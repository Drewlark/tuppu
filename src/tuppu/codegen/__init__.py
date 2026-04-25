"""Tuppu codegen: AST -> LLVM IR.

The entry point is the module-level `codegen(program)` function,
which walks the AST and emits an `llvmlite.ir.Module`.

Organization: the Codegen class is split into mixins by concern —
`sex.py` holds all Babylonian runtime helpers and literal lowering,
`rat.py` holds rat arithmetic + reduction, `tablets.py` holds the
growable-storage runtime. Everything else (statements, expressions,
types, coercion, prints, structs, strings, tables, basic literals)
lives in this file. Shared constants and dataclasses live in
`_common.py`, imported by every mixin."""
from __future__ import annotations

from dataclasses import dataclass

from llvmlite import binding as llvm
from llvmlite import ir

from .. import ast as A
from ..comptime import Comptime, ComptimeError
from ._common import (
    CodegenError,
    I1, I8, I16, I32, I64,
    INTRINSICS, INT_WIDTH,
    RAT, SEX, SEX_MAX_DIGITS,
    SEX_IDX_DIGITS, SEX_IDX_RADIX, SEX_IDX_COUNT, SEX_IDX_SIGN,
    TabletsInfo, Variable,
)
from .access import AccessMixin
from .expr import ExprMixin
from .intrinsics import IntrinsicsMixin
from .rat import RatMixin
from .seals import SealsMixin
from .sex import SexMixin
from .stmt import StmtMixin
from .strs import StrsMixin
from .tablets import TabletsMixin
from .types import TypesMixin


class Codegen(
    SexMixin, RatMixin, TabletsMixin, StrsMixin, SealsMixin,
    ExprMixin, StmtMixin, IntrinsicsMixin, AccessMixin, TypesMixin,
):
    def __init__(self, checker=None) -> None:
        # Checker provides monomorphization sidebands — `mono_call_args`
        # keyed by id(Call) and `mono_struct_args` keyed by id(StructLit).
        # Plus `struct_type_params` / `fn_type_params` for name → params.
        self._checker = checker
        # Fresh LLVM context per Codegen so identified struct types
        # (e.g. user `seal Point { ... }`) don't leak across test runs
        # or multiple compilations in the same process.
        self.module = ir.Module(name="tuppu", context=ir.Context())
        self.module.triple = llvm.get_default_triple()
        self.builder: ir.IRBuilder | None = None
        self.functions: dict[str, ir.Function] = {}
        # Per-fn parameter mut-ness — populated by `_declare_fn` /
        # `_declare_colophon` / `_declare_gloss` / monomorph paths.
        # Consulted by `_gen_call` to decide whether a struct-with-
        # cleanup arg needs field neutering: only mut params do (so
        # the callee's cleanup frame doesn't double-free), non-mut
        # params read the caller's data as-is.
        self._fn_param_mut: dict[str, list[bool]] = {}
        self.scopes: list[dict[str, Variable]] = []
        self._strings: dict[bytes, ir.GlobalVariable] = {}
        self._str_counter = 0
        self._rat_reduce: ir.Function | None = None  # built lazily
        self._sex_to_rat: ir.Function | None = None
        self._sex_print: ir.Function | None = None
        self._sex_add: ir.Function | None = None
        self._sex_cmp: ir.Function | None = None
        self._int_to_sex: ir.Function | None = None
        self._rat_to_sex: ir.Function | None = None
        self._trap: ir.Function | None = None
        # Dynamic-string runtime: release (free if cap > 0), concat, slice,
        # value→str conversions. All built lazily on first use.
        self._str_release: ir.Function | None = None
        self._str_concat: ir.Function | None = None
        self._str_slice: ir.Function | None = None
        self._int_to_str: ir.Function | None = None
        self._sex_to_str: ir.Function | None = None
        self._memcpy: ir.Function | None = None
        self._snprintf: ir.Function | None = None
        # Per-struct release fns, keyed by id() of the LLVM struct type.
        # Built lazily; only emitted for structs that transitively hold
        # cleanup-bearing fields (str / tablets / nested cleanup struct).
        self._struct_release_cache: dict[int, ir.Function] = {}
        # Per-struct clone fns, keyed by id(struct_ty). Emitted lazily
        # on first use at a Field/Index return site, where the
        # returned value needs independently-owned bytes so the
        # caller doesn't UAF when the source's cleanup fires first.
        self._struct_clone_cache: dict[int, ir.Function] = {}
        # Per-seal release + clone fns, keyed by id(seal_ty). Emitted
        # lazily; only built for seals that transitively hold cleanup-
        # bearing payload fields.
        self._seal_release_cache: dict[int, ir.Function] = {}
        self._seal_clone_cache: dict[int, ir.Function] = {}
        # table name -> (global array, length, lo bound, element LLVM type)
        self._tables: dict[str, tuple[ir.GlobalVariable, int, int, ir.Type]] = {}
        # Tablets monomorphizations: key is (N, str(elem_type)).
        self._tablets_types: dict[tuple[int, str], "TabletsInfo"] = {}
        # User-defined structs: name -> LLVM struct type + ordered fields.
        # Per-block stack of cleanups (just tablets releases for now).
        # Each entry: (release_fn, ptr, source_name). Pushed at block
        # entry, popped at block exit (emitting releases along the way).
        # Mirrors the scope stack — same push/pop cadence.
        self._cleanup_frames: list[list[tuple[ir.Function, ir.Value, str]]] = []
        # Parallel to _cleanup_frames: count of GC roots pushed into
        # the innermost frame, popped at frame exit.
        self._gc_root_counts: list[int] = []
        self._struct_types: dict[str, ir.LiteralStructType] = {}
        self._struct_fields: dict[str, list[tuple[str, ir.Type]]] = {}
        # Seal (sum type) state. `_seal_types` keys are seal names for
        # non-generic seals and `(name, arg_tys)` tuples for concrete
        # monomorphizations. Each seal value is laid out as
        # `{ i8 tag, [N x i64] payload }` where N is chosen to fit the
        # widest variant. Variant payloads are accessed via a bitcast
        # of the payload slot to a per-variant "payload struct".
        self._seal_types: dict = {}
        # seal key -> [(variant_name, payload_struct_ty), ...] in
        # source order. payload_struct_ty is a LiteralStructType whose
        # elements are the variant's field LLVM types (empty tuple for
        # nullary variants).
        self._seal_variants: dict = {}
        # Declarations for generic seals, keyed by seal name. Populated
        # in `_register_seals` and consumed by `_get_monomorph_seal`.
        self._generic_seal_decls: dict = {}
        # Generic monomorphizations. Keys are (name, tuple-of-LLVM-types).
        # Values are the specialized LLVM types / functions. Populated
        # on demand via `_get_monomorph_struct` / `_get_monomorph_fn`.
        self._struct_monomorphs: dict[tuple, ir.IdentifiedStructType] = {}
        self._struct_mono_fields: dict[tuple, list[tuple[str, ir.Type]]] = {}
        self._fn_monomorphs: dict[tuple, ir.Function] = {}
        # Current generic-body type-arg substitution, source-param name
        # → concrete LLVM type. Set by `_emit_fn_specialization` while
        # walking a specialization of a generic fn body.
        self._type_arg_subst: dict[str, ir.Type] = {}
        # Most-recent AST source location, updated as we walk statements
        # and expressions. Used to attach line:col to codegen errors that
        # don't otherwise carry one.
        self._current_loc: tuple[int, int] = (0, 0)
        self._init_runtime_externs()

    def _init_runtime_externs(self) -> None:
        """Declare the libc functions our intrinsics lower to."""
        i8ptr = I8.as_pointer()
        self.printf = self._get_or_declare_libc(
            "printf", ir.FunctionType(I32, [i8ptr], var_arg=True),
        )
        self.scanf = self._get_or_declare_libc(
            "scanf", ir.FunctionType(I32, [i8ptr], var_arg=True),
        )
        self._malloc: ir.Function | None = None  # lazy
        self._free: ir.Function | None = None
        self._write: ir.Function | None = None
        self._fflush: ir.Function | None = None
        self._strlen: ir.Function | None = None
        # GC runtime externs (see runtime/tuppu_gc.c). Lazy.
        self._gc_alloc_bytes: ir.Function | None = None
        self._gc_push_root: ir.Function | None = None
        self._gc_pop_roots: ir.Function | None = None
        # Cache type descriptors keyed by a stable string form of the
        # value type — one LLVM global per distinct type.
        self._type_descs: dict[str, ir.GlobalVariable] = {}
        # LLVM type for tuppu_type_t (see runtime/tuppu_gc.c):
        # { i8* name; i64 size; i64 n_ptrs; i64* ptr_offsets;
        #   void(i8*)* trace }.
        self._trace_fn_ty = ir.FunctionType(
            ir.VoidType(), [I8.as_pointer()],
        )
        self._type_desc_ty = ir.LiteralStructType([
            I8.as_pointer(), I64, I64, I64.as_pointer(),
            self._trace_fn_ty.as_pointer(),
        ])
        # Colophon decls by Tuppu-level name, for call-site marshaling.
        self._colophon_decls: dict[str, A.ColophonDecl] = {}

    def _get_or_declare_libc(
        self, name: str, fn_type: ir.FunctionType,
    ) -> ir.Function:
        """Declare `name` as an LLVM extern with the given signature,
        or return the existing declaration if one is already present
        (e.g. from a user `colophon fn`). Raises if a pre-existing
        declaration has an incompatible signature — that would
        silently miscompile calls through it, and the fix is to pick
        a different Tuppu-side name (C-symbol renames aren't in the
        syntax yet)."""
        existing = self.module.globals.get(name)
        if existing is not None:
            existing_ty = getattr(existing, "function_type", None)
            if existing_ty != fn_type:
                raise CodegenError(
                    f"compiler needs extern {name!r} with signature "
                    f"{fn_type}, but the module already has one with "
                    f"{existing_ty} (likely from a user `colophon fn "
                    f"{name}(...)` declaration). Rename the colophon."
                )
            return existing
        return ir.Function(self.module, fn_type, name=name)

    def _get_malloc(self) -> ir.Function:
        """Heap allocator for Tuppu-emitted code. Routes through the
        GC's raw-bytes allocator — used for str contents, where the
        buffer is a byte leaf with no internal pointers to trace.
        Tablets chunks deliberately do NOT go through here; they
        allocate via `_get_gc_alloc_typed` with a chunk descriptor
        so GC can trace their elements and next-pointer."""
        return self._get_gc_alloc_bytes()

    def _get_gc_alloc_typed(self) -> ir.Function:
        """`__tuppu_gc_alloc(size, *type_desc) -> i8*` — typed GC
        allocation used for objects whose internal layout the GC
        needs to trace (tablets chunks, composite heap objects)."""
        existing = self.module.globals.get("__tuppu_gc_alloc")
        if existing is not None:
            return existing
        fn_type = ir.FunctionType(
            I8.as_pointer(), [I64, I8.as_pointer()],
        )
        return ir.Function(self.module, fn_type, name="__tuppu_gc_alloc")

    def _get_free(self) -> ir.Function:
        """Free path for code that still emits explicit free() calls
        (tablets release, etc.). Routes to a no-op in the runtime so
        GC-owned buffers aren't corrupted by a bogus libc free during
        the migration. Once Stage 2.5 removes the free-call sites,
        this can go away entirely."""
        if self._free is None:
            self._free = self._get_or_declare_libc(
                "__tuppu_gc_noop_free",
                ir.FunctionType(ir.VoidType(), [I8.as_pointer()]),
            )
        return self._free

    def _get_gc_alloc_bytes(self) -> ir.Function:
        """`__tuppu_gc_alloc_bytes(size) -> i8*` — GC-managed byte
        buffer allocator. Used for leaf allocations (str contents,
        tablet chunks) — things with no internal pointer fields."""
        if self._gc_alloc_bytes is None:
            self._gc_alloc_bytes = self._get_or_declare_libc(
                "__tuppu_gc_alloc_bytes",
                ir.FunctionType(I8.as_pointer(), [I64]),
            )
        return self._gc_alloc_bytes

    def _get_gc_push_root(self) -> ir.Function:
        """`__tuppu_gc_push_root(slot, type_desc)` — register a stack
        slot as a GC root. `slot` points at an alloca holding the
        object (by value); `type_desc` describes which offsets inside
        that object are pointer fields the GC should trace."""
        if self._gc_push_root is None:
            self._gc_push_root = self._get_or_declare_libc(
                "__tuppu_gc_push_root",
                ir.FunctionType(
                    ir.VoidType(),
                    [I8.as_pointer(), I8.as_pointer()],
                ),
            )
        return self._gc_push_root

    def _get_gc_pop_roots(self) -> ir.Function:
        """`__tuppu_gc_pop_roots(n)` — pop the top-n root entries off
        the shadow stack, matching the push_root calls at fn entry."""
        if self._gc_pop_roots is None:
            self._gc_pop_roots = self._get_or_declare_libc(
                "__tuppu_gc_pop_roots",
                ir.FunctionType(ir.VoidType(), [I64]),
            )
        return self._gc_pop_roots

    def _type_desc_key(self, value_ty: ir.Type) -> str | None:
        """Stable string key identifying a type for descriptor
        caching. Returns None for types that don't need GC tracing
        (scalars, pointers, fn values, seals with only scalar
        payloads)."""
        if self._is_str_value(value_ty):
            return "__tuppu_str"
        info = self._tablets_info_for(value_ty)
        if info is not None:
            return f"__tuppu_tbls_{info.elem_ty}_{info.N}".replace(" ", "_")
        if isinstance(value_ty, ir.IdentifiedStructType):
            if self._seal_key_for_ty(value_ty) is not None:
                if not self._seal_needs_cleanup(value_ty):
                    return None
                return f"__tuppu_seal_{value_ty.name}"
            # Plain structs: only produce a desc when there's something
            # to trace. `_get_type_desc` and the chokepoint both key off
            # this, so keeping them agreed prevents push/pop counter
            # drift (alloca a rooted slot but skip the push, vs. pop
            # expecting a push that never happened).
            if self._struct_needs_cleanup(value_ty):
                return f"__tuppu_struct_{value_ty.name}"
            return None
        return None

    def _type_ptr_offsets(self, value_ty: ir.Type) -> list[int]:
        """Return byte offsets of pointer fields inside `value_ty`
        that the GC needs to trace.

        - str `{i8* ptr, i64 len, i64 cap}` — ptr at offset 0.
        - tablets `{*Node head, *Node tail, i64 len}` — head at 0,
          tail at 8. Marking both also reaches the chunk chain since
          each chunk's own descriptor lists its `next` ptr.
        - Struct (identified or literal): walk fields, composing
          each field's offsets at the field's *aligned* start.
          Alignment-aware — a `{i8, str}` variant payload keeps the
          str at offset 8, not 1.
        - Array: fan out offsets across N elements. Covers buffers
          or struct fields that happen to be arrays of cleanup-
          bearing elements.
        - Chunk (Node_...): built separately via
          `_chunk_ptr_offsets` since it has a fixed layout derived
          from (N, elem_ty).
        """
        if self._is_str_value(value_ty):
            return [0]
        info = self._tablets_info_for(value_ty)
        if info is not None:
            return [0, 8]   # head, tail ptrs
        if isinstance(value_ty, (ir.IdentifiedStructType, ir.LiteralStructType)):
            elements = getattr(value_ty, "elements", None) or ()
            offsets: list[int] = []
            offset = 0
            for fty in elements:
                align = self._align_of(fty)
                offset = (offset + align - 1) // align * align
                for inner_off in self._type_ptr_offsets(fty):
                    offsets.append(offset + inner_off)
                offset += self._size_of(fty)
            return offsets
        if isinstance(value_ty, ir.ArrayType):
            inner = self._type_ptr_offsets(value_ty.element)
            if not inner:
                return []
            elem_size = self._size_of(value_ty.element)
            offsets = []
            for i in range(value_ty.count):
                base = i * elem_size
                for off in inner:
                    offsets.append(base + off)
            return offsets
        return []

    def _chunk_ptr_offsets(
        self, N: int, elem_ty: ir.Type,
    ) -> list[int]:
        """Byte offsets of pointer fields inside a tablets chunk.
        Chunk layout: `[elem[0]..elem[N-1], used: i64, next: *Node]`.
        Each slot contributes offsets per its own type; `next` sits
        at the first 8-aligned offset after the items array plus the
        8-byte `used` field. Uses alignment-aware `_size_of` so
        mixed-align element types (e.g. a variant payload `{i8, str}`)
        stride correctly."""
        elem_size = self._size_of(elem_ty)
        inner = self._type_ptr_offsets(elem_ty)
        offsets: list[int] = []
        for i in range(N):
            base = i * elem_size
            for inner_off in inner:
                offsets.append(base + inner_off)
        items_end = N * elem_size
        used_off = (items_end + 7) & ~7   # align to i64 for used
        next_off = used_off + 8
        offsets.append(next_off)
        return offsets

    def _get_chunk_type_desc(
        self, N: int, elem_ty: ir.Type, node_ty: ir.IdentifiedStructType,
    ) -> ir.GlobalVariable:
        """Emit (or return cached) `tuppu_type_t` for a tablets chunk.
        Chunks allocate via `__tuppu_gc_alloc(size, &chunk_desc)` so
        GC marks through them. Element types that recursively hold a
        seal field (whose payload layout is tag-dispatched) get a
        per-chunk trace fn that walks each slot via the same
        alignment-aware composition as the struct trace fns. Plain
        elements stick to a flat ptr_offsets table."""
        key = f"__tuppu_chunk_{elem_ty}_{N}".replace(" ", "_")
        cached = self._type_descs.get(key)
        if cached is not None:
            return cached
        size = N * self._size_of(elem_ty) + 16
        # If the element type's layout includes any seal-with-cleanup
        # field (directly or nested), the flat ptr-offsets approach
        # can't see the variant-dependent payload ptrs. Emit a chunk
        # trace fn that loops over slots and recurses through the
        # element's full tracing logic.
        needs_trace = self._contains_seal_anywhere(elem_ty)
        if needs_trace:
            offsets: list[int] = []
            trace_fn: ir.Function | None = self._get_chunk_trace_fn(
                key, N, elem_ty, node_ty,
            )
        else:
            offsets = self._chunk_ptr_offsets(N, elem_ty)
            trace_fn = None
        offsets_arr_ty = ir.ArrayType(I64, max(len(offsets), 1))
        offsets_arr = ir.GlobalVariable(
            self.module, offsets_arr_ty, f"{key}_offsets",
        )
        offsets_arr.linkage = "internal"
        offsets_arr.global_constant = True
        offsets_arr.initializer = ir.Constant(
            offsets_arr_ty,
            [ir.Constant(I64, o) for o in offsets] or [ir.Constant(I64, 0)],
        )
        name_bytes = (key + "\0").encode("utf-8")
        name_arr_ty = ir.ArrayType(I8, len(name_bytes))
        name_arr = ir.GlobalVariable(
            self.module, name_arr_ty, f"{key}_name",
        )
        name_arr.linkage = "internal"
        name_arr.global_constant = True
        name_arr.initializer = ir.Constant(
            name_arr_ty, bytearray(name_bytes),
        )
        trace_init: ir.Constant | ir.Function = (
            trace_fn if trace_fn is not None
            else ir.Constant(self._trace_fn_ty.as_pointer(), None)
        )
        desc = ir.GlobalVariable(self.module, self._type_desc_ty, key)
        desc.linkage = "internal"
        desc.global_constant = True
        desc.initializer = ir.Constant(self._type_desc_ty, [
            name_arr.bitcast(I8.as_pointer()),
            ir.Constant(I64, size),
            ir.Constant(I64, len(offsets)),
            offsets_arr.bitcast(I64.as_pointer()),
            trace_init,
        ])
        self._type_descs[key] = desc
        return desc

    def _get_chunk_trace_fn(
        self, key: str, N: int, elem_ty: ir.Type,
        node_ty: ir.IdentifiedStructType,
    ) -> ir.Function:
        """Per-chunk trace fn for chunks whose element layout includes
        a seal (or anything else flat ptr_offsets can't express).
        Walks all N slots — the chunk header's `used` field tells the
        runtime how many to mind, but unused slots are calloc-zero, so
        marking them is a safe no-op via mark_ptr's null check. Also
        marks the `next` chunk pointer."""
        fn_name = f"{key}_trace"
        cached = self.module.globals.get(fn_name)
        if isinstance(cached, ir.Function):
            return cached
        fn = ir.Function(self.module, self._trace_fn_ty, fn_name)
        fn.linkage = "internal"
        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        base = fn.args[0]  # i8* to chunk start
        elem_size = self._size_of(elem_ty)
        for i in range(N):
            self._emit_trace_mark_calls(b, base, i * elem_size, elem_ty)
        # Mark the `next` chunk pointer at offset N*elem_size + 8
        # (alignment-padded past `used: i64`).
        items_end = N * elem_size
        used_off = (items_end + 7) & ~7
        next_off = used_off + 8
        next_i8 = b.gep(base, [ir.Constant(I64, next_off)], inbounds=True)
        next_pp = b.bitcast(next_i8, I8.as_pointer().as_pointer())
        next_p = b.load(next_pp)
        b.call(self._get_gc_mark_ptr(), [next_p])
        b.ret_void()
        return fn

    def _get_type_desc(self, value_ty: ir.Type) -> ir.GlobalVariable | None:
        """Fetch or emit a `tuppu_type_t` global for `value_ty`. Returns
        None if the type needs no tracing (no pointer fields)."""
        key = self._type_desc_key(value_ty)
        if key is None:
            return None
        cached = self._type_descs.get(key)
        if cached is not None:
            return cached
        # Seals dispatch tracing via a per-seal fn because a flat
        # ptr_offsets table can't express the tag-dependent payload
        # layout. Structs that transitively contain a seal field need
        # the same escape hatch so GC can recurse into the seal's
        # trace fn. Everything else gets a flat ptr_offsets table.
        is_seal = (
            isinstance(value_ty, ir.IdentifiedStructType)
            and self._seal_key_for_ty(value_ty) is not None
        )
        trace_fn: ir.Function | None = None
        offsets: list[int] = []
        if is_seal:
            trace_fn = self._get_seal_trace_fn(value_ty)
        elif (
            isinstance(value_ty, ir.IdentifiedStructType)
            and self._struct_contains_seal(value_ty)
        ):
            trace_fn = self._get_struct_trace_fn(value_ty)
        else:
            offsets = self._type_ptr_offsets(value_ty)
            if not offsets:
                return None
        # Offsets table as an LLVM global (possibly empty for seals).
        offsets_arr_ty = ir.ArrayType(I64, max(len(offsets), 1))
        offsets_arr = ir.GlobalVariable(
            self.module, offsets_arr_ty, f"{key}_offsets",
        )
        offsets_arr.linkage = "internal"
        offsets_arr.global_constant = True
        offsets_arr.initializer = ir.Constant(
            offsets_arr_ty,
            [ir.Constant(I64, o) for o in offsets] or [ir.Constant(I64, 0)],
        )
        # Name string as a global.
        name_bytes = (key + "\0").encode("utf-8")
        name_arr_ty = ir.ArrayType(I8, len(name_bytes))
        name_arr = ir.GlobalVariable(
            self.module, name_arr_ty, f"{key}_name",
        )
        name_arr.linkage = "internal"
        name_arr.global_constant = True
        name_arr.initializer = ir.Constant(
            name_arr_ty, bytearray(name_bytes),
        )
        trace_init: ir.Constant | ir.Function
        if trace_fn is None:
            trace_init = ir.Constant(self._trace_fn_ty.as_pointer(), None)
        else:
            trace_init = trace_fn
        desc = ir.GlobalVariable(self.module, self._type_desc_ty, key)
        desc.linkage = "internal"
        desc.global_constant = True
        desc.initializer = ir.Constant(self._type_desc_ty, [
            name_arr.bitcast(I8.as_pointer()),
            ir.Constant(I64, self._size_of(value_ty)),
            ir.Constant(I64, len(offsets)),
            offsets_arr.bitcast(I64.as_pointer()),
            trace_init,
        ])
        self._type_descs[key] = desc
        return desc

    def _get_seal_trace_fn(self, seal_ty: ir.Type) -> ir.Function:
        """Emit (or return cached) a per-seal trace function that
        dispatches on the tag byte and marks each variant's cleanup-
        bearing payload fields. The GC runtime calls this via the
        `trace` field on tuppu_type_t.

        The fn takes an `i8*` pointing at the seal's start address.
        It bitcasts to the seal type, loads the tag, and switches
        to variant-specific blocks. Each arm walks the variant's
        payload fields — plain str / tablets contribute mark_ptr
        calls at their offsets; nested seals recurse via the inner
        seal's own trace fn; nested structs compose into flat ptr
        offsets."""
        assert isinstance(seal_ty, ir.IdentifiedStructType)
        seal_key = self._seal_key_for_ty(seal_ty)
        assert seal_key is not None
        fn_name = f"__tuppu_seal_{seal_ty.name}_trace"
        cached = self.module.globals.get(fn_name)
        if isinstance(cached, ir.Function):
            return cached
        fn = ir.Function(self.module, self._trace_fn_ty, fn_name)
        fn.linkage = "internal"
        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        seal_ptr = b.bitcast(fn.args[0], seal_ty.as_pointer())
        tag_ptr = b.gep(
            seal_ptr, [ir.Constant(I32, 0), ir.Constant(I32, 0)], inbounds=True,
        )
        tag = b.load(tag_ptr)
        payload_raw_ptr = b.gep(
            seal_ptr, [ir.Constant(I32, 0), ir.Constant(I32, 1)],
            inbounds=True,
        )
        payload_base = b.bitcast(payload_raw_ptr, I8.as_pointer())
        merge_bb = fn.append_basic_block("trace.done")
        variants = self._seal_variants.get(seal_key, [])
        if not variants:
            b.branch(merge_bb)
        else:
            switch = b.switch(tag, merge_bb)
            for idx, (vname, payload_ty) in enumerate(variants):
                arm = fn.append_basic_block(f"trace.{vname}")
                switch.add_case(ir.Constant(I8, idx), arm)
                b.position_at_end(arm)
                fld_off = 0
                for fty in payload_ty.elements:
                    align = self._align_of(fty)
                    fld_off = (fld_off + align - 1) // align * align
                    self._emit_trace_mark_calls(b, payload_base, fld_off, fty)
                    fld_off += self._size_of(fty)
                b.branch(merge_bb)
        b.position_at_end(merge_bb)
        b.ret_void()
        return fn

    def _emit_trace_mark_calls(
        self,
        b: ir.IRBuilder,
        base: ir.Value,
        offset: int,
        field_ty: ir.Type,
    ) -> None:
        """Emit the IR that marks every GC-reachable pointer inside
        `field_ty` at `base + offset`. Nested seals dispatch into
        their own trace fn; structs that (transitively) contain a
        seal also dispatch via a struct trace fn so the tag-based
        recursion chains through. Everything else falls back to a
        flat ptr_offsets walk. Scalars are no-ops."""
        if isinstance(field_ty, ir.IdentifiedStructType):
            seal_key = self._seal_key_for_ty(field_ty)
            if seal_key is not None:
                if self._seal_needs_cleanup(field_ty):
                    inner_fn = self._get_seal_trace_fn(field_ty)
                    sub_ptr = b.gep(
                        base, [ir.Constant(I64, offset)], inbounds=True,
                    )
                    b.call(inner_fn, [sub_ptr])
                return
            if self._struct_contains_seal(field_ty):
                inner_fn = self._get_struct_trace_fn(field_ty)
                sub_ptr = b.gep(
                    base, [ir.Constant(I64, offset)], inbounds=True,
                )
                b.call(inner_fn, [sub_ptr])
                return
        # Literal struct (e.g. variant payload tuple) that contains a
        # seal field still needs recursion — walk fields one by one
        # so nested seal fields re-enter the dispatch.
        if isinstance(field_ty, ir.LiteralStructType):
            if any(self._contains_seal_anywhere(el) for el in field_ty.elements):
                inner_off = 0
                for el in field_ty.elements:
                    align = self._align_of(el)
                    inner_off = (inner_off + align - 1) // align * align
                    self._emit_trace_mark_calls(b, base, offset + inner_off, el)
                    inner_off += self._size_of(el)
                return
        offsets = self._type_ptr_offsets(field_ty)
        if not offsets:
            return
        mark_fn = self._get_gc_mark_ptr()
        for inner in offsets:
            total = offset + inner
            field_i8 = b.gep(
                base, [ir.Constant(I64, total)], inbounds=True,
            )
            field_ptr_ptr = b.bitcast(field_i8, I8.as_pointer().as_pointer())
            field_ptr = b.load(field_ptr_ptr)
            b.call(mark_fn, [field_ptr])

    def _contains_seal_anywhere(self, ty: ir.Type) -> bool:
        """Does `ty` anywhere in its layout hold a cleanup-bearing
        seal? Used to decide whether a composite field's trace needs
        the full field-by-field recursion or can use flat offsets."""
        if isinstance(ty, ir.IdentifiedStructType):
            if self._seal_key_for_ty(ty) is not None:
                return self._seal_needs_cleanup(ty)
            if self._struct_contains_seal(ty):
                return True
        if isinstance(ty, ir.LiteralStructType):
            return any(self._contains_seal_anywhere(el) for el in ty.elements)
        return False

    def _struct_contains_seal(self, struct_ty: ir.Type) -> bool:
        """Does this struct transitively contain a cleanup-bearing
        seal field? Those fields need tag-dispatch tracing, which
        a flat ptr_offsets table can't express."""
        if not isinstance(struct_ty, ir.IdentifiedStructType):
            return False
        if self._seal_key_for_ty(struct_ty) is not None:
            return self._seal_needs_cleanup(struct_ty)
        fields = self._struct_fields_for(struct_ty)
        if fields is None:
            return False
        for _name, fty in fields:
            if self._struct_contains_seal(fty):
                return True
        return False

    def _get_struct_trace_fn(self, struct_ty: ir.Type) -> ir.Function:
        """Emit (or return cached) a trace fn for a struct whose
        layout can't be expressed as a flat ptr_offsets table —
        today, structs that transitively hold a seal-with-cleanup
        field. The fn walks each field and recurses via
        `_emit_trace_mark_calls`, which handles seals by calling
        their own trace fn in turn."""
        assert isinstance(struct_ty, ir.IdentifiedStructType)
        fn_name = f"__tuppu_struct_{struct_ty.name}_trace"
        cached = self.module.globals.get(fn_name)
        if isinstance(cached, ir.Function):
            return cached
        fn = ir.Function(self.module, self._trace_fn_ty, fn_name)
        fn.linkage = "internal"
        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        base = fn.args[0]
        fields = self._struct_fields_for(struct_ty) or []
        offset = 0
        for _name, fty in fields:
            align = self._align_of(fty)
            offset = (offset + align - 1) // align * align
            self._emit_trace_mark_calls(b, base, offset, fty)
            offset += self._size_of(fty)
        b.ret_void()
        return fn

    def _get_gc_mark_ptr(self) -> ir.Function:
        """`__tuppu_gc_mark_ptr(ptr)` — runtime callback for trace fns
        to mark a discovered pointer as reachable."""
        cached = self.module.globals.get("__tuppu_gc_mark_ptr")
        if isinstance(cached, ir.Function):
            return cached
        fty = ir.FunctionType(ir.VoidType(), [I8.as_pointer()])
        return ir.Function(self.module, fty, "__tuppu_gc_mark_ptr")

    def _emit_gc_push_root(self, slot: ir.Value, value_ty: ir.Type) -> bool:
        """Emit a `__tuppu_gc_push_root(slot, type_desc)` call if the
        type has pointer fields to trace. Returns True on emit so the
        caller can count how many pops are needed at frame exit."""
        desc = self._get_type_desc(value_ty)
        if desc is None:
            return False
        b = self.builder
        assert b is not None
        b.call(
            self._get_gc_push_root(),
            [
                b.bitcast(slot, I8.as_pointer()),
                b.bitcast(desc, I8.as_pointer()),
            ],
        )
        return True

    def _emit_gc_pop_roots(self, n: int) -> None:
        if n <= 0:
            return
        b = self.builder
        assert b is not None
        b.call(self._get_gc_pop_roots(), [ir.Constant(I64, n)])

    def _get_write(self) -> ir.Function:
        if self._write is None:
            self._write = self._get_or_declare_libc(
                "write", ir.FunctionType(I64, [I32, I8.as_pointer(), I64]),
            )
        return self._write

    def _get_fflush(self) -> ir.Function:
        if self._fflush is None:
            self._fflush = self._get_or_declare_libc(
                "fflush", ir.FunctionType(I32, [I8.as_pointer()]),
            )
        return self._fflush

    def _get_strlen(self) -> ir.Function:
        if self._strlen is None:
            self._strlen = self._get_or_declare_libc(
                "strlen", ir.FunctionType(I64, [I8.as_pointer()]),
            )
        return self._strlen

    # --- top level ---

    def gen(self, prog: A.Program) -> ir.Module:
        self.comptime = Comptime(prog)

        # Phase 0: build struct + seal LLVM types. Interleave declaration
        # and body-resolution so a struct field of seal type (or vice
        # versa) can see the identified type of the other form before
        # we compute layouts.
        struct_decls = [
            d for d in prog.decls if isinstance(d, A.StructDecl)
        ]
        seal_decls = [
            d for d in prog.decls if isinstance(d, A.SealDecl)
        ]
        self._register_structs_declare(struct_decls)
        self._register_seals_declare(seal_decls)
        self._register_structs_resolve(struct_decls)
        self._register_seals_resolve(seal_decls)

        # Generic fns are monomorphized lazily at call sites, so we
        # don't declare/emit them here — just stash the AST.
        self._generic_fn_decls: dict[str, A.FnDecl] = {
            d.name: d for d in prog.decls
            if isinstance(d, A.FnDecl) and d.type_params
        }

        # Phase 1: forward-declare all non-generic user functions, plus
        # colophon externs (C functions the compiler marshals to / from
        # at each call site).
        for decl in prog.decls:
            if isinstance(decl, A.FnDecl):
                if decl.type_params:
                    continue
                self._declare_fn(decl)
            elif isinstance(decl, A.ColophonDecl):
                self._declare_colophon(decl)
            elif isinstance(decl, A.GlossDecl):
                self._declare_gloss(decl)
            elif isinstance(decl, A.TableDecl):
                pass  # handled in phase 2 after function decls are visible
            elif isinstance(decl, A.StructDecl):
                pass  # already handled in phase 0
            elif isinstance(decl, A.SealDecl):
                pass  # already handled in phase 0c
            else:
                raise CodegenError(
                    f"unsupported top-level: {type(decl).__name__}"
                )

        # Phase 2: evaluate tables at compile time and emit them as static
        # globals. Done in declaration order so later tables may reference
        # earlier ones.
        for decl in prog.decls:
            if isinstance(decl, A.TableDecl):
                self._emit_table(decl)

        # Phase 3: emit bodies of non-generic functions. Generic fn
        # specializations are emitted on demand when we see a call to
        # them (see `_get_monomorph_fn`).
        for decl in prog.decls:
            if isinstance(decl, A.FnDecl):
                if decl.type_params:
                    continue
                self._gen_fn_body(decl)
            elif isinstance(decl, A.GlossDecl):
                self._gen_gloss_body(decl)
        return self.module

    def _declare_fn(self, fn: A.FnDecl) -> None:
        if fn.name in INTRINSICS:
            raise CodegenError(
                f"cannot define {fn.name!r}: it is a built-in intrinsic"
            )
        if fn.name in self.functions:
            raise CodegenError(f"duplicate function {fn.name!r}")
        param_types = []
        for p in fn.params:
            t = self._lower_type(p.type)
            # Mut tablets params are passed by reference so mutations
            # (push, release) persist to the caller's storage. Without
            # this the caller's tablets header (head/tail/len) stays
            # unchanged and any chunks the callee allocated would leak.
            if p.is_mut and self._tablets_info_for(t) is not None:
                t = t.as_pointer()
            # Variadic `tablets[...]T` param: call site builds the
            # literal in the caller's frame; callee receives a pointer
            # so indexing and iteration see the actual chunks.
            elif isinstance(p.type, A.TypeVariadicTablets):
                t = t.as_pointer()
            # Mut user-struct param — pass by pointer so callee
            # mutations persist to the caller's storage. Previously
            # mut structs were pass-by-value, which made
            # `fn add_route(mut app: App) { app.routes.push(...) }`
            # silently no-op from the caller's perspective. Matches
            # the mut-tablets and colophon-mut-struct conventions.
            # `str` is excluded: it has its own cap-sentinel ownership
            # model and reassignment-release machinery that assumes
            # by-value passing with call-site neutering.
            elif (
                p.is_mut
                and self._struct_fields_for(t) is not None
                and not self._is_str_value(t)
            ):
                t = t.as_pointer()
            param_types.append(t)
        ret_type = self._lower_type(fn.return_type) if fn.return_type else ir.VoidType()
        fn_type = ir.FunctionType(ret_type, param_types)
        llvm_fn = ir.Function(self.module, fn_type, name=fn.name)
        for i, p in enumerate(fn.params):
            llvm_fn.args[i].name = p.name
        self.functions[fn.name] = llvm_fn
        self._fn_param_mut[fn.name] = [p.is_mut for p in fn.params]

    def _declare_gloss(self, g: A.GlossDecl) -> None:
        """Forward-declare a gloss fn under its mangled internal name.
        Mirrors `_declare_fn` but resolves the name through the
        checker's mangle scheme so operator dispatch can `self.functions
        [mangled]` like any other fn."""
        from ..typecheck import GLOSS_OPS
        if self._checker is None:
            raise CodegenError("gloss decl requires a typechecker pass")
        # Rebuild the mangled name from the decl's operand types.
        param_tys = tuple(
            self._checker._resolve_type(p.type, "gloss param")
            for p in g.params
        )
        _sym, arity, _ = GLOSS_OPS[g.op]
        rhs_ty = param_tys[1] if arity == "bin" else None
        mangled = self._checker._gloss_mangled_name(g.op, param_tys[0], rhs_ty)
        fake_fn = A.FnDecl(
            name=mangled,
            params=g.params,
            return_type=g.return_type,
            body=g.body,
            line=g.line,
            col=g.col,
        )
        self._declare_fn(fake_fn)

    def _gen_gloss_body(self, g: A.GlossDecl) -> None:
        """Emit the body of a gloss decl — identical to a regular fn
        body, just under the mangled name registered during
        `_declare_gloss`."""
        from ..typecheck import GLOSS_OPS
        assert self._checker is not None
        param_tys = tuple(
            self._checker._resolve_type(p.type, "gloss param")
            for p in g.params
        )
        _sym, arity, _ = GLOSS_OPS[g.op]
        rhs_ty = param_tys[1] if arity == "bin" else None
        mangled = self._checker._gloss_mangled_name(g.op, param_tys[0], rhs_ty)
        fake_fn = A.FnDecl(
            name=mangled,
            params=g.params,
            return_type=g.return_type,
            body=g.body,
            line=g.line,
            col=g.col,
        )
        self._gen_fn_body(fake_fn)

    def _declare_colophon(self, c: A.ColophonDecl) -> None:
        """Forward-declare a libc extern. The LLVM signature uses C-ABI
        types (i8* for Tuppu str, i8 for bool, ints pass through) so the
        Tuppu-level call site can marshal values at each boundary —
        caller-side str gets a fresh NUL-terminated heap buffer, return
        str gets copied into a Tuppu-owned heap str via strlen + memcpy.

        Reserves the Tuppu name in both the fn table and a per-colophon
        sideband so the call-site dispatch can recognise colophon calls
        and pick the marshaling path."""
        if c.name in INTRINSICS:
            raise CodegenError(
                f"cannot declare colophon {c.name!r}: name is a built-in intrinsic"
            )
        if c.name in self.functions:
            raise CodegenError(f"duplicate declaration {c.name!r}")
        c_sym = c.c_name or c.name
        param_types = []
        for p in c.params:
            ty = self._lower_type(p.type)
            # Mut user-tablet params cross the C ABI by pointer
            # (mirrors `mut tablets[N]T` semantics; matches how libc
            # writes through `struct sockaddr *addr`). Non-mut user
            # tablets pass by value — LLVM lowers them to the
            # platform's struct-arg ABI.
            if p.is_mut and self._struct_fields_for(ty) is not None:
                param_types.append(ty.as_pointer())
            else:
                # Buffers always pass as `T*` regardless of mut —
                # arrays can't be passed by value across C at all.
                param_types.append(self._colophon_c_ty(ty))
        ret_type = self._colophon_c_ty(
            self._lower_type(c.return_type) if c.return_type
            else ir.VoidType()
        )
        fn_type = ir.FunctionType(ret_type, param_types)
        existing = self.module.globals.get(c_sym)
        if existing is not None:
            # Another declaration (internal runtime helper or a prior
            # colophon resolved through the same C symbol) already
            # exists. Refuse to reuse it unless the signatures match —
            # a silent mismatch would emit correct-looking IR that
            # miscalls the C function. Users can always pick a
            # different Tuppu-side name; we reserve an explicit
            # C-symbol override for a future syntax pass.
            existing_ty = getattr(existing, "function_type", None)
            if existing_ty != fn_type:
                raise CodegenError(
                    f"colophon {c.name!r} collides with the compiler's "
                    f"internal {c_sym!r} extern (signature mismatch: "
                    f"declared {fn_type}, internal {existing_ty}). Pick a "
                    f"different name — the marshaler would silently "
                    f"misbehave otherwise."
                )
            llvm_fn = existing
        else:
            llvm_fn = ir.Function(self.module, fn_type, name=c_sym)
            for i, p in enumerate(c.params):
                llvm_fn.args[i].name = p.name
        self.functions[c.name] = llvm_fn
        self._colophon_decls[c.name] = c
        self._fn_param_mut[c.name] = [p.is_mut for p in c.params]

    def _colophon_c_ty(self, ty: ir.Type) -> ir.Type:
        """Map a Tuppu-side LLVM type to its C-ABI counterpart for
        extern signatures. `str` becomes `i8*` (pointer to NUL-
        terminated bytes); `bool` widens to `i8` for cross-platform
        stability; integer types pass through unchanged. A
        `buffer[N]T` decays to `T*` — the natural C-side shape for
        byte-buffer-taking fns like `recv`/`send`."""
        if isinstance(ty, ir.VoidType):
            return ty
        if self._is_str_value(ty):
            return I8.as_pointer()
        if isinstance(ty, ir.ArrayType):
            return ty.element.as_pointer()
        if ty == I1:
            return I8
        return ty

    def _str_to_cstr(self, s_val: ir.Value) -> ir.Value:
        """Emit `malloc(len+1) + memcpy(ptr, len) + NUL` to produce a
        fresh NUL-terminated C string from a Tuppu str value. The
        returned i8* is heap-owned by the call-site — it must be
        freed after the extern call returns."""
        assert self.builder is not None
        b = self.builder
        ptr = b.extract_value(s_val, 0)
        length = b.extract_value(s_val, 1)
        alloc_size = b.add(length, ir.Constant(I64, 1))
        raw = b.call(self._get_malloc(), [alloc_size])
        b.call(self._get_memcpy(), [raw, ptr, length])
        b.store(ir.Constant(I8, 0), b.gep(raw, [length], inbounds=True))
        return raw

    def _cstr_to_str(self, cstr: ir.Value) -> ir.Value:
        """Turn a C-returned i8* into a heap-owned Tuppu str via
        `strlen + malloc + memcpy`. The original C pointer is left
        untouched — Tuppu owns a copy — so callers returning pointers
        into static storage (getenv) or the stack don't force
        premature frees on the caller's side.

        NULL returns (getenv on a missing var, etc.) yield an empty
        borrow: `{ptr=null, len=0, cap=0}`. This collapses "not found"
        with "found empty string"; stdlib wrappers can distinguish by
        querying the raw env before the marshal if needed."""
        assert self.builder is not None
        b = self.builder
        fn = b.function
        is_null = b.icmp_signed(
            "==", cstr, ir.Constant(I8.as_pointer(), None),
        )
        null_bb = fn.append_basic_block("cstr.null")
        copy_bb = fn.append_basic_block("cstr.copy")
        done_bb = fn.append_basic_block("cstr.done")
        b.cbranch(is_null, null_bb, copy_bb)

        b.position_at_end(null_bb)
        empty = self._str_build_value_in(
            b, ir.Constant(I8.as_pointer(), None),
            ir.Constant(I64, 0), ir.Constant(I64, 0),
        )
        b.branch(done_bb)

        b.position_at_end(copy_bb)
        length = b.call(self._get_strlen(), [cstr])
        alloc_size = b.add(length, ir.Constant(I64, 1))
        raw = b.call(self._get_malloc(), [alloc_size])
        b.call(self._get_memcpy(), [raw, cstr, length])
        b.store(ir.Constant(I8, 0), b.gep(raw, [length], inbounds=True))
        copied = self._str_build_value_in(b, raw, length, length)
        b.branch(done_bb)

        b.position_at_end(done_bb)
        phi = b.phi(self._str_ty())
        phi.add_incoming(empty, null_bb)
        phi.add_incoming(copied, copy_bb)
        return phi

    def _gen_fn_value_call(
        self, fn_ptr: ir.Value, fn_ty: ir.FunctionType,
        arg_exprs: list[A.Expr],
    ) -> ir.Value | None:
        """Emit an indirect call through a precomputed fn-pointer value.
        Arg marshaling mirrors the direct-call path — str gets cap=0
        borrow, cleanup-bearing structs get field neutering, etc. — so
        users can't leak or UAF by routing a call through a pointer
        instead of calling by name."""
        assert self.builder is not None
        if len(arg_exprs) != len(fn_ty.args):
            raise CodegenError(
                f"fn-value call expects {len(fn_ty.args)} args, "
                f"got {len(arg_exprs)}"
            )
        call_args: list[ir.Value] = []
        for arg, expected_ty in zip(arg_exprs, fn_ty.args):
            v = self._gen_expr(arg)
            if v is None:
                raise CodegenError("fn-value call arg has no value")
            coerced = self._coerce(v, expected_ty)
            if self._is_str_value(expected_ty):
                coerced = self._str_as_borrow(coerced)
            elif (
                self._struct_fields_for(expected_ty) is not None
                and self._struct_needs_cleanup(expected_ty)
            ):
                coerced = self._struct_as_borrow(coerced, expected_ty)
            call_args.append(coerced)
        return self.builder.call(fn_ptr, call_args)

    def _gen_colophon_call(
        self, decl: A.ColophonDecl, llvm_fn: ir.Function,
        arg_exprs: list[A.Expr],
    ) -> ir.Value | None:
        """Lower a call to a colophon-declared extern. Marshals each
        str arg to a fresh cstr buffer, widens bool to i8, passes
        ints through; after the call, frees every cstr we allocated
        and converts an i8* return back into a heap-owned Tuppu str.
        Void return yields None."""
        if len(arg_exprs) != len(decl.params):
            raise CodegenError(
                f"colophon {decl.name!r} expects {len(decl.params)} args, "
                f"got {len(arg_exprs)}"
            )
        assert self.builder is not None
        b = self.builder
        call_args: list[ir.Value] = []
        temp_cstrs: list[ir.Value] = []
        for arg_expr, param in zip(arg_exprs, decl.params):
            param_ty = self._lower_type(param.type)
            # Buffer arg: decay to element pointer via GEP [0, 0]. The
            # arg must name a buffer-typed mut binding so we can take
            # the address of its alloca directly.
            if isinstance(param_ty, ir.ArrayType):
                if not isinstance(arg_expr, A.Ident):
                    raise CodegenError(
                        f"colophon {decl.name!r}: buffer arg must be a "
                        f"buffer-typed Ident, got {type(arg_expr).__name__}"
                    )
                var = self._lookup(arg_expr.name)
                if not var.is_mut or var.value_ty != param_ty:
                    raise CodegenError(
                        f"colophon {decl.name!r}: buffer arg "
                        f"{arg_expr.name!r} must be a mut binding of "
                        f"type {param_ty}"
                    )
                elem_ptr = b.gep(
                    var.ir_ref,
                    [ir.Constant(I32, 0), ir.Constant(I32, 0)],
                    inbounds=True,
                )
                call_args.append(elem_ptr)
                continue
            # Mut user-tablet arg: pass the caller's alloca address so
            # the callee can read/write through it (sockaddr out-params,
            # mut pointer-to-struct libc conventions). The call site
            # must be a mut-bound Ident naming a matching struct.
            if (
                param.is_mut
                and self._struct_fields_for(param_ty) is not None
            ):
                if not isinstance(arg_expr, A.Ident):
                    raise CodegenError(
                        f"colophon {decl.name!r}: mut struct arg must be "
                        f"a mut-bound Ident, got {type(arg_expr).__name__}"
                    )
                var = self._lookup(arg_expr.name)
                if not var.is_mut or var.value_ty != param_ty:
                    raise CodegenError(
                        f"colophon {decl.name!r}: mut struct arg "
                        f"{arg_expr.name!r} must be a mut binding "
                        f"of type {param_ty}"
                    )
                call_args.append(var.ir_ref)
                continue
            v = self._gen_expr(arg_expr)
            if v is None:
                raise CodegenError(
                    f"colophon {decl.name!r} arg has no value"
                )
            v = self._coerce(v, param_ty)
            if self._is_str_value(param_ty):
                cstr = self._str_to_cstr(v)
                temp_cstrs.append(cstr)
                call_args.append(cstr)
            elif param_ty == I1:
                call_args.append(b.zext(v, I8))
            else:
                call_args.append(v)
        result = b.call(llvm_fn, call_args)
        for cstr in temp_cstrs:
            b.call(self._get_free(), [cstr])
        if decl.return_type is None:
            return None
        ret_ty = self._lower_type(decl.return_type)
        if self._is_str_value(ret_ty):
            return self._cstr_to_str(result)
        if ret_ty == I1:
            return b.icmp_signed("!=", result, ir.Constant(I8, 0))
        return result

    def _gen_fn_body(self, fn: A.FnDecl) -> None:
        if fn.name == "main":
            if not (isinstance(fn.return_type, A.TypeName) and fn.return_type.name == "i32"):
                raise CodegenError("main must declare -> i32")

        llvm_fn = self.functions[fn.name]
        entry = llvm_fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        self.scopes = [{}]

        # Params live in a dedicated cleanup frame that wraps the fn body.
        # A mut str param needs release at scope exit so a reassignment
        # to a heap-owned str doesn't leak; the incoming value is a
        # borrow (caller forced cap=0), so the initial release is a no-op.
        # Non-mut str params stay SSA — they can't be reassigned, and the
        # cap=0 borrow has nothing to free.
        self._push_cleanup_frame()
        try:
            # Parameters: step-bound (direct SSA ref) unless the user wrote
            # `mut` — in which case we alloca + store the incoming arg and
            # bind the alloca, so methods requiring a mut binding (notably
            # `tablets.push`) work on the parameter.
            #
            # Special case: mut tablets params arrive already as a pointer
            # to the caller's storage (see `_declare_fn`). We bind the
            # incoming pointer directly as the Variable's ir_ref — no
            # alloca+store — so method dispatch gets a stable pointer to
            # the caller's tablets and mutations persist.
            for i, p in enumerate(fn.params):
                arg = llvm_fn.args[i]
                param_decl_ty = self._lower_type(p.type)
                is_mut_tablets = (
                    p.is_mut and self._tablets_info_for(param_decl_ty) is not None
                )
                is_variadic = isinstance(p.type, A.TypeVariadicTablets)
                is_mut_struct = (
                    p.is_mut
                    and self._struct_fields_for(param_decl_ty) is not None
                    and not self._is_str_value(param_decl_ty)
                )
                if is_mut_tablets or is_variadic or is_mut_struct:
                    # Either shape arrives as a pointer to the caller's
                    # tablets or struct storage; bind the incoming
                    # pointer directly as the Variable's ir_ref so the
                    # body's indexing, iteration, field access, and
                    # method dispatch all work on the caller's actual
                    # storage. No cleanup registration — the caller
                    # owns the memory.
                    self.scopes[-1][p.name] = Variable(
                        is_mut=True, ir_ref=arg, value_ty=param_decl_ty,
                    )
                elif p.is_mut:
                    slot = self._alloca_entry(arg.type, p.name)
                    self.builder.store(arg, slot)
                    self.scopes[-1][p.name] = Variable(
                        is_mut=True, ir_ref=slot, value_ty=arg.type,
                    )
                    self._maybe_register_cleanup(p.name, arg.type, slot)
                elif self._type_desc_key(arg.type) is not None:
                    # Non-mut cleanup-bearing param: spill to a shadow-
                    # stack-rooted slot so GC cycles during the body
                    # see it as a root. Without this, a param passed
                    # as-is through SSA is invisible to the collector
                    # and can be prematurely reclaimed when a callee
                    # triggers GC. Cleanup release is still a no-op
                    # for borrowed-semantics params (caller owns).
                    slot = self._alloca_entry(arg.type, p.name)
                    self.builder.store(arg, slot)
                    # Bind SSA (the incoming value) so reads don't go
                    # through the slot — the slot is a root spill only.
                    # `.ir_ref` remains the SSA value for downstream
                    # ident reads that expect a value, not a pointer.
                    self.scopes[-1][p.name] = Variable(
                        is_mut=False, ir_ref=arg, value_ty=arg.type,
                    )
                    self._register_gc_root(slot, arg.type)
                else:
                    self.scopes[-1][p.name] = Variable(
                        is_mut=False, ir_ref=arg, value_ty=arg.type,
                    )

            value = self._gen_expr(fn.body)

            if self._is_terminated():
                # Body already returned via yield — the yield path unwound
                # every live cleanup frame (including this one).
                return
            if fn.return_type is None:
                self._emit_frame_cleanups(self._cleanup_frames[-1])
                self._emit_all_gc_root_pops_for_early_return()
                self.builder.ret_void()
            else:
                if value is None:
                    raise CodegenError(
                        f"function {fn.name!r} must produce a value for return type "
                        f"{fn.return_type}, but its body has no trailing expression"
                    )
                expected = self._lower_type(fn.return_type)
                coerced = self._coerce(value, expected)
                # Block-level codegen already clones Field/Index tails
                # so the caller gets independently-owned bytes
                # (see `_gen_block`). No second neuter here; cloning
                # twice would leave the first clone's heap bytes
                # unrooted across the second clone's allocation.
                self._emit_frame_cleanups(self._cleanup_frames[-1])
                self._emit_all_gc_root_pops_for_early_return()
                self.builder.ret(coerced)
        finally:
            self._pop_cleanup_frame()

    def _block_tail_expr(self, e: "A.Expr") -> "A.Expr | None":
        """Find the source expression for a fn/block's tail value, if
        any. Drills through nested blocks so `{ ... { x.y } }` returns
        the same expr as `x.y`. Returns None if the tail is missing
        or the expression has no value."""
        if isinstance(e, A.Block):
            if e.tail is None:
                return None
            return self._block_tail_expr(e.tail)
        return e

    def _is_terminated(self) -> bool:
        assert self.builder is not None
        return self.builder.block.is_terminated


    # --- scope / bindings ---

    def _bind(self, name: str, var: Variable) -> None:
        if name in self.scopes[-1]:
            raise CodegenError(f"redefinition of {name!r} in same scope")
        self.scopes[-1][name] = var

    def _lookup(self, name: str) -> Variable:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise CodegenError(f"undefined name {name!r}")

    def _alloca_entry(self, ty: ir.Type, name: str) -> ir.Value:
        """Emit an alloca in the entry block so mem2reg can promote it later."""
        assert self.builder is not None
        saved = self.builder.block
        entry = self.builder.function.entry_basic_block
        self.builder.position_at_start(entry)
        slot = self.builder.alloca(ty, name=name)
        self.builder.position_at_end(saved)
        return slot

def codegen(program: A.Program, checker=None) -> ir.Module:
    return Codegen(checker=checker).gen(program)
