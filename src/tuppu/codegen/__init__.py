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
    get_gc_framework,
    I1, I8, I16, I32, I64,
    INTRINSICS, INT_WIDTH,
    RAT, SEX, SEX_MAX_DIGITS,
    SEX_IDX_DIGITS, SEX_IDX_RADIX, SEX_IDX_COUNT, SEX_IDX_SIGN,
    TabletsInfo, Variable,
)


# Sentinel tag value for empty / cleared seal slots in `llvm` GC
# framework mode. Any slot rooted via `@llvm.gcroot` is rooted for
# the function's full lifetime; we mark a slot as "empty — do not
# trace" by writing this byte at the tag offset, and the codegen-
# emitted seal trace fns short-circuit on it. The choice of 0xFF
# is structural: no real seal in the language can have 256 variants
# (variant indexing is i8, but we'd run out of space long before),
# so 0xFF can't collide with a live tag. Using a sentinel rather
# than zero-clearing keeps slot-clear cost at one store while
# being safe regardless of the per-seal "is variant 0 trace-safe
# under zero payload" property that zero-clearing would rely on.
GC_SEAL_EMPTY_TAG = 0xFF
from .access import AccessMixin
from .dvec import DVecMixin
from .expr import ExprMixin
from .intrinsics import IntrinsicsMixin
from .ivec import IVecMixin
from .module import ModuleMixin
from .rat import RatMixin
from .seals import SealsMixin
from .sex import SexMixin
from .stmt import StmtMixin
from .strs import StrsMixin
from .tablets import TabletsMixin
from .types import TypesMixin


class Codegen(
    SexMixin, RatMixin, TabletsMixin, IVecMixin, DVecMixin, StrsMixin, SealsMixin,
    ExprMixin, StmtMixin, IntrinsicsMixin, AccessMixin, TypesMixin, ModuleMixin,
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
        # ivec per-T helper-fn cache. Keyed by (str(elem_ty), is_wedge).
        # See ivec.py for layout discussion.
        self._ivec_types: dict = {}
        # dvec per-T cache (helper fns + per-T storage descriptor).
        # Keyed by (str(elem_ty), is_wedge); each entry holds a
        # buffer trace fn that walks inline T slots.
        self._dvec_types: dict = {}
        # User-defined structs: name -> LLVM struct type + ordered fields.
        # Per-block stack of cleanups (just tablets releases for now).
        # Each entry: (release_fn, ptr, source_name). Pushed at block
        # entry, popped at block exit (emitting releases along the way).
        # Mirrors the scope stack — same push/pop cadence.
        self._cleanup_frames: list[list[tuple[ir.Function, ir.Value, str]]] = []
        # Parallel to _cleanup_frames: count of GC roots pushed into
        # the innermost frame, popped at frame exit. Used in `shadow`
        # GC mode to balance push_root / pop_roots; in `llvm` mode
        # it counts the same per-frame roots, but the pop site emits
        # slot-clear stores instead (see `_gc_root_slots_per_frame`).
        self._gc_root_counts: list[int] = []
        # Parallel to _cleanup_frames in `llvm` GC mode: each frame
        # carries the list of (slot, kind, value_ty) tuples whose
        # contents must be sentinel-cleared at frame exit. `kind`
        # is one of "seal" / "scalar" / "wedge" — drives whether we
        # write the 0xFF tag, zero-fill the slot, or null the wedge
        # ptr. Empty in `shadow` mode (the count is sufficient there).
        self._gc_root_slots_per_frame: list[list[tuple[ir.Value, str, ir.Type]]] = []
        # GC mode (frozen at codegen construction time — set via the
        # TUPPU_GC_FRAMEWORK env var). `shadow` keeps the legacy
        # push/pop array; `llvm` switches to @llvm.gcroot + shadow-
        # stack strategy. Read by `_register_gc_root` and
        # `_emit_gc_frame_pop` to dispatch. See _common.py for
        # rationale. Re-read per-instance so a single test process
        # can sweep both modes by toggling env var between compiles.
        self._gc_mode = get_gc_framework()
        # `llvm` mode: pending @llvm.gcroot calls to emit at the
        # entry block once the fn body is fully generated. Each entry
        # is (shadow_slot, real_slot, descriptor_global, kind, value_ty).
        # `shadow_slot` is an i8* alloca whose stored value is the
        # i8*-cast address of the real (struct-shaped) slot — that's
        # what gcroot wants. The runtime sees `*shadow_slot ==
        # &real_slot` and dispatches via the descriptor's trace_fn.
        # Reset at fn-entry; finalized by `_finalize_pending_gcroots`.
        self._pending_gcroots: list[tuple[
            ir.Value, ir.Value, ir.GlobalVariable, str, ir.Type,
        ]] = []
        # Set of LLVM fn objects that need the `gc "shadow-stack"`
        # attribute injected during the IR-text post-process pass
        # (llvmlite doesn't expose Function.gc directly). Populated
        # by `_finalize_pending_gcroots` only for fns that actually
        # registered at least one gcroot — fns without roots get no
        # attribute so the strategy emitter doesn't insert empty
        # StackEntries. See driver._inject_gc_strategy.
        self._fns_needing_gc_attr: set[str] = set()
        # `llvm` mode: lazy-declared @llvm.gcroot intrinsic.
        self._llvm_gcroot: ir.Function | None = None
        # Tracks the slot the most recent `_force_root_cleanup_value`
        # call registered (or None if it didn't register one — borrow
        # source, helper-fn emission, no descriptor). Reset on entry to
        # the chokepoint so the value after a `_gen_expr` call reflects
        # ONLY that call's outermost chokepoint, never a stale earlier
        # one. `_gen_fn_body` reads this to transfer the return value's
        # cleanup out by slot identity rather than by frame-position.
        self._last_rvalue_root_slot: ir.Value | None = None
        # Module context for type-name lookup. Set by phase-emit
        # callers per-decl so `_lower_type(TypeName(name="Foo"))`
        # consults the right module's visible scope to translate
        # `Foo` to its mangled flat key in `_struct_types`.
        self._codegen_current_module: tuple[str, ...] = ()
        self._struct_types: dict[str, ir.LiteralStructType] = {}
        self._struct_fields: dict[str, list[tuple[str, ir.Type]]] = {}
        # Per-struct set of field indices declared as `wedge T` at the
        # source level. LLVM type-level info loses the distinction
        # between `wedge T` and `*T` (both lower to `T*`); we keep it
        # here so the GC trace fn emitter knows which pointer slots
        # need interior-pointer marking via `__tuppu_gc_mark_wedge`
        # vs. regular `__tuppu_gc_mark_ptr`. Mirrored by
        # `_struct_mono_wedge_idxs` for monomorphizations.
        self._struct_wedge_idxs: dict[str, set[int]] = {}
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
        # Wedge field indices for monomorphized structs — see
        # `_struct_wedge_idxs` for rationale.
        self._struct_mono_wedge_idxs: dict[tuple, set[int]] = {}
        # Per-seal, per-variant set of payload field indices declared
        # as `wedge T`. Keyed by (seal_key, variant_idx). Mirrors the
        # struct-side wedge tracking; consumed by the seal trace fn.
        self._seal_wedge_idxs: dict[tuple, set[int]] = {}
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
        payloads).

        A type needs a descriptor if it carries cleanup-bearing fields
        (str / tablets / nested struct/seal-with-cleanup) OR if it
        carries a wedge anywhere in its layout — wedges aren't
        cleanup-bearing themselves (they're non-owning) but the GC
        still needs to mark through them via interior-pointer lookup
        to keep the pointed-into chunk alive."""
        if self._is_str_value(value_ty):
            return "__tuppu_str"
        if self._is_ivec_value(value_ty):
            # All ivec values share one descriptor — the buf pointer at
            # offset 0 is GC-traced (the storage's runtime trace fn
            # walks the per-cap pointer slots), and len/cap are scalar.
            return "__tuppu_ivec"
        if self._is_dvec_value(value_ty):
            # Like ivec, the dvec value's descriptor is shared (one
            # ptr-field at offset 0). The PER-T variation lives on the
            # buffer's descriptor, not the dvec value's.
            return "__tuppu_dvec"
        info = self._tablets_info_for(value_ty)
        if info is not None:
            wedge_tag = "_w" if info.elem_is_wedge else ""
            return f"__tuppu_tbls_{info.elem_ty}_{info.N}{wedge_tag}".replace(" ", "_")
        if isinstance(value_ty, ir.IdentifiedStructType):
            if self._seal_key_for_ty(value_ty) is not None:
                if not (
                    self._seal_needs_cleanup(value_ty)
                    or self._contains_wedge_anywhere(value_ty)
                ):
                    return None
                return f"__tuppu_seal_{value_ty.name}"
            # Plain structs: only produce a desc when there's something
            # to trace. `_get_type_desc` and the chokepoint both key off
            # this, so keeping them agreed prevents push/pop counter
            # drift (alloca a rooted slot but skip the push, vs. pop
            # expecting a push that never happened).
            if (
                self._struct_needs_cleanup(value_ty)
                or self._contains_wedge_anywhere(value_ty)
            ):
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
        if self._is_ivec_value(value_ty):
            # buf at 0 (leaf bytes — kept alive but not traced through),
            # head_node at 24, tail_node at 32. The chunks reached
            # through head/tail trace their own per-T slot contents.
            return [0, 24, 32]
        if self._is_dvec_value(value_ty):
            return [0]   # buf ptr; len/cap are scalar i64
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
        elem_is_wedge: bool = False,
    ) -> ir.GlobalVariable:
        """Emit (or return cached) `tuppu_type_t` for a tablets chunk.
        Chunks allocate via `__tuppu_gc_alloc(size, &chunk_desc)` so
        GC marks through them. Element types that recursively hold a
        seal or wedge field — and tablets-of-wedge themselves — get a
        per-chunk trace fn that walks each slot via the same
        alignment-aware composition as the struct trace fns. Plain
        elements stick to a flat ptr_offsets table.

        `elem_is_wedge` reflects the source-level `tablets[N]wedge T`
        case: each chunk slot holds a single interior pointer that
        must be dispatched through `__tuppu_gc_mark_wedge` rather than
        a flat ptr_offsets entry (which would always emit mark_ptr)."""
        wedge_tag = "_w" if elem_is_wedge else ""
        key = f"__tuppu_chunk_{elem_ty}_{N}{wedge_tag}".replace(" ", "_")
        cached = self._type_descs.get(key)
        if cached is not None:
            return cached
        size = N * self._size_of(elem_ty) + 16
        # If the element layout transitively includes a seal-with-
        # cleanup field, a wedge field, or this tablets is itself a
        # tablets-of-wedge, the flat ptr-offsets approach can't model
        # the dispatch correctly. Emit a chunk trace fn that loops
        # over slots and recurses through the element's full tracing
        # logic, with mark_wedge for wedge slots.
        needs_trace = (
            elem_is_wedge
            or self._contains_seal_anywhere(elem_ty)
            or self._contains_wedge_anywhere(elem_ty)
        )
        if needs_trace:
            offsets: list[int] = []
            trace_fn: ir.Function | None = self._get_chunk_trace_fn(
                key, N, elem_ty, node_ty, elem_is_wedge=elem_is_wedge,
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
        elem_is_wedge: bool = False,
    ) -> ir.Function:
        """Per-chunk trace fn for chunks whose element layout includes
        a seal, a wedge, or whose element is itself a wedge T.
        Walks all N slots — the chunk header's `used` field tells the
        runtime how many to mind, but unused slots are calloc-zero, so
        marking them is a safe no-op via mark_ptr's null check. Also
        marks the `next` chunk pointer.

        When `elem_is_wedge`, each slot holds a `wedge T` (interior
        pointer); we dispatch through `_emit_trace_mark_calls` with
        `is_wedge_field=True` so the slot value gets routed through
        mark_wedge."""
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
            self._emit_trace_mark_calls(
                b, base, i * elem_size, elem_ty,
                is_wedge_field=elem_is_wedge,
            )
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
        # layout. Structs that transitively contain a seal or a wedge
        # field need the same escape hatch — seals so GC can recurse
        # into the seal's trace fn, wedges so the trace fn can route
        # them through mark_wedge instead of mark_ptr (flat
        # ptr_offsets always uses mark_ptr, which would silently
        # collect the chunk a wedge points into). Everything else
        # gets a flat ptr_offsets table.
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
            and (
                self._struct_contains_seal(value_ty)
                or self._contains_wedge_anywhere(value_ty)
            )
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
                wedge_idxs = self._seal_wedge_idxs.get((seal_key, idx), set())
                fld_off = 0
                for fi, fty in enumerate(payload_ty.elements):
                    align = self._align_of(fty)
                    fld_off = (fld_off + align - 1) // align * align
                    self._emit_trace_mark_calls(
                        b, payload_base, fld_off, fty,
                        is_wedge_field=(fi in wedge_idxs),
                    )
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
        is_wedge_field: bool = False,
    ) -> None:
        """Emit the IR that marks every GC-reachable pointer inside
        `field_ty` at `base + offset`. Nested seals dispatch into
        their own trace fn; structs that (transitively) contain a
        seal or wedge also dispatch via a struct trace fn so the
        tag-/wedge-aware recursion chains through. Everything else
        falls back to a flat ptr_offsets walk. Scalars are no-ops.

        `is_wedge_field` is set by the caller (a struct/seal/chunk
        trace fn) when the parent declared this slot as `wedge T`.
        In that case the slot holds a single interior pointer; we
        load it and call mark_wedge so the GC can find the chunk
        the wedge points into. Without this flag, wedge slots would
        fall through the flat ptr_offsets path and get mark_ptr'd —
        which only handles object-start pointers (chunk's HDR has
        the magic byte; an interior wedge does not), so the chunk
        would be silently swept. That was the v0.4.1 soundness bug."""
        if is_wedge_field:
            field_i8 = b.gep(
                base, [ir.Constant(I64, offset)], inbounds=True,
            )
            ptr_ptr = b.bitcast(field_i8, I8.as_pointer().as_pointer())
            wedge_ptr = b.load(ptr_ptr)
            b.call(self._get_gc_mark_wedge(), [wedge_ptr])
            return
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
            if (
                self._struct_contains_seal(field_ty)
                or self._contains_wedge_anywhere(field_ty)
            ):
                inner_fn = self._get_struct_trace_fn(field_ty)
                sub_ptr = b.gep(
                    base, [ir.Constant(I64, offset)], inbounds=True,
                )
                b.call(inner_fn, [sub_ptr])
                return
        # Literal struct (e.g. variant payload tuple) that contains a
        # seal or wedge field still needs recursion — walk fields one
        # by one so nested seal/wedge fields re-enter the dispatch.
        if isinstance(field_ty, ir.LiteralStructType):
            if any(
                self._contains_seal_anywhere(el)
                or self._contains_wedge_anywhere(el)
                for el in field_ty.elements
            ):
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

    def _contains_wedge_anywhere(self, ty: ir.Type) -> bool:
        """Does `ty` anywhere in its layout hold a `wedge T` slot the
        parent's trace fn needs to dispatch via mark_wedge? Returns
        True for direct wedge fields and for sub-fields whose own type
        recursively contains wedges (so the parent must call into the
        inner type's trace fn rather than using flat ptr_offsets that
        would mark wedges as object-start pointers).

        Returns False for tablets-of-wedge — the chunk descriptor
        handles its own wedge slots, so a parent struct holding a
        `tablets[N]wedge T` field can still use flat offsets for
        head/tail."""
        if isinstance(ty, ir.IdentifiedStructType):
            seal_key = self._seal_key_for_ty(ty)
            if seal_key is not None:
                variants = self._seal_variants.get(seal_key, [])
                for vi, (_vn, payload) in enumerate(variants):
                    if self._seal_wedge_idxs.get((seal_key, vi)):
                        return True
                    for el in payload.elements:
                        if self._contains_wedge_anywhere(el):
                            return True
                return False
            if self._struct_wedge_idxs_for(ty):
                return True
            fields = self._struct_fields_for(ty) or []
            for _name, fty in fields:
                if self._contains_wedge_anywhere(fty):
                    return True
            return False
        if isinstance(ty, ir.LiteralStructType):
            return any(self._contains_wedge_anywhere(el) for el in ty.elements)
        if isinstance(ty, ir.ArrayType):
            return self._contains_wedge_anywhere(ty.element)
        return False

    def _get_struct_trace_fn(self, struct_ty: ir.Type) -> ir.Function:
        """Emit (or return cached) a trace fn for a struct whose
        layout can't be expressed as a flat ptr_offsets table —
        structs that transitively hold a seal-with-cleanup field or
        a `wedge T` field. The fn walks each field and recurses via
        `_emit_trace_mark_calls`, which handles seals by calling
        their own trace fn in turn and dispatches wedge fields via
        mark_wedge."""
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
        wedge_idxs = self._struct_wedge_idxs_for(struct_ty)
        offset = 0
        for fi, (_name, fty) in enumerate(fields):
            align = self._align_of(fty)
            offset = (offset + align - 1) // align * align
            self._emit_trace_mark_calls(
                b, base, offset, fty, is_wedge_field=(fi in wedge_idxs),
            )
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

    def _get_gc_data_size(self) -> ir.Function:
        """`__tuppu_gc_data_size(p) -> size_t` — return the bytes-of-
        data following p's GC header, or 0 if p isn't a GC allocation.
        Used by codegen-emitted trace fns that need to recover an
        allocation's element count without mirroring the header
        layout."""
        cached = self.module.globals.get("__tuppu_gc_data_size")
        if isinstance(cached, ir.Function):
            return cached
        fty = ir.FunctionType(I64, [I8.as_pointer()])
        return ir.Function(self.module, fty, "__tuppu_gc_data_size")

    def _get_gc_mark_wedge(self) -> ir.Function:
        """`__tuppu_gc_mark_wedge(ptr)` — interior-pointer mark for
        wedge fields. The runtime walks the live-list to find the
        chunk whose `[start, end)` contains the wedge, then marks
        that chunk via the standard mark_ptr path so its descriptor
        keeps the rest of the forward chain alive. Trace fns dispatch
        here for any field declared as `wedge T` at the source level
        (LLVM type alone can't tell `wedge T` from `*T`)."""
        cached = self.module.globals.get("__tuppu_gc_mark_wedge")
        if isinstance(cached, ir.Function):
            return cached
        fty = ir.FunctionType(ir.VoidType(), [I8.as_pointer()])
        return ir.Function(self.module, fty, "__tuppu_gc_mark_wedge")

    def _get_wedge_trace_fn(self) -> ir.Function:
        """Trace fn for the shared wedge descriptor. Loads the wedge
        pointer out of the rooted slot and dispatches to mark_wedge so
        the GC reaches the chunk the wedge points into. Without this,
        a wedge held across allocations whose source ivec / tablets has
        gone out of scope (the marquee arena property) is the only
        path to its chunk — and the chunk is silently swept."""
        cached = self.module.globals.get("__tuppu_wedge_trace")
        if isinstance(cached, ir.Function):
            return cached
        fn = ir.Function(self.module, self._trace_fn_ty, "__tuppu_wedge_trace")
        fn.linkage = "internal"
        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        slot_i8 = fn.args[0]
        # Slot holds an i8* (the wedge value). Load it and mark.
        ptr_ptr = b.bitcast(slot_i8, I8.as_pointer().as_pointer())
        wedge_val = b.load(ptr_ptr)
        b.call(self._get_gc_mark_wedge(), [wedge_val])
        b.ret_void()
        return fn

    def _get_wedge_descriptor(self) -> ir.GlobalVariable:
        """Single shared `tuppu_type_t` for any `wedge T` value: the
        wedge descriptor is T-independent because mark_wedge does its
        own interior-pointer chunk lookup. Size 8 (one pointer), no
        flat ptr_offsets — the trace fn takes over and dispatches
        through mark_wedge."""
        key = "__tuppu_wedge"
        cached = self._type_descs.get(key)
        if cached is not None:
            return cached
        # Empty offsets array (n_ptrs = 0; the trace fn supersedes it).
        offsets_arr_ty = ir.ArrayType(I64, 1)
        offsets_arr = ir.GlobalVariable(
            self.module, offsets_arr_ty, f"{key}_offsets",
        )
        offsets_arr.linkage = "internal"
        offsets_arr.global_constant = True
        offsets_arr.initializer = ir.Constant(
            offsets_arr_ty, [ir.Constant(I64, 0)],
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
        desc = ir.GlobalVariable(self.module, self._type_desc_ty, key)
        desc.linkage = "internal"
        desc.global_constant = True
        desc.initializer = ir.Constant(self._type_desc_ty, [
            name_arr.bitcast(I8.as_pointer()),
            ir.Constant(I64, 8),  # size: a single pointer
            ir.Constant(I64, 0),  # n_ptrs: 0 — trace fn supersedes
            offsets_arr.bitcast(I64.as_pointer()),
            self._get_wedge_trace_fn(),
        ])
        self._type_descs[key] = desc
        return desc

    def _emit_gc_push_root(self, slot: ir.Value, value_ty: ir.Type) -> bool:
        """Register a stack slot as a GC root.

        In `shadow` mode: emits an inline `__tuppu_gc_push_root(slot,
        &type_desc)` call at the current builder position. The matching
        pop_roots(n) at frame exit balances out — see `_emit_gc_pop_roots`.

        In `llvm` mode: queues an `@llvm.gcroot(shadow_slot,
        descriptor_meta)` for emission in the fn's entry block. Also
        emits an inline "safe-state" store at the current builder
        position so the slot is trace-safe before any subsequent GC
        trigger. The slot is rooted for the fn's full lifetime; clearing
        at the original pop site is handled by `_emit_gc_pop_roots`.

        Returns True if a root was registered (the type had traceable
        pointer fields), False otherwise so the caller skips
        bookkeeping."""
        desc = self._get_type_desc(value_ty)
        if desc is None:
            return False
        b = self.builder
        assert b is not None
        if self._gc_mode == "shadow":
            b.call(
                self._get_gc_push_root(),
                [
                    b.bitcast(slot, I8.as_pointer()),
                    b.bitcast(desc, I8.as_pointer()),
                ],
            )
            return True
        kind = "seal" if (
            isinstance(value_ty, ir.IdentifiedStructType)
            and self._seal_key_for_ty(value_ty) is not None
        ) else "scalar"
        self._queue_gcroot(slot, desc, kind, value_ty)
        return True

    def _emit_gc_pop_roots(self, n: int) -> None:
        """Mark the most recently rooted N slots as no-longer-live.

        In `shadow` mode: emits the matching `__tuppu_gc_pop_roots(n)`
        call so the runtime's flat array stops scanning them.

        In `llvm` mode: emits a slot-clear store for each — `tag = 0xFF`
        for seal slots, zero-fill for everything else. The gcroot
        intrinsic still keeps the slot rooted for the fn lifetime, but
        the trace fns short-circuit on the sentinel so a stale entry
        contributes nothing to the mark phase."""
        if n <= 0:
            return
        b = self.builder
        assert b is not None
        if self._gc_mode == "shadow":
            b.call(self._get_gc_pop_roots(), [ir.Constant(I64, n)])
            return
        # `llvm` mode: walk the innermost frame's slot list, peel off
        # the last n entries (in registration order — outermost first),
        # and emit a sentinel-clear for each. We iterate the per-frame
        # list in reverse so the most-recent registration clears first.
        if not self._gc_root_slots_per_frame:
            return
        slots = self._gc_root_slots_per_frame[-1]
        if not slots:
            return
        to_clear = slots[-n:] if n <= len(slots) else slots[:]
        for slot, kind, value_ty in to_clear:
            self._emit_slot_clear(slot, kind, value_ty)

    # ------------------------------------------------------------------
    # `llvm` GC mode helpers
    # ------------------------------------------------------------------

    def _get_llvm_gcroot(self) -> ir.Function:
        """`@llvm.gcroot(i8** %ptrloc, i8* %metadata)` — the LLVM
        framework intrinsic. The slot must be an `i8**`-typed alloca
        the strategy emitter rewrites to a slot inside the fn's
        StackEntry; metadata is opaque (we pass our type descriptor's
        i8*-cast address) and ends up in `FrameMap::Meta[]` for the
        runtime walker to dispatch on."""
        if self._llvm_gcroot is None:
            fty = ir.FunctionType(
                ir.VoidType(), [I8.as_pointer().as_pointer(), I8.as_pointer()],
            )
            self._llvm_gcroot = self._get_or_declare_libc(
                "llvm.gcroot", fty,
            )
        return self._llvm_gcroot

    def _queue_gcroot(
        self, real_slot: ir.Value, desc: ir.GlobalVariable,
        kind: str, value_ty: ir.Type,
    ) -> None:
        """Queue a `@llvm.gcroot` for fn-entry emission, alongside the
        shadow slot setup and cleanup-frame bookkeeping. The actual
        gcroot intrinsic call AND the init slot-clear are deferred
        to `_finalize_pending_gcroots` so they both land at the start
        of the entry block — before any user store could observe the
        slot. Emitting the init-clear at the current builder position
        (which is mid-body, AFTER the chokepoint's store-to-slot)
        would overwrite live data; that was the v0.4.2-prerelease bug.

        `kind` ∈ {seal, scalar, wedge} drives the matching slot-clear
        at every pop site. The shadow slot store DOES happen at the
        current position — we need `*shadow_slot = &real_slot` to be
        true by the time the first GC could fire after registration,
        which is here. The fn-entry init-clear handles the "GC fires
        before this point" hazard separately."""
        b = self.builder
        assert b is not None
        # Allocate the shadow i8* slot in the entry block and stash the
        # real slot's address there. The strategy emitter rewrites this
        # alloca to point into the fn's StackEntry — that's how the
        # runtime walker reads the value out at mark time.
        shadow_slot = self._alloca_entry(I8.as_pointer(), "gc.shadow")
        b.store(b.bitcast(real_slot, I8.as_pointer()), shadow_slot)
        self._pending_gcroots.append(
            (shadow_slot, real_slot, desc, kind, value_ty),
        )
        # Per-frame slot list: each `_emit_gc_pop_roots(n)` site walks
        # this to know which slots to sentinel-clear.
        if self._gc_root_slots_per_frame:
            self._gc_root_slots_per_frame[-1].append(
                (real_slot, kind, value_ty),
            )

    def _emit_slot_clear(
        self, slot: ir.Value, kind: str, value_ty: ir.Type,
    ) -> None:
        """Write the sentinel "no-trace" state into a rooted slot.
        Drives the always-rooted-hazard strategy: in `llvm` mode every
        slot stays rooted for the fn lifetime, so we need the *contents*
        to read as empty whenever the fn isn't actively using it.

        - `seal`: store 0xFF at the tag byte (offset 0). Generated
          seal trace fns dispatch on tag and short-circuit on this
          sentinel (see seals.py:_get_seal_trace_fn).
        - `wedge`: store NULL at the slot. `mark_wedge(NULL)` no-ops.
        - `scalar`: zero-fill the whole slot via memset. Pointer fields
          read as NULL, which is null-safe for `mark_ptr`."""
        b = self.builder
        assert b is not None
        if kind == "seal":
            # Tag byte at slot offset 0. The seal type is
            # `{i8 tag, [N x i64] payload}`; we write only the tag.
            tag_ptr = b.bitcast(slot, I8.as_pointer())
            b.store(ir.Constant(I8, GC_SEAL_EMPTY_TAG), tag_ptr)
            return
        if kind == "wedge":
            # The slot holds a `T*` (the wedge value); store a null of
            # the same pointee type so llvmlite's type-checker is
            # happy. mark_wedge(NULL) is a no-op.
            null = ir.Constant(slot.type.pointee, None)
            b.store(null, slot)
            return
        # `scalar`: zero-fill the whole slot. memset is the simplest
        # path — doing one store per pointer field would also work
        # but adds codegen complexity for marginal IR savings.
        size = self._size_of(value_ty)
        if size == 0:
            return
        slot_i8 = b.bitcast(slot, I8.as_pointer())
        b.call(
            self._get_memset(),
            [slot_i8, ir.Constant(I8, 0), ir.Constant(I64, size),
             ir.Constant(I1, False)],
        )

    def _get_memset(self) -> ir.Function:
        """`@llvm.memset.p0i8.i64(i8* dst, i8 val, i64 len, i1 isvolatile)`
        — used by `_emit_slot_clear` to zero a rooted slot at every
        pop site in `llvm` GC mode."""
        existing = self.module.globals.get("llvm.memset.p0i8.i64")
        if isinstance(existing, ir.Function):
            return existing
        fty = ir.FunctionType(
            ir.VoidType(),
            [I8.as_pointer(), I8, I64, I1],
        )
        return ir.Function(self.module, fty, "llvm.memset.p0i8.i64")

    def _finalize_pending_gcroots(self, llvm_fn: ir.Function) -> None:
        """Emit all queued `@llvm.gcroot` calls into the entry block,
        immediately after the alloca prologue. Called at fn body end
        once the full pending set is known. Also marks `llvm_fn` for
        the IR-text post-processor's `gc "shadow-stack"` attribute
        injection if any roots were queued — fns without roots get no
        attribute so the strategy emitter skips per-call StackEntry
        bookkeeping for them."""
        if self._gc_mode != "llvm":
            self._pending_gcroots = []
            return
        if not self._pending_gcroots:
            return
        # Mark the fn for the gc-attribute post-process.
        self._fns_needing_gc_attr.add(llvm_fn.name)
        # Position at the end of the entry block, before its terminator
        # (if any — usually the entry block just falls through to the
        # next, so it has no terminator yet at the point this runs).
        entry = llvm_fn.entry_basic_block
        b = ir.IRBuilder(entry)
        # Position the builder right after the entry block's last
        # alloca. We walk forwards until we hit the first non-alloca
        # instruction — that's where gcroot calls go. (The strategy
        # emitter requires gcroot calls to dominate every use; emitting
        # in the entry block satisfies this.)
        insert_pos = None
        for inst in entry.instructions:
            if not isinstance(inst, ir.AllocaInstr):
                insert_pos = inst
                break
        if insert_pos is None:
            # Entry block is all-allocas (or empty) — append at the end.
            b.position_at_end(entry)
        else:
            b.position_before(insert_pos)
        intrinsic = self._get_llvm_gcroot()
        # Save / restore self.builder so callees in the loop (notably
        # `_emit_slot_clear`'s memset path) emit at the entry-block
        # position, not whatever stale position the body finalize
        # left. We're about to be torn down anyway, but keeping the
        # mutation contained makes the code less fragile.
        saved_builder = self.builder
        self.builder = b
        try:
            for shadow_slot, real_slot, desc, kind, value_ty in self._pending_gcroots:
                # @llvm.gcroot's metadata operand must be a constant —
                # using `b.bitcast(desc, ...)` would emit a runtime
                # cast and trip the verifier. `desc.bitcast(...)`
                # returns a ConstantExpr which the verifier accepts.
                b.call(
                    intrinsic,
                    [
                        shadow_slot,
                        desc.bitcast(I8.as_pointer()),
                    ],
                )
                # Init-clear the real slot at fn entry so the slot's
                # contents read as "empty / no-trace" before any user
                # code runs. Without this, the slot is undef-memory
                # (alloca semantics) and a GC firing before the
                # chokepoint's store would see garbage. The user's
                # actual store later overwrites the zero/sentinel —
                # one wasted store per rooted slot, paid once per
                # fn entry. Cheap.
                self._emit_slot_clear(real_slot, kind, value_ty)
        finally:
            self.builder = saved_builder
        self._pending_gcroots = []

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
