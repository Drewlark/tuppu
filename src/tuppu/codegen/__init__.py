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
from .rat import RatMixin
from .sex import SexMixin
from .strs import StrsMixin
from .tablets import TabletsMixin


class Codegen(SexMixin, RatMixin, TabletsMixin, StrsMixin):
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
        self.printf = ir.Function(
            self.module,
            ir.FunctionType(I32, [i8ptr], var_arg=True),
            name="printf",
        )
        self.scanf = ir.Function(
            self.module,
            ir.FunctionType(I32, [i8ptr], var_arg=True),
            name="scanf",
        )
        self._malloc: ir.Function | None = None  # lazy
        self._free: ir.Function | None = None
        self._write: ir.Function | None = None
        self._fflush: ir.Function | None = None
        self._strlen: ir.Function | None = None
        # Colophon decls by Tuppu-level name, for call-site marshaling.
        self._colophon_decls: dict[str, A.ColophonDecl] = {}

    def _get_malloc(self) -> ir.Function:
        if self._malloc is None:
            self._malloc = ir.Function(
                self.module,
                ir.FunctionType(I8.as_pointer(), [I64]),
                name="malloc",
            )
        return self._malloc

    def _get_free(self) -> ir.Function:
        if self._free is None:
            self._free = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), [I8.as_pointer()]),
                name="free",
            )
        return self._free

    def _get_write(self) -> ir.Function:
        if self._write is None:
            self._write = ir.Function(
                self.module,
                ir.FunctionType(I64, [I32, I8.as_pointer(), I64]),
                name="write",
            )
        return self._write

    def _get_fflush(self) -> ir.Function:
        if self._fflush is None:
            self._fflush = ir.Function(
                self.module,
                ir.FunctionType(I32, [I8.as_pointer()]),
                name="fflush",
            )
        return self._fflush

    def _get_strlen(self) -> ir.Function:
        if self._strlen is None:
            existing = self.module.globals.get("strlen")
            if existing is not None:
                self._strlen = existing
            else:
                self._strlen = ir.Function(
                    self.module,
                    ir.FunctionType(I64, [I8.as_pointer()]),
                    name="strlen",
                )
        return self._strlen

    # --- top level ---

    def gen(self, prog: A.Program) -> ir.Module:
        self.comptime = Comptime(prog)

        # Phase 0: build struct LLVM types. Ordered so a struct referenced by
        # a later struct (or by function signatures) is always ready.
        self._register_structs(
            [d for d in prog.decls if isinstance(d, A.StructDecl)]
        )
        self._register_seals(
            [d for d in prog.decls if isinstance(d, A.SealDecl)]
        )

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
            param_types.append(t)
        ret_type = self._lower_type(fn.return_type) if fn.return_type else ir.VoidType()
        fn_type = ir.FunctionType(ret_type, param_types)
        llvm_fn = ir.Function(self.module, fn_type, name=fn.name)
        for i, p in enumerate(fn.params):
            llvm_fn.args[i].name = p.name
        self.functions[fn.name] = llvm_fn

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
        param_types = [
            self._colophon_c_ty(self._lower_type(p.type)) for p in c.params
        ]
        ret_type = self._colophon_c_ty(
            self._lower_type(c.return_type) if c.return_type
            else ir.VoidType()
        )
        fn_type = ir.FunctionType(ret_type, param_types)
        existing = self.module.globals.get(c_sym)
        if existing is not None:
            llvm_fn = existing
        else:
            llvm_fn = ir.Function(self.module, fn_type, name=c_sym)
            for i, p in enumerate(c.params):
                llvm_fn.args[i].name = p.name
        self.functions[c.name] = llvm_fn
        self._colophon_decls[c.name] = c

    def _colophon_c_ty(self, ty: ir.Type) -> ir.Type:
        """Map a Tuppu-side LLVM type to its C-ABI counterpart for
        extern signatures. `str` becomes `i8*` (pointer to NUL-
        terminated bytes); `bool` widens to `i8` for cross-platform
        stability; integer types pass through unchanged."""
        if isinstance(ty, ir.VoidType):
            return ty
        if self._is_str_value(ty):
            return I8.as_pointer()
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
            v = self._gen_expr(arg_expr)
            if v is None:
                raise CodegenError(
                    f"colophon {decl.name!r} arg has no value"
                )
            param_ty = self._lower_type(param.type)
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
        self._cleanup_frames.append([])
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
                if is_mut_tablets:
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
                self.builder.ret_void()
            else:
                if value is None:
                    raise CodegenError(
                        f"function {fn.name!r} must produce a value for return type "
                        f"{fn.return_type}, but its body has no trailing expression"
                    )
                expected = self._lower_type(fn.return_type)
                coerced = self._coerce(value, expected)
                self._emit_frame_cleanups(self._cleanup_frames[-1])
                self.builder.ret(coerced)
        finally:
            self._cleanup_frames.pop()

    def _is_terminated(self) -> bool:
        assert self.builder is not None
        return self.builder.block.is_terminated

    # --- types ---

    def _lower_type(self, t: A.TypeExpr) -> ir.Type:
        if isinstance(t, A.TypeName):
            if t.name in INT_WIDTH:
                return ir.IntType(INT_WIDTH[t.name])
            if t.name == "bool":
                return I1
            if t.name == "rat":
                return RAT
            # sex/dish now has a distinct digit-form representation so its
            # Babylonian identity survives to runtime. Coercion between sex
            # and rat is a real conversion, not a no-op.
            if t.name in ("sex", "dish"):
                return SEX
            # Generic-body type parameter in scope — resolve to the
            # current specialization's concrete LLVM type.
            if t.name in self._type_arg_subst:
                return self._type_arg_subst[t.name]
            if t.name in self._struct_types:
                return self._struct_types[t.name]
            if t.name in self._seal_types:
                return self._seal_types[t.name]
            raise CodegenError(f"type {t.name!r} not supported in this stage")
        if isinstance(t, A.TypeApply):
            arg_tys = tuple(self._lower_type(a) for a in t.args)
            if t.name in self._generic_seal_decls:
                return self._get_monomorph_seal(t.name, arg_tys)
            return self._get_monomorph_struct(t.name, arg_tys)
        if isinstance(t, A.TypeTablets):
            elem = self._lower_type(t.element)
            return self._get_tablets(t.size, elem).tablets_ty
        if isinstance(t, A.TypePointer):
            elem = self._lower_type(t.element)
            return elem.as_pointer()
        if isinstance(t, A.TypeHandle):
            # `tablet T` — runtime is a pointer to T, distinct from
            # `*T` at the source level but same LLVM representation.
            elem = self._lower_type(t.element)
            return elem.as_pointer()
        raise CodegenError(
            f"complex types not supported in this stage: {type(t).__name__}"
        )

    def _register_structs(self, decls: list[A.StructDecl]) -> None:
        """Build LLVM types for user-defined tablets.

        Two phases enable recursive and mutually-recursive types: first
        we declare every tablet name as an empty identified LLVM type;
        then we resolve field types now that every name is visible, so
        `wedge Node` inside `Node`'s body resolves cleanly.

        Generic tablets (those with type parameters) are NOT emitted
        here — we can't compute a layout without concrete type args.
        Their AST declarations are stashed for on-demand specialization
        via `_get_monomorph_struct(name, concrete_arg_tys)`."""
        self._generic_struct_decls: dict[str, A.StructDecl] = {
            d.name: d for d in decls if d.type_params
        }
        decls = [d for d in decls if not d.type_params]
        by_name = {d.name: d for d in decls}

        # Phase A: declare empty identified types for every struct.
        for d in decls:
            if d.name in self._struct_types:
                raise CodegenError(f"duplicate struct {d.name!r}")
            ident_ty = self.module.context.get_identified_type(d.name)
            self._struct_types[d.name] = ident_ty

        # Phase B: detect direct cycles (cycle in the "inline contains"
        # graph). A field whose type is another struct by value — or an
        # array of that struct — contributes a direct edge. A field
        # that's a pointer or tablets does NOT (the recursion goes
        # through heap indirection, so size is finite).
        direct_deps: dict[str, set[str]] = {}
        for d in decls:
            deps: set[str] = set()
            for _fname, ftype in d.fields:
                if isinstance(ftype, A.TypeName) and ftype.name in by_name:
                    deps.add(ftype.name)
                elif isinstance(ftype, A.TypeArray):
                    elem = ftype.element
                    if isinstance(elem, A.TypeName) and elem.name in by_name:
                        deps.add(elem.name)
            direct_deps[d.name] = deps

        color: dict[str, int] = {name: 0 for name in by_name}  # 0 white, 1 gray, 2 black
        def visit(name: str) -> None:
            if color[name] == 2:
                return
            if color[name] == 1:
                raise CodegenError(
                    f"tablet {name!r} is recursively contained without "
                    f"indirection — use `wedge {name}` for a "
                    f"recursive reference (it gives finite size via "
                    f"a tablets-backed pointer)"
                )
            color[name] = 1
            for dep in direct_deps[name]:
                visit(dep)
            color[name] = 2

        for name in by_name:
            visit(name)

        # Phase C: resolve all bodies. Identified types support
        # `set_body(*field_tys)` exactly once.
        for d in decls:
            field_tys = [self._lower_type(ftype) for _, ftype in d.fields]
            self._struct_types[d.name].set_body(*field_tys)
            self._struct_fields[d.name] = list(
                zip([n for n, _ in d.fields], field_tys)
            )

    def _get_monomorph_struct(
        self, name: str, arg_tys: tuple,
    ) -> ir.IdentifiedStructType:
        """Return (building once, caching thereafter) the specialized
        identified LLVM struct type for a generic tablet at a specific
        type-arg tuple. E.g. `_get_monomorph_struct("Node", (I64,))`
        yields `Node_i64`.

        Field bodies are set by substituting the declaration's type
        parameters with `arg_tys` and lowering through `_lower_type` —
        which means fields of type `wedge Node<T>` correctly produce
        a pointer to this same monomorphized type (thanks to the
        identified type being registered before we compute the body,
        matching the non-generic case's two-phase approach)."""
        if not arg_tys:
            # Delegate to the non-generic path.
            return self._struct_types[name]
        key = (name, arg_tys)
        cached = self._struct_monomorphs.get(key)
        if cached is not None:
            return cached
        decl = self._generic_struct_decls.get(name)
        if decl is None:
            raise CodegenError(
                f"unknown generic tablet {name!r} for monomorphization"
            )
        # Build a stable identified-type name from the args. llvmlite
        # keeps the string as-is so we escape special chars lightly.
        arg_tag = "_".join(str(a).replace(" ", "").replace('"', "")
                           for a in arg_tys)
        mono_name = f"{name}__{arg_tag}"
        ident_ty = self.module.context.get_identified_type(mono_name)
        self._struct_monomorphs[key] = ident_ty
        # Set the subst so any reference to a type param inside the
        # body resolves to the concrete arg. Also register this
        # monomorph under `self._struct_types[decl.name]` temporarily
        # so `wedge Node<T>` resolves back to the same identified type
        # via the name lookup path. We pop the shadowing afterward.
        saved_subst = self._type_arg_subst
        self._type_arg_subst = dict(zip(decl.type_params, arg_tys))
        saved_struct_ty = self._struct_types.get(name)
        self._struct_types[name] = ident_ty
        try:
            field_tys = [self._lower_type(ftype) for _, ftype in decl.fields]
        finally:
            self._type_arg_subst = saved_subst
            if saved_struct_ty is None:
                del self._struct_types[name]
            else:
                self._struct_types[name] = saved_struct_ty
        ident_ty.set_body(*field_tys)
        self._struct_mono_fields[key] = list(
            zip([n for n, _ in decl.fields], field_tys)
        )
        return ident_ty

    def _get_monomorph_fn(
        self, name: str, arg_tys: tuple,
    ) -> ir.Function:
        """Emit (once, caching thereafter) a specialization of a
        generic fn at a concrete type-arg tuple. Walks the fn body
        AST with `_type_arg_subst` set so type-parameter references
        resolve to the concrete LLVM type."""
        key = (name, arg_tys)
        cached = self._fn_monomorphs.get(key)
        if cached is not None:
            return cached
        decl = self._generic_fn_decls.get(name)
        if decl is None:
            raise CodegenError(
                f"unknown generic fn {name!r} for monomorphization"
            )
        saved_subst = self._type_arg_subst
        saved_builder = self.builder
        saved_scopes = self.scopes
        saved_cleanup = self._cleanup_frames
        saved_loc = self._current_loc
        # Give the specialization a fresh scope + cleanup stack so it
        # doesn't inherit state from whichever outer emit we're nested
        # inside. _gen_fn_body will overwrite self.scopes anyway but
        # the cleanup stack needs to start empty here.
        self._cleanup_frames = []
        self._type_arg_subst = dict(zip(decl.type_params, arg_tys))
        # Declare with a tagged name. Fresh function — distinct from
        # the generic AST decl.name which we never emit directly.
        arg_tag = "_".join(str(a).replace(" ", "").replace('"', "")
                           for a in arg_tys)
        mono_name = f"{name}__{arg_tag}"
        param_types = []
        for p in decl.params:
            t = self._lower_type(p.type)
            if p.is_mut and self._tablets_info_for(t) is not None:
                t = t.as_pointer()
            param_types.append(t)
        ret_type = (
            self._lower_type(decl.return_type)
            if decl.return_type else ir.VoidType()
        )
        fn_type = ir.FunctionType(ret_type, param_types)
        llvm_fn = ir.Function(self.module, fn_type, name=mono_name)
        for i, p in enumerate(decl.params):
            llvm_fn.args[i].name = p.name
        self._fn_monomorphs[key] = llvm_fn
        # Temporarily install this specialization under the decl's
        # source name so recursive calls inside the body find it and
        # don't trigger a second monomorphization pass.
        saved_functions = self.functions.get(name)
        self.functions[name] = llvm_fn
        try:
            self._gen_fn_body(decl)
        finally:
            self._type_arg_subst = saved_subst
            self.builder = saved_builder
            self.scopes = saved_scopes
            self._cleanup_frames = saved_cleanup
            self._current_loc = saved_loc
            if saved_functions is None:
                del self.functions[name]
            else:
                self.functions[name] = saved_functions
        return llvm_fn

    # --- seals (sum types) ---------------------------------------------

    def _register_seals(self, decls: list["A.SealDecl"]) -> None:
        """Phase 0c for sum types. Declares an identified LLVM type per
        seal and computes its layout `{ i8 tag, [N x i64] payload }` so
        variants can be constructed / matched by bitcasting the payload
        slot to the appropriate variant struct.

        Generic seals (those with type parameters) are stashed for on-
        demand monomorphization — layout depends on concrete arg types
        (payload width differs between `Option<i64>` and `Option<str>`),
        so we can't emit one up front."""
        self._generic_seal_decls = {
            d.name: d for d in decls if d.type_params
        }
        concrete = [d for d in decls if not d.type_params]
        # Phase A: declare empty identified types so mutually-recursive
        # seal-to-seal references resolve during body lowering.
        for d in concrete:
            if d.name in self._seal_types:
                raise CodegenError(f"duplicate seal {d.name!r}")
            ident_ty = self.module.context.get_identified_type(d.name)
            self._seal_types[d.name] = ident_ty
        # Phase B: compute each seal's payload layout.
        for d in concrete:
            self._finalize_seal(d.name, d, arg_tys=())

    def _finalize_seal(
        self,
        name: str,
        decl: "A.SealDecl",
        arg_tys: tuple,
    ) -> None:
        """Compute and assign the LLVM body for a (possibly monomorphized)
        seal. Stores variant payload struct types under the seal's key
        for later variant construction / match destructuring."""
        # Set the subst so type-parameter references inside variant
        # fields resolve to concrete arg types.
        saved_subst = self._type_arg_subst
        if decl.type_params:
            self._type_arg_subst = dict(zip(decl.type_params, arg_tys))
        try:
            variants: list[tuple[str, ir.LiteralStructType]] = []
            for v in decl.variants:
                field_tys = [self._lower_type(ft) for ft in v.fields]
                payload = ir.LiteralStructType(field_tys)
                variants.append((v.name, payload))
        finally:
            self._type_arg_subst = saved_subst
        key = name if not arg_tys else (name, arg_tys)
        self._seal_variants[key] = variants
        max_bytes = 0
        for _, payload in variants:
            b = self._size_of(payload)
            if b > max_bytes:
                max_bytes = b
        # Payload chunk: array of i64, sized up to fit the widest variant.
        # Using i64 gives 8-byte alignment, which covers every primitive
        # we currently allow in variants.
        n = (max_bytes + 7) // 8
        if n == 0:
            body = [I8]
        else:
            body = [I8, ir.ArrayType(I64, n)]
        seal_ty = self._seal_types[key] if not arg_tys else self._seal_types[key]
        seal_ty.set_body(*body)

    def _get_monomorph_seal(
        self, name: str, arg_tys: tuple,
    ) -> ir.IdentifiedStructType:
        """Return the specialized seal LLVM type for a generic seal at
        concrete type args. Mirrors `_get_monomorph_struct` in shape."""
        if not arg_tys:
            return self._seal_types[name]
        key = (name, arg_tys)
        cached = self._seal_types.get(key)
        if cached is not None:
            return cached
        decl = self._generic_seal_decls.get(name)
        if decl is None:
            raise CodegenError(
                f"unknown generic seal {name!r} for monomorphization"
            )
        arg_tag = "_".join(str(a).replace(" ", "").replace('"', "")
                           for a in arg_tys)
        mono_name = f"{name}__{arg_tag}"
        ident_ty = self.module.context.get_identified_type(mono_name)
        self._seal_types[key] = ident_ty
        self._finalize_seal(name, decl, arg_tys)
        return ident_ty

    def _seal_key_for_ty(self, llvm_ty: ir.Type):
        """Find the seal key (name or (name, args) tuple) for a given
        LLVM type. Returns None if not a registered seal."""
        for k, v in self._seal_types.items():
            if v is llvm_ty:
                return k
        return None

    def _size_of(self, ty: ir.Type) -> int:
        """Conservative-to-LLVM byte size. Handles primitives, pointers,
        arrays, and (possibly identified) struct types including rat/
        sex/user-tablets/nested seals. Good enough for picking the
        widest variant's payload width — we round up to i64 anyway."""
        if isinstance(ty, ir.IntType):
            return max(1, ty.width // 8)
        if isinstance(ty, ir.PointerType):
            return 8
        if isinstance(ty, ir.ArrayType):
            return ty.count * self._size_of(ty.element)
        if isinstance(ty, (ir.LiteralStructType, ir.IdentifiedStructType)):
            offset = 0
            max_align = 1
            elements = getattr(ty, "elements", None) or ()
            for m in elements:
                align = self._align_of(m)
                if align > max_align:
                    max_align = align
                offset = (offset + align - 1) // align * align
                offset += self._size_of(m)
            return (offset + max_align - 1) // max_align * max_align
        raise CodegenError(f"cannot compute size of {ty}")

    def _align_of(self, ty: ir.Type) -> int:
        if isinstance(ty, ir.IntType):
            return max(1, ty.width // 8)
        if isinstance(ty, ir.PointerType):
            return 8
        if isinstance(ty, ir.ArrayType):
            return self._align_of(ty.element)
        if isinstance(ty, (ir.LiteralStructType, ir.IdentifiedStructType)):
            elements = getattr(ty, "elements", None) or ()
            return max((self._align_of(m) for m in elements), default=1)
        return 1

    def _variant_lookup(
        self, seal_key, variant_name: str,
    ) -> tuple[int, ir.LiteralStructType]:
        """Return (tag_index, payload_struct_ty) for a variant within a
        seal. Raises CodegenError if the variant isn't registered."""
        variants = self._seal_variants.get(seal_key)
        if variants is None:
            raise CodegenError(f"seal {seal_key!r} not registered")
        for idx, (vn, payload) in enumerate(variants):
            if vn == variant_name:
                return idx, payload
        raise CodegenError(
            f"seal {seal_key!r} has no variant {variant_name!r}"
        )

    def _gen_variant_ctor(self, node) -> ir.Value:
        """Build a seal value from a variant construction. `node` is
        either a bare Ident (nullary variant) or a Call whose callee
        is an Ident naming a variant.

        Layout: alloca the seal struct, zero-init, write the tag byte,
        and for variants with fields bitcast the payload slot to the
        per-variant payload struct and store each field. Load and
        return the final aggregate value."""
        assert self.builder is not None
        assert self._checker is not None
        if isinstance(node, A.Call):
            variant_name = node.callee.name  # type: ignore[attr-defined]
            args = node.args
        else:
            variant_name = node.name
            args = []
        seal_name, _, vidx = self._checker.variant_of_node[id(node)]
        type_args = self._checker.mono_variant_args[id(node)]
        if type_args:
            llvm_args = tuple(self._lower_ty(a) for a in type_args)
            seal_ty = self._get_monomorph_seal(seal_name, llvm_args)
            seal_key = (seal_name, llvm_args)
        else:
            seal_ty = self._seal_types[seal_name]
            seal_key = seal_name
        _, payload_ty = self._variant_lookup(seal_key, variant_name)

        slot = self._alloca_entry(seal_ty, f"{seal_name}.{variant_name}")
        # Zero the whole thing so any unused payload bytes are well-defined
        # (lets `== lost`-style tag comparisons work deterministically).
        self.builder.store(ir.Constant(seal_ty, None), slot)
        tag_ptr = self.builder.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, 0)], inbounds=True,
        )
        self.builder.store(ir.Constant(I8, vidx), tag_ptr)

        if payload_ty.elements and args:
            payload_ptr = self._seal_payload_ptr(slot)
            typed_ptr = self.builder.bitcast(
                payload_ptr, payload_ty.as_pointer(),
            )
            for i, (arg, expected_ty) in enumerate(
                zip(args, payload_ty.elements)
            ):
                v = self._gen_expr(arg)
                if v is None:
                    raise CodegenError(
                        f"variant {variant_name!r} arg {i} has no value"
                    )
                field_ptr = self.builder.gep(
                    typed_ptr,
                    [ir.Constant(I32, 0), ir.Constant(I32, i)],
                    inbounds=True,
                )
                self.builder.store(self._coerce(v, expected_ty), field_ptr)
        return self.builder.load(slot)

    def _seal_payload_ptr(self, slot: ir.Value) -> ir.Value:
        """GEP to the payload slot (field index 1) of a seal alloca.
        Asserts that the seal actually has a payload field."""
        assert self.builder is not None
        seal_ty = slot.type.pointee
        if not isinstance(seal_ty, ir.IdentifiedStructType):
            raise CodegenError(f"expected seal pointer, got {slot.type}")
        if len(seal_ty.elements) < 2:
            raise CodegenError(
                "seal has no payload (all variants are nullary) "
                "— caller should have taken the fast path"
            )
        return self.builder.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, 1)], inbounds=True,
        )

    def _gen_match(self, e: "A.MatchExpr") -> ir.Value | None:
        """Lower a match expression to a switch on the tag byte.

        Each arm becomes its own basic block; a VariantPattern's arm
        bitcasts the payload to the variant's payload struct and binds
        pattern binders to extracted fields. A wildcard arm becomes
        the default block; without one the default is `unreachable`
        (exhaustiveness is already checked by the type checker).

        Arm values are joined via a phi in the merge block, mirroring
        the `if`-expression pattern."""
        assert self.builder is not None
        assert self._checker is not None
        scrutinee = self._gen_expr(e.scrutinee)
        if scrutinee is None:
            raise CodegenError("match scrutinee diverged")
        seal_key = self._seal_key_for_ty(scrutinee.type)
        if seal_key is None:
            raise CodegenError(
                f"match scrutinee has type {scrutinee.type}, not a seal"
            )
        variants = self._seal_variants[seal_key]
        name_to_index = {vn: i for i, (vn, _) in enumerate(variants)}
        # Spill scrutinee so we can GEP into it for the payload.
        slot = self._alloca_entry(scrutinee.type, "match.scrut")
        self.builder.store(scrutinee, slot)
        tag_ptr = self.builder.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, 0)], inbounds=True,
        )
        tag = self.builder.load(tag_ptr)

        fn = self.builder.function
        merge_bb = fn.append_basic_block("match.merge")
        default_bb = fn.append_basic_block("match.default")

        # Pick the default block: if an arm has a wildcard, its body
        # becomes the default. Otherwise default is an unreachable trap.
        wildcard_arm: "A.MatchArm | None" = None
        variant_arms: list[tuple["A.MatchArm", int, ir.Block]] = []
        for arm in e.arms:
            if isinstance(arm.pattern, A.WildcardPattern):
                wildcard_arm = arm
            else:
                vidx = name_to_index[arm.pattern.name]
                bb = fn.append_basic_block(f"match.{arm.pattern.name}")
                variant_arms.append((arm, vidx, bb))

        switch_inst = self.builder.switch(tag, default_bb)
        for _arm, vidx, bb in variant_arms:
            switch_inst.add_case(ir.Constant(I8, vidx), bb)

        results: list[tuple[ir.Value | None, ir.Block]] = []

        # Emit variant arms.
        for arm, vidx, bb in variant_arms:
            self.builder.position_at_end(bb)
            self.scopes.append({})
            self._cleanup_frames.append([])
            try:
                self._bind_variant_pattern(arm.pattern, slot, seal_key)
                val = self._gen_expr(arm.body)
                end_bb = self.builder.block
                diverged = end_bb.is_terminated
                if not diverged:
                    self._emit_frame_cleanups(self._cleanup_frames[-1])
                    results.append((val, self.builder.block))
                    self.builder.branch(merge_bb)
            finally:
                self.scopes.pop()
                self._cleanup_frames.pop()

        # Emit default (wildcard or unreachable trap).
        self.builder.position_at_end(default_bb)
        if wildcard_arm is not None:
            self.scopes.append({})
            self._cleanup_frames.append([])
            try:
                val = self._gen_expr(wildcard_arm.body)
                end_bb = self.builder.block
                diverged = end_bb.is_terminated
                if not diverged:
                    self._emit_frame_cleanups(self._cleanup_frames[-1])
                    results.append((val, self.builder.block))
                    self.builder.branch(merge_bb)
            finally:
                self.scopes.pop()
                self._cleanup_frames.pop()
        else:
            self.builder.unreachable()

        self.builder.position_at_end(merge_bb)
        if not results:
            # All arms diverged.
            self.builder.unreachable()
            return None
        if all(r[0] is None for r in results):
            return None
        # Find a non-None representative to pick the phi type.
        rep = next((r for r in results if r[0] is not None), None)
        if rep is None:
            return None
        phi = self.builder.phi(rep[0].type)
        for val, bb in results:
            if val is None:
                # Diverged arms don't contribute to the phi; but arms
                # that produced unit in a unit-typed match shouldn't
                # reach here in practice.
                continue
            phi.add_incoming(val, bb)
        return phi

    def _bind_variant_pattern(
        self,
        pattern: "A.VariantPattern",
        scrut_slot: ir.Value,
        seal_key,
    ) -> None:
        """Inside a match arm for a variant pattern: bitcast the
        scrutinee's payload to the variant's payload struct and bind
        each named pattern binder to a loaded field."""
        assert self.builder is not None
        vidx, payload_ty = self._variant_lookup(seal_key, pattern.name)
        if not payload_ty.elements:
            return
        payload_ptr = self._seal_payload_ptr(scrut_slot)
        typed_ptr = self.builder.bitcast(payload_ptr, payload_ty.as_pointer())
        for i, binder in enumerate(pattern.binders):
            if binder is None:
                continue
            field_ptr = self.builder.gep(
                typed_ptr,
                [ir.Constant(I32, 0), ir.Constant(I32, i)],
                inbounds=True,
            )
            val = self.builder.load(field_ptr)
            self._bind(binder, Variable(
                is_mut=False, ir_ref=val, value_ty=val.type,
            ))

    def _lower_ty(self, ty) -> ir.Type:
        """Convert a `typecheck.Ty` object (the resolved-type form the
        checker works in) to an `ir.Type`. Used by monomorphization
        paths where we have checker-resolved types, not AST nodes."""
        from ..typecheck import (
            TyInt, TyBool, TyRat, TyDish, TyUnit, TyHandle, TyTablets,
            TyStruct, TySeal, TyVar,
        )
        if isinstance(ty, TyVar):
            # Inside a generic fn specialization, a TyVar that survived
            # typechecking refers to one of this specialization's type
            # parameters. Look it up via the current subst.
            if ty.name in self._type_arg_subst:
                return self._type_arg_subst[ty.name]
            raise CodegenError(
                f"unbound type variable {ty.name!r} during codegen"
            )
        if isinstance(ty, TyInt):
            return ir.IntType(ty.width)
        if isinstance(ty, TyBool):
            return I1
        if isinstance(ty, TyRat):
            return RAT
        if isinstance(ty, TyDish):
            return SEX
        if isinstance(ty, TyUnit):
            return ir.VoidType()
        if isinstance(ty, TyHandle):
            return self._lower_ty(ty.element).as_pointer()
        if isinstance(ty, TyTablets):
            return self._get_tablets(ty.size, self._lower_ty(ty.element)).tablets_ty
        if isinstance(ty, TyStruct):
            if ty.args:
                arg_tys = tuple(self._lower_ty(a) for a in ty.args)
                return self._get_monomorph_struct(ty.name, arg_tys)
            return self._struct_types[ty.name]
        if isinstance(ty, TySeal):
            if ty.args:
                arg_tys = tuple(self._lower_ty(a) for a in ty.args)
                return self._get_monomorph_seal(ty.name, arg_tys)
            return self._seal_types[ty.name]
        raise CodegenError(f"cannot lower checker type {ty!r} to LLVM")

    def _struct_name_for(self, llvm_ty: ir.Type) -> str | None:
        for name, ty in self._struct_types.items():
            if ty is llvm_ty:
                return name
        for (name, _args), ty in self._struct_monomorphs.items():
            if ty is llvm_ty:
                return name
        return None

    def _struct_fields_for(self, llvm_ty: ir.Type) -> list[tuple[str, ir.Type]] | None:
        """Look up field list by LLVM type (for either non-generic or
        monomorphized tablets). Returns None if not a known tablet."""
        for name, ty in self._struct_types.items():
            if ty is llvm_ty:
                return self._struct_fields[name]
        for key, ty in self._struct_monomorphs.items():
            if ty is llvm_ty:
                return self._struct_mono_fields[key]
        return None

    def _coerce(self, value: ir.Value, target_ty: ir.Type) -> ir.Value:
        """Insert a cast instruction if value's type differs from target_ty.
        Handles integer widening (sext/zext), integer narrowing (trunc),
        and i64<->rat and sex<->rat conversions."""
        if value.type == target_ty:
            return value
        assert self.builder is not None

        # Sex conversions. Sex is a compile-time-distinct type now; going
        # to rat requires a runtime reduction of the digit sequence.
        if value.type == SEX:
            if target_ty == RAT:
                return self.builder.call(self._get_sex_to_rat(), [value])
            if isinstance(target_ty, ir.IntType):
                # sex → iN: reduce to rat, then truncate.
                as_rat = self._coerce(value, RAT)
                return self._coerce(as_rat, target_ty)
        if target_ty == SEX:
            # int → sex: decompose into base-60 digits via a runtime helper.
            # Always lands in integer form (no fractional digits).
            if isinstance(value.type, ir.IntType):
                n_i64 = self._coerce(value, I64)
                return self.builder.call(self._get_int_to_sex(), [n_i64])
            # rat → sex: regularity-checked reconstruction. Traps at
            # runtime if the denominator isn't 2^a·3^b·5^c (non-
            # terminating sexagesimal), or if it would need more than
            # SEX_MAX_DIGITS fractional digits.
            if value.type == RAT:
                return self.builder.call(self._get_rat_to_sex(), [value])

        # Rat conversions.
        if value.type == RAT and isinstance(target_ty, ir.IntType):
            # rat as iN: truncate toward zero via signed division of num/den.
            num = self.builder.extract_value(value, 0)
            den = self.builder.extract_value(value, 1)
            result = self.builder.sdiv(num, den)
            return self._coerce(result, target_ty)  # narrow/widen to target width
        if target_ty == RAT and isinstance(value.type, ir.IntType):
            # iN as rat: widen to i64, then build {num: x, den: 1} (already reduced).
            num_i64 = self._coerce(value, I64)
            undef = ir.Constant(RAT, ir.Undefined)
            with_num = self.builder.insert_value(undef, num_i64, 0)
            return self.builder.insert_value(with_num, ir.Constant(I64, 1), 1)

        if isinstance(value.type, ir.IntType) and isinstance(target_ty, ir.IntType):
            sw, tw = value.type.width, target_ty.width
            if tw > sw:
                # Widening: zero-extend booleans, sign-extend other integers.
                if sw == 1:
                    return self.builder.zext(value, target_ty)
                return self.builder.sext(value, target_ty)
            if tw < sw:
                return self.builder.trunc(value, target_ty)
            return value
        # Pointer-to-pointer bitcasts handle `lost` → any `tablet T`
        # and any handle-handle coercion at the LLVM level. Typecheck
        # has already verified the source was `lost` or a compatible
        # handle before we land here.
        if (
            isinstance(value.type, ir.PointerType)
            and isinstance(target_ty, ir.PointerType)
        ):
            return self.builder.bitcast(value, target_ty)
        line, col = self._current_loc
        raise CodegenError(
            f"cannot coerce {value.type} to {target_ty}", line, col,
        )

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

    # --- statements ---

    def _gen_stmt(self, s: A.Stmt) -> None:
        self._current_loc = (getattr(s, "line", 0), getattr(s, "col", 0))
        if isinstance(s, A.Binding):
            self._gen_binding(s); return
        if isinstance(s, A.Assign):
            self._gen_assign(s); return
        if isinstance(s, A.While):
            self._gen_while(s); return
        if isinstance(s, A.ForStmt):
            self._gen_for(s); return
        if isinstance(s, A.YieldStmt):
            self._gen_yield(s); return
        if isinstance(s, A.ReleaseStmt):
            self._gen_release(s); return
        if isinstance(s, A.ExprStmt):
            val = self._gen_expr(s.expr)
            if val is not None:
                self._register_str_rvalue_cleanup(val, s.expr)
            return
        raise CodegenError(f"statement not supported yet: {type(s).__name__}")

    def _gen_release(self, s: A.ReleaseStmt) -> None:
        var = self._lookup(s.name)
        info = self._tablets_info_for(var.value_ty)
        if info is None:
            raise CodegenError(f"release requires a tablets, got {var.value_ty}")
        if not var.is_mut:
            raise CodegenError(f"cannot release step-bound tablets {s.name!r}")
        assert self.builder is not None
        self.builder.call(info.release, [var.ir_ref])
        # Remove this variable from its cleanup frame so the auto-
        # release at scope exit doesn't double-free. We walk frames
        # outermost-in since explicit release can target an outer
        # binding shadowed by an inner one (unusual but legal).
        for frame in reversed(self._cleanup_frames):
            for i, (_fn, _ptr, name) in enumerate(frame):
                if name == s.name:
                    frame.pop(i)
                    return

    def _gen_for(self, f: A.ForStmt) -> None:
        """Generate a `for name in iter { body }` loop.

        Three iterable shapes are supported; each picks a different loop
        body:

        - **str**: walk 0..len, load s.ptr[i] as u8.
        - **tablets[N]T**: walk the chain via the cached `get` helper.
        - **table**: walk the global array in memory order.

        The loop variable is bound as a fresh `step` (SSA) per iteration
        so it cannot be assigned inside the body."""
        assert self.builder is not None

        # Comptime table iteration — recognise the table by name before
        # we try to produce a value for the iter expression.
        if isinstance(f.iter, A.Ident) and f.iter.name in self._tables:
            self._gen_for_table(f, f.iter.name)
            return

        iter_val = self._gen_expr(f.iter)
        if iter_val is None:
            raise CodegenError("for: iter expression has no value")

        if self._is_str_value(iter_val.type):
            self._gen_for_str(f, iter_val)
            return

        info = self._tablets_info_for(iter_val.type)
        if info is not None:
            self._gen_for_tablets(f, iter_val, info)
            return

        raise CodegenError(
            f"for: cannot iterate over value of type {iter_val.type}"
        )

    def _gen_for_str(self, f: A.ForStmt, str_val: ir.Value) -> None:
        """Lower `for c in s { body }` — c is u8, bounds-safe by construction
        since we walk 0..len."""
        assert self.builder is not None
        ptr = self.builder.extract_value(str_val, 0)
        length = self.builder.extract_value(str_val, 1)
        self._emit_counted_loop(
            length,
            lambda i: self.builder.load(
                self.builder.gep(ptr, [i], inbounds=True),
            ),
            f,
        )

    def _gen_for_tablets(
        self, f: A.ForStmt, tbl_val: ir.Value, info: "TabletsInfo",
    ) -> None:
        """Iterate over a tablets value. We reuse the cached `get` helper;
        mem2reg + the existing optimizer clean up the redundant chain walks
        for dense access patterns."""
        assert self.builder is not None
        # tbl_val is a value (loaded struct). We need an address to pass
        # to the get helper, so spill it to a temp alloca.
        slot = self._alloca_entry(info.tablets_ty, "for.tbl")
        self.builder.store(tbl_val, slot)
        len_addr = self.builder.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, 2)], inbounds=True,
        )
        length = self.builder.load(len_addr)
        self._emit_counted_loop(
            length,
            lambda i: self.builder.call(info.get, [slot, i]),
            f,
        )

    def _gen_for_table(self, f: A.ForStmt, name: str) -> None:
        """Walk a compile-time table in declaration order (element index
        0..size-1), regardless of the table's `lo` bound."""
        assert self.builder is not None
        g, size, _lo, _elem_ty = self._tables[name]
        length = ir.Constant(I64, size)
        zero = ir.Constant(I32, 0)

        def load_at(i: ir.Value) -> ir.Value:
            return self.builder.load(
                self.builder.gep(g, [zero, i], inbounds=True),
            )

        self._emit_counted_loop(length, load_at, f)

    def _emit_counted_loop(
        self,
        length: ir.Value,
        load_element,
        f: A.ForStmt,
    ) -> None:
        """Emit the shared 0..length loop skeleton and bind `f.name` to the
        current element inside the body. `load_element(i_i64)` returns the
        value to bind."""
        assert self.builder is not None
        fn = self.builder.function
        header = fn.append_basic_block("for.header")
        body = fn.append_basic_block("for.body")
        exit_ = fn.append_basic_block("for.exit")

        i_slot = self._alloca_entry(I64, "for.i")
        self.builder.store(ir.Constant(I64, 0), i_slot)
        self.builder.branch(header)

        self.builder.position_at_end(header)
        i_val = self.builder.load(i_slot)
        cond = self.builder.icmp_signed("<", i_val, length)
        self.builder.cbranch(cond, body, exit_)

        self.builder.position_at_end(body)
        element = load_element(i_val)
        self.scopes.append({})
        try:
            self._bind(f.name, Variable(
                is_mut=False, ir_ref=element, value_ty=element.type,
            ))
            self._gen_block(f.body)
        finally:
            self.scopes.pop()
        if not self._is_terminated():
            next_i = self.builder.add(i_val, ir.Constant(I64, 1))
            self.builder.store(next_i, i_slot)
            self.builder.branch(header)

        self.builder.position_at_end(exit_)

    def _gen_while(self, w: A.While) -> None:
        assert self.builder is not None
        fn = self.builder.function
        header = fn.append_basic_block("while.header")
        body = fn.append_basic_block("while.body")
        exit_ = fn.append_basic_block("while.exit")

        self.builder.branch(header)

        self.builder.position_at_end(header)
        cond = self._gen_expr(w.cond)
        if cond is None or cond.type != I1:
            raise CodegenError("while condition must be a bool expression")
        self.builder.cbranch(cond, body, exit_)

        self.builder.position_at_end(body)
        self._gen_block(w.body)
        if not self._is_terminated():
            self.builder.branch(header)

        self.builder.position_at_end(exit_)

    def _gen_yield(self, y: A.YieldStmt) -> None:
        assert self.builder is not None
        ret_ty = self.builder.function.ftype.return_type
        if y.value is None:
            self._emit_all_cleanups_for_early_return()
            if isinstance(ret_ty, ir.VoidType):
                self.builder.ret_void()
            else:
                raise CodegenError("bare yield in non-void function")
            return
        val = self._gen_expr(y.value)
        if val is None:
            raise CodegenError("yield value diverged")
        coerced = self._coerce(val, ret_ty)
        # Unwind every live cleanup frame (inner-to-outer) before the
        # ret. The return value has already been captured into `coerced`
        # so it doesn't matter if the cleanup invalidates heap memory
        # — escape analysis rejects programs that return handles into
        # soon-released tablets.
        self._emit_all_cleanups_for_early_return()
        self.builder.ret(coerced)

    def _emit_all_cleanups_for_early_return(self) -> None:
        """Emit release calls for every live cleanup frame in the
        current function, innermost first. Used by yield to unwind
        before the ret."""
        for frame in reversed(self._cleanup_frames):
            self._emit_frame_cleanups(frame)

    def _gen_binding(self, b: A.Binding) -> None:
        # Uninitialized mut binding with explicit type: zero-initialize.
        if b.init is None:
            assert b.is_mut and b.type_ann is not None  # parser enforces this
            ty = self._lower_type(b.type_ann)
            slot = self._alloca_entry(ty, b.name)
            assert self.builder is not None
            self.builder.store(ir.Constant(ty, None), slot)
            self._bind(b.name, Variable(is_mut=True, ir_ref=slot, value_ty=ty))
            self._maybe_register_cleanup(b.name, ty, slot)
            return

        init_val = self._gen_expr(b.init)
        if init_val is None:
            raise CodegenError(f"binding {b.name!r} has no value (initializer diverged)")
        if b.type_ann is not None:
            expected = self._lower_type(b.type_ann)
            init_val = self._coerce(init_val, expected)
        if b.is_mut:
            slot = self._alloca_entry(init_val.type, b.name)
            assert self.builder is not None
            self.builder.store(init_val, slot)
            self._bind(b.name, Variable(is_mut=True, ir_ref=slot, value_ty=init_val.type))
            self._maybe_register_cleanup(b.name, init_val.type, slot)
        else:
            # Step-bound cleanup-bearing values need a slot so the
            # scope-exit release can see them. Covers the built-in str
            # and any user struct that transitively holds cleanup-
            # bearing fields. The SSA value stays the read path (reads
            # remain direct, reassignment impossible), the slot exists
            # purely for release dispatch.
            #
            # `step x = y` (Ident-init) is a BORROW: x shares y's
            # heap bytes, y already owns, registering x would
            # double-free at scope exit. Skip cleanup; record
            # `transfer_on_tail` so if x flows out as a block-tail
            # expression, we transfer ownership of the underlying
            # owner instead of x itself. Field-init (`step x = r.name`)
            # follows the same reasoning — the enclosing struct owns,
            # x is a borrow — but there's no single Variable to
            # transfer from; ownership stays with the struct.
            needs_cleanup = (
                self._is_str_value(init_val.type)
                or (
                    self._struct_fields_for(init_val.type) is not None
                    and self._struct_needs_cleanup(init_val.type)
                )
            )
            transfer_on_tail = None
            is_borrow_init = isinstance(b.init, (A.Ident, A.Field, A.StringLit))
            if needs_cleanup and not is_borrow_init:
                assert self.builder is not None
                cleanup_slot = self._alloca_entry(init_val.type, f"{b.name}.cleanup")
                self.builder.store(init_val, cleanup_slot)
                self._maybe_register_cleanup(b.name, init_val.type, cleanup_slot)
            elif needs_cleanup and isinstance(b.init, A.Ident):
                # Redirect tail-transfer to the source. If the source is
                # itself a borrow, chain through; if it's a param or
                # untracked binding, transfer_on_tail stays None and the
                # borrowed value leaves as-is (safe when the true owner
                # lives in an outer scope).
                try:
                    src_var = self._lookup(b.init.name)
                except CodegenError:
                    src_var = None
                if src_var is not None:
                    if src_var.transfer_on_tail is not None:
                        transfer_on_tail = src_var.transfer_on_tail
                    elif self._frame_has_entry(b.init.name):
                        transfer_on_tail = b.init.name
            self._bind(b.name, Variable(
                is_mut=False, ir_ref=init_val, value_ty=init_val.type,
                transfer_on_tail=transfer_on_tail,
            ))

    def _maybe_register_cleanup(
        self, name: str, value_ty: ir.Type, slot: ir.Value,
    ) -> None:
        """If `value_ty` is a cleanup-having type, record a release call
        for the innermost cleanup frame so it fires automatically at
        scope exit. Handled: tablets, the built-in str, and user
        structs that (transitively) hold any of those."""
        if not self._cleanup_frames:
            return
        info = self._tablets_info_for(value_ty)
        if info is not None:
            self._cleanup_frames[-1].append((info.release, slot, name))
            return
        if self._is_str_value(value_ty):
            self._cleanup_frames[-1].append(
                (self._get_str_release(), slot, name),
            )
            return
        if (
            self._struct_fields_for(value_ty) is not None
            and self._struct_needs_cleanup(value_ty)
        ):
            self._cleanup_frames[-1].append(
                (self._get_struct_release(value_ty), slot, name),
            )

    def _struct_needs_cleanup(self, struct_ty: ir.Type) -> bool:
        """Does this user struct (transitively) hold any cleanup-bearing
        fields? Walks the declared field list — str, tablets, and nested
        user structs that themselves need cleanup all count. Pointer /
        handle fields don't — they borrow into some other storage whose
        owner does the release."""
        fields = self._struct_fields_for(struct_ty)
        if fields is None:
            return False
        for _name, fty in fields:
            if self._is_str_value(fty):
                return True
            if self._tablets_info_for(fty) is not None:
                return True
            if self._struct_fields_for(fty) is not None:
                if self._struct_needs_cleanup(fty):
                    return True
        return False

    def _struct_as_borrow(
        self, val: ir.Value, struct_ty: ir.Type,
    ) -> ir.Value:
        """Produce a view of `val` with every cleanup-bearing field
        neutered: str fields get cap=0, tablets fields get zero-init
        {head=null, tail=null, len=0}, nested cleanup structs recurse.
        Used at call sites for struct-valued args so the callee's
        cleanup-frame release becomes a no-op on every owning field —
        caller retains sole ownership of the heap bytes."""
        assert self.builder is not None
        fields = self._struct_fields_for(struct_ty)
        if fields is None:
            return val
        b = self.builder
        result = val
        for i, (_fname, fty) in enumerate(fields):
            if self._is_str_value(fty):
                old = b.extract_value(result, i)
                borrowed = self._str_as_borrow(old)
                result = b.insert_value(result, borrowed, i)
                continue
            if self._tablets_info_for(fty) is not None:
                result = b.insert_value(result, ir.Constant(fty, None), i)
                continue
            if (
                self._struct_fields_for(fty) is not None
                and self._struct_needs_cleanup(fty)
            ):
                old = b.extract_value(result, i)
                borrowed = self._struct_as_borrow(old, fty)
                result = b.insert_value(result, borrowed, i)
        return result

    def _register_struct_rvalue_cleanup(
        self, val: ir.Value, src: A.Expr, struct_ty: ir.Type,
    ) -> None:
        """Anonymous cleanup for a cleanup-bearing struct rvalue the
        caller doesn't bind — e.g. `take(build_row())`. Skips Idents
        and Fields (already owned by someone tracked); struct literals,
        calls, etc. genuinely produce fresh owners that this scope
        now holds."""
        if isinstance(src, (A.Ident, A.Field)):
            return
        if not self._cleanup_frames:
            return
        assert self.builder is not None
        slot = self._alloca_entry(val.type, ".struct.temp")
        self.builder.store(val, slot)
        self._cleanup_frames[-1].append(
            (self._get_struct_release(struct_ty), slot, ".struct.temp"),
        )

    def _get_struct_release(self, struct_ty: ir.Type) -> ir.Function:
        """Build (once, caching by LLVM-type identity) a release fn for
        a user struct: `__tuppu_struct_<name>_release(s: *struct_ty)`.
        GEPs to each cleanup-bearing field and dispatches to the
        appropriate release — str, tablets, or nested struct. Fields
        without cleanup are skipped entirely."""
        cached = self._struct_release_cache.get(id(struct_ty))
        if cached is not None:
            return cached
        name = self._struct_name_for(struct_ty) or "anon"
        fn = ir.Function(
            self.module,
            ir.FunctionType(ir.VoidType(), [struct_ty.as_pointer()]),
            name=f"__tuppu_struct_{name}_release",
        )
        # Cache before body-build so any recursive call through a nested
        # struct field (via another _get_struct_release) sees the in-
        # progress function rather than rebuilding it.
        self._struct_release_cache[id(struct_ty)] = fn

        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        s_ptr = fn.args[0]
        fields = self._struct_fields_for(struct_ty) or []
        for i, (_fname, fty) in enumerate(fields):
            if self._is_str_value(fty):
                field_ptr = b.gep(
                    s_ptr, [ir.Constant(I32, 0), ir.Constant(I32, i)],
                    inbounds=True,
                )
                b.call(self._get_str_release(), [field_ptr])
                continue
            info = self._tablets_info_for(fty)
            if info is not None:
                field_ptr = b.gep(
                    s_ptr, [ir.Constant(I32, 0), ir.Constant(I32, i)],
                    inbounds=True,
                )
                b.call(info.release, [field_ptr])
                continue
            if (
                self._struct_fields_for(fty) is not None
                and self._struct_needs_cleanup(fty)
            ):
                field_ptr = b.gep(
                    s_ptr, [ir.Constant(I32, 0), ir.Constant(I32, i)],
                    inbounds=True,
                )
                b.call(self._get_struct_release(fty), [field_ptr])
        b.ret_void()
        return fn

    def _register_str_rvalue_cleanup(
        self, val: ir.Value, src: A.Expr,
    ) -> None:
        """Anonymous-temp auto-release for a str rvalue the caller doesn't
        bind. The heap bytes in `val` need an owner somewhere; if the
        source expression can produce a freshly-owned str (a Call) we
        spill it to a local slot and register release at current-scope
        exit. Idents and Fields are intentionally skipped — they read a
        value someone else already owns. String literals carry cap=0,
        so there's nothing to free."""
        if not self._is_str_value(val.type):
            return
        if isinstance(src, (A.Ident, A.Field, A.StringLit)):
            return
        if not self._cleanup_frames:
            return
        assert self.builder is not None
        slot = self._alloca_entry(val.type, ".str.temp")
        self.builder.store(val, slot)
        self._cleanup_frames[-1].append(
            (self._get_str_release(), slot, ".str.temp"),
        )

    def _gen_assign(self, a: A.Assign) -> None:
        assert self.builder is not None
        # Resolve the target to (slot_ptr, value_type). For an Ident target
        # the slot is the alloca itself. For a Field chain, we GEP from
        # the root alloca down to the innermost field.
        slot_ptr, slot_ty = self._lvalue_slot(a.target)
        value = self._gen_expr(a.value)
        if value is None:
            raise CodegenError("assignment RHS has no value")
        # Reassignment: release the old value before overwriting.
        # Covers every cleanup-bearing slot type — str (cap-sentinel
        # no-ops borrows), tablets (frees the chunk chain), or a
        # user struct that transitively owns cleanup-bearing fields.
        # Without this, any prior heap state leaks on reassign.
        if self._is_str_value(slot_ty):
            self.builder.call(self._get_str_release(), [slot_ptr])
        else:
            info = self._tablets_info_for(slot_ty)
            if info is not None:
                self.builder.call(info.release, [slot_ptr])
            elif (
                self._struct_fields_for(slot_ty) is not None
                and self._struct_needs_cleanup(slot_ty)
            ):
                self.builder.call(
                    self._get_struct_release(slot_ty), [slot_ptr],
                )
        self.builder.store(self._coerce(value, slot_ty), slot_ptr)

    def _lvalue_slot(self, target: A.Expr) -> tuple[ir.Value, ir.Type]:
        """Resolve an lvalue to (pointer-to-slot, value-type-at-slot).

        Root must be a mut-bound Ident. Each Field step GEPs one level
        deeper through the appropriate user-struct LLVM type."""
        assert self.builder is not None
        if isinstance(target, A.Ident):
            var = self._lookup(target.name)
            if not var.is_mut:
                raise CodegenError(
                    f"cannot assign to step binding {target.name!r}"
                )
            return var.ir_ref, var.value_ty
        if isinstance(target, A.Field):
            parent_ptr, parent_ty = self._lvalue_slot(target.target)
            fields = self._struct_fields_for(parent_ty)
            if fields is None:
                raise CodegenError(
                    f"field assignment: {parent_ty} is not a user tablet"
                )
            for i, (fname, fty) in enumerate(fields):
                if fname == target.name:
                    field_ptr = self.builder.gep(
                        parent_ptr,
                        [ir.Constant(I32, 0), ir.Constant(I32, i)],
                        inbounds=True,
                    )
                    return field_ptr, fty
            raise CodegenError(
                f"tablet has no field {target.name!r}"
            )
        raise CodegenError(
            f"assignment target must be a variable or field chain, "
            f"got {type(target).__name__}"
        )

    # --- expressions ---

    def _gen_expr(self, e: A.Expr) -> ir.Value | None:
        """Generate code for an expression. Returns None if the expression
        diverges (e.g. a block where all paths yield) or has no value (e.g.
        an `if` without `else`, which produces unit)."""
        line = getattr(e, "line", 0)
        col = getattr(e, "col", 0)
        if line:
            self._current_loc = (line, col)
        if isinstance(e, A.IntLit):
            return ir.Constant(I64, e.value)
        if isinstance(e, A.CharLit):
            return ir.Constant(I8, e.value)
        if isinstance(e, A.BoolLit):
            return ir.Constant(I1, 1 if e.value else 0)
        if isinstance(e, A.LostLit):
            # Lowered as a typed-but-generic null — an `i8*` null that
            # `_coerce` bitcasts to the actual `tablet T` pointer type
            # at every use site.
            return ir.Constant(I8.as_pointer(), None)
        if isinstance(e, A.StringLit):
            return self._gen_string_lit(e.value)
        if isinstance(e, A.SexLit):
            return self._gen_sex_lit(e)
        if isinstance(e, A.StructLit):
            return self._gen_struct_lit(e)
        if isinstance(e, A.Field):
            return self._gen_field(e)
        if isinstance(e, A.Index):
            return self._gen_index(e)
        if isinstance(e, A.Slice):
            return self._gen_slice(e)
        if isinstance(e, A.Ident):
            return self._gen_ident(e)
        if isinstance(e, A.Block):
            return self._gen_block(e)
        if isinstance(e, A.IfExpr):
            return self._gen_if_expr(e)
        if isinstance(e, A.Unary):
            return self._gen_unary(e)
        if isinstance(e, A.Binary):
            return self._gen_binary(e)
        if isinstance(e, A.Call):
            return self._gen_call(e)
        if isinstance(e, A.Cast):
            value = self._gen_expr(e.value)
            if value is None:
                raise CodegenError("cannot cast a diverging expression")
            target = self._lower_type(e.type)
            return self._coerce(value, target)
        if isinstance(e, A.MatchExpr):
            return self._gen_match(e)
        raise CodegenError(f"expression not supported yet: {type(e).__name__}")

    def _gen_if_expr(self, e: A.IfExpr) -> ir.Value | None:
        assert self.builder is not None
        cond = self._gen_expr(e.cond)
        if cond is None:
            raise CodegenError("if condition diverged")
        if cond.type != I1:
            raise CodegenError(f"if condition must be bool, got {cond.type}")

        fn = self.builder.function
        then_bb = fn.append_basic_block("if.then")
        merge_bb = fn.append_basic_block("if.merge")

        # No else — value is always None (unit). Useful only in statement position.
        if e.else_ is None:
            self.builder.cbranch(cond, then_bb, merge_bb)
            self.builder.position_at_end(then_bb)
            self._gen_expr(e.then)
            if not self._is_terminated():
                self.builder.branch(merge_bb)
            self.builder.position_at_end(merge_bb)
            return None

        else_bb = fn.append_basic_block("if.else")
        self.builder.cbranch(cond, then_bb, else_bb)

        self.builder.position_at_end(then_bb)
        then_val = self._gen_expr(e.then)
        then_end = self.builder.block
        # Snapshot whether the arm diverged BEFORE we insert the fall-through
        # branch to merge (which itself is a terminator).
        then_diverged = then_end.is_terminated
        if not then_diverged:
            self.builder.branch(merge_bb)

        self.builder.position_at_end(else_bb)
        else_val = self._gen_expr(e.else_)
        else_end = self.builder.block
        else_diverged = else_end.is_terminated
        if not else_diverged:
            self.builder.branch(merge_bb)

        self.builder.position_at_end(merge_bb)

        # Both arms diverged: make the merge block a valid (unreachable) block
        # so the IR verifier is happy, and any outer code sees that we're
        # terminated too. This is the "diverging if" case.
        if then_diverged and else_diverged:
            self.builder.unreachable()
            return None

        # One side diverged — the value, if any, comes from the other.
        if then_diverged:
            return else_val
        if else_diverged:
            return then_val

        # Both sides reach merge.
        if then_val is None and else_val is None:
            return None
        if then_val is None or else_val is None:
            raise CodegenError(
                "if arms disagree: one has a trailing expression, the other does not"
            )
        if then_val.type != else_val.type:
            raise CodegenError(
                f"if arms have different types: {then_val.type} vs {else_val.type}"
            )
        phi = self.builder.phi(then_val.type)
        phi.add_incoming(then_val, then_end)
        phi.add_incoming(else_val, else_end)
        return phi

    def _gen_ident(self, e: A.Ident) -> ir.Value:
        # A bare identifier may also name a nullary seal variant —
        # recognise it via the checker's sideband so `None` / `Empty` /
        # etc. construct the correct seal value.
        if (
            self._checker is not None
            and id(e) in self._checker.variant_of_node
        ):
            return self._gen_variant_ctor(e)
        var = self._lookup(e.name)
        assert self.builder is not None
        if var.is_mut:
            return self.builder.load(var.ir_ref, name=e.name)
        return var.ir_ref

    def _gen_block(self, b: A.Block) -> ir.Value | None:
        """Evaluate a block. Returns the value of its trailing expression, or
        None if the block has no tail or diverged before reaching it."""
        self.scopes.append({})
        self._cleanup_frames.append([])
        try:
            for stmt in b.stmts:
                if self._is_terminated():
                    break   # dead code after a yield
                self._gen_stmt(stmt)
            if self._is_terminated():
                return None
            if b.tail is None:
                tail_val: ir.Value | None = None
            else:
                tail_val = self._gen_expr(b.tail)
            # Ownership transfer on fall-through: if the tail is an Ident
            # bound in this frame with a heap-owning type, drop its
            # cleanup entry so the caller receives the live value. Without
            # this the scope-exit release frees the heap bytes before the
            # return, leaving the caller with a dangling pointer.
            if (
                not self._is_terminated()
                and tail_val is not None
                and isinstance(b.tail, A.Ident)
            ):
                self._transfer_ownership_out(b.tail.name)
            # Emit cleanups for this frame on fall-through (not on early
            # return — yield emits its own chain before the ret).
            if not self._is_terminated():
                self._emit_frame_cleanups(self._cleanup_frames[-1])
            return tail_val
        finally:
            self.scopes.pop()
            self._cleanup_frames.pop()

    def _transfer_ownership_out(self, name: str) -> None:
        """Remove the cleanup entry that owns the value flowing out via
        `name`'s tail position. If `name` is a borrow (its Variable has
        `transfer_on_tail` set), redirect to that source — the actual
        heap owner. Used when a block's tail expression returns a
        locally-bound value so the scope-exit release doesn't fire on
        the escaping heap."""
        if not self._cleanup_frames:
            return
        try:
            var = self._lookup(name)
        except CodegenError:
            var = None
        entry_name = (
            var.transfer_on_tail if var is not None and var.transfer_on_tail
            else name
        )
        frame = self._cleanup_frames[-1]
        for i, (_fn, _ptr, fname) in enumerate(frame):
            if fname == entry_name:
                frame.pop(i)
                return

    def _frame_has_entry(self, name: str) -> bool:
        if not self._cleanup_frames:
            return False
        return any(n == name for _fn, _ptr, n in self._cleanup_frames[-1])

    def _emit_frame_cleanups(
        self, frame: list[tuple[ir.Function, ir.Value, str]],
    ) -> None:
        """Emit release calls for the given cleanup frame, in reverse
        declaration order — matches C++ RAII / Rust Drop ordering so
        references between bindings unwind safely."""
        assert self.builder is not None
        for release_fn, ptr, _name in reversed(frame):
            self.builder.call(release_fn, [ptr])

    def _gen_unary(self, e: A.Unary) -> ir.Value:
        assert self.builder is not None
        operand = self._gen_expr(e.operand)
        if operand is None:
            raise CodegenError(f"unary {e.op} operand has no value")
        if e.op == "-":
            if operand.type == SEX:
                # Flip sign byte in place; digits untouched.
                sign = self.builder.extract_value(operand, SEX_IDX_SIGN)
                flipped = self.builder.xor(sign, ir.Constant(I8, 1))
                return self.builder.insert_value(operand, flipped, SEX_IDX_SIGN)
            if operand.type == RAT:
                num = self.builder.extract_value(operand, 0)
                return self.builder.insert_value(operand, self.builder.neg(num), 0)
            if not isinstance(operand.type, ir.IntType) or operand.type.width < 8:
                raise CodegenError(f"unary - requires integer, got {operand.type}")
            return self.builder.neg(operand)
        if e.op == "!":
            if operand.type != I1:
                raise CodegenError(f"unary ! requires bool, got {operand.type}")
            return self.builder.not_(operand)
        raise CodegenError(f"unknown unary op: {e.op}")

    def _gen_binary(self, e: A.Binary) -> ir.Value:
        assert self.builder is not None
        lhs = self._gen_expr(e.lhs)
        rhs = self._gen_expr(e.rhs)
        if lhs is None or rhs is None:
            raise CodegenError(f"operand of binary {e.op} has no value")
        op = e.op

        # Mixed sex + int: promote the int to sex (int→sex is a
        # lossless base-60 decomposition) so the native digit-form path
        # handles the op.
        if op in ("+", "-", "*", "/"):
            if lhs.type == SEX and isinstance(rhs.type, ir.IntType):
                rhs = self._coerce(rhs, SEX)
            elif isinstance(lhs.type, ir.IntType) and rhs.type == SEX:
                lhs = self._coerce(lhs, SEX)

        # Native Babylonian arithmetic for sex+sex / sex-sex. The type
        # checker has already declared the result type as sex here, so no
        # warning is emitted; digit form is preserved through the op.
        if lhs.type == SEX and rhs.type == SEX and op in ("+", "-"):
            if op == "-":
                # a - b = a + (-b); negation is a sign-byte flip, free.
                rhs_sign = self.builder.extract_value(rhs, SEX_IDX_SIGN)
                flipped = self.builder.xor(rhs_sign, ir.Constant(I8, 1))
                rhs = self.builder.insert_value(rhs, flipped, SEX_IDX_SIGN)
            return self.builder.call(self._get_sex_add(), [lhs, rhs])

        # Native sex*sex and sex/sex: lower through rat, then reconstruct
        # a sex via the regularity-checked helper. Traps at runtime if
        # the result isn't a regular number (den not 2^a·3^b·5^c), or
        # on divide-by-zero (rat_reduce's existing trap).
        if lhs.type == SEX and rhs.type == SEX and op in ("*", "/"):
            lhs_rat = self._coerce(lhs, RAT)
            rhs_rat = self._coerce(rhs, RAT)
            result_rat = self._gen_rat_binary(op, lhs_rat, rhs_rat)
            return self.builder.call(self._get_rat_to_sex(), [result_rat])

        # Everything else still lowers sex to rat — the warning path the
        # type checker announced. Phase 3 will replace more of this with
        # native digit-sequence operations (multiplication, division).
        if lhs.type == SEX:
            lhs = self._coerce(lhs, RAT)
        if rhs.type == SEX:
            rhs = self._coerce(rhs, RAT)

        # --- rat arithmetic and comparison ---
        if lhs.type == RAT and rhs.type == RAT:
            return self._gen_rat_binary(op, lhs, rhs)

        if op in ("+", "-", "*", "/", "%"):
            if lhs.type != rhs.type or not isinstance(lhs.type, ir.IntType):
                raise CodegenError(
                    f"{op} requires matching integer types, got {lhs.type} and {rhs.type}"
                )
            return {
                "+": self.builder.add,
                "-": self.builder.sub,
                "*": self.builder.mul,
                "/": self.builder.sdiv,
                "%": self.builder.srem,
            }[op](lhs, rhs)

        if op in ("<", "<=", ">", ">=", "==", "!="):
            # Mixed-width integer compare: promote to the wider type
            # (matches _unify_if_arms on the checker side).
            if (
                isinstance(lhs.type, ir.IntType)
                and isinstance(rhs.type, ir.IntType)
                and lhs.type.width != rhs.type.width
            ):
                target = lhs.type if lhs.type.width >= rhs.type.width else rhs.type
                lhs = self._coerce(lhs, target)
                rhs = self._coerce(rhs, target)
            # Tablet handle / pointer equality: either side may be the
            # generic `lost` (i8* null) and need bitcasting to the other
            # side's pointer type for icmp to accept.
            if op in ("==", "!=") and (
                isinstance(lhs.type, ir.PointerType)
                or isinstance(rhs.type, ir.PointerType)
            ):
                if isinstance(lhs.type, ir.PointerType) and isinstance(rhs.type, ir.PointerType):
                    if lhs.type != rhs.type:
                        rhs = self._coerce(rhs, lhs.type)
                    return self.builder.icmp_unsigned(op, lhs, rhs)
            if lhs.type != rhs.type or not isinstance(lhs.type, ir.IntType):
                raise CodegenError(
                    f"comparison requires matching types, got {lhs.type} and {rhs.type}"
                )
            return self.builder.icmp_signed(op, lhs, rhs)

        if op in ("&&", "||"):
            if lhs.type != I1 or rhs.type != I1:
                raise CodegenError(f"{op} requires bool operands")
            # Non-short-circuit for now — see §7 for branch-based impl.
            return self.builder.and_(lhs, rhs) if op == "&&" else self.builder.or_(lhs, rhs)

        raise CodegenError(f"unsupported binary op: {op}")

    def _gen_call(self, e: A.Call) -> ir.Value | None:
        # Method call on a tablets receiver — plain Ident or a field
        # chain rooted at one. For the field-chain case (buf.bytes.push)
        # we GEP through the struct to the tablets slot and dispatch on
        # a synthetic mut Variable referencing that inner slot.
        if isinstance(e.callee, A.Field):
            if isinstance(e.callee.target, A.Ident):
                try:
                    var = self._lookup(e.callee.target.name)
                except CodegenError:
                    var = None
                if var is not None:
                    info = self._tablets_info_for(var.value_ty)
                    if info is not None:
                        return self._gen_tablets_method(
                            info, var, e.callee.name, e.args,
                        )
            elif isinstance(e.callee.target, A.Field):
                try:
                    slot_ptr, slot_ty = self._lvalue_slot(e.callee.target)
                except CodegenError:
                    slot_ptr = None
                if slot_ptr is not None:
                    info = self._tablets_info_for(slot_ty)
                    if info is not None:
                        inner = Variable(
                            is_mut=True, ir_ref=slot_ptr, value_ty=slot_ty,
                        )
                        return self._gen_tablets_method(
                            info, inner, e.callee.name, e.args,
                        )

        if not isinstance(e.callee, A.Ident):
            raise CodegenError("only direct function calls are supported")
        name = e.callee.name

        # Variant constructor call: `Some(x)`, `Circle(r)`, etc.
        # Checker has already resolved this to (seal, variant, type args).
        if (
            self._checker is not None
            and id(e) in self._checker.variant_of_node
        ):
            return self._gen_variant_ctor(e)

        # Intrinsics dispatch first so user-defined shadows can't occur
        # (they'd have been rejected at declaration time anyway).
        if name == "print":
            return self._gen_print(e.args, newline=False)
        if name == "println":
            return self._gen_print(e.args, newline=True)
        if name == "read_int":
            return self._gen_read_int(e.args)
        if name == "rat":
            return self._gen_rat_ctor(e.args)
        if name == "str_concat":
            return self._gen_str_concat_call(e.args)
        if name == "str_slice":
            return self._gen_str_slice_call(e.args)
        if name == "int_to_str":
            return self._gen_to_str_call(e.args, self._get_int_to_str(), I64)
        if name == "sex_to_str":
            from ._common import SEX as _SEX
            return self._gen_to_str_call(e.args, self._get_sex_to_str(), _SEX)
        if name == "bytes_to_str":
            return self._gen_bytes_to_str_call(e.args)

        # Generic fn call: look up the concrete type args inferred by
        # the checker and dispatch to (emitting if necessary) the
        # matching monomorphization. Non-generic calls take the normal
        # path via self.functions.
        mono_args = None
        if self._checker is not None:
            mono_args = self._checker.mono_call_args.get(id(e))
        if mono_args is not None:
            arg_tys_llvm = tuple(self._lower_ty(a) for a in mono_args)
            fn = self._get_monomorph_fn(name, arg_tys_llvm)
        else:
            fn = self.functions.get(name)
        if fn is None:
            raise CodegenError(f"unknown function {name!r}")
        # Colophon calls use a dedicated dispatch path that marshals
        # each arg and return across the C-ABI boundary.
        if name in self._colophon_decls:
            return self._gen_colophon_call(
                self._colophon_decls[name], fn, e.args,
            )
        if len(e.args) != len(fn.args):
            raise CodegenError(
                f"{name} expects {len(fn.args)} args, got {len(e.args)}"
            )
        assert self.builder is not None
        call_args = []
        for i, arg in enumerate(e.args):
            expected_ty = fn.args[i].type
            # Mut tablets param: callee expects a pointer to the caller's
            # tablets storage. If the arg is an Ident naming a mut
            # tablets binding, pass its alloca directly (no load).
            if (
                isinstance(expected_ty, ir.PointerType)
                and self._tablets_info_for(expected_ty.pointee) is not None
                and isinstance(arg, A.Ident)
            ):
                var = self._lookup(arg.name)
                if var.is_mut and self._tablets_info_for(var.value_ty) is not None:
                    call_args.append(var.ir_ref)
                    continue
                raise CodegenError(
                    f"argument {i} of {name!r}: mut tablets parameter "
                    f"requires a mut tablets argument, got {var.value_ty}"
                )
            v = self._gen_expr(arg)
            if v is None:
                raise CodegenError(f"argument {i} of call to {name} has no value")
            coerced = self._coerce(v, expected_ty)
            # Cleanup-bearing args: transfer ownership of any fresh
            # heap-owning rvalue to an anonymous slot in the current
            # cleanup frame (so the bytes outlive the call and free at
            # scope exit), then hand the callee a borrow — cap=0 for
            # str, cleanup markers zeroed for struct fields — so the
            # callee's own scope-exit release is a no-op on every
            # heap-owning field. Caller retains sole ownership.
            if self._is_str_value(expected_ty):
                self._register_str_rvalue_cleanup(coerced, arg)
                coerced = self._str_as_borrow(coerced)
            elif (
                self._struct_fields_for(expected_ty) is not None
                and self._struct_needs_cleanup(expected_ty)
            ):
                self._register_struct_rvalue_cleanup(
                    coerced, arg, expected_ty,
                )
                coerced = self._struct_as_borrow(coerced, expected_ty)
            call_args.append(coerced)
        return self.builder.call(fn, call_args)

    # --- intrinsics: stdlib I/O -----------------------------------------

    def _str_ptr(self, data: bytes) -> ir.Value:
        """Return an i8* pointing to a global, NUL-terminated copy of `data`.
        Deduplicates identical strings via `self._strings`."""
        assert self.builder is not None
        g = self._strings.get(data)
        if g is None:
            payload = data + b"\0"
            ty = ir.ArrayType(I8, len(payload))
            g = ir.GlobalVariable(self.module, ty, name=f".str.{self._str_counter}")
            self._str_counter += 1
            g.linkage = "internal"
            g.global_constant = True
            g.initializer = ir.Constant(ty, bytearray(payload))
            self._strings[data] = g
        zero = ir.Constant(I32, 0)
        return self.builder.gep(g, [zero, zero], inbounds=True)

    def _gen_print(self, args: list[A.Expr], *, newline: bool) -> None:
        if not args:
            raise CodegenError(
                f"{'println' if newline else 'print'} takes at least one argument"
            )
        assert self.builder is not None
        # Each argument is emitted without a newline; if `newline=True`
        # the trailing newline goes AFTER the last argument only.
        for i, arg in enumerate(args):
            val = self._gen_expr(arg)
            if val is None:
                raise CodegenError("print argument has no value")
            self._register_str_rvalue_cleanup(val, arg)
            last = (i == len(args) - 1)
            self._emit_one_print(val, newline=(newline and last))

    def _emit_one_print(self, val: ir.Value, *, newline: bool) -> None:
        assert self.builder is not None
        # Dispatch on runtime IR type.
        if val.type == I1:
            fmt = "%s\n" if newline else "%s"
            choice = self.builder.select(
                val, self._str_ptr(b"true"), self._str_ptr(b"false"),
            )
            self.builder.call(self.printf, [self._str_ptr(fmt.encode()), choice])
            return

        if isinstance(val.type, ir.IntType):
            fmt = "%lld\n" if newline else "%lld"
            v64 = self._coerce(val, I64)
            self.builder.call(self.printf, [self._str_ptr(fmt.encode()), v64])
            return

        if val.type == SEX:
            self._emit_sex_print(val, newline=newline)
            return

        # Seal types must be checked before RAT, since a user seal may be
        # structurally equal to the rat struct at the LLVM level.
        if self._is_str_value(val.type):
            ptr = self.builder.extract_value(val, 0)
            length = self.builder.extract_value(val, 1)
            null_file = ir.Constant(I8.as_pointer(), None)
            self.builder.call(self._get_fflush(), [null_file])
            stdout_fd = ir.Constant(I32, 1)
            self.builder.call(self._get_write(), [stdout_fd, ptr, length])
            if newline:
                self.builder.call(self._get_write(), [
                    stdout_fd, self._str_ptr(b"\n"), ir.Constant(I64, 1),
                ])
            return

        if val.type == RAT:
            num = self.builder.extract_value(val, 0)
            den = self.builder.extract_value(val, 1)
            fmt = "%lld/%lld\n" if newline else "%lld/%lld"
            self.builder.call(self.printf, [self._str_ptr(fmt.encode()), num, den])
            return

        raise CodegenError(f"print: unsupported value type {val.type}")

    def _gen_read_int(self, args: list[A.Expr]) -> ir.Value:
        if args:
            raise CodegenError("read_int takes no arguments")
        assert self.builder is not None
        slot = self._alloca_entry(I64, "readint_slot")
        self.builder.call(self.scanf, [self._str_ptr(b"%lld"), slot])
        return self.builder.load(slot, name="readint_val")

    # --- intrinsics: rat constructor ------------------------------------

    def _gen_rat_ctor(self, args: list[A.Expr]) -> ir.Value:
        if len(args) != 2:
            raise CodegenError("rat() takes exactly two arguments (num, den)")
        assert self.builder is not None
        num = self._gen_expr(args[0])
        den = self._gen_expr(args[1])
        if num is None or den is None:
            raise CodegenError("rat() argument has no value")
        num = self._coerce(num, I64)
        den = self._coerce(den, I64)
        return self.builder.call(self._get_rat_reduce(), [num, den])

    # --- dynamic-string intrinsic emitters ----------------------------

    # Ownership rule: str intrinsic results are heap-owned (cap > 0).
    # When consumed as an arg to another call — intrinsic or user fn —
    # the consumer registers an anonymous cleanup slot so the heap bytes
    # outlive the call and get freed at scope exit. User fn calls
    # additionally zero the callee's cap so the callee's own cleanup
    # frame can register the param uniformly without double-free.

    def _gen_str_concat_call(self, args: list[A.Expr]) -> ir.Value:
        assert self.builder is not None
        a = self._gen_expr(args[0])
        b = self._gen_expr(args[1])
        if a is None or b is None:
            raise CodegenError("str_concat argument has no value")
        self._register_str_rvalue_cleanup(a, args[0])
        self._register_str_rvalue_cleanup(b, args[1])
        return self.builder.call(self._get_str_concat(), [a, b])

    def _gen_bytes_to_str_call(self, args: list[A.Expr]) -> ir.Value:
        """Lower `bytes_to_str(t)` — flatten a `tablets[N]u8` into a
        heap-owned str via the per-N monomorph. The arg is evaluated
        by value; for a mut tablets Ident that's `load(alloca)`, which
        hands the intrinsic the current {head, tail, len} metadata."""
        assert self.builder is not None
        if len(args) != 1:
            raise CodegenError("bytes_to_str takes exactly one argument")
        v = self._gen_expr(args[0])
        if v is None:
            raise CodegenError("bytes_to_str argument has no value")
        info = self._tablets_info_for(v.type)
        if info is None or info.elem_ty != I8:
            raise CodegenError(
                f"bytes_to_str: argument must be tablets[N]u8, got {v.type}"
            )
        return self.builder.call(self._get_bytes_to_str(info.N), [v])

    def _gen_str_slice_call(self, args: list[A.Expr]) -> ir.Value:
        assert self.builder is not None
        s = self._gen_expr(args[0])
        lo = self._gen_expr(args[1])
        hi = self._gen_expr(args[2])
        if s is None or lo is None or hi is None:
            raise CodegenError("str_slice argument has no value")
        self._register_str_rvalue_cleanup(s, args[0])
        lo = self._coerce(lo, I64)
        hi = self._coerce(hi, I64)
        return self.builder.call(self._get_str_slice(), [s, lo, hi])

    def _gen_to_str_call(
        self, args: list[A.Expr], fn: ir.Function, arg_ty: ir.Type,
    ) -> ir.Value:
        assert self.builder is not None
        v = self._gen_expr(args[0])
        if v is None:
            raise CodegenError("to_str argument has no value")
        v = self._coerce(v, arg_ty)
        return self.builder.call(fn, [v])

    # --- field access ---------------------------------------------------

    def _gen_field(self, e: A.Field) -> ir.Value:
        assert self.builder is not None

        # Fast path: field on a named tablets variable — GEP directly, skip
        # loading the whole struct.
        if isinstance(e.target, A.Ident):
            try:
                var = self._lookup(e.target.name)
            except CodegenError:
                var = None
            if var is not None and self._tablets_info_for(var.value_ty) is not None:
                return self._gen_tablets_field(var, e.name)

        target = self._gen_expr(e.target)
        if target is None:
            raise CodegenError("field access target has no value")
        # Tablet handle: auto-deref. The handle is a pointer to the
        # underlying struct; GEP into it to the field slot, then load.
        if isinstance(target.type, ir.PointerType):
            pointee = target.type.pointee
            fields = self._struct_fields_for(pointee)
            if fields is not None:
                for i, (fname, _fty) in enumerate(fields):
                    if fname == e.name:
                        field_ptr = self.builder.gep(
                            target,
                            [ir.Constant(I32, 0), ir.Constant(I32, i)],
                            inbounds=True,
                        )
                        return self.builder.load(field_ptr)
                raise CodegenError(
                    f"tablet has no field {e.name!r}"
                )
        # Check user-defined tablets BEFORE rat: a `tablet P { x: i64, y: i64 }`
        # is structurally equal to RAT at the LLVM level, but identity
        # comparison against _struct_types distinguishes them correctly.
        fields = self._struct_fields_for(target.type)
        if fields is not None:
            for i, (fname, _fty) in enumerate(fields):
                if fname == e.name:
                    return self.builder.extract_value(target, i)
            raise CodegenError(
                f"tablet has no field {e.name!r}"
            )
        if target.type == RAT:
            if e.name == "num":
                return self.builder.extract_value(target, 0)
            if e.name == "den":
                return self.builder.extract_value(target, 1)
            raise CodegenError(f"rat has no field {e.name!r}; only num and den")
        if target.type == SEX and e.name in ("num", "den"):
            # Sex has no literal num/den fields; reduce first.
            as_rat = self._coerce(target, RAT)
            return self.builder.extract_value(as_rat, 0 if e.name == "num" else 1)
        raise CodegenError(f"field access on {target.type} not supported yet")

    def _gen_string_lit(self, data: bytes) -> ir.Value:
        """Lower a string literal to a `str` tablet: `{ ptr, len, cap }`.
        Literals carry cap=0 to mark them as borrowed (immortal global
        storage — `str_release` is a no-op for cap=0)."""
        assert self.builder is not None
        if "str" not in self._struct_types:
            raise CodegenError(
                "string literal used but `str` tablet is not registered "
                "(driver should have auto-injected it)"
            )
        struct_ty = self._struct_types["str"]
        ptr = self._str_ptr(data)                      # i8*
        length = ir.Constant(I64, len(data))
        value: ir.Value = ir.Constant(struct_ty, ir.Undefined)
        value = self.builder.insert_value(value, ptr, 0)
        value = self.builder.insert_value(value, length, 1)
        value = self.builder.insert_value(value, ir.Constant(I64, 0), 2)
        return value

    def _is_str_value(self, llvm_ty: ir.Type) -> bool:
        ty = self._struct_types.get("str")
        return ty is not None and ty is llvm_ty

    def _gen_struct_lit(self, e: A.StructLit) -> ir.Value:
        assert self.builder is not None
        # Generic tablet: consult the checker's mono_struct_args to
        # find the concrete type-arg tuple inferred for this literal,
        # then monomorphize.
        mono_args = None
        if self._checker is not None:
            mono_args = self._checker.mono_struct_args.get(id(e))
        if mono_args is not None:
            arg_tys = tuple(self._lower_ty(a) for a in mono_args)
            struct_ty = self._get_monomorph_struct(e.name, arg_tys)
            fields = self._struct_mono_fields[(e.name, arg_tys)]
        else:
            if e.name not in self._struct_types:
                raise CodegenError(f"unknown tablet {e.name!r}")
            struct_ty = self._struct_types[e.name]
            fields = self._struct_fields[e.name]
        provided: dict[str, A.Expr] = dict(e.fields)
        value: ir.Value = ir.Constant(struct_ty, ir.Undefined)
        for i, (fname, fty) in enumerate(fields):
            if fname not in provided:
                raise CodegenError(
                    f"tablet {e.name!r}: missing field {fname!r}"
                )
            fv = self._gen_expr(provided[fname])
            if fv is None:
                raise CodegenError(
                    f"tablet {e.name!r} field {fname!r}: initializer has no value"
                )
            value = self.builder.insert_value(value, self._coerce(fv, fty), i)
        return value

    # --- tablets method/field/index dispatch -----------------------------

    def _emit_table(self, decl: A.TableDecl) -> None:
        try:
            values = self.comptime.eval_table(decl)
            lo = self.comptime.eval_constant_expr(decl.lo)
        except ComptimeError as e:
            raise CodegenError(f"table {decl.name!r}: {e}") from None

        elem_ty = self._lower_type(decl.element_type)
        array_ty = ir.ArrayType(elem_ty, len(values))

        try:
            constants = [self._py_value_to_constant(v, elem_ty) for v in values]
        except CodegenError as e:
            raise CodegenError(f"table {decl.name!r}: {e}") from None

        g = ir.GlobalVariable(self.module, array_ty, name=decl.name)
        g.linkage = "internal"
        g.global_constant = True
        g.initializer = ir.Constant(array_ty, constants)

        self._tables[decl.name] = (g, len(values), lo, elem_ty)

    def _py_value_to_constant(self, v, target_ty: ir.Type) -> ir.Constant:
        if target_ty == I1:
            if isinstance(v, bool):
                return ir.Constant(I1, 1 if v else 0)
            raise CodegenError(f"expected bool for i1, got {type(v).__name__}")
        if isinstance(target_ty, ir.IntType):
            if isinstance(v, int) and not isinstance(v, bool):
                return ir.Constant(target_ty, v)
            raise CodegenError(
                f"expected int for {target_ty}, got {type(v).__name__}"
            )
        if target_ty == RAT:
            if isinstance(v, tuple) and len(v) == 2:
                return ir.Constant(RAT, (
                    ir.Constant(I64, v[0]),
                    ir.Constant(I64, v[1]),
                ))
            raise CodegenError(
                f"expected (num, den) tuple for rat, got {type(v).__name__}"
            )
        raise CodegenError(f"cannot lower comptime {v!r} to {target_ty}")

    def _gen_index(self, e: A.Index) -> ir.Value:
        assert self.builder is not None
        # Comptime table lookup
        if isinstance(e.target, A.Ident) and e.target.name in self._tables:
            g, size, lo, _elem_ty = self._tables[e.target.name]
            idx = self._gen_expr(e.index)
            if idx is None:
                raise CodegenError("table index has no value")
            idx = self._coerce(idx, I64)
            if lo != 0:
                idx = self.builder.sub(idx, ir.Constant(I64, lo))
            self._emit_bounds_trap(idx, size)
            zero = ir.Constant(I32, 0)
            elem_ptr = self.builder.gep(g, [zero, idx], inbounds=True)
            return self.builder.load(elem_ptr)

        # Tablets indexing (dynamic bounds check vs len)
        if isinstance(e.target, A.Ident):
            try:
                var = self._lookup(e.target.name)
            except CodegenError:
                var = None
            if var is not None:
                info = self._tablets_info_for(var.value_ty)
                if info is not None:
                    return self._gen_tablets_index(info, var, e.index)

        # str indexing: bounds-checked byte load through s.ptr.
        target = self._gen_expr(e.target)
        if target is not None and self._is_str_value(target.type):
            idx = self._gen_expr(e.index)
            if idx is None:
                raise CodegenError("str index has no value")
            idx_i64 = self._coerce(idx, I64)
            ptr = self.builder.extract_value(target, 0)    # i8*
            length = self.builder.extract_value(target, 1) # i64
            self._emit_dynamic_bounds_trap(idx_i64, length)
            byte_ptr = self.builder.gep(ptr, [idx_i64], inbounds=True)
            return self.builder.load(byte_ptr)

        raise CodegenError("indexing is only supported on tables, tablets, and str")

    def _gen_slice(self, e: A.Slice) -> ir.Value:
        """Lower `s[lo:hi]` (and its elided variants) to a call into
        `__tuppu_str_slice`. Missing lo defaults to 0; missing hi
        defaults to `s.len` — matching Python's open-ended half-slice
        semantics. The result is heap-owned; the surrounding consumer
        site registers the anonymous cleanup, same as any other
        str-returning call."""
        assert self.builder is not None
        target = self._gen_expr(e.target)
        if target is None or not self._is_str_value(target.type):
            raise CodegenError(
                "slice expression target must be a str value"
            )
        # If the target itself is a heap-producing rvalue (e.g.
        # `str_concat(a,b)[0:3]`), register it for cleanup so the
        # bytes don't orphan after the slice call reads them.
        self._register_str_rvalue_cleanup(target, e.target)
        if e.lo is None:
            lo = ir.Constant(I64, 0)
        else:
            lo_val = self._gen_expr(e.lo)
            if lo_val is None:
                raise CodegenError("slice lo bound has no value")
            lo = self._coerce(lo_val, I64)
        if e.hi is None:
            hi = self.builder.extract_value(target, 1)
        else:
            hi_val = self._gen_expr(e.hi)
            if hi_val is None:
                raise CodegenError("slice hi bound has no value")
            hi = self._coerce(hi_val, I64)
        return self.builder.call(self._get_str_slice(), [target, lo, hi])

def codegen(program: A.Program, checker=None) -> ir.Module:
    return Codegen(checker=checker).gen(program)
