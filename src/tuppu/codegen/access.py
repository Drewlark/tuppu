"""Member-access and aggregate-literal codegen: `_gen_field`,
`_gen_index`, `_gen_slice`, `_gen_struct_lit`, `_gen_string_lit`,
`_gen_tablets_lit`, tablets method/field/index dispatch, and the
comptime-backed `_emit_table` / `_py_value_to_constant`. Extracted
from `codegen/__init__.py` as `AccessMixin`."""
from __future__ import annotations

from llvmlite import ir

from .. import ast as A
from ..comptime import ComptimeError
from ._common import (
    CodegenError, Variable,
    I1, I8, I16, I32, I64,
    RAT, SEX,
)


class AccessMixin:
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
            if var is not None and isinstance(var.value_ty, ir.ArrayType):
                if e.name == "len":
                    return ir.Constant(I64, var.value_ty.count)
                raise CodegenError(
                    f"buffer has no field {e.name!r}; only len"
                )

        target = self._gen_expr(e.target)
        if target is None:
            raise CodegenError("field access target has no value")
        # Tablets value read as an SSA (e.g. from a struct field or a
        # fn return). The fast path above only fires for direct Ident
        # bindings; here we cover the general case. Only `.len` is
        # readable off the value — indexing needs a pointer and goes
        # through `_gen_index`'s spill-to-alloca path.
        if self._tablets_info_for(target.type) is not None:
            if e.name == "len":
                return self.builder.extract_value(target, 2)
            raise CodegenError(
                f"tablets has no field {e.name!r}; only len"
            )
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
                        # Wedge-deref reads BORROW the container's
                        # storage — the tablets owns the underlying
                        # bytes. Neuter cleanup markers so passing
                        # this value to a container-owning site
                        # doesn't create a second owner.
                        return self._read_borrow(self.builder.load(field_ptr))
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
                    # Field read is a borrow — see the wedge-deref
                    # comment above. Same neutering rule.
                    return self._read_borrow(
                        self.builder.extract_value(target, i),
                    )
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

    def _gen_tablets_lit(self, e: A.TabletsLit) -> ir.Value:
        """Build a fresh `tablets[N]T` populated with the literal's
        elements. Alloca the header in the current function's entry
        block (zero-init `{head=null, tail=null, len=0}`), push each
        evaluated element via the per-(N, T) push fn, and register a
        release in the current cleanup frame so the chunks free at
        scope exit. Returns the loaded tablets value (callers that
        need the pointer — e.g. the variadic call-site path — look
        through `_gen_tablets_lit_addr` below)."""
        slot = self._gen_tablets_lit_addr(e)
        assert self.builder is not None
        return self.builder.load(slot)

    def _gen_tablets_lit_addr(
        self, e: A.TabletsLit, elem_ty_hint: ir.Type | None = None,
    ) -> ir.Value:
        """Like `_gen_tablets_lit` but returns the alloca pointer. Used
        by the variadic-call path so the callee sees the caller's
        storage directly (same convention as a `mut tablets` param).
        `elem_ty_hint` lets the variadic caller supply the element
        type for zero-arity literals where inference has nothing to
        look at."""
        assert self.builder is not None
        # Resolve the element type. The parser always spells one out in
        # tablets[N]T literals; synthesised variadic literals leave it
        # None, and we take the hint from the caller if provided, else
        # probe the first field's expression type.
        if e.element is not None:
            elem_ty = self._lower_type(e.element)
        elif elem_ty_hint is not None:
            elem_ty = elem_ty_hint
        else:
            if not e.fields:
                raise CodegenError(
                    "variadic literal: cannot infer element type from "
                    "empty field list (use explicit tablets[N]T { ... })"
                )
            probe = self._gen_expr(e.fields[0])
            if probe is None:
                raise CodegenError(
                    "variadic literal: element probe has no value",
                )
            elem_ty = probe.type
        info = self._get_tablets(e.size, elem_ty)
        slot = self._alloca_entry(info.tablets_ty, ".tbls.lit")
        self.builder.store(ir.Constant(info.tablets_ty, None), slot)
        # Register cleanup BEFORE pushing so a push-then-error path
        # still frees what was already allocated. Anonymous entry.
        # GC root push so tablets chunks (head/tail ptrs inside the
        # tablets value) stay reachable when the fields being stored
        # trigger collections mid-build.
        if self._cleanup_frames:
            self._cleanup_frames[-1].append(
                (self._get_tablets_release(info), slot, ".tbls.lit"),
            )
            self._register_gc_root(slot, info.tablets_ty)
        push_fn = self._get_tablets_push(info)
        for fexpr in e.fields:
            v = self._gen_expr(fexpr)
            if v is None:
                raise CodegenError("tablets literal field has no value")
            v = self._coerce(v, info.elem_ty)
            # Cleanup-bearing element (str or cleanup-struct): neuter
            # the element so the tablets holds a borrow-view. The
            # true owner stays in the caller's frame — same convention
            # as passing through a cap=0 str param.
            if self._is_str_value(info.elem_ty):
                v = self._str_as_borrow(v)
            elif (
                self._struct_fields_for(info.elem_ty) is not None
                and self._struct_needs_cleanup(info.elem_ty)
            ):
                v = self._struct_as_borrow(v, info.elem_ty)
            self.builder.call(push_fn, [slot, v])
        return slot

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
            fexpr = provided[fname]
            fv = self._gen_expr(fexpr)
            if fv is None:
                raise CodegenError(
                    f"tablet {e.name!r} field {fname!r}: initializer has no value"
                )
            # Ownership into the field: transfer from an owning Ident,
            # pass through a fresh-owned rvalue unchanged, or deep-
            # clone a borrow source (alias into existing storage, or
            # an Ident naming a borrow binding with no cleanup to
            # transfer). Three-way split avoids the redundant clone
            # that `Box { s: make() }` used to perform.
            if self._is_cleanup_bearing_ty(fty):
                if isinstance(fexpr, A.Ident):
                    transferred = self._transfer_cleanup_into_container(
                        fexpr.name,
                    )
                    if not transferred:
                        fv = self._deep_clone_if_cleanup_bearing(fv)
                elif self._is_borrow_source_expr(fexpr):
                    fv = self._deep_clone_if_cleanup_bearing(fv)
                # else: fresh rvalue already rooted by the `_gen_expr`
                # chokepoint when `fexpr` was evaluated.
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
                if isinstance(var.value_ty, ir.ArrayType):
                    # Buffer indexing — GEP + bounds-trap + load.
                    idx = self._gen_expr(e.index)
                    if idx is None:
                        raise CodegenError("buffer index has no value")
                    idx = self._coerce(idx, I64)
                    self._emit_bounds_trap(idx, var.value_ty.count)
                    elem_ptr = self.builder.gep(
                        var.ir_ref,
                        [ir.Constant(I32, 0), idx],
                        inbounds=True,
                    )
                    return self.builder.load(elem_ptr)

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

        # Tablets value accessed via struct-field or fn-return — SSA
        # form. The Ident fast path above only fires for direct
        # tablets bindings; here we spill to a temp alloca so the
        # runtime get() call has a pointer to walk. Reads only; writes
        # would need an lvalue slot rooted at a mut binding. The read
        # is a borrow — cleanup markers neutered so the caller can't
        # double-free against the container's own release walk.
        if target is not None:
            info = self._tablets_info_for(target.type)
            if info is not None:
                idx = self._gen_expr(e.index)
                if idx is None:
                    raise CodegenError("tablets index has no value")
                idx = self._coerce(idx, I64)
                slot = self._alloca_entry(target.type, ".tbls.view")
                self.builder.store(target, slot)
                length = self.builder.extract_value(target, 2)
                self._emit_dynamic_bounds_trap(idx, length)
                val = self.builder.call(
                    self._get_tablets_get(info), [slot, idx],
                )
                return self._read_borrow(val)

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
        # Heap-producing rvalue targets (`str_concat(a,b)[0:3]`) were
        # already rooted by `_gen_expr`'s chokepoint; no extra spill.
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

