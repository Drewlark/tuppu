"""Type lowering and descriptor codegen: `_lower_type` (the AST →
LLVM type translator), `_get_type_desc` and the GC type-descriptor
emitters, struct/seal trace_fn builders, and `_coerce` (the
universal int / pointer / sex / rat conversion helper used at every
binding, return, and call site). Extracted from `codegen/__init__.py`
as `TypesMixin`."""
from __future__ import annotations

from llvmlite import ir

from .. import ast as A
from ._common import (
    CodegenError, Variable,
    I1, I8, I16, I32, I64,
    INT_WIDTH, RAT, SEX, SEX_MAX_DIGITS,
)


class TypesMixin:
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
        if isinstance(t, A.TypeBuffer):
            elem = self._lower_type(t.element)
            return ir.ArrayType(elem, t.size)
        if isinstance(t, A.TypeVariadicTablets):
            # Resolved identically to a `tablets[VARIADIC_CHUNK_SIZE]T`
            # — see typecheck VARIADIC_CHUNK_SIZE. The variadic marker
            # is only meaningful at call sites.
            from ..typecheck import VARIADIC_CHUNK_SIZE
            elem = self._lower_type(t.element)
            return self._get_tablets(VARIADIC_CHUNK_SIZE, elem).tablets_ty
        if isinstance(t, A.TypePointer):
            elem = self._lower_type(t.element)
            return elem.as_pointer()
        if isinstance(t, A.TypeHandle):
            # `tablet T` — runtime is a pointer to T, distinct from
            # `*T` at the source level but same LLVM representation.
            elem = self._lower_type(t.element)
            return elem.as_pointer()
        if isinstance(t, A.TypeFn):
            param_tys = [self._lower_type(p) for p in t.params]
            ret_ty = (
                self._lower_type(t.return_type) if t.return_type
                else ir.VoidType()
            )
            return ir.FunctionType(ret_ty, param_tys).as_pointer()
        raise CodegenError(
            f"complex types not supported in this stage: {type(t).__name__}"
        )

    def _register_structs(self, decls: list[A.StructDecl]) -> None:
        """Facade: declare then resolve. Callers that don't need to
        interleave with seals can use this single-shot entry point."""
        self._register_structs_declare(decls)
        self._register_structs_resolve(decls)

    def _register_structs_declare(self, decls: list[A.StructDecl]) -> None:
        """Phase A of struct registration: declare every tablet as an
        empty identified LLVM type. Splitting declare from resolve lets
        us interleave with seal registration so struct fields of seal
        type (and vice versa) can see the identified type."""
        self._generic_struct_decls: dict[str, A.StructDecl] = {
            d.name: d for d in decls if d.type_params
        }
        concrete = [d for d in decls if not d.type_params]
        for d in concrete:
            if d.name in self._struct_types:
                raise CodegenError(f"duplicate struct {d.name!r}")
            ident_ty = self.module.context.get_identified_type(d.name)
            self._struct_types[d.name] = ident_ty

    def _register_structs_resolve(self, decls: list[A.StructDecl]) -> None:
        """Phase B/C: cycle-check and resolve each struct's body."""
        concrete = [d for d in decls if not d.type_params]
        by_name = {d.name: d for d in concrete}

        # Phase B: detect direct cycles (cycle in the "inline contains"
        # graph). A field whose type is another struct by value — or an
        # array of that struct — contributes a direct edge. A field
        # that's a pointer or tablets does NOT (the recursion goes
        # through heap indirection, so size is finite).
        direct_deps: dict[str, set[str]] = {}
        for d in concrete:
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
        for d in concrete:
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
        saved_root_counts = self._gc_root_counts
        saved_loc = self._current_loc
        # Give the specialization a fresh scope + cleanup stack so it
        # doesn't inherit state from whichever outer emit we're nested
        # inside. _gen_fn_body will overwrite self.scopes anyway but
        # the cleanup stack needs to start empty here.
        self._cleanup_frames = []
        self._gc_root_counts = []
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
            elif isinstance(p.type, A.TypeVariadicTablets):
                t = t.as_pointer()
            elif (
                p.is_mut
                and self._struct_fields_for(t) is not None
                and not self._is_str_value(t)
            ):
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
        self._fn_param_mut[mono_name] = [p.is_mut for p in decl.params]
        # Temporarily install this specialization under the decl's
        # source name so recursive calls inside the body find it and
        # don't trigger a second monomorphization pass.
        saved_functions = self.functions.get(name)
        self.functions[name] = llvm_fn
        # Mirror the param-mut list under the source name for the
        # duration of recursive body emission.
        saved_param_mut = self._fn_param_mut.get(name)
        self._fn_param_mut[name] = self._fn_param_mut[mono_name]
        try:
            self._gen_fn_body(decl)
        finally:
            self._type_arg_subst = saved_subst
            self.builder = saved_builder
            self.scopes = saved_scopes
            self._cleanup_frames = saved_cleanup
            self._gc_root_counts = saved_root_counts
            self._current_loc = saved_loc
            if saved_functions is None:
                del self.functions[name]
            else:
                self.functions[name] = saved_functions
            if saved_param_mut is None:
                del self._fn_param_mut[name]
            else:
                self._fn_param_mut[name] = saved_param_mut
        return llvm_fn


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
