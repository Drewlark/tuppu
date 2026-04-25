"""Seal (sum type) codegen: decl registration, monomorphization,
variant constructors, pattern matching, and the generic `_size_of`
/ `_align_of` LLVM helpers. Extracted from `codegen/__init__.py` as
`SealsMixin`."""
from __future__ import annotations

from llvmlite import ir

from .. import ast as A
from ._common import (
    CodegenError, Variable,
    I1, I8, I16, I32, I64,
)


class SealsMixin:
    # --- seals (sum types) ---------------------------------------------

    def _register_seals(self, decls: list["A.SealDecl"]) -> None:
        """Facade: declare then resolve. Used when seals don't need to
        interleave with struct registration."""
        self._register_seals_declare(decls)
        self._register_seals_resolve(decls)

    def _register_seals_declare(self, decls: list["A.SealDecl"]) -> None:
        """Phase A of seal registration: declare an empty identified
        LLVM type per concrete seal. Generic seals are stashed for on-
        demand monomorphization — their layout depends on concrete
        type args so we can't emit one up front."""
        self._generic_seal_decls = {
            d.name: d for d in decls if d.type_params
        }
        for d in decls:
            if d.type_params:
                continue
            if d.name in self._seal_types:
                raise CodegenError(f"duplicate seal {d.name!r}")
            ident_ty = self.module.context.get_identified_type(d.name)
            self._seal_types[d.name] = ident_ty

    def _register_seals_resolve(self, decls: list["A.SealDecl"]) -> None:
        """Phase B: compute each concrete seal's payload layout."""
        for d in decls:
            if d.type_params:
                continue
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
                coerced = self._coerce(v, expected_ty)
                # Variant payload is a long-lived container: same
                # three-way split as push / struct-lit / assign.
                # Owning Ident transfers; fresh-owned rvalue passes
                # through; borrow (or Ident naming a borrow) gets
                # deep-cloned.
                if self._is_cleanup_bearing_ty(expected_ty):
                    if isinstance(arg, A.Ident):
                        transferred = self._transfer_cleanup_into_container(
                            arg.name,
                        )
                        if not transferred:
                            coerced = self._deep_clone_if_cleanup_bearing(coerced)
                    elif self._is_borrow_source_expr(arg):
                        coerced = self._deep_clone_if_cleanup_bearing(coerced)
                    # else: fresh-owned rvalue is already rooted by the
                    # `_gen_expr` chokepoint; no extra spill here.
                field_ptr = self.builder.gep(
                    typed_ptr,
                    [ir.Constant(I32, 0), ir.Constant(I32, i)],
                    inbounds=True,
                )
                self.builder.store(coerced, field_ptr)
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
        # Spill scrutinee so we can GEP into it for the payload. No
        # runtime deep-clone needed: the freeze-while-borrow rule
        # rejects the UAF shape (mut-reach to the scrutinee's source
        # while a binder is live) at typecheck, so binders into the
        # shallow match.scrut copy are safe.
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
            self._push_cleanup_frame()
            try:
                self._bind_variant_pattern(arm.pattern, slot, seal_key)
                val = self._gen_expr(arm.body)
                end_bb = self.builder.block
                diverged = end_bb.is_terminated
                if not diverged:
                    val = self._finalize_arm_tail(arm.body, val)
                    self._emit_frame_cleanups(self._cleanup_frames[-1])
                    self._emit_gc_frame_pop()
                    results.append((val, self.builder.block))
                    self.builder.branch(merge_bb)
            finally:
                self.scopes.pop()
                self._pop_cleanup_frame()

        # Emit default (wildcard or unreachable trap).
        self.builder.position_at_end(default_bb)
        if wildcard_arm is not None:
            self.scopes.append({})
            self._push_cleanup_frame()
            try:
                val = self._gen_expr(wildcard_arm.body)
                end_bb = self.builder.block
                diverged = end_bb.is_terminated
                if not diverged:
                    val = self._finalize_arm_tail(wildcard_arm.body, val)
                    self._emit_frame_cleanups(self._cleanup_frames[-1])
                    self._emit_gc_frame_pop()
                    results.append((val, self.builder.block))
                    self.builder.branch(merge_bb)
            finally:
                self.scopes.pop()
                self._pop_cleanup_frame()
        else:
            self.builder.unreachable()

        self.builder.position_at_end(merge_bb)
        if not results:
            # All arms diverged.
            self.builder.unreachable()
            return None
        # Treat void-typed arm values (e.g. a bare `noop()` call tail)
        # as "no value" — LLVM doesn't allow a phi of void type, and
        # the match is being consumed as a statement anyway. Equivalent
        # to the unit-producing branches in a stmt-position if.
        def _usable(v: ir.Value | None) -> bool:
            return v is not None and not isinstance(v.type, ir.VoidType)
        if not any(_usable(r[0]) for r in results):
            return None
        rep = next(r for r in results if _usable(r[0]))
        phi = self.builder.phi(rep[0].type)
        for val, bb in results:
            if not _usable(val):
                # Diverged / unit / void arms don't contribute. Safe
                # because the typechecker already rejects a match
                # whose arms produce mismatched value types.
                continue
            phi.add_incoming(val, bb)
        return phi

    def _finalize_arm_tail(self, body: "A.Expr", val: ir.Value | None) -> ir.Value | None:
        """Apply block-tail ownership-flow to a direct match arm body.
        `_gen_block` already handles Block tails internally; arm bodies
        that ARE a direct Ident / Field / Index don't go through that
        path, so we mirror the same logic here so cleanups fire
        correctly for returned borrows / owned values."""
        if val is None:
            return None
        if isinstance(body, A.Ident):
            # Owning Ident: transfer cleanup out so the arm's scope-
            # exit release doesn't free bytes the match result is
            # about to carry.
            self._transfer_ownership_out(body.name)
        elif isinstance(body, (A.Field, A.Index)):
            # Field/Index tail: the aliased bytes live in some
            # container; clone so the returned match value is
            # independently owned and safe after scope-exit releases.
            val = self._deep_clone_if_cleanup_bearing(val)
        return val

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
            # Match binders on cleanup-bearing payloads are implicit-
            # copied so the arm body can freely mutate the scrutinee's
            # source without dangling the binder. Deep-cloned binder
            # becomes a regular owning step binding — transfer-on-tail
            # moves it out on return, scope-exit releases it
            # otherwise. The alternative was requiring users to write
            # `step n = copy name` at the top of every arm that
            # touches the scrutinee, which turned out to be the
            # majority of match arms in real parsers.
            if self._is_cleanup_bearing_ty(val.type):
                val = self._deep_clone_if_cleanup_bearing(val)
                slot = self._alloca_entry(val.type, f"{binder}.cleanup")
                self.builder.store(val, slot)
                self._maybe_register_cleanup(binder, val.type, slot)
            self._bind(binder, Variable(
                is_mut=False, ir_ref=val, value_ty=val.type,
            ))
