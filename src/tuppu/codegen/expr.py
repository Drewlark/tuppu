"""Expression codegen: the `_gen_expr` dispatcher, literal / binary /
unary / if / match arm / block / call / copy handlers, and the
cleanup-bearing-rvalue plumbing that most of them share. Extracted
from `codegen/__init__.py` as `ExprMixin`."""
from __future__ import annotations

from llvmlite import ir

from .. import ast as A
from ._common import (
    CodegenError, Variable,
    I1, I8, I16, I32, I64,
    RAT, SEX, SEX_IDX_SIGN,
)


class ExprMixin:
    # --- expressions ---

    def _gen_expr(self, e: A.Expr) -> ir.Value | None:
        """Generate code for an expression. Returns None if the expression
        diverges (e.g. a block where all paths yield) or has no value (e.g.
        an `if` without `else`, which produces unit).

        Every cleanup-bearing result is funneled through the chokepoint
        at the bottom — one site covers Call, Binary, Copy, Index, Field
        reads, Cast, and anything else that might yield a GC-tracked
        value. The chokepoint no-ops on scalars / pointers / types
        without a descriptor, so over-rooting is cheap."""
        line = getattr(e, "line", 0)
        col = getattr(e, "col", 0)
        if line:
            self._current_loc = (line, col)
        val: ir.Value | None
        if isinstance(e, A.IntLit):
            return ir.Constant(I64, e.value)
        if isinstance(e, A.CharLit):
            return ir.Constant(I8, e.value)
        if isinstance(e, A.BoolLit):
            return ir.Constant(I1, 1 if e.value else 0)
        if isinstance(e, A.LostLit):
            return ir.Constant(I8.as_pointer(), None)
        if isinstance(e, A.StringLit):
            return self._gen_string_lit(e.value)
        if isinstance(e, A.SexLit):
            return self._gen_sex_lit(e)
        if isinstance(e, A.StructLit):
            val = self._gen_struct_lit(e)
        elif isinstance(e, A.TabletsLit):
            val = self._gen_tablets_lit(e)
        elif isinstance(e, A.Field):
            val = self._gen_field(e)
        elif isinstance(e, A.Index):
            val = self._gen_index(e)
        elif isinstance(e, A.Slice):
            val = self._gen_slice(e)
        elif isinstance(e, A.Ident):
            val = self._gen_ident(e)
        elif isinstance(e, A.Block):
            val = self._gen_block(e)
        elif isinstance(e, A.IfExpr):
            val = self._gen_if_expr(e)
        elif isinstance(e, A.Unary):
            val = self._gen_unary(e)
        elif isinstance(e, A.Binary):
            val = self._gen_binary(e)
        elif isinstance(e, A.Call):
            val = self._gen_call(e)
        elif isinstance(e, A.Cast):
            value = self._gen_expr(e.value)
            if value is None:
                raise CodegenError("cannot cast a diverging expression")
            target = self._lower_type(e.type)
            val = self._coerce(value, target)
        elif isinstance(e, A.Copy):
            val = self._gen_copy(e)
        elif isinstance(e, A.MatchExpr):
            val = self._gen_match(e)
        else:
            raise CodegenError(f"expression not supported yet: {type(e).__name__}")
        if val is not None:
            self._force_root_cleanup_value(val)
        return val

    def _gen_copy(self, e: A.Copy) -> ir.Value | None:
        """`copy x` → deep-clone of x. Scalars and handles pass through
        unchanged (no-op). The result is a freshly-owned rvalue; its
        cleanup is managed by the consumer — `step n = copy x` gets
        cleanup on the binding slot, `container.push(copy x)` transfers
        into the container, and so on. This matches how other fresh-
        rvalue paths (Call results, str concat, etc.) flow through the
        consumer-registers-cleanup discipline."""
        val = self._gen_expr(e.value)
        if val is None:
            raise CodegenError("cannot copy a diverging expression")
        return self._deep_clone_if_cleanup_bearing(val)

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

        # Counter snapshot so arm-internal chokepoint pushes land the
        # same delta on both paths. Without this, then+else compile-
        # time counters both add to the outer frame, but runtime runs
        # one arm — pop at merge would underflow. We emit a balancing
        # pop before each arm's branch to merge and re-push once
        # after the phi so the outer counter advances by exactly one
        # regardless of which arm executes.
        counter_before = (
            self._gc_root_counts[-1] if self._gc_root_counts else 0
        )

        self.builder.position_at_end(then_bb)
        then_val = self._gen_expr(e.then)
        then_end = self.builder.block
        then_diverged = then_end.is_terminated
        if not then_diverged:
            then_delta = (
                (self._gc_root_counts[-1] - counter_before)
                if self._gc_root_counts else 0
            )
            if then_delta > 0:
                self._emit_gc_pop_roots(then_delta)
            self.builder.branch(merge_bb)
        # Reset the Python counter so the else arm starts at the same
        # baseline as the then arm did. Otherwise else_val's chokepoint
        # would stack on top of then's accumulated push count.
        if self._gc_root_counts:
            self._gc_root_counts[-1] = counter_before

        self.builder.position_at_end(else_bb)
        else_val = self._gen_expr(e.else_)
        else_end = self.builder.block
        else_diverged = else_end.is_terminated
        if not else_diverged:
            else_delta = (
                (self._gc_root_counts[-1] - counter_before)
                if self._gc_root_counts else 0
            )
            if else_delta > 0:
                self._emit_gc_pop_roots(else_delta)
            self.builder.branch(merge_bb)
        if self._gc_root_counts:
            self._gc_root_counts[-1] = counter_before

        self.builder.position_at_end(merge_bb)

        # Both arms diverged: make the merge block a valid (unreachable) block
        # so the IR verifier is happy, and any outer code sees that we're
        # terminated too. This is the "diverging if" case.
        if then_diverged and else_diverged:
            self.builder.unreachable()
            return None

        # Statement-position if (typechecker marked it as discarded):
        # arms may have different shapes/types — the value is unused,
        # so fall through to merge and return None. Check this BEFORE
        # the arm-shape / type-match checks because a then-arm ending
        # in a void call vs an else-arm ending in an Assign is a
        # shape mismatch the user shouldn't have to reconcile.
        in_stmt_position = (
            self._checker is not None
            and id(e) in self._checker.stmt_if_nodes
        )
        if in_stmt_position:
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
                "if arms disagree: one has a trailing expression, the other does not",
                e.line, e.col,
            )
        if then_val.type != else_val.type:
            raise CodegenError(
                f"if arms have different types: {then_val.type} vs {else_val.type}",
                e.line, e.col,
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
        # If the name isn't in any local scope but IS a declared fn,
        # evaluate to the LLVM function pointer (first-class value).
        # Colophons are excluded — the typechecker already rejects
        # taking their address.
        try:
            var = self._lookup(e.name)
        except CodegenError:
            fn = self.functions.get(e.name)
            if fn is not None and e.name not in self._colophon_decls:
                return fn
            raise
        assert self.builder is not None
        if var.is_mut:
            return self.builder.load(var.ir_ref, name=e.name)
        return var.ir_ref

    def _gen_block(self, b: A.Block) -> ir.Value | None:
        """Evaluate a block. Returns the value of its trailing expression, or
        None if the block has no tail or diverged before reaching it."""
        self.scopes.append({})
        self._push_cleanup_frame()
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
            # Field / Index tail of a cleanup-bearing type: the bytes
            # are owned elsewhere (by a local struct, a container,
            # etc.). Clone BEFORE firing scope-exit cleanups — the
            # source's cleanup may run here and free the original
            # bytes. Cloning into a caller-owned value sidesteps both
            # the UAF-via-local-cleanup case and the double-free-via-
            # container-walk case.
            if (
                not self._is_terminated()
                and tail_val is not None
                and isinstance(b.tail, (A.Field, A.Index))
            ):
                tail_val = self._deep_clone_if_cleanup_bearing(tail_val)
            # Emit cleanups for this frame on fall-through (not on early
            # return — yield emits its own chain before the ret).
            if not self._is_terminated():
                self._emit_frame_cleanups(self._cleanup_frames[-1])
            return tail_val
        finally:
            self.scopes.pop()
            self._pop_cleanup_frame()

    def _is_borrow_source_expr(self, expr: "A.Expr") -> bool:
        """Does this AST expression read through to bytes someone else
        owns? Ident (may be a borrow binding), Field, Index, match
        pattern binders (bound as Ident), and StringLit all alias into
        existing storage. Everything else — Call, Binary(str+), Copy,
        StructLit, TabletsLit, variant construction via Call, Slice,
        Cast — produces a fresh-owned rvalue that storage sites can
        take over directly without cloning."""
        return isinstance(expr, (A.Ident, A.Field, A.Index, A.StringLit))

    def _is_cleanup_bearing_ty(self, ty: ir.Type) -> bool:
        """Does this LLVM type hold heap state that needs a release
        call on scope exit? Used to decide whether a container push /
        struct-lit field needs to transfer ownership from its Ident
        source to avoid double-free."""
        if self._is_str_value(ty):
            return True
        if self._tablets_info_for(ty) is not None:
            return True
        if (
            self._struct_fields_for(ty) is not None
            and self._struct_needs_cleanup(ty)
        ):
            return True
        if (
            self._seal_key_for_ty(ty) is not None
            and self._seal_needs_cleanup(ty)
        ):
            return True
        return False

    def _transfer_cleanup_into_container(
        self, name: str, *, defer_zero: bool = False,
    ) -> "ir.Value | bool":
        """Remove the cleanup entry owning `name` — its value is
        flowing into a long-lived container (tablets push, struct-lit
        field bound for push, etc.) which takes over ownership. Walks
        `transfer_on_tail` chains so borrow bindings redirect to their
        true owners.

        Default: zero-inits the source alloca and returns True. With
        `defer_zero=True`, returns the source slot without zeroing
        so the caller can zero AFTER an intermediate op that must
        still see live chunks (e.g. tablets.push allocates a chunk
        before storing the value; zeroing first would strand the
        chunks from GC reachability through the shadow stack).
        Returns False if no cleanup entry was found."""
        if not self._cleanup_frames:
            return False
        try:
            var = self._lookup(name)
        except CodegenError:
            return False
        entry_name = (
            var.transfer_on_tail if var.transfer_on_tail else name
        )
        # Innermost-first: with variable shadowing (inner scope
        # re-binds `x`), `_lookup` returns the inner, so cleanup
        # eviction should also target the inner frame's entry.
        # Matches `_gen_release`'s walk order.
        for frame in reversed(self._cleanup_frames):
            for i, (_fn, _ptr, fname) in enumerate(frame):
                if fname == entry_name:
                    slot = _ptr
                    frame.pop(i)
                    if defer_zero:
                        return slot
                    self._zero_transferred_slot(slot)
                    return True
        return False

    def _zero_transferred_slot(self, slot: ir.Value) -> None:
        """Write a zero-initialized value to a transferred alloca so any
        subsequent release walk (explicit or auto) is a no-op on it.
        Safe because the container now holds the real pointers; the
        source slot's value is semantically "moved-from" and users
        shouldn't rely on reading it back."""
        assert self.builder is not None
        if not isinstance(slot.type, ir.PointerType):
            return
        slot_ty = slot.type.pointee
        self.builder.store(ir.Constant(slot_ty, None), slot)

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

    def _gen_gloss_call(
        self, mangled: str, arg_exprs: list[A.Expr],
    ) -> ir.Value:
        """Emit a call to a gloss-registered fn under its mangled name.
        Reuses the regular fn-call marshaling — str cap=0 neutering,
        struct-field zeroing, anonymous cleanup for heap-owning rvalue
        args — so operator overloads inherit the same ownership rules
        every other Tuppu call has. No marshaling wrapper: the callee
        is just a regular Tuppu fn, dispatched by typechecker lookup
        instead of source-level name."""
        assert self.builder is not None
        fn = self.functions.get(mangled)
        if fn is None:
            raise CodegenError(
                f"gloss dispatch: fn {mangled!r} not declared"
            )
        call_args: list[ir.Value] = []
        for arg_expr, expected_ty in zip(arg_exprs, fn.args):
            v = self._gen_expr(arg_expr)
            if v is None:
                raise CodegenError("gloss arg has no value")
            coerced = self._coerce(v, expected_ty.type)
            if self._is_str_value(expected_ty.type):
                coerced = self._str_as_borrow(coerced)
            elif (
                self._struct_fields_for(expected_ty.type) is not None
                and self._struct_needs_cleanup(expected_ty.type)
            ):
                coerced = self._struct_as_borrow(coerced, expected_ty.type)
            call_args.append(coerced)
        return self.builder.call(fn, call_args)

    def _gen_unary(self, e: A.Unary) -> ir.Value:
        assert self.builder is not None
        # User-defined overload: the checker marked the node with a
        # mangled fn name. Dispatch by emitting a regular call — the
        # fn lives in self.functions under the mangle.
        if (
            self._checker is not None
            and id(e) in self._checker.gloss_call_for_node
        ):
            return self._gen_gloss_call(
                self._checker.gloss_call_for_node[id(e)], [e.operand],
            )
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
        # User-defined overload: checker marked the node with a mangled
        # fn name. Emit a call; for `!=` the checker routes via `eq`
        # and we negate the result here.
        if (
            self._checker is not None
            and id(e) in self._checker.gloss_call_for_node
        ):
            mangled = self._checker.gloss_call_for_node[id(e)]
            result = self._gen_gloss_call(mangled, [e.lhs, e.rhs])
            if e.op == "!=":
                return self.builder.not_(result)
            return result
        lhs = self._gen_expr(e.lhs)
        rhs = self._gen_expr(e.rhs)
        if lhs is None or rhs is None:
            raise CodegenError(f"operand of binary {e.op} has no value")
        op = e.op

        # str + str = concat. Reuses the same single-malloc emitter as
        # the intrinsic, so `s + t` and `s += t` (which the parser
        # desugars to `s = s + t`) both produce one heap allocation
        # per combined op, not a chain.
        if (
            op == "+"
            and self._is_str_value(lhs.type)
            and self._is_str_value(rhs.type)
        ):
            return self._emit_str_concat([(lhs, e.lhs), (rhs, e.rhs)])

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
                    if self._is_ivec_value(var.value_ty):
                        # ivec method dispatch — recover the per-T
                        # IVecInfo from the typecheck-resolved element
                        # type recorded on the call expression.
                        elem_ty = self._ivec_elem_for_call(e)
                        if elem_ty is not None:
                            iv_info = self._get_ivec(elem_ty)
                            return self._gen_ivec_method(
                                iv_info, var, e.callee.name, e.args,
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
                    if self._is_ivec_value(slot_ty):
                        elem_ty = self._ivec_elem_for_call(e)
                        if elem_ty is not None:
                            iv_info = self._get_ivec(elem_ty)
                            inner = Variable(
                                is_mut=True, ir_ref=slot_ptr,
                                value_ty=slot_ty,
                            )
                            return self._gen_ivec_method(
                                iv_info, inner, e.callee.name, e.args,
                            )

            # Struct field holding a fn-value: `obj.run(x)` loads the
            # field (a function pointer) and calls through it. Not a
            # method dispatch — the callee has no implicit receiver.
            field_val = self._gen_expr(e.callee)
            if (
                field_val is not None
                and isinstance(field_val.type, ir.PointerType)
                and isinstance(field_val.type.pointee, ir.FunctionType)
            ):
                return self._gen_fn_value_call(
                    field_val, field_val.type.pointee, e.args,
                )

        if not isinstance(e.callee, A.Ident):
            raise CodegenError("only direct function calls are supported")
        name = e.callee.name

        # Indirect call through a fn-valued local binding —
        # `step f = foo; f(x)`. Local bindings shadow global fns by
        # design, so check scopes first and dispatch indirectly when
        # the binding is a fn pointer.
        try:
            local_var = self._lookup(name)
        except CodegenError:
            local_var = None
        if local_var is not None:
            vty = local_var.value_ty
            if isinstance(vty, ir.PointerType) and isinstance(
                vty.pointee, ir.FunctionType
            ):
                fn_ptr = self._gen_expr(e.callee)
                assert fn_ptr is not None
                return self._gen_fn_value_call(fn_ptr, vty.pointee, e.args)

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
        if name == "str_slice":
            return self._gen_str_slice_call(e.args)
        if name == "int_to_str":
            return self._gen_to_str_call(e.args, self._get_int_to_str(), I64)
        if name == "sex_to_str":
            from ._common import SEX as _SEX
            return self._gen_to_str_call(e.args, self._get_sex_to_str(), _SEX)
        if name == "bytes_to_str":
            return self._gen_bytes_to_str_call(e.args)
        if name == "buffer_to_str":
            return self._gen_buffer_to_str_call(e.args)

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
            # Variadic tablets param receives a pointer; the arg is the
            # synthesised literal from the checker. Build it in the
            # caller's frame and pass the alloca pointer directly so
            # the callee sees the real chunks. Cleanup registration
            # happens inside `_gen_tablets_lit_addr`.
            if (
                isinstance(expected_ty, ir.PointerType)
                and self._tablets_info_for(expected_ty.pointee) is not None
                and isinstance(arg, A.TabletsLit)
            ):
                info = self._tablets_info_for(expected_ty.pointee)
                call_args.append(
                    self._gen_tablets_lit_addr(arg, elem_ty_hint=info.elem_ty),
                )
                continue
            # Mut struct param: expects a pointer to the caller's
            # struct. Distinguished from a wedge handle (also Struct*
            # at the LLVM level) by the per-param mut-ness sideband.
            # The arg must be a mut-bound Ident so we can hand over
            # the alloca address; literals and step bindings have no
            # stable address a callee could mutate through. The
            # callee doesn't register cleanup — caller retains sole
            # ownership.
            is_mut_struct_param = (
                isinstance(expected_ty, ir.PointerType)
                and self._struct_fields_for(expected_ty.pointee) is not None
                and self._fn_param_mut.get(fn.name) is not None
                and i < len(self._fn_param_mut[fn.name])
                and self._fn_param_mut[fn.name][i]
            )
            if is_mut_struct_param:
                if not isinstance(arg, A.Ident):
                    raise CodegenError(
                        f"argument {i} of {name!r}: mut struct param "
                        f"needs a mut-bound Ident (pass by reference); "
                        f"got {type(arg).__name__}"
                    )
                var = self._lookup(arg.name)
                if not var.is_mut or var.value_ty != expected_ty.pointee:
                    raise CodegenError(
                        f"argument {i} of {name!r}: mut struct param "
                        f"{arg.name!r} must be a mut binding of type "
                        f"{expected_ty.pointee}, got "
                        f"{'mut ' if var.is_mut else 'step '}{var.value_ty}"
                    )
                call_args.append(var.ir_ref)
                continue
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
            #
            # Struct neutering only fires for MUT struct params: those
            # are the only ones that alloca + register cleanup in the
            # callee. Non-mut struct params read the SSA value as-is,
            # no cleanup frame entry — so neutering would pointlessly
            # zero out tablets fields the callee wants to read.
            param_is_mut = False
            param_mut_list = self._fn_param_mut.get(fn.name)
            if param_mut_list is not None and i < len(param_mut_list):
                param_is_mut = param_mut_list[i]
            if self._is_str_value(expected_ty):
                coerced = self._str_as_borrow(coerced)
            elif (
                self._struct_fields_for(expected_ty) is not None
                and self._struct_needs_cleanup(expected_ty)
            ):
                if param_is_mut:
                    coerced = self._struct_as_borrow(coerced, expected_ty)
            call_args.append(coerced)
        return self.builder.call(fn, call_args)
