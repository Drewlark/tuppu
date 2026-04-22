"""Rat arithmetic codegen: `_gen_rat_binary` (arithmetic and
comparison lowering) and `__tuppu_rat_reduce` (gcd-normalizing
helper). Methods assume the containing class has the usual Codegen
attributes (`self.builder`, `self.module`, `self._rat_reduce`)."""
from __future__ import annotations

from llvmlite import ir

from ._common import CodegenError, I64, RAT


class RatMixin:
    def _gen_rat_binary(self, op: str, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        assert self.builder is not None
        b = self.builder
        a_num = b.extract_value(lhs, 0)
        a_den = b.extract_value(lhs, 1)
        b_num = b.extract_value(rhs, 0)
        b_den = b.extract_value(rhs, 1)

        if op in ("+", "-"):
            # (a/p ± b/q) = (a*q ± b*p) / (p*q)
            left  = b.mul(a_num, b_den)
            right = b.mul(b_num, a_den)
            num = b.add(left, right) if op == "+" else b.sub(left, right)
            den = b.mul(a_den, b_den)
            return b.call(self._get_rat_reduce(), [num, den])
        if op == "*":
            return b.call(self._get_rat_reduce(),
                          [b.mul(a_num, b_num), b.mul(a_den, b_den)])
        if op == "/":
            # a/p ÷ b/q = (a*q) / (p*b) — reduce handles sign and zero-trap.
            return b.call(self._get_rat_reduce(),
                          [b.mul(a_num, b_den), b.mul(a_den, b_num)])
        if op in ("==", "!="):
            # Since rats are always reduced (gcd=1, den>0), equal iff fields match.
            num_eq = b.icmp_signed("==", a_num, b_num)
            den_eq = b.icmp_signed("==", a_den, b_den)
            eq = b.and_(num_eq, den_eq)
            return eq if op == "==" else b.not_(eq)
        if op in ("<", "<=", ">", ">="):
            # With den>0 on both sides: a/p < b/q  <=>  a*q < b*p.
            left  = b.mul(a_num, b_den)
            right = b.mul(b_num, a_den)
            return b.icmp_signed(op, left, right)
        raise CodegenError(f"rat does not support operator {op}")

    # --- tables (comptime lookup) ---------------------------------------

    def _get_rat_reduce(self) -> ir.Function:
        """Emit (once per module) __tuppu_rat_reduce(i64 num, i64 den) -> rat.
        Traps on den == 0. Normalizes so den > 0, then divides both fields by
        gcd(|num|, den) using Euclidean iteration."""
        if self._rat_reduce is not None:
            return self._rat_reduce

        fn = ir.Function(
            self.module,
            ir.FunctionType(RAT, [I64, I64]),
            name="__tuppu_rat_reduce",
        )
        fn.args[0].name = "num"
        fn.args[1].name = "den"
        num_arg, den_arg = fn.args[0], fn.args[1]

        entry      = fn.append_basic_block("entry")
        trap_bb    = fn.append_basic_block("trap")
        normalize  = fn.append_basic_block("normalize")
        gcd_loop   = fn.append_basic_block("gcd.loop")
        gcd_body   = fn.append_basic_block("gcd.body")
        gcd_done   = fn.append_basic_block("gcd.done")

        b = ir.IRBuilder(entry)
        is_den_zero = b.icmp_signed("==", den_arg, ir.Constant(I64, 0))
        b.cbranch(is_den_zero, trap_bb, normalize)

        b.position_at_end(trap_bb)
        b.call(self._get_trap(), [])
        b.unreachable()

        # Normalize: if den < 0, flip both signs. Then gcd on (|num|, den>0).
        b.position_at_end(normalize)
        den_neg = b.icmp_signed("<", den_arg, ir.Constant(I64, 0))
        num_norm = b.select(den_neg, b.neg(num_arg), num_arg)
        den_norm = b.select(den_neg, b.neg(den_arg), den_arg)
        num_neg = b.icmp_signed("<", num_norm, ir.Constant(I64, 0))
        num_abs = b.select(num_neg, b.neg(num_norm), num_norm)
        b.branch(gcd_loop)

        # gcd loop: while b != 0: a, b = b, a % b.  Final a is gcd.
        b.position_at_end(gcd_loop)
        a_phi = b.phi(I64, "gcd.a")
        bb_phi = b.phi(I64, "gcd.b")
        a_phi.add_incoming(num_abs, normalize)
        bb_phi.add_incoming(den_norm, normalize)
        b_is_zero = b.icmp_signed("==", bb_phi, ir.Constant(I64, 0))
        b.cbranch(b_is_zero, gcd_done, gcd_body)

        b.position_at_end(gcd_body)
        rem = b.srem(a_phi, bb_phi)
        a_phi.add_incoming(bb_phi, gcd_body)
        bb_phi.add_incoming(rem, gcd_body)
        b.branch(gcd_loop)

        # Divide through by gcd, build the struct, return.
        b.position_at_end(gcd_done)
        # gcd > 0 because den > 0. If num == 0, gcd = den, and 0/den = 0. Safe.
        result_num = b.sdiv(num_norm, a_phi)
        result_den = b.sdiv(den_norm, a_phi)
        undef = ir.Constant(RAT, ir.Undefined)
        with_n = b.insert_value(undef, result_num, 0)
        final  = b.insert_value(with_n, result_den, 1)
        b.ret(final)

        self._rat_reduce = fn
        return fn


