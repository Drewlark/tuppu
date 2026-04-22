"""Sex/dish codegen: literal lowering and all runtime helpers
(`__tuppu_sex_print`, `__tuppu_sex_to_rat`, `__tuppu_int_to_sex`,
`__tuppu_rat_to_sex`, `__tuppu_sex_add`). Methods assume the
containing class has the usual Codegen attributes — the caches
`self._sex_print`, `self._sex_to_rat`, `self._int_to_sex`,
`self._rat_to_sex`, `self._sex_add`, and the builder/module/rat
helpers they call through `self`."""
from __future__ import annotations

from llvmlite import ir

from ._common import (
    CodegenError, I1, I8, I16, I32, I64,
    RAT, SEX, SEX_MAX_DIGITS,
    SEX_IDX_DIGITS, SEX_IDX_RADIX, SEX_IDX_COUNT, SEX_IDX_SIGN,
)


class SexMixin:
    def _gen_sex_lit(self, e: A.SexLit) -> ir.Value:
        """Lower a sex literal to a digit-form constant. The lexer has
        already validated each digit is in [0, 60)."""
        int_digits = e.int_digits
        frac_digits = e.frac_digits if e.frac_digits is not None else []
        all_digits = int_digits + frac_digits
        if len(all_digits) > SEX_MAX_DIGITS:
            raise CodegenError(
                f"sex literal has {len(all_digits)} digits; max is "
                f"{SEX_MAX_DIGITS}"
            )
        # Pad to fixed width so every sex value has identical layout.
        padded = all_digits + [0] * (SEX_MAX_DIGITS - len(all_digits))
        digit_arr = ir.Constant(
            ir.ArrayType(I8, SEX_MAX_DIGITS),
            padded,
        )
        radix = len(int_digits)
        count = len(all_digits)
        return ir.Constant(SEX, (
            digit_arr,
            ir.Constant(I8, radix),
            ir.Constant(I8, count),
            ir.Constant(I8, 0),   # positive by construction; unary - flips
            ir.Constant(I8, 0),   # pad
        ))

    def _get_sex_print(self) -> ir.Function:
        """Emit (once) `__tuppu_sex_print(sex, newline: i1)` — prints a sex
        value in Babylonian notation: integer digits space-separated,
        a semicolon before the fractional digits (if any), then fractional
        digits space-separated. Negative sign printed as a leading `-`."""
        if self._sex_print is not None:
            return self._sex_print

        fn = ir.Function(
            self.module,
            ir.FunctionType(ir.VoidType(), [SEX, I1]),
            name="__tuppu_sex_print",
        )
        fn.args[0].name = "sx"
        fn.args[1].name = "newline"
        sx, want_nl = fn.args

        entry     = fn.append_basic_block("entry")
        neg_bb    = fn.append_basic_block("print.neg")
        int_loop  = fn.append_basic_block("int.loop")
        int_body  = fn.append_basic_block("int.body")
        int_next  = fn.append_basic_block("int.next")
        radix_bb  = fn.append_basic_block("radix")
        has_frac  = fn.append_basic_block("print.semi")
        frac_loop = fn.append_basic_block("frac.loop")
        frac_body = fn.append_basic_block("frac.body")
        frac_next = fn.append_basic_block("frac.next")
        maybe_nl  = fn.append_basic_block("maybe.nl")
        do_nl     = fn.append_basic_block("do.nl")
        done      = fn.append_basic_block("done")

        b = ir.IRBuilder(entry)
        # _str_ptr emits GEPs into self.builder — temporarily point it at
        # our local builder so the format constants live in this function.
        saved_builder = self.builder
        self.builder = b
        try:
            fmt_dash  = self._str_ptr(b"-")
            fmt_sp    = self._str_ptr(b" ")
            fmt_semi  = self._str_ptr(b";")
            fmt_nl    = self._str_ptr(b"\n")
            fmt_digit = self._str_ptr(b"%d")
        finally:
            self.builder = saved_builder
        slot = b.alloca(SEX, name="sex.slot")
        b.store(sx, slot)
        digits_addr = b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_DIGITS)],
            inbounds=True,
        )
        radix = b.sext(b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_RADIX)],
            inbounds=True,
        )), I64)
        count = b.sext(b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_COUNT)],
            inbounds=True,
        )), I64)
        sign = b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_SIGN)],
            inbounds=True,
        ))
        is_neg = b.icmp_signed("!=", sign, ir.Constant(I8, 0))
        b.cbranch(is_neg, neg_bb, int_loop)

        b.position_at_end(neg_bb)
        b.call(self.printf, [fmt_dash])
        b.branch(int_loop)

        # Integer-digit loop: print each digit, space-separated.
        b.position_at_end(int_loop)
        i_phi = b.phi(I64, "i")
        i_phi.add_incoming(ir.Constant(I64, 0), neg_bb)
        i_phi.add_incoming(ir.Constant(I64, 0), entry)
        done_int = b.icmp_signed(">=", i_phi, radix)
        b.cbranch(done_int, radix_bb, int_body)

        b.position_at_end(int_body)
        need_space = b.icmp_signed(">", i_phi, ir.Constant(I64, 0))
        with b.if_then(need_space):
            b.call(self.printf, [fmt_sp])
        dptr = b.gep(digits_addr, [ir.Constant(I32, 0), i_phi], inbounds=True)
        dval = b.zext(b.load(dptr), I32)
        b.call(self.printf, [fmt_digit, dval])
        b.branch(int_next)

        b.position_at_end(int_next)
        next_i = b.add(i_phi, ir.Constant(I64, 1))
        i_phi.add_incoming(next_i, int_next)
        b.branch(int_loop)

        # After integer digits, maybe print ';' and fractional digits.
        b.position_at_end(radix_bb)
        fractional = b.icmp_signed(">", count, radix)
        b.cbranch(fractional, has_frac, maybe_nl)

        b.position_at_end(has_frac)
        b.call(self.printf, [fmt_semi])
        b.branch(frac_loop)

        b.position_at_end(frac_loop)
        j_phi = b.phi(I64, "j")
        j_phi.add_incoming(radix, has_frac)
        done_frac = b.icmp_signed(">=", j_phi, count)
        b.cbranch(done_frac, maybe_nl, frac_body)

        b.position_at_end(frac_body)
        # First fractional digit immediately follows `;` with no space.
        need_sp2 = b.icmp_signed(">", j_phi, radix)
        with b.if_then(need_sp2):
            b.call(self.printf, [fmt_sp])
        fptr = b.gep(digits_addr, [ir.Constant(I32, 0), j_phi], inbounds=True)
        fval = b.zext(b.load(fptr), I32)
        b.call(self.printf, [fmt_digit, fval])
        b.branch(frac_next)

        b.position_at_end(frac_next)
        next_j = b.add(j_phi, ir.Constant(I64, 1))
        j_phi.add_incoming(next_j, frac_next)
        b.branch(frac_loop)

        b.position_at_end(maybe_nl)
        b.cbranch(want_nl, do_nl, done)

        b.position_at_end(do_nl)
        b.call(self.printf, [fmt_nl])
        b.branch(done)

        b.position_at_end(done)
        b.ret_void()

        self._sex_print = fn
        return fn

    def _emit_sex_print(self, val: ir.Value, *, newline: bool) -> None:
        assert self.builder is not None
        nl_flag = ir.Constant(I1, 1 if newline else 0)
        self.builder.call(self._get_sex_print(), [val, nl_flag])

    def _get_sex_to_rat(self) -> ir.Function:
        """Emit (once per module) `__tuppu_sex_to_rat(sex) -> rat`.

        Reconstructs an integer numerator by Horner-style evaluation over
        the digit sequence (each digit × 60^place), computes the implied
        denominator from (count - radix) fractional places, applies the
        sign bit, then delegates to `__tuppu_rat_reduce` for gcd reduction.
        The result is a normal rat value — all invariants preserved."""
        if self._sex_to_rat is not None:
            return self._sex_to_rat

        fn = ir.Function(
            self.module,
            ir.FunctionType(RAT, [SEX]),
            name="__tuppu_sex_to_rat",
        )
        fn.args[0].name = "sx"
        sx = fn.args[0]

        entry = fn.append_basic_block("entry")
        num_loop = fn.append_basic_block("num.loop")
        num_body = fn.append_basic_block("num.body")
        den_loop = fn.append_basic_block("den.loop")
        den_body = fn.append_basic_block("den.body")
        apply_sign = fn.append_basic_block("apply.sign")
        do_reduce = fn.append_basic_block("reduce")

        b = ir.IRBuilder(entry)

        # Spill the sex value to a stack slot so we can GEP into the digit
        # array by a runtime index. (LLVM can't index a struct field by a
        # non-constant, but it can GEP into an alloca.)
        slot = b.alloca(SEX, name="sex.slot")
        b.store(sx, slot)
        digits_addr = b.gep(
            slot,
            [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_DIGITS)],
            inbounds=True,
        )
        radix = b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_RADIX)],
            inbounds=True,
        ))
        count = b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_COUNT)],
            inbounds=True,
        ))
        sign = b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_SIGN)],
            inbounds=True,
        ))
        count_i64 = b.sext(count, I64)
        radix_i64 = b.sext(radix, I64)
        frac_places = b.sub(count_i64, radix_i64)
        b.branch(num_loop)

        # num = sum(digits[i] * 60^(count-1-i)) computed Horner-style:
        #   num = 0; for i in 0..count: num = num*60 + digits[i]
        b.position_at_end(num_loop)
        num_phi = b.phi(I64, "num")
        i_phi = b.phi(I64, "i")
        num_phi.add_incoming(ir.Constant(I64, 0), entry)
        i_phi.add_incoming(ir.Constant(I64, 0), entry)
        done_num = b.icmp_signed(">=", i_phi, count_i64)
        b.cbranch(done_num, den_loop, num_body)

        b.position_at_end(num_body)
        digit_ptr = b.gep(
            digits_addr,
            [ir.Constant(I32, 0), i_phi],
            inbounds=True,
        )
        digit_byte = b.load(digit_ptr)
        digit = b.zext(digit_byte, I64)
        scaled = b.mul(num_phi, ir.Constant(I64, 60))
        next_num = b.add(scaled, digit)
        next_i = b.add(i_phi, ir.Constant(I64, 1))
        num_phi.add_incoming(next_num, num_body)
        i_phi.add_incoming(next_i, num_body)
        b.branch(num_loop)

        # den = 60^frac_places
        b.position_at_end(den_loop)
        den_phi = b.phi(I64, "den")
        k_phi = b.phi(I64, "k")
        den_phi.add_incoming(ir.Constant(I64, 1), num_loop)
        k_phi.add_incoming(ir.Constant(I64, 0), num_loop)
        done_den = b.icmp_signed(">=", k_phi, frac_places)
        b.cbranch(done_den, apply_sign, den_body)

        b.position_at_end(den_body)
        next_den = b.mul(den_phi, ir.Constant(I64, 60))
        next_k = b.add(k_phi, ir.Constant(I64, 1))
        den_phi.add_incoming(next_den, den_body)
        k_phi.add_incoming(next_k, den_body)
        b.branch(den_loop)

        # Apply sign (any nonzero sign byte means negative).
        b.position_at_end(apply_sign)
        is_neg = b.icmp_signed("!=", sign, ir.Constant(I8, 0))
        signed_num = b.select(is_neg, b.neg(num_phi), num_phi)
        b.branch(do_reduce)

        b.position_at_end(do_reduce)
        result = b.call(self._get_rat_reduce(), [signed_num, den_phi])
        b.ret(result)

        self._sex_to_rat = fn
        return fn

    def _get_int_to_sex(self) -> ir.Function:
        """Emit `__tuppu_int_to_sex(i64) -> sex` — decompose a 64-bit
        integer into its sexagesimal digit sequence. Result is always
        in integer form (no fractional digits). i64 max fits in 11
        base-60 digits, well under SEX_MAX_DIGITS."""
        if self._int_to_sex is not None:
            return self._int_to_sex

        fn = ir.Function(
            self.module,
            ir.FunctionType(SEX, [I64]),
            name="__tuppu_int_to_sex",
        )
        fn.args[0].name = "n"
        n_arg = fn.args[0]

        arr16_ty = ir.ArrayType(I8, SEX_MAX_DIGITS)
        vec16 = ir.VectorType(I8, SEX_MAX_DIGITS)
        ZERO_I32 = ir.Constant(I32, 0)

        entry        = fn.append_basic_block("entry")
        is_zero_bb   = fn.append_basic_block("is.zero")
        neg_bb       = fn.append_basic_block("negate")
        decomp_hdr   = fn.append_basic_block("decomp.hdr")
        decomp_body  = fn.append_basic_block("decomp.body")
        copy_hdr     = fn.append_basic_block("copy.hdr")
        copy_body    = fn.append_basic_block("copy.body")
        pack         = fn.append_basic_block("pack")

        b = ir.IRBuilder(entry)
        tmp = b.alloca(arr16_ty)
        tmp.align = SEX_MAX_DIGITS
        out_buf = b.alloca(arr16_ty)
        out_buf.align = SEX_MAX_DIGITS
        zero_vec = ir.Constant(vec16, [0] * SEX_MAX_DIGITS)
        for buf in (tmp, out_buf):
            st = b.store(zero_vec, b.bitcast(buf, vec16.as_pointer()))
            st.align = SEX_MAX_DIGITS

        n_is_zero = b.icmp_signed("==", n_arg, ir.Constant(I64, 0))
        b.cbranch(n_is_zero, is_zero_bb, neg_bb)

        # Zero path — single digit 0, integer form, sign 0.
        b.position_at_end(is_zero_bb)
        zero_result: ir.Value = ir.Constant(SEX, ir.Undefined)
        zero_result = b.insert_value(
            zero_result,
            ir.Constant(arr16_ty, [0] * SEX_MAX_DIGITS),
            SEX_IDX_DIGITS,
        )
        zero_result = b.insert_value(zero_result, ir.Constant(I8, 1), SEX_IDX_RADIX)
        zero_result = b.insert_value(zero_result, ir.Constant(I8, 1), SEX_IDX_COUNT)
        zero_result = b.insert_value(zero_result, ir.Constant(I8, 0), SEX_IDX_SIGN)
        zero_result = b.insert_value(zero_result, ir.Constant(I8, 0), 4)
        b.ret(zero_result)

        # Negate if needed. Sign byte = 1 iff the input was negative.
        b.position_at_end(neg_bb)
        is_neg = b.icmp_signed("<", n_arg, ir.Constant(I64, 0))
        abs_n = b.select(is_neg, b.neg(n_arg), n_arg)
        sign_byte = b.select(is_neg, ir.Constant(I8, 1), ir.Constant(I8, 0))
        b.branch(decomp_hdr)

        # Decompose into base-60 digits, MSB first in `tmp[]`, written
        # from the right (index = start_idx..15) so we know how many
        # digits we used.
        b.position_at_end(decomp_hdr)
        n_phi = b.phi(I64, "n")
        idx_phi = b.phi(I32, "idx")
        n_phi.add_incoming(abs_n, neg_bb)
        idx_phi.add_incoming(ir.Constant(I32, SEX_MAX_DIGITS - 1), neg_bb)
        done = b.icmp_signed("==", n_phi, ir.Constant(I64, 0))
        b.cbranch(done, copy_hdr, decomp_body)

        b.position_at_end(decomp_body)
        digit = b.trunc(b.srem(n_phi, ir.Constant(I64, 60)), I8)
        next_n = b.sdiv(n_phi, ir.Constant(I64, 60))
        b.store(digit, b.gep(
            tmp, [ZERO_I32, b.sext(idx_phi, I64)], inbounds=True,
        ))
        n_phi.add_incoming(next_n, decomp_body)
        idx_phi.add_incoming(b.sub(idx_phi, ir.Constant(I32, 1)), decomp_body)
        b.branch(decomp_hdr)

        # Copy tmp[start_idx+1 .. 16) to out_buf[0 .. digit_count).
        b.position_at_end(copy_hdr)
        start_idx = b.add(idx_phi, ir.Constant(I32, 1))
        digit_count = b.sub(ir.Constant(I32, SEX_MAX_DIGITS), start_idx)
        b.branch(copy_body)

        copy_body_hdr = fn.append_basic_block("copy.body.hdr")
        copy_body_step = fn.append_basic_block("copy.body.step")
        b.position_at_end(copy_body)
        b.branch(copy_body_hdr)

        b.position_at_end(copy_body_hdr)
        j_phi = b.phi(I32, "j")
        j_phi.add_incoming(ZERO_I32, copy_body)
        j_done = b.icmp_signed(">=", j_phi, digit_count)
        b.cbranch(j_done, pack, copy_body_step)

        b.position_at_end(copy_body_step)
        src_idx = b.add(start_idx, j_phi)
        src_byte = b.load(b.gep(
            tmp, [ZERO_I32, b.sext(src_idx, I64)], inbounds=True,
        ))
        b.store(src_byte, b.gep(
            out_buf, [ZERO_I32, b.sext(j_phi, I64)], inbounds=True,
        ))
        j_phi.add_incoming(b.add(j_phi, ir.Constant(I32, 1)), copy_body_step)
        b.branch(copy_body_hdr)

        b.position_at_end(pack)
        final_digits = b.load(out_buf)
        result: ir.Value = ir.Constant(SEX, ir.Undefined)
        result = b.insert_value(result, final_digits, SEX_IDX_DIGITS)
        result = b.insert_value(result, b.trunc(digit_count, I8), SEX_IDX_RADIX)
        result = b.insert_value(result, b.trunc(digit_count, I8), SEX_IDX_COUNT)
        result = b.insert_value(result, sign_byte, SEX_IDX_SIGN)
        result = b.insert_value(result, ir.Constant(I8, 0), 4)
        b.ret(result)

        self._int_to_sex = fn
        return fn

    def _get_rat_to_sex(self) -> ir.Function:
        """Emit `__tuppu_rat_to_sex(rat) -> sex` — convert a reduced rat
        to its Babylonian digit form.

        Regularity check: the denominator must factor as 2^a·3^b·5^c
        (a "regular number" in Old Babylonian terms). Non-regular rats
        have no terminating sexagesimal representation — we trap.

        Algorithm:
          1. If num == 0, return zero-sex.
          2. Sign = (num < 0); work with |num|.
          3. Regularity check: strip 2s, 3s, 5s from den; trap if not
             reduced to 1.
          4. Decompose |num|/den into integer digits (base-60, MSB-first)
             via repeated (%60, /60).
          5. Extract fractional digits via iterated (rem*60)/den, until
             rem == 0 (regular: guaranteed) or SEX_MAX_DIGITS hit (trap).
          6. Pack.
        """
        if self._rat_to_sex is not None:
            return self._rat_to_sex

        fn = ir.Function(
            self.module,
            ir.FunctionType(SEX, [RAT]),
            name="__tuppu_rat_to_sex",
        )
        fn.args[0].name = "r"
        r_arg = fn.args[0]

        arr16_ty = ir.ArrayType(I8, SEX_MAX_DIGITS)
        vec16 = ir.VectorType(I8, SEX_MAX_DIGITS)
        ZERO_I32 = ir.Constant(I32, 0)

        entry          = fn.append_basic_block("entry")
        zero_path      = fn.append_basic_block("zero.path")
        sign_bb        = fn.append_basic_block("sign")
        reg_hdr        = fn.append_basic_block("reg.hdr")
        reg_try2       = fn.append_basic_block("reg.try2")
        reg_div2       = fn.append_basic_block("reg.div2")
        reg_try3       = fn.append_basic_block("reg.try3")
        reg_div3       = fn.append_basic_block("reg.div3")
        reg_try5       = fn.append_basic_block("reg.try5")
        reg_div5       = fn.append_basic_block("reg.div5")
        reg_trap       = fn.append_basic_block("reg.trap")
        reg_ok         = fn.append_basic_block("reg.ok")
        int_hdr        = fn.append_basic_block("int.hdr")
        int_body       = fn.append_basic_block("int.body")
        int_copy_prep  = fn.append_basic_block("int.copy.prep")
        int_copy_hdr   = fn.append_basic_block("int.copy.hdr")
        int_copy_body  = fn.append_basic_block("int.copy.body")
        int_force_zero = fn.append_basic_block("int.force.zero")
        int_done       = fn.append_basic_block("int.done")
        frac_hdr       = fn.append_basic_block("frac.hdr")
        frac_ov_trap   = fn.append_basic_block("frac.ov.trap")
        frac_body      = fn.append_basic_block("frac.body")
        pack           = fn.append_basic_block("pack")

        b = ir.IRBuilder(entry)
        num = b.extract_value(r_arg, 0)
        den = b.extract_value(r_arg, 1)
        is_zero = b.icmp_signed("==", num, ir.Constant(I64, 0))
        b.cbranch(is_zero, zero_path, sign_bb)

        # Zero path — single int digit 0, sign 0.
        b.position_at_end(zero_path)
        zero_result: ir.Value = ir.Constant(SEX, ir.Undefined)
        zero_result = b.insert_value(
            zero_result,
            ir.Constant(arr16_ty, [0] * SEX_MAX_DIGITS),
            SEX_IDX_DIGITS,
        )
        zero_result = b.insert_value(zero_result, ir.Constant(I8, 1), SEX_IDX_RADIX)
        zero_result = b.insert_value(zero_result, ir.Constant(I8, 1), SEX_IDX_COUNT)
        zero_result = b.insert_value(zero_result, ir.Constant(I8, 0), SEX_IDX_SIGN)
        zero_result = b.insert_value(zero_result, ir.Constant(I8, 0), 4)
        b.ret(zero_result)

        # Sign extraction.
        b.position_at_end(sign_bb)
        is_neg = b.icmp_signed("<", num, ir.Constant(I64, 0))
        abs_num = b.select(is_neg, b.neg(num), num)
        sign_byte = b.select(is_neg, ir.Constant(I8, 1), ir.Constant(I8, 0))
        b.branch(reg_hdr)

        # Regularity check: strip factors of 2, 3, 5 until d == 1 (ok) or
        # none divide (trap).
        b.position_at_end(reg_hdr)
        d_phi = b.phi(I64, "d")
        d_phi.add_incoming(den, sign_bb)
        d_eq_1 = b.icmp_signed("==", d_phi, ir.Constant(I64, 1))
        b.cbranch(d_eq_1, reg_ok, reg_try2)

        b.position_at_end(reg_try2)
        r2 = b.srem(d_phi, ir.Constant(I64, 2))
        r2_zero = b.icmp_signed("==", r2, ir.Constant(I64, 0))
        b.cbranch(r2_zero, reg_div2, reg_try3)
        b.position_at_end(reg_div2)
        d_next_2 = b.sdiv(d_phi, ir.Constant(I64, 2))
        d_phi.add_incoming(d_next_2, reg_div2)
        b.branch(reg_hdr)

        b.position_at_end(reg_try3)
        r3 = b.srem(d_phi, ir.Constant(I64, 3))
        r3_zero = b.icmp_signed("==", r3, ir.Constant(I64, 0))
        b.cbranch(r3_zero, reg_div3, reg_try5)
        b.position_at_end(reg_div3)
        d_next_3 = b.sdiv(d_phi, ir.Constant(I64, 3))
        d_phi.add_incoming(d_next_3, reg_div3)
        b.branch(reg_hdr)

        b.position_at_end(reg_try5)
        r5 = b.srem(d_phi, ir.Constant(I64, 5))
        r5_zero = b.icmp_signed("==", r5, ir.Constant(I64, 0))
        b.cbranch(r5_zero, reg_div5, reg_trap)
        b.position_at_end(reg_div5)
        d_next_5 = b.sdiv(d_phi, ir.Constant(I64, 5))
        d_phi.add_incoming(d_next_5, reg_div5)
        b.branch(reg_hdr)

        b.position_at_end(reg_trap)
        b.call(self._get_trap(), [])
        b.unreachable()

        # Regularity established. Separate integer quotient and remainder.
        b.position_at_end(reg_ok)
        int_quot = b.sdiv(abs_num, den)
        frac_rem0 = b.srem(abs_num, den)
        tmp = b.alloca(arr16_ty)
        tmp.align = SEX_MAX_DIGITS
        out_buf = b.alloca(arr16_ty)
        out_buf.align = SEX_MAX_DIGITS
        zero_vec = ir.Constant(vec16, [0] * SEX_MAX_DIGITS)
        for buf in (tmp, out_buf):
            st = b.store(zero_vec, b.bitcast(buf, vec16.as_pointer()))
            st.align = SEX_MAX_DIGITS
        b.branch(int_hdr)

        # Int decomposition: write digits MSB-first into tmp[15..start_idx]
        # by walking right-to-left from index 15. Same shape as
        # __tuppu_int_to_sex.
        b.position_at_end(int_hdr)
        n_phi = b.phi(I64, "n")
        idx_phi = b.phi(I32, "idx")
        n_phi.add_incoming(int_quot, reg_ok)
        idx_phi.add_incoming(ir.Constant(I32, SEX_MAX_DIGITS - 1), reg_ok)
        n_zero = b.icmp_signed("==", n_phi, ir.Constant(I64, 0))
        b.cbranch(n_zero, int_copy_prep, int_body)

        b.position_at_end(int_body)
        digit_i = b.trunc(b.srem(n_phi, ir.Constant(I64, 60)), I8)
        next_n = b.sdiv(n_phi, ir.Constant(I64, 60))
        b.store(digit_i, b.gep(
            tmp, [ZERO_I32, b.sext(idx_phi, I64)], inbounds=True,
        ))
        n_phi.add_incoming(next_n, int_body)
        idx_phi.add_incoming(b.sub(idx_phi, ir.Constant(I32, 1)), int_body)
        b.branch(int_hdr)

        # Copy tmp[start_idx..16) left-aligned into out_buf[0..int_count).
        # If int_quot was 0, int_count will be 0 — force a single 0 digit.
        b.position_at_end(int_copy_prep)
        start_idx = b.add(idx_phi, ir.Constant(I32, 1))
        int_digits_count = b.sub(ir.Constant(I32, SEX_MAX_DIGITS), start_idx)
        has_int = b.icmp_signed(">", int_digits_count, ZERO_I32)
        b.cbranch(has_int, int_copy_hdr, int_force_zero)

        b.position_at_end(int_force_zero)
        b.store(ir.Constant(I8, 0), b.gep(
            out_buf, [ZERO_I32, ir.Constant(I64, 0)], inbounds=True,
        ))
        b.branch(int_done)

        b.position_at_end(int_copy_hdr)
        j_phi = b.phi(I32, "j")
        j_phi.add_incoming(ZERO_I32, int_copy_prep)
        j_done = b.icmp_signed(">=", j_phi, int_digits_count)
        b.cbranch(j_done, int_done, int_copy_body)

        b.position_at_end(int_copy_body)
        src_idx = b.add(start_idx, j_phi)
        src_byte = b.load(b.gep(
            tmp, [ZERO_I32, b.sext(src_idx, I64)], inbounds=True,
        ))
        b.store(src_byte, b.gep(
            out_buf, [ZERO_I32, b.sext(j_phi, I64)], inbounds=True,
        ))
        j_phi.add_incoming(b.add(j_phi, ir.Constant(I32, 1)), int_copy_body)
        b.branch(int_copy_hdr)

        b.position_at_end(int_done)
        # int_count: 1 if we forced a zero, else int_digits_count.
        int_count = b.phi(I32, "int_count")
        int_count.add_incoming(ir.Constant(I32, 1), int_force_zero)
        int_count.add_incoming(int_digits_count, int_copy_hdr)
        b.branch(frac_hdr)

        # Fractional digits: while rem > 0, write (rem*60)/den to
        # out_buf[write_idx]; rem = (rem*60) % den. Regularity => this
        # terminates. Trap if we'd exceed SEX_MAX_DIGITS anyway (e.g.
        # den = 2^30 needs > 16 frac digits).
        b.position_at_end(frac_hdr)
        rem_phi = b.phi(I64, "rem")
        write_idx = b.phi(I32, "write_idx")
        rem_phi.add_incoming(frac_rem0, int_done)
        write_idx.add_incoming(int_count, int_done)
        rem_is_zero = b.icmp_signed("==", rem_phi, ir.Constant(I64, 0))
        b.cbranch(rem_is_zero, pack, frac_ov_trap)

        b.position_at_end(frac_ov_trap)
        at_cap = b.icmp_signed(">=", write_idx, ir.Constant(I32, SEX_MAX_DIGITS))
        b.cbranch(at_cap, reg_trap, frac_body)

        b.position_at_end(frac_body)
        rem_scaled = b.mul(rem_phi, ir.Constant(I64, 60))
        digit_f = b.trunc(b.sdiv(rem_scaled, den), I8)
        next_rem = b.srem(rem_scaled, den)
        b.store(digit_f, b.gep(
            out_buf, [ZERO_I32, b.sext(write_idx, I64)], inbounds=True,
        ))
        rem_phi.add_incoming(next_rem, frac_body)
        write_idx.add_incoming(b.add(write_idx, ir.Constant(I32, 1)), frac_body)
        b.branch(frac_hdr)

        # Pack.
        b.position_at_end(pack)
        final_digits = b.load(out_buf)
        result: ir.Value = ir.Constant(SEX, ir.Undefined)
        result = b.insert_value(result, final_digits, SEX_IDX_DIGITS)
        result = b.insert_value(result, b.trunc(int_count, I8), SEX_IDX_RADIX)
        result = b.insert_value(result, b.trunc(write_idx, I8), SEX_IDX_COUNT)
        result = b.insert_value(result, sign_byte, SEX_IDX_SIGN)
        result = b.insert_value(result, ir.Constant(I8, 0), 4)
        b.ret(result)

        self._rat_to_sex = fn
        return fn

    def _get_sex_add(self) -> ir.Function:
        """Emit `__tuppu_sex_add(sex, sex) -> sex` — native Babylonian
        digit-form addition.

        Algorithm:
        1. Align operands: compute max_int = max(a.radix, b.radix) and
           max_frac = max(a_frac, b_frac). Write digits into 16-byte
           buffers right-aligned in the int zone and left-aligned in
           the frac zone; everything else stays zero.
        2. On same sign: digit-wise SIMD add, then scalar carry
           propagation. A final carry extends the int zone by one digit.
        3. On different sign: lexicographic magnitude compare, then
           digit-wise sub (borrow propagation) of smaller from larger;
           the result takes the sign of the larger magnitude.
        """
        if self._sex_add is not None:
            return self._sex_add

        fn = ir.Function(
            self.module,
            ir.FunctionType(SEX, [SEX, SEX]),
            name="__tuppu_sex_add",
        )
        fn.args[0].name = "a"
        fn.args[1].name = "b"
        a_arg, b_arg = fn.args

        vec16 = ir.VectorType(I8, SEX_MAX_DIGITS)
        arr16_ty = ir.ArrayType(I8, SEX_MAX_DIGITS)
        ZERO_I32 = ir.Constant(I32, 0)

        # --- declare every basic block up front for clarity -----------------
        entry       = fn.append_basic_block("entry")
        overflow_bb = fn.append_basic_block("overflow")
        align_start = fn.append_basic_block("align.start")
        align_a_hdr = fn.append_basic_block("align.a.hdr")
        align_a_stp = fn.append_basic_block("align.a.step")
        align_b_hdr = fn.append_basic_block("align.b.hdr")
        align_b_stp = fn.append_basic_block("align.b.step")
        signs_check = fn.append_basic_block("signs.check")
        same_sign   = fn.append_basic_block("same.sign")
        add_carry   = fn.append_basic_block("add.carry")
        add_cbody   = fn.append_basic_block("add.carry.body")
        add_fcarry  = fn.append_basic_block("add.final.carry")
        add_shift   = fn.append_basic_block("add.shift")
        add_sbody   = fn.append_basic_block("add.shift.body")
        add_sfinish = fn.append_basic_block("add.shift.finish")
        add_nocarry = fn.append_basic_block("add.no.carry")
        mag_cmp     = fn.append_basic_block("mag.cmp")
        mag_cbody   = fn.append_basic_block("mag.cmp.body")
        mag_equal   = fn.append_basic_block("mag.equal")
        mag_sub     = fn.append_basic_block("mag.sub")
        mag_sbody   = fn.append_basic_block("mag.sub.body")
        mag_send    = fn.append_basic_block("mag.sub.end")
        pack        = fn.append_basic_block("pack")

        b = ir.IRBuilder(entry)

        # --- entry: spill args, zero buffers, compute widths ---------------
        a_slot = b.alloca(SEX)
        b_slot = b.alloca(SEX)
        b.store(a_arg, a_slot)
        b.store(b_arg, b_slot)
        a_buf = b.alloca(arr16_ty); a_buf.align = SEX_MAX_DIGITS
        bbuf  = b.alloca(arr16_ty); bbuf.align  = SEX_MAX_DIGITS
        out_buf = b.alloca(arr16_ty); out_buf.align = SEX_MAX_DIGITS
        zero_vec = ir.Constant(vec16, [0] * SEX_MAX_DIGITS)
        for buf in (a_buf, bbuf, out_buf):
            st = b.store(zero_vec, b.bitcast(buf, vec16.as_pointer()))
            st.align = SEX_MAX_DIGITS

        def load_field(slot, idx):
            return b.load(b.gep(
                slot, [ZERO_I32, ir.Constant(I32, idx)], inbounds=True,
            ))

        a_radix = b.zext(load_field(a_slot, SEX_IDX_RADIX), I32)
        a_count = b.zext(load_field(a_slot, SEX_IDX_COUNT), I32)
        a_sign  = load_field(a_slot, SEX_IDX_SIGN)
        b_radix = b.zext(load_field(b_slot, SEX_IDX_RADIX), I32)
        b_count = b.zext(load_field(b_slot, SEX_IDX_COUNT), I32)
        b_sign  = load_field(b_slot, SEX_IDX_SIGN)
        a_frac  = b.sub(a_count, a_radix)
        b_frac  = b.sub(b_count, b_radix)
        max_int = b.select(
            b.icmp_signed(">", a_radix, b_radix), a_radix, b_radix,
        )
        max_frac = b.select(
            b.icmp_signed(">", a_frac, b_frac), a_frac, b_frac,
        )
        new_count = b.add(max_int, max_frac)
        overflow = b.icmp_signed(
            ">", new_count, ir.Constant(I32, SEX_MAX_DIGITS - 1),
        )
        b.cbranch(overflow, overflow_bb, align_start)

        b.position_at_end(overflow_bb)
        b.call(self._get_trap(), [])
        b.unreachable()

        # --- align.start: prepare offsets and digit-array pointers ---------
        b.position_at_end(align_start)
        a_int_offset = b.sub(max_int, a_radix)
        b_int_offset = b.sub(max_int, b_radix)
        a_digits = b.gep(
            a_slot, [ZERO_I32, ir.Constant(I32, SEX_IDX_DIGITS)], inbounds=True,
        )
        b_digits = b.gep(
            b_slot, [ZERO_I32, ir.Constant(I32, SEX_IDX_DIGITS)], inbounds=True,
        )
        # Pre-compute the first iteration value once, in this block, so the
        # phi-incoming edges below come from dominating instructions.
        zero_i32 = ZERO_I32
        b.branch(align_a_hdr)

        # --- align.a: copy a.count digits into a_buf at aligned positions --
        b.position_at_end(align_a_hdr)
        j_phi = b.phi(I32, "j")
        j_phi.add_incoming(zero_i32, align_start)
        done_a = b.icmp_signed(">=", j_phi, a_count)
        b.cbranch(done_a, align_b_hdr, align_a_stp)

        b.position_at_end(align_a_stp)
        src = b.load(b.gep(
            a_digits, [ZERO_I32, b.sext(j_phi, I64)], inbounds=True,
        ))
        is_int = b.icmp_signed("<", j_phi, a_radix)
        dst_int  = b.add(a_int_offset, j_phi)
        dst_frac = b.add(max_int, b.sub(j_phi, a_radix))
        dst_idx  = b.select(is_int, dst_int, dst_frac)
        b.store(src, b.gep(
            a_buf, [ZERO_I32, b.sext(dst_idx, I64)], inbounds=True,
        ))
        j_next = b.add(j_phi, ir.Constant(I32, 1))
        j_phi.add_incoming(j_next, align_a_stp)
        b.branch(align_a_hdr)

        # --- align.b: same structure, into bbuf ----------------------------
        b.position_at_end(align_b_hdr)
        k_phi = b.phi(I32, "k")
        k_phi.add_incoming(zero_i32, align_a_hdr)
        done_b = b.icmp_signed(">=", k_phi, b_count)
        b.cbranch(done_b, signs_check, align_b_stp)

        b.position_at_end(align_b_stp)
        src_b = b.load(b.gep(
            b_digits, [ZERO_I32, b.sext(k_phi, I64)], inbounds=True,
        ))
        is_int_b = b.icmp_signed("<", k_phi, b_radix)
        dst_int_b  = b.add(b_int_offset, k_phi)
        dst_frac_b = b.add(max_int, b.sub(k_phi, b_radix))
        dst_idx_b  = b.select(is_int_b, dst_int_b, dst_frac_b)
        b.store(src_b, b.gep(
            bbuf, [ZERO_I32, b.sext(dst_idx_b, I64)], inbounds=True,
        ))
        k_next = b.add(k_phi, ir.Constant(I32, 1))
        k_phi.add_incoming(k_next, align_b_stp)
        b.branch(align_b_hdr)

        # --- signs_check: dispatch same-sign vs mixed-sign -----------------
        b.position_at_end(signs_check)
        # Compute starting-i for the loops we're about to launch, so the
        # phi nodes can reference already-dominating values.
        nc_minus_one = b.sub(new_count, ir.Constant(I32, 1))
        same = b.icmp_signed("==", a_sign, b_sign)
        b.cbranch(same, same_sign, mag_cmp)

        # --- same_sign: SIMD raw add + scalar carry propagation ------------
        b.position_at_end(same_sign)
        a_vec = b.load(b.bitcast(a_buf, vec16.as_pointer()))
        a_vec.align = SEX_MAX_DIGITS
        b_vec = b.load(b.bitcast(bbuf, vec16.as_pointer()))
        b_vec.align = SEX_MAX_DIGITS
        raw = b.add(a_vec, b_vec)
        rs = b.store(raw, b.bitcast(out_buf, vec16.as_pointer()))
        rs.align = SEX_MAX_DIGITS
        b.branch(add_carry)

        b.position_at_end(add_carry)
        i_phi = b.phi(I32, "i")
        carry_phi = b.phi(I8, "carry")
        i_phi.add_incoming(nc_minus_one, same_sign)
        carry_phi.add_incoming(ir.Constant(I8, 0), same_sign)
        cont = b.icmp_signed(">=", i_phi, ZERO_I32)
        b.cbranch(cont, add_cbody, add_fcarry)

        b.position_at_end(add_cbody)
        cell_ptr = b.gep(
            out_buf, [ZERO_I32, b.sext(i_phi, I64)], inbounds=True,
        )
        cell = b.load(cell_ptr)
        combined = b.add(cell, carry_phi)
        over = b.icmp_signed(">=", combined, ir.Constant(I8, 60))
        corrected = b.select(over, b.sub(combined, ir.Constant(I8, 60)), combined)
        next_carry = b.select(over, ir.Constant(I8, 1), ir.Constant(I8, 0))
        b.store(corrected, cell_ptr)
        i_phi.add_incoming(b.sub(i_phi, ir.Constant(I32, 1)), add_cbody)
        carry_phi.add_incoming(next_carry, add_cbody)
        b.branch(add_carry)

        b.position_at_end(add_fcarry)
        has_final = b.icmp_signed("!=", carry_phi, ir.Constant(I8, 0))
        b.cbranch(has_final, add_shift, add_nocarry)

        # Shift out_buf right by one, then write 1 at position 0.
        b.position_at_end(add_shift)
        b.branch(add_sbody)

        b.position_at_end(add_sbody)
        s_phi = b.phi(I32, "s")
        s_phi.add_incoming(nc_minus_one, add_shift)
        s_cont = b.icmp_signed(">=", s_phi, ZERO_I32)
        shift_do = fn.append_basic_block("add.shift.do")
        b.cbranch(s_cont, shift_do, add_sfinish)

        b.position_at_end(shift_do)
        s_i64 = b.sext(s_phi, I64)
        src_cell = b.load(b.gep(out_buf, [ZERO_I32, s_i64], inbounds=True))
        s_plus_i64 = b.sext(b.add(s_phi, ir.Constant(I32, 1)), I64)
        b.store(src_cell, b.gep(out_buf, [ZERO_I32, s_plus_i64], inbounds=True))
        s_phi.add_incoming(b.sub(s_phi, ir.Constant(I32, 1)), shift_do)
        b.branch(add_sbody)

        b.position_at_end(add_sfinish)
        b.store(
            ir.Constant(I8, 1),
            b.gep(out_buf, [ZERO_I32, ir.Constant(I64, 0)], inbounds=True),
        )
        shifted_count = b.add(new_count, ir.Constant(I32, 1))
        shifted_radix = b.add(max_int, ir.Constant(I32, 1))
        b.branch(pack)

        b.position_at_end(add_nocarry)
        b.branch(pack)

        # --- mag_cmp: MSB-first lexicographic magnitude compare ------------
        b.position_at_end(mag_cmp)
        # For the subtract loop below we also need nc_minus_one here; it's
        # already defined in signs_check which dominates mag_cmp.
        b.branch(mag_cbody)   # enter via header for simpler phi wiring

        b.position_at_end(mag_cbody)
        m_phi = b.phi(I32, "m")
        m_phi.add_incoming(ZERO_I32, mag_cmp)
        m_done = b.icmp_signed(">=", m_phi, new_count)
        mag_cload = fn.append_basic_block("mag.cmp.load")
        b.cbranch(m_done, mag_equal, mag_cload)

        b.position_at_end(mag_cload)
        m_i64 = b.sext(m_phi, I64)
        av = b.load(b.gep(a_buf, [ZERO_I32, m_i64], inbounds=True))
        bv = b.load(b.gep(bbuf,  [ZERO_I32, m_i64], inbounds=True))
        ne = b.icmp_signed("!=", av, bv)
        m_next = b.add(m_phi, ir.Constant(I32, 1))
        m_phi.add_incoming(m_next, mag_cload)
        a_larger_here = b.icmp_signed(">", av, bv)
        b.cbranch(ne, mag_sub, mag_cbody)

        # `a_larger_here` is only defined in mag_cload; phi it at mag_sub.
        b.position_at_end(mag_sub)
        a_larger = b.phi(I1, "a.larger")
        a_larger.add_incoming(a_larger_here, mag_cload)
        b.branch(mag_sbody)

        b.position_at_end(mag_sbody)
        sub_i = b.phi(I32, "sub.i")
        borrow_phi = b.phi(I8, "borrow")
        sub_i.add_incoming(nc_minus_one, mag_sub)
        borrow_phi.add_incoming(ir.Constant(I8, 0), mag_sub)
        sub_cont = b.icmp_signed(">=", sub_i, ZERO_I32)
        sub_do = fn.append_basic_block("mag.sub.do")
        b.cbranch(sub_cont, sub_do, mag_send)

        b.position_at_end(sub_do)
        si64 = b.sext(sub_i, I64)
        a_cell = b.load(b.gep(a_buf, [ZERO_I32, si64], inbounds=True))
        b_cell = b.load(b.gep(bbuf,  [ZERO_I32, si64], inbounds=True))
        minuend    = b.select(a_larger, a_cell, b_cell)
        subtrahend = b.select(a_larger, b_cell, a_cell)
        diff = b.sub(
            b.sub(b.sext(minuend, I16), b.sext(subtrahend, I16)),
            b.sext(borrow_phi, I16),
        )
        neg = b.icmp_signed("<", diff, ir.Constant(I16, 0))
        bumped = b.select(neg, b.add(diff, ir.Constant(I16, 60)), diff)
        new_borrow = b.select(neg, ir.Constant(I8, 1), ir.Constant(I8, 0))
        b.store(b.trunc(bumped, I8),
                b.gep(out_buf, [ZERO_I32, si64], inbounds=True))
        sub_i.add_incoming(b.sub(sub_i, ir.Constant(I32, 1)), sub_do)
        borrow_phi.add_incoming(new_borrow, sub_do)
        b.branch(mag_sbody)

        b.position_at_end(mag_send)
        mixed_sign_val = b.select(a_larger, a_sign, b_sign)
        b.branch(pack)

        # --- mag_equal: zero result --------------------------------------
        b.position_at_end(mag_equal)
        ze = b.store(zero_vec, b.bitcast(out_buf, vec16.as_pointer()))
        ze.align = SEX_MAX_DIGITS
        b.branch(pack)

        # --- pack: phi result metadata, assemble struct ------------------
        b.position_at_end(pack)
        sign_phi  = b.phi(I8,  "final.sign")
        count_phi = b.phi(I32, "final.count")
        radix_phi = b.phi(I32, "final.radix")
        # add_nocarry: same sign, no overflow
        sign_phi.add_incoming(a_sign, add_nocarry)
        count_phi.add_incoming(new_count, add_nocarry)
        radix_phi.add_incoming(max_int, add_nocarry)
        # add_sfinish: same sign, overflow extended one digit
        sign_phi.add_incoming(a_sign, add_sfinish)
        count_phi.add_incoming(shifted_count, add_sfinish)
        radix_phi.add_incoming(shifted_radix, add_sfinish)
        # mag_send: mixed sign, sign of larger
        sign_phi.add_incoming(mixed_sign_val, mag_send)
        count_phi.add_incoming(new_count, mag_send)
        radix_phi.add_incoming(max_int, mag_send)
        # mag_equal: magnitudes equal, result is zero
        sign_phi.add_incoming(ir.Constant(I8, 0), mag_equal)
        count_phi.add_incoming(new_count, mag_equal)
        radix_phi.add_incoming(max_int, mag_equal)

        final_digits = b.load(out_buf)
        result: ir.Value = ir.Constant(SEX, ir.Undefined)
        result = b.insert_value(result, final_digits, SEX_IDX_DIGITS)
        result = b.insert_value(result, b.trunc(radix_phi, I8), SEX_IDX_RADIX)
        result = b.insert_value(result, b.trunc(count_phi, I8), SEX_IDX_COUNT)
        result = b.insert_value(result, sign_phi, SEX_IDX_SIGN)
        result = b.insert_value(result, ir.Constant(I8, 0), 4)
        b.ret(result)

        self._sex_add = fn
        return fn

