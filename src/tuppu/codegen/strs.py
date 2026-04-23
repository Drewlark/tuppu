"""Dynamic-string runtime helpers. Lazily built, each cached in a
`self._str_*` slot on the Codegen instance.

Design:

- `str` is `{ ptr: *u8, len: i64, cap: i64 }`. `cap == 0` marks
  borrowed (literal / global storage); any `cap > 0` marks heap
  ownership (malloc-backed).
- `__tuppu_str_release(s: *str)` reads `cap`, frees `ptr` only when
  `cap > 0`. Literal bindings pass through as no-ops. Registered
  in the scope's cleanup frame for every `str` binding so scope
  exit auto-releases — same shape as the tablets mixin.
- `__tuppu_str_concat(a, b) -> str` mallocs `a.len + b.len + 1`,
  memcpies both halves, appends a trailing NUL, returns a heap str.
- `__tuppu_str_slice(s, lo, hi) -> str` bounds-checks
  `0 <= lo <= hi <= s.len`, copies `hi - lo` bytes into a fresh
  heap buffer. v0.1 always copies; views would need lifetimes.
- `int_to_str` / `rat_to_str` / `bool_to_str` / `sex_to_str` each
  return heap strs. `sex_to_str` mirrors the existing Babylonian
  printer shape so the string form matches what `println(s)`
  shows — semicolon radix, space-separated digits, leading `-`.

Ownership across fn boundaries: the call site forces cap=0 on every
str arg via `_str_as_borrow`, so the callee's copy is a borrow. That
lets the callee register the param in its cleanup frame uniformly with
every other str binding — `__tuppu_str_release` on a cap=0 value is a
no-op, so no double-free. The caller retains sole ownership of the
bytes and frees them at its own scope exit. Returns go the other way:
the producer's cap survives, the caller binds and owns."""
from __future__ import annotations

from llvmlite import ir

from ._common import (
    CodegenError,
    I1, I8, I32, I64,
    RAT, SEX,
    SEX_IDX_DIGITS, SEX_IDX_RADIX, SEX_IDX_COUNT, SEX_IDX_SIGN,
)


class StrsMixin:
    # --- layout / accessors --------------------------------------------

    def _str_ty(self) -> ir.IdentifiedStructType:
        ty = self._struct_types.get("str")
        if ty is None:
            raise CodegenError(
                "str tablet not registered (driver should inject it)"
            )
        return ty

    def _str_extract(self, val: ir.Value, idx: int) -> ir.Value:
        """Compiler-controlled str field access. Keeping every read behind
        this helper means we can add a discriminator branch later (SSO,
        refcount tag, etc.) without touching callers."""
        assert self.builder is not None
        return self.builder.extract_value(val, idx)

    def _str_build_value_in(
        self, b: ir.IRBuilder, ptr: ir.Value,
        length: ir.Value, cap: ir.Value,
    ) -> ir.Value:
        struct_ty = self._str_ty()
        v: ir.Value = ir.Constant(struct_ty, ir.Undefined)
        v = b.insert_value(v, ptr, 0)
        v = b.insert_value(v, length, 1)
        v = b.insert_value(v, cap, 2)
        return v

    def _str_as_borrow(self, val: ir.Value) -> ir.Value:
        """Return a view of `val` with cap forced to 0. Used at call sites
        for every str arg: the callee sees a borrow, registers it in its
        cleanup frame uniformly with every other str binding, and the
        release call becomes a no-op — caller keeps sole ownership of the
        heap bytes."""
        assert self.builder is not None
        return self.builder.insert_value(val, ir.Constant(I64, 0), 2)

    def _bare_str_ptr(self, b: ir.IRBuilder, data: bytes) -> ir.Value:
        """Emit / reuse an immortal global byte array and return an i8*
        to its first byte. Usable from any builder position — the
        public `_str_ptr` only works from `self.builder`."""
        g = self._strings.get(data)
        if g is None:
            payload = data + b"\0"
            ty = ir.ArrayType(I8, len(payload))
            g = ir.GlobalVariable(
                self.module, ty, name=f".str.{self._str_counter}",
            )
            self._str_counter += 1
            g.linkage = "internal"
            g.global_constant = True
            g.initializer = ir.Constant(ty, bytearray(payload))
            self._strings[data] = g
        zero = ir.Constant(I32, 0)
        return b.gep(g, [zero, zero], inbounds=True)

    # --- libc externs --------------------------------------------------

    def _get_memcpy(self) -> ir.Function:
        if self._memcpy is None:
            i8p = I8.as_pointer()
            self._memcpy = ir.Function(
                self.module,
                ir.FunctionType(i8p, [i8p, i8p, I64]),
                name="memcpy",
            )
        return self._memcpy

    def _get_snprintf(self) -> ir.Function:
        if self._snprintf is None:
            i8p = I8.as_pointer()
            self._snprintf = ir.Function(
                self.module,
                ir.FunctionType(I32, [i8p, I64, i8p], var_arg=True),
                name="snprintf",
            )
        return self._snprintf

    # --- release -------------------------------------------------------

    def _get_str_release(self) -> ir.Function:
        """`__tuppu_str_release(s: *str)` — `if s.cap > 0 free(s.ptr)`.
        No-op for borrowed (literal) strings."""
        if self._str_release is not None:
            return self._str_release
        str_ty = self._str_ty()
        fn = ir.Function(
            self.module,
            ir.FunctionType(ir.VoidType(), [str_ty.as_pointer()]),
            name="__tuppu_str_release",
        )
        fn.args[0].name = "s"
        entry = fn.append_basic_block("entry")
        do_free = fn.append_basic_block("free")
        skip = fn.append_basic_block("skip")
        b = ir.IRBuilder(entry)
        cap_ptr = b.gep(
            fn.args[0], [ir.Constant(I32, 0), ir.Constant(I32, 2)],
            inbounds=True,
        )
        cap = b.load(cap_ptr)
        is_owned = b.icmp_signed(">", cap, ir.Constant(I64, 0))
        b.cbranch(is_owned, do_free, skip)

        b.position_at_end(do_free)
        ptr_ptr = b.gep(
            fn.args[0], [ir.Constant(I32, 0), ir.Constant(I32, 0)],
            inbounds=True,
        )
        ptr = b.load(ptr_ptr)
        b.call(self._get_free(), [ptr])
        b.branch(skip)

        b.position_at_end(skip)
        b.ret_void()
        self._str_release = fn
        return fn

    # --- concat --------------------------------------------------------

    def _get_str_concat(self) -> ir.Function:
        """`__tuppu_str_concat(a, b) -> str`. Allocates `a.len + b.len + 1`
        bytes, memcpies both halves, appends a trailing NUL (belt and
        suspenders — none of our routines require it, but cheap). Returns
        a heap-owned str."""
        if self._str_concat is not None:
            return self._str_concat
        str_ty = self._str_ty()
        fn = ir.Function(
            self.module,
            ir.FunctionType(str_ty, [str_ty, str_ty]),
            name="__tuppu_str_concat",
        )
        fn.args[0].name = "a"
        fn.args[1].name = "b"
        a, b_val = fn.args
        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)

        a_ptr = b.extract_value(a, 0)
        a_len = b.extract_value(a, 1)
        b_ptr = b.extract_value(b_val, 0)
        b_len = b.extract_value(b_val, 1)
        total = b.add(a_len, b_len, name="total")
        alloc_size = b.add(total, ir.Constant(I64, 1), name="alloc_size")
        raw = b.call(self._get_malloc(), [alloc_size])
        b.call(self._get_memcpy(), [raw, a_ptr, a_len])
        second = b.gep(raw, [a_len], inbounds=True)
        b.call(self._get_memcpy(), [second, b_ptr, b_len])
        end = b.gep(raw, [total], inbounds=True)
        b.store(ir.Constant(I8, 0), end)

        out = self._str_build_value_in(b, raw, total, total)
        b.ret(out)
        self._str_concat = fn
        return fn

    # --- slice ---------------------------------------------------------

    def _get_str_slice(self) -> ir.Function:
        """`__tuppu_str_slice(s, lo, hi) -> str`. Bounds-checks
        `0 <= lo <= hi <= s.len`; OOB traps. Allocates `hi - lo` bytes
        + trailing NUL, copies, returns heap-owned."""
        if self._str_slice is not None:
            return self._str_slice
        str_ty = self._str_ty()
        fn = ir.Function(
            self.module,
            ir.FunctionType(str_ty, [str_ty, I64, I64]),
            name="__tuppu_str_slice",
        )
        fn.args[0].name = "s"
        fn.args[1].name = "lo"
        fn.args[2].name = "hi"
        s, lo, hi = fn.args
        entry = fn.append_basic_block("entry")
        trap_bb = fn.append_basic_block("trap")
        ok_bb = fn.append_basic_block("ok")
        b = ir.IRBuilder(entry)

        s_ptr = b.extract_value(s, 0)
        s_len = b.extract_value(s, 1)
        lo_bad = b.icmp_signed("<", lo, ir.Constant(I64, 0))
        hi_lo  = b.icmp_signed("<", hi, lo)
        hi_bad = b.icmp_signed(">", hi, s_len)
        bad = b.or_(b.or_(lo_bad, hi_lo), hi_bad)
        b.cbranch(bad, trap_bb, ok_bb)

        b.position_at_end(trap_bb)
        b.call(self._get_trap(), [])
        b.unreachable()

        b.position_at_end(ok_bb)
        length = b.sub(hi, lo, name="slice_len")
        alloc_size = b.add(length, ir.Constant(I64, 1))
        raw = b.call(self._get_malloc(), [alloc_size])
        src = b.gep(s_ptr, [lo], inbounds=True)
        b.call(self._get_memcpy(), [raw, src, length])
        end = b.gep(raw, [length], inbounds=True)
        b.store(ir.Constant(I8, 0), end)
        out = self._str_build_value_in(b, raw, length, length)
        b.ret(out)
        self._str_slice = fn
        return fn

    # --- int / rat / bool → str ---------------------------------------

    def _get_int_to_str(self) -> ir.Function:
        """`__tuppu_int_to_str(v: i64) -> str`. 21-byte buffer covers
        every i64 including `-9223372036854775808\\0`."""
        if self._int_to_str is not None:
            return self._int_to_str
        str_ty = self._str_ty()
        fn = ir.Function(
            self.module,
            ir.FunctionType(str_ty, [I64]),
            name="__tuppu_int_to_str",
        )
        fn.args[0].name = "v"
        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        BUF = 24
        raw = b.call(self._get_malloc(), [ir.Constant(I64, BUF)])
        fmt = self._bare_str_ptr(b, b"%lld")
        written = b.call(
            self._get_snprintf(),
            [raw, ir.Constant(I64, BUF), fmt, fn.args[0]],
        )
        length = b.sext(written, I64)
        out = self._str_build_value_in(b, raw, length, length)
        b.ret(out)
        self._int_to_str = fn
        return fn

    def _get_rat_to_str(self) -> ir.Function:
        """`__tuppu_rat_to_str(r: rat) -> str`. Formats as `num/den`."""
        if self._rat_to_str is not None:
            return self._rat_to_str
        str_ty = self._str_ty()
        fn = ir.Function(
            self.module,
            ir.FunctionType(str_ty, [RAT]),
            name="__tuppu_rat_to_str",
        )
        fn.args[0].name = "r"
        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        BUF = 48
        raw = b.call(self._get_malloc(), [ir.Constant(I64, BUF)])
        num = b.extract_value(fn.args[0], 0)
        den = b.extract_value(fn.args[0], 1)
        fmt = self._bare_str_ptr(b, b"%lld/%lld")
        written = b.call(
            self._get_snprintf(),
            [raw, ir.Constant(I64, BUF), fmt, num, den],
        )
        length = b.sext(written, I64)
        out = self._str_build_value_in(b, raw, length, length)
        b.ret(out)
        self._rat_to_str = fn
        return fn

    def _get_bool_to_str(self) -> ir.Function:
        """`__tuppu_bool_to_str(b: i1) -> str`. Returns a fresh heap
        copy (not a borrowed view of a static "true" / "false") so
        the cleanup path treats it uniformly with other to_str outputs."""
        if self._bool_to_str is not None:
            return self._bool_to_str
        str_ty = self._str_ty()
        fn = ir.Function(
            self.module,
            ir.FunctionType(str_ty, [I1]),
            name="__tuppu_bool_to_str",
        )
        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        true_ptr = self._bare_str_ptr(b, b"true")
        false_ptr = self._bare_str_ptr(b, b"false")
        chosen_ptr = b.select(fn.args[0], true_ptr, false_ptr)
        chosen_len = b.select(
            fn.args[0], ir.Constant(I64, 4), ir.Constant(I64, 5),
        )
        alloc_size = b.add(chosen_len, ir.Constant(I64, 1))
        raw = b.call(self._get_malloc(), [alloc_size])
        b.call(self._get_memcpy(), [raw, chosen_ptr, chosen_len])
        end = b.gep(raw, [chosen_len], inbounds=True)
        b.store(ir.Constant(I8, 0), end)
        out = self._str_build_value_in(b, raw, chosen_len, chosen_len)
        b.ret(out)
        self._bool_to_str = fn
        return fn

    # --- sex → str (Babylonian form, mirrors __tuppu_sex_print) -------

    def _get_sex_to_str(self) -> ir.Function:
        """`__tuppu_sex_to_str(s: sex) -> str`. Produces the same
        Babylonian notation that `println(s)` renders: optional leading
        `-`, integer digits space-separated, `;` at the radix, fractional
        digits space-separated. Mirrors `__tuppu_sex_print` but writes
        into a freshly malloc'd buffer rather than calling printf. The
        result is heap-owned, so the cleanup path frees it.

        Buffer sizing: 16 digits × 3 chars (max "59" + sep) + sign +
        radix + NUL ≤ 80 bytes. Pre-allocate that; trim the final
        len to what we actually wrote."""
        if self._sex_to_str is not None:
            return self._sex_to_str
        str_ty = self._str_ty()
        fn = ir.Function(
            self.module,
            ir.FunctionType(str_ty, [SEX]),
            name="__tuppu_sex_to_str",
        )
        fn.args[0].name = "sx"
        sx = fn.args[0]
        BUF = 80

        entry     = fn.append_basic_block("entry")
        neg_bb    = fn.append_basic_block("neg")
        after_sign= fn.append_basic_block("after.sign")
        zero_bb   = fn.append_basic_block("empty.zero")
        int_loop  = fn.append_basic_block("int.loop")
        int_body  = fn.append_basic_block("int.body")
        int_next  = fn.append_basic_block("int.next")
        radix_bb  = fn.append_basic_block("radix")
        has_frac  = fn.append_basic_block("semi")
        frac_loop = fn.append_basic_block("frac.loop")
        frac_body = fn.append_basic_block("frac.body")
        frac_next = fn.append_basic_block("frac.next")
        done      = fn.append_basic_block("done")

        b = ir.IRBuilder(entry)

        # buf: u8[BUF]. cursor + running length live in stack slots so
        # each writing block can advance them uniformly.
        raw = b.call(self._get_malloc(), [ir.Constant(I64, BUF)])
        cur_slot = b.alloca(I8.as_pointer(), name="cur")
        len_slot = b.alloca(I64, name="len")
        b.store(raw, cur_slot)
        b.store(ir.Constant(I64, 0), len_slot)

        fmt_digit = self._bare_str_ptr(b, b"%d")

        def advance(b: ir.IRBuilder, delta: ir.Value) -> None:
            cur = b.load(cur_slot)
            b.store(b.gep(cur, [delta], inbounds=True), cur_slot)
            ln = b.load(len_slot)
            b.store(b.add(ln, delta), len_slot)

        def put_char(b: ir.IRBuilder, ch: int) -> None:
            cur = b.load(cur_slot)
            b.store(ir.Constant(I8, ch), cur)
            advance(b, ir.Constant(I64, 1))

        # Spill sex so we can index digits dynamically.
        sx_slot = b.alloca(SEX, name="sx.slot")
        b.store(sx, sx_slot)
        digits_addr = b.gep(
            sx_slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_DIGITS)],
            inbounds=True,
        )
        radix = b.sext(b.load(b.gep(
            sx_slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_RADIX)],
            inbounds=True,
        )), I64)
        count = b.sext(b.load(b.gep(
            sx_slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_COUNT)],
            inbounds=True,
        )), I64)
        sign = b.load(b.gep(
            sx_slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_SIGN)],
            inbounds=True,
        ))
        is_neg = b.icmp_signed("!=", sign, ir.Constant(I8, 0))
        b.cbranch(is_neg, neg_bb, after_sign)

        b.position_at_end(neg_bb)
        put_char(b, ord("-"))
        b.branch(after_sign)

        b.position_at_end(after_sign)
        # count==0 → emit "0" once and done.
        is_zero = b.icmp_signed("==", count, ir.Constant(I64, 0))
        b.cbranch(is_zero, zero_bb, int_loop)

        b.position_at_end(zero_bb)
        put_char(b, ord("0"))
        b.branch(done)

        # Integer-digit loop.
        b.position_at_end(int_loop)
        i_phi = b.phi(I64, "i")
        i_phi.add_incoming(ir.Constant(I64, 0), after_sign)
        done_int = b.icmp_signed(">=", i_phi, radix)
        b.cbranch(done_int, radix_bb, int_body)

        b.position_at_end(int_body)
        need_space = b.icmp_signed(">", i_phi, ir.Constant(I64, 0))
        sp_bb   = fn.append_basic_block("int.sp")
        no_sp   = fn.append_basic_block("int.nosp")
        b.cbranch(need_space, sp_bb, no_sp)
        b.position_at_end(sp_bb)
        put_char(b, ord(" "))
        b.branch(no_sp)
        b.position_at_end(no_sp)
        dptr = b.gep(digits_addr, [ir.Constant(I32, 0), i_phi], inbounds=True)
        dval = b.zext(b.load(dptr), I32)
        cur = b.load(cur_slot)
        ln_now = b.load(len_slot)
        remaining = b.sub(ir.Constant(I64, BUF), ln_now)
        written = b.call(
            self._get_snprintf(),
            [cur, remaining, fmt_digit, dval],
        )
        advance(b, b.sext(written, I64))
        b.branch(int_next)

        b.position_at_end(int_next)
        next_i = b.add(i_phi, ir.Constant(I64, 1))
        i_phi.add_incoming(next_i, int_next)
        b.branch(int_loop)

        b.position_at_end(radix_bb)
        fractional = b.icmp_signed(">", count, radix)
        b.cbranch(fractional, has_frac, done)

        b.position_at_end(has_frac)
        put_char(b, ord(";"))
        b.branch(frac_loop)

        b.position_at_end(frac_loop)
        j_phi = b.phi(I64, "j")
        j_phi.add_incoming(radix, has_frac)
        done_frac = b.icmp_signed(">=", j_phi, count)
        b.cbranch(done_frac, done, frac_body)

        b.position_at_end(frac_body)
        need_sp2 = b.icmp_signed(">", j_phi, radix)
        sp2_bb = fn.append_basic_block("frac.sp")
        no_sp2 = fn.append_basic_block("frac.nosp")
        b.cbranch(need_sp2, sp2_bb, no_sp2)
        b.position_at_end(sp2_bb)
        put_char(b, ord(" "))
        b.branch(no_sp2)
        b.position_at_end(no_sp2)
        fptr = b.gep(digits_addr, [ir.Constant(I32, 0), j_phi], inbounds=True)
        fval = b.zext(b.load(fptr), I32)
        cur = b.load(cur_slot)
        ln_now = b.load(len_slot)
        remaining = b.sub(ir.Constant(I64, BUF), ln_now)
        written = b.call(
            self._get_snprintf(),
            [cur, remaining, fmt_digit, fval],
        )
        advance(b, b.sext(written, I64))
        b.branch(frac_next)

        b.position_at_end(frac_next)
        next_j = b.add(j_phi, ir.Constant(I64, 1))
        j_phi.add_incoming(next_j, frac_next)
        b.branch(frac_loop)

        b.position_at_end(done)
        # NUL-terminate (belt and suspenders).
        cur = b.load(cur_slot)
        b.store(ir.Constant(I8, 0), cur)
        length = b.load(len_slot)
        out = self._str_build_value_in(b, raw, length, length)
        b.ret(out)

        self._sex_to_str = fn
        return fn
