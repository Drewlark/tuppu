"""Intrinsic emitters: stdlib I/O (`print`, `println`, `read_int`), the
`rat(num, den)` constructor, and dynamic-string intrinsics that return
heap-owned str values (`str_concat`, `int_to_str`, `sex_to_str`,
`bytes_to_str`, etc.). Extracted from `codegen/__init__.py` as
`IntrinsicsMixin`."""
from __future__ import annotations

from llvmlite import ir

from .. import ast as A
from ._common import (
    CodegenError, Variable,
    I1, I8, I16, I32, I64,
    RAT, SEX,
)


class IntrinsicsMixin:
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
        """Variadic str concat: `str_concat(a, b, ..., z)` emits a
        single linear-time join. See `_emit_str_concat` for the
        mechanics."""
        if len(args) < 2:
            raise CodegenError(
                "str_concat takes at least two arguments"
            )
        parts: list[tuple[ir.Value, A.Expr]] = []
        for arg in args:
            v = self._gen_expr(arg)
            if v is None:
                raise CodegenError("str_concat argument has no value")
            parts.append((v, arg))
        return self._emit_str_concat(parts)

    def _emit_str_concat(
        self, parts: list[tuple[ir.Value, A.Expr]],
    ) -> ir.Value:
        """Emit a single-malloc linear-time concat over pre-evaluated
        str values — sum all part lengths, malloc once, memcpy each
        part at a running offset, NUL-terminate. Linear in the total
        output size regardless of arity, so `str_concat(h1, h2, h3,
        h4, body)` reads like a log line and runs in one pass rather
        than four nested chain allocations. The per-part AST is
        carried through for rvalue-cleanup dispatch so any heap
        intermediate (`foo() + "x"`) is released at scope exit."""
        assert self.builder is not None
        b = self.builder
        # Extract ptr / len up front so the two passes (sum lengths,
        # copy bytes) share the same SSA values.
        ptrs_lens = [
            (b.extract_value(v, 0), b.extract_value(v, 1))
            for v, _src in parts
        ]
        total: ir.Value = ir.Constant(I64, 0)
        for _, ln in ptrs_lens:
            total = b.add(total, ln)
        alloc_size = b.add(total, ir.Constant(I64, 1))
        raw = b.call(self._get_malloc(), [alloc_size])
        offset: ir.Value = ir.Constant(I64, 0)
        for ptr, ln in ptrs_lens:
            dst = b.gep(raw, [offset], inbounds=True)
            b.call(self._get_memcpy(), [dst, ptr, ln])
            offset = b.add(offset, ln)
        b.store(ir.Constant(I8, 0), b.gep(raw, [total], inbounds=True))
        return self._str_build_value_in(b, raw, total, total)

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

    def _gen_buffer_to_str_call(self, args: list[A.Expr]) -> ir.Value:
        """Lower `buffer_to_str(buf, n)` — copy the first `n` bytes of
        a `buffer[N]u8` into a fresh heap-owned str. The arg must be
        a buffer-typed Ident (so we can take the alloca's address);
        `n` is bounds-checked against the buffer's compile-time size
        at runtime."""
        assert self.builder is not None
        if len(args) != 2:
            raise CodegenError("buffer_to_str takes exactly two arguments")
        buf_expr = args[0]
        if not isinstance(buf_expr, A.Ident):
            raise CodegenError(
                f"buffer_to_str: buffer argument must be an Ident, "
                f"got {type(buf_expr).__name__}"
            )
        var = self._lookup(buf_expr.name)
        if not isinstance(var.value_ty, ir.ArrayType) or var.value_ty.element != I8:
            raise CodegenError(
                f"buffer_to_str: {buf_expr.name!r} is not a buffer[N]u8 "
                f"(got {var.value_ty})"
            )
        n = self._gen_expr(args[1])
        if n is None:
            raise CodegenError("buffer_to_str length argument has no value")
        n = self._coerce(n, I64)
        b = self.builder
        # Runtime bounds check: n must be in [0, N]. Saturation would
        # be friendlier, but trapping keeps the "buffer is always
        # safe" invariant.
        self._emit_bounds_trap_inclusive(n, var.value_ty.count)
        elem_ptr = b.gep(
            var.ir_ref,
            [ir.Constant(I32, 0), ir.Constant(I32, 0)],
            inbounds=True,
        )
        # malloc(n+1); memcpy(raw, buf_ptr, n); NUL-terminate.
        alloc_size = b.add(n, ir.Constant(I64, 1))
        raw = b.call(self._get_malloc(), [alloc_size])
        b.call(self._get_memcpy(), [raw, elem_ptr, n])
        b.store(ir.Constant(I8, 0), b.gep(raw, [n], inbounds=True))
        return self._str_build_value_in(b, raw, n, n)

    def _emit_bounds_trap_inclusive(self, n: ir.Value, size: int) -> None:
        """Trap if `n < 0` or `n > size` — buffer_to_str permits `n == N`
        (copies the full buffer), which the exclusive bounds trap
        rejects. One-off helper; the tablets/str bounds paths stay
        exclusive."""
        assert self.builder is not None
        b = self.builder
        oob_lo = b.icmp_signed("<", n, ir.Constant(I64, 0))
        oob_hi = b.icmp_signed(">", n, ir.Constant(I64, size))
        oob = b.or_(oob_lo, oob_hi)
        fn = b.function
        trap_bb = fn.append_basic_block("bounds.trap")
        ok_bb = fn.append_basic_block("bounds.ok")
        b.cbranch(oob, trap_bb, ok_bb)
        b.position_at_end(trap_bb)
        b.call(self._get_trap(), [])
        b.unreachable()
        b.position_at_end(ok_bb)

    def _gen_str_slice_call(self, args: list[A.Expr]) -> ir.Value:
        assert self.builder is not None
        s = self._gen_expr(args[0])
        lo = self._gen_expr(args[1])
        hi = self._gen_expr(args[2])
        if s is None or lo is None or hi is None:
            raise CodegenError("str_slice argument has no value")
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

