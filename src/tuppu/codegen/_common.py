"""Shared codegen definitions: LLVM type aliases, the runtime SEX/RAT
struct layouts, error class, and utility dataclasses used across the
codegen mixins.

Kept as a separate module so each mixin file (sex.py, rat.py,
tablets.py) can import these without pulling in the full Codegen
class — avoids circular-import pain."""
from __future__ import annotations

from dataclasses import dataclass

from llvmlite import ir

from ..errors import CompileError


class CodegenError(CompileError):
    def __init__(self, message: str, line: int = 0, col: int = 0) -> None:
        if line:
            super().__init__(f"{line}:{col}: {message}")
        else:
            super().__init__(message)
        self.message = message
        self.line = line
        self.col = col


I1 = ir.IntType(1)
I8 = ir.IntType(8)
I16 = ir.IntType(16)
I32 = ir.IntType(32)
I64 = ir.IntType(64)

# rat is a literal struct { i64 num, i64 den }, always reduced (gcd=1) and
# normalized so den > 0 at construction time. Field 0 is num, field 1 is den.
RAT = ir.LiteralStructType([I64, I64])

# Sex: Babylonian-faithful sexagesimal representation. A fixed-width digit
# sequence with explicit radix position and sign. Each digit is in [0, 60).
# Layout (20 bytes):
#   digits : [16]u8   fixed buffer, int digits first then fractional
#   radix  : u8       index where fractional part begins (also = int digit count)
#   count  : u8       total digits used (0..=16)
#   sign   : i8       0 = positive, non-zero = negative
#   _pad   : u8       alignment filler
# Values beyond 16 total digits are a compile-time error.
SEX_MAX_DIGITS = 16
SEX = ir.LiteralStructType([
    ir.ArrayType(I8, SEX_MAX_DIGITS),
    I8, I8, I8, I8,
])
SEX_IDX_DIGITS = 0
SEX_IDX_RADIX = 1
SEX_IDX_COUNT = 2
SEX_IDX_SIGN = 3

INT_WIDTH: dict[str, int] = {
    "i8": 8, "i16": 16, "i32": 32, "i64": 64,
    "u8": 8, "u16": 16, "u32": 32, "u64": 64,
}


@dataclass
class Variable:
    is_mut: bool
    ir_ref: ir.Value       # SSA value for step; alloca pointer for mut
    value_ty: ir.Type      # logical value type (not the pointer type)
    # If this binding is a borrow — `step x = y` where y owns a
    # cleanup-bearing value — records the cleanup-frame entry name to
    # transfer instead of this binding's own name when the value
    # escapes via tail-return. None for bindings that own their value
    # (no source to redirect to) or that hold non-cleanup types.
    transfer_on_tail: str | None = None


@dataclass
class TabletsInfo:
    """Monomorphized helper functions and struct types for one (N, T) pair."""
    N: int
    elem_ty: ir.Type
    node_ty: ir.IdentifiedStructType   # {[N x T], used: i64, next: Node*}
    tablets_ty: ir.LiteralStructType   # {head: Node*, tail: Node*, len: i64}
    push: ir.Function
    get: ir.Function
    get_addr: ir.Function
    release: ir.Function
    # `clone(src_ptr) -> tablets_ty` — deep-copy. Built lazily on first
    # use via `_get_tablets_clone` so plain scalar-element tablets that
    # never get cloned don't pay for the helper.
    clone: ir.Function | None = None


# Names that the user cannot shadow — they resolve to compiler intrinsics.
# "rat" is both a type and a construction intrinsic (rat(num, den) -> rat).
INTRINSICS: frozenset[str] = frozenset({
    "print", "println", "read_int", "rat",
    # Dynamic-string intrinsics that return heap-owned `str` values.
    # Only the ones requiring native support (heap allocation or
    # internal-field digit decomposition) live here; `bool_to_str`,
    # `rat_to_str`, and `str_concat` are expressible in Tuppu itself,
    # so they moved to stdlib/str.tpu. The `str + str` binary operator
    # still uses the native `_emit_str_concat` single-malloc fast path
    # for the two-argument case.
    "str_slice",
    "int_to_str", "sex_to_str",
    # Flatten a tablets[N]u8 into a heap-owned str — underpins the
    # growable str_buf pattern without a quadratic-rebuild trap.
    "bytes_to_str",
    # buffer_to_str(buf, n) — copy n bytes out of a buffer[N]u8 into
    # a heap-owned str. Paired with buffer[N]u8 for the FFI story.
    "buffer_to_str",
})
