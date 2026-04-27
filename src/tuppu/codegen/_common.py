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

# ivec value layout — `{ buf: i8**, len: i64, cap: i64 }`. Shared
# across every `ivec<T>`; T-specific behavior lives in per-T helper
# fns (push, get, get_addr). The buffer is a heap-allocated array of
# pointers (each pointing to a separately heap-allocated T), traced
# by the runtime's `__tuppu_ivec_storage_desc`. One LLVM type for
# every ivec means one shared descriptor too — the buffer pointer
# at offset 0 is always a regular GC-traced pointer.
IVEC_STRUCT = ir.LiteralStructType([
    I8.as_pointer().as_pointer(),  # buf: i8**
    I64,                           # len
    I64,                           # cap
])
IVEC_IDX_BUF = 0
IVEC_IDX_LEN = 1
IVEC_IDX_CAP = 2
# Initial cap on the first push (0 means "lazy alloc"). Subsequent
# grows double; chosen as a single allocation rather than a sequence
# of 1→2→4→8 to keep small ivecs off the GC's small-object pile.
IVEC_INITIAL_CAP = 8

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
    """Monomorphized helper functions and struct types for one (N, T) pair.

    Only `node_ty` / `tablets_ty` / `suffix` get populated eagerly at
    registration — `_lower_type` needs those to resolve struct and
    seal fields referring to `tablets[N]T`. Helper fn bodies defer
    until first call; the chunk descriptor they depend on can't be
    computed until every struct + seal is fully resolved (an element
    type that's a seal is opaque during struct-field resolution),
    and emitting bodies eagerly captures the desc against the wrong
    size. Use `_get_tablets_push`, `_get_tablets_get`, etc. to
    materialize them on demand."""
    N: int
    elem_ty: ir.Type
    node_ty: ir.IdentifiedStructType   # {[N x T], used: i64, next: Node*}
    tablets_ty: ir.LiteralStructType   # {head: Node*, tail: Node*, len: i64}
    suffix: str
    # True iff the element type was declared as `wedge T` at the source
    # level. LLVM types collapse `wedge T` and `*T` to the same `T*`,
    # so we keep this flag separately to decide whether each chunk slot
    # should be traced via mark_wedge (interior-pointer lookup, keeps
    # the source arena alive) or mark_ptr (treat as a regular GC obj
    # start). Set at `_get_tablets` time from the lowering call site.
    elem_is_wedge: bool = False
    push: ir.Function | None = None
    get: ir.Function | None = None
    get_addr: ir.Function | None = None
    release: ir.Function | None = None
    clone: ir.Function | None = None


@dataclass
class IVecInfo:
    """Per-T monomorphized helper fns for `ivec<T>`. The struct LLVM
    type is shared (`IVEC_STRUCT`); only the helpers vary per element
    type. Helpers are emitted lazily on first call so a type that
    never gets indexed / pushed produces zero IR."""
    elem_ty: ir.Type
    suffix: str
    elem_is_wedge: bool = False
    push: ir.Function | None = None
    get: ir.Function | None = None
    get_addr: ir.Function | None = None


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
