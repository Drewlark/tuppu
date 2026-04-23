"""`buffer[N]u8` — fixed-size, stack-allocated, bounds-checked byte
buffers for FFI and byte-level programming. Exercised here: basic
index read/write, `.len`, `buffer_to_str` roundtrip, bounds trap,
colophon FFI arg passing, and the parts the typechecker rejects."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_to_binary, compile_to_ir
from tuppu.errors import CompileError


def run(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return r.returncode, r.stdout, r.stderr


# --- basic use -------------------------------------------------------------

def test_buffer_declare_and_index(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[8]u8\n"
        "  buf[0] = 72 as u8\n"
        "  buf[1] = 73 as u8\n"
        "  println(buf[0])\n"
        "  println(buf[1])\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"72\n73\n"


def test_buffer_len_is_compile_time(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[1024]u8\n"
        "  println(buf.len)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1024\n"


def test_buffer_zero_init_on_declaration(tmp_path):
    # Unwritten slots must read back as 0.
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[16]u8\n"
        "  mut i: i64 = 0\n"
        "  mut sum: i64 = 0\n"
        "  while i < 16 {\n"
        "    sum = sum + buf[i] as i64\n"
        "    i = i + 1\n"
        "  }\n"
        "  println(sum)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\n"


def test_buffer_aug_assign_into_slot(tmp_path):
    # `buf[i] += x` goes through the lvalue-indexing machinery; same
    # path as `arr[n] += x` for tablets.
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[4]u8\n"
        "  buf[0] = 10 as u8\n"
        "  buf[0] += 5 as u8\n"
        "  println(buf[0])\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"15\n"


# --- bounds checking -------------------------------------------------------

def test_buffer_read_bounds_trap(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[4]u8\n"
        "  step b = buf[10]\n"
        "  println(b)\n"
        "  0\n"
        "}\n"
    )
    rc, _, _ = run(src, tmp_path)
    assert rc != 0  # trap (SIGABRT / SIGILL)


def test_buffer_write_bounds_trap(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[4]u8\n"
        "  buf[10] = 1 as u8\n"
        "  0\n"
        "}\n"
    )
    rc, _, _ = run(src, tmp_path)
    assert rc != 0


def test_buffer_negative_index_traps(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[4]u8\n"
        "  step b = buf[-1 as i64]\n"
        "  println(b)\n"
        "  0\n"
        "}\n"
    )
    rc, _, _ = run(src, tmp_path)
    assert rc != 0


# --- buffer_to_str ---------------------------------------------------------

def test_buffer_to_str_basic(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[16]u8\n"
        "  buf[0] = 72 as u8\n"
        "  buf[1] = 105 as u8\n"
        "  step s = buffer_to_str(buf, 2 as i64)\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"Hi\n"


def test_buffer_to_str_zero_length_empty(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[16]u8\n"
        "  step s = buffer_to_str(buf, 0 as i64)\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"\n"


def test_buffer_to_str_full_length_allowed(tmp_path):
    # `n == N` copies the whole buffer — the inclusive bounds rule
    # that separates buffer_to_str's check from buf[N]'s (exclusive).
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[3]u8\n"
        "  buf[0] = 65 as u8\n"
        "  buf[1] = 66 as u8\n"
        "  buf[2] = 67 as u8\n"
        "  step s = buffer_to_str(buf, 3 as i64)\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"ABC\n"


def test_buffer_to_str_length_too_big_traps(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[4]u8\n"
        "  step s = buffer_to_str(buf, 99 as i64)\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    rc, _, _ = run(src, tmp_path)
    assert rc != 0


# --- colophon FFI ----------------------------------------------------------

def test_buffer_via_colophon_memset(tmp_path):
    # memset writes n bytes to buf; we then read them back.
    src = (
        'colophon fn memset(buf: buffer[32]u8, c: i32, n: u64) -> buffer[32]u8\n'
    )
    # memset actually returns void* but we don't use the return here.
    src = (
        "colophon fn memset(mut buf: buffer[32]u8, c: i32, n: u64) -> i64\n"
        "fn main() -> i32 {\n"
        "  mut buf: buffer[32]u8\n"
        "  memset(buf, 65 as i32, 5 as u64)\n"
        "  step s = buffer_to_str(buf, 5 as i64)\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    # memset's actual sig is (void*, int, size_t) -> void*. We declare
    # it as -> i64 because we ignore the return; the C side doesn't
    # know or care. Passing buf decays to a pointer, so memset writes
    # through the caller's storage.
    _, out, _ = run(src, tmp_path)
    assert out == b"AAAAA\n"


# --- typecheck rejections --------------------------------------------------

def test_buffer_cannot_be_return_type(tmp_path):
    src = (
        "fn make() -> buffer[16]u8 {\n"
        "  mut buf: buffer[16]u8\n"
        "  buf\n"
        "}\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="cannot return a buffer"):
        compile_to_ir(src)


def test_buffer_cannot_be_struct_field(tmp_path):
    src = (
        "tablet Packet { bytes: buffer[16]u8 }\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="cannot be a buffer"):
        compile_to_ir(src)


def test_buffer_non_u8_element_rejected(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[16]i64\n"
        "  0\n"
        "}\n"
    )
    with pytest.raises(CompileError, match="buffer element must be u8"):
        compile_to_ir(src)


def test_buffer_index_non_int_rejected(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[16]u8\n"
        "  step b = buf[true]\n"
        "  println(b)\n"
        "  0\n"
        "}\n"
    )
    with pytest.raises(CompileError, match="buffer index must be integer"):
        compile_to_ir(src)


def test_buffer_unknown_field_rejected(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: buffer[16]u8\n"
        "  println(buf.size)\n"
        "  0\n"
        "}\n"
    )
    with pytest.raises(CompileError, match="buffer has no field 'size'"):
        compile_to_ir(src)
