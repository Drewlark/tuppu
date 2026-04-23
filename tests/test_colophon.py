"""Colophon declarations — Tuppu's typed FFI to libc. Each `colophon
fn` becomes an LLVM extern whose call sites marshal `str <-> cstr` and
pass primitives through. Here we exercise the round-trip for getenv /
atoi / exit, the NULL-return path (missing env var), and the error
cases typecheck rejects."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_to_binary, compile_to_ir
from tuppu.errors import CompileError


def run(src: str, tmp_path: Path, stdin: bytes = b"", env=None
        ) -> tuple[int, bytes, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run(
        [str(binary)], input=stdin, capture_output=True, env=env,
    )
    return r.returncode, r.stdout, r.stderr


# --- basic marshaling ------------------------------------------------------

def test_colophon_atoi_primitive_roundtrip(tmp_path):
    # Str arg in (marshaled to cstr) + i32 return (passes through).
    src = (
        'colophon fn atoi(s: str) -> i32\n'
        'fn main() -> i32 {\n'
        '  println(atoi("42"))\n'
        '  println(atoi("-17"))\n'
        '  println(atoi("   99 trailing"))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"42\n-17\n99\n"


def test_colophon_getenv_str_return(tmp_path):
    # Str return: C returns i8*, compiler emits strlen+malloc+memcpy to
    # hand a heap-owned Tuppu str back to user code.
    src = (
        'colophon fn getenv(name: str) -> str\n'
        'fn main() -> i32 {\n'
        '  step home = getenv("TUPPU_TEST_HOME")\n'
        '  println(home)\n'
        '  println(home.len)\n'
        '  0\n'
        '}\n'
    )
    env = {"TUPPU_TEST_HOME": "/tmp/tablet-archive"}
    _, out, _ = run(src, tmp_path, env=env)
    assert out == b"/tmp/tablet-archive\n19\n"


def test_colophon_getenv_null_becomes_empty(tmp_path):
    # Missing env var → C returns NULL → marshaling yields an empty
    # str (`{ptr=null, len=0, cap=0}`). Collapses "not found" with
    # "found empty"; stdlib can wrap if the distinction matters.
    src = (
        'colophon fn getenv(name: str) -> str\n'
        'fn main() -> i32 {\n'
        '  step missing = getenv("TUPPU_NO_SUCH_VAR_XYZZY")\n'
        '  println("len =", missing.len)\n'
        '  0\n'
        '}\n'
    )
    # Explicitly unset the env var by passing a minimal env.
    _, out, _ = run(src, tmp_path, env={"PATH": "/usr/bin"})
    assert out == b"len =0\n"


def test_colophon_exit_controls_return_code(tmp_path):
    # Void-return colophon + a specific exit status. The println must
    # be flushed before exit; we include write() via println which
    # ends with fflush-free (write is direct), so the line is out
    # before exit() fires.
    src = (
        'colophon fn exit(code: i32)\n'
        'fn main() -> i32 {\n'
        '  println("before exit")\n'
        '  exit(7 as i32)\n'
        '  0\n'
        '}\n'
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 7
    assert out == b"before exit\n"


def test_colophon_cstr_copies_so_caller_retains_bytes(tmp_path):
    # Passing the same Tuppu str to a colophon twice should work —
    # each call mallocs its own cstr, copies, frees after. The Tuppu
    # str is untouched across the calls.
    src = (
        'colophon fn atoi(s: str) -> i32\n'
        'fn main() -> i32 {\n'
        '  step n = "100"\n'
        '  println(atoi(n))\n'
        '  println(atoi(n))\n'
        '  println(n)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"100\n100\n100\n"


# --- error / rejection ----------------------------------------------------

def test_colophon_accepts_user_tablet_arg(tmp_path):
    # User tablets pass by value across the C boundary (LLVM handles
    # the struct-arg ABI). This smoke-tests the shape end-to-end
    # without needing a real C library: a tablet with identity layout
    # gets handed to `atoi` via a tiny demo where we only read one
    # of its fields — here we just compile to IR and check that the
    # extern declaration has the struct parameter type.
    ir = compile_to_ir(
        'tablet Point { x: i64, y: i64 }\n'
        'colophon fn weird(p: Point) -> i64\n'
        'fn main() -> i32 { 0 }\n'
    )
    # Extern declaration should take a `%"Point"` struct parameter.
    assert "declare i64 @\"weird\"" in ir or "declare i64 @weird" in ir
    assert "Point" in ir


def test_colophon_rejects_struct_return():
    # Struct returns across the FFI aren't exposed yet (platform-
    # dependent layouts for stat-shaped things, ABI quirks we haven't
    # confronted). Parameters are fine; returns will land later.
    with pytest.raises(CompileError, match="isn't marshalable"):
        compile_to_ir(
            'tablet Point { x: i64, y: i64 }\n'
            'colophon fn make_point() -> Point\n'
            'fn main() -> i32 { 0 }\n'
        )


def test_colophon_tcp_bind_roundtrip(tmp_path):
    # The payoff test: declare a real sockaddr_in in Tuppu, call
    # socket + bind + close through raw libc, verify the kernel
    # accepted our struct layout. macOS-specific sockaddr_in
    # (sin_len byte up front); on Linux the layout differs and
    # this test would need a conditional compile story we haven't
    # built yet.
    import platform
    if platform.system() != "Darwin":
        pytest.skip("sockaddr_in layout assumed macOS-specific here")
    src = (
        'tablet sockaddr_in {\n'
        '  sin_len: u8, sin_family: u8, sin_port: u16,\n'
        '  sin_addr: u32, sin_zero: u64\n'
        '}\n'
        'colophon fn socket(domain: i32, ty: i32, proto: i32) -> i32\n'
        'colophon fn bind(fd: i32, mut addr: sockaddr_in, addrlen: u32) -> i32\n'
        'colophon fn close(fd: i32) -> i32\n'
        'colophon fn htons(val: u16) -> u16\n'
        'colophon fn htonl(val: u32) -> u32\n'
        'fn main() -> i32 {\n'
        '  step fd: i32 = socket(2 as i32, 1 as i32, 0 as i32)\n'
        '  if fd < (0 as i32) { yield 1 }\n'
        '  mut addr: sockaddr_in\n'
        '  addr.sin_len    = 16 as u8\n'
        '  addr.sin_family = 2 as u8\n'
        '  addr.sin_port   = htons(0 as u16)\n'
        '  addr.sin_addr   = htonl(2130706433 as u32)\n'
        '  addr.sin_zero   = 0 as u64\n'
        '  step rc: i32 = bind(fd, addr, 16 as u32)\n'
        '  close(fd)\n'
        '  if rc == (0 as i32) { 0 } else { 2 }\n'
        '}\n'
    )
    rc, _, _ = run(src, tmp_path)
    assert rc == 0


def test_colophon_name_collision_with_fn():
    with pytest.raises(CompileError, match="duplicate"):
        compile_to_ir(
            'fn foo() -> i32 { 0 }\n'
            'colophon fn foo() -> i32\n'
            'fn main() -> i32 { 0 }\n'
        )


def test_colophon_name_collision_with_intrinsic():
    with pytest.raises(CompileError, match="built-in intrinsic"):
        compile_to_ir(
            'colophon fn str_concat(a: str, b: str) -> str\n'
            'fn main() -> i32 { 0 }\n'
        )


def test_colophon_matching_signature_shares_internal_extern(tmp_path):
    # A user declaring `write` with the same signature the compiler
    # uses internally shares the single LLVM extern — both the user's
    # calls and `println`'s `write(2)` go through the same symbol
    # without redeclaration issues.
    src = (
        'colophon fn write(fd: i32, buf: str, n: i64) -> i64\n'
        'fn main() -> i32 {\n'
        '  println("hello from tuppu")\n'
        '  write(1 as i32, "direct syscall\\n", 15 as i64)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"hello from tuppu\ndirect syscall\n"


def test_colophon_signature_mismatch_with_internal_errors():
    # User redeclares `write` with a wrong signature — previously this
    # silently reused the internal extern and miscalled it. Now we
    # refuse with a clear error pointing at the collision.
    with pytest.raises(CompileError, match="compiler needs extern"):
        compile_to_ir(
            'colophon fn write(x: i32) -> i32\n'
            'fn main() -> i32 {\n'
            '  println("trigger write extern")\n'
            '  0\n'
            '}\n'
        )
