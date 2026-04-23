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

def test_colophon_rejects_struct_arg():
    # User tablets in colophon signatures will come in the follow-up
    # landing (for sockaddr_in etc.); first cut only marshalS ints,
    # bool, and str.
    with pytest.raises(CompileError, match="isn't marshalable"):
        compile_to_ir(
            'tablet Point { x: i64, y: i64 }\n'
            'colophon fn weird(p: Point) -> i64\n'
            'fn main() -> i32 { 0 }\n'
        )


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
