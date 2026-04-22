"""Variable-length strings as the built-in `str` seal.

Covers: string literal → str tablet value, print dispatch via %.*s, field
access `s.len`, byte indexing `s[i]`, passing strings as function args,
and the stdlib helpers in `stdlib/str.tpu`."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import (
    compile_files_to_binary,
    compile_to_binary,
    compile_to_ir,
    stdlib_files,
)
from tuppu.errors import CompileError


def run(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return r.returncode, r.stdout, r.stderr


def run_with_stdlib(src: str, tmp_path: Path) -> tuple[int, bytes, bytes]:
    user_file = tmp_path / "main.tpu"
    user_file.write_text(src)
    binary = compile_files_to_binary(
        stdlib_files() + [user_file], tmp_path, name="prog",
    )
    r = subprocess.run([str(binary)], capture_output=True)
    return r.returncode, r.stdout, r.stderr


# --- literal, print, len, index -------------------------------------------

def test_string_literal_print(tmp_path):
    _, out, _ = run('fn main() -> i32 { println("hello")\n 0 }', tmp_path)
    assert out == b"hello\n"


def test_print_vs_println(tmp_path):
    _, out, _ = run(
        'fn main() -> i32 { print("one ")\n print("two ")\n println("three")\n 0 }',
        tmp_path,
    )
    assert out == b"one two three\n"


def test_s_dot_len(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  step s = "hello"\n'
        '  println(s.len)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"5\n"


def test_empty_string_len_and_print(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  step s = ""\n'
        '  println(s.len)\n'
        '  print(s)\n'
        '  println("done")\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\ndone\n"


def test_s_at_index_returns_u8(tmp_path):
    # "hello"[1] == 'e' (0x65 = 101)
    src = (
        'fn main() -> i32 {\n'
        '  step s = "hello"\n'
        '  println(s[1] as i64)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"101\n"


def test_string_index_bounds_trap(tmp_path):
    # Reading past the end should trap (SIGILL via llvm.trap on macOS).
    src = (
        'fn main() -> i32 {\n'
        '  step s = "hi"\n'
        '  println(s[2] as i64)\n'   # out of bounds
        '  0\n'
        '}\n'
    )
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], capture_output=True)
    assert r.returncode != 0, "expected trap on OOB index"


# --- escape sequences survive the redesign --------------------------------

def test_escapes_newline_and_tab(tmp_path):
    src = 'fn main() -> i32 { println("a\\tb\\nc")\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"a\tb\nc\n"


def test_escapes_with_internal_nul(tmp_path):
    # %.*s respects the length field and must print past an embedded NUL.
    src = 'fn main() -> i32 { print("ab\\0cd")\n println("!")\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"ab\x00cd!\n"


# --- strings as function parameters and return values ---------------------

def test_string_as_parameter(tmp_path):
    src = (
        'fn greet(name: str) {\n'
        '  print("hi, ")\n'
        '  println(name)\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  greet("Drew")\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"hi, Drew\n"


def test_string_as_return_value(tmp_path):
    src = (
        'fn label(tag: i64) -> str {\n'
        '  if tag == 0 { "zero" } else { "nonzero" }\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  println(label(0))\n'
        '  println(label(7))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"zero\nnonzero\n"


# --- stdlib helpers --------------------------------------------------------

def test_stdlib_str_eq(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(str_eq("foo", "foo"))\n'
        '  println(str_eq("foo", "bar"))\n'
        '  println(str_eq("foo", "foobar"))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"true\nfalse\nfalse\n"


def test_stdlib_starts_with(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(str_starts_with("hello, world", "hello"))\n'
        '  println(str_starts_with("hello", "hi"))\n'
        '  println(str_starts_with("hi", "hello"))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"true\nfalse\nfalse\n"


def test_stdlib_ends_with(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(str_ends_with("hello.tpu", ".tpu"))\n'
        '  println(str_ends_with("hello.tpu", ".py"))\n'
        '  println(str_ends_with("hi", "hello"))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"true\nfalse\nfalse\n"


def test_stdlib_index_of(tmp_path):
    # 'l' = 108
    src = (
        'fn main() -> i32 {\n'
        '  println(str_index_of("hello", 108 as u8))\n'   # first 'l' at 2
        '  println(str_index_of("hello", 122 as u8))\n'   # 'z' absent
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"2\n-1\n"


def test_stdlib_is_empty(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println(str_is_empty(""))\n'
        '  println(str_is_empty("x"))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"true\nfalse\n"


# --- error cases -----------------------------------------------------------

def test_bracket_array_type_rejected():
    # The old `[N]u8` shorthand is retired; users should write `str`.
    with pytest.raises(CompileError, match="array types are not supported"):
        compile_to_ir(
            "fn f(s: [5]u8) -> i32 { 0 }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_str_index_needs_integer():
    with pytest.raises(CompileError, match="str index must be integer"):
        compile_to_ir(
            'fn main() -> i32 {\n'
            '  step s = "hi"\n'
            '  println(s[true] as i64)\n'
            '  0\n'
            '}\n'
        )
