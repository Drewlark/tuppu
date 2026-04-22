"""Tests for the built-in I/O intrinsics: print, println, read_int.

Each test compiles a source snippet, runs the binary in a subprocess,
and asserts on its stdout (and optionally stdin pipe)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.codegen import CodegenError
from tuppu.errors import CompileError
from tuppu.driver import compile_to_binary, compile_to_ir


def run(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    result = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return result.returncode, result.stdout


# --- print / println -------------------------------------------------------

def test_println_integer(tmp_path):
    _, out = run('fn main() -> i32 { println(42)\n 0 }', tmp_path)
    assert out == b"42\n"


def test_print_no_newline(tmp_path):
    _, out = run('fn main() -> i32 { print(42)\n print(7)\n 0 }', tmp_path)
    assert out == b"427"


def test_println_string(tmp_path):
    _, out = run('fn main() -> i32 { println("hello")\n 0 }', tmp_path)
    assert out == b"hello\n"


def test_print_multiple_strings(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  print("a")\n'
        '  print("bc")\n'
        '  println("def")\n'
        '  0\n'
        '}\n'
    )
    _, out = run(src, tmp_path)
    assert out == b"abcdef\n"


def test_print_bool_true(tmp_path):
    _, out = run('fn main() -> i32 { println(true)\n 0 }', tmp_path)
    assert out == b"true\n"


def test_print_bool_false(tmp_path):
    _, out = run('fn main() -> i32 { println(false)\n 0 }', tmp_path)
    assert out == b"false\n"


def test_print_computed_comparison(tmp_path):
    _, out = run('fn main() -> i32 { println(3 < 5)\n 0 }', tmp_path)
    assert out == b"true\n"


def test_print_negative_integer(tmp_path):
    _, out = run('fn main() -> i32 { println(-17)\n 0 }', tmp_path)
    assert out == b"-17\n"


def test_string_dedup_emits_one_global():
    ir = compile_to_ir(
        'fn main() -> i32 { println("hi")\n println("hi")\n 0 }'
    )
    # Identical strings reuse the same .str.N global.
    assert ir.count('constant [3 x i8]') == 1


# --- read_int --------------------------------------------------------------

def test_read_int_doubles(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  step n = read_int()\n"
        "  println(n * 2)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path, stdin=b"21\n")
    assert out == b"42\n"


def test_read_int_then_compute(tmp_path):
    # Read two numbers, print their sum.
    src = (
        "fn main() -> i32 {\n"
        "  step a = read_int()\n"
        "  step b = read_int()\n"
        "  println(a + b)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path, stdin=b"10\n32\n")
    assert out == b"42\n"


def test_read_int_used_in_conditional(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  step n = read_int()\n"
        "  if n > 0 { println(\"positive\") } else { println(\"non-positive\") }\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path, stdin=b"7\n")
    assert out == b"positive\n"
    _, out = run(src, tmp_path, stdin=b"-1\n")
    assert out == b"non-positive\n"


# --- interactive programs that use everything ------------------------------

def test_factorial_of_input(tmp_path):
    src = (
        "fn fact(n: i64) -> i64 {\n"
        "  if n < 2 { 1 } else { n * fact(n - 1) }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  print(\"n = \")\n"
        "  step n = read_int()\n"
        "  print(\"n! = \")\n"
        "  println(fact(n))\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path, stdin=b"6\n")
    assert out == b"n = n! = 720\n"


# --- error cases -----------------------------------------------------------

def test_cannot_define_print():
    with pytest.raises(CompileError, match="built-in intrinsic"):
        compile_to_ir(
            "fn print(n: i64) { }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_cannot_define_read_int():
    with pytest.raises(CompileError, match="built-in intrinsic"):
        compile_to_ir(
            "fn read_int() -> i64 { 0 }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_print_wrong_arity_errors():
    with pytest.raises(CompileError, match="exactly one argument"):
        compile_to_ir("fn main() -> i32 { println(1, 2)\n 0 }")


def test_read_int_takes_no_args():
    with pytest.raises(CompileError, match="takes no arguments"):
        compile_to_ir("fn main() -> i32 { step n = read_int(42)\n 0 }")
