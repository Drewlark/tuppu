"""Tests for rat (rational) type and sexagesimal arithmetic."""
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


# --- literals (sexagesimal) ------------------------------------------------

def test_sex_literal_one_and_a_half(tmp_path):
    # print(sex) now renders in Babylonian notation; cast to rat for the
    # reduced-fraction form these tests are checking.
    _, out = run('fn main() -> i32 { println((1;30) as rat)\n 0 }', tmp_path)
    assert out == b"3/2\n"


def test_sex_literal_one_third_exact(tmp_path):
    # 0;20 = 1/3 exactly — cannot be represented in f64.
    _, out = run('fn main() -> i32 { println((0;20) as rat)\n 0 }', tmp_path)
    assert out == b"1/3\n"


def test_sex_literal_tiny(tmp_path):
    # 0;0 45 = 45/3600 = 1/80 (space-separator convention)
    _, out = run('fn main() -> i32 { println((0;0 45) as rat)\n 0 }', tmp_path)
    assert out == b"1/80\n"


def test_sex_literal_reduces_at_parse_time(tmp_path):
    # 1;30 0 0 — trailing fractional zeros still reduce to 3/2 via `as rat`.
    _, out = run('fn main() -> i32 { println((1;30 0 0) as rat)\n 0 }', tmp_path)
    assert out == b"3/2\n"


# --- constructor -----------------------------------------------------------

def test_rat_constructor_reduces(tmp_path):
    _, out = run('fn main() -> i32 { println(rat(6, 4))\n 0 }', tmp_path)
    assert out == b"3/2\n"


def test_rat_constructor_normalizes_sign(tmp_path):
    # den < 0 should flip signs so den > 0.
    _, out = run('fn main() -> i32 { println(rat(1, -2))\n 0 }', tmp_path)
    assert out == b"-1/2\n"


def test_rat_constructor_zero(tmp_path):
    # rat(0, anything) should be 0/1 after reduction.
    _, out = run('fn main() -> i32 { println(rat(0, 7))\n 0 }', tmp_path)
    assert out == b"0/1\n"


def test_rat_constructor_zero_den_traps(tmp_path):
    # rat(_, 0) must abort at runtime.
    src = "fn main() -> i32 { println(rat(1, 0))\n 0 }"
    binary = compile_to_binary(src, tmp_path, name="prog")
    result = subprocess.run([str(binary)], capture_output=True)
    # llvm.trap() on macOS terminates with SIGILL (exit code -4 via Popen
    # convention, or 132+signo under sh). Just assert nonzero exit.
    assert result.returncode != 0


# --- arithmetic ------------------------------------------------------------

def test_rat_add(tmp_path):
    # 3/2 + 1/3 = 11/6 (LCD = 6; 9 + 2 = 11). Cast to rat because sex+sex
    # now stays in digit form — we're specifically testing rat arithmetic.
    _, out = run(
        'fn main() -> i32 { println((1;30) as rat + (0;20) as rat)\n 0 }',
        tmp_path,
    )
    assert out == b"11/6\n"


def test_rat_sub(tmp_path):
    # 3/2 - 1/3 = 7/6
    _, out = run(
        'fn main() -> i32 { println((1;30) as rat - (0;20) as rat)\n 0 }',
        tmp_path,
    )
    assert out == b"7/6\n"


def test_rat_mul(tmp_path):
    # 3/2 * 1/3 = 1/2
    _, out = run('fn main() -> i32 { println(1;30 * 0;20)\n 0 }', tmp_path)
    assert out == b"1/2\n"


def test_rat_div(tmp_path):
    # 3/2 / 1/3 = 9/2
    _, out = run('fn main() -> i32 { println(1;30 / 0;20)\n 0 }', tmp_path)
    assert out == b"9/2\n"


def test_rat_arithmetic_reduces_result(tmp_path):
    # 1/4 + 1/4 = 2/4 = 1/2 after reduction
    _, out = run('fn main() -> i32 { println(rat(1, 4) + rat(1, 4))\n 0 }', tmp_path)
    assert out == b"1/2\n"


# --- comparisons -----------------------------------------------------------

def test_rat_eq(tmp_path):
    _, out = run('fn main() -> i32 { println(1;30 == rat(3, 2))\n 0 }', tmp_path)
    assert out == b"true\n"


def test_rat_ne(tmp_path):
    _, out = run('fn main() -> i32 { println(1;30 != rat(1, 2))\n 0 }', tmp_path)
    assert out == b"true\n"


def test_rat_lt(tmp_path):
    # 1/3 < 1/2
    _, out = run('fn main() -> i32 { println(rat(1, 3) < rat(1, 2))\n 0 }', tmp_path)
    assert out == b"true\n"


def test_rat_gt(tmp_path):
    _, out = run('fn main() -> i32 { println(rat(2, 3) > rat(1, 2))\n 0 }', tmp_path)
    assert out == b"true\n"


def test_rat_le(tmp_path):
    _, out = run(
        'fn main() -> i32 { println(rat(1, 2) <= rat(1, 2))\n 0 }', tmp_path
    )
    assert out == b"true\n"


# --- field access ----------------------------------------------------------

def test_rat_num_field(tmp_path):
    _, out = run('fn main() -> i32 { println((1;30).num)\n 0 }', tmp_path)
    assert out == b"3\n"


def test_rat_den_field(tmp_path):
    _, out = run('fn main() -> i32 { println((1;30).den)\n 0 }', tmp_path)
    assert out == b"2\n"


# --- casts -----------------------------------------------------------------

def test_int_to_rat(tmp_path):
    _, out = run('fn main() -> i32 { println(7 as rat)\n 0 }', tmp_path)
    assert out == b"7/1\n"


def test_rat_to_int_truncates(tmp_path):
    # 7/2 = 3.5 → 3 (truncation toward zero)
    _, out = run(
        'fn main() -> i32 { println(rat(7, 2) as i64)\n 0 }', tmp_path
    )
    assert out == b"3\n"


def test_rat_to_int_negative(tmp_path):
    # -7/2 = -3.5 → -3 (truncation toward zero, not floor)
    _, out = run(
        'fn main() -> i32 { println(rat(-7, 2) as i64)\n 0 }', tmp_path
    )
    assert out == b"-3\n"


# --- user functions can take and return rat --------------------------------

def test_rat_as_parameter_and_return(tmp_path):
    src = (
        "fn rat_reciprocal(x: rat) -> rat { rat(x.den, x.num) }\n"
        "fn main() -> i32 {\n"
        "  println(rat_reciprocal(1;30))\n"  # 3/2 → 2/3
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"2/3\n"


def test_rat_absolute_value(tmp_path):
    src = (
        "fn rat_abs(x: rat) -> rat {\n"
        "  if x.num < 0 { rat(-x.num, x.den) } else { x }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  println(rat_abs(rat(-5, 3)))\n"
        "  println(rat_abs(rat(5, 3)))\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"5/3\n5/3\n"


# --- error cases -----------------------------------------------------------

def test_cannot_define_rat():
    # `rat` is a type keyword, so the parser rejects it as a function name
    # before codegen ever sees it. Either error message is acceptable.
    from tuppu.parser import ParseError
    with pytest.raises((ParseError, CodegenError)):
        compile_to_ir(
            "fn rat(a: i64, b: i64) -> i64 { a + b }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_rat_ctor_wrong_arity():
    with pytest.raises(CompileError, match="exactly two arguments"):
        compile_to_ir("fn main() -> i32 { step r = rat(1)\n 0 }")


def test_rat_unknown_field():
    with pytest.raises(CompileError, match="no field"):
        compile_to_ir(
            "fn main() -> i32 { step x: rat = 1;30\n println(x.foo)\n 0 }"
        )