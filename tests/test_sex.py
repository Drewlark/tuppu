"""The sex/dish type — Babylonian digit-sequence literals.

Covers: space-separated lexing, radix semantics, coercion, explicit casts,
arithmetic warnings when sex is mixed with rat, and the YBC 7289 √2
approximation as an end-to-end test.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tuppu.driver import compile_to_binary, compile_to_ir
from tuppu.errors import CompileError
from tuppu.typecheck import CheckError


def run(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return r.returncode, r.stdout, r.stderr


# --- syntax + print ---------------------------------------------------------

def test_sex_prints_babylonian(tmp_path):
    # After the Phase 1 redesign sex prints in its own notation, not the
    # rat-reduced form. `1;30` is displayed `1;30`, preserving the digits.
    _, out, _ = run('fn main() -> i32 { println(1;30)\n 0 }', tmp_path)
    assert out == b"1;30\n"


def test_sex_coerced_to_rat_prints_rat(tmp_path):
    # Binding to `: rat` forces the conversion; now we see the reduced form.
    src = (
        "fn main() -> i32 {\n"
        "  step r: rat = 1;30\n"
        "  println(r)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"3/2\n"


def test_sex_integer_form_via_spaces(tmp_path):
    # `1 30 as i64` yields 90, which exits the process.
    src = "fn main() -> i32 { (1 30 as i64) as i32 }"
    binary = compile_to_binary(src, tmp_path, name="prog")
    assert subprocess.run([str(binary)]).returncode == 90


def test_ybc_7289_sqrt2(tmp_path):
    # YBC 7289: the Babylonian √2 approximation baked into a cuneiform tablet.
    # Printing as sex preserves the scribal digits; casting to rat reduces.
    src = (
        "fn main() -> i32 {\n"
        "  step s = 1;24 51 10\n"
        "  println(s)\n"
        "  println(s as rat)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1;24 51 10\n30547/21600\n"


def test_spaces_around_semicolon_all_work(tmp_path):
    for src_expr in ("1;30", "1; 30", "1 ; 30", "1 ;30"):
        src = f"fn main() -> i32 {{ println({src_expr})\n 0 }}"
        _, out, _ = run(src, tmp_path)
        assert out == b"1;30\n", f"{src_expr!r} produced {out!r}"


# --- dish as a type name ---------------------------------------------------

def test_dish_type_name_works(tmp_path):
    src = (
        "fn identity(x: dish) -> dish { x }\n"
        "fn main() -> i32 { println(identity(1;30))\n 0 }\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1;30\n"


def test_sex_and_dish_are_same_type(tmp_path):
    # A fn takes `sex`, caller supplies value via `dish` context — both
    # work since they're aliases.
    src = (
        "fn identity(x: sex) -> sex { x }\n"
        "fn main() -> i32 {\n"
        "  step d: dish = 1;30\n"
        "  println(identity(d))\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1;30\n"


# --- coercion rules --------------------------------------------------------

def test_sex_coerces_silently_to_rat(tmp_path):
    # Binding at `: rat` accepts a sex literal with no cast, no warning.
    src = (
        "fn main() -> i32 {\n"
        "  step x: rat = 1;30\n"
        "  println(x)\n"
        "  0\n"
        "}\n"
    )
    _, out, err = run(src, tmp_path)
    assert out == b"3/2\n"
    assert b"warning" not in err.lower()


def test_sex_coerces_silently_to_i64(tmp_path):
    # Integer-form sex assigned to an i64 binding — truncates via sdiv.
    src = (
        "fn main() -> i32 {\n"
        "  step x: i64 = 1 30\n"       # 90
        "  println(x)\n"
        "  0\n"
        "}\n"
    )
    _, out, err = run(src, tmp_path)
    assert out == b"90\n"
    assert b"warning" not in err.lower()


def test_sex_to_i64_truncates_fractional(tmp_path):
    # 0;20 = 1/3 → truncates to 0.
    src = (
        "fn main() -> i32 {\n"
        "  step x: i64 = 0;20\n"
        "  println(x)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\n"


# --- arithmetic warning ----------------------------------------------------

def test_sex_plus_sex_warns_and_lowers_to_rat(tmp_path, capsys):
    # The warning is emitted by the compiler (parent process), not the
    # compiled binary — so we capture via capsys, not subprocess stderr.
    src = (
        "fn main() -> i32 {\n"
        "  step a = 1;30\n"
        "  step b = 0;20\n"
        "  println(a + b)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    captured = capsys.readouterr()
    assert out == b"11/6\n"
    assert "warning" in captured.err.lower()
    assert "sex arithmetic" in captured.err.lower() or "native sex" in captured.err.lower()


def test_sex_comparison_does_not_warn(tmp_path, capsys):
    src = (
        "fn main() -> i32 {\n"
        "  step a = 1;30\n"
        "  step b = 0;20\n"
        "  println(a > b)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    captured = capsys.readouterr()
    assert out == b"true\n"
    assert "warning" not in captured.err.lower()


def test_explicit_cast_then_rat_arithmetic_no_warning(tmp_path, capsys):
    # Once the user casts to rat, they've opted in — no warning.
    src = (
        "fn main() -> i32 {\n"
        "  step a = (1;30) as rat\n"
        "  step b = (0;20) as rat\n"
        "  println(a + b)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    captured = capsys.readouterr()
    assert out == b"11/6\n"
    assert "warning" not in captured.err.lower()


# --- error cases -----------------------------------------------------------

def test_sex_as_f64_errors():
    with pytest.raises(CompileError, match="not yet supported"):
        compile_to_ir("fn main() -> i32 { (1;30) as f64\n 0 }")


def test_two_semicolons_errors():
    with pytest.raises(CompileError, match="two sexagesimal radix points"):
        compile_to_ir("fn main() -> i32 { println(1;30;45)\n 0 }")


def test_sex_place_over_59_errors():
    with pytest.raises(CompileError, match="must be < 60"):
        compile_to_ir("fn main() -> i32 { println(1 60)\n 0 }")


# --- digit-form runtime (Phase 1 redesign) ---------------------------------

def test_sex_preserves_trailing_fractional_zeros(tmp_path):
    # 1;30 0 0 still reduces to 3/2 as a rat, but the sex view keeps the
    # scribe's digits.
    src = (
        "fn main() -> i32 {\n"
        "  step s = 1;30 0 0\n"
        "  println(s)\n"
        "  println(s as rat)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1;30 0 0\n3/2\n"


def test_sex_integer_form_prints_with_spaces(tmp_path):
    src = 'fn main() -> i32 { println(1 30)\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"1 30\n"


def test_sex_negation_flips_sign(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  step s = -(1;30)\n"
        "  println(s)\n"
        "  println(s as rat)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"-1;30\n-3/2\n"


def test_double_negation_sex(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  step s = -(-(1;30))\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1;30\n"


def test_sex_to_int_truncates_via_rat(tmp_path):
    # 1;30 as i64 = 1 (truncates 3/2)
    src = (
        "fn main() -> i32 {\n"
        "  step n: i64 = 1;30\n"
        "  println(n)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1\n"


def test_sex_and_rat_print_differently(tmp_path):
    # Regression: sex is no longer structurally identical to rat, so
    # print dispatches on the distinct type.
    src = (
        'fn main() -> i32 {\n'
        '  step s: sex = 1;30\n'
        '  step r: rat = 1;30\n'
        '  println(s)\n'
        '  println(r)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1;30\n3/2\n"


def test_sex_as_function_param_and_return(tmp_path):
    # Passing sex through a function preserves the digit sequence.
    src = (
        "fn identity(x: dish) -> dish { x }\n"
        "fn main() -> i32 {\n"
        "  println(identity(1;24 51 10))\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1;24 51 10\n"


# --- CLI emits warnings to stderr ------------------------------------------

def test_cli_sex_arithmetic_warning(tmp_path):
    src = tmp_path / "warn.tpu"
    src.write_text(
        "fn main() -> i32 {\n"
        "  step a = 1;30\n"
        "  step b = 0;20\n"
        "  println(a + b)\n"
        "  0\n"
        "}\n"
    )
    r = subprocess.run(
        [sys.executable, "-m", "tuppu", "run", str(src), "--no-stdlib"],
        capture_output=True,
    )
    assert r.returncode == 0
    assert r.stdout == b"11/6\n"
    assert b"warning" in r.stderr.lower()
