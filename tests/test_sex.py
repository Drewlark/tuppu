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

def test_sex_plus_sex_native(tmp_path, capsys):
    # Phase 2: sex + sex stays in digit form, no warning, no rat lowering.
    # 1;30 + 0;20 = 1;50.
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
    assert out == b"1;50\n"
    assert "warning" not in captured.err.lower()


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


def test_native_sex_add_same_radix(tmp_path, capsys):
    # 1;30 + 0;20 = 1;50 (base 60: 30 + 20 = 50, no carry)
    src = 'fn main() -> i32 { println((1;30) + (0;20))\n 0 }'
    _, out, _ = run(src, tmp_path)
    captured = capsys.readouterr()
    assert out == b"1;50\n"
    assert "warning" not in captured.err.lower()


def test_native_sex_add_with_carry(tmp_path):
    # 0;40 + 0;30 = 1;10 (fractional 40 + 30 = 70, carry one into int place)
    src = 'fn main() -> i32 { println((0;40) + (0;30))\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"1;10\n"


def test_native_sex_add_cascading_carry(tmp_path):
    # 0;59 + 0;1 = 1;0 — carry ripples to new integer position.
    src = 'fn main() -> i32 { println((0;59) + (0;1))\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"1;0\n"


def test_native_sex_add_misaligned_radix(tmp_path):
    # 1 30 (integer form, =90) + 0;20 (fractional, =1/3) = 1 30;20
    # In base 60: int part [1, 30], frac part [20] → `1 30;20`.
    src = 'fn main() -> i32 { println((1 30) + (0;20))\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"1 30;20\n"


def test_native_sex_add_misaligned_frac(tmp_path):
    # 1;30 + 0;0 20 — pad a with trailing zero in frac: 1;30 0 + 0;0 20 = 1;30 20
    src = 'fn main() -> i32 { println((1;30) + (0;0 20))\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"1;30 20\n"


def test_native_sex_sub_same_sign(tmp_path, capsys):
    # 1;30 - 0;20 = 1;10 (base 60: 30 - 20 = 10)
    src = 'fn main() -> i32 { println((1;30) - (0;20))\n 0 }'
    _, out, _ = run(src, tmp_path)
    captured = capsys.readouterr()
    assert out == b"1;10\n"
    assert "warning" not in captured.err.lower()


def test_native_sex_sub_with_borrow(tmp_path):
    # 1;10 - 0;20 → fractional 10 < 20, borrow 1 from int.
    # 1;10 - 0;20 = 0;50
    src = 'fn main() -> i32 { println((1;10) - (0;20))\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"0;50\n"


def test_native_sex_add_mixed_sign_via_sub(tmp_path):
    # a + (-b) where b > a → result has b's sign (negative).
    # 1;30 + (-(1;40)) → magnitudes 1;30 vs 1;40, second larger, result = -(0;10)
    src = 'fn main() -> i32 { println((1;30) + -(1;40))\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"-0;10\n"


def test_native_sex_sub_underflow_flips_sign(tmp_path):
    # 0;20 - 0;30 = -0;10
    src = 'fn main() -> i32 { println((0;20) - (0;30))\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"-0;10\n"


def test_native_sex_ybc_plus_small(tmp_path):
    # 1;24 51 10 + 0;0 0 5 = 1;24 51 15 (add at smallest place)
    src = 'fn main() -> i32 { println((1;24 51 10) + (0;0 0 5))\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"1;24 51 15\n"


def test_native_sex_round_trip_via_rat(tmp_path):
    # Verify that a + b digit-form matches what rat arithmetic would give.
    src = (
        "fn main() -> i32 {\n"
        "  step s = (1;30) + (0;20)\n"
        "  println(s)\n"           # 1;50
        "  println(s as rat)\n"    # 11/6
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1;50\n11/6\n"


def test_int_to_sex_zero(tmp_path):
    # An int literal in a sex context decomposes to digit form.
    src = (
        "fn main() -> i32 {\n"
        "  step s: sex = 0\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\n"


def test_int_to_sex_small(tmp_path):
    # 42 is less than 60, so it's one digit.
    src = (
        "fn main() -> i32 {\n"
        "  step s: sex = 42\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"42\n"


def test_int_to_sex_two_digits(tmp_path):
    # 123 = 2*60 + 3 → `2 3`
    src = (
        "fn main() -> i32 {\n"
        "  step s: sex = 123\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"2 3\n"


def test_int_to_sex_negative(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  step s: sex = -123\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"-2 3\n"


def test_int_to_sex_in_struct_field(tmp_path):
    # The scratch-file motivating case: integer literal at a sex field
    # slot should decompose via int→sex, not error out.
    src = (
        "seal Point { x: sex, y: sex }\n"
        "fn main() -> i32 {\n"
        "  step p = Point { x: 0, y: 0 }\n"
        "  println(p.x)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\n"


def test_rat_to_sex_coerces_regular(tmp_path):
    # rat → sex is now supported for regular numbers (den = 2^a·3^b·5^c).
    # Storing a rat in a sex-typed field should reconstruct the Babylonian
    # digit form and print 1;30.
    src = (
        "seal P { x: sex }\n"
        "fn main() -> i32 {\n"
        "  step r: rat = rat(3, 2)\n"
        "  step p = P { x: r }\n"
        "  println(p.x)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1;30\n"


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


# --- Phase 3a: native sex * and rat → sex with regularity check -----------

def test_rat_to_sex_pure_integer(tmp_path):
    # rat(5, 1) → sex is integer-form 5, prints "5".
    src = (
        "fn main() -> i32 {\n"
        "  step r: rat = rat(5, 1)\n"
        "  step s: sex = r\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"5\n"


def test_rat_to_sex_pure_fraction(tmp_path):
    # 1/3 → 0;20 (single leading-zero int digit).
    src = (
        "fn main() -> i32 {\n"
        "  step r: rat = rat(1, 3)\n"
        "  step s: sex = r\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0;20\n"


def test_rat_to_sex_multi_frac(tmp_path):
    # 1/8 = 0;7 30. (den = 8 = 2^3; rem path emits 7 then 30.)
    src = (
        "fn main() -> i32 {\n"
        "  step r: rat = rat(1, 8)\n"
        "  step s: sex = r\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0;7 30\n"


def test_rat_to_sex_negative(tmp_path):
    # -3/2 → -1;30. Sign byte propagates.
    src = (
        "fn main() -> i32 {\n"
        "  step r: rat = rat(0, 1) - rat(3, 2)\n"
        "  step s: sex = r\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"-1;30\n"


def test_rat_to_sex_nonregular_traps(tmp_path):
    # 1/7 is not regular (7 isn't 2^a·3^b·5^c). Runtime trap.
    src = (
        "fn main() -> i32 {\n"
        "  step r: rat = rat(1, 7)\n"
        "  step s: sex = r\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    rc, _, _ = run(src, tmp_path)
    assert rc != 0


def test_sex_mul_native(tmp_path):
    # 3/2 · 1/3 = 1/2 → 0;30 in digit form, no warning.
    src = (
        "fn main() -> i32 {\n"
        "  println(1;30 * 0;20)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0;30\n"


def test_sex_mul_integer_form(tmp_path):
    # 1 2 (= 62) · 1 3 (= 63) = 3906 = 1 5 6 in base 60.
    src = (
        "fn main() -> i32 {\n"
        "  println(1 2 * 1 3)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1 5 6\n"


def test_sex_mul_no_warning(tmp_path):
    # sex*sex now stays in digit form — no Phase 2-style warning anymore.
    src_path = tmp_path / "mul.tpu"
    src_path.write_text(
        "fn main() -> i32 {\n"
        "  println(1;30 * 0;20)\n"
        "  0\n"
        "}\n"
    )
    r = subprocess.run(
        [sys.executable, "-m", "tuppu", "run", str(src_path), "--no-stdlib"],
        capture_output=True,
    )
    assert r.returncode == 0
    assert b"warning" not in r.stderr.lower()


def test_sex_mul_round_trip_via_int(tmp_path):
    # Integer sex times integer sex should equal the i64 product. 7·8=56,
    # which in base-60 is a single digit "56".
    src = (
        "fn main() -> i32 {\n"
        "  step a: sex = 7\n"
        "  step b: sex = 8\n"
        "  step c = a * b\n"
        "  println(c)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"56\n"


# --- CLI emits warnings to stderr ------------------------------------------

def test_cli_sex_div_warning(tmp_path):
    # Division still lowers to rat and warns. (Multiplication now stays
    # in digit form — see test_sex_mul_native.)
    src = tmp_path / "warn.tpu"
    src.write_text(
        "fn main() -> i32 {\n"
        "  step a = 1;30\n"
        "  step b = 0;20\n"
        "  println(a / b)\n"
        "  0\n"
        "}\n"
    )
    r = subprocess.run(
        [sys.executable, "-m", "tuppu", "run", str(src), "--no-stdlib"],
        capture_output=True,
    )
    assert r.returncode == 0
    assert r.stdout == b"9/2\n"
    assert b"warning" in r.stderr.lower()
