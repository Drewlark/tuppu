"""End-to-end codegen tests: compile Tuppu source to a native ARM64 binary
and check its exit code. Each test takes ~100ms (llvmlite + clang link)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.codegen import CodegenError
from tuppu.errors import CompileError
from tuppu.driver import compile_to_binary, compile_to_ir


def run(source: str, tmp_path: Path) -> int:
    binary = compile_to_binary(source, tmp_path, name="prog")
    return subprocess.run([str(binary)]).returncode


# --- integer arithmetic -----------------------------------------------------

def test_returns_literal(tmp_path):
    assert run("fn main() -> i32 { 0 }", tmp_path) == 0


def test_returns_forty_two(tmp_path):
    assert run("fn main() -> i32 { 42 }", tmp_path) == 42


def test_addition(tmp_path):
    assert run("fn main() -> i32 { 1 + 2 + 3 }", tmp_path) == 6


def test_multiplication(tmp_path):
    assert run("fn main() -> i32 { 6 * 7 }", tmp_path) == 42


def test_precedence(tmp_path):
    # 2 + 3 * 4 = 14 (not 20)
    assert run("fn main() -> i32 { 2 + 3 * 4 }", tmp_path) == 14


def test_parentheses_override(tmp_path):
    assert run("fn main() -> i32 { (2 + 3) * 4 }", tmp_path) == 20


def test_subtraction(tmp_path):
    assert run("fn main() -> i32 { 100 - 58 }", tmp_path) == 42


def test_signed_division(tmp_path):
    assert run("fn main() -> i32 { 100 / 4 }", tmp_path) == 25


def test_remainder(tmp_path):
    assert run("fn main() -> i32 { 17 % 5 }", tmp_path) == 2


def test_unary_minus_in_expression(tmp_path):
    # -5 + 10 = 5
    assert run("fn main() -> i32 { -5 + 10 }", tmp_path) == 5


# --- booleans and comparisons -----------------------------------------------

def test_true_returns_one(tmp_path):
    assert run("fn main() -> i32 { true }", tmp_path) == 1


def test_false_returns_zero(tmp_path):
    assert run("fn main() -> i32 { false }", tmp_path) == 0


def test_comparison_true(tmp_path):
    assert run("fn main() -> i32 { 1 < 2 }", tmp_path) == 1


def test_comparison_false(tmp_path):
    assert run("fn main() -> i32 { 5 < 2 }", tmp_path) == 0


def test_equality(tmp_path):
    assert run("fn main() -> i32 { 42 == 42 }", tmp_path) == 1
    assert run("fn main() -> i32 { 42 != 42 }", tmp_path) == 0


def test_bool_and(tmp_path):
    assert run("fn main() -> i32 { true && true }", tmp_path) == 1
    assert run("fn main() -> i32 { true && false }", tmp_path) == 0


def test_bool_or(tmp_path):
    assert run("fn main() -> i32 { false || true }", tmp_path) == 1
    assert run("fn main() -> i32 { false || false }", tmp_path) == 0


def test_bool_not(tmp_path):
    assert run("fn main() -> i32 { !false }", tmp_path) == 1
    assert run("fn main() -> i32 { !true }", tmp_path) == 0


# --- IR-level sanity checks (fast; no binary build) -------------------------

def test_ir_contains_module_triple():
    ir = compile_to_ir("fn main() -> i32 { 0 }")
    assert "arm64-apple-darwin" in ir


def test_ir_has_trunc_for_i64_to_i32():
    ir = compile_to_ir("fn main() -> i32 { 42 }")
    assert "trunc i64" in ir and "to i32" in ir


def test_ir_uses_signed_comparison():
    ir = compile_to_ir("fn main() -> i32 { 1 < 2 }")
    assert "icmp slt" in ir


# --- error cases ------------------------------------------------------------

def test_main_missing_return_type_errors():
    with pytest.raises(CompileError, match="main must declare -> i32"):
        compile_to_ir("fn main() { 0 }")


# --- step bindings ----------------------------------------------------------

def test_step_binding(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  step x = 6\n"
        "  step y = 7\n"
        "  x * y\n"
        "}\n"
    )
    assert run(src, tmp_path) == 42


def test_step_binding_uses_ssa_not_alloca():
    """Verify step bindings lower to direct SSA values — no alloca/load/store.
    This is the core payoff of the step keyword from SPEC §5."""
    src = (
        "fn main() -> i32 {\n"
        "  step x = 10\n"
        "  step y = 4\n"
        "  x - y\n"
        "}\n"
    )
    ir = compile_to_ir(src)
    assert "alloca" not in ir, f"step bindings should not alloca:\n{ir}"
    assert "store" not in ir
    # The only load, if any, should not touch x/y — they're SSA-direct.


def test_step_binding_with_explicit_type(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  step x: i64 = 40\n"
        "  x + 2\n"
        "}\n"
    )
    assert run(src, tmp_path) == 42


def test_step_assignment_is_error():
    with pytest.raises(CompileError, match="cannot assign to step"):
        compile_to_ir(
            "fn main() -> i32 { step x = 1\n x = 2\n x }"
        )


# --- mut bindings -----------------------------------------------------------

def test_mut_binding_and_assign(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut x = 0\n"
        "  x = x + 42\n"
        "  x\n"
        "}\n"
    )
    assert run(src, tmp_path) == 42


def test_mut_multiple_assigns(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut total = 0\n"
        "  total = total + 10\n"
        "  total = total + 20\n"
        "  total = total + 12\n"
        "  total\n"
        "}\n"
    )
    assert run(src, tmp_path) == 42


def test_mut_uses_alloca():
    src = "fn main() -> i32 { mut x = 5\n x }"
    ir = compile_to_ir(src)
    assert "alloca" in ir
    assert "store" in ir
    assert "load" in ir


# --- user functions ---------------------------------------------------------

def test_simple_function_call(tmp_path):
    src = (
        "fn double(n: i64) -> i64 { n * 2 }\n"
        "fn main() -> i32 { double(21) }\n"
    )
    assert run(src, tmp_path) == 42


def test_function_with_multiple_params(tmp_path):
    src = (
        "fn add(a: i64, b: i64) -> i64 { a + b }\n"
        "fn main() -> i32 { add(19, 23) }\n"
    )
    assert run(src, tmp_path) == 42


def test_chained_function_calls(tmp_path):
    src = (
        "fn double(n: i64) -> i64 { n * 2 }\n"
        "fn triple(n: i64) -> i64 { n * 3 }\n"
        "fn main() -> i32 { double(triple(7)) }\n"
    )
    assert run(src, tmp_path) == 42


def test_forward_reference(tmp_path):
    """Functions declared later should be callable from earlier ones."""
    src = (
        "fn main() -> i32 { helper(42) }\n"
        "fn helper(x: i64) -> i64 { x }\n"
    )
    assert run(src, tmp_path) == 42


def test_function_with_step_bindings(tmp_path):
    # From SPEC §12.3 quadratic discriminant — b^2 - 4ac
    src = (
        "fn disc(a: i64, b: i64, c: i64) -> i64 {\n"
        "  step bsq = b * b\n"
        "  step ac = a * c\n"
        "  step four_ac = 4 * ac\n"
        "  bsq - four_ac\n"
        "}\n"
        "fn main() -> i32 { disc(1, 5, 6) }\n"  # 25 - 24 = 1
    )
    assert run(src, tmp_path) == 1


def test_function_with_mut_binding(tmp_path):
    src = (
        "fn sum_three(a: i64, b: i64, c: i64) -> i64 {\n"
        "  mut total = a\n"
        "  total = total + b\n"
        "  total = total + c\n"
        "  total\n"
        "}\n"
        "fn main() -> i32 { sum_three(10, 20, 12) }\n"
    )
    assert run(src, tmp_path) == 42


# --- integer width coercion -------------------------------------------------

def test_explicit_cast_narrows(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  step x: i64 = 300\n"
        "  x as i32\n"
        "}\n"
    )
    # 300 fits in i32; exit code on macOS truncates to u8 but 300 & 255 = 44
    assert run(src, tmp_path) == 44


def test_call_arg_coerces_widening():
    """If a function takes i64 and the literal is i64, no coercion needed.
    This test just verifies the IR is well-formed for the trivial case."""
    src = (
        "fn id(n: i64) -> i64 { n }\n"
        "fn main() -> i32 { id(7) }\n"
    )
    ir = compile_to_ir(src)
    assert "call i64 @\"id\"(i64 7)" in ir


# --- error cases for the new features ---------------------------------------

def test_unknown_function_errors():
    with pytest.raises(CompileError, match="unknown function"):
        compile_to_ir("fn main() -> i32 { foo(1) }")


def test_wrong_arity_errors():
    with pytest.raises(CompileError, match="expects 2 args"):
        compile_to_ir(
            "fn add(a: i64, b: i64) -> i64 { a + b }\n"
            "fn main() -> i32 { add(1) }\n"
        )


def test_undefined_name_errors():
    with pytest.raises(CompileError, match="undefined name"):
        compile_to_ir("fn main() -> i32 { x }")


def test_duplicate_function_errors():
    with pytest.raises(CompileError, match="duplicate"):
        compile_to_ir(
            "fn foo() -> i64 { 1 }\n"
            "fn foo() -> i64 { 2 }\n"
            "fn main() -> i32 { 0 }\n"
        )


# --- if/else ----------------------------------------------------------------

def test_if_true_branch(tmp_path):
    src = "fn main() -> i32 { if true { 42 } else { 0 } }"
    assert run(src, tmp_path) == 42


def test_if_false_branch(tmp_path):
    src = "fn main() -> i32 { if false { 0 } else { 42 } }"
    assert run(src, tmp_path) == 42


def test_if_with_condition(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  step x = 10\n"
        "  if x < 5 { 1 } else { 42 }\n"
        "}\n"
    )
    assert run(src, tmp_path) == 42


def test_if_else_if_chain(tmp_path):
    # grade classifier — returns 42 for x=85
    src = (
        "fn grade(x: i64) -> i64 {\n"
        "  if x >= 90 { 1 }\n"
        "  else if x >= 80 { 42 }\n"
        "  else if x >= 70 { 3 }\n"
        "  else { 4 }\n"
        "}\n"
        "fn main() -> i32 { grade(85) }\n"
    )
    assert run(src, tmp_path) == 42


def test_if_emits_phi():
    src = "fn main() -> i32 { if true { 1 } else { 2 } }"
    ir = compile_to_ir(src)
    assert "phi" in ir


# --- recursion (fact, fib) --------------------------------------------------

def test_recursive_factorial(tmp_path):
    # 5! = 120
    src = (
        "fn fact(n: i64) -> i64 {\n"
        "  if n < 2 { 1 } else { n * fact(n - 1) }\n"
        "}\n"
        "fn main() -> i32 { fact(5) }\n"
    )
    assert run(src, tmp_path) == 120


def test_recursive_fibonacci(tmp_path):
    # fib(10) = 55
    src = (
        "fn fib(n: i64) -> i64 {\n"
        "  if n < 2 { n } else { fib(n - 1) + fib(n - 2) }\n"
        "}\n"
        "fn main() -> i32 { fib(10) }\n"
    )
    assert run(src, tmp_path) == 55


def test_factorial_10_truncates_correctly(tmp_path):
    # 10! = 3628800. Exit code is (3628800 & 255) = 0
    src = (
        "fn fact(n: i64) -> i64 {\n"
        "  if n < 2 { 1 } else { n * fact(n - 1) }\n"
        "}\n"
        "fn main() -> i32 { fact(10) }\n"
    )
    assert run(src, tmp_path) == 0


# --- while loops ------------------------------------------------------------

def test_while_counts_down(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut i = 42\n"
        "  while i > 0 {\n"
        "    i = i - 1\n"
        "  }\n"
        "  i\n"
        "}\n"
    )
    assert run(src, tmp_path) == 0


def test_while_sum_to_n(tmp_path):
    # sum(0..9) = 45, then +13 = 58 — arbitrary to distinguish from zero
    src = (
        "fn sum_to(n: i64) -> i64 {\n"
        "  mut total = 0\n"
        "  mut i = 0\n"
        "  while i < n {\n"
        "    total = total + i\n"
        "    i = i + 1\n"
        "  }\n"
        "  total\n"
        "}\n"
        "fn main() -> i32 { sum_to(10) }\n"  # 0+1+...+9 = 45
    )
    assert run(src, tmp_path) == 45


def test_iterative_factorial(tmp_path):
    src = (
        "fn fact(n: i64) -> i64 {\n"
        "  mut result = 1\n"
        "  mut i = 1\n"
        "  while i <= n {\n"
        "    result = result * i\n"
        "    i = i + 1\n"
        "  }\n"
        "  result\n"
        "}\n"
        "fn main() -> i32 { fact(5) }\n"
    )
    assert run(src, tmp_path) == 120


def test_while_emits_loop_blocks():
    src = (
        "fn main() -> i32 {\n"
        "  mut i = 0\n"
        "  while i < 10 { i = i + 1 }\n"
        "  i\n"
        "}\n"
    )
    ir = compile_to_ir(src)
    assert "while.header" in ir
    assert "while.body" in ir
    assert "while.exit" in ir


# --- yield -----------------------------------------------------------------

def test_yield_early_return(tmp_path):
    src = (
        "fn first_multiple_of_seven(n: i64) -> i64 {\n"
        "  mut i = 1\n"
        "  while i < n {\n"
        "    if i % 7 == 0 { yield i }\n"
        "    i = i + 1\n"
        "  }\n"
        "  0\n"
        "}\n"
        "fn main() -> i32 { first_multiple_of_seven(100) }\n"
    )
    assert run(src, tmp_path) == 7


def test_yield_in_if_branch(tmp_path):
    src = (
        "fn sign(n: i64) -> i64 {\n"
        "  if n > 0 { yield 1 }\n"
        "  if n < 0 { yield 0 }\n"  # exit code 0 means negative here
        "  42\n"  # zero case — but any code past yields is unreachable when taken
        "}\n"
        "fn main() -> i32 { sign(5) }\n"
    )
    assert run(src, tmp_path) == 1


# --- error cases for control flow -------------------------------------------

def test_if_arms_different_types_errors():
    with pytest.raises(CompileError, match="different types"):
        compile_to_ir(
            "fn main() -> i32 { if true { 1 } else { true } }"
        )


def test_if_cond_must_be_bool():
    with pytest.raises(CompileError, match="must be bool"):
        compile_to_ir(
            "fn main() -> i32 { if 1 { 1 } else { 2 } }"
        )


def test_while_cond_must_be_bool():
    with pytest.raises(CompileError, match="must be a bool"):
        compile_to_ir(
            "fn main() -> i32 { mut i = 0\n while i { i = i + 1 }\n i }"
        )
