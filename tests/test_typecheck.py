"""Type checker — dedicated tests for domain-level error messages with
source positions. These programs would all previously have failed (or
succeeded incorrectly) at codegen with cryptic errors; now they're
caught up front."""
from __future__ import annotations

import pytest

from tuppu.driver import compile_to_ir
from tuppu.errors import CompileError
from tuppu.typecheck import CheckError


def fails(src: str, message_fragment: str) -> CheckError:
    with pytest.raises(CheckError, match=message_fragment) as ei:
        compile_to_ir(src)
    return ei.value


# --- if in statement position (arms need not unify) ------------------------

def test_if_stmt_arms_may_mismatch():
    # `if cond { unit } else { wedge u8 }` is fine when the value is
    # provably discarded — the if sits as an ExprStmt inside a block.
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: tablets[4]u8\n"
        "  mut flag: bool = false\n"
        "  if flag { flag = true } else { buf.push(7 as u8) }\n"
        "  0\n"
        "}\n"
    )
    # Must not raise.
    compile_to_ir(src)


def test_if_expr_arms_still_must_unify():
    # When the if's value IS used (step binding, block tail, etc.),
    # arms must still unify — the relaxation is scoped to statement
    # position only.
    with pytest.raises(CompileError, match="different types"):
        compile_to_ir(
            "fn main() -> i32 {\n"
            "  step x = if true { 1 } else { false }\n"
            "  0\n"
            "}\n"
        )


def test_if_stmt_elif_chain_also_relaxed():
    # A `else if` inside an already-relaxed if inherits the flag.
    # Without propagation, the inner if would reject.
    src = (
        "fn main() -> i32 {\n"
        "  mut buf: tablets[4]u8\n"
        "  mut n: i64 = 0\n"
        "  if n < 0 { buf.push(1 as u8) }\n"
        "  elif n == 0 { n = n + 1 }\n"
        "  else { buf.push(2 as u8) }\n"
        "  0\n"
        "}\n"
    )
    compile_to_ir(src)


def test_if_stmt_arms_can_be_divergent():
    # One arm diverges (colophon declared to return i32 but actually
    # terminates), the other runs ordinary statements. Without the
    # relaxation the user had to pad with `0 as i32` dead code.
    src = (
        "colophon fn _exit(status: i32) -> i32\n"
        "fn main() -> i32 {\n"
        "  step pid: i32 = 0 as i32\n"
        "  if pid == 0 as i32 {\n"
        "    _exit(pid)\n"
        "  } else {\n"
        "    println(\"parent\")\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    compile_to_ir(src)


# --- domain-level messages ---------------------------------------------

def test_return_type_mismatch():
    # rat → i64 is NOT auto-coerced (requires explicit `as i64`). Dish
    # DOES auto-coerce to i64 silently (it's the Babylonian convenience),
    # so we use an explicit `rat(...)` construction here instead of a
    # sex literal.
    err = fails(
        "fn f() -> i64 { rat(3, 2) }\n"
        "fn main() -> i32 { 0 }\n",
        "body produces rat, expected i64",
    )
    assert err.line > 0


def test_binary_op_type_mismatch():
    err = fails(
        "fn main() -> i32 {\n"
        "  step x: i64 = 1\n"
        "  step y: bool = true\n"
        "  if x + y == 0 { 0 } else { 1 }\n"
        "}\n",
        "requires matching",
    )
    assert err.line >= 4, f"expected error on line 4ish, got {err.line}"


def test_undefined_name_has_position():
    err = fails(
        "fn main() -> i32 {\n"
        "  step x = y + 1\n"        # y is undefined
        "  0\n"
        "}\n",
        "undefined name 'y'",
    )
    # The Ident `y` sits on line 2 (starting after the `{`).
    assert err.line == 2


def test_wrong_arity():
    fails(
        "fn add(a: i64, b: i64) -> i64 { a + b }\n"
        "fn main() -> i32 { add(1) }\n",
        "add expects 2 args, got 1",
    )


def test_arg_type_mismatch():
    fails(
        "fn want_bool(b: bool) -> i64 { 0 }\n"
        "fn main() -> i32 { want_bool(42)\n 0 }\n",
        "arg 0 has type i64, expected bool",
    )


def test_if_condition_not_bool():
    fails(
        "fn main() -> i32 {\n"
        "  if 1 { 1 } else { 0 }\n"
        "}\n",
        "if condition must be bool, got i64",
    )


def test_while_condition_not_bool():
    fails(
        "fn main() -> i32 {\n"
        "  while 1 { 0 }\n"
        "  0\n"
        "}\n",
        "while condition must be a bool",
    )


def test_if_arms_different_types():
    fails(
        "fn main() -> i32 {\n"
        "  if true { 1 } else { false }\n"
        "}\n",
        "if arms have different types",
    )


def test_assignment_type_mismatch():
    fails(
        "fn main() -> i32 {\n"
        "  mut x: i64 = 0\n"
        "  x = rat(3, 2)\n"              # rat -> i64 isn't an auto-coercion
        "  0\n"
        "}\n",
        "assignment target has type i64, value has type rat",
    )


def test_rat_field_unknown():
    # `(1;30)` is now a sex/dish literal, which shares the error shape.
    fails(
        "fn main() -> i32 { println((1;30).foo)\n 0 }",
        "has no field 'foo'",
    )


def test_tablets_method_unknown():
    fails(
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.pop()\n"
        "  0\n"
        "}\n",
        "tablets has no method 'pop'",
    )


def test_tablets_push_wrong_element_type():
    # rat into tablets[4]i64 — rat doesn't coerce to int. (dish would.)
    fails(
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(rat(3, 2))\n"
        "  0\n"
        "}\n",
        "tablets.push: value has type rat",
    )


def test_release_non_tablets():
    fails(
        "fn main() -> i32 {\n"
        "  mut x: i64 = 5\n"
        "  release x\n"
        "  0\n"
        "}\n",
        "not a tablets",
    )


def test_main_must_return_i32():
    fails(
        "fn main() -> i64 { 0 }\n",
        "main must declare -> i32",
    )


def test_duplicate_function():
    fails(
        "fn foo() -> i64 { 1 }\n"
        "fn foo() -> i64 { 2 }\n"
        "fn main() -> i32 { 0 }\n",
        "duplicate function 'foo'",
    )


def test_cannot_define_intrinsic():
    fails(
        "fn println(s: i64) { }\n"
        "fn main() -> i32 { 0 }\n",
        "built-in intrinsic",
    )


def test_unknown_type_in_annotation():
    fails(
        "fn main() -> i32 {\n"
        "  step x: not_a_type = 0\n"
        "  0\n"
        "}\n",
        "unknown type",
    )


def test_cast_rat_to_bool_rejected():
    fails(
        "fn main() -> i32 {\n"
        "  step x: rat = 1;30\n"
        "  if x as bool { 1 } else { 0 }\n"
        "}\n",
        "cannot cast rat to bool",
    )


# --- programs that used to typecheck-fail but shouldn't ---------------

def test_good_programs_still_compile():
    # A small regression check that our existing language features still
    # make it through the type checker cleanly.
    sources = [
        # fib
        "fn fib(n: i64) -> i64 {\n"
        "  if n < 2 { n } else { fib(n - 1) + fib(n - 2) }\n"
        "}\n"
        "fn main() -> i32 { fib(10) as i32 }\n",
        # interactive
        "fn main() -> i32 {\n"
        "  step n = read_int()\n"
        "  println(n * 2)\n"
        "  0\n"
        "}\n",
        # rat and tablets together
        "fn main() -> i32 {\n"
        "  mut rs: tablets[4]rat\n"
        "  rs.push(1;30)\n"
        "  println(rs[0])\n"
        "  0\n"
        "}\n",
    ]
    for s in sources:
        compile_to_ir(s)  # must not raise


def test_error_carries_line_and_col():
    err = fails(
        "fn f() -> i64 {\n"
        "  42 + true\n"
        "}\n",
        "requires matching",
    )
    assert err.line >= 2
    assert err.col >= 1


# --- did-you-mean suggestions ---------------------------------------------

def test_suggest_typo_undefined_name():
    err = fails(
        "fn main() -> i32 {\n"
        "  step velocity = 10\n"
        "  println(veocity)\n"
        "  0\n"
        "}\n",
        r"undefined name 'veocity' \(did you mean 'velocity'\?\)",
    )
    assert err.line == 3


def test_suggest_typo_unknown_function():
    fails(
        "fn compute(n: i64) -> i64 { n * 2 }\n"
        "fn main() -> i32 {\n"
        "  println(comute(5))\n"
        "  0\n"
        "}\n",
        r"unknown function 'comute' \(did you mean 'compute'\?\)",
    )


def test_suggest_typo_struct_field():
    fails(
        "tablet Point { x: i64, y: i64 }\n"
        "fn main() -> i32 {\n"
        "  step p = Point { x: 1, y: 2 }\n"
        "  println(p.ex)\n"
        "  0\n"
        "}\n",
        r"has no field 'ex' \(did you mean 'x'\?\)",
    )


def test_suggest_typo_unknown_struct():
    fails(
        "tablet Tablet { id: i64 }\n"
        "fn main() -> i32 {\n"
        "  step t = Tablt { id: 1 }\n"
        "  0\n"
        "}\n",
        r"unknown tablet 'Tablt' \(did you mean 'Tablet'\?\)",
    )


def test_no_suggestion_when_nothing_close():
    # Completely unrelated name — we should NOT hallucinate a suggestion.
    err = fails(
        "fn main() -> i32 {\n"
        "  step velocity = 10\n"
        "  println(xyzzy)\n"
        "  0\n"
        "}\n",
        r"undefined name 'xyzzy'",
    )
    assert "did you mean" not in str(err)
