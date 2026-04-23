"""Operator overloads via the `gloss` keyword.

A `gloss <op>(a, b)` declaration registers a fn in an operator-
dispatch table keyed on (op, lhs_ty, rhs_ty). Binary/unary ops on
user types look up the table before erroring, so user tablets gain
`+`/`-`/`*`/etc. naturally. Internal mangling (`__gloss_add_Vector_
Vector`) keeps the user's `fn add` namespace free, and the
typechecker emits a did-you-mean warning if someone writes a plain
fn that looks like an operator overload on user types."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_to_binary, compile_sources_to_ir, compile_to_ir
from tuppu.errors import CompileError


def run(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return r.returncode, r.stdout, r.stderr


# --- basic dispatch --------------------------------------------------------

def test_gloss_add_on_vector(tmp_path):
    src = (
        "tablet Vector { x: i64, y: i64 }\n"
        "gloss add(a: Vector, b: Vector) -> Vector {\n"
        "  Vector { x: a.x + b.x, y: a.y + b.y }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  step p = Vector { x: 1, y: 2 }\n"
        "  step q = Vector { x: 10, y: 20 }\n"
        "  step r = p + q\n"
        "  println(r.x)\n"
        "  println(r.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"11\n22\n"


def test_gloss_all_binary_arith(tmp_path):
    # Every binary-arith gloss op dispatches when declared.
    src = (
        "tablet N { v: i64 }\n"
        "gloss add(a: N, b: N) -> N { N { v: a.v + b.v } }\n"
        "gloss sub(a: N, b: N) -> N { N { v: a.v - b.v } }\n"
        "gloss mul(a: N, b: N) -> N { N { v: a.v * b.v } }\n"
        "gloss div(a: N, b: N) -> N { N { v: a.v / b.v } }\n"
        "gloss mod(a: N, b: N) -> N { N { v: a.v % b.v } }\n"
        "fn main() -> i32 {\n"
        "  step a = N { v: 12 }\n"
        "  step b = N { v: 5 }\n"
        "  println((a + b).v)\n"
        "  println((a - b).v)\n"
        "  println((a * b).v)\n"
        "  println((a / b).v)\n"
        "  println((a % b).v)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"17\n7\n60\n2\n2\n"


def test_gloss_unary_neg_and_not(tmp_path):
    src = (
        "tablet Vec { x: i64 }\n"
        "tablet Flag { b: bool }\n"
        "gloss neg(a: Vec) -> Vec { Vec { x: -a.x } }\n"
        "gloss not(a: Flag) -> Flag { Flag { b: !a.b } }\n"
        "fn main() -> i32 {\n"
        "  step v = Vec { x: 7 }\n"
        "  println((-v).x)\n"
        "  step f = Flag { b: true }\n"
        "  println((!f).b)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"-7\nfalse\n"


def test_gloss_eq_derives_ne(tmp_path):
    # Only `gloss eq` is declared; `!=` auto-derives.
    src = (
        "tablet Point { x: i64, y: i64 }\n"
        "gloss eq(a: Point, b: Point) -> bool {\n"
        "  a.x == b.x && a.y == b.y\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  step p = Point { x: 1, y: 2 }\n"
        "  step q = Point { x: 1, y: 2 }\n"
        "  step r = Point { x: 9, y: 9 }\n"
        "  println(p == q)\n"
        "  println(p == r)\n"
        "  println(p != q)\n"
        "  println(p != r)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"true\nfalse\nfalse\ntrue\n"


def test_gloss_comparisons(tmp_path):
    src = (
        "tablet N { v: i64 }\n"
        "gloss lt(a: N, b: N) -> bool { a.v < b.v }\n"
        "gloss le(a: N, b: N) -> bool { a.v <= b.v }\n"
        "gloss gt(a: N, b: N) -> bool { a.v > b.v }\n"
        "gloss ge(a: N, b: N) -> bool { a.v >= b.v }\n"
        "fn main() -> i32 {\n"
        "  step a = N { v: 3 }\n"
        "  step b = N { v: 5 }\n"
        "  println(a < b)\n"
        "  println(a <= b)\n"
        "  println(a > b)\n"
        "  println(b >= a)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"true\ntrue\nfalse\ntrue\n"


def test_gloss_mixed_operand_types(tmp_path):
    # `Vec * i64` scales; dispatch keys on the pair so both orderings
    # need their own gloss if you want commutativity.
    src = (
        "tablet Vec { x: i64, y: i64 }\n"
        "gloss mul(a: Vec, s: i64) -> Vec { Vec { x: a.x * s, y: a.y * s } }\n"
        "fn main() -> i32 {\n"
        "  step v = Vec { x: 2, y: 3 }\n"
        "  step u = v * 5\n"
        "  println(u.x)\n"
        "  println(u.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"10\n15\n"


def test_gloss_coexists_with_fn_add(tmp_path):
    # Users can keep a regular `fn add` for their own API; `gloss add`
    # occupies a separate (mangled) slot in the symbol table.
    src = (
        "tablet Vec { x: i64 }\n"
        "fn add(a: i64, b: i64) -> i64 { a + b + 1000 }\n"  # sentinel offset
        "gloss add(a: Vec, b: Vec) -> Vec { Vec { x: a.x + b.x } }\n"
        "fn main() -> i32 {\n"
        "  println(add(1, 2))\n"                             # regular fn
        "  step v = Vec { x: 7 }\n"
        "  step w = Vec { x: 3 }\n"
        "  println((v + w).x)\n"                             # gloss dispatch
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1003\n10\n"


# --- typecheck rejections --------------------------------------------------

def test_gloss_unknown_op_rejected():
    with pytest.raises(CompileError, match="unknown operator name"):
        compile_to_ir(
            "tablet V { x: i64 }\n"
            "gloss foo(a: V, b: V) -> V { a }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_gloss_arity_mismatch_rejected():
    # `add` is binary; one-arg decl is a decl-time error.
    with pytest.raises(CompileError, match="expects 2 param"):
        compile_to_ir(
            "tablet V { x: i64 }\n"
            "gloss add(a: V) -> V { a }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_gloss_primitive_only_rejected():
    with pytest.raises(CompileError, match="user tablet or seal"):
        compile_to_ir(
            "gloss add(a: i64, b: i64) -> i64 { a + b }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_gloss_eq_must_return_bool():
    with pytest.raises(CompileError, match="return type must be bool"):
        compile_to_ir(
            "tablet V { x: i64 }\n"
            "gloss eq(a: V, b: V) -> i64 { 0 }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_gloss_duplicate_rejected():
    with pytest.raises(CompileError, match="duplicate definition"):
        compile_to_ir(
            "tablet V { x: i64 }\n"
            "gloss add(a: V, b: V) -> V { a }\n"
            "gloss add(a: V, b: V) -> V { b }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_fn_looking_like_gloss_warns(capsys):
    # A plain `fn add(V, V) -> V` triggers a did-you-mean warning but
    # still compiles — the user might genuinely have meant it. The
    # driver prints warnings to stderr.
    src = (
        "tablet V { x: i64 }\n"
        "fn add(a: V, b: V) -> V { a }\n"
        "fn main() -> i32 { 0 }\n"
    )
    # compile_sources_to_ir writes warnings to stderr via driver glue;
    # route through the driver so we observe the message.
    compile_sources_to_ir([("<src>", src)])
    captured = capsys.readouterr()
    assert "did you mean `gloss add`" in captured.err


def test_gloss_doesnt_apply_to_non_user_operand_miss(tmp_path):
    # `gloss add(Vec, Vec)` is declared, but `Vec + i64` has no
    # matching entry and falls through to the regular error.
    with pytest.raises(CompileError, match="matching"):
        compile_to_ir(
            "tablet Vec { x: i64 }\n"
            "gloss add(a: Vec, b: Vec) -> Vec { a }\n"
            "fn main() -> i32 {\n"
            "  step v = Vec { x: 1 }\n"
            "  step w = v + 5\n"                 # no gloss for (Vec, i64)
            "  println(w.x)\n"
            "  0\n"
            "}\n"
        )
