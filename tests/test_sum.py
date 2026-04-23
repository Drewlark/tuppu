"""Sum types: seal declarations, variant construction, and match.

Covers non-generic and generic seals, wildcard patterns, exhaustiveness
checking, and a handful of integration shapes (seal values as function
args/returns, nested seals, reuse at multiple monomorphizations)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_to_binary, compile_to_ir
from tuppu.errors import CompileError


def run(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return r.returncode, r.stdout, r.stderr


# --- non-generic seals -----------------------------------------------------

def test_non_generic_seal_match(tmp_path):
    src = (
        "seal Shape {\n"
        "  Circle(rat),\n"
        "  Rect(rat, rat),\n"
        "  Point,\n"
        "}\n"
        "fn area(s: Shape) -> rat {\n"
        "  match s {\n"
        "    Circle(r) => r * r * rat(314, 100),\n"
        "    Rect(w, h) => w * h,\n"
        "    Point => rat(0, 1),\n"
        "  }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  println(area(Circle(rat(2, 1))))\n"
        "  println(area(Rect(rat(3, 1), rat(4, 1))))\n"
        "  println(area(Point))\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"314/25\n12/1\n0/1\n"


def test_wildcard_arm(tmp_path):
    src = (
        "seal Color { Red, Green, Blue, Custom(i64, i64, i64) }\n"
        "fn describe(c: Color) -> i64 {\n"
        "  match c {\n"
        "    Red => 1,\n"
        "    Green => 2,\n"
        "    _ => 99,\n"
        "  }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  println(describe(Red))\n"
        "  println(describe(Blue))\n"
        "  println(describe(Custom(10, 20, 30)))\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1\n99\n99\n"


def test_underscore_binder(tmp_path):
    # A wildcard binder inside a variant pattern discards that field.
    src = (
        "seal Pair { Both(i64, i64) }\n"
        "fn snd(p: Pair) -> i64 {\n"
        "  match p {\n"
        "    Both(_, b) => b,\n"
        "  }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  println(snd(Both(10, 42)))\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"42\n"


# --- generic seals ---------------------------------------------------------

def test_generic_option_inference_from_args(tmp_path):
    src = (
        "seal Option<T> { Some(T), None }\n"
        "fn main() -> i32 {\n"
        "  step x = Some(42)\n"
        "  match x {\n"
        "    Some(v) => println(v),\n"
        "    None => println(-1),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"42\n"


def test_generic_option_nullary_needs_annotation(tmp_path):
    src = (
        "seal Option<T> { Some(T), None }\n"
        "fn main() -> i32 {\n"
        "  step x: Option<i64> = None\n"
        "  match x {\n"
        "    Some(v) => println(v),\n"
        "    None => println(0),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\n"


def test_generic_option_reused_at_multiple_type_args(tmp_path):
    src = (
        "seal Option<T> { Some(T), None }\n"
        "fn main() -> i32 {\n"
        "  step a: Option<i64> = Some(7)\n"
        "  step b: Option<rat> = Some(rat(1, 2))\n"
        "  match a { Some(v) => println(v), None => println(0) }\n"
        "  match b { Some(v) => println(v), None => println(rat(0, 1)) }\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"7\n1/2\n"


def test_generic_result_two_type_params(tmp_path):
    src = (
        "seal Result<T, E> { Ok(T), Err(E) }\n"
        "fn div(a: i64, b: i64) -> Result<i64, i64> {\n"
        "  if b == 0 { Err(0 - 1) } else { Ok(a / b) }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  match div(10, 3) {\n"
        "    Ok(v) => println(v),\n"
        "    Err(c) => println(c),\n"
        "  }\n"
        "  match div(10, 0) {\n"
        "    Ok(v) => println(v),\n"
        "    Err(c) => println(c),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"3\n-1\n"


# --- expected-type threading -----------------------------------------------

def test_nullary_variant_in_return_position(tmp_path):
    # The fn's return type should propagate to the variant literal, so
    # bare `None` works without a binding annotation.
    src = (
        "seal Option<T> { Some(T), None }\n"
        "fn maybe(flag: bool) -> Option<i64> {\n"
        "  if flag { Some(99) } else { None }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  match maybe(true) {\n"
        "    Some(v) => println(v),\n"
        "    None => println(-1),\n"
        "  }\n"
        "  match maybe(false) {\n"
        "    Some(v) => println(v),\n"
        "    None => println(-1),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"99\n-1\n"


def test_nullary_variant_in_call_arg(tmp_path):
    # Param type provides the hint.
    src = (
        "seal Option<T> { Some(T), None }\n"
        "fn unwrap(o: Option<i64>, dflt: i64) -> i64 {\n"
        "  match o { Some(v) => v, None => dflt }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  println(unwrap(Some(5), 99))\n"
        "  println(unwrap(None, 99))\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"5\n99\n"


# --- checker errors --------------------------------------------------------

def test_non_exhaustive_match_rejected(tmp_path):
    src = (
        "seal Color { Red, Green, Blue }\n"
        "fn f(c: Color) -> i64 {\n"
        "  match c { Red => 1, Green => 2 }\n"
        "}\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="not exhaustive"):
        compile_to_ir(src)


def test_duplicate_pattern_rejected(tmp_path):
    src = (
        "seal Color { Red, Green }\n"
        "fn f(c: Color) -> i64 {\n"
        "  match c { Red => 1, Red => 2, Green => 3 }\n"
        "}\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="duplicate pattern"):
        compile_to_ir(src)


def test_unknown_variant_in_pattern_rejected(tmp_path):
    src = (
        "seal Color { Red, Green }\n"
        "fn f(c: Color) -> i64 {\n"
        "  match c { Red => 1, Green => 2, Purple => 3 }\n"
        "}\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="not a variant of seal"):
        compile_to_ir(src)


def test_variant_name_collision_rejected(tmp_path):
    # V0.1: variant names must be globally unique.
    src = (
        "seal A { X, Y }\n"
        "seal B { X, Z }\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="already declared in seal"):
        compile_to_ir(src)


def test_nullary_generic_variant_without_context_errors(tmp_path):
    src = (
        "seal Option<T> { Some(T), None }\n"
        "fn main() -> i32 {\n"
        "  step x = None\n"
        "  0\n"
        "}\n"
    )
    with pytest.raises(CompileError, match="cannot infer"):
        compile_to_ir(src)


def test_variant_wrong_arity_rejected(tmp_path):
    src = (
        "seal Pair { P(i64, i64) }\n"
        "fn main() -> i32 {\n"
        "  step x = P(1)\n"
        "  0\n"
        "}\n"
    )
    with pytest.raises(CompileError, match="takes 2 argument"):
        compile_to_ir(src)


def test_match_scrutinee_must_be_seal(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  step x = 5\n"
        "  match x { _ => 0 }\n"
        "  0\n"
        "}\n"
    )
    with pytest.raises(CompileError, match="scrutinee must be a seal"):
        compile_to_ir(src)


@pytest.mark.xfail(
    reason=(
        "KNOWN_BUGS.md Bug 2: recursive seal via tablet wrapper — "
        "seal E references tablet N which holds an E field. Fails "
        "with \"type E not supported in this stage\" because codegen "
        "lowers seal fields before the identified type is declared. "
        "Needs the two-phase pattern that tablets-recursion already "
        "uses."
    ),
    strict=True,
)
def test_recursive_seal_via_tablet_wrapper(tmp_path):
    src = (
        "tablet N { e: E }\n"
        "seal E { Num(i64), Add(wedge N, wedge N) }\n"
        "fn main() -> i32 {\n"
        "  mut store: tablets[8]N\n"
        "  step a: wedge N = store.push(N { e: Num(1) })\n"
        "  step b: wedge N = store.push(N { e: Add(a, a) })\n"
        "  match b.e {\n"
        "    Num(n) => println(\"leaf \", n),\n"
        "    Add(x, y) => println(\"branch\"),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    assert out == b"branch\n"
