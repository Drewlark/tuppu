"""User-defined structs: declaration, construction, field access, and use
as function parameters, return values, table elements, and mut bindings."""
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


# --- declaration + construction + field access -----------------------------

def test_simple_struct_roundtrip(tmp_path):
    src = (
        "struct Point { x: i64, y: i64 }\n"
        "fn main() -> i32 {\n"
        "  step p: Point = Point { x: 3, y: 4 }\n"
        "  println(p.x)\n"
        "  println(p.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"3\n4\n"


def test_struct_field_out_of_order_in_literal(tmp_path):
    # Literal order doesn't need to match declaration order.
    src = (
        "struct Point { x: i64, y: i64 }\n"
        "fn main() -> i32 {\n"
        "  step p = Point { y: 10, x: 20 }\n"
        "  println(p.x)\n"
        "  println(p.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"20\n10\n"


def test_struct_with_mixed_field_types(tmp_path):
    src = (
        "struct Flagged { n: i64, ok: bool }\n"
        "fn main() -> i32 {\n"
        "  step f = Flagged { n: 42, ok: true }\n"
        "  println(f.n)\n"
        "  println(f.ok)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"42\ntrue\n"


def test_struct_trailing_comma_in_decl_and_lit(tmp_path):
    src = (
        "struct Point { x: i64, y: i64, }\n"
        "fn main() -> i32 {\n"
        "  step p = Point { x: 1, y: 2, }\n"
        "  println(p.x + p.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"3\n"


def test_struct_decl_spans_lines(tmp_path):
    src = (
        "struct Point {\n"
        "  x: i64,\n"
        "  y: i64,\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  step p = Point {\n"
        "    x: 7,\n"
        "    y: 8,\n"
        "  }\n"
        "  println(p.x + p.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"15\n"


# --- struct as parameter / return type ------------------------------------

def test_struct_as_parameter_and_return(tmp_path):
    src = (
        "struct Point { x: i64, y: i64 }\n"
        "fn translate(p: Point, dx: i64, dy: i64) -> Point {\n"
        "  Point { x: p.x + dx, y: p.y + dy }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  step q = translate(Point { x: 1, y: 2 }, 10, 20)\n"
        "  println(q.x)\n"
        "  println(q.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"11\n22\n"


# --- nested structs --------------------------------------------------------

def test_nested_struct(tmp_path):
    src = (
        "struct Point { x: i64, y: i64 }\n"
        "struct Line { a: Point, b: Point }\n"
        "fn main() -> i32 {\n"
        "  step l = Line {\n"
        "    a: Point { x: 0, y: 0 },\n"
        "    b: Point { x: 3, y: 4 },\n"
        "  }\n"
        "  println(l.a.x)\n"
        "  println(l.b.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\n4\n"


def test_forward_declared_struct_reference(tmp_path):
    # Source-order forward reference: Line references Point declared later.
    # Codegen's topo sort must resolve this.
    src = (
        "struct Line { a: Point, b: Point }\n"
        "struct Point { x: i64, y: i64 }\n"
        "fn main() -> i32 {\n"
        "  step l = Line {\n"
        "    a: Point { x: 1, y: 2 },\n"
        "    b: Point { x: 9, y: 8 },\n"
        "  }\n"
        "  println(l.a.x + l.b.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"9\n"


# --- mut struct -----------------------------------------------------------

def test_mut_struct_reassign(tmp_path):
    src = (
        "struct Point { x: i64, y: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut p: Point = Point { x: 1, y: 2 }\n"
        "  p = Point { x: p.x + 10, y: p.y + 20 }\n"
        "  println(p.x)\n"
        "  println(p.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"11\n22\n"


# --- struct with rat field ------------------------------------------------

def test_struct_with_rat_field(tmp_path):
    src = (
        "struct Weighted { value: rat, weight: i64 }\n"
        "fn main() -> i32 {\n"
        "  step w = Weighted { value: 1;30, weight: 5 }\n"
        "  println(w.value)\n"
        "  println(w.weight)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"3/2\n5\n"


# --- type-checker errors --------------------------------------------------

def test_missing_field_rejected():
    with pytest.raises(CompileError, match="missing field"):
        compile_to_ir(
            "struct Point { x: i64, y: i64 }\n"
            "fn main() -> i32 { step p = Point { x: 1 }\n 0 }\n"
        )


def test_extra_field_rejected():
    with pytest.raises(CompileError, match="unknown field"):
        compile_to_ir(
            "struct Point { x: i64, y: i64 }\n"
            "fn main() -> i32 { step p = Point { x: 1, y: 2, z: 3 }\n 0 }\n"
        )


def test_wrong_field_type_rejected():
    # bool → int coerces silently, so choose types with no coercion path:
    # i64 → bool is explicitly-cast-only.
    with pytest.raises(CompileError, match="expected"):
        compile_to_ir(
            "struct Flagged { n: i64, ok: bool }\n"
            "fn main() -> i32 { step f = Flagged { n: 1, ok: 42 }\n 0 }\n"
        )


def test_unknown_field_access_rejected():
    with pytest.raises(CompileError, match="no field"):
        compile_to_ir(
            "struct Point { x: i64, y: i64 }\n"
            "fn main() -> i32 { step p = Point { x: 1, y: 2 }\n println(p.z)\n 0 }\n"
        )


def test_duplicate_struct_rejected():
    with pytest.raises(CompileError, match="duplicate struct"):
        compile_to_ir(
            "struct Point { x: i64 }\n"
            "struct Point { y: i64 }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_struct_shadowing_primitive_rejected():
    # Parser rejects `i64` as a struct name because `i64` is a type keyword,
    # not an identifier — giving defense-in-depth against clashes with
    # built-in types.
    with pytest.raises(CompileError, match="struct name"):
        compile_to_ir(
            "struct i64 { x: i64 }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_empty_struct_rejected():
    with pytest.raises(CompileError, match="at least one field"):
        compile_to_ir(
            "struct Empty {}\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_unknown_struct_in_literal_rejected():
    with pytest.raises(CompileError, match="unknown struct"):
        compile_to_ir(
            "fn main() -> i32 { step p = Point { x: 1 }\n 0 }\n"
        )


def test_recursive_struct_rejected():
    with pytest.raises(CompileError, match="recursive"):
        compile_to_ir(
            "struct Node { value: i64, next: Node }\n"
            "fn main() -> i32 { 0 }\n"
        )


# --- seal alias ------------------------------------------------------------

def test_seal_is_alias_for_struct(tmp_path):
    # `seal` is the Babylonian-flavored alias — same AST, same semantics.
    src = (
        "seal Point { x: i64, y: i64 }\n"
        "fn main() -> i32 {\n"
        "  step p = Point { x: 5, y: 6 }\n"
        "  println(p.x + p.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"11\n"


def test_seal_and_struct_interop(tmp_path):
    # One type declared with `struct`, another with `seal`, used together.
    src = (
        "struct Point { x: i64, y: i64 }\n"
        "seal   Line  { a: Point, b: Point }\n"
        "fn main() -> i32 {\n"
        "  step l = Line {\n"
        "    a: Point { x: 1, y: 2 },\n"
        "    b: Point { x: 4, y: 6 },\n"
        "  }\n"
        "  println(l.b.x - l.a.x)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"3\n"


# --- does-not-collide-with-blocks regression ------------------------------

def test_if_block_still_parses_as_block(tmp_path):
    # Ensures the struct-lit lookahead (IDENT LBRACE IDENT COLON) does not
    # accidentally absorb `if cond { body }` when body starts with an IDENT.
    src = (
        "fn main() -> i32 {\n"
        "  step cond = true\n"
        "  step x = if cond { 42 } else { 0 }\n"
        "  println(x)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"42\n"


def test_while_with_ident_body_still_parses(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut i: i64 = 0\n"
        "  while i < 3 {\n"
        "    i = i + 1\n"
        "  }\n"
        "  println(i)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"3\n"
