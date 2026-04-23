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
        "tablet Point { x: i64, y: i64 }\n"
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
        "tablet Point { x: i64, y: i64 }\n"
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
        "tablet Flagged { n: i64, ok: bool }\n"
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
        "tablet Point { x: i64, y: i64, }\n"
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
        "tablet Point {\n"
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


# --- tablet as parameter / return type ------------------------------------

def test_struct_as_parameter_and_return(tmp_path):
    src = (
        "tablet Point { x: i64, y: i64 }\n"
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
        "tablet Point { x: i64, y: i64 }\n"
        "tablet Line { a: Point, b: Point }\n"
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
        "tablet Line { a: Point, b: Point }\n"
        "tablet Point { x: i64, y: i64 }\n"
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
        "tablet Point { x: i64, y: i64 }\n"
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


# --- tablet with rat field ------------------------------------------------

def test_struct_with_rat_field(tmp_path):
    src = (
        "tablet Weighted { value: rat, weight: i64 }\n"
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
            "tablet Point { x: i64, y: i64 }\n"
            "fn main() -> i32 { step p = Point { x: 1 }\n 0 }\n"
        )


def test_extra_field_rejected():
    with pytest.raises(CompileError, match="unknown field"):
        compile_to_ir(
            "tablet Point { x: i64, y: i64 }\n"
            "fn main() -> i32 { step p = Point { x: 1, y: 2, z: 3 }\n 0 }\n"
        )


def test_wrong_field_type_rejected():
    # bool → int coerces silently, so choose types with no coercion path:
    # i64 → bool is explicitly-cast-only.
    with pytest.raises(CompileError, match="expected"):
        compile_to_ir(
            "tablet Flagged { n: i64, ok: bool }\n"
            "fn main() -> i32 { step f = Flagged { n: 1, ok: 42 }\n 0 }\n"
        )


def test_unknown_field_access_rejected():
    with pytest.raises(CompileError, match="no field"):
        compile_to_ir(
            "tablet Point { x: i64, y: i64 }\n"
            "fn main() -> i32 { step p = Point { x: 1, y: 2 }\n println(p.z)\n 0 }\n"
        )


def test_duplicate_struct_rejected():
    with pytest.raises(CompileError, match="duplicate tablet"):
        compile_to_ir(
            "tablet Point { x: i64 }\n"
            "tablet Point { y: i64 }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_struct_shadowing_primitive_rejected():
    # Parser rejects `i64` as a tablet name because `i64` is a type keyword,
    # not an identifier — giving defense-in-depth against clashes with
    # built-in types.
    with pytest.raises(CompileError, match="tablet name"):
        compile_to_ir(
            "tablet i64 { x: i64 }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_empty_struct_rejected():
    with pytest.raises(CompileError, match="at least one field"):
        compile_to_ir(
            "tablet Empty {}\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_unknown_struct_in_literal_rejected():
    with pytest.raises(CompileError, match="unknown tablet"):
        compile_to_ir(
            "fn main() -> i32 { step p = Point { x: 1 }\n 0 }\n"
        )


def test_recursive_struct_rejected():
    with pytest.raises(CompileError, match="recursive"):
        compile_to_ir(
            "tablet Node { value: i64, next: Node }\n"
            "fn main() -> i32 { 0 }\n"
        )


# --- `seal` reserved for future sum types ----------------------------------

def test_seal_keyword_is_reserved(tmp_path):
    # Using `seal` as a product-type decl now fails — the keyword is
    # reserved for future sum types. Users should write `tablet` instead.
    with pytest.raises(CompileError):
        compile_to_ir(
            "seal Point { x: i64, y: i64 }\n"
            "fn main() -> i32 { 0 }\n"
        )


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


# --- field mutation (p.x = 5) ----------------------------------------------

def test_field_assign_simple(tmp_path):
    src = (
        "tablet Point { x: i64, y: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut p: Point = Point { x: 3, y: 4 }\n"
        "  p.x = 99\n"
        "  println(p.x)\n"
        "  println(p.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"99\n4\n"


def test_field_assign_aug_op(tmp_path):
    # `p.x += 1` also parses — aug-assign works on field targets now.
    src = (
        "tablet Point { x: i64, y: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut p: Point = Point { x: 10, y: 0 }\n"
        "  p.x += 5\n"
        "  p.x *= 2\n"
        "  println(p.x)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"30\n"


def test_field_assign_nested(tmp_path):
    # Multi-level field chains GEP through each tablet level.
    src = (
        "tablet Point { x: i64, y: i64 }\n"
        "tablet Line { a: Point, b: Point }\n"
        "fn main() -> i32 {\n"
        "  mut l: Line = Line { a: Point { x: 0, y: 0 }, b: Point { x: 0, y: 0 } }\n"
        "  l.a.x = 7\n"
        "  l.b.y += 100\n"
        "  println(l.a.x)\n"
        "  println(l.b.y)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"7\n100\n"


def test_field_assign_preserves_other_fields(tmp_path):
    # Mutating one field must not disturb siblings — confirms we GEP
    # into the alloca rather than reassigning the whole struct.
    src = (
        "tablet P { x: i64, y: i64, z: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut p: P = P { x: 1, y: 2, z: 3 }\n"
        "  p.y = 99\n"
        "  println(p.x)\n"
        "  println(p.y)\n"
        "  println(p.z)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1\n99\n3\n"


def test_field_assign_to_step_rejected():
    # step-bound structs are immutable; field assign is a codegen error.
    with pytest.raises(CompileError, match="step binding"):
        compile_to_ir(
            "tablet P { x: i64 }\n"
            "fn main() -> i32 {\n"
            "  step p = P { x: 1 }\n"
            "  p.x = 2\n"
            "  0\n"
            "}\n"
        )


def test_field_assign_type_mismatch():
    with pytest.raises(CompileError, match="assignment target has type i64"):
        compile_to_ir(
            "tablet P { x: i64 }\n"
            "fn main() -> i32 {\n"
            "  mut p: P = P { x: 0 }\n"
            "  p.x = rat(1, 2)\n"
            "  0\n"
            "}\n"
        )


def test_non_lvalue_assignment_rejected():
    with pytest.raises(CompileError, match="assignment target"):
        compile_to_ir(
            "fn main() -> i32 {\n"
            "  3 + 4 = 5\n"
            "  0\n"
            "}\n"
        )


# --- wedge handles + recursive structs ------------------------------------

def test_linked_list_basic(tmp_path):
    # Build a linked list via tablets.push returning a handle, walk via
    # auto-deref on `cur.value` / `cur.next`, terminate on `cur == lost`.
    src = (
        "tablet Node { value: i64, next: wedge Node }\n"
        "fn main() -> i32 {\n"
        "  mut lib: tablets[16]Node\n"
        "  mut head: wedge Node = lost\n"
        "  head = lib.push(Node { value: 3, next: head })\n"
        "  head = lib.push(Node { value: 2, next: head })\n"
        "  head = lib.push(Node { value: 1, next: head })\n"
        "  mut cur: wedge Node = head\n"
        "  while cur != lost {\n"
        "    println(cur.value)\n"
        "    cur = cur.next\n"
        "  }\n"
        "  release lib\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1\n2\n3\n"


def test_mutually_recursive_structs(tmp_path):
    # A references B via wedge B and vice versa — identified types
    # resolve both forward references cleanly.
    src = (
        "tablet Even { v: i64, down: wedge Odd }\n"
        "tablet Odd  { v: i64, down: wedge Even }\n"
        "fn main() -> i32 {\n"
        "  mut pool_e: tablets[8]Even\n"
        "  mut pool_o: tablets[8]Odd\n"
        "  step e0: wedge Even = pool_e.push(Even { v: 0, down: lost })\n"
        "  step o1: wedge Odd  = pool_o.push(Odd  { v: 1, down: e0 })\n"
        "  step e2: wedge Even = pool_e.push(Even { v: 2, down: o1 })\n"
        "  println(e2.v)\n"
        "  println(e2.down.v)\n"
        "  println(e2.down.down.v)\n"
        "  release pool_e\n"
        "  release pool_o\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"2\n1\n0\n"


def test_lost_equality(tmp_path):
    src = (
        "tablet N { v: i64, next: wedge N }\n"
        "fn main() -> i32 {\n"
        "  mut lib: tablets[4]N\n"
        "  step a: wedge N = lost\n"
        "  step b: wedge N = lib.push(N { v: 7, next: lost })\n"
        "  if a == lost { println(1) } else { println(0) }\n"
        "  if b == lost { println(0) } else { println(1) }\n"
        "  if a == b    { println(0) } else { println(1) }\n"
        "  release lib\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1\n1\n1\n"


def test_recursive_tablet_without_indirection_rejected():
    with pytest.raises(CompileError, match="recursively contained without indirection"):
        compile_to_ir(
            "tablet Node { v: i64, child: Node }\n"
            "fn main() -> i32 { 0 }\n"
        )


# --- escape check: no returning local-rooted handles -----------------------

def test_escape_rejects_return_of_local_push():
    with pytest.raises(CompileError, match="cannot return a wedge handle"):
        compile_to_ir(
            "tablet Node { value: i64, next: wedge Node }\n"
            "fn build() -> wedge Node {\n"
            "  mut lib: tablets[8]Node\n"
            "  lib.push(Node { value: 1, next: lost })\n"
            "}\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_escape_rejects_return_of_tainted_binding():
    with pytest.raises(CompileError, match="cannot return a wedge handle"):
        compile_to_ir(
            "tablet Node { value: i64, next: wedge Node }\n"
            "fn build() -> wedge Node {\n"
            "  mut lib: tablets[8]Node\n"
            "  step h: wedge Node = lib.push(Node { value: 1, next: lost })\n"
            "  h\n"
            "}\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_escape_rejects_yielded_local_handle():
    with pytest.raises(CompileError, match="cannot return a wedge handle"):
        compile_to_ir(
            "tablet Node { value: i64, next: wedge Node }\n"
            "fn build() -> wedge Node {\n"
            "  mut lib: tablets[8]Node\n"
            "  yield lib.push(Node { value: 1, next: lost })\n"
            "}\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_escape_accepts_lost_return(tmp_path):
    # `lost` has no provenance — always safe to return.
    src = (
        "tablet Node { value: i64, next: wedge Node }\n"
        "fn empty() -> wedge Node { lost }\n"
        "fn main() -> i32 {\n"
        "  step h: wedge Node = empty()\n"
        "  if h == lost { println(1) } else { println(0) }\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1\n"


# --- mut tablets params (by-ref semantics) --------------------------------

def test_mut_tablets_param_pass_through(tmp_path):
    # Callee pushes into caller's tablets — mutations must persist so
    # the eventual release actually frees the chunks allocated here.
    src = (
        "tablet Node { value: i64, next: wedge Node }\n"
        "fn push_front(mut store: tablets[16]Node, head: wedge Node, v: i64) -> wedge Node {\n"
        "  store.push(Node { value: v, next: head })\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  mut lib: tablets[16]Node\n"
        "  mut head: wedge Node = lost\n"
        "  head = push_front(lib, head, 3)\n"
        "  head = push_front(lib, head, 2)\n"
        "  head = push_front(lib, head, 1)\n"
        "  println(lib.len)\n"
        "  mut cur: wedge Node = head\n"
        "  while cur != lost {\n"
        "    println(cur.value)\n"
        "    cur = cur.next\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    # lib.len should reflect all three pushes (by-ref), and the walk
    # prints 1, 2, 3.
    assert out == b"3\n1\n2\n3\n"


# --- generics --------------------------------------------------------------

def test_generic_tablet_basic(tmp_path):
    # Box<T> over i64 — simplest generic tablet, no recursion.
    src = (
        "tablet Box<T> { value: T }\n"
        "fn main() -> i32 {\n"
        "  step b = Box { value: 42 }\n"
        "  println(b.value)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"42\n"


def test_generic_fn_inference_from_arg(tmp_path):
    # identity<T>(x) — T inferred from the argument's concrete type.
    src = (
        "fn identity<T>(x: T) -> T { x }\n"
        "fn main() -> i32 {\n"
        "  println(identity(7))\n"
        "  println(identity(true))\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"7\ntrue\n"


def test_generic_tablet_two_instantiations(tmp_path):
    # Box<i64> and Box<bool> as distinct monomorphs in the same program.
    src = (
        "tablet Box<T> { v: T }\n"
        "fn main() -> i32 {\n"
        "  step a = Box { v: 10 }\n"
        "  step b = Box { v: false }\n"
        "  println(a.v)\n"
        "  println(b.v)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"10\nfalse\n"


def test_generic_recursive_with_wedge(tmp_path):
    # Node<T> with wedge Node<T> next — the stdlib List pattern.
    src = (
        "tablet Node<T> { value: T, next: wedge Node<T> }\n"
        "fn main() -> i32 {\n"
        "  mut lib: tablets[8]Node<i64>\n"
        "  mut head: wedge Node<i64> = lost\n"
        "  head = lib.push(Node { value: 1, next: head })\n"
        "  head = lib.push(Node { value: 2, next: head })\n"
        "  println(head.value)\n"
        "  println(head.next.value)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"2\n1\n"


def test_generic_arity_mismatch_rejected():
    with pytest.raises(CompileError, match="expects 1 type argument"):
        compile_to_ir(
            "tablet Box<T> { v: T }\n"
            "fn main() -> i32 {\n"
            "  step b: Box<i64, i64> = Box { v: 1 }\n"
            "  0\n"
            "}\n"
        )


def test_generic_missing_type_arg_rejected():
    with pytest.raises(CompileError, match="expects 1 type argument"):
        compile_to_ir(
            "tablet Box<T> { v: T }\n"
            "fn main() -> i32 {\n"
            "  step b: Box = Box { v: 1 }\n"
            "  0\n"
            "}\n"
        )
