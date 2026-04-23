"""User-defined structs: declaration, construction, field access, and use
as function parameters, return values, table elements, and mut bindings."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import (
    compile_files_to_binary, compile_to_binary, compile_to_ir, stdlib_files,
)
from tuppu.errors import CompileError


def run(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return r.returncode, r.stdout, r.stderr


def run_with_stdlib(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes, bytes]:
    """Compile with the bundled stdlib so tests using `str_concat`,
    `int_to_str`, etc. can reach them."""
    user_file = tmp_path / "main.tpu"
    user_file.write_text(src)
    binary = compile_files_to_binary(
        stdlib_files() + [user_file], tmp_path, name="prog",
    )
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


# --- struct field auto-release --------------------------------------------
#
# A user struct that transitively holds cleanup-bearing fields (str,
# tablets, or another such struct) participates in scope-exit release.
# Without this, every Foo { name: str_concat(...) } would leak the
# name bytes the moment Foo went out of scope.

def test_struct_with_str_field_no_leak(tmp_path):
    # Stress loop — a per-iteration leak would show up as unbounded
    # RSS. Correctness proxy: finish cleanly with expected output.
    src = (
        "tablet Row { label: str }\n"
        "fn main() -> i32 {\n"
        "  mut i: i64 = 0\n"
        "  while i < 1000 {\n"
        "    step r: Row = Row { label: str_concat(\"hi \", int_to_str(i)) }\n"
        "    println(r.label)\n"
        "    i = i + 1\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run_with_stdlib(src, tmp_path)
    assert rc == 0
    lines = out.split(b"\n")
    assert lines[0] == b"hi 0"
    assert lines[999] == b"hi 999"


def test_struct_with_tablets_field_no_leak(tmp_path):
    src = (
        "tablet Buf { bytes: tablets[4]u8 }\n"
        "fn main() -> i32 {\n"
        "  mut i: i64 = 0\n"
        "  while i < 500 {\n"
        "    mut b: Buf\n"
        "    b.bytes.push(72 as u8)\n"
        "    b.bytes.push(105 as u8)\n"
        "    i = i + 1\n"
        "  }\n"
        "  println(\"done\")\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    assert out == b"done\n"


def test_struct_nested_cleanup(tmp_path):
    # Outer struct's release recursively drains Inner, which drains
    # its str field. IR should contain both struct releases plus the
    # str release that Inner dispatches to.
    src = (
        "tablet Inner { s: str }\n"
        "tablet Outer { inner: Inner }\n"
        "fn main() -> i32 {\n"
        "  mut o: Outer = Outer { inner: Inner { s: str_concat(\"a\", \"b\") } }\n"
        "  println(o.inner.s)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"ab\n"
    # For the IR spot-check use a stdlib-free equivalent (`+` operator)
    # so compile_to_ir doesn't need the fn table populated.
    ir_src = src.replace('str_concat("a", "b")', '"a" + "b"')
    ir = compile_to_ir(ir_src)
    assert "__tuppu_struct_Outer_release" in ir
    assert "__tuppu_struct_Inner_release" in ir


def test_struct_without_cleanup_has_no_release_fn():
    # A struct of plain i64 fields shouldn't emit a release fn —
    # keeps the IR lean and makes the predicate observable.
    src = (
        "tablet Point { x: i64, y: i64 }\n"
        "fn main() -> i32 {\n"
        "  step p: Point = Point { x: 1, y: 2 }\n"
        "  println(p.x + p.y)\n"
        "  0\n"
        "}\n"
    )
    ir = compile_to_ir(src)
    assert "__tuppu_struct_Point_release" not in ir


def test_struct_returned_transfers_cleanup(tmp_path):
    # Ownership transfer on tail return extends to struct returns:
    # the heap str inside the returned Row must survive the callee's
    # scope-exit release.
    src = (
        "tablet Row { label: str }\n"
        "fn build() -> Row {\n"
        "  step r: Row = Row { label: str_concat(\"hello\", \"!\") }\n"
        "  r\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  step r2: Row = build()\n"
        "  println(r2.label)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"hello!\n"


def test_mut_struct_param_caller_retains_heap(tmp_path):
    # `fn f(mut r: Row)` used to double-free: the callee's cleanup
    # frame registered the param's alloca, so the str field's bytes
    # got released at callee exit while the caller still held cap>0
    # metadata for the same ptr. Call-site neutering zeros the
    # cleanup markers in the callee's view so its release no-ops.
    src = (
        "tablet Row { label: str }\n"
        "fn take(mut r: Row) {\n"
        "  println(r.label)\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  mut r: Row = Row { label: str_concat(\"hi\", \"!\") }\n"
        "  take(r)\n"
        "  println(r.label)\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run_with_stdlib(src, tmp_path)
    assert rc == 0
    assert out == b"hi!\nhi!\n"


def test_mut_struct_param_callee_reassigns_persistently(tmp_path):
    # Mut struct params now pass by pointer (matching mut tablets and
    # colophon mut-struct), so the callee's reassignment persists to
    # the caller. The old value is released via the standard
    # reassignment path (release old, store new) — caller's eventual
    # scope-exit release frees the `local` bytes once.
    src = (
        "tablet Row { label: str }\n"
        "fn swap(mut r: Row) {\n"
        "  r = Row { label: str_concat(\"local\", \"\") }\n"
        "  println(r.label)\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  mut r: Row = Row { label: str_concat(\"outer\", \"\") }\n"
        "  swap(r)\n"
        "  println(r.label)\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run_with_stdlib(src, tmp_path)
    assert rc == 0
    assert out == b"local\nlocal\n"


def test_helper_fn_pushes_into_caller_tablets_field(tmp_path):
    # The webserver-framework pattern: a helper fn takes `mut app: App`
    # and pushes into `app.routes`. With mut-struct pass-by-pointer,
    # the caller's `a` sees the pushes. Before: callee got a local
    # copy, pushes were invisible to the caller.
    src = (
        "tablet Route { code: i64 }\n"
        "tablet App { routes: tablets[8]Route, port: i32 }\n"
        "fn add_route(mut app: App, code: i64) {\n"
        "  app.routes.push(Route { code: code })\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  mut a: App\n"
        "  add_route(a, 1)\n"
        "  add_route(a, 2)\n"
        "  add_route(a, 3)\n"
        "  println(a.routes.len)\n"
        "  println(a.routes[0].code)\n"
        "  println(a.routes[2].code)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"3\n1\n3\n"


def test_mut_struct_rejects_step_arg():
    # Step bindings have no stable address to pass; mut struct
    # requires a caller-side mut binding. Clear error at codegen.
    with pytest.raises(CompileError, match="mut binding"):
        compile_to_ir(
            "tablet Row { n: i64 }\n"
            "fn take(mut r: Row) { r.n = 1 }\n"
            "fn main() -> i32 {\n"
            "  step r: Row = Row { n: 0 }\n"
            "  take(r)\n"
            "  0\n"
            "}\n"
        )


def test_mut_struct_rejects_literal_arg():
    # Literals have no address — the mut param would have nothing
    # to mutate that the caller could observe. Rejected.
    with pytest.raises(CompileError, match="mut-bound Ident"):
        compile_to_ir(
            "tablet Row { n: i64 }\n"
            "fn take(mut r: Row) { r.n = 1 }\n"
            "fn main() -> i32 {\n"
            "  take(Row { n: 0 })\n"
            "  0\n"
            "}\n"
        )


def test_rvalue_struct_arg_anonymous_cleanup(tmp_path):
    # Passing a freshly-built struct-with-heap-field as an rvalue to
    # a non-mut param must register an anonymous cleanup at the call
    # site so the heap field's bytes don't orphan after the callee
    # returns. Stress loop sanity-checks no leak.
    src = (
        "tablet Row { label: str }\n"
        "fn build(n: i64) -> Row { Row { label: int_to_str(n) } }\n"
        "fn take(r: Row) { println(r.label) }\n"
        "fn main() -> i32 {\n"
        "  mut i: i64 = 0\n"
        "  while i < 500 {\n"
        "    take(build(i))\n"
        "    i = i + 1\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    assert out.split(b"\n")[0] == b"0"
    assert out.split(b"\n")[499] == b"499"


def test_struct_passed_to_fn_caller_retains_ownership(tmp_path):
    # Caller holds a Row with a heap str; show() reads through its
    # non-mut param. The callee must not register cleanup for the
    # non-mut struct param (would double-free), and the caller must
    # still be able to read the str after the call.
    src = (
        "tablet Row { label: str }\n"
        "fn show(r: Row) { println(r.label) }\n"
        "fn main() -> i32 {\n"
        "  step r: Row = Row { label: str_concat(\"hi\", \"!\") }\n"
        "  show(r)\n"
        "  println(r.label)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"hi!\nhi!\n"


def test_struct_reassignment_releases_old(tmp_path):
    # Reassigning a whole struct-with-cleanup binding must release the
    # old value first so the old heap str / chunks don't leak.
    src = (
        "tablet Row { label: str }\n"
        "fn main() -> i32 {\n"
        "  mut r: Row = Row { label: str_concat(\"first\", \"\") }\n"
        "  r = Row { label: str_concat(\"second\", \"\") }\n"
        "  println(r.label)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"second\n"
