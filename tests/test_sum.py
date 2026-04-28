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


def test_match_stmt_arm_with_void_call_tail(tmp_path):
    # match used as a statement (value discarded), one arm body ends
    # in a void-returning call like `{ noop() }`. Codegen used to
    # emit `phi void` at the merge block — LLVM rejects it (void
    # types only allowed as fn returns). The fix normalizes void-
    # typed arm values to "no value" so the phi is skipped entirely.
    # Surfaced by the Lua parser's `match p.cur { TkRParen =>
    # { p_bump(p) }, _ => {} }` consume-optional shape.
    src = (
        "seal T { A, B }\n"
        "fn noop() {}\n"
        "fn f(x: T) -> i64 {\n"
        "  match x {\n"
        "    A => { noop() },\n"
        "    _ => {},\n"
        "  }\n"
        "  42\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  println(f(A))\n"
        "  println(f(B))\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    assert out == b"42\n42\n"


def test_match_binder_survives_scrutinee_mutation_via_implicit_copy(tmp_path):
    # Match binders on cleanup-bearing payloads are implicit-copied
    # at codegen — so the parser idiom `match p.cur { TkIdent(name)
    # => { p_bump(p); use(name) } }` Just Works. The arm body can
    # mutate the scrutinee's source without dangling the binder.
    # The alternative — rejecting and forcing `step n = copy name`
    # at every arm — was ergonomically dead in real parser code.
    src = (
        "seal Tok { Ident(str), EOF }\n"
        "tablet Parser { cur: Tok }\n"
        "fn p_bump(mut p: Parser) {\n"
        "  p.cur = EOF\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  mut p: Parser = Parser { cur: Ident(\"hel\" + \"lo\") }\n"
        "  match p.cur {\n"
        "    Ident(name) => {\n"
        "      p_bump(p)\n"
        "      println(name)\n"
        "    },\n"
        "    EOF => println(\"eof\"),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_match_stmt_multiple_void_call_arms(tmp_path):
    # Mixed arms: void-call tail, empty body, another void-call tail.
    # All should normalize to no-value; no phi construction.
    src = (
        "seal T { A, B, C }\n"
        "fn noop1() {}\n"
        "fn noop2() {}\n"
        "fn f(x: T) -> i64 {\n"
        "  match x {\n"
        "    A => { noop1() },\n"
        "    B => { noop2() },\n"
        "    C => {},\n"
        "  }\n"
        "  7\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  println(f(A))\n"
        "  println(f(B))\n"
        "  println(f(C))\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    assert out == b"7\n7\n7\n"


def test_variant_ctor_transfers_ownership_from_ident(tmp_path):
    # `Ok(s)` where `s` is an owned str binding must take over s's
    # cleanup — otherwise s's scope-exit release frees the bytes that
    # now live in the seal payload, and the caller reads garbage.
    # Surfaced by buffer_to_str flowing through two fn boundaries:
    # build() -> str; wrap() wraps in Ok; main() matches + prints.
    src = (
        "seal Result { Ok(str), Err(str) }\n"
        "fn build() -> str {\n"
        "  mut buf: buffer[16]u8\n"
        "  buf[0] = 104 as u8\n"
        "  buf[1] = 105 as u8\n"
        "  buffer_to_str(buf, 2)\n"
        "}\n"
        "fn wrap() -> Result {\n"
        "  step s = build()\n"
        "  Ok(s)\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  match wrap() {\n"
        "    Ok(s) => println(s),\n"
        "    Err(e) => println(e),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    assert out == b"hi\n"


def test_variant_ctor_deep_clones_borrow_str(tmp_path):
    # `Ok(p)` where `p` borrows into a container must deep-clone so
    # releasing the container doesn't dangle the seal's payload.
    src = (
        "seal Holder { Some(str) }\n"
        "fn wrap(store: tablets[4]str) -> Holder {\n"
        "  Some(store[0])\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  mut store: tablets[4]str\n"
        "  step _x = store.push(\"ab\" + \"cd\")\n"
        "  step h = wrap(store)\n"
        "  release store\n"
        "  match h {\n"
        "    Some(s) => println(s),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    assert out == b"abcd\n"


def test_generic_struct_holding_tablets_of_seal_with_str(tmp_path):
    # Regression for the Map<JValue> shape: pushing a seal-with-str-
    # payload into a tablets that lives inside a generic struct.
    # Pre-fix, every read through the generic getter ran the function-
    # exit cleanup walk, and tablets_release for cleanup-bearing T
    # called seal_release on every chunk slot — zeroing the tag of
    # every value the caller still owned. Subsequent reads then saw
    # tag=0 (VNull). The fix routes borrow-source reads through the
    # chokepoint with `for_transfer=True` (GC-rooted but no cleanup
    # entry), and likewise transfers the chokepoint cleanup of a
    # function's tail return out before frame teardown.
    src = (
        "seal V {\n"
        "  VNull,\n"
        "  VInt(i64),\n"
        "  VStr(str),\n"
        "}\n"
        "\n"
        "tablet Box<T> { items: tablets[64]T }\n"
        "\n"
        "fn box_push<T>(mut b: Box<T>, x: T) {\n"
        "  b.items.push(x)\n"
        "}\n"
        "\n"
        "fn box_get<T>(b: Box<T>, i: i64) -> T {\n"
        "  b.items[i]\n"
        "}\n"
        "\n"
        "fn show(v: V) {\n"
        "  match v {\n"
        "    VNull   => println(\"null\"),\n"
        "    VInt(n) => println(\"int \", n),\n"
        "    VStr(s) => println(\"str \", s),\n"
        "  }\n"
        "}\n"
        "\n"
        "fn main() -> i32 {\n"
        "  mut c: Box<V>\n"
        "  box_push(c, VInt(10))\n"
        "  box_push(c, VStr(\"hello\"))\n"
        "  box_push(c, VInt(20))\n"
        "  box_push(c, VStr(\"world\"))\n"
        "  show(box_get(c, 0))\n"
        "  show(box_get(c, 1))\n"
        "  show(box_get(c, 2))\n"
        "  show(box_get(c, 3))\n"
        "  show(box_get(c, 0))\n"
        "  0\n"
        "}\n"
    )
    rc, out, _ = run(src, tmp_path)
    assert rc == 0
    assert out == (
        b"int 10\n"
        b"str hello\n"
        b"int 20\n"
        b"str world\n"
        b"int 10\n"
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
