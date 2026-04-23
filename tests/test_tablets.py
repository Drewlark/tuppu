"""Tablets — chained-chunk growable storage (SPEC §4.4)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.codegen import CodegenError
from tuppu.errors import CompileError
from tuppu.driver import (
    compile_files_to_binary, compile_to_binary, compile_to_ir, stdlib_files,
)


def run(src: str, tmp_path: Path) -> tuple[int, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    result = subprocess.run([str(binary)], capture_output=True)
    return result.returncode, result.stdout


def run_with_stdlib(src: str, tmp_path: Path) -> tuple[int, bytes]:
    user_file = tmp_path / "main.tpu"
    user_file.write_text(src)
    binary = compile_files_to_binary(
        stdlib_files() + [user_file], tmp_path, name="prog",
    )
    result = subprocess.run([str(binary)], capture_output=True)
    return result.returncode, result.stdout


# --- basic usage -----------------------------------------------------------

def test_empty_tablets_has_zero_len(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  println(t.len)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"0\n"


def test_push_bumps_len(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(10)\n"
        "  t.push(20)\n"
        "  t.push(30)\n"
        "  println(t.len)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"3\n"


def test_indexed_access(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(100)\n"
        "  t.push(200)\n"
        "  t.push(300)\n"
        "  println(t[0])\n"
        "  println(t[1])\n"
        "  println(t[2])\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"100\n200\n300\n"


def test_chain_across_multiple_tablets(tmp_path):
    # Chunk size 4, push 10 elements, exercise chain walk on access.
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  mut i: i64 = 0\n"
        "  while i < 10 {\n"
        "    t.push(i * i)\n"
        "    i = i + 1\n"
        "  }\n"
        "  println(t.len)\n"
        "  println(t[0])\n"   # 0
        "  println(t[3])\n"   # 9 (end of first chunk)
        "  println(t[4])\n"   # 16 (start of second chunk)
        "  println(t[7])\n"   # 49 (end of second chunk)
        "  println(t[9])\n"   # 81 (middle of third chunk)
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"10\n0\n9\n16\n49\n81\n"


# --- rat element type ------------------------------------------------------

def test_tablets_of_rat(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut r: tablets[2]rat\n"
        "  r.push(1;30)\n"
        "  r.push(0;20)\n"
        "  r.push(rat(5, 6))\n"
        "  println(r[0])\n"   # 3/2
        "  println(r[1])\n"   # 1/3
        "  println(r[2])\n"   # 5/6
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"3/2\n1/3\n5/6\n"


# --- release ---------------------------------------------------------------

def test_release_resets_len(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(1)\n"
        "  t.push(2)\n"
        "  release t\n"
        "  println(t.len)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"0\n"


def test_release_then_push_works(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(1)\n"
        "  t.push(2)\n"
        "  release t\n"
        "  t.push(99)\n"
        "  println(t.len)\n"
        "  println(t[0])\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"1\n99\n"


# --- bounds checks ---------------------------------------------------------

def test_index_out_of_range_traps(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(42)\n"
        "  println(t[1])\n"   # only index 0 exists
        "  0\n"
        "}\n"
    )
    binary = compile_to_binary(src, tmp_path, name="oob")
    r = subprocess.run([str(binary)], capture_output=True)
    assert r.returncode != 0


def test_negative_index_traps(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(42)\n"
        "  println(t[-1])\n"
        "  0\n"
        "}\n"
    )
    binary = compile_to_binary(src, tmp_path, name="neg")
    r = subprocess.run([str(binary)], capture_output=True)
    assert r.returncode != 0


# --- tablets as a value from a field / fn return ---------------------------

def test_tablets_len_from_struct_field(tmp_path):
    # `.len` on a tablets accessed as a struct field (not a direct
    # Ident). Before the fix, `_gen_field`'s fast path only fired for
    # Ident-rooted tablets, so struct-field access fell through to the
    # generic "field access on X not supported yet" error.
    src = (
        "tablet Route { code: i64 }\n"
        "tablet App { routes: tablets[8]Route, port: i32 }\n"
        "fn main() -> i32 {\n"
        "  mut a: App\n"
        "  a.routes.push(Route { code: 1 })\n"
        "  a.routes.push(Route { code: 2 })\n"
        "  a.routes.push(Route { code: 3 })\n"
        "  step n = a.routes.len\n"
        "  println(n)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"3\n"


def test_tablets_field_readable_through_nonmut_struct_arg(tmp_path):
    # Passing a struct-with-tablets-field by value (non-mut) must
    # preserve the tablets metadata so the callee can read it. Before
    # the fix, `_struct_as_borrow` zeroed tablets fields for every
    # struct arg — mut or not — destroying the data the callee needed
    # to read. Only mut struct params need the neutering (to prevent
    # the callee's cleanup frame from double-releasing caller chunks).
    src = (
        "tablet Route { code: i64 }\n"
        "tablet App { routes: tablets[8]Route, port: i32 }\n"
        "fn count(app: App) -> i64 { app.routes.len }\n"
        "fn main() -> i32 {\n"
        "  mut a: App\n"
        "  a.routes.push(Route { code: 1 })\n"
        "  a.routes.push(Route { code: 2 })\n"
        "  println(count(a))\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"2\n"


def test_tablets_str_element_transfer_ownership_on_push(tmp_path):
    # The hashmap pattern the Tupsarru community hit: a local `step
    # owned_k` holds a cap>0 str, gets pushed into a tablets inside
    # an Entry struct. Previously owned_k's cleanup fired at block-
    # end, freeing the bytes the Entry still pointed at → UAF.
    # Fixed via push/struct-lit ownership transfer + per-element
    # tablets release on scope exit.
    src = (
        "tablet Entry { key: str, count: i64 }\n"
        "fn build(n: i64) -> i64 {\n"
        "  mut store: tablets[4]Entry\n"
        "  mut i: i64 = 0\n"
        "  while i < n {\n"
        "    step owned_k = \"key\" + int_to_str(i)\n"
        "    step _ = store.push(Entry { key: owned_k, count: i })\n"
        "    i = i + 1\n"
        "  }\n"
        "  mut total: i64 = 0\n"
        "  mut j: i64 = 0\n"
        "  while j < store.len {\n"
        "    step e = store[j]\n"
        "    total = total + e.count + e.key.len\n"
        "    j = j + 1\n"
        "  }\n"
        "  total\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  println(build(3))\n"  # 0+1+2 + 4*3 = 15
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"15\n"


def test_tablets_str_element_inline_push_no_leak(tmp_path):
    # The "inline rvalue" pattern (str_clone result flows directly
    # into the struct lit). Pre-fix this was safe-but-leaky (tablets
    # release freed chunks without walking str elements). Post-fix
    # the walk fires, bytes are reclaimed.
    src = (
        "tablet Entry { key: str, count: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut store: tablets[4]Entry\n"
        "  mut i: i64 = 0\n"
        "  while i < 5 {\n"
        "    step _ = store.push(Entry {\n"
        "      key: \"k\" + int_to_str(i), count: i,\n"
        "    })\n"
        "    i = i + 1\n"
        "  }\n"
        "  println(store.len)\n"
        "  step first = store[0]\n"
        "  step last = store[4]\n"
        "  println(first.key)\n"
        "  println(last.key)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"5\nk0\nk4\n"


def test_tablets_of_str_release_frees_elements(tmp_path):
    # Sanity: a tablets[N]str with heap strings, then explicit release,
    # should free every element. Before the fix only the chunks freed;
    # element bytes leaked.
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]str\n"
        "  mut i: i64 = 0\n"
        "  while i < 10 {\n"
        "    step _ = t.push(\"s\" + int_to_str(i))\n"
        "    i = i + 1\n"
        "  }\n"
        "  println(t.len)\n"
        "  println(t[7])\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"10\ns7\n"


def test_tablets_index_returns_borrow(tmp_path):
    # `step e = store[j]` is a borrow of the tablets element, not an
    # owning copy — so its cleanup is a no-op. Without this, the
    # binding's scope-exit release would double-free against the
    # tablets's own release walk.
    src = (
        "tablet Entry { key: str }\n"
        "fn main() -> i32 {\n"
        "  mut store: tablets[2]Entry\n"
        "  step _a = store.push(Entry { key: \"a\" + \"b\" })\n"
        "  step _b = store.push(Entry { key: \"c\" + \"d\" })\n"
        "  mut round: i64 = 0\n"
        "  while round < 100 {\n"
        "    step e = store[0]\n"             # borrowed read
        "    step f = store[1]\n"
        "    round = round + 1\n"
        "  }\n"
        "  println(store[0].key)\n"
        "  println(store[1].key)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"ab\ncd\n"


@pytest.mark.xfail(
    reason=(
        "Copying an Index-borrowed struct value and re-pushing into a "
        "tablets creates two entries sharing the same cap>0 str bytes; "
        "the release walk frees them both, double-freeing. The fix "
        "belongs with the broader 'struct-copy ownership' story — "
        "either neuter field caps on Index reads, or clone fields on "
        "push of an Index-sourced struct. Filed under NEXT.md §7."
    ),
    strict=True,
)
def test_reindex_and_repush_struct_double_free(tmp_path):
    src = (
        "tablet Entry { key: str, val: str }\n"
        "fn find(mut store: tablets[4]Entry, key: str) -> wedge Entry {\n"
        "  mut i: i64 = 0\n"
        "  while i < store.len {\n"
        "    step cur = store[i]\n"
        "    if str_eq(cur.key, key) { yield store.push(cur) }\n"
        "    i = i + 1\n"
        "  }\n"
        "  lost\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  mut store: tablets[4]Entry\n"
        "  step _a = store.push(Entry { key: \"k\", val: \"v\" + \"1\" })\n"
        "  step h = find(store, \"k\")\n"
        "  if h != lost { println(h.val) }\n"
        "  0\n"
        "}\n"
    )
    rc, out = run_with_stdlib(src, tmp_path)
    assert rc == 0
    assert out == b"v1\n"


def test_yield_field_of_wedge_no_double_free(tmp_path):
    # The hashmap `get(key) -> str` pattern. A fn walks a tablets to
    # find a matching Entry and yields `cur.val` (a Field read of a
    # wedge-dereferenced struct). The returned str's bytes live in
    # the tablets, not in the callee — so the callee must hand the
    # caller a borrow (cap=0) so the caller's scope-exit cleanup
    # doesn't race with the tablets's own release walk.
    src = (
        "tablet Entry { key: str, val: str }\n"
        "fn get(mut store: tablets[4]Entry, key: str) -> str {\n"
        "  mut i: i64 = 0\n"
        "  while i < store.len {\n"
        "    step cur = store[i]\n"
        "    if str_eq(cur.key, key) { yield cur.val }\n"
        "    i = i + 1\n"
        "  }\n"
        "  \"none\"\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  mut store: tablets[4]Entry\n"
        "  step _a = store.push(Entry {\n"
        "    key: \"a\" + \"1\", val: \"x\" + \"1\",\n"
        "  })\n"
        "  step _b = store.push(Entry {\n"
        "    key: \"b\" + \"2\", val: \"y\" + \"2\",\n"
        "  })\n"
        "  step found = get(store, \"b2\")\n"
        "  println(found)\n"
        "  step missing = get(store, \"nope\")\n"
        "  println(missing)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run_with_stdlib(src, tmp_path)
    assert rc == 0
    assert out == b"y2\nnone\n"


def test_tablets_len_from_fn_return(tmp_path):
    # Same general case — tablets returned from a fn, with `.len` read
    # off the returned SSA value.
    src = (
        "fn build() -> tablets[4]i64 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(10)\n"
        "  t.push(20)\n"
        "  t\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  step t = build()\n"
        "  println(t.len)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"2\n"


# --- generated helpers live in the IR --------------------------------------

def test_helpers_emitted():
    ir = compile_to_ir(
        "fn main() -> i32 { mut t: tablets[256]i64\n t.push(1)\n 0 }"
    )
    assert "__tuppu_tbls_" in ir
    assert "push" in ir
    assert "malloc" in ir


def test_monomorphization_per_elem_type():
    """tablets[N]i64 and tablets[N]rat should each get their own helpers."""
    ir = compile_to_ir(
        "fn main() -> i32 {\n"
        "  mut a: tablets[4]i64\n"
        "  mut b: tablets[4]rat\n"
        "  a.push(1)\n"
        "  b.push(1;30)\n"
        "  0\n"
        "}\n"
    )
    push_fns = [line for line in ir.splitlines()
                if "__tuppu_tbls_" in line and "push" in line and "define" in line]
    assert len(push_fns) == 2, f"expected 2 distinct push fns, got {push_fns}"


# --- error cases -----------------------------------------------------------

def test_step_tablets_without_init_errors():
    with pytest.raises(Exception, match="step"):
        compile_to_ir(
            "fn main() -> i32 { step t: tablets[4]i64\n 0 }"
        )


def test_mut_no_init_no_type_errors():
    with pytest.raises(Exception, match="type annotation"):
        compile_to_ir("fn main() -> i32 { mut x\n 0 }")


def test_tablets_unknown_method():
    with pytest.raises(CompileError, match="no method"):
        compile_to_ir(
            "fn main() -> i32 { mut t: tablets[4]i64\n t.pop()\n 0 }"
        )


def test_tablets_unknown_field():
    with pytest.raises(CompileError, match="no field"):
        compile_to_ir(
            "fn main() -> i32 { mut t: tablets[4]i64\n println(t.capacity)\n 0 }"
        )


# --- auto-release at scope exit --------------------------------------------

def test_auto_release_on_fn_exit(tmp_path):
    # No explicit `release lib`, yet the IR must contain exactly one
    # release call (inserted by codegen at scope exit).
    src = (
        "fn main() -> i32 {\n"
        "  mut xs: tablets[4]i64\n"
        "  xs.push(1)\n"
        "  xs.push(2)\n"
        "  println(xs.len)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"2\n"
    from tuppu.driver import compile_to_ir
    ir = compile_to_ir(src)
    release_calls = [l for l in str(ir).splitlines() if "_release" in l and "call " in l]
    assert len(release_calls) == 1, release_calls


def test_explicit_release_not_doubled(tmp_path):
    # Explicit `release xs` should still produce exactly one release —
    # auto-release must unregister the binding on explicit release.
    src = (
        "fn main() -> i32 {\n"
        "  mut xs: tablets[4]i64\n"
        "  xs.push(9)\n"
        "  release xs\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b""
    from tuppu.driver import compile_to_ir
    ir = compile_to_ir(src)
    release_calls = [l for l in str(ir).splitlines() if "_release" in l and "call " in l]
    assert len(release_calls) == 1, release_calls


def test_auto_release_fires_on_yield(tmp_path):
    # Early return via `yield` must still emit the release — cleanup is
    # part of the unwind chain, not just the fall-through exit.
    src = (
        "fn main() -> i32 {\n"
        "  mut xs: tablets[4]i64\n"
        "  xs.push(1)\n"
        "  if xs.len > 0 { yield 0 }\n"
        "  0\n"
        "}\n"
    )
    rc, _ = run(src, tmp_path)
    assert rc == 0
    from tuppu.driver import compile_to_ir
    ir = compile_to_ir(src)
    release_calls = [l for l in str(ir).splitlines() if "_release" in l and "call " in l]
    # At minimum one release — could emit on both the yield path and
    # the fall-through path. Both must free `xs`.
    assert len(release_calls) >= 1, release_calls


def test_lvalue_index_assign_whole_element(tmp_path):
    src = (
        "tablet P { x: i64, y: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut arr: tablets[4]P\n"
        "  arr.push(P { x: 0, y: 0 })\n"
        "  arr[0] = P { x: 3, y: 4 }\n"
        "  println(arr[0].x)\n"
        "  println(arr[0].y)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"3\n4\n"


def test_lvalue_index_assign_field(tmp_path):
    # `arr[n].field = v` writes one field of an indexed element
    # without rebuilding the whole struct.
    src = (
        "tablet P { x: i64, y: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut arr: tablets[4]P\n"
        "  arr.push(P { x: 1, y: 1 })\n"
        "  arr.push(P { x: 2, y: 2 })\n"
        "  arr[0].x = 99\n"
        "  arr[1].y = 42\n"
        "  println(arr[0].x)\n"
        "  println(arr[0].y)\n"
        "  println(arr[1].x)\n"
        "  println(arr[1].y)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"99\n1\n2\n42\n"


def test_lvalue_index_assign_nested_field(tmp_path):
    src = (
        "tablet Inner { v: i64 }\n"
        "tablet Outer { i: Inner, tag: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut arr: tablets[4]Outer\n"
        "  arr.push(Outer { i: Inner { v: 0 }, tag: 0 })\n"
        "  arr[0].i.v = 77\n"
        "  arr[0].tag = 5\n"
        "  println(arr[0].i.v)\n"
        "  println(arr[0].tag)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"77\n5\n"


def test_lvalue_index_aug_assign(tmp_path):
    src = (
        "tablet P { x: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut arr: tablets[4]P\n"
        "  arr.push(P { x: 10 })\n"
        "  arr[0].x += 5\n"
        "  arr[0].x *= 2\n"
        "  println(arr[0].x)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"30\n"


def test_lvalue_index_bounds_trap(tmp_path):
    src = (
        "tablet P { x: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut arr: tablets[4]P\n"
        "  arr.push(P { x: 1 })\n"
        "  arr[5].x = 0\n"
        "  0\n"
        "}\n"
    )
    rc, _ = run(src, tmp_path)
    assert rc != 0


def test_lvalue_index_rejects_step_binding():
    with pytest.raises(CompileError, match="step binding"):
        compile_to_ir(
            "tablet P { x: i64 }\n"
            "fn main() -> i32 {\n"
            "  mut src: tablets[4]P\n"
            "  src.push(P { x: 1 })\n"
            "  step arr = src\n"
            "  arr[0].x = 0\n"
            "  0\n"
            "}\n"
        )


def test_auto_release_inner_block_only(tmp_path):
    # A mut tablets declared in a nested block should release at that
    # block's end, independent of the outer function.
    src = (
        "fn main() -> i32 {\n"
        "  mut outer: tablets[4]i64\n"
        "  if true {\n"
        "    mut inner: tablets[4]i64\n"
        "    inner.push(1)\n"
        "  }\n"
        "  outer.push(2)\n"
        "  0\n"
        "}\n"
    )
    from tuppu.driver import compile_to_ir
    ir = compile_to_ir(src)
    release_calls = [l for l in str(ir).splitlines() if "_release" in l and "call " in l]
    # One for inner (at if-block exit), one for outer (at fn exit).
    assert len(release_calls) == 2, release_calls
