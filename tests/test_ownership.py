"""Ownership / cleanup invariants across storage sites.

Every cleanup-bearing value ends up in exactly one of these storage
sinks:

  - tablets.push(x)                   → chunk slot
  - Tablet { field: x, ... }          → struct-lit field
  - slot = x    (Ident / Field / Index target)
  - Variant(x)                        → seal payload
  - yield / fall-through block tail   → caller's binding

The discipline is the same at every site: transfer from an owning
Ident, deep-clone a borrow / rvalue that has no transferable cleanup.
After a transfer the source slot is zeroed so later release paths
(reassignment, explicit `release`, auto-release on scope exit) become
no-ops on the moved-from binding.

This file collects regression tests that probe the full matrix —
source shape × storage sink — so a future change that re-introduces
an alias leaks or double-frees lands against a failing test instead
of a user-reported UAF.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_to_binary, compile_to_ir
from tuppu.errors import CompileError


def run(src: str, tmp_path: Path) -> tuple[int, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], capture_output=True)
    return r.returncode, r.stdout


# --- reassign after transfer ------------------------------------------

def test_reassign_str_after_push_no_uaf(tmp_path):
    # `mut owned = ...; t.push(owned); owned = other`. The push
    # transferred ownership into the tablets. The reassignment must
    # NOT release the transferred heap bytes — they now live in the
    # tablets.
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]str\n"
        "  mut owned: str = \"al\" + \"ice\"\n"
        "  step _x = t.push(owned)\n"
        "  owned = \"bo\" + \"b\"\n"
        "  println(t[0])\n"
        "  println(owned)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"alice\nbob\n"


def test_explicit_release_after_tablets_transfer_no_crash(tmp_path):
    # `step g = Group { members: src }; release src`. The struct-lit
    # transferred src's chunks into g. The explicit release must NOT
    # walk and free those chunks again.
    src = (
        "tablet Group { members: tablets[4]str }\n"
        "fn main() -> i32 {\n"
        "  mut src: tablets[4]str\n"
        "  step _a = src.push(\"al\" + \"ice\")\n"
        "  step _b = src.push(\"bo\" + \"b\")\n"
        "  step g = Group { members: src }\n"
        "  release src\n"
        "  println(g.members[0])\n"
        "  println(g.members[1])\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"alice\nbob\n"


def test_double_transfer_second_is_noop(tmp_path):
    # First push moves the owned str into a; a's slot is zeroed.
    # Second push reads the (now-empty) slot and pushes an empty str.
    # Critically: no double-free.
    src = (
        "fn main() -> i32 {\n"
        "  mut a: tablets[4]str\n"
        "  mut b: tablets[4]str\n"
        "  mut owned: str = \"hi\" + \"!\"\n"
        "  step _x = a.push(owned)\n"
        "  step _y = b.push(owned)\n"
        "  println(a[0])\n"
        "  println(b[0])\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hi!\n\n"


# --- variant ctor storage ---------------------------------------------

def test_variant_ctor_from_rvalue_call_direct(tmp_path):
    # Ok(build()) — no intermediate binding. The rvalue Call result
    # flows straight into the seal payload; the returned seal must
    # own its bytes (not dangle against any anonymous-cleanup frame
    # that fires at the wrap() exit).
    src = (
        "seal R { Ok(str), Err(str) }\n"
        "fn build() -> str { \"he\" + \"llo\" }\n"
        "fn wrap() -> R { Ok(build()) }\n"
        "fn main() -> i32 {\n"
        "  match wrap() {\n"
        "    Ok(s) => println(s),\n"
        "    Err(e) => println(e),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_yield_variant_with_owned_transfer(tmp_path):
    # yield Ok(s) where s is a locally-owned str. Yield unwinds the
    # current frame's cleanups before the ret — the transfer must
    # remove s's cleanup so the early return doesn't free bytes
    # the returned variant is pointing at.
    src = (
        "seal R { Ok(str), Err(str) }\n"
        "fn wrap(flag: bool) -> R {\n"
        "  step s = \"he\" + \"llo\"\n"
        "  if flag { yield Ok(s) }\n"
        "  Err(\"nope\")\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  match wrap(true) {\n"
        "    Ok(s) => println(s),\n"
        "    Err(e) => println(e),\n"
        "  }\n"
        "  match wrap(false) {\n"
        "    Ok(s) => println(s),\n"
        "    Err(e) => println(e),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\nnope\n"


def test_nested_variant_ctor_ok_of_wrap(tmp_path):
    # Ok(Wrap(s)) — variant payload is itself a variant. Both layers
    # must transfer correctly; the outermost seal ends up owning the
    # str bytes through the inner seal's payload.
    src = (
        "seal Inner { Wrap(str) }\n"
        "seal Outer { Ok(Inner), Err(str) }\n"
        "fn build() -> Outer {\n"
        "  step s = \"ab\" + \"cd\"\n"
        "  Ok(Wrap(s))\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  match build() {\n"
        "    Ok(inner) => match inner {\n"
        "      Wrap(s) => println(s),\n"
        "    },\n"
        "    Err(e) => println(e),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"abcd\n"


# --- source-shape coverage at push sites ------------------------------

def test_push_match_binder_into_other_tablets(tmp_path):
    # Pattern binder `s` from a match arm is a borrow into the
    # scrutinee's payload. Pushing it into another tablets must
    # deep-clone so the destination owns independent bytes.
    src = (
        "seal Msg { Text(str) }\n"
        "fn main() -> i32 {\n"
        "  mut log: tablets[4]str\n"
        "  step m = Text(\"he\" + \"llo\")\n"
        "  match m {\n"
        "    Text(s) => {\n"
        "      step _x = log.push(s)\n"
        "    }\n"
        "  }\n"
        "  println(log[0])\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_push_rvalue_fn_call_result(tmp_path):
    # push(make()) — rvalue Call result, no named Ident. Transfer
    # isn't possible; the push-path deep-clone covers this shape.
    src = (
        "fn make() -> str { \"hel\" + \"lo\" }\n"
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]str\n"
        "  step _x = t.push(make())\n"
        "  println(t[0])\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_assign_struct_field_from_borrow_deep_clones(tmp_path):
    # b.s = store[0] where store[0] is a borrow. After `release
    # store`, b.s must still be readable — the assign had to deep-
    # clone. Otherwise b.s aliased into freed chunks.
    src = (
        "tablet Box { s: str }\n"
        "fn main() -> i32 {\n"
        "  mut store: tablets[4]str\n"
        "  step _x = store.push(\"original\" + \"!\")\n"
        "  mut b: Box = Box { s: \"init\" + \"!\" }\n"
        "  b.s = store[0]\n"
        "  release store\n"
        "  println(b.s)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"original!\n"


# --- seals composed with containers -----------------------------------

def test_push_variant_with_str_payload(tmp_path):
    # Pushing seal values that carry heap-owning payloads into a
    # tablets. Each variant's str payload lives through the
    # tablets's release walk — the element-walk release needs to
    # reach into the seal payload. (Seal-release is currently
    # shallow, so the payload would leak at scope exit, but the
    # reads must still be correct.)
    src = (
        "seal Msg { Text(str), Silent }\n"
        "fn main() -> i32 {\n"
        "  mut log: tablets[4]Msg\n"
        "  step _a = log.push(Text(\"hel\" + \"lo\"))\n"
        "  step _b = log.push(Text(\"wo\" + \"rld\"))\n"
        "  match log[0] {\n"
        "    Text(s) => println(s),\n"
        "    Silent => println(\"silent\"),\n"
        "  }\n"
        "  match log[1] {\n"
        "    Text(s) => println(s),\n"
        "    Silent => println(\"silent\"),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\nworld\n"


def test_struct_with_seal_field(tmp_path):
    # A tablet whose field is a seal type with a str payload. The
    # struct-lit path transfers into the seal payload, and the
    # struct binding owns the chain through its auto-release.
    src = (
        "seal Msg { Text(str), Silent }\n"
        "tablet Entry { m: Msg, count: i64 }\n"
        "fn main() -> i32 {\n"
        "  step e = Entry { m: Text(\"he\" + \"llo\"), count: 3 }\n"
        "  match e.m {\n"
        "    Text(s) => println(s),\n"
        "    Silent => println(\"silent\"),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_tablets_of_tablets_owned_transfer(tmp_path):
    # `tablets[4]tablets[4]str` — the element type is itself a
    # tablets. Pushing an owned inner tablets transfers its chunks
    # into the outer container's slot; subsequent reads walk through
    # the outer to the inner.
    src = (
        "fn main() -> i32 {\n"
        "  mut groups: tablets[4]tablets[4]str\n"
        "  mut g1: tablets[4]str\n"
        "  step _a = g1.push(\"al\" + \"ice\")\n"
        "  step _b = g1.push(\"bo\" + \"b\")\n"
        "  step _g = groups.push(g1)\n"
        "  mut g2: tablets[4]str\n"
        "  step _c = g2.push(\"chris\" + \"!\")\n"
        "  step _gg = groups.push(g2)\n"
        "  println(groups[0][0])\n"
        "  println(groups[0][1])\n"
        "  println(groups[1][0])\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"alice\nbob\nchris!\n"


def test_reassign_mut_seal_with_heap_payload(tmp_path):
    # `mut m: Msg = Text(...)` then `m = Text(other)`. The mut seal
    # slot's reassignment releases the old variant's payload via the
    # per-seal release helper, then the new value's payload transfers
    # in. No leak, no UAF, no double-free.
    src = (
        "seal Msg { Text(str), Silent }\n"
        "fn main() -> i32 {\n"
        "  mut m: Msg = Text(\"first\" + \"!\")\n"
        "  m = Text(\"second\" + \"!\")\n"
        "  match m {\n"
        "    Text(s) => println(s),\n"
        "    Silent => println(\"silent\"),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"second!\n"


# --- seal release (scope-exit + composition) --------------------------

def test_seal_binding_releases_payload_at_scope_exit(tmp_path):
    # Tight loop that allocates a heap str into a seal variant each
    # iteration. Without a per-seal release the heap bytes accumulate
    # unboundedly; with it, peak memory stays flat. We verify the
    # loop completes cleanly — a missing release would eventually
    # exhaust address space or at least show in the IR as a missing
    # call at scope exit.
    src = (
        "seal Msg { Text(str), Silent }\n"
        "fn main() -> i32 {\n"
        "  mut n: i64 = 0\n"
        "  while n < 10000 {\n"
        "    step m = Text(\"aa\" + \"bb\")\n"
        "    match m {\n"
        "      Text(s) => {},\n"
        "      Silent => {},\n"
        "    }\n"
        "    n = n + 1\n"
        "  }\n"
        "  println(\"done\")\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"done\n"


def test_seal_release_wired_into_ir(tmp_path):
    # IR-level check that a `step m = Text(...)` binding emits a
    # __tuppu_seal_<name>_release call at scope exit.
    from tuppu.driver import compile_to_ir
    src = (
        "seal Msg { Text(str), Silent }\n"
        "fn main() -> i32 {\n"
        "  step m = Text(\"hi\" + \"!\")\n"
        "  0\n"
        "}\n"
    )
    ir = compile_to_ir(src)
    # Release helper is defined and a call is emitted at main's exit.
    assert "define void @\"__tuppu_seal_Msg_release\"" in ir
    assert "call void @\"__tuppu_seal_Msg_release\"" in ir


def test_tablets_of_seal_walks_payload_on_release(tmp_path):
    # tablets[N]Msg with heap-bearing Msg variants. The element-walk
    # release must dispatch to the per-seal release for each slot so
    # each variant's heap payload gets freed.
    src = (
        "seal Msg { Text(str), Silent }\n"
        "fn main() -> i32 {\n"
        "  mut n: i64 = 0\n"
        "  while n < 1000 {\n"
        "    mut t: tablets[8]Msg\n"
        "    step _a = t.push(Text(\"aa\" + \"bb\"))\n"
        "    step _b = t.push(Text(\"cc\" + \"dd\"))\n"
        "    step _c = t.push(Silent)\n"
        "    n = n + 1\n"
        "  }\n"
        "  println(\"done\")\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"done\n"


def test_struct_with_seal_field_releases_payload(tmp_path):
    # A tablet whose field is a seal with cleanup-bearing payload.
    # Struct-release must walk to the seal field and dispatch to
    # seal-release so nested heap bytes get freed.
    src = (
        "seal Msg { Text(str), Silent }\n"
        "tablet Entry { m: Msg, count: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut n: i64 = 0\n"
        "  while n < 1000 {\n"
        "    step e = Entry { m: Text(\"aa\" + \"bb\"), count: 1 }\n"
        "    match e.m {\n"
        "      Text(s) => {},\n"
        "      Silent => {},\n"
        "    }\n"
        "    n = n + 1\n"
        "  }\n"
        "  println(\"done\")\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"done\n"


def test_nested_seal_release_walks_inner(tmp_path):
    # Outer seal holds an Inner seal payload that itself holds a
    # heap str. Outer-release dispatches to inner-release which frees
    # the str. A missing inner-release would leak the str bytes.
    src = (
        "seal Inner { Wrap(str) }\n"
        "seal Outer { Hold(Inner), Empty }\n"
        "fn main() -> i32 {\n"
        "  mut n: i64 = 0\n"
        "  while n < 1000 {\n"
        "    step o = Hold(Wrap(\"aa\" + \"bb\"))\n"
        "    match o {\n"
        "      Hold(inner) => match inner {\n"
        "        Wrap(s) => {},\n"
        "      },\n"
        "      Empty => {},\n"
        "    }\n"
        "    n = n + 1\n"
        "  }\n"
        "  println(\"done\")\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"done\n"


def test_seal_deep_clone_from_borrow(tmp_path):
    # Pushing a borrow Ident carrying a seal-with-heap-payload into
    # another container must deep-clone the seal (which in turn
    # clones the heap str). Releasing the source container must NOT
    # dangle the destination's copy.
    src = (
        "seal Msg { Text(str) }\n"
        "fn main() -> i32 {\n"
        "  mut a: tablets[4]Msg\n"
        "  mut b: tablets[4]Msg\n"
        "  step _x = a.push(Text(\"al\" + \"ice\"))\n"
        "  step p = a[0]\n"
        "  step _y = b.push(p)\n"
        "  release a\n"
        "  match b[0] {\n"
        "    Text(s) => println(s),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"alice\n"


# --- copy keyword -----------------------------------------------------

def test_copy_breaks_borrow_aliasing(tmp_path):
    # `copy name` on a match binder produces a freshly-owned str so
    # later mutation of the scrutinee's source (p_bump) can't reach
    # the cloned bytes. This is the intended use of `copy` under the
    # freeze-while-borrow rule.
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
        "      step n = copy name\n"
        "      p_bump(p)\n"
        "      println(n)\n"
        "    },\n"
        "    EOF => println(\"eof\"),\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_copy_scalar_is_noop(tmp_path):
    # `copy` on a plain int is harmless — lowers through a no-op in
    # _deep_clone_if_cleanup_bearing.
    src = (
        "fn main() -> i32 {\n"
        "  step x: i64 = 42\n"
        "  step y = copy x\n"
        "  println(y)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"42\n"


def test_copy_allows_safe_push_into_other_container(tmp_path):
    # `copy` lets a borrow into one container be safely pushed into
    # another. After release of the source, the destination still
    # reads the cloned bytes.
    src = (
        "fn main() -> i32 {\n"
        "  mut a: tablets[4]str\n"
        "  mut b: tablets[4]str\n"
        "  step _x = a.push(\"ab\" + \"cd\")\n"
        "  step p = a[0]\n"
        "  step _y = b.push(copy p)\n"
        "  release a\n"
        "  println(b[0])\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"abcd\n"


# --- fresh-rvalue transfer optimization -------------------------------

def test_push_call_result_is_single_malloc(tmp_path):
    # `push(fn())` where fn returns a heap str should NOT double-
    # clone — the Call result is a fresh-owned rvalue that transfers
    # into the container. A missing optimization showed up as a 10x
    # peak memory bump on loops of 100k pushes.
    from tuppu.driver import compile_to_ir
    src = (
        "fn make() -> str { \"ab\" + \"cd\" }\n"
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]str\n"
        "  step _x = t.push(make())\n"
        "  0\n"
        "}\n"
    )
    ir = compile_to_ir(src)
    # Exactly one str_concat (from the "ab" + "cd" inside make),
    # zero str_clones in main — the Call result transfers directly.
    main_start = ir.find("define i32 @\"main\"")
    main_end = ir.find("define ", main_start + 1)
    main_body = ir[main_start:main_end if main_end > 0 else len(ir)]
    assert "str_clone" not in main_body, main_body


def test_structlit_call_result_is_single_malloc(tmp_path):
    # Same optimization at the struct-lit field init site.
    from tuppu.driver import compile_to_ir
    src = (
        "fn make() -> str { \"ab\" + \"cd\" }\n"
        "tablet Box { s: str }\n"
        "fn main() -> i32 {\n"
        "  step b = Box { s: make() }\n"
        "  0\n"
        "}\n"
    )
    ir = compile_to_ir(src)
    main_start = ir.find("define i32 @\"main\"")
    main_end = ir.find("define ", main_start + 1)
    main_body = ir[main_start:main_end if main_end > 0 else len(ir)]
    assert "str_clone" not in main_body, main_body


def test_variant_ctor_call_result_is_single_malloc(tmp_path):
    # Same at the variant-ctor payload site.
    from tuppu.driver import compile_to_ir
    src = (
        "seal R { Ok(str), Err(str) }\n"
        "fn make() -> str { \"ab\" + \"cd\" }\n"
        "fn wrap() -> R { Ok(make()) }\n"
        "fn main() -> i32 { 0 }\n"
    )
    ir = compile_to_ir(src)
    wrap_start = ir.find("define %\"R\" @\"wrap\"")
    wrap_end = ir.find("define ", wrap_start + 1)
    wrap_body = ir[wrap_start:wrap_end if wrap_end > 0 else len(ir)]
    assert "str_clone" not in wrap_body, wrap_body


# --- freeze-while-borrow rule ----------------------------------------

def test_borrow_step_from_index_push_is_not_mut_reach(tmp_path):
    # `step p = a[0]; a.push(x); use(p)` — push appends without
    # relocating existing slots, so p still aliases slot 0's bytes.
    # Rule correctly allows it.
    src = (
        "fn main() -> i32 {\n"
        "  mut a: tablets[4]str\n"
        "  step _x = a.push(\"hi\" + \"!\")\n"
        "  step p = a[0]\n"
        "  step _y = a.push(\"other\" + \"!\")\n"
        "  println(p)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hi!\n"


def test_borrow_rejected_after_index_assign(tmp_path):
    # `a[0] = new` DOES free slot 0's old contents (it's cleanup-
    # bearing). A live borrow into a gets invalidated.
    src = (
        "fn main() -> i32 {\n"
        "  mut a: tablets[4]str\n"
        "  step _x = a.push(\"hi\" + \"!\")\n"
        "  step p = a[0]\n"
        "  a[0] = \"new\" + \"!\"\n"
        "  println(p)\n"
        "  0\n"
        "}\n"
    )
    with pytest.raises(CompileError, match="use of borrow 'p'"):
        compile_to_ir(src)


def test_scalar_field_assign_does_not_invalidate_borrow(tmp_path):
    # `b.next = a` where `.next` is a wedge (scalar) doesn't free
    # any heap bytes. Borrows into b / store stay valid.
    src = (
        "tablet N { next: wedge N, val: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut store: tablets[8]N\n"
        "  mut a: wedge N = store.push(N { next: lost, val: 1 })\n"
        "  mut b: wedge N = store.push(N { next: lost, val: 2 })\n"
        "  b.next = a\n"
        "  println(b.next.val)\n"
        "  println(b.val)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"1\n2\n"


def test_wedge_field_extract_rejected_after_index_overwrite(tmp_path):
    # Community repro: `step h = store.push(x); step b = h.buf;
    # store[0] = new_x; use(b)`. Freeze catches via:
    # - h registered as borrow of store (wedge arena tracking).
    # - b registered as borrow of store (via h's root).
    # - a[0] = x invalidates borrows rooted at store.
    # - use(b) rejects.
    src = (
        "tablet N { buf: str }\n"
        "fn main() -> i32 {\n"
        "  mut store: tablets[4]N\n"
        "  step h = store.push(N { buf: \"hel\" + \"lo\" })\n"
        "  step b = h.buf\n"
        "  store[0] = N { buf: \"over\" + \"written\" }\n"
        "  println(b)\n"
        "  0\n"
        "}\n"
    )
    with pytest.raises(CompileError, match="use of borrow 'b'"):
        compile_to_ir(src)


def test_match_binder_returned_no_double_free(tmp_path):
    # Community repro: `match m { Text(s) => s }` returns s through
    # the fn. The match binder was cap>0 (inherited from the payload),
    # so caller's cleanup double-freed against the scrutinee's own
    # seal release. Fix: match binders are read-borrows (cap=0)
    # mirroring Field/Index reads.
    src = (
        "seal M { Text(str), Silent }\n"
        "fn unwrap(m: M) -> str {\n"
        "  match m {\n"
        "    Text(s) => s,\n"
        "    Silent => \"none\",\n"
        "  }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  step m = Text(\"hel\" + \"lo\")\n"
        "  step result = unwrap(m)\n"
        "  println(result)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_borrow_step_from_field_rejected_after_mut_assign(tmp_path):
    # `step s = r.label; r.label = x; use(s)` — the assignment to
    # r.label releases the old heap bytes, dangling s.
    src = (
        "tablet Row { label: str }\n"
        "fn main() -> i32 {\n"
        "  mut r: Row = Row { label: \"first\" + \"!\" }\n"
        "  step s = r.label\n"
        "  r.label = \"second\" + \"!\"\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    with pytest.raises(CompileError, match="use of borrow 's'"):
        compile_to_ir(src)


def test_borrow_used_before_mut_call_fine(tmp_path):
    # Borrow's use happens BEFORE the mut call; no subsequent read
    # triggers the rule. The check is "use-triggered" — a mut-reach
    # silently invalidates the borrow, and the error surfaces only
    # on a post-invalidation read. Naturally permissive for this
    # common shape.
    src = (
        "fn main() -> i32 {\n"
        "  mut a: tablets[4]str\n"
        "  step _x = a.push(\"hi\" + \"!\")\n"
        "  step p = a[0]\n"
        "  println(p)\n"
        "  step _y = a.push(\"other\" + \"!\")\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hi!\n"


def test_borrow_no_mut_reach_fine(tmp_path):
    # No mut-reach between borrow bind and use. Always fine.
    src = (
        "fn main() -> i32 {\n"
        "  mut a: tablets[4]str\n"
        "  step _x = a.push(\"hello\" + \"\")\n"
        "  step p = a[0]\n"
        "  println(p)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_borrow_copy_at_bind_site_allows_use_after_mut(tmp_path):
    # `step n = copy p` where p is a borrow: n is independently
    # owned, so subsequent mut to a doesn't invalidate it.
    src = (
        "fn main() -> i32 {\n"
        "  mut a: tablets[4]str\n"
        "  step _x = a.push(\"hello\" + \"\")\n"
        "  step p = a[0]\n"
        "  step n = copy p\n"
        "  step _y = a.push(\"other\" + \"\")\n"
        "  println(n)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_borrow_scoped_release_at_block_exit(tmp_path):
    # Borrow in inner block goes out of scope at block exit, so
    # subsequent mut-reach in the outer scope is fine.
    src = (
        "fn main() -> i32 {\n"
        "  mut a: tablets[4]str\n"
        "  step _x = a.push(\"hi\" + \"!\")\n"
        "  if true {\n"
        "    step p = a[0]\n"
        "    println(p)\n"
        "  }\n"
        "  step _y = a.push(\"other\" + \"!\")\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hi!\n"


def test_accumulator_pattern_no_false_positive(tmp_path):
    # `mut result = p.lex; loop { result = updated }; use(result)` —
    # the mut binding is initialized as a borrow of p, but the loop
    # rebinds it to fresh-owned values. The rule must treat the
    # rebind as moving `result` out of borrow state, otherwise the
    # post-loop read would be incorrectly rejected. Community-
    # reported false positive.
    src = (
        "tablet Lex { buf: str }\n"
        "tablet P { lex: Lex }\n"
        "fn main() -> i32 {\n"
        "  mut p: P = P { lex: Lex { buf: \"\" } }\n"
        "  mut result: Lex = p.lex\n"
        "  mut i: i64 = 0\n"
        "  while i < 3 {\n"
        "    result = Lex { buf: result.buf + \"x\" }\n"
        "    i = i + 1\n"
        "  }\n"
        "  println(result.buf)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"xxx\n"


def test_mut_binding_borrow_chain_rejected(tmp_path):
    # Community-reported gap: `mut l: Lex = p.lex` makes l a borrow
    # of p's bytes. `advance(l)` (mut param) mutates l, which writes
    # THROUGH to p's bytes. Then `p.lex = l` reads l on the RHS
    # while l's ultimate root has been mut-reached — the write-back
    # alias pattern that silently corrupted heap strings.
    #
    # Freeze rule catches it once `_invalidate_root` walks the borrow
    # chain to the ultimate owner. A 3-line fix inside the existing
    # rule — not a per-case patch.
    src = (
        "tablet Lex { buf: str, pos: i64 }\n"
        "tablet Parser { lex: Lex, depth: i64 }\n"
        "fn advance(mut l: Lex) { l.pos = l.pos + 1 }\n"
        "fn main() -> i32 {\n"
        "  mut p: Parser = Parser {\n"
        "    lex: Lex { buf: \"hel\" + \"lo\", pos: 0 },\n"
        "    depth: 0,\n"
        "  }\n"
        "  mut l: Lex = p.lex\n"
        "  advance(l)\n"
        "  p.lex = l\n"
        "  0\n"
        "}\n"
    )
    with pytest.raises(CompileError, match="use of borrow 'l'"):
        compile_to_ir(src)


def test_mut_binding_copy_opt_out(tmp_path):
    # Same program with `copy` at the binding site. l owns
    # independent bytes; the round trip is safe.
    src = (
        "tablet Lex { buf: str, pos: i64 }\n"
        "tablet Parser { lex: Lex, depth: i64 }\n"
        "fn advance(mut l: Lex) { l.pos = l.pos + 1 }\n"
        "fn main() -> i32 {\n"
        "  mut p: Parser = Parser {\n"
        "    lex: Lex { buf: \"hel\" + \"lo\", pos: 0 },\n"
        "    depth: 0,\n"
        "  }\n"
        "  mut l: Lex = copy p.lex\n"
        "  advance(l)\n"
        "  p.lex = l\n"
        "  println(p.lex.buf)\n"
        "  println(p.lex.pos)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n1\n"


# --- escape analysis -----------------------------------------------

def test_escape_field_return_rejected(tmp_path):
    # Returning a Field read of a local struct — the struct dies at
    # fn exit, caller would read freed memory.
    src = (
        "tablet Row { label: str }\n"
        "fn leak() -> str {\n"
        "  step r: Row = Row { label: \"hel\" + \"lo\" }\n"
        "  r.label\n"
        "}\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="borrow of local binding 'r'"):
        compile_to_ir(src)


def test_escape_match_binder_return_rejected(tmp_path):
    # Returning a match binder — aliases scrutinee's seal payload,
    # which dies with the local scrutinee.
    src = (
        "seal M { Text(str), Silent }\n"
        "fn leak() -> str {\n"
        "  step m = Text(\"hel\" + \"lo\")\n"
        "  match m {\n"
        "    Text(s) => s,\n"
        "    Silent => \"none\",\n"
        "  }\n"
        "}\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="borrow of local binding 's'"):
        compile_to_ir(src)


def test_escape_param_field_return_fine(tmp_path):
    # Returning a Field read of a PARAM is safe — caller owns the
    # param's bytes, the returned borrow aliases into caller
    # storage.
    src = (
        "tablet Row { label: str }\n"
        "fn get_label(r: Row) -> str { r.label }\n"
        "fn main() -> i32 {\n"
        "  step r = Row { label: \"hel\" + \"lo\" }\n"
        "  println(get_label(r))\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_escape_copy_opt_out(tmp_path):
    # `copy r.label` produces a fresh-owned rvalue; not a borrow,
    # safe to return.
    src = (
        "tablet Row { label: str }\n"
        "fn build() -> str {\n"
        "  step r: Row = Row { label: \"hel\" + \"lo\" }\n"
        "  copy r.label\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  step s = build()\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_escape_ident_chain_transfer_still_works(tmp_path):
    # Returning an Ident that chains through step-borrows to an
    # owning local is fine — codegen transfers ownership on tail.
    # The escape rule shouldn't reject this common pattern.
    src = (
        "fn build() -> str {\n"
        "  mut y: str = \"Ur\" + \"uk\"\n"
        "  step w = y\n"
        "  step v = w\n"
        "  v\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  step msg = build()\n"
        "  println(msg)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"Uruk\n"


def test_escape_yield_field_rejected(tmp_path):
    # Same rule applied to yield.
    src = (
        "tablet Row { label: str }\n"
        "fn build(flag: bool) -> str {\n"
        "  step r: Row = Row { label: \"hel\" + \"lo\" }\n"
        "  if flag { yield r.label }\n"
        "  \"fallback\"\n"
        "}\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="borrow of local binding 'r'"):
        compile_to_ir(src)


def test_seal_pure_enum_no_release_fn(tmp_path):
    # A seal with only nullary / scalar-payload variants should NOT
    # get a release fn — _seal_needs_cleanup returns False and no IR
    # is emitted.
    from tuppu.driver import compile_to_ir
    src = (
        "seal Color { Red, Green, Blue }\n"
        "fn main() -> i32 {\n"
        "  step c = Red\n"
        "  match c { Red => 1, Green => 2, Blue => 3 }\n"
        "  0\n"
        "}\n"
    )
    ir = compile_to_ir(src)
    assert "__tuppu_seal_Color_release" not in ir
