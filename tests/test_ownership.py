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

from tuppu.driver import compile_to_binary


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
    # slot's reassignment releases... currently shallow (seals have
    # no release helper yet), but the read after reassignment must
    # hit the new payload, not the old, and execution must complete
    # cleanly.
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
