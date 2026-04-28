"""Method-call dispatch for `Map<T>`. The stdlib `map_set` /
`map_get` / `map_has` / `map_len` / `map_index` free functions are
exposed through method syntax (`m.set(k, v)`, etc.) so receivers can
be any lvalue path — Field, Index, or Ident — without forcing the
caller to bind the map to a mut Ident first. Each test runs in both
normal and GC-stress mode."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_files_to_binary, stdlib_files


@pytest.fixture(params=[False, True], ids=["normal", "stress"])
def stress(request):
    return request.param


def run(src: str, tmp_path: Path, stress: bool) -> tuple[int, bytes]:
    user = tmp_path / "main.tpu"
    user.write_text(src)
    binary = compile_files_to_binary(
        stdlib_files() + [user], tmp_path, name="prog",
    )
    env = dict(os.environ)
    if stress:
        env["TUPPU_GC_STRESS"] = "1"
    r = subprocess.run([str(binary)], capture_output=True, env=env)
    return r.returncode, r.stdout


def test_map_methods_on_ident(tmp_path, stress):
    # Baseline: receiver is a mut-bound Ident. All five methods should
    # behave identically to their free-function counterparts.
    src = """
fn main() -> i32 {
  mut m: Map<i64>
  m.set("a", 1)
  m.set("b", 2)
  m.set("c", 3)
  m.set("a", 10)
  println("len=", m.len())
  println(m.get("a", -1))
  println(m.get("b", -1))
  println(m.get("c", -1))
  println(m.get("z", -99))
  if m.has("b") { println("has b") }
  if m.has("nope") { println("nope") }
  println("idx=", m.index("c"))
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"len=3\n10\n2\n3\n-99\nhas b\nidx=2\n"


def test_map_set_via_field_lvalue(tmp_path, stress):
    # Receiver is a Field path on a wedge into tablets storage. The
    # free-function form `map_set(mut wedge.field, ...)` is rejected by
    # the mut-Ident rule; method dispatch accepts the lvalue path.
    src = """
tablet Bag { items: Map<i64> }

fn main() -> i32 {
  mut bags: tablets[4]Bag
  mut empty: Bag
  step bag: wedge Bag = bags.push(empty)
  bag.items.set("alpha", 100)
  bag.items.set("beta",  200)
  bag.items.set("alpha", 999)
  println("len=", bag.items.len())
  println(bag.items.get("alpha", -1))
  println(bag.items.get("beta",  -1))
  if bag.items.has("alpha") { println("has alpha") }
  if bag.items.has("zeta")  { println("zeta") }
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"len=2\n999\n200\nhas alpha\n"


def test_map_set_via_index_lvalue(tmp_path, stress):
    # Receiver is an Index path through tablets — `slots[i].m.set(...)`.
    # Each slot gets its own per-i payload so we can spot any aliasing
    # between map instances.
    src = """
tablet Slot { m: Map<i64> }

fn main() -> i32 {
  mut slots: tablets[8]Slot
  mut empty: Slot
  slots.push(empty)
  slots.push(empty)
  slots.push(empty)
  mut i: i64 = 0
  while i < 3 {
    slots[i].m.set("x", i * 10)
    slots[i].m.set("y", i * 100)
    i += 1
  }
  println(slots[0].m.get("x", -1), " ", slots[0].m.get("y", -1))
  println(slots[1].m.get("x", -1), " ", slots[1].m.get("y", -1))
  println(slots[2].m.get("x", -1), " ", slots[2].m.get("y", -1))
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"0 0\n10 100\n20 200\n"


def test_map_str_payload(tmp_path, stress):
    # Map<str> exercises the heap-string ownership path in `set` and
    # `get`. The `set("name", ...)` overwrite must not double-free the
    # previous value's bytes.
    src = """
fn main() -> i32 {
  mut env: Map<str>
  env.set("name", "tuppu")
  env.set("ver",  "0.1")
  env.set("name", "Tuppu")
  println("len=", env.len())
  println(env.get("name", "?"))
  println(env.get("ver",  "?"))
  println(env.get("missing", "fallback"))
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"len=2\nTuppu\n0.1\nfallback\n"


def test_map_seal_payload(tmp_path, stress):
    # Map<seal-with-str> — the JSON-tree shape that motivated this
    # whole branch. `set("name", JStr("Tuppu"))` after a previous
    # `set("name", JStr("tuppu"))` must release the old payload and
    # store the new one without corrupting either.
    src = """
seal JV {
  JNull,
  JStr(str),
  JInt(i64),
}

fn main() -> i32 {
  mut m: Map<JV>
  m.set("name", JStr("tuppu"))
  m.set("ver",  JInt(1))
  m.set("name", JStr("Tuppu"))
  println("len=", m.len())
  step name: JV = m.get("name", JNull)
  step ver:  JV = m.get("ver",  JNull)
  match name {
    JStr(s) => println("name=", s),
    JInt(i) => println("name int ", i),
    JNull   => println("name null"),
  }
  match ver {
    JStr(s) => println("ver=", s),
    JInt(i) => println("ver=", i),
    JNull   => println("ver null"),
  }
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"len=2\nname=Tuppu\nver=1\n"


def test_map_methods_coexist_with_free_fns(tmp_path, stress):
    # The free-function forms must keep working — this slice is purely
    # additive. Mix method calls and `map_set` / `map_get` on the same
    # binding to prove there's no shadowing or release double-up.
    src = """
fn main() -> i32 {
  mut m: Map<i64>
  m.set("a", 1)
  map_set(m, "b", 2)
  m.set("c", 3)
  println(map_get(m, "a", -1))
  println(m.get("b", -1))
  println(map_has(m, "c"))
  println(m.has("c"))
  println(map_len(m))
  println(m.len())
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"1\n2\ntrue\ntrue\n3\n3\n"


def test_map_unknown_method_rejected(tmp_path, stress):
    # Misspelled methods produce a typecheck error that lists what's
    # available. Compile-time only — no need to run.
    src = """
fn main() -> i32 {
  mut m: Map<i64>
  m.bogus("x", 1)
  0
}
"""
    user = tmp_path / "main.tpu"
    user.write_text(src)
    with pytest.raises(Exception) as excinfo:
        compile_files_to_binary(
            stdlib_files() + [user], tmp_path, name="prog",
        )
    msg = str(excinfo.value)
    assert "Map has no method 'bogus'" in msg
    assert "set" in msg and "get" in msg


def test_map_method_arity_mismatch_rejected(tmp_path, stress):
    # Wrong number of args at a method site goes through the same
    # generic-fn unification as the free-function form, so the error
    # path is shared. Also compile-time only.
    src = """
fn main() -> i32 {
  mut m: Map<i64>
  m.set("a")
  0
}
"""
    user = tmp_path / "main.tpu"
    user.write_text(src)
    with pytest.raises(Exception) as excinfo:
        compile_files_to_binary(
            stdlib_files() + [user], tmp_path, name="prog",
        )
    assert "map_set" in str(excinfo.value)
