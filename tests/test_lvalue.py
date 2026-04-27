"""Lvalue torture: every shape of `lhs = rhs` we want to support
when `lhs` is a chain of field/index access. The parser and codegen
both walk the chain recursively; these tests pin every realistic
combination so the next regression points at exactly one shape."""
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


def test_field_then_index_assign(tmp_path, stress):
    # m.values[i] = v — the canonical Map-style write that motivated
    # the lvalue extension. Field navigates to the inner tablets,
    # Index addresses the slot.
    src = """
tablet Holder { values: tablets[16]i64 }

fn main() -> i32 {
  mut h: Holder
  step _a = h.values.push(10)
  step _b = h.values.push(20)
  step _c = h.values.push(30)
  h.values[1] = 99
  println(h.values[0])
  println(h.values[1])
  println(h.values[2])
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"10\n99\n30\n"


def test_field_then_index_assign_str_payload(tmp_path, stress):
    # Same but the element is a heap-cleanup-bearing str. The
    # overwrite has to free / re-root through the GC layer.
    src = """
tablet Inbox { msgs: tablets[16]str }

fn main() -> i32 {
  mut x: Inbox
  step _a = x.msgs.push("first" + "_msg")
  step _b = x.msgs.push("second" + "_msg")
  step _c = x.msgs.push("third" + "_msg")
  x.msgs[1] = "replaced" + "!"
  println(x.msgs[0])
  println(x.msgs[1])
  println(x.msgs[2])
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"first_msg\nreplaced!\nthird_msg\n"


def test_index_then_field_assign_through_step_handle(tmp_path, stress):
    # `arr[n].field = v` was already supported pre-fix — the lvalue
    # walk terminates at a tablets binding (Index whose target is the
    # Ident). Pinned here as a cross-check that the new generalized
    # walk doesn't regress it.
    src = """
tablet Row { label: str, count: i64 }

fn main() -> i32 {
  mut t: tablets[8]Row
  step _r = t.push(Row { label: "alpha", count: 1 })
  t[0].count = 42
  t[0].label = "beta"
  println(t[0].label)
  println(t[0].count)
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"beta\n42\n"


def test_nested_field_then_index_two_levels_deep(tmp_path, stress):
    # outer.inner.values[i] = v — two levels of field navigation
    # before the index. Confirms the recursive _lvalue_slot walk
    # handles arbitrary chain depth.
    src = """
tablet Inner { values: tablets[16]i64 }
tablet Outer { inner: Inner, name: str }

fn main() -> i32 {
  mut o: Outer = Outer { inner: Inner { values: tablets[16]i64 { } }, name: "root" }
  step _a = o.inner.values.push(1)
  step _b = o.inner.values.push(2)
  step _c = o.inner.values.push(3)
  o.inner.values[1] = 88
  println(o.name)
  println(o.inner.values[0])
  println(o.inner.values[1])
  println(o.inner.values[2])
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"root\n1\n88\n3\n"


def test_field_index_then_field_assign(tmp_path, stress):
    # m.entries[i].count = n — Index produces a struct slot, Field
    # navigates inside it. Mixes both extensions.
    src = """
tablet Entry { name: str, count: i64 }
tablet Bag { entries: tablets[16]Entry }

fn main() -> i32 {
  mut b: Bag
  step _a = b.entries.push(Entry { name: "alice", count: 0 })
  step _b = b.entries.push(Entry { name: "bob", count: 0 })
  b.entries[0].count = 5
  b.entries[1].count = 9
  println(b.entries[0].name)
  println(b.entries[0].count)
  println(b.entries[1].name)
  println(b.entries[1].count)
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"alice\n5\nbob\n9\n"


def test_compound_field_then_index_assign(tmp_path, stress):
    # `+=` should desugar into the same lvalue path. Pin both arith
    # and string += through the new chain.
    src = """
tablet H { vs: tablets[8]i64 }

fn main() -> i32 {
  mut h: H
  step _a = h.vs.push(10)
  step _b = h.vs.push(20)
  h.vs[0] += 5
  h.vs[1] -= 3
  println(h.vs[0])
  println(h.vs[1])
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"15\n17\n"


def test_local_buffer_indexed_assign(tmp_path, stress):
    # Buffer is a fixed-size inline array — separate codegen path
    # inside _lvalue_slot (ArrayType vs tablets). Buffer-as-field is
    # forbidden by typecheck, so the buffer lives as a local.
    src = """
fn main() -> i32 {
  mut buf: buffer[8]u8
  buf[0] = 65 as u8
  buf[1] = 66 as u8
  buf[2] = 67 as u8
  step s: str = buffer_to_str(buf, 3)
  println(s)
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"ABC\n"


def test_generic_struct_field_index_assign(tmp_path, stress):
    # The Map<T> shape: a generic struct wrapping `tablets[N]T` whose
    # mut method writes into the field by index. Combines the lvalue
    # extension with the generic-struct mut-param fix.
    src = """
tablet Cache<T> { items: tablets[16]T }

fn cache_set<T>(mut c: Cache<T>, i: i64, v: T) {
  c.items[i] = v
}

fn main() -> i32 {
  mut c: Cache<i64>
  step _a = c.items.push(0)
  step _b = c.items.push(0)
  step _c = c.items.push(0)
  cache_set(c, 0, 11)
  cache_set(c, 1, 22)
  cache_set(c, 2, 33)
  println(c.items[0])
  println(c.items[1])
  println(c.items[2])
  0
}
"""
    rc, out = run(src, tmp_path, stress)
    assert rc == 0
    assert out == b"11\n22\n33\n"
