"""dvec MVP tests.

`dvec<T>` is a contiguous heap-allocated array of T values stored
inline. The two facts worth pinning:

1. **O(1) random access via single load**: `buf + i*sizeof(T)` is
   the slot's address; we load T directly from there.
2. **Buffer trace fn walks T-typed slots**: each `dvec<T>` has its
   own per-T trace fn that reads cap from the GC header (via
   `__tuppu_gc_data_size`) and recurses through T's full tracing
   logic for each slot. Composite T (struct holding str, etc.) keeps
   its inner heap state alive correctly.

Pointer-instability is a deliberate non-feature: grow memcpys the
inline T bytes, so `push` returns unit (no handle to dangle on next
push). Compare with `ivec<T>` — pointer-stable but two loads per
index, and per-element heap allocation.
"""
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


def test_dvec_push_get_basic(tmp_path, stress):
    src = """
fn main() -> i32 {
  mut dv: dvec<i64>
  dv.push(10)
  dv.push(20)
  dv.push(30)
  println(dv.len)
  println(dv[0])
  println(dv[1])
  println(dv[2])
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"3\n10\n20\n30\n"


def test_dvec_for_iteration(tmp_path, stress):
    src = """
fn main() -> i32 {
  mut dv: dvec<i64>
  dv.push(1)
  dv.push(2)
  dv.push(3)
  dv.push(4)
  for x in dv { println(x) }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"1\n2\n3\n4\n"


def test_dvec_grows_across_doublings(tmp_path, stress):
    # Same shape as the ivec test — 200 elements, multiple grows,
    # all earlier elements survive memcpy on every doubling.
    src = """
fn main() -> i32 {
  mut dv: dvec<i64>
  mut i: i64 = 0
  while i < 200 {
    dv.push(i * 7)
    i = i + 1
  }
  println(dv.len)
  println(dv[0])
  println(dv[42])
  println(dv[199])
  mut j: i64 = 0
  mut sum: i64 = 0
  while j < dv.len {
    sum = sum + dv[j]
    j = j + 1
  }
  println(sum)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"200\n0\n294\n1393\n139300\n"


def test_dvec_set(tmp_path, stress):
    src = """
fn main() -> i32 {
  mut dv: dvec<i64>
  dv.push(1)
  dv.push(2)
  dv.push(3)
  dv[1] = 999
  println(dv[0])
  println(dv[1])
  println(dv[2])
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"1\n999\n3\n"


def test_dvec_str_payload(tmp_path, stress):
    # The marquee correctness test for the per-T buffer trace fn.
    # Each str slot stores `{ptr, len, cap}` inline. The trace fn
    # must recurse into each slot's str descriptor (mark_ptr at
    # offset 0) so the str's heap bytes stay live across stress-mode
    # collections. Pre-fix this returned blank strings.
    src = """
fn main() -> i32 {
  mut dv: dvec<str>
  mut i: i64 = 0
  while i < 50 {
    dv.push("item_" + int_to_str(i))
    i = i + 1
  }
  println(dv.len)
  mut acc: str = ""
  mut j: i64 = 0
  while j < 100 {
    acc = acc + "x"
    j = j + 1
  }
  println(dv[0])
  println(dv[25])
  println(dv[49])
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"50\nitem_0\nitem_25\nitem_49\n"


def test_dvec_struct_payload_with_str(tmp_path, stress):
    # Composite T inline. Trace must recurse into each struct slot's
    # str field at the right offset.
    src = """
tablet Entry { name: str, count: i64 }

fn main() -> i32 {
  mut dv: dvec<Entry>
  dv.push(Entry { name: "barley" + "1", count: 10 })
  dv.push(Entry { name: "emmer" + "2",  count: 6  })
  dv.push(Entry { name: "wheat" + "3",  count: 0  })
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "."
    i = i + 1
  }
  for e in dv {
    println(e.name)
    println(e.count)
  }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == (
        b"barley1\n10\n"
        b"emmer2\n6\n"
        b"wheat3\n0\n"
    )


def test_dvec_returned_from_fn_survives_caller_gc(tmp_path, stress):
    src = """
fn build() -> dvec<i64> {
  mut dv: dvec<i64>
  mut i: i64 = 0
  while i < 50 {
    dv.push(i)
    i = i + 1
  }
  dv
}

fn main() -> i32 {
  step v = build()
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "x"
    i = i + 1
  }
  println(v.len)
  println(v[0])
  println(v[49])
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"50\n0\n49\n"


def test_dvec_push_returns_unit(tmp_path, stress):
    # Sanity: dvec.push doesn't return a useful value (unlike
    # ivec.push). Used as a statement; the typechecker and codegen
    # must accept it without trying to bind a result.
    src = """
fn main() -> i32 {
  mut dv: dvec<i64>
  dv.push(42)
  println(dv[0])
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"42\n"


def test_dvec_indexed_through_struct_field(tmp_path):
    """Parallel to the ivec field-routed-index regression: a `dvec<T>`
    accessed via a struct field (or `self.v` in an edubba method)
    used to error in the `_gen_index` fallback because only tablets
    and str had SSA-value branches there."""
    import os, subprocess
    from tuppu.driver import compile_files_to_binary, stdlib_files
    src = """
tablet Box {
  v: dvec<i64>
}

edubba Box {
  fn first(self) -> i64 { self.v[0] }
}

fn main() -> i32 {
  mut b: Box
  b.v.push(40)
  b.v.push(2)
  (b.first() + b.v[1]) as i32
}
"""
    user = tmp_path / "main.tpu"
    user.write_text(src)
    binary = compile_files_to_binary(stdlib_files() + [user], tmp_path, name="prog")
    r = subprocess.run([str(binary)], capture_output=True)
    assert r.returncode == 42
