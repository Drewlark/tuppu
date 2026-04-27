"""ivec MVP tests.

`ivec<T>` is a contiguous heap-allocated array of pointers to per-T
heap allocations. The two facts worth pinning:

1. **O(1) random access**: index into the pointer array, then deref.
2. **Pointer stability**: each T's address is independent of the
   pointer-array buffer, so resize-on-grow leaves wedges valid.

Each test runs in normal and stress modes (the `stress` fixture).
Stress mode forces a collection on every allocation, surfacing
missed root pushes immediately.
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


def test_ivec_push_get_basic(tmp_path, stress):
    src = """
fn main() -> i32 {
  mut iv: ivec<i64>
  iv.push(10)
  iv.push(20)
  iv.push(30)
  println(iv.len)
  println(iv[0])
  println(iv[1])
  println(iv[2])
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"3\n10\n20\n30\n"


def test_ivec_for_iteration(tmp_path, stress):
    src = """
fn main() -> i32 {
  mut iv: ivec<i64>
  iv.push(1)
  iv.push(2)
  iv.push(3)
  iv.push(4)
  for x in iv { println(x) }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"1\n2\n3\n4\n"


def test_ivec_grows_across_doublings(tmp_path, stress):
    # 200 i64 elements — past the initial cap 8 and several doublings
    # (8 → 16 → 32 → 64 → 128 → 256). All earlier elements must
    # survive each grow, including under stress mode.
    src = """
fn main() -> i32 {
  mut iv: ivec<i64>
  mut i: i64 = 0
  while i < 200 {
    iv.push(i * 7)
    i = i + 1
  }
  println(iv.len)
  println(iv[0])
  println(iv[42])
  println(iv[199])
  mut j: i64 = 0
  mut sum: i64 = 0
  while j < iv.len {
    sum = sum + iv[j]
    j = j + 1
  }
  println(sum)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    # sum = 7 * (0+1+...+199) = 7 * 19900 = 139300
    assert stdout == b"200\n0\n294\n1393\n139300\n"


def test_ivec_pointer_stability_across_grow(tmp_path, stress):
    # The marquee ivec property: a wedge into iv[0] taken before a
    # grow is still valid afterwards. With a contiguous T array
    # (dvec) this would dangle.
    src = """
tablet Item { v: i64 }

fn main() -> i32 {
  mut iv: ivec<Item>
  step h0 = iv.push(Item { v: 100 })
  step h1 = iv.push(Item { v: 200 })
  mut i: i64 = 0
  while i < 500 {
    iv.push(Item { v: i })
    i = i + 1
  }
  println(h0.v)
  println(h1.v)
  println(iv.len)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"100\n200\n502\n"


def test_ivec_set(tmp_path, stress):
    # Set writes through to the existing per-element heap allocation,
    # so the slot's address remains stable.
    src = """
fn main() -> i32 {
  mut iv: ivec<i64>
  iv.push(1)
  iv.push(2)
  iv.push(3)
  iv[1] = 999
  println(iv[0])
  println(iv[1])
  println(iv[2])
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"1\n999\n3\n"


def test_ivec_str_payload(tmp_path, stress):
    # Cleanup-bearing T: str. Each str's heap bytes must stay live
    # while the ivec holds them, even under stress.
    src = """
fn main() -> i32 {
  mut iv: ivec<str>
  iv.push("hello" + "_world")
  iv.push("babylonian" + "_tablets")
  iv.push("ivec" + "_lives")
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "."
    i = i + 1
  }
  for s in iv { println(s) }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"hello_world\nbabylonian_tablets\nivec_lives\n"


def test_ivec_struct_payload_with_str(tmp_path, stress):
    # Composite T: struct holding a str. Per-element heap allocation
    # is sized for the struct; GC must trace inside via T's
    # descriptor (set up by codegen at allocation time).
    src = """
tablet Entry { name: str, count: i64 }

fn main() -> i32 {
  mut iv: ivec<Entry>
  iv.push(Entry { name: "barley" + "1", count: 10 })
  iv.push(Entry { name: "emmer" + "2",  count: 6  })
  iv.push(Entry { name: "wheat" + "3",  count: 0  })
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 {
    acc = acc + "."
    i = i + 1
  }
  for e in iv {
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


def test_ivec_returned_from_fn_survives_caller_gc(tmp_path, stress):
    # Build an ivec inside a callee and return it. The GC must keep
    # the ivec value, the buffer, and every per-element heap T alive
    # across caller-side allocations.
    src = """
fn build() -> ivec<i64> {
  mut iv: ivec<i64>
  mut i: i64 = 0
  while i < 50 {
    iv.push(i)
    i = i + 1
  }
  iv
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
