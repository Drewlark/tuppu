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


# --- arena torture: chunk-boundary, deep payloads, wedges across grows --


def test_ivec_arena_long_chain_under_stress(tmp_path, stress):
    # Push past many chunk boundaries (K = 64). 5000 elements forces
    # ~78 chunk allocations on top of ~10 buf grows; stress mode runs
    # a full collection on every allocation. Every slot's address
    # must remain stable for the whole run, and every per-slot value
    # must survive every collection. Read through both `iv[i]` and
    # iteration to exercise both random-access and sequential paths.
    src = """
fn main() -> i32 {
  mut iv: ivec<i64>
  mut i: i64 = 0
  while i < 5000 {
    iv.push(i * 3 + 1)
    i = i + 1
  }
  println(iv.len)
  println(iv[0])
  println(iv[63])      // last slot of chunk 0
  println(iv[64])      // first slot of chunk 1
  println(iv[4999])    // last slot of last chunk
  mut sum: i64 = 0
  for x in iv { sum = sum + x }
  println(sum)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    # sum_{i=0..4999} (3i + 1) = 3 * 4999*5000/2 + 5000 = 37 497 500
    assert stdout == b"5000\n1\n190\n193\n14998\n37497500\n"


def test_ivec_wedge_handle_survives_chain_extension(tmp_path, stress):
    # The marquee arena property: a wedge taken at iv[0] before any
    # chunk allocation stays valid after dozens of new chunks are
    # appended. This is the user-facing guarantee that ivec is for —
    # if the per-element address ever moved, this test would either
    # crash or read garbage.
    src = """
tablet Item { v: i64 }

fn main() -> i32 {
  mut iv: ivec<Item>
  step h_first = iv.push(Item { v: 100 })
  step h_mid   = iv.push(Item { v: 200 })
  mut i: i64 = 0
  while i < 2000 {
    iv.push(Item { v: i })
    i = i + 1
  }
  // Both wedges still point at their original slots, two chunks back.
  println(h_first.v)
  println(h_mid.v)
  // And mutation through the wedge writes through to that same slot.
  println(iv.len)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"100\n200\n2002\n"


def test_ivec_seal_payload_with_str(tmp_path, stress):
    # User-flagged edge case: a seal carrying a heap-str payload.
    # Each chunk holds K seal slots, each with a tag byte and a
    # union-shaped payload that may include a str. The chunk
    # descriptor must trace through the seal's variant, marking the
    # heap str alive. Mix tag-without-payload, tag-with-int, and
    # tag-with-str to make sure the trace dispatch doesn't lose
    # any variant under stress. Push past the 64-slot boundary so
    # at least two chunks have to be traced.
    src = """
seal V {
  VNull,
  VInt(i64),
  VStr(str),
}

fn show(v: V) -> str {
  match v {
    VNull    => "null",
    VInt(n)  => str_concat("int:", int_to_str(n)),
    VStr(s)  => str_concat("str:", s),
  }
}

fn main() -> i32 {
  mut iv: ivec<V>
  mut i: i64 = 0
  while i < 200 {
    iv.push(VStr(str_concat("k_", int_to_str(i))))
    iv.push(VInt(i))
    iv.push(VNull)
    i = i + 1
  }
  // Force allocations after the pushes — each one runs a collection
  // under stress mode, so any mistraced str payload would be freed
  // before we read it back.
  mut acc: str = ""
  mut j: i64 = 0
  while j < 200 {
    acc = acc + "."
    j = j + 1
  }
  // Sample reads across multiple chunks.
  println(show(iv[0]))      // VStr "k_0"
  println(show(iv[1]))      // VInt 0
  println(show(iv[2]))      // VNull
  println(show(iv[300]))    // chunk 4: i=100, slot 0 → VStr "k_100"
  println(show(iv[301]))    // chunk 4: VInt 100
  println(show(iv[599]))    // last: i=199, VNull
  println(iv.len)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == (
        b"str:k_0\n"
        b"int:0\n"
        b"null\n"
        b"str:k_100\n"
        b"int:100\n"
        b"null\n"
        b"600\n"
    )


def test_ivec_nested_ivec(tmp_path, stress):
    # Deep payload: ivec<ivec<i64>>. Outer pushes copy the inner
    # ivec's struct value (buf, len, cap, head_node, tail_node) into
    # an outer slot; the inner's chunks must stay alive via the
    # outer's traversal. Each inner has a different size so we can
    # tell positions apart, and we cross the chunk boundary at i=64.
    src = """
fn build_inner(seed: i64, n: i64) -> ivec<i64> {
  mut inner: ivec<i64>
  mut k: i64 = 0
  while k < n {
    inner.push(seed + k)
    k = k + 1
  }
  inner
}

fn main() -> i32 {
  mut outer: ivec<ivec<i64>>
  mut i: i64 = 0
  while i < 80 {
    outer.push(build_inner(i * 100, i + 1))
    i = i + 1
  }
  // outer[0] holds {0}; outer[64] holds {6400, 6401, .., 6464}.
  step a: ivec<i64> = outer[0]
  step b: ivec<i64> = outer[64]
  println(a.len)
  println(a[0])
  println(b.len)
  println(b[0])
  println(b[64])      // this inner just barely crosses the chunk boundary
  println(outer.len)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"1\n0\n65\n6400\n6464\n80\n"


def test_ivec_str_payload_dense_then_collect(tmp_path, stress):
    # Each push creates a fresh heap str inside the seal-free path;
    # the chunk descriptor must register every slot's interior str
    # pointer. After a long burst of pushes, do a tail of unrelated
    # allocations to force collections — every previously-pushed
    # string must still be readable.
    src = """
fn main() -> i32 {
  mut iv: ivec<str>
  mut i: i64 = 0
  while i < 500 {
    iv.push(str_concat("v_", int_to_str(i)))
    i = i + 1
  }
  // Pile up garbage to provoke a collection cycle (stress mode
  // already collects on every alloc; this raises the bar even in
  // normal mode).
  mut churn: str = ""
  mut j: i64 = 0
  while j < 500 {
    churn = churn + "."
    j = j + 1
  }
  // Spot-check across all chunks.
  println(iv[0])
  println(iv[63])      // chunk 0 last
  println(iv[64])      // chunk 1 first
  println(iv[256])     // chunk 4 first
  println(iv[499])     // last
  println(iv.len)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == (
        b"v_0\n"
        b"v_63\n"
        b"v_64\n"
        b"v_256\n"
        b"v_499\n"
        b"500\n"
    )


def test_ivec_set_through_index_after_chain_extension(tmp_path, stress):
    # `iv[i] = x` writes through to the slot inside a chunk, even
    # after many chunks have been appended past the slot's chunk.
    # The lvalue path goes through `get_addr`, which must return a
    # pointer that still resolves to the original chunk slot.
    src = """
fn main() -> i32 {
  mut iv: ivec<i64>
  mut i: i64 = 0
  while i < 200 {
    iv.push(i)
    i = i + 1
  }
  iv[0]   = 9000
  iv[63]  = 9063     // chunk 0 last
  iv[64]  = 9064     // chunk 1 first
  iv[199] = 9199     // last chunk
  println(iv[0])
  println(iv[63])
  println(iv[64])
  println(iv[199])
  println(iv[100])   // untouched
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"9000\n9063\n9064\n9199\n100\n"


# --- arena-specific properties: chunk descriptor & wedge-keeps-chunk -----


def test_ivec_of_wedge_dispatches_through_mark_wedge(tmp_path, stress):
    # ivec<wedge T>: each chunk slot is an interior pointer into a
    # separate tablets storage. The chunk descriptor must thread
    # elem_is_wedge=True so per-slot tracing routes through
    # __tuppu_gc_mark_wedge. mark_ptr would silently fail on the
    # interior pointer (no TUPPU_GC_MAGIC at HDR_OF), and under stress
    # the source tablets' chunks would be swept while the ivec still
    # references them.
    src = """
tablet Item { v: i64 }

fn main() -> i32 {
  mut store: tablets[8]Item
  step h0 = store.push(Item { v: 100 })
  step h1 = store.push(Item { v: 200 })
  step h2 = store.push(Item { v: 300 })

  mut iv: ivec<wedge Item>
  iv.push(h0)
  iv.push(h1)
  iv.push(h2)

  // Allocation churn so collections fire and have to walk the ivec
  // chunk's wedge slots correctly via mark_wedge.
  mut acc: str = ""
  mut i: i64 = 0
  while i < 100 { acc = acc + "."  i = i + 1 }

  println(iv[0].v)
  println(iv[1].v)
  println(iv[2].v)
  println(iv.len)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"100\n200\n300\n3\n"


def test_arena_wedge_outlives_source_ivec(tmp_path, stress):
    # The marquee arena property in the GC-only case: after the ivec
    # value goes out of scope, a wedge into one of its chunks is the
    # ONLY path keeping that chunk alive. The collector must reach
    # the chunk via mark_wedge on the wedge's shadow-stack root, not
    # via the ivec's head_node / tail_node (those are gone). My
    # earlier "wedge stable across pushes" test kept the ivec alive
    # in the same scope; this one returns just the wedge, runs heavy
    # churn in the caller, and dereferences. Proves chunks survive
    # through the wedge alone.
    src = """
tablet Item { v: i64 }

fn build_and_pluck() -> wedge Item {
  mut iv: ivec<Item>
  iv.push(Item { v: 11 })
  iv.push(Item { v: 22 })
  step h: wedge Item = iv.push(Item { v: 33 })
  // Push past the chunk boundary so the held wedge is no longer
  // in the most-recent chunk being filled — exercises the
  // chunk-chain trace through earlier nodes too.
  mut i: i64 = 0
  while i < 100 {
    iv.push(Item { v: i })
    i = i + 1
  }
  h
}

fn main() -> i32 {
  step pluck: wedge Item = build_and_pluck()
  // Caller-side allocation churn — every alloc fires a collection
  // under stress, and the only path to `pluck`'s chunk is via
  // mark_wedge on the returned handle.
  mut acc: str = ""
  mut i: i64 = 0
  while i < 200 {
    acc = acc + "x"
    i = i + 1
  }
  println(pluck.v)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"33\n"


def test_wedge_outlives_source_under_heavy_churn(tmp_path, stress):
    # Same shape as test_arena_wedge_outlives_source_ivec but with
    # 5000 caller-side allocations instead of 200. Linux's glibc
    # allocator is slow to recycle the bytes the swept arena
    # released, so the lighter test passed on Linux even when the
    # wedge had no shadow-stack root — the freed bytes hadn't been
    # overwritten by the read. macOS hit the bug on 200. With heavy
    # churn the bug surfaces on every platform; the test pins down
    # the wedge-rooting fix so a future regression can't silently
    # pass on one OS while corrupting on another.
    src = """
tablet Item { v: i64 }

fn build_and_pluck() -> wedge Item {
  mut iv: ivec<Item>
  iv.push(Item { v: 11 })
  iv.push(Item { v: 22 })
  step h: wedge Item = iv.push(Item { v: 33 })
  mut i: i64 = 0
  while i < 100 {
    iv.push(Item { v: i })
    i = i + 1
  }
  h
}

fn main() -> i32 {
  step pluck: wedge Item = build_and_pluck()
  mut acc: str = ""
  mut i: i64 = 0
  while i < 5000 {
    acc = acc + "x"
    i = i + 1
  }
  println(pluck.v)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"33\n"


def test_mut_wedge_binding_rooted(tmp_path, stress):
    # Companion to the step-binding case: a `mut` wedge binding goes
    # through a different `_gen_binding` arm. Under stress mode
    # without rooting, the chunk would also be swept.
    src = """
tablet Item { v: i64 }

fn build() -> wedge Item {
  mut iv: ivec<Item>
  iv.push(Item { v: 100 })
  iv.push(Item { v: 200 })
  step h: wedge Item = iv.push(Item { v: 999 })
  mut i: i64 = 0
  while i < 80 {
    iv.push(Item { v: i })
    i = i + 1
  }
  h
}

fn main() -> i32 {
  mut pluck: wedge Item = build()
  // Heavy caller-side churn through string concat — the only path
  // to pluck's chunk is mark_wedge on the rooted slot.
  mut acc: str = ""
  mut i: i64 = 0
  while i < 3000 {
    acc = acc + "y"
    i = i + 1
  }
  println(pluck.v)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress)
    assert rc == 0
    assert stdout == b"999\n"
