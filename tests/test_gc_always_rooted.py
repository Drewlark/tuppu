"""Always-rooted hazard tests for the LLVM gc-framework migration
(issue #8).

In `TUPPU_GC_FRAMEWORK=llvm` mode every gcroot-tracked slot is rooted
for the function's full lifetime — that's how `@llvm.gcroot`'s
semantics work. The runtime visits every entry on `llvm_gc_root_chain`
at every collection, regardless of whether the user has stored to the
slot yet. This file pins the cases where that "always rooted" property
historically caused trouble:

- alloca-undef contents observed before the user's first store
- a slot read after a logical "pop" point (we sentinel-clear there)
- loop-carried bindings where the per-iter clear must fire
- nested seals whose payload zero-clear hazard was flagged in the
  SROA experiment
- early-return paths past live cleanup-bearing bindings
- emitted-helper internal allocas

Each test runs under both `normal` and `stress` cadences. Stress mode
forces a collect on every allocation, so any not-yet-cleared slot
would dereference garbage on the very first iteration.

These tests target the `llvm` mode end-to-end by also forcing the
codegen mode via env var. In `shadow` mode they pass trivially (the
hazard simply doesn't exist), so we run them under both — matching
the "both modes pass full suite" verification gate in issue #8."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_files_to_binary, stdlib_files


@pytest.fixture(params=[False, True], ids=["normal", "stress"])
def stress(request):
    return request.param


@pytest.fixture(
    params=["shadow", "llvm"], ids=["framework=shadow", "framework=llvm"],
)
def gc_framework(request):
    """Run each test under both root-tracking schemes. The
    `compile_files_to_binary` import below is module-level, so we
    overwrite the env var at process-start time and reload — the
    Tuppu-driver subprocess (compile + run) sees the right value."""
    return request.param


def run(
    src: str, tmp_path: Path, stress: bool, gc_framework: str,
) -> tuple[int, bytes]:
    user = tmp_path / "main.tpu"
    user.write_text(src)
    # Codegen re-reads TUPPU_GC_FRAMEWORK on every instantiation
    # (see codegen/__init__.py:_gc_mode); we don't need module reloads
    # — just stash the desired value in env and let the next
    # `compile_files_to_binary` call pick it up.
    saved = os.environ.get("TUPPU_GC_FRAMEWORK")
    os.environ["TUPPU_GC_FRAMEWORK"] = gc_framework
    try:
        binary = compile_files_to_binary(
            stdlib_files() + [user], tmp_path, name="prog",
        )
    finally:
        if saved is None:
            os.environ.pop("TUPPU_GC_FRAMEWORK", None)
        else:
            os.environ["TUPPU_GC_FRAMEWORK"] = saved
    env = dict(os.environ)
    if stress:
        env["TUPPU_GC_STRESS"] = "1"
    env["TUPPU_GC_FRAMEWORK"] = gc_framework
    r = subprocess.run([str(binary)], capture_output=True, env=env)
    return r.returncode, r.stdout


from tuppu.driver import compile_files_to_binary


# --- baseline: zero-init seal slot is trace-safe -----------------------


def test_zero_init_seal_traces_safely(tmp_path, stress, gc_framework):
    # Allocate a seal slot but trigger GC before storing into it. The
    # init-clear at fn entry should leave it sentinel-tagged so the
    # trace fn short-circuits. Without the clear, the trace fn would
    # dispatch on undef and crash.
    src = """
seal Status { Ok(str), Err(str) }

fn alloc_pressure() -> str {
  // Each iter allocates a fresh str; under stress mode every alloc
  // collects, so the gc walks the rooted-but-uninitialised seal slot
  // 100 times before its first store happens at the call site.
  mut s: str = ""
  mut i: i64 = 0
  while i < 100 {
    s = s + "x"
    i = i + 1
  }
  s
}

fn main() -> i32 {
  // The seal binding's slot is rooted from fn entry. We immediately
  // call alloc_pressure, which in stress mode triggers GC at every
  // alloc — long before the seal is initialised at the binding's
  // assignment below.
  step junk = alloc_pressure()
  step result = Ok(junk)
  match result {
    Ok(s) => println(s.len),
    Err(_) => println(-1)
  }
  0
}
"""
    rc, stdout = run(src, tmp_path, stress, gc_framework)
    assert rc == 0
    assert stdout == b"100\n"


# --- post-pop: cleared slot still traces safely ------------------------


def test_seal_after_clear_traces_safely(tmp_path, stress, gc_framework):
    # Bind a seal in an inner scope, exit the scope (triggers
    # sentinel-clear), then stress GC. The cleared slot should still
    # trace safely from the outer fn's perspective.
    src = """
seal Tag { A(str), B(str) }

fn churn() -> i64 {
  mut s: str = ""
  mut i: i64 = 0
  while i < 200 {
    s = s + "y"
    i = i + 1
  }
  s.len
}

fn main() -> i32 {
  // Inner block: seal lives, then dies (slot cleared).
  if true {
    step inner = A("hello")
    println(inner.len)  // borrow read of the str payload via .len
  }
  // Outer fn keeps running; the cleared slot is still rooted by gcroot
  // and walked at mark time. Without the sentinel clear, the slot
  // would still hold a stale Tag whose payload pointer was freed.
  println(churn())
  0
}

fn .len(t: Tag) -> i64 {
  match t {
    A(s) => s.len,
    B(s) => s.len
  }
}
"""
    # Skip if the gloss syntax I sketched isn't supported — fall back
    # to plain match.
    src_simple = """
seal Tag { A(str), B(str) }

fn churn() -> i64 {
  mut s: str = ""
  mut i: i64 = 0
  while i < 200 {
    s = s + "y"
    i = i + 1
  }
  s.len
}

fn tag_len(t: Tag) -> i64 {
  match t {
    A(s) => s.len,
    B(s) => s.len
  }
}

fn main() -> i32 {
  if true {
    step inner = A("hello")
    println(tag_len(inner))
  }
  println(churn())
  0
}
"""
    rc, stdout = run(src_simple, tmp_path, stress, gc_framework)
    assert rc == 0
    assert stdout == b"5\n200\n"


# --- loop-carried seal binding -----------------------------------------


def test_loop_carried_seal_binding(tmp_path, stress, gc_framework):
    # Each iteration binds a fresh seal, exits the iter (sentinel
    # clear), and re-enters. 100 iters under stress mode mean every
    # iter's slot starts as sentinel and gets a real store during
    # the body.
    src = """
seal Result { Hit(i64), Miss(str) }

fn classify(n: i64) -> Result {
  if n % 7 == 0 { Hit(n) } else { Miss("nope") }
}

fn main() -> i32 {
  mut hits: i64 = 0
  mut i: i64 = 0
  while i < 100 {
    step r = classify(i)
    match r {
      Hit(_) => { hits = hits + 1 },
      Miss(_) => { }
    }
    i = i + 1
  }
  println(hits)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress, gc_framework)
    assert rc == 0
    # 0, 7, 14, ..., 98 → 15 multiples of 7 in [0, 100).
    assert stdout == b"15\n"


# --- nested seal payload (the SROA-experiment hazard) ------------------


def test_nested_seal_str_payload(tmp_path, stress, gc_framework):
    # `Outer { Inner(seal { Payload(str), Empty }), Done }` — the
    # nested seal whose zero-clear was flagged as fragile in the
    # always-rooted-hazard discussion. Under sentinel clearing the
    # inner seal traces as no-op when its outer is sentinel-tagged.
    src = """
seal Inner { Payload(str), Empty }
seal Outer { WithInner(Inner), Done }

fn build(n: i64) -> Outer {
  if n == 0 { Done } else { WithInner(Payload("x")) }
}

fn main() -> i32 {
  mut total: i64 = 0
  mut i: i64 = 0
  while i < 50 {
    step out = build(i)
    match out {
      WithInner(inner) => {
        match inner {
          Payload(s) => { total = total + s.len },
          Empty => { }
        }
      },
      Done => { }
    }
    i = i + 1
  }
  println(total)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress, gc_framework)
    assert rc == 0
    # i=0 → Done, no contribution. i=1..49 → Payload("x") (len=1).
    # 49 iterations contribute 1 each.
    assert stdout == b"49\n"


# --- early-return clears slot ------------------------------------------


def test_early_return_clears_seal_slot(tmp_path, stress, gc_framework):
    # `yield` (Tuppu's early return) past a live seal binding. The
    # yield path emits a clear for every active gcroot before the
    # ret. Without that, a caller GC after our return would walk a
    # slot whose contents reference our (now-freed) frame.
    src = """
seal Found { Yes(str), No }

fn lookup(n: i64) -> Found {
  step r = if n > 0 { Yes("hit") } else { No }
  // Force an early return so the yield path runs the slot-clear
  // sequence rather than the natural fn-end one.
  if n < 0 { yield No }
  r
}

fn main() -> i32 {
  mut hits: i64 = 0
  mut i: i64 = -10
  while i < 10 {
    step r = lookup(i)
    match r {
      Yes(_) => { hits = hits + 1 },
      No => { }
    }
    i = i + 1
  }
  println(hits)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress, gc_framework)
    assert rc == 0
    # i in [-10, 10): negative -> yield No (10 entries). 0 -> No. 1..9
    # -> Yes (9 entries). hits = 9.
    assert stdout == b"9\n"


# --- helper-fn with local seal -----------------------------------------


def test_helper_fn_with_local_seal(tmp_path, stress, gc_framework):
    # Codegen-emitted helpers (sex_add, struct release fns, etc.)
    # have their own internal allocas. In `llvm` mode each such alloca
    # gets its own gcroot if cleanup-bearing. The init-clear must fire
    # at helper entry too, not only at user-fn entry.
    src = """
tablet Wrap { s: str, n: i64 }

fn make(n: i64) -> Wrap {
  // Recursive call, so the helper-style emission stresses param
  // rooting + return-value transfer at the same time.
  if n == 0 {
    Wrap { s: "base", n: 0 }
  } else {
    step prev = make(n - 1)
    Wrap { s: copy prev.s + "+", n: prev.n + 1 }
  }
}

fn main() -> i32 {
  step r = make(10)
  println(r.n)
  println(r.s.len)
  0
}
"""
    rc, stdout = run(src, tmp_path, stress, gc_framework)
    assert rc == 0
    # n bottoms out at 0 with len("base")=4; each recursion appends "+"
    # so depth 10 means len = 4 + 10 = 14, n = 10.
    assert stdout == b"10\n14\n"


# --- mixed wedge + seal in same fn -------------------------------------


def test_mixed_wedge_and_seal_slots(tmp_path, stress, gc_framework):
    # A fn with both a wedge slot and a seal slot active simultaneously.
    # Their clear-paths use different sentinels (NULL ptr vs 0xFF tag);
    # any mix-up between the two would surface here.
    src = """
seal Step { Take(i64), Stop }
tablet Box { v: i64 }

fn drive(t: tablets[8]Box) -> i64 {
  mut sum: i64 = 0
  for b in t {
    step s = if b.v > 0 { Take(b.v) } else { Stop }
    match s {
      Take(n) => { sum = sum + n },
      Stop => { }
    }
  }
  sum
}

fn main() -> i32 {
  mut bs: tablets[8]Box
  bs.push(Box { v: 1 })
  bs.push(Box { v: 0 })
  bs.push(Box { v: 2 })
  bs.push(Box { v: 3 })
  println(drive(bs))
  0
}
"""
    rc, stdout = run(src, tmp_path, stress, gc_framework)
    assert rc == 0
    assert stdout == b"6\n"
