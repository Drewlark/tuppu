# Pre-GC-migration baselines (issue #8 — verification gate 1)

Captured on the `claude/gc-framework-migration` branch off `main` at
commit `fd60967`. Purpose: pin numbers we can compare against once
the LLVM gc-framework migration lands, so "did opt actually start
working" stops being a vibes question.

Toolchain:

- llvmlite 0.47.0 / LLVM 20 (driver-side IR emit + lowering)
- clang 18.1.3 (link)
- Linux 6.18.5 / x86_64
- `./tuppu build <src> -o <out>` (default `-O0`, stdlib bundled)

Each bench is timed best-of-5 wall-clock under `subprocess.run` from
warm-cache. Numbers are illustrative for this hardware — what matters
is the **ratio** vs. the post-migration capture.

## Wall-clock (-O0)

| Bench | ms |
|---|---|
| `bench/gc_migration/fib35.tpu` (recursion, 9.2M calls) | 53.9 |
| `bench/gc_migration/counter_bump.tpu` (1M edubba `Counter.bump()`) | 2.7 |
| `bench/gc_migration/str_concat.tpu` (100K `s = s + "x"`) | 248.0 |
| `bench/gc_migration/struct_copy.tpu` (10K struct-by-value pass) | 2.0 |
| `examples/lua_interp.tpu` (full run) | 3.5 |

Interpretation:

- `fib35` is the call-overhead reference — pure recursion, no GC
  values. Targets ≥30% speedup post-migration once mem2reg promotes
  the i64 stack slots and the recursion can use registers.
- `counter_bump` is the edubba inline target. The Counter is a single
  i64, so today no gcroot fires — the 2.7ms is a pure-call-overhead
  baseline. Useful as the unobstructed "best case" floor that the
  inlined-with-GC variant should approach.
- `str_concat` is the heap-pressure reference. 100K iters * one
  per-iter alloc + per-iter root push/pop. Targets ≥30%
  speedup; the dominant cost is the alloc itself, not the rooting,
  so don't expect a multiplier.
- `struct_copy` measures struct-pass-by-value with a str field.
  Targets mem2reg promotion of the param slot; 1.5-2× expected.
- `lua_interp` is the integrated reference — small fixture, but
  it touches every Tuppu codegen path. Wedge-into-tablets, seals,
  edubba dispatch (none today, but planned).

## Allocas: pre-migration mem2reg behavior

The whole point of the migration is to teach LLVM about our GC roots
through its first-class framework so optimizations preserve them.
Today, `__tuppu_gc_push_root` is opaque to opt — every alloca that
flows into one stays unpromoted. Empirically:

```
$ opt -passes=mem2reg -S build/preflight/lua_interp.ll \
    -o /tmp/m2r.ll
$ grep -c "alloca " build/preflight/lua_interp.ll      # 1229
$ grep -c "alloca " /tmp/m2r.ll                        # 1194
```

35 allocas promoted (2.8%) out of 1229. Reading the diff: every
promoted alloca is a scalar local that flows into no rooted slot;
every survivor is either a struct/seal slot held across a GC-rootable
expression or behind a `__tuppu_gc_push_root` barrier.

The migration's targeted promotion floor: the slots whose pointer
fields are unconditionally live for the fn duration (anything that
lowered through `_register_gc_root` in `codegen/stmt.py`) become
gcroot-tracked allocas. mem2reg leaves them alone (correctly — gcroot
forces the slot addressable), but their **adjacent scalar loads /
stores** unblock; SROA splits non-pointer fields out; LICM hoists
loop-invariant pointer reads.

We expect the post-migration mem2reg-only count to climb to ~70-80%
promoted. The bigger win is at `default<O2>` where SROA + GVN
+ LICM + inlining all chain.

## `-O2` SIGSEGV repro: not reproduced today

The issue describes "anything above `-O0` segfaults the lua
interpreter." On this branch / hardware, that's not what we see:

```
$ opt -passes='default<O2>' -S build/preflight/lua_interp.ll \
      -o /tmp/o.ll
$ llc -filetype=obj -relocation-model=pic /tmp/o.ll -o /tmp/o.o
$ clang /tmp/o.o build/preflight/tuppu_gc.o -o /tmp/o.bin
$ /tmp/o.bin > /dev/null; echo $?
0
$ TUPPU_GC_STRESS=1 /tmp/o.bin > /dev/null; echo $?
0
```

Same outcome at `-O1` and `-O3`. Possible explanations:

1. The original repro was platform- or input-specific (macOS, larger
   Lua source) and our ad-hoc fixture doesn't pressure it enough.
2. Recent freeze-rule + wedge-rooting fixes already eliminated the
   structural class of bugs that prior `-O2` runs surfaced.
3. The crash needs an LLVM-version-specific opt sequence we're not
   triggering with `default<O2>`.

We don't claim "no `-O2` bugs exist" — opt-time behavior is
exhaustive only in the limit. What we DO claim: today's `-O0`
suite passes; post-migration we'll require the **full test suite**
to pass under `-O2`, not just lua_interp. That's a stronger gate
than the original issue specified, and we got it for free by
having the suite already.

## Captured artifacts

- `bench/gc_migration/{fib35,counter_bump,str_concat,struct_copy}.tpu`
  — minimal benches, runnable via `./tuppu build && /usr/bin/time`.
- `bench/gc_migration/run_preflight.py` — wraps the timing dance so
  the post-migration capture uses the same harness (forthcoming
  `BENCH_POST_GC_MIGRATION.md` reads numbers from this).

## Verification gate status

Gate 1 (this doc): captured. Gates 2-7 are post-implementation.
