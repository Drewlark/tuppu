# Post-GC-migration measurements (issue #8 — verification gate 6)

Captured on `claude/gc-framework-migration` after Phases 1-3 land.
Same hardware / toolchain as `BENCH_PRE_GC_MIGRATION.md`:

- llvmlite 0.47.0 / LLVM 20 (driver-side IR emit + lowering)
- clang 18.1.3 (link); `opt` 18.1.3 for `-O2` runs
- Linux 6.18.5 / x86_64

`shadow` and `llvm` columns refer to the `TUPPU_GC_FRAMEWORK` env
flag. `O0` runs the IR through `llc` directly; `O2` runs `opt
-passes='default<O2>'` first. `shadow`+`O2` is intentionally absent
— the legacy push/pop scheme silently de-roots under opt and the
runtime aborts. That's exactly the regime issue #8 was filed for.

## Edubba dispatch

`bench/edubba_calls/run.py` reproduces these numbers; check it in to
re-measure post-merge:

| bench | mode | opt | ms (best of 5) |
|---|---|---|---|
| inline (Counter.bump 1M, no GC fields) | shadow | O0 | 2.76 |
| inline | llvm | O0 | 2.90 |
| inline | llvm | O2 | **1.34** |
| inline_gc (Tagged.bump 1M, str field) | shadow | O0 | 5.95 |
| inline_gc | llvm | O0 | 6.56 |
| inline_gc | llvm | O2 | **3.10** |
| outofline (Bag.set 100K, hash-style body) | shadow | O0 | 14.22 |
| outofline | llvm | O0 | 11.26 |
| outofline | llvm | O2 | **8.46** |

### Issue-8 target check

| bench | llvm@O2 / shadow@O0 | issue target | status |
|---|---|---|---|
| inline | 0.49 (~2× win) | ≤ 0.50 (≥2×) | **pass** |
| inline_gc | 0.52 (~2× win) | ≤ 0.20 (≥5×) | miss |
| outofline | 0.59 (~1.7× win) | ≤ 0.67 (≥1.5×) | **pass** |

`inline_gc` underdelivered. The expected mechanism — gcroot blocking
removal lets opt fold per-call push/pop and inline the body — fires;
the post-inline body is a single `count++` plus the entry-block
gcroot. But `opt -O2` keeps the slot addressable (gcroot forces
mem2reg to skip it), so the loop ends up doing a load-add-store on
the slot's `count` field rather than promoting the counter to a
register. We're seeing the call-cost win without the loop
register-promotion win. **Fixable as a follow-up** — the `count`
field is a non-pointer scalar, so a per-field SROA hint would let
mem2reg pick it up. That belongs in a v2 patch; documenting here so
it doesn't get lost.

`outofline` hit the target: mem2reg's job there is to promote the
`key` and `value` ABI-arg slots, which it does cleanly even when
the surrounding struct slot stays gcroot'd.

## Comparison vs. pre-flight (`BENCH_PRE_GC_MIGRATION.md`)

Pre-flight numbers were under `shadow` (pre-migration default). Same
binaries, no opt:

| bench | pre-flight ms | post-flight @ llvm O2 ms | speedup |
|---|---|---|---|
| `fib35.tpu` | 53.9 | 41.7 | 1.29× |
| `counter_bump.tpu` | 2.7 | 1.3 | 2.08× |
| `str_concat.tpu` | 248.0 | 226.4 | 1.10× |
| `struct_copy.tpu` | 2.0 | 1.7 | 1.18× |
| `lua_interp.tpu` | 3.5 | 2.8 | 1.25× |

The wins are smaller than the issue's stated targets (≥30% on fib,
lua_interp). Why:

- These benches hit shape-specific bottlenecks the migration didn't
  target. `str_concat` is dominated by the GC allocator path
  (`__tuppu_gc_alloc_bytes` + memcpy); the migration left that
  untouched.
- `fib(35)` is recursion-bound. The shadow-stack push/pop calls
  WOULD have been per-call costs, but `fib` has no GC fields, so
  no gcroot is emitted in either mode. The 1.29× win is from
  ordinary opt (mem2reg of i64 locals, register allocation).
- `lua_interp` is an integration fixture; many small wins compound
  to a modest overall improvement.

Conclusion: the migration delivers what it can deliver — opt now
runs, edubba dispatch wins ~2×, hash-style methods win ~1.7×. The
SROA-of-non-pointer-fields-in-rooted-structs follow-up is the next
lever to unlock further. Issue #8's headline targets are partly met
and partly require additional work; this doc says exactly which.

## Runtime ABI sanity

The runtime's `gc_init` constructor asserts the `StackEntry` and
`FrameMap` layouts match what the LLVM strategy emitter produces.
Currently active on every `llvm`-mode binary; trips the binary
before main() runs if a future LLVM upgrade reshapes the structures.

## Verification gate status

Gates 1-5 (pre-flight, three phases each at -O0, both modes) pass.
Gate 6 (bench targets) is partial — see table above. Gate 7 (runtime
version assertion active) ships as part of the runtime change.
