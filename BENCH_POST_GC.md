# Post-GC metrics (2026-04-24)

Captured on `main` at commit `c5cca87` (lazy tablets helper emission +
GC torture tests), directly comparable to `BENCH_BASELINE.md` from
pre-GC `c8c4b3e`. Same laptop, same build flags (`./tuppu build`,
default optimizer, clang link), same single-run `/usr/bin/time -l`
methodology.

## Binary sizes (bytes, stripped)

| Program | Pre-GC | Post-GC | Δ |
|---|---|---|---|
| `examples/hello.tpu` | 34,432 | 51,344 | +16,912 |
| `examples/fib.tpu` | 34,368 | 51,280 | +16,912 |
| `examples/higher_order.tpu` | 34,600 | 51,512 | +16,912 |
| `examples/linked_list.tpu` | 34,648 | 51,544 | +16,896 |
| `examples/reciprocal_table.tpu` | 34,504 | 51,424 | +16,920 |
| `examples/lua_interp.tpu` | 69,456 | 152,336 | +82,880 |

Hello-program overhead: ~+17 KB for the linked-in GC runtime + shadow
stack + the minimum type-descriptor set. Baseline predicted
"+50–200 KB, ballooning past ~300 KB means something went wrong" —
we're under the lower bound for trivial programs.

Lua interp grew +83 KB. The delta is almost entirely type descriptors:
every seal variant, every `tablets[N]T` monomorphization, and every
struct that transitively contains a seal gets its own descriptor +
trace_fn.

## Synthetic benchmarks

| Benchmark | Pre-GC Instr | Post-GC Instr | Δ | Pre-GC Cycles | Post-GC Cycles | Δ | Pre-GC RSS | Post-GC RSS |
|---|---|---|---|---|---|---|---|---|
| `fib(35)` | 337,090,991 | 338,136,071 | ~0% | 173,727,019 | 173,608,810 | ~0% | 1,049,408 | 1,082,176 |
| `str += "x"` × 50k | 280,623,734 | 417,774,897 | +49% | 63,057,838 | 130,340,962 | +107% | 1,491,776 | 1,639,232 |
| `tablets.push` × 100k × 50 | 270,888,211 | 201,340,485 | −26% | 58,149,674 | 50,940,909 | −12% | 1,884,928 | 3,162,944 |
| struct copy × 20k | 30,432,453 | 24,021,741 | −20% | 7,922,625 | 10,937,441 | +38% | 1,016,576 | 1,082,176 |
| `lua_interp.tpu` (12 programs) | 24,887,348 | 29,334,551 | +18% | 7,242,513 | 14,369,690 | +98% | 1,098,496 | 1,278,784 |

## Readout vs baseline expectations

- **Fib — unchanged.** Baseline: "Pure stack, no heap. Should be
  UNCHANGED." Instructions 0.3% delta, cycles 0.07%. The GC doesn't
  disturb non-heap-bearing code — the shadow-stack push/pop pairs
  around function entry have ~zero measurable cost when no roots fire.

- **Str concat — 1.5× instructions, 2× cycles.** Baseline: "Expect
  1.5-3× slowdown on this specific shape — it's an adversarial
  workload for any allocator." We landed at the low end of that
  range. Each concat still mallocs; the GC adds shadow-stack spills
  on the loop carry and periodic mark phases. The quadratic shape
  itself is the dominant cost, not the GC.

- **Tablets — faster.** −26% instructions, −12% cycles. The old
  per-chunk `malloc + release-fn registered in cleanup frame` path
  had more bookkeeping than the new `gc_alloc + trace_fn` path.
  Peak RSS is up +68% (3.16 MB vs 1.88 MB) because the arena
  doesn't give memory back mid-run; this is a trade we explicitly
  chose.

- **Struct copy — mixed.** Fewer instructions (−20%) but more
  cycles (+38%). The SSA-rooting chokepoint adds cache-pressure
  on the copy path; instruction-level savings don't always
  translate to cycle savings. Absolute numbers are tiny (24M
  instr total) so noise is louder here.

- **Lua interp — right at the 2× red-flag line.** Cycles 7.2M →
  14.4M (+98%). Baseline: "Any regression past 2× is a red flag."
  We're at 1.98×. Lua does the most allocation-per-instruction of
  anything in the suite: every token, every AST node, every
  environment binding hits the heap. This is the benchmark we
  should measure first when root-elision optimizations land
  (tasks #136, #137).

## Peak RSS commentary

Baseline predicted "Hello-world RSS... if it climbs past ~4 MB for
a trivial program, the initial heap is oversized." Hello is at
1.08 MB unchanged from baseline (the GC arena allocates lazily),
safely under.

Tablets bench shows the real arena-growth shape: 1.88 → 3.16 MB.
The arena doubles when it runs out of space; after 50 × 100k pushes,
it settled at ~3 MB. A future region-allocator (task #137) for
local tablets that don't escape would reclaim most of this.

## Tuppu compiler (self) metrics

Full `pytest` run: **688 passing** (667 pre-GC baseline + 20 GC
torture tests + 1 lua_interp example test), ~72 s wall — no
regression in test suite runtime.

Compiler code counts (lines, including comments):

| File | Pre-GC | Post-GC | Δ |
|---|---|---|---|
| `src/tuppu/typecheck.py` | ~3,040 | 3,171 | +131 |
| `src/tuppu/codegen/__init__.py` | ~4,062 | 4,830 | +768 |
| `src/tuppu/codegen/_common.py` | — | 120 | +120 |
| `src/tuppu/codegen/rat.py` | — | 118 | +118 |
| `src/tuppu/codegen/sex.py` | — | 994 | +994 |
| `src/tuppu/codegen/strs.py` | — | 610 | +610 |
| `src/tuppu/codegen/tablets.py` | — | 672 | +672 |
| `src/tuppu/effects.py` | ~280 | 315 | +35 |
| `runtime/tuppu_gc.c` | — | 309 | +309 |

The codegen package split (rat/sex/strs/tablets extracted into
separate files during the GC work) accounts for most of the
apparent growth — 2,514 of the 3,282 new codegen lines are
extracted-from-`__init__.py` code, not new code. Net-new codegen
is closer to ~768 lines in `__init__.py` for GC plumbing
(descriptors, shadow stack, chunk trace_fns, universal chokepoint).

Baseline target was "typecheck shrinks by ~800-1200 lines (freeze/
escape machinery removed)". That deletion hasn't landed yet —
task #133 is still pending. Until it does, the "net negative code"
promise is unpaid: we've added ~1,100 lines across typecheck +
codegen + runtime. Freeze/escape removal should flip the sign.

## What to compare against in future

- **Root-elision optimization (task #136):** should cut lua_interp
  cycles by 20-40%, bringing it back under the 2× threshold.
- **Region-alloc for local tablets (task #137):** should drop the
  tablets-bench peak RSS back near the baseline's 1.88 MB while
  keeping instruction counts where they are or improving them.
- **Freeze/escape machinery deletion (task #133):** should remove
  800-1200 lines from typecheck.py and 300-600 from codegen.
- **Binary size after freeze removal:** the ~17 KB overhead
  should shrink slightly — some descriptor emission can be
  elided for types proven not to escape.
