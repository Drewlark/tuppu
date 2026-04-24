# Pre-GC baseline metrics (2026-04-24)

Captured on `main` at commit `c8c4b3e` — the freeze-rule fix for
the writeback UAF. Purpose: establish a honest reference for the
GC migration branch so we can compare binary size, peak memory, and
runtime cost before / after, rather than argue in abstractions.

Build: `./tuppu build <src> -o <out>` (stdlib bundled, default
optimizer, clang link).

Hardware: the laptop this was recorded on; apples-to-apples with
whatever you measure next.

## Binary sizes (bytes)

| Program | Unstripped | Stripped |
|---|---|---|
| `examples/hello.tpu` | 35,912 | 34,432 |
| `examples/fib.tpu` | 35,824 | 34,368 |
| `examples/higher_order.tpu` | 36,368 | 34,600 |
| `examples/linked_list.tpu` | 36,584 | 34,648 |
| `examples/reciprocal_table.tpu` | 36,144 | 34,504 |
| `lua_interp.tpu` | 76,472 | 69,456 |

Baseline-program overhead (hello, stripped): ~34 KB. Nearly all of
that is libc + clang startup. Tuppu emits a tiny main + called
stdlib fns.

Lua interp jumps to ~70 KB stripped — ~36 KB of Tuppu-emitted code
for the lexer + parser + evaluator + closure support.

## Synthetic benchmarks

Single-run, instruments from `/usr/bin/time -l`.

| Benchmark | Instructions | Cycles | Peak RSS |
|---|---|---|---|
| `fib(35)` (recursion) | 337,090,991 | 173,727,019 | 1,049,408 |
| `str += "x"` × 50,000 (quadratic concat) | 280,623,734 | 63,057,838 | 1,491,776 |
| `tablets.push` × 100,000 × 50 rounds | 270,888,211 | 58,149,674 | 1,884,928 |
| struct copy × 20,000 | 30,432,453 | 7,922,625 | 1,016,576 |
| `lua_interp.tpu` (12 sample programs) | 24,887,348 | 7,242,513 | 1,098,496 |

5-run wall times (shell loop, average-of-five real time):

| Benchmark | real (s) |
|---|---|
| `fib_bench` | 0.23 |
| `str_bench` | 0.10 |
| `tablets_bench` | 0.09 |
| `struct_bench` | 0.22 |
| `lua_interp` | 0.16 |

## What to compare against post-GC

- **Binary size.** Expect +50-200 KB for the GC runtime + type
  descriptors. If it balloons past ~300 KB for `hello`, something
  went wrong.
- **Hello-world RSS.** Currently ~1 MB (mostly libc's normal
  slop). Post-GC will have an initial heap arena — 64 KB or 256 KB
  initial size is typical. If RSS at program start climbs past
  ~4 MB for a trivial program, the initial heap is oversized.
- **Fib.** Pure stack, no heap. Should be UNCHANGED. This validates
  that the GC isn't disturbing non-heap-bearing code paths.
- **Str bench.** Quadratic concat; each concat is a malloc + free
  today. Under GC, the frees become marker work. Expect 1.5-3x
  slowdown on this specific shape — it's an adversarial workload
  for any allocator.
- **Tablets bench.** Same allocation pressure; same expected range.
- **Struct bench.** All stack if structs don't escape; if they do,
  heap pressure. Measure whether effect analysis elides.
- **Lua interp.** The workload that matters for the "compiler gets
  simpler" narrative. Should stay under ~500 ms on the 12-program
  suite. Any regression past 2x is a red flag.

## Tuppu compiler (self) metrics

Full `pytest` run: 667 passing, ~72 s. If this crosses 90 s or
tests start failing on the GC branch, we've regressed.

Compiler code counts (lines, including comments):

| File | Lines |
|---|---|
| `src/tuppu/typecheck.py` | ~3040 |
| `src/tuppu/codegen/__init__.py` | ~4062 |
| `src/tuppu/effects.py` | ~280 |

Target for post-GC: typecheck shrinks by ~800-1200 lines (freeze/
escape machinery removed). Codegen shrinks by ~300-600 lines
(transfer-or-clone, cap-sentinel branches, cleanup frames, release
fn generation). New GC runtime lives outside the Python compiler —
either a C file bundled with emitted binaries, or generated at
codegen time.
