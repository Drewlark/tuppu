# bench

Microbenches comparing `dvec<T>` (contiguous T inline) and `ivec<T>`
(chunk-chain + slot-pointer buf, pointer-stable) on the workloads that
matter for picking between them in real code. Each bench is a
self-contained subdirectory with `dvec.tpu` and `ivec.tpu` doing the
same work under the same name; the top-level `run.sh` builds both,
times them 3× each, and reports best wall time plus the ivec-vs-dvec
ratio.

The two existing benches kept their own conventions:

- `ivec_vs_dvec/` — push-throughput sweep across element sizes
  (1, 4, 16, 32, 64, 128 × i64). Has its own `run.sh` because it
  generates `dvec_T*.tpu` / `ivec_T*.tpu` per size rather than the
  `dvec.tpu` / `ivec.tpu` pattern. Useful for finding the size
  threshold where ivec's smaller grow cost overtakes dvec's
  one-load index.
- `ivec_dvec_workload/` — build-and-read of a 256 B `BigStruct`,
  1000× build with 1000 elements each. Closer to a real
  build-then-read shape; the original push-only T-sweep overstates
  ivec's wins at large T.

The four benches added on top:

- `read_iter_l1/` / `read_iter_l3/` / `read_iter_dram/` —
  read-dominated, split per cache regime so the runner can time
  each separately. L1 (4k × 8B = 32 KB), L3 (256k × 8B = 2 MB),
  DRAM (4M × 8B = 32 MB). Splitting was deliberate after a botched
  earlier version that combined all three phases in one binary —
  the harness can only time the whole binary, not phases inside,
  so per-regime numbers were guesswork until split.
- `random_access/` — defeat the prefetcher with a linear-congruential
  index walk over 1 M i64s. Tests whether dvec's single-load index
  matters when memory-access patterns aren't streamable.
- `grow_heavy/` — repeated build/drop of a 64 B struct vec, no reads.
  Measures realloc cost differential: dvec memcpys `cap × sizeof(T)`,
  ivec memcpys `cap × 8`.
- `nested_composite/` — `vec<Entry>` where `Entry` holds a heap
  `str` field plus i64s. Exercises the trace path through
  per-element heap-bearing structs — closer to JSON-shaped data.

## Running

```sh
# All benches:
bash bench/run.sh

# Just one or two:
bash bench/run.sh read_iter random_access

# The element-size sweep (separate harness):
bash bench/ivec_vs_dvec/run.sh
```

The runner discards stdout (the `.tpu` files print accumulators only
to defeat dead-code elimination) and reports best-of-3 wall time.
Variance from GC, OS scheduling, and thermal throttling is real;
treat anything within ~10% as "wash."

## Snapshot results (Apple Silicon, post-ivec-redesign)

Best-of-3 wall times. Treat ≤10% deltas as wash; the >50% deltas
are the ones worth puzzling over.

| bench               | dvec ms | ivec ms | ivec/dvec |
|---------------------|---------|---------|-----------|
| read_iter_l1        |     299 |     175 |       59% |
| read_iter_l3        |     353 |     381 |      108% |
| read_iter_dram      |     348 |     187 |       54% |
| random_access       |      89 |     380 |      427% |
| grow_heavy          |      97 |     108 |      111% |
| nested_composite    |      27 |      29 |      107% |
| ivec_dvec_workload  |      31 |      35 |      113% |

Reads:

- **random_access is the headline for dvec.** ivec's two-load index
  path turns one cache miss into two on adversarial access patterns;
  with the prefetcher disabled by the LCG walk, both misses are
  ~200 cycles on DRAM and the gap balloons. dvec is 4.3× faster.
  This is the "indexing-not-scanning" workload (JSON traversal,
  agent state lookup, hash-table probing) — exactly where most
  modern programs live.
- **Streaming reads favor ivec at L1 and DRAM** by 41–46%, but tie
  at L3. The chunk-walk codegen probably pipelines better than
  dvec's `idx * sizeof(T)` mul, or autovectorizes more cleanly on
  the chunked layout — would take an IR/asm dive to confirm. Either
  way, both designs are bandwidth-bound at DRAM scale and the gap
  there is doing real work.
- **Composite / workload / grow benches all favor dvec by ~7–13%.**
  Individually within noise, but the directional consistency across
  three different workloads suggests a small real edge under
  typical build-and-read shapes.

Net read for the dvec / ivec / autovec design question: **both
belong in the language.** dvec wins random access by 4×, wins small
read/write/grow workloads by ~10%; ivec wins streaming reads by
~50% and gives pointer stability for free. Neither is universally
faster, and crucially, *the right choice depends on access pattern,
which the compiler can't see from the type alone.* Autovec by
`sizeof(T)` was already wrong; autovec by anything is wrong. Users
have to pick — and the property they're picking on is best
exposed in the type name.

## Cache regime caveat

These benches are timed on whatever machine they run on. Cache
hierarchies vary widely — a number that looks like an L3 hit on
an Apple M-series core might be a DRAM miss on a smaller mobile
chip. The per-machine numbers are most useful as relative
comparisons (ivec vs dvec on this machine) rather than absolute
performance claims. If you want machine-comparable numbers, lock
CPU frequency, disable turbo, and pin to a single core; that's
beyond this harness's scope.
