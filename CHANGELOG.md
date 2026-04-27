# Changelog

All notable changes to the Tuppu compiler are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
in pre-1.0 mode: the **MINOR** number bumps on every notable feature
or breaking change, **PATCH** on bug fixes.

The version of record lives in `pyproject.toml`; this file is the
narrative.

## [Unreleased]

### Added

- **`type Name = TypeExpr` aliases.** Transparent — every use site
  resolves the alias's target. Works with primitives, tablets, seals,
  generic struct types (`type Counts = tablets[64]i64`), and chains
  through other aliases. Cannot collide with existing tablet / seal /
  primitive names. No type-parameter list yet.
- **`tuppu check` subcommand.** Parses + typechecks without emitting
  IR or linking. Fast feedback loop for "did I write valid Tuppu?"
  Skips most of a build's wall time and surfaces the same errors.
- **Source context in lex / parse errors.** Errors now show the
  offending source line + a caret pointer beneath the column, in the
  gcc/clang-style. `tuppu: foo.tpu:4:12: <message>` followed by the
  line and a `^`. CompileError gained an optional `path` attribute
  the driver attaches; `format_error(e, source_text)` does the
  rendering. Typecheck and codegen errors don't yet carry a path,
  see LIMITATIONS.md.

### Removed

- **`_neuter_return_if_borrow` and its call site in `_gen_yield`.**
  This was a UAF guard pre-GC that deep-cloned the return value of
  every fn whose body tail (or `yield`) was a `Field` or `Index`
  expression. Under the GC, the returned value's type descriptor
  (e.g. str.ptr → byte buffer) keeps the underlying allocation
  reachable from the caller's binding, so the clone was wasted
  allocation. Four new GC torture tests pin the behavior under
  `TUPPU_GC_STRESS=1`: returning a Field of a local struct, an
  Index of a local tablets, a Field through a match arm, and a
  nested Field-of-Field — each followed by ~100 allocations in
  the caller before the value is read.

### Changed

- **`step _ = expr` is no longer the idiom for discarding a return.**
  Bare expression-statements have always worked (`local.push(x)` is
  a valid statement); the `step _ = ...` wrapper added nothing the
  bare form didn't, but had been cargo-culted across stdlib, examples,
  and tests. Swept ~57 occurrences. README + CONTRIBUTING note the
  preferred form.

## [0.4.0] — 2026-04-26

This is the first release after the version reset. Earlier history
is captured in git; the project tracked an inconsistent "v0.1 draft"
label for a long time and the move to a real GC-backed runtime is a
big enough turning point to be honest about it. From here forward,
every notable change appears in this file.

### Added

- **Mark-sweep garbage collector.** `runtime/tuppu_gc.c` is a small
  precise GC with shadow-stack rooting, type descriptors emitted by
  codegen, and a stress mode (`TUPPU_GC_STRESS=1`) that forces a
  collection on every allocation for testing. All `str`, `tablets`,
  and seal-payload allocations route through it; user code never
  calls `malloc` / `free` directly.
- **Generics.** `tablet Name<T>`, `fn name<T>(...)`, monomorphization
  on use, type-arg inference at call sites. `seal` is included.
- **Sum types via `seal`.** Tagged-union variants (nullary or carrying
  payload), pattern matching with binding patterns, and per-seal
  release / clone helpers driven by type descriptors.
- **Sexagesimal literals + `rat` arithmetic.** Babylonian-faithful
  digit sequences (`1;30`, `1;24 51 10`) lower to a `sex` type that
  auto-promotes to exact rational `rat = {num: i64, den: i64}`. No
  `f64` rounding in the path. The `dish` keyword is an alias for
  `sex` to match SPEC vocabulary.
- **Comptime tables.** `table foo: T = (lo..hi) -> generator()`
  evaluated at build time, baked into the binary as static data.
- **`colophon` typed FFI.** Declares C externs with Tuppu-side
  signatures; primitives + `*u8` + struct-by-pointer + variadic
  slice parameters supported. Callbacks restricted to primitive-
  only signatures.
- **`gloss` operator overloads.** User-declared operator methods
  (e.g. `gloss eq(a: str, b: str) -> bool`) wired into the parser's
  binary operator dispatch.
- **Generic stdlib containers.**
  - `stdlib/vec.tpu`: `Vec<T>` over `tablets[64]T` — push, get, set,
    swap, reverse, find, contains.
  - `stdlib/map.tpu`: `Map<T>` (string-keyed, insertion-ordered,
    linear-scan) — get, set, has, len.
  - `stdlib/list.tpu`: `Node<T>` + `list_push` / `list_len` /
    `list_find` / `list_contains`.
- **Lvalue chains.** `m.values[i] = v`, `outer.inner.values[i] = v`,
  `m.entries[i].count += 5`, etc. — any chain of `.field` and `[i]`
  accesses rooted at an `Ident` is a legal assignment target. Both
  the parser's `_check_lvalue` and codegen's `_lvalue_slot` walk
  the chain recursively.
- **Examples:** `lua_interp.tpu` (1500-line Lua subset interpreter,
  the regression magnet for GC + closure handling) and
  `scribe_ledger.tpu` (Vec + Map cooperating).
- **Test suites:** `tests/test_gc_torture.py` (32 cursed-composition
  tests × 2 GC modes), `tests/test_lvalue.py` (10 chain shapes ×
  2 GC modes including RHS-pushes-into-target stress).
- **Documentation:** `README.md` (quickstart, types reference,
  keyword index, stdlib + examples tour), `CONTRIBUTING.md` (the
  correctness-paramount stance), `BENCH_BASELINE.md` /
  `BENCH_POST_GC.md` (perf measurements before / after GC).

### Changed

- **`codegen/__init__.py` split into mixins.** Previously a 4,830-
  line monolith; now a 791-line scaffolding file plus seven concern-
  oriented mixins (`module`, `stmt`, `expr`, `types`, `seals`,
  `intrinsics`, `access`) on top of the existing `tablets`, `strs`,
  `sex`, `rat` mixins. No behavior change, but every file is now
  navigable.
- **Freeze / escape machinery deleted from `typecheck.py`.** Pre-GC,
  the type checker tracked aliasing and inserted implicit deep-clones
  to prevent UAF at "escape sites." Under GC, shadow-stack rooting
  makes those clones unnecessary and the rules false-positive. ~690
  lines removed; the kept escape rule is just the wedge-handle-
  out-of-local-arena check (wedges aren't GC-traced).
- **Universal SSA-rooting chokepoint in codegen.** A single
  post-dispatch hook in `_gen_expr` registers heap-bearing
  intermediate values as GC roots, replacing the prior consumer-side
  rooting helpers.
- **Tablets helper emission deferred until first call.** Previously
  the push / get / release functions were materialized at type-
  registration time, capturing chunk descriptors before recursive
  seal payloads were finalized. Now the type is reserved eagerly
  but the helper bodies are emitted on demand.

### Fixed

- **Mut-struct-param lookup keys on monomorphized fn name.** Generic
  `tablet Vec<T> { storage: tablets[N]T }` + `mut Vec<T>` parameter
  used to fail with `cannot coerce %"Vec__i64" to %"Vec__i64"*`
  because `_gen_call` queried `_fn_param_mut` with the AST-level
  name (`vec_push`) instead of the mangled mono name (`vec_push__i64`).
- **Chunk trace_fn for elements that transitively contain seals.**
  Tablets of structs whose fields included seals had a too-flat
  trace function that missed payload pointers; cross-collection
  pattern matches in the lua interpreter would print empty strings.
- **Lazy tablets helper emission** (see Changed) — closes a sub-
  variety of the same chunk-descriptor-too-early bug.
- **GC torture coverage.** Six previously-flaky scenarios (struct
  alignment, shadowed cleanup eviction, anonymous rvalue block-tail,
  yield mid-expression, match-binder implicit clone, tablets
  circular wedge assignment) now have explicit regression tests.

### Removed

- The pre-GC freeze / escape rules and their associated implicit-
  copy warnings. Programs that used to compile with a warning now
  compile silently; the deep-clones the warnings flagged are
  unnecessary under the GC.
- `Tok.RAT` (the comma-separated sexagesimal literal). Sexagesimal
  digit groups are space-separated only — `1;30 0 0`, not `1,30`.

### Known limitations

- `_neuter_return_if_borrow` in codegen still inserts a defensive
  deep-clone when a function returns a `Field` or `Index` expression.
  This was a UAF guard pre-GC and is wasted work post-GC; the
  followup is to delete it.
- Removal from `Map<T>` is not supported. Tablets are append-only
  at the runtime level, so a real `map_delete` needs tombstone slots.
- `Vec<T>` chunk size is hardcoded to 64. Numeric generics (so a user
  can write `Vec<T, 16>` vs `Vec<T, 1024>`) are pending.
- No module / import system. The bundled stdlib is auto-discovered
  from `stdlib/*.tpu`; user programs are single-file.
- `f32` / `f64` are reserved keywords with no implementation. Casts
  to / from float types raise a clear "not yet supported" error.

## Earlier history

Pre-0.4.0 commit history is in git (`git log`). The project went
through roughly: lexer + parser + basic codegen → structs + strings +
ownership rules → sex/rat + comptime tables → generics + seals + FFI
→ GC migration. None of those phases were tagged or version-marked
at the time. The 0.4.0 baseline above is the first version with
formal versioning hygiene.
