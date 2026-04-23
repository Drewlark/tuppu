# Tuppu — what's next

Pickup notes. The Babylonian sexagesimal redesign is the active
front; imports and dynamic strings are queued behind it.

## Where we are (as of 2026-04-22)

- **v0.1 feature-complete** per SPEC.md — lexer, Pratt parser, type
  checker, LLVM codegen via llvmlite.
- Private repo: https://github.com/Drewlark/tuppu (branch `main`).
- **411 tests passing.**
- CLI: `./tuppu run file.tpu` and `./tuppu build ... -o out`.
- Bundled stdlib auto-included; pass `--no-stdlib` to opt out.
- Compiler's in Python (`src/tuppu/`); stdlib's in Tuppu
  (`stdlib/*.tpu`).
- `scratch/` is gitignored (along with `fun.tpu`, `scratch.tpu`) for
  user experimentation without polluting history.

What works:
- Primitives: `i8..i64`, `u8..u64`, `bool`, `rat`, `sex`/`dish`.
- **User-defined structs / tablets** — nominal, by-value, forward refs
  resolved via topological sort. `tablet`/`tablet` are interchangeable
  keywords. Construction `Point { x: 3, y: 4 }`, field access `p.x`,
  structs as params/returns and in `mut` bindings. **Field mutation**
  `p.x = 5` and chained `l.a.x = 5` via GEP into the alloca; `p.x +=
  1` aug-assign also works. `step`-bound structs remain immutable.
- **Variable-length strings** — `str` is a built-in tablet
  `{ ptr: *u8, len: i64 }`, auto-injected. `fn f(s: str)`, `s.len`,
  `s[i]`, `print(s)`/`println(s)` via `write(2)` (embedded NULs
  survive). Stdlib `str_eq`/`str_starts_with`/`str_ends_with`/
  `str_is_empty`/`str_index_of`.
- Raw pointer type `*T` — type-only, no expression-level ops.
- **Recursive structs** via identified LLVM types. `wedge Node {
  next: wedge Node }` works, including mutually-recursive pairs.
  Direct (no-indirection) cycles are rejected with a clear fix hint.
- **Tablet handles** — `wedge T` is a handle into some
  `tablets[N]T`. Obtained from `tablets.push(x)`, compared with
  `==` / `!=`, auto-dereffed on `.field`. `lost` is the null handle.
- **Auto-release** on `mut tablets[N]T` at scope exit — codegen
  inserts the release call; explicit `release` still works for
  early release and is de-duplicated against the auto path. Covers
  fall-through and `yield` (early-return) paths.
- **Escape check** rejects returning a `wedge T` whose underlying
  tablets is declared locally — the common UAF pattern. Parameters
  and `lost` are safe to return.
- **Mut parameters** — `fn f(mut x: T)` allocas the param so methods
  work on it. For `tablets[N]T` specifically the param is passed by
  pointer so mutations persist to the caller's storage.
- **Char literals** `'a'`, `'\n'`, etc. — type `u8`.
- **Multi-arg `print`/`println`** — `println("x=", x, " y=", y)`.
- **Augmented assignment** `+= -= *= /= %=` (parser-desugared).
- **`elif` keyword** for chained conditionals — sugar over
  `else if`, both forms still parse.
- **Did-you-mean suggestions** on undefined name, unknown function,
  unknown tablet, and unknown tablet-field errors.
- **`colophon` reserved keyword** — no semantics yet, held for a
  future use (file-level metadata preamble or tablets debug-name).
- **`for name in iter { ... }`** — works over `str` (u8 bytes),
  `tablets[N]T`, and comptime tables. Loop variable is step-bound.
- **Mixed-width int comparisons** promote to the wider type (matches
  `if`-arm unification).
- **Sex/dish Phases 1–3b:**
  - Distinct 20-byte runtime `{ [16]u8 digits, u8 radix, u8 count,
    i8 sign, u8 pad }`.
  - Native Babylonian printing (`1;30`, `1;24 51 10`, `-1;30`).
  - `sex as rat` is a real reduce; `sex as i64` truncates via rat.
  - `int → sex` decomposes an i64 into base-60 digits (integer form).
    So `Point { x: 0, y: 0 }` with sex fields works.
  - `rat → sex` via `__tuppu_rat_to_sex` — regularity check (den =
    2^a·3^b·5^c), integer-digit decomposition + fractional-digit
    extraction. Runtime trap on non-regular denominators or digit-
    buffer overflow.
  - Native `sex + sex` and `sex - sex` via `__tuppu_sex_add` — radix
    alignment, 16-lane SIMD digit add, scalar carry, mixed-sign via
    magnitude compare + borrow-propagating subtract. No warning.
  - Native `sex * sex` and `sex / sex` via the rat path:
    `(a as rat) op (b as rat)` → `__tuppu_rat_to_sex`. Result stays
    dish-typed. Traps at runtime on non-regular results or div-zero.
  - Mixed `sex op int` (any of +, -, *, /) promotes the int to sex
    via `__tuppu_int_to_sex` — digit form preserved end-to-end, no
    warning.
  - Unary `-` is a free sign-byte flip.
- `step` (SSA) and `mut` (alloca) bindings; assignment.
- `if`/`else` as expression; `while`; `yield` early-return.
- User functions with recursion; multi-file compilation.
- `table name[lo..hi]: T = genfn` — comptime-evaluated static tables.
- `tablets[N]T` — chained-chunk growable storage, `release` to free.
- Intrinsics: `print`, `println`, `read_int`, `rat(n, d)`.
- Type errors with source `line:col`; codegen errors now also carry
  `line:col` via `Codegen._current_loc` tracking.
- Compile-time warning infrastructure.

What doesn't yet:
- Native sex `%` (still warn-lower to rat; rat % itself isn't
  implemented either, so this errors at codegen).
- Mixed sex × rat ops (still warn, stay as rat — deliberate, since
  the rat operand has abandoned digit form).
- Escape-analysis rat-fallback for rat-only sex values (**Phase 3c**
  — the big compiler-learning chunk).
- `--strict-dish` flag (see FUTURE_OPTIMIZATIONS.md for the sketch).
- Imports / namespacing.
- Dynamic string ops (concat, slice, case) — needs tablets-backed
  alloc plus a lifetime/ownership story.
- `read_line() -> str` — same blocker.
- Expression-level pointer ops (`*p`, `&x`, `p + 1`) — intentional.
- Recursive structs (would need identified types + heap indirection).
- `impress` reinterpret cast (documented in SPEC §14).
- f64 (lexed but no codegen; `sex as f64` errors cleanly).
- Closures / first-class functions.
- Self-hosting.

## Priority order (agreed)

1. ~~**Tablets**~~ — **done.** See `examples/points.tpu`,
   `tests/test_struct.py`.
2. ~~**Better strings**~~ — **done.** See `examples/greeting.tpu`,
   `tests/test_string.py`, `stdlib/str.tpu`.
3. ~~**Ergonomics bundle**~~ (char lits, multi-arg print, `+=`, for,
   mixed-width int compare) — **done.** `tests/test_ergonomics.py`.
4. ~~**Sex/dish Phase 1**~~ — **done.** Digit-form runtime +
   Babylonian printing + `sex as rat` as a real reduce helper.
5. ~~**Sex/dish Phase 2**~~ — **done.** Native `sex + sex` / `-`,
   int→sex, error location tracking. See `__tuppu_sex_add`.
6. ~~**Sex/dish Phase 3a**~~ — **done.** `__tuppu_rat_to_sex` with
   regularity check, `rat → sex` coercion, native `sex * sex` via the
   rat path. See `tests/test_sex.py::test_rat_to_sex_*` and
   `::test_sex_mul_*`.
7. ~~**Sex/dish Phase 3b**~~ — **done.** Native `sex / sex` via the
   same rat-path dispatch. Mixed `sex op int` (+/-/*//) also works
   now — int promotes to sex via `__tuppu_int_to_sex`. See
   `::test_sex_div_*` and `::test_sex_mixed_int_*`.
8. ~~**Language QoL pass**~~ — **done.** Struct field mutation,
   codegen.py split into a mixins package, elif, did-you-mean
   suggestions, recursive tablets + wedge handles + auto-release +
   escape check, tablet/wedge/seal rename.
9. ~~**Minimal generics**~~ — **done.** `tablet Node<T>`,
   `fn push<T>(...)`, HM-style inference at call sites and struct
   literals, lazy monomorphization. `stdlib/list.tpu` rewritten as
   `List<T>`. See `tests/test_struct.py::test_generic_*`.
10. **Sum types** — **Next.** Landing the `seal` keyword for Option/
    Result/ADT shapes, plus minimal pattern matching. Biggest single
    step toward functional-style code and toward self-hosting. See
    "Roadmap (agreed 2026-04-22)" below for the full menu.
11. **Sex/dish Phase 3c** — escape-analysis pass for rat-fallback
    specialization. Unblocks the future `--strict-dish` flag idea.
12. **Imports** — cleanup; pay when stdlib grows enough to need
    namespaces.

## Roadmap (agreed 2026-04-22)

Cross-cutting goal: incrementally fold pieces of the compiler into
Tuppu itself (self-hosting). Minimum viable self-host set: sum types
(AST shape), closures or fn-as-value (visitors), dynamic strings
(error messages), maps (symbol tables), file I/O.

Ranked by "unblocks most / lays groundwork for the self-host path":

| Feature                          | Cost    | Unlocks                                       | Notes                                                                        |
|----------------------------------|---------|-----------------------------------------------|------------------------------------------------------------------------------|
| **Sum types + pattern matching** | ~6-8 h  | Option, Result, ADTs, AST-shaped data         | `seal` is already reserved. Biggest single step toward self-hosting.         |
| **Fn-as-value (no capture)**     | ~2-3 h  | Higher-order functions, minimal visitors      | Cheap. Pointers-to-functions, no environment capture.                        |
| **Full closures (with capture)** | ~4-6 h  | Inline callbacks, state-carrying fns          | Needs a captured-env layout story.                                           |
| **Overloads (`__repr__`-style)** | ~2-3 h  | Generic `print`, user-extensible operators    | Type-dispatched name resolution. No trait system needed.                     |
| **Operator overloads**           | +1-2 h  | `a + b` for user types                        | Layers on the same dispatch mechanism as overloads above.                    |
| **Dynamic strings + slicing**    | ~4-6 h  | `s[i:j]`, concat, real manipulation           | Needs an arena-for-strings story (tablets-backed, auto-release).             |
| **Maps / hash tables**           | ~4-6 h  | Symbol tables, interning                      | Can be built in Tuppu on top of `tablets`; probably starts as `stdlib/map.tpu`. |
| **File I/O (read_line, etc.)**   | ~2-4 h  | Real programs, step toward self-hosting input | Libc wrappers via the same extern pattern as `write`/`fflush`.               |

Agreed order (approximate): sum types → fn-as-value → full closures
→ overloads → operator overloads → dynamic strings → maps → file I/O.

**Sum types design sketch** (to pick up after compact):

```
seal Option<T> {
  Some(T),
  None,
}

seal Shape {
  Circle(radius: rat),
  Square(side: rat),
}

fn area(s: Shape) -> rat {
  match s {
    Circle(r) => r * r * rat(314, 100),
    Square(s) => s * s,
  }
}
```

- Tagged-union runtime: `{ tag: i8, payload: <union of variants> }` —
  payload is a `max(sizeof(each variant))`-byte buffer. Identified
  LLVM type per-seal (same machinery as tablets, different shape).
- Variant constructors are callable-like: `Some(42)` produces an
  `Option<i64>`; `None` is a zero-arg constructor.
- Pattern matching exhaustive over variants. v0.1 scope: no guards,
  no nested patterns, bindings only at the variant level.
- Generic seals (`seal Option<T>`) use the same monomorphization
  machinery as generic tablets.

Skip for v0.1: nested patterns (`Some(Circle(r))`), guards
(`Some(x) if x > 0`), or-patterns (`Some(1) | Some(2)`). Defer to
v0.2 if we want real ML-family matching.

## 1. Future: `impress` reinterpret cast

Planned for v0.2 (documented in `SPEC.md` §14). Keeps `as` purely for
safe numeric conversions while giving same-layout tablet reinterpretation
its own vivid, mildly-scary spelling:

```
step v: Vector = impress p as Vector
```

Rules: source and target must be tablets (not primitives) with
byte-identical field layouts — same field *types* in declaration
order, names need not match. Codegen is a no-op bitcast. A third
mechanism (trait-driven `into`-style conversion) can layer on later
without colliding with either `as` or `impress`.

## 2. Imports — design sketch

### Grammar
```
use_decl = "use" path
path     = IDENT ("/" IDENT)*
```
Example: `use stdlib/rat` imports all public decls from that file.

### Visibility
- `fn foo(...)` is file-local by default.
- `pub fn foo(...)` is exported / visible to `use`rs.
- Same for `tablet` / `tablet` with a `pub` prefix.

### Name resolution
Simplest: `use stdlib/rat` copies all `pub` names from
`stdlib/rat.tpu` into the current file's namespace unqualified (Go's
dot import, Python's `from x import *`). Qualified
`use stdlib/rat as r` + `r.rat_abs(x)` comes later.

### Driver changes
When the driver sees `use stdlib/rat` it resolves the file:
- Same-directory relative — fine for single-project.
- `TUPPU_PATH` env var — search path for modules.
- Bundled stdlib lives at `stdlib/`, resolve as
  `stdlib/rat` → `stdlib/rat.tpu`.
- Existing stdlib auto-include becomes opt-in via `use`.

### Files to touch
- `src/tuppu/ast.py` — `UseDecl(path: list[str])`, `pub: bool` on
  FnDecl/StructDecl.
- `src/tuppu/parser.py` — parse `use`.
- `src/tuppu/driver.py` — resolve `use` to additional source files;
  parse/typecheck in dependency order.
- `src/tuppu/typecheck.py` — per-file visibility, import resolution.
- Migration: existing stdlib functions marked `pub`; examples add
  `use stdlib/rat` / `use stdlib/sex` at top.

## 3. Sex Phase 3 — /, escape analysis (3a done)

### Status: 3a + 3b done, 3c remaining

### 3a. ~~Native `sex * sex`~~ — **done.**

Shipped via the rat path: `sex * sex → (a as rat) * (b as rat) →
__tuppu_rat_to_sex`. `__tuppu_rat_to_sex` lives in codegen.py, runs
the regularity check (strip 2s, 3s, 5s from den), decomposes the
integer quotient into base-60 digits, then iteratively extracts
fractional digits until rem == 0 or we hit SEX_MAX_DIGITS (trap).

This also implemented `rat → sex` as a coercion. Tests:
`tests/test_sex.py::test_rat_to_sex_*`, `test_sex_mul_*`.

Future-precision note: if the 16-digit buffer starts cramping real
workloads, the "digit cross-product" alternative would let sex*sex
exceed i64·i64 precision by staying in digit space — but that needs
sign handling, alignment via radix-shift, and a non-SIMD carry chain
across partial sums. Not on the roadmap unless someone asks.

### 3b. ~~Native `sex / sex`~~ — **done.**

Shipped alongside 3a's `*` path — same dispatch, same trap semantics
(rat_reduce traps on divisor-zero, rat_to_sex traps on non-regular
quotient). Bonus: mixed `sex op int` now works for +, -, *, / via a
typecheck promotion to DISH + codegen call to `__tuppu_int_to_sex`.

Remaining: `%` could follow the same shape but requires implementing
`rat %` first; deferred unless needed.

### 3c. Escape-analysis rat-fallback

The compile-time optimization pass that makes the digit form not
secretly slow.

**Question:** does a given sex value's **digit sequence** ever escape?

A sex value is **digit-observed** if any of these reach it:
- Printed via `print`/`println` (renders Babylonian).
- Returned from a function whose return type is `sex` and the caller
  digit-observes the result.
- Stored in a tablet field of type `sex` that is later digit-observed.
- Passed as arg to a function whose param slot is `sex` and is
  digit-observed inside the callee.

A sex value is **rat-only** if its sole uses are:
- Cast `as rat` / `as i64`.
- Arithmetic (both operands rat-only, result inherits).
- Comparison (inspects num/den of aligned rats).
- Bound to a `: rat` typed slot (coercion forces to rat).

For rat-only values, emit LLVM IR that uses the compact `{num, den}`
tablet directly and skip `__tuppu_sex_to_rat`.

**Analysis shape:**
- Whole-program (we don't have separate compilation).
- **First cut: intraprocedural, function boundaries always
  digit-form.** Catches local temporaries — the common case for
  tight arithmetic loops. Simple to implement and verify.
- **Follow-up: inter-procedural.** Function signatures get two
  specializations (digit vs rat), callers pick.

**Files to touch:**
- `src/tuppu/escape.py` (new) — the analysis pass: walk AST, tag
  each sex-typed binding / expression with `digit_observed: bool`.
- `src/tuppu/codegen.py` — consult escape tags at lowering time;
  emit rat-form tablet for rat-only sex values; the builtins
  (add/sub, to_rat, int_to_sex) become no-ops for rat-only values.
- `tests/test_sex.py` — verify the optimization fires on expected
  shapes (e.g., a purely arithmetic function should have zero
  calls to `__tuppu_sex_add` in its IR post-optimization), and
  doesn't fire when digits escape.

### SIMD hook (already in Phase 2, note for future)

The current `__tuppu_sex_add` uses `<16 x i8>` for the raw digit
sum. Carry propagation is scalar because of the serial dependency.
For multiplication we'd get another SIMD surface on the partial
products, but carry propagation dominates — so SIMD wins are
smaller. Not the first priority.

## Common pitfalls users hit

Notes for future-self (or future-user) reading scratch files:

- **`1 3` vs `1;3`:** space-separated is integer-form sex (1·60+3 =
  63); semicolon-separated is fractional (1 + 3/60 = 21/20). They
  are very different values that print very differently. Get this
  wrong and `length_sq` gives `1613/720` instead of `8065/1`.
- **Struct literal lookahead:** `Name { field : ...` parses as a
  tablet literal. To write a block with a leading identifier, avoid
  `{ ident : ... }` shape — but blocks should start with `step`,
  `mut`, etc., so this rarely bites.
- **Tuppu uses newlines**, not `;`, as statement terminator.  `;`
  is *exclusively* a sexagesimal radix marker. `,` is *exclusively*
  an argument separator.

## Conventions / preferences to keep

- **No Claude attribution on commits or PRs.** Already saved in
  `~/.claude/.../memory/` and enforced via `~/.claude/settings.json`
  `attribution.commit = ""` / `attribution.pr = ""`.
- **Git:** user.name is `drewlark`, user.email is NOT globally set.
  Commits use `git -c user.email='drewlarkplusplus@gmail.com' commit ...`
  — never modify global config.
- **Commit style:** concise imperative title, blank line, then a
  bulleted body for non-trivial changes.
- **Statement terminator:** newlines only.
- **Testing:** `.venv/bin/pytest` runs all. Keep tests in sync with
  code; `test_examples.py` is a gatekeeper that forces every new
  `.tpu` in `examples/` to be registered.
- **CLI entry:** `./tuppu` (bash wrapper) and `tuppu` (inside venv
  once `pip install -e .` picks up `[project.scripts]`).
- **Scratch files:** put experiments in `scratch/` or at repo root
  as `fun.tpu` / `scratch.tpu`. All gitignored.

## Session rehydration checklist

If starting a fresh session after this compact:

1. `cd /Users/drew/code/compilerfun` and read this file.
2. `.venv/bin/pytest` — expect 417 passing.
3. `git log --oneline -12` — recent timeline: sex Phase 3a/3b,
   struct field mutation, codegen.py split into mixins package,
   elif + did-you-mean, recursive tablets + wedge handles + auto-
   release + escape check, tablet/wedge/seal rename, minimal
   generics (+ stdlib/list.tpu rewritten to `List<T>`).
4. Read `SPEC.md` §4.5 (tablets), §4.6 (wedges), §14 (non-goals).
5. **Agreed next task: sum types.** `seal` is reserved and ready to
   claim. See "Roadmap (agreed 2026-04-22)" section above for the
   design sketch (tagged union runtime, variant constructors,
   exhaustive match — v0.1 scope is flat patterns only).
   Ultimate goal: fold pieces of the compiler into Tuppu
   (self-hosting). Min self-host set: sum types → fn-as-value →
   closures → dynamic strings → maps → file I/O. The full ranked
   roadmap is in the same section above.
6. `FUTURE_OPTIMIZATIONS.md` (gitignored) captures design sketches
   for a `--strict-dish` flag, the SEX 20→16 byte shrink, SIMD carry
   in sex_add, and other perf/language ideas. Don't forget on the
   next perf pass.
