# Tuppu — what's next

Pickup notes. The Babylonian sexagesimal redesign is the active
front; imports and dynamic strings are queued behind it.

## Where we are (as of 2026-04-22)

- **v0.1 feature-complete** per SPEC.md — lexer, Pratt parser, type
  checker, LLVM codegen via llvmlite.
- Private repo: https://github.com/Drewlark/tuppu (branch `main`).
- **365 tests passing.**
- CLI: `./tuppu run file.tpu` and `./tuppu build ... -o out`.
- Bundled stdlib auto-included; pass `--no-stdlib` to opt out.
- Compiler's in Python (`src/tuppu/`); stdlib's in Tuppu
  (`stdlib/*.tpu`).
- `scratch/` is gitignored (along with `fun.tpu`, `scratch.tpu`) for
  user experimentation without polluting history.

What works:
- Primitives: `i8..i64`, `u8..u64`, `bool`, `rat`, `sex`/`dish`.
- **User-defined structs / seals** — nominal, by-value, forward refs
  resolved via topological sort. `struct`/`seal` are interchangeable
  keywords. Construction `Point { x: 3, y: 4 }`, field access `p.x`,
  structs as params/returns and in `mut` bindings.
- **Variable-length strings** — `str` is a built-in seal
  `{ ptr: *u8, len: i64 }`, auto-injected. `fn f(s: str)`, `s.len`,
  `s[i]`, `print(s)`/`println(s)` via `write(2)` (embedded NULs
  survive). Stdlib `str_eq`/`str_starts_with`/`str_ends_with`/
  `str_is_empty`/`str_index_of`.
- Raw pointer type `*T` — type-only, no expression-level ops.
- **Char literals** `'a'`, `'\n'`, etc. — type `u8`.
- **Multi-arg `print`/`println`** — `println("x=", x, " y=", y)`.
- **Augmented assignment** `+= -= *= /= %=` (parser-desugared).
- **`for name in iter { ... }`** — works over `str` (u8 bytes),
  `tablets[N]T`, and comptime tables. Loop variable is step-bound.
- **Mixed-width int comparisons** promote to the wider type (matches
  `if`-arm unification).
- **Sex/dish Phase 1 + 2:**
  - Distinct 20-byte runtime `{ [16]u8 digits, u8 radix, u8 count,
    i8 sign, u8 pad }`.
  - Native Babylonian printing (`1;30`, `1;24 51 10`, `-1;30`).
  - `sex as rat` is a real reduce; `sex as i64` truncates via rat.
  - `int → sex` decomposes an i64 into base-60 digits (integer form).
    So `Point { x: 0, y: 0 }` with sex fields works.
  - Native `sex + sex` and `sex - sex` via `__tuppu_sex_add` — radix
    alignment, 16-lane SIMD digit add, scalar carry, mixed-sign via
    magnitude compare + borrow-propagating subtract. No warning.
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
- Native sex `*`, `/`, `%` (still warn-lower to rat — **Phase 3**).
- `rat → sex` (needs regularity check — Phase 3 too).
- Imports / namespacing.
- Dynamic string ops (concat, slice, case) — needs tablets-backed
  alloc plus a lifetime/ownership story.
- `read_line() -> str` — same blocker.
- Expression-level pointer ops (`*p`, `&x`, `p + 1`) — intentional.
- Recursive structs (would need identified types + heap indirection).
- Struct field mutation (`p.x = 5`) — only whole-struct reassign.
- `impress` reinterpret cast (documented in SPEC §14).
- f64 (lexed but no codegen; `sex as f64` errors cleanly).
- Closures / first-class functions.
- Self-hosting.

## Priority order (agreed)

1. ~~**Structs**~~ — **done.** See `examples/points.tpu`,
   `tests/test_struct.py`.
2. ~~**Better strings**~~ — **done.** See `examples/greeting.tpu`,
   `tests/test_string.py`, `stdlib/str.tpu`.
3. ~~**Ergonomics bundle**~~ (char lits, multi-arg print, `+=`, for,
   mixed-width int compare) — **done.** `tests/test_ergonomics.py`.
4. ~~**Sex/dish Phase 1**~~ — **done.** Digit-form runtime +
   Babylonian printing + `sex as rat` as a real reduce helper.
5. ~~**Sex/dish Phase 2**~~ — **done.** Native `sex + sex` / `-`,
   int→sex, error location tracking. See `__tuppu_sex_add`.
6. **Sex/dish Phase 3** — **Next.** Native `*` and `/` with
   Babylonian regularity check, plus the escape-analysis pass for
   rat-fallback specialization. See §3 below for the full design.
7. **Imports** — cleanup; pay when stdlib grows enough to need
   namespaces. After sex Phase 3.

## 1. Future: `impress` reinterpret cast

Planned for v0.2 (documented in `SPEC.md` §14). Keeps `as` purely for
safe numeric conversions while giving same-layout seal reinterpretation
its own vivid, mildly-scary spelling:

```
step v: Vector = impress p as Vector
```

Rules: source and target must be seals (not primitives) with
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
- Same for `struct` / `seal` with a `pub` prefix.

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

## 3. Sex Phase 3 — native *, /, and escape analysis

The remaining sex/dish work. Three sub-tasks, all worth doing in the
order listed; multiplication is the most visible win, escape analysis
is the learning chunk.

### 3a. Native `sex * sex`

Two implementation paths to choose between:

- **Via rat (easy).** `sex * sex → (sex as rat) * rat → rat`. The
  digit form is lost in the result — but you can reduce back via a
  new `__tuppu_rat_to_sex` helper that runs a regularity check first
  (denominator must be 2^a · 3^b · 5^c) and decomposes into
  fractional digits via repeated × 60 / den. This also implements
  `rat as sex`. Good first cut; covers the regular-number case
  exactly, errors on non-regular.
- **Digit-cross-product (hard).** O(n·m) digit multiplications, each
  in base 60, each producing a partial sum with a carry. Handle sign
  separately. Alignment is simpler than add (just shift the radix by
  the sum of frac counts). No SIMD easy path here because of the
  carry chain across partial sums.

**Recommend: path 1 first.** It gives `rat → sex` and native-ish
multiplication in one shot. Path 2 can come later if the precision
ceiling bites.

**New helper:** `__tuppu_rat_to_sex(rat) -> sex`. Algorithm:
1. Extract num, den. Handle sign.
2. Factor den to check regularity: repeatedly divide by 2, 3, 5
   until den == 1 (regular) or none of those divide (not regular).
3. If not regular, **runtime trap** (or emit `CompileError` at
   compile time for literal operands — we know at codegen-time).
4. If regular, produce digits:
   - int_digits = Horner-decompose (num / den) into base 60.
   - For fractional, repeatedly multiply remainder by 60 and divide
     by den to extract the next frac digit, until remainder is 0.
   - Track digit counts; trap on SEX_MAX_DIGITS overflow.

Regularity check can also be done at compile time for literal
operands — cleaner UX since the user sees the error immediately.

### 3b. Native `sex / sex`

Once 3a lands, `sex / sex` is (sex as rat) / rat → rat → sex. Same
regularity check applies and errors the same way. Write it as a
small helper that chains `__tuppu_rat_to_sex` and `rat_div`.

### 3c. Escape-analysis rat-fallback

The compile-time optimization pass that makes the digit form not
secretly slow.

**Question:** does a given sex value's **digit sequence** ever escape?

A sex value is **digit-observed** if any of these reach it:
- Printed via `print`/`println` (renders Babylonian).
- Returned from a function whose return type is `sex` and the caller
  digit-observes the result.
- Stored in a struct field of type `sex` that is later digit-observed.
- Passed as arg to a function whose param slot is `sex` and is
  digit-observed inside the callee.

A sex value is **rat-only** if its sole uses are:
- Cast `as rat` / `as i64`.
- Arithmetic (both operands rat-only, result inherits).
- Comparison (inspects num/den of aligned rats).
- Bound to a `: rat` typed slot (coercion forces to rat).

For rat-only values, emit LLVM IR that uses the compact `{num, den}`
struct directly and skip `__tuppu_sex_to_rat`.

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
  emit rat-form struct for rat-only sex values; the builtins
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
  struct literal. To write a block with a leading identifier, avoid
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
2. `.venv/bin/pytest` — expect 365 passing.
3. `git log --oneline -6` — timeline: initial import, bundled v0.1
   features, untrack fun.tpu, sex Phase 1 + ergonomics, sex Phase 2,
   int→sex + error locations.
4. Read `SPEC.md` §4.3 for the current sex spec, §14 for non-goals.
5. **Agreed next task: sex Phase 3.** Start with 3a (`rat → sex`
   with regularity check, then native `*` built on top of it), then
   3b (`/`), then 3c (escape analysis — the big compiler-learning
   chunk). See §3 in this file for the design.
