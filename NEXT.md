# Tuppu — what's next

Pickup notes. Written after finishing the sex/dish Babylonian redesign
and the initial GitHub push.

## Where we are (as of 2026-04-22)

- **v0.1 feature-complete** per SPEC.md — lexer, Pratt parser, type
  checker, LLVM codegen via llvmlite.
- Private repo: https://github.com/Drewlark/tuppu (branch `main`).
- **359 tests passing** (277 base + 21 struct + 17 string + 21
  ergonomics + 7 sex-identity + 11 native-sex-arithmetic + examples).
- CLI: `./tuppu run file.tpu` and `./tuppu build ... -o out`.
- Bundled stdlib auto-included; pass `--no-stdlib` to opt out.
- Compiler's in Python (`src/tuppu/`); stdlib's in Tuppu
  (`stdlib/*.tpu`).

What works:
- Primitives: `i8..i64`, `u8..u64`, `bool`, `rat`, `sex`/`dish`.
- **User-defined structs / seals** — nominal, by-value, forward refs
  resolved via topological sort. `struct`/`seal` are interchangeable
  keywords. Construction `Point { x: 3, y: 4 }`, field access `p.x`,
  structs as params/returns and in `mut` bindings.
- **Variable-length strings** — `str` is a built-in seal
  `{ ptr: *u8, len: i64 }`, auto-injected. String literals produce
  `str` values, `fn f(s: str)` works, `s.len` / `s[i]` work,
  `print(s)`/`println(s)` go through `write(2)` so embedded NULs
  survive. Stdlib `str_eq`/`str_starts_with`/`str_ends_with`/
  `str_is_empty`/`str_index_of`.
- Raw pointer type `*T` — type-only, no expression-level ops.
- **Char literals** `'a'`, `'\\n'`, etc. — type `u8`.
- **Multi-arg `print`/`println`** — `println("x=", x, " y=", y)`.
- **Augmented assignment** `+= -= *= /= %=` (parser-desugared).
- **`for name in iter { … }`** — works over `str` (u8 bytes), tablets,
  and comptime tables. Loop variable is step-bound.
- **Mixed-width int comparisons** promote to the wider type, matching
  the `if`-arm unification rule.
- `step` (SSA) and `mut` (alloca) bindings; assignment
- `if`/`else` as expression; `while`; `yield` early-return
- User functions with recursion; multi-file compilation
- rat + sex arithmetic (sex lowers to rat with a compile warning)
- `table name[lo..hi]: T = genfn` — comptime-evaluated static tables
- `tablets[N]T` — chained-chunk growable storage, `release` to free
- Intrinsics: `print`, `println`, `read_int`, `rat(n, d)`
- Proper type errors with source `line:col`, CLI prints cleanly
- Compile-time warning infrastructure

What doesn't yet:
- Imports / namespacing
- Dynamic string ops (concat, slice, case) — needs tablets-backed alloc
- `read_line()` — same blocker
- Expression-level pointer ops (`*p`, `&x`, `p + 1`) — intentional
- Recursive structs (would need identified types + heap indirection)
- Struct field mutation (`p.x = 5`) — only whole-struct reassign
- `impress` reinterpret cast (documented in SPEC.md §14)
- f64 (lexed but no codegen; `sex as f64` errors cleanly)
- Closures / first-class functions
- Self-hosting

## Priority order (agreed)

1. ~~**Structs**~~ — **done.** Nominal structs with field access,
   pass-by-value, nested, and forward-reference support via topo sort
   in codegen. `seal` shipped as the Babylonian-flavored alias for
   `struct`. See `examples/points.tpu` and `tests/test_struct.py`.
2. ~~**Better strings**~~ — **done.** `str` is now a built-in seal
   `{ ptr: *u8, len: i64 }` auto-injected by the driver. Literals
   produce `str` values, `s.len`/`s[i]`/print/params all work,
   `stdlib/str.tpu` has the non-allocating byte ops. See
   `examples/greeting.tpu` and `tests/test_string.py`. Dynamic
   (allocating) string ops deferred until we wire tablets into a
   dynamic-string story — see §4 below.
3. **Sex/dish redesign** — **Phase 2 shipped.**
   - Phase 1: digit-form runtime + Babylonian printing + `sex as rat`
     as an explicit reduce.
   - Phase 2: native `sex + sex` and `sex - sex` — radix alignment,
     16-lane SIMD digit add, scalar carry propagation, mixed-sign
     handling via magnitude compare + borrow-propagating subtract.
     No more warning for `+`/`-`. See `__tuppu_sex_add` in codegen.py.
   - Phase 3 (remaining): native `*`, `/` with regularity checks, and
     the escape-analysis pass for rat-fallback specialization.
     See §5 for the design.
4. **Imports** — cleanup; pay when stdlib grows enough to need
   namespaces. After sex phase 3.

### Future: `impress` reinterpret cast

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

## 1. Structs — shipped (2026-04-22)

Final implementation notes (as built, for future self-reference):

- `TyStruct(name)` is nominal and name-equal; field layout lives in
  `Checker.struct_fields` keyed by struct name (not on the TyStruct).
  This keeps the type itself hashable and cheap to compare.
- Codegen uses `ir.LiteralStructType` — anonymous, non-recursive.
  A topological sort over struct decls (`_register_structs`) resolves
  forward references so declaration source order doesn't matter.
  Direct cycles (`struct Node { next: Node }`) are rejected with a
  clear message; recursive structs are an explicit non-goal until we
  pick up identified types for heap indirection.
- Struct literals (`Point { x: 3, y: 4 }`) are parsed in prefix position
  via a 3-token lookahead (`IDENT LBRACE IDENT COLON`). This avoids
  colliding with `if cond { body }` because a block never starts with
  `IDENT COLON` (statements require `step`/`mut`).
- Field access on a struct value generates `extract_value`. Because a
  user struct `{ i64, i64 }` is structurally equal to `rat` at the LLVM
  level, `_gen_field` checks `_struct_name_for(target.type)` via
  *identity* before falling back to the rat comparison.
- Empty structs and field mutation (`p.x = 5`) are intentionally out of
  scope for this round. Whole-struct reassignment works via `mut`.

## 2. Strings — shipped (2026-04-22)

Final shape, for future-self reference:

- `str` is a built-in seal, *not* a stdlib file. `driver._builtin_decls`
  prepends `StructDecl("str", [("ptr", *u8), ("len", i64)])` to every
  parsed program. This works with `--no-stdlib` and keeps string
  literals usable in bare programs.
- String literals (`A.StringLit`) lower to `str` seal values — the bytes
  live in a deduped global constant, the seal references it with a
  pre-baked length. Arbitrary bytes including embedded NUL are
  permitted.
- `print(s)` / `println(s)` go through `write(2)` (fd 1), not printf,
  to preserve embedded NULs. `fflush(NULL)` is called before each
  string write so ordering with `print(int)` / `print(rat)` (which use
  printf) stays stable.
- `s.len` is just ordinary struct field access — free once structs
  landed.
- `s[i]` is compiler-recognized for the `str` seal specifically: GEP on
  `s.ptr` with a bounds check against `s.len`. `*u8` itself is *not*
  user-indexable — that would expose pointer arithmetic, which we
  explicitly don't want at this level.
- Retired: the old `[N]u8` special case in the typechecker, and `TyStr`
  entirely. Array types (`[N]T`) now error out cleanly; we can
  reintroduce them when there's a real need.
- Stdlib: `stdlib/str.tpu` holds `str_eq`, `str_starts_with`,
  `str_ends_with`, `str_is_empty`, `str_index_of`. All non-allocating
  byte-level. Anything producing a new `str` is deferred.

## 3. Imports — design sketch

### Grammar
```
use_decl = "use" path
path     = IDENT ("/" IDENT)*
```
Example: `use stdlib/rat` imports all public decls from that file.

### Visibility
Default: `fn foo(...)` is file-local.
`pub fn foo(...)` is exported / visible to `use`rs.

### Name resolution model
Simplest: `use stdlib/rat` copies all `pub` names from `stdlib/rat.tpu`
into the current file's namespace unqualified (like Go's dot import or
Python's `from x import *`). No `rat.rat_abs(x)` qualified form yet —
keeps lookup simple.

Later: qualified `use stdlib/rat as r` + `r.rat_abs(x)`. That needs
module-aware scoping in the type checker.

### Driver changes
When the driver sees `use stdlib/rat` it needs to locate the file.
Options:
- Same-directory relative — fine for single-project use
- `TUPPU_PATH` env var — search path for modules
- Bundled stdlib lives at `stdlib/`, resolve as `stdlib/rat` → `stdlib/rat.tpu`

### Files to touch
- `src/tuppu/ast.py` — `UseDecl(path: list[str])`, `pub: bool` on FnDecl/StructDecl
- `src/tuppu/parser.py` — parse `use`
- `src/tuppu/driver.py` — resolve `use` to additional source files;
  repeat parse/typecheck per file in dependency order; current stdlib
  auto-include becomes opt-in via `use`
- `src/tuppu/typecheck.py` — per-file visibility, import resolution
- Migration: existing stdlib functions marked `pub`; examples add
  `use stdlib/rat` / `use stdlib/sex` at top

## 5. Sex phase 2 — native digit arithmetic + escape analysis

Goal: make sex arithmetic stay in digit form (so the warning goes away
and the feature becomes *honest*), while ensuring the digit form isn't
secretly expensive when users don't actually look at the digits.

### Native digit arithmetic

Target runtime helpers:

- `__tuppu_sex_add(a, b) -> sex` — handles same-sign add via digit-wise
  base-60 add with carry propagation; mixed signs dispatch to sub.
- `__tuppu_sex_sub(a, b) -> sex` — borrow propagation.
- `__tuppu_sex_mul(a, b) -> sex` — probably goes through the i128 path
  (convert to int via digits, multiply, convert back) or a full n² digit
  cross-product. Start with the simple path.
- `__tuppu_sex_cmp(a, b) -> i32` — align radix, lexicographic compare.
- Division: explicit `sex / sex` emits a **regularity check** — the
  denominator's int value must be 2^a · 3^b · 5^c. If not, error at
  compile time for literals, or trap at runtime otherwise. Result
  fractional digit count bounded by max(a, b) + some small slack.

### SIMD hook

The digit buffer is 16 × u8 — fits exactly in an SSE register. The
inner add loop is:

```
; load a.digits, b.digits as <16 x i8>
%sum = add <16 x i8> %a, %b
; sum now has raw per-digit sums, each ≤ 118
; carry propagation is sequential, but we can scan or peel the first
;   iteration in parallel
```

For carry propagation: not vectorizable in general (serial dependency),
but we can do it in a scalar loop over the 16-lane vector extracted to
a stack array. Or split: first SIMD reduces simultaneous adds, then a
one-pass scalar loop does carries.

Alignment: current SEX struct has natural alignment 1 (all fields are
u8). For vectorized loads we want 16-byte alignment — set an explicit
`.align 16` on the digit array field when emitting loads, or allocate
via an aligned alloca. Need to test whether LLVM's auto-vectorizer
picks up the scalar loop for us; probably not because of the carry
dependency.

### Escape analysis

The question: does a given sex value's **digit sequence** ever escape?

A sex value is **digit-observed** if any of these reach it:
- Passed to `print` / `println` (always renders in Babylonian)
- Returned from a function whose return type is `sex` (the caller may
  then observe digits; propagates backwards in a fixpoint)
- Stored in a struct field of type `sex` (the field may be later read
  and printed)
- Passed as an arg to a function whose param is `sex` in a digit-
  observed position

A sex value is **rat-only** if its sole uses are:
- Cast `as rat` / `as i64`
- Arithmetic (both operands rat-only, result inherits rat-only-ness)
- Comparison (inspects num/den of aligned rats)
- Bound to a `: rat` typed slot (coercion forces to rat)

For rat-only values, emit LLVM IR that uses the compact
`{num, den}` struct directly. The `__tuppu_sex_to_rat` helper is
skipped; sex + sex becomes rat + rat with no digit overhead.

Analysis shape: whole-program, intraprocedural first, then a fixpoint
over function signatures (which args/returns need digit form vs rat
form). Each function may be emitted in two specializations — one for
each form. Callers pick based on their own analysis.

First implementable cut: **purely intraprocedural, function-boundary
always digit-form**. Catches local temporaries (the common case for
tight arithmetic loops). Cross-function specialization comes next.

### Files to touch (rough)

- `src/tuppu/codegen.py` — `_get_sex_add`, `_get_sex_sub`, optional
  `_get_sex_mul`, `_get_sex_cmp`. Alignment hint on SEX struct or on
  allocas. Replace the `_gen_binary` warn-and-lower path with dispatch
  on the native helpers; keep the warning only for the unsupported
  cases (div by non-regular) for now.
- `src/tuppu/typecheck.py` — drop the warning for supported native ops.
- `src/tuppu/escape.py` (new) — the analysis pass: walk AST, tag each
  sex-typed binding / expression with `digit_observed` or not.
- `src/tuppu/codegen.py` again — consult escape tags at lowering time;
  emit rat-form struct for rat-only sex values.
- `tests/test_sex.py` — native arithmetic cases, escape-analysis
  specialization cases, regularity check cases.

### Deferred past Phase 2

- Cross-function escape analysis (monomorphization of functions with
  sex params/returns per caller-observed form).
- SIMD-optimized multiplication.
- Regularity check for explicit `rat as sex`.


## Conventions / preferences to keep

- **No Claude attribution on commits or PRs.** Already saved in
  `~/.claude/.../memory/` and enforced via `~/.claude/settings.json`
  `attribution.commit = ""` / `attribution.pr = ""`.
- **Git:** user.name is `drewlark`, user.email is NOT globally set.
  Commits use `git -c user.email='drewlarkplusplus@gmail.com' commit ...`
  — never modify global config.
- **Commit style:** concise imperative title, a blank line, then a
  bulleted body for non-trivial changes.
- **Statement terminator:** newlines only. `;` is *exclusively* for
  sexagesimal literals. `,` is *exclusively* an argument separator.
- **Testing:** `.venv/bin/pytest` runs all. Keep tests in sync with
  code; the `test_examples.py` gatekeeper test forces every new `.tpu`
  in `examples/` to be registered.
- **CLI entry:** `./tuppu` (bash wrapper at repo root) and `tuppu`
  (inside the venv once `pip install -e .` picks up `[project.scripts]`).

## Session rehydration checklist

If starting a fresh session after this compact, do these first:

1. `cd /Users/drew/code/compilerfun` and read this file.
2. `.venv/bin/pytest` to confirm 359 tests still pass.
3. `git log --oneline -5` to see where we are on the timeline.
4. Read `SPEC.md` §4 for the type grammar and §14 for explicit non-goals.
5. Agreed next task: **sex phase 3** — native digit-form
   multiplication, division with regularity checks, and the
   escape-analysis rat-fallback specialization. See §5 below for
   rules. Imports wait until we're actually hurting for namespaces.
