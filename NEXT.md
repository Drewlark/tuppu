# Tuppu — what's next

Pickup notes. The Babylonian sexagesimal redesign is the active
front; imports and dynamic strings are queued behind it.

## Where we are (as of 2026-04-22)

- **v0.1 feature-complete** per SPEC.md — lexer, Pratt parser, type
  checker, LLVM codegen via llvmlite.
- Private repo: https://github.com/Drewlark/tuppu (branch `main`).
- **484 tests passing.**
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
  `{ ptr: *u8, len: i64, cap: i64 }`, auto-injected. `cap == 0`
  marks a borrow (string literals, fn params, reads of tracked
  values); `cap > 0` marks heap ownership. `__tuppu_str_release`
  frees only when `cap > 0` — so passing a str through a call is
  safe: the call site forces cap=0 on every str arg, the callee
  registers the param in its cleanup frame uniformly, release
  becomes a no-op, caller keeps sole ownership. Unbound rvalues
  (`println(str_concat(a, b))`) get an anonymous cleanup slot at
  the consumer site so heap bytes are freed at the surrounding
  scope's exit — no leaks in tight loops. `fn f(s: str)`, `s.len`,
  `s[i]`, `print(s)`/`println(s)` via `write(2)` (embedded NULs
  survive). Python-style slice syntax `s[lo:hi]`, `s[:hi]`, `s[lo:]`,
  `s[:]` lowers to `__tuppu_str_slice` with elided bounds defaulting
  to `0` / `s.len`; bounds-checked, copies into a fresh heap str.
  **Ownership transfers on tail return** — a block whose trailing
  expression is an Ident naming a local heap-owned binding
  deregisters that entry from the scope-exit cleanup, so the caller
  receives the live value instead of a dangling ptr.
  **Growable byte buffer (a.k.a. str_buf)** — a `mut tablets[64]u8`
  accumulates bytes via the amortized-O(1) `push`, then
  `bytes_to_str(buf)` flattens the chain into a fresh heap str in
  one pass. Total cost to build an n-byte string is O(n); no
  quadratic-rebuild trap. `stdlib/str_buf_append` wraps the common
  "append a str's bytes" shape. Stdlib `str_eq`/`str_starts_with`/
  `str_ends_with`/`str_is_empty`/`str_index_of`/`str_find`/
  `str_contains`/`str_repeat`/`bool_to_str`/`rat_to_str`
  (last two were native, migrated to Tuppu now that the language
  can express them).
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
- **Struct field auto-release** — a user struct (mut or step) that
  transitively holds cleanup-bearing fields (str, tablets, nested
  structs) gets a generated `__tuppu_struct_<name>_release` that
  GEPs to each owning field and dispatches the appropriate
  release. Recursive: Outer release calls Inner release calls
  str/tablets release. Plain structs (no cleanup fields) emit no
  release fn. Reassignment of a cleanup-bearing struct slot
  releases the old value before the store. Tail-return ownership
  transfer extends to structs — returning a local mut/step struct
  binding hands ownership out to the caller's binding.
- **Step-binding borrow rule** — `step x = y` where y is an Ident
  naming a cleanup-bearing binding is a borrow: x shares y's
  metadata and doesn't register its own cleanup. `Variable.transfer_on_tail`
  records the owner's entry name so returning x from a block
  transfers y (the real owner) out of the cleanup frame.
  Transitive: `step w = y; step v = w` has v transfer-chain
  through to y. Prevents the double-free that plain duplicate-
  registration produced at scope exit.
- **Call-site neutering for cleanup-bearing struct args** — passing
  a struct (str / tablets / nested cleanup) to any fn arg zeros
  the cleanup markers on every heap-owning field in the callee's
  view (cap=0 for str, zero-init for tablets, recurses). The
  callee's mut-param cleanup then no-ops on every field while
  the caller retains sole ownership. Fresh rvalues (struct
  literals, Call results) get an anonymous slot in the caller's
  frame so their heap fields free at scope exit.
- **Nested tablets-method dispatch** — `buf.bytes.push(b)` works
  when `buf` is a mut struct and `bytes` is a `tablets[N]T` field;
  codegen GEPs to the inner slot and dispatches on a synthetic mut
  reference. Root must still be a mut Ident so the lvalue
  machinery has a slot to mutate.
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
- **`colophon fn name(params) -> type`** — typed FFI to libc.
  The scribe's endnote pointing outside the tablet. Declares an
  extern the compiler emits at LLVM level and marshals at every
  call site: Tuppu `str` becomes a fresh malloc'd NUL-terminated
  cstr on the way out (freed after the call) and a heap-owned
  Tuppu str on the way back via `strlen + malloc + memcpy`; NULL
  returns yield an empty str. Integer primitives pass through;
  `bool` widens to `i8` for C-ABI stability. First cut allows
  ints, bool, and str; user tablets (for e.g. `sockaddr_in`) and
  opaque handles are the next FFI landings. See
  `tests/test_colophon.py` — `atoi`, `getenv`, `exit` as proofs.
- **`for name in iter { ... }`** — works over `str` (u8 bytes),
  `tablets[N]T`, and comptime tables. Loop variable is step-bound.
- **Mixed-width int comparisons** promote to the wider type (matches
  `if`-arm unification).
- **Sum types (`seal`) + pattern matching.** `seal Option<T> { Some(T),
  None }`; non-generic and generic; positional variant fields; `match`
  as an expression with wildcard arms (`_ => ...`) and per-slot
  wildcards (`Both(_, b)`); exhaustiveness enforced unless a wildcard
  arm is present. Runtime is `{ i8 tag, [N x i64] payload }` — payload
  bitcast to a per-variant struct for read/write. Generic seals use the
  same lazy monomorphization as generic tablets. Bidirectional type
  checking threads an optional `expected` type through bindings, yield,
  fn body tails, assignment RHS, and call args — so `None` works
  without an annotation when context pins `T`. See
  `examples/omens.tpu`, `tests/test_sum.py`.
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
- Fn-as-value / closures — the next planned chunk after dynamic strings.
- Native sex `%` (still warn-lower to rat; rat % itself isn't
  implemented either, so this errors at codegen).
- Mixed sex × rat ops (still warn, stay as rat — deliberate, since
  the rat operand has abandoned digit form).
- Escape-analysis rat-fallback for rat-only sex values (**Phase 3c**
  — the big compiler-learning chunk).
- `--strict-dish` flag (see FUTURE_OPTIMIZATIONS.md for the sketch).
- Imports / namespacing.
- `read_line() -> str` — needs stdin wrapper, not yet added.
- Expression-level pointer ops (`*p`, `&x`, `p + 1`) — intentional.
- `impress` reinterpret cast (documented in SPEC §14).
- f64 (lexed but no codegen; `sex as f64` errors cleanly).
- Closures / first-class functions.
- Self-hosting.

## Known v0.1 limitations (deliberate cuts)

Things that are shipping with known reductions, documented here so
future-us doesn't trip on them and a v0.2 pass can revisit them:

### Sum types (`seal` / `match`)
- **Flat patterns only.** No nesting (`Some(Circle(r))`), no guards
  (`Some(x) if x > 0`), no or-patterns (`Some(1) | Some(2)`).
  Adding these is a v0.2 effort — ML-family matching is a bigger
  design lift than it looks (exhaustiveness for nesting is subtle).
- **Positional variant fields only.** `Circle(rat)` works, but
  `Circle(radius: rat)` doesn't parse. Trivial to add once we want
  named-field patterns.
- **Globally-unique variant names.** `seal A { X }; seal B { X }`
  is rejected because variants are looked up in one flat table.
  Qualified syntax (`A::X`) would unblock this; not yet designed.
- **Nullary variants of generic seals need context to pin T.**
  `step x = None` (no annotation) errors with "cannot infer T".
  Works fine in return position, call args, or annotated bindings
  (`step x: Option<i64> = None`) thanks to the expected-type
  threading. Turbofish (`None::<i64>`) isn't supported.
- **SPEC.md doesn't cover `seal` / `match` yet.** Source of truth
  is `tests/test_sum.py` and `examples/omens.tpu` until the spec
  pass catches up.

### Dynamic strings
- **Slicing always copies.** `str_slice(s, lo, hi)` allocates a
  fresh buffer and memcpies. Zero-copy views would need lifetime
  or refcount machinery we don't have; the copy keeps the model
  coherent and is the honest price of value semantics. Move/borrow
  analysis later can elide the copy when the source is provably
  unused past the call.
- **Immutable strings only.** All ops produce new strings; `mut s`
  allows reassignment (old value is released before storing new),
  but there's no in-place `str_push` or growable `str_buf` type.
- **No f-strings or format mini-language.** Build via
  `str_concat` + `int_to_str` / `rat_to_str` / `sex_to_str` /
  `bool_to_str`. Format syntax is a post-overloads feature.
- **Accessor door kept open.** Reads of `.ptr` / `.len` / `.cap` go
  through `StrsMixin._str_extract` in codegen (not raw field
  extract) so a future small-string optimization can add a
  discriminator branch without touching callers. Stdlib
  `str.tpu` still uses `s.ptr` / `s.len` as field access; if we
  switch layouts, those become accessor-dispatches transparently.
- **SSO (small-string optimization) deferred.** Design discussed
  during this pass; layout was kept 24-byte `{ptr, len, cap}`
  rather than the SSO variant to keep v0.1 simple. Switchable
  later without user-code breakage thanks to the accessor path.

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
10. ~~**Sum types**~~ — **done.** `seal Option<T> { Some(T), None }`,
    generic seals monomorphized like tablets, `match` as an expression
    with wildcard arms and exhaustiveness checking. Bidirectional
    expected-type threading so `None` works in return / call-arg /
    annotated-binding contexts. See `tests/test_sum.py` and
    `examples/omens.tpu`.
11. **Fn-as-value (no capture)** — **Next.** Higher-order functions
    without environment capture. Lets us pass `fn(i64) -> i64`
    values as params and store them in bindings; unlocks visitor
    patterns and minimal higher-order stdlib helpers. Short step
    toward closures.
12. **Sex/dish Phase 3c** — escape-analysis pass for rat-fallback
    specialization. Unblocks the future `--strict-dish` flag idea.
13. **Imports** — cleanup; pay when stdlib grows enough to need
    namespaces.

## Roadmap (agreed 2026-04-22)

Cross-cutting goal: incrementally fold pieces of the compiler into
Tuppu itself (self-hosting). Minimum viable self-host set: sum types
(AST shape), closures or fn-as-value (visitors), dynamic strings
(error messages), maps (symbol tables), file I/O.

Ranked by "unblocks most / lays groundwork for the self-host path":

| Feature                          | Cost    | Unlocks                                       | Notes                                                                        |
|----------------------------------|---------|-----------------------------------------------|------------------------------------------------------------------------------|
| ~~**Sum types + pattern matching**~~ | **done**  | Option, Result, ADTs, AST-shaped data         | Landed 2026-04-22 — see `tests/test_sum.py`.                          |
| **Fn-as-value (no capture)**     | ~2-3 h  | Higher-order functions, minimal visitors      | Cheap. Pointers-to-functions, no environment capture.                        |
| **Full closures (with capture)** | ~4-6 h  | Inline callbacks, state-carrying fns          | Needs a captured-env layout story.                                           |
| **Overloads (`__repr__`-style)** | ~2-3 h  | Generic `print`, user-extensible operators    | Type-dispatched name resolution. No trait system needed.                     |
| **Operator overloads**           | +1-2 h  | `a + b` for user types                        | Layers on the same dispatch mechanism as overloads above.                    |
| **Dynamic strings + slicing**    | ~4-6 h  | `s[i:j]`, concat, real manipulation           | Needs an arena-for-strings story (tablets-backed, auto-release).             |
| **Maps / hash tables**           | ~4-6 h  | Symbol tables, interning                      | Can be built in Tuppu on top of `tablets`; probably starts as `stdlib/map.tpu`. |
| **File I/O (read_line, etc.)**   | ~2-4 h  | Real programs, step toward self-hosting input | Libc wrappers via the same extern pattern as `write`/`fflush`.               |

Agreed order (approximate): ~~sum types~~ → **fn-as-value** → full
closures → overloads → operator overloads → dynamic strings → maps →
file I/O.

**Sum types — what shipped** (2026-04-22):

- `seal Name<T> { Variant, Variant(T), Variant(T, U), ... }` declares
  a tagged sum. Variant fields are positional (names elided); named
  variant fields are a possible follow-up.
- Runtime: `{ i8 tag, [N x i64] payload }` where N fits the widest
  variant's payload. Variant constructors alloca the seal, store tag,
  bitcast payload to the per-variant struct, fill the fields, and
  load. Nullary variants skip the payload writes.
- `match scrutinee { Pat => expr, ..., _ => expr, }` lowers to an
  LLVM `switch` on the tag. Exhaustiveness enforced unless a `_`
  arm is present; default block is `unreachable` when exhaustive.
- Patterns: `VariantName`, `VariantName(bind_or_underscore, ...)`,
  and `_`. **Flat patterns only** — no nesting, no guards, no or-
  patterns. (Those stay v0.2.)
- Generic seals use the same monomorphization machinery as generic
  tablets (`_get_monomorph_seal(name, arg_tys)`).
- **Bidirectional inference for nullary generic variants.** `_tc_expr`
  takes an optional `expected` that's threaded through bindings with
  type annotations, yield, fn-body tails, assignment RHS, and call
  args. So `step x: Option<i64> = None` and `fn f() -> Option<i64> {
  if cond { Some(1) } else { None } }` both work. A bare `step x =
  None` without context errors with a "cannot infer T" message.
- Typecheck: `TySeal` added alongside `TyStruct`; variant names live
  in a global `variant_lookup` (they must be globally unique in v0.1).
- Sidebands: `mono_variant_args` (id(node) → concrete type args) and
  `variant_of_node` (id(node) → (seal, variant, index)) so codegen
  knows exactly what to emit without re-running inference.

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
2. `.venv/bin/pytest` — expect 484 passing.
3. `git log --oneline -12` — recent timeline: sex Phase 3a/3b,
   struct field mutation, codegen.py split into mixins package,
   elif + did-you-mean, recursive tablets + wedge handles + auto-
   release + escape check, tablet/wedge/seal rename, minimal
   generics (+ stdlib/list.tpu rewritten to `List<T>`), sum types
   via `seal` + flat pattern `match` + bidirectional expected-type
   threading.
4. Read `SPEC.md` §4.5 (tablets), §4.6 (wedges), §14 (non-goals).
   SPEC.md does NOT yet describe `seal` / `match`; the source of
   truth is `tests/test_sum.py` and `examples/omens.tpu`. A spec
   update for sum types is queued.
5. **Agreed next task: fn-as-value (no capture).** Function values
   passed as params / stored in bindings / returned — no environment
   capture yet. Type `fn(i64) -> i64`. Unlocks visitor patterns over
   sums and is the smallest step toward full closures. After that:
   closures → overloads → operator overloads → dynamic strings →
   maps → file I/O. Self-hosting remains the multi-quarter goal.
6. `FUTURE_OPTIMIZATIONS.md` (gitignored) captures design sketches
   for a `--strict-dish` flag, the SEX 20→16 byte shrink, SIMD carry
   in sex_add, and other perf/language ideas. Don't forget on the
   next perf pass.
