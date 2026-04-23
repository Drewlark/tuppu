# Tuppu — what's next

Pickup notes. The Babylonian sexagesimal redesign is the active
front; imports and dynamic strings are queued behind it.

## Where we are (as of 2026-04-23)

- **v0.1 feature-complete** per SPEC.md — lexer, Pratt parser, type
  checker, LLVM codegen via llvmlite.
- Private repo: https://github.com/Drewlark/tuppu (branch `main`).
- **585 tests passing.**
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
  can express them). **Variadic `str_concat`** — any arity ≥ 2,
  emits one linear-time single-malloc join (sum lens, malloc,
  memcpy each part at a running offset), so
  `str_concat(h1, h2, h3, body)` is one allocation, not a nested
  chain. **`s + t` and `s += t` on str values** — binary `+` on
  two str operands lowers to the same emitter, so
  `acc += int_to_str(i)` reads naturally. (Still O(n²) if used in
  a loop — use the str_buf pattern for hot paths.)
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
- **Lvalue indexing into tablets** — `arr[n] = v`, `arr[n].f = v`,
  `arr[n].inner.f += v` all work. Runtime walks the chunk chain
  via a new `get_addr(t, n) -> *T` helper, bounds-checks (trap on
  OOB mirroring the read path), and hands back a slot pointer the
  existing Field-GEP machinery composes on top of. Parser's
  `_check_lvalue` was tightened accordingly: an Index rooted at
  an Ident is a valid assignment target alongside the existing
  variable / field-chain shapes.
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
  `bool` widens to `i8` for C-ABI stability. **User-tablet args
  pass by value** (LLVM handles the platform struct-arg ABI), and
  **`mut` struct args pass by pointer** (mirrors `mut tablets`;
  used for libc's `struct sockaddr *addr` shape). Struct returns
  aren't exposed yet (platform-dependent layouts). See
  `examples/tcp_bind.tpu` for a real-libc TCP `socket` + `bind`
  +`close` roundtrip with a user-declared `sockaddr_in`, and
  `tests/test_colophon.py` for the full grid.
- **`buffer[N]u8` — fixed-size byte buffer for FFI.** `mut buf:
  buffer[1024]u8` allocates on the stack, zero-inits, and bounds-
  checks both read and write. `.len` is the compile-time constant.
  Crosses the C ABI as a `T*` pointer decay — `colophon fn recv(fd,
  mut buf: buffer[1024]u8, n, flags)` Just Works. `buffer_to_str(buf,
  n)` copies the first `n` bytes into a fresh heap str (inclusive
  bounds check: `n == N` allowed, unlike indexing). Rejected in
  return types and struct fields (would dangle). See `tests/
  test_buffer.py`.
- **Tablets literal + variadic slice params.** `tablets[N]T {
  a, b, c }` builds a pre-populated tablets in one expression;
  `tablets[...]T` on the last fn param collects trailing call args
  into a synthesised tablets literal passed by pointer. Enables
  self-hosted variadic stdlib fns. `str_concat` has been migrated
  from compiler intrinsic to `stdlib/str.tpu` as a plain Tuppu fn.
  The binary `+` / `+=` operator on strs still uses the native
  single-malloc fast path. See `tests/test_variadic.py`.
- **Operator overloads via `gloss` keyword.** `gloss add(a: Vec,
  b: Vec) -> Vec { ... }` registers a Tuppu fn as the dispatch
  target for `+` on two Vecs. Internal mangling (`__gloss_add_
  Vec_Vec`) keeps the user's `fn add` namespace free — you can
  have both a plain `fn add` and a `gloss add` coexist. Supported
  ops: `add sub mul div mod` (binary arith), `eq` (auto-derives
  `!=`), `lt le gt ge` (separate, no `Ordering` type at v1),
  `neg not` (unary). At least one operand must be a user tablet
  or seal; primitive-primitive overloads are rejected. Did-you-
  mean warning fires when a user writes `fn add(T, T) -> T` where
  T is a user type — nudges them toward `gloss add` without
  rejecting. See `tests/test_gloss.py`.
- **Primitive-only fn pointers across colophon.** `colophon fn
  atexit(cb: fn())` and `colophon fn signal(sig: i32, handler:
  fn(i32)) -> fn(i32)` both type-check and run. Tuppu fn names
  pass straight through as LLVM function pointers (no wrapper,
  no marshaling — LLVM's default calling convention is
  C-compatible for primitive signatures). Callback signatures
  are restricted to `fn(prim, ...) -> prim` (int / bool / unit)
  at v0.1 — str/struct/wedge/nested-fn rejected at colophon
  decl. Returned C fn pointers can be bound, rebound via a
  later colophon call, or invoked directly from Tuppu (all
  three tested). See `tests/test_colophon.py::test_colophon_*fn*`.
- **If-as-statement relaxation.** When an `if` sits in statement
  position (a bare ExprStmt inside a block, not the block's tail
  and not assigned anywhere), its arms no longer need to unify.
  Fixes the recurring friction when one arm diverges via a
  colophon (`_exit`, `abort`) and the other returns a different
  shape, or when `push()` returns a `wedge` that nobody asked for.
  When the `if`'s value IS used (step binding, block tail, etc.),
  the usual arm-unification rule still applies. Elif chains inherit
  the flag — relaxing the outer `if` relaxes every nested
  `else if`. See `tests/test_typecheck.py::test_if_stmt_*`.
- **First-class function values (no capture)** — a bare fn name is
  a value of type `fn(params) -> ret`. Pass as arg, store in a
  binding, reassign through `mut`, return from a fn, hold in a
  struct field, and call through any of those. Codegen produces
  an LLVM function-pointer; calls through the pointer reuse the
  same cap=0 / struct-neutering marshaling as direct calls so
  there's no ownership escape via the indirect path. Colophons
  and generic fns can't be taken as values (no monomorphic
  address to hand out). Closures with environment capture remain
  a future pass. See `examples/higher_order.tpu`,
  `tests/test_fn_value.py`.
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

## 4. Fixed-size byte buffers — `buffer[N]u8`

**Goal.** Raise the ceiling on byte-level FFI without breaking
memory safety. User programs that talk to the kernel (the HTTP
server + Mandelbrot examples) currently invent ugly workarounds
like `tablet ByteSlot { b: u8 }` to get a single mutable byte
they can pass to `recv`. A fixed-size, stack-allocated,
bounds-checked byte buffer fixes this without reintroducing raw
pointer ops.

### Grammar / syntax

```
mut buf: buffer[1024]u8              // declaration + zero-init
buf[i] = 0 as u8                      // write
step b = buf[i]                       // read (bounds-checked)
step n = buf.len                      // i64 constant = N

colophon fn recv(fd: i32, mut buf: buffer[1024]u8, n: u64, flags: i32) -> i64
colophon fn send(fd: i32, buf: buffer[1024]u8, n: u64, flags: i32) -> i64
```

The `mut buffer[N]T` param shape is the byte-buffer analogue of
`mut tablets[N]T` — passes by pointer, callee reads/writes
through it. Non-mut `buffer[N]T` also passes by pointer at the C
ABI (array-to-pointer decay); our struct-by-value convention
doesn't apply.

### Why "buffer", not reintroduce `[N]u8`

The old `[N]u8` syntax was retired when `str` landed. Different
name avoids stale muscle memory. Also semantically distinct:
`str` is value-semantics with hidden heap, `buffer` is
stack-allocated and FFI-facing.

### Memory safety preservation

- Stack-allocated, lifetime = scope: no free semantics to leak.
- Zero-init on declaration (same as `mut x: SomeStruct`).
- Bounds-checked on `buf[i]` read and write (same
  `_emit_dynamic_bounds_trap` we already use for tablets/str).
- Returning `buffer[N]T` from a fn: **rejected at typecheck**
  (the slot would dangle). Same rule should apply to storing in
  a struct field initially — defer structs-of-buffer to a
  follow-up if anyone asks.
- FFI boundary risk (caller lying about `n`): identical to
  today's `str` FFI where `write(fd, s, 999)` with a 10-byte str
  already instructs the kernel to read past. Not a new hole.

### Typecheck

- New AST node `TypeBuffer(size: int, element: TypeExpr)`.
- New `Ty` node `TyBuffer(size: int, element: Ty)`.
- `TypeBuffer` resolves to `TyBuffer` in `_resolve_type`.
- Index `buf[i]`: element type, integer-index required (mirrors
  `str` / tablets indexing).
- `buf.len`: `I64`, value known at compile time.
- Return type `buffer[N]T`: error.
- `buffer[N]T` as struct field: error for now.
- Colophon FFI allow-list: accept `buffer[N]T` in parameters
  (both mut and non-mut).

### Codegen

- `_lower_type(TypeBuffer)` → `ir.ArrayType(elem, N)`.
- Binding: `alloca [N x elem]`, zero-init via
  `store [N x elem] zeroinitializer, [N x elem]* %slot`.
- Index read: `gep inbounds [N x i8], ptr %slot, i32 0, i64 %idx`
  + `load`, preceded by bounds trap against `N`.
- Index write: same GEP + `store` + bounds trap.
- `.len`: compile-time `i64 N`, no load.
- FFI call with buffer arg (mut or non-mut):
  `gep inbounds [N x T], ptr %slot, i32 0, i32 0` → `T*`, pass
  as `i8*`-decayed pointer (matches C calling convention).
- Colophon signature LLVM type for `buffer[N]T` → `T*` (pointer
  to element type).

### Conversions

- `buffer_to_str(buf: buffer[N]u8, n: i64) -> str` — new
  intrinsic. Allocates heap `n+1`, memcpies `n` bytes, NUL-
  terminates, returns heap-owned str. Parallel to
  `bytes_to_str(tablets[N]u8)` but takes an explicit length
  because buffers don't track "used bytes".
- `str_to_buffer(s: str, mut buf: buffer[N]u8) -> i64` — stdlib
  helper. memcpies `min(s.len, N)` bytes, returns bytes-copied.
  Can be Tuppu-level on top of indexing.

### Files to touch

- `src/tuppu/ast.py` — `TypeBuffer` dataclass.
- `src/tuppu/parser.py` — recognise `buffer[N]T` in `parse_type`.
- `src/tuppu/lexer.py` — no new keyword (reuse IDENT or add
  `BUFFER` token).
- `src/tuppu/typecheck.py` — `TyBuffer`, `_resolve_type` branch,
  index / field handling, FFI allow-list, return-type rejection.
- `src/tuppu/codegen/__init__.py` — `_lower_type` branch, binding
  path with zero-init, index read/write with bounds trap, FFI
  arg passing, `buffer_to_str` intrinsic + dispatch.
- `stdlib/str.tpu` — optional `str_to_buffer` helper.
- `tests/test_buffer.py` — new. Basic use, bounds trap, FFI
  roundtrip with a mock libc `memset`-style fn, return-type
  rejection.
- `examples/tcp_echo.tpu` or update `tcp_bind.tpu` to use
  `buffer[1024]u8` for recv.

### Payoff: rewriting `read_request` from the user's HTTP server

```tuppu
fn read_request(cfd: i32) {
  mut buf: buffer[1024]u8
  mut run: i64 = 0
  mut done: bool = false
  while !done {
    step n: i64 = recv(cfd, buf, 1024 as u64, 0 as i32)
    if n <= 0 { done = true }
    else {
      mut i: i64 = 0
      while i < n {
        step b: u8 = buf[i]
        // state machine on \r\n\r\n ...
        i = i + 1
      }
    }
    if run == 4 { done = true }
  }
}
```

No more single-byte `ByteSlot` sink, no per-byte syscall.


## 5. Tablets-literal syntax + variadic slice params

**Goal.** Let users write `str_concat(a, b, c, d)` that reaches a
Tuppu-level fn (not a compiler intrinsic) — the unlock that lets
us migrate `str_concat`, future `print_fmt`, and any "take a
bunch of the same thing" API into stdlib. Unlocks self-hosting
of more of the stdlib and makes the language feel grown-up.

### Two pieces, land together

**Piece A: tablets-literal.** Syntax for constructing a
pre-populated tablets value in one expression.

```
step nums: tablets[4]i64 = tablets[4]i64 { 1, 2, 3, 4 }
step words = tablets[4]str { "alpha", "beta", "gamma", "delta" }
```

Without the explicit type annotation, the type can be inferred
if the chunk size is known contextually — otherwise require the
explicit `tablets[N]T { ... }` form. Keep inference minimal in
v1.

**Piece B: variadic param.** A single `tablets[...]T` param
(must be last) collects the trailing args at the call site into
a tablets literal.

```
fn str_concat(parts: tablets[...]str) -> str {
  mut buf: tablets[64]u8
  for p in parts {
    str_buf_append(buf, p)
  }
  bytes_to_str(buf)
}

// Call site desugar:
str_concat("a", "b", "c")
// becomes
str_concat(tablets[CANONICAL_N]str { "a", "b", "c" })
```

`CANONICAL_N` is a compiler-chosen chunk size (probably 16 — big
enough to hold typical variadic calls in one chunk, not so big
we waste memory on small calls).

Splat syntax (`str_concat(existing_tablets...)`) is deferred —
orthogonal, easy to add later.

### Why this shape (not C varargs, not ownership tricks)

Slice-passing is type-safe at every call site, composes with
our existing iteration (`for p in parts`), needs no new runtime
primitives beyond tablets literal construction. The tablets
runtime struct `{ head, tail, len }` already doubles as a slice
header — we just need a way to build one without `.push` in a
loop.

### Typecheck

- New AST `TypeVariadicTablets(element: TypeExpr)` parsed from
  `tablets[...]T`.
- New AST `TabletsLit(size: int, element: TypeExpr | None,
  fields: list[Expr])`.
- `TyFn` gains `is_variadic: bool`; the last param's type
  carries an "any N" flag.
- Call typecheck: if callee is variadic, split args into
  (fixed, variadic tail). Fixed args typecheck against named
  params. Variadic tail typechecks against the element type and
  gets wrapped into a synthetic `TabletsLit` node before
  codegen.
- In fn bodies, a variadic-typed param is treated as a regular
  `tablets[CANONICAL_N]T` for all indexing / iteration / method
  dispatch purposes.

### Codegen

- Tablets-literal: alloca a tablets header, push each element
  one by one via the existing `__tuppu_tbls_<suffix>_push`.
  Register for cleanup per the normal rules (mut binding
  registers; rvalue in a call-arg position registers an
  anonymous cleanup). Optimization for later: fast path for
  known-small literals that fits in one chunk (single
  `malloc(sizeof(node))`, write slots directly, bump `used` and
  `len`).
- Variadic call: emit the synthetic TabletsLit at the call site.
  Pass by the mut-tablets-param convention (by pointer) so the
  callee sees the same storage.
- Cleanup on the synthetic literal: anonymous cleanup in the
  caller's frame, same shape as other Call-rvalue args. Chunks
  get released at caller scope exit.

### Migration: self-hosted `str_concat`

Once both pieces land, delete the compiler intrinsic and move
the fn into `stdlib/str.tpu`:

```
// stdlib/str.tpu
fn str_concat(parts: tablets[...]str) -> str {
  mut buf: tablets[64]u8
  for p in parts {
    str_buf_append(buf, p)
  }
  bytes_to_str(buf)
}
```

Same API, no compiler knowledge needed. The binary-plus operator
`a + b` on strs either stays a compiler shortcut (2-arg fast
path) or lowers to a 2-element `TabletsLit` followed by a call
— performance-wise the fast path is worth keeping.

### Files to touch

- `src/tuppu/ast.py` — `TypeVariadicTablets`, `TabletsLit`.
- `src/tuppu/parser.py` — parse `tablets[...]T` in types; parse
  `tablets[N]T { a, b, c }` literal (add to `parse_prefix`
  discrimination alongside struct lit).
- `src/tuppu/typecheck.py` — `TyFn.is_variadic`,
  `_tc_call`'s variadic branch, `TabletsLit` typecheck.
- `src/tuppu/codegen/__init__.py` — `_gen_tablets_lit`, call-
  site variadic argument collection.
- `stdlib/str.tpu` — rewrite `str_concat` in Tuppu; delete
  compiler intrinsic in follow-up commit once parity is proven.
- `tests/test_variadic.py` — new. Arity 0, 1, 2+, mixed types
  error, passing an existing tablets via splat (deferred /
  error for now), self-hosted `str_concat` parity with old
  intrinsic.

### Edge cases to watch

- Zero-arity variadic: `str_concat()` with no args. Call-site
  builds an empty tablets literal. Callee handles empty
  iteration. Output: empty str.
- Variadic + expected-type threading for literals: a
  `None`-shaped nullary variant in a variadic arg list needs
  the per-arg expected type to pin `T`. Same bidirectional
  machinery we use for seal inference already applies.
- Recursion through variadic: no special handling — generic fn
  rules apply if the variadic is polymorphic (`fn first<T>(xs:
  tablets[...]T) -> T`).
- Nested variadic calls (`f(g(a, b), c)`): ownership of the
  inner tablets literal is handled by the Call-rvalue cleanup
  registration — same as other heap-producing rvalues.


## 6. Fn pointers across colophon (primitives-only)

**Goal.** Let users hand a Tuppu fn to libc callbacks (`signal`,
`atexit`, `qsort`) and receive C fn pointers back (`signal`'s
return value). Unlocks a big chunk of real programs — signal
handlers for SIGCHLD-reaping in the HTTP-server example, atexit
for cleanup, qsort for any sorting workload — without committing
to the full closure machinery or the "marshaled callback" design
that would be required for `str`/`struct`/`wedge` across the C
boundary.

### The rule: primitives-only in callback signatures

A `colophon fn` may contain `fn(prim, prim, ...) -> prim` (or
`fn(prims) -> void`) in its parameter or return types, where
`prim` is any integer type (`i8..i64`, `u8..u64`) or `bool`. Any
other type — `str`, user tablets, `tablets[N]T`, `wedge T`,
`buffer[N]T`, nested `fn` — is **rejected at colophon
declaration time** with a clear "callback signatures are
primitives-only for now" error.

This is a semantic restriction, not a safety one: the C side
invokes the fn with raw register-passing conventions, so any
Tuppu-level marshaling (cap=0 str neutering, struct-field zeroing)
would never fire. Primitives already round-trip by the C-ABI
default, which is also LLVM's default calling convention for
Tuppu fns — so no wrapper code is needed at either end.

### Grammar / syntax

Already parses (we have `TypeFn` from first-class fn values).
This proposal is purely a typecheck allow-list extension —
no grammar changes.

```
colophon fn signal(signum: i32, handler: fn(i32)) -> fn(i32)
colophon fn atexit(cb: fn()) -> i32

fn on_sigchld(sig: i32) { /* reap zombies */ }
fn main() -> i32 {
  step old = signal(17 as i32, on_sigchld)   // SIGCHLD
  0
}
```

### Typecheck

- `_register_colophon`'s FFI allow-list grows: `TyFn` accepted
  iff all params are primitive-or-bool AND ret is primitive-or-
  bool-or-unit. Nested `TyFn` inside a `TyFn` signature rejected
  at v0.1 (keeps the rule dead simple; can relax later).
- Passing a Tuppu fn name at the call site: the existing
  first-class-fn-value path types it as `TyFn(params, ret)`.
  `_coerces_to(TyFn, TyFn)` check verifies signature match —
  already almost-working because colophons can't be passed as
  values (we keep that rejection), but regular fns can.
- Receiving a `fn(...)` return from a colophon: the return value
  is typed as the declared `TyFn`, which can then be bound,
  passed to another colophon, or invoked (the existing
  `_tc_fn_value_call` already handles this).

### Codegen

- `_colophon_c_ty`: recognize `ir.FunctionType.as_pointer()` and
  pass through unchanged — LLVM lowers to the platform's fn-
  pointer ABI. C's `void (*)(int)` becomes `void (i32)*` on our
  side; identical representation.
- `_gen_colophon_call`: when a fn-value arg is emitted, the
  existing `_gen_expr(Ident(fn_name))` already yields the LLVM
  function pointer — no marshaling wrapper needed. No cleanup,
  no neutering; primitives go straight across.
- `_declare_colophon`: lower `fn(prim) -> prim` param types
  directly. No special case.

### Edge cases

- **Calling a received C fn pointer**: `old(1 as i32)` goes
  through `_gen_fn_value_call`, which emits an indirect call.
  Same path as a Tuppu fn-value call. The cleanup/neutering
  logic in that path is currently keyed on element types, so
  primitives skip it automatically.
- **Passing `lost` or `fn`-valued null**: defer. `lost` is
  handle-typed; we'd need a fn-type null literal. Users who
  need "no handler" can define a Tuppu no-op fn and pass that.
- **Var-args callbacks** (`printf`-style): rejected. Callbacks
  with `...` C-level varargs aren't common and our FFI doesn't
  model them anywhere.

### Files to touch

- `src/tuppu/typecheck.py` — `_register_colophon`'s allow-list:
  accept `TyFn` with primitive-only signature; reject nested /
  str / struct / wedge inside the TyFn.
- `src/tuppu/codegen/__init__.py` — `_colophon_c_ty` pass-through
  for fn-pointer types; signal to the signature assembly that no
  marshaling wrapper is needed.
- `tests/test_colophon_fnptr.py` — new. `signal` roundtrip, the
  "SIGCHLD reaper" use case, `atexit`, rejection for
  `fn(str) -> i32` and `fn() -> Row`, receiving-and-calling a
  returned fn pointer.

### Payoff

The HTTP server's zombie-reaping story: `signal(SIGCHLD, SIG_IGN
or reaper_fn)`. The `atexit` path for clean shutdown. `qsort`
once we also add a way to pass mutable storage (closely coupled
to pointer-to-primitive access; out of scope for this proposal
but unblocked by primitives-only FFI). ~50 lines of compiler
code. Strict subset of full closures — doesn't preclude closures
later.


## 7. Todo backlog (ideas floated, not specced)

Rough list of things we've discussed but haven't designed in
detail. Ordered by gut priority — sanity-check against user
pressure before committing to one.

### High-leverage, small-medium effort

- **Print dispatch via `gloss show`.** Extend the gloss table
  with `show(x: T) -> str`; `println(x)` on a user type looks it
  up and calls it. Makes user tablets first-class in `println`
  without forcing users to write `println(v.x, ", ", v.y)` by
  hand. Probably 30–50 lines — dispatch mirrors the existing
  gloss machinery, `_emit_one_print` adds a branch that checks
  the sideband and calls the show fn before printing the
  resulting str. Compounds with the gloss work we just did.

- **File I/O wrappers in stdlib.** A new `stdlib/fs.tpu` that
  wraps `open` / `read` / `write` / `close` / `stat` via
  colophons + `buffer[N]u8`. Zero compiler work — pure
  proof-of-capability. `read_file(path) -> str`,
  `write_file(path, content)`, `read_line()` for stdin. Unblocks
  a class of real programs (config readers, log processors,
  one-shot scripts). Adjacent: `stdin` read wrapper for REPLs /
  interactive tools.

- **Maps / hash tables in Tuppu.** `stdlib/map.tpu` with a
  robin-hood or open-addressed table, generic over `<K, V>`.
  Unlocks symbol tables (so we can self-host the typechecker
  one day), interning, lookup tables keyed by str. Medium-large
  effort — needs a hashing story (probably start with u64 keys
  only, add str hashing after) and a strategy for deletions.
  All in pure Tuppu on top of tablets.

### Nice-to-have, smaller effort

- **Default `gloss eq` / `gloss ord` for more built-ins.** We
  added `gloss eq` for `str`; same treatment could go to
  `rat`, `dish` if they're not already handled by the primitive
  path. Audit which types support `==` today.

- **Type conversion dispatch.** User mentioned Python-style
  `__repr__` / a general "how do I turn X into Y?" mechanism.
  Could be `gloss to_str(x: T) -> str` (overlaps with `show`
  above) or a broader `cast` dispatch that makes `v as Other`
  callable for user types. Deferred — not a pressing need
  while `show` + `to_str` cover the common case.

### Bigger lifts, lower immediate demand

- **Closures with environment capture.** The `gloss` + fn-as-
  value + primitive-fn-pointer-across-colophon combo covers a
  lot of what closures would. Real need hasn't surfaced yet.
  Full closures need a captured-env layout story (stack vs heap,
  lifetime/escape analysis, or a Rust-style `move` keyword).
  Defer until a user says "I'm blocked on closures."

- **Imports.** Sketched in §2 above. Stdlib auto-include still
  works fine; pay this cost when stdlib outgrows the "one big
  namespace" model. Probably when maps + fs + net wrappers all
  land.

- **`dish` 16-byte refactor.** Sketched in FUTURE_OPTIMIZATIONS.md.
  Narrow perf win; don't do it until a profile demands it.

- **Sex Phase 3c — escape-analysis rat-fallback.** Compile-time
  pass that detects "this dish never leaks its digit form" and
  lowers it to rat directly. Sketched in §3c above. Medium-large;
  useful for hot sexagesimal arithmetic, not broadly critical.

- **SPEC.md catch-up.** The spec hasn't covered `seal`,
  `colophon`, fn-as-value, `buffer`, variadic, gloss, or the
  full str ownership story. Overdue but purely editorial — cut
  a half-day when the feature surface stabilizes.

### Known friction spots worth a look

- **Struct returns across colophon.** Currently rejected — we
  punted on the platform-specific ABI story. Some libc fns
  would benefit (e.g., `gettimeofday`-shaped). Probably narrow
  and low-demand.

- **`waitpid`-style int-output params.** Need pointer-to-primitive
  across colophon. Would go well with the (deferred) primitive
  pointer story.

- **`SO_RCVTIMEO` / other socket options.** Needs
  `setsockopt(fd, level, name, *void, len)` — opaque pointer
  arg. Blocked on having some pointer-to-primitive story.


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
2. `.venv/bin/pytest` — expect 585 passing.
3. `git log --oneline -15` — recent timeline: sum types + generic
   monomorphization, str ownership sentinel on fn args, slicing,
   str_buf pattern via tablets-backed byte buffer + `bytes_to_str`,
   ownership-transfer-on-tail-return, struct-field auto-release,
   step-borrow rule + mut-struct-param neutering, lvalue indexing
   (`arr[n].f = v`), colophon typed FFI (primitive / str / user-
   tablet args, `mut` structs by pointer for sockaddr_in), real-
   libc TCP bind demo, fn-as-value (no capture), variadic
   `str_concat` + `s + t` / `s += t` operator, newline-inside-
   `{}` is a statement terminator.
4. Read `SPEC.md` §4.5 (tablets), §4.6 (wedges), §14 (non-goals).
   SPEC.md does NOT yet describe `seal` / `match`, `colophon`,
   fn-as-value, the full str ownership story, or lvalue indexing;
   source of truth is the test files + this document. A SPEC
   catch-up pass is overdue.
5. **Agreed next tasks (in order):**
   - ~~**§4. `buffer[N]u8`**~~ — done 2026-04-23.
   - ~~**§5. Tablets literal + variadic slice params**~~ — done
     2026-04-23. `str_concat` is now a stdlib fn.
   - ~~**If-as-statement relaxation**~~ — done 2026-04-23.
   After those: full closures (with capture) → overloads →
   operator overloads → maps → file I/O.
   - ~~**§6. Fn pointers across colophon (primitives-only)**~~ —
     done 2026-04-23. `signal` / `atexit` work; callback sigs must
     be `fn(prim, ...) -> prim`.
   - ~~**Operator overloads via `gloss`**~~ — done 2026-04-23.
     `gloss add(a: Vec, b: Vec) -> Vec`. See `tests/test_gloss.py`.
6. `FUTURE_OPTIMIZATIONS.md` (gitignored) captures design sketches
   for a `--strict-dish` flag, the SEX 20→16 byte shrink, SIMD carry
   in sex_add, and other perf/language ideas. Don't forget on the
   next perf pass.
