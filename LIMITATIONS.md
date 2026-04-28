# Limitations

A consolidated list of things Tuppu doesn't have yet that a serious
language of this type ought to. The point of this file is **visibility**:
nothing on this list should be excused with "we're early" anymore —
either it's intentional and documented here, or it's a bug and gets
fixed. New limitations land here at the same time they land in code,
so the list is the source of truth. When something is fixed, delete
the bullet (and add a CHANGELOG entry).

Format: each section groups by area. Bullets prefix with severity:

- **[blocker]** — actively prevents real programs from being written.
- **[gap]** — missing feature with a real workflow it would unblock.
- **[polish]** — nice-to-have that doesn't block anything load-bearing.

## Memory model / GC

- **[blocker]** No LLVM optimization passes can run on the emitted
  IR. The shadow-stack rooting calls our GC depends on
  (`__tuppu_gc_push_root(slot, ...)`) look like opaque extern calls
  to LLVM; standard passes (SROA, mem2reg, instcombine) promote
  rooted allocas to SSA without realizing the GC needs the address
  to persist. Even `tail_call_elimination` desyncs the push/pop
  accounting on recursive calls. Confirmed empirically: any opt
  level above `-O0` SIGSEGVs the lua interp. The fix is to migrate
  to LLVM's first-class GC framework — `gc "shadow-stack"` fn
  attribute + `@llvm.gcroot` intrinsics — so optimizations become
  GC-aware. Until then, every Tuppu binary ships with whatever the
  IR generator emits, no inlining, no DCE, no register promotion.
  This leaves a lot of perf on the table; on hot recursive loops
  (lua interp, fib) it's the single biggest unlocked win available.
- **[gap]** No region-allocator for non-escaping `mut tablets` —
  every push goes through the GC arena even when the lifetime is
  trivially the function body. A region-allocator for the common
  case would cut allocation pressure on hot paths (parser arenas,
  per-frame scratch buffers).
- **[gap]** No root-elision pass. `wedge T` walks already skip the
  shadow stack, but `step` bindings of cleanup-bearing values are
  unconditionally rooted even when their use is provably non-allocating.
- **[polish]** GC arena never gives memory back to the OS. Long-
  running programs grow until exit. Mark-sweep can compact + return
  pages; we don't.

## Type system

- **[blocker]** No numeric generics. Container chunk size is hardcoded
  per stdlib type (`tablets[16]Node<T>` in `list.tpu`,
  `tablets[64]T` in `vec.tpu`, `tablets[64]str/T` in `map.tpu`).
  A user can't write `Vec<T, 16>` vs `Vec<T, 1024>`.
- (Cross-module same-name decls coexist via module-prefix LLVM
  mangling — see `tests/test_modules.py::test_two_modules_can_each_declare_same_fn_name`
  and `::test_two_modules_can_each_declare_same_tablet_name`.)
- **[gap]** Pattern matching is flat only. No nested patterns
  (`Some(Circle(r))`), no guards (`Some(x) if x > 0`), no or-patterns
  (`Some(1) | Some(2)`). Exhaustiveness for nested patterns is
  subtle and the design lift hasn't happened.
- **[gap]** Variant fields are positional only. `Circle(rat)` works,
  `Circle(radius: rat)` doesn't parse.
- **[gap]** `f32` / `f64` are reserved keywords with no codegen.
  Casts to / from float types raise a clear "not yet supported"
  error rather than silently doing the wrong thing, but float
  arithmetic is genuinely missing.
- **[gap]** Bare array types (`[N]T`) aren't supported as a type
  expression. Use `tablets[N]T` for growable, `buffer[N]u8` for
  stack-lifetime byte-only.
- **[gap]** No traits / typeclasses. Generic constraints are implicit
  ("monomorphization will fail if `T` doesn't support `==`"), with
  errors deferred to instantiation time. Real bounded generics need
  a constraint syntax.

## Runtime / FFI

- **[gap]** Buffers (`buffer[N]T`) are u8-only. Lifting requires an
  ownership story for struct / heap-bearing element types inside a
  stack-lifetime container — not yet decided.
- **[gap]** Buffers can't be struct fields (stack lifetime would
  outlive struct field accessors). Adding requires the same
  decision.
- **[gap]** Raw pointers (`*T`) can be held and passed but have no
  user-side dereference, arithmetic, or construction syntax. Adding
  requires an unsafe-block decision (which functions can do unsafe
  work, what does the boundary look like).
- **[gap]** Colophon callbacks are primitives-only. `fn(prim, ...) -> prim`
  signatures cross the FFI boundary; `str` / struct / wedge / nested
  fn arguments are rejected because we have no marshaling story for
  them.
- **[gap]** Closures don't capture environment. `fn` values pass
  function pointers only; lambdas with captured variables are
  pending.

## Strings

- **[gap]** `str_slice` always copies. Zero-copy slice views need a
  lifetime story we don't have.
- **[gap]** No mutable / growable string type. `str` is value-
  semantics-immutable; string building goes through repeated
  `str_concat` allocations or a `tablets[N]u8` buffer that gets
  flattened at the end.
- **[gap]** No format mini-language / f-strings. Build via
  `str_concat` + `int_to_str` / `rat_to_str` etc.
- **[polish]** No SSO (small-string optimization). `str` is always
  `{ptr, len, cap}` (24 bytes). Accessor codegen leaves the door
  open to switch to a tagged variant later without breaking callers.

## Containers (stdlib)

- **[gap]** `Map<T>` is linear-scan (O(n) lookup). Hash-based
  variant pending — needs numeric generics for bucket count + a
  stdlib hash function.
- **[gap]** `Map<T>` can't remove entries. Tablets are append-only
  at runtime; a real `map_delete` needs tombstone slots and
  compaction logic.
- **[gap]** `Vec<T>` can't pop. Tablets are append-only at runtime;
  pop would need the tail chunk to track holes or a separate
  shrink path.
- **[gap]** No `Vec<T>` map / filter / fold helpers. They need
  generic fn values across two type parameters which works in
  principle but isn't tested; once verified, the helpers are
  trivial.

## Modules / packaging

- **[gap]** No package manager / external dependency story. Cargo /
  npm / pip equivalent doesn't exist. Project-local modules under
  `src/` only. Vendoring is fine.
- **[gap]** No conditional compilation. No `cfg` / feature flags
  / target-specific code. Everything compiles for the host triple.
- **[gap]** No re-exports. `import x.y` brings names into the
  importer's scope literally; if downstream consumers need to see
  `x.y`'s names they import them directly. `pub use x.y` /
  `export from` is the natural follow-up.
- (Cross-module same-name fns and tablets coexist via the LLVM
  `__M_<mod>__<short>` mangle — module-prefix mangling is on for
  `fn`, `tablet`, and `seal` decls. Duplicate-name constraints now
  apply only within a single module.)
- **[gap]** Qualified-name struct literals. `mod.Tablet { ... }` in
  expression position doesn't parse — the struct-lit parser doesn't
  see the dotted form as a type-position name. Wildcard `import mod`
  brings the short name into local scope, so `Tablet { ... }`
  works directly. Type-position annotations (`step x: mod.Tablet`)
  do support the qualified form.

## Tooling

- **[gap]** No language server. Editor integrations rely on
  `./tuppu run` / `./tuppu build` and reading errors out of stderr.
- **[gap]** No formatter. Tests check `pytest` — the language has
  no `gofmt` equivalent.
- **[gap]** No incremental compilation. Every build re-typechecks
  + re-codegens the whole program (stdlib included).
- **[gap]** Lex / parse errors render with source context (line +
  caret pointer), but typecheck and codegen errors don't yet — they
  fall back to bare `line:col: message` format because the AST doesn't
  carry the source-file label down to those passes. Wiring the label
  through (either as a per-decl attribute or via a checker-side label
  stack) would unify the rendering.
- **[polish]** No debugger integration. Compiled binaries have
  basic LLVM debug info but no Tuppu-aware lldb / gdb pretty-
  printers.

## Standard library coverage

- **[gap]** No `read_line() -> str`. `read_int` exists; reading
  whitespace-delimited or newline-delimited strings doesn't.
- **[gap]** No file I/O. No `open` / `read` / `write` outside of
  what `print` / `println` provide via libc.
- **[gap]** No collections beyond `Vec<T>` / `Map<T>` / `Node<T>`.
  No set, no priority queue, no deque.
- **[gap]** No date / time. `clock_gettime` would route through
  colophon but no wrapper exists.
- **[gap]** No regex.
- **[gap]** No JSON / serialization.

## Spec / docs gaps

- **[gap]** SPEC.md doesn't yet cover `seal` / `match`. Source of
  truth is `tests/test_sum.py` and `examples/omens.tpu`.
- **[gap]** SPEC.md predates the GC migration. Sections describing
  the "tablets memory model" still talk about explicit `release`
  even though the GC handles it now.
- **[polish]** No prose-level "how to write idiomatic Tuppu" guide.
  README + examples cover the surface; deeper patterns (when to
  use `wedge` vs by-value, when to use `step` vs `mut`) are
  implicit.

---

If you're working on a feature and you find a limitation that's
actually blocking your work, **promote the bullet to a real task,
file a focused PR, and delete it from this list**. Don't add
workarounds that pretend the limitation isn't there — that's
exactly the failure mode CONTRIBUTING.md calls out.
