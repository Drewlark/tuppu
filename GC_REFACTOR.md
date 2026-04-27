# GC refactor — atomic-transition plan

Target: one session. Bring the `gc-migration` branch from its current
partial state (667/667 normal, 16/667 stress failures, duplicated
rooting code scattered across composite-site callers) to a clean
state where:

- There is exactly ONE site where a cleanup-bearing SSA value gets
  spilled to a shadow-stack-rooted slot.
- The per-site `clone + maybe-root` duplication is gone.
- The monolithic `codegen/__init__.py` is split along cohesive
  concerns.
- The freeze / escape correctness machinery in `typecheck.py` is
  deleted (GC is the safety net).
- `stress` mode passes the full suite.

This file is the plan to be mechanically executed next session. It
is NOT a scratchpad. Changes made while executing go into the file,
not replace it.

---

## Context — why this is necessary

The partial migration revealed that shadow-stack precise GC needs
one invariant: every cleanup-bearing value held across any
allocating call must be on the shadow stack at the moment the
collector runs. Our current codegen violates that invariant at
dozens of sites because cleanup-bearing values flow through SSA
between evaluation and consumption, and each consumer "fixes it up"
locally with its own spill-and-root dance. 16 stress failures
remain because some consumers weren't hooked. Meanwhile each
hooked consumer carries a near-identical 10-line block.

The right answer is a chokepoint: spill+root at the point of
PRODUCTION (calls / clones / binary-op results), not at each point
of CONSUMPTION. Consumers then don't need rooting logic at all —
they just read SSA or load from the rooted slot, and GC sees the
live value via its slot.

Simultaneously, `codegen/__init__.py` has grown to 4,500+ lines
through this same pattern of "add the new thing to the main file."
A clean split makes the chokepoint plus the freeze-machinery
deletion tractable without stepping on unrelated concerns.

---

## Type model — exhaustive enumeration

For each type: layout, where it lives, what GC does with it, what
needs a type descriptor, what the chokepoint does.

### Scalars — `i8`..`i64`, `u8`..`u64`, `bool`

- **Layout.** LLVM int width as-is.
- **Lives.** SSA / stack.
- **GC.** Untracked. No ptrs inside.
- **Type descriptor.** None.
- **Chokepoint.** Pass-through. No spill.

### `rat`, `dish` / `sex`

- **Layout.** `{i64 num, i64 den}` — 16 bytes.
- **Lives.** SSA / stack (by-value).
- **GC.** Untracked. Two scalar fields, no ptrs.
- **Type descriptor.** None.
- **Chokepoint.** Pass-through.

### `str`

- **Layout.** `{i8* ptr, i64 len, i64 cap}` — 24 bytes. `ptr`
  at offset 0.
- **Lives.** The STRUCT lives SSA / stack. The BYTES behind `ptr`
  live on the GC heap (or in an immortal global for string literals,
  cap=0).
- **GC.** The ptr field is traced. Literals are skipped by
  `mark_ptr` via the magic-header check (the address isn't a
  GC-owned object).
- **Type descriptor.** `__tuppu_str` with `ptr_offsets=[0]`.
- **Chokepoint.** Any SSA `str` value produced by a call, concat,
  slice, clone, or other allocating op gets spilled to a rooted
  slot.

### `tablets[N]T`

- **Layout of the value.** `{*Node head, *Node tail, i64 len}` —
  24 bytes. head at 0, tail at 8.
- **Layout of a `Node` (chunk).** `{[N x T] items, i64 used, *Node next}`.
  Allocated via `__tuppu_gc_alloc(size, &chunk_desc)` so the
  collector can trace it.
- **Lives.** Value: SSA / stack. Chunks: GC heap.
- **GC.** Value descriptor traces `[head, tail]`. Marking the head
  chunk follows the `next` ptr via the chunk's descriptor and marks
  each slot's ptr fields.
- **Type descriptors.** Two per `(N, T)`:
  - `__tuppu_tbls_{T}_{N}` — value desc, `ptr_offsets=[0, 8]`.
  - `__tuppu_chunk_{T}_{N}` — chunk desc, offsets = each slot's
    own ptr offsets shifted by slot index × sizeof(T), plus the
    `next` offset at `N*sizeof(T) + 8`.
- **Chokepoint.** The tablets VALUE is usually a named binding or a
  literal — both sites root it already. Chunks are GC-allocated
  with the typed allocator, so their reachability falls out of the
  chunk descriptor.

### `buffer[N]T`

- **Layout.** `[N x T]` — fixed-size array, stack-allocated.
- **Lives.** Stack only. Typecheck rejects buffer-typed return
  values and struct fields.
- **GC.** If `T` has ptrs, those ptrs need tracing when a buffer
  local is in scope. Under current design we root the buffer
  binding with a descriptor that lists each slot's ptr offsets.
  For `buffer[N]u8` (the common case), no ptrs, no descriptor.
- **Type descriptor.** `__tuppu_buffer_{T}_{N}` only when `T` has
  traceable ptrs. For `buffer[1024]u8`: none.
- **Chokepoint.** Pass-through. Buffers don't flow as rvalues.

### `wedge T` (handle)

- **Layout.** A single `*Node` pointing at a specific slot inside
  some tablets' chunk.
- **Lives.** SSA / stack. Non-owning.
- **GC.** Tracing a wedge would mark its target chunk. Since
  wedges are non-owning handles INTO a tablets the caller already
  owns, the tablets' own root covers the chunks. A wedge does NOT
  need to be a root in its own right — its validity is scope-
  governed (we forbid wedge escape via static analysis). Keep the
  escape rule as an optimization, not a safety requirement.
- **Type descriptor.** None (we don't trace wedges; they piggyback
  on the tablets' reachability).
- **Chokepoint.** Pass-through.

### `lost`

- **Layout.** Null pointer typed as `wedge T`.
- **GC.** Nothing. Literal null.
- **Chokepoint.** Pass-through.

### `*T` raw pointer

- **Layout.** Single pointer. Type-only — no construction or
  deref syntax.
- **Lives.** SSA / stack.
- **GC.** Untracked. Intentional escape hatch for FFI.
- **Chokepoint.** Pass-through.

### User structs (`tablet Name { ... }`)

- **Layout.** Packed fields in declaration order. Each field's type
  determines its sub-layout.
- **Lives.** Value by default (SSA / stack). Heap allocation
  happens only via tablets chunks (a struct element of a chunk).
- **GC.** Recursive type descriptor: for each field, compose its
  own ptr offsets at the field's byte offset. Leaves
  (scalars / wedges / fn-pointers) contribute nothing.
- **Type descriptor.** `__tuppu_struct_{Name}` with composed
  `ptr_offsets`.
- **Chokepoint.** A struct SSA value produced by a StructLit /
  Call that is cleanup-bearing gets spilled.

### User seals (`seal Name { Variant1(T), Variant2, ... }`)

- **Layout.** `{i8 tag, [K x i64] payload}` where K = ceil(widest_variant_bytes / 8).
  Payload is opaque bytes whose interpretation depends on tag.
- **Lives.** Value by default.
- **GC.** This is the tricky one. Payload interpretation is
  tag-dependent. A flat `ptr_offsets` can't express "if tag==0,
  trace these offsets; if tag==1, those." **Solution: custom
  `trace_fn`.** Codegen emits a per-seal trace fn that:
    1. Reads the tag byte.
    2. Dispatches to variant-specific tracing.
    3. For each variant with cleanup-bearing payload fields, marks
       those ptrs relative to the payload base.
    4. No-op for nullary variants.
  The trace_fn field on `tuppu_type_t` already exists — it's used
  here.
- **Type descriptor.** `__tuppu_seal_{Name}` with `trace_fn` set.
- **Chokepoint.** A seal SSA value produced by a variant ctor
  (which is a `Call`) gets spilled.

### Fn values (`fn(T1, T2) -> U`)

- **Layout.** Single code pointer.
- **Lives.** SSA / stack.
- **GC.** No environment capture today. Code pointers point at
  static globals (emitted fns). Not traceable / not traced.
- **Type descriptor.** None.
- **Chokepoint.** Pass-through.

---

## Composition cases worth spelling out

| Shape | What GC does |
|---|---|
| `struct { key: str, count: i64 }` | struct desc: `ptr_offsets=[0]` (key's ptr at offset 0, shifted by 0). |
| `struct { name: str, locals: tablets[4]str }` | struct desc: `ptr_offsets=[0, 24, 32]` (name's ptr, locals.head, locals.tail). Chunk desc on the tablets handles slot contents. |
| `tablets[N]str` | value desc `[0, 8]`; chunk desc lists each of N slots' ptr offsets plus `next`. |
| `tablets[N]Entry` where Entry contains a str | chunk desc lists each slot's composed Entry offsets (i.e. for Entry=`{str, i64}`, slot i contributes `[i*32 + 0]`) plus `next`. |
| `tablets[N]tablets[M]T` | chunk desc lists slot ptrs for each inner tablets' `[head, tail]` (offsets 0+0, 0+8 within each slot), plus `next`. Inner tablets chunks trace through their own desc. |
| `seal X { Text(str), Silent }` | trace_fn: switch on tag. Tag 0 → mark payload[0] as str ptr. Tag 1 → no-op. |
| `seal X { Entry(struct{str, i64}), Empty }` | trace_fn: switch on tag. Tag 0 → mark payload[0] as the str ptr inside the Entry struct (composed offsets). Tag 1 → no-op. |
| `seal X { Nested(seal Y) }` | trace_fn: switch on X's tag. Tag 0 → recurse into Y's trace_fn with the payload bytes as Y's object. |
| `wedge Entry` where Entry has a str | wedge value is not rooted. Its target chunk is rooted via the owning tablets. Reads through the wedge work because the chunk is alive. |
| `buffer[1024]u8` | no descriptor; no ptrs. |
| `fn(str) -> str` | code pointer, no descriptor. The result str gets rooted at the chokepoint. |

---

## Root discipline — the chokepoint

### Rule

A cleanup-bearing value becomes GC-visible exactly when it is
spilled to a shadow-stack slot via `_force_root_cleanup_value(val)`.
This is the ONLY rooting primitive; all other helpers route through
it.

### Where the chokepoint fires

1. **After any `A.Call` that returns a cleanup-bearing type.** In
   `_gen_expr`'s Call branch. One hook.
2. **After any `A.Binary` op that returns a cleanup-bearing type.**
   Currently only `str + str`. In `_gen_binary`'s str-concat branch.
3. **After `_deep_clone_if_cleanup_bearing(val)`** when `cloned is
   not val` (i.e. an actual clone happened). Already factored into
   `_deep_clone_and_root` — keep it.
4. **After `A.Copy`** (the explicit `copy x` keyword). Currently
   returns SSA; wrap with root.
5. **At `_gen_tablets_lit_addr` for the tablets value slot.** Already
   rooted — keep.

### Where it does NOT fire (over-rooting to avoid)

- Reads that produce aliases (`Ident` / `Field` / `Index` / `StringLit`).
  These are borrow sources; the OWNER's root covers them.
- Scope-exit cleanup registration (`_maybe_register_cleanup`) —
  this binds a named slot; the binding site roots it once. Rvalue
  cleanup (anonymous temp) and binding cleanup should NOT BOTH root
  the same value.

### Consumer sites — remove their local rooting

Every one of these currently has a `clone + maybe-root` block;
delete the rooting half, keep only the ownership-transfer logic
(which is still needed to decide transfer vs clone):

- `_gen_struct_lit` — field eval loop.
- `_gen_variant_ctor` — payload arg eval.
- `_gen_tablets_method` push — arg eval.
- `_gen_assign` — RHS eval.
- `_register_str_rvalue_cleanup` — becomes DELETE (its job is now
  the chokepoint's).
- `_register_struct_rvalue_cleanup` — same, DELETE.
- Non-mut cleanup-bearing param spill in `_gen_fn_body` — keep (a
  param is a producer from the caller's perspective; the callee
  can't see the caller's chokepoint).

### Rvalue cleanup — does it still exist?

Yes, but simplified. A cleanup-bearing SSA value that becomes
anonymous (flows into a container, gets printed, etc.) needs a
scope-exit cleanup entry so the scope-exit release fires. Under
GC, release is a no-op — but we keep the registration so Stage 2.5
can remove the whole release machinery as a single change.

The chokepoint already adds a cleanup entry when it spills. So:
one primitive, two effects (root + cleanup registration).

---

## File split plan for `codegen/__init__.py`

Monolithic today (~4500 lines). Split into cohesive modules. Each
becomes a mixin on the main `Codegen` class, same pattern as
`strs.py` / `tablets.py` / `_common.py` already use.

### `codegen/cleanup.py` (NEW)

- `_push_cleanup_frame` / `_pop_cleanup_frame`.
- `_emit_frame_cleanups` / `_emit_gc_frame_pop` / `_emit_all_cleanups_for_early_return`.
- `_maybe_register_cleanup`.
- `_force_root_cleanup_value` (the one primitive).
- `_transfer_cleanup_into_container`.
- `_transfer_ownership_out`.
- `_zero_transferred_slot`.
- GC runtime externs (`_get_gc_alloc_bytes` / `_get_gc_alloc_typed` / `_get_gc_push_root` / `_get_gc_pop_roots`).
- Type descriptor emission (`_type_desc_key` / `_type_ptr_offsets` / `_chunk_ptr_offsets` / `_get_type_desc` / `_get_chunk_type_desc`).
- Seal trace-fn emission (new).

Roughly 600–800 lines.

### `codegen/call.py` (NEW)

- `_gen_call`.
- `_gen_fn_value_call`.
- `_gen_tablets_method` (move from tablets.py? or keep — tablets is logically about tablets).
- Variant ctor (`_gen_variant_ctor`).
- Arg marshaling (including the cleanup-bearing-param spill).
- Gloss dispatch (`_gen_gloss_call`).
- println / print (`_gen_print`, `_emit_one_print`).

Roughly 800–1000 lines.

### `codegen/expr.py` (NEW)

- `_gen_expr` main dispatcher.
- Small arithmetic / logic expr helpers (`_gen_unary`, `_gen_binary`, `_gen_if_expr`, `_gen_match`, `_gen_block`).
- `_gen_copy` (the `copy` keyword — can go away entirely post-GC since clones are cheap under shadow-stack, but keep for source-level visibility).

Roughly 600–800 lines.

### `codegen/stmt.py` (NEW)

- `_gen_stmt` dispatcher.
- `_gen_binding`, `_gen_assign`, `_gen_while`, `_gen_for`, `_gen_yield`, `_gen_release`.

Roughly 400–500 lines.

### `codegen/userty.py` (NEW)

- StructLit / seal variant ctor / TabletsLit emission.
- Struct / seal identified-type registration.
- Struct / seal release fn generation (goes away in Stage 2.5).
- Struct / seal clone fn generation (stays — explicit `copy` still compiles).

Roughly 500–700 lines.

### `codegen/__init__.py` (SLIMS DOWN)

- The main `Codegen` class definition + mixin composition.
- Module-level setup (init, externs, globals).
- Driver `codegen(prog, checker)` entry point.
- `_lower_type` / `_lower_ty` / `_coerce`.
- `_alloca_entry`, `_lookup`, `_bind`.
- Small top-level utilities that don't fit elsewhere.

Target: under 1500 lines.

### Mechanics

Each mixin gets the relevant methods moved verbatim. No behavior
change expected from the split alone. Tests guard it.

---

## Freeze-machinery deletion

After the chokepoint lands and stress mode passes, the analyzer's
correctness role dissolves. Cuts from `typecheck.py`:

- `_borrow_sources` (per-scope list of borrow-root mappings).
- `_all_borrow_sources`, `_borrow_paths`, `_field_borrows`.
- `_invalidated`, `_invalidate_root`, `_invalidate_mut_call_args`.
- `_borrow_binding_nodes`.
- `_register_borrow`, `_rebind_borrow`, `_is_borrow_binding`.
- `_borrow_source_root`, `_borrow_source_path`, `_root_for`.
- `_check_use_not_invalidated`.
- `_wrap_escape_sites`, `_return_borrow_escape`, `_infer_return_alias`.
- `_fn_params`, `fn_return_alias`, `fn_param_names`.
- `_wedge_arena_root`, `_needs_borrow_tracking`.

What stays (as optimization hints for Stage 3, not as correctness):

- Phase B effect analysis (`effects.py`) — proves non-allocation for
  root elision.
- `_local_tablets` / `_tainted` — wedge escape rule (stays a static
  check because wedges aren't rooted; escaping a wedge really would
  be a bug).

Rough delete: 1200–1500 lines of typecheck.

## Typecheck-side fallout

- `A.Copy` stays in AST, parser, typecheck. Codegen lowers it
  via `_deep_clone_and_root`. Semantically redundant under GC
  (every allocation is already tracked) but useful as a source-
  level hint meaning "I want a fresh owner here."
- `CompileWarning` stays; used by effect analysis / stage-3 lints.
- Implicit-copy rewrite in `_check_use_not_invalidated` goes away
  with the analyzer it patches.
- Some tests that asserted specific freeze-rule behavior get
  deleted or rewritten to test GC behavior.

## Test churn

Currently 667 tests. Expected after this refactor:
- `test_ownership.py` — mostly deleted or rewritten; the ownership
  model it was testing no longer exists.
- `test_effects.py` — keeps the analyzer unit tests; deletes the
  end-to-end freeze-rule tests.
- `test_string.py`, `test_tablets.py`, `test_struct.py`, `test_sum.py`,
  `test_examples.py` — stay largely as-is. These test runtime
  correctness, which the GC now provides.
- New: `test_gc.py` — stress-mode regression tests, ensuring
  specific shapes (tablets-of-str, struct-with-tablets, seal-with-
  str-payload) survive forced collection.

Expect test count to drop by ~100 after deletion, grow by ~20 from
new GC-specific tests. Target: ~580 passing in both normal and
stress modes.

---

## Execution order — mechanical checklist for next session

1. **Read this file first** to reconstitute context.
2. **Chokepoint lands atomically:**
   a. Add spill+root in `_gen_expr` Call branch.
   b. Add spill+root in `_gen_binary` str-concat branch.
   c. Add spill+root in `_gen_copy`.
   d. DELETE `_register_str_rvalue_cleanup` + every call site
      (~6 call sites; they become no-ops).
   e. DELETE `_register_struct_rvalue_cleanup` + call sites.
   f. In `_gen_struct_lit` / `_gen_variant_ctor` / `_gen_tablets_method`
      push: simplify the clone branches (drop the `cloned ? force_root`
      flag; `_deep_clone_and_root` handles it).
3. **Add seal trace_fn emission** so seal-with-str-payload shapes
   trace correctly under stress. New code in `cleanup.py`.
4. **Run `TUPPU_GC_STRESS=1 pytest`** — target 0 failures. If any
   remain, they'll point at specific flows the chokepoint missed
   (probably unusual call shapes like method dispatch on a struct
   field holding a fn pointer — audit those).
5. **File split.** Move methods between files per the plan above.
   Verify `pytest` stays green after each file moves. No behavior
   change.
6. **Freeze-machinery deletion.** One commit removes every item in
   the list above from `typecheck.py`. Delete or rewrite
   `test_ownership.py`. Verify `pytest` green.
7. **Delete `copy` keyword** surface if we're going all-in on
   GC: lexer token, parser handler, `A.Copy` AST node, codegen
   `_gen_copy`, typecheck `_tc_copy`. OR keep it for source
   visibility — decide at review time.
8. **Benchmark** vs `BENCH_BASELINE.md`. Expect binary +50–150KB,
   runtime +20–40% on allocation-heavy hot paths under stress-off
   mode. If worse, diagnose.

---

## Critical files to touch (by the refactor)

- `src/tuppu/codegen/__init__.py` — shrinks dramatically.
- `src/tuppu/codegen/cleanup.py` — NEW.
- `src/tuppu/codegen/call.py` — NEW.
- `src/tuppu/codegen/expr.py` — NEW.
- `src/tuppu/codegen/stmt.py` — NEW.
- `src/tuppu/codegen/userty.py` — NEW.
- `src/tuppu/codegen/tablets.py` — small change to `_gen_tablets_method` push; drop local rooting.
- `src/tuppu/codegen/strs.py` — small; drop `_register_str_rvalue_cleanup` call sites inside str helpers.
- `src/tuppu/typecheck.py` — heavy deletion.
- `src/tuppu/effects.py` — stays.
- `src/tuppu/ast.py` — maybe drop `A.Copy` if we delete the keyword.
- `src/tuppu/parser.py` / `src/tuppu/lexer.py` — same.
- `runtime/tuppu_gc.c` — add variant-aware trace_fn dispatch
  (already has the field; codegen now emits fns into it).
- `tests/test_ownership.py` — heavy deletion / rewrite.
- `tests/test_gc.py` — NEW.

## Verification

- `pytest` (normal) — target 580ish passing, zero failing.
- `TUPPU_GC_STRESS=1 pytest` — same set passes.
- `./tuppu run lua_interp.tpu` with `TUPPU_GC_STRESS=1` — the
  community workload that motivated this. Must run clean.
- `./tuppu build examples/hello.tpu` — binary size under 150KB.
  Compare with `BENCH_BASELINE.md`.
- Line-count check on `codegen/__init__.py` — should be under
  1500 lines after split.
- Line-count check on `typecheck.py` — should be under 1800 after
  deletion.
