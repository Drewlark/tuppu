# Tuppu runtime

C code that links into every Tuppu binary. Currently one file:

- `tuppu_gc.c` — mark-sweep GC allocator + shadow-stack root tracking.

## Why a separate runtime

Tuppu's compiler emits LLVM IR; the IR gets compiled to an object via
llvmlite + clang and linked. Everything that's too complex or too
platform-sensitive to emit as IR lives here: the free-list allocator,
the mark traversal, the shadow-stack machinery. LLVM IR calls out to
these via the `__tuppu_gc_*` externs.

The driver (see `src/tuppu/driver.py::link`) compiles this file to
an object alongside the user program and links both into the final
binary.

## GC runtime spec (minimum viable spike)

Interface (LLVM externs emitted by codegen):

```
i8*  __tuppu_gc_alloc(i64 size, i8* type_desc)
i8*  __tuppu_gc_alloc_bytes(i64 n)
void __tuppu_gc_push_root(i8* slot, i8* type_desc)
void __tuppu_gc_pop_roots(i64 n)
void __tuppu_gc_collect(void)
```

### Allocation

`__tuppu_gc_alloc(size, type)` returns a zero-initialized block of
`size` bytes, headed by a GC header. `type` is a pointer to a
static `tuppu_type_t` describing the object's layout for tracing.

`__tuppu_gc_alloc_bytes(n)` is the leaf variant — type=NULL, no
internal pointers, used for str contents and tablet chunk backing
storage.

The allocator triggers a GC when `live_bytes >= threshold`. Threshold
grows adaptively so collection rate stays roughly constant.

### Type descriptors

Each user type (tablet, seal variant, str) gets one `tuppu_type_t`
global emitted by codegen:

```c
typedef struct {
    const char* name;
    size_t      size;
    size_t      n_ptrs;
    const size_t* ptr_offsets;
} tuppu_type_t;
```

At mark time the GC reads `n_ptrs` and `ptr_offsets`, loads the
pointer field at each offset, and recurses. Seals get one descriptor
per variant; runtime dispatch on the tag.

### Roots

Codegen emits at every fn entry, for each local with a GC-tracked
type:

```
%str_slot = alloca %str
call void @__tuppu_gc_push_root(i8* %str_slot, i8* @__tuppu_str_type_desc)
```

And at every exit path (normal return, yield, early-return):

```
call void @__tuppu_gc_pop_roots(i64 N)
```

The shadow-stack entry is `(slot_ptr, type)`. Slot is a pointer to
the stack-allocated struct (alloca); type tells the GC which fields
inside that struct are pointers.

Borrow-style bindings (str params whose bytes are caller-owned) don't
need roots — the caller's root already covers them. This is a small
optimization; omitting it is sound but bloats the shadow stack.

### Not in the spike

- Write barriers (no generational GC yet)
- Concurrent mark (STW only)
- Compaction / bump allocation (free-list only)
- Finalizers
- Weak references

All of those are future optimizations if profiles demand them. The
current shape is "simplest thing that works."

## Migration state

### Stage 1 — runtime wired in (LANDED)

- [x] Runtime compiled & linked via driver.
- [x] Baseline captured in BENCH_BASELINE.md.

### Stage 2 — GC on, analyzer still correctness

**Progress so far (all LANDED, still gated behind libc malloc):**
- GC runtime (`tuppu_gc.c`) with magic-guarded mark-sweep + shadow
  stack. `TUPPU_GC_DEBUG=1` opt-in trace.
- Codegen helpers: `_get_gc_alloc_bytes`, `_get_gc_push_root`,
  `_get_gc_pop_roots`.
- Per-type descriptor emitter `_get_type_desc(value_ty)` for str
  and user structs with str fields.
- `_push_cleanup_frame` / `_pop_cleanup_frame` / `_register_gc_root`
  wrappers wired into every cleanup-frame lifecycle site.
- `_maybe_register_cleanup`, `_register_str_rvalue_cleanup`,
  `_register_struct_rvalue_cleanup` all push GC roots.
- Non-mut cleanup-bearing params spill to shadow-stack-rooted
  slots at fn entry (critical: SSA params are invisible to the
  collector otherwise).
- Early-return / yield paths emit a cumulative `pop_roots(total)`.

**The atomic switch is NOT yet flipped.** `_get_malloc` still
returns libc malloc. When it flips to `__tuppu_gc_alloc_bytes`,
the following work has to land together in one change:

- [ ] **Tablets tracing.** This is the blocker. `tablets[N]T` uses
      a chunk chain (`{elements: [N×T], used: i64, next: *node}`)
      that a flat ptr_offsets table can't walk. Runtime needs a
      `trace_fn` field on `tuppu_type_t`; if non-null, GC calls it
      instead of iterating offsets. Codegen emits a per-
      (N, elem_ty) trace fn that walks chunks and calls
      `mark_ptr` on each pointer-bearing field of each used slot.
- [ ] Runtime: add `trace_fn` field + dispatch to `tuppu_type_t`,
      update `trace_struct` / `mark_ptr` to call it.
- [ ] Codegen: `_type_desc_key` / `_type_ptr_offsets` / new
      `_type_trace_fn` cover tablets + struct-with-tablets-field
      + seal variants with cleanup payloads.
- [ ] Flip `_get_malloc` to return `_get_gc_alloc_bytes()`. Flip
      `_get_free` to return a no-op (add `__tuppu_gc_noop_free`
      to runtime, which exists already in this branch's earlier
      exploration).
- [ ] Diagnose two known failure modes from the earlier spike:
      - `test_slice_of_concat_no_leak` crashes at ~100 iterations
        because the stdlib `str_concat` uses `mut buf: tablets[64]u8`
        internally; without tablets tracing, the chunk gets GC'd
        mid-iteration.
      - `test_str_repeat_linear_on_large_n` same root cause.
- [ ] Update `test_helpers_emitted` to match new IR
      (`__tuppu_gc_alloc_bytes` in place of `malloc`).
- [ ] Capture Stage-2 benchmark pass vs BENCH_BASELINE.md.

**Crucial correctness invariant** (when Stage 2 lands): every live
cleanup-bearing local must be on the shadow stack between any two
allocation sites. An allocation can trigger a GC, which will
reclaim anything not rooted. Conservative rule — push a root at
every alloca, pop at every return — is unconditionally safe.
Optimization (Stage 3) loosens this.

**Lesson from the partial attempt:** SSA register values aren't
roots. Non-mut cleanup-bearing params had to be spilled to
shadow-stack slots at fn entry. Same discipline applies to any
intermediate result held in SSA across an allocating call — the
existing `_register_str_rvalue_cleanup` / `_register_struct_rvalue_cleanup`
sites cover the common cases (anonymous call-arg temporaries), but
sweep through once the atomic flip happens to verify no SSA
borrow-return survives across a GC trigger.

### Stage 2.5 — delete correctness analyzer

Analyzer's role shifts from "catch UAFs" to "enable optimizations."
GC is now the safety net. `copy` keyword goes away; cleanup frames
go away; transfer-or-clone storage-site discipline goes away.

- [ ] Typecheck: delete freeze/escape correctness rules.
- [ ] Delete `copy` keyword, lexer/parser/ast/codegen paths.
- [ ] Delete cleanup-frame machinery in codegen.
- [ ] Delete transfer-or-clone at storage sites.
- [ ] Delete cap-sentinel branches in str_release.

### Stage 3 — re-layer analyzer as optimization

Existing machinery repurposed. Same code paths, different
consumers — instead of raising CheckError, they emit codegen
hints.

- [ ] `_return_borrow_escape` / escape detection → stack-alloc hint.
- [ ] Phase B effect analysis → "does this fn allocate?" summary.
- [ ] Caller-side root elision for provably-pure callees.
- [ ] Region allocation for non-escaping `mut` tablets.
- [ ] Stage-3 benchmark pass; compare all three stages side by side.
