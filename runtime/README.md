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

Everything is allocated via GC. Analyzer still runs and still
enforces UAF-prevention. Worst-case perf — every alloc goes
through the GC, every cleanup-bearing local is on the shadow
stack. This stage is deliberately unoptimized so we can measure
"pure GC cost" vs. baseline and then measure what Stage 3
recovers.

**Must migrate atomically.** A half-migrated state crashes: if
any allocation path still calls malloc while the matching release
path calls __tuppu_gc_alloc_bytes (or vice versa), `free()` on a
GC-owned buffer corrupts the libc heap. No partial-migration
intermediate state is safe.

Migration checklist (all in one commit or one tight sequence):

- [x] Codegen: GC-extern helpers + type descriptor emitter.
      (commit abc028b)
- [ ] Codegen: type descriptor for str (one ptr field at offset 0,
      24-byte size). Call via `_get_type_desc`.
- [ ] Codegen: type descriptors for user structs (transitively
      cleanup-bearing). Walk fields, pick up str-field offsets;
      extend later to nested struct / seal fields.
- [ ] Codegen: type descriptor for tablets — CUSTOM TRACE needed.
      Tablets chunks are a linked-list of blocks; a simple
      ptr_offsets array can't express walk-the-chain. Add a
      `trace_fn` field to `tuppu_type_t`, emit a per-tablets-type
      trace fn that walks chunks and calls `mark_ptr` on each
      element's ptr field. Runtime change: if `trace != NULL`,
      call `trace(obj)` instead of iterating `ptr_offsets`.
- [ ] Codegen: replace `_get_malloc()` with `_get_gc_alloc_bytes()`
      at every str / tablets allocation site — there are ~12 total
      across codegen/__init__.py, codegen/strs.py, codegen/tablets.py.
      Consider just changing `_get_malloc` itself to return the GC
      allocator; then every caller transparently moves.
- [ ] Codegen: remove `free()` calls in str_release, tablets
      release, cstr-to-str cleanup. Make these paths no-ops (GC
      reclaims). Alternatively route `_get_free` to a no-op.
- [ ] Codegen: at `_maybe_register_cleanup`, also emit
      `_emit_gc_push_root(slot, value_ty)`. Track the count per
      cleanup frame.
- [ ] Codegen: at `_emit_frame_cleanups`, after the existing
      release walks, emit `_emit_gc_pop_roots(count)`.
- [ ] Runtime: add `trace_fn` field + call site in tuppu_gc.c.
- [ ] Run full test suite — expect some adjustments for the
      changed malloc ↔ GC boundary.
- [ ] Capture Stage-2 benchmark pass.

**Crucial correctness invariant** (when stage 2 lands): every live
cleanup-bearing local must be on the shadow stack between any two
allocation sites. An allocation can trigger a GC, which will
reclaim anything not rooted. The conservative rule — push a root
at every alloca, pop at every return — is unconditionally safe.
Optimization (stage 3) loosens this.

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
