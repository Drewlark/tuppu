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

- [x] Runtime compiled & linked via driver.
- [x] Baseline captured in BENCH_BASELINE.md.
- [ ] Codegen: emit type descriptors for str.
- [ ] Codegen: emit push_root / pop_roots at fn entry / exit.
- [ ] Codegen: replace malloc for str bytes with __tuppu_gc_alloc_bytes.
- [ ] Codegen: stop emitting free in str_release.
- [ ] Codegen: type descriptors + alloc calls for tablets chunks.
- [ ] Codegen: type descriptors for user structs / seals.
- [ ] Typecheck: delete freeze/escape machinery.
- [ ] Delete `copy` keyword surface.
- [ ] Re-run BENCH_BASELINE comparisons.
