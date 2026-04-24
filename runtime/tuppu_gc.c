/*
 * Tuppu GC runtime — stage 1 spike.
 *
 * Precise, single-generation, stop-the-world mark-sweep. No write
 * barriers, no compaction, no concurrency. Intended as the minimal
 * floor that validates the architecture; grows only if profiles
 * demand it.
 *
 * Root discovery: shadow stack. Codegen emits
 *   __tuppu_gc_push_root(&my_local, &my_type_desc)
 * at every fn entry for each cleanup-bearing local, and
 *   __tuppu_gc_pop_roots(n)
 * on every fn exit path. The GC scans this stack at mark time.
 *
 * Allocation: `__tuppu_gc_alloc(size, type)` for typed objects whose
 * layout has a descriptor, `__tuppu_gc_alloc_bytes(n)` for raw
 * leaf buffers (str contents, tablet chunks — no internal pointers).
 * Every object carries a small header with a type descriptor (or
 * NULL for bytes), total size, mark bit, and a link into the
 * intrusive live-list used by sweep.
 *
 * The Tuppu compiler emits the per-type descriptors as LLVM globals
 * at codegen time and wires the allocator + shadow-stack calls into
 * the same places that used to call malloc / free.
 *
 * This file is bundled into every Tuppu binary by the driver (see
 * driver.py:link).
 */
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>

/* Public interface — kept in sync with the LLVM externs in codegen. */

typedef struct tuppu_type {
    const char* name;          /* debug only */
    size_t      size;          /* object size in bytes (0 for byte buffers) */
    size_t      n_ptrs;        /* pointer-field count */
    const size_t* ptr_offsets; /* byte offsets of pointer fields within object */
} tuppu_type_t;

void* __tuppu_gc_alloc(size_t obj_size, const tuppu_type_t* type);
void* __tuppu_gc_alloc_bytes(size_t n);
void  __tuppu_gc_push_root(void* slot, const tuppu_type_t* type);
void  __tuppu_gc_pop_roots(size_t n);
void  __tuppu_gc_collect(void);

/* Magic precedes every GC header so mark_ptr can distinguish a
 * GC-tracked allocation from a malloc'd buffer or random stack
 * garbage. Chosen so it doesn't collide with common byte patterns
 * (all zeros, all ones, pointer-looking values). */
#define TUPPU_GC_MAGIC 0x7475707075475443ULL   /* "tuppuGTC" */

/* Object header precedes every GC-tracked allocation. */
typedef struct tuppu_hdr {
    uint64_t            magic;  /* TUPPU_GC_MAGIC — sentinel */
    const tuppu_type_t* type;   /* NULL = raw bytes (leaf) */
    size_t              size;   /* total allocation size including header */
    uint8_t             mark;
    uint8_t             _pad[7];
    struct tuppu_hdr*   next;   /* intrusive live-list link */
} tuppu_hdr_t;

#define HDR_SIZE (sizeof(tuppu_hdr_t))
#define OBJ_OF(h) ((void*)((char*)(h) + HDR_SIZE))
#define HDR_OF(obj) ((tuppu_hdr_t*)((char*)(obj) - HDR_SIZE))

/* --- heap state ---------------------------------------------------- */

static tuppu_hdr_t* live_list = NULL;
static size_t       live_bytes = 0;
static size_t       gc_threshold = 64 * 1024;  /* grow threshold: 64 KB */

/* --- shadow stack -------------------------------------------------- */

#define SHADOW_MAX 65536

typedef struct {
    void*               slot;  /* ptr to the alloca / struct holding the object */
    const tuppu_type_t* type;  /* descriptor for the struct AT slot */
} tuppu_root_t;

static tuppu_root_t shadow[SHADOW_MAX];
static size_t       shadow_top = 0;

void __tuppu_gc_push_root(void* slot, const tuppu_type_t* type) {
    if (shadow_top >= SHADOW_MAX) {
        fprintf(stderr, "tuppu: GC shadow stack overflow\n");
        abort();
    }
    shadow[shadow_top].slot = slot;
    shadow[shadow_top].type = type;
    shadow_top++;
}

void __tuppu_gc_pop_roots(size_t n) {
    if (n > shadow_top) {
        fprintf(stderr, "tuppu: GC shadow stack underflow (pop %zu, have %zu)\n",
                n, shadow_top);
        abort();
    }
    shadow_top -= n;
}

/* --- mark ---------------------------------------------------------- */

/* Opt-in tracing via the `TUPPU_GC_DEBUG=1` env var. Enables
 * per-allocation and per-collection stderr lines, used during
 * migration work to verify shadow-stack state at collect time. */
static int gc_debug = 0;

/* Stress mode: `TUPPU_GC_STRESS=1` forces a collection on every
 * allocation, so any missed root registration fails immediately
 * rather than hiding behind the normal threshold. Turn this on
 * in CI / local testing to catch root-tracking bugs at the fastest
 * possible cadence. */
static int gc_stress = 0;

static void mark_ptr(void* p);

static void trace_struct(char* obj, const tuppu_type_t* type) {
    if (!type || !type->ptr_offsets) return;
    for (size_t i = 0; i < type->n_ptrs; i++) {
        void* p = *(void**)(obj + type->ptr_offsets[i]);
        mark_ptr(p);
    }
}

static void mark_ptr(void* p) {
    if (!p) return;
    tuppu_hdr_t* hdr = HDR_OF(p);
    /* Pointer-validation: during the migration window, Tuppu has
     * strs and tablets backed by GC memory but other leaf pointers
     * (colophon returns, string literals pointing into globals,
     * malloc'd buffers) also flow through the same type-descriptor
     * machinery. The magic byte distinguishes a GC header from
     * everything else; unknown ptrs are harmless to ignore. */
    if (hdr->magic != TUPPU_GC_MAGIC) return;
    if (hdr->mark) return;
    hdr->mark = 1;
    if (hdr->type) {
        trace_struct((char*)p, hdr->type);
    }
    /* Leaf bytes — nothing to trace into. */
}

static void mark_all(void) {
    for (size_t i = 0; i < shadow_top; i++) {
        tuppu_root_t r = shadow[i];
        if (gc_debug) {
            fprintf(stderr, "gc: root[%zu] slot=%p type=%s n_ptrs=%zu\n",
                    i, r.slot,
                    r.type ? r.type->name : "(bytes)",
                    r.type ? r.type->n_ptrs : 0);
            if (r.type) {
                for (size_t j = 0; j < r.type->n_ptrs; j++) {
                    void* p = *(void**)((char*)r.slot + r.type->ptr_offsets[j]);
                    fprintf(stderr, "gc:   ptr[%zu off=%zu] = %p\n",
                            j, r.type->ptr_offsets[j], p);
                }
            }
        }
        if (r.type) {
            trace_struct((char*)r.slot, r.type);
        } else {
            mark_ptr(*(void**)r.slot);
        }
    }
}

/* --- sweep --------------------------------------------------------- */

static void sweep(void) {
    tuppu_hdr_t** prev = &live_list;
    tuppu_hdr_t*  cur  = live_list;
    while (cur) {
        tuppu_hdr_t* next = cur->next;
        if (cur->mark) {
            cur->mark = 0;
            prev = &cur->next;
        } else {
            *prev = next;
            live_bytes -= cur->size;
            free(cur);
        }
        cur = next;
    }
}

/* --- collect / alloc ---------------------------------------------- */

void __tuppu_gc_collect(void) {
    mark_all();
    sweep();
}

static void maybe_collect(void) {
    if (gc_stress || live_bytes >= gc_threshold) {
        if (gc_debug) fprintf(stderr, "gc: collect start live=%zu top=%zu\n",
                              live_bytes, shadow_top);
        __tuppu_gc_collect();
        if (gc_debug) fprintf(stderr, "gc: collect end   live=%zu\n",
                              live_bytes);
        if (live_bytes * 2 > gc_threshold) {
            gc_threshold = live_bytes * 2;
        }
    }
}

__attribute__((destructor))
static void gc_fini(void) {
    /* Program exit: the shadow stack should be empty if every fn
     * entry's push was matched by a paired pop. A non-zero depth
     * means codegen emitted unbalanced push/pop IR somewhere —
     * silent leak today, deterministic memory bug later. Abort so
     * it's caught in testing. Suppress in production builds if
     * ever needed, but during the migration this check is the
     * cheapest correctness oracle we have. */
    if (shadow_top != 0) {
        fprintf(stderr,
                "tuppu: GC shadow-stack leak at exit: %zu roots still pushed\n",
                shadow_top);
        abort();
    }
}

__attribute__((constructor))
static void gc_init(void) {
    const char* dbg = getenv("TUPPU_GC_DEBUG");
    gc_debug = (dbg && dbg[0] == '1');
    const char* str = getenv("TUPPU_GC_STRESS");
    gc_stress = (str && str[0] == '1');
}

static void* raw_alloc(size_t obj_size, const tuppu_type_t* type) {
    maybe_collect();
    if (gc_debug) fprintf(stderr, "gc: alloc %zu bytes live=%zu\n",
                          obj_size, live_bytes);
    size_t total = HDR_SIZE + obj_size;
    tuppu_hdr_t* hdr = (tuppu_hdr_t*)calloc(1, total);
    if (!hdr) {
        fprintf(stderr, "tuppu: out of memory allocating %zu bytes\n", total);
        abort();
    }
    hdr->magic = TUPPU_GC_MAGIC;
    hdr->type = type;
    hdr->size = total;
    hdr->mark = 0;
    hdr->next = live_list;
    live_list = hdr;
    live_bytes += total;
    return OBJ_OF(hdr);
}

void* __tuppu_gc_alloc(size_t obj_size, const tuppu_type_t* type) {
    return raw_alloc(obj_size, type);
}

void* __tuppu_gc_alloc_bytes(size_t n) {
    return raw_alloc(n, NULL);
}

/* During the Stage 2 migration, codegen still emits `free(ptr)`
 * calls from the old release paths (str_release, tablet chunk free).
 * We route those to this no-op so the libc heap doesn't get
 * corrupted by a free on a GC-owned buffer. Once Stage 2.5 deletes
 * the release machinery outright, this symbol becomes unused. */
void __tuppu_gc_noop_free(void* p) {
    (void)p;
}
