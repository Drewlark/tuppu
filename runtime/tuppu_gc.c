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

/* Object header precedes every GC-tracked allocation. */
typedef struct tuppu_hdr {
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
        if (r.type) {
            /* Slot IS a struct by value — trace its pointer fields in place. */
            trace_struct((char*)r.slot, r.type);
        } else {
            /* Slot holds a bare pointer (e.g. raw bytes). */
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
    if (live_bytes >= gc_threshold) {
        __tuppu_gc_collect();
        /* Adaptive threshold: keep GC frequency roughly constant by
         * aiming for live_bytes at collect time ~= half threshold. */
        if (live_bytes * 2 > gc_threshold) {
            gc_threshold = live_bytes * 2;
        }
    }
}

static void* raw_alloc(size_t obj_size, const tuppu_type_t* type) {
    maybe_collect();
    size_t total = HDR_SIZE + obj_size;
    tuppu_hdr_t* hdr = (tuppu_hdr_t*)calloc(1, total);
    if (!hdr) {
        fprintf(stderr, "tuppu: out of memory allocating %zu bytes\n", total);
        abort();
    }
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
