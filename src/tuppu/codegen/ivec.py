"""ivec codegen — `ivec<T>` is a contiguous slot-pointer array backed
by a chunk-chain of `Node_<T>_K` arenas. The `buf` array gives O(1)
random access; chunks (reused from the tablets infrastructure) hold
the actual T values K-at-a-time, amortizing one heap allocation
across K pushes.

Layout shared across every T (`IVEC_STRUCT`):
    { buf: i8**, len: i64, cap: i64, head_node: i8*, tail_node: i8* }

Slot storage:
  * `head_node` / `tail_node` are the chunk chain. Each chunk is a
    `Node_<T>_K` (same shape tablets uses): K inline T slots plus
    `used` and `next`. Allocations go through
    `__tuppu_gc_alloc(sizeof(Node), &chunk_desc)` so the per-T chunk
    descriptor traces inside every T slot — strs, nested vecs, seals,
    and wedges all reach correctly.
  * `buf` is a leaf-byte allocation (`__tuppu_gc_alloc_bytes`); its
    contents are interior pointers into chunks. The GC keeps `buf`
    alive (it's a normal `mark_ptr` target) but does NOT trace
    through it — the chunks are kept alive independently via the
    `head_node` / `tail_node` chain. That avoids the soundness hazard
    of mark_ptr on interior pointers (which can't tell an interior
    pointer from a stale start pointer except by the magic byte) and
    the cost of walking N slot pointers per collection.

Wedges into `iv[i]` stay valid across both `buf` grows and further
pushes: `buf` may relocate, but each slot's address is a fixed offset
inside its chunk and chunks never move once allocated.
"""
from __future__ import annotations

from llvmlite import ir

from .. import ast as A
from ._common import (
    CodegenError, IVecInfo,
    I1, I8, I32, I64,
    IVEC_STRUCT, IVEC_IDX_BUF, IVEC_IDX_LEN, IVEC_IDX_CAP,
    IVEC_IDX_HEAD, IVEC_IDX_TAIL,
    IVEC_INITIAL_CAP, IVEC_CHUNK_K,
)


class IVecMixin:
    # --- type access + cache --------------------------------------------

    def _get_ivec(
        self, elem_ty: ir.Type, elem_is_wedge: bool = False,
    ) -> IVecInfo:
        """Lookup-or-create the IVecInfo for a given element type.
        The struct LLVM type is shared across every T (literal struct
        `{ i8**, i64, i64 }`), so only the helper fns are per-T."""
        key = (str(elem_ty), elem_is_wedge)
        cache = self._ivec_types.setdefault("__map__", {})
        existing = cache.get(key)
        if existing is not None:
            return existing
        suffix = f"{elem_ty}".replace(" ", "_").replace("{", "").replace("}", "")
        if elem_is_wedge:
            suffix = f"w_{suffix}"
        info = IVecInfo(
            elem_ty=elem_ty, suffix=suffix, elem_is_wedge=elem_is_wedge,
        )
        cache[key] = info
        return info

    def _ivec_info_for(self, value_ty: ir.Type) -> IVecInfo | None:
        """Reverse lookup: which IVecInfo do we have for this LLVM
        type? Returns None if not an ivec value type. The struct type
        is shared, so this just checks shape — callers needing the
        per-T helper must know T from another source."""
        # All ivecs share IVEC_STRUCT, so a single shape match suffices.
        if value_ty is IVEC_STRUCT:
            return None  # ambiguous — caller needs T context
        return None

    def _is_ivec_value(self, value_ty: ir.Type) -> bool:
        return value_ty is IVEC_STRUCT

    # --- storage allocator + grow --------------------------------------

    def _emit_ivec_grow(
        self, b: ir.IRBuilder, iv_ptr: ir.Value, new_cap: ir.Value,
    ) -> None:
        """Allocate a new buf of `new_cap * 8` bytes as leaf bytes
        (the chunk chain keeps slot targets alive; buf contents need
        no tracing), copy old slot pointers if any, update the ivec's
        buf and cap. The old buf becomes orphan and gets swept on the
        next collection."""
        ZERO_I32 = ir.Constant(I32, 0)
        BUF_I32  = ir.Constant(I32, IVEC_IDX_BUF)
        CAP_I32  = ir.Constant(I32, IVEC_IDX_CAP)

        buf_addr = b.gep(iv_ptr, [ZERO_I32, BUF_I32], inbounds=True)
        cap_addr = b.gep(iv_ptr, [ZERO_I32, CAP_I32], inbounds=True)
        old_buf = b.load(buf_addr)
        old_cap = b.load(cap_addr)

        new_size_bytes = b.mul(new_cap, ir.Constant(I64, 8))
        raw = b.call(self._get_gc_alloc_bytes(), [new_size_bytes])
        new_buf = b.bitcast(raw, I8.as_pointer().as_pointer())

        # If there's existing data, memcpy it into the new buffer.
        # `cap == 0` means "no buffer yet" — skip the copy. We test
        # via cap rather than buf-null so a freshly-zero-init ivec
        # (mut iv: ivec<T>) is handled even before any push.
        fn = b.function
        do_copy = fn.append_basic_block("ivec.grow.copy")
        after   = fn.append_basic_block("ivec.grow.after")
        has_old = b.icmp_signed("!=", old_cap, ir.Constant(I64, 0))
        b.cbranch(has_old, do_copy, after)

        b.position_at_end(do_copy)
        old_size_bytes = b.mul(old_cap, ir.Constant(I64, 8))
        memcpy = self._get_memcpy_for_ivec()
        b.call(
            memcpy,
            [
                b.bitcast(new_buf, I8.as_pointer()),
                b.bitcast(old_buf, I8.as_pointer()),
                old_size_bytes,
                ir.Constant(I1, 0),
            ],
        )
        b.branch(after)

        b.position_at_end(after)
        b.store(new_buf, buf_addr)
        b.store(new_cap, cap_addr)

    def _get_memcpy_for_ivec(self) -> ir.Function:
        """`llvm.memcpy.p0i8.p0i8.i64` — used to copy the old pointer
        array into a freshly-allocated larger one on grow. The str
        runtime already declares this; reuse if present."""
        name = "llvm.memcpy.p0i8.p0i8.i64"
        existing = self.module.globals.get(name)
        if isinstance(existing, ir.Function):
            return existing
        fty = ir.FunctionType(
            ir.VoidType(),
            [I8.as_pointer(), I8.as_pointer(), I64, I1],
        )
        return ir.Function(self.module, fty, name)

    # --- per-T helper fns -----------------------------------------------

    def _get_ivec_push(self, info: IVecInfo) -> ir.Function:
        if info.push is None:
            info.push = self._build_ivec_push(info)
        return info.push

    def _get_ivec_get(self, info: IVecInfo) -> ir.Function:
        if info.get is None:
            info.get = self._build_ivec_get(info)
        return info.get

    def _get_ivec_get_addr(self, info: IVecInfo) -> ir.Function:
        if info.get_addr is None:
            info.get_addr = self._build_ivec_get_addr(info)
        return info.get_addr

    def _build_ivec_push(self, info: IVecInfo) -> ir.Function:
        """Push a T into an ivec. Steps:
          1. If `len == cap`, grow the slot-pointer buf (leaf bytes).
          2. If the tail chunk is null or full (`used == K`),
             allocate a fresh chunk via the per-T chunk descriptor
             and link it into the chain.
          3. Compute the slot address inside the tail chunk, store
             `val` into it, and bump the chunk's `used`.
          4. Save the slot pointer at `buf[len]` and bump `iv.len`.
          5. Return the slot pointer — usable as a `wedge T`. The
             chunk that owns the slot is reachable from the ivec's
             head/tail pointers, so the wedge stays valid for as
             long as either the ivec or the wedge itself is rooted.
        """
        elem_ty = info.elem_ty
        K = IVEC_CHUNK_K
        # Reuse tablets infrastructure for chunks: same Node_T layout
        # ([K x T] items, used: i64, next: *Node), same per-T chunk
        # descriptor with all the seal/wedge dispatch already wired.
        tinfo = self._get_tablets(K, elem_ty, elem_is_wedge=info.elem_is_wedge)
        node_ty = tinfo.node_ty
        node_ptr_ty = node_ty.as_pointer()
        null_node = ir.Constant(node_ptr_ty, None)

        fn = ir.Function(
            self.module,
            ir.FunctionType(
                elem_ty.as_pointer(),
                [IVEC_STRUCT.as_pointer(), elem_ty],
            ),
            name=f"__tuppu_ivec_{info.suffix}_push",
        )
        fn.args[0].name = "iv"
        fn.args[1].name = "val"
        iv_ptr, val = fn.args

        ZERO_I32 = ir.Constant(I32, 0)
        BUF_I32  = ir.Constant(I32, IVEC_IDX_BUF)
        LEN_I32  = ir.Constant(I32, IVEC_IDX_LEN)
        CAP_I32  = ir.Constant(I32, IVEC_IDX_CAP)
        HEAD_I32 = ir.Constant(I32, IVEC_IDX_HEAD)
        TAIL_I32 = ir.Constant(I32, IVEC_IDX_TAIL)

        entry      = fn.append_basic_block("entry")
        need_grow  = fn.append_basic_block("grow.buf")
        after_grow = fn.append_basic_block("after.grow")
        check_full = fn.append_basic_block("check.full")
        need_chunk = fn.append_basic_block("need.chunk")
        link_head  = fn.append_basic_block("link.head")
        link_tail  = fn.append_basic_block("link.tail")
        do_insert  = fn.append_basic_block("insert")

        b = ir.IRBuilder(entry)
        len_addr  = b.gep(iv_ptr, [ZERO_I32, LEN_I32],  inbounds=True)
        cap_addr  = b.gep(iv_ptr, [ZERO_I32, CAP_I32],  inbounds=True)
        head_addr = b.gep(iv_ptr, [ZERO_I32, HEAD_I32], inbounds=True)
        tail_addr = b.gep(iv_ptr, [ZERO_I32, TAIL_I32], inbounds=True)

        # 1. Grow buf (slot-pointer cache) if full.
        cur_len = b.load(len_addr)
        cur_cap = b.load(cap_addr)
        is_full_buf = b.icmp_signed("==", cur_len, cur_cap)
        b.cbranch(is_full_buf, need_grow, check_full)

        b.position_at_end(need_grow)
        cap_is_zero = b.icmp_signed("==", cur_cap, ir.Constant(I64, 0))
        new_cap_initial = ir.Constant(I64, IVEC_INITIAL_CAP)
        new_cap_doubled = b.mul(cur_cap, ir.Constant(I64, 2))
        new_cap = b.select(cap_is_zero, new_cap_initial, new_cap_doubled)
        self._emit_ivec_grow(b, iv_ptr, new_cap)
        b.branch(after_grow)

        b.position_at_end(after_grow)
        b.branch(check_full)

        # 2. Need new chunk if tail is null or its `used == K`.
        b.position_at_end(check_full)
        tail_i8 = b.load(tail_addr)
        tail = b.bitcast(tail_i8, node_ptr_ty)
        tail_is_null = b.icmp_signed("==", tail, null_node)
        # Two-step dispatch — branch on null first to avoid
        # dereferencing a null tail's `used` field.
        check_used = fn.append_basic_block("check.used")
        b.cbranch(tail_is_null, need_chunk, check_used)

        b.position_at_end(check_used)
        used_addr_existing = b.gep(
            tail, [ZERO_I32, ir.Constant(I32, 1)], inbounds=True,
        )
        used_existing = b.load(used_addr_existing)
        is_chunk_full = b.icmp_signed("==", used_existing, ir.Constant(I64, K))
        b.cbranch(is_chunk_full, need_chunk, do_insert)

        # need.chunk: alloc a fresh Node_T via the per-T chunk
        # descriptor, init `used = 0` and `next = null`, then link
        # into the chain. gc_alloc_typed may collect; the new chunk
        # is rooted as soon as it's stored into head_node / tail_node
        # (both are listed in IVEC_STRUCT's ptr_offsets).
        b.position_at_end(need_chunk)
        size_ptr = b.gep(null_node, [ir.Constant(I32, 1)], inbounds=False)
        node_size = b.ptrtoint(size_ptr, I64)
        chunk_desc = self._get_chunk_type_desc(
            K, elem_ty, node_ty, elem_is_wedge=info.elem_is_wedge,
        )
        raw_chunk = b.call(
            self._get_gc_alloc_typed(),
            [node_size, b.bitcast(chunk_desc, I8.as_pointer())],
        )
        new_chunk = b.bitcast(raw_chunk, node_ptr_ty)
        b.store(
            ir.Constant(I64, 0),
            b.gep(new_chunk, [ZERO_I32, ir.Constant(I32, 1)], inbounds=True),
        )
        b.store(
            null_node,
            b.gep(new_chunk, [ZERO_I32, ir.Constant(I32, 2)], inbounds=True),
        )
        # Link: empty chain → set head; non-empty → splice onto old
        # tail's next. Both branches end at do_insert with the new
        # chunk as the live tail.
        was_empty = b.icmp_signed("==", tail, null_node)
        b.cbranch(was_empty, link_head, link_tail)

        b.position_at_end(link_head)
        new_chunk_i8 = b.bitcast(new_chunk, I8.as_pointer())
        b.store(new_chunk_i8, head_addr)
        b.store(new_chunk_i8, tail_addr)
        b.branch(do_insert)

        b.position_at_end(link_tail)
        old_tail_next_addr = b.gep(
            tail, [ZERO_I32, ir.Constant(I32, 2)], inbounds=True,
        )
        b.store(new_chunk, old_tail_next_addr)
        new_chunk_i8b = b.bitcast(new_chunk, I8.as_pointer())
        b.store(new_chunk_i8b, tail_addr)
        b.branch(do_insert)

        # 3. Insert into the live tail chunk: write val at
        #    items[used], bump used, then mirror the slot pointer
        #    into buf[len] and bump len.
        b.position_at_end(do_insert)
        cur_tail = b.bitcast(b.load(tail_addr), node_ptr_ty)
        used_addr = b.gep(
            cur_tail, [ZERO_I32, ir.Constant(I32, 1)], inbounds=True,
        )
        cur_used = b.load(used_addr)
        slot_typed = b.gep(
            cur_tail,
            [ZERO_I32, ZERO_I32, cur_used],
            inbounds=True,
        )
        b.store(val, slot_typed)
        b.store(b.add(cur_used, ir.Constant(I64, 1)), used_addr)

        # 4. Cache slot pointer in buf[len] for O(1) random access.
        cur_len2 = b.load(len_addr)
        cur_buf = b.load(b.gep(iv_ptr, [ZERO_I32, BUF_I32], inbounds=True))
        slot_in_buf_i8 = b.gep(cur_buf, [cur_len2], inbounds=True)
        slot_in_buf_typed = b.bitcast(
            slot_in_buf_i8, elem_ty.as_pointer().as_pointer(),
        )
        b.store(slot_typed, slot_in_buf_typed)
        b.store(b.add(cur_len2, ir.Constant(I64, 1)), len_addr)
        b.ret(slot_typed)
        return fn

    def _build_ivec_get(self, info: IVecInfo) -> ir.Function:
        """Load buf[i] -> T*, then *T*.  Bounds-check is at the call
        site (mirrors tablets `get`). Returns T by value."""
        elem_ty = info.elem_ty
        fn = ir.Function(
            self.module,
            ir.FunctionType(
                elem_ty,
                [IVEC_STRUCT.as_pointer(), I64],
            ),
            name=f"__tuppu_ivec_{info.suffix}_get",
        )
        fn.args[0].name = "iv"
        fn.args[1].name = "idx"
        iv_ptr, idx = fn.args

        ZERO_I32 = ir.Constant(I32, 0)
        BUF_I32  = ir.Constant(I32, IVEC_IDX_BUF)

        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        buf = b.load(b.gep(iv_ptr, [ZERO_I32, BUF_I32], inbounds=True))
        slot_addr_i8 = b.gep(buf, [idx], inbounds=True)
        slot_addr = b.bitcast(
            slot_addr_i8, elem_ty.as_pointer().as_pointer(),
        )
        slot_ptr = b.load(slot_addr)  # T*
        val = b.load(slot_ptr)
        b.ret(val)
        return fn

    def _build_ivec_get_addr(self, info: IVecInfo) -> ir.Function:
        """Like `get`, but returns the slot pointer (T*). Used for
        lvalues — `iv[i] = x` writes through this pointer, which
        keeps the heap T's address stable (a wedge into iv[i] taken
        before the assignment stays valid afterwards)."""
        elem_ty = info.elem_ty
        fn = ir.Function(
            self.module,
            ir.FunctionType(
                elem_ty.as_pointer(),
                [IVEC_STRUCT.as_pointer(), I64],
            ),
            name=f"__tuppu_ivec_{info.suffix}_get_addr",
        )
        fn.args[0].name = "iv"
        fn.args[1].name = "idx"
        iv_ptr, idx = fn.args

        ZERO_I32 = ir.Constant(I32, 0)
        BUF_I32  = ir.Constant(I32, IVEC_IDX_BUF)

        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        buf = b.load(b.gep(iv_ptr, [ZERO_I32, BUF_I32], inbounds=True))
        slot_addr_i8 = b.gep(buf, [idx], inbounds=True)
        slot_addr = b.bitcast(
            slot_addr_i8, elem_ty.as_pointer().as_pointer(),
        )
        slot_ptr = b.load(slot_addr)  # T*
        b.ret(slot_ptr)
        return fn

    # --- expression dispatch helpers ------------------------------------

    def _gen_ivec_method(
        self, info: IVecInfo, var, method: str, args,
    ) -> ir.Value | None:
        """Lower `iv.push(x)`. Receiver must be a mut binding (the
        push mutates the ivec struct in place). Mirrors tablets
        method-dispatch in spirit."""
        if not var.is_mut:
            raise CodegenError("ivec methods require a mut binding")
        assert self.builder is not None
        ptr = var.ir_ref
        if method == "push":
            if len(args) != 1:
                raise CodegenError("ivec.push takes exactly one argument")
            v = self._gen_expr(args[0])
            if v is None:
                raise CodegenError("ivec.push argument has no value")
            v = self._coerce(v, info.elem_ty)
            # Cleanup-bearing element types follow the same
            # transfer/clone discipline as tablets.push: owning Ident
            # transfers ownership into the ivec; borrows clone; fresh
            # rvalues pass through.
            if self._is_cleanup_bearing_ty(info.elem_ty):
                arg_expr = args[0]
                if isinstance(arg_expr, A.Ident):
                    res = self._transfer_cleanup_into_container(
                        arg_expr.name,
                    )
                    if not res:
                        v = self._deep_clone_if_cleanup_bearing(
                            v, for_transfer=True,
                        )
                elif self._is_borrow_source_expr(arg_expr):
                    v = self._deep_clone_if_cleanup_bearing(
                        v, for_transfer=True,
                    )
            return self.builder.call(self._get_ivec_push(info), [ptr, v])
        raise CodegenError(f"ivec has no method {method!r}")

    def _gen_ivec_field(self, var, field_name: str) -> ir.Value:
        """`iv.len` — loads the len field. No other fields exposed."""
        assert self.builder is not None
        if field_name == "len":
            if isinstance(var.ir_ref.type, ir.PointerType):
                len_addr = self.builder.gep(
                    var.ir_ref,
                    [ir.Constant(I32, 0), ir.Constant(I32, IVEC_IDX_LEN)],
                    inbounds=True,
                )
                return self.builder.load(len_addr)
            return self.builder.extract_value(var.ir_ref, IVEC_IDX_LEN)
        raise CodegenError(f"ivec has no field {field_name!r}; only .len")

    def _ivec_elem_for_call(self, e: A.Call) -> ir.Type | None:
        """Forward T from typecheck's `ivec_elem_at_call` sideband.
        Returns None if the checker didn't tag this call (e.g., the
        receiver isn't actually an ivec). Caller should fall back to
        the existing dispatch in that case."""
        if self._checker is None:
            return None
        ty = self._checker.ivec_elem_at_call.get(id(e))
        if ty is None:
            return None
        return self._lower_ty(ty)

    def _ivec_elem_for_index(self, e: A.Index) -> ir.Type | None:
        if self._checker is None:
            return None
        ty = self._checker.ivec_elem_at_index.get(id(e))
        if ty is None:
            return None
        return self._lower_ty(ty)

    def _ivec_elem_for_for(self, f: A.ForStmt) -> ir.Type | None:
        if self._checker is None:
            return None
        ty = self._checker.ivec_elem_at_for.get(id(f))
        if ty is None:
            return None
        return self._lower_ty(ty)

    def _gen_ivec_index(
        self, info: IVecInfo, var, idx_expr,
    ) -> ir.Value:
        """`iv[i]` — bounds-check vs len, then call get. Returns T."""
        assert self.builder is not None
        if not isinstance(var.ir_ref.type, ir.PointerType):
            slot = self._alloca_entry(var.ir_ref.type, ".ivec.view")
            self.builder.store(var.ir_ref, slot)
            iv_ptr = slot
        else:
            iv_ptr = var.ir_ref
        idx = self._gen_expr(idx_expr)
        if idx is None:
            raise CodegenError("ivec index has no value")
        idx = self._coerce(idx, I64)
        len_addr = self.builder.gep(
            iv_ptr,
            [ir.Constant(I32, 0), ir.Constant(I32, IVEC_IDX_LEN)],
            inbounds=True,
        )
        length = self.builder.load(len_addr)
        self._emit_dynamic_bounds_trap(idx, length)
        val = self.builder.call(self._get_ivec_get(info), [iv_ptr, idx])
        # Reads from a container are borrows — neuter cleanup markers
        # so the value can be assigned/copied without double-free risk.
        return self._read_borrow(val)
