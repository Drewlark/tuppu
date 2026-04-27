"""ivec codegen — `ivec<T>` is a contiguous heap-allocated array of
pointers to per-element T allocations. Different shape from
`tablets[N]T`: contiguous instead of chunk-chained, one T per heap
object instead of N-per-chunk, O(1) random access (two loads — one
into the pointer array, one through the slot pointer).

Layout shared across every T (`IVEC_STRUCT`):
    { buf: i8**, len: i64, cap: i64 }

Storage allocation goes through `__tuppu_gc_alloc(cap*8,
&__tuppu_ivec_storage_desc)` — the runtime descriptor's trace fn
walks `cap` pointer slots regardless of T (each slot is just an i8*).

Per-element heap allocation goes through `__tuppu_gc_alloc(sizeof(T),
T's descriptor or NULL for leaf T)` so the GC traces inside T
correctly. With smart wedges, taking a `wedge T` to an ivec slot
remains valid across grows — only the pointer array moves, not the
T allocations themselves.
"""
from __future__ import annotations

from llvmlite import ir

from .. import ast as A
from ._common import (
    CodegenError, IVecInfo,
    I1, I8, I32, I64,
    IVEC_STRUCT, IVEC_IDX_BUF, IVEC_IDX_LEN, IVEC_IDX_CAP,
    IVEC_INITIAL_CAP,
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

    def _get_ivec_storage_desc(self) -> ir.GlobalVariable:
        """Reference the runtime's static `__tuppu_ivec_storage_desc`
        as an extern global. The runtime trace fn walks
        `(allocation_size - HDR_SIZE) / sizeof(void*)` slots, so we
        don't need a per-cap descriptor at codegen time."""
        existing = self.module.globals.get("__tuppu_ivec_storage_desc")
        if existing is not None:
            return existing
        desc = ir.GlobalVariable(
            self.module, self._type_desc_ty, "__tuppu_ivec_storage_desc",
        )
        # External — defined in runtime/tuppu_gc.c. No initializer here.
        return desc

    def _emit_ivec_grow(
        self, b: ir.IRBuilder, iv_ptr: ir.Value, new_cap: ir.Value,
    ) -> None:
        """Allocate a new buffer of `new_cap * 8` bytes, copy old
        contents (if any), update the ivec's buf and cap. Old buffer
        becomes orphan and gets swept on the next collection."""
        ZERO_I32 = ir.Constant(I32, 0)
        BUF_I32  = ir.Constant(I32, IVEC_IDX_BUF)
        CAP_I32  = ir.Constant(I32, IVEC_IDX_CAP)

        buf_addr = b.gep(iv_ptr, [ZERO_I32, BUF_I32], inbounds=True)
        cap_addr = b.gep(iv_ptr, [ZERO_I32, CAP_I32], inbounds=True)
        old_buf = b.load(buf_addr)
        old_cap = b.load(cap_addr)

        new_size_bytes = b.mul(new_cap, ir.Constant(I64, 8))
        desc = self._get_ivec_storage_desc()
        raw = b.call(
            self._get_gc_alloc_typed(),
            [new_size_bytes, b.bitcast(desc, I8.as_pointer())],
        )
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
          1. If len == cap, grow the buffer (initial cap = 8, double
             from there).
          2. Allocate a fresh T on the heap with T's descriptor (so
             the GC traces inside T correctly).
          3. Store `val` into the heap slot.
          4. Save the slot pointer at `buf[len]`.
          5. len++.
        Returns a `T*` pointing at the heap slot — usable as a
        `wedge T` since smart wedges trace through it."""
        elem_ty = info.elem_ty
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

        entry     = fn.append_basic_block("entry")
        need_grow = fn.append_basic_block("grow")
        after_grow = fn.append_basic_block("after.grow")
        do_insert = fn.append_basic_block("insert")

        b = ir.IRBuilder(entry)
        len_addr = b.gep(iv_ptr, [ZERO_I32, LEN_I32], inbounds=True)
        cap_addr = b.gep(iv_ptr, [ZERO_I32, CAP_I32], inbounds=True)
        cur_len = b.load(len_addr)
        cur_cap = b.load(cap_addr)
        is_full = b.icmp_signed("==", cur_len, cur_cap)
        b.cbranch(is_full, need_grow, do_insert)

        b.position_at_end(need_grow)
        cap_is_zero = b.icmp_signed("==", cur_cap, ir.Constant(I64, 0))
        new_cap_initial = ir.Constant(I64, IVEC_INITIAL_CAP)
        new_cap_doubled = b.mul(cur_cap, ir.Constant(I64, 2))
        new_cap = b.select(cap_is_zero, new_cap_initial, new_cap_doubled)
        self._emit_ivec_grow(b, iv_ptr, new_cap)
        b.branch(after_grow)

        b.position_at_end(after_grow)
        b.branch(do_insert)

        # Insert path. Allocate the per-element heap slot, store val
        # into it, and write the slot pointer at buf[len].
        b.position_at_end(do_insert)
        elem_size = ir.Constant(I64, self._size_of(elem_ty))
        elem_desc = self._get_type_desc(elem_ty)
        if elem_desc is None:
            desc_arg = ir.Constant(I8.as_pointer(), None)
        else:
            desc_arg = b.bitcast(elem_desc, I8.as_pointer())
        raw_slot = b.call(
            self._get_gc_alloc_typed(),
            [elem_size, desc_arg],
        )
        slot_typed = b.bitcast(raw_slot, elem_ty.as_pointer())
        b.store(val, slot_typed)

        cur_len2 = b.load(len_addr)
        cur_buf = b.load(b.gep(iv_ptr, [ZERO_I32, BUF_I32], inbounds=True))
        # buf is i8** — index gives an i8*-slot; bitcast to T**-slot
        # so we can store our typed slot pointer.
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
                        v = self._deep_clone_if_cleanup_bearing(v)
                elif self._is_borrow_source_expr(arg_expr):
                    v = self._deep_clone_if_cleanup_bearing(v)
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
