"""dvec codegen — `dvec<T>` is a contiguous heap-allocated array of
T values stored inline. Different shape from `ivec<T>` (array of
pointers) and from `tablets[N]T` (chunk-chained); same shape as
Rust's `Vec<T>` or C++'s `std::vector<T>` minus iterator
invalidation guarantees we don't promise.

Layout shared across every T (`DVEC_STRUCT`):
    { buf: i8*, len: i64, cap: i64 }

The buffer is `cap * sizeof(T)` bytes. Indexing GEPs by
`i * sizeof(T)` and bitcasts to `T*`. Grow allocates a new buffer
of the doubled byte size, memcpys the old contents, and repoints
`buf`. T values' addresses move on grow, so `push` returns unit and
no `wedge T` is ever derived from `dv[i]` — that prevents users from
holding handles that would dangle on the next push.

Per-T descriptor: each `dvec<T>` needs its own buffer descriptor
because the trace fn must walk T-typed slots, calling
`_emit_trace_mark_calls(b, base, i*sizeof(T), T)` per slot. The
trace fn reads its own GC header to compute cap dynamically, so a
single trace fn handles every grown size.
"""
from __future__ import annotations

from llvmlite import ir

from .. import ast as A
from ._common import (
    CodegenError, DVecInfo,
    I1, I8, I32, I64,
    DVEC_STRUCT, DVEC_IDX_BUF, DVEC_IDX_LEN, DVEC_IDX_CAP,
    DVEC_INITIAL_CAP,
)


class DVecMixin:
    def _get_dvec(
        self, elem_ty: ir.Type, elem_is_wedge: bool = False,
    ) -> DVecInfo:
        """Lookup-or-create per-T DVecInfo. Cache key includes
        elem_is_wedge so `dvec<wedge T>` and `dvec<*T>` (which lower
        to the same LLVM `T*`) get distinct descriptors."""
        key = (str(elem_ty), elem_is_wedge)
        cache = self._dvec_types.setdefault("__map__", {})
        existing = cache.get(key)
        if existing is not None:
            return existing
        suffix = f"{elem_ty}".replace(" ", "_").replace("{", "").replace("}", "")
        if elem_is_wedge:
            suffix = f"w_{suffix}"
        info = DVecInfo(
            elem_ty=elem_ty, suffix=suffix, elem_is_wedge=elem_is_wedge,
        )
        cache[key] = info
        return info

    def _is_dvec_value(self, value_ty: ir.Type) -> bool:
        return value_ty is DVEC_STRUCT

    # --- buffer trace fn + descriptor ----------------------------------

    def _get_dvec_storage_desc(self, info: DVecInfo) -> ir.GlobalVariable:
        """Per-T `tuppu_type_t` global for a dvec storage buffer.
        Trace fn walks the inline T slots; size field is 0 (variable,
        runtime reads from header). Cached on info.desc."""
        if info.desc is not None:
            return info.desc
        trace_fn = self._build_dvec_buffer_trace_fn(info)
        key = f"__tuppu_dvec_buf_{info.suffix}"
        name_bytes = (key + "\0").encode("utf-8")
        name_arr_ty = ir.ArrayType(I8, len(name_bytes))
        name_arr = ir.GlobalVariable(
            self.module, name_arr_ty, f"{key}_name",
        )
        name_arr.linkage = "internal"
        name_arr.global_constant = True
        name_arr.initializer = ir.Constant(
            name_arr_ty, bytearray(name_bytes),
        )
        # Empty offsets array — descriptor uses trace fn dispatch.
        offsets_arr_ty = ir.ArrayType(I64, 1)
        offsets_arr = ir.GlobalVariable(
            self.module, offsets_arr_ty, f"{key}_offsets",
        )
        offsets_arr.linkage = "internal"
        offsets_arr.global_constant = True
        offsets_arr.initializer = ir.Constant(
            offsets_arr_ty, [ir.Constant(I64, 0)],
        )
        desc = ir.GlobalVariable(self.module, self._type_desc_ty, key)
        desc.linkage = "internal"
        desc.global_constant = True
        desc.initializer = ir.Constant(self._type_desc_ty, [
            name_arr.bitcast(I8.as_pointer()),
            ir.Constant(I64, 0),                    # variable size
            ir.Constant(I64, 0),                    # n_ptrs unused
            offsets_arr.bitcast(I64.as_pointer()),  # ptr_offsets unused
            trace_fn,
        ])
        info.desc = desc
        info.trace_fn = trace_fn
        return desc

    def _build_dvec_buffer_trace_fn(self, info: DVecInfo) -> ir.Function:
        """Trace fn for a per-T dvec buffer.

        Calls the runtime helper `__tuppu_gc_data_size` to get the
        bytes-of-data behind the buffer pointer (header layout stays
        a runtime detail), divides by sizeof(T) to get the current
        cap, and walks each slot through `_emit_trace_mark_calls`.
        Slots beyond the dvec's `len` are calloc-zero (gc_alloc
        zeroes via calloc), and mark_ptr's null check makes those a
        safe no-op — so we trace cap slots, avoiding the need for
        the trace fn to know which `dvec` value owns this buffer."""
        elem_ty = info.elem_ty
        fn_name = f"__tuppu_dvec_buf_{info.suffix}_trace"
        cached = self.module.globals.get(fn_name)
        if isinstance(cached, ir.Function):
            return cached
        fn = ir.Function(self.module, self._trace_fn_ty, fn_name)
        fn.linkage = "internal"
        entry = fn.append_basic_block("entry")
        loop  = fn.append_basic_block("loop")
        body  = fn.append_basic_block("body")
        done  = fn.append_basic_block("done")
        b = ir.IRBuilder(entry)
        buf = fn.args[0]  # i8* pointing at start of inline T data
        data_size = b.call(self._get_gc_data_size(), [buf])
        elem_size = self._size_of(elem_ty)
        cap = b.sdiv(data_size, ir.Constant(I64, elem_size))
        b.branch(loop)

        b.position_at_end(loop)
        i_phi = b.phi(I64, "i")
        i_phi.add_incoming(ir.Constant(I64, 0), entry)
        cont = b.icmp_signed("<", i_phi, cap)
        b.cbranch(cont, body, done)

        b.position_at_end(body)
        offset_bytes = b.mul(i_phi, ir.Constant(I64, elem_size))
        # Pass the slot's base pointer and offset 0 — the helper
        # operates relative to its `base` arg.
        slot_base = b.gep(buf, [offset_bytes], inbounds=True)
        saved_builder = self.builder
        self.builder = b
        try:
            self._emit_trace_mark_calls(
                b, slot_base, 0, elem_ty,
                is_wedge_field=info.elem_is_wedge,
            )
        finally:
            self.builder = saved_builder
        i_next = b.add(i_phi, ir.Constant(I64, 1))
        i_phi.add_incoming(i_next, b.block)
        b.branch(loop)

        b.position_at_end(done)
        b.ret_void()
        return fn

    # --- per-T helper fns -----------------------------------------------

    def _get_dvec_push(self, info: DVecInfo) -> ir.Function:
        if info.push is None:
            info.push = self._build_dvec_push(info)
        return info.push

    def _get_dvec_get(self, info: DVecInfo) -> ir.Function:
        if info.get is None:
            info.get = self._build_dvec_get(info)
        return info.get

    def _get_dvec_get_addr(self, info: DVecInfo) -> ir.Function:
        if info.get_addr is None:
            info.get_addr = self._build_dvec_get_addr(info)
        return info.get_addr

    def _emit_dvec_grow(
        self, b: ir.IRBuilder, dv_ptr: ir.Value, info: DVecInfo,
        new_cap: ir.Value,
    ) -> None:
        """Allocate a new buffer of `new_cap * sizeof(T)` bytes with
        T's per-T descriptor, memcpy the inline T bytes from the old
        buffer, repoint dv.buf and update dv.cap. Old buffer becomes
        orphan and gets swept on the next collection."""
        ZERO_I32 = ir.Constant(I32, 0)
        BUF_I32  = ir.Constant(I32, DVEC_IDX_BUF)
        CAP_I32  = ir.Constant(I32, DVEC_IDX_CAP)

        elem_size = self._size_of(info.elem_ty)
        buf_addr = b.gep(dv_ptr, [ZERO_I32, BUF_I32], inbounds=True)
        cap_addr = b.gep(dv_ptr, [ZERO_I32, CAP_I32], inbounds=True)
        old_buf = b.load(buf_addr)
        old_cap = b.load(cap_addr)

        new_size_bytes = b.mul(new_cap, ir.Constant(I64, elem_size))
        desc = self._get_dvec_storage_desc(info)
        raw = b.call(
            self._get_gc_alloc_typed(),
            [new_size_bytes, b.bitcast(desc, I8.as_pointer())],
        )
        new_buf = raw  # already i8*

        fn = b.function
        do_copy = fn.append_basic_block("dvec.grow.copy")
        after   = fn.append_basic_block("dvec.grow.after")
        has_old = b.icmp_signed("!=", old_cap, ir.Constant(I64, 0))
        b.cbranch(has_old, do_copy, after)

        b.position_at_end(do_copy)
        old_size_bytes = b.mul(old_cap, ir.Constant(I64, elem_size))
        memcpy = self._get_memcpy_for_ivec()  # same llvm.memcpy intrinsic
        b.call(
            memcpy,
            [new_buf, old_buf, old_size_bytes, ir.Constant(I1, 0)],
        )
        b.branch(after)

        b.position_at_end(after)
        b.store(new_buf, buf_addr)
        b.store(new_cap, cap_addr)

    def _build_dvec_push(self, info: DVecInfo) -> ir.Function:
        """Push a T into a dvec.

        1. If len == cap, grow the buffer (initial cap = 8, double
           thereafter).
        2. Compute slot address: buf + len * sizeof(T).
        3. Bitcast to T* and store val.
        4. len++.

        Returns void — dvec.push hands back no handle because the
        slot's address invalidates on the next grow."""
        elem_ty = info.elem_ty
        fn = ir.Function(
            self.module,
            ir.FunctionType(
                ir.VoidType(),
                [DVEC_STRUCT.as_pointer(), elem_ty],
            ),
            name=f"__tuppu_dvec_{info.suffix}_push",
        )
        fn.args[0].name = "dv"
        fn.args[1].name = "val"
        dv_ptr, val = fn.args

        ZERO_I32 = ir.Constant(I32, 0)
        BUF_I32  = ir.Constant(I32, DVEC_IDX_BUF)
        LEN_I32  = ir.Constant(I32, DVEC_IDX_LEN)
        CAP_I32  = ir.Constant(I32, DVEC_IDX_CAP)

        entry      = fn.append_basic_block("entry")
        need_grow  = fn.append_basic_block("grow")
        after_grow = fn.append_basic_block("after.grow")
        do_insert  = fn.append_basic_block("insert")

        b = ir.IRBuilder(entry)
        len_addr = b.gep(dv_ptr, [ZERO_I32, LEN_I32], inbounds=True)
        cap_addr = b.gep(dv_ptr, [ZERO_I32, CAP_I32], inbounds=True)
        cur_len = b.load(len_addr)
        cur_cap = b.load(cap_addr)
        is_full = b.icmp_signed("==", cur_len, cur_cap)
        b.cbranch(is_full, need_grow, do_insert)

        b.position_at_end(need_grow)
        cap_is_zero = b.icmp_signed("==", cur_cap, ir.Constant(I64, 0))
        new_cap = b.select(
            cap_is_zero,
            ir.Constant(I64, DVEC_INITIAL_CAP),
            b.mul(cur_cap, ir.Constant(I64, 2)),
        )
        self._emit_dvec_grow(b, dv_ptr, info, new_cap)
        b.branch(after_grow)

        b.position_at_end(after_grow)
        b.branch(do_insert)

        b.position_at_end(do_insert)
        elem_size = self._size_of(elem_ty)
        cur_buf = b.load(b.gep(dv_ptr, [ZERO_I32, BUF_I32], inbounds=True))
        cur_len2 = b.load(len_addr)
        offset_bytes = b.mul(cur_len2, ir.Constant(I64, elem_size))
        slot_i8 = b.gep(cur_buf, [offset_bytes], inbounds=True)
        slot_typed = b.bitcast(slot_i8, elem_ty.as_pointer())
        b.store(val, slot_typed)
        b.store(b.add(cur_len2, ir.Constant(I64, 1)), len_addr)
        b.ret_void()
        return fn

    def _build_dvec_get(self, info: DVecInfo) -> ir.Function:
        elem_ty = info.elem_ty
        fn = ir.Function(
            self.module,
            ir.FunctionType(
                elem_ty,
                [DVEC_STRUCT.as_pointer(), I64],
            ),
            name=f"__tuppu_dvec_{info.suffix}_get",
        )
        fn.args[0].name = "dv"
        fn.args[1].name = "idx"
        dv_ptr, idx = fn.args

        ZERO_I32 = ir.Constant(I32, 0)
        BUF_I32  = ir.Constant(I32, DVEC_IDX_BUF)

        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        elem_size = self._size_of(elem_ty)
        buf = b.load(b.gep(dv_ptr, [ZERO_I32, BUF_I32], inbounds=True))
        offset_bytes = b.mul(idx, ir.Constant(I64, elem_size))
        slot_i8 = b.gep(buf, [offset_bytes], inbounds=True)
        slot_typed = b.bitcast(slot_i8, elem_ty.as_pointer())
        b.ret(b.load(slot_typed))
        return fn

    def _build_dvec_get_addr(self, info: DVecInfo) -> ir.Function:
        """Return a T* into the inline buffer. Caller must use the
        pointer immediately — it dangles on the next grow. Used for
        `dv[i] = x` lvalue, where the assignment happens before any
        chance of a grow, so it's safe."""
        elem_ty = info.elem_ty
        fn = ir.Function(
            self.module,
            ir.FunctionType(
                elem_ty.as_pointer(),
                [DVEC_STRUCT.as_pointer(), I64],
            ),
            name=f"__tuppu_dvec_{info.suffix}_get_addr",
        )
        fn.args[0].name = "dv"
        fn.args[1].name = "idx"
        dv_ptr, idx = fn.args

        ZERO_I32 = ir.Constant(I32, 0)
        BUF_I32  = ir.Constant(I32, DVEC_IDX_BUF)

        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        elem_size = self._size_of(elem_ty)
        buf = b.load(b.gep(dv_ptr, [ZERO_I32, BUF_I32], inbounds=True))
        offset_bytes = b.mul(idx, ir.Constant(I64, elem_size))
        slot_i8 = b.gep(buf, [offset_bytes], inbounds=True)
        slot_typed = b.bitcast(slot_i8, elem_ty.as_pointer())
        b.ret(slot_typed)
        return fn

    # --- expression dispatch helpers ------------------------------------

    def _dvec_elem_for_call(self, e: A.Call) -> ir.Type | None:
        if self._checker is None:
            return None
        ty = self._checker.dvec_elem_at_call.get(id(e))
        if ty is None:
            return None
        return self._lower_ty(ty)

    def _dvec_elem_for_index(self, e: A.Index) -> ir.Type | None:
        if self._checker is None:
            return None
        ty = self._checker.dvec_elem_at_index.get(id(e))
        if ty is None:
            return None
        return self._lower_ty(ty)

    def _dvec_elem_for_for(self, f: A.ForStmt) -> ir.Type | None:
        if self._checker is None:
            return None
        ty = self._checker.dvec_elem_at_for.get(id(f))
        if ty is None:
            return None
        return self._lower_ty(ty)

    def _gen_dvec_method(
        self, info: DVecInfo, var, method: str, args,
    ) -> ir.Value | None:
        if not var.is_mut:
            raise CodegenError("dvec methods require a mut binding")
        assert self.builder is not None
        ptr = var.ir_ref
        if method == "push":
            if len(args) != 1:
                raise CodegenError("dvec.push takes exactly one argument")
            v = self._gen_expr(args[0])
            if v is None:
                raise CodegenError("dvec.push argument has no value")
            v = self._coerce(v, info.elem_ty)
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
            self.builder.call(self._get_dvec_push(info), [ptr, v])
            return None
        raise CodegenError(f"dvec has no method {method!r}")

    def _gen_dvec_field(self, var, field_name: str) -> ir.Value:
        assert self.builder is not None
        if field_name == "len":
            if isinstance(var.ir_ref.type, ir.PointerType):
                len_addr = self.builder.gep(
                    var.ir_ref,
                    [ir.Constant(I32, 0), ir.Constant(I32, DVEC_IDX_LEN)],
                    inbounds=True,
                )
                return self.builder.load(len_addr)
            return self.builder.extract_value(var.ir_ref, DVEC_IDX_LEN)
        raise CodegenError(f"dvec has no field {field_name!r}; only .len")

    def _gen_dvec_index(
        self, info: DVecInfo, var, idx_expr,
    ) -> ir.Value:
        assert self.builder is not None
        if not isinstance(var.ir_ref.type, ir.PointerType):
            slot = self._alloca_entry(var.ir_ref.type, ".dvec.view")
            self.builder.store(var.ir_ref, slot)
            dv_ptr = slot
        else:
            dv_ptr = var.ir_ref
        idx = self._gen_expr(idx_expr)
        if idx is None:
            raise CodegenError("dvec index has no value")
        idx = self._coerce(idx, I64)
        len_addr = self.builder.gep(
            dv_ptr,
            [ir.Constant(I32, 0), ir.Constant(I32, DVEC_IDX_LEN)],
            inbounds=True,
        )
        length = self.builder.load(len_addr)
        self._emit_dynamic_bounds_trap(idx, length)
        val = self.builder.call(self._get_dvec_get(info), [dv_ptr, idx])
        return self._read_borrow(val)
