"""Tablets codegen: per-(N, T) monomorphization (`_get_tablets` and
its helper builders), tablets field/method/index access, bounds-
check emitters, and the shared trap helper (`__tuppu_trap`)."""
from __future__ import annotations

from llvmlite import ir

from .. import ast as A
from ._common import (
    CodegenError, TabletsInfo, Variable,
    I1, I8, I32, I64,
)


class TabletsMixin:
    def _gen_tablets_method(
        self, info: TabletsInfo, var: Variable, method: str, args: list[A.Expr],
    ) -> ir.Value | None:
        if not var.is_mut:
            raise CodegenError(f"tablets methods require a mut binding")
        assert self.builder is not None
        ptr = var.ir_ref
        if method == "push":
            if len(args) != 1:
                raise CodegenError("tablets.push takes exactly one argument")
            arg_expr = args[0]
            val = self._gen_expr(arg_expr)
            if val is None:
                raise CodegenError("tablets.push argument has no value")
            val = self._coerce(val, info.elem_ty)
            # Container ownership: a cleanup-bearing value going into
            # the container either transfers from its current owner
            # (owning Ident → remove cleanup), passes through as a
            # fresh-owned rvalue (Call / Copy / StructLit — nobody
            # else aliases it), or gets deep-cloned (any borrow
            # source — Field/Index/StringLit, or an Ident naming a
            # borrow binding). The three-way split keeps
            # push(fresh_call()) as a single malloc.
            deferred_slot: ir.Value | None = None
            if self._is_cleanup_bearing_ty(info.elem_ty):
                if isinstance(arg_expr, A.Ident):
                    # Defer zero so the source slot still reaches the
                    # element chunks through GC during push's chunk
                    # allocation. We zero after push returns.
                    res = self._transfer_cleanup_into_container(
                        arg_expr.name, defer_zero=True,
                    )
                    if res is False:
                        val = self._deep_clone_if_cleanup_bearing(
                            val, for_transfer=True,
                        )
                    else:
                        deferred_slot = res  # type: ignore[assignment]
                elif self._is_borrow_source_expr(arg_expr):
                    val = self._deep_clone_if_cleanup_bearing(
                        val, for_transfer=True,
                    )
                # else: fresh-owned rvalue, transfer by default.
            pushed = self.builder.call(self._get_tablets_push(info), [ptr, val])
            if deferred_slot is not None:
                self._zero_transferred_slot(deferred_slot)
            return pushed
        raise CodegenError(f"tablets has no method {method!r}")

    def _gen_tablets_field(self, var: Variable, field_name: str) -> ir.Value:
        assert self.builder is not None
        if field_name == "len":
            # Works on any tablets binding whose ir_ref is a pointer to
            # the header (both mut and the slot-backed step case covered
            # by tablets-literal bindings). Pure-SSA step bindings (e.g.
            # a tablets-returning fn call stored into a step) would be
            # rare and we can fold them in later.
            if isinstance(var.ir_ref.type, ir.PointerType):
                len_addr = self.builder.gep(
                    var.ir_ref,
                    [ir.Constant(I32, 0), ir.Constant(I32, 2)],
                    inbounds=True,
                )
                return self.builder.load(len_addr)
            # Fall-through: step SSA value, just extract the len field.
            return self.builder.extract_value(var.ir_ref, 2)
        raise CodegenError(f"tablets has no field {field_name!r}; only .len")

    # --- rat arithmetic -------------------------------------------------

    def _gen_tablets_index(
        self, info: TabletsInfo, var: Variable, idx_expr: A.Expr,
    ) -> ir.Value:
        assert self.builder is not None
        # Non-pointer-backed binding (e.g. a non-mut tablets fn param):
        # spill the SSA value to a temp alloca so info.get has a ptr
        # to chain-walk. The spill is a local copy of the metadata
        # only; the chunks still live in the caller's storage, so
        # reads return the current state.
        if not isinstance(var.ir_ref.type, ir.PointerType):
            slot = self._alloca_entry(var.ir_ref.type, ".tbls.view")
            self.builder.store(var.ir_ref, slot)
            t_ptr = slot
        else:
            t_ptr = var.ir_ref
        idx = self._gen_expr(idx_expr)
        if idx is None:
            raise CodegenError("tablets index has no value")
        idx = self._coerce(idx, I64)
        len_addr = self.builder.gep(
            t_ptr,
            [ir.Constant(I32, 0), ir.Constant(I32, 2)],
            inbounds=True,
        )
        length = self.builder.load(len_addr)
        self._emit_dynamic_bounds_trap(idx, length)
        val = self.builder.call(self._get_tablets_get(info), [t_ptr, idx])
        # Reads from a container are borrows — the container owns the
        # bytes; the caller gets a view. Neuter cleanup markers so
        # copying the value (into another struct, another container,
        # etc.) doesn't create a second "owner" that would double-free
        # at release-walk time.
        return self._read_borrow(val)

    def _emit_dynamic_bounds_trap(self, idx: ir.Value, length: ir.Value) -> None:
        assert self.builder is not None
        b = self.builder
        oob_lo = b.icmp_signed("<", idx, ir.Constant(I64, 0))
        oob_hi = b.icmp_signed(">=", idx, length)
        oob = b.or_(oob_lo, oob_hi)
        fn = b.function
        trap_bb = fn.append_basic_block("bounds.trap")
        ok_bb = fn.append_basic_block("bounds.ok")
        b.cbranch(oob, trap_bb, ok_bb)
        b.position_at_end(trap_bb)
        b.call(self._get_trap(), [])
        b.unreachable()
        b.position_at_end(ok_bb)

    def _emit_bounds_trap(self, idx: ir.Value, size: int) -> None:
        assert self.builder is not None
        b = self.builder
        oob_lo = b.icmp_signed("<", idx, ir.Constant(I64, 0))
        oob_hi = b.icmp_signed(">=", idx, ir.Constant(I64, size))
        oob = b.or_(oob_lo, oob_hi)
        fn = b.function
        trap_bb = fn.append_basic_block("bounds.trap")
        ok_bb = fn.append_basic_block("bounds.ok")
        b.cbranch(oob, trap_bb, ok_bb)
        b.position_at_end(trap_bb)
        b.call(self._get_trap(), [])
        b.unreachable()
        b.position_at_end(ok_bb)

    # --- tablets (chained-chunk growable storage) -----------------------

    def _get_tablets(
        self, N: int, elem_ty: ir.Type, elem_is_wedge: bool = False,
    ) -> TabletsInfo:
        """Eagerly register (once, cache thereafter) the type half of
        `tablets[N]T` — `node_ty` and `tablets_ty`. Helper fn bodies
        (`push`, `get`, `get_addr`, `release`) defer to their
        accessor methods, which build-and-cache on first use. Called
        during struct/seal field resolution; at that point a seal
        element type may still be opaque, so committing to the
        chunk descriptor (which encodes `sizeof(T)`) would bake in
        garbage. Lazy helpers dodge that ordering hazard.

        `elem_is_wedge` is True iff the source-level element was
        `wedge T`. LLVM collapses `wedge T` / `*T` / `T*` to the same
        pointer type, so the cache key keeps them distinct: a
        tablets-of-wedge needs chunk slots traced via mark_wedge, a
        tablets-of-raw-ptr does not."""
        key = (N, str(elem_ty), elem_is_wedge)
        existing = self._tablets_types.get(key)
        if existing is not None:
            return existing

        suffix = f"{elem_ty}_{N}".replace(" ", "_").replace("{", "").replace("}", "")
        if elem_is_wedge:
            suffix = f"w_{suffix}"
        node_ty = self.module.context.get_identified_type(f"Node_{suffix}")
        if node_ty.is_opaque:
            node_ty.set_body(
                ir.ArrayType(elem_ty, N),
                I64,
                node_ty.as_pointer(),
            )
        tablets_ty = ir.LiteralStructType([
            node_ty.as_pointer(),
            node_ty.as_pointer(),
            I64,
        ])

        info = TabletsInfo(
            N=N, elem_ty=elem_ty,
            node_ty=node_ty, tablets_ty=tablets_ty,
            suffix=suffix,
            elem_is_wedge=elem_is_wedge,
        )
        self._tablets_types[key] = info
        return info

    def _get_tablets_push(self, info: TabletsInfo) -> ir.Function:
        if info.push is None:
            info.push = self._build_tablets_push(
                info.N, info.elem_ty, info.node_ty, info.tablets_ty, info.suffix,
                elem_is_wedge=info.elem_is_wedge,
            )
        return info.push

    def _get_tablets_get(self, info: TabletsInfo) -> ir.Function:
        if info.get is None:
            info.get = self._build_tablets_get(
                info.N, info.elem_ty, info.node_ty, info.tablets_ty, info.suffix,
            )
        return info.get

    def _get_tablets_get_addr(self, info: TabletsInfo) -> ir.Function:
        if info.get_addr is None:
            info.get_addr = self._build_tablets_get_addr(
                info.N, info.elem_ty, info.node_ty, info.tablets_ty, info.suffix,
            )
        return info.get_addr

    def _get_tablets_release(self, info: TabletsInfo) -> ir.Function:
        if info.release is None:
            info.release = self._build_tablets_release(
                info.N, info.elem_ty, info.node_ty, info.tablets_ty, info.suffix,
            )
        return info.release

    def _build_tablets_push(
        self, N: int, elem_ty: ir.Type,
        node_ty: ir.IdentifiedStructType, tablets_ty: ir.LiteralStructType,
        suffix: str, elem_is_wedge: bool = False,
    ) -> ir.Function:
        # Returns a pointer to the just-pushed element slot — this is
        # the `tablet T` handle the user sees from `tablets.push(...)`.
        fn = ir.Function(
            self.module,
            ir.FunctionType(elem_ty.as_pointer(), [tablets_ty.as_pointer(), elem_ty]),
            name=f"__tuppu_tbls_{suffix}_push",
        )
        fn.args[0].name = "t"
        fn.args[1].name = "value"
        t_ptr, val = fn.args

        ZERO_I32 = ir.Constant(I32, 0)
        ONE_I32  = ir.Constant(I32, 1)
        TWO_I32  = ir.Constant(I32, 2)
        null_node = ir.Constant(node_ty.as_pointer(), None)

        entry      = fn.append_basic_block("entry")
        check_full = fn.append_basic_block("check.full")
        need_new   = fn.append_basic_block("need.new")
        link_head  = fn.append_basic_block("link.head")
        link_tail  = fn.append_basic_block("link.tail")
        do_insert  = fn.append_basic_block("do.insert")

        b = ir.IRBuilder(entry)
        head_addr = b.gep(t_ptr, [ZERO_I32, ZERO_I32], inbounds=True)
        tail_addr = b.gep(t_ptr, [ZERO_I32, ONE_I32],  inbounds=True)
        len_addr  = b.gep(t_ptr, [ZERO_I32, TWO_I32],  inbounds=True)
        tail = b.load(tail_addr)
        tail_is_null = b.icmp_signed("==", tail, null_node)
        b.cbranch(tail_is_null, need_new, check_full)

        b.position_at_end(check_full)
        used_addr_existing = b.gep(tail, [ZERO_I32, ONE_I32], inbounds=True)
        used_existing = b.load(used_addr_existing)
        is_full = b.icmp_signed("==", used_existing, ir.Constant(I64, N))
        b.cbranch(is_full, need_new, do_insert)

        # need.new: allocate a GC-tracked chunk + link into chain.
        # Chunks go through __tuppu_gc_alloc(size, &chunk_desc) so the
        # collector can trace through each slot's pointer fields and
        # the chunk's `next` via the descriptor's ptr_offsets table.
        b.position_at_end(need_new)
        # sizeof(node_ty) via GEP-from-null trick.
        size_ptr = b.gep(null_node, [ONE_I32], inbounds=False)
        node_size = b.ptrtoint(size_ptr, I64)
        chunk_desc = self._get_chunk_type_desc(
            N, elem_ty, node_ty, elem_is_wedge=elem_is_wedge,
        )
        raw = b.call(
            self._get_gc_alloc_typed(),
            [node_size, b.bitcast(chunk_desc, I8.as_pointer())],
        )
        new_node = b.bitcast(raw, node_ty.as_pointer())
        b.store(ir.Constant(I64, 0), b.gep(new_node, [ZERO_I32, ONE_I32], inbounds=True))
        b.store(null_node, b.gep(new_node, [ZERO_I32, TWO_I32], inbounds=True))
        was_empty = b.icmp_signed("==", tail, null_node)
        b.cbranch(was_empty, link_head, link_tail)

        b.position_at_end(link_head)
        b.store(new_node, head_addr)
        b.store(new_node, tail_addr)
        b.branch(do_insert)

        b.position_at_end(link_tail)
        tail_next_addr = b.gep(tail, [ZERO_I32, TWO_I32], inbounds=True)
        b.store(new_node, tail_next_addr)
        b.store(new_node, tail_addr)
        b.branch(do_insert)

        # do.insert: tail is non-null and has room. Write and bump counts.
        b.position_at_end(do_insert)
        cur_tail = b.load(tail_addr)
        used_addr = b.gep(cur_tail, [ZERO_I32, ONE_I32], inbounds=True)
        cur_used = b.load(used_addr)
        slot = b.gep(
            cur_tail,
            [ZERO_I32, ZERO_I32, cur_used],
            inbounds=True,
        )
        b.store(val, slot)
        b.store(b.add(cur_used, ir.Constant(I64, 1)), used_addr)
        cur_len = b.load(len_addr)
        b.store(b.add(cur_len, ir.Constant(I64, 1)), len_addr)
        b.ret(slot)

        return fn

    def _build_tablets_get(
        self, N: int, elem_ty: ir.Type,
        node_ty: ir.IdentifiedStructType, tablets_ty: ir.LiteralStructType,
        suffix: str,
    ) -> ir.Function:
        """Walk the chain (i/N) nodes to find the i-th element."""
        fn = ir.Function(
            self.module,
            ir.FunctionType(elem_ty, [tablets_ty.as_pointer(), I64]),
            name=f"__tuppu_tbls_{suffix}_get",
        )
        fn.args[0].name = "t"
        fn.args[1].name = "idx"
        t_ptr, idx_arg = fn.args

        ZERO_I32 = ir.Constant(I32, 0)

        entry   = fn.append_basic_block("entry")
        loop    = fn.append_basic_block("loop")
        advance = fn.append_basic_block("advance")
        done    = fn.append_basic_block("done")

        b = ir.IRBuilder(entry)
        head = b.load(b.gep(t_ptr, [ZERO_I32, ZERO_I32], inbounds=True))
        b.branch(loop)

        b.position_at_end(loop)
        cur_phi = b.phi(node_ty.as_pointer(), "cur")
        idx_phi = b.phi(I64, "i")
        cur_phi.add_incoming(head, entry)
        idx_phi.add_incoming(idx_arg, entry)
        need_adv = b.icmp_signed(">=", idx_phi, ir.Constant(I64, N))
        b.cbranch(need_adv, advance, done)

        b.position_at_end(advance)
        next_node = b.load(b.gep(cur_phi, [ZERO_I32, ir.Constant(I32, 2)], inbounds=True))
        new_idx = b.sub(idx_phi, ir.Constant(I64, N))
        cur_phi.add_incoming(next_node, advance)
        idx_phi.add_incoming(new_idx, advance)
        b.branch(loop)

        b.position_at_end(done)
        slot = b.gep(cur_phi, [ZERO_I32, ZERO_I32, idx_phi], inbounds=True)
        b.ret(b.load(slot))
        return fn

    def _build_tablets_get_addr(
        self, N: int, elem_ty: ir.Type,
        node_ty: ir.IdentifiedStructType, tablets_ty: ir.LiteralStructType,
        suffix: str,
    ) -> ir.Function:
        """Like `get`, but returns a pointer to the slot instead of
        loading it. Used by the lvalue path so `arr[n].field = v` can
        compute the address of `arr[n]` and GEP through to the field.
        Caller is responsible for bounds-checking; this mirrors `get`
        which doesn't bounds-check either (the checker/emitter at the
        call site does)."""
        fn = ir.Function(
            self.module,
            ir.FunctionType(elem_ty.as_pointer(), [tablets_ty.as_pointer(), I64]),
            name=f"__tuppu_tbls_{suffix}_get_addr",
        )
        fn.args[0].name = "t"
        fn.args[1].name = "idx"
        t_ptr, idx_arg = fn.args

        ZERO_I32 = ir.Constant(I32, 0)

        entry   = fn.append_basic_block("entry")
        loop    = fn.append_basic_block("loop")
        advance = fn.append_basic_block("advance")
        done    = fn.append_basic_block("done")

        b = ir.IRBuilder(entry)
        head = b.load(b.gep(t_ptr, [ZERO_I32, ZERO_I32], inbounds=True))
        b.branch(loop)

        b.position_at_end(loop)
        cur_phi = b.phi(node_ty.as_pointer(), "cur")
        idx_phi = b.phi(I64, "i")
        cur_phi.add_incoming(head, entry)
        idx_phi.add_incoming(idx_arg, entry)
        need_adv = b.icmp_signed(">=", idx_phi, ir.Constant(I64, N))
        b.cbranch(need_adv, advance, done)

        b.position_at_end(advance)
        next_node = b.load(b.gep(cur_phi, [ZERO_I32, ir.Constant(I32, 2)], inbounds=True))
        new_idx = b.sub(idx_phi, ir.Constant(I64, N))
        cur_phi.add_incoming(next_node, advance)
        idx_phi.add_incoming(new_idx, advance)
        b.branch(loop)

        b.position_at_end(done)
        slot = b.gep(cur_phi, [ZERO_I32, ZERO_I32, idx_phi], inbounds=True)
        b.ret(slot)
        return fn

    def _build_tablets_release(
        self, N: int, elem_ty: ir.Type,
        node_ty: ir.IdentifiedStructType, tablets_ty: ir.LiteralStructType,
        suffix: str,
    ) -> ir.Function:
        # When the element type is cleanup-bearing (str / struct-with-
        # cleanup / nested tablets), the tablets release walks each
        # chunk's occupied slots and dispatches the appropriate release
        # on each, then frees the chunk. This ensures `mut t:
        # tablets[N]str` with heap-owned str elements doesn't leak its
        # bytes at scope exit. Ownership-transfer semantics at push /
        # struct-lit sites make the caller's cleanup no-op, so the
        # element-walk here can safely free without risking double-free.
        element_release = self._element_release_fn(elem_ty)

        fn = ir.Function(
            self.module,
            ir.FunctionType(ir.VoidType(), [tablets_ty.as_pointer()]),
            name=f"__tuppu_tbls_{suffix}_release",
        )
        fn.args[0].name = "t"
        t_ptr = fn.args[0]

        ZERO_I32 = ir.Constant(I32, 0)
        null_node = ir.Constant(node_ty.as_pointer(), None)

        entry = fn.append_basic_block("entry")
        loop  = fn.append_basic_block("loop")
        body  = fn.append_basic_block("body")
        done  = fn.append_basic_block("done")

        b = ir.IRBuilder(entry)
        head_addr = b.gep(t_ptr, [ZERO_I32, ZERO_I32], inbounds=True)
        tail_addr = b.gep(t_ptr, [ZERO_I32, ir.Constant(I32, 1)], inbounds=True)
        len_addr  = b.gep(t_ptr, [ZERO_I32, ir.Constant(I32, 2)], inbounds=True)
        head = b.load(head_addr)
        b.branch(loop)

        b.position_at_end(loop)
        cur_phi = b.phi(node_ty.as_pointer(), "cur")
        cur_phi.add_incoming(head, entry)
        is_null = b.icmp_signed("==", cur_phi, null_node)
        b.cbranch(is_null, done, body)

        b.position_at_end(body)

        # Per-element release walk for cleanup-bearing element types.
        # Iterates `used` slots of the current chunk; each slot gets
        # the appropriate release. Scalar elements skip this entirely
        # and we just free the chunk as before.
        if element_release is not None:
            used = b.load(b.gep(
                cur_phi, [ZERO_I32, ir.Constant(I32, 1)], inbounds=True,
            ))
            rel_loop = fn.append_basic_block("rel.loop")
            rel_body = fn.append_basic_block("rel.body")
            rel_done = fn.append_basic_block("rel.done")
            start_bb = b.block
            b.branch(rel_loop)

            b.position_at_end(rel_loop)
            i_phi = b.phi(I64, "i")
            i_phi.add_incoming(ir.Constant(I64, 0), start_bb)
            cont = b.icmp_signed("<", i_phi, used)
            b.cbranch(cont, rel_body, rel_done)

            b.position_at_end(rel_body)
            slot = b.gep(
                cur_phi, [ZERO_I32, ZERO_I32, i_phi], inbounds=True,
            )
            b.call(element_release, [slot])
            i_next = b.add(i_phi, ir.Constant(I64, 1))
            i_phi.add_incoming(i_next, b.block)
            b.branch(rel_loop)

            b.position_at_end(rel_done)

        next_node = b.load(b.gep(cur_phi, [ZERO_I32, ir.Constant(I32, 2)], inbounds=True))
        raw = b.bitcast(cur_phi, I8.as_pointer())
        b.call(self._get_free(), [raw])
        cur_phi.add_incoming(next_node, b.block)
        b.branch(loop)

        b.position_at_end(done)
        b.store(null_node, head_addr)
        b.store(null_node, tail_addr)
        b.store(ir.Constant(I64, 0), len_addr)
        b.ret_void()
        return fn

    def _get_tablets_clone(self, info: TabletsInfo) -> ir.Function:
        """Lazily build + cache the deep-clone helper for a tablets type.
        Clone walks the source chain, clones each element if it's
        cleanup-bearing (str/struct/nested-tablets), and pushes into a
        fresh tablets. Result is returned by value — caller allocas the
        slot. Used by struct-clone when a field is tablets-typed and
        by the push path when the source is a borrow."""
        if info.clone is not None:
            return info.clone
        info.clone = self._build_tablets_clone(
            info.N, info.elem_ty, info.node_ty, info.tablets_ty,
            self._get_tablets_push(info), info.suffix,
        )
        return info.clone

    def _build_tablets_clone(
        self, N: int, elem_ty: ir.Type,
        node_ty: ir.IdentifiedStructType, tablets_ty: ir.LiteralStructType,
        push: ir.Function, suffix: str,
    ) -> ir.Function:
        fn = ir.Function(
            self.module,
            ir.FunctionType(tablets_ty, [tablets_ty.as_pointer()]),
            name=f"__tuppu_tbls_{suffix}_clone",
        )
        fn.args[0].name = "src"
        src_ptr = fn.args[0]

        ZERO_I32 = ir.Constant(I32, 0)
        null_node = ir.Constant(node_ty.as_pointer(), None)

        entry    = fn.append_basic_block("entry")
        chunks   = fn.append_basic_block("chunks")
        chunk_bd = fn.append_basic_block("chunk.body")
        slots    = fn.append_basic_block("slots")
        slot_bd  = fn.append_basic_block("slot.body")
        advance  = fn.append_basic_block("advance")
        done     = fn.append_basic_block("done")

        b = ir.IRBuilder(entry)
        # Fresh dest tablets lives on the stack for the duration of
        # this fn; we load-and-return it at the end.
        dest_slot = b.alloca(tablets_ty, name="dest")
        b.store(ir.Constant(tablets_ty, None), dest_slot)
        # Per-iteration spill slot for the cloned element, hoisted
        # to entry so the loop body reuses one alloca instead of
        # growing the stack each iteration.
        elem_desc = self._get_type_desc(elem_ty)
        elem_spill: ir.Value | None = None
        if elem_desc is not None:
            elem_spill = b.alloca(elem_ty, name=".elem.spill")
        # Helper-fn body emission: isolate from caller's cleanup state
        # so deep_clone's chokepoint doesn't pollute the caller. Root
        # dest_slot so accumulated element clones stay reachable if a
        # later alloc triggers GC. Paired pop emitted at `done`.
        saved_frames = self._cleanup_frames
        saved_counts = self._gc_root_counts
        saved_helper = getattr(self, "_in_helper_emission", False)
        saved_builder = self.builder
        self._cleanup_frames = [[]]
        self._gc_root_counts = [0]
        self._in_helper_emission = True
        self.builder = b
        dest_desc = self._get_type_desc(tablets_ty)
        dest_pushed = False
        if dest_desc is not None:
            b.call(
                self._get_gc_push_root(),
                [
                    b.bitcast(dest_slot, I8.as_pointer()),
                    b.bitcast(dest_desc, I8.as_pointer()),
                ],
            )
            dest_pushed = True
        head = b.load(b.gep(src_ptr, [ZERO_I32, ZERO_I32], inbounds=True))
        b.branch(chunks)

        b.position_at_end(chunks)
        cur_phi = b.phi(node_ty.as_pointer(), "cur")
        cur_phi.add_incoming(head, entry)
        is_null = b.icmp_signed("==", cur_phi, null_node)
        b.cbranch(is_null, done, chunk_bd)

        b.position_at_end(chunk_bd)
        used = b.load(b.gep(
            cur_phi, [ZERO_I32, ir.Constant(I32, 1)], inbounds=True,
        ))
        b.branch(slots)

        b.position_at_end(slots)
        i_phi = b.phi(I64, "i")
        i_phi.add_incoming(ir.Constant(I64, 0), chunk_bd)
        cont = b.icmp_signed("<", i_phi, used)
        b.cbranch(cont, slot_bd, advance)

        b.position_at_end(slot_bd)
        slot_ptr = b.gep(
            cur_phi, [ZERO_I32, ZERO_I32, i_phi], inbounds=True,
        )
        elem = b.load(slot_ptr)
        saved = self.builder
        self.builder = b
        try:
            cloned = self._deep_clone_if_cleanup_bearing(elem)
        finally:
            self.builder = saved
        # Protect the cloned element across push's internal chunk
        # allocation. Without this root, a stress-mode collect fired
        # inside push would reclaim the freshly cloned bytes before
        # the dest chunk holds them. Popped immediately after push
        # returns so the loop-iteration push/pop pair balances.
        if elem_spill is not None:
            b.store(cloned, elem_spill)
            b.call(
                self._get_gc_push_root(),
                [
                    b.bitcast(elem_spill, I8.as_pointer()),
                    b.bitcast(elem_desc, I8.as_pointer()),
                ],
            )
            cloned = b.load(elem_spill)
        b.call(push, [dest_slot, cloned])
        if elem_spill is not None:
            b.call(self._get_gc_pop_roots(), [ir.Constant(I64, 1)])
        i_next = b.add(i_phi, ir.Constant(I64, 1))
        i_phi.add_incoming(i_next, b.block)
        b.branch(slots)

        b.position_at_end(advance)
        next_node = b.load(b.gep(
            cur_phi, [ZERO_I32, ir.Constant(I32, 2)], inbounds=True,
        ))
        cur_phi.add_incoming(next_node, b.block)
        b.branch(chunks)

        b.position_at_end(done)
        if dest_pushed:
            b.call(self._get_gc_pop_roots(), [ir.Constant(I64, 1)])
        b.ret(b.load(dest_slot))
        self._cleanup_frames = saved_frames
        self._gc_root_counts = saved_counts
        self._in_helper_emission = saved_helper
        self.builder = saved_builder
        return fn

    def _element_release_fn(self, elem_ty: ir.Type) -> "ir.Function | None":
        """Return the per-slot release fn for a tablets element type,
        or None if the element has no cleanup (plain scalars, rats,
        sexes, struct-without-cleanup). The returned fn takes a
        pointer-to-element; it's the same shape the struct and str
        release fns already have."""
        if self._is_str_value(elem_ty):
            return self._get_str_release()
        inner = self._tablets_info_for(elem_ty)
        if inner is not None:
            return inner.release
        if (
            self._struct_fields_for(elem_ty) is not None
            and self._struct_needs_cleanup(elem_ty)
        ):
            return self._get_struct_release(elem_ty)
        if (
            self._seal_key_for_ty(elem_ty) is not None
            and self._seal_needs_cleanup(elem_ty)
        ):
            return self._get_seal_release(elem_ty)
        return None

    def _tablets_info_for(self, value_ty: ir.Type) -> TabletsInfo | None:
        """Given an LLVM type, return the TabletsInfo if it's a tablets struct."""
        for info in self._tablets_types.values():
            if info.tablets_ty == value_ty:
                return info
        return None

    def _get_trap(self) -> ir.Function:
        if self._trap is None:
            self._trap = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), []),
                name="llvm.trap",
            )
        return self._trap

