"""Statement codegen: `_gen_stmt` dispatcher, loop handlers (`_gen_for*`,
`_gen_while`, `_emit_counted_loop`), `_gen_yield`, `_gen_release`,
`_gen_binding`, and the cleanup-frame / GC-root / struct + seal
clone/release machinery that binding and yield both depend on.
Extracted from `codegen/__init__.py` as `StmtMixin`."""
from __future__ import annotations

from llvmlite import ir

from .. import ast as A
from ._common import (
    CodegenError, TabletsInfo, Variable,
    I1, I8, I16, I32, I64,
)


class StmtMixin:
    # --- statements ---

    def _gen_stmt(self, s: A.Stmt) -> None:
        self._current_loc = (getattr(s, "line", 0), getattr(s, "col", 0))
        if isinstance(s, A.Binding):
            self._gen_binding(s); return
        if isinstance(s, A.Assign):
            self._gen_assign(s); return
        if isinstance(s, A.While):
            self._gen_while(s); return
        if isinstance(s, A.ForStmt):
            self._gen_for(s); return
        if isinstance(s, A.YieldStmt):
            self._gen_yield(s); return
        if isinstance(s, A.ReleaseStmt):
            self._gen_release(s); return
        if isinstance(s, A.ExprStmt):
            # The `_gen_expr` chokepoint has already rooted any
            # cleanup-bearing rvalue produced by `s.expr`; discard
            # the value.
            self._gen_expr(s.expr)
            return
        raise CodegenError(f"statement not supported yet: {type(s).__name__}")

    def _gen_release(self, s: A.ReleaseStmt) -> None:
        var = self._lookup(s.name)
        info = self._tablets_info_for(var.value_ty)
        if info is None:
            raise CodegenError(f"release requires a tablets, got {var.value_ty}")
        if not var.is_mut:
            raise CodegenError(f"cannot release step-bound tablets {s.name!r}")
        assert self.builder is not None
        self.builder.call(self._get_tablets_release(info), [var.ir_ref])
        # Remove this variable from its cleanup frame so the auto-
        # release at scope exit doesn't double-free. We walk frames
        # outermost-in since explicit release can target an outer
        # binding shadowed by an inner one (unusual but legal).
        for frame in reversed(self._cleanup_frames):
            for i, (_fn, _ptr, name) in enumerate(frame):
                if name == s.name:
                    frame.pop(i)
                    return

    def _gen_for(self, f: A.ForStmt) -> None:
        """Generate a `for name in iter { body }` loop.

        Three iterable shapes are supported; each picks a different loop
        body:

        - **str**: walk 0..len, load s.ptr[i] as u8.
        - **tablets[N]T**: walk the chain via the cached `get` helper.
        - **table**: walk the global array in memory order.

        The loop variable is bound as a fresh `step` (SSA) per iteration
        so it cannot be assigned inside the body."""
        assert self.builder is not None

        # Comptime table iteration — recognise the table by name before
        # we try to produce a value for the iter expression.
        if isinstance(f.iter, A.Ident) and f.iter.name in self._tables:
            self._gen_for_table(f, f.iter.name)
            return

        iter_val = self._gen_expr(f.iter)
        if iter_val is None:
            raise CodegenError("for: iter expression has no value")

        if self._is_str_value(iter_val.type):
            self._gen_for_str(f, iter_val)
            return

        info = self._tablets_info_for(iter_val.type)
        if info is not None:
            self._gen_for_tablets(f, iter_val, info)
            return

        if self._is_ivec_value(iter_val.type):
            elem_ty = self._ivec_elem_for_for(f)
            if elem_ty is not None:
                iv_info = self._get_ivec(elem_ty)
                self._gen_for_ivec(f, iter_val, iv_info)
                return

        if self._is_dvec_value(iter_val.type):
            elem_ty = self._dvec_elem_for_for(f)
            if elem_ty is not None:
                dv_info = self._get_dvec(elem_ty)
                self._gen_for_dvec(f, iter_val, dv_info)
                return

        raise CodegenError(
            f"for: cannot iterate over value of type {iter_val.type}"
        )

    def _gen_for_ivec(
        self, f: A.ForStmt, iv_val: ir.Value, info,
    ) -> None:
        """Iterate over an ivec value via the cached `get` helper.
        Spills the ivec value to a temp alloca so we can pass an
        address to `get`."""
        assert self.builder is not None
        from ._common import IVEC_STRUCT, IVEC_IDX_LEN
        slot = self._alloca_entry(IVEC_STRUCT, "for.iv")
        self.builder.store(iv_val, slot)
        len_addr = self.builder.gep(
            slot,
            [ir.Constant(I32, 0), ir.Constant(I32, IVEC_IDX_LEN)],
            inbounds=True,
        )
        length = self.builder.load(len_addr)
        get_fn = self._get_ivec_get(info)
        self._emit_counted_loop(
            length,
            lambda i: self.builder.call(get_fn, [slot, i]),
            f,
        )

    def _gen_for_dvec(
        self, f: A.ForStmt, dv_val: ir.Value, info,
    ) -> None:
        assert self.builder is not None
        from ._common import DVEC_STRUCT, DVEC_IDX_LEN
        slot = self._alloca_entry(DVEC_STRUCT, "for.dv")
        self.builder.store(dv_val, slot)
        len_addr = self.builder.gep(
            slot,
            [ir.Constant(I32, 0), ir.Constant(I32, DVEC_IDX_LEN)],
            inbounds=True,
        )
        length = self.builder.load(len_addr)
        get_fn = self._get_dvec_get(info)
        self._emit_counted_loop(
            length,
            lambda i: self.builder.call(get_fn, [slot, i]),
            f,
        )

    def _gen_for_str(self, f: A.ForStmt, str_val: ir.Value) -> None:
        """Lower `for c in s { body }` — c is u8, bounds-safe by construction
        since we walk 0..len."""
        assert self.builder is not None
        ptr = self.builder.extract_value(str_val, 0)
        length = self.builder.extract_value(str_val, 1)
        self._emit_counted_loop(
            length,
            lambda i: self.builder.load(
                self.builder.gep(ptr, [i], inbounds=True),
            ),
            f,
        )

    def _gen_for_tablets(
        self, f: A.ForStmt, tbl_val: ir.Value, info: "TabletsInfo",
    ) -> None:
        """Iterate over a tablets value. We reuse the cached `get` helper;
        mem2reg + the existing optimizer clean up the redundant chain walks
        for dense access patterns."""
        assert self.builder is not None
        # tbl_val is a value (loaded struct). We need an address to pass
        # to the get helper, so spill it to a temp alloca.
        slot = self._alloca_entry(info.tablets_ty, "for.tbl")
        self.builder.store(tbl_val, slot)
        len_addr = self.builder.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, 2)], inbounds=True,
        )
        length = self.builder.load(len_addr)
        get_fn = self._get_tablets_get(info)
        self._emit_counted_loop(
            length,
            lambda i: self.builder.call(get_fn, [slot, i]),
            f,
        )

    def _gen_for_table(self, f: A.ForStmt, name: str) -> None:
        """Walk a compile-time table in declaration order (element index
        0..size-1), regardless of the table's `lo` bound."""
        assert self.builder is not None
        g, size, _lo, _elem_ty = self._tables[name]
        length = ir.Constant(I64, size)
        zero = ir.Constant(I32, 0)

        def load_at(i: ir.Value) -> ir.Value:
            return self.builder.load(
                self.builder.gep(g, [zero, i], inbounds=True),
            )

        self._emit_counted_loop(length, load_at, f)

    def _emit_counted_loop(
        self,
        length: ir.Value,
        load_element,
        f: A.ForStmt,
    ) -> None:
        """Emit the shared 0..length loop skeleton and bind `f.name` to the
        current element inside the body. `load_element(i_i64)` returns the
        value to bind."""
        assert self.builder is not None
        fn = self.builder.function
        header = fn.append_basic_block("for.header")
        body = fn.append_basic_block("for.body")
        exit_ = fn.append_basic_block("for.exit")

        i_slot = self._alloca_entry(I64, "for.i")
        self.builder.store(ir.Constant(I64, 0), i_slot)
        self.builder.branch(header)

        self.builder.position_at_end(header)
        i_val = self.builder.load(i_slot)
        cond = self.builder.icmp_signed("<", i_val, length)
        self.builder.cbranch(cond, body, exit_)

        self.builder.position_at_end(body)
        element = load_element(i_val)
        self.scopes.append({})
        try:
            self._bind(f.name, Variable(
                is_mut=False, ir_ref=element, value_ty=element.type,
            ))
            self._gen_block(f.body)
        finally:
            self.scopes.pop()
        if not self._is_terminated():
            next_i = self.builder.add(i_val, ir.Constant(I64, 1))
            self.builder.store(next_i, i_slot)
            self.builder.branch(header)

        self.builder.position_at_end(exit_)

    def _gen_while(self, w: A.While) -> None:
        assert self.builder is not None
        fn = self.builder.function
        header = fn.append_basic_block("while.header")
        body = fn.append_basic_block("while.body")
        exit_ = fn.append_basic_block("while.exit")

        self.builder.branch(header)

        self.builder.position_at_end(header)
        # Any chokepoint roots the cond expression pushes are scoped
        # to this iteration's cond evaluation — they shouldn't leak
        # into the body/exit paths since each iteration re-runs the
        # header. Snapshot the counter, pop the delta before the
        # cbranch so cond-eval roots balance per iteration.
        cond_before = (
            self._gc_root_counts[-1] if self._gc_root_counts else 0
        )
        cond = self._gen_expr(w.cond)
        if cond is None or cond.type != I1:
            raise CodegenError("while condition must be a bool expression")
        cond_delta = (
            (self._gc_root_counts[-1] - cond_before)
            if self._gc_root_counts else 0
        )
        if cond_delta > 0:
            self._emit_gc_pop_roots(cond_delta)
            self._gc_root_counts[-1] = cond_before
        self.builder.cbranch(cond, body, exit_)

        self.builder.position_at_end(body)
        self._gen_block(w.body)
        if not self._is_terminated():
            self.builder.branch(header)

        self.builder.position_at_end(exit_)

    def _gen_yield(self, y: A.YieldStmt) -> None:
        assert self.builder is not None
        ret_ty = self.builder.function.ftype.return_type
        if y.value is None:
            self._emit_all_cleanups_for_early_return()
            if isinstance(ret_ty, ir.VoidType):
                self.builder.ret_void()
            else:
                raise CodegenError("bare yield in non-void function")
            return
        val = self._gen_expr(y.value)
        if val is None:
            raise CodegenError("yield value diverged")
        coerced = self._coerce(val, ret_ty)
        # Pre-GC, this site deep-cloned cleanup-bearing Field/Index
        # returns to dodge UAF: the local source struct would be
        # released at frame pop, freeing bytes the caller had just
        # received. Under the current GC that's no longer a hazard —
        # the returned value's type descriptor (e.g. str.ptr → byte
        # buffer) keeps the underlying allocation reachable from the
        # caller's binding. The wedge-handle escape rule is what
        # still rules out genuinely dangling returns; everything
        # else just rides the shadow stack.
        self._emit_all_cleanups_for_early_return()
        self.builder.ret(coerced)

    def _read_borrow(self, val: ir.Value) -> ir.Value:
        """Neuter cleanup markers on a value read from a container or
        aggregate — str gets cap=0, struct-with-cleanup gets all its
        cleanup-bearing fields zeroed (recursively). Represents the
        "borrow view" of the read: the underlying container still owns
        the bytes; the caller sees a view that won't double-free when
        copied, compared, or passed along."""
        if self._is_str_value(val.type):
            return self._str_as_borrow(val)
        if (
            self._struct_fields_for(val.type) is not None
            and self._struct_needs_cleanup(val.type)
        ):
            return self._struct_as_borrow(val, val.type)
        return val

    def _register_gc_root(self, slot: ir.Value, value_ty: ir.Type) -> None:
        """If `value_ty` has traceable pointer fields, emit a
        `__tuppu_gc_push_root(slot, &type_desc)` call at the current
        IRBuilder position and bump the innermost cleanup frame's
        pending-root count so the matching `pop_roots(n)` at frame
        exit balances out. Safe to call for types with no pointer
        fields — the emit is skipped and nothing is counted."""
        if not self._cleanup_frames:
            return
        pushed = self._emit_gc_push_root(slot, value_ty)
        if pushed:
            # Parallel counter: one entry per cleanup frame tracking
            # how many roots were pushed within it.
            while len(self._gc_root_counts) < len(self._cleanup_frames):
                self._gc_root_counts.append(0)
            self._gc_root_counts[-1] += 1

    def _push_cleanup_frame(self) -> None:
        """Push a fresh cleanup frame + GC-root counter in lockstep."""
        self._cleanup_frames.append([])
        self._gc_root_counts.append(0)

    def _emit_gc_frame_pop(self) -> None:
        """Emit `__tuppu_gc_pop_roots(n)` for the innermost cleanup
        frame's accumulated push count, WITHOUT mutating the
        Python-side counter. Callers emit this just before closing
        the basic block (branch / ret) so the pop lands in the live
        control-flow path. The subsequent `_pop_cleanup_frame` still
        tears down the Python-side bookkeeping."""
        if (
            self._gc_root_counts
            and self._gc_root_counts[-1] > 0
            and self.builder is not None
            and not self.builder.block.is_terminated
        ):
            self._emit_gc_pop_roots(self._gc_root_counts[-1])

    def _pop_cleanup_frame(self) -> None:
        """Tear down the innermost cleanup frame's Python-side state.
        If the current block is still open and no matching
        `_emit_gc_frame_pop` call has landed, also emit the
        balancing `pop_roots(n)` IR. Already-terminated blocks skip
        the emit; the caller was responsible for placing the pop
        inline before the terminator."""
        if self._gc_root_counts:
            n = self._gc_root_counts.pop()
            if (
                n > 0
                and self.builder is not None
                and not self.builder.block.is_terminated
            ):
                self._emit_gc_pop_roots(n)
        self._cleanup_frames.pop()

    def _emit_all_gc_root_pops_for_early_return(self) -> None:
        """At an early-return site, unwind every currently-active
        GC root in one shot. Doesn't mutate `_gc_root_counts` — the
        Python-side bookkeeping is still needed to satisfy the
        normal per-frame pops that fire when the (now-dead) body
        code path is still walked by codegen. LLVM optimizes the
        dead post-ret IR away."""
        total = sum(self._gc_root_counts)
        if total > 0 and self.builder is not None:
            self._emit_gc_pop_roots(total)

    def _deep_clone_if_cleanup_bearing(self, val: ir.Value) -> ir.Value:
        """Deep-clone `val` if it's a cleanup-bearing type — str, a
        user struct whose fields recursively require cloning, a
        tablets value, or a seal carrying cleanup-bearing payload.
        Scalars pass through unchanged. When a clone actually
        happens, the fresh heap bytes are routed through the same
        chokepoint as Call results — spilled to a rooted slot so a
        subsequent allocating op can't reclaim them before the
        consumer stores or transfers the value."""
        assert self.builder is not None
        if self._is_str_value(val.type):
            cloned = self.builder.call(self._get_str_clone(), [val])
            return self._force_root_cleanup_value(cloned)
        if (
            self._struct_fields_for(val.type) is not None
            and self._struct_needs_cleanup(val.type)
        ):
            cloned = self.builder.call(
                self._get_struct_clone(val.type), [val],
            )
            return self._force_root_cleanup_value(cloned)
        info = self._tablets_info_for(val.type)
        if info is not None:
            # Tablets clone takes a pointer; spill the SSA to a temp.
            src_slot = self._alloca_entry(val.type, ".tbls.clone.src")
            self.builder.store(val, src_slot)
            cloned = self.builder.call(
                self._get_tablets_clone(info), [src_slot],
            )
            return self._force_root_cleanup_value(cloned)
        if (
            self._seal_key_for_ty(val.type) is not None
            and self._seal_needs_cleanup(val.type)
        ):
            cloned = self.builder.call(
                self._get_seal_clone(val.type), [val],
            )
            return self._force_root_cleanup_value(cloned)
        return val

    def _emit_all_cleanups_for_early_return(self) -> None:
        """Emit release calls for every live cleanup frame in the
        current function, innermost first. Used by yield to unwind
        before the ret. Also emits a cumulative GC pop_roots over
        every live frame so the shadow stack balance matches the
        runtime call path."""
        for frame in reversed(self._cleanup_frames):
            self._emit_frame_cleanups(frame)
        self._emit_all_gc_root_pops_for_early_return()

    def _gen_binding(self, b: A.Binding) -> None:
        # Uninitialized mut binding with explicit type: zero-initialize.
        if b.init is None:
            assert b.is_mut and b.type_ann is not None  # parser enforces this
            ty = self._lower_type(b.type_ann)
            slot = self._alloca_entry(ty, b.name)
            assert self.builder is not None
            self.builder.store(ir.Constant(ty, None), slot)
            self._bind(b.name, Variable(is_mut=True, ir_ref=slot, value_ty=ty))
            self._maybe_register_cleanup(b.name, ty, slot)
            return

        # Tablets literal as initializer: `_gen_tablets_lit_addr` already
        # alloca'd a slot, pushed elements, and registered a cleanup.
        # Reuse that slot as the binding's storage — creating a second
        # alloca would double-register cleanup and cause a double free.
        if isinstance(b.init, A.TabletsLit):
            slot = self._gen_tablets_lit_addr(b.init)
            assert self.builder is not None
            value_ty = slot.type.pointee
            # Rename the anonymous cleanup entry for readable IR.
            if self._cleanup_frames and self._cleanup_frames[-1]:
                fn_rel, _ptr, _old = self._cleanup_frames[-1][-1]
                self._cleanup_frames[-1][-1] = (fn_rel, slot, b.name)
            # Step-bound tablets keeps pointer semantics too — reads go
            # through the slot so `nums.len` and `nums[i]` don't need a
            # mut binding. Reassignment is still rejected at typecheck
            # / parse level.
            self._bind(b.name, Variable(
                is_mut=b.is_mut, ir_ref=slot, value_ty=value_ty,
            ))
            return

        init_val = self._gen_expr(b.init)
        if init_val is None:
            raise CodegenError(f"binding {b.name!r} has no value (initializer diverged)")
        if b.type_ann is not None:
            expected = self._lower_type(b.type_ann)
            init_val = self._coerce(init_val, expected)
        if b.is_mut:
            slot = self._alloca_entry(init_val.type, b.name)
            assert self.builder is not None
            self.builder.store(init_val, slot)
            self._bind(b.name, Variable(is_mut=True, ir_ref=slot, value_ty=init_val.type))
            self._maybe_register_cleanup(b.name, init_val.type, slot)
        else:
            # Step-bound cleanup-bearing values need a slot so the
            # scope-exit release can see them. Covers the built-in str
            # and any user struct that transitively holds cleanup-
            # bearing fields. The SSA value stays the read path (reads
            # remain direct, reassignment impossible), the slot exists
            # purely for release dispatch.
            #
            # `step x = y` (Ident-init) is a BORROW: x shares y's
            # heap bytes, y already owns, registering x would
            # double-free at scope exit. Skip cleanup; record
            # `transfer_on_tail` so if x flows out as a block-tail
            # expression, we transfer ownership of the underlying
            # owner instead of x itself. Field-init (`step x = r.name`)
            # follows the same reasoning — the enclosing struct owns,
            # x is a borrow — but there's no single Variable to
            # transfer from; ownership stays with the struct.
            needs_cleanup = (
                self._is_str_value(init_val.type)
                or (
                    self._struct_fields_for(init_val.type) is not None
                    and self._struct_needs_cleanup(init_val.type)
                )
                or (
                    self._seal_key_for_ty(init_val.type) is not None
                    and self._seal_needs_cleanup(init_val.type)
                )
            )
            transfer_on_tail = None
            # Indexing a container yields a borrow of the container's
            # element — same semantic as Ident/Field reads. Registering
            # cleanup on the binding would double-free against the
            # container's own release on scope exit.
            is_borrow_init = isinstance(
                b.init, (A.Ident, A.Field, A.Index, A.StringLit),
            )
            if needs_cleanup and not is_borrow_init:
                assert self.builder is not None
                cleanup_slot = self._alloca_entry(init_val.type, f"{b.name}.cleanup")
                self.builder.store(init_val, cleanup_slot)
                self._maybe_register_cleanup(b.name, init_val.type, cleanup_slot)
            elif needs_cleanup and isinstance(b.init, A.Ident):
                # Redirect tail-transfer to the source. If the source is
                # itself a borrow, chain through; if it's a param or
                # untracked binding, transfer_on_tail stays None and the
                # borrowed value leaves as-is (safe when the true owner
                # lives in an outer scope).
                try:
                    src_var = self._lookup(b.init.name)
                except CodegenError:
                    src_var = None
                if src_var is not None:
                    if src_var.transfer_on_tail is not None:
                        transfer_on_tail = src_var.transfer_on_tail
                    elif self._frame_has_entry(b.init.name):
                        transfer_on_tail = b.init.name
            self._bind(b.name, Variable(
                is_mut=False, ir_ref=init_val, value_ty=init_val.type,
                transfer_on_tail=transfer_on_tail,
            ))

    def _maybe_register_cleanup(
        self, name: str, value_ty: ir.Type, slot: ir.Value,
    ) -> None:
        """If `value_ty` is a cleanup-having type, record a release call
        for the innermost cleanup frame so it fires automatically at
        scope exit. Handled: tablets, the built-in str, user structs
        that (transitively) hold any of those, and seals whose
        variants carry cleanup-bearing payloads.

        Also emits a GC root push for the slot if the type has
        pointer fields. Tracked count is popped at frame exit."""
        if not self._cleanup_frames:
            return
        info = self._tablets_info_for(value_ty)
        if info is not None:
            self._cleanup_frames[-1].append(
                (self._get_tablets_release(info), slot, name),
            )
            self._register_gc_root(slot, value_ty)
            return
        if self._is_ivec_value(value_ty):
            # ivec has no manual release — GC handles the storage and
            # per-element heap allocs. We still need to root the slot
            # so the buf pointer stays reachable across collections.
            self._register_gc_root(slot, value_ty)
            return
        if self._is_dvec_value(value_ty):
            # dvec is the same story as ivec: no release fn, but the
            # buf pointer needs rooting so the inline T storage stays
            # reachable across collections.
            self._register_gc_root(slot, value_ty)
            return
        if self._is_str_value(value_ty):
            self._cleanup_frames[-1].append(
                (self._get_str_release(), slot, name),
            )
            self._register_gc_root(slot, value_ty)
            return
        if (
            self._struct_fields_for(value_ty) is not None
            and self._struct_needs_cleanup(value_ty)
        ):
            self._cleanup_frames[-1].append(
                (self._get_struct_release(value_ty), slot, name),
            )
            # Struct/seal bindings need shadow-stack rooting too —
            # without it, any heap the struct transitively holds
            # (e.g. a tablets field's chunks) becomes unreachable
            # between the binding's own allocations. Under stress
            # mode the chunks get collected mid-function.
            self._register_gc_root(slot, value_ty)
            return
        if (
            self._seal_key_for_ty(value_ty) is not None
            and self._seal_needs_cleanup(value_ty)
        ):
            self._cleanup_frames[-1].append(
                (self._get_seal_release(value_ty), slot, name),
            )
            self._register_gc_root(slot, value_ty)

    def _struct_needs_cleanup(self, struct_ty: ir.Type) -> bool:
        """Does this user struct (transitively) hold any cleanup-bearing
        fields? Walks the declared field list — str, tablets, nested
        user structs that themselves need cleanup, and seals that carry
        cleanup-bearing payloads all count. Pointer / handle fields
        don't — they borrow into some other storage whose owner does
        the release."""
        fields = self._struct_fields_for(struct_ty)
        if fields is None:
            return False
        for _name, fty in fields:
            if self._is_str_value(fty):
                return True
            if self._tablets_info_for(fty) is not None:
                return True
            if self._struct_fields_for(fty) is not None:
                if self._struct_needs_cleanup(fty):
                    return True
            if (
                self._seal_key_for_ty(fty) is not None
                and self._seal_needs_cleanup(fty)
            ):
                return True
        return False

    def _seal_needs_cleanup(self, seal_ty: ir.Type) -> bool:
        """Does this seal (transitively) hold any cleanup-bearing
        payload fields? Walks each variant's payload tuple looking for
        str / tablets / nested struct-with-cleanup / nested seal-with-
        cleanup. A seal with only nullary variants or scalar payloads
        returns False — no release fn needed."""
        seal_key = self._seal_key_for_ty(seal_ty)
        if seal_key is None:
            return False
        variants = self._seal_variants.get(seal_key)
        if variants is None:
            return False
        for _vname, payload_ty in variants:
            for fty in payload_ty.elements:
                if self._is_str_value(fty):
                    return True
                if self._tablets_info_for(fty) is not None:
                    return True
                if (
                    self._struct_fields_for(fty) is not None
                    and self._struct_needs_cleanup(fty)
                ):
                    return True
                if (
                    self._seal_key_for_ty(fty) is not None
                    and self._seal_needs_cleanup(fty)
                ):
                    return True
        return False

    def _struct_as_borrow(
        self, val: ir.Value, struct_ty: ir.Type,
    ) -> ir.Value:
        """Produce a view of `val` with every str field forced to cap=0
        and every nested cleanup struct recursively neutered. Tablets
        and seal fields are left intact — neither has a cap-style
        sentinel, and zeroing would destroy read access. The push /
        struct-lit / assign / variant-ctor code paths deep-clone on
        borrow inputs, so the borrow view is safe to copy around
        without aliasing the container's storage at a release site."""
        assert self.builder is not None
        fields = self._struct_fields_for(struct_ty)
        if fields is None:
            return val
        b = self.builder
        result = val
        for i, (_fname, fty) in enumerate(fields):
            if self._is_str_value(fty):
                old = b.extract_value(result, i)
                borrowed = self._str_as_borrow(old)
                result = b.insert_value(result, borrowed, i)
                continue
            if (
                self._struct_fields_for(fty) is not None
                and self._struct_needs_cleanup(fty)
            ):
                old = b.extract_value(result, i)
                borrowed = self._struct_as_borrow(old, fty)
                result = b.insert_value(result, borrowed, i)
        return result

    def _get_struct_release(self, struct_ty: ir.Type) -> ir.Function:
        """Build (once, caching by LLVM-type identity) a release fn for
        a user struct: `__tuppu_struct_<name>_release(s: *struct_ty)`.
        GEPs to each cleanup-bearing field and dispatches to the
        appropriate release — str, tablets, or nested struct. Fields
        without cleanup are skipped entirely."""
        cached = self._struct_release_cache.get(id(struct_ty))
        if cached is not None:
            return cached
        name = self._struct_name_for(struct_ty) or "anon"
        fn = ir.Function(
            self.module,
            ir.FunctionType(ir.VoidType(), [struct_ty.as_pointer()]),
            name=f"__tuppu_struct_{name}_release",
        )
        # Cache before body-build so any recursive call through a nested
        # struct field (via another _get_struct_release) sees the in-
        # progress function rather than rebuilding it.
        self._struct_release_cache[id(struct_ty)] = fn

        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        s_ptr = fn.args[0]
        fields = self._struct_fields_for(struct_ty) or []
        for i, (_fname, fty) in enumerate(fields):
            if self._is_str_value(fty):
                field_ptr = b.gep(
                    s_ptr, [ir.Constant(I32, 0), ir.Constant(I32, i)],
                    inbounds=True,
                )
                b.call(self._get_str_release(), [field_ptr])
                continue
            info = self._tablets_info_for(fty)
            if info is not None:
                field_ptr = b.gep(
                    s_ptr, [ir.Constant(I32, 0), ir.Constant(I32, i)],
                    inbounds=True,
                )
                b.call(self._get_tablets_release(info), [field_ptr])
                continue
            if (
                self._struct_fields_for(fty) is not None
                and self._struct_needs_cleanup(fty)
            ):
                field_ptr = b.gep(
                    s_ptr, [ir.Constant(I32, 0), ir.Constant(I32, i)],
                    inbounds=True,
                )
                b.call(self._get_struct_release(fty), [field_ptr])
                continue
            if (
                self._seal_key_for_ty(fty) is not None
                and self._seal_needs_cleanup(fty)
            ):
                field_ptr = b.gep(
                    s_ptr, [ir.Constant(I32, 0), ir.Constant(I32, i)],
                    inbounds=True,
                )
                b.call(self._get_seal_release(fty), [field_ptr])
        b.ret_void()
        return fn

    def _get_struct_clone(self, struct_ty: ir.Type) -> ir.Function:
        """Build (once, caching by LLVM-type identity) a deep-clone fn
        for a user struct: `__tuppu_struct_<name>_clone(src: struct_ty)
        -> struct_ty`. Returns a fresh value with cloned str fields
        (new heap allocations), recursively-cloned nested struct
        fields, deep-cloned tablets fields, and scalar fields copied
        by value."""
        cached = self._struct_clone_cache.get(id(struct_ty))
        if cached is not None:
            return cached
        name = self._struct_name_for(struct_ty) or "anon"
        fn = ir.Function(
            self.module,
            ir.FunctionType(struct_ty, [struct_ty]),
            name=f"__tuppu_struct_{name}_clone",
        )
        # Cache up front so recursive struct clones (nested fields of
        # the same type) see the in-progress function rather than
        # rebuilding it into an infinite loop.
        self._struct_clone_cache[id(struct_ty)] = fn

        entry = fn.append_basic_block("entry")
        b = ir.IRBuilder(entry)
        src = fn.args[0]
        # Same isolation dance as seal_clone / tablets_clone: don't leak
        # cleanup entries or root counts into the caller's frames, root
        # the dst slot so accumulated field clones stay reachable, emit
        # the paired pop before ret.
        saved_frames = self._cleanup_frames
        saved_counts = self._gc_root_counts
        saved_helper = getattr(self, "_in_helper_emission", False)
        saved_builder = self.builder
        self._cleanup_frames = [[]]
        self._gc_root_counts = [0]
        self._in_helper_emission = True
        self.builder = b
        dst_slot = b.alloca(struct_ty, name="dst")
        b.store(ir.Constant(struct_ty, None), dst_slot)
        dst_desc = self._get_type_desc(struct_ty)
        dst_pushed = False
        if dst_desc is not None:
            b.call(
                self._get_gc_push_root(),
                [
                    b.bitcast(dst_slot, I8.as_pointer()),
                    b.bitcast(dst_desc, I8.as_pointer()),
                ],
            )
            dst_pushed = True
        fields = self._struct_fields_for(struct_ty) or []
        for i, (_fname, fty) in enumerate(fields):
            field_val = b.extract_value(src, i)
            dst_field_ptr = b.gep(
                dst_slot, [ir.Constant(I32, 0), ir.Constant(I32, i)],
                inbounds=True,
            )
            if self._is_str_value(fty):
                cloned = b.call(self._get_str_clone(), [field_val])
                b.store(cloned, dst_field_ptr)
                continue
            info = self._tablets_info_for(fty)
            if info is not None:
                src_slot = b.alloca(fty, name=".tbls.field.src")
                b.store(field_val, src_slot)
                cloned = b.call(
                    self._get_tablets_clone(info), [src_slot],
                )
                b.store(cloned, dst_field_ptr)
                continue
            if (
                self._struct_fields_for(fty) is not None
                and self._struct_needs_cleanup(fty)
            ):
                cloned = b.call(self._get_struct_clone(fty), [field_val])
                b.store(cloned, dst_field_ptr)
                continue
            if (
                self._seal_key_for_ty(fty) is not None
                and self._seal_needs_cleanup(fty)
            ):
                cloned = b.call(self._get_seal_clone(fty), [field_val])
                b.store(cloned, dst_field_ptr)
                continue
            # Scalars, pointers, wedges: copy by value.
            b.store(field_val, dst_field_ptr)
        if dst_pushed:
            b.call(self._get_gc_pop_roots(), [ir.Constant(I64, 1)])
        b.ret(b.load(dst_slot))
        self._cleanup_frames = saved_frames
        self._gc_root_counts = saved_counts
        self._in_helper_emission = saved_helper
        self.builder = saved_builder
        return fn

    def _get_seal_release(self, seal_ty: ir.Type) -> ir.Function:
        """Build (once, cached by LLVM-type identity) a release fn for a
        seal: `__tuppu_seal_<tag>_release(*seal_ty)`. Loads the tag,
        switches on each variant index, and for variants with
        cleanup-bearing payload fields bitcasts the payload slot to
        the variant struct and releases each field via its
        appropriate release helper. Variants with no cleanup-bearing
        fields (including nullary) fall through to the default
        (no-op) block."""
        cached = self._seal_release_cache.get(id(seal_ty))
        if cached is not None:
            return cached
        tag = self._seal_fn_suffix(seal_ty)
        fn = ir.Function(
            self.module,
            ir.FunctionType(ir.VoidType(), [seal_ty.as_pointer()]),
            name=f"__tuppu_seal_{tag}_release",
        )
        self._seal_release_cache[id(seal_ty)] = fn
        slot = fn.args[0]
        slot.name = "s"

        entry = fn.append_basic_block("entry")
        done = fn.append_basic_block("done")
        b = ir.IRBuilder(entry)
        seal_key = self._seal_key_for_ty(seal_ty)
        variants = self._seal_variants[seal_key]
        tag_ptr = b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, 0)], inbounds=True,
        )
        # GEP the payload slot BEFORE the switch (the switch is a
        # terminator — anything after it in `entry` is invalid IR).
        payload_ptr = None
        if len(seal_ty.elements) >= 2:
            payload_ptr = b.gep(
                slot, [ir.Constant(I32, 0), ir.Constant(I32, 1)],
                inbounds=True,
            )
        tag_val = b.load(tag_ptr)
        switch = b.switch(tag_val, done)

        for vidx, (vname, payload_ty) in enumerate(variants):
            needs = any(
                self._field_needs_cleanup(fty) for fty in payload_ty.elements
            )
            if not needs:
                continue
            case_bb = fn.append_basic_block(f"release.{vname}")
            switch.add_case(ir.Constant(I8, vidx), case_bb)
            b.position_at_end(case_bb)
            assert payload_ptr is not None
            typed = b.bitcast(payload_ptr, payload_ty.as_pointer())
            for fi, fty in enumerate(payload_ty.elements):
                field_ptr = b.gep(
                    typed,
                    [ir.Constant(I32, 0), ir.Constant(I32, fi)],
                    inbounds=True,
                )
                if self._is_str_value(fty):
                    b.call(self._get_str_release(), [field_ptr])
                    continue
                info = self._tablets_info_for(fty)
                if info is not None:
                    b.call(self._get_tablets_release(info), [field_ptr])
                    continue
                if (
                    self._struct_fields_for(fty) is not None
                    and self._struct_needs_cleanup(fty)
                ):
                    b.call(self._get_struct_release(fty), [field_ptr])
                    continue
                if (
                    self._seal_key_for_ty(fty) is not None
                    and self._seal_needs_cleanup(fty)
                ):
                    b.call(self._get_seal_release(fty), [field_ptr])
                    continue
            b.branch(done)

        b.position_at_end(done)
        # After releasing, zero the tag so repeat releases (shouldn't
        # happen, but the transfer-zeroing path relies on it) walk the
        # default no-op block.
        b.store(ir.Constant(I8, 0), tag_ptr)
        b.ret_void()
        return fn

    def _get_seal_clone(self, seal_ty: ir.Type) -> ir.Function:
        """Build (once, cached by LLVM-type identity) a deep-clone fn
        for a seal: `__tuppu_seal_<tag>_clone(seal_ty) -> seal_ty`.
        Loads the tag, switches on the variant, and deep-clones each
        cleanup-bearing payload field into a fresh seal value. Scalar
        payload fields copy by value."""
        cached = self._seal_clone_cache.get(id(seal_ty))
        if cached is not None:
            return cached
        tag = self._seal_fn_suffix(seal_ty)
        fn = ir.Function(
            self.module,
            ir.FunctionType(seal_ty, [seal_ty]),
            name=f"__tuppu_seal_{tag}_clone",
        )
        self._seal_clone_cache[id(seal_ty)] = fn
        src_arg = fn.args[0]
        src_arg.name = "src"

        entry = fn.append_basic_block("entry")
        merge = fn.append_basic_block("merge")
        b = ir.IRBuilder(entry)
        # Spill src to a stack slot so we can GEP/bitcast the payload.
        src_slot = b.alloca(seal_ty, name="src.slot")
        b.store(src_arg, src_slot)
        dst_slot = b.alloca(seal_ty, name="dst.slot")
        # Start with a scalar copy — covers the tag + any scalar payload
        # bytes. Deep-cloned fields overwrite on top.
        b.store(src_arg, dst_slot)
        # Root dst so cleanup-bearing fields stay reachable across any
        # GC that fires during subsequent field clones. One root, one
        # pop at the single merge ret. Helper-fn body emission suppresses
        # the deep_clone chokepoint (see `_in_helper_emission`) so no
        # extra pushes leak through branch-divergent paths.
        saved_frames = self._cleanup_frames
        saved_counts = self._gc_root_counts
        saved_helper = getattr(self, "_in_helper_emission", False)
        saved_builder = self.builder
        self._cleanup_frames = [[]]
        self._gc_root_counts = [0]
        self._in_helper_emission = True
        self.builder = b
        dst_desc = self._get_type_desc(seal_ty)
        dst_pushed = False
        if dst_desc is not None:
            b.call(
                self._get_gc_push_root(),
                [
                    b.bitcast(dst_slot, I8.as_pointer()),
                    b.bitcast(dst_desc, I8.as_pointer()),
                ],
            )
            dst_pushed = True

        seal_key = self._seal_key_for_ty(seal_ty)
        variants = self._seal_variants[seal_key]
        tag_ptr = b.gep(
            src_slot, [ir.Constant(I32, 0), ir.Constant(I32, 0)], inbounds=True,
        )
        # GEP the payload slots BEFORE the switch terminator.
        src_payload_ptr = None
        dst_payload_ptr = None
        if len(seal_ty.elements) >= 2:
            src_payload_ptr = b.gep(
                src_slot, [ir.Constant(I32, 0), ir.Constant(I32, 1)],
                inbounds=True,
            )
            dst_payload_ptr = b.gep(
                dst_slot, [ir.Constant(I32, 0), ir.Constant(I32, 1)],
                inbounds=True,
            )
        tag_val = b.load(tag_ptr)
        switch = b.switch(tag_val, merge)

        for vidx, (vname, payload_ty) in enumerate(variants):
            needs = any(
                self._field_needs_cleanup(fty) for fty in payload_ty.elements
            )
            if not needs:
                continue
            case_bb = fn.append_basic_block(f"clone.{vname}")
            switch.add_case(ir.Constant(I8, vidx), case_bb)
            b.position_at_end(case_bb)
            assert src_payload_ptr is not None and dst_payload_ptr is not None
            src_typed = b.bitcast(src_payload_ptr, payload_ty.as_pointer())
            dst_typed = b.bitcast(dst_payload_ptr, payload_ty.as_pointer())
            for fi, fty in enumerate(payload_ty.elements):
                if not self._field_needs_cleanup(fty):
                    continue
                src_field_ptr = b.gep(
                    src_typed,
                    [ir.Constant(I32, 0), ir.Constant(I32, fi)],
                    inbounds=True,
                )
                dst_field_ptr = b.gep(
                    dst_typed,
                    [ir.Constant(I32, 0), ir.Constant(I32, fi)],
                    inbounds=True,
                )
                src_field_val = b.load(src_field_ptr)
                saved = self.builder
                self.builder = b
                try:
                    cloned = self._deep_clone_if_cleanup_bearing(src_field_val)
                finally:
                    self.builder = saved
                b.store(cloned, dst_field_ptr)
            b.branch(merge)

        b.position_at_end(merge)
        if dst_pushed:
            b.call(self._get_gc_pop_roots(), [ir.Constant(I64, 1)])
        b.ret(b.load(dst_slot))
        self._cleanup_frames = saved_frames
        self._gc_root_counts = saved_counts
        self._in_helper_emission = saved_helper
        self.builder = saved_builder
        return fn

    def _field_needs_cleanup(self, fty: ir.Type) -> bool:
        """Helper: does this LLVM type need a release fn? (str, tablets,
        struct-with-cleanup, or seal-with-cleanup-payload.)"""
        if self._is_str_value(fty):
            return True
        if self._tablets_info_for(fty) is not None:
            return True
        if (
            self._struct_fields_for(fty) is not None
            and self._struct_needs_cleanup(fty)
        ):
            return True
        if (
            self._seal_key_for_ty(fty) is not None
            and self._seal_needs_cleanup(fty)
        ):
            return True
        return False

    def _seal_fn_suffix(self, seal_ty: ir.Type) -> str:
        """Turn a seal LLVM type into a stable fn-name suffix. Uses the
        seal name for non-generic seals and `Name__arg_tag` for
        monomorphs, matching the identified-type naming scheme."""
        key = self._seal_key_for_ty(seal_ty)
        if isinstance(key, str):
            return key
        if isinstance(key, tuple):
            name, arg_tys = key
            arg_tag = "_".join(
                str(a).replace(" ", "").replace('"', "")
                for a in arg_tys
            )
            return f"{name}__{arg_tag}"
        return "anon_seal"

    def _force_root_cleanup_value(self, val: ir.Value) -> ir.Value:
        """Chokepoint primitive. Unconditionally spill+root a
        cleanup-bearing rvalue — used at PRODUCTION sites (Call /
        Binary str-concat / Copy) so consumers don't need to re-root.
        Returns `val` unchanged (the spill slot is a side channel
        for the collector; the SSA register still feeds consumers).

        Scalars and non-traceable types pass through untouched —
        there's nothing for the collector to trace. Helper-fn body
        emission (clone/release/trace) sets `_in_helper_emission`
        so clone results don't pollute the outer caller's cleanup
        frame — the helper roots its own dst slot at entry to keep
        sequential field clones alive across reallocs."""
        if getattr(self, "_in_helper_emission", False):
            return val
        if not self._cleanup_frames:
            return val
        if self._type_desc_key(val.type) is None:
            return val
        assert self.builder is not None
        slot = self._alloca_entry(val.type, ".rvalue.root")
        self.builder.store(val, slot)
        # Release entry — dispatch by type. Keeps the scope-exit
        # release machinery's shape consistent, so post-GC-delete
        # work can sweep them all at once.
        if self._is_str_value(val.type):
            release = self._get_str_release()
        else:
            info = self._tablets_info_for(val.type)
            if info is not None:
                release = self._get_tablets_release(info)
            elif (
                self._struct_fields_for(val.type) is not None
                and self._struct_needs_cleanup(val.type)
            ):
                release = self._get_struct_release(val.type)
            elif (
                self._seal_key_for_ty(val.type) is not None
                and self._seal_needs_cleanup(val.type)
            ):
                release = self._get_seal_release(val.type)
            else:
                release = None
        if release is not None:
            self._cleanup_frames[-1].append(
                (release, slot, ".rvalue.root"),
            )
        self._register_gc_root(slot, val.type)
        return val

    def _gen_assign(self, a: A.Assign) -> None:
        assert self.builder is not None
        # Resolve the target to (slot_ptr, value_type). For an Ident target
        # the slot is the alloca itself. For a Field chain, we GEP from
        # the root alloca down to the innermost field.
        slot_ptr, slot_ty = self._lvalue_slot(a.target)
        value = self._gen_expr(a.value)
        if value is None:
            raise CodegenError("assignment RHS has no value")
        # Reassignment: release the old value before overwriting.
        # Covers every cleanup-bearing slot type — str (cap-sentinel
        # no-ops borrows), tablets (frees the chunk chain), or a
        # user struct that transitively owns cleanup-bearing fields.
        # Without this, any prior heap state leaks on reassign.
        if self._is_str_value(slot_ty):
            self.builder.call(self._get_str_release(), [slot_ptr])
        else:
            info = self._tablets_info_for(slot_ty)
            if info is not None:
                self.builder.call(self._get_tablets_release(info), [slot_ptr])
            elif (
                self._struct_fields_for(slot_ty) is not None
                and self._struct_needs_cleanup(slot_ty)
            ):
                self.builder.call(
                    self._get_struct_release(slot_ty), [slot_ptr],
                )
            elif (
                self._seal_key_for_ty(slot_ty) is not None
                and self._seal_needs_cleanup(slot_ty)
            ):
                self.builder.call(
                    self._get_seal_release(slot_ty), [slot_ptr],
                )
        # Ownership into the slot: three-way split matching push /
        # struct-lit. Owning Ident transfers its cleanup; a fresh-
        # owned rvalue passes through unchanged (no wasted clone);
        # borrow source (or Ident naming a borrow) gets a deep-clone.
        coerced = self._coerce(value, slot_ty)
        if self._is_cleanup_bearing_ty(slot_ty):
            if isinstance(a.value, A.Ident):
                transferred = self._transfer_cleanup_into_container(
                    a.value.name,
                )
                if not transferred:
                    coerced = self._deep_clone_if_cleanup_bearing(coerced)
            elif self._is_borrow_source_expr(a.value):
                coerced = self._deep_clone_if_cleanup_bearing(coerced)
        self.builder.store(coerced, slot_ptr)

    def _lvalue_slot(self, target: A.Expr) -> tuple[ir.Value, ir.Type]:
        """Resolve an lvalue to (pointer-to-slot, value-type-at-slot).

        Root must be a mut-bound Ident, or a step-bound wedge (step
        bindings are SSA-immutable but writing THROUGH the handle into
        the underlying tablets slot is fine, same as `arr[n].field = v`
        going through a mut tablets). Each Field step GEPs one level
        deeper through the appropriate user-struct LLVM type."""
        assert self.builder is not None
        if isinstance(target, A.Ident):
            var = self._lookup(target.name)
            if var.is_mut:
                return var.ir_ref, var.value_ty
            # Step-bound wedge: the binding is an SSA pointer value.
            # Spill to a throwaway slot so Field auto-deref has a
            # pointer-to-pointer to load from. The spill is local —
            # mutations happen in the pointee, not in the binding.
            if isinstance(var.value_ty, ir.PointerType):
                slot = self._alloca_entry(var.value_ty, f".{target.name}.lv")
                self.builder.store(var.ir_ref, slot)
                return slot, var.value_ty
            raise CodegenError(
                f"cannot assign to step binding {target.name!r}"
            )
        if isinstance(target, A.Field):
            parent_ptr, parent_ty = self._lvalue_slot(target.target)
            # Wedge (tablet handle) auto-deref for lvalue field access:
            # the parent slot holds a `*T` pointer into a tablets-owned
            # slot. Load the handle, then GEP through the pointee to
            # the field — mirrors the read-side wedge-deref in
            # `_gen_field`. Without this, `w.field = x` on a `wedge T`
            # errored with "not a user tablet" since the direct path
            # sees the pointer type.
            if isinstance(parent_ty, ir.PointerType):
                pointee = parent_ty.pointee
                pointee_fields = self._struct_fields_for(pointee)
                if pointee_fields is not None:
                    wedge_val = self.builder.load(parent_ptr)
                    for i, (fname, fty) in enumerate(pointee_fields):
                        if fname == target.name:
                            field_ptr = self.builder.gep(
                                wedge_val,
                                [ir.Constant(I32, 0), ir.Constant(I32, i)],
                                inbounds=True,
                            )
                            return field_ptr, fty
                    raise CodegenError(
                        f"tablet has no field {target.name!r}"
                    )
            fields = self._struct_fields_for(parent_ty)
            if fields is None:
                raise CodegenError(
                    f"field assignment: {parent_ty} is not a user tablet"
                )
            for i, (fname, fty) in enumerate(fields):
                if fname == target.name:
                    field_ptr = self.builder.gep(
                        parent_ptr,
                        [ir.Constant(I32, 0), ir.Constant(I32, i)],
                        inbounds=True,
                    )
                    return field_ptr, fty
            raise CodegenError(
                f"tablet has no field {target.name!r}"
            )
        if isinstance(target, A.Index):
            # lvalue indexing: `arr[n] = v`, `arr[n].field = v`,
            # `obj.field[n] = v`, `m.keys[i] = v`, etc. Resolve the
            # parent (which may itself be a Field / Index chain) to
            # the pointer-to-container slot, then GEP / call get_addr
            # on top.
            parent_ptr, parent_ty = self._lvalue_slot(target.target)
            idx_val = self._gen_expr(target.index)
            if idx_val is None:
                raise CodegenError("lvalue index has no value")
            idx_val = self._coerce(idx_val, I64)
            if isinstance(parent_ty, ir.ArrayType):
                self._emit_bounds_trap(idx_val, parent_ty.count)
                slot = self.builder.gep(
                    parent_ptr,
                    [ir.Constant(I32, 0), idx_val],
                    inbounds=True,
                )
                return slot, parent_ty.element
            if self._is_ivec_value(parent_ty):
                # ivec lvalue index — write through get_addr so the
                # underlying T allocation's address stays stable
                # (a wedge taken before this assignment still sees
                # the new value via the same pointer).
                from ._common import IVEC_IDX_LEN
                elem_ty = self._ivec_elem_for_index(target)
                if elem_ty is None:
                    raise CodegenError(
                        "lvalue ivec index: missing element type "
                        "from typecheck sideband"
                    )
                iv_info = self._get_ivec(elem_ty)
                len_addr = self.builder.gep(
                    parent_ptr,
                    [ir.Constant(I32, 0), ir.Constant(I32, IVEC_IDX_LEN)],
                    inbounds=True,
                )
                length = self.builder.load(len_addr)
                self._emit_dynamic_bounds_trap(idx_val, length)
                slot = self.builder.call(
                    self._get_ivec_get_addr(iv_info),
                    [parent_ptr, idx_val],
                )
                return slot, elem_ty
            if self._is_dvec_value(parent_ty):
                from ._common import DVEC_IDX_LEN
                elem_ty = self._dvec_elem_for_index(target)
                if elem_ty is None:
                    raise CodegenError(
                        "lvalue dvec index: missing element type "
                        "from typecheck sideband"
                    )
                dv_info = self._get_dvec(elem_ty)
                len_addr = self.builder.gep(
                    parent_ptr,
                    [ir.Constant(I32, 0), ir.Constant(I32, DVEC_IDX_LEN)],
                    inbounds=True,
                )
                length = self.builder.load(len_addr)
                self._emit_dynamic_bounds_trap(idx_val, length)
                slot = self.builder.call(
                    self._get_dvec_get_addr(dv_info),
                    [parent_ptr, idx_val],
                )
                return slot, elem_ty
            info = self._tablets_info_for(parent_ty)
            if info is None:
                raise CodegenError(
                    f"lvalue indexing: parent is not a tablets or buffer "
                    f"(got {parent_ty})"
                )
            len_addr = self.builder.gep(
                parent_ptr,
                [ir.Constant(I32, 0), ir.Constant(I32, 2)],
                inbounds=True,
            )
            length = self.builder.load(len_addr)
            self._emit_dynamic_bounds_trap(idx_val, length)
            slot = self.builder.call(
                self._get_tablets_get_addr(info), [parent_ptr, idx_val],
            )
            return slot, info.elem_ty
        raise CodegenError(
            f"assignment target must be a variable or field chain, "
            f"got {type(target).__name__}"
        )


