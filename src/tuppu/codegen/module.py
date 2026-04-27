"""Top-level codegen: the `gen(program)` driver, fn / colophon / gloss
declaration, fn body emission, generic monomorphization
(`_get_monomorph_struct`, `_get_monomorph_fn`, `_emit_fn_specialization`),
the parameter-prelude plumbing (param spill + cleanup-frame seeding),
the closure-aware tail-return pass, and `_block_tail_expr`. Extracted
from `codegen/__init__.py` as `ModuleMixin`."""
from __future__ import annotations

from llvmlite import ir

from .. import ast as A
from ..comptime import Comptime
from ._common import (
    CodegenError, Variable,
    I1, I8, I16, I32, I64,
    INTRINSICS,
)


class ModuleMixin:
    # --- top level ---

    def gen(self, prog: A.Program) -> ir.Module:
        self.comptime = Comptime(prog)

        # Phase 0: build struct + seal LLVM types. Interleave declaration
        # and body-resolution so a struct field of seal type (or vice
        # versa) can see the identified type of the other form before
        # we compute layouts.
        struct_decls = [
            d for d in prog.decls if isinstance(d, A.StructDecl)
        ]
        seal_decls = [
            d for d in prog.decls if isinstance(d, A.SealDecl)
        ]
        self._register_structs_declare(struct_decls)
        self._register_seals_declare(seal_decls)
        self._register_structs_resolve(struct_decls)
        self._register_seals_resolve(seal_decls)

        # Generic fns are monomorphized lazily at call sites, so we
        # don't declare/emit them here — just stash the AST.
        self._generic_fn_decls: dict[str, A.FnDecl] = {
            d.name: d for d in prog.decls
            if isinstance(d, A.FnDecl) and d.type_params
        }

        # Phase 1: forward-declare all non-generic user functions, plus
        # colophon externs (C functions the compiler marshals to / from
        # at each call site).
        for decl in prog.decls:
            if isinstance(decl, A.FnDecl):
                if decl.type_params:
                    continue
                self._declare_fn(decl)
            elif isinstance(decl, A.ColophonDecl):
                self._declare_colophon(decl)
            elif isinstance(decl, A.GlossDecl):
                self._declare_gloss(decl)
            elif isinstance(decl, A.TableDecl):
                pass  # handled in phase 2 after function decls are visible
            elif isinstance(decl, A.StructDecl):
                pass  # already handled in phase 0
            elif isinstance(decl, A.SealDecl):
                pass  # already handled in phase 0c
            elif isinstance(decl, A.AliasDecl):
                pass  # transparent — resolved on-demand in _lower_type
            else:
                raise CodegenError(
                    f"unsupported top-level: {type(decl).__name__}"
                )

        # Phase 2: evaluate tables at compile time and emit them as static
        # globals. Done in declaration order so later tables may reference
        # earlier ones.
        for decl in prog.decls:
            if isinstance(decl, A.TableDecl):
                self._emit_table(decl)

        # Phase 3: emit bodies of non-generic functions. Generic fn
        # specializations are emitted on demand when we see a call to
        # them (see `_get_monomorph_fn`).
        for decl in prog.decls:
            if isinstance(decl, A.FnDecl):
                if decl.type_params:
                    continue
                self._gen_fn_body(decl)
            elif isinstance(decl, A.GlossDecl):
                self._gen_gloss_body(decl)
        return self.module

    def _declare_fn(self, fn: A.FnDecl) -> None:
        if fn.name in INTRINSICS:
            raise CodegenError(
                f"cannot define {fn.name!r}: it is a built-in intrinsic"
            )
        if fn.name in self.functions:
            raise CodegenError(f"duplicate function {fn.name!r}")
        param_types = []
        for p in fn.params:
            t = self._lower_type(p.type)
            # Mut tablets params are passed by reference so mutations
            # (push, release) persist to the caller's storage. Without
            # this the caller's tablets header (head/tail/len) stays
            # unchanged and any chunks the callee allocated would leak.
            if p.is_mut and self._tablets_info_for(t) is not None:
                t = t.as_pointer()
            # Variadic `tablets[...]T` param: call site builds the
            # literal in the caller's frame; callee receives a pointer
            # so indexing and iteration see the actual chunks.
            elif isinstance(p.type, A.TypeVariadicTablets):
                t = t.as_pointer()
            # Mut user-struct param — pass by pointer so callee
            # mutations persist to the caller's storage. Previously
            # mut structs were pass-by-value, which made
            # `fn add_route(mut app: App) { app.routes.push(...) }`
            # silently no-op from the caller's perspective. Matches
            # the mut-tablets and colophon-mut-struct conventions.
            # `str` is excluded: it has its own cap-sentinel ownership
            # model and reassignment-release machinery that assumes
            # by-value passing with call-site neutering.
            elif (
                p.is_mut
                and self._struct_fields_for(t) is not None
                and not self._is_str_value(t)
            ):
                t = t.as_pointer()
            param_types.append(t)
        ret_type = self._lower_type(fn.return_type) if fn.return_type else ir.VoidType()
        fn_type = ir.FunctionType(ret_type, param_types)
        llvm_fn = ir.Function(self.module, fn_type, name=fn.name)
        for i, p in enumerate(fn.params):
            llvm_fn.args[i].name = p.name
        self.functions[fn.name] = llvm_fn
        self._fn_param_mut[fn.name] = [p.is_mut for p in fn.params]

    def _declare_gloss(self, g: A.GlossDecl) -> None:
        """Forward-declare a gloss fn under its mangled internal name.
        Mirrors `_declare_fn` but resolves the name through the
        checker's mangle scheme so operator dispatch can `self.functions
        [mangled]` like any other fn."""
        from ..typecheck import GLOSS_OPS
        if self._checker is None:
            raise CodegenError("gloss decl requires a typechecker pass")
        # Rebuild the mangled name from the decl's operand types.
        param_tys = tuple(
            self._checker._resolve_type(p.type, "gloss param")
            for p in g.params
        )
        _sym, arity, _ = GLOSS_OPS[g.op]
        rhs_ty = param_tys[1] if arity == "bin" else None
        mangled = self._checker._gloss_mangled_name(g.op, param_tys[0], rhs_ty)
        fake_fn = A.FnDecl(
            name=mangled,
            params=g.params,
            return_type=g.return_type,
            body=g.body,
            line=g.line,
            col=g.col,
        )
        self._declare_fn(fake_fn)

    def _gen_gloss_body(self, g: A.GlossDecl) -> None:
        """Emit the body of a gloss decl — identical to a regular fn
        body, just under the mangled name registered during
        `_declare_gloss`."""
        from ..typecheck import GLOSS_OPS
        assert self._checker is not None
        param_tys = tuple(
            self._checker._resolve_type(p.type, "gloss param")
            for p in g.params
        )
        _sym, arity, _ = GLOSS_OPS[g.op]
        rhs_ty = param_tys[1] if arity == "bin" else None
        mangled = self._checker._gloss_mangled_name(g.op, param_tys[0], rhs_ty)
        fake_fn = A.FnDecl(
            name=mangled,
            params=g.params,
            return_type=g.return_type,
            body=g.body,
            line=g.line,
            col=g.col,
        )
        self._gen_fn_body(fake_fn)

    def _declare_colophon(self, c: A.ColophonDecl) -> None:
        """Forward-declare a libc extern. The LLVM signature uses C-ABI
        types (i8* for Tuppu str, i8 for bool, ints pass through) so the
        Tuppu-level call site can marshal values at each boundary —
        caller-side str gets a fresh NUL-terminated heap buffer, return
        str gets copied into a Tuppu-owned heap str via strlen + memcpy.

        Reserves the Tuppu name in both the fn table and a per-colophon
        sideband so the call-site dispatch can recognise colophon calls
        and pick the marshaling path."""
        if c.name in INTRINSICS:
            raise CodegenError(
                f"cannot declare colophon {c.name!r}: name is a built-in intrinsic"
            )
        if c.name in self.functions:
            raise CodegenError(f"duplicate declaration {c.name!r}")
        c_sym = c.c_name or c.name
        param_types = []
        for p in c.params:
            ty = self._lower_type(p.type)
            # Mut user-tablet params cross the C ABI by pointer
            # (mirrors `mut tablets[N]T` semantics; matches how libc
            # writes through `struct sockaddr *addr`). Non-mut user
            # tablets pass by value — LLVM lowers them to the
            # platform's struct-arg ABI.
            if p.is_mut and self._struct_fields_for(ty) is not None:
                param_types.append(ty.as_pointer())
            else:
                # Buffers always pass as `T*` regardless of mut —
                # arrays can't be passed by value across C at all.
                param_types.append(self._colophon_c_ty(ty))
        ret_type = self._colophon_c_ty(
            self._lower_type(c.return_type) if c.return_type
            else ir.VoidType()
        )
        fn_type = ir.FunctionType(ret_type, param_types)
        existing = self.module.globals.get(c_sym)
        if existing is not None:
            # Another declaration (internal runtime helper or a prior
            # colophon resolved through the same C symbol) already
            # exists. Refuse to reuse it unless the signatures match —
            # a silent mismatch would emit correct-looking IR that
            # miscalls the C function. Users can always pick a
            # different Tuppu-side name; we reserve an explicit
            # C-symbol override for a future syntax pass.
            existing_ty = getattr(existing, "function_type", None)
            if existing_ty != fn_type:
                raise CodegenError(
                    f"colophon {c.name!r} collides with the compiler's "
                    f"internal {c_sym!r} extern (signature mismatch: "
                    f"declared {fn_type}, internal {existing_ty}). Pick a "
                    f"different name — the marshaler would silently "
                    f"misbehave otherwise."
                )
            llvm_fn = existing
        else:
            llvm_fn = ir.Function(self.module, fn_type, name=c_sym)
            for i, p in enumerate(c.params):
                llvm_fn.args[i].name = p.name
        self.functions[c.name] = llvm_fn
        self._colophon_decls[c.name] = c
        self._fn_param_mut[c.name] = [p.is_mut for p in c.params]

    def _colophon_c_ty(self, ty: ir.Type) -> ir.Type:
        """Map a Tuppu-side LLVM type to its C-ABI counterpart for
        extern signatures. `str` becomes `i8*` (pointer to NUL-
        terminated bytes); `bool` widens to `i8` for cross-platform
        stability; integer types pass through unchanged. A
        `buffer[N]T` decays to `T*` — the natural C-side shape for
        byte-buffer-taking fns like `recv`/`send`."""
        if isinstance(ty, ir.VoidType):
            return ty
        if self._is_str_value(ty):
            return I8.as_pointer()
        if isinstance(ty, ir.ArrayType):
            return ty.element.as_pointer()
        if ty == I1:
            return I8
        return ty

    def _str_to_cstr(self, s_val: ir.Value) -> ir.Value:
        """Emit `malloc(len+1) + memcpy(ptr, len) + NUL` to produce a
        fresh NUL-terminated C string from a Tuppu str value. The
        returned i8* is heap-owned by the call-site — it must be
        freed after the extern call returns."""
        assert self.builder is not None
        b = self.builder
        ptr = b.extract_value(s_val, 0)
        length = b.extract_value(s_val, 1)
        alloc_size = b.add(length, ir.Constant(I64, 1))
        raw = b.call(self._get_malloc(), [alloc_size])
        b.call(self._get_memcpy(), [raw, ptr, length])
        b.store(ir.Constant(I8, 0), b.gep(raw, [length], inbounds=True))
        return raw

    def _cstr_to_str(self, cstr: ir.Value) -> ir.Value:
        """Turn a C-returned i8* into a heap-owned Tuppu str via
        `strlen + malloc + memcpy`. The original C pointer is left
        untouched — Tuppu owns a copy — so callers returning pointers
        into static storage (getenv) or the stack don't force
        premature frees on the caller's side.

        NULL returns (getenv on a missing var, etc.) yield an empty
        borrow: `{ptr=null, len=0, cap=0}`. This collapses "not found"
        with "found empty string"; stdlib wrappers can distinguish by
        querying the raw env before the marshal if needed."""
        assert self.builder is not None
        b = self.builder
        fn = b.function
        is_null = b.icmp_signed(
            "==", cstr, ir.Constant(I8.as_pointer(), None),
        )
        null_bb = fn.append_basic_block("cstr.null")
        copy_bb = fn.append_basic_block("cstr.copy")
        done_bb = fn.append_basic_block("cstr.done")
        b.cbranch(is_null, null_bb, copy_bb)

        b.position_at_end(null_bb)
        empty = self._str_build_value_in(
            b, ir.Constant(I8.as_pointer(), None),
            ir.Constant(I64, 0), ir.Constant(I64, 0),
        )
        b.branch(done_bb)

        b.position_at_end(copy_bb)
        length = b.call(self._get_strlen(), [cstr])
        alloc_size = b.add(length, ir.Constant(I64, 1))
        raw = b.call(self._get_malloc(), [alloc_size])
        b.call(self._get_memcpy(), [raw, cstr, length])
        b.store(ir.Constant(I8, 0), b.gep(raw, [length], inbounds=True))
        copied = self._str_build_value_in(b, raw, length, length)
        b.branch(done_bb)

        b.position_at_end(done_bb)
        phi = b.phi(self._str_ty())
        phi.add_incoming(empty, null_bb)
        phi.add_incoming(copied, copy_bb)
        return phi

    def _gen_fn_value_call(
        self, fn_ptr: ir.Value, fn_ty: ir.FunctionType,
        arg_exprs: list[A.Expr],
    ) -> ir.Value | None:
        """Emit an indirect call through a precomputed fn-pointer value.
        Arg marshaling mirrors the direct-call path — str gets cap=0
        borrow, cleanup-bearing structs get field neutering, etc. — so
        users can't leak or UAF by routing a call through a pointer
        instead of calling by name."""
        assert self.builder is not None
        if len(arg_exprs) != len(fn_ty.args):
            raise CodegenError(
                f"fn-value call expects {len(fn_ty.args)} args, "
                f"got {len(arg_exprs)}"
            )
        call_args: list[ir.Value] = []
        for arg, expected_ty in zip(arg_exprs, fn_ty.args):
            v = self._gen_expr(arg)
            if v is None:
                raise CodegenError("fn-value call arg has no value")
            coerced = self._coerce(v, expected_ty)
            if self._is_str_value(expected_ty):
                coerced = self._str_as_borrow(coerced)
            elif (
                self._struct_fields_for(expected_ty) is not None
                and self._struct_needs_cleanup(expected_ty)
            ):
                coerced = self._struct_as_borrow(coerced, expected_ty)
            call_args.append(coerced)
        return self.builder.call(fn_ptr, call_args)

    def _gen_colophon_call(
        self, decl: A.ColophonDecl, llvm_fn: ir.Function,
        arg_exprs: list[A.Expr],
    ) -> ir.Value | None:
        """Lower a call to a colophon-declared extern. Marshals each
        str arg to a fresh cstr buffer, widens bool to i8, passes
        ints through; after the call, frees every cstr we allocated
        and converts an i8* return back into a heap-owned Tuppu str.
        Void return yields None."""
        if len(arg_exprs) != len(decl.params):
            raise CodegenError(
                f"colophon {decl.name!r} expects {len(decl.params)} args, "
                f"got {len(arg_exprs)}"
            )
        assert self.builder is not None
        b = self.builder
        call_args: list[ir.Value] = []
        temp_cstrs: list[ir.Value] = []
        for arg_expr, param in zip(arg_exprs, decl.params):
            param_ty = self._lower_type(param.type)
            # Buffer arg: decay to element pointer via GEP [0, 0]. The
            # arg must name a buffer-typed mut binding so we can take
            # the address of its alloca directly.
            if isinstance(param_ty, ir.ArrayType):
                if not isinstance(arg_expr, A.Ident):
                    raise CodegenError(
                        f"colophon {decl.name!r}: buffer arg must be a "
                        f"buffer-typed Ident, got {type(arg_expr).__name__}"
                    )
                var = self._lookup(arg_expr.name)
                if not var.is_mut or var.value_ty != param_ty:
                    raise CodegenError(
                        f"colophon {decl.name!r}: buffer arg "
                        f"{arg_expr.name!r} must be a mut binding of "
                        f"type {param_ty}"
                    )
                elem_ptr = b.gep(
                    var.ir_ref,
                    [ir.Constant(I32, 0), ir.Constant(I32, 0)],
                    inbounds=True,
                )
                call_args.append(elem_ptr)
                continue
            # Mut user-tablet arg: pass the caller's alloca address so
            # the callee can read/write through it (sockaddr out-params,
            # mut pointer-to-struct libc conventions). The call site
            # must be a mut-bound Ident naming a matching struct.
            if (
                param.is_mut
                and self._struct_fields_for(param_ty) is not None
            ):
                if not isinstance(arg_expr, A.Ident):
                    raise CodegenError(
                        f"colophon {decl.name!r}: mut struct arg must be "
                        f"a mut-bound Ident, got {type(arg_expr).__name__}"
                    )
                var = self._lookup(arg_expr.name)
                if not var.is_mut or var.value_ty != param_ty:
                    raise CodegenError(
                        f"colophon {decl.name!r}: mut struct arg "
                        f"{arg_expr.name!r} must be a mut binding "
                        f"of type {param_ty}"
                    )
                call_args.append(var.ir_ref)
                continue
            v = self._gen_expr(arg_expr)
            if v is None:
                raise CodegenError(
                    f"colophon {decl.name!r} arg has no value"
                )
            v = self._coerce(v, param_ty)
            if self._is_str_value(param_ty):
                cstr = self._str_to_cstr(v)
                temp_cstrs.append(cstr)
                call_args.append(cstr)
            elif param_ty == I1:
                call_args.append(b.zext(v, I8))
            else:
                call_args.append(v)
        result = b.call(llvm_fn, call_args)
        for cstr in temp_cstrs:
            b.call(self._get_free(), [cstr])
        if decl.return_type is None:
            return None
        ret_ty = self._lower_type(decl.return_type)
        if self._is_str_value(ret_ty):
            return self._cstr_to_str(result)
        if ret_ty == I1:
            return b.icmp_signed("!=", result, ir.Constant(I8, 0))
        return result

    def _gen_fn_body(self, fn: A.FnDecl) -> None:
        if fn.name == "main":
            if not (isinstance(fn.return_type, A.TypeName) and fn.return_type.name == "i32"):
                raise CodegenError("main must declare -> i32")

        llvm_fn = self.functions[fn.name]
        entry = llvm_fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        self.scopes = [{}]

        # Params live in a dedicated cleanup frame that wraps the fn body.
        # A mut str param needs release at scope exit so a reassignment
        # to a heap-owned str doesn't leak; the incoming value is a
        # borrow (caller forced cap=0), so the initial release is a no-op.
        # Non-mut str params stay SSA — they can't be reassigned, and the
        # cap=0 borrow has nothing to free.
        self._push_cleanup_frame()
        try:
            # Parameters: step-bound (direct SSA ref) unless the user wrote
            # `mut` — in which case we alloca + store the incoming arg and
            # bind the alloca, so methods requiring a mut binding (notably
            # `tablets.push`) work on the parameter.
            #
            # Special case: mut tablets params arrive already as a pointer
            # to the caller's storage (see `_declare_fn`). We bind the
            # incoming pointer directly as the Variable's ir_ref — no
            # alloca+store — so method dispatch gets a stable pointer to
            # the caller's tablets and mutations persist.
            for i, p in enumerate(fn.params):
                arg = llvm_fn.args[i]
                param_decl_ty = self._lower_type(p.type)
                is_mut_tablets = (
                    p.is_mut and self._tablets_info_for(param_decl_ty) is not None
                )
                is_variadic = isinstance(p.type, A.TypeVariadicTablets)
                is_mut_struct = (
                    p.is_mut
                    and self._struct_fields_for(param_decl_ty) is not None
                    and not self._is_str_value(param_decl_ty)
                )
                if is_mut_tablets or is_variadic or is_mut_struct:
                    # Either shape arrives as a pointer to the caller's
                    # tablets or struct storage; bind the incoming
                    # pointer directly as the Variable's ir_ref so the
                    # body's indexing, iteration, field access, and
                    # method dispatch all work on the caller's actual
                    # storage. No cleanup registration — the caller
                    # owns the memory.
                    self.scopes[-1][p.name] = Variable(
                        is_mut=True, ir_ref=arg, value_ty=param_decl_ty,
                    )
                elif p.is_mut:
                    slot = self._alloca_entry(arg.type, p.name)
                    self.builder.store(arg, slot)
                    self.scopes[-1][p.name] = Variable(
                        is_mut=True, ir_ref=slot, value_ty=arg.type,
                    )
                    self._maybe_register_cleanup(p.name, arg.type, slot)
                elif self._type_desc_key(arg.type) is not None:
                    # Non-mut cleanup-bearing param: spill to a shadow-
                    # stack-rooted slot so GC cycles during the body
                    # see it as a root. Without this, a param passed
                    # as-is through SSA is invisible to the collector
                    # and can be prematurely reclaimed when a callee
                    # triggers GC. Cleanup release is still a no-op
                    # for borrowed-semantics params (caller owns).
                    slot = self._alloca_entry(arg.type, p.name)
                    self.builder.store(arg, slot)
                    # Bind SSA (the incoming value) so reads don't go
                    # through the slot — the slot is a root spill only.
                    # `.ir_ref` remains the SSA value for downstream
                    # ident reads that expect a value, not a pointer.
                    self.scopes[-1][p.name] = Variable(
                        is_mut=False, ir_ref=arg, value_ty=arg.type,
                    )
                    self._register_gc_root(slot, arg.type)
                else:
                    self.scopes[-1][p.name] = Variable(
                        is_mut=False, ir_ref=arg, value_ty=arg.type,
                    )

            value = self._gen_expr(fn.body)

            if self._is_terminated():
                # Body already returned via yield — the yield path unwound
                # every live cleanup frame (including this one).
                return
            if fn.return_type is None:
                self._emit_frame_cleanups(self._cleanup_frames[-1])
                self._emit_all_gc_root_pops_for_early_return()
                self.builder.ret_void()
            else:
                if value is None:
                    raise CodegenError(
                        f"function {fn.name!r} must produce a value for return type "
                        f"{fn.return_type}, but its body has no trailing expression"
                    )
                expected = self._lower_type(fn.return_type)
                coerced = self._coerce(value, expected)
                # Block-level codegen already clones Field/Index tails
                # so the caller gets independently-owned bytes
                # (see `_gen_block`). No second neuter here; cloning
                # twice would leave the first clone's heap bytes
                # unrooted across the second clone's allocation.
                self._emit_frame_cleanups(self._cleanup_frames[-1])
                self._emit_all_gc_root_pops_for_early_return()
                self.builder.ret(coerced)
        finally:
            self._pop_cleanup_frame()

    def _block_tail_expr(self, e: "A.Expr") -> "A.Expr | None":
        """Find the source expression for a fn/block's tail value, if
        any. Drills through nested blocks so `{ ... { x.y } }` returns
        the same expr as `x.y`. Returns None if the tail is missing
        or the expression has no value."""
        if isinstance(e, A.Block):
            if e.tail is None:
                return None
            return self._block_tail_expr(e.tail)
        return e
