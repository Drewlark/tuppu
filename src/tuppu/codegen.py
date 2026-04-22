"""Tuppu codegen: AST -> LLVM IR.

Covers v0.1 integer/bool arithmetic, user functions with parameters,
`step`/`mut` bindings, control flow (if/else/while, yield), string
literals, and the built-in I/O intrinsics `print`, `println`, and
`read_int` — which lower to `printf`/`scanf` calls against libc.

Does *not* yet handle: `rat`, `tablets`, comptime `table` declarations,
or first-class functions.
"""
from __future__ import annotations

from dataclasses import dataclass

from llvmlite import binding as llvm
from llvmlite import ir

from . import ast as A
from .comptime import Comptime, ComptimeError
from .errors import CompileError


class CodegenError(CompileError):
    pass


I1 = ir.IntType(1)
I8 = ir.IntType(8)
I16 = ir.IntType(16)
I32 = ir.IntType(32)
I64 = ir.IntType(64)

# rat is a literal struct { i64 num, i64 den }, always reduced (gcd=1) and
# normalized so den > 0 at construction time. Field 0 is num, field 1 is den.
RAT = ir.LiteralStructType([I64, I64])

# Sex: Babylonian-faithful sexagesimal representation. A fixed-width digit
# sequence with explicit radix position and sign. Each digit is in [0, 60).
# Layout (20 bytes):
#   digits : [16]u8   fixed buffer, int digits first then fractional
#   radix  : u8       index where fractional part begins (also = int digit count)
#   count  : u8       total digits used (0..=16)
#   sign   : i8       0 = positive, non-zero = negative
#   _pad   : u8       alignment filler
# Values beyond 16 total digits are a compile-time error.
SEX_MAX_DIGITS = 16
SEX = ir.LiteralStructType([
    ir.ArrayType(I8, SEX_MAX_DIGITS),
    I8, I8, I8, I8,
])
SEX_IDX_DIGITS = 0
SEX_IDX_RADIX = 1
SEX_IDX_COUNT = 2
SEX_IDX_SIGN = 3

INT_WIDTH: dict[str, int] = {
    "i8": 8, "i16": 16, "i32": 32, "i64": 64,
    "u8": 8, "u16": 16, "u32": 32, "u64": 64,
}


@dataclass
class Variable:
    is_mut: bool
    ir_ref: ir.Value       # SSA value for step; alloca pointer for mut
    value_ty: ir.Type      # logical value type (not the pointer type)


@dataclass
class TabletsInfo:
    """Monomorphized helper functions and struct types for one (N, T) pair."""
    N: int
    elem_ty: ir.Type
    node_ty: ir.IdentifiedStructType   # {[N x T], used: i64, next: Node*}
    tablets_ty: ir.LiteralStructType   # {head: Node*, tail: Node*, len: i64}
    push: ir.Function
    get: ir.Function
    release: ir.Function


# Names that the user cannot shadow — they resolve to compiler intrinsics.
# "rat" is both a type and a construction intrinsic (rat(num, den) -> rat).
INTRINSICS: frozenset[str] = frozenset({"print", "println", "read_int", "rat"})


class Codegen:
    def __init__(self) -> None:
        self.module = ir.Module(name="tuppu")
        self.module.triple = llvm.get_default_triple()
        self.builder: ir.IRBuilder | None = None
        self.functions: dict[str, ir.Function] = {}
        self.scopes: list[dict[str, Variable]] = []
        self._strings: dict[bytes, ir.GlobalVariable] = {}
        self._str_counter = 0
        self._rat_reduce: ir.Function | None = None  # built lazily
        self._sex_to_rat: ir.Function | None = None
        self._sex_print: ir.Function | None = None
        self._trap: ir.Function | None = None
        # table name -> (global array, length, lo bound, element LLVM type)
        self._tables: dict[str, tuple[ir.GlobalVariable, int, int, ir.Type]] = {}
        # Tablets monomorphizations: key is (N, str(elem_type)).
        self._tablets_types: dict[tuple[int, str], "TabletsInfo"] = {}
        # User-defined structs: name -> LLVM struct type + ordered fields.
        self._struct_types: dict[str, ir.LiteralStructType] = {}
        self._struct_fields: dict[str, list[tuple[str, ir.Type]]] = {}
        self._init_runtime_externs()

    def _init_runtime_externs(self) -> None:
        """Declare the libc functions our intrinsics lower to."""
        i8ptr = I8.as_pointer()
        self.printf = ir.Function(
            self.module,
            ir.FunctionType(I32, [i8ptr], var_arg=True),
            name="printf",
        )
        self.scanf = ir.Function(
            self.module,
            ir.FunctionType(I32, [i8ptr], var_arg=True),
            name="scanf",
        )
        self._malloc: ir.Function | None = None  # lazy
        self._free: ir.Function | None = None
        self._write: ir.Function | None = None
        self._fflush: ir.Function | None = None

    def _get_malloc(self) -> ir.Function:
        if self._malloc is None:
            self._malloc = ir.Function(
                self.module,
                ir.FunctionType(I8.as_pointer(), [I64]),
                name="malloc",
            )
        return self._malloc

    def _get_free(self) -> ir.Function:
        if self._free is None:
            self._free = ir.Function(
                self.module,
                ir.FunctionType(ir.VoidType(), [I8.as_pointer()]),
                name="free",
            )
        return self._free

    def _get_write(self) -> ir.Function:
        if self._write is None:
            self._write = ir.Function(
                self.module,
                ir.FunctionType(I64, [I32, I8.as_pointer(), I64]),
                name="write",
            )
        return self._write

    def _get_fflush(self) -> ir.Function:
        if self._fflush is None:
            self._fflush = ir.Function(
                self.module,
                ir.FunctionType(I32, [I8.as_pointer()]),
                name="fflush",
            )
        return self._fflush

    # --- top level ---

    def gen(self, prog: A.Program) -> ir.Module:
        self.comptime = Comptime(prog)

        # Phase 0: build struct LLVM types. Ordered so a struct referenced by
        # a later struct (or by function signatures) is always ready.
        self._register_structs(
            [d for d in prog.decls if isinstance(d, A.StructDecl)]
        )

        # Phase 1: forward-declare all user functions.
        for decl in prog.decls:
            if isinstance(decl, A.FnDecl):
                self._declare_fn(decl)
            elif isinstance(decl, A.TableDecl):
                pass  # handled in phase 2 after function decls are visible
            elif isinstance(decl, A.StructDecl):
                pass  # already handled in phase 0
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

        # Phase 3: emit function bodies. Tables are already visible so
        # table[i] access works from user code.
        for decl in prog.decls:
            if isinstance(decl, A.FnDecl):
                self._gen_fn_body(decl)
        return self.module

    def _declare_fn(self, fn: A.FnDecl) -> None:
        if fn.name in INTRINSICS:
            raise CodegenError(
                f"cannot define {fn.name!r}: it is a built-in intrinsic"
            )
        if fn.name in self.functions:
            raise CodegenError(f"duplicate function {fn.name!r}")
        param_types = [self._lower_type(p.type) for p in fn.params]
        ret_type = self._lower_type(fn.return_type) if fn.return_type else ir.VoidType()
        fn_type = ir.FunctionType(ret_type, param_types)
        llvm_fn = ir.Function(self.module, fn_type, name=fn.name)
        for i, p in enumerate(fn.params):
            llvm_fn.args[i].name = p.name
        self.functions[fn.name] = llvm_fn

    def _gen_fn_body(self, fn: A.FnDecl) -> None:
        if fn.name == "main":
            if not (isinstance(fn.return_type, A.TypeName) and fn.return_type.name == "i32"):
                raise CodegenError("main must declare -> i32")

        llvm_fn = self.functions[fn.name]
        entry = llvm_fn.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry)
        self.scopes = [{}]

        # Parameters: bound like step bindings — immutable, direct SSA refs.
        for i, p in enumerate(fn.params):
            arg = llvm_fn.args[i]
            self.scopes[-1][p.name] = Variable(
                is_mut=False, ir_ref=arg, value_ty=arg.type,
            )

        value = self._gen_expr(fn.body)

        if self._is_terminated():
            # Body already returned via yield.
            return
        if fn.return_type is None:
            self.builder.ret_void()
        else:
            if value is None:
                raise CodegenError(
                    f"function {fn.name!r} must produce a value for return type "
                    f"{fn.return_type}, but its body has no trailing expression"
                )
            expected = self._lower_type(fn.return_type)
            self.builder.ret(self._coerce(value, expected))

    def _is_terminated(self) -> bool:
        assert self.builder is not None
        return self.builder.block.is_terminated

    # --- types ---

    def _lower_type(self, t: A.TypeExpr) -> ir.Type:
        if isinstance(t, A.TypeName):
            if t.name in INT_WIDTH:
                return ir.IntType(INT_WIDTH[t.name])
            if t.name == "bool":
                return I1
            if t.name == "rat":
                return RAT
            # sex/dish now has a distinct digit-form representation so its
            # Babylonian identity survives to runtime. Coercion between sex
            # and rat is a real conversion, not a no-op.
            if t.name in ("sex", "dish"):
                return SEX
            if t.name in self._struct_types:
                return self._struct_types[t.name]
            raise CodegenError(f"type {t.name!r} not supported in this stage")
        if isinstance(t, A.TypeTablets):
            elem = self._lower_type(t.element)
            return self._get_tablets(t.size, elem).tablets_ty
        if isinstance(t, A.TypePointer):
            elem = self._lower_type(t.element)
            return elem.as_pointer()
        raise CodegenError(
            f"complex types not supported in this stage: {type(t).__name__}"
        )

    def _register_structs(self, decls: list[A.StructDecl]) -> None:
        """Build LLVM types for user-defined structs. Dependencies resolved
        via topological sort so a struct can reference another regardless of
        source order. Direct cycles are rejected — v0.1 has no recursive
        structs (which would require identified-type heap indirection)."""
        by_name = {d.name: d for d in decls}
        state: dict[str, int] = {}  # 0=unseen, 1=in-progress, 2=done

        def visit(d: A.StructDecl) -> None:
            st = state.get(d.name, 0)
            if st == 2:
                return
            if st == 1:
                raise CodegenError(
                    f"struct {d.name!r}: recursive structs are not supported"
                )
            state[d.name] = 1
            for _fname, ftype in d.fields:
                if isinstance(ftype, A.TypeName) and ftype.name in by_name:
                    visit(by_name[ftype.name])
            state[d.name] = 2
            field_tys = [self._lower_type(ftype) for _, ftype in d.fields]
            struct_ty = ir.LiteralStructType(field_tys)
            self._struct_types[d.name] = struct_ty
            self._struct_fields[d.name] = list(
                zip([n for n, _ in d.fields], field_tys)
            )

        for d in decls:
            visit(d)

    def _struct_name_for(self, llvm_ty: ir.Type) -> str | None:
        for name, ty in self._struct_types.items():
            if ty is llvm_ty:
                return name
        return None

    def _coerce(self, value: ir.Value, target_ty: ir.Type) -> ir.Value:
        """Insert a cast instruction if value's type differs from target_ty.
        Handles integer widening (sext/zext), integer narrowing (trunc),
        and i64<->rat and sex<->rat conversions."""
        if value.type == target_ty:
            return value
        assert self.builder is not None

        # Sex conversions. Sex is a compile-time-distinct type now; going
        # to rat requires a runtime reduction of the digit sequence.
        if value.type == SEX:
            if target_ty == RAT:
                return self.builder.call(self._get_sex_to_rat(), [value])
            if isinstance(target_ty, ir.IntType):
                # sex → iN: reduce to rat, then truncate.
                as_rat = self._coerce(value, RAT)
                return self._coerce(as_rat, target_ty)
        if target_ty == SEX:
            # For now we refuse implicit construction of sex from numeric
            # types. A future phase will add regularity-checked rat → sex.
            raise CodegenError(
                f"converting {value.type} to sex is not yet supported"
            )

        # Rat conversions.
        if value.type == RAT and isinstance(target_ty, ir.IntType):
            # rat as iN: truncate toward zero via signed division of num/den.
            num = self.builder.extract_value(value, 0)
            den = self.builder.extract_value(value, 1)
            result = self.builder.sdiv(num, den)
            return self._coerce(result, target_ty)  # narrow/widen to target width
        if target_ty == RAT and isinstance(value.type, ir.IntType):
            # iN as rat: widen to i64, then build {num: x, den: 1} (already reduced).
            num_i64 = self._coerce(value, I64)
            undef = ir.Constant(RAT, ir.Undefined)
            with_num = self.builder.insert_value(undef, num_i64, 0)
            return self.builder.insert_value(with_num, ir.Constant(I64, 1), 1)

        if isinstance(value.type, ir.IntType) and isinstance(target_ty, ir.IntType):
            sw, tw = value.type.width, target_ty.width
            if tw > sw:
                # Widening: zero-extend booleans, sign-extend other integers.
                if sw == 1:
                    return self.builder.zext(value, target_ty)
                return self.builder.sext(value, target_ty)
            if tw < sw:
                return self.builder.trunc(value, target_ty)
            return value
        raise CodegenError(f"cannot coerce {value.type} to {target_ty}")

    # --- scope / bindings ---

    def _bind(self, name: str, var: Variable) -> None:
        if name in self.scopes[-1]:
            raise CodegenError(f"redefinition of {name!r} in same scope")
        self.scopes[-1][name] = var

    def _lookup(self, name: str) -> Variable:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise CodegenError(f"undefined name {name!r}")

    def _alloca_entry(self, ty: ir.Type, name: str) -> ir.Value:
        """Emit an alloca in the entry block so mem2reg can promote it later."""
        assert self.builder is not None
        saved = self.builder.block
        entry = self.builder.function.entry_basic_block
        self.builder.position_at_start(entry)
        slot = self.builder.alloca(ty, name=name)
        self.builder.position_at_end(saved)
        return slot

    # --- statements ---

    def _gen_stmt(self, s: A.Stmt) -> None:
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
            self._gen_expr(s.expr); return  # discard value
        raise CodegenError(f"statement not supported yet: {type(s).__name__}")

    def _gen_release(self, s: A.ReleaseStmt) -> None:
        var = self._lookup(s.name)
        info = self._tablets_info_for(var.value_ty)
        if info is None:
            raise CodegenError(f"release requires a tablets, got {var.value_ty}")
        if not var.is_mut:
            raise CodegenError(f"cannot release step-bound tablets {s.name!r}")
        assert self.builder is not None
        self.builder.call(info.release, [var.ir_ref])

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

        raise CodegenError(
            f"for: cannot iterate over value of type {iter_val.type}"
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
        self._emit_counted_loop(
            length,
            lambda i: self.builder.call(info.get, [slot, i]),
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
        cond = self._gen_expr(w.cond)
        if cond is None or cond.type != I1:
            raise CodegenError("while condition must be a bool expression")
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
            if isinstance(ret_ty, ir.VoidType):
                self.builder.ret_void()
            else:
                raise CodegenError("bare yield in non-void function")
            return
        val = self._gen_expr(y.value)
        if val is None:
            raise CodegenError("yield value diverged")
        self.builder.ret(self._coerce(val, ret_ty))

    def _gen_binding(self, b: A.Binding) -> None:
        # Uninitialized mut binding with explicit type: zero-initialize.
        if b.init is None:
            assert b.is_mut and b.type_ann is not None  # parser enforces this
            ty = self._lower_type(b.type_ann)
            slot = self._alloca_entry(ty, b.name)
            assert self.builder is not None
            self.builder.store(ir.Constant(ty, None), slot)
            self._bind(b.name, Variable(is_mut=True, ir_ref=slot, value_ty=ty))
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
        else:
            self._bind(b.name, Variable(is_mut=False, ir_ref=init_val, value_ty=init_val.type))

    def _gen_assign(self, a: A.Assign) -> None:
        var = self._lookup(a.name)
        if not var.is_mut:
            raise CodegenError(f"cannot assign to step binding {a.name!r}")
        assert self.builder is not None
        value = self._gen_expr(a.value)
        if value is None:
            raise CodegenError(f"assignment RHS for {a.name!r} has no value")
        self.builder.store(self._coerce(value, var.value_ty), var.ir_ref)

    # --- expressions ---

    def _gen_expr(self, e: A.Expr) -> ir.Value | None:
        """Generate code for an expression. Returns None if the expression
        diverges (e.g. a block where all paths yield) or has no value (e.g.
        an `if` without `else`, which produces unit)."""
        if isinstance(e, A.IntLit):
            return ir.Constant(I64, e.value)
        if isinstance(e, A.CharLit):
            return ir.Constant(I8, e.value)
        if isinstance(e, A.BoolLit):
            return ir.Constant(I1, 1 if e.value else 0)
        if isinstance(e, A.StringLit):
            return self._gen_string_lit(e.value)
        if isinstance(e, A.SexLit):
            return self._gen_sex_lit(e)
        if isinstance(e, A.StructLit):
            return self._gen_struct_lit(e)
        if isinstance(e, A.Field):
            return self._gen_field(e)
        if isinstance(e, A.Index):
            return self._gen_index(e)
        if isinstance(e, A.Ident):
            return self._gen_ident(e)
        if isinstance(e, A.Block):
            return self._gen_block(e)
        if isinstance(e, A.IfExpr):
            return self._gen_if_expr(e)
        if isinstance(e, A.Unary):
            return self._gen_unary(e)
        if isinstance(e, A.Binary):
            return self._gen_binary(e)
        if isinstance(e, A.Call):
            return self._gen_call(e)
        if isinstance(e, A.Cast):
            value = self._gen_expr(e.value)
            if value is None:
                raise CodegenError("cannot cast a diverging expression")
            target = self._lower_type(e.type)
            return self._coerce(value, target)
        raise CodegenError(f"expression not supported yet: {type(e).__name__}")

    def _gen_if_expr(self, e: A.IfExpr) -> ir.Value | None:
        assert self.builder is not None
        cond = self._gen_expr(e.cond)
        if cond is None:
            raise CodegenError("if condition diverged")
        if cond.type != I1:
            raise CodegenError(f"if condition must be bool, got {cond.type}")

        fn = self.builder.function
        then_bb = fn.append_basic_block("if.then")
        merge_bb = fn.append_basic_block("if.merge")

        # No else — value is always None (unit). Useful only in statement position.
        if e.else_ is None:
            self.builder.cbranch(cond, then_bb, merge_bb)
            self.builder.position_at_end(then_bb)
            self._gen_expr(e.then)
            if not self._is_terminated():
                self.builder.branch(merge_bb)
            self.builder.position_at_end(merge_bb)
            return None

        else_bb = fn.append_basic_block("if.else")
        self.builder.cbranch(cond, then_bb, else_bb)

        self.builder.position_at_end(then_bb)
        then_val = self._gen_expr(e.then)
        then_end = self.builder.block
        # Snapshot whether the arm diverged BEFORE we insert the fall-through
        # branch to merge (which itself is a terminator).
        then_diverged = then_end.is_terminated
        if not then_diverged:
            self.builder.branch(merge_bb)

        self.builder.position_at_end(else_bb)
        else_val = self._gen_expr(e.else_)
        else_end = self.builder.block
        else_diverged = else_end.is_terminated
        if not else_diverged:
            self.builder.branch(merge_bb)

        self.builder.position_at_end(merge_bb)

        # Both arms diverged: make the merge block a valid (unreachable) block
        # so the IR verifier is happy, and any outer code sees that we're
        # terminated too. This is the "diverging if" case.
        if then_diverged and else_diverged:
            self.builder.unreachable()
            return None

        # One side diverged — the value, if any, comes from the other.
        if then_diverged:
            return else_val
        if else_diverged:
            return then_val

        # Both sides reach merge.
        if then_val is None and else_val is None:
            return None
        if then_val is None or else_val is None:
            raise CodegenError(
                "if arms disagree: one has a trailing expression, the other does not"
            )
        if then_val.type != else_val.type:
            raise CodegenError(
                f"if arms have different types: {then_val.type} vs {else_val.type}"
            )
        phi = self.builder.phi(then_val.type)
        phi.add_incoming(then_val, then_end)
        phi.add_incoming(else_val, else_end)
        return phi

    def _gen_ident(self, e: A.Ident) -> ir.Value:
        var = self._lookup(e.name)
        assert self.builder is not None
        if var.is_mut:
            return self.builder.load(var.ir_ref, name=e.name)
        return var.ir_ref

    def _gen_block(self, b: A.Block) -> ir.Value | None:
        """Evaluate a block. Returns the value of its trailing expression, or
        None if the block has no tail or diverged before reaching it."""
        self.scopes.append({})
        try:
            for stmt in b.stmts:
                if self._is_terminated():
                    break   # dead code after a yield
                self._gen_stmt(stmt)
            if self._is_terminated():
                return None
            if b.tail is None:
                return None
            return self._gen_expr(b.tail)
        finally:
            self.scopes.pop()

    def _gen_unary(self, e: A.Unary) -> ir.Value:
        assert self.builder is not None
        operand = self._gen_expr(e.operand)
        if operand is None:
            raise CodegenError(f"unary {e.op} operand has no value")
        if e.op == "-":
            if operand.type == SEX:
                # Flip sign byte in place; digits untouched.
                sign = self.builder.extract_value(operand, SEX_IDX_SIGN)
                flipped = self.builder.xor(sign, ir.Constant(I8, 1))
                return self.builder.insert_value(operand, flipped, SEX_IDX_SIGN)
            if operand.type == RAT:
                num = self.builder.extract_value(operand, 0)
                return self.builder.insert_value(operand, self.builder.neg(num), 0)
            if not isinstance(operand.type, ir.IntType) or operand.type.width < 8:
                raise CodegenError(f"unary - requires integer, got {operand.type}")
            return self.builder.neg(operand)
        if e.op == "!":
            if operand.type != I1:
                raise CodegenError(f"unary ! requires bool, got {operand.type}")
            return self.builder.not_(operand)
        raise CodegenError(f"unknown unary op: {e.op}")

    def _gen_binary(self, e: A.Binary) -> ir.Value:
        assert self.builder is not None
        lhs = self._gen_expr(e.lhs)
        rhs = self._gen_expr(e.rhs)
        if lhs is None or rhs is None:
            raise CodegenError(f"operand of binary {e.op} has no value")
        op = e.op

        # Sex operands lower to rat arithmetic for now — this is the
        # warning path the type checker already announced. Phase 3 will
        # replace this with native digit-sequence arithmetic.
        if lhs.type == SEX:
            lhs = self._coerce(lhs, RAT)
        if rhs.type == SEX:
            rhs = self._coerce(rhs, RAT)

        # --- rat arithmetic and comparison ---
        if lhs.type == RAT and rhs.type == RAT:
            return self._gen_rat_binary(op, lhs, rhs)

        if op in ("+", "-", "*", "/", "%"):
            if lhs.type != rhs.type or not isinstance(lhs.type, ir.IntType):
                raise CodegenError(
                    f"{op} requires matching integer types, got {lhs.type} and {rhs.type}"
                )
            return {
                "+": self.builder.add,
                "-": self.builder.sub,
                "*": self.builder.mul,
                "/": self.builder.sdiv,
                "%": self.builder.srem,
            }[op](lhs, rhs)

        if op in ("<", "<=", ">", ">=", "==", "!="):
            # Mixed-width integer compare: promote to the wider type
            # (matches _unify_if_arms on the checker side).
            if (
                isinstance(lhs.type, ir.IntType)
                and isinstance(rhs.type, ir.IntType)
                and lhs.type.width != rhs.type.width
            ):
                target = lhs.type if lhs.type.width >= rhs.type.width else rhs.type
                lhs = self._coerce(lhs, target)
                rhs = self._coerce(rhs, target)
            if lhs.type != rhs.type or not isinstance(lhs.type, ir.IntType):
                raise CodegenError(
                    f"comparison requires matching types, got {lhs.type} and {rhs.type}"
                )
            return self.builder.icmp_signed(op, lhs, rhs)

        if op in ("&&", "||"):
            if lhs.type != I1 or rhs.type != I1:
                raise CodegenError(f"{op} requires bool operands")
            # Non-short-circuit for now — see §7 for branch-based impl.
            return self.builder.and_(lhs, rhs) if op == "&&" else self.builder.or_(lhs, rhs)

        raise CodegenError(f"unsupported binary op: {op}")

    def _gen_call(self, e: A.Call) -> ir.Value | None:
        # Method call on a tablets value: t.push(x), etc.
        if isinstance(e.callee, A.Field) and isinstance(e.callee.target, A.Ident):
            try:
                var = self._lookup(e.callee.target.name)
            except CodegenError:
                var = None
            if var is not None:
                info = self._tablets_info_for(var.value_ty)
                if info is not None:
                    return self._gen_tablets_method(info, var, e.callee.name, e.args)

        if not isinstance(e.callee, A.Ident):
            raise CodegenError("only direct function calls are supported")
        name = e.callee.name

        # Intrinsics dispatch first so user-defined shadows can't occur
        # (they'd have been rejected at declaration time anyway).
        if name == "print":
            return self._gen_print(e.args, newline=False)
        if name == "println":
            return self._gen_print(e.args, newline=True)
        if name == "read_int":
            return self._gen_read_int(e.args)
        if name == "rat":
            return self._gen_rat_ctor(e.args)

        fn = self.functions.get(name)
        if fn is None:
            raise CodegenError(f"unknown function {name!r}")
        if len(e.args) != len(fn.args):
            raise CodegenError(
                f"{name} expects {len(fn.args)} args, got {len(e.args)}"
            )
        assert self.builder is not None
        call_args = []
        for i, arg in enumerate(e.args):
            v = self._gen_expr(arg)
            if v is None:
                raise CodegenError(f"argument {i} of call to {name} has no value")
            call_args.append(self._coerce(v, fn.args[i].type))
        return self.builder.call(fn, call_args)

    # --- intrinsics: stdlib I/O -----------------------------------------

    def _str_ptr(self, data: bytes) -> ir.Value:
        """Return an i8* pointing to a global, NUL-terminated copy of `data`.
        Deduplicates identical strings via `self._strings`."""
        assert self.builder is not None
        g = self._strings.get(data)
        if g is None:
            payload = data + b"\0"
            ty = ir.ArrayType(I8, len(payload))
            g = ir.GlobalVariable(self.module, ty, name=f".str.{self._str_counter}")
            self._str_counter += 1
            g.linkage = "internal"
            g.global_constant = True
            g.initializer = ir.Constant(ty, bytearray(payload))
            self._strings[data] = g
        zero = ir.Constant(I32, 0)
        return self.builder.gep(g, [zero, zero], inbounds=True)

    def _gen_print(self, args: list[A.Expr], *, newline: bool) -> None:
        if not args:
            raise CodegenError(
                f"{'println' if newline else 'print'} takes at least one argument"
            )
        assert self.builder is not None
        # Each argument is emitted without a newline; if `newline=True`
        # the trailing newline goes AFTER the last argument only.
        for i, arg in enumerate(args):
            val = self._gen_expr(arg)
            if val is None:
                raise CodegenError("print argument has no value")
            last = (i == len(args) - 1)
            self._emit_one_print(val, newline=(newline and last))

    def _emit_one_print(self, val: ir.Value, *, newline: bool) -> None:
        assert self.builder is not None
        # Dispatch on runtime IR type.
        if val.type == I1:
            fmt = "%s\n" if newline else "%s"
            choice = self.builder.select(
                val, self._str_ptr(b"true"), self._str_ptr(b"false"),
            )
            self.builder.call(self.printf, [self._str_ptr(fmt.encode()), choice])
            return

        if isinstance(val.type, ir.IntType):
            fmt = "%lld\n" if newline else "%lld"
            v64 = self._coerce(val, I64)
            self.builder.call(self.printf, [self._str_ptr(fmt.encode()), v64])
            return

        if val.type == SEX:
            self._emit_sex_print(val, newline=newline)
            return

        # Seal types must be checked before RAT, since a user seal may be
        # structurally equal to the rat struct at the LLVM level.
        if self._is_str_value(val.type):
            ptr = self.builder.extract_value(val, 0)
            length = self.builder.extract_value(val, 1)
            null_file = ir.Constant(I8.as_pointer(), None)
            self.builder.call(self._get_fflush(), [null_file])
            stdout_fd = ir.Constant(I32, 1)
            self.builder.call(self._get_write(), [stdout_fd, ptr, length])
            if newline:
                self.builder.call(self._get_write(), [
                    stdout_fd, self._str_ptr(b"\n"), ir.Constant(I64, 1),
                ])
            return

        if val.type == RAT:
            num = self.builder.extract_value(val, 0)
            den = self.builder.extract_value(val, 1)
            fmt = "%lld/%lld\n" if newline else "%lld/%lld"
            self.builder.call(self.printf, [self._str_ptr(fmt.encode()), num, den])
            return

        raise CodegenError(f"print: unsupported value type {val.type}")

    def _gen_read_int(self, args: list[A.Expr]) -> ir.Value:
        if args:
            raise CodegenError("read_int takes no arguments")
        assert self.builder is not None
        slot = self._alloca_entry(I64, "readint_slot")
        self.builder.call(self.scanf, [self._str_ptr(b"%lld"), slot])
        return self.builder.load(slot, name="readint_val")

    # --- intrinsics: rat constructor ------------------------------------

    def _gen_rat_ctor(self, args: list[A.Expr]) -> ir.Value:
        if len(args) != 2:
            raise CodegenError("rat() takes exactly two arguments (num, den)")
        assert self.builder is not None
        num = self._gen_expr(args[0])
        den = self._gen_expr(args[1])
        if num is None or den is None:
            raise CodegenError("rat() argument has no value")
        num = self._coerce(num, I64)
        den = self._coerce(den, I64)
        return self.builder.call(self._get_rat_reduce(), [num, den])

    # --- field access ---------------------------------------------------

    def _gen_field(self, e: A.Field) -> ir.Value:
        assert self.builder is not None

        # Fast path: field on a named tablets variable — GEP directly, skip
        # loading the whole struct.
        if isinstance(e.target, A.Ident):
            try:
                var = self._lookup(e.target.name)
            except CodegenError:
                var = None
            if var is not None and self._tablets_info_for(var.value_ty) is not None:
                return self._gen_tablets_field(var, e.name)

        target = self._gen_expr(e.target)
        if target is None:
            raise CodegenError("field access target has no value")
        # Check user-defined structs BEFORE rat: a `struct P { x: i64, y: i64 }`
        # is structurally equal to RAT at the LLVM level, but identity
        # comparison against _struct_types distinguishes them correctly.
        struct_name = self._struct_name_for(target.type)
        if struct_name is not None:
            for i, (fname, _fty) in enumerate(self._struct_fields[struct_name]):
                if fname == e.name:
                    return self.builder.extract_value(target, i)
            raise CodegenError(
                f"struct {struct_name!r} has no field {e.name!r}"
            )
        if target.type == RAT:
            if e.name == "num":
                return self.builder.extract_value(target, 0)
            if e.name == "den":
                return self.builder.extract_value(target, 1)
            raise CodegenError(f"rat has no field {e.name!r}; only num and den")
        if target.type == SEX and e.name in ("num", "den"):
            # Sex has no literal num/den fields; reduce first.
            as_rat = self._coerce(target, RAT)
            return self.builder.extract_value(as_rat, 0 if e.name == "num" else 1)
        raise CodegenError(f"field access on {target.type} not supported yet")

    def _gen_sex_lit(self, e: A.SexLit) -> ir.Value:
        """Lower a sex literal to a digit-form constant. The lexer has
        already validated each digit is in [0, 60)."""
        int_digits = e.int_digits
        frac_digits = e.frac_digits if e.frac_digits is not None else []
        all_digits = int_digits + frac_digits
        if len(all_digits) > SEX_MAX_DIGITS:
            raise CodegenError(
                f"sex literal has {len(all_digits)} digits; max is "
                f"{SEX_MAX_DIGITS}"
            )
        # Pad to fixed width so every sex value has identical layout.
        padded = all_digits + [0] * (SEX_MAX_DIGITS - len(all_digits))
        digit_arr = ir.Constant(
            ir.ArrayType(I8, SEX_MAX_DIGITS),
            padded,
        )
        radix = len(int_digits)
        count = len(all_digits)
        return ir.Constant(SEX, (
            digit_arr,
            ir.Constant(I8, radix),
            ir.Constant(I8, count),
            ir.Constant(I8, 0),   # positive by construction; unary - flips
            ir.Constant(I8, 0),   # pad
        ))

    def _gen_string_lit(self, data: bytes) -> ir.Value:
        """Lower a string literal to a `str` seal value: `{ ptr: *u8, len: i64 }`.
        Backing bytes live in a deduped internal global."""
        assert self.builder is not None
        if "str" not in self._struct_types:
            raise CodegenError(
                "string literal used but `str` seal is not registered "
                "(driver should have auto-injected it)"
            )
        struct_ty = self._struct_types["str"]
        ptr = self._str_ptr(data)                      # i8*
        length = ir.Constant(I64, len(data))
        value: ir.Value = ir.Constant(struct_ty, ir.Undefined)
        value = self.builder.insert_value(value, ptr, 0)
        value = self.builder.insert_value(value, length, 1)
        return value

    def _is_str_value(self, llvm_ty: ir.Type) -> bool:
        ty = self._struct_types.get("str")
        return ty is not None and ty is llvm_ty

    def _gen_struct_lit(self, e: A.StructLit) -> ir.Value:
        assert self.builder is not None
        if e.name not in self._struct_types:
            raise CodegenError(f"unknown struct {e.name!r}")
        struct_ty = self._struct_types[e.name]
        fields = self._struct_fields[e.name]
        provided: dict[str, A.Expr] = dict(e.fields)
        value: ir.Value = ir.Constant(struct_ty, ir.Undefined)
        for i, (fname, fty) in enumerate(fields):
            if fname not in provided:
                raise CodegenError(
                    f"struct {e.name!r}: missing field {fname!r}"
                )
            fv = self._gen_expr(provided[fname])
            if fv is None:
                raise CodegenError(
                    f"struct {e.name!r} field {fname!r}: initializer has no value"
                )
            value = self.builder.insert_value(value, self._coerce(fv, fty), i)
        return value

    # --- tablets method/field/index dispatch -----------------------------

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
            val = self._gen_expr(args[0])
            if val is None:
                raise CodegenError("tablets.push argument has no value")
            val = self._coerce(val, info.elem_ty)
            self.builder.call(info.push, [ptr, val])
            return None
        raise CodegenError(f"tablets has no method {method!r}")

    def _gen_tablets_field(self, var: Variable, field_name: str) -> ir.Value:
        assert self.builder is not None
        if not var.is_mut:
            raise CodegenError("tablets must be a mut binding")
        if field_name == "len":
            len_addr = self.builder.gep(
                var.ir_ref,
                [ir.Constant(I32, 0), ir.Constant(I32, 2)],
                inbounds=True,
            )
            return self.builder.load(len_addr)
        raise CodegenError(f"tablets has no field {field_name!r}; only .len")

    # --- rat arithmetic -------------------------------------------------

    def _gen_rat_binary(self, op: str, lhs: ir.Value, rhs: ir.Value) -> ir.Value:
        assert self.builder is not None
        b = self.builder
        a_num = b.extract_value(lhs, 0)
        a_den = b.extract_value(lhs, 1)
        b_num = b.extract_value(rhs, 0)
        b_den = b.extract_value(rhs, 1)

        if op in ("+", "-"):
            # (a/p ± b/q) = (a*q ± b*p) / (p*q)
            left  = b.mul(a_num, b_den)
            right = b.mul(b_num, a_den)
            num = b.add(left, right) if op == "+" else b.sub(left, right)
            den = b.mul(a_den, b_den)
            return b.call(self._get_rat_reduce(), [num, den])
        if op == "*":
            return b.call(self._get_rat_reduce(),
                          [b.mul(a_num, b_num), b.mul(a_den, b_den)])
        if op == "/":
            # a/p ÷ b/q = (a*q) / (p*b) — reduce handles sign and zero-trap.
            return b.call(self._get_rat_reduce(),
                          [b.mul(a_num, b_den), b.mul(a_den, b_num)])
        if op in ("==", "!="):
            # Since rats are always reduced (gcd=1, den>0), equal iff fields match.
            num_eq = b.icmp_signed("==", a_num, b_num)
            den_eq = b.icmp_signed("==", a_den, b_den)
            eq = b.and_(num_eq, den_eq)
            return eq if op == "==" else b.not_(eq)
        if op in ("<", "<=", ">", ">="):
            # With den>0 on both sides: a/p < b/q  <=>  a*q < b*p.
            left  = b.mul(a_num, b_den)
            right = b.mul(b_num, a_den)
            return b.icmp_signed(op, left, right)
        raise CodegenError(f"rat does not support operator {op}")

    # --- tables (comptime lookup) ---------------------------------------

    def _emit_table(self, decl: A.TableDecl) -> None:
        try:
            values = self.comptime.eval_table(decl)
            lo = self.comptime.eval_constant_expr(decl.lo)
        except ComptimeError as e:
            raise CodegenError(f"table {decl.name!r}: {e}") from None

        elem_ty = self._lower_type(decl.element_type)
        array_ty = ir.ArrayType(elem_ty, len(values))

        try:
            constants = [self._py_value_to_constant(v, elem_ty) for v in values]
        except CodegenError as e:
            raise CodegenError(f"table {decl.name!r}: {e}") from None

        g = ir.GlobalVariable(self.module, array_ty, name=decl.name)
        g.linkage = "internal"
        g.global_constant = True
        g.initializer = ir.Constant(array_ty, constants)

        self._tables[decl.name] = (g, len(values), lo, elem_ty)

    def _py_value_to_constant(self, v, target_ty: ir.Type) -> ir.Constant:
        if target_ty == I1:
            if isinstance(v, bool):
                return ir.Constant(I1, 1 if v else 0)
            raise CodegenError(f"expected bool for i1, got {type(v).__name__}")
        if isinstance(target_ty, ir.IntType):
            if isinstance(v, int) and not isinstance(v, bool):
                return ir.Constant(target_ty, v)
            raise CodegenError(
                f"expected int for {target_ty}, got {type(v).__name__}"
            )
        if target_ty == RAT:
            if isinstance(v, tuple) and len(v) == 2:
                return ir.Constant(RAT, (
                    ir.Constant(I64, v[0]),
                    ir.Constant(I64, v[1]),
                ))
            raise CodegenError(
                f"expected (num, den) tuple for rat, got {type(v).__name__}"
            )
        raise CodegenError(f"cannot lower comptime {v!r} to {target_ty}")

    def _gen_index(self, e: A.Index) -> ir.Value:
        assert self.builder is not None
        # Comptime table lookup
        if isinstance(e.target, A.Ident) and e.target.name in self._tables:
            g, size, lo, _elem_ty = self._tables[e.target.name]
            idx = self._gen_expr(e.index)
            if idx is None:
                raise CodegenError("table index has no value")
            idx = self._coerce(idx, I64)
            if lo != 0:
                idx = self.builder.sub(idx, ir.Constant(I64, lo))
            self._emit_bounds_trap(idx, size)
            zero = ir.Constant(I32, 0)
            elem_ptr = self.builder.gep(g, [zero, idx], inbounds=True)
            return self.builder.load(elem_ptr)

        # Tablets indexing (dynamic bounds check vs len)
        if isinstance(e.target, A.Ident):
            try:
                var = self._lookup(e.target.name)
            except CodegenError:
                var = None
            if var is not None:
                info = self._tablets_info_for(var.value_ty)
                if info is not None:
                    return self._gen_tablets_index(info, var, e.index)

        # str indexing: bounds-checked byte load through s.ptr.
        target = self._gen_expr(e.target)
        if target is not None and self._is_str_value(target.type):
            idx = self._gen_expr(e.index)
            if idx is None:
                raise CodegenError("str index has no value")
            idx_i64 = self._coerce(idx, I64)
            ptr = self.builder.extract_value(target, 0)    # i8*
            length = self.builder.extract_value(target, 1) # i64
            self._emit_dynamic_bounds_trap(idx_i64, length)
            byte_ptr = self.builder.gep(ptr, [idx_i64], inbounds=True)
            return self.builder.load(byte_ptr)

        raise CodegenError("indexing is only supported on tables, tablets, and str")

    def _gen_tablets_index(
        self, info: TabletsInfo, var: Variable, idx_expr: A.Expr,
    ) -> ir.Value:
        if not var.is_mut:
            raise CodegenError("tablets indexing requires a mut binding")
        assert self.builder is not None
        idx = self._gen_expr(idx_expr)
        if idx is None:
            raise CodegenError("tablets index has no value")
        idx = self._coerce(idx, I64)
        len_addr = self.builder.gep(
            var.ir_ref,
            [ir.Constant(I32, 0), ir.Constant(I32, 2)],
            inbounds=True,
        )
        length = self.builder.load(len_addr)
        self._emit_dynamic_bounds_trap(idx, length)
        return self.builder.call(info.get, [var.ir_ref, idx])

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

    def _get_tablets(self, N: int, elem_ty: ir.Type) -> TabletsInfo:
        """Return (building once, caching thereafter) the struct types and
        helper functions for tablets[N]T with the given element type."""
        key = (N, str(elem_ty))
        existing = self._tablets_types.get(key)
        if existing is not None:
            return existing

        suffix = f"{elem_ty}_{N}".replace(" ", "_").replace("{", "").replace("}", "")
        node_ty = ir.global_context.get_identified_type(f"Node_{suffix}")
        # Identified types live in the global context and persist across
        # Codegen instances. Only set the body the first time we see one.
        if node_ty.is_opaque:
            node_ty.set_body(
                ir.ArrayType(elem_ty, N),   # items
                I64,                         # used
                node_ty.as_pointer(),        # next
            )
        tablets_ty = ir.LiteralStructType([
            node_ty.as_pointer(),        # head
            node_ty.as_pointer(),        # tail
            I64,                         # len
        ])

        info = TabletsInfo(
            N=N, elem_ty=elem_ty, node_ty=node_ty, tablets_ty=tablets_ty,
            push=self._build_tablets_push(N, elem_ty, node_ty, tablets_ty, suffix),
            get=self._build_tablets_get(N, elem_ty, node_ty, tablets_ty, suffix),
            release=self._build_tablets_release(N, elem_ty, node_ty, tablets_ty, suffix),
        )
        self._tablets_types[key] = info
        return info

    def _build_tablets_push(
        self, N: int, elem_ty: ir.Type,
        node_ty: ir.IdentifiedStructType, tablets_ty: ir.LiteralStructType,
        suffix: str,
    ) -> ir.Function:
        fn = ir.Function(
            self.module,
            ir.FunctionType(ir.VoidType(), [tablets_ty.as_pointer(), elem_ty]),
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

        # need.new: allocate via malloc, initialize, link into chain.
        b.position_at_end(need_new)
        # sizeof(node_ty) via GEP-from-null trick.
        size_ptr = b.gep(null_node, [ONE_I32], inbounds=False)
        node_size = b.ptrtoint(size_ptr, I64)
        raw = b.call(self._get_malloc(), [node_size])
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
        b.ret_void()

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

    def _build_tablets_release(
        self, N: int, elem_ty: ir.Type,
        node_ty: ir.IdentifiedStructType, tablets_ty: ir.LiteralStructType,
        suffix: str,
    ) -> ir.Function:
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
        next_node = b.load(b.gep(cur_phi, [ZERO_I32, ir.Constant(I32, 2)], inbounds=True))
        raw = b.bitcast(cur_phi, I8.as_pointer())
        b.call(self._get_free(), [raw])
        cur_phi.add_incoming(next_node, body)
        b.branch(loop)

        b.position_at_end(done)
        b.store(null_node, head_addr)
        b.store(null_node, tail_addr)
        b.store(ir.Constant(I64, 0), len_addr)
        b.ret_void()
        return fn

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

    def _get_sex_print(self) -> ir.Function:
        """Emit (once) `__tuppu_sex_print(sex, newline: i1)` — prints a sex
        value in Babylonian notation: integer digits space-separated,
        a semicolon before the fractional digits (if any), then fractional
        digits space-separated. Negative sign printed as a leading `-`."""
        if self._sex_print is not None:
            return self._sex_print

        fn = ir.Function(
            self.module,
            ir.FunctionType(ir.VoidType(), [SEX, I1]),
            name="__tuppu_sex_print",
        )
        fn.args[0].name = "sx"
        fn.args[1].name = "newline"
        sx, want_nl = fn.args

        entry     = fn.append_basic_block("entry")
        neg_bb    = fn.append_basic_block("print.neg")
        int_loop  = fn.append_basic_block("int.loop")
        int_body  = fn.append_basic_block("int.body")
        int_next  = fn.append_basic_block("int.next")
        radix_bb  = fn.append_basic_block("radix")
        has_frac  = fn.append_basic_block("print.semi")
        frac_loop = fn.append_basic_block("frac.loop")
        frac_body = fn.append_basic_block("frac.body")
        frac_next = fn.append_basic_block("frac.next")
        maybe_nl  = fn.append_basic_block("maybe.nl")
        do_nl     = fn.append_basic_block("do.nl")
        done      = fn.append_basic_block("done")

        b = ir.IRBuilder(entry)
        # _str_ptr emits GEPs into self.builder — temporarily point it at
        # our local builder so the format constants live in this function.
        saved_builder = self.builder
        self.builder = b
        try:
            fmt_dash  = self._str_ptr(b"-")
            fmt_sp    = self._str_ptr(b" ")
            fmt_semi  = self._str_ptr(b";")
            fmt_nl    = self._str_ptr(b"\n")
            fmt_digit = self._str_ptr(b"%d")
        finally:
            self.builder = saved_builder
        slot = b.alloca(SEX, name="sex.slot")
        b.store(sx, slot)
        digits_addr = b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_DIGITS)],
            inbounds=True,
        )
        radix = b.sext(b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_RADIX)],
            inbounds=True,
        )), I64)
        count = b.sext(b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_COUNT)],
            inbounds=True,
        )), I64)
        sign = b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_SIGN)],
            inbounds=True,
        ))
        is_neg = b.icmp_signed("!=", sign, ir.Constant(I8, 0))
        b.cbranch(is_neg, neg_bb, int_loop)

        b.position_at_end(neg_bb)
        b.call(self.printf, [fmt_dash])
        b.branch(int_loop)

        # Integer-digit loop: print each digit, space-separated.
        b.position_at_end(int_loop)
        i_phi = b.phi(I64, "i")
        i_phi.add_incoming(ir.Constant(I64, 0), neg_bb)
        i_phi.add_incoming(ir.Constant(I64, 0), entry)
        done_int = b.icmp_signed(">=", i_phi, radix)
        b.cbranch(done_int, radix_bb, int_body)

        b.position_at_end(int_body)
        need_space = b.icmp_signed(">", i_phi, ir.Constant(I64, 0))
        with b.if_then(need_space):
            b.call(self.printf, [fmt_sp])
        dptr = b.gep(digits_addr, [ir.Constant(I32, 0), i_phi], inbounds=True)
        dval = b.zext(b.load(dptr), I32)
        b.call(self.printf, [fmt_digit, dval])
        b.branch(int_next)

        b.position_at_end(int_next)
        next_i = b.add(i_phi, ir.Constant(I64, 1))
        i_phi.add_incoming(next_i, int_next)
        b.branch(int_loop)

        # After integer digits, maybe print ';' and fractional digits.
        b.position_at_end(radix_bb)
        fractional = b.icmp_signed(">", count, radix)
        b.cbranch(fractional, has_frac, maybe_nl)

        b.position_at_end(has_frac)
        b.call(self.printf, [fmt_semi])
        b.branch(frac_loop)

        b.position_at_end(frac_loop)
        j_phi = b.phi(I64, "j")
        j_phi.add_incoming(radix, has_frac)
        done_frac = b.icmp_signed(">=", j_phi, count)
        b.cbranch(done_frac, maybe_nl, frac_body)

        b.position_at_end(frac_body)
        # First fractional digit immediately follows `;` with no space.
        need_sp2 = b.icmp_signed(">", j_phi, radix)
        with b.if_then(need_sp2):
            b.call(self.printf, [fmt_sp])
        fptr = b.gep(digits_addr, [ir.Constant(I32, 0), j_phi], inbounds=True)
        fval = b.zext(b.load(fptr), I32)
        b.call(self.printf, [fmt_digit, fval])
        b.branch(frac_next)

        b.position_at_end(frac_next)
        next_j = b.add(j_phi, ir.Constant(I64, 1))
        j_phi.add_incoming(next_j, frac_next)
        b.branch(frac_loop)

        b.position_at_end(maybe_nl)
        b.cbranch(want_nl, do_nl, done)

        b.position_at_end(do_nl)
        b.call(self.printf, [fmt_nl])
        b.branch(done)

        b.position_at_end(done)
        b.ret_void()

        self._sex_print = fn
        return fn

    def _emit_sex_print(self, val: ir.Value, *, newline: bool) -> None:
        assert self.builder is not None
        nl_flag = ir.Constant(I1, 1 if newline else 0)
        self.builder.call(self._get_sex_print(), [val, nl_flag])

    def _get_sex_to_rat(self) -> ir.Function:
        """Emit (once per module) `__tuppu_sex_to_rat(sex) -> rat`.

        Reconstructs an integer numerator by Horner-style evaluation over
        the digit sequence (each digit × 60^place), computes the implied
        denominator from (count - radix) fractional places, applies the
        sign bit, then delegates to `__tuppu_rat_reduce` for gcd reduction.
        The result is a normal rat value — all invariants preserved."""
        if self._sex_to_rat is not None:
            return self._sex_to_rat

        fn = ir.Function(
            self.module,
            ir.FunctionType(RAT, [SEX]),
            name="__tuppu_sex_to_rat",
        )
        fn.args[0].name = "sx"
        sx = fn.args[0]

        entry = fn.append_basic_block("entry")
        num_loop = fn.append_basic_block("num.loop")
        num_body = fn.append_basic_block("num.body")
        den_loop = fn.append_basic_block("den.loop")
        den_body = fn.append_basic_block("den.body")
        apply_sign = fn.append_basic_block("apply.sign")
        do_reduce = fn.append_basic_block("reduce")

        b = ir.IRBuilder(entry)

        # Spill the sex value to a stack slot so we can GEP into the digit
        # array by a runtime index. (LLVM can't index a struct field by a
        # non-constant, but it can GEP into an alloca.)
        slot = b.alloca(SEX, name="sex.slot")
        b.store(sx, slot)
        digits_addr = b.gep(
            slot,
            [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_DIGITS)],
            inbounds=True,
        )
        radix = b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_RADIX)],
            inbounds=True,
        ))
        count = b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_COUNT)],
            inbounds=True,
        ))
        sign = b.load(b.gep(
            slot, [ir.Constant(I32, 0), ir.Constant(I32, SEX_IDX_SIGN)],
            inbounds=True,
        ))
        count_i64 = b.sext(count, I64)
        radix_i64 = b.sext(radix, I64)
        frac_places = b.sub(count_i64, radix_i64)
        b.branch(num_loop)

        # num = sum(digits[i] * 60^(count-1-i)) computed Horner-style:
        #   num = 0; for i in 0..count: num = num*60 + digits[i]
        b.position_at_end(num_loop)
        num_phi = b.phi(I64, "num")
        i_phi = b.phi(I64, "i")
        num_phi.add_incoming(ir.Constant(I64, 0), entry)
        i_phi.add_incoming(ir.Constant(I64, 0), entry)
        done_num = b.icmp_signed(">=", i_phi, count_i64)
        b.cbranch(done_num, den_loop, num_body)

        b.position_at_end(num_body)
        digit_ptr = b.gep(
            digits_addr,
            [ir.Constant(I32, 0), i_phi],
            inbounds=True,
        )
        digit_byte = b.load(digit_ptr)
        digit = b.zext(digit_byte, I64)
        scaled = b.mul(num_phi, ir.Constant(I64, 60))
        next_num = b.add(scaled, digit)
        next_i = b.add(i_phi, ir.Constant(I64, 1))
        num_phi.add_incoming(next_num, num_body)
        i_phi.add_incoming(next_i, num_body)
        b.branch(num_loop)

        # den = 60^frac_places
        b.position_at_end(den_loop)
        den_phi = b.phi(I64, "den")
        k_phi = b.phi(I64, "k")
        den_phi.add_incoming(ir.Constant(I64, 1), num_loop)
        k_phi.add_incoming(ir.Constant(I64, 0), num_loop)
        done_den = b.icmp_signed(">=", k_phi, frac_places)
        b.cbranch(done_den, apply_sign, den_body)

        b.position_at_end(den_body)
        next_den = b.mul(den_phi, ir.Constant(I64, 60))
        next_k = b.add(k_phi, ir.Constant(I64, 1))
        den_phi.add_incoming(next_den, den_body)
        k_phi.add_incoming(next_k, den_body)
        b.branch(den_loop)

        # Apply sign (any nonzero sign byte means negative).
        b.position_at_end(apply_sign)
        is_neg = b.icmp_signed("!=", sign, ir.Constant(I8, 0))
        signed_num = b.select(is_neg, b.neg(num_phi), num_phi)
        b.branch(do_reduce)

        b.position_at_end(do_reduce)
        result = b.call(self._get_rat_reduce(), [signed_num, den_phi])
        b.ret(result)

        self._sex_to_rat = fn
        return fn

    def _get_rat_reduce(self) -> ir.Function:
        """Emit (once per module) __tuppu_rat_reduce(i64 num, i64 den) -> rat.
        Traps on den == 0. Normalizes so den > 0, then divides both fields by
        gcd(|num|, den) using Euclidean iteration."""
        if self._rat_reduce is not None:
            return self._rat_reduce

        fn = ir.Function(
            self.module,
            ir.FunctionType(RAT, [I64, I64]),
            name="__tuppu_rat_reduce",
        )
        fn.args[0].name = "num"
        fn.args[1].name = "den"
        num_arg, den_arg = fn.args[0], fn.args[1]

        entry      = fn.append_basic_block("entry")
        trap_bb    = fn.append_basic_block("trap")
        normalize  = fn.append_basic_block("normalize")
        gcd_loop   = fn.append_basic_block("gcd.loop")
        gcd_body   = fn.append_basic_block("gcd.body")
        gcd_done   = fn.append_basic_block("gcd.done")

        b = ir.IRBuilder(entry)
        is_den_zero = b.icmp_signed("==", den_arg, ir.Constant(I64, 0))
        b.cbranch(is_den_zero, trap_bb, normalize)

        b.position_at_end(trap_bb)
        b.call(self._get_trap(), [])
        b.unreachable()

        # Normalize: if den < 0, flip both signs. Then gcd on (|num|, den>0).
        b.position_at_end(normalize)
        den_neg = b.icmp_signed("<", den_arg, ir.Constant(I64, 0))
        num_norm = b.select(den_neg, b.neg(num_arg), num_arg)
        den_norm = b.select(den_neg, b.neg(den_arg), den_arg)
        num_neg = b.icmp_signed("<", num_norm, ir.Constant(I64, 0))
        num_abs = b.select(num_neg, b.neg(num_norm), num_norm)
        b.branch(gcd_loop)

        # gcd loop: while b != 0: a, b = b, a % b.  Final a is gcd.
        b.position_at_end(gcd_loop)
        a_phi = b.phi(I64, "gcd.a")
        bb_phi = b.phi(I64, "gcd.b")
        a_phi.add_incoming(num_abs, normalize)
        bb_phi.add_incoming(den_norm, normalize)
        b_is_zero = b.icmp_signed("==", bb_phi, ir.Constant(I64, 0))
        b.cbranch(b_is_zero, gcd_done, gcd_body)

        b.position_at_end(gcd_body)
        rem = b.srem(a_phi, bb_phi)
        a_phi.add_incoming(bb_phi, gcd_body)
        bb_phi.add_incoming(rem, gcd_body)
        b.branch(gcd_loop)

        # Divide through by gcd, build the struct, return.
        b.position_at_end(gcd_done)
        # gcd > 0 because den > 0. If num == 0, gcd = den, and 0/den = 0. Safe.
        result_num = b.sdiv(num_norm, a_phi)
        result_den = b.sdiv(den_norm, a_phi)
        undef = ir.Constant(RAT, ir.Undefined)
        with_n = b.insert_value(undef, result_num, 0)
        final  = b.insert_value(with_n, result_den, 1)
        b.ret(final)

        self._rat_reduce = fn
        return fn


def codegen(program: A.Program) -> ir.Module:
    return Codegen().gen(program)
