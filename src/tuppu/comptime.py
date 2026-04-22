"""Compile-time evaluator for Tuppu.

Used by codegen to populate `table` declarations at build time. Implements
a tree-walking interpreter over the AST for the subset of Tuppu listed in
SPEC.md §13: arithmetic + comparison on primitive types and rat, step/mut
bindings, if/else, while, calls to other comptime-evaluable functions,
indexing into already-declared tables. I/O intrinsics are rejected.

Comptime values are plain Python values for simplicity:
  integers -> int    |    bools -> bool    |    rats -> (num, den) tuple
  strings  -> bytes  |    tables -> list of any of the above
"""
from __future__ import annotations

from math import gcd

from . import ast as A


class ComptimeError(Exception):
    pass


class _YieldSignal(Exception):
    """Internal: propagates a `yield` back to the enclosing function call."""
    def __init__(self, value):
        self.value = value


def _rat_reduce(num: int, den: int) -> tuple[int, int]:
    if den == 0:
        raise ComptimeError("comptime: division by zero")
    if den < 0:
        num, den = -num, -den
    g = gcd(abs(num), den)
    if g == 0:
        g = 1
    return num // g, den // g


def _trunc_div(a: int, b: int) -> int:
    """Signed integer division that truncates toward zero (like LLVM sdiv)."""
    if b == 0:
        raise ComptimeError("comptime: division by zero")
    q, r = divmod(a, b)
    if r and (a < 0) != (b < 0):
        q += 1
    return q


class _Env:
    """Stack of scopes for lexical lookup and assignment."""
    def __init__(self) -> None:
        self.scopes: list[dict] = [{}]

    def push(self) -> None: self.scopes.append({})
    def pop(self) -> None: self.scopes.pop()

    def define(self, name: str, value) -> None:
        self.scopes[-1][name] = value

    def lookup(self, name: str):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise ComptimeError(f"comptime: undefined name {name!r}")

    def assign(self, name: str, value) -> None:
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = value
                return
        raise ComptimeError(f"comptime: cannot assign to undefined {name!r}")


class Comptime:
    def __init__(self, program: A.Program) -> None:
        self.fns: dict[str, A.FnDecl] = {
            d.name: d for d in program.decls if isinstance(d, A.FnDecl)
        }
        self.tables: dict[str, list] = {}

    # --- public entry points ------------------------------------------

    def eval_table(self, table: A.TableDecl) -> list:
        env = _Env()
        lo = self._expect_int(self.eval_expr(table.lo, env), "table range lo")
        hi = self._expect_int(self.eval_expr(table.hi, env), "table range hi")
        if hi < lo:
            raise ComptimeError(
                f"table {table.name!r}: empty or inverted range [{lo}..{hi})"
            )

        if not isinstance(table.generator, A.Ident):
            raise ComptimeError(
                f"table {table.name!r}: generator must be a function name"
            )
        fn_name = table.generator.name
        fn = self.fns.get(fn_name)
        if fn is None:
            raise ComptimeError(
                f"table {table.name!r}: generator {fn_name!r} is not a function"
            )
        if len(fn.params) != 1:
            raise ComptimeError(
                f"table {table.name!r}: generator {fn_name!r} must take 1 argument"
            )

        values = [self._call_fn(fn, [i]) for i in range(lo, hi)]
        self.tables[table.name] = values
        return values

    def eval_constant_expr(self, e: A.Expr) -> object:
        return self.eval_expr(e, _Env())

    # --- dispatch -----------------------------------------------------

    def eval_expr(self, e: A.Expr, env: _Env):
        if isinstance(e, A.IntLit):
            return e.value
        if isinstance(e, A.BoolLit):
            return e.value
        if isinstance(e, A.SexLit):
            return (e.num, e.den)
        if isinstance(e, A.CharLit):
            return e.value
        if isinstance(e, A.StructLit):
            return {fname: self.eval_expr(fexpr, env) for fname, fexpr in e.fields}
        if isinstance(e, A.StringLit):
            return e.value
        if isinstance(e, A.Ident):
            return env.lookup(e.name)
        if isinstance(e, A.Unary):
            return self._eval_unary(e.op, self.eval_expr(e.operand, env))
        if isinstance(e, A.Binary):
            return self._eval_binary(e, env)
        if isinstance(e, A.Call):
            return self._eval_call(e, env)
        if isinstance(e, A.Field):
            return self._eval_field(e, env)
        if isinstance(e, A.Block):
            return self.eval_block(e, env)
        if isinstance(e, A.IfExpr):
            return self._eval_if(e, env)
        if isinstance(e, A.Cast):
            return self._eval_cast(self.eval_expr(e.value, env), e.type)
        if isinstance(e, A.Index):
            return self._eval_index(e, env)
        raise ComptimeError(f"unsupported expr at comptime: {type(e).__name__}")

    def eval_block(self, b: A.Block, env: _Env):
        env.push()
        try:
            for stmt in b.stmts:
                self._eval_stmt(stmt, env)
            if b.tail is None:
                return None
            return self.eval_expr(b.tail, env)
        finally:
            env.pop()

    # --- details ------------------------------------------------------

    def _call_fn(self, fn: A.FnDecl, args: list):
        env = _Env()
        for p, a in zip(fn.params, args):
            env.define(p.name, a)
        try:
            return self.eval_expr(fn.body, env)
        except _YieldSignal as y:
            return y.value

    def _eval_unary(self, op: str, v):
        if op == "-":
            if isinstance(v, tuple):              # rat negation
                return (-v[0], v[1])
            if isinstance(v, bool):
                raise ComptimeError("comptime: cannot negate a bool with -")
            return -v
        if op == "!":
            if not isinstance(v, bool):
                raise ComptimeError("comptime: ! requires bool")
            return not v
        raise ComptimeError(f"comptime: unknown unary op {op!r}")

    def _eval_binary(self, e: A.Binary, env: _Env):
        l = self.eval_expr(e.lhs, env)
        r = self.eval_expr(e.rhs, env)
        op = e.op

        if isinstance(l, tuple) and isinstance(r, tuple):
            return self._rat_binop(op, l, r)

        if op == "+":  return l + r
        if op == "-":  return l - r
        if op == "*":  return l * r
        if op == "/":  return _trunc_div(l, r)
        if op == "%":
            q = _trunc_div(l, r)
            return l - q * r
        if op == "==": return l == r
        if op == "!=": return l != r
        if op == "<":  return l < r
        if op == "<=": return l <= r
        if op == ">":  return l > r
        if op == ">=": return l >= r
        if op == "&&":
            if not isinstance(l, bool) or not isinstance(r, bool):
                raise ComptimeError("comptime: && requires bool operands")
            return l and r
        if op == "||":
            if not isinstance(l, bool) or not isinstance(r, bool):
                raise ComptimeError("comptime: || requires bool operands")
            return l or r
        raise ComptimeError(f"comptime: unknown binop {op!r}")

    def _rat_binop(self, op: str, a: tuple, b: tuple):
        an, ad = a
        bn, bd = b
        if op == "+":  return _rat_reduce(an * bd + bn * ad, ad * bd)
        if op == "-":  return _rat_reduce(an * bd - bn * ad, ad * bd)
        if op == "*":  return _rat_reduce(an * bn, ad * bd)
        if op == "/":  return _rat_reduce(an * bd, ad * bn)
        if op in ("==", "!="):
            eq = (an == bn) and (ad == bd)
            return eq if op == "==" else not eq
        # comparison: a/p op b/q  <=>  a*q op b*p  (both dens positive)
        left = an * bd
        right = bn * ad
        if op == "<":  return left < right
        if op == "<=": return left <= right
        if op == ">":  return left > right
        if op == ">=": return left >= right
        raise ComptimeError(f"comptime: unknown rat op {op!r}")

    def _eval_call(self, e: A.Call, env: _Env):
        if not isinstance(e.callee, A.Ident):
            raise ComptimeError("comptime: only direct function calls")
        name = e.callee.name
        if name == "rat":
            args = [self.eval_expr(a, env) for a in e.args]
            if len(args) != 2:
                raise ComptimeError("comptime: rat() takes 2 arguments")
            return _rat_reduce(args[0], args[1])
        if name in ("print", "println", "read_int"):
            raise ComptimeError(
                f"comptime: I/O intrinsic {name!r} cannot be called at build time"
            )
        fn = self.fns.get(name)
        if fn is None:
            raise ComptimeError(f"comptime: unknown function {name!r}")
        if len(e.args) != len(fn.params):
            raise ComptimeError(f"comptime: {name!r} wrong arg count")
        args = [self.eval_expr(a, env) for a in e.args]
        return self._call_fn(fn, args)

    def _eval_field(self, e: A.Field, env: _Env):
        target = self.eval_expr(e.target, env)
        if isinstance(target, tuple) and len(target) == 2:
            if e.name == "num": return target[0]
            if e.name == "den": return target[1]
            raise ComptimeError(f"comptime: rat has no field {e.name!r}")
        if isinstance(target, dict):
            if e.name in target:
                return target[e.name]
            raise ComptimeError(
                f"comptime: struct has no field {e.name!r}"
            )
        raise ComptimeError(
            f"comptime: field {e.name!r} on unsupported value"
        )

    def _eval_if(self, e: A.IfExpr, env: _Env):
        cond = self.eval_expr(e.cond, env)
        if not isinstance(cond, bool):
            raise ComptimeError("comptime: if condition must be bool")
        if cond:
            return self.eval_block(e.then, env)
        if e.else_ is None:
            return None
        if isinstance(e.else_, A.IfExpr):
            return self._eval_if(e.else_, env)
        return self.eval_block(e.else_, env)

    def _eval_cast(self, v, type_expr: A.TypeExpr):
        if not isinstance(type_expr, A.TypeName):
            raise ComptimeError(f"comptime: unsupported cast target")
        name = type_expr.name
        if name in ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"):
            if isinstance(v, bool):
                return 1 if v else 0
            if isinstance(v, tuple):
                return _trunc_div(v[0], v[1])
            if isinstance(v, int):
                return v
        if name == "rat":
            if isinstance(v, bool):
                return (1, 1) if v else (0, 1)
            if isinstance(v, int):
                return (v, 1)
            if isinstance(v, tuple):
                return v
        if name == "bool":
            if isinstance(v, bool):
                return v
        raise ComptimeError(f"comptime: cannot cast {v!r} to {name}")

    def _eval_index(self, e: A.Index, env: _Env):
        if not (isinstance(e.target, A.Ident) and e.target.name in self.tables):
            raise ComptimeError(
                "comptime: indexing is only allowed on already-declared tables"
            )
        idx = self.eval_expr(e.index, env)
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise ComptimeError("comptime: table index must be integer")
        tbl = self.tables[e.target.name]
        if not 0 <= idx < len(tbl):
            raise ComptimeError(
                f"comptime: table {e.target.name!r} index {idx} out of range"
            )
        return tbl[idx]

    # --- statements ---------------------------------------------------

    def _eval_stmt(self, s: A.Stmt, env: _Env):
        if isinstance(s, A.Binding):
            env.define(s.name, self.eval_expr(s.init, env))
            return
        if isinstance(s, A.Assign):
            if not isinstance(s.target, A.Ident):
                raise ComptimeError(
                    "comptime: field assignment not supported"
                )
            env.assign(s.target.name, self.eval_expr(s.value, env))
            return
        if isinstance(s, A.While):
            limit, count = 10_000_000, 0
            while self.eval_expr(s.cond, env):
                self.eval_block(s.body, env)
                count += 1
                if count > limit:
                    raise ComptimeError(
                        f"comptime: while loop exceeded {limit} iterations"
                    )
            return
        if isinstance(s, A.YieldStmt):
            v = self.eval_expr(s.value, env) if s.value else None
            raise _YieldSignal(v)
        if isinstance(s, A.ExprStmt):
            self.eval_expr(s.expr, env)
            return
        raise ComptimeError(f"comptime: unsupported stmt {type(s).__name__}")

    # --- helpers ------------------------------------------------------

    @staticmethod
    def _expect_int(v, where: str) -> int:
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        raise ComptimeError(f"{where} must be an integer, got {v!r}")
