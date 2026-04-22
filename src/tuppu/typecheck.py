"""Tuppu type checker.

Runs between the parser and codegen. Walks the AST, infers/verifies the
type of every expression, and rejects ill-typed programs with clear
domain-level errors attached to source positions.

After this pass, codegen may trust its input — existing CodegenError
raises remain as internal assertions but should be unreachable for any
program this pass accepts.

Coercion rules (match existing codegen behavior):
  i*/u* -> i*/u*  : any width, auto-widen (sext/zext) or narrow (trunc)
  bool  -> i*/u*  : zext
  i*/u* -> rat    : promote to rat(x, 1)
  rat   -> i*/u*  : requires explicit `as` cast (narrowing / lossy)
  rat   -> bool, bool -> rat : always requires explicit cast
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Iterable

from . import ast as A
from .errors import CompileError, CompileWarning


def _suggest(name: str, candidates: Iterable[str]) -> str:
    """Return a ' (did you mean 'x'?)' suffix if a single close match
    exists, otherwise an empty string. Tunes cutoff/n to avoid noisy
    suggestions — we only speak up when confident."""
    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    if not matches:
        return ""
    return f" (did you mean {matches[0]!r}?)"


class CheckError(CompileError):
    def __init__(self, message: str, line: int = 0, col: int = 0) -> None:
        if line:
            super().__init__(f"{line}:{col}: {message}")
        else:
            super().__init__(message)
        self.message = message
        self.line = line
        self.col = col


# --- the type lattice -------------------------------------------------------

@dataclass(frozen=True)
class TyInt:
    width: int
    signed: bool = True
    def __str__(self) -> str:
        return f"{'i' if self.signed else 'u'}{self.width}"

@dataclass(frozen=True)
class TyBool:
    def __str__(self) -> str: return "bool"

@dataclass(frozen=True)
class TyRat:
    def __str__(self) -> str: return "rat"

@dataclass(frozen=True)
class TyDish:
    """Babylonian sexagesimal — the raw digit-with-radix form.
    Runtime representation is the same {i64 num, i64 den} struct as rat;
    sex/dish is a compile-time distinction. Arithmetic on dish values
    auto-lowers to rat and emits a warning."""
    def __str__(self) -> str: return "dish"

@dataclass(frozen=True)
class TyPtr:
    """A raw pointer. Held and passed around but not directly
    dereferenced or manipulated from user code — the compiler uses
    these internally for string and FFI byte access."""
    element: "Ty"
    def __str__(self) -> str: return f"*{self.element}"

@dataclass(frozen=True)
class TyUnit:
    def __str__(self) -> str: return "()"

@dataclass(frozen=True)
class TyDiverge:
    """The type of an expression that never returns (e.g. function whose
    body is `yield expr`)."""
    def __str__(self) -> str: return "!"

@dataclass(frozen=True)
class TyTablets:
    size: int
    element: "Ty"
    def __str__(self) -> str: return f"tablets[{self.size}]{self.element}"

@dataclass(frozen=True)
class TyStruct:
    """A user-defined struct. Nominally typed — equal by name only.
    Field layout lives in Checker.struct_fields, keyed by name."""
    name: str
    def __str__(self) -> str: return self.name

@dataclass(frozen=True)
class TyTable:
    element: "Ty"
    def __str__(self) -> str: return f"table[_]{self.element}"

@dataclass(frozen=True)
class TyFn:
    params: tuple
    ret: "Ty"
    def __str__(self) -> str:
        args = ", ".join(str(p) for p in self.params)
        return f"fn({args}) -> {self.ret}"


Ty = object  # structural union; actual nodes listed above

I8  = TyInt(8);  I16 = TyInt(16); I32 = TyInt(32); I64 = TyInt(64)
U8  = TyInt(8, False); U16 = TyInt(16, False); U32 = TyInt(32, False); U64 = TyInt(64, False)
BOOL = TyBool(); RAT = TyRat(); DISH = TyDish(); UNIT = TyUnit(); DIV = TyDiverge()

PRIM_TYPES: dict[str, Ty] = {
    "i8": I8, "i16": I16, "i32": I32, "i64": I64,
    "u8": U8, "u16": U16, "u32": U32, "u64": U64,
    "bool": BOOL, "rat": RAT,
    # `sex` and `dish` are aliases for the same compile-time type.
    "sex": DISH, "dish": DISH,
}

INTRINSIC_NAMES = {"print", "println", "read_int", "rat"}


# --- helpers ----------------------------------------------------------------

def _is_int(t: Ty) -> bool:
    return isinstance(t, TyInt)


def _coerces_to(src: Ty, dst: Ty) -> bool:
    """Would `src` be implicitly coerced to `dst` at a coercion site?"""
    if src == dst:
        return True
    if isinstance(src, TyDiverge):
        return True    # divergent expressions are type-compatible everywhere
    if _is_int(src) and _is_int(dst):
        return True
    if isinstance(src, TyBool) and _is_int(dst):
        return True
    if _is_int(src) and isinstance(dst, TyRat):
        return True
    # dish/sex: silently coerces to rat (no-op at runtime, same struct) and
    # to any integer type (truncating sdiv). Rat and int coerce back to dish.
    if isinstance(src, TyDish) and isinstance(dst, TyRat):
        return True
    if isinstance(src, TyDish) and _is_int(dst):
        return True
    if isinstance(src, TyRat) and isinstance(dst, TyDish):
        return True
    if _is_int(src) and isinstance(dst, TyDish):
        return True
    return False


def _unify_if_arms(a: Ty, b: Ty) -> Ty | None:
    """What type does `if cond { <a> } else { <b> }` produce? None if the
    arms disagree."""
    if isinstance(a, TyDiverge): return b
    if isinstance(b, TyDiverge): return a
    if a == b: return a
    if _is_int(a) and _is_int(b):
        # Pick the wider signed (matches codegen's sext rule).
        sign = a.signed if a.width >= b.width else b.signed  # type: ignore
        return TyInt(max(a.width, b.width), sign)  # type: ignore
    return None


# --- the checker ------------------------------------------------------------

class Checker:
    def __init__(self, prog: A.Program) -> None:
        self.prog = prog
        self.fns: dict[str, TyFn] = {}
        self.tables: dict[str, TyTable] = {}
        self.structs: dict[str, TyStruct] = {}
        self.struct_fields: dict[str, tuple[tuple[str, Ty], ...]] = {}
        self.scopes: list[dict[str, Ty]] = [{}]
        self.current_fn: str = "<top>"
        self.warnings: list[CompileWarning] = []

    def _warn(self, message: str, line: int = 0, col: int = 0) -> None:
        self.warnings.append(CompileWarning(message=message, line=line, col=col))

    def check(self) -> None:
        # Phase 0a: collect struct names so struct-referencing-struct works
        # regardless of source order.
        for d in self.prog.decls:
            if isinstance(d, A.StructDecl):
                self._register_struct_name(d)
        # Phase 0b: resolve each struct's fields. Struct names are visible
        # at this point; primitive types always have been.
        for d in self.prog.decls:
            if isinstance(d, A.StructDecl):
                self._resolve_struct_fields(d)
        # Phase 1: function signatures (parameter and return types can now
        # reference any struct).
        for d in self.prog.decls:
            if isinstance(d, A.FnDecl):
                self._register_fn(d)
        for d in self.prog.decls:
            if isinstance(d, A.TableDecl):
                self._register_table(d)
        for d in self.prog.decls:
            if isinstance(d, A.FnDecl):
                self._check_fn_body(d)

    # --- registration ------------------------------------------------

    def _register_struct_name(self, s: A.StructDecl) -> None:
        if s.name in PRIM_TYPES:
            raise CheckError(
                f"struct {s.name!r}: name shadows a built-in type", s.line, s.col,
            )
        if s.name in self.structs:
            raise CheckError(
                f"duplicate struct {s.name!r}", s.line, s.col,
            )
        self.structs[s.name] = TyStruct(name=s.name)

    def _resolve_struct_fields(self, s: A.StructDecl) -> None:
        seen: set[str] = set()
        resolved: list[tuple[str, Ty]] = []
        for fname, ftype in s.fields:
            if fname in seen:
                raise CheckError(
                    f"struct {s.name!r}: duplicate field {fname!r}",
                    s.line, s.col,
                )
            seen.add(fname)
            resolved.append((
                fname,
                self._resolve_type(ftype, f"field {fname!r} of struct {s.name!r}"),
            ))
        self.struct_fields[s.name] = tuple(resolved)

    def _register_fn(self, fn: A.FnDecl) -> None:
        if fn.name in INTRINSIC_NAMES:
            raise CheckError(
                f"cannot define {fn.name!r}: it is a built-in intrinsic",
                fn.line, fn.col,
            )
        if fn.name in self.fns:
            raise CheckError(
                f"duplicate function {fn.name!r}", fn.line, fn.col,
            )
        params = tuple(
            self._resolve_type(p.type, f"parameter {p.name!r} of {fn.name!r}")
            for p in fn.params
        )
        ret = (
            self._resolve_type(fn.return_type, f"return type of {fn.name!r}")
            if fn.return_type else UNIT
        )
        self.fns[fn.name] = TyFn(params=params, ret=ret)
        if fn.name == "main":
            if ret != I32:
                raise CheckError(
                    f"main must declare -> i32, got -> {ret}", fn.line, fn.col,
                )

    def _register_table(self, t: A.TableDecl) -> None:
        elem = self._resolve_type(t.element_type, f"element type of table {t.name!r}")
        if t.name in self.tables:
            raise CheckError(
                f"duplicate table {t.name!r}", t.line, t.col,
            )
        self.tables[t.name] = TyTable(element=elem)
        if not isinstance(t.generator, A.Ident):
            raise CheckError(
                f"table {t.name!r}: generator must be a function name",
                t.line, t.col,
            )
        gen_name = t.generator.name
        gen = self.fns.get(gen_name)
        if gen is None:
            raise CheckError(
                f"table {t.name!r}: generator {gen_name!r} is not a function",
                t.line, t.col,
            )
        if len(gen.params) != 1 or not _is_int(gen.params[0]):
            raise CheckError(
                f"table {t.name!r}: generator {gen_name!r} must take 1 argument",
                t.line, t.col,
            )
        if not _coerces_to(gen.ret, elem):
            raise CheckError(
                f"table {t.name!r}: generator returns {gen.ret}, "
                f"element type is {elem}",
                t.line, t.col,
            )

    # --- function bodies --------------------------------------------

    def _check_fn_body(self, fn: A.FnDecl) -> None:
        self.current_fn = fn.name
        self.scopes = [{}]
        fn_ty = self.fns[fn.name]
        for param, pty in zip(fn.params, fn_ty.params):
            self.scopes[0][param.name] = pty
        body_ty = self._tc_expr(fn.body)
        expected = fn_ty.ret
        if not _coerces_to(body_ty, expected) and not isinstance(body_ty, TyDiverge):
            raise CheckError(
                f"in fn {fn.name!r}: body produces {body_ty}, expected {expected}",
                fn.line, fn.col,
            )

    # --- type resolution --------------------------------------------

    def _resolve_type(self, t: A.TypeExpr, where: str) -> Ty:
        if isinstance(t, A.TypeName):
            if t.name in PRIM_TYPES:
                return PRIM_TYPES[t.name]
            if t.name in self.structs:
                return self.structs[t.name]
            raise CheckError(
                f"{where}: unknown type {t.name!r}", t.line, t.col,
            )
        if isinstance(t, A.TypeTablets):
            inner = self._resolve_type(t.element, f"{where} element")
            return TyTablets(size=t.size, element=inner)
        if isinstance(t, A.TypeArray):
            raise CheckError(
                f"{where}: array types are not supported in v0.1",
                t.line, t.col,
            )
        if isinstance(t, A.TypePointer):
            inner = self._resolve_type(t.element, f"{where} element")
            return TyPtr(element=inner)
        raise CheckError(
            f"{where}: unsupported type expression",
            getattr(t, "line", 0), getattr(t, "col", 0),
        )

    # --- scope ------------------------------------------------------

    def _push(self) -> None: self.scopes.append({})
    def _pop(self) -> None: self.scopes.pop()

    def _bind(self, name: str, ty: Ty, line: int, col: int) -> None:
        if name in self.scopes[-1]:
            raise CheckError(
                f"redefinition of {name!r} in same scope", line, col,
            )
        self.scopes[-1][name] = ty

    def _lookup(self, name: str, line: int, col: int) -> Ty:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        # Gather everything nameable in this program so the suggestion
        # can point to a function / struct / table if the user typo'd
        # across categories, not just a local.
        candidates: set[str] = set()
        for scope in self.scopes:
            candidates.update(scope)
        candidates.update(self.fns)
        candidates.update(self.structs)
        candidates.update(self.tables)
        raise CheckError(
            f"in fn {self.current_fn!r}: undefined name {name!r}"
            f"{_suggest(name, candidates)}",
            line, col,
        )

    # --- statements -------------------------------------------------

    def _tc_stmt(self, s: A.Stmt) -> None:
        if isinstance(s, A.Binding):       return self._tc_binding(s)
        if isinstance(s, A.Assign):        return self._tc_assign(s)
        if isinstance(s, A.While):         return self._tc_while(s)
        if isinstance(s, A.ForStmt):       return self._tc_for(s)
        if isinstance(s, A.YieldStmt):     return self._tc_yield(s)
        if isinstance(s, A.ReleaseStmt):   return self._tc_release(s)
        if isinstance(s, A.ExprStmt):      self._tc_expr(s.expr); return
        raise CheckError(
            f"unsupported statement: {type(s).__name__}", s.line, s.col,
        )

    def _tc_binding(self, b: A.Binding) -> None:
        declared: Ty | None = None
        if b.type_ann is not None:
            declared = self._resolve_type(b.type_ann, f"type of {b.name!r}")
        if b.init is None:
            assert declared is not None  # parser enforces this
            self._bind(b.name, declared, b.line, b.col)
            return
        init_ty = self._tc_expr(b.init)
        if declared is not None:
            if not _coerces_to(init_ty, declared):
                raise CheckError(
                    f"binding {b.name!r}: initializer has type {init_ty}, "
                    f"expected {declared}",
                    b.line, b.col,
                )
            self._bind(b.name, declared, b.line, b.col)
        else:
            self._bind(b.name, init_ty, b.line, b.col)

    def _tc_assign(self, a: A.Assign) -> None:
        # Type-check the target as an expression: this both validates
        # the chain (field names must exist on their struct types) and
        # yields the expected type of the RHS.
        target_ty = self._tc_expr(a.target)
        value_ty = self._tc_expr(a.value)
        if not _coerces_to(value_ty, target_ty):
            raise CheckError(
                f"assignment target has type {target_ty}, "
                f"value has type {value_ty}",
                a.line, a.col,
            )

    def _tc_for(self, f: A.ForStmt) -> None:
        # Resolve element type before we look at the body so errors about
        # the iterable itself are reported at the for-stmt position.
        element_ty = self._for_element_type(f)
        self._push()
        try:
            self._bind(f.name, element_ty, f.line, f.col)
            self._tc_block(f.body, allow_nonbool_tail=True)
        finally:
            self._pop()

    def _for_element_type(self, f: A.ForStmt) -> Ty:
        """Resolve the per-iteration type for a `for name in iter` loop.
        Tables are recognized by identifier name (like `_tc_index`); other
        iterables come through the normal type of the iter expression."""
        if isinstance(f.iter, A.Ident) and f.iter.name in self.tables:
            return self.tables[f.iter.name].element
        iter_ty = self._tc_expr(f.iter)
        if isinstance(iter_ty, TyTablets):
            return iter_ty.element
        str_ty = self.structs.get("str")
        if str_ty is not None and iter_ty == str_ty:
            return U8
        raise CheckError(
            f"for loop: cannot iterate over {iter_ty}", f.line, f.col,
        )

    def _tc_while(self, w: A.While) -> None:
        cond_ty = self._tc_expr(w.cond)
        if not isinstance(cond_ty, TyBool):
            raise CheckError(
                f"while condition must be a bool, got {cond_ty}",
                w.line, w.col,
            )
        self._tc_block(w.body, allow_nonbool_tail=True)

    def _tc_yield(self, y: A.YieldStmt) -> None:
        expected = self.fns[self.current_fn].ret
        if y.value is None:
            if expected != UNIT:
                raise CheckError(
                    f"bare `yield`: fn {self.current_fn!r} returns {expected}, "
                    f"not ()",
                    y.line, y.col,
                )
            return
        val_ty = self._tc_expr(y.value)
        if not _coerces_to(val_ty, expected):
            raise CheckError(
                f"yield: value has type {val_ty}, fn {self.current_fn!r} "
                f"returns {expected}",
                y.line, y.col,
            )

    def _tc_release(self, r: A.ReleaseStmt) -> None:
        ty = self._lookup(r.name, r.line, r.col)
        if not isinstance(ty, TyTablets):
            raise CheckError(
                f"release: {r.name!r} is {ty}, not a tablets value",
                r.line, r.col,
            )

    # --- expressions ------------------------------------------------

    def _tc_expr(self, e: A.Expr) -> Ty:
        if isinstance(e, A.IntLit):    return I64
        if isinstance(e, A.CharLit):   return U8
        if isinstance(e, A.BoolLit):   return BOOL
        if isinstance(e, A.SexLit):    return DISH
        if isinstance(e, A.StringLit):
            if "str" not in self.structs:
                raise CheckError(
                    "string literal used but `str` type is not in scope "
                    "(stdlib may be missing)",
                    e.line, e.col,
                )
            return self.structs["str"]
        if isinstance(e, A.Ident):
            return self._lookup(e.name, e.line, e.col)
        if isinstance(e, A.Unary):     return self._tc_unary(e)
        if isinstance(e, A.Binary):    return self._tc_binary(e)
        if isinstance(e, A.Call):      return self._tc_call(e)
        if isinstance(e, A.Field):     return self._tc_field(e)
        if isinstance(e, A.Index):     return self._tc_index(e)
        if isinstance(e, A.Cast):      return self._tc_cast(e)
        if isinstance(e, A.StructLit): return self._tc_struct_lit(e)
        if isinstance(e, A.Block):     return self._tc_block(e)
        if isinstance(e, A.IfExpr):    return self._tc_if(e)
        raise CheckError(
            f"unsupported expression: {type(e).__name__}",
            getattr(e, "line", 0), getattr(e, "col", 0),
        )

    def _tc_unary(self, e: A.Unary) -> Ty:
        ty = self._tc_expr(e.operand)
        if e.op == "-":
            if _is_int(ty): return ty
            if isinstance(ty, TyRat): return ty
            # Negating a dish preserves the dish type — it's lossless.
            if isinstance(ty, TyDish): return ty
            raise CheckError(
                f"unary -: requires an integer, rat, or dish, got {ty}", e.line, e.col,
            )
        if e.op == "!":
            if isinstance(ty, TyBool): return BOOL
            raise CheckError(
                f"unary !: requires a bool, got {ty}", e.line, e.col,
            )
        raise CheckError(f"unknown unary operator {e.op!r}", e.line, e.col)

    def _tc_binary(self, e: A.Binary) -> Ty:
        lhs = self._tc_expr(e.lhs)
        rhs = self._tc_expr(e.rhs)
        op = e.op

        if op in ("&&", "||"):
            if not (isinstance(lhs, TyBool) and isinstance(rhs, TyBool)):
                raise CheckError(
                    f"{op} requires bool operands, got {lhs} and {rhs}",
                    e.line, e.col,
                )
            return BOOL

        # Dish arithmetic: lower to rat and emit a warning. Native sex
        # arithmetic (which could in principle exceed i64/i64 precision
        # via radix-shift tricks) isn't implemented yet.
        dish_involved = isinstance(lhs, TyDish) or isinstance(rhs, TyDish)

        if op in ("+", "-", "*", "/", "%"):
            if _is_int(lhs) and _is_int(rhs) and lhs == rhs:
                return lhs
            if isinstance(lhs, TyRat) and isinstance(rhs, TyRat):
                return RAT
            # Native sex arithmetic:
            #   +, -    stay in digit form end-to-end (Phase 2).
            #   *, /    go through rat internally but the result is
            #           reconstructed as sex via __tuppu_rat_to_sex,
            #           which traps at runtime on non-regular results
            #           (Phase 3a/3b). Result type is sex — no warning.
            # Mixed sex + int promotes the int to sex (int→sex is a
            # lossless decomposition into digits, so digit form is
            # preserved end-to-end).
            dish_or_int = lambda t: isinstance(t, TyDish) or _is_int(t)
            if (
                isinstance(lhs, TyDish) and dish_or_int(rhs)
                and op in ("+", "-", "*", "/")
            ) or (
                _is_int(lhs) and isinstance(rhs, TyDish)
                and op in ("+", "-", "*", "/")
            ):
                return DISH
            # Remaining sex-involved ops still lower to rat and warn —
            # `%` is deferred.
            if dish_involved and _coerces_to(lhs, RAT) and _coerces_to(rhs, RAT):
                self._warn(
                    f"sex {op} lowers to rat (native digit-form "
                    f"{op} not yet implemented for this case)",
                    e.line, e.col,
                )
                return RAT
            raise CheckError(
                f"{op} requires matching integer, rat, or dish operands, "
                f"got {lhs} and {rhs}",
                e.line, e.col,
            )

        if op in ("==", "!=", "<", "<=", ">", ">="):
            if lhs == rhs and (_is_int(lhs) or isinstance(lhs, (TyBool, TyRat, TyDish))):
                return BOOL
            # Mixed-width integer comparison: promote both to the wider
            # type (same rule as `if` arm unification).
            if _is_int(lhs) and _is_int(rhs):
                return BOOL
            # Dish vs rat (or dish vs dish that isn't the "equal" case)
            # is cheap to compare via rat, no warning necessary.
            if dish_involved and _coerces_to(lhs, RAT) and _coerces_to(rhs, RAT):
                return BOOL
            raise CheckError(
                f"{op} requires matching comparable operands, got {lhs} and {rhs}",
                e.line, e.col,
            )

        raise CheckError(f"unknown binary operator {op!r}", e.line, e.col)

    def _tc_call(self, e: A.Call) -> Ty:
        # Method call on a tablets variable: t.push(x), t.release(), ...
        if isinstance(e.callee, A.Field) and isinstance(e.callee.target, A.Ident):
            return self._tc_method_call(e)

        if not isinstance(e.callee, A.Ident):
            raise CheckError(
                "only direct function calls are supported", e.line, e.col,
            )
        name = e.callee.name

        if name in ("print", "println"):
            if not e.args:
                raise CheckError(
                    f"{name} takes at least one argument", e.line, e.col,
                )
            str_ty = self.structs.get("str")
            for arg in e.args:
                at = self._tc_expr(arg)
                if not (
                    _is_int(at)
                    or isinstance(at, (TyBool, TyRat, TyDish))
                    or (str_ty is not None and at == str_ty)
                ):
                    raise CheckError(
                        f"{name}: unsupported argument type {at}",
                        e.line, e.col,
                    )
            return UNIT

        if name == "read_int":
            if e.args:
                raise CheckError(
                    "read_int takes no arguments", e.line, e.col,
                )
            return I64

        if name == "rat":
            if len(e.args) != 2:
                raise CheckError(
                    "rat() takes exactly two arguments (num, den)",
                    e.line, e.col,
                )
            for i, a in enumerate(e.args):
                at = self._tc_expr(a)
                if not _is_int(at):
                    raise CheckError(
                        f"rat(): argument {i} must be integer, got {at}",
                        e.line, e.col,
                    )
            return RAT

        fn = self.fns.get(name)
        if fn is None:
            raise CheckError(
                f"in fn {self.current_fn!r}: unknown function {name!r}"
                f"{_suggest(name, self.fns)}",
                e.line, e.col,
            )
        if len(e.args) != len(fn.params):
            raise CheckError(
                f"{name} expects {len(fn.params)} args, got {len(e.args)}",
                e.line, e.col,
            )
        for i, (arg, pty) in enumerate(zip(e.args, fn.params)):
            at = self._tc_expr(arg)
            if not _coerces_to(at, pty):
                raise CheckError(
                    f"call to {name!r}: arg {i} has type {at}, expected {pty}",
                    e.line, e.col,
                )
        return fn.ret

    def _tc_method_call(self, e: A.Call) -> Ty:
        assert isinstance(e.callee, A.Field) and isinstance(e.callee.target, A.Ident)
        recv_name = e.callee.target.name
        method = e.callee.name
        recv_ty = self._lookup(recv_name, e.line, e.col)

        if isinstance(recv_ty, TyTablets):
            if method == "push":
                if len(e.args) != 1:
                    raise CheckError(
                        "tablets.push takes one argument", e.line, e.col,
                    )
                at = self._tc_expr(e.args[0])
                if not _coerces_to(at, recv_ty.element):
                    raise CheckError(
                        f"tablets.push: value has type {at}, "
                        f"element type is {recv_ty.element}",
                        e.line, e.col,
                    )
                return UNIT
            raise CheckError(
                f"tablets has no method {method!r}", e.line, e.col,
            )

        raise CheckError(
            f"method call: {recv_name!r} is {recv_ty}, not a tablets",
            e.line, e.col,
        )

    def _tc_field(self, e: A.Field) -> Ty:
        target_ty = self._tc_expr(e.target)
        # Dish shares fields (.num, .den) with rat at runtime.
        if isinstance(target_ty, (TyRat, TyDish)):
            if e.name in ("num", "den"):
                return I64
            raise CheckError(
                f"{target_ty} has no field {e.name!r}; only num and den",
                e.line, e.col,
            )
        if isinstance(target_ty, TyTablets):
            if e.name == "len":
                return I64
            raise CheckError(
                f"tablets has no field {e.name!r}; only len",
                e.line, e.col,
            )
        if isinstance(target_ty, TyStruct):
            fields = self.struct_fields[target_ty.name]
            for fname, fty in fields:
                if fname == e.name:
                    return fty
            field_names = [n for n, _ in fields]
            raise CheckError(
                f"struct {target_ty.name!r} has no field {e.name!r}"
                f"{_suggest(e.name, field_names)}",
                e.line, e.col,
            )
        raise CheckError(
            f"field access on type {target_ty} is not supported",
            e.line, e.col,
        )

    def _tc_struct_lit(self, e: A.StructLit) -> Ty:
        if e.name not in self.structs:
            raise CheckError(
                f"unknown struct {e.name!r}"
                f"{_suggest(e.name, self.structs)}",
                e.line, e.col,
            )
        declared = self.struct_fields[e.name]
        declared_map = {n: t for n, t in declared}
        seen: set[str] = set()
        for fname, fexpr in e.fields:
            if fname not in declared_map:
                raise CheckError(
                    f"struct {e.name!r}: unknown field {fname!r}"
                    f"{_suggest(fname, declared_map)}",
                    e.line, e.col,
                )
            if fname in seen:
                raise CheckError(
                    f"struct {e.name!r}: duplicate field {fname!r} in literal",
                    e.line, e.col,
                )
            seen.add(fname)
            val_ty = self._tc_expr(fexpr)
            if not _coerces_to(val_ty, declared_map[fname]):
                raise CheckError(
                    f"struct {e.name!r} field {fname!r}: got {val_ty}, "
                    f"expected {declared_map[fname]}",
                    e.line, e.col,
                )
        missing = [n for n, _ in declared if n not in seen]
        if missing:
            raise CheckError(
                f"struct {e.name!r}: missing field(s) {', '.join(repr(n) for n in missing)}",
                e.line, e.col,
            )
        return self.structs[e.name]

    def _tc_index(self, e: A.Index) -> Ty:
        if isinstance(e.target, A.Ident) and e.target.name in self.tables:
            tbl = self.tables[e.target.name]
            idx_ty = self._tc_expr(e.index)
            if not _is_int(idx_ty):
                raise CheckError(
                    f"table index must be integer, got {idx_ty}",
                    e.line, e.col,
                )
            return tbl.element
        target_ty = self._tc_expr(e.target)
        if isinstance(target_ty, TyTablets):
            idx_ty = self._tc_expr(e.index)
            if not _is_int(idx_ty):
                raise CheckError(
                    f"tablets index must be integer, got {idx_ty}",
                    e.line, e.col,
                )
            return target_ty.element
        # str indexing yields u8 — the compiler inserts a bounds check
        # against s.len and reads through s.ptr.
        str_ty = self.structs.get("str")
        if str_ty is not None and target_ty == str_ty:
            idx_ty = self._tc_expr(e.index)
            if not _is_int(idx_ty):
                raise CheckError(
                    f"str index must be integer, got {idx_ty}",
                    e.line, e.col,
                )
            return U8
        raise CheckError(
            f"cannot index into {target_ty}", e.line, e.col,
        )

    def _tc_cast(self, e: A.Cast) -> Ty:
        src = self._tc_expr(e.value)
        # `f32`/`f64` aren't yet in our type system; intercept here to give
        # the user a clear "not yet supported" message at the cast site.
        if isinstance(e.type, A.TypeName) and e.type.name in ("f32", "f64"):
            raise CheckError(
                f"cast target {e.type.name!r} is not yet supported",
                e.line, e.col,
            )
        dst = self._resolve_type(e.type, "cast target")

        if src == dst: return dst
        if _is_int(src) and _is_int(dst):                    return dst
        if isinstance(src, TyBool) and _is_int(dst):         return dst
        if _is_int(src) and isinstance(dst, TyRat):          return dst
        if isinstance(src, TyRat) and _is_int(dst):          return dst
        # Dish casts: sex ↔ rat (no-op at runtime), sex → int (truncate).
        if isinstance(src, TyDish) and isinstance(dst, TyRat):   return dst
        if isinstance(src, TyRat) and isinstance(dst, TyDish):   return dst
        if isinstance(src, TyDish) and _is_int(dst):             return dst
        if _is_int(src) and isinstance(dst, TyDish):             return dst
        raise CheckError(
            f"cannot cast {src} to {dst}", e.line, e.col,
        )

    def _tc_block(self, b: A.Block, *, allow_nonbool_tail: bool = False) -> Ty:
        self._push()
        try:
            for stmt in b.stmts:
                self._tc_stmt(stmt)
            if b.tail is None:
                return UNIT
            return self._tc_expr(b.tail)
        finally:
            self._pop()

    def _tc_if(self, e: A.IfExpr) -> Ty:
        cond_ty = self._tc_expr(e.cond)
        if not isinstance(cond_ty, TyBool):
            raise CheckError(
                f"if condition must be bool, got {cond_ty}", e.line, e.col,
            )
        then_ty = self._tc_block(e.then)
        if e.else_ is None:
            return UNIT
        if isinstance(e.else_, A.IfExpr):
            else_ty = self._tc_if(e.else_)
        else:
            else_ty = self._tc_block(e.else_)
        unified = _unify_if_arms(then_ty, else_ty)
        if unified is None:
            raise CheckError(
                f"if arms have different types: {then_ty} vs {else_ty}",
                e.line, e.col,
            )
        return unified


def check(program: A.Program) -> list[CompileWarning]:
    """Run type-checking. Raises CheckError on any problem. Returns any
    non-fatal warnings collected during the pass (may be empty)."""
    checker = Checker(program)
    checker.check()
    return checker.warnings
