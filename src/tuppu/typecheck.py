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
from dataclasses import dataclass, field
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


class _UnifyError(Exception):
    """Internal: structural unification failure. Wrapped into a
    CheckError by the caller so the message includes call-site
    context and source position."""
    def __init__(self, detail: str = "") -> None:
        super().__init__(detail)
        self.detail = detail


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
class TyHandle:
    """`wedge T` — a handle into some `tablets[N]T` storage. At
    runtime a pointer to T; at the type level it's distinct from
    `*T` so we can restrict how it's obtained (only through
    `tablets.push`, not `&x`) and what operations it supports
    (auto-deref field access, equality, `lost` comparison).

    The name comes from cuneiform being "wedge-writing" — a wedge
    is the atom of a Mesopotamian mark, a single pointer to
    something larger."""
    element: "Ty"
    def __str__(self) -> str: return f"wedge {self.element}"

@dataclass(frozen=True)
class TyLost:
    """The bare `lost` literal's type — coerces to any `wedge T`.
    Never appears in a type annotation; only as the transient type
    of a `LostLit` expression that hasn't been placed yet."""
    def __str__(self) -> str: return "lost"

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
class TyBuffer:
    """`buffer[N]T` — a fixed-size, stack-allocated buffer. `.len` is
    the compile-time constant `N`. Zero-init on declaration, bounds
    checked on index, rejected in return types and struct fields."""
    size: int
    element: "Ty"
    def __str__(self) -> str: return f"buffer[{self.size}]{self.element}"

@dataclass(frozen=True)
class TyStruct:
    """A user-defined tablet (product) type. Nominally typed — equal
    by name only, with an optional tuple of instantiated type args
    for generic tablets. `Node<i64>` and `Node<str>` are different
    TyStructs; `Node` (non-generic) has args=()."""
    name: str
    args: tuple = ()    # tuple of Ty, type arguments for generic tablets
    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}<{', '.join(str(a) for a in self.args)}>"

@dataclass(frozen=True)
class TySeal:
    """A user-defined seal (sum) type. Like TyStruct it's nominal and
    carries an optional tuple of type args for generic seals — so
    `Option<i64>` and `Option<str>` are distinct TySeals."""
    name: str
    args: tuple = ()
    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}<{', '.join(str(a) for a in self.args)}>"

@dataclass(frozen=True)
class TyVar:
    """A type variable — appears during generic-fn body checking and
    during unification at a call site. Resolved away by the end of
    typechecking a concrete call."""
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
    # A variadic fn's last param is a `tablets[...]T`; the checker
    # collects trailing call-site args into a synthetic TabletsLit
    # wrapped around the element type. Non-variadic fns set this False.
    is_variadic: bool = False
    # Per-param mut flag — parallels `params`. Used by the freeze-
    # while-borrow analysis: a `mut`-annotated param is a mut-reach on
    # its argument root, invalidating any live borrow rooted there.
    # `compare=False` keeps mut-ness out of TyFn equality so a fn
    # literal (no mut info) compares equal to its registered decl.
    param_muts: tuple = field(default=(), compare=False)
    def __str__(self) -> str:
        args_list = []
        for i, p in enumerate(self.params):
            if self.is_variadic and i == len(self.params) - 1 and isinstance(p, TyTablets):
                args_list.append(f"tablets[...]{p.element}")
            else:
                args_list.append(str(p))
        return f"fn({', '.join(args_list)}) -> {self.ret}"


Ty = object  # structural union; actual nodes listed above

I8  = TyInt(8);  I16 = TyInt(16); I32 = TyInt(32); I64 = TyInt(64)
U8  = TyInt(8, False); U16 = TyInt(16, False); U32 = TyInt(32, False); U64 = TyInt(64, False)
BOOL = TyBool(); RAT = TyRat(); DISH = TyDish(); UNIT = TyUnit(); DIV = TyDiverge()
LOST = TyLost()

PRIM_TYPES: dict[str, Ty] = {
    "i8": I8, "i16": I16, "i32": I32, "i64": I64,
    "u8": U8, "u16": U16, "u32": U32, "u64": U64,
    "bool": BOOL, "rat": RAT,
    # `sex` and `dish` are aliases for the same compile-time type.
    "sex": DISH, "dish": DISH,
}

INTRINSIC_NAMES = {
    "print", "println", "read_int", "rat",
    "str_slice",
    "int_to_str", "sex_to_str",
    "bytes_to_str", "buffer_to_str",
}

# The fixed set of operator-overload op names users may declare with
# `gloss <op>(...)`. Each maps to the operator symbol it implements,
# an arity ("bin" / "un"), and whether the return type is fixed.
#
# `eq` produces `==`; the typechecker also derives `!=` by negating
# the result, so users don't (and shouldn't) declare `gloss ne`.
# Ordering ops are separate for v1 — no `cmp`-returning-`Ordering`
# convenience layer. Might add one later.
GLOSS_OPS: dict[str, tuple[str, str, "str | None"]] = {
    # name      -> (op_symbol, arity, fixed_return_type_name_or_None)
    "add":      ("+",  "bin", None),
    "sub":      ("-",  "bin", None),
    "mul":      ("*",  "bin", None),
    "div":      ("/",  "bin", None),
    "mod":      ("%",  "bin", None),
    # Comparisons must return bool — the control-flow machinery
    # downstream of `if` / `while` depends on it.
    "eq":       ("==", "bin", "bool"),
    "lt":       ("<",  "bin", "bool"),
    "le":       ("<=", "bin", "bool"),
    "gt":       (">",  "bin", "bool"),
    "ge":       (">=", "bin", "bool"),
    # Unary ops are free to return any type — e.g. `!flag` on a
    # user Flag might flip it and return another Flag, or `-v` on
    # a Vector returns a Vector.
    "neg":      ("-",  "un",  None),
    "not":      ("!",  "un",  None),
}

# Canonical tablets chunk size used when synthesising literals for
# variadic call arguments and when resolving `tablets[...]T` param
# markers. Sixteen is big enough to hold most variadic calls in one
# chunk while staying cheap on the "tiny call" path.
VARIADIC_CHUNK_SIZE = 16


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
    # Tablet handles: `lost` coerces to any `tablet T`. Two handle
    # types are compatible only when their element types match
    # (nominal, no subtyping — a tablet Node isn't a tablet Tree).
    if isinstance(src, TyLost) and isinstance(dst, TyHandle):
        return True
    if isinstance(src, TyHandle) and isinstance(dst, TyHandle):
        return src.element == dst.element
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
        # Seals (sum types) — registered alongside structs in phase 0.
        self.seals: dict[str, TySeal] = {}
        self.seal_type_params: dict[str, tuple[str, ...]] = {}
        # seal name → tuple of (variant_name, tuple of field Ty).
        # Order is source order so codegen can assign stable tag indices.
        self.seal_variants: dict[str, tuple[tuple[str, tuple[Ty, ...]], ...]] = {}
        # variant_name → (seal_name, variant_index, declared_field_tys).
        # Variant names are globally unique for v0.1 — ambiguity would
        # require qualified syntax we haven't designed yet.
        self.variant_lookup: dict[str, tuple[str, int, tuple[Ty, ...]]] = {}
        # Generics: per-tablet type-parameter names, in declaration order.
        self.struct_type_params: dict[str, tuple[str, ...]] = {}
        # Generics: per-fn type-parameter names.
        self.fn_type_params: dict[str, tuple[str, ...]] = {}
        # Names of colophon-declared externs — codegen emits extern
        # declarations for these rather than bodies.
        self.colophons: set[str] = set()
        # Type variables visible while resolving a specific generic body.
        # Maps the source-level name ("T") to a TyVar sentinel.
        self._active_type_vars: dict[str, "TyVar"] = {}
        # Per-AST-node monomorphization sidebands, keyed by id(node).
        # Filled by the checker at calls / literals; codegen consults
        # these to know which specialization to emit.
        self.mono_call_args: dict[int, tuple] = {}   # Call node → tuple of Ty
        self.mono_struct_args: dict[int, tuple] = {} # StructLit → tuple of Ty
        # Variadic calls: Call node id → synthesized TabletsLit holding
        # the collected trailing arguments. Codegen consults this to
        # emit the literal once for the last param slot.
        self.variadic_lit_for_call: dict[int, "A.TabletsLit"] = {}
        # Operator-overload dispatch table. Keys are (op, lhs_ty, rhs_ty)
        # for binary ops or (op, operand_ty, None) for unary, where
        # `op` is one of the GLOSS_OPS names. Values are the mangled
        # internal fn name — codegen resolves through self.functions
        # like any other fn call. Populated by `_register_gloss`.
        self.gloss_dispatch: dict[tuple[str, Ty, "Ty | None"], str] = {}
        # Per-expression sideband: Binary / Unary node id → mangled
        # gloss fn name. Codegen consults this to emit a call through
        # the user-defined overload instead of the built-in op lowering.
        self.gloss_call_for_node: dict[int, str] = {}
        # IfExpr AST node ids appearing in statement position — i.e.
        # as a bare `ExprStmt` inside a block or a while body, where
        # the value is provably discarded. Populated by `_tc_stmt`,
        # consulted by `_tc_if` to relax the arms-must-unify rule, and
        # by codegen to skip phi construction when arm types differ.
        # Elif chains propagate: if an outer if is in stmt position,
        # every nested `else if` inherits the flag.
        self.stmt_if_nodes: set[int] = set()
        # Variant construction sidebands — both Call (`Some(x)`) and
        # bare Ident (`None`) forms. Keyed by id(AST node).
        self.mono_variant_args: dict[int, tuple] = {}
        self.variant_of_node: dict[int, tuple[str, str, int]] = {}
        # ^ id(node) → (seal_name, variant_name, variant_index)
        self.scopes: list[dict[str, Ty]] = [{}]
        self.current_fn: str = "<top>"
        self.warnings: list[CompileWarning] = []
        # Escape tracking: name → True if this binding holds a handle
        # (or a tablets) whose storage is rooted in a mut tablets
        # declared inside the current function. Returning such a value
        # is a use-after-free and gets rejected.
        self._local_tablets: set[str] = set()
        self._tainted: dict[str, bool] = {}
        # Freeze-while-borrow state. Per-scope, `_borrow_sources` maps
        # a borrow binding's name → its source root (the outermost
        # owning Ident whose mutation would invalidate the borrow).
        # `_invalidated` records borrows whose root has been mut-
        # reached since their binding — any read of an invalidated
        # borrow is rejected with a suggestion to `copy`.
        self._borrow_sources: list[dict[str, str]] = [{}]
        self._invalidated: set[str] = set()

    def _warn(self, message: str, line: int = 0, col: int = 0) -> None:
        self.warnings.append(CompileWarning(message=message, line=line, col=col))

    def check(self) -> None:
        # Phase 0a: collect struct + seal names so type bodies can refer
        # to each other and to user types regardless of source order.
        for d in self.prog.decls:
            if isinstance(d, A.StructDecl):
                self._register_struct_name(d)
            elif isinstance(d, A.SealDecl):
                self._register_seal_name(d)
        # Phase 0b: resolve fields (structs) and variants (seals) now
        # that all user type names are in scope.
        for d in self.prog.decls:
            if isinstance(d, A.StructDecl):
                self._resolve_struct_fields(d)
            elif isinstance(d, A.SealDecl):
                self._resolve_seal_variants(d)
        # Phase 1: function signatures (parameter and return types can now
        # reference any struct). Colophons declare externs and join the
        # same fn table so call sites resolve uniformly. Gloss decls
        # register in both the fn table (under a mangled name) and the
        # operator dispatch table.
        for d in self.prog.decls:
            if isinstance(d, A.FnDecl):
                self._register_fn(d)
            elif isinstance(d, A.ColophonDecl):
                self._register_colophon(d)
            elif isinstance(d, A.GlossDecl):
                self._register_gloss(d)
        for d in self.prog.decls:
            if isinstance(d, A.TableDecl):
                self._register_table(d)
        for d in self.prog.decls:
            if isinstance(d, A.FnDecl):
                self._check_fn_body(d)
            elif isinstance(d, A.GlossDecl):
                self._check_gloss_body(d)

    # --- registration ------------------------------------------------

    def _register_struct_name(self, s: A.StructDecl) -> None:
        if s.name in PRIM_TYPES:
            raise CheckError(
                f"tablet {s.name!r}: name shadows a built-in type", s.line, s.col,
            )
        if s.name in self.structs:
            raise CheckError(
                f"duplicate tablet {s.name!r}", s.line, s.col,
            )
        self.structs[s.name] = TyStruct(name=s.name)
        self.struct_type_params[s.name] = tuple(s.type_params)

    def _register_seal_name(self, s: A.SealDecl) -> None:
        if s.name in PRIM_TYPES:
            raise CheckError(
                f"seal {s.name!r}: name shadows a built-in type", s.line, s.col,
            )
        if s.name in self.structs:
            raise CheckError(
                f"seal {s.name!r}: name collides with an existing tablet",
                s.line, s.col,
            )
        if s.name in self.seals:
            raise CheckError(
                f"duplicate seal {s.name!r}", s.line, s.col,
            )
        self.seals[s.name] = TySeal(name=s.name)
        self.seal_type_params[s.name] = tuple(s.type_params)

    def _resolve_seal_variants(self, s: A.SealDecl) -> None:
        seen_names: set[str] = set()
        resolved_variants: list[tuple[str, tuple[Ty, ...]]] = []
        saved = self._active_type_vars
        self._active_type_vars = {name: TyVar(name) for name in s.type_params}
        try:
            for idx, v in enumerate(s.variants):
                if v.name in seen_names:
                    raise CheckError(
                        f"seal {s.name!r}: duplicate variant {v.name!r}",
                        v.line, v.col,
                    )
                seen_names.add(v.name)
                if v.name in self.variant_lookup:
                    prev_seal = self.variant_lookup[v.name][0]
                    raise CheckError(
                        f"variant {v.name!r} is already declared in seal "
                        f"{prev_seal!r}; variant names must be globally "
                        f"unique in v0.1",
                        v.line, v.col,
                    )
                field_tys = tuple(
                    self._resolve_type(
                        ft, f"field of variant {v.name!r} of seal {s.name!r}",
                    )
                    for ft in v.fields
                )
                resolved_variants.append((v.name, field_tys))
                self.variant_lookup[v.name] = (s.name, idx, field_tys)
        finally:
            self._active_type_vars = saved
        self.seal_variants[s.name] = tuple(resolved_variants)

    def _resolve_struct_fields(self, s: A.StructDecl) -> None:
        seen: set[str] = set()
        resolved: list[tuple[str, Ty]] = []
        # Inside a generic tablet's body, its type parameters are in
        # scope as type variables. `Node<T> { next: wedge Node<T> }`
        # resolves the field type with T bound to TyVar("T").
        saved = self._active_type_vars
        self._active_type_vars = {name: TyVar(name) for name in s.type_params}
        try:
            for fname, ftype in s.fields:
                if fname in seen:
                    raise CheckError(
                        f"tablet {s.name!r}: duplicate field {fname!r}",
                        s.line, s.col,
                    )
                seen.add(fname)
                fty = self._resolve_type(ftype, f"field {fname!r} of tablet {s.name!r}")
                # Buffers are scope-lifetime: if the enclosing struct
                # outlives the scope, the buffer doesn't. Reject here
                # until we have an ownership story for it.
                if isinstance(fty, TyBuffer):
                    raise CheckError(
                        f"tablet {s.name!r}: field {fname!r} cannot be a buffer "
                        f"(buffers live on the stack; use a str or tablets)",
                        s.line, s.col,
                    )
                resolved.append((fname, fty))
        finally:
            self._active_type_vars = saved
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
        # Generic fns: type parameters are in scope as TyVars while we
        # resolve the signature and (later) check the body.
        self.fn_type_params[fn.name] = tuple(fn.type_params)
        saved = self._active_type_vars
        self._active_type_vars = {name: TyVar(name) for name in fn.type_params}
        try:
            params = tuple(
                self._resolve_type(p.type, f"parameter {p.name!r} of {fn.name!r}")
                for p in fn.params
            )
            ret = (
                self._resolve_type(fn.return_type, f"return type of {fn.name!r}")
                if fn.return_type else UNIT
            )
        finally:
            self._active_type_vars = saved
        # Variadic param must be last and unique. Parse-time already
        # resolved the AST marker; detect by looking for TypeVariadicTablets
        # in the source-level param types.
        is_variadic = False
        for i, p in enumerate(fn.params):
            if isinstance(p.type, A.TypeVariadicTablets):
                if i != len(fn.params) - 1:
                    raise CheckError(
                        f"fn {fn.name!r}: variadic `tablets[...]T` parameter "
                        f"must be the last parameter",
                        p.line, p.col,
                    )
                is_variadic = True
        if isinstance(ret, TyBuffer):
            raise CheckError(
                f"fn {fn.name!r}: cannot return a buffer (its storage "
                f"would dangle — use a str or tablets instead)",
                fn.line, fn.col,
            )
        self.fns[fn.name] = TyFn(
            params=params, ret=ret, is_variadic=is_variadic,
            param_muts=tuple(p.is_mut for p in fn.params),
        )
        # Did-you-mean warning: users who write `fn add(a: Vec, b: Vec)
        # -> Vec` on user types probably meant `gloss add`. Warn but
        # don't reject — the regular fn may be intentional.
        if fn.name in GLOSS_OPS and not is_variadic:
            _sym, arity, _fixed_ret = GLOSS_OPS[fn.name]
            expected_arity = 2 if arity == "bin" else 1
            if len(params) == expected_arity and any(
                isinstance(p, (TyStruct, TySeal)) for p in params
            ):
                self._warn(
                    f"fn {fn.name!r} has a signature that looks like an "
                    f"operator overload on user types; did you mean "
                    f"`gloss {fn.name}`? (if you meant a regular fn, "
                    f"ignore this)",
                    fn.line, fn.col,
                )
        if fn.name == "main":
            if ret != I32:
                raise CheckError(
                    f"main must declare -> i32, got -> {ret}", fn.line, fn.col,
                )

    def _register_colophon(self, c: A.ColophonDecl) -> None:
        """Register an extern declaration in the fn table so call sites
        resolve it like any other function. Signature types are
        restricted to what the call-site marshaling knows how to
        bridge: integer primitives, bool, and the built-in str
        (marshaled to / from NUL-terminated C strings). Anything else
        fails typecheck with a clear error — user tablets, tablets,
        and handles will land in a follow-up."""
        if c.name in INTRINSIC_NAMES:
            raise CheckError(
                f"cannot declare colophon {c.name!r}: name is a built-in intrinsic",
                c.line, c.col,
            )
        if c.name in self.fns:
            raise CheckError(
                f"duplicate declaration {c.name!r}", c.line, c.col,
            )
        str_ty = self.structs.get("str")

        def is_primitive(t: Ty) -> bool:
            """Types that round-trip unchanged across the C ABI with no
            marshaling. Used to gate callback-signature acceptance."""
            return _is_int(t) or isinstance(t, (TyBool, TyUnit))

        def check_ffi_type(ty: Ty, where: str, *, is_return: bool) -> None:
            if _is_int(ty) or isinstance(ty, TyBool):
                return
            if str_ty is not None and ty == str_ty:
                return
            # User tablet: passes by value across the C ABI, or by
            # pointer when the param is `mut`. Restricted to parameters
            # here — struct returns across the FFI aren't exposed yet
            # (most libc fns that return structs do so for 'stat'-
            # like shapes that are platform-dependent; defer).
            if isinstance(ty, TyStruct) and not is_return:
                return
            # buffer[N]T decays to a T* at the C boundary — the natural
            # shape for `recv`/`send`-style fns. Params only; no return.
            if isinstance(ty, TyBuffer) and not is_return:
                return
            # Callback fn pointer — `fn(prim, ...) -> prim`. Accepted
            # only when every param and the return are primitive
            # (int, bool, unit); anything else would need marshaling
            # inside the C-invoked callback, which we don't have a
            # story for. Nested fn types (fns returning fns) are
            # rejected at v0.1.
            if isinstance(ty, TyFn):
                if all(is_primitive(p) for p in ty.params) and is_primitive(ty.ret):
                    return
                raise CheckError(
                    f"colophon {c.name!r}: {where} has type {ty}; "
                    f"callback signatures are primitives-only (int / "
                    f"bool / unit) at v0.1 — str / struct / wedge / "
                    f"nested fn aren't marshalable across a C-invoked "
                    f"callback",
                    c.line, c.col,
                )
            raise CheckError(
                f"colophon {c.name!r}: {where} has type {ty}, which isn't "
                f"marshalable across the C boundary yet (allowed: ints, "
                f"bool, str, buffer, fn(prim)->prim, and — for "
                f"parameters — user tablets)",
                c.line, c.col,
            )

        params = tuple(
            self._resolve_type(p.type, f"parameter {p.name!r} of colophon {c.name!r}")
            for p in c.params
        )
        for p, pty in zip(c.params, params):
            check_ffi_type(pty, f"parameter {p.name!r}", is_return=False)
        ret = (
            self._resolve_type(c.return_type, f"return type of colophon {c.name!r}")
            if c.return_type else UNIT
        )
        if c.return_type is not None:
            check_ffi_type(ret, "return type", is_return=True)
        self.fns[c.name] = TyFn(
            params=params, ret=ret,
            param_muts=tuple(p.is_mut for p in c.params),
        )
        # Tracked so codegen can emit extern declarations instead of
        # trying to lower a body.
        self.colophons.add(c.name)

    def _register_gloss(self, g: A.GlossDecl) -> None:
        """Validate a gloss decl, register it in the dispatch table
        keyed by (op, lhs_ty, rhs_ty), and mint a mangled name for
        the fn table so codegen can emit a regular call. Enforces:
          - op must be a known gloss-op name
          - arity matches the op (binary vs unary)
          - operand types must be user tablets or seals (not primitives)
          - return type matches any fixed constraint (`eq` -> bool, etc.)
          - no duplicate registration for the same (op, lhs, rhs)"""
        if g.op not in GLOSS_OPS:
            valid = ", ".join(sorted(GLOSS_OPS))
            raise CheckError(
                f"gloss {g.op!r}: unknown operator name; valid names are "
                f"{valid}",
                g.line, g.col,
            )
        _sym, arity, fixed_ret = GLOSS_OPS[g.op]
        expected_arity = 2 if arity == "bin" else 1
        if len(g.params) != expected_arity:
            raise CheckError(
                f"gloss {g.op!r}: expects {expected_arity} param(s), "
                f"got {len(g.params)}",
                g.line, g.col,
            )

        params = tuple(
            self._resolve_type(p.type, f"gloss {g.op!r} param {p.name!r}")
            for p in g.params
        )
        ret = (
            self._resolve_type(g.return_type, f"gloss {g.op!r} return type")
            if g.return_type else UNIT
        )

        # Operand-type restriction: at least one operand must be a
        # user-defined type (TyStruct or TySeal). Overloading
        # primitive+primitive would shadow built-ins and produce
        # surprising behavior; we reject at decl.
        def is_user_type(t: Ty) -> bool:
            return isinstance(t, (TyStruct, TySeal))
        if not any(is_user_type(p) for p in params):
            raise CheckError(
                f"gloss {g.op!r}: at least one operand must be a user "
                f"tablet or seal (can't overload operators for "
                f"primitive+primitive combinations)",
                g.line, g.col,
            )

        if fixed_ret is not None:
            expected_ret = PRIM_TYPES[fixed_ret]
            if ret != expected_ret:
                raise CheckError(
                    f"gloss {g.op!r}: return type must be {fixed_ret}, "
                    f"got {ret}",
                    g.line, g.col,
                )

        # Dispatch key: (op, lhs, rhs). For unary ops, rhs=None.
        lhs_ty = params[0]
        rhs_ty = params[1] if arity == "bin" else None
        key = (g.op, lhs_ty, rhs_ty)
        if key in self.gloss_dispatch:
            raise CheckError(
                f"gloss {g.op!r}: duplicate definition for operand "
                f"types ({lhs_ty}" + (f", {rhs_ty}" if rhs_ty else "") + ")",
                g.line, g.col,
            )

        mangled = self._gloss_mangled_name(g.op, lhs_ty, rhs_ty)
        if mangled in self.fns:
            raise CheckError(
                f"gloss {g.op!r}: internal mangled name {mangled!r} "
                f"collides with an existing declaration",
                g.line, g.col,
            )
        self.fns[mangled] = TyFn(
            params=params, ret=ret,
            param_muts=tuple(p.is_mut for p in g.params),
        )
        self.gloss_dispatch[key] = mangled
        self.fn_type_params[mangled] = ()

    def _gloss_mangled_name(
        self, op: str, lhs: Ty, rhs: "Ty | None",
    ) -> str:
        """Deterministic internal symbol for a gloss dispatch entry.
        Stable across compilations (used only within one program, so
        uniqueness is the only requirement)."""
        def tag(t: Ty) -> str:
            if isinstance(t, TyStruct):
                base = t.name
                if t.args:
                    base += "__" + "_".join(tag(a) for a in t.args)
                return base
            if isinstance(t, TySeal):
                base = t.name
                if t.args:
                    base += "__" + "_".join(tag(a) for a in t.args)
                return base
            if isinstance(t, TyInt):
                return str(t)
            if isinstance(t, TyBool):
                return "bool"
            return str(t).replace(" ", "_")
        if rhs is None:
            return f"__gloss_{op}_{tag(lhs)}"
        return f"__gloss_{op}_{tag(lhs)}_{tag(rhs)}"

    def _check_gloss_body(self, g: A.GlossDecl) -> None:
        """Typecheck a gloss body as if it were a regular fn — same
        rules, same param binding, same tail-return semantics. The
        only difference is the name under which it's registered."""
        _sym, arity, _ = GLOSS_OPS[g.op]
        params = tuple(
            self._resolve_type(p.type, f"gloss {g.op!r} param {p.name!r}")
            for p in g.params
        )
        rhs_ty = params[1] if arity == "bin" else None
        mangled = self._gloss_mangled_name(g.op, params[0], rhs_ty)
        fake_fn = A.FnDecl(
            name=mangled,
            params=g.params,
            return_type=g.return_type,
            body=g.body,
            line=g.line,
            col=g.col,
        )
        self._check_fn_body(fake_fn)

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
        self._local_tablets = set()
        self._tainted = {}
        # Type params are in scope while we check the body so local
        # bindings with annotations like `mut cur: wedge Node<T>` work.
        saved = self._active_type_vars
        self._active_type_vars = {name: TyVar(name) for name in fn.type_params}
        fn_ty = self.fns[fn.name]
        try:
            for param, pty in zip(fn.params, fn_ty.params):
                self.scopes[0][param.name] = pty
            body_ty = self._tc_expr(fn.body, expected=fn_ty.ret)
        finally:
            self._active_type_vars = saved
        expected = fn_ty.ret
        # Unit-returning fn: the body's tail value is discarded, so
        # any tail type is fine. `fn add_route(mut a: App) { a.routes.
        # push(r) }` wrote the push call as a tail — push returns a
        # wedge the user doesn't care about. Previously this errored
        # with "body produces wedge Route, expected ()"; now we accept
        # and silently drop the value. Same relaxation spirit as the
        # if-in-statement-position rule.
        if isinstance(expected, TyUnit):
            pass
        elif not _coerces_to(body_ty, expected) and not isinstance(body_ty, TyDiverge):
            raise CheckError(
                f"in fn {fn.name!r}: body produces {body_ty}, expected {expected}",
                fn.line, fn.col,
            )
        # Trailing-expression return (no yield): apply escape check.
        if isinstance(expected, TyHandle) and self._expr_escapes(fn.body):
            raise CheckError(
                f"in fn {fn.name!r}: cannot return a wedge handle whose "
                f"tablets is declared locally — auto-release at scope "
                f"exit would free the storage while the caller still "
                f"holds the handle. Take the tablets as a parameter "
                f"instead.",
                fn.line, fn.col,
            )

    # --- type resolution --------------------------------------------

    def _resolve_type(self, t: A.TypeExpr, where: str) -> Ty:
        if isinstance(t, A.TypeName):
            if t.name in PRIM_TYPES:
                return PRIM_TYPES[t.name]
            # Inside a generic decl body, the type-parameter names are
            # in scope as fresh type variables.
            if t.name in self._active_type_vars:
                return self._active_type_vars[t.name]
            if t.name in self.structs:
                # Using a generic tablet's name without type args is
                # only valid if the tablet is non-generic.
                params = self.struct_type_params.get(t.name, ())
                if params:
                    raise CheckError(
                        f"{where}: tablet {t.name!r} expects "
                        f"{len(params)} type argument(s): "
                        f"write `{t.name}<...>`",
                        t.line, t.col,
                    )
                return self.structs[t.name]
            if t.name in self.seals:
                params = self.seal_type_params.get(t.name, ())
                if params:
                    raise CheckError(
                        f"{where}: seal {t.name!r} expects "
                        f"{len(params)} type argument(s): "
                        f"write `{t.name}<...>`",
                        t.line, t.col,
                    )
                return self.seals[t.name]
            raise CheckError(
                f"{where}: unknown type {t.name!r}", t.line, t.col,
            )
        if isinstance(t, A.TypeApply):
            if t.name in self.structs:
                params = self.struct_type_params.get(t.name, ())
                if len(params) != len(t.args):
                    raise CheckError(
                        f"{where}: tablet {t.name!r} expects "
                        f"{len(params)} type argument(s), got {len(t.args)}",
                        t.line, t.col,
                    )
                resolved_args = tuple(
                    self._resolve_type(a, f"{where} type arg") for a in t.args
                )
                return TyStruct(name=t.name, args=resolved_args)
            if t.name in self.seals:
                params = self.seal_type_params.get(t.name, ())
                if len(params) != len(t.args):
                    raise CheckError(
                        f"{where}: seal {t.name!r} expects "
                        f"{len(params)} type argument(s), got {len(t.args)}",
                        t.line, t.col,
                    )
                resolved_args = tuple(
                    self._resolve_type(a, f"{where} type arg") for a in t.args
                )
                return TySeal(name=t.name, args=resolved_args)
            raise CheckError(
                f"{where}: unknown type {t.name!r}",
                t.line, t.col,
            )
        if isinstance(t, A.TypeTablets):
            inner = self._resolve_type(t.element, f"{where} element")
            return TyTablets(size=t.size, element=inner)
        if isinstance(t, A.TypeBuffer):
            inner = self._resolve_type(t.element, f"{where} element")
            # v0.1 scope: only byte buffers. Narrowing this avoids
            # committing to a story for struct-element buffers, which
            # would intersect with ownership in ways we haven't spec'd.
            if not (isinstance(inner, TyInt) and inner.width == 8 and not inner.signed):
                raise CheckError(
                    f"{where}: buffer element must be u8 in v0.1, got {inner}",
                    t.line, t.col,
                )
            return TyBuffer(size=t.size, element=inner)
        if isinstance(t, A.TypeVariadicTablets):
            # Variadic param marker — only legal on the last parameter
            # of a fn declaration. Resolved into a TyTablets with a
            # canonical chunk size that codegen will also use at the
            # call site when synthesising the literal.
            inner = self._resolve_type(t.element, f"{where} element")
            return TyTablets(size=VARIADIC_CHUNK_SIZE, element=inner)
        if isinstance(t, A.TypeArray):
            raise CheckError(
                f"{where}: array types are not supported in v0.1",
                t.line, t.col,
            )
        if isinstance(t, A.TypePointer):
            inner = self._resolve_type(t.element, f"{where} element")
            return TyPtr(element=inner)
        if isinstance(t, A.TypeHandle):
            inner = self._resolve_type(t.element, f"{where} element")
            return TyHandle(element=inner)
        if isinstance(t, A.TypeFn):
            params = tuple(
                self._resolve_type(p, f"{where} fn-type parameter")
                for p in t.params
            )
            ret = (
                self._resolve_type(t.return_type, f"{where} fn-type return")
                if t.return_type else UNIT
            )
            return TyFn(params=params, ret=ret)
        raise CheckError(
            f"{where}: unsupported type expression",
            getattr(t, "line", 0), getattr(t, "col", 0),
        )

    # --- scope ------------------------------------------------------

    def _push(self) -> None:
        self.scopes.append({})
        self._borrow_sources.append({})

    def _pop(self) -> None:
        self.scopes.pop()
        # Borrows go out of scope when the block ends; their
        # invalidation state no longer matters. Clear any leftover
        # invalidation entries pointing at borrows we're about to drop.
        dropped = self._borrow_sources.pop()
        for name in dropped:
            self._invalidated.discard(name)

    def _register_borrow(self, name: str, root: str) -> None:
        """Mark `name` as a borrow whose heap bytes are rooted in
        `root` (the outermost owning Ident). Used by `_tc_binding` for
        Ident/Field/Index initializers and by match-pattern binders."""
        self._borrow_sources[-1][name] = root
        self._invalidated.discard(name)

    def _borrow_source_root(self, e: A.Expr) -> str | None:
        """Walk a borrow-source expression to its root Ident. Returns
        None for expressions that produce fresh ownership (Call,
        Copy, literals), which don't establish aliasing relationships."""
        if isinstance(e, A.Ident):
            # Follow the borrow chain: `step q = p; step r = q` — r's
            # root is whatever p's root was (or p itself if p is an
            # owning binding).
            return self._root_for(e.name)
        if isinstance(e, A.Field):
            return self._borrow_source_root(e.target)
        if isinstance(e, A.Index):
            return self._borrow_source_root(e.target)
        return None

    def _root_for(self, name: str) -> str:
        """If `name` is itself a borrow, return the root its chain
        points at. Otherwise `name` IS the root."""
        for scope in reversed(self._borrow_sources):
            if name in scope:
                return scope[name]
        return name

    def _invalidate_root(self, root: str) -> None:
        """Flag every live borrow rooted at `root` as invalidated. A
        subsequent read of any such binding will error out. Called when
        `root` (or a path through `root`) is mut-reached by a call or
        by a direct assignment."""
        for scope in self._borrow_sources:
            for name, src in scope.items():
                if src == root:
                    self._invalidated.add(name)

    def _check_use_not_invalidated(
        self, name: str, line: int, col: int,
    ) -> None:
        """At each Ident read, enforce the freeze-while-borrow rule:
        the borrow hasn't been invalidated by a mut-reach to its root
        since it was bound. The fix the user needs is `step n = copy
        name` at the binding site."""
        if name in self._invalidated:
            raise CheckError(
                f"use of borrow {name!r} after its source may have "
                f"been mutated — bind a fresh copy at the borrow "
                f"site with `step n = copy {name}` and use `n` after "
                f"the mutation instead",
                line, col,
            )

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
        if isinstance(s, A.ExprStmt):
            # `if` used as a bare statement — the value is provably
            # discarded, so arms don't need to unify. Mark the node
            # before typechecking so `_tc_if` sees the flag.
            self._mark_discarded_expr(s.expr)
            self._tc_expr(s.expr)
            return
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
            if isinstance(declared, TyTablets) and b.is_mut:
                self._local_tablets.add(b.name)
            if isinstance(declared, TyHandle):
                self._tainted[b.name] = False
            return
        init_ty = self._tc_expr(b.init, expected=declared)
        if declared is not None:
            if not _coerces_to(init_ty, declared):
                raise CheckError(
                    f"binding {b.name!r}: initializer has type {init_ty}, "
                    f"expected {declared}",
                    b.line, b.col,
                )
            self._bind(b.name, declared, b.line, b.col)
            final_ty = declared
        else:
            self._bind(b.name, init_ty, b.line, b.col)
            final_ty = init_ty
        # Track provenance for handles and locally-declared tablets so
        # the escape check at function return can reject UAFs.
        if isinstance(final_ty, TyTablets) and b.is_mut:
            self._local_tablets.add(b.name)
        if isinstance(final_ty, TyHandle):
            self._tainted[b.name] = self._expr_escapes(b.init)
        # Freeze-while-borrow: if this binding aliases into some
        # existing storage (borrow-init via Ident/Field/Index), remember
        # its source root. A later mut-reach to that root invalidates
        # this binding; subsequent reads error out with a `copy` hint.
        if self._needs_borrow_tracking(final_ty):
            root = self._borrow_source_root(b.init)
            if root is not None:
                self._register_borrow(b.name, root)

    def _needs_borrow_tracking(self, ty: Ty) -> bool:
        """Which types are susceptible to borrow invalidation? Any
        cleanup-bearing type — str, user struct, tablets, or seal with
        cleanup-bearing payload. Scalars, wedges, and fn pointers don't
        carry heap state so can't UAF from mutation."""
        str_ty = self.structs.get("str")
        if str_ty is not None and ty == str_ty:
            return True
        if isinstance(ty, TyStruct):
            return True
        if isinstance(ty, TyTablets):
            return True
        if isinstance(ty, TySeal):
            return True
        return False

    def _tc_assign(self, a: A.Assign) -> None:
        # Type-check the target as an expression: this both validates
        # the chain (field names must exist on their struct types) and
        # yields the expected type of the RHS.
        target_ty = self._tc_expr(a.target)
        value_ty = self._tc_expr(a.value, expected=target_ty)
        if not _coerces_to(value_ty, target_ty):
            raise CheckError(
                f"assignment target has type {target_ty}, "
                f"value has type {value_ty}",
                a.line, a.col,
            )
        # Taint propagation for mut handle bindings: once tainted,
        # always tainted (covers the case `head = lib.push(...)` where
        # `head` later gets returned).
        if isinstance(target_ty, TyHandle) and isinstance(a.target, A.Ident):
            if self._expr_escapes(a.value):
                self._tainted[a.target.name] = True
        # Freeze-while-borrow: assignment to a path invalidates every
        # live borrow rooted at the path's base. Covers `s.field = x`,
        # `arr[i] = x`, and `r = x` — all paths whose mutation could
        # free or overwrite the bytes a borrow aliases into.
        root = self._assign_target_root(a.target)
        if root is not None:
            self._invalidate_root(root)

    def _invalidate_mut_call_args(
        self, fn_ty: "TyFn", args: list[A.Expr],
    ) -> None:
        """For each call argument passed as `mut`, invalidate every
        live borrow rooted at that arg's base. Used by both direct fn
        calls and first-class fn-value calls."""
        if not fn_ty.param_muts:
            return
        for is_mut, arg in zip(fn_ty.param_muts, args):
            if not is_mut:
                continue
            root = self._assign_target_root(arg)
            if root is not None:
                self._invalidate_root(root)

    def _assign_target_root(self, target: A.Expr) -> str | None:
        """Root of an assignment lvalue. `r.field.field2 = x` roots at
        `r`; `arr[i] = x` roots at `arr`; `x = y` roots at `x`."""
        if isinstance(target, A.Ident):
            return target.name
        if isinstance(target, A.Field):
            return self._assign_target_root(target.target)
        if isinstance(target, A.Index):
            return self._assign_target_root(target.target)
        return None

    def _tc_for(self, f: A.ForStmt) -> None:
        # Resolve element type before we look at the body so errors about
        # the iterable itself are reported at the for-stmt position.
        element_ty = self._for_element_type(f)
        self._push()
        try:
            self._bind(f.name, element_ty, f.line, f.col)
            self._mark_discarded_expr(f.body.tail)
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
        # Loop body's tail value is always discarded — if its tail is
        # an IfExpr, arms don't need to unify.
        self._mark_discarded_expr(w.body.tail)
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
        val_ty = self._tc_expr(y.value, expected=expected)
        if not _coerces_to(val_ty, expected):
            raise CheckError(
                f"yield: value has type {val_ty}, fn {self.current_fn!r} "
                f"returns {expected}",
                y.line, y.col,
            )
        if isinstance(expected, TyHandle) and self._expr_escapes(y.value):
            raise CheckError(
                f"yield: cannot return a wedge handle whose tablets "
                f"is declared locally — auto-release at scope exit "
                f"would free the storage while the caller still holds "
                f"the handle. Take the tablets as a parameter instead.",
                y.line, y.col,
            )

    def _expr_escapes(self, e: A.Expr) -> bool:
        """Would returning `e` hand the caller a handle into a tablets
        that the current function owns (and will therefore auto-release
        before control returns to the caller)?

        Conservative analysis: walk the expression, find its root. If
        the root is a locally-declared mut tablets (or an already-
        tainted handle binding), the answer is yes. Parameters and
        `lost` are safe; calls to other user functions trust the
        callee's own escape check."""
        if isinstance(e, A.LostLit):
            return False
        if isinstance(e, A.Ident):
            if e.name in self._local_tablets:
                return True
            return self._tainted.get(e.name, False)
        if isinstance(e, A.Field):
            return self._expr_escapes(e.target)
        if isinstance(e, A.Call):
            # tablets.push(...) — root is the tablets receiver.
            if (
                isinstance(e.callee, A.Field)
                and isinstance(e.callee.target, A.Ident)
                and e.callee.name == "push"
            ):
                return self._expr_escapes(e.callee.target)
            # Plain function call: trust the callee's own escape check.
            return False
        if isinstance(e, A.IfExpr):
            # Either arm could produce the returned value; taint is
            # the OR of both (a conservative-but-sound approximation).
            then_esc = any(
                self._expr_escapes(s.expr) if isinstance(s, A.ExprStmt) else False
                for s in [*e.then.stmts, *([A.ExprStmt(expr=e.then.tail)] if e.then.tail else [])]
            ) if e.then.tail else False
            if e.then.tail is not None:
                then_esc = self._expr_escapes(e.then.tail)
            else:
                then_esc = False
            if e.else_ is None:
                return then_esc
            if isinstance(e.else_, A.IfExpr):
                else_esc = self._expr_escapes(e.else_)
            elif e.else_.tail is not None:
                else_esc = self._expr_escapes(e.else_.tail)
            else:
                else_esc = False
            return then_esc or else_esc
        if isinstance(e, A.Block):
            return self._expr_escapes(e.tail) if e.tail is not None else False
        return False

    def _tc_release(self, r: A.ReleaseStmt) -> None:
        ty = self._lookup(r.name, r.line, r.col)
        if not isinstance(ty, TyTablets):
            raise CheckError(
                f"release: {r.name!r} is {ty}, not a tablets value",
                r.line, r.col,
            )

    # --- expressions ------------------------------------------------

    def _tc_expr(self, e: A.Expr, expected: Ty | None = None) -> Ty:
        if isinstance(e, A.IntLit):    return I64
        if isinstance(e, A.CharLit):   return U8
        if isinstance(e, A.BoolLit):   return BOOL
        if isinstance(e, A.LostLit):   return LOST
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
            # Nullary variants look like bare identifiers. Check the
            # variant registry first so `None` / `Circle` etc. resolve
            # to their seal type rather than "undefined name".
            if e.name in self.variant_lookup:
                return self._tc_variant_ident(e, expected)
            # Bare fn name used as an expression = first-class function
            # value. Its type is the fn signature (TyFn). Local
            # bindings shadow — if a user wrote `step print = 0`
            # locally, that wins, which is fine because intrinsics
            # can't be shadowed and user fn names collide at decl
            # time.
            if e.name not in self.scopes[-1] and not any(
                e.name in s for s in self.scopes
            ):
                if e.name in self.fns:
                    if e.name in self.colophons:
                        raise CheckError(
                            f"{e.name!r} is a colophon extern; taking its "
                            f"address as a value isn't supported yet",
                            e.line, e.col,
                        )
                    if self.fn_type_params.get(e.name):
                        raise CheckError(
                            f"{e.name!r} is generic; can't take its address "
                            f"as a fn value (monomorphize via a call first)",
                            e.line, e.col,
                        )
                    return self.fns[e.name]
            # Freeze-while-borrow: reject reads of a borrow whose
            # source has been mut-reached since it was bound.
            self._check_use_not_invalidated(e.name, e.line, e.col)
            return self._lookup(e.name, e.line, e.col)
        if isinstance(e, A.Unary):     return self._tc_unary(e)
        if isinstance(e, A.Binary):    return self._tc_binary(e)
        if isinstance(e, A.Call):      return self._tc_call(e, expected)
        if isinstance(e, A.Field):     return self._tc_field(e)
        if isinstance(e, A.Index):     return self._tc_index(e)
        if isinstance(e, A.Slice):     return self._tc_slice(e)
        if isinstance(e, A.Cast):      return self._tc_cast(e)
        if isinstance(e, A.Copy):      return self._tc_copy(e, expected)
        if isinstance(e, A.StructLit): return self._tc_struct_lit(e)
        if isinstance(e, A.TabletsLit): return self._tc_tablets_lit(e, expected)
        if isinstance(e, A.Block):     return self._tc_block(e, expected=expected)
        if isinstance(e, A.IfExpr):    return self._tc_if(e, expected)
        if isinstance(e, A.MatchExpr): return self._tc_match(e, expected)
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
            # User type: dispatch to gloss_neg if declared.
            gloss_ret = self._gloss_lookup_unary("neg", ty, e)
            if gloss_ret is not None:
                return gloss_ret
            raise CheckError(
                f"unary -: requires an integer, rat, or dish, got {ty}", e.line, e.col,
            )
        if e.op == "!":
            if isinstance(ty, TyBool): return BOOL
            gloss_ret = self._gloss_lookup_unary("not", ty, e)
            if gloss_ret is not None:
                return gloss_ret
            raise CheckError(
                f"unary !: requires a bool, got {ty}", e.line, e.col,
            )
        raise CheckError(f"unknown unary operator {e.op!r}", e.line, e.col)

    def _gloss_lookup_unary(
        self, op_name: str, operand: Ty, e: A.Unary,
    ) -> "Ty | None":
        key = (op_name, operand, None)
        mangled = self.gloss_dispatch.get(key)
        if mangled is None:
            return None
        fn_ty = self.fns[mangled]
        # Record so codegen can route to the mangled fn call.
        self.gloss_call_for_node[id(e)] = mangled
        return fn_ty.ret

    def _gloss_lookup_binary(
        self, op_name: str, lhs: Ty, rhs: Ty, e: A.Binary,
    ) -> "Ty | None":
        key = (op_name, lhs, rhs)
        mangled = self.gloss_dispatch.get(key)
        if mangled is None:
            return None
        fn_ty = self.fns[mangled]
        self.gloss_call_for_node[id(e)] = mangled
        return fn_ty.ret

    # Reverse map: operator symbol -> gloss-op name for dispatch lookup.
    # Comparisons map to their matching gloss-op; `!=` derives from
    # `eq` with a negation applied at typecheck time (see below).
    _BIN_OP_GLOSS = {
        "+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod",
        "==": "eq", "!=": "eq",
        "<": "lt", "<=": "le", ">": "gt", ">=": "ge",
    }

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
            # str + str = str_concat. Covers `s += t` via the parser's
            # aug-assign desugar. Not a general operator-overload path —
            # we know both sides statically and emit the concat inline.
            str_ty = self.structs.get("str")
            if op == "+" and str_ty is not None and lhs == str_ty and rhs == str_ty:
                return str_ty
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
            # User-defined operator overload via `gloss`: dispatch on
            # the exact (lhs, rhs) type pair. Registered fns live in
            # `gloss_dispatch`; a miss falls through to the error.
            gloss_name = self._BIN_OP_GLOSS.get(op)
            if gloss_name is not None:
                gloss_ret = self._gloss_lookup_binary(gloss_name, lhs, rhs, e)
                if gloss_ret is not None:
                    return gloss_ret
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
            # Tablet handles: == / != only (ordering isn't meaningful).
            # Either both are `wedge T` with matching T, or one side
            # is `lost` which coerces to any handle type.
            if op in ("==", "!="):
                handle_involved = isinstance(lhs, (TyHandle, TyLost)) or isinstance(rhs, (TyHandle, TyLost))
                if handle_involved and (
                    (isinstance(lhs, TyHandle) and isinstance(rhs, TyHandle) and lhs.element == rhs.element)
                    or isinstance(lhs, TyLost) or isinstance(rhs, TyLost)
                ):
                    return BOOL
            # Inside a generic fn body, two TyVars with the same name
            # stand for whatever the eventual specialization binds.
            # Accept `==`/`!=` here; if the specialization picks a type
            # that doesn't actually support equality, codegen will
            # surface the problem.
            if (
                op in ("==", "!=")
                and isinstance(lhs, TyVar) and isinstance(rhs, TyVar)
                and lhs.name == rhs.name
            ):
                return BOOL
            # User-defined comparison via `gloss`. `!=` dispatches to
            # `gloss eq` and the codegen emits `!eq(a, b)`.
            gloss_name = self._BIN_OP_GLOSS.get(op)
            if gloss_name is not None:
                gloss_ret = self._gloss_lookup_binary(gloss_name, lhs, rhs, e)
                if gloss_ret is not None:
                    return gloss_ret
            raise CheckError(
                f"{op} requires matching comparable operands, got {lhs} and {rhs}",
                e.line, e.col,
            )

        raise CheckError(f"unknown binary operator {op!r}", e.line, e.col)

    # --- generics: unification & substitution --------------------------

    def _unify(self, pattern: Ty, concrete: Ty, subst: dict[str, Ty]) -> None:
        """Match a parameterized `pattern` against a concrete type,
        accumulating type-variable bindings into `subst`. Raises
        `_UnifyError` if the shapes disagree. Accepts coercion-friendly
        shapes on the leaves (lost → handle, int → dish, etc.) so call
        sites pass the same way they did pre-generics.

        Note: we apply the current subst to the pattern as we go, so
        if T was already bound to i64 in a previous param, a later
        pattern reference to T gets the concrete form."""
        pattern = self._substitute(pattern, subst)
        concrete = self._substitute(concrete, subst)
        if isinstance(pattern, TyVar) and isinstance(concrete, TyVar):
            if pattern.name == concrete.name:
                return   # same variable — already aligned, no-op
            subst[pattern.name] = concrete
            return
        if isinstance(pattern, TyVar):
            # Bind the variable. If it's `lost`, leave unbound for now;
            # the variable gets pinned later when another param supplies
            # a concrete element type.
            if isinstance(concrete, TyLost):
                return
            if self._occurs_in(pattern.name, concrete):
                raise _UnifyError(f"occurs check: {pattern} in {concrete}")
            subst[pattern.name] = concrete
            return
        if isinstance(concrete, TyVar):
            if self._occurs_in(concrete.name, pattern):
                raise _UnifyError(f"occurs check: {concrete} in {pattern}")
            subst[concrete.name] = pattern
            return
        if isinstance(pattern, TyHandle) and isinstance(concrete, TyHandle):
            self._unify(pattern.element, concrete.element, subst)
            return
        if isinstance(pattern, TyHandle) and isinstance(concrete, TyLost):
            return
        if isinstance(pattern, TyTablets) and isinstance(concrete, TyTablets):
            if pattern.size != concrete.size:
                raise _UnifyError(
                    f"tablets size mismatch: {pattern.size} vs {concrete.size}",
                )
            self._unify(pattern.element, concrete.element, subst)
            return
        if isinstance(pattern, TyStruct) and isinstance(concrete, TyStruct):
            if pattern.name != concrete.name:
                raise _UnifyError(f"{pattern.name} vs {concrete.name}")
            if len(pattern.args) != len(concrete.args):
                raise _UnifyError(
                    f"{pattern.name}: arity mismatch "
                    f"{len(pattern.args)} vs {len(concrete.args)}",
                )
            for p, c in zip(pattern.args, concrete.args):
                self._unify(p, c, subst)
            return
        if isinstance(pattern, TySeal) and isinstance(concrete, TySeal):
            if pattern.name != concrete.name:
                raise _UnifyError(f"{pattern.name} vs {concrete.name}")
            if len(pattern.args) != len(concrete.args):
                raise _UnifyError(
                    f"{pattern.name}: arity mismatch "
                    f"{len(pattern.args)} vs {len(concrete.args)}",
                )
            for p, c in zip(pattern.args, concrete.args):
                self._unify(p, c, subst)
            return
        # Leaves — rely on the coercion rules that already govern call
        # sites. If pattern == concrete exactly, trivially fine;
        # otherwise let `_coerces_to` handle it later. Here we only
        # raise on obvious structural disagreement.
        if pattern == concrete:
            return
        # Permit integer / dish / rat mixes — the caller will still
        # gate those via `_coerces_to` after unification.
        if _coerces_to(concrete, pattern):
            return
        raise _UnifyError(f"{pattern} vs {concrete}")

    def _occurs_in(self, var_name: str, ty: Ty) -> bool:
        """Does TyVar(var_name) appear anywhere in `ty`? Used by unify's
        occurs check to reject cyclic bindings like T = wedge T."""
        if isinstance(ty, TyVar):
            return ty.name == var_name
        if isinstance(ty, TyHandle):
            return self._occurs_in(var_name, ty.element)
        if isinstance(ty, TyTablets):
            return self._occurs_in(var_name, ty.element)
        if isinstance(ty, TyStruct):
            return any(self._occurs_in(var_name, a) for a in ty.args)
        if isinstance(ty, TySeal):
            return any(self._occurs_in(var_name, a) for a in ty.args)
        return False

    def _substitute(self, ty: Ty, subst: dict[str, Ty]) -> Ty:
        if isinstance(ty, TyVar):
            bound = subst.get(ty.name)
            return self._substitute(bound, subst) if bound is not None else ty
        if isinstance(ty, TyHandle):
            inner = self._substitute(ty.element, subst)
            return TyHandle(element=inner)
        if isinstance(ty, TyTablets):
            inner = self._substitute(ty.element, subst)
            return TyTablets(size=ty.size, element=inner)
        if isinstance(ty, TyStruct) and ty.args:
            new_args = tuple(self._substitute(a, subst) for a in ty.args)
            return TyStruct(name=ty.name, args=new_args)
        if isinstance(ty, TySeal) and ty.args:
            new_args = tuple(self._substitute(a, subst) for a in ty.args)
            return TySeal(name=ty.name, args=new_args)
        return ty

    def _tc_call(self, e: A.Call, expected: Ty | None = None) -> Ty:
        # Method call on a tablets receiver. The receiver may be a bare
        # Ident (t.push) or a field chain rooted at one (buf.bytes.push);
        # _tc_method_call routes through _tc_expr on the receiver so
        # nested-field method dispatch works uniformly.
        if isinstance(e.callee, A.Field):
            return self._tc_method_call(e)

        if not isinstance(e.callee, A.Ident):
            raise CheckError(
                "only direct function calls are supported", e.line, e.col,
            )
        name = e.callee.name

        # Variant constructor: `Some(42)`, `Circle(rat(1, 2))`, etc.
        if name in self.variant_lookup:
            return self._tc_variant_call(e, expected)

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

        if name == "str_slice":
            return self._tc_str_intrinsic(e, name)

        if name in ("int_to_str", "sex_to_str"):
            return self._tc_to_str_intrinsic(e, name)

        if name == "bytes_to_str":
            return self._tc_bytes_to_str(e)

        if name == "buffer_to_str":
            return self._tc_buffer_to_str(e)

        # Local binding named `name` shadows any global fn. If it's a
        # fn-value binding (TyFn), do an indirect call through it;
        # any other type is not callable and errors early.
        for scope in reversed(self.scopes):
            if name in scope:
                bound = scope[name]
                if not isinstance(bound, TyFn):
                    raise CheckError(
                        f"{name!r} has type {bound}, not callable",
                        e.line, e.col,
                    )
                return self._tc_fn_value_call(e, bound)

        fn = self.fns.get(name)
        if fn is None:
            raise CheckError(
                f"in fn {self.current_fn!r}: unknown function {name!r}"
                f"{_suggest(name, self.fns)}",
                e.line, e.col,
            )
        # Variadic call: split args into (fixed, tail). The fixed args
        # typecheck against the first (n-1) params; the tail becomes a
        # synthesized TabletsLit that gets typechecked against the last
        # param (a TyTablets) and stored on the sideband for codegen.
        if fn.is_variadic:
            fixed_count = len(fn.params) - 1
            if len(e.args) < fixed_count:
                raise CheckError(
                    f"{name} expects at least {fixed_count} arg(s), got "
                    f"{len(e.args)}",
                    e.line, e.col,
                )
            tail_param = fn.params[-1]
            assert isinstance(tail_param, TyTablets)
            lit = A.TabletsLit(
                size=tail_param.size,
                element=None,
                fields=list(e.args[fixed_count:]),
            )
            lit.line = e.line
            lit.col = e.col
            self.variadic_lit_for_call[id(e)] = lit
            # Rewrite e.args to the (fixed + literal) shape so the
            # existing matching loop handles it uniformly.
            new_args: list[A.Expr] = list(e.args[:fixed_count])
            new_args.append(lit)
            e.args = new_args
        if len(e.args) != len(fn.params):
            raise CheckError(
                f"{name} expects {len(fn.params)} args, got {len(e.args)}",
                e.line, e.col,
            )
        # Infer type-parameter substitutions for a generic fn by
        # unifying each param's declared type against the actual arg
        # type. For non-generic fns the instantiation map is empty
        # and everything works as before.
        #
        # We freshen the callee's type-parameter TyVars at this call
        # site. Without freshening, a generic-in-generic call (list_
        # contains<T> → list_find<T>) would unify TyVar("T") against
        # TyVar("T") and learn nothing — the callee's T and the
        # caller's T share a name but are distinct bindings.
        type_params = self.fn_type_params.get(name, ())
        inst_names = [f"{name}.{tp}#{id(e)}" for tp in type_params]
        inst_subst = {tp: TyVar(fresh) for tp, fresh in zip(type_params, inst_names)}
        subst: dict[str, Ty] = {}
        # Expected-type hint per arg: the freshened param type if it
        # contains no open type variables, otherwise None. Supplies
        # context to variant/match construction inside args.
        arg_hints: list[Ty | None] = []
        for pty in fn.params:
            h = self._substitute(pty, inst_subst)
            arg_hints.append(
                h if not self._has_open_tyvar(h, inst_names) else None
            )
        arg_tys = [self._tc_expr(a, expected=h) for a, h in zip(e.args, arg_hints)]
        # Freeze-while-borrow: each mut-annotated param invalidates
        # every live borrow rooted at the root of its arg expression.
        # The callee can write through the mut param, which may free
        # or overwrite whatever a borrow aliases.
        self._invalidate_mut_call_args(fn, e.args)
        for i, (at, pty) in enumerate(zip(arg_tys, fn.params)):
            freshened = self._substitute(pty, inst_subst)
            try:
                self._unify(freshened, at, subst)
            except _UnifyError as ex:
                raise CheckError(
                    f"call to {name!r}: arg {i} has type {at}, expected {pty}"
                    f"{f' ({ex.detail})' if ex.detail else ''}",
                    e.line, e.col,
                ) from None
        for i, (at, pty) in enumerate(zip(arg_tys, fn.params)):
            inst = self._substitute(self._substitute(pty, inst_subst), subst)
            if not _coerces_to(at, inst):
                raise CheckError(
                    f"call to {name!r}: arg {i} has type {at}, expected {inst}",
                    e.line, e.col,
                )
        ret = self._substitute(self._substitute(fn.ret, inst_subst), subst)
        # Record the concrete type-arg tuple for codegen monomorphization.
        if type_params:
            missing = [tp for tp, fresh in zip(type_params, inst_names) if fresh not in subst]
            if missing:
                raise CheckError(
                    f"call to {name!r}: could not infer type parameter(s) "
                    f"{', '.join(missing)} from argument types",
                    e.line, e.col,
                )
            self.mono_call_args[id(e)] = tuple(
                self._substitute(subst[fresh], subst) for fresh in inst_names
            )
        return ret

    def _tc_str_intrinsic(self, e: A.Call, name: str) -> Ty:
        str_ty = self.structs.get("str")
        if str_ty is None:
            raise CheckError(
                f"{name}: str type not in scope (stdlib may be missing)",
                e.line, e.col,
            )
        if name == "str_slice":
            if len(e.args) != 3:
                raise CheckError(
                    "str_slice(s, lo, hi) takes exactly three arguments",
                    e.line, e.col,
                )
            at = self._tc_expr(e.args[0])
            if at != str_ty:
                raise CheckError(
                    f"str_slice: first argument must be str, got {at}",
                    e.line, e.col,
                )
            for i in (1, 2):
                bt = self._tc_expr(e.args[i])
                if not _is_int(bt):
                    raise CheckError(
                        f"str_slice: argument {i} must be integer, got {bt}",
                        e.line, e.col,
                    )
            return str_ty
        raise CheckError(f"unknown str intrinsic {name!r}", e.line, e.col)

    def _tc_to_str_intrinsic(self, e: A.Call, name: str) -> Ty:
        str_ty = self.structs.get("str")
        if str_ty is None:
            raise CheckError(
                f"{name}: str type not in scope (stdlib may be missing)",
                e.line, e.col,
            )
        if len(e.args) != 1:
            raise CheckError(
                f"{name} takes exactly one argument", e.line, e.col,
            )
        at = self._tc_expr(e.args[0])
        if name == "int_to_str":
            if not _is_int(at):
                raise CheckError(
                    f"int_to_str: argument must be integer, got {at}",
                    e.line, e.col,
                )
        elif name == "sex_to_str":
            if not isinstance(at, TyDish):
                raise CheckError(
                    f"sex_to_str: argument must be sex/dish, got {at}",
                    e.line, e.col,
                )
        return str_ty

    def _tc_fn_value_call(self, e: A.Call, fn_ty: TyFn) -> Ty:
        """Typecheck an indirect call through a fn-valued binding —
        `step f = some_fn; f(args)`. No generics at call time (fn
        values aren't polymorphic), just arity + param/arg unification
        against the fn-pointer signature."""
        if len(fn_ty.params) != len(e.args):
            raise CheckError(
                f"fn-value call: signature expects {len(fn_ty.params)} "
                f"arg(s), got {len(e.args)}",
                e.line, e.col,
            )
        for i, (arg, pty) in enumerate(zip(e.args, fn_ty.params)):
            at = self._tc_expr(arg, expected=pty)
            if not _coerces_to(at, pty):
                raise CheckError(
                    f"fn-value call: arg {i} has type {at}, expected {pty}",
                    e.line, e.col,
                )
        return fn_ty.ret

    def _tc_bytes_to_str(self, e: A.Call) -> Ty:
        str_ty = self.structs.get("str")
        if str_ty is None:
            raise CheckError(
                "bytes_to_str: str type not in scope (stdlib may be missing)",
                e.line, e.col,
            )
        if len(e.args) != 1:
            raise CheckError(
                "bytes_to_str takes exactly one argument", e.line, e.col,
            )
        at = self._tc_expr(e.args[0])
        if not isinstance(at, TyTablets) or at.element != U8:
            raise CheckError(
                f"bytes_to_str: argument must be tablets[N]u8, got {at}",
                e.line, e.col,
            )
        return str_ty

    def _tc_buffer_to_str(self, e: A.Call) -> Ty:
        """`buffer_to_str(buf, n)` — copy the first `n` bytes of `buf`
        into a fresh heap-owned str. Buffers don't track "used bytes",
        so the user passes the length explicitly (matches `recv`'s
        return-value convention). Bounds-checked at runtime."""
        str_ty = self.structs.get("str")
        if str_ty is None:
            raise CheckError(
                "buffer_to_str: str type not in scope (stdlib may be missing)",
                e.line, e.col,
            )
        if len(e.args) != 2:
            raise CheckError(
                "buffer_to_str(buf, n) takes exactly two arguments",
                e.line, e.col,
            )
        at = self._tc_expr(e.args[0])
        if not isinstance(at, TyBuffer) or at.element != U8:
            raise CheckError(
                f"buffer_to_str: first argument must be buffer[N]u8, got {at}",
                e.line, e.col,
            )
        nt = self._tc_expr(e.args[1])
        if not _is_int(nt):
            raise CheckError(
                f"buffer_to_str: length argument must be integer, got {nt}",
                e.line, e.col,
            )
        return str_ty

    def _tc_method_call(self, e: A.Call) -> Ty:
        assert isinstance(e.callee, A.Field)
        method = e.callee.name
        # Resolve the receiver's type via the general expression path so
        # a chained `buf.bytes` (Field) routes through field-typecheck
        # and yields the correct TyTablets for the inner field.
        recv_ty = self._tc_expr(e.callee.target)
        # Report receiver shape in errors: an Ident receiver gives the
        # old "recv is X, not a tablets" message, field chains name the
        # expression structure.
        recv_name = (
            e.callee.target.name if isinstance(e.callee.target, A.Ident)
            else "<field expression>"
        )

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
                # Freeze-while-borrow: `a.push(x)` mut-reaches `a`.
                # Invalidate every live borrow rooted there — a push
                # may reallocate or append into the same chunk a
                # borrow aliases.
                recv_root = self._assign_target_root(e.callee.target)
                if recv_root is not None:
                    self._invalidate_root(recv_root)
                # push returns a handle to the newly-pushed element —
                # discarded automatically in statement position.
                return TyHandle(element=recv_ty.element)
            raise CheckError(
                f"tablets has no method {method!r}", e.line, e.col,
            )

        # Struct field that happens to be a fn value — `obj.run(x)` is
        # not a method dispatch but a field-access-then-indirect-call.
        # Resolve the field's type via the struct registry; if it's a
        # TyFn, treat the whole expression as a fn-value call.
        if isinstance(recv_ty, TyStruct):
            fields = self.struct_fields.get(recv_ty.name, ())
            for fname, fty in fields:
                if fname == method and isinstance(fty, TyFn):
                    return self._tc_fn_value_call(e, fty)

        raise CheckError(
            f"method call: {recv_name!r} is {recv_ty}, not a tablets",
            e.line, e.col,
        )

    def _tc_field(self, e: A.Field) -> Ty:
        target_ty = self._tc_expr(e.target)
        # Tablet handle: auto-deref for field access on `tablet T`
        # where T is a struct. `h.value` is shorthand for "load the
        # T this handle points to and take its `value` field." Rest
        # of the method treats it as if the user wrote on T directly.
        if isinstance(target_ty, TyHandle):
            target_ty = target_ty.element
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
        if isinstance(target_ty, TyBuffer):
            if e.name == "len":
                return I64
            raise CheckError(
                f"buffer has no field {e.name!r}; only len",
                e.line, e.col,
            )
        if isinstance(target_ty, TyStruct):
            fields = self.struct_fields[target_ty.name]
            # For generic tablets, substitute the instantiated type
            # args into the declared field type before returning it.
            # Skip no-op self-bindings (T → TyVar("T")) which would
            # otherwise produce a cyclic subst when the arg is still
            # a type variable from the enclosing fn's scope.
            type_params = self.struct_type_params.get(target_ty.name, ())
            subst = {}
            for tp, arg in zip(type_params, target_ty.args):
                if isinstance(arg, TyVar) and arg.name == tp:
                    continue
                subst[tp] = arg
            for fname, fty in fields:
                if fname == e.name:
                    return self._substitute(fty, subst)
            field_names = [n for n, _ in fields]
            raise CheckError(
                f"tablet {target_ty.name!r} has no field {e.name!r}"
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
                f"unknown tablet {e.name!r}"
                f"{_suggest(e.name, self.structs)}",
                e.line, e.col,
            )
        declared = self.struct_fields[e.name]
        type_params = self.struct_type_params.get(e.name, ())
        # Instantiate fresh type variables for the tablet's type params
        # so unification doesn't alias them to identically-named TyVars
        # from the enclosing fn's scope. `Node.T#<id(e)>` is unique
        # per literal, so inferred bindings stay scoped to this site.
        fresh_names = [f"{e.name}.{tp}#{id(e)}" for tp in type_params]
        inst_subst = {tp: TyVar(fresh) for tp, fresh in zip(type_params, fresh_names)}
        declared_map = {n: self._substitute(t, inst_subst) for n, t in declared}
        seen: set[str] = set()
        subst: dict[str, Ty] = {}
        field_val_tys: dict[str, Ty] = {}
        for fname, fexpr in e.fields:
            if fname not in declared_map:
                raise CheckError(
                    f"tablet {e.name!r}: unknown field {fname!r}"
                    f"{_suggest(fname, declared_map)}",
                    e.line, e.col,
                )
            if fname in seen:
                raise CheckError(
                    f"tablet {e.name!r}: duplicate field {fname!r} in literal",
                    e.line, e.col,
                )
            seen.add(fname)
            val_ty = self._tc_expr(fexpr)
            field_val_tys[fname] = val_ty
            if type_params:
                try:
                    self._unify(declared_map[fname], val_ty, subst)
                except _UnifyError:
                    # fall through to the concrete coercion check below,
                    # which produces a sharper error message for the user.
                    pass
        # After inference, check each field concretely with the subst.
        for fname, val_ty in field_val_tys.items():
            inst = self._substitute(declared_map[fname], subst)
            if not _coerces_to(val_ty, inst):
                raise CheckError(
                    f"tablet {e.name!r} field {fname!r}: got {val_ty}, "
                    f"expected {inst}",
                    e.line, e.col,
                )
        missing = [n for n, _ in declared if n not in seen]
        if missing:
            raise CheckError(
                f"tablet {e.name!r}: missing field(s) {', '.join(repr(n) for n in missing)}",
                e.line, e.col,
            )
        if type_params:
            missing_params = [
                tp for tp, fresh in zip(type_params, fresh_names)
                if fresh not in subst
            ]
            if missing_params:
                raise CheckError(
                    f"tablet {e.name!r}: could not infer type parameter(s) "
                    f"{', '.join(missing_params)} from field initializers "
                    f"(a type annotation on the binding would fix this)",
                    e.line, e.col,
                )
            concrete_args = tuple(
                self._substitute(subst[fresh], subst) for fresh in fresh_names
            )
            self.mono_struct_args[id(e)] = concrete_args
            return TyStruct(name=e.name, args=concrete_args)
        return self.structs[e.name]

    def _tc_tablets_lit(self, e: A.TabletsLit, expected: Ty | None) -> Ty:
        """`tablets[N]T { a, b, c }` — typecheck each element against T
        (either the spelled element type or one inferred from context).
        The parser only emits TabletsLit nodes with an explicit element
        type, but the variadic-call desugar synthesises nodes without
        one and relies on `expected` to pin T."""
        if e.element is not None:
            elem_ty = self._resolve_type(e.element, "tablets literal element type")
        elif isinstance(expected, TyTablets):
            elem_ty = expected.element
        else:
            raise CheckError(
                "tablets literal: cannot infer element type; add "
                "`tablets[N]T { ... }` form with an explicit element",
                e.line, e.col,
            )
        for i, fexpr in enumerate(e.fields):
            fty = self._tc_expr(fexpr, expected=elem_ty)
            if not _coerces_to(fty, elem_ty):
                raise CheckError(
                    f"tablets literal: element {i} has type {fty}, "
                    f"expected {elem_ty}",
                    e.line, e.col,
                )
        return TyTablets(size=e.size, element=elem_ty)

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
        if isinstance(target_ty, TyBuffer):
            idx_ty = self._tc_expr(e.index)
            if not _is_int(idx_ty):
                raise CheckError(
                    f"buffer index must be integer, got {idx_ty}",
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

    def _tc_slice(self, e: A.Slice) -> Ty:
        target_ty = self._tc_expr(e.target)
        str_ty = self.structs.get("str")
        if str_ty is None or target_ty != str_ty:
            raise CheckError(
                f"slice syntax is only supported on str, got {target_ty}",
                e.line, e.col,
            )
        for label, bound in (("lo", e.lo), ("hi", e.hi)):
            if bound is None:
                continue
            bt = self._tc_expr(bound)
            if not _is_int(bt):
                raise CheckError(
                    f"slice {label} bound must be integer, got {bt}",
                    e.line, e.col,
                )
        return str_ty

    def _tc_copy(self, e: A.Copy, expected: Ty | None = None) -> Ty:
        """`copy x` has the same type as its operand. At codegen we
        dispatch to `_deep_clone_if_cleanup_bearing`, which is a no-op
        on scalars/handles/fn-pointers — so `copy` on a plain int is
        harmless redundancy rather than an error. Future lint could
        warn, but v0.1 stays permissive."""
        return self._tc_expr(e.value, expected=expected)

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

    def _tc_block(
        self, b: A.Block, *,
        allow_nonbool_tail: bool = False,
        expected: Ty | None = None,
    ) -> Ty:
        self._push()
        try:
            for stmt in b.stmts:
                self._tc_stmt(stmt)
            if b.tail is None:
                return UNIT
            return self._tc_expr(b.tail, expected=expected)
        finally:
            self._pop()

    def _mark_stmt_if(self, e: A.IfExpr) -> None:
        """Back-compat shim: mark an IfExpr as stmt-position. Prefer
        `_mark_discarded_expr` for new call sites — it also dives into
        block tails and arm tails."""
        self._mark_discarded_expr(e)

    def _mark_discarded_expr(self, e: A.Expr | None) -> None:
        """Walk every tail position whose value is provably discarded and
        mark each IfExpr found along the way. Handles `else if` chains
        (both elif-form and nested `if` in an arm's tail) and blocks
        standing in for either arm."""
        if e is None:
            return
        if isinstance(e, A.IfExpr):
            self.stmt_if_nodes.add(id(e))
            # Arm tails inherit stmt-position: the arm's value is what
            # the if yields, and the if's value is discarded.
            if isinstance(e.then, A.Block):
                self._mark_discarded_expr(e.then.tail)
            if e.else_ is None:
                return
            if isinstance(e.else_, A.IfExpr):
                self._mark_discarded_expr(e.else_)
            elif isinstance(e.else_, A.Block):
                self._mark_discarded_expr(e.else_.tail)
        elif isinstance(e, A.Block):
            self._mark_discarded_expr(e.tail)

    def _tc_if(self, e: A.IfExpr, expected: Ty | None = None) -> Ty:
        cond_ty = self._tc_expr(e.cond)
        if not isinstance(cond_ty, TyBool):
            raise CheckError(
                f"if condition must be bool, got {cond_ty}", e.line, e.col,
            )
        in_stmt_position = id(e) in self.stmt_if_nodes
        # In statement position the arm values are discarded, so
        # neither arm is expected to produce a particular type. Drop
        # the `expected` hint so arms aren't forced to coerce their
        # tails toward an unused sink type.
        arm_expected = None if in_stmt_position else expected
        then_ty = self._tc_block(e.then, expected=arm_expected)
        if e.else_ is None:
            return UNIT
        if isinstance(e.else_, A.IfExpr):
            else_ty = self._tc_if(e.else_, arm_expected)
        else:
            else_ty = self._tc_block(e.else_, expected=arm_expected)
        if in_stmt_position:
            # Value is discarded; skip the unify and report () upward.
            return UNIT
        unified = _unify_if_arms(then_ty, else_ty)
        if unified is None:
            raise CheckError(
                f"if arms have different types: {then_ty} vs {else_ty}",
                e.line, e.col,
            )
        return unified

    # --- variants and match ----------------------------------------------

    def _tc_variant_ident(self, e: A.Ident, expected: Ty | None) -> Ty:
        """Typecheck a bare variant name like `None`. Must be nullary;
        generic seals require `expected` to pin their type parameters."""
        seal_name, vidx, field_tys = self.variant_lookup[e.name]
        if field_tys:
            raise CheckError(
                f"variant {e.name!r} takes {len(field_tys)} argument(s); "
                f"call it like `{e.name}(...)`",
                e.line, e.col,
            )
        type_params = self.seal_type_params.get(seal_name, ())
        self.variant_of_node[id(e)] = (seal_name, e.name, vidx)
        if not type_params:
            self.mono_variant_args[id(e)] = ()
            return TySeal(name=seal_name, args=())
        # Generic seal: try to pin type args from expected.
        if (
            isinstance(expected, TySeal)
            and expected.name == seal_name
            and len(expected.args) == len(type_params)
            and all(not isinstance(a, TyVar) for a in expected.args)
        ):
            self.mono_variant_args[id(e)] = expected.args
            return TySeal(name=seal_name, args=expected.args)
        raise CheckError(
            f"variant {e.name!r} of generic seal {seal_name!r}: cannot "
            f"infer type parameter(s) from the context. Add a type "
            f"annotation on the binding or call site (e.g. "
            f"`step x: {seal_name}<...> = {e.name}`).",
            e.line, e.col,
        )

    def _tc_variant_call(self, e: A.Call, expected: Ty | None) -> Ty:
        assert isinstance(e.callee, A.Ident)
        vname = e.callee.name
        seal_name, vidx, field_tys = self.variant_lookup[vname]
        self.variant_of_node[id(e)] = (seal_name, vname, vidx)
        if len(e.args) != len(field_tys):
            raise CheckError(
                f"variant {vname!r} takes {len(field_tys)} argument(s), "
                f"got {len(e.args)}",
                e.line, e.col,
            )
        type_params = self.seal_type_params.get(seal_name, ())
        # Freshen the seal's type parameters at this call site so they
        # don't collide with identically-named TyVars from an enclosing
        # generic fn/tablet body.
        fresh_names = [f"{seal_name}.{tp}#{id(e)}" for tp in type_params]
        inst_subst = {tp: TyVar(fn) for tp, fn in zip(type_params, fresh_names)}
        freshened_fields = [self._substitute(t, inst_subst) for t in field_tys]
        subst: dict[str, Ty] = {}
        # If the sink provides concrete type args, seed the subst so
        # arg inference can reuse them.
        if (
            isinstance(expected, TySeal)
            and expected.name == seal_name
            and len(expected.args) == len(type_params)
        ):
            for fn, exp_arg in zip(fresh_names, expected.args):
                subst[fn] = exp_arg
        arg_tys: list[Ty] = []
        for i, arg in enumerate(e.args):
            # Pass along the field's expected type if fully resolved,
            # so nested variants like `Some(None)` can find their T.
            hint = self._substitute(freshened_fields[i], subst)
            arg_hint = hint if not self._has_open_tyvar(hint, fresh_names) else None
            arg_tys.append(self._tc_expr(arg, expected=arg_hint))
        for i, (at, pty) in enumerate(zip(arg_tys, freshened_fields)):
            try:
                self._unify(pty, at, subst)
            except _UnifyError as ex:
                raise CheckError(
                    f"variant {vname!r}: arg {i} has type {at}, "
                    f"expected {self._substitute(pty, subst)}"
                    f"{f' ({ex.detail})' if ex.detail else ''}",
                    e.line, e.col,
                ) from None
        for i, (at, pty) in enumerate(zip(arg_tys, freshened_fields)):
            inst = self._substitute(pty, subst)
            if not _coerces_to(at, inst):
                raise CheckError(
                    f"variant {vname!r}: arg {i} has type {at}, expected {inst}",
                    e.line, e.col,
                )
        if type_params:
            resolved: list[Ty] = []
            missing: list[str] = []
            for tp, fn in zip(type_params, fresh_names):
                r = self._substitute(TyVar(fn), subst)
                if isinstance(r, TyVar) and r.name == fn:
                    missing.append(tp)
                resolved.append(r)
            if missing:
                raise CheckError(
                    f"variant {vname!r} of generic seal {seal_name!r}: "
                    f"could not infer type parameter(s) "
                    f"{', '.join(missing)}. A type annotation on the "
                    f"binding or an explicit cast would fix this.",
                    e.line, e.col,
                )
            resolved_tuple = tuple(resolved)
            self.mono_variant_args[id(e)] = resolved_tuple
            return TySeal(name=seal_name, args=resolved_tuple)
        self.mono_variant_args[id(e)] = ()
        return TySeal(name=seal_name, args=())

    def _has_open_tyvar(self, ty: Ty, fresh_names: list[str]) -> bool:
        if isinstance(ty, TyVar):
            return ty.name in fresh_names
        if isinstance(ty, (TyHandle, TyTablets)):
            return self._has_open_tyvar(ty.element, fresh_names)
        if isinstance(ty, TyStruct):
            return any(self._has_open_tyvar(a, fresh_names) for a in ty.args)
        if isinstance(ty, TySeal):
            return any(self._has_open_tyvar(a, fresh_names) for a in ty.args)
        return False

    def _tc_match(self, e: A.MatchExpr, expected: Ty | None = None) -> Ty:
        scrut_ty = self._tc_expr(e.scrutinee)
        if not isinstance(scrut_ty, TySeal):
            raise CheckError(
                f"match scrutinee must be a seal value, got {scrut_ty}",
                e.line, e.col,
            )
        seal_name = scrut_ty.name
        variants = self.seal_variants[seal_name]
        type_params = self.seal_type_params.get(seal_name, ())
        # Build a subst from the seal's declared type params to the
        # scrutinee's concrete type args, so pattern binder types get
        # the right concrete form.
        subst: dict[str, Ty] = {}
        for tp, ta in zip(type_params, scrut_ty.args):
            if isinstance(ta, TyVar) and ta.name == tp:
                continue
            subst[tp] = ta

        seen: set[str] = set()
        has_wildcard = False
        arm_tys: list[Ty] = []
        for arm in e.arms:
            self._push()
            try:
                if isinstance(arm.pattern, A.WildcardPattern):
                    has_wildcard = True
                elif isinstance(arm.pattern, A.VariantPattern):
                    matched = None
                    for i, (vn, vfs) in enumerate(variants):
                        if vn == arm.pattern.name:
                            matched = (i, vn, vfs)
                            break
                    if matched is None:
                        raise CheckError(
                            f"pattern {arm.pattern.name!r}: not a variant "
                            f"of seal {seal_name!r}",
                            arm.line, arm.col,
                        )
                    _, vn, vfs = matched
                    if vn in seen:
                        raise CheckError(
                            f"duplicate pattern for variant {vn!r}",
                            arm.line, arm.col,
                        )
                    seen.add(vn)
                    self.variant_of_node[id(arm.pattern)] = (seal_name, vn, matched[0])
                    if len(arm.pattern.binders) != len(vfs):
                        raise CheckError(
                            f"pattern {vn!r} has "
                            f"{len(arm.pattern.binders)} binder(s); "
                            f"variant has {len(vfs)} field(s)",
                            arm.line, arm.col,
                        )
                    # Match binders alias into the scrutinee's
                    # payload — register each as a borrow rooted at
                    # the scrutinee expression's root. Any mut-reach
                    # to that root during the arm body invalidates
                    # the binder.
                    scrut_root = self._borrow_source_root(e.scrutinee)
                    for binder, fty in zip(arm.pattern.binders, vfs):
                        if binder is None:
                            continue
                        concrete = self._substitute(fty, subst)
                        self._bind(binder, concrete, arm.line, arm.col)
                        if (
                            scrut_root is not None
                            and self._needs_borrow_tracking(concrete)
                        ):
                            self._register_borrow(binder, scrut_root)
                else:
                    raise CheckError(
                        f"unsupported pattern: {type(arm.pattern).__name__}",
                        arm.line, arm.col,
                    )
                body_ty = self._tc_expr(arm.body, expected=expected)
                arm_tys.append(body_ty)
            finally:
                self._pop()

        if not has_wildcard:
            all_names = {vn for vn, _ in variants}
            missing = sorted(all_names - seen)
            if missing:
                raise CheckError(
                    f"match on seal {seal_name!r} is not exhaustive; "
                    f"missing variant(s): {', '.join(missing)}",
                    e.line, e.col,
                )

        result = arm_tys[0]
        for ty in arm_tys[1:]:
            uni = _unify_if_arms(result, ty)
            if uni is None:
                raise CheckError(
                    f"match arms have different types: {result} vs {ty}",
                    e.line, e.col,
                )
            result = uni
        return result


def check(program: A.Program) -> "Checker":
    """Run type-checking. Raises CheckError on any problem. Returns the
    Checker instance so downstream passes can consult the mono sidebands
    (mono_call_args, mono_struct_args) and any collected warnings."""
    checker = Checker(program)
    checker.check()
    return checker
