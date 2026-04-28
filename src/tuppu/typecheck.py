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
class TyIVec:
    """`ivec<T>` — indirect vector: contiguous heap-allocated array of
    pointers to per-element T allocations. Random access is O(1); T
    values are pointer-stable across grow (only the pointer array
    moves). Parallel to `tablets[N]T` but with different shape:
    contiguous instead of chunk-chained, and one heap object per
    element instead of N-per-chunk."""
    element: "Ty"
    def __str__(self) -> str: return f"ivec<{self.element}>"

@dataclass(frozen=True)
class TyDVec:
    """`dvec<T>` — direct vector: contiguous heap-allocated array of
    T values inline. Random access is O(1) (one load). Grow moves T
    bytes, so wedges into individual elements are invalidated; for
    that reason push returns unit, unlike ivec/tablets which hand
    back stable handles."""
    element: "Ty"
    def __str__(self) -> str: return f"dvec<{self.element}>"

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

# Names that are universally visible regardless of module context.
# These are language-level: the built-in `str` tablet that the driver
# auto-prepends, plus the keywords-but-actually-names like `lost`. The
# primitive types (i64, bool, ...) are handled separately in
# `_resolve_type` because they aren't stored in `self.structs`.
BUILTIN_NAMES: set[str] = {"str"}


def _mangle_module_name(module: tuple[str, ...], short: str) -> str:
    """Compose a module-qualified flat name for the global symbol
    tables. Built-in names (universally visible) and decls in the
    root module are not mangled — they keep their short form so
    single-source compiles, the auto-prepended `str` tablet, and
    LLVM-level intrinsic / extern symbols all remain unambiguous.

    Module segments are joined with `__` so the resulting symbol is a
    valid C / LLVM identifier; the leading `__M_` distinguishes
    compiler-mangled names from user-chosen ones (which can't start
    with `__M_` because lex disallows the prefix? no — they can, but
    by convention the user-namespace doesn't collide here)."""
    if not module or short in BUILTIN_NAMES:
        return short
    return "__M_" + "__".join(module) + "__" + short


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
        # Type aliases — `type Bytes = buffer[1024]u8`. Stored as the
        # raw AST TypeExpr; resolved on demand inside `_resolve_type`
        # so the alias target can itself reference user structs / seals
        # / other aliases declared anywhere in the program.
        self.type_aliases: dict[str, A.TypeExpr] = {}
        # Seals (sum types) — registered alongside structs in phase 0.
        self.seals: dict[str, TySeal] = {}
        self.seal_type_params: dict[str, tuple[str, ...]] = {}
        # seal name → tuple of (variant_name, tuple of field Ty).
        # Order is source order so codegen can assign stable tag indices.
        self.seal_variants: dict[str, tuple[tuple[str, tuple[Ty, ...]], ...]] = {}
        # variant_name → list of (seal_name, variant_idx, declared_field_tys,
        # declaring_module). Multi-keyed because two seals in different
        # modules may declare the same variant name (the LIMITATIONS.md
        # flat-seal-variant gap is fixed via per-module disambiguation
        # at use sites). Within a single module two seals still can't
        # share a variant name.
        self.variant_lookup: dict[str, list[tuple[str, int, tuple[Ty, ...], tuple[str, ...]]]] = {}
        # mod → {variant_name: list[(seal_name, idx, field_tys, decl_module)]}.
        # Populated by `_build_module_visible_variants` after seal
        # variants are resolved. The list shape lets the use-site
        # resolver detect ambiguity when two imported seals each
        # contribute the same variant name into one module's scope.
        self.module_visible_variants: dict[
            tuple[str, ...],
            dict[str, list[tuple[str, int, tuple[Ty, ...], tuple[str, ...]]]],
        ] = {}
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
        # `ivec<T>` calls need T at codegen; the LLVM struct loses it.
        # Recorded at typecheck for any method/index/iter on an ivec.
        self.ivec_elem_at_call: dict[int, "Ty"] = {}     # Call → elem Ty
        self.ivec_elem_at_index: dict[int, "Ty"] = {}    # Index → elem Ty
        self.ivec_elem_at_for: dict[int, "Ty"] = {}      # ForStmt → elem Ty
        # Same shape for `dvec<T>` — different runtime layout but the
        # codegen still needs T at every call/index/iter site.
        self.dvec_elem_at_call: dict[int, "Ty"] = {}
        self.dvec_elem_at_index: dict[int, "Ty"] = {}
        self.dvec_elem_at_for: dict[int, "Ty"] = {}
        # Bindings whose resolved type is `wedge T`. Codegen consults
        # this set to spill the wedge value to a stack slot and push it
        # as a GC root with a trace fn that calls `mark_wedge` — without
        # this, a wedge held across allocations whose source ivec /
        # tablets has gone out of scope can be the only path to its
        # chunk, and the chunk is silently swept (the LLVM type is just
        # a plain pointer, so the chokepoint's `_type_desc_key` returns
        # None and no root push happens).
        self.wedge_bindings: set[int] = set()
        # Method-call dispatch: when a tablet exposes operations through
        # `edubba T<...> { fn ... }`, each method becomes a regular
        # mangled fn (`<TypeName>__<method>`) and lands in this registry
        # under its short name. `_tc_method_call` resolves a receiver's
        # type to its method table, picks the mangled fn name, and
        # records both the dispatch target on `method_dispatch_target`
        # and the mono args (if any) on `mono_call_args` so codegen can
        # emit the call exactly like a regular generic free fn call.
        self.tablet_methods: dict[str, dict[str, str]] = {}
        self.method_dispatch_target: dict[int, str] = {}  # Call → fn name
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
        # Set of fn parameter names in scope. Retained for diagnostic
        # phrasing in a few error messages; no longer load-bearing for
        # any escape rule (smart wedges trace through GC instead).
        self._fn_params: set[str] = set()

        # --- module support --------------------------------------------
        # Each top-level decl is tagged with its declaring module path
        # via `Program.module_of`. Phase 0 builds per-module tables and
        # all subsequent lookups go through them.

        # All module paths that contributed at least one decl. The root
        # module `()` is included implicitly so single-source compiles
        # still resolve the auto-prepended `str` tablet.
        self._known_modules: set[tuple[str, ...]] = {()}
        # mod -> {short_name: decl}. Decls declared IN that module,
        # by their parser-given short name. Includes `_`-prefixed
        # private decls (visibility filtering happens at use-site).
        self.module_decls: dict[tuple[str, ...], dict[str, A.Decl]] = {}
        # `id(decl) -> flat_name`. The global-symbol-table key for
        # each top-level decl. Mangled (`__M_mod__short`) when the
        # short name collides across modules; otherwise the short
        # form so existing single-module code paths stay untouched.
        # All `self.fns` / `self.structs` / etc. registrations and
        # lookups go through this — it's how cross-module same-name
        # decls coexist.
        self.flat_name_for: dict[int, str] = {}
        # mod -> {short_name: mangled_name}. The visible scope of each
        # module: own decls + imports + universally-visible builtins.
        # Lookups for top-level user names route through this table.
        # `mangled_name` is what's keyed in `self.fns` / `self.structs`
        # / `self.seals`. Builtins map to their unmangled form.
        self.module_visible: dict[tuple[str, ...], dict[str, str]] = {}
        # Set of Call-node ids whose callee was rewritten by the
        # module-qualified-access branch in `_tc_call`. The downstream
        # `_check_module_scope` skips these — the qualifier already
        # validated visibility against the source module's exports.
        self._qualified_call_resolved: set[int] = set()
        # mod -> {alias: source_module_path}. Populated from
        # `import x.y as z` (so `z.foo` becomes a module-qualified
        # reference into module x.y at use sites).
        self.module_aliases: dict[tuple[str, ...], dict[str, tuple[str, ...]]] = {}
        # mod -> set of source modules referenced by wildcard `import x`
        # (so `x.foo` works as a module-qualified reference). The
        # wildcard form also pulls every public name into the local
        # visible scope; this set just records that the prefix `x`
        # itself is also valid as a qualifier.
        self.module_qualified_refs: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
        # Updated at every top-level decl visit so name resolution
        # knows whose visible scope to consult. Body-checking phases
        # set this at the start of each fn / gloss; phase-0 already
        # sets it per-decl.
        self.current_module: tuple[str, ...] = ()

    def _warn(self, message: str, line: int = 0, col: int = 0) -> None:
        self.warnings.append(CompileWarning(message=message, line=line, col=col))

    def check(self) -> None:
        # Phase 0-modules: catalogue all module paths, validate imports,
        # build per-module visible scopes (own decls + imports +
        # universally-visible builtins). After this every later phase
        # can resolve top-level names via `_resolve_top_level` and
        # `_resolve_module_qualified`.
        self._build_module_scopes()
        # Phase 0-cycles: import graph must be acyclic. The strategy
        # doc explicitly lists circular imports as a non-goal for v1
        # (see `scratch/NEXT_BIG_FEATURE.md`); rejecting them now keeps
        # later passes' single-traversal model honest.
        self._reject_import_cycles()
        # Phase 0a: collect struct + seal names so type bodies can refer
        # to each other and to user types regardless of source order.
        # Aliases register their target AST eagerly; the target gets
        # resolved lazily on first use so it can name struct / seal /
        # alias siblings declared later in the file.
        for d in self.prog.decls:
            self.current_module = self.prog.module_of.get(id(d), ())
            if isinstance(d, A.StructDecl):
                self._register_struct_name(d)
            elif isinstance(d, A.SealDecl):
                self._register_seal_name(d)
            elif isinstance(d, A.AliasDecl):
                self._register_alias(d)
        # Phase 0b: resolve fields (structs) and variants (seals) now
        # that all user type names are in scope.
        for d in self.prog.decls:
            self.current_module = self.prog.module_of.get(id(d), ())
            if isinstance(d, A.StructDecl):
                self._resolve_struct_fields(d)
            elif isinstance(d, A.SealDecl):
                self._resolve_seal_variants(d)
        # Phase 0b': now that every seal's variants are resolved into
        # `self.variant_lookup`, build the per-module visible-variants
        # table. Use sites consult this to pick the right variant when
        # cross-module variant-name reuse exists.
        self._build_module_visible_variants()
        # Phase 0c: lower edubba blocks to flat FnDecls. By the time we
        # reach the fn-signature phase, methods need to look like regular
        # generic fns, so we splice them into `self.prog.decls` here and
        # populate the method registry. The host tablet must exist (we
        # validate against `self.structs` collected in 0a). The
        # EdubbaDecl is dropped — every later phase iterates the new
        # flat fns directly and never sees the wrapper.
        self._lower_edubbas()
        # Phase 1: function signatures (parameter and return types can now
        # reference any struct). Colophons declare externs and join the
        # same fn table so call sites resolve uniformly. Gloss decls
        # register in both the fn table (under a mangled name) and the
        # operator dispatch table.
        for d in self.prog.decls:
            self.current_module = self.prog.module_of.get(id(d), ())
            if isinstance(d, A.FnDecl):
                self._register_fn(d)
            elif isinstance(d, A.ColophonDecl):
                self._register_colophon(d)
            elif isinstance(d, A.GlossDecl):
                self._register_gloss(d)
                # Gloss decls don't expose a user-named top-level slot;
                # their mangled name is internal-only, so we skip the
                # visibility table for them.
        for d in self.prog.decls:
            self.current_module = self.prog.module_of.get(id(d), ())
            if isinstance(d, A.TableDecl):
                self._register_table(d)
        for d in self.prog.decls:
            self.current_module = self.prog.module_of.get(id(d), ())
            if isinstance(d, A.FnDecl):
                self._check_fn_body(d)
            elif isinstance(d, A.GlossDecl):
                self._check_gloss_body(d)
        # Reset to the root module after all decls have been visited so
        # any subsequent invocations (e.g. tests reusing the Checker
        # instance) start from a clean slate.
        self.current_module = ()

    def _build_module_scopes(self) -> None:
        """Build the per-module decl tables and visible scopes.

        Phase 0 of the checker. After this, every later phase can
        consult `self.module_visible[mod]` to translate a short name
        into the global mangled name keyed in `self.fns` /
        `self.structs` / `self.seals` / etc.

        Steps:

        1. Catalogue every (module, short_name) pair from the program's
           non-import top-level decls. This is `module_decls`.
        2. Validate import paths and selected names. Unknown modules,
           unknown names, and private (`_`-prefixed) names referenced
           in `from ... import` forms are rejected here.
        3. Build `module_visible` per module:
             - own decls (including private)
             - explicit imports (`from x import a [as b]`)
             - explicit wildcard imports (`import x.y` — every public
               name of x.y in scope; AND `y` registered as a qualifier
               for module-qualified access)
             - aliased imports (`import x.y as z` — only `z` registered
               as a qualifier; no flat-name pollution)
             - the universally-visible builtin `str`
        4. Record module-qualifier aliases in `module_aliases` and
           `module_qualified_refs` so module-qualified access (e.g.
           `parser.parse(x)` after `import parser`) can resolve.
        """
        # Step 1: catalogue decls per module. First pass: every decl
        # contributes its module path to `_known_modules` (so files that
        # contain only edubbas / imports / gloss still get a visible-
        # scope entry).
        for d in self.prog.decls:
            mod = self.prog.module_of.get(id(d), ())
            self._known_modules.add(mod)
        # Second pass: populate `module_decls` with the user-named
        # top-level decls. Skip imports (no decl name), gloss (mangled
        # internally), and edubba (lowered to `<Type>__<method>` fns
        # by phase 0c).
        for d in self.prog.decls:
            if isinstance(d, (A.ImportDecl, A.GlossDecl, A.EdubbaDecl)):
                continue
            name = getattr(d, "name", None)
            if not name:
                continue
            mod = self.prog.module_of.get(id(d), ())
            decls = self.module_decls.setdefault(mod, {})
            # Tolerate duplicates here — the per-decl-type registration
            # phases (`_register_fn`, `_register_struct_name`, etc.)
            # raise their own typed error messages that callers depend
            # on for diagnostics. We just record the first one we see
            # so module_visible has something to point at.
            decls.setdefault(name, d)

        # Compute the flat (global-symbol-table) name for every decl.
        # When two modules each declare the same short name, both
        # decls get module-prefix-mangled so they coexist in the global
        # tables. When a short name is unique across the program, no
        # mangling is needed and the short form is used as-is — keeps
        # existing single-module code paths untouched.
        short_count: dict[str, int] = {}
        for mod, decls in self.module_decls.items():
            for short in decls:
                short_count[short] = short_count.get(short, 0) + 1
        # Builtins are universally visible by their short name and
        # never get mangled (the auto-prepended `str` tablet has its
        # own global symbol).
        for b in BUILTIN_NAMES:
            short_count[b] = 1
        for mod, decls in self.module_decls.items():
            for short, decl in decls.items():
                # Colophons are externs — their C-ABI symbol is the
                # short name, period. Never mangled. (Two modules
                # declaring the same colophon name will collide at the
                # global fn table, which is the right behavior — only
                # one extern can claim a given C symbol.)
                if isinstance(decl, A.ColophonDecl):
                    self.flat_name_for[id(decl)] = short
                elif short in BUILTIN_NAMES or short_count.get(short, 0) <= 1:
                    self.flat_name_for[id(decl)] = short
                else:
                    self.flat_name_for[id(decl)] = _mangle_module_name(mod, short)

        # Initialize each known module's visible scope with its own
        # decls. The visible mapping is short → flat (global-symbol)
        # name. Private decls (`_`-prefixed) are visible only within
        # their declaring module — their entry lands here but not in
        # any other module's scope.
        for mod in self._known_modules:
            scope = {}
            for short, decl in self.module_decls.get(mod, {}).items():
                scope[short] = self.flat_name_for.get(id(decl), short)
            # Builtins always visible.
            for b in BUILTIN_NAMES:
                scope.setdefault(b, b)
            self.module_visible[mod] = scope
            self.module_aliases[mod] = {}
            self.module_qualified_refs[mod] = set()

        # Root-module ergonomic shortcut: when code lives outside `src/`
        # / `stdlib/` (single-source compiles, ad-hoc scripts in
        # `tmp_path`, the `examples/` directory), every public stdlib
        # decl is in scope unprefixed. Inside `src/` and inside
        # `stdlib/`, the shortcut is OFF — those modules must explicitly
        # `from stdlib.X import Y`. The split is between "script mode"
        # (root: ergonomics) and "project mode" (src/, stdlib/: import
        # discipline). The strategy doc didn't anticipate `<source>`-
        # style ad-hoc compiles, but the cwd-as-project-root convention
        # naturally accommodates both: you opt into project mode by
        # putting code under `src/`.
        if () in self.module_visible:
            for mod in list(self._known_modules):
                if mod and mod[0] == "stdlib":
                    for short, decl in self.module_decls.get(mod, {}).items():
                        if short.startswith("_"):
                            continue
                        self.module_visible[()].setdefault(
                            short, self.flat_name_for.get(id(decl), short),
                        )
                    # Also register the stdlib module's last segment as
                    # a module-qualifier in root, so `list.list_push`
                    # syntax (when qualified access lands) resolves.
                    self.module_aliases[()].setdefault(mod[-1], mod)
                    self.module_qualified_refs[()].add(mod)

        # Step 2 + 3 + 4: process imports.
        for d in self.prog.decls:
            if not isinstance(d, A.ImportDecl):
                continue
            importer = self.prog.module_of.get(id(d), ())
            src_mod = tuple(d.path)
            if src_mod not in self._known_modules:
                pretty = ".".join(d.path)
                raise CheckError(
                    f"unknown module {pretty!r} in import",
                    d.line, d.col,
                )
            src_decls = self.module_decls.get(src_mod, {})

            pretty = ".".join(d.path)
            if d.names is None:
                # Wildcard `import x.y` (or `import x.y as z`).
                if d.wildcard_alias is None:
                    # Bring every public name into local scope.
                    for short, decl in src_decls.items():
                        if short.startswith("_"):
                            continue
                        new_flat = self.flat_name_for.get(id(decl), short)
                        existing = self.module_visible[importer].get(short)
                        if existing is not None and existing != new_flat:
                            raise CheckError(
                                f"import of {short!r} from {pretty!r} "
                                f"conflicts with an existing name in "
                                f"this file",
                                d.line, d.col,
                            )
                        self.module_visible[importer][short] = new_flat
                    # The last segment of the path is also registered
                    # as a module-qualifier so `<seg>.foo` works at use
                    # sites.
                    last = src_mod[-1]
                    self.module_aliases[importer][last] = src_mod
                    self.module_qualified_refs[importer].add(src_mod)
                else:
                    # `import x.y as z` — alias-only. No flat-name
                    # pollution; access is exclusively via `z.foo`.
                    alias = d.wildcard_alias
                    if alias in self.module_aliases[importer]:
                        raise CheckError(
                            f"duplicate import alias {alias!r}",
                            d.line, d.col,
                        )
                    self.module_aliases[importer][alias] = src_mod
                    self.module_qualified_refs[importer].add(src_mod)
            else:
                # `from x.y import a, b as c` — selective.
                for src_name, alias in d.names:
                    if src_name.startswith("_"):
                        raise CheckError(
                            f"cannot import private name {src_name!r} "
                            f"from {pretty!r} (names beginning with "
                            f"'_' are private to their module)",
                            d.line, d.col,
                        )
                    if src_name not in src_decls:
                        raise CheckError(
                            f"module {pretty!r} has no export named "
                            f"{src_name!r}",
                            d.line, d.col,
                        )
                    local = alias or src_name
                    src_decl = src_decls[src_name]
                    new_flat = self.flat_name_for.get(id(src_decl), src_name)
                    existing = self.module_visible[importer].get(local)
                    if existing is not None and existing != new_flat:
                        raise CheckError(
                            f"import of {local!r} from {pretty!r} "
                            f"conflicts with an existing name in this "
                            f"file",
                            d.line, d.col,
                        )
                    self.module_visible[importer][local] = new_flat

    def _resolve_top_level(self, short_name: str) -> str | None:
        """Map a short name to its mangled global form via the current
        module's visible scope. Returns None when the name isn't
        visible — callers either fall back to other tables (locals,
        intrinsics, primitives) or raise."""
        scope = self.module_visible.get(self.current_module)
        if scope is None:
            return None
        return scope.get(short_name)

    def _resolve_module_qualified(
        self, qualifier: str, short_name: str,
    ) -> str | None:
        """Resolve `qualifier.short_name` against the current module's
        registered module aliases. Returns the mangled global name if
        `qualifier` matches an `import x.y` (last segment) or
        `import x.y as qualifier` entry AND `short_name` is a public
        decl of that source module. Returns None otherwise."""
        aliases = self.module_aliases.get(self.current_module, {})
        src_mod = aliases.get(qualifier)
        if src_mod is None:
            return None
        src_decls = self.module_decls.get(src_mod, {})
        if short_name not in src_decls:
            return None
        if short_name.startswith("_"):
            return None
        return _mangle_module_name(src_mod, short_name)

    def _resolve_qualified_type(
        self, dotted: str, line: int, col: int,
        type_args: "list[A.TypeExpr] | None" = None,
        where: str = "",
    ) -> "Ty | None":
        """Resolve a dotted type name like `qual.Map` (or `qual.Map<T>`
        when `type_args` is given) against the current module's import
        aliases. The qualifier is the leading segment; the rest is the
        short type name. Returns the resolved `Ty` or None if the
        qualifier isn't a registered import alias (caller falls
        through to the unknown-type error)."""
        head, _, tail = dotted.partition(".")
        if not tail:
            return None
        aliases = self.module_aliases.get(self.current_module, {})
        src_mod = aliases.get(head)
        if src_mod is None:
            return None
        decl = self.module_decls.get(src_mod, {}).get(tail)
        if decl is None:
            pretty = ".".join(src_mod) or "<root>"
            raise CheckError(
                f"module {pretty!r} has no public type {tail!r}",
                line, col,
            )
        if isinstance(decl, A.StructDecl):
            params = self.struct_type_params.get(tail, ())
            if type_args is None:
                if params:
                    raise CheckError(
                        f"{where or 'type'}: tablet {tail!r} expects "
                        f"{len(params)} type argument(s); write "
                        f"`{dotted}<...>`",
                        line, col,
                    )
                return self.structs[tail]
            if len(params) != len(type_args):
                raise CheckError(
                    f"{where or 'type'}: tablet {tail!r} expects "
                    f"{len(params)} type argument(s), got "
                    f"{len(type_args)}",
                    line, col,
                )
            resolved = tuple(
                self._resolve_type(a, f"{where or 'type'} type arg")
                for a in type_args
            )
            return TyStruct(name=tail, args=resolved)
        if isinstance(decl, A.SealDecl):
            params = self.seal_type_params.get(tail, ())
            if type_args is None:
                if params:
                    raise CheckError(
                        f"{where or 'type'}: seal {tail!r} expects "
                        f"{len(params)} type argument(s); write "
                        f"`{dotted}<...>`",
                        line, col,
                    )
                return self.seals[tail]
            if len(params) != len(type_args):
                raise CheckError(
                    f"{where or 'type'}: seal {tail!r} expects "
                    f"{len(params)} type argument(s), got "
                    f"{len(type_args)}",
                    line, col,
                )
            resolved = tuple(
                self._resolve_type(a, f"{where or 'type'} type arg")
                for a in type_args
            )
            return TySeal(name=tail, args=resolved)
        # decl exists but isn't a tablet/seal — type position requires
        # a type-bearing decl.
        raise CheckError(
            f"{where or 'type'}: {dotted!r} resolves to a "
            f"{type(decl).__name__}, not a tablet or seal",
            line, col,
        )

    def _reject_import_cycles(self) -> None:
        """Reject any cycle in the module import graph. The graph
        edge `A → B` means module A has an `import B` or `from B import
        ...` decl. Cycles are detected with a depth-first traversal.

        v1 doesn't need a topological order to typecheck — the existing
        flat-namespace pass already sees all decls — but rejecting
        cycles now keeps the door open for a real per-module-scope
        typecheck that walks modules in dependency order. See the
        strategy doc's explicit non-goal: 'No circular imports.
        Reject at link time.'"""
        edges: dict[tuple[str, ...], list[tuple[tuple[str, ...], int, int]]] = {}
        for d in self.prog.decls:
            if not isinstance(d, A.ImportDecl):
                continue
            src = self.prog.module_of.get(id(d), ())
            dst = tuple(d.path)
            edges.setdefault(src, []).append((dst, d.line, d.col))

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[tuple[str, ...], int] = {m: WHITE for m in self._known_modules}
        stack: list[tuple[str, ...]] = []

        def visit(node: tuple[str, ...]) -> None:
            if color.get(node, WHITE) == BLACK:
                return
            if color.get(node) == GRAY:
                # Cycle. Slice the stack from the recurrence point.
                idx = stack.index(node)
                cycle = stack[idx:] + [node]
                pretty = " -> ".join(".".join(m) or "<root>" for m in cycle)
                raise CheckError(
                    f"circular import detected: {pretty}",
                    0, 0,
                )
            color[node] = GRAY
            stack.append(node)
            for tgt, _line, _col in edges.get(node, []):
                visit(tgt)
            stack.pop()
            color[node] = BLACK

        for mod in list(self._known_modules):
            if color.get(mod, WHITE) == WHITE:
                visit(mod)

    def _build_module_visible_variants(self) -> None:
        """Populate `self.module_visible_variants[mod]` for every known
        module. The visible variants in module M are:
          - own seals' variants
          - variants of seals imported via `from X import SealName` or
            wildcard `import X` (where X has at least one seal)
          - (root only) variants of every public stdlib seal — matches
            the auto-import shortcut in `_build_module_scopes`

        Variant names that resolve to multiple distinct (seal, module)
        pairs in the same scope land here as the single registered
        entry; ambiguity is detected at use time when the user picks
        the name. Per-module same-name-twice is already rejected at
        registration in `_resolve_seal_variants`."""
        # Map: (seal_name, decl_mod) -> {variant_name: (seal, idx, fields, mod)}
        per_seal: dict[tuple[str, tuple[str, ...]], dict[str, tuple[str, int, tuple[Ty, ...], tuple[str, ...]]]] = {}
        for vname, entries in self.variant_lookup.items():
            for seal, idx, fields, decl_mod in entries:
                per_seal.setdefault((seal, decl_mod), {})[vname] = (
                    seal, idx, fields, decl_mod,
                )

        def _add(visible_map, vname, info):
            """Append `info` to visible_map[vname] unless an identical
            entry is already there (avoids spurious duplicates from
            wildcard + selective imports of the same seal)."""
            entries = visible_map.setdefault(vname, [])
            if info not in entries:
                entries.append(info)

        for mod in self._known_modules:
            visible: dict[str, list[tuple[str, int, tuple[Ty, ...], tuple[str, ...]]]] = {}
            # Own seals.
            for name, decl in self.module_decls.get(mod, {}).items():
                if isinstance(decl, A.SealDecl):
                    for vname, info in per_seal.get((name, mod), {}).items():
                        _add(visible, vname, info)
            self.module_visible_variants[mod] = visible

        # Process imports: bring in the variants of every imported seal.
        for d in self.prog.decls:
            if not isinstance(d, A.ImportDecl):
                continue
            importer = self.prog.module_of.get(id(d), ())
            src_mod = tuple(d.path)
            src_decls = self.module_decls.get(src_mod, {})
            if d.names is None and d.wildcard_alias is None:
                # `import x.y` (wildcard) — every public seal's variants.
                for sname, decl in src_decls.items():
                    if sname.startswith("_"):
                        continue
                    if not isinstance(decl, A.SealDecl):
                        continue
                    for vname, info in per_seal.get((sname, src_mod), {}).items():
                        _add(self.module_visible_variants[importer], vname, info)
            elif d.names is not None:
                # `from x import a, b as c` — only the named decls.
                for src_name, _alias in d.names:
                    decl = src_decls.get(src_name)
                    if not isinstance(decl, A.SealDecl):
                        continue
                    for vname, info in per_seal.get((src_name, src_mod), {}).items():
                        _add(self.module_visible_variants[importer], vname, info)

        # Root-mode auto-stdlib: bring every public stdlib seal's variants
        # into root's visible variants. Mirrors the auto-import
        # shortcut for top-level decls in `_build_module_scopes`.
        for mod in list(self._known_modules):
            if mod and mod[0] == "stdlib":
                for sname, decl in self.module_decls.get(mod, {}).items():
                    if sname.startswith("_"):
                        continue
                    if not isinstance(decl, A.SealDecl):
                        continue
                    for vname, info in per_seal.get((sname, mod), {}).items():
                        _add(self.module_visible_variants.setdefault((), {}), vname, info)

    def _resolve_variant(
        self, name: str, line: int, col: int,
    ) -> tuple[str, int, tuple[Ty, ...], tuple[str, ...]] | None:
        """Look up `name` in the current module's visible variants.

        Outcomes:
          - exactly one entry → returns (seal, idx, fields, decl_mod)
          - zero entries, registered elsewhere → CheckError telling the
            user to import the seal that contains it
          - zero entries, not registered → returns None (caller continues
            to "undefined name" path)
          - more than one entry (ambiguous) → CheckError listing the
            candidate seals so the user can be specific
        """
        visible = self.module_visible_variants.get(self.current_module, {}).get(name)
        if visible:
            if len(visible) == 1:
                return visible[0]
            # Ambiguity: more than one imported seal contributes a
            # variant by this name. Tell the user which seals collide.
            choices = ", ".join(
                f"seal {seal!r} in module {('.'.join(mod) or '<root>')!r}"
                for seal, _idx, _f, mod in visible
            )
            raise CheckError(
                f"variant {name!r} is declared in multiple seals; "
                f"import the specific seal you mean: {choices}",
                line, col,
            )
        # Not visible. Is it declared anywhere?
        registered = self.variant_lookup.get(name, [])
        if not registered:
            return None
        # Variant exists but isn't in scope. Point the user at one of
        # the seals that contains it.
        seals = sorted({(seal, mod) for seal, _idx, _f, mod in registered})
        if len(seals) == 1:
            seal, mod = seals[0]
            pretty = ".".join(mod) or "<root>"
            raise CheckError(
                f"variant {name!r} (of seal {seal!r}) is declared in "
                f"module {pretty!r} but the seal isn't in scope here; "
                f"add `from {pretty} import {seal}` (or `import "
                f"{pretty}`) at the top of this file",
                line, col,
            )
        # Multiple seals declare it elsewhere — list them.
        choices = ", ".join(
            f"seal {seal!r} in module {('.'.join(mod) or '<root>')!r}"
            for seal, mod in seals
        )
        raise CheckError(
            f"variant {name!r} is declared in multiple seals; import "
            f"the specific seal you mean: {choices}",
            line, col,
        )

    def _check_module_scope(self, name: str, line: int, col: int) -> None:
        """Enforce per-module visibility for top-level user-defined
        names. The name must be in the current module's visible scope
        (own decls + imports + builtins), or be a non-top-level
        thing the caller will handle (a not-yet-bound local, or the
        `not declared anywhere` case). Raises with a targeted message
        if the name is declared in some other module but not imported
        here; otherwise returns silently and lets the caller continue
        its existing not-found path."""
        scope = self.module_visible.get(self.current_module, {})
        if name in scope:
            return
        # Find where this name is declared, if anywhere. If it lives in
        # a different module, the user almost certainly forgot to
        # import it.
        for mod, decls in self.module_decls.items():
            if name in decls:
                if mod == self.current_module:
                    # Decl exists in this module but isn't visible —
                    # only happens if `_build_module_scopes` skipped
                    # it (shouldn't); fall through to the generic path.
                    return
                pretty = ".".join(mod) or "<root>"
                if name.startswith("_"):
                    raise CheckError(
                        f"name {name!r} is private to module {pretty!r} "
                        f"and cannot be imported",
                        line, col,
                    )
                here = ".".join(self.current_module) or "<root>"
                raise CheckError(
                    f"name {name!r} is declared in module {pretty!r} "
                    f"but not in scope here ({here!r}); add `from "
                    f"{pretty} import {name}` (or `import {pretty}`) at "
                    f"the top of this file",
                    line, col,
                )

    # --- registration ------------------------------------------------

    def _register_alias(self, a: A.AliasDecl) -> None:
        if a.name in PRIM_TYPES:
            raise CheckError(
                f"type alias {a.name!r}: name shadows a built-in type",
                a.line, a.col,
            )
        flat = self._flat_name(a)
        if flat in self.structs or flat in self.seals:
            raise CheckError(
                f"type alias {a.name!r}: name collides with an existing "
                f"tablet or seal",
                a.line, a.col,
            )
        if flat in self.type_aliases:
            raise CheckError(
                f"duplicate type alias {a.name!r}", a.line, a.col,
            )
        self.type_aliases[flat] = a.target

    def _register_struct_name(self, s: A.StructDecl) -> None:
        if s.name in PRIM_TYPES:
            raise CheckError(
                f"tablet {s.name!r}: name shadows a built-in type", s.line, s.col,
            )
        flat = self.flat_name_for.get(id(s), s.name)
        if flat in self.structs:
            raise CheckError(
                f"duplicate tablet {s.name!r}", s.line, s.col,
            )
        if flat in self.type_aliases:
            raise CheckError(
                f"tablet {s.name!r}: name collides with an existing "
                f"type alias",
                s.line, s.col,
            )
        self.structs[flat] = TyStruct(name=flat)
        self.struct_type_params[flat] = tuple(s.type_params)

    def _lower_edubbas(self) -> None:
        """Splice each `edubba T<...> { fn ... }` block's methods into
        the program's top-level decl list as flat FnDecls (with type
        params copied from the host edubba) and register them in
        `self.tablet_methods`. The EdubbaDecl wrapper is dropped — by
        the time later phases iterate `self.prog.decls`, methods look
        like ordinary generic free fns. Validation here:
        - Host tablet must exist (registered in phase 0a).
        - Edubba arity matches the tablet's type-param count.
        - No two methods on the same tablet share a name.
        - `mut self` is only allowed when the host tablet is itself
          a struct codegen lowers to a mut-pointer parameter. (All
          tablets currently qualify; the check is here so a future
          built-in-only tablet shape gets a clear error.)
        """
        new_decls: list[A.Decl] = []
        for d in self.prog.decls:
            if not isinstance(d, A.EdubbaDecl):
                new_decls.append(d)
                continue
            if d.type_name not in self.structs:
                raise CheckError(
                    f"edubba {d.type_name!r}: no tablet by that name",
                    d.line, d.col,
                )
            # An edubba block can only extend a tablet declared in the
            # same module — outside modules can't bolt methods onto
            # someone else's tablet (that's both a module-pollution
            # vector and an evolution-trap for the host module's
            # invariants). Multiple edubba blocks for the same tablet
            # within one module are fine and additive — the existing
            # method registry below catches duplicate method names.
            edubba_mod = self.prog.module_of.get(id(d), ())
            host_mod = None
            for mod, decls in self.module_decls.items():
                if d.type_name in decls and isinstance(
                    decls[d.type_name], A.StructDecl,
                ):
                    host_mod = mod
                    break
            if host_mod is not None and host_mod != edubba_mod:
                here = ".".join(edubba_mod) or "<root>"
                there = ".".join(host_mod) or "<root>"
                raise CheckError(
                    f"edubba {d.type_name!r}: tablet is declared in "
                    f"module {there!r}, but this edubba block lives in "
                    f"{here!r}; methods can only be added in the host "
                    f"tablet's own module",
                    d.line, d.col,
                )
            host_params = self.struct_type_params.get(d.type_name, ())
            if len(d.type_params) != len(host_params):
                raise CheckError(
                    f"edubba {d.type_name}: type-param arity {len(d.type_params)} "
                    f"does not match tablet arity {len(host_params)}",
                    d.line, d.col,
                )
            methods = self.tablet_methods.setdefault(d.type_name, {})
            for m in d.methods:
                # The parser mangled the method name as `<Type>__<name>`.
                # Recover the short name for the registry.
                if not m.name.startswith(f"{d.type_name}__"):
                    raise CheckError(
                        f"edubba {d.type_name}: malformed method name "
                        f"{m.name!r} (compiler bug — parser should have "
                        f"mangled this)",
                        m.line, m.col,
                    )
                short = m.name[len(d.type_name) + 2:]
                if short in methods:
                    raise CheckError(
                        f"edubba {d.type_name}: duplicate method "
                        f"{short!r}",
                        m.line, m.col,
                    )
                m.type_params = list(d.type_params)
                methods[short] = m.name
                new_decls.append(m)
        self.prog.decls[:] = new_decls

    def _register_seal_name(self, s: A.SealDecl) -> None:
        if s.name in PRIM_TYPES:
            raise CheckError(
                f"seal {s.name!r}: name shadows a built-in type", s.line, s.col,
            )
        flat = self._flat_name(s)
        if flat in self.structs:
            raise CheckError(
                f"seal {s.name!r}: name collides with an existing tablet",
                s.line, s.col,
            )
        if flat in self.seals:
            raise CheckError(
                f"duplicate seal {s.name!r}", s.line, s.col,
            )
        self.seals[flat] = TySeal(name=flat)
        self.seal_type_params[flat] = tuple(s.type_params)

    def _resolve_seal_variants(self, s: A.SealDecl) -> None:
        seen_names: set[str] = set()
        resolved_variants: list[tuple[str, tuple[Ty, ...]]] = []
        saved = self._active_type_vars
        self._active_type_vars = {name: TyVar(name) for name in s.type_params}
        decl_mod = self.prog.module_of.get(id(s), ())
        flat = self._flat_name(s)
        try:
            for idx, v in enumerate(s.variants):
                if v.name in seen_names:
                    raise CheckError(
                        f"seal {s.name!r}: duplicate variant {v.name!r}",
                        v.line, v.col,
                    )
                seen_names.add(v.name)
                # Within a single module two seals can't share a variant
                # name (one flat namespace per module). Across modules
                # is fine — disambiguation happens at use site via the
                # importer's visible-variants table.
                existing = self.variant_lookup.get(v.name, [])
                same_mod = [e for e in existing if e[3] == decl_mod]
                if same_mod:
                    prev_seal = same_mod[0][0]
                    raise CheckError(
                        f"variant {v.name!r} is already declared in seal "
                        f"{prev_seal!r} in this module",
                        v.line, v.col,
                    )
                field_tys = tuple(
                    self._resolve_type(
                        ft, f"field of variant {v.name!r} of seal {s.name!r}",
                    )
                    for ft in v.fields
                )
                resolved_variants.append((v.name, field_tys))
                self.variant_lookup.setdefault(v.name, []).append(
                    (s.name, idx, field_tys, decl_mod),
                )
        finally:
            self._active_type_vars = saved
        self.seal_variants[flat] = tuple(resolved_variants)

    def _flat_name(self, decl: A.Decl) -> str:
        """Convenience: return the flat (global-table) name for a decl,
        defaulting to its short name if no module-prefix mangle is
        needed. Always-visible callers should prefer this over
        manually consulting `flat_name_for`."""
        return self.flat_name_for.get(id(decl), decl.name)

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
        self.struct_fields[self._flat_name(s)] = tuple(resolved)

    def _register_fn(self, fn: A.FnDecl) -> None:
        if fn.name in INTRINSIC_NAMES:
            raise CheckError(
                f"cannot define {fn.name!r}: it is a built-in intrinsic",
                fn.line, fn.col,
            )
        # Generic fns: type parameters are in scope as TyVars while we
        # resolve the signature and (later) check the body. Duplicate
        # detection moved below to use the flat name (so cross-module
        # same-name fns coexist via mangling).
        flat = self._flat_name(fn)
        self.fn_type_params[flat] = tuple(fn.type_params)
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
        if flat in self.fns:
            raise CheckError(
                f"duplicate function {fn.name!r}", fn.line, fn.col,
            )
        self.fns[flat] = TyFn(
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
        if self._flat_name(c) in self.fns:
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
            # rejected for the same reason.
            if isinstance(ty, TyFn):
                if all(is_primitive(p) for p in ty.params) and is_primitive(ty.ret):
                    return
                raise CheckError(
                    f"colophon {c.name!r}: {where} has type {ty}; "
                    f"callback signatures must be primitives-only (int / "
                    f"bool / unit) — str / struct / wedge / nested fn "
                    f"aren't marshalable across a C-invoked callback",
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
        flat = self._flat_name(c)
        self.fns[flat] = TyFn(
            params=params, ret=ret,
            param_muts=tuple(p.is_mut for p in c.params),
        )
        # Tracked so codegen can emit extern declarations instead of
        # trying to lower a body.
        self.colophons.add(self._flat_name(c))

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
        self.tables[self._flat_name(t)] = TyTable(element=elem)
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
        self.current_fn = self._flat_name(fn)
        self.scopes = [{}]
        self._fn_params = {p.name for p in fn.params}
        # Type params are in scope while we check the body so local
        # bindings with annotations like `mut cur: wedge Node<T>` work.
        saved = self._active_type_vars
        self._active_type_vars = {name: TyVar(name) for name in fn.type_params}
        fn_ty = self.fns[self.current_fn]
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

    # --- type resolution --------------------------------------------

    def _resolve_type(self, t: A.TypeExpr, where: str) -> Ty:
        if isinstance(t, A.TypeName):
            if t.name in PRIM_TYPES:
                return PRIM_TYPES[t.name]
            # Inside a generic decl body, the type-parameter names are
            # in scope as fresh type variables.
            if t.name in self._active_type_vars:
                return self._active_type_vars[t.name]
            # Module-qualified type: `qual.Tablet`. Resolves through
            # the module_aliases entry the import set up.
            if "." in t.name:
                resolved = self._resolve_qualified_type(t.name, t.line, t.col)
                if resolved is not None:
                    return resolved
            # Translate short name to flat (mangled) form via the
            # current module's visible scope. Falls back to short for
            # cases where module scoping isn't engaged (e.g. type-aliases
            # whose short name isn't in any module table).
            flat = self.module_visible.get(self.current_module, {}).get(t.name, t.name)
            if flat in self.type_aliases:
                # Aliases are transparent: resolve the target in the
                # alias's stead. Cycles surface as RecursionError —
                # cheap detection but the user gets a stack trace
                # rather than a clean diagnostic. Improve later.
                return self._resolve_type(self.type_aliases[flat], where)
            if flat in self.structs:
                # Using a generic tablet's name without type args is
                # only valid if the tablet is non-generic.
                params = self.struct_type_params.get(flat, ())
                if params:
                    raise CheckError(
                        f"{where}: tablet {t.name!r} expects "
                        f"{len(params)} type argument(s): "
                        f"write `{t.name}<...>`",
                        t.line, t.col,
                    )
                self._check_module_scope(t.name, t.line, t.col)
                return self.structs[flat]
            if flat in self.seals:
                params = self.seal_type_params.get(flat, ())
                if params:
                    raise CheckError(
                        f"{where}: seal {t.name!r} expects "
                        f"{len(params)} type argument(s): "
                        f"write `{t.name}<...>`",
                        t.line, t.col,
                    )
                self._check_module_scope(t.name, t.line, t.col)
                return self.seals[flat]
            raise CheckError(
                f"{where}: unknown type {t.name!r}", t.line, t.col,
            )
        if isinstance(t, A.TypeApply):
            # Module-qualified generic application: `qual.Map<T>`.
            if "." in t.name:
                resolved = self._resolve_qualified_type(
                    t.name, t.line, t.col, type_args=t.args, where=where,
                )
                if resolved is not None:
                    return resolved
            flat = self.module_visible.get(self.current_module, {}).get(t.name, t.name)
            if flat in self.structs:
                params = self.struct_type_params.get(flat, ())
                if len(params) != len(t.args):
                    raise CheckError(
                        f"{where}: tablet {t.name!r} expects "
                        f"{len(params)} type argument(s), got {len(t.args)}",
                        t.line, t.col,
                    )
                self._check_module_scope(t.name, t.line, t.col)
                resolved_args = tuple(
                    self._resolve_type(a, f"{where} type arg") for a in t.args
                )
                return TyStruct(name=flat, args=resolved_args)
            if flat in self.seals:
                params = self.seal_type_params.get(flat, ())
                if len(params) != len(t.args):
                    raise CheckError(
                        f"{where}: seal {t.name!r} expects "
                        f"{len(params)} type argument(s), got {len(t.args)}",
                        t.line, t.col,
                    )
                self._check_module_scope(t.name, t.line, t.col)
                resolved_args = tuple(
                    self._resolve_type(a, f"{where} type arg") for a in t.args
                )
                return TySeal(name=flat, args=resolved_args)
            raise CheckError(
                f"{where}: unknown type {t.name!r}",
                t.line, t.col,
            )
        if isinstance(t, A.TypeTablets):
            inner = self._resolve_type(t.element, f"{where} element")
            return TyTablets(size=t.size, element=inner)
        if isinstance(t, A.TypeBuffer):
            inner = self._resolve_type(t.element, f"{where} element")
            # Buffers carry only u8 today. Lifting the restriction needs
            # an ownership story for struct / heap-bearing element types
            # inside a stack-lifetime container — we haven't decided
            # whether to copy on overwrite, forbid overwrite, or treat
            # the buffer as a borrow window into something else.
            if not (isinstance(inner, TyInt) and inner.width == 8 and not inner.signed):
                raise CheckError(
                    f"{where}: buffer element must be u8, got {inner}",
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
                f"{where}: bare array types are not supported — use "
                f"`tablets[N]T` for a growable arena or `buffer[N]u8` "
                f"for a stack-lifetime byte buffer",
                t.line, t.col,
            )
        if isinstance(t, A.TypePointer):
            inner = self._resolve_type(t.element, f"{where} element")
            return TyPtr(element=inner)
        if isinstance(t, A.TypeHandle):
            inner = self._resolve_type(t.element, f"{where} element")
            return TyHandle(element=inner)
        if isinstance(t, A.TypeIVec):
            inner = self._resolve_type(t.element, f"{where} element")
            return TyIVec(element=inner)
        if isinstance(t, A.TypeDVec):
            inner = self._resolve_type(t.element, f"{where} element")
            return TyDVec(element=inner)
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

    def _pop(self) -> None:
        self.scopes.pop()

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
        # Variant-not-in-scope check. If `name` is a registered variant
        # but the seal that contains it isn't imported, give a targeted
        # "import the seal" error before the generic "undefined name".
        if name in self.variant_lookup:
            self._resolve_variant(name, line, col)
        # Top-level visibility: the name must be in the current
        # module's visible scope (own decls + imports + builtins).
        # If `name` is a top-level decl declared elsewhere but not
        # imported, raise a clear "not in scope" error rather than
        # falling through to the generic "undefined name" — directs
        # users to add an import.
        self._check_module_scope(name, line, col)
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
            self._maybe_mark_wedge_binding(b, declared)
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
            self._maybe_mark_wedge_binding(b, declared)
        else:
            self._bind(b.name, init_ty, b.line, b.col)
            self._maybe_mark_wedge_binding(b, init_ty)

    def _maybe_mark_wedge_binding(self, b: A.Binding, ty: Ty) -> None:
        """If the binding holds a `wedge T` value, record it so codegen
        emits the GC-root spill. The LLVM lowering loses wedge-ness
        (just a pointer), so we have to surface it from here."""
        if isinstance(ty, TyHandle):
            self.wedge_bindings.add(id(b))

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
        if isinstance(iter_ty, TyIVec):
            self.ivec_elem_at_for[id(f)] = iter_ty.element
            return iter_ty.element
        if isinstance(iter_ty, TyDVec):
            self.dvec_elem_at_for[id(f)] = iter_ty.element
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
            # current module's visible variants first so `None` /
            # `Circle` etc. resolve to their seal type rather than
            # "undefined name". Cross-module variant-name reuse is
            # disambiguated here too.
            if e.name in self.module_visible_variants.get(self.current_module, {}):
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
        if isinstance(ty, TyIVec):
            return self._occurs_in(var_name, ty.element)
        if isinstance(ty, TyDVec):
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
        if isinstance(ty, TyIVec):
            inner = self._substitute(ty.element, subst)
            return TyIVec(element=inner)
        if isinstance(ty, TyDVec):
            inner = self._substitute(ty.element, subst)
            return TyDVec(element=inner)
        if isinstance(ty, TyStruct) and ty.args:
            new_args = tuple(self._substitute(a, subst) for a in ty.args)
            return TyStruct(name=ty.name, args=new_args)
        if isinstance(ty, TySeal) and ty.args:
            new_args = tuple(self._substitute(a, subst) for a in ty.args)
            return TySeal(name=ty.name, args=new_args)
        return ty

    def _tc_call(self, e: A.Call, expected: Ty | None = None) -> Ty:
        # Module-qualified call: `parser.parse(x)` after `import parser`
        # (or `import x.y.parser as parser`). Dispatches to the resolved
        # public fn in the source module, bypassing the method-call path
        # below. Detected before the Field-as-method case because module
        # qualifiers shadow the same-name local-binding rule (a local
        # named `parser` shadows the import; method dispatch then fires).
        if (
            isinstance(e.callee, A.Field)
            and isinstance(e.callee.target, A.Ident)
            and not any(e.callee.target.name in s for s in self.scopes)
        ):
            qualifier = e.callee.target.name
            method = e.callee.name
            aliases = self.module_aliases.get(self.current_module, {})
            if qualifier in aliases:
                src_mod = aliases[qualifier]
                src_decls = self.module_decls.get(src_mod, {})
                target = src_decls.get(method)
                if target is None or method.startswith("_"):
                    pretty = ".".join(src_mod) or "<root>"
                    raise CheckError(
                        f"module {pretty!r} has no public name "
                        f"{method!r}",
                        e.line, e.col,
                    )
                # Dispatch as a regular call by substituting the callee
                # with a bare Ident pointing at the resolved name (which
                # is in the global flat fn table because cross-module
                # mangling isn't on yet — that's the next commit). The
                # `_qualified_call_resolved` flag tells
                # `_check_module_scope` to trust the qualifier-side
                # check we just did instead of re-checking visibility
                # against the importer's local scope (where the
                # method name doesn't appear, by design of `import x as y`).
                new_ident = A.Ident(name=method)
                new_ident.line = e.callee.line
                new_ident.col = e.callee.col
                e.callee = new_ident
                self._qualified_call_resolved.add(id(e))
                # Drop through to the regular fn-call path below.

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
        # Resolved via current module's visible variants.
        if name in self.module_visible_variants.get(self.current_module, {}):
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

        # Translate the short name through the current module's
        # visible scope to find the mangled flat key in self.fns.
        # When no collision exists across modules, mangled == short
        # and this is a no-op. When two modules each declare the
        # same fn name, this picks the right one for the caller.
        flat = self.module_visible.get(self.current_module, {}).get(name, name)
        fn = self.fns.get(flat)
        if fn is None:
            raise CheckError(
                f"in fn {self.current_fn!r}: unknown function {name!r}"
                f"{_suggest(name, self.fns)}",
                e.line, e.col,
            )
        # Rewrite the callee to the flat name so codegen sees the
        # exact LLVM symbol to dispatch to.
        if flat != name:
            e.callee.name = flat
        # Skip the local-scope visibility check when the call was
        # dispatched via module-qualified access — qualifier resolution
        # already validated the source module's exports.
        if id(e) not in self._qualified_call_resolved:
            self._check_module_scope(name, e.line, e.col)
        name = flat
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
                # Freeze-while-borrow: `a.push(x)` does NOT invalidate
                # existing borrows into `a`. Tuppu's tablets is a
                # chunk-chain — push appends (allocating a new chunk
                # if needed) but never relocates existing slots, so
                # wedges and element borrows into earlier slots stay
                # valid. Writes via `a[i] = x` (handled in _tc_assign)
                # ARE mut-reaches since they can free a slot's old
                # contents.
                # push returns a handle to the newly-pushed element —
                # discarded automatically in statement position.
                return TyHandle(element=recv_ty.element)
            raise CheckError(
                f"tablets has no method {method!r}", e.line, e.col,
            )

        if isinstance(recv_ty, TyIVec):
            if method == "push":
                if len(e.args) != 1:
                    raise CheckError(
                        "ivec.push takes one argument", e.line, e.col,
                    )
                at = self._tc_expr(e.args[0])
                if not _coerces_to(at, recv_ty.element):
                    raise CheckError(
                        f"ivec.push: value has type {at}, "
                        f"element type is {recv_ty.element}",
                        e.line, e.col,
                    )
                # Record elem-ty for codegen — IVEC_STRUCT loses T at
                # the LLVM level, so we forward it via sideband.
                self.ivec_elem_at_call[id(e)] = recv_ty.element
                # ivec.push allocates a fresh per-element T on the heap
                # and stores its pointer in the array. Existing element
                # heap addresses don't move; only the pointer array
                # might realloc. So push returns nothing useful — unlike
                # tablets, where the returned wedge points at a stable
                # arena slot, ivec elements are independently allocated
                # and the stable address is the returned wedge into the
                # heap-allocated T. Return a wedge for the same UX.
                return TyHandle(element=recv_ty.element)
            raise CheckError(
                f"ivec has no method {method!r}", e.line, e.col,
            )

        if isinstance(recv_ty, TyDVec):
            if method == "push":
                if len(e.args) != 1:
                    raise CheckError(
                        "dvec.push takes one argument", e.line, e.col,
                    )
                at = self._tc_expr(e.args[0])
                if not _coerces_to(at, recv_ty.element):
                    raise CheckError(
                        f"dvec.push: value has type {at}, "
                        f"element type is {recv_ty.element}",
                        e.line, e.col,
                    )
                self.dvec_elem_at_call[id(e)] = recv_ty.element
                # Unlike ivec / tablets, dvec.push returns unit. Inline
                # T storage means a grow can memcpy elements to a new
                # buffer; handing back a wedge to the just-pushed slot
                # would dangle on the very next push. Users index into
                # `dv[i]` for read access; that path is freshly
                # bounds-checked + dereferenced each time.
                return UNIT
            raise CheckError(
                f"dvec has no method {method!r}", e.line, e.col,
            )

        # Method dispatch on a tablet receiver — `m.set(k, v)` resolves
        # through `tablet_methods`, populated when the tablet's
        # `edubba T { ... }` block was lowered. Receiver becomes the
        # first arg; for mut methods codegen passes the lvalue pointer
        # so receivers can be Field/Index paths without forcing the
        # caller to bind to a mut Ident.
        if (
            isinstance(recv_ty, TyStruct)
            and recv_ty.name in self.tablet_methods
        ):
            methods = self.tablet_methods[recv_ty.name]
            if method in methods:
                return self._tc_struct_method_dispatch(
                    e, recv_ty, methods[method],
                )
            if methods:
                available = ", ".join(sorted(methods))
                raise CheckError(
                    f"{recv_ty.name} has no method {method!r}; "
                    f"available: {available}",
                    e.line, e.col,
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

    def _tc_struct_method_dispatch(
        self, e: A.Call, recv_ty: "TyStruct", fn_name: str,
    ) -> Ty:
        """Route a method call on a stdlib tablet to its underlying free
        function. The receiver was already typechecked by `_tc_method_call`
        — we re-run the standard generic-fn unification flow here against
        the synthetic arg list `[receiver, *e.args]`, pinning the
        type-parameter substitution from the receiver's type up front so
        callers don't need to spell `T` explicitly.

        Records `mono_call_args[id(e)]` (when the underlying fn is generic)
        and `method_dispatch_target[id(e)]` so codegen can find the target
        and decide whether the receiver gets passed by pointer or value."""
        fn = self.fns.get(fn_name)
        if fn is None:
            raise CheckError(
                f"method dispatch on {recv_ty}: stdlib fn {fn_name!r} "
                f"is missing — make sure the matching stdlib file is "
                f"loaded",
                e.line, e.col,
            )
        if len(fn.params) != len(e.args) + 1:
            raise CheckError(
                f"{fn_name} expects {len(fn.params) - 1} arg(s) after "
                f"the receiver, got {len(e.args)}",
                e.line, e.col,
            )
        type_params = self.fn_type_params.get(fn_name, ())
        inst_names = [f"{fn_name}.{tp}#{id(e)}" for tp in type_params]
        inst_subst = {
            tp: TyVar(fresh)
            for tp, fresh in zip(type_params, inst_names)
        }
        subst: dict[str, Ty] = {}
        # Pin T from the receiver type before typechecking explicit args
        # so their hints are concrete (e.g. `Map<JValue>.set(k, v)` knows
        # v should be a JValue, not an open TyVar).
        receiver_param = self._substitute(fn.params[0], inst_subst)
        try:
            self._unify(receiver_param, recv_ty, subst)
        except _UnifyError as ex:
            raise CheckError(
                f"method dispatch: receiver has type {recv_ty}, but "
                f"{fn_name}'s first param expects {fn.params[0]}"
                f"{f' ({ex.detail})' if ex.detail else ''}",
                e.line, e.col,
            ) from None
        arg_hints: list[Ty | None] = []
        for pty in fn.params[1:]:
            h = self._substitute(self._substitute(pty, inst_subst), subst)
            arg_hints.append(
                h if not self._has_open_tyvar(h, inst_names) else None
            )
        arg_tys = [
            self._tc_expr(a, expected=h)
            for a, h in zip(e.args, arg_hints)
        ]
        for i, (at, pty) in enumerate(zip(arg_tys, fn.params[1:]), start=1):
            freshened = self._substitute(pty, inst_subst)
            try:
                self._unify(freshened, at, subst)
            except _UnifyError as ex:
                raise CheckError(
                    f"call to {fn_name!r}: arg {i} has type {at}, "
                    f"expected {pty}"
                    f"{f' ({ex.detail})' if ex.detail else ''}",
                    e.line, e.col,
                ) from None
        for i, (at, pty) in enumerate(zip(arg_tys, fn.params[1:]), start=1):
            inst = self._substitute(self._substitute(pty, inst_subst), subst)
            if not _coerces_to(at, inst):
                raise CheckError(
                    f"call to {fn_name!r}: arg {i} has type {at}, "
                    f"expected {inst}",
                    e.line, e.col,
                )
        ret = self._substitute(self._substitute(fn.ret, inst_subst), subst)
        if type_params:
            missing = [
                tp for tp, fresh in zip(type_params, inst_names)
                if fresh not in subst
            ]
            if missing:
                raise CheckError(
                    f"call to {fn_name!r}: could not infer type "
                    f"parameter(s) {', '.join(missing)} from receiver "
                    f"and argument types",
                    e.line, e.col,
                )
            self.mono_call_args[id(e)] = tuple(
                self._substitute(subst[fresh], subst)
                for fresh in inst_names
            )
        self.method_dispatch_target[id(e)] = fn_name
        return ret

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
        if isinstance(target_ty, TyIVec):
            if e.name == "len":
                return I64
            raise CheckError(
                f"ivec has no field {e.name!r}; only len",
                e.line, e.col,
            )
        if isinstance(target_ty, TyDVec):
            if e.name == "len":
                return I64
            raise CheckError(
                f"dvec has no field {e.name!r}; only len",
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
        flat = self.module_visible.get(self.current_module, {}).get(e.name, e.name)
        if flat not in self.structs:
            raise CheckError(
                f"unknown tablet {e.name!r}"
                f"{_suggest(e.name, self.structs)}",
                e.line, e.col,
            )
        declared = self.struct_fields[flat]
        type_params = self.struct_type_params.get(flat, ())
        # Instantiate fresh type variables for the tablet's type params
        # so unification doesn't alias them to identically-named TyVars
        # from the enclosing fn's scope. `Node.T#<id(e)>` is unique
        # per literal, so inferred bindings stay scoped to this site.
        fresh_names = [f"{flat}.{tp}#{id(e)}" for tp in type_params]
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
            return TyStruct(name=flat, args=concrete_args)
        return self.structs[flat]

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
        if isinstance(target_ty, TyIVec):
            idx_ty = self._tc_expr(e.index)
            if not _is_int(idx_ty):
                raise CheckError(
                    f"ivec index must be integer, got {idx_ty}",
                    e.line, e.col,
                )
            self.ivec_elem_at_index[id(e)] = target_ty.element
            return target_ty.element
        if isinstance(target_ty, TyDVec):
            idx_ty = self._tc_expr(e.index)
            if not _is_int(idx_ty):
                raise CheckError(
                    f"dvec index must be integer, got {idx_ty}",
                    e.line, e.col,
                )
            self.dvec_elem_at_index[id(e)] = target_ty.element
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
        on scalars / handles / fn-pointers — so `copy` on a plain int
        is harmless redundancy rather than an error. A future lint
        could warn on those redundant copies; the type checker stays
        permissive for now."""
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
        info = self._resolve_variant(e.name, e.line, e.col)
        assert info is not None  # caller already checked visibility
        seal_name, vidx, field_tys, _decl_mod = info
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
        info = self._resolve_variant(vname, e.line, e.col)
        assert info is not None  # caller already checked visibility
        seal_name, vidx, field_tys, _decl_mod = info
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
                    # Match binders on cleanup-bearing payloads are
                    # implicit-copied at codegen (see
                    # `_bind_variant_pattern`), so each binder is an
                    # owning step — not a borrow. No freeze
                    # registration, no field-origin tag, no escape
                    # restriction on returning a binder. The
                    # ergonomic cost of requiring explicit `copy` at
                    # every parser-style arm outweighed the one
                    # extra clone per matched payload.
                    for binder, fty in zip(arm.pattern.binders, vfs):
                        if binder is None:
                            continue
                        concrete = self._substitute(fty, subst)
                        self._bind(binder, concrete, arm.line, arm.col)
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
