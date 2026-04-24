"""Per-fn effect analysis for the freeze-while-borrow rule.

Computes, for each user-defined fn and each parameter index, the set
of field paths on the param that the fn may write — directly via
`param.field = ...` / `param = ...`, or indirectly via a recursive
call that propagates writes back to the param.

The result lets the call-site invalidation code be precise: instead
of blindly invalidating every live borrow rooted at a mut arg, only
borrows whose own path overlaps a write path get invalidated.

Paths overlap if one is a prefix of the other. `("src",)` overlaps
`("src", "inner")` (writing a parent frees the child), and
`("src", "inner")` overlaps `("src",)` (writing a nested field
updates the parent's memory). Two unrelated siblings like `("pos",)`
and `("src",)` do not overlap — a write to `l.pos` leaves borrows
of `l.src` intact.

Conservative fallbacks (marked `full=True` on the ParamEffects):
- Colophon / extern callees: unknown body, treat as writes-all.
- Fn-value calls / generic callees / unresolved callees: treat as
  writes-all on every param whose arg syntactically roots at one of
  our params.
- Container-shape ops (`release p.field`, method calls on a param
  field, index assigns through a param): conservative.

Fixed-point: start from empty (pure) summaries and iterate. Each
iteration re-walks every body using the previous iteration's
summaries; we stop when nothing changes. Always terminates because
summaries grow monotonically (paths accumulate, `full` only flips
from False→True).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from . import ast as A


@dataclass(frozen=True)
class ParamEffects:
    """One param's summary.

    `full=True` → conservative; treat as "anything in this param may
    have been written". Callers fall back to full invalidation.
    `paths` → specific field paths that may be written (each a tuple
    of field names; `("__index__",)` marks an opaque index write).
    """
    full: bool = False
    paths: frozenset[tuple[str, ...]] = field(default_factory=frozenset)

    def conflicts_with(self, borrow_path: tuple[str, ...]) -> bool:
        """Does this effect potentially invalidate a borrow at
        `borrow_path` (relative to the same root)?"""
        if self.full:
            return True
        for wp in self.paths:
            if _paths_overlap(wp, borrow_path):
                return True
        return False

    def is_pure(self) -> bool:
        return not self.full and not self.paths


PURE = ParamEffects()
CONSERVATIVE = ParamEffects(full=True)


def _paths_overlap(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Two field paths overlap if one is a prefix of the other."""
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def _expr_root_and_path(e: A.Expr) -> tuple[str, tuple[str, ...]] | None:
    """Walk Field/Index chain down to an Ident. Returns
    `(root_name, path)` or None if the expression isn't rooted at
    an Ident (literals, binary exprs, etc.)."""
    parts: list[str] = []
    cur: A.Expr = e
    while True:
        if isinstance(cur, A.Ident):
            return (cur.name, tuple(reversed(parts)))
        if isinstance(cur, A.Field):
            parts.append(cur.name)
            cur = cur.target
            continue
        if isinstance(cur, A.Index):
            parts.append("__index__")
            cur = cur.target
            continue
        return None


class EffectAnalyzer:
    """Fixed-point effect analysis over the set of user fns. Extern/
    colophon fns aren't analyzed — they live only in `extern_names`
    and are looked up to produce conservative summaries at callsites."""

    def __init__(
        self, fn_decls: Iterable[A.FnDecl], extern_names: set[str],
    ) -> None:
        self.fn_decls: dict[str, A.FnDecl] = {d.name: d for d in fn_decls}
        self.extern_names = extern_names
        # Seed with empty (pure) summaries so the first iteration
        # sees optimistic assumptions for recursive / mutually-
        # recursive calls.
        self.effects: dict[str, list[ParamEffects]] = {
            name: [PURE for _ in d.params]
            for name, d in self.fn_decls.items()
        }

    def run(self, max_iter: int = 32) -> dict[str, list[ParamEffects]]:
        for _ in range(max_iter):
            changed = False
            for name, fn in self.fn_decls.items():
                new_summary = self._analyze(fn)
                if new_summary != self.effects[name]:
                    self.effects[name] = new_summary
                    changed = True
            if not changed:
                break
        return self.effects

    # ---- analysis --------------------------------------------------

    def _analyze(self, fn: A.FnDecl) -> list[ParamEffects]:
        param_names = [p.name for p in fn.params]
        n = len(param_names)
        full = [False] * n
        paths: list[set[tuple[str, ...]]] = [set() for _ in range(n)]
        ctx = (param_names, full, paths)
        self._visit_block(fn.body, ctx)
        out: list[ParamEffects] = []
        for i in range(n):
            if full[i]:
                out.append(ParamEffects(full=True))
            else:
                out.append(ParamEffects(paths=frozenset(paths[i])))
        return out

    def _mark(self, ctx, root: str, path: tuple[str, ...]) -> None:
        param_names, full, paths = ctx
        if root not in param_names:
            return
        idx = param_names.index(root)
        if not path:
            full[idx] = True
        else:
            paths[idx].add(path)

    def _mark_full(self, ctx, root: str) -> None:
        param_names, full, _paths = ctx
        if root not in param_names:
            return
        full[param_names.index(root)] = True

    def _visit_block(self, b: A.Block, ctx) -> None:
        for s in b.stmts:
            self._visit(s, ctx)
        if b.tail is not None:
            self._visit(b.tail, ctx)

    def _visit(self, node, ctx) -> None:
        # Statements
        if isinstance(node, A.Assign):
            rp = _expr_root_and_path(node.target)
            if rp is not None:
                self._mark(ctx, rp[0], rp[1])
            self._visit(node.value, ctx)
            return
        if isinstance(node, A.Binding):
            if node.init is not None:
                self._visit(node.init, ctx)
            return
        if isinstance(node, A.ExprStmt):
            self._visit(node.expr, ctx)
            return
        if isinstance(node, A.While):
            self._visit(node.cond, ctx)
            self._visit_block(node.body, ctx)
            return
        if isinstance(node, A.ForStmt):
            self._visit(node.iter, ctx)
            self._visit_block(node.body, ctx)
            return
        if isinstance(node, A.YieldStmt):
            if node.value is not None:
                self._visit(node.value, ctx)
            return
        if isinstance(node, A.ReleaseStmt):
            # `release x` frees any heap x owns — conservative on x.
            self._mark_full(ctx, node.name)
            return

        # Expressions
        if isinstance(node, A.Call):
            self._on_call(node, ctx)
            for a in node.args:
                self._visit(a, ctx)
            if isinstance(node.callee, A.Field):
                self._visit(node.callee.target, ctx)
            return
        if isinstance(node, A.Unary):
            self._visit(node.operand, ctx)
            return
        if isinstance(node, (A.Cast, A.Copy)):
            self._visit(node.value, ctx)
            return
        if isinstance(node, A.Binary):
            self._visit(node.lhs, ctx)
            self._visit(node.rhs, ctx)
            return
        if isinstance(node, A.Field):
            self._visit(node.target, ctx)
            return
        if isinstance(node, A.Index):
            self._visit(node.target, ctx)
            self._visit(node.index, ctx)
            return
        if isinstance(node, A.Slice):
            self._visit(node.target, ctx)
            if node.lo is not None:
                self._visit(node.lo, ctx)
            if node.hi is not None:
                self._visit(node.hi, ctx)
            return
        if isinstance(node, A.Block):
            self._visit_block(node, ctx)
            return
        if isinstance(node, A.IfExpr):
            self._visit(node.cond, ctx)
            self._visit(node.then, ctx)
            if node.else_ is not None:
                self._visit(node.else_, ctx)
            return
        if isinstance(node, A.MatchExpr):
            self._visit(node.scrutinee, ctx)
            for arm in node.arms:
                self._visit(arm.body, ctx)
            return
        if isinstance(node, A.StructLit):
            for _name, val in node.fields:
                self._visit(val, ctx)
            return
        if isinstance(node, A.TabletsLit):
            for el in node.fields:
                self._visit(el, ctx)
            return
        # Leaves (IntLit, SexLit, StringLit, CharLit, BoolLit, Ident,
        # LostLit): nothing to do.

    def _on_call(self, c: A.Call, ctx) -> None:
        callee = c.callee
        # Method call: conservative on receiver's root. Tablets.push
        # is append-only (doesn't invalidate existing borrows), but
        # it can still free nothing on the receiver — treat it as
        # "writes on the receiver container" which the callsite
        # already treats specially. For other methods, err wide.
        if isinstance(callee, A.Field):
            target_rp = _expr_root_and_path(callee.target)
            if target_rp is not None:
                root, path = target_rp
                if not path:
                    self._mark_full(ctx, root)
                else:
                    self._mark(ctx, root, path)
            return
        if not isinstance(callee, A.Ident):
            # Paren call, index call, etc. Conservative for every
            # arg that roots at one of our params.
            for a in c.args:
                rp = _expr_root_and_path(a)
                if rp is not None:
                    self._mark_full(ctx, rp[0])
            return
        name = callee.name
        if name in self.extern_names:
            # Extern: we don't know what the C side does.
            for a in c.args:
                rp = _expr_root_and_path(a)
                if rp is not None:
                    self._mark_full(ctx, rp[0])
            return
        summary = self.effects.get(name)
        if summary is None:
            # Unknown user fn (generic instance, variant ctor, etc.).
            # Conservative.
            for a in c.args:
                rp = _expr_root_and_path(a)
                if rp is not None:
                    self._mark_full(ctx, rp[0])
            return
        # Propagate: project callee's per-param effects through each
        # arg's path into our params.
        for arg_idx, arg in enumerate(c.args):
            if arg_idx >= len(summary):
                continue
            pe = summary[arg_idx]
            if pe.is_pure():
                continue
            rp = _expr_root_and_path(arg)
            if rp is None:
                continue
            arg_root, arg_path = rp
            if pe.full:
                if not arg_path:
                    self._mark_full(ctx, arg_root)
                else:
                    self._mark(ctx, arg_root, arg_path)
                continue
            for pp in pe.paths:
                self._mark(ctx, arg_root, arg_path + pp)
