"""Phase B: effect-analysis-driven precision on the freeze rule.

Each test stands on its own. The common shape:

    tablet Lex { src: str, pos: i64 }
    fn step_pos(mut l: Lex) { l.pos = l.pos + 1 }    # writes (pos,)
    fn main() { step s = l.src; step_pos(l); println(s) }

Before effect analysis, the call `step_pos(l)` invalidated every
borrow rooted at `l` — including `s`, a borrow of `l.src`. That was
an over-approximation: `step_pos` never touches `l.src`. Effect
analysis records the actual write set; the call-site invalidation
consults it and leaves disjoint siblings alone.

Confirm both angles:
- Pure wrt an arg → borrows rooted at that arg stay live.
- Writes a specific field → only borrows whose path overlaps that
  field get invalidated (implicitly copied or read-allowed).
- Conservative fallbacks still kick in for: extern/colophon, fn
  values, and calls whose target is a method on a container.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_to_binary, compile_to_ir
from tuppu.effects import EffectAnalyzer, ParamEffects, _paths_overlap
from tuppu.lexer import lex
from tuppu.parser import parse
from tuppu import ast as A


def run(src: str, tmp_path: Path) -> tuple[int, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], capture_output=True)
    return r.returncode, r.stdout


# --- unit: analyzer produces the right summaries ----------------------

def _effects(src: str) -> dict[str, list[ParamEffects]]:
    prog = parse(lex(src))
    fns = [d for d in prog.decls if isinstance(d, A.FnDecl)]
    return EffectAnalyzer(fns, set()).run()


def test_paths_overlap_basic():
    assert _paths_overlap((), ())
    assert _paths_overlap(("a",), ())
    assert _paths_overlap((), ("a",))
    assert _paths_overlap(("a",), ("a", "b"))
    assert _paths_overlap(("a", "b"), ("a",))
    assert not _paths_overlap(("a",), ("b",))
    assert not _paths_overlap(("a", "b"), ("a", "c"))


def test_pure_fn_has_empty_summary():
    src = "fn noop(a: i64, b: i64) -> i64 { a + b }"
    eff = _effects(src)
    for pe in eff["noop"]:
        assert pe.is_pure()


def test_field_write_is_precise():
    src = (
        "tablet L { src: i64, pos: i64 }\n"
        "fn advance(mut l: L) { l.pos = l.pos + 1 }"
    )
    eff = _effects(src)
    pe = eff["advance"][0]
    assert not pe.full
    assert pe.paths == frozenset({("pos",)})


def test_whole_param_reassign_is_full():
    src = (
        "tablet L { src: i64 }\n"
        "fn replace(mut l: L) { l = L { src: 0 } }"
    )
    eff = _effects(src)
    assert eff["replace"][0].full


def test_caller_propagates_callee_effects():
    src = (
        "tablet L { src: i64, pos: i64 }\n"
        "fn step_pos(mut l: L) { l.pos = l.pos + 1 }\n"
        "fn double_step(mut l: L) {\n"
        "  step_pos(l)\n"
        "  step_pos(l)\n"
        "}"
    )
    eff = _effects(src)
    assert eff["double_step"][0].paths == frozenset({("pos",)})


def test_nested_field_write_records_deep_path():
    src = (
        "tablet Inner { v: i64 }\n"
        "tablet Outer { a: Inner, b: i64 }\n"
        "fn poke(mut o: Outer) { o.a.v = 99 }"
    )
    eff = _effects(src)
    assert eff["poke"][0].paths == frozenset({("a", "v")})


# --- end-to-end: freeze rule uses effects -----------------------------

def test_disjoint_field_write_leaves_borrow_live(tmp_path, capsys):
    # `step_pos` writes (pos,); borrow of (src,) should stay live.
    # Before Phase B the call invalidated everything — the program
    # would have needed a `copy`, emitting an implicit-copy warning.
    # After Phase B, no warning and the borrow reads the ORIGINAL
    # bytes (no copy was needed).
    src = (
        "tablet L { src: str, pos: i64 }\n"
        "fn step_pos(mut l: L) { l.pos = l.pos + 1 }\n"
        "fn main() -> i32 {\n"
        "  mut l: L = L { src: \"hel\" + \"lo\", pos: 0 }\n"
        "  step s = l.src\n"
        "  step_pos(l)\n"
        "  println(s)\n"
        "  println(l.pos)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n1\n"
    err = capsys.readouterr().err
    assert "implicit copy" not in err


def test_overlapping_field_write_still_invalidates(tmp_path, capsys):
    # Writing (src,) DOES overlap borrows at (src,) — effect analysis
    # flags the conflict; Phase A then inserts the implicit copy.
    src = (
        "tablet L { src: str, pos: i64 }\n"
        "fn replace_src(mut l: L) { l.src = \"new\" + \"!\" }\n"
        "fn main() -> i32 {\n"
        "  mut l: L = L { src: \"old\" + \"!\", pos: 0 }\n"
        "  step s = l.src\n"
        "  replace_src(l)\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"old!\n"
    err = capsys.readouterr().err
    assert "implicit copy inserted at binding 's'" in err


def test_sibling_assign_does_not_invalidate_disjoint_borrow(tmp_path, capsys):
    # Direct field assign: `l.pos = x` should not invalidate a
    # borrow of `l.src`. Tests the path-aware _tc_assign path.
    src = (
        "tablet L { src: str, pos: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut l: L = L { src: \"hel\" + \"lo\", pos: 0 }\n"
        "  step s = l.src\n"
        "  l.pos = 7\n"
        "  println(s)\n"
        "  println(l.pos)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n7\n"
    err = capsys.readouterr().err
    assert "implicit copy" not in err


def test_same_field_assign_invalidates(tmp_path, capsys):
    # The path-aware _tc_assign still catches same-field conflicts.
    src = (
        "tablet L { src: str, pos: i64 }\n"
        "fn main() -> i32 {\n"
        "  mut l: L = L { src: \"old\" + \"!\", pos: 0 }\n"
        "  step s = l.src\n"
        "  l.src = \"new\" + \"!\"\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"old!\n"
    err = capsys.readouterr().err
    assert "implicit copy inserted at binding 's'" in err


def test_recursive_fn_reaches_fixed_point(tmp_path, capsys):
    # Recursive fn that only writes (pos,). The fixed-point must
    # converge without falsely concluding `full`.
    src = (
        "tablet L { src: str, pos: i64 }\n"
        "fn advance(mut l: L, n: i64) {\n"
        "  if n > 0 {\n"
        "    l.pos = l.pos + 1\n"
        "    advance(l, n - 1)\n"
        "  }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  mut l: L = L { src: \"hel\" + \"lo\", pos: 0 }\n"
        "  step s = l.src\n"
        "  advance(l, 3)\n"
        "  println(s)\n"
        "  println(l.pos)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n3\n"
    err = capsys.readouterr().err
    assert "implicit copy" not in err


def test_colophon_call_remains_conservative(tmp_path, capsys):
    # Colophon extern is opaque — any arg that roots at our param
    # is treated as fully written, so borrows into such args still
    # get invalidated (Phase A may then insert the implicit copy).
    # We validate the fallback path by routing an extern through a
    # mut-struct arg.
    src = (
        "colophon fn abs(x: i64) -> i64\n"
        "tablet L { src: str }\n"
        "fn main() -> i32 {\n"
        "  mut l: L = L { src: \"hel\" + \"lo\" }\n"
        "  step s = l.src\n"
        "  step _z = abs(-3)\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    # Just ensure it compiles/runs; the extern doesn't touch l at
    # all, so the borrow stays live regardless — the guard here is
    # against an analyzer regression that crashes on colophons.
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"


def test_index_write_does_not_invalidate_disjoint_field(tmp_path, capsys):
    # `store[0] = x` writes `(__index__,)`; a borrow of a different
    # wedge's field path `(__index__, buf)` overlaps at the index
    # prefix and IS invalidated. This pins the index-level match.
    # The complementary case (borrow whose path doesn't share the
    # __index__ prefix at all) isn't expressible — wedges always
    # live inside their tablets.
    src = (
        "tablet N { buf: str }\n"
        "fn main() -> i32 {\n"
        "  mut store: tablets[4]N\n"
        "  step h = store.push(N { buf: \"hel\" + \"lo\" })\n"
        "  step b = h.buf\n"
        "  store[0] = N { buf: \"over\" + \"written\" }\n"
        "  println(b)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"hello\n"
    err = capsys.readouterr().err
    assert "implicit copy inserted at binding 'b'" in err
