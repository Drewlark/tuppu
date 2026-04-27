"""Type aliases — `type Name = TypeExpr`. Aliases are transparent at
every use site: codegen lowers the target, the typechecker's nominal
identity is the target's name, errors report the alias-side spelling.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_to_binary, compile_to_ir
from tuppu.errors import CompileError


def run(src: str, tmp_path: Path) -> tuple[int, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], capture_output=True)
    return r.returncode, r.stdout


def test_alias_to_primitive(tmp_path):
    src = (
        "type Score = i64\n"
        "fn main() -> i32 {\n"
        "  step s: Score = 42\n"
        "  println(s)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"42\n"


def test_alias_to_tablets(tmp_path):
    # Aliases over generic-shaped types — the canonical motivation:
    # name a long type once.
    src = (
        "type Counts = tablets[64]i64\n"
        "fn main() -> i32 {\n"
        "  mut c: Counts\n"
        "  c.push(10)\n"
        "  c.push(20)\n"
        "  println(c[0])\n"
        "  println(c[1])\n"
        "  println(c.len)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"10\n20\n2\n"


def test_alias_in_struct_field(tmp_path):
    src = (
        "type Name = str\n"
        "type Counts = tablets[16]i64\n"
        "tablet Person { name: Name, scores: Counts }\n"
        "fn main() -> i32 {\n"
        "  mut p: Person\n"
        "  p.name = \"alice\"\n"
        "  p.scores.push(7)\n"
        "  p.scores.push(9)\n"
        "  println(p.name)\n"
        "  println(p.scores[0])\n"
        "  println(p.scores[1])\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"alice\n7\n9\n"


def test_alias_chains_through_other_alias(tmp_path):
    # Alias of an alias: B -> A -> i64. Both spellings must resolve.
    src = (
        "type Inner = i64\n"
        "type Outer = Inner\n"
        "fn main() -> i32 {\n"
        "  step x: Outer = 99\n"
        "  println(x)\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"99\n"


def test_alias_unifies_with_target(tmp_path):
    # An alias and its target should be interchangeable in fn args.
    src = (
        "type IntCount = i64\n"
        "fn double(x: i64) -> i64 { x * 2 }\n"
        "fn main() -> i32 {\n"
        "  step c: IntCount = 21\n"
        "  println(double(c))\n"
        "  0\n"
        "}\n"
    )
    rc, out = run(src, tmp_path)
    assert rc == 0
    assert out == b"42\n"


def test_alias_name_collides_with_struct():
    with pytest.raises(CompileError, match="collides"):
        compile_to_ir(
            "type Foo = i64\n"
            "tablet Foo { x: i64 }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_alias_name_shadows_primitive():
    # The parser rejects primitive type names as alias targets (they
    # tokenize as TYPE_KW, not IDENT) — same enforcement, surfaces
    # at parse time rather than the typecheck shadow-builtin path.
    with pytest.raises(CompileError, match="expected alias name"):
        compile_to_ir(
            "type i64 = bool\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_duplicate_alias_rejected():
    with pytest.raises(CompileError, match="duplicate type alias"):
        compile_to_ir(
            "type X = i64\n"
            "type X = bool\n"
            "fn main() -> i32 { 0 }\n"
        )
