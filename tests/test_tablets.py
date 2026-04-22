"""Tablets — chained-chunk growable storage (SPEC §4.4)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.codegen import CodegenError
from tuppu.errors import CompileError
from tuppu.driver import compile_to_binary, compile_to_ir


def run(src: str, tmp_path: Path) -> tuple[int, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    result = subprocess.run([str(binary)], capture_output=True)
    return result.returncode, result.stdout


# --- basic usage -----------------------------------------------------------

def test_empty_tablets_has_zero_len(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  println(t.len)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"0\n"


def test_push_bumps_len(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(10)\n"
        "  t.push(20)\n"
        "  t.push(30)\n"
        "  println(t.len)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"3\n"


def test_indexed_access(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(100)\n"
        "  t.push(200)\n"
        "  t.push(300)\n"
        "  println(t[0])\n"
        "  println(t[1])\n"
        "  println(t[2])\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"100\n200\n300\n"


def test_chain_across_multiple_tablets(tmp_path):
    # Chunk size 4, push 10 elements, exercise chain walk on access.
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  mut i: i64 = 0\n"
        "  while i < 10 {\n"
        "    t.push(i * i)\n"
        "    i = i + 1\n"
        "  }\n"
        "  println(t.len)\n"
        "  println(t[0])\n"   # 0
        "  println(t[3])\n"   # 9 (end of first chunk)
        "  println(t[4])\n"   # 16 (start of second chunk)
        "  println(t[7])\n"   # 49 (end of second chunk)
        "  println(t[9])\n"   # 81 (middle of third chunk)
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"10\n0\n9\n16\n49\n81\n"


# --- rat element type ------------------------------------------------------

def test_tablets_of_rat(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut r: tablets[2]rat\n"
        "  r.push(1;30)\n"
        "  r.push(0;20)\n"
        "  r.push(rat(5, 6))\n"
        "  println(r[0])\n"   # 3/2
        "  println(r[1])\n"   # 1/3
        "  println(r[2])\n"   # 5/6
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"3/2\n1/3\n5/6\n"


# --- release ---------------------------------------------------------------

def test_release_resets_len(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(1)\n"
        "  t.push(2)\n"
        "  release t\n"
        "  println(t.len)\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"0\n"


def test_release_then_push_works(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(1)\n"
        "  t.push(2)\n"
        "  release t\n"
        "  t.push(99)\n"
        "  println(t.len)\n"
        "  println(t[0])\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"1\n99\n"


# --- bounds checks ---------------------------------------------------------

def test_index_out_of_range_traps(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(42)\n"
        "  println(t[1])\n"   # only index 0 exists
        "  0\n"
        "}\n"
    )
    binary = compile_to_binary(src, tmp_path, name="oob")
    r = subprocess.run([str(binary)], capture_output=True)
    assert r.returncode != 0


def test_negative_index_traps(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[4]i64\n"
        "  t.push(42)\n"
        "  println(t[-1])\n"
        "  0\n"
        "}\n"
    )
    binary = compile_to_binary(src, tmp_path, name="neg")
    r = subprocess.run([str(binary)], capture_output=True)
    assert r.returncode != 0


# --- generated helpers live in the IR --------------------------------------

def test_helpers_emitted():
    ir = compile_to_ir(
        "fn main() -> i32 { mut t: tablets[256]i64\n t.push(1)\n 0 }"
    )
    assert "__tuppu_tbls_" in ir
    assert "push" in ir
    assert "malloc" in ir


def test_monomorphization_per_elem_type():
    """tablets[N]i64 and tablets[N]rat should each get their own helpers."""
    ir = compile_to_ir(
        "fn main() -> i32 {\n"
        "  mut a: tablets[4]i64\n"
        "  mut b: tablets[4]rat\n"
        "  a.push(1)\n"
        "  b.push(1;30)\n"
        "  0\n"
        "}\n"
    )
    push_fns = [line for line in ir.splitlines()
                if "__tuppu_tbls_" in line and "push" in line and "define" in line]
    assert len(push_fns) == 2, f"expected 2 distinct push fns, got {push_fns}"


# --- error cases -----------------------------------------------------------

def test_step_tablets_without_init_errors():
    with pytest.raises(Exception, match="step"):
        compile_to_ir(
            "fn main() -> i32 { step t: tablets[4]i64\n 0 }"
        )


def test_mut_no_init_no_type_errors():
    with pytest.raises(Exception, match="type annotation"):
        compile_to_ir("fn main() -> i32 { mut x\n 0 }")


def test_tablets_unknown_method():
    with pytest.raises(CompileError, match="no method"):
        compile_to_ir(
            "fn main() -> i32 { mut t: tablets[4]i64\n t.pop()\n 0 }"
        )


def test_tablets_unknown_field():
    with pytest.raises(CompileError, match="no field"):
        compile_to_ir(
            "fn main() -> i32 { mut t: tablets[4]i64\n println(t.capacity)\n 0 }"
        )
