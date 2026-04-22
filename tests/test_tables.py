"""Comptime tables: the hero feature from SPEC §9.

Each test compiles a program, asserts the static table contents are baked
into the emitted IR, and (for runtime tests) verifies `table[i]` reads
the expected value or traps on out-of-bounds."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.codegen import CodegenError
from tuppu.errors import CompileError
from tuppu.driver import compile_to_binary, compile_to_ir


def run(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    result = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return result.returncode, result.stdout


FACT_SRC = (
    "fn fact(n: i64) -> i64 {\n"
    "  if n < 2 { 1 } else { n * fact(n - 1) }\n"
    "}\n"
    "table fact_table[0..10]: i64 = fact\n"
)


# --- comptime evaluation produces correct static data ----------------------

def test_table_is_baked_into_ir():
    ir = compile_to_ir(FACT_SRC + "fn main() -> i32 { fact_table[0] as i32 }\n")
    # All 10 factorials, in order, appear as LLVM constants.
    expected = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880]
    assert (
        "[" + ", ".join(f"i64 {v}" for v in expected) + "]"
    ) in ir, f"expected factorials not found in IR:\n{ir}"


def test_table_lookup_at_runtime(tmp_path):
    # Read an index, print the table value.
    src = FACT_SRC + (
        "fn main() -> i32 {\n"
        "  step i: i64 = read_int()\n"
        "  println(fact_table[i])\n"
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path, stdin=b"0\n")
    assert out == b"1\n"
    _, out = run(src, tmp_path, stdin=b"5\n")
    assert out == b"120\n"
    _, out = run(src, tmp_path, stdin=b"9\n")
    assert out == b"362880\n"


def test_table_function_can_be_elided():
    # The generator function fact should not be *required* to exist in the
    # final binary for a program that only touches the table — a real
    # language would let LLVM DCE it. For now we just assert that the table
    # data is present.
    src = FACT_SRC + (
        "fn main() -> i32 { (fact_table[5] / 10) as i32 }\n"  # 12
    )
    binary = compile_to_binary(src, Path("build/table_test1"), name="t1")
    assert subprocess.run([str(binary)]).returncode == 12


# --- lo != 0 offsets --------------------------------------------------------

def test_table_nonzero_lo(tmp_path):
    src = (
        "fn square(n: i64) -> i64 { n * n }\n"
        "table squares[5..10]: i64 = square\n"
        "fn main() -> i32 {\n"
        "  println(squares[5])\n"  # 25
        "  println(squares[9])\n"  # 81
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"25\n81\n"


# --- out of bounds traps at runtime ----------------------------------------

def test_oob_below_range_traps(tmp_path):
    src = FACT_SRC + (
        "fn main() -> i32 {\n"
        "  step i: i64 = read_int()\n"
        "  println(fact_table[i])\n"
        "  0\n"
        "}\n"
    )
    binary = compile_to_binary(src, tmp_path, name="oob")
    r = subprocess.run([str(binary)], input=b"-1\n", capture_output=True)
    assert r.returncode != 0


def test_oob_above_range_traps(tmp_path):
    src = FACT_SRC + (
        "fn main() -> i32 {\n"
        "  step i: i64 = read_int()\n"
        "  println(fact_table[i])\n"
        "  0\n"
        "}\n"
    )
    binary = compile_to_binary(src, tmp_path, name="oob2")
    r = subprocess.run([str(binary)], input=b"10\n", capture_output=True)
    assert r.returncode != 0


# --- rat-valued tables -----------------------------------------------------

def test_rat_table(tmp_path):
    # Babylonian reciprocal table: 1/n for n in 1..8 (skip 0).
    src = (
        "fn reciprocal(n: i64) -> rat { rat(1, n) }\n"
        "table recip[1..8]: rat = reciprocal\n"
        "fn main() -> i32 {\n"
        "  println(recip[1])\n"  # 1/1
        "  println(recip[2])\n"  # 1/2
        "  println(recip[3])\n"  # 1/3 — exact rational
        "  println(recip[7])\n"  # 1/7
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"1/1\n1/2\n1/3\n1/7\n"


# --- tables can reference earlier tables -----------------------------------

def test_table_can_depend_on_earlier_table(tmp_path):
    # squares of fact(i) for i in 0..5 -> (1, 1, 4, 36, 576, 14400)
    src = (
        "fn fact(n: i64) -> i64 {\n"
        "  if n < 2 { 1 } else { n * fact(n - 1) }\n"
        "}\n"
        "table facts[0..5]: i64 = fact\n"
        "fn square_fact(i: i64) -> i64 { facts[i] * facts[i] }\n"
        "table fact_sq[0..5]: i64 = square_fact\n"
        "fn main() -> i32 {\n"
        "  println(fact_sq[0])\n"   # 1
        "  println(fact_sq[3])\n"   # 36
        "  println(fact_sq[4])\n"   # 576
        "  0\n"
        "}\n"
    )
    _, out = run(src, tmp_path)
    assert out == b"1\n36\n576\n"


# --- error cases -----------------------------------------------------------

def test_io_intrinsic_at_comptime_rejected():
    src = (
        "fn bad(n: i64) -> i64 { println(n)\n n }\n"
        "table t[0..3]: i64 = bad\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="cannot be called at build time"):
        compile_to_ir(src)


def test_table_generator_must_be_function():
    src = (
        "table t[0..5]: i64 = 42\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="must be a function name"):
        compile_to_ir(src)


def test_table_generator_unknown_function():
    src = (
        "table t[0..5]: i64 = does_not_exist\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="not a function"):
        compile_to_ir(src)


def test_table_generator_wrong_arity():
    src = (
        "fn two(a: i64, b: i64) -> i64 { a + b }\n"
        "table t[0..3]: i64 = two\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="must take 1 argument"):
        compile_to_ir(src)


def test_empty_range_rejected():
    src = (
        "fn f(n: i64) -> i64 { n }\n"
        "table t[10..5]: i64 = f\n"
        "fn main() -> i32 { 0 }\n"
    )
    with pytest.raises(CompileError, match="empty or inverted"):
        compile_to_ir(src)
