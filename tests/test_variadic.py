"""Tablets-literal syntax (`tablets[N]T { a, b, c }`) plus variadic
`tablets[...]T` params — the two pieces that land together to unlock
self-hosted stdlib fns like `str_concat`. Exercised here: literal
construction, zero-arity + arity-n calls, iteration, and the
typecheck rejects for bad shapes."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import (
    compile_files_to_binary,
    compile_to_binary,
    compile_to_ir,
    stdlib_files,
)
from tuppu.errors import CompileError


def run(src: str, tmp_path: Path) -> tuple[int, bytes, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], capture_output=True)
    return r.returncode, r.stdout, r.stderr


def run_with_stdlib(src: str, tmp_path: Path) -> tuple[int, bytes, bytes]:
    user_file = tmp_path / "main.tpu"
    user_file.write_text(src)
    binary = compile_files_to_binary(
        stdlib_files() + [user_file], tmp_path, name="prog",
    )
    r = subprocess.run([str(binary)], capture_output=True)
    return r.returncode, r.stdout, r.stderr


# --- tablets literal -------------------------------------------------------

def test_tablets_lit_basic(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut nums: tablets[4]i64 = tablets[4]i64 { 1, 2, 3, 4 }\n"
        "  mut i: i64 = 0\n"
        "  while i < nums.len {\n"
        "    println(nums[i])\n"
        "    i = i + 1\n"
        "  }\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1\n2\n3\n4\n"


def test_tablets_lit_step_bound(tmp_path):
    # Step-bound tablets literal works — binding inherits the slot
    # created by _gen_tablets_lit_addr so reads (.len, [i]) go through
    # the same pointer.
    src = (
        "fn main() -> i32 {\n"
        "  step nums: tablets[4]i64 = tablets[4]i64 { 10, 20, 30 }\n"
        "  println(nums.len)\n"
        "  println(nums[2])\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"3\n30\n"


def test_tablets_lit_empty_requires_explicit_type(tmp_path):
    # An empty literal is fine so long as the type is spelled out.
    src = (
        "fn main() -> i32 {\n"
        "  mut nums: tablets[4]i64 = tablets[4]i64 { }\n"
        "  println(nums.len)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\n"


def test_tablets_lit_spans_chunks(tmp_path):
    # More elements than N — the push path chains chunks.
    src = (
        "fn main() -> i32 {\n"
        "  mut xs: tablets[2]i64 = tablets[2]i64 { 1, 2, 3, 4, 5 }\n"
        "  println(xs.len)\n"
        "  mut i: i64 = 0\n"
        "  mut sum: i64 = 0\n"
        "  while i < xs.len {\n"
        "    sum = sum + xs[i]\n"
        "    i = i + 1\n"
        "  }\n"
        "  println(sum)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"5\n15\n"


# --- variadic param --------------------------------------------------------

def test_variadic_sum_various_arities(tmp_path):
    src = (
        "fn sum(parts: tablets[...]i64) -> i64 {\n"
        "  mut total: i64 = 0\n"
        "  mut i: i64 = 0\n"
        "  while i < parts.len {\n"
        "    total = total + parts[i]\n"
        "    i = i + 1\n"
        "  }\n"
        "  total\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  println(sum())\n"
        "  println(sum(42))\n"
        "  println(sum(1, 2, 3, 4, 5))\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"0\n42\n15\n"


def test_variadic_with_str_element(tmp_path):
    # tablets[...]str — element cleanup is handled via cap=0 neutering.
    src = (
        "fn total_len(parts: tablets[...]str) -> i64 {\n"
        "  mut total: i64 = 0\n"
        "  mut i: i64 = 0\n"
        "  while i < parts.len {\n"
        "    total = total + parts[i].len\n"
        "    i = i + 1\n"
        "  }\n"
        "  total\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  println(total_len(\"hello\", \" \", \"world\"))\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"11\n"


def test_variadic_with_fixed_prefix(tmp_path):
    # Fixed args before the variadic — the first is the delimiter;
    # trailing args collect into the tablets.
    src = (
        "fn sum_offset(base: i64, parts: tablets[...]i64) -> i64 {\n"
        "  mut total: i64 = base\n"
        "  mut i: i64 = 0\n"
        "  while i < parts.len {\n"
        "    total = total + parts[i]\n"
        "    i = i + 1\n"
        "  }\n"
        "  total\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  println(sum_offset(100, 1, 2, 3))\n"
        "  println(sum_offset(50))\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"106\n50\n"


def test_variadic_for_loop_iteration(tmp_path):
    # `for p in parts` inside the callee — same shape as iterating
    # a regular tablets binding.
    src = (
        "fn print_all(parts: tablets[...]i64) {\n"
        "  for p in parts { println(p) }\n"
        "}\n"
        "fn main() -> i32 {\n"
        "  print_all(7, 8, 9)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"7\n8\n9\n"


# --- self-hosted str_concat ------------------------------------------------

def test_self_hosted_str_concat_basic(tmp_path):
    # str_concat now lives in stdlib/str.tpu as a variadic fn.
    # Exercises the full path: parse, typecheck a variadic call site,
    # build the tablets literal at the call, pass by pointer to the
    # callee, iterate via indexing inside the callee.
    src = (
        "fn main() -> i32 {\n"
        "  println(str_concat(\"a\", \"b\", \"c\"))\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"abc\n"


def test_self_hosted_str_concat_many_args(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        '  println(str_concat("x=", int_to_str(7), " y=", int_to_str(9)))\n'
        "  println(str_concat())\n"
        '  println(str_concat("only"))\n'
        "  0\n"
        "}\n"
    )
    _, out, _ = run_with_stdlib(src, tmp_path)
    assert out == b"x=7 y=9\n\nonly\n"


# --- typecheck rejections --------------------------------------------------

def test_variadic_param_must_be_last():
    # The variadic marker is only legal on the final parameter.
    with pytest.raises(CompileError, match="must be the last parameter"):
        compile_to_ir(
            "fn bad(parts: tablets[...]i64, trailing: i64) -> i64 { 0 }\n"
            "fn main() -> i32 { 0 }\n"
        )


def test_tablets_lit_element_type_mismatch():
    with pytest.raises(CompileError, match="element 1 has type"):
        compile_to_ir(
            "fn main() -> i32 {\n"
            "  mut xs: tablets[4]str = tablets[4]str { \"a\", 42, \"c\" }\n"
            "  0\n"
            "}\n"
        )


def test_variadic_arg_type_mismatch():
    with pytest.raises(CompileError, match="element 0 has type"):
        compile_to_ir(
            "fn first(parts: tablets[...]str) -> i64 { parts[0].len }\n"
            "fn main() -> i32 {\n"
            "  println(first(42))\n"
            "  0\n"
            "}\n"
        )
