"""Tests for the ergonomic additions: char literals, int-width promotion
in comparisons, multi-arg print, augmented assignment, and `for x in iter`."""
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


def run(src: str, tmp_path: Path, stdin: bytes = b"") -> tuple[int, bytes, bytes]:
    binary = compile_to_binary(src, tmp_path, name="prog")
    r = subprocess.run([str(binary)], input=stdin, capture_output=True)
    return r.returncode, r.stdout, r.stderr


def run_with_stdlib(src: str, tmp_path: Path) -> tuple[int, bytes, bytes]:
    user_file = tmp_path / "main.tpu"
    user_file.write_text(src)
    binary = compile_files_to_binary(
        stdlib_files() + [user_file], tmp_path, name="prog",
    )
    r = subprocess.run([str(binary)], capture_output=True)
    return r.returncode, r.stdout, r.stderr


# --- char literals --------------------------------------------------------

def test_char_literal_is_u8(tmp_path):
    # 'a' == 97 should hold; printing the char as i64 gives 97.
    src = 'fn main() -> i32 { println(\'a\' as i64)\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"97\n"


def test_char_escape_newline(tmp_path):
    src = 'fn main() -> i32 { println(\'\\n\' as i64)\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"10\n"


def test_char_escape_backslash_and_quote(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  println('\\\\' as i64)\n"
        "  println('\\'' as i64)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"92\n39\n"


def test_char_literal_compares_with_u8_from_str(tmp_path):
    # The motivating case: no `as u8` dance needed.
    src = (
        'fn main() -> i32 {\n'
        '  step s = "hi"\n'
        '  if s[0] == \'h\' { println("match") } else { println("nope") }\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"match\n"


def test_empty_char_literal_rejected():
    with pytest.raises(CompileError, match="empty char literal"):
        compile_to_ir("fn main() -> i32 { step c = ''\n 0 }")


def test_multichar_literal_rejected():
    # Two ASCII chars tries to consume the second as the closing quote.
    with pytest.raises(CompileError, match="char literal"):
        compile_to_ir("fn main() -> i32 { step c = 'ab'\n 0 }")


# --- int comparison auto-promote ------------------------------------------

def test_i64_vs_i32_comparison(tmp_path):
    src = (
        "fn small() -> i32 { 42 }\n"
        "fn main() -> i32 {\n"
        "  step big: i64 = 42\n"
        "  if small() == big { println(1) } else { println(0) }\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"1\n"


def test_u8_vs_int_literal(tmp_path):
    # No explicit cast needed on the literal.
    src = (
        'fn main() -> i32 {\n'
        '  step s = "A"\n'
        '  if s[0] == 65 { println("yes") } else { println("no") }\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"yes\n"


# --- multi-arg print ------------------------------------------------------

def test_multi_arg_println(tmp_path):
    # Replaces the print("foo: "); println(x) pattern.
    src = 'fn main() -> i32 { println("count: ", 42)\n 0 }'
    _, out, _ = run(src, tmp_path)
    assert out == b"count: 42\n"


def test_multi_arg_print_no_newline(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        '  print("a", "b", "c")\n'
        '  println("!")\n'
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"abc!\n"


def test_multi_arg_mixed_types(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  println("x=", 3, " y=", 4, " p=", 3 * 4)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"x=3 y=4 p=12\n"


# --- augmented assignment -------------------------------------------------

def test_plus_eq(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut i: i64 = 0\n"
        "  i += 5\n"
        "  i += 7\n"
        "  println(i)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"12\n"


def test_all_aug_assign_ops(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut x: i64 = 20\n"
        "  x += 5\n"     # 25
        "  x -= 3\n"     # 22
        "  x *= 2\n"     # 44
        "  x /= 4\n"     # 11
        "  x %= 3\n"     # 2
        "  println(x)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"2\n"


def test_plus_eq_in_loop(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut sum: i64 = 0\n"
        "  mut i: i64 = 1\n"
        "  while i <= 10 {\n"
        "    sum += i\n"
        "    i += 1\n"
        "  }\n"
        "  println(sum)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"55\n"


# --- for x in iter --------------------------------------------------------

def test_for_over_str(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  mut vowels: i64 = 0\n'
        '  for c in "education" {\n'
        '    if c == \'a\' || c == \'e\' || c == \'i\' || c == \'o\' || c == \'u\' {\n'
        '      vowels += 1\n'
        '    }\n'
        '  }\n'
        '  println(vowels)\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"5\n"


def test_for_over_str_preserves_order(tmp_path):
    src = (
        'fn main() -> i32 {\n'
        '  for c in "abc" { println(c as i64) }\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"97\n98\n99\n"


def test_for_over_tablets(tmp_path):
    src = (
        "fn main() -> i32 {\n"
        "  mut t: tablets[2]i64\n"
        "  t.push(10)\n"
        "  t.push(20)\n"
        "  t.push(30)\n"
        "  mut sum: i64 = 0\n"
        "  for x in t { sum += x }\n"
        "  release t\n"
        "  println(sum)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"60\n"


def test_for_over_table(tmp_path):
    src = (
        "fn sq(i: i64) -> i64 { i * i }\n"
        "table squares[0..5]: i64 = sq\n"
        "fn main() -> i32 {\n"
        "  mut sum: i64 = 0\n"
        "  for x in squares { sum += x }\n"
        "  println(sum)\n"
        "  0\n"
        "}\n"
    )
    _, out, _ = run(src, tmp_path)
    assert out == b"30\n"


def test_for_loop_var_is_immutable():
    # Assigning to the loop variable should fail — it's step-bound.
    with pytest.raises(CompileError, match="cannot assign"):
        compile_to_ir(
            'fn main() -> i32 {\n'
            '  for c in "abc" { c = \'z\' }\n'
            '  0\n'
            '}\n'
        )


def test_for_over_non_iterable_rejected():
    with pytest.raises(CompileError, match="cannot iterate over"):
        compile_to_ir(
            "fn main() -> i32 {\n"
            "  step x: i64 = 42\n"
            "  for c in x { println(c) }\n"
            "  0\n"
            "}\n"
        )


# --- idiomatic rewrite end-to-end -----------------------------------------

def test_idiomatic_vowel_count(tmp_path):
    # The shape we wanted in greeting.tpu — compare with the original:
    # no `as u8` casts, no manual index counter, `+=` for accumulation.
    src = (
        'fn count_vowels(s: str) -> i64 {\n'
        '  mut n: i64 = 0\n'
        '  for c in s {\n'
        '    if c == \'a\' || c == \'e\' || c == \'i\' || c == \'o\' || c == \'u\' {\n'
        '      n += 1\n'
        '    }\n'
        '  }\n'
        '  n\n'
        '}\n'
        'fn main() -> i32 {\n'
        '  println("vowels: ", count_vowels("Mesopotamia"))\n'
        '  0\n'
        '}\n'
    )
    _, out, _ = run(src, tmp_path)
    # Mesopotamia → e, o, o, a, i, a = 6
    assert out == b"vowels: 6\n"
