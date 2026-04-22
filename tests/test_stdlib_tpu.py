"""Tests for the Tuppu stdlib (stdlib/*.tpu).

Each test compiles a tiny user program together with the full stdlib
and checks stdout. This is the real dogfooding suite — proves the
language has enough expressive power to write its own library code."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_files_to_binary, stdlib_files


def run_with_stdlib(user_src: str, tmp_path: Path) -> bytes:
    user = tmp_path / "user.tpu"
    user.write_text(user_src)
    files = stdlib_files() + [user]
    binary = compile_files_to_binary(files, tmp_path / "build", name="prog")
    return subprocess.run([str(binary)], capture_output=True).stdout


# --- sanity: the stdlib compiles by itself ---------------------------------

def test_stdlib_compiles_alone(tmp_path):
    # Give it a minimal main; stdlib should compile without error.
    user = "fn main() -> i32 { 0 }"
    assert run_with_stdlib(user, tmp_path) == b""


def test_stdlib_has_expected_files():
    names = {p.name for p in stdlib_files()}
    assert "rat.tpu" in names
    assert "sex.tpu" in names


# --- rat.tpu ---------------------------------------------------------------

def test_rat_neg(tmp_path):
    user = 'fn main() -> i32 { println(rat_neg(rat(3, 2)))\n 0 }'
    assert run_with_stdlib(user, tmp_path) == b"-3/2\n"


def test_rat_abs(tmp_path):
    user = (
        "fn main() -> i32 {\n"
        "  println(rat_abs(rat(-5, 3)))\n"
        "  println(rat_abs(rat(5, 3)))\n"
        "  0\n"
        "}\n"
    )
    assert run_with_stdlib(user, tmp_path) == b"5/3\n5/3\n"


def test_rat_reciprocal(tmp_path):
    user = 'fn main() -> i32 { println(rat_reciprocal(1;30))\n 0 }'
    assert run_with_stdlib(user, tmp_path) == b"2/3\n"


def test_rat_min_max(tmp_path):
    user = (
        "fn main() -> i32 {\n"
        "  println(rat_min(rat(1, 2), rat(2, 3)))\n"  # 1/2
        "  println(rat_max(rat(1, 2), rat(2, 3)))\n"  # 2/3
        "  0\n"
        "}\n"
    )
    assert run_with_stdlib(user, tmp_path) == b"1/2\n2/3\n"


def test_rat_mean(tmp_path):
    # mean of 1/2 and 1/3 is 5/12
    user = (
        "fn main() -> i32 { println(rat_mean(rat(1, 2), rat(1, 3)))\n 0 }"
    )
    assert run_with_stdlib(user, tmp_path) == b"5/12\n"


def test_rat_half(tmp_path):
    user = 'fn main() -> i32 { println(rat_half(rat(5, 3)))\n 0 }'
    assert run_with_stdlib(user, tmp_path) == b"5/6\n"


def test_rat_predicates(tmp_path):
    user = (
        "fn main() -> i32 {\n"
        "  println(rat_is_zero(rat(0, 5)))\n"          # true
        "  println(rat_is_zero(rat(1, 5)))\n"          # false
        "  println(rat_is_negative(rat(-1, 2)))\n"     # true
        "  println(rat_is_negative(rat(1, 2)))\n"      # false
        "  0\n"
        "}\n"
    )
    assert run_with_stdlib(user, tmp_path) == b"true\nfalse\ntrue\nfalse\n"


# --- sex.tpu ---------------------------------------------------------------

def test_hms_to_rat_hours(tmp_path):
    user = (
        "fn main() -> i32 {\n"
        "  println(hms_to_rat_hours(3, 30, 0))\n"      # 7/2
        "  println(hms_to_rat_hours(0, 20, 0))\n"      # 1/3
        "  println(hms_to_rat_hours(1, 0, 0))\n"       # 1/1
        "  0\n"
        "}\n"
    )
    assert run_with_stdlib(user, tmp_path) == b"7/2\n1/3\n1/1\n"


def test_rat_hours_to_seconds(tmp_path):
    user = (
        "fn main() -> i32 {\n"
        "  println(rat_hours_to_seconds(rat(7, 2)))\n"    # 12600
        "  println(rat_hours_to_seconds(rat(1, 3)))\n"    # 1200
        "  0\n"
        "}\n"
    )
    assert run_with_stdlib(user, tmp_path) == b"12600\n1200\n"


def test_degrees_turns_roundtrip(tmp_path):
    # 180° = 1/2 turn → 180° again
    user = (
        "fn main() -> i32 {\n"
        "  step d: rat = rat(180, 1)\n"
        "  step t: rat = degrees_to_rat_turns(d)\n"
        "  println(t)\n"                               # 1/2
        "  println(rat_turns_to_degrees(t))\n"         # 180/1
        "  0\n"
        "}\n"
    )
    assert run_with_stdlib(user, tmp_path) == b"1/2\n180/1\n"


# --- opt-out: --no-stdlib should not pull stdlib in ------------------------

def test_no_stdlib_flag_excludes_library(tmp_path):
    # This program references rat_abs. With stdlib, it works; without, error.
    import sys as _sys
    src = (
        "fn main() -> i32 {\n"
        "  println(rat_abs(rat(-1, 2)))\n"
        "  0\n"
        "}\n"
    )
    user = tmp_path / "u.tpu"
    user.write_text(src)
    # With stdlib: runs cleanly.
    r1 = subprocess.run(
        [_sys.executable, "-m", "tuppu", "run", str(user)],
        capture_output=True,
    )
    assert r1.returncode == 0
    assert r1.stdout == b"1/2\n"

    # Without stdlib: rat_abs is unknown -> compile error.
    r2 = subprocess.run(
        [_sys.executable, "-m", "tuppu", "run", str(user), "--no-stdlib"],
        capture_output=True,
    )
    assert r2.returncode != 0
    assert b"unknown function" in r2.stderr or b"rat_abs" in r2.stderr
