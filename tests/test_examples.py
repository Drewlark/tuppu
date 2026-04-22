"""Compile every .tpu file under examples/ and verify it behaves as
expected. Each entry specifies the expected exit code and optionally
expected stdout. Acts as a living integration suite — when a new
example is added, register it here."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import compile_files_to_binary, stdlib_files


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

# Each value is {"exit": int, "stdout": bytes | None}.
# stdout=None means "don't check".
EXPECTED: dict[str, dict] = {
    "fact.tpu":              {"exit": 120},
    "fact_iter.tpu":         {"exit": 120},
    "fib.tpu":               {"exit": 55},
    "discriminant.tpu":      {"exit": 1},
    "multiple_of_seven.tpu": {"exit": 7},
    "grade.tpu":             {"exit": 66},
    "hello.tpu":             {"exit": 0, "stdout": b"Hello, Tuppu!\n"},
    "fizzbuzz.tpu":          {
        "exit": 0,
        "stdout": (
            b"1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n"
            b"11\nFizz\n13\n14\nFizzBuzz\n"
        ),
    },
    "sexagesimal.tpu":       {
        "exit": 0,
        "stdout": (
            b"a = 1;30\n"
            b"b = 0;20\n"
            b"a + b = 1;50\n"
            b"0;40 + 0;30 = 1;10\n"
            b"1 30 + 0;20 = 1 30;20\n"
            b"1;30 + -(1;40) = -0;10\n"
            b"(a + b) as rat = 11/6\n"
            b"mean = 11/12\n"
            b"|-5/3| = 5/3\n"
            b"reciprocal of 0;20 = 3/1\n"
            b"3:30:00 in rat hours = 7/2\n"
            b"back to seconds = 12600\n"
            b"YBC 7289 sqrt2 = 1;24 51 10\n"
            b"              as rat = 30547/21600\n"
        ),
    },

    "collect_squares.tpu":   {
        "exit": 0,
        "stdout": (
            b"squares 0..9:\n"
            b"  0^2 = 0\n"
            b"  1^2 = 1\n"
            b"  2^2 = 4\n"
            b"  3^2 = 9\n"
            b"  4^2 = 16\n"
            b"  5^2 = 25\n"
            b"  6^2 = 36\n"
            b"  7^2 = 49\n"
            b"  8^2 = 64\n"
            b"  9^2 = 81\n"
            b"sum = 285\n"
        ),
    },
    "points.tpu":            {
        "exit": 0,
        "stdout": b"3\n4\n25\n",
    },
    "greeting.tpu":          {
        "exit": 0,
        "stdout": (
            b"message: Ur is in Mesopotamia\n"
            b"length:  20\n"
            b"vowels:  8\n"
            b"starts with 'Ur': true\n"
            b"ends with '.tpu': false\n"
        ),
    },
    "reciprocal_table.tpu":  {
        "exit": 0,
        "stdout": (
            b"The Babylonian reciprocal table (1 through 9):\n"
            b"  1/1 = 1/1\n"
            b"  1/2 = 1/2\n"
            b"  1/3 = 1/3\n"
            b"  1/4 = 1/4\n"
            b"  1/5 = 1/5\n"
            b"  1/6 = 1/6\n"
            b"  1/7 = 1/7\n"
            b"  1/8 = 1/8\n"
            b"  1/9 = 1/9\n"
        ),
    },
}


@pytest.mark.parametrize("filename,expected", sorted(EXPECTED.items()))
def test_example_runs_as_expected(filename, expected, tmp_path):
    # Examples are compiled with the bundled stdlib, matching `tuppu run`.
    files = stdlib_files() + [EXAMPLES_DIR / filename]
    binary = compile_files_to_binary(files, tmp_path, name=filename[:-4])
    result = subprocess.run([str(binary)], capture_output=True)
    assert result.returncode == expected["exit"], (
        f"{filename}: expected exit {expected['exit']}, got {result.returncode}"
    )
    if "stdout" in expected:
        assert result.stdout == expected["stdout"], (
            f"{filename}: stdout mismatch.\n"
            f"expected: {expected['stdout']!r}\n"
            f"actual:   {result.stdout!r}"
        )


def test_every_example_is_covered():
    """Gatekeeper: if someone drops a new .tpu into examples/, force
    them to register it here, and flag orphans in the registry."""
    actual = {p.name for p in EXAMPLES_DIR.glob("*.tpu")}
    declared = set(EXPECTED.keys())
    missing_from_map = actual - declared
    missing_from_disk = declared - actual
    assert not missing_from_map, f"examples without expected values: {missing_from_map}"
    assert not missing_from_disk, f"registered examples missing on disk: {missing_from_disk}"
