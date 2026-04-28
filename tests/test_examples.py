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
    "scribe.tpu":            {
        "exit": 0,
        "stdout": (
            b"Day 1, Eanna Granary:\n"
            b"Eanna Granary\n"
            b"Day 1\n"
            b"barley: 3600 (1 0 0)\n"
            b"emmer: 150 (2 30)\n"
            b"sesame: 60 (1 0)\n"
            b"total items: 3\n"
            b"Eanna\n"
        ),
    },
    "tcp_bind.tpu":          {
        "exit": 0,
        "stdout": b"bind rc = 0\n",
    },
    "higher_order.tpu":      {
        "exit": 0,
        "stdout": (
            b"sq(1) = 1\n"
            b"sq(2) = 4\n"
            b"sq(3) = 9\n"
            b"cb(1) = 1\n"
            b"cb(2) = 8\n"
            b"cb(3) = 27\n"
            b"incr(41) =42\n"
        ),
    },
    "linked_list.tpu":       {
        "exit": 0,
        "stdout": (
            b"length: 5\n"
            b"items:\n"
            b"1\n2\n3\n4\n5\n"
            b"found 3; its successor is 4\n"
            b"99 not in list\n"
        ),
    },
    "scribe_ledger.tpu":     {
        "exit": 0,
        "stdout": (
            b"today's offerings (6 total):\n"
            b"  barley\n"
            b"  emmer\n"
            b"  barley\n"
            b"  sesame\n"
            b"  emmer\n"
            b"  barley\n"
            b"tally:\n"
            b"  barley = 10\n"
            b"  emmer = 6\n"
            b"  sesame = 1\n"
            b"found barley: true\n"
            b"found wheat:  false\n"
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
    "omens.tpu":             {
        "exit": 0,
        "stdout": (
            b"the scribe reads:\n"
            b"favorable: +3/2\n"
            b"ambiguous: ignored\n"
            b"unfavorable: -1/4\n"
            b"favorable: +2/1\n"
            b"ambiguous: ignored\n"
            b"unfavorable: -3/4\n"
            b"tally: 5/2\n"
            b"the gods favor this venture\n"
        ),
    },
    # 1500+ lines of recursive-descent Lua interpreter — exercises
    # everything heavy: strs interned in seal payloads stored in
    # tablets-of-structs, mutually recursive ASTs, closures captured
    # in seals, repeated `run()` calls so cross-call GC integrity
    # gets stressed. Historically the worst regression magnet in
    # the project; if a test catches the next GC bug, this one will.
    "lua_interp.tpu":        {
        "exit": 0,
        "stdout": (
            b"=== Tuppu-hosted Lua interpreter ===\n"
            b"\n"
            b"------------------------------------\n"
            b"> print(1 + 2 * 3)\n"
            b"---\n"
            b"7\n"
            b"------------------------------------\n"
            b"> print((1 + 2) * 3)\n"
            b"---\n"
            b"9\n"
            b"------------------------------------\n"
            b"> local x = 5  print(x * 2)\n"
            b"---\n"
            b"10\n"
            b"------------------------------------\n"
            b"> local s = \"hello, \" .. \"world\"  print(s)\n"
            b"---\n"
            b"hello, world\n"
            b"------------------------------------\n"
            b"> if 10 > 5 then print(\"big\") else print(\"small\") end\n"
            b"---\n"
            b"big\n"
            b"------------------------------------\n"
            b"> local i = 1  while i <= 5 do print(i)  i = i + 1 end\n"
            b"---\n"
            b"1\n2\n3\n4\n5\n"
            b"------------------------------------\n"
            b"> local function sq(x) return x * x end  print(sq(7))\n"
            b"---\n"
            b"49\n"
            b"------------------------------------\n"
            b"> local function fact(n) if n <= 1 then return 1 else return n * fact(n - 1) end end  print(fact(5))\n"
            b"---\n"
            b"120\n"
            b"------------------------------------\n"
            b"> local function fib(n) if n < 2 then return n else return fib(n - 1) + fib(n - 2) end end  print(fib(10))\n"
            b"---\n"
            b"55\n"
            b"------------------------------------\n"
            b"> local function makeAdder(x) return function(y) return x + y end end  local add5 = makeAdder(5)  print(add5(3))  print(add5(100))\n"
            b"---\n"
            b"8\n105\n"
            b"------------------------------------\n"
            b"> local function apply(f, v) return f(v) end  local function dbl(n) return n * 2 end  print(apply(dbl, 21))\n"
            b"---\n"
            b"42\n"
            b"------------------------------------\n"
            b"> local function counter() local n = 0  local function step() n = n + 1  return n end  return step end  local c = counter()  print(c())  print(c())  print(c())\n"
            b"---\n"
            b"1\n2\n3\n"
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
