"""Multi-file compilation: decls across files share one namespace."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tuppu.codegen import CodegenError
from tuppu.errors import CompileError
from tuppu.driver import compile_files_to_binary


def write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_function_called_across_files(tmp_path):
    a = write(tmp_path, "a.tpu", "fn double(n: i64) -> i64 { n * 2 }\n")
    b = write(tmp_path, "b.tpu", "fn main() -> i32 { double(21) }\n")
    binary = compile_files_to_binary([a, b], tmp_path / "build", name="prog")
    result = subprocess.run([str(binary)])
    assert result.returncode == 42


def test_forward_reference_across_files(tmp_path):
    # main calls double, which is declared in the OTHER file passed first.
    main = write(tmp_path, "main.tpu", "fn main() -> i32 { double(21) }\n")
    helper = write(tmp_path, "helper.tpu", "fn double(n: i64) -> i64 { n * 2 }\n")
    binary = compile_files_to_binary([main, helper], tmp_path / "build", name="prog")
    assert subprocess.run([str(binary)]).returncode == 42


def test_stdout_across_files(tmp_path):
    # Helper uses println; main calls helper.
    helper = write(
        tmp_path, "greet.tpu",
        'fn greet() { println("hi") }\n',
    )
    main = write(
        tmp_path, "main.tpu",
        "fn main() -> i32 { greet()\n 0 }\n",
    )
    binary = compile_files_to_binary([helper, main], tmp_path / "build", name="prog")
    r = subprocess.run([str(binary)], capture_output=True)
    assert r.stdout == b"hi\n"


def test_rat_helper_in_separate_file(tmp_path):
    # Preview of what stdlib dogfooding will look like.
    lib = write(tmp_path, "rat.tpu", (
        "fn rat_reciprocal(x: rat) -> rat { rat(x.den, x.num) }\n"
        "fn rat_double(x: rat) -> rat { x + x }\n"
    ))
    main = write(tmp_path, "main.tpu", (
        "fn main() -> i32 {\n"
        "  println(rat_reciprocal(1;30))\n"    # 3/2 → 2/3
        "  println(rat_double(rat(1, 3)))\n"   # 2/3
        "  0\n"
        "}\n"
    ))
    binary = compile_files_to_binary([lib, main], tmp_path / "build", name="prog")
    r = subprocess.run([str(binary)], capture_output=True)
    assert r.stdout == b"2/3\n2/3\n"


def test_duplicate_function_across_files_errors(tmp_path):
    a = write(tmp_path, "a.tpu", "fn foo() -> i64 { 1 }\n")
    b = write(tmp_path, "b.tpu", "fn foo() -> i64 { 2 }\n")
    main = write(tmp_path, "main.tpu", "fn main() -> i32 { 0 }\n")
    with pytest.raises(CompileError, match="duplicate"):
        compile_files_to_binary([a, b, main], tmp_path / "build", name="prog")


def test_parse_error_includes_filename(tmp_path):
    from tuppu.parser import ParseError
    bad = write(tmp_path, "broken.tpu", "fn main() -> i32 { 1 + }\n")
    with pytest.raises(ParseError, match="broken.tpu"):
        compile_files_to_binary([bad], tmp_path / "build", name="prog")


# --- CLI tests --------------------------------------------------------------

def _cli(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tuppu", *args],
        capture_output=True, **kwargs,
    )


def test_cli_build_then_run(tmp_path):
    src = write(tmp_path, "hello.tpu", 'fn main() -> i32 { println("hi from cli")\n 0 }')
    out = tmp_path / "hello"
    r = _cli(["build", str(src), "-o", str(out)])
    assert r.returncode == 0, r.stderr.decode()
    assert out.exists()
    r2 = subprocess.run([str(out)], capture_output=True)
    assert r2.stdout == b"hi from cli\n"


def test_cli_run_returns_program_exit_code(tmp_path):
    src = write(tmp_path, "exit42.tpu", "fn main() -> i32 { 42 }")
    r = _cli(["run", str(src)])
    assert r.returncode == 42


def test_cli_build_multiple_files(tmp_path):
    a = write(tmp_path, "a.tpu", "fn seven() -> i64 { 7 }\n")
    b = write(tmp_path, "b.tpu", "fn main() -> i32 { seven() * 6 }\n")
    out = tmp_path / "prog"
    r = _cli(["build", str(a), str(b), "-o", str(out)])
    assert r.returncode == 0, r.stderr.decode()
    assert subprocess.run([str(out)]).returncode == 42
