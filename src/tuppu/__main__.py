"""Tuppu command-line entry point.

Usage:
    python -m tuppu build file.tpu... [-o output] [--no-stdlib]
    python -m tuppu run   file.tpu... [--no-stdlib]

By default the bundled stdlib (all of <repo>/stdlib/*.tpu) is included in
the compilation. Pass --no-stdlib to compile user files alone.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from .driver import compile_files_to_binary, stdlib_files
from .errors import CompileError


def _resolve_inputs(user_files: list[Path], include_stdlib: bool) -> list[Path]:
    """Stdlib first so user code can forward-reference it and collisions
    surface as duplicate-definition errors against user code, not stdlib."""
    if include_stdlib:
        return stdlib_files() + list(user_files)
    return list(user_files)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuppu")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, help_ in [("build", "compile files into a native binary"),
                        ("run",   "compile and execute immediately")]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("files", nargs="+", type=Path, help=".tpu source files")
        s.add_argument(
            "--no-stdlib", action="store_true",
            help="compile without the bundled Tuppu stdlib",
        )
        if name == "build":
            s.add_argument(
                "-o", "--output", type=Path, default=Path("./a.out"),
                help="output path (default: ./a.out)",
            )

    args = p.parse_args(argv)
    files = _resolve_inputs(args.files, include_stdlib=not args.no_stdlib)

    try:
        if args.cmd == "build":
            out = args.output
            out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as td:
                binary = compile_files_to_binary(files, Path(td), name=out.stem or "a")
                out.write_bytes(binary.read_bytes())
                out.chmod(0o755)
            print(f"tuppu: wrote {out}", file=sys.stderr)
            return 0

        if args.cmd == "run":
            with tempfile.TemporaryDirectory() as td:
                binary = compile_files_to_binary(files, Path(td), name="a")
                return subprocess.run([str(binary)]).returncode
    except CompileError as e:
        print(f"tuppu: {e}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    sys.exit(main())
