"""Tuppu command-line entry point.

Usage:
    python -m tuppu build file.tpu... [-o output] [--no-stdlib]
    python -m tuppu run   file.tpu... [--no-stdlib]
    python -m tuppu check file.tpu... [--no-stdlib]

`check` parses + typechecks without emitting IR — fast feedback for
"did I write valid Tuppu" without paying for codegen + linking.

By default the bundled stdlib (all of <repo>/stdlib/*.tpu) is included in
the compilation. Pass --no-stdlib to compile user files alone.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from .driver import check_files, compile_files_to_binary, stdlib_files
from .errors import CompileError, format_error


def _resolve_inputs(user_files: list[Path], include_stdlib: bool) -> list[Path]:
    """Stdlib first so user code can forward-reference it and collisions
    surface as duplicate-definition errors against user code, not stdlib.

    Each user input may be a file or a directory; directories are walked
    recursively for `.tpu` files (sorted for determinism). This is the
    foundation for project-shaped Tuppu code where one `tuppu run src/`
    invocation compiles every module under `src/`."""
    expanded: list[Path] = []
    for f in user_files:
        if f.is_dir():
            expanded.extend(sorted(f.rglob("*.tpu")))
        else:
            expanded.append(f)
    if include_stdlib:
        return stdlib_files() + expanded
    return expanded


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuppu")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, help_ in [("build", "compile files into a native binary"),
                        ("run",   "compile and execute immediately"),
                        ("check", "parse + typecheck only; no IR emission")]:
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

    # path -> source text for error rendering. Built lazily in the
    # except branch so happy-path compiles don't pay for it.
    def _read_source(path_str: str) -> str | None:
        for f in files:
            if str(f) == path_str:
                try:
                    return f.read_text()
                except OSError:
                    return None
        return None

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

        if args.cmd == "check":
            check_files(files)
            print(f"tuppu: ok ({len(files)} file{'s' if len(files) != 1 else ''})",
                  file=sys.stderr)
            return 0
    except CompileError as e:
        path = getattr(e, "path", None)
        text = _read_source(path) if path else None
        print(f"tuppu: {format_error(e, text)}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    sys.exit(main())
