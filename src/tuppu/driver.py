"""Tuppu compiler driver.

Orchestrates the full pipeline:

    source text  ->  tokens  ->  AST  ->  LLVM IR  ->  object  ->  binary

Supports two input shapes:

- Single source string — the original API, `compile_to_binary(src, ...)`.
- Multiple files — `compile_files_to_binary([path1, path2, ...], ...)`,
  where top-level decls across all files share a single namespace and
  forward references across files Just Work. This is what makes a
  Tuppu-in-Tuppu stdlib possible.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from llvmlite import binding as llvm

from . import ast as A
from .codegen import codegen
from .errors import CompileWarning
from .lexer import LexError, lex
from .parser import ParseError, parse
from .typecheck import CheckError, check


# --- IR generation --------------------------------------------------------

def _builtin_decls() -> list[A.Decl]:
    """Decls injected into every compilation. Currently the `str` tablet:
    `{ ptr: *u8, len: i64, cap: i64 }`. The `cap` sentinel discriminates
    ownership: cap == 0 means borrowed (string literals pointing into an
    immortal global); cap > 0 means heap-owned (freed by the scope-exit
    cleanup). Keeping the definition here (rather than in a .tpu stdlib
    file) means string literals work even with `--no-stdlib`."""
    return [
        A.StructDecl(
            name="str",
            fields=[
                ("ptr", A.TypePointer(element=A.TypeName(name="u8"))),
                ("len", A.TypeName(name="i64")),
                ("cap", A.TypeName(name="i64")),
            ],
        ),
    ]


def _parse_labeled(sources: list[tuple[str, str]]) -> A.Program:
    """Parse a list of (label, source_text) pairs and merge their top-level
    decls into one Program. Labels are used only for error-message context.
    Built-in declarations (e.g. the `str` seal) are prepended automatically."""
    decls: list[A.Decl] = list(_builtin_decls())
    # Built-in `str` tablet is auto-prepended above.
    for label, text in sources:
        try:
            prog = parse(lex(text))
        except (LexError, ParseError) as e:
            raise type(e)(f"{label}: {e.message}", e.line, e.col) from None
        decls.extend(prog.decls)
    return A.Program(decls=decls)


def compile_sources_to_ir(sources: list[tuple[str, str]]) -> str:
    """Generate LLVM IR text from a list of (label, source_text) pairs.
    Any non-fatal warnings from the type checker are written to stderr."""
    prog = _parse_labeled(sources)
    checker = check(prog)
    _emit_warnings(checker.warnings)
    return str(codegen(prog, checker))


def _emit_warnings(warnings: list[CompileWarning]) -> None:
    for w in warnings:
        print(w.format(), file=sys.stderr)


def compile_to_ir(source: str) -> str:
    """Generate LLVM IR text from a single source string."""
    return compile_sources_to_ir([("<source>", source)])


# --- object and link ------------------------------------------------------

def emit_object(ir_text: str, out: Path) -> None:
    """Lower Tuppu-emitted LLVM IR to a native object file.

    NOTE: no LLVM optimization passes are run on the emitted IR. The
    standard pipelines (`-O1`+) all include passes that don't know
    about our shadow-stack GC roots:

      - SROA / mem2reg promote allocas whose uses are loads/stores,
        but they treat `__tuppu_gc_push_root(slot, ...)` as an opaque
        extern call and don't track the slot as a persistent address —
        so a rooted alloca gets promoted, the GC traces freed memory.
      - tail_call_elimination rewrites recursive calls to reuse the
        caller's frame, which desyncs our push/pop accounting.

    The fix is to migrate to LLVM's first-class GC framework
    (`gc "shadow-stack"` fn attribute + `@llvm.gcroot` intrinsics),
    which makes the optimizer GC-aware. Tracked in LIMITATIONS.md
    as a [blocker] for any opt-level work."""
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    ref = llvm.parse_assembly(ir_text)
    ref.verify()
    tm = llvm.Target.from_default_triple().create_target_machine(reloc="pic")
    out.write_bytes(tm.emit_object(ref))


def _runtime_object(build_dir: Path) -> Path:
    """Compile the bundled GC runtime to an object file, caching the
    result in `build_dir`. Rebuilt if the source file is newer."""
    src = Path(__file__).resolve().parents[2] / "runtime" / "tuppu_gc.c"
    obj = build_dir / "tuppu_gc.o"
    if obj.exists() and obj.stat().st_mtime >= src.stat().st_mtime:
        return obj
    subprocess.run(
        ["clang", "-c", "-O2", "-Wall", "-o", str(obj), str(src)],
        check=True,
    )
    return obj


def link(obj: Path, out: Path) -> None:
    runtime_obj = _runtime_object(obj.parent)
    subprocess.run(
        ["clang", str(obj), str(runtime_obj), "-o", str(out)],
        check=True,
    )


# --- end-to-end entry points ----------------------------------------------

def _compile_to_binary(
    sources: list[tuple[str, str]], build_dir: Path, name: str,
) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    ir_text = compile_sources_to_ir(sources)
    (build_dir / f"{name}.ll").write_text(ir_text)
    emit_object(ir_text, build_dir / f"{name}.o")
    binary = build_dir / name
    link(build_dir / f"{name}.o", binary)
    return binary


def compile_to_binary(source: str, build_dir: Path, name: str = "a") -> Path:
    """Compile a single source string to a native binary."""
    return _compile_to_binary([("<source>", source)], build_dir, name)


def compile_files_to_binary(
    files: list[Path], build_dir: Path, name: str = "a",
) -> Path:
    """Compile multiple .tpu files as a single compilation unit. Decls
    across files share a namespace and can reference each other freely."""
    sources = [(str(p), p.read_text()) for p in files]
    return _compile_to_binary(sources, build_dir, name)


# --- stdlib discovery -----------------------------------------------------

def stdlib_dir() -> Path:
    """Absolute path to the bundled Tuppu stdlib directory.
    Assumes editable / source install: <repo>/src/tuppu/driver.py's
    grandparent-of-parent is the repo root containing <repo>/stdlib."""
    return Path(__file__).resolve().parents[2] / "stdlib"


def stdlib_files() -> list[Path]:
    """All .tpu files in the bundled stdlib, sorted for deterministic order."""
    d = stdlib_dir()
    if not d.is_dir():
        return []
    return sorted(d.glob("*.tpu"))
