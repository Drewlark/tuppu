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
    decls into one Program. Labels are file paths; on lex/parse errors
    they're attached to the error as `e.path` so the driver can render
    source context. Type-checker and codegen errors don't currently get
    the path attached — see LIMITATIONS.md — they fall back to the bare
    line:col format. Built-in declarations (e.g. the `str` seal) are
    prepended automatically.

    Each label is also mapped to a module path (a tuple of segments
    derived from the file's location relative to the project root or
    stdlib) and stamped onto every parsed decl in the program's
    `module_of` sideband. Built-in injections live in the root module
    (empty tuple). Single-string sources (the simple test path) get a
    synthetic root-module label."""
    decls: list[A.Decl] = list(_builtin_decls())
    module_of: dict[int, tuple[str, ...]] = {}
    for label, text in sources:
        try:
            prog = parse(lex(text))
        except (LexError, ParseError) as e:
            e.path = label
            raise
        mod_path = _module_path_for_label(label)
        for d in prog.decls:
            module_of[id(d)] = mod_path
        decls.extend(prog.decls)
    return A.Program(decls=decls, module_of=module_of)


def _module_path_for_label(label: str) -> tuple[str, ...]:
    """Derive a dotted module path (as a tuple of segments) from a
    file's label. Labels are absolute or repo-relative paths to .tpu
    files; the path is taken relative to its containing 'src/' or
    'stdlib/' directory, with the .tpu extension stripped.

    Examples:
        /repo/stdlib/list.tpu              -> ('stdlib', 'list')
        /repo/stdlib/sub/foo.tpu           -> ('stdlib', 'sub', 'foo')
        /repo/src/parser.tpu               -> ('parser',)
        /repo/src/sema/typecheck.tpu       -> ('sema', 'typecheck')
        /tmp/a.tpu                         -> ()    (root — no anchor)
        /tmp/b.tpu                         -> ()    (root — same)
        <source>                           -> ()    (root)

    Files outside any recognized root (`stdlib/` or `src/`) all collapse
    to the root module so multi-file scripts in tmp_path behave like
    one program (matching the existing single-namespace expectation
    test_multifile depends on). Project-shaped code that wants
    real module isolation lives under `src/` or `stdlib/`."""
    if not label or label == "<source>":
        return ()
    p = Path(label)
    parts = list(p.parts)
    # Strip .tpu suffix off the final segment.
    stem = p.stem
    parts[-1] = stem
    # Anchor at the first 'stdlib' or 'src' segment we see; any path
    # prefix above that is the project root and isn't part of the
    # module name. For 'src/...' the 'src' itself is dropped (it's a
    # build convention, not a module). For 'stdlib/...' we keep
    # 'stdlib' as the leading segment so user-imported names like
    # 'stdlib.list' line up with how callers spell them.
    for i, seg in enumerate(parts):
        if seg == "stdlib":
            return tuple(parts[i:])
        if seg == "src":
            return tuple(parts[i + 1:])
    # No recognized root — root module. Multi-file ad-hoc scripts share
    # one namespace.
    return ()


def compile_sources_to_ir(sources: list[tuple[str, str]]) -> str:
    """Generate LLVM IR text from a list of (label, source_text) pairs.
    Any non-fatal warnings from the type checker are written to stderr.

    In `TUPPU_GC_FRAMEWORK=llvm` mode also injects the
    `gc "shadow-stack"` attribute into every Tuppu-emitted fn that
    queued at least one `@llvm.gcroot` — see `_inject_gc_strategy`."""
    prog = _parse_labeled(sources)
    checker = check(prog)
    _emit_warnings(checker.warnings)
    cg = _gen_module(prog, checker)
    text = str(cg.module)
    if cg._gc_mode == "llvm" and cg._fns_needing_gc_attr:
        text = _inject_gc_strategy(text, cg._fns_needing_gc_attr)
    return text


def _gen_module(prog: A.Program, checker):
    """Run codegen and return the Codegen instance (not just its IR
    module) so the driver can read post-codegen state — currently the
    set of fns that need `gc "shadow-stack"` injected into IR text.
    Mirrors `codegen()` but exposes the holder."""
    from .codegen import Codegen
    cg = Codegen(checker=checker)
    cg.gen(prog)
    return cg


# Pre-compiled regex catches `define <type> @"<name>"(...)` lines, with
# or without the leading `internal`/`linkonce_odr`/etc. linkage marker.
# llvmlite always emits in the `define <linkage?> <ret> @"<name>"(<args>)`
# form so the trailing `\s*$|\s*\{` lets us match before either the
# block opening brace or end-of-line (some llvmlite versions wrap).
import re as _re

_DEFINE_LINE_RE = _re.compile(
    r'^(define\b[^@]*@"(?P<name>[^"]+)"\([^)]*\))(?P<rest>.*)$',
    _re.MULTILINE,
)


def _inject_gc_strategy(ir_text: str, fns: set[str]) -> str:
    """Inject `gc "shadow-stack"` into the `define ...` line for every
    fn in `fns`. llvmlite (as of 0.47) doesn't expose `Function.gc` on
    its IR builder, so we post-process the textual IR before handoff to
    the LLVM lowering machinery. The attribute lives between the
    closing `)` of the param list and the opening `{` of the body — or
    at end-of-line if the body opens on the next line.

    Idempotent: if the attribute is already present (e.g. driver was
    re-run on the same text) we don't double-inject. Externs (`declare
    ...`) don't match the regex so they're left alone — the attribute
    only applies to defined fns."""
    def repl(m: _re.Match[str]) -> str:
        prefix, name, rest = m.group(1), m.group("name"), m.group("rest")
        if name not in fns:
            return m.group(0)
        if 'gc "shadow-stack"' in rest:
            return m.group(0)
        return f'{prefix} gc "shadow-stack"{rest}'
    return _DEFINE_LINE_RE.sub(repl, ir_text)


def check_sources(sources: list[tuple[str, str]]) -> None:
    """Parse and typecheck a list of (label, source_text) pairs without
    emitting IR. Used by `tuppu check` for fast feedback. Raises
    CompileError on any lex / parse / type problem; prints warnings
    to stderr otherwise."""
    prog = _parse_labeled(sources)
    checker = check(prog)
    _emit_warnings(checker.warnings)


def check_files(paths: list[Path]) -> None:
    """File-list flavor of check_sources — mirrors compile_files_to_binary."""
    sources = [(str(p), p.read_text()) for p in paths]
    check_sources(sources)


def _emit_warnings(warnings: list[CompileWarning]) -> None:
    for w in warnings:
        print(w.format(), file=sys.stderr)


def compile_to_ir(source: str) -> str:
    """Generate LLVM IR text from a single source string."""
    return compile_sources_to_ir([("<source>", source)])


# --- object and link ------------------------------------------------------

def emit_object(ir_text: str, out: Path) -> None:
    """Lower Tuppu-emitted LLVM IR to a native object file. No
    optimization passes run — see GC_REQS.md and LIMITATIONS.md
    for why."""
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
