"""Module / import support — v1.

v1 scope:
- `import x.y` and `from x.y import a, b as c` parse and validate.
- Unknown modules and unknown names are rejected at typecheck.
- Private (`_`-prefixed) top-level names are visible only within
  their declaring module; reads from another module error.

Out-of-scope for v1 (will land later):
- Per-module name resolution (the global flat namespace is preserved
  for backward compat with existing code that relies on it).
- Module-prefix LLVM mangling.
- Cross-module variant disambiguation (the LIMITATIONS.md flat-seal-
  variant bug).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tuppu.driver import (
    _module_path_for_label,
    check_sources,
    compile_files_to_binary,
)
from tuppu.errors import CompileError
from tuppu.lexer import Tok, lex
from tuppu.parser import parse
from tuppu import ast as A


def write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


# --- lexer / parser -------------------------------------------------------


def test_lexer_emits_import_and_from_tokens():
    toks = lex("import a.b\nfrom c import d as e\n")
    kinds = [t.kind for t in toks if t.kind not in (Tok.NEWLINE, Tok.EOF)]
    assert Tok.IMPORT in kinds
    assert Tok.FROM in kinds
    assert Tok.AS in kinds


def test_parser_wildcard_import():
    prog = parse(lex("import stdlib.list\nfn main() -> i32 { 0 }\n"))
    imports = [d for d in prog.decls if isinstance(d, A.ImportDecl)]
    assert len(imports) == 1
    imp = imports[0]
    assert imp.path == ["stdlib", "list"]
    assert imp.names is None


def test_parser_from_import_with_aliases():
    src = "from stdlib.map import Map, get as map_get\nfn main() -> i32 { 0 }\n"
    prog = parse(lex(src))
    imports = [d for d in prog.decls if isinstance(d, A.ImportDecl)]
    assert len(imports) == 1
    imp = imports[0]
    assert imp.path == ["stdlib", "map"]
    assert imp.names == [("Map", None), ("get", "map_get")]


def test_parser_import_as_alias():
    # `import x.y as z` parses and stores the alias on the ImportDecl.
    prog = parse(lex("import stdlib.list as l\nfn main() -> i32 { 0 }\n"))
    imp = prog.decls[0]
    assert isinstance(imp, A.ImportDecl)
    assert imp.path == ["stdlib", "list"]
    assert imp.names is None
    assert imp.wildcard_alias == "l"


def test_parser_dotted_module_path():
    prog = parse(lex("import a.b.c.d\nfn main() -> i32 { 0 }\n"))
    imp = prog.decls[0]
    assert isinstance(imp, A.ImportDecl)
    assert imp.path == ["a", "b", "c", "d"]


# --- driver: module path derivation --------------------------------------


def test_module_path_from_stdlib():
    assert _module_path_for_label("/repo/stdlib/list.tpu") == ("stdlib", "list")


def test_module_path_from_nested_stdlib():
    assert _module_path_for_label("/repo/stdlib/sub/foo.tpu") == ("stdlib", "sub", "foo")


def test_module_path_from_src():
    assert _module_path_for_label("/repo/src/parser.tpu") == ("parser",)


def test_module_path_from_nested_src():
    assert _module_path_for_label("/repo/src/sema/typecheck.tpu") == ("sema", "typecheck")


def test_module_path_for_synthetic_source():
    assert _module_path_for_label("<source>") == ()
    assert _module_path_for_label("") == ()


def test_module_path_unrooted_falls_to_root():
    # No 'src' or 'stdlib' anchor — collapse to root module so multi-
    # file scripts in tmp_path share one namespace (matches the
    # existing test_multifile expectations).
    assert _module_path_for_label("/tmp/scratch.tpu") == ()
    assert _module_path_for_label("/tmp/a.tpu") == ()
    assert _module_path_for_label("/tmp/b.tpu") == ()


# --- import validation ----------------------------------------------------


def test_import_unknown_module_errors(tmp_path):
    main = write(tmp_path, "src/main.tpu", "import a.nonexistent\nfn main() -> i32 { 0 }\n")
    with pytest.raises(CompileError, match="unknown module"):
        check_sources([(str(main), main.read_text())])


def test_from_import_unknown_name_errors(tmp_path):
    helper = write(tmp_path, "src/helper.tpu", "fn add(a: i64, b: i64) -> i64 { a + b }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "from helper import absent\nfn main() -> i32 { 0 }\n",
    )
    with pytest.raises(CompileError, match="no export named"):
        check_sources([
            (str(helper), helper.read_text()),
            (str(main), main.read_text()),
        ])


def test_from_import_private_name_errors(tmp_path):
    # A name starting with `_` cannot be `from`-imported even if it
    # exists in the source module.
    helper = write(tmp_path, "src/helper.tpu", "fn _hidden() -> i64 { 7 }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "from helper import _hidden\nfn main() -> i32 { 0 }\n",
    )
    with pytest.raises(CompileError, match="private name"):
        check_sources([
            (str(helper), helper.read_text()),
            (str(main), main.read_text()),
        ])


def test_wildcard_import_to_known_module_ok(tmp_path):
    helper = write(tmp_path, "src/helper.tpu", "fn add(a: i64, b: i64) -> i64 { a + b }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "import helper\nfn main() -> i32 { add(1, 2) as i32 }\n",
    )
    # Should not raise — wildcard `import helper` brings `add` into scope.
    check_sources([
        (str(helper), helper.read_text()),
        (str(main), main.read_text()),
    ])


def test_unimported_name_rejected_in_src(tmp_path):
    """Project-shaped code under `src/` requires explicit imports;
    cross-module names without an import are rejected with a targeted
    message that points at the fix."""
    helper = write(tmp_path, "src/helper.tpu", "fn add(a: i64, b: i64) -> i64 { a + b }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "fn main() -> i32 { add(1, 2) as i32 }\n",
    )
    with pytest.raises(CompileError, match="not in scope"):
        check_sources([
            (str(helper), helper.read_text()),
            (str(main), main.read_text()),
        ])


# --- visibility: `_`-prefix is private to declaring module ---------------


def test_private_fn_visible_within_module(tmp_path):
    helper = write(
        tmp_path, "src/helper.tpu",
        "fn _internal() -> i64 { 7 }\n"
        "fn add(a: i64, b: i64) -> i64 { a + b + _internal() }\n",
    )
    main = write(
        tmp_path, "src/main.tpu",
        "from helper import add\n"
        "fn main() -> i32 { add(1, 2) as i32 }\n",
    )
    binary = compile_files_to_binary([helper, main], tmp_path / "build", name="prog")
    assert subprocess.run([str(binary)]).returncode == 10


def test_private_fn_invisible_from_other_module(tmp_path):
    helper = write(tmp_path, "src/helper.tpu", "fn _internal() -> i64 { 7 }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "fn main() -> i32 { _internal() as i32 }\n",
    )
    with pytest.raises(CompileError, match="private to module"):
        check_sources([
            (str(helper), helper.read_text()),
            (str(main), main.read_text()),
        ])


def test_private_tablet_invisible_from_other_module(tmp_path):
    helper = write(
        tmp_path, "src/helper.tpu",
        "tablet _Hidden { x: i64 }\n",
    )
    main = write(
        tmp_path, "src/main.tpu",
        "fn main() -> i32 { step h: _Hidden = _Hidden { x: 1 }\n h.x as i32 }\n",
    )
    with pytest.raises(CompileError, match="private to module"):
        check_sources([
            (str(helper), helper.read_text()),
            (str(main), main.read_text()),
        ])


def test_public_fn_visible_across_modules(tmp_path):
    """Cross-module reference works through an explicit `from` import."""
    helper = write(tmp_path, "src/helper.tpu", "fn add(a: i64, b: i64) -> i64 { a + b }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "from helper import add\n"
        "fn main() -> i32 { add(40, 2) as i32 }\n",
    )
    binary = compile_files_to_binary([helper, main], tmp_path / "build", name="prog")
    assert subprocess.run([str(binary)]).returncode == 42


# --- cross-module variant disambiguation (LIMITATIONS.md fix) ------------


def test_seals_in_different_modules_can_share_variant_names(tmp_path):
    """`seal A { X }; seal B { X }` works as long as the seals live
    in different modules. Previously rejected by the flat variant
    table — the LIMITATIONS.md gap. Each importer picks the variant
    via its own visible seals (only the seal it imports brings X
    into scope)."""
    foo = write(
        tmp_path, "src/foo.tpu",
        "seal Foo { X, Y }\n"
        "fn make_foo() -> Foo { X }\n",
    )
    bar = write(
        tmp_path, "src/bar.tpu",
        "seal Bar { X, Z }\n"
        "fn make_bar() -> Bar { X }\n",
    )
    main = write(
        tmp_path, "src/main.tpu",
        "from foo import Foo, make_foo\n"
        "from bar import Bar, make_bar\n"
        "fn main() -> i32 {\n"
        "  step f: Foo = make_foo()\n"
        "  step b: Bar = make_bar()\n"
        "  match f { X => 0, Y => 1 }\n"
        "}\n",
    )
    binary = compile_files_to_binary([foo, bar, main], tmp_path / "build", name="prog")
    assert subprocess.run([str(binary)]).returncode == 0


def test_variant_ambiguity_when_both_seals_imported(tmp_path):
    """If a use site imports two seals that each have variant X, using
    bare `X` is ambiguous and rejected with a message that names the
    candidate seals."""
    foo = write(tmp_path, "src/foo.tpu", "seal Foo { X }\n")
    bar = write(tmp_path, "src/bar.tpu", "seal Bar { X }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "from foo import Foo\n"
        "from bar import Bar\n"
        "fn main() -> i32 {\n"
        "  step f: Foo = X\n"
        "  0\n"
        "}\n",
    )
    with pytest.raises(CompileError, match="multiple seals"):
        check_sources([
            (str(foo), foo.read_text()),
            (str(bar), bar.read_text()),
            (str(main), main.read_text()),
        ])


def test_variant_unimported_seal_errors_with_hint(tmp_path):
    """Using a variant whose seal isn't imported gives a targeted
    'add `from X import Seal`' message, not a generic 'undefined name'."""
    foo = write(tmp_path, "src/foo.tpu", "seal Foo { X, Y }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "fn main() -> i32 { step f: i64 = 0\n match f { 0 => 0, _ => 1 } }\n",
    )
    # Compile succeeds (main doesn't reference X). Now try to use X
    # without importing Foo — should error with the hint.
    main2 = write(
        tmp_path, "src/main.tpu",
        "fn main() -> i32 { step _: i64 = 0\n 0 }\n",
    )
    main3 = write(
        tmp_path, "src/main3.tpu",
        "seal Local { Q }\n"
        "fn main() -> i32 {\n"
        "  step l: Local = Q\n"
        "  step l2: Local = X\n"  # X is from foo, not imported
        "  0\n"
        "}\n",
    )
    with pytest.raises(CompileError, match="seal isn't in scope here"):
        check_sources([
            (str(foo), foo.read_text()),
            (str(main3), main3.read_text()),
        ])


# --- module-qualified access (`x.foo(args)`) -----------------------------


def test_module_qualified_call_via_import_as(tmp_path):
    """`import x.y as z` enables qualified access `z.foo(args)`. The
    aliased form does NOT pollute the local namespace — `foo` is only
    reachable via `z.foo`."""
    helper = write(tmp_path, "src/helper.tpu", "fn add(a: i64, b: i64) -> i64 { a + b }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "import helper as h\n"
        "fn main() -> i32 { h.add(40, 2) as i32 }\n",
    )
    binary = compile_files_to_binary([helper, main], tmp_path / "build", name="prog")
    assert subprocess.run([str(binary)]).returncode == 42


def test_module_qualified_call_via_wildcard_import(tmp_path):
    """`import x` (wildcard) brings public names in unprefixed AND
    registers `x` as a qualifier so `x.foo(args)` is also valid."""
    helper = write(tmp_path, "src/helper.tpu", "fn add(a: i64, b: i64) -> i64 { a + b }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "import helper\n"
        "fn main() -> i32 { helper.add(40, 2) as i32 }\n",
    )
    binary = compile_files_to_binary([helper, main], tmp_path / "build", name="prog")
    assert subprocess.run([str(binary)]).returncode == 42


def test_qualified_access_to_unknown_member_errors(tmp_path):
    """`h.no_such_fn` after `import helper as h` is a clear error
    (module exists, but the name doesn't)."""
    helper = write(tmp_path, "src/helper.tpu", "fn add(a: i64, b: i64) -> i64 { a + b }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "import helper as h\n"
        "fn main() -> i32 { h.no_such_fn(1, 2) as i32 }\n",
    )
    with pytest.raises(CompileError, match="has no public name"):
        check_sources([
            (str(helper), helper.read_text()),
            (str(main), main.read_text()),
        ])


# --- regression: existing single-source tests keep working ---------------


def test_single_source_still_compiles():
    """A single-source compile (no file path) lives in the root module
    and has no imports; visibility checks must not fire on built-in
    names like `str`."""
    from tuppu.driver import compile_to_ir
    ir = compile_to_ir(
        'fn main() -> i32 { println("hi") 0 }\n'
    )
    assert "define" in ir
