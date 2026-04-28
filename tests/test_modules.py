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


def test_module_qualified_type_in_annotation(tmp_path):
    """`mod.Tablet` works as a type-position name after `import mod`,
    parallel to expression-level `mod.foo(args)`. The struct literal
    uses the unqualified form (wildcard import brought it into local
    scope) — qualified-name struct literals would need parser support
    in a follow-up."""
    other = write(
        tmp_path, "src/other.tpu",
        "tablet Counter { x: i64 }\n",
    )
    main = write(
        tmp_path, "src/main.tpu",
        "import other\n"
        "fn make() -> other.Counter { Counter { x: 7 } }\n"
        "fn main() -> i32 {\n"
        "  step c: other.Counter = make()\n"
        "  c.x as i32\n"
        "}\n",
    )
    binary = compile_files_to_binary([other, main], tmp_path / "build", name="prog")
    assert subprocess.run([str(binary)]).returncode == 7


def test_module_qualified_generic_type(tmp_path):
    """`mod.Foo<T>` for generic tablets after `import mod`. The
    annotation pins T; the struct literal infers from context."""
    other = write(
        tmp_path, "src/other.tpu",
        "tablet Box<T> { value: T }\n",
    )
    main = write(
        tmp_path, "src/main.tpu",
        "import other\n"
        "fn main() -> i32 {\n"
        "  step b: other.Box<i64> = Box { value: 42 }\n"
        "  b.value as i32\n"
        "}\n",
    )
    binary = compile_files_to_binary([other, main], tmp_path / "build", name="prog")
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


# --- topological order + cycle detection --------------------------------


def test_circular_import_rejected(tmp_path):
    """Mutual `import` between two modules is rejected at typecheck.
    Strategy doc lists circular imports as a non-goal for v1; the
    rejection keeps later passes' single-traversal model honest."""
    a = write(tmp_path, "src/a.tpu", "import b\nfn fa() -> i64 { 1 }\n")
    b = write(tmp_path, "src/b.tpu", "import a\nfn fb() -> i64 { 2 }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "from a import fa\nfn main() -> i32 { fa() as i32 }\n",
    )
    with pytest.raises(CompileError, match="circular import"):
        check_sources([
            (str(a), a.read_text()),
            (str(b), b.read_text()),
            (str(main), main.read_text()),
        ])


def test_self_import_rejected(tmp_path):
    """A module that imports itself trips the cycle detector — the
    self-edge is the smallest cycle."""
    a = write(
        tmp_path, "src/a.tpu",
        "import a\n"
        "fn helper() -> i64 { 7 }\n"
        "fn main() -> i32 { helper() as i32 }\n",
    )
    with pytest.raises(CompileError, match="circular import"):
        check_sources([(str(a), a.read_text())])


# --- edubba additive across files in same module ------------------------


def test_edubba_additive_within_one_file_in_a_module():
    """Two `edubba T { ... }` blocks in the same file merge — methods
    from both blocks are reachable on a `T` value. This is the v1
    additive-edubba feature; directory-as-module multi-file additive
    edubbas would need a `mod foo;`-style module declaration syntax
    that v1 deliberately avoids."""
    from tuppu.driver import compile_to_ir
    src = """
    tablet Box { x: i64 }
    edubba Box { fn one(self) -> i64 { self.x } }
    edubba Box { fn two(self) -> i64 { self.x * 2 } }
    fn main() -> i32 {
      step b: Box = Box { x: 21 }
      (b.one() + b.two()) as i32
    }
    """
    ir = compile_to_ir(src)
    # Both methods get materialised as fns on Box.
    assert "Box__one" in ir
    assert "Box__two" in ir


def test_edubba_cannot_extend_foreign_tablet(tmp_path):
    """An edubba block declared outside the host tablet's module is
    rejected — modules don't get to bolt methods onto someone else's
    tablet."""
    host = write(tmp_path, "src/host.tpu", "tablet Foo { x: i64 }\n")
    intruder = write(
        tmp_path, "src/intruder.tpu",
        "from host import Foo\n"
        "edubba Foo { fn bad(self) -> i64 { 0 } }\n",
    )
    with pytest.raises(CompileError, match="can only be added in the host"):
        check_sources([
            (str(host), host.read_text()),
            (str(intruder), intruder.read_text()),
        ])


# --- cross-module same-name decls (LLVM mangling) -----------------------


def test_two_modules_can_each_declare_same_fn_name(tmp_path):
    """`fn helper()` in two different modules coexists. Each module's
    `helper` is mangled to `__M_<mod>__helper` at the global symbol
    layer; importers reach the right one through their visible scope."""
    foo = write(tmp_path, "src/foo.tpu", "fn helper() -> i64 { 10 }\n")
    bar = write(tmp_path, "src/bar.tpu", "fn helper() -> i64 { 32 }\n")
    main = write(
        tmp_path, "src/main.tpu",
        "from foo import helper as foo_helper\n"
        "from bar import helper as bar_helper\n"
        "fn main() -> i32 { (foo_helper() + bar_helper()) as i32 }\n",
    )
    binary = compile_files_to_binary([foo, bar, main], tmp_path / "build", name="prog")
    assert subprocess.run([str(binary)]).returncode == 42


def test_edubba_works_on_cross_module_collision_tablets(tmp_path):
    """Two modules each declare `tablet Counter` AND an
    `edubba Counter { ... }` on it. Each tablet's edubba methods need
    to land in distinct global symbols (`__M_foo__Counter__bump` vs
    `__M_bar__Counter__bump`) — without the host's flat name flowing
    through `_lower_edubbas`, the host-not-found check spuriously
    fired and cross-module same-name edubbas couldn't ship methods at
    all."""
    foo = write(
        tmp_path, "src/foo.tpu",
        "tablet Counter { x: i64 }\n"
        "edubba Counter { fn bump(self) -> i64 { self.x + 1 } }\n",
    )
    bar = write(
        tmp_path, "src/bar.tpu",
        "tablet Counter { y: i64 }\n"
        "edubba Counter { fn bump(self) -> i64 { self.y * 10 } }\n",
    )
    main = write(
        tmp_path, "src/main.tpu",
        "from foo import Counter as FC\n"
        "from bar import Counter as BC\n"
        "fn main() -> i32 {\n"
        "  step f: FC = FC { x: 1 }\n"
        "  step b: BC = BC { y: 4 }\n"
        "  (f.bump() + b.bump()) as i32\n"
        "}\n",
    )
    binary = compile_files_to_binary([foo, bar, main], tmp_path / "build", name="prog")
    assert subprocess.run([str(binary)]).returncode == 42


def test_two_modules_can_each_declare_same_tablet_name(tmp_path):
    """`tablet Foo` in two different modules coexists end-to-end —
    each becomes a distinct LLVM identified type via module-prefix
    mangling. The two `Foo`s are distinct nominal types: a `foo.Foo`
    value can't be assigned to a `bar.Foo` slot."""
    foo = write(
        tmp_path, "src/foo.tpu",
        "tablet Foo { x: i64 }\n"
        "fn make_foo_x() -> Foo { Foo { x: 7 } }\n",
    )
    bar = write(
        tmp_path, "src/bar.tpu",
        "tablet Foo { y: i64 }\n"
        "fn make_bar_y() -> Foo { Foo { y: 35 } }\n",
    )
    main = write(
        tmp_path, "src/main.tpu",
        "from foo import Foo as FFoo, make_foo_x\n"
        "from bar import Foo as BFoo, make_bar_y\n"
        "fn main() -> i32 {\n"
        "  step a: FFoo = make_foo_x()\n"
        "  step b: BFoo = make_bar_y()\n"
        "  (a.x + b.y) as i32\n"
        "}\n",
    )
    binary = compile_files_to_binary([foo, bar, main], tmp_path / "build", name="prog")
    assert subprocess.run([str(binary)]).returncode == 42


# --- diagnostics: pretty-printed mangled names --------------------------


def test_error_messages_prettify_mangled_type_names(tmp_path):
    """A type mismatch involving cross-module same-name tablets
    surfaces them as `foo.Counter` / `bar.Counter` in the error,
    not the raw `__M_foo__Counter` mangle."""
    foo = write(
        tmp_path, "src/foo.tpu",
        "tablet Counter { x: i64 }\n"
        "fn make_foo() -> Counter { Counter { x: 7 } }\n",
    )
    bar = write(
        tmp_path, "src/bar.tpu",
        "tablet Counter { y: i64 }\n",
    )
    main = write(
        tmp_path, "src/main.tpu",
        "from foo import Counter as FC, make_foo\n"
        "from bar import Counter as BC\n"
        "fn main() -> i32 {\n"
        "  step b: BC = make_foo()\n"  # type mismatch
        "  0\n"
        "}\n",
    )
    with pytest.raises(CompileError) as ei:
        check_sources([
            (str(foo), foo.read_text()),
            (str(bar), bar.read_text()),
            (str(main), main.read_text()),
        ])
    msg = str(ei.value)
    # The pretty form contains the dotted module path; the raw mangle
    # has the `__M_` prefix users shouldn't see.
    assert "__M_" not in msg
    assert "foo.Counter" in msg or "bar.Counter" in msg


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
