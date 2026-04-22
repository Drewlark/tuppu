from __future__ import annotations

import pytest

from tuppu import ast as A
from tuppu.lexer import lex
from tuppu.parser import ParseError, parse


def parse_expr_only(source: str) -> A.Expr:
    """Wrap an expression in a trivial program so we can reuse the main parser.

    We synthesize `fn _(){ <expr> }` and extract the block's tail.
    """
    toks = lex("fn _(){ " + source + " }")
    prog = parse(toks)
    assert len(prog.decls) == 1
    fn = prog.decls[0]
    assert isinstance(fn, A.FnDecl)
    tail = fn.body.tail
    assert tail is not None, f"no tail expression in: {source!r}"
    return tail


# --- literals ---------------------------------------------------------------

def test_int_literal():
    assert parse_expr_only("42") == A.IntLit(value=42)


def test_sex_literal_parses():
    # `1;30` → SexLit with int_digits=[1], frac_digits=[30], num=3, den=2.
    assert parse_expr_only("1;30") == A.SexLit(
        int_digits=[1], frac_digits=[30], num=3, den=2,
    )


def test_sex_integer_form_parses():
    # `1 30` → SexLit with frac_digits=None, num=90, den=1.
    assert parse_expr_only("1 30") == A.SexLit(
        int_digits=[1, 30], frac_digits=None, num=90, den=1,
    )


def test_string_literal():
    assert parse_expr_only('"hi"') == A.StringLit(value=b"hi")


def test_bool_literals():
    assert parse_expr_only("true") == A.BoolLit(value=True)
    assert parse_expr_only("false") == A.BoolLit(value=False)


def test_ident():
    assert parse_expr_only("foo") == A.Ident(name="foo")


# --- precedence -------------------------------------------------------------

def test_add_and_mul_precedence():
    # 1 + 2 * 3  ==>  Binary(+, 1, Binary(*, 2, 3))
    e = parse_expr_only("1 + 2 * 3")
    assert isinstance(e, A.Binary) and e.op == "+"
    assert isinstance(e.rhs, A.Binary) and e.rhs.op == "*"


def test_left_associative_addition():
    # 1 + 2 + 3  ==>  Binary(+, Binary(+, 1, 2), 3)
    e = parse_expr_only("1 + 2 + 3")
    assert isinstance(e, A.Binary) and e.op == "+"
    assert isinstance(e.lhs, A.Binary) and e.lhs.op == "+"
    assert isinstance(e.rhs, A.IntLit) and e.rhs.value == 3


def test_comparison_lower_than_arith():
    # a + b < c * d
    e = parse_expr_only("a + b < c * d")
    assert isinstance(e, A.Binary) and e.op == "<"
    assert isinstance(e.lhs, A.Binary) and e.lhs.op == "+"
    assert isinstance(e.rhs, A.Binary) and e.rhs.op == "*"


def test_and_or_precedence():
    # a || b && c  ==>  Binary(||, a, Binary(&&, b, c))
    e = parse_expr_only("a || b && c")
    assert isinstance(e, A.Binary) and e.op == "||"
    assert isinstance(e.rhs, A.Binary) and e.rhs.op == "&&"


def test_unary_minus_binds_tighter_than_mul():
    # -a * b  ==>  Binary(*, Unary(-, a), b)
    e = parse_expr_only("-a * b")
    assert isinstance(e, A.Binary) and e.op == "*"
    assert isinstance(e.lhs, A.Unary) and e.lhs.op == "-"


def test_parentheses_override():
    e = parse_expr_only("(1 + 2) * 3")
    assert isinstance(e, A.Binary) and e.op == "*"
    assert isinstance(e.lhs, A.Binary) and e.lhs.op == "+"


# --- postfix ----------------------------------------------------------------

def test_call_no_args():
    e = parse_expr_only("foo()")
    assert isinstance(e, A.Call) and isinstance(e.callee, A.Ident) and e.callee.name == "foo"
    assert e.args == []


def test_call_with_args():
    e = parse_expr_only("foo(1, 2, x + y)")
    assert isinstance(e, A.Call) and len(e.args) == 3
    assert isinstance(e.args[2], A.Binary) and e.args[2].op == "+"


def test_index_and_call_chain():
    e = parse_expr_only("foo[3](x)")
    assert isinstance(e, A.Call)
    assert isinstance(e.callee, A.Index)


def test_field_access():
    e = parse_expr_only("r.num")
    assert isinstance(e, A.Field) and e.name == "num"


def test_cast():
    e = parse_expr_only("x as f64")
    assert isinstance(e, A.Cast) and isinstance(e.type, A.TypeName) and e.type.name == "f64"


# --- if expression ----------------------------------------------------------

def test_if_then_else():
    e = parse_expr_only("if x < 2 { 1 } else { 2 }")
    assert isinstance(e, A.IfExpr)
    assert isinstance(e.cond, A.Binary) and e.cond.op == "<"
    assert isinstance(e.then, A.Block) and e.then.tail == A.IntLit(value=1)
    assert isinstance(e.else_, A.Block) and e.else_.tail == A.IntLit(value=2)


def test_if_without_else():
    e = parse_expr_only("if x { 1 }")
    assert isinstance(e, A.IfExpr) and e.else_ is None


def test_if_else_if_chain():
    e = parse_expr_only("if a { 1 } else if b { 2 } else { 3 }")
    assert isinstance(e, A.IfExpr)
    assert isinstance(e.else_, A.IfExpr)
    assert isinstance(e.else_.else_, A.Block)


def test_if_with_newline_before_else():
    # `}\n else {` — NEWLINE between block and else must be tolerated
    e = parse_expr_only("if a { 1 }\nelse { 2 }")
    assert isinstance(e, A.IfExpr) and isinstance(e.else_, A.Block)


# --- blocks and statements --------------------------------------------------

def test_block_with_bindings_and_tail():
    e = parse_expr_only("{ step x = 1\n step y = 2\n x + y }")
    assert isinstance(e, A.Block)
    assert len(e.stmts) == 2
    assert all(isinstance(s, A.Binding) for s in e.stmts)
    assert isinstance(e.tail, A.Binary) and e.tail.op == "+"


def test_step_binding_with_type():
    e = parse_expr_only("{ step x: i64 = 5\n x }")
    stmt = e.stmts[0]
    assert isinstance(stmt, A.Binding) and not stmt.is_mut
    assert stmt.type_ann == A.TypeName(name="i64")


def test_mut_binding():
    e = parse_expr_only("{ mut x = 0\n x = x + 1\n x }")
    assert isinstance(e.stmts[0], A.Binding) and e.stmts[0].is_mut
    assert isinstance(e.stmts[1], A.Assign)


def test_while_stmt():
    e = parse_expr_only("{ mut i = 0\n while i < 10 { i = i + 1 } }")
    assert isinstance(e.stmts[1], A.While)


def test_yield_with_value():
    e = parse_expr_only("{ yield 42 }")
    # yield is a statement, so the block has no tail
    assert isinstance(e.stmts[0], A.YieldStmt)
    assert e.stmts[0].value == A.IntLit(value=42)
    assert e.tail is None


def test_yield_without_value():
    e = parse_expr_only("{ yield }")
    assert isinstance(e.stmts[0], A.YieldStmt) and e.stmts[0].value is None


# --- functions and tables ---------------------------------------------------

def test_fn_decl_no_return():
    prog = parse(lex("fn hello() { }"))
    fn = prog.decls[0]
    assert isinstance(fn, A.FnDecl) and fn.name == "hello"
    assert fn.params == []
    assert fn.return_type is None


def test_fn_decl_with_params_and_return():
    prog = parse(lex("fn add(a: i64, b: i64) -> i64 { a + b }"))
    fn = prog.decls[0]
    assert isinstance(fn, A.FnDecl)
    assert fn.name == "add"
    assert len(fn.params) == 2
    assert fn.params[0].type == A.TypeName(name="i64")
    assert fn.return_type == A.TypeName(name="i64")


def test_table_decl():
    prog = parse(lex("table fact_table[0..20]: i64 = fact"))
    td = prog.decls[0]
    assert isinstance(td, A.TableDecl)
    assert td.name == "fact_table"
    assert td.lo == A.IntLit(value=0)
    assert td.hi == A.IntLit(value=20)
    assert td.element_type == A.TypeName(name="i64")
    assert td.generator == A.Ident(name="fact")


# --- types ------------------------------------------------------------------

def test_array_type():
    prog = parse(lex("fn f(buf: [32]u8) -> i64 { 0 }"))
    fn = prog.decls[0]
    assert fn.params[0].type == A.TypeArray(size=32, element=A.TypeName(name="u8"))


def test_tablets_type():
    prog = parse(lex("fn f(toks: tablets[256]i64) -> i64 { 0 }"))
    fn = prog.decls[0]
    assert fn.params[0].type == A.TypeTablets(size=256, element=A.TypeName(name="i64"))


# --- full program sanity ----------------------------------------------------

def test_fact_program_parses():
    src = (
        "fn fact(n: i64) -> i64 {\n"
        "  if n < 2 { 1 } else { n * fact(n - 1) }\n"
        "}\n"
        "\n"
        "fn main() -> i32 {\n"
        "  println(fact(10))\n"
        "  0\n"
        "}\n"
    )
    prog = parse(lex(src))
    assert len(prog.decls) == 2
    fact = prog.decls[0]
    assert isinstance(fact, A.FnDecl) and fact.name == "fact"
    # body is a block whose tail is the if-expr
    assert isinstance(fact.body.tail, A.IfExpr)

    main = prog.decls[1]
    assert isinstance(main, A.FnDecl) and main.name == "main"
    assert main.return_type == A.TypeName(name="i32")
    # main's body: [ExprStmt(println(fact(10)))], tail = 0
    assert len(main.body.stmts) == 1
    assert isinstance(main.body.stmts[0], A.ExprStmt)
    assert main.body.tail == A.IntLit(value=0)


# --- error cases ------------------------------------------------------------

def test_missing_rparen_errors():
    with pytest.raises(ParseError):
        parse(lex("fn f() { foo(1, 2 }"))


def test_missing_rbrace_errors():
    with pytest.raises(ParseError):
        parse(lex("fn f() { step x = 1"))


def test_top_level_non_decl_errors():
    with pytest.raises(ParseError, match="top level"):
        parse(lex("step x = 5"))
