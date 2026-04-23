"""Tuppu parser.

Recursive descent for declarations and statements; Pratt parsing for
expressions. Input is a token stream from the lexer; output is an
ast.Program whose nodes carry source positions for later error reporting.
Fails with ParseError at the first unexpected token.
"""
from __future__ import annotations

from . import ast as A
from .errors import CompileError
from .lexer import Tok, Token


class ParseError(CompileError):
    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"{line}:{col}: {message}")
        self.message = message
        self.line = line
        self.col = col


# Infix binding power. Higher binds tighter. 0 means "not an infix operator".
INFIX_PREC: dict[Tok, int] = {
    Tok.PIPEPIPE: 1,
    Tok.AMPAMP: 2,
    Tok.EQEQ: 3, Tok.BANGEQ: 3,
    Tok.LT: 4, Tok.LE: 4, Tok.GT: 4, Tok.GE: 4,
    Tok.PLUS: 5, Tok.MINUS: 5,
    Tok.STAR: 6, Tok.SLASH: 6, Tok.PERCENT: 6,
    Tok.AS: 7,
    Tok.LPAREN: 9,
    Tok.LBRACKET: 9,
    Tok.DOT: 9,
}

BINARY_OP_NAME: dict[Tok, str] = {
    Tok.PLUS: "+", Tok.MINUS: "-", Tok.STAR: "*", Tok.SLASH: "/", Tok.PERCENT: "%",
    Tok.EQEQ: "==", Tok.BANGEQ: "!=",
    Tok.LT: "<", Tok.LE: "<=", Tok.GT: ">", Tok.GE: ">=",
    Tok.AMPAMP: "&&", Tok.PIPEPIPE: "||",
}

# `x += y` desugars to `x = x + y` at parse time. This map drives that.
_AUG_ASSIGN_OPS: dict[Tok, str] = {
    Tok.PLUSEQ: "+", Tok.MINUSEQ: "-", Tok.STAREQ: "*",
    Tok.SLASHEQ: "/", Tok.PERCENTEQ: "%",
}


def _at(tok: Token, node):
    """Stamp the starting token's position onto an AST node."""
    node.line = tok.line
    node.col = tok.col
    return node


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # --- cursor -------------------------------------------------------

    def peek(self, offset: int = 0) -> Token:
        i = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[i]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        if t.kind is not Tok.EOF:
            self.pos += 1
        return t

    def check(self, kind: Tok) -> bool:
        return self.peek().kind is kind

    def eat(self, kind: Tok, what: str | None = None) -> Token:
        t = self.peek()
        if t.kind is not kind:
            expected = what or kind.name
            raise ParseError(f"expected {expected}, got {t.kind.name}", t.line, t.col)
        return self.advance()

    def skip_newlines(self) -> None:
        while self.check(Tok.NEWLINE):
            self.advance()

    # --- program / declarations ---------------------------------------

    def parse_program(self) -> A.Program:
        decls: list[A.Decl] = []
        self.skip_newlines()
        while not self.check(Tok.EOF):
            if self.check(Tok.FN):
                decls.append(self.parse_fn())
            elif self.check(Tok.TABLE):
                decls.append(self.parse_table())
            elif self.check(Tok.STRUCT):
                decls.append(self.parse_struct_decl())
            else:
                t = self.peek()
                raise ParseError(
                    f"expected 'fn', 'table', or 'tablet' at top level, "
                    f"got {t.kind.name}",
                    t.line, t.col,
                )
            self.skip_newlines()
        return A.Program(decls)

    def parse_fn(self) -> A.FnDecl:
        start = self.eat(Tok.FN)
        name = self.eat(Tok.IDENT, "function name").value
        type_params = self.parse_type_params()
        self.eat(Tok.LPAREN)
        params: list[A.Param] = []
        if not self.check(Tok.RPAREN):
            params.append(self.parse_param())
            while self.check(Tok.COMMA):
                self.advance()
                params.append(self.parse_param())
        self.eat(Tok.RPAREN)

        return_type: A.TypeExpr | None = None
        if self.check(Tok.ARROW):
            self.advance()
            return_type = self.parse_type()

        body = self.parse_block()
        return _at(start, A.FnDecl(
            name=name, params=params, return_type=return_type, body=body,
            type_params=type_params,
        ))

    def parse_type_params(self) -> list[str]:
        """Parse an optional `<T>` / `<T, U>` type-parameter list.
        Returns an empty list if the next token isn't `<`."""
        if not self.check(Tok.LT):
            return []
        self.advance()
        params: list[str] = []
        if not self.check(Tok.GT):
            params.append(self.eat(Tok.IDENT, "type parameter").value)
            while self.check(Tok.COMMA):
                self.advance()
                params.append(self.eat(Tok.IDENT, "type parameter").value)
        self.eat(Tok.GT)
        return params

    def parse_param(self) -> A.Param:
        start = self.peek()
        is_mut = False
        if self.check(Tok.MUT):
            self.advance()
            is_mut = True
        name = self.eat(Tok.IDENT, "parameter name").value
        self.eat(Tok.COLON)
        ty = self.parse_type()
        return _at(start, A.Param(name=name, type=ty, is_mut=is_mut))

    def parse_struct_decl(self) -> A.StructDecl:
        start = self.eat(Tok.STRUCT)
        name = self.eat(Tok.IDENT, "tablet name").value
        type_params = self.parse_type_params()
        self.eat(Tok.LBRACE)
        fields: list[tuple[str, A.TypeExpr]] = []
        self.skip_newlines()
        while not self.check(Tok.RBRACE):
            field_name = self.eat(Tok.IDENT, "field name").value
            self.eat(Tok.COLON)
            field_type = self.parse_type()
            fields.append((field_name, field_type))
            self.skip_newlines()
            if self.check(Tok.COMMA):
                self.advance()
                self.skip_newlines()
            else:
                break
        self.skip_newlines()
        self.eat(Tok.RBRACE)
        if not fields:
            raise ParseError(
                f"tablet {name!r} must declare at least one field",
                start.line, start.col,
            )
        return _at(start, A.StructDecl(
            name=name, fields=fields, type_params=type_params,
        ))

    def parse_struct_lit(self) -> A.StructLit:
        name_tok = self.eat(Tok.IDENT, "tablet name")
        self.eat(Tok.LBRACE)
        fields: list[tuple[str, A.Expr]] = []
        self.skip_newlines()
        while not self.check(Tok.RBRACE):
            field_name = self.eat(Tok.IDENT, "field name").value
            self.eat(Tok.COLON)
            value = self.parse_expr()
            fields.append((field_name, value))
            self.skip_newlines()
            if self.check(Tok.COMMA):
                self.advance()
                self.skip_newlines()
            else:
                break
        self.skip_newlines()
        self.eat(Tok.RBRACE)
        return _at(name_tok, A.StructLit(name=name_tok.value, fields=fields))

    def parse_table(self) -> A.TableDecl:
        start = self.eat(Tok.TABLE)
        name = self.eat(Tok.IDENT, "table name").value
        self.eat(Tok.LBRACKET)
        lo = self.parse_expr()
        self.eat(Tok.DOTDOT)
        hi = self.parse_expr()
        self.eat(Tok.RBRACKET)
        self.eat(Tok.COLON)
        element_type = self.parse_type()
        self.eat(Tok.EQ)
        generator = self.parse_expr()
        return _at(start, A.TableDecl(
            name=name, lo=lo, hi=hi,
            element_type=element_type, generator=generator,
        ))

    # --- types --------------------------------------------------------

    def parse_type(self) -> A.TypeExpr:
        t = self.peek()
        if t.kind is Tok.TYPE_KW:
            self.advance()
            return _at(t, A.TypeName(name=t.value))
        if t.kind is Tok.IDENT:
            self.advance()
            # Generic type application: `Name<arg1, arg2>`. In type
            # position the `<` is always the type-arg-list bracket —
            # no ambiguity with less-than because type positions don't
            # admit arbitrary expressions.
            if self.check(Tok.LT):
                self.advance()
                args: list[A.TypeExpr] = []
                if not self.check(Tok.GT):
                    args.append(self.parse_type())
                    while self.check(Tok.COMMA):
                        self.advance()
                        args.append(self.parse_type())
                self.eat(Tok.GT)
                return _at(t, A.TypeApply(name=t.value, args=args))
            return _at(t, A.TypeName(name=t.value))
        if t.kind is Tok.TABLETS:
            self.advance()
            self.eat(Tok.LBRACKET)
            size = self.eat(Tok.INT, "tablet size (integer)").value
            self.eat(Tok.RBRACKET)
            element = self.parse_type()
            return _at(t, A.TypeTablets(size=size, element=element))
        if t.kind is Tok.WEDGE:
            # `wedge T` — a handle into some `tablets[N]T`. Runtime
            # footprint is a pointer; you get one from `tablets.push`.
            # "Wedge" because cuneiform is literally wedge-writing, and
            # a single wedge is the atom of a Mesopotamian mark.
            self.advance()
            element = self.parse_type()
            return _at(t, A.TypeHandle(element=element))
        if t.kind is Tok.LBRACKET:
            self.advance()
            size = self.eat(Tok.INT, "array size (integer)").value
            self.eat(Tok.RBRACKET)
            element = self.parse_type()
            return _at(t, A.TypeArray(size=size, element=element))
        if t.kind is Tok.STAR:
            self.advance()
            element = self.parse_type()
            return _at(t, A.TypePointer(element=element))
        raise ParseError(f"expected type, got {t.kind.name}", t.line, t.col)

    # --- blocks and statements ---------------------------------------

    def parse_block(self) -> A.Block:
        start = self.eat(Tok.LBRACE)
        stmts: list[A.Stmt] = []
        tail: A.Expr | None = None
        self.skip_newlines()
        while not self.check(Tok.RBRACE):
            stmt = self.parse_stmt()
            self.skip_newlines()
            if self.check(Tok.RBRACE) and isinstance(stmt, A.ExprStmt):
                tail = stmt.expr
                break
            stmts.append(stmt)
        self.eat(Tok.RBRACE)
        return _at(start, A.Block(stmts=stmts, tail=tail))

    def parse_stmt(self) -> A.Stmt:
        t = self.peek()
        if t.kind is Tok.STEP:   return self.parse_binding(is_mut=False)
        if t.kind is Tok.MUT:    return self.parse_binding(is_mut=True)
        if t.kind is Tok.WHILE:  return self.parse_while()
        if t.kind is Tok.FOR:    return self.parse_for()
        if t.kind is Tok.YIELD:  return self.parse_yield()
        if t.kind is Tok.RELEASE: return self.parse_release()
        # Parse an expression; if it's followed by `=` or an aug-op,
        # treat it as an assignment whose target is that expression.
        # Otherwise it's just an expression statement.
        expr = self.parse_expr()
        peek = self.peek()
        if peek.kind is Tok.EQ:
            self.advance()
            value = self.parse_expr()
            self._check_lvalue(expr, t)
            return _at(t, A.Assign(target=expr, value=value))
        if peek.kind in _AUG_ASSIGN_OPS:
            op_tok = self.advance()
            op = _AUG_ASSIGN_OPS[op_tok.kind]
            rhs = self.parse_expr()
            self._check_lvalue(expr, t)
            combined = _at(op_tok, A.Binary(op=op, lhs=expr, rhs=rhs))
            return _at(t, A.Assign(target=expr, value=combined))
        return _at(t, A.ExprStmt(expr=expr))

    def _check_lvalue(self, expr: A.Expr, start) -> None:
        """An assignment target must be an Ident or a `.`-chain of Field
        accesses rooted at an Ident. No calls, indexes, or literals."""
        node = expr
        while isinstance(node, A.Field):
            node = node.target
        if not isinstance(node, A.Ident):
            raise ParseError(
                "assignment target must be a variable or a `.`-chain "
                "of fields rooted at a variable",
                start.line, start.col,
            )

    def parse_binding(self, *, is_mut: bool) -> A.Binding:
        tok = self.advance()  # step or mut
        name = self.eat(Tok.IDENT, "binding name").value
        type_ann: A.TypeExpr | None = None
        if self.check(Tok.COLON):
            self.advance()
            type_ann = self.parse_type()
        init: A.Expr | None = None
        if self.check(Tok.EQ):
            self.advance()
            init = self.parse_expr()
        if init is None:
            if not is_mut:
                raise ParseError(
                    "step bindings require an initializer", tok.line, tok.col,
                )
            if type_ann is None:
                raise ParseError(
                    "mut binding without initializer requires a type annotation",
                    tok.line, tok.col,
                )
        return _at(tok, A.Binding(
            is_mut=is_mut, name=name, type_ann=type_ann, init=init,
        ))

    def parse_release(self) -> A.ReleaseStmt:
        start = self.eat(Tok.RELEASE)
        name = self.eat(Tok.IDENT, "name after release").value
        return _at(start, A.ReleaseStmt(name=name))

    def parse_while(self) -> A.While:
        start = self.eat(Tok.WHILE)
        cond = self.parse_expr()
        body = self.parse_block()
        return _at(start, A.While(cond=cond, body=body))

    def parse_for(self) -> A.ForStmt:
        start = self.eat(Tok.FOR)
        name = self.eat(Tok.IDENT, "loop variable name").value
        self.eat(Tok.IN, "expected 'in' after for loop variable")
        iter_expr = self.parse_expr()
        body = self.parse_block()
        return _at(start, A.ForStmt(name=name, iter=iter_expr, body=body))

    def parse_yield(self) -> A.YieldStmt:
        start = self.eat(Tok.YIELD)
        if self.check(Tok.NEWLINE) or self.check(Tok.RBRACE) or self.check(Tok.EOF):
            return _at(start, A.YieldStmt(value=None))
        return _at(start, A.YieldStmt(value=self.parse_expr()))

    # --- expressions (Pratt) -----------------------------------------

    def parse_expr(self, min_prec: int = 0) -> A.Expr:
        left = self.parse_prefix()
        while True:
            t = self.peek()
            prec = INFIX_PREC.get(t.kind, 0)
            if prec <= min_prec:
                break
            self.advance()
            left = self.parse_infix(left, t, prec)
        return left

    def parse_prefix(self) -> A.Expr:
        t = self.peek()
        if t.kind is Tok.INT:
            self.advance(); return _at(t, A.IntLit(value=t.value))
        if t.kind is Tok.SEX:
            self.advance()
            int_digits, frac_digits, num, den = t.value
            return _at(t, A.SexLit(
                int_digits=int_digits, frac_digits=frac_digits,
                num=num, den=den,
            ))
        if t.kind is Tok.STRING:
            self.advance(); return _at(t, A.StringLit(value=t.value))
        if t.kind is Tok.CHAR:
            self.advance(); return _at(t, A.CharLit(value=t.value))
        if t.kind is Tok.TRUE:
            self.advance(); return _at(t, A.BoolLit(value=True))
        if t.kind is Tok.FALSE:
            self.advance(); return _at(t, A.BoolLit(value=False))
        if t.kind is Tok.LOST:
            self.advance(); return _at(t, A.LostLit())
        if t.kind is Tok.IDENT:
            # `Name { field : ...` is a struct literal. The 3-token lookahead
            # (LBRACE IDENT COLON) avoids colliding with block expressions or
            # `if x { body }` — a block's first token is never IDENT-COLON
            # because bindings require `step`/`mut`.
            if (
                self.peek(1).kind is Tok.LBRACE
                and self.peek(2).kind is Tok.IDENT
                and self.peek(3).kind is Tok.COLON
            ):
                return self.parse_struct_lit()
            self.advance(); return _at(t, A.Ident(name=t.value))
        if t.kind is Tok.TYPE_KW:
            # Type names double as identifiers in expression position so
            # `rat(n, d)` parses as Call(Ident("rat"), [n, d]).
            self.advance(); return _at(t, A.Ident(name=t.value))
        if t.kind is Tok.LPAREN:
            self.advance()
            e = self.parse_expr()
            self.eat(Tok.RPAREN)
            return e
        if t.kind is Tok.MINUS or t.kind is Tok.BANG:
            op = "-" if t.kind is Tok.MINUS else "!"
            self.advance()
            operand = self.parse_expr(min_prec=8)
            return _at(t, A.Unary(op=op, operand=operand))
        if t.kind is Tok.LBRACE:
            return self.parse_block()
        if t.kind is Tok.IF:
            return self.parse_if()
        raise ParseError(f"expected expression, got {t.kind.name}", t.line, t.col)

    def parse_infix(self, left: A.Expr, op_tok: Token, prec: int) -> A.Expr:
        if op_tok.kind is Tok.LPAREN:
            args: list[A.Expr] = []
            if not self.check(Tok.RPAREN):
                args.append(self.parse_expr())
                while self.check(Tok.COMMA):
                    self.advance()
                    args.append(self.parse_expr())
            self.eat(Tok.RPAREN)
            return _at(op_tok, A.Call(callee=left, args=args))
        if op_tok.kind is Tok.LBRACKET:
            idx = self.parse_expr()
            self.eat(Tok.RBRACKET)
            return _at(op_tok, A.Index(target=left, index=idx))
        if op_tok.kind is Tok.DOT:
            name = self.eat(Tok.IDENT, "field name").value
            return _at(op_tok, A.Field(target=left, name=name))
        if op_tok.kind is Tok.AS:
            ty = self.parse_type()
            return _at(op_tok, A.Cast(value=left, type=ty))
        if op_tok.kind in BINARY_OP_NAME:
            rhs = self.parse_expr(min_prec=prec)
            return _at(op_tok, A.Binary(
                op=BINARY_OP_NAME[op_tok.kind], lhs=left, rhs=rhs,
            ))
        raise ParseError(f"not an infix operator: {op_tok.kind.name}", op_tok.line, op_tok.col)

    def parse_if(self) -> A.IfExpr:
        # Accept either `if` or `elif` as the starter — elif is just
        # `else if` without the brace layer. Produces the same AST
        # (nested IfExpr in else_).
        start = self.peek()
        if start.kind not in (Tok.IF, Tok.ELIF):
            raise ParseError(
                f"expected `if` or `elif`, got {start.kind.name}",
                start.line, start.col,
            )
        self.advance()
        cond = self.parse_expr()
        then = self.parse_block()
        else_: A.Block | A.IfExpr | None = None
        save = self.pos
        self.skip_newlines()
        if self.check(Tok.ELIF):
            else_ = self.parse_if()
        elif self.check(Tok.ELSE):
            self.advance()
            # Still accept the two-token `else if` form for backward
            # compatibility — harmless and common muscle memory.
            if self.check(Tok.IF):
                else_ = self.parse_if()
            else:
                else_ = self.parse_block()
        else:
            self.pos = save
        return _at(start, A.IfExpr(cond=cond, then=then, else_=else_))


def parse(tokens: list[Token]) -> A.Program:
    return Parser(tokens).parse_program()
