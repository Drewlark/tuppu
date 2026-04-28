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
            elif self.check(Tok.SEAL):
                decls.append(self.parse_seal_decl())
            elif self.check(Tok.COLOPHON):
                decls.append(self.parse_colophon())
            elif self.check(Tok.GLOSS):
                decls.append(self.parse_gloss())
            elif self.check(Tok.EDUBBA):
                decls.append(self.parse_edubba())
            elif self.check(Tok.TYPE_ALIAS):
                decls.append(self.parse_type_alias())
            elif self.check(Tok.IMPORT):
                decls.append(self.parse_import())
            elif self.check(Tok.FROM):
                decls.append(self.parse_from_import())
            else:
                t = self.peek()
                raise ParseError(
                    f"expected 'fn', 'table', 'tablet', 'seal', 'colophon', "
                    f"'gloss', 'edubba', 'type', 'import', or 'from' at top "
                    f"level, got {t.kind.name}",
                    t.line, t.col,
                )
            self.skip_newlines()
        return A.Program(decls)

    def parse_gloss(self) -> A.GlossDecl:
        """`gloss <op>(params) -> type { body }` — operator overload.
        `<op>` is one of the fixed gloss-op names (`add`, `eq`, etc.);
        the typechecker validates the specific name + arity against
        the operator it implements."""
        start = self.eat(Tok.GLOSS)
        op = self.eat(Tok.IDENT, "operator name after 'gloss'").value
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
        return _at(start, A.GlossDecl(
            op=op, params=params, return_type=return_type, body=body,
        ))

    def parse_edubba(self) -> A.EdubbaDecl:
        """`edubba T<...> { fn ... fn ... }` — methods block on a
        tablet. Each method's first param is implicit `self` or
        `mut self` (no type annotation; the receiver type is the
        block's `T<...>` filled in here). The synthesized `Param` is
        prepended so the rest of the compiler sees a perfectly normal
        generic fn with a struct first param. Method names are mangled
        as `<TypeName>__<method>` to keep the user-visible global fn
        namespace clean — the typecheck registry exposes them through
        method-call syntax only."""
        start = self.eat(Tok.EDUBBA)
        type_name = self.eat(Tok.IDENT, "tablet name after 'edubba'").value
        type_params = self.parse_type_params()
        # Build the receiver type expression once — every method param
        # references it. For non-generic tablets we emit a TypeName;
        # generic ones get a TypeApply with TypeName children.
        if type_params:
            receiver_ty: A.TypeExpr = A.TypeApply(
                name=type_name,
                args=[A.TypeName(name=tp) for tp in type_params],
            )
        else:
            receiver_ty = A.TypeName(name=type_name)
        self.eat(Tok.LBRACE)
        methods: list[A.FnDecl] = []
        self.skip_newlines()
        while not self.check(Tok.RBRACE):
            if not self.check(Tok.FN):
                t = self.peek()
                raise ParseError(
                    f"expected 'fn' inside edubba block, got {t.kind.name}",
                    t.line, t.col,
                )
            method = self._parse_edubba_method(type_name, receiver_ty)
            methods.append(method)
            self.skip_newlines()
        self.eat(Tok.RBRACE)
        return _at(start, A.EdubbaDecl(
            type_name=type_name,
            type_params=type_params,
            methods=methods,
        ))

    def _parse_edubba_method(
        self, type_name: str, receiver_ty: "A.TypeExpr",
    ) -> A.FnDecl:
        """Parse one fn inside an edubba block. The first param must be
        `self` or `mut self` (no explicit type) — we synthesize the
        Param here and prepend it. The fn name is mangled with the
        host tablet so two tablets can each have a `len` without
        clashing in the global symbol table."""
        start = self.eat(Tok.FN)
        method_name = self.eat(Tok.IDENT, "method name").value
        # Methods do not get their own type-param list — they inherit
        # the host edubba's. Disallow `<...>` to avoid silent confusion.
        if self.check(Tok.LT):
            t = self.peek()
            raise ParseError(
                "edubba methods inherit the block's type parameters; "
                "drop the per-method `<...>` list",
                t.line, t.col,
            )
        self.eat(Tok.LPAREN)
        # First arg must be `self` or `mut self`.
        self_start = self.peek()
        is_mut_self = False
        if self.check(Tok.MUT):
            self.advance()
            is_mut_self = True
        sname_tok = self.eat(Tok.IDENT, "'self' as receiver")
        if sname_tok.value != "self":
            raise ParseError(
                f"edubba method receiver must be named 'self', got "
                f"{sname_tok.value!r}",
                sname_tok.line, sname_tok.col,
            )
        # Disallow an explicit type annotation — receiver is implicit.
        if self.check(Tok.COLON):
            t = self.peek()
            raise ParseError(
                "edubba method receiver carries no type annotation — "
                "`self` is the host tablet",
                t.line, t.col,
            )
        self_param = _at(self_start, A.Param(
            name="self", type=receiver_ty, is_mut=is_mut_self,
        ))
        params: list[A.Param] = [self_param]
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
            name=f"{type_name}__{method_name}",
            params=params,
            return_type=return_type,
            body=body,
            type_params=[],  # filled in at lowering from the host edubba
        ))

    def parse_import(self) -> A.ImportDecl:
        """`import x.y.z` — wildcard form: every public top-level name
        from `x.y.z` is brought into the current file's scope, AND the
        last segment `z` becomes a module-qualifier (`z.foo` works at
        use sites for module-qualified access).

        `import x.y.z as w` — same as above but `w` (rather than `z`)
        is the qualifier alias, and the wildcard names are NOT brought
        into local scope (the alias is the only access path).
        """
        start = self.eat(Tok.IMPORT)
        path = self._parse_module_path()
        alias: str | None = None
        if self.check(Tok.AS):
            self.advance()
            alias_tok = self.eat(Tok.IDENT, "alias name after 'as'")
            alias = alias_tok.value
        return _at(start, A.ImportDecl(
            path=path, names=None, wildcard_alias=alias,
        ))

    def parse_from_import(self) -> A.ImportDecl:
        """`from x.y import a, b as c` — selective form. Brings only the
        named decls into scope, optionally with a local alias each."""
        start = self.eat(Tok.FROM)
        path = self._parse_module_path()
        self.eat(Tok.IMPORT, "expected 'import' after module path")
        names: list[tuple[str, str | None]] = []
        names.append(self._parse_import_name())
        while self.check(Tok.COMMA):
            self.advance()
            names.append(self._parse_import_name())
        return _at(start, A.ImportDecl(path=path, names=names))

    def _parse_module_path(self) -> list[str]:
        """Read a dotted identifier path like `stdlib.list` into a list
        of segments. Each segment is a plain IDENT; the dots are pure
        path separators (no expression-level meaning)."""
        first = self.eat(Tok.IDENT, "module path segment")
        path = [first.value]
        while self.check(Tok.DOT):
            self.advance()
            seg = self.eat(Tok.IDENT, "module path segment after '.'")
            path.append(seg.value)
        return path

    def _parse_import_name(self) -> tuple[str, str | None]:
        """One entry in a `from ... import ...` list: `name` or
        `name as alias`. Returns (source_name, local_alias_or_None)."""
        n = self.eat(Tok.IDENT, "imported name").value
        alias: str | None = None
        if self.check(Tok.AS):
            self.advance()
            alias = self.eat(Tok.IDENT, "alias name after 'as'").value
        return (n, alias)

    def parse_colophon(self) -> A.ColophonDecl:
        """`colophon fn name(params) -> type` — declare an external C
        function the compiler will emit as an extern and marshal at
        every call site. No body, no generics."""
        start = self.eat(Tok.COLOPHON)
        self.eat(Tok.FN, "expected 'fn' after 'colophon'")
        name = self.eat(Tok.IDENT, "external function name").value
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
        return _at(start, A.ColophonDecl(
            name=name, params=params, return_type=return_type, c_name=name,
        ))

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

    def parse_type_alias(self) -> A.AliasDecl:
        """`type Name = TypeExpr` — declares a transparent alias.
        No type-parameter list yet (would need first-class tuples /
        higher-kinded handling for the common cases users actually
        want)."""
        start = self.eat(Tok.TYPE_ALIAS)
        name = self.eat(Tok.IDENT, "alias name").value
        self.eat(Tok.EQ)
        target = self.parse_type()
        return _at(start, A.AliasDecl(name=name, target=target))

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

    def parse_seal_decl(self) -> A.SealDecl:
        start = self.eat(Tok.SEAL)
        name = self.eat(Tok.IDENT, "seal name").value
        type_params = self.parse_type_params()
        self.eat(Tok.LBRACE)
        variants: list[A.Variant] = []
        seen_names: set[str] = set()
        self.skip_newlines()
        while not self.check(Tok.RBRACE):
            vtok = self.peek()
            vname = self.eat(Tok.IDENT, "variant name").value
            if vname in seen_names:
                raise ParseError(
                    f"seal {name!r}: duplicate variant {vname!r}",
                    vtok.line, vtok.col,
                )
            seen_names.add(vname)
            fields: list[A.TypeExpr] = []
            if self.check(Tok.LPAREN):
                self.advance()
                if not self.check(Tok.RPAREN):
                    fields.append(self.parse_type())
                    while self.check(Tok.COMMA):
                        self.advance()
                        fields.append(self.parse_type())
                self.eat(Tok.RPAREN)
            variants.append(_at(vtok, A.Variant(name=vname, fields=fields)))
            self.skip_newlines()
            if self.check(Tok.COMMA):
                self.advance()
                self.skip_newlines()
            else:
                break
        self.skip_newlines()
        self.eat(Tok.RBRACE)
        if not variants:
            raise ParseError(
                f"seal {name!r} must declare at least one variant",
                start.line, start.col,
            )
        return _at(start, A.SealDecl(
            name=name, variants=variants, type_params=type_params,
        ))

    def parse_struct_lit(self) -> A.StructLit:
        name_tok = self.eat(Tok.IDENT, "tablet name")
        # Dotted module-qualified form: `mod.Tablet { ... }` or
        # `mod.sub.Tablet { ... }`. Collapse the segments into a single
        # dotted string on `StructLit.name`; the typechecker splits at
        # the last `.` to find the qualifier and the short tablet name.
        full_name = name_tok.value
        while self.check(Tok.DOT):
            self.advance()
            seg = self.eat(Tok.IDENT, "struct-lit segment after '.'")
            full_name = f"{full_name}.{seg.value}"
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
        return _at(name_tok, A.StructLit(name=full_name, fields=fields))

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
            # Module-qualified type name: `mod.Foo` or `mod.Foo<T>`. The
            # name is collapsed into a single dotted string here so the
            # downstream typechecker can split it once and look up the
            # qualifier in `module_aliases`.
            full_name = t.value
            while self.check(Tok.DOT):
                self.advance()
                seg = self.eat(Tok.IDENT, "type-name segment after '.'")
                full_name = f"{full_name}.{seg.value}"
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
                return _at(t, A.TypeApply(name=full_name, args=args))
            return _at(t, A.TypeName(name=full_name))
        if t.kind is Tok.TABLETS:
            self.advance()
            self.eat(Tok.LBRACKET)
            # `tablets[...]T` is the variadic-param shape. The lexer
            # collapses `..` → DOTDOT so `...` lexes as DOTDOT DOT.
            # Last-param enforcement happens in typecheck; the parser
            # just shapes the AST here.
            if self.check(Tok.DOTDOT) and self.peek(1).kind is Tok.DOT:
                self.advance()  # ..
                self.advance()  # .
                self.eat(Tok.RBRACKET)
                element = self.parse_type()
                return _at(t, A.TypeVariadicTablets(element=element))
            size = self.eat(Tok.INT, "tablet size (integer)").value
            self.eat(Tok.RBRACKET)
            element = self.parse_type()
            return _at(t, A.TypeTablets(size=size, element=element))
        if t.kind is Tok.BUFFER:
            self.advance()
            self.eat(Tok.LBRACKET)
            size = self.eat(Tok.INT, "buffer size (integer)").value
            self.eat(Tok.RBRACKET)
            element = self.parse_type()
            return _at(t, A.TypeBuffer(size=size, element=element))
        if t.kind is Tok.WEDGE:
            # `wedge T` — a handle into some `tablets[N]T`. Runtime
            # footprint is a pointer; you get one from `tablets.push`.
            # "Wedge" because cuneiform is literally wedge-writing, and
            # a single wedge is the atom of a Mesopotamian mark.
            self.advance()
            element = self.parse_type()
            return _at(t, A.TypeHandle(element=element))
        if t.kind is Tok.IVEC:
            # `ivec<T>` — indirect vector: contiguous heap-allocated
            # array of pointers, each pointing to a separately-allocated
            # T. O(1) random access; T values are pointer-stable across
            # resize.
            self.advance()
            self.eat(Tok.LT)
            element = self.parse_type()
            self.eat(Tok.GT)
            return _at(t, A.TypeIVec(element=element))
        if t.kind is Tok.DVEC:
            # `dvec<T>` — direct vector: contiguous heap-allocated
            # array of T values inline. O(1) random access (one load),
            # but grow invalidates T addresses, so push returns unit.
            self.advance()
            self.eat(Tok.LT)
            element = self.parse_type()
            self.eat(Tok.GT)
            return _at(t, A.TypeDVec(element=element))
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
        if t.kind is Tok.FN:
            # First-class function type: `fn(T1, T2) -> U`. Used as
            # param / binding / return annotations. No capture: these
            # are plain function pointers, no environment.
            self.advance()
            self.eat(Tok.LPAREN)
            params: list[A.TypeExpr] = []
            if not self.check(Tok.RPAREN):
                params.append(self.parse_type())
                while self.check(Tok.COMMA):
                    self.advance()
                    params.append(self.parse_type())
            self.eat(Tok.RPAREN)
            return_type: A.TypeExpr | None = None
            if self.check(Tok.ARROW):
                self.advance()
                return_type = self.parse_type()
            return _at(t, A.TypeFn(params=params, return_type=return_type))
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
        """An assignment target must be a chain of `.`-field and `[]`-
        index access steps rooted at an Ident. `r.field = v`,
        `arr[n] = v`, `arr[n].field = v`, `m.values[idx] = v` — all
        legal. Codegen's `_lvalue_slot` walks the same chain to GEP
        the slot pointer + emit bounds checks."""
        node = expr
        while isinstance(node, (A.Field, A.Index)):
            node = node.target
        if not isinstance(node, A.Ident):
            raise ParseError(
                "assignment target must be a variable or a chain of "
                "field / index accesses rooted at one",
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
            # `Name { field : ...` and `mod.Name { field : ...` are
            # struct literals. The lookahead skips any NEWLINE tokens
            # between the `{` and the first field so multi-line struct
            # literals still trigger the struct-lit parse path. A
            # block's first token is never IDENT-COLON because bindings
            # require `step`/`mut`. The dotted form is unambiguous —
            # a field-access chain followed by `{ IDENT :` doesn't
            # form a valid expression elsewhere in the grammar (block
            # bodies start with a stmt keyword, not a label).
            #
            # Walk past any `.IDENT` segments, then check for the
            # struct-lit shape.
            i = 1
            while self.peek(i).kind is Tok.DOT and self.peek(i + 1).kind is Tok.IDENT:
                i += 2
            if self.peek(i).kind is Tok.LBRACE:
                j = i + 1
                while self.peek(j).kind is Tok.NEWLINE:
                    j += 1
                if (
                    self.peek(j).kind is Tok.IDENT
                    and self.peek(j + 1).kind is Tok.COLON
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
        if t.kind is Tok.COPY:
            # `copy expr` — deep-clone prefix. Binds tight like unary
            # so `copy name.field` and `copy arr[i]` parse as the
            # clone of the accessed value, not a clone of the root.
            self.advance()
            operand = self.parse_expr(min_prec=8)
            return _at(t, A.Copy(value=operand))
        if t.kind is Tok.MINUS or t.kind is Tok.BANG:
            op = "-" if t.kind is Tok.MINUS else "!"
            self.advance()
            operand = self.parse_expr(min_prec=8)
            return _at(t, A.Unary(op=op, operand=operand))
        if t.kind is Tok.LBRACE:
            return self.parse_block()
        if t.kind is Tok.IF:
            return self.parse_if()
        if t.kind is Tok.MATCH:
            return self.parse_match()
        if t.kind is Tok.TABLETS:
            return self.parse_tablets_lit()
        raise ParseError(f"expected expression, got {t.kind.name}", t.line, t.col)

    def parse_tablets_lit(self) -> A.TabletsLit:
        """`tablets[N]T { e1, e2, ... }` — pre-populated tablets literal.
        The size and element type are both required here; call-site
        variadic desugaring synthesizes a TabletsLit with a canonical
        size and uses typecheck to pin the element type."""
        start = self.eat(Tok.TABLETS)
        self.eat(Tok.LBRACKET)
        size = self.eat(Tok.INT, "tablet size (integer)").value
        self.eat(Tok.RBRACKET)
        element = self.parse_type()
        self.eat(Tok.LBRACE)
        fields: list[A.Expr] = []
        self.skip_newlines()
        while not self.check(Tok.RBRACE):
            fields.append(self.parse_expr())
            self.skip_newlines()
            if self.check(Tok.COMMA):
                self.advance()
                self.skip_newlines()
            else:
                break
        self.skip_newlines()
        self.eat(Tok.RBRACE)
        return _at(start, A.TabletsLit(size=size, element=element, fields=fields))

    def parse_match(self) -> A.MatchExpr:
        start = self.eat(Tok.MATCH)
        scrutinee = self.parse_expr()
        self.eat(Tok.LBRACE)
        arms: list[A.MatchArm] = []
        self.skip_newlines()
        while not self.check(Tok.RBRACE):
            arm_start = self.peek()
            pattern = self.parse_pattern()
            self.eat(Tok.FATARROW)
            body = self.parse_expr()
            arms.append(_at(arm_start, A.MatchArm(pattern=pattern, body=body)))
            self.skip_newlines()
            if self.check(Tok.COMMA):
                self.advance()
                self.skip_newlines()
            else:
                break
        self.skip_newlines()
        self.eat(Tok.RBRACE)
        if not arms:
            raise ParseError(
                "match expression requires at least one arm",
                start.line, start.col,
            )
        return _at(start, A.MatchExpr(scrutinee=scrutinee, arms=arms))

    def parse_pattern(self) -> A.Pattern:
        t = self.peek()
        # `_` is a bare wildcard. It lexes as an IDENT whose value is "_".
        if t.kind is Tok.IDENT and t.value == "_":
            self.advance()
            return _at(t, A.WildcardPattern())
        if t.kind is Tok.IDENT:
            self.advance()
            binders: list[str | None] = []
            has_paren = False
            if self.check(Tok.LPAREN):
                has_paren = True
                self.advance()
                if not self.check(Tok.RPAREN):
                    binders.append(self._parse_pattern_binder())
                    while self.check(Tok.COMMA):
                        self.advance()
                        binders.append(self._parse_pattern_binder())
                self.eat(Tok.RPAREN)
            return _at(t, A.VariantPattern(name=t.value, binders=binders))
        raise ParseError(
            f"expected pattern, got {t.kind.name}", t.line, t.col,
        )

    def _parse_pattern_binder(self) -> str | None:
        t = self.peek()
        if t.kind is not Tok.IDENT:
            raise ParseError(
                f"expected identifier or `_` in pattern, got {t.kind.name}",
                t.line, t.col,
            )
        self.advance()
        return None if t.value == "_" else t.value

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
            # Distinguish `x[expr]` (Index) from `x[lo:hi]` (Slice).
            # Either bound may be elided — `[:]`, `[lo:]`, `[:hi]`.
            lo: A.Expr | None = None
            hi: A.Expr | None = None
            if self.check(Tok.COLON):
                self.advance()
                if not self.check(Tok.RBRACKET):
                    hi = self.parse_expr()
                self.eat(Tok.RBRACKET)
                return _at(op_tok, A.Slice(target=left, lo=lo, hi=hi))
            lo = self.parse_expr()
            if self.check(Tok.COLON):
                self.advance()
                if not self.check(Tok.RBRACKET):
                    hi = self.parse_expr()
                self.eat(Tok.RBRACKET)
                return _at(op_tok, A.Slice(target=left, lo=lo, hi=hi))
            self.eat(Tok.RBRACKET)
            return _at(op_tok, A.Index(target=left, index=lo))
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
