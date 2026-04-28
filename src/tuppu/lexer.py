"""Tuppu lexer.

Produces a token stream from source text per SPEC.md §3. Newlines are
significant — a NEWLINE token is emitted where a statement can end —
except inside unmatched brackets or after a continuer token (binary
operator, `=`, `,`, `->`, `.`, `..`, `:`).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import gcd


class Tok(Enum):
    # literals
    INT = auto()        # value: int
    SEX = auto()        # value: (int_digits, frac_digits, num, den)
                        #   - int_digits: list[int] — pre-radix sexagesimal digits
                        #   - frac_digits: list[int] | None — post-radix digits, or None for integer form
                        #   - num, den: pre-reduced rat form, den > 0
    STRING = auto()     # value: bytes (after escape processing)
    CHAR = auto()       # value: int (0..255) — a single-byte char literal 'a'
    TRUE = auto()
    FALSE = auto()

    # identifiers
    IDENT = auto()      # value: str

    # keywords
    FN = auto()
    STEP = auto()
    MUT = auto()
    VAR = auto()
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    YIELD = auto()
    AS = auto()
    TABLE = auto()
    TABLETS = auto()
    BUFFER = auto()      # emitted by `buffer` keyword — fixed-size byte buffer
    IVEC = auto()        # emitted by `ivec` keyword — indirect vector (array of pointers)
    DVEC = auto()        # emitted by `dvec` keyword — direct vector (contiguous T values)
    RELEASE = auto()
    STRUCT = auto()      # emitted by `tablet` keyword — product type decl
    WEDGE = auto()       # emitted by `wedge` keyword — handle type (tablet ref)
    SEAL = auto()        # emitted by `seal` keyword — sum type decl
    MATCH = auto()       # emitted by `match` keyword — pattern-match expr
    LOST = auto()
    COLOPHON = auto()    # reserved; no semantics yet (see NEXT.md)
    GLOSS = auto()       # emitted by `gloss` keyword — operator overload decl
    EDUBBA = auto()      # emitted by `edubba` keyword — methods block
    COPY = auto()        # emitted by `copy` keyword — deep-clone prefix op
    TYPE_ALIAS = auto()  # emitted by `type` keyword — alias decl
    IMPORT = auto()      # emitted by `import` keyword — module import
    FROM = auto()        # emitted by `from` keyword — selective import

    # type keywords (value: str — "i64", "bool", etc.)
    TYPE_KW = auto()

    # operators / punctuation
    PLUS = auto()       # +
    MINUS = auto()      # -
    STAR = auto()       # *
    SLASH = auto()      # /
    PERCENT = auto()    # %
    EQ = auto()         # =
    EQEQ = auto()       # ==
    PLUSEQ = auto()     # +=
    MINUSEQ = auto()    # -=
    STAREQ = auto()     # *=
    SLASHEQ = auto()    # /=
    PERCENTEQ = auto()  # %=
    BANG = auto()       # !
    BANGEQ = auto()     # !=
    LT = auto()         # <
    LE = auto()         # <=
    GT = auto()         # >
    GE = auto()         # >=
    AMPAMP = auto()     # &&
    PIPEPIPE = auto()   # ||
    ARROW = auto()      # ->
    FATARROW = auto()   # =>
    DOT = auto()        # .
    DOTDOT = auto()     # ..
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    COLON = auto()

    NEWLINE = auto()
    EOF = auto()


KEYWORDS: dict[str, Tok] = {
    "fn": Tok.FN,
    "step": Tok.STEP,
    "mut": Tok.MUT,
    "var": Tok.VAR,
    "if": Tok.IF,
    "elif": Tok.ELIF,
    "else": Tok.ELSE,
    "while": Tok.WHILE,
    "for": Tok.FOR,
    "in": Tok.IN,
    "yield": Tok.YIELD,
    "as": Tok.AS,
    "table": Tok.TABLE,
    # `tablet Name { field: ty, ... }` is a product type decl —
    # an inscribed clay tablet with named fields. (Previously: struct/seal.)
    "tablet": Tok.STRUCT,
    # `tablets[N]T` is a chained-chunk growable storage of T. Plural of
    # `tablet` — a chest of tablets.
    "tablets": Tok.TABLETS,
    # `buffer[N]T` is a fixed-size, stack-allocated byte buffer, chiefly
    # for FFI. Not a tablets — no chain, no growth, no release.
    "buffer": Tok.BUFFER,
    # `ivec<T>` is an indirect vector — contiguous heap-allocated array of
    # pointers to per-element T allocations. O(1) random access (two
    # loads), pointer-stable T values (resize moves only the pointer
    # array). Cousin: `dvec<T>` for direct contiguous T storage, better
    # for primitive T at the cost of pointer-instability.
    "ivec": Tok.IVEC,
    # `dvec<T>` is a direct vector — contiguous heap-allocated array of
    # T values inline. O(1) random access (one load), but T's address
    # is invalidated by grow (memcpy moves inline T bytes), so push
    # does not hand back a handle. Pick over ivec for primitive /
    # small T where the per-element heap allocation is wasteful.
    "dvec": Tok.DVEC,
    # `wedge T` is a handle into some `tablets[N]T` — cuneiform is
    # "wedge-writing", so a wedge is the atom of a Mesopotamian mark:
    # a single small reference to something larger.
    "wedge": Tok.WEDGE,
    "release": Tok.RELEASE,
    # `seal Name { Variant, Variant(T), ... }` declares a sum type.
    # Kotlin/Scala precedent: a cylinder seal produces one of a fixed
    # set of stamp designs — here, one of a fixed set of variants.
    "seal": Tok.SEAL,
    # `match e { Pattern => expr, ... }` is a pattern-match expression
    # that dispatches on a seal value's variant.
    "match": Tok.MATCH,
    "lost": Tok.LOST,
    # `colophon` is reserved for a future use (file-level metadata
    # preamble, tablets debug-name, something along those lines) — the
    # Babylonian colophon being the scribe's tag at the end of a tablet.
    # Reserving the word now so users can't accidentally claim it.
    "colophon": Tok.COLOPHON,
    # `gloss <op>(a: T, b: U) -> V { ... }` is an operator overload —
    # a marginal annotation that assigns meaning to an operator symbol
    # for user types, mirroring the ancient scribal practice of
    # writing a gloss above a word to explain its reading.
    "gloss": Tok.GLOSS,
    # `edubba T<...> { fn ... }` declares a methods block on a tablet.
    # E2-DUB-BA-A was the Sumerian/Akkadian scribal school — literally
    # "tablet-house" — where students learned to read, write, and
    # operate on tablets. Methods on a type are the teachings of its
    # edubba: a tablet holds the data, the edubba records what you
    # can do with it. Multiple edubba blocks may be declared on the
    # same tablet (different scribes, same tablet).
    "edubba": Tok.EDUBBA,
    # `copy x` deep-clones a cleanup-bearing value into independent
    # ownership. Assyriological use: archival scribes made "copies" of
    # originals. Syntactic escape hatch under the freeze-while-borrow
    # rule — when a borrow would cross a mut-reaching call, `copy`
    # makes the cost visible in source.
    "copy": Tok.COPY,
    # `type Bytes = buffer[1024]u8` declares a type alias. Aliases are
    # transparent — the alias name is interchangeable with its target
    # at every use site, no nominal-type wrapping.
    "type": Tok.TYPE_ALIAS,
    # `import path.to.module` and `from path.to.module import name [as alias]`
    # bring decls from another file's module into the current file's scope.
    # `import x.y` is sugar for `from x.y import *` (all public decls).
    # Visibility within a module is public-by-default; names starting with
    # `_` (Python convention) are private to the declaring module.
    "import": Tok.IMPORT,
    "from": Tok.FROM,
    "true": Tok.TRUE,
    "false": Tok.FALSE,
}

TYPE_KEYWORDS: set[str] = {
    "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64",
    "bool", "f32", "f64", "rat",
    "sex", "dish",
}

def _sex_to_rat(int_digits: list[int], frac_digits: list[int] | None) -> tuple[int, int]:
    """Compute the exact rational (num, den) denoted by a sex digit sequence.

    - With frac_digits=None (integer form): den=1, num=sum of int_digits in base 60.
    - With frac_digits present: value is (int_part) + (frac_part)/60^k, reduced.
    """
    int_val = 0
    for d in int_digits:
        int_val = int_val * 60 + d
    if frac_digits is None:
        return int_val, 1
    k = len(frac_digits)
    den = 60 ** k
    num = int_val * den
    for i, d in enumerate(frac_digits):
        num += d * (60 ** (k - 1 - i))
    g = gcd(num, den) or 1
    return num // g, den // g


# After one of these tokens, a newline does not end a statement.
CONTINUERS: frozenset[Tok] = frozenset({
    Tok.PLUS, Tok.MINUS, Tok.STAR, Tok.SLASH, Tok.PERCENT,
    Tok.EQ, Tok.EQEQ, Tok.BANGEQ,
    Tok.PLUSEQ, Tok.MINUSEQ, Tok.STAREQ, Tok.SLASHEQ, Tok.PERCENTEQ,
    Tok.LT, Tok.LE, Tok.GT, Tok.GE,
    Tok.AMPAMP, Tok.PIPEPIPE, Tok.BANG,
    Tok.ARROW, Tok.FATARROW,
    Tok.COMMA, Tok.COLON,
    Tok.DOT, Tok.DOTDOT,
})


@dataclass
class Token:
    kind: Tok
    value: object
    line: int
    col: int

    def __repr__(self) -> str:
        if self.value is None:
            return f"{self.kind.name}@{self.line}:{self.col}"
        return f"{self.kind.name}({self.value!r})@{self.line}:{self.col}"


from .errors import CompileError


class LexError(CompileError):
    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"{line}:{col}: {message}")
        self.message = message
        self.line = line
        self.col = col


class Lexer:
    def __init__(self, source: str) -> None:
        self.src = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.bracket_depth = 0
        self.tokens: list[Token] = []

    # --- cursor helpers ---

    def _peek(self, offset: int = 0) -> str:
        i = self.pos + offset
        return self.src[i] if i < len(self.src) else ""

    def _advance(self) -> str:
        c = self.src[self.pos]
        self.pos += 1
        if c == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return c

    def _read_while(self, pred) -> str:
        start = self.pos
        while self.pos < len(self.src) and pred(self.src[self.pos]):
            self._advance()
        return self.src[start:self.pos]

    # --- emission ---

    def _emit(self, kind: Tok, value: object, line: int, col: int) -> None:
        self.tokens.append(Token(kind, value, line, col))

    def _last_significant(self) -> Tok | None:
        for t in reversed(self.tokens):
            if t.kind != Tok.NEWLINE:
                return t.kind
        return None

    def _maybe_emit_newline(self, line: int, col: int) -> None:
        if self.bracket_depth > 0:
            return
        last = self._last_significant()
        if last is None or last in CONTINUERS:
            return
        if self.tokens and self.tokens[-1].kind == Tok.NEWLINE:
            return  # merge consecutive newlines
        self._emit(Tok.NEWLINE, None, line, col)

    # --- main loop ---

    def tokenize(self) -> list[Token]:
        while self.pos < len(self.src):
            c = self._peek()

            if c == "\n":
                line, col = self.line, self.col
                self._advance()
                self._maybe_emit_newline(line, col)
                continue

            if c in " \t\r":
                self._advance()
                continue

            if c == "/" and self._peek(1) == "/":
                while self.pos < len(self.src) and self._peek() != "\n":
                    self._advance()
                continue

            if c.isalpha() or c == "_":
                self._scan_ident()
                continue

            if c.isdigit():
                self._scan_number()
                continue

            if c == '"':
                self._scan_string()
                continue

            if c == "'":
                self._scan_char()
                continue

            self._scan_operator()

        # trailing newline so the parser sees a final statement terminator
        if self.tokens and self.tokens[-1].kind != Tok.NEWLINE:
            self._maybe_emit_newline(self.line, self.col)
        self._emit(Tok.EOF, None, self.line, self.col)
        return self.tokens

    # --- scanners ---

    def _scan_ident(self) -> None:
        line, col = self.line, self.col
        name = self._read_while(lambda c: c.isalnum() or c == "_")
        if name in KEYWORDS:
            self._emit(KEYWORDS[name], None, line, col)
        elif name in TYPE_KEYWORDS:
            self._emit(Tok.TYPE_KW, name, line, col)
        else:
            self._emit(Tok.IDENT, name, line, col)

    def _scan_number(self) -> None:
        line, col = self.line, self.col

        # hex / binary
        if self._peek() == "0" and self._peek(1) in "xX":
            self._advance(); self._advance()
            digits = self._read_while(lambda c: c in "0123456789abcdefABCDEF_")
            if not digits.replace("_", ""):
                raise LexError("expected hex digits after 0x", line, col)
            self._emit(Tok.INT, int(digits.replace("_", ""), 16), line, col)
            return
        if self._peek() == "0" and self._peek(1) in "bB":
            self._advance(); self._advance()
            digits = self._read_while(lambda c: c in "01_")
            if not digits.replace("_", ""):
                raise LexError("expected binary digits after 0b", line, col)
            self._emit(Tok.INT, int(digits.replace("_", ""), 2), line, col)
            return

        # Decimal integer or sexagesimal. The rule: a digit group followed
        # by inline whitespace followed by another digit group (or `;`)
        # combines into a sex literal. A lone digit group with no `;` and
        # no space-continued follower stays as INT.
        int_places: list[str] = [self._read_while(lambda c: c.isdigit())]

        # Space-separated integer-part digit groups.
        while self._peek_after_inline_spaces_is_continuation_digit():
            self._skip_inline_spaces()
            int_places.append(self._read_while(lambda c: c.isdigit()))

        frac_places: list[str] | None = None
        if self._peek_after_inline_spaces_is(";"):
            # Tentatively enter fractional mode: consume spaces and `;`,
            # then require at least one following digit group.
            save = self.pos
            save_line, save_col = self.line, self.col
            self._skip_inline_spaces()
            self._advance()  # ';'
            # allow whitespace after the `;`
            self._skip_inline_spaces()
            if self._peek().isdigit():
                frac_places = [self._read_while(lambda c: c.isdigit())]
                while self._peek_after_inline_spaces_is_continuation_digit():
                    self._skip_inline_spaces()
                    frac_places.append(self._read_while(lambda c: c.isdigit()))
                # A second `;` in the same literal is an error.
                if self._peek_after_inline_spaces_is(";"):
                    raise LexError(
                        "two sexagesimal radix points in one literal",
                        self.line, self.col,
                    )
            else:
                # `;` not followed by a digit — not part of this literal.
                self.pos = save
                self.line, self.col = save_line, save_col

        is_sex = frac_places is not None or len(int_places) > 1
        if is_sex:
            all_places = int_places + (frac_places or [])
            for p in all_places:
                v = int(p)
                if v >= 60:
                    raise LexError(f"sexagesimal place {v} must be < 60", line, col)

        if is_sex:
            int_ds = [int(p) for p in int_places]
            frac_ds = [int(p) for p in frac_places] if frac_places is not None else None
            num, den = _sex_to_rat(int_ds, frac_ds)
            self._emit(Tok.SEX, (int_ds, frac_ds, num, den), line, col)
        else:
            self._emit(Tok.INT, int(int_places[0]), line, col)

    # --- lookahead helpers for sex-literal space continuation ----------

    def _skip_inline_spaces(self) -> None:
        while self._peek() in (" ", "\t"):
            self._advance()

    def _peek_after_inline_spaces_is(self, ch: str) -> bool:
        """Is the next non-inline-space char `ch`? Newlines stop the scan."""
        i = self.pos
        while i < len(self.src) and self.src[i] in (" ", "\t"):
            i += 1
        return i < len(self.src) and self.src[i] == ch

    def _peek_after_inline_spaces_is_continuation_digit(self) -> bool:
        """Is the next non-inline-space char a digit? Only true if there's
        at least one space/tab separating (otherwise the digit is already
        part of the current digit run)."""
        i = self.pos
        saw_space = False
        while i < len(self.src) and self.src[i] in (" ", "\t"):
            i += 1
            saw_space = True
        if not saw_space:
            return False  # adjacent digits are already one run
        return i < len(self.src) and self.src[i].isdigit()

    def _scan_string(self) -> None:
        line, col = self.line, self.col
        self._advance()  # opening "
        out = bytearray()
        while True:
            if self.pos >= len(self.src):
                raise LexError("unterminated string literal", line, col)
            c = self._peek()
            if c == '"':
                self._advance()
                break
            if c == "\\":
                self._advance()
                esc = self._peek()
                if esc == "":
                    raise LexError("unterminated escape", self.line, self.col)
                self._advance()
                out.extend(self._decode_escape(esc).encode("utf-8"))
                continue
            if c == "\n":
                raise LexError("unterminated string literal (newline)", line, col)
            self._advance()
            out.extend(c.encode("utf-8"))
        self._emit(Tok.STRING, bytes(out), line, col)

    def _scan_char(self) -> None:
        """Scan a char literal like `'a'` or `'\\n'`. Body is exactly one
        byte post-escape; emits Tok.CHAR with integer byte value."""
        line, col = self.line, self.col
        self._advance()  # opening '
        if self.pos >= len(self.src):
            raise LexError("unterminated char literal", line, col)
        c = self._peek()
        if c == "'":
            raise LexError("empty char literal", line, col)
        if c == "\n":
            raise LexError("unterminated char literal (newline)", line, col)
        if c == "\\":
            self._advance()
            esc = self._peek()
            if esc == "":
                raise LexError("unterminated escape", self.line, self.col)
            self._advance()
            payload = self._decode_escape(esc).encode("utf-8")
        else:
            self._advance()
            payload = c.encode("utf-8")
        if len(payload) != 1:
            raise LexError(
                "char literal must encode to exactly one byte",
                line, col,
            )
        if self._peek() != "'":
            raise LexError("expected closing ' in char literal", self.line, self.col)
        self._advance()  # closing '
        self._emit(Tok.CHAR, payload[0], line, col)

    def _decode_escape(self, esc: str) -> str:
        table = {"n": "\n", "r": "\r", "t": "\t", "0": "\0", '"': '"', "'": "'", "\\": "\\"}
        if esc in table:
            return table[esc]
        raise LexError(f"unknown escape \\{esc}", self.line, self.col)

    def _scan_operator(self) -> None:
        line, col = self.line, self.col
        c = self._peek()
        c2 = self._peek(1)

        two = c + c2
        two_map = {
            "==": Tok.EQEQ, "!=": Tok.BANGEQ, "<=": Tok.LE, ">=": Tok.GE,
            "&&": Tok.AMPAMP, "||": Tok.PIPEPIPE, "->": Tok.ARROW,
            "=>": Tok.FATARROW, "..": Tok.DOTDOT,
            "+=": Tok.PLUSEQ, "-=": Tok.MINUSEQ, "*=": Tok.STAREQ,
            "/=": Tok.SLASHEQ, "%=": Tok.PERCENTEQ,
        }
        if two in two_map:
            self._advance(); self._advance()
            self._emit(two_map[two], None, line, col)
            return

        one_map = {
            "+": Tok.PLUS, "-": Tok.MINUS, "*": Tok.STAR, "/": Tok.SLASH,
            "%": Tok.PERCENT, "=": Tok.EQ, "!": Tok.BANG,
            "<": Tok.LT, ">": Tok.GT,
            "(": Tok.LPAREN, ")": Tok.RPAREN,
            "{": Tok.LBRACE, "}": Tok.RBRACE,
            "[": Tok.LBRACKET, "]": Tok.RBRACKET,
            ",": Tok.COMMA, ":": Tok.COLON, ".": Tok.DOT,
        }
        if c in one_map:
            kind = one_map[c]
            self._advance()
            # Only `(` and `[` suppress newlines — they're always
            # expression-continuation contexts (fn args, subscripts).
            # `{` delimits both statement blocks (where newlines are
            # genuine statement terminators) and struct / match /
            # enum bodies (which use COMMA for separation); letting
            # newlines through inside braces lets the parser see the
            # terminator between `call()` on one line and `(expr)` on
            # the next, while the specific parsers that care about
            # multi-line layouts call `skip_newlines` explicitly.
            if kind in (Tok.LPAREN, Tok.LBRACKET):
                self.bracket_depth += 1
            elif kind in (Tok.RPAREN, Tok.RBRACKET):
                if self.bracket_depth > 0:
                    self.bracket_depth -= 1
            self._emit(kind, None, line, col)
            return

        # `;` shows up here only when it doesn't sit between two digit
        # groups (which is the sexagesimal radix form `1;30` handled in
        # `_scan_number`). A bare `;` almost always means the user
        # typed it as a statement separator — tell them what Tuppu
        # uses instead so they don't hunt for the real issue.
        if c == ";":
            raise LexError(
                "unexpected ';' — Tuppu uses newlines as statement "
                "separators; ';' is reserved for sexagesimal literals "
                "like `1;30`",
                line, col,
            )
        raise LexError(f"unexpected character {c!r}", line, col)


def lex(source: str) -> list[Token]:
    return Lexer(source).tokenize()
