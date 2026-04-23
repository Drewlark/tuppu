from __future__ import annotations

import pytest

from tuppu.lexer import LexError, Tok, lex


def kinds(source: str) -> list[Tok]:
    """Return the sequence of token kinds, stripping EOF and any trailing
    NEWLINE (which the lexer emits so the parser sees a terminator on the
    last statement)."""
    ks = [t.kind for t in lex(source) if t.kind is not Tok.EOF]
    while ks and ks[-1] is Tok.NEWLINE:
        ks.pop()
    return ks


def values(source: str) -> list[object]:
    return [t.value for t in lex(source) if t.kind not in (Tok.EOF, Tok.NEWLINE)]


# --- identifiers and keywords -----------------------------------------------

def test_keywords_are_distinct_from_idents():
    toks = lex("fn foo step mut if else while")
    assert [t.kind for t in toks if t.kind is not Tok.EOF and t.kind is not Tok.NEWLINE] == [
        Tok.FN, Tok.IDENT, Tok.STEP, Tok.MUT, Tok.IF, Tok.ELSE, Tok.WHILE,
    ]


def test_type_keywords_carry_name():
    toks = [t for t in lex("i64 u8 bool rat f64") if t.kind is Tok.TYPE_KW]
    assert [t.value for t in toks] == ["i64", "u8", "bool", "rat", "f64"]


def test_true_false():
    assert kinds("true false") == [Tok.TRUE, Tok.FALSE]


# --- integer literals -------------------------------------------------------

def test_decimal_int():
    toks = [t for t in lex("42") if t.kind is Tok.INT]
    assert toks[0].value == 42


def test_hex_int():
    toks = [t for t in lex("0x1A") if t.kind is Tok.INT]
    assert toks[0].value == 0x1A


def test_binary_int():
    toks = [t for t in lex("0b1010") if t.kind is Tok.INT]
    assert toks[0].value == 10


def test_underscore_in_hex():
    toks = [t for t in lex("0xDEAD_BEEF") if t.kind is Tok.INT]
    assert toks[0].value == 0xDEADBEEF


# --- sexagesimal ------------------------------------------------------------

def _sex_num_den(source: str) -> tuple[int, int]:
    """Helper — pull (num, den) from the first SEX token in a source."""
    toks = [t for t in lex(source) if t.kind is Tok.SEX]
    assert toks, f"no SEX token in {source!r}"
    _int_digits, _frac_digits, num, den = toks[0].value
    return num, den


def _sex_digits(source: str) -> tuple[list[int], list[int] | None]:
    toks = [t for t in lex(source) if t.kind is Tok.SEX]
    int_digits, frac_digits, _num, _den = toks[0].value
    return int_digits, frac_digits


def test_sex_integer_form_lowers_via_spaces():
    # `1 30` = 1*60 + 30 = 90, integer-form sex (no radix).
    num, den = _sex_num_den("1 30")
    assert (num, den) == (90, 1)
    assert _sex_digits("1 30") == ([1, 30], None)


def test_sex_integer_three_places_via_spaces():
    # `1 30 0` = 1*3600 + 30*60 + 0 = 5400
    num, den = _sex_num_den("1 30 0")
    assert (num, den) == (5400, 1)


def test_sex_fractional_half():
    # 1;30 = 3/2
    num, den = _sex_num_den("1;30")
    assert (num, den) == (3, 2)
    assert _sex_digits("1;30") == ([1], [30])


def test_sex_fractional_one_third_exact():
    # 0;20 = 1/3 exactly — cannot be represented in f64.
    assert _sex_num_den("0;20") == (1, 3)


def test_sex_fractional_multiple_places_via_spaces():
    # 0;0 45 = 45/3600 = 1/80
    assert _sex_num_den("0;0 45") == (1, 80)


def test_sex_fractional_trailing_zeros_reduce():
    # 1;30 0 0 — trailing fractional zeros reduce to 3/2.
    assert _sex_num_den("1;30 0 0") == (3, 2)


def test_sex_integer_then_frac():
    # 1 30;0 — integer part 1,30 (= 90) with zero fraction. Fractional form
    # because it contains `;`.
    assert _sex_num_den("1 30;0") == (90, 1)


def test_ybc_7289_sqrt2_approximation():
    # YBC 7289: √2 ≈ 1;24 51 10 = 30547/21600 ≈ 1.41421296...
    num, den = _sex_num_den("1;24 51 10")
    assert (num, den) == (30547, 21600)


def test_spaces_around_semicolon_accepted():
    # `1;30`, `1; 30`, `1 ; 30`, `1 ;30` should all lex identically.
    for src in ("1;30", "1; 30", "1 ; 30", "1 ;30"):
        assert _sex_num_den(src) == (3, 2), f"{src!r} should be 3/2"


def test_comma_is_always_just_comma():
    # `foo(1, 30)` — comma is purely an arg separator now.
    ks = kinds("foo(1, 30)")
    assert ks == [Tok.IDENT, Tok.LPAREN, Tok.INT, Tok.COMMA, Tok.INT, Tok.RPAREN]


def test_comma_separated_is_two_ints_not_sex():
    # Old syntax `1,30` is now two separate INT tokens with a COMMA between.
    ks = kinds("1,30")
    assert ks == [Tok.INT, Tok.COMMA, Tok.INT]


def test_two_semicolons_in_sex_is_error():
    with pytest.raises(LexError, match="two sexagesimal radix points"):
        lex("1;30;45")


def test_bare_semicolon_gives_statement_separator_hint():
    # Muscle memory from C/JS/Rust puts `;` between statements. The
    # plain "unexpected character" error is useless for onboarding —
    # tell users where `;` actually belongs.
    with pytest.raises(
        LexError,
        match="newlines as statement separators",
    ):
        lex("foo(); bar()")


def test_sex_place_over_59_is_error():
    with pytest.raises(LexError, match="must be < 60"):
        lex("1 60")


def test_sex_place_over_59_in_fraction_is_error():
    with pytest.raises(LexError, match="must be < 60"):
        lex("1;60")


def test_single_digit_run_is_not_sex():
    # Regular decimal literal: lone digit group with no `;` and no
    # space-continuation stays an INT, even if > 60.
    toks = [t for t in lex("100") if t.kind is Tok.INT]
    assert toks[0].value == 100


def test_newline_breaks_sex_continuation():
    # `1\n30` is two separate INT tokens — newline is not an inline
    # whitespace under the sex continuation rule.
    ks = [k for k in kinds("1\n30") if k is not Tok.NEWLINE]
    assert ks == [Tok.INT, Tok.INT]


def test_space_continuation_stops_at_non_digit():
    # `1 30 + 5` — sex literal `1 30` (= 90), then `+ 5`. The `+` breaks
    # the continuation chain.
    ks = kinds("1 30 + 5")
    assert ks == [Tok.SEX, Tok.PLUS, Tok.INT]


# --- strings ----------------------------------------------------------------

def test_simple_string():
    toks = [t for t in lex('"hello"') if t.kind is Tok.STRING]
    assert toks[0].value == b"hello"


def test_string_with_escapes():
    toks = [t for t in lex('"a\\nb\\tc"') if t.kind is Tok.STRING]
    assert toks[0].value == b"a\nb\tc"


def test_unterminated_string_raises():
    with pytest.raises(LexError, match="unterminated"):
        lex('"nope')


# --- operators --------------------------------------------------------------

def test_two_char_operators():
    assert kinds("== != <= >= && || -> => ..") == [
        Tok.EQEQ, Tok.BANGEQ, Tok.LE, Tok.GE,
        Tok.AMPAMP, Tok.PIPEPIPE, Tok.ARROW, Tok.FATARROW, Tok.DOTDOT,
    ]


def test_single_char_operators():
    assert kinds("+ - * / % = ! < > . , :") == [
        Tok.PLUS, Tok.MINUS, Tok.STAR, Tok.SLASH, Tok.PERCENT,
        Tok.EQ, Tok.BANG, Tok.LT, Tok.GT, Tok.DOT, Tok.COMMA, Tok.COLON,
    ]


def test_brackets_track_depth():
    assert kinds("( [ { } ] )") == [
        Tok.LPAREN, Tok.LBRACKET, Tok.LBRACE,
        Tok.RBRACE, Tok.RBRACKET, Tok.RPAREN,
    ]


# --- comments ---------------------------------------------------------------

def test_line_comment_consumed():
    # Comment followed by newline then `y`
    assert kinds("x // this is a comment\ny") == [Tok.IDENT, Tok.NEWLINE, Tok.IDENT]


def test_comment_at_eof():
    assert kinds("x // trailing") == [Tok.IDENT]


# --- newlines ---------------------------------------------------------------

def test_simple_newline_between_stmts():
    ks = kinds("x = 1\ny = 2")
    assert ks == [Tok.IDENT, Tok.EQ, Tok.INT, Tok.NEWLINE, Tok.IDENT, Tok.EQ, Tok.INT]


def test_blank_lines_are_one_newline():
    ks = kinds("x = 1\n\n\n\ny = 2")
    nl = sum(1 for k in ks if k is Tok.NEWLINE)
    assert nl == 1


def test_newline_suppressed_inside_parens():
    ks = kinds("foo(\n  x,\n  y,\n)")
    # no NEWLINE between LPAREN and RPAREN
    depth = 0
    for t in lex("foo(\n  x,\n  y,\n)"):
        if t.kind is Tok.LPAREN:
            depth += 1
        elif t.kind is Tok.RPAREN:
            depth -= 1
        elif t.kind is Tok.NEWLINE and depth > 0:
            pytest.fail("newline emitted inside parens")


def test_newline_suppressed_after_binary_op():
    ks = kinds("a = b +\nc")
    # no NEWLINE between `+` and `c`
    saw_plus = False
    for k in ks:
        if k is Tok.PLUS:
            saw_plus = True
            continue
        if saw_plus:
            assert k is not Tok.NEWLINE
            break


def test_newline_suppressed_after_comma():
    ks = kinds("[1,\n 2]")
    assert Tok.NEWLINE not in ks


def test_newline_suppressed_after_arrow():
    ks = kinds("fn f() ->\ni64 { 0 }")
    # ARROW should immediately be followed by TYPE_KW, no NEWLINE
    for i, k in enumerate(ks):
        if k is Tok.ARROW:
            assert ks[i + 1] is Tok.TYPE_KW
            break


# --- full program sketch ----------------------------------------------------

def test_fact_source_tokenizes():
    src = (
        "fn fact(n: i64) -> i64 {\n"
        "  if n < 2 { 1 } else { n * fact(n - 1) }\n"
        "}\n"
    )
    # just verify it doesn't raise and produces a reasonable token count
    toks = lex(src)
    assert toks[0].kind is Tok.FN
    assert toks[-1].kind is Tok.EOF
    # sanity: at least contains all the important bits
    kset = {t.kind for t in toks}
    assert Tok.IF in kset
    assert Tok.ELSE in kset
    assert Tok.LT in kset
    assert Tok.ARROW in kset


def test_sex_examples_from_spec():
    """Every entry in SPEC.md §3.6's example table should tokenize
    correctly under the space-separator convention. SEX tokens expose
    their (num, den) via the last two slots of `value`."""
    # (source, expected_kind, expected_check)
    # For SEX tokens, check is (num, den). For INT tokens, check is the int.
    expected: list[tuple[str, Tok, object]] = [
        ("1 30",       Tok.SEX, (90, 1)),
        ("1 30 0",     Tok.SEX, (5400, 1)),
        ("1;30",       Tok.SEX, (3, 2)),
        ("0;30",       Tok.SEX, (1, 2)),
        ("0;20",       Tok.SEX, (1, 3)),
        ("0;0 45",     Tok.SEX, (1, 80)),
        ("1;30 0 0",   Tok.SEX, (3, 2)),
        ("1;24 51 10", Tok.SEX, (30547, 21600)),  # YBC 7289's √2
    ]
    for src, want_kind, want_val in expected:
        toks = [t for t in lex(src) if t.kind in (Tok.INT, Tok.SEX)]
        assert len(toks) == 1, f"{src!r} tokenized to {toks}"
        assert toks[0].kind is want_kind, f"{src!r}: wanted {want_kind}, got {toks[0].kind}"
        if want_kind is Tok.SEX:
            _int_digits, _frac_digits, num, den = toks[0].value
            assert (num, den) == want_val, f"{src!r}: wanted {want_val}, got {(num, den)}"
        else:
            assert toks[0].value == want_val, f"{src!r}: wanted {want_val}, got {toks[0].value}"
