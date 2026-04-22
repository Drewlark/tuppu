# Tuppu — Language Specification

> Named for *tuppu*, the Akkadian word for a clay tablet — the medium
> on which all Babylonian knowledge was recorded. A Tuppu program is
> literally written on a tablet, and at runtime it writes onto tablets
> of its own (see §4.5).
> File extension: `.tpu`

## 0. Status

Draft v0.1. Everything in this document is negotiable. This spec exists to
give the compiler implementation a fixed target — we iterate on the spec,
not the code.

## 1. Philosophy

Tuppu is a small, statically-typed, ahead-of-time compiled language. It
exists to teach its author how to write a compiler, with a few design
choices picked for both pedagogical value and thematic identity:

- **Static types, no inference** (v0.1). Annotations are mandatory on
  function signatures and tablet declarations. Inside bodies, binding
  types are inferred from their initializer.
- **Expression-oriented.** Blocks evaluate to their last expression.
  There is no `return` keyword; use `yield` for early exit from a
  function body.
- **Single-assignment by default.** The `step` keyword introduces an
  immutable binding that lowers to exactly one LLVM SSA value — no
  `alloca`, no `load`/`store`. `mut` is the opt-in escape hatch.
- **Compile-time tables.** Lookup tables are a first-class declaration,
  populated by evaluating a generator at build time and baked into the
  binary as static data. Babylonian mathematics lived by its tables.
- **Sexagesimal literals.** Base-60 is a native literal form, with a
  native `rat` (rational) type so finite sexagesimal fractions are
  represented exactly rather than via `f64` approximation.
- **Tablet memory model.** No `malloc`, no GC. Fixed-size allocations
  (`tablet`) plus a chained-chunk primitive (`tablets`) that grows by
  appending new tablets — the Gilgamesh strategy. This is a real,
  well-known allocation pattern (unrolled list / bump-chunk arena) with
  the useful property that pointers into allocated data never move.

## 2. Notation

In grammar fragments:

- `UPPER` names are terminals (tokens).
- `lower` names are non-terminals.
- `?` means optional, `*` means zero or more, `+` means one or more.
- `|` separates alternatives.

## 3. Lexical structure

### 3.1 Whitespace and comments

Comments are line comments introduced by `//` and run to end of line.
Block comments are not supported in v0.1.

**Newlines are significant.** A newline terminates the current statement
*unless* one of the following is true, in which case it is ignored:

- We are inside unmatched `(`, `[`, or `{` brackets.
- The preceding token is a binary operator, `=`, `,`, or `->`.
- The preceding token is `if`, `else`, `while`, `table`, `yield`, or
  any other keyword that explicitly expects a following expression.

Blank lines are always ignored. Tuppu has no statement terminator
character — `;` is reserved exclusively for sexagesimal literals
(§3.6).

Long expressions break naturally:

```
step total = a + b +
             c + d              // one statement
foo(
  x,
  y,
)                               // one statement
```

### 3.2 Identifiers

```
IDENT   = [a-zA-Z_] [a-zA-Z0-9_]*
```

### 3.3 Keywords (reserved)

```
fn        step      mut       if        else      while
for       in        yield     true      false     as
table     tablet    tablets   release   struct    seal
i8 i16 i32 i64 u8 u16 u32 u64 bool f32 f64 rat  sex  dish

Char literals: `'a'`, `'\n'`, `'\\'`, `'\''` — single-byte, type `u8`.
```

### 3.4 Operators and punctuation

```
+  -  *  /  %
== != <  <= >  >=
&& || !
=  ->  =>
( ) { } [ ] ,  ;  .  ..
```

Note: `;` is used *only* inside sexagesimal literals. It is not a
statement terminator anywhere in the grammar.

### 3.5 Integer literals

```
DEC_INT = [0-9]+
HEX_INT = 0x [0-9a-fA-F]+
BIN_INT = 0b [01]+
```

Decimal integer literals default to `i64`. Suffixes not in v0.1.

### 3.6 Sexagesimal literals

Sexagesimal literals model Babylonian notation faithfully. A literal is
a sequence of decimal **digit groups** (each in `[0, 60)`) separated by
**inline whitespace**, with at most one `;` marking the **radix point**.
All sex literals have type `sex` (aka `dish`) — which is a distinct
type from `rat`, even though the two share a runtime representation.

We use **space** (not `,`) as the digit separator for two reasons: `,`
is purely an argument separator in the expression grammar, and
Python/C++ have already set a precedent for whitespace-inside-numeric-
literals.

```
DIGIT_GROUP = [0-9]+                       // value must be < 60 in sex form
SPACES      = (" " | "\t")+                // inline whitespace only, not "\n"
SEX_INT     = DIGIT_GROUP (SPACES DIGIT_GROUP)+
SEX_FRAC    = DIGIT_GROUP (SPACES? ";" SPACES? DIGIT_GROUP (SPACES DIGIT_GROUP)*)
SEX_LIT     = SEX_INT | SEX_FRAC
```

A literal becomes a **sex token** iff it contains a `;` OR has more
than one space-separated digit group. A single digit group with no
`;` stays an `INT`. Two or more `;` characters is a syntax error
("two radix points"); place values ≥ 60 are a lex error.

Examples:

| Literal       | Form        | Value exactly       | Default type |
|---------------|-------------|---------------------|--------------|
| `42`          | —           | 42                  | `i64`        |
| `1 30`        | integer     | 90                  | `sex`        |
| `1 30 0`      | integer     | 5400                | `sex`        |
| `1;30`        | fractional  | 3/2                 | `sex`        |
| `0;30`        | fractional  | 1/2                 | `sex`        |
| `0;20`        | fractional  | 1/3 (exact!)        | `sex`        |
| `0;0 45`      | fractional  | 45/3600 = 1/80      | `sex`        |
| `1;30 0 0`    | fractional  | 3/2 (reduced)       | `sex`        |
| `1;24 51 10`  | fractional  | 30547/21600 ≈ √2    | `sex`        |
| `1; 30`, `1 ; 30`, `1 ;30` | fractional | 3/2 (whitespace around `;` tolerated) | `sex` |

Sex values silently coerce to `rat` (no-op at runtime; same struct) or
to any integer type (via truncating `sdiv num, den`). Arithmetic on sex
values (`+`, `-`, `*`, `/`) is auto-lowered to rat arithmetic and emits
a compile-time warning, as native sex arithmetic — which would avoid
the i64/i64 bottleneck by keeping digit sequences and radix shifts
first-class — is a planned future feature. Comparisons (`==`, `<`, ...)
do not warn.

### 3.7 Boolean and string literals

```
BOOL     = "true" | "false"
STRING   = "\"" ([^"\\] | "\\" ["\\nrt0])* "\""
```

String literals have type `str` — a built-in seal (see §4.7) whose
contents live in the static section. Arbitrary bytes (including NUL)
are permitted; printing goes through `write(2)` so embedded NULs
survive unchanged.

## 4. Types

```
type = prim
     | "tablets" "[" INT "]" type    // chained growable
     | "*" type                      // raw pointer (see §4.7)
     | IDENT                         // primitive, user seal, or `str`

prim = "i8"  | "i16" | "i32" | "i64"
     | "u8"  | "u16" | "u32" | "u64"
     | "bool" | "f32" | "f64" | "rat"
     | "sex" | "dish"         // aliases for the sexagesimal type
```

Array size precedes the element type, matching Zig. This keeps `;`
reserved for sexagesimal literals only.

Examples:

```
tablets[256]Token  // chained storage, 256 tokens per tablet
*u8                // raw pointer to a byte (used inside `str`)
Point              // a user seal declared elsewhere
```

### 4.1 Primitives

Integer types are two's complement, fixed width. Arithmetic overflow on
signed types is undefined behavior (matching LLVM `nsw`); on unsigned
types it wraps. Floating types are IEEE 754.

### 4.2 `rat`

A rational number: `struct rat { num: i64, den: i64 }`, always stored
reduced, with `den > 0`. Arithmetic operations reduce on construction.
Overflow of either field traps at runtime.

Conversions:

```
i64  as rat    // exact, den = 1
rat  as f64    // deferred: "f64 not yet supported" in v0.1
rat  as i64    // truncation toward zero (signed division of num by den)
```

### 4.3 `sex` (alias: `dish`)

The Babylonian sexagesimal type — a digit sequence with a positional
radix, modelled faithfully and **distinct from `rat` at runtime**.
`sex` and `dish` are both keywords for the same compile-time type;
`dish` (Sumerian for the cuneiform vertical wedge) is the preferred
name in docs and library code.

Runtime layout (20 bytes):

```
seal sex {
  digits : [16]u8,   // base-60 digits; int digits first, then fractional
  radix  : u8,       // where fractional begins (also = count of int digits)
  count  : u8,       // total digits used (0..=16)
  sign   : i8,       // 0 positive, non-zero negative
  _pad   : u8,
}
```

Printing shows the scribe's digits: `println(1;30)` emits `1;30`,
`println(1;24 51 10)` emits `1;24 51 10`, and `println(-(1;30))`
emits `-1;30`.

Conversions:

```
sex  as rat    // real work: reconstruct num/den, reduce via gcd
sex  as i64    // reduces to rat then truncates toward zero
rat  as sex    // regularity-checked reconstruction — traps at runtime
               //   if den isn't 2^a·3^b·5^c (non-terminating sexagesimal)
i64  as sex    // decompose into base-60 digits (integer form)
sex  as f64    // deferred: "f64 not yet supported"
```

Sex values coerce silently to `rat` at typed binding sites
(`step x: rat = 1;30` works) — the conversion happens implicitly at
the coercion point.

**Native digit-form arithmetic:**

- `sex + sex` and `sex - sex` (Phase 2) stay in digit form end-to-end.
  No warning, no rat lowering. The runtime helper aligns radix points,
  SIMD-adds the 16-lane digit buffer, then propagates base-60 carries.
  Mixed signs dispatch to a magnitude compare + borrow-propagating
  subtract.
- `sex * sex` (Phase 3a) computes the product through rat internally
  (`(a as rat) * (b as rat)`), then reconstructs the sex digit form
  via `__tuppu_rat_to_sex` — which runs the regularity check and
  traps on non-terminating products. No compile-time warning; result
  type is sex. The integer ≤ 14-digit headroom covers any i64 · i64
  product (max 22 digits reduced is well within our 16-digit buffer
  because regular denominators cap frac digits sharply).
- `sex / sex` and `sex % sex` still lower to rat and emit a compile-
  time warning for the moment — native digit-form division is Phase
  3b. Escape-analysis rat-fallback specialization (where rat-only sex
  values skip the digit-form runtime entirely) is Phase 3c.

Field access: `x.num` and `x.den` are still allowed on sex (the
compiler reduces to rat first), but prefer an explicit `as rat`
cast at the call site if you care about the conversion cost.

### 4.4 Tablets — the memory model

- `tablet[N]T` is a single fixed-size allocation of `N` elements of
  `T`. In v0.1 this is a synonym for `[N]T`; distinction reserved for
  future use (distinct storage-class annotation).
- `tablets[N]T` is a chained sequence backed by a linked list of
  tablets, each of capacity `N`. Grows by appending new tablets; never
  resizes or moves existing data. Stable pointers.

Core operations on a `var toks: tablets[N]T`:

| Operation         | Effect                                                   |
|-------------------|----------------------------------------------------------|
| `toks.push(x)`    | Append; allocate a new tablet if the current one is full |
| `toks.len`        | Total number of elements across all tablets              |
| `toks[i]`         | Indexed access; bounds-checked; O(i/N)                   |
| `for x in toks`   | Iterate in insertion order                               |
| `release toks`    | Free the entire chain of tablets                         |

There is no general-purpose heap. All dynamic allocation in user code
goes through `tablets`.

### 4.5 Structs (alias: `seal`)

A user-defined composite type with named fields. Nominally typed —
two structurally identical structs with different names are different
types and do not implicitly convert.

```
struct Point { x: i64, y: i64 }
seal   Line  { a: Point, b: Point }
```

`struct` and `seal` are both keywords for the same declaration form;
`seal` is the preferred name in Babylonian-flavored examples because a
cylinder seal was the native metaphor for nominal identity — a named
composite design whose impressions in clay are the struct values.

Construction uses a trailing-braces literal:

```
step p: Point = Point { x: 3, y: 4 }
step l = Line { a: Point { x: 0, y: 0 }, b: p }
```

Fields may appear in any order in a literal but all declared fields
must be present. Field access is `p.x`. Forward references across
declarations are supported (declaration order is resolved by the
compiler); direct recursion is an error in v0.1.

Structs are passed and returned by value. Struct field mutation
(`p.x = 5`) is not yet supported; reassign the whole struct via `mut`.

Conversions: no implicit `as` casts involving structs. Explicit
reinterpret between same-layout structs is planned as a separate
operator, `impress` (see §14).

The built-in `str` type (§4.7) is itself a seal — nothing about it
is special at the type-system level, just at literal-production
time.

### 4.6 Raw pointers (`*T`)

`*T` is a raw, unmanaged pointer type — a machine address interpreted
as pointing to a value of type `T`. In v0.1 raw pointers are
intentionally **type-only**: you can hold one (as a field of a seal,
a function parameter, or a binding), pass it through, and the
compiler can use it internally, but there is no expression-level
syntax for dereferencing (`*p`), pointer arithmetic, or address-of.

Their purpose in v0.1 is to let the built-in `str` seal describe its
own contents and to leave room for future FFI and low-level stdlib
code.

### 4.7 `str` — variable-length strings

```
seal str {
  ptr: *u8,
  len: i64,
}
```

`str` is a built-in seal, auto-injected into every compilation
(including `--no-stdlib`). String literals produce `str` values;
the bytes live in a deduped global constant.

Operations:

| Syntax             | Result | Notes                                   |
|--------------------|--------|-----------------------------------------|
| `"literal"`        | `str`  | `.ptr` into static bytes, `.len` in bytes |
| `s.len`            | `i64`  | Byte length (free — ordinary field access) |
| `s[i]`             | `u8`   | Bounds-checked byte load through `s.ptr` |
| `print(s)`         | `()`   | Writes bytes verbatim via `write(2)`     |
| `println(s)`       | `()`   | Same, followed by `\n`                   |
| `fn f(s: str)`     | —      | Pass-by-value (pointer + length, 16 B)   |

Stdlib functions in `stdlib/str.tpu` cover the non-allocating byte-
level operations: `str_eq`, `str_starts_with`, `str_ends_with`,
`str_is_empty`, `str_index_of`. Anything that would need to
*produce* a new string (concat, slice, case-fold) is deferred until
v0.2 when we commit to a dynamic-string allocation strategy
(likely tablets-backed).

### 4.8 Function types

Functions are not first-class in v0.1. A `fn` declaration introduces a
name bound to a function; it can be called but not passed, stored, or
returned. (Tables get a targeted exception — see §9.)

## 5. Bindings

```
binding   = "step" IDENT (":" type)? "=" expr
          | "mut"  IDENT (":" type)? "=" expr
```

A `step` binding is immutable. It lowers to a single SSA value. It
cannot appear on the left of `=`.

A `mut` binding is mutable. It lowers to an `alloca` slot. Assignment
uses `=`:

```
mut count: i64 = 0
count = count + 1
```

Type annotations are optional on bindings; the initializer's type
determines the binding's type.

## 6. Expressions

Precedence, highest to lowest (same associativity as C unless noted):

```
1.  call, index, field           foo(x)   a[i]   p.field
2.  unary                        -x   !x
3.  cast                         x as T    (non-associative)
4.  * / %                        left
5.  + -                          left
6.  < <= > >=                    non-assoc
7.  == !=                        non-assoc
8.  &&                           left
9.  ||                           left
```

Blocks are expressions:

```
block_expr = "{" stmt* expr? "}"
```

A block evaluates to its trailing expression (if present), otherwise to
unit (`()`, with type `()`; unit does not have a literal form and
cannot be stored — the compiler elides it).

`if`/`else` is an expression. Both arms must produce the same type (or
one must be unreachable, e.g. `yield`).

```
if_expr = "if" expr block_expr ("else" (if_expr | block_expr))?
```

`while` is a statement; it has no value.

## 7. Statements

```
stmt = binding
     | assign
     | expr
     | while_stmt
     | yield_stmt

assign      = IDENT "=" expr               // IDENT must be mut
while_stmt  = "while" expr block_expr
yield_stmt  = "yield" expr?                // early return from fn
```

There are no semicolons. Statements are separated by newlines or by
the `}` that closes the enclosing block. Whitespace otherwise ignored.

## 8. Functions

```
fn_decl = "fn" IDENT "(" params? ")" ("->" type)? block_expr
params  = param ("," param)*
param   = IDENT ":" type
```

A function with no `->` clause returns unit. The body is a block; its
value is the return value. `yield expr` returns early.

```
fn add(a: i64, b: i64) -> i64 {
  a + b
}

fn fact(n: i64) -> i64 {
  if n < 2 { 1 } else { n * fact(n - 1) }
}
```

## 9. Tables — comptime lookup

```
table_decl = "table" IDENT "[" range "]" ":" type "=" expr
range      = expr ".." expr
```

Semantics: at compile time, for each integer `i` in the half-open range
`[lo, hi)`, the compiler evaluates `<expr>(i)` (the generator applied
to the index) and records the result at position `i - lo` in a static
array. `<expr>` must be a function reference or a lambda of type
`(i64) -> T`, and must be evaluable at comptime (no I/O, no mutable
state, no `tablets` allocation — see §13).

```
fn fact(n: i64) -> i64 {
  if n < 2 { 1 } else { n * fact(n - 1) }
}

table fact_table[0..20]: i64 = fact
// fact_table[10] is a compile-time-known load from a static array
```

Accessing `fact_table[i]` checks bounds at runtime (traps on OOB) and
loads from the static array. The generator function is **not** present
in the output binary unless it is also called at runtime from elsewhere.

Tables are the only place where function references escape their
declaration in v0.1.

## 10. Program structure

A program is a sequence of top-level declarations:

```
program = decl*
decl    = fn_decl | table_decl | struct_decl   // struct_decl v0.2
```

Compilation requires a function `fn main()` (unit return) or
`fn main() -> i32` (process exit code).

## 11. Standard library (v0.1)

Minimal, implemented as intrinsics lowered directly by the compiler.
The intrinsic names are reserved — user code cannot shadow them.

- `print(x)` — overloaded on `i8..i64`, `u8..u64`, `bool`, `rat` (future),
  and `[N]u8` strings. Writes to stdout without a newline.
- `println(x)` — same, with a trailing `\n`.
- `read_int() -> i64` — reads a decimal integer from stdin. Lowers to a
  `scanf("%lld", ...)` into an entry-block alloca.

Internally these lower to calls against libc (`printf`, `scanf`) declared
as external functions at module load. Program linking is done by
`clang`, which pulls in libc automatically.

Everything else — string manipulation, file I/O, formatted output
beyond the primitives — lives in user code or comes later.

## 12. Worked examples

### 12.1 Hello, world

```
fn main() -> i32 {
  println("Hello, Tuppu!")
  0
}
```

### 12.2 Recursive factorial

```
fn fact(n: i64) -> i64 {
  if n < 2 { 1 } else { n * fact(n - 1) }
}

fn main() -> i32 {
  println(fact(10))     // 3628800
  0
}
```

### 12.3 Named-step SSA style

```
fn quadratic(a: f64, b: f64, c: f64) -> f64 {
  step disc = b*b - 4.0*a*c
  step root = sqrt(disc)
  (-b + root) / (2.0*a)
}
```

Each `step` is one SSA value in the emitted IR — no stack slots.

### 12.4 Mutable loop counter

```
fn sum_to(n: i64) -> i64 {
  mut total: i64 = 0
  mut i:     i64 = 0
  while i < n {
    total = total + i
    i = i + 1
  }
  total
}
```

### 12.5 Sexagesimal arithmetic

```
fn main() -> i32 {
  step a: rat = 1;30                   // sex 1;30 silently coerces to rat
  step b: rat = 0;20                   // 1/3 exactly — not 0.3333...
  println(a + b)                       // 11/6  (exact, no f64 rounding)

  // YBC 7289 (c. 1800 BCE): the Babylonian √2 approximation.
  step sqrt2_approx: rat = 1;24 51 10  // = 30547/21600 ≈ 1.41421296
  println(sqrt2_approx)
  0
}
```

Sex arithmetic (mixing two sex values with `+`, `-`, `*`, `/`) emits a
compile-time warning that the operation was lowered to rat. To silence
the warning, cast to `rat` explicitly: `(a as rat) + (b as rat)`.

### 12.6 Babylonian table: Pythagorean triples up to N

```
// Plimpton-322 style: enumerate (a, b, c) with a^2 + b^2 = c^2, a < b
// Here: precompute a[i] = smallest b such that i^2 + b^2 is square, or 0.
fn smallest_companion(a: i64) -> i64 {
  mut b: i64 = 1
  while b < 10000 {
    step c2 = a*a + b*b
    // naive int sqrt; trap on overflow
    mut s: i64 = 0
    while (s+1) * (s+1) <= c2 { s = s + 1 }
    if s*s == c2 { yield b }
    b = b + 1
  }
  0
}

table companion[1..101]: i64 = smallest_companion
// The first 100 companions are baked into the binary at compile time.
```

### 12.7 Tablets (growable storage)

```
fn main() {
  var toks: tablets[256]i64
  mut i: i64 = 0
  while i < 10000 {
    toks.push(i * i)
    i = i + 1
  }
  println(toks.len)         // 10000
  println(toks[9999])       // 99980001
  release toks
}
```

`toks` grows by appending new 256-element tablets (~40 tablets total
here). Pointers into earlier tablets remain valid the whole time.

## 13. What is evaluable at comptime (v0.1)

A function is comptime-evaluable if its body uses only:

- Arithmetic and comparison on primitive types and `rat`
- `step` and `mut` bindings (local mutation is fine during comptime eval)
- `if`/`else`, `while`
- Calls to other comptime-evaluable functions
- Indexing into other `table`s already declared

Not comptime-evaluable: `tablets` allocation, `print`, file I/O,
calls into the runtime. A violation is a compile-time error at the
table declaration site.

## 14. Non-goals for v0.1

Explicitly out of scope — to keep the compiler reachable:

- Generics / polymorphism
- Sum types / pattern matching
- Closures or first-class functions (beyond table generators)
- Garbage collection
- General `malloc`/`free`
- Async, concurrency, threads
- Trait/interface system
- Modules (everything is one file in v0.1)
- Macros / metaprogramming beyond `table`
- String manipulation that allocates: concatenation, slicing, case
  conversion, case-insensitive compare (deferred until we settle on
  tablets-backed dynamic strings in v0.2)
- `read_line()` or any stdin-to-`str` intrinsic (same reason)
- Struct field mutation (`p.x = 5`); whole-struct `mut` reassignment works
- Recursive struct types (would require identified types + heap
  indirection)
- Reinterpret casts between same-layout structs — planned as the
  `impress` operator in v0.2 (see below)
- Package manager, build tool, formatter
- Type inference across function boundaries

v0.2 candidates (rough): modules / imports, dynamic strings (concat /
slice, tablets-backed), sum types with pattern match, first-class
functions, simple type inference in bodies, and the `impress`
reinterpret cast:

```
// v0.2 sketch — same-layout reinterpret, explicitly unsafe-flavored:
step v: Vector = impress p as Vector
```

Rationale for separating `impress` from `as`: `as` stays the safe,
semantics-preserving numeric conversion operator; `impress` is the
Babylonian-flavored escape hatch that treats the bits of one seal as
though they had been pressed from a different-named seal of identical
layout. Requires field-type-by-field-type equality between source and
target structs; codegen is a no-op. A future trait-driven
`into`-style conversion would be a third, distinct mechanism.

## 15. Grammar summary

```
program     = decl*
decl        = fn_decl | table_decl | struct_decl
fn_decl     = "fn" IDENT "(" params? ")" ("->" type)? block_expr
params      = param ("," param)*
param       = IDENT ":" type
table_decl  = "table" IDENT "[" expr ".." expr "]" ":" type "=" expr
struct_decl = ("struct" | "seal") IDENT "{" field ("," field)* ","? "}"
field       = IDENT ":" type

stmt        = binding | assign | aug_assign
            | while_stmt | for_stmt | yield_stmt | expr
binding     = ("step" | "mut") IDENT (":" type)? "=" expr
assign      = IDENT "=" expr
aug_assign  = IDENT ("+=" | "-=" | "*=" | "/=" | "%=") expr
while_stmt  = "while" expr block_expr
for_stmt    = "for" IDENT "in" expr block_expr
yield_stmt  = "yield" expr?

expr        = if_expr | block_expr | binary | unary | call | atom
if_expr     = "if" expr block_expr ("else" (if_expr | block_expr))?
block_expr  = "{" stmt* expr? "}"
call        = expr "(" args? ")"
args        = expr ("," expr)*
struct_lit  = IDENT "{" (IDENT ":" expr ("," IDENT ":" expr)* ","?)? "}"
atom        = IDENT | literal | struct_lit | "(" expr ")"
literal     = DEC_INT | HEX_INT | BIN_INT | SEX_LIT | CHAR | STRING | BOOL

type        = prim
            | "tablets" "[" DEC_INT "]" type
            | "*" type                      // raw pointer
            | IDENT                         // primitive, struct, or `str`
```
