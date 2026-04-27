# Tuppu

A small, statically-typed, AOT-compiled language with a Babylonian
flavor. Written end-to-end in Python on top of `llvmlite`, with a
mark-sweep GC runtime in C. The language is the byproduct of trying
to learn how compilers work in earnest — the design picks its
constraints to keep the surface area small while still leaving room
for real ideas.

> *tuppu* is the Akkadian word for a clay tablet — the medium on
> which Babylonian knowledge was recorded. A Tuppu program is
> literally written on a tablet (`.tpu`), and at runtime it writes
> onto tablets of its own.

## Quickstart

You need Python 3.11+, `clang`, and a working LLVM (any version
recent enough to keep `llvmlite` happy). Then:

```sh
git clone https://github.com/Drewlark/tuppu
cd tuppu
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Compile + run any example:
./tuppu run examples/hello.tpu

# Build a binary you can keep:
./tuppu build examples/fib.tpu -o fib
./fib
```

`./tuppu` is a shell wrapper around `python -m tuppu`. The bundled
stdlib (`stdlib/*.tpu`) is auto-discovered and linked into every
build — there's no `import` keyword in v0.1.

## A taste

```tuppu
// Hello.
fn main() -> i32 {
  println("Hello, Tuppu!")
  0
}
```

```tuppu
// Sexagesimal arithmetic, exact via the rat type.
fn main() -> i32 {
  step a: rat = 1;30           // 1.5 in base-60
  step b: rat = 0;20           // 1/3 exactly
  println(a + b)               // 11/6 — no f64 rounding
  0
}
```

```tuppu
// Generic linked list, all nodes arena-allocated.
fn main() -> i32 {
  mut store: tablets[16]Node<i64>
  mut head: wedge Node<i64> = lost
  head = list_push(store, head, 1)
  head = list_push(store, head, 2)
  println(list_len(head))      // 2
  0
}
```

```tuppu
// Vec<T> + Map<T> from stdlib.
fn main() -> i32 {
  mut tally: Map<i64>
  map_set(tally, "barley", 10)
  map_set(tally, "emmer",  6)
  println(map_get(tally, "barley", 0))   // 10
  println(map_has(tally, "wheat"))       // false
  0
}
```

## Why it looks the way it does

Tuppu trades familiar features for fewer moving parts. A few
opinionated picks:

- **`step` vs `mut`.** Bindings default to single-assignment via the
  `step` keyword, which lowers to exactly one SSA value — no alloca,
  no load/store. `mut` is the explicit opt-in for alloca-backed
  mutable bindings. Most code doesn't need it.
- **`yield`, not `return`.** Tuppu is expression-oriented; blocks
  evaluate to their tail expression. `yield <value>` is the early
  exit. The fn body's tail expression is its return.
- **Sexagesimal literals.** `1;30` is base-60 (= 3/2). Any literal
  containing `;` (or with multiple digit groups separated by spaces)
  is a `sex` ("dish") value — a Babylonian-faithful digit sequence
  that auto-promotes to `rat` for arithmetic. Useful for the dataset
  Babylonian mathematics actually uses; pleasant for non-base-10
  fractions in any program.
- **Tablets are the memory model.** A `tablet` is a fixed-size
  product type (struct). A `tablets[N]T` is a chunk-chained,
  pointer-stable, append-only growable arena of T — chunks of N
  elements linked by a tail pointer, allocated by the GC. Pushing
  never moves existing elements, so `wedge T` (tablet handles) into
  the arena stay valid for its lifetime.
- **Mark-sweep GC.** Every heap allocation goes through a precise
  collector with shadow-stack rooting. The runtime is in
  `runtime/tuppu_gc.c` — small, no thread story, designed to be
  comprehensible in one sitting. `TUPPU_GC_STRESS=1` forces a
  collection on every allocation, used in tests to surface missed
  roots.
- **Comptime tables.** `table foo: i64 = (lo..hi) -> generator()` is
  a first-class declaration evaluated at compile time and baked into
  the binary as static data. The reciprocal-table example in
  `examples/reciprocal_table.tpu` is the prototypical use.

The full grammar + semantics live in [`SPEC.md`](./SPEC.md).

## Standard library

Bundled `.tpu` files in `stdlib/`, all written in pure Tuppu (no
"compiler magic" beyond the intrinsics):

| File | What it provides |
|---|---|
| `str.tpu` | `str_eq`, `str_starts_with`, `str_index_of`, `str_repeat`, `str_concat` (variadic), `int_to_str` / `bool_to_str` / `rat_to_str`, etc. |
| `sex.tpu` | Sexagesimal helpers — `hms_to_rat_hours`, `rat_hours_to_seconds`, etc. |
| `rat.tpu` | Rational helpers — `abs`, `reciprocal`, `mean`. |
| `list.tpu` | `Node<T>`, `list_push`, `list_len`, `list_find`, `list_contains`. |
| `vec.tpu` | `Vec<T>` over `tablets[64]T` — push, get, set, swap, reverse, find, contains. |
| `map.tpu` | `Map<T>` (string-keyed, insertion-ordered, linear-scan) — get, set, has, len. |

Adding a `.tpu` file to `stdlib/` is the only step — the build picks
it up automatically.

## Examples

Each `.tpu` in `examples/` is registered in `tests/test_examples.py`
with its expected stdout / exit code. Notable ones:

- `hello.tpu` — `Hello, Tuppu!`.
- `fib.tpu` / `fact.tpu` — recursion baseline.
- `fizzbuzz.tpu` — control flow practice.
- `sexagesimal.tpu` — rat arithmetic + the YBC 7289 √2 demo.
- `linked_list.tpu` — `Node<T>` from stdlib.
- `scribe_ledger.tpu` — `Vec<str>` + `Map<i64>` from stdlib.
- `reciprocal_table.tpu` — comptime table.
- `omens.tpu` — sum types (`seal`) and pattern matching.
- `lua_interp.tpu` — 1,500-line Lua subset interpreter, the
  regression magnet for GC + closure handling.

## How the compiler is laid out

For anyone reading the source:

```
src/tuppu/
├── lexer.py              tokens
├── parser.py             AST
├── ast.py                AST node defs
├── typecheck.py          types + monomorphization
├── effects.py            (currently unused — kept for future opt passes)
├── comptime.py           comptime table evaluation
├── codegen/              AST -> LLVM IR
│   ├── __init__.py       Codegen class scaffolding + entry point
│   ├── module.py         top-level driver, fn / decl emission, mono
│   ├── stmt.py           statements + cleanup-frame / GC plumbing
│   ├── expr.py           expressions + dispatch
│   ├── access.py         field / index / slice / aggregate literals
│   ├── types.py          _lower_type, type descriptors, _coerce
│   ├── seals.py          sum-type codegen + pattern matching
│   ├── intrinsics.py     print / read_int / int_to_str / etc.
│   ├── tablets.py        tablets monomorphization + helpers
│   ├── strs.py           str runtime (concat, slice, etc.)
│   ├── rat.py            rat reduction + arithmetic
│   ├── sex.py            sexagesimal literal lowering
│   └── _common.py        shared LLVM types + Variable dataclass
├── driver.py             CLI orchestration
└── runtime/tuppu_gc.c    mark-sweep GC (linked into every binary)
```

Tests live in `tests/`. `test_gc_torture.py` and `test_lvalue.py` are
the most useful regression checks during compiler work — both run in
both normal and `TUPPU_GC_STRESS=1` modes.

## Status

Pre-1.0. The language is real enough to host its own Lua interpreter
end-to-end (see `examples/lua_interp.tpu`) but small parts of the
surface are still moving. `NEXT.md` tracks what's next; `SPEC.md` is
the source of truth for syntax + semantics.
