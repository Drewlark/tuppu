"""Generate Tuppu benchmark sources comparing ivec<T>::push vs dvec<T>::push
for a struct T whose size we vary. Emits two .tpu files per (size, n) pair:
one that runs only an ivec push loop, one that runs only a dvec push loop.

Each program touches the value before sinking to len, so the optimizer
can't elide the construction. Element size is 8 * fields bytes (i64 fields).
"""
from __future__ import annotations

import sys
from pathlib import Path


def big_decl(fields: int) -> str:
    body = ", ".join(f"x{i}: i64" for i in range(fields))
    return f"tablet Big {{ {body} }}"


def make_big_fn(fields: int) -> str:
    inits = ", ".join(f"x{i}: seed + {i}" for i in range(fields))
    return (
        "fn make_big(seed: i64) -> Big {\n"
        f"  Big {{ {inits} }}\n"
        "}\n"
    )


def ivec_program(fields: int, n: int) -> str:
    return (
        big_decl(fields) + "\n\n"
        + make_big_fn(fields) + "\n"
        + "fn main() -> i32 {\n"
        + "  mut v: ivec<Big>\n"
        + "  mut i: i64 = 0\n"
        + f"  while i < {n} {{\n"
        + "    v.push(make_big(i))\n"
        + "    i = i + 1\n"
        + "  }\n"
        + "  println(v.len)\n"
        + "  println(v[0].x0)\n"
        + f"  println(v[{n - 1}].x0)\n"
        + "  0\n"
        + "}\n"
    )


def dvec_program(fields: int, n: int) -> str:
    return (
        big_decl(fields) + "\n\n"
        + make_big_fn(fields) + "\n"
        + "fn main() -> i32 {\n"
        + "  mut v: dvec<Big>\n"
        + "  mut i: i64 = 0\n"
        + f"  while i < {n} {{\n"
        + "    v.push(make_big(i))\n"
        + "    i = i + 1\n"
        + "  }\n"
        + "  println(v.len)\n"
        + "  println(v[0].x0)\n"
        + f"  println(v[{n - 1}].x0)\n"
        + "  0\n"
        + "}\n"
    )


def main() -> int:
    out = Path(__file__).parent
    sizes = [1, 4, 16, 32, 64, 128]   # i64 fields => bytes = 8 * size
    n_default = 200000
    for fields in sizes:
        n = n_default
        (out / f"ivec_T{fields}.tpu").write_text(ivec_program(fields, n))
        (out / f"dvec_T{fields}.tpu").write_text(dvec_program(fields, n))
    print(f"wrote {len(sizes)*2} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
