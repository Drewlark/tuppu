#!/usr/bin/env python3
"""Time the GC-migration benches and print one row per bench.

Usage:
    .venv/bin/python bench/gc_migration/run_preflight.py [opt-level]

Default opt-level is `O0` (the driver default — no opt run). Pass
`O2` to additionally optimize each IR through `opt -passes='default<O2>'`
before linking. The lua_interp fixture is included.

Each bench builds once, then runs 5× and reports best wall-clock.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "preflight"


def _ensure_runtime() -> Path:
    """Compile / cache the GC runtime object the bench's -O2 path needs."""
    BUILD.mkdir(parents=True, exist_ok=True)
    src = ROOT / "runtime" / "tuppu_gc.c"
    obj = BUILD / "tuppu_gc.o"
    if obj.exists() and obj.stat().st_mtime >= src.stat().st_mtime:
        return obj
    subprocess.run(
        ["clang", "-c", "-O2", "-Wall", "-o", str(obj), str(src)],
        check=True,
    )
    return obj


def _build(tpu: Path, name: str, opt: str) -> Path:
    """Build a Tuppu source. For O0, defer to `tuppu build`; for higher
    opt levels, drive the IR -> opt -> llc -> clang chain directly."""
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / name
    if opt == "O0":
        subprocess.run(
            [str(ROOT / "tuppu"), "build", str(tpu), "-o", str(out)],
            check=True, stderr=subprocess.DEVNULL,
        )
        return out
    # Higher opt levels: emit IR via tuppu, then opt + llc + clang.
    sys.path.insert(0, str(ROOT / "src"))
    from tuppu.driver import compile_sources_to_ir, stdlib_files
    files = stdlib_files() + [tpu]
    sources = [(str(p), p.read_text()) for p in files]
    ir = compile_sources_to_ir(sources)
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        (td_p / "in.ll").write_text(ir)
        subprocess.run(
            ["opt", "-passes=" + f"default<{opt}>", "-S",
             str(td_p / "in.ll"), "-o", str(td_p / "opt.ll")],
            check=True,
        )
        subprocess.run(
            ["llc", "-filetype=obj", "-relocation-model=pic",
             str(td_p / "opt.ll"), "-o", str(td_p / "opt.o")],
            check=True,
        )
        subprocess.run(
            ["clang", str(td_p / "opt.o"), str(_ensure_runtime()),
             "-o", str(out)],
            check=True,
        )
    return out


def _time_best(bin_path: Path, runs: int = 5, env: dict | None = None) -> float:
    """Best-of-N wall-clock of running `bin_path`. Stdout is dropped so
    the bench programs can `println` without polluting the harness."""
    times = []
    for _ in range(runs):
        t0 = time.monotonic()
        subprocess.run(
            [str(bin_path)], stdout=subprocess.DEVNULL,
            env={**os.environ, **(env or {})}, check=True,
        )
        times.append((time.monotonic() - t0) * 1000)
    return min(times)


BENCHES = [
    ("fib35", "bench/gc_migration/fib35.tpu"),
    ("counter_bump", "bench/gc_migration/counter_bump.tpu"),
    ("str_concat", "bench/gc_migration/str_concat.tpu"),
    ("struct_copy", "bench/gc_migration/struct_copy.tpu"),
    ("lua_interp", "examples/lua_interp.tpu"),
]


def main(argv: list[str]) -> int:
    opt = argv[1] if len(argv) > 1 else "O0"
    if opt not in {"O0", "O1", "O2", "O3"}:
        print(f"unknown opt level {opt!r}", file=sys.stderr)
        return 2
    print(f"# pre-flight bench at -{opt}")
    print(f"{'bench':<18s} {'ms (best of 5)':>14s}")
    for name, src in BENCHES:
        bin_path = _build(ROOT / src, f"{name}_{opt}", opt)
        ms = _time_best(bin_path)
        print(f"{name:<18s} {ms:>14.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
