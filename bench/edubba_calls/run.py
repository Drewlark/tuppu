#!/usr/bin/env python3
"""Time the edubba-call benches in both `shadow` and `llvm` GC modes,
at -O0 and -O2. Targets from issue #8:

- `inline.tpu` (Counter.bump 1M): expect 5-10× post-migration -O2 win
  vs. shadow -O0, once @llvm.gcroot lets the optimizer inline.
- `outoflineheavier.tpu` (Bag.set 100K): expect 1.5-2× post-migration
  -O2 win, even though full inlining isn't realistic — mem2reg of
  param slots is the lever.

Usage:
    .venv/bin/python bench/edubba_calls/run.py

Prints a table to stdout. The post-GC-migration capture lands in
`BENCH_POST_GC_MIGRATION.md` from this output.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "edubba_bench"


def _ensure_runtime() -> Path:
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


def _build(tpu: Path, name: str, opt: str, framework: str) -> Path:
    """Build once for a (file, opt-level, framework) combination.
    `framework` is "shadow" or "llvm" — passed through env so the
    Codegen instance reads it at construction. `opt` ∈ {O0, O2}."""
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / f"{name}_{framework}_{opt}"
    sys.path.insert(0, str(ROOT / "src"))
    env_save = os.environ.get("TUPPU_GC_FRAMEWORK")
    os.environ["TUPPU_GC_FRAMEWORK"] = framework
    try:
        from tuppu.driver import compile_sources_to_ir, stdlib_files
        files = stdlib_files() + [tpu]
        sources = [(str(p), p.read_text()) for p in files]
        ir = compile_sources_to_ir(sources)
    finally:
        if env_save is None:
            os.environ.pop("TUPPU_GC_FRAMEWORK", None)
        else:
            os.environ["TUPPU_GC_FRAMEWORK"] = env_save
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        (td_p / "in.ll").write_text(ir)
        if opt == "O0":
            subprocess.run(
                ["llc", "-filetype=obj", "-relocation-model=pic",
                 str(td_p / "in.ll"), "-o", str(td_p / "obj.o")],
                check=True,
            )
        else:
            subprocess.run(
                ["opt", f"-passes=default<{opt}>", "-S",
                 str(td_p / "in.ll"), "-o", str(td_p / "opt.ll")],
                check=True,
            )
            subprocess.run(
                ["llc", "-filetype=obj", "-relocation-model=pic",
                 str(td_p / "opt.ll"), "-o", str(td_p / "obj.o")],
                check=True,
            )
        subprocess.run(
            ["clang", str(td_p / "obj.o"), str(_ensure_runtime()),
             "-o", str(out)],
            check=True,
        )
    return out


def _time_best(bin_path: Path, runs: int = 5) -> float:
    times = []
    for _ in range(runs):
        t0 = time.monotonic()
        subprocess.run(
            [str(bin_path)], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, check=True,
        )
        times.append((time.monotonic() - t0) * 1000)
    return min(times)


BENCHES = [
    # `inline.tpu` has only an i64 field, so no gcroot emits — measures
    # pure call-overhead removal at -O2.
    ("inline", "bench/edubba_calls/inline.tpu"),
    # `inline_with_gc.tpu` has a str field, so every `bump` call would
    # pay push_root/pop_roots in shadow mode. The win at -O2 reflects
    # both inlining AND gcroot-blocking removal.
    ("inline_gc", "bench/edubba_calls/inline_with_gc.tpu"),
    ("outofline", "bench/edubba_calls/outoflineheavier.tpu"),
]


def main() -> int:
    print(f"# edubba-call bench")
    header = f"{'bench':<14s} {'mode':<10s} {'opt':<4s} {'ms':>10s}"
    print(header)
    print("-" * len(header))
    matrix: dict[tuple[str, str, str], float] = {}
    for name, src in BENCHES:
        for framework in ("shadow", "llvm"):
            for opt in ("O0", "O2"):
                if framework == "shadow" and opt == "O2":
                    # Issue #8: shadow mode at -O2 is the SIGSEGV
                    # regime we're avoiding. Skip — recording the
                    # absence is the point.
                    continue
                bin_path = _build(ROOT / src, name, opt, framework)
                ms = _time_best(bin_path)
                matrix[(name, framework, opt)] = ms
                print(f"{name:<14s} {framework:<10s} {opt:<4s} {ms:>10.2f}")
    # Summary ratios — issue #8 targets
    print()
    print("# issue-8 target check")
    print(f"{'bench':<14s} {'metric':<28s} {'value':<10s} {'target':<10s} {'status':<6s}")
    print("-" * 70)
    targets = [
        # `inline.tpu` (no GC fields): target is pure call-elim — call
        # overhead is small (2-3ns/call), loop body is one i64 add, so
        # post-inline we expect ~2× rather than the 5-10× the issue
        # spec quotes for the gc-blocked case. `inline_gc.tpu` is the
        # one that exercises the issue's stated target.
        ("inline", "llvm@O2 vs shadow@O0", 0.50, "<= 0.50"),
        ("inline_gc", "llvm@O2 vs shadow@O0", 0.20, "<= 0.20"),
        ("outofline", "llvm@O2 vs shadow@O0", 0.67, "<= 0.67"),
    ]
    for name, label, target_ratio, target_str in targets:
        base = matrix.get((name, "shadow", "O0"))
        opt = matrix.get((name, "llvm", "O2"))
        if base is None or opt is None:
            ratio_str = "n/a"
            status = "??"
        else:
            ratio = opt / base
            ratio_str = f"{ratio:.2f}"
            status = "PASS" if ratio <= target_ratio else "MISS"
        print(f"{name:<14s} {label:<28s} {ratio_str:<10s} {target_str:<10s} {status:<6s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
