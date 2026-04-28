#!/usr/bin/env bash
# Sweep all dvec-vs-ivec benches in this directory and print a summary
# table. A subdir is a "bench" iff it contains both `dvec.tpu` and
# `ivec.tpu` — we time those, take the best of 3 wall-clock runs each,
# and tabulate. Subdirs without that pair (e.g. ivec_vs_dvec/, which
# has its own per-T-size sweep harness) are skipped here; run their
# own scripts separately.
#
# Usage: bash bench/run.sh [bench_name ...]
#   No args runs every bench; named args restrict to those subdirs.
#
# Quirks worth knowing:
#   - Compile time is excluded — `tuppu build` runs upfront.
#   - 3 runs, take the minimum. Variance from GC / OS scheduling /
#     thermal throttling is real; the min is the closest thing to a
#     "no-noise" datapoint we can get without locking the CPU.
#   - The per-bench `.tpu` files print to stdout to defeat the
#     compiler's dead-code elimination on the loop's accumulator;
#     this script throws stdout away.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

# Collect bench dirs: every immediate subdir of bench/ holding both
# dvec.tpu and ivec.tpu.
benches=()
if [ "$#" -gt 0 ]; then
  for arg in "$@"; do
    benches+=("$arg")
  done
else
  for d in "$HERE"/*/; do
    name="$(basename "$d")"
    if [ -f "$d/dvec.tpu" ] && [ -f "$d/ivec.tpu" ]; then
      benches+=("$name")
    fi
  done
fi

# 3 runs, report best wall-clock ms for each binary. Build once.
time_best() {
  local bin="$1"
  local best=999999
  for i in 1 2 3; do
    local ms
    ms=$(.venv/bin/python -c "
import subprocess, time
t0 = time.monotonic()
subprocess.run(['$bin'], stdout=subprocess.DEVNULL, check=True)
print(int((time.monotonic() - t0) * 1000))
")
    if [ "$ms" -lt "$best" ]; then best=$ms; fi
  done
  echo "$best"
}

printf "%-22s %-12s %-12s %s\n" "bench" "dvec_ms" "ivec_ms" "ivec_vs_dvec"
printf "%-22s %-12s %-12s %s\n" "----------------------" "-------" "-------" "------------"
for b in "${benches[@]}"; do
  d="$HERE/$b"
  if [ ! -f "$d/dvec.tpu" ] || [ ! -f "$d/ivec.tpu" ]; then
    echo "skip: $b (missing dvec.tpu or ivec.tpu)" >&2
    continue
  fi
  ./tuppu build "$d/dvec.tpu" -o "$d/dvec_bin" >/dev/null 2>&1
  ./tuppu build "$d/ivec.tpu" -o "$d/ivec_bin" >/dev/null 2>&1
  d_ms=$(time_best "$d/dvec_bin")
  i_ms=$(time_best "$d/ivec_bin")
  # Ratio: ivec / dvec, expressed as a percentage. >100% means ivec slower.
  if [ "$d_ms" -eq 0 ]; then
    ratio="n/a"
  else
    ratio=$(.venv/bin/python -c "print(f'{$i_ms / $d_ms * 100:.0f}%')")
  fi
  printf "%-22s %-12s %-12s %s\n" "$b" "$d_ms" "$i_ms" "$ratio"
done
