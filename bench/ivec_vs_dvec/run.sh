#!/usr/bin/env bash
# Build and time each ivec/dvec push benchmark, 3 runs, report best wall time.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

echo "fields  bytes  variant  best_ms"
for fields in 1 4 16 32 64 128; do
  bytes=$((fields * 8))
  for variant in dvec ivec; do
    src="$HERE/${variant}_T${fields}.tpu"
    bin="$HERE/${variant}_T${fields}"
    ./tuppu build "$src" -o "$bin" >/dev/null 2>&1
    best=999999
    for i in 1 2 3; do
      ms=$(.venv/bin/python -c "
import subprocess, time
t0 = time.monotonic()
subprocess.run(['$bin'], stdout=subprocess.DEVNULL, check=True)
print(int((time.monotonic() - t0) * 1000))
")
      if [ "$ms" -lt "$best" ]; then best=$ms; fi
    done
    printf "%-7s %-6s %-7s %s\n" "$fields" "$bytes" "$variant" "$best"
  done
done
