#!/usr/bin/env bash
# Campaign 4 (2026-08-10): final citable numbers on a fresh server running
# the precise-inference-timing code (PERF-004/005 re-target verification).
set -u
cd "$(dirname "$0")/.."
source ~/.venv/bin/activate
R="loadtest/results"
PROG="$R/campaign4.log"
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$PROG"; }

mkdir -p "$R/current-wsl10"
say "BOOT starting"
python -u -m loadtest.server_wrapper --out "$R/current-wsl10" \
  > "$R/current-wsl10/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 150); do
  grep -qE "gRPC server started" "$R/current-wsl10/server.log" && break
  sleep 3
done
say "BOOT ready"
SI="$R/current-wsl10/server.json"

run() { local out="$1"; shift
  say "START $out"
  python -m loadtest.run "$@" --environment E-M --server-info "$SI" \
    --out "$R/$out" > "$R/$out-console.log" 2>&1
  say "DONE $out exit=$?"
}

run em4-baseline     baseline
run em4-baseline-ms  baseline --model multisensory
run em4-load-ms      load --sessions 1 --model multisensory
run em4-smoke        smoke
kill $SRV
say CAMPAIGN4-COMPLETE
