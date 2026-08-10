#!/usr/bin/env bash
# Campaign 3 (2026-08-10): verify the D-P5-1 fix (ingest/inference overlap +
# slack-aware window_min_advance=4480) and regression-check the shared
# stream loop for the SoP paths; finish the deconfounded PERF-011/012 soak.
set -u
cd "$(dirname "$0")/.."
source ~/.venv/bin/activate
R="loadtest/results"
PROG="$R/campaign3.log"
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$PROG"; }
wait_ready() { for i in $(seq 1 150); do
    grep -qE "gRPC server started" "$1" 2>/dev/null && return 0; sleep 3
  done; return 1; }

mkdir -p "$R/current-wsl9"
say "BOOT starting"
python -u -m loadtest.server_wrapper --out "$R/current-wsl9" \
  > "$R/current-wsl9/server.log" 2>&1 &
SRV=$!
wait_ready "$R/current-wsl9/server.log" || { say "BOOT FAILED"; exit 1; }
say "BOOT ready"
SI="$R/current-wsl9/server.json"; SL="$R/current-wsl9/server.log"
CLOSE='Stream closed for StreamProcess'

run() { local out="$1"; shift
  say "START $out"
  python -m loadtest.run "$@" --environment E-M --server-info "$SI" \
    --out "$R/$out" > "$R/$out-console.log" 2>&1
  say "DONE $out exit=$?"
}

run em3-baseline-ms   baseline --model multisensory
run em3-load-ms       load --sessions 1 --model multisensory
run em3-baseline      baseline
run em3-baseline-replay baseline --video data/test_instruments.mp4 \
  --runs 2 --ttfr-probe-seconds 10 --duration 60 --replay-runs 3
run em3-soak-clean    soak --duration 1800 --model sonicsight \
  --server-log "$SL" --close-marker "$CLOSE"
run em3-smoke         smoke
kill $SRV
say CAMPAIGN3-COMPLETE
