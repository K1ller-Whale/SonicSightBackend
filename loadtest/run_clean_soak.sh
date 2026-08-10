#!/usr/bin/env bash
# Clean (no-dump) 30-min halves soak: PERF-011/012 without the diagnostic
# recorder's own RSS growth confound (the dump-mode soak buffers ~5.3
# MiB/min of PCM in process memory by design). Chained after the stride
# diagnostic.
set -u
cd "$(dirname "$0")/.."
source ~/.venv/bin/activate
R="loadtest/results"
PROG="$R/clean_soak.log"
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$PROG"; }

until grep -q "STRIDE-DIAG-COMPLETE" "$R/stride_diag.log" 2>/dev/null; do
  sleep 60
done
say "starting clean soak"

mkdir -p "$R/current-wsl6"
python -u -m loadtest.server_wrapper --out "$R/current-wsl6" \
  > "$R/current-wsl6/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 150); do
  grep -qE "gRPC server started" "$R/current-wsl6/server.log" && break
  sleep 3
done
say "server ready"
python -m loadtest.run soak --duration 1800 --model sonicsight \
  --environment E-M --server-info "$R/current-wsl6/server.json" \
  --server-log "$R/current-wsl6/server.log" \
  --close-marker "Stream closed for StreamProcess" \
  --out "$R/em2-soak-clean" > "$R/em2-soak-clean-console.log" 2>&1
say "clean soak exit=$?"
kill $SRV
say CLEAN-SOAK-COMPLETE
