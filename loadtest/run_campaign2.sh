#!/usr/bin/env bash
# Post-fix verification campaign (2026-08-10): D1 stride fix, D4 close-line
# parity, REL-005 60-min halves soak under SONICSIGHT_DUMP_STREAM=1.
set -u
cd "$(dirname "$0")/.."
source ~/.venv/bin/activate
R="loadtest/results"
PROG="$R/campaign2.log"
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$PROG"; }

wait_ready() { # wait_ready <log>
  for i in $(seq 1 150); do
    grep -qE "gRPC server started" "$1" 2>/dev/null && return 0
    sleep 3
  done
  return 1
}

# ── Boot A: normal server, all engines ────────────────────────────────
mkdir -p "$R/current-wsl2"
say "BOOT-A starting"
python -u -m loadtest.server_wrapper --out "$R/current-wsl2" \
  > "$R/current-wsl2/server.log" 2>&1 &
SRV=$!
wait_ready "$R/current-wsl2/server.log" || { say "BOOT-A FAILED"; exit 1; }
say "BOOT-A ready (pid $SRV)"
SI="$R/current-wsl2/server.json"; SL="$R/current-wsl2/server.log"
CLOSE='Stream closed for StreamProcess'

run() { local out="$1"; shift
  say "START $out"
  python -m loadtest.run "$@" --environment E-M --server-info "$SI" \
    --out "$R/$out" > "$R/$out-console.log" 2>&1
  say "DONE $out exit=$?"
}

run em2-baseline-ms  baseline --model multisensory
run em2-load-ms      load --sessions 1 --model multisensory
run em2-failure      failure-injection --server-log "$SL" --close-marker "$CLOSE"
run em2-smoke        smoke
kill $SRV; sleep 5
say "BOOT-A stopped"

# ── Boot B: dump-enabled server for the 60-min halves soak ────────────
mkdir -p "$R/current-wsl3"
say "BOOT-B starting (SONICSIGHT_DUMP_STREAM=1)"
SONICSIGHT_DUMP_STREAM=1 python -u -m loadtest.server_wrapper \
  --out "$R/current-wsl3" > "$R/current-wsl3/server.log" 2>&1 &
SRV=$!
wait_ready "$R/current-wsl3/server.log" || { say "BOOT-B FAILED"; exit 1; }
say "BOOT-B ready (pid $SRV)"
SI="$R/current-wsl3/server.json"; SL="$R/current-wsl3/server.log"
run em2-soak-halves  soak --duration 3600 --model sonicsight \
  --server-log "$SL" --close-marker "$CLOSE"
kill $SRV
say CAMPAIGN2-COMPLETE
