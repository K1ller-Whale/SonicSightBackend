#!/usr/bin/env bash
# Full E-M measurement campaign against the long-lived wrapper server.
# Exit codes 1 (FAIL) / 2 (not-measured) are findings, not aborts.
set -u
cd "$(dirname "$0")/.."
source ~/.venv/bin/activate
SI="loadtest/results/current-wsl/server.json"
SL="loadtest/results/current-wsl/server.log"
R="loadtest/results"
PROG="$R/campaign.log"

say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$PROG"; }
vram() { nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu \
  --format=csv,noheader | sed "s/^/$(date -u +%H:%M:%S) $1 /" >> "$R/vram_stages.log"; }

run() { # run <outdir> <args...>
  local out="$1"; shift
  say "START $out"
  python -m loadtest.run "$@" --environment E-M --server-info "$SI" \
    --out "$R/$out" > "$R/$out-console.log" 2>&1
  say "DONE $out exit=$?"
  vram "after-$out"
}

vram idle-warm-server
run em-baseline        baseline
run em-baseline-pixel  baseline --model sonicsight-pixel
run em-baseline-ms     baseline --model multisensory
run em-load            load --sessions 2
run em-load-ms         load --sessions 1 --model multisensory
run em-stress          stress --max-sessions 6
run em-stress-ms       stress --max-sessions 3 --model multisensory
run em-spike           spike --burst-sessions 4
run em-switching-ms    switching --cycles 100 --models sonicsight,multisensory
run em-failure         failure-injection --server-log "$SL"
run em-soak-ms         soak --duration 1800 --model multisensory --server-log "$SL"
say CAMPAIGN-COMPLETE
