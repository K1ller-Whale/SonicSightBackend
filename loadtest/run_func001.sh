#!/usr/bin/env bash
# FUNC-001 discharge on the in-domain clip (guitar+cello duet,
# sha16 f33e3e700cc86f64). Waits for campaign2 to finish so the replay
# never co-tenants a timing measurement, then:
#   1. streaming replay x3 via the baseline scenario (NFR-FUNC-001)
#   2. offline /predict-path pearson on the same clip (for the record)
set -u
cd "$(dirname "$0")/.."
source ~/.venv/bin/activate
R="loadtest/results"
PROG="$R/func001.log"
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$PROG"; }

until grep -q "CAMPAIGN2-COMPLETE\|FAILED" "$R/campaign2.log" 2>/dev/null; do
  sleep 60
done
say "campaign2 finished — starting FUNC-001 runs"

mkdir -p "$R/current-wsl4"
python -u -m loadtest.server_wrapper --out "$R/current-wsl4" \
  > "$R/current-wsl4/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 150); do
  grep -qE "gRPC server started" "$R/current-wsl4/server.log" && break
  sleep 3
done
say "server ready"

python -m loadtest.run baseline --video data/test_instruments.mp4 \
  --runs 2 --ttfr-probe-seconds 10 --duration 60 --replay-runs 3 \
  --environment E-M --server-info "$R/current-wsl4/server.json" \
  --out "$R/em2-baseline-replay" > "$R/em2-baseline-replay-console.log" 2>&1
say "replay scenario exit=$?"
kill $SRV; sleep 5

python3 - <<'PYEOF' >> "$PROG" 2>&1
import os, sys, numpy as np, soundfile as sf
sys.path.insert(0, "src")
os.chdir("src")
from video_preprocessor import VideoPreprocessor
from inference import inference
inference.load_model()
out = "/tmp/offline_instr"
os.makedirs(out, exist_ok=True)
job = VideoPreprocessor().preprocess("../data/test_instruments.mp4", out)
inference.eval_single_heatmap(job.audio_path, job.frames_dir, job.frame_count, out)
l, _ = sf.read(os.path.join(out, "pred_left.wav"))
r, _ = sf.read(os.path.join(out, "pred_right.wav"))
n = min(len(l), len(r))
print("OFFLINE pearson(L,R) test_instruments:", round(float(np.corrcoef(l[:n], r[:n])[0,1]), 4))
PYEOF
say FUNC001-COMPLETE
