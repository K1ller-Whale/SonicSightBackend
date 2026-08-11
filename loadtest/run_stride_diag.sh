#!/usr/bin/env bash
# D-P5-1 stride diagnostic: dump-enabled server + one 90 s multisensory
# session; cycles.csv window_delta is the direct evidence of the actual
# window-advance quantisation. Waits for the FUNC-001 runs to finish so it
# never co-tenants a measurement.
set -u
cd "$(dirname "$0")/.."
source ~/.venv/bin/activate
R="loadtest/results"
PROG="$R/stride_diag.log"
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$PROG"; }

until grep -q "FUNC001-COMPLETE" "$R/func001.log" 2>/dev/null; do sleep 60; done
say "starting stride diagnostic"

mkdir -p "$R/current-wsl5"
SONICSIGHT_DUMP_STREAM=1 SONICSIGHT_DUMP_DIR="$R/stride-dump" \
  python -u -m loadtest.server_wrapper --out "$R/current-wsl5" \
  > "$R/current-wsl5/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 150); do
  grep -qE "gRPC server started" "$R/current-wsl5/server.log" && break
  sleep 3
done
say "server ready"

python - <<'PYEOF' >> "$PROG" 2>&1
import asyncio, sys
sys.path.insert(0, "loadtest")
from loadtest import driver, media
prof = driver.PROFILES["multisensory"]
src = media.SyntheticSource(seed=0, sample_rate=prof["sample_rate"],
                            frame_rate=prof["frame_rate"],
                            frame_kind=prof["frame_kind"])
st = asyncio.run(driver.run_session("127.0.0.1", 50051, "multisensory",
                                    src, 90.0))
iv = st.inter_result_intervals_ms()
import statistics
print("results:", len(st.results),
      "interval median:", round(statistics.median(iv), 1) if iv else None)
PYEOF
kill $SRV; sleep 3

d=$(ls -dt "$R"/stride-dump/*/ 2>/dev/null | head -1)
say "dump dir: $d"
if [ -n "$d" ] && [ -f "$d/cycles.csv" ]; then
  awk -F, 'NR>1 {print $5}' "$d/cycles.csv" | sort -n | uniq -c \
    | tee -a "$PROG"
fi
say STRIDE-DIAG-COMPLETE
