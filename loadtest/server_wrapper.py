"""Launch the backend under measurement scaffolding — no server source edits.

Runs the unmodified server (load_all_engines + serve from src/grpc_server.py)
on an asyncio loop that also carries a loop-lag sampler task, and writes:

- results-dir/server.json  — pid, start time, loop-lag CSV path
- results-dir/loop_lag.csv — "<epoch_seconds>,<lag_ms>" at >= 10 Hz
  (NFR-PERF-006; epoch timestamps let each scenario window the samples to
  its own run, so a long-lived wrapper cannot contaminate later assertions)

Loop lag is measured as the drift of `asyncio.sleep(period)` wakeups on the
SAME loop that serves inference: if a coroutine blocks the loop, the sampler
wakes late by exactly the blocked time. This is the standard single-process
lag probe and needs no change to application behaviour (working-rules
scaffolding allowance).

Usage (from SonicSightBackend/):
    python -m loadtest.server_wrapper --out loadtest/results/current
"""

import argparse
import asyncio
import json
import logging
import os
import time

from . import paths  # noqa: F401  (sys.path bootstrap)

# The server modules log via module loggers but only run_servers.py/main.py
# call basicConfig; without a handler every INFO line (including the
# "gRPC server started" readiness line and all log-based metrics' inputs)
# would be dropped. Same format as src/run_servers.py.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

import grpc_server  # noqa: E402  — the unmodified server module


SAMPLE_PERIOD_S = 0.05  # 20 Hz sampling, above the >=10 Hz condition


async def loop_lag_sampler(csv_path):
    with open(csv_path, "w", encoding="utf-8", buffering=1) as f:
        while True:
            before = time.monotonic()
            await asyncio.sleep(SAMPLE_PERIOD_S)
            after = time.monotonic()
            lag_ms = max(0.0, (after - before - SAMPLE_PERIOD_S) * 1000.0)
            f.write(f"{time.time():.3f},{lag_ms:.3f}\n")


async def main(out_dir, port):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "loop_lag.csv")
    info = {"pid": os.getpid(), "port": port,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "loop_lag_csv": csv_path}
    with open(os.path.join(out_dir, "server.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    sampler = asyncio.ensure_future(loop_lag_sampler(csv_path))
    try:
        await grpc_server.serve(port=port)
    finally:
        sampler.cancel()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="results directory for server.json/loop_lag.csv")
    ap.add_argument("--port", type=int, default=50051)
    args = ap.parse_args()

    grpc_server.load_all_engines()
    asyncio.run(main(args.out, args.port))
