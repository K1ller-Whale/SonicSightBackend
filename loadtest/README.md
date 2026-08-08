# SonicSight backend load and stress suite

Produces the numbers Section 10 of `docs/ANALYSIS_REPORT.md` cites. Every
assertion threshold is read at run time from `docs/nfr/nfr_targets.yaml`
(the single source of truth, NFR-MAINT-004) — no threshold literal exists in
this package; stimulus parameters (payload sizes, cycle counts) are CLI
arguments, not thresholds.

**Driver choice.** The service is gRPC bidirectional streaming with in-band
control messages, per-stream metadata model selection, and per-result
metrics; generic tools (`ghz`, `k6`+xk6-grpc, Locust) model none of that
well. This suite is an asyncio driver on the project's own generated stubs,
generalising the validated `replay_client.py` protocol to N concurrent
sessions, injections, and switching — the full paragraph of justification is
at the top of `loadtest/driver.py`.

**Perceived-lag proxy definition.** `audio_age_lag_ms` = result arrival time
minus the send time of the capture chunk whose `timestamp_ms` the result
echoes. It measures capture-to-result age at the driver, excluding the
client's playback jitter buffer; the true at-the-ear lag adds that buffer.
Reported as characterisation, not thresholded (report §5.9).

## Setup

```powershell
cd SonicSightBackend
pip install -r requirements.txt -r loadtest\requirements.txt
```

## Start the server under measurement scaffolding

The wrapper runs the *unmodified* server with a loop-lag sampler on the same
asyncio loop (20 Hz; NFR-PERF-006) and writes `server.json` + `loop_lag.csv`:

```powershell
mkdir loadtest\results\current -Force
python -u -m loadtest.server_wrapper --out loadtest\results\current *> loadtest\results\current\server.log
```

(`-u` disables block buffering so the log stays current for log-based
metrics; `*>` merges streams without PowerShell 5.1's per-line
NativeCommandError wrapping of `2>&1`. Watch readiness with
`Get-Content -Wait -Tail 5 loadtest\results\current\server.log` — the line
is "gRPC server started".)

Pass `--server-info loadtest\results\current\server.json` to every scenario
so RSS/CPU/GPU monitoring and loop-lag assertions attach; pass
`--server-log loadtest\results\current\server.log` for log-based metrics
(exception counts, open/close parity).

## Scenarios — exact commands

Common: `--host 127.0.0.1 --port 50051 --model sonicsight`. Results land in
`loadtest/results/<runid>/{results.json, summary.md}` with automatic run
metadata (git heads, checkpoints, GPU/driver, versions, YAML hash).

| Scenario | Purpose | Command |
|---|---|---|
| smoke | CI subset: proto identity, unit suite, report↔YAML match, default-model + health probes | `python -m loadtest.run smoke --server-info ...` (add `--offline` for no-server CI) |
| baseline | TTFR (20 openings) + one ≥7-min steady session; add `--model sonicsight-pixel` for query RTT; add `--video <clip>` for replay sanity (NFR-FUNC-001; clip is operator-supplied and SHA-256-hashed into metadata) | `python -m loadtest.run baseline --server-info ...` |
| load | Hold at `--sessions 2`; NFR-PERF-007 floor, GPU steady ceiling | `python -m loadtest.run load --sessions 2 --server-info ...` |
| stress | Step 1..`--max-sessions` until a threshold breaches — the knee | `python -m loadtest.run stress --max-sessions 6 --server-info ...` |
| spike | `--burst-sessions 4` simultaneous opens on an idle server | `python -m loadtest.run spike --server-info ...` |
| soak | `--duration 1800` (or 3600) single session; leak detection | `python -m loadtest.run soak --duration 1800 --server-info ... --server-log ...` |
| failure-injection | Malformed/degenerate media vs a co-running healthy session; oversized payload; 50 abrupt disconnects | `python -m loadtest.run failure-injection --server-info ... --server-log ...` |
| switching | 100 cancel-and-reopen cycles alternating `sonicsight,sonicsight-pixel` | `python -m loadtest.run switching --server-info ...` |

Exit code is non-zero when any applicable assertion FAILs. E-M-conditional
and matrix-conditional targets are reported as *skipped with reason* on E-D,
never silently dropped (`--environment E-M` on an E-M-class host enables
them).

## Reproducing every figure in report §10

1. Start the wrapper (above) and wait for "gRPC server started".
2. `python -m loadtest.run smoke --server-info <...>\server.json`
3. `python -m loadtest.run baseline --server-info <...>` and again with
   `--model sonicsight-pixel`, and with `--video <reference clip>`.
4. `python -m loadtest.run load --sessions 2 --server-info <...>`
5. `python -m loadtest.run stress --max-sessions 6 --server-info <...>`
6. `python -m loadtest.run spike --server-info <...>`
7. `python -m loadtest.run soak --duration 1800 --server-info <...> --server-log <...>`
8. `python -m loadtest.run failure-injection --server-info <...> --server-log <...>`
9. `python -m loadtest.run switching --server-info <...>`

Each `summary.md` is committed; raw `results.json` is committed when small
(the working rules exclude multi-MB captures — summarise instead).

## Known limits (declared, not discovered later)

- The GPU-OOM injection (NFR-REL-007) is not automated; run the server with
  a squeezed `MS_GPU_MEM_FRACTION`/reduced VRAM to induce it, or discharge
  the target by the documented analysis fallback.
- Log-based metrics need `--server-log`. The server logs a stream-open line
  ("Client connected for StreamProcess", the `--open-marker` default) but
  **no stream-close line**, so NFR-REL-002's open/close parity stays
  not-measured unless a `--close-marker` is supplied for a future close
  line — a recorded server-observability limitation, not a suite gap.
- `splice_fallback_rate` (NFR-REL-005) parses the server's stream-dump
  summary, which only prints when the server runs with
  `SONICSIGHT_DUMP_STREAM=1`; soak runs that need it must set that flag.
- GPU memory readings are device-global `nvidia-smi` values, not
  per-process: keep the E-D desktop otherwise idle during runs, or the
  ±64 MiB deltas absorb foreign allocations.
- `multisensory` scenarios require an E-M-class host; on E-D they are
  reported as skipped (TensorFlow absent).
- Targets whose scenario is `unit`/`inspection`/`analysis`/`mobile-*`
  (NFR-FUNC-003, INT-001–004, SEC-002, MAINT-002/003, FLEX-002/003) are
  outside this suite and appear in every summary as
  *not-evaluable-by-this-suite*; the unit subset is Phase 5 scaffolding,
  the mobile subset Phase 6.
- Soak's `crashes` metric is a server-aliveness probe after the run; a
  per-stream error surfaces separately in extras, not as a crash.
