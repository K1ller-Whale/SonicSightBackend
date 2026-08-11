# Load-suite run `20260810T075055Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-10T08:02:56Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:98f7877bbe5216fc (17587 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 720.6 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-002 | time_to_first_nonbuffering_result | p95 | <= 4.000 s | 2.768 | pass |
| NFR-PERF-002 | inter_result_interval_abs_dev_from_125ms | p95 | <= 125.000 ms | 128.759 | FAIL |
| NFR-PERF-003 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-004 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-005 | total_server_time_ms | p95 | <= 200 ms | 152.000 | pass |
| NFR-PERF-005 | total_server_time_ms | max | <= 1000 ms | 312 | pass |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 2.122 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 5.655 | pass |
| NFR-PERF-008 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-FUNC-001 | — | — |  —  | — | condition-mismatch |
| NFR-FUNC-002 | results_meeting_multisensory_contract | proportion | >= 1.000 ratio | 1.000 | pass |
| NFR-FUNC-003 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-SEC-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-001 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-003 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-004 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-MAINT-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-MAINT-003 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-FLEX-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-FLEX-003 | — | — |  —  | — | not-evaluable-by-this-suite |

## Measured metrics

| Metric | Statistic | Value |
|---|---|---|
| event_loop_lag | p99 | 2.122 |
| event_loop_lag | max | 5.655 |
| inter_result_interval_abs_dev_from_125ms | p95 | 128.759 |
| results_meeting_multisensory_contract | proportion | 1.000 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 2.768 |
| total_server_time_ms | p95 | 152.000 |
| total_server_time_ms | p99 | 154.000 |
| total_server_time_ms | max | 312 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 2.6661390970475187,
    "min": 2.2893150839990994,
    "max": 6.97642909500064,
    "p50": 2.451736620999327,
    "p90": 2.529845785999896,
    "p95": 2.5461554800003796,
    "p99": 6.090374372000591
  },
  "inter_result_interval_ms": {
    "count": 2033,
    "mean": 326.8632709542547,
    "min": 3.2557999993514386,
    "max": 393.7082980000923,
    "p50": 373.5173989998657,
    "p90": 377.79181559981225,
    "p95": 378.80862440015335,
    "p99": 381.6039693202765
  },
  "cadence_abs_dev_ms": {
    "count": 2033,
    "mean": 86.03738610141113,
    "min": 0.001304263189467747,
    "max": 246.72152426368712,
    "p50": 123.71278073665735,
    "p90": 127.99711193698158,
    "p95": 129.2483779367626,
    "p99": 145.80975993485072
  },
  "total_server_time_ms": {
    "count": 2033,
    "mean": 152.76586325627153,
    "min": 134.0,
    "max": 4844.0,
    "p50": 148.0,
    "p90": 151.0,
    "p95": 153.0,
    "p99": 164.0400000000002
  },
  "audio_age_lag_ms": {
    "count": 2054,
    "mean": 1309.9814686616292,
    "min": 1261.6377879994616,
    "max": 5974.51318200001,
    "p50": 1276.2889509999695,
    "p90": 1279.547738099427,
    "p95": 1281.647917650571,
    "p99": 2381.159010470331
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 688,
      "mean": 2911.2402627634447,
      "min": 2612.27734375,
      "max": 2924.51953125,
      "p50": 2911.828125,
      "p90": 2916.09375,
      "p95": 2916.1220703125,
      "p99": 2924.4692578125
    },
    "cpu_pct": {
      "count": 688,
      "mean": 39.160755813953486,
      "min": 0.0,
      "max": 123.3,
      "p50": 40.0,
      "p90": 43.9,
      "p95": 44.9,
      "p99": 82.47800000000001
    },
    "gpu_used_mib": {
      "count": 688,
      "mean": 2039.8706395348838,
      "min": 1514.0,
      "max": 2051.0,
      "p50": 2042.0,
      "p90": 2045.0,
      "p95": 2045.0,
      "p99": 2045.0
    },
    "gpu_util_pct": {
      "count": 688,
      "mean": 36.911337209302324,
      "min": 0.0,
      "max": 83.0,
      "p50": 38.0,
      "p90": 39.0,
      "p95": 40.0,
      "p99": 60.129999999999995
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:43260"
    ]
  }
}
```
