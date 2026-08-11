# Load-suite run `20260810T171820Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-10T17:23:21Z
- **SonicSightBackend:** `7cb179bb4064` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:052388852cee79cb (37692 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 300.9 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | time_to_first_nonbuffering_result | p95 | <= 8.000 s | 6.253 | pass |
| NFR-PERF-002 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-003 | inter_result_interval_abs_dev_from_125ms | p95 | <= 62.500 ms | 0.944 | pass |
| NFR-PERF-004 | inference_time_ms | p95 | <= 140 ms | 123.000 | pass |
| NFR-PERF-004 | inference_time_ms | p99 | <= 250 ms | 124.000 | pass |
| NFR-PERF-004 | inference_time_ms | max | <= 1500 ms | 374 | pass |
| NFR-PERF-004 | total_server_time_ms | report | report —  | {'p95': 124.0, 'p99': 125.0, 'max': 375} | characterisation |
| NFR-PERF-005 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 2.055 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 2.266 | pass |
| NFR-PERF-008 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | — | not-measured |
| NFR-PERF-015 | — | — |  —  | — | condition-mismatch |
| NFR-FUNC-001 | replay_left_right_pearson | max_over_runs | <= 0.350 correlation | 0.187 | pass |
| NFR-FUNC-002 | — | — |  —  | — | condition-mismatch |
| NFR-FUNC-004 | — | — |  —  | — | condition-mismatch |
| NFR-FUNC-003 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-001 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-003 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-004 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-SEC-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-MAINT-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-MAINT-003 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-FLEX-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-FLEX-003 | — | — |  —  | — | not-evaluable-by-this-suite |

## Measured metrics

| Metric | Statistic | Value |
|---|---|---|
| event_loop_lag | p99 | 2.055 |
| event_loop_lag | max | 2.266 |
| inference_time_ms | p95 | 123.000 |
| inference_time_ms | p99 | 124.000 |
| inference_time_ms | max | 374 |
| inter_result_interval_abs_dev_from_125ms | p95 | 0.944 |
| replay_left_right_pearson | max_over_runs | 0.187 |
| time_to_first_nonbuffering_result | p95 | 6.253 |
| total_server_time_ms | p95 | 124.000 |
| total_server_time_ms | p99 | 125.000 |
| total_server_time_ms | max | 375 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 3,
    "mean": 6.252925593666987,
    "min": 6.252053801999864,
    "max": 6.253444505000516,
    "p50": 6.253278474000581,
    "p90": 6.253411298800529,
    "p95": 6.253427901900523,
    "p99": 6.2534411843805175
  },
  "inter_result_interval_ms": {
    "count": 490,
    "mean": 124.58475788571377,
    "min": 57.05800000032468,
    "max": 126.72350000048027,
    "p50": 125.18529999988459,
    "p90": 125.5275991002236,
    "p95": 125.7418800004416,
    "p99": 126.09839499989903
  },
  "cadence_abs_dev_ms": {
    "count": 490,
    "mean": 0.8688648111936587,
    "min": 0.004462131774971567,
    "max": 67.9306621311946,
    "p50": 0.3729873680902074,
    "p90": 0.8704622310426658,
    "p95": 0.9482126816885736,
    "p99": 1.7393395378425571
  },
  "total_server_time_ms": {
    "count": 490,
    "mean": 125.23673469387755,
    "min": 120.0,
    "max": 375.0,
    "p50": 124.0,
    "p90": 124.0,
    "p95": 124.0,
    "p99": 125.0
  },
  "audio_age_lag_ms": {
    "count": 493,
    "mean": 3253.6577539655104,
    "min": 3249.9361849995694,
    "max": 3502.679903000171,
    "p50": 3251.754085000357,
    "p90": 3252.473576599914,
    "p95": 3252.6837766001336,
    "p99": 3308.8203222803713
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 288,
      "mean": 3172.7725287543403,
      "min": 3137.734375,
      "max": 3182.15234375,
      "p50": 3174.80859375,
      "p90": 3182.01953125,
      "p95": 3182.140625,
      "p99": 3182.15234375
    },
    "cpu_pct": {
      "count": 288,
      "mean": 44.38263888888889,
      "min": 0.0,
      "max": 60.3,
      "p50": 50.6,
      "p90": 53.63,
      "p95": 54.6,
      "p99": 56.113
    },
    "gpu_used_mib": {
      "count": 288,
      "mean": 2036.6805555555557,
      "min": 2032.0,
      "max": 2055.0,
      "p50": 2036.0,
      "p90": 2040.0,
      "p95": 2044.0,
      "p99": 2055.0
    },
    "gpu_util_pct": {
      "count": 288,
      "mean": 28.05902777777778,
      "min": 0.0,
      "max": 36.0,
      "p50": 32.0,
      "p90": 33.0,
      "p95": 33.0,
      "p99": 34.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:33812",
      "::ffff:127.0.0.1:41590",
      "::ffff:127.0.0.1:44568",
      "::ffff:127.0.0.1:55518"
    ]
  }
}
```
