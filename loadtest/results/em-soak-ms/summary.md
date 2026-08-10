# Load-suite run `20260810T100449Z-soak`

- **Scenario:** soak
- **Timestamp (UTC):** 2026-08-10T10:34:50Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:e2c1c712b035b6ac (17981 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 1800.8 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 2.103 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 40.979 | pass |
| NFR-PERF-011 | rss_slope_final_20min | least_squares_slope | <= 1.000 MiB/min | 0.086 | pass |
| NFR-PERF-012 | gpu_memory_growth_after_5min | max | <= 64 MiB | 93.000 | FAIL |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-REL-005 | unhandled_exceptions_in_logs | count | == 0 count | 0 | pass |
| NFR-REL-005 | splice_fallback_rate | proportion | <= 0.010 ratio | — | not-measured |
| NFR-SEC-001 | non_lan_endpoints_contacted | count | == 0 endpoints | 0 | pass |
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
| crashes | count | 0 |
| event_loop_lag | p99 | 2.103 |
| event_loop_lag | max | 40.979 |
| gpu_memory_growth_after_5min | max | 93.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 130.643 |
| non_lan_endpoints_contacted | count | 0 |
| rss_slope_final_20min | least_squares_slope | 0.086 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 2.690 |
| total_server_time_ms | p95 | 157.000 |
| total_server_time_ms | p99 | 166.000 |
| total_server_time_ms | max | 558 |
| unhandled_exceptions_in_logs | count | 0 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 1,
    "mean": 2.6902995169984933,
    "min": 2.6902995169984933,
    "max": 2.6902995169984933,
    "p50": 2.6902995169984933,
    "p90": 2.6902995169984933,
    "p95": 2.6902995169984933,
    "p99": 2.6902995169984933
  },
  "inter_result_interval_ms": {
    "count": 5391,
    "mean": 333.36870199109643,
    "min": 100.26084200035257,
    "max": 422.80774600112636,
    "p50": 373.2546609990095,
    "p90": 378.7206310007605,
    "p95": 380.6138854997698,
    "p99": 385.74155230071483
  },
  "cadence_abs_dev_ms": {
    "count": 5391,
    "mean": 84.6209526707471,
    "min": 0.0030832630204713496,
    "max": 172.8304217380878,
    "p50": 123.27743773717657,
    "p90": 128.751488737596,
    "p95": 130.64310623709162,
    "p99": 135.78782293776163
  },
  "total_server_time_ms": {
    "count": 5391,
    "mean": 151.20422927100722,
    "min": 135.0,
    "max": 558.0,
    "p50": 150.0,
    "p90": 155.0,
    "p95": 157.0,
    "p99": 166.0
  },
  "audio_age_lag_ms": {
    "count": 5392,
    "mean": 1362.1279974491742,
    "min": 1263.0045610003435,
    "max": 1688.7455849991966,
    "p50": 1401.8556014989372,
    "p90": 1406.2583235001512,
    "p95": 1408.2520511993607,
    "p99": 1415.6493536400376
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 1714,
      "mean": 3281.765066638711,
      "min": 3204.74609375,
      "max": 3289.87890625,
      "p50": 3283.84765625,
      "p90": 3284.140625,
      "p95": 3284.29296875,
      "p99": 3289.83984375
    },
    "cpu_pct": {
      "count": 1714,
      "mean": 41.53838973162194,
      "min": 0.0,
      "max": 75.3,
      "p50": 41.1,
      "p90": 44.9,
      "p95": 45.8,
      "p99": 47.7
    },
    "gpu_used_mib": {
      "count": 1714,
      "mean": 2641.4084014002333,
      "min": 2619.0,
      "max": 2750.0,
      "p50": 2639.0,
      "p90": 2655.0,
      "p95": 2673.0,
      "p99": 2707.87
    },
    "gpu_util_pct": {
      "count": 1714,
      "mean": 38.82030338389732,
      "min": 0.0,
      "max": 54.0,
      "p50": 39.0,
      "p90": 40.0,
      "p95": 40.0,
      "p99": 42.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:55336"
    ]
  }
}
```
