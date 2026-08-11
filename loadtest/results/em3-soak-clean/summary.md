# Load-suite run `20260810T172323Z-soak`

- **Scenario:** soak
- **Timestamp (UTC):** 2026-08-10T17:53:24Z
- **SonicSightBackend:** `7cb179bb4064` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:052388852cee79cb (37692 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 1800.3 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-011 | rss_slope_final_20min | least_squares_slope | <= 1.000 MiB/min | 1.030 | FAIL |
| NFR-PERF-012 | gpu_memory_growth_after_5min | max | <= 128 MiB | 1038.000 | FAIL |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-REL-005 | crashes | count | == 0 count | 0 | pass |
| NFR-REL-005 | unhandled_exceptions_in_logs | count | == 0 count | 0 | pass |
| NFR-REL-005 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-REL-005 | splice_fallback_rate | proportion | <= 0.010 ratio | — | not-measured |
| NFR-SEC-001 | non_lan_endpoints_contacted | count | == 0 endpoints | 0 | pass |
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
| crashes | count | 0 |
| event_loop_lag | p99 | 2.082 |
| event_loop_lag | max | 4.299 |
| gpu_memory_growth_after_5min | max | 1038.000 |
| inference_time_ms | p95 | 123.000 |
| inference_time_ms | p99 | 123.000 |
| inference_time_ms | max | 1248 |
| inter_result_interval_abs_dev_from_125ms | p95 | 0.893 |
| non_lan_endpoints_contacted | count | 0 |
| rss_slope_final_20min | least_squares_slope | 1.030 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 6.128 |
| total_server_time_ms | p95 | 124.000 |
| total_server_time_ms | p99 | 125.000 |
| total_server_time_ms | max | 1249 |
| unhandled_exceptions_in_logs | count | 0 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 1,
    "mean": 6.1280246870001065,
    "min": 6.1280246870001065,
    "max": 6.1280246870001065,
    "p50": 6.1280246870001065,
    "p90": 6.1280246870001065,
    "p95": 6.1280246870001065,
    "p99": 6.1280246870001065
  },
  "inter_result_interval_ms": {
    "count": 14333,
    "mean": 125.15246683820557,
    "min": 60.48350099990785,
    "max": 1250.1391089999743,
    "p50": 125.1890009998533,
    "p90": 125.46416019995377,
    "p95": 125.56840039997041,
    "p99": 125.97232400028588
  },
  "cadence_abs_dev_ms": {
    "count": 14333,
    "mean": 0.5926799542494243,
    "min": 6.213192250470456e-05,
    "max": 1125.150446868455,
    "p50": 0.3659818687344796,
    "p90": 0.8214211311235433,
    "p95": 0.8930021314620262,
    "p99": 1.1570215486324291
  },
  "total_server_time_ms": {
    "count": 14333,
    "mean": 123.89123002860532,
    "min": 120.0,
    "max": 1249.0,
    "p50": 124.0,
    "p90": 124.0,
    "p95": 124.0,
    "p99": 125.0
  },
  "audio_age_lag_ms": {
    "count": 14334,
    "mean": 3363.931837021003,
    "min": 3250.4884690006293,
    "max": 4502.008956000282,
    "p50": 3376.655145500081,
    "p90": 3377.5309850998383,
    "p95": 3501.387239100495,
    "p99": 3502.22233769975
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 1718,
      "mean": 3155.7349547984577,
      "min": 3132.42578125,
      "max": 3186.34765625,
      "p50": 3150.40625,
      "p90": 3180.115625,
      "p95": 3182.21875,
      "p99": 3182.22265625
    },
    "cpu_pct": {
      "count": 1718,
      "mean": 50.89481955762515,
      "min": 0.0,
      "max": 73.8,
      "p50": 51.4,
      "p90": 53.5,
      "p95": 54.1,
      "p99": 55.282999999999994
    },
    "gpu_used_mib": {
      "count": 1718,
      "mean": 2047.4965075669384,
      "min": 1904.0,
      "max": 2942.0,
      "p50": 2042.0,
      "p90": 2076.0,
      "p95": 2076.0,
      "p99": 2076.0
    },
    "gpu_util_pct": {
      "count": 1718,
      "mean": 32.118742724097785,
      "min": 0.0,
      "max": 62.0,
      "p50": 32.0,
      "p90": 33.0,
      "p95": 33.0,
      "p99": 34.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:44356"
    ]
  }
}
```
