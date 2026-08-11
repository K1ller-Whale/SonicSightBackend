# Load-suite run `20260810T131349Z-soak`

- **Scenario:** soak
- **Timestamp (UTC):** 2026-08-10T14:13:58Z
- **SonicSightBackend:** `f840ce16afdd` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:c1b14b0c73772c80 (36367 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 3608.2 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-011 | rss_slope_final_20min | least_squares_slope | <= 1.000 MiB/min | 6.329 | FAIL |
| NFR-PERF-012 | gpu_memory_growth_after_5min | max | <= 128 MiB | 676.000 | FAIL |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-REL-005 | crashes | count | == 0 count | 0 | pass |
| NFR-REL-005 | unhandled_exceptions_in_logs | count | == 0 count | 0 | pass |
| NFR-REL-005 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-REL-005 | splice_fallback_rate | proportion | <= 0.010 ratio | 0.000 | pass |
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
| event_loop_lag | p99 | 2.045 |
| event_loop_lag | max | 7557.391 |
| gpu_memory_growth_after_5min | max | 676.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 7.579 |
| non_lan_endpoints_contacted | count | 0 |
| rss_slope_final_20min | least_squares_slope | 6.329 |
| sequence_gap_rate | proportion | 0.000 |
| splice_fallback_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 6.041 |
| total_server_time_ms | p95 | 57.000 |
| total_server_time_ms | p99 | 64.000 |
| total_server_time_ms | max | 2128 |
| unhandled_exceptions_in_logs | count | 0 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 1,
    "mean": 6.040665225002158,
    "min": 6.040665225002158,
    "max": 6.040665225002158,
    "p50": 6.040665225002158,
    "p90": 6.040665225002158,
    "p95": 6.040665225002158,
    "p99": 6.040665225002158
  },
  "inter_result_interval_ms": {
    "count": 28753,
    "mean": 124.99217020776955,
    "min": 0.21290899894665927,
    "max": 2132.2525149989815,
    "p50": 124.96501300120144,
    "p90": 128.2981469994411,
    "p95": 130.01402519876137,
    "p99": 136.8212222409784
  },
  "cadence_abs_dev_ms": {
    "count": 28753,
    "mean": 3.0852527555883085,
    "min": 1.0133381920240936e-05,
    "max": 2007.2638528674622,
    "p50": 1.3917588714582507,
    "p90": 5.119276868780388,
    "p95": 7.579084668240054,
    "p99": 24.659256973255793
  },
  "total_server_time_ms": {
    "count": 28753,
    "mean": 49.171669043230274,
    "min": 29.0,
    "max": 2128.0,
    "p50": 48.0,
    "p90": 53.0,
    "p95": 57.0,
    "p99": 64.0
  },
  "audio_age_lag_ms": {
    "count": 28754,
    "mean": 3380.12521068682,
    "min": 2950.3142310022668,
    "max": 6794.562553001015,
    "p50": 3421.6349519992946,
    "p90": 3550.4874741000094,
    "p95": 3557.3011540500374,
    "p99": 3563.077046240687
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 3100,
      "mean": 2643.383874747984,
      "min": 2362.90234375,
      "max": 3105.28515625,
      "p50": 2640.4609375,
      "p90": 2754.425390625,
      "p95": 2763.2845703125,
      "p99": 2769.3526171875
    },
    "cpu_pct": {
      "count": 3100,
      "mean": 59.26877419354839,
      "min": 0.0,
      "max": 108.5,
      "p50": 58.8,
      "p90": 64.0,
      "p95": 66.3,
      "p99": 76.80299999999993
    },
    "gpu_used_mib": {
      "count": 3097,
      "mean": 1804.1656441717791,
      "min": 1302.0,
      "max": 2383.0,
      "p50": 1800.0,
      "p90": 1897.0,
      "p95": 1915.1999999999998,
      "p99": 1933.0
    },
    "gpu_util_pct": {
      "count": 3097,
      "mean": 31.73103002906038,
      "min": 0.0,
      "max": 54.0,
      "p50": 32.0,
      "p90": 33.0,
      "p95": 33.0,
      "p99": 34.0
    },
    "nvidia_smi_available": false,
    "remote_endpoints": [
      "::ffff:127.0.0.1:47846"
    ]
  }
}
```
