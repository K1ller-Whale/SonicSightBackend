# Load-suite run `20260809T113939Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-09T11:51:38Z
- **SonicSightBackend:** `1a1842a8d159` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1650, 595.97, 4096 MiB
- **Host:** Windows-10-10.0.26200-SP0 — Python 3.11.9, torch 2.6.0+cu124, grpcio 1.78.0
- **nfr_targets.yaml:** sha256:afe89a851eb00392 (25614 B)
- **Checkpoints:** frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 718.9 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | time_to_first_nonbuffering_result | p95 | <= 8.000 s | 6.338 | pass |
| NFR-PERF-003 | inter_result_interval_abs_dev_from_125ms | p95 | <= 62.500 ms | 18.013 | pass |
| NFR-PERF-004 | total_server_time_ms | p95 | <= 140 ms | 50.000 | pass |
| NFR-PERF-004 | total_server_time_ms | max | <= 500 ms | 328 | pass |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 13.000 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 44.000 | pass |
| NFR-PERF-008 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-FUNC-001 | replay_left_right_pearson | max_over_runs | <= 0.350 correlation | — | not-measured |
| NFR-PERF-002 | — | — | — | — | skipped: e-m-conditional: requires an E-M-class host (TensorFlow) |
| NFR-PERF-005 | — | — | — | — | skipped: e-m-conditional: requires an E-M-class host (TensorFlow) |
| NFR-FUNC-002 | — | — | — | — | skipped: e-m-conditional: requires an E-M-class host (TensorFlow) |
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
| event_loop_lag | p99 | 13.000 |
| event_loop_lag | max | 44.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 18.013 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 6.338 |
| total_server_time_ms | p95 | 50.000 |
| total_server_time_ms | p99 | 58.000 |
| total_server_time_ms | max | 328 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 6.247298985715523,
    "min": 5.943515699997079,
    "max": 8.103821200013044,
    "p50": 6.169702099999995,
    "p90": 6.244002100007492,
    "p95": 6.245440500002587,
    "p99": 7.7321450600109545
  },
  "inter_result_interval_ms": {
    "count": 4773,
    "mean": 123.09422191493445,
    "min": 1.7003999964799732,
    "max": 232.46060000383295,
    "p50": 125.69960000109859,
    "p90": 137.69337999692652,
    "p95": 141.75858000526205,
    "p99": 149.08522399724455
  },
  "cadence_abs_dev_ms": {
    "count": 4773,
    "mean": 8.061429521666668,
    "min": 0.0007378605266694649,
    "max": 123.2882621350393,
    "p50": 2.467862135537189,
    "p90": 17.288597865107256,
    "p95": 20.752717874555596,
    "p99": 78.39261412927148
  },
  "total_server_time_ms": {
    "count": 4773,
    "mean": 48.94887911166981,
    "min": 40.0,
    "max": 2217.0,
    "p50": 47.0,
    "p90": 50.0,
    "p95": 51.0,
    "p99": 67.55999999999949
  },
  "audio_age_lag_ms": {
    "count": 4794,
    "mean": 3206.4841251358357,
    "min": 3100.602299993625,
    "max": 5349.938100000145,
    "p50": 3177.2587499945075,
    "p90": 3298.375500002294,
    "p95": 3307.103240005381,
    "p99": 3357.442011001839
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 674,
      "mean": 1016.2176894009644,
      "min": 680.53125,
      "max": 1074.0703125,
      "p50": 1042.6328125,
      "p90": 1073.1640625,
      "p95": 1073.721484375,
      "p99": 1073.95703125
    },
    "cpu_pct": {
      "count": 674,
      "mean": 439.0344213649852,
      "min": 0.0,
      "max": 579.1,
      "p50": 529.0,
      "p90": 538.5,
      "p95": 540.9,
      "p99": 546.5079999999999
    },
    "gpu_used_mib": {
      "count": 674,
      "mean": 1321.632047477745,
      "min": 1090.0,
      "max": 2584.0,
      "p50": 1316.0,
      "p90": 1347.0,
      "p95": 1354.0,
      "p99": 1388.0
    },
    "gpu_util_pct": {
      "count": 674,
      "mean": 27.90652818991098,
      "min": 1.0,
      "max": 70.0,
      "p50": 32.0,
      "p90": 33.700000000000045,
      "p95": 35.0,
      "p99": 47.26999999999998
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "127.0.0.1:52677",
      "127.0.0.1:52700",
      "127.0.0.1:56545",
      "127.0.0.1:56546",
      "127.0.0.1:56547",
      "127.0.0.1:56548",
      "127.0.0.1:56766",
      "127.0.0.1:56790",
      "127.0.0.1:58558",
      "127.0.0.1:60811",
      "127.0.0.1:62240",
      "127.0.0.1:62715",
      "127.0.0.1:62746",
      "127.0.0.1:63781"
    ]
  }
}
```
