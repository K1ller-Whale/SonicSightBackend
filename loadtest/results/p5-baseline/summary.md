# Load-suite run `20260808T142635Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-08T14:38:49Z
- **SonicSightBackend:** `1a1842a8d159` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1650, 595.97, 4096 MiB
- **Host:** Windows-10-10.0.26200-SP0 — Python 3.11.9, torch 2.6.0+cu124, grpcio 1.78.0
- **nfr_targets.yaml:** sha256:b43885f80c4f1c08 (25497 B)
- **Checkpoints:** frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 732.8 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | time_to_first_nonbuffering_result | p95 | <= 8.000 s | 6.255 | pass |
| NFR-PERF-003 | inter_result_interval_abs_dev_from_125ms | p95 | <= 62.500 ms | 31.011 | pass |
| NFR-PERF-004 | total_server_time_ms | p95 | <= 140 ms | 59.400 | pass |
| NFR-PERF-004 | total_server_time_ms | max | <= 500 ms | 256 | pass |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 13.000 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 29.000 | pass |
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
| event_loop_lag | max | 29.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 31.011 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 6.255 |
| total_server_time_ms | p95 | 59.400 |
| total_server_time_ms | max | 256 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 6.156333333346993,
    "min": 5.938000000081956,
    "max": 6.359999999869615,
    "p50": 6.17200000025332,
    "p90": 6.25,
    "p95": 6.25,
    "p99": 6.3379999998956915
  },
  "inter_result_interval_ms": {
    "count": 4773,
    "mean": 126.39094908842137,
    "min": 0.0,
    "max": 12125.0,
    "p50": 125.0,
    "p90": 140.99999982863665,
    "p95": 141.00000029429793,
    "p99": 176.20000015943862
  },
  "cadence_abs_dev_ms": {
    "count": 4773,
    "mean": 12.774213227096475,
    "min": 0.011337868480723046,
    "max": 12000.01133786848,
    "p50": 0.011337868480723046,
    "p90": 30.988662090541084,
    "p95": 46.98866191917773,
    "p99": 78.99501124846151
  },
  "total_server_time_ms": {
    "count": 4773,
    "mean": 55.35784621831134,
    "min": 42.0,
    "max": 12029.0,
    "p50": 49.0,
    "p90": 55.0,
    "p95": 62.0,
    "p99": 128.5599999999995
  },
  "audio_age_lag_ms": {
    "count": 4794,
    "mean": 3236.191489323169,
    "min": 3139.999999664724,
    "max": 15375.0,
    "p50": 3186.9999999180436,
    "p90": 3311.9999999180436,
    "p95": 3313.0000000819564,
    "p99": 3375.0
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 678,
      "mean": 621.1547520280236,
      "min": 464.90234375,
      "max": 822.18359375,
      "p50": 638.458984375,
      "p90": 736.4296875,
      "p95": 741.399609375,
      "p99": 821.973203125
    },
    "cpu_pct": {
      "count": 678,
      "mean": 414.1656342182891,
      "min": 0.0,
      "max": 552.2,
      "p50": 510.05,
      "p90": 536.5,
      "p95": 540.0,
      "p99": 545.938
    },
    "gpu_used_mib": {
      "count": 678,
      "mean": 1619.924778761062,
      "min": 1300.0,
      "max": 3558.0,
      "p50": 1598.0,
      "p90": 1676.3000000000002,
      "p95": 1682.0,
      "p99": 1693.0
    },
    "gpu_util_pct": {
      "count": 678,
      "mean": 27.72566371681416,
      "min": 0.0,
      "max": 70.0,
      "p50": 32.0,
      "p90": 34.0,
      "p95": 35.0,
      "p99": 46.0
    },
    "nvidia_smi_available": true
  }
}
```
