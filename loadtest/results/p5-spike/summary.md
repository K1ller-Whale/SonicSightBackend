# Load-suite run `20260808T150850Z-spike`

- **Scenario:** spike
- **Timestamp (UTC):** 2026-08-08T15:15:19Z
- **SonicSightBackend:** `1a1842a8d159` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1650, 595.97, 4096 MiB
- **Host:** Windows-10-10.0.26200-SP0 — Python 3.11.9, torch 2.6.0+cu124, grpcio 1.78.0
- **nfr_targets.yaml:** sha256:b43885f80c4f1c08 (25497 B)
- **Checkpoints:** frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 386.9 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-014 | sessions_surviving_burst | count | >= 3 sessions | 4 | pass |
| NFR-PERF-014 | time_to_first_compliant_10s_window | max_over_survivors | <= 30 s | inf | FAIL |
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
| event_loop_lag | p99 | 28.000 |
| event_loop_lag | max | 44.000 |
| sessions_surviving_burst | count | 4 |
| time_to_first_compliant_10s_window | max_over_survivors | inf |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 4,
    "mean": 6.320871125091799,
    "min": 6.205541600007564,
    "max": 6.420988900121301,
    "p50": 6.328477000119165,
    "p90": 6.402004870120436,
    "p95": 6.411496885120869,
    "p99": 6.419090497121215
  },
  "inter_result_interval_ms": {
    "count": 3652,
    "mean": 415.6873301477432,
    "min": 3.6522001028060913,
    "max": 2527.104000095278,
    "p50": 361.7234998382628,
    "p90": 627.638559928164,
    "p95": 728.8849601289255,
    "p99": 1038.5286950506254
  },
  "cadence_abs_dev_ms": {
    "count": 3652,
    "mean": 290.9630503540192,
    "min": 95.24773754513149,
    "max": 2402.115337963759,
    "p50": 236.73483770674352,
    "p90": 502.64989779664467,
    "p95": 603.8962979974062,
    "p99": 913.5400329191061
  },
  "total_server_time_ms": {
    "count": 3652,
    "mean": 410.0739320920044,
    "min": 216.0,
    "max": 2521.0,
    "p50": 356.0,
    "p90": 621.8000000000002,
    "p95": 720.4499999999998,
    "p99": 1025.449999999999
  },
  "audio_age_lag_ms": {
    "count": 3656,
    "mean": 132129.33815166945,
    "min": 3450.5741000175476,
    "max": 258030.97749967128,
    "p50": 125919.72725000232,
    "p90": 239554.6723001171,
    "p95": 253120.80824992154,
    "p99": 256803.16320497077
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 341,
      "mean": 166.63929618768327,
      "min": 103.51171875,
      "max": 173.6796875,
      "p50": 167.4140625,
      "p90": 169.23828125,
      "p95": 170.7578125,
      "p99": 170.9921875
    },
    "cpu_pct": {
      "count": 341,
      "mean": 492.8961876832844,
      "min": 0.0,
      "max": 680.6,
      "p50": 509.6,
      "p90": 620.3,
      "p95": 638.2,
      "p99": 667.2400000000001
    },
    "gpu_used_mib": {
      "count": 341,
      "mean": 1527.8387096774193,
      "min": 1508.0,
      "max": 1694.0,
      "p50": 1527.0,
      "p90": 1544.0,
      "p95": 1553.0,
      "p99": 1572.8000000000002
    },
    "gpu_util_pct": {
      "count": 341,
      "mean": 39.57478005865103,
      "min": 1.0,
      "max": 64.0,
      "p50": 41.0,
      "p90": 54.0,
      "p95": 57.0,
      "p99": 61.60000000000002
    },
    "nvidia_smi_available": true
  }
}
```
