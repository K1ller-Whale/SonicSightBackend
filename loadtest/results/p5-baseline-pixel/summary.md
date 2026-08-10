# Load-suite run `20260808T144007Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-08T14:52:11Z
- **SonicSightBackend:** `1a1842a8d159` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1650, 595.97, 4096 MiB
- **Host:** Windows-10-10.0.26200-SP0 — Python 3.11.9, torch 2.6.0+cu124, grpcio 1.78.0
- **nfr_targets.yaml:** sha256:b43885f80c4f1c08 (25497 B)
- **Checkpoints:** frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 722.9 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-003 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-004 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 28.000 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 153.000 | pass |
| NFR-PERF-008 | query_round_trip | p95 | <= 250 ms | 944.800 | FAIL |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-FUNC-001 | — | — |  —  | — | condition-mismatch |
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
| event_loop_lag | p99 | 28.000 |
| event_loop_lag | max | 153.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 78.011 |
| query_round_trip | p95 | 944.800 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 6.291 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 6.1799047618572205,
    "min": 6.030999999959022,
    "max": 6.483999999705702,
    "p50": 6.186999999918044,
    "p90": 6.25,
    "p95": 6.280999999959022,
    "p99": 6.443399999756366
  },
  "inter_result_interval_ms": {
    "count": 4773,
    "mean": 124.1751518960993,
    "min": 0.0,
    "max": 1250.0,
    "p50": 125.0,
    "p90": 171.99999978765845,
    "p95": 203.00000021234155,
    "p99": 282.0000001229346
  },
  "cadence_abs_dev_ms": {
    "count": 4773,
    "mean": 27.34040923148288,
    "min": 0.011337868480723046,
    "max": 1125.0113378684807,
    "p50": 15.98866242581721,
    "p90": 62.98866221347566,
    "p95": 78.01133808082227,
    "p99": 157.0113379914153
  },
  "total_server_time_ms": {
    "count": 0
  },
  "audio_age_lag_ms": {
    "count": 4794,
    "mean": 3364.1906550238987,
    "min": 3141.000000294298,
    "max": 6157.000000122935,
    "p50": 3203.999999910593,
    "p90": 3729.500000132245,
    "p95": 4141.000000294298,
    "p99": 4844.000000040978
  },
  "query_rtt_ms": {
    "count": 419,
    "mean": 138.59665872493585,
    "min": 0.0,
    "max": 2750.0,
    "p50": 0.0,
    "p90": 494.1999998874957,
    "p95": 944.7999999392752,
    "p99": 1778.239999888463
  },
  "resources": {
    "rss_mib": {
      "count": 649,
      "mean": 255.9426280816641,
      "min": 106.5,
      "max": 467.140625,
      "p50": 139.17578125,
      "p90": 457.234375,
      "p95": 457.234375,
      "p99": 465.88109375
    },
    "cpu_pct": {
      "count": 649,
      "mean": 421.02080123266563,
      "min": 0.0,
      "max": 762.7,
      "p50": 479.7,
      "p90": 643.9399999999999,
      "p95": 673.42,
      "p99": 715.6639999999999
    },
    "gpu_used_mib": {
      "count": 649,
      "mean": 1458.7981510015409,
      "min": 1322.0,
      "max": 1708.0,
      "p50": 1395.0,
      "p90": 1631.0,
      "p95": 1639.0,
      "p99": 1646.0
    },
    "gpu_util_pct": {
      "count": 649,
      "mean": 38.23112480739599,
      "min": 0.0,
      "max": 80.0,
      "p50": 43.0,
      "p90": 51.0,
      "p95": 59.0,
      "p99": 73.03999999999996
    },
    "nvidia_smi_available": true
  }
}
```
