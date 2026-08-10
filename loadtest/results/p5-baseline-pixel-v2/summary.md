# Load-suite run `20260808T145512Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-08T15:07:20Z
- **SonicSightBackend:** `1a1842a8d159` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1650, 595.97, 4096 MiB
- **Host:** Windows-10-10.0.26200-SP0 — Python 3.11.9, torch 2.6.0+cu124, grpcio 1.78.0
- **nfr_targets.yaml:** sha256:b43885f80c4f1c08 (25497 B)
- **Checkpoints:** frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 727.1 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-003 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-004 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 28.000 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 59.000 | pass |
| NFR-PERF-008 | query_round_trip | p95 | <= 250 ms | 1182.878 | FAIL |
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
| event_loop_lag | max | 59.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 62.192 |
| query_round_trip | p95 | 1182.878 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 6.302 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 6.223656576193337,
    "min": 5.957379600033164,
    "max": 7.693672299850732,
    "p50": 6.170379700139165,
    "p90": 6.228749500121921,
    "p95": 6.269129800144583,
    "p99": 7.4087637999095035
  },
  "inter_result_interval_ms": {
    "count": 4773,
    "mean": 124.88143754451406,
    "min": 1.9574002362787724,
    "max": 2177.863399963826,
    "p50": 124.61009994149208,
    "p90": 153.5005199722946,
    "p95": 193.12848001718498,
    "p99": 298.48254788666867
  },
  "cadence_abs_dev_ms": {
    "count": 4773,
    "mean": 25.055237896231045,
    "min": 0.0008619915854524152,
    "max": 2052.874737832307,
    "p50": 12.779661972464282,
    "p90": 58.00496242606755,
    "p95": 73.76377791499934,
    "p99": 173.49388575514936
  },
  "total_server_time_ms": {
    "count": 0
  },
  "audio_age_lag_ms": {
    "count": 4794,
    "mean": 3421.1617206019523,
    "min": 3108.432499691844,
    "max": 8945.098599884659,
    "p50": 3207.000350113958,
    "p90": 3750.3914101514965,
    "p95": 4742.310370062477,
    "p99": 6357.279731789601
  },
  "query_rtt_ms": {
    "count": 419,
    "mean": 177.57952767635445,
    "min": 1.317600253969431,
    "max": 3565.5400999821723,
    "p50": 1.7995997332036495,
    "p90": 438.0894602276377,
    "p95": 1182.8777200542363,
    "p99": 3064.2322479840363
  },
  "resources": {
    "rss_mib": {
      "count": 654,
      "mean": 110.12670823776759,
      "min": 102.31640625,
      "max": 118.89453125,
      "p50": 113.89453125,
      "p90": 116.43359375,
      "p95": 117.077734375,
      "p99": 118.886484375
    },
    "cpu_pct": {
      "count": 654,
      "mean": 433.7382262996942,
      "min": 0.0,
      "max": 775.5,
      "p50": 518.7,
      "p90": 641.24,
      "p95": 667.385,
      "p99": 716.3430000000002
    },
    "gpu_used_mib": {
      "count": 654,
      "mean": 1434.454128440367,
      "min": 1378.0,
      "max": 1664.0,
      "p50": 1440.0,
      "p90": 1467.0,
      "p95": 1494.35,
      "p99": 1501.0
    },
    "gpu_util_pct": {
      "count": 654,
      "mean": 37.857798165137616,
      "min": 0.0,
      "max": 80.0,
      "p50": 43.0,
      "p90": 50.0,
      "p95": 57.0,
      "p99": 69.47000000000003
    },
    "nvidia_smi_available": true
  }
}
```
