# Load-suite run `20260808T151646Z-load`

- **Scenario:** load
- **Timestamp (UTC):** 2026-08-08T15:32:39Z
- **SonicSightBackend:** `1a1842a8d159` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1650, 595.97, 4096 MiB
- **Host:** Windows-10-10.0.26200-SP0 — Python 3.11.9, torch 2.6.0+cu124, grpcio 1.78.0
- **nfr_targets.yaml:** sha256:b43885f80c4f1c08 (25497 B)
- **Checkpoints:** frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 952.2 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-007 | sessions_meeting_perf_003_and_004 | count | >= 2 sessions | 0 | FAIL |
| NFR-PERF-007 | concurrency_ceiling | report | report —  | — | characterisation |
| NFR-PERF-009 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-SEC-001 | non_lan_endpoints_contacted | count | == 0 endpoints | — | not-measured |
| NFR-PERF-010 | — | — | — | — | skipped: e-m-conditional: requires an E-M-class host (TensorFlow) |
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
| event_loop_lag | max | 121.000 |
| gpu_memory_used | max_over_steady_state | 1577.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 191.628 |
| sequence_gap_rate | proportion | 0.000 |
| sessions_meeting_perf_003_and_004 | count | 0 |
| time_to_first_nonbuffering_result | p95 | 6.127 |
| total_server_time_ms | p95 | 310.750 |
| total_server_time_ms | max | 4475 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 2,
    "mean": 6.059804100077599,
    "min": 5.985692299902439,
    "max": 6.1339159002527595,
    "p50": 6.059804100077599,
    "p90": 6.119093540217728,
    "p95": 6.126504720235244,
    "p99": 6.132433664249256
  },
  "inter_result_interval_ms": {
    "count": 11426,
    "mean": 165.41645724660395,
    "min": 4.148799926042557,
    "max": 4533.087499905378,
    "p50": 131.75430009141564,
    "p90": 256.27580005675554,
    "p95": 316.6169250616804,
    "p99": 482.4462999822572
  },
  "cadence_abs_dev_ms": {
    "count": 11426,
    "mean": 47.10202940575615,
    "min": 0.0014617633235047833,
    "max": 4408.098837773859,
    "p50": 12.952762009964545,
    "p90": 131.28713792523627,
    "p95": 191.6282629301611,
    "p99": 357.4576378507379
  },
  "total_server_time_ms": {
    "count": 11426,
    "mean": 159.12077717486434,
    "min": 44.0,
    "max": 4475.0,
    "p50": 126.0,
    "p90": 249.0,
    "p95": 310.75,
    "p99": 473.0
  },
  "audio_age_lag_ms": {
    "count": 11428,
    "mean": 77153.8647201003,
    "min": 3145.241599995643,
    "max": 123454.29139956832,
    "p50": 87922.83700010739,
    "p90": 111666.13837997429,
    "p95": 115267.06316026393,
    "p99": 121592.73857797495
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 840,
      "mean": 138.64840029761905,
      "min": 114.6796875,
      "max": 139.828125,
      "p50": 138.8046875,
      "p90": 139.3828125,
      "p95": 139.4453125,
      "p99": 139.60546875
    },
    "cpu_pct": {
      "count": 840,
      "mean": 522.6882142857143,
      "min": 0.0,
      "max": 879.8,
      "p50": 549.9,
      "p90": 657.0,
      "p95": 682.0699999999998,
      "p99": 725.3230000000001
    },
    "gpu_used_mib": {
      "count": 840,
      "mean": 1543.3738095238095,
      "min": 1525.0,
      "max": 1577.0,
      "p50": 1541.0,
      "p90": 1548.0,
      "p95": 1564.0,
      "p99": 1565.0
    },
    "gpu_util_pct": {
      "count": 840,
      "mean": 48.81785714285714,
      "min": 0.0,
      "max": 72.0,
      "p50": 52.0,
      "p90": 65.0,
      "p95": 67.0,
      "p99": 70.0
    },
    "nvidia_smi_available": true
  }
}
```
