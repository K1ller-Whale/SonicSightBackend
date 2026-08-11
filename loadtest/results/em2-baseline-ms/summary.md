# Load-suite run `20260810T115950Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-10T12:11:48Z
- **SonicSightBackend:** `f840ce16afdd` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:c1b14b0c73772c80 (36367 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 717.6 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-002 | time_to_first_nonbuffering_result | p95 | <= 3.500 s | 2.732 | pass |
| NFR-PERF-003 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-004 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-005 | total_server_time_ms | p95 | <= 200 ms | 153.000 | pass |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 2.084 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 5.233 | pass |
| NFR-PERF-008 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-PERF-015 | inter_result_interval_abs_dev_from_125ms | p95 | <= 125.000 ms | 128.726 | FAIL |
| NFR-FUNC-001 | — | — |  —  | — | condition-mismatch |
| NFR-FUNC-002 | cross_check_margin | value | >= 0.300 correlation-margin | — | not-measured |
| NFR-FUNC-004 | results_meeting_multisensory_contract | proportion | >= 1.000 ratio | 1.000 | pass |
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
| event_loop_lag | p99 | 2.084 |
| event_loop_lag | max | 5.233 |
| inter_result_interval_abs_dev_from_125ms | p95 | 128.726 |
| results_meeting_multisensory_contract | proportion | 1.000 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 2.732 |
| total_server_time_ms | p95 | 153.000 |
| total_server_time_ms | p99 | 158.470 |
| total_server_time_ms | max | 350 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 2.658656838047353,
    "min": 2.397617844002525,
    "max": 6.664885017002234,
    "p50": 2.4700843379978323,
    "p90": 2.5091145709993725,
    "p95": 2.525318141000753,
    "p99": 5.836971641801941
  },
  "inter_result_interval_ms": {
    "count": 2034,
    "mean": 325.3722339798451,
    "min": 3.3251669992750976,
    "max": 390.05030499902205,
    "p50": 373.2208689998515,
    "p90": 377.87299109950254,
    "p95": 378.7750629004222,
    "p99": 382.24597447831917
  },
  "cadence_abs_dev_ms": {
    "count": 2034,
    "mean": 82.74241591926686,
    "min": 0.0027782621532423946,
    "max": 246.65215726376346,
    "p50": 123.3748822382563,
    "p90": 128.015947934598,
    "p95": 129.29769878524667,
    "p99": 143.55941251877528
  },
  "total_server_time_ms": {
    "count": 2034,
    "mean": 153.26352015732547,
    "min": 136.0,
    "max": 4532.0,
    "p50": 149.0,
    "p90": 152.0,
    "p95": 154.0,
    "p99": 233.66000000000713
  },
  "audio_age_lag_ms": {
    "count": 2055,
    "mean": 1307.5084257260264,
    "min": 1263.6382279997633,
    "max": 5663.611071999185,
    "p50": 1276.8146599992178,
    "p90": 1280.4195190001337,
    "p95": 1283.8116811999498,
    "p99": 2040.2299920402615
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 684,
      "mean": 2694.2499714455407,
      "min": 2368.125,
      "max": 2714.6875,
      "p50": 2708.55859375,
      "p90": 2709.35546875,
      "p95": 2709.37890625,
      "p99": 2714.6796875
    },
    "cpu_pct": {
      "count": 684,
      "mean": 39.75043859649123,
      "min": 0.0,
      "max": 124.3,
      "p50": 40.6,
      "p90": 44.8,
      "p95": 45.7,
      "p99": 79.79099999999991
    },
    "gpu_used_mib": {
      "count": 684,
      "mean": 1875.830409356725,
      "min": 1339.0,
      "max": 1940.0,
      "p50": 1876.0,
      "p90": 1885.0,
      "p95": 1885.0,
      "p99": 1886.0
    },
    "gpu_util_pct": {
      "count": 684,
      "mean": 37.078947368421055,
      "min": 0.0,
      "max": 83.0,
      "p50": 38.0,
      "p90": 39.0,
      "p95": 40.0,
      "p99": 54.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:47190"
    ]
  }
}
```
