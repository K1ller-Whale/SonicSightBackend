# Load-suite run `20260811T054723Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-11T05:59:22Z
- **SonicSightBackend:** `7cb179bb4064` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:052388852cee79cb (37692 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 718.7 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | time_to_first_nonbuffering_result | p95 | <= 8.000 s | 6.415 | pass |
| NFR-PERF-002 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-003 | inter_result_interval_abs_dev_from_125ms | p95 | <= 62.500 ms | 0.958 | pass |
| NFR-PERF-004 | inference_time_ms | p95 | <= 140 ms | 60.000 | pass |
| NFR-PERF-004 | inference_time_ms | p99 | <= 250 ms | 65.000 | pass |
| NFR-PERF-004 | inference_time_ms | max | <= 1500 ms | 321 | pass |
| NFR-PERF-004 | total_server_time_ms | report | report —  | {'p95': 124.0, 'p99': 125.0, 'max': 375} | characterisation |
| NFR-PERF-005 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 2.028 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 87.518 | pass |
| NFR-PERF-008 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-PERF-015 | — | — |  —  | — | condition-mismatch |
| NFR-FUNC-001 | replay_left_right_pearson | max_over_runs | <= 0.350 correlation | — | not-measured |
| NFR-FUNC-002 | — | — |  —  | — | condition-mismatch |
| NFR-FUNC-004 | — | — |  —  | — | condition-mismatch |
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
| event_loop_lag | p99 | 2.028 |
| event_loop_lag | max | 87.518 |
| inference_time_ms | p95 | 60.000 |
| inference_time_ms | p99 | 65.000 |
| inference_time_ms | max | 321 |
| inter_result_interval_abs_dev_from_125ms | p95 | 0.958 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 6.415 |
| total_server_time_ms | p95 | 124.000 |
| total_server_time_ms | p99 | 125.000 |
| total_server_time_ms | max | 375 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 6.246967536571428,
    "min": 6.127548324000003,
    "max": 7.128539942999993,
    "p50": 6.252501376999987,
    "p90": 6.25309403,
    "p95": 6.37749826999999,
    "p99": 6.9783316083999924
  },
  "inter_result_interval_ms": {
    "count": 4708,
    "mean": 124.78241034876807,
    "min": 55.97420300000522,
    "max": 403.96419500001457,
    "p50": 125.11999699995613,
    "p90": 125.48791300004041,
    "p95": 125.66076854997448,
    "p99": 126.23893878002036
  },
  "cadence_abs_dev_ms": {
    "count": 4708,
    "mean": 0.8062172637913535,
    "min": 5.4131552360559e-05,
    "max": 278.9755328684953,
    "p50": 0.30520199999273245,
    "p90": 0.8900731314918177,
    "p95": 0.9980978815286787,
    "p99": 1.924853041472265
  },
  "total_server_time_ms": {
    "count": 4708,
    "mean": 124.91758708581139,
    "min": 95.0,
    "max": 1250.0,
    "p50": 124.0,
    "p90": 124.0,
    "p95": 124.0,
    "p99": 125.0
  },
  "audio_age_lag_ms": {
    "count": 4729,
    "mean": 3277.053558207443,
    "min": 3137.196804000013,
    "max": 4377.072279999993,
    "p50": 3251.8427340000926,
    "p90": 3376.778465999996,
    "p95": 3377.1330554000087,
    "p99": 3377.626360159984
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 680,
      "mean": 2621.376390165441,
      "min": 2367.4765625,
      "max": 2628.83203125,
      "p50": 2622.25,
      "p90": 2623.89453125,
      "p95": 2628.8046875,
      "p99": 2628.828125
    },
    "cpu_pct": {
      "count": 680,
      "mean": 42.87838235294118,
      "min": 0.0,
      "max": 84.5,
      "p50": 49.75,
      "p90": 53.6,
      "p95": 54.5,
      "p99": 58.36300000000001
    },
    "gpu_used_mib": {
      "count": 680,
      "mean": 1641.7926470588236,
      "min": 1363.0,
      "max": 2847.0,
      "p50": 1641.0,
      "p90": 1648.1,
      "p95": 1665.0,
      "p99": 1705.42
    },
    "gpu_util_pct": {
      "count": 680,
      "mean": 26.679411764705883,
      "min": 0.0,
      "max": 57.0,
      "p50": 32.0,
      "p90": 33.0,
      "p95": 33.0,
      "p99": 38.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:52182"
    ]
  }
}
```
