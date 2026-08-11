# Load-suite run `20260810T165416Z-load`

- **Scenario:** load
- **Timestamp (UTC):** 2026-08-10T17:06:17Z
- **SonicSightBackend:** `7cb179bb4064` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:052388852cee79cb (37692 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 720.1 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-007 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-010 | gpu_memory_used | max_over_steady_state | <= 5632 MiB | 1662.000 | pass |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-PERF-016 | sessions_meeting_perf_015_and_005 | count | >= 1 sessions | 0 | FAIL |
| NFR-SEC-001 | non_lan_endpoints_contacted | count | == 0 endpoints | 0 | pass |
| NFR-PERF-009 | — | — | — | — | skipped: environment E-M not in ['E-D'] |
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
| event_loop_lag | p99 | 2.131 |
| event_loop_lag | max | 5.073 |
| gpu_memory_used | max_over_steady_state | 1662.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 1.460 |
| non_lan_endpoints_contacted | count | 0 |
| sequence_gap_rate | proportion | 0.000 |
| sessions_meeting_perf_015_and_005 | count | 0 |
| time_to_first_nonbuffering_result | p95 | 2.628 |
| total_server_time_ms | p95 | 249.000 |
| total_server_time_ms | p99 | 250.000 |
| total_server_time_ms | max | 497 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 1,
    "mean": 2.628270220000104,
    "min": 2.628270220000104,
    "max": 2.628270220000104,
    "p50": 2.628270220000104,
    "p90": 2.628270220000104,
    "p95": 2.628270220000104,
    "p99": 2.628270220000104
  },
  "inter_result_interval_ms": {
    "count": 2870,
    "mean": 249.96783738153303,
    "min": 157.03850099998817,
    "max": 256.67871499990724,
    "p50": 249.90652449992012,
    "p90": 250.86747280001873,
    "p95": 251.09515759975238,
    "p99": 251.82684130983034
  },
  "cadence_abs_dev_ms": {
    "count": 2870,
    "mean": 0.5778519464149723,
    "min": 8.262798019131878e-06,
    "max": 92.93882326305038,
    "p50": 0.4200077368818711,
    "p90": 1.1481128628861366,
    "p95": 1.4596216129507396,
    "p99": 2.157766889994354
  },
  "total_server_time_ms": {
    "count": 2870,
    "mean": 248.2923344947735,
    "min": 242.0,
    "max": 497.0,
    "p50": 248.0,
    "p90": 249.0,
    "p95": 249.0,
    "p99": 250.0
  },
  "audio_age_lag_ms": {
    "count": 2871,
    "mean": 1408.0245861386293,
    "min": 1375.2833029998328,
    "max": 1659.573607999846,
    "p50": 1377.622796000196,
    "p90": 1502.3637429999326,
    "p95": 1502.746949999846,
    "p99": 1503.4264561999862
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 683,
      "mean": 2954.9561733620058,
      "min": 2796.23046875,
      "max": 3040.17578125,
      "p50": 2955.4609375,
      "p90": 3038.5625,
      "p95": 3039.8046875,
      "p99": 3039.8203125
    },
    "cpu_pct": {
      "count": 683,
      "mean": 56.40775988286969,
      "min": 0.0,
      "max": 71.4,
      "p50": 56.3,
      "p90": 60.1,
      "p95": 60.9,
      "p99": 62.13599999999999
    },
    "gpu_used_mib": {
      "count": 683,
      "mean": 1650.3016105417278,
      "min": 1644.0,
      "max": 1662.0,
      "p50": 1650.0,
      "p90": 1651.0,
      "p95": 1654.0,
      "p99": 1655.0
    },
    "gpu_util_pct": {
      "count": 683,
      "mean": 54.33089311859444,
      "min": 0.0,
      "max": 57.0,
      "p50": 55.0,
      "p90": 56.0,
      "p95": 56.0,
      "p99": 56.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:60854"
    ]
  }
}
```
