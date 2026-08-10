# Load-suite run `20260810T072653Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-10T07:38:52Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:98f7877bbe5216fc (17587 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 718.5 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | time_to_first_nonbuffering_result | p95 | <= 8.000 s | 6.220 | pass |
| NFR-PERF-002 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-003 | inter_result_interval_abs_dev_from_125ms | p95 | <= 62.500 ms | 3.229 | pass |
| NFR-PERF-004 | total_server_time_ms | p95 | <= 140 ms | 50.000 | pass |
| NFR-PERF-004 | total_server_time_ms | max | <= 500 ms | 191 | pass |
| NFR-PERF-005 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 2.026 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 2.474 | pass |
| NFR-PERF-008 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-FUNC-001 | replay_left_right_pearson | max_over_runs | <= 0.350 correlation | — | not-measured |
| NFR-FUNC-002 | — | — |  —  | — | condition-mismatch |
| NFR-FUNC-003 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-SEC-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-001 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-003 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-INT-004 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-MAINT-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-MAINT-003 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-FLEX-002 | — | — |  —  | — | not-evaluable-by-this-suite |
| NFR-FLEX-003 | — | — |  —  | — | not-evaluable-by-this-suite |

## Measured metrics

| Metric | Statistic | Value |
|---|---|---|
| event_loop_lag | p99 | 2.026 |
| event_loop_lag | max | 2.474 |
| inter_result_interval_abs_dev_from_125ms | p95 | 3.229 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 6.220 |
| total_server_time_ms | p95 | 50.000 |
| total_server_time_ms | p99 | 52.000 |
| total_server_time_ms | max | 191 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 6.123901552190459,
    "min": 6.02584675900016,
    "max": 6.232917837000059,
    "p50": 6.121867554000346,
    "p90": 6.1882805590003045,
    "p95": 6.219737141999758,
    "p99": 6.230281697999999
  },
  "inter_result_interval_ms": {
    "count": 4773,
    "mean": 123.58209330358251,
    "min": 1.5038999999887892,
    "max": 155.5696999994325,
    "p50": 124.96011199982604,
    "p90": 127.38188540006377,
    "p95": 128.48801760010247,
    "p99": 131.4878652002517
  },
  "cadence_abs_dev_ms": {
    "count": 4773,
    "mean": 2.9059271212190745,
    "min": 0.000563131626449831,
    "max": 123.48476213153049,
    "p50": 0.9598621316138178,
    "p90": 3.5451319209980756,
    "p95": 5.015744026745092,
    "p99": 88.01337017111209
  },
  "total_server_time_ms": {
    "count": 4773,
    "mean": 46.377330819191286,
    "min": 28.0,
    "max": 354.0,
    "p50": 46.0,
    "p90": 49.0,
    "p95": 50.0,
    "p99": 53.0
  },
  "audio_age_lag_ms": {
    "count": 4794,
    "mean": 3197.7698269770463,
    "min": 3154.361687999881,
    "max": 3481.7739860000074,
    "p50": 3173.85505700031,
    "p90": 3298.4852424003293,
    "p95": 3299.4184794002194,
    "p99": 3301.8914607697207
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 686,
      "mean": 2604.161243850219,
      "min": 2359.40234375,
      "max": 2610.28515625,
      "p50": 2605.46875,
      "p90": 2606.52734375,
      "p95": 2607.9326171875,
      "p99": 2609.015625
    },
    "cpu_pct": {
      "count": 686,
      "mean": 47.55991253644315,
      "min": 0.0,
      "max": 80.2,
      "p50": 55.3,
      "p90": 60.3,
      "p95": 62.075,
      "p99": 66.53
    },
    "gpu_used_mib": {
      "count": 686,
      "mean": 1496.0014577259476,
      "min": 894.0,
      "max": 1524.0,
      "p50": 1489.0,
      "p90": 1520.0,
      "p95": 1520.0,
      "p99": 1520.0
    },
    "gpu_util_pct": {
      "count": 686,
      "mean": 27.32215743440233,
      "min": 0.0,
      "max": 52.0,
      "p50": 32.0,
      "p90": 33.0,
      "p95": 33.0,
      "p99": 42.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:58030"
    ]
  }
}
```
