# Load-suite run `20260810T103536Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-10T10:38:23Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:5954cce68a9e5b9a (20139 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 166.5 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | time_to_first_nonbuffering_result | p95 | <= 8.000 s | 6.125 | pass |
| NFR-PERF-002 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-003 | inter_result_interval_abs_dev_from_125ms | p95 | <= 62.500 ms | 9.281 | pass |
| NFR-PERF-004 | total_server_time_ms | p95 | <= 140 ms | 62.000 | pass |
| NFR-PERF-004 | total_server_time_ms | max | <= 500 ms | 289 | pass |
| NFR-PERF-005 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 2.034 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 2.414 | pass |
| NFR-PERF-008 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | — | not-measured |
| NFR-FUNC-001 | replay_left_right_pearson | max_over_runs | <= 0.350 correlation | 0.961 | FAIL |
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
| event_loop_lag | p99 | 2.034 |
| event_loop_lag | max | 2.414 |
| inter_result_interval_abs_dev_from_125ms | p95 | 9.281 |
| replay_left_right_pearson | max_over_runs | 0.961 |
| time_to_first_nonbuffering_result | p95 | 6.125 |
| total_server_time_ms | p95 | 62.000 |
| total_server_time_ms | p99 | 66.680 |
| total_server_time_ms | max | 289 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 3,
    "mean": 6.118179507666355,
    "min": 6.059552459999395,
    "max": 6.16699106499982,
    "p50": 6.127994997999849,
    "p90": 6.159191851599826,
    "p95": 6.163091458299823,
    "p99": 6.166211143659821
  },
  "inter_result_interval_ms": {
    "count": 499,
    "mean": 123.14783804608878,
    "min": 1.6283690001728246,
    "max": 142.26313200015284,
    "p50": 125.01420499938831,
    "p90": 130.4032597996411,
    "p95": 131.7665192000277,
    "p99": 136.16289070112543
  },
  "cadence_abs_dev_ms": {
    "count": 499,
    "mean": 5.092260521702739,
    "min": 0.006253867676377922,
    "max": 123.36029313134645,
    "p50": 2.4905698685222433,
    "p90": 7.75834713219524,
    "p95": 10.89445119558422,
    "p99": 78.90851873148789
  },
  "total_server_time_ms": {
    "count": 499,
    "mean": 55.673346693386776,
    "min": 38.0,
    "max": 289.0,
    "p50": 55.0,
    "p90": 60.0,
    "p95": 62.0,
    "p99": 68.03999999999996
  },
  "audio_age_lag_ms": {
    "count": 502,
    "mean": 3183.7916521035513,
    "min": 3164.7091369995906,
    "max": 3415.73538800003,
    "p50": 3182.315877000292,
    "p90": 3187.577332899673,
    "p95": 3189.879459000076,
    "p99": 3263.0471784993647
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 157,
      "mean": 3237.9913166799365,
      "min": 3200.73828125,
      "max": 3286.41015625,
      "p50": 3230.1796875,
      "p90": 3285.75625,
      "p95": 3286.3609375,
      "p99": 3286.41015625
    },
    "cpu_pct": {
      "count": 157,
      "mean": 32.49745222929936,
      "min": 0.0,
      "max": 61.8,
      "p50": 47.4,
      "p90": 53.4,
      "p95": 54.1,
      "p99": 56.967999999999996
    },
    "gpu_used_mib": {
      "count": 157,
      "mean": 2631.522292993631,
      "min": 2621.0,
      "max": 2640.0,
      "p50": 2639.0,
      "p90": 2640.0,
      "p95": 2640.0,
      "p99": 2640.0
    },
    "gpu_util_pct": {
      "count": 157,
      "mean": 20.67515923566879,
      "min": 0.0,
      "max": 42.0,
      "p50": 30.0,
      "p90": 33.0,
      "p95": 33.0,
      "p99": 37.19999999999999
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:35888",
      "::ffff:127.0.0.1:44060",
      "::ffff:127.0.0.1:49406",
      "::ffff:127.0.0.1:59042"
    ]
  }
}
```
