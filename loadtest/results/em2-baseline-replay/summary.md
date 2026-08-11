# Load-suite run `20260810T141553Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-10T14:20:58Z
- **SonicSightBackend:** `f840ce16afdd` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:c1b14b0c73772c80 (36367 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 304.4 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | time_to_first_nonbuffering_result | p95 | <= 8.000 s | 6.183 | pass |
| NFR-PERF-002 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-003 | inter_result_interval_abs_dev_from_125ms | p95 | <= 62.500 ms | 66.852 | FAIL |
| NFR-PERF-004 | total_server_time_ms | p95 | <= 140 ms | 52.000 | pass |
| NFR-PERF-004 | total_server_time_ms | p99 | <= 250 ms | 60.360 | pass |
| NFR-PERF-004 | total_server_time_ms | max | <= 1500 ms | 239 | pass |
| NFR-PERF-005 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 1.955 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 438.795 | pass |
| NFR-PERF-008 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | — | not-measured |
| NFR-PERF-015 | — | — |  —  | — | condition-mismatch |
| NFR-FUNC-001 | replay_left_right_pearson | max_over_runs | <= 0.350 correlation | 0.187 | pass |
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
| event_loop_lag | p99 | 1.955 |
| event_loop_lag | max | 438.795 |
| inter_result_interval_abs_dev_from_125ms | p95 | 66.852 |
| replay_left_right_pearson | max_over_runs | 0.187 |
| time_to_first_nonbuffering_result | p95 | 6.183 |
| total_server_time_ms | p95 | 52.000 |
| total_server_time_ms | p99 | 60.360 |
| total_server_time_ms | max | 239 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 3,
    "mean": 6.103595683998719,
    "min": 6.032543613000598,
    "max": 6.191008752997732,
    "p50": 6.087234685997828,
    "p90": 6.170253939597751,
    "p95": 6.180631346297742,
    "p99": 6.188933271657734
  },
  "inter_result_interval_ms": {
    "count": 499,
    "mean": 124.92986164730547,
    "min": 0.5135229985171463,
    "max": 838.0070400016848,
    "p50": 124.88203600150882,
    "p90": 128.58259260246996,
    "p95": 133.8155845001893,
    "p99": 345.2837683208053
  },
  "cadence_abs_dev_ms": {
    "count": 499,
    "mean": 13.118712876307173,
    "min": 0.0015091334455519245,
    "max": 713.0183778701655,
    "p50": 1.3724391299285799,
    "p90": 30.593453331103976,
    "p95": 76.74074293151646,
    "p99": 220.29510618928597
  },
  "total_server_time_ms": {
    "count": 499,
    "mean": 48.3186372745491,
    "min": 28.0,
    "max": 743.0,
    "p50": 46.0,
    "p90": 50.0,
    "p95": 52.0,
    "p99": 210.25999999999976
  },
  "audio_age_lag_ms": {
    "count": 502,
    "mean": 3186.0993487888013,
    "min": 3055.0521929981187,
    "max": 4043.6186509978143,
    "p50": 3173.5641510003916,
    "p90": 3181.2143981027475,
    "p95": 3272.879775797992,
    "p99": 3497.744895331853
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 250,
      "mean": 2632.815703125,
      "min": 2388.2421875,
      "max": 2639.62890625,
      "p50": 2636.0,
      "p90": 2636.15625,
      "p95": 2638.43203125,
      "p99": 2639.6269921875
    },
    "cpu_pct": {
      "count": 250,
      "mean": 51.3024,
      "min": 0.0,
      "max": 97.3,
      "p50": 56.9,
      "p90": 62.01,
      "p95": 63.455,
      "p99": 85.23399999999997
    },
    "gpu_used_mib": {
      "count": 250,
      "mean": 2337.436,
      "min": 2100.0,
      "max": 3588.0,
      "p50": 2335.0,
      "p90": 2354.0,
      "p95": 2362.0,
      "p99": 2362.0
    },
    "gpu_util_pct": {
      "count": 250,
      "mean": 27.98,
      "min": 0.0,
      "max": 49.0,
      "p50": 32.0,
      "p90": 33.0,
      "p95": 33.0,
      "p99": 46.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:40468",
      "::ffff:127.0.0.1:48714",
      "::ffff:127.0.0.1:49010",
      "::ffff:127.0.0.1:50674"
    ]
  }
}
```
