# Load-suite run `20260810T121152Z-load`

- **Scenario:** load
- **Timestamp (UTC):** 2026-08-10T12:23:52Z
- **SonicSightBackend:** `f840ce16afdd` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:c1b14b0c73772c80 (36367 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 719.9 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-007 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-010 | gpu_memory_used | max_over_steady_state | <= 5632 MiB | 1955.000 | pass |
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
| event_loop_lag | p99 | 2.109 |
| event_loop_lag | max | 5.092 |
| gpu_memory_used | max_over_steady_state | 1955.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 129.482 |
| non_lan_endpoints_contacted | count | 0 |
| sequence_gap_rate | proportion | 0.000 |
| sessions_meeting_perf_015_and_005 | count | 0 |
| time_to_first_nonbuffering_result | p95 | 2.480 |
| total_server_time_ms | p95 | 156.000 |
| total_server_time_ms | p99 | 161.000 |
| total_server_time_ms | max | 348 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 1,
    "mean": 2.479852126998594,
    "min": 2.479852126998594,
    "max": 2.479852126998594,
    "p50": 2.479852126998594,
    "p90": 2.479852126998594,
    "p95": 2.479852126998594,
    "p99": 2.479852126998594
  },
  "inter_result_interval_ms": {
    "count": 2153,
    "mean": 333.21108824524026,
    "min": 106.68981700291624,
    "max": 392.8205970005365,
    "p50": 373.6221680010203,
    "p90": 378.0665008001961,
    "p95": 379.4482995988801,
    "p99": 384.2531347979093
  },
  "cadence_abs_dev_ms": {
    "count": 2153,
    "mean": 84.48169164832146,
    "min": 2.773636651909328e-05,
    "max": 143.2875072601223,
    "p50": 123.64572473639555,
    "p90": 128.09891613765785,
    "p95": 129.48182553487413,
    "p99": 134.44128397602753
  },
  "total_server_time_ms": {
    "count": 2153,
    "mean": 150.56479331165815,
    "min": 146.0,
    "max": 348.0,
    "p50": 150.0,
    "p90": 154.0,
    "p95": 156.0,
    "p99": 161.0
  },
  "audio_age_lag_ms": {
    "count": 2154,
    "mean": 1298.2500942172105,
    "min": 1273.04423300302,
    "max": 1506.8202490001568,
    "p50": 1278.0829985003948,
    "p90": 1401.803840298453,
    "p95": 1403.0385225494683,
    "p99": 1405.0626838300013
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 686,
      "mean": 2724.726260705175,
      "min": 2678.48828125,
      "max": 2743.28125,
      "p50": 2714.609375,
      "p90": 2743.2421875,
      "p95": 2743.2578125,
      "p99": 2743.27734375
    },
    "cpu_pct": {
      "count": 686,
      "mean": 41.28746355685131,
      "min": 0.0,
      "max": 47.7,
      "p50": 41.1,
      "p90": 44.9,
      "p95": 45.7,
      "p99": 46.714999999999996
    },
    "gpu_used_mib": {
      "count": 686,
      "mean": 1889.628279883382,
      "min": 1885.0,
      "max": 1955.0,
      "p50": 1889.0,
      "p90": 1891.0,
      "p95": 1891.0,
      "p99": 1896.0
    },
    "gpu_util_pct": {
      "count": 686,
      "mean": 38.673469387755105,
      "min": 0.0,
      "max": 59.0,
      "p50": 39.0,
      "p90": 40.0,
      "p95": 41.0,
      "p99": 41.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:33682"
    ]
  }
}
```
