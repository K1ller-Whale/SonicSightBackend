# Load-suite run `20260810T103834Z-load`

- **Scenario:** load
- **Timestamp (UTC):** 2026-08-10T10:50:35Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:5954cce68a9e5b9a (20139 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 721.1 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-007 | sessions_meeting_perf_003_and_004 | count | >= 2 sessions | 2 | pass |
| NFR-PERF-007 | concurrency_ceiling | report | report —  | — | characterisation |
| NFR-PERF-009 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-010 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-SEC-001 | non_lan_endpoints_contacted | count | == 0 endpoints | 0 | pass |
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
| event_loop_lag | p99 | 2.114 |
| event_loop_lag | max | 2.940 |
| gpu_memory_used | max_over_steady_state | 2673.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 6.864 |
| non_lan_endpoints_contacted | count | 0 |
| sequence_gap_rate | proportion | 0.000 |
| sessions_meeting_perf_003_and_004 | count | 2 |
| time_to_first_nonbuffering_result | p95 | 6.215 |
| total_server_time_ms | p95 | 81.000 |
| total_server_time_ms | p99 | 87.000 |
| total_server_time_ms | max | 350 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 2,
    "mean": 6.0982262845009245,
    "min": 5.967952438000793,
    "max": 6.228500131001056,
    "p50": 6.0982262845009245,
    "p90": 6.202445361701029,
    "p95": 6.215472746351042,
    "p99": 6.225894654071053
  },
  "inter_result_interval_ms": {
    "count": 11426,
    "mean": 124.94673812436535,
    "min": 1.8358739980612881,
    "max": 154.7442939991015,
    "p50": 124.96933049988002,
    "p90": 128.3382919991709,
    "p95": 129.96307500088733,
    "p99": 134.66655600086597
  },
  "cadence_abs_dev_ms": {
    "count": 11426,
    "mean": 2.2901541192756447,
    "min": 0.0005431318379578443,
    "max": 123.15278813345799,
    "p50": 1.5083606329164922,
    "p90": 4.969802000232448,
    "p95": 6.864269882745475,
    "p99": 13.508796380184549
  },
  "total_server_time_ms": {
    "count": 11426,
    "mean": 58.50376334675302,
    "min": 35.0,
    "max": 350.0,
    "p50": 56.0,
    "p90": 79.0,
    "p95": 81.0,
    "p99": 87.0
  },
  "audio_age_lag_ms": {
    "count": 11428,
    "mean": 3257.8351411534863,
    "min": 3161.629782000091,
    "max": 3476.8743270014966,
    "p50": 3289.9873120004486,
    "p90": 3330.2398652000193,
    "p95": 3331.6635085002417,
    "p99": 3336.2866117306476
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 682,
      "mean": 3250.5985153958945,
      "min": 3205.265625,
      "max": 3254.4296875,
      "p50": 3251.71484375,
      "p90": 3253.08984375,
      "p95": 3253.9453125,
      "p99": 3254.42578125
    },
    "cpu_pct": {
      "count": 682,
      "mean": 72.76876832844576,
      "min": 0.0,
      "max": 100.5,
      "p50": 73.2,
      "p90": 76.2,
      "p95": 77.8,
      "p99": 83.20899999999993
    },
    "gpu_used_mib": {
      "count": 682,
      "mean": 2654.526392961877,
      "min": 2621.0,
      "max": 2696.0,
      "p50": 2662.0,
      "p90": 2667.0,
      "p95": 2673.0,
      "p99": 2680.19
    },
    "gpu_util_pct": {
      "count": 682,
      "mean": 36.28299120234604,
      "min": 3.0,
      "max": 50.0,
      "p50": 36.0,
      "p90": 37.0,
      "p95": 38.0,
      "p99": 38.18999999999994
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:33122"
    ]
  }
}
```
