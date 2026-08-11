# Load-suite run `20260810T084808Z-stress`

- **Scenario:** stress
- **Timestamp (UTC):** 2026-08-10T08:55:08Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:e2c1c712b035b6ac (17981 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 419.9 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-007 | — | — |  —  | — | condition-mismatch |
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
| concurrency_ceiling | first_breaching_concurrency | 1 |
| concurrency_ceiling | note | None = no breach up to --max-sessions |
| event_loop_lag | p99 | 2.157 |
| event_loop_lag | max | 387.064 |
| sessions_meeting_perf_002_and_005 | count | 0 |

## Characterisation

```json
{
  "steps": {
    "1": {
      "sessions_ok": 0,
      "of": 1,
      "time_to_first_nonbuffering_result": {
        "p95": 2.513456195998515
      },
      "inter_result_interval_abs_dev_from_125ms": {
        "p95": 132.2303847359376
      },
      "total_server_time_ms": {
        "p95": 166.0,
        "p99": 180.96000000000004,
        "max": 382
      },
      "sequence_gap_rate": {
        "proportion": 0.0
      }
    }
  },
  "resources": {
    "rss_mib": {
      "count": 389,
      "mean": 3130.7131567320052,
      "min": 3052.609375,
      "max": 3146.20703125,
      "p50": 3139.87109375,
      "p90": 3142.34375,
      "p95": 3142.40625,
      "p99": 3146.20703125
    },
    "cpu_pct": {
      "count": 389,
      "mean": 42.78303341902314,
      "min": 0.0,
      "max": 66.2,
      "p50": 42.3,
      "p90": 46.919999999999995,
      "p95": 48.919999999999995,
      "p99": 58.46400000000001
    },
    "gpu_used_mib": {
      "count": 389,
      "mean": 2460.051413881748,
      "min": 2389.0,
      "max": 2593.0,
      "p50": 2480.0,
      "p90": 2520.0,
      "p95": 2529.6,
      "p99": 2557.84
    },
    "gpu_util_pct": {
      "count": 389,
      "mean": 39.84575835475579,
      "min": 0.0,
      "max": 46.0,
      "p50": 40.0,
      "p90": 42.0,
      "p95": 42.0,
      "p99": 43.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:34608"
    ]
  }
}
```
