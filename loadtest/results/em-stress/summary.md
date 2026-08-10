# Load-suite run `20260810T082703Z-stress`

- **Scenario:** stress
- **Timestamp (UTC):** 2026-08-10T08:48:06Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:98f7877bbe5216fc (17587 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 1262.9 s

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
| concurrency_ceiling | first_breaching_concurrency | 3 |
| concurrency_ceiling | note | None = no breach up to --max-sessions |
| event_loop_lag | p99 | 2.097 |
| event_loop_lag | max | 3.145 |
| sessions_meeting_perf_003_and_004 | count | 2 |

## Characterisation

```json
{
  "steps": {
    "1": {
      "sessions_ok": 1,
      "of": 1,
      "time_to_first_nonbuffering_result": {
        "p95": 6.131879378999656
      },
      "inter_result_interval_abs_dev_from_125ms": {
        "p95": 7.018861668283304
      },
      "total_server_time_ms": {
        "p95": 61.0,
        "p99": 64.0,
        "max": 254
      },
      "sequence_gap_rate": {
        "proportion": 0.0
      }
    },
    "2": {
      "sessions_ok": 2,
      "of": 2,
      "time_to_first_nonbuffering_result": {
        "p95": 6.1901967280987265
      },
      "inter_result_interval_abs_dev_from_125ms": {
        "p95": 6.995033381618896
      },
      "total_server_time_ms": {
        "p95": 84.0,
        "p99": 89.0,
        "max": 323
      },
      "sequence_gap_rate": {
        "proportion": 0.0
      }
    },
    "3": {
      "sessions_ok": 0,
      "of": 3,
      "time_to_first_nonbuffering_result": {
        "p95": 6.797172443499221
      },
      "inter_result_interval_abs_dev_from_125ms": {
        "p95": 6.66555573095252
      },
      "total_server_time_ms": {
        "p95": 124.0,
        "p99": 130.0,
        "max": 1231
      },
      "sequence_gap_rate": {
        "proportion": 0.0
      }
    }
  },
  "resources": {
    "rss_mib": {
      "count": 1204,
      "mean": 2996.150815640573,
      "min": 2959.19921875,
      "max": 3052.609375,
      "p50": 2983.80078125,
      "p90": 3044.625,
      "p95": 3046.26953125,
      "p99": 3051.6588671875
    },
    "cpu_pct": {
      "count": 1204,
      "mean": 77.6218438538206,
      "min": 0.0,
      "max": 118.6,
      "p50": 73.8,
      "p90": 111.97000000000001,
      "p95": 113.0,
      "p99": 115.994
    },
    "gpu_used_mib": {
      "count": 1204,
      "mean": 2299.737541528239,
      "min": 2020.0,
      "max": 3552.0,
      "p50": 2294.0,
      "p90": 2337.0,
      "p95": 2361.0,
      "p99": 2398.0
    },
    "gpu_util_pct": {
      "count": 1204,
      "mean": 41.352159468438536,
      "min": 0.0,
      "max": 65.0,
      "p50": 37.0,
      "p90": 58.0,
      "p95": 58.0,
      "p99": 59.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:50476"
    ]
  }
}
```
