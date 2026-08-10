# Load-suite run `20260810T080257Z-load`

- **Scenario:** load
- **Timestamp (UTC):** 2026-08-10T08:14:59Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:98f7877bbe5216fc (17587 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 721.0 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-007 | sessions_meeting_perf_003_and_004 | count | >= 2 sessions | 0 | FAIL |
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
| event_loop_lag | p99 | 2.124 |
| event_loop_lag | max | 5.954 |
| gpu_memory_used | max_over_steady_state | 2407.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 10.442 |
| non_lan_endpoints_contacted | count | 0 |
| sequence_gap_rate | proportion | 0.000 |
| sessions_meeting_perf_003_and_004 | count | 0 |
| time_to_first_nonbuffering_result | p95 | 6.542 |
| total_server_time_ms | p95 | 88.000 |
| total_server_time_ms | p99 | 109.750 |
| total_server_time_ms | max | 1258 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 2,
    "mean": 6.332363550000082,
    "min": 6.09984201000043,
    "max": 6.564885089999734,
    "p50": 6.332363550000082,
    "p90": 6.518380781999804,
    "p95": 6.541632935999769,
    "p99": 6.560234659199741
  },
  "inter_result_interval_ms": {
    "count": 11426,
    "mean": 124.90594898783476,
    "min": 1.711601999886625,
    "max": 1278.568943999744,
    "p50": 124.9140729996725,
    "p90": 128.7739029999102,
    "p95": 130.84233724976002,
    "p99": 138.85584950048724
  },
  "cadence_abs_dev_ms": {
    "count": 11426,
    "mean": 3.7734192253294516,
    "min": 0.00013486917737282056,
    "max": 1153.5802818682248,
    "p50": 1.6462859994135215,
    "p90": 6.629174000408966,
    "p95": 10.441808565825,
    "p99": 38.30340681578093
  },
  "total_server_time_ms": {
    "count": 11426,
    "mean": 62.23148958515666,
    "min": 36.0,
    "max": 1258.0,
    "p50": 76.0,
    "p90": 83.0,
    "p95": 88.0,
    "p99": 109.75
  },
  "audio_age_lag_ms": {
    "count": 11428,
    "mean": 3277.087225530104,
    "min": 3158.385131000614,
    "max": 5416.787563000071,
    "p50": 3291.205275500033,
    "p90": 3333.3348613001363,
    "p95": 3336.029618550174,
    "p99": 3954.06697698022
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 686,
      "mean": 2925.0708591472303,
      "min": 2913.27734375,
      "max": 2964.07421875,
      "p50": 2921.8515625,
      "p90": 2927.93359375,
      "p95": 2951.55859375,
      "p99": 2963.63671875
    },
    "cpu_pct": {
      "count": 686,
      "mean": 76.9695335276968,
      "min": 0.0,
      "max": 125.9,
      "p50": 75.4,
      "p90": 83.5,
      "p95": 91.25,
      "p99": 121.46499999999993
    },
    "gpu_used_mib": {
      "count": 686,
      "mean": 2311.844023323615,
      "min": 2030.0,
      "max": 2407.0,
      "p50": 2312.0,
      "p90": 2355.0,
      "p95": 2360.0,
      "p99": 2383.15
    },
    "gpu_util_pct": {
      "count": 686,
      "mean": 37.76967930029154,
      "min": 0.0,
      "max": 68.0,
      "p50": 38.0,
      "p90": 39.0,
      "p95": 40.0,
      "p99": 51.899999999999864
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:43000"
    ]
  }
}
```
