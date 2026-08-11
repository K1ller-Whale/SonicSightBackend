# Load-suite run `20260810T170619Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-10T17:18:18Z
- **SonicSightBackend:** `7cb179bb4064` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:052388852cee79cb (37692 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 719.0 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | time_to_first_nonbuffering_result | p95 | <= 8.000 s | 6.303 | pass |
| NFR-PERF-002 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-003 | inter_result_interval_abs_dev_from_125ms | p95 | <= 62.500 ms | 0.865 | pass |
| NFR-PERF-004 | inference_time_ms | p95 | <= 140 ms | 123.000 | pass |
| NFR-PERF-004 | inference_time_ms | p99 | <= 250 ms | 123.000 | pass |
| NFR-PERF-004 | inference_time_ms | max | <= 1500 ms | 249 | pass |
| NFR-PERF-004 | total_server_time_ms | report | report —  | {'p95': 124.0, 'p99': 125.0, 'max': 249} | characterisation |
| NFR-PERF-005 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 2.065 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 2.487 | pass |
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
| event_loop_lag | p99 | 2.065 |
| event_loop_lag | max | 2.487 |
| inference_time_ms | p95 | 123.000 |
| inference_time_ms | p99 | 123.000 |
| inference_time_ms | max | 249 |
| inter_result_interval_abs_dev_from_125ms | p95 | 0.865 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 6.303 |
| total_server_time_ms | p95 | 124.000 |
| total_server_time_ms | p99 | 125.000 |
| total_server_time_ms | max | 249 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 6.2409183747142505,
    "min": 6.127186272000017,
    "max": 7.25303366099979,
    "p50": 6.252107358000103,
    "p90": 6.253313301000162,
    "p95": 6.253492092000215,
    "p99": 7.053125347199876
  },
  "inter_result_interval_ms": {
    "count": 4616,
    "mean": 127.36586243349204,
    "min": 55.08800000006886,
    "max": 1375.1745029999256,
    "p50": 125.17385550017934,
    "p90": 125.4906994997782,
    "p95": 125.63494999972136,
    "p99": 126.16190499988988
  },
  "cadence_abs_dev_ms": {
    "count": 4616,
    "mean": 3.3780559443582434,
    "min": 0.00048186869094024587,
    "max": 1250.1858408684063,
    "p50": 0.3624428684167924,
    "p90": 0.8309121315244283,
    "p95": 0.92171188159341,
    "p99": 2.3562487182096135
  },
  "total_server_time_ms": {
    "count": 4616,
    "mean": 127.24805025996534,
    "min": 117.0,
    "max": 1375.0,
    "p50": 124.0,
    "p90": 124.0,
    "p95": 124.0,
    "p99": 125.0
  },
  "audio_age_lag_ms": {
    "count": 4637,
    "mean": 3280.348210786287,
    "min": 3247.4457760004043,
    "max": 5385.180988000229,
    "p50": 3251.902427999994,
    "p90": 3376.806403200044,
    "p95": 3377.1207980002146,
    "p99": 3377.766700720149
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 686,
      "mean": 3169.979153380102,
      "min": 3040.17578125,
      "max": 3243.0,
      "p50": 3157.12109375,
      "p90": 3186.05859375,
      "p95": 3186.08984375,
      "p99": 3242.25234375
    },
    "cpu_pct": {
      "count": 686,
      "mean": 43.69285714285714,
      "min": 0.0,
      "max": 79.5,
      "p50": 50.6,
      "p90": 53.8,
      "p95": 54.6,
      "p99": 74.09499999999997
    },
    "gpu_used_mib": {
      "count": 686,
      "mean": 2063.17055393586,
      "min": 1650.0,
      "max": 5879.0,
      "p50": 2052.0,
      "p90": 2116.0,
      "p95": 2116.0,
      "p99": 2324.0
    },
    "gpu_util_pct": {
      "count": 686,
      "mean": 27.889212827988338,
      "min": 0.0,
      "max": 68.0,
      "p50": 32.0,
      "p90": 33.0,
      "p95": 33.0,
      "p99": 64.14999999999998
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:33066"
    ]
  }
}
```
