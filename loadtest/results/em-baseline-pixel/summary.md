# Load-suite run `20260810T073854Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-10T07:50:53Z
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
| NFR-PERF-001 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-002 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-003 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-004 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-005 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 2.054 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 2.535 | pass |
| NFR-PERF-008 | query_round_trip | p95 | <= 250 ms | 1.560 | pass |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-FUNC-001 | — | — |  —  | — | condition-mismatch |
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
| event_loop_lag | p99 | 2.054 |
| event_loop_lag | max | 2.535 |
| inter_result_interval_abs_dev_from_125ms | p95 | 4.918 |
| query_round_trip | p95 | 1.560 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 6.245 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 6.150628751619176,
    "min": 6.057645312000204,
    "max": 6.374614170000314,
    "p50": 6.156183371000225,
    "p90": 6.229432819999602,
    "p95": 6.237754842999493,
    "p99": 6.34724230460015
  },
  "inter_result_interval_ms": {
    "count": 4773,
    "mean": 123.46684639891025,
    "min": 1.0537000007388997,
    "max": 1123.295839000093,
    "p50": 124.90600299952348,
    "p90": 127.92156680006883,
    "p95": 128.98492139993323,
    "p99": 131.34723672010293
  },
  "cadence_abs_dev_ms": {
    "count": 4773,
    "mean": 3.7886756041108556,
    "min": 0.00014286909276961524,
    "max": 998.3071768685738,
    "p50": 1.4772431311678247,
    "p90": 4.266927520927539,
    "p95": 5.726739573988876,
    "p99": 84.42353981202471
  },
  "total_server_time_ms": {
    "count": 0
  },
  "audio_age_lag_ms": {
    "count": 4794,
    "mean": 3201.9040968919394,
    "min": 3158.7102209996374,
    "max": 4165.074718999676,
    "p50": 3175.913654500164,
    "p90": 3300.8971728002507,
    "p95": 3303.407876549909,
    "p99": 3308.824020579268
  },
  "query_rtt_ms": {
    "count": 419,
    "mean": 1.5941954725392617,
    "min": 1.1101000000053318,
    "max": 83.20590400035144,
    "p50": 1.3944000002084067,
    "p90": 1.4783999999053776,
    "p95": 1.5596499999446676,
    "p99": 1.9473021804151356
  },
  "resources": {
    "rss_mib": {
      "count": 687,
      "mean": 2612.0465850163755,
      "min": 2601.6953125,
      "max": 2613.8359375,
      "p50": 2612.3046875,
      "p90": 2613.83203125,
      "p95": 2613.8359375,
      "p99": 2613.8359375
    },
    "cpu_pct": {
      "count": 687,
      "mean": 37.343522561863175,
      "min": 0.0,
      "max": 83.5,
      "p50": 43.1,
      "p90": 47.0,
      "p95": 48.76999999999999,
      "p99": 56.31399999999999
    },
    "gpu_used_mib": {
      "count": 687,
      "mean": 1513.1164483260552,
      "min": 1451.0,
      "max": 2504.0,
      "p50": 1516.0,
      "p90": 1516.0,
      "p95": 1519.0,
      "p99": 1652.6799999999998
    },
    "gpu_util_pct": {
      "count": 687,
      "mean": 26.29985443959243,
      "min": 0.0,
      "max": 61.0,
      "p50": 31.0,
      "p90": 33.0,
      "p95": 34.0,
      "p99": 45.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:58430"
    ]
  }
}
```
