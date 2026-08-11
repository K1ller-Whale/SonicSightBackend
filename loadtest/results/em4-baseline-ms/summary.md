# Load-suite run `20260811T055927Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-11T06:11:25Z
- **SonicSightBackend:** `7cb179bb4064` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:f00d1071195028c6 (38683 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 718.1 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-002 | time_to_first_nonbuffering_result | p95 | <= 3.500 s | 2.730 | pass |
| NFR-PERF-003 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-004 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-005 | inference_time_ms | p95 | <= 200 ms | 154.000 | pass |
| NFR-PERF-005 | total_server_time_ms | report | report —  | {'p95': 249.0, 'p99': 249.0, 'max': 372} | characterisation |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 2.143 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 23.155 | pass |
| NFR-PERF-008 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-PERF-015 | inter_result_interval_abs_dev_from_125ms | p95 | <= 125.000 ms | 1.197 | pass |
| NFR-FUNC-001 | — | — |  —  | — | condition-mismatch |
| NFR-FUNC-002 | cross_check_margin | value | >= 0.300 correlation-margin | — | not-measured |
| NFR-FUNC-004 | results_meeting_multisensory_contract | proportion | >= 1.000 ratio | 1.000 | pass |
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
| event_loop_lag | p99 | 2.143 |
| event_loop_lag | max | 23.155 |
| inference_time_ms | p95 | 154.000 |
| inference_time_ms | p99 | 159.000 |
| inference_time_ms | max | 362 |
| inter_result_interval_abs_dev_from_125ms | p95 | 1.197 |
| results_meeting_multisensory_contract | proportion | 1.000 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 2.730 |
| total_server_time_ms | p95 | 249.000 |
| total_server_time_ms | p99 | 249.000 |
| total_server_time_ms | max | 372 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 2.7176875546666843,
    "min": 2.502649705999943,
    "max": 7.003830461999996,
    "p50": 2.5031981009999527,
    "p90": 2.5047176240000226,
    "p95": 2.504814084000145,
    "p99": 6.104027186400029
  },
  "inter_result_interval_ms": {
    "count": 2652,
    "mean": 249.24776817420806,
    "min": 145.9614010000223,
    "max": 254.06613600000583,
    "p50": 249.92500000007567,
    "p90": 250.79231069998968,
    "p95": 250.99767349998388,
    "p99": 251.56564507995654
  },
  "cadence_abs_dev_ms": {
    "count": 2652,
    "mean": 1.224897081954734,
    "min": 0.00042426294612596394,
    "max": 104.01592326301625,
    "p50": 0.3706975000454804,
    "p90": 1.050874362924788,
    "p95": 1.321944623682156,
    "p99": 2.6588579930250225
  },
  "total_server_time_ms": {
    "count": 2652,
    "mean": 250.8891402714932,
    "min": 244.0,
    "max": 4872.0,
    "p50": 248.0,
    "p90": 249.0,
    "p95": 249.0,
    "p99": 250.0
  },
  "audio_age_lag_ms": {
    "count": 2673,
    "mean": 1381.1934547366245,
    "min": 1375.1529010000922,
    "max": 6001.591807999944,
    "p50": 1377.304406999997,
    "p90": 1378.0513581999003,
    "p95": 1378.305715399938,
    "p99": 1501.6085121599826
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 676,
      "mean": 2932.0820601423816,
      "min": 2629.0,
      "max": 2990.453125,
      "p50": 2917.26171875,
      "p90": 2990.1796875,
      "p95": 2990.37890625,
      "p99": 2990.3984375
    },
    "cpu_pct": {
      "count": 676,
      "mean": 50.62662721893492,
      "min": 0.0,
      "max": 133.0,
      "p50": 52.6,
      "p90": 56.6,
      "p95": 57.425,
      "p99": 59.6
    },
    "gpu_used_mib": {
      "count": 676,
      "mean": 2144.5724852071007,
      "min": 1633.0,
      "max": 2164.0,
      "p50": 2147.0,
      "p90": 2152.0,
      "p95": 2157.0,
      "p99": 2164.0
    },
    "gpu_util_pct": {
      "count": 676,
      "mean": 49.02218934911242,
      "min": 0.0,
      "max": 71.0,
      "p50": 52.0,
      "p90": 53.0,
      "p95": 54.0,
      "p99": 57.5
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:52580"
    ]
  }
}
```
