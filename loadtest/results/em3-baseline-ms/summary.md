# Load-suite run `20260810T164214Z-baseline`

- **Scenario:** baseline
- **Timestamp (UTC):** 2026-08-10T16:54:13Z
- **SonicSightBackend:** `7cb179bb4064` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:bcf70086bf00a854 (36566 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 718.1 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-001 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-002 | time_to_first_nonbuffering_result | p95 | <= 3.500 s | 2.680 | pass |
| NFR-PERF-003 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-004 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-005 | total_server_time_ms | p95 | <= 200 ms | 249.000 | FAIL |
| NFR-PERF-006 | event_loop_lag | p99 | <= 100 ms | 2.135 | pass |
| NFR-PERF-006 | event_loop_lag | max | <= 500 ms | 5.115 | pass |
| NFR-PERF-008 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-PERF-015 | inter_result_interval_abs_dev_from_125ms | p95 | <= 125.000 ms | 1.180 | pass |
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
| event_loop_lag | p99 | 2.135 |
| event_loop_lag | max | 5.115 |
| inter_result_interval_abs_dev_from_125ms | p95 | 1.180 |
| results_meeting_multisensory_contract | proportion | 1.000 |
| sequence_gap_rate | proportion | 0.000 |
| time_to_first_nonbuffering_result | p95 | 2.680 |
| total_server_time_ms | p95 | 249.000 |
| total_server_time_ms | p99 | 249.000 |
| total_server_time_ms | max | 370 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 21,
    "mean": 2.670098602142876,
    "min": 2.502662911999778,
    "max": 6.003880303000187,
    "p50": 2.503370813000174,
    "p90": 2.5041371250003976,
    "p95": 2.504675247000023,
    "p99": 5.304039291800157
  },
  "inter_result_interval_ms": {
    "count": 2656,
    "mean": 249.2349123396087,
    "min": 149.40389999992476,
    "max": 255.10669799996322,
    "p50": 249.91660199998478,
    "p90": 250.7554574999631,
    "p95": 250.9746985001584,
    "p99": 251.44657729988467
  },
  "cadence_abs_dev_ms": {
    "count": 2656,
    "mean": 1.2361124626123072,
    "min": 2.126288791259867e-05,
    "max": 100.5734242631138,
    "p50": 0.39943950014276197,
    "p90": 0.9957500001291919,
    "p95": 1.2357373816678319,
    "p99": 2.3065971762246087
  },
  "total_server_time_ms": {
    "count": 2656,
    "mean": 250.51807228915663,
    "min": 243.0,
    "max": 3872.0,
    "p50": 248.0,
    "p90": 249.0,
    "p95": 249.0,
    "p99": 250.0
  },
  "audio_age_lag_ms": {
    "count": 2677,
    "mean": 1380.8296410190508,
    "min": 1375.190918000044,
    "max": 5002.279620000081,
    "p50": 1377.3576040002808,
    "p90": 1378.0610404000072,
    "p95": 1378.322880200085,
    "p99": 1501.589118879856
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 683,
      "mean": 2741.7912358162516,
      "min": 2369.06640625,
      "max": 2861.3828125,
      "p50": 2693.07421875,
      "p90": 2833.078125,
      "p95": 2860.80859375,
      "p99": 2861.3203125
    },
    "cpu_pct": {
      "count": 683,
      "mean": 51.39663250366032,
      "min": 0.0,
      "max": 158.9,
      "p50": 53.5,
      "p90": 57.28000000000001,
      "p95": 58.0,
      "p99": 59.91799999999999
    },
    "gpu_used_mib": {
      "count": 683,
      "mean": 1647.9941434846266,
      "min": 1123.0,
      "max": 1658.0,
      "p50": 1650.0,
      "p90": 1651.0,
      "p95": 1653.0,
      "p99": 1655.0
    },
    "gpu_util_pct": {
      "count": 683,
      "mean": 49.74670571010249,
      "min": 0.0,
      "max": 81.0,
      "p50": 53.0,
      "p90": 54.0,
      "p95": 54.0,
      "p99": 60.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:60518"
    ]
  }
}
```
