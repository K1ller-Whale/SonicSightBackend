# Load-suite run `20260810T081501Z-load`

- **Scenario:** load
- **Timestamp (UTC):** 2026-08-10T08:27:01Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:98f7877bbe5216fc (17587 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 720.0 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-007 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-009 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-010 | sessions_meeting_perf_002_and_005 | count | >= 1 sessions | 0 | FAIL |
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
| event_loop_lag | p99 | 2.126 |
| event_loop_lag | max | 5.371 |
| gpu_memory_used | max_over_steady_state | 2371.000 |
| inter_result_interval_abs_dev_from_125ms | p95 | 131.225 |
| non_lan_endpoints_contacted | count | 0 |
| sequence_gap_rate | proportion | 0.000 |
| sessions_meeting_perf_002_and_005 | count | 0 |
| time_to_first_nonbuffering_result | p95 | 2.441 |
| total_server_time_ms | p95 | 164.000 |
| total_server_time_ms | p99 | 171.000 |
| total_server_time_ms | max | 309 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 1,
    "mean": 2.4409580459996505,
    "min": 2.4409580459996505,
    "max": 2.4409580459996505,
    "p50": 2.4409580459996505,
    "p90": 2.4409580459996505,
    "p95": 2.4409580459996505,
    "p99": 2.4409580459996505
  },
  "inter_result_interval_ms": {
    "count": 2153,
    "mean": 333.2440674830468,
    "min": 6.573599000148533,
    "max": 404.00569299981726,
    "p50": 373.15479399967444,
    "p90": 379.1666394001368,
    "p95": 381.1771270002282,
    "p99": 386.19734388052166
  },
  "cadence_abs_dev_ms": {
    "count": 2153,
    "mean": 84.76177732209392,
    "min": 0.01377673693923498,
    "max": 243.40372526289002,
    "p50": 123.18027573697464,
    "p90": 129.1926787369971,
    "p95": 131.2249563367527,
    "p99": 136.28488745694767
  },
  "total_server_time_ms": {
    "count": 2153,
    "mean": 154.7473293079424,
    "min": 148.0,
    "max": 309.0,
    "p50": 153.0,
    "p90": 161.0,
    "p95": 164.0,
    "p99": 171.0
  },
  "audio_age_lag_ms": {
    "count": 2154,
    "mean": 1302.5238249596011,
    "min": 1274.2965959996582,
    "max": 1438.8430549997793,
    "p50": 1282.4748509997335,
    "p90": 1403.8907192998522,
    "p95": 1405.655066400277,
    "p99": 1408.6551725298887
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 687,
      "mean": 2974.7091748544394,
      "min": 2925.08203125,
      "max": 2996.85546875,
      "p50": 2974.21875,
      "p90": 2992.1015625,
      "p95": 2996.824609375,
      "p99": 2996.8515625
    },
    "cpu_pct": {
      "count": 687,
      "mean": 42.25254730713246,
      "min": 0.0,
      "max": 54.7,
      "p50": 42.0,
      "p90": 45.8,
      "p95": 46.8,
      "p99": 48.769999999999996
    },
    "gpu_used_mib": {
      "count": 687,
      "mean": 2333.4192139737993,
      "min": 2251.0,
      "max": 2399.0,
      "p50": 2340.0,
      "p90": 2357.0,
      "p95": 2371.0,
      "p99": 2381.0
    },
    "gpu_util_pct": {
      "count": 687,
      "mean": 40.04221251819505,
      "min": 3.0,
      "max": 55.0,
      "p50": 40.0,
      "p90": 42.0,
      "p95": 43.0,
      "p99": 44.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:59104"
    ]
  }
}
```
