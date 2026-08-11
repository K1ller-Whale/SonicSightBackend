# Load-suite run `20260811T061131Z-load`

- **Scenario:** load
- **Timestamp (UTC):** 2026-08-11T06:23:31Z
- **SonicSightBackend:** `7cb179bb4064` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:f00d1071195028c6 (38683 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 720.0 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-007 | — | — |  —  | — | condition-mismatch |
| NFR-PERF-010 | gpu_memory_used | max_over_steady_state | <= 5632 MiB | 2299.000 | pass |
| NFR-PERF-013 | sequence_gap_rate | proportion | <= 0.005 ratio | 0.000 | pass |
| NFR-PERF-016 | sessions_meeting_perf_015_and_005 | count | >= 1 sessions | 1 | pass |
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
| event_loop_lag | p99 | 2.125 |
| event_loop_lag | max | 17.493 |
| gpu_memory_used | max_over_steady_state | 2299.000 |
| inference_time_ms | p95 | 157.000 |
| inference_time_ms | p99 | 172.640 |
| inference_time_ms | max | 362 |
| inter_result_interval_abs_dev_from_125ms | p95 | 1.172 |
| non_lan_endpoints_contacted | count | 0 |
| sequence_gap_rate | proportion | 0.000 |
| sessions_meeting_perf_015_and_005 | count | 1 |
| time_to_first_nonbuffering_result | p95 | 2.504 |
| total_server_time_ms | p95 | 249.000 |
| total_server_time_ms | p99 | 249.000 |
| total_server_time_ms | max | 384 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 1,
    "mean": 2.5037421080000968,
    "min": 2.5037421080000968,
    "max": 2.5037421080000968,
    "p50": 2.5037421080000968,
    "p90": 2.5037421080000968,
    "p95": 2.5037421080000968,
    "p99": 2.5037421080000968
  },
  "inter_result_interval_ms": {
    "count": 2869,
    "mean": 250.0549787682119,
    "min": 157.56030200009263,
    "max": 386.80670200028544,
    "p50": 249.94173799996133,
    "p90": 250.71666219996587,
    "p95": 250.93920220006112,
    "p99": 251.4263233200927
  },
  "cadence_abs_dev_ms": {
    "count": 2869,
    "mean": 0.5404848087683787,
    "min": 0.00022526277689394192,
    "max": 136.8293777372469,
    "p50": 0.2929777369513431,
    "p90": 0.9586301579706743,
    "p95": 1.171795547399142,
    "p99": 1.5537197370099451
  },
  "total_server_time_ms": {
    "count": 2869,
    "mean": 248.31718368769606,
    "min": 236.0,
    "max": 384.0,
    "p50": 248.0,
    "p90": 249.0,
    "p95": 249.0,
    "p99": 249.0
  },
  "audio_age_lag_ms": {
    "count": 2870,
    "mean": 1401.453025975253,
    "min": 1375.3513980000207,
    "max": 1660.0005229997805,
    "p50": 1377.4966595001388,
    "p90": 1502.1870064001178,
    "p95": 1502.55869599996,
    "p99": 1503.1037376898257
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 677,
      "mean": 3162.3277384139587,
      "min": 2990.1796875,
      "max": 3272.55078125,
      "p50": 3179.34765625,
      "p90": 3266.1890624999996,
      "p95": 3272.515625,
      "p99": 3272.55078125
    },
    "cpu_pct": {
      "count": 677,
      "mean": 53.433530280649926,
      "min": 0.0,
      "max": 75.8,
      "p50": 53.2,
      "p90": 57.0,
      "p95": 58.0,
      "p99": 60.724000000000004
    },
    "gpu_used_mib": {
      "count": 677,
      "mean": 2184.440177252585,
      "min": 2146.0,
      "max": 2299.0,
      "p50": 2155.0,
      "p90": 2269.0,
      "p95": 2271.0,
      "p99": 2286.24
    },
    "gpu_util_pct": {
      "count": 677,
      "mean": 52.12850812407681,
      "min": 0.0,
      "max": 57.0,
      "p50": 52.0,
      "p90": 53.0,
      "p95": 53.0,
      "p99": 55.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:60452"
    ]
  }
}
```
