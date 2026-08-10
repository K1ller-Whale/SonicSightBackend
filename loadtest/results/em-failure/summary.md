# Load-suite run `20260810T091610Z-failure-injection`

- **Scenario:** failure-injection
- **Timestamp (UTC):** 2026-08-10T10:04:46Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:e2c1c712b035b6ac (17981 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 2914.8 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-REL-001 | injected_cases_contained | proportion | >= 1.000 ratio | 1.000 | pass |
| NFR-REL-001 | healthy_session_perf_003_holds_during_injection | boolean | == True bool | True | pass |
| NFR-REL-002 | rss_delta_vs_baseline | max | <= 5 percent | -0.150 | pass |
| NFR-REL-002 | gpu_memory_delta_vs_baseline | max | <= 64 MiB | -13.000 | pass |
| NFR-REL-002 | stream_open_minus_close_log_lines | count | == 0 count | — | not-measured |
| NFR-REL-003 | oversized_rejected_at_transport | proportion | >= 1.000 ratio | 1.000 | pass |
| NFR-REL-003 | next_session_succeeds | boolean | == True bool | True | pass |
| NFR-REL-004 | injections_with_defined_behaviour | proportion | >= 1.000 ratio | 1.000 | pass |
| NFR-REL-004 | total_server_time_ms_during_injection | p95 | <= 500 ms | 84.000 | pass |
| NFR-REL-007 | gpu_oom_stream_fails_server_survives | boolean |  —  | — | inspection |
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
| event_loop_lag | p99 | 2.096 |
| event_loop_lag | max | 62.227 |
| gpu_memory_delta_vs_baseline | max | -13.000 |
| healthy_session_perf_003_holds_during_injection | boolean | True |
| injected_cases_contained | proportion | 1.000 |
| injections_with_defined_behaviour | proportion | 1.000 |
| next_session_succeeds | boolean | True |
| oversized_rejected_at_transport | proportion | 1.000 |
| rss_delta_vs_baseline | max | -0.150 |
| total_server_time_ms_during_injection | p95 | 84.000 |

## Characterisation

```json
{
  "injections": {
    "truncated_pcm": {
      "victim_error": "buffer size must be a multiple of element size",
      "victim_results": 0,
      "healthy_results": 2594,
      "server_alive_after": true
    },
    "bad_jpeg": {
      "victim_error": null,
      "victim_results": 21,
      "healthy_results": 2594,
      "server_alive_after": true
    },
    "zero_len_audio": {
      "victim_error": null,
      "victim_results": 2593,
      "healthy_results": 2594,
      "server_alive_after": true
    },
    "wrong_rate_field": {
      "victim_error": null,
      "victim_results": 170,
      "healthy_results": 2594,
      "server_alive_after": true
    },
    "silent_audio": {
      "victim_error": null,
      "victim_results": 2594,
      "healthy_results": 2594,
      "server_alive_after": true
    },
    "clipping_audio": {
      "victim_error": null,
      "victim_results": 2594,
      "healthy_results": 2594,
      "server_alive_after": true
    },
    "black_frames": {
      "victim_error": null,
      "victim_results": 2594,
      "healthy_results": 2594,
      "server_alive_after": true
    }
  },
  "resources": {
    "rss_mib": {
      "count": 2777,
      "mean": 3233.5117820489736,
      "min": 3175.796875,
      "max": 3263.12890625,
      "p50": 3233.5859375,
      "p90": 3246.41015625,
      "p95": 3250.79296875,
      "p99": 3263.12890625
    },
    "cpu_pct": {
      "count": 2777,
      "mean": 54.853042851998566,
      "min": 0.0,
      "max": 98.7,
      "p50": 55.5,
      "p90": 74.6,
      "p95": 76.0,
      "p99": 80.21999999999989
    },
    "gpu_used_mib": {
      "count": 2777,
      "mean": 2595.175369103349,
      "min": 2516.0,
      "max": 2732.0,
      "p50": 2622.0,
      "p90": 2661.4,
      "p95": 2674.0,
      "p99": 2704.4799999999996
    },
    "gpu_util_pct": {
      "count": 2777,
      "mean": 29.58228303925099,
      "min": 0.0,
      "max": 68.0,
      "p50": 33.0,
      "p90": 37.0,
      "p95": 37.0,
      "p99": 39.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:35002",
      "::ffff:127.0.0.1:49852"
    ]
  }
}
```
