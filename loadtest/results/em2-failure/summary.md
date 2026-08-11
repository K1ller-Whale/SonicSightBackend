# Load-suite run `20260810T122354Z-failure-injection`

- **Scenario:** failure-injection
- **Timestamp (UTC):** 2026-08-10T13:12:30Z
- **SonicSightBackend:** `f840ce16afdd` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:c1b14b0c73772c80 (36367 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 2915.2 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-REL-001 | injected_cases_contained | proportion | >= 1.000 ratio | 1.000 | pass |
| NFR-REL-001 | healthy_session_perf_003_holds_during_injection | boolean | == True bool | True | pass |
| NFR-REL-002 | rss_delta_vs_baseline | max | <= 5 percent | -0.137 | pass |
| NFR-REL-002 | gpu_memory_delta_vs_baseline | max | <= 64 MiB | 59.000 | pass |
| NFR-REL-002 | stream_open_minus_close_log_lines | count | == 0 count | 0 | pass |
| NFR-REL-003 | oversized_rejected_at_transport | proportion | >= 1.000 ratio | 1.000 | pass |
| NFR-REL-003 | next_session_succeeds | boolean | == True bool | True | pass |
| NFR-REL-004 | injections_with_defined_behaviour | proportion | >= 1.000 ratio | 1.000 | pass |
| NFR-REL-004 | total_server_time_ms_during_injection | p95 | <= 280 ms | 91.000 | pass |
| NFR-REL-007 | error_result_delivered_and_stream_closed_within | max | <= 5 s | — | not-measured |
| NFR-REL-007 | subsequent_session_succeeds_without_restart | boolean | == True bool | — | not-measured |
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
| event_loop_lag | p99 | 2.098 |
| event_loop_lag | max | 5.284 |
| gpu_memory_delta_vs_baseline | max | 59.000 |
| healthy_session_perf_003_holds_during_injection | boolean | True |
| injected_cases_contained | proportion | 1.000 |
| injections_with_defined_behaviour | proportion | 1.000 |
| next_session_succeeds | boolean | True |
| oversized_rejected_at_transport | proportion | 1.000 |
| rss_delta_vs_baseline | max | -0.137 |
| stream_open_minus_close_log_lines | count | 0 |
| total_server_time_ms_during_injection | p95 | 91.000 |

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
      "count": 2613,
      "mean": 3015.399102743494,
      "min": 2743.2421875,
      "max": 3081.07421875,
      "p50": 3028.390625,
      "p90": 3077.21484375,
      "p95": 3079.046875,
      "p99": 3081.07421875
    },
    "cpu_pct": {
      "count": 2613,
      "mean": 55.94994259471872,
      "min": 0.0,
      "max": 125.5,
      "p50": 54.4,
      "p90": 77.4,
      "p95": 81.14,
      "p99": 96.80400000000009
    },
    "gpu_used_mib": {
      "count": 2613,
      "mean": 2357.7409108304632,
      "min": 1896.0,
      "max": 3780.0,
      "p50": 2351.0,
      "p90": 2511.6000000000004,
      "p95": 2576.0,
      "p99": 2577.0
    },
    "gpu_util_pct": {
      "count": 2613,
      "mean": 30.413318025258324,
      "min": 0.0,
      "max": 68.0,
      "p50": 33.0,
      "p90": 38.80000000000018,
      "p95": 40.0,
      "p99": 56.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:50260"
    ]
  }
}
```
