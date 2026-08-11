# Load-suite run `20260810T085800Z-switching`

- **Scenario:** switching
- **Timestamp (UTC):** 2026-08-10T09:16:09Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:e2c1c712b035b6ac (17981 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 1088.4 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-REL-006 | results_echoing_own_stream_model | proportion | >= 1.000 ratio | 1.000 | pass |
| NFR-REL-006 | rss_delta_vs_baseline | max | <= 5 percent | -0.041 | pass |
| NFR-REL-006 | gpu_memory_delta_vs_baseline | max | <= 64 MiB | 11.000 | pass |
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
| event_loop_lag | p99 | 2.076 |
| event_loop_lag | max | 4.911 |
| gpu_memory_delta_vs_baseline | max | 11.000 |
| results_echoing_own_stream_model | proportion | 1.000 |
| rss_delta_vs_baseline | max | -0.041 |

## Characterisation

```json
{
  "resources": {
    "rss_mib": {
      "count": 1038,
      "mean": 3259.0074361753373,
      "min": 3152.375,
      "max": 3266.5078125,
      "p50": 3263.78515625,
      "p90": 3264.29296875,
      "p95": 3266.1640625,
      "p99": 3266.5078125
    },
    "cpu_pct": {
      "count": 1038,
      "mean": 26.75279383429672,
      "min": 0.0,
      "max": 66.8,
      "p50": 38.1,
      "p90": 46.16000000000001,
      "p95": 49.8,
      "p99": 55.81499999999994
    },
    "gpu_used_mib": {
      "count": 1038,
      "mean": 2523.734104046243,
      "min": 2504.0,
      "max": 2579.0,
      "p50": 2524.0,
      "p90": 2526.0,
      "p95": 2531.0,
      "p99": 2575.0
    },
    "gpu_util_pct": {
      "count": 1038,
      "mean": 21.815028901734102,
      "min": 0.0,
      "max": 65.0,
      "p50": 25.0,
      "p90": 39.0,
      "p95": 42.0,
      "p99": 61.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:35870"
    ]
  }
}
```
