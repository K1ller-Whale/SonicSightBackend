# Load-suite run `20260809T115246Z-switching`

- **Scenario:** switching
- **Timestamp (UTC):** 2026-08-09T12:10:50Z
- **SonicSightBackend:** `1a1842a8d159` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1650, 595.97, 4096 MiB
- **Host:** Windows-10-10.0.26200-SP0 — Python 3.11.9, torch 2.6.0+cu124, grpcio 1.78.0
- **nfr_targets.yaml:** sha256:afe89a851eb00392 (25614 B)
- **Checkpoints:** frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 1084.4 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-REL-006 | results_echoing_own_stream_model | proportion | >= 1.000 ratio | 1.000 | pass |
| NFR-REL-006 | rss_delta_vs_baseline | max | <= 5 percent | -26.826 | pass |
| NFR-REL-006 | gpu_memory_delta_vs_baseline | max | <= 64 MiB | 250.000 | FAIL |
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
| event_loop_lag | p99 | 28.000 |
| event_loop_lag | max | 29.000 |
| gpu_memory_delta_vs_baseline | max | 250.000 |
| results_echoing_own_stream_model | proportion | 1.000 |
| rss_delta_vs_baseline | max | -26.826 |

## Characterisation

```json
{
  "resources": {
    "rss_mib": {
      "count": 1022,
      "mean": 557.3632086289139,
      "min": 498.38671875,
      "max": 684.37890625,
      "p50": 535.51171875,
      "p90": 681.692578125,
      "p95": 682.83984375,
      "p99": 683.7475390625
    },
    "cpu_pct": {
      "count": 1022,
      "mean": 269.54911937377693,
      "min": 0.0,
      "max": 886.3,
      "p50": 99.25,
      "p90": 795.2,
      "p95": 825.495,
      "p99": 859.9239999999999
    },
    "gpu_used_mib": {
      "count": 1022,
      "mean": 1456.96771037182,
      "min": 1170.0,
      "max": 1644.0,
      "p50": 1461.5,
      "p90": 1490.0,
      "p95": 1510.0,
      "p99": 1546.0
    },
    "gpu_util_pct": {
      "count": 1022,
      "mean": 20.65753424657534,
      "min": 0.0,
      "max": 77.0,
      "p50": 16.0,
      "p90": 47.0,
      "p95": 54.0,
      "p99": 66.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "127.0.0.1:50611",
      "127.0.0.1:51296",
      "127.0.0.1:52232",
      "127.0.0.1:52687",
      "127.0.0.1:52723",
      "127.0.0.1:52760",
      "127.0.0.1:53162",
      "127.0.0.1:53379",
      "127.0.0.1:54845",
      "127.0.0.1:54872",
      "127.0.0.1:56403",
      "127.0.0.1:56433",
      "127.0.0.1:56545",
      "127.0.0.1:56546",
      "127.0.0.1:56547",
      "127.0.0.1:56548",
      "127.0.0.1:57968",
      "127.0.0.1:57993",
      "127.0.0.1:58924",
      "127.0.0.1:58957",
      "127.0.0.1:59246",
      "127.0.0.1:59272",
      "127.0.0.1:61043",
      "127.0.0.1:62120",
      "127.0.0.1:62251",
      "127.0.0.1:62440",
      "127.0.0.1:62466",
      "127.0.0.1:62718",
      "127.0.0.1:62793",
      "127.0.0.1:62817",
      "127.0.0.1:62840",
      "127.0.0.1:62958",
      "127.0.0.1:62993",
      "127.0.0.1:63019",
      "127.0.0.1:64440",
      "127.0.0.1:64525"
    ]
  }
}
```
