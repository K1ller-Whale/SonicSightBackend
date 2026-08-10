# Load-suite run `20260808T153356Z-stress`

- **Scenario:** stress
- **Timestamp (UTC):** 2026-08-08T15:40:57Z
- **SonicSightBackend:** `1a1842a8d159` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1650, 595.97, 4096 MiB
- **Host:** Windows-10-10.0.26200-SP0 — Python 3.11.9, torch 2.6.0+cu124, grpcio 1.78.0
- **nfr_targets.yaml:** sha256:afe89a851eb00392 (25614 B)
- **Checkpoints:** frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 420.2 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-007 | sessions_meeting_perf_003_and_004 | count | >= 2 sessions | 0 | FAIL |
| NFR-PERF-007 | concurrency_ceiling | report | report —  | {'first_breaching_concurrency': 1, 'note': 'None = no breach up to --max-sessions'} | characterisation |
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
| concurrency_ceiling | first_breaching_concurrency | 1 |
| concurrency_ceiling | note | None = no breach up to --max-sessions |
| event_loop_lag | p99 | 28.000 |
| event_loop_lag | max | 153.000 |
| sessions_meeting_perf_003_and_004 | count | 0 |

## Characterisation

```json
{
  "steps": {
    "1": {
      "sessions_ok": 0,
      "of": 1,
      "time_to_first_nonbuffering_result": {
        "p95": 6.116283499635756
      },
      "inter_result_interval_abs_dev_from_125ms": {
        "p95": 60.62162207105869
      },
      "total_server_time_ms": {
        "p95": 89.0,
        "max": 1476
      },
      "sequence_gap_rate": {
        "proportion": 0.0
      }
    }
  },
  "resources": {
    "rss_mib": {
      "count": 383,
      "mean": 111.00010199086162,
      "min": 104.2890625,
      "max": 123.75,
      "p50": 108.24609375,
      "p90": 112.41796875,
      "p95": 123.7265625,
      "p99": 123.75
    },
    "cpu_pct": {
      "count": 383,
      "mean": 451.79634464751956,
      "min": 0.0,
      "max": 549.3,
      "p50": 486.0,
      "p90": 528.16,
      "p95": 531.29,
      "p99": 540.026
    },
    "gpu_used_mib": {
      "count": 383,
      "mean": 1554.4699738903394,
      "min": 1546.0,
      "max": 1580.0,
      "p50": 1554.0,
      "p90": 1558.0,
      "p95": 1562.0,
      "p99": 1573.0
    },
    "gpu_util_pct": {
      "count": 383,
      "mean": 31.966057441253263,
      "min": 0.0,
      "max": 58.0,
      "p50": 32.0,
      "p90": 35.0,
      "p95": 36.0,
      "p99": 46.0
    },
    "nvidia_smi_available": true
  }
}
```
