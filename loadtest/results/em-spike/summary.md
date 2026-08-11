# Load-suite run `20260810T085512Z-spike`

- **Scenario:** spike
- **Timestamp (UTC):** 2026-08-10T08:57:57Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:e2c1c712b035b6ac (17981 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 165.0 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-PERF-014 | sessions_surviving_burst | count | >= 3 sessions | 4 | pass |
| NFR-PERF-014 | time_to_first_compliant_10s_window | max_over_survivors | <= 30 s | 11.848 | pass |
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
| event_loop_lag | p99 | 2.087 |
| event_loop_lag | max | 2.975 |
| sessions_surviving_burst | count | 4 |
| time_to_first_compliant_10s_window | max_over_survivors | 11.848 |

## Characterisation

```json
{
  "time_to_first_nonbuffering_s": {
    "count": 4,
    "mean": 6.190253758750714,
    "min": 6.040775064000627,
    "max": 6.2881736180006556,
    "p50": 6.216033176500787,
    "p90": 6.273728939300781,
    "p95": 6.2809512786507185,
    "p99": 6.286729150130668
  },
  "inter_result_interval_ms": {
    "count": 3652,
    "mean": 173.77707183844495,
    "min": 1.8464589993527625,
    "max": 2316.6173089994118,
    "p50": 166.71089999999822,
    "p90": 176.02099229970918,
    "p95": 180.18424255014907,
    "p99": 190.25305188963097
  },
  "cadence_abs_dev_ms": {
    "count": 3652,
    "mean": 49.05688075704957,
    "min": 32.0737128697408,
    "max": 2191.6286468678927,
    "p50": 41.7318458687339,
    "p90": 51.07384476780567,
    "p95": 55.305354367491184,
    "p99": 65.93978319869046
  },
  "total_server_time_ms": {
    "count": 3652,
    "mean": 170.89430449069005,
    "min": 154.0,
    "max": 2314.0,
    "p50": 164.0,
    "p90": 173.0,
    "p95": 176.44999999999982,
    "p99": 187.0
  },
  "audio_age_lag_ms": {
    "count": 3656,
    "mean": 28938.057608125,
    "min": 3289.662049000981,
    "max": 48158.099524000136,
    "p50": 29230.201467500592,
    "p90": 44408.99839099984,
    "p95": 46245.165435749645,
    "p99": 47775.22125625009
  },
  "query_rtt_ms": {
    "count": 0
  },
  "resources": {
    "rss_mib": {
      "count": 157,
      "mean": 3156.597755772293,
      "min": 3079.9453125,
      "max": 3164.4609375,
      "p50": 3160.64453125,
      "p90": 3163.921875,
      "p95": 3163.94921875,
      "p99": 3164.328125
    },
    "cpu_pct": {
      "count": 157,
      "mean": 108.45095541401275,
      "min": 0.0,
      "max": 118.8,
      "p50": 112.9,
      "p90": 116.4,
      "p95": 117.11999999999999,
      "p99": 118.344
    },
    "gpu_used_mib": {
      "count": 157,
      "mean": 2495.140127388535,
      "min": 2129.0,
      "max": 3122.0,
      "p50": 2511.0,
      "p90": 2515.0,
      "p95": 2515.0,
      "p99": 2515.0
    },
    "gpu_util_pct": {
      "count": 157,
      "mean": 55.24203821656051,
      "min": 1.0,
      "max": 66.0,
      "p50": 57.0,
      "p90": 58.0,
      "p95": 59.0,
      "p99": 65.0
    },
    "nvidia_smi_available": true,
    "remote_endpoints": [
      "::ffff:127.0.0.1:39498"
    ]
  }
}
```
