# Load-suite run `20260810T072548Z-smoke`

- **Scenario:** smoke
- **Timestamp (UTC):** 2026-08-10T07:25:58Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:98f7877bbe5216fc (17587 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 10.2 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-COMPAT-001 | differing_bytes_between_proto_copies | count | == 0 bytes | 0 | pass |
| NFR-COMPAT-002 | metadata_less_streams_running_sonicsight | proportion | >= 1.000 ratio | 1.000 | pass |
| NFR-COMPAT-003 | pytorch_modes_served_without_tf | count | >= 2 models | 2 | pass |
| NFR-COMPAT-003 | health_lists_exactly_loaded_models | boolean | == True bool | True | pass |
| NFR-MAINT-001 | unit_test_failures | count | == 0 count | 0 | pass |
| NFR-MAINT-001 | unit_test_collection_errors | count | == 0 count | 0 | pass |
| NFR-MAINT-004 | hardcoded_threshold_literals_in_loadtest | count |  —  | — | inspection |
| NFR-MAINT-004 | report_yaml_threshold_mismatches | count | == 0 mismatches | 0 | pass |
| NFR-FLEX-001 | models_served_on_e_d_without_tf | count | >= 2 models | 2 | pass |
| NFR-FLEX-001 | models_served_on_e_m | count | >= 3 models | 3 | pass |
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
| differing_bytes_between_proto_copies | count | 0 |
| health_lists_exactly_loaded_models | boolean | True |
| metadata_less_streams_running_sonicsight | proportion | 1.000 |
| models_served_on_e_d_without_tf | count | 2 |
| models_served_on_e_m | count | 3 |
| pytorch_modes_served_without_tf | count | 2 |
| report_yaml_threshold_mismatches | count | 0 |
| unit_test_collection_errors | count | 0 |
| unit_test_failures | count | 0 |

## Characterisation

```json
{
  "health": {
    "model_loaded": true,
    "device": "cuda",
    "loaded_models": [
      "sonicsight",
      "multisensory",
      "sonicsight-pixel"
    ]
  },
  "resources": {
    "rss_mib": {
      "count": 10,
      "mean": 2358.078125,
      "min": 2358.078125,
      "max": 2358.078125,
      "p50": 2358.078125,
      "p90": 2358.078125,
      "p95": 2358.078125,
      "p99": 2358.078125
    },
    "cpu_pct": {
      "count": 10,
      "mean": 0.39,
      "min": 0.0,
      "max": 1.0,
      "p50": 0.0,
      "p90": 1.0,
      "p95": 1.0,
      "p99": 1.0
    },
    "gpu_used_mib": {
      "count": 10,
      "mean": 1084.1,
      "min": 892.0,
      "max": 1495.0,
      "p50": 1049.5,
      "p90": 1253.8,
      "p95": 1374.3999999999996,
      "p99": 1470.88
    },
    "gpu_util_pct": {
      "count": 10,
      "mean": 13.8,
      "min": 2.0,
      "max": 61.0,
      "p50": 10.0,
      "p90": 18.69999999999998,
      "p95": 39.84999999999995,
      "p99": 56.77000000000001
    },
    "nvidia_smi_available": true,
    "remote_endpoints": []
  }
}
```
