# Load-suite run `20260810T105037Z-smoke`

- **Scenario:** smoke
- **Timestamp (UTC):** 2026-08-10T10:50:58Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:5954cce68a9e5b9a (20139 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 21.4 s

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
      "count": 19,
      "mean": 3211.156866776316,
      "min": 3025.4765625,
      "max": 3251.23828125,
      "p50": 3247.2734375,
      "p90": 3251.23828125,
      "p95": 3251.23828125,
      "p99": 3251.23828125
    },
    "cpu_pct": {
      "count": 19,
      "mean": 0.45789473684210524,
      "min": 0.0,
      "max": 1.9,
      "p50": 0.0,
      "p90": 1.1199999999999997,
      "p95": 1.6299999999999994,
      "p99": 1.846
    },
    "gpu_used_mib": {
      "count": 19,
      "mean": 2167.842105263158,
      "min": 1325.0,
      "max": 3451.0,
      "p50": 2662.0,
      "p90": 2711.0,
      "p95": 2961.3999999999987,
      "p99": 3353.08
    },
    "gpu_util_pct": {
      "count": 19,
      "mean": 3.1578947368421053,
      "min": 0.0,
      "max": 29.0,
      "p50": 1.0,
      "p90": 6.399999999999995,
      "p95": 13.699999999999964,
      "p99": 25.940000000000005
    },
    "nvidia_smi_available": true,
    "remote_endpoints": []
  }
}
```
