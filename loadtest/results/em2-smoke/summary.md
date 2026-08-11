# Load-suite run `20260810T131233Z-smoke`

- **Scenario:** smoke
- **Timestamp (UTC):** 2026-08-10T13:13:21Z
- **SonicSightBackend:** `f840ce16afdd` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:c1b14b0c73772c80 (36367 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 47.2 s

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
| NFR-FLEX-001 | models_served_on_e_m_class_with_tf | count | == 3 models | 3 | pass |
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
| differing_bytes_between_proto_copies | count | 0 |
| health_lists_exactly_loaded_models | boolean | True |
| metadata_less_streams_running_sonicsight | proportion | 1.000 |
| models_served_on_e_d_without_tf | count | 2 |
| models_served_on_e_m_class_with_tf | count | 3 |
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
      "count": 27,
      "mean": 3066.4327256944443,
      "min": 2983.14453125,
      "max": 3076.84375,
      "p50": 3076.84375,
      "p90": 3076.84375,
      "p95": 3076.84375,
      "p99": 3076.84375
    },
    "cpu_pct": {
      "count": 27,
      "mean": 0.4740740740740741,
      "min": 0.0,
      "max": 1.7,
      "p50": 0.5,
      "p90": 0.9,
      "p95": 0.97,
      "p99": 1.517999999999999
    },
    "gpu_used_mib": {
      "count": 27,
      "mean": 1893.4444444444443,
      "min": 1360.0,
      "max": 3270.0,
      "p50": 1376.0,
      "p90": 2573.0,
      "p95": 2735.6,
      "p99": 3148.8399999999992
    },
    "gpu_util_pct": {
      "count": 27,
      "mean": 9.555555555555555,
      "min": 1.0,
      "max": 52.0,
      "p50": 7.0,
      "p90": 15.400000000000023,
      "p95": 23.4,
      "p99": 44.719999999999956
    },
    "nvidia_smi_available": true,
    "remote_endpoints": []
  }
}
```
