# Load-suite run `20260811T062334Z-smoke`

- **Scenario:** smoke
- **Timestamp (UTC):** 2026-08-11T06:23:53Z
- **SonicSightBackend:** `7cb179bb4064` on `em-verification` (dirty)
- **SonicSightMobile:** `09af71a599ea` on `em-verification`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:f00d1071195028c6 (38683 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 18.3 s

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
      "count": 17,
      "mean": 3272.5546875,
      "min": 3272.5546875,
      "max": 3272.5546875,
      "p50": 3272.5546875,
      "p90": 3272.5546875,
      "p95": 3272.5546875,
      "p99": 3272.5546875
    },
    "cpu_pct": {
      "count": 17,
      "mean": 0.6529411764705884,
      "min": 0.0,
      "max": 3.8,
      "p50": 0.0,
      "p90": 1.3000000000000003,
      "p95": 2.2799999999999985,
      "p99": 3.4959999999999996
    },
    "gpu_used_mib": {
      "count": 17,
      "mean": 2186.294117647059,
      "min": 1450.0,
      "max": 5839.0,
      "p50": 2242.0,
      "p90": 2340.6,
      "p95": 3157.399999999998,
      "p99": 5302.679999999999
    },
    "gpu_util_pct": {
      "count": 17,
      "mean": 5.882352941176471,
      "min": 0.0,
      "max": 33.0,
      "p50": 0.0,
      "p90": 27.800000000000004,
      "p95": 32.2,
      "p99": 32.84
    },
    "nvidia_smi_available": true,
    "remote_endpoints": []
  }
}
```
