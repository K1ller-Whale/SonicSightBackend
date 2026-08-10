# Load-suite run `20260808T142525Z-smoke`

- **Scenario:** smoke
- **Timestamp (UTC):** 2026-08-08T14:25:52Z
- **SonicSightBackend:** `1a1842a8d159` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1650, 595.97, 4096 MiB
- **Host:** Windows-10-10.0.26200-SP0 — Python 3.11.9, torch 2.6.0+cu124, grpcio 1.78.0
- **nfr_targets.yaml:** sha256:b43885f80c4f1c08 (25497 B)
- **Checkpoints:** frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 26.4 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-COMPAT-001 | differing_bytes_between_proto_copies | count | == 0 bytes | 0 | pass |
| NFR-COMPAT-002 | metadata_less_streams_running_sonicsight | proportion | >= 1.000 ratio | 1.000 | pass |
| NFR-COMPAT-003 | pytorch_modes_served_without_tf | count | >= 2 models | 2 | pass |
| NFR-COMPAT-003 | health_lists_exactly_loaded_models | boolean | == True bool | True | pass |
| NFR-MAINT-001 | unit_test_failures | count | == 0 count | 0 | pass |
| NFR-MAINT-001 | unit_test_collection_errors | count | == 0 count | 2 | FAIL |
| NFR-MAINT-004 | hardcoded_threshold_literals_in_loadtest | count |  —  | — | inspection |
| NFR-MAINT-004 | report_yaml_threshold_mismatches | count | == 0 mismatches | 0 | pass |
| NFR-FLEX-001 | models_served_on_e_d_without_tf | count | >= 2 models | 2 | pass |
| NFR-FLEX-001 | — | — | — | — | skipped: assertion is e-m-conditional |
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
| pytorch_modes_served_without_tf | count | 2 |
| report_yaml_threshold_mismatches | count | 0 |
| unit_test_collection_errors | count | 2 |
| unit_test_failures | count | 0 |

## Characterisation

```json
{
  "health": {
    "model_loaded": true,
    "device": "cuda",
    "loaded_models": [
      "sonicsight",
      "sonicsight-pixel"
    ]
  },
  "resources": {
    "rss_mib": {
      "count": 25,
      "mean": 1015.645,
      "min": 894.015625,
      "max": 1055.69140625,
      "p50": 1054.69140625,
      "p90": 1055.29140625,
      "p95": 1055.69140625,
      "p99": 1055.69140625
    },
    "cpu_pct": {
      "count": 25,
      "mean": 0.5760000000000001,
      "min": 0.0,
      "max": 4.4,
      "p50": 0.0,
      "p90": 1.4600000000000002,
      "p95": 1.5,
      "p99": 3.7039999999999944
    },
    "gpu_used_mib": {
      "count": 25,
      "mean": 1650.48,
      "min": 1286.0,
      "max": 3721.0,
      "p50": 1520.0,
      "p90": 1592.4,
      "p95": 2181.999999999998,
      "p99": 3386.199999999997
    },
    "gpu_util_pct": {
      "count": 25,
      "mean": 9.8,
      "min": 2.0,
      "max": 26.0,
      "p50": 10.0,
      "p90": 12.600000000000001,
      "p95": 23.399999999999963,
      "p99": 26.0
    },
    "nvidia_smi_available": true
  }
}
```
