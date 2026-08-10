# Load-suite run `20260810T072353Z-smoke`

- **Scenario:** smoke
- **Timestamp (UTC):** 2026-08-10T07:24:26Z
- **SonicSightBackend:** `59adf8a3a92d` on `main` (dirty)
- **SonicSightMobile:** `cfbe64f096c8` on `master`
- **GPU:** NVIDIA GeForce GTX 1660 Ti, 610.88, 6144 MiB
- **Host:** Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 — Python 3.12.13, torch 2.11.0+cu128, grpcio 1.83.0
- **nfr_targets.yaml:** sha256:98f7877bbe5216fc (17587 B)
- **Checkpoints:** .txt (0 B), frame_best.pth (45356134 B), sound_best.pth (121145339 B), synthesizer_best.pth (652 B)
- **Wall time:** 32.7 s

## Assertions (thresholds from docs/nfr/nfr_targets.yaml)

| NFR | Metric | Stat | Target | Measured | Outcome |
|---|---|---|---|---|---|
| NFR-COMPAT-001 | differing_bytes_between_proto_copies | count | == 0 bytes | 0 | pass |
| NFR-COMPAT-002 | metadata_less_streams_running_sonicsight | proportion | >= 1.000 ratio | — | not-measured |
| NFR-COMPAT-003 | pytorch_modes_served_without_tf | count | >= 2 models | — | not-measured |
| NFR-COMPAT-003 | health_lists_exactly_loaded_models | boolean | == True bool | — | not-measured |
| NFR-MAINT-001 | unit_test_failures | count | == 0 count | 0 | pass |
| NFR-MAINT-001 | unit_test_collection_errors | count | == 0 count | 2 | FAIL |
| NFR-MAINT-004 | hardcoded_threshold_literals_in_loadtest | count |  —  | — | inspection |
| NFR-MAINT-004 | report_yaml_threshold_mismatches | count | == 0 mismatches | 0 | pass |
| NFR-FLEX-001 | models_served_on_e_d_without_tf | count | >= 2 models | — | not-measured |
| NFR-FLEX-001 | — | — | — | — | skipped: assertion is e-m-conditional |
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
| report_yaml_threshold_mismatches | count | 0 |
| unit_test_collection_errors | count | 2 |
| unit_test_failures | count | 0 |
