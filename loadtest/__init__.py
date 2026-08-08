# SonicSight backend load/stress suite. See loadtest/README.md.
#
# Thresholds live in docs/nfr/nfr_targets.yaml (single source of truth,
# NFR-MAINT-004): no assertion threshold may appear as a literal in this
# package. Stimulus parameters (payload sizes, injection counts) are not
# thresholds and live in scenario args with a comment naming the FR they
# exercise.
