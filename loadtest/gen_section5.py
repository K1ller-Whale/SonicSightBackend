"""Generate the ANALYSIS_REPORT.md skeleton whose section 5 table is
mechanically derived from docs/nfr/nfr_targets.yaml, so the smoke
scenario's report<->YAML matcher (NFR-MAINT-004) checks a generated table,
never a hand-typed one."""
import yaml, io, os, sys

root = os.path.expanduser("~/SonicSight")
doc = yaml.safe_load(open(os.path.join(root, "docs/nfr/nfr_targets.yaml")))

UNIT_TOK = {"ms": " ms", "s": " s", "MiB": " MiB", "MiB/min": " MiB/min",
            "percent": " %", "ratio": "", "sessions": " sessions",
            "correlation": "", "count": "", "bytes": " bytes",
            "models": " models", "endpoints": " endpoints",
            "mismatches": "", "bool": ""}

rows = []
for tid, t in doc["targets"].items():
    cells = []
    for a in t.get("assertions", []):
        v = a["value"]
        if isinstance(v, bool):
            cells.append(f"{a['metric']} {a['op']} {v}")
            continue
        stat = a.get("statistic", "")
        vtxt = f"{v:g}" if isinstance(v, float) else str(v)
        cells.append(f"{a['metric']} {a['op']} **{vtxt}**"
                     f"{UNIT_TOK.get(a.get('unit') or '', '')} ({stat})")
    scen = t.get("scenario")
    scen = ", ".join(scen) if isinstance(scen, list) else scen
    flags = []
    if t.get("e_m_conditional"):
        flags.append("E-M only")
    if t.get("matrix_conditional"):
        flags.append("device-matrix")
    if t.get("model"):
        flags.append(f"model={t['model']}")
    rows.append(f"| {tid} | {t.get('title','')} | "
                f"{'; '.join(cells) if cells else '—'} | {scen}"
                f"{(' — ' + ', '.join(flags)) if flags else ''} |")

table = "\n".join(rows)
out = f"""# SonicSight — Analysis report (RECONSTRUCTED SKELETON)

> **Provenance warning.** The Phase 1–3 analysis report was authored on the
> DESKTOP-RF4V9RH host and never committed (docs/ sits outside all three
> repositories). This file is a skeleton created 2026-08-10 so the load
> suite's report-to-YAML consistency check (NFR-MAINT-004) has a section 5
> to verify. Its section 5 table is GENERATED from docs/nfr/nfr_targets.yaml
> by loadtest scaffolding (gen_section5.py); no threshold below is
> hand-typed. Merge the original report over this skeleton when it is
> recovered, and re-run the smoke scenario to re-verify.

## 5. Non-functional requirements (thresholds generated from nfr_targets.yaml)

| ID | Title | Assertions (metric, bound, statistic) | Scenario |
|---|---|---|---|
{table}

## 6. Use case model

(Placeholder — the original section lives in the uncommitted report.)
"""
path = os.path.join(root, "docs/ANALYSIS_REPORT.md")
open(path, "w", encoding="utf-8").write(out)
print("wrote", path, len(out), "bytes,", len(rows), "target rows")
