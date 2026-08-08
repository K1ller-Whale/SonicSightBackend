"""Markdown summary generator: one summary.md per run, citable by §10."""


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_summary(payload, path):
    meta = payload["meta"]
    lines = []
    a = lines.append
    a(f"# Load-suite run `{payload['run_id']}`")
    a("")
    a(f"- **Scenario:** {meta['scenario']}")
    a(f"- **Timestamp (UTC):** {meta['timestamp_utc']}")
    git = meta["git"]
    for repo, g in git.items():
        if g and g.get("head"):
            dirty = " (dirty)" if g.get("dirty") else ""
            a(f"- **{repo}:** `{g['head'][:12]}` on `{g.get('branch')}`{dirty}")
    a(f"- **GPU:** {meta.get('gpu') or 'nvidia-smi unavailable'}")
    host = meta["host"]
    a(f"- **Host:** {host['platform']} — Python {host['python']}, "
      f"torch {meta['torch']['version'] if meta.get('torch') else 'absent'}, "
      f"grpcio {meta.get('grpcio')}")
    y = meta.get("nfr_targets_yaml")
    if y:
        a(f"- **nfr_targets.yaml:** sha256:{y['sha256_16']} ({y['bytes']} B)")
    ck = meta.get("checkpoints") or {}
    if ck:
        a(f"- **Checkpoints:** " + ", ".join(
            f"{k} ({v['bytes']} B)" for k, v in ck.items() if v))
    a(f"- **Wall time:** {payload['wall_seconds']:.1f} s")
    a("")
    a("## Assertions (thresholds from docs/nfr/nfr_targets.yaml)")
    a("")
    a("| NFR | Metric | Stat | Target | Measured | Outcome |")
    a("|---|---|---|---|---|---|")
    for o in payload["assertions"]:
        if o.get("outcome") == "skipped":
            a(f"| {o['id']} | — | — | — | — | skipped: {o['reason']} |")
            continue
        a(f"| {o['id']} | {o.get('metric','—')} | {o.get('statistic','—')} | "
          f"{o.get('op','')} {_fmt(o.get('target'))} {o.get('unit') or ''} | "
          f"{_fmt(o.get('measured'))} | {o['outcome']} |")
    a("")
    a("## Measured metrics")
    a("")
    a("| Metric | Statistic | Value |")
    a("|---|---|---|")
    for metric, stats in sorted(payload["measured"].items()):
        if isinstance(stats, dict):
            for stat, v in stats.items():
                a(f"| {metric} | {stat} | {_fmt(v)} |")
    extras = payload.get("extras") or {}
    char = {k: v for k, v in extras.items() if isinstance(v, dict)}
    if char:
        a("")
        a("## Characterisation")
        a("")
        a("```json")
        import json
        a(json.dumps(char, indent=2, default=str)[:8000])
        a("```")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
