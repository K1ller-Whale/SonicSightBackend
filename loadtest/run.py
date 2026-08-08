"""CLI entry for the load suite.

    python -m loadtest.run <scenario> [options]

Scenarios: baseline | load | stress | spike | soak | failure-injection |
switching | smoke. Every run writes results/<runid>/{results.json,
summary.md} with full run metadata; assertions come exclusively from
docs/nfr/nfr_targets.yaml.
"""

import argparse
import asyncio
import json
import os
import time

from . import driver, monitors, nfr, paths, report, runmeta, scenarios


def build_parser():
    ap = argparse.ArgumentParser(prog="loadtest.run", description=__doc__)
    ap.add_argument("scenario", choices=sorted(scenarios.SCENARIOS))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--model", default="sonicsight",
                    choices=sorted(driver.PROFILES))
    ap.add_argument("--environment", default="E-D",
                    help="reference-environment id for assertion selection")
    ap.add_argument("--duration", type=float, default=None,
                    help="steady/hold duration seconds; default per scenario "
                         "(baseline 420, load 720, soak 1800, spike 120)")
    ap.add_argument("--runs", type=int, default=20,
                    help="TTFR probe stream openings (baseline)")
    ap.add_argument("--ttfr-probe-seconds", type=float, default=15.0)
    ap.add_argument("--sessions", type=int, default=2, help="load concurrency")
    ap.add_argument("--stagger", type=float, default=1.0)
    ap.add_argument("--max-sessions", type=int, default=6, help="stress upper bound")
    ap.add_argument("--step-duration", type=float, default=420.0,
                    help="per-step hold; >= fill + the 5-min cadence window")
    ap.add_argument("--burst-sessions", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=100, help="switching cycles")
    ap.add_argument("--cycle-duration", type=float, default=10.0)
    ap.add_argument("--models", default="sonicsight,sonicsight-pixel")
    ap.add_argument("--injections", default="",
                    help="comma list; empty = all: " + ",".join(scenarios.INJECTIONS))
    ap.add_argument("--injection-duration", type=float, default=330.0)
    ap.add_argument("--disconnect-cycles", type=int, default=50)
    ap.add_argument("--disconnect-after", type=float, default=10.0)
    ap.add_argument("--settle-seconds", type=float, default=30.0)
    ap.add_argument("--video", default=None,
                    help="reference clip for the replay-sanity check (NFR-FUNC-001)")
    ap.add_argument("--replay-runs", type=int, default=3)
    ap.add_argument("--server-info", default=None,
                    help="server.json written by loadtest.server_wrapper")
    ap.add_argument("--server-log", default=None,
                    help="captured server stdout/stderr, for log-based metrics")
    ap.add_argument("--open-marker",
                    default=r"Client connected for StreamProcess",
                    help="regex for the server's stream-open log line")
    ap.add_argument("--close-marker", default=None,
                    help="regex for a stream-close log line; the server has "
                         "no such line today, so open/close parity is "
                         "not-measured unless one is provided")
    ap.add_argument("--offline", action="store_true",
                    help="smoke only: skip live-server probes")
    ap.add_argument("--out", default=None, help="results directory override")
    return ap


DURATION_DEFAULTS = {"baseline": 420.0, "load": 720.0, "soak": 1800.0,
                     "spike": 120.0}


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.duration is None:
        args.duration = DURATION_DEFAULTS.get(args.scenario, 420.0)
    doc = nfr.load_targets()

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + args.scenario
    out_dir = args.out or os.path.join(paths.RESULTS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    ctx = {"run_start_epoch": time.time()}
    server_info = monitors.read_server_info(args.server_info) if args.server_info else None
    ctx["server_info"] = server_info
    mon = None
    if server_info:
        mon = monitors.ResourceMonitor(server_pid=server_info["pid"]).start()
        ctx["monitor"] = mon

    if not args.offline and args.scenario != "smoke":
        health = asyncio.run(driver.health_check(args.host, args.port))
        if args.model not in health["loaded_models"]:
            raise SystemExit(
                f"model '{args.model}' not loaded on {args.host}:{args.port} "
                f"(loaded: {health['loaded_models']})")

    t0 = time.time()
    measured, extras = scenarios.SCENARIOS[args.scenario](args, doc, ctx)
    wall = time.time() - t0

    if mon is not None:
        mon.stop()
        extras["resources"] = mon.snapshot()

    context = {"model": args.model,
               "sessions": args.sessions if args.scenario == "load" else 1}
    outcomes = nfr.evaluate(doc, args.scenario, measured,
                            environment=args.environment, context=context)
    outcomes += nfr.coverage_gaps(doc, set(scenarios.SCENARIOS))
    payload = {
        "run_id": run_id,
        "meta": runmeta.collect(args.scenario, vars(args)),
        "wall_seconds": wall,
        "measured": measured,
        "assertions": outcomes,
        "extras": extras,
    }
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    summary_path = os.path.join(out_dir, "summary.md")
    report.write_summary(payload, summary_path)

    fails = [o for o in outcomes if o.get("outcome") == "FAIL"]
    not_measured = [o for o in outcomes if o.get("outcome") == "not-measured"]
    print(f"run {run_id}: {len(fails)} failing assertion(s), "
          f"{len(not_measured)} not-measured; results: {results_path}")
    for o in not_measured:
        print(f"  WARNING not-measured: {o['id']} {o.get('metric')}/"
              f"{o.get('statistic')} — a masked defect, not a pass")
    if fails:
        return 1
    if not_measured:
        return 2  # distinct exit code: nothing failed, but coverage gaps
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
