"""Scenario implementations. Each is independently runnable via run.py and
parameterised; each returns (measured, extras) where `measured` maps
nfr_targets.yaml metric names to {statistic: value} and `extras` carries
characterisation data for the report.

No assertion threshold appears here as a literal (NFR-MAINT-004): wherever a
scenario needs a bound or a measurement-window definition, it reads it from
the YAML through helpers in this module. Stimulus parameters (payload sizes,
cycle counts) are scenario arguments, not thresholds.
"""

import asyncio
import math
import os
import re
import subprocess
import sys
import time

from . import driver, media, metrics, monitors, nfr, paths


# ── helpers ────────────────────────────────────────────────────────────

def _assertion_value(doc, target_id, metric, statistic):
    """Read one threshold from the YAML (the only sanctioned source)."""
    for a in doc["targets"][target_id]["assertions"]:
        if a["metric"] == metric and a["statistic"] == statistic:
            return a["value"]
    raise KeyError(f"{target_id} has no assertion {metric}/{statistic}")


def _target_field(doc, target_id, field, default=None):
    """Read a condition field (duration_min, cycles, ...) from a target."""
    return doc["targets"][target_id].get(field, default)


def _session_meets_cadence(doc, stats):
    dev_p95 = metrics.percentile(stats.cadence_abs_dev_ms(), 95)
    if dev_p95 is None:
        return False
    return dev_p95 <= _assertion_value(
        doc, "NFR-PERF-003", "inter_result_interval_abs_dev_from_125ms", "p95")


def _session_meets_cadence_and_cycle(doc, stats):
    """Per-session NFR-PERF-003 + NFR-PERF-004 check, thresholds from YAML."""
    srv = stats.server_times_ms()
    srv_p95 = metrics.percentile(srv, 95)
    if srv_p95 is None:
        return False
    return (_session_meets_cadence(doc, stats)
            and srv_p95 <= _assertion_value(
                doc, "NFR-PERF-004", "total_server_time_ms", "p95")
            and max(srv) <= _assertion_value(
                doc, "NFR-PERF-004", "total_server_time_ms", "max"))


GAP_WINDOW_S = 300.0  # NFR-PERF-013's "consecutive non-overlapping 5-min
# windows" definition (yaml `window:` prose); duplicated here as a
# measurement definition, not a threshold.


def _gap_rate_windows(stats_list, window_s=GAP_WINDOW_S):
    """Max sequence-gap rate over consecutive non-overlapping windows,
    post-first-nonbuffering-result, per session (NFR-PERF-013 definition).
    Gap counting uses strictly increasing sequence pairs only (halves
    numbers from 0; the flush result is unnumbered)."""
    worst = 0.0
    any_window = False
    for st in stats_list:
        if st.t_first_nonbuffering is None or not st.results:
            continue
        t0 = st.t_first_nonbuffering
        t_end = st.results[-1]["t"]
        w = t0
        while w + window_s <= t_end:
            rows = [r for r in st.results if w <= r["t"] < w + window_s]
            seqs = [r["seq"] for r in rows]
            gaps = 0
            pairs = 0
            for a, b in zip(seqs, seqs[1:]):
                if b > a:
                    pairs += 1
                    gaps += b - a - 1
            if pairs:
                worst = max(worst, gaps / float(pairs + 1 + gaps))
                any_window = True
            w += window_s
    return worst if any_window else None


def _loop_lag_measured(ctx):
    """NFR-PERF-006 samples windowed to THIS run's wall-clock span, so a
    long-lived wrapper's earlier scenarios cannot contaminate the
    assertion (the wrapper stamps epoch time into loop_lag.csv)."""
    info = ctx.get("server_info")
    if not info:
        return {}
    lag = monitors.read_loop_lag_csv(info.get("loop_lag_csv"))
    t0 = ctx.get("run_start_epoch")
    vals = lag.window(t0, time.time()) if t0 else lag.v
    b = metrics.stats_bundle(vals)
    if b.get("count", 0) == 0:
        return {}
    return {"event_loop_lag": {"p99": b["p99"], "max": b["max"]}}


def _aggregate_single(stats_list, doc):
    """Common single-model aggregation onto YAML metric names."""
    ttfr = [st.time_to_first_nonbuffering() for st in stats_list
            if st.time_to_first_nonbuffering() is not None]
    devs = [d for st in stats_list for d in st.cadence_abs_dev_ms()]
    srv = [v for st in stats_list for v in st.server_times_ms()]
    measured = {}
    if ttfr:
        measured["time_to_first_nonbuffering_result"] = {
            "p95": metrics.percentile(ttfr, 95)}
    if devs:
        measured["inter_result_interval_abs_dev_from_125ms"] = {
            "p95": metrics.percentile(devs, 95)}
    if srv:
        measured["total_server_time_ms"] = {
            "p95": metrics.percentile(srv, 95), "max": max(srv)}
    gap = _gap_rate_windows(stats_list)
    if gap is not None:
        measured["sequence_gap_rate"] = {"proportion": gap}
    return measured


def _characterise(stats_list):
    """Full p50/p90/p95/p99/max bundles for the report (§10 metric list)."""
    def bundle(vals):
        return metrics.stats_bundle(vals)
    return {
        "time_to_first_nonbuffering_s": bundle(
            [st.time_to_first_nonbuffering() for st in stats_list
             if st.time_to_first_nonbuffering() is not None]),
        "inter_result_interval_ms": bundle(
            [iv for st in stats_list for iv in st.inter_result_intervals_ms()]),
        "cadence_abs_dev_ms": bundle(
            [d for st in stats_list for d in st.cadence_abs_dev_ms()]),
        "total_server_time_ms": bundle(
            [v for st in stats_list for v in st.server_times_ms()]),
        "audio_age_lag_ms": bundle(
            [v for st in stats_list for v in st.audio_ages_ms()]),
        "query_rtt_ms": bundle(
            [v for st in stats_list for v in st.query_rtts_ms]),
        "results_per_session": [len(st.results) for st in stats_list],
        "errors": [st.error for st in stats_list if st.error],
        "error_results": sum(st.error_results for st in stats_list),
        "buffering_results": sum(st.buffering_count for st in stats_list),
    }


def _make_source(model_id):
    prof = driver.PROFILES[model_id]
    def make(i):
        return media.SyntheticSource(
            seed=i, sample_rate=prof["sample_rate"],
            frame_rate=prof["frame_rate"], frame_kind=prof["frame_kind"])
    return make


def _mon_now(mon):
    return time.monotonic() - mon._t0 if mon else None


def _wait_for_sample(mon, timeout_s=5.0):
    """The monitor thread samples at 1 Hz; wait for a first reading so a
    baseline resource point is never taken from an empty series."""
    t0 = time.time()
    while mon and not mon.rss_mib.v and time.time() - t0 < timeout_s:
        time.sleep(0.2)


def _resource_point(mon):
    rss = mon.rss_mib.v[-1] if mon and mon.rss_mib.v else None
    gpu = mon.gpu_used_mib.v[-1] if mon and mon.gpu_used_mib.v else None
    return rss, gpu


# ── scenarios ──────────────────────────────────────────────────────────

def baseline(args, doc, ctx):
    """1 session: TTFR probe runs + one steady session (+ queries in pixel
    mode; + replay-harness sanity when --video is supplied)."""
    model = args.model
    make = _make_source(model)
    qps = 1.0 if model == "sonicsight-pixel" else 0.0

    async def go():
        ttfr_stats = []
        for i in range(args.runs):
            st = await driver.run_session(
                args.host, args.port, model, make(i),
                duration_s=args.ttfr_probe_seconds, session_id=i)
            ttfr_stats.append(st)
        steady = await driver.run_session(
            args.host, args.port, model, make(999),
            duration_s=args.duration, session_id=999, queries_per_s=qps)
        return ttfr_stats, steady

    ttfr_stats, steady = asyncio.run(go())
    measured = _aggregate_single([steady], doc)
    ttfr_only = _aggregate_single(ttfr_stats, doc)
    if "time_to_first_nonbuffering_result" in ttfr_only:
        measured["time_to_first_nonbuffering_result"] = \
            ttfr_only["time_to_first_nonbuffering_result"]
    if steady.query_rtts_ms:
        measured["query_round_trip"] = {
            "p95": metrics.percentile(steady.query_rtts_ms, 95)}
    measured.update(_loop_lag_measured(ctx))

    extras = _characterise(ttfr_stats + [steady])
    if args.video:
        pearson, warn = _replay_pearson(args)
        extras["replay_pearson_runs"] = pearson
        if warn:
            extras["replay_warning"] = warn
        if pearson:
            measured["replay_left_right_pearson"] = {
                "max_over_runs": max(pearson)}
    return measured, extras


def _replay_pearson(args):
    """Run replay_client.py N times, parse the L/R pearson from its report
    line '  pearson(L,R)    : %+.6f' (NFR-FUNC-001; the clip is
    operator-supplied — its SHA-256 lands in run metadata)."""
    vals, warn = [], None
    for i in range(args.replay_runs):
        out_dir = os.path.join(paths.RESULTS_DIR, "replay", f"run{i}")
        os.makedirs(out_dir, exist_ok=True)
        try:
            subprocess.run(
                [sys.executable,
                 os.path.join(paths.BACKEND_ROOT, "replay_client.py"),
                 "--video", args.video, "--host", f"{args.host}:{args.port}",
                 "--out", out_dir],
                cwd=paths.BACKEND_ROOT, timeout=1800)
        except subprocess.TimeoutExpired:
            warn = f"replay run {i} timed out"
            continue
        report = os.path.join(out_dir, "replay_report.txt")
        if os.path.exists(report):
            text = open(report, "r", encoding="utf-8", errors="replace").read()
            m = re.search(r"pearson\s*\(L,\s*R\)\s*:\s*([-+0-9.eE]+)", text)
            if m:
                vals.append(float(m.group(1)))
    if not vals and warn is None:
        warn = "no pearson value parsed from any replay report"
    return vals, warn


def load(args, doc, ctx):
    """Ramp to N sessions, hold; NFR-PERF-007 floor + GPU steady ceiling."""
    make = _make_source(args.model)
    mon = ctx.get("monitor")
    t_start = _mon_now(mon)

    stats = asyncio.run(driver.run_many(
        args.host, args.port, args.model, make, args.sessions,
        duration_s=args.duration, stagger_s=args.stagger))

    measured = _aggregate_single(stats, doc)
    ok = sum(1 for st in stats if _session_meets_cadence_and_cycle(doc, st))
    measured["sessions_meeting_perf_003_and_004"] = {"count": ok}
    if mon is not None:
        # steady state: after the first 5 min of the session (yaml
        # steady_state prose), on the monitor axis from session start.
        steady = mon.gpu_used_mib.window((t_start or 0) + 300.0, float("inf"))
        if steady:
            measured["gpu_memory_used"] = {"max_over_steady_state": max(steady)}
    measured.update(_loop_lag_measured(ctx))
    return measured, _characterise(stats)


def stress(args, doc, ctx):
    """Step concurrency upward until a per-step threshold breach: the knee.
    Reports the ceiling as characterisation (NFR-PERF-007 report metric);
    the PERF-007 floor count comes from the highest completed step at or
    below 2 sessions (FAILing honestly when even n=1 breaches)."""
    make = _make_source(args.model)
    ceiling = None
    steps = {}
    floor_count = 0
    for n in range(1, args.max_sessions + 1):
        stats = asyncio.run(driver.run_many(
            args.host, args.port, args.model, make, n,
            duration_s=args.step_duration, stagger_s=args.stagger))
        ok = sum(1 for st in stats if _session_meets_cadence_and_cycle(doc, st))
        steps[n] = {"sessions_ok": ok, "of": n,
                    **_aggregate_single(stats, doc)}
        if n <= 2:
            floor_count = ok
        if ok < n:
            ceiling = n
            break
    measured = {
        "concurrency_ceiling": {
            "first_breaching_concurrency": ceiling,
            "note": "None = no breach up to --max-sessions"},
        "sessions_meeting_perf_003_and_004": {"count": floor_count},
    }
    measured.update(_loop_lag_measured(ctx))
    return measured, {"steps": steps}


def spike(args, doc, ctx):
    """Burst simultaneous opens on an idle server (NFR-PERF-014). A
    survivor that never reaches a compliant window scores +inf so the
    30 s bound FAILs rather than vanishing."""
    make = _make_source(args.model)
    cadence_bound = _assertion_value(
        doc, "NFR-PERF-003", "inter_result_interval_abs_dev_from_125ms", "p95")

    stats = asyncio.run(driver.run_many(
        args.host, args.port, args.model, make, args.burst_sessions,
        duration_s=args.duration, stagger_s=0.0))

    survivors = [st for st in stats if st.error is None and st.results]
    times = []
    for st in survivors:
        t = _first_compliant_window(st, cadence_bound, window_s=10.0)
        times.append(t if t is not None else float("inf"))
    measured = {"sessions_surviving_burst": {"count": len(survivors)}}
    if times:
        measured["time_to_first_compliant_10s_window"] = {
            "max_over_survivors": max(times)}
    measured.update(_loop_lag_measured(ctx))
    return measured, _characterise(stats)


def _first_compliant_window(st, cadence_bound_ms, window_s):
    """Seconds from burst (t_open) to the start of the first sliding
    window whose p95 |interval - hop| is within bound. A window must be
    fully inside the stream and reasonably covered (>= 60 % of the results
    its hop implies), so 3 tight results before 9 s of silence do not
    qualify."""
    if not st.results:
        return None
    hop = st.profile["hop_ms"]
    min_results = max(3, int(math.ceil(0.6 * window_s * 1000.0 / hop)))
    rows = st.results
    t_last = rows[-1]["t"]
    for i in range(len(rows)):
        w0 = rows[i]["t"]
        if w0 + window_s > t_last:
            break
        wts = [r["t"] for r in rows if w0 <= r["t"] < w0 + window_s]
        if len(wts) < min_results:
            continue
        devs = [abs((b - a) * 1000.0 - hop) for a, b in zip(wts, wts[1:])]
        if metrics.percentile(devs, 95) <= cadence_bound_ms:
            return w0 - st.t_open
    return None


def soak(args, doc, ctx):
    """Sustained single session; leak detection (NFR-PERF-011/012, REL-005).
    The RSS-slope metric is emitted only when the run satisfies the YAML's
    duration condition; splice counts need the server's stream-dump log
    (see README)."""
    make = _make_source(args.model)
    mon = ctx.get("monitor")
    t_start = _mon_now(mon)

    stats = asyncio.run(driver.run_session(
        args.host, args.port, args.model, make(0),
        duration_s=args.duration, session_id=0))
    t_end = _mon_now(mon)

    measured = _aggregate_single([stats], doc)
    server_alive = asyncio.run(_alive(args))
    measured["crashes"] = {"count": 0 if server_alive else 1}
    extras = _characterise([stats])
    extras["server_alive_after"] = server_alive
    if stats.error:
        extras["stream_error"] = stats.error

    if mon is not None and t_end is not None:
        min_minutes = _target_field(doc, "NFR-PERF-011", "duration_min", 30)
        if args.duration >= min_minutes * 60:
            tail = [(t, v) for t, v in zip(mon.rss_mib.t, mon.rss_mib.v)
                    if t_end - 1200.0 <= t <= t_end]
            if len(tail) >= 2:
                measured["rss_slope_final_20min"] = {
                    "least_squares_slope": metrics.least_squares_slope(
                        [t for t, _ in tail], [v for _, v in tail])}
        else:
            extras["rss_slope_note"] = (
                f"duration {args.duration}s below the YAML condition "
                f"({min_minutes} min); NFR-PERF-011 not evaluated")
        warm = mon.gpu_used_mib.window((t_start or 0) + 300.0, t_end)
        if warm:
            measured["gpu_memory_growth_after_5min"] = {
                "max": max(warm) - min(warm)}
    if args.server_log and os.path.exists(args.server_log):
        text = open(args.server_log, "r", encoding="utf-8",
                    errors="replace").read()
        measured["unhandled_exceptions_in_logs"] = {
            "count": len(re.findall(r"Traceback \(most recent call last\)", text))}
        splice = re.findall(r"ola_splice_fallbacks'?\s*[=:]\s*(\d+)", text)
        if splice and stats.results:
            measured["splice_fallback_rate"] = {
                "proportion": int(splice[-1]) / float(len(stats.results))}
    measured.update(_loop_lag_measured(ctx))
    return measured, extras


# ── failure injections ─────────────────────────────────────────────────
# Mutators receive (key, payload) where key is an int tick for audio chunks
# and a ("frame", tick) tuple for frame dicts. Every injector must handle
# both shapes; _audio_only/_frame_only enforce that. Sizes/indices are
# STIMULI exercising FR-014/FR-011/FR-003, not thresholds.

def _audio_only(fn):
    def wrapped(key, payload):
        if isinstance(key, tuple):
            return payload
        return fn(key, payload)
    return wrapped


def _frame_only(fn):
    def wrapped(key, payload):
        if not isinstance(key, tuple):
            return payload
        return fn(key, payload)
    return wrapped


@_audio_only
def _inject_truncated_pcm(i, chunk):
    if i == 40 and chunk.audio_pcm:
        chunk.audio_pcm = chunk.audio_pcm[:101]  # odd length: not int16-aligned
    return chunk


@_frame_only
def _inject_bad_jpeg(key, fr):
    if key[1] >= 40:
        return {k: b"\xff\xd8NOT-A-JPEG" for k in fr}
    return fr


@_audio_only
def _inject_zero_len(i, chunk):
    if i == 40:
        chunk.audio_pcm = b""
    return chunk


@_audio_only
def _inject_wrong_rate_field(i, chunk):
    if i % 7 == 0 and chunk.audio_pcm:
        # halves-mode audio sent on the 22050 Hz field: wrong field+rate
        chunk.audio_pcm_hi = chunk.audio_pcm
        chunk.audio_pcm = b""
    return chunk


@_audio_only
def _inject_silent(i, chunk):
    if chunk.audio_pcm:
        chunk.audio_pcm = b"\x00" * len(chunk.audio_pcm)
    return chunk


@_audio_only
def _inject_clipping(i, chunk):
    if chunk.audio_pcm:
        n = len(chunk.audio_pcm) // 2
        chunk.audio_pcm = (b"\xff\x7f\x00\x80" * ((n // 2) + 1))[:n * 2]
    return chunk


@_frame_only
def _inject_black_frames(key, fr):
    from PIL import Image
    import io as _io
    from config import IMG_SIZE
    buf = _io.BytesIO()
    Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0)).save(
        buf, "JPEG", quality=90)
    return {k: buf.getvalue() for k in fr}


# NFR-REL-001's malformed set vs NFR-REL-004's degenerate-media set: the
# two requirements assert different metrics over different stimulus classes.
MALFORMED = ("truncated_pcm", "bad_jpeg", "zero_len_audio")
DEGENERATE = ("wrong_rate_field", "silent_audio", "clipping_audio",
              "black_frames")

INJECTIONS = {
    "truncated_pcm": _inject_truncated_pcm,
    "bad_jpeg": _inject_bad_jpeg,
    "zero_len_audio": _inject_zero_len,
    "wrong_rate_field": _inject_wrong_rate_field,
    "silent_audio": _inject_silent,
    "clipping_audio": _inject_clipping,
    "black_frames": _inject_black_frames,
}


def failure(args, doc, ctx):
    """Failure injection: malformed input (REL-001) and degenerate media
    (REL-004) against a co-running healthy session, oversized payload
    (REL-003), and abrupt disconnect cycles (REL-002)."""
    make = _make_source(args.model)
    mon = ctx.get("monitor")
    measured, extras = {}, {"injections": {}}

    async def injected_pair(fn):
        healthy_task = driver.run_session(
            args.host, args.port, args.model, make(1),
            duration_s=args.injection_duration, session_id=1)
        victim_task = driver.run_session(
            args.host, args.port, args.model, make(2),
            duration_s=args.injection_duration, session_id=2, inject=fn)
        return await asyncio.gather(healthy_task, victim_task)

    names = args.injections.split(",") if args.injections else list(INJECTIONS)
    outcomes = {}
    degenerate_srv = []
    healthy_ok = True
    for name in names:
        healthy, victim = asyncio.run(injected_pair(INJECTIONS[name]))
        server_alive = asyncio.run(_alive(args))
        contained = server_alive and bool(victim.error is not None
                                          or victim.results)
        outcomes[name] = contained
        if name in DEGENERATE:
            degenerate_srv += victim.server_times_ms()
        # REL-001's containment condition: the healthy session's cadence
        # (NFR-PERF-003 only, per its definition) holds during injection.
        if not _session_meets_cadence(doc, healthy):
            healthy_ok = False
        extras["injections"][name] = {
            "victim_error": victim.error,
            "victim_results": len(victim.results),
            "healthy_results": len(healthy.results),
            "server_alive_after": server_alive,
        }

    mal = [n for n in names if n in MALFORMED]
    deg = [n for n in names if n in DEGENERATE]
    if mal:
        measured["injected_cases_contained"] = {
            "proportion": sum(outcomes[n] for n in mal) / float(len(mal))}
        measured["healthy_session_perf_003_holds_during_injection"] = {
            "boolean": healthy_ok}
    if deg:
        measured["injections_with_defined_behaviour"] = {
            "proportion": sum(outcomes[n] for n in deg) / float(len(deg))}
        if degenerate_srv:
            measured["total_server_time_ms_during_injection"] = {
                "p95": metrics.percentile(degenerate_srv, 95)}

    # oversized payload: stimulus is one message over the 16 MB transport
    # cap (FR-014); expectation read from NFR-REL-003.
    oversized_ok, next_ok = asyncio.run(_oversized_probe(args))
    measured["oversized_rejected_at_transport"] = {
        "proportion": 1.0 if oversized_ok else 0.0}
    measured["next_session_succeeds"] = {"boolean": next_ok}

    # disconnect cycles (REL-002). Baseline resource point after a warmup
    # session + settle, so first-use allocator growth is not booked as
    # leakage; log parity needs only --server-log.
    if args.disconnect_cycles > 0:
        if mon is not None:
            asyncio.run(driver.run_session(
                args.host, args.port, args.model, make(0),
                duration_s=30.0, session_id=0))
            time.sleep(args.settle_seconds)
            _wait_for_sample(mon)
            rss0, gpu0 = _resource_point(mon)
        else:
            rss0 = gpu0 = None

        async def cycles():
            for i in range(args.disconnect_cycles):
                await driver.run_session(
                    args.host, args.port, args.model, make(i),
                    duration_s=30.0, session_id=i,
                    abrupt_cancel_after_s=args.disconnect_after)
        asyncio.run(cycles())
        time.sleep(args.settle_seconds)
        rss1, gpu1 = _resource_point(mon)
        if rss0 is not None and rss1 is not None:
            measured["rss_delta_vs_baseline"] = {
                "max": 100.0 * (rss1 - rss0) / rss0}
        if gpu0 is not None and gpu1 is not None:
            measured["gpu_memory_delta_vs_baseline"] = {"max": gpu1 - gpu0}
        if args.server_log and os.path.exists(args.server_log):
            text = open(args.server_log, encoding="utf-8",
                        errors="replace").read()
            opens = len(re.findall(args.open_marker, text))
            closes = (len(re.findall(args.close_marker, text))
                      if args.close_marker else None)
            extras["stream_open_lines"] = opens
            if closes is not None and opens >= args.disconnect_cycles:
                measured["stream_open_minus_close_log_lines"] = {
                    "count": opens - closes}
            else:
                extras["open_close_parity_note"] = (
                    "not evaluated: no close-side log line exists in the "
                    "server today (recorded limitation), or the open marker "
                    "matched fewer lines than cycles ran")
    measured.update(_loop_lag_measured(ctx))
    return measured, extras


async def _alive(args):
    try:
        await driver.health_check(args.host, args.port)
        return True
    except Exception:
        return False


async def _oversized_probe(args):
    """Send one >16 MB message; expect transport rejection, then a healthy
    short session (NFR-REL-003)."""
    import grpc
    pb2, pb2_grpc = paths.import_stubs()
    channel = grpc.aio.insecure_channel(f"{args.host}:{args.port}")
    stub = pb2_grpc.SonicSightServiceStub(channel)

    async def gen():
        yield pb2.StreamChunk(timestamp_ms=0,
                              audio_pcm=b"\x00" * (17 * 1024 * 1024))
        yield pb2.StreamChunk(timestamp_ms=1, is_last=True)

    rejected = False
    try:
        async for _ in stub.StreamProcess(gen()):
            pass
    except grpc.aio.AioRpcError:
        rejected = True
    finally:
        await channel.close()

    make = _make_source(args.model)
    st = await driver.run_session(args.host, args.port, args.model, make(0),
                                  duration_s=15.0)
    return rejected, st.error is None


def switching(args, doc, ctx):
    """Cancel-and-reopen churn alternating the two E-D-loadable models
    (NFR-REL-006). Zero results across all cycles scores 0.0 (FAIL),
    never not-measured."""
    mon = ctx.get("monitor")
    models = args.models.split(",")

    if mon is not None:
        # warmup both models once so first-use allocation is outside the
        # churn measurement, then take the baseline.
        for m in models:
            asyncio.run(driver.run_session(
                args.host, args.port, m, _make_source(m)(0),
                duration_s=15.0, session_id=0))
        time.sleep(args.settle_seconds)
        _wait_for_sample(mon)
    rss0, gpu0 = _resource_point(mon)

    async def cycles():
        all_stats = []
        for i in range(args.cycles):
            m = models[i % len(models)]
            st = await driver.run_session(
                args.host, args.port, m, _make_source(m)(i),
                duration_s=args.cycle_duration, session_id=i)
            all_stats.append(st)
        return all_stats

    stats = asyncio.run(cycles())
    total = sum(len(st.results) for st in stats)
    mismatches = sum(st.model_id_mismatches for st in stats)
    measured = {"results_echoing_own_stream_model": {
        "proportion": ((total - mismatches) / total) if total else 0.0}}
    if mon is not None:
        time.sleep(args.settle_seconds)
        rss1, gpu1 = _resource_point(mon)
        if rss0 is not None and rss1 is not None:
            measured["rss_delta_vs_baseline"] = {
                "max": 100.0 * (rss1 - rss0) / rss0}
        if gpu0 is not None and gpu1 is not None:
            measured["gpu_memory_delta_vs_baseline"] = {"max": gpu1 - gpu0}
    measured.update(_loop_lag_measured(ctx))
    return measured, {"cycles": args.cycles, "models": models,
                      "total_results": total,
                      "errors": [st.error for st in stats if st.error]}


def smoke(args, doc, ctx):
    """Fast CI subset: proto identity (COMPAT-001), no-metadata default
    (COMPAT-002), TF-absent service consistency (COMPAT-003, FLEX-001 E-D
    half), unit suite (MAINT-001), and the report↔YAML mechanical match
    (MAINT-004's second assertion; its no-literal assertion is discharged
    by inspection, declared in the YAML)."""
    measured = {}

    # NFR-COMPAT-001
    with open(paths.BACKEND_PROTO, "rb") as f:
        backend = f.read()
    if os.path.exists(paths.MOBILE_PROTO):
        with open(paths.MOBILE_PROTO, "rb") as f:
            mobile = f.read()
        diff = sum(1 for a, b in zip(backend, mobile) if a != b) \
            + abs(len(backend) - len(mobile))
        measured["differing_bytes_between_proto_copies"] = {"count": diff}

    # NFR-MAINT-001
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=paths.BACKEND_ROOT, capture_output=True, text=True, timeout=600)
    out = proc.stdout + proc.stderr
    failures = _pytest_count(out, "failed")
    errors = _pytest_count(out, "error")
    if proc.returncode != 0 and failures == 0 and errors == 0:
        errors = 1  # pytest died without a parsable summary: not a pass
    measured["unit_test_failures"] = {"count": failures}
    measured["unit_test_collection_errors"] = {"count": errors}

    # NFR-MAINT-004 — §5 tables vs YAML mechanical check
    measured["report_yaml_threshold_mismatches"] = {
        "count": _report_yaml_mismatches(doc)}

    if not args.offline:
        # NFR-COMPAT-002/003, NFR-FLEX-001 (E-D half) — live probes
        health = asyncio.run(driver.health_check(args.host, args.port))
        served = set(health["loaded_models"])
        pytorch_served = served - {"multisensory"}
        measured["pytorch_modes_served_without_tf"] = {
            "count": len(pytorch_served)}
        measured["models_served_on_e_d_without_tf"] = {
            "count": len(pytorch_served)}
        # "exactly": the served set must be one of the two legitimate
        # states of the registry — all models, or all minus multisensory —
        # and the any-loaded flag must agree.
        registry_ids = set(driver.PROFILES)
        measured["health_lists_exactly_loaded_models"] = {
            "boolean": (served in (registry_ids,
                                   registry_ids - {"multisensory"})
                        and health["model_loaded"] == bool(served))}

        default_ok = asyncio.run(_no_metadata_probe(args))
        measured["metadata_less_streams_running_sonicsight"] = {
            "proportion": 1.0 if default_ok else 0.0}
        extras_health = health
    else:
        extras_health = None
    return measured, {"pytest_tail": out.strip().splitlines()[-3:],
                      "health": extras_health}


def _pytest_count(out, word):
    m = re.search(r"(\d+)\s+" + word, out)
    return int(m.group(1)) if m else 0


async def _no_metadata_probe(args):
    """Open a stream with NO metadata; the echoed model id must be the
    default (FR-012). The expected id and its rates come from the server's
    own registry, not literals."""
    from model_registry import DEFAULT_MODEL_ID
    pb2, pb2_grpc = paths.import_stubs()
    import grpc
    prof = driver.PROFILES[DEFAULT_MODEL_ID]
    samples = int(round(prof["sample_rate"] * 0.125))
    src = media.SyntheticSource(seed=0, sample_rate=prof["sample_rate"],
                                frame_rate=prof["frame_rate"],
                                frame_kind=prof["frame_kind"])
    channel = grpc.aio.insecure_channel(f"{args.host}:{args.port}")
    stub = pb2_grpc.SonicSightServiceStub(channel)

    async def gen():
        for i in range(40):  # ~5 s of audio: enough for early-mode results
            yield pb2.StreamChunk(timestamp_ms=i * 125,
                                  audio_pcm=src.audio_block(i, samples))
            await asyncio.sleep(0.0)
        yield pb2.StreamChunk(timestamp_ms=5000, is_last=True)

    ok = True
    try:
        async for r in stub.StreamProcess(gen()):
            if r.model_id and r.model_id != DEFAULT_MODEL_ID:
                ok = False
    finally:
        await channel.close()
    return ok


def _report_yaml_mismatches(doc):
    """Mechanical §5↔YAML check (NFR-MAINT-004): scoped to the report's
    section 5, and anchored — a threshold must appear as a bold token
    (optionally with its unit) inside its NFR's own table row, and rows
    asserting a percentile must name that percentile. Value-substring
    matching alone would false-pass small integers against incidental
    digits."""
    report_path = os.path.join(paths.WORKSPACE_ROOT, "docs",
                               "ANALYSIS_REPORT.md")
    text = open(report_path, "r", encoding="utf-8").read()
    m = re.search(r"\n## 5\. .*?(?=\n## 6\. )", text, re.S)
    if not m:
        return len(doc["targets"])
    section5 = m.group(0)

    def norm(s):
        return (s.replace(" ", "").replace(" ", "")
                 .replace(" ", ""))

    mismatches = 0
    for tid, t in doc["targets"].items():
        row = re.search(r"\|\s*" + re.escape(tid) + r"\s*\|(.*?)\n", section5)
        if not row:
            mismatches += 1
            continue
        row_norm = norm(row.group(1))
        for a in t.get("assertions", []):
            v = a["value"]
            if v is None or isinstance(v, bool):
                continue
            stat = a.get("statistic", "")
            if stat in ("p50", "p90", "p95", "p99") and stat not in row_norm:
                mismatches += 1
                continue
            vnorm = norm(f"{v:g}") if isinstance(v, float) else str(v)
            unit = a.get("unit") or ""
            unit_tok = {"ms": "ms", "MiB": "MiB", "MiB/min": "MiB/min",
                        "percent": "%"}.get(unit, "")
            candidates = ["**" + vnorm, vnorm + unit_tok]
            if isinstance(v, float) and v.is_integer():
                candidates += ["**" + str(int(v)), str(int(v)) + unit_tok]
            if unit == "ratio":
                candidates.append(norm(f"{v * 100:g}") + "%")
            if not any(c and c in row_norm for c in candidates):
                mismatches += 1
    return mismatches


SCENARIOS = {
    "baseline": baseline,
    "load": load,
    "stress": stress,
    "spike": spike,
    "soak": soak,
    "failure-injection": failure,
    "switching": switching,
    "smoke": smoke,
}
