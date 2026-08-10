"""Asyncio session driver on the project's own generated stubs.

Why a custom driver (the brief's tool question, answered): the service is
gRPC *bidirectional streaming* with in-band control fields (pixel queries,
freeze, sticky), per-stream model selection via metadata, and a cadence
contract measured per result. Generic load tools handle this poorly: `ghz`
drives unary/streaming RPCs from canned payload files but cannot express
paced two-message-per-tick capture, mid-stream cancellation, or per-result
metric extraction; `k6` needs the xk6-grpc extension build and still models
streams awkwardly from JS; Locust is HTTP-first with gRPC bolted on and adds
a coordination layer this single-host suite does not need. The project
already ships a validated single-session driver (replay_client.py) on its
own stubs — this module generalises that exact protocol to N concurrent
asyncio sessions, injections, and switching, so the harness speaks the real
wire contract by construction and cannot drift from config.py constants.
"""

import asyncio
import bisect
import time

import grpc

from . import paths

sonicsight_pb2, sonicsight_pb2_grpc = paths.import_stubs()
from config import IMG_SIZE  # noqa: E402

from model_registry import MODEL_METADATA_KEY, REGISTRY  # noqa: E402

# Client-side capture profiles, mirrored from the registry specs the same
# way the mobile ModelProfile table is (FR-065). Values are read from the
# server's own registry so the driver cannot drift. hop_samples is defined
# in the wire (capture) domain (model_registry.py:52), hence the divisor.
PROFILES = {}
for _id, _spec in REGISTRY.items():
    PROFILES[_id] = {
        "model_id": _id,
        "sample_rate": _spec.capture_sample_rate,
        "frame_rate": _spec.frame_rate,
        "frame_kind": _spec.frame_kind,
        "audio_field": _spec.audio_chunk_field,
        "hop_ms": 1000.0 * _spec.hop_samples / _spec.capture_sample_rate,
    }


class SessionStats:
    """Per-session recorder; all timestamps are time.perf_counter() seconds."""

    def __init__(self, session_id, profile):
        self.session_id = session_id
        self.profile = profile
        self.t_open = None            # set just before the RPC is created;
        # grpc.aio connects lazily, so TTFR includes channel establishment —
        # a conservative (over-)estimate of "from stream establishment".
        self.t_first_result = None
        self.t_first_nonbuffering = None
        self.results = []             # dicts per non-buffering window result
        self.buffering_count = 0
        self.error_results = 0
        self.model_id_mismatches = 0
        self.seq_gaps = 0
        self.last_seq = None
        self.error = None
        self.cancelled = False
        self.query_rtts_ms = []       # pixel mode
        self.query_send_times = {}    # query_id -> monotonic send time
        self._audio_ts = []           # sorted capture timestamps (ms)
        self._audio_sent_at = {}      # timestamp_ms -> monotonic send time

    def record_audio_sent(self, ts_ms, t_mono):
        self._audio_ts.append(ts_ms)
        self._audio_sent_at[ts_ms] = t_mono

    def _send_time_at_or_before(self, ts_ms):
        i = bisect.bisect_right(self._audio_ts, ts_ms)
        if i == 0:
            return None
        return self._audio_sent_at[self._audio_ts[i - 1]]

    def on_result(self, r, now):
        if self.t_first_result is None:
            self.t_first_result = now
        if getattr(r, "is_buffering", False):
            self.buffering_count += 1
            return
        if not r.success and r.error_message:
            self.error = r.error_message
            self.error_results += 1
            return  # error results carry no window payload; keep them out
            # of the cadence/gap series (they have seq 0, zero timings).
        if r.model_id and r.model_id != self.profile["model_id"]:
            self.model_id_mismatches += 1
        if r.pixel_audio and not r.sequence_number:
            return  # pixel query answer (seq 0): RTT is booked separately;
            # it is not a window result and must not enter cadence math.
        if self.t_first_nonbuffering is None:
            self.t_first_nonbuffering = now
        # sequence numbering: halves numbers windows from 0, pixel from 1,
        # and the final flush result is unnumbered (0) — count gaps only
        # across strictly increasing pairs.
        if (self.last_seq is not None and r.sequence_number > self.last_seq + 1):
            self.seq_gaps += r.sequence_number - self.last_seq - 1
        if self.last_seq is None or r.sequence_number > self.last_seq:
            self.last_seq = r.sequence_number
        # audio-age proxy for perceived lag: arrival minus send time of the
        # newest capture chunk at or before the result's echoed timestamp
        # (the server echoes a computed window-center time, not a capture
        # tick, so this must be a floor lookup, not an exact match).
        sent = self._send_time_at_or_before(r.timestamp_ms)
        lag_ms = (now - sent) * 1000.0 if sent is not None else None
        self.results.append({
            "t": now,
            "seq": r.sequence_number,
            "timestamp_ms": r.timestamp_ms,
            "inference_ms": r.inference_time_ms,
            "post_ms": r.post_processing_time_ms,
            "total_server_ms": r.total_server_time_ms,
            "audio_samples": r.audio_sample_count,
            "audio_age_ms": lag_ms,
        })

    # ── derived metrics ────────────────────────────────────────────────
    def time_to_first_nonbuffering(self):
        if self.t_first_nonbuffering is None or self.t_open is None:
            return None
        return self.t_first_nonbuffering - self.t_open

    def inter_result_intervals_ms(self):
        ts = [r["t"] for r in self.results]
        return [(b - a) * 1000.0 for a, b in zip(ts, ts[1:])]

    def cadence_abs_dev_ms(self):
        hop = self.profile["hop_ms"]
        return [abs(iv - hop) for iv in self.inter_result_intervals_ms()]

    def server_times_ms(self):
        return [r["total_server_ms"] for r in self.results if r["total_server_ms"]]

    def audio_ages_ms(self):
        return [r["audio_age_ms"] for r in self.results if r["audio_age_ms"] is not None]

    def sequence_gap_rate(self):
        n = len(self.results)
        if n == 0:
            return None
        return self.seq_gaps / float(n + self.seq_gaps)


async def chunk_stream(stats, source, duration_s, queries_per_s=0.0,
                       inject=None, stop_event=None):
    """Paced capture generator: one audio chunk per 125 ms tick, frames at
    the profile rate (catching up on every frame due within the tick, each
    stamped with its own frame time), one is_last terminal chunk."""
    prof = stats.profile
    sr = prof["sample_rate"]
    tick_ms = 125.0  # audio block cadence, streamRate/8 per FR-011
    samples = int(round(sr * tick_ms / 1000.0))
    frame_interval = 1000.0 / prof["frame_rate"]
    audio_field = prof["audio_field"]

    n_ticks = int(duration_s * 1000.0 / tick_ms)
    t0 = time.perf_counter()
    next_frame_at = 0.0
    frame_idx = 0
    next_query_at = 1.0 / queries_per_s if queries_per_s > 0 else None
    qid = 0

    for i in range(n_ticks):
        if stop_event is not None and stop_event.is_set():
            break
        target = t0 + i * tick_ms / 1000.0
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        ts_ms = int(i * tick_ms)
        audio = source.audio_block(i, samples)
        chunk = sonicsight_pb2.StreamChunk(timestamp_ms=ts_ms,
                                           **{audio_field: audio})

        elapsed_ms = i * tick_ms
        if next_query_at is not None and elapsed_ms / 1000.0 >= next_query_at:
            next_query_at += 1.0 / queries_per_s
            qid += 1
            chunk.queries.append(sonicsight_pb2.PixelQuery(
                query_id=qid, x_norm=0.5, y_norm=0.5, radius_norm=0.15,
                window_id=0))
            stats.query_send_times[qid] = time.perf_counter()

        if inject is not None:
            chunk = inject(i, chunk)
        if chunk is not None:
            stats.record_audio_sent(ts_ms, time.perf_counter())
            yield chunk

        # catch-up loop: emit every frame due by now (a 30 fps profile has
        # ~3-4 frames per 125 ms tick), each with its own frame timestamp.
        while elapsed_ms >= next_frame_at:
            fr = source.frame(frame_idx)
            frame_ts = int(next_frame_at)
            next_frame_at += frame_interval
            frame_idx += 1
            if fr is None:
                continue
            if inject is not None:
                fr = inject(("frame", i), fr)
                if fr is None:
                    continue
            yield sonicsight_pb2.StreamChunk(
                timestamp_ms=frame_ts, frame_width=IMG_SIZE,
                frame_height=IMG_SIZE, **fr)

    yield sonicsight_pb2.StreamChunk(timestamp_ms=int(n_ticks * tick_ms),
                                     is_last=True)


async def run_session(host, port, model_id, source, duration_s,
                      session_id=0, queries_per_s=0.0, inject=None,
                      abrupt_cancel_after_s=None):
    """One full session; returns SessionStats. In-band errors land in
    stats.error; only external cancellation propagates."""
    prof = PROFILES[model_id]
    stats = SessionStats(session_id, prof)
    stop_event = asyncio.Event()

    channel = grpc.aio.insecure_channel(f"{host}:{port}")
    stub = sonicsight_pb2_grpc.SonicSightServiceStub(channel)
    metadata = ((MODEL_METADATA_KEY, model_id),)

    stats.t_open = time.perf_counter()
    call = stub.StreamProcess(
        chunk_stream(stats, source, duration_s, queries_per_s, inject,
                     stop_event),
        metadata=metadata)

    async def canceller():
        await asyncio.sleep(abrupt_cancel_after_s)
        stats.cancelled = True
        stop_event.set()
        call.cancel()

    cancel_task = (asyncio.ensure_future(canceller())
                   if abrupt_cancel_after_s is not None else None)
    try:
        async for r in call:
            now = time.perf_counter()
            for pa in r.pixel_audio:
                sent = stats.query_send_times.get(pa.query_id)
                if sent is not None:
                    stats.query_rtts_ms.append((now - sent) * 1000.0)
            stats.on_result(r, now)
    except asyncio.CancelledError:
        if not stats.cancelled:
            raise  # outer cancellation (gather teardown, Ctrl-C) propagates
    except grpc.aio.AioRpcError as e:
        if stats.cancelled and e.code() == grpc.StatusCode.CANCELLED:
            pass
        else:
            stats.error = f"{e.code().name}: {e.details()}"
    finally:
        if cancel_task is not None:
            cancel_task.cancel()
        await channel.close()
    return stats


async def run_many(host, port, model_id, make_source, n_sessions, duration_s,
                   stagger_s=0.0, **kwargs):
    """N concurrent sessions, optionally staggered; returns list of stats."""
    async def one(i):
        if stagger_s:
            await asyncio.sleep(i * stagger_s)
        return await run_session(host, port, model_id, make_source(i),
                                 duration_s, session_id=i, **kwargs)
    return await asyncio.gather(*(one(i) for i in range(n_sessions)))


async def health_check(host, port):
    channel = grpc.aio.insecure_channel(f"{host}:{port}")
    try:
        stub = sonicsight_pb2_grpc.SonicSightServiceStub(channel)
        resp = await stub.HealthCheck(sonicsight_pb2.Empty(), timeout=10)
        return {"model_loaded": resp.model_loaded, "device": resp.device,
                "loaded_models": list(resp.loaded_models)}
    finally:
        await channel.close()

