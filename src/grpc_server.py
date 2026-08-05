import os
import io
import shutil
import time
import asyncio
from uuid import uuid4
import logging

import numpy as np
import grpc
from grpc import aio
import torch

import sonicsight_pb2
import sonicsight_pb2_grpc
from inference import inference, StreamingBuffer
from overlap_add_buffer import OverlapAddBuffer
from video_preprocessor import VideoPreprocessor
from config import AUD_RATE, STREAM_DUMP, STREAM_DUMP_DIR
from stream_dump import StreamDumper
from model_registry import (
    DEFAULT_MODEL_ID,
    MODEL_METADATA_KEY,
    REGISTRY,
    get_engine,
    load_all_engines,
    loaded_model_ids,
)

logger = logging.getLogger(__name__)


class SonicSightServicer(sonicsight_pb2_grpc.SonicSightServiceServicer):
    """gRPC servicer implementing the SonicSight service."""

    def __init__(self):
        self._preprocessor = VideoPreprocessor()
        self._inference_lock = asyncio.Lock()

    async def HealthCheck(self, request, context):
        device_str = str(inference.device) if inference.device else "not loaded"
        loaded = loaded_model_ids()
        return sonicsight_pb2.HealthResponse(
            model_loaded=bool(loaded),
            device=device_str,
            loaded_models=loaded,
        )

    async def StreamProcess(self, request_iterator, context):
        # Model selection: gRPC metadata key "sonicsight-model". Absent ->
        # default model, so pre-registry clients keep working unchanged.
        # Switching models is always cancel-stream + reopen; the capture
        # profiles are incompatible mid-stream.
        metadata = dict(context.invocation_metadata())
        model_id = metadata.get(MODEL_METADATA_KEY, DEFAULT_MODEL_ID)
        spec = REGISTRY.get(model_id)
        if spec is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"Unknown model '{model_id}'. Available: {', '.join(sorted(REGISTRY))}",
            )
        engine = get_engine(model_id)
        if not engine.is_loaded:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"Model '{model_id}' is not loaded on this server.",
            )

        # Pixel mode gets its own loop: cache-and-query architecture, no
        # left/right synthesis. The halves/multisensory path below is not
        # shared with it and stays byte-identical.
        if spec.mode == "pixel":
            async for result in self._pixel_stream(
                request_iterator, context, spec, engine, model_id
            ):
                yield result
            return

        logger.info("Client connected for StreamProcess (model=%s)...", model_id)
        start_process_time = time.time()

        # Initialize streaming buffer for this connection, shaped by the spec
        buffer = StreamingBuffer(
            target_audio_len=spec.window_samples,
            target_sr=spec.capture_sample_rate,
            num_frames=spec.num_frames,
            frame_ring_cap=spec.frame_ring_cap,
            early_min_samples=spec.early_min_samples,
            frame_selection=spec.frame_selection,
            window_min_advance=spec.window_min_advance,
        )
        left_ola = OverlapAddBuffer(
            window_len=spec.window_samples, hop_samples=spec.hop_samples
        )
        right_ola = OverlapAddBuffer(
            window_len=spec.window_samples, hop_samples=spec.hop_samples
        )

        # Diagnostic recorder (SONICSIGHT_DUMP_STREAM=1). control_ola is fed
        # the UNSEPARATED input window through the identical overlap-add path,
        # giving an identity-mask reference: whatever damage the windowing and
        # stitching do to the audio shows up in control.wav without the model
        # being involved at all.
        dumper = None
        control_ola = None
        if STREAM_DUMP:
            session_dir = os.path.join(STREAM_DUMP_DIR, time.strftime("%Y%m%d_%H%M%S"))
            dumper = StreamDumper(session_dir)
            control_ola = OverlapAddBuffer(
                window_len=spec.window_samples, hop_samples=spec.hop_samples
            )
            logger.info("STREAM DUMP ENABLED -> %s", session_dir)

        # Track processing state
        chunks_received = 0
        prev_window_start = None
        last_yielded_timestamp = -1
        cycles_processed = 0
        seq_number = 0

        try:
            async for chunk in request_iterator:
                chunk_receive_time = time.time()
                chunks_received += 1

                # 1. Add data to buffer. Which chunk fields feed this stream
                # comes from the spec: audio_pcm + left/right JPEG halves for
                # the sonicsight branch, audio_pcm_hi + full_jpeg for models
                # taking the whole frame.
                audio_bytes = getattr(chunk, spec.audio_chunk_field)
                if audio_bytes:
                    buffer.add_audio_chunk(audio_bytes, chunk.timestamp_ms)
                    # DEBUG: Check if audio is silent
                    audio_arr = np.frombuffer(audio_bytes, dtype=np.int16)
                    if dumper is not None:
                        dumper.add_mic(audio_bytes, chunk.timestamp_ms)
                    if chunks_received % 20 == 0:
                        logger.info(f"Incoming audio max amp: {np.max(np.abs(audio_arr))}")

                if spec.frame_kind == "left_right_halves":
                    if chunk.left_jpeg and chunk.right_jpeg:
                        try:
                            timestamp_ms, left_img, right_img = await asyncio.to_thread(
                                buffer.decode_images,
                                chunk.timestamp_ms,
                                chunk.left_jpeg,
                                chunk.right_jpeg
                            )
                            buffer.add_decoded_frame(timestamp_ms, left_img, right_img)
                        except Exception as e:
                            logger.info(f"Error decoding frame: {e}")
                elif chunk.full_jpeg:
                    try:
                        timestamp_ms, full_img = await asyncio.to_thread(
                            buffer.decode_full_image,
                            chunk.timestamp_ms,
                            chunk.full_jpeg,
                        )
                        buffer.add_decoded_frame(timestamp_ms, None, None, full_img)
                    except Exception as e:
                        logger.info(f"Error decoding frame: {e}")

                # 2. Check if we have enough data to run inference
                if not buffer.has_enough_data():
                    # Stream back a "buffering" status
                    if chunks_received % 10 == 0:  # Don't spam, send every 10th chunk
                        yield sonicsight_pb2.StreamResult(
                            success=True,
                            is_buffering=True,
                            timestamp_ms=chunk.timestamp_ms,
                            model_id=model_id,
                            heatmap_count=spec.heatmap_count,
                        )
                    continue

                # 3. Get the latest valid window
                window_start_time = time.time()
                audio_window, frames, center_timestamp, window_start_sample = (
                    buffer.get_latest_window()
                )

                if audio_window is None:
                    # Still waiting for future audio to arrive to match the current frames
                    if chunks_received % 10 == 0:
                        yield sonicsight_pb2.StreamResult(
                            success=True,
                            is_buffering=True,
                            timestamp_ms=chunk.timestamp_ms,
                            model_id=model_id,
                            heatmap_count=spec.heatmap_count,
                        )
                    continue

                # Don't re-process the same window if no new frames have arrived
                if center_timestamp == last_yielded_timestamp:
                    continue

                # 4. Run inference (synchronously in a thread to not block event loop)
                logger.info(f"Running inference for timestamp {center_timestamp}ms... (Buffer prep took {int((window_start_time - chunk_receive_time)*1000)}ms)")

                inference_start_time = time.time()
                async with self._inference_lock:
                    result = await asyncio.to_thread(
                        engine.eval_stream_window,
                        audio_window,
                        frames
                    )
                inference_end_time = time.time()
                cycles_processed += 1

                # 5. Stitch full windows together with overlap-add.
                post_start_time = time.time()
                if (
                    left_ola.latest_window_start is not None
                    and window_start_sample <= left_ola.latest_window_start
                ):
                    logger.warning(
                        "Skipping stale streaming window before OLA: start_sample=%s last_start=%s center=%sms",
                        window_start_sample,
                        left_ola.latest_window_start,
                        center_timestamp,
                    )
                    continue

                # Determine center-slice extraction offset.
                # In early inference mode the real audio sits at the START of the
                # 65536-sample window (the rest is zero-padded).  We tell the OLA
                # buffer to extract from a position that falls inside the real audio,
                # rather than the mathematical center of the window.
                ola_center_offset = None
                if getattr(buffer, '_last_window_mode', 'normal') == 'early':
                    early_len = getattr(buffer, '_last_early_audio_len', 0)
                    hop = left_ola.hop_samples
                    if early_len > hop:
                        # Place extraction in the center of the real audio portion
                        ola_center_offset = max(0, (early_len - hop) // 2)
                    else:
                        ola_center_offset = 0

                # Early-mode windows (zero-padded, low-quality audio, centre placed near
                # sample 0) must NOT be fed into the OLA buffer. If they were, the next
                # normal-mode window would drain the entire gap between the early centre
                # (~0) and the normal centre (~32079) in one shot – a burst of ~26000
                # samples / 53 KB. The phone JitterBuffer holds only 33 KB, so the write
                # wraps 1.6x around the ring and corrupts every byte already in it,
                # producing static for the rest of the session.
                is_early_mode = getattr(buffer, '_last_window_mode', 'normal') == 'early'
                if is_early_mode:
                    # Skip audio output entirely; keep sending buffering status so the
                    # client stays alive. The heatmap is also low-quality here, so skip.
                    logger.debug("Skipping OLA add for early-mode window (jitter-overflow guard).")
                    yield sonicsight_pb2.StreamResult(
                        success=True,
                        is_buffering=True,
                        timestamp_ms=center_timestamp,
                        model_id=model_id,
                        heatmap_count=spec.heatmap_count,
                    )
                    last_yielded_timestamp = center_timestamp
                    continue

                left_ola.add_window(result["left_audio"], start_sample=window_start_sample, center_offset=ola_center_offset)
                right_ola.add_window(result["right_audio"], start_sample=window_start_sample, center_offset=ola_center_offset)
                left_pcm = left_ola.drain()
                right_pcm = right_ola.drain()

                if len(left_pcm) != len(right_pcm):
                    raise RuntimeError("Left/right OLA drains returned mismatched PCM lengths.")

                # Safety net: if a drain is still unexpectedly large (should not happen
                # after the early-mode guard above, but protects against future regressions)
                # trim to the newest 1.5 seconds and log loudly so we can investigate.
                SAFE_DRAIN_SAMPLES = int(AUD_RATE * 1.5)  # 16537 samples = jitter cap
                if len(left_pcm) // 2 > SAFE_DRAIN_SAMPLES:
                    logger.warning(
                        "Oversized OLA drain detected (%d samples > %d cap) – trimming to prevent "
                        "JitterBuffer overflow. This should not happen; check early-mode logic.",
                        len(left_pcm) // 2,
                        SAFE_DRAIN_SAMPLES,
                    )
                    left_pcm = left_pcm[-SAFE_DRAIN_SAMPLES * 2:]
                    right_pcm = right_pcm[-SAFE_DRAIN_SAMPLES * 2:]

                if dumper is not None:
                    control_ola.add_window(
                        audio_window,
                        start_sample=window_start_sample,
                        center_offset=ola_center_offset,
                    )
                    control_pcm = control_ola.drain()
                    dumper.add_output(left_pcm, right_pcm, control_pcm)
                    dumper.maybe_dump_window(
                        cycles_processed,
                        audio_window,
                        result["left_audio"],
                        result["right_audio"],
                    )
                    d = result.get("diag") or {}
                    lh_d = result["left_heatmap"]
                    if lh_d is None:  # confidence-gated window
                        lh_d = np.zeros((1,), dtype=np.float32)
                    dumper.add_cycle(
                        {
                            "cycle": cycles_processed,
                            "seq": seq_number,
                            "center_ts_ms": center_timestamp,
                            "window_start_sample": window_start_sample,
                            "window_delta": (
                                window_start_sample - prev_window_start
                                if prev_window_start is not None
                                else 0
                            ),
                            "mode": getattr(buffer, "_last_window_mode", "normal"),
                            "emitted_samples": len(left_pcm) // 2,
                            "audio_gain": d.get("audio_gain", 0.0),
                            "in_rms": d.get("in_rms", 0.0),
                            "out_rms_left": d.get("out_rms_left", 0.0),
                            "out_rms_right": d.get("out_rms_right", 0.0),
                            "mask_mean_left": d.get("mask_mean_left", 0.0),
                            "mask_mean_right": d.get("mask_mean_right", 0.0),
                            "heatmap_min": float(lh_d.min()),
                            "heatmap_max": float(lh_d.max()),
                            "heatmap_std": float(lh_d.std()),
                            "gap_fills": left_ola.gap_fills,
                            "splice_fallbacks": left_ola.splice_fallbacks,
                            "lagging_windows": left_ola.lagging_windows,
                            "inference_ms": int(
                                (inference_end_time - inference_start_time) * 1000
                            ),
                        }
                    )
                    prev_window_start = window_start_sample

                # Diagnostic: log audio levels periodically
                if cycles_processed % 10 == 0 and len(left_pcm) > 0:
                    left_samples = np.frombuffer(left_pcm, dtype=np.int16)
                    logger.info(
                        f"Audio drain: {len(left_samples)} samples, "
                        f"max_amp={np.max(np.abs(left_samples))}, "
                        f"rms={np.sqrt(np.mean(left_samples.astype(np.float64)**2)):.1f}"
                    )

                # 6. Stream back the result
                # OPTIMIZATION: Quantize heatmap to uint8 (0-255) to save 4x bandwidth
                # OPTIMIZATION: Stop sending center_frame_jpeg, mobile will use local cache
                # OPTIMIZATION: Skip PIL to JPEG encoding (saves ~500ms+)

                total_cycle_time = int((time.time() - chunk_receive_time) * 1000)
                inf_time = int((inference_end_time - inference_start_time) * 1000)
                # Ensure post-processing time is at least 1ms so the client
                # can clearly see that post-processing happened.
                post_time = max(1, int((time.time() - post_start_time) * 1000))

                # A model with heatmap_count == 1 returns right_heatmap=None;
                # its single map rides in left_heatmap and right stays empty.
                left_hm = result["left_heatmap"]
                right_hm = result.get("right_heatmap")
                yield sonicsight_pb2.StreamResult(
                    success=True,
                    is_buffering=False,
                    timestamp_ms=center_timestamp,
                    left_audio_pcm=left_pcm,
                    right_audio_pcm=right_pcm,
                    left_heatmap=(left_hm * 255).astype(np.uint8).tobytes() if left_hm is not None else b"",
                    right_heatmap=(right_hm * 255).astype(np.uint8).tobytes() if right_hm is not None else b"",
                    center_frame_jpeg=b"", # Empty to save bandwidth
                    inference_time_ms=inf_time,
                    post_processing_time_ms=post_time,
                    total_server_time_ms=total_cycle_time,
                    sequence_number=seq_number,
                    audio_sample_count=len(left_pcm) // 2,  # PCM16 = 2 bytes per sample
                    model_id=model_id,
                    heatmap_count=spec.heatmap_count,
                    # Raw-CAM positive fraction for confidence-gated models;
                    # 0.0 for models without a gate. When gated off, the
                    # heatmap bytes above are already empty.
                    cam_confidence=result.get("confidence", 0.0),
                )
                seq_number += 1

                last_yielded_timestamp = center_timestamp
                # Throttle per-cycle logs: reduce console I/O overhead and only
                # print every 10 cycles (or immediately when cycle is slow).
                if cycles_processed % 10 == 0 or total_cycle_time > 140:
                    lh = result["left_heatmap"]
                    hm_desc = (
                        f"min={lh.min():.4f} max={lh.max():.4f} std={lh.std():.4f}"
                        if lh is not None
                        else "gated (low CAM confidence)"
                    )
                    logger.info(
                        f"Cycle Complete: Inf={inf_time}ms, Post={post_time}ms, "
                        f"Total={total_cycle_time}ms | Heatmap L {hm_desc}"
                    )

                if chunk.is_last:
                    break

            final_left_pcm = left_ola.flush()
            final_right_pcm = right_ola.flush()
            if dumper is not None:
                dumper.add_output(
                    final_left_pcm, final_right_pcm, control_ola.flush()
                )
            if final_left_pcm or final_right_pcm:
                if len(final_left_pcm) != len(final_right_pcm):
                    raise RuntimeError("Left/right OLA flush returned mismatched PCM lengths.")

                yield sonicsight_pb2.StreamResult(
                    success=True,
                    is_buffering=False,
                    timestamp_ms=max(0, last_yielded_timestamp),
                    left_audio_pcm=final_left_pcm,
                    right_audio_pcm=final_right_pcm,
                    left_heatmap=b"",
                    right_heatmap=b"",
                    center_frame_jpeg=b"",
                    inference_time_ms=0,
                    post_processing_time_ms=0,
                    total_server_time_ms=0,
                    model_id=model_id,
                    heatmap_count=spec.heatmap_count,
                )

        except grpc.aio.AbortError:
            raise
        except torch.cuda.OutOfMemoryError:
            yield sonicsight_pb2.StreamResult(
                success=False,
                error_message="GPU out of memory.",
                model_id=model_id,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield sonicsight_pb2.StreamResult(
                success=False,
                error_message=str(e),
                model_id=model_id,
            )
        finally:
            if dumper is not None:
                summary = dumper.close(
                    {
                        "chunks_received": chunks_received,
                        "cycles_processed": cycles_processed,
                        "session_seconds": round(time.time() - start_process_time, 2),
                        "ola_gap_fills": left_ola.gap_fills,
                        "ola_splice_fallbacks": left_ola.splice_fallbacks,
                        "ola_lagging_windows": left_ola.lagging_windows,
                        "ola_total_emitted": left_ola.total_emitted,
                    }
                )
                logger.info("STREAM DUMP WRITTEN: %s", summary)

    async def _pixel_stream(self, request_iterator, context, spec, engine, model_id):
        """Pixel-mode stream loop (PIXEL_PLAN.md sections 2 and 5).

        Per completed window: one feature pass, cached in a small ring; the
        energy map and the mixture go out with the window result. Spatial
        queries arrive in-band and are answered against the cache — one
        InnerProd bmm + ISTFT, never a network forward, so a tap costs
        milliseconds while the window cadence stays at the 125 ms hop.
        """
        from pixel_cache import (
            GRID_H,
            GRID_W,
            EnergyMapSmoother,
            WindowCache,
            WindowCacheRing,
            cell_analysis,
            region_disc,
            synthesize_regions,
        )
        from clustering import discover_sources

        logger.info("Client connected for StreamProcess (model=%s, pixel mode)...", model_id)

        buffer = StreamingBuffer(
            target_audio_len=spec.window_samples,
            target_sr=spec.capture_sample_rate,
            num_frames=spec.num_frames,
            frame_ring_cap=spec.frame_ring_cap,
            early_min_samples=spec.early_min_samples,
            frame_selection=spec.frame_selection,
            window_min_advance=spec.window_min_advance,
        )
        # The mixture baseline rides left_audio_pcm: identity "mask" through
        # the same center-slice OLA stitching the other modes use.
        mix_ola = OverlapAddBuffer(
            window_len=spec.window_samples, hop_samples=spec.hop_samples
        )
        ring = WindowCacheRing(capacity=3)
        smoother = EnergyMapSmoother()

        MAX_QUERIES = 4  # proto contract; excess answered with an error
        # Safety net shared with the halves path: never hand the phone more
        # than its ~1.5 s jitter buffer in one drain.
        SAFE_DRAIN_SAMPLES = int(spec.output_sample_rate * 1.5)

        chunks_received = 0
        # Window results number from 1: sequence 0 is the reserved marker for
        # query answers (outside client audio gap detection).
        seq_number = 1
        last_yielded_timestamp = -1
        # Freeze is TTL-latched exactly like request_clusters: the client's
        # audio and frame chunks come from different threads, and a strict
        # per-chunk level would unpin ~8 times a second on flag-less audio
        # chunks (review finding). Rising edge pins; TTL expiry is the
        # falling edge.
        FREEZE_TTL = 16
        freeze_ttl = 0
        pinned_id = None
        # Source discovery enable: a TTL latch, not a per-chunk level. The
        # phone's audio and frame chunks are built by different threads, so a
        # per-chunk level would flap off on every chunk that omits the flag
        # (review finding). Any request_clusters=true arms the latch for
        # CLUSTERS_TTL chunks (~1 s at ~16 chunks/s); expiry is the falling
        # edge and resets persistence state, so stale identities cannot be
        # resurrected minutes later.
        CLUSTERS_TTL = 16
        clusters_ttl = 0
        cluster_state = None  # label persistence across windows

        try:
            async for chunk in request_iterator:
                chunks_received += 1

                audio_bytes = getattr(chunk, spec.audio_chunk_field)
                if audio_bytes:
                    buffer.add_audio_chunk(audio_bytes, chunk.timestamp_ms)

                if chunk.full_jpeg:
                    try:
                        timestamp_ms, full_img = await asyncio.to_thread(
                            buffer.decode_full_image,
                            chunk.timestamp_ms,
                            chunk.full_jpeg,
                        )
                        buffer.add_decoded_frame(timestamp_ms, None, None, full_img)
                    except Exception as e:
                        logger.info(f"Error decoding frame: {e}")

                # Rising edge pins; while latched, window_id=0 queries resolve
                # to the pinned window — the client never guesses server ids.
                if chunk.freeze:
                    if freeze_ttl == 0:
                        pinned_id = ring.pin_newest()
                        logger.info("Pixel freeze: pinned window_id=%s", pinned_id)
                    freeze_ttl = FREEZE_TTL
                elif freeze_ttl > 0:
                    freeze_ttl -= 1
                    if freeze_ttl == 0:
                        ring.unpin_all()
                        pinned_id = None
                        logger.info("Pixel freeze released")
                frozen = freeze_ttl > 0

                if chunk.request_clusters:
                    clusters_ttl = CLUSTERS_TTL
                elif clusters_ttl > 0:
                    clusters_ttl -= 1
                    if clusters_ttl == 0:
                        cluster_state = None  # falling edge: no stale revivals
                clusters_on = clusters_ttl > 0

                # ── Answer queries against the cache (no inference) ──
                pixel_answers = []
                for i, q in enumerate(chunk.queries):
                    if i >= MAX_QUERIES:
                        pixel_answers.append(sonicsight_pb2.PixelAudio(
                            query_id=q.query_id,
                            error=f"query cap ({MAX_QUERIES}) exceeded",
                        ))
                        continue
                    wid = q.window_id
                    if wid == 0 and frozen and pinned_id is not None:
                        wid = pinned_id
                    cache = ring.get(wid)
                    if cache is None:
                        pixel_answers.append(sonicsight_pb2.PixelAudio(
                            query_id=q.query_id,
                            window_id=q.window_id,
                            error="window not available (not cached yet, or evicted)",
                        ))
                        continue
                    weights = region_disc(q.x_norm, q.y_norm, q.radius_norm)
                    # Honest silence: if no cell in the tapped region clears
                    # the measured contrast gate, say "nothing here" instead
                    # of synthesizing the ~40%-of-everything leakage a
                    # background sigmoid produces (calibrated 2026-08-05).
                    if cache.energy_map is not None:
                        from clustering import silence_threshold
                        thr = silence_threshold(cache.energy_map.reshape(-1))
                        if not (cache.energy_map[weights > 0.5] > thr).any():
                            pixel_answers.append(sonicsight_pb2.PixelAudio(
                                query_id=q.query_id,
                                energy=0.0,
                                window_id=cache.window_id,
                            ))
                            continue
                    w = torch.from_numpy(weights)
                    # Region + complement, renormed together: the pair
                    # partitions the mixture (PIXEL_PLAN 2.2). The region
                    # track is returned; the complement exists so the math
                    # matches the mixer semantics.
                    async with self._inference_lock:
                        wavs, energies = await asyncio.to_thread(
                            synthesize_regions, engine, cache, [w, 1.0 - w]
                        )
                    pcm = (np.clip(wavs[0], -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    # RELATIVE energy: fraction of the window's mixture energy
                    # captured by the region, 0..1 — scale-free, so the
                    # client's "no sound detected here" gate is meaningful.
                    rel_energy = energies[0] / max(cache.mixture_energy(), 1e-12)
                    pixel_answers.append(sonicsight_pb2.PixelAudio(
                        query_id=q.query_id,
                        pcm=pcm,
                        sample_count=len(wavs[0]),
                        energy=min(1.0, rel_energy),
                        window_id=cache.window_id,
                    ))

                # Query answers go out IMMEDIATELY — never parked on the next
                # window result, where an inference failure could drop them
                # and where the wait would break the "a tap feels instant"
                # contract. sequence_number=0 keeps them outside the client's
                # audio gap detection (which tracks only results with seq > 0).
                if pixel_answers:
                    newest = ring.newest()
                    yield sonicsight_pb2.StreamResult(
                        success=True,
                        is_buffering=False,
                        timestamp_ms=chunk.timestamp_ms,
                        sequence_number=0,
                        model_id=model_id,
                        heatmap_count=spec.heatmap_count,
                        pixel_audio=pixel_answers,
                        window_id=(
                            pinned_id if (frozen and pinned_id is not None)
                            else (newest.window_id if newest else 0)
                        ),
                    )

                # ── Window advance (normal-mode windows only) ──
                yielded_window = False
                if buffer.has_enough_data():
                    audio_window, frames, center_timestamp, window_start_sample = (
                        buffer.get_latest_window()
                    )
                    if (
                        audio_window is not None
                        and getattr(buffer, "_last_window_mode", "normal") == "normal"
                        and center_timestamp != last_yielded_timestamp
                    ):
                        # Feature pass AND the per-cell map pass hold the same
                        # lock every other GPU entry point in this servicer
                        # holds — cell_maps is real GPU work, not bookkeeping.
                        async with self._inference_lock:
                            feats = await asyncio.to_thread(
                                engine.eval_pixel_window, audio_window, frames
                            )
                            cache = WindowCache(
                                window_id=0,  # assigned by the ring
                                feat_sound=feats["feat_sound"],
                                feats_img=feats["feats_img"],
                                mag_lin=feats["mag_lin"],
                                phase=feats["phase"],
                                audio_gain=feats["audio_gain"],
                                start_sample=window_start_sample,
                                center_timestamp_ms=center_timestamp,
                            )
                            energy_map_arr, _activation_map, cell_feats = (
                                await asyncio.to_thread(
                                    cell_analysis, engine, cache, 28, clusters_on
                                )
                            )
                        cache.energy_map = energy_map_arr
                        window_id = ring.push(cache)
                        energy_bytes = smoother.update(energy_map_arr)

                        cluster_labels_bytes = b""
                        cluster_protos = []
                        if clusters_on and cell_feats is not None:
                            label_map, found, cluster_state = discover_sources(
                                cell_feats, energy_map_arr, cluster_state
                            )
                            cluster_labels_bytes = label_map.tobytes()
                            cluster_protos = [
                                sonicsight_pb2.SourceCluster(
                                    cluster_id=c["cluster_id"],
                                    centroid_x=c["centroid_x"],
                                    centroid_y=c["centroid_y"],
                                    energy=c["energy"],
                                    rgb=c["rgb"],
                                )
                                for c in found
                            ]

                        mix_ola.add_window(
                            audio_window, start_sample=window_start_sample
                        )
                        mix_pcm = mix_ola.drain()
                        if len(mix_pcm) // 2 > SAFE_DRAIN_SAMPLES:
                            logger.warning(
                                "Oversized pixel mixture drain (%d samples) – trimming "
                                "to protect the client jitter buffer.",
                                len(mix_pcm) // 2,
                            )
                            mix_pcm = mix_pcm[-SAFE_DRAIN_SAMPLES * 2:]

                        yield sonicsight_pb2.StreamResult(
                            success=True,
                            is_buffering=False,
                            timestamp_ms=center_timestamp,
                            left_audio_pcm=mix_pcm,   # the MIXTURE (mixer baseline)
                            right_audio_pcm=b"",
                            left_heatmap=b"",          # energy_map is the only spatial payload
                            right_heatmap=b"",
                            sequence_number=seq_number,
                            audio_sample_count=len(mix_pcm) // 2,
                            model_id=model_id,
                            heatmap_count=spec.heatmap_count,
                            energy_map=energy_bytes,
                            grid_width=GRID_W,
                            grid_height=GRID_H,
                            cluster_labels=cluster_labels_bytes,
                            clusters=cluster_protos,
                            window_id=window_id,
                        )
                        seq_number += 1
                        last_yielded_timestamp = center_timestamp
                        yielded_window = True

                if not yielded_window and not pixel_answers and chunks_received % 10 == 0:
                    # Keepalive on the same cadence as the halves path — sent
                    # even when a window is already cached, so an audio stall
                    # never leaves the stream silent for an unbounded time.
                    yield sonicsight_pb2.StreamResult(
                        success=True,
                        is_buffering=True,
                        timestamp_ms=chunk.timestamp_ms,
                        model_id=model_id,
                        heatmap_count=spec.heatmap_count,
                    )

                if chunk.is_last:
                    break

            # Release the mixture's held-back crossfade tail, like the halves
            # path does on stream end.
            final_mix = mix_ola.flush()
            if final_mix:
                yield sonicsight_pb2.StreamResult(
                    success=True,
                    is_buffering=False,
                    timestamp_ms=max(0, last_yielded_timestamp),
                    left_audio_pcm=final_mix,
                    sequence_number=seq_number,
                    audio_sample_count=len(final_mix) // 2,
                    model_id=model_id,
                    heatmap_count=spec.heatmap_count,
                )

        except grpc.aio.AbortError:
            raise
        except torch.cuda.OutOfMemoryError:
            yield sonicsight_pb2.StreamResult(
                success=False,
                error_message="GPU out of memory.",
                model_id=model_id,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield sonicsight_pb2.StreamResult(
                success=False,
                error_message=str(e),
                model_id=model_id,
            )

    async def ProcessVideo(self, request_iterator, context):
        request_id = uuid4().hex
        temp_dir = os.path.join("src", "outputs", request_id)
        os.makedirs(temp_dir, exist_ok=True)
        raw_path = os.path.join(temp_dir, "raw_input.mp4")

        start_time = time.time()

        # Phase 3.7: Total stream duration timeout (120s)
        total_timeout = 120

        try:
            # ── 1. Receive and reassemble video chunks ──
            metadata = None
            total_bytes = 0
            expected_next_index = 0

            async def receive_chunks():
                nonlocal metadata, total_bytes, expected_next_index
                with open(raw_path, "wb") as f:
                    while True:
                        try:
                            # Phase 3.7: Stream inactivity timeout (30s)
                            # We manually call __anext__ to apply timeout to each chunk
                            chunk = await asyncio.wait_for(request_iterator.__anext__(), timeout=30.0)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            logger.info(f"[{request_id}] Stream inactivity timeout (30s)")
                            raise

                        # Validate metadata on first chunk
                        if chunk.HasField('metadata') and metadata is None:
                            metadata = chunk.metadata
                            logger.info(f"[{request_id}] Receiving: {metadata.filename} ({metadata.total_size} bytes)")

                        # Validate chunk ordering (Phase 3.6)
                        if chunk.chunk_index != expected_next_index:
                            await context.abort(grpc.StatusCode.OUT_OF_RANGE, f"Expected chunk {expected_next_index}")

                        f.write(chunk.data)
                        total_bytes += len(chunk.data)
                        expected_next_index += 1
                        if chunk.is_last:
                            break
                return True

            await asyncio.wait_for(receive_chunks(), timeout=float(total_timeout))

            if metadata is None:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "First chunk must include metadata",
                )

            # Validate total size
            if metadata.total_size > 0 and total_bytes != metadata.total_size:
                logger.info(
                    f"[{request_id}] Warning: Expected {metadata.total_size} bytes, "
                    f"received {total_bytes} bytes"
                )

            logger.info(
                f"[{request_id}] Received {total_bytes} bytes in "
                f"{expected_next_index} chunks"
            )

            # ── 2. FFmpeg preprocessing (in thread pool to avoid blocking) ──
            preprocessor = self._preprocessor
            processed = await asyncio.to_thread(
                preprocessor.preprocess, raw_path, temp_dir
            )

            logger.info(
                f"[{request_id}] Preprocessed: {processed.frame_count} frames extracted"
            )

            # ── 3. Run inference (one at a time via lock) ──
            async with self._inference_lock:
                result = await asyncio.to_thread(
                    inference.eval_for_grpc,
                    processed.audio_path,
                    processed.frames_dir,
                    processed.frame_count,
                )

            # ── 4. Serialize response ──
            # Convert separated audio from float32 to int16 PCM
            left_pcm = (result["left_audio"] * 32767).astype(np.int16).tobytes()
            right_pcm = (result["right_audio"] * 32767).astype(np.int16).tobytes()

            # Heatmaps as raw float32 bytes
            left_heatmap = result["left_heatmap"].astype(np.float32).tobytes()
            right_heatmap = result["right_heatmap"].astype(np.float32).tobytes()

            # Center frames as JPEG
            left_frame_bytes = self._pil_to_jpeg_bytes(result["left_center_frame"])
            right_frame_bytes = self._pil_to_jpeg_bytes(result["right_center_frame"])

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"[{request_id}] Inference complete in {elapsed_ms}ms")

            return sonicsight_pb2.InferenceResult(
                success=True,
                left_audio_pcm=left_pcm,
                right_audio_pcm=right_pcm,
                audio_sample_rate=AUD_RATE,
                left_heatmap=left_heatmap,
                right_heatmap=right_heatmap,
                left_center_frame=left_frame_bytes,
                right_center_frame=right_frame_bytes,
                processing_time_ms=elapsed_ms,
            )

        except grpc.aio.AbortError:
            raise
        except torch.cuda.OutOfMemoryError:
            return sonicsight_pb2.InferenceResult(
                success=False,
                error_message="GPU out of memory. Try a shorter video.",
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return sonicsight_pb2.InferenceResult(
                success=False,
                error_message=str(e),
            )
        finally:
            # Clean up temp files
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _pil_to_jpeg_bytes(pil_image):
        """Convert a PIL Image to JPEG bytes."""
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


async def serve(port=50051):
    """Start the gRPC server."""
    server = aio.server(
        options=[
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),  # 16MB
            ("grpc.max_send_message_length", 16 * 1024 * 1024),     # 16MB
            # Enable gzip compression — reduces payload size by 30-50% on
            # degraded networks at the cost of ~1ms CPU overhead per message.
            ("grpc.default_compression_algorithm", 2),  # 2 = gzip
            ("grpc.default_compression_level", 1),  # 1 = low (fastest)
        ]
    )
    sonicsight_pb2_grpc.add_SonicSightServiceServicer_to_server(
        SonicSightServicer(), server
    )
    server.add_insecure_port(f"0.0.0.0:{port}")
    await server.start()
    logger.info(f"gRPC server started on port {port}")
    await server.wait_for_termination()


if __name__ == "__main__":
    load_all_engines()
    asyncio.run(serve())
