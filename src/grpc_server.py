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
from config import AUD_RATE

logger = logging.getLogger(__name__)


class SonicSightServicer(sonicsight_pb2_grpc.SonicSightServiceServicer):
    """gRPC servicer implementing the SonicSight service."""

    def __init__(self):
        self._preprocessor = VideoPreprocessor()
        self._inference_lock = asyncio.Lock()

    async def HealthCheck(self, request, context):
        device_str = str(inference.device) if inference.device else "not loaded"
        return sonicsight_pb2.HealthResponse(
            model_loaded=inference.is_loaded,
            device=device_str,
        )

    async def StreamProcess(self, request_iterator, context):
        logger.info("Client connected for StreamProcess...")
        start_process_time = time.time()

        # Initialize streaming buffer for this connection
        buffer = StreamingBuffer()
        left_ola = OverlapAddBuffer()
        right_ola = OverlapAddBuffer()

        # Track processing state
        chunks_received = 0
        last_yielded_timestamp = -1
        cycles_processed = 0

        try:
            async for chunk in request_iterator:
                chunk_receive_time = time.time()
                chunks_received += 1

                # 1. Add data to buffer
                if chunk.audio_pcm:
                    buffer.add_audio_chunk(chunk.audio_pcm, chunk.timestamp_ms)
                    # DEBUG: Check if audio is silent
                    audio_arr = np.frombuffer(chunk.audio_pcm, dtype=np.int16)
                    if chunks_received % 20 == 0:
                        logger.info(f"Incoming audio max amp: {np.max(np.abs(audio_arr))}")

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

                # 2. Check if we have enough data to run inference
                if not buffer.has_enough_data():
                    # Stream back a "buffering" status
                    if chunks_received % 10 == 0:  # Don't spam, send every 10th chunk
                        yield sonicsight_pb2.StreamResult(
                            success=True,
                            is_buffering=True,
                            timestamp_ms=chunk.timestamp_ms
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
                            timestamp_ms=chunk.timestamp_ms
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
                        inference.eval_stream_window,
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

                left_ola.add_window(result["left_audio"], start_sample=window_start_sample)
                right_ola.add_window(result["right_audio"], start_sample=window_start_sample)
                left_pcm = left_ola.drain()
                right_pcm = right_ola.drain()

                if len(left_pcm) != len(right_pcm):
                    raise RuntimeError("Left/right OLA drains returned mismatched PCM lengths.")

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

                yield sonicsight_pb2.StreamResult(
                    success=True,
                    is_buffering=False,
                    timestamp_ms=center_timestamp,
                    left_audio_pcm=left_pcm,
                    right_audio_pcm=right_pcm,
                    left_heatmap=(result["left_heatmap"] * 255).astype(np.uint8).tobytes(),
                    right_heatmap=(result["right_heatmap"] * 255).astype(np.uint8).tobytes(),
                    center_frame_jpeg=b"", # Empty to save bandwidth
                    inference_time_ms=inf_time,
                    post_processing_time_ms=post_time,
                    total_server_time_ms=total_cycle_time
                )

                last_yielded_timestamp = center_timestamp
                # Throttle per-cycle logs: reduce console I/O overhead and only
                # print every 10 cycles (or immediately when cycle is slow).
                if cycles_processed % 10 == 0 or total_cycle_time > 140:
                    lh = result["left_heatmap"]
                    logger.info(
                        f"Cycle Complete: Inf={inf_time}ms, Post={post_time}ms, "
                        f"Total={total_cycle_time}ms | Heatmap L min={lh.min():.4f} "
                        f"max={lh.max():.4f} std={lh.std():.4f}"
                    )

                if chunk.is_last:
                    break

            final_left_pcm = left_ola.flush()
            final_right_pcm = right_ola.flush()
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
                )

        except grpc.aio.AbortError:
            raise
        except torch.cuda.OutOfMemoryError:
            yield sonicsight_pb2.StreamResult(
                success=False,
                error_message="GPU out of memory."
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield sonicsight_pb2.StreamResult(
                success=False,
                error_message=str(e),
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
            # Disable compression for low-latency streaming
            ("grpc.default_compression_algorithm", grpc.Compression.NoCompression),
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
    inference.load_model()
    asyncio.run(serve())
