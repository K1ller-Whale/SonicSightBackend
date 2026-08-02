"""Streaming diagnostic recorder.

Enabled with SONICSIGHT_DUMP_STREAM=1. Records, for a single StreamProcess
connection:

  mic.wav       - raw microphone PCM exactly as it arrived from the phone,
                  concatenated in arrival order. If this is already static,
                  the problem is upstream of the model entirely.

  control.wav   - the UNSEPARATED input window pushed through an identical
                  OverlapAddBuffer. This is the identity-mask reference: it is
                  what the client would receive if the model were replaced by a
                  pass-through. If mic.wav is clean and this is static, the
                  fault is in windowing / timeline / overlap-add, not the model.

  left.wav      - the exact PCM sent to the client, byte for byte.
  right.wav

  window_NN_*.wav - full 65536-sample windows (input, left, right) straight
                  out of the model, before overlap-add. Separates 'the model
                  emits noise' from 'the stitching creates noise'.

  chunks.csv    - one row per inbound audio chunk: timestamp, sample count,
                  cumulative samples, peak amplitude, and the drift between
                  the wall-clock timestamp and the sample count. Non-zero
                  growing drift means chunks are being dropped in transit.

  cycles.csv    - one row per inference cycle.
  summary.json  - totals and counters.

Everything is buffered in memory and written on close, so the recording adds
no per-cycle disk I/O to the latency path.
"""

import json
import os
import time
import wave

import numpy as np

from config import AUD_RATE


def _write_wav(path, pcm_bytes, sample_rate=AUD_RATE):
    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(bytes(pcm_bytes))


def _float_to_pcm16(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return b""
    a = np.clip(a, -1.0, 1.0)
    return np.round(a * 32767.0).astype(np.int16).tobytes()


def _rows_to_csv(rows, columns):
    lines = [",".join(columns)]
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append("%.6f" % v)
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    return "\n".join(lines) + "\n"


class StreamDumper:
    """Collects a full diagnostic record of one streaming session."""

    MAX_WINDOW_DUMPS = 6
    WINDOW_DUMP_EVERY = 8

    CHUNK_COLUMNS = [
        "index",
        "timestamp_ms",
        "samples",
        "cum_samples",
        "cum_ms_by_samples",
        "drift_ms",
        "peak_int16",
    ]

    CYCLE_COLUMNS = [
        "cycle",
        "seq",
        "center_ts_ms",
        "window_start_sample",
        "window_delta",
        "mode",
        "emitted_samples",
        "audio_gain",
        "in_rms",
        "out_rms_left",
        "out_rms_right",
        "mask_mean_left",
        "mask_mean_right",
        "heatmap_min",
        "heatmap_max",
        "heatmap_std",
        "gap_fills",
        "splice_fallbacks",
        "lagging_windows",
        "inference_ms",
    ]

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.mic = bytearray()
        self.left = bytearray()
        self.right = bytearray()
        self.control = bytearray()

        self.chunk_rows = []
        self.cycle_rows = []

        self.mic_chunks = 0
        self.mic_samples = 0
        self.first_chunk_ts = None
        self.windows_dumped = 0
        self.started = time.time()

    # -- inbound ---------------------------------------------------------

    def add_mic(self, pcm_bytes, timestamp_ms):
        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        self.mic += pcm_bytes
        self.mic_chunks += 1
        self.mic_samples += int(arr.size)

        ts = int(timestamp_ms)
        if self.first_chunk_ts is None:
            self.first_chunk_ts = ts

        cum_ms = (self.mic_samples / float(AUD_RATE)) * 1000.0
        elapsed_ms = ts - self.first_chunk_ts
        peak = int(np.max(np.abs(arr))) if arr.size else 0

        self.chunk_rows.append(
            {
                "index": self.mic_chunks,
                "timestamp_ms": ts,
                "samples": int(arr.size),
                "cum_samples": self.mic_samples,
                "cum_ms_by_samples": cum_ms,
                # Positive and growing => audio chunks are being lost in
                # transit, so the reassembled timeline is shorter than
                # real elapsed time.
                "drift_ms": elapsed_ms - cum_ms,
                "peak_int16": peak,
            }
        )

    # -- outbound --------------------------------------------------------

    def add_output(self, left_pcm, right_pcm, control_pcm):
        self.left += left_pcm
        self.right += right_pcm
        self.control += control_pcm

    def add_cycle(self, row):
        self.cycle_rows.append(row)

    def maybe_dump_window(self, cycle_index, mix_window, left_window, right_window):
        if self.windows_dumped >= self.MAX_WINDOW_DUMPS:
            return
        if cycle_index % self.WINDOW_DUMP_EVERY != 0:
            return
        tag = "%02d" % cycle_index
        _write_wav(
            os.path.join(self.out_dir, "window_%s_input.wav" % tag),
            _float_to_pcm16(mix_window),
        )
        _write_wav(
            os.path.join(self.out_dir, "window_%s_left.wav" % tag),
            _float_to_pcm16(left_window),
        )
        _write_wav(
            os.path.join(self.out_dir, "window_%s_right.wav" % tag),
            _float_to_pcm16(right_window),
        )
        self.windows_dumped += 1

    # -- finish ----------------------------------------------------------

    def close(self, extra=None):
        try:
            _write_wav(os.path.join(self.out_dir, "mic.wav"), self.mic)
            _write_wav(os.path.join(self.out_dir, "control.wav"), self.control)
            _write_wav(os.path.join(self.out_dir, "left.wav"), self.left)
            _write_wav(os.path.join(self.out_dir, "right.wav"), self.right)

            with open(os.path.join(self.out_dir, "chunks.csv"), "w") as f:
                f.write(_rows_to_csv(self.chunk_rows, self.CHUNK_COLUMNS))
            with open(os.path.join(self.out_dir, "cycles.csv"), "w") as f:
                f.write(_rows_to_csv(self.cycle_rows, self.CYCLE_COLUMNS))

            mic_arr = np.frombuffer(bytes(self.mic), dtype=np.int16)
            left_arr = np.frombuffer(bytes(self.left), dtype=np.int16)

            summary = {
                "duration_s": round(time.time() - self.started, 2),
                "mic_chunks": self.mic_chunks,
                "mic_samples": self.mic_samples,
                "mic_seconds": round(self.mic_samples / float(AUD_RATE), 2),
                "mic_peak": int(np.max(np.abs(mic_arr))) if mic_arr.size else 0,
                "mic_rms": (
                    float(np.sqrt(np.mean(mic_arr.astype(np.float64) ** 2)))
                    if mic_arr.size
                    else 0.0
                ),
                "emitted_samples_left": len(self.left) // 2,
                "emitted_seconds_left": round(
                    (len(self.left) // 2) / float(AUD_RATE), 2
                ),
                "emitted_peak_left": (
                    int(np.max(np.abs(left_arr))) if left_arr.size else 0
                ),
                "control_samples": len(self.control) // 2,
                "cycles": len(self.cycle_rows),
                "windows_dumped": self.windows_dumped,
                "final_chunk_drift_ms": (
                    round(self.chunk_rows[-1]["drift_ms"], 1)
                    if self.chunk_rows
                    else 0.0
                ),
            }
            if extra:
                summary.update(extra)

            with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            return summary
        except Exception as e:
            return {"dump_error": str(e)}
