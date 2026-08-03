import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import AUD_LEN, AUD_RATE, FRAME_RATE, NUM_FRAMES
from inference import StreamingBuffer
from model_registry import (
    DEFAULT_MODEL_ID,
    MODEL_METADATA_KEY,
    REGISTRY,
    get_engine,
)


# ---------------------------------------------------------------------------
# Registry / spec
# ---------------------------------------------------------------------------

def test_default_model_registered():
    assert DEFAULT_MODEL_ID == "sonicsight"
    assert DEFAULT_MODEL_ID in REGISTRY
    assert MODEL_METADATA_KEY == "sonicsight-model"


def test_sonicsight_spec_matches_pre_registry_constants():
    """The registry must reproduce the exact values the server used before
    the refactor — this is the 'no behavioural change' contract in numbers."""
    spec = REGISTRY["sonicsight"]
    assert spec.window_samples == AUD_LEN == 65536
    assert spec.model_sample_rate == AUD_RATE == 11025
    assert spec.capture_sample_rate == 11025
    assert spec.output_sample_rate == 11025
    assert spec.hop_samples == int(round(AUD_RATE / FRAME_RATE)) == 1378
    assert spec.num_frames == NUM_FRAMES == 3
    assert spec.frame_ring_cap == 60
    assert spec.early_min_samples == AUD_RATE  # 1 second, the old class attr
    assert spec.frame_selection == "centered_triple"
    assert spec.frame_kind == "left_right_halves"
    assert spec.audio_chunk_field == "audio_pcm"
    assert spec.heatmap_count == 2
    assert spec.stream_labels == ("Left", "Right")
    assert spec.confidence_gated is False


def test_spec_is_frozen():
    spec = REGISTRY["sonicsight"]
    with pytest.raises(Exception):
        spec.window_samples = 1234


def test_get_engine_cached_and_delegates_to_singleton():
    e1 = get_engine("sonicsight")
    e2 = get_engine("sonicsight")
    assert e1 is e2
    from inference import inference as engine_singleton

    # The adapter is plumbing only: same underlying callable, no wrapper math.
    assert e1.eval_stream_window.__func__.__qualname__.startswith("SonicSightEngine")
    assert e1.is_loaded == engine_singleton.is_loaded


# ---------------------------------------------------------------------------
# StreamingBuffer instance state
# ---------------------------------------------------------------------------

def test_buffer_defaults_match_pre_registry_behaviour():
    buf = StreamingBuffer()
    assert buf.target_audio_len == AUD_LEN
    assert buf.target_sr == AUD_RATE
    assert buf.num_frames == 3
    assert buf.frame_ring_cap == 60
    assert buf.early_min_samples == AUD_RATE
    assert buf.frame_selection == "centered_triple"
    # The old class attribute is gone: instance state only.
    assert "EARLY_INFERENCE_MIN_SAMPLES" not in StreamingBuffer.__dict__


def test_two_buffers_do_not_collide():
    a = StreamingBuffer()
    b = StreamingBuffer(
        target_audio_len=44144,
        target_sr=21000,
        num_frames=63,
        frame_ring_cap=90,
        early_min_samples=44144,
        frame_selection="consecutive_span",
    )
    assert a.early_min_samples == 11025
    assert b.early_min_samples == 44144
    assert a.num_frames == 3 and b.num_frames == 63
    assert a.frame_ring_cap == 60 and b.frame_ring_cap == 90


def _img(w=8, h=8, color=(0, 0, 0)):
    return Image.new("RGB", (w, h), color)


def _add_frames(buf, n, full=False, start_ms=0, step_ms=125):
    for i in range(n):
        ts = start_ms + i * step_ms
        if full:
            buf.add_decoded_frame(ts, None, None, _img(color=(i % 256, 0, 0)))
        else:
            buf.add_decoded_frame(ts, _img(color=(i % 256, 0, 0)), _img(color=(0, i % 256, 0)))


def test_ring_cap_eviction_respects_instance_value():
    small = StreamingBuffer(frame_ring_cap=5)
    _add_frames(small, 12)
    assert len(small.frame_buffer) == 5

    default = StreamingBuffer()
    _add_frames(default, 70)
    assert len(default.frame_buffer) == 60  # the pre-registry cap, unchanged


def test_frame_entries_carry_full_img_field():
    buf = StreamingBuffer()
    _add_frames(buf, 1)
    entry = buf.frame_buffer[0]
    assert set(entry.keys()) == {"timestamp_ms", "left_img", "right_img", "full_img"}
    assert entry["full_img"] is None  # sonicsight path stores no full frame


def test_centered_triple_selection_is_verbatim():
    buf = StreamingBuffer()
    _add_frames(buf, 5)
    frames, trim = buf._select_frames(2)
    left, right = frames
    assert len(left) == len(right) == 3
    assert left[0] is buf.frame_buffer[1]["left_img"]
    assert left[1] is buf.frame_buffer[2]["left_img"]
    assert left[2] is buf.frame_buffer[3]["left_img"]
    assert right[1] is buf.frame_buffer[2]["right_img"]
    assert trim == 1  # old code trimmed at valid_center_idx - 1


def test_consecutive_span_selection():
    buf = StreamingBuffer(num_frames=5, frame_selection="consecutive_span", frame_ring_cap=90)
    _add_frames(buf, 9, full=True)
    frames, trim = buf._select_frames(4)
    assert isinstance(frames, list) and len(frames) == 5
    assert frames[0] is buf.frame_buffer[2]["full_img"]  # centered: 4 - 5//2
    assert trim == 2

    # clamped at the buffer edges
    frames_lo, trim_lo = buf._select_frames(0)
    assert len(frames_lo) == 5 and trim_lo == 0
    frames_hi, trim_hi = buf._select_frames(8)
    assert len(frames_hi) == 5 and trim_hi == 4


def test_early_window_shape_unchanged_for_sonicsight_defaults():
    """1.5 s of audio + 3 frames -> early-mode window: 65536 samples,
    zero-padded, frames as (left[3], right[3]) — the pre-registry contract."""
    buf = StreamingBuffer()
    n_samples = int(AUD_RATE * 1.5)
    pcm = (np.zeros(n_samples, dtype=np.int16) + 100).tobytes()
    buf.add_audio_chunk(pcm, int(1500))
    _add_frames(buf, 3, start_ms=100)

    audio, frames, center_ts, start_sample = buf.get_latest_window()
    assert audio is not None
    assert len(audio) == AUD_LEN
    assert buf._last_window_mode == "early"
    left, right = frames
    assert len(left) == 3 and len(right) == 3


def test_early_mode_disabled_when_min_equals_window():
    buf = StreamingBuffer(
        target_audio_len=44144,
        target_sr=21000,
        num_frames=3,  # keep frame demands satisfiable in the test
        early_min_samples=44144,
        frame_selection="centered_triple",
    )
    pcm = np.zeros(21000, dtype=np.int16).tobytes()  # 1 s — below the window
    buf.add_audio_chunk(pcm, 1000)
    _add_frames(buf, 3)
    assert not buf.has_enough_data()  # no early inference for this profile
