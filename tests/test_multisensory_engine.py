"""Multisensory spec + TF-free engine helpers (CAM postprocess, resampling).

Everything here runs without TensorFlow or a checkpoint; the model-in-the-loop
validation (probe comparison) runs on the machine that has both.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from engines.multisensory_engine import (
    CAM_CONFIDENCE_MIN,
    HEATMAP_WIRE_SIDE,
    MODEL_SR,
    NUM_SAMPLES,
    VID_DUR,
    WIRE_SR,
    MultisensoryEngine,
    cam_confidence,
    default_multisensory_root,
    model_to_wire,
    normalize_cam,
    reduce_cam,
    upsample_cam,
    wire_to_model,
)
from model_registry import REGISTRY, loaded_model_ids


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

def test_multisensory_spec_registered_with_validated_constants():
    spec = REGISTRY["multisensory"]
    assert spec.frame_rate == 30
    assert spec.capture_sample_rate == WIRE_SR == 22050
    assert spec.model_sample_rate == MODEL_SR == 21000
    assert spec.frame_kind == "full_letterboxed"
    assert spec.audio_chunk_field == "audio_pcm_hi"
    # The LOCKED 2.135 s window, expressed at wire rate: engine converts to
    # exactly NUM_SAMPLES (44144) at 21000 Hz internally.
    assert spec.window_samples == 46352
    assert spec.hop_samples == 5512  # 250 ms hop — 125 ms cannot be sustained
    assert spec.num_frames == 63
    assert spec.frame_selection == "consecutive_span"
    assert spec.early_min_samples == spec.window_samples  # early mode disabled
    assert spec.heatmap_count == 1
    assert spec.stream_labels == ("On-screen", "Off-screen")
    assert spec.confidence_gated is True
    assert VID_DUR == 2.135
    assert CAM_CONFIDENCE_MIN == 0.10


def test_wire_window_matches_locked_model_window():
    # 46352 @ 22050 resampled by 20/21 covers >= 44144 @ 21000
    assert int(np.floor(46352 * 20 / 21)) + 1 >= NUM_SAMPLES


# ---------------------------------------------------------------------------
# Resampling (22050 <-> 21000, exact 21:20)
# ---------------------------------------------------------------------------

def test_wire_to_model_length_and_tone_preservation():
    t = np.arange(46352) / WIRE_SR
    tone = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    out = wire_to_model(tone)
    assert len(out) == NUM_SAMPLES
    # 440 Hz stays 440 Hz: peak bin of the resampled tone
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak_hz = np.argmax(spec) * MODEL_SR / len(out)
    assert abs(peak_hz - 440.0) < 2.0


def test_model_to_wire_roundtrip_shape_and_energy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(NUM_SAMPLES) * 0.1
    y = model_to_wire(x, 46352)
    assert len(y) == 46352
    # band-limited content survives the 20/21 up/down within a few percent
    assert np.sqrt(np.mean(y**2)) == pytest.approx(np.sqrt(np.mean(x**2)), rel=0.05)


# ---------------------------------------------------------------------------
# CAM postprocess — probe-identical semantics
# ---------------------------------------------------------------------------

def test_cam_confidence_is_raw_positive_fraction():
    cam = np.full((8, 7, 7), -1.0)
    assert cam_confidence(cam) == 0.0
    cam[:4] = 1.0  # half the cells positive
    assert cam_confidence(cam) == pytest.approx(0.5)


def test_reduce_cam_clamps_then_means_over_time():
    cam = np.zeros((8, 7, 7), dtype=np.float32)
    cam[0, 0, 0] = 8.0    # one hot cell in one timestep
    cam[1, 0, 0] = -8.0   # negative must NOT cancel it (that's what abs/raw would do)
    spatial = reduce_cam(cam)
    assert spatial.shape == (7, 7)
    assert spatial[0, 0] == pytest.approx(1.0)  # 8.0 clamped-neg-mean over 8 steps
    assert spatial.min() == 0.0


def test_normalize_cam_probe_divisor_and_ema_scale():
    spatial = np.array([[0.0, 2.0], [1.0, 4.0]], dtype=np.float32)
    n = normalize_cam(spatial)
    assert n.max() == pytest.approx(1.0, abs=1e-6)
    # EMA'd divisor changes brightness, not shape; values clipped to [0, 1]
    n2 = normalize_cam(spatial, scale=8.0)
    assert n2.max() == pytest.approx(0.5, abs=1e-6)
    assert np.all(normalize_cam(spatial, scale=1.0) <= 1.0)


def test_upsample_cam_shape_and_range():
    spatial = np.random.default_rng(1).random((7, 7)).astype(np.float32)
    up = upsample_cam(spatial)
    assert up.shape == (HEATMAP_WIRE_SIDE, HEATMAP_WIRE_SIDE)
    assert up.min() >= 0.0 and up.max() <= 1.0 + 1e-6
    # uint8 wire encode round-trips through the client's sqrt(size) inference
    wire = (up * 255).astype(np.uint8).tobytes()
    assert len(wire) == HEATMAP_WIRE_SIDE**2 == 3136


# ---------------------------------------------------------------------------
# Loading behaviour in an environment without TF / checkpoint
# ---------------------------------------------------------------------------

def test_load_fails_cleanly_without_checkpoint():
    root = default_multisensory_root()
    if not (root / "src" / "sep_video.py").exists():
        pytest.skip("multisensory checkout not present")
    if (root / "results" / "nets" / "sep" / "full" / "net.tf-160000.index").exists():
        pytest.skip("checkpoint present; this test is for the empty environment")
    with pytest.raises(FileNotFoundError):
        MultisensoryEngine().load()


def test_loaded_model_ids_empty_without_checkpoints():
    # Neither model can load here; the registry must say so instead of lying.
    assert loaded_model_ids() == []
