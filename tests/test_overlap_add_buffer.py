import os
import sys

import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from overlap_add_buffer import OverlapAddBuffer


def _pcm16_to_float64(pcm_bytes):
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64) / 32767.0


def test_center_slice_preserves_constant_signal():
    """A constant-valued window should produce a constant-valued center slice."""
    ola = OverlapAddBuffer(window_len=256, hop_samples=64, crossfade_len=8)

    signal = np.ones(256, dtype=np.float64) * 0.5

    ola.add_window(signal, start_sample=0)
    pcm = ola.drain()
    output = _pcm16_to_float64(pcm)

    # Center slice of a constant signal should be constant (minus the crossfade tail)
    expected_len = 64 - 8  # hop - crossfade tail held back
    assert len(output) == expected_len, f"Expected {expected_len} samples, got {len(output)}"
    np.testing.assert_allclose(output, 0.5, atol=1e-3)


def test_sinusoid_amplitude_preserved():
    """A sinusoid fed through center-slice should maintain its amplitude."""
    ola = OverlapAddBuffer(window_len=1024, hop_samples=128, crossfade_len=16)

    freq = 440.0
    sr = 11025.0
    t = np.arange(1024) / sr
    signal = np.sin(2 * np.pi * freq * t) * 0.8

    ola.add_window(signal, start_sample=0)
    pcm = ola.drain()
    output = _pcm16_to_float64(pcm)

    # Output should have amplitude close to 0.8 (within PCM16 quantization)
    assert np.max(np.abs(output)) > 0.5, f"Signal too quiet: max={np.max(np.abs(output))}"


def test_crossfade_smooths_discontinuity():
    """Two different signals should crossfade smoothly without a hard jump."""
    ola = OverlapAddBuffer(window_len=256, hop_samples=64, crossfade_len=16)

    signal_a = np.ones(256, dtype=np.float64) * 0.5
    signal_b = np.ones(256, dtype=np.float64) * -0.5

    ola.add_window(signal_a, start_sample=0)
    ola.add_window(signal_b, start_sample=64)

    pcm = ola.drain()
    output = _pcm16_to_float64(pcm)

    # There should be no discontinuity larger than the crossfade transition
    diff = np.abs(np.diff(output))
    max_jump = np.max(diff)
    assert max_jump < 0.15, f"Crossfade didn't smooth transition: max_jump={max_jump}"


def test_flush_emits_remaining_tail():
    """Flushing should emit the held-back crossfade tail."""
    ola = OverlapAddBuffer(window_len=256, hop_samples=64, crossfade_len=16)

    signal = np.ones(256, dtype=np.float64) * 0.3
    ola.add_window(signal, start_sample=0)

    drain_pcm = ola.drain()
    drain_samples = len(drain_pcm) // 2  # int16 = 2 bytes

    flush_pcm = ola.flush()
    flush_samples = len(flush_pcm) // 2

    # Flush should produce the 16-sample tail (faded out)
    assert flush_samples == 16, f"Expected 16 tail samples from flush, got {flush_samples}"


def test_monotonic_start_sample_enforced():
    """Adding a window with a start_sample before the last should raise."""
    ola = OverlapAddBuffer(window_len=256, hop_samples=64, crossfade_len=8)

    ola.add_window(np.zeros(256), start_sample=100)

    try:
        ola.add_window(np.zeros(256), start_sample=50)
        assert False, "Should have raised ValueError for non-monotonic start_sample"
    except ValueError:
        pass  # Expected


def test_multi_window_total_output_length():
    """Multiple windows should produce proportional output."""
    ola = OverlapAddBuffer(window_len=256, hop_samples=64, crossfade_len=16)

    for i in range(5):
        ola.add_window(np.random.randn(256) * 0.3, start_sample=i * 64)

    drain_pcm = ola.drain()
    flush_pcm = ola.flush()
    total_samples = (len(drain_pcm) + len(flush_pcm)) // 2

    # Each window emits (hop - crossfade) = 48 samples via drain,
    # except the tail from the last window is held back for flush.
    # Total = 5 * 48 + 16 (flush tail) = 256
    expected = 5 * (64 - 16) + 16
    assert total_samples == expected, f"Expected {expected} total samples, got {total_samples}"
