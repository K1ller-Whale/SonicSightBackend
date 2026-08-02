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
    output = _pcm16_to_float64(ola.drain())

    # hop minus the tail held back for the next crossfade
    assert len(output) == 64 - 8, f"Expected 56 samples, got {len(output)}"
    np.testing.assert_allclose(output, 0.5, atol=1e-3)


def test_sinusoid_amplitude_preserved():
    ola = OverlapAddBuffer(window_len=1024, hop_samples=128, crossfade_len=16)

    t = np.arange(1024) / 11025.0
    signal = np.sin(2 * np.pi * 440.0 * t) * 0.8

    ola.add_window(signal, start_sample=0)
    output = _pcm16_to_float64(ola.drain())

    assert np.max(np.abs(output)) > 0.5, f"Signal too quiet: {np.max(np.abs(output))}"


def test_crossfade_smooths_discontinuity():
    ola = OverlapAddBuffer(window_len=256, hop_samples=64, crossfade_len=16)

    ola.add_window(np.ones(256) * 0.5, start_sample=0)
    ola.add_window(np.ones(256) * -0.5, start_sample=64)

    output = _pcm16_to_float64(ola.drain())
    max_jump = np.max(np.abs(np.diff(output)))
    assert max_jump < 0.15, f"Crossfade didn't smooth transition: {max_jump}"


def test_flush_emits_remaining_tail():
    ola = OverlapAddBuffer(window_len=256, hop_samples=64, crossfade_len=16)

    ola.add_window(np.ones(256) * 0.3, start_sample=0)
    ola.drain()

    flush_samples = len(ola.flush()) // 2
    assert flush_samples == 16, f"Expected 16 tail samples, got {flush_samples}"


def test_monotonic_start_sample_enforced():
    ola = OverlapAddBuffer(window_len=256, hop_samples=64, crossfade_len=8)
    ola.add_window(np.zeros(256), start_sample=100)
    try:
        ola.add_window(np.zeros(256), start_sample=50)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Regression tests for the timeline contract (Bug #1 / #2)
# ---------------------------------------------------------------------------

def test_no_sample_loss_ideal_cadence():
    """Total emitted audio must equal the real advance of the timeline.

    The previous implementation dropped `crossfade_len` samples per window
    (9.2% of all audio at the production config), which time-compressed and
    pitch-shifted the stream.
    """
    hop, cf, n = 64, 16, 5
    ola = OverlapAddBuffer(window_len=256, hop_samples=hop, crossfade_len=cf)

    for i in range(n):
        ola.add_window(np.random.randn(256) * 0.3, start_sample=i * hop)

    total = (len(ola.drain()) + len(ola.flush())) // 2
    assert total == n * hop, f"Expected {n * hop} samples, got {total}"


def test_no_sample_loss_jittered_cadence():
    """Real window deltas are 256-quantized and never equal hop_samples.

    Output length must track the ACTUAL timeline advance, not a constant hop.
    """
    hop, cf = 64, 16
    ola = OverlapAddBuffer(window_len=1024, hop_samples=hop, crossfade_len=cf)

    starts = [0, 48, 128, 176, 288, 320, 432]
    for s in starts:
        ola.add_window(np.random.randn(1024) * 0.3, start_sample=s)

    total = (len(ola.drain()) + len(ola.flush())) // 2
    cs = (1024 - hop) // 2
    expected = (starts[-1] + cs + hop) - (starts[0] + cs)
    assert total == expected, f"Expected {expected} samples, got {total}"
    assert ola.gap_fills == 0
    assert ola.splice_fallbacks == 0


def test_identity_mask_reconstructs_input():
    """With an identity mask the OLA must return the input essentially exactly.

    This is the decisive end-to-end guarantee: any timing or accounting defect
    in the reassembly layer shows up here as waveform error.
    """
    sr, hop, cf = 11025, 1378, 128
    wl = 65536
    n_total = sr * 12
    t = np.arange(n_total) / sr
    gt = 0.8 * np.sin(2 * np.pi * (200.0 * t + 45.0 * t**2))

    def window_at(start):
        w = np.zeros(wl)
        lo, hi = max(0, start), min(n_total, start + wl)
        if hi > lo:
            w[lo - start:hi - start] = gt[lo:hi]
        return w

    ola = OverlapAddBuffer(window_len=wl, hop_samples=hop, crossfade_len=cf)
    starts = [i * 1536 for i in range(40)]  # 256-quantized, != hop
    out = []
    for s in starts:
        ola.add_window(window_at(s), start_sample=s)
        out.append(ola.drain())
    out.append(ola.flush())

    y = _pcm16_to_float64(b"".join(out))
    cs = (wl - hop) // 2
    ref = gt[starts[0] + cs: starts[0] + cs + len(y)]

    # Exclude the intentional end-of-stream fade-out tail.
    peak_err = np.max(np.abs(y[:-cf] - ref[:-cf]))
    assert peak_err * 32767 <= 1.01, f"Not bit-exact: {peak_err * 32767:.3f} LSB"


def test_stall_is_reported_not_silently_swallowed():
    """A capture stall must surface as a gap_fill, not vanish."""
    hop, cf, wl = 64, 16, 1024
    ola = OverlapAddBuffer(window_len=wl, hop_samples=hop, crossfade_len=cf)
    ola.add_window(np.random.randn(wl), start_sample=0)
    ola.add_window(np.random.randn(wl), start_sample=5000)  # huge jump
    assert ola.gap_fills == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
