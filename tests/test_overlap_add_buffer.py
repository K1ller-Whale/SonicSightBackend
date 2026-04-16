import os
import sys

import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from overlap_add_buffer import OverlapAddBuffer


def _pcm16_to_float32(pcm_bytes):
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0


def test_overlap_add_reconstructs_with_explicit_start_samples():
    ola = OverlapAddBuffer(window_len=16, hop_samples=4)

    full_signal = np.linspace(-0.5, 0.5, 25, dtype=np.float32)
    start_samples = [0, 5, 9]

    reconstructed_chunks = []
    for start_sample in start_samples:
        ola.add_window(
            full_signal[start_sample : start_sample + ola.window_len],
            start_sample=start_sample,
        )
        reconstructed_chunks.append(_pcm16_to_float32(ola.drain()))

    reconstructed_chunks.append(_pcm16_to_float32(ola.flush()))
    reconstructed = np.concatenate(reconstructed_chunks)

    assert reconstructed.shape[0] == full_signal.shape[0]
    np.testing.assert_allclose(reconstructed, full_signal, atol=1e-4)
