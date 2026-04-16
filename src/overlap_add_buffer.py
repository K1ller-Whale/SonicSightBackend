import numpy as np

from config import AUD_LEN, AUD_RATE, FRAME_RATE


class OverlapAddBuffer:
    """
    Overlap-add reconstruction for streaming separation windows.

    Windows may be written at explicit absolute sample positions so the stream
    stays sample-accurate even when consecutive inference windows do not land on
    a perfectly fixed hop.
    """

    def __init__(self, window_len=AUD_LEN, hop_samples=int(AUD_RATE / FRAME_RATE)):
        self.window_len = int(window_len)
        self.hop_samples = int(hop_samples)

        initial_capacity = self.window_len * 10
        self.acc = np.zeros(initial_capacity, dtype=np.float64)
        self.norm = np.zeros(initial_capacity, dtype=np.float64)

        # At SonicSight's tiny streaming hop, rectangular synthesis is the
        # correct OLA window. A Hann taper would create deep gain valleys.
        self.synthesis_window = np.ones(self.window_len, dtype=np.float64)

        self.buffer_offset = 0
        self.next_start_sample = 0
        self.read_pos = 0
        self.latest_window_start = None
        self.max_written_end = 0

    def add_window(self, audio_array, start_sample=None):
        """Add one full inference window at an absolute sample position."""
        audio_array = np.asarray(audio_array, dtype=np.float64)

        if len(audio_array) > self.window_len:
            audio_array = audio_array[: self.window_len]
        elif len(audio_array) < self.window_len:
            audio_array = np.pad(audio_array, (0, self.window_len - len(audio_array)))

        if start_sample is None:
            start_sample = self.next_start_sample

        start_sample = int(start_sample)
        if self.latest_window_start is not None and start_sample < self.latest_window_start:
            raise ValueError("OLA windows must be added in non-decreasing sample order.")
        if start_sample < self.buffer_offset:
            raise ValueError("Cannot add audio before the compacted OLA buffer start.")

        end_sample = start_sample + self.window_len
        self._expand_buffers(end_sample)

        start_idx = start_sample - self.buffer_offset
        end_idx = start_idx + self.window_len

        self.acc[start_idx:end_idx] += audio_array * self.synthesis_window
        self.norm[start_idx:end_idx] += self.synthesis_window

        self.latest_window_start = start_sample
        self.max_written_end = max(self.max_written_end, end_sample)
        self.next_start_sample = start_sample + self.hop_samples

    def drain(self):
        """
        Drain the committed PCM16 region.

        Once a new window arrives, everything strictly before that window's
        start sample is final and can be emitted safely.
        """
        if self.latest_window_start is None:
            return b""
        return self._drain_until(self.latest_window_start)

    def flush(self):
        """Drain the remaining tail after the stream ends."""
        return self._drain_until(self.max_written_end)

    def _drain_until(self, drain_pos):
        drain_pos = int(drain_pos)
        if self.read_pos >= drain_pos:
            return b""

        start_idx = self.read_pos - self.buffer_offset
        end_idx = drain_pos - self.buffer_offset

        chunk_acc = self.acc[start_idx:end_idx]
        chunk_norm = self.norm[start_idx:end_idx]

        safe_norm = np.where(chunk_norm > 1e-10, chunk_norm, 1.0)
        normalized_audio = chunk_acc / safe_norm
        pcm_bytes = (
            (np.clip(normalized_audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        )

        self.read_pos = drain_pos
        self._compact()
        return pcm_bytes

    def _expand_buffers(self, min_end_sample):
        required_size = min_end_sample - self.buffer_offset
        if required_size <= len(self.acc):
            return

        new_size = len(self.acc)
        while required_size > new_size:
            new_size *= 2

        new_acc = np.zeros(new_size, dtype=np.float64)
        new_norm = np.zeros(new_size, dtype=np.float64)
        new_acc[: len(self.acc)] = self.acc
        new_norm[: len(self.norm)] = self.norm

        self.acc = new_acc
        self.norm = new_norm

    def _compact(self):
        consumed = self.read_pos - self.buffer_offset
        if consumed <= 0:
            return

        threshold = max(self.window_len, len(self.acc) // 4)
        if consumed < threshold:
            return

        valid_len = len(self.acc) - consumed
        self.acc[:valid_len] = self.acc[consumed:]
        self.acc[valid_len:] = 0.0

        self.norm[:valid_len] = self.norm[consumed:]
        self.norm[valid_len:] = 0.0

        self.buffer_offset += consumed
