import numpy as np

from config import AUD_LEN, AUD_RATE, FRAME_RATE


class OverlapAddBuffer:
    """
    Center-slice extraction with crossfade for streaming separation windows.

    Instead of averaging the full 65536-sample inference window across ~47
    overlapping calls (which destroys separation quality by blending masks
    from different video frames), we extract only the center `hop_samples`
    from each window — the region most accurately described by the current
    visual context.

    A short cosine crossfade at the splice boundary prevents audible clicks.
    """

    def __init__(
        self,
        window_len: int = AUD_LEN,
        hop_samples: int = int(AUD_RATE / FRAME_RATE),
        crossfade_len: int = 128,
    ):
        self.window_len = int(window_len)
        self.hop_samples = int(hop_samples)
        self.crossfade_len = min(int(crossfade_len), self.hop_samples // 2)

        # Pre-compute crossfade ramps (half-cosine curves)
        t = np.linspace(0, np.pi / 2, self.crossfade_len, dtype=np.float64)
        self.fade_in = np.sin(t) ** 2   # 0 → 1
        self.fade_out = np.cos(t) ** 2   # 1 → 0

        # State
        self._prev_tail: np.ndarray | None = None  # tail of previous slice for crossfade
        self._pcm_queue: list[bytes] = []
        self.latest_window_start: int | None = None
        self.next_start_sample: int = 0

    def add_window(self, audio_array, start_sample=None):
        """Add one full inference window; extract the center hop slice."""
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

        # ── Extract center slice ──
        center_start = (self.window_len - self.hop_samples) // 2
        center_end = center_start + self.hop_samples
        raw_slice = audio_array[center_start:center_end].copy()

        cf = self.crossfade_len

        # ── Crossfade with previous tail ──
        if self._prev_tail is not None and cf > 0:
            raw_slice[:cf] = (
                raw_slice[:cf] * self.fade_in
                + self._prev_tail * self.fade_out
            )

        # Save this slice's tail for the NEXT crossfade
        if cf > 0:
            self._prev_tail = raw_slice[-cf:].copy()
            # Emit everything EXCEPT the tail (it will be blended into the next slice)
            emit_slice = raw_slice[:-cf]
        else:
            self._prev_tail = None
            emit_slice = raw_slice

        # Convert to PCM16 and queue
        pcm_bytes = (
            (np.clip(emit_slice, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        )
        self._pcm_queue.append(pcm_bytes)

        self.latest_window_start = start_sample
        self.next_start_sample = start_sample + self.hop_samples

    def drain(self):
        """
        Return all queued PCM16 bytes and clear the queue.

        Each call to `add_window` queues one center-slice worth of audio.
        The caller should send the drained bytes to the client immediately.
        """
        if not self._pcm_queue:
            return b""
        result = b"".join(self._pcm_queue)
        self._pcm_queue.clear()
        return result

    def flush(self):
        """Drain the remaining crossfade tail after the stream ends."""
        tail_pcm = b""
        if self._prev_tail is not None and len(self._prev_tail) > 0:
            # Fade out the final tail to silence
            faded_tail = self._prev_tail * self.fade_out
            tail_pcm = (
                (np.clip(faded_tail, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            )
            self._prev_tail = None
        queued = self.drain()
        return queued + tail_pcm
