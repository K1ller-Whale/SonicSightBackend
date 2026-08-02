import numpy as np

from config import AUD_LEN, AUD_RATE, FRAME_RATE


class OverlapAddBuffer:
    """
    Center-slice extraction with crossfade for streaming separation windows.

    TIMELINE-DRIVEN CONTRACT
    ------------------------
    The emitted stream is defined in ABSOLUTE sample coordinates, not by a
    fixed per-window hop. `_next_abs` always holds the absolute index of the
    next sample that must be emitted. Each window emits exactly the samples
    from `_next_abs` up to the end of its center region, so the total emitted
    count equals the real advance of the timeline, with no gaps and no
    duplicates, regardless of how far apart consecutive windows actually are.

    This matters because frame timestamps are jittery and are then quantized
    to the 256-sample STFT hop, so real window deltas are 1280 / 1536 / 1792
    and never equal int(AUD_RATE / FRAME_RATE) == 1378.

    The crossfade tail is HELD BACK and re-read from the next window, so the
    two blended segments cover the SAME absolute instants (a true crossfade
    between two independent estimates of that instant) instead of two
    adjacent, non-overlapping regions.
    """

    def __init__(self, window_len=AUD_LEN, hop_samples=None, crossfade_len=128):
        self.window_len = int(window_len)
        # Nominal hop sizes the center region and seeds the first window.
        # It is deliberately NOT used to advance the timeline.
        if hop_samples is None:
            hop_samples = AUD_RATE / FRAME_RATE
        self.hop_samples = int(round(hop_samples))
        self.crossfade_len = min(int(crossfade_len), self.hop_samples // 2)

        t = np.linspace(0, np.pi / 2, self.crossfade_len, dtype=np.float64)
        self.fade_in = np.sin(t) ** 2
        self.fade_out = np.cos(t) ** 2

        self._prev_tail = None
        self._tail_abs = None
        self._next_abs = None
        self._pcm_queue = []
        self.latest_window_start = None
        self.next_start_sample = 0

        self.total_emitted = 0
        self.gap_fills = 0
        self.splice_fallbacks = 0
        self.lagging_windows = 0

    def _center_start(self, center_offset):
        if center_offset is not None:
            return max(0, min(int(center_offset), self.window_len - self.hop_samples))
        return (self.window_len - self.hop_samples) // 2

    def add_window(self, audio_array, start_sample=None, center_offset=None):
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

        cs = self._center_start(center_offset)
        cf = self.crossfade_len

        emit_to_abs = start_sample + cs + self.hop_samples
        if self._next_abs is None:
            self._next_abs = start_sample + cs

        lo = self._next_abs - start_sample
        hi = emit_to_abs - start_sample

        if lo < 0:
            self.gap_fills += 1
            lo = 0
        if hi > self.window_len:
            hi = self.window_len
        if hi < lo:
            hi = lo

        if hi - lo <= cf:
            self.lagging_windows += 1
            self.latest_window_start = start_sample
            self.next_start_sample = start_sample + self.hop_samples
            return

        seg = audio_array[lo:hi].copy()

        if self._prev_tail is not None and cf > 0:
            if self._tail_abs - start_sample == lo:
                seg[:cf] = seg[:cf] * self.fade_in + self._prev_tail * self.fade_out
            else:
                self.splice_fallbacks += 1

        if cf > 0:
            self._prev_tail = seg[-cf:].copy()
            self._tail_abs = start_sample + hi - cf
            emit_slice = seg[:-cf]
            self._next_abs = self._tail_abs
        else:
            self._prev_tail = None
            self._tail_abs = None
            emit_slice = seg
            self._next_abs = start_sample + hi

        pcm_bytes = (
            np.round(np.clip(emit_slice, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        )
        self._pcm_queue.append(pcm_bytes)
        self.total_emitted += len(emit_slice)

        self.latest_window_start = start_sample
        self.next_start_sample = start_sample + self.hop_samples

    def drain(self):
        if not self._pcm_queue:
            return b""
        result = b"".join(self._pcm_queue)
        self._pcm_queue.clear()
        return result

    def flush(self):
        tail_pcm = b""
        if self._prev_tail is not None and len(self._prev_tail) > 0:
            faded_tail = self._prev_tail * self.fade_out
            tail_pcm = (
                np.round(np.clip(faded_tail, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            )
            self.total_emitted += len(self._prev_tail)
            self._prev_tail = None
            self._tail_abs = None
        queued = self.drain()
        return queued + tail_pcm
