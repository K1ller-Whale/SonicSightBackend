"""Deterministic media source for the driver.

SyntheticSource generates procedural tone audio and moving-shape frames:
fully deterministic (seeded per session index), no external file, no ffmpeg.
Every scenario uses it. The one requirement that needs real content
(NFR-FUNC-001, separation sanity) shells out to the project's own
replay_client.py with an operator-supplied clip whose SHA-256 is recorded in
run metadata (see scenarios._replay_pearson and runmeta.collect).

Frame kinds mirror the client's two capture geometries: left/right halves
(sonicsight) and full letterboxed (sonicsight-pixel, multisensory).
"""

import io
import math

import numpy as np
from PIL import Image, ImageDraw

from . import paths

paths.import_stubs()
from config import AUD_RATE, IMG_SIZE, FRAME_RATE  # noqa: E402

JPEG_QUALITY = 90  # mirrors the client and replay_client


def _jpeg(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


class SyntheticSource:
    """Deterministic tones + moving rectangles.

    Audio: two tone stacks (left ~220/330 Hz, right ~554/831 Hz) with slow
    amplitude envelopes phase-offset between halves, at the wire rate. The
    content is irrelevant to load behaviour but must be non-silent (the
    server logs amplitude and the pixel silence gate reacts to silence) and
    deterministic across runs.
    """

    def __init__(self, seed=0, sample_rate=AUD_RATE, frame_rate=FRAME_RATE,
                 frame_kind="left_right_halves"):
        self.seed = seed
        self.sample_rate = int(sample_rate)
        self.frame_rate = float(frame_rate)
        self.frame_kind = frame_kind
        self._frame_cache = {}

    def audio_block(self, index, samples):
        t0 = index * samples / self.sample_rate
        t = t0 + np.arange(samples, dtype=np.float64) / self.sample_rate
        envL = 0.5 + 0.5 * np.sin(2 * math.pi * 0.25 * t + self.seed)
        envR = 0.5 + 0.5 * np.sin(2 * math.pi * 0.25 * t + self.seed + math.pi)
        left = envL * (np.sin(2 * math.pi * 220 * t) + 0.5 * np.sin(2 * math.pi * 330 * t))
        right = envR * (np.sin(2 * math.pi * 554 * t) + 0.5 * np.sin(2 * math.pi * 831 * t))
        mix = 0.35 * (left + right)
        return (np.clip(mix, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def _draw(self, phase, panel):
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (24, 24, 24))
        d = ImageDraw.Draw(img)
        x = int((0.15 + 0.6 * (0.5 + 0.5 * math.sin(phase))) * IMG_SIZE)
        y = IMG_SIZE // 2 + (40 if panel == "right" else -40)
        color = (200, 160, 40) if panel == "left" else (60, 160, 220)
        d.rectangle([x - 30, y - 30, x + 30, y + 30], fill=color)
        return img

    def frame(self, index):
        """Returns a dict shaped for the frame chunk of self.frame_kind."""
        key = index % 64  # cycle so long soaks reuse encodes
        if key in self._frame_cache:
            return self._frame_cache[key]
        phase = 2 * math.pi * key / 64 + self.seed
        if self.frame_kind == "left_right_halves":
            out = {"left_jpeg": _jpeg(self._draw(phase, "left")),
                   "right_jpeg": _jpeg(self._draw(phase, "right"))}
        else:  # full_letterboxed
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            inner = self._draw(phase, "left").resize((IMG_SIZE, IMG_SIZE * 9 // 16))
            img.paste(inner, (0, (IMG_SIZE - inner.height) // 2))
            out = {"full_jpeg": _jpeg(img)}
        self._frame_cache[key] = out
        return out

    def describe(self):
        return {"kind": "synthetic", "seed": self.seed,
                "sample_rate": self.sample_rate, "frame_rate": self.frame_rate,
                "frame_kind": self.frame_kind}
