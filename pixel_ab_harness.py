"""Offline A/B harness for the pixel-mode design questions (PIXEL_PLAN 5, B1, B2).

Runs on the machine with the checkpoints (GPU or patient CPU). Three
comparisons from one fixed clip, no server needed:

  A/B  halves-from-crops (the deployed path) vs halves-from-region-pooling
       (left/right content-column regions against the full-frame cache).
       NOTE the writeup rule: the deployed halves path is video-pooled over
       half-CROPS; region pooling sees a letterboxed FULL frame — pooling
       rule is held equal (max), but the pixels differ. Say so in any report.
  B1   the same tap rendered through ratio+renorm AND binary masks — the
       perceptual separation-vs-mixer-coherence tradeoff, for your ears.
  B2   the linear-domain energy map vs the validated warped activation map,
       side by side as PNGs + npy.

Usage:
    python pixel_ab_harness.py --video clip.mp4 --start 5 --out ab_out
    python pixel_ab_harness.py --video clip.mp4 --start 5 --tap 0.3,0.5

Outputs in --out: mixture.wav, A_left/right.wav, B_left/right.wav,
tap_ratio/binary(.complement).wav when --tap is given, energy_map.png/.npy,
activation_map.png/.npy, report.txt with RMS + A-vs-B correlations.

Do not change any default on the strength of this without a human listen.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from PIL import Image
import soundfile as sf
import torch

from config import AUD_LEN, AUD_RATE, FRAME_RATE
from inference import inference
from pixel_cache import (
    GRID_H,
    GRID_W,
    WindowCache,
    cell_analysis,
    region_disc,
    synthesize_regions,
)


def extract(video, start, seconds, tmp):
    """Frames at 8 fps 1280x720 + mono 11025 Hz wav, like the live capture."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(start), "-i", video,
         "-t", str(seconds), "-r", str(FRAME_RATE), "-vf", "scale=1280:720",
         os.path.join(tmp, "f_%04d.png")],
        check=True,
    )
    wav = os.path.join(tmp, "a.wav")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(start), "-i", video,
         "-t", str(seconds), "-ac", "1", "-ar", str(AUD_RATE), wav],
        check=True,
    )
    frames = sorted(
        Image.open(os.path.join(tmp, f)).convert("RGB")
        for f in os.listdir(tmp) if f.startswith("f_")
    )
    audio, _ = sf.read(wav, dtype="float32")
    return frames, audio


def client_halves(frame):
    """Mirror of ImageTransform.processAndCompressHalves (PIL bilinear —
    close to, not bit-identical with, Android's scaler; the A/B compares
    pipelines from the same source, which is the honest comparison)."""
    w, h = frame.size
    halves = []
    for box in [(0, 0, w // 2, h), (w // 2, 0, w, h)]:
        half = frame.crop(box)
        hw, hh = half.size
        if hw < hh:
            sw, sh = 256, int(256.0 / hw * hh)
        else:
            sh, sw = 256, int(256.0 / hh * hw)
        half = half.resize((sw, sh), Image.BILINEAR)
        left = (sw - 224) // 2
        top = (sh - 224) // 2
        halves.append(half.crop((left, top, left + 224, top + 224)))
    return halves  # [left224, right224]


def client_letterbox(frame, dim=224):
    """Mirror of ImageTransform.letterboxAndCompress."""
    w, h = frame.size
    s = dim / max(w, h)
    nw, nh = round(w * s), round(h * s)
    scaled = frame.resize((nw, nh), Image.BILINEAR)
    out = Image.new("RGB", (dim, dim), (128, 128, 128))
    out.paste(scaled, ((dim - nw) // 2, (dim - nh) // 2))
    return out


def content_rows():
    """Grid rows covered by 16:9 content in the 224 letterbox (49 px bars)."""
    top = int(49 / 16)          # 3
    bottom = int((224 - 49) / 16)  # 10 (inclusive)
    return top, bottom


def halves_region_masks():
    top, bottom = content_rows()
    left = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    right = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    left[top:bottom + 1, : GRID_W // 2] = 1.0
    right[top:bottom + 1, GRID_W // 2:] = 1.0
    return left, right


def save_map_png(arr, path):
    a = arr.astype(np.float32)
    a = a / max(float(a.max()), 1e-12)
    Image.fromarray((a * 255).astype(np.uint8), "L").resize(
        (GRID_W * 24, GRID_H * 24), Image.NEAREST
    ).save(path)


def rms(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


def corr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--out", default="ab_out")
    p.add_argument("--tap", default=None,
                   help="x,y in normalized letterbox coords for the B1 listen")
    arg = p.parse_args()

    os.makedirs(arg.out, exist_ok=True)
    inference.load_model()

    # One centered window: AUD_LEN needs ~6 s; extract 8 with margin.
    seconds = AUD_LEN / AUD_RATE + 2.0
    tmp = tempfile.mkdtemp(prefix="pixel_ab_")
    try:
        frames, audio = extract(arg.video, arg.start, seconds, tmp)
        if len(audio) < AUD_LEN or len(frames) < 3:
            sys.exit("clip too short after --start for one full window")
        audio_win = audio[:AUD_LEN]
        center = min(len(frames) - 2, int((AUD_LEN / AUD_RATE / 2) * FRAME_RATE))
        triple = [frames[center - 1], frames[center], frames[center + 1]]

        sf.write(os.path.join(arg.out, "mixture.wav"), audio_win, AUD_RATE)

        # ── A: deployed halves (crops) ──
        left3 = []; right3 = []
        for f in triple:
            l, r = client_halves(f)
            left3.append(l); right3.append(r)
        res_a = inference.eval_stream_window(audio_win.copy(), (left3, right3))
        sf.write(os.path.join(arg.out, "A_left.wav"), res_a["left_audio"], AUD_RATE)
        sf.write(os.path.join(arg.out, "A_right.wav"), res_a["right_audio"], AUD_RATE)

        # ── B: region-pooled halves from the full-frame cache ──
        full3 = [client_letterbox(f) for f in triple]
        feats = inference.eval_pixel_window(audio_win.copy(), full3)
        cache = WindowCache(
            window_id=1, feat_sound=feats["feat_sound"],
            feats_img=feats["feats_img"], mag_lin=feats["mag_lin"],
            phase=feats["phase"], audio_gain=feats["audio_gain"],
            start_sample=0, center_timestamp_ms=0,
        )
        lm, rm = halves_region_masks()
        wavs, energies = synthesize_regions(
            inference, cache, [torch.from_numpy(lm), torch.from_numpy(rm)]
        )
        sf.write(os.path.join(arg.out, "B_left.wav"), wavs[0], AUD_RATE)
        sf.write(os.path.join(arg.out, "B_right.wav"), wavs[1], AUD_RATE)

        # ── B2: both cell maps ──
        energy, activation, _ = cell_analysis(inference, cache)
        np.save(os.path.join(arg.out, "energy_map.npy"), energy)
        np.save(os.path.join(arg.out, "activation_map.npy"), activation)
        save_map_png(energy, os.path.join(arg.out, "energy_map.png"))
        save_map_png(activation, os.path.join(arg.out, "activation_map.png"))

        # ── B1: one tap, both mask variants ──
        tap_lines = []
        if arg.tap:
            x, y = (float(v) for v in arg.tap.split(","))
            w = torch.from_numpy(region_disc(x, y, 1.0 / GRID_W))
            for name, binary in (("ratio", False), ("binary", True)):
                tw, te = synthesize_regions(
                    inference, cache, [w, 1.0 - w], binary=binary
                )
                sf.write(os.path.join(arg.out, f"tap_{name}.wav"), tw[0], AUD_RATE)
                sf.write(os.path.join(arg.out, f"tap_{name}_complement.wav"), tw[1], AUD_RATE)
                tap_lines.append(f"tap {name:6s}: region rms {rms(tw[0]):.5f}  "
                                 f"complement rms {rms(tw[1]):.5f}  energy {te[0]:.4g}")

        report = [
            f"clip={arg.video} start={arg.start}",
            f"A (deployed crops):  left rms {rms(res_a['left_audio']):.5f}  "
            f"right rms {rms(res_a['right_audio']):.5f}",
            f"B (region-pooled):   left rms {rms(wavs[0]):.5f}  "
            f"right rms {rms(wavs[1]):.5f}   energies {energies[0]:.4g}/{energies[1]:.4g}",
            f"A-vs-B correlation:  left {corr(res_a['left_audio'], wavs[0]):+.4f}  "
            f"right {corr(res_a['right_audio'], wavs[1]):+.4f}",
            "",
            "Caveat for any conclusion: A sees half-CROPS (87.5%x77.8% of each",
            "half at crop scale), B sees the letterboxed FULL frame — pooling",
            "rule is equal (max), the pixels are not. Listen before judging;",
            "do not change the default mode on numbers alone.",
            *tap_lines,
        ]
        with open(os.path.join(arg.out, "report.txt"), "w") as f:
            f.write("\n".join(report) + "\n")
        print("\n".join(report))
        print(f"\nwavs + maps in {arg.out}/")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
