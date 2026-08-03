"""Multisensory engine adapter (Owens & Efros, arXiv 1804.03641v2).

One TF forward pass produces both outputs: fg/bg separated speech from the
u-net head and the alignment CAM from joint/logits. The two audio streams
mean ON-SCREEN / OFF-SCREEN (not left/right); labels come from the spec.

TensorFlow is imported only inside load(), so this module (and the CAM /
resampling helpers below, which are unit-tested without a checkpoint) stays
importable on machines without TF. The multisensory repo location and
checkpoint come from MULTISENSORY_ROOT / MULTISENSORY_CHECKPOINT, defaulting
to the sibling checkout layout used by this project.
"""

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import resample_poly

from engines import EngineAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validated constants. LOCKED — these reproduced the measurement record; both
# --vid_dur and --cam_hires produced confident, plausible, WRONG output when
# deviated from (HANDOFF 8.1, 8.2, 11.5). Not tunables.
# ---------------------------------------------------------------------------

# The analysis window. Enters the graph as
# spec_len = 128 * 2**round(log2(vid_dur / 2.135)), so only power-of-two
# multiples are even coherent; at 4.27 s the decisive mask test flipped from
# PASS (+0.8248) to a confident-looking FAIL (+0.0924) on the identical clip.
VID_DUR = 2.135

MODEL_SR = 21000       # rate the checkpoint was trained at
WIRE_SR = 22050        # capture/playback rate on the wire (ratio 21:20 exactly)
NUM_SAMPLES = 44144    # model window length @ MODEL_SR
NUM_FRAMES = 63        # sampled_frames for the 2.135 s window

# joint/logits is an ALIGNMENT head the separation loss never touches
# (HANDOFF 8.3). When the raw CAM's positive fraction is below this, the map
# means "this audio does not match this video" and must not be rendered as a
# localization. Separated audio is unaffected.
CAM_CONFIDENCE_MIN = 0.10

# Wire format shared with the Sound of Pixels path: square uint8 grid, side
# inferred client-side from sqrt(byte count). 56x56 matches the existing pipe.
HEATMAP_WIRE_SIDE = 56

# Input level the net expects; sep_video.main() and sep_cam_probe set the
# same value outside sep_params.
INPUT_RMS = float(np.sqrt(0.1 ** 2 + 0.1 ** 2))

# EMA weight for the display-normalisation divisor only (see normalize_cam).
CAM_SCALE_EMA = 0.3


# ---------------------------------------------------------------------------
# CAM post-processing — probe-identical (cam_analyze.reduce_positive), so a
# live heatmap stays directly comparable to the validated sep_cam_probe /
# cam_analyze output. TF-free and unit-tested.
# ---------------------------------------------------------------------------

def cam_confidence(cam):
    """Positive fraction of the RAW signed CAM — cam_analyze's pos_frac."""
    cam = np.asarray(cam)
    return float((cam > 0).mean())


def reduce_cam(cam):
    """Time reduction + sign handling, exactly cam_analyze.reduce_positive:
    clamp negatives to zero, then MEAN over the 8 time steps.

    Mean (not the centre slice): it is the reduction every validated number
    in the handoff was computed on, and it is steadier at the 4 Hz update
    rate. The 8 steps span only 2.1 s, so the centre slice buys almost no
    temporal precision in exchange for losing comparability.
    Clamp (not abs): the CAM is signed; abs conflates evidence AGAINST
    alignment with evidence for it. The all-negative case is handled by the
    confidence gate, computed on the raw CAM before this clamp.
    """
    cam = np.asarray(cam, dtype=np.float32)
    return np.maximum(cam, 0.0).mean(axis=0)


def normalize_cam(spatial, scale=None):
    """Per-window max-normalisation, the probe's exact choice
    (norm / (norm.max() + 1e-8); minimum is already 0 after the clamp).

    `scale` optionally substitutes an EMA-smoothed running max as the
    divisor: same map shape, steadier brightness between windows. Raw
    per-window values stay in diag so cam_analyze comparisons remain exact.
    """
    spatial = np.asarray(spatial, dtype=np.float32)
    denom = float(spatial.max() if scale is None else scale) + 1e-8
    return np.clip(spatial / denom, 0.0, 1.0)


def upsample_cam(spatial, side=HEATMAP_WIRE_SIDE):
    """7x7 -> 56x56 bilinear, matching the wire format the client already
    decodes (square uint8 grid, side inferred from byte count)."""
    return cv2.resize(
        np.asarray(spatial, dtype=np.float32), (side, side), interpolation=cv2.INTER_LINEAR
    )


# ---------------------------------------------------------------------------
# Wire-rate <-> model-rate resampling (22050 <-> 21000, exact 21:20 ratio)
# ---------------------------------------------------------------------------

def _fit_length(x, n):
    x = np.asarray(x)
    if len(x) > n:
        return x[:n]
    if len(x) < n:
        return np.pad(x, (0, n - len(x)))
    return x


def wire_to_model(x):
    """22050 Hz -> 21000 Hz, fitted to the model's window length."""
    return _fit_length(resample_poly(np.asarray(x, dtype=np.float64), 20, 21), NUM_SAMPLES)


def model_to_wire(x, n_out):
    """21000 Hz -> 22050 Hz, fitted to the wire window length."""
    return _fit_length(resample_poly(np.asarray(x, dtype=np.float64), 21, 20), n_out)


def default_multisensory_root():
    # SonicSight/SonicSightBackend/src/engines/ -> SonicSight/multisensory
    return Path(__file__).resolve().parents[3] / "multisensory"


class MultisensoryEngine(EngineAdapter):
    id = "multisensory"

    def __init__(self):
        self._net = None
        self._pr = None
        self._device = "not loaded"
        self._cam_scale_ema = None

    @property
    def is_loaded(self):
        return self._net is not None

    @property
    def device(self):
        return self._device

    def load(self):
        if self._net is not None:
            return

        root = Path(os.environ.get("MULTISENSORY_ROOT", "") or default_multisensory_root())
        src = root / "src"
        if not (src / "sep_video.py").exists():
            raise FileNotFoundError(
                f"multisensory repo not found at {root} (set MULTISENSORY_ROOT)"
            )
        ckpt = os.environ.get("MULTISENSORY_CHECKPOINT", "") or str(
            root / "results" / "nets" / "sep" / "full" / "net.tf-160000"
        )
        if not Path(str(ckpt) + ".index").exists():
            raise FileNotFoundError(f"multisensory checkpoint not found: {ckpt}")

        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        # Heavy imports deferred to here: sep_video sets
        # TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512 at module level (required — the
        # uncapped BFC pool holds 2.9 GB for a 477 MB working set) and builds
        # its session from MS_GPU_ALLOW_GROWTH / MS_GPU_MEM_FRACTION, which
        # is what lets PyTorch share the card.
        import sep_params
        import sep_video

        pr = sep_params.full()  # vid_dur defaults to the LOCKED 2.135 s
        if abs(pr.vid_dur - VID_DUR) > 1e-9 or pr.num_samples != NUM_SAMPLES:
            raise RuntimeError(
                f"sep_params.full() drifted: vid_dur={pr.vid_dur} "
                f"num_samples={pr.num_samples}; expected {VID_DUR}/{NUM_SAMPLES}"
            )
        if pr.cam:
            # pr.cam=True is the 14x14 CAM: 0.0684 correlation with the
            # validated 7x7, plus latency. Never ship it (HANDOFF 8.2).
            raise RuntimeError("pr.cam must be False (7x7 CAM is the validated one)")
        pr.input_rms = INPUT_RMS
        pr.model_path = str(ckpt)

        # MS_GPU: integer GPU index (default 0); empty/"cpu" -> unpinned (CPU).
        gpu_env = os.environ.get("MS_GPU", "0").strip().lower()
        gpu = None if gpu_env in ("", "cpu", "none") else f"/gpu:{int(gpu_env)}"

        net = sep_video.NetClf(pr, gpu=gpu)
        net.init()
        if net.cam_op is None:
            raise RuntimeError(
                f"No CAM tensor in the graph (net_style={pr.net_style!r}); need 'full'."
            )

        self._net = net
        self._pr = pr
        self._device = gpu or "cpu"
        logger.info(
            "Multisensory loaded: ckpt=%s device=%s window=%.3fs (%d samples @ %d Hz)",
            ckpt, self._device, VID_DUR, NUM_SAMPLES, MODEL_SR,
        )

    def eval_stream_window(self, audio_window, frames):
        """audio_window: float32 mono at WIRE_SR (22050), one spec window.
        frames: list of NUM_FRAMES PIL RGB images, 224x224 letterboxed.
        """
        if any(f is None for f in frames):
            raise ValueError(
                "multisensory stream received frames without full_img — "
                "client is sending the wrong capture profile"
            )
        ims = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
        if ims.shape != (NUM_FRAMES, 224, 224, 3):
            raise ValueError(f"expected ({NUM_FRAMES}, 224, 224, 3) frames, got {ims.shape}")

        # Mirror sep_cam_probe.load_audio: mono duplicated to 2 channels,
        # normalised to pr.input_rms (normalize_rms_np semantics, eps 1e-4).
        # The gain is undone on the output so playback matches mic level.
        mono = wire_to_model(audio_window)
        rms = max(1e-4, float(np.sqrt(np.mean(np.square(mono)))))
        gain = INPUT_RMS / rms
        samples = np.stack([mono, mono], axis=1).astype(np.float32) * gain

        ret = self._net.predict_with_cam(ims[None], samples[None])

        n_out = len(audio_window)
        fg = model_to_wire(np.asarray(ret["samples_pred_fg"][0], dtype=np.float64) / gain, n_out)
        bg = model_to_wire(np.asarray(ret["samples_pred_bg"][0], dtype=np.float64) / gain, n_out)

        cam = np.asarray(ret["cam"])  # (8, 7, 7), signed, near-zero mean
        confidence = cam_confidence(cam)
        spatial = reduce_cam(cam)
        raw_max = float(spatial.max())

        if confidence < CAM_CONFIDENCE_MIN:
            # Meaningless map ("this audio does not match this video") — do
            # not render it as a localization, and do not let its max poison
            # the display-scale EMA. Audio still plays.
            heatmap = None
        else:
            if self._cam_scale_ema is None:
                self._cam_scale_ema = raw_max
            else:
                self._cam_scale_ema = (
                    CAM_SCALE_EMA * raw_max + (1.0 - CAM_SCALE_EMA) * self._cam_scale_ema
                )
            heatmap = upsample_cam(normalize_cam(spatial, scale=self._cam_scale_ema))

        # NOTE: no std/|mean| "contrast" anywhere — on a near-zero-mean signed
        # CAM it is an artifact, not a signal (HANDOFF 8.4).
        diag = {
            "audio_gain": float(gain),
            "cam_pos_frac": confidence,
            "cam_raw_min": float(cam.min()),
            "cam_raw_max": float(cam.max()),
            "cam_raw_std": float(cam.std()),
            "cam_spatial_max_raw": raw_max,   # probe-comparable, pre-EMA
            "cam_scale_ema": float(self._cam_scale_ema) if self._cam_scale_ema is not None else 0.0,
        }

        return {
            "left_audio": np.clip(fg, -1.0, 1.0).astype(np.float32),   # on-screen
            "right_audio": np.clip(bg, -1.0, 1.0).astype(np.float32),  # off-screen
            "left_heatmap": heatmap,
            "right_heatmap": None,
            "diag": diag,
            "confidence": confidence,
        }
