"""Pixel-mode window cache and the region-synthesis primitive.

Architecture (PIXEL_PLAN.md section 2): the two expensive networks run once
per window; everything a spatial selection needs afterwards is cached, and any
region's audio is one InnerProd bmm + sigmoid + inverse log-frequency resample
+ ISTFT against that cache. A tap never costs a network forward pass.

All cached tensors are float32 regardless of AMP — the existing pixelwise
paths force fp32 for numeric stability, and queries must match the streamed
math exactly.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    AUD_LEN,
    BINARY_MASK,
    MASK_RENORM,
    MASK_RENORM_EPS,
    MASK_RENORM_FLOOR,
    MASK_THRES,
    STFT_FRAME,
    STFT_HOP,
)

logger = logging.getLogger(__name__)

# The model's native visual feature grid for a 224x224 input: stride-16
# dilated ResNet-18 (vision_net.py dilate_scale=16) => 14x14. Verified, not
# assumed — see PIXEL_PLAN.md section 1.
GRID_W = 14
GRID_H = 14

# A region's "support" for max-pooling: cells whose weight exceeds this.
# Max (not weighted mean) because feats_img is post-sigmoid and sigmoid is
# monotonic, so region max-pooling reproduces the deployed video-level
# pooling restricted to the region (review amendment A1).
REGION_SUPPORT_THRESHOLD = 0.5

# EMA for the wire energy map (display stability; raw per-window values stay
# in the cache). PROVISIONAL 0.4 — to be tuned by eye on a device per
# PIXEL_PLAN section 2.3; the number and its reasoning get updated then.
ENERGY_EMA = 0.4


@dataclass
class WindowCache:
    """Everything needed to answer any spatial query about one window."""

    window_id: int
    feat_sound: torch.Tensor      # [1, K, 256, 256] fp32 — s_k, warped log-freq domain
    feats_img: torch.Tensor       # [1, K, 14, 14] fp32 — i_k, post-sigmoid, temporal-maxed
    mag_lin: torch.Tensor         # [512, 256] fp32 — linear-frequency mixture magnitude
    phase: torch.Tensor           # [512, 256] fp32 — mixture phase
    audio_gain: float             # input normalisation gain (undone on synthesis)
    start_sample: int             # absolute wire-domain start of the window
    center_timestamp_ms: int
    pinned: bool = False

    def mixture_energy(self) -> float:
        """Total linear-domain energy of the window's mixture (gain undone).
        The denominator for RELATIVE region energies: raw sums over 512x256
        bins are mic-level and window-length dependent, useless as an
        absolute client-side threshold (review finding)."""
        return float(((self.mag_lin / self.audio_gain) ** 2).sum().item())

    def bytes_estimate(self) -> int:
        return sum(
            t.element_size() * t.nelement()
            for t in (self.feat_sound, self.feats_img, self.mag_lin, self.phase)
        )


class WindowCacheRing:
    """Small ring of recent windows. A pinned (frozen) window is exempt from
    eviction — the server keeps advancing; the pin only protects the cache
    entry (review amendment A2)."""

    def __init__(self, capacity: int = 3):
        self.capacity = int(capacity)
        self._entries: List[WindowCache] = []
        self._next_id = 1

    def push(self, cache: WindowCache) -> int:
        cache.window_id = self._next_id
        self._next_id += 1
        self._entries.append(cache)
        # Evict oldest unpinned entries beyond capacity (pinned ones ride along
        # as at most one extra slot each).
        unpinned = [c for c in self._entries if not c.pinned]
        while len(unpinned) > self.capacity:
            victim = unpinned.pop(0)
            self._entries.remove(victim)
        logger.debug(
            "pixel cache: %d windows resident, ~%.1f MB",
            len(self._entries),
            sum(c.bytes_estimate() for c in self._entries) / 1e6,
        )
        return cache.window_id

    def newest(self) -> Optional[WindowCache]:
        return self._entries[-1] if self._entries else None

    def get(self, window_id: int) -> Optional[WindowCache]:
        if window_id == 0:
            return self.newest()
        for c in self._entries:
            if c.window_id == window_id:
                return c
        return None

    def pin_newest(self) -> Optional[int]:
        """Pin the newest window; unpin all others (one frozen scene at a time).
        Callers must invoke this on the freeze RISING EDGE only — re-pinning
        every chunk while frozen would move the pin to ever-newer windows and
        protect nothing."""
        newest = self.newest()
        if newest is None:
            return None
        for c in self._entries:
            c.pinned = False
        newest.pinned = True
        return newest.window_id

    def unpin_all(self) -> None:
        for c in self._entries:
            c.pinned = False


def region_disc(x_norm: float, y_norm: float, radius_norm: float,
                grid_w: int = GRID_W, grid_h: int = GRID_H) -> np.ndarray:
    """Region weights for a tap: the touched cell, or a disc around it.

    Coordinates are normalized letterboxed-frame space (the proto contract).
    radius_norm is a fraction of the frame width; 0 = exactly one cell. The
    tapped cell is always included.
    """
    col = min(grid_w - 1, max(0, int(x_norm * grid_w)))
    row = min(grid_h - 1, max(0, int(y_norm * grid_h)))
    w = np.zeros((grid_h, grid_w), dtype=np.float32)
    w[row, col] = 1.0
    if radius_norm > 0:
        radius_cells = radius_norm * grid_w
        cy, cx = row + 0.5, col + 0.5
        for r in range(grid_h):
            for c in range(grid_w):
                if math.hypot(r + 0.5 - cy, c + 0.5 - cx) <= radius_cells:
                    w[r, c] = 1.0
    return w


def _pool_region_vector(feats_img: torch.Tensor, weights: torch.Tensor) -> Optional[torch.Tensor]:
    """Max-pool the K-vectors over the region's support. None = empty region."""
    support = weights > REGION_SUPPORT_THRESHOLD
    if not bool(support.any()):
        return None
    cells = feats_img[0][:, support]          # [K, n_cells]
    return cells.amax(dim=1, keepdim=True).T  # [1, K]


def synthesize_regions(engine, cache: WindowCache, regions: List[torch.Tensor],
                       binary: Optional[bool] = None):
    """The primitive: N region-weight maps -> N PCM tracks + linear energies.

    Pool each region's K-vector FIRST (max over support), synthesize ONE mask
    per region via InnerProd, then post-process in the DEPLOYED halves order
    (inference.py ~1156-1203): unwarp to the linear 512-bin domain FIRST, then
    renorm across the N masks together (gated on MASK_RENORM, divisor clamped
    at max(MASK_RENORM_FLOOR, MASK_RENORM_EPS)), then optional binary — all in
    the linear domain. Renorm/binary do NOT commute with the bilinear unwarp,
    so doing them pre-unwarp would silently diverge from the streamed halves
    math (caught in review; verified up to 0.54 absolute mask error).
    The renormed set partitions the mixture: faders sum back to the original;
    a single tap passes [region, complement].

    Returns (list of float32 [AUD_LEN] arrays, list of float energies).
    Empty-support regions return silence with energy 0 (an honest "nothing
    here", not a globally-biased mask).
    """
    if binary is None:
        binary = bool(BINARY_MASK)

    with torch.inference_mode():
        device = cache.feat_sound.device
        masks = []
        empty_flags = []
        for w in regions:
            wt = torch.as_tensor(w, dtype=torch.float32, device=device)
            vec = _pool_region_vector(cache.feats_img, wt)
            if vec is None:
                masks.append(torch.zeros(1, 1, 256, 256, device=device))
                empty_flags.append(True)
                continue
            z = engine.net_synth(vec, cache.feat_sound)   # [1, 1, 256, 256] logits
            masks.append(torch.sigmoid(z))
            empty_flags.append(False)

        m = torch.cat(masks, dim=0)  # [N, 1, 256, 256], warped domain

        # 1) Inverse log-frequency resample FIRST (deployed order), with the
        #    engine's existing 512-bin grid.
        grid = engine.grid_unwarp.to(device=device, dtype=m.dtype).expand(m.size(0), -1, -1, -1)
        m_lin = F.grid_sample(m, grid, align_corners=True).squeeze(1)  # [N, 512, 256]

        # 2) Renorm across the region set in the LINEAR domain, honoring the
        #    same config gates as the halves path. Empty regions stay zero.
        if MASK_RENORM:
            msum = torch.clamp(
                m_lin.sum(dim=0, keepdim=True),
                min=max(MASK_RENORM_FLOOR, MASK_RENORM_EPS),
            )
            m_lin = m_lin / msum

        # 3) Optional binary threshold, linear domain, after renorm.
        if binary:
            m_lin = (m_lin > MASK_THRES).float()

        pred_mag = cache.mag_lin.unsqueeze(0) * m_lin                  # [N, 512, 256]
        spec = torch.polar(pred_mag, cache.phase.unsqueeze(0).expand_as(pred_mag))
        window = engine.hann_window.to(device)
        wav = torch.istft(spec, n_fft=STFT_FRAME, hop_length=STFT_HOP,
                          window=window, length=AUD_LEN)
        if cache.audio_gain != 1.0:
            wav = wav / cache.audio_gain
        wav = torch.clamp(wav, -1.0, 1.0)

        # Linear-domain energy of what would actually play (gain undone).
        energies = ((pred_mag / cache.audio_gain) ** 2).sum(dim=(1, 2))

    out_wavs, out_energy = [], []
    for i in range(m.size(0)):
        if empty_flags[i]:
            out_wavs.append(np.zeros(AUD_LEN, dtype=np.float32))
            out_energy.append(0.0)
        else:
            out_wavs.append(wav[i].detach().cpu().numpy().astype(np.float32))
            out_energy.append(float(energies[i].item()))
    return out_wavs, out_energy


def cell_analysis(engine, cache: WindowCache, chunk: int = 28, with_features: bool = False):
    """Per-cell quantities for the whole grid, ONE vectorised pass, chunked.

    Returns (energy_map, activation_map, features):
      energy_map     — float32 [GRID_H, GRID_W]: sum((M_lin * |S_lin|)/gain)^2
                       per cell, LINEAR domain: the paper-faithful quantity
                       that matches what plays (PIXEL_PLAN 2.3). Per-cell
                       masks are raw sigmoids (no renorm — there is no
                       partition across 196 cells).
      activation_map — float32 [GRID_H, GRID_W]: mean of the sigmoid mask over
                       the WARPED spectral dims — the only quantity ever
                       validated live on a phone (the existing heatmap). Both
                       maps ship from one pass so they can be compared side by
                       side before one is chosen (amendment B2).
      features       — None unless with_features: float32 [GRID_H*GRID_W, 256]
                       per-cell sound features for clustering — each cell's
                       warped sigmoid mask average-pooled to 16 freq bands x
                       16 time bins and L2-normalized (PIXEL_PLAN 4.1;
                       computed in the same chunk loop so clustering does not
                       pay a second 196-cell synthesizer pass).
    """
    with torch.inference_mode():
        device = cache.feat_sound.device
        K = cache.feats_img.shape[1]
        vecs = cache.feats_img[0].reshape(K, GRID_H * GRID_W).T.contiguous()  # [196, K]
        n = vecs.shape[0]

        energy = torch.empty(n, device=device)
        activation = torch.empty(n, device=device)
        feats = torch.empty(n, 16 * 16, device=device) if with_features else None
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            v = vecs[lo:hi]                                   # [b, K]
            fs = cache.feat_sound.expand(hi - lo, -1, -1, -1)  # [b, K, 256, 256]
            z = engine.net_synth(v, fs)                        # [b, 1, 256, 256]
            msk = torch.sigmoid(z)
            activation[lo:hi] = msk.mean(dim=(1, 2, 3))
            if with_features:
                pooled = F.adaptive_avg_pool2d(msk, (16, 16)).reshape(hi - lo, -1)
                feats[lo:hi] = F.normalize(pooled, dim=1)
            grid = engine.grid_unwarp.to(device=device, dtype=msk.dtype).expand(hi - lo, -1, -1, -1)
            m_lin = F.grid_sample(msk, grid, align_corners=True).squeeze(1)  # [b, 512, 256]
            pm = (m_lin * cache.mag_lin.unsqueeze(0)) / cache.audio_gain
            energy[lo:hi] = (pm ** 2).sum(dim=(1, 2))

    return (
        energy.reshape(GRID_H, GRID_W).detach().cpu().numpy().astype(np.float32),
        activation.reshape(GRID_H, GRID_W).detach().cpu().numpy().astype(np.float32),
        feats.detach().cpu().numpy().astype(np.float32) if with_features else None,
    )


def cell_maps(engine, cache: WindowCache, chunk: int = 28):
    """Back-compat wrapper: (energy_map, activation_map) only."""
    energy, activation, _ = cell_analysis(engine, cache, chunk=chunk, with_features=False)
    return energy, activation


class EnergyMapSmoother:
    """EMA over per-window energy maps + a stable uint8 quantizer.

    Raw per-window maps strobe; the EMA steadies them. Quantization divides by
    a slow-release PEAK-HOLD of the map maximum — deliberately NOT an EMA of
    the peak: dividing a decaying map by a peak that decays with it is
    scale-invariant, so a stale bright pattern would linger long after the
    sound stopped (caught in review). With a held peak, silence decays to
    black at the map EMA's own rate. Constants provisional until tuned
    on-device (PIXEL_PLAN 2.3).
    """

    PEAK_FLOOR = 1e-6
    # Peak-hold release per window: ~0.98^8 ≈ 0.85/s at the 125 ms hop, so the
    # scale adapts down over a few seconds without tracking transient decay.
    PEAK_RELEASE = 0.98

    def __init__(self, alpha: float = ENERGY_EMA):
        self.alpha = float(alpha)
        self._ema: Optional[np.ndarray] = None
        self._peak_hold: float = 0.0

    def update(self, energy_map: np.ndarray) -> bytes:
        if self._ema is None:
            self._ema = energy_map.astype(np.float32).copy()
        else:
            self._ema = self.alpha * energy_map + (1.0 - self.alpha) * self._ema
        peak_now = float(self._ema.max())
        self._peak_hold = max(peak_now, self.PEAK_RELEASE * self._peak_hold)
        norm = np.clip(self._ema / max(self._peak_hold, self.PEAK_FLOOR), 0.0, 1.0)
        return (norm * 255.0).astype(np.uint8).tobytes()
