"""Pixel-mode cache + region primitive, tested on CPU without a checkpoint.

The synthesizer here is the real InnerProd module with random-but-fixed
weights; references are brute-force computations (forward_pixelwise, direct
ISTFT of the mixture), so a wrong shape, transpose, or pooling rule fails
loudly rather than plausibly.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from config import AUD_LEN, STFT_FRAME, STFT_HOP
from models.synthesizer_net import InnerProd
from utils.utils import warpgrid
from pixel_cache import (
    GRID_H,
    GRID_W,
    EnergyMapSmoother,
    WindowCache,
    WindowCacheRing,
    _pool_region_vector,
    cell_maps,
    region_disc,
    synthesize_regions,
)

K = 32


class StubEngine:
    """The three attributes pixel_cache needs, with the REAL synthesizer."""

    def __init__(self):
        torch.manual_seed(7)
        self.net_synth = InnerProd(fc_dim=K)
        with torch.no_grad():
            self.net_synth.scale.copy_(torch.rand(K) * 1.5 + 0.25)
            self.net_synth.bias.copy_(torch.tensor([-0.4]))
        self.grid_unwarp = torch.from_numpy(warpgrid(1, 512, 256, warp=False))
        self.hann_window = torch.hann_window(STFT_FRAME)


@pytest.fixture(scope="module")
def engine():
    return StubEngine()


@pytest.fixture(scope="module")
def cache():
    torch.manual_seed(11)
    y = torch.randn(AUD_LEN) * 0.1
    stft = torch.stft(
        y, n_fft=STFT_FRAME, hop_length=STFT_HOP,
        window=torch.hann_window(STFT_FRAME), return_complex=True,
    )[:512, :256]
    return WindowCache(
        window_id=1,
        feat_sound=torch.randn(1, K, 256, 256) * 0.5,
        feats_img=torch.sigmoid(torch.randn(1, K, GRID_H, GRID_W) * 2.0),
        mag_lin=stft.abs().float(),
        phase=stft.angle().float(),
        audio_gain=1.7,
        start_sample=0,
        center_timestamp_ms=1000,
    )


# ---------------------------------------------------------------------------
# Pooling + mask math against brute-force references
# ---------------------------------------------------------------------------

def test_single_cell_vector_and_mask_match_forward_pixelwise(engine, cache):
    r, c = 5, 9
    w = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    w[r, c] = 1.0
    vec = _pool_region_vector(cache.feats_img, torch.from_numpy(w))
    assert torch.allclose(vec[0], cache.feats_img[0, :, r, c])

    z = engine.net_synth(vec, cache.feat_sound)
    z_ref = engine.net_synth.forward_pixelwise(cache.feats_img, cache.feat_sound)
    assert torch.allclose(z[0, 0], z_ref[0, r, c], atol=1e-5)


def test_full_grid_region_equals_spatial_max(engine, cache):
    w = np.ones((GRID_H, GRID_W), dtype=np.float32)
    vec = _pool_region_vector(cache.feats_img, torch.from_numpy(w))
    ref = cache.feats_img[0].amax(dim=(1, 2))
    assert torch.allclose(vec[0], ref)


def test_partition_property_region_plus_complement_reconstructs_mixture(engine, cache):
    w = torch.from_numpy(region_disc(0.3, 0.4, 0.15))
    wavs, energies = synthesize_regions(engine, cache, [w, 1.0 - w], binary=False)

    mix_ref = torch.istft(
        torch.polar(cache.mag_lin, cache.phase),
        n_fft=STFT_FRAME, hop_length=STFT_HOP,
        window=engine.hann_window, length=AUD_LEN,
    ) / cache.audio_gain
    total = wavs[0] + wavs[1]
    assert np.allclose(total, mix_ref.numpy(), atol=2e-4), (
        "renormed region+complement must sum back to the mixture"
    )
    assert energies[0] > 0 and energies[1] > 0


def test_renorm_happens_in_linear_domain_after_unwarp(engine, cache):
    """The deployed halves order is unwarp -> renorm; renorm and the bilinear
    unwarp do not commute, so this pins the order (review finding)."""
    import torch.nn.functional as F
    from config import MASK_RENORM_EPS

    w = torch.from_numpy(region_disc(0.25, 0.25, 0.2))
    regions = [w, 1.0 - w]
    # Legacy conditioning so the hand-built reference below stays valid; the
    # unwarp->renorm ordering under test is shared by both conditioning modes.
    wavs, _ = synthesize_regions(
        engine, cache, regions, binary=False, conditioning="feature_maxpool"
    )

    # Independent reference computed the deployed way.
    with torch.inference_mode():
        masks = []
        for rw in regions:
            vec = _pool_region_vector(cache.feats_img, rw)
            masks.append(torch.sigmoid(engine.net_synth(vec, cache.feat_sound)))
        m = torch.cat(masks, dim=0)
        grid = engine.grid_unwarp.expand(2, -1, -1, -1)
        m_lin = F.grid_sample(m, grid, align_corners=True).squeeze(1)
        m_lin = m_lin / torch.clamp(m_lin.sum(dim=0, keepdim=True), min=MASK_RENORM_EPS)
        pred = cache.mag_lin.unsqueeze(0) * m_lin
        ref = torch.istft(
            torch.polar(pred, cache.phase.unsqueeze(0).expand_as(pred)),
            n_fft=STFT_FRAME, hop_length=STFT_HOP,
            window=engine.hann_window, length=AUD_LEN,
        ) / cache.audio_gain
    assert np.allclose(wavs[0], np.clip(ref[0].numpy(), -1, 1), atol=1e-5)

    # And the WRONG order (renorm pre-unwarp) must NOT match — proves the
    # test can actually tell the two apart.
    with torch.inference_mode():
        m_bad = torch.cat(masks, dim=0)
        m_bad = m_bad / torch.clamp(m_bad.sum(dim=0, keepdim=True), min=MASK_RENORM_EPS)
        m_bad_lin = F.grid_sample(m_bad, grid, align_corners=True).squeeze(1)
        pred_bad = cache.mag_lin.unsqueeze(0) * m_bad_lin
        bad = torch.istft(
            torch.polar(pred_bad, cache.phase.unsqueeze(0).expand_as(pred_bad)),
            n_fft=STFT_FRAME, hop_length=STFT_HOP,
            window=engine.hann_window, length=AUD_LEN,
        ) / cache.audio_gain
    assert not np.allclose(wavs[0], np.clip(bad[0].numpy(), -1, 1), atol=1e-5)


def test_mask_renorm_flag_is_honored(engine, cache, monkeypatch):
    import pixel_cache as pc
    w = torch.from_numpy(region_disc(0.5, 0.5, 0.2))
    on, _ = synthesize_regions(engine, cache, [w, 1.0 - w], binary=False)
    monkeypatch.setattr(pc, "MASK_RENORM", 0)
    off, _ = synthesize_regions(engine, cache, [w, 1.0 - w], binary=False)
    assert not np.allclose(on[0], off[0], atol=1e-5), (
        "SONICSIGHT_MASK_RENORM=0 must change pixel output like it changes halves"
    )


def test_tap_binary_default_env_controls_mask_variant(engine, cache, monkeypatch):
    """binary=None must resolve to PIXEL_TAP_BINARY (pixel-path default,
    measured decision 2026-08-15: ratio taps leak — cello tap corr(region,
    complement)=0.989 — binary taps separate cleanly), not the halves-shared
    BINARY_MASK."""
    import pixel_cache as pc
    w = torch.from_numpy(region_disc(0.5, 0.5, 0.2))
    explicit_bin, _ = synthesize_regions(engine, cache, [w, 1.0 - w], binary=True)
    explicit_soft, _ = synthesize_regions(engine, cache, [w, 1.0 - w], binary=False)

    monkeypatch.setattr(pc, "PIXEL_TAP_BINARY", 1)
    default_on, _ = synthesize_regions(engine, cache, [w, 1.0 - w])
    assert np.allclose(default_on[0], explicit_bin[0], atol=1e-6)

    monkeypatch.setattr(pc, "PIXEL_TAP_BINARY", 0)
    default_off, _ = synthesize_regions(engine, cache, [w, 1.0 - w])
    assert np.allclose(default_off[0], explicit_soft[0], atol=1e-6)


def test_sticky_ema_smooths_masks_across_windows(engine, cache, monkeypatch):
    """The sticky mask-EMA lever (OFF by default — measured no benefit on
    canon_d; see config): when enabled, an ema dict carried across calls
    must smooth the linear-domain masks pre-renorm, mirroring the halves
    MASK_SMOOTH order (unwarp -> EMA -> renorm -> binary)."""
    import pixel_cache as pc
    monkeypatch.setattr(pc, "PIXEL_STICKY_SMOOTH", 0.3)
    w = torch.from_numpy(region_disc(0.3, 0.4, 0.15))

    # Second window with different sound features -> different raw masks.
    torch.manual_seed(23)
    cache2 = WindowCache(
        window_id=2,
        feat_sound=torch.randn(1, K, 256, 256) * 0.5,
        feats_img=cache.feats_img,
        mag_lin=cache.mag_lin,
        phase=cache.phase,
        audio_gain=cache.audio_gain,
        start_sample=0,
        center_timestamp_ms=1125,
    )

    stateless2, _ = synthesize_regions(engine, cache2, [w, 1.0 - w], binary=False)

    ema = {}
    first, _ = synthesize_regions(engine, cache, [w, 1.0 - w], binary=False, ema=ema)
    stateless1, _ = synthesize_regions(engine, cache, [w, 1.0 - w], binary=False)
    # First call seeds the state: output identical to stateless.
    assert np.allclose(first[0], stateless1[0], atol=1e-6)
    assert "m" in ema

    smoothed2, _ = synthesize_regions(engine, cache2, [w, 1.0 - w], binary=False, ema=ema)
    # Second window's smoothed output must differ from its stateless output
    # (history pulls it back toward window 1)...
    assert not np.allclose(smoothed2[0], stateless2[0], atol=1e-5)
    # ...and partition must still hold exactly (renorm runs after the EMA).
    mix_ref = torch.istft(
        torch.polar(cache2.mag_lin, cache2.phase),
        n_fft=STFT_FRAME, hop_length=STFT_HOP,
        window=engine.hann_window, length=AUD_LEN,
    ) / cache2.audio_gain
    assert np.allclose(smoothed2[0] + smoothed2[1], mix_ref.numpy(), atol=2e-4)


def test_no_grad_graph_on_outputs(engine, cache):
    # net_synth params require grad; without the inference_mode guard these
    # calls would silently build autograd graphs (review finding).
    assert engine.net_synth.scale.requires_grad
    wavs, _ = synthesize_regions(engine, cache, [torch.ones(GRID_H, GRID_W)])
    e, a = cell_maps(engine, cache, chunk=49)
    assert isinstance(wavs[0], np.ndarray) and isinstance(e, np.ndarray)


def test_empty_region_is_honest_silence(engine, cache):
    w = torch.zeros(GRID_H, GRID_W)
    wavs, energies = synthesize_regions(engine, cache, [w, 1.0 - w])
    assert not wavs[0].any()
    assert energies[0] == 0.0
    assert wavs[1].any()  # the complement (whole grid) still plays


def test_binary_full_region_returns_the_mixture(engine, cache):
    # Region = whole grid, complement empty: renorm makes the mask 1
    # everywhere, binary keeps it 1 — output must be the mixture exactly.
    w = torch.ones(GRID_H, GRID_W)
    wavs, _ = synthesize_regions(engine, cache, [w, torch.zeros_like(w)], binary=True)
    mix_ref = torch.istft(
        torch.polar(cache.mag_lin, cache.phase),
        n_fft=STFT_FRAME, hop_length=STFT_HOP,
        window=engine.hann_window, length=AUD_LEN,
    ) / cache.audio_gain
    assert np.allclose(wavs[0], np.clip(mix_ref.numpy(), -1, 1), atol=2e-4)


# ---------------------------------------------------------------------------
# Region conditioning selectivity (the long-press "both instruments" bug)
# ---------------------------------------------------------------------------

class _TwoChanEngine:
    """Real InnerProd, 2 channels, identity scale — masks are readable by hand."""

    def __init__(self):
        self.net_synth = InnerProd(fc_dim=2)
        with torch.no_grad():
            self.net_synth.scale.copy_(torch.ones(2))
            self.net_synth.bias.copy_(torch.tensor([0.0]))
        self.grid_unwarp = torch.from_numpy(warpgrid(1, 512, 256, warp=False))
        self.hann_window = torch.hann_window(STFT_FRAME)


def _two_instrument_setup():
    """Synthetic scene mirroring the live failure: two spectrally disjoint
    instruments at known cells, plus a heterogeneous leaky background whose
    per-channel maxima saturate a whole-frame feature max-pool (the measured
    live condition: every cell claims 30-60% of peak energy)."""
    torch.manual_seed(3)
    engine = _TwoChanEngine()

    # Sound features: channel 0 votes +4 logits for the low warped band,
    # channel 1 for the high band.
    feat_sound = torch.empty(1, 2, 256, 256)
    feat_sound[0, 0, :128, :] = 4.0
    feat_sound[0, 0, 128:, :] = -4.0
    feat_sound[0, 1] = -feat_sound[0, 0]

    # Vision features: instrument A at (2,2) purely ch0, B at (11,11) purely
    # ch1; background cells carry BOTH channels at 0.3-0.9, so max-pooling
    # 191 of them saturates both channels.
    feats_img = torch.rand(1, 2, GRID_H, GRID_W) * 0.6 + 0.3
    feats_img[0, :, 2, 2] = torch.tensor([1.0, 0.0])
    feats_img[0, :, 11, 11] = torch.tensor([0.0, 1.0])

    y = torch.randn(AUD_LEN) * 0.1
    stft = torch.stft(
        y, n_fft=STFT_FRAME, hop_length=STFT_HOP,
        window=torch.hann_window(STFT_FRAME), return_complex=True,
    )[:512, :256]
    cache = WindowCache(
        window_id=1,
        feat_sound=feat_sound,
        feats_img=feats_img,
        mag_lin=torch.ones(512, 256),
        phase=stft.angle().float(),
        audio_gain=1.0,
        start_sample=0,
        center_timestamp_ms=0,
    )
    # Energy map as the live server always provides it before any query:
    # instruments are the peaks, background at the measured ~30% floor.
    em = np.full((GRID_H, GRID_W), 3.0, dtype=np.float32)
    em[2, 2] = 10.0
    em[11, 11] = 10.0
    cache.energy_map = em
    return engine, cache


def _band_energies(wav, engine):
    """Energy of a track split into instrument A's vs B's linear-domain rows."""
    import torch.nn.functional as F

    band = torch.zeros(1, 1, 256, 256)
    band[:, :, :128, :] = 1.0
    lin = F.grid_sample(band, engine.grid_unwarp, align_corners=True).squeeze()
    a_rows = lin.mean(dim=1) > 0.5
    s = torch.stft(
        torch.from_numpy(wav), n_fft=STFT_FRAME, hop_length=STFT_HOP,
        window=engine.hann_window, return_complex=True,
    )[:512, :256].abs()
    e = s ** 2
    return float(e[a_rows].sum()), float(e[~a_rows].sum())


def test_long_press_region_suppresses_the_other_instrument():
    """A tap on instrument A must not play a mildly re-weighted mixture:
    conditioning each region on its most sound-active CELL MASKS (the
    deployed halves recipe) has to suppress instrument B by a clear margin."""
    engine, cache = _two_instrument_setup()
    w = torch.from_numpy(region_disc(2.5 / GRID_W, 2.5 / GRID_H, 1.0 / GRID_W))
    wavs, _ = synthesize_regions(engine, cache, [w, 1.0 - w], binary=False)
    e_a, e_b = _band_energies(wavs[0], engine)
    assert e_a / max(e_b, 1e-12) >= 4.0, (
        f"tap on A keeps too much of B: E_A/E_B = {e_a / max(e_b, 1e-12):.2f}"
    )


# ---------------------------------------------------------------------------
# Cell maps
# ---------------------------------------------------------------------------

def test_cell_maps_shapes_and_chunk_equivalence(engine, cache):
    e28, a28 = cell_maps(engine, cache, chunk=28)
    e_all, a_all = cell_maps(engine, cache, chunk=196)
    assert e28.shape == a28.shape == (GRID_H, GRID_W)
    assert np.allclose(e28, e_all, rtol=1e-5, atol=1e-3)
    assert np.allclose(a28, a_all, rtol=1e-5, atol=1e-6)
    assert (e28 >= 0).all()
    assert (a28 > 0).all() and (a28 < 1).all()  # mean of sigmoids


def test_cell_analysis_features_shape_norm_and_chunk_equivalence(engine, cache):
    from pixel_cache import cell_analysis

    e1, a1, f1 = cell_analysis(engine, cache, chunk=28, with_features=True)
    e2, a2, f2 = cell_analysis(engine, cache, chunk=196, with_features=True)
    assert f1.shape == (GRID_H * GRID_W, 256)
    assert np.allclose(np.linalg.norm(f1, axis=1), 1.0, atol=1e-5)
    assert np.allclose(f1, f2, atol=1e-5)
    assert np.allclose(e1, e2, rtol=1e-5, atol=1e-3)
    # with_features=False returns None and identical maps
    e3, a3, f3 = cell_analysis(engine, cache, chunk=28, with_features=False)
    assert f3 is None and np.allclose(e1, e3, rtol=1e-5, atol=1e-3)


def test_cell_maps_silent_mixture_has_zero_energy(engine, cache):
    silent = WindowCache(
        window_id=2,
        feat_sound=cache.feat_sound,
        feats_img=cache.feats_img,
        mag_lin=torch.zeros_like(cache.mag_lin),
        phase=cache.phase,
        audio_gain=1.0,
        start_sample=0,
        center_timestamp_ms=0,
    )
    energy, activation = cell_maps(engine, silent, chunk=49)
    assert not energy.any()
    assert activation.any()  # activation is magnitude-free by design


# ---------------------------------------------------------------------------
# Ring + pinning (freeze semantics, amendment A2)
# ---------------------------------------------------------------------------

def _mini_cache(i):
    return WindowCache(
        window_id=0,
        feat_sound=torch.zeros(1, 1, 1, 1),
        feats_img=torch.zeros(1, 1, 1, 1),
        mag_lin=torch.zeros(1, 1),
        phase=torch.zeros(1, 1),
        audio_gain=1.0,
        start_sample=i,
        center_timestamp_ms=i,
    )


def test_ring_eviction_and_lookup():
    ring = WindowCacheRing(capacity=3)
    ids = [ring.push(_mini_cache(i)) for i in range(5)]
    assert ids == [1, 2, 3, 4, 5]
    assert ring.get(0).window_id == 5
    assert ring.get(1) is None and ring.get(2) is None  # evicted
    assert ring.get(3).window_id == 3


def test_pinned_window_survives_eviction():
    ring = WindowCacheRing(capacity=3)
    for i in range(3):
        ring.push(_mini_cache(i))
    pinned_id = ring.pin_newest()
    assert pinned_id == 3
    for i in range(4):
        ring.push(_mini_cache(10 + i))
    assert ring.get(pinned_id) is not None, "pinned window must not be evicted"
    assert ring.get(4) is None  # unpinned windows still age out
    # re-pinning moves the exemption
    new_pin = ring.pin_newest()
    assert new_pin == 7
    assert not ring.get(pinned_id).pinned
    # release: unpin_all clears every exemption (freeze falling edge)
    ring.unpin_all()
    assert all(not c.pinned for c in ring._entries)


# ---------------------------------------------------------------------------
# Disc geometry + smoother
# ---------------------------------------------------------------------------

def test_region_disc_geometry():
    w = region_disc(0.5, 0.5, 0.0)
    assert w.sum() == 1.0 and w[7, 7] == 1.0  # radius 0 = exactly one cell

    w = region_disc(1.5, -0.2, 0.0)  # out-of-range clamps
    assert w[0, GRID_W - 1] == 1.0 and w.sum() == 1.0

    w = region_disc(0.5, 0.5, 0.1)  # 0.1 * 14 = 1.4-cell radius disc
    assert w[7, 7] == 1.0 and w.sum() > 1.0
    assert w.sum() <= 9  # stays a small neighbourhood, not the whole grid


def test_energy_smoother_bytes_and_stability():
    s = EnergyMapSmoother(alpha=0.5)
    flat = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    assert s.update(flat) == b"\x00" * (GRID_H * GRID_W)
    hot = flat.copy(); hot[3, 3] = 100.0
    b1 = s.update(hot)
    assert len(b1) == GRID_H * GRID_W
    assert max(b1) > 0
    # repeated identical input converges rather than flickering
    b2 = s.update(hot); b3 = s.update(hot)
    assert abs(max(b3) - max(b2)) <= abs(max(b2) - max(b1)) + 1


def test_energy_smoother_silence_goes_dark():
    """Peak-hold (not peak-EMA) normalization: after the sound stops, the
    map must actually fade to black instead of a scale-invariant lingering
    glow (review finding)."""
    s = EnergyMapSmoother(alpha=0.5)
    hot = np.zeros((GRID_H, GRID_W), dtype=np.float32); hot[3, 3] = 1000.0
    for _ in range(4):
        bright = s.update(hot)
    assert max(bright) == 255
    dark = b""
    for _ in range(10):
        dark = s.update(np.zeros((GRID_H, GRID_W), dtype=np.float32))
    assert max(dark) < 10, "10 silent windows (~1.25 s) must decay near black"


# ---------------------------------------------------------------------------
# Registry + frame selection
# ---------------------------------------------------------------------------

def test_pixel_spec_registered():
    from model_registry import REGISTRY
    halves = REGISTRY["sonicsight"]
    pixel = REGISTRY["sonicsight-pixel"]
    assert halves.mode == "halves"  # default untouched — bit-exact promise
    assert pixel.mode == "pixel"
    assert pixel.frame_selection == "centered_triple_full"
    assert pixel.frame_kind == "full_letterboxed"
    assert pixel.window_samples == AUD_LEN and pixel.capture_sample_rate == 11025
    assert pixel.heatmap_count == 0  # legacy heatmaps empty in pixel mode
    assert pixel.num_frames == 3     # consecutive triple, like deployed halves


def test_centered_triple_full_selection():
    from PIL import Image
    from inference import StreamingBuffer

    buf = StreamingBuffer(frame_selection="centered_triple_full", frame_ring_cap=60)
    imgs = []
    for i in range(5):
        img = Image.new("RGB", (8, 8), (i, 0, 0))
        imgs.append(img)
        buf.add_decoded_frame(i * 125, None, None, img)
    frames, trim = buf._select_frames(2)
    assert isinstance(frames, list) and len(frames) == 3
    assert frames[0] is imgs[1] and frames[1] is imgs[2] and frames[2] is imgs[3]
    assert trim == 1  # same trim rule as the deployed triple
