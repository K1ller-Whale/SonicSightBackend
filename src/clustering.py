"""Source discovery for pixel mode (PIXEL_PLAN section 4).

Silence first, then a FIXED number of k-means clusters over the surviving
cells, then label persistence across windows so colours do not strobe.

Design decisions carried from the plan review:
- C1: N is FIXED at 3 clusters + a silence class, empty clusters allowed.
  Per-window N selection would fight the persistence matcher (3-vs-4
  matching strobes colours), and solos-and-duets territory does not need
  adaptive N.
- C2: the silence thresholds are PROVISIONAL and must be set from measured
  per-cell energy distributions on real clips. discover_sources logs energy
  percentiles (throttled) exactly so those numbers can be read off a live
  run; update the constants with the measured values and the reasoning.

Sources get colours and positions, never instrument names — channel
activations carry no labels and naming them is a tempting dead end
(46.2%/68.9% one-channel-per-category accuracy in the paper).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from pixel_cache import GRID_H, GRID_W

logger = logging.getLogger(__name__)

N_CLUSTERS = 3          # fixed + silence class (amendment C1)
SILENCE_LABEL = 255
# Track ids live in [1, MAX_TRACK_ID] and are allocated from a REUSE pool
# (lowest free id): the label map is uint8, so ids must fit beside the
# silence sentinel, and unbounded monotonic ids would collide mod-255 with
# long-lived tracks (review finding). An id is recycled only after its track
# ages out; with <= 3 live tracks the pool never exhausts.
MAX_TRACK_ID = 254

# ── PROVISIONAL silence thresholds (amendment C2) ──
# A cell is active when energy > max(SILENCE_ABS_FLOOR, SILENCE_REL_FRAC * max).
# The relative gate keeps one loud source from silencing everything else at
# 5%; the absolute floor guards empty rooms. Both UNMEASURED until real-clip
# distributions are read from the percentile log below — do not trust them
# past that first measurement session.
SILENCE_ABS_FLOOR = 1e-3
SILENCE_REL_FRAC = 0.05

# Persistence: a track survives this many windows unmatched before it dies.
TRACK_MAX_MISSED = 4
# Match acceptance: MEAN-CENTERED cosine similarity of pooled sound features,
# minus a penalty on normalized centroid distance. Centering matters:
# production features are pooled sigmoids — all-positive vectors whose RAW
# cosine never drops below ~0.65 even between unrelated sources, which made
# any positive threshold vacuous (review finding). Centered, unrelated
# features score near 0 and the gate actually gates.
MATCH_MIN_SCORE = 0.35
MATCH_SPATIAL_WEIGHT = 0.5

# Cluster colours: distinct under the common red-green colour-vision
# deficiencies (no red/green opposition; separated in lightness and in
# blue-yellow). Sources are distinguished BY colour, so this is load-bearing
# (brief: Phase 6 design constraint, enforced at the data source).
PALETTE = (0x4477AA, 0xCCBB44, 0xAA3377)  # blue, sand, purple

def _centered(f: np.ndarray) -> np.ndarray:
    """Remove the DC component all-positive features share, renormalize."""
    c = f - f.mean()
    return c / max(np.linalg.norm(c), 1e-12)


@dataclass
class Track:
    track_id: int
    feature: np.ndarray          # [256] L2-normalized pooled sound feature
    centroid: Tuple[float, float]  # normalized (x, y), letterboxed frame space
    color: int
    energy: float = 0.0
    missed: int = 0


@dataclass
class ClusterState:
    tracks: List[Track] = field(default_factory=list)
    log_counter: int = 0  # per-stream C2 instrumentation cadence

    def _alloc_id(self) -> int:
        """Lowest free id in [1, MAX_TRACK_ID] — recycled after age-out, so
        the uint8 label map and SourceCluster.cluster_id carry the SAME
        number with no modulo anywhere."""
        used = {t.track_id for t in self.tracks}
        for i in range(1, MAX_TRACK_ID + 1):
            if i not in used:
                return i
        return MAX_TRACK_ID  # unreachable with <= 3 live tracks

    def _free_color(self) -> int:
        # Every live track owns its colour, INCLUDING briefly-missed ones —
        # they keep it when re-matched (review finding).
        used = {t.color for t in self.tracks}
        for c in PALETTE:
            if c not in used:
                return c
        return PALETTE[len(self.tracks) % len(PALETTE)]


def _kmeans(features: np.ndarray, k: int, iters: int = 20, seed: int = 0):
    """Tiny deterministic k-means (k-means++ init). <=196 points, k<=3 —
    sub-millisecond; no sklearn dependency."""
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    centers = np.empty((k, features.shape[1]), dtype=features.dtype)
    centers[0] = features[rng.integers(n)]
    for i in range(1, k):
        d2 = np.min(
            ((features[:, None, :] - centers[None, :i, :]) ** 2).sum(-1), axis=1
        )
        total = float(d2.sum())
        if total <= 0.0:
            # All remaining points are bit-identical to chosen centers
            # (saturated sigmoids pool to identical vectors) — a zero
            # probability vector would crash rng.choice. Uniform fallback;
            # duplicate centers are harmless (argmin ties break low).
            centers[i] = features[rng.integers(n)]
            continue
        probs = d2.astype(np.float64) / total
        probs = probs / probs.sum()
        centers[i] = features[rng.choice(n, p=probs)]
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = d.argmin(axis=1)
        if (new_labels == labels).all() and _ > 0:
            break
        labels = new_labels
        for j in range(k):
            member = features[labels == j]
            if len(member):
                centers[j] = member.mean(axis=0)
    return labels, centers


def discover_sources(
    features: np.ndarray,          # [GRID_H*GRID_W, F] L2-normalized
    energy_map: np.ndarray,        # [GRID_H, GRID_W] linear-domain energies
    state: Optional[ClusterState],
):
    """-> (label_map [GRID_H, GRID_W] uint8 (SILENCE_LABEL = silence),
           clusters: list of dicts {cluster_id, centroid_x, centroid_y, energy, rgb},
           new_state)

    Silence gate first (or the clusterer happily partitions the noise floor),
    fixed-N k-means on the survivors, then greedy nearest-track matching so a
    source keeps its id and colour across windows.
    """
    state = state or ClusterState()

    energy = energy_map.reshape(-1)
    threshold = max(SILENCE_ABS_FLOOR, SILENCE_REL_FRAC * float(energy.max()))
    active = energy > threshold

    # C2 instrumentation: the numbers the provisional thresholds must be
    # re-derived from, readable off any live run. Per-stream counter so
    # concurrent streams do not starve each other's log cadence.
    state.log_counter += 1
    if state.log_counter % 40 == 1:  # ~every 5 s at the 125 ms hop
        q = np.percentile(energy, [50, 75, 90, 99])
        logger.info(
            "pixel energy percentiles p50=%.3g p75=%.3g p90=%.3g p99=%.3g "
            "max=%.3g threshold=%.3g active=%d/196",
            q[0], q[1], q[2], q[3], float(energy.max()), threshold, int(active.sum()),
        )

    labels_map = np.full(GRID_H * GRID_W, SILENCE_LABEL, dtype=np.uint8)

    n_active = int(active.sum())
    if n_active == 0:
        for t in state.tracks:
            t.missed += 1
        state.tracks = [t for t in state.tracks if t.missed <= TRACK_MAX_MISSED]
        return labels_map.reshape(GRID_H, GRID_W), [], state

    k = min(N_CLUSTERS, n_active)  # empty clusters allowed by construction
    feats_active = features[active]
    # CONSTANT seed: a data-derived seed flips the k-means init between
    # near-identical windows (any float wiggle), churning how a blob splits
    # under fixed k and defeating persistence (review finding). Same seed +
    # near-same data converges near-same; residual churn is absorbed by the
    # track matcher.
    kl, centers = _kmeans(feats_active, k, seed=0)

    # Build raw clusters: energy-weighted spatial centroid in normalized
    # letterboxed-frame coordinates (cell centers / grid size).
    rows, cols = np.divmod(np.flatnonzero(active), GRID_W)
    raw = []
    for j in range(k):
        member = kl == j
        if not member.any():
            continue
        e = energy[active][member]
        wsum = max(float(e.sum()), 1e-12)
        cx = float((((cols[member] + 0.5) / GRID_W) * e).sum() / wsum)
        cy = float((((rows[member] + 0.5) / GRID_H) * e).sum() / wsum)
        raw.append({
            "kidx": j,
            "feature": centers[j] / max(np.linalg.norm(centers[j]), 1e-12),
            "centroid": (cx, cy),
            "energy": float(e.sum()),
        })

    # ── Label persistence: greedy best-match against surviving tracks ──
    # Cosine on MEAN-CENTERED features (see MATCH_MIN_SCORE comment).
    candidates = []
    for ri, r in enumerate(raw):
        rc = _centered(r["feature"])
        for t in state.tracks:
            cos = float(np.dot(rc, _centered(t.feature)))
            dist = float(np.hypot(r["centroid"][0] - t.centroid[0],
                                  r["centroid"][1] - t.centroid[1]))
            candidates.append((cos - MATCH_SPATIAL_WEIGHT * dist, ri, t))
    candidates.sort(key=lambda c: -c[0])
    if candidates and state.log_counter % 40 == 1:
        # Instrumentation for MATCH_MIN_SCORE (same C2 spirit): read real
        # match-score distributions off a live run before trusting the gate.
        logger.info(
            "pixel match scores best=%.3f worst=%.3f n=%d",
            candidates[0][0], candidates[-1][0], len(candidates),
        )

    assigned_r, assigned_t = set(), set()
    for score, ri, t in candidates:
        if score < MATCH_MIN_SCORE or ri in assigned_r or t.track_id in assigned_t:
            continue
        r = raw[ri]
        t.feature = 0.5 * t.feature + 0.5 * r["feature"]
        t.feature /= max(np.linalg.norm(t.feature), 1e-12)
        t.centroid = r["centroid"]
        t.energy = r["energy"]
        t.missed = 0
        r["track"] = t
        assigned_r.add(ri)
        assigned_t.add(t.track_id)

    for ri, r in enumerate(raw):
        if ri in assigned_r:
            continue
        track = Track(
            track_id=state._alloc_id(),
            feature=r["feature"],
            centroid=r["centroid"],
            color=state._free_color(),
            energy=r["energy"],
        )
        state.tracks.append(track)
        r["track"] = track

    for t in state.tracks:
        if t.track_id not in {r["track"].track_id for r in raw}:
            t.missed += 1
    state.tracks = [t for t in state.tracks if t.missed <= TRACK_MAX_MISSED]

    # Label map carries TRACK ids (stable across windows, <= MAX_TRACK_ID by
    # construction — same number as SourceCluster.cluster_id, no modulo).
    kidx_to_track = {r["kidx"]: r["track"].track_id for r in raw}
    active_idx = np.flatnonzero(active)
    for pos, j in zip(active_idx, kl):
        labels_map[pos] = kidx_to_track.get(int(j), SILENCE_LABEL)

    clusters = [{
        "cluster_id": r["track"].track_id,
        "centroid_x": r["centroid"][0],
        "centroid_y": r["centroid"][1],
        "energy": r["energy"],
        "rgb": r["track"].color,
    } for r in raw]

    return labels_map.reshape(GRID_H, GRID_W), clusters, state


def region_weights_for_track(label_map: np.ndarray, track_id: int) -> np.ndarray:
    """Region-weight mask for one discovered source — feed to
    synthesize_regions to give the mixer its per-source track."""
    return (label_map == track_id).astype(np.float32)
