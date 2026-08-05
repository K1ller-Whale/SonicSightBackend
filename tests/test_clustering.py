"""Source discovery: silence gate, fixed-N k-means, label persistence."""

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from clustering import (
    N_CLUSTERS,
    PALETTE,
    SILENCE_LABEL,
    TRACK_MAX_MISSED,
    ClusterState,
    discover_sources,
    region_weights_for_track,
)
from pixel_cache import GRID_H, GRID_W

F = 256
N = GRID_H * GRID_W


def _unit(v):
    return v / np.linalg.norm(v)


def _scene(sources):
    """sources: list of (row, col, base_feature). Builds features + energy:
    3x3 blob of that feature/energy around each source; silence elsewhere."""
    rng = np.random.default_rng(3)
    features = rng.normal(size=(N, F)).astype(np.float32) * 0.01
    energy = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for row, col, base in sources:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if 0 <= r < GRID_H and 0 <= c < GRID_W:
                    idx = r * GRID_W + c
                    features[idx] = base + rng.normal(size=F).astype(np.float32) * 0.02
                    energy[r, c] = 50.0
    features = features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True), 1e-12
    )
    return features.astype(np.float32), energy


BASE_A = _unit(np.concatenate([np.ones(F // 2), np.zeros(F - F // 2)]).astype(np.float32))
BASE_B = _unit(np.concatenate([np.zeros(F - F // 2), np.ones(F // 2)]).astype(np.float32))


def test_all_silence_yields_no_clusters():
    features = np.zeros((N, F), dtype=np.float32)
    energy = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    labels, clusters, state = discover_sources(features, energy, None)
    assert (labels == SILENCE_LABEL).all()
    assert clusters == []


def test_two_distinct_sources_get_two_tracks():
    features, energy = _scene([(3, 3, BASE_A), (10, 11, BASE_B)])
    labels, clusters, state = discover_sources(features, energy, None)

    active_labels = set(labels[energy > 0].tolist())
    assert SILENCE_LABEL not in active_labels
    assert (labels[energy == 0] == SILENCE_LABEL).all()
    assert len(clusters) >= 2  # fixed k=3 may split one blob; never merges silence

    # The two blob centers must land in DIFFERENT tracks.
    assert labels[3, 3] != labels[10, 11]
    ids = {c["cluster_id"] for c in clusters}
    assert labels[3, 3] in ids  # label bytes carry the SAME id as the proto
    # colours come from the CVD-safe palette
    assert all(c["rgb"] in PALETTE for c in clusters)


def test_labels_persist_across_windows():
    f1, e1 = _scene([(3, 3, BASE_A), (10, 11, BASE_B)])
    labels1, clusters1, state = discover_sources(f1, e1, None)
    id_a1 = labels1[3, 3]
    id_b1 = labels1[10, 11]
    color_by_id = {c["cluster_id"]: c["rgb"] for c in clusters1}

    # Next window: same sources, slightly moved — ids and colours must hold.
    f2, e2 = _scene([(4, 3, BASE_A), (10, 12, BASE_B)])
    labels2, clusters2, state = discover_sources(f2, e2, state)
    assert labels2[4, 3] == id_a1, "source A must keep its track id"
    assert labels2[10, 12] == id_b1, "source B must keep its track id"
    for c in clusters2:
        if c["cluster_id"] in color_by_id:
            assert c["rgb"] == color_by_id[c["cluster_id"]], "colour must not strobe"


def _point_scene(sources):
    """Single-cell sources: track counts are exactly len(sources), because
    k = min(3, n_active) and each source is one active cell."""
    features = np.zeros((N, F), dtype=np.float32)
    energy = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for row, col, base in sources:
        features[row * GRID_W + col] = base
        energy[row, col] = 50.0
    return features, energy


def test_vanished_source_ages_out():
    f1, e1 = _point_scene([(3, 3, BASE_A), (10, 11, BASE_B)])
    labels1, _, state = discover_sources(f1, e1, None)
    id_b = int(labels1[10, 11])

    # B disappears; A alone for TRACK_MAX_MISSED+1 windows.
    for _ in range(TRACK_MAX_MISSED + 1):
        fa, ea = _point_scene([(3, 3, BASE_A)])
        _, _, state = discover_sources(fa, ea, state)
    assert all(t.track_id != id_b for t in state.tracks), "B must age out"

    # A genuinely new source appears: distinct from A. Its numeric id MAY
    # legally reuse B's — ids are a reuse pool <= 254 by design — but it must
    # never share A's live id.
    BASE_C = _unit(np.abs(np.sin(np.arange(F))).astype(np.float32))
    fc, ec = _point_scene([(3, 3, BASE_A), (7, 7, BASE_C)])
    labels_c, clusters_c, state = discover_sources(fc, ec, state)
    assert labels_c[7, 7] != labels_c[3, 3]
    assert {int(labels_c[3, 3]), int(labels_c[7, 7])} == {c["cluster_id"] for c in clusters_c}


def test_kmeans_survives_bit_identical_features():
    """Saturated sigmoids pool to identical vectors; k-means++ must not crash
    the stream on a zero probability vector (review finding, critical)."""
    features = np.zeros((N, F), dtype=np.float32)
    energy = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for r, c in [(2, 2), (2, 3), (3, 2), (3, 3), (9, 9)]:
        features[r * GRID_W + c] = BASE_A  # five ACTIVE cells, ONE distinct vector
        energy[r, c] = 20.0
    labels, clusters, _ = discover_sources(features, energy, None)  # must not raise
    assert (labels[energy > 0] != SILENCE_LABEL).all()
    assert len(clusters) >= 1


def test_track_ids_are_recycled_and_stay_uint8_safe():
    """Ids come from a reuse pool <= 254: no unbounded growth, no modulo
    collision between the label map and cluster_id (review finding)."""
    state = None
    seen_ids = set()
    # Churn many births/deaths: alternate two different single-source scenes
    # far apart with dissimilar features, letting each age out repeatedly.
    for i in range(30):
        base = BASE_A if i % 2 == 0 else BASE_B
        pos = (2, 2) if i % 2 == 0 else (11, 11)
        f, e = _scene([(pos[0], pos[1], base)])
        for _ in range(TRACK_MAX_MISSED + 2):  # let the other track die off
            labels, clusters, state = discover_sources(f, e, state)
        seen_ids.update(c["cluster_id"] for c in clusters)
    assert max(seen_ids) <= 10, f"ids must be recycled, saw {sorted(seen_ids)}"
    assert SILENCE_LABEL not in seen_ids


def test_spatially_close_new_source_is_not_identity_theft():
    """A different instrument appearing near a dead track's last position
    must get a NEW id: the centered-cosine gate must reject the match even
    though the spatial penalty is tiny (review finding — the raw-cosine gate
    was vacuous for all-positive features)."""
    f1, e1 = _scene([(5, 5, BASE_A)])
    labels1, clusters1, state = discover_sources(f1, e1, None)
    id_a = clusters1[0]["cluster_id"]
    # One silent window: track A now missed=1, still alive with its centroid.
    _, _, state = discover_sources(
        np.zeros((N, F), dtype=np.float32), np.zeros((GRID_H, GRID_W), dtype=np.float32), state
    )
    # BASE_B at almost the same spot: disjoint band pattern, centered cosine -1.
    f2, e2 = _scene([(5, 6, BASE_B)])
    labels2, clusters2, state = discover_sources(f2, e2, state)
    assert clusters2[0]["cluster_id"] != id_a


def test_missed_track_keeps_colour_ownership():
    """A briefly-missed track still owns its colour; a new source must not
    steal it and collide on re-match (review finding)."""
    f1, e1 = _point_scene([(3, 3, BASE_A), (10, 11, BASE_B)])
    labels1, clusters1, state = discover_sources(f1, e1, None)
    id_b = int(labels1[10, 11])
    color_b = next(c["rgb"] for c in clusters1 if c["cluster_id"] == id_b)

    # B misses this window while a NEW source C appears elsewhere.
    BASE_C = _unit(np.abs(np.sin(np.arange(F) * 0.37)).astype(np.float32))
    f2, e2 = _point_scene([(3, 3, BASE_A), (6, 6, BASE_C)])
    _, clusters2, state = discover_sources(f2, e2, state)

    known_ids = {c["cluster_id"] for c in clusters1}
    for c in clusters2:
        if c["cluster_id"] not in known_ids:  # the newcomer(s)
            assert c["rgb"] != color_b, "missed track's colour must not be reassigned"


def test_fewer_active_cells_than_k_is_fine():
    features = np.zeros((N, F), dtype=np.float32)
    features[5 * GRID_W + 5] = BASE_A
    energy = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    energy[5, 5] = 10.0
    labels, clusters, _ = discover_sources(features, energy, None)
    assert len(clusters) == 1  # k = min(3, n_active) — empty clusters allowed
    assert labels[5, 5] == clusters[0]["cluster_id"]


def test_determinism_same_input_same_output():
    f, e = _scene([(3, 3, BASE_A), (10, 11, BASE_B)])
    l1, c1, _ = discover_sources(f, e, None)
    l2, c2, _ = discover_sources(f, e, None)
    assert (l1 == l2).all()
    assert [c["centroid_x"] for c in c1] == [c["centroid_x"] for c in c2]


def test_region_weights_for_track_roundtrip():
    f, e = _scene([(3, 3, BASE_A)])
    labels, clusters, _ = discover_sources(f, e, None)
    # fixed k=3 may split the blob; pick the cluster that owns cell (3,3)
    target = int(labels[3, 3])
    assert target in {c["cluster_id"] for c in clusters}
    w = region_weights_for_track(labels, target)
    assert w.shape == (GRID_H, GRID_W)
    assert w[3, 3] == 1.0
    assert w[(e == 0)].sum() == 0.0


def test_centroids_are_normalized_frame_coordinates():
    f, e = _scene([(3, 3, BASE_A)])
    _, clusters, _ = discover_sources(f, e, None)
    c = clusters[0]
    assert 0.0 <= c["centroid_x"] <= 1.0 and 0.0 <= c["centroid_y"] <= 1.0
    # blob centered at col 3.5/14, row 3.5/14
    assert abs(c["centroid_x"] - 0.25) < 0.08
    assert abs(c["centroid_y"] - 0.25) < 0.08
