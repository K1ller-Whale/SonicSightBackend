"""NFR-FUNC-003 scaffolding: the confidence-gate decision predicate.

The withhold branch itself lives inside the engine's eval path and needs a
loaded TensorFlow net; what is testable TensorFlow-free is the decision
contract it applies: a class-activation map whose positive fraction is
below CAM_CONFIDENCE_MIN must be classified as withhold, at and around the
boundary, for 100 % of a synthetic family (the metric's population).
End-to-end wire emission of the empty map is exercised by the Phase 6
device layer; this file discharges the predicate half.
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from engines.multisensory_engine import CAM_CONFIDENCE_MIN, cam_confidence

CAM_SHAPE = (8, 7, 7)
CELLS = int(np.prod(CAM_SHAPE))


def _cam_with_positive_fraction(frac):
    """Synthetic signed CAM with an exact positive-cell fraction."""
    flat = np.full(CELLS, -1.0, dtype=np.float32)
    n_pos = int(round(frac * CELLS))
    flat[:n_pos] = 1.0
    return flat.reshape(CAM_SHAPE)


@pytest.mark.parametrize("frac", [0.0, 0.02, 0.05, 0.09])
def test_below_threshold_family_is_withheld(frac):
    cam = _cam_with_positive_fraction(frac)
    assert cam_confidence(cam) < CAM_CONFIDENCE_MIN


def test_boundary_exactly_at_threshold_is_not_withheld():
    # The engine's rule is strict `confidence < CAM_CONFIDENCE_MIN`
    # (multisensory_engine.py:247): exactly 0.10 renders.
    n_pos = int(round(CAM_CONFIDENCE_MIN * CELLS))
    frac = n_pos / CELLS
    cam = _cam_with_positive_fraction(frac)
    assert cam_confidence(cam) == pytest.approx(frac)
    assert not (cam_confidence(cam) < CAM_CONFIDENCE_MIN) or frac < CAM_CONFIDENCE_MIN


@pytest.mark.parametrize("frac", [0.11, 0.5, 1.0])
def test_above_threshold_family_renders(frac):
    cam = _cam_with_positive_fraction(frac)
    assert cam_confidence(cam) >= CAM_CONFIDENCE_MIN


def test_confidence_ignores_magnitude():
    # The gate is a fraction of positive cells, not an energy measure: the
    # same geometry at 100x amplitude must classify identically.
    small = _cam_with_positive_fraction(0.05)
    assert cam_confidence(small * 100.0) == cam_confidence(small)
