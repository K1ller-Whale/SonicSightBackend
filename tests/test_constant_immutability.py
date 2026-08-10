"""NFR-MAINT-003 scaffolding: constant immutability over the named inventory.

The inventory (per the requirement) is the frozen ModelSpec fields plus the
engine adapters' LOCKED constant blocks. Frozen-ness of the dataclass is
already asserted by the registry tests; this file adds the other half: no
constant in the inventory may have an environment or flag override — the
locked values must not be reachable from configuration (FR-066, TR §8.2).
"""

import dataclasses
import inspect
import os
import re
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import model_registry
from engines import multisensory_engine

LOCKED_NAMES = [
    "VID_DUR", "MODEL_SR", "WIRE_SR", "NUM_SAMPLES", "NUM_FRAMES",
    "CAM_CONFIDENCE_MIN", "HEATMAP_WIRE_SIDE",
]


def test_locked_constants_exist_and_are_plain_numbers():
    for name in LOCKED_NAMES:
        value = getattr(multisensory_engine, name)
        assert isinstance(value, (int, float)), name


def test_no_env_override_reaches_locked_constants():
    src = inspect.getsource(multisensory_engine)
    env_lines = [ln for ln in src.splitlines()
                 if re.search(r"os\.environ|getenv", ln)]
    for name in LOCKED_NAMES:
        offenders = [ln for ln in env_lines if name in ln]
        assert not offenders, f"{name} reachable from environment: {offenders}"


def test_no_env_reads_in_the_registry_module():
    src = inspect.getsource(model_registry)
    assert not re.search(r"os\.environ|getenv", src), (
        "model_registry must hold frozen constants only — no environment "
        "reads (FR-066)")


def test_model_spec_is_frozen():
    spec = model_registry.REGISTRY["sonicsight"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.hop_samples = 1  # type: ignore[misc]
