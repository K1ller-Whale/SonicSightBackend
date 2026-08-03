"""Model registry: one ModelSpec per user-selectable model.

Every per-model difference lives in the spec. The numbers are the VALIDATED
constants for each model (window, hop, frame count, sample rates) — frozen
dataclass fields, deliberately not config keys, env vars, or CLI flags.
Deviating from them has produced confident-looking wrong output before
(see HANDOFF section 11.5), so a new model means a new spec entry, not a knob.

The client selects a model per stream via the gRPC metadata key
"sonicsight-model"; its value must equal a ModelSpec.id registered here.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Tuple

from config import AUD_LEN, AUD_RATE, FRAME_RATE, NUM_FRAMES, IMG_SIZE

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "sonicsight"

# gRPC metadata key carrying the model id. Absent -> DEFAULT_MODEL_ID so
# existing clients keep working unchanged.
MODEL_METADATA_KEY = "sonicsight-model"


@dataclass(frozen=True)
class ModelSpec:
    # identity
    id: str
    display_name: str
    description: str
    # () -> engine adapter instance. Creation must be cheap and side-effect
    # free; heavyweight work happens in the engine's load().
    engine_factory: Callable

    # -- capture profile the client must use (mirrored by the mobile app) --
    frame_rate: int              # frames per second the client sends
    capture_sample_rate: int     # PCM rate on the wire, client -> server
    frame_kind: str              # "left_right_halves" | "full_letterboxed"
    frame_dim: int               # JPEG side length (square)
    audio_chunk_field: str       # StreamChunk field carrying this model's audio

    # -- server-side window --
    model_sample_rate: int       # rate the model consumes
    window_samples: int          # samples per inference window, at model rate
    hop_samples: int             # OLA hop, at model rate (exact int, no ms rounding)
    num_frames: int              # frames per inference window
    frame_selection: str         # "centered_triple" | "consecutive_span"
    frame_ring_cap: int          # max frames retained in the ring buffer
    early_min_samples: int      # min audio before early inference; == window_samples disables early mode

    # -- outputs --
    output_sample_rate: int      # PCM rate of separated audio, server -> client
    heatmap_count: int           # 2 = left/right, 1 = single map in left_heatmap
    stream_labels: Tuple[str, str]
    confidence_gated: bool       # True: heatmap withheld when CAM confidence is low


def _sonicsight_engine():
    from engines.sonicsight_engine import SonicSightEngine

    return SonicSightEngine()


SONICSIGHT_SPEC = ModelSpec(
    id="sonicsight",
    display_name="Music & Instruments",
    description=(
        "Sound of Pixels: separates by screen region. The left and right "
        "halves of the frame become two audio tracks with per-region heatmaps. "
        "Strongest on music and instruments."
    ),
    engine_factory=_sonicsight_engine,
    frame_rate=FRAME_RATE,                       # 8
    capture_sample_rate=AUD_RATE,                # 11025
    frame_kind="left_right_halves",
    frame_dim=IMG_SIZE,                          # 224
    audio_chunk_field="audio_pcm",
    model_sample_rate=AUD_RATE,                  # 11025
    window_samples=AUD_LEN,                      # 65536
    hop_samples=int(round(AUD_RATE / FRAME_RATE)),  # 1378, same value OverlapAddBuffer defaulted to
    num_frames=NUM_FRAMES,                       # 3
    frame_selection="centered_triple",
    frame_ring_cap=60,
    early_min_samples=AUD_RATE,                  # 1 second, the pre-registry threshold
    output_sample_rate=AUD_RATE,                 # 11025
    heatmap_count=2,
    stream_labels=("Left", "Right"),
    confidence_gated=False,
)


REGISTRY = {
    SONICSIGHT_SPEC.id: SONICSIGHT_SPEC,
    # The multisensory spec is added together with its engine adapter.
}


_engines = {}


def get_engine(model_id):
    """Return the (cached) engine adapter for a registered model id."""
    if model_id not in _engines:
        _engines[model_id] = REGISTRY[model_id].engine_factory()
    return _engines[model_id]


def load_all_engines():
    """Try to load every registered engine. A model whose checkpoint or
    runtime is unavailable is skipped with a warning instead of killing the
    server; HealthCheck advertises what actually loaded."""
    for model_id in REGISTRY:
        try:
            get_engine(model_id).load()
            logger.info("Model '%s' loaded.", model_id)
        except Exception:
            logger.warning("Model '%s' failed to load; streams requesting it "
                           "will be rejected.", model_id, exc_info=True)


def loaded_model_ids():
    """Ids of models whose engines report loaded, in registry order."""
    out = []
    for model_id in REGISTRY:
        try:
            if get_engine(model_id).is_loaded:
                out.append(model_id)
        except Exception:
            pass
    return out
