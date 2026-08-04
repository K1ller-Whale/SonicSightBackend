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
    # window/hop are in the WIRE domain (capture_sample_rate): the streaming
    # buffer and overlap-add run at the wire rate end to end. An engine whose
    # model rate differs (multisensory: 21000 vs 22050) converts at its own
    # boundary and returns audio back at the wire rate.
    model_sample_rate: int       # rate the model consumes internally
    window_samples: int          # samples per inference window, at capture_sample_rate
    hop_samples: int             # OLA hop, at capture_sample_rate (exact int, no ms rounding)
    num_frames: int              # frames per inference window
    frame_selection: str         # "centered_triple" | "consecutive_span"
    frame_ring_cap: int          # max frames retained in the ring buffer
    early_min_samples: int      # min audio before early inference; == window_samples disables early mode

    # -- outputs --
    output_sample_rate: int      # PCM rate of separated audio, server -> client
    heatmap_count: int           # 2 = left/right, 1 = single map in left_heatmap
    stream_labels: Tuple[str, str]
    confidence_gated: bool       # True: heatmap withheld when CAM confidence is low

    # Minimum timeline advance (samples at the wire rate) between consecutive
    # inference windows. 1 = pace by frame arrival (Sound of Pixels: 8 fps
    # already IS the 125 ms hop). Models whose frame rate outruns their hop
    # must set this to hop_samples, or a new window exists after every frame
    # and inference runs back-to-back: GPU pegged, the chunk-ingest loop
    # starves, and the stream falls behind real time.
    window_min_advance: int = 1


def _sonicsight_engine():
    from engines.sonicsight_engine import SonicSightEngine

    return SonicSightEngine()


def _multisensory_engine():
    from engines.multisensory_engine import MultisensoryEngine

    return MultisensoryEngine()


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


MULTISENSORY_SPEC = ModelSpec(
    id="multisensory",
    display_name="Speech",
    description=(
        "Multisensory: separates the on-screen speaker from off-screen sound "
        "using audio-video alignment, with one full-frame heatmap showing how "
        "strongly each region explains the audio. Strongest on speech."
    ),
    engine_factory=_multisensory_engine,
    frame_rate=30,                               # ~29.97 fps capture
    capture_sample_rate=22050,                   # 44100/2, exact on-device decimation
    frame_kind="full_letterboxed",               # 1280x720 -> 224x126 + 49 grey rows
    frame_dim=224,
    audio_chunk_field="audio_pcm_hi",
    model_sample_rate=21000,                     # engine resamples 22050<->21000 (21:20)
    window_samples=46352,                        # the LOCKED 2.135 s window, at wire rate
    hop_samples=5512,                            # 250 ms hop: inference is 142-148 ms, so
                                                 # the 125 ms sonicsight hop cannot be
                                                 # sustained and the lag compounds
    num_frames=63,
    frame_selection="consecutive_span",
    frame_ring_cap=90,                           # 63/window + lag margin (~3 s @ 30 fps)
    early_min_samples=46352,                     # == window: early mode disabled; the
                                                 # window is only 2.1 s and the early-mode
                                                 # machinery exists to shave SoP's 5.9 s fill
    output_sample_rate=22050,                    # separated audio returns at wire rate
    heatmap_count=1,
    stream_labels=("On-screen", "Off-screen"),   # NEVER "Left / Right" on this branch
    confidence_gated=True,
    window_min_advance=5512,                     # = hop: at 30 fps every frame offers a
                                                 # newer window; without this the 250 ms
                                                 # hop is never honoured and inference
                                                 # runs back-to-back at ~150 ms
)


REGISTRY = {
    SONICSIGHT_SPEC.id: SONICSIGHT_SPEC,
    MULTISENSORY_SPEC.id: MULTISENSORY_SPEC,
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
