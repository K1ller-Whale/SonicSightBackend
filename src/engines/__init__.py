"""Engine adapters.

An engine adapter binds one model to the streaming server. The contract is
the return shape of the original eval_stream_window (inference.py):

    eval_stream_window(audio_window, frames) -> {
        "left_audio":    np.float32 [window_samples]   (model output rate),
        "right_audio":   np.float32 [window_samples],
        "left_heatmap":  np.float32 [H, W] in [0, 1],
        "right_heatmap": np.float32 [H, W] in [0, 1] or None (heatmap_count == 1),
        "diag":          dict or None,
    }

`frames` is whatever the model's frame_selection produced:
    "centered_triple"  -> (left_frames[3], right_frames[3]) of PIL images
    "consecutive_span" -> [full_img, ...] list of num_frames PIL images

What the two audio streams MEAN differs per model (left/right region vs
on-screen/off-screen); the labels come from ModelSpec.stream_labels, never
from the field names.
"""


class EngineAdapter:
    """Minimal interface every engine adapter implements."""

    id = None  # ModelSpec.id this engine serves

    def load(self):
        raise NotImplementedError

    @property
    def is_loaded(self):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplementedError

    def eval_stream_window(self, audio_window, frames):
        raise NotImplementedError
