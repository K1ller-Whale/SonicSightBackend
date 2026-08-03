"""Sound of Pixels engine adapter.

Pure plumbing around the existing global InferenceEngine. No tensor op,
constant, or ordering changes here — the numerical path must stay
byte-identical to the pre-registry server.
"""

from engines import EngineAdapter
from inference import inference


class SonicSightEngine(EngineAdapter):
    id = "sonicsight"

    def load(self):
        inference.load_model()

    @property
    def is_loaded(self):
        return inference.is_loaded

    @property
    def device(self):
        return str(inference.device) if inference.device else "not loaded"

    def eval_stream_window(self, audio_window, frames):
        return inference.eval_stream_window(audio_window, frames)
