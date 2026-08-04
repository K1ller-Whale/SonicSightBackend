# Adding a model to SonicSight

A model is a registry entry, not an `if` statement. Everything the server and
client need to know about a model lives in two places:

- **server:** a `ModelSpec` in `src/model_registry.py` + an engine adapter in
  `src/engines/`
- **client:** a `ModelProfile` in
  `app/src/main/java/.../data/model/ModelProfile.kt` (SonicSightMobile)

Adding a third model touches those two files, possibly the proto (only if the
model needs a capture format that doesn't exist yet), and nothing else.

---

## 1. The ModelSpec (server)

Frozen dataclass; one entry in `REGISTRY`. Every field is a **validated
constant**, not a tunable — the numbers you ship are the numbers you measured.
Both existing models produced confident-looking wrong output when someone
"reasonably" changed a window or resolution parameter, so the registry is
deliberately hostile to knobs.

| field | meaning | sonicsight | multisensory |
|---|---|---|---|
| `id` | metadata value + echo | `sonicsight` | `multisensory` |
| `display_name` | what the user points the camera at | Music & Instruments | Speech |
| `engine_factory` | lazy engine constructor | — | — |
| `frame_rate` | client capture fps | 8 | 30 |
| `capture_sample_rate` | PCM rate on the wire | 11025 | 22050 |
| `frame_kind` | `left_right_halves` \| `full_letterboxed` | halves | full |
| `audio_chunk_field` | which StreamChunk field carries audio | `audio_pcm` | `audio_pcm_hi` |
| `model_sample_rate` | model-internal rate (engine converts at its boundary) | 11025 | 21000 |
| `window_samples` | inference window **at the wire rate** | 65536 | 46352 |
| `hop_samples` | OLA hop at the wire rate, exact int | 1378 | 5512 |
| `num_frames` | frames per window | 3 | 63 |
| `frame_selection` | `centered_triple` \| `consecutive_span` | triple | span |
| `frame_ring_cap` | ring size ≥ num_frames + lag margin | 60 | 90 |
| `early_min_samples` | == window_samples disables early mode | 11025 | 46352 |
| `output_sample_rate` | playback rate the client configures | 11025 | 22050 |
| `heatmap_count` | 2 = left/right, 1 = single in left_heatmap | 2 | 1 |
| `stream_labels` | what the two audio streams MEAN | Left/Right | On-screen/Off-screen |
| `confidence_gated` | server may withhold a meaningless heatmap | no | yes |

Rules worth internalizing:

- **The buffer and OLA run at the wire rate end to end.** If your model wants a
  different internal rate, resample inside the engine, in and out (see the
  multisensory adapter's 22050↔21000 boundary). Never make the streaming
  pipeline itself rate-aware.
- **`hop_samples` must exceed your measured inference time.** The multisensory
  model runs 142–148 ms/window; a 125 ms hop falls behind ~20 ms per window and
  the lag compounds. Measure, then pick the hop.
- **`frame_ring_cap`** must hold `num_frames` plus the frames that arrive while
  the window lags real time — ring too small silently evicts frames the next
  window still needs.
- **Early-inference mode** (partial zero-padded windows before the buffer fills)
  is an optimization for long windows. For a short window, disable it
  (`early_min_samples = window_samples`) — the machinery has bitten before.

## 2. The engine adapter (server)

`src/engines/` — implement `EngineAdapter`:

```python
class MyEngine(EngineAdapter):
    id = "mymodel"
    def load(self): ...                # heavy: checkpoints, session/graph. Raise on failure;
                                       # load_all_engines() logs it and HealthCheck omits the id.
    @property
    def is_loaded(self): ...
    @property
    def device(self): ...
    def eval_stream_window(self, audio_window, frames): ...
```

`eval_stream_window` contract (identical to the original Sound of Pixels path):

- **in:** `audio_window` — float32 mono in [-1,1], `window_samples` long, at the
  wire rate. `frames` — `(left[3], right[3])` PIL images for `centered_triple`,
  or a flat list of `num_frames` full PIL images for `consecutive_span`.
- **out:** dict with `left_audio`, `right_audio` (float32, [-1,1], same length
  and rate as the input window), `left_heatmap`, `right_heatmap` (float32 [H,W]
  in [0,1]; `right_heatmap=None` when `heatmap_count == 1`; `left_heatmap=None`
  when confidence-gated off), `diag` (dict or None), and optionally
  `confidence` (float in [0,1], shipped as `cam_confidence`).

Import your ML framework **inside `load()`**, not at module level — the module
must import cleanly on machines that don't have your runtime, so the rest of
the server still works and tests still run.

Heatmaps: return any square map in [0,1]; the server quantizes to uint8 and the
client infers the grid side from the byte count. 56×56 is the convention.

## 3. The ModelProfile (client)

Mirror of the spec, in `ModelProfile.kt`: frame interval, decimation factor from
44100 (must be an exact integer: 4→11025, 2→22050), anti-alias cutoff
(< 44100/(2·factor)), wire/playback rate, frame kind, heatmap count, the two
stream labels, the legend meaning labels, confidence gating, and the expected
first-result time for honest buffering copy. Add the entry to
`ModelProfile.ALL`; the segmented model switch, solo chips, legend, and
buffering copy all read from the profile — no other client code should need to
know your model exists.

## 4. The proto (only if capture differs)

The contract is already model-generic: `model_id` echo, `heatmap_count`,
`cam_confidence`, size-agnostic heatmap bytes, and two audio fields
(`audio_pcm` @ 11025, `audio_pcm_hi` @ 22050). A third model that captures like
an existing one needs **no proto change**. If it genuinely needs a new capture
format, add fields (never renumber), keep both proto copies byte-identical, and
regenerate stubs on both sides.

## 5. Validation expectations

Before a model ships in the registry, it should have the equivalent of the
multisensory measurement record: a correctness experiment that would FAIL if
the model were wired wrong (not just "output looks plausible"), measured
inference latency on target hardware (sets `hop_samples`), and VRAM alongside
the already-resident models. Write the numbers into the spec's comments with
the evidence. See TESTPLAN.md for the template.
