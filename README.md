# 🔊 SonicSight Backend

<div align="center">
  <h3>Deep Learning-Based Audio-Visual Source Separation</h3>
  <i>Powered by PyTorch, TensorFlow, FastAPI, and gRPC</i>
</div>

---

## 📖 Overview

SonicSight Backend is the inference server for the SonicSight platform. It performs
real-time audio-visual source separation: the phone streams camera frames plus
microphone audio, the server returns separated audio plus a heatmap showing where
the sound is coming from.

The server hosts **multiple models behind one registry**. The client picks one
model per stream; that model alone produces both the separated audio and the
heatmap. There is no fusion, gating, or blending between models.

| model id | separates | streams mean | spatial payload | strongest on |
|---|---|---|---|---|
| `sonicsight` | by screen halves (Sound of Pixels) | Left / Right half of frame | 2 heatmaps, 56×56 uint8 | music, instruments |
| `sonicsight-pixel` | by touched region (same checkpoint, touch mode) | Selection / Everything else + mixture | native 14×14 energy map + source clusters | music, instruments |
| `multisensory` | by audio-video alignment (Owens & Efros) | On-screen / Off-screen | 1 heatmap (CAM), 56×56 uint8 | speech |

## 👆 Touch (pixel) mode

Verified model facts this mode is built on: the visual feature grid is
**14×14** (stride-16 dilated ResNet-18 on 224×224; only ~14×8 cells carry
scene content after 16:9 letterboxing), **K = 32** feature channels
(`NUM_CHANNELS`, not the paper's 16), and the synthesizer is a 33-parameter
linear inner product — which is why this is cheap:

- The two expensive nets run **once per window** (`eval_pixel_window`); the
  intermediates are cached (~**9.5 MB/window** fp32: `feat_sound`
  [1,32,256,256], pooled image features, mixture magnitude+phase; ring of 3
  ≈ 28 MB, a frozen window pins one extra slot).
- **Any region's audio is one bmm + sigmoid + unwarp + ISTFT** against the
  cache (`pixel_cache.synthesize_regions`) — a tap never costs a network
  forward pass. Regions are renormed as a set in the linear domain (the
  deployed halves order), so they partition the mixture and faders sum back
  to the original.
- The per-cell **energy map** ships at the native grid with explicit
  `grid_width`/`grid_height` (196 bytes uint8) — the server never resamples
  up to fake resolution. Source discovery (silence gate → fixed-3 k-means →
  persistent track ids ≤254 with CVD-safe colours) rides `cluster_labels` +
  `SourceCluster`.
- Queries travel **in-band** (`PixelQuery` on the stream that owns the
  cache), answered with `sequence_number = 0`; `freeze` and
  `request_clusters` are ~1 s TTL latches. `PixelAudio.energy` is the
  region's **fraction of window mixture energy** (0..1).

The deployed halves path is untouched by all of this — `mode="halves"` is
the default and dispatches around the pixel loop entirely. See
[MODELS.md](MODELS.md) for the region-mode contract and
[PIXEL_PLAN.md](../PIXEL_PLAN.md) for the full design record.
`pixel_ab_harness.py` renders the pending listening questions (crop-halves
vs region-halves, ratio vs binary masks, energy vs activation maps) from a
fixed clip offline.

Two interfaces:
1. **gRPC (bidirectional streaming)** — the primary path. `StreamProcess` accepts
   `StreamChunk` (PCM + JPEG) and yields `StreamResult` continuously.
2. **FastAPI (REST)** — batched video file processing via `POST /predict`
   (Sound of Pixels only).

## 🧭 Model selection

- The client sends gRPC metadata key **`sonicsight-model`** with value
  `sonicsight` or `multisensory`. Absent → `sonicsight` (old clients keep working).
  Unknown value or unloaded model → `FAILED_PRECONDITION`.
- Every `StreamResult` echoes `model_id`; the client drops results whose id does
  not match its current selection (this is how in-flight results from a cancelled
  stream are discarded after a switch).
- `HealthCheck` returns `loaded_models` — the ids whose checkpoints actually loaded.
- Switching models is always **cancel the stream, reopen with new metadata**.
  The capture profiles are incompatible mid-stream.

Every per-model constant (window, hop, frame count, sample rates, labels) lives in
a frozen `ModelSpec` in `src/model_registry.py`. These are validated constants,
deliberately not config keys. See [MODELS.md](MODELS.md) for how to add a model.

## 📡 Wire format facts (source of truth: the code)

- Proto contract: **`sonicsight.proto` at the repo root** (not `proto/`). The
  mobile repo keeps a byte-identical copy at `app/src/main/proto/sonicsight.proto`;
  if you change one, change the other in the same change set and regenerate stubs
  (`python -m grpc_tools.protoc -I. --python_out=src --grpc_python_out=src sonicsight.proto`).
- Streaming heatmaps: **56×56 uint8** (3136 bytes/side), per-window normalized,
  quantized from [0,1]. The client infers the grid side from sqrt(byte count),
  so any square size decodes.
- `ProcessVideo` (REST-style one-shot) heatmaps: **224×224 float32 little-endian**
  (200704 bytes/side).
- Streaming separated audio: PCM16 mono at the model's wire rate — 11025 Hz
  (`sonicsight`) or 22050 Hz (`multisensory`).
- `cam_confidence` (field 17): raw-CAM positive fraction for confidence-gated
  models. Below 0.10 the heatmap is withheld (sent empty) because the alignment
  head is reporting "this audio does not match this video".

## 🗂️ Project Structure

```text
SonicSightBackend/
├── sonicsight.proto            # gRPC contract (byte-identical copy in mobile repo)
├── replay_client.py            # deterministic file-replay client for the streaming path
├── src/
│   ├── run_servers.py          # entrypoint: FastAPI + gRPC, loads all engines
│   ├── grpc_server.py          # StreamProcess / ProcessVideo / HealthCheck
│   ├── model_registry.py       # ModelSpec + REGISTRY (all per-model constants)
│   ├── engines/                # engine adapters (sonicsight, multisensory)
│   ├── inference.py            # Sound of Pixels engine + StreamingBuffer
│   ├── overlap_add_buffer.py   # timeline-driven OLA with crossfade
│   ├── main.py                 # FastAPI application (Sound of Pixels only)
│   ├── config.py               # Sound of Pixels constants + env toggles
│   ├── models/                 # PyTorch nets (UNet7, ResNet18dilated, synthesizer)
│   ├── utils/                  # STFT helpers, transforms
│   └── ckpt/                   # PyTorch weights (sound/frame/synthesizer_best.pth)
├── tests/
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- NVIDIA GPU strongly recommended (validated on a GTX 1660 Ti 6 GB).
- **For the multisensory model:** TensorFlow 2.x with GPU requires **WSL2/Linux**
  (TF ≥ 2.11 has no native Windows GPU support). The whole server runs in one
  process so both models share one CUDA context budget — on a 6 GB card that is
  the difference between fitting and not. Install per the multisensory repo's
  environment notes (`tensorflow[and-cuda]==2.21`, `tf_keras`, `tf_slim`, and the
  `LD_LIBRARY_PATH` venv fix).

### Installation

```bash
pip install -r requirements.txt
```

Place the Sound of Pixels checkpoints in `src/ckpt/`
(`sound_best.pth`, `frame_best.pth`, `synthesizer_best.pth`).

### Multisensory model configuration (optional)

The multisensory engine is loaded lazily; if its repo or checkpoint is missing
the server still starts and `HealthCheck` simply omits it from `loaded_models`.

| env var | default | meaning |
|---|---|---|
| `MULTISENSORY_ROOT` | sibling `../multisensory` checkout | repo containing `src/sep_video.py` |
| `MULTISENSORY_CHECKPOINT` | `<root>/results/nets/sep/full/net.tf-160000` | separation checkpoint |
| `MS_GPU` | `0` | GPU index; `cpu` to unpin |
| `MS_GPU_ALLOW_GROWTH` / `MS_GPU_MEM_FRACTION` | `1` / unset | TF VRAM behaviour (required for sharing the card with PyTorch) |

`TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512` is set as a default inside the multisensory
repo itself; do not remove it (uncapped, the allocator holds ~2.9 GB for a
~0.5 GB working set).

### Running the Server

```bash
python src/run_servers.py
```

- **FastAPI**: `http://localhost:8000/`
- **gRPC**: port `50051`

Phones cannot reach a WSL2 server directly: either set `networkingMode=mirrored`
in `.wslconfig` or add a `netsh interface portproxy` rule for 50051.

### Verifying the streaming path

`replay_client.py` streams a video file through the identical `StreamProcess`
path deterministically (no phone, no mic, no room):

```bash
python replay_client.py --video myclip.mp4
```

Use it for regression gates: same input, compare output bytes across code
changes. See [TESTPLAN.md](TESTPLAN.md).

## 🛡️ License
Distribute your license here. All rights reserved by the original project contributors.
