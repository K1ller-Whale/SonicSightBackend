# 🔊 SonicSight Backend

<div align="center">
  <h3>Deep Learning-Based Audio-Visual Source Separation</h3>
  <i>Powered by PyTorch, FastAPI, and gRPC</i>
</div>

---

## 📖 Overview

SonicSight Backend is the core artificial intelligence inference server for the SonicSight platform. It uses advanced deep learning—based on the *Sound-of-Pixels* architecture—to perform robust audio-visual source separation. By analyzing both the audio waveform and the accompanying video frames simultaneously, the model can spatially localize sounds within the video and isolate specific audio sources from complex mixed environments.

This backend serves predictions efficiently via two distinct interfaces:
1. **FastAPI (REST)**: Handles batched video processing through standard HTTP POST endpoints.
2. **gRPC (Bidirectional Streaming)**: Handles ultra low-latency, near real-time stream processing, sending audio chunks and raw JPEGs directly to inference without relying on intermediate disk I/O or FFmpeg layers.

## ✨ Features

- 👀 **Audio-Visual Learning**: Processes video frames (vision) and audio (sound) simultaneously to localize and separate multi-source sound mixtures into isolated components.
- ⚡ **High-Performance Inference**: Powered by an optimized PyTorch pipeline handling both forward inference and reverse STFT synthesis natively.
- 📡 **Low Latency gRPC Streaming**: Enables near real-time processing directly from mobile applications by streaming raw arrays and optimizing bandwidth (e.g., uint8 heatmap quantization).
- 🌐 **Robust REST API**: Provides a seamless REST API for standard integrations.
- 🧠 **Dynamic Resource Management**: Handles background tasks, memory cleanups, GPU management, and automatically falls back to CPU if a CUDA device is unavailable.

## 🏗️ Model Architecture (Sound of Pixels)

The inferencing pipeline implements a multi-module deep learning architecture:
- **Visual Network**: `ResNet18Dilated` extracts contextual visual semantics from input frames.
- **Audio Network**: `UNet7` extracts high-dimensional spectrogram features from the mixed audio inputs.
- **Synthesizer**: `Linear` synthesizes the learned features into independent audio masks.
- **Configurations**: Processes at a sample rate of `11025 Hz`, an STFT frame size of `1022`, and a frame extraction target of `8 FPS`.

The model successfully returns spatial localizations (Heatmaps) defining where the target sound occurs and the separated Audio Tracks (Left and Right Channels).

## 🗂️ Project Structure

```text
SonicSightBackend/
├── src/
│   ├── main.py                 # FastAPI application and REST endpoints
│   ├── grpc_server.py          # gRPC streaming server implementation
│   ├── run_servers.py          # Entrypoint to run both FastAPI and gRPC servers
│   ├── inference.py            # AI deep learning inference operations and streaming buffer
│   ├── video_preprocessor.py   # Legacy video and audio extraction tools
│   ├── config.py               # Central architecture constants and model configurations
│   ├── models/                 # Model PyTorch class definitions (UNet, ResNet, etc.)
│   ├── utils/                  # Assorted AI utilities (STFT manipulation, audio tools)
│   ├── ckpt/                   # PyTorch pre-trained model weights directory (.pth files)
│   └── outputs/                # Temporary backend storage for REST batched tasks
├── proto/                      # gRPC Protocol Buffers shared contract defining communication
└── requirements.txt            # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.13
- (Optional but Recommended) NVIDIA GPU with CUDA Toolkit installed for hardware-accelerated inference.

### Installation

1. **Clone the repository and access the backend directory:**
   ```bash
   git clone <repository-url>
   cd SonicSightV1/SonicSightBackend
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Model Weights:**
   Ensure your `.pth` model checkpoints are downloaded and correctly placed inside the `src/ckpt/` directory.

### Running the Server

To launch both the FastAPI endpoint and the gRPC server simultaneously:

```bash
python src/run_servers.py
```

By default:
- **FastAPI** will be available at: `http://localhost:8000/`
- **gRPC Server** will listen on port: `50051`

## 📡 API Usage

### REST Endpoint
- `GET /`: Health check to verify model loading status.
- `POST /predict`: Submit an MP4 video file (`multipart/form-data`). Returns JSON containing Base64 encoded separated WAV files and spatial heatmaps.

### gRPC Endpoints
- `HealthCheck`: Verifies the server stream and CUDA device stability.
- `StreamProcess`: Specialized bidirectional streaming endpoint. Accepts streams of `StreamChunk` (PCM and JPEG) and yields `StreamResult` continuously back to the client.

## 🛡️ License
Distribute your license here. All rights reserved by the original project contributors.
