# VoIP Avatar Service

GPU-backed LiveKit + MuseTalk avatar service that joins a LiveKit room, listens to an audio track, runs MuseTalk inference, and publishes a talking-head video track in real-time.

## Features

- **Real-time lip-sync**: Processes audio and generates video at 1.2x realtime on RTX 4090
- **TensorRT acceleration**: 2-3x faster inference with TensorRT-optimized UNet and VAE
- **Smooth transitions**: Natural mouth closing when speech ends (8-frame fadeout)
- **Silence detection**: Skips GPU inference during silent periods
- **Avatar caching**: Pre-computed latents cached to disk for fast startup
- **LiveKit integration**: Full bi-directional audio/video with LiveKit rooms

## Project Structure

```
voipavatar/
├── service/
│   ├── __init__.py               # Package init
│   ├── musetalk_adapter.py       # MuseTalk wrapper (model loading, inference)
│   ├── livekit_avatar_service.py # LiveKit <-> MuseTalk glue, main entrypoint
│   ├── tensorrt_runtime.py       # TensorRT engine wrapper
│   ├── export_tensorrt.py        # ONNX/TensorRT export script
│   ├── gpu_blending.py           # GPU-accelerated frame blending (optional)
│   ├── personas.py               # Persona config loader
│   ├── personas.yaml             # Mapping persona names -> avatar config
│   ├── health_server.py          # HTTP health check endpoint
│   ├── test_avatar.py            # Standalone pipeline test
│   ├── test_realtime.py          # Real-time performance benchmark
│   └── test_gpu_blending.py      # GPU blending benchmark
├── scripts/
│   └── download_samples.py       # Download sample audio for testing
├── Dockerfile                    # NVIDIA Docker container definition
├── entrypoint.sh                 # Container entrypoint script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Performance

Tested on NVIDIA RTX 4090 with 25fps avatar video:

| Configuration | Realtime Ratio | Latency (0.4s chunk) |
|--------------|----------------|----------------------|
| PyTorch FP16 | 0.5x | ~800ms |
| TensorRT FP16 | **1.2x** | ~330ms |

**Key optimizations:**
- TensorRT engines for UNet and VAE decoder
- CUDA streams with pinned memory transfers
- Parallel post-processing (4-thread blending)
- Optimized resize (INTER_LINEAR vs LANCZOS4)

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 11.8+
- MuseTalk repository cloned to `./MuseTalk`
- MuseTalk models downloaded to `./models`
- Conda environment with dependencies

### Setup

```bash
# Clone MuseTalk
git clone https://github.com/TMElyralab/MuseTalk.git

# Create conda environment
conda create -n voipavatar python=3.11
conda activate voipavatar

# Install dependencies
pip install -r requirements.txt
pip install -r MuseTalk/requirements.txt

# Download MuseTalk models (follow MuseTalk README)
```

### Export TensorRT Engines (Recommended)

```bash
cd MuseTalk
PYTHONPATH=. python -m service.export_tensorrt \
    --output-dir ../models/tensorrt \
    --batch-size 1 25 50
```

### Test Real-time Performance

```bash
cd MuseTalk
PYTHONPATH=.:../service python -m service.test_realtime \
    --video /path/to/avatar.mp4 \
    --audio /path/to/speech.wav \
    --tensorrt \
    --chunk-duration 0.4
```

### Run with LiveKit

```bash
export LIVEKIT_URL="wss://your.livekit.url"
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."

python -m service.livekit_avatar_service \
    --persona default \
    --room demo-room \
    --use-tensorrt
```

## Avatar Video Requirements

The avatar video quality directly affects output quality.

| Property | Requirement | Notes |
|----------|-------------|-------|
| **Frame rate** | **25 fps** | Must match output for proper sync |
| Duration | 3-10 seconds | Longer = more memory |
| Resolution | 512-720p | Higher = slower blending |
| Format | MP4 (H.264) | |
| Face coverage | 40-60% of frame | Front-facing, well-lit |

### Preparing Avatar Videos

Convert existing video to optimal format:

```bash
# Convert to 25fps, 640px width
ffmpeg -i input.mp4 -vf "scale=640:-1,fps=25" -c:v libx264 -crf 18 avatar_25fps.mp4
```

**Important**: Use 25fps to ensure audio-video sync. Other frame rates cause drift.

## Audio Processing

- **Sample rate**: 16kHz mono (resampled automatically)
- **Chunk duration**: 0.4s recommended (10 frames at 25fps)
- **Silence threshold**: 0.01 RMS (configurable)

**Why 0.4s chunks?** At 25fps, 0.4s = exactly 10 frames. Using 0.5s would produce 12.5 frames, causing sync drift.

## Configuration

### personas.yaml

```yaml
default:
  display_name: "Default Avatar"
  video_path: "/app/assets/avatar.mp4"
  inference_config: "/opt/MuseTalk/configs/inference/realtime.yaml"
  bbox_shift: 0
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LIVEKIT_URL` | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | API key for authentication |
| `LIVEKIT_API_SECRET` | API secret for token generation |
| `LIVEKIT_ROOM` | Room name to join |
| `PERSONA_NAME` | Key into personas.yaml |

## Docker

```bash
docker build -t voipavatar .

docker run --gpus all --rm \
  -e LIVEKIT_URL="wss://your.livekit.url" \
  -e LIVEKIT_API_KEY="..." \
  -e LIVEKIT_API_SECRET="..." \
  -e LIVEKIT_ROOM="demo-room" \
  -e PERSONA_NAME="default" \
  -v /path/to/models:/app/models \
  -p 8080:8080 \
  voipavatar
```

## Health Endpoints

- `GET /health` - Returns `{"status": "ok"}`
- `GET /personas` - Returns list of available personas

## Troubleshooting

### Video faster than audio
- Use 0.4s chunks (not 0.5s) for 25fps video
- Ensure avatar video is exactly 25fps

### Mouth closes too abruptly
- Smooth fadeout is enabled by default (8 frames)
- Adjust `_silence_fadeout_frames` in musetalk_adapter.py

### TensorRT batch size error
- Default engines support batch 1-50
- For longer audio, process in chunks

### Out of memory
- Reduce avatar video length (fewer cached frames)
- Use smaller resolution avatar video

## License

This project wraps [MuseTalk](https://github.com/TMElyralab/MuseTalk) which has its own license terms. See MuseTalk repository for details.
