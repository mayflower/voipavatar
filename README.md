# VoIP Avatar Service

GPU-backed LiveKit + MuseTalk avatar service that joins a LiveKit room, listens to an audio track, runs MuseTalk, and publishes a talking-head video track.

## Project Structure

```
voipavatar/
├── service/
│   ├── __init__.py               # Package init
│   ├── musetalk_adapter.py       # MuseTalk wrapper (model loading, inference)
│   ├── livekit_avatar_service.py # LiveKit <-> MuseTalk glue, main entrypoint
│   ├── personas.py               # Persona config loader
│   ├── personas.yaml             # Mapping persona names -> avatar config
│   ├── health_server.py          # HTTP health check endpoint
│   └── test_avatar.py            # Standalone pipeline test script
├── scripts/
│   └── download_samples.py       # Download sample audio for testing
├── assets/                       # Avatar videos and audio samples
├── Dockerfile                    # NVIDIA Docker container definition
├── entrypoint.sh                 # Container entrypoint script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Components

### musetalk_adapter.py
Wraps MuseTalk's real-time pipeline into a clean Python class:
- `MuseTalkAvatar.__init__()`: Load models (UNet, VAE, Whisper)
- `MuseTalkAvatar.prepare_avatar()`: Process reference video
- `MuseTalkAvatar.generate_from_audio_chunk()`: Convert audio PCM to video frames

### livekit_avatar_service.py
Main service that:
1. Connects to LiveKit room
2. Subscribes to remote audio tracks
3. Processes audio through MuseTalk
4. Publishes video track with generated frames

### personas.yaml
Configuration file mapping persona names to:
- `display_name`: Human-readable name
- `video_path`: Path to avatar reference video
- `inference_config`: MuseTalk config path
- `bbox_shift`: Face crop adjustment

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LIVEKIT_URL` | LiveKit server WebSocket URL |
| `LIVEKIT_API_KEY` | API key for authentication |
| `LIVEKIT_API_SECRET` | API secret for token generation |
| `LIVEKIT_ROOM` | Room name to join |
| `LIVEKIT_IDENTITY` | Bot identity in the room |
| `PERSONA_NAME` | Key into personas.yaml |

## Usage

### Docker (Production)

```bash
docker build -t voipavatar .

docker run --gpus all --rm \
  -e LIVEKIT_URL="wss://your.livekit.url" \
  -e LIVEKIT_API_KEY="..." \
  -e LIVEKIT_API_SECRET="..." \
  -e LIVEKIT_ROOM="demo-room" \
  -e LIVEKIT_IDENTITY="avatar-bot" \
  -e PERSONA_NAME="default" \
  -p 8080:8080 \
  voipavatar
```

### Local Development (with conda)

```bash
conda activate voipavatar
cd service
python livekit_avatar_service.py --persona default --room demo-room
```

## Avatar Video Requirements

The avatar video is the reference footage that MuseTalk uses to generate lip-synced output. Quality of this video directly affects output quality.

### Technical Requirements

| Property | Requirement | Recommended |
|----------|-------------|-------------|
| Duration | 2-30 seconds | 3-10 seconds |
| Resolution | 256x256 minimum | 512x512 or higher |
| Frame rate | 24-30 fps | 25 fps |
| Format | MP4, MOV, AVI | MP4 (H.264) |
| Face coverage | Face clearly visible | Face fills 40-60% of frame |

### Content Guidelines

- **Framing**: Head and shoulders, front-facing camera
- **Lighting**: Even, soft lighting on face (no harsh shadows)
- **Background**: Plain or simple background preferred
- **Expression**: Neutral or slight natural movement
- **Motion**: Small head movements are fine; avoid large motions
- **Audio**: Not required (only video frames are used)

### Good vs Bad Examples

```
✓ Good:                          ✗ Bad:
- Front-facing webcam clip       - Side profile
- Evenly lit office setting      - Strong backlighting
- Person looking at camera       - Looking away/down
- Slight talking motion          - Rapid head movements
- Clear face visibility          - Sunglasses, masks, obstructions
```

### Recording Tips

1. Use a webcam or phone front camera
2. Position camera at eye level
3. Ensure face is well-lit (natural light or ring light)
4. Record 5-10 seconds of natural talking or slight movement
5. Keep head relatively still but not frozen

## Getting Sample Assets

Download sample audio files for testing:

```bash
# Download sample audio files
python scripts/download_samples.py --audio-only

# Generate synthetic test files (no downloads)
python scripts/download_samples.py --synthetic

# Download everything and update personas.yaml
python scripts/download_samples.py --update-personas
```

**Note**: You must provide your own avatar video. Sample videos cannot be distributed due to likeness rights.

## Testing Without LiveKit

Test the MuseTalk pipeline locally without a LiveKit server:

```bash
# Test with your avatar video and an audio file
python -m service.test_avatar --video your_avatar.mp4 --audio speech.wav -o output.mp4

# Test with synthetic audio (no audio file needed)
python -m service.test_avatar --video your_avatar.mp4 --synthetic -o output.mp4

# Test using a persona from personas.yaml
python -m service.test_avatar --persona default --audio speech.wav -o output.mp4

# Quick component test (no GPU required)
python -m service.test_avatar --test-components
```

## Adding New Personas

1. Record a 3-10 second video following the [Avatar Video Requirements](#avatar-video-requirements)
2. Save to `assets/<name>.mp4`
3. Add entry to `service/personas.yaml`:
   ```yaml
   new_persona:
     display_name: "New Persona"
     video_path: "/app/assets/new_persona.mp4"
     inference_config: "/opt/MuseTalk/configs/inference/realtime.yaml"
     bbox_shift: 0
   ```
4. Test locally: `python -m service.test_avatar --persona new_persona --synthetic`
5. Restart the service with `PERSONA_NAME=new_persona`

### bbox_shift Parameter

The `bbox_shift` parameter adjusts the vertical position of the detected face bounding box:
- **Positive values**: Shift box down (use if chin is cut off)
- **Negative values**: Shift box up (use if forehead is cut off)
- **Typical range**: -10 to +10 pixels
- **Start with 0** and adjust if face alignment looks wrong

## Health Endpoints

- `GET /health` - Returns `{"status": "ok"}`
- `GET /personas` - Returns list of available persona names
