# VoIP Avatar Service

GPU-backed LiveKit + MuseTalk avatar service that joins a LiveKit room, listens to an audio track, runs MuseTalk, and publishes a talking-head video track.

## Project Structure

```
voipavatar/
├── service/
│   ├── __init__.py               # Package init
│   ├── musetalk_adapter.py       # Wrapper around MuseTalk (one avatar per instance)
│   ├── livekit_avatar_service.py # LiveKit <-> MuseTalk glue, entrypoint
│   ├── personas.py               # Persona config loader
│   ├── personas.yaml             # Mapping persona names -> avatar config
│   └── health_server.py          # HTTP health check endpoint
├── assets/
│   └── personas/                 # Avatar video files
├── Dockerfile                    # NVIDIA Docker container definition
├── entrypoint.sh                 # Container entrypoint script
└── docs/
    └── starter.md                # Implementation guide
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

## Adding New Personas

1. Record a 5-15 second video of the person (frontal face, centered)
2. Save to `assets/personas/<name>.mp4`
3. Add entry to `service/personas.yaml`:
   ```yaml
   new_persona:
     display_name: "New Persona"
     video_path: "/opt/MuseTalk/assets/personas/new_persona.mp4"
     inference_config: "/opt/MuseTalk/configs/inference/realtime.yaml"
     bbox_shift: 0
   ```
4. Restart the service with `PERSONA_NAME=new_persona`

## Health Endpoints

- `GET /health` - Returns `{"status": "ok"}`
- `GET /personas` - Returns list of available persona names
