"""
LiveKit Avatar Service - LiveKit <-> MuseTalk glue and entrypoint.

This module provides the main service that:
1. Connects to a LiveKit room as a bot participant
2. Subscribes to remote audio tracks
3. Feeds audio to MuseTalkAvatar for processing
4. Publishes generated video frames back to the room

The service uses async/await patterns for efficient handling of
real-time audio and video streams.

Environment variables:
    LIVEKIT_URL: WebSocket URL for LiveKit server
    LIVEKIT_API_KEY: API key for authentication
    LIVEKIT_API_SECRET: API secret for token generation
    LIVEKIT_ROOM: Room name to join
    LIVEKIT_IDENTITY: Bot identity in the room
    PERSONA_NAME: Key into personas.yaml for avatar config

Usage:
    python livekit_avatar_service.py [--persona NAME] [--room ROOM]
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from collections import deque

import numpy as np
from livekit import api, rtc

from .musetalk_adapter import MuseTalkAvatar
from .personas import PersonaConfig, PersonaNotFoundError, load_persona


logger = logging.getLogger(__name__)

# Audio configuration
AUDIO_SAMPLE_RATE = 16000  # 16 kHz mono PCM
AUDIO_CHANNELS = 1
AUDIO_CHUNK_DURATION_S = 1.0  # Accumulate 1 second of audio before processing
SAMPLES_PER_CHUNK = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_DURATION_S)

# Video configuration
VIDEO_FPS = 25  # Target frame rate for video publishing


class LiveKitAvatarService:
    """
    Main service class that bridges LiveKit audio streams with MuseTalk video generation.

    This class handles:
    - Connecting to a LiveKit room
    - Subscribing to remote audio tracks
    - Buffering audio into chunks
    - Generating avatar video frames via MuseTalk
    - Publishing video frames to the room
    """

    def __init__(self, persona_config: PersonaConfig) -> None:
        """
        Initialize the LiveKit Avatar Service.

        Args:
            persona_config: Configuration for the avatar persona
        """
        self.persona_config = persona_config
        self.avatar: MuseTalkAvatar | None = None
        self.room: rtc.Room | None = None
        self.video_source: rtc.VideoSource | None = None
        self.video_track: rtc.LocalVideoTrack | None = None

        # Audio buffer for accumulating samples
        self._audio_buffer: deque[np.ndarray] = deque()
        self._audio_buffer_samples = 0

        # Task management
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """
        Initialize the MuseTalk avatar with the configured persona.

        This loads the MuseTalk models and prepares the avatar from the
        reference video. This is a heavy operation and should be done
        once at startup.
        """
        logger.info(f"Initializing avatar for persona: {self.persona_config.name}")

        self.avatar = MuseTalkAvatar(
            inference_config_path=self.persona_config.inference_config,
        )

        self.avatar.prepare_avatar(
            video_path=self.persona_config.video_path,
            bbox_shift=self.persona_config.bbox_shift,
        )

        logger.info(
            f"Avatar initialized: {self.avatar.frame_size[0]}x{self.avatar.frame_size[1]} "
            f"@ {self.avatar.fps} fps"
        )

    async def connect(
        self,
        livekit_url: str,
        room_name: str,
        identity: str,
        api_key: str,
        api_secret: str,
    ) -> None:
        """
        Connect to a LiveKit room and set up audio/video tracks.

        Args:
            livekit_url: WebSocket URL for LiveKit server
            room_name: Name of the room to join
            identity: Bot identity in the room
            api_key: LiveKit API key
            api_secret: LiveKit API secret
        """
        if self.avatar is None:
            raise RuntimeError("Avatar not initialized. Call initialize() first.")

        # Generate access token
        token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(identity)
            .with_name(self.persona_config.display_name)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )

        # Create room and set up event handlers
        self.room = rtc.Room()
        self._setup_room_events()

        # Create video source and track
        width, height = self.avatar.frame_size
        self.video_source = rtc.VideoSource(width, height)
        self.video_track = rtc.LocalVideoTrack.create_video_track(
            "avatar_video", self.video_source
        )

        # Connect to room
        logger.info(f"Connecting to room '{room_name}' as '{identity}'")
        await self.room.connect(livekit_url, token)
        logger.info("Connected to LiveKit room")

        # Publish video track
        await self.room.local_participant.publish_track(
            self.video_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
        )
        logger.info("Published video track")

    def _setup_room_events(self) -> None:
        """Set up event handlers for the LiveKit room."""
        if self.room is None:
            return

        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Subscribed to audio track from {participant.identity}")
                # Start processing audio from this track
                task = asyncio.create_task(self._process_audio_track(track))
                self._tasks.append(task)

        @self.room.on("track_unsubscribed")
        def on_track_unsubscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            logger.info(f"Unsubscribed from track from {participant.identity}")

        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
            logger.info(f"Participant connected: {participant.identity}")

        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            logger.info(f"Participant disconnected: {participant.identity}")

        @self.room.on("disconnected")
        def on_disconnected() -> None:
            logger.info("Disconnected from room")
            self._shutdown_event.set()

    async def _process_audio_track(self, track: rtc.Track) -> None:
        """
        Process audio frames from a subscribed track.

        Accumulates audio samples into chunks and triggers frame generation
        when enough audio has been buffered.

        Args:
            track: The audio track to process
        """
        audio_stream = rtc.AudioStream(track)

        async for frame_event in audio_stream:
            if not self._running:
                break

            # Convert frame to numpy array
            audio_frame = frame_event.frame
            samples = np.frombuffer(audio_frame.data, dtype=np.int16)

            # Handle multi-channel audio (convert to mono)
            if audio_frame.num_channels > 1:
                samples = samples.reshape(-1, audio_frame.num_channels)
                samples = samples.mean(axis=1).astype(np.int16)

            # Resample if necessary
            if audio_frame.sample_rate != AUDIO_SAMPLE_RATE:
                # Simple resampling (for production, use proper resampling)
                ratio = AUDIO_SAMPLE_RATE / audio_frame.sample_rate
                new_length = int(len(samples) * ratio)
                samples = np.interp(
                    np.linspace(0, len(samples), new_length),
                    np.arange(len(samples)),
                    samples,
                ).astype(np.int16)

            # Add to buffer
            self._audio_buffer.append(samples)
            self._audio_buffer_samples += len(samples)

            # Process when we have enough samples
            if self._audio_buffer_samples >= SAMPLES_PER_CHUNK:
                await self._process_audio_buffer()

    async def _process_audio_buffer(self) -> None:
        """
        Process accumulated audio buffer and generate video frames.

        Combines buffered audio chunks, generates avatar frames via MuseTalk,
        and publishes them to the video track.
        """
        if self.avatar is None or self.video_source is None:
            return

        # Combine audio chunks
        audio_chunks = list(self._audio_buffer)
        self._audio_buffer.clear()
        self._audio_buffer_samples = 0

        if not audio_chunks:
            return

        audio_pcm = np.concatenate(audio_chunks)

        # Generate frames from audio
        try:
            frames = await asyncio.get_event_loop().run_in_executor(
                None,
                self.avatar.generate_from_audio_chunk,
                audio_pcm,
                AUDIO_SAMPLE_RATE,
            )
        except Exception as e:
            logger.error(f"Error generating frames: {e}")
            return

        if not frames:
            return

        # Publish frames at the target frame rate
        frame_interval = 1.0 / VIDEO_FPS

        for frame in frames:
            if not self._running:
                break

            # Convert RGB frame to VideoFrame
            height, width = frame.shape[:2]

            # Create RGBA frame (LiveKit expects RGBA)
            rgba_frame = np.zeros((height, width, 4), dtype=np.uint8)
            rgba_frame[:, :, :3] = frame
            rgba_frame[:, :, 3] = 255  # Full opacity

            video_frame = rtc.VideoFrame(
                width=width,
                height=height,
                type=rtc.VideoBufferType.RGBA,
                data=rgba_frame.tobytes(),
            )

            self.video_source.capture_frame(video_frame)

            # Wait for next frame time
            await asyncio.sleep(frame_interval)

    async def run(self) -> None:
        """
        Run the main service loop.

        Keeps the service running until shutdown is requested.
        """
        self._running = True
        logger.info("Avatar service running")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        logger.info("Shutdown signal received")
        await self.shutdown()

    async def shutdown(self) -> None:
        """Clean up resources and disconnect from the room."""
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Disconnect from room
        if self.room is not None:
            await self.room.disconnect()
            self.room = None

        logger.info("Avatar service shut down")

    def request_shutdown(self) -> None:
        """Request a graceful shutdown of the service."""
        self._shutdown_event.set()


async def main(
    persona_name: str,
    room_name: str | None = None,
    identity: str | None = None,
    livekit_url: str | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> None:
    """
    Main entry point for the LiveKit Avatar Service.

    Args:
        persona_name: Name of the persona to use
        room_name: LiveKit room name (defaults to LIVEKIT_ROOM env var)
        identity: Bot identity (defaults to LIVEKIT_IDENTITY env var)
        livekit_url: LiveKit server URL (defaults to LIVEKIT_URL env var)
        api_key: LiveKit API key (defaults to LIVEKIT_API_KEY env var)
        api_secret: LiveKit API secret (defaults to LIVEKIT_API_SECRET env var)
    """
    # Load configuration from environment with overrides
    livekit_url = livekit_url or os.environ.get("LIVEKIT_URL")
    room_name = room_name or os.environ.get("LIVEKIT_ROOM")
    identity = identity or os.environ.get("LIVEKIT_IDENTITY", "avatar-bot")
    api_key = api_key or os.environ.get("LIVEKIT_API_KEY")
    api_secret = api_secret or os.environ.get("LIVEKIT_API_SECRET")

    # Validate required configuration
    if not livekit_url:
        raise ValueError("LIVEKIT_URL environment variable or --url required")
    if not room_name:
        raise ValueError("LIVEKIT_ROOM environment variable or --room required")
    if not api_key:
        raise ValueError("LIVEKIT_API_KEY environment variable required")
    if not api_secret:
        raise ValueError("LIVEKIT_API_SECRET environment variable required")

    # Load persona configuration
    try:
        persona_config = load_persona(persona_name)
    except PersonaNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Using persona: {persona_config.display_name}")

    # Create and initialize service
    service = LiveKitAvatarService(persona_config)

    # Set up signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler() -> None:
        logger.info("Received shutdown signal")
        service.request_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Initialize avatar
        await service.initialize()

        # Connect to LiveKit room
        await service.connect(
            livekit_url=livekit_url,
            room_name=room_name,
            identity=identity,
            api_key=api_key,
            api_secret=api_secret,
        )

        # Run main loop
        await service.run()

    except Exception as e:
        logger.error(f"Service error: {e}")
        raise
    finally:
        await service.shutdown()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LiveKit Avatar Service - MuseTalk talking head video bot"
    )
    parser.add_argument(
        "--persona",
        default=os.environ.get("PERSONA_NAME", "default"),
        help="Persona name from personas.yaml (default: $PERSONA_NAME or 'default')",
    )
    parser.add_argument(
        "--room",
        default=None,
        help="LiveKit room name (default: $LIVEKIT_ROOM)",
    )
    parser.add_argument(
        "--identity",
        default=None,
        help="Bot identity in the room (default: $LIVEKIT_IDENTITY or 'avatar-bot')",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="LiveKit server URL (default: $LIVEKIT_URL)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the service
    asyncio.run(
        main(
            persona_name=args.persona,
            room_name=args.room,
            identity=args.identity,
            livekit_url=args.url,
        )
    )
