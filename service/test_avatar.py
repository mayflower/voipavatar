"""
Standalone Avatar Test Script - Test MuseTalk pipeline without LiveKit.

This script validates the entire ML pipeline by:
1. Loading a persona configuration
2. Initializing the MuseTalkAvatar with models
3. Processing a WAV audio file
4. Outputting video frames to an MP4 file

Usage:
    python -m service.test_avatar --persona default --audio sample.wav --output result.mp4
    python -m service.test_avatar --video avatar.mp4 --audio sample.wav --output result.mp4
"""

import argparse
import logging
import sys
import time

import cv2
import numpy as np
import soundfile as sf


logger = logging.getLogger(__name__)


def create_test_audio(duration_s: float = 3.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Create synthetic test audio (sine wave) for testing without a real audio file.

    Args:
        duration_s: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio samples as int16 numpy array
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    # Generate a simple tone with some variation
    freq = 440  # A4 note
    audio = np.sin(2 * np.pi * freq * t) * 0.5
    # Add some harmonics for richer sound
    audio += np.sin(2 * np.pi * freq * 2 * t) * 0.25
    audio += np.sin(2 * np.pi * freq * 3 * t) * 0.125
    # Convert to int16
    return (audio * 32767).astype(np.int16)


def write_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: float = 25.0,
) -> None:
    """
    Write frames to an MP4 video file.

    Args:
        frames: List of RGB frames as numpy arrays
        output_path: Output file path
        fps: Frames per second
    """
    if not frames:
        raise ValueError("No frames to write")

    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
    finally:
        writer.release()

    logger.info(f"Wrote {len(frames)} frames to {output_path}")


def test_avatar_pipeline(
    video_path: str,
    audio_path: str | None,
    output_path: str,
    inference_config: str,
    bbox_shift: int = 0,
    use_synthetic_audio: bool = False,
) -> dict:
    """
    Test the full avatar pipeline.

    Args:
        video_path: Path to avatar reference video
        audio_path: Path to input audio file (WAV)
        output_path: Path for output video
        inference_config: Path to MuseTalk inference config
        bbox_shift: Bounding box shift for face detection
        use_synthetic_audio: If True, generate synthetic test audio

    Returns:
        Dict with test results and timing info
    """
    # Import here to allow running --help without MuseTalk installed
    from .musetalk_adapter import MuseTalkAvatar

    results = {
        "success": False,
        "timings": {},
        "frame_count": 0,
        "errors": [],
    }

    # Step 1: Initialize avatar
    logger.info("Step 1: Initializing MuseTalkAvatar...")
    t0 = time.perf_counter()
    try:
        avatar = MuseTalkAvatar(inference_config_path=inference_config)
        results["timings"]["model_load"] = time.perf_counter() - t0
        logger.info(f"  Models loaded in {results['timings']['model_load']:.2f}s")
    except Exception as e:
        results["errors"].append(f"Model load failed: {e}")
        logger.error(f"  Failed: {e}")
        return results

    # Step 2: Prepare avatar from video
    logger.info(f"Step 2: Preparing avatar from {video_path}...")
    t0 = time.perf_counter()
    try:
        avatar.prepare_avatar(video_path=video_path, bbox_shift=bbox_shift)
        results["timings"]["avatar_prepare"] = time.perf_counter() - t0
        logger.info(f"  Avatar prepared in {results['timings']['avatar_prepare']:.2f}s")
        logger.info(f"  Frame size: {avatar.frame_size}, FPS: {avatar.fps}")
    except Exception as e:
        results["errors"].append(f"Avatar prepare failed: {e}")
        logger.error(f"  Failed: {e}")
        return results

    # Step 3: Load or generate audio
    logger.info("Step 3: Loading audio...")
    t0 = time.perf_counter()
    try:
        if use_synthetic_audio or audio_path is None:
            logger.info("  Using synthetic test audio (3 seconds)")
            audio_pcm = create_test_audio(duration_s=3.0)
            sample_rate = 16000
        else:
            logger.info(f"  Loading {audio_path}")
            audio_data, sample_rate = sf.read(audio_path, dtype="int16")
            if len(audio_data.shape) > 1:
                # Convert stereo to mono
                audio_pcm = audio_data.mean(axis=1).astype(np.int16)
            else:
                audio_pcm = audio_data

        results["timings"]["audio_load"] = time.perf_counter() - t0
        duration = len(audio_pcm) / sample_rate
        logger.info(f"  Audio: {duration:.2f}s @ {sample_rate}Hz")
    except Exception as e:
        results["errors"].append(f"Audio load failed: {e}")
        logger.error(f"  Failed: {e}")
        return results

    # Step 4: Generate frames
    logger.info("Step 4: Generating frames from audio...")
    t0 = time.perf_counter()
    try:
        frames = avatar.generate_from_audio_chunk(audio_pcm, sample_rate)
        results["timings"]["frame_generation"] = time.perf_counter() - t0
        results["frame_count"] = len(frames)
        logger.info(
            f"  Generated {len(frames)} frames in {results['timings']['frame_generation']:.2f}s"
        )
        if frames:
            fps_achieved = len(frames) / results["timings"]["frame_generation"]
            logger.info(f"  Processing speed: {fps_achieved:.1f} fps")
    except Exception as e:
        results["errors"].append(f"Frame generation failed: {e}")
        logger.error(f"  Failed: {e}")
        return results

    # Step 5: Write output video
    if frames:
        logger.info(f"Step 5: Writing output to {output_path}...")
        t0 = time.perf_counter()
        try:
            write_video(frames, output_path, fps=avatar.fps)
            results["timings"]["video_write"] = time.perf_counter() - t0
            logger.info(f"  Video written in {results['timings']['video_write']:.2f}s")
        except Exception as e:
            results["errors"].append(f"Video write failed: {e}")
            logger.error(f"  Failed: {e}")
            return results
    else:
        results["errors"].append("No frames generated")
        logger.warning("  No frames to write")
        return results

    results["success"] = True
    total_time = sum(results["timings"].values())
    logger.info(f"\nTotal time: {total_time:.2f}s")

    return results


def test_persona_loading() -> bool:
    """Test that persona loading works."""
    from .personas import list_persona_names, load_persona

    logger.info("Testing persona loading...")
    try:
        names = list_persona_names()
        logger.info(f"  Available personas: {names}")

        for name in names:
            persona = load_persona(name)
            logger.info(f"  Loaded '{name}': {persona.display_name}")

        return True
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return False


def test_health_server() -> bool:
    """Test that health server starts and responds."""
    import socket
    import urllib.request

    from .health_server import HealthServer

    logger.info("Testing health server...")

    # Find an available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    server = HealthServer(port=port)
    try:
        server.start()

        # Test /health endpoint
        url = f"http://localhost:{port}/health"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = response.read().decode()
            logger.info(f"  GET /health: {data}")

        # Test /personas endpoint
        url = f"http://localhost:{port}/personas"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = response.read().decode()
            logger.info(f"  GET /personas: {data[:100]}...")

        return True
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return False
    finally:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test MuseTalk avatar pipeline without LiveKit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with a persona and audio file
  python -m service.test_avatar --persona default --audio speech.wav -o output.mp4

  # Test with custom video and synthetic audio
  python -m service.test_avatar --video avatar.mp4 --synthetic -o output.mp4

  # Quick test of all components (no GPU required for persona/health tests)
  python -m service.test_avatar --test-components
        """,
    )

    parser.add_argument(
        "--persona",
        help="Persona name from personas.yaml",
    )
    parser.add_argument(
        "--video",
        help="Path to avatar reference video (overrides persona)",
    )
    parser.add_argument(
        "--config",
        default="/opt/MuseTalk/configs/inference/realtime.yaml",
        help="Path to MuseTalk inference config",
    )
    parser.add_argument(
        "--audio",
        "-a",
        help="Path to input audio file (WAV)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic test audio instead of a file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="test_output.mp4",
        help="Output video path (default: test_output.mp4)",
    )
    parser.add_argument(
        "--bbox-shift",
        type=int,
        default=0,
        help="Bounding box shift for face detection",
    )
    parser.add_argument(
        "--test-components",
        action="store_true",
        help="Test persona loading and health server only (no GPU needed)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Component-only test mode
    if args.test_components:
        logger.info("=" * 50)
        logger.info("Component Tests (no GPU required)")
        logger.info("=" * 50)

        results = []
        results.append(("Persona Loading", test_persona_loading()))
        results.append(("Health Server", test_health_server()))

        logger.info("\n" + "=" * 50)
        logger.info("Results:")
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {name}: {status}")

        sys.exit(0 if all(r[1] for r in results) else 1)

    # Full pipeline test
    if not args.persona and not args.video:
        parser.error("Either --persona or --video is required for pipeline test")

    if not args.audio and not args.synthetic:
        parser.error("Either --audio or --synthetic is required")

    # Resolve video path
    if args.video:
        video_path = args.video
        inference_config = args.config
        bbox_shift = args.bbox_shift
    else:
        from .personas import load_persona

        persona = load_persona(args.persona)
        video_path = persona.video_path
        inference_config = persona.inference_config
        bbox_shift = persona.bbox_shift
        logger.info(f"Using persona '{args.persona}': {persona.display_name}")

    logger.info("=" * 50)
    logger.info("Avatar Pipeline Test")
    logger.info("=" * 50)

    results = test_avatar_pipeline(
        video_path=video_path,
        audio_path=args.audio,
        output_path=args.output,
        inference_config=inference_config,
        bbox_shift=bbox_shift,
        use_synthetic_audio=args.synthetic,
    )

    logger.info("\n" + "=" * 50)
    if results["success"]:
        logger.info("✓ Test PASSED")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Frames: {results['frame_count']}")
    else:
        logger.error("✗ Test FAILED")
        for error in results["errors"]:
            logger.error(f"  - {error}")

    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
