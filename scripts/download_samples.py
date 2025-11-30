#!/usr/bin/env python3
"""
Download sample avatar videos and audio files for testing.

This script downloads publicly available sample media for testing the
VoIP Avatar service without requiring users to provide their own assets.

Usage:
    python scripts/download_samples.py
    python scripts/download_samples.py --output-dir ./my_assets
    python scripts/download_samples.py --audio-only
"""

import argparse
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path


logger = logging.getLogger(__name__)

# Sample video sources (public domain / CC0 licensed)
# These are placeholder URLs - replace with actual working URLs
SAMPLE_VIDEOS = {
    "sample_avatar": {
        "url": "https://www.pexels.com/download/video/3252106/",
        "filename": "sample_avatar.mp4",
        "description": "Sample talking head video for avatar",
        # Alternative: direct mp4 link if pexels doesn't work
        "alt_url": None,
    },
}

# Sample audio sources (public domain)
SAMPLE_AUDIO = {
    "sample_speech": {
        # Open Speech Repository - Harvard sentences (public domain)
        "url": "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav",
        "filename": "sample_speech.wav",
        "description": "Harvard sentences (Open Speech Repository)",
    },
    "sample_short": {
        # Mozilla DeepSpeech test sample (CC0)
        "url": "https://raw.githubusercontent.com/mozilla/DeepSpeech/master/data/smoke_test/LDC93S1.wav",
        "filename": "sample_short.wav",
        "description": "Short speech sample (LDC93S1)",
    },
}


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """
    Download a file from URL to local path.

    Args:
        url: Source URL
        output_path: Local destination path
        description: Human-readable description for logging

    Returns:
        True if successful, False otherwise
    """
    if output_path.exists():
        logger.info(f"  Already exists: {output_path.name}")
        return True

    logger.info(f"  Downloading: {description or url}")

    try:
        # Try urllib first
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.read())
        logger.info(f"  Saved: {output_path}")
        return True
    except Exception as e:
        logger.warning(f"  urllib failed: {e}, trying curl...")

    try:
        # Fallback to curl
        subprocess.run(
            ["curl", "-L", "-o", str(output_path), url],
            check=True,
            capture_output=True,
        )
        logger.info(f"  Saved: {output_path}")
        return True
    except Exception as e:
        logger.error(f"  Download failed: {e}")
        return False


def create_synthetic_avatar(output_path: Path, duration: float = 3.0) -> bool:
    """
    Create a synthetic avatar video using ffmpeg (colored rectangle with motion).

    This is a fallback when real videos can't be downloaded.

    Args:
        output_path: Output video path
        duration: Video duration in seconds

    Returns:
        True if successful
    """
    logger.info("  Creating synthetic test avatar...")

    try:
        # Create a simple animated test pattern
        # This won't work with MuseTalk (needs real face) but allows testing the pipeline
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s=512x512:d={duration},format=rgb24",
            "-vf",
            "drawtext=text='TEST AVATAR':fontsize=48:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"  Created synthetic avatar: {output_path}")
        logger.warning(
            "  Note: Synthetic avatar won't work with MuseTalk face detection"
        )
        return True
    except Exception as e:
        logger.error(f"  Failed to create synthetic avatar: {e}")
        return False


def generate_test_audio(output_path: Path, duration: float = 3.0) -> bool:
    """
    Generate synthetic test audio using ffmpeg.

    Args:
        output_path: Output WAV path
        duration: Audio duration in seconds

    Returns:
        True if successful
    """
    logger.info("  Generating synthetic test audio...")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"  Created: {output_path}")
        return True
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return False


def update_personas_yaml(assets_dir: Path, service_dir: Path) -> None:
    """
    Update personas.yaml to use downloaded assets.

    Args:
        assets_dir: Directory containing downloaded assets
        service_dir: Directory containing personas.yaml
    """
    personas_path = service_dir / "personas.yaml"

    if not personas_path.exists():
        logger.warning(f"  personas.yaml not found at {personas_path}")
        return

    # Read current content
    with open(personas_path) as f:
        content = f.read()

    # Check if already using local paths
    if str(assets_dir) in content:
        logger.info("  personas.yaml already configured for local assets")
        return

    # Create updated content with local paths
    avatar_path = assets_dir / "sample_avatar.mp4"
    if avatar_path.exists():
        new_content = f"""# Persona configuration file
# Maps persona names to their avatar configuration
# Updated to use local sample assets

default:
  display_name: "Default Avatar"
  video_path: "{avatar_path}"
  inference_config: "/opt/MuseTalk/configs/inference/realtime.yaml"
  bbox_shift: 0

# Add your own personas below:
# my_avatar:
#   display_name: "My Custom Avatar"
#   video_path: "/path/to/your/video.mp4"
#   inference_config: "/opt/MuseTalk/configs/inference/realtime.yaml"
#   bbox_shift: 0
"""
        with open(personas_path, "w") as f:
            f.write(new_content)
        logger.info(f"  Updated {personas_path} with local asset paths")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download sample avatar videos and audio for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "assets",
        help="Output directory for downloaded files (default: ./assets)",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Download only audio samples (faster, no large video files)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic test files instead of downloading",
    )
    parser.add_argument(
        "--update-personas",
        action="store_true",
        help="Update personas.yaml to use downloaded assets",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    assets_dir = args.output_dir.resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("VoIP Avatar Sample Downloader")
    logger.info("=" * 50)
    logger.info(f"Output directory: {assets_dir}")

    success_count = 0
    total_count = 0

    # Download/generate audio samples
    logger.info("\nAudio samples:")
    for name, info in SAMPLE_AUDIO.items():
        total_count += 1
        output_path = assets_dir / info["filename"]
        if download_file(info["url"], output_path, info["description"]):
            success_count += 1

    # Generate synthetic audio as fallback
    synth_audio = assets_dir / "synthetic_test.wav"
    if not synth_audio.exists():
        total_count += 1
        if generate_test_audio(synth_audio, duration=3.0):
            success_count += 1

    # Download/generate video samples
    if not args.audio_only:
        logger.info("\nVideo samples:")

        if args.synthetic:
            # Generate synthetic test video
            total_count += 1
            synth_video = assets_dir / "synthetic_avatar.mp4"
            if create_synthetic_avatar(synth_video, duration=3.0):
                success_count += 1
        else:
            # Try to download real videos
            for name, info in SAMPLE_VIDEOS.items():
                total_count += 1
                output_path = assets_dir / info["filename"]

                # Try primary URL
                if download_file(info["url"], output_path, info["description"]):
                    success_count += 1
                elif info.get("alt_url"):
                    # Try alternate URL
                    if download_file(info["alt_url"], output_path, "alternate source"):
                        success_count += 1
                else:
                    logger.warning(f"  Could not download {name}")
                    logger.info("  Creating synthetic fallback...")
                    if create_synthetic_avatar(output_path):
                        success_count += 1

    # Update personas.yaml if requested
    if args.update_personas:
        logger.info("\nUpdating configuration:")
        service_dir = Path(__file__).parent.parent / "service"
        update_personas_yaml(assets_dir, service_dir)

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info(f"Downloaded/created: {success_count}/{total_count} files")
    logger.info(f"Assets directory: {assets_dir}")

    if success_count > 0:
        logger.info("\nNext steps:")
        logger.info("  1. Provide your own avatar video (2-10s talking head clip)")
        logger.info("  2. Update service/personas.yaml with your video path")
        logger.info(
            "  3. Run: python -m service.test_avatar --video YOUR_VIDEO.mp4 --synthetic"
        )
        logger.info("\nOr use downloaded audio for testing:")
        logger.info(
            f"  python -m service.test_avatar --video YOUR_VIDEO.mp4 --audio {assets_dir}/sample_speech.wav"
        )

    sys.exit(0 if success_count == total_count else 1)


if __name__ == "__main__":
    main()
