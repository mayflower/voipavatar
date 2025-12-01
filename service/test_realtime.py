#!/usr/bin/env python3
"""
Real-time Avatar Test - Tests chunked audio processing for LiveKit integration.

This test simulates real-time audio streaming by:
1. Loading the avatar once (expensive operation)
2. Processing audio in small chunks (simulating streaming)
3. Measuring frame generation latency and throughput
"""

import argparse
import logging
import sys
import time

import numpy as np
import soundfile as sf


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test real-time avatar processing")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to avatar reference video",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to test audio file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/inference/realtime.yaml",
        help="Path to MuseTalk config",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.5,
        help="Audio chunk duration in seconds (default: 0.5 for ~0.9x realtime)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="Target frames per second (default: 25.0)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile() for faster inference (slower first run)",
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Use TensorRT engines for fastest inference",
    )
    parser.add_argument(
        "--tensorrt-dir",
        type=str,
        default="./models/tensorrt",
        help="Path to TensorRT engine files (default: ./models/tensorrt)",
    )
    parser.add_argument(
        "--gpu-blending",
        action="store_true",
        help="Use GPU-accelerated frame blending (~3x faster)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Real-time Avatar Test")
    logger.info("=" * 60)
    logger.info(f"  Chunk duration: {args.chunk_duration}s")
    logger.info(f"  Target FPS: {args.target_fps}")
    logger.info(f"  Compile models: {args.compile}")
    logger.info(f"  TensorRT: {args.tensorrt}")
    logger.info(f"  GPU blending: {args.gpu_blending}")

    # Import here to avoid slow startup for --help
    from service.musetalk_adapter import MuseTalkAvatar

    # Step 1: Initialize avatar (one-time cost)
    logger.info("\nStep 1: Initializing MuseTalkAvatar...")
    init_start = time.time()
    avatar = MuseTalkAvatar(
        inference_config_path=args.config,
        models_dir="./models",
        device="cuda",
        compile_models=args.compile,
        use_tensorrt=args.tensorrt,
        tensorrt_dir=args.tensorrt_dir,
        use_gpu_blending=args.gpu_blending,
    )
    init_time = time.time() - init_start
    logger.info(f"  Model initialization: {init_time:.2f}s")

    # Step 2: Prepare avatar (one-time cost per avatar)
    logger.info("\nStep 2: Preparing avatar...")
    prep_start = time.time()
    avatar.prepare_avatar(args.video)
    prep_time = time.time() - prep_start
    logger.info(f"  Avatar preparation: {prep_time:.2f}s")
    logger.info(f"  Frame size: {avatar.frame_size}, FPS: {avatar.fps}")

    # Step 3: Load and chunk audio
    logger.info("\nStep 3: Loading audio...")
    audio_data, sample_rate = sf.read(args.audio, dtype="int16")
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Take first channel if stereo

    total_duration = len(audio_data) / sample_rate
    logger.info(f"  Audio: {total_duration:.2f}s @ {sample_rate}Hz")

    # Calculate chunk size
    chunk_samples = int(args.chunk_duration * sample_rate)
    num_chunks = int(np.ceil(len(audio_data) / chunk_samples))
    logger.info(f"  Chunk size: {chunk_samples} samples ({args.chunk_duration}s)")
    logger.info(f"  Number of chunks: {num_chunks}")

    # Step 4: Process chunks and measure performance
    logger.info("\nStep 4: Processing audio chunks...")
    logger.info("-" * 60)

    chunk_times = []
    frame_counts = []
    total_frames = 0

    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = min((i + 1) * chunk_samples, len(audio_data))
        chunk = audio_data[start_idx:end_idx]

        if len(chunk) < sample_rate * 0.1:  # Skip very short chunks
            logger.info(f"  Chunk {i + 1}: Skipping (too short)")
            continue

        chunk_start = time.time()
        frames = avatar.generate_from_audio_chunk(chunk, sample_rate)
        chunk_time = time.time() - chunk_start

        chunk_times.append(chunk_time)
        frame_counts.append(len(frames))
        total_frames += len(frames)

        # Calculate metrics
        chunk_audio_duration = len(chunk) / sample_rate
        realtime_ratio = chunk_audio_duration / chunk_time if chunk_time > 0 else 0
        fps = len(frames) / chunk_time if chunk_time > 0 else 0

        status = "✓" if realtime_ratio >= 1.0 else "✗"
        logger.info(
            f"  Chunk {i + 1}/{num_chunks}: "
            f"{len(frames)} frames in {chunk_time:.2f}s "
            f"({fps:.1f} fps, {realtime_ratio:.2f}x realtime) {status}"
        )

    # Step 5: Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Performance Summary")
    logger.info("=" * 60)

    if chunk_times:
        avg_chunk_time = np.mean(chunk_times)
        avg_frames = np.mean(frame_counts)
        avg_fps = avg_frames / avg_chunk_time if avg_chunk_time > 0 else 0
        avg_realtime_ratio = args.chunk_duration / avg_chunk_time

        total_process_time = sum(chunk_times)
        overall_fps = total_frames / total_process_time if total_process_time > 0 else 0
        overall_realtime_ratio = total_duration / total_process_time

        logger.info(f"  Total frames generated: {total_frames}")
        logger.info(f"  Total processing time: {total_process_time:.2f}s")
        logger.info(f"  Audio duration: {total_duration:.2f}s")
        logger.info(f"  Overall FPS: {overall_fps:.1f}")
        logger.info(f"  Overall realtime ratio: {overall_realtime_ratio:.2f}x")
        logger.info("")
        logger.info(f"  Average chunk time: {avg_chunk_time:.2f}s")
        logger.info(f"  Average frames/chunk: {avg_frames:.1f}")
        logger.info(f"  Average FPS: {avg_fps:.1f}")
        logger.info(f"  Average realtime ratio: {avg_realtime_ratio:.2f}x")

        # Determine if real-time is feasible
        logger.info("")
        if overall_realtime_ratio >= 1.0:
            logger.info("✓ REAL-TIME CAPABLE: Processing is faster than real-time!")
            if overall_realtime_ratio >= 1.5:
                logger.info("  Excellent: Good headroom for network/encoding overhead")
            elif overall_realtime_ratio >= 1.2:
                logger.info("  Good: Some headroom available")
            else:
                logger.info("  Marginal: May struggle with additional overhead")
        else:
            logger.info("✗ NOT REAL-TIME: Processing is slower than real-time")
            logger.info(
                f"  Need {1 / overall_realtime_ratio:.1f}x speedup for real-time"
            )
            logger.info(
                "  Consider: batch processing, lower resolution, or GPU upgrade"
            )

        # LiveKit-specific recommendations
        logger.info("")
        logger.info("LiveKit Integration Notes:")
        logger.info(f"  - Target video FPS: {args.target_fps}")
        logger.info(f"  - Achieved FPS: {overall_fps:.1f}")
        if overall_fps >= args.target_fps:
            logger.info(f"  ✓ Can sustain {args.target_fps} fps video output")
        else:
            logger.info(f"  ✗ Cannot sustain {args.target_fps} fps (will drop frames)")

    else:
        logger.error("No chunks were processed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
