#!/usr/bin/env python3
"""Test GPU blending performance vs CPU blending."""

import time

import numpy as np
import torch
from PIL import Image

from service.gpu_blending import GPUFrameBlender


def create_test_data(num_frames: int = 10, frame_size: tuple = (512, 512)):
    """Create synthetic test data for blending."""
    height, width = frame_size

    # Create reference frames (BGR uint8)
    ref_frames = [
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]

    # Create face bboxes and crop boxes
    coords = []
    crop_boxes = []
    masks = []

    for _ in range(num_frames):
        # Face bbox (x1, y1, x2, y2) - 256x256 face in center-ish
        cx, cy = (
            width // 2 + np.random.randint(-50, 50),
            height // 2 + np.random.randint(-50, 50),
        )
        x1, y1 = cx - 128, cy - 128
        x2, y2 = cx + 128, cy + 128
        coords.append((x1, y1, x2, y2))

        # Expanded crop box with 50px margin
        margin = 50
        x_s = max(0, x1 - margin)
        y_s = max(0, y1 - margin)
        x_e = min(width, x2 + margin)
        y_e = min(height, y2 + margin)
        crop_boxes.append((x_s, y_s, x_e, y_e))

        # Create mask for crop box size
        crop_h, crop_w = y_e - y_s, x_e - x_s
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

        # Create elliptical mask in the face region within the crop
        face_y_offset = y1 - y_s
        face_x_offset = x1 - x_s
        y_coords, x_coords = np.ogrid[0:256, 0:256]
        center_y, center_x = 128, 128
        # Elliptical distance
        dist = ((x_coords - center_x) / 100) ** 2 + ((y_coords - center_y) / 120) ** 2
        face_mask = (dist <= 1).astype(np.uint8) * 255

        # Place face mask in crop mask
        mask[
            face_y_offset : face_y_offset + 256, face_x_offset : face_x_offset + 256
        ] = face_mask
        masks.append(mask)

    return ref_frames, masks, coords, crop_boxes


def cpu_blend_single(ref_frame, face_image, mask, coord, crop_box):
    """CPU blending using PIL (similar to MuseTalk's method)."""
    x1, y1, x2, y2 = coord
    x_s, y_s, x_e, y_e = crop_box

    crop_h, crop_w = y2 - y1, x2 - x1

    # Resize face to crop size
    face_pil = Image.fromarray(face_image[:, :, ::-1])  # BGR to RGB
    face_resized = face_pil.resize((crop_w, crop_h), Image.BILINEAR)

    # Get crop region
    ref_pil = Image.fromarray(ref_frame[:, :, ::-1])  # BGR to RGB
    crop_region = ref_pil.crop((x_s, y_s, x_e, y_e))

    # Create face in crop
    face_in_crop = crop_region.copy()
    face_offset_x = x1 - x_s
    face_offset_y = y1 - y_s
    face_in_crop.paste(face_resized, (face_offset_x, face_offset_y))

    # Apply mask
    mask_pil = Image.fromarray(mask)
    blended_crop = Image.composite(face_in_crop, crop_region, mask_pil)

    # Paste back
    result = ref_pil.copy()
    result.paste(blended_crop, (x_s, y_s))

    return np.array(result)[:, :, ::-1].copy()  # RGB to BGR


def benchmark_cpu_blending(
    ref_frames, masks, coords, crop_boxes, face_images, num_runs=5
):
    """Benchmark CPU blending."""
    num_frames = len(face_images)

    times = []
    for run in range(num_runs):
        start = time.perf_counter()
        results = []
        for i in range(num_frames):
            idx = i % len(ref_frames)
            result = cpu_blend_single(
                ref_frames[idx],
                face_images[i],
                masks[idx],
                coords[idx],
                crop_boxes[idx],
            )
            results.append(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times


def benchmark_gpu_blending(
    ref_frames, masks, coords, crop_boxes, face_images_tensor, num_runs=5
):
    """Benchmark GPU blending."""
    num_frames = face_images_tensor.shape[0]
    frame_indices = [i % len(ref_frames) for i in range(num_frames)]

    # Initialize blender
    blender = GPUFrameBlender(device="cuda")
    blender.prepare(ref_frames, masks, coords, crop_boxes)

    # Warmup
    _ = blender.blend_batch_to_numpy(face_images_tensor, frame_indices)
    torch.cuda.synchronize()

    times = []
    for run in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        results = blender.blend_batch_to_numpy(face_images_tensor, frame_indices)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times, results


def main():
    print("=" * 60)
    print("GPU vs CPU Blending Benchmark")
    print("=" * 60)

    # Test parameters
    num_ref_frames = 10
    num_blend_frames = 12  # Typical for 500ms chunk at 25fps
    frame_size = (512, 512)
    num_runs = 5

    print("\nParameters:")
    print(f"  Reference frames: {num_ref_frames}")
    print(f"  Frames to blend: {num_blend_frames}")
    print(f"  Frame size: {frame_size}")
    print(f"  Benchmark runs: {num_runs}")

    # Create test data
    print("\nCreating test data...")
    ref_frames, masks, coords, crop_boxes = create_test_data(num_ref_frames, frame_size)

    # Create fake generated face images (256x256 BGR)
    face_images_np = [
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        for _ in range(num_blend_frames)
    ]

    # Convert faces to GPU tensor (B, 3, 256, 256) float32 RGB
    face_images_tensor = torch.stack(
        [
            torch.from_numpy(f[:, :, ::-1].copy()).float().permute(2, 0, 1) / 255.0
            for f in face_images_np
        ]
    ).cuda()

    # Benchmark CPU
    print("\nBenchmarking CPU blending...")
    cpu_times = benchmark_cpu_blending(
        ref_frames, masks, coords, crop_boxes, face_images_np, num_runs
    )
    cpu_avg = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    cpu_per_frame = cpu_avg / num_blend_frames * 1000

    print(f"  Total time: {cpu_avg * 1000:.1f} ± {cpu_std * 1000:.1f} ms")
    print(f"  Per frame: {cpu_per_frame:.2f} ms")

    # Benchmark GPU
    print("\nBenchmarking GPU blending...")
    gpu_times, gpu_results = benchmark_gpu_blending(
        ref_frames, masks, coords, crop_boxes, face_images_tensor, num_runs
    )
    gpu_avg = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    gpu_per_frame = gpu_avg / num_blend_frames * 1000

    print(f"  Total time: {gpu_avg * 1000:.1f} ± {gpu_std * 1000:.1f} ms")
    print(f"  Per frame: {gpu_per_frame:.2f} ms")

    # Comparison
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    speedup = cpu_avg / gpu_avg
    print(f"  CPU: {cpu_avg * 1000:.1f} ms ({cpu_per_frame:.2f} ms/frame)")
    print(f"  GPU: {gpu_avg * 1000:.1f} ms ({gpu_per_frame:.2f} ms/frame)")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Time saved per chunk: {(cpu_avg - gpu_avg) * 1000:.1f} ms")

    # Verify output shapes
    print("\n  Output verification:")
    print(f"    Number of frames: {len(gpu_results)}")
    print(f"    Frame shape: {gpu_results[0].shape}")
    print(f"    Frame dtype: {gpu_results[0].dtype}")


if __name__ == "__main__":
    main()
