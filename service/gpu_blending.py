"""
GPU-accelerated frame blending for MuseTalk.

This module provides GPU-based alternatives to the CPU PIL blending,
keeping all operations on GPU until final output.

PERFORMANCE NOTES:
------------------
GPU blending provides ~3x speedup over CPU blending ONLY when reference
frames are pre-loaded on GPU. For large avatar videos (>300 frames), the
memory-efficient version that transfers frames per-batch is actually SLOWER
than CPU blending due to CPU->GPU transfer overhead.

Benchmarks (12 frames at 512x512):
- CPU (PIL): ~67ms
- GPU (pre-loaded): ~23ms (2.9x faster)
- GPU (memory-efficient): ~100ms+ (slower than CPU!)

RECOMMENDATION:
- For small avatars (<100 frames): GPU blending is beneficial
- For large avatars (>300 frames): Use CPU blending (default)
- The synthetic_avatar.mp4 (typically short) benefits from GPU blending
- Full-length avatar videos should use CPU blending
"""

import torch
import torch.nn.functional as F


class GPUFrameBlender:
    """
    GPU-accelerated frame blending for avatar face compositing.

    Memory-efficient version: keeps reference frames on CPU, transfers
    only needed frames to GPU per batch. Trades some speed for much
    lower GPU memory usage.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the GPU blender.

        Args:
            device: CUDA device string
        """
        self.device = device
        self.ref_frames_cpu = None  # List of numpy BGR frames
        self.ref_frames_tensor = None  # List of CPU tensors (3, H, W) float32
        self.masks_cpu = None  # List of CPU tensors (1, crop_H, crop_W) float32
        self.coords = None  # List of (x1, y1, x2, y2)
        self.crop_boxes = None  # List of (x_s, y_s, x_e, y_e)
        self.frame_height = 0
        self.frame_width = 0

    def prepare(
        self,
        ref_frames: list,
        masks: list,
        coords: list,
        crop_boxes: list,
    ) -> None:
        """
        Prepare reference frames and masks (kept on CPU for memory efficiency).

        Uses pinned memory for faster CPU->GPU transfers during blending.

        Args:
            ref_frames: List of numpy BGR frames (H, W, 3) uint8
            masks: List of numpy masks (crop_H, crop_W) uint8 - sized for crop_box
            coords: List of (x1, y1, x2, y2) face bboxes
            crop_boxes: List of (x_s, y_s, x_e, y_e) expanded crop boxes
        """
        self.coords = coords
        self.crop_boxes = crop_boxes
        self.ref_frames_cpu = ref_frames
        self.frame_height, self.frame_width = ref_frames[0].shape[:2]

        # Convert frames to pinned CPU tensors (faster GPU transfers)
        self.ref_frames_tensor = []
        for frame in ref_frames:
            # BGR to RGB, HWC to CHW, normalize to [0, 1]
            t = torch.from_numpy(frame[:, :, ::-1].copy()).float() / 255.0
            t = t.permute(2, 0, 1)  # HWC -> CHW
            t = t.pin_memory()  # Pin for faster async transfers
            self.ref_frames_tensor.append(t)

        # Convert masks to pinned CPU tensors
        self.masks_cpu = []
        for mask in masks:
            t = torch.from_numpy(mask.copy()).float() / 255.0
            t = t.unsqueeze(0)  # Add channel dim (1, H, W)
            t = t.pin_memory()
            self.masks_cpu.append(t)

    def blend_batch(
        self,
        face_images: torch.Tensor,
        frame_indices: list[int],
    ) -> torch.Tensor:
        """
        Blend generated face images into reference frames on GPU.

        Memory-efficient: transfers only needed frames from CPU to GPU per batch.

        Args:
            face_images: Generated faces (B, 3, 256, 256) float32 RGB on GPU
            frame_indices: List of indices into the reference frame cycle

        Returns:
            Blended frames (B, 3, H, W) float32 RGB on GPU
        """
        batch_size = face_images.shape[0]

        # Transfer only needed reference frames to GPU (non_blocking with pinned memory)
        ref_batch = torch.stack(
            [
                self.ref_frames_tensor[idx].to(self.device, non_blocking=True)
                for idx in frame_indices
            ]
        )  # (B, 3, H, W)

        # Process each frame (different crop sizes)
        for i in range(batch_size):
            idx = frame_indices[i]
            x1, y1, x2, y2 = self.coords[idx]
            x_s, y_s, x_e, y_e = self.crop_boxes[idx]

            crop_h, crop_w = y2 - y1, x2 - x1

            # Resize face to crop size using bilinear interpolation
            face_resized = F.interpolate(
                face_images[i : i + 1],  # (1, 3, 256, 256)
                size=(crop_h, crop_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # (3, crop_h, crop_w)

            # Get the crop region from reference
            crop_region = ref_batch[i, :, y_s:y_e, x_s:x_e]  # (3, crop_H, crop_W)

            # Get the mask for this crop (non_blocking transfer from pinned CPU)
            mask_region = self.masks_cpu[idx].to(
                self.device, non_blocking=True
            )  # (1, crop_H, crop_W)

            # Place face into the crop region at the correct offset
            face_offset_y = y1 - y_s
            face_offset_x = x1 - x_s

            # Create face_in_crop by pasting face into crop region
            face_in_crop = crop_region.clone()
            face_in_crop[
                :,
                face_offset_y : face_offset_y + crop_h,
                face_offset_x : face_offset_x + crop_w,
            ] = face_resized

            # Apply mask blending: original * (1-mask) + new * mask
            blended_crop = crop_region * (1 - mask_region) + face_in_crop * mask_region

            # Put blended crop back into the frame
            ref_batch[i, :, y_s:y_e, x_s:x_e] = blended_crop

        return ref_batch

    def blend_batch_to_numpy(
        self,
        face_images: torch.Tensor,
        frame_indices: list[int],
    ) -> list:
        """
        Blend and convert to numpy BGR frames.

        Args:
            face_images: Generated faces (B, 3, 256, 256) float32 RGB on GPU
            frame_indices: List of indices into the reference frame cycle

        Returns:
            List of numpy BGR frames (H, W, 3) uint8
        """
        blended = self.blend_batch(face_images, frame_indices)

        # Convert to numpy BGR
        # (B, 3, H, W) -> (B, H, W, 3) -> numpy
        blended = blended.permute(0, 2, 3, 1)  # BCHW -> BHWC
        blended = (blended * 255).clamp(0, 255).to(torch.uint8)
        blended = blended.cpu().numpy()

        # RGB to BGR
        frames = [frame[:, :, ::-1].copy() for frame in blended]
        return frames


def gpu_blend_single(
    ref_frame: torch.Tensor,
    face: torch.Tensor,
    mask: torch.Tensor,
    coord: tuple,
    crop_box: tuple,
) -> torch.Tensor:
    """
    Blend a single face into a reference frame on GPU.

    Args:
        ref_frame: Reference frame (3, H, W) float32 RGB
        face: Generated face (3, 256, 256) float32 RGB
        mask: Blending mask (1, crop_H, crop_W) float32
        coord: (x1, y1, x2, y2) face bbox
        crop_box: (x_s, y_s, x_e, y_e) expanded crop box

    Returns:
        Blended frame (3, H, W) float32 RGB
    """
    x1, y1, x2, y2 = coord
    x_s, y_s, x_e, y_e = crop_box

    crop_h, crop_w = y2 - y1, x2 - x1

    # Resize face to crop size
    face_resized = F.interpolate(
        face.unsqueeze(0),  # (1, 3, 256, 256)
        size=(crop_h, crop_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # (3, crop_h, crop_w)

    # Get crop region
    output = ref_frame.clone()
    crop_region = output[:, y_s:y_e, x_s:x_e]

    # Face offset in crop space
    face_offset_y = y1 - y_s
    face_offset_x = x1 - x_s

    # Place face in crop
    face_in_crop = crop_region.clone()
    face_in_crop[
        :,
        face_offset_y : face_offset_y + crop_h,
        face_offset_x : face_offset_x + crop_w,
    ] = face_resized

    # Blend with mask
    blended_crop = crop_region * (1 - mask) + face_in_crop * mask

    # Put back
    output[:, y_s:y_e, x_s:x_e] = blended_crop

    return output
