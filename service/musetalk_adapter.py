"""
MuseTalk Adapter - Wrapper around MuseTalk's real-time inference pipeline.

This module provides a clean Python class (MuseTalkAvatar) that wraps
MuseTalk's functionality for:
- Loading and initializing MuseTalk models (UNet, VAE, Whisper, etc.)
- Preparing an avatar from a reference video (face detection, landmarks, etc.)
- Generating talking-head frames from audio chunks

The class is designed to be used by the LiveKit service, which feeds it
audio PCM data and receives RGB frames in return.
"""

import logging
import os
import tempfile

import cv2
import numpy as np
import soundfile as sf
import torch
import yaml
from musetalk.utils.blending import get_image_blending, get_image_prepare_material
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs

# MuseTalk imports (available when /opt/MuseTalk is on PYTHONPATH)
from musetalk.utils.utils import get_video_fps, load_all_model


logger = logging.getLogger(__name__)


class AvatarNotPreparedError(Exception):
    """Raised when generate_from_audio_chunk is called before prepare_avatar."""


class FaceNotFoundError(Exception):
    """Raised when no face is detected in the avatar video."""


class MuseTalkAvatar:
    """
    Wrapper around MuseTalk's real-time inference pipeline.

    This class encapsulates all MuseTalk functionality needed to:
    1. Load the required models (UNet, VAE, Whisper audio processor)
    2. Prepare an avatar from a reference video
    3. Generate talking-head frames from audio chunks

    Attributes:
        device: PyTorch device (cuda/cpu)
        dtype: PyTorch dtype for inference
        audio_processor: Whisper-based audio feature extractor
        vae: VAE model for encoding/decoding
        unet: UNet model for generation
        pe: Positional encoding model
    """

    def __init__(
        self,
        inference_config_path: str,
        models_dir: str = "./models",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Initialize MuseTalkAvatar with models and configuration.

        Args:
            inference_config_path: Path to MuseTalk inference config YAML
                (e.g., configs/inference/realtime.yaml)
            models_dir: Directory containing MuseTalk model weights
            device: PyTorch device string ("cuda" or "cpu")
            dtype: PyTorch dtype for inference (torch.float16 recommended for GPU)

        Raises:
            FileNotFoundError: If config file or models directory doesn't exist
        """
        self.device = device
        self.dtype = dtype
        self.models_dir = models_dir
        self._prepared = False

        # Load inference config
        if not os.path.exists(inference_config_path):
            raise FileNotFoundError(f"Config not found: {inference_config_path}")

        with open(inference_config_path) as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loading MuseTalk models from {models_dir} on {device}")

        # Load all models using MuseTalk's utility
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()

        # Move models to device
        self.vae = self.vae.to(device, dtype=dtype)
        self.unet = self.unet.to(device, dtype=dtype)
        self.pe = self.pe.to(device, dtype=dtype)

        # Avatar state (populated by prepare_avatar)
        self.coord_list: list = []
        self.frame_list: list = []
        self.frame_list_cycle: list = []
        self.input_latent_list_cycle: list = []
        self.mask_coords_list_cycle: list = []
        self.mask_list_cycle: list = []
        self.fps: float = 25.0
        self.avatar_width: int = 0
        self.avatar_height: int = 0

        logger.info("MuseTalk models loaded successfully")

    def prepare_avatar(self, video_path: str, bbox_shift: int = 0) -> None:
        """
        Prepare an avatar from a reference video.

        This performs the heavy preprocessing work:
        - Extract frames from the video
        - Detect faces and landmarks
        - Compute bounding boxes with optional shift
        - Build coordinate and frame lists for inference
        - Pre-compute VAE latents for the reference frames

        Args:
            video_path: Path to the avatar reference video file
            bbox_shift: Vertical shift for face bounding box (pixels).
                Positive values shift down, negative shift up.

        Raises:
            FileNotFoundError: If video file doesn't exist
            FaceNotFoundError: If no face is detected in the video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Avatar video not found: {video_path}")

        logger.info(f"Preparing avatar from {video_path} (bbox_shift={bbox_shift})")

        # Get video properties
        self.fps = get_video_fps(video_path)

        # Create temp directory for extracted frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract frames from video
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            # Use ffmpeg to extract frames (using subprocess for security)
            import subprocess

            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-qscale:v",
                    "2",
                    f"{frames_dir}/%06d.png",
                    "-loglevel",
                    "error",
                ],
                check=True,
            )

            # Read extracted frames
            frame_files = sorted(
                [
                    os.path.join(frames_dir, f)
                    for f in os.listdir(frames_dir)
                    if f.endswith(".png")
                ]
            )

            if not frame_files:
                raise FaceNotFoundError(f"No frames extracted from {video_path}")

            # Read frames as images
            input_img_list = read_imgs(frame_files)

            if not input_img_list:
                raise FaceNotFoundError(f"Failed to read frames from {video_path}")

            # Get image dimensions
            self.avatar_height, self.avatar_width = input_img_list[0].shape[:2]

            # Detect faces and get landmarks/bboxes
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)

            if not coord_list or all(c is None for c in coord_list):
                raise FaceNotFoundError(
                    f"No face detected in avatar video: {video_path}"
                )

            # Handle frames where face wasn't detected (use placeholder)
            self.coord_list = []
            self.frame_list = []

            for i, (coord, frame) in enumerate(zip(coord_list, frame_list)):
                if coord is None:
                    # Use placeholder for missing detections
                    if self.coord_list:
                        coord = self.coord_list[-1]
                        frame = self.frame_list[-1]
                    else:
                        continue
                self.coord_list.append(coord)
                self.frame_list.append(frame)

            if not self.coord_list:
                raise FaceNotFoundError(
                    f"No valid face coordinates found in {video_path}"
                )

            logger.info(f"Detected faces in {len(self.coord_list)} frames")

            # Prepare cyclic lists for inference
            self.frame_list_cycle = self.frame_list + self.frame_list[::-1]
            self.coord_list_cycle = self.coord_list + self.coord_list[::-1]

            # Pre-compute latents and masks for all frames
            self.input_latent_list_cycle = []
            self.mask_coords_list_cycle = []
            self.mask_list_cycle = []

            for i, (frame, coord) in enumerate(
                zip(self.frame_list_cycle, self.coord_list_cycle)
            ):
                # Get blending materials
                x1, y1, x2, y2 = coord
                crop_frame = frame[y1:y2, x1:x2]

                # Resize crop to expected size (256x256 for MuseTalk)
                crop_frame_resized = cv2.resize(
                    crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
                )

                # Prepare mask and latent
                mask_crop, mask_coords = get_image_prepare_material(crop_frame_resized)

                # Encode to latent space
                crop_tensor = (
                    torch.from_numpy(crop_frame_resized.transpose(2, 0, 1))
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                crop_tensor = crop_tensor.to(self.device, dtype=self.dtype)

                with torch.no_grad():
                    latent = (
                        self.vae.encode(crop_tensor * 2 - 1).latent_dist.sample()
                        * 0.18215
                    )

                self.input_latent_list_cycle.append(latent)
                self.mask_coords_list_cycle.append(mask_coords)
                self.mask_list_cycle.append(mask_crop)

        self._prepared = True
        logger.info(f"Avatar prepared: {len(self.frame_list_cycle)} frames ready")

    def generate_from_audio_chunk(
        self, audio_pcm: np.ndarray, audio_sample_rate: int
    ) -> list[np.ndarray]:
        """
        Generate talking-head frames from an audio chunk.

        Takes mono PCM audio and produces RGB video frames of the avatar
        speaking the audio.

        Args:
            audio_pcm: Mono audio samples as int16 numpy array
            audio_sample_rate: Sample rate of the audio (e.g., 16000)

        Returns:
            List of RGB frames as uint8 numpy arrays with shape (H, W, 3)

        Raises:
            AvatarNotPreparedError: If prepare_avatar() hasn't been called
            ValueError: If audio_pcm is empty or invalid
        """
        if not self._prepared:
            raise AvatarNotPreparedError(
                "Must call prepare_avatar() before generating frames"
            )

        if audio_pcm is None or len(audio_pcm) == 0:
            raise ValueError("audio_pcm cannot be empty")

        # Ensure audio is float32 normalized
        if audio_pcm.dtype == np.int16:
            audio_float = audio_pcm.astype(np.float32) / 32768.0
        elif audio_pcm.dtype == np.float32:
            audio_float = audio_pcm
        else:
            audio_float = audio_pcm.astype(np.float32)

        # Write to temporary WAV file (required by MuseTalk's audio processor)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        try:
            sf.write(tmp_wav_path, audio_float, audio_sample_rate)

            # Extract audio features using Whisper
            whisper_feature = self.audio_processor.audio2feat(tmp_wav_path)

            # Get whisper chunks for each frame
            whisper_chunks = self.audio_processor.feature2chunks(
                feature_array=whisper_feature, fps=self.fps
            )

            if not whisper_chunks:
                logger.warning("No audio features extracted")
                return []

            # Generate frames
            output_frames = []

            for i, whisper_chunk in enumerate(whisper_chunks):
                # Cycle through avatar frames
                frame_idx = i % len(self.frame_list_cycle)

                # Get precomputed data
                ref_frame = self.frame_list_cycle[frame_idx]
                coord = self.coord_list_cycle[frame_idx]
                input_latent = self.input_latent_list_cycle[frame_idx]
                mask = self.mask_list_cycle[frame_idx]

                # Prepare audio features tensor
                audio_feat = torch.from_numpy(whisper_chunk).unsqueeze(0)
                audio_feat = audio_feat.to(self.device, dtype=self.dtype)

                # Run UNet to generate face
                with torch.no_grad():
                    # Get positional encoding
                    pe_cond = self.pe(audio_feat)

                    # Run UNet
                    pred_latent = self.unet(
                        input_latent,
                        timesteps=torch.tensor([0], device=self.device),
                        encoder_hidden_states=pe_cond,
                    ).sample

                    # Decode latent to image
                    pred_image = self.vae.decode(pred_latent / 0.18215).sample

                    # Convert to numpy
                    pred_image = (pred_image + 1) / 2
                    pred_image = pred_image.clamp(0, 1)
                    pred_image = pred_image[0].permute(1, 2, 0).cpu().numpy()
                    pred_image = (pred_image * 255).astype(np.uint8)

                # Blend generated face into reference frame
                x1, y1, x2, y2 = coord

                # Resize prediction to match crop size
                crop_h, crop_w = y2 - y1, x2 - x1
                pred_resized = cv2.resize(
                    pred_image, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4
                )

                # Blend into frame
                output_frame = ref_frame.copy()
                output_frame = get_image_blending(
                    output_frame, pred_resized, coord, mask
                )

                # Ensure RGB format
                if output_frame.shape[2] == 4:
                    output_frame = output_frame[:, :, :3]

                output_frames.append(output_frame)

            logger.debug(f"Generated {len(output_frames)} frames from audio chunk")
            return output_frames

        finally:
            # Clean up temp file
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)

    @property
    def is_prepared(self) -> bool:
        """Check if avatar has been prepared."""
        return self._prepared

    @property
    def frame_size(self) -> tuple[int, int]:
        """Get avatar frame dimensions (width, height)."""
        return (self.avatar_width, self.avatar_height)
