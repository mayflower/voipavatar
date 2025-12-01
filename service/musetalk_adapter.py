"""
MuseTalk Adapter - Wrapper around MuseTalk's real-time inference pipeline.

This module provides a clean Python class (MuseTalkAvatar) that wraps
MuseTalk's functionality for:
- Loading and initializing MuseTalk models (UNet, VAE, Whisper, etc.)
- Preparing an avatar from a reference video (face detection, landmarks, etc.)
- Generating talking-head frames from audio chunks

The class is designed to be used by the LiveKit service, which feeds it
audio PCM data and receives RGB frames in return.

Optimizations implemented:
- Disk caching of VAE latents, masks, and coordinates for fast avatar reload
- Silence detection with idle animation (skips GPU inference for silent audio)
- Model warm-up to avoid first-inference latency
- Producer/consumer threading for decoupled audio processing
"""

import hashlib
import json
import logging
import os
import pickle
import queue
import subprocess
import tempfile
import threading

import cv2
import numpy as np
import torch
import yaml

# MuseTalk imports (available when /opt/MuseTalk is on PYTHONPATH)
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image_blending, get_image_prepare_material
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.utils import get_video_fps, load_all_model
from transformers import WhisperModel


logger = logging.getLogger(__name__)

# Cache version - increment when cache format changes
CACHE_VERSION = "1.0"


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
        fp: FaceParsing model for face segmentation
    """

    def __init__(
        self,
        inference_config_path: str,
        models_dir: str = "./models",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        cache_dir: str = "./results/avatars",
        silence_threshold: float = 0.01,
        compile_models: bool = False,
        use_tensorrt: bool = False,
        tensorrt_dir: str = "./models/tensorrt",
    ) -> None:
        """
        Initialize MuseTalkAvatar with models and configuration.

        Args:
            inference_config_path: Path to MuseTalk inference config YAML
                (e.g., configs/inference/realtime.yaml)
            models_dir: Directory containing MuseTalk model weights
            device: PyTorch device string ("cuda" or "cpu")
            dtype: PyTorch dtype for inference (torch.float16 recommended for GPU)
            cache_dir: Directory for caching pre-computed avatar data
            silence_threshold: RMS threshold for silence detection (0.01 default)
            compile_models: If True, use torch.compile() for 2-4x speedup (slower first run)
            use_tensorrt: If True, use TensorRT engines for UNet/VAE (fastest)
            tensorrt_dir: Directory containing TensorRT engine files

        Raises:
            FileNotFoundError: If config file or models directory doesn't exist
        """
        self.compile_models = compile_models
        self.use_tensorrt = use_tensorrt
        self.tensorrt_dir = tensorrt_dir
        self.trt_model = None
        self.device = device
        self.dtype = dtype
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        self.silence_threshold = silence_threshold
        self._prepared = False

        # Load inference config
        if not os.path.exists(inference_config_path):
            raise FileNotFoundError(f"Config not found: {inference_config_path}")

        with open(inference_config_path) as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loading MuseTalk models from {models_dir} on {device}")

        # Load VAE, UNet, and positional encoding using MuseTalk's utility
        self.vae, self.unet, self.pe = load_all_model()

        # Load audio processor and whisper model separately
        whisper_dir = os.path.join(models_dir, "whisper")
        self.audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
        self.whisper = WhisperModel.from_pretrained(whisper_dir)
        self.whisper = self.whisper.to(device=device, dtype=dtype).eval()
        self.whisper.requires_grad_(False)

        # Initialize face parsing for blending masks
        self.fp = FaceParsing()

        # Move models to device and dtype
        # VAE is a wrapper class - access internal vae model
        self.vae.vae = self.vae.vae.to(device).to(dtype)
        # UNet is also a wrapper - access internal model
        self.unet.model = self.unet.model.to(device).to(dtype)
        # pe is a standard nn.Module
        self.pe = self.pe.to(device).to(dtype)

        # Apply torch.compile() for inference speedup
        if compile_models and not use_tensorrt:
            logger.info(
                "Compiling models with torch.compile() (first inference will be slow)..."
            )
            # Use default mode which caches compiled graphs to disk
            self.unet.model = torch.compile(self.unet.model, mode="default")
            self.vae.vae = torch.compile(self.vae.vae, mode="default")
            logger.info("Models compiled successfully")

        # Load TensorRT engines if available
        if use_tensorrt:
            unet_engine = os.path.join(tensorrt_dir, "unet.engine")
            vae_engine = os.path.join(tensorrt_dir, "vae_decoder.engine")
            if os.path.exists(unet_engine) and os.path.exists(vae_engine):
                from service.tensorrt_runtime import TensorRTMuseTalk

                logger.info("Loading TensorRT engines...")
                self.trt_model = TensorRTMuseTalk(unet_engine, vae_engine, device)
                logger.info("TensorRT engines loaded successfully")
            else:
                logger.warning(
                    f"TensorRT engines not found in {tensorrt_dir}, falling back to PyTorch"
                )
                self.use_tensorrt = False

        # Avatar state (populated by prepare_avatar)
        self.coord_list: list = []
        self.frame_list: list = []
        self.frame_list_cycle: list = []
        self.coord_list_cycle: list = []
        self.input_latent_list_cycle: list = []
        self.mask_coords_list_cycle: list = []
        self.mask_list_cycle: list = []
        self.fps: float = 25.0
        self.avatar_width: int = 0
        self.avatar_height: int = 0

        # Idle animation state
        self._idle_frame_index: int = 0
        self._frame_index: int = 0  # Current frame index for inference

        # Threading state
        self._request_queue: queue.Queue = queue.Queue(maxsize=2)
        self._result_queue: queue.Queue = queue.Queue(maxsize=2)
        self._worker_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        logger.info("MuseTalk models loaded successfully")

    def _get_avatar_id(self, video_path: str, bbox_shift: int) -> str:
        """Generate a unique avatar ID based on video path and parameters."""
        # Use video filename + bbox_shift for human-readable ID
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # Add hash of full path for uniqueness
        path_hash = hashlib.md5(video_path.encode()).hexdigest()[:8]
        return f"{video_name}_shift{bbox_shift}_{path_hash}"

    def _get_cache_dir(self, video_path: str, bbox_shift: int) -> str:
        """Get the cache directory for an avatar."""
        avatar_id = self._get_avatar_id(video_path, bbox_shift)
        return os.path.join(self.cache_dir, avatar_id)

    def _cache_valid(self, cache_dir: str, video_path: str) -> bool:
        """Check if cache exists and is valid."""
        metadata_path = os.path.join(cache_dir, "metadata.json")
        latents_path = os.path.join(cache_dir, "latents.pt")
        data_path = os.path.join(cache_dir, "avatar_data.pkl")

        if not all(os.path.exists(p) for p in [metadata_path, latents_path, data_path]):
            return False

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Check version
            if metadata.get("cache_version") != CACHE_VERSION:
                logger.info("Cache version mismatch, will regenerate")
                return False

            # Check if source video was modified
            video_mtime = os.path.getmtime(video_path)
            if metadata.get("video_mtime") != video_mtime:
                logger.info("Source video modified, will regenerate cache")
                return False

            return True
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Cache validation failed: {e}")
            return False

    def _save_to_cache(self, cache_dir: str, video_path: str, bbox_shift: int) -> None:
        """Save pre-computed avatar data to disk cache."""
        os.makedirs(cache_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "cache_version": CACHE_VERSION,
            "video_path": video_path,
            "video_mtime": os.path.getmtime(video_path),
            "bbox_shift": bbox_shift,
            "fps": self.fps,
            "avatar_width": self.avatar_width,
            "avatar_height": self.avatar_height,
            "num_frames": len(self.frame_list_cycle),
        }
        with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save latents (torch tensors)
        torch.save(self.input_latent_list_cycle, os.path.join(cache_dir, "latents.pt"))

        # Save all other data in single pickle
        avatar_data = {
            "frame_list_cycle": self.frame_list_cycle,
            "coord_list_cycle": self.coord_list_cycle,
            "mask_list_cycle": self.mask_list_cycle,
            "mask_coords_list_cycle": self.mask_coords_list_cycle,
        }
        with open(os.path.join(cache_dir, "avatar_data.pkl"), "wb") as f:
            pickle.dump(avatar_data, f)

        logger.info(f"Saved avatar cache to {cache_dir}")

    def _load_from_cache(self, cache_dir: str) -> None:
        """Load pre-computed avatar data from disk cache."""
        logger.info(f"Loading avatar from cache: {cache_dir}")

        # Load metadata
        with open(os.path.join(cache_dir, "metadata.json")) as f:
            metadata = json.load(f)

        self.fps = metadata["fps"]
        self.avatar_width = metadata["avatar_width"]
        self.avatar_height = metadata["avatar_height"]

        # Load latents
        self.input_latent_list_cycle = torch.load(
            os.path.join(cache_dir, "latents.pt"), map_location=self.device
        )

        # Load other data
        with open(os.path.join(cache_dir, "avatar_data.pkl"), "rb") as f:
            avatar_data = pickle.load(f)

        self.frame_list_cycle = avatar_data["frame_list_cycle"]
        self.coord_list_cycle = avatar_data["coord_list_cycle"]
        self.mask_list_cycle = avatar_data["mask_list_cycle"]
        self.mask_coords_list_cycle = avatar_data["mask_coords_list_cycle"]

        self._prepared = True
        logger.info(f"Loaded {len(self.frame_list_cycle)} frames from cache")

    def prepare_avatar(self, video_path: str, bbox_shift: int = 0) -> None:
        """
        Prepare an avatar from a reference video.

        This performs the heavy preprocessing work:
        - Extract frames from the video
        - Detect faces and landmarks
        - Compute bounding boxes with optional shift
        - Build coordinate and frame lists for inference
        - Pre-compute VAE latents for the reference frames

        Results are cached to disk for fast subsequent loads.

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

        # Check cache first
        cache_dir = self._get_cache_dir(video_path, bbox_shift)
        if self._cache_valid(cache_dir, video_path):
            self._load_from_cache(cache_dir)
            return

        logger.info(f"Preparing avatar from {video_path} (bbox_shift={bbox_shift})")

        # Get video properties
        self.fps = get_video_fps(video_path)

        # Create temp directory for extracted frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract frames from video
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

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

            # Detect faces and get landmarks/bboxes
            coord_list, frame_list = get_landmark_and_bbox(frame_files, bbox_shift)

            if not frame_list:
                raise FaceNotFoundError(f"Failed to read frames from {video_path}")

            # Get image dimensions
            self.avatar_height, self.avatar_width = frame_list[0].shape[:2]

            if not coord_list or all(c is None for c in coord_list):
                raise FaceNotFoundError(
                    f"No face detected in avatar video: {video_path}"
                )

            # Handle frames where face wasn't detected (use placeholder)
            self.coord_list = []
            self.frame_list = []

            for coord, frame in zip(coord_list, frame_list):
                if coord is None:
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

            for frame, coord in zip(self.frame_list_cycle, self.coord_list_cycle):
                x1, y1, x2, y2 = coord
                face_box = [x1, y1, x2, y2]
                mask, crop_box = get_image_prepare_material(frame, face_box, fp=self.fp)

                crop_frame = frame[y1:y2, x1:x2]
                crop_frame_resized = cv2.resize(
                    crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
                )

                latent = self.vae.get_latents_for_unet(crop_frame_resized)

                self.input_latent_list_cycle.append(latent)
                self.mask_coords_list_cycle.append(crop_box)
                self.mask_list_cycle.append(mask)

        self._prepared = True
        logger.info(f"Avatar prepared: {len(self.frame_list_cycle)} frames ready")

        # Save to cache
        self._save_to_cache(cache_dir, video_path, bbox_shift)

    def _is_silent(self, audio_float: np.ndarray) -> bool:
        """Check if audio is silent based on RMS energy."""
        rms = np.sqrt(np.mean(audio_float**2))
        return rms < self.silence_threshold

    def _generate_idle_frames(self, duration_seconds: float) -> list[np.ndarray]:
        """Generate idle animation frames (cycling reference frames without lip sync)."""
        num_frames = int(duration_seconds * self.fps)
        frames = []
        for i in range(num_frames):
            frame_idx = (self._idle_frame_index + i) % len(self.frame_list_cycle)
            frames.append(self.frame_list_cycle[frame_idx].copy())
        # Update idle frame index for continuity
        self._idle_frame_index = (self._idle_frame_index + num_frames) % len(
            self.frame_list_cycle
        )
        return frames

    def _extract_audio_features_direct(self, audio_16k: np.ndarray) -> list:
        """Extract Whisper features directly from numpy array (bypasses file I/O)."""
        # Split audio into 30s segments (same as AudioProcessor.get_audio_feature)
        sampling_rate = 16000
        segment_length = 30 * sampling_rate
        segments = [
            audio_16k[i : i + segment_length]
            for i in range(0, len(audio_16k), segment_length)
        ]

        features = []
        for segment in segments:
            audio_feature = self.audio_processor.feature_extractor(
                segment, return_tensors="pt", sampling_rate=sampling_rate
            ).input_features
            if self.dtype is not None:
                audio_feature = audio_feature.to(dtype=self.dtype)
            features.append(audio_feature)

        return features

    def generate_from_audio_chunk(
        self, audio_pcm: np.ndarray, audio_sample_rate: int
    ) -> list[np.ndarray]:
        """
        Generate talking-head frames from an audio chunk.

        Takes mono PCM audio and produces RGB video frames of the avatar
        speaking the audio. If audio is silent, returns idle animation frames.

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

        # Check for silence - return idle animation instead
        if self._is_silent(audio_float):
            duration = len(audio_pcm) / audio_sample_rate
            logger.debug(f"Silence detected, returning {duration:.2f}s idle animation")
            return self._generate_idle_frames(duration)

        # Resample to 16kHz if needed (Whisper requires 16kHz)
        if audio_sample_rate != 16000:
            import librosa

            audio_16k = librosa.resample(
                audio_float, orig_sr=audio_sample_rate, target_sr=16000
            )
        else:
            audio_16k = audio_float

        # Extract audio features directly (bypass file I/O)
        whisper_input_features = self._extract_audio_features_direct(audio_16k)
        librosa_length = len(audio_16k)

        if whisper_input_features is None or len(whisper_input_features) == 0:
            logger.warning("No audio features extracted")
            return []

        # Get whisper chunks for each frame
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features=whisper_input_features,
            device=self.device,
            weight_dtype=self.dtype,
            whisper=self.whisper,
            librosa_length=librosa_length,
            fps=self.fps,
        )

        if whisper_chunks is None or len(whisper_chunks) == 0:
            logger.warning("No whisper chunks generated")
            return []

        # Generate frames using batched inference
        output_frames = []
        num_frames = len(whisper_chunks)

        # Gather all inputs for batched processing
        frame_indices = [
            (self._frame_index + i) % len(self.frame_list_cycle)
            for i in range(num_frames)
        ]
        input_latents = torch.cat(
            [self.input_latent_list_cycle[idx] for idx in frame_indices], dim=0
        ).to(device=self.device, dtype=torch.float16)

        # Batched UNet + VAE inference
        with torch.no_grad():
            # Process all audio features in batch
            pe_cond = self.pe(whisper_chunks)

            if self.use_tensorrt and self.trt_model is not None:
                # TensorRT inference path
                timesteps = torch.zeros(
                    num_frames, dtype=torch.int64, device=self.device
                )
                pred_latents = self.trt_model.unet_inference(
                    input_latents, timesteps, pe_cond
                )
                pred_images = self.trt_model.decode_to_numpy(pred_latents)
            else:
                # PyTorch inference path
                timesteps = torch.zeros(
                    num_frames, dtype=torch.long, device=self.device
                )
                pred_latents = self.unet.model(
                    input_latents,
                    timesteps,
                    encoder_hidden_states=pe_cond,
                ).sample

                # Batched VAE decode
                pred_latents = pred_latents.to(
                    device=self.device, dtype=self.vae.vae.dtype
                )
                pred_images = self.vae.decode_latents(pred_latents)

        # Post-process each frame (CPU operations)
        for i in range(num_frames):
            frame_idx = frame_indices[i]

            # Get precomputed data
            ref_frame = self.frame_list_cycle[frame_idx]
            coord = self.coord_list_cycle[frame_idx]
            mask = self.mask_list_cycle[frame_idx]
            crop_box = self.mask_coords_list_cycle[frame_idx]

            pred_image = pred_images[i]

            # Blend generated face into reference frame
            x1, y1, x2, y2 = coord
            crop_h, crop_w = y2 - y1, x2 - x1
            pred_resized = cv2.resize(
                pred_image, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4
            )

            output_frame = ref_frame.copy()
            output_frame = get_image_blending(
                output_frame, pred_resized, coord, mask, crop_box
            )

            if output_frame.shape[2] == 4:
                output_frame = output_frame[:, :, :3]

            output_frames.append(output_frame)

        # Update frame index for continuity
        self._frame_index = (self._frame_index + num_frames) % len(
            self.frame_list_cycle
        )
        # Sync idle frame index
        self._idle_frame_index = self._frame_index

        logger.debug(f"Generated {len(output_frames)} frames from audio chunk")
        return output_frames

    def warmup(self) -> None:
        """
        Warm up GPU kernels by running a dummy inference.

        This should be called after prepare_avatar() to avoid first-inference
        latency when real audio arrives.
        """
        if not self._prepared:
            logger.warning("Cannot warmup: avatar not prepared")
            return

        logger.info("Warming up models...")
        # Generate 0.5s of dummy audio
        dummy_audio = np.zeros(8000, dtype=np.int16)
        try:
            # Force inference (bypass silence detection temporarily)
            old_threshold = self.silence_threshold
            self.silence_threshold = 0.0  # Accept any audio
            _ = self.generate_from_audio_chunk(dummy_audio, 16000)
            self.silence_threshold = old_threshold
            torch.cuda.synchronize()
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
            self.silence_threshold = old_threshold

    # ==================== Threading Support ====================

    def start_worker(self) -> None:
        """Start background worker thread for async frame generation."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("Worker thread already running")
            return

        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Started background worker thread")

    def stop_worker(self) -> None:
        """Stop background worker thread."""
        if self._worker_thread is None:
            return

        self._shutdown_event.set()
        # Unblock the queue
        try:
            self._request_queue.put_nowait(None)
        except queue.Full:
            pass

        self._worker_thread.join(timeout=5)
        if self._worker_thread.is_alive():
            logger.warning("Worker thread did not stop cleanly")
        else:
            logger.info("Worker thread stopped")
        self._worker_thread = None

    def _worker_loop(self) -> None:
        """Background worker loop that processes audio requests."""
        while not self._shutdown_event.is_set():
            try:
                request = self._request_queue.get(timeout=0.1)
                if request is None:
                    break

                audio_pcm, sample_rate = request
                frames = self.generate_from_audio_chunk(audio_pcm, sample_rate)
                self._result_queue.put(frames)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
                self._result_queue.put([])

    def generate_from_audio_chunk_threaded(
        self, audio_pcm: np.ndarray, audio_sample_rate: int
    ) -> list[np.ndarray]:
        """
        Submit audio to worker thread and get frames (blocking).

        This is an alternative to generate_from_audio_chunk() that uses the
        background worker thread. The worker must be started first with
        start_worker().

        Args:
            audio_pcm: Mono audio samples as int16 numpy array
            audio_sample_rate: Sample rate of the audio

        Returns:
            List of RGB frames
        """
        if self._worker_thread is None or not self._worker_thread.is_alive():
            raise RuntimeError("Worker thread not running. Call start_worker() first.")

        self._request_queue.put((audio_pcm, audio_sample_rate))
        return self._result_queue.get()

    # ==================== Properties ====================

    @property
    def is_prepared(self) -> bool:
        """Check if avatar has been prepared."""
        return self._prepared

    @property
    def frame_size(self) -> tuple[int, int]:
        """Get avatar frame dimensions (width, height)."""
        return (self.avatar_width, self.avatar_height)
