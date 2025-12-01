"""
TensorRT runtime for MuseTalk inference.

This module provides TensorRT-accelerated inference for UNet and VAE models.
"""

import logging
import os

import numpy as np
import tensorrt as trt
import torch


logger = logging.getLogger(__name__)


class TensorRTEngine:
    """Wrapper for TensorRT engine execution."""

    def __init__(self, engine_path: str, device: str = "cuda"):
        """
        Load a TensorRT engine from file.

        Args:
            engine_path: Path to the .engine file
            device: CUDA device string
        """
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        self.device = device
        self.engine_path = engine_path

        # Initialize CUDA via PyTorch first (fixes TRT CUDA init issues)
        _ = torch.zeros(1, device=device)

        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get input/output info
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes[name] = shape
            else:
                self.output_names.append(name)
                self.output_shapes[name] = shape

        logger.info(
            f"Loaded TensorRT engine: {os.path.basename(engine_path)} "
            f"({len(self.input_names)} inputs, {len(self.output_names)} outputs)"
        )

    def infer(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Run inference on the TensorRT engine.

        Args:
            inputs: Dictionary mapping input names to torch tensors on GPU

        Returns:
            Dictionary mapping output names to torch tensors on GPU
        """
        # Set input shapes (for dynamic batch)
        for name, tensor in inputs.items():
            shape = list(tensor.shape)
            self.context.set_input_shape(name, shape)

        # Allocate output tensors
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(tuple(shape), dtype=dtype, device=self.device)

        # Set tensor addresses
        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Run inference
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        return outputs

    def _trt_dtype_to_torch(self, trt_dtype):
        """Convert TensorRT dtype to PyTorch dtype."""
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.int8: torch.int8,
            trt.bool: torch.bool,
        }
        return mapping.get(trt_dtype, torch.float32)


class TensorRTMuseTalk:
    """TensorRT-accelerated MuseTalk inference."""

    def __init__(
        self,
        unet_engine_path: str,
        vae_engine_path: str,
        device: str = "cuda",
    ):
        """
        Initialize TensorRT engines for MuseTalk.

        Args:
            unet_engine_path: Path to UNet .engine file
            vae_engine_path: Path to VAE decoder .engine file
            device: CUDA device string
        """
        self.device = device

        logger.info("Loading TensorRT engines...")
        self.unet = TensorRTEngine(unet_engine_path, device)
        self.vae = TensorRTEngine(vae_engine_path, device)
        logger.info("TensorRT engines loaded successfully")

    def unet_inference(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run UNet inference.

        Args:
            latents: Input latents (batch, 8, 32, 32) fp16
            timesteps: Timesteps (batch,) int64
            encoder_hidden_states: Audio features (batch, 50, 384) fp16

        Returns:
            Predicted latents (batch, 4, 32, 32) fp16
        """
        inputs = {
            "sample": latents.contiguous(),
            "timestep": timesteps.contiguous(),
            "encoder_hidden_states": encoder_hidden_states.contiguous(),
        }
        outputs = self.unet.infer(inputs)
        return outputs["output"]

    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Run VAE decoder inference.

        Args:
            latents: Latents to decode (batch, 4, 32, 32) fp16

        Returns:
            Decoded images (batch, 3, 256, 256) fp16
        """
        inputs = {"latent": latents.contiguous()}
        outputs = self.vae.infer(inputs)
        return outputs["image"]

    def decode_to_numpy(self, latents: torch.Tensor) -> np.ndarray:
        """
        Decode latents to numpy images (uint8).

        Args:
            latents: Latents (batch, 4, 32, 32)

        Returns:
            Images as numpy array (batch, 256, 256, 3) uint8 BGR
        """
        # VAE decode
        images = self.vae_decode(latents)

        # Post-process: (batch, 3, 256, 256) -> (batch, 256, 256, 3) uint8
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC
        images = (images * 255).round().to(torch.uint8)
        images = images.cpu().numpy()

        # RGB to BGR
        images = images[..., ::-1].copy()

        return images
