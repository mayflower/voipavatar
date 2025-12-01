#!/usr/bin/env python3
"""
Export MuseTalk models to TensorRT format for faster inference.

This script:
1. Loads the MuseTalk models (UNet, VAE)
2. Exports them to ONNX format with dynamic batch sizes
3. Converts ONNX to TensorRT engines

Usage:
    python export_tensorrt.py --output-dir ./models/tensorrt
"""

import argparse
import logging
import os
import sys

import onnx
import tensorrt as trt
import torch


# Add MuseTalk to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MuseTalk"))

from musetalk.utils.utils import load_all_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_unet_to_onnx(unet_model, output_path: str, batch_size: int = 25):
    """Export UNet model to ONNX format."""
    logger.info("Exporting UNet to ONNX...")

    unet_model = unet_model.eval()
    device = next(unet_model.parameters()).device
    dtype = next(unet_model.parameters()).dtype

    # Create dummy inputs matching actual inference shapes
    # MuseTalk UNet2DConditionModel inputs:
    # - sample: (batch, 8, 32, 32) - concatenated masked + reference latents
    #   (4 channels masked latent + 4 channels reference latent = 8 channels)
    # - timestep: (batch,) - always 0 for single-step
    # - encoder_hidden_states: (batch, seq_len, embed_dim) - from PE (Whisper features)
    batch = batch_size
    dummy_latent = torch.randn(batch, 8, 32, 32, device=device, dtype=dtype)
    dummy_timestep = torch.zeros(batch, dtype=torch.long, device=device)
    dummy_encoder_hidden_states = torch.randn(batch, 50, 384, device=device, dtype=dtype)

    # Export with dynamic axes for batch dimension
    torch.onnx.export(
        unet_model,
        (dummy_latent, dummy_timestep, dummy_encoder_hidden_states),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["output"],
        dynamic_axes={
            "sample": {0: "batch"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "output": {0: "batch"},
        },
    )

    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info(f"UNet ONNX exported to {output_path}")


def export_vae_decoder_to_onnx(vae_model, output_path: str, batch_size: int = 25):
    """Export VAE decoder to ONNX format."""
    logger.info("Exporting VAE decoder to ONNX...")

    vae_model = vae_model.eval()
    device = next(vae_model.parameters()).device
    dtype = next(vae_model.parameters()).dtype

    # Create wrapper for VAE decoder only
    class VAEDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            # Scale latents (same as in diffusers)
            latents = latents / self.vae.config.scaling_factor
            # Decode
            return self.vae.decode(latents, return_dict=False)[0]

    decoder = VAEDecoder(vae_model)

    # Dummy input: (batch, 4, 32, 32)
    batch = batch_size
    dummy_latent = torch.randn(batch, 4, 32, 32, device=device, dtype=dtype)

    torch.onnx.export(
        decoder,
        dummy_latent,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={
            "latent": {0: "batch"},
            "image": {0: "batch"},
        },
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info(f"VAE decoder ONNX exported to {output_path}")


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    min_batch: int = 1,
    opt_batch: int = 25,
    max_batch: int = 50,
):
    """Convert ONNX model to TensorRT engine."""
    logger.info(f"Building TensorRT engine from {onnx_path}...")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    logger.info(f"Network has {network.num_inputs} inputs and {network.num_outputs} outputs")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 mode")

    # Set optimization profiles for dynamic batch
    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        name = input_tensor.name
        shape = input_tensor.shape

        # Replace -1 (dynamic) with min/opt/max batch sizes
        min_shape = [min_batch if d == -1 else d for d in shape]
        opt_shape = [opt_batch if d == -1 else d for d in shape]
        max_shape = [max_batch if d == -1 else d for d in shape]

        logger.info(f"  Input '{name}': min={min_shape}, opt={opt_shape}, max={max_shape}")
        profile.set_shape(name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)

    # Build engine
    logger.info("Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(f"TensorRT engine saved to {engine_path}")
    return engine_path


def main():
    parser = argparse.ArgumentParser(description="Export MuseTalk models to TensorRT")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/tensorrt",
        help="Directory to save exported models",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Only export to ONNX, skip TensorRT conversion",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 precision for TensorRT (default: True)",
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
        help="Minimum batch size for TensorRT optimization",
    )
    parser.add_argument(
        "--opt-batch",
        type=int,
        default=25,
        help="Optimal batch size for TensorRT optimization",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=50,
        help="Maximum batch size for TensorRT optimization",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading MuseTalk models...")
    vae, unet, pe = load_all_model()

    # Move to GPU with FP16
    device = "cuda"
    dtype = torch.float16
    vae.vae = vae.vae.to(device).to(dtype)
    unet.model = unet.model.to(device).to(dtype)

    # Export UNet
    unet_onnx_path = os.path.join(args.output_dir, "unet.onnx")
    export_unet_to_onnx(unet.model, unet_onnx_path, batch_size=args.opt_batch)

    # Export VAE decoder
    vae_onnx_path = os.path.join(args.output_dir, "vae_decoder.onnx")
    export_vae_decoder_to_onnx(vae.vae, vae_onnx_path, batch_size=args.opt_batch)

    if args.onnx_only:
        logger.info("ONNX export complete (skipping TensorRT conversion)")
        return 0

    # Convert to TensorRT
    unet_engine_path = os.path.join(args.output_dir, "unet.engine")
    build_tensorrt_engine(
        unet_onnx_path,
        unet_engine_path,
        fp16=args.fp16,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
    )

    vae_engine_path = os.path.join(args.output_dir, "vae_decoder.engine")
    build_tensorrt_engine(
        vae_onnx_path,
        vae_engine_path,
        fp16=args.fp16,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
    )

    logger.info("=" * 60)
    logger.info("TensorRT export complete!")
    logger.info(f"  UNet engine: {unet_engine_path}")
    logger.info(f"  VAE engine: {vae_engine_path}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
