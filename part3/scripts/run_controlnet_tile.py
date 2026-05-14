import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image


CURRENT_DIR = Path(__file__).resolve().parent
PART3_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PART3_DIR / "src"))

from io_utils import ensure_dir, list_images


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dtype_from_name(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def resize_for_diffusion(image: Image.Image, max_side: int, multiple_of: int):
    original_size = image.size
    if max_side <= 0:
        return image, original_size

    width, height = image.size
    scale = min(1.0, float(max_side) / float(max(width, height)))
    new_w = max(multiple_of, int(round(width * scale / multiple_of)) * multiple_of)
    new_h = max(multiple_of, int(round(height * scale / multiple_of)) * multiple_of)

    if (new_w, new_h) == image.size:
        return image, original_size
    return image.resize((new_w, new_h), Image.BICUBIC), original_size


def main():
    parser = argparse.ArgumentParser(description="Optional Stable Diffusion + ControlNet-Tile enhancer.")
    parser.add_argument("--config", type=str, required=True, help="Path to controlnet_tile_*.yaml")
    args = parser.parse_args()

    try:
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing diffusion dependencies. Install them with:\n"
            "pip install diffusers transformers accelerate controlnet-aux safetensors"
        ) from exc

    cfg = load_config(args.config)
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]

    input_dir = paths["input_dir"]
    output_dir = paths["output_dir"]
    ensure_dir(output_dir)

    device_name = model_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    dtype = dtype_from_name(model_cfg.get("torch_dtype", "float16"))
    if device.type == "cpu":
        dtype = torch.float32

    controlnet = ControlNetModel.from_pretrained(
        model_cfg["controlnet_model"],
        torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_cfg["base_model"],
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    image_paths = list_images(input_dir)
    max_frames = gen_cfg.get("max_frames")
    if max_frames is not None:
        image_paths = image_paths[: int(max_frames)]

    generator = torch.Generator(device=device).manual_seed(int(gen_cfg.get("seed", 42)))
    max_side = int(gen_cfg.get("max_side", 768))
    multiple_of = int(gen_cfg.get("multiple_of", 8))
    print(f"Enhancing {len(image_paths)} frames with ControlNet-Tile...")
    print(f"Diffusion max side: {max_side}, multiple_of: {multiple_of}")

    for idx, path in enumerate(image_paths, 1):
        image = Image.open(path).convert("RGB")
        image, original_size = resize_for_diffusion(
            image,
            max_side=max_side,
            multiple_of=multiple_of,
        )
        result = pipe(
            prompt=gen_cfg["prompt"],
            negative_prompt=gen_cfg.get("negative_prompt", ""),
            image=image,
            control_image=image,
            num_inference_steps=int(gen_cfg.get("num_inference_steps", 20)),
            guidance_scale=float(gen_cfg.get("guidance_scale", 6.0)),
            controlnet_conditioning_scale=float(gen_cfg.get("controlnet_conditioning_scale", 0.75)),
            strength=float(gen_cfg.get("strength", 0.28)),
            generator=generator,
        ).images[0]
        if result.size != original_size:
            result = result.resize(original_size, Image.BICUBIC)
        out_path = os.path.join(output_dir, os.path.basename(path))
        result.save(out_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if idx == 1 or idx % 10 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] {out_path}")

    print(f"Saved ControlNet-Tile outputs to: {output_dir}")


if __name__ == "__main__":
    main()
