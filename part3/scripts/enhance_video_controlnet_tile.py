import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image


CURRENT_DIR = Path(__file__).resolve().parent
PART3_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PART3_DIR / "src"))

from io_utils import ensure_dir, list_images
from video_utils import images_to_video


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def dtype_from_name(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_video_fps(video_path: str, fallback: float = 30.0) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        if "/" in out:
            num, den = out.split("/")
            den = float(den)
            if den != 0:
                return float(num) / den
        return float(out)
    except Exception:
        return fallback


def extract_video_frames(video_path: str, frames_dir: str, max_frames=None):
    ensure_dir(frames_dir)
    pattern = os.path.join(frames_dir, "%08d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-start_number",
        "0",
    ]
    if max_frames is not None:
        cmd.extend(["-vframes", str(max_frames)])
    cmd.extend(["-vsync", "0", pattern])
    run_cmd(cmd)


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
    parser = argparse.ArgumentParser(description="Enhance a video with low-strength ControlNet-Tile img2img.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input", type=str, default=None, help="Override input video")
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Override max frames")
    parser.add_argument("--strength", type=float, default=None, help="Override img2img strength")
    parser.add_argument("--max-side", type=int, default=None, help="Override diffusion max side")
    args = parser.parse_args()

    try:
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing diffusion dependencies. Activate part3_env and install diffusers stack."
        ) from exc

    cfg = load_config(args.config)
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]

    input_video = args.input or paths["input_video"]
    output_dir = args.output or paths["output_dir"]
    max_frames = args.max_frames if args.max_frames is not None else gen_cfg.get("max_frames")
    strength = args.strength if args.strength is not None else float(gen_cfg.get("strength", 0.15))
    max_side = args.max_side if args.max_side is not None else int(gen_cfg.get("max_side", 960))

    ensure_dir(output_dir)
    input_frames_dir = os.path.join(output_dir, "input_frames")
    enhanced_frames_dir = os.path.join(output_dir, "frames")
    video_dir = os.path.join(output_dir, "videos")

    if os.path.isdir(input_frames_dir):
        shutil.rmtree(input_frames_dir)
    if os.path.isdir(enhanced_frames_dir):
        shutil.rmtree(enhanced_frames_dir)
    ensure_dir(enhanced_frames_dir)
    ensure_dir(video_dir)

    fps = gen_cfg.get("fps")
    if fps is None:
        fps = get_video_fps(input_video)
    else:
        fps = float(fps)

    print(f"Extracting frames from: {input_video}")
    extract_video_frames(input_video, input_frames_dir, max_frames=max_frames)

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

    image_paths = list_images(input_frames_dir)
    generator = torch.Generator(device=device).manual_seed(int(gen_cfg.get("seed", 42)))
    multiple_of = int(gen_cfg.get("multiple_of", 8))

    print("=" * 72)
    print("ControlNet-Tile Video Enhancement")
    print("=" * 72)
    print(f"Frames     : {len(image_paths)}")
    print(f"FPS        : {fps}")
    print(f"Strength   : {strength}")
    print(f"Max side   : {max_side}")
    print(f"Output dir : {output_dir}")
    print("=" * 72)

    for idx, path in enumerate(image_paths, 1):
        image = Image.open(path).convert("RGB")
        work_image, original_size = resize_for_diffusion(
            image,
            max_side=max_side,
            multiple_of=multiple_of,
        )
        result = pipe(
            prompt=gen_cfg["prompt"],
            negative_prompt=gen_cfg.get("negative_prompt", ""),
            image=work_image,
            control_image=work_image,
            num_inference_steps=int(gen_cfg.get("num_inference_steps", 10)),
            guidance_scale=float(gen_cfg.get("guidance_scale", 4.5)),
            controlnet_conditioning_scale=float(gen_cfg.get("controlnet_conditioning_scale", 0.85)),
            strength=float(strength),
            generator=generator,
        ).images[0]
        if result.size != original_size:
            result = result.resize(original_size, Image.BICUBIC)
        out_path = os.path.join(enhanced_frames_dir, os.path.basename(path))
        result.save(out_path)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if idx == 1 or idx % 10 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] {out_path}")

    video_path = os.path.join(video_dir, "enhanced.mp4")
    images_to_video(list_images(enhanced_frames_dir), video_path, fps=int(round(fps)))
    print(f"Saved enhanced frames to: {enhanced_frames_dir}")
    print(f"Saved enhanced video to: {video_path}")

    if not bool(gen_cfg.get("keep_frames", True)):
        shutil.rmtree(input_frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

