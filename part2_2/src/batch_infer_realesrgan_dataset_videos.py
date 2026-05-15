import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from infer import load_named_model_for_inference
from tiler import RealESRGANTiler
from video_utils import FFmpegVideoWriter


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def iter_sequence_dirs(root: Path):
    for path, dirs, files in os.walk(root):
        del dirs
        if any(Path(name).suffix.lower() in IMG_EXTS for name in files):
            yield Path(path)


def list_frames(folder: Path):
    return sorted([
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    ])


def load_tensor(path: Path):
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).clamp(0, 1)


def write_csv(path: str, rows):
    ensure_dir(os.path.dirname(path))
    fields = [
        "dataset",
        "sequence",
        "frames",
        "input_width",
        "input_height",
        "output_width",
        "output_height",
        "video",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Real-ESRGAN inference for folder datasets, saving videos only.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--ckpt",
        default="/home/fc/Coding/CV/part2_2/experiments/realesrgan_x4plus_official_conservative_4epoch/checkpoints/net_g_best.pth",
    )
    parser.add_argument("--model-name", default="RealESRGAN_x4plus")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--tile-pad", type=int, default=10)
    parser.add_argument("--pre-pad", type=int, default=0)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    sequence_dirs = sorted(iter_sequence_dirs(input_root))
    if args.max_seqs > 0:
        sequence_dirs = sequence_dirs[: args.max_seqs]
    if not sequence_dirs:
        raise ValueError(f"No image sequence folders found in {input_root}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, native_scale, used_key = load_named_model_for_inference(args.model_name, args.ckpt, device)
    tiler = RealESRGANTiler(
        model=model,
        scale=native_scale,
        device=device,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=(device.type == "cuda" and not args.fp32),
    )

    print("=" * 72)
    print("Batch Real-ESRGAN Video Inference")
    print("=" * 72)
    print(f"Dataset     : {args.dataset_name}")
    print(f"Input root  : {input_root}")
    print(f"Output root : {output_root}")
    print(f"Checkpoint  : {args.ckpt}")
    print(f"Load key    : {used_key}")
    print(f"Sequences   : {len(sequence_dirs)}")
    print(f"Device      : {device}")
    print("=" * 72)

    rows = []
    for seq_dir in tqdm(sequence_dirs, desc=args.dataset_name):
        rel = seq_dir.relative_to(input_root)
        frame_paths = list_frames(seq_dir)
        if not frame_paths:
            continue

        with Image.open(frame_paths[0]) as first:
            in_w, in_h = first.size
        out_w, out_h = in_w * native_scale, in_h * native_scale
        video_path = output_root / rel / "sr.mp4"

        if args.force or not video_path.is_file():
            ensure_dir(str(video_path.parent))
            writer = FFmpegVideoWriter(str(video_path), (out_w, out_h), fps=args.fps)
            try:
                for path in frame_paths:
                    pred = tiler.enhance_tensor(load_tensor(path)).squeeze(0).cpu().clamp(0, 1)
                    image = to_pil_image(pred)
                    writer.append_pil(image)
                    del pred
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            finally:
                writer.close()

        rows.append({
            "dataset": args.dataset_name,
            "sequence": str(rel),
            "frames": len(frame_paths),
            "input_width": in_w,
            "input_height": in_h,
            "output_width": out_w,
            "output_height": out_h,
            "video": str(video_path),
        })

    write_csv(str(output_root / "manifest.csv"), rows)
    print(f"Saved manifest: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
