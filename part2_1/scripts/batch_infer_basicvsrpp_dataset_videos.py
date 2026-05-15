import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from mmagic.models import BasicVSRPlusPlusNet
from PIL import Image
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from infer_basicvsrpp_video import (
    infer_and_write_chunked_sequence,
    list_frames,
    load_basicvsrpp_generator_checkpoint,
)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def direct_image_count(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMG_EXTS)


def iter_sequence_dirs(root: Path):
    for path, dirs, files in os.walk(root):
        del dirs
        folder = Path(path)
        if any(Path(name).suffix.lower() in IMG_EXTS for name in files):
            yield folder


def read_size(path: str):
    with Image.open(path) as img:
        return img.size


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
    parser = argparse.ArgumentParser(description="Batch BasicVSR++ inference for folder datasets, saving videos only.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--ckpt",
        default="/home/fc/Coding/CV/part2_1/mmagic/work_dirs/basicvsr-pp_c64n7_fc_finetune/basicvsr-pp_c64n7_fc_finetune/best_PSNR_iter_20000.pth",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--chunk-overlap", type=int, default=2)
    parser.add_argument("--cpu-cache-length", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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
    model = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=7, cpu_cache_length=args.cpu_cache_length)
    load_basicvsrpp_generator_checkpoint(model, args.ckpt)
    model = model.to(device).eval()

    print("=" * 72)
    print("Batch BasicVSR++ Video Inference")
    print("=" * 72)
    print(f"Dataset     : {args.dataset_name}")
    print(f"Input root  : {input_root}")
    print(f"Output root : {output_root}")
    print(f"Sequences   : {len(sequence_dirs)}")
    print(f"Device      : {device}")
    print("=" * 72)

    rows = []
    for seq_dir in tqdm(sequence_dirs, desc=args.dataset_name):
        rel = seq_dir.relative_to(input_root)
        video_path = output_root / rel / "sr.mp4"
        frame_paths = list_frames(str(seq_dir))
        if not frame_paths:
            continue
        in_w, in_h = read_size(frame_paths[0])
        out_w, out_h = in_w * 4, in_h * 4

        if args.force or not video_path.is_file():
            infer_and_write_chunked_sequence(
                model=model,
                frame_paths=frame_paths,
                out_video_path=str(video_path),
                fps=args.fps,
                device=device,
                chunk_size=args.chunk_size,
                overlap=args.chunk_overlap,
            )
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
