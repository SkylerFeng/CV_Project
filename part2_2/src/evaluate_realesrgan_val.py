import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from infer import load_named_model_for_inference
from tiler import RealESRGANTiler
from evaluate import calculate_psnr, calculate_ssim


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_frames(folder: str):
    return sorted([
        name for name in os.listdir(folder)
        if os.path.splitext(name.lower())[1] in IMG_EXTS
    ])


def load_tensor(path: str) -> torch.Tensor:
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).clamp(0, 1)


def write_csv(path: str, fieldnames, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate Real-ESRGAN on full val set.")
    parser.add_argument("--lr-root", default="/home/fc/Coding/CV/data/val/val_sharp_bicubic/X4")
    parser.add_argument("--gt-root", default="/home/fc/Coding/CV/data/val/val_sharp")
    parser.add_argument("--ckpt", default="/home/fc/Coding/CV/part2_2/models/RealESRGAN_x4plus.pth")
    parser.add_argument("--output", default="/home/fc/Coding/CV/part2_2/results/metrics_val_realesrgan_x4plus_full")
    parser.add_argument("--model-name", default="RealESRGAN_x4plus")
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--tile-pad", type=int, default=10)
    parser.add_argument("--pre-pad", type=int, default=0)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seqs", type=int, default=0)
    args = parser.parse_args()

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

    seqs = sorted(d for d in os.listdir(args.lr_root) if os.path.isdir(os.path.join(args.lr_root, d)))
    if args.max_seqs > 0:
        seqs = seqs[: args.max_seqs]

    per_seq = []
    per_frame = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_frames = 0

    print("=" * 72)
    print("Real-ESRGAN Full Val Evaluation")
    print("=" * 72)
    print(f"LR root   : {args.lr_root}")
    print(f"GT root   : {args.gt_root}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Load key  : {used_key}")
    print(f"Output    : {args.output}")
    print(f"Sequences : {len(seqs)}")
    print(f"Tile      : {args.tile}")
    print("=" * 72)

    for seq in tqdm(seqs, desc="Sequences"):
        lr_dir = os.path.join(args.lr_root, seq)
        gt_dir = os.path.join(args.gt_root, seq)
        names = list_frames(lr_dir)
        seq_psnr = 0.0
        seq_ssim = 0.0

        for name in tqdm(names, desc=seq, leave=False):
            lr = load_tensor(os.path.join(lr_dir, name))
            gt = load_tensor(os.path.join(gt_dir, name)).to(device)
            pred = tiler.enhance_tensor(lr).to(device)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
            pred = pred.clamp(0, 1)
            gt = gt.clamp(0, 1)
            frame_psnr = calculate_psnr(pred, gt)
            frame_ssim = calculate_ssim(pred, gt)
            seq_psnr += frame_psnr
            seq_ssim += frame_ssim
            total_psnr += frame_psnr
            total_ssim += frame_ssim
            total_frames += 1
            per_frame.append({
                "sequence": seq,
                "frame": name,
                "psnr": frame_psnr,
                "ssim": frame_ssim,
            })
            del lr, gt, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

        n = len(names)
        per_seq.append({
            "sequence": seq,
            "frames": n,
            "psnr": seq_psnr / n,
            "ssim": seq_ssim / n,
        })
        print(f"{seq}: frames={n}, PSNR={seq_psnr / n:.4f}, SSIM={seq_ssim / n:.4f}", flush=True)

    summary = [{
        "method": args.model_name,
        "sequences": len(seqs),
        "frames": total_frames,
        "psnr": total_psnr / total_frames,
        "ssim": total_ssim / total_frames,
    }]

    ensure_dir(args.output)
    write_csv(os.path.join(args.output, "summary.csv"), ["method", "sequences", "frames", "psnr", "ssim"], summary)
    write_csv(os.path.join(args.output, "per_sequence.csv"), ["sequence", "frames", "psnr", "ssim"], per_seq)
    write_csv(os.path.join(args.output, "per_frame.csv"), ["sequence", "frame", "psnr", "ssim"], per_frame)
    with open(os.path.join(args.output, "summary.md"), "w", encoding="utf-8") as f:
        f.write("| Method | Sequences | Frames | PSNR | SSIM |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        row = summary[0]
        f.write(f"| {row['method']} | {row['sequences']} | {row['frames']} | {row['psnr']:.4f} | {row['ssim']:.4f} |\n")

    row = summary[0]
    print("| Method | Sequences | Frames | PSNR | SSIM |")
    print("|---|---:|---:|---:|---:|")
    print(f"| {row['method']} | {row['sequences']} | {row['frames']} | {row['psnr']:.4f} | {row['ssim']:.4f} |")
    print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
