import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm


PART1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART1_DIR))

from src.model import SRCNN


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_frames(folder: str):
    return sorted([
        name for name in os.listdir(folder)
        if os.path.splitext(name.lower())[1] in IMG_EXTS
    ])


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred.clamp(0, 1) - gt.clamp(0, 1)) ** 2).item()
    if mse <= 1e-10:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d.view(1, 1, window_size, window_size)


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.clamp(0, 1).unsqueeze(0)
    gt = gt.clamp(0, 1).unsqueeze(0)
    _, channels, h, w = pred.shape
    window_size = min(11, h, w)
    if window_size % 2 == 0:
        window_size -= 1
    kernel = gaussian_kernel(window_size).expand(channels, 1, window_size, window_size)
    padding = window_size // 2
    mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_gt = F.conv2d(gt, kernel, padding=padding, groups=channels)
    mu_pred_sq = mu_pred.pow(2)
    mu_gt_sq = mu_gt.pow(2)
    mu_pred_gt = mu_pred * mu_gt
    sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_pred_sq
    sigma_gt_sq = F.conv2d(gt * gt, kernel, padding=padding, groups=channels) - mu_gt_sq
    sigma_pred_gt = F.conv2d(pred * gt, kernel, padding=padding, groups=channels) - mu_pred_gt
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score = ((2 * mu_pred_gt + c1) * (2 * sigma_pred_gt + c2)) / (
        (mu_pred_sq + mu_gt_sq + c1) * (sigma_pred_sq + sigma_gt_sq + c2)
    )
    return float(score.mean().item())


def load_srcnn(ckpt_path: str, device: torch.device):
    model = SRCNN().to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return model


def weighted_temporal(frames, idx, scale: int, weights, apply_unsharp: bool):
    radius = len(weights) // 2
    acc = None
    for offset, weight in zip(range(-radius, radius + 1), weights):
        nidx = max(0, min(idx + offset, len(frames) - 1))
        lr = frames[nidx]
        up = lr.resize((lr.width * scale, lr.height * scale), resample=Image.BICUBIC)
        arr = np.array(up.convert("RGB")).astype(np.float32) * float(weight)
        acc = arr if acc is None else acc + arr
    out = Image.fromarray(acc.clip(0, 255).astype(np.uint8))
    if apply_unsharp:
        out = out.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    return out


def write_csv(path: str, fieldnames, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Part 1 baselines on validation set.")
    parser.add_argument("--lr-root", default="/home/fc/Coding/CV/data/val/val_sharp_bicubic/X4")
    parser.add_argument("--gt-root", default="/home/fc/Coding/CV/data/val/val_sharp")
    parser.add_argument("--ckpt", default="/home/fc/Coding/CV/part1/checkpoints/srcnn_x4_epoch20.pth")
    parser.add_argument("--output", default="/home/fc/Coding/CV/part1/outputs/metrics_val_part1")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_srcnn(args.ckpt, device)
    methods = ["Bicubic", "Lanczos", "SRCNN", "Temporal"]
    totals = {m: {"psnr": 0.0, "ssim": 0.0, "frames": 0} for m in methods}
    per_seq = []
    per_frame = []

    seqs = sorted(d for d in os.listdir(args.lr_root) if os.path.isdir(os.path.join(args.lr_root, d)))
    if args.max_seqs > 0:
        seqs = seqs[: args.max_seqs]

    print(f"Evaluating Part 1 on {len(seqs)} sequences")
    for seq in tqdm(seqs, desc="Sequences"):
        lr_dir = os.path.join(args.lr_root, seq)
        gt_dir = os.path.join(args.gt_root, seq)
        names = list_frames(lr_dir)
        lr_images = [Image.open(os.path.join(lr_dir, name)).convert("RGB") for name in names]
        seq_totals = {m: {"psnr": 0.0, "ssim": 0.0, "frames": 0} for m in methods}

        for idx, name in enumerate(tqdm(names, desc=seq, leave=False)):
            gt_img = Image.open(os.path.join(gt_dir, name)).convert("RGB")
            gt_t = pil_to_tensor(gt_img)

            lr = lr_images[idx]
            preds = {
                "Bicubic": lr.resize(gt_img.size, resample=Image.BICUBIC),
                "Lanczos": lr.resize(gt_img.size, resample=Image.LANCZOS),
                "Temporal": weighted_temporal(lr_images, idx, args.scale, [0.25, 0.5, 0.25], True),
            }

            with torch.no_grad():
                x = pil_to_tensor(preds["Bicubic"]).unsqueeze(0).to(device)
                sr = model(x).clamp(0, 1).squeeze(0).cpu()

            pred_tensors = {
                "Bicubic": pil_to_tensor(preds["Bicubic"]),
                "Lanczos": pil_to_tensor(preds["Lanczos"]),
                "Temporal": pil_to_tensor(preds["Temporal"]),
                "SRCNN": sr,
            }

            for method, pred_t in pred_tensors.items():
                frame_psnr = psnr(pred_t, gt_t)
                frame_ssim = ssim(pred_t, gt_t)
                totals[method]["psnr"] += frame_psnr
                totals[method]["ssim"] += frame_ssim
                totals[method]["frames"] += 1
                seq_totals[method]["psnr"] += frame_psnr
                seq_totals[method]["ssim"] += frame_ssim
                seq_totals[method]["frames"] += 1
                per_frame.append({
                    "sequence": seq,
                    "frame": name,
                    "method": method,
                    "psnr": frame_psnr,
                    "ssim": frame_ssim,
                })

        for method in methods:
            n = seq_totals[method]["frames"]
            per_seq.append({
                "sequence": seq,
                "method": method,
                "frames": n,
                "psnr": seq_totals[method]["psnr"] / n,
                "ssim": seq_totals[method]["ssim"] / n,
            })

    summary = []
    for method in methods:
        n = totals[method]["frames"]
        summary.append({
            "method": method,
            "frames": n,
            "psnr": totals[method]["psnr"] / n,
            "ssim": totals[method]["ssim"] / n,
        })

    ensure_dir(args.output)
    write_csv(os.path.join(args.output, "summary.csv"), ["method", "frames", "psnr", "ssim"], summary)
    write_csv(os.path.join(args.output, "per_sequence.csv"), ["sequence", "method", "frames", "psnr", "ssim"], per_seq)
    write_csv(os.path.join(args.output, "per_frame.csv"), ["sequence", "frame", "method", "psnr", "ssim"], per_frame)
    with open(os.path.join(args.output, "summary.md"), "w", encoding="utf-8") as f:
        f.write("| Method | Frames | PSNR | SSIM |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in summary:
            f.write(f"| {row['method']} | {row['frames']} | {row['psnr']:.4f} | {row['ssim']:.4f} |\n")

    print("| Method | Frames | PSNR | SSIM |")
    print("|---|---:|---:|---:|")
    for row in summary:
        print(f"| {row['method']} | {row['frames']} | {row['psnr']:.4f} | {row['ssim']:.4f} |")
    print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
