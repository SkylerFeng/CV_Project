import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmagic.models import BasicVSRPlusPlusNet
from PIL import Image
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from infer_basicvsrpp_video import (
    infer_full_sequence,
    list_frames,
    load_basicvsrpp_generator_checkpoint,
    load_frame_paths,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_rgb_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).float()


def crop_border_tensor(x: torch.Tensor, crop_border: int) -> torch.Tensor:
    if crop_border <= 0:
        return x
    _, h, w = x.shape
    if crop_border * 2 >= h or crop_border * 2 >= w:
        raise ValueError(f"crop_border={crop_border} is too large for image size {w}x{h}")
    return x[:, crop_border:-crop_border, crop_border:-crop_border]


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, crop_border: int = 0) -> float:
    pred = crop_border_tensor(pred.detach().cpu().clamp(0, 1), crop_border)
    target = crop_border_tensor(target.detach().cpu().clamp(0, 1), crop_border)
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-10:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d.view(1, 1, window_size, window_size)


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, crop_border: int = 0) -> float:
    pred = crop_border_tensor(pred.detach().cpu().clamp(0, 1), crop_border).unsqueeze(0)
    target = crop_border_tensor(target.detach().cpu().clamp(0, 1), crop_border).unsqueeze(0)

    _, channels, h, w = pred.shape
    window_size = min(11, h, w)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return 1.0 if torch.equal(pred, target) else 0.0

    kernel = gaussian_kernel(window_size=window_size).to(pred.device, pred.dtype)
    kernel = kernel.expand(channels, 1, window_size, window_size)
    padding = window_size // 2

    mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_pred_target

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = (
        (2 * mu_pred_target + c1)
        * (2 * sigma_pred_target + c2)
        / ((mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2))
    )
    return float(ssim_map.mean().item())


def build_gt_map(gt_dir: str):
    return {os.path.basename(path): path for path in list_frames(gt_dir)}


def evaluate_sequence(
    model,
    lr_dir: str,
    gt_dir: str,
    device: torch.device,
    chunk_size: int,
    overlap: int,
    crop_border: int,
):
    frame_paths = list_frames(lr_dir)
    gt_map = build_gt_map(gt_dir)
    filenames = [os.path.basename(path) for path in frame_paths]
    missing = [name for name in filenames if name not in gt_map]
    if missing:
        raise ValueError(f"{lr_dir}: missing GT frames, e.g. {missing[:5]}")

    total_frames = len(frame_paths)
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2 - 1))
    step = chunk_size - 2 * overlap
    if step <= 0:
        raise ValueError("chunk_size must be larger than 2 * chunk_overlap")

    frame_rows = []
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for start in tqdm(range(0, total_frames, step), desc=os.path.basename(lr_dir), leave=False):
        end = min(start + chunk_size, total_frames)
        chunk = load_frame_paths(frame_paths[start:end])
        chunk_out = infer_full_sequence(model, chunk, device)

        keep_start = 0 if start == 0 else overlap
        keep_end = chunk_out.shape[0] if end == total_frames else chunk_out.shape[0] - overlap

        for local_idx in range(keep_start, keep_end):
            global_idx = start + local_idx
            filename = filenames[global_idx]
            gt = load_rgb_tensor(gt_map[filename])
            frame_psnr = calculate_psnr(chunk_out[local_idx], gt, crop_border=crop_border)
            frame_ssim = calculate_ssim(chunk_out[local_idx], gt, crop_border=crop_border)
            total_psnr += frame_psnr
            total_ssim += frame_ssim
            count += 1
            frame_rows.append({
                "sequence": os.path.basename(lr_dir),
                "frame": filename,
                "psnr": frame_psnr,
                "ssim": frame_ssim,
            })

        del chunk, chunk_out
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if end == total_frames:
            break

    if count != total_frames:
        raise RuntimeError(f"{lr_dir}: evaluated {count} frames, expected {total_frames}")

    return total_psnr / count, total_ssim / count, frame_rows


def write_csv(path: str, fieldnames, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: str, overall: dict, sequence_rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("| Split | Sequences | Frames | PSNR | SSIM |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        f.write(
            f"| val | {overall['sequences']} | {overall['frames']} | "
            f"{overall['psnr']:.4f} | {overall['ssim']:.4f} |\n\n"
        )
        f.write("| Sequence | Frames | PSNR | SSIM |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in sequence_rows:
            f.write(
                f"| {row['sequence']} | {row['frames']} | "
                f"{row['psnr']:.4f} | {row['ssim']:.4f} |\n"
            )


def write_metric_outputs(output_dir: str, overall: dict, sequence_rows, frame_rows):
    ensure_dir(output_dir)
    write_csv(
        os.path.join(output_dir, "summary.csv"),
        ["sequences", "frames", "psnr", "ssim"],
        [overall],
    )
    write_csv(
        os.path.join(output_dir, "per_sequence.csv"),
        ["sequence", "frames", "psnr", "ssim"],
        sequence_rows,
    )
    write_csv(
        os.path.join(output_dir, "per_frame.csv"),
        ["sequence", "frame", "psnr", "ssim"],
        frame_rows,
    )
    write_summary_md(os.path.join(output_dir, "summary.md"), overall, sequence_rows)


def load_existing_metrics(output_dir: str):
    seq_path = os.path.join(output_dir, "per_sequence.csv")
    frame_path = os.path.join(output_dir, "per_frame.csv")
    sequence_rows = []
    frame_rows = []

    if os.path.isfile(seq_path):
        with open(seq_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sequence_rows.append({
                    "sequence": row["sequence"],
                    "frames": int(row["frames"]),
                    "psnr": float(row["psnr"]),
                    "ssim": float(row["ssim"]),
                })

    if os.path.isfile(frame_path):
        with open(frame_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                frame_rows.append({
                    "sequence": row["sequence"],
                    "frame": row["frame"],
                    "psnr": float(row["psnr"]),
                    "ssim": float(row["ssim"]),
                })

    completed = {row["sequence"] for row in sequence_rows}
    return sequence_rows, frame_rows, completed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate current BasicVSR++ checkpoint on the full val set.")
    parser.add_argument(
        "--lr-root",
        type=str,
        default="/home/fc/Coding/CV/data/val/val_sharp_bicubic/X4",
        help="Root directory of LR validation sequences.",
    )
    parser.add_argument(
        "--gt-root",
        type=str,
        default="/home/fc/Coding/CV/data/val/val_sharp",
        help="Root directory of GT validation sequences.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/fc/Coding/CV/part2_1/mmagic/work_dirs/basicvsr-pp_c64n7_fc_finetune/basicvsr-pp_c64n7_fc_finetune/best_PSNR_iter_20000.pth",
        help="BasicVSR++ generator checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/fc/Coding/CV/part2_1/results/metrics_val_basicvsrpp_current",
        help="Output metrics directory.",
    )
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--crop-border", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-seqs", type=int, default=0, help="Debug option: evaluate only first N sequences.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing per_sequence.csv/per_frame.csv.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    seq_names = sorted([
        name for name in os.listdir(args.lr_root)
        if os.path.isdir(os.path.join(args.lr_root, name))
    ])
    if args.max_seqs > 0:
        seq_names = seq_names[: args.max_seqs]
    if not seq_names:
        raise ValueError(f"No validation sequences found in: {args.lr_root}")

    print("=" * 72)
    print("BasicVSR++ Full Val PSNR Evaluation")
    print("=" * 72)
    print(f"LR root     : {args.lr_root}")
    print(f"GT root     : {args.gt_root}")
    print(f"Checkpoint  : {args.ckpt}")
    print(f"Output      : {args.output}")
    print(f"Device      : {device}")
    print(f"Sequences   : {len(seq_names)}")
    print(f"Chunk       : size={args.chunk_size}, overlap={args.chunk_overlap}")
    print(f"Crop border : {args.crop_border}")
    print("=" * 72)

    model = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=7, cpu_cache_length=30)
    load_basicvsrpp_generator_checkpoint(model, args.ckpt)
    model = model.to(device).eval()

    if args.resume:
        sequence_rows, frame_rows, completed = load_existing_metrics(args.output)
        print(f"Resume enabled: found {len(completed)} completed sequence(s).")
    else:
        sequence_rows = []
        frame_rows = []
        completed = set()

    total_psnr = sum(row["psnr"] * row["frames"] for row in sequence_rows)
    total_ssim = sum(row["ssim"] * row["frames"] for row in sequence_rows)
    total_frames = sum(row["frames"] for row in sequence_rows)

    for seq_name in tqdm(seq_names, desc="Sequences"):
        if seq_name in completed:
            print(f"{seq_name}: skip existing metrics", flush=True)
            continue

        lr_dir = os.path.join(args.lr_root, seq_name)
        gt_dir = os.path.join(args.gt_root, seq_name)
        if not os.path.isdir(gt_dir):
            raise FileNotFoundError(f"GT sequence not found: {gt_dir}")

        seq_psnr, seq_ssim, seq_frame_rows = evaluate_sequence(
            model=model,
            lr_dir=lr_dir,
            gt_dir=gt_dir,
            device=device,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            crop_border=args.crop_border,
        )
        seq_frames = len(seq_frame_rows)
        sequence_rows.append({
            "sequence": seq_name,
            "frames": seq_frames,
            "psnr": seq_psnr,
            "ssim": seq_ssim,
        })
        frame_rows.extend(seq_frame_rows)
        total_psnr += seq_psnr * seq_frames
        total_ssim += seq_ssim * seq_frames
        total_frames += seq_frames
        print(
            f"{seq_name}: frames={seq_frames}, PSNR={seq_psnr:.4f}, SSIM={seq_ssim:.4f}",
            flush=True,
        )

        current_overall = {
            "sequences": len(sequence_rows),
            "frames": total_frames,
            "psnr": total_psnr / total_frames,
            "ssim": total_ssim / total_frames,
        }
        write_metric_outputs(args.output, current_overall, sequence_rows, frame_rows)

    overall = {
        "sequences": len(sequence_rows),
        "frames": total_frames,
        "psnr": total_psnr / total_frames,
        "ssim": total_ssim / total_frames,
    }

    write_metric_outputs(args.output, overall, sequence_rows, frame_rows)

    print("| Split | Sequences | Frames | PSNR | SSIM |")
    print("|---|---:|---:|---:|---:|")
    print(
        f"| val | {overall['sequences']} | {overall['frames']} | "
        f"{overall['psnr']:.4f} | {overall['ssim']:.4f} |"
    )
    print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
