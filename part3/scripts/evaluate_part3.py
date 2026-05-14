import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from pytorch_msssim import ssim
from torchvision.transforms.functional import to_tensor


CURRENT_DIR = Path(__file__).resolve().parent
PART3_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PART3_DIR / "src"))

from io_utils import build_image_map, ensure_dir, load_rgb, resize_like


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return to_tensor(image).unsqueeze(0).clamp(0, 1)


def crop_border_tensor(x: torch.Tensor, crop_border: int) -> torch.Tensor:
    if crop_border <= 0:
        return x
    return x[..., crop_border:-crop_border, crop_border:-crop_border]


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def ssim_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(ssim(pred, target, data_range=1.0, size_average=True).item())


def evaluate_method(name: str, method_cfg: dict, gt_map: dict, crop_border: int):
    pred_map = build_image_map(method_cfg["dir"], suffix=method_cfg.get("suffix", ""))
    keys = sorted(set(gt_map) & set(pred_map))
    if not keys:
        raise ValueError(f"No matching frames for method {name}: {method_cfg['dir']}")

    rows = []
    total_psnr = 0.0
    total_ssim = 0.0

    for key in keys:
        gt = load_rgb(gt_map[key])
        pred = resize_like(load_rgb(pred_map[key]), gt)

        gt_t = crop_border_tensor(image_to_tensor(gt), crop_border)
        pred_t = crop_border_tensor(image_to_tensor(pred), crop_border)

        frame_psnr = psnr(pred_t, gt_t)
        frame_ssim = ssim_score(pred_t, gt_t)
        total_psnr += frame_psnr
        total_ssim += frame_ssim

        rows.append({
            "method": name,
            "frame": key,
            "psnr": frame_psnr,
            "ssim": frame_ssim,
        })

    summary = {
        "method": name,
        "frames": len(keys),
        "psnr": total_psnr / len(keys),
        "ssim": total_ssim / len(keys),
    }
    return summary, rows


def write_csv(path: str, fieldnames, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, summaries):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("| Method | Frames | PSNR | SSIM |\n")
        f.write("|---|---:|---:|---:|\n")
        for item in summaries:
            f.write(
                f"| {item['method']} | {item['frames']} | "
                f"{item['psnr']:.4f} | {item['ssim']:.4f} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Part 3 PSNR/SSIM.")
    parser.add_argument("--config", type=str, required=True, help="Path to eval_part3.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gt_map = build_image_map(cfg["paths"]["gt_dir"])
    out_dir = cfg["paths"]["output_dir"]
    crop_border = int(cfg.get("metrics", {}).get("crop_border", 0))

    summaries = []
    all_rows = []
    for name, method_cfg in cfg["methods"].items():
        summary, rows = evaluate_method(name, method_cfg, gt_map, crop_border)
        summaries.append(summary)
        all_rows.extend(rows)

    write_csv(
        os.path.join(out_dir, "summary.csv"),
        ["method", "frames", "psnr", "ssim"],
        summaries,
    )
    write_csv(
        os.path.join(out_dir, "per_frame.csv"),
        ["method", "frame", "psnr", "ssim"],
        all_rows,
    )
    write_markdown(os.path.join(out_dir, "summary.md"), summaries)

    print("| Method | Frames | PSNR | SSIM |")
    print("|---|---:|---:|---:|")
    for item in summaries:
        print(
            f"| {item['method']} | {item['frames']} | "
            f"{item['psnr']:.4f} | {item['ssim']:.4f} |"
        )
    print(f"Saved metrics to: {out_dir}")


if __name__ == "__main__":
    main()

