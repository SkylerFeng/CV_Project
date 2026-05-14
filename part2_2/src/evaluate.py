import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def strip_suffix(stem: str, suffix: str) -> str:
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def build_image_map(root: str, suffix: str = "") -> Dict[str, str]:
    root_path = Path(root).resolve()
    if root_path.is_file():
        stem = strip_suffix(root_path.stem, suffix)
        return {stem: str(root_path)}

    if not root_path.is_dir():
        raise FileNotFoundError(f"Image path not found: {root}")

    image_map = {}
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or not is_image_file(str(path)):
            continue
        rel = path.relative_to(root_path)
        key_path = rel.with_suffix("")
        key_parts = list(key_path.parts)
        key_parts[-1] = strip_suffix(key_parts[-1], suffix)
        key = "/".join(key_parts)
        if key in image_map:
            raise ValueError(f"Duplicate frame key after suffix stripping: {key}")
        image_map[key] = str(path)
    return image_map


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_like(pred: Image.Image, gt: Image.Image) -> Image.Image:
    if pred.size == gt.size:
        return pred
    return pred.resize(gt.size, Image.BICUBIC)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return to_tensor(image).unsqueeze(0).clamp(0, 1)


def crop_border_tensor(x: torch.Tensor, crop_border: int) -> torch.Tensor:
    if crop_border <= 0:
        return x
    _, _, h, w = x.shape
    if crop_border * 2 >= h or crop_border * 2 >= w:
        raise ValueError(
            f"crop_border={crop_border} is too large for image size {w}x{h}"
        )
    return x[..., crop_border:-crop_border, crop_border:-crop_border]


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d.view(1, 1, window_size, window_size)


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
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
    sigma_pred_target = (
        F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_pred_target
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = (
        (2 * mu_pred_target + c1)
        * (2 * sigma_pred_target + c2)
        / ((mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2))
    )
    return float(ssim_map.mean().item())


def write_csv(path: str, fieldnames: List[str], rows: List[dict]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, summary: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("| Frames | PSNR | SSIM |\n")
        f.write("|---:|---:|---:|\n")
        f.write(
            f"| {summary['frames']} | {summary['psnr']:.4f} | "
            f"{summary['ssim']:.4f} |\n"
        )


def evaluate(pred_dir: str, gt_dir: str, pred_suffix: str, crop_border: int, resize: bool):
    pred_map = build_image_map(pred_dir, suffix=pred_suffix)
    gt_map = build_image_map(gt_dir)
    keys = sorted(set(pred_map) & set(gt_map))

    if not keys:
        pred_examples = sorted(pred_map)[:5]
        gt_examples = sorted(gt_map)[:5]
        raise ValueError(
            "No matching images found.\n"
            f"pred examples: {pred_examples}\n"
            f"gt examples  : {gt_examples}\n"
            f"Try setting --pred-suffix, e.g. --pred-suffix _out"
        )

    rows = []
    total_psnr = 0.0
    total_ssim = 0.0

    for key in keys:
        gt = load_rgb(gt_map[key])
        pred = load_rgb(pred_map[key])

        if pred.size != gt.size:
            if not resize:
                raise ValueError(
                    f"Size mismatch for {key}: pred={pred.size}, gt={gt.size}. "
                    "Use default resize behavior or remove --no-resize."
                )
            pred = resize_like(pred, gt)

        gt_t = crop_border_tensor(image_to_tensor(gt), crop_border)
        pred_t = crop_border_tensor(image_to_tensor(pred), crop_border)

        frame_psnr = calculate_psnr(pred_t, gt_t)
        frame_ssim = calculate_ssim(pred_t, gt_t)

        rows.append({
            "frame": key,
            "psnr": frame_psnr,
            "ssim": frame_ssim,
            "pred_path": pred_map[key],
            "gt_path": gt_map[key],
        })
        total_psnr += frame_psnr
        total_ssim += frame_ssim

    summary = {
        "frames": len(keys),
        "psnr": total_psnr / len(keys),
        "ssim": total_ssim / len(keys),
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate Part 2.2 outputs with PSNR/SSIM.")
    parser.add_argument("--pred", type=str, required=True, help="Predicted SR image folder or image.")
    parser.add_argument("--gt", type=str, required=True, help="Ground-truth HR image folder or image.")
    parser.add_argument(
        "--output",
        type=str,
        default="part2_2/results/metrics",
        help="Directory for summary.csv, per_frame.csv, and summary.md.",
    )
    parser.add_argument(
        "--pred-suffix",
        type=str,
        default="_out",
        help="Suffix to strip from predicted filenames before matching.",
    )
    parser.add_argument(
        "--crop-border",
        type=int,
        default=4,
        help="Pixels cropped from each border before metric calculation.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Fail on size mismatch instead of resizing prediction to GT size.",
    )
    args = parser.parse_args()

    summary, rows = evaluate(
        pred_dir=args.pred,
        gt_dir=args.gt,
        pred_suffix=args.pred_suffix,
        crop_border=args.crop_border,
        resize=not args.no_resize,
    )

    ensure_dir(args.output)
    write_csv(
        os.path.join(args.output, "summary.csv"),
        ["frames", "psnr", "ssim"],
        [summary],
    )
    write_csv(
        os.path.join(args.output, "per_frame.csv"),
        ["frame", "psnr", "ssim", "pred_path", "gt_path"],
        rows,
    )
    write_markdown(os.path.join(args.output, "summary.md"), summary)

    print("| Frames | PSNR | SSIM |")
    print("|---:|---:|---:|")
    print(f"| {summary['frames']} | {summary['psnr']:.4f} | {summary['ssim']:.4f} |")
    print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
