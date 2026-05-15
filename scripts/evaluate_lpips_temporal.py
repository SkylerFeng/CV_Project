import argparse
import csv
import os
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def strip_suffix(stem: str, suffix: str) -> str:
    if suffix and stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def build_image_map(root: str, suffix: str = ""):
    root_path = Path(root).resolve()
    image_map = {}
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or not is_image_file(path):
            continue
        rel = path.relative_to(root_path).with_suffix("")
        parts = list(rel.parts)
        parts[-1] = strip_suffix(parts[-1], suffix)
        image_map["/".join(parts)] = str(path)
    return image_map


def load_tensor(path: str, size_hw=None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = to_tensor(image).unsqueeze(0).clamp(0, 1)
    if size_hw is not None and tensor.shape[-2:] != size_hw:
        tensor = F.interpolate(tensor, size=size_hw, mode="bicubic", align_corners=False).clamp(0, 1)
    return tensor


def lpips_input(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def temporal_delta(x_prev: torch.Tensor, x_cur: torch.Tensor) -> torch.Tensor:
    # Map frame differences from [-1, 1] back into [0, 1], then LPIPS expects [-1, 1].
    return ((x_cur - x_prev) + 1.0) * 0.5


def write_csv(path: str, fieldnames, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_method(name, pred_dir, pred_suffix, gt_map, loss_fn, device):
    pred_map = build_image_map(pred_dir, suffix=pred_suffix)
    keys = sorted(set(gt_map) & set(pred_map))
    if not keys:
        raise ValueError(f"No matched frames for method={name}, pred_dir={pred_dir}")

    rows = []
    total_lpips = 0.0
    total_temporal_mse = 0.0
    total_tlpips = 0.0
    temporal_count = 0

    prev_gt = None
    prev_pred = None
    for key in tqdm(keys, desc=name):
        gt = load_tensor(gt_map[key]).to(device)
        pred = load_tensor(pred_map[key], size_hw=gt.shape[-2:]).to(device)

        frame_lpips = float(loss_fn(lpips_input(pred), lpips_input(gt)).item())
        total_lpips += frame_lpips

        frame_temporal_mse = ""
        frame_tlpips = ""
        if prev_gt is not None and prev_pred is not None:
            gt_delta = temporal_delta(prev_gt, gt)
            pred_delta = temporal_delta(prev_pred, pred)
            frame_temporal_mse = float(torch.mean((pred_delta - gt_delta) ** 2).item())
            frame_tlpips = float(loss_fn(lpips_input(pred_delta), lpips_input(gt_delta)).item())
            total_temporal_mse += frame_temporal_mse
            total_tlpips += frame_tlpips
            temporal_count += 1

        rows.append({
            "method": name,
            "frame": key,
            "lpips": frame_lpips,
            "temporal_mse": frame_temporal_mse,
            "tlpips_delta": frame_tlpips,
        })

        prev_gt = gt
        prev_pred = pred

    summary = {
        "method": name,
        "frames": len(keys),
        "lpips": total_lpips / len(keys),
        "temporal_pairs": temporal_count,
        "temporal_mse": total_temporal_mse / temporal_count if temporal_count else 0.0,
        "tlpips_delta": total_tlpips / temporal_count if temporal_count else 0.0,
    }
    return summary, rows


def parse_method(text: str):
    parts = text.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--method must use name=dir format")
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate LPIPS and simple temporal consistency metrics.")
    parser.add_argument("--gt", required=True, help="GT frame directory.")
    parser.add_argument("--method", action="append", required=True, help="name=prediction_dir. Can be repeated.")
    parser.add_argument("--output", required=True, help="Output metrics directory.")
    parser.add_argument("--pred-suffix", default="", help="Suffix stripped from prediction filenames.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    gt_map = build_image_map(args.gt)
    loss_fn = lpips.LPIPS(net=args.net).to(device).eval()

    summaries = []
    all_rows = []
    for method_text in args.method:
        name, pred_dir = parse_method(method_text)
        summary, rows = evaluate_method(name, pred_dir, args.pred_suffix, gt_map, loss_fn, device)
        summaries.append(summary)
        all_rows.extend(rows)

    ensure_dir(args.output)
    write_csv(
        os.path.join(args.output, "summary.csv"),
        ["method", "frames", "lpips", "temporal_pairs", "temporal_mse", "tlpips_delta"],
        summaries,
    )
    write_csv(
        os.path.join(args.output, "per_frame.csv"),
        ["method", "frame", "lpips", "temporal_mse", "tlpips_delta"],
        all_rows,
    )
    with open(os.path.join(args.output, "summary.md"), "w", encoding="utf-8") as f:
        f.write("| Method | Frames | LPIPS ↓ | tMSE ↓ | tLPIPS-delta ↓ |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for item in summaries:
            f.write(
                f"| {item['method']} | {item['frames']} | {item['lpips']:.4f} | "
                f"{item['temporal_mse']:.6f} | {item['tlpips_delta']:.4f} |\n"
            )

    print("| Method | Frames | LPIPS ↓ | tMSE ↓ | tLPIPS-delta ↓ |")
    print("|---|---:|---:|---:|---:|")
    for item in summaries:
        print(
            f"| {item['method']} | {item['frames']} | {item['lpips']:.4f} | "
            f"{item['temporal_mse']:.6f} | {item['tlpips_delta']:.4f} |"
        )
    print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
