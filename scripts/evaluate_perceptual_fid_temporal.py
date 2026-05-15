import argparse
import csv
import math
import os
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def is_image(path: Path):
    return path.suffix.lower() in IMG_EXTS


def build_map(root: Path):
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and is_image(path):
            out["/".join(path.relative_to(root).with_suffix("").parts)] = path
    return out


def load_tensor(path: Path, size_hw=None):
    img = Image.open(path).convert("RGB")
    t = to_tensor(img).unsqueeze(0).clamp(0, 1)
    if size_hw is not None and t.shape[-2:] != size_hw:
        t = F.interpolate(t, size=size_hw, mode="bicubic", align_corners=False).clamp(0, 1)
    return t


def lpips_input(x):
    return x * 2.0 - 1.0


def temporal_delta(prev, cur):
    return ((cur - prev) + 1.0) * 0.5


class InceptionFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / x.new_tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)
        return self.model(x)


def frechet_distance(mu1, sigma1, mu2, sigma2):
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def stats(features):
    feats = np.asarray(features, dtype=np.float64)
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


def write_csv(path, fieldnames, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_method(name, pred_root, gt_map, lpips_fn, inception_fn, gt_features, device):
    pred_map = build_map(pred_root)
    keys = sorted(set(gt_map) & set(pred_map))
    if not keys:
        raise ValueError(f"No matched frames for {name}: {pred_root}")

    rows = []
    pred_features = []
    total_lpips = 0.0
    total_temporal_mse = 0.0
    total_tlpips = 0.0
    temporal_count = 0
    prev_gt = None
    prev_pred = None

    for key in tqdm(keys, desc=name):
        gt = load_tensor(gt_map[key]).to(device)
        pred = load_tensor(pred_map[key], size_hw=gt.shape[-2:]).to(device)

        frame_lpips = float(lpips_fn(lpips_input(pred), lpips_input(gt)).item())
        total_lpips += frame_lpips

        pred_features.append(inception_fn(pred).detach().cpu().numpy()[0])

        frame_tmse = ""
        frame_tlpips = ""
        if prev_gt is not None:
            gt_delta = temporal_delta(prev_gt, gt)
            pred_delta = temporal_delta(prev_pred, pred)
            frame_tmse = float(torch.mean((pred_delta - gt_delta) ** 2).item())
            frame_tlpips = float(lpips_fn(lpips_input(pred_delta), lpips_input(gt_delta)).item())
            total_temporal_mse += frame_tmse
            total_tlpips += frame_tlpips
            temporal_count += 1

        rows.append({
            "method": name,
            "frame": key,
            "lpips": frame_lpips,
            "temporal_mse": frame_tmse,
            "tlpips_delta": frame_tlpips,
        })
        prev_gt = gt
        prev_pred = pred

    mu_gt, sig_gt = stats(gt_features)
    mu_pred, sig_pred = stats(pred_features)
    fid = frechet_distance(mu_gt, sig_gt, mu_pred, sig_pred)
    summary = {
        "method": name,
        "frames": len(keys),
        "lpips": total_lpips / len(keys),
        "fid": fid,
        "temporal_pairs": temporal_count,
        "temporal_mse": total_temporal_mse / temporal_count if temporal_count else 0.0,
        "tlpips_delta": total_tlpips / temporal_count if temporal_count else 0.0,
    }
    return summary, rows


def parse_method(text):
    name, path = text.split("=", 1)
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LPIPS, FID, and tLPIPS-style proxy.")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    gt_map = build_map(Path(args.gt))
    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    inception_fn = InceptionFeatures().to(device).eval()

    gt_features = []
    for key in tqdm(sorted(gt_map), desc="GT Inception"):
        gt = load_tensor(gt_map[key]).to(device)
        gt_features.append(inception_fn(gt).detach().cpu().numpy()[0])

    summaries = []
    all_rows = []
    for method_text in args.method:
        name, pred_root = parse_method(method_text)
        summary, rows = evaluate_method(name, pred_root, gt_map, lpips_fn, inception_fn, gt_features, device)
        summaries.append(summary)
        all_rows.extend(rows)

    ensure_dir(args.output)
    write_csv(
        os.path.join(args.output, "summary.csv"),
        ["method", "frames", "lpips", "fid", "temporal_pairs", "temporal_mse", "tlpips_delta"],
        summaries,
    )
    write_csv(
        os.path.join(args.output, "per_frame.csv"),
        ["method", "frame", "lpips", "temporal_mse", "tlpips_delta"],
        all_rows,
    )
    with open(os.path.join(args.output, "summary.md"), "w", encoding="utf-8") as f:
        f.write("| Method | Frames | LPIPS ↓ | FID ↓ | tMSE ↓ | tLPIPS-proxy ↓ |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for item in summaries:
            f.write(
                f"| {item['method']} | {item['frames']} | {item['lpips']:.4f} | "
                f"{item['fid']:.2f} | {item['temporal_mse']:.6f} | {item['tlpips_delta']:.4f} |\n"
            )

    print("| Method | Frames | LPIPS ↓ | FID ↓ | tMSE ↓ | tLPIPS-proxy ↓ |")
    print("|---|---:|---:|---:|---:|---:|")
    for item in summaries:
        print(
            f"| {item['method']} | {item['frames']} | {item['lpips']:.4f} | "
            f"{item['fid']:.2f} | {item['temporal_mse']:.6f} | {item['tlpips_delta']:.4f} |"
        )
    print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
