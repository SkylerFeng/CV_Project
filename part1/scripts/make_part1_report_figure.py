import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont


PART1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PART1_DIR))

from src.model import SRCNN


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    arr = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def load_srcnn(ckpt_path: str, device: torch.device):
    model = SRCNN().to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return model


def weighted_temporal(frame_paths, frame_idx: int, size, weights, apply_unsharp: bool):
    radius = len(weights) // 2
    acc = None
    for offset, weight in zip(range(-radius, radius + 1), weights):
        idx = max(0, min(frame_idx + offset, len(frame_paths) - 1))
        lr = Image.open(frame_paths[idx]).convert("RGB")
        up = lr.resize(size, Image.BICUBIC)
        arr = np.asarray(up).astype(np.float32) * float(weight)
        acc = arr if acc is None else acc + arr
    out = Image.fromarray(acc.clip(0, 255).round().astype(np.uint8), mode="RGB")
    if apply_unsharp:
        out = out.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    return out


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.BICUBIC)


def label_panel(image: Image.Image, label: str, height: int) -> Image.Image:
    image = resize_to_height(image.convert("RGB"), height)
    label_h = 34
    canvas = Image.new("RGB", (image.width, image.height + label_h), (255, 255, 255))
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, label_h), fill=(20, 20, 20))
    draw.text((8, 11), label, fill=(255, 255, 255), font=font)
    return canvas


def hstack(panels, output: str):
    total_w = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (total_w, height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    ensure_dir(os.path.dirname(output))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description="Create a report-ready Part 1 baseline comparison figure.")
    parser.add_argument("--lr-dir", default="data/val/val_sharp_bicubic/X4/000")
    parser.add_argument("--gt-dir", default="data/val/val_sharp/000")
    parser.add_argument("--frame", default="00000049.png")
    parser.add_argument("--ckpt", default="part1/checkpoints/srcnn_x4_epoch20.pth")
    parser.add_argument("--output", default="figures/part1_baseline_comparison.png")
    parser.add_argument("--height", type=int, default=220)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    lr_path = os.path.join(args.lr_dir, args.frame)
    gt_path = os.path.join(args.gt_dir, args.frame)
    if not os.path.isfile(lr_path):
        raise FileNotFoundError(lr_path)
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(gt_path)

    lr = Image.open(lr_path).convert("RGB")
    gt = Image.open(gt_path).convert("RGB")
    bicubic = lr.resize(gt.size, Image.BICUBIC)
    lanczos = lr.resize(gt.size, Image.LANCZOS)

    names = sorted([
        name for name in os.listdir(args.lr_dir)
        if os.path.splitext(name.lower())[1] in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ])
    frame_idx = names.index(args.frame)
    frame_paths = [os.path.join(args.lr_dir, name) for name in names]
    temporal = weighted_temporal(frame_paths, frame_idx, gt.size, [0.25, 0.5, 0.25], True)

    model = load_srcnn(args.ckpt, device)
    with torch.no_grad():
        x = pil_to_tensor(bicubic).unsqueeze(0).to(device)
        srcnn = tensor_to_pil(model(x).squeeze(0))

    lr_up = lr.resize(gt.size, Image.NEAREST)
    panels = [
        label_panel(lr_up, "LR up", args.height),
        label_panel(bicubic, "Bicubic", args.height),
        label_panel(lanczos, "Lanczos", args.height),
        label_panel(srcnn, "SRCNN", args.height),
        label_panel(temporal, "Temporal Avg.", args.height),
        label_panel(gt, "GT", args.height),
    ]
    hstack(panels, args.output)
    print(f"Saved Part 1 report figure to: {args.output}")


if __name__ == "__main__":
    main()
