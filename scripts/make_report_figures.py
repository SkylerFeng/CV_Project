import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_to_height(img: Image.Image, height: int) -> Image.Image:
    if img.height == height:
        return img
    width = round(img.width * height / img.height)
    return img.resize((width, height), Image.BICUBIC)


def draw_label(img: Image.Image, label: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    pad = 8
    box_h = 24
    draw.rectangle((0, 0, out.width, box_h), fill=(0, 0, 0))
    draw.text((pad, 7), label, fill=(255, 255, 255), font=font)
    return out


def hstack(items, labels, out_path: str, height: int = 360):
    imgs = [draw_label(resize_to_height(load_rgb(path), height), label) for path, label in zip(items, labels)]
    total_w = sum(img.width for img in imgs)
    canvas = Image.new("RGB", (total_w, height), (255, 255, 255))
    x = 0
    for img in imgs:
        canvas.paste(img, (x, 0))
        x += img.width
    ensure_dir(os.path.dirname(out_path))
    canvas.save(out_path)


def crop_columns(grid_path: str, labels, out_path: str):
    grid = load_rgb(grid_path)
    col_w = grid.width // len(labels)
    imgs = []
    for idx, label in enumerate(labels):
        crop = grid.crop((idx * col_w, 0, (idx + 1) * col_w, grid.height))
        imgs.append(draw_label(crop, label))
    canvas = Image.new("RGB", (grid.width, grid.height), (255, 255, 255))
    for idx, img in enumerate(imgs):
        canvas.paste(img, (idx * col_w, 0))
    ensure_dir(os.path.dirname(out_path))
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Create report-ready comparison figures.")
    parser.add_argument("--frame", default="00000049.png")
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()

    ensure_dir(args.output)

    # Part 1: reuse the saved LRup / Bicubic / Lanczos / SRCNN / HR grid.
    part1_grid = "part1/outputs/vis/00000000_LRup_BI_LZ_SR_HR.png"
    if os.path.isfile(part1_grid):
        crop_columns(
            part1_grid,
            ["LR up", "Bicubic", "Lanczos", "SRCNN", "GT"],
            os.path.join(args.output, "part1_baseline_comparison.png"),
        )

    # Part 2: BasicVSR++ vs Real-ESRGAN on val011.
    frame = args.frame
    basic = os.path.join("part2_1/results/val011_basicvsrpp_x4/frames", frame)
    real = os.path.join("part2_2/results/val011_official_x4plus", frame.replace(".png", "_out.png"))
    gt = os.path.join("data/val/val_sharp/011", frame)
    if all(os.path.isfile(p) for p in [basic, real, gt]):
        hstack(
            [basic, real, gt],
            ["BasicVSR++", "Real-ESRGAN", "GT"],
            os.path.join(args.output, "part2_basicvsrpp_vs_realesrgan.png"),
            height=360,
        )

    # Part 3: mask + Basic + RealESRGAN + Hybrid + GT.
    base = "part3/results/adaptive_hybrid_000_directionc_official"
    basic = os.path.join(base, "basic_frames", frame)
    real = os.path.join(base, "generative_frames", frame)
    mask = os.path.join(base, "masks", frame)
    hybrid = os.path.join(base, "frames", frame)
    gt = os.path.join("data/val/val_sharp/000", frame)
    if all(os.path.isfile(p) for p in [basic, real, mask, hybrid, gt]):
        hstack(
            [basic, real, mask, hybrid, gt],
            ["BasicVSR++", "Real-ESRGAN", "Mask", "Hybrid", "GT"],
            os.path.join(args.output, "part3_mask_basic_real_hybrid.png"),
            height=360,
        )

    print(f"Saved report figures to: {args.output}")


if __name__ == "__main__":
    main()
