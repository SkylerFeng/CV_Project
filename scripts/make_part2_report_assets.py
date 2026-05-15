import argparse
import csv
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


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
    ensure_dir(os.path.dirname(output))
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(output)


def read_basic_summary(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return {
        "method": "Fine-tuned BasicVSR++",
        "sequences": int(row["sequences"]),
        "frames": int(row["frames"]),
        "psnr": float(row["psnr"]),
        "ssim": float(row["ssim"]),
    }


def read_real_summary(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return {
        "method": "Fine-tuned Real-ESRGAN",
        "sequences": int(row["sequences"]),
        "frames": int(row["frames"]),
        "psnr": float(row["psnr"]),
        "ssim": float(row["ssim"]),
    }


def write_summary(rows, output_dir: str):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "part2_finetuned_full_val_summary.csv")
    md_path = os.path.join(output_dir, "part2_finetuned_full_val_summary.md")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "sequences", "frames", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Method | Sequences | Frames | PSNR | SSIM |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['sequences']} | {row['frames']} | "
                f"{row['psnr']:.4f} | {row['ssim']:.4f} |\n"
            )
    print(f"Saved metrics summary to: {csv_path}")
    print(f"Saved metrics summary to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Create Part 2 report assets.")
    parser.add_argument("--frame", default="00000049.png")
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--basic-dir", default="part2_1/results/report_val000_basicvsrpp_finetuned/frames")
    parser.add_argument("--real-dir", default="part2_2/results/report_val000_realesrgan_finetuned")
    parser.add_argument("--gt-dir", default="data/val/val_sharp/000")
    parser.add_argument("--basic-summary", default="part2_1/results/metrics_val_basicvsrpp_finetuned_full/summary.csv")
    parser.add_argument("--real-summary", default="part2_2/results/metrics_val_realesrgan_finetuned_full/summary.csv")
    args = parser.parse_args()

    frame = args.frame
    paths = [
        (os.path.join(args.basic_dir, frame), "BasicVSR++ FT"),
        (os.path.join(args.real_dir, frame), "Real-ESRGAN FT"),
        (os.path.join(args.gt_dir, frame), "GT"),
    ]
    for path, _ in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    panels = [label_panel(load_rgb(path), label, args.height) for path, label in paths]
    out_path = os.path.join(args.output_dir, "part2_basicvsrpp_vs_realesrgan_finetuned.png")
    hstack(panels, out_path)
    print(f"Saved Part 2 report figure to: {out_path}")

    rows = [
        read_basic_summary(args.basic_summary),
        read_real_summary(args.real_summary),
    ]
    write_summary(rows, args.output_dir)


if __name__ == "__main__":
    main()
