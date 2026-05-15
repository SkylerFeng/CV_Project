import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw


PANEL_SIZE = (360, 202)
LABEL_H = 32
PAD = 10
COLS = 4


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit(image: Image.Image, size=PANEL_SIZE) -> Image.Image:
    return image.convert("RGB").resize(size, Image.BICUBIC)


def save_asset(image: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def add_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str):
    draw.rectangle((x, y, x + PANEL_SIZE[0], y + LABEL_H), fill=(255, 255, 255))
    draw.text((x + 8, y + 9), text, fill=(20, 20, 20))


def make_showcase(panels):
    rows = (len(panels) + COLS - 1) // COLS
    width = COLS * PANEL_SIZE[0] + (COLS + 1) * PAD
    height = rows * (PANEL_SIZE[1] + LABEL_H) + (rows + 1) * PAD
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for idx, (label, image) in enumerate(panels):
        row = idx // COLS
        col = idx % COLS
        x = PAD + col * (PANEL_SIZE[0] + PAD)
        y = PAD + row * (PANEL_SIZE[1] + LABEL_H + PAD)
        add_label(draw, x, y, label)
        canvas.paste(fit(image), (x, y + LABEL_H))
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Make Direction C showcase image.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--frame", default="00000049")
    parser.add_argument("--generative-label", default="Real-ESRGAN")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    frame = args.frame
    assets_dir = result_dir / "showcase_assets"
    output = Path(args.output) if args.output else result_dir / f"showcase_{frame}.png"

    paths = {
        "BasicVSR++": result_dir / "basic_frames" / f"{frame}.png",
        args.generative_label: result_dir / "generative_frames" / f"{frame}.png",
        "Direction C Hybrid": result_dir / "frames" / f"{frame}.png",
        "Alpha mask": result_dir / "masks" / f"{frame}.png",
        "Structure protect": result_dir / "maps" / "structure_protect" / f"{frame}.png",
        "Uncertain texture": result_dir / "maps" / "uncertain_texture" / f"{frame}.png",
        "Hallucination risk": result_dir / "maps" / "hallucination_risk" / f"{frame}.png",
        "Flicker risk": result_dir / "maps" / "flicker_risk" / f"{frame}.png",
    }

    panels = []
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing panel for {label}: {path}")
        image = load_rgb(str(path))
        asset_name = label.lower().replace("++", "pp").replace(" ", "_").replace("-", "_")
        save_asset(image, str(assets_dir / f"{asset_name}_{frame}.png"))
        panels.append((label, image))

    os.makedirs(output.parent, exist_ok=True)
    make_showcase(panels).save(output)
    print(f"Saved showcase: {output}")
    print(f"Saved assets: {assets_dir}")


if __name__ == "__main__":
    main()
