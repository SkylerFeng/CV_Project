from typing import Dict

from PIL import Image, ImageDraw


def _fit(image: Image.Image, size) -> Image.Image:
    if image.size == size:
        return image.convert("RGB")
    return image.convert("RGB").resize(size, Image.BICUBIC)


def _label(draw: ImageDraw.ImageDraw, x: int, text: str):
    draw.rectangle((x, 0, x + 260, 28), fill=(255, 255, 255))
    draw.text((x + 8, 8), text, fill=(20, 20, 20))


def make_adaptive_grid(
    lr_up: Image.Image,
    basic: Image.Image,
    generative: Image.Image,
    mask: Image.Image,
    fused: Image.Image,
    maps: Dict[str, Image.Image],
) -> Image.Image:
    size = basic.size
    panels = [
        ("Bicubic/LR up", _fit(lr_up, size)),
        ("BasicVSR++", _fit(basic, size)),
        ("Real-ESRGAN", _fit(generative, size)),
        ("Adaptive alpha", _fit(mask.convert("RGB"), size)),
        ("Hybrid", _fit(fused, size)),
    ]
    for name in ["texture", "edges", "temporal", "disagreement"]:
        if name in maps:
            panels.append((name, _fit(maps[name].convert("RGB"), size)))

    cols = 3
    rows = (len(panels) + cols - 1) // cols
    w, h = size
    label_h = 30
    canvas = Image.new("RGB", (w * cols, (h + label_h) * rows), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (label, image) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = col * w
        y = row * (h + label_h)
        canvas.paste(image, (x, y + label_h))
        _label(draw, x, label)
    return canvas
