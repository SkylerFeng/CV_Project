import numpy as np
from PIL import Image, ImageDraw


def blend_with_mask(basic: Image.Image, generative: Image.Image, mask: Image.Image) -> Image.Image:
    basic_arr = np.asarray(basic.convert("RGB")).astype(np.float32)
    gen_arr = np.asarray(generative.convert("RGB")).astype(np.float32)
    alpha = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    fused = basic_arr * (1.0 - alpha) + gen_arr * alpha
    return Image.fromarray(np.clip(fused, 0, 255).round().astype(np.uint8), mode="RGB")


def make_grid(basic: Image.Image, generative: Image.Image, mask: Image.Image, fused: Image.Image) -> Image.Image:
    basic = basic.convert("RGB")
    generative = generative.convert("RGB").resize(basic.size, Image.BICUBIC)
    fused = fused.convert("RGB").resize(basic.size, Image.BICUBIC)
    mask_rgb = mask.convert("RGB").resize(basic.size, Image.BICUBIC)

    labels = ["BasicVSR++", "Generative", "Mask", "Hybrid"]
    images = [basic, generative, mask_rgb, fused]
    w, h = basic.size
    label_h = 32
    canvas = Image.new("RGB", (w * 4, h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(zip(labels, images)):
        canvas.paste(image, (idx * w, label_h))
        draw.text((idx * w + 8, 9), label, fill=(20, 20, 20))
    return canvas

