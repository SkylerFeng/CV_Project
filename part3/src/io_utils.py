import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMG_EXTS


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Image folder not found: {folder}")
    paths = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if is_image_file(name)
    ]
    return paths


def normalized_key(path: str, suffix: str = "") -> str:
    stem = Path(path).stem
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def build_image_map(folder: str, suffix: str = "") -> Dict[str, str]:
    image_map = {}
    for path in list_images(folder):
        key = normalized_key(path, suffix=suffix)
        if key.startswith("debug_"):
            continue
        image_map[key] = path
    return image_map


def paired_image_paths(
    basic_dir: str,
    generative_dir: str,
    generative_suffix: str = "",
    max_frames: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    basic_map = build_image_map(basic_dir)
    gen_map = build_image_map(generative_dir, suffix=generative_suffix)
    keys = sorted(set(basic_map) & set(gen_map))
    if max_frames is not None:
        keys = keys[:max_frames]
    if not keys:
        raise ValueError(
            "No matching frames found between basic and generative folders.\n"
            f"basic_dir={basic_dir}\n"
            f"generative_dir={generative_dir}"
        )
    return [(key, basic_map[key], gen_map[key]) for key in keys]


def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def extract_panel(image: Image.Image, panel_index: int, panel_count: int) -> Image.Image:
    if panel_count <= 1:
        return image.copy()
    width, height = image.size
    panel_width = width // panel_count
    left = panel_width * panel_index
    right = left + panel_width
    if panel_index < 0 or panel_index >= panel_count:
        raise ValueError(f"panel_index={panel_index} out of range for panel_count={panel_count}")
    return image.crop((left, 0, right, height))


def resize_like(image: Image.Image, reference: Image.Image) -> Image.Image:
    if image.size == reference.size:
        return image
    return image.resize(reference.size, Image.BICUBIC)


def save_rgb(image: Image.Image, path: str):
    ensure_dir(os.path.dirname(path))
    image.save(path)


def iter_previous_current_next(items: List[Tuple[str, str, str]]):
    for idx, item in enumerate(items):
        prev_item = items[idx - 1] if idx > 0 else None
        next_item = items[idx + 1] if idx + 1 < len(items) else None
        yield prev_item, item, next_item

