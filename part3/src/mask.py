from typing import Optional

import numpy as np
from PIL import Image, ImageFilter


def _gray_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L")).astype(np.float32) / 255.0


def _normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo = float(np.percentile(x, 2))
    hi = float(np.percentile(x, 98))
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)


def _box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return x
    pad = radius
    padded = np.pad(x, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return total / float(size * size)


def gradient_strength(image: Image.Image) -> np.ndarray:
    gray = _gray_float(image)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :]) * 0.5
    return _normalize(np.sqrt(gx * gx + gy * gy))


def local_texture_strength(image: Image.Image, radius: int = 3) -> np.ndarray:
    gray_arr = _gray_float(image)
    mean_arr = _box_mean(gray_arr, radius)
    mean_sq_arr = _box_mean(gray_arr * gray_arr, radius)
    var = np.maximum(mean_sq_arr - mean_arr * mean_arr, 0.0)
    return _normalize(var)


def temporal_change_strength(
    current: Image.Image,
    previous: Optional[Image.Image],
    next_image: Optional[Image.Image],
) -> np.ndarray:
    cur = _gray_float(current)
    diffs = []
    if previous is not None:
        diffs.append(np.abs(cur - _gray_float(previous)))
    if next_image is not None:
        diffs.append(np.abs(cur - _gray_float(next_image)))
    if not diffs:
        return np.zeros_like(cur)
    return _normalize(np.mean(diffs, axis=0))


def build_uncertainty_mask(
    basic: Image.Image,
    previous_basic: Optional[Image.Image] = None,
    next_basic: Optional[Image.Image] = None,
    texture_gain: float = 1.25,
    edge_protect_strength: float = 1.15,
    temporal_protect_strength: float = 0.50,
    min_alpha: float = 0.0,
    max_alpha: float = 0.55,
    blur_radius: float = 5.0,
) -> Image.Image:
    texture = local_texture_strength(basic)
    edges = gradient_strength(basic)
    temporal = temporal_change_strength(basic, previous_basic, next_basic)

    alpha = texture_gain * texture
    alpha *= 1.0 - np.clip(edge_protect_strength * edges, 0.0, 1.0)
    alpha *= 1.0 - np.clip(temporal_protect_strength * temporal, 0.0, 1.0)
    alpha = np.clip(alpha, min_alpha, max_alpha)

    mask = Image.fromarray((alpha * 255.0).round().astype(np.uint8), mode="L")
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    return mask
