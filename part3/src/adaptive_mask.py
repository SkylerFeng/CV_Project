from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter


def _rgb_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB")).astype(np.float32) / 255.0


def _gray_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L")).astype(np.float32) / 255.0


def _normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo = float(np.percentile(x, 2))
    hi = float(np.percentile(x, 98))
    return np.clip((x - lo) / (hi - lo + eps), 0.0, 1.0)


def _box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return x
    pad = int(radius)
    padded = np.pad(x, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    size = 2 * pad + 1
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
    gray = _gray_float(image)
    mean = _box_mean(gray, radius)
    mean_sq = _box_mean(gray * gray, radius)
    return _normalize(np.maximum(mean_sq - mean * mean, 0.0))


def flat_color_score(image: Image.Image, radius: int = 5) -> np.ndarray:
    rgb = _rgb_float(image)
    local_vars = []
    for channel in range(3):
        x = rgb[..., channel]
        mean = _box_mean(x, radius)
        mean_sq = _box_mean(x * x, radius)
        local_vars.append(np.maximum(mean_sq - mean * mean, 0.0))
    var = np.mean(local_vars, axis=0)
    return 1.0 - _normalize(var)


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


def branch_disagreement(basic: Image.Image, generative: Image.Image) -> np.ndarray:
    basic_arr = _rgb_float(basic)
    gen_arr = _rgb_float(generative.resize(basic.size, Image.BICUBIC))
    return _normalize(np.mean(np.abs(basic_arr - gen_arr), axis=2))


def estimate_anime_score(image: Image.Image) -> float:
    edges = gradient_strength(image)
    flat = flat_color_score(image)
    texture = local_texture_strength(image)
    edge_density = float((edges > 0.45).mean())
    flat_area = float((flat > 0.70).mean())
    texture_area = float((texture > 0.45).mean())
    score = 0.55 * flat_area + 0.35 * edge_density + 0.10 * (1.0 - texture_area)
    return float(np.clip(score, 0.0, 1.0))


@dataclass
class AdaptiveMaskConfig:
    mode: str = "auto"
    anime_threshold: float = 0.50
    real_max_alpha: float = 0.28
    anime_max_alpha: float = 0.62
    min_alpha: float = 0.0
    texture_gain: float = 1.10
    line_gain: float = 0.85
    flat_gain: float = 0.45
    edge_protect_strength: float = 0.75
    temporal_protect_strength: float = 0.60
    disagreement_protect_strength: float = 0.85
    blur_radius: float = 4.0
    gamma: float = 1.15


def resolve_content_weight(mode: str, anime_score: float, threshold: float) -> Tuple[str, float]:
    if mode == "anime":
        return "anime", 1.0
    if mode == "real":
        return "real", 0.0
    weight = np.clip((anime_score - threshold + 0.20) / 0.40, 0.0, 1.0)
    return ("anime" if weight >= 0.5 else "real"), float(weight)


def build_adaptive_alpha(
    basic: Image.Image,
    generative: Image.Image,
    previous_basic: Optional[Image.Image],
    next_basic: Optional[Image.Image],
    cfg: AdaptiveMaskConfig,
) -> Tuple[Image.Image, Dict[str, Image.Image], Dict[str, float]]:
    generative = generative.resize(basic.size, Image.BICUBIC)
    anime_score = estimate_anime_score(basic)
    content_label, anime_weight = resolve_content_weight(cfg.mode, anime_score, cfg.anime_threshold)

    texture = local_texture_strength(basic)
    edges = gradient_strength(basic)
    flat = flat_color_score(basic)
    temporal = temporal_change_strength(basic, previous_basic, next_basic)
    disagreement = branch_disagreement(basic, generative)

    line_candidate = edges * flat
    texture_candidate = texture * (1.0 - np.clip(edges * cfg.edge_protect_strength, 0.0, 1.0))
    flat_candidate = flat * (1.0 - texture)

    real_prior = cfg.texture_gain * texture_candidate
    anime_prior = (
        cfg.line_gain * line_candidate
        + cfg.flat_gain * flat_candidate
        + 0.35 * cfg.texture_gain * texture_candidate
    )
    alpha = (1.0 - anime_weight) * real_prior + anime_weight * anime_prior

    alpha *= 1.0 - np.clip(cfg.temporal_protect_strength * temporal, 0.0, 1.0)
    alpha *= 1.0 - np.clip(cfg.disagreement_protect_strength * disagreement, 0.0, 1.0)

    max_alpha = (1.0 - anime_weight) * cfg.real_max_alpha + anime_weight * cfg.anime_max_alpha
    alpha = np.clip(alpha, cfg.min_alpha, max_alpha)
    if cfg.gamma > 0:
        alpha = np.power(alpha / max(max_alpha, 1e-6), cfg.gamma) * max_alpha

    mask = Image.fromarray((np.clip(alpha, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    if cfg.blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(cfg.blur_radius))

    maps = {
        "texture": Image.fromarray((texture * 255.0).round().astype(np.uint8), mode="L"),
        "edges": Image.fromarray((edges * 255.0).round().astype(np.uint8), mode="L"),
        "flat": Image.fromarray((flat * 255.0).round().astype(np.uint8), mode="L"),
        "temporal": Image.fromarray((temporal * 255.0).round().astype(np.uint8), mode="L"),
        "disagreement": Image.fromarray((disagreement * 255.0).round().astype(np.uint8), mode="L"),
    }
    stats = {
        "anime_score": anime_score,
        "anime_weight": anime_weight,
        "max_alpha": float(max_alpha),
        "mean_alpha": float(np.asarray(mask).astype(np.float32).mean() / 255.0),
        "content_label": content_label,
    }
    return mask, maps, stats
