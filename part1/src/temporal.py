import os
from PIL import Image, ImageFilter

from .utils import ensure_dir


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _list_frames(folder: str):
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    ])


def _clamp_index(idx: int, low: int, high: int) -> int:
    return max(low, min(idx, high))


def _default_weights(radius: int):
    """
    radius=1 -> [0.25, 0.5, 0.25]
    radius=2 -> [0.1, 0.2, 0.4, 0.2, 0.1]
    """
    if radius == 1:
        return [0.25, 0.5, 0.25]
    if radius == 2:
        return [0.1, 0.2, 0.4, 0.2, 0.1]

    n = 2 * radius + 1
    weights = [1.0] * n
    weights[radius] = 2.0
    s = sum(weights)
    return [w / s for w in weights]


def _normalize_weights(weights):
    s = sum(weights)
    if s <= 0:
        raise ValueError("Sum of weights must be positive")
    return [w / s for w in weights]


def _weighted_average_pil(images, weights):
    """
    images: list of PIL.Image with same size
    weights: normalized weights
    """
    if len(images) != len(weights):
        raise ValueError("images and weights must have the same length")
    if len(images) == 0:
        raise ValueError("images must not be empty")

    base = images[0].convert("RGB")
    acc = Image.new("RGB", base.size)

    # 用逐步 blend 的方式不方便精确加权，所以转像素更稳
    import numpy as np

    arr = None
    for img, w in zip(images, weights):
        img_arr = np.array(img.convert("RGB")).astype("float32")
        if arr is None:
            arr = img_arr * w
        else:
            arr += img_arr * w

    arr = arr.clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def _unsharp_mask(img: Image.Image, radius: float = 1.0, percent: int = 120, threshold: int = 3):
    """
    PIL built-in unsharp mask
    """
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def run_temporal_baseline(cfg: dict):
    """
    Temporal baseline:
      1. read neighboring LR frames
      2. bicubic upscale each neighbor
      3. weighted average
      4. optional unsharp mask
      5. save output
    """
    scale = int(cfg["scale"])
    tcfg = cfg["temporal"]

    lr_root = tcfg["lr_root"]
    out_root = tcfg["out_dir"]

    radius = int(tcfg.get("radius", 1))
    weights = tcfg.get("weights", None)
    apply_unsharp = bool(tcfg.get("apply_unsharp", False))

    unsharp_radius = float(tcfg.get("unsharp_radius", 1.0))
    unsharp_percent = int(tcfg.get("unsharp_percent", 120))
    unsharp_threshold = int(tcfg.get("unsharp_threshold", 3))

    if weights is None:
        weights = _default_weights(radius)
    else:
        if len(weights) != 2 * radius + 1:
            raise ValueError(
                f"weights length must be {2 * radius + 1} when radius={radius}, "
                f"but got {len(weights)}"
            )
    weights = _normalize_weights(weights)

    lr_root = os.path.join(lr_root, f"X{scale}")
    if not os.path.isdir(lr_root):
        raise ValueError(f"Temporal LR root not found: {lr_root}")

    ensure_dir(out_root)

    video_folders = sorted([
        d for d in os.listdir(lr_root)
        if os.path.isdir(os.path.join(lr_root, d))
    ])

    if len(video_folders) == 0:
        raise ValueError(f"No video folders found in {lr_root}")

    total_videos = len(video_folders)
    print(f"[TemporalBaseline] Found {total_videos} videos in {lr_root}")

    for vid_idx, video in enumerate(video_folders, start=1):
        video_lr_dir = os.path.join(lr_root, video)
        video_out_dir = os.path.join(out_root, video)
        ensure_dir(video_out_dir)

        frames = _list_frames(video_lr_dir)
        if len(frames) == 0:
            print(f"[TemporalBaseline] Skip empty folder: {video_lr_dir}")
            continue

        print(f"[TemporalBaseline] Processing video {vid_idx}/{total_videos}: {video} ({len(frames)} frames)")

        for t in range(len(frames)):
            neighbor_imgs = []

            # 先拿当前帧尺寸，推 HR 尺寸
            center_path = os.path.join(video_lr_dir, frames[t])
            center_lr = Image.open(center_path).convert("RGB")
            lr_w, lr_h = center_lr.size
            hr_size = (lr_w * scale, lr_h * scale)

            for offset in range(-radius, radius + 1):
                ni = _clamp_index(t + offset, 0, len(frames) - 1)
                npath = os.path.join(video_lr_dir, frames[ni])

                lr = Image.open(npath).convert("RGB")
                up = lr.resize(hr_size, resample=Image.BICUBIC)
                neighbor_imgs.append(up)

            fused = _weighted_average_pil(neighbor_imgs, weights)

            if apply_unsharp:
                fused = _unsharp_mask(
                    fused,
                    radius=unsharp_radius,
                    percent=unsharp_percent,
                    threshold=unsharp_threshold
                )

            out_path = os.path.join(video_out_dir, frames[t])
            fused.save(out_path)

        print(f"[TemporalBaseline] Saved to: {video_out_dir}")

    print("[TemporalBaseline] Done.")