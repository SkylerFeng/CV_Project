import os
import random
import yaml
import math
import torch
import numpy as np
from PIL import Image


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_images_recursive(root: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = []
    for dp, _, fnames in os.walk(root):
        for fn in fnames:
            if os.path.splitext(fn.lower())[1] in exts:
                files.append(os.path.join(dp, fn))
    return sorted(files)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0  # HWC, [0,1]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = arr.transpose(2, 0, 1)  # CHW
    return torch.from_numpy(arr)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().clamp(0, 1).cpu().numpy()
    t = (t * 255.0 + 0.5).astype(np.uint8)
    t = t.transpose(1, 2, 0)  # HWC
    return Image.fromarray(t)


def make_lr_bicubic(hr: Image.Image, scale: int):
    """HR -> LR (down) using bicubic"""
    w, h = hr.size
    lw, lh = w // scale, h // scale
    lr = hr.resize((lw, lh), resample=Image.BICUBIC)
    return lr


def upsample_bicubic(lr: Image.Image, size_wh):
    """LR -> upsampled to HR size (classic SRCNN pipeline)"""
    return lr.resize(size_wh, resample=Image.BICUBIC)


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    """pred/target: (C,H,W) or (N,C,H,W) in [0,1]"""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def make_vis_grid(lr_up: Image.Image, sr: Image.Image, hr: Image.Image, bicubic: Image.Image = None):
    """Return a single PIL image concatenating LR_up / Bicubic / SR / HR"""
    if bicubic is None:
        bicubic = lr_up

    images = [lr_up, bicubic, sr, hr]
    w, h = hr.size
    canvas = Image.new("RGB", (w * 4, h))
    for i, im in enumerate(images):
        canvas.paste(im.convert("RGB"), (i * w, 0))
    return canvas