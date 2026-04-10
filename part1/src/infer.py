import os
import torch
from PIL import Image

from .model import SRCNN
from .utils import ensure_dir, pil_to_tensor, tensor_to_pil, make_lr_bicubic, upsample_bicubic


def run_infer(cfg: dict, ckpt_path: str, input_path: str, output_path: str = None):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    scale = int(cfg["scale"])

    if ckpt_path is None:
        raise ValueError("--ckpt is required for infer (please provide a trained checkpoint)")

    model = SRCNN().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    hr = Image.open(input_path).convert("RGB")

    # inference: treat input as HR, create LR then upsample, then SR
    w, h = hr.size
    w = (w // scale) * scale
    h = (h // scale) * scale
    hr = hr.crop((0, 0, w, h))

    lr = make_lr_bicubic(hr, scale)
    lr_up = upsample_bicubic(lr, hr.size)

    x = pil_to_tensor(lr_up).unsqueeze(0).to(device)

    with torch.no_grad():
        sr = model(x).clamp(0, 1)[0]

    sr_pil = tensor_to_pil(sr)

    out_dir = cfg["paths"]["out_dir"]
    ensure_dir(out_dir)

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(out_dir, f"{base}_SR_x{scale}.png")

    sr_pil.save(output_path)
    print(f"Saved SR image: {output_path}")