import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PairSRDataset
from .model import SRCNN
from .utils import ensure_dir, psnr_torch, tensor_to_pil, make_lr_bicubic, upsample_bicubic, make_vis_grid


def run_test(cfg: dict, ckpt_path: str = None):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    scale = int(cfg["scale"])

    test_set = PairSRDataset(
    cfg["paths"]["test_lr"],
    cfg["paths"]["test_hr"],
    patch_size=int(cfg["patch_size"]),
    scale=scale,
    is_train=False
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=int(cfg["num_workers"]))

    model = SRCNN().to(device)
    if ckpt_path is None:
        # try find latest in ckpt_dir
        ckpt_dir = cfg["paths"]["ckpt_dir"]
        if not os.path.isdir(ckpt_dir):
            raise ValueError("No checkpoint provided and ckpt_dir not found.")
        cands = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pth")])
        if len(cands) == 0:
            raise ValueError("No checkpoint found in ckpt_dir.")
        ckpt_path = cands[-1]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = cfg["paths"]["out_dir"]
    vis_dir = os.path.join(out_dir, "vis")
    ensure_dir(vis_dir)

    total_psnr_sr = 0.0
    total_psnr_bi = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            lr_up = batch["lr_up"].to(device)
            hr = batch["hr"].to(device)

            sr = model(lr_up).clamp(0, 1)

            # PSNR: bicubic(=lr_up) vs HR, SR vs HR
            total_psnr_bi += psnr_torch(lr_up.clamp(0, 1), hr)
            total_psnr_sr += psnr_torch(sr, hr)

            # save a few visualizations
            if i < 10:
                hr_pil = tensor_to_pil(hr[0])
                lr_up_pil = tensor_to_pil(lr_up[0])
                sr_pil = tensor_to_pil(sr[0])

                # explicit bicubic from HR (same as lr_up) to show baseline
                lr = make_lr_bicubic(hr_pil, scale)
                bicubic = upsample_bicubic(lr, hr_pil.size)

                grid = make_vis_grid(lr_up_pil, sr_pil, hr_pil, bicubic=bicubic)
                name = os.path.splitext(os.path.basename(batch["path"][0]))[0]
                grid.save(os.path.join(vis_dir, f"{name}_LRup_BI_SR_HR.png"))

    n = len(test_loader)
    avg_bi = total_psnr_bi / max(1, n)
    avg_sr = total_psnr_sr / max(1, n)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Average PSNR (Bicubic) = {avg_bi:.2f} dB")
    print(f"Average PSNR (SRCNN)   = {avg_sr:.2f} dB")
    print(f"Saved visualization to: {vis_dir}")