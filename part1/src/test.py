import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PairSRDataset
from .model import SRCNN
from .utils import (
    ensure_dir,
    psnr_torch,
    ssim_torch,
    tensor_to_pil,
    pil_to_tensor,
    make_lr_bicubic,
    upsample_bicubic,
    upsample_lanczos,
    make_vis_grid,
)
from .video_utils import FFmpegVideoWriter

def run_test(cfg: dict, ckpt_path: str = None):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    scale = int(cfg["scale"])
    video_cfg = cfg.get("test_video", {})
    export_video = bool(video_cfg.get("enabled", True))
    max_video_sequences = int(video_cfg.get("max_sequences", 1))
    video_fps = int(video_cfg.get("fps", 30))
    test_videos = cfg.get("test_videos", None)

    test_set = PairSRDataset(
        cfg["paths"]["test_lr"],
        cfg["paths"]["test_hr"],
        patch_size=int(cfg["patch_size"]),
        scale=scale,
        is_train=False,
        video_names=test_videos,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["num_workers"])
    )

    model = SRCNN().to(device)
    if ckpt_path is None:
        ckpt_dir = cfg["paths"]["ckpt_dir"]
        if not os.path.isdir(ckpt_dir):
            raise ValueError("No checkpoint provided and ckpt_dir not found.")
        cands = sorted(
            [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        )
        if len(cands) == 0:
            raise ValueError("No checkpoint found in ckpt_dir.")
        ckpt_path = cands[-1]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = cfg["paths"]["out_dir"]
    vis_dir = os.path.join(out_dir, "vis")
    video_dir = os.path.join(out_dir, "videos")
    ensure_dir(vis_dir)
    ensure_dir(video_dir)

    total_psnr_sr = 0.0
    total_psnr_bi = 0.0
    total_psnr_lz = 0.0

    total_ssim_sr = 0.0
    total_ssim_bi = 0.0
    total_ssim_lz = 0.0

    comparison_writers = {}
    sr_writers = {}
    try:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
                lr_up = batch["lr_up"].to(device)
                hr = batch["hr"].to(device)

                sr = model(lr_up).clamp(0, 1)

                # 构造真实的 LR，再分别做 Bicubic / Lanczos 上采样
                hr_pil = tensor_to_pil(hr[0])
                lr_pil = make_lr_bicubic(hr_pil, scale)

                bicubic_pil = upsample_bicubic(lr_pil, hr_pil.size)
                lanczos_pil = upsample_lanczos(lr_pil, hr_pil.size)

                bicubic_tensor = pil_to_tensor(bicubic_pil).unsqueeze(0).to(device).clamp(0, 1)
                lanczos_tensor = pil_to_tensor(lanczos_pil).unsqueeze(0).to(device).clamp(0, 1)

                total_psnr_bi += psnr_torch(bicubic_tensor, hr)
                total_psnr_lz += psnr_torch(lanczos_tensor, hr)
                total_psnr_sr += psnr_torch(sr, hr)

                total_ssim_bi += ssim_torch(bicubic_tensor, hr)
                total_ssim_lz += ssim_torch(lanczos_tensor, hr)
                total_ssim_sr += ssim_torch(sr, hr)

                lr_up_pil = tensor_to_pil(lr_up[0])
                sr_pil = tensor_to_pil(sr[0])

                grid = make_vis_grid(
                    lr_up=lr_up_pil,
                    sr=sr_pil,
                    hr=hr_pil,
                    bicubic=bicubic_pil,
                    lanczos=lanczos_pil,
                )

                rel_path = batch["path"][0]
                video_name = os.path.dirname(rel_path) or "root"
                should_export_video = (
                    export_video and
                    (video_name in comparison_writers or
                     len(comparison_writers) < max_video_sequences)
                )

                if should_export_video and video_name not in comparison_writers:
                    comparison_writers[video_name] = FFmpegVideoWriter(
                        os.path.join(video_dir, f"{video_name}_comparison.mp4"),
                        grid.size,
                        fps=video_fps,
                    )
                    sr_writers[video_name] = FFmpegVideoWriter(
                        os.path.join(video_dir, f"{video_name}_srcnn.mp4"),
                        sr_pil.size,
                        fps=video_fps,
                    )

                if should_export_video:
                    comparison_writers[video_name].append_pil(grid)
                    sr_writers[video_name].append_pil(sr_pil)

                # save a few visualization images for report figures
                if i < 10:
                    name = os.path.splitext(os.path.basename(rel_path))[0]
                    grid.save(os.path.join(vis_dir, f"{name}_LRup_BI_LZ_SR_HR.png"))
    finally:
        for writer in comparison_writers.values():
            writer.close()
        for writer in sr_writers.values():
            writer.close()

    n = len(test_loader)
    avg_bi = total_psnr_bi / max(1, n)
    avg_lz = total_psnr_lz / max(1, n)
    avg_sr = total_psnr_sr / max(1, n)

    avg_ssim_bi = total_ssim_bi / max(1, n)
    avg_ssim_lz = total_ssim_lz / max(1, n)
    avg_ssim_sr = total_ssim_sr / max(1, n)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Average PSNR (Bicubic) = {avg_bi:.2f} dB")
    print(f"Average PSNR (Lanczos) = {avg_lz:.2f} dB")
    print(f"Average PSNR (SRCNN)   = {avg_sr:.2f} dB")
    print(f"Average SSIM (Bicubic) = {avg_ssim_bi:.4f}")
    print(f"Average SSIM (Lanczos) = {avg_ssim_lz:.4f}")
    print(f"Average SSIM (SRCNN)   = {avg_ssim_sr:.4f}")
    print(f"Saved visualization to: {vis_dir}")
    if export_video:
        print(f"Saved up to {max_video_sequences} video sequence(s) to: {video_dir}")
    else:
        print("Video export disabled.")
