import argparse
import csv
import json
import math
import os
import subprocess

import torch
import torch.nn.functional as F


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def run_command(cmd):
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def probe_video(path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        path,
    ]
    data = json.loads(run_command(cmd).stdout.decode("utf-8"))
    stream = data["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream.get("r_frame_rate", "30/1"),
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None,
        "duration": float(stream.get("duration", 0.0) or 0.0),
    }


def start_reader(path: str, width: int, height: int, scale_to=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if scale_to is not None:
        cmd.extend(["-vf", f"scale={scale_to[0]}:{scale_to[1]}:flags=bicubic"])
    cmd.extend(["-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * height * 3)


def frame_to_tensor(frame_bytes: bytes, width: int, height: int) -> torch.Tensor:
    frame = torch.frombuffer(bytearray(frame_bytes), dtype=torch.uint8)
    frame = frame.view(height, width, 3).permute(2, 0, 1).float().div(255.0)
    return frame


def crop_border_tensor(x: torch.Tensor, crop_border: int) -> torch.Tensor:
    if crop_border <= 0:
        return x
    _, h, w = x.shape
    if crop_border * 2 >= h or crop_border * 2 >= w:
        raise ValueError(f"crop_border={crop_border} is too large for image size {w}x{h}")
    return x[:, crop_border:-crop_border, crop_border:-crop_border]


def psnr(pred: torch.Tensor, gt: torch.Tensor, crop_border: int = 0) -> float:
    pred = crop_border_tensor(pred.clamp(0, 1), crop_border)
    gt = crop_border_tensor(gt.clamp(0, 1), crop_border)
    mse = torch.mean((pred - gt) ** 2).item()
    if mse <= 1e-10:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d.view(1, 1, window_size, window_size)


def ssim(pred: torch.Tensor, gt: torch.Tensor, crop_border: int = 0) -> float:
    pred = crop_border_tensor(pred.clamp(0, 1), crop_border).unsqueeze(0)
    gt = crop_border_tensor(gt.clamp(0, 1), crop_border).unsqueeze(0)
    _, channels, h, w = pred.shape
    window_size = min(11, h, w)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return 1.0 if torch.equal(pred, gt) else 0.0

    kernel = gaussian_kernel(window_size).expand(channels, 1, window_size, window_size)
    padding = window_size // 2
    mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_gt = F.conv2d(gt, kernel, padding=padding, groups=channels)
    mu_pred_sq = mu_pred.pow(2)
    mu_gt_sq = mu_gt.pow(2)
    mu_pred_gt = mu_pred * mu_gt
    sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_pred_sq
    sigma_gt_sq = F.conv2d(gt * gt, kernel, padding=padding, groups=channels) - mu_gt_sq
    sigma_pred_gt = F.conv2d(pred * gt, kernel, padding=padding, groups=channels) - mu_pred_gt
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score = ((2 * mu_pred_gt + c1) * (2 * sigma_pred_gt + c2)) / (
        (mu_pred_sq + mu_gt_sq + c1) * (sigma_pred_sq + sigma_gt_sq + c2)
    )
    return float(score.mean().item())


def write_csv(path: str, fieldnames, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Compute video PSNR/SSIM frame by frame.")
    parser.add_argument("--gt", required=True, help="Ground-truth/reference video.")
    parser.add_argument("--pred", required=True, help="Predicted video.")
    parser.add_argument("--output", required=True, help="Output metrics directory.")
    parser.add_argument("--name", default="method", help="Method name in summary.")
    parser.add_argument("--crop-border", type=int, default=0)
    parser.add_argument(
        "--scale-gt-to-pred",
        action="store_true",
        help="Resize GT video to prediction resolution before metric calculation.",
    )
    args = parser.parse_args()

    gt_info = probe_video(args.gt)
    pred_info = probe_video(args.pred)
    out_w = pred_info["width"]
    out_h = pred_info["height"]
    if (gt_info["width"], gt_info["height"]) != (out_w, out_h):
        if not args.scale_gt_to_pred:
            raise ValueError(
                f"Resolution mismatch: gt={gt_info['width']}x{gt_info['height']}, "
                f"pred={out_w}x{out_h}. Use --scale-gt-to-pred."
            )
        gt_scale = (out_w, out_h)
    else:
        gt_scale = None

    gt_reader = start_reader(args.gt, out_w, out_h, scale_to=gt_scale)
    pred_reader = start_reader(args.pred, out_w, out_h)
    frame_size = out_w * out_h * 3

    rows = []
    total_psnr = 0.0
    total_ssim = 0.0
    frame_idx = 0

    try:
        while True:
            gt_bytes = gt_reader.stdout.read(frame_size)
            pred_bytes = pred_reader.stdout.read(frame_size)
            if not gt_bytes and not pred_bytes:
                break
            if len(gt_bytes) != frame_size or len(pred_bytes) != frame_size:
                break

            gt = frame_to_tensor(gt_bytes, out_w, out_h)
            pred = frame_to_tensor(pred_bytes, out_w, out_h)
            frame_psnr = psnr(pred, gt, crop_border=args.crop_border)
            frame_ssim = ssim(pred, gt, crop_border=args.crop_border)
            rows.append({
                "frame": frame_idx,
                "psnr": frame_psnr,
                "ssim": frame_ssim,
            })
            total_psnr += frame_psnr
            total_ssim += frame_ssim
            frame_idx += 1

            if frame_idx == 1 or frame_idx % 25 == 0:
                print(f"[{frame_idx}] PSNR={frame_psnr:.4f}, SSIM={frame_ssim:.4f}", flush=True)
    finally:
        if gt_reader.stdout:
            gt_reader.stdout.close()
        if pred_reader.stdout:
            pred_reader.stdout.close()
        gt_reader.wait()
        pred_reader.wait()

    if not rows:
        raise RuntimeError("No frames were evaluated.")

    summary = {
        "method": args.name,
        "frames": len(rows),
        "width": out_w,
        "height": out_h,
        "psnr": total_psnr / len(rows),
        "ssim": total_ssim / len(rows),
    }

    ensure_dir(args.output)
    write_csv(os.path.join(args.output, "summary.csv"), ["method", "frames", "width", "height", "psnr", "ssim"], [summary])
    write_csv(os.path.join(args.output, "per_frame.csv"), ["frame", "psnr", "ssim"], rows)
    with open(os.path.join(args.output, "summary.md"), "w", encoding="utf-8") as f:
        f.write("| Method | Frames | Resolution | PSNR | SSIM |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        f.write(
            f"| {summary['method']} | {summary['frames']} | "
            f"{summary['width']}x{summary['height']} | "
            f"{summary['psnr']:.4f} | {summary['ssim']:.4f} |\n"
        )

    print("| Method | Frames | Resolution | PSNR | SSIM |")
    print("|---|---:|---:|---:|---:|")
    print(
        f"| {summary['method']} | {summary['frames']} | {summary['width']}x{summary['height']} | "
        f"{summary['psnr']:.4f} | {summary['ssim']:.4f} |"
    )
    print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()
