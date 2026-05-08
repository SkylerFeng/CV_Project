import os
import subprocess
import torch
import torch.nn.functional as F
from mmagic.models import BasicVSRPlusPlusNet
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_basicvsrpp_generator_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if "params" in state_dict:
        state_dict = state_dict["params"]

    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    if any(key.startswith("generator.") for key in state_dict):
        state_dict = {
            key.removeprefix("generator."): value
            for key, value in state_dict.items()
            if key.startswith("generator.")
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match BasicVSRPlusPlusNet.\n"
            f"Missing keys: {missing[:10]}\n"
            f"Unexpected keys: {unexpected[:10]}")


def load_sequence(folder, max_frames=None):
    files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    if max_frames is not None:
        files = files[:max_frames]

    imgs = []
    for f in tqdm(files, desc=f"Loading {os.path.basename(folder)}"):
        img = Image.open(os.path.join(folder, f)).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        imgs.append(img)

    if not imgs:
        raise ValueError(f"No images found in folder: {folder}")

    return torch.stack(imgs), files


def tensor_to_uint8_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return x


class FFmpegVideoWriter:
    def __init__(self, out_path, frame_size, fps):
        width, height = frame_size
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            out_path,
        ]
        self.out_path = out_path
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        self.closed = False

    def append_data(self, frame):
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        self.process.wait()
        if self.process.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed while writing {self.out_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        elif not self.closed:
            self.closed = True
            self.process.kill()


def pad_to_multiple_of_4(x):
    _, _, h, w = x.shape
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4

    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, pad_h, pad_w


def save_comparison_sequence(sr_seq, hr_seq, lr_seq, out_folder, filenames, fps=30):
    images_folder = os.path.join(out_folder, "images")
    videos_folder = os.path.join(out_folder, "videos")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(videos_folder, exist_ok=True)

    sr_video_path = os.path.join(videos_folder, "sr.mp4")
    comparison_video_path = os.path.join(videos_folder, "comparison.mp4")

    sr_writer = None
    comparison_writer = None
    try:
        for i in tqdm(range(sr_seq.shape[0]), desc="Saving images and videos"):
            sr_img = sr_seq[i:i + 1]   # 1, C, H, W
            hr_img = hr_seq[i:i + 1]   # 1, C, H, W
            lr_img = lr_seq[i:i + 1]   # 1, C, h, w

            # bicubic upsample LR to HR size
            lr_up = F.interpolate(
                lr_img,
                size=hr_img.shape[-2:],
                mode="bicubic",
                align_corners=False
            )

            lr_np = tensor_to_uint8_image(lr_up.squeeze(0))
            sr_np = tensor_to_uint8_image(sr_img.squeeze(0))
            hr_np = tensor_to_uint8_image(hr_img.squeeze(0))
            if i == 0:
                Image.fromarray(lr_np).save(os.path.join(images_folder, "debug_lr.png"))
                Image.fromarray(sr_np).save(os.path.join(images_folder, "debug_sr.png"))
                Image.fromarray(hr_np).save(os.path.join(images_folder, "debug_hr.png"))

            h, w, _ = hr_np.shape
            canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)

            # left: bicubic LR, middle: SR, right: HR
            canvas[:, 0:w, :] = lr_np
            canvas[:, w:2 * w, :] = sr_np
            canvas[:, 2 * w:3 * w, :] = hr_np

            if sr_writer is None:
                sr_h, sr_w, _ = sr_np.shape
                sr_writer = FFmpegVideoWriter(sr_video_path, (sr_w, sr_h), fps)
                comparison_writer = FFmpegVideoWriter(
                    comparison_video_path, (w * 3, h), fps)

            sr_writer.append_data(sr_np)
            comparison_writer.append_data(canvas)
            Image.fromarray(canvas).save(os.path.join(images_folder, filenames[i]))
    finally:
        if sr_writer is not None:
            sr_writer.close()
        if comparison_writer is not None:
            comparison_writer.close()

    print(f"Saved SR video to: {sr_video_path}")
    print(f"Saved comparison video to: {comparison_video_path}")
    print(f"Saved comparison images to: {images_folder}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====== 改这里 ======
    lr_folder = "/home/fc/Coding/CV/data/val/val_sharp_bicubic/X4/000"
    hr_folder = "/home/fc/Coding/CV/data/val/val_sharp/000"
    out_folder = "/home/fc/Coding/CV/part2_1/results/basicvsrpp_compare_000"
    ckpt_path = "/home/fc/Coding/CV/part2_1/mmagic/work_dirs/basicvsr-pp_c64n7_fc_finetune/basicvsr-pp_c64n7_fc_finetune/best_PSNR_iter_20000.pth"
    max_frames = 100
    fps = 30
    cpu_cache_length = 30
    # ===================

    # load model
    model = BasicVSRPlusPlusNet(
        mid_channels=64,
        num_blocks=7,
        cpu_cache_length=cpu_cache_length)
    load_basicvsrpp_generator_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()

    # load input
    lr_seq, filenames = load_sequence(lr_folder, max_frames=max_frames)
    hr_seq, hr_filenames = load_sequence(hr_folder, max_frames=max_frames)

    if filenames != hr_filenames:
        raise ValueError("LR and HR filenames do not match.")

    lr_input = lr_seq.unsqueeze(0).to(device)

    # padding
    b, t, c, h, w = lr_input.shape
    lr_input = lr_input.view(-1, c, h, w)

    lr_input, pad_h, pad_w = pad_to_multiple_of_4(lr_input)

    # reshape 回去
    _, _, h_new, w_new = lr_input.shape
    lr_input = lr_input.view(1, t, c, h_new, w_new)

    if device == "cuda":
        torch.cuda.empty_cache()

    # inference
    with torch.no_grad():
        print("Running inference...")
        sr_seq = model(lr_input)  # 1, T, C, H, W

    sr_seq = sr_seq.squeeze(0).cpu()   # T, C, H, W
    if pad_h > 0 or pad_w > 0:
        sr_seq = sr_seq[:, :, :h*4, :w*4]
    hr_seq = hr_seq.cpu()
    lr_seq = lr_seq.cpu()
    

    # save comparison images
    save_comparison_sequence(sr_seq, hr_seq, lr_seq, out_folder, filenames, fps=fps)

    print(f"Saved outputs to: {out_folder}")


if __name__ == "__main__":
    main()
