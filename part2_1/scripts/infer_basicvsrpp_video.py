import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmagic.models import BasicVSRPlusPlusNet
from PIL import Image
from tqdm import tqdm


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMG_EXTS


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


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
            f"Unexpected keys: {unexpected[:10]}"
        )


def extract_video_frames(video_path: str, frames_dir: str, max_frames=None, input_max_side=None):
    ensure_dir(frames_dir)
    pattern = os.path.join(frames_dir, "%08d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-start_number",
        "0",
    ]
    if max_frames is not None:
        cmd.extend(["-vframes", str(max_frames)])
    if input_max_side is not None and input_max_side > 0:
        scale_filter = (
            "scale="
            f"'if(gte(iw,ih),min({input_max_side},iw),-2)':"
            f"'if(gte(iw,ih),-2,min({input_max_side},ih))'"
        )
        cmd.extend(["-vf", scale_filter])
    cmd.extend([
        "-vsync",
        "0",
        pattern,
    ])
    run_cmd(cmd)


def get_video_fps(video_path: str, fallback: int = 30) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        if "/" in out:
            num, den = out.split("/")
            den = float(den)
            if den != 0:
                return float(num) / den
        return float(out)
    except Exception:
        return float(fallback)


def list_frames(folder: str):
    return sorted([
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if is_image_file(name)
    ])


def load_sequence(folder, max_frames=None):
    paths = list_frames(folder)
    if max_frames is not None:
        paths = paths[:max_frames]
    if not paths:
        raise ValueError(f"No image frames found in: {folder}")

    imgs = []
    names = []
    for path in tqdm(paths, desc=f"Loading {os.path.basename(folder)}"):
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        imgs.append(tensor)
        names.append(os.path.basename(path))
    return torch.stack(imgs), names


def load_frame_paths(paths):
    imgs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        imgs.append(tensor)
    return torch.stack(imgs)


def tensor_to_uint8_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    return (x * 255.0).round().astype(np.uint8)


def pad_to_multiple_of_4(x):
    _, _, h, w = x.shape
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w


def save_sr_frames(sr_seq: torch.Tensor, out_dir: str, filenames):
    ensure_dir(out_dir)
    for idx in tqdm(range(sr_seq.shape[0]), desc="Saving SR frames"):
        arr = tensor_to_uint8_image(sr_seq[idx])
        Image.fromarray(arr).save(os.path.join(out_dir, filenames[idx]))


def save_sr_frame(sr_frame: torch.Tensor, out_dir: str, filename: str):
    ensure_dir(out_dir)
    arr = tensor_to_uint8_image(sr_frame)
    Image.fromarray(arr).save(os.path.join(out_dir, filename))


def start_video_writer(out_path: str, width: int, height: int, fps: float):
    ensure_dir(os.path.dirname(out_path))
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_sr_frame_to_video(sr_frame: torch.Tensor, writer):
    arr = tensor_to_uint8_image(sr_frame)
    writer.stdin.write(arr.tobytes())


def infer_full_sequence(model, lr_seq: torch.Tensor, device: torch.device):
    dtype = next(model.parameters()).dtype
    lr_input = lr_seq.unsqueeze(0).to(device)
    if dtype != lr_input.dtype:
        lr_input = lr_input.to(dtype=dtype)
    _, t, c, h, w = lr_input.shape
    lr_input = lr_input.view(-1, c, h, w)
    lr_input, pad_h, pad_w = pad_to_multiple_of_4(lr_input)
    _, _, h_new, w_new = lr_input.shape
    lr_input = lr_input.view(1, t, c, h_new, w_new)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        sr_seq = model(lr_input)

    sr_seq = sr_seq.squeeze(0).cpu()
    if pad_h > 0 or pad_w > 0:
        sr_seq = sr_seq[:, :, : h * 4, : w * 4]
    return sr_seq


def infer_and_save_chunked_sequence(
    model,
    frame_paths,
    filenames,
    out_dir: str,
    device: torch.device,
    chunk_size: int,
    overlap: int,
):
    total_frames = len(frame_paths)
    if total_frames == 0:
        raise ValueError("No frames to infer.")

    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2 - 1))
    step = chunk_size - 2 * overlap
    if step <= 0:
        raise ValueError("chunk_size must be larger than 2 * chunk_overlap")

    ensure_dir(out_dir)
    print(
        f"Running BasicVSR++ in streaming chunks: "
        f"chunk_size={chunk_size}, overlap={overlap}, step={step}"
    )

    saved = 0
    for start in tqdm(range(0, total_frames, step), desc="Infer/save chunks"):
        end = min(start + chunk_size, total_frames)
        chunk = load_frame_paths(frame_paths[start:end])
        chunk_out = infer_full_sequence(model, chunk, device)

        keep_start = 0 if start == 0 else overlap
        keep_end = chunk_out.shape[0] if end == total_frames else chunk_out.shape[0] - overlap

        for local_idx in range(keep_start, keep_end):
            global_idx = start + local_idx
            save_sr_frame(chunk_out[local_idx], out_dir, filenames[global_idx])
            saved += 1

        del chunk, chunk_out
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if end == total_frames:
            break

    if saved != total_frames:
        raise RuntimeError(f"Streaming inference saved {saved} frames, expected {total_frames}.")


def infer_and_write_chunked_sequence(
    model,
    frame_paths,
    out_video_path: str,
    fps: float,
    device: torch.device,
    chunk_size: int,
    overlap: int,
):
    total_frames = len(frame_paths)
    if total_frames == 0:
        raise ValueError("No frames to infer.")

    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2 - 1))
    step = chunk_size - 2 * overlap
    if step <= 0:
        raise ValueError("chunk_size must be larger than 2 * chunk_overlap")

    first = Image.open(frame_paths[0]).convert("RGB")
    width, height = first.size[0] * 4, first.size[1] * 4
    first.close()
    writer = start_video_writer(out_video_path, width=width, height=height, fps=fps)

    print(
        f"Running BasicVSR++ in streaming chunks to video: "
        f"chunk_size={chunk_size}, overlap={overlap}, step={step}"
    )

    written = 0
    try:
        for start in tqdm(range(0, total_frames, step), desc="Infer/write chunks"):
            end = min(start + chunk_size, total_frames)
            chunk = load_frame_paths(frame_paths[start:end])
            chunk_out = infer_full_sequence(model, chunk, device)

            keep_start = 0 if start == 0 else overlap
            keep_end = chunk_out.shape[0] if end == total_frames else chunk_out.shape[0] - overlap

            for local_idx in range(keep_start, keep_end):
                write_sr_frame_to_video(chunk_out[local_idx], writer)
                written += 1

            del chunk, chunk_out
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if end == total_frames:
                break
    finally:
        if writer.stdin:
            writer.stdin.close()
        ret = writer.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg video writer failed with exit code {ret}.")

    if written != total_frames:
        raise RuntimeError(f"Streaming inference wrote {written} frames, expected {total_frames}.")


def infer_chunked_sequence(
    model,
    lr_seq: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    overlap: int,
):
    total_frames, c, h, w = lr_seq.shape
    if chunk_size <= 0 or chunk_size >= total_frames:
        return infer_full_sequence(model, lr_seq, device)

    overlap = max(0, min(overlap, chunk_size // 2 - 1))
    step = chunk_size - 2 * overlap
    if step <= 0:
        raise ValueError("chunk_size must be larger than 2 * chunk_overlap")

    outputs = []
    print(
        f"Running BasicVSR++ in chunks: "
        f"chunk_size={chunk_size}, overlap={overlap}, step={step}"
    )

    for start in tqdm(range(0, total_frames, step), desc="Infer chunks"):
        end = min(start + chunk_size, total_frames)
        chunk = lr_seq[start:end]
        chunk_out = infer_full_sequence(model, chunk, device)

        keep_start = 0 if start == 0 else overlap
        keep_end = chunk_out.shape[0] if end == total_frames else chunk_out.shape[0] - overlap
        if keep_start < keep_end:
            outputs.append(chunk_out[keep_start:keep_end])

        if end == total_frames:
            break

    sr_seq = torch.cat(outputs, dim=0)
    if sr_seq.shape[0] != total_frames:
        raise RuntimeError(
            f"Chunked inference produced {sr_seq.shape[0]} frames, "
            f"expected {total_frames}."
        )
    return sr_seq


def frames_to_video(frames_dir: str, out_path: str, fps: float):
    ensure_dir(os.path.dirname(out_path))
    frame_pattern = os.path.join(frames_dir, "%08d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    run_cmd(cmd)


def prepare_input_frames(input_path: str, output_dir: str, max_frames=None, input_max_side=None):
    input_path = os.path.abspath(input_path)
    if os.path.isdir(input_path):
        return input_path, None

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    extracted_dir = os.path.join(output_dir, "input_frames")
    print(f"Extracting video frames to: {extracted_dir}")
    if os.path.isdir(extracted_dir):
        shutil.rmtree(extracted_dir)
    extract_video_frames(
        input_path,
        extracted_dir,
        max_frames=max_frames,
        input_max_side=input_max_side,
    )
    return extracted_dir, input_path


def parse_args():
    parser = argparse.ArgumentParser(description="BasicVSR++ inference for a real video or frame folder.")
    parser.add_argument("--input", type=str, required=True, help="Input mp4/video file or frame folder")
    parser.add_argument("--output", type=str, required=True, help="Output folder")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/fc/Coding/CV/part2_1/mmagic/work_dirs/basicvsr-pp_c64n7_fc_finetune/basicvsr-pp_c64n7_fc_finetune/best_PSNR_iter_20000.pth",
        help="BasicVSR++ generator checkpoint",
    )
    parser.add_argument("--fps", type=float, default=None, help="Output FPS. Defaults to source FPS for video input, or 30 for folder input.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick testing")
    parser.add_argument(
        "--input-max-side",
        type=int,
        default=None,
        help="Resize video input before VSR so its longer side is at most this value. Useful for large real videos.",
    )
    parser.add_argument("--cpu-cache-length", type=int, default=30)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Number of frames processed per forward pass. Lower this to reduce GPU memory.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=2,
        help="Neighbor frames shared between chunks to reduce boundary artifacts.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on CUDA to reduce memory usage.")
    parser.add_argument("--keep-input-frames", action="store_true", help="Keep extracted input frames for video input")
    parser.add_argument("--video-only", action="store_true", help="Write SR video directly without saving SR PNG frames.")
    parser.add_argument(
        "--no-stream-save",
        action="store_true",
        help="Keep all SR tensors in memory before saving. Not recommended for long videos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ensure_dir(args.output)
    input_frames_dir, source_video = prepare_input_frames(
        args.input,
        args.output,
        max_frames=args.max_frames,
        input_max_side=args.input_max_side,
    )
    fps = args.fps
    if fps is None:
        fps = get_video_fps(source_video) if source_video else 30.0

    print("=" * 72)
    print("BasicVSR++ Video Inference")
    print("=" * 72)
    print(f"Input frames : {input_frames_dir}")
    print(f"Output       : {args.output}")
    print(f"Checkpoint   : {args.ckpt}")
    print(f"Device       : {device}")
    print(f"FPS          : {fps}")
    print("=" * 72)

    model = BasicVSRPlusPlusNet(
        mid_channels=64,
        num_blocks=7,
        cpu_cache_length=args.cpu_cache_length,
    )
    load_basicvsrpp_generator_checkpoint(model, args.ckpt)
    model = model.to(device).eval()
    if args.half:
        if device.type != "cuda":
            print("[WARN] --half is only supported on CUDA. Keeping FP32.")
        else:
            model = model.half()
            print("[INFO] Using FP16 inference.")

    print("Running BasicVSR++...")
    sr_frames_dir = os.path.join(args.output, "frames")
    videos_dir = os.path.join(args.output, "videos")

    frame_paths = list_frames(input_frames_dir)
    if not source_video and args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]
    filenames = [os.path.basename(path) for path in frame_paths]

    sr_video_path = os.path.join(videos_dir, "sr.mp4")
    if args.video_only:
        infer_and_write_chunked_sequence(
            model,
            frame_paths,
            sr_video_path,
            fps=fps,
            device=device,
            chunk_size=int(args.chunk_size),
            overlap=int(args.chunk_overlap),
        )
    elif args.no_stream_save:
        lr_seq = load_frame_paths(frame_paths)
        sr_seq = infer_chunked_sequence(
            model,
            lr_seq,
            device,
            chunk_size=int(args.chunk_size),
            overlap=int(args.chunk_overlap),
        )
        save_sr_frames(sr_seq, sr_frames_dir, filenames)
    else:
        infer_and_save_chunked_sequence(
            model,
            frame_paths,
            filenames,
            sr_frames_dir,
            device,
            chunk_size=int(args.chunk_size),
            overlap=int(args.chunk_overlap),
        )

        frames_to_video(sr_frames_dir, sr_video_path, fps=fps)

    if source_video and not args.keep_input_frames:
        shutil.rmtree(input_frames_dir, ignore_errors=True)

    if not args.video_only:
        print(f"Saved SR frames to: {sr_frames_dir}")
    print(f"Saved SR video to: {sr_video_path}")


if __name__ == "__main__":
    main()
