import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from infer import load_named_model_for_inference
from tiler import RealESRGANTiler


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
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        path,
    ]
    result = run_command(cmd)
    data = json.loads(result.stdout.decode("utf-8"))
    stream = data["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1",
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None,
        "duration": float(stream.get("duration", 0.0) or 0.0),
    }


def has_audio(path: str) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return bool(result.stdout.strip())


def start_reader(input_path: str, width: int, height: int, max_frames: int = 0):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        input_path,
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    if max_frames > 0:
        cmd[-3:-3] = ["-frames:v", str(max_frames)]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * height * 3)


def start_writer(output_path: str, width: int, height: int, fps: str, crf: int, preset: str):
    ensure_dir(os.path.dirname(output_path))
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        fps,
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def mux_audio(input_video: str, video_only: str, output_path: str):
    ensure_dir(os.path.dirname(output_path))
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        video_only,
        "-i",
        input_video,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
    run_command(cmd)


def frame_bytes_to_tensor(frame_bytes: bytes, width: int, height: int) -> torch.Tensor:
    frame = torch.frombuffer(bytearray(frame_bytes), dtype=torch.uint8)
    frame = frame.view(height, width, 3).permute(2, 0, 1).unsqueeze(0)
    return frame.float().div(255.0)


def tensor_to_frame_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    tensor = tensor.squeeze(0).permute(1, 2, 0).mul(255.0).round().byte().contiguous()
    return tensor.numpy().tobytes()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Stream video inference with Real-ESRGAN.")
    parser.add_argument("--input", type=str, required=True, help="Input video path.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--output", type=str, required=True, help="Output mp4 path.")
    parser.add_argument("--model-name", type=str, default="RealESRGAN_x4plus")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--tile-pad", type=int, default=10)
    parser.add_argument("--pre-pad", type=int, default=0)
    parser.add_argument("--fp32", action="store_true", help="Disable half precision.")
    parser.add_argument(
        "--outscale",
        type=float,
        default=None,
        help="Final scale. Defaults to model native scale; use 2 for 1080p->4K.",
    )
    parser.add_argument("--crf", type=int, default=18, help="x264 quality; lower is better.")
    parser.add_argument("--preset", type=str, default="medium", help="x264 preset.")
    parser.add_argument("--no-audio", action="store_true", help="Do not copy source audio.")
    parser.add_argument("--max-frames", type=int, default=0, help="Process only the first N frames.")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    info = probe_video(args.input)
    in_w = info["width"]
    in_h = info["height"]
    fps = info["fps"]
    total_frames = info["frames"]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, native_scale, used_key = load_named_model_for_inference(args.model_name, args.ckpt, device)
    outscale = float(args.outscale) if args.outscale is not None else float(native_scale)
    out_w = int(round(in_w * outscale))
    out_h = int(round(in_h * outscale))

    tiler = RealESRGANTiler(
        model=model,
        scale=native_scale,
        device=device,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=(device.type == "cuda" and not args.fp32),
    )

    output_path = os.path.abspath(args.output)
    temp_path = output_path.replace(".mp4", ".video_only.mp4")

    print("=" * 60, flush=True)
    print("Real-ESRGAN Video Inference", flush=True)
    print("=" * 60, flush=True)
    print(f"Input    : {args.input}", flush=True)
    print(f"Output   : {output_path}", flush=True)
    print(f"Ckpt     : {args.ckpt}", flush=True)
    print(f"Model    : {args.model_name}", flush=True)
    print(f"Load key : {used_key}", flush=True)
    print(f"Device   : {device}", flush=True)
    print(f"Frames   : {total_frames if total_frames is not None else 'unknown'}", flush=True)
    print(f"Input    : {in_w}x{in_h} @ {fps}", flush=True)
    print(f"Output   : {out_w}x{out_h} (outscale={outscale:g})", flush=True)
    print(f"Tile     : {args.tile}", flush=True)
    print("=" * 60, flush=True)

    reader = start_reader(args.input, in_w, in_h, max_frames=args.max_frames)
    writer = start_writer(temp_path, out_w, out_h, fps, args.crf, args.preset)
    frame_size = in_w * in_h * 3
    frame_idx = 0

    try:
        while True:
            frame_bytes = reader.stdout.read(frame_size)
            if not frame_bytes:
                break
            if len(frame_bytes) != frame_size:
                raise RuntimeError(f"Incomplete frame read at frame {frame_idx + 1}")

            img = frame_bytes_to_tensor(frame_bytes, in_w, in_h)
            pred = tiler.enhance_tensor(img)
            if outscale != float(native_scale):
                pred = F.interpolate(
                    pred,
                    size=(out_h, out_w),
                    mode="bicubic",
                    align_corners=False,
                ).clamp(0, 1)

            writer.stdin.write(tensor_to_frame_bytes(pred))
            frame_idx += 1

            if frame_idx == 1 or frame_idx % 10 == 0:
                if total_frames:
                    print(f"[{frame_idx}/{total_frames}] processed", flush=True)
                else:
                    print(f"[{frame_idx}] processed", flush=True)

            del img, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        if reader.stdout:
            reader.stdout.close()
        reader.wait()
        if writer.stdin:
            writer.stdin.close()
        writer.wait()

    if writer.returncode != 0:
        raise RuntimeError(f"ffmpeg writer failed with code {writer.returncode}")

    if not args.no_audio and has_audio(args.input):
        mux_audio(args.input, temp_path, output_path)
        os.remove(temp_path)
    else:
        os.replace(temp_path, output_path)

    print("=" * 60, flush=True)
    print(f"Finished {frame_idx} frames.", flush=True)
    print(f"Saved: {output_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
