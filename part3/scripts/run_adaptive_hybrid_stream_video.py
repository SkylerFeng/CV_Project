import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image


CURRENT_DIR = Path(__file__).resolve().parent
PART3_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PART3_DIR / "src"))

from adaptive_mask import AdaptiveMaskConfig, build_adaptive_alpha
from fusion import blend_with_mask
from io_utils import build_image_map, ensure_dir, load_rgb, resize_like
from video_utils import FFmpegVideoWriter


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def adaptive_config(raw: dict) -> AdaptiveMaskConfig:
    return AdaptiveMaskConfig(
        mode=raw.get("mode", "real"),
        anime_threshold=float(raw.get("anime_threshold", 0.50)),
        real_max_alpha=float(raw.get("real_max_alpha", 0.24)),
        anime_max_alpha=float(raw.get("anime_max_alpha", 0.46)),
        min_alpha=float(raw.get("min_alpha", 0.0)),
        texture_gain=float(raw.get("texture_gain", 0.82)),
        line_gain=float(raw.get("line_gain", 0.40)),
        flat_gain=float(raw.get("flat_gain", 0.10)),
        edge_protect_strength=float(raw.get("edge_protect_strength", 1.05)),
        temporal_protect_strength=float(raw.get("temporal_protect_strength", 0.75)),
        disagreement_protect_strength=float(raw.get("disagreement_protect_strength", 1.10)),
        structure_protect_strength=float(raw.get("structure_protect_strength", 1.00)),
        hallucination_protect_strength=float(raw.get("hallucination_protect_strength", 0.90)),
        flicker_protect_strength=float(raw.get("flicker_protect_strength", 0.80)),
        blur_radius=float(raw.get("mask_blur_radius", 5.0)),
        gamma=float(raw.get("gamma", 1.35)),
    )


def probe_video(path: str):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,duration",
        "-of",
        "json",
        path,
    ]
    out = subprocess.check_output(cmd, text=True)
    stream = json.loads(out)["streams"][0]
    fps = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None,
    }


def fps_to_float(value: str) -> float:
    if "/" in value:
        num, den = value.split("/")
        den = float(den)
        return float(num) / den if den else 30.0
    return float(value)


def start_video_reader(path: str, width: int, height: int):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * height * 3)


def read_frame(reader, width: int, height: int):
    frame_size = width * height * 3
    data = reader.stdout.read(frame_size)
    if not data:
        return None
    if len(data) != frame_size:
        raise RuntimeError("Incomplete frame read from generative video.")
    return Image.frombytes("RGB", (width, height), data)


def write_stats(path: str, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "content_label",
                "anime_score",
                "anime_weight",
                "max_alpha",
                "mean_alpha",
                "mean_structure_protect",
                "mean_uncertain_texture",
                "mean_hallucination_risk",
                "mean_flicker_risk",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Stream Part 3 adaptive hybrid directly to mp4.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--generative-video", type=str, required=True)
    parser.add_argument("--output-video", type=str, required=True)
    parser.add_argument("--stats", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    mask_cfg = adaptive_config(cfg.get("adaptive", {}))
    fps = float(cfg.get("output", {}).get("fps", 0) or 0)

    basic_map = build_image_map(paths["basic_dir"])
    keys = sorted(basic_map)
    if args.max_frames > 0:
        keys = keys[: args.max_frames]
    if not keys:
        raise ValueError(f"No BasicVSR++ frames found: {paths['basic_dir']}")

    video_info = probe_video(args.generative_video)
    if fps <= 0:
        fps = fps_to_float(video_info["fps"])
    reader = start_video_reader(args.generative_video, video_info["width"], video_info["height"])

    first_basic = load_rgb(basic_map[keys[0]])
    ensure_dir(os.path.dirname(args.output_video))
    writer = FFmpegVideoWriter(args.output_video, first_basic.size, fps=int(round(fps)))
    stats_rows = []

    print("=" * 72)
    print("Part 3 Streaming Adaptive Hybrid")
    print("=" * 72)
    print(f"Basic frames    : {paths['basic_dir']}")
    print(f"Generative video: {args.generative_video}")
    print(f"Output video    : {args.output_video}")
    print(f"Frames          : {len(keys)}")
    print(f"Mode            : {mask_cfg.mode}")
    print("=" * 72)

    try:
        for idx, key in enumerate(keys):
            gen = read_frame(reader, video_info["width"], video_info["height"])
            if gen is None:
                break

            basic = load_rgb(basic_map[key])
            gen = resize_like(gen, basic)
            prev_basic = load_rgb(basic_map[keys[idx - 1]]) if idx > 0 else None
            next_basic = load_rgb(basic_map[keys[idx + 1]]) if idx + 1 < len(keys) else None

            mask, _, stats = build_adaptive_alpha(
                basic=basic,
                generative=gen,
                previous_basic=prev_basic,
                next_basic=next_basic,
                cfg=mask_cfg,
            )
            fused = blend_with_mask(basic, gen, mask)
            writer.append_pil(fused)

            stats["frame"] = key
            stats_rows.append(stats)

            frame_no = idx + 1
            if frame_no == 1 or frame_no % 25 == 0 or frame_no == len(keys):
                print(
                    f"[{frame_no}/{len(keys)}] {key}: "
                    f"content={stats['content_label']}, mean_alpha={stats['mean_alpha']:.3f}",
                    flush=True,
                )
    finally:
        writer.close()
        if reader.stdout:
            reader.stdout.close()
        reader.wait()

    write_stats(args.stats, stats_rows)
    print(f"Saved video: {args.output_video}")
    print(f"Saved stats: {args.stats}")


if __name__ == "__main__":
    main()
