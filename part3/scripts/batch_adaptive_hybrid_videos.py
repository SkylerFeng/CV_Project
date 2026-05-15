import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
PART3_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PART3_DIR / "src"))

from adaptive_mask import AdaptiveMaskConfig, build_adaptive_alpha
from fusion import blend_with_mask
from video_utils import FFmpegVideoWriter


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def adaptive_config(raw: dict) -> AdaptiveMaskConfig:
    return AdaptiveMaskConfig(
        mode=raw.get("mode", "real"),
        anime_threshold=float(raw.get("anime_threshold", 0.56)),
        real_max_alpha=float(raw.get("real_max_alpha", 0.36)),
        anime_max_alpha=float(raw.get("anime_max_alpha", 0.45)),
        min_alpha=float(raw.get("min_alpha", 0.0)),
        texture_gain=float(raw.get("texture_gain", 1.75)),
        line_gain=float(raw.get("line_gain", 0.25)),
        flat_gain=float(raw.get("flat_gain", 0.08)),
        edge_protect_strength=float(raw.get("edge_protect_strength", 0.55)),
        temporal_protect_strength=float(raw.get("temporal_protect_strength", 0.45)),
        disagreement_protect_strength=float(raw.get("disagreement_protect_strength", 0.55)),
        structure_protect_strength=float(raw.get("structure_protect_strength", 0.55)),
        hallucination_protect_strength=float(raw.get("hallucination_protect_strength", 0.55)),
        flicker_protect_strength=float(raw.get("flicker_protect_strength", 0.65)),
        blur_radius=float(raw.get("mask_blur_radius", 4.0)),
        gamma=float(raw.get("gamma", 0.85)),
    )


def iter_videos(root: Path):
    return sorted(root.rglob("sr.mp4"))


def probe_video(path: Path):
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
        str(path),
    ]
    stream = json.loads(subprocess.check_output(cmd, text=True))["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None,
    }


def start_reader(path: Path, width: int, height: int):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
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
        raise RuntimeError("Incomplete frame read from video stream.")
    return Image.frombytes("RGB", (width, height), data)


def write_frame_stats(path: Path, rows):
    ensure_dir(path.parent)
    fields = [
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
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, rows):
    ensure_dir(path.parent)
    fields = [
        "dataset",
        "sequence",
        "frames",
        "width",
        "height",
        "mean_alpha",
        "basic_video",
        "generative_video",
        "hybrid_video",
        "stats",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fuse_video_pair(
    basic_video: Path,
    gen_video: Path,
    out_video: Path,
    stats_path: Path,
    cfg: AdaptiveMaskConfig,
    fps: int,
):
    basic_info = probe_video(basic_video)
    gen_info = probe_video(gen_video)

    basic_reader = start_reader(basic_video, basic_info["width"], basic_info["height"])
    gen_reader = start_reader(gen_video, gen_info["width"], gen_info["height"])

    ensure_dir(out_video.parent)
    writer = FFmpegVideoWriter(str(out_video), (basic_info["width"], basic_info["height"]), fps=fps)
    rows = []
    frame_count = 0

    prev_basic = None
    prev_gen = None
    cur_basic = read_frame(basic_reader, basic_info["width"], basic_info["height"])
    cur_gen = read_frame(gen_reader, gen_info["width"], gen_info["height"])
    next_basic = read_frame(basic_reader, basic_info["width"], basic_info["height"])
    next_gen = read_frame(gen_reader, gen_info["width"], gen_info["height"])

    try:
        while cur_basic is not None and cur_gen is not None:
            frame_count += 1
            cur_gen = cur_gen.resize(cur_basic.size, Image.BICUBIC)
            next_gen_for_mask = next_gen.resize(cur_basic.size, Image.BICUBIC) if next_gen is not None else None
            prev_gen_for_mask = prev_gen.resize(cur_basic.size, Image.BICUBIC) if prev_gen is not None else None

            mask, _, stats = build_adaptive_alpha(
                basic=cur_basic,
                generative=cur_gen,
                previous_basic=prev_basic,
                next_basic=next_basic,
                cfg=cfg,
                previous_generative=prev_gen_for_mask,
                next_generative=next_gen_for_mask,
            )
            fused = blend_with_mask(cur_basic, cur_gen, mask)
            writer.append_pil(fused)

            stats["frame"] = f"{frame_count:08d}"
            rows.append(stats)

            prev_basic, prev_gen = cur_basic, cur_gen
            cur_basic, cur_gen = next_basic, next_gen
            next_basic = read_frame(basic_reader, basic_info["width"], basic_info["height"])
            next_gen = read_frame(gen_reader, gen_info["width"], gen_info["height"])
    finally:
        writer.close()
        if basic_reader.stdout:
            basic_reader.stdout.close()
        if gen_reader.stdout:
            gen_reader.stdout.close()
        basic_reader.wait()
        gen_reader.wait()

    write_frame_stats(stats_path, rows)
    mean_alpha = sum(float(row["mean_alpha"]) for row in rows) / max(len(rows), 1)
    return {
        "frames": frame_count,
        "width": basic_info["width"],
        "height": basic_info["height"],
        "mean_alpha": mean_alpha,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Direction C adaptive hybrid videos.")
    parser.add_argument("--basic-root", required=True)
    parser.add_argument("--generative-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--adaptive-config",
        default="/home/fc/Coding/CV/part3/configs/adaptive_hybrid_000_directionc_official.yaml",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    basic_root = Path(args.basic_root).resolve()
    gen_root = Path(args.generative_root).resolve()
    output_root = Path(args.output_root).resolve()
    cfg = adaptive_config(load_config(args.adaptive_config).get("adaptive", {}))

    basic_videos = iter_videos(basic_root)
    if args.max_seqs > 0:
        basic_videos = basic_videos[: args.max_seqs]
    if not basic_videos:
        raise ValueError(f"No sr.mp4 files found in {basic_root}")

    print("=" * 72)
    print("Batch Direction C Adaptive Hybrid")
    print("=" * 72)
    print(f"Dataset          : {args.dataset_name}")
    print(f"Basic root       : {basic_root}")
    print(f"Generative root  : {gen_root}")
    print(f"Output root      : {output_root}")
    print(f"Adaptive config  : {args.adaptive_config}")
    print(f"Sequences        : {len(basic_videos)}")
    print("=" * 72)

    manifest_rows = []
    for basic_video in tqdm(basic_videos, desc=args.dataset_name):
        rel = basic_video.relative_to(basic_root)
        gen_video = gen_root / rel
        if not gen_video.is_file():
            raise FileNotFoundError(f"Missing generative video for {rel}: {gen_video}")

        seq = rel.parent
        out_video = output_root / seq / "adaptive_hybrid.mp4"
        stats_path = output_root / seq / "frame_stats.csv"
        if args.force or not out_video.is_file() or not stats_path.is_file():
            info = fuse_video_pair(
                basic_video=basic_video,
                gen_video=gen_video,
                out_video=out_video,
                stats_path=stats_path,
                cfg=cfg,
                fps=args.fps,
            )
        else:
            info = probe_video(out_video)
            info["mean_alpha"] = ""

        manifest_rows.append({
            "dataset": args.dataset_name,
            "sequence": str(seq),
            "frames": info["frames"],
            "width": info["width"],
            "height": info["height"],
            "mean_alpha": info["mean_alpha"],
            "basic_video": str(basic_video),
            "generative_video": str(gen_video),
            "hybrid_video": str(out_video),
            "stats": str(stats_path),
        })

    write_manifest(output_root / "manifest.csv", manifest_rows)
    print(f"Saved manifest: {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
