import argparse
import csv
import os
import sys
from pathlib import Path

import yaml


CURRENT_DIR = Path(__file__).resolve().parent
PART3_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PART3_DIR / "src"))

from adaptive_mask import AdaptiveMaskConfig, build_adaptive_alpha
from fusion import blend_with_mask
from io_utils import (
    build_image_map,
    ensure_dir,
    extract_panel,
    iter_previous_current_next,
    load_rgb,
    paired_image_paths,
    resize_like,
    save_rgb,
)
from video_utils import images_to_video
from visualization import make_adaptive_grid


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def basic_panel(path: str, panel_index: int, panel_count: int):
    return extract_panel(load_rgb(path), panel_index=panel_index, panel_count=panel_count)


def adaptive_config(raw: dict) -> AdaptiveMaskConfig:
    return AdaptiveMaskConfig(
        mode=raw.get("mode", "auto"),
        anime_threshold=float(raw.get("anime_threshold", 0.50)),
        real_max_alpha=float(raw.get("real_max_alpha", 0.28)),
        anime_max_alpha=float(raw.get("anime_max_alpha", 0.62)),
        min_alpha=float(raw.get("min_alpha", 0.0)),
        texture_gain=float(raw.get("texture_gain", 1.10)),
        line_gain=float(raw.get("line_gain", 0.85)),
        flat_gain=float(raw.get("flat_gain", 0.45)),
        edge_protect_strength=float(raw.get("edge_protect_strength", 0.75)),
        temporal_protect_strength=float(raw.get("temporal_protect_strength", 0.60)),
        disagreement_protect_strength=float(raw.get("disagreement_protect_strength", 0.85)),
        structure_protect_strength=float(raw.get("structure_protect_strength", 1.00)),
        hallucination_protect_strength=float(raw.get("hallucination_protect_strength", 0.90)),
        flicker_protect_strength=float(raw.get("flicker_protect_strength", 0.80)),
        blur_radius=float(raw.get("mask_blur_radius", 4.0)),
        gamma=float(raw.get("gamma", 1.15)),
    )


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
    parser = argparse.ArgumentParser(description="Content-adaptive Direction C hybrid VSR.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    output_cfg = cfg.get("output", {})
    mask_cfg = adaptive_config(cfg.get("adaptive", {}))

    basic_dir = paths["basic_dir"]
    gen_dir = paths["generative_dir"]
    out_dir = paths["output_dir"]
    gen_suffix = paths.get("generative_suffix", "")
    panel_count = int(paths.get("basic_panel_count", 1))
    basic_panel_index = int(paths.get("basic_panel_index", 0))
    lr_panel_index = int(paths.get("lr_panel_index", 0))
    lr_dir = paths.get("lr_dir")
    lr_map = build_image_map(lr_dir) if lr_dir else None

    frame_dir = os.path.join(out_dir, "frames")
    basic_frame_dir = os.path.join(out_dir, "basic_frames")
    gen_frame_dir = os.path.join(out_dir, "generative_frames")
    mask_dir = os.path.join(out_dir, "masks")
    grid_dir = os.path.join(out_dir, "grids")
    map_root = os.path.join(out_dir, "maps")
    video_dir = os.path.join(out_dir, "videos")
    for folder in [frame_dir, basic_frame_dir, gen_frame_dir, mask_dir, grid_dir, map_root, video_dir]:
        ensure_dir(folder)

    pairs = paired_image_paths(
        basic_dir=basic_dir,
        generative_dir=gen_dir,
        generative_suffix=gen_suffix,
        max_frames=args.max_frames,
    )

    print("=" * 72)
    print("Part 3 Direction C: Content-Adaptive Hybrid VSR")
    print("=" * 72)
    print(f"Basic branch     : {basic_dir}")
    print(f"Generative branch: {gen_dir}")
    print(f"Output           : {out_dir}")
    print(f"Mode             : {mask_cfg.mode}")
    print(f"Frames           : {len(pairs)}")
    print("=" * 72)

    saved_frames = []
    saved_grids = []
    stats_rows = []

    for idx, (prev_item, item, next_item) in enumerate(iter_previous_current_next(pairs), 1):
        key, basic_path, gen_path = item
        basic = basic_panel(basic_path, panel_index=basic_panel_index, panel_count=panel_count)
        if lr_map is not None and key in lr_map:
            lr_up = resize_like(load_rgb(lr_map[key]), basic)
        else:
            lr_up = basic_panel(basic_path, panel_index=lr_panel_index, panel_count=panel_count)
        generative = resize_like(load_rgb(gen_path), basic)

        previous_basic = None
        next_basic = None
        previous_generative = None
        next_generative = None
        if prev_item is not None:
            previous_basic = basic_panel(prev_item[1], panel_index=basic_panel_index, panel_count=panel_count)
            previous_generative = resize_like(load_rgb(prev_item[2]), basic)
        if next_item is not None:
            next_basic = basic_panel(next_item[1], panel_index=basic_panel_index, panel_count=panel_count)
            next_generative = resize_like(load_rgb(next_item[2]), basic)

        mask, maps, stats = build_adaptive_alpha(
            basic=basic,
            generative=generative,
            previous_basic=previous_basic,
            next_basic=next_basic,
            cfg=mask_cfg,
            previous_generative=previous_generative,
            next_generative=next_generative,
        )
        fused = blend_with_mask(basic, generative, mask)

        frame_path = os.path.join(frame_dir, f"{key}.png")
        basic_path_out = os.path.join(basic_frame_dir, f"{key}.png")
        gen_path_out = os.path.join(gen_frame_dir, f"{key}.png")
        mask_path = os.path.join(mask_dir, f"{key}.png")

        save_rgb(fused, frame_path)
        save_rgb(basic, basic_path_out)
        save_rgb(generative, gen_path_out)
        mask.save(mask_path)
        saved_frames.append(frame_path)

        if bool(output_cfg.get("save_maps", True)):
            for map_name, map_image in maps.items():
                map_dir = os.path.join(map_root, map_name)
                ensure_dir(map_dir)
                map_image.save(os.path.join(map_dir, f"{key}.png"))

        if bool(output_cfg.get("save_grids", True)):
            grid = make_adaptive_grid(lr_up, basic, generative, mask, fused, maps)
            grid_path = os.path.join(grid_dir, f"{key}.png")
            save_rgb(grid, grid_path)
            saved_grids.append(grid_path)

        stats["frame"] = key
        stats_rows.append(stats)

        if idx == 1 or idx % 25 == 0 or idx == len(pairs):
            print(
                f"[{idx}/{len(pairs)}] {key}: "
                f"content={stats['content_label']}, "
                f"anime_score={stats['anime_score']:.3f}, "
                f"mean_alpha={stats['mean_alpha']:.3f}"
            )

    write_stats(os.path.join(out_dir, "frame_stats.csv"), stats_rows)

    if bool(output_cfg.get("export_video", True)):
        fps = int(output_cfg.get("fps", 30))
        images_to_video(saved_frames, os.path.join(video_dir, "adaptive_hybrid.mp4"), fps=fps)
        if saved_grids:
            images_to_video(saved_grids, os.path.join(video_dir, "adaptive_grid.mp4"), fps=fps)
        print(f"Saved videos to: {video_dir}")

    print(f"Saved fused frames to: {frame_dir}")
    print(f"Saved masks to: {mask_dir}")
    print(f"Saved frame stats to: {os.path.join(out_dir, 'frame_stats.csv')}")


if __name__ == "__main__":
    main()
