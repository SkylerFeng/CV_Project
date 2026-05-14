import argparse
import os
import sys
from pathlib import Path

import yaml


CURRENT_DIR = Path(__file__).resolve().parent
PART3_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(PART3_DIR / "src"))

from fusion import blend_with_mask, make_grid
from io_utils import (
    ensure_dir,
    extract_panel,
    iter_previous_current_next,
    load_rgb,
    paired_image_paths,
    resize_like,
    save_rgb,
)
from mask import build_uncertainty_mask
from video_utils import images_to_video


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_basic_frame(path: str, panel_index: int, panel_count: int):
    return extract_panel(load_rgb(path), panel_index=panel_index, panel_count=panel_count)


def main():
    parser = argparse.ArgumentParser(description="Uncertainty-aware hybrid fusion for Part 3.")
    parser.add_argument("--config", type=str, required=True, help="Path to hybrid_fusion.yaml")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    fusion_cfg = cfg.get("fusion", {})

    basic_dir = paths["basic_dir"]
    gen_dir = paths["generative_dir"]
    out_dir = paths["output_dir"]
    gen_suffix = paths.get("generative_suffix", "")
    panel_index = int(paths.get("basic_panel_index", 0))
    panel_count = int(paths.get("basic_panel_count", 1))

    frame_dir = os.path.join(out_dir, "frames")
    basic_frame_dir = os.path.join(out_dir, "basic_frames")
    mask_dir = os.path.join(out_dir, "masks")
    grid_dir = os.path.join(out_dir, "grids")
    video_dir = os.path.join(out_dir, "videos")

    for folder in [frame_dir, basic_frame_dir, mask_dir, grid_dir, video_dir]:
        ensure_dir(folder)

    pairs = paired_image_paths(
        basic_dir=basic_dir,
        generative_dir=gen_dir,
        generative_suffix=gen_suffix,
        max_frames=args.max_frames,
    )

    print("=" * 72)
    print("Part 3 Hybrid Fusion")
    print("=" * 72)
    print(f"Basic dir      : {basic_dir}")
    print(f"Generative dir : {gen_dir}")
    print(f"Output dir     : {out_dir}")
    print(f"Frames         : {len(pairs)}")
    print("=" * 72)

    saved_frames = []
    saved_grids = []

    for idx, (prev_item, item, next_item) in enumerate(iter_previous_current_next(pairs), 1):
        key, basic_path, gen_path = item
        basic = maybe_basic_frame(basic_path, panel_index=panel_index, panel_count=panel_count)
        generative = resize_like(load_rgb(gen_path), basic)

        previous_basic = None
        next_basic = None
        if prev_item is not None:
            previous_basic = maybe_basic_frame(prev_item[1], panel_index=panel_index, panel_count=panel_count)
        if next_item is not None:
            next_basic = maybe_basic_frame(next_item[1], panel_index=panel_index, panel_count=panel_count)

        mask = build_uncertainty_mask(
            basic=basic,
            previous_basic=previous_basic,
            next_basic=next_basic,
            texture_gain=float(fusion_cfg.get("texture_gain", 1.25)),
            edge_protect_strength=float(fusion_cfg.get("edge_protect_strength", 1.15)),
            temporal_protect_strength=float(fusion_cfg.get("temporal_protect_strength", 0.50)),
            min_alpha=float(fusion_cfg.get("min_alpha", 0.0)),
            max_alpha=float(fusion_cfg.get("max_alpha", 0.55)),
            blur_radius=float(fusion_cfg.get("mask_blur_radius", 5.0)),
        )
        fused = blend_with_mask(basic, generative, mask)

        frame_path = os.path.join(frame_dir, f"{key}.png")
        basic_frame_path = os.path.join(basic_frame_dir, f"{key}.png")
        mask_path = os.path.join(mask_dir, f"{key}.png")
        grid_path = os.path.join(grid_dir, f"{key}.png")

        save_rgb(fused, frame_path)
        save_rgb(basic, basic_frame_path)
        mask.save(mask_path)
        saved_frames.append(frame_path)

        if bool(fusion_cfg.get("save_grids", True)):
            grid = make_grid(basic, generative, mask, fused)
            save_rgb(grid, grid_path)
            saved_grids.append(grid_path)

        if idx == 1 or idx % 25 == 0 or idx == len(pairs):
            print(f"[{idx}/{len(pairs)}] {key}")

    if bool(fusion_cfg.get("export_video", True)):
        fps = int(fusion_cfg.get("fps", 30))
        images_to_video(saved_frames, os.path.join(video_dir, "hybrid.mp4"), fps=fps)
        if saved_grids:
            images_to_video(saved_grids, os.path.join(video_dir, "comparison_grid.mp4"), fps=fps)
        print(f"Saved videos to: {video_dir}")

    print(f"Saved fused frames to: {frame_dir}")
    print(f"Saved masks to: {mask_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

