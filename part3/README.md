# Part 3: Direction C - Content-Adaptive Hybrid VSR

This part implements the guideline's Direction C: an uncertainty-aware hybrid pipeline.

Our empirical observation is:

- BasicVSR++ is stronger on real-world videos because it preserves temporal consistency and natural image fidelity.
- Real-ESRGAN is stronger on anime/line-art style videos because it produces sharper contours and cleaner flat-color regions.
- Real-ESRGAN can hallucinate unrealistic textures on real scenes, so it should not globally replace BasicVSR++.

The final Part 3 pipeline therefore uses BasicVSR++ as the reliable reconstruction branch and Real-ESRGAN as a local detail branch. A content-adaptive uncertainty mask decides where Real-ESRGAN is allowed to contribute.

## Method

For each frame, the adaptive pipeline computes:

- `anime_score`: estimates whether the frame is closer to anime/line-art or real-world content.
- `texture map`: identifies regions where generative details may help.
- `edge/flat-color maps`: identify anime-like line art and color blocks.
- `temporal change map`: suppresses Real-ESRGAN in unstable/moving regions.
- `branch disagreement map`: suppresses Real-ESRGAN where it deviates too much from BasicVSR++.

The final alpha mask is conservative for real-world videos and stronger for anime-like videos. This keeps the fidelity and temporal stability of BasicVSR++ while selectively borrowing the sharper local appearance of Real-ESRGAN.

## Main Command

Run the conservative final configuration:

```bash
cd /home/fc/Coding/CV
python part3/scripts/run_adaptive_hybrid.py \
  --config part3/configs/adaptive_hybrid_000_conservative.yaml
```

Evaluate PSNR/SSIM:

```bash
cd /home/fc/Coding/CV
.venv/bin/python part3/scripts/evaluate_part3.py \
  --config part3/configs/eval_adaptive_hybrid_000_conservative.yaml
```

Outputs:

- fused frames: `part3/results/adaptive_hybrid_000_conservative/frames`
- BasicVSR++ frames: `part3/results/adaptive_hybrid_000_conservative/basic_frames`
- Real-ESRGAN frames: `part3/results/adaptive_hybrid_000_conservative/generative_frames`
- alpha masks: `part3/results/adaptive_hybrid_000_conservative/masks`
- diagnostic maps: `part3/results/adaptive_hybrid_000_conservative/maps`
- comparison grids: `part3/results/adaptive_hybrid_000_conservative/grids`
- videos: `part3/results/adaptive_hybrid_000_conservative/videos`
- frame-level mask statistics: `part3/results/adaptive_hybrid_000_conservative/frame_stats.csv`

## Current Result on Validation Sequence 000

| Method | Frames | PSNR | SSIM |
|---|---:|---:|---:|
| BasicVSR++ | 100 | 31.4585 | 0.8871 |
| Real-ESRGAN | 100 | 27.8395 | 0.7877 |
| Adaptive-Hybrid-Conservative | 100 | 31.4529 | 0.8861 |

The conservative hybrid nearly preserves the pixel metrics of BasicVSR++ while adding controlled Real-ESRGAN detail in high-confidence regions. This is the recommended result for the final report because it demonstrates the intended Direction C behavior without sacrificing fidelity.

## Other Variants

The less conservative adaptive configuration is:

```bash
python part3/scripts/run_adaptive_hybrid.py \
  --config part3/configs/adaptive_hybrid_000.yaml
```

It uses a stronger Real-ESRGAN contribution:

| Method | Frames | PSNR | SSIM |
|---|---:|---:|---:|
| BasicVSR++ | 100 | 31.4585 | 0.8871 |
| Real-ESRGAN | 100 | 27.8395 | 0.7877 |
| Adaptive-Hybrid | 100 | 31.3046 | 0.8792 |

The older `run_hybrid_fusion.py` and ControlNet-Tile scripts are kept for ablation. They are useful for discussing why unrestricted diffusion-style enhancement can reduce PSNR/SSIM and introduce hallucination.
