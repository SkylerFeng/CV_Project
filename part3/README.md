# Part 3: Direction C Adaptive Hybrid Pipeline

Part 3 implements Direction C from the guideline:

```text
Uncertainty-Aware Refinement / Adaptive Hybrid Pipeline
```

The goal is to identify limitations of generative enhancement, especially hallucination artifacts and temporal flickering, and attempt an improvement by selectively fusing BasicVSR++ and Real-ESRGAN.

## Core Idea

BasicVSR++ is the reliable reconstruction branch:

- stronger PSNR/SSIM;
- better temporal stability;
- better for reliable structures and real-world videos;
- sometimes visually soft.

Real-ESRGAN is the generative enhancement branch:

- sharper local detail;
- better for anime or line-art content;
- can hallucinate textures;
- can flicker when applied frame by frame.

The final hybrid result is:

```text
I_hybrid = (1 - alpha) * I_BasicVSR++ + alpha * I_RealESRGAN
```

The alpha map is a heuristic proxy uncertainty/risk map. It is not a learned probabilistic uncertainty estimator.

## Files

```text
part3/
  configs/
    adaptive_hybrid_val000_finetuned_report.yaml
    eval_adaptive_hybrid_val000_finetuned_report.yaml
    directionc_val000_006_batch.yaml
    perceptual300_conservative.yaml
  scripts/
    run_adaptive_hybrid.py
    run_adaptive_hybrid_stream_video.py
    run_directionc_batch.py
    batch_adaptive_hybrid_videos.py
    evaluate_part3.py
    make_directionc_showcase.py
    run_perceptual300_hybrid.py
  src/
    adaptive_mask.py
    fusion.py
    io_utils.py
    video_utils.py
    visualization.py
```

## Diagnostic Maps

The pipeline can save:

- `alpha mask`: final Real-ESRGAN blending weight;
- `structure protect`: regions where BasicVSR++ should dominate;
- `uncertain texture`: regions where generative detail may help;
- `hallucination risk`: regions where Real-ESRGAN may be unfaithful;
- `flicker risk`: regions where frame-wise generative enhancement may be unstable.

These maps make the hybrid decision interpretable.

## Environment

Use the general environment:

```bash
.venv/bin/python
```

Part 3 assumes BasicVSR++ and Real-ESRGAN branch outputs already exist or are specified in the config.

## Run One Report Experiment

```bash
.venv/bin/python part3/scripts/run_adaptive_hybrid.py \
  --config part3/configs/adaptive_hybrid_val000_finetuned_report.yaml
```

Expected output:

```text
part3/results/adaptive_hybrid_val000_finetuned_report/
  frames/
  masks/
  maps/
  grids/
  videos/
  frame_stats.csv
```

The `grids/` folder contains visual comparison grids for checking BasicVSR++, Real-ESRGAN, alpha mask, and hybrid output frame by frame.

## Evaluate PSNR/SSIM

```bash
.venv/bin/python part3/scripts/evaluate_part3.py \
  --config part3/configs/eval_adaptive_hybrid_val000_finetuned_report.yaml
```

Expected output:

```text
part3/results/metrics_adaptive_hybrid_val000_finetuned_report/
  summary.csv
  summary.md
  per_frame.csv
```

Example report result on validation sequence 000:

| Method | Frames | PSNR | SSIM |
|---|---:|---:|---:|
| Fine-tuned BasicVSR++ | 100 | 30.5047 | 0.8642 |
| Fine-tuned Real-ESRGAN | 100 | 27.2124 | 0.7666 |
| Direction C Hybrid | 100 | 30.4978 | 0.8637 |

## Run Direction C Batch Variants

Conservative variant:

```bash
.venv/bin/python part3/scripts/run_directionc_batch.py \
  --variant conservative \
  --run-hybrid \
  --evaluate \
  --showcase
```

Anime-stronger variant:

```bash
.venv/bin/python part3/scripts/run_directionc_batch.py \
  --variant anime_stronger \
  --run-hybrid \
  --evaluate \
  --showcase
```

Final validation 000-006 ablation:

| Variant | Frames | PSNR | SSIM | Mean alpha |
|---|---:|---:|---:|---|
| Conservative | 700 | 32.8052 | 0.9075 | about 0.11-0.15 |
| Anime-stronger | 700 | 32.1390 | 0.8934 | about 0.29-0.32 |

Use the conservative result as the formal validation setting. Use anime-stronger as an ablation and qualitative setting for anime content.

## Generate Showcase Figure

```bash
.venv/bin/python part3/scripts/make_directionc_showcase.py \
  --result-dir part3/results/adaptive_hybrid_val000_finetuned_report \
  --frame 00000049
```

Report-ready figure:

```text
figures/part3_directionc_finetuned_showcase.png
```

The showcase includes:

```text
BasicVSR++ / Real-ESRGAN / Direction C Hybrid / Alpha mask
Structure protect / Uncertain texture / Hallucination risk / Flicker risk
```

## Batch No-Reference Hybrid Videos

After Part 2 has generated BasicVSR++ and Real-ESRGAN videos, run:

```bash
.venv/bin/python part3/scripts/batch_adaptive_hybrid_videos.py \
  --basic-root part2_extra_outputs/basicvsrpp_finetuned/REDS-sample \
  --generative-root part2_extra_outputs/realesrgan_finetuned/REDS-sample \
  --output-root part3_extra_outputs/directionc_hybrid_official/REDS-sample \
  --dataset-name REDS-sample \
  --fps 30
```

Quick test:

```bash
.venv/bin/python part3/scripts/batch_adaptive_hybrid_videos.py \
  --basic-root part2_extra_outputs/basicvsrpp_finetuned/REDS-sample \
  --generative-root part2_extra_outputs/realesrgan_finetuned/REDS-sample \
  --output-root part3_extra_outputs/directionc_hybrid_debug/REDS-sample \
  --dataset-name REDS-sample \
  --fps 30 \
  --max-seqs 1
```

Repeat with `vimeo-RL` paths if needed.

## LPIPS / FID / tLPIPS-Proxy on 300 Frames

Generate the 300-frame hybrid set:

```bash
.venv/bin/python part3/scripts/run_perceptual300_hybrid.py
```

Evaluate perceptual and temporal metrics:

```bash
.venv/bin/python scripts/evaluate_perceptual_fid_temporal.py \
  --gt part3/results/perceptual300_evalset/gt \
  --method "Fine-tuned BasicVSR++=part3/results/perceptual300_evalset/basicvsrpp" \
  --method "Fine-tuned Real-ESRGAN=part3/results/perceptual300_evalset/realesrgan" \
  --method "Direction C Hybrid=part3/results/perceptual300_evalset/hybrid" \
  --output part3/results/metrics_perceptual300 \
  --device cuda
```

Report output:

```text
part3/results/metrics_perceptual300/summary.csv
figures/part3_perceptual300_summary.csv
figures/part3_perceptual300_summary.md
```

Result:

| Method | Frames | LPIPS | FID | tMSE | tLPIPS-proxy |
|---|---:|---:|---:|---:|---:|
| Fine-tuned BasicVSR++ | 300 | 0.1531 | 9.27 | 0.000369 | 0.1447 |
| Fine-tuned Real-ESRGAN | 300 | 0.2669 | 21.80 | 0.000849 | 0.2600 |
| Direction C Hybrid | 300 | 0.1550 | 9.46 | 0.000370 | 0.1470 |

Important: this is a tLPIPS-style proxy, not official flow-warped tLPIPS.

## Interpretation

Use Part 3 to argue:

- standalone Real-ESRGAN is sharp but risky;
- BasicVSR++ is stable and faithful;
- the hybrid mask keeps the result close to BasicVSR++ where structure or temporal stability matters;
- Real-ESRGAN is only allowed to contribute in uncertain texture regions;
- conservative and anime-stronger settings show content-dependent fusion behavior.

## Report Wording

Use:

```text
proxy uncertainty/risk map
tLPIPS-proxy
content-dependent alpha setting
```

Avoid:

```text
learned uncertainty
official tLPIPS
PSNR/SSIM for no-reference videos
```

