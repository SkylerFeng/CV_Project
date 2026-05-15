# Results Manifest

This file records report-ready results and their local paths. Large result folders are ignored by git, so this manifest is the stable reference for the report.

## Part 1

Full validation PSNR/SSIM:

```text
part1/outputs/metrics_val_part1/summary.csv
```

Reported values:

| Method | Frames | PSNR | SSIM |
|---|---:|---:|---:|
| Bicubic | 3000 | 26.2949 | 0.7233 |
| Lanczos | 3000 | 26.5041 | 0.7302 |
| SRCNN | 3000 | 27.3264 | 0.7655 |
| Temporal | 3000 | 23.1046 | 0.6403 |

## Part 2.1 BasicVSR++

Full / partial validation metrics:

```text
part2_1/results/metrics_val_basicvsrpp_current/summary.csv
part2_1/results/metrics_val_basicvsrpp_final/summary.csv
```

Key sequence-level comparisons:

```text
part2_1/results/metrics_val000_basicvsrpp_original/summary.csv
part2_1/results/metrics_val000_basicvsrpp_finetuned_ssim/summary.csv
```

## Part 2.2 Real-ESRGAN

Official full validation metrics:

```text
part2_2/results/metrics_val_realesrgan_x4plus_full/summary.csv
```

Val000 original vs fine-tuned comparison:

| Model | Frames | PSNR | SSIM |
|---|---:|---:|---:|
| Official Real-ESRGAN | 100 | 24.1037 | 0.6696 |
| Conservative fine-tuned Real-ESRGAN | 100 | 27.1930 | 0.7697 |

Paths:

```text
part2_2/results/metrics_val000_official_x4plus_original_eval
part2_2/results/metrics_val000_conservative_4epoch_best_eval
```

Fine-tuned checkpoint:

```text
part2_2/experiments/realesrgan_x4plus_official_conservative_4epoch/checkpoints/net_g_best.pth
```

## Part 3 Direction C

### Validation 000 Official Real-ESRGAN

Metrics:

```text
part3/results/metrics_adaptive_hybrid_000_directionc_official
part3/results/lpips_temporal_adaptive_hybrid_000_directionc_official
```

Key values:

| Method | PSNR | SSIM | LPIPS | tLPIPS proxy |
|---|---:|---:|---:|---:|
| BasicVSR++ | 31.4585 | 0.8871 | 0.1590 | 0.1541 |
| Official Real-ESRGAN | 24.1239 | 0.6656 | 0.2537 | 0.2477 |
| Direction C Hybrid | 31.3772 | 0.8860 | 0.1610 | 0.1565 |

Showcase:

```text
part3/results/adaptive_hybrid_000_directionc_official/showcase_00000049.png
```

### Val000-006 Conservative vs Anime-stronger

Conservative:

```text
part3/results/adaptive_hybrid_val000_006_official_conservative_backup
part3/results/metrics_adaptive_hybrid_val000_006_official_conservative_backup/summary.csv
```

Anime stronger:

```text
part3/results/adaptive_hybrid_val000_006_official_anime_stronger
part3/results/metrics_adaptive_hybrid_val000_006_official_anime_stronger/summary.csv
```

Summary:

| Variant | Frames | PSNR | SSIM | Mean alpha |
|---|---:|---:|---:|---|
| conservative | 700 | 32.8052 | 0.9075 | about 0.11-0.15 |
| anime stronger | 700 | 32.1390 | 0.8934 | about 0.29-0.32 |

Showcase examples:

```text
part3/results/adaptive_hybrid_val000_006_official_conservative_backup/000/showcase_00000049.png
part3/results/adaptive_hybrid_val000_006_official_anime_stronger/000/showcase_00000049.png
```

## Recommended Report Figures

```text
figures/part1_baseline_comparison.png
figures/part2_basicvsrpp_vs_realesrgan.png
figures/part3_mask_basic_real_hybrid.png
```

Additional Direction C showcase panels are under each Part 3 result directory.
