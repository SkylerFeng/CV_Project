# Content-Adaptive Hybrid Video Super-Resolution with BasicVSR++ and Real-ESRGAN

This file is a working report draft. It is written in a paper-like style so that each section can be moved into the CVPR LaTeX template later.

## Figure and Table Checklist

Figures to include:

| ID | Content | Local path |
|---|---|---|
| Fig. 1 | Overall pipeline: BasicVSR++ branch, Real-ESRGAN branch, proxy uncertainty mask, hybrid fusion | create from method diagram |
| Fig. 2 | Part 1 baseline comparison | `figures/part1_baseline_comparison.png` |
| Fig. 3 | Part 2 BasicVSR++ vs Real-ESRGAN qualitative comparison | `figures/part2_basicvsrpp_vs_realesrgan.png` |
| Fig. 4 | Part 3 Direction C masks and hybrid result | `figures/part3_mask_basic_real_hybrid.png` |
| Fig. 5 | Full Direction C showcase | `part3/results/adaptive_hybrid_000_directionc_official/showcase_00000049.png` |
| Fig. 6 | Conservative vs anime-stronger qualitative comparison | `part3/results/adaptive_hybrid_val000_006_official_conservative_backup/000/showcase_00000049.png`, `part3/results/adaptive_hybrid_val000_006_official_anime_stronger/000/showcase_00000049.png` |

Tables to include:

| ID | Content | Source |
|---|---|---|
| Table 1 | Part 1 full validation PSNR/SSIM | `part1/outputs/metrics_val_part1/summary.csv` |
| Table 2 | Part 2 Real-ESRGAN official vs fine-tuned on val000 | `docs/results_manifest.md` |
| Table 3 | Direction C val000 PSNR/SSIM/LPIPS/tLPIPS proxy | `docs/results_manifest.md` |
| Table 4 | Direction C val000-006 conservative vs anime-stronger | `docs/results_manifest.md` |

## Abstract

Video super-resolution aims to recover high-resolution video frames from low-resolution inputs while preserving both spatial details and temporal consistency. In this project, we first implement classical and CNN-based baselines, including bicubic interpolation, Lanczos interpolation, SRCNN, and a simple temporal baseline. We then study two stronger restoration branches: BasicVSR++, which provides temporally stable and high-fidelity reconstruction, and Real-ESRGAN, which produces sharper perceptual details but may introduce hallucinated textures and frame-wise instability.

Motivated by this fidelity-perception trade-off, we develop a Direction C adaptive hybrid pipeline. The method estimates a heuristic proxy pixel-wise uncertainty and risk map from texture strength, structure protection, branch disagreement, hallucination risk, and flicker risk. The final output selectively uses BasicVSR++ in reliable structure regions and introduces Real-ESRGAN only in uncertain texture regions. On validation sequence 000, the hybrid result achieves PSNR/SSIM close to BasicVSR++ while substantially avoiding the degradation of standalone Real-ESRGAN. On validation sequences 000-006, the conservative hybrid variant obtains 32.8052 PSNR and 0.9075 SSIM, while a stronger anime-style variant increases Real-ESRGAN contribution at the cost of lower fidelity metrics. Code and results are available at: `[GitHub link]`.

## 1. Introduction

Video super-resolution (VSR) is more challenging than single-image super-resolution because the restored video must be both spatially detailed and temporally coherent. A method that sharpens each frame independently can improve local texture appearance, but it may also create inconsistent details across frames. Conversely, a conservative temporal model can produce stable videos but may look blurry on fine textures and line-art content.

This project follows the three-part structure of the assignment. In Part 1, we establish simple baselines using interpolation, SRCNN, and temporal processing. In Part 2, we reproduce and evaluate BasicVSR++ and Real-ESRGAN. BasicVSR++ is used as the reliable temporal reconstruction branch, while Real-ESRGAN is studied as a perceptual enhancement branch. Our experiments show a clear trade-off: BasicVSR++ achieves better PSNR/SSIM and temporal stability on real videos, whereas Real-ESRGAN often produces sharper edges and cleaner line art but may hallucinate details on real-world content.

Based on this observation, Part 3 implements the guideline's Direction C: uncertainty-aware refinement. Instead of replacing BasicVSR++ globally with a generative model, we compute a proxy uncertainty/risk map and fuse the two branches spatially. The main contributions are:

1. A complete VSR evaluation pipeline covering classical baselines, BasicVSR++, Real-ESRGAN, and real custom videos.
2. A conservative Real-ESRGAN fine-tuning setup using reconstruction, perceptual, edge, Laplacian, color, and initial-weight regularization terms.
3. A Direction C hybrid system that identifies hallucination and flicker risks and selectively applies generative enhancement in uncertain texture regions.
4. Quantitative evaluation with PSNR/SSIM and perceptual/temporal proxy metrics, plus qualitative diagnostic maps for interpretability.

## 2. Related Work

Classical interpolation methods such as bicubic and Lanczos are fast and deterministic, but they cannot recover missing high-frequency details. SRCNN introduced an early CNN-based approach for image super-resolution, showing that learned mappings from low-resolution to high-resolution images can outperform handcrafted interpolation. Later image SR models such as SRGAN and EDSR further improved perceptual quality and reconstruction accuracy.

For video super-resolution, temporal information is essential. EDVR and TDAN use alignment mechanisms to aggregate neighboring frames, while BasicVSR and BasicVSR++ improve temporal propagation through recurrent feature propagation and refinement. These models are strong choices for faithful reconstruction because they use information across frames rather than enhancing each frame independently.

Generative super-resolution methods, including Real-ESRGAN and diffusion-based approaches such as SR3, focus on perceptual realism and sharp detail synthesis. These models can produce visually sharper results, but their generated textures are not always faithful to the ground truth. In video settings, hallucinated details can also vary between frames, causing temporal flickering. This motivates hybrid strategies that combine reliable temporal reconstruction with controlled generative refinement.

## 3. Method

### 3.1 Part 1 Baselines

We evaluate four baseline methods. Bicubic and Lanczos interpolation directly upsample each low-resolution frame. SRCNN performs learned single-image super-resolution. The temporal baseline applies simple frame-level temporal processing to test whether neighboring-frame information helps without a modern VSR architecture.

These baselines are useful for establishing a lower bound. They are simple and efficient, but they generally produce blurred textures and lack robust temporal modeling.

### 3.2 BasicVSR++ Reconstruction Branch

BasicVSR++ is used as the main temporally stable reconstruction branch. Given a low-resolution input video, the model propagates information across frames and reconstructs high-resolution outputs with strong structural consistency. In our pipeline, BasicVSR++ is preferred for reliable regions such as object boundaries, text-like structures, faces, and areas where frame-to-frame consistency is more important than aggressive texture synthesis.

Implementation details:

- Main inference script: `part2_1/scripts/infer_basicvsrpp_video.py`
- Full validation evaluation: `part2_1/scripts/evaluate_basicvsrpp_val.py`
- Environment: `mmagic_env`
- Large-video inference uses chunking, optional FP16, and direct video writing to reduce memory pressure.

### 3.3 Real-ESRGAN Perceptual Branch

Real-ESRGAN is used as the perceptual/generative enhancement branch. It can sharpen edges and synthesize high-frequency details, especially for anime or line-art style content. However, when applied to real videos, it may introduce artificial textures, block-like artifacts, color shifts, or inconsistent details across frames.

We evaluate both the official Real-ESRGAN weights and a conservative fine-tuned model. The final fine-tuning configuration is:

```text
part2_2/configs/train_conservative.yaml
```

The fine-tuning objective combines:

```text
Charbonnier + perceptual + edge + Laplacian + color + initial-weight regularization
```

The purpose of this fine-tuning is not to make Real-ESRGAN dominate the final system, but to improve its fidelity when used as a candidate generative branch.

### 3.4 Direction C Adaptive Hybrid

The final Part 3 method follows Direction C from the guideline. The hybrid output is computed as:

```text
I_hybrid = (1 - alpha) * I_BasicVSR++ + alpha * I_RealESRGAN
```

Here, `alpha` is a proxy pixel-wise uncertainty/risk map. It is important that this is not a learned probabilistic uncertainty estimator. Instead, it is a heuristic diagnostic map computed from several interpretable signals:

- Uncertain texture: encourages Real-ESRGAN in textured regions where detail synthesis may help.
- Structure protect: suppresses Real-ESRGAN around important edges, text-like strokes, and smooth structure regions.
- Branch disagreement: reduces Real-ESRGAN contribution when the two branches differ strongly.
- Hallucination risk: detects regions where the generative branch may introduce unfaithful details.
- Flicker risk: penalizes regions with unstable frame-to-frame generative changes.

This design directly targets the observed limitations of generative SR: hallucination artifacts and temporal flickering.

## 4. Experiments

### 4.1 Datasets and Inputs

We use the provided training and validation sequences, including low-resolution bicubic inputs under:

```text
data/val/val_sharp_bicubic/X4
data/val/val_sharp
```

For real-video analysis, we also process custom videos such as:

```text
data/custom/3.mp4
data/custom/4.mp4
data/custom/wild_01.mp4
data/custom/wild_05
```

The validation set provides ground-truth frames, so PSNR and SSIM can be computed directly. For custom real videos without native high-resolution ground truth, we use qualitative comparison and, where applicable, resized references only as approximate fidelity checks.

### 4.2 Metrics

We report PSNR and SSIM as distortion/fidelity metrics. These metrics reward pixel-level similarity to the ground truth and are useful for comparing reconstruction accuracy.

We also report LPIPS and a simplified tLPIPS-style temporal proxy on selected Part 3 outputs. LPIPS measures perceptual distance in a learned feature space. The temporal proxy compares perceptual changes across adjacent frames and is used as an approximate indicator of temporal instability. Since this proxy does not use optical-flow warping, it should be described as a simplified temporal perceptual metric rather than a full official tLPIPS implementation.

### 4.3 Part 1 Results

Table 1 reports the full validation results for the Part 1 baselines.

| Method | Frames | PSNR | SSIM |
|---|---:|---:|---:|
| Bicubic | 3000 | 26.2949 | 0.7233 |
| Lanczos | 3000 | 26.5041 | 0.7302 |
| SRCNN | 3000 | 27.3264 | 0.7655 |
| Temporal | 3000 | 23.1046 | 0.6403 |

SRCNN improves over bicubic and Lanczos, confirming that learned image SR can recover more structure than fixed interpolation. However, all Part 1 methods remain limited compared with modern VSR models. The temporal baseline performs worse in PSNR/SSIM, suggesting that naive temporal processing can blur or misalign content instead of improving reconstruction.

Insert Fig. 2 here: `figures/part1_baseline_comparison.png`.

### 4.4 Part 2 Results

BasicVSR++ provides the strongest reliable reconstruction branch. On validation data, it achieves substantially higher PSNR/SSIM than standalone Real-ESRGAN. This agrees with visual inspection: BasicVSR++ is more stable and faithful on real-world videos, but it can look soft on anime-style line art.

Real-ESRGAN behaves differently. It often looks sharper, especially for anime and line-art content, but its pixel-level fidelity is lower. On val000, the official and fine-tuned Real-ESRGAN comparison is:

| Model | Frames | PSNR | SSIM |
|---|---:|---:|---:|
| Official Real-ESRGAN | 100 | 24.1037 | 0.6696 |
| Conservative fine-tuned Real-ESRGAN | 100 | 27.1930 | 0.7697 |

The fine-tuned Real-ESRGAN improves PSNR and SSIM over the official model, showing that conservative fine-tuning makes the generative branch more faithful. Nevertheless, even the improved Real-ESRGAN remains less reliable than BasicVSR++ for distortion metrics, which motivates a controlled hybrid rather than a global replacement.

Insert Fig. 3 here: `figures/part2_basicvsrpp_vs_realesrgan.png`.

### 4.5 Part 3 Direction C Results

On validation sequence 000, we compare BasicVSR++, official Real-ESRGAN, and the Direction C hybrid.

| Method | PSNR | SSIM | LPIPS ↓ | tLPIPS proxy ↓ |
|---|---:|---:|---:|---:|
| BasicVSR++ | 31.4585 | 0.8871 | 0.1590 | 0.1541 |
| Official Real-ESRGAN | 24.1239 | 0.6656 | 0.2537 | 0.2477 |
| Direction C Hybrid | 31.3772 | 0.8860 | 0.1610 | 0.1565 |

The official Real-ESRGAN branch alone has much lower PSNR/SSIM and worse LPIPS/tLPIPS proxy, confirming that unrestricted generative enhancement is risky on this real validation sequence. The Direction C hybrid stays close to BasicVSR++ in all metrics. This indicates that the alpha mask successfully limits the generative branch to selected regions instead of allowing it to dominate the output.

Insert Fig. 4 here: `figures/part3_mask_basic_real_hybrid.png`.

We also evaluate two parameter variants on validation sequences 000-006:

| Variant | Frames | PSNR | SSIM | Mean alpha |
|---|---:|---:|---:|---|
| conservative | 700 | 32.8052 | 0.9075 | about 0.11-0.15 |
| anime stronger | 700 | 32.1390 | 0.8934 | about 0.29-0.32 |

The conservative variant is better for formal PSNR/SSIM reporting because it keeps BasicVSR++ dominant and uses Real-ESRGAN only cautiously. The anime-stronger variant increases mean alpha and therefore introduces more Real-ESRGAN detail, but this reduces PSNR/SSIM. This result captures the perception-fidelity trade-off: stronger generative enhancement may look sharper, but it is less faithful to the ground truth.

Insert Fig. 5 here: `part3/results/adaptive_hybrid_000_directionc_official/showcase_00000049.png`.

### 4.6 Qualitative Analysis

The diagnostic showcase contains eight panels: BasicVSR++, Real-ESRGAN, Direction C Hybrid, alpha mask, structure protect, uncertain texture, hallucination risk, and flicker risk. The alpha mask is generally dark in reliable structure regions, meaning the final output stays close to BasicVSR++. Real-ESRGAN receives more weight in texture-like regions where perceptual detail may help.

For anime-style videos, pure Real-ESRGAN often produces sharper line art than the conservative hybrid. This is expected because anime frames contain clean edges, flat color regions, and less natural stochastic texture. To study this behavior, we evaluate an anime-stronger variant with higher alpha and weaker structure protection. The result is visually sharper but has lower PSNR/SSIM on validation data. Therefore, our final report treats the conservative setting as the formal Direction C result and the anime-stronger setting as an ablation.

## 5. Conclusion

This project demonstrates a full video super-resolution pipeline from classical baselines to modern VSR and generative enhancement. BasicVSR++ is the most reliable reconstruction model in terms of fidelity and temporal stability. Real-ESRGAN provides sharper perceptual details but can hallucinate textures and introduce temporal inconsistency. The proposed Direction C hybrid pipeline addresses this limitation by estimating a proxy uncertainty/risk map and selectively blending the two branches.

The main limitation is that the uncertainty map is heuristic rather than learned. The temporal metric is also a simplified proxy instead of full optical-flow-warped tLPIPS. Future work could learn the uncertainty estimator from data, use optical flow for stronger temporal evaluation, and automatically choose alpha parameters based on video content.

## Reproduction Summary

The most important commands are collected in:

```text
docs/commands.md
```

The report-ready result paths are collected in:

```text
docs/results_manifest.md
```
