# Report Outline

Suggested title:

```text
Content-Adaptive Hybrid Video Super-Resolution with BasicVSR++ and Real-ESRGAN
```

## Abstract

- State the video super-resolution task.
- Summarize Part 1 baselines, Part 2 BasicVSR++ / Real-ESRGAN, and Part 3 Direction C.
- Main observation: BasicVSR++ is stable but conservative; Real-ESRGAN is sharp but can hallucinate and flicker.
- Main contribution: a proxy uncertainty-aware hybrid pipeline.
- End with the public GitHub repository link.

## 1. Introduction

- Explain spatial detail restoration and temporal consistency.
- Introduce the fidelity-perception trade-off.
- Summarize project contributions:
  - classical and CNN baselines
  - BasicVSR++ reproduction
  
  - Real-ESRGAN reproduction and conservative fine-tuning
  - Direction C hybrid with diagnostic maps

## 2. Related Work

Cover at least the papers required by the guideline:

- Classic / image SR: SRCNN, SRGAN, EDSR.
- Video SR: EDVR, TDAN, BasicVSR, BasicVSR++.
- Generative SR: Real-ESRGAN, SR3, Flow Matching, ControlNet.
- Evaluation: PSNR, SSIM, LPIPS, tLPIPS.

## 3. Method

### 3.1 Part 1 Baselines

- Bicubic and Lanczos interpolation.
- SRCNN.
- Temporal averaging / sharpening baseline.
- Expected limitation: blurry texture and weak temporal reasoning.

### 3.2 BasicVSR++ Branch

- Describe recurrent temporal propagation and alignment.
- Use it as the reliable high-fidelity branch.
- It is preferred for text, faces, edges, and stable structures.

### 3.3 Real-ESRGAN Branch

- Describe it as the perceptual / generative detail branch.
- Discuss original official weights and conservative fine-tuning.
- Limitations: hallucinated textures, color shifts, frame-wise instability.

### 3.4 Direction C Hybrid

Core equation:

```text
I_hybrid = (1 - alpha) * I_BasicVSR++ + alpha * I_RealESRGAN
```

Alpha is a proxy pixel-wise uncertainty/risk map from:

- uncertain texture
- structure protection
- branch disagreement
- hallucination risk
- flicker risk

Important wording:

```text
We estimate a heuristic proxy uncertainty map rather than learning a probabilistic uncertainty model.
```

## 4. Experiments

### 4.1 Datasets

- Wild videos.
- Provided sample clips.
- REDS-style train/val sequences.
- Standard benchmark subset: validation sequences 000-006.

### 4.2 Metrics

- PSNR / SSIM for fidelity.
- LPIPS and tLPIPS proxy for perceptual and temporal quality where available.
- Qualitative visual comparisons.

### 4.3 Part 1 Results

Report Bicubic, Lanczos, SRCNN, and temporal baseline.

### 4.4 Part 2 Results

Compare:

- BasicVSR++
- official Real-ESRGAN
- fine-tuned Real-ESRGAN

Emphasize that GAN-based enhancement can look sharper while scoring lower on PSNR/SSIM.

### 4.5 Part 3 Results

Use the final table:

| Variant | Frames | PSNR | SSIM | Mean alpha |
|---|---:|---:|---:|---|
| conservative | 700 | 32.8052 | 0.9075 | about 0.11-0.15 |
| anime stronger | 700 | 32.1390 | 0.8934 | about 0.29-0.32 |

Interpretation:

- Conservative parameters preserve fidelity and are best for the formal Direction C result.
- Anime-stronger parameters improve subjective sharpness but reduce PSNR/SSIM.

### 4.6 Qualitative Analysis

Include showcase panels:

- BasicVSR++
- Official Real-ESRGAN
- Direction C Hybrid
- Alpha mask
- Structure protect
- Uncertain texture
- Hallucination risk
- Flicker risk

## 5. Conclusion

- Summarize what worked.
- Discuss limitations:
  - uncertainty is heuristic
  - no learned uncertainty estimator
  - tLPIPS is a proxy unless optical-flow warping is added
  - anime and real video prefer different alpha settings
- Future work:
  - learned uncertainty network
  - optical-flow-based temporal metrics
  - content-adaptive parameter selection
  - ControlNet-Tile or Flow Matching integration
