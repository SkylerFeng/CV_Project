# Part 2.2: Real-ESRGAN

Part 2.2 uses Real-ESRGAN as the perceptual enhancement branch. It produces sharper local details than BasicVSR++, especially on anime or line-art content, but it can also hallucinate textures and introduce frame-wise inconsistency on real videos.

## Files

```text
part2_2/
  configs/
    train_conservative.yaml
  models/
    RealESRGAN_x4plus.pth
  src/
    dataset.py
    evaluate.py
    evaluate_realesrgan_val.py
    infer.py
    infer_video.py
    losses.py
    model.py
    tiler.py
    train.py
    video_utils.py
  tools/
    download_official_weights.py
```

## Environment

Use the general environment:

```bash
.venv/bin/python
```

Install dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

If perceptual loss or LPIPS metrics are needed:

```bash
.venv/bin/pip install lpips scipy
```

## Checkpoints

Official checkpoint:

```text
part2_2/models/RealESRGAN_x4plus.pth
```

Fine-tuned checkpoint produced by this project:

```text
part2_2/experiments/realesrgan_x4plus_official_conservative_4epoch/checkpoints/net_g_best.pth
```

If the official checkpoint is missing, download it:

```bash
.venv/bin/python part2_2/tools/download_official_weights.py
```

## Conservative Fine-Tuning

The final training config is:

```text
part2_2/configs/train_conservative.yaml
```

Run training:

```bash
.venv/bin/python part2_2/src/train.py \
  --config part2_2/configs/train_conservative.yaml
```

The loss combines:

```text
Charbonnier reconstruction
+ perceptual loss
+ edge loss
+ Laplacian loss
+ color loss
+ initial-weight regularization
```

Why this loss is used:

- pure reconstruction loss improves PSNR but tends to blur detail;
- strong perceptual or edge losses can over-sharpen and hallucinate textures;
- color loss reduces color shifts;
- initial-weight regularization prevents the model from drifting too far from the official Real-ESRGAN prior.

The fine-tuned model is intended to be a controlled generative branch, not a global replacement for BasicVSR++.

## Image-Sequence Inference

Run on one validation sequence with the fine-tuned checkpoint:

```bash
.venv/bin/python part2_2/src/infer.py \
  --input data/val/val_sharp_bicubic/X4/000 \
  --ckpt part2_2/experiments/realesrgan_x4plus_official_conservative_4epoch/checkpoints/net_g_best.pth \
  --output part2_2/results/val000_realesrgan_finetuned \
  --model-name RealESRGAN_x4plus \
  --device cuda \
  --tile 256 \
  --suffix "" \
  --fps 30
```

Run with the official checkpoint:

```bash
.venv/bin/python part2_2/src/infer.py \
  --input data/val/val_sharp_bicubic/X4/000 \
  --ckpt part2_2/models/RealESRGAN_x4plus.pth \
  --output part2_2/results/val000_realesrgan_official \
  --model-name RealESRGAN_x4plus \
  --device cuda \
  --tile 256 \
  --suffix "" \
  --fps 30
```

## Video Inference

```bash
.venv/bin/python part2_2/src/infer_video.py \
  --input data/custom/3.mp4 \
  --ckpt part2_2/models/RealESRGAN_x4plus.pth \
  --output part2_2/results/custom3_realesrgan_official_full/realesrgan_official_x4.mp4 \
  --model-name RealESRGAN_x4plus \
  --device cuda \
  --tile 256 \
  --tile-pad 10 \
  --crf 18 \
  --preset medium \
  --no-audio
```

Use `--tile 256` or smaller if CUDA memory is limited.

## Full Validation Evaluation

```bash
.venv/bin/python part2_2/src/evaluate_realesrgan_val.py \
  --lr-root data/val/val_sharp_bicubic/X4 \
  --gt-root data/val/val_sharp \
  --ckpt part2_2/experiments/realesrgan_x4plus_official_conservative_4epoch/checkpoints/net_g_best.pth \
  --output part2_2/results/metrics_val_realesrgan_finetuned \
  --device cuda \
  --tile 256
```

Final report result:

| Method | Sequences | Frames | PSNR | SSIM |
|---|---:|---:|---:|---:|
| Fine-tuned Real-ESRGAN | 30 | 3000 | 27.9661 | 0.7854 |

## Batch No-Reference Video Output

For folders without ground truth:

```bash
.venv/bin/python part2_2/src/batch_infer_realesrgan_dataset_videos.py \
  --input-root data/REDS-sample \
  --output-root part2_extra_outputs/realesrgan_finetuned/REDS-sample \
  --dataset-name REDS-sample \
  --ckpt part2_2/experiments/realesrgan_x4plus_official_conservative_4epoch/checkpoints/net_g_best.pth \
  --model-name RealESRGAN_x4plus \
  --fps 30 \
  --tile 256 \
  --device cuda
```

Quick test:

```bash
.venv/bin/python part2_2/src/batch_infer_realesrgan_dataset_videos.py \
  --input-root data/REDS-sample \
  --output-root part2_extra_outputs/realesrgan_finetuned_debug/REDS-sample \
  --dataset-name REDS-sample \
  --ckpt part2_2/models/RealESRGAN_x4plus.pth \
  --model-name RealESRGAN_x4plus \
  --fps 30 \
  --tile 256 \
  --device cuda \
  --max-seqs 1
```

## Interpretation

Real-ESRGAN should be described as:

- sharper and more perceptual than BasicVSR++;
- especially useful for anime and line-art content;
- weaker in PSNR/SSIM than BasicVSR++ on the validation set;
- risky as a frame-wise video method because it can hallucinate details and flicker.

