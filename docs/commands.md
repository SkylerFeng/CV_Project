# Reproduction Commands

Run commands from the repository root:

```bash
cd /home/fc/Coding/CV
```

## Part 1 Evaluation

```bash
python part1/scripts/evaluate_part1_val.py
```

Output:

```text
part1/outputs/metrics_val_part1/summary.csv
```

## Part 2.1 BasicVSR++

Use the MMagic environment:

```bash
mmagic_env/bin/python part2_1/scripts/evaluate_basicvsrpp_val.py \
  --lr-root data/val/val_sharp_bicubic/X4 \
  --gt-root data/val/val_sharp \
  --output part2_1/results/metrics_val_basicvsrpp_final \
  --device cuda
```

Video/folder inference:

```bash
mmagic_env/bin/python part2_1/scripts/infer_basicvsrpp_video.py \
  --input data/val/val_sharp_bicubic/X4/000 \
  --output part2_1/results/basicvsrpp_val000_x4 \
  --fps 30 \
  --device cuda
```

For large videos, use FP16 and direct video writing:

```bash
mmagic_env/bin/python part2_1/scripts/infer_basicvsrpp_video.py \
  --input data/custom/wild_01.mp4 \
  --output part2_1/results/wild_01_basicvsrpp_4k \
  --input-max-side 960 \
  --chunk-size 2 \
  --chunk-overlap 0 \
  --half \
  --video-only \
  --device cuda
```

## Part 2.2 Real-ESRGAN

Final 4-epoch conservative fine-tuning:

```bash
.venv/bin/python part2_2/src/train.py \
  --config part2_2/configs/train_conservative.yaml
```

Official image-sequence inference:

```bash
.venv/bin/python part2_2/src/infer.py \
  --input data/val/val_sharp_bicubic/X4/000 \
  --ckpt part2_2/models/RealESRGAN_x4plus.pth \
  --output part2_2/results/val000_official_x4plus_original \
  --model-name RealESRGAN_x4plus \
  --device cuda \
  --tile 256 \
  --suffix "" \
  --fps 30
```

Fine-tuned image-sequence inference:

```bash
.venv/bin/python part2_2/src/infer.py \
  --input data/val/val_sharp_bicubic/X4/000 \
  --ckpt part2_2/experiments/realesrgan_x4plus_official_conservative_4epoch/checkpoints/net_g_best.pth \
  --output part2_2/results/val000_conservative_4epoch_best \
  --model-name RealESRGAN_x4plus \
  --device cuda \
  --tile 256 \
  --suffix "" \
  --fps 30
```

Video inference:

```bash
.venv/bin/python part2_2/src/infer_video.py \
  --input data/custom/3.mp4 \
  --ckpt part2_2/models/RealESRGAN_x4plus.pth \
  --output part2_2/results/video3_official_x4plus_original/sr.mp4 \
  --model-name RealESRGAN_x4plus \
  --device cuda \
  --tile 256 \
  --tile-pad 10 \
  --crf 18 \
  --preset medium \
  --no-audio
```

Evaluation:

```bash
.venv/bin/python part2_2/src/evaluate.py \
  --pred part2_2/results/val000_official_x4plus_original \
  --gt data/val/val_sharp/000 \
  --output part2_2/results/metrics_val000_official_x4plus_original_eval \
  --pred-suffix ""
```

## Part 3 Direction C

Single config:

```bash
.venv/bin/python part3/scripts/run_adaptive_hybrid.py \
  --config part3/configs/adaptive_hybrid_000_directionc_official.yaml
```

Batch wrapper for final val000-006 variants:

```bash
.venv/bin/python part3/scripts/run_directionc_batch.py \
  --variant conservative \
  --run-hybrid \
  --evaluate \
  --showcase

.venv/bin/python part3/scripts/run_directionc_batch.py \
  --variant anime_stronger \
  --run-hybrid \
  --evaluate \
  --showcase
```

Generate one showcase image:

```bash
.venv/bin/python part3/scripts/make_directionc_showcase.py \
  --result-dir part3/results/adaptive_hybrid_val000_006_official_conservative_backup/000 \
  --frame 00000049
```

## Report Figures

```bash
python scripts/make_report_figures.py
```
