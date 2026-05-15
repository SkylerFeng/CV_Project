# Part 1: Baselines

Part 1 implements the required baseline methods for video super-resolution:

- Bicubic interpolation
- Lanczos interpolation
- SRCNN
- Simple temporal averaging without motion alignment

These methods establish lower-bound performance before moving to BasicVSR++ and Real-ESRGAN.

## Files

```text
part1/
  config.yaml
  main.py
  checkpoints/
    srcnn_x4_epoch20.pth
  src/
    dataset.py
    infer.py
    model.py
    temporal.py
    test.py
    train.py
    utils.py
    video_utils.py
  scripts/
    evaluate_part1_val.py
    make_part1_report_figure.py
```

## Environment

Run Part 1 with the general environment:

```bash
.venv/bin/python
```

Install dependencies from the repository root:

```bash
.venv/bin/pip install -r requirements.txt
```

## Dataset

Expected validation layout:

```text
data/val/val_sharp_bicubic/X4/<sequence>/*.png
data/val/val_sharp/<sequence>/*.png
```

The low-resolution frames are restored to the high-resolution size and compared with the paired ground truth.

## Evaluate All Part 1 Baselines

From the repository root:

```bash
.venv/bin/python part1/scripts/evaluate_part1_val.py \
  --lr-root data/val/val_sharp_bicubic/X4 \
  --gt-root data/val/val_sharp \
  --ckpt part1/checkpoints/srcnn_x4_epoch20.pth \
  --output part1/outputs/metrics_val_part1 \
  --device cuda
```

For CPU only:

```bash
.venv/bin/python part1/scripts/evaluate_part1_val.py \
  --lr-root data/val/val_sharp_bicubic/X4 \
  --gt-root data/val/val_sharp \
  --ckpt part1/checkpoints/srcnn_x4_epoch20.pth \
  --output part1/outputs/metrics_val_part1 \
  --device cpu
```

For a quick smoke test on a few sequences:

```bash
.venv/bin/python part1/scripts/evaluate_part1_val.py \
  --lr-root data/val/val_sharp_bicubic/X4 \
  --gt-root data/val/val_sharp \
  --ckpt part1/checkpoints/srcnn_x4_epoch20.pth \
  --output part1/outputs/metrics_val_part1_debug \
  --max-seqs 2 \
  --device cuda
```

## Outputs

```text
part1/outputs/metrics_val_part1/
  summary.csv
  summary.md
  per_sequence.csv
  per_frame.csv
```

Final full-validation results:

| Method | Frames | PSNR | SSIM |
|---|---:|---:|---:|
| Bicubic | 3000 | 26.2949 | 0.7233 |
| Lanczos | 3000 | 26.5041 | 0.7302 |
| SRCNN | 3000 | 27.3264 | 0.7655 |
| Temporal Avg. w/o Align. | 3000 | 23.1046 | 0.6403 |

## Generate Report Figure

```bash
.venv/bin/python part1/scripts/make_part1_report_figure.py
```

Report-ready output:

```text
figures/part1_baseline_comparison.png
```

## Single-Image SRCNN Inference

```bash
.venv/bin/python part1/main.py \
  --mode infer \
  --ckpt part1/checkpoints/srcnn_x4_epoch20.pth \
  --input path/to/input.png \
  --output path/to/output.png
```

## Interpretation

Use Part 1 to show:

- fixed interpolation is fast but smooth;
- SRCNN improves structure over bicubic/Lanczos;
- naive temporal averaging can hurt because it does not align motion.

