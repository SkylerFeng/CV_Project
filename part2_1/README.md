# Part 2.1: BasicVSR++

Part 2.1 uses BasicVSR++ as the temporally stable video reconstruction branch. It is the main high-fidelity branch in the project and is preferred for reliable structures and real-world videos.

## Files

```text
part2_1/
  checkpoints/
    basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth
    spynet_20210409-c6c1bd09.pth
  mmagic/                                      # local MMagic dependency
  scripts/
    infer_basicvsrpp_video.py                 # folder/video inference
    batch_infer_basicvsrpp_dataset_videos.py  # no-reference dataset videos
    evaluate_basicvsrpp_val.py                # validation metrics
    prepare_meta_info.py
```

## Environment

Use the MMagic environment:

```bash
mmagic_env/bin/python
```

Do not run BasicVSR++ scripts with `.venv/bin/python`, because they require the local MMagic package.

Verify:

```bash
mmagic_env/bin/python -c "import mmagic; print(mmagic.__version__)"
```

## Dataset

Expected validation layout:

```text
data/val/val_sharp_bicubic/X4/<sequence>/*.png
data/val/val_sharp/<sequence>/*.png
```

## Inference on One Validation Sequence

```bash
mmagic_env/bin/python part2_1/scripts/infer_basicvsrpp_video.py \
  --input data/val/val_sharp_bicubic/X4/000 \
  --output part2_1/results/basicvsrpp_val000_x4 \
  --fps 30 \
  --device cuda
```

Typical output:

```text
part2_1/results/basicvsrpp_val000_x4/
  frames/
  videos/
```

## Inference on a Custom Video

```bash
mmagic_env/bin/python part2_1/scripts/infer_basicvsrpp_video.py \
  --input data/custom/3.mp4 \
  --output part2_1/results/custom3_basicvsrpp_x4 \
  --fps 30 \
  --device cuda \
  --half \
  --video-only \
  --chunk-size 2 \
  --chunk-overlap 0
```

Use `--half`, `--video-only`, and small chunk sizes on 8GB GPUs.

## Full Validation Evaluation

```bash
mmagic_env/bin/python part2_1/scripts/evaluate_basicvsrpp_val.py \
  --lr-root data/val/val_sharp_bicubic/X4 \
  --gt-root data/val/val_sharp \
  --output part2_1/results/metrics_val_basicvsrpp \
  --device cuda
```

Output:

```text
part2_1/results/metrics_val_basicvsrpp/
  summary.csv
  per_sequence.csv
  per_frame.csv
```

Final report result:

| Method | Sequences | Frames | PSNR | SSIM |
|---|---:|---:|---:|---:|
| Fine-tuned BasicVSR++ | 30 | 3000 | 31.0021 | 0.8730 |

## Batch No-Reference Video Output

For folders without ground truth, generate videos only:

```bash
mmagic_env/bin/python part2_1/scripts/batch_infer_basicvsrpp_dataset_videos.py \
  --input-root data/REDS-sample \
  --output-root part2_extra_outputs/basicvsrpp_finetuned/REDS-sample \
  --dataset-name REDS-sample \
  --fps 30 \
  --device cuda
```

For a quick test:

```bash
mmagic_env/bin/python part2_1/scripts/batch_infer_basicvsrpp_dataset_videos.py \
  --input-root data/REDS-sample \
  --output-root part2_extra_outputs/basicvsrpp_finetuned_debug/REDS-sample \
  --dataset-name REDS-sample \
  --fps 30 \
  --device cuda \
  --max-seqs 1
```

Repeat for `data/vimeo-RL` by changing the input and output roots.

## Meta Info

If the dataset layout changes, rebuild MMagic meta information:

```bash
mmagic_env/bin/python part2_1/scripts/prepare_meta_info.py \
  --train-root data/train/train_sharp \
  --val-root data/val/val_sharp
```

## Interpretation

BasicVSR++ should be described as:

- temporally stable;
- faithful to ground truth;
- strong on real-world validation videos;
- sometimes visually soft on fine textures or anime-style line art.

