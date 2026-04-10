art 1 Overview

Part 1 focuses on single image super-resolution.
The goal is to train a baseline model that reconstructs a high-resolution (HR) image from a low-resolution (LR) image.

Main characteristics
Paired LR-HR image training
Baseline super-resolution pipeline
PSNR-based evaluation
Used as a reference baseline for comparison with Part 2
Part 2 Overview

Part 2 focuses on video super-resolution (VSR) using a BasicVSR-style framework.

Instead of processing a single image, Part 2 takes a sequence of frames as input and predicts a sequence of high-resolution frames.

Main characteristics
Sequence-based LR-HR training
Temporal modeling across neighboring frames
Bidirectional propagation
Reconstruction of HR video frames
Evaluation with PSNR on frame sequences
Important note

The current implementation in part2 is a trainable BasicVSR-style scaffold / simplified implementation, not a full official reproduction with pretrained weights.

Environment

It is recommended to use Python 3.10 and a virtual environment.

Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
Install dependencies
pip install -r requirements.txt

If requirements.txt is incomplete, install at least:

pip install torch torchvision pyyaml numpy pillow tqdm
Dataset Organization
Part 1

Part 1 uses paired LR-HR images.

Example:

data/
├── train/
│   ├── train_sharp/
│   └── train_sharp_bicubic/
└── val/
    ├── val_sharp/
    └── val_sharp_bicubic/
Part 2

Part 2 expects video frames grouped by sequence/video folder.

Example:

data/
├── train/
│   ├── train_sharp/
│   │   ├── video_001/
│   │   │   ├── 00000000.png
│   │   │   ├── 00000001.png
│   │   │   └── ...
│   │   └── video_002/
│   │
│   └── train_sharp_bicubic/
│       └── X4/
│           ├── video_001/
│           │   ├── 00000000.png
│           │   ├── 00000001.png
│           │   └── ...
│           └── video_002/
│
└── val/
    ├── val_sharp/
    └── val_sharp_bicubic/
        └── X4/
Notes
LR and HR folders must have matching video folder names
Frame names inside each video folder must align one-to-one
For scale: 4, LR data is usually read from the X4/ subfolder
Running Part 1

Move into the part1 directory:

cd part1

Train:

python3 main.py --mode train --cfg config.yaml

Test:

python3 main.py --mode test --cfg config.yaml
Running Part 2

Move into the part2 directory:

cd part2

Train:

python3 main.py --mode train --cfg config.yaml

Test:

python3 main.py --mode test --cfg config.yaml
Part 2 Configuration

Example fields in part2/config.yaml:

seed: 42
device: "cuda"
num_workers: 2

scale: 4
patch_size: 64
seq_len: 3
interval: 1

train:
  batch_size: 2
  epochs: 5
  lr: 0.0001

model:
  mid_channels: 64
  num_blocks: 10

paths:
  train_hr: "/path/to/train_hr"
  train_lr: "/path/to/train_lr"
  val_hr: "/path/to/val_hr"
  val_lr: "/path/to/val_lr"
  test_hr: "/path/to/test_hr"
  test_lr: "/path/to/test_lr"
  ckpt_dir: "checkpoints"
  output_dir: "outputs"
Important note

For Part 2, model parameters should be defined under:

model:
  mid_channels: ...
  num_blocks: ...

This ensures training and testing use the same architecture.

Outputs
Part 1
Trained checkpoints in part1/checkpoints/
Restored SR results in part1/outputs/
Part 2
Trained checkpoints in part2/checkpoints/
Restored SR frame sequences in part2/outputs/
Evaluation

This project mainly uses:

PSNR for reconstruction quality
Sequence-level PSNR for Part 2

For Part 2, qualitative visual inspection is also important because temporal consistency and edge reconstruction matter in video super-resolution.

Current Limitations
Part 1
Baseline-focused
Limited perceptual enhancement
Part 2
Simplified BasicVSR-style implementation
Not a full official BasicVSR / BasicVSR++ reproduction
No full pretrained SpyNet integration yet
Training is memory-intensive on GPUs with limited VRAM
Practical Notes
Video SR is significantly slower than image SR
Small batch size is normal for Part 2 due to GPU memory limits
If CUDA runs out of memory, reduce:
batch_size
seq_len
patch_size
mid_channels
num_blocks

Recommended smaller configuration for limited VRAM:

patch_size: 64
seq_len: 3

train:
  batch_size: 1

model:
  mid_channels: 32
  num_blocks: 5
Future Improvements

Possible future extensions include:

Full optical-flow-based alignment
Stronger BasicVSR reproduction
BasicVSR++ style propagation
Perceptual loss and GAN-based enhancement
Better temporal consistency evaluation
Author

Course project by FC.
