import os
import random
from PIL import Image
from torch.utils.data import Dataset
from .utils import pil_to_tensor


class PairSRDataset(Dataset):
    """
    REDS-style paired dataset:
      HR: data/train/train_sharp/000/00000000.png
      LR: data/train/train_sharp_bicubic/X4/000/00000000.png

    For SRCNN:
      input  = bicubic-upsampled LR
      target = HR
    """
    def __init__(self, lr_root: str, hr_root: str, patch_size: int = 96, scale: int = 4, is_train: bool = True):
        self.lr_root = os.path.join(lr_root, f"X{scale}")
        self.hr_root = hr_root
        self.patch_size = int(patch_size)
        self.scale = int(scale)
        self.is_train = bool(is_train)

        self.pairs = []
        video_folders = sorted(os.listdir(self.hr_root))

        for video in video_folders:
            hr_video_dir = os.path.join(self.hr_root, video)
            lr_video_dir = os.path.join(self.lr_root, video)

            if not os.path.isdir(hr_video_dir):
                continue
            if not os.path.isdir(lr_video_dir):
                continue

            hr_frames = sorted([
                f for f in os.listdir(hr_video_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])

            for frame in hr_frames:
                hr_path = os.path.join(hr_video_dir, frame)
                lr_path = os.path.join(lr_video_dir, frame)
                if os.path.exists(lr_path):
                    self.pairs.append((lr_path, hr_path, os.path.join(video, frame)))

        if len(self.pairs) == 0:
            raise ValueError(
                f"No LR/HR pairs found.\n"
                f"hr_root={self.hr_root}\n"
                f"lr_root={self.lr_root}"
            )

        print(f"[PairSRDataset] Loaded {len(self.pairs)} pairs from {self.hr_root}")

    def __len__(self):
        return len(self.pairs)

    def _paired_random_crop(self, lr: Image.Image, hr: Image.Image):
        """
        Crop HR patch of size patch_size x patch_size
        and corresponding LR patch of size (patch_size/scale).
        """
        hr_w, hr_h = hr.size
        lr_w, lr_h = lr.size

        ps_hr = self.patch_size
        ps_lr = ps_hr // self.scale

        # 保证尺寸匹配
        hr_w = min(hr_w, lr_w * self.scale)
        hr_h = min(hr_h, lr_h * self.scale)
        hr = hr.crop((0, 0, hr_w, hr_h))
        lr = lr.crop((0, 0, hr_w // self.scale, hr_h // self.scale))

        hr_w, hr_h = hr.size
        lr_w, lr_h = lr.size

        if hr_w < ps_hr or hr_h < ps_hr:
            # 太小就退化成最小可裁尺寸
            ps_hr = min(hr_w, hr_h)
            ps_hr = (ps_hr // self.scale) * self.scale
            ps_lr = max(1, ps_hr // self.scale)

        x_lr = random.randint(0, lr_w - ps_lr) if lr_w > ps_lr else 0
        y_lr = random.randint(0, lr_h - ps_lr) if lr_h > ps_lr else 0

        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale

        lr_patch = lr.crop((x_lr, y_lr, x_lr + ps_lr, y_lr + ps_lr))
        hr_patch = hr.crop((x_hr, y_hr, x_hr + ps_hr, y_hr + ps_hr))

        return lr_patch, hr_patch

    def __getitem__(self, idx):
        lr_path, hr_path, rel_path = self.pairs[idx]

        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")

        # 保证 HR 与 LR*scale 对齐
        hr_w, hr_h = hr.size
        lr_w, lr_h = lr.size

        hr_w = min(hr_w, lr_w * self.scale)
        hr_h = min(hr_h, lr_h * self.scale)

        hr = hr.crop((0, 0, hr_w, hr_h))
        lr = lr.crop((0, 0, hr_w // self.scale, hr_h // self.scale))

        if self.is_train:
            lr, hr = self._paired_random_crop(lr, hr)

        # 关键：把 LR 上采样到 HR 大小，供 SRCNN 输入
        lr_up = lr.resize(hr.size, resample=Image.BICUBIC)

        return {
            "lr_up": pil_to_tensor(lr_up),   # 输入模型
            "hr": pil_to_tensor(hr),         # 监督目标
            "path": rel_path,
        }