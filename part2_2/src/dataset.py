import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def list_image_files_recursive(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            if is_image_file(name):
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, folder)
                files.append(rel_path)
    return sorted(files)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return TF.to_tensor(img)


class PairedImageDataset(Dataset):
    """
    Paired LR-HR dataset for Real-ESRGAN finetuning.

    Supports nested folder structures, e.g.
        HR/000/00000000.png
        LR/000/00000000.png

    Pairing is done by relative path.
    """

    def __init__(
        self,
        lr_root: str,
        hr_root: str,
        scale: int = 4,
        patch_size: int = 64,
        is_train: bool = True,
        use_hflip: bool = True,
        use_rot: bool = True,
    ):
        super().__init__()
        self.lr_root = os.path.abspath(lr_root)
        self.hr_root = os.path.abspath(hr_root)
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train
        self.use_hflip = use_hflip
        self.use_rot = use_rot

        lr_files = list_image_files_recursive(self.lr_root)
        hr_files = list_image_files_recursive(self.hr_root)

        lr_set = set(lr_files)
        hr_set = set(hr_files)

        only_in_lr = sorted(lr_set - hr_set)
        only_in_hr = sorted(hr_set - lr_set)

        if only_in_lr or only_in_hr:
            msg = []
            if only_in_lr:
                msg.append(f"{len(only_in_lr)} files only in LR, e.g. {only_in_lr[:5]}")
            if only_in_hr:
                msg.append(f"{len(only_in_hr)} files only in HR, e.g. {only_in_hr[:5]}")
            raise ValueError("LR/HR filenames do not match. " + " | ".join(msg))

        self.rel_paths = sorted(lr_set & hr_set)

        if len(self.rel_paths) == 0:
            raise ValueError(
                f"No paired image files found.\n"
                f"lr_root={self.lr_root}\n"
                f"hr_root={self.hr_root}"
            )

    def __len__(self) -> int:
        return len(self.rel_paths)

    def _load_pair(self, index: int) -> Tuple[Image.Image, Image.Image, str]:
        rel_path = self.rel_paths[index]
        lr_path = os.path.join(self.lr_root, rel_path)
        hr_path = os.path.join(self.hr_root, rel_path)

        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")
        return lr, hr, rel_path

    def _check_size(self, lr: Image.Image, hr: Image.Image, name: str):
        lr_w, lr_h = lr.size
        hr_w, hr_h = hr.size

        if hr_w != lr_w * self.scale or hr_h != lr_h * self.scale:
            raise ValueError(
                f"Size mismatch for {name}: "
                f"LR=({lr_w},{lr_h}), HR=({hr_w},{hr_h}), "
                f"expected HR=({lr_w * self.scale},{lr_h * self.scale})"
            )

    def _paired_random_crop(
        self,
        lr: Image.Image,
        hr: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        lr_w, lr_h = lr.size
        hr_w, hr_h = hr.size

        lr_patch = self.patch_size
        hr_patch = lr_patch * self.scale

        if lr_w < lr_patch or lr_h < lr_patch:
            raise ValueError(
                f"LR image is smaller than patch_size={lr_patch}, got size=({lr_w},{lr_h})"
            )
        if hr_w < hr_patch or hr_h < hr_patch:
            raise ValueError(
                f"HR image is smaller than expected patch size={hr_patch}, got size=({hr_w},{hr_h})"
            )

        x = random.randint(0, lr_w - lr_patch)
        y = random.randint(0, lr_h - lr_patch)

        lr_crop = lr.crop((x, y, x + lr_patch, y + lr_patch))

        hx = x * self.scale
        hy = y * self.scale
        hr_crop = hr.crop((hx, hy, hx + hr_patch, hy + hr_patch))

        return lr_crop, hr_crop

    def _augment(
        self,
        lr: Image.Image,
        hr: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        if self.use_hflip and random.random() < 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)

        if self.use_rot:
            if random.random() < 0.5:
                lr = TF.vflip(lr)
                hr = TF.vflip(hr)

            k = random.randint(0, 3)
            if k > 0:
                angle = 90 * k
                lr = TF.rotate(lr, angle)
                hr = TF.rotate(hr, angle)

        return lr, hr

    def __getitem__(self, index: int):
        lr, hr, rel_path = self._load_pair(index)
        self._check_size(lr, hr, rel_path)

        if self.is_train:
            lr, hr = self._paired_random_crop(lr, hr)
            lr, hr = self._augment(lr, hr)

        lr_tensor = pil_to_tensor(lr)
        hr_tensor = pil_to_tensor(hr)

        return {
            "lq": lr_tensor,
            "gt": hr_tensor,
            "name": rel_path,
        }


if __name__ == "__main__":
    dataset = PairedImageDataset(
        lr_root="/home/fc/Coding/CV/data/train/train_sharp_bicubic/X4",
        hr_root="/home/fc/Coding/CV/data/train/train_sharp",
        scale=4,
        patch_size=64,
        is_train=True,
    )
    sample = dataset[0]
    print("Dataset size:", len(dataset))
    print("Name:", sample["name"])
    print("LQ shape:", sample["lq"].shape)
    print("GT shape:", sample["gt"].shape)