import os
import random
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def _list_frames(video_dir: str) -> List[str]:
    names = []
    for name in os.listdir(video_dir):
        if os.path.splitext(name.lower())[1] in IMG_EXTS:
            names.append(name)
    return sorted(names)


class VideoSequenceDataset(Dataset):
    """
    Expected directory layout:
    root/
      video_001/
        00000000.png
        00000001.png
      video_002/
        ...

    lr_root and hr_root must share the same video folder names and frame names.
    """

    def __init__(
        self,
        lr_root: str,
        hr_root: str,
        seq_len: int = 5,
        crop_size: int = None,
        scale: int = 4,
        training: bool = True,
    ):
        self.lr_root = lr_root
        self.hr_root = hr_root
        self.seq_len = seq_len
        self.crop_size = crop_size
        self.scale = scale
        self.training = training
        self.samples: List[Tuple[str, List[str]]] = []
        self._build_index()

    def _build_index(self) -> None:
        if not os.path.isdir(self.lr_root):
            raise ValueError(f'LR root does not exist: {self.lr_root}')
        if not os.path.isdir(self.hr_root):
            raise ValueError(f'HR root does not exist: {self.hr_root}')

        video_names = sorted([d for d in os.listdir(self.lr_root) if os.path.isdir(os.path.join(self.lr_root, d))])
        for video_name in video_names:
            lr_video_dir = os.path.join(self.lr_root, video_name)
            hr_video_dir = os.path.join(self.hr_root, video_name)
            if not os.path.isdir(hr_video_dir):
                continue

            lr_frames = _list_frames(lr_video_dir)
            hr_frames = _list_frames(hr_video_dir)
            if lr_frames != hr_frames:
                raise ValueError(f'Frame mismatch in video {video_name}')
            if len(lr_frames) < self.seq_len:
                continue

            for start in range(0, len(lr_frames) - self.seq_len + 1):
                seq_names = lr_frames[start:start + self.seq_len]
                self.samples.append((video_name, seq_names))

        if not self.samples:
            raise ValueError('No valid video sequences found.')

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sequence(self, root: str, video_name: str, frame_names: List[str]) -> List[torch.Tensor]:
        frames = []
        for frame_name in frame_names:
            path = os.path.join(root, video_name, frame_name)
            image = Image.open(path).convert('RGB')
            frames.append(to_tensor(image))
        return frames

    def _paired_random_crop(self, lr_seq: List[torch.Tensor], hr_seq: List[torch.Tensor]):
        if self.crop_size is None:
            return lr_seq, hr_seq

        _, h, w = lr_seq[0].shape
        if h < self.crop_size or w < self.crop_size:
            return lr_seq, hr_seq

        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        hr_top = top * self.scale
        hr_left = left * self.scale
        hr_crop = self.crop_size * self.scale

        lr_seq = [x[:, top:top + self.crop_size, left:left + self.crop_size] for x in lr_seq]
        hr_seq = [x[:, hr_top:hr_top + hr_crop, hr_left:hr_left + hr_crop] for x in hr_seq]
        return lr_seq, hr_seq

    def __getitem__(self, idx: int):
        video_name, frame_names = self.samples[idx]
        lr_seq = self._load_sequence(self.lr_root, video_name, frame_names)
        hr_seq = self._load_sequence(self.hr_root, video_name, frame_names)

        if self.training:
            lr_seq, hr_seq = self._paired_random_crop(lr_seq, hr_seq)

        lr_seq = torch.stack(lr_seq, dim=0)   # [T, C, H, W]
        hr_seq = torch.stack(hr_seq, dim=0)   # [T, C, Hs, Ws]
        return {
            'lr': lr_seq,
            'hr': hr_seq,
            'video_name': video_name,
            'frame_names': frame_names,
        }
