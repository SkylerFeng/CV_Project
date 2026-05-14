import os
import subprocess
from typing import Iterable, List

import numpy as np
from PIL import Image


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_image_files(folder: str) -> List[str]:
    return sorted([
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(IMG_EXTS)
    ])


class FFmpegVideoWriter:
    def __init__(self, out_path: str, frame_size, fps: int = 30):
        width, height = frame_size
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            out_path,
        ]
        self.out_path = out_path
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.closed = False

    def append_pil(self, image: Image.Image):
        frame = np.asarray(image.convert("RGB"))
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        self.process.wait()
        if self.process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed while writing {self.out_path}")


def images_to_video(image_paths: Iterable[str], out_path: str, fps: int = 30):
    image_paths = list(image_paths)
    if not image_paths:
        return

    first = Image.open(image_paths[0]).convert("RGB")
    writer = FFmpegVideoWriter(out_path, first.size, fps=fps)
    try:
        writer.append_pil(first)
        for path in image_paths[1:]:
            writer.append_pil(Image.open(path).convert("RGB"))
    finally:
        writer.close()
