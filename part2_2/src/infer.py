import os
import sys
import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
import torchvision.transforms.functional as TF


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from model import build_realesrgan_x4plus_generator, load_generator_checkpoint


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def list_images_recursive(root: str) -> List[str]:
    if os.path.isfile(root):
        return [os.path.abspath(root)]

    files = []
    for folder, _, names in os.walk(root):
        for name in names:
            if is_image_file(name):
                files.append(os.path.abspath(os.path.join(folder, name)))
    return sorted(files)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(img).unsqueeze(0)  # [1, 3, H, W]
    return tensor


def save_image(tensor: torch.Tensor, save_path: str):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0)
    img = TF.to_pil_image(tensor)
    ensure_dir(os.path.dirname(save_path))
    img.save(save_path)


def load_model_for_inference(ckpt_path: str, device: torch.device):
    model = build_realesrgan_x4plus_generator().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    used_key = None

    # 1) finetune checkpoints saved by our train.py
    if isinstance(ckpt, dict) and "params" in ckpt:
        try:
            model.load_state_dict(ckpt["params"], strict=True)
            used_key = "params"
            return model.eval(), used_key
        except Exception:
            pass

    # 2) pretrained RealESRGAN_x4plus.pth
    used_key, load_msg = load_generator_checkpoint(
        model=model,
        ckpt_path=ckpt_path,
        map_location="cpu",
        strict=True,
    )
    print(f"[INFO] Loaded checkpoint with key: {used_key}")
    print(load_msg)
    return model.eval(), used_key


@torch.no_grad()
def infer_one(model, img_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    img_tensor = img_tensor.to(device)
    pred = model(img_tensor)
    pred = pred.clamp(0.0, 1.0)
    return pred


def build_output_path(input_path: str, input_root: str, output_root: str, suffix: str) -> str:
    input_path = os.path.abspath(input_path)
    input_root = os.path.abspath(input_root)

    if os.path.isfile(input_root):
        name, ext = os.path.splitext(os.path.basename(input_path))
        return os.path.join(output_root, f"{name}{suffix}{ext}")

    rel_path = os.path.relpath(input_path, input_root)
    rel_dir = os.path.dirname(rel_path)
    name, ext = os.path.splitext(os.path.basename(rel_path))
    return os.path.join(output_root, rel_dir, f"{name}{suffix}{ext}")


def main():
    parser = argparse.ArgumentParser(description="Inference for finetuned Real-ESRGAN x4plus.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image path or folder path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint path, e.g. net_g_best.pth or RealESRGAN_x4plus.pth",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output folder",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_out",
        help="Suffix appended to output filename",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path not found: {args.input}")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ensure_dir(args.output)

    model, used_key = load_model_for_inference(args.ckpt, device)

    image_paths = list_images_recursive(args.input)
    if len(image_paths) == 0:
        raise ValueError(f"No image files found in: {args.input}")

    print("=" * 60)
    print("Real-ESRGAN x4plus Inference")
    print("=" * 60)
    print(f"Input   : {args.input}")
    print(f"Output  : {args.output}")
    print(f"Ckpt    : {args.ckpt}")
    print(f"Device  : {device}")
    print(f"Images  : {len(image_paths)}")
    print(f"Load key: {used_key}")
    print("=" * 60)

    for i, img_path in enumerate(image_paths, 1):
        img_tensor = load_image(img_path)
        pred = infer_one(model, img_tensor, device)

        save_path = build_output_path(
            input_path=img_path,
            input_root=args.input,
            output_root=args.output,
            suffix=args.suffix,
        )
        save_image(pred, save_path)

        print(f"[{i}/{len(image_paths)}] {img_path} -> {save_path}")

    print("=" * 60)
    print("Inference finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()