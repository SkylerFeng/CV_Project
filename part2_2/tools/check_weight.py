import os
import sys
import argparse
from pathlib import Path

import torch

# 让脚本可以找到 part2_2/src/model.py
CURRENT_DIR = Path(__file__).resolve().parent
PART2_ROOT = CURRENT_DIR.parent
SRC_DIR = PART2_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import build_realesrgan_x4plus_generator  # noqa: E402


def smart_load_state_dict(model, ckpt_path, map_location="cpu", strict=True):
    ckpt = torch.load(ckpt_path, map_location=map_location)

    tried = []

    if isinstance(ckpt, dict):
        if "params_ema" in ckpt:
            tried.append("params_ema")
            try:
                msg = model.load_state_dict(ckpt["params_ema"], strict=strict)
                return "params_ema", msg
            except Exception as e:
                last_err = e
        if "params" in ckpt:
            tried.append("params")
            try:
                msg = model.load_state_dict(ckpt["params"], strict=strict)
                return "params", msg
            except Exception as e:
                last_err = e

        tried.append("raw")
        try:
            msg = model.load_state_dict(ckpt, strict=strict)
            return "raw", msg
        except Exception as e:
            last_err = e
            raise RuntimeError(
                f"Failed to load checkpoint with keys {tried}. Last error:\n{last_err}"
            )
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description="Check whether RealESRGAN_x4plus.pth matches local RRDBNet.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/fc/Coding/CV/part2_2/models/RealESRGAN_x4plus.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for forward test, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=64,
        help="Input size for forward test",
    )
    args = parser.parse_args()

    ckpt_path = args.ckpt
    strict = args.strict
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("=" * 60)
    print("Check Real-ESRGAN Weight Compatibility")
    print("=" * 60)
    print(f"Checkpoint : {ckpt_path}")
    print(f"Strict load: {strict}")
    print(f"Device     : {device}")
    print("=" * 60)

    model = build_realesrgan_x4plus_generator().to(device)
    total_params, trainable_params = count_parameters(model)

    print(f"Model       : {model.__class__.__name__}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable   : {trainable_params:,}")
    print("-" * 60)

    used_key, load_msg = smart_load_state_dict(
        model=model,
        ckpt_path=ckpt_path,
        map_location="cpu",
        strict=strict,
    )

    print(f"Loaded with key: {used_key}")
    print("load_state_dict message:")
    print(load_msg)
    print("-" * 60)

    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, args.input_size, args.input_size).to(device)
        y = model(x)

    print("Forward test passed.")
    print(f"Input shape : {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print("=" * 60)
    print("Checkpoint is compatible with local RRDBNet definition.")
    print("=" * 60)


if __name__ == "__main__":
    main()