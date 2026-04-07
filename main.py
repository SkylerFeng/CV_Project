import argparse
from src.utils import load_config, set_seed
from src.train import run_train
from src.test import run_test
from src.infer import run_infer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["train", "test", "infer"])
    p.add_argument("--cfg", type=str, default="config.yaml")
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint path for test/infer")
    p.add_argument("--input", type=str, default=None, help="input image path for infer")
    p.add_argument("--output", type=str, default=None, help="output image path for infer")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    set_seed(cfg.get("seed", 42))

    if args.mode == "train":
        run_train(cfg)

    elif args.mode == "test":
        run_test(cfg, ckpt_path=args.ckpt)

    elif args.mode == "infer":
        if args.input is None:
            raise ValueError("--input is required for infer mode")
        run_infer(cfg, ckpt_path=args.ckpt, input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()