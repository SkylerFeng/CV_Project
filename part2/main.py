import argparse
from src.utils import load_config, set_seed
from src.train import run_train
from src.test import run_test


def parse_args():
    parser = argparse.ArgumentParser(description='Part2 Video SR entry point')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
    parser.add_argument('--cfg', type=str, default='config.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args.cfg)
    set_seed(cfg.get('seed', 42))

    if args.mode == 'train':
        run_train(cfg)
    else:
        run_test(cfg)
