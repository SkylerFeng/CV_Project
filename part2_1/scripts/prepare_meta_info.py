import argparse
from pathlib import Path


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXTS


def build_meta_info(video_root: str, out_file: str):
    video_root = Path(video_root)
    if not video_root.is_dir():
        raise FileNotFoundError(f'Video root not found: {video_root}')

    lines = []
    video_dirs = sorted([p for p in video_root.iterdir() if p.is_dir()])

    for video_dir in video_dirs:
        frames = sorted([p.name for p in video_dir.iterdir() if p.is_file() and is_image_file(p.name)])
        if len(frames) == 0:
            continue
        lines.append(f'{video_dir.name} {len(frames)}')

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

    print(f'Saved meta info to: {out_path}')
    print(f'Found {len(lines)} video sequences.')


def main():
    parser = argparse.ArgumentParser(description='Build BasicVSR++ meta-info files from frame folders.')
    parser.add_argument('--train-root', default='data/train/train_sharp', help='Training HR sequence root.')
    parser.add_argument('--val-root', default='data/val/val_sharp', help='Validation HR sequence root.')
    parser.add_argument('--train-out', default='part2_1/data/meta_info_train.txt', help='Output train meta-info file.')
    parser.add_argument('--val-out', default='part2_1/data/meta_info_val.txt', help='Output validation meta-info file.')
    args = parser.parse_args()

    build_meta_info(args.train_root, args.train_out)
    build_meta_info(args.val_root, args.val_out)


if __name__ == '__main__':
    main()
