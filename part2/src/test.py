import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import VideoSequenceDataset
from src.metrics import calculate_sequence_psnr
from src.models import BasicVSRTiny
from src.utils import ensure_dir, get_device, load_checkpoint, save_image


def _unwrap_video_name(video_name):
    """把 DataLoader 返回的 video_name 安全还原成字符串。"""
    while isinstance(video_name, (list, tuple)):
        video_name = video_name[0]
    return video_name


def _unwrap_frame_names(frame_names):
    """
    把 DataLoader 返回的 frame_names 还原成:
        ['00000000.png', '00000001.png', ...]
    兼容多种 batch_size=1 下的嵌套形式。
    """
    # 常见情况1: [['00000000.png'], ['00000001.png'], ...]
    if isinstance(frame_names, list) and len(frame_names) > 0:
        result = []
        for item in frame_names:
            while isinstance(item, (list, tuple)):
                item = item[0]
            result.append(item)
        return result

    # 兜底：如果直接是 tuple
    if isinstance(frame_names, tuple):
        result = []
        for item in frame_names:
            while isinstance(item, (list, tuple)):
                item = item[0]
            result.append(item)
        return result

    # 再兜底：单个字符串
    if isinstance(frame_names, str):
        return [frame_names]

    raise TypeError(f"Unsupported frame_names type: {type(frame_names)}")


def run_test(cfg):
    device = get_device(cfg)
    paths = cfg['paths']
    model_cfg = cfg['model']

    test_set = VideoSequenceDataset(
        lr_root=paths['test_lr'],
        hr_root=paths['test_hr'],
        seq_len=cfg['seq_len'],
        crop_size=None,
        scale=cfg['scale'],
        training=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
    )

    model = BasicVSRTiny(
        mid_channels=model_cfg.get('mid_channels', cfg.get('channels', 64)),
        num_blocks=model_cfg.get('num_blocks', cfg.get('num_blocks', 5)),
        scale=cfg['scale'],
    ).to(device)

    checkpoint_path = cfg.get(
        'checkpoint_path',
        cfg.get('test', {}).get('checkpoint', paths.get('checkpoint_path', 'checkpoints/best.pth'))
    )
    load_checkpoint(checkpoint_path, model, optimizer=None, map_location=device)
    model.eval()

    out_root = paths.get('output_dir', paths.get('out_dir', 'outputs'))
    ensure_dir(out_root)

    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)

            pred = model(lr)

            total_psnr += calculate_sequence_psnr(pred, hr)
            count += 1

            video_name = _unwrap_video_name(batch['video_name'])
            frame_names = _unwrap_frame_names(batch['frame_names'])

            video_dir = os.path.join(out_root, video_name)
            ensure_dir(video_dir)

            num_pred_frames = pred.size(1)
            num_name_frames = len(frame_names)
            num_save = min(num_pred_frames, num_name_frames)

            for t in range(num_save):
                frame_name = frame_names[t]
                save_path = os.path.join(video_dir, frame_name)
                save_image(pred[0, t], save_path)

    mean_psnr = total_psnr / max(count, 1)
    print(f'Test PSNR: {mean_psnr:.4f}')