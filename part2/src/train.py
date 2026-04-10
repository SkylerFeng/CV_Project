import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import VideoSequenceDataset
from src.metrics import calculate_sequence_psnr
from src.models import BasicVSRTiny
from src.utils import ensure_dir, get_device, save_checkpoint, load_checkpoint


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    count = 0
    for batch in loader:
        lr = batch['lr'].to(device)
        hr = batch['hr'].to(device)
        pred = model(lr)
        total_psnr += calculate_sequence_psnr(pred, hr)
        count += 1
    return total_psnr / max(count, 1)


def run_train(cfg):
    device = get_device(cfg)
    paths = cfg['paths']
    train_cfg = cfg['train']
    model_cfg = cfg['model']

    train_set = VideoSequenceDataset(
        lr_root=paths['train_lr'],
        hr_root=paths['train_hr'],
        seq_len=cfg['seq_len'],
        crop_size=cfg.get('crop_size'),
        scale=cfg['scale'],
        training=True,
    )
    val_set = VideoSequenceDataset(
        lr_root=paths['val_lr'],
        hr_root=paths['val_hr'],
        seq_len=cfg['seq_len'],
        crop_size=None,
        scale=cfg['scale'],
        training=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
    )

    model = BasicVSRTiny(
        mid_channels=model_cfg.get('mid_channels', 64),
        num_blocks=model_cfg.get('num_blocks', 5),
        scale=cfg['scale'],
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = float(train_cfg['lr']),
        weight_decay=train_cfg.get('weight_decay', 0.0),
    )

    start_epoch = 0
    best_psnr = -1.0
    resume_path = cfg.get('resume')
    if resume_path:
        ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, map_location=device)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('best_psnr', -1.0)

    save_dir = paths['ckpt_dir']
    ensure_dir(save_dir)

    epochs = train_cfg['epochs']
    log_interval = train_cfg.get('log_interval', 10)
    val_interval = train_cfg.get('val_interval', 1)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train {epoch + 1}/{epochs}')
        for step, batch in pbar:
            lr = batch['lr'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(lr)
            loss = criterion(pred, hr)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % log_interval == 0 or (step + 1) == len(train_loader):
                pbar.set_postfix(loss=f'{running_loss / (step + 1):.4f}')

        if (epoch + 1) % val_interval == 0:
            val_psnr = validate(model, val_loader, device)
            print(f'Epoch {epoch + 1}: val_psnr={val_psnr:.4f}')

            latest_path = os.path.join(save_dir, 'latest.pth')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'config': cfg,
            }
            save_checkpoint(state, latest_path)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                state['best_psnr'] = best_psnr
                best_path = os.path.join(save_dir, 'best.pth')
                save_checkpoint(state, best_path)
                print(f'Saved best checkpoint to {best_path}')
