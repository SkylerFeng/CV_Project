import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from .dataset import PairSRDataset
from .model import SRCNN
from .utils import ensure_dir, psnr_torch


def run_train(cfg: dict):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    scale = int(cfg["scale"])
    patch = int(cfg["patch_size"])
    bs = int(cfg["batch_size"])
    nw = int(cfg["num_workers"])

    train_set = PairSRDataset(
    cfg["paths"]["train_lr"],
    cfg["paths"]["train_hr"],
    patch_size=patch,
    scale=scale,
    is_train=True
    )

    val_set = PairSRDataset(
        cfg["paths"]["val_lr"],
        cfg["paths"]["val_hr"],
        patch_size=patch,
        scale=scale,
        is_train=False
    )

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=nw, pin_memory=True)

    model = SRCNN().to(device)

    loss_name = cfg["train"].get("loss", "l1").lower()
    criterion = nn.L1Loss() if loss_name == "l1" else nn.MSELoss()
    optim = Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    ckpt_dir = cfg["paths"]["ckpt_dir"]
    out_dir = cfg["paths"]["out_dir"]
    ensure_dir(ckpt_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))

    epochs = int(cfg["train"]["epochs"])
    save_every = int(cfg["train"].get("save_every", 1))

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Train epoch {ep}/{epochs}")
        for batch in pbar:
            lr_up = batch["lr_up"].to(device)
            hr = batch["hr"].to(device)

            sr = model(lr_up)
            loss = criterion(sr, hr)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = running / max(1, len(train_loader))

        # quick val psnr
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for vb in val_loader:
                lr_up = vb["lr_up"].to(device)
                hr = vb["hr"].to(device)
                sr = model(lr_up).clamp(0, 1)
                val_psnr += psnr_torch(sr, hr)
        val_psnr /= max(1, len(val_loader))

        print(f"[Epoch {ep}] avg_loss={avg_loss:.6f} val_psnr={val_psnr:.2f} dB")

        if ep % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"srcnn_x{scale}_epoch{ep}.pth")
            torch.save(
                {"epoch": ep, "model": model.state_dict(), "scale": scale},
                ckpt_path
            )
            print(f"Saved: {ckpt_path}")