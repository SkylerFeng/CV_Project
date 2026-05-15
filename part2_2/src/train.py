import os
import sys
import time
import math
import argparse
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from dataset import PairedImageDataset
from model import build_realesrgan_x4plus_generator, load_generator_checkpoint
from losses import build_sr_loss


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tensor_to_image_range(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def calculate_psnr_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = tensor_to_image_range(pred)
    target = tensor_to_image_range(target)
    mse = torch.mean((pred - target) ** 2).item()
    if mse < eps:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def maybe_subset(dataset, max_samples: int, name: str):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    print(f"[INFO] Using {max_samples}/{len(dataset)} samples for {name}.")
    return Subset(dataset, list(range(max_samples)))


def clone_trainable_params(model: torch.nn.Module):
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def initial_weight_regularization(model: torch.nn.Module, initial_params, weight: float) -> torch.Tensor:
    if weight <= 0 or not initial_params:
        return next(model.parameters()).new_tensor(0.0)
    reg = next(model.parameters()).new_tensor(0.0)
    count = 0
    for param, initial in zip((p for p in model.parameters() if p.requires_grad), initial_params):
        reg = reg + torch.mean((param - initial) ** 2)
        count += 1
    return float(weight) * reg / max(count, 1)


def save_checkpoint(save_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, step: int, best_val_loss: float):
    state = {
        "params": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, save_path)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    count = 0

    for batch in val_loader:
        lq = batch["lq"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        pred = model(lq)
        loss = criterion(pred, gt)

        total_loss += loss.item()
        total_psnr += calculate_psnr_torch(pred, gt)
        count += 1

    if count == 0:
        return 0.0, 0.0

    return total_loss / count, total_psnr / count


def main():
    parser = argparse.ArgumentParser(description="Finetune RealESRGAN_x4plus with paired LR/HR data.")
    parser.add_argument(
        "--config",
        type=str,
        default="part2_2/configs/train_conservative.yaml",
        help="Path to config yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    paths_cfg = cfg["paths"]
    train_cfg = cfg["train"]
    resume_cfg = cfg.get("resume", {})

    train_lr = paths_cfg["train_lr"]
    train_hr = paths_cfg["train_hr"]
    val_lr = paths_cfg["val_lr"]
    val_hr = paths_cfg["val_hr"]
    pretrain = paths_cfg["pretrain"]
    save_dir = paths_cfg["save_dir"]

    scale = int(train_cfg.get("scale", 4))
    patch_size = int(train_cfg.get("patch_size", 64))
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 4))

    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    loss_type = str(train_cfg.get("loss_type", "l1"))
    loss_weight = float(train_cfg.get("loss_weight", 1.0))
    perceptual_weight = float(train_cfg.get("perceptual_weight", 0.0))
    edge_weight = float(train_cfg.get("edge_weight", 0.0))
    laplacian_weight = float(train_cfg.get("laplacian_weight", 0.0))
    color_weight = float(train_cfg.get("color_weight", 0.0))
    init_weight_reg = float(train_cfg.get("init_weight_reg", 0.0))
    max_train_samples = int(train_cfg.get("max_train_samples", 0))
    max_val_samples = int(train_cfg.get("max_val_samples", 0))
    amp = bool(train_cfg.get("amp", False))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    log_interval = int(train_cfg.get("log_interval", 10))

    save_freq = int(train_cfg.get("save_freq", 1))
    val_freq = int(train_cfg.get("val_freq", 1))
    device_name = str(train_cfg.get("device", "cuda"))
    seed = int(train_cfg.get("seed", 42))

    resume_path = str(resume_cfg.get("path", "")).strip()
    resume_optimizer = bool(resume_cfg.get("optimizer", True))

    set_seed(seed)

    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    ensure_dir(save_dir)
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    print("=" * 60)
    print("Real-ESRGAN x4plus Finetune")
    print("=" * 60)
    print(f"Config      : {args.config}")
    print(f"Device      : {device}")
    print(f"Train LR    : {train_lr}")
    print(f"Train HR    : {train_hr}")
    print(f"Val LR      : {val_lr}")
    print(f"Val HR      : {val_hr}")
    print(f"Pretrain    : {pretrain}")
    print(f"Save dir    : {save_dir}")
    print(f"Scale       : x{scale}")
    print(f"Patch size  : {patch_size}")
    print(f"Batch size  : {batch_size}")
    print(f"Epochs      : {epochs}")
    print(f"LR          : {lr}")
    print(
        f"Loss        : {loss_type}*{loss_weight} + "
        f"perceptual*{perceptual_weight} + edge*{edge_weight} + "
        f"laplacian*{laplacian_weight} + color*{color_weight} + "
        f"init_reg*{init_weight_reg}"
    )
    print(f"AMP         : {amp}")
    print(f"Grad clip   : {grad_clip}")
    print(f"Log interval: {log_interval}")
    print("=" * 60)

    train_set = PairedImageDataset(
        lr_root=train_lr,
        hr_root=train_hr,
        scale=scale,
        patch_size=patch_size,
        is_train=True,
        use_hflip=True,
        use_rot=True,
    )
    val_set = PairedImageDataset(
        lr_root=val_lr,
        hr_root=val_hr,
        scale=scale,
        patch_size=patch_size,
        is_train=False,
        use_hflip=False,
        use_rot=False,
    )
    train_set = maybe_subset(train_set, max_train_samples, "train")
    val_set = maybe_subset(val_set, max_val_samples, "val")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_realesrgan_x4plus_generator().to(device)

    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"[INFO] Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["params"], strict=True)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    else:
        if not os.path.isfile(pretrain):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrain}")
        used_key, load_msg = load_generator_checkpoint(
            model=model,
            ckpt_path=pretrain,
            map_location="cpu",
            strict=True,
        )
        print(f"[INFO] Loaded pretrained generator from key: {used_key}")
        print(load_msg)
        start_epoch = 1
        global_step = 0
        best_val_loss = float("inf")

    criterion = build_sr_loss(train_cfg).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    initial_params = clone_trainable_params(model) if init_weight_reg > 0 else []
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    if resume_path and resume_optimizer:
        ckpt = torch.load(resume_path, map_location="cpu")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("[INFO] Optimizer state resumed.")
    elif resume_path:
        print("[INFO] Optimizer state is not resumed.")

    log_path = os.path.join(save_dir, "train_log.txt")
    progress_log_path = os.path.join(save_dir, "progress_log.txt")

    def write_progress(message: str):
        print(message, flush=True)
        with open(progress_log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader, 1):
            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                pred = model(lq)
                loss = criterion(pred, gt)
                if init_weight_reg > 0:
                    loss = loss + initial_weight_regularization(model, initial_params, init_weight_reg)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % max(log_interval, 1) == 0 or batch_idx == 1:
                elapsed = time.time() - epoch_start
                batches_done = batch_idx
                batches_left = max(len(train_loader) - batch_idx, 0)
                sec_per_batch = elapsed / max(batches_done, 1)
                eta = batches_left * sec_per_batch
                write_progress(
                    f"[Epoch {epoch}/{epochs}] "
                    f"[Batch {batch_idx}/{len(train_loader)}] "
                    f"step={global_step} "
                    f"loss={loss.item():.6f} "
                    f"elapsed={elapsed / 60:.1f}m "
                    f"eta_epoch={eta / 60:.1f}m"
                )

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        epoch_time = time.time() - epoch_start

        log_msg = (
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"time={epoch_time:.2f}s"
        )

        if epoch % val_freq == 0:
            val_loss, val_psnr = validate(model, val_loader, criterion, device)
            log_msg += f" | val_loss={val_loss:.6f} | val_psnr={val_psnr:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(ckpt_dir, "net_g_best.pth")
                save_checkpoint(best_path, model, optimizer, epoch, global_step, best_val_loss)
                print(f"[INFO] Saved best checkpoint to: {best_path}")

        print(log_msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

        latest_path = os.path.join(ckpt_dir, "net_g_latest.pth")
        save_checkpoint(latest_path, model, optimizer, epoch, global_step, best_val_loss)

        if epoch % save_freq == 0:
            epoch_path = os.path.join(ckpt_dir, f"net_g_epoch_{epoch:03d}.pth")
            save_checkpoint(epoch_path, model, optimizer, epoch, global_step, best_val_loss)
            print(f"[INFO] Saved checkpoint to: {epoch_path}")

    print("=" * 60)
    print("Training finished.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Logs saved to: {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
