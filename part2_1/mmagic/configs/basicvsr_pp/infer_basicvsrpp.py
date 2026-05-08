import os
import torch
from mmagic.models import BasicVSRPlusPlusNet
from PIL import Image
import numpy as np


def load_basicvsrpp_generator_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    if 'params' in state_dict:
        state_dict = state_dict['params']

    state_dict = {
        key.removeprefix('module.'): value
        for key, value in state_dict.items()
    }
    if any(key.startswith('generator.') for key in state_dict):
        state_dict = {
            key.removeprefix('generator.'): value
            for key, value in state_dict.items()
            if key.startswith('generator.')
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            'Checkpoint does not match BasicVSRPlusPlusNet.\n'
            f'Missing keys: {missing[:10]}\n'
            f'Unexpected keys: {unexpected[:10]}')


def load_sequence(folder):
    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    imgs = []
    for f in files:
        img = Image.open(os.path.join(folder, f)).convert('RGB')
        img = np.array(img) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        imgs.append(img)
    return torch.stack(imgs)  # (T, C, H, W)


def save_sequence(tensor, folder):
    os.makedirs(folder, exist_ok=True)
    tensor = tensor.clamp(0, 1)

    for i, img in enumerate(tensor):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(
            os.path.join(folder, f"{i:08d}.png")
        )


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ====== 改这里 ======
    lr_folder = "demo_lr"
    out_folder = "demo_sr"
    ckpt_path = "/home/fc/Coding/CV/part2_1/checkpoints/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"
    # ===================

    # load model
    model = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=7)
    load_basicvsrpp_generator_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()

    # load input
    lr = load_sequence(lr_folder).unsqueeze(0).to(device)  # (1, T, C, H, W)

    # inference
    with torch.no_grad():
        sr = model(lr)

    sr = sr.squeeze(0)  # (T, C, H, W)

    # save
    save_sequence(sr, out_folder)


if __name__ == "__main__":
    main()
