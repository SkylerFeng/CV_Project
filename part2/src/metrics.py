import math
import torch


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    pred = pred.detach().clamp(0, 1)
    target = target.detach().clamp(0, 1)
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def calculate_sequence_psnr(pred_seq: torch.Tensor, target_seq: torch.Tensor) -> float:
    # input: [B, T, C, H, W] or [T, C, H, W]
    if pred_seq.dim() == 4:
        pred_seq = pred_seq.unsqueeze(0)
        target_seq = target_seq.unsqueeze(0)
    scores = []
    for b in range(pred_seq.size(0)):
        for t in range(pred_seq.size(1)):
            scores.append(calculate_psnr(pred_seq[b, t], target_seq[b, t]))
    return sum(scores) / max(len(scores), 1)
