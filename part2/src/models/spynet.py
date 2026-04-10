import torch
import torch.nn as nn


class SpyNetStub(nn.Module):
    """
    Placeholder optical flow module.
    It returns zero flow so the overall training pipeline can run first.
    Replace this with a real SpyNet implementation or pretrained module later.
    """

    def __init__(self):
        super().__init__()

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        b, _, h, w = ref.shape
        return torch.zeros(b, 2, h, w, device=ref.device, dtype=ref.dtype)
