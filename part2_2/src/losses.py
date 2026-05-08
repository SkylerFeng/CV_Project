import torch
import torch.nn as nn


class PixelL1Loss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * self.criterion(pred, target)


class PixelMSELoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * self.criterion(pred, target)


def build_pixel_loss(loss_type: str = "l1", loss_weight: float = 1.0) -> nn.Module:
    loss_type = loss_type.lower()

    if loss_type == "l1":
        return PixelL1Loss(loss_weight=loss_weight)
    elif loss_type in ["mse", "l2"]:
        return PixelMSELoss(loss_weight=loss_weight)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Expected one of ['l1', 'mse', 'l2'].")


if __name__ == "__main__":
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    l1_loss = build_pixel_loss("l1")
    mse_loss = build_pixel_loss("mse")

    print("L1 :", l1_loss(pred, target).item())
    print("MSE:", mse_loss(pred, target).item())