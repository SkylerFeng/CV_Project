import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19


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


class CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * torch.sqrt((pred - target) ** 2 + self.eps).mean()


def build_pixel_loss(loss_type: str = "l1", loss_weight: float = 1.0) -> nn.Module:
    loss_type = loss_type.lower()

    if loss_type == "l1":
        return PixelL1Loss(loss_weight=loss_weight)
    elif loss_type in ["mse", "l2"]:
        return PixelMSELoss(loss_weight=loss_weight)
    elif loss_type in ["charbonnier", "cb"]:
        return CharbonnierLoss(loss_weight=loss_weight)
    else:
        raise ValueError(
            f"Unsupported loss_type: {loss_type}. "
            "Expected one of ['l1', 'mse', 'l2', 'charbonnier']."
        )


class VGGPerceptualLoss(nn.Module):
    def __init__(
        self,
        layer_ids=(3, 8, 17, 26),
        layer_weights=(1.0, 1.0, 1.0, 1.0),
        loss_weight: float = 0.01,
    ):
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1
        features = vgg19(weights=weights).features.eval()
        for param in features.parameters():
            param.requires_grad = False
        self.features = features
        self.layer_ids = set(int(idx) for idx in layer_ids)
        self.layer_weights = {
            int(idx): float(weight)
            for idx, weight in zip(layer_ids, layer_weights)
        }
        self.loss_weight = float(loss_weight)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(0, 1) - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._normalize(pred)
        target = self._normalize(target)
        loss = pred.new_tensor(0.0)
        for idx, layer in enumerate(self.features):
            pred = layer(pred)
            with torch.no_grad():
                target = layer(target)
            if idx in self.layer_ids:
                loss = loss + self.layer_weights[idx] * F.l1_loss(pred, target)
        return self.loss_weight * loss


class EdgeLoss(nn.Module):
    def __init__(self, loss_weight: float = 0.05):
        super().__init__()
        self.loss_weight = float(loss_weight)
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x, persistent=False)
        self.register_buffer("kernel_y", kernel_y, persistent=False)

    def _gray(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        gray = self._gray(x.clamp(0, 1))
        gx = F.conv2d(gray, self.kernel_x, padding=1)
        gy = F.conv2d(gray, self.kernel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * F.l1_loss(self._edges(pred), self._edges(target))


class LaplacianLoss(nn.Module):
    def __init__(self, loss_weight: float = 0.02):
        super().__init__()
        self.loss_weight = float(loss_weight)
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel, persistent=False)

    def _gray(self, x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(self._gray(x.clamp(0, 1)), self.kernel, padding=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * F.l1_loss(self._laplacian(pred), self._laplacian(target))


class ColorLoss(nn.Module):
    def __init__(self, loss_weight: float = 0.03):
        super().__init__()
        self.loss_weight = float(loss_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.clamp(0, 1).mean(dim=(2, 3))
        target_mean = target.clamp(0, 1).mean(dim=(2, 3))
        return self.loss_weight * F.l1_loss(pred_mean, target_mean)


class CombinedSRLoss(nn.Module):
    def __init__(
        self,
        pixel_type: str = "l1",
        pixel_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        edge_weight: float = 0.0,
        laplacian_weight: float = 0.0,
        color_weight: float = 0.0,
    ):
        super().__init__()
        self.pixel_loss = build_pixel_loss(pixel_type, pixel_weight)
        self.perceptual_loss = (
            VGGPerceptualLoss(loss_weight=perceptual_weight)
            if perceptual_weight > 0
            else None
        )
        self.edge_loss = EdgeLoss(loss_weight=edge_weight) if edge_weight > 0 else None
        self.laplacian_loss = (
            LaplacianLoss(loss_weight=laplacian_weight)
            if laplacian_weight > 0
            else None
        )
        self.color_loss = ColorLoss(loss_weight=color_weight) if color_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.pixel_loss(pred, target)
        if self.perceptual_loss is not None:
            loss = loss + self.perceptual_loss(pred, target)
        if self.edge_loss is not None:
            loss = loss + self.edge_loss(pred, target)
        if self.laplacian_loss is not None:
            loss = loss + self.laplacian_loss(pred, target)
        if self.color_loss is not None:
            loss = loss + self.color_loss(pred, target)
        return loss


def build_sr_loss(train_cfg: dict) -> nn.Module:
    return CombinedSRLoss(
        pixel_type=str(train_cfg.get("loss_type", "l1")),
        pixel_weight=float(train_cfg.get("loss_weight", 1.0)),
        perceptual_weight=float(train_cfg.get("perceptual_weight", 0.0)),
        edge_weight=float(train_cfg.get("edge_weight", 0.0)),
        laplacian_weight=float(train_cfg.get("laplacian_weight", 0.0)),
        color_weight=float(train_cfg.get("color_weight", 0.0)),
    )


if __name__ == "__main__":
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    l1_loss = build_pixel_loss("l1")
    mse_loss = build_pixel_loss("mse")

    print("L1 :", l1_loss(pred, target).item())
    print("MSE:", mse_loss(pred, target).item())
