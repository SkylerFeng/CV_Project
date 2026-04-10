import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConvResidualBlocks(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels))
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))
