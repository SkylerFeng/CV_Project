import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvResidualBlocks, PixelShufflePack
from .spynet import SpyNetStub


class BasicVSRTiny(nn.Module):
    """
    A simplified BasicVSR-style baseline.

    What it already does:
    - accepts video sequence input [B, T, C, H, W]
    - performs shallow feature extraction per frame
    - performs bidirectional recurrent feature propagation
    - reconstructs HR frame for every timestep

    What it does not yet do:
    - real optical-flow-based feature alignment
    - full BasicVSR paper details

    This is intended as a clean, trainable scaffold for Part 2.
    """

    def __init__(self, mid_channels: int = 64, num_blocks: int = 5, scale: int = 4):
        super().__init__()
        self.mid_channels = mid_channels
        self.scale = scale

        self.spynet = SpyNetStub()
        self.feat_extract = ConvResidualBlocks(3, mid_channels, num_blocks)

        self.backward_trunk = ConvResidualBlocks(mid_channels * 2, mid_channels, num_blocks)
        self.forward_trunk = ConvResidualBlocks(mid_channels * 2, mid_channels, num_blocks)

        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0)

        if scale == 4:
            self.upsample = nn.Sequential(
                PixelShufflePack(mid_channels, mid_channels, 2),
                nn.ReLU(inplace=True),
                PixelShufflePack(mid_channels, mid_channels, 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, 3, 3, 1, 1),
            )
        elif scale == 2:
            self.upsample = nn.Sequential(
                PixelShufflePack(mid_channels, mid_channels, 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, 3, 3, 1, 1),
            )
        else:
            raise ValueError('Only scale 2 or 4 is supported in this scaffold.')

    def _compute_feats(self, x: torch.Tensor):
        b, t, c, h, w = x.shape
        feats = []
        for i in range(t):
            feats.append(self.feat_extract(x[:, i]))
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        feats = self._compute_feats(x)

        backward_feats = [None] * t
        feat_prop = torch.zeros(b, self.mid_channels, h, w, device=x.device, dtype=x.dtype)
        for i in range(t - 1, -1, -1):
            feat_prop = self.backward_trunk(torch.cat([feats[i], feat_prop], dim=1))
            backward_feats[i] = feat_prop

        outputs = []
        feat_prop = torch.zeros(b, self.mid_channels, h, w, device=x.device, dtype=x.dtype)
        for i in range(t):
            feat_prop = self.forward_trunk(torch.cat([feats[i], feat_prop], dim=1))
            fused = self.fusion(torch.cat([feat_prop, backward_feats[i]], dim=1))
            out = self.upsample(fused)
            base = F.interpolate(x[:, i], scale_factor=self.scale, mode='bilinear', align_corners=False)
            outputs.append(out + base)

        return torch.stack(outputs, dim=1)
