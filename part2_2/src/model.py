import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block used in ESRGAN / Real-ESRGAN generator.
    Structure:
        x -> conv1
          -> conv2([x, x1])
          -> conv3([x, x1, x2])
          -> conv4([x, x1, x2, x3])
          -> conv5([x, x1, x2, x3, x4])
    Final output uses residual scaling.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + num_grow_ch * 2, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + num_grow_ch * 3, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + num_grow_ch * 4, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block.
    It stacks 3 ResidualDenseBlocks with outer residual scaling.
    """

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out


class RRDBNet(nn.Module):
    """
    RRDBNet generator used by RealESRGAN_x4plus.

    Official RealESRGAN_x4plus generator setting:
        num_in_ch=3
        num_out_ch=3
        num_feat=64
        num_block=23
        num_grow_ch=32
        scale=4
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        if scale not in [1, 2, 4]:
            raise ValueError(f"Unsupported scale: {scale}. Expected 1, 2, or 4.")

        self.scale = scale
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_block = num_block
        self.num_grow_ch = num_grow_ch

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = nn.Sequential(*[
            RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch)
            for _ in range(num_block)
        ])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # upsampling
        if scale == 4:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        elif scale == 2:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = None
        else:
            self.conv_up1 = None
            self.conv_up2 = None

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        if self.scale == 4:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        elif self.scale == 2:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))

        out = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(out)
        return out


def build_realesrgan_x4plus_generator() -> RRDBNet:
    """
    Build generator matching RealESRGAN_x4plus.
    """
    return RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )


def load_generator_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
    strict: bool = True,
):
    """
    Load checkpoint with automatic support for common checkpoint formats:
    - raw state_dict
    - {'params': ...}
    - {'params_ema': ...}

    Returns:
        used_key: one of ['params_ema', 'params', 'raw']
        load_msg: torch load_state_dict return message
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "params_ema" in ckpt:
            state_dict = ckpt["params_ema"]
            used_key = "params_ema"
        elif "params" in ckpt:
            state_dict = ckpt["params"]
            used_key = "params"
        else:
            # maybe raw state_dict but wrapped in dict of tensor params
            state_dict = ckpt
            used_key = "raw"
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    load_msg = model.load_state_dict(state_dict, strict=strict)
    return used_key, load_msg


if __name__ == "__main__":
    model = build_realesrgan_x4plus_generator()
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(model.__class__.__name__)
    print("Input :", x.shape)
    print("Output:", y.shape)