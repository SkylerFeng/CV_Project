import torch
import torch.nn.functional as F


def flow_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    x: [B, C, H, W]
    flow: [B, 2, H, W], dx/dy in pixel units
    This is a reusable utility. In the current tiny model it is not yet used.
    """
    b, c, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=x.device),
        torch.arange(0, w, device=x.device),
        indexing='ij',
    )
    base_grid = torch.stack((grid_x, grid_y), dim=0).float()  # [2, H, W]
    base_grid = base_grid.unsqueeze(0).repeat(b, 1, 1, 1)
    vgrid = base_grid + flow

    vgrid_x = 2.0 * vgrid[:, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1] / max(h - 1, 1) - 1.0
    vgrid = torch.stack((vgrid_x, vgrid_y), dim=-1)
    return F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border', align_corners=True)
