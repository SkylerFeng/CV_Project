import math

import torch
import torch.nn.functional as F


class RealESRGANTiler:
    """
    Lightweight tiling wrapper adapted from the official Real-ESRGAN inference helper.
    It keeps part2_2 independent from the external basicsr/realesrgan package while
    supporting large images on small GPUs.
    """

    def __init__(
        self,
        model,
        scale: int,
        device,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        half: bool = True,
    ):
        self.model = model.to(device).eval()
        self.scale = int(scale)
        self.device = device
        self.tile_size = int(tile)
        self.tile_pad = int(tile_pad)
        self.pre_pad = int(pre_pad)
        self.half = bool(half)
        if self.half:
            self.model = self.model.half()

    def _pre_process(self, img):
        if self.half:
            img = img.half()
        if self.pre_pad > 0:
            img = F.pad(img, (0, self.pre_pad, 0, self.pre_pad), mode="reflect")
        return img

    def _post_process(self, output, original_h: int, original_w: int):
        out_h = original_h * self.scale
        out_w = original_w * self.scale
        return output[:, :, :out_h, :out_w]

    @torch.no_grad()
    def enhance_tensor(self, img):
        """
        Args:
            img: torch tensor [1, 3, H, W] in RGB order, range [0, 1].
        Returns:
            torch tensor [1, 3, H*scale, W*scale] in range [0, 1].
        """
        if img.dim() != 4 or img.shape[0] != 1:
            raise ValueError("Expected input tensor shape [1, 3, H, W].")
        _, _, original_h, original_w = img.shape
        img = img.to(self.device)
        img = self._pre_process(img)

        if self.tile_size > 0:
            output = self._tile_process(img)
        else:
            output = self.model(img)

        output = self._post_process(output, original_h, original_w)
        return output.float().clamp(0, 1)

    def _tile_process(self, img):
        batch, channel, height, width = img.shape
        output = img.new_zeros((batch, channel, height * self.scale, width * self.scale))
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                input_start_x = x * self.tile_size
                input_end_x = min(input_start_x + self.tile_size, width)
                input_start_y = y * self.tile_size
                input_end_y = min(input_start_y + self.tile_size, height)

                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                input_tile = img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]
                output_tile = self.model(input_tile)

                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                output[
                    :,
                    :,
                    output_start_y:output_end_y,
                    output_start_x:output_end_x,
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

        return output

