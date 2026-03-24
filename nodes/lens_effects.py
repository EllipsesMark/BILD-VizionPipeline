"""
BILDLensEffects — Chromatic aberration and vignette simulation.

Chromatic Aberration:
  Simulates lateral CA by displacing the red and blue channels in opposite
  directions from the image center. The green channel stays fixed.
  Displacement increases with distance from center (radial).

Vignette:
  Simulates natural corner darkening from lens optics using a radial
  falloff mask: brightness = 1 - strength * r^2.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

from ..utils.tensor_ops import bhwc_to_bchw, bchw_to_bhwc


class BILDLensEffects:
    """Simulates lens chromatic aberration and vignette darkening."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ca_strength": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Chromatic aberration strength in pixels of max displacement.",
                    },
                ),
                "vignette_strength": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Vignette darkening strength. 0 = none, 1 = heavy corner darkening.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline/Authenticate"
    DESCRIPTION = "Simulates lens chromatic aberration (RGB channel displacement) and vignette (corner darkening)."

    def apply(self, image: torch.Tensor, ca_strength: float, vignette_strength: float):
        B, H, W, C = image.shape
        device = image.device
        dtype = image.dtype
        result = image.clone()

        # ─── Chromatic Aberration ───
        if ca_strength > 0:
            yy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            xx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

            r = torch.sqrt(grid_x ** 2 + grid_y ** 2).clamp(min=1e-6)

            pixel_to_norm_x = 2.0 / W
            pixel_to_norm_y = 2.0 / H
            ca_norm = ca_strength * 0.5 * (pixel_to_norm_x + pixel_to_norm_y)

            dir_x = grid_x / r
            dir_y = grid_y / r
            displacement = ca_norm * r

            dx_r = dir_x * displacement
            dy_r = dir_y * displacement
            dx_b = -dir_x * displacement
            dy_b = -dir_y * displacement

            base_grid = torch.stack([grid_x, grid_y], dim=-1)
            base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

            grid_r = base_grid.clone()
            grid_r[..., 0] = grid_r[..., 0] + dx_r.unsqueeze(0)
            grid_r[..., 1] = grid_r[..., 1] + dy_r.unsqueeze(0)

            grid_b = base_grid.clone()
            grid_b[..., 0] = grid_b[..., 0] + dx_b.unsqueeze(0)
            grid_b[..., 1] = grid_b[..., 1] + dy_b.unsqueeze(0)

            img_bchw = bhwc_to_bchw(image)

            r_channel = F.grid_sample(
                img_bchw[:, 0:1], grid_r, mode="bilinear",
                padding_mode="border", align_corners=True
            )
            b_channel = F.grid_sample(
                img_bchw[:, 2:3], grid_b, mode="bilinear",
                padding_mode="border", align_corners=True
            )

            ca_result = torch.cat([r_channel, img_bchw[:, 1:2], b_channel], dim=1)
            result = bchw_to_bhwc(ca_result)

        # ─── Vignette ───
        if vignette_strength > 0:
            yy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            xx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

            r_sq = grid_x ** 2 + grid_y ** 2
            vignette_mask = 1.0 - vignette_strength * r_sq
            vignette_mask = torch.clamp(vignette_mask, 0.0, 1.0)
            vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(-1)

            result = result * vignette_mask

        return (torch.clamp(result, 0.0, 1.0),)
