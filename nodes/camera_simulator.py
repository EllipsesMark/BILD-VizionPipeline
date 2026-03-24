"""
BILDCameraSimulator — Sensor noise, hot pixels, and Bayer CFA demosaic simulation.

Sub-stages:
1. Bayer CFA mosaic + bilinear demosaic → introduces cross-channel aliasing
2. Poisson-Gaussian sensor noise (shot noise + read noise)
3. Hot pixel injection (stuck bright pixels)

All operations use torch for GPU acceleration.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


class BILDCameraSimulator:
    """Simulates digital camera sensor artifacts: noise, hot pixels, Bayer demosaic."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_strength": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Overall sensor noise intensity. Controls both shot noise and read noise.",
                    },
                ),
                "hot_pixel_density": (
                    "FLOAT",
                    {
                        "default": 0.0001,
                        "min": 0.0,
                        "max": 0.001,
                        "step": 0.00001,
                        "tooltip": "Fraction of pixels that become 'stuck' hot pixels.",
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline/Authenticate"
    DESCRIPTION = (
        "Simulates camera sensor artifacts: Bayer CFA demosaic bleeding, "
        "Poisson-Gaussian sensor noise, and hot pixel defects."
    )

    def _bayer_demosaic(self, image: torch.Tensor) -> torch.Tensor:
        """
        Simulate Bayer CFA mosaic and bilinear demosaic reconstruction.
        Introduces subtle cross-channel bleeding artifacts characteristic
        of real single-sensor cameras with RGGB color filter arrays.
        """
        B, H, W, C = image.shape
        device = image.device
        dtype = image.dtype

        mosaic = torch.zeros_like(image)

        row_even = torch.arange(H, device=device) % 2 == 0
        col_even = torch.arange(W, device=device) % 2 == 0

        r_mask = row_even.unsqueeze(1) & col_even.unsqueeze(0)
        g_mask1 = row_even.unsqueeze(1) & (~col_even.unsqueeze(0))
        g_mask2 = (~row_even.unsqueeze(1)) & col_even.unsqueeze(0)
        b_mask = (~row_even.unsqueeze(1)) & (~col_even.unsqueeze(0))

        mosaic[..., 0] = image[..., 0] * r_mask.float().unsqueeze(0)
        mosaic[..., 1] = image[..., 1] * (g_mask1 | g_mask2).float().unsqueeze(0)
        mosaic[..., 2] = image[..., 2] * b_mask.float().unsqueeze(0)

        mosaic_bchw = mosaic.movedim(-1, 1).contiguous()

        r_count = r_mask.float().unsqueeze(0).unsqueeze(0)
        g_count = (g_mask1 | g_mask2).float().unsqueeze(0).unsqueeze(0)
        b_count = b_mask.float().unsqueeze(0).unsqueeze(0)
        count_mask = torch.cat([r_count, g_count, b_count], dim=1)

        kernel = torch.ones(1, 1, 3, 3, device=device, dtype=dtype) / 9.0
        kernel_3ch = kernel.repeat(3, 1, 1, 1)

        padded_mosaic = F.pad(mosaic_bchw, (1, 1, 1, 1), mode="replicate")
        padded_count = F.pad(count_mask.expand(B, -1, -1, -1), (1, 1, 1, 1), mode="replicate")

        value_sum = F.conv2d(padded_mosaic, kernel_3ch, padding=0, groups=3)
        count_sum = F.conv2d(padded_count, kernel_3ch, padding=0, groups=3)

        demosaiced = value_sum / (count_sum + 1e-6)

        original_bchw = image.movedim(-1, 1).contiguous()
        has_data = count_mask.expand(B, -1, -1, -1)
        result_bchw = torch.where(has_data > 0.5, original_bchw, demosaiced)

        blend_factor = 0.15
        result_bchw = result_bchw * (1.0 - blend_factor) + demosaiced * blend_factor

        return result_bchw.movedim(1, -1).contiguous()

    def _sensor_noise(self, image: torch.Tensor, strength: float, generator: torch.Generator) -> torch.Tensor:
        """Poisson-Gaussian noise model: shot noise + read noise."""
        if strength <= 0:
            return image

        photon_scale = 50.0 / (strength + 0.01)

        scaled = image * photon_scale
        shot_noise = torch.randn_like(image, generator=generator if image.is_cpu else None) * torch.sqrt(scaled.clamp(min=0.01))
        noisy = (scaled + shot_noise) / photon_scale

        read_noise_std = strength * 0.02
        read_noise = torch.randn_like(image) * read_noise_std
        noisy = noisy + read_noise

        return noisy

    def _hot_pixels(self, image: torch.Tensor, density: float, generator: torch.Generator) -> torch.Tensor:
        """Inject stuck bright pixels at random positions."""
        if density <= 0:
            return image

        B, H, W, C = image.shape
        result = image.clone()

        num_pixels = int(H * W * density)
        if num_pixels < 1:
            return image

        for b in range(B):
            hot_y = torch.randint(0, H, (num_pixels,), device=image.device)
            hot_x = torch.randint(0, W, (num_pixels,), device=image.device)
            hot_values = 0.8 + 0.2 * torch.rand(num_pixels, C, device=image.device, dtype=image.dtype)
            result[b, hot_y, hot_x, :] = hot_values

        return result

    def apply(self, image: torch.Tensor, noise_strength: float, hot_pixel_density: float, seed: int):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed % (2**63))

        result = image.clone()

        if noise_strength > 0:
            result = self._bayer_demosaic(result)

        result = self._sensor_noise(result, noise_strength, gen)
        result = self._hot_pixels(result, hot_pixel_density, gen)

        return (torch.clamp(result, 0.0, 1.0),)
