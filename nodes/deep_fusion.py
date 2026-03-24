"""
BILDDeepFusion — Micro-contrast sharpening and skin tone smoothing.

iPhone Deep Fusion:
- Applies aggressive micro-contrast enhancement (sharpening at texture frequencies)
- Detects skin tones and selectively smooths them (reduces pore visibility)
- Creates that "hyper-detailed yet smooth skin" iPhone look
"""

from __future__ import annotations
import torch

from ..utils.tensor_ops import separable_gaussian_blur


class BILDDeepFusion:
    """Simulates iPhone Deep Fusion: micro-contrast sharpening with selective skin smoothing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpness": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Micro-contrast sharpening amount. iPhone Deep Fusion aggressively sharpens textures.",
                    },
                ),
                "skin_smooth": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Skin tone smoothing intensity. iPhones selectively smooth skin while keeping textures sharp.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline/Authenticate"
    DESCRIPTION = (
        "Simulates iPhone Deep Fusion: micro-contrast sharpening on textures "
        "with selective skin tone detection and smoothing."
    )

    def _detect_skin(self, image: torch.Tensor) -> torch.Tensor:
        """
        Detect skin tones using RGB ratio rules.
        Returns a soft mask [B, H, W, 1] where 1.0 = likely skin.
        """
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]

        warm = (r > g - 0.02) & (g > b - 0.02)

        rb_diff = r - b
        rb_ok = (rb_diff > 0.03) & (rb_diff < 0.45)

        max_rgb = torch.max(image, dim=-1).values
        min_rgb = torch.min(image, dim=-1).values
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
        sat_ok = saturation < 0.55

        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        bright_ok = (luminance > 0.15) & (luminance < 0.85)

        skin_mask = (warm & rb_ok & sat_ok & bright_ok).float()

        mask_bchw = skin_mask.unsqueeze(1)
        mask_smooth = separable_gaussian_blur(mask_bchw, sigma=3.0)
        mask_smooth = mask_smooth.squeeze(1).unsqueeze(-1)

        return mask_smooth.clamp(0, 1)

    def apply(self, image: torch.Tensor, sharpness: float, skin_smooth: float):
        B, H, W, C = image.shape
        result = image.clone()

        # ─── Micro-Contrast Sharpening (Unsharp Mask) ───
        if sharpness > 0:
            img_bchw = image.movedim(-1, 1).contiguous()

            sigma = 1.5
            blurred = separable_gaussian_blur(img_bchw, sigma)
            high_freq = img_bchw - blurred
            sharpened = img_bchw + sharpness * high_freq

            # Second pass at larger scale for multi-scale enhancement
            blurred_large = separable_gaussian_blur(img_bchw, sigma * 2.5)
            high_freq_large = img_bchw - blurred_large
            sharpened = sharpened + (sharpness * 0.3) * high_freq_large

            result = sharpened.movedim(1, -1).contiguous()

        # ─── Skin Tone Smoothing ───
        if skin_smooth > 0:
            skin_mask = self._detect_skin(image)

            img_bchw = result.movedim(-1, 1).contiguous()
            smooth_sigma = 2.0 + skin_smooth * 3.0
            smoothed = separable_gaussian_blur(img_bchw, smooth_sigma)
            smoothed_bhwc = smoothed.movedim(1, -1).contiguous()

            blend = skin_smooth * skin_mask
            result = result * (1.0 - blend) + smoothed_bhwc * blend

        return (torch.clamp(result, 0.0, 1.0),)
