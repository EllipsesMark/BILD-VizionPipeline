"""
BILDAutoWhiteBalance — Gray world white balance with iPhone warm bias.

Algorithm:
1. Compute per-channel spatial mean of the input image (gray world assumption).
2. Compute a neutral target (equal channel means) at the desired luminance.
3. Blend the neutral target toward an iPhone-characteristic warm bias
   (slightly elevated red, slightly suppressed blue) controlled by iphone_bias.
4. Adjust color temperature by shifting the blue-red balance.
5. Scale each channel by target_mean / source_mean.
6. Blend result with original by strength (implicit via bias slider).
"""

from __future__ import annotations
import torch


class BILDAutoWhiteBalance:
    """Applies gray-world auto white balance with an iPhone warm color bias."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_temperature": (
                    "FLOAT",
                    {
                        "default": 5500.0,
                        "min": 3000.0,
                        "max": 9000.0,
                        "step": 100.0,
                        "tooltip": "Color temperature in Kelvin. Lower = warmer (more orange), higher = cooler (more blue). 5500 = neutral daylight.",
                    },
                ),
                "iphone_bias": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Strength of the iPhone warm-tone bias. 0 = pure gray world, 1 = full iPhone warmth.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline/Authenticate"
    DESCRIPTION = "Gray-world auto white balance with iPhone's characteristic warm color bias."

    def apply(self, image: torch.Tensor, color_temperature: float, iphone_bias: float):
        eps = 1e-6

        # Step 1: per-channel mean across spatial dims (gray world)
        source_mean = image.mean(dim=(1, 2), keepdim=True)

        # Step 2: neutral gray target
        avg_luminance = source_mean.mean(dim=-1, keepdim=True)
        neutral_target = avg_luminance.expand_as(source_mean)

        # Step 3: iPhone warm bias — R +3%, G +1%, B -4% relative to neutral
        iphone_warm = torch.tensor(
            [1.03, 1.01, 0.96], device=image.device, dtype=image.dtype
        ).view(1, 1, 1, 3)
        biased_target = neutral_target * torch.lerp(
            torch.ones_like(iphone_warm), iphone_warm, iphone_bias
        )

        # Step 4: color temperature adjustment
        temp_norm = (color_temperature - 5500.0) / 3500.0
        temp_norm = max(-1.0, min(1.0, temp_norm))

        temp_shift = torch.tensor(
            [-temp_norm * 0.04, 0.0, temp_norm * 0.04],
            device=image.device, dtype=image.dtype
        ).view(1, 1, 1, 3)
        biased_target = biased_target + temp_shift * avg_luminance

        # Step 5: per-channel scale factors
        scale = biased_target / (source_mean + eps)
        scale = torch.clamp(scale, 0.5, 2.0)

        # Step 6: apply correction
        corrected = image * scale

        return (torch.clamp(corrected, 0.0, 1.0),)
