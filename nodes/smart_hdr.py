"""
BILDSmartHDR — iPhone Smart HDR simulation.

iPhones aggressively process HDR:
- Shadows are lifted to reveal detail
- Highlights are compressed to prevent clipping
- Overall contrast is flattened (the "HDR look")
- Results in that distinctive "everything is visible" iPhone aesthetic
"""

from __future__ import annotations
import torch


class BILDSmartHDR:
    """Simulates iPhone Smart HDR: shadow lifting, highlight compression, contrast flattening."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_lift": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "How much to lift shadow regions. 0 = no lift, 1 = aggressive HDR shadow recovery.",
                    },
                ),
                "highlight_compress": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "How much to compress highlights. 0 = no compression, 1 = heavy highlight roll-off.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline/Authenticate"
    DESCRIPTION = (
        "Simulates iPhone Smart HDR: lifts shadows, compresses highlights, "
        "and flattens contrast for that characteristic iPhone look."
    )

    def apply(self, image: torch.Tensor, shadow_lift: float, highlight_compress: float):
        if shadow_lift <= 0 and highlight_compress <= 0:
            return (image,)

        result = image.clone()

        luminance = (
            0.2126 * result[..., 0:1]
            + 0.7152 * result[..., 1:2]
            + 0.0722 * result[..., 2:3]
        )

        # ─── Shadow Lifting ───
        if shadow_lift > 0:
            shadow_mask = torch.pow(1.0 - luminance.clamp(0, 1), 2.0)
            gamma = 1.0 - shadow_lift * 0.5
            lifted = torch.pow(result.clamp(min=1e-6), gamma)
            lift_amount = shadow_lift * shadow_mask
            result = result * (1.0 - lift_amount) + lifted * lift_amount

        # ─── Highlight Compression ───
        if highlight_compress > 0:
            highlight_mask = torch.pow(luminance.clamp(0, 1), 2.0)
            threshold = 1.0 - highlight_compress * 0.5
            above = (result - threshold).clamp(min=0)
            compress_factor = 1.0 + highlight_compress * 3.0
            compressed = threshold + above / compress_factor
            compress_amount = highlight_compress * highlight_mask
            result = torch.where(
                result > threshold,
                result * (1.0 - compress_amount) + compressed * compress_amount,
                result,
            )

        # ─── Subtle contrast flattening ───
        overall_strength = (shadow_lift + highlight_compress) * 0.1
        if overall_strength > 0:
            mean_val = result.mean(dim=(1, 2), keepdim=True)
            result = result * (1.0 - overall_strength) + mean_val * overall_strength

        return (torch.clamp(result, 0.0, 1.0),)
