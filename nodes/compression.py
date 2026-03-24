"""
BILDCompressionArtifacts — Multi-cycle JPEG compression and HEIF-like banding.

Simulates:
1. Multi-cycle JPEG compression (re-encoding that happens on social media sharing)
2. HEIF-like gradient banding in smooth areas (iPhone's default format artifact)

Requires PIL for actual JPEG encode/decode.
"""

from __future__ import annotations
import io
import torch
import numpy as np
from PIL import Image


class BILDCompressionArtifacts:
    """Simulates JPEG re-compression cycles and HEIF-like gradient banding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "quality": (
                    "INT",
                    {
                        "default": 85,
                        "min": 60,
                        "max": 98,
                        "step": 1,
                        "tooltip": "JPEG quality for each compression cycle. Lower = more artifacts.",
                    },
                ),
                "cycles": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 3,
                        "step": 1,
                        "tooltip": "Number of JPEG compression-decompression cycles.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline/Authenticate"
    DESCRIPTION = (
        "Simulates multi-cycle JPEG compression artifacts and HEIF-like "
        "gradient banding that occurs in real iPhone image processing."
    )

    def _jpeg_cycle(self, img_np: np.ndarray, quality: int) -> np.ndarray:
        """Single JPEG encode-decode cycle using PIL."""
        pil_img = Image.fromarray(img_np, mode="RGB")
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality, subsampling=0)
        buffer.seek(0)
        decoded = Image.open(buffer)
        decoded.load()
        return np.array(decoded)

    def _add_banding(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """Add subtle gradient banding in smooth areas (HEIF compression artifact)."""
        if strength <= 0:
            return image

        img_bchw = image.movedim(-1, 1).contiguous()
        B, C, H, W = img_bchw.shape

        grad_x = torch.abs(img_bchw[:, :, :, 1:] - img_bchw[:, :, :, :-1])
        grad_y = torch.abs(img_bchw[:, :, 1:, :] - img_bchw[:, :, :-1, :])

        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0), mode="replicate")
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1), mode="replicate")

        grad_mag = (grad_x + grad_y).mean(dim=1, keepdim=True)

        smooth_mask = torch.exp(-grad_mag * 50.0)
        smooth_mask = smooth_mask.movedim(1, -1)

        num_levels = int(256 - strength * 200)
        num_levels = max(8, num_levels)

        quantized = torch.round(image * num_levels) / num_levels

        banding_amount = strength * 0.5
        result = image * (1.0 - smooth_mask * banding_amount) + quantized * (smooth_mask * banding_amount)

        return result

    def apply(self, image: torch.Tensor, quality: int, cycles: int):
        B, H, W, C = image.shape
        device = image.device
        dtype = image.dtype

        results = []
        for b in range(B):
            img_np = (image[b].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            for cycle in range(cycles):
                q = quality + (cycle % 2) * 3 - 1
                q = max(1, min(100, q))
                img_np = self._jpeg_cycle(img_np, q)

            result = torch.from_numpy(img_np.astype(np.float32) / 255.0)
            results.append(result)

        result = torch.stack(results, dim=0).to(device=device, dtype=dtype)

        banding_strength = (98 - quality) / 98.0 * 0.3
        result = self._add_banding(result, banding_strength)

        return (torch.clamp(result, 0.0, 1.0),)
