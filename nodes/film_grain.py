"""
BILDFilmGrain — Procedural film grain with luminance-weighted noise.

Real digital sensor noise is:
- Stronger in shadows (shot noise is proportional to signal)
- Primarily luminance noise (achromatic) with subtle color component
- Has a characteristic grain size that depends on sensor/ISO

Algorithm:
1. Generate Gaussian noise at base resolution
2. Optionally upscale noise for larger grain size
3. Weight noise by luminance (stronger in shadows, weaker in highlights)
4. Add luminance noise to all channels equally (achromatic component)
5. Add small per-channel noise (chromatic component)
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


class BILDFilmGrain:
    """Procedural film grain: luminance-weighted noise with controllable grain size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Overall grain intensity. 0.05-0.15 = subtle film grain, 0.3+ = heavy grain.",
                    },
                ),
                "size": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": "Grain particle size. 1.0 = pixel-level, 2.0+ = coarser, film-like grain.",
                    },
                ),
                "color_amount": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "How much of the grain is chromatic (color noise). 0 = pure luminance noise, 1 = full color noise.",
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline/Authenticate"
    DESCRIPTION = (
        "Procedural film grain with luminance-weighted noise distribution. "
        "Grain is stronger in shadows, weaker in highlights, matching real sensor behavior."
    )

    def apply(self, image: torch.Tensor, amount: float, size: float, color_amount: float, seed: int):
        if amount <= 0:
            return (image,)

        B, H, W, C = image.shape
        device = image.device
        dtype = image.dtype

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        # Generate grain at potentially reduced resolution for larger grain size
        if size > 1.0:
            grain_h = max(1, int(H / size))
            grain_w = max(1, int(W / size))
        else:
            grain_h, grain_w = H, W

        luma_noise = torch.randn(B, grain_h, grain_w, 1, generator=gen, dtype=dtype, device="cpu").to(device)
        chroma_noise = torch.randn(B, grain_h, grain_w, C, generator=gen, dtype=dtype, device="cpu").to(device)

        # Upscale noise if grain size > 1
        if size > 1.0:
            luma_noise = luma_noise.movedim(-1, 1)
            luma_noise = F.interpolate(luma_noise, size=(H, W), mode="bilinear", align_corners=False)
            luma_noise = luma_noise.movedim(1, -1)

            chroma_noise = chroma_noise.movedim(-1, 1)
            chroma_noise = F.interpolate(chroma_noise, size=(H, W), mode="bilinear", align_corners=False)
            chroma_noise = chroma_noise.movedim(1, -1)

        # Luminance-dependent weighting
        luminance = (
            0.299 * image[..., 0:1]
            + 0.587 * image[..., 1:2]
            + 0.114 * image[..., 2:3]
        )

        shadow_weight = 1.0 - 0.6 * torch.sqrt(luminance.clamp(0, 1))

        # Combine achromatic and chromatic components
        achromatic = luma_noise.expand_as(image) * (1.0 - color_amount)
        chromatic = chroma_noise * color_amount

        combined_noise = (achromatic + chromatic) * shadow_weight * amount

        result = image + combined_noise

        return (torch.clamp(result, 0.0, 1.0),)
