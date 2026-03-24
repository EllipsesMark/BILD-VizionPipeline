"""
BILD Post-Processing nodes for ComfyUI IMAGE tensors [B,H,W,C], float, [0,1].

Nodes:
- BILDMotionBlur: Linear motion blur via line kernel convolution
- BILDBrightnessContrast: Linear brightness/contrast adjustment
- BILDGaussianBlur: Separable Gaussian blur
- BILDVignette: Radial corner darkening
- BILDSaturation: Rec.709 luminance-based saturation
- BILDUnsharpMask: High-frequency sharpening boost
- BILDGamma: Power-curve gamma adjustment
- BILDFilmGrainSimple: Basic additive Gaussian film grain (luminance or RGB)
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from ..utils.tensor_ops import (
    bhwc_to_bchw,
    bchw_to_bhwc,
    clamp01,
    rec709_luminance,
    separable_gaussian_blur,
    depthwise_conv2d_same,
)


def _motion_blur_kernel(angle_deg: float, length_px: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Uniform weights along a line segment; angle in degrees (0 = horizontal streak, 90 = vertical)."""
    length_px = max(0.5, float(length_px))
    half = int(max(1, math.ceil(length_px * 0.5)))
    ks = half * 2 + 1
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    kernel = torch.zeros((ks, ks), device=device, dtype=dtype)
    cx = cy = ks // 2
    for t in range(-half, half + 1):
        xi = int(round(cx + c * t))
        yi = int(round(cy + s * t))
        if 0 <= xi < ks and 0 <= yi < ks:
            kernel[yi, xi] += 1.0
    ssum = kernel.sum()
    if ssum > 0:
        kernel /= ssum
    else:
        kernel[cy, cx] = 1.0
    return kernel


def _film_grain_noise(
    shape: tuple[int, ...],
    mode: Literal["luminance", "rgb"],
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mode == "rgb":
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
    n = torch.randn(shape[0], shape[1], shape[2], 1, generator=generator, device=device, dtype=dtype)
    return n.expand(-1, -1, -1, 3)


class BILDMotionBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -360.0,
                        "max": 360.0,
                        "step": 0.1,
                        "tooltip": "Direction of the blur streak in degrees (0 = horizontal, 90 = vertical).",
                    },
                ),
                "distance": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.0,
                        "max": 512.0,
                        "step": 0.5,
                        "tooltip": "Approximate streak length in pixels (kernel extent).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline"
    DESCRIPTION = (
        "Simulates linear motion during exposure by convolving with a line kernel "
        "(uniform sampling along a segment)."
    )

    def apply(self, image: torch.Tensor, angle: float, distance: float):
        if distance <= 0:
            return (image,)
        k = _motion_blur_kernel(angle, distance, image.device, image.dtype)
        x = bhwc_to_bchw(image)
        y = depthwise_conv2d_same(x, k)
        return (clamp01(bchw_to_bhwc(y)),)


class BILDFilmGrainSimple:
    """Basic additive film grain — Gaussian noise in luminance or RGB mode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.005,
                        "tooltip": "Scale of additive Gaussian noise (typical 0.02–0.15).",
                    },
                ),
                "grain_type": (
                    ["luminance", "rgb"],
                    {
                        "tooltip": "luminance: same noise on R,G,B (neutral grain). rgb: independent per channel (color speckle).",
                    },
                ),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "tooltip": "Leave at 0 for random."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline"
    DESCRIPTION = (
        "Simple additive film-style grain: Gaussian noise. Luminance mode ties channels (silver-halide style); "
        "RGB mode decorrelates channels for stronger color noise."
    )

    def apply(self, image: torch.Tensor, strength: float, grain_type: str, seed: int = 0):
        import random as _rand
        if seed == 0:
            seed = _rand.randint(1, 2**62)
        if strength <= 0:
            return (image,)
        gen = torch.Generator(device=image.device)
        gen.manual_seed(int(seed) % (2**31))
        n = _film_grain_noise(image.shape, grain_type, gen, image.device, image.dtype)
        return (clamp01(image + strength * n),)


class BILDBrightnessContrast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Added after contrast, in linear [0,1] space.",
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "Scaled around mid-gray 0.5 (1.0 = unchanged).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline"
    DESCRIPTION = "Linear brightness (offset) and contrast (gain around 0.5), then clamp to [0,1]."

    def apply(self, image: torch.Tensor, brightness: float, contrast: float):
        out = (image - 0.5) * contrast + 0.5 + brightness
        return (clamp01(out),)


class BILDGaussianBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sigma": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.01,
                        "max": 64.0,
                        "step": 0.05,
                        "tooltip": "Gaussian standard deviation in pixels (separable blur).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline"
    DESCRIPTION = "Separable Gaussian blur with replicate padding (edge-friendly)."

    def apply(self, image: torch.Tensor, sigma: float):
        x = bhwc_to_bchw(image)
        y = separable_gaussian_blur(x, sigma)
        return (clamp01(bchw_to_bhwc(y)),)


class BILDVignette:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How dark the corners become.",
                    },
                ),
                "feather": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.05,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Higher = tighter falloff toward corners (power on radial distance).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline"
    DESCRIPTION = "Multiplies image by a radial mask (center 1, corners darker)."

    def apply(self, image: torch.Tensor, strength: float, feather: float):
        if strength <= 0:
            return (image,)
        b, h, w, _ = image.shape
        device, dtype = image.device, image.dtype
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, h, 1).expand(b, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, w).expand(b, h, w)
        r = torch.sqrt(xx * xx + yy * yy)
        r = torch.clamp(r / 1.41421356, 0.0, 1.0)
        mask = 1.0 - strength * (r**feather)
        mask = mask.unsqueeze(-1)
        return (clamp01(image * mask),)


class BILDSaturation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "saturation": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "1.0 = original; 0 = grayscale; >1 boosts chroma.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline"
    DESCRIPTION = "Rec.709 luminance vs chroma blend (saturation adjustment)."

    def apply(self, image: torch.Tensor, saturation: float):
        gray = rec709_luminance(image)
        gray_rgb = gray.expand(-1, -1, -1, 3)
        out = gray_rgb + saturation * (image - gray_rgb)
        return (clamp01(out),)


class BILDUnsharpMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": "Strength of high-frequency boost.",
                    },
                ),
                "sigma": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 32.0,
                        "step": 0.05,
                        "tooltip": "Blur radius used to extract detail (Gaussian sigma).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline"
    DESCRIPTION = "Classic sharpening: image + amount * (image - gaussian_blur(image))."

    def apply(self, image: torch.Tensor, amount: float, sigma: float):
        if amount <= 0:
            return (image,)
        x = bhwc_to_bchw(image)
        blur = separable_gaussian_blur(x, sigma)
        high = x - blur
        y = x + amount * high
        return (clamp01(bchw_to_bhwc(y)),)


class BILDGamma:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gamma": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.01,
                        "tooltip": "Output = input^gamma. >1 darkens midtones; <1 brightens. 1.0 = unchanged.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline"
    DESCRIPTION = "Power-curve adjustment: out = in^gamma (common photo-style gamma slider)."

    def apply(self, image: torch.Tensor, gamma: float):
        gamma = max(0.1, float(gamma))
        return (clamp01(image**gamma),)
