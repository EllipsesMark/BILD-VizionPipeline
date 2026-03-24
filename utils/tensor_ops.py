"""
Shared tensor utilities for BILD-VizionPipeline nodes.

Common operations on ComfyUI IMAGE tensors [B, H, W, C] float32 [0,1].
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ── Layout conversions ──────────────────────────────────────────────

def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI IMAGE [B,H,W,C] → PyTorch conv layout [B,C,H,W]."""
    return x.movedim(-1, 1).contiguous()


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Convert PyTorch conv layout [B,C,H,W] → ComfyUI IMAGE [B,H,W,C]."""
    return x.movedim(1, -1).contiguous()


# ── Clamping ─────────────────────────────────────────────────────────

def clamp01(x: torch.Tensor) -> torch.Tensor:
    """Clamp tensor values to [0, 1]."""
    return torch.clamp(x, 0.0, 1.0)


# ── Luminance ────────────────────────────────────────────────────────

def rec709_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Rec.709 luminance from [B,H,W,C] RGB tensor → [B,H,W,1]."""
    r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ── Gaussian blur ────────────────────────────────────────────────────

def gaussian_kernel_1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a normalized 1D Gaussian kernel."""
    sigma = max(0.01, float(sigma))
    radius = max(1, int(math.ceil(sigma * 3.0)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def separable_gaussian_blur(bchw: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable Gaussian blur to a [B,C,H,W] tensor with replicate padding."""
    device, dtype = bchw.device, bchw.dtype
    k1d = gaussian_kernel_1d(sigma, device, dtype)
    k = k1d.numel()
    pad = k // 2
    _, c, _, _ = bchw.shape

    # Horizontal pass
    x = F.pad(bchw, (pad, pad, 0, 0), mode="replicate")
    w_h = k1d.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    x = F.conv2d(x, w_h, padding=0, groups=c)

    # Vertical pass
    x = F.pad(x, (0, 0, pad, pad), mode="replicate")
    w_v = k1d.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(x, w_v, padding=0, groups=c)

    return x


# ── Depthwise 2D convolution ────────────────────────────────────────

def depthwise_conv2d_same(bchw: torch.Tensor, kernel_2d: torch.Tensor) -> torch.Tensor:
    """Apply the same 2D kernel to each channel independently with replicate padding."""
    _, c, _, _ = bchw.shape
    kh, kw = kernel_2d.shape
    pad_h, pad_w = kh // 2, kw // 2
    x = F.pad(bchw, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
    weight = kernel_2d.view(1, 1, kh, kw).repeat(c, 1, 1, 1)
    return F.conv2d(x, weight, padding=0, groups=c)
