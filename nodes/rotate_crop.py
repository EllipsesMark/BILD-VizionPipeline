"""
BILD Rotate & Crop — rotate an image and crop to the largest inscribed
rectangle that contains no blank space, matching Photoshop's
"crop to original size" rotation behavior.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..utils.tensor_ops import bhwc_to_bchw, bchw_to_bhwc, clamp01


def _largest_rotated_rect(w: int, h: int, angle_rad: float) -> tuple[float, float]:
    """Return (crop_w, crop_h) of the largest axis-aligned rectangle that
    fits entirely inside a w×h rectangle rotated by *angle_rad*.

    Uses the closed-form solution for the maximal-area inscribed rectangle.
    """
    angle_rad = abs(angle_rad) % math.pi
    if angle_rad > math.pi / 2:
        angle_rad = math.pi - angle_rad

    if abs(angle_rad) < 1e-10:
        return float(w), float(h)

    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)

    if w <= 2.0 * sin_a * cos_a * h or h <= 2.0 * sin_a * cos_a * w:
        # Fully constrained case: the crop rectangle is limited by the
        # shorter dimension of the rotated image.
        x = 0.5 * min(w, h) / sin_a
        if w < h:
            cw = x * cos_a - (x * sin_a - w / 2.0) * sin_a / cos_a
            ch = x * sin_a + (x * cos_a - w / (2.0 * cos_a)) * cos_a / sin_a
        else:
            cw = x * sin_a + (x * cos_a - h / (2.0 * cos_a)) * cos_a / sin_a
            ch = x * cos_a - (x * sin_a - h / 2.0) * sin_a / cos_a
        # Fallback: ensure positive
        cw, ch = abs(cw), abs(ch)
    else:
        cw = (w * cos_a - h * sin_a) / (cos_a * cos_a - sin_a * sin_a)
        ch = (h * cos_a - w * sin_a) / (cos_a * cos_a - sin_a * sin_a)

    return max(1.0, cw), max(1.0, ch)


class BILDRotateCrop:
    """Rotate an image by an arbitrary angle and crop to the largest
    rectangle that fits without any blank/empty corners."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -180.0,
                        "max": 180.0,
                        "step": 0.1,
                        "tooltip": "Rotation angle in degrees. Positive = counter-clockwise.",
                    },
                ),
                "interpolation": (
                    ["bilinear", "bicubic", "nearest"],
                    {
                        "default": "bilinear",
                        "tooltip": "Resampling method for the rotation.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline"
    DESCRIPTION = (
        "Rotates the image by the given angle and crops to the largest "
        "axis-aligned rectangle that fits entirely inside — no blank corners, "
        "like Photoshop's content-aware crop after rotation."
    )

    def apply(self, image: torch.Tensor, angle: float, interpolation: str):
        if abs(angle) < 1e-6:
            return (image,)

        b, h, w, c = image.shape
        angle_rad = math.radians(angle)

        crop_w, crop_h = _largest_rotated_rect(w, h, angle_rad)
        crop_w = max(1, int(math.floor(crop_w)))
        crop_h = max(1, int(math.floor(crop_h)))

        x = bhwc_to_bchw(image)  # [B, C, H, W]

        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)

        # Affine matrix that rotates around center then selects the crop region
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0],
             [sin_a,  cos_a, 0.0]],
            dtype=image.dtype,
            device=image.device,
        ).unsqueeze(0).expand(b, -1, -1)

        mode_map = {
            "bilinear": "bilinear",
            "bicubic": "bicubic",
            "nearest": "nearest",
        }
        grid = F.affine_grid(theta, [b, c, crop_h, crop_w], align_corners=False)
        rotated = F.grid_sample(
            x, grid,
            mode=mode_map[interpolation],
            padding_mode="zeros",
            align_corners=False,
        )

        return (clamp01(bchw_to_bhwc(rotated)),)
