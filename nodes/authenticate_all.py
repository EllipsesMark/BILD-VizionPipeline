"""
BILDAuthenticateAll — All-in-one iPhone authentication node.

Chains all BILD-Authenticate processing stages in the correct order.
Single "strength" slider (0.0-1.0) proportionally controls all sub-effects.
Camera model selection affects visual processing (not just metadata).
"""

from __future__ import annotations
import torch

from .auto_white_balance import BILDAutoWhiteBalance
from .smart_hdr import BILDSmartHDR
from .deep_fusion import BILDDeepFusion
from .film_grain import BILDFilmGrain
from .camera_simulator import BILDCameraSimulator
from .lens_effects import BILDLensEffects
from .compression import BILDCompressionArtifacts
from .metadata_inject import BILDMetadataInject


# Camera-specific processing profiles
CAMERA_PROFILES = {
    "Main 1x": {
        "awb_temp": 5800,       # Main sensor warm bias
        "awb_bias": 0.12,       # Subtle iPhone warmth
        "hdr_shadow": 0.15,     # Mild shadow lift
        "hdr_highlight": 0.10,  
        "fusion_sharp": 0.25,   # Moderate Deep Fusion
        "fusion_skin": 0.15,    
        "grain_amount": 0.06,   # Light grain
        "grain_size": 1.0,
        "noise_strength": 0.05, # Very light sensor noise
        "ca_strength": 0.4,     # Subtle chromatic aberration
        "vignette": 0.06,       # Barely there vignette
    },
    "Telephoto 5x": {
        "awb_temp": 5600,
        "awb_bias": 0.10,
        "hdr_shadow": 0.12,
        "hdr_highlight": 0.08,
        "fusion_sharp": 0.35,   # More sharpening (telephoto needs it)
        "fusion_skin": 0.12,
        "grain_amount": 0.08,   # Slightly more grain (smaller sensor area)
        "grain_size": 0.8,
        "noise_strength": 0.08,
        "ca_strength": 0.2,     # Less CA on telephoto
        "vignette": 0.03,       # Minimal vignette
    },
    "Ultra-Wide 0.5x": {
        "awb_temp": 5500,
        "awb_bias": 0.10,
        "hdr_shadow": 0.20,     # More HDR on ultra-wide
        "hdr_highlight": 0.15,
        "fusion_sharp": 0.20,
        "fusion_skin": 0.10,
        "grain_amount": 0.07,
        "grain_size": 1.2,
        "noise_strength": 0.06,
        "ca_strength": 0.8,     # More CA on ultra-wide edges
        "vignette": 0.12,       # Noticeable vignette on ultra-wide
    },
    "Selfie": {
        "awb_temp": 6000,       # Warmer for skin tones
        "awb_bias": 0.15,
        "hdr_shadow": 0.18,
        "hdr_highlight": 0.12,
        "fusion_sharp": 0.15,   # Less sharpening on selfie
        "fusion_skin": 0.25,    # More skin smoothing
        "grain_amount": 0.05,
        "grain_size": 0.9,
        "noise_strength": 0.04,
        "ca_strength": 0.3,
        "vignette": 0.05,
    },
    "Portrait Mode": {
        "awb_temp": 5700,
        "awb_bias": 0.12,
        "hdr_shadow": 0.10,
        "hdr_highlight": 0.08,
        "fusion_sharp": 0.30,   # Strong subject sharpening
        "fusion_skin": 0.20,    # Good skin smoothing
        "grain_amount": 0.04,   # Less grain in portrait mode
        "grain_size": 1.0,
        "noise_strength": 0.03,
        "ca_strength": 0.3,
        "vignette": 0.04,
    },
}


class BILDAuthenticateAll:
    """All-in-one iPhone authentication with camera-specific processing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Overall authentication strength. 0.3-0.5 recommended for subtle, 0.6-0.8 for noticeable.",
                    },
                ),
                "camera_model": (
                    ["iPhone 15 Pro Max", "iPhone 16 Pro Max", "iPhone 17 Pro Max"],
                    {"default": "iPhone 17 Pro Max"},
                ),
                "camera_mode": (
                    list(CAMERA_PROFILES.keys()),
                    {"default": "Main 1x"},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "metadata_json",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline/Authenticate"
    DESCRIPTION = (
        "All-in-one iPhone photo authentication. Camera mode selection changes "
        "processing characteristics (lens, noise, HDR, grain). Strength 0.3-0.5 "
        "for subtle authenticity, 0.6+ for stronger effect."
    )

    def apply(self, image: torch.Tensor, strength: float, camera_model: str, camera_mode: str, seed: int):
        if strength <= 0:
            return (image, "{}",)

        s = strength
        profile = CAMERA_PROFILES.get(camera_mode, CAMERA_PROFILES["Main 1x"])
        result = image

        # Stage 1: Auto White Balance — subtle iPhone warmth
        awb = BILDAutoWhiteBalance()
        result = awb.apply(result, profile["awb_temp"], profile["awb_bias"] * s)[0]

        # Stage 2: Smart HDR — lift shadows, compress highlights
        hdr = BILDSmartHDR()
        result = hdr.apply(result, profile["hdr_shadow"] * s, profile["hdr_highlight"] * s)[0]

        # Stage 3: Deep Fusion — texture sharpening + skin smoothing
        fusion = BILDDeepFusion()
        result = fusion.apply(result, profile["fusion_sharp"] * s, profile["fusion_skin"] * s)[0]

        # Stage 4: Film Grain — luminance-weighted
        grain = BILDFilmGrain()
        result = grain.apply(
            result,
            profile["grain_amount"] * s,
            profile["grain_size"],
            0.05 * s,  # Very subtle color grain
            seed % (2**63),
        )[0]

        # Stage 5: Camera Simulator — sensor noise + hot pixels
        cam = BILDCameraSimulator()
        result = cam.apply(
            result,
            profile["noise_strength"] * s,
            0.00003 * s,  # Very few hot pixels
            (seed + 1) % (2**63),
        )[0]

        # Stage 6: Lens Effects — chromatic aberration + vignette
        lens = BILDLensEffects()
        result = lens.apply(result, profile["ca_strength"] * s, profile["vignette"] * s)[0]

        # Stage 7: Compression — only at higher strengths
        if s > 0.3:
            comp = BILDCompressionArtifacts()
            quality = int(96 - s * 6)  # 93-96 range (very subtle)
            result = comp.apply(result, quality, 1)[0]

        # Metadata
        meta_mode = camera_mode if camera_mode != "Portrait Mode" else "Main 1x"
        meta = BILDMetadataInject()
        _, metadata_json = meta.apply(result, camera_model, meta_mode, True)

        return (result, metadata_json,)
