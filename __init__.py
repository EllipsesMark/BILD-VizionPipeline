"""
BILD-VizionPipeline — Unified ComfyUI custom node pack.

Combines post-processing tools, LoRA batch loading, and iPhone photo
authentication into a single organized pack under the BILD brand.
"""

# ── LoRA nodes ───────────────────────────────────────────────────────
from .nodes.lora_batch import (
    BILDLoraLoaderFromFolder,
    BILDLoraLoaderModelOnly,
)

# ── Post-processing nodes ────────────────────────────────────────────
from .nodes.post_processing import (
    BILDMotionBlur,
    BILDFilmGrainSimple,
    BILDBrightnessContrast,
    BILDGaussianBlur,
    BILDVignette,
    BILDSaturation,
    BILDUnsharpMask,
    BILDGamma,
)

# ── iPhone Authenticate nodes ────────────────────────────────────────
from .nodes.auto_white_balance import BILDAutoWhiteBalance
from .nodes.lens_effects import BILDLensEffects
from .nodes.camera_simulator import BILDCameraSimulator
from .nodes.smart_hdr import BILDSmartHDR
from .nodes.deep_fusion import BILDDeepFusion
from .nodes.compression import BILDCompressionArtifacts
from .nodes.film_grain import BILDFilmGrain
from .nodes.metadata_inject import BILDMetadataInject
from .nodes.authenticate_all import BILDAuthenticateAll

# ── Registration ─────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    # LoRA
    "BILDLoraLoaderFromFolder": BILDLoraLoaderFromFolder,
    "BILDLoraLoaderModelOnly": BILDLoraLoaderModelOnly,
    # Post-processing
    "BILDMotionBlur": BILDMotionBlur,
    "BILDFilmGrainSimple": BILDFilmGrainSimple,
    "BILDBrightnessContrast": BILDBrightnessContrast,
    "BILDGaussianBlur": BILDGaussianBlur,
    "BILDVignette": BILDVignette,
    "BILDSaturation": BILDSaturation,
    "BILDUnsharpMask": BILDUnsharpMask,
    "BILDGamma": BILDGamma,
    # iPhone Authenticate
    "BILDAutoWhiteBalance": BILDAutoWhiteBalance,
    "BILDLensEffects": BILDLensEffects,
    "BILDCameraSimulator": BILDCameraSimulator,
    "BILDSmartHDR": BILDSmartHDR,
    "BILDDeepFusion": BILDDeepFusion,
    "BILDCompressionArtifacts": BILDCompressionArtifacts,
    "BILDFilmGrain": BILDFilmGrain,
    "BILDMetadataInject": BILDMetadataInject,
    "BILDAuthenticateAll": BILDAuthenticateAll,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # LoRA
    "BILDLoraLoaderFromFolder": "BILD Load LoRAs From Folder",
    "BILDLoraLoaderModelOnly": "BILD Load LoRAs From Folder (Model Only)",
    # Post-processing
    "BILDMotionBlur": "BILD Motion Blur",
    "BILDFilmGrainSimple": "BILD Film Grain (Simple)",
    "BILDBrightnessContrast": "BILD Brightness & Contrast",
    "BILDGaussianBlur": "BILD Gaussian Blur",
    "BILDVignette": "BILD Vignette",
    "BILDSaturation": "BILD Saturation",
    "BILDUnsharpMask": "BILD Unsharp Mask",
    "BILDGamma": "BILD Gamma",
    # iPhone Authenticate
    "BILDAutoWhiteBalance": "BILD Auto White Balance",
    "BILDLensEffects": "BILD Lens Effects",
    "BILDCameraSimulator": "BILD Camera Simulator",
    "BILDSmartHDR": "BILD Smart HDR",
    "BILDDeepFusion": "BILD Deep Fusion",
    "BILDCompressionArtifacts": "BILD Compression Artifacts",
    "BILDFilmGrain": "BILD Film Grain",
    "BILDMetadataInject": "BILD Metadata Inject",
    "BILDAuthenticateAll": "BILD Authenticate All",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
