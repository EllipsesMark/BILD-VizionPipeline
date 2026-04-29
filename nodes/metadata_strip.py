"""
BILDMetadataStrip — Strip EXIF/metadata from images.

Passes through the image unchanged but outputs an empty metadata JSON string,
effectively clearing any metadata that was injected by BILDMetadataInject or
other upstream nodes. Can also strip EXIF from image files on disk.
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
import torch
from PIL import Image


class BILDMetadataStrip:
    """Passes the image through and outputs an empty metadata JSON string,
    neutralizing any upstream BILDMetadataInject output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "metadata_json": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": "Optional metadata JSON from BILDMetadataInject — will be discarded.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "metadata_json",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline"
    DESCRIPTION = (
        "Strips all EXIF/metadata. The image passes through unchanged while "
        "any incoming metadata_json is replaced with an empty object. "
        "Pair with BILDMetadataInject to selectively clear metadata in a pipeline."
    )

    def apply(self, image: torch.Tensor, metadata_json: str = ""):
        return (image, "{}",)


class BILDMetadataStripFile:
    """Strips EXIF metadata from image files on disk and re-saves them clean."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "file_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Path to an image file on disk to strip EXIF from. "
                            "If empty, only the pass-through image is returned "
                            "(no file operation)."
                        ),
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If True, overwrites the original file. If False, saves as <name>_clean.<ext>.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "status",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline"
    DESCRIPTION = (
        "Strips ALL EXIF/metadata from an image file on disk by re-encoding it "
        "without metadata. The ComfyUI image tensor passes through unchanged."
    )

    def apply(self, image: torch.Tensor, file_path: str, overwrite: bool):
        file_path = file_path.strip()
        if not file_path or not os.path.isfile(file_path):
            return (image, json.dumps({"status": "skipped", "reason": "no valid file path"}),)

        try:
            img = Image.open(file_path)
            fmt = img.format or "JPEG"

            clean = Image.new(img.mode, img.size)
            clean.putdata(list(img.getdata()))

            if overwrite:
                out_path = file_path
            else:
                base, ext = os.path.splitext(file_path)
                out_path = f"{base}_clean{ext}"

            save_kwargs = {}
            if fmt.upper() in ("JPEG", "JPG"):
                save_kwargs["quality"] = 98
                save_kwargs["subsampling"] = 0
            elif fmt.upper() == "PNG":
                save_kwargs["compress_level"] = 6

            clean.save(out_path, format=fmt, **save_kwargs)

            status = {
                "status": "stripped",
                "input": file_path,
                "output": out_path,
                "format": fmt,
            }
        except Exception as e:
            status = {"status": "error", "message": str(e)}

        return (image, json.dumps(status, indent=2),)
