"""
BILD LoRA Batch Loader nodes.

Loads and applies every LoRA file in a folder to MODEL (and optionally CLIP)
in alphabetical order — like chaining multiple Load LoRA nodes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import comfy.sd
import comfy.utils

_LORA_EXTENSIONS = (".safetensors", ".pt")


def _comfy_root() -> str:
    """Parent of custom_nodes — ComfyUI install root."""
    here = Path(__file__).resolve().parent
    return str(here.parent.parent.parent)


def _resolve_folder_path(folder_path: str) -> str:
    raw = (folder_path or "").strip()
    if not raw:
        raise ValueError("folder_path is empty — set a folder path (absolute or relative to ComfyUI root).")
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(_comfy_root(), raw))


def _collect_lora_paths(folder: str, recursive: bool) -> list[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {folder}")

    paths: list[str] = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                lower = name.lower()
                if any(lower.endswith(ext) for ext in _LORA_EXTENSIONS):
                    paths.append(os.path.join(root, name))
    else:
        try:
            for name in os.listdir(folder):
                full = os.path.join(folder, name)
                if not os.path.isfile(full):
                    continue
                lower = name.lower()
                if any(lower.endswith(ext) for ext in _LORA_EXTENSIONS):
                    paths.append(full)
        except OSError as e:
            raise OSError(f"Cannot read folder: {folder}") from e

    paths.sort(key=lambda p: p.lower())
    return paths


def _apply_loras_sequential(model, clip, paths: list[str], strength_model: float, strength_clip: float, cache: dict):
    if strength_model == 0 and strength_clip == 0:
        return model, clip

    m, c = model, clip
    for lora_path in paths:
        lora = cache.get(lora_path)
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            cache[lora_path] = lora
        m, c = comfy.sd.load_lora_for_models(m, c, lora, strength_model, strength_clip)
    return m, c


class BILDLoraLoaderFromFolder:
    """Apply every LoRA file in a folder to MODEL and CLIP (same strengths each)."""

    def __init__(self):
        self._lora_cache: dict[str, object] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base diffusion model."}),
                "clip": ("CLIP", {"tooltip": "Base CLIP model."}),
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Absolute path, or path relative to ComfyUI root (e.g. models/loras/my_pack).",
                    },
                ),
                "recursive": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "If true, include LoRA files in subfolders."},
                ),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Strength applied to each LoRA on the diffusion model.",
                    },
                ),
                "strength_clip": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Strength applied to each LoRA on CLIP.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("Model with all folder LoRAs applied in sorted order.", "CLIP with all folder LoRAs applied.")
    FUNCTION = "load_loras_from_folder"
    CATEGORY = "BILD/VizionPipeline/LoRA"
    DESCRIPTION = (
        "Loads and applies every .safetensors / .pt LoRA in a folder in alphabetical order "
        "(like chaining multiple Load LoRA nodes)."
    )

    def load_loras_from_folder(self, model, clip, folder_path: str, recursive: bool, strength_model: float, strength_clip: float):
        resolved = _resolve_folder_path(folder_path)
        paths = _collect_lora_paths(resolved, recursive)
        if not paths:
            logging.warning(
                "BILDLoraLoaderFromFolder: no LoRA files (%s) in %s — returning inputs unchanged.",
                ", ".join(_LORA_EXTENSIONS),
                resolved,
            )
            return (model, clip)

        out = _apply_loras_sequential(model, clip, paths, strength_model, strength_clip, self._lora_cache)
        return out


class BILDLoraLoaderModelOnly(BILDLoraLoaderFromFolder):
    """Same as folder LoRA loader but only patches the diffusion model."""

    DESCRIPTION = (
        "Loads every .safetensors / .pt LoRA in a folder onto the diffusion model only "
        "(alphabetical order, same strength each)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Absolute path, or path relative to ComfyUI root.",
                    },
                ),
                "recursive": ("BOOLEAN", {"default": False}),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("Model with all folder LoRAs applied.",)
    FUNCTION = "load_loras_model_only_from_folder"
    CATEGORY = "BILD/VizionPipeline/LoRA"

    def load_loras_model_only_from_folder(self, model, folder_path: str, recursive: bool, strength_model: float):
        resolved = _resolve_folder_path(folder_path)
        paths = _collect_lora_paths(resolved, recursive)
        if not paths:
            logging.warning(
                "BILDLoraLoaderModelOnly: no LoRA files (%s) in %s — returning model unchanged.",
                ", ".join(_LORA_EXTENSIONS),
                resolved,
            )
            return (model,)

        m, _ = _apply_loras_sequential(model, None, paths, strength_model, 0.0, self._lora_cache)
        return (m,)
