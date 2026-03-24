# BILD-VizionPipeline

Unified ComfyUI custom node pack combining post-processing tools, LoRA batch loading, and iPhone photo authentication — all under one roof.

## Node List

### LoRA Tools — `BILD/VizionPipeline/LoRA`

| Node | Class | Description |
|------|-------|-------------|
| **BILD Load LoRAs From Folder** | `BILDLoraLoaderFromFolder` | Apply every LoRA in a folder to MODEL + CLIP (alphabetical order, same strength) |
| **BILD Load LoRAs From Folder (Model Only)** | `BILDLoraLoaderModelOnly` | Same as above but patches diffusion model only (no CLIP) |

### Post-Processing — `BILD/VizionPipeline`

| Node | Class | Description |
|------|-------|-------------|
| **BILD Motion Blur** | `BILDMotionBlur` | Linear motion blur via line kernel convolution |
| **BILD Film Grain (Simple)** | `BILDFilmGrainSimple` | Basic additive Gaussian grain (luminance or RGB mode) |
| **BILD Brightness & Contrast** | `BILDBrightnessContrast` | Linear brightness offset + contrast gain around mid-gray |
| **BILD Gaussian Blur** | `BILDGaussianBlur` | Separable Gaussian blur with replicate padding |
| **BILD Vignette** | `BILDVignette` | Radial corner darkening with configurable feather |
| **BILD Saturation** | `BILDSaturation` | Rec.709 luminance-based saturation adjustment |
| **BILD Unsharp Mask** | `BILDUnsharpMask` | Classic high-frequency sharpening boost |
| **BILD Gamma** | `BILDGamma` | Power-curve gamma correction |

### iPhone Authenticate — `BILD/VizionPipeline/Authenticate`

| Node | Class | Description |
|------|-------|-------------|
| **BILD Auto White Balance** | `BILDAutoWhiteBalance` | Gray world AWB with iPhone warm color bias |
| **BILD Lens Effects** | `BILDLensEffects` | Chromatic aberration (RGB displacement) + vignette |
| **BILD Camera Simulator** | `BILDCameraSimulator` | Bayer CFA demosaic, Poisson-Gaussian sensor noise, hot pixels |
| **BILD Smart HDR** | `BILDSmartHDR` | Shadow lifting, highlight compression, contrast flattening |
| **BILD Deep Fusion** | `BILDDeepFusion` | Micro-contrast sharpening + selective skin smoothing |
| **BILD Compression Artifacts** | `BILDCompressionArtifacts` | Multi-cycle JPEG compression + HEIF-like gradient banding |
| **BILD Film Grain** | `BILDFilmGrain` | Luminance-weighted procedural grain with grain size + color control |
| **BILD Metadata Inject** | `BILDMetadataInject` | iPhone EXIF metadata (camera model, lens, GPS, datetime) |
| **BILD Authenticate All** | `BILDAuthenticateAll` | All-in-one: chains all auth nodes with a single strength slider |

### Film Grain: Two Versions

- **BILD Film Grain** (`BILDFilmGrain`) — Advanced luminance-weighted grain with controllable size and color amount. Stronger in shadows, weaker in highlights. Use this for realistic sensor noise.
- **BILD Film Grain (Simple)** (`BILDFilmGrainSimple`) — Basic additive Gaussian noise in luminance or RGB mode. Simpler, faster, good for quick stylization.

## Authenticate All — Processing Order

1. **Auto White Balance** → correct color, add iPhone warmth
2. **Smart HDR** → lift shadows, compress highlights
3. **Deep Fusion** → sharpen textures, smooth skin
4. **Film Grain** → add luminance-weighted sensor noise
5. **Camera Simulator** → Bayer artifacts, shot noise, hot pixels
6. **Lens Effects** → chromatic aberration, vignette
7. **Compression** → JPEG cycles, gradient banding

## Installation

1. Clone or copy to `ComfyUI/custom_nodes/BILD-VizionPipeline/`
2. Install dependencies: `pip install -r requirements.txt`
3. Restart ComfyUI

## Requirements

- PyTorch (included with ComfyUI)
- numpy >= 1.24
- Pillow >= 10.0
- scipy >= 1.11

## Structure

```
BILD-VizionPipeline/
├── __init__.py              # Registers all 19 nodes
├── requirements.txt
├── README.md
├── nodes/
│   ├── __init__.py
│   ├── lora_batch.py        # LoRA batch loading
│   ├── post_processing.py   # Motion blur, brightness, gaussian blur, vignette, saturation, unsharp, gamma, simple grain
│   ├── auto_white_balance.py
│   ├── lens_effects.py
│   ├── camera_simulator.py
│   ├── smart_hdr.py
│   ├── deep_fusion.py
│   ├── compression.py
│   ├── film_grain.py        # Advanced luminance-weighted grain
│   ├── metadata_inject.py
│   └── authenticate_all.py
└── utils/
    ├── __init__.py
    └── tensor_ops.py        # Shared tensor utilities
```

## Usage Tips

- **Subtle authenticity** (strength ~0.3–0.5): Light touch, preserves original quality
- **Standard iPhone look** (strength ~0.6): Recommended default for Authenticate All
- **Heavy processing** (strength ~0.8–1.0): Very obvious iPhone processing artifacts
- Use individual nodes for fine-grained control over each effect
- The Metadata Inject node outputs a JSON string — use with custom save nodes for EXIF writing

## Categories in ComfyUI Menu

- **BILD/VizionPipeline** — Post-processing nodes
- **BILD/VizionPipeline/LoRA** — LoRA batch tools
- **BILD/VizionPipeline/Authenticate** — iPhone authenticity simulation
