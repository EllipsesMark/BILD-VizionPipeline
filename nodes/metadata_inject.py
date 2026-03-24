"""
BILDMetadataInject — iPhone EXIF metadata injection.

Passes through the image unchanged but outputs a JSON string with realistic
iPhone EXIF metadata that downstream save nodes can use.
"""

from __future__ import annotations
import random
from datetime import datetime, timedelta
import torch


# iPhone camera specifications per model
CAMERA_SPECS = {
    "iPhone 15 Pro Max": {
        "make": "Apple",
        "model": "iPhone 15 Pro Max",
        "software": "17.5.1",
        "Main 1x": {
            "lens_make": "Apple",
            "lens_model": "iPhone 15 Pro Max back triple camera 6.765mm f/1.78",
            "focal_length": 6.765,
            "focal_length_35mm": 24,
            "f_number": 1.78,
            "iso_range": (50, 2500),
            "shutter_range": (1/8000, 1/4),
        },
        "Ultra-Wide 0.5x": {
            "lens_make": "Apple",
            "lens_model": "iPhone 15 Pro Max back triple camera 2.22mm f/2.2",
            "focal_length": 2.22,
            "focal_length_35mm": 13,
            "f_number": 2.2,
            "iso_range": (50, 3200),
            "shutter_range": (1/8000, 1/4),
        },
        "Telephoto 5x": {
            "lens_make": "Apple",
            "lens_model": "iPhone 15 Pro Max back triple camera 15.94mm f/2.8",
            "focal_length": 15.94,
            "focal_length_35mm": 120,
            "f_number": 2.8,
            "iso_range": (50, 3200),
            "shutter_range": (1/4000, 1/4),
        },
        "Selfie": {
            "lens_make": "Apple",
            "lens_model": "iPhone 15 Pro Max front TrueDepth camera 2.69mm f/1.9",
            "focal_length": 2.69,
            "focal_length_35mm": 23,
            "f_number": 1.9,
            "iso_range": (50, 2000),
            "shutter_range": (1/4000, 1/4),
        },
    },
    "iPhone 16 Pro Max": {
        "make": "Apple",
        "model": "iPhone 16 Pro Max",
        "software": "18.3.2",
        "Main 1x": {
            "lens_make": "Apple",
            "lens_model": "iPhone 16 Pro Max back quad camera 6.765mm f/1.78",
            "focal_length": 6.765,
            "focal_length_35mm": 24,
            "f_number": 1.78,
            "iso_range": (50, 3200),
            "shutter_range": (1/8000, 1/4),
        },
        "Ultra-Wide 0.5x": {
            "lens_make": "Apple",
            "lens_model": "iPhone 16 Pro Max back quad camera 2.22mm f/2.2",
            "focal_length": 2.22,
            "focal_length_35mm": 13,
            "f_number": 2.2,
            "iso_range": (50, 3200),
            "shutter_range": (1/8000, 1/4),
        },
        "Telephoto 5x": {
            "lens_make": "Apple",
            "lens_model": "iPhone 16 Pro Max back quad camera 15.94mm f/2.8",
            "focal_length": 15.94,
            "focal_length_35mm": 120,
            "f_number": 2.8,
            "iso_range": (50, 3200),
            "shutter_range": (1/4000, 1/4),
        },
        "Selfie": {
            "lens_make": "Apple",
            "lens_model": "iPhone 16 Pro Max front TrueDepth camera 2.69mm f/1.9",
            "focal_length": 2.69,
            "focal_length_35mm": 23,
            "f_number": 1.9,
            "iso_range": (50, 2000),
            "shutter_range": (1/4000, 1/4),
        },
    },
    "iPhone 17 Pro Max": {
        "make": "Apple",
        "model": "iPhone 17 Pro Max",
        "software": "19.1",
        "Main 1x": {
            "lens_make": "Apple",
            "lens_model": "iPhone 17 Pro Max back triple camera 6.765mm f/1.78",
            "focal_length": 6.765,
            "focal_length_35mm": 24,
            "f_number": 1.78,
            "iso_range": (25, 3200),
            "shutter_range": (1/8000, 1/4),
        },
        "Ultra-Wide 0.5x": {
            "lens_make": "Apple",
            "lens_model": "iPhone 17 Pro Max back triple camera 2.22mm f/2.2",
            "focal_length": 2.22,
            "focal_length_35mm": 13,
            "f_number": 2.2,
            "iso_range": (25, 3200),
            "shutter_range": (1/8000, 1/4),
        },
        "Telephoto 5x": {
            "lens_make": "Apple",
            "lens_model": "iPhone 17 Pro Max back triple camera 15.94mm f/2.8",
            "focal_length": 15.94,
            "focal_length_35mm": 120,
            "f_number": 2.8,
            "iso_range": (25, 3200),
            "shutter_range": (1/4000, 1/4),
        },
        "Selfie": {
            "lens_make": "Apple",
            "lens_model": "iPhone 17 Pro Max front TrueDepth camera 2.69mm f/1.9",
            "focal_length": 2.69,
            "focal_length_35mm": 23,
            "f_number": 1.9,
            "iso_range": (25, 2500),
            "shutter_range": (1/4000, 1/4),
        },
    },
}

# Major city GPS coordinates for realistic location data
GPS_CITIES = [
    # US Major
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
    {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    {"name": "Austin", "lat": 30.2672, "lon": -97.7431},
    {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
    {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
    {"name": "Nashville", "lat": 36.1627, "lon": -86.7816},
    {"name": "Portland", "lat": 45.5152, "lon": -122.6784},
    {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740},
    {"name": "Las Vegas", "lat": 36.1699, "lon": -115.1398},
    {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    {"name": "Atlanta", "lat": 33.7490, "lon": -84.3880},
    {"name": "Boston", "lat": 42.3601, "lon": -71.0589},
    {"name": "San Diego", "lat": 32.7157, "lon": -117.1611},
    {"name": "New Orleans", "lat": 29.9511, "lon": -90.0715},
    {"name": "Detroit", "lat": 42.3314, "lon": -83.0458},
    {"name": "Minneapolis", "lat": 44.9778, "lon": -93.2650},
    {"name": "Philadelphia", "lat": 39.9526, "lon": -75.1652},
    {"name": "Tampa", "lat": 27.9506, "lon": -82.4572},
    {"name": "Orlando", "lat": 28.5383, "lon": -81.3792},
    {"name": "Salt Lake City", "lat": 40.7608, "lon": -111.8910},
    {"name": "Honolulu", "lat": 21.3069, "lon": -157.8583},
    # Europe
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
    {"name": "Barcelona", "lat": 41.3874, "lon": 2.1686},
    {"name": "Berlin", "lat": 52.5200, "lon": 13.4050},
    {"name": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
    {"name": "Rome", "lat": 41.9028, "lon": 12.4964},
    {"name": "Prague", "lat": 50.0755, "lon": 14.4378},
    {"name": "Stockholm", "lat": 59.3293, "lon": 18.0686},
    {"name": "Lisbon", "lat": 38.7223, "lon": -9.1393},
    {"name": "Dublin", "lat": 53.3498, "lon": -6.2603},
    {"name": "Copenhagen", "lat": 55.6761, "lon": 12.5683},
    {"name": "Vienna", "lat": 48.2082, "lon": 16.3738},
    {"name": "Milan", "lat": 45.4642, "lon": 9.1900},
    # Asia / Pacific
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"name": "Seoul", "lat": 37.5665, "lon": 126.9780},
    {"name": "Bangkok", "lat": 13.7563, "lon": 100.5018},
    {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
    {"name": "Melbourne", "lat": -37.8136, "lon": 144.9631},
    {"name": "Bali", "lat": -8.3405, "lon": 115.0920},
    # Middle East / Latin America
    {"name": "Dubai", "lat": 25.2048, "lon": 55.2708},
    {"name": "Tel Aviv", "lat": 32.0853, "lon": 34.7818},
    {"name": "Mexico City", "lat": 19.4326, "lon": -99.1332},
    {"name": "Cancun", "lat": 21.1619, "lon": -86.8515},
    {"name": "Rio de Janeiro", "lat": -22.9068, "lon": -43.1729},
    {"name": "Buenos Aires", "lat": -34.6037, "lon": -58.3816},
    {"name": "Medellin", "lat": 6.2476, "lon": -75.5658},
]


class BILDMetadataInject:
    """Injects iPhone EXIF metadata. Image passes through unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "camera_model": (
                    ["iPhone 15 Pro Max", "iPhone 16 Pro Max", "iPhone 17 Pro Max"],
                    {"default": "iPhone 16 Pro Max"},
                ),
                "camera_mode": (
                    ["Main 1x", "Ultra-Wide 0.5x", "Telephoto 5x", "Selfie"],
                    {"default": "Main 1x"},
                ),
                "include_gps": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Include randomized GPS coordinates from selected city.",
                    },
                ),
                "gps_city": (
                    ["Random"] + [c["name"] for c in GPS_CITIES],
                    {"default": "Random"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "metadata_json",)
    FUNCTION = "apply"
    CATEGORY = "BILD/VizionPipeline/Authenticate"
    DESCRIPTION = (
        "Injects realistic iPhone EXIF metadata. The image passes through unchanged. "
        "Outputs a JSON string with metadata that custom save nodes can use."
    )

    def apply(self, image: torch.Tensor, camera_model: str, camera_mode: str, include_gps: bool, gps_city: str = "Random"):
        import json

        specs = CAMERA_SPECS[camera_model]
        mode_specs = specs[camera_mode]

        iso_min, iso_max = mode_specs["iso_range"]
        mean_brightness = image.mean().item()

        if mean_brightness > 0.5:
            iso = random.randint(iso_min, min(iso_min * 4, iso_max))
            shutter_speed = random.uniform(1/4000, 1/500)
        elif mean_brightness > 0.25:
            iso = random.randint(iso_min * 2, min(iso_min * 8, iso_max))
            shutter_speed = random.uniform(1/500, 1/60)
        else:
            iso = random.randint(iso_max // 4, iso_max)
            shutter_speed = random.uniform(1/60, 1/15)

        now = datetime.now()
        offset = timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        capture_time = now - offset

        metadata = {
            "Make": specs["make"],
            "Model": specs["model"],
            "Software": specs["software"],
            "LensMake": mode_specs["lens_make"],
            "LensModel": mode_specs["lens_model"],
            "FocalLength": mode_specs["focal_length"],
            "FocalLengthIn35mmFormat": mode_specs["focal_length_35mm"],
            "FNumber": mode_specs["f_number"],
            "ISO": iso,
            "ExposureTime": round(shutter_speed, 6),
            "ExposureProgram": 2,
            "MeteringMode": 5,
            "Flash": 16,
            "WhiteBalance": 0,
            "DateTimeOriginal": capture_time.strftime("%Y:%m:%d %H:%M:%S"),
            "CreateDate": capture_time.strftime("%Y:%m:%d %H:%M:%S"),
            "OffsetTimeOriginal": "-07:00",
            "ColorSpace": 65535,
            "ExifImageWidth": image.shape[2],
            "ExifImageHeight": image.shape[1],
            "Orientation": 1,
            "XResolution": 72,
            "YResolution": 72,
            "ResolutionUnit": 2,
            "SceneType": 1,
            "SceneCaptureType": 0,
            "SensingMethod": 2,
            "ExposureMode": 0,
            "DigitalZoomRatio": 1.0,
            "BrightnessValue": round(mean_brightness * 12 - 1, 2),
        }

        if include_gps:
            if gps_city == "Random":
                city = random.choice(GPS_CITIES)
            else:
                city = next((c for c in GPS_CITIES if c["name"] == gps_city), random.choice(GPS_CITIES))
            lat = city["lat"] + random.gauss(0, 0.009)
            lon = city["lon"] + random.gauss(0, 0.009)

            metadata["GPSLatitude"] = abs(lat)
            metadata["GPSLatitudeRef"] = "N" if lat >= 0 else "S"
            metadata["GPSLongitude"] = abs(lon)
            metadata["GPSLongitudeRef"] = "E" if lon >= 0 else "W"
            metadata["GPSAltitude"] = round(random.uniform(0, 500), 1)
            metadata["GPSAltitudeRef"] = 0
            metadata["GPSSpeed"] = 0.0
            metadata["GPSSpeedRef"] = "K"

        metadata_json = json.dumps(metadata, indent=2)

        return (image, metadata_json,)
