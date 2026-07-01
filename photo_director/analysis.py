from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rawpy
from PIL import Image

from .schema import GlobalAdjustments


def _read_exif(path: Path) -> dict[str, str]:
    try:
        import exifread
    except ImportError:
        return {}

    with path.open("rb") as raw_file:
        tags = exifread.process_file(raw_file, details=False)

    wanted = {
        "Image Make": "camera_make",
        "Image Model": "camera_model",
        "EXIF LensModel": "lens",
        "EXIF ISOSpeedRatings": "iso",
        "EXIF FNumber": "aperture",
        "EXIF ExposureTime": "shutter_speed",
        "EXIF FocalLength": "focal_length",
        "EXIF DateTimeOriginal": "captured_at",
    }
    return {out_key: str(tags[tag_key]) for tag_key, out_key in wanted.items() if tag_key in tags}


def _render_preview_array(path: Path, max_size: int) -> np.ndarray:
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_bps=8,
            no_auto_bright=True,
            gamma=(2.222, 4.5),
        )
    image = Image.fromarray(rgb)
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return np.asarray(image)


def _histogram(rgb: np.ndarray) -> dict[str, Any]:
    channels = {}
    for index, name in enumerate(("red", "green", "blue")):
        counts, bins = np.histogram(rgb[:, :, index], bins=32, range=(0, 255))
        channels[name] = {
            "bins": [round(float(value), 3) for value in bins[:-1].tolist()],
            "counts": counts.astype(int).tolist(),
        }

    luminance = (
        0.2126 * rgb[:, :, 0].astype(np.float32)
        + 0.7152 * rgb[:, :, 1].astype(np.float32)
        + 0.0722 * rgb[:, :, 2].astype(np.float32)
    )
    lum_counts, lum_bins = np.histogram(luminance, bins=32, range=(0, 255))
    return {
        "channels": channels,
        "luminance": {
            "bins": [round(float(value), 3) for value in lum_bins[:-1].tolist()],
            "counts": lum_counts.astype(int).tolist(),
        },
        "shadow_clip_percent": round(float(np.mean(luminance <= 2) * 100), 3),
        "highlight_clip_percent": round(float(np.mean(luminance >= 253) * 100), 3),
        "mean_luminance": round(float(np.mean(luminance)), 3),
        "median_luminance": round(float(np.median(luminance)), 3),
    }


def _dominant_palette(rgb: np.ndarray, sample_size: int = 20000, colors: int = 6) -> list[dict[str, Any]]:
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    if len(pixels) > sample_size:
        indices = np.linspace(0, len(pixels) - 1, sample_size, dtype=np.int64)
        pixels = pixels[indices]

    rng = np.random.default_rng(13)
    centers = pixels[rng.choice(len(pixels), size=min(colors, len(pixels)), replace=False)]
    for _ in range(10):
        distances = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = []
        for color_index in range(len(centers)):
            members = pixels[labels == color_index]
            new_centers.append(members.mean(axis=0) if len(members) else centers[color_index])
        centers = np.asarray(new_centers)

    distances = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    palette = []
    for color_index, center in enumerate(centers):
        share = float(np.mean(labels == color_index))
        rgb_values = [int(round(value)) for value in center.clip(0, 255).tolist()]
        palette.append(
            {
                "hex": "#{:02x}{:02x}{:02x}".format(*rgb_values),
                "rgb": rgb_values,
                "percent": round(share * 100, 2),
            }
        )
    return sorted(palette, key=lambda item: item["percent"], reverse=True)


def _baseline_settings(histogram: dict[str, Any]) -> dict[str, float]:
    mean_luminance = float(histogram.get("mean_luminance", 128.0))
    highlight_clip = float(histogram.get("highlight_clip_percent", 0.0))
    shadow_clip = float(histogram.get("shadow_clip_percent", 0.0))

    exposure = max(-0.75, min(1.25, (128.0 - mean_luminance) / 90.0))
    shadows = 12.0 if mean_luminance < 105.0 or shadow_clip > 0.5 else 0.0
    highlights = -12.0 if highlight_clip > 0.1 else -6.0
    blacks = -4.0 if shadow_clip < 0.5 else 2.0

    return GlobalAdjustments(
        exposure=round(exposure, 2),
        contrast=6.0,
        highlights=highlights,
        shadows=shadows,
        whites=4.0,
        blacks=blacks,
        vibrance=6.0,
        clarity=4.0,
    ).to_dict()


def analyze_raw(raw_path: Path, preview_path: Path, preview_size: int = 1400) -> dict[str, Any]:
    raw_path = raw_path.expanduser().resolve()
    preview_path = preview_path.expanduser().resolve()
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = _render_preview_array(raw_path, preview_size)
    Image.fromarray(rgb).save(preview_path, quality=92, optimize=True)

    with rawpy.imread(str(raw_path)) as raw:
        raw_shape = list(raw.raw_image_visible.shape)
        color_desc = raw.color_desc.decode("ascii", errors="ignore")
        black_levels = [int(value) for value in raw.black_level_per_channel]
        white_level = int(raw.white_level)
        camera_white_balance = [float(value) for value in raw.camera_whitebalance]
        daylight_white_balance = [float(value) for value in raw.daylight_whitebalance]

    histogram = _histogram(rgb)
    return {
        "source": str(raw_path),
        "preview": str(preview_path),
        "baseline_settings": _baseline_settings(histogram),
        "baseline_strategy": (
            "Deterministic normalized RAW baseline. The LLM should return deltas from these settings, "
            "not absolute final slider values."
        ),
        "raw_specs": {
            "raw_shape": raw_shape,
            "color_desc": color_desc,
            "black_levels": black_levels,
            "white_level": white_level,
            "camera_white_balance": camera_white_balance,
            "daylight_white_balance": daylight_white_balance,
        },
        "exif": _read_exif(raw_path),
        "histogram": histogram,
        "dominant_palette": _dominant_palette(rgb),
    }


def write_analysis(analysis: dict[str, Any], output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
