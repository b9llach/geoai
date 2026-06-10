"""Equirectangular → 4 perspective crops.

PIGEON renders 4 cardinal-heading views at 90° FOV from each Street View
panorama; SigLIP2-SO400M-patch14-384 wants 384×384 inputs. We do the same.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import py360convert
from PIL import Image

from geoai.config import CROP_FOV_DEG, CROP_HEADINGS_DEG, CROP_SIZE


def equirect_to_perspective_crops(
    equirect_path: Path,
    out_dir: Path,
    pano_id: str,
    *,
    headings: Iterable[int] = CROP_HEADINGS_DEG,
    fov_deg: float = CROP_FOV_DEG,
    size: int = CROP_SIZE,
    pitch_deg: float = 0.0,
    quality: int = 92,
) -> list[Path]:
    """Render perspective JPEGs and return their paths."""
    img = np.asarray(Image.open(equirect_path).convert("RGB"))
    out_paths: list[Path] = []
    for heading in headings:
        crop = py360convert.e2p(
            img,
            fov_deg=(fov_deg, fov_deg),
            u_deg=heading,
            v_deg=pitch_deg,
            out_hw=(size, size),
            in_rot_deg=0,
            mode="bilinear",
        )
        out_path = out_dir / f"{pano_id}_{heading:03d}.jpg"
        Image.fromarray(crop).save(out_path, quality=quality)
        out_paths.append(out_path)
    return out_paths


def equirect_to_panorama_strip(
    equirect_image: Image.Image,
    *,
    headings: Iterable[int] = CROP_HEADINGS_DEG,
    fov_deg: float = CROP_FOV_DEG,
    size: int = 768,
    pitch_deg: float = 0.0,
) -> Image.Image:
    """Render the 4 perspective views and concatenate horizontally — a clean
    "unwrapped" panoramic strip suitable for Stage 2 VLM analysis.

    Stage 1 uses 384x384 crops because that's SigLIP2's input. Stage 2's
    Qwen2.5-VL benefits from higher resolution to read small/distant signs,
    so we default to 768x768 per heading → a 3072x768 horizontal strip.
    The full 360° fits in one image so the VLM sees all sides at once
    without us picking the "right" heading.

    For Surya OCR, prefer `equirect_to_perspective_tiles()` instead —
    Surya downscales extreme-aspect-ratio inputs aggressively, so it gets
    better text detection from individual square crops than from one
    4:1 strip.
    """
    arr = np.asarray(equirect_image.convert("RGB"))
    views = []
    for heading in headings:
        crop = py360convert.e2p(
            arr,
            fov_deg=(fov_deg, fov_deg),
            u_deg=heading,
            v_deg=pitch_deg,
            out_hw=(size, size),
            in_rot_deg=0,
            mode="bilinear",
        )
        views.append(crop)
    concat = np.concatenate(views, axis=1)
    return Image.fromarray(concat)


# 8 headings = 4 cardinals (0, 90, 180, 270) + 4 diagonals (45, 135, 225, 315).
# The diagonals overlap the cardinals by 50% on each side, so any sign that
# straddles a cardinal boundary lands fully within at least one diagonal view.
_OCR_HEADINGS_8 = (0, 45, 90, 135, 180, 225, 270, 315)


def equirect_to_perspective_tiles(
    equirect_image: Image.Image,
    *,
    headings: Iterable[int] = _OCR_HEADINGS_8,
    fov_deg: float = CROP_FOV_DEG,
    size: int = 1024,
    pitch_deg: float = 0.0,
) -> list[Image.Image]:
    """Render N independent square perspective views, each at full size,
    for OCR consumption. Unlike the panoramic strip, each tile keeps a
    1:1 aspect ratio (what Surya was trained on) and full per-view
    resolution.

    With the default 8 headings + 90° FOV, every spot in the panorama
    appears in at least 2 tiles (each tile overlaps its neighbors by
    50%), so text that straddles a cardinal boundary in the strip is
    fully visible in at least one diagonal tile. Costs 2× the per-tile
    OCR latency vs the strip but typically catches ~20-40% more text on
    sign-heavy panos.
    """
    arr = np.asarray(equirect_image.convert("RGB"))
    tiles: list[Image.Image] = []
    for heading in headings:
        crop = py360convert.e2p(
            arr,
            fov_deg=(fov_deg, fov_deg),
            u_deg=heading,
            v_deg=pitch_deg,
            out_hw=(size, size),
            in_rot_deg=0,
            mode="bilinear",
        )
        tiles.append(Image.fromarray(crop))
    return tiles
