"""Filesystem and resource paths.

The training corpus and reference rasters live on a separate volume
(`/data/geolocation`) — way too big to vendor into the repo. This module
centralizes the paths so production code never hard-codes them and tests
can override via env vars.

Environment variables (all optional, sane defaults):
    GEOAI_DATA_ROOT        — root of the imported corpus (default /data/geolocation)
    GEOAI_PROCESSED_DIR    — where catalog DB + rendered crops live
                              (default <DATA_ROOT>/processed)
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("GEOAI_DATA_ROOT", "/data/geolocation"))

# Equirectangular pano sources. All three directories share the same naming
# (`{pano_id}.jpg`) and we treat them as one logical pool, with `panos/`
# preferred when the same id appears in multiple (~720 known dupes).
# `panos_supplement/images/` is where new scrapes (locs9+) land.
PANOS_DIRS: tuple[Path, ...] = (
    DATA_ROOT / "panos",
    DATA_ROOT / "panos_new",
    DATA_ROOT / "panos_supplement" / "images",
)

# Master metadata (pano_id, lat, lng) — 1.04M rows, ~62 MB.
PANO_LOG_CSV = DATA_ROOT / "panorama_log_new.csv"

# GADM 4.1 admin-0/1/2 polygons in a single GeoPackage (4.7 GB).
GADM_PATH = DATA_ROOT / "gadm_410-levels.gpkg"

# Köppen-Geiger present-climate classification, 0.0083° (~1 km).
# The top-level `_conf_` file is per-pixel confidence (0-100%), NOT the class
# label — use the no-suffix variant under `custom/` for actual class IDs.
KOPPEN_PATH = DATA_ROOT / "custom" / "Beck_KG_V1_present_0p0083.tif"

# GHS Population 2020, Mollweide 1 km grid. The top-level copy is mode 700
# (root-only); the duplicate under `custom/` is mode 644 and readable to us.
GHS_POP_PATH = DATA_ROOT / "custom" / "GHS_POP_E2020_GLOBE_R2022A_54009_1000_V1_0.tif"

# Outputs of Phase 2 — kept on the same volume as the source equirects so
# we don't pay copy cost.
PROCESSED_DIR = Path(
    os.environ.get("GEOAI_PROCESSED_DIR", str(DATA_ROOT / "processed"))
)
METADATA_DB = PROCESSED_DIR / "metadata.db"
CROPS_DIR = PROCESSED_DIR / "crops"

# Stage 1 input geometry. Four perspective views per pano matches PIGEON
# and GeoGuessr's default player FOV; SigLIP2-SO400M-patch14-384 wants
# 384x384 inputs.
CROP_SIZE = 384
CROP_FOV_DEG = 90
CROP_HEADINGS_DEG: tuple[int, ...] = (0, 90, 180, 270)


# --- Stage 2 (verification VLM + OCR + langid + translation) ---
# Surya OCR auto-downloads its own weights to ~/.cache/datalab/.
# fasttext + NLLB live under GEOAI_STAGE2_MODELS_DIR (defaults to
# <DATA_ROOT>/models) so they share the data volume with the corpus.
STAGE2_MODELS_DIR = Path(
    os.environ.get("GEOAI_STAGE2_MODELS_DIR", str(DATA_ROOT / "models"))
)
FASTTEXT_LID_PATH = STAGE2_MODELS_DIR / "fasttext" / "lid.176.bin"
NLLB_CT2_DIR = STAGE2_MODELS_DIR / "nllb-200-distilled-1.3B-ct2"
NLLB_HF_DIR = STAGE2_MODELS_DIR / "nllb-200-distilled-1.3B"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
QWEN_VL_MODEL = os.environ.get("GEOAI_QWEN_VL_MODEL", "qwen3.5:9b")

# Stage-2 reasoning VLM. The hybrid pinpointer (Surya OCR → text → VLM reasons)
# talks to an OpenAI-compatible endpoint (LM Studio) serving Gemma 4 26B A4B —
# the MoE/vision/tool-use model Stage 2 was originally specced around. It runs
# on a SEPARATE box from the training GPUs, so it never contends for VRAM.
# Gemma's thinking mode spirals when asked to read illegible pixels directly,
# but stays bounded (~8 s) when fed pre-extracted OCR text under the tight
# pinpoint system prompt — hence the hybrid.
LMSTUDIO_HOST = os.environ.get("GEOAI_LMSTUDIO_HOST", "http://localhost:1234")
GEMMA_VL_MODEL = os.environ.get("GEOAI_GEMMA_VL_MODEL", "gemma-4-26b-a4b-it-qat")


def find_equirect(pano_id: str) -> Path | None:
    """Return the on-disk path for a pano_id, or None if not found.

    Prefers `panos/` over `panos_new/` when the same id exists in both.
    """
    for d in PANOS_DIRS:
        p = d / f"{pano_id}.jpg"
        if p.exists():
            return p
    return None
