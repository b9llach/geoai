"""PyTorch Dataset over the cataloged + cropped pano corpus.

One sample = one pano = a 4-view stack (heading 0/90/180/270) at 384×384,
with S2 cell labels at four levels and the true (lat, lng).

Augmentation policy (training only):
    * Random *cyclic shift* of the 4 views — gives heading-invariance for
      free without breaking the per-view structure that concat pooling
      relies on. ([0,90,180,270] becomes [90,180,270,0] etc.)
    * Light photometric jitter — brightness, contrast, saturation ±10%.
    * NO horizontal flip (mirrors right-side traffic to left, breaks
      country signal). NO random crop (FOV is geometric, not aesthetic).

Image normalization is SigLIP2's mean/std = 0.5/0.5, NOT ImageNet's.
"""
from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from geoai.config import CROP_HEADINGS_DEG, CROP_SIZE, METADATA_DB
from geoai.stage1.cells import PRUNED_LABEL, CellVocab
from geoai.stage1.country_vocab import CountryVocab

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


class PanoDataset(Dataset):
    """One row per training pano. Constructed from SQLite catalog."""

    def __init__(
        self,
        cells: CellVocab,
        split: str = "train",
        db_path: Path = METADATA_DB,
        crop_size: int = CROP_SIZE,
        augment: bool | None = None,
        country_filter: str | None = None,
        countries: CountryVocab | None = None,
    ):
        if augment is None:
            augment = (split == "train")
        self.cells = cells
        self.countries = countries
        self.augment = augment
        self.headings = CROP_HEADINGS_DEG

        # Pull only what training needs. Filter to panos that have ALL 4 crops
        # so __getitem__ never has to handle missing files.
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        q = f"""
            SELECT p.pano_id, p.lat, p.lng,
                   p.s2_l3, p.s2_l6, p.s2_l9, p.s2_l12,
                   p.country_code
            FROM panos p
            WHERE p.split = ?
              AND p.equirect_path IS NOT NULL
              AND (SELECT COUNT(*) FROM crops c WHERE c.pano_id = p.pano_id) = 4
        """
        params: list = [split]
        if country_filter:
            q += " AND p.country_code = ?"
            params.append(country_filter)
        self.rows = list(conn.execute(q, params))
        conn.close()

        # Resolve crop paths once; SQLite lookups in __getitem__ are too slow.
        from geoai.config import CROPS_DIR
        self._crops_root = CROPS_DIR

        self._color_jitter = (
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            if augment else None
        )
        self._tx = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def _crop_path(self, pano_id: str, heading: int) -> Path:
        return self._crops_root / pano_id[:2] / f"{pano_id}_{heading:03d}.jpg"

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        pano_id = row["pano_id"]

        headings = list(self.headings)
        if self.augment:
            shift = random.randint(0, len(headings) - 1)
            headings = headings[shift:] + headings[:shift]

        views: list[torch.Tensor] = []
        for h in headings:
            img = Image.open(self._crop_path(pano_id, h)).convert("RGB")
            if self._color_jitter is not None:
                img = self._color_jitter(img)
            views.append(self._tx(img))
        pixel_values = torch.stack(views, dim=0)  # [4, 3, H, W]

        cell_indices = {
            lvl: self.cells.index(lvl, row[f"s2_l{lvl}"])
            for lvl in self.cells.levels
        }
        country_code = row["country_code"] or ""
        country_idx = (
            self.countries.index(country_code) if self.countries is not None
            else PRUNED_LABEL
        )

        return {
            "pano_id": pano_id,
            "pixel_values": pixel_values,
            "cell_indices": cell_indices,
            "latlng": torch.tensor([row["lat"], row["lng"]], dtype=torch.float32),
            "country_code": country_code,
            "country_idx": country_idx,
        }


def collate(batch: list[dict]) -> dict:
    """Stack tensors, keep per-level cell labels as long tensors with -1 sentinels."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    latlng = torch.stack([b["latlng"] for b in batch], dim=0)
    levels = batch[0]["cell_indices"].keys()
    cell_indices = {
        lvl: torch.tensor([b["cell_indices"][lvl] for b in batch], dtype=torch.long)
        for lvl in levels
    }
    country_idx = torch.tensor(
        [b["country_idx"] for b in batch], dtype=torch.long
    )
    return {
        "pano_ids": [b["pano_id"] for b in batch],
        "pixel_values": pixel_values,
        "cell_indices": cell_indices,
        "latlng": latlng,
        "country_codes": [b["country_code"] for b in batch],
        "country_idx": country_idx,
    }
