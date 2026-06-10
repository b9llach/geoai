"""Per-level S2 cell vocabulary, built from cells.parquet.

`CellVocab` maps between the canonical S2 cell ID (stored as a decimal
string in SQLite, see `geoai/scraper/db.py`) and a dense integer index
0..vocab_size(level)-1 used as a classifier output index.

Cells with fewer than `min_count[level]` training panos are *pruned* —
they don't get an output slot. A pano falling in a pruned cell at level L
returns the sentinel `PRUNED_LABEL = -1` for that level and the loss
function masks it out, but the same pano still contributes to the loss
at coarser levels where its cell *was* kept.

Defaults are tuned to our ~920k-pano corpus (see analysis in CLAUDE.md):
    L3: min=1   (168 cells, all kept — continental partitioning)
    L6: min=1   (3.7k cells, all kept — country/state partitioning)
    L9: min=2   (65k cells, 97.5% pano coverage — city-sized)
    L12: min=5  (26k cells, 35% pano coverage — dense urban areas only)

The L12 cut focuses the head's ~26M params on cells where sub-km accuracy
is achievable (dense data); rural panos rely on L9 + ProtoNet refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from geoai.config import PROCESSED_DIR

PRUNED_LABEL = -1
DEFAULT_LEVELS: tuple[int, ...] = (3, 6, 9, 12)
DEFAULT_MIN_COUNT: Mapping[int, int] = {3: 1, 6: 1, 9: 2, 12: 5}


@dataclass
class _LevelVocab:
    cell_to_index: dict[str, int]
    centroids: np.ndarray   # [N, 2] — (lat, lng), kept-cell order
    counts: np.ndarray      # [N]    — train-pano count per kept cell

    def __len__(self) -> int:
        return len(self.cell_to_index)


class CellVocab:
    def __init__(self, by_level: dict[int, _LevelVocab], levels: tuple[int, ...]):
        self.by_level = by_level
        self.levels = levels

    @classmethod
    def from_parquet(
        cls,
        path: Path = PROCESSED_DIR / "cells.parquet",
        levels: tuple[int, ...] = DEFAULT_LEVELS,
        min_count: Mapping[int, int] = DEFAULT_MIN_COUNT,
    ) -> "CellVocab":
        df = pd.read_parquet(path)
        by_level: dict[int, _LevelVocab] = {}
        for lvl in levels:
            sub = df[(df.level == lvl) & (df["count"] >= min_count.get(lvl, 1))]
            sub = sub.sort_values("cell_id_str").reset_index(drop=True)
            by_level[lvl] = _LevelVocab(
                cell_to_index={c: i for i, c in enumerate(sub.cell_id_str.tolist())},
                centroids=sub[["centroid_lat", "centroid_lng"]].to_numpy(np.float32),
                counts=sub["count"].to_numpy(np.int64),
            )
        return cls(by_level=by_level, levels=levels)

    def vocab_size(self, level: int) -> int:
        return len(self.by_level[level])

    def index(self, level: int, cell_id_str: str | None) -> int:
        """Return dense index for a cell, or PRUNED_LABEL if pruned/unknown."""
        if cell_id_str is None:
            return PRUNED_LABEL
        return self.by_level[level].cell_to_index.get(cell_id_str, PRUNED_LABEL)

    def centroids_tensor(self, level: int, device=None) -> torch.Tensor:
        return torch.from_numpy(self.by_level[level].centroids.copy()).to(device)

    def counts_tensor(self, level: int, device=None) -> torch.Tensor:
        return torch.from_numpy(self.by_level[level].counts.copy()).to(device)

    def cell_id_at(self, level: int, dense_idx: int) -> str:
        """Inverse of `index()` — get the S2 cell ID string for a dense index.
        Builds a reverse lookup table lazily on first call."""
        if not hasattr(self, "_idx_to_id"):
            self._idx_to_id: dict[int, list[str]] = {}
        if level not in self._idx_to_id:
            cell_to_idx = self.by_level[level].cell_to_index
            ordered: list[str] = [""] * len(cell_to_idx)
            for cid, idx in cell_to_idx.items():
                ordered[idx] = cid
            self._idx_to_id[level] = ordered
        return self._idx_to_id[level][dense_idx]

    def cell_polygon(self, level: int, dense_idx: int) -> list[list[float]]:
        """Return the 4 lat/lng corners of an S2 cell as [[lat, lng], ...].
        Used for rendering the cell as a polygon on a Leaflet/GeoJSON map.
        Order follows S2's get_vertex(0..3) — should form a closed quad."""
        import s2sphere
        cell_id_str = self.cell_id_at(level, dense_idx)
        cell = s2sphere.Cell(s2sphere.CellId(int(cell_id_str)))
        out: list[list[float]] = []
        for i in range(4):
            ll = s2sphere.LatLng.from_point(cell.get_vertex(i))
            out.append([ll.lat().degrees, ll.lng().degrees])
        return out

    def parent_map(self, child_level: int, parent_level: int) -> torch.Tensor:
        """Map each child-level cell to its dense parent-level index.

        Returns a LongTensor [vocab_size(child_level)] where entry i is the
        index in the parent-level vocab of cell i's S2 parent at
        `parent_level`, or -1 if that parent isn't in the parent vocab
        (pruned). Cached on the instance after first call.

        Used at inference to enforce hierarchical constraint: after picking
        the top-K cells at a coarser level, mask the finer-level logits to
        only the children of those K parents. Standard fix for the
        "northeast US pano → Ottawa" failure mode (small/dense foreign
        cells outscoring diffuse correct-region cells).
        """
        if not hasattr(self, "_parent_maps"):
            self._parent_maps: dict[tuple[int, int], torch.Tensor] = {}
        key = (child_level, parent_level)
        if key in self._parent_maps:
            return self._parent_maps[key]
        if parent_level >= child_level:
            raise ValueError(f"parent_level ({parent_level}) must be coarser "
                             f"than child_level ({child_level})")
        import s2sphere
        child_vocab = self.by_level[child_level]
        parent_to_idx = self.by_level[parent_level].cell_to_index
        result = np.full(len(child_vocab), -1, dtype=np.int64)
        for child_str, child_idx in child_vocab.cell_to_index.items():
            parent_id = s2sphere.CellId(int(child_str)).parent(parent_level).id()
            result[child_idx] = parent_to_idx.get(str(parent_id), -1)
        t = torch.from_numpy(result)
        self._parent_maps[key] = t
        return t

    def __repr__(self) -> str:
        sizes = ", ".join(f"L{lvl}={self.vocab_size(lvl)}" for lvl in self.levels)
        return f"CellVocab({sizes})"
