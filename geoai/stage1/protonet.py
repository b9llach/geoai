"""ProtoNet sub-cell refinement on L9 cells.

After the hierarchical model picks an L9 cell, we don't have to settle for
that cell's centroid. The cell already contains hundreds-to-thousands of
training panos with known (lat, lng) — we can use feature-space nearest
neighbors among them and return a feature-similarity-weighted (lat, lng)
inside the cell.

L9 cells average ~18 km on a side. The centroid alone gives ~9 km
sub-cell error in the worst case (and ~6 km in expectation). ProtoNet on
L9 typically delivers 1-3 km median error within-cell — a meaningful
upgrade for "right metro, where exactly" cases (correct L9 prediction,
which the model gets ~70% of the time after the L9-fallback rule).

Why L9 and not L12:
    L12 cells are ~2.3 km — already small. They also have only 5-15
    prototypes per cell after pruning. ProtoNet's variance is dominated by
    prototype sparsity at L12. L9 cells, with 30-200+ prototypes each, are
    where feature-NN is statistically reliable.

Index format (a single torch-saved dict):
    features:    [N_total, D] float32 (or float16)
    latlngs:     [N_total, 2] float32
    cell_starts: [vocab_size + 1] int64
    feature_dim: int (=proj_dim, e.g. 1024)
    level:       int (the level this index is built for, currently always 9)

Cell `i`'s prototypes live at `features[cell_starts[i] : cell_starts[i+1]]`.
Built once offline by `scripts/build_protonet_index.py`; loaded lazily at
inference time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class ProtoNetRefinement:
    """Result of a ProtoNet lookup against one L9 cell."""
    lat: float
    lng: float
    n_prototypes: int           # how many prototypes lived in this cell
    n_used: int                 # how many were used for the weighted average (top-K)
    top_similarity: float       # cosine sim to nearest prototype


class ProtoNetIndex:
    def __init__(
        self,
        features: torch.Tensor,         # [N_total, D]
        latlngs: torch.Tensor,          # [N_total, 2]
        cell_starts: torch.Tensor,      # [vocab_size + 1]
        level: int = 9,
    ):
        assert features.shape[0] == latlngs.shape[0]
        assert features.ndim == 2 and latlngs.shape[1] == 2
        assert cell_starts.ndim == 1
        self.features = features.contiguous()
        self.latlngs = latlngs.contiguous()
        self.cell_starts = cell_starts.contiguous()
        self.level = level
        self._normalised = None  # lazy-cached L2-normalised features

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_cells(self) -> int:
        # cell_starts has vocab_size + 1 entries
        return int(self.cell_starts.numel() - 1)

    def cell_size(self, cell_idx: int) -> int:
        return int(self.cell_starts[cell_idx + 1] - self.cell_starts[cell_idx])

    def _normalised_features(self) -> torch.Tensor:
        # Normalize in fp32 (precision) but STORE in fp16 — for 1.94M×1024 this
        # is ~4GB instead of ~8GB, the difference between fitting alongside the
        # model+raw-features on a 24GB GPU and OOM-ing a prediction. The tiny
        # per-cell slice is upcast back to fp32 in refine() for the matmul.
        if self._normalised is None:
            self._normalised = F.normalize(self.features.float(), dim=-1).half()
        return self._normalised

    def to(self, device: torch.device | str) -> "ProtoNetIndex":
        """Move all tensors to device (call once at startup; keep on GPU for fast lookups)."""
        self.features = self.features.to(device)
        self.latlngs = self.latlngs.to(device)
        self.cell_starts = self.cell_starts.to(device)
        if self._normalised is not None:
            self._normalised = self._normalised.to(device)
        return self

    @torch.no_grad()
    def refine(
        self,
        query_features: torch.Tensor,    # [D] or [1, D]
        cell_idx: int,
        k: int = 5,
        temperature: float = 0.1,
        min_prototypes: int = 3,
    ) -> ProtoNetRefinement | None:
        """Feature-NN refinement within the predicted cell.

        Returns None if the cell has fewer than `min_prototypes` panos, or
        if `cell_idx` is out of range — caller should fall back to centroid
        in that case.

        Algorithm:
            1. Pull the cell's prototype features (slice via cell_starts).
            2. Cosine-similarity query against each.
            3. Take top-K, softmax with temperature τ.
            4. Weighted average of (lat, lng) of the top-K.

        Smaller τ → more peaked; the top-1 NN dominates. Larger τ → more
        averaging across the K. τ=0.1 is a reasonable starting point;
        sweep on val.
        """
        if cell_idx < 0 or cell_idx >= self.num_cells:
            return None
        if query_features.ndim == 2:
            assert query_features.shape[0] == 1, "ProtoNet refine handles one query at a time"
            query_features = query_features.squeeze(0)

        start = int(self.cell_starts[cell_idx].item())
        end = int(self.cell_starts[cell_idx + 1].item())
        n_proto = end - start
        if n_proto < min_prototypes:
            return None

        # Move query to the index's device — caller might pass a CPU tensor
        # if the model produced features on CPU, or a different CUDA device.
        device = self.features.device
        q = query_features.to(device).float()

        # Cosine similarity = normalised dot product. The normalized index is
        # stored fp16 to save GPU memory; upcast the small per-cell slice to
        # fp32 to match the query for the matmul (negligible — n_proto ~16).
        proto = self._normalised_features()[start:end].float()       # [n_proto, D]
        q = F.normalize(q.unsqueeze(0), dim=-1).squeeze(0)           # [D]
        sims = proto @ q                                             # [n_proto]
        k_eff = min(k, n_proto)
        top = sims.topk(k_eff)
        top_idx = top.indices                                        # [k_eff]
        top_sims = top.values                                        # [k_eff]

        # Softmax weighting on the top-K similarities.
        weights = F.softmax(top_sims / temperature, dim=0)           # [k_eff]
        latlngs_topk = self.latlngs[start:end][top_idx]              # [k_eff, 2]
        weighted = (weights.unsqueeze(-1) * latlngs_topk).sum(dim=0) # [2]

        return ProtoNetRefinement(
            lat=float(weighted[0].item()),
            lng=float(weighted[1].item()),
            n_prototypes=n_proto,
            n_used=int(k_eff),
            top_similarity=float(top_sims[0].item()),
        )

    @classmethod
    def from_file(cls, path: Path | str, device: torch.device | str = "cpu") -> "ProtoNetIndex":
        blob = torch.load(str(path), map_location=device, weights_only=True)
        return cls(
            features=blob["features"],
            latlngs=blob["latlngs"],
            cell_starts=blob["cell_starts"],
            level=int(blob.get("level", 9)),
        )

    def to_file(self, path: Path | str) -> None:
        torch.save({
            "features": self.features.cpu(),
            "latlngs": self.latlngs.cpu(),
            "cell_starts": self.cell_starts.cpu(),
            "level": self.level,
            "feature_dim": self.feature_dim,
        }, str(path))

    def __repr__(self) -> str:
        n_total = int(self.features.shape[0])
        avg = n_total / max(self.num_cells, 1)
        return (
            f"ProtoNetIndex(level=L{self.level}, cells={self.num_cells:,}, "
            f"prototypes={n_total:,}, avg/cell={avg:.1f}, dim={self.feature_dim})"
        )
