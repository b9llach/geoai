"""Distribution-based country prior + per-level cell masking.

V1 has no auxiliary country head (that's a V2 retrain change). But we can
still derive a country prior from one of the existing classifier heads:

    P(country | image) = Σ_cell P(country | cell) · P(cell | image)

where `P(cell | image)` is the head's softmax output and
`P(country | cell)` comes from the per-cell country distribution among
training panos. Aggregating with a *distribution* (not just modal country)
matters because L3 cells in particular span multiple countries — a
modal-country aggregation incorrectly funnels probability into whichever
country happens to be the majority in each L3 cell, biasing toward
geographically-large countries (Russia, USA) and away from small
clustered ones (Germany, Czechia).

We default to **L6** as the source level — L6 cells are ~144 km, usually
inside one country, so their distributions are sharp and the resulting
country probs match intuition. L3 (~1150 km) is too coarse and was the
source of the bug in the first prototype. L9 (~18 km) would be sharper
still but its softmax is harder to read for a country signal.

Apply: at inference, take the source level's logits → softmax →
country distribution. Pick top-K countries. Mask L9 / L12 logits whose
cell distribution puts most of its mass outside the top-K.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


class CountryPrior:
    def __init__(
        self,
        countries: Sequence[str],
        per_level_dist: dict[int, torch.Tensor],   # {lvl: [V_lvl, num_countries]}
        source_level: int = 6,
    ):
        self.codes = list(countries)
        self._cc_to_idx = {c: i for i, c in enumerate(countries)}
        self.per_level_dist = {lvl: m.contiguous() for lvl, m in per_level_dist.items()}
        if source_level not in self.per_level_dist:
            raise ValueError(
                f"source_level={source_level} not in available distributions "
                f"{sorted(self.per_level_dist)}"
            )
        self.source_level = source_level

    @classmethod
    def from_files(
        cls,
        dir: Path | str,
        source_level: int = 6,
        levels: tuple[int, ...] = (3, 6, 9, 12),
    ) -> "CountryPrior":
        d = Path(dir)
        per_level: dict[int, torch.Tensor] = {}
        countries: list[str] = []
        for lvl in levels:
            p = d / f"l{lvl}_country_dist.pt"
            if not p.exists():
                continue
            blob = torch.load(str(p), map_location="cpu", weights_only=True)
            per_level[lvl] = blob["matrix"]
            if not countries:
                countries = blob["countries"]
        if not per_level:
            raise FileNotFoundError(f"no l*_country_dist.pt files in {d}")
        return cls(countries, per_level, source_level=source_level)

    def to(self, device: torch.device | str) -> "CountryPrior":
        self.per_level_dist = {
            lvl: m.to(device) for lvl, m in self.per_level_dist.items()
        }
        return self

    @property
    def num_countries(self) -> int:
        return len(self.codes)

    @torch.no_grad()
    def country_probs_from(
        self,
        logits: torch.Tensor,    # [B, V_lvl]
        level: int,
    ) -> torch.Tensor:
        """Returns [B, num_countries] — Σ_cell softmax(logits)[c] · P(country | cell c)."""
        if level not in self.per_level_dist:
            raise ValueError(f"no distribution for level {level}")
        cell_probs = logits.float().softmax(dim=-1)                            # [B, V] in fp32
        return cell_probs @ self.per_level_dist[level].to(cell_probs.device)   # [B, K]

    @torch.no_grad()
    def top_k(
        self,
        logits: torch.Tensor,
        k: int = 3,
        level: int | None = None,
    ) -> tuple[torch.Tensor, list[list[str]]]:
        """Returns (indices [B, k], codes [list of lists of strings])."""
        lvl = level if level is not None else self.source_level
        cprobs = self.country_probs_from(logits, lvl)        # [B, K]
        top = cprobs.topk(k=min(k, self.num_countries), dim=-1)
        codes = [[self.codes[int(i)] for i in row] for row in top.indices.cpu().tolist()]
        return top.indices, codes

    @torch.no_grad()
    def mask_logits(
        self,
        logits: torch.Tensor,           # [B, V_target_lvl]
        target_level: int,
        top_indices: torch.Tensor,      # [B, k] long
        keep_threshold: float = 0.30,
    ) -> torch.Tensor:
        """Mask cells in `target_level`'s logits when their country
        distribution puts < `keep_threshold` of mass on any of the top-K
        countries. Cells with no training-pano data in any country (zero
        row in the distribution matrix) are NEVER masked.

        keep_threshold trades off precision vs safety:
          - 0.50 → only keep cells where ≥50% of training panos in the cell
                   are from a top-K country (strict)
          - 0.30 → looser; keeps a cell if any single top-K country has
                   ≥30% of its mass (recommended default)
          - 0.0  → keeps any cell that has any mass on a top-K country
        """
        if target_level not in self.per_level_dist:
            return logits
        dist = self.per_level_dist[target_level].to(logits.device)  # [V, K]
        no_data_mask = dist.sum(dim=-1) == 0                        # [V]
        masked = logits.clone()
        for b in range(logits.shape[0]):
            top_set = top_indices[b].to(logits.device)              # [k]
            # For each cell, the maximum fraction over the top-K countries.
            mass_in_top = dist[:, top_set].max(dim=-1).values       # [V]
            keep = (mass_in_top >= keep_threshold) | no_data_mask
            masked[b, ~keep] = float("-inf")
        return masked

    def __repr__(self) -> str:
        sizes = ", ".join(f"L{lvl}={m.shape[0]}" for lvl, m in self.per_level_dist.items())
        return (
            f"CountryPrior(source=L{self.source_level}, levels=[{sizes}], "
            f"countries={self.num_countries})"
        )


# Backward-compatible alias for any code that imported the old name.
L3CountryPrior = CountryPrior
