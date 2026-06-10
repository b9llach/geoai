"""Hierarchical haversine-smoothed cross-entropy.

For each S2 level, the target distribution over cells is a Gaussian over
great-circle distance from the true (lat, lng): cells geographically near
the truth get partial credit, far cells get ~0. This is dramatically more
sample-efficient than flat one-hot CE because nearly-right answers train
the model toward the right region rather than away from it.

Sigma per level (PLAN.md §"Phase 4 — Loss"): {3: 2000, 6: 500, 9: 100, 12: 20} km.
The ratio σ:cell-side stays in the 1.5–8× band across levels, so each level
keeps a similar "neighborhood softness."

Pruned-cell samples (label = -1) are masked out at *that* level only. They
still contribute at coarser levels where their cell *was* kept.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from geoai.stage1.cells import PRUNED_LABEL

DEFAULT_SIGMAS_KM: Mapping[int, float] = {3: 2000.0, 6: 500.0, 9: 100.0, 12: 20.0}
_EARTH_R_KM = 6371.0


def _haversine_km(
    lat1: torch.Tensor, lng1: torch.Tensor,
    lat2: torch.Tensor, lng2: torch.Tensor,
) -> torch.Tensor:
    """Broadcast haversine. Inputs in degrees, output in km."""
    deg2rad = math.pi / 180.0
    lat1, lat2 = lat1 * deg2rad, lat2 * deg2rad
    dlat = lat2 - lat1
    dlng = (lng2 - lng1) * deg2rad
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng / 2) ** 2
    return 2 * _EARTH_R_KM * torch.asin(torch.clamp(torch.sqrt(a), max=1.0))


def haversine_smooth_targets(
    true_latlng: torch.Tensor,    # [B, 2]   degrees
    cell_centroids: torch.Tensor, # [N, 2]   degrees
    sigma_km: float,
) -> torch.Tensor:
    """[B, N] soft-label distribution. Each row sums to 1."""
    true_lat = true_latlng[:, 0:1]      # [B, 1]
    true_lng = true_latlng[:, 1:2]      # [B, 1]
    cell_lat = cell_centroids[:, 0]     # [N]
    cell_lng = cell_centroids[:, 1]     # [N]
    d_km = _haversine_km(true_lat, true_lng, cell_lat[None, :], cell_lng[None, :])  # [B, N]
    # softmax of -d²/(2σ²) is the Gaussian normalised over cells
    log_w = -(d_km ** 2) / (2.0 * sigma_km ** 2)
    return F.softmax(log_w, dim=-1)


def hierarchical_loss(
    logits_per_level: dict[int, torch.Tensor],   # {lvl: [B, N_lvl]}
    true_latlng: torch.Tensor,                   # [B, 2]
    cell_indices: dict[int, torch.Tensor],       # {lvl: [B] long, -1 = pruned}
    centroids_per_level: dict[int, torch.Tensor],# {lvl: [N_lvl, 2]}
    sigmas_km: Mapping[int, float] = DEFAULT_SIGMAS_KM,
    level_weights: Mapping[int, float] | None = None,
    country_idx: torch.Tensor | None = None,                # [B] long, -1 ok
    cell_to_country: Mapping[int, torch.Tensor] | None = None,  # {lvl: [N_lvl] long}
    border_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Sum of soft-CE across levels. Returns (total_loss, per_level_loss_dict).

    Pruned samples (cell_indices[lvl] == -1) are masked out of that level's
    loss term. If every sample in the batch is pruned at a level, that
    level contributes 0 (and we skip it to avoid NaN).

    Border-aware penalty (V2+): if `border_lambda > 0` and `country_idx` +
    `cell_to_country` are provided, samples whose argmax-cell falls in a
    different country than the truth get their per-level CE multiplied by
    `1 + border_lambda`. Trains the model to prefer "right country, wrong
    town" over "right look, wrong country" — directly attacks the EU
    lookalike-confusion failure mode (PL/CZ/RO/HU). The penalty is
    non-differentiable on the multiplier itself (argmax + comparison),
    but the gradient flows through the scaled CE normally.
    """
    level_weights = level_weights or {lvl: 1.0 for lvl in logits_per_level}
    per_level: dict[int, torch.Tensor] = {}
    total = torch.zeros((), device=true_latlng.device)
    use_border = (
        border_lambda > 0.0
        and country_idx is not None
        and cell_to_country is not None
    )

    for lvl, logits in logits_per_level.items():
        mask = cell_indices[lvl] != PRUNED_LABEL  # [B]
        if not mask.any():
            # Every sample at this level was pruned in this batch. Touch the
            # logits with a 0-weighted term so DDP sees this level's head
            # parameters as "used" and doesn't deadlock on grad sync.
            zero_touch = (logits.sum() * 0.0)
            per_level[lvl] = zero_touch
            total = total + zero_touch
            continue
        l = logits[mask]                         # [B', N]
        latlng = true_latlng[mask]               # [B', 2]
        smooth = haversine_smooth_targets(latlng, centroids_per_level[lvl], sigmas_km[lvl])
        log_probs = F.log_softmax(l, dim=-1)
        ce_per_sample = -(smooth * log_probs).sum(dim=-1)  # [B']

        if use_border and lvl in cell_to_country:
            true_country = country_idx[mask]                       # [B']
            pred_idx = l.argmax(dim=-1)                            # [B']
            lookup = cell_to_country[lvl].to(l.device)             # [N_lvl]
            pred_country = lookup[pred_idx]                        # [B']
            valid = (true_country != PRUNED_LABEL) & (pred_country != PRUNED_LABEL)
            border_violation = (pred_country != true_country) & valid
            border_mult = 1.0 + border_lambda * border_violation.float()
            ce_per_sample = border_mult * ce_per_sample

        ce = ce_per_sample.mean()
        per_level[lvl] = ce
        total = total + level_weights[lvl] * ce

    return total, per_level


def country_loss(
    country_logits: torch.Tensor,    # [B, num_countries]
    country_idx: torch.Tensor,       # [B] long, -1 = unknown / null country
) -> torch.Tensor:
    """Standard cross-entropy on the auxiliary country head.

    Samples whose `country_code` was NULL in the catalog (about 7% of train)
    carry `country_idx == PRUNED_LABEL` and are masked out — same convention
    as `hierarchical_loss`. If every sample in the batch has unknown country,
    we return a 0-weighted touch on the logits so DDP sees the head's
    parameters as used and doesn't deadlock.
    """
    mask = country_idx != PRUNED_LABEL
    if not mask.any():
        return country_logits.sum() * 0.0
    return F.cross_entropy(country_logits[mask], country_idx[mask])
