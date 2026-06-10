"""Inference-time post-processing for hierarchical predictions.

The Stage 1 model's natural output is the L12 cell argmax centroid. But L12 is
pruned (only cells with ≥5 train samples are kept), so for ~65% of test panos
there is no "correct" L12 cell — the model can only choose from L12 cells
that exist in our vocabulary, which sometimes lands thousands of km from the
true location.

L9 is coarser (~20 km cells) but always populated and never makes the
"catastrophic country jump" mistake. The rule: if the L12 prediction is
geographically far from the L9 prediction, the L12 has gone off the rails —
fall back to L9.

Empirically (epoch 3, 1000-pano test set, threshold 500 km):
    mean error           603.7 km → 174.1 km   (−71%)
    p90 error           1510   km → 419   km   (−72%)
    median error          83.8 km →  78.3 km   (slightly better)
    within-25 km coverage 26.2% →  26.1%       (unchanged)
    within-750 km         85.5% →  97.3%       (+11.8 pp)
    fallback triggered      —    →  ~20% of predictions

The catastrophic-tail cases (USA, Brazil, Argentina at 1900+ km median) drop
to country-bounded errors. Close-correct cases keep their precise L12 answer.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

# L9 cell side is ~20 km. A correct L12 cell sits inside its parent L9 region,
# so L12 centroids should be within ~30 km of L9 centroids in normal cases. We
# use 500 km as a conservative threshold that triggers only on the catastrophic
# tail (model jumped to the wrong country/region) and avoids false positives in
# urban areas where L12 may legitimately drift further than its L9 cell radius.
DEFAULT_FALLBACK_THRESHOLD_KM = 500.0


class FallbackResult(NamedTuple):
    final_lat: torch.Tensor       # [B]
    final_lng: torch.Tensor       # [B]
    fallback_used: torch.Tensor   # [B] bool


def _haversine_km(
    lat1: torch.Tensor, lng1: torch.Tensor,
    lat2: torch.Tensor, lng2: torch.Tensor,
) -> torch.Tensor:
    R = 6371.0
    d = math.pi / 180.0
    lat1, lat2 = lat1 * d, lat2 * d
    dlat = lat2 - lat1
    dlng = (lng2 - lng1) * d
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng / 2) ** 2
    return 2 * R * torch.asin(torch.clamp(torch.sqrt(a), max=1.0))


def apply_l9_fallback(
    l9_centroid_pred: torch.Tensor,    # [B, 2] (lat, lng)
    l12_centroid_pred: torch.Tensor,   # [B, 2] (lat, lng)
    threshold_km: float = DEFAULT_FALLBACK_THRESHOLD_KM,
) -> FallbackResult:
    """Replace L12's prediction with L9's where they are geographically far apart.

    Args:
        l9_centroid_pred:  per-sample L9 argmax centroid, shape [B, 2].
        l12_centroid_pred: per-sample L12 argmax centroid, shape [B, 2].
        threshold_km: haversine distance above which to use L9.

    Returns:
        FallbackResult with the chosen lat/lng per sample plus a bool mask of
        which samples used the fallback.
    """
    d = _haversine_km(
        l9_centroid_pred[:, 0], l9_centroid_pred[:, 1],
        l12_centroid_pred[:, 0], l12_centroid_pred[:, 1],
    )
    use_l9 = d > threshold_km
    final_lat = torch.where(use_l9, l9_centroid_pred[:, 0], l12_centroid_pred[:, 0])
    final_lng = torch.where(use_l9, l9_centroid_pred[:, 1], l12_centroid_pred[:, 1])
    return FallbackResult(final_lat, final_lng, use_l9)
