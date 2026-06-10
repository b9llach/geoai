"""Validation metrics for Stage 1.

Two families of metrics, computed per batch and aggregated externally:

1. **Cell classification** at each level: top-1 / top-5 accuracy. Useful
   for tracking convergence per level.
2. **Geographic accuracy**: take the model's L12 (or finest available)
   prediction → cell centroid → haversine distance to truth. Reports
   median, mean, p10, p90, and within-{1, 5, 25, 200, 750}km fractions.
   These match PLAN.md §"Phase 7 — Metrics".

Aggregating across batches is straightforward — concatenate the per-batch
distance arrays and re-summarise.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from geoai.stage1.cells import PRUNED_LABEL, CellVocab

WITHIN_KM_THRESHOLDS = (0.1, 1.0, 5.0, 25.0, 200.0, 750.0, 2500.0)


def _haversine_np(lat1, lng1, lat2, lng2):
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlng/2)**2
    return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def topk_accuracy(
    logits: torch.Tensor,            # [B, N]
    labels: torch.Tensor,            # [B] long, -1 = pruned/skip
    k: int = 1,
) -> tuple[int, int]:
    """Returns (n_correct, n_evaluated). Pruned labels are skipped."""
    mask = labels != PRUNED_LABEL
    if not mask.any():
        return 0, 0
    pred = logits[mask].topk(k=min(k, logits.shape[-1]), dim=-1).indices
    correct = (pred == labels[mask].unsqueeze(-1)).any(dim=-1).sum().item()
    return correct, int(mask.sum().item())


def predicted_latlng(
    logits: torch.Tensor,            # [B, N]
    centroids: torch.Tensor,         # [N, 2]
) -> torch.Tensor:
    """Argmax cell → its centroid (lat, lng). Returns [B, 2]."""
    pred_idx = logits.argmax(dim=-1)
    return centroids[pred_idx]


def haversine_summary(distances_km: np.ndarray) -> dict:
    if distances_km.size == 0:
        return {}
    out = {
        "median_km": float(np.median(distances_km)),
        "mean_km": float(np.mean(distances_km)),
        "p10_km": float(np.percentile(distances_km, 10)),
        "p90_km": float(np.percentile(distances_km, 90)),
    }
    for thr in WITHIN_KM_THRESHOLDS:
        out[f"within_{thr:g}km"] = float(np.mean(distances_km < thr))
    return out


@torch.no_grad()
def evaluate_batch(
    logits_per_level: dict[int, torch.Tensor],
    cell_indices: dict[int, torch.Tensor],
    true_latlng: torch.Tensor,
    cells: CellVocab,
    finest_level_for_distance: int | None = None,
    country_logits: torch.Tensor | None = None,
    country_idx: torch.Tensor | None = None,
) -> dict:
    """Per-batch metric dict; combine across batches for epoch summary."""
    finest = finest_level_for_distance or max(logits_per_level)
    metrics: dict = {}

    # Top-K accuracy at every available level
    for lvl, logits in logits_per_level.items():
        for k in (1, 5):
            n_corr, n_eval = topk_accuracy(logits, cell_indices[lvl], k=k)
            metrics[f"L{lvl}_top{k}_correct"] = n_corr
            metrics[f"L{lvl}_top{k}_n"] = n_eval

    # Auxiliary country-head accuracy (V2+) — present only if the model exposed it.
    if country_logits is not None and country_idx is not None:
        for k in (1, 3):
            n_corr, n_eval = topk_accuracy(country_logits, country_idx, k=k)
            metrics[f"country_top{k}_correct"] = n_corr
            metrics[f"country_top{k}_n"] = n_eval

    # Distance metric uses the finest level's argmax → centroid
    centroids = cells.centroids_tensor(finest, device=true_latlng.device)
    pred = predicted_latlng(logits_per_level[finest], centroids)  # [B, 2]
    d_km = _haversine_np(
        true_latlng[:, 0].cpu().numpy(), true_latlng[:, 1].cpu().numpy(),
        pred[:, 0].cpu().numpy(), pred[:, 1].cpu().numpy(),
    )
    metrics["distances_km"] = d_km
    return metrics


def aggregate_metrics(per_batch: list[dict], levels: tuple[int, ...]) -> dict:
    """Stitch per-batch dicts into single epoch summary."""
    out = {}
    for lvl in levels:
        for k in (1, 5):
            corr = sum(b[f"L{lvl}_top{k}_correct"] for b in per_batch)
            n = sum(b[f"L{lvl}_top{k}_n"] for b in per_batch)
            out[f"L{lvl}_top{k}_acc"] = corr / max(n, 1)
            out[f"L{lvl}_top{k}_n"] = n

    # Country head — present only if every batch reported it (i.e., V2+ run)
    if per_batch and "country_top1_correct" in per_batch[0]:
        for k in (1, 3):
            corr = sum(b[f"country_top{k}_correct"] for b in per_batch)
            n = sum(b[f"country_top{k}_n"] for b in per_batch)
            out[f"country_top{k}_acc"] = corr / max(n, 1)
            out[f"country_top{k}_n"] = n

    distances = np.concatenate([b["distances_km"] for b in per_batch])
    out.update(haversine_summary(distances))
    return out
