"""Hierarchical S2-cell classifier (Stage 1).

```
4 perspective views (384x384) ──► SigLIP2 vision encoder ──► 4 × [B, D]
                                                              │
                                              concat across views
                                                              │ [B, 4D]
                                              project to 1024
                                                              │
                                ┌───────── L3 head ──────► logits[L3]
                                │  embed L3 prediction (256)
                                │           │
                                ├──────  + L6 head ──────► logits[L6]
                                │  embed L6 prediction (256)
                                │           │
                                ├──────  + L9 head ──────► logits[L9]
                                │  embed L9 prediction (256)
                                │           │
                                └──────  + L12 head ─────► logits[L12]
```

Pooling = concat per the project's design choice (PLAN.md §"Open Questions"
#2): preserves per-view information that mean-pooling discards. The
dataset's random heading-rotation augmentation gives heading-invariance.

Conditioning between levels follows GeoToken (arXiv:2511.01082): each
level's head sees both the pooled image features AND a 256-dim embedding
of the previously-predicted cell. During training we teacher-force on
ground-truth cells; during inference we use argmax (or beam search at
serving time).
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
from transformers import AutoModel

from geoai.stage1.cells import PRUNED_LABEL, CellVocab

DEFAULT_BACKBONE = "google/siglip2-so400m-patch14-384"
PROJ_DIM = 1024
COND_DIM = 256


class HierarchicalGeocellClassifier(nn.Module):
    def __init__(
        self,
        cells: CellVocab,
        backbone_name: str = DEFAULT_BACKBONE,
        proj_dim: int = PROJ_DIM,
        cond_dim: int = COND_DIM,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = True,
        num_countries: int = 0,
    ):
        super().__init__()
        self.cells = cells
        self.levels = cells.levels
        self.num_countries = num_countries

        # SigLIP2 ships as a vision+text dual encoder; we want the vision tower only.
        # Gradient checkpointing must be toggled on the parent SiglipModel (the
        # inner SiglipVisionTransformer doesn't expose gradient_checkpointing_enable
        # in transformers 4.50+) — it propagates to submodules.
        full = AutoModel.from_pretrained(backbone_name)
        if gradient_checkpointing:
            # Cuts activation memory ~3-4x for the backbone in exchange for one
            # extra forward pass. Required to fit BS≥2 on a 24 GB 4090 once
            # DDP gradient buckets are accounted for.
            full.gradient_checkpointing_enable()
        self.backbone = full.vision_model
        self.backbone_dim = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Concat 4 views, project to a manageable hidden dim.
        self.feat_proj = nn.Sequential(
            nn.Linear(4 * self.backbone_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )

        # One head per level. Each head sees the pooled features concatenated
        # with the prior level's cell embedding (or just the features, for
        # the first level). Cell embedding tables exist only for levels whose
        # output conditions a *later* level — the last level's embedding
        # would be unused, and DDP complains about dead parameters.
        self.heads = nn.ModuleList()
        self.cell_embeds = nn.ModuleList()
        prev_cond = 0
        for i, lvl in enumerate(self.levels):
            self.heads.append(nn.Linear(proj_dim + prev_cond, cells.vocab_size(lvl)))
            if i < len(self.levels) - 1:
                self.cell_embeds.append(nn.Embedding(cells.vocab_size(lvl), cond_dim))
                prev_cond = cond_dim

        # Auxiliary country head (V2+). Reads the same pooled features as the
        # hierarchical heads — no extra encoder pass. Disabled when
        # num_countries == 0 so V1-shape checkpoints load cleanly.
        if num_countries > 0:
            self.country_head = nn.Linear(proj_dim, num_countries)
        else:
            self.country_head = None

    def encode_views_unconcat(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, V, 3, H, W] → [B, V, backbone_dim] per-view embeddings.

        The expensive SigLIP forward runs B*V times here. Concat-pooling and
        feat_proj happen separately in `encode_views_concat` so test-time
        augmentation (heading-shift TTA) can reuse these per-view embeddings
        across multiple shift permutations without re-encoding.
        """
        B, V, C, H, W = pixel_values.shape
        flat = pixel_values.reshape(B * V, C, H, W)
        out = self.backbone(pixel_values=flat)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            view_emb = out.pooler_output
        else:
            view_emb = out.last_hidden_state.mean(dim=1)
        return view_emb.reshape(B, V, -1)

    def encode_views(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, V, 3, H, W] → [B, V*backbone_dim] concat-pooled across views."""
        view_emb = self.encode_views_unconcat(pixel_values)
        B, V, D = view_emb.shape
        return view_emb.reshape(B, V * D)

    def encode_pooled(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 4, 3, H, W] → [B, proj_dim] post-projection pooled features.

        Same vector that gets fed to the hierarchical heads. Useful for
        ProtoNet feature-NN refinement where we want a learned, head-aligned
        embedding of the query pano.
        """
        return self.feat_proj(self.encode_views(pixel_values))

    def project_pooled(self, view_emb: torch.Tensor, shift: int = 0) -> torch.Tensor:
        """[B, V, backbone_dim] (already encoded) → [B, proj_dim] pooled features.

        Applies an optional cyclic heading shift along the V dimension before
        concat. Used by heading-shift TTA: we encode views once, then call
        this 4 times with different shifts to get 4 cheap pooled features.
        """
        if shift != 0:
            view_emb = torch.roll(view_emb, shifts=shift, dims=1)
        B, V, D = view_emb.shape
        return self.feat_proj(view_emb.reshape(B, V * D))

    def heads_forward(
        self,
        feat: torch.Tensor,                                        # [B, proj_dim]
        teacher_forcing: Mapping[int, torch.Tensor] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run only the hierarchical heads on already-pooled features.

        Split out so callers (e.g. predict.py + ProtoNet) can compute features
        once and reuse them for both classification and refinement.
        """
        logits_per_level: dict[int, torch.Tensor] = {}
        prev_embed: torch.Tensor | None = None

        last_i = len(self.levels) - 1
        for i, lvl in enumerate(self.levels):
            x = feat if prev_embed is None else torch.cat([feat, prev_embed], dim=-1)
            logits = self.heads[i](x)
            logits_per_level[lvl] = logits

            if i == last_i:
                # No further levels condition on this prediction — skip the embed lookup.
                break
            if teacher_forcing is not None and lvl in teacher_forcing:
                tf = teacher_forcing[lvl]
                # Pruned (-1) samples can't index the embedding; replace with argmax for those.
                argmax = logits.argmax(dim=-1)
                next_idx = torch.where(tf == PRUNED_LABEL, argmax, tf)
            else:
                next_idx = logits.argmax(dim=-1)
            prev_embed = self.cell_embeds[i](next_idx)

        return logits_per_level

    def forward(
        self,
        pixel_values: torch.Tensor,                       # [B, 4, 3, H, W]
        teacher_forcing: Mapping[int, torch.Tensor] | None = None,  # {lvl: [B] cell idx, -1 ok}
        return_country: bool = False,
        return_features: bool = False,
    ):
        """Returns:
            * dict (default)  — {lvl: logits}
            * (dict, country_logits)            if return_country and head exists
            * (dict, features)                  if return_features (and not return_country)
            * (dict, country_logits, features)  if both flags set and head exists

        Country always precedes features when both are returned.
        """
        feat = self.encode_pooled(pixel_values)
        logits_per_level = self.heads_forward(feat, teacher_forcing=teacher_forcing)

        extras: list[torch.Tensor] = []
        if return_country and self.country_head is not None:
            extras.append(self.country_head(feat))
        if return_features:
            extras.append(feat)
        if extras:
            return (logits_per_level, *extras)
        return logits_per_level

    def predict_topk(
        self,
        pixel_values: torch.Tensor,
        k_per_level: int = 5,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """Greedy beam-of-1 down the hierarchy, returning top-K cells *at each level*.

        For Stage 2 we want top-K *final* candidates with their (lat, lng) so the
        MLLM has multiple regions to verify. v1 returns per-level top-K decoupled;
        proper beam search across levels is a v1.1 upgrade.
        """
        logits_per_level = self.forward(pixel_values)
        out = {}
        for lvl, logits in logits_per_level.items():
            probs = logits.softmax(dim=-1)
            topk = probs.topk(k=min(k_per_level, probs.shape[-1]), dim=-1)
            out[lvl] = {"indices": topk.indices, "probs": topk.values}
        return out
