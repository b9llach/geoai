"""`geoai-predict-stage1` — load a checkpoint and inspect predictions.

Usage:
    geoai-predict-stage1 --ckpt <path> --pano-id <id>
    geoai-predict-stage1 --ckpt <path> --n-random 5
    geoai-predict-stage1 --ckpt <path> --equirect /path/to/foo.jpg

For each pano, prints:
    * per-level top-K cell predictions with their centroid (lat, lng)
    * the cell at the finest level's argmax → final predicted (lat, lng)
    * ground-truth (lat, lng) + haversine error (if pano is in the catalog)
    * GADM country/admin1 lookup for each top candidate (for human reading)
"""
from __future__ import annotations

import logging
import math
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import torch
import typer
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from geoai.config import (
    CROP_HEADINGS_DEG,
    CROP_SIZE,
    METADATA_DB,
    PROCESSED_DIR,
    find_equirect,
)
from geoai.processing.perspective import equirect_to_perspective_crops
from geoai.processing import reverse_geo
from geoai.stage1.cells import CellVocab
from geoai.stage1.country_prior import CountryPrior
from geoai.stage1.country_vocab import CountryVocab
from geoai.stage1.dataset import SIGLIP_MEAN, SIGLIP_STD
from geoai.stage1.model import HierarchicalGeocellClassifier
from geoai.stage1.protonet import ProtoNetIndex, ProtoNetRefinement
from geoai.stage1.refinement import apply_l9_fallback

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def _strip_prefixes(state_dict: dict) -> dict:
    """Accelerate may save with `_orig_mod.` (torch.compile) or `module.` (DDP) wrappers."""
    out = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("_orig_mod.", "module."):
            while nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def load_checkpoint(
    ckpt_dir: Path, cells: CellVocab, device: torch.device
) -> HierarchicalGeocellClassifier:
    """Build a model and load the weights from `<ckpt_dir>/model.safetensors`.

    Auto-detects whether the checkpoint includes a V2-style country head by
    inspecting the state_dict for `country_head.weight`. If present, the
    model is built with `num_countries` matching the checkpoint's head size
    so the weights load cleanly instead of being silently dropped under
    strict=False.
    """
    state = load_file(ckpt_dir / "model.safetensors")
    state = _strip_prefixes(state)

    num_countries = 0
    if "country_head.weight" in state:
        num_countries = int(state["country_head.weight"].shape[0])
        log.info(f"checkpoint has a country head with {num_countries} classes")

    model = HierarchicalGeocellClassifier(
        cells=cells, num_countries=num_countries,
        gradient_checkpointing=False,  # no need at inference
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        log.warning(f"unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model = model.to(device).eval()
    return model


def load_country_vocab_for_ckpt(ckpt_dir: Path) -> CountryVocab | None:
    """Load `country_vocab.json` from a V2 checkpoint dir if it exists.

    Returns None for V1 checkpoints (no such file).
    """
    path = Path(ckpt_dir) / "country_vocab.json"
    if not path.exists():
        return None
    return CountryVocab.from_json(path)


_TX = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
])


def load_crops_from_dir(crops_dir: Path, pano_id: str) -> torch.Tensor:
    """Load 4 already-rendered crops from /data/geolocation/processed/crops/.

    Falls back to the canonical sharded layout `<crops_dir>/<pano_id[:2]>/<pano_id>_<heading>.jpg`.
    """
    shard = crops_dir / pano_id[:2]
    views = []
    for h in CROP_HEADINGS_DEG:
        p = shard / f"{pano_id}_{h:03d}.jpg"
        if not p.exists():
            raise FileNotFoundError(f"{p}")
        views.append(_TX(Image.open(p).convert("RGB")))
    return torch.stack(views, dim=0)  # [4, 3, 384, 384]


def render_crops_on_the_fly(equirect_path: Path) -> torch.Tensor:
    """Render the 4 crops in-memory from an equirectangular JPEG."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        paths = equirect_to_perspective_crops(
            equirect_path, Path(tmp), pano_id="adhoc"
        )
        views = [_TX(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(views, dim=0)


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dl = math.radians(lat2 - lat1); dn = math.radians(lng2 - lng1)
    a = math.sin(dl / 2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dn / 2)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def _expected_location(cands: list[dict], topk: int,
                       max_span_km: float) -> tuple[float, float]:
    """Probability-weighted mean of the top-K cell centroids — the model's
    EXPECTED location instead of the single argmax centroid.

    Restricted to candidates within `max_span_km` of the top-1, so we never
    average into the empty space between two distant, confused cells (a model
    torn between two countries should commit to one, not split the difference
    into the ocean). When the mass is spread over neighbouring cells this
    escapes the argmax-centroid quantization; when it's concentrated it just
    returns ~the top-1 centroid. `cands` is an out[lvl]-style prob-sorted list
    of {prob, lat, lng}.
    """
    cands = cands[:max(1, topk)]
    a_lat, a_lng = cands[0]["lat"], cands[0]["lng"]
    coh = [c for c in cands
           if haversine_km(a_lat, a_lng, c["lat"], c["lng"]) <= max_span_km]
    w = sum(c["prob"] for c in coh)
    if w <= 0:
        return a_lat, a_lng
    return (sum(c["lat"] * c["prob"] for c in coh) / w,
            sum(c["lng"] * c["prob"] for c in coh) / w)


def _l6_topk_from_l9_mass(
    logits_per_level: dict[int, torch.Tensor],
    cells: CellVocab,
    topk: int,
) -> torch.Tensor:
    """Return the top-K L6 cell indices, RANKED BY AGGREGATED L9 CHILD MASS
    instead of L6's own logits. Fixes the case where L6 picks a small dense
    cell whose 1 high-prob L9 child beats a different L6 region with many
    medium-prob L9 children (whose sum is actually larger). Returns
    LongTensor [k] of L6 indices."""
    # Cast to float32 — index_add_ + downstream numpy/sort prefer float32.
    l9_probs = logits_per_level[9].softmax(dim=-1)[0].float()    # [n_l9]
    pmap = cells.parent_map(9, 6).to(l9_probs.device)            # [n_l9]
    # Treat pruned-parent L9 cells (pmap == -1) as a sentinel bucket we
    # then discard. We accumulate into a [n_l6 + 1] vector and slice off
    # the last bucket (index = -1 stored at position n_l6).
    n_l6 = cells.vocab_size(6)
    bucket = torch.zeros(n_l6 + 1, device=l9_probs.device, dtype=l9_probs.dtype)
    safe_idx = torch.where(pmap >= 0, pmap, torch.tensor(n_l6, device=pmap.device))
    bucket.index_add_(0, safe_idx, l9_probs)
    l6_mass = bucket[:n_l6]
    k = min(topk, n_l6)
    return l6_mass.topk(k=k).indices                              # [k]


def _joint_rerank(
    logits_per_level: dict[int, torch.Tensor],
    cells: CellVocab,
) -> None:
    """Replace each level's logits with joint log-probabilities, computed as

        joint(L12=c) = log P(L3 = ancestor3(c))
                     + log P(L6 = ancestor6(c))
                     + log P(L9 = ancestor9(c))
                     + log P(L12 = c)

    and aggregated up to L9 / L6 / L3 by summing exp(joint) over descendants.
    Operates IN-PLACE. After this call, every level's argmax / top-K refers
    to the joint ranking — so the L9 polygon shown contains the joint-argmax
    L12, the L6 polygon contains the L9 polygon, and so on. Pruned-ancestor
    L12 cells are pushed to -1e9 instead of -inf so softmax stays finite.

    Mathematically what this does: it's the principled way to combine the
    four hierarchical heads when the model was trained autoregressively —
    instead of taking each head's argmax independently (and possibly getting
    inconsistent regions), we score each L12 cell by the agreement of every
    head about its parent chain. The downstream code paths don't need to
    change; they just see "logits" that already encode the joint.
    """
    needed = (3, 6, 9, 12)
    if not all(lvl in logits_per_level for lvl in needed):
        return

    # Per-level log probs in fp32 (softmax precision matters, downstream
    # numpy needs non-bf16).
    log_p = {
        lvl: logits_per_level[lvl].log_softmax(dim=-1)[0].float()  # [n_lvl]
        for lvl in needed
    }
    device = log_p[12].device
    NEG = -1e9

    def gather_parent(child_log_idx_target_lvl: int, child_lvl: int) -> torch.Tensor:
        """Return [n_child] = log P(parent of each child at target level).
        Pruned-parent entries get NEG."""
        pmap = cells.parent_map(child_lvl, child_log_idx_target_lvl).to(device)
        safe = torch.clamp(pmap, min=0)
        gathered = log_p[child_log_idx_target_lvl].gather(0, safe)
        return torch.where(pmap >= 0, gathered, torch.full_like(gathered, NEG))

    # Joint at L12 = sum of log-probs across the parent chain.
    joint_l12 = (
        log_p[12]
        + gather_parent(9, 12)
        + gather_parent(6, 12)
        + gather_parent(3, 12)
    )  # [n_l12]

    # Aggregate joint upward by summing exp(joint) across descendants.
    # We do this in log-space via logsumexp-on-buckets for numerical stability.
    def agg_to(parent_lvl: int) -> torch.Tensor:
        n_parent = cells.vocab_size(parent_lvl)
        pmap = cells.parent_map(12, parent_lvl).to(device)
        # Sentinel bucket at index n_parent absorbs pruned (-1) parents.
        safe = torch.where(pmap >= 0, pmap, torch.tensor(n_parent, device=device))
        # logsumexp via the max-shift trick, bucketed.
        m = torch.full((n_parent + 1,), NEG, device=device)
        m.scatter_reduce_(0, safe, joint_l12, reduce="amax", include_self=True)
        shifted = (joint_l12 - m[safe]).exp()  # in (0, 1]
        s = torch.zeros(n_parent + 1, device=device, dtype=joint_l12.dtype)
        s.index_add_(0, safe, shifted)
        agg = m[:n_parent] + s[:n_parent].clamp_min(1e-30).log()
        # Parents with zero mass (all children pruned) come out near NEG; fine.
        return agg

    # Write joint scores back into logits_per_level. The downstream code
    # softmaxes these to produce per-level top-K rankings + probs, which is
    # exactly what we want now: ranks are by joint mass.
    logits_per_level[12] = joint_l12.unsqueeze(0)
    logits_per_level[9] = agg_to(9).unsqueeze(0)
    logits_per_level[6] = agg_to(6).unsqueeze(0)
    logits_per_level[3] = agg_to(3).unsqueeze(0)


def _apply_hier_mask(
    logits_per_level: dict[int, torch.Tensor],
    cells: CellVocab,
    parent_lvl: int,
    child_lvls: tuple[int, ...],
    topk: int,
    top_parents_override: torch.Tensor | None = None,
) -> None:
    """Mask the child-level logits to only cells whose S2 parent at
    `parent_lvl` is among the top-K of the parent level's softmax — OR
    the supplied `top_parents_override` if given (used to swap in
    L9-mass-derived L6 ranking).

    Operates IN-PLACE on the entries of `logits_per_level`. Skips levels
    that aren't present in the dict.
    """
    parent_logits = logits_per_level[parent_lvl]
    device = parent_logits.device
    if top_parents_override is not None:
        top_parents = top_parents_override.to(device)
    else:
        parent_probs = parent_logits.softmax(dim=-1)
        k = min(topk, parent_probs.shape[-1])
        top_parents = parent_probs.topk(k=k, dim=-1).indices[0]  # [k]

    for clvl in child_lvls:
        if clvl not in logits_per_level:
            continue
        try:
            pmap = cells.parent_map(clvl, parent_lvl).to(device)  # [n_child]
        except (KeyError, ValueError):
            continue
        # valid_mask[i] = True iff child i's parent is one of the top-K parents
        valid_mask = torch.isin(pmap, top_parents.to(pmap.device))
        # If every cell at this level would be masked (e.g. all L12 children
        # of the chosen L6 cells happen to be pruned), skip masking — better
        # to fall back to the unconstrained head than emit NaN softmax.
        if not valid_mask.any():
            continue
        # Use a very-negative finite value rather than -inf: softmax over a
        # row of all -inf is NaN, which crashes JSON serialization. -1e9
        # makes masked positions effectively zero post-softmax while
        # remaining numerically safe.
        logits_per_level[clvl] = logits_per_level[clvl].masked_fill(
            ~valid_mask.unsqueeze(0), -1e9,
        )


@torch.no_grad()
def predict_one(
    model: HierarchicalGeocellClassifier,
    cells: CellVocab,
    pixel_values: torch.Tensor,        # [4, 3, H, W]
    device: torch.device,
    k: int = 5,
    protonet: ProtoNetIndex | None = None,
    protonet_k: int = 5,
    protonet_temperature: float = 0.1,
    tta: bool = True,
    country_prior: CountryPrior | None = None,
    # top-K was 3 originally (so a top-1 country-head error could be rescued
    # by top-2/3 saving the right country). But at country_top1 ≈ 0.986 the
    # head is essentially never wrong, and the looser top-3 was leaking
    # cross-country argmaxes (NE US → Ottawa: head said "USA 90% / Canada
    # 8%", mask kept both, L12 argmax landed in Canada). top-K=1 trusts the
    # country head fully — safe given its accuracy.
    country_prior_topk: int = 1,
    country_prior_threshold: float = 0.3,
    l12_min_count_to_trust: int = 10,
    country_vocab: CountryVocab | None = None,
    use_country_head: bool = True,
    hier_l6_topk: int = 5,
    hier_l3_topk: int = 3,
    hier_l9_topk: int = 3,
    protonet_topk_l9: int = 1,
    protonet_topk_max_span_km: float = 100.0,
    # ProtoNet as a CONTENT-AWARE SELECTOR over the top-K L9 candidates (V3+).
    # When > 0, refine each of the top-`protonet_select_topk` L9 cells and pick
    # the one whose prototypes best MATCH the query image (max cosine
    # top_similarity), using its refined coordinate as the final answer. The
    # model's probability is a density-biased ranker that can't select the right
    # cell; this routes around it. Diagnostics (E4, test split): greedy ~73km /
    # 19% <25km → select K=200 ~26km / 49% <25km. ~0.13ms/candidate on GPU.
    # 0 = disabled (legacy prob-weighted top-`protonet_topk_l9` refinement).
    protonet_select_topk: int = 0,
    # Probability-weighted EXPECTED location instead of the argmax centroid.
    # When > 0, the final coord (in the L9/L12 centroid branches — NOT when
    # ProtoNet fires) becomes the prob-weighted mean of the top-`expected_topk`
    # coherent cells at that level, within `expected_max_span_km` of the top-1.
    # 0 = legacy argmax-centroid behavior.
    expected_topk: int = 0,
    expected_max_span_km: float = 50.0,
    # When True, replace per-level argmax with joint-probability prediction:
    # P(L12=c) ∝ P(L3=anc3(c)) * P(L6=anc6(c)) * P(L9=anc9(c)) * P(L12=c).
    # All four levels share a single coherent ranking, so the final pin's
    # parent chain is always shown in the visualized polygons. Implies that
    # hier-cascade masking and L6-from-L9 reranking are skipped (joint
    # achieves coherency natively).
    use_joint: bool = False,
) -> dict:
    """Single-pano forward + (optional) ProtoNet L9 refinement + L9 fallback.

    The flow is: encode → hierarchical heads → top-K per level → on the
    L9 top-1 cell, optionally replace centroid with ProtoNet-refined
    (lat, lng) → then apply the L9-vs-L12 catastrophic-tail fallback.

    Order matters: ProtoNet refines the L9 *cell's* prediction inside the
    cell. The fallback rule decides between (refined) L9 and L12 — so when
    the model is right at L9, ProtoNet's sub-cell precision survives;
    when L12 has jumped, the fallback still rescues.

    TTA (test-time augmentation) is on by default. We run SigLIP once on
    the 4 views, then run feat_proj + heads 4 times under different cyclic
    heading shifts of the V dimension, and average the per-level logits.
    Only the cheap projection+heads happen 4×; the expensive SigLIP backbone
    pass runs once. Net cost ~+10% latency, typically -2 to -5 km median.
    """
    pix = pixel_values.unsqueeze(0).to(device)
    has_country_head = use_country_head and (model.country_head is not None)
    country_logits: torch.Tensor | None = None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        if tta:
            view_emb = model.encode_views_unconcat(pix)        # [1, V, backbone_dim]
            n_shifts = view_emb.shape[1]
            features = []
            level_logits_acc: dict[int, torch.Tensor] = {}
            country_logits_acc: torch.Tensor | None = None
            for shift in range(n_shifts):
                f = model.project_pooled(view_emb, shift=shift)  # [1, proj_dim]
                features.append(f)
                ll = model.heads_forward(f)
                for lvl, lg in ll.items():
                    level_logits_acc[lvl] = (level_logits_acc.get(lvl, 0) + lg)
                if has_country_head:
                    cl = model.country_head(f)
                    country_logits_acc = cl if country_logits_acc is None else (country_logits_acc + cl)
            feat = torch.stack(features, dim=0).mean(dim=0)      # [1, proj_dim]
            logits_per_level = {lvl: v / n_shifts for lvl, v in level_logits_acc.items()}
            if country_logits_acc is not None:
                country_logits = country_logits_acc / n_shifts
        else:
            feat = model.encode_pooled(pix)                       # [1, proj_dim]
            logits_per_level = model.heads_forward(feat)          # {lvl: [1, vocab]}
            if has_country_head:
                country_logits = model.country_head(feat)

    # ---- Country-head top-K (V2) — the trained aux-head signal -----------
    # When V2's country head is present, use its direct logits to pick top-K
    # countries (and report them). For V1 we fall back to country_prior's L6-
    # derived distribution. The country_prior masking on L9/L12 candidates
    # still uses per-cell country distributions so the gating is fair.
    top_country_codes: list[str] = []
    top_country_probs: list[float] = []
    if country_logits is not None and country_vocab is not None:
        probs = country_logits.float().softmax(dim=-1)[0]
        topk = probs.topk(k=min(country_prior_topk, probs.shape[-1]))
        top_country_codes = [
            country_vocab.code_at(int(i)) or "?"
            for i in topk.indices.cpu().tolist()
        ]
        top_country_probs = [float(p) for p in topk.values.cpu().tolist()]
        # When the country prior is also configured AND its `codes` list
        # matches the country_vocab, mask candidate cells via the prior's
        # cell-level country distributions using indices derived from the
        # country head. This is the V2-aware version of the V1 hack.
        if country_prior is not None and 9 in logits_per_level:
            cp_idx = torch.tensor(
                [country_prior._cc_to_idx.get(c, -1) for c in top_country_codes],
                dtype=torch.long, device=logits_per_level[9].device,
            ).unsqueeze(0)  # [1, k]
            valid_mask = (cp_idx >= 0)
            if valid_mask.any():
                cp_idx = cp_idx[valid_mask].unsqueeze(0)  # drop unmapped countries
                for lvl in (9, 12):
                    if lvl in logits_per_level:
                        logits_per_level[lvl] = country_prior.mask_logits(
                            logits_per_level[lvl], lvl, cp_idx,
                            keep_threshold=country_prior_threshold,
                        )

    # ---- Country-prior masking via L6-derived signal (V1 fallback) -------
    # Only fires when the model has no country head OR we don't have a
    # CountryVocab. V2 inference goes through the country_logits path above.
    elif country_prior is not None:
        src_lvl = country_prior.source_level
        if src_lvl in logits_per_level:
            top_idx, top_codes = country_prior.top_k(
                logits_per_level[src_lvl], k=country_prior_topk
            )
            top_country_codes = top_codes[0]
            for lvl in (9, 12):
                if lvl in logits_per_level:
                    logits_per_level[lvl] = country_prior.mask_logits(
                        logits_per_level[lvl], lvl, top_idx,
                        keep_threshold=country_prior_threshold,
                    )

    # ---- Hierarchical S2 cell constraint ---------------------------------
    # Force finer-level argmax to stay inside the coarser-level top-K
    # parents. Fixes the "NE-US pano → Ottawa" / "Sweden → Denmark" /
    # "Italy → Slovenia" failure mode where a dense small/foreign L9 cell
    # outscores diffuse correct-region cells. Soft (top-K) so a wrong L3
    # doesn't doom everything below.
    #
    # Cascade:
    #   1. L3 top-K → mask L6/L9/L12 to only children of those L3 cells
    #   2. L6 top-K (re-evaluated AFTER L3 mask) → mask L9/L12 to only
    #      children of those L6 cells
    # When joint mode is requested, do the rerank here (after country masking
    # so foreign cells are excluded from the joint search) and SKIP the
    # hier-cascade / L6-from-L9 / L9→L12 blocks below — joint already produces
    # a single coherent ranking shared by all four levels.
    if use_joint:
        _joint_rerank(logits_per_level, cells)
        log.info("[joint] reranked all levels by joint log-probability")
    if not use_joint and hier_l3_topk > 0 and 3 in logits_per_level:
        _apply_hier_mask(logits_per_level, cells, parent_lvl=3,
                         child_lvls=(6, 9, 12), topk=hier_l3_topk)
    if not use_joint and hier_l6_topk > 0 and 6 in logits_per_level and 9 in logits_per_level:
        # Pick L6 cells whose L9 children carry the most combined probability
        # rather than the L6 head's own argmax. Empirically the L6 head suffers
        # the same density bias as L9 (a tiny dense L6 cell can beat a large
        # diffuse cluster), and using the L6 head's top-K filters out exactly
        # the diffuse-but-correct L6 regions where the right answer lives.
        l6_top = _l6_topk_from_l9_mass(logits_per_level, cells, hier_l6_topk)
        log.info(f"[hier-L6-from-L9] L6 top-{hier_l6_topk} (by L9 child mass): "
                 f"{l6_top.cpu().tolist()}")
        _apply_hier_mask(logits_per_level, cells, parent_lvl=6,
                         child_lvls=(9, 12), topk=hier_l6_topk,
                         top_parents_override=l6_top)
        # Stash for overriding out[6] below so the visualization shows the
        # L6 cells we actually used (by L9 mass), not L6 head's argmax.
        _l6_top_override = l6_top
    else:
        _l6_top_override = None
    # Flag: True when L9 top-K had NO L12 children in vocab. Sparse rural
    # cells (Russia, Mongolia, Canada interior) often have no L12 children
    # passing min_count, so the L9→L12 mask would empty out the L12
    # candidate set entirely. We track this so the final-point-selection
    # below can ignore the unconstrained L12 and use L9 centroid instead.
    _l12_unconstrained = False
    if not use_joint and hier_l9_topk > 0 and 9 in logits_per_level:
        # Cast to float32 to avoid bf16 numpy errors and ensure clean topk.
        _l9_probs = logits_per_level[9].softmax(dim=-1)[0].float()
        _l9_top = _l9_probs.topk(hier_l9_topk)
        _l9_top_idx = _l9_top.indices.cpu().tolist()
        _pmap = cells.parent_map(12, 9)
        _valid_l12 = torch.isin(_pmap, torch.tensor(_l9_top_idx))
        log.info(
            f"[hier-L9→L12] L9 top-{hier_l9_topk} indices: {_l9_top_idx} "
            f"(probs: {[f'{p:.3f}' for p in _l9_top.values.cpu().tolist()]})  "
            f"L12 valid count: {int(_valid_l12.sum())} / {_pmap.shape[0]}"
        )
        if not _valid_l12.any():
            _l12_unconstrained = True
            log.info("[hier-L9→L12] No L12 children in vocab for L9 top-K — "
                     "will force L9 centroid as final coord.")
        _apply_hier_mask(logits_per_level, cells, parent_lvl=9,
                         child_lvls=(12,), topk=hier_l9_topk)
        try:
            _l12_probs = logits_per_level[12].softmax(dim=-1)[0]
            _l12_top1 = int(_l12_probs.argmax().item())
            _l12_parent = int(cells.parent_map(12, 9)[_l12_top1].item())
            log.info(
                f"[hier-L9→L12] L12 top-1 after mask: idx={_l12_top1}  "
                f"its L9 parent idx={_l12_parent}  "
                f"in L9 top-{hier_l9_topk}? "
                f"{_l12_parent in _l9_top.indices.cpu().tolist()}"
            )
        except Exception as e:
            log.warning(f"[hier-L9→L12] post-mask diag failed: {e}")

    # ---- top-K extraction (after masking) ---------------------------------
    out: dict[int, list[dict]] = {}
    for lvl, logits in logits_per_level.items():
        probs = logits.softmax(dim=-1)
        topk = probs.topk(k=min(k, probs.shape[-1]), dim=-1)
        idxs = topk.indices[0].cpu().tolist()
        ps = topk.values[0].cpu().tolist()
        centroids = cells.centroids_tensor(lvl).cpu().numpy()
        out[lvl] = [
            {"cell_idx": idx, "prob": float(p),
             "lat": float(centroids[idx, 0]), "lng": float(centroids[idx, 1]),
             "polygon": cells.cell_polygon(lvl, idx)}
            for idx, p in zip(idxs, ps)
        ]

    # Override out[6] with the L9-mass-derived L6 cells (the ones actually
    # used for masking). Without this, the visualization would show L6
    # head's own top-K — which can be in a different region than where the
    # cascade actually constrains, making the pin appear to be "outside L6".
    if _l6_top_override is not None and 6 in out:
        centroids_l6 = cells.centroids_tensor(6).cpu().numpy()
        # Cast to float32 before numpy() — bf16 is not numpy-compatible.
        l9_probs_final = logits_per_level[9].softmax(dim=-1)[0].float()
        pmap_l9 = cells.parent_map(9, 6).to(l9_probs_final.device)
        n_l6 = cells.vocab_size(6)
        bucket = torch.zeros(n_l6 + 1, device=l9_probs_final.device,
                              dtype=l9_probs_final.dtype)
        safe = torch.where(pmap_l9 >= 0, pmap_l9,
                            torch.tensor(n_l6, device=pmap_l9.device))
        bucket.index_add_(0, safe, l9_probs_final)
        l6_mass = bucket[:n_l6].cpu().numpy()
        out[6] = [
            {"cell_idx": int(idx), "prob": float(l6_mass[int(idx)]),
             "lat": float(centroids_l6[int(idx), 0]),
             "lng": float(centroids_l6[int(idx), 1]),
             "polygon": cells.cell_polygon(6, int(idx))}
            for idx in _l6_top_override.cpu().tolist()
        ]

    # ---- ProtoNet refinement on the top-K L9 cells -------------------------
    # Refine each of the top-K L9 cells; weighted-average the refined points
    # by their L9 logit prob — but only if all refinements are spatially
    # close (max pairwise span ≤ protonet_topk_max_span_km). If they aren't,
    # the model is uncertain between distant L9 cells (likely confusing two
    # countries), and averaging into "no man's land" between them would be
    # worse than committing to top-1. Falls back to top-1 in that case.
    proto_info: ProtoNetRefinement | None = None
    _protonet_selected = False
    if protonet is not None and protonet_select_topk > 0:
        # SELECT mode: pick the top-K L9 cell whose prototypes best match the
        # query image (cosine top_similarity), and take its refined coordinate.
        # Candidates come straight from the (post-masking) L9 logits so we can
        # rank far more than the viz top-K. This is the headline V3 win.
        n_l9 = logits_per_level[9].shape[-1]
        kk = min(protonet_select_topk, n_l9)
        cand_idx = logits_per_level[9].softmax(dim=-1)[0].topk(kk).indices.cpu().tolist()
        best_sim, best_ref, best_ci = -1.0, None, None
        for ci in cand_idx:
            ref = protonet.refine(feat, cell_idx=ci, k=protonet_k,
                                  temperature=protonet_temperature)
            if ref is not None and ref.top_similarity > best_sim:
                best_sim, best_ref, best_ci = ref.top_similarity, ref, ci
        if best_ref is not None:
            proto_info = best_ref
            _final_pn_lat, _final_pn_lng = best_ref.lat, best_ref.lng
            _protonet_selected = True
            # Reflect the selected cell in the L9 viz entry (HUD / polygons).
            cents9 = cells.centroids_tensor(9).cpu().numpy()
            out[9][0]["lat_centroid"] = out[9][0]["lat"]
            out[9][0]["lng_centroid"] = out[9][0]["lng"]
            out[9][0]["lat"] = best_ref.lat
            out[9][0]["lng"] = best_ref.lng
            out[9][0]["cell_idx"] = best_ci
            out[9][0]["protonet_used"] = True
            out[9][0]["protonet_selected"] = True
            out[9][0]["protonet_n_used"] = best_ref.n_used
            out[9][0]["protonet_top_sim"] = best_ref.top_similarity
            out[9][0]["protonet_select_k"] = kk
        else:
            _final_pn_lat = _final_pn_lng = None
    elif protonet is not None:
        topk_l9 = max(1, min(protonet_topk_l9, len(out[9])))
        refinements: list[tuple[float, float, float, ProtoNetRefinement]] = []  # (lat, lng, weight, info)
        for cand in out[9][:topk_l9]:
            ref = protonet.refine(
                feat, cell_idx=cand["cell_idx"],
                k=protonet_k, temperature=protonet_temperature,
            )
            if ref is not None:
                refinements.append((ref.lat, ref.lng, cand["prob"], ref))

        if refinements:
            # Use top-1's refinement as the "anchor" for the spatial-spread test.
            anchor_lat, anchor_lng, _, anchor_ref = refinements[0]
            spans_ok = all(
                haversine_km(anchor_lat, anchor_lng, la, ln) <= protonet_topk_max_span_km
                for la, ln, _, _ in refinements
            )
            if spans_ok and len(refinements) > 1:
                total_w = sum(w for _, _, w, _ in refinements)
                final_lat_pn = sum(la * w for la, _, w, _ in refinements) / total_w
                final_lng_pn = sum(ln * w for _, ln, w, _ in refinements) / total_w
                proto_info = anchor_ref  # carry top-1's diagnostics
            else:
                final_lat_pn, final_lng_pn = anchor_lat, anchor_lng
                proto_info = anchor_ref

            out[9][0]["lat_centroid"] = out[9][0]["lat"]
            out[9][0]["lng_centroid"] = out[9][0]["lng"]
            out[9][0]["lat"] = final_lat_pn
            out[9][0]["lng"] = final_lng_pn
            out[9][0]["protonet_used"] = True
            out[9][0]["protonet_n_used"] = anchor_ref.n_used
            out[9][0]["protonet_top_sim"] = anchor_ref.top_similarity
            out[9][0]["protonet_refinements_used"] = len(refinements)
            # Stash final to be returned below
            _final_pn_lat, _final_pn_lng = final_lat_pn, final_lng_pn
        else:
            _final_pn_lat = _final_pn_lng = None
    else:
        _final_pn_lat = _final_pn_lng = None

    # ---- final point selection ---------------------------------------------
    # The right policy depends on whether L12 is a well-supported cell or a
    # sparse "default-for-the-whole-country" cell (Malaysia: only 1
    # qualifying L12 cell → every pano predicts Jerantut). We gate on the
    # L12 cell's training-pano count: ≥l12_min_count_to_trust means dense
    # urban-ish cell where the centroid is precise; below that means sparse
    # cell where ProtoNet on L9 is more reliable.
    #
    # Decision order:
    #   1. L12 close to L9 (no catastrophic jump):
    #        a. L12 cell densely supported → L12 centroid (V1 default —
    #           Manhattan / Madrid / Tokyo cells are tight).
    #        b. L12 cell sparsely supported → ProtoNet refined L9 if fired,
    #           else L12 centroid (Malaysia / Denmark / sparse-vocab fix).
    #   2. L12 far from L9 (catastrophic jump):
    #        a. ProtoNet fired → ProtoNet refined L9 (sub-cell precision on
    #           the rescue cell).
    #        b. ProtoNet didn't fire → L9 centroid (V1 catastrophic rescue).
    l9_top = out[9][0]
    l12_top = out[12][0]
    l9_pt = torch.tensor([[l9_top["lat"], l9_top["lng"]]])
    l12_pt = torch.tensor([[l12_top["lat"], l12_top["lng"]]])
    fb_naive = apply_l9_fallback(l9_pt, l12_pt)
    catastrophic = bool(fb_naive.fallback_used[0])

    # Probability-weighted expected location (escapes argmax-centroid
    # quantization). Falls back to the argmax centroid when disabled.
    if expected_topk > 0:
        l9_lat, l9_lng = _expected_location(out[9], expected_topk, expected_max_span_km)
        l12_lat, l12_lng = _expected_location(out[12], expected_topk, expected_max_span_km)
    else:
        l9_lat, l9_lng = l9_top["lat"], l9_top["lng"]
        l12_lat, l12_lng = l12_top["lat"], l12_top["lng"]

    # L12 cell density signal
    try:
        l12_count = int(cells.counts_tensor(12)[l12_top["cell_idx"]].item())
    except Exception:
        l12_count = 0

    pn_avail = (proto_info is not None and _final_pn_lat is not None)
    if _protonet_selected and _final_pn_lat is not None:
        # ProtoNet SELECT mode: the similarity-selected L9 cell's refined coord
        # IS the answer. It beat the L12-density/centroid cascade by a wide
        # margin in eval (median ~73→26km), so use it directly — no L12-density
        # gating, which would just reintroduce the density-biased L12 argmax.
        final_lat, final_lng = _final_pn_lat, _final_pn_lng
        fallback_used = False
    elif _l12_unconstrained:
        # Sparse-region case: L9 top-K cells had NO L12 children in vocab,
        # so L12's argmax is unconstrained and untrustworthy. Use the same
        # rescue path as a catastrophic jump.
        log.info("[final-coord] L12 unconstrained — using L9 path "
                 f"(protonet={'yes' if pn_avail else 'no'}).")
        if pn_avail:
            final_lat, final_lng = _final_pn_lat, _final_pn_lng
        else:
            final_lat, final_lng = l9_lat, l9_lng
        fallback_used = True
    elif catastrophic:
        if pn_avail:
            final_lat, final_lng = _final_pn_lat, _final_pn_lng
        else:
            # Catastrophic L12 jump → L9 (expected location or its centroid).
            final_lat, final_lng = l9_lat, l9_lng
        fallback_used = True
    else:
        if l12_count >= l12_min_count_to_trust:
            # Trust L12 — well-supported dense cell.
            final_lat, final_lng = l12_lat, l12_lng
        elif pn_avail:
            # Sparse L12 cell — likely the "lone qualifying cell for this
            # country" pattern. Prefer ProtoNet on L9.
            final_lat, final_lng = _final_pn_lat, _final_pn_lng
        else:
            # Sparse L12 AND no ProtoNet: still use L12 centroid (no better option).
            final_lat, final_lng = l12_lat, l12_lng
        fallback_used = False
    out[12][0]["cell_count"] = l12_count

    return {
        "per_level": out,
        "final_lat": float(final_lat),
        "final_lng": float(final_lng),
        "fallback_used": fallback_used,
        "protonet_used": proto_info is not None,
        "protonet_info": (
            None if proto_info is None
            else {"n_prototypes": proto_info.n_prototypes,
                  "n_used": proto_info.n_used,
                  "top_similarity": proto_info.top_similarity,
                  "refinements_used": out[9][0].get("protonet_refinements_used", 1)}
        ),
        "country_prior_top": top_country_codes,
        "country_prior_probs": top_country_probs,
        "country_head_used": country_logits is not None,
    }


def _print_prediction(result: dict, truth: Optional[dict]) -> None:
    lvl_names = {3: "L3 (~1150km)", 6: "L6 (~144km)", 9: "L9 (~18km)", 12: "L12 (~2.3km)"}
    tags = []
    if result.get("fallback_used"):
        tags.append("L9 fallback")
    if result.get("protonet_used"):
        info = result.get("protonet_info") or {}
        tags.append(f"ProtoNet n={info.get('n_used','?')} sim={info.get('top_similarity', 0):.2f}")
    suffix = f" [{', '.join(tags)}]" if tags else ""
    print()
    if truth:
        print(f"  TRUTH:  ({truth['lat']:.4f}, {truth['lng']:.4f})  "
              f"{truth.get('country_name','')} / {truth.get('admin1_name','')}")
        d = haversine_km(truth["lat"], truth["lng"], result["final_lat"], result["final_lng"])
        print(f"  PRED:   ({result['final_lat']:.4f}, {result['final_lng']:.4f})  "
              f"haversine error = {d:.1f} km{suffix}")
    else:
        print(f"  PRED:   ({result['final_lat']:.4f}, {result['final_lng']:.4f})  (no ground truth){suffix}")
    print()
    for lvl, candidates in sorted(result["per_level"].items()):
        print(f"  {lvl_names.get(lvl, f'L{lvl}')}:")
        for c in candidates:
            geo = reverse_geo.reverse_geocode(c["lat"], c["lng"])
            cc = (geo.get("country_code") or "??")
            a1 = (geo.get("admin1_name") or "")
            print(f"    p={c['prob']:.3f}  ({c['lat']:7.3f}, {c['lng']:8.3f})  {cc}  {a1}")


@app.command()
def main(
    ckpt: Path = typer.Option(..., exists=True, help="Checkpoint dir (e.g. .../epoch_00)"),
    pano_id: Optional[str] = typer.Option(None, help="Inspect a specific pano from the catalog"),
    n_random: int = typer.Option(0, help="Sample N random test-split panos and predict on each"),
    equirect: Optional[Path] = typer.Option(None, exists=True, help="Path to an arbitrary equirectangular JPG"),
    k: int = typer.Option(5, help="Top-K candidates per level"),
    device: str = typer.Option("cuda:0", help="cuda:0 / cuda:1 / cpu"),
    db_path: Path = typer.Option(METADATA_DB),
    protonet_path: Optional[Path] = typer.Option(
        None, help="Optional ProtoNet L9 index (.pt). When provided, L9 centroid is replaced with feature-NN-refined (lat, lng)."
    ),
    protonet_k: int = typer.Option(5, help="ProtoNet: top-K neighbors averaged"),
    protonet_temperature: float = typer.Option(0.1, help="ProtoNet: softmax τ over similarities"),
    protonet_topk_l9: int = typer.Option(1, help="Refine top-K L9 cells (1=top-1 only). Top-K results are weight-averaged when within max-span."),
    tta: bool = typer.Option(True, "--tta/--no-tta", help="Heading-shift test-time augmentation"),
    country_prior_dir: Optional[Path] = typer.Option(
        None,
        help="Directory holding l3_to_country.json + l{6,9,12}_to_country.json. "
             "Enables L3-derived country masking on L9/L12 candidates.",
    ),
    country_prior_topk: int = typer.Option(3, help="Top-K countries kept by the L3 prior mask"),
    cells_parquet: Optional[Path] = typer.Option(
        None,
        help="Path to cells.parquet for the cell vocab. Defaults to PROCESSED_DIR/cells.parquet.",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # vocab + model
    if cells_parquet is not None:
        cells = CellVocab.from_parquet(path=cells_parquet)
    else:
        cells = CellVocab.from_parquet()
    log.info(f"vocab: {cells} (from {cells_parquet or 'default'})")
    dev = torch.device(device)
    model = load_checkpoint(ckpt, cells, dev)
    log.info(f"loaded checkpoint: {ckpt}")
    reverse_geo.load_gadm()  # warm up for human-readable cell labels

    protonet: ProtoNetIndex | None = None
    if protonet_path is not None:
        log.info(f"loading ProtoNet index from {protonet_path} ...")
        protonet = ProtoNetIndex.from_file(protonet_path, device=dev)
        log.info(f"  {protonet}")

    country_prior: CountryPrior | None = None
    if country_prior_dir is not None:
        log.info(f"loading country prior from {country_prior_dir} ...")
        country_prior = CountryPrior.from_files(country_prior_dir, source_level=6).to(dev)
        log.info(f"  {country_prior}")

    # decide what to predict
    panos: list[tuple[str, Optional[dict]]] = []  # (pano_id, ground_truth_or_None)
    if equirect:
        pixel_values = render_crops_on_the_fly(equirect)
        result = predict_one(
            model, cells, pixel_values, dev, k=k,
            protonet=protonet, protonet_k=protonet_k,
            protonet_temperature=protonet_temperature,
            tta=tta,
            country_prior=country_prior, country_prior_topk=country_prior_topk,
            protonet_topk_l9=protonet_topk_l9,
        )
        print(f"\n=== {equirect.name} ===")
        _print_prediction(result, truth=None)
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if pano_id:
        row = conn.execute(
            "SELECT pano_id, lat, lng, country_name, admin1_name FROM panos WHERE pano_id = ?",
            (pano_id,),
        ).fetchone()
        if row is None:
            sys.exit(f"pano_id {pano_id} not found in catalog")
        panos.append((row["pano_id"], dict(row)))
    elif n_random > 0:
        rows = conn.execute(
            f"""SELECT pano_id, lat, lng, country_name, admin1_name FROM panos
                WHERE split = 'test'
                  AND (SELECT COUNT(*) FROM crops c WHERE c.pano_id = panos.pano_id) = 4
                ORDER BY RANDOM() LIMIT ?""",
            (n_random,),
        ).fetchall()
        panos = [(r["pano_id"], dict(r)) for r in rows]
    else:
        sys.exit("Pass exactly one of --pano-id, --n-random, or --equirect")

    distances = []
    for pid, truth in panos:
        try:
            pixel_values = load_crops_from_dir(PROCESSED_DIR / "crops", pid)
        except FileNotFoundError as e:
            log.warning(f"skipping {pid}: {e}")
            continue
        result = predict_one(
            model, cells, pixel_values, dev, k=k,
            protonet=protonet, protonet_k=protonet_k,
            protonet_temperature=protonet_temperature,
            tta=tta,
            country_prior=country_prior, country_prior_topk=country_prior_topk,
            protonet_topk_l9=protonet_topk_l9,
        )
        print(f"\n=== {pid} ===")
        _print_prediction(result, truth)
        if truth:
            distances.append(haversine_km(
                truth["lat"], truth["lng"], result["final_lat"], result["final_lng"]
            ))

    if len(distances) > 1:
        import numpy as np
        arr = np.array(distances)
        print()
        print(f"=== summary over {len(arr)} panos ===")
        print(f"  median: {np.median(arr):.1f} km")
        print(f"  mean:   {np.mean(arr):.1f} km")
        for thr in (1, 5, 25, 200, 750):
            print(f"  within {thr:>3} km: {100*(arr < thr).mean():.1f}%")


if __name__ == "__main__":
    app()
