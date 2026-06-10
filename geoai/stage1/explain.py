"""Occlusion attribution — visualize *where* Stage 1 looked.

The deployed prediction picks an L9 cell by cosine similarity between the
pano's pooled features and that cell's nearest stored prototype (ProtoNet
select). To show what drove that choice we slide a gray patch over each of the
4 perspective crops and measure how much it drops that similarity: a big drop
means the model leaned on that region. The per-view heatmaps are composited
onto the unwrapped panorama strip and returned as a PNG.

This is attribution against the *actual* decision (the selected cell's
prototype match), not a generic saliency map.
"""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from py360convert import utils as _p360

from geoai.config import CROP_FOV_DEG, CROP_HEADINGS_DEG

# Display equirect width for the overlay (height follows the pano's aspect).
# The per-view heat is splat at this granularity, then lightly filled/blurred.
_DISP_W = 2048
_SPLAT = 640  # resolution at which each crop's heat is projected back


@torch.inference_mode()
def occlusion_explain(
    model,
    protonet,
    device,
    pixel_values: torch.Tensor,      # [4, 3, 384, 384]
    pano_image: Image.Image,         # equirect pano (used as crops were rendered)
    *,
    grid: int = 14,
    select_topk: int = 500,
    batch: int = 24,
) -> tuple[bytes, dict]:
    """Return (PNG bytes of the equirect-with-heatmap, metadata dict)."""
    crops = pixel_values.to(device)
    V, _, H, _ = crops.shape

    # Base pooled features + the ProtoNet-selected L9 cell (deployed path:
    # top-K candidates from the L9 head, pick the best prototype match).
    view_emb = model.encode_views_unconcat(crops.unsqueeze(0))   # [1, V, D]
    feat = model.project_pooled(view_emb)                        # [1, P]
    l9 = model.heads_forward(feat)[9]
    kk = min(select_topk, l9.shape[-1])
    cand = l9[0].topk(kk).indices.tolist()
    best_ci, best_sim = cand[0], -1.0
    for ci in cand:
        r = protonet.refine(feat, cell_idx=ci)
        if r is not None and r.top_similarity > best_sim:
            best_sim, best_ci = r.top_similarity, ci
    ref = protonet.refine(feat, cell_idx=best_ci)

    # Similarity scorer against the selected cell's prototypes.
    start = int(protonet.cell_starts[best_ci].item())
    end = int(protonet.cell_starts[best_ci + 1].item())
    proto_t = protonet._normalised_features()[start:end].float().t().to(device)  # [P, n]

    def sims_of(feats: torch.Tensor) -> torch.Tensor:  # [B, P] -> [B]
        return (F.normalize(feats, dim=-1).float() @ proto_t).max(dim=1).values

    base_sim = float(sims_of(feat).item())

    # Per-view occlusion. Mask a grid block (gray = 0 in SigLIP 0.5/0.5 norm
    # space), re-encode only that view, re-pool, measure the similarity drop.
    step = H // grid
    heats: list[np.ndarray] = []
    for v in range(V):
        variants = []
        for gy in range(grid):
            for gx in range(grid):
                m = crops[v].clone()
                m[:, gy * step:(gy + 1) * step, gx * step:(gx + 1) * step] = 0.0
                variants.append(m)
        variants = torch.stack(variants)                          # [G*G, 3, H, W]

        drops = []
        for i in range(0, variants.shape[0], batch):
            chunk = variants[i:i + batch]
            emb_v = model.encode_views_unconcat(chunk.unsqueeze(1))[:, 0, :]
            ve = view_emb.repeat(chunk.shape[0], 1, 1).clone()
            ve[:, v, :] = emb_v
            fb = model.project_pooled(ve)
            drops.append((base_sim - sims_of(fb)).clamp(min=0))
        h = torch.cat(drops).reshape(grid, grid)
        h = (h / (h.max() + 1e-8)).cpu().numpy()
        heats.append(h)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    png = _composite_equirect(pano_image, heats)
    return png, {
        "cell": int(best_ci),
        "sim": round(base_sim, 4),
        "lat": round(ref.lat, 5) if ref else None,
        "lng": round(ref.lng, 5) if ref else None,
    }


def _composite_equirect(pano_image: Image.Image, heats: list[np.ndarray]) -> bytes:
    """Project per-view heat back onto the true equirect and overlay it.

    Each crop was rendered with py360convert's `e2p` (heading = CROP_HEADINGS,
    FOV = CROP_FOV, pitch 0). We reuse the exact same forward mapping
    (xyzpers → xyz2uv → uv2coor) to find, for every heat pixel, where it lands
    on the equirect, and splat it there. The result lines up 1:1 with the pano
    (and thus GeoGuessr's view), rather than 4 separate panels.
    """
    disp = pano_image.convert("RGB")
    if disp.width > _DISP_W:
        disp = disp.resize((_DISP_W, round(_DISP_W * disp.height / disp.width)), Image.BICUBIC)
    W, H = disp.size

    fov = float(np.deg2rad(CROP_FOV_DEG))
    heat_eq = np.zeros((H, W), np.float32)
    for v, heading in enumerate(CROP_HEADINGS_DEG):
        hm = np.asarray(
            Image.fromarray((heats[v] * 255).astype("uint8")).resize((_SPLAT, _SPLAT), Image.BICUBIC)
        ).astype(np.float32) / 255.0
        # crop pixel -> equirect (x, y); e2p uses u = -heading.
        xyz = _p360.xyzpers(fov, fov, -float(np.deg2rad(heading)), 0.0, (_SPLAT, _SPLAT), 0.0)
        uu, vv = _p360.xyz2uv(xyz)
        cx, cy = _p360.uv2coor(uu, vv, H, W)
        xi = np.clip(np.round(cx), 0, W - 1).astype(np.int64).ravel()
        yi = np.clip(np.round(cy), 0, H - 1).astype(np.int64).ravel()
        np.maximum.at(heat_eq, (yi, xi), hm.ravel())

    # Fill the sparse-splat pinholes and smooth (the heat is low-frequency).
    himg = Image.fromarray((heat_eq * 255).astype("uint8"))
    himg = himg.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(4))
    hm = np.asarray(himg).astype(np.float32) / 255.0

    base = np.asarray(disp).astype(np.float32)
    a = (hm * 0.62)[..., None]
    color = np.stack([255 * np.ones_like(hm), 210 - 170 * hm, 40 - 10 * hm], axis=-1)
    out = (base * (1 - a) + color * a).clip(0, 255).astype("uint8")

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue()
