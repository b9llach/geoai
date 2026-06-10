"""`geoai-serve` — minimal FastAPI server for interactive Stage 1 inference.

The model and reverse-geocoder are loaded once on startup (lifespan event).
Subsequent /predict requests reuse the same in-memory state — no per-request
checkpoint reload, no per-request GADM load.

Two input shapes accepted:
    - Equirectangular pano (~2:1 aspect): rendered to 4 perspective crops
      before going through the model, exactly matching training.
    - Single image (any aspect): tiled 4× into the 4-view slot. Quality is
      degraded vs a real Street View pano but works for ad-hoc photos.

Run with:
    geoai-serve --ckpt /data/geolocation/processed/checkpoints/stage1/epoch_13 \
                --device cuda:1 --port 8000
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
import torch
import typer
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from geoai.config import CROP_SIZE, METADATA_DB, PROCESSED_DIR
from geoai.processing import reverse_geo
from geoai.stage1.cells import CellVocab
from geoai.stage1.dataset import SIGLIP_MEAN, SIGLIP_STD
from geoai.stage1.predict import (
    load_checkpoint,
    load_country_vocab_for_ckpt,
    predict_one,
    render_crops_on_the_fly,
)
from geoai.stage1.country_prior import CountryPrior
from geoai.stage1.country_vocab import CountryVocab
from geoai.stage1.protonet import ProtoNetIndex

log = logging.getLogger(__name__)

# State populated in lifespan, reused across requests
_state: dict = {}

_TX = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
])

# Default checkpoint — overridable via env var
DEFAULT_CKPT = PROCESSED_DIR / "checkpoints" / "stage1" / "epoch_13"
DEFAULT_DEVICE = "cuda:1"

# ProtoNet index for sub-cell L9 refinement. If the file exists next to the
# checkpoint (built once via `scripts/build_protonet_index.py`), the server
# loads it at startup and refines every L9 prediction in-place. Override the
# path via env var GEOAI_PROTONET_PATH (or pass empty string to disable).
DEFAULT_PROTONET_PATH = DEFAULT_CKPT / "protonet_l9.pt"

# ProtoNet as a content-aware SELECTOR over the top-K L9 candidates (the big V3
# win: median ~73→25km, <25km ~18→50%). Refines the top-K L9 cells and picks
# the one whose prototypes best match the query image, sidestepping the
# density-biased probability ranker. 0 disables (legacy prob-weighted refine).
# ~0.13ms/candidate on GPU, so 500 ≈ 65ms. K plateaus ~500; beyond that the
# median is flat but the mean worsens (the selector chases global look-alikes),
# so 500 is the sweet spot. Override via env.
PROTONET_SELECT_TOPK = int(os.environ.get("GEOAI_PROTONET_SELECT_TOPK", "500"))

# L6-derived country prior. OFF by default in serve until we have a larger
# eval validating it — the L6 head shares V1's EU-lookalike confusion, so
# masking with it is net-neutral (helps when the prior's top-K includes the
# true country, hurts when the L6 head's confusion has already pushed the
# true country out of top-K). Opt in via GEOAI_COUNTRY_PRIOR_DIR=<dir>.
DEFAULT_COUNTRY_PRIOR_DIR = ""  # explicit empty = disabled

# Stitched panos uploaded via /predict (and tile-stitched by /api/v1/predict)
# are saved here for after-the-fact verification of what the model actually saw.
# Lives inside the repo so the captures are easy to grep/scroll alongside code.
PREDICTION_SAVE_DIR = Path(__file__).resolve().parents[2] / "saved_panos"
PREDICTION_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _save_pano_bytes(image_bytes: bytes, hint: str | None) -> Path | None:
    """Persist the assembled JPEG to disk. `hint` is normally the upload filename
    (e.g. '<panoID>.jpg' from the userscript); we strip directory components and
    fall back to a timestamp if it's missing or unsafe."""
    safe = ""
    if hint:
        safe = Path(hint).name
        if not safe.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            safe = ""
    if not safe:
        safe = f"{int(time.time() * 1000)}.jpg"
    path = PREDICTION_SAVE_DIR / safe
    try:
        path.write_bytes(image_bytes)
        return path
    except OSError as e:
        log.warning(f"failed to save pano to {path}: {e}")
        return None

# Same URL + browser-style headers data_scraper.py uses successfully (50k+
# panos/day). cbk0 with output=tile started 403'ing; this one is what Google
# Maps itself requests today.
_TILE_URL = (
    "https://streetviewpixels-pa.googleapis.com/v1/tile"
    "?cb_client=maps_sv.tactile&panoid={pid}&x={x}&y={y}&zoom={z}&nbt=1&fover=2"
)
_TILE_HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.google.com",
    "Referer": "https://www.google.com/",
    "Sec-Ch-Ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    ),
}


async def _fetch_tile(session: aiohttp.ClientSession, pid: str, z: int, x: int, y: int):
    url = _TILE_URL.format(pid=pid, z=z, x=x, y=y)
    try:
        async with session.get(
            url, headers=_TILE_HEADERS,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as r:
            if r.status == 200:
                body = await r.read()
                if body:
                    return (x, y, body)
    except Exception:
        pass
    return (x, y, None)


async def fetch_pano(pano_id: str, zoom: int = 3, tile_size: int = 512) -> Image.Image:
    """Download all tiles for a pano in parallel and stitch them. Black-bar
    trimming happens separately."""
    n_h = 1 << zoom
    n_v = n_h // 2
    canvas = Image.new("RGB", (n_h * tile_size, n_v * tile_size))
    max_x = max_y = -1

    async with aiohttp.ClientSession() as session:
        coros = [_fetch_tile(session, pano_id, zoom, x, y)
                 for x in range(n_h) for y in range(n_v)]
        results = await asyncio.gather(*coros)

    for x, y, data in results:
        if not data:
            continue
        try:
            tile = Image.open(io.BytesIO(data))
            canvas.paste(tile, (x * tile_size, y * tile_size))
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        except Exception:
            continue

    if max_x < 0:
        raise ValueError(
            f"no tiles fetched for pano_id {pano_id!r} — wrong id, deleted, "
            "or Google soft-blocked this IP"
        )
    return canvas.crop((0, 0, (max_x + 1) * tile_size, (max_y + 1) * tile_size))


def crop_black_bars(img: Image.Image) -> Image.Image:
    """Trim solid-black borders left over from missing edge tiles. Vectorized
    via numpy; ~50× faster than the pixel-by-pixel approach in the user's
    original Python script."""
    arr = np.asarray(img.convert("RGB"))
    nonblack = arr.sum(axis=2) > 0
    if not nonblack.any():
        return img
    rows = nonblack.any(axis=1)
    cols = nonblack.any(axis=0)
    rmin = int(rows.argmax())
    rmax = int(len(rows) - 1 - rows[::-1].argmax())
    cmin = int(cols.argmax())
    cmax = int(len(cols) - 1 - cols[::-1].argmax())
    if rmin == 0 and cmin == 0 and rmax == img.height - 1 and cmax == img.width - 1:
        return img
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + GADM once at server startup. ~30s on first run."""
    import os
    ckpt = Path(os.environ.get("GEOAI_CKPT", str(DEFAULT_CKPT)))
    device = torch.device(os.environ.get("GEOAI_DEVICE", DEFAULT_DEVICE))

    log.info(f"loading vocab + model from {ckpt} on {device} ...")
    cells_parquet_env = os.environ.get("GEOAI_CELLS_PARQUET", "")
    if cells_parquet_env:
        cells = CellVocab.from_parquet(path=Path(cells_parquet_env))
        log.info(f"  cell vocab from {cells_parquet_env}")
    else:
        cells = CellVocab.from_parquet()
    model = load_checkpoint(ckpt, cells, device)

    # Auto-load V2 country vocab if it lives next to the checkpoint.
    country_vocab: Optional[CountryVocab] = load_country_vocab_for_ckpt(ckpt)
    if country_vocab is not None:
        log.info(f"loaded country vocab from {ckpt}/country_vocab.json: {country_vocab}")

    # Optional ProtoNet index. Empty-string env var explicitly disables;
    # missing-file is silent (server boots without refinement).
    # Defaults to <actual ckpt>/protonet_l9.pt so swapping --ckpt doesn't
    # accidentally load a stale V1 index against a V2 vocab.
    pn_path_str = os.environ.get("GEOAI_PROTONET_PATH", str(ckpt / "protonet_l9.pt"))
    protonet: Optional[ProtoNetIndex] = None
    if pn_path_str:
        pn_path = Path(pn_path_str)
        if pn_path.exists():
            log.info(f"loading ProtoNet index from {pn_path} ...")
            protonet = ProtoNetIndex.from_file(pn_path, device=device)
            log.info(f"  {protonet}")
            # Pre-warm the lazy feature normalization (a one-time large alloc +
            # cache on first refine). Doing it at startup moves the cost off the
            # first prediction — which otherwise risks a userscript request
            # timeout (the observed 408) — and surfaces any GPU-memory problem
            # here at boot rather than killing a live request.
            log.info("  warming ProtoNet feature normalization ...")
            protonet._normalised_features()
            if device.type == "cuda":
                # Release the fp32 normalization scratch back to the driver so
                # the forward pass has real headroom (not just PyTorch-pool).
                torch.cuda.empty_cache()
                free_b, total_b = torch.cuda.mem_get_info(device)
                log.info(f"  ProtoNet warm. GPU free {free_b/1e9:.1f}/{total_b/1e9:.1f} GB")
        else:
            log.warning(f"protonet path set but file missing: {pn_path}")

    # Optional L3 country prior. Same convention: empty env var disables.
    cp_dir_str = os.environ.get("GEOAI_COUNTRY_PRIOR_DIR", DEFAULT_COUNTRY_PRIOR_DIR)
    country_prior: Optional[CountryPrior] = None
    if cp_dir_str:
        cp_dir = Path(cp_dir_str)
        if (cp_dir / "l6_country_dist.pt").exists():
            log.info(f"loading country prior from {cp_dir} ...")
            country_prior = CountryPrior.from_files(cp_dir, source_level=6).to(device)
            log.info(f"  {country_prior}")
        else:
            log.warning(f"country prior dir set but l6_country_dist.pt missing: {cp_dir}")

    log.info("loading GADM for reverse geocoding ...")
    reverse_geo.load_gadm()

    _state["cells"] = cells
    _state["model"] = model
    _state["device"] = device
    _state["ckpt"] = str(ckpt)
    _state["protonet"] = protonet
    _state["country_prior"] = country_prior
    _state["country_vocab"] = country_vocab
    # Serializes /predict against /load-model so a checkpoint swap doesn't
    # change `_state["model"]` mid-forward-pass. Predicts queue behind a swap,
    # not behind each other in any extra way (GPU already serializes compute).
    _state["model_lock"] = asyncio.Lock()
    log.info(
        f"ready — vocab {cells}, ckpt {ckpt}"
        + (f", protonet ON ({protonet.num_cells} cells)" if protonet else ", protonet OFF")
        + (f", country_prior ON" if country_prior else ", country_prior OFF")
        + (f", country_head ON ({country_vocab.size} classes)" if country_vocab else ", country_head OFF")
    )
    yield
    _state.clear()


app = FastAPI(title="geoai", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _process_image(image_bytes: bytes, mode: str) -> tuple[torch.Tensor, str]:
    """Bytes + mode → (4, 3, 384, 384) tensor + a label describing the path taken.

    mode is one of:
        "auto"      — pick based on aspect ratio (1.85–2.25 → equirect, else single)
        "panorama"  — force the equirect-rendering path
        "single"    — force the tile-4× single-image path
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    aspect = img.size[0] / img.size[1]

    if mode == "auto":
        is_equirect = 1.85 <= aspect <= 2.25
    elif mode == "panorama":
        is_equirect = True
    elif mode == "single":
        is_equirect = False
    else:
        raise HTTPException(400, f"Unknown mode: {mode!r}")

    if is_equirect:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name, quality=92)
            tmp_path = Path(f.name)
        try:
            pixel_values = render_crops_on_the_fly(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        label = f"panorama (4 crops rendered, aspect={aspect:.2f})"
        if mode == "auto":
            label = "auto → " + label
        elif mode == "panorama":
            label = "panorama (forced) → 4 crops rendered"
        return pixel_values, label
    else:
        single = _TX(img)
        pixel_values = single.unsqueeze(0).expand(4, -1, -1, -1).contiguous()
        label = f"single image (tiled 4×, aspect={aspect:.2f}, expect degraded accuracy)"
        if mode == "auto":
            label = "auto → " + label
        elif mode == "single":
            label = "single (forced, tiled 4×, expect degraded accuracy)"
        return pixel_values, label


def _build_response(pixel_values: torch.Tensor, input_label: str,
                    pano_id: Optional[str] = None,
                    cascade: str = "country_only",
                    original_image: Optional[Image.Image] = None,
                    ocr_tiles: Optional[list] = None,
                    on_stage2_progress=None) -> dict:
    """Run inference + assemble the JSON payload both endpoints return.

    cascade modes:
      "plain"        → V1-style argmax-per-level. Nothing extra.
      "country_only" → country head + country prior mask L9/L12, but NO hier cascade,
                       NO L6-from-L9 ranking, NO L12-unconstrained fallback.
                       Lets L12 head pick freely within the predicted country.
      "joint"        → joint-probability prediction: rank L12 cells by
                       P(L3)·P(L6)·P(L9)·P(L12) along the parent chain.
                       Pin is the joint-argmax L12 centroid; L9/L6/L3 polygons
                       shown are the parent chain of that pick. Stacks with
                       country masking. Skips hier cascade (joint achieves
                       coherency natively).
      "fancy"        → all heuristic masks on (hier cascade + L6-from-L9 +
                       L12 fallback + country mask).
      "refined"      → "country_only" Stage 1 + Stage 2 (OCR + VLM + geocode)
                       precision refinement. Returns extra stage2_* fields
                       and overrides top-level lat/lng when Stage 2 commits.
    """
    # Stage 1 is now a single path: raw L9 logits → ProtoNet content-selector
    # over the top-K candidates. Eval (E4): raw+select beats every cascade —
    # the hier/country masks only strip truth-near cells before ProtoNet picks
    # them (fancy: median 30→35, <25km 47→43%, worse tail). So the legacy
    # cascade modes are collapsed; the only meaningful switch is whether Stage 2
    # runs. `refined` adds the OCR+VLM+geocode layer on top of the same Stage 1.
    refined = cascade == "refined"
    result = predict_one(
        _state["model"], _state["cells"], pixel_values, _state["device"], k=5,
        protonet=_state.get("protonet"),
        protonet_select_topk=PROTONET_SELECT_TOPK,
        hier_l3_topk=0, hier_l6_topk=0, hier_l9_topk=0,
        use_joint=False,
    )
    final_geo = reverse_geo.reverse_geocode(result["final_lat"], result["final_lng"])

    per_level = {}
    for lvl, candidates in result["per_level"].items():
        per_level[str(lvl)] = [
            {
                "prob": c["prob"], "lat": c["lat"], "lng": c["lng"],
                "polygon": c.get("polygon"),
                "country_code": (
                    reverse_geo.reverse_geocode(c["lat"], c["lng"])["country_code"] or "??"
                ),
            }
            for c in candidates[:5]
        ]

    resp = {
        # Top-level lat/lng aliases for the userscript that expects {data.lat, data.lng}.
        "lat": result["final_lat"],
        "lng": result["final_lng"],
        "panoID": pano_id,
        "input_type": input_label,
        "final_lat": result["final_lat"],
        "final_lng": result["final_lng"],
        "fallback_used": bool(result.get("fallback_used")),
        "protonet_used": bool(result.get("protonet_used")),
        "protonet_info": result.get("protonet_info"),
        # ProtoNet-as-selector summary for the HUD: which-of-top-K was chosen by
        # image similarity, and how confident the match was.
        "protonet_select": (
            {
                "selected": bool(_l9_top0.get("protonet_selected")),
                "top_sim": _l9_top0.get("protonet_top_sim"),
                "select_k": _l9_top0.get("protonet_select_k"),
            }
            if (_l9_top0 := (result.get("per_level", {}).get(9) or [{}])[0]).get("protonet_selected")
            else None
        ),
        "country_prior_top": result.get("country_prior_top") or [],
        "country_prior_probs": result.get("country_prior_probs") or [],
        "country_head_used": bool(result.get("country_head_used")),
        "cascade": cascade,
        "country": final_geo.get("country_name"),
        "admin1": final_geo.get("admin1_name") if final_geo.get("admin1_name") != "NA" else None,
        "admin2": final_geo.get("admin2_name") if final_geo.get("admin2_name") != "NA" else None,
        "country_code": final_geo.get("country_code"),
        "per_level": per_level,
        "ckpt": _state.get("ckpt"),
    }

    # Stage 2 refinement: only run when cascade=refined AND we have the
    # raw image (refined called without an image silently falls through).
    if refined and original_image is not None:
        from geoai.stage2.refine import refine as stage2_refine

        s1_hub = (
            f"{resp.get('admin1') or ''}, {resp.get('country') or 'unknown'}"
        ).lstrip(", ")
        try:
            rr = stage2_refine(
                original_image,
                resp["final_lat"], resp["final_lng"], s1_hub,
                on_progress=on_stage2_progress,
                ocr_tiles=ocr_tiles,
            )
            resp["stage2_used"] = rr.used_refinement
            resp["stage2_source"] = rr.source
            resp["stage2_precision"] = rr.effective_precision
            resp["stage2_explanation"] = rr.explanation
            resp["stage2_seconds"] = rr.total_seconds
            resp["stage2_extract_seconds"] = rr.extract_seconds
            resp["stage2_pinpoint_seconds"] = rr.pinpoint_seconds
            resp["stage2_geocode_seconds"] = rr.geocode_seconds
            if rr.pinpoint:
                resp["stage2_queryable"] = rr.pinpoint.queryable_name
                resp["stage2_confidence"] = rr.pinpoint.confidence
            if rr.geocode_hit:
                resp["stage2_match_name"] = rr.geocode_hit.display_name
                resp["stage2_match_query"] = rr.geocode_hit.query
            # If Stage 2 committed, override top-level lat/lng for the pin
            # but preserve final_lat/final_lng as the Stage 1 result for
            # debugging.
            if rr.used_refinement:
                resp["stage1_lat"] = resp["final_lat"]
                resp["stage1_lng"] = resp["final_lng"]
                resp["lat"] = rr.final_lat
                resp["lng"] = rr.final_lng
                resp["final_lat"] = rr.final_lat
                resp["final_lng"] = rr.final_lng
        except Exception as e:
            log.exception("stage 2 refine failed: %r", e)
            resp["stage2_used"] = False
            resp["stage2_error"] = repr(e)

    return resp


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  mode: str = Form("auto"),
                  cascade: str = Form("country_only")):
    if not _state.get("model"):
        raise HTTPException(503, "Model not loaded yet")
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Failed to decode image: {e!r}")

    # Trim solid-black borders from missing edge tiles before further
    # processing. Apply on the equirect path; aspect threshold matches
    # _process_image's auto-mode detection.
    aspect = img.size[0] / img.size[1]
    is_equirect = mode == "panorama" or (mode == "auto" and 1.85 <= aspect <= 2.25)
    if is_equirect:
        img = crop_black_bars(img)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    final_bytes = buf.getvalue()
    saved = _save_pano_bytes(final_bytes, file.filename)
    if saved:
        log.info(f"saved pano to {saved} ({img.size[0]}x{img.size[1]}, cropped={is_equirect})")

    try:
        pixel_values, input_label = _process_image(final_bytes, mode)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to process image: {e!r}")

    c = (cascade or "").lower()
    cascade_clean = c if c in ("plain", "fancy", "joint", "refined") else "country_only"
    # Stage 2 needs TWO renderings of the same pano:
    #   - strip (3072x768): the VLM sees all 360° at once
    #   - tiles (8x 1024x1024, 50% overlap): Surya OCR with full per-view
    #     resolution and overlapping coverage so signs at cardinal seams
    #     are still captured in at least one tile
    # For already-perspective inputs (user-uploaded screenshots), we
    # pass the single image to both — no tiling possible.
    stage2_image: Optional[Image.Image] = None
    stage2_tiles: Optional[list] = None
    if cascade_clean == "refined":
        if is_equirect:
            from geoai.processing.perspective import (
                equirect_to_panorama_strip, equirect_to_perspective_tiles,
            )
            stage2_image = equirect_to_panorama_strip(img)
            stage2_tiles = equirect_to_perspective_tiles(img)
        else:
            stage2_image = img
    async with _state["model_lock"]:
        return JSONResponse(_build_response(
            pixel_values, input_label, cascade=cascade_clean,
            original_image=stage2_image,
            ocr_tiles=stage2_tiles,
        ))


class PanoIDRequest(BaseModel):
    """Body for /api/v1/predict — accepts both panoID (matches the userscript)
    and pano_id (snake-case) for convenience."""
    panoID: Optional[str] = None
    pano_id: Optional[str] = None
    # "fancy" (default) = full inference cascade. "plain" = vanilla V1-style
    # argmax-per-level. Anything else falls back to "fancy".
    cascade: Optional[str] = None

    def get_id(self) -> str:
        return self.panoID or self.pano_id or ""

    def get_cascade(self) -> str:
        c = (self.cascade or "").lower()
        if c in ("plain", "fancy", "joint", "refined"):
            return c
        return "country_only"  # default when unset or unrecognized


@app.get("/api/v1/info")
async def api_info():
    """Lightweight endpoint for the userscript to fetch model metadata at
    overlay-boot, before any prediction runs. Returns the active checkpoint
    path and parsed epoch number."""
    import re
    ckpt = _state.get("ckpt", "") or ""
    m = re.search(r'epoch_(\d+)', ckpt)
    return {
        "ckpt": ckpt,
        "epoch": int(m.group(1)) if m else None,
    }


def _family_short(dir_name: str) -> str:
    """'stage1_v3_long' → 'v3_long', 'stage1_v2' → 'v2', 'stage1' → 'v1'."""
    if dir_name == "stage1":
        return "v1"
    return dir_name.replace("stage1_", "", 1) or dir_name


@app.get("/api/v1/models")
async def api_models():
    """List checkpoints from the SAME model family as the one currently
    loaded. We scope to the loaded family (e.g. stage1_v3_long) because every
    epoch there shares the loaded cells vocab — swapping across families
    (v2↔v3) would mismatch the cell-head shapes and crash. Each entry carries
    a `family` + `label` so the UI can disambiguate (v3_long e03 ≠ v2 e03)."""
    import re
    from pathlib import Path
    ckpt = _state.get("ckpt", "") or ""
    base = Path(ckpt).parent if ckpt else None
    family = _family_short(base.name) if base else ""
    cur_resolved = str(Path(ckpt).resolve()) if ckpt else ""
    available = []
    if base and base.exists():
        for d in sorted(base.glob("epoch_*")):
            if (d / "model.safetensors").exists():
                m = re.search(r"epoch_(\d+)", d.name)
                ep = int(m.group(1)) if m else None
                available.append({
                    "id": d.name,                       # e.g. "epoch_10"
                    "epoch": ep,
                    "family": family,                   # e.g. "v3_long"
                    "label": (f"{family} e{ep:02d}" if ep is not None
                              else f"{family} {d.name}"),
                    "current": str(d.resolve()) == cur_resolved,
                    "path": str(d),
                })
    return {"current": ckpt, "family": family, "available": available}


class LoadModelRequest(BaseModel):
    """`epoch` accepts either "epoch_10" or just "10" / 10."""
    epoch: str | int


@app.post("/api/v1/models/load")
async def api_load_model(req: LoadModelRequest):
    """Swap the in-memory model to another epoch IN THE SAME FAMILY without
    restarting the server. Scoped to the loaded checkpoint's family dir so the
    new epoch shares the loaded cells vocab (cross-family swaps would crash on
    cell-head shape mismatch — restart with a new --ckpt/--cells-parquet for
    that). Uses the model_lock so any in-flight predict finishes first. ~5-10s.
    """
    from pathlib import Path
    cur = _state.get("ckpt", "") or ""
    if not cur:
        raise HTTPException(409, "no checkpoint loaded yet")
    base = Path(cur).parent.resolve()         # the loaded family dir
    raw = str(req.epoch).strip()
    name = raw if raw.startswith("epoch_") else f"epoch_{int(raw):02d}"
    target = (base / name).resolve()
    # Security: refuse anything that escapes the loaded family dir.
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(400, f"checkpoint must live under {base}")
    if not (target / "model.safetensors").exists():
        raise HTTPException(404, f"no model.safetensors at {target}")

    async with _state["model_lock"]:
        device = _state["device"]
        # Free the old model before loading the new one to avoid 2× GPU mem.
        old = _state.pop("model", None)
        if old is not None:
            del old
            if device.type == "cuda":
                torch.cuda.empty_cache()
        log.info(f"loading new checkpoint: {target}")
        new_model = load_checkpoint(target, _state["cells"], device).eval()
        _state["model"] = new_model
        _state["ckpt"] = str(target)
        # Re-attach the country vocab next to this ckpt (V2 epochs each ship
        # their own country_vocab.json — same 114 classes today, but could
        # differ if a post-init_v2 ckpt is loaded).
        _state["country_vocab"] = load_country_vocab_for_ckpt(target)
        log.info(f"  swapped to {target}")

    return {
        "status": "loaded",
        "ckpt": str(target),
        "epoch": int(name.removeprefix("epoch_")),
    }


@app.post("/api/v1/predict")
async def api_predict(req: PanoIDRequest):
    """Userscript-facing endpoint. Fetches the pano from Google by pano_id,
    stitches tiles, trims black bars, runs inference, returns {lat, lng, ...}.
    """
    pid = req.get_id().strip()
    if not pid:
        raise HTTPException(400, "panoID required in body")
    if not _state.get("model"):
        raise HTTPException(503, "Model not loaded yet")

    try:
        pano = await fetch_pano(pid)
        pano = crop_black_bars(pano)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch pano {pid}: {e!r}")

    buf = io.BytesIO()
    pano.save(buf, format="JPEG", quality=92)
    saved = _save_pano_bytes(buf.getvalue(), f"{pid}.jpg")
    if saved:
        log.info(f"saved pano to {saved}")

    # Round-trip through the existing equirect renderer (writes a temp file
    # because py360convert is path-based).
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        pano.save(f.name, quality=92)
        tmp_path = Path(f.name)
    try:
        pixel_values = render_crops_on_the_fly(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    label = f"pano_id ({pid}) → {pano.size[0]}×{pano.size[1]} equirect → 4 crops"
    cc = req.get_cascade()
    # Stage 2 needs both: panorama strip (for the VLM) and perspective
    # tiles (for Surya OCR — better recall than the strip).
    stage2_image: Optional[Image.Image] = None
    stage2_tiles: Optional[list] = None
    if cc == "refined":
        from geoai.processing.perspective import (
            equirect_to_panorama_strip, equirect_to_perspective_tiles,
        )
        stage2_image = equirect_to_panorama_strip(pano)
        stage2_tiles = equirect_to_perspective_tiles(pano)
    async with _state["model_lock"]:
        return JSONResponse(_build_response(
            pixel_values, label, pano_id=pid, cascade=cc,
            original_image=stage2_image,
            ocr_tiles=stage2_tiles,
        ))


@app.post("/api/v1/explain")
async def api_explain(req: PanoIDRequest):
    """Occlusion attribution: returns a PNG of the unwrapped pano with a
    heatmap of which regions drove the ProtoNet-selected location. Reuses the
    already-loaded model + index; needs ProtoNet (it's what we attribute to).
    """
    from geoai.stage1.explain import occlusion_explain

    pid = req.get_id().strip()
    if not pid:
        raise HTTPException(400, "panoID required in body")
    model = _state.get("model")
    protonet = _state.get("protonet")
    if model is None:
        raise HTTPException(503, "Model not loaded yet")
    if protonet is None:
        raise HTTPException(400, "ProtoNet index not loaded; /explain requires it")

    try:
        pano = await fetch_pano(pid)
        pano = crop_black_bars(pano)
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch pano {pid}: {e!r}")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        pano.save(f.name, quality=92)
        tmp_path = Path(f.name)
    try:
        pixel_values = render_crops_on_the_fly(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    async with _state["model_lock"]:
        png, meta = occlusion_explain(
            model, protonet, _state["device"], pixel_values, pano,
        )
    log.info(f"[explain] {pid[:8]}… cell={meta['cell']} sim={meta['sim']}")
    return Response(content=png, media_type="image/png", headers={
        "X-Explain-Cell": str(meta["cell"]),
        "X-Explain-Sim": str(meta["sim"]),
        "X-Explain-Lat": str(meta["lat"]),
        "X-Explain-Lng": str(meta["lng"]),
        "Access-Control-Expose-Headers":
            "X-Explain-Cell,X-Explain-Sim,X-Explain-Lat,X-Explain-Lng",
    })


@app.post("/api/v1/predict_stream")
async def api_predict_stream(req: PanoIDRequest):
    """Streaming version of /api/v1/predict. Emits one JSON event per line
    (NDJSON, `application/x-ndjson`) as the pipeline progresses so the
    userscript can show a live "Stage 1 → OCR → VLM → Geocode → Done"
    status indicator instead of staring at a spinner for 30 s.

    Event sequence:
        fetch_pano_start    fetched from Google
        fetch_pano_done     {width, height, seconds}
        stage1_start
        stage1_done         {lat, lng, country, admin1}
        extract_start       Stage 2 OCR begins (refined cascade only)
        extract_done        {script, raw_text, translation, seconds}
        pinpoint_start      VLM begins
        pinpoint_done       {precision, confidence, queryable, seconds}
        geocode_start       {queries, country_code}    (only if VLM trusts itself)
        geocode_done        {hit, query, lat, lng, seconds}
        done                full final response dict

    Final non-refined cascades just emit stage1_done then done.
    """
    import asyncio
    import json as _json
    import threading
    main_thread_id = threading.get_ident()

    pid = req.get_id().strip()
    if not pid:
        raise HTTPException(400, "panoID required in body")
    if not _state.get("model"):
        raise HTTPException(503, "Model not loaded yet")
    cc = req.get_cascade()
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _json_default(o):
        """Coerce numpy / torch / Path / set / bytes etc into JSON-safe
        primitives so emit() never blows up on a stray non-Python type."""
        try:
            import numpy as _np
            if isinstance(o, _np.generic):
                return o.item()
        except ImportError:
            pass
        try:
            import torch as _torch
            if isinstance(o, _torch.Tensor):
                return o.detach().cpu().tolist()
        except ImportError:
            pass
        if hasattr(o, "tolist"):
            return o.tolist()
        if hasattr(o, "__fspath__"):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return list(o)
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        return repr(o)

    def emit(event: str, data: Optional[dict] = None) -> None:
        """Thread-safe push of a (event, data) pair into the async queue.
        Uses a tolerant JSON encoder so numpy/torch types don't blow up
        the whole pipeline — the userscript would otherwise see only
        'stream ended without done event'."""
        try:
            msg = _json.dumps(
                {"event": event, "data": data or {}},
                default=_json_default,
            ) + "\n"
        except Exception as e:
            log.exception("emit(%r) failed to serialize: %r", event, e)
            msg = _json.dumps(
                {"event": "emit_error",
                 "data": {"original_event": event, "error": repr(e)}}
            ) + "\n"
        # From the loop's own thread, put_nowait is synchronous — the item
        # is in the queue before this function returns. From a worker thread
        # (where refine() runs via to_thread), we must marshal via
        # run_coroutine_threadsafe. Picking the wrong one when called from
        # the loop thread races the finally-block's put(None) and silently
        # drops events.
        if threading.get_ident() == main_thread_id:
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # shouldn't happen — queue has no maxsize
        else:
            try:
                asyncio.run_coroutine_threadsafe(queue.put(msg), loop)
            except RuntimeError:
                pass  # loop closed (client disconnected)

    async def pipeline() -> None:
        """Run the full pipeline, pushing events at each boundary."""
        try:
            # 1) Fetch the pano from Google.
            emit("fetch_pano_start", {"pano_id": pid})
            t = time.time()
            pano = await fetch_pano(pid)
            pano = crop_black_bars(pano)
            emit("fetch_pano_done", {
                "width": pano.size[0], "height": pano.size[1],
                "seconds": time.time() - t,
            })
            saved = _save_pano_bytes(
                (lambda: (b := io.BytesIO(), pano.save(b, format="JPEG", quality=92), b.getvalue())[2])(),
                f"{pid}.jpg",
            )
            if saved:
                log.info(f"saved pano to {saved}")

            # 2) Stage 1 inference.
            emit("stage1_start", {"cascade": cc})
            t = time.time()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                pano.save(f.name, quality=92)
                tmp_path = Path(f.name)
            try:
                pixel_values = render_crops_on_the_fly(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

            stage2_image: Optional[Image.Image] = None
            stage2_tiles: Optional[list] = None
            if cc == "refined":
                from geoai.processing.perspective import (
                    equirect_to_panorama_strip,
                    equirect_to_perspective_tiles,
                )
                stage2_image = equirect_to_panorama_strip(pano)
                stage2_tiles = equirect_to_perspective_tiles(pano)

            label = f"pano_id ({pid}) → {pano.size[0]}×{pano.size[1]} equirect → 4 crops"
            # Run Stage 1 (and, for refined, ALSO Stage 2 inside _build_response)
            # in a worker thread so we don't block the event loop. We hook
            # Stage 2's on_progress to feed our emit() queue.
            def _do_work() -> dict:
                return _build_response(
                    pixel_values, label, pano_id=pid, cascade=cc,
                    original_image=stage2_image,
                    ocr_tiles=stage2_tiles,
                    on_stage2_progress=(lambda ev, d: emit("stage2_" + ev, d)),
                )

            log.info(f"predict_stream({pid}): acquiring model_lock")
            async with _state["model_lock"]:
                log.info(f"predict_stream({pid}): lock acquired, dispatching _do_work to thread")
                resp = await asyncio.to_thread(_do_work)
                log.info(f"predict_stream({pid}): _do_work returned, keys={list(resp.keys())[:6]}")
            log.info(f"predict_stream({pid}): lock released, emitting stage1_done")
            emit("stage1_done", {
                "lat": resp.get("stage1_lat", resp.get("final_lat")),
                "lng": resp.get("stage1_lng", resp.get("final_lng")),
                "country": resp.get("country"),
                "admin1": resp.get("admin1"),
                "seconds": time.time() - t,
            })

            # 3) The full resp dict — already merged Stage 1 + Stage 2 fields.
            emit("done", resp)
        except Exception as e:
            log.exception("predict_stream pipeline error")
            emit("error", {"error": repr(e)})
        finally:
            await queue.put(None)  # sentinel for the generator

    async def heartbeat():
        """Emit a noop event every few seconds so the consumer's HTTP
        connection isn't silent for >30 s during VLM thinking. Chrome's
        MV3 service worker kills GM_xmlhttpRequest after ~30 s of stream
        inactivity ("background shutdown"), losing the result. Heartbeats
        keep bytes flowing on the wire so the SW stays alive."""
        try:
            while True:
                await asyncio.sleep(5.0)
                # ts is for client-side diagnostics; userscript ignores
                # the event itself but reading it keeps the SW awake.
                emit("heartbeat", {"ts": time.time()})
        except asyncio.CancelledError:
            return

    async def event_generator():
        # Kick the pipeline off as a task so it runs concurrently with our
        # queue-drain loop. Heartbeats run in parallel and are cancelled
        # the moment the pipeline drops its sentinel.
        pipe_task = asyncio.create_task(pipeline())
        hb_task = asyncio.create_task(heartbeat())
        try:
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield msg
            hb_task.cancel()
            await pipe_task  # surface any exception
        except asyncio.CancelledError:
            hb_task.cancel()
            pipe_task.cancel()
            raise
        finally:
            # Belt-and-suspenders cancel in case we exit through some
            # other path (e.g. client disconnects mid-stream).
            if not hb_task.done():
                hb_task.cancel()

    return StreamingResponse(
        event_generator(), media_type="application/x-ndjson",
    )


@app.get("/api/v1/reverse-geocode")
async def api_reverse_geocode(lat: float, lng: float):
    """Cheap in-memory GADM lookup for the userscript's per-country stats.
    Returns the country code (ISO alpha-3) and human-readable admin chain
    for an arbitrary lat/lng — used to bucket round results by *truth*
    country, not predicted country."""
    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        raise HTTPException(400, "lat/lng out of range")
    geo = reverse_geo.reverse_geocode(lat, lng)
    return {
        "lat": lat, "lng": lng,
        "country":      geo.get("country"),
        "country_code": geo.get("country_code"),
        "admin1":       geo.get("admin1"),
        "admin2":       geo.get("admin2"),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return _PAGE_HTML


_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>geoai — geolocation inference</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
      crossorigin="" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
        crossorigin=""></script>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    max-width: 880px; margin: 2rem auto; padding: 0 1rem;
    color: #1a1a1a; background: #fafaf9;
  }
  h1 { font-size: 1.4rem; margin-bottom: 0.25rem; }
  p.sub { color: #666; margin-top: 0; font-size: 0.95rem; }
  .drop-zone {
    border: 2px dashed #cbd5e1; border-radius: 0.5rem;
    padding: 3.5rem 2rem; text-align: center; cursor: pointer;
    background: #fff; transition: all 0.15s;
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: #3b82f6; background: #eff6ff;
  }
  .drop-zone p { margin: 0; color: #666; }
  .drop-zone strong { color: #1e293b; }
  .drop-zone input[type=file] { display: none; }
  .mode-row {
    display: flex; gap: 1rem; margin-bottom: 0.75rem; align-items: center;
    background: #fff; padding: 0.65rem 0.9rem; border-radius: 0.4rem;
    box-shadow: 0 1px 2px rgba(0,0,0,.05); font-size: 0.9rem;
  }
  .mode-row label { display: inline-flex; align-items: center; gap: 0.35rem; cursor: pointer; }
  .mode-row span.lab { color: #475569; font-weight: 500; margin-right: 0.4rem; }
  .mode-row input[type=radio] { margin: 0; cursor: pointer; }
  .pano-row {
    display: flex; gap: 0.5rem; margin-top: 0.75rem; margin-bottom: 0.5rem;
    background: #fff; padding: 0.65rem 0.9rem; border-radius: 0.4rem;
    box-shadow: 0 1px 2px rgba(0,0,0,.05);
  }
  .pano-row input[type=text] {
    flex: 1; padding: 0.45rem 0.6rem; border: 1px solid #cbd5e1;
    border-radius: 0.3rem; font-family: ui-monospace, monospace; font-size: 0.88rem;
  }
  .pano-row button {
    padding: 0.45rem 1rem; border: 0; border-radius: 0.3rem;
    background: #3b82f6; color: #fff; cursor: pointer; font-weight: 500;
  }
  .pano-row button:hover { background: #2563eb; }
  .or-divider {
    text-align: center; color: #94a3b8; font-size: 0.85rem;
    margin: 0.5rem 0; text-transform: uppercase; letter-spacing: 0.05em;
  }
  .preview {
    max-width: 100%; max-height: 400px; margin-top: 1rem;
    border-radius: 0.375rem; box-shadow: 0 1px 3px rgba(0,0,0,.1);
  }
  .result { margin-top: 1.5rem; }
  .card {
    background: #fff; padding: 1rem 1.25rem;
    border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.08);
    margin-bottom: 1rem;
  }
  .card h2 { margin: 0 0 0.5rem; font-size: 1.05rem; }
  .latlng { font-size: 1.4rem; font-weight: 600; color: #1e40af; font-family: ui-monospace, monospace; }
  .where { color: #475569; font-size: 1rem; margin-top: 0.25rem; }
  .badge {
    display: inline-block; padding: 0.15rem 0.55rem; margin-left: 0.5rem;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600; vertical-align: middle;
  }
  .badge.fb { background: #fef3c7; color: #92400e; }
  .badge.l12 { background: #dbeafe; color: #1e40af; }
  .map-link { display: inline-block; margin-top: 0.5rem; color: #3b82f6; text-decoration: none; }
  .map-link:hover { text-decoration: underline; }
  .meta { color: #94a3b8; font-size: 0.82rem; margin-top: 0.4rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th, td { padding: 0.4rem 0.5rem; text-align: left; }
  th { color: #64748b; font-weight: 500; border-bottom: 1px solid #e2e8f0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }
  tr:not(:last-child) td { border-bottom: 1px solid #f1f5f9; }
  td.num { font-family: ui-monospace, monospace; color: #475569; }
  .lvl { color: #64748b; font-size: 0.85rem; margin-bottom: 0.25rem; }
  .loading { color: #64748b; font-style: italic; }
  .err { color: #dc2626; }
  #map {
    height: 380px; width: 100%; border-radius: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-top: 0.5rem;
  }
</style>
</head>
<body>
<h1>geoai — Stage 1 inference</h1>
<p class="sub">Drop an image. Equirectangular panos (2:1) are rendered to 4 crops as the model expects. Anything else is tiled 4× (degraded but works).</p>

<div class="mode-row">
  <span class="lab">Image type:</span>
  <label><input type="radio" name="mode" value="auto" checked> Auto-detect</label>
  <label><input type="radio" name="mode" value="panorama"> Panorama (equirectangular)</label>
  <label><input type="radio" name="mode" value="single"> Single image</label>
</div>

<div class="pano-row">
  <input type="text" id="panoIdInput" placeholder="paste a Street View panoID (e.g. 7cLZqEp3eN5bvQdvH4UT8g) — fetches & predicts">
  <button id="panoBtn">Fetch & predict</button>
</div>
<div class="or-divider">— or upload an image —</div>

<div class="drop-zone" id="dropZone">
  <input type="file" id="fileInput" accept="image/*">
  <p><strong>Drag and drop</strong>, click to choose, or <strong>paste</strong> (⌘/Ctrl-V) an image</p>
</div>

<img id="preview" class="preview" style="display:none">
<div id="result" class="result"></div>

<script>
const dz = document.getElementById('dropZone');
const fi = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

dz.addEventListener('click', () => fi.click());
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('dragover');
  if (e.dataTransfer.files.length) handle(e.dataTransfer.files[0]);
});
fi.addEventListener('change', () => fi.files.length && handle(fi.files[0]));

// Pano-id fetch path
const panoBtn = document.getElementById('panoBtn');
const panoInput = document.getElementById('panoIdInput');
panoBtn.addEventListener('click', () => {
  const id = panoInput.value.trim();
  if (id) handlePano(id);
});
panoInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); panoBtn.click(); }
});

async function handlePano(panoId) {
  preview.style.display = 'none';
  result.innerHTML = '<div class="card"><span class="loading">Fetching pano ' + panoId + '…</span></div>';
  try {
    const r = await fetch('/api/v1/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ panoID: panoId }),
    });
    if (!r.ok) throw new Error(await r.text());
    render(await r.json());
  } catch (e) {
    result.innerHTML = '<div class="card err">Error: ' + e.message + '</div>';
  }
}

// Paste anywhere on the page: grab the first image item from the clipboard
// and feed it through the same handle() flow.
document.addEventListener('paste', e => {
  const items = (e.clipboardData || window.clipboardData)?.items || [];
  for (const it of items) {
    if (it.type && it.type.startsWith('image/')) {
      const file = it.getAsFile();
      if (file) {
        e.preventDefault();
        handle(file);
        return;
      }
    }
  }
});

async function handle(file) {
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  result.innerHTML = '<div class="card"><span class="loading">Analyzing…</span></div>';
  const mode = document.querySelector('input[name=mode]:checked').value;
  const fd = new FormData();
  fd.append('file', file);
  fd.append('mode', mode);
  try {
    const r = await fetch('/predict', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await r.text());
    render(await r.json());
  } catch (e) {
    result.innerHTML = '<div class="card err">Error: ' + e.message + '</div>';
  }
}

function render(d) {
  const lat = d.final_lat.toFixed(4), lng = d.final_lng.toFixed(4);
  const where = [d.country, d.admin1, d.admin2].filter(x => x).join(' / ') || '(unknown)';
  const fb = d.fallback_used
    ? '<span class="badge fb">L9 fallback</span>'
    : '<span class="badge l12">L12</span>';

  let html = '<div class="card">';
  html += '<h2>Prediction' + fb + '</h2>';
  html += '<div class="latlng">' + lat + ', ' + lng + '</div>';
  html += '<div class="where">' + where + '</div>';
  html += '<a class="map-link" href="https://www.google.com/maps/@' + lat + ',' + lng + ',8z" target="_blank">open in google maps →</a>';
  html += '<div class="meta">input: ' + d.input_type + ' · ckpt: ' + (d.ckpt || '?').split('/').pop() + '</div>';
  html += '<div id="map"></div>';
  html += '</div>';

  const lvlNames = {3: 'L3 (~1300 km)', 6: 'L6 (~165 km)', 9: 'L9 (~20 km)', 12: 'L12 (~2.5 km)'};
  for (const lvl of [3, 6, 9, 12]) {
    const cands = d.per_level[lvl];
    if (!cands) continue;
    html += '<div class="card">';
    html += '<div class="lvl">' + lvlNames[lvl] + '</div>';
    html += '<table><thead><tr><th>p</th><th>lat</th><th>lng</th><th>country</th></tr></thead><tbody>';
    for (const c of cands) {
      html += '<tr>';
      html += '<td class="num">' + c.prob.toFixed(3) + '</td>';
      html += '<td class="num">' + c.lat.toFixed(3) + '</td>';
      html += '<td class="num">' + c.lng.toFixed(3) + '</td>';
      html += '<td>' + c.country_code + '</td>';
      html += '</tr>';
    }
    html += '</tbody></table></div>';
  }
  result.innerHTML = html;

  // Drop a red pin on the predicted location. Since #map is rebuilt with the
  // result HTML each time, we just spin up a fresh Leaflet instance per render.
  const mapEl = document.getElementById('map');
  if (mapEl) {
    const m = L.map('map').setView([d.final_lat, d.final_lng], 7);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors', maxZoom: 19,
    }).addTo(m);
    const redPin = L.divIcon({
      className: 'red-pin',
      html: '<svg width="28" height="40" viewBox="0 0 28 40" xmlns="http://www.w3.org/2000/svg">' +
        '<path d="M14 0C6.3 0 0 6.3 0 14c0 10.5 14 26 14 26s14-15.5 14-26C28 6.3 21.7 0 14 0z" ' +
        'fill="#dc2626" stroke="#7f1d1d" stroke-width="1"/>' +
        '<circle cx="14" cy="14" r="5.5" fill="#fff"/></svg>',
      iconSize: [28, 40], iconAnchor: [14, 40], popupAnchor: [0, -36],
    });
    L.marker([d.final_lat, d.final_lng], { icon: redPin })
      .addTo(m)
      .bindPopup(d.final_lat.toFixed(4) + ', ' + d.final_lng.toFixed(4));
  }
}
</script>
</body>
</html>
"""


cli = typer.Typer(add_completion=False)


@cli.command()
def main(
    ckpt: Path = typer.Option(DEFAULT_CKPT, exists=True, help="Stage 1 checkpoint dir"),
    device: str = typer.Option(DEFAULT_DEVICE),
    host: str = typer.Option("0.0.0.0", help="0.0.0.0 binds to all interfaces (LAN-accessible)"),
    port: int = typer.Option(6301, help="Default 6301 matches the userscript's expected port"),
    protonet_path: Optional[Path] = typer.Option(
        None,
        help="Path to a ProtoNet L9 index (.pt). Defaults to <ckpt>/protonet_l9.pt if it exists. "
             "Pass empty string '' to disable explicitly.",
    ),
    cells_parquet: Optional[Path] = typer.Option(
        None,
        help="Path to cells.parquet for the cell vocab. Defaults to PROCESSED_DIR/cells.parquet. "
             "Set to a V1-vocab backup (e.g. cells_v1.parquet) when serving a V1 checkpoint "
             "after the default cells.parquet has been overwritten for V2 prep.",
    ),
) -> None:
    import os
    os.environ["GEOAI_CKPT"] = str(ckpt)
    os.environ["GEOAI_DEVICE"] = device
    if protonet_path is not None:
        os.environ["GEOAI_PROTONET_PATH"] = str(protonet_path)
    if cells_parquet is not None:
        os.environ["GEOAI_CELLS_PARQUET"] = str(cells_parquet)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import uvicorn
    uvicorn.run("geoai.serve.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    cli()
