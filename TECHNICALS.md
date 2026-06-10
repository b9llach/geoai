# geoai — Technical Walkthrough

How the system works, component by component: what each piece is, *why* it exists,
and *how* it does its job. This is the code-oriented companion to `README.md`
(overview + results) and [`paper/`](paper/) (the research report: the diagnostics
and the selection finding in depth).

---

## 1. The problem

Given a single Street View panorama, predict where on Earth it was taken. Direct
latitude/longitude regression fails — the loss surface is full of bad minima (the
model learns to predict mid-ocean to minimize average error). The standard answer,
which we follow, is **hierarchical classification**: chop the world into S2 cells at
several resolutions, classify the cell at each resolution, then convert the chosen
cell to a coordinate.

Two stages:

- **Stage 1 (vision, ~100 ms):** SigLIP2 encoder + hierarchical S2-cell heads +
  a ProtoNet content selector. This is the production system.
- **Stage 2 (language, seconds, optional):** a vision-language model that reads
  on-image text and reasons to pinpoint *within* Stage 1's region.

Everything is local, open-weights, zero recurring API cost (a hard constraint).

---

## 2. Data corpus and pipeline

The training set is a pre-existing corpus at `/data/geolocation/`: ~2.5M
equirectangular Street View panoramas globally (≈2.39M train / 75K held-out), with
a SQLite index (`metadata.db`) mapping each pano to its `(lat, lng)`, S2 cells, GADM
country, and climate/population features.

Two image forms matter:

- **Equirectangular panorama** — the full 360°×180° sphere flattened to one image.
  The raw input.
- **Perspective crops** (`384×384`) — the equirect un-warped into 4 flat photos
  looking N/E/S/W (90° FOV). What the model actually sees, because the backbone was
  pretrained on normal photos, not warped spheres.

Four resumable CLI stages turn raw panos into trainable rows:

| Stage | CLI | What it does |
|---|---|---|
| Catalog | `geoai-catalog` | GADM point-in-polygon → country; sample S2 cells L3/6/9/12; sample Köppen + GHS-POP rasters → `metadata.db` |
| Render crops | `geoai-render-crops` | equirect → 4× `384²` perspective JPEGs, sharded `crops/<id[:2]>/` |
| Split | `geoai-split` | deterministic held-out split, keyed on **location** not pano |
| Cell-stats | `geoai-cell-stats` | count panos per cell, prune rare cells, write `cells_vN.parquet` (the output vocabulary) |

Each is idempotent (`INSERT OR IGNORE` + on-disk dedupe), so kill/restart is safe.

---

## 3. S2 cells: the hierarchical vocabulary

[S2 geometry](http://s2geometry.io/) recursively subdivides the sphere; "level N" =
N subdivisions = smaller cell. We use four levels:

| Level | Avg edge | Intuition |
|---|---|---|
| L3 | ~1150 km | continental region |
| L6 | ~144 km | country / large state |
| L9 | ~18 km | city / metro |
| L12 | ~2.3 km | neighborhood |

Each pano belongs to one nested cell per level. We keep only cells with enough
support (`min_count = {L3:1, L6:1, L9:2, L12:5}`); the current `cells_v3` vocab is
175 / 4024 / 119,026 / 45,946 cells. At inference a chosen cell maps to its
**centroid** (or a retrieval-refined point — §7).

---

## 4. Stage 1 architecture

File: `geoai/stage1/model.py`. The forward pass:

```
4 perspective views [B, 4, 3, 384, 384]
        ▼  SigLIP2 vision tower (so400m), run 4× per pano
   4 × [B, 1152]                          ← one embedding per view
        ▼  concat across views            [B, 4608]
        ▼  feat_proj: Linear→1024, GELU, LayerNorm   [B, 1024]  "pooled features"
        │
   L3 head  Linear(1024 → 175)   → logits[L3];  embed(argmax) → 256-d
   L6 head  Linear(1024+256 → 4024)   → logits[L6];  embed → 256-d
   L9 head  Linear(1024+256 → 119026) → logits[L9];  embed → 256-d
   L12 head Linear(1024+256 → 45946)  → logits[L12]
        └─  country head  Linear(1024 → 135)   (auxiliary)
```

Design choices and why:

- **Concat 4 views, not mean-pool** — preserves *which direction* evidence came
  from. Heading-invariance is recovered by randomly rotating the heading order
  during training augmentation.
- **Autoregressive heads (GeoToken-style)** — each finer head sees the pooled
  features plus a 256-d embedding of the coarser prediction. Teacher-forced on
  ground-truth cells in training, argmax at inference. Lets fine heads condition on
  coarse context.
- **Auxiliary country head** — reads the same pooled features (nearly free) and
  predicts the country directly, sharpening visually-similar-country boundaries; at
  inference it can mask cells in the wrong country (not used in the deployed path).
- **`encode_views_unconcat` is split out on purpose** — the SigLIP forward runs
  `B×4` times; caching per-view embeddings lets test-time augmentation re-pool under
  heading shifts without re-encoding.

The backbone is **full-finetuned**. **SigLIP2 normalization is mean/std = 0.5/0.5**,
not ImageNet's values — watch for this in any new dataset code.

---

## 5. Loss

File: `geoai/stage1/loss.py`. Three terms:

1. **Haversine-smoothed soft cross-entropy, per level.** The target is a *soft*
   distribution over cells, decaying with great-circle distance from the true cell,
   bandwidth `sigma = {L3:2000, L6:500, L9:100, L12:20} km`. Predicting a nearby
   cell is *almost* right; a far cell is punished hard.
2. **Country cross-entropy** (weight 0.3).
3. **Border penalty** — extra cost when the predicted cell's country ≠ the truth's,
   attacking cross-border look-alike confusion.

---

## 6. Training

`geoai/stage1/train.py`, launched via `accelerate` on 2× RTX 4090: per-GPU batch 4
× grad-accum → effective ~256, AdamW lr 2e-5, cosine schedule + 5% warmup, bf16,
grad-clip 1.0, gradient checkpointing, `torch.compile`. A `--init-from` warm-start
loads a prior checkpoint's backbone + heads (remapping by cell-id / country-code
when the vocab grew) with a fresh optimizer/schedule. The deployed checkpoint is
`stage1_v3_long/epoch_04`.

---

## 7. Inference: ProtoNet content-selection (the key piece)

Raw argmax-per-level → centroid is the floor, and it is badly limited: the model
gets the *region* right but rarely the exact L9 cell, because its probability is
biased toward data-dense cells and cannot rank the true cell first. (The full
diagnosis — oracle floors, recall@K, the selection ceiling — is in `paper/`.)

The fix is a **content-aware selector**. ProtoNet is a prebuilt L9 feature index:
for each cell we store the pooled embeddings ("prototypes") of its training panos
(`cells_v3` index: 1.94M prototypes across 119,026 cells, avg ~16/cell). At
inference (`geoai/stage1/predict.py`, `protonet_select_topk`):

1. Take the top-$K$ L9 candidates from the raw logits (default $K{=}500$).
2. For each, compute the cosine similarity of the query's pooled features to that
   cell's nearest prototype.
3. **Select the highest-similarity cell** — picking by *image match*, not the
   density-biased probability — and use its retrieval-refined coordinate.

This cuts median error from ~73 km to ~25 km and roughly doubles within-25 km, with
no retraining. It also *simplifies* inference: candidate-masking "cascade modes"
all hurt once selection is active (they strip truth-near cells before the selector
can pick them), so **Stage 1 is a single path: raw logits → ProtoNet-select**.

The index stores normalized features in fp16 (~4 GB) so it fits alongside the model
on a 24 GB GPU; the server pre-warms the normalization at startup (`geoai-serve`).

Build it with `scripts/build_protonet_index.py` (supports `--shard/--num-shards`
for dual-GPU encoding, merged by `scripts/merge_protonet_shards.py`).

---

## 8. Stage 2 — optional reasoning layer

`geoai/stage2/`. For text-rich panoramas, `refined` mode runs:

```
equirect ─► perspective tiles ─► Surya OCR (90+ scripts) ─► fastText langid
         ─► NLLB-200 translation ─► Gemma 4 26B (LM Studio, OpenAI-compatible)
         ─► Nominatim geocode ─► reverse-geocode country sanity check
```

Gemma reasons over the *pre-extracted* OCR text plus the image (feeding it raw
pixels to "read" sends thinking-mode into a token spiral; pre-extracted text keeps
it bounded). It overrides Stage 1 **only** when it resolves a specific OSM *point*
feature near Stage 1's guess — area centroids (a city/region) and far-drift matches
are rejected, so a bare "Geneva" never pulls a rural guess to the city center.

Its chief value is **visually-homogeneous countries**. Japan, for example, has
ample coverage (68K panos) and ProtoNet finds near-perfect visual matches
(similarity ~0.98), yet still errs ~131 km — convenience stores, suburban housing,
and road markings look identical nationwide, so the pixels are genuinely ambiguous
and only on-image text (prefecture names, area codes) disambiguates. (Per-country
analysis is in `paper/`.)

---

## 9. Serving

`geoai/serve/api.py` (`geoai-serve`, FastAPI, port 6301) loads the model + ProtoNet
index + GADM once and answers predictions. Key endpoints:

- `POST /api/v1/predict` / `POST /predict` — fetch the equirect by pano ID (or
  accept an upload), render crops, run raw → ProtoNet-select, return lat/lng +
  per-level cells + reverse-geocoded country.
- `POST /api/v1/predict_stream` — NDJSON with progress events (for the Stage 2 UI).
- `GET /api/v1/info`, `GET /api/v1/models`, `POST /api/v1/models/load` — checkpoint
  metadata + same-family hot-swap.

The `script.js` Tampermonkey userscript plays live on geoguessr.com against the
server, with two modes: **ProtoNet (Stage 1)** and **+ Stage 2 (OCR/VLM)**.

---

## 10. Non-obvious gotchas

Things that have bitten us — preserve them.

- **S2 cell IDs are stored as TEXT, not INTEGER.** S2 cells are 64-bit *unsigned*;
  SQLite INTEGER is signed 64-bit, so high-bit cells overflow on insert. Cast back
  with `int(cell_str)` in dataset code.
- **SigLIP2 normalization is mean/std = 0.5/0.5**, not ImageNet's 0.485/0.456/0.406.
- **Held-out splits are at the location level**, not the pano level — a pano's
  temporal siblings must all land in the same split or the test set leaks into
  training.
- **Use the `custom/` raster copies.** The top-level `Beck_KG_*conf*.tif` is a
  *confidence* map, not the class map; the top-level GHS-POP is root-owned mode 700.
- **GHS-POP is in ESRI:54009 (Mollweide), not WGS84** — lookups reproject
  `(lng, lat)` before sampling (`rasters.population_density` handles it).
- **Köppen group letters: 14/15/16 (Cfa/Cfb/Cfc) are group C, not D.** Group D
  starts at 17. An off-by-2 mislabels every temperate-marine pano (London, PNW, NZ).
- **GADM 4.1 has ~14 countries with no ADM_2 subdivision** (Qatar, Singapore,
  Puerto Rico, the Euro microstates, …); an ADM_2-only point-in-polygon lookup
  returns `country_code=NULL` for them. `reverse_geo.py` chains ADM_2 → ADM_1 →
  ADM_0 plus bbox overrides for HK/Macau.
- **Soft-blocks, not deprecation.** `cbk0.google.com/cbk` and the tile endpoints
  work fine; "endpoint deprecated"-looking failures are IP-level soft-blocks from
  bursty parallel requests, and clear in minutes.

---

## 11. Glossary

- **Equirectangular** — the 360°×180° sphere flattened to one rectangular image.
- **Perspective crop** — a flat photo un-warped from the equirect at a heading/FOV.
- **S2 cell** — a cell in Google's hierarchical sphere grid; the prediction target.
- **Centroid** — the lat/lng center of a cell; the default coordinate for a cell.
- **Autoregressive head / GeoToken** — finer cell heads conditioned on an embedding
  of the coarser head's prediction.
- **ProtoNet selector** — picking the final L9 cell by image-feature similarity to
  stored prototypes, over the top-K candidates, instead of by model probability.
