"""Build the ProtoNet feature index for L9 cells.

For every train pano whose L9 cell is in the vocab, run the trained model's
encoder + projection to produce a [proj_dim] feature vector. Group the
features by L9 cell index. Save as a single torch file consumed at
inference by `geoai.stage1.protonet.ProtoNetIndex`.

Usage:
    .venv/bin/python scripts/build_protonet_index.py \\
        --ckpt /data/geolocation/processed/checkpoints/stage1/epoch_13 \\
        --out  /data/geolocation/processed/checkpoints/stage1/epoch_13/protonet_l9.pt \\
        --batch-size 16 --num-workers 4 --device cuda:0

Optional flags:
    --max-per-cell N   cap each L9 cell's prototypes at N (saves memory).
                       0 = no cap. Sensible defaults: 50-100.
    --subset N         build from a random N-pano subset (smoke testing).
                       0 = full train split.

Runtime estimate on the V1 epoch_13 checkpoint:
    ~870k qualifying L9 panos × ~20 ms/pano ≈ 5 hours single-GPU.
    Cap at --max-per-cell 50 to drop ~70% of them in dense cells: ~1-2h.
"""
from __future__ import annotations

import logging
import random
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import typer
from torch.utils.data import DataLoader

from geoai.config import METADATA_DB, PROCESSED_DIR
from geoai.stage1.cells import PRUNED_LABEL, CellVocab
from geoai.stage1.dataset import PanoDataset, collate
from geoai.stage1.predict import load_checkpoint
from geoai.stage1.protonet import ProtoNetIndex

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


@app.command()
def main(
    ckpt: Path = typer.Option(..., exists=True, help="Stage 1 checkpoint dir"),
    out: Path = typer.Option(..., help="Output .pt path for the ProtoNet index"),
    level: int = typer.Option(9, help="S2 level for the index. Currently only L9 is recommended."),
    batch_size: int = typer.Option(16, help="Pano batch size for the encoder pass"),
    num_workers: int = typer.Option(4),
    device: str = typer.Option("cuda:0"),
    db_path: Path = typer.Option(METADATA_DB),
    max_per_cell: int = typer.Option(0, help="Cap prototypes per cell (0=no cap)"),
    subset: int = typer.Option(0, help="Random N-pano subset for smoke testing (0=full)"),
    shard: int = typer.Option(0, help="This shard's index (0-based). For multi-GPU encoding."),
    num_shards: int = typer.Option(1, help="Total shards. >1 saves a RAW partial (not an index); merge with merge_protonet_shards.py."),
    seed: int = typer.Option(42),
    feature_dtype: str = typer.Option("float16", help="float16|float32 storage dtype"),
    cells_parquet: Optional[Path] = typer.Option(
        None,
        help="Path to cells.parquet for the cell vocab. Defaults to PROCESSED_DIR/cells.parquet. "
             "Must match the vocab the model was trained against.",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if cells_parquet is not None:
        cells = CellVocab.from_parquet(path=cells_parquet)
    else:
        cells = CellVocab.from_parquet()
    log.info(f"vocab: {cells} (from {cells_parquet or 'default'})")
    dev = torch.device(device)

    model = load_checkpoint(ckpt, cells, dev)
    model.eval()
    log.info(f"loaded checkpoint: {ckpt}")

    # ---- candidate set: train panos whose L{level} cell is in vocab ---------
    ds = PanoDataset(cells, split="train", augment=False)
    log.info(f"train dataset: {len(ds):,} panos")

    # Filter rows down to those with a valid (vocab-qualifying) L{level} cell.
    # The dataset already yields cell_indices; we pre-compute here on rows to
    # also enforce per-cell caps without having to load images first.
    keep_idx: list[int] = []
    cell_idx_per_keep: list[int] = []
    per_cell_count: dict[int, int] = defaultdict(int)
    for i, row in enumerate(ds.rows):
        cidx = cells.index(level, row[f"s2_l{level}"])
        if cidx == PRUNED_LABEL:
            continue
        if max_per_cell > 0 and per_cell_count[cidx] >= max_per_cell:
            continue
        keep_idx.append(i)
        cell_idx_per_keep.append(cidx)
        per_cell_count[cidx] += 1

    if subset > 0 and subset < len(keep_idx):
        rng = random.Random(seed)
        sample_pos = rng.sample(range(len(keep_idx)), subset)
        keep_idx = [keep_idx[p] for p in sample_pos]
        cell_idx_per_keep = [cell_idx_per_keep[p] for p in sample_pos]

    log.info(
        f"selected {len(keep_idx):,} panos with valid L{level} cells "
        f"({len(set(cell_idx_per_keep)):,} unique cells)"
    )
    if max_per_cell > 0:
        log.info(f"per-cell cap: {max_per_cell}")

    # Multi-GPU sharding: the selection above is deterministic, so every shard
    # agrees on the full prototype set. Each shard encodes a DISJOINT strided
    # slice; merge_protonet_shards.py concatenates them (the index is an
    # order-independent per-cell bag, so a strided split + concat is exact).
    if num_shards > 1:
        keep_idx = keep_idx[shard::num_shards]
        cell_idx_per_keep = cell_idx_per_keep[shard::num_shards]
        log.info(f"shard {shard}/{num_shards}: encoding {len(keep_idx):,} of the selected panos")

    # Restrict the dataset's row list to our kept indices.
    ds.rows = [ds.rows[i] for i in keep_idx]
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate, pin_memory=True,
    )

    # ---- encode in batches --------------------------------------------------
    feature_dim = model.feat_proj[0].out_features  # 1024 for V1
    log.info(f"feature dim: {feature_dim}")
    dtype = torch.float16 if feature_dtype == "float16" else torch.float32

    n_total = len(ds.rows)
    all_features = torch.empty((n_total, feature_dim), dtype=dtype)
    all_latlngs = torch.empty((n_total, 2), dtype=torch.float32)
    all_cell_idx = torch.tensor(cell_idx_per_keep, dtype=torch.long)

    t0 = time.time()
    cursor = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            pix = batch["pixel_values"].to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                feat = model.encode_pooled(pix)  # [B, D]
            B = feat.shape[0]
            all_features[cursor:cursor + B] = feat.to(dtype).cpu()
            all_latlngs[cursor:cursor + B] = batch["latlng"].cpu()
            cursor += B
            if bi % 100 == 0 or bi == len(loader) - 1:
                elapsed = time.time() - t0
                pps = cursor / max(elapsed, 1e-6)
                eta = (n_total - cursor) / max(pps, 1)
                log.info(
                    f"  encoded {cursor:,}/{n_total:,} "
                    f"({100 * cursor / n_total:.1f}%, {pps:.0f} pano/s, ETA {eta/60:.1f}min)"
                )
    log.info(f"encoding done in {(time.time() - t0)/60:.1f} min")

    # Sharded run: save a RAW partial (features + latlngs + cell_idx) and stop.
    # merge_protonet_shards.py concatenates all partials and builds the index.
    if num_shards > 1:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"features": all_features, "latlngs": all_latlngs,
             "cell_idx": all_cell_idx, "level": level,
             "shard": shard, "num_shards": num_shards},
            out,
        )
        log.info(f"saved shard {shard}/{num_shards} partial → {out} "
                 f"({n_total:,} prototypes, {out.stat().st_size / 1e6:.1f} MB)")
        return

    # ---- group by cell index, build flat layout -----------------------------
    # Sort everything by cell index so each cell's features are contiguous.
    log.info("grouping by cell index ...")
    sort_order = torch.argsort(all_cell_idx, stable=True)
    all_features = all_features[sort_order]
    all_latlngs = all_latlngs[sort_order]
    sorted_cells = all_cell_idx[sort_order]

    vocab_size = cells.vocab_size(level)
    cell_starts = torch.zeros(vocab_size + 1, dtype=torch.long)
    # cumulative counts: count per cell, then prefix-sum.
    counts = torch.bincount(sorted_cells, minlength=vocab_size)
    cell_starts[1:] = torch.cumsum(counts, dim=0)
    assert cell_starts[-1].item() == n_total, (cell_starts[-1].item(), n_total)

    n_filled = int((counts > 0).sum().item())
    avg = n_total / max(n_filled, 1)
    log.info(
        f"cells with prototypes: {n_filled:,}/{vocab_size:,} "
        f"({100 * n_filled / vocab_size:.1f}%), avg {avg:.1f} prototypes/cell"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    index = ProtoNetIndex(all_features, all_latlngs, cell_starts, level=level)
    index.to_file(out)
    log.info(f"saved index → {out} ({out.stat().st_size / 1e6:.1f} MB)")
    log.info(f"summary: {index}")


if __name__ == "__main__":
    app()
