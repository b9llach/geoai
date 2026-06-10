"""Merge raw ProtoNet shard partials (from build_protonet_index.py --num-shards N)
into a single ProtoNetIndex .pt.

Each shard saved {features, latlngs, cell_idx} for a disjoint strided slice of
the SAME deterministic prototype set. The index is an order-independent per-cell
bag, so merging is just concatenation + the standard group-by-cell layout
(identical to the single-GPU build's final stage).

Usage:
    .venv/bin/python scripts/merge_protonet_shards.py \\
        --shards out.shard0.pt,out.shard1.pt \\
        --out    .../protonet_l9.pt \\
        --cells-parquet .../cells_v3.parquet --level 9
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import typer

from geoai.config import PROCESSED_DIR
from geoai.stage1.cells import CellVocab
from geoai.stage1.protonet import ProtoNetIndex

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


@app.command()
def main(
    shards: str = typer.Option(..., help="Comma-separated shard partial .pt paths"),
    out: Path = typer.Option(..., help="Final ProtoNetIndex .pt path"),
    level: int = typer.Option(9),
    cells_parquet: Optional[Path] = typer.Option(None, help="Cells parquet (must match the build)"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cells = CellVocab.from_parquet(path=cells_parquet) if cells_parquet else CellVocab.from_parquet()
    vocab_size = cells.vocab_size(level)

    paths = [Path(p.strip()) for p in shards.split(",") if p.strip()]
    feats, lls, cidx = [], [], []
    seen = set()
    for p in paths:
        d = torch.load(str(p), map_location="cpu", weights_only=True)
        assert d["level"] == level, f"{p}: level {d['level']} != {level}"
        sh = int(d.get("shard", -1))
        if sh in seen:
            raise SystemExit(f"duplicate shard index {sh} ({p}) — shards must be distinct")
        seen.add(sh)
        feats.append(d["features"]); lls.append(d["latlngs"]); cidx.append(d["cell_idx"])
        log.info(f"loaded {p.name}: shard {sh}/{d.get('num_shards','?')}, {d['features'].shape[0]:,} prototypes")

    all_features = torch.cat(feats, dim=0)
    all_latlngs = torch.cat(lls, dim=0)
    all_cell_idx = torch.cat(cidx, dim=0)
    n_total = all_features.shape[0]
    log.info(f"merged {len(paths)} shards → {n_total:,} prototypes")

    # ---- identical group-by-cell layout as the single-GPU build -------------
    sort_order = torch.argsort(all_cell_idx, stable=True)
    all_features = all_features[sort_order]
    all_latlngs = all_latlngs[sort_order]
    sorted_cells = all_cell_idx[sort_order]

    cell_starts = torch.zeros(vocab_size + 1, dtype=torch.long)
    counts = torch.bincount(sorted_cells, minlength=vocab_size)
    cell_starts[1:] = torch.cumsum(counts, dim=0)
    assert cell_starts[-1].item() == n_total, (cell_starts[-1].item(), n_total)

    n_filled = int((counts > 0).sum().item())
    log.info(f"cells with prototypes: {n_filled:,}/{vocab_size:,} "
             f"({100*n_filled/vocab_size:.1f}%), avg {n_total/max(n_filled,1):.1f}/cell")

    out.parent.mkdir(parents=True, exist_ok=True)
    index = ProtoNetIndex(all_features, all_latlngs, cell_starts, level=level)
    index.to_file(out)
    log.info(f"saved merged index → {out} ({out.stat().st_size/1e6:.1f} MB)")
    log.info(f"summary: {index}")


if __name__ == "__main__":
    app()
