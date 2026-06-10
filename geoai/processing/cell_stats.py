"""`geoai-cell-stats` — derive per-S2-cell counts and empirical centroids.

For each level in {3, 6, 9, 12} we compute the rows you need for Stage 1:
    * `count`         — number of TRAIN-split panos that fall in this cell
    * `centroid_lat`  — mean lat of those panos (NOT the S2 cell's geometric center)
    * `centroid_lng`  — mean lng

Empirical centroids beat S2 geometric centers because S2 cells often span
ocean / empty land; the average of training points is what the haversine
loss should snap to and what inference should report as the candidate
location.

Output: parquet at `<PROCESSED_DIR>/cells.parquet` with columns
`(level, cell_id_str, count, centroid_lat, centroid_lng)`.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from geoai.config import METADATA_DB, PROCESSED_DIR

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

LEVELS = (3, 6, 9, 12)


@app.command()
def main(
    db_path: Path = typer.Option(METADATA_DB, exists=True),
    out_path: Path = typer.Option(PROCESSED_DIR / "cells.parquet"),
    split: str = typer.Option("train", help="Compute centroids over this split only"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    conn = sqlite3.connect(db_path)
    has_split = conn.execute(
        "SELECT 1 FROM pragma_table_info('panos') WHERE name='split'"
    ).fetchone() is not None
    where = f"WHERE split = '{split}'" if has_split else ""
    if not has_split:
        log.warning("no `split` column yet — using ALL panos. Re-run after geoai-split.")

    frames = []
    for lvl in LEVELS:
        col = f"s2_l{lvl}"
        log.info(f"aggregating level {lvl}...")
        df = pd.read_sql_query(
            f"""SELECT {col} AS cell_id_str,
                       COUNT(*) AS count,
                       AVG(lat) AS centroid_lat,
                       AVG(lng) AS centroid_lng
                FROM panos
                {where}
                GROUP BY {col}""",
            conn,
        )
        df.insert(0, "level", lvl)
        log.info(f"  level {lvl}: {len(df):,} cells, total panos {df['count'].sum():,}")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info(f"wrote {len(out):,} rows → {out_path}")


if __name__ == "__main__":
    app()
