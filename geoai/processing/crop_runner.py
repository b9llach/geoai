"""`geoai-render-crops` — render 4 perspective views per pano in parallel.

Reads the catalog from SQLite, finds panos that don't yet have all 4 crops,
distributes them across worker processes, and writes the JPEGs into a
sharded directory tree. After each pano completes, its 4 crop paths are
inserted into the `crops` table (with INSERT OR IGNORE so reruns are safe).

Sharding scheme: `crops/<pano_id[0:2]>/<pano_id>_<heading>.jpg`. This keeps
any single directory under ~4k files so `ls` and FUSE filesystems stay sane.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sqlite3
import time
from pathlib import Path
from typing import Iterator

import typer
from tqdm import tqdm

from geoai.config import CROPS_DIR, CROP_HEADINGS_DEG, METADATA_DB
from geoai.processing.perspective import equirect_to_perspective_crops

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def _shard_dir(pano_id: str) -> Path:
    return CROPS_DIR / pano_id[:2]


def _expected_crop_paths(pano_id: str) -> list[Path]:
    d = _shard_dir(pano_id)
    return [d / f"{pano_id}_{h:03d}.jpg" for h in CROP_HEADINGS_DEG]


def _pano_already_done(pano_id: str) -> bool:
    return all(p.exists() for p in _expected_crop_paths(pano_id))


# ---- worker ----------------------------------------------------------------

_WORKER_CONN: sqlite3.Connection | None = None
_WORKER_DB_PATH: str | None = None


def _worker_init(db_path: str) -> None:
    """Open a per-worker SQLite connection in WAL mode for concurrent writes."""
    global _WORKER_CONN, _WORKER_DB_PATH
    _WORKER_DB_PATH = db_path
    _WORKER_CONN = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    _WORKER_CONN.execute("PRAGMA journal_mode=WAL")
    _WORKER_CONN.execute("PRAGMA synchronous=NORMAL")


def _render_one(args: tuple[str, str]) -> tuple[str, str | None]:
    """Worker: render 4 crops for one pano. Returns (pano_id, error|None)."""
    pano_id, equirect_path = args
    try:
        if _pano_already_done(pano_id):
            return pano_id, None
        out_dir = _shard_dir(pano_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = equirect_to_perspective_crops(Path(equirect_path), out_dir, pano_id)
        assert _WORKER_CONN is not None
        rows = [
            (pano_id, h, str(p))
            for h, p in zip(CROP_HEADINGS_DEG, paths)
        ]
        _WORKER_CONN.executemany(
            "INSERT OR REPLACE INTO crops VALUES (?, ?, ?)", rows
        )
        return pano_id, None
    except Exception as e:
        return pano_id, repr(e)


# ---- driver ----------------------------------------------------------------

def _iter_pending(
    conn: sqlite3.Connection, batch_size: int = 5_000
) -> Iterator[tuple[str, str]]:
    """Yield (pano_id, equirect_path) for panos missing crops, in batches."""
    cur = conn.cursor()
    cur.execute(
        """SELECT p.pano_id, p.equirect_path FROM panos p
           WHERE p.equirect_path IS NOT NULL
             AND p.download_status = 'success'
             AND NOT EXISTS (
               SELECT 1 FROM crops c
                WHERE c.pano_id = p.pano_id AND c.heading = ?
             )""",
        (CROP_HEADINGS_DEG[-1],),  # last heading written wins → its presence ⇒ all 4 done
    )
    while True:
        chunk = cur.fetchmany(batch_size)
        if not chunk:
            return
        yield from chunk


@app.command()
def main(
    db_path: Path = typer.Option(METADATA_DB, exists=True, help="Catalog SQLite DB"),
    workers: int = typer.Option(
        max(1, (os.cpu_count() or 4) - 2),
        help="Worker process count (defaults to ncpu - 2)",
    ),
    limit: int = typer.Option(0, help="Stop after N panos (0 = no limit). Use for smoke testing."),
    chunksize: int = typer.Option(8, help="Pool.imap_unordered chunksize"),
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    main_conn = sqlite3.connect(db_path, timeout=30.0)
    main_conn.execute("PRAGMA journal_mode=WAL")
    total_q = main_conn.execute("SELECT COUNT(*) FROM panos WHERE equirect_path IS NOT NULL").fetchone()[0]
    log.info(f"catalog has {total_q:,} panos with equirect; planning crop render")

    pending = list(_iter_pending(main_conn))
    main_conn.close()
    if limit:
        pending = pending[:limit]
    log.info(f"{len(pending):,} panos pending crops; spawning {workers} workers")

    if not pending:
        log.info("nothing to do — every pano already has 4 crops")
        return

    t0 = time.time()
    n_done = n_err = 0

    ctx = mp.get_context("spawn")  # avoids fork-safety issues with PIL/rasterio/etc
    with ctx.Pool(workers, initializer=_worker_init, initargs=(str(db_path),)) as pool:
        bar = tqdm(
            pool.imap_unordered(_render_one, pending, chunksize=chunksize),
            total=len(pending), unit="pano",
        )
        for pano_id, err in bar:
            if err:
                n_err += 1
                if n_err <= 10:
                    log.warning(f"render failed: {pano_id}: {err}")
            else:
                n_done += 1
            if (n_done + n_err) % 1000 == 0:
                bar.set_postfix(
                    done=n_done, err=n_err,
                    rate=f"{(n_done + n_err) / (time.time() - t0):.0f}/s",
                )

    elapsed = time.time() - t0
    log.info(
        f"done: rendered={n_done:,}  errors={n_err:,}  "
        f"elapsed={elapsed/60:.1f}min  rate={(n_done+n_err)/elapsed:.0f}/s"
    )


if __name__ == "__main__":
    app()
