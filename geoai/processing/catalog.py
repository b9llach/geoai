"""`geoai-catalog` — one-shot ingestion of the imported pano corpus into SQLite.

Reads the master CSV, walks the on-disk pano directories, joins them, and
for every linked pano computes:
    - reverse-geocoded country / admin1 / admin2 (GADM 4.1)
    - S2 cell IDs at levels 3, 6, 9, 12
    - Köppen-Geiger climate class + group letter
    - GHS 2020 population per km², plus an is_urban flag

Writes everything into the `panos` table. Resumable: rerun and already-
catalogued pano_ids are skipped (PK conflict + INSERT OR IGNORE).

Crops are NOT rendered here — that's `geoai-render-crops`.
"""
from __future__ import annotations

import csv
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
from tqdm import tqdm

from geoai.config import (
    METADATA_DB,
    PANO_LOG_CSV,
    PANOS_DIRS,
    PROCESSED_DIR,
)
from geoai.processing import rasters, reverse_geo
from geoai.processing.s2_cells import s2_cells_for_point
from geoai.scraper.db import init_db

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_file_index(
    dirs: list[Path], source_label: str | None = None,
) -> dict[str, tuple[str, str]]:
    """pano_id → (source_dir_label, absolute_equirect_path).

    `source_label` overrides the label stored in the DB (default = dir basename).
    First dir in the list wins on duplicate pano_ids (preserves V1 behavior:
    `panos` beats `panos_new`).
    """
    idx: dict[str, tuple[str, str]] = {}
    for d in dirs:
        if not d.is_dir():
            log.warning(f"missing source dir: {d}")
            continue
        tag = source_label or d.name
        for entry in d.iterdir():
            if entry.suffix == ".jpg":
                idx.setdefault(entry.stem, (tag, str(entry.resolve())))
    return idx


def _existing_pano_ids(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT pano_id FROM panos")}


def _ingest_row(
    conn: sqlite3.Connection,
    pano_id: str,
    lat: float,
    lng: float,
    source_dir: str,
    equirect_path: str,
    run_id: str,
    discovered_at: str,
) -> None:
    geo = reverse_geo.reverse_geocode(lat, lng)
    cells = s2_cells_for_point(lat, lng)
    kg_class, kg_group = rasters.koppen_class(lat, lng)
    pop = rasters.population_density(lat, lng)
    urban = pop is not None and pop >= rasters.URBAN_POP_THRESHOLD

    conn.execute(
        """INSERT OR IGNORE INTO panos (
            pano_id, lat, lng, capture_date, camera_gen, is_official, description,
            country_code, country_name, admin1_code, admin1_name, admin2_code, admin2_name,
            s2_l3, s2_l6, s2_l9, s2_l12,
            koppen_class, koppen_group, pop_per_km2, is_urban,
            equirect_path, source_dir, zoom_level,
            download_status, download_error, discovered_at, sample_run_id
        ) VALUES (?, ?, ?, NULL, NULL, ?, NULL,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?,
            'success', NULL, ?, ?)""",
        (
            pano_id, lat, lng, int(len(pano_id) == 22),
            geo["country_code"], geo["country_name"],
            geo["admin1_code"], geo["admin1_name"],
            geo["admin2_code"], geo["admin2_name"],
            str(cells[3]), str(cells[6]), str(cells[9]), str(cells[12]),
            kg_class, kg_group, pop, int(urban),
            equirect_path, source_dir, 3,
            discovered_at, run_id,
        ),
    )


@app.command()
def main(
    csv_path: Path = typer.Option(PANO_LOG_CSV, exists=True, help="Master metadata CSV"),
    source_dirs: list[Path] = typer.Option(
        None, "--source-dir",
        help="Pano JPG directory. Pass multiple times for multi-source ingest. "
             "Defaults to the configured PANOS_DIRS (the main corpus).",
    ),
    source_label: str = typer.Option(
        None, "--source-label",
        help="Override `source_dir` column value in DB. Default: scanned dir's basename. "
             "Use 'panos_supplement' when ingesting the supplement scrape so it's tagged distinctly.",
    ),
    run_label: str = typer.Option(
        "imported_corpus", "--run-label",
        help="Goes into the sample_runs table for this ingest pass.",
    ),
    db_path: Path = typer.Option(METADATA_DB, help="Output SQLite database"),
    limit: int = typer.Option(0, help="Stop after N panos (0 = no limit). Use for smoke testing."),
    commit_every: int = typer.Option(2_000, help="Rows between SQLite commits"),
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    conn = init_db(db_path)

    dirs = list(source_dirs) if source_dirs else list(PANOS_DIRS)
    log.info(f"scanning {len(dirs)} directories for pano JPGs: {[str(d) for d in dirs]}")
    log.info(f"csv source: {csv_path}")

    t0 = time.time()
    file_idx = _build_file_index(dirs, source_label=source_label)
    log.info(f"indexed {len(file_idx):,} files in {time.time()-t0:.1f}s")

    log.info("loading GADM ADM_2 (one-time, ~3-30s)...")
    reverse_geo.load_gadm()

    already = _existing_pano_ids(conn)
    log.info(f"{len(already):,} panos already in DB; will skip on re-encounter")

    run_id = uuid.uuid4().hex
    started = _now()
    conn.execute(
        "INSERT INTO sample_runs VALUES (?, ?, NULL, ?, NULL, 'running')",
        (run_id, run_label, started),
    )
    conn.commit()

    n_seen = n_ingested = n_orphan_csv = 0
    t_loop = time.time()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        bar = tqdm(reader, unit="pano")
        for row in bar:
            n_seen += 1
            pano_id = row["pano_id"]
            if pano_id in already:
                continue
            entry = file_idx.get(pano_id)
            if entry is None:
                n_orphan_csv += 1
                continue
            source_tag, equirect_path = entry
            try:
                lat = float(row["lat"])
                lng = float(row["lng"])
            except ValueError:
                continue
            _ingest_row(conn, pano_id, lat, lng, source_tag, equirect_path, run_id, _now())
            n_ingested += 1

            if n_ingested % commit_every == 0:
                conn.commit()
                bar.set_postfix(
                    ingested=n_ingested,
                    csv_orphans=n_orphan_csv,
                    rate=f"{n_ingested / (time.time() - t_loop):.0f}/s",
                )
            if limit and n_ingested >= limit:
                break

    conn.commit()

    # Files present on disk but missing from CSV → can't train on them (no lat/lng)
    n_orphan_file = sum(
        1 for pid in file_idx
        if conn.execute("SELECT 1 FROM panos WHERE pano_id=?", (pid,)).fetchone() is None
    )

    conn.execute(
        "UPDATE sample_runs SET completed_at=?, status='complete' WHERE run_id=?",
        (_now(), run_id),
    )
    conn.commit()
    log.info(
        f"done: ingested={n_ingested:,}  csv-without-file={n_orphan_csv:,}  "
        f"file-without-csv={n_orphan_file:,}  total-csv-rows={n_seen:,}  "
        f"elapsed={time.time()-t_loop:.1f}s"
    )


if __name__ == "__main__":
    app()
