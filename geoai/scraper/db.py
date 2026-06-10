"""SQLite schema and helpers for the scraper.

The pano_id is the primary key — reruns skip already-seen panos, so the
scraper is resumable by construction.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS panos (
    pano_id          TEXT PRIMARY KEY,
    lat              REAL NOT NULL,
    lng              REAL NOT NULL,
    capture_date     TEXT,
    camera_gen       INTEGER,
    is_official      INTEGER NOT NULL,
    description      TEXT,
    -- Reverse-geocoded admin hierarchy (from GADM 4.1 ADM_2 layer)
    country_code     TEXT,
    country_name     TEXT,
    admin1_code      TEXT,
    admin1_name      TEXT,
    admin2_code      TEXT,
    admin2_name      TEXT,
    -- Multi-level S2 cell IDs (Stage 1 hierarchical classifier targets).
    -- Stored as decimal strings: S2 cells are 64-bit UNSIGNED, but SQLite
    -- INTEGER is signed 64-bit, so high-bit cells overflow on insert.
    s2_l3            TEXT,
    s2_l6            TEXT,
    s2_l9            TEXT,
    s2_l12           TEXT,
    -- Auxiliary features from rasters
    koppen_class     INTEGER,   -- Beck et al. 2018 numeric class 1..30
    koppen_group     TEXT,      -- 'A'..'E'
    pop_per_km2      REAL,      -- GHS_POP 2020, 1km Mollweide pixel
    is_urban         INTEGER,   -- 0/1, GHS-SMOD urban-centre threshold
    -- File-system pointers
    equirect_path    TEXT,      -- absolute, since data may live on a separate volume
    source_dir       TEXT,      -- 'panos', 'panos_new', 'scraped', etc.
    zoom_level       INTEGER NOT NULL DEFAULT 3,
    -- Provenance / status (populated by both scraper and catalog importer)
    download_status  TEXT NOT NULL,
    download_error   TEXT,
    discovered_at    TEXT NOT NULL,
    sample_run_id    TEXT
);

CREATE INDEX IF NOT EXISTS idx_panos_country ON panos(country_code);
CREATE INDEX IF NOT EXISTS idx_panos_s2_l3   ON panos(s2_l3);
CREATE INDEX IF NOT EXISTS idx_panos_s2_l6   ON panos(s2_l6);
CREATE INDEX IF NOT EXISTS idx_panos_status  ON panos(download_status);

-- Same-location historical sibling captures (populated by the scraper only;
-- the imported corpus has no time[] data, so this stays empty for v1).
CREATE TABLE IF NOT EXISTS pano_timeline (
    pano_id    TEXT NOT NULL,
    sibling_id TEXT NOT NULL,
    PRIMARY KEY (pano_id, sibling_id)
);

-- One row per scrape run OR catalog import.
CREATE TABLE IF NOT EXISTS sample_runs (
    run_id       TEXT PRIMARY KEY,
    polygon_name TEXT NOT NULL,
    target_count INTEGER,
    started_at   TEXT NOT NULL,
    completed_at TEXT,
    status       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS crops (
    pano_id   TEXT NOT NULL,
    heading   INTEGER NOT NULL,
    crop_path TEXT NOT NULL,
    PRIMARY KEY (pano_id, heading)
);

CREATE INDEX IF NOT EXISTS idx_crops_pano ON crops(pano_id);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def pano_exists(conn: sqlite3.Connection, pano_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM panos WHERE pano_id = ? LIMIT 1", (pano_id,)
    ).fetchone()
    return row is not None


def record_pano(
    conn: sqlite3.Connection,
    *,
    pano_id: str,
    lat: float,
    lng: float,
    capture_date: Optional[str],
    camera_gen: Optional[int],
    is_official: bool,
    description: Optional[str],
    equirect_path: Optional[str],
    zoom_level: int,
    download_status: str,
    sample_run_id: str,
    discovered_at: str,
    download_error: Optional[str] = None,
) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO panos (
            pano_id, lat, lng, capture_date, camera_gen, is_official,
            description, equirect_path, zoom_level, download_status,
            download_error, discovered_at, sample_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            pano_id, lat, lng, capture_date, camera_gen, int(is_official),
            description, equirect_path, zoom_level, download_status,
            download_error, discovered_at, sample_run_id,
        ),
    )


def record_timeline_edge(
    conn: sqlite3.Connection, pano_id: str, sibling_id: str
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO pano_timeline VALUES (?, ?)",
        (pano_id, sibling_id),
    )
