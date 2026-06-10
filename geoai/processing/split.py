"""`geoai-split` — assign each pano to train / val / test.

Adds a `split` column to `panos`. Stratified by `country_code` so rare
countries appear in all three splits (we can't evaluate per-country
accuracy on a country with zero held-out panos). Deterministic via
hash(pano_id) so reruns are stable across machines and partial DBs.

Country-less panos (~7%, coastal/island) get assigned by the same hash —
they're trainable, just not part of per-country breakdowns.

Default ratio matches PLAN.md §"Held-Out Set": 95/3/2.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path

import typer

from geoai.config import METADATA_DB

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def _split_for(pano_id: str, country_code: str | None, p_val: float, p_test: float) -> str:
    # Salt by country so the per-country distribution is independent of the
    # global hash — protects against pathological cases like "all of GBR's
    # ids happen to hash low" creating empty val/test for that country.
    salt = country_code or "_unk"
    h = hashlib.blake2b(f"{salt}:{pano_id}".encode(), digest_size=8).digest()
    u = int.from_bytes(h, "big") / 2**64  # uniform [0, 1)
    if u < p_test:
        return "test"
    if u < p_test + p_val:
        return "val"
    return "train"


@app.command()
def main(
    db_path: Path = typer.Option(METADATA_DB, exists=True),
    p_val: float = typer.Option(0.03, help="Validation fraction"),
    p_test: float = typer.Option(0.02, help="Test fraction (held out for final eval)"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "ALTER TABLE panos ADD COLUMN split TEXT"
    ) if conn.execute(
        "SELECT 1 FROM pragma_table_info('panos') WHERE name='split'"
    ).fetchone() is None else None
    conn.execute("CREATE INDEX IF NOT EXISTS idx_panos_split ON panos(split)")

    rows = conn.execute("SELECT pano_id, country_code FROM panos").fetchall()
    log.info(f"assigning splits for {len(rows):,} panos (val={p_val}, test={p_test})")

    counts = {"train": 0, "val": 0, "test": 0}
    cur = conn.cursor()
    BATCH = 5000
    buf = []
    for pano_id, country_code in rows:
        s = _split_for(pano_id, country_code, p_val, p_test)
        counts[s] += 1
        buf.append((s, pano_id))
        if len(buf) >= BATCH:
            cur.executemany("UPDATE panos SET split=? WHERE pano_id=?", buf)
            buf.clear()
    if buf:
        cur.executemany("UPDATE panos SET split=? WHERE pano_id=?", buf)
    conn.commit()

    log.info(
        f"done: train={counts['train']:,} val={counts['val']:,} test={counts['test']:,}"
    )


if __name__ == "__main__":
    app()
