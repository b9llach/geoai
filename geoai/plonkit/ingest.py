"""`geoai-ingest-country-guides` — read curated country guide markdown files
into a SQLite knowledge base for Stage 2 RAG retrieval.

Source: `data/country_guides/{ISO3}.md` — one file per country, with YAML
front-matter + ## section headers. Each file is read, section-split, and
inserted (or updated) in `/data/geolocation/processed/country_guides.db`.

Schema:
    countries(country_code PK, country_name, driving_side, languages, scripts, last_updated)
    country_sections(country_code, section, text, PRIMARY KEY (country_code, section))

The CLI is idempotent: re-running it picks up edits and new files. Removed
sections are not auto-deleted (would lose history); `--prune` flag does that.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
import yaml

from geoai.config import PROCESSED_DIR

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

# Repo-tracked source location. Country guides are small text — they belong
# under version control with the code, not on the giant data volume.
GUIDES_DIR = Path(__file__).resolve().parents[2] / "data" / "country_guides"
DB_PATH = PROCESSED_DIR / "country_guides.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS countries (
    country_code  TEXT PRIMARY KEY,
    country_name  TEXT,
    driving_side  TEXT,
    languages     TEXT,    -- JSON array, e.g. ["Catalan"]
    scripts       TEXT,    -- JSON array, e.g. ["Latin"]
    source        TEXT,
    last_updated  TEXT
);

CREATE TABLE IF NOT EXISTS country_sections (
    country_code  TEXT NOT NULL,
    section       TEXT NOT NULL,
    text          TEXT NOT NULL,
    PRIMARY KEY (country_code, section)
);

CREATE INDEX IF NOT EXISTS idx_country_sections_cc ON country_sections(country_code);
"""


@dataclass
class CountryGuide:
    country_code: str
    country_name: Optional[str]
    driving_side: Optional[str]
    languages: list[str]
    scripts: list[str]
    source: Optional[str]
    last_updated: Optional[str]
    sections: dict[str, str]   # section name -> full text


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_SECTION_RE = re.compile(r"^##\s+(\S[^\n]*)\s*\n", re.MULTILINE)


def parse_guide(path: Path) -> CountryGuide:
    raw = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        raise ValueError(f"{path}: missing YAML frontmatter (--- ... ---)")
    fm = yaml.safe_load(m.group(1)) or {}
    body = m.group(2)

    # Split body at "## section" headers. Anything before the first header is dropped.
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(body))
    for i, mm in enumerate(matches):
        name = mm.group(1).strip().lower().replace(" ", "_")
        start = mm.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[start:end].strip()
        if chunk:
            sections[name] = chunk

    return CountryGuide(
        country_code=str(fm.get("country_code", path.stem)).upper(),
        country_name=fm.get("country_name"),
        driving_side=fm.get("driving_side"),
        languages=list(fm.get("languages", []) or []),
        scripts=list(fm.get("scripts", []) or []),
        source=fm.get("source"),
        last_updated=str(fm.get("last_updated")) if fm.get("last_updated") else None,
        sections=sections,
    )


def upsert(conn: sqlite3.Connection, guide: CountryGuide) -> None:
    conn.execute(
        """INSERT INTO countries
           (country_code, country_name, driving_side, languages, scripts, source, last_updated)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(country_code) DO UPDATE SET
               country_name = excluded.country_name,
               driving_side = excluded.driving_side,
               languages    = excluded.languages,
               scripts      = excluded.scripts,
               source       = excluded.source,
               last_updated = excluded.last_updated""",
        (
            guide.country_code,
            guide.country_name,
            guide.driving_side,
            json.dumps(guide.languages),
            json.dumps(guide.scripts),
            guide.source,
            guide.last_updated,
        ),
    )
    for section, text in guide.sections.items():
        conn.execute(
            """INSERT INTO country_sections (country_code, section, text)
               VALUES (?, ?, ?)
               ON CONFLICT(country_code, section) DO UPDATE SET text = excluded.text""",
            (guide.country_code, section, text),
        )


@app.command()
def main(
    guides_dir: Path = typer.Option(GUIDES_DIR, exists=True, file_okay=False),
    db_path: Path = typer.Option(DB_PATH),
    prune: bool = typer.Option(
        False, "--prune", help="Delete DB rows for countries/sections no longer in source files."
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)

    seen_codes: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    for md in sorted(guides_dir.glob("*.md")):
        guide = parse_guide(md)
        upsert(conn, guide)
        seen_codes.add(guide.country_code)
        for s in guide.sections:
            seen_pairs.add((guide.country_code, s))
        log.info(f"  {guide.country_code} {guide.country_name or '':30s} sections={len(guide.sections)}")

    if prune:
        existing = conn.execute("SELECT country_code FROM countries").fetchall()
        for (cc,) in existing:
            if cc not in seen_codes:
                conn.execute("DELETE FROM country_sections WHERE country_code=?", (cc,))
                conn.execute("DELETE FROM countries WHERE country_code=?", (cc,))
                log.info(f"  pruned {cc}")
        for (cc, s) in conn.execute("SELECT country_code, section FROM country_sections").fetchall():
            if (cc, s) not in seen_pairs:
                conn.execute("DELETE FROM country_sections WHERE country_code=? AND section=?", (cc, s))

    conn.commit()
    n_countries = conn.execute("SELECT COUNT(*) FROM countries").fetchone()[0]
    n_sections = conn.execute("SELECT COUNT(*) FROM country_sections").fetchone()[0]
    log.info(f"done: {n_countries} countries, {n_sections} sections in {db_path}")


if __name__ == "__main__":
    app()
