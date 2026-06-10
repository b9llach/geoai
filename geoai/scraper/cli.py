"""CLI entry point — `geoai-scrape`."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
from shapely.geometry import shape

from geoai.scraper.coverage import has_coverage_near
from geoai.scraper.db import (
    init_db,
    pano_exists,
    record_pano,
    record_timeline_edge,
)
from geoai.scraper.sampler import iter_samples
from geoai.scraper.streetview_client import (
    PanoFilter,
    StreetViewClient,
    camera_generation,
    passes_filter,
)

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_polygon(polygon_file: Path):
    """Accept either a bare Geometry, a Feature, or a FeatureCollection."""
    data = json.loads(polygon_file.read_text())
    t = data.get("type")
    if t == "FeatureCollection":
        from shapely.ops import unary_union
        return unary_union([shape(f["geometry"]) for f in data["features"]])
    if t == "Feature":
        return shape(data["geometry"])
    return shape(data)


async def run_scrape(
    polygon_file: Path,
    target_count: int,
    output_dir: Path,
    zoom: int = 3,
    concurrency: int = 4,
    collect_timeline: bool = True,
    oversample: int = 10,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "processed" / "metadata.db"
    conn = init_db(db_path)
    run_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO sample_runs VALUES (?, ?, ?, ?, NULL, 'running')",
        (run_id, polygon_file.name, target_count, _now()),
    )
    conn.commit()

    poly = _load_polygon(polygon_file)
    pano_filter = PanoFilter(
        official_only=True, min_camera_gen=2, reject_trekkers=True
    )

    collected = 0
    rejected_no_coverage = 0
    rejected_filter = 0

    async with StreetViewClient(max_concurrent=concurrency) as client:
        for lat, lng in iter_samples(poly, n=target_count * oversample):
            if collected >= target_count:
                break

            if not has_coverage_near(lat, lng, radius_m=50):
                rejected_no_coverage += 1
                continue

            pano = await client.find_nearest(lat, lng, radius_m=50)
            if pano is None:
                rejected_no_coverage += 1
                continue
            if pano_exists(conn, pano.id):
                continue
            if not passes_filter(pano, pano_filter):
                rejected_filter += 1
                continue

            to_download = [pano]
            if collect_timeline:
                to_download = await client.get_timeline(pano)

            for p in to_download:
                if pano_exists(conn, p.id):
                    continue
                eqr_rel = Path("raw") / "equirectangular" / f"{p.id}.jpg"
                eqr_abs = output_dir / eqr_rel
                eqr_abs.parent.mkdir(parents=True, exist_ok=True)
                ok = await client.download_equirect(p, str(eqr_abs), zoom=zoom)
                record_pano(
                    conn,
                    pano_id=p.id,
                    lat=p.lat,
                    lng=p.lon,
                    capture_date=str(p.date) if p.date else None,
                    camera_gen=camera_generation(p),
                    is_official=(len(p.id) == 22),
                    description=None,  # streetlevel 0.12.5 has no description field
                    equirect_path=str(eqr_rel) if ok else None,
                    zoom_level=zoom,
                    download_status="success" if ok else "failed",
                    sample_run_id=run_id,
                    discovered_at=_now(),
                )
                if p.id != pano.id:
                    record_timeline_edge(conn, pano.id, p.id)

            collected += 1
            if collected % 50 == 0:
                log.info(
                    f"Progress: {collected}/{target_count} "
                    f"(no-coverage: {rejected_no_coverage}, filter: {rejected_filter})"
                )
            conn.commit()

    conn.execute(
        "UPDATE sample_runs SET completed_at=?, status='complete' WHERE run_id=?",
        (_now(), run_id),
    )
    conn.commit()
    log.info(
        f"Done. {collected} unique locations collected "
        f"(no-coverage: {rejected_no_coverage}, filter: {rejected_filter})"
    )


@app.command()
def main(
    polygon: Path = typer.Option(..., exists=True, help="GeoJSON polygon file"),
    target: int = typer.Option(20_000, help="Number of unique locations to collect"),
    output: Path = typer.Option(Path("data"), help="Output directory"),
    zoom: int = typer.Option(3, help="Panorama zoom level (3 default, 5 for hi-res subset)"),
    concurrency: int = typer.Option(4, help="Max concurrent requests"),
    timeline: bool = typer.Option(True, help="Also collect historical captures"),
    oversample: int = typer.Option(10, help="Draw this multiple of `target` raw samples before giving up"),
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    asyncio.run(
        run_scrape(polygon, target, output, zoom, concurrency, timeline, oversample)
    )


if __name__ == "__main__":
    app()
