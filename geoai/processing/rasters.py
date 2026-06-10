"""Raster lookups: Köppen-Geiger climate class and GHS_POP population density.

Both rasters are global single-band GeoTIFFs we open lazily and query
point-by-point. Köppen is in EPSG:4326 (lat/lng direct), GHS_POP is in
ESRI:54009 (Mollweide), so we reproject query points before sampling.

Usage pattern: long-running scripts open the raster once via the module
helpers and reuse the open dataset. Worker processes should reopen in
their own address space — rasterio handles aren't fork-safe.
"""
from __future__ import annotations

from typing import Optional

import rasterio
from rasterio.warp import transform as rio_transform

from geoai.config import GHS_POP_PATH, KOPPEN_PATH

# Köppen-Geiger class IDs 1..30 (Beck et al. 2018). 0 = ocean / no data.
# Group code is the first letter of the standard label (A/B/C/D/E).
# We expose just the group letter; the full label is overkill for ML labels.
_KOPPEN_GROUP = {
    # Beck et al. 2018 numbering: 14=Cfa, 15=Cfb, 16=Cfc are still group C —
    # group D ("Continental") starts at 17 (Dsa).
    **{i: "A" for i in (1, 2, 3)},                                  # Tropical
    **{i: "B" for i in (4, 5, 6, 7)},                               # Arid
    **{i: "C" for i in (8, 9, 10, 11, 12, 13, 14, 15, 16)},         # Temperate
    **{i: "D" for i in (17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)},  # Continental
    **{i: "E" for i in (29, 30)},                                   # Polar
}

_kg_ds: Optional[rasterio.io.DatasetReader] = None
_pop_ds: Optional[rasterio.io.DatasetReader] = None


def _open_koppen() -> rasterio.io.DatasetReader:
    global _kg_ds
    if _kg_ds is None:
        _kg_ds = rasterio.open(KOPPEN_PATH)
    return _kg_ds


def _open_pop() -> rasterio.io.DatasetReader:
    global _pop_ds
    if _pop_ds is None:
        _pop_ds = rasterio.open(GHS_POP_PATH)
    return _pop_ds


def koppen_class(lat: float, lng: float) -> tuple[Optional[int], Optional[str]]:
    """Return (numeric class 1..30, group letter A-E) or (None, None) for ocean."""
    ds = _open_koppen()
    row, col = ds.index(lng, lat)  # EPSG:4326 ⇒ (lng, lat) ordering
    if not (0 <= row < ds.height and 0 <= col < ds.width):
        return None, None
    val = int(ds.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
    if val == 0:
        return None, None
    return val, _KOPPEN_GROUP.get(val)


def population_density(lat: float, lng: float) -> Optional[float]:
    """Population per ~1 km² Mollweide pixel at this point. None if out-of-bounds."""
    ds = _open_pop()
    # Reproject query point from WGS84 → Mollweide.
    xs, ys = rio_transform("EPSG:4326", ds.crs, [lng], [lat])
    row, col = ds.index(xs[0], ys[0])
    if not (0 <= row < ds.height and 0 <= col < ds.width):
        return None
    val = float(ds.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
    if ds.nodata is not None and val == ds.nodata:
        return None
    return max(val, 0.0)


# GHS-SMOD's "urban centre" threshold is 1500 inhabitants/km² in a
# contiguous 1 km grid. We use the same cutoff as a per-cell heuristic.
URBAN_POP_THRESHOLD = 1500.0


def is_urban(lat: float, lng: float) -> bool:
    pop = population_density(lat, lng)
    return pop is not None and pop >= URBAN_POP_THRESHOLD
