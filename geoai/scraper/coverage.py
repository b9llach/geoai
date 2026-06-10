"""Coverage tile pre-filter — reject points not near Street View blue lines.

Google's coverage tiles are small Protobuf blobs keyed by (z=17, x, y) that
list every official pano rooted on that tile. Fetching one tile and caching
it in-process lets many sample points reuse a single round-trip. Without
this pre-filter, ~80% of random rural US points return nothing from
`find_panorama_async`, wasting a network round-trip each time.
"""
from __future__ import annotations

import math
from functools import lru_cache

from streetlevel import streetview

_COVERAGE_ZOOM = 17


def latlng_to_tile(lat: float, lng: float, zoom: int = _COVERAGE_ZOOM) -> tuple[int, int]:
    """Standard slippy-map tile coordinates at the given zoom."""
    n = 2.0 ** zoom
    x = int((lng + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


@lru_cache(maxsize=20_000)
def _coverage_panos_cached(tile_x: int, tile_y: int) -> tuple:
    """Tuple (immutable) so @lru_cache is safe. One tile covers many sample points."""
    try:
        panos = streetview.get_coverage_tile(tile_x, tile_y)
    except Exception:
        return ()
    return tuple(panos) if panos else ()


def has_coverage_near(lat: float, lng: float, radius_m: int = 50) -> bool:
    """True iff any official pano exists within `radius_m` of (lat, lng)."""
    tx, ty = latlng_to_tile(lat, lng)
    panos = _coverage_panos_cached(tx, ty)
    if not panos:
        return False
    for p in panos:
        if haversine_m(lat, lng, p.lat, p.lon) <= radius_m:
            return True
    return False


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
