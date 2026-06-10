"""Offline reverse geocoding via GADM 4.1 admin polygons.

Loads ADM_2 once into memory (a few minutes the first time, after which
lookups are O(log N) via the spatial index). ADM_2 already includes the
parent admin1 / admin0 GIDs and names, so where ADM_2 exists a single
layer covers all three levels we use in the schema.

BUT: ~14 countries/territories have NO admin-2 subdivision in GADM (Qatar,
Singapore, Puerto Rico, Israel, Lesotho, Montenegro, N. Macedonia, Bahrain,
Kuwait, the Euro microstates, …). They exist only in ADM_0/ADM_1. Querying
ADM_2 alone returns country_code=None for every pano in those places — which
silently dropped ~95k panos (4.5% of the corpus) from the country head's
training signal. So we also load ADM_1 and fall back to it on an ADM_2 miss,
filling country_code + admin1 (admin2 stays None, as it genuinely has none).

Costs ~1.6 GB of RAM after load. Worker processes should call `load_gadm`
in their own process — geopandas DataFrames don't share well via fork.
"""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
from shapely.geometry import Point

from geoai.config import GADM_PATH

_GADM: Optional[gpd.GeoDataFrame] = None
_GADM1: Optional[gpd.GeoDataFrame] = None
_GADM0: Optional[gpd.GeoDataFrame] = None

# Manual country-bbox overrides for places GADM 4.1 doesn't separate from
# their parent state. Each entry: (cc, country_name, lat_min, lat_max,
# lng_min, lng_max). Checked BEFORE GADM lookup; first hit wins. Bboxes are
# slightly conservative on the mainland-China-adjacent borders (Shenzhen for
# HKG, Zhuhai for MAC) so we don't mis-relabel border-crossing panos.
_BBOX_OVERRIDES: tuple = (
    ("MAC", "Macau",     22.108, 22.222, 113.526, 113.601),
    # HK/Shenzhen border isn't a clean horizontal — western frontier is at
    # ~22.50 (Deep Bay), eastern frontier ~22.55 (Sha Tau Kok). 22.51 is the
    # pragmatic compromise: keeps Shenzhen center (~22.54+) well out while
    # preserving the densely-covered south HK (Island + Kowloon + most NT).
    # Loses a thin northern HK strip (Sha Tau Kok area) — sparse SV anyway.
    ("HKG", "Hong Kong", 22.135, 22.510, 113.820, 114.510),
)


def load_gadm(
    path=GADM_PATH, layer: str = "ADM_2", columns: tuple[str, ...] | None = None
) -> gpd.GeoDataFrame:
    """Load GADM into the module global. Subsequent calls are no-ops."""
    global _GADM
    if _GADM is not None:
        return _GADM
    if columns is None:
        columns = (
            "GID_0", "COUNTRY",
            "GID_1", "NAME_1",
            "GID_2", "NAME_2",
            "geometry",
        )
    _GADM = gpd.read_file(path, layer=layer, columns=list(columns))
    # Touch the spatial index so it's built on the main thread.
    _ = _GADM.sindex
    return _GADM


def load_gadm0(path=GADM_PATH) -> gpd.GeoDataFrame:
    """Load the ADM_0 (country-outline only) fallback layer. Used when both
    ADM_2 AND ADM_1 miss — chiefly for tiny territories that GADM stores only
    at ADM_0 (Cocos Keeling = CCK, Christmas Island = CXR)."""
    global _GADM0
    if _GADM0 is not None:
        return _GADM0
    _GADM0 = gpd.read_file(
        path, layer="ADM_0", columns=["GID_0", "COUNTRY", "geometry"],
    )
    _ = _GADM0.sindex
    return _GADM0


def load_gadm1(path=GADM_PATH) -> gpd.GeoDataFrame:
    """Load the ADM_1 fallback layer (country + admin1, no admin2). Used when
    an ADM_2 lookup misses — chiefly for countries with no admin-2 level."""
    global _GADM1
    if _GADM1 is not None:
        return _GADM1
    _GADM1 = gpd.read_file(
        path, layer="ADM_1",
        columns=["GID_0", "COUNTRY", "GID_1", "NAME_1", "geometry"],
    )
    _ = _GADM1.sindex
    return _GADM1


def reverse_geocode(lat: float, lng: float) -> dict[str, Optional[str]]:
    """Return GADM admin codes + names for a (lat, lng), or None fields if no hit.

    Coastal points and outlying islands sometimes miss ADM_2 polygons; the
    caller should treat None fields as 'unknown' rather than 'invalid'.
    """
    # Manual bbox overrides FIRST — they handle places GADM 4.1 folds into a
    # parent country (HK → CHN, Macau → CHN). Otherwise the GADM lookup below
    # would stamp them with the parent's code.
    for cc, cn, la0, la1, lo0, lo1 in _BBOX_OVERRIDES:
        if la0 <= lat <= la1 and lo0 <= lng <= lo1:
            return {
                "country_code": cc,
                "country_name": cn,
                "admin1_code": None,
                "admin1_name": None,
                "admin2_code": None,
                "admin2_name": None,
            }
    if _GADM is None:
        load_gadm()
    pt = Point(lng, lat)
    hits = list(_GADM.sindex.query(pt, predicate="intersects"))
    for idx in hits:
        row = _GADM.iloc[idx]
        if row.geometry.contains(pt):
            return {
                "country_code": _norm_gid0(row.get("GID_0")),
                "country_name": row.get("COUNTRY"),
                "admin1_code": row.get("GID_1"),
                "admin1_name": row.get("NAME_1"),
                "admin2_code": row.get("GID_2"),
                "admin2_name": row.get("NAME_2"),
            }
    # ADM_2 miss → fall back to ADM_1 (country + admin1; no admin2 here).
    if _GADM1 is None:
        load_gadm1()
    hits1 = list(_GADM1.sindex.query(pt, predicate="intersects"))
    for idx in hits1:
        row = _GADM1.iloc[idx]
        if row.geometry.contains(pt):
            return {
                "country_code": _norm_gid0(row.get("GID_0")),
                "country_name": row.get("COUNTRY"),
                "admin1_code": row.get("GID_1"),
                "admin1_name": row.get("NAME_1"),
                "admin2_code": None,
                "admin2_name": None,
            }
    # ADM_1 miss → fall back to ADM_0 (country-only outline). Catches tiny
    # territories GADM stores only at ADM_0 — Cocos Keeling (CCK), Christmas
    # Island (CXR) — that have no ADM_1/ADM_2 subdivisions.
    if _GADM0 is None:
        load_gadm0()
    hits0 = list(_GADM0.sindex.query(pt, predicate="intersects"))
    for idx in hits0:
        row = _GADM0.iloc[idx]
        if row.geometry.contains(pt):
            return {
                "country_code": _norm_gid0(row.get("GID_0")),
                "country_name": row.get("COUNTRY"),
                "admin1_code": None,
                "admin1_name": None,
                "admin2_code": None,
                "admin2_name": None,
            }
    return {
        "country_code": None,
        "country_name": None,
        "admin1_code": None,
        "admin1_name": None,
        "admin2_code": None,
        "admin2_name": None,
    }


def _norm_gid0(gid0: Optional[str]) -> Optional[str]:
    """GADM GID_0 is ISO-3 alpha for normal countries but uses Z-prefixed
    placeholder codes for a few disputed regions (e.g. Z01/Z07 ≈ India-
    administered Kashmir). Those fragment a country's signal across phantom
    'countries'. Map the known ones back to their parent ISO-3 code."""
    if gid0 is None:
        return None
    return _GID0_REMAP.get(gid0, gid0)


# Disputed-region GADM placeholder → parent ISO-3, verified against the GADM
# 4.1 ADM_1 table (NAME_1 in comments). NOT all India — they span the
# India/China/Pakistan Kashmir & Himalaya disputes.
_GID0_REMAP = {
    "Z01": "IND",  # Jammu and Kashmir
    "Z02": "CHN",  # Xinjiang Uygur
    "Z03": "CHN",  # Xinjiang Uygur / Xizang
    "Z04": "IND",  # Himachal Pradesh
    "Z05": "IND",  # Uttarakhand
    "Z06": "PAK",  # Azad Kashmir / Gilgit-Baltistan
    "Z07": "IND",  # Arunachal Pradesh
    "Z08": "CHN",  # Xizang
    "Z09": "IND",  # Himachal Pradesh / Uttarakhand
}
