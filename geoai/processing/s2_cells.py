"""S2 cell IDs at the four levels Stage 1 conditions on.

Level guide (approx side at the equator):
    L3   ≈  1300 km   (continental)
    L6   ≈   165 km   (state-sized)
    L9   ≈    20 km   (city-sized)
    L12  ≈     2.5 km (neighborhood)
"""
from __future__ import annotations

from typing import Iterable

import s2sphere

DEFAULT_LEVELS: tuple[int, ...] = (3, 6, 9, 12)


def s2_cells_for_point(
    lat: float, lng: float, levels: Iterable[int] = DEFAULT_LEVELS
) -> dict[int, int]:
    """Return {level: 64-bit cell id} for the leaf cell at (lat, lng)."""
    cell = s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(lat, lng))
    return {lvl: cell.parent(lvl).id() for lvl in levels}
