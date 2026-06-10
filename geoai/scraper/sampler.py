"""Uniform random sampling inside a polygon, latitude-corrected.

Earth surface-area-uniform sampling: draw latitude from its sine (to correct
for the cos(lat) area distortion of a naive uniform-in-lat draw), longitude
uniform. Rejection-sample against the polygon.
"""
from __future__ import annotations

import math
import random
from typing import Iterator, Union

from shapely.geometry import MultiPolygon, Point, Polygon

PolyLike = Union[Polygon, MultiPolygon]


def random_point_in_polygon(poly: PolyLike) -> tuple[float, float]:
    minx, miny, maxx, maxy = poly.bounds
    sin_min = math.sin(math.radians(miny))
    sin_max = math.sin(math.radians(maxy))
    while True:
        u = random.random()
        lat = math.degrees(math.asin(sin_min + u * (sin_max - sin_min)))
        lng = random.uniform(minx, maxx)
        if poly.contains(Point(lng, lat)):
            return lat, lng


def iter_samples(poly: PolyLike, n: int) -> Iterator[tuple[float, float]]:
    for _ in range(n):
        yield random_point_in_polygon(poly)
