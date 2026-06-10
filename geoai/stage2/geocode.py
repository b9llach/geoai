"""Thin client for the local Nominatim instance.

We use Nominatim only as a name → (lat, lng) resolver for the VLM's
`queryable_name` field. The VLM has already done the hard part
(turning OCR noise + visual cues into a canonical place string); the
geocoder's job is just to look that string up in OSM.

Two tiers of result quality:
    - `importance` ≥ 0.4    most landmarks, named businesses, big roads
    - `importance` < 0.4    ambiguous, generic names; caller may distrust

The VLM emits an `alternate_queries` list; the caller is expected to try
queryable_name first, then walk the alternates and return the first hit.

By default this points at OSM's PUBLIC Nominatim, which enforces a strict
usage policy: at most 1 request/second and a real identifying User-Agent.
We honour that with a process-wide throttle (`_throttle`) that spaces every
outbound request by `MIN_INTERVAL`. Point `GEOAI_NOMINATIM_URL` at a
self-hosted instance to lift the limit (set `GEOAI_NOMINATIM_MIN_INTERVAL=0`).
On a miss / down / error the calls just return None.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import requests

log = logging.getLogger(__name__)

# Default to OSM's free public Nominatim. Override with GEOAI_NOMINATIM_URL to
# use a self-hosted instance (and set GEOAI_NOMINATIM_MIN_INTERVAL=0 there).
NOMINATIM_URL = os.environ.get(
    "GEOAI_NOMINATIM_URL", "https://nominatim.openstreetmap.org"
)

# Public-API rate limit: max 1 req/sec. We space by a hair over 1 s to stay
# safely under it even with clock jitter. Set to 0 for a self-hosted instance.
MIN_INTERVAL = float(os.environ.get("GEOAI_NOMINATIM_MIN_INTERVAL", "1.1"))
_throttle_lock = threading.Lock()
_last_request_ts = 0.0


def _throttle() -> None:
    """Block until at least MIN_INTERVAL has elapsed since the previous
    request, then stamp 'now'. Process-wide and thread-safe so concurrent
    refine calls can't burst past the public API's 1 req/sec ceiling."""
    global _last_request_ts
    if MIN_INTERVAL <= 0:
        return
    with _throttle_lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.monotonic()

# OSM's public Nominatim requires a User-Agent identifying the application.
# Our self-hosted instance ignores it but it costs nothing to send. Override
# via GEOAI_GEOCODE_UA if you want to identify a specific deployment.
USER_AGENT = os.environ.get(
    "GEOAI_GEOCODE_UA",
    "geomllm-stage2/0.1 (https://github.com/local; contact: local)",
)


@dataclass
class GeocodeHit:
    """A single Nominatim result, normalized.

    `importance` is Nominatim's PageRank-derived ranking (0..1). Higher
    means the OSM object is more prominent — the Eiffel Tower is ~0.9,
    a corner cafe is ~0.05.

    `place_type` is the OSM key=value pair (e.g. "amenity=bank",
    "place=city"). The caller can use this to filter — e.g. drop
    `boundary=administrative` matches when the VLM was hoping for a
    specific business.
    """
    lat: float
    lng: float
    display_name: str
    importance: float
    place_type: str
    osm_id: Optional[int]
    query: str  # the query string that produced this hit


def _is_up() -> bool:
    """Cheap status check — used by callers to decide whether to skip
    geocoding entirely when the container isn't ready yet."""
    try:
        _throttle()
        r = requests.get(f"{NOMINATIM_URL}/status",
                         headers={"User-Agent": USER_AGENT}, timeout=5.0)
        return r.status_code == 200
    except requests.RequestException:
        return False


def geocode(query: str, *,
            country_code: Optional[str] = None,
            min_importance: float = 0.0,
            limit: int = 1,
            timeout: float = 10.0) -> Optional[GeocodeHit]:
    """Look up a query string in the local Nominatim. Returns the top hit
    (filtered by min_importance) or None on miss / down / error.

    Args:
        query:           the VLM's queryable_name (or an alternate).
        country_code:    optional ISO-2 (e.g. "kh", "br") to anchor the
                         search. The VLM's queryable_name USUALLY includes
                         the country so this is rarely needed.
        min_importance:  drop hits below this score. 0 = accept anything;
                         0.3 = drop noise; 0.6 = landmarks only.
        limit:           Nominatim returns at most N; we always pick top.
        timeout:         HTTP timeout in seconds.
    """
    if not query.strip():
        return None
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": str(limit),
        "addressdetails": "0",
        "extratags": "0",
    }
    if country_code:
        params["countrycodes"] = country_code.lower()
    _throttle()
    try:
        r = requests.get(f"{NOMINATIM_URL}/search", params=params,
                         headers={"User-Agent": USER_AGENT},
                         timeout=timeout)
        r.raise_for_status()
        results = r.json()
    except requests.RequestException as e:
        log.warning("geocode: HTTP error for %r: %s", query, e)
        return None
    except ValueError as e:  # bad JSON
        log.warning("geocode: bad JSON for %r: %s", query, e)
        return None

    for obj in results:
        importance = float(obj.get("importance", 0.0) or 0.0)
        if importance < min_importance:
            continue
        try:
            lat = float(obj["lat"])
            lng = float(obj["lon"])
        except (KeyError, ValueError, TypeError):
            continue
        place_type = (
            f"{obj.get('category', '?')}={obj.get('type', '?')}"
        )
        return GeocodeHit(
            lat=lat, lng=lng,
            display_name=str(obj.get("display_name", "")),
            importance=importance,
            place_type=place_type,
            osm_id=obj.get("osm_id"),
            query=query,
        )
    return None


def geocode_first_hit(queries: Iterable[str], *,
                      country_code: Optional[str] = None,
                      min_importance: float = 0.0) -> Optional[GeocodeHit]:
    """Walk the candidate queries in order, return the first that hits
    Nominatim with sufficient importance. Used by refine.py to consume
    the VLM's primary + alternate_queries list."""
    for q in queries:
        hit = geocode(q, country_code=country_code,
                      min_importance=min_importance)
        if hit is not None:
            return hit
    return None
