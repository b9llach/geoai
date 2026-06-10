"""Wrapper around streetlevel with retries, filters, and rate limiting.

streetlevel hits internal unauthenticated Google endpoints (`cbk0.google.com`
and `GeoPhotoService`). These tolerate far more traffic than the JS SDK's
keyed quota, but rate limits are IP-based — default to 4 concurrent requests
and exponentially back off on failure.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from aiohttp import ClientSession, ClientTimeout
from streetlevel import streetview
from streetlevel.streetview.panorama import StreetViewPanorama

log = logging.getLogger(__name__)


@dataclass
class PanoFilter:
    official_only: bool = True
    min_camera_gen: int = 2  # Skip Gen1 (visually degraded)
    reject_trekkers: bool = True
    date_after: Optional[str] = None   # YYYY-MM
    date_before: Optional[str] = None


def camera_generation(pano: StreetViewPanorama) -> int:
    """Infer camera generation from full-res world height (per tzhf's scheme).

    Returns 1, 23, 4, or 0 (unknown).
    """
    height = pano.image_sizes[-1].y if pano.image_sizes else 0
    if height == 1664:
        return 1
    if height == 6656:
        return 23
    if height == 8192:
        return 4
    return 0


def passes_filter(pano: StreetViewPanorama, f: PanoFilter) -> bool:
    if f.official_only:
        # 22-char pano IDs are official car captures; longer IDs are
        # user-uploaded photospheres. A non-None `uploader` is another tell.
        if len(pano.id) != 22 or getattr(pano, "uploader", None) is not None:
            return False
    gen = camera_generation(pano)
    if gen and gen < f.min_camera_gen:
        return False
    if f.reject_trekkers:
        # streetlevel exposes `source`: 'launch' = car, 'scout' = trekker/backpack.
        # Anything other than 'launch' is a non-vehicle capture for our purposes.
        src = getattr(pano, "source", None)
        if src is not None and src != "launch":
            return False
    date = str(pano.date) if pano.date else None
    if f.date_after and date and date < f.date_after:
        return False
    if f.date_before and date and date > f.date_before:
        return False
    return True


class StreetViewClient:
    def __init__(self, max_concurrent: int = 4, timeout_s: int = 30):
        self.sem = asyncio.Semaphore(max_concurrent)
        self.timeout = ClientTimeout(total=timeout_s)
        self._session: Optional[ClientSession] = None

    async def __aenter__(self) -> "StreetViewClient":
        self._session = ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()

    async def find_nearest(
        self, lat: float, lng: float, radius_m: int = 50
    ) -> Optional[StreetViewPanorama]:
        """Nearest pano to a point; None if no coverage within radius."""
        assert self._session is not None, "Use `async with StreetViewClient(...)`"
        async with self.sem:
            for attempt in range(3):
                try:
                    return await streetview.find_panorama_async(
                        lat, lng, self._session, radius=radius_m
                    )
                except Exception as e:
                    log.warning(f"find_panorama_async failed (try {attempt}): {e}")
                    await asyncio.sleep(2 ** attempt)
            return None

    async def get_timeline(
        self, pano: StreetViewPanorama
    ) -> list[StreetViewPanorama]:
        """Fetch all historical captures for a pano's location.

        `pano.historical` entries are shallow stubs (no `image_sizes`, depth,
        etc.), so downloading them directly fails. We re-fetch each by ID
        to get a full panorama. Each returned pano is a distinct Street View
        capture at the same geographic location from a different date — the
        single biggest free data multiplier available to us.
        """
        assert self._session is not None
        historical = getattr(pano, "historical", None) or []
        if not historical:
            return [pano]
        result = [pano]
        for h in historical:
            sibling_id = getattr(h, "id", None)
            if not sibling_id or sibling_id == pano.id:
                continue
            async with self.sem:
                try:
                    sib = await streetview.find_panorama_by_id_async(
                        sibling_id, self._session
                    )
                    if sib:
                        result.append(sib)
                except Exception as e:
                    log.warning(f"Timeline fetch failed for {sibling_id}: {e}")
        return result

    async def download_equirect(
        self, pano: StreetViewPanorama, path: str, zoom: int = 3
    ) -> bool:
        """Download and stitch the equirectangular panorama to `path`."""
        assert self._session is not None
        async with self.sem:
            try:
                await streetview.download_panorama_async(
                    pano, path, self._session, zoom=zoom
                )
                return True
            except Exception as e:
                log.error(f"Download failed for {pano.id}: {e}")
                return False
