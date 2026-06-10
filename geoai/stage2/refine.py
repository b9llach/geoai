"""Stage 2 orchestrator: extract → pinpoint → geocode → decide.

Pipeline:
    image + Stage 1 (lat, lng, hub)
        ↓
    Surya OCR → script ID → NLLB translation     [extract.py]
        ↓
    Qwen2.5-VL refinement (image + OCR + S1 hub) [pinpoint.py]
        ↓                                                            ↓
    if VLM gave a queryable_name with conf ≥ MIN_CONF                │
      → look it up in local Nominatim          [geocode.py]          │
      → on hit (with good importance): use those coords              │
      → on miss: try alternate_queries                               │
      → on still-miss: fall through to VLM's refined_lat/lng         │
    else: skip geocode                                               │
        ↓                                                            │
    if VLM precision ≥ neighborhood AND conf ≥ MIN_CONF              │
      → use VLM coords                                               │
    else                                                             │
      → use Stage 1 coords (never regress)  ←──────────────────────┘
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
from PIL import Image

from geoai.processing import reverse_geo
from geoai.stage1.predict import haversine_km
from geoai.stage2.extract import (
    ExtractResult,
    extract_from_image,
    extract_from_tiles,
)
from geoai.stage2.geocode import GeocodeHit, geocode_first_hit
from geoai.stage2.pinpoint import (
    PinpointResult,
    PRECISION_LEVELS,
    pinpoint,
)

log = logging.getLogger(__name__)


# Only these precision levels may OVERRIDE Stage 1. "city" was removed
# deliberately: at city precision the VLM is telling us it can't name a
# specific point — and geocoding a bare city name returns the city CENTROID,
# which is almost never where the pano is. Most Street View panos sit in the
# outskirts/suburbs, so snapping to the centre usually moves us AWAY from
# Stage 1's real coordinate (the "outskirts of Geneva → Genève centroid" bug).
# At city-or-coarser precision we keep Stage 1's point.
TRUSTED_PRECISIONS = {"street", "neighborhood"}
MIN_CONFIDENCE = 0.7

# A geocode hit is only a useful REFINEMENT if it's a POINT FEATURE (a specific
# building / amenity / landmark / address / named street). When the match is an
# AREA — an administrative boundary or a settlement (`place=city/town/...`) —
# its coordinate is just the area's centroid, which we must NOT treat as a
# precise location. These are the OSM category / type tokens that mark an area.
_CENTROID_CATEGORIES = {"boundary"}
_CENTROID_PLACE_TYPES = {
    "city", "town", "village", "hamlet", "suburb", "quarter", "neighbourhood",
    "municipality", "county", "state", "province", "region", "country",
    "district", "borough", "city_district", "administrative", "political",
}
# Reject point-feature hits farther than this from Stage 1 — at that distance
# it's almost certainly a same-name match in the wrong place (e.g. a "Springfield"
# in the wrong state), not a genuine local refinement of Stage 1's guess.
MAX_GEOCODE_DRIFT_KM = 75.0


def _is_area_centroid(place_type: str) -> bool:
    """True when an OSM `category=type` string denotes an AREA (whose geocoded
    coordinate is just a centroid), not a point feature we can pinpoint to."""
    category, _, typ = place_type.partition("=")
    return category in _CENTROID_CATEGORIES or typ in _CENTROID_PLACE_TYPES

# Hard wall-clock cap on the VLM call — the backstop against a thinking-mode
# spiral. The tight pinpoint prompt keeps Gemma bounded, but feeding multiple
# high-res tiles grows the reasoning (more to look at): a single strip finishes
# ~8 s, 4 tiles ~20 s, 8 tiles ~30-40 s. Cap generously so the tiled path isn't
# clipped, while still aborting a true runaway. Override via env.
PINPOINT_TIMEOUT = float(os.environ.get("GEOAI_PINPOINT_TIMEOUT", "60.0"))
# How many of the rendered OCR tiles to also show Gemma (it sees small detail —
# flags, distant signs, plates — that the downscaled strip loses). 8 = full
# 360° overlap coverage; lower (e.g. 4 cardinals) is faster. 0 = strip only.
PINPOINT_NUM_TILES = int(os.environ.get("GEOAI_PINPOINT_NUM_TILES", "8"))
# Drop Nominatim hits below this. OSM uses `importance` for landmark rank
# (Eiffel Tower ~0.9, town center ~0.4) but most business chain branches
# get importance=0.000 by default — they're real coords, just not famous.
# Keep this at 0.0 so we don't filter out branch hits, which are the
# common case for our VLM-produced queryable_names.
MIN_GEOCODE_IMPORTANCE = 0.0


@dataclass
class RefineResult:
    final_lat: float
    final_lng: float
    used_refinement: bool
    effective_precision: str
    # Which source produced the final coords. Only ever "stage1" (deferred) or
    # "nominatim" (OSM-grounded point feature). The VLM never sets coords
    # directly — it only proposes the queryable_name that Nominatim resolves.
    source: str  # "stage1" | "nominatim"
    pinpoint: Optional[PinpointResult]
    extract: Optional[ExtractResult]
    geocode_hit: Optional[GeocodeHit]
    extract_seconds: float = 0.0
    pinpoint_seconds: float = 0.0
    geocode_seconds: float = 0.0
    total_seconds: float = 0.0
    # Why the wrapper made the trust/fall-back call (internal trace).
    decision_reason: str = ""

    @property
    def explanation(self) -> str:
        """One-line human-readable summary suitable for showing to a user
        in a tooltip / userscript HUD. Concatenates the VLM's reasoning
        with the geocoder match name when available."""
        parts: list[str] = []
        if self.pinpoint and self.pinpoint.reasoning:
            parts.append(self.pinpoint.reasoning)
        if self.geocode_hit:
            parts.append(
                f"Matched in OSM: {self.geocode_hit.display_name.split(',')[0]} "
                f"(via query \"{self.geocode_hit.query}\")"
            )
        if not self.used_refinement:
            parts.append(f"Refinement skipped — {self.decision_reason}")
        return " · ".join(parts) if parts else "(no explanation)"


def _country_code_from_pinpoint(pp: PinpointResult) -> Optional[str]:
    """Best-effort ISO-2 country code from the VLM's confirmed_country.
    Used to scope Nominatim's search. Returns None on unknown."""
    if not pp.confirmed_country or pp.confirmed_country == "unknown":
        return None
    # Tiny mapping for common cases; Nominatim handles the rest via name.
    name = pp.confirmed_country.lower()
    NAMES = {
        "cambodia": "kh", "thailand": "th", "laos": "la", "myanmar": "mm",
        "vietnam": "vn", "south korea": "kr", "north korea": "kp",
        "japan": "jp", "china": "cn", "taiwan": "tw", "singapore": "sg",
        "india": "in", "indonesia": "id", "philippines": "ph",
        "malaysia": "my", "russia": "ru", "ukraine": "ua",
        "belarus": "by", "kazakhstan": "kz", "mongolia": "mn",
        "iran": "ir", "iraq": "iq", "israel": "il", "saudi arabia": "sa",
        "turkey": "tr", "greece": "gr", "italy": "it", "spain": "es",
        "france": "fr", "germany": "de", "united kingdom": "gb",
        "great britain": "gb", "ireland": "ie", "netherlands": "nl",
        "belgium": "be", "switzerland": "ch", "austria": "at",
        "poland": "pl", "czech republic": "cz", "czechia": "cz",
        "hungary": "hu", "romania": "ro", "bulgaria": "bg",
        "serbia": "rs", "croatia": "hr", "portugal": "pt",
        "sweden": "se", "norway": "no", "finland": "fi", "denmark": "dk",
        "iceland": "is", "estonia": "ee", "latvia": "lv", "lithuania": "lt",
        "united states": "us", "usa": "us", "america": "us",
        "canada": "ca", "mexico": "mx", "panama": "pa",
        "costa rica": "cr", "guatemala": "gt", "honduras": "hn",
        "nicaragua": "ni", "el salvador": "sv",
        "brazil": "br", "argentina": "ar", "chile": "cl",
        "peru": "pe", "colombia": "co", "venezuela": "ve",
        "ecuador": "ec", "uruguay": "uy", "paraguay": "py", "bolivia": "bo",
        "south africa": "za", "kenya": "ke", "ethiopia": "et",
        "egypt": "eg", "morocco": "ma", "tunisia": "tn",
        "australia": "au", "new zealand": "nz",
    }
    return NAMES.get(name)


def refine(
    image: Image.Image,
    stage1_lat: float,
    stage1_lng: float,
    stage1_hub: str,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    trusted_precisions: set[str] = TRUSTED_PRECISIONS,
    use_geocoder: bool = True,
    on_progress: Optional[Callable[[str, dict], None]] = None,
    ocr_tiles: Optional[list] = None,
) -> RefineResult:
    """Run extract + pinpoint + (optional) geocode, then decide. Never
    regresses past Stage 1's coords.

    Args:
        on_progress: optional callback invoked at each stage boundary as
            `on_progress(event_name, data_dict)`. Used by the streaming
            API endpoint to push live status updates to the client.
            Events emitted (in order): extract_start, extract_done,
            pinpoint_start, pinpoint_done, geocode_start, geocode_done,
            done.
    """
    def _emit(event: str, **data) -> None:
        if on_progress is not None:
            try:
                on_progress(event, data)
            except Exception:
                log.exception("on_progress callback raised; ignoring")

    t0 = time.time()
    extract_seconds = 0.0
    pinpoint_seconds = 0.0
    geocode_seconds = 0.0

    # Extract. When the caller provides perspective tiles (rendered from
    # the panorama at 1024x1024 each, 8 overlapping headings), Surya gets
    # square aspect-ratio inputs at full per-tile resolution + redundant
    # coverage — substantially better recall than feeding the 3072x768
    # strip (which Surya downscales aggressively because of the 4:1 AR).
    _emit("extract_start", tiles=(len(ocr_tiles) if ocr_tiles else 0))
    t = time.time()
    if ocr_tiles:
        extract = extract_from_tiles(ocr_tiles)
    else:
        extract = extract_from_image(image)
    extract_seconds = time.time() - t
    _emit("extract_done",
          script=(extract.language_name if extract else None),
          raw_text=(extract.raw_text[:200] if extract else None),
          translation=(extract.english_translation[:200] if extract else None),
          elapsed=extract_seconds)

    # FAST-PATH: no readable text in the image → skip the VLM entirely.
    # Rural / featureless scenes have no exploitable signal for refinement,
    # and the Thinking VLM will burn ~30-60s trying to come up with
    # something before giving up. Just take Stage 1's prediction now.
    if extract is None:
        _emit("done", used_refinement=False, reason="no_text",
              final_lat=stage1_lat, final_lng=stage1_lng)
        return RefineResult(
            final_lat=stage1_lat, final_lng=stage1_lng,
            used_refinement=False, effective_precision="country",
            source="stage1", pinpoint=None, extract=None,
            geocode_hit=None,
            extract_seconds=extract_seconds,
            total_seconds=time.time() - t0,
            decision_reason=("no readable text in image — skipping VLM, "
                             "keeping Stage 1's prediction"),
        )

    # Pinpoint — with a hard wall-clock timeout to bound Thinking-mode
    # spirals on ambiguous inputs. If the VLM hasn't returned in this
    # many seconds, abort and fall back to Stage 1 cleanly.
    _emit("pinpoint_start")
    t = time.time()
    # Bridge pinpoint's token-stream events into refine's progress channel.
    # Each Ollama chunk → "pinpoint_<event>" up the stack → "stage2_pinpoint_<event>"
    # in the API stream → live token-count + latest-text in the userscript.
    def _on_pinpoint_event(event_name: str, data: dict) -> None:
        _emit("pinpoint_" + event_name, **data)

    # Show Gemma high-res tiles (small detail the strip loses) when we have
    # them. Evenly subsample to PINPOINT_NUM_TILES so a lower setting keeps
    # full 360° spread (e.g. 4 → cardinals) rather than one side of the view.
    vlm_images = None
    if ocr_tiles and PINPOINT_NUM_TILES > 0:
        n = min(PINPOINT_NUM_TILES, len(ocr_tiles))
        step = len(ocr_tiles) / n
        vlm_images = [ocr_tiles[int(i * step)] for i in range(n)]
    try:
        pp = pinpoint(image, stage1_lat, stage1_lng, stage1_hub,
                      extract=extract, images=vlm_images,
                      timeout=PINPOINT_TIMEOUT,
                      on_progress=_on_pinpoint_event)
    except requests.exceptions.Timeout:
        pinpoint_seconds = time.time() - t
        _emit("pinpoint_timeout", elapsed=pinpoint_seconds,
              limit_seconds=PINPOINT_TIMEOUT)
        _emit("done", used_refinement=False, reason="vlm_timeout",
              final_lat=stage1_lat, final_lng=stage1_lng)
        return RefineResult(
            final_lat=stage1_lat, final_lng=stage1_lng,
            used_refinement=False, effective_precision="country",
            source="stage1", pinpoint=None, extract=extract,
            geocode_hit=None,
            extract_seconds=extract_seconds,
            pinpoint_seconds=pinpoint_seconds,
            total_seconds=time.time() - t0,
            decision_reason=(f"VLM exceeded {PINPOINT_TIMEOUT}s timeout — "
                             "fell back to Stage 1"),
        )
    pinpoint_seconds = time.time() - t
    _emit("pinpoint_done",
          precision=pp.precision_level,
          confidence=pp.confidence,
          queryable=pp.queryable_name,
          confirmed_country=pp.confirmed_country,
          elapsed=pinpoint_seconds)

    # Early outs that bypass the geocoder.
    if pp.precision_level not in trusted_precisions:
        return RefineResult(
            final_lat=stage1_lat, final_lng=stage1_lng,
            used_refinement=False, effective_precision="country",
            source="stage1", pinpoint=pp, extract=extract,
            geocode_hit=None,
            extract_seconds=extract_seconds,
            pinpoint_seconds=pinpoint_seconds,
            total_seconds=time.time() - t0,
            decision_reason=(f"VLM precision={pp.precision_level} too coarse"),
        )
    if pp.confidence < min_confidence:
        return RefineResult(
            final_lat=stage1_lat, final_lng=stage1_lng,
            used_refinement=False, effective_precision="country",
            source="stage1", pinpoint=pp, extract=extract,
            geocode_hit=None,
            extract_seconds=extract_seconds,
            pinpoint_seconds=pinpoint_seconds,
            total_seconds=time.time() - t0,
            decision_reason=(f"VLM conf {pp.confidence:.2f} < {min_confidence}"),
        )

    # Try the geocoder if we have a queryable_name.
    hit: Optional[GeocodeHit] = None
    if use_geocoder and pp.queryable_name:
        queries = [pp.queryable_name, *pp.alternate_queries]
        cc = _country_code_from_pinpoint(pp)
        _emit("geocode_start", queries=queries, country_code=cc)
        t = time.time()
        hit = geocode_first_hit(
            queries, country_code=cc,
            min_importance=MIN_GEOCODE_IMPORTANCE,
        )
        geocode_seconds = time.time() - t
        _emit("geocode_done",
              hit=(hit.display_name if hit else None),
              query=(hit.query if hit else None),
              lat=(hit.lat if hit else None),
              lng=(hit.lng if hit else None),
              elapsed=geocode_seconds)

    if hit is not None:
        # COUNTRY-SANITY CHECK. Stage 1's country head is ~99% accurate.
        # If the geocoded hit lands in a different country, the VLM almost
        # certainly built a misleading queryable (e.g., "Walmart" without
        # geo context returned a US Walmart) or matched a same-name place
        # in a different country. Reject the hit and fall back to Stage 1.
        hit_geo = reverse_geo.reverse_geocode(hit.lat, hit.lng)
        hit_cc = (hit_geo.get("country_code") or "").upper()
        s1_hub_lower = stage1_hub.lower()
        # We don't have stage1's ISO-3 directly, but the hub string ends
        # in the country name; check it doesn't contradict the hit's
        # country name. Both come from the same GADM, so a country name
        # mismatch is a strong rejection signal.
        s1_country_name = (
            stage1_hub.rsplit(",", 1)[-1].strip().lower()
            if "," in stage1_hub else stage1_hub.strip().lower()
        )
        hit_country_name = (hit_geo.get("country_name") or "").lower()
        if hit_country_name and s1_country_name and (
            hit_country_name != s1_country_name
            and s1_country_name not in hit_country_name
            and hit_country_name not in s1_country_name
        ):
            log.warning(
                "stage2: country mismatch — Stage 1 says %r, geocoder hit is in %r. "
                "Rejecting hit (query=%r); falling back to Stage 1.",
                s1_country_name, hit_country_name, hit.query,
            )
            # Treat as if geocoder missed; fall through to VLM-coords or Stage 1.
            hit = None

    if hit is not None and _is_area_centroid(hit.place_type):
        # The match is an area (city/region/admin boundary), so its coordinate
        # is a centroid — not where the pano is. Snapping to it would override
        # Stage 1's real point with a city centre. Keep Stage 1 instead.
        log.info(
            "stage2: geocode hit %r is an area centroid (%s), not a point "
            "feature — keeping Stage 1's coordinate.",
            hit.query, hit.place_type,
        )
        _emit("geocode_rejected", reason="area_centroid",
              query=hit.query, place_type=hit.place_type)
        hit = None

    if hit is not None:
        drift_km = haversine_km(stage1_lat, stage1_lng, hit.lat, hit.lng)
        if drift_km > MAX_GEOCODE_DRIFT_KM:
            # A point feature this far from Stage 1 is almost certainly a
            # same-name match in the wrong place, not a refinement.
            log.info(
                "stage2: geocode hit %r is %.0f km from Stage 1 (> %.0f) — "
                "rejecting as a likely wrong-place match.",
                hit.query, drift_km, MAX_GEOCODE_DRIFT_KM,
            )
            _emit("geocode_rejected", reason="too_far",
                  query=hit.query, drift_km=round(drift_km, 1))
            hit = None

    if hit is not None:
        return RefineResult(
            final_lat=hit.lat, final_lng=hit.lng,
            used_refinement=True, effective_precision=pp.precision_level,
            source="nominatim", pinpoint=pp, extract=extract,
            geocode_hit=hit,
            extract_seconds=extract_seconds,
            pinpoint_seconds=pinpoint_seconds,
            geocode_seconds=geocode_seconds,
            total_seconds=time.time() - t0,
            decision_reason=(f"Nominatim hit '{hit.query}' ({hit.place_type}, "
                             f"importance {hit.importance:.2f})"),
        )

    # No usable geocode hit → DEFER TO STAGE 1. We deliberately do NOT fall
    # back to the VLM's own refined_lat/lng: that coordinate is the VLM's
    # parametric-memory guess (Phase-1 design, see pinpoint.py), ungrounded and
    # often inconsistent with its own reasoning — e.g. it "reasons Crowsnest
    # Pass" but emits a Calgary coordinate. Stage 2 only ever MOVES the pin to
    # an OSM-grounded point feature near Stage 1; otherwise Stage 1 stands.
    return RefineResult(
        final_lat=stage1_lat, final_lng=stage1_lng,
        used_refinement=False, effective_precision="country",
        source="stage1", pinpoint=pp, extract=extract,
        geocode_hit=None,
        extract_seconds=extract_seconds,
        pinpoint_seconds=pinpoint_seconds,
        geocode_seconds=geocode_seconds,
        total_seconds=time.time() - t0,
        decision_reason=("VLM trusted but geocoder missed and no VLM coord "
                         "delta vs Stage 1"),
    )
