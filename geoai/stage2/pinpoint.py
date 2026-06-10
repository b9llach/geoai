"""VLM-driven location refinement.

Stage 1 already gets the country right ~99% of the time. The job of
Stage 2 is to NARROW the prediction — go from "somewhere in Cambodia"
to "Norodom Blvd, Phnom Penh" when the image contains actionable cues
(storefront name, landmark, street sign).

The VLM sees three things in one call:
    1. The street-view image (raw pixels — it can read logos, building
       style, vegetation, traffic side, even small signage Surya missed)
    2. Surya's OCR output, with language ID and English translation
    3. Stage 1's country/region guess

It outputs a structured refinement (canonical place name + best-guess
lat/lng + precision level + confidence). The wrapper then decides
whether to *use* the refinement (high confidence + neighborhood-or-better
precision) or *discard* it (low confidence — fall back to Stage 1).

Note: this is the Phase-1 design from the conversation. The VLM emits
lat/lng directly from its training-time geography knowledge. Phase 2
would route the `queryable_name` field through a self-hosted Nominatim
to convert names to coords more reliably.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from PIL import Image

from geoai.config import GEMMA_VL_MODEL, LMSTUDIO_HOST
from geoai.stage2.extract import ExtractResult

log = logging.getLogger(__name__)


PRECISION_LEVELS = ("street", "neighborhood", "city", "region", "country", "none")


@dataclass
class PinpointResult:
    """Output of the refinement VLM call.

    refined_lat / refined_lng are the VLM's own coordinate guess. They
    may be the SAME as Stage 1's coordinates (when the VLM has nothing
    more specific to add) or substantially different (when it recognizes
    a landmark / business chain / neighborhood).

    precision_level tells the wrapper how seriously to trust the
    coordinate refinement. Anything coarser than `neighborhood` is
    treated as "no refinement" by the caller.
    """
    queryable_name: str            # canonical query string for future geocoder
    alternate_queries: list[str]   # backups if queryable_name doesn't hit
    refined_lat: Optional[float]
    refined_lng: Optional[float]
    precision_level: str           # one of PRECISION_LEVELS
    confidence: float              # 0..1
    confirmed_country: str         # VLM-determined country (anchor check)
    reasoning: str = ""
    raw_response: str = ""


_SYSTEM = """You are a Geolocation PINPOINTER. Stage 1's country is \
**99% accurate — trust it and do NOT re-verify**. Your one job is to \
narrow WITHIN that country using the OCR text + image. Think briefly and \
decisively, then emit JSON.

INPUT
  - image (street view crop strip)
  - OCR text + script (from Surya, pre-extracted; trust it)
  - NLLB English translation (often WRONG on brand names — use your own \
brand knowledge instead)
  - Stage 1 country (CORRECT — narrow within it, don't second-guess)

WHAT TO DO
  Find the most specific recognizable thing in the image OR OCR text:
    landmark > brand/chain logo > street name > district name > city name
  Build a geocoder query around it. Brand names beat OCR text beat NLLB \
translations. Don't fabricate.

QUERYABLE_NAME — pick the most specific you can defend:
  - "Sagrada Familia Barcelona Spain"  (named landmark)
  - "ABA Bank Phnom Penh Cambodia"     (named brand + city)
  - "Independence Square Lusaka Zambia"
  - "Norodom Boulevard Phnom Penh"     (named street)

NLLB CAVEAT — DON'T USE NLLB OUTPUT FOR PROPER NOUNS:
  - Khmer "ធនាគារវឌ្ឍនៈអាស៊ី" is **ABA Bank / Advanced Bank of Asia**, \
NOT "Asian Development Bank" as NLLB says. Trust your brand knowledge.

ALTERNATE_QUERIES — exactly 3 entries, each MUST be present:
  1. Same brand/place, different phrasing (full vs short form, etc.)
  2. Raw OCR text + country in English (OSM may only have local names)
  3. Generic category + city ("bank Phnom Penh", "restaurant Bangkok") \
— mandatory safety net; lands on SOME building of the right category in \
the right city when brand-specific lookups miss.

PRECISION_LEVEL — pick the most specific evidence supports:
  - "street"       ±50m — named landmark or unique business at known address
  - "neighborhood" ±2km — district name or recognized brand chain
  - "city"         ±15km — image has any text in the country's script AND \
shows urban context (storefront / signage / streetlights). Default to the \
country's capital coords if you can't pin the city — most Street View urban \
panos are in the capital.
  - "region"       ±100km — broad province / state
  - "country"      use ONLY when image is featureless rural + no text
  - "none"         image degraded / unreadable

CONFIDENCE (0.0-1.0):
  famous landmark .95  ·  brand-chain match .80-.90  ·  capital-city scene \
.70-.80  ·  generic urban .60-.70  ·  rural no-markers .30  ·  degraded .10
Caller uses refinement only if confidence ≥ 0.7. Being TOO cautious is \
just as bad as overconfident — both give the user a Stage-1-only answer.

THINKING BUDGET — KEEP IT SHORT.
You don't need to verify the country, debate translation accuracy, or \
re-derive what you see in the image. Identify the thing, pick the query, \
emit JSON. 2-4 sentences of thinking total is enough.

OUTPUT — JSON only, no prose around it:
{
  "queryable_name": "string",
  "alternate_queries": ["string", "string", "string"],
  "refined_lat": 0.0,
  "refined_lng": 0.0,
  "precision_level": "street|neighborhood|city|region|country|none",
  "confidence": 0.00,
  "confirmed_country": "string (same as Stage 1 unless you're 95% sure it's wrong)",
  "reasoning": "one short sentence on what you identified"
}"""


def _image_to_data_url(image: Image.Image) -> str:
    """Encode as a JPEG data URL for the OpenAI-compatible image_url field.

    JPEG (not PNG) keeps the payload small — these are photographic panorama
    strips where lossless buys nothing but megabytes, and a smaller upload
    means fewer vision tokens for Gemma to chew through.
    """
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _build_user_prompt(lat: float, lng: float, hub: str,
                       extract: Optional[ExtractResult]) -> str:
    parts = [
        "STAGE 1 COARSE GUESS:",
        f"- Latitude: {lat:.6f}",
        f"- Longitude: {lng:.6f}",
        f"- Country/Region: {hub}",
        "",
    ]
    if extract is not None:
        parts.extend([
            "OCR (do not re-run; correct obvious slips silently):",
            f"- Detected Script/Language: {extract.language_name}",
            f"- Raw text: {extract.raw_text}",
            f"- English translation: \"{extract.english_translation}\"",
            "",
        ])
    else:
        parts.extend([
            "OCR: (no readable text in image)",
            "",
        ])
    parts.append(
        "Identify the most specific recognizable feature and emit the JSON "
        "block per the schema. Remember: be conservative on confidence; if "
        "you can only say 'somewhere in Cambodia', use precision_level "
        "'country' and Stage 1's coordinates verbatim."
    )
    return "\n".join(parts)


_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(
    r"(\{[^{}]*\"precision_level\"[^{}]*\})", re.DOTALL,
)


def _parse(raw: str, fallback_lat: float, fallback_lng: float,
           fallback_hub: str) -> PinpointResult:
    """Parse the VLM's JSON. With `format=json` set on the Ollama call the
    content IS the JSON object string (no fences). We also tolerate the
    legacy ```json fenced form for compatibility with non-format models."""
    obj = None
    raw_stripped = raw.strip()
    # Try direct-JSON first (the format=json path).
    if raw_stripped.startswith("{"):
        try:
            obj = json.loads(raw_stripped)
        except json.JSONDecodeError:
            obj = None
    # Fall back to the legacy fenced-block path.
    if obj is None:
        m = _JSON_FENCE_RE.search(raw) or _BARE_JSON_RE.search(raw)
        if m is None:
            log.warning("pinpoint: no JSON in VLM response; falling back")
            return PinpointResult(
                queryable_name="", alternate_queries=[],
                refined_lat=fallback_lat, refined_lng=fallback_lng,
                precision_level="none", confidence=0.0,
                confirmed_country=fallback_hub.split(",")[-1].strip() or "unknown",
                reasoning="verifier returned no JSON",
                raw_response=raw,
            )
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            log.warning("pinpoint: bad JSON (%s); falling back", e)
            return PinpointResult(
                queryable_name="", alternate_queries=[],
                refined_lat=fallback_lat, refined_lng=fallback_lng,
                precision_level="none", confidence=0.0,
                confirmed_country="unknown", reasoning=f"JSON decode error: {e}",
                raw_response=raw,
            )
    pl = str(obj.get("precision_level", "none") or "none").lower()
    if pl not in PRECISION_LEVELS:
        pl = "none"
    return PinpointResult(
        queryable_name=str(obj.get("queryable_name", "") or ""),
        alternate_queries=list(obj.get("alternate_queries") or []),
        refined_lat=_to_float(obj.get("refined_lat"), fallback_lat),
        refined_lng=_to_float(obj.get("refined_lng"), fallback_lng),
        precision_level=pl,
        confidence=float(obj.get("confidence", 0.0) or 0.0),
        confirmed_country=str(obj.get("confirmed_country", "") or "unknown"),
        reasoning=str(obj.get("reasoning", "") or ""),
        raw_response=raw,
    )


def _to_float(v, default: float) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def pinpoint(
    image: Image.Image,
    lat: float,
    lng: float,
    hub: str,
    extract: Optional[ExtractResult] = None,
    *,
    images: Optional[list[Image.Image]] = None,
    timeout: float = 120.0,
    temperature: float = 0.0,
    on_progress: Optional["callable"] = None,
) -> PinpointResult:
    """Ask the VLM to refine Stage 1's guess to a specific place if it
    can. Returns Stage 1's coords as the fallback when the VLM can't
    pinpoint anything.

    `images`: optional list of high-resolution perspective tiles to send
    instead of the single `image`. The downscaled panorama strip loses small
    detail (flags, distant signs, plates); feeding the overlapping 1024² tiles
    lets Gemma actually SEE those. When omitted, falls back to `[image]`.

    Streams the LM Studio (OpenAI-compatible) response line-by-line. Two
    benefits:
      1. Live progress — `on_progress(event_name, data)` fires for each
         token batch (~1s cadence), letting callers show "VLM thinking..."
         updates AND keeping any downstream HTTP stream's TCP packets
         flowing (which keeps Chrome's MV3 service worker alive when a
         userscript is consuming the upstream stream).
      2. Tighter latency feedback — caller learns the moment generation
         starts, mid-stream token counts, and final completion separately.

    Backend is Gemma 4 26B A4B via LM Studio. We feed PRE-EXTRACTED Surya
    OCR text (not raw pixels to decipher) under the tight pinpoint system
    prompt — that keeps Gemma's thinking bounded (~800 tok / ~8 s) instead
    of spiralling (15k tok / 138 s) the way it does when asked to read
    illegible signage directly. The wall-clock cap below is the backstop.
    """
    img_list = images if images else [image]
    user_text = _build_user_prompt(lat, lng, hub, extract)
    if len(img_list) > 1:
        user_text += (
            f"\n\nYou are given {len(img_list)} overlapping perspective views "
            "covering the full 360° around the camera. Scan ALL of them for "
            "small details the panorama strip loses — flags, distant street "
            "signs, license plates, brand logos, shop names."
        )
    user_content = [{"type": "text", "text": user_text}]
    for im in img_list:
        user_content.append(
            {"type": "image_url", "image_url": {"url": _image_to_data_url(im)}}
        )
    payload = {
        "model": GEMMA_VL_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "stream": True,
        "temperature": temperature,
        # Generous ceiling: thinking + JSON. The tight system prompt keeps
        # real usage modest, but multi-tile input grows the reasoning (more to
        # look at), so leave plenty of headroom. The wall-clock timeout (not
        # this cap) is what protects latency. A small cap would CLIP the answer
        # mid-thought (the 138 s / 0-content failure we hit during bring-up).
        "max_tokens": 12288,
    }

    def _emit(name: str, **data) -> None:
        if on_progress is None:
            return
        try:
            on_progress(name, data)
        except Exception:
            log.exception("pinpoint on_progress callback raised; ignoring")

    content_parts: list[str] = []
    thinking_parts: list[str] = []
    last_progress = time.time()
    started = False
    # Wall-clock cap on TOTAL generation. `requests.post(timeout=...)` with
    # stream=True is per-read, not total — Thinking-mode VLMs that drip
    # tokens steadily can run 60-90 s without triggering it. We enforce a
    # real wall-clock here and gracefully fall back when exceeded.
    t_start = time.time()
    timed_out = False

    with requests.post(
        f"{LMSTUDIO_HOST}/v1/chat/completions", json=payload,
        timeout=timeout, stream=True,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            # Hard wall-clock timeout. Closes the connection cleanly.
            if time.time() - t_start > timeout:
                log.warning(
                    "pinpoint: wall-clock timeout (%.1fs > %.1fs) — "
                    "aborting VLM call; will fall back via refine.",
                    time.time() - t_start, timeout,
                )
                timed_out = True
                _emit("wallclock_timeout", elapsed=time.time() - t_start)
                break
            if not line:
                continue
            # OpenAI SSE framing: "data: {json}\n\n", terminal "data: [DONE]".
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line:
                continue
            if line == "[DONE]":
                _emit("token_done",
                      content_chars=sum(len(p) for p in content_parts),
                      thinking_chars=sum(len(p) for p in thinking_parts))
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta", {}) or {}
            piece = delta.get("content") or ""
            # LM Studio surfaces Gemma's thinking in reasoning_content
            # (some builds use 'reasoning'); accept either.
            think_piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if piece:
                content_parts.append(piece)
            if think_piece:
                thinking_parts.append(think_piece)
            if not started and (piece or think_piece):
                started = True
                _emit("token_start")
            # Throttle progress events to ~1 Hz so we don't flood the
            # downstream queue. Emit content and thinking byte counts so
            # the userscript can show "VLM thinking, 3247 chars so far…".
            now = time.time()
            if now - last_progress > 1.0:
                _emit("token",
                      content_chars=sum(len(p) for p in content_parts),
                      thinking_chars=sum(len(p) for p in thinking_parts),
                      latest=(piece or think_piece)[-80:])
                last_progress = now
            # Server signals completion via finish_reason on the last chunk;
            # the [DONE] sentinel (handled above) is the other terminator.
            if choice.get("finish_reason"):
                _emit("token_done",
                      content_chars=sum(len(p) for p in content_parts),
                      thinking_chars=sum(len(p) for p in thinking_parts))
                break

    if timed_out:
        # Raise so refine.py's existing timeout path handles the fallback.
        raise requests.exceptions.Timeout(
            f"VLM exceeded wall-clock timeout of {timeout}s"
        )
    raw = "".join(content_parts)
    return _parse(raw, lat, lng, hub)
