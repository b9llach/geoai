"""Macro-verification of a Stage 1 prediction using Qwen3-VL via Ollama.

The verifier never produces a coordinate. It only answers: given this
image and this (lat, lng, hub) guess, is the macro story self-consistent?
Script + language + landscape + traffic-side must match the country
the guess lands in. Micro-distance errors (5 km down the wrong road
within the right city) are explicitly allowed — RESEARCH.md §4
calls this MACRO-VERIFICATION.

The prompt is lifted from RESEARCH.md §4 with two tweaks:
    1. The "EXTERNAL DATA" block is omitted when extract.py found no text
       (rural / featureless cases). The VLM falls back to image-only.
    2. We force the model to emit a JSON block fenced as ```json so we
       can parse it deterministically; non-JSON-fenced outputs go through
       a permissive regex extractor as a fallback.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import requests
from PIL import Image

from geoai.config import OLLAMA_HOST, QWEN_VL_MODEL
from geoai.stage2.extract import ExtractResult

log = logging.getLogger(__name__)


@dataclass
class VerifyVerdict:
    is_plausible: bool
    confidence: float
    confirmed_country: str
    primary_script: str
    contradictions: list[str] = field(default_factory=list)
    reasoning: str = ""
    raw_response: str = ""


_SYSTEM = """You are a Geolocation Verification Agent. Your job is to decide \
whether an upstream vision model's COORDINATE GUESS for a street-view image \
is plausible at the country/region level.

# Inputs

You are given:
  - An image (a street-view crop).
  - The COORDINATE GUESS: (lat, lng) and a "Guessed Region" label like \
"Krong Pursat, Cambodia". THIS IS THE PREDICTION YOU ARE EVALUATING.
  - Pre-extracted OCR data: the detected script, the raw transcribed text, \
and an English translation. TREAT THIS AS GROUND-TRUTH OCR — do NOT re-OCR, \
do NOT re-translate, do NOT second-guess the script identification.

# How to reason (in order)

## Step 1 — list contradictions
Before reaching any verdict, list EVERY hard contradiction between the image \
and the COORDINATE GUESS. A hard contradiction is a fact in the image that is \
incompatible with the country the coordinates land in. Examples:

  - Image text is in script X; coordinates point to a country that does not \
use script X. (Cyrillic script in Cambodia, Khmer in Thailand, Hangul in \
Vietnam, Devanagari in Egypt, Hebrew in Iran, Arabic in Greece, Thai in \
Russia, Amharic in Saudi Arabia, etc. — these are AUTOMATIC contradictions.)
  - Vehicles driving on the right side of the road; coordinates in a \
left-driving country (UK, Japan, Australia, India, Indonesia, Thailand, \
South Africa, etc.) — or vice versa.
  - Architectural style fundamentally incompatible with the region (e.g., \
Japanese pagoda roofs in coords pointing to Brazil).
  - Visible signage in a language not spoken in the guessed country.

If no hard contradiction exists, list "none".

## Step 2 — make the decision

If contradictions == ["none"], set is_guess_plausible = true.
If ANY hard contradiction exists, set is_guess_plausible = false.

# Anchor rules (script → set of plausible countries)

These map the detected script to its plausible countries. If the coordinate \
guess's country is NOT in the script's plausible set, that is an AUTOMATIC \
hard contradiction and the verdict is FALSE.

  - Khmer       → Cambodia
  - Thai        → Thailand
  - Lao         → Laos
  - Burmese     → Myanmar
  - Vietnamese  → Vietnam (Latin + diacritics)
  - Hangul      → South Korea, North Korea
  - Japanese    → Japan
  - Chinese     → China, Taiwan, Singapore (signage), HK/Macau
  - Hebrew      → Israel
  - Arabic      → MENA + Maghreb + Gulf countries
  - Persian     → Iran, parts of Afghanistan/Tajikistan
  - Amharic     → Ethiopia, Eritrea
  - Tigrinya    → Ethiopia, Eritrea
  - Georgian    → Georgia
  - Armenian    → Armenia
  - Greek       → Greece, Cyprus
  - Cyrillic    → Russia, Belarus, Ukraine, Serbia, Bulgaria, N.Macedonia, \
Kazakhstan, Kyrgyzstan, Mongolia, Tajikistan
  - Devanagari  → India, Nepal
  - Sinhala     → Sri Lanka
  - Tamil       → India (south), Sri Lanka, Singapore
  - Bengali     → Bangladesh, India (West Bengal)
  - Latin       → too broad to anchor; use other cues instead

# Macro-accuracy rule

The coordinates do NOT need to be street-accurate. A correct guess can be \
5-50 km away from the actual building. Only reject for COUNTRY-level or \
clear REGION-level mismatches; do not reject because the guess is in the \
wrong neighborhood within the right country.

# Output

End your response with a JSON block fenced as ```json. No prose after it.
The schema is exact:

```json
{
  "upstream_coordinates_evaluated": {"lat": LAT, "lng": LNG},
  "primary_script_detected": "string",
  "confirmed_country_or_state": "string",
  "is_guess_plausible": true,
  "confidence_score": 0.00,
  "contradictions_found": ["string"],
  "reasoning_summary": "string"
}
```

`confirmed_country_or_state` is the country YOU determine the image is FROM \
(based on script/landscape/signage), not the upstream guess. If you find \
contradictions, this will differ from the upstream's country.

Keep `reasoning_summary` to one or two sentences."""


def _image_to_b64(image: Image.Image) -> str:
    """Encode the PIL image as base64 PNG for Ollama's images[] field."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_user_prompt(lat: float, lng: float, hub: str,
                       extract: Optional[ExtractResult]) -> str:
    """Format the user message. Omits the EXTERNAL DATA block when extract
    is None so the VLM doesn't try to verify against absent translation."""
    parts = [
        "Upstream Geolocation Prediction:",
        f"- Latitude: {lat:.6f}",
        f"- Longitude: {lng:.6f}",
        f"- Guessed Region: {hub}",
        "",
    ]
    if extract is not None:
        parts.extend([
            "EXTERNAL DATA VERIFICATION LAYOUT:",
            f"- Detected Language/Script: {extract.language_name}",
            f"- Extracted Raw String: {extract.raw_text}",
            f"- Verified English Translation: \"{extract.english_translation}\"",
            "",
        ])
    else:
        parts.extend([
            "EXTERNAL DATA VERIFICATION LAYOUT:",
            "- Detected Language/Script: (none — no readable text in image)",
            "",
        ])
    parts.append(
        "Execute your verification steps on the attached image. Cross-reference"
        " the verified translation (if any) and script against the coordinate"
        " guess, and emit the final JSON block."
    )
    return "\n".join(parts)


_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{[^{}]*\"is_guess_plausible\"[^{}]*\})", re.DOTALL)


def _parse_verdict(raw: str) -> VerifyVerdict:
    """Pull the JSON block out of the VLM's response. Falls back to a
    permissive 'is_plausible' regex if there's no fence."""
    m = _JSON_FENCE_RE.search(raw)
    if m is None:
        m = _BARE_JSON_RE.search(raw)
    if m is None:
        log.warning("verifier: no JSON block found; defaulting to plausible=False")
        return VerifyVerdict(
            is_plausible=False, confidence=0.0,
            confirmed_country="unknown", primary_script="unknown",
            contradictions=["verifier returned no JSON block"],
            reasoning=raw[-500:], raw_response=raw,
        )
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        log.warning("verifier: bad JSON: %s; defaulting to plausible=False", e)
        return VerifyVerdict(
            is_plausible=False, confidence=0.0,
            confirmed_country="unknown", primary_script="unknown",
            contradictions=[f"JSON decode error: {e}"],
            reasoning=raw[-500:], raw_response=raw,
        )
    return VerifyVerdict(
        is_plausible=bool(obj.get("is_guess_plausible", False)),
        confidence=float(obj.get("confidence_score", 0.0) or 0.0),
        confirmed_country=str(obj.get("confirmed_country_or_state", "") or "unknown"),
        primary_script=str(obj.get("primary_script_detected", "") or "unknown"),
        contradictions=list(obj.get("contradictions_found") or []),
        reasoning=str(obj.get("reasoning_summary", "") or ""),
        raw_response=raw,
    )


def verify_prediction(
    image: Image.Image,
    lat: float,
    lng: float,
    hub: str,
    extract: Optional[ExtractResult] = None,
    *,
    timeout: float = 240.0,
    temperature: float = 0.0,
) -> VerifyVerdict:
    """Ask Qwen3-VL whether (lat, lng) is macro-plausible for this image.

    Args:
        image:   PIL image (any size; Qwen3-VL handles its own resize)
        lat, lng: the Stage 1 candidate's coordinate
        hub:     reverse-geocoded label, e.g. "Phnom Penh, Cambodia"
        extract: optional ExtractResult from extract.py
        timeout: HTTP timeout in seconds; default 90 (Thinking mode is slow)
        temperature: 0 for determinism; raise for sampling variety
    """
    payload = {
        "model": QWEN_VL_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _build_user_prompt(lat, lng, hub, extract),
                "images": [_image_to_b64(image)],
            },
        ],
        "stream": False,
        # Default num_ctx in Ollama is 256k for qwen3-vl, which spills to
        # CPU and balloons memory to ~50 GB. Our prompts are <4k tokens
        # plus one image; 16k is plenty and fits the whole model on GPU.
        # num_predict bounds Thinking-mode generation — without it, the
        # 8B can loop on featureless inputs and burn the entire context.
        # keep_alive=24h prevents the per-request cold start.
        "keep_alive": "24h",
        "options": {
            "temperature": temperature,
            "num_ctx": 16384,
            "num_predict": 2048,
        },
    }
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    return _parse_verdict(raw)
