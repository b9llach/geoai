"""OCR + script identification + translation, run before the VLM verifier.

Pulling text out of the image first — with a specialist OCR model — and
labelling its script with fasttext keeps the VLM from hallucinating a
familiar script when it sees one it doesn't recognize (the "Khmer-as-Thai"
failure mode in RESEARCH.md §1). The VLM then gets a structured payload
("Detected script: Khmer; raw: ធនាគារ…; English: Bank of Asia") instead
of having to do OCR itself.

Three stages, gated by output:
    1. Surya OCR — detect text regions and transcribe across 90+ scripts
                   incl. Khmer, Thai, Burmese, Lao, Amharic, Devanagari, etc.
                   (Document-parsers like Nemotron-Parse can detect text
                   regions in any script but only transcribe Latin/CJK —
                   their fallback is `<unknown>` for Khmer/Thai/...)
    2. fasttext lid.176 — identify the script's language (176-way)
    3. NLLB-200 — translate the raw text to English

Each stage's model is lazy-loaded on first call and cached process-wide,
so a long-running server pays the load cost once.

If stage 1 returns empty (the common case for a featureless rural street),
this function returns None and the verifier degrades to image-only macro
analysis.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from geoai.config import (
    FASTTEXT_LID_PATH,
    NLLB_CT2_DIR,
    NLLB_HF_DIR,
)

log = logging.getLogger(__name__)


@dataclass
class ExtractResult:
    """Output of the text-extraction pipeline.

    raw_text is what the OCR model emitted (multiple lines joined by \\n).
    language_code follows fasttext's __label__xx convention but with the
    prefix stripped (e.g. "khm", "rus", "fra"). english_translation is
    NLLB's output, which may equal raw_text when the source is English.
    """
    raw_text: str
    language_code: str
    language_name: str
    language_confidence: float
    english_translation: str


# Module-level singletons. _lock prevents double-init when callers race.
_lock = threading.Lock()
_surya_rec = None        # surya RecognitionPredictor
_surya_det = None        # surya DetectionPredictor
_fasttext = None
_nllb = None
_nllb_tokenizer = None


# fasttext returns ISO-639-3 codes; NLLB uses BCP-47-ish "<code>_<script>"
# names ("rus_Cyrl", "khm_Khmr", "eng_Latn"). This is the slice we actually
# need at runtime — pano text is overwhelmingly one of these scripts.
# Extend on demand; an unmapped lang falls through to skip translation.
_NLLB_LANG = {
    "en": "eng_Latn", "eng": "eng_Latn",
    "fr": "fra_Latn", "fra": "fra_Latn",
    "es": "spa_Latn", "spa": "spa_Latn",
    "pt": "por_Latn", "por": "por_Latn",
    "de": "deu_Latn", "deu": "deu_Latn",
    "it": "ita_Latn", "ita": "ita_Latn",
    "nl": "nld_Latn", "nld": "nld_Latn",
    "pl": "pol_Latn", "pol": "pol_Latn",
    "ro": "ron_Latn", "ron": "ron_Latn",
    "tr": "tur_Latn", "tur": "tur_Latn",
    "vi": "vie_Latn", "vie": "vie_Latn",
    "id": "ind_Latn", "ind": "ind_Latn",
    "ru": "rus_Cyrl", "rus": "rus_Cyrl",
    "uk": "ukr_Cyrl", "ukr": "ukr_Cyrl",
    "bg": "bul_Cyrl", "bul": "bul_Cyrl",
    "sr": "srp_Cyrl", "srp": "srp_Cyrl",
    "ja": "jpn_Jpan", "jpn": "jpn_Jpan",
    "ko": "kor_Hang", "kor": "kor_Hang",
    "zh": "zho_Hans", "zho": "zho_Hans",
    "th": "tha_Thai", "tha": "tha_Thai",
    "km": "khm_Khmr", "khm": "khm_Khmr",
    "lo": "lao_Laoo", "lao": "lao_Laoo",
    "my": "mya_Mymr", "mya": "mya_Mymr",
    "ar": "arb_Arab", "ara": "arb_Arab", "arb": "arb_Arab",
    "fa": "pes_Arab", "fas": "pes_Arab", "pes": "pes_Arab",
    "he": "heb_Hebr", "heb": "heb_Hebr",
    "hi": "hin_Deva", "hin": "hin_Deva",
    "el": "ell_Grek", "ell": "ell_Grek",
    "am": "amh_Ethi", "amh": "amh_Ethi",
    "ka": "kat_Geor", "kat": "kat_Geor",
    "hy": "hye_Armn", "hye": "hye_Armn",
}

_LANG_NAME = {
    "eng_Latn": "English", "fra_Latn": "French", "spa_Latn": "Spanish",
    "por_Latn": "Portuguese", "deu_Latn": "German", "ita_Latn": "Italian",
    "nld_Latn": "Dutch", "pol_Latn": "Polish", "ron_Latn": "Romanian",
    "tur_Latn": "Turkish", "vie_Latn": "Vietnamese", "ind_Latn": "Indonesian",
    "rus_Cyrl": "Russian", "ukr_Cyrl": "Ukrainian", "bul_Cyrl": "Bulgarian",
    "srp_Cyrl": "Serbian", "jpn_Jpan": "Japanese", "kor_Hang": "Korean",
    "zho_Hans": "Chinese", "tha_Thai": "Thai", "khm_Khmr": "Khmer",
    "lao_Laoo": "Lao", "mya_Mymr": "Burmese", "arb_Arab": "Arabic",
    "pes_Arab": "Persian", "heb_Hebr": "Hebrew", "hin_Deva": "Hindi",
    "ell_Grek": "Greek", "amh_Ethi": "Amharic", "kat_Geor": "Georgian",
    "hye_Armn": "Armenian",
}


def _get_surya():
    """Surya OCR — detection + recognition. Models auto-download to
    ~/.cache/datalab/models/ on first call (~150 MB combined)."""
    global _surya_rec, _surya_det
    if _surya_rec is not None:
        return _surya_rec, _surya_det
    with _lock:
        if _surya_rec is not None:
            return _surya_rec, _surya_det
        # Set Surya's torch device via env var BEFORE the import — Surya
        # reads it at module load time.
        import os
        os.environ.setdefault("TORCH_DEVICE", "cuda:0")
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        log.info("loading Surya OCR detector + recognizer")
        _surya_det = DetectionPredictor()
        _surya_rec = RecognitionPredictor()
        return _surya_rec, _surya_det


def _get_fasttext():
    """fasttext lid.176 — supports 176 languages including every script we'd
    expect in pano text. fasttext-wheel 0.9.x has a numpy-2.x incompatibility
    on its single-text predict() path (uses the removed `np.array(copy=False)`
    spelling); we monkey-patch the single line to use `np.asarray()` instead.
    The batch-predict path is already numpy-2 compatible upstream."""
    global _fasttext
    if _fasttext is not None:
        return _fasttext
    with _lock:
        if _fasttext is not None:
            return _fasttext
        import numpy as np
        import fasttext
        import fasttext.FastText as _ft_mod

        if not FASTTEXT_LID_PATH.exists():
            raise FileNotFoundError(
                f"fasttext lid model missing: {FASTTEXT_LID_PATH} — "
                "download via the curl command in the Stage 2 docs."
            )

        _orig_predict = _ft_mod._FastText.predict

        def _patched_predict(self, text, k=1, threshold=0.0,
                             on_unicode_error="strict"):
            if isinstance(text, list):
                return _orig_predict(self, text, k, threshold, on_unicode_error)
            # Single-string path: bypass the broken np.array(copy=False) line.
            # The C binding returns [(prob, label), ...]; we mirror upstream's
            # (labels_tuple, probs_array) return shape via np.asarray.
            checked = _ft_mod.check(text) if hasattr(_ft_mod, "check") else text
            predictions = self.f.predict(checked, k, threshold,
                                          on_unicode_error)
            if predictions:
                probs, labels = zip(*predictions)
            else:
                probs, labels = ([], ())
            return labels, np.asarray(probs)

        _ft_mod._FastText.predict = _patched_predict

        # Suppress fasttext's deprecation warning on load.
        _ft_mod.eprint = lambda *a, **k: None
        _fasttext = fasttext.load_model(str(FASTTEXT_LID_PATH))
        return _fasttext


def _get_nllb():
    """Try CTranslate2 first (faster, int8). Fall back to HF transformers
    if the CT2 conversion wasn't run yet."""
    global _nllb, _nllb_tokenizer
    if _nllb is not None:
        return _nllb, _nllb_tokenizer
    with _lock:
        if _nllb is not None:
            return _nllb, _nllb_tokenizer
        from transformers import AutoTokenizer

        # NLLB runs on CPU. Translation fires at most once per pano (rural
        # panos have no text and skip it entirely), so the ~1 s/call CPU
        # latency is invisible against the 3-5 s VLM call. CPU placement
        # also avoids fighting Surya + Ollama for GPU 0 VRAM.
        if NLLB_CT2_DIR.exists():
            import ctranslate2

            log.info("loading NLLB-200 (CT2 int8 CPU) from %s", NLLB_CT2_DIR)
            _nllb = ctranslate2.Translator(str(NLLB_CT2_DIR),
                                            device="cpu",
                                            compute_type="int8",
                                            inter_threads=2,
                                            intra_threads=4)
            _nllb_tokenizer = AutoTokenizer.from_pretrained(str(NLLB_HF_DIR))
        else:
            import torch
            from transformers import AutoModelForSeq2SeqLM

            log.info("loading NLLB-200 (HF CPU) from %s", NLLB_HF_DIR)
            _nllb_tokenizer = AutoTokenizer.from_pretrained(str(NLLB_HF_DIR))
            _nllb = AutoModelForSeq2SeqLM.from_pretrained(
                str(NLLB_HF_DIR), torch_dtype=torch.float32,
            ).eval()
        return _nllb, _nllb_tokenizer


_SURYA_TAG_RE = re.compile(r"<(br|sup|sub|/[a-z]+)\s*/?>", re.IGNORECASE)


def _clean_surya_line(text: str) -> str:
    """Strip Surya's layout markers and collapse whitespace."""
    t = _SURYA_TAG_RE.sub(" ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _run_surya(image: Image.Image, min_conf: float = 0.5) -> str:
    """Single-image OCR via Surya. Returns concatenated text lines or ''.

    Kept for backward compatibility; the multi-tile path below is
    preferred when the input is a panorama (better recall on text that
    straddles cardinal boundaries).
    """
    rec, det = _get_surya()
    preds = rec([image], task_names=["ocr_with_boxes"], det_predictor=det)
    if not preds or not preds[0].text_lines:
        return ""
    lines: list[str] = []
    for tl in preds[0].text_lines:
        if tl.confidence < min_conf:
            continue
        t = _clean_surya_line(tl.text)
        if t:
            lines.append(t)
    return "\n".join(lines)


def _run_surya_tiles(tiles: list, min_conf: float = 0.5) -> str:
    """Batch-OCR a list of overlapping perspective tiles, dedupe lines
    across tiles, return joined text. Surya batches the whole list in a
    single forward pass so this is cheaper than 8 sequential calls.

    Dedupe is exact-match (case-sensitive) on the cleaned line text.
    Overlapping tiles will detect the same sign multiple times; one
    instance is enough for langid + the VLM prompt.
    """
    if not tiles:
        return ""
    rec, det = _get_surya()
    preds = rec(tiles, task_names=["ocr_with_boxes"] * len(tiles),
                det_predictor=det)
    seen: set[str] = set()
    ordered: list[str] = []
    for p in preds or []:
        for tl in p.text_lines or []:
            if tl.confidence < min_conf:
                continue
            t = _clean_surya_line(tl.text)
            if not t or t in seen:
                continue
            seen.add(t)
            ordered.append(t)
    return "\n".join(ordered)


def _run_langid(text: str) -> tuple[str, float]:
    """Identify language. Returns (iso-639-1-or-3 code, confidence in 0..1)."""
    if not text.strip():
        return "unk", 0.0
    model = _get_fasttext()
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    iso = labels[0].replace("__label__", "")
    return iso, float(probs[0])


def _run_nllb(text: str, src_nllb: str, tgt_nllb: str = "eng_Latn") -> str:
    """Translate text src→tgt using NLLB. Returns the English string."""
    if src_nllb == tgt_nllb:
        return text
    model, tokenizer = _get_nllb()
    tokenizer.src_lang = src_nllb
    # CT2 vs HF paths differ in API
    import ctranslate2
    if isinstance(model, ctranslate2.Translator):
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        results = model.translate_batch(
            [tokens],
            target_prefix=[[tgt_nllb]],
            max_decoding_length=256,
            beam_size=2,
        )
        out_tokens = results[0].hypotheses[0][1:]  # drop target_prefix
        return tokenizer.decode(
            tokenizer.convert_tokens_to_ids(out_tokens),
            skip_special_tokens=True,
        ).strip()
    else:
        import torch
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_nllb)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, forced_bos_token_id=forced_bos,
                                  max_new_tokens=256, num_beams=2)
        return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def _finalize_extract(raw: str) -> Optional[ExtractResult]:
    """Shared post-OCR pipeline: langid + translate. Returns None when
    the OCR turned up nothing worth translating."""
    if len(raw.strip()) < 2:
        return None
    iso, conf = _run_langid(raw)
    nllb_code = _NLLB_LANG.get(iso)
    lang_name = _LANG_NAME.get(nllb_code, iso) if nllb_code else iso
    if nllb_code is None or nllb_code == "eng_Latn":
        english = raw
    else:
        try:
            english = _run_nllb(raw, nllb_code, "eng_Latn")
        except Exception as e:
            log.warning("NLLB translate failed (%s -> en): %s", nllb_code, e)
            english = raw
    return ExtractResult(
        raw_text=raw,
        language_code=nllb_code or iso,
        language_name=lang_name,
        language_confidence=conf,
        english_translation=english,
    )


def extract_from_image(image: Image.Image) -> Optional[ExtractResult]:
    """Full extract pipeline on a single image (perspective crop or user
    upload). Returns None when no text was found."""
    return _finalize_extract(_run_surya(image))


def extract_from_tiles(tiles: list) -> Optional[ExtractResult]:
    """Full extract pipeline on a list of overlapping perspective tiles
    (e.g. the 8 tiles from `equirect_to_perspective_tiles`). Surya
    batches all tiles in one forward pass, results are deduped before
    langid + translation. Returns None when nothing was found across
    any tile."""
    return _finalize_extract(_run_surya_tiles(tiles))


def extract_from_path(path: str | Path) -> Optional[ExtractResult]:
    """Convenience wrapper; opens the path then calls extract_from_image."""
    return extract_from_image(Image.open(path).convert("RGB"))
