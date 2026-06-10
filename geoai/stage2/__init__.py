"""Stage 2: macro-verification of Stage 1 predictions via VLM + OCR.

The pipeline never produces a coordinate of its own. Instead it scores
each Stage 1 top-K candidate by plausibility — script + language + macro
landscape vs the candidate's country/region — and promotes the first
candidate that survives verification. If all reject, falls back to
Stage 1's top-1.

Layout:
    extract.py  — Nemotron-Parse OCR -> fasttext langid -> NLLB translation
    verify.py   — Qwen3-VL via Ollama, prompted as in RESEARCH.md §4
    refine.py   — top-K hill-climb wrapper; the public entry point
"""
