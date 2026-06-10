"""Country vocabulary for the auxiliary country-classification head.

Mirrors `CellVocab` in spirit: deterministic ordering, integer indices,
sentinel `PRUNED_LABEL = -1` for null/unknown countries (so the country
loss can mask the same way `hierarchical_loss` masks pruned cells).

Built once at training time from the SQLite catalog (`from_catalog`),
and persisted next to the checkpoint as `country_vocab.json` so inference
loads without a SQLite roundtrip.

The country head is V2-onward — V1 checkpoints don't have one. Code that
loads checkpoints should treat the head as optional.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from geoai.stage1.cells import PRUNED_LABEL


class CountryVocab:
    def __init__(self, codes: list[str]):
        # Sorted, deterministic. codes[i] is the country at index i.
        self.codes = list(codes)
        self._idx = {c: i for i, c in enumerate(self.codes)}

    def __len__(self) -> int:
        return len(self.codes)

    @property
    def size(self) -> int:
        return len(self.codes)

    def index(self, code: str | None) -> int:
        """Vocab index, or PRUNED_LABEL for null/empty/unknown."""
        if not code:
            return PRUNED_LABEL
        return self._idx.get(code, PRUNED_LABEL)

    def code_at(self, idx: int) -> str | None:
        if 0 <= idx < len(self.codes):
            return self.codes[idx]
        return None

    @classmethod
    def from_catalog(cls, db_path: Path | str, split: str = "train") -> "CountryVocab":
        """Distinct non-null country_codes in the given split, sorted alphabetically."""
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT DISTINCT country_code FROM panos "
                "WHERE country_code IS NOT NULL AND split = ? "
                "ORDER BY country_code",
                (split,),
            ).fetchall()
        finally:
            conn.close()
        return cls(codes=[r[0] for r in rows])

    @classmethod
    def from_json(cls, path: Path | str) -> "CountryVocab":
        return cls(codes=json.loads(Path(path).read_text())["codes"])

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps({"codes": self.codes}))

    def __repr__(self) -> str:
        return f"CountryVocab(n={len(self.codes)})"
