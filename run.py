#!/usr/bin/env python3
"""Convenience wrapper for the project's services and CLIs.

Usage:
    python run.py <subcommand> [args...]

Subcommands:
    setup       Create .venv and pip install -e . (run this first on a fresh box)
    serve       FastAPI inference server (geoai-serve)
    scrape      Supplement scraper (data_scraper.py)
    train       Stage 1 trainer (geoai-train-stage1)
    predict     Batch predict (geoai-predict-stage1)
    catalog     Build SQLite metadata.db (geoai-catalog)
    crops       Render 4-perspective crops (geoai-render-crops)
    cell-stats  Cell-vocab pruning stats (geoai-cell-stats)
    split       Held-out splits (geoai-split)
    ingest      PlonkIt country-guide ingest (geoai-ingest-country-guides)
    sv-scrape   Original Phase 1 Street View scraper (geoai-scrape)

Most subcommands forward extra arguments to the underlying CLI:
    python run.py serve --help
    python run.py train --resume-from-checkpoint /data/.../epoch_05
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_BIN = VENV / "bin"
PY = VENV_BIN / "python"

# Map subcommand -> argv. Each is resolved against .venv/bin so the project
# always uses the project's pinned interpreter, even if you're outside the venv.
COMMANDS = {
    "serve":      [str(VENV_BIN / "geoai-serve")],
    "scrape":     [str(PY), str(ROOT / "data_scraper.py")],
    "train":      [str(VENV_BIN / "geoai-train-stage1")],
    "predict":    [str(VENV_BIN / "geoai-predict-stage1")],
    "catalog":    [str(VENV_BIN / "geoai-catalog")],
    "crops":      [str(VENV_BIN / "geoai-render-crops")],
    "cell-stats": [str(VENV_BIN / "geoai-cell-stats")],
    "split":      [str(VENV_BIN / "geoai-split")],
    "ingest":     [str(VENV_BIN / "geoai-ingest-country-guides")],
    "sv-scrape":  [str(VENV_BIN / "geoai-scrape")],
}


def setup() -> int:
    """Bootstrap on a fresh machine: create .venv and pip install -e ."""
    if VENV.exists():
        print(f"venv already exists at {VENV} — skipping creation")
    else:
        print(f"creating venv at {VENV} ...")
        subprocess.run(["python3.11", "-m", "venv", str(VENV)], check=True)
    pip = VENV_BIN / "pip"
    print("installing project (editable) ...")
    subprocess.run([str(pip), "install", "-e", "."], cwd=str(ROOT), check=True)
    print()
    print("Note: torch + torchvision must be installed separately from PyTorch's CUDA wheel index:")
    print(f"  {pip} install --index-url https://download.pytorch.org/whl/cu128 torch torchvision")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    sub = sys.argv[1]
    rest = sys.argv[2:]

    if sub == "setup":
        return setup()

    if sub not in COMMANDS:
        print(f"unknown subcommand: {sub}\n")
        print(__doc__)
        return 1

    if not VENV.exists():
        print(f"error: no venv at {VENV} — run `python run.py setup` first")
        return 1

    cmd = COMMANDS[sub] + rest
    target = cmd[0]
    if not Path(target).exists():
        print(f"error: {target} not found — run `python run.py setup` to (re)install")
        return 1

    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
