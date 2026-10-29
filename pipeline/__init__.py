"""The export -> validate -> train -> recommend pipeline.

Each stage is a module you run with -m from the repo root, e.g.:

    python -m pipeline.train_local

Paths live here so no stage computes its own and they cannot disagree about
where the data or the deployed model lives.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"

LIBRARY_CSV = DATA_DIR / "library.csv"
METRICS_HISTORY_CSV = DATA_DIR / "metrics_history.csv"
