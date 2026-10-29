"""Feature encoding shared by training and serving.

This module is the single source of truth for turning library.csv rows into
model inputs. Training calls fit() to learn the encoding and save() to store
it next to the weights; serving calls load() + transform() to apply the exact
same encoding.

Why it's one module: training and serving used to each own a copy of this
logic, and they had already drifted - training imputed a missing year with the
column median, serving imputed it with the stored mean. A model scored with a
different encoding than it was trained on is train/serve skew, and it fails
silently. Everything the encoding depends on (genre vocabulary, the year fill
value, the year mean/std) is therefore learned in one place and persisted, not
recomputed at serve time.
"""
import json
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

GENRE_SEPARATOR = "|"
REQUIRED_COLUMNS = ("item_id", "title", "year", "genres", "label")


@dataclass(frozen=True)
class Encoding:
    """Everything needed to turn a library row into a feature vector."""

    genre_vocab: List[str]
    year_fill: float
    year_mean: float
    year_std: float

    @property
    def n_features(self) -> int:
        return len(self.genre_vocab) + 1


def _genre_lists(df: pd.DataFrame) -> pd.Series:
    return df["genres"].fillna("").apply(
        lambda cell: [part for part in str(cell).split(GENRE_SEPARATOR) if part]
    )


def _years(df: pd.DataFrame) -> pd.Series:
    # errors="coerce" so a blank or junk year becomes NaN and gets imputed
    # rather than blowing up the whole run.
    return pd.to_numeric(df["year"], errors="coerce")


def fit(df: pd.DataFrame) -> Encoding:
    genre_vocab = sorted({g for genre_list in _genre_lists(df) for g in genre_list})

    year = _years(df)
    year_fill = float(year.median()) if year.notna().any() else 0.0
    year = year.fillna(year_fill)

    year_mean = float(year.mean())
    year_std = float(year.std())
    # std is NaN for a single row and 0.0 when every year is identical; both
    # would poison every feature vector with NaN/inf, so fall back to 1.0.
    if not np.isfinite(year_std) or year_std == 0.0:
        year_std = 1.0

    return Encoding(genre_vocab, year_fill, year_mean, year_std)


def transform(df: pd.DataFrame, encoding: Encoding) -> np.ndarray:
    genre_position = {g: i for i, g in enumerate(encoding.genre_vocab)}

    genre_features = np.zeros((len(df), len(encoding.genre_vocab)), dtype=np.float64)
    for row, genre_list in enumerate(_genre_lists(df)):
        for g in genre_list:
            position = genre_position.get(g)
            # A genre the model never saw during fit has no weight to apply,
            # so it contributes nothing rather than shifting the vector.
            if position is not None:
                genre_features[row, position] = 1.0

    year = _years(df).fillna(encoding.year_fill)
    year_feature = ((year - encoding.year_mean) / encoding.year_std).to_numpy(
        dtype=np.float64
    ).reshape(-1, 1)

    return np.concatenate([genre_features, year_feature], axis=1)


def labels(df: pd.DataFrame) -> np.ndarray:
    return df["label"].to_numpy(dtype=np.float64)


def save(encoding: Encoding, models_dir) -> None:
    (models_dir / "genre_vocab.json").write_text(json.dumps(encoding.genre_vocab))
    (models_dir / "year_stats.json").write_text(
        json.dumps(
            {
                "fill": encoding.year_fill,
                "mean": encoding.year_mean,
                "std": encoding.year_std,
            }
        )
    )


def load(models_dir) -> Encoding:
    genre_vocab = json.loads((models_dir / "genre_vocab.json").read_text())
    year_stats = json.loads((models_dir / "year_stats.json").read_text())

    if "fill" not in year_stats:
        # Deliberately a hard error: silently substituting the mean here is
        # exactly the skew this module exists to prevent.
        raise SystemExit(
            "models/year_stats.json predates the shared feature encoding "
            "(no 'fill' key). Rerun train_local.py to regenerate it."
        )

    return Encoding(
        genre_vocab=genre_vocab,
        year_fill=year_stats["fill"],
        year_mean=year_stats["mean"],
        year_std=year_stats["std"],
    )
