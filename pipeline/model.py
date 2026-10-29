"""Loading and scoring the deployed model.

Two stages score candidates now - recommend (what you own) and discover (what
you don't) - so the artifact loading and the sigmoid live here instead of in
each stage. Same reasoning that put feature encoding in features.py: a second
copy is a second thing that can drift.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import MODELS_DIR, features


@dataclass(frozen=True)
class Model:
    weights: np.ndarray
    bias: float
    encoding: features.Encoding


def load(models_dir=MODELS_DIR) -> Model:
    weights = np.load(models_dir / "weights.npy")
    bias = json.loads((models_dir / "bias.json").read_text())["bias"]
    return Model(weights=weights, bias=bias, encoding=features.load(models_dir))


def score(df: pd.DataFrame, model: Model) -> np.ndarray:
    """Probability that you'd watch each row, in the frame's own order."""
    X = features.transform(df, model.encoding)
    return 1 / (1 + np.exp(-(X @ model.weights + model.bias)))
