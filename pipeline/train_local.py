"""Retrain the recommender locally from library.csv - no torch, no Colab.

Same model as train.ipynb (genre + year -> watched probability), reimplemented
with plain numpy gradient descent since the model is tiny enough that training
here costs nothing. train.ipynb stays as the one-off "trained on Colab" demo;
this script is the one meant to run on a schedule.

Feature encoding lives in features.py so training and serving cannot drift.

Includes a promotion gate: the new model only overwrites models/ if its
validation accuracy is at least as good as the currently deployed model's.
Every attempt - promoted or not - is appended to metrics_history.csv so you
can see the trend over time.

Training is deterministic (fixed seed, fixed split), so the same library.csv
always produces the same model and the same metrics. That is what lets CI
assert on accuracy instead of just checking the script doesn't crash.
"""
import argparse
import csv
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import LIBRARY_CSV, METRICS_HISTORY_CSV, MODELS_DIR, features

SEED = 42
LR = 0.05
EPOCHS = 500
L2 = 1e-3
TRAIN_FRACTION = 0.8


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train(X, y):
    n, n_features = X.shape
    weight = np.zeros(n_features)
    bias = 0.0

    # Watched items are the minority class; without this the model can score
    # well by calling everything unwatched.
    n_pos = max(y.sum(), 1)
    n_neg = max(len(y) - y.sum(), 1)
    sample_weight = np.where(y == 1, n_neg / n_pos, 1.0)

    for _ in range(EPOCHS):
        residual = (sigmoid(X @ weight + bias) - y) * sample_weight
        weight -= LR * (X.T @ residual / n + L2 * weight)
        bias -= LR * residual.mean()

    return weight, bias


def evaluate(X, y, weight, bias, k=10):
    logits = X @ weight + bias
    accuracy = float(((sigmoid(logits) > 0.5) == y).mean())

    k = min(k, len(y))
    ranked = y[np.argsort(-logits)]
    precision_at_k = float(ranked[:k].mean()) if k else float("nan")
    return accuracy, precision_at_k, k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=LIBRARY_CSV)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="train and evaluate but write nothing (used by CI)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="exit non-zero if validation accuracy falls below this floor",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.library)
    encoding = features.fit(df)
    X = features.transform(df, encoding)
    y = features.labels(df)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(df))
    split = int(TRAIN_FRACTION * len(df))
    train_idx, val_idx = perm[:split], perm[split:]

    weight, bias = train(X[train_idx], y[train_idx])
    val_accuracy, precision_at_k, k = evaluate(X[val_idx], y[val_idx], weight, bias)

    new_metrics = {
        "val_accuracy": val_accuracy,
        f"val_precision_at_{k}": precision_at_k,
        "n_items": len(df),
        "n_genres": len(encoding.genre_vocab),
        "n_positive": int(y.sum()),
    }

    deployed_metrics_path = MODELS_DIR / "metrics.json"
    deployed_accuracy = None
    if deployed_metrics_path.exists():
        deployed_accuracy = json.loads(deployed_metrics_path.read_text()).get("val_accuracy")

    # Note: as library.csv grows the validation split changes, so this compares
    # scores measured on different data. A frozen holdout would make the
    # comparison exact; for now the gate only catches large regressions.
    promoted = deployed_accuracy is None or val_accuracy >= deployed_accuracy

    if args.dry_run:
        print(f"dry run: val_accuracy={val_accuracy:.3f} (wrote nothing)")
    elif promoted:
        MODELS_DIR.mkdir(exist_ok=True)
        np.save(MODELS_DIR / "weights.npy", weight)
        (MODELS_DIR / "bias.json").write_text(json.dumps({"bias": bias}))
        features.save(encoding, MODELS_DIR)
        deployed_metrics_path.write_text(json.dumps(new_metrics, indent=2))
        print(f"promoted new model (val_accuracy={val_accuracy:.3f})")
    else:
        print(
            f"kept existing model (new val_accuracy={val_accuracy:.3f} "
            f"< deployed {deployed_accuracy:.3f})"
        )

    if not args.dry_run:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "promoted": promoted,
            **new_metrics,
        }
        write_header = not METRICS_HISTORY_CSV.exists()
        with METRICS_HISTORY_CSV.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    if args.min_accuracy is not None and val_accuracy < args.min_accuracy:
        raise SystemExit(
            f"validation accuracy {val_accuracy:.3f} is below the required "
            f"floor of {args.min_accuracy:.3f}"
        )


if __name__ == "__main__":
    main()
