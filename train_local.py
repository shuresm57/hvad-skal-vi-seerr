"""Retrain the recommender locally from library.csv - no torch, no Colab.

Same model as train.ipynb (genre + year -> watched probability), reimplemented
with plain numpy gradient descent since the model is tiny enough that training
here costs nothing. train.ipynb stays as the one-off "trained on Colab" demo;
this script is the one meant to run on a schedule (cron, Docker, whatever).

Includes a promotion gate: the new model only overwrites models/ if its
validation accuracy is at least as good as the currently deployed model's.
Every attempt - promoted or not - is appended to metrics_history.csv so you
can see the trend over time.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"
HISTORY_PATH = Path(__file__).parent / "metrics_history.csv"

SEED = 42
LR = 0.05
EPOCHS = 500
L2 = 1e-3


def build_features(df):
    genres = df["genres"].fillna("").apply(lambda g: g.split("|") if g else [])
    genre_vocab = sorted({g for genre_list in genres for g in genre_list})
    genre_index = {g: i for i, g in enumerate(genre_vocab)}

    genre_features = np.zeros((len(df), len(genre_vocab)), dtype=np.float64)
    for row, genre_list in enumerate(genres):
        for g in genre_list:
            genre_features[row, genre_index[g]] = 1.0

    year = df["year"].fillna(df["year"].median())
    year_mean, year_std = float(year.mean()), float(year.std() or 1.0)
    year_feature = ((year - year_mean) / year_std).to_numpy(dtype=np.float64).reshape(-1, 1)

    X = np.concatenate([genre_features, year_feature], axis=1)
    y = df["label"].to_numpy(dtype=np.float64)
    return X, y, genre_vocab, year_mean, year_std


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train(X, y, rng):
    n, n_features = X.shape
    weight = np.zeros(n_features)
    bias = 0.0

    n_pos = max(y.sum(), 1)
    n_neg = max(len(y) - y.sum(), 1)
    pos_weight = n_neg / n_pos
    sample_weight = np.where(y == 1, pos_weight, 1.0)

    for _ in range(EPOCHS):
        pred = sigmoid(X @ weight + bias)
        diff = (pred - y) * sample_weight
        grad_w = X.T @ diff / n + L2 * weight
        grad_b = diff.mean()
        weight -= LR * grad_w
        bias -= LR * grad_b

    return weight, bias


def evaluate(X, y, weight, bias, k=10):
    logits = X @ weight + bias
    pred = sigmoid(logits)
    accuracy = float(((pred > 0.5) == y).mean())

    order = np.argsort(-logits)
    k = min(k, len(y))
    precision_at_k = float(y[order][:k].mean()) if k else float("nan")
    return accuracy, precision_at_k, k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="library.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    df = pd.read_csv(args.library)
    X, y, genre_vocab, year_mean, year_std = build_features(df)

    perm = rng.permutation(len(df))
    split = int(0.8 * len(df))
    train_idx, val_idx = perm[:split], perm[split:]

    weight, bias = train(X[train_idx], y[train_idx], rng)
    val_accuracy, precision_at_k, k = evaluate(X[val_idx], y[val_idx], weight, bias)

    new_metrics = {
        "val_accuracy": val_accuracy,
        f"val_precision_at_{k}": precision_at_k,
        "n_items": len(df),
        "n_genres": len(genre_vocab),
        "n_positive": int(y.sum()),
    }

    current_metrics_path = MODELS_DIR / "metrics.json"
    promoted = True
    if current_metrics_path.exists():
        old_metrics = json.loads(current_metrics_path.read_text())
        if new_metrics["val_accuracy"] < old_metrics.get("val_accuracy", 0):
            promoted = False

    if promoted:
        MODELS_DIR.mkdir(exist_ok=True)
        np.save(MODELS_DIR / "weights.npy", weight)
        (MODELS_DIR / "bias.json").write_text(json.dumps({"bias": bias}))
        (MODELS_DIR / "genre_vocab.json").write_text(json.dumps(genre_vocab))
        (MODELS_DIR / "year_stats.json").write_text(json.dumps({"mean": year_mean, "std": year_std}))
        (MODELS_DIR / "metrics.json").write_text(json.dumps(new_metrics, indent=2))
        print(f"promoted new model (val_accuracy={val_accuracy:.3f})")
    else:
        print(f"kept existing model (new val_accuracy={val_accuracy:.3f} < deployed {old_metrics['val_accuracy']:.3f})")

    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "promoted": promoted, **new_metrics}
    write_header = not HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
