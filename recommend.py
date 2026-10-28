"""Local inference: recommend movies using the model trained in train.ipynb on Colab.

Needs only numpy + pandas (see requirements-infer.txt) - no torch required here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"


def load_artifacts():
    user_emb = np.load(MODELS_DIR / "user_emb.npy")
    item_emb = np.load(MODELS_DIR / "item_emb.npy")
    user_bias = np.load(MODELS_DIR / "user_bias.npy")
    item_bias = np.load(MODELS_DIR / "item_bias.npy")
    global_bias = json.loads((MODELS_DIR / "global_bias.json").read_text())["global_bias"]
    user_id_map = json.loads((MODELS_DIR / "user_id_map.json").read_text())
    item_id_map = json.loads((MODELS_DIR / "item_id_map.json").read_text())
    seen_items = json.loads((MODELS_DIR / "seen_items.json").read_text())
    movies = pd.read_csv(MODELS_DIR / "movies.csv")
    return user_emb, item_emb, user_bias, item_bias, global_bias, user_id_map, item_id_map, seen_items, movies


def recommend(user_id: str, top_n: int = 10):
    (user_emb, item_emb, user_bias, item_bias, global_bias,
     user_id_map, item_id_map, seen_items, movies) = load_artifacts()

    if user_id not in user_id_map:
        sample = ", ".join(list(user_id_map)[:5])
        raise SystemExit(f"Unknown user_id '{user_id}'. Example valid ids: {sample}, ...")

    u = user_id_map[user_id]
    scores = global_bias + user_bias[u] + item_bias + item_emb @ user_emb[u]

    already_seen = {item_id_map[i] for i in seen_items.get(user_id, []) if i in item_id_map}
    ranked = np.argsort(-scores)
    idx_to_item_id = {v: k for k, v in item_id_map.items()}
    titles_by_id = movies.set_index("item_id")["title"]

    print(f"Top {top_n} recommendations for user {user_id}:")
    shown = 0
    for item_idx in ranked:
        if item_idx in already_seen:
            continue
        item_id = idx_to_item_id[item_idx]
        title = titles_by_id.get(int(item_id), "Unknown title")
        print(f"{shown + 1:>2}. {title}  (predicted rating: {scores[item_idx]:.2f})")
        shown += 1
        if shown == top_n:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend movies from the matrix-factorization model trained on Colab.")
    parser.add_argument("--user_id", required=True, help="MovieLens user id, e.g. 42")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()
    recommend(args.user_id, args.top_n)
