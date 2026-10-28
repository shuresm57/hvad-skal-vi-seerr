"""Local inference: rank your own unwatched Jellyfin library using the classifier
trained in train.ipynb on Colab from your exported library.csv, then push the
top picks into a Jellyfin Collection so they show up in the Jellyfin app.

Needs numpy + pandas + requests (see requirements-infer.txt) - no torch required here.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

MODELS_DIR = Path(__file__).parent / "models"


def load_model():
    weights = np.load(MODELS_DIR / "weights.npy")
    bias = json.loads((MODELS_DIR / "bias.json").read_text())["bias"]
    genre_vocab = json.loads((MODELS_DIR / "genre_vocab.json").read_text())
    year_stats = json.loads((MODELS_DIR / "year_stats.json").read_text())
    return weights, bias, genre_vocab, year_stats


def featurize(df, genre_vocab, year_stats):
    genre_index = {g: i for i, g in enumerate(genre_vocab)}
    genre_lists = df["genres"].fillna("").apply(lambda g: g.split("|") if g else [])

    genre_features = np.zeros((len(df), len(genre_vocab)), dtype=np.float32)
    for row, genre_list in enumerate(genre_lists):
        for g in genre_list:
            if g in genre_index:
                genre_features[row, genre_index[g]] = 1.0

    year = df["year"].fillna(year_stats["mean"])
    year_feature = ((year - year_stats["mean"]) / year_stats["std"]).to_numpy(dtype=np.float32).reshape(-1, 1)

    return np.concatenate([genre_features, year_feature], axis=1)


def jellyfin_push(item_ids, collection_name):
    base_url = os.environ["JELLYFIN_URL"].rstrip("/")
    headers = {"X-Emby-Token": os.environ["JELLYFIN_API_KEY"]}
    user_id = os.environ["JELLYFIN_USER_ID"]

    search = requests.get(
        f"{base_url}/Users/{user_id}/Items",
        headers=headers,
        params={"IncludeItemTypes": "BoxSet", "Recursive": "true", "SearchTerm": collection_name},
        timeout=15,
    )
    search.raise_for_status()
    existing = next((i for i in search.json()["Items"] if i["Name"] == collection_name), None)

    if existing is None:
        requests.post(
            f"{base_url}/Collections",
            headers=headers,
            params={"Name": collection_name, "Ids": ",".join(item_ids), "userId": user_id},
            timeout=15,
        ).raise_for_status()
        print(f"created Jellyfin collection '{collection_name}' with {len(item_ids)} items")
        return

    collection_id = existing["Id"]
    current = requests.get(f"{base_url}/Items", headers=headers, params={"ParentId": collection_id}, timeout=15)
    current.raise_for_status()
    current_ids = [i["Id"] for i in current.json()["Items"]]
    if current_ids:
        requests.delete(
            f"{base_url}/Collections/{collection_id}/Items",
            headers=headers,
            params={"ids": ",".join(current_ids)},
            timeout=15,
        ).raise_for_status()

    requests.post(
        f"{base_url}/Collections/{collection_id}/Items",
        headers=headers,
        params={"ids": ",".join(item_ids)},
        timeout=15,
    ).raise_for_status()
    print(f"updated Jellyfin collection '{collection_name}' with {len(item_ids)} items")


def recommend(library_path: str, top_n: int, collection_name: str):
    weights, bias, genre_vocab, year_stats = load_model()
    df = pd.read_csv(library_path)

    unwatched = df[df["label"] == 0].reset_index(drop=True)
    if unwatched.empty:
        print("Nothing unwatched in this library export - time to request more.")
        return

    X = featurize(unwatched, genre_vocab, year_stats)
    scores = 1 / (1 + np.exp(-(X @ weights + bias)))
    unwatched = unwatched.assign(score=scores).sort_values("score", ascending=False)
    top = unwatched.head(top_n)

    print(f"Top {top_n} picks from what's already in your library:")
    for rank, row in enumerate(top.itertuples(), start=1):
        print(f"{rank:>2}. {row.title} ({row.year})  [{row.genres}]  score={row.score:.2f}")

    if os.environ.get("JELLYFIN_URL"):
        jellyfin_push([str(i) for i in top["item_id"]], collection_name)
    else:
        print("JELLYFIN_URL not set - skipped pushing to Jellyfin, printed only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank your unwatched Jellyfin library with the model trained on Colab.")
    parser.add_argument("--library", default="library.csv", help="path to the CSV from export_history.py")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--collection", default="Recommended for you", help="Jellyfin collection name to create/update")
    args = parser.parse_args()
    recommend(args.library, args.top_n, args.collection)
