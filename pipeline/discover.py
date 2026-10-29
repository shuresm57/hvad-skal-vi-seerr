"""Score titles you don't own yet, and optionally request them.

recommend.py ranks what is already in your library; this ranks what isn't.
Candidates come from Jellyseerr's discover feed and are scored by the same
deployed model through the same feature encoding, so a brand new data source
goes through exactly one code path.

Anything Jellyseerr already knows about is skipped: an item it has a mediaInfo
record for is owned, requested, or downloading.

Requesting is opt-in. --request N asks Jellyseerr to actually download the top
N, which is a real side effect, so it does nothing unless you pass it.
"""
import argparse
import os
from datetime import date

import pandas as pd
import requests

from . import model

SOURCES = {
    "popular": "/api/v1/discover/movies",
    "trending": "/api/v1/discover/trending",
    "upcoming": "/api/v1/discover/movies/upcoming",
}


def _session():
    base_url = os.environ["JELLYSEERR_URL"].rstrip("/")
    headers = {"X-Api-Key": os.environ["JELLYSEERR_API_KEY"]}
    return base_url, headers


def genre_names(base_url, headers) -> dict:
    """TMDB genre id -> name. The names line up with the genres Jellyfin
    reports, which is why the model's vocabulary applies to these candidates
    at all."""
    resp = requests.get(f"{base_url}/api/v1/genres/movie", headers=headers, timeout=30)
    resp.raise_for_status()
    return {g["id"]: g["name"] for g in resp.json()}


def fetch_candidates(base_url, headers, source, pages) -> list:
    items = []
    for page in range(1, pages + 1):
        resp = requests.get(
            f"{base_url}{SOURCES[source]}", headers=headers, params={"page": page}, timeout=30
        )
        resp.raise_for_status()
        items.extend(resp.json().get("results", []))
    return items


def is_eligible(item, min_votes, today=None) -> bool:
    today = today or date.today()

    if item.get("mediaType") != "movie":
        return False
    # A mediaInfo record means Jellyseerr already has it: owned, requested, or
    # mid-download. Either way it does not belong in a "what should I get" list.
    if item.get("mediaInfo") is not None:
        return False
    if item.get("adult"):
        return False
    if (item.get("voteCount") or 0) < min_votes:
        return False

    release = item.get("releaseDate") or ""
    if len(release) < 4:
        return False
    # Discover leans heavily on unreleased films; they are useless tonight.
    if release > today.isoformat():
        return False

    return True


def to_frame(items, genres_by_id) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tmdb_id": item["id"],
                "title": item.get("title", ""),
                "year": int(item["releaseDate"][:4]),
                "genres": "|".join(
                    genres_by_id[g] for g in item.get("genreIds", []) if g in genres_by_id
                ),
                "vote_average": item.get("voteAverage"),
            }
            for item in items
        ]
    )


def request_movie(base_url, headers, tmdb_id) -> None:
    resp = requests.post(
        f"{base_url}/api/v1/request",
        headers=headers,
        json={"mediaType": "movie", "mediaId": int(tmdb_id)},
        timeout=60,
    )
    resp.raise_for_status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), default="popular")
    parser.add_argument("--pages", type=int, default=5, help="discover pages to scan (20 per page)")
    parser.add_argument("--min-votes", type=int, default=50, help="skip obscure titles")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument(
        "--request",
        type=int,
        default=0,
        metavar="N",
        help="actually ask Jellyseerr to download the top N (default 0 = just print)",
    )
    args = parser.parse_args()

    base_url, headers = _session()
    genres_by_id = genre_names(base_url, headers)

    raw = fetch_candidates(base_url, headers, args.source, args.pages)
    eligible = [item for item in raw if is_eligible(item, args.min_votes)]
    if not eligible:
        print(f"no eligible candidates in {len(raw)} {args.source} results")
        return

    candidates = to_frame(eligible, genres_by_id)
    candidates = candidates.assign(score=model.score(candidates, model.load()))
    candidates = candidates.sort_values("score", ascending=False).reset_index(drop=True)

    print(
        f"scanned {len(raw)} {args.source} titles, {len(candidates)} not already in Jellyseerr"
    )
    print(f"Top {args.top_n} you don't own:")
    for rank, row in enumerate(candidates.head(args.top_n).itertuples(), start=1):
        print(
            f"{rank:>2}. {row.title} ({row.year})  [{row.genres}]  "
            f"score={row.score:.2f}  tmdb={row.tmdb_id}"
        )

    if args.request:
        for row in candidates.head(args.request).itertuples():
            request_movie(base_url, headers, row.tmdb_id)
            print(f"requested: {row.title} ({row.year})")


if __name__ == "__main__":
    main()
