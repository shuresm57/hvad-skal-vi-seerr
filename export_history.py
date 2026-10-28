"""Export your personal watch + request history from Jellyfin and Jellyseerr/Overseerr
into a single CSV used to train the model in train.ipynb.

Run this on a machine that can reach your media server (usually your home network) -
not in a cloud sandbox. The output CSV contains your personal viewing data and is
gitignored; never commit it.

Required environment variables:
  JELLYFIN_URL        e.g. http://192.168.1.10:8096
  JELLYFIN_API_KEY    Jellyfin dashboard > API keys
  JELLYFIN_USER_ID    run with --list-users if you don't know it

Optional (skip to export Jellyfin data only):
  JELLYSEERR_URL      e.g. http://192.168.1.10:5055 (Jellyseerr or Overseerr both work)
  JELLYSEERR_API_KEY  Jellyseerr/Overseerr settings > API key
"""
import argparse
import csv
import os

import requests


def jellyfin_headers(api_key):
    return {"X-Emby-Token": api_key}


def list_jellyfin_users(base_url, api_key):
    resp = requests.get(f"{base_url}/Users", headers=jellyfin_headers(api_key), timeout=15)
    resp.raise_for_status()
    for user in resp.json():
        print(f"{user['Id']}  {user['Name']}")


def fetch_jellyfin_library(base_url, api_key, user_id):
    resp = requests.get(
        f"{base_url}/Users/{user_id}/Items",
        headers=jellyfin_headers(api_key),
        params={
            "Recursive": "true",
            "IncludeItemTypes": "Movie,Series",
            "Fields": "Genres,ProductionYear,UserData,ProviderIds",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["Items"]


def fetch_jellyseerr_requested_tmdb_ids(base_url, api_key):
    tmdb_ids = set()
    skip = 0
    while True:
        resp = requests.get(
            f"{base_url}/api/v1/request",
            headers={"X-Api-Key": api_key},
            params={"take": 100, "skip": skip},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        for req in payload.get("results", []):
            tmdb_id = (req.get("media") or {}).get("tmdbId")
            if tmdb_id is not None:
                tmdb_ids.add(str(tmdb_id))
        total = (payload.get("pageInfo") or {}).get("results", 0)
        skip += 100
        if skip >= total:
            break
    return tmdb_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-users", action="store_true", help="print Jellyfin users and exit")
    parser.add_argument("--out", default="library.csv")
    args = parser.parse_args()

    jellyfin_url = os.environ["JELLYFIN_URL"].rstrip("/")
    jellyfin_key = os.environ["JELLYFIN_API_KEY"]

    if args.list_users:
        list_jellyfin_users(jellyfin_url, jellyfin_key)
        return

    jellyfin_user_id = os.environ["JELLYFIN_USER_ID"]
    items = fetch_jellyfin_library(jellyfin_url, jellyfin_key, jellyfin_user_id)

    requested_tmdb_ids = set()
    jellyseerr_url = os.environ.get("JELLYSEERR_URL")
    jellyseerr_key = os.environ.get("JELLYSEERR_API_KEY")
    if jellyseerr_url and jellyseerr_key:
        requested_tmdb_ids = fetch_jellyseerr_requested_tmdb_ids(jellyseerr_url.rstrip("/"), jellyseerr_key)

    rows = []
    for item in items:
        user_data = item.get("UserData", {})
        watched = bool(user_data.get("Played")) or user_data.get("PlayCount", 0) > 0
        tmdb_id = (item.get("ProviderIds") or {}).get("Tmdb")
        requested = tmdb_id in requested_tmdb_ids if tmdb_id else False
        label = watched or requested or bool(user_data.get("IsFavorite"))
        rows.append({
            "item_id": item["Id"],
            "title": item.get("Name", ""),
            "year": item.get("ProductionYear", ""),
            "genres": "|".join(item.get("Genres", [])),
            "watched": int(watched),
            "requested": int(requested),
            "label": int(label),
        })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "title", "year", "genres", "watched", "requested", "label"])
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(r["label"] for r in rows)
    print(f"wrote {len(rows)} items to {args.out} ({positives} positive / {len(rows) - positives} negative)")


if __name__ == "__main__":
    main()
