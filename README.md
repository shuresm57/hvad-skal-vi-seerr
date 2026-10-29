# hvad-skal-vi-seerr

[![ci](https://github.com/shuresm57/hvad-skal-vi-seerr/actions/workflows/ci.yml/badge.svg)](https://github.com/shuresm57/hvad-skal-vi-seerr/actions/workflows/ci.yml)

Recommends what to watch next from your own Jellyfin library, trained on your
watch history + Jellyseerr requests, pushed back into Jellyfin as a Collection.
Content-based, not collaborative filtering - it's one person's history.

```
export_history -> validate_library -> train_local -> recommend
```

- `pipeline/features.py` is the only encoder both training and serving use, so
  they can't drift apart (they had: median vs. mean imputation).
- `validate_library` refuses a saturated or empty label before training on it.
- `train_local` only promotes a model if validation accuracy holds up; every
  attempt lands in `data/metrics_history.csv`.
- `data/library.csv` is committed, so CI trains the model on every push with no
  credentials and no network. Training is deterministic, so CI asserts an
  accuracy floor.

Caveats: the promotion gate compares runs whose validation splits differ, and
at 53 items / 10 positives the accuracy is mostly noise. The pipeline is the point.

## Run

```
export JELLYFIN_URL=... JELLYFIN_API_KEY=... JELLYFIN_USER_ID=...
export JELLYSEERR_URL=... JELLYSEERR_API_KEY=...   # optional

pip install -r requirements/infer.txt
python -m pipeline.export_history --list-users   # find your user id, once
python -m pipeline.export_history                # -> data/library.csv
python -m pipeline.validate_library
python -m pipeline.train_local
python -m pipeline.recommend
```

Tests: `pip install -r requirements/dev.txt && pytest -q`

## Cron

```
0 4 * * * cd /root/hvad-skal-vi-seerr && set -a && . .env.local && set +a && nix-shell -p python3 python3Packages.numpy python3Packages.pandas python3Packages.requests --run "python3 -m pipeline.export_history && python3 -m pipeline.validate_library && python3 -m pipeline.train_local && python3 -m pipeline.recommend" >> /var/log/hvad-skal-vi-seerr.log 2>&1
```

`set -a` is required - plain `. .env.local` doesn't export to the Python
processes. The `&&` chain stops a retrain on a failed validation.
