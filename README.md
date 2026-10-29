# hvad-skal-vi-seerr

[![ci](https://github.com/shuresm57/hvad-skal-vi-seerr/actions/workflows/ci.yml/badge.svg)](https://github.com/shuresm57/hvad-skal-vi-seerr/actions/workflows/ci.yml)

Recommends what to watch next from your own Jellyfin library, trained on your
Jellyfin watch history + Jellyseerr/Overseerr requests, and pushed back into
Jellyfin as a Collection so it shows up in the app.

Content-based, not collaborative filtering - it's one person's history, no
crowd to collaborate with.

```
export_history -> validate_library -> train_local -> recommend
pulls Jellyfin +   refuses to train    promotion-     ranks unwatched items,
Jellyseerr into    on a broken         gated          pushes top picks into
data/library.csv   export              retrain        a Jellyfin Collection
```

## Layout

```
pipeline/      the four stages above, plus features.py (shared encoding)
tests/         unit tests, run in CI
data/          library.csv (your export) and metrics_history.csv
models/        the deployed model - weights plus its feature encoding
notebooks/     the original one-off Colab training demo
requirements/  dependencies, split per environment
```

Each stage is a module, run from the repo root:

```
python -m pipeline.export_history
python -m pipeline.validate_library
python -m pipeline.train_local
python -m pipeline.recommend
```

## The MLOps bits

**One feature encoding, shared.** `pipeline/features.py` is the only place in
the pipeline that turns a library row into a model input. Training fits the
encoding and saves it next to the weights; serving loads and applies it. This
was a real bug, not a hypothetical: training used to impute a missing year with
the column median while serving used the stored mean, so the model scored items
with a different encoding than it was trained on. Train/serve skew fails
silently, which is why `tests/test_features.py` pins the two paths to produce
identical matrices. (`notebooks/train.ipynb` still carries its own copy, since
it has to run on Colab without the repo checked out - a decent illustration of
why notebooks drift away from the code that ships.)

**Validation before training.** `pipeline/validate_library.py` refuses a
library whose label has saturated, lost its genres, or shrunk to nothing. Also
from experience: the first export labelled anything ever *requested* in
Jellyseerr as positive, and since requesting is how items get downloaded, 51 of
53 rows came back positive. Training "succeeded" and learned nothing.

**A promotion gate.** `pipeline/train_local.py` only overwrites the deployed
model in `models/` if the new one's validation accuracy is at least as good.
Every attempt, promoted or not, is appended to `data/metrics_history.csv`. The
model artifacts are committed, so the repo doubles as its own model history.

**CI that actually trains the model.** `data/library.csv` is committed, so
GitHub Actions runs the unit tests, validates the data, and trains on every
push - with no credentials and no network access to the media server. Training
is deterministic (fixed seed, fixed split), so the workflow asserts an accuracy
floor rather than just checking the script doesn't crash.

### Known limitations

- The promotion gate compares accuracy across runs, but as `library.csv` grows
  the validation split changes, so the two numbers come from different data. A
  frozen holdout set would make the comparison exact.
- 53 items with 10 positives means the reported accuracy is mostly noise. The
  pipeline is the point; the model gets meaningful as the history grows.

## Run it

```
export JELLYFIN_URL=... JELLYFIN_API_KEY=... JELLYFIN_USER_ID=...
export JELLYSEERR_URL=... JELLYSEERR_API_KEY=...   # optional

pip install -r requirements/infer.txt        # covers all four stages
python -m pipeline.export_history --list-users   # find your user id, once
python -m pipeline.export_history            # writes data/library.csv
python -m pipeline.validate_library          # sanity-check the export
python -m pipeline.train_local               # retrains, promotion-gated
python -m pipeline.recommend                 # ranks + pushes to Jellyfin
```

Tests:

```
pip install -r requirements/dev.txt
pytest -q
```

## Automating it

Put the pipeline in a cron job. On a NixOS box without a system-wide Python:

```
0 4 * * * cd /root/hvad-skal-vi-seerr && set -a && . .env.local && set +a && nix-shell -p python3 python3Packages.numpy python3Packages.pandas python3Packages.requests --run "python3 -m pipeline.export_history && python3 -m pipeline.validate_library && python3 -m pipeline.train_local && python3 -m pipeline.recommend" >> /var/log/hvad-skal-vi-seerr.log 2>&1
```

`set -a` before sourcing `.env.local` is required - plain `. .env.local` only
sets shell variables, it doesn't export them to the Python processes. The `&&`
chain matters too: a failed validation stops the retrain instead of letting it
proceed on a broken export.
