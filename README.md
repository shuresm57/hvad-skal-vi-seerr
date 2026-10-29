# hvad-skal-vi-seerr

[![ci](https://github.com/shuresm57/hvad-skal-vi-seerr/actions/workflows/ci.yml/badge.svg)](https://github.com/shuresm57/hvad-skal-vi-seerr/actions/workflows/ci.yml)

Recommends what to watch next from your own Jellyfin library, trained on your
Jellyfin watch history + Jellyseerr/Overseerr requests, and pushed back into
Jellyfin as a Collection so it shows up in the app.

Content-based, not collaborative filtering - it's one person's history, no
crowd to collaborate with.

```
export_history.py -> validate_library.py -> train_local.py -> recommend.py
pulls Jellyfin +     refuses to train on    promotion-gated   ranks unwatched items,
Jellyseerr history   a broken export        retrain           pushes top picks into
into library.csv                                              a Jellyfin Collection
```

## The MLOps bits

**One feature encoding, shared.** `features.py` is the only place in the
scheduled pipeline that turns a library row into a model input. Training fits
the encoding and saves it next to the weights; serving loads and applies it.
This was a real bug, not a hypothetical: training used to impute a missing year
with the column median while serving used the stored mean, so the model scored
items with a different encoding than it was trained on. Train/serve skew fails
silently, which is why `test_features.py` pins the two paths to produce
identical matrices. (`train.ipynb` still carries its own copy, since it has to
run on Colab without the repo checked out - a decent illustration of why
notebooks drift away from the code that ships.)

**Validation before training.** `validate_library.py` refuses a library whose
label has saturated, lost its genres, or shrunk to nothing. Also from
experience: the first export labelled anything ever *requested* in Jellyseerr
as positive, and since requesting is how items get downloaded, 51 of 53 rows
came back positive. Training "succeeded" and learned nothing.

**A promotion gate.** `train_local.py` only overwrites the deployed model in
`models/` if the new one's validation accuracy is at least as good. Every
attempt, promoted or not, is appended to `metrics_history.csv`.

**CI that actually trains the model.** `library.csv` is committed, so GitHub
Actions runs the unit tests, validates the data, and trains on every push -
with no credentials and no network access to the media server. Training is
deterministic (fixed seed, fixed split), so the workflow can assert an
accuracy floor rather than just checking the script doesn't crash.

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

pip install -r requirements-infer.txt   # covers export, train_local, recommend
python export_history.py --list-users   # find your user id, once
python export_history.py                # writes library.csv
python validate_library.py              # sanity-check the export
python train_local.py                   # retrains, promotion-gated
python recommend.py                     # ranks + pushes to Jellyfin
```

Tests:

```
pip install -r requirements-dev.txt
pytest -q
```

## Automating it

Put the pipeline in a cron job. On a NixOS box without a system-wide Python:

```
0 4 * * * cd /root/hvad-skal-vi-seerr && set -a && . .env.local && set +a && nix-shell -p python3 python3Packages.numpy python3Packages.pandas python3Packages.requests --run "python3 export_history.py && python3 validate_library.py && python3 train_local.py && python3 recommend.py" >> /var/log/hvad-skal-vi-seerr.log 2>&1
```

`set -a` before sourcing `.env.local` is required - plain `. .env.local` only
sets shell variables, it doesn't export them to the Python processes. The `&&`
chain matters too: a failed validation stops the retrain instead of letting it
proceed on a broken export.

## Files

- `export_history.py` - Jellyfin/Jellyseerr -> `library.csv`
- `validate_library.py` - data guards; exits non-zero on a bad export
- `features.py` - the shared feature encoding (single source of truth)
- `train_local.py` - the real retraining entrypoint (numpy, promotion-gated)
- `recommend.py` - ranks unwatched items, pushes the top N into a Jellyfin Collection
- `train.ipynb` - the original one-off Colab training demo (PyTorch)
- `test_features.py`, `test_pipeline.py` - unit tests, run in CI
- `metrics_history.csv` - one row per retrain attempt, promoted or not
- `.github/workflows/ci.yml` - tests, data validation, and a training run per push
