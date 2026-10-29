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

## Scheduling (NixOS)

`nix/service.nix` is a systemd timer that runs the four stages nightly. Add it
to your configuration and rebuild:

```nix
imports = [ /root/hvad-skal-vi-seerr/nix/service.nix ];
```

Credentials come from a gitignored `.env.local` in the repo root (`KEY=VALUE`
per line, no `export`) via systemd's `EnvironmentFile`:

```
JELLYFIN_URL=http://...
JELLYFIN_API_KEY=...
JELLYFIN_USER_ID=...
JELLYSEERR_URL=http://...
JELLYSEERR_API_KEY=...
```

The unit pins its own Python (`python3.withPackages`) rather than calling
`nix-shell` at runtime, and `Type = "oneshot"` runs the stages in order and
skips the rest if one fails - so a failed validation stops the retrain.

```
systemctl start hvad-skal-vi-seerr     # run it now
systemctl list-timers hvad-skal-vi-seerr
journalctl -u hvad-skal-vi-seerr -n 50
```
