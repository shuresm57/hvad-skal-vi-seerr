# hvad-skal-vi-seerr

Recommends what to watch next from your own Jellyfin library, trained on your
Jellyfin watch history + Jellyseerr/Overseerr requests, and pushed back into
Jellyfin as a Collection so it shows up in the app.

Content-based, not collaborative filtering - it's one person's history, no
crowd to collaborate with.

```
export_history.py  ->  train_local.py / train.ipynb  ->  recommend.py
pulls Jellyfin + Jellyseerr   trains genre/year classifier,     ranks unwatched items,
history into library.csv      promotion-gated against the       pushes top picks into
                               currently deployed model          a Jellyfin Collection
```

`train.ipynb` is the one-off "trained on Colab" version. `train_local.py` is
the same model reimplemented in plain numpy (no torch needed - it's a linear
model on a few dozen rows, training costs milliseconds) and is what actually
runs on a schedule: it only overwrites the deployed model in `models/` if the
new one's validation accuracy is at least as good, and every attempt gets
logged to `metrics_history.csv` either way.

## Run it

```
export JELLYFIN_URL=... JELLYFIN_API_KEY=... JELLYFIN_USER_ID=...
export JELLYSEERR_URL=... JELLYSEERR_API_KEY=...   # optional

pip install -r requirements-infer.txt   # covers export + train_local + recommend
python export_history.py --list-users   # find your user id, once
python export_history.py                # writes library.csv
python train_local.py                   # retrains, promotion-gated
python recommend.py                     # ranks + pushes to Jellyfin
```

Put those three commands (export, train, recommend) in a cron job to automate it, e.g.:

```
0 4 * * * cd /root/hvad-skal-vi-seerr && . .env.local && python export_history.py && python train_local.py && python recommend.py >> /var/log/hvad-skal-vi-seerr.log 2>&1
```

## Files

- `export_history.py` - Jellyfin/Jellyseerr -> `library.csv`
- `train.ipynb` - one-off Colab training demo (PyTorch)
- `train_local.py` - the real retraining entrypoint (numpy, promotion-gated)
- `recommend.py` - ranks unwatched items, pushes the top N into a Jellyfin Collection
- `metrics_history.csv` - one row per retrain attempt, promoted or not
- `Dockerfile` - the whole pipeline as one container
