# hvad-skal-vi-seerr

Recommends what to watch next from your own Jellyfin library, trained on your
Jellyfin watch history + Jellyseerr/Overseerr requests. Small on purpose:
train on Colab, serve locally with almost no dependencies.

```
export_history.py (your PC)  ->  train.ipynb (Colab)  ->  recommend.py (your PC)
pulls Jellyfin + Jellyseerr       trains a tiny genre/year      ranks your unwatched
history into library.csv          classifier, exports it        library items
```

Content-based, not collaborative filtering - it's one person's history, no
crowd to collaborate with.

## Run it

```
# 1. export your history (needs network access to your server)
pip install -r requirements-export.txt
export JELLYFIN_URL=... JELLYFIN_API_KEY=... JELLYFIN_USER_ID=...
export JELLYSEERR_URL=... JELLYSEERR_API_KEY=...   # optional
python export_history.py --list-users   # find your user id
python export_history.py                # writes library.csv (gitignored)

# 2. train on Colab: open train.ipynb, Runtime > Run all, upload library.csv,
#    download models.zip, unzip into ./models

# 3. recommend locally
pip install -r requirements-infer.txt
python recommend.py --library library.csv
```

## Files

- `export_history.py` - Jellyfin/Jellyseerr -> `library.csv`
- `train.ipynb` - Colab-only training
- `recommend.py` - local inference, numpy/pandas only
- `requirements-{export,train,infer}.txt` - deliberately separate envs
