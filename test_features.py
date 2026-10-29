"""Tests for the shared feature encoding.

The important one is test_train_and_serve_encode_identically: it is a
regression test for real train/serve skew, where training imputed a missing
year with the column median while serving imputed it with the stored mean.
"""
import json

import numpy as np
import pandas as pd
import pytest

import features


def make_df(years=None, genres=None, labels=None):
    """Build a small library frame. Row count follows whichever argument is
    given, so callers can pass just one list of any length."""
    lengths = {len(a) for a in (years, genres, labels) if a is not None}
    if len(lengths) > 1:
        raise ValueError(f"make_df got mismatched lengths: {sorted(lengths)}")
    n = lengths.pop() if lengths else 4

    def cycled(default):
        return [default[i % len(default)] for i in range(n)]

    years = years if years is not None else cycled([2000, 2010, 2020, 1990])
    genres = genres if genres is not None else cycled(["Action", "Drama", "Action|Drama", ""])
    labels = labels if labels is not None else cycled([1, 0, 1, 0])

    return pd.DataFrame(
        {
            "item_id": [str(i) for i in range(n)],
            "title": [f"title {i}" for i in range(n)],
            "year": years,
            "genres": genres,
            "label": labels,
        }
    )


def test_train_and_serve_encode_identically(tmp_path):
    """The encoding that training fit must survive a save/load round trip
    and produce a byte-identical matrix at serve time."""
    df = make_df(years=[2000, 2010, 2020, None])

    encoding = features.fit(df)
    train_X = features.transform(df, encoding)

    features.save(encoding, tmp_path)
    serve_X = features.transform(df, features.load(tmp_path))

    np.testing.assert_array_equal(train_X, serve_X)


def test_missing_year_is_imputed_with_the_median():
    """Pinned explicitly because the old serve path reached for the mean, and
    on skewed data the two differ enough to move the feature."""
    df = make_df(years=[2000, 2001, 2002, 2100, None])

    encoding = features.fit(df)

    assert encoding.year_fill == 2001.5  # median of the four known years
    assert encoding.year_fill != pytest.approx(np.mean([2000, 2001, 2002, 2100]))


def test_genre_vocabulary_is_sorted_and_positional():
    df = make_df(genres=["Drama", "Action", "Action|Drama", ""])

    encoding = features.fit(df)
    X = features.transform(df, encoding)

    assert encoding.genre_vocab == ["Action", "Drama"]
    assert X[0, 0] == 0.0 and X[0, 1] == 1.0  # Drama only
    assert X[1, 0] == 1.0 and X[1, 1] == 0.0  # Action only
    assert X[2, 0] == 1.0 and X[2, 1] == 1.0  # both
    assert X[3, 0] == 0.0 and X[3, 1] == 0.0  # neither


def test_genre_unseen_during_fit_is_ignored_at_serve_time():
    """A newly added genre must not crash scoring or shift the vector, since
    the model has no weight for it."""
    encoding = features.fit(make_df(genres=["Action", "Drama", "Action", "Drama"]))

    serve_df = make_df(genres=["Action|Documentary", "Documentary", "Action", "Drama"])
    X = features.transform(serve_df, encoding)

    assert X.shape[1] == encoding.n_features
    assert X[0, 0] == 1.0  # Action still registers
    assert X[1, :-1].sum() == 0.0  # Documentary alone contributes nothing


def test_single_row_does_not_produce_nan_features():
    """std is NaN for one row and 0.0 when all years match; either would
    poison every feature vector."""
    for years in ([2000], [2000, 2000, 2000]):
        X = features.transform(make_df(years=years), features.fit(make_df(years=years)))
        assert np.isfinite(X).all(), f"non-finite features for years={years}"


def test_all_years_missing_still_encodes():
    df = make_df(years=[None, None, None, None])

    X = features.transform(df, features.fit(df))

    assert np.isfinite(X).all()


def test_non_numeric_year_is_treated_as_missing():
    df = make_df(years=[2000, 2010, "", "not a year"])

    encoding = features.fit(df)
    X = features.transform(df, encoding)

    assert np.isfinite(X).all()
    assert X[2, -1] == X[3, -1]  # both imputed to the same fill value


def test_load_rejects_an_encoding_without_a_fill_value(tmp_path):
    """Old artifacts lack 'fill'. Guessing one would silently reintroduce the
    skew, so loading must fail loudly instead."""
    (tmp_path / "genre_vocab.json").write_text(json.dumps(["Action"]))
    (tmp_path / "year_stats.json").write_text(json.dumps({"mean": 2000.0, "std": 10.0}))

    with pytest.raises(SystemExit, match="rerun train_local.py|Rerun train_local.py"):
        features.load(tmp_path)


def test_labels_are_floats():
    y = features.labels(make_df(labels=[1, 0, 1, 0]))

    assert y.dtype == np.float64
    assert y.tolist() == [1.0, 0.0, 1.0, 0.0]
