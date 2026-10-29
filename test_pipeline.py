"""Tests for the training loop and the library validation guards."""
import numpy as np
import pandas as pd

import features
import train_local
import validate_library
from test_features import make_df


def healthy_df(n=40):
    years = [1990 + (i % 30) for i in range(n)]
    genres = ["Action", "Drama", "Action|Drama", "Comedy"] * (n // 4)
    labels = [1 if i % 4 == 0 else 0 for i in range(n)]
    return pd.DataFrame(
        {
            "item_id": [str(i) for i in range(n)],
            "title": [f"title {i}" for i in range(n)],
            "year": years,
            "genres": genres[:n],
            "label": labels,
        }
    )


def test_training_is_deterministic():
    """CI asserts on an accuracy floor, which only means something if the same
    data always trains the same model."""
    df = healthy_df()
    X, y = features.transform(df, features.fit(df)), features.labels(df)

    first = train_local.train(X, y)
    second = train_local.train(X, y)

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_training_separates_a_learnable_signal():
    """Items labelled watched are all Action here, so the model should score
    Action above Comedy."""
    n = 40
    df = pd.DataFrame(
        {
            "item_id": [str(i) for i in range(n)],
            "title": [f"title {i}" for i in range(n)],
            "year": [2000] * n,
            "genres": ["Action" if i % 2 == 0 else "Comedy" for i in range(n)],
            "label": [1 if i % 2 == 0 else 0 for i in range(n)],
        }
    )
    encoding = features.fit(df)
    X, y = features.transform(df, encoding), features.labels(df)

    weight, bias = train_local.train(X, y)
    scores = train_local.sigmoid(X @ weight + bias)

    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_evaluate_is_perfect_on_separable_scores():
    y = np.array([1.0, 1.0, 0.0, 0.0])
    # a single feature that already separates the classes
    X = np.array([[10.0], [10.0], [-10.0], [-10.0]])

    accuracy, precision_at_k, k = train_local.evaluate(X, y, np.array([1.0]), 0.0, k=2)

    assert accuracy == 1.0
    assert precision_at_k == 1.0
    assert k == 2


def test_validation_accepts_a_healthy_library():
    assert validate_library.validate(healthy_df()) == []


def test_validation_rejects_a_saturated_label():
    """The regression that started this: 51 of 53 rows positive because
    'requested' was folded into the label."""
    df = healthy_df()
    df["label"] = 1

    problems = validate_library.validate(df)

    assert any("saturated" in p for p in problems)


def test_validation_rejects_an_empty_label():
    df = healthy_df()
    df["label"] = 0

    problems = validate_library.validate(df)

    assert any("no signal to learn" in p for p in problems)


def test_validation_rejects_missing_columns():
    problems = validate_library.validate(healthy_df().drop(columns=["genres"]))

    assert len(problems) == 1
    assert "genres" in problems[0]


def test_validation_rejects_duplicate_item_ids():
    df = healthy_df()
    df.loc[1, "item_id"] = df.loc[0, "item_id"]

    problems = validate_library.validate(df)

    assert any("duplicate item_id" in p for p in problems)


def test_validation_rejects_a_tiny_library():
    problems = validate_library.validate(healthy_df(n=4))

    assert any("expected at least" in p for p in problems)


def test_validation_rejects_an_implausible_year():
    df = healthy_df()
    df.loc[0, "year"] = 12000

    problems = validate_library.validate(df)

    assert any("implausible year" in p for p in problems)


def test_validation_rejects_a_library_with_no_genres():
    df = healthy_df()
    df["genres"] = ""

    problems = validate_library.validate(df)

    assert any("distinct genre" in p for p in problems)


def test_real_library_passes_validation():
    """The committed library.csv is what CI trains on, so it must stay valid."""
    df = pd.read_csv("library.csv")

    assert validate_library.validate(df) == []
