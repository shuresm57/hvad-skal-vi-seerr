"""Tests for the discover stage's candidate filtering and conversion.

Both are pure functions over Jellyseerr's response shape, so they are tested
without touching the network.
"""
from datetime import date

from pipeline import discover

GENRES_BY_ID = {28: "Action", 18: "Drama", 878: "Science Fiction"}
TODAY = date(2026, 10, 29)


def candidate(**overrides):
    item = {
        "id": 1234,
        "mediaType": "movie",
        "title": "Some Movie",
        "releaseDate": "2020-05-01",
        "genreIds": [28, 18],
        "voteCount": 500,
        "voteAverage": 7.1,
        "adult": False,
    }
    item.update(overrides)
    return item


def test_plain_released_movie_is_eligible():
    assert discover.is_eligible(candidate(), min_votes=50, today=TODAY)


def test_titles_jellyseerr_already_knows_are_skipped():
    """A mediaInfo record means owned, requested, or downloading."""
    owned = candidate(mediaInfo={"status": 5})

    assert not discover.is_eligible(owned, min_votes=50, today=TODAY)


def test_unreleased_titles_are_skipped():
    assert not discover.is_eligible(
        candidate(releaseDate="2027-01-01"), min_votes=50, today=TODAY
    )


def test_missing_release_date_is_skipped():
    for bad in ("", None):
        assert not discover.is_eligible(
            candidate(releaseDate=bad), min_votes=50, today=TODAY
        )


def test_obscure_and_adult_titles_are_skipped():
    assert not discover.is_eligible(candidate(voteCount=3), min_votes=50, today=TODAY)
    assert not discover.is_eligible(candidate(adult=True), min_votes=50, today=TODAY)


def test_non_movies_are_skipped():
    assert not discover.is_eligible(
        candidate(mediaType="tv"), min_votes=50, today=TODAY
    )


def test_missing_vote_count_is_treated_as_zero():
    assert not discover.is_eligible(
        candidate(voteCount=None), min_votes=50, today=TODAY
    )


def test_to_frame_maps_genre_ids_to_the_models_vocabulary():
    df = discover.to_frame([candidate()], GENRES_BY_ID)

    assert df.loc[0, "genres"] == "Action|Drama"
    assert df.loc[0, "year"] == 2020
    assert df.loc[0, "tmdb_id"] == 1234


def test_to_frame_drops_unknown_genre_ids():
    df = discover.to_frame([candidate(genreIds=[28, 99999])], GENRES_BY_ID)

    assert df.loc[0, "genres"] == "Action"


def test_to_frame_output_is_scorable():
    """The frame must carry the columns the shared encoder needs, since the
    whole point is that discover reuses the model unchanged."""
    from pipeline import features

    df = discover.to_frame([candidate(), candidate(genreIds=[878])], GENRES_BY_ID)

    for column in ("year", "genres"):
        assert column in df.columns

    encoding = features.fit(df)
    assert features.transform(df, encoding).shape == (2, encoding.n_features)
