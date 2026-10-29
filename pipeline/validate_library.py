"""Validate library.csv before it is trained on.

Run this between export_history.py and train_local.py. It exists because of a
real failure: the first export labelled an item positive if it had ever been
*requested* in Jellyseerr, and since requesting is how items get downloaded in
the first place, 51 of 53 rows came back positive. Training "succeeded" and
produced a model that had learned nothing. A saturated label is not a crash,
so only an explicit check catches it.

Exits non-zero with a readable reason on failure, so a cron chain stops
instead of silently retraining on garbage.
"""
import argparse
import sys

import pandas as pd

from . import LIBRARY_CSV, features

# A single-user library is naturally imbalanced - you've watched a minority of
# what you own - so these bounds are wide. They are here to catch a label that
# has collapsed to one class, not to enforce a particular balance.
MIN_POSITIVE_RATE = 0.02
MAX_POSITIVE_RATE = 0.90
MIN_ROWS = 10
MIN_GENRES = 2
PLAUSIBLE_YEARS = (1880, 2100)


def validate(df: pd.DataFrame) -> list:
    problems = []

    missing_columns = [c for c in features.REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        # Everything below indexes these columns, so stop here.
        return [f"missing required column(s): {', '.join(missing_columns)}"]

    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows, expected at least {MIN_ROWS}")

    bad_labels = set(df["label"].dropna().unique()) - {0, 1}
    if bad_labels:
        problems.append(f"label must be 0 or 1, found {sorted(bad_labels)}")

    if df["item_id"].isna().any():
        problems.append("item_id has blank values")
    duplicates = int(df["item_id"].duplicated().sum())
    if duplicates:
        problems.append(f"{duplicates} duplicate item_id value(s)")

    if len(df):
        positive_rate = float(df["label"].mean())
        if positive_rate < MIN_POSITIVE_RATE:
            problems.append(
                f"positive rate {positive_rate:.2%} is below {MIN_POSITIVE_RATE:.0%} - "
                "almost nothing is labelled watched, so there is no signal to learn"
            )
        elif positive_rate > MAX_POSITIVE_RATE:
            problems.append(
                f"positive rate {positive_rate:.2%} is above {MAX_POSITIVE_RATE:.0%} - "
                "the label has saturated and the model cannot learn a useful boundary"
            )

    genre_vocab = features.fit(df).genre_vocab
    if len(genre_vocab) < MIN_GENRES:
        problems.append(
            f"only {len(genre_vocab)} distinct genre(s); genres are the main "
            "feature, so this usually means the export lost its metadata"
        )

    years = pd.to_numeric(df["year"], errors="coerce").dropna()
    low, high = PLAUSIBLE_YEARS
    implausible = years[(years < low) | (years > high)]
    if len(implausible):
        problems.append(
            f"{len(implausible)} row(s) have an implausible year "
            f"(e.g. {int(implausible.iloc[0])}), expected {low}-{high}"
        )

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default=LIBRARY_CSV)
    args = parser.parse_args()

    df = pd.read_csv(args.library)
    problems = validate(df)

    if problems:
        print(f"{args.library} failed validation:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(1)

    print(
        f"{args.library} looks sane: {len(df)} rows, "
        f"{int(df['label'].sum())} positive ({df['label'].mean():.1%})"
    )


if __name__ == "__main__":
    main()
