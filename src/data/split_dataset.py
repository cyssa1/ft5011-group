"""
split_dataset.py
----------------
Splits master_dataset.csv into train / validation / test sets using a
strict chronological split.

Rules
-----
- Split is performed on unique trading DATES, not on rows, so that all
  10 tickers share identical split boundaries.  A single trading day
  never appears in two different splits.
- No shuffling.  Earlier dates always go to train; later dates to test.
- Ratio: 70% train / 15% validation / 15% test  (applied to unique dates).
- fwd_ret_5d is kept in all splits for post-hoc trading evaluation but
  must never be used as a model feature.

Outputs (saved to data/splits/)
--------------------------------
  train.csv   — rows whose Date falls in the training window
  val.csv     — rows whose Date falls in the validation window
  test.csv    — rows whose Date falls in the test window
  split_info.json — exact date boundaries for full reproducibility
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_PATH   = Path("data/master_dataset.csv")
OUTPUT_DIR  = Path("data/splits")
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test ratio = 1 - TRAIN_RATIO - VAL_RATIO = 0.15 (implicit)


def main() -> None:
    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    print(f"  Shape: {df.shape}  |  Tickers: {sorted(df['Ticker'].unique())}")

    # ── 2. Verify all tickers have the same date range ────────────────────────
    date_counts = df.groupby("Ticker")["Date"].count()
    assert date_counts.nunique() == 1, (
        f"Unequal row counts across tickers:\n{date_counts}"
    )
    rows_per_ticker = date_counts.iloc[0]
    print(f"  Rows per ticker: {rows_per_ticker}  (uniform — OK)")

    # ── 3. Compute split boundaries on unique dates ───────────────────────────
    unique_dates = df["Date"].drop_duplicates().sort_values().reset_index(drop=True)
    n_dates = len(unique_dates)

    train_end_idx = int(n_dates * TRAIN_RATIO)          # last index IN train
    val_end_idx   = int(n_dates * (TRAIN_RATIO + VAL_RATIO))  # last index IN val
    # test = everything after val_end_idx

    train_dates = set(unique_dates.iloc[:train_end_idx])
    val_dates   = set(unique_dates.iloc[train_end_idx:val_end_idx])
    test_dates  = set(unique_dates.iloc[val_end_idx:])

    print(f"\n  Total unique dates : {n_dates}")
    print(f"  Train dates        : {len(train_dates)}"
          f"  ({len(train_dates)/n_dates*100:.1f}%)"
          f"  {unique_dates.iloc[0].date()} → {unique_dates.iloc[train_end_idx-1].date()}")
    print(f"  Val dates          : {len(val_dates)}"
          f"  ({len(val_dates)/n_dates*100:.1f}%)"
          f"  {unique_dates.iloc[train_end_idx].date()} → {unique_dates.iloc[val_end_idx-1].date()}")
    print(f"  Test dates         : {len(test_dates)}"
          f"  ({len(test_dates)/n_dates*100:.1f}%)"
          f"  {unique_dates.iloc[val_end_idx].date()} → {unique_dates.iloc[-1].date()}")

    # ── 4. Sanity checks ──────────────────────────────────────────────────────
    assert len(train_dates & val_dates)  == 0, "Train / val date overlap!"
    assert len(train_dates & test_dates) == 0, "Train / test date overlap!"
    assert len(val_dates   & test_dates) == 0, "Val / test date overlap!"
    assert len(train_dates) + len(val_dates) + len(test_dates) == n_dates, \
        "Dates don't sum to total!"
    print("\n  Overlap checks passed.")

    # ── 5. Assign rows to splits ──────────────────────────────────────────────
    train_df = df[df["Date"].isin(train_dates)].copy()
    val_df   = df[df["Date"].isin(val_dates)].copy()
    test_df  = df[df["Date"].isin(test_dates)].copy()

    # Each split should have exactly (n_split_dates * 10) rows
    assert len(train_df) == len(train_dates) * 10, "Train row count mismatch!"
    assert len(val_df)   == len(val_dates)   * 10, "Val row count mismatch!"
    assert len(test_df)  == len(test_dates)  * 10, "Test row count mismatch!"
    print("  Row count checks passed.")

    # ── 6. Check label distribution in each split ─────────────────────────────
    print("\n  Label distribution:")
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split["label"].value_counts().sort_index()
        total  = len(split)
        dist   = {int(k): f"{v} ({v/total*100:.1f}%)" for k, v in counts.items()}
        print(f"    {name:5s}: {dist}")

    # ── 7. Save splits ────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR  / "val.csv",   index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv",  index=False)

    # Save exact boundaries for reproducibility
    split_info = {
        "source_file"      : str(DATA_PATH),
        "total_rows"       : int(len(df)),
        "total_unique_dates": int(n_dates),
        "train": {
            "n_dates"   : int(len(train_dates)),
            "n_rows"    : int(len(train_df)),
            "date_start": str(unique_dates.iloc[0].date()),
            "date_end"  : str(unique_dates.iloc[train_end_idx - 1].date()),
        },
        "val": {
            "n_dates"   : int(len(val_dates)),
            "n_rows"    : int(len(val_df)),
            "date_start": str(unique_dates.iloc[train_end_idx].date()),
            "date_end"  : str(unique_dates.iloc[val_end_idx - 1].date()),
        },
        "test": {
            "n_dates"   : int(len(test_dates)),
            "n_rows"    : int(len(test_df)),
            "date_start": str(unique_dates.iloc[val_end_idx].date()),
            "date_end"  : str(unique_dates.iloc[-1].date()),
        },
    }

    with open(OUTPUT_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n  Saved to {OUTPUT_DIR}/")
    print(f"    train.csv  — {len(train_df):,} rows")
    print(f"    val.csv    — {len(val_df):,} rows")
    print(f"    test.csv   — {len(test_df):,} rows")
    print(f"    split_info.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
