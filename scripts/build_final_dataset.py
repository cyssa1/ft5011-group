"""Build data/final_dataset.csv from labeled_dataset.csv + technical_dataset.csv.

- Keep technical feature columns defined by technical_dataset.csv.
  Overlapping cols come from labeled_dataset (used for label computation);
  technical-only cols are merged in on (ticker, date).
- Add label_updown_1d (binary, next-day close-to-close return sign) and
  label_excess_5d (forward 5d return minus cross-sectional mean per date).
- Drop rows where either label is NaN.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LABELED_PATH = DATA / "labeled_dataset.csv"
TECHNICAL_PATH = DATA / "technical_dataset.csv"
OUT_PATH = DATA / "final_dataset.csv"

ID_COLS_TECH = {"Date", "Ticker"}
LABEL_COLS_TECH = {"signal", "label", "fwd_ret_5d"}
ID_COLS_LABELED = {"ticker", "date"}


def main() -> None:
    labeled = pd.read_csv(LABELED_PATH)
    technical = pd.read_csv(TECHNICAL_PATH)

    tech_feature_cols = [
        c for c in technical.columns
        if c not in ID_COLS_TECH and c not in LABEL_COLS_TECH
    ]
    overlap_cols = [c for c in tech_feature_cols if c in labeled.columns]
    extra_cols = [c for c in tech_feature_cols if c not in labeled.columns]

    df = labeled[["ticker", "date"] + overlap_cols + ["label"]].copy()
    df = df.rename(columns={"label": "label_multi"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    tech = technical.rename(columns={"Ticker": "ticker", "Date": "date"})[
        ["ticker", "date"] + extra_cols
    ].copy()
    tech["date"] = pd.to_datetime(tech["date"])
    df = df.merge(tech, on=["ticker", "date"], how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    feature_cols = overlap_cols + extra_cols

    grp = df.groupby("ticker", group_keys=False)
    next_close = grp["Close"].shift(-1)
    fwd_ret_1d = next_close / df["Close"] - 1.0
    df["label_updown_1d"] = (fwd_ret_1d > 0).astype("float")
    df.loc[fwd_ret_1d.isna(), "label_updown_1d"] = np.nan

    fwd_close_5 = grp["Close"].shift(-5)
    r_5d = fwd_close_5 / df["Close"] - 1.0
    cs_mean = r_5d.groupby(df["date"]).transform("mean")
    df["label_excess_5d"] = r_5d - cs_mean

    before = len(df)
    df = df.dropna(subset=["label_updown_1d", "label_excess_5d"]).reset_index(drop=True)
    df["label_updown_1d"] = df["label_updown_1d"].astype(int)
    dropped = before - len(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved -> {OUT_PATH}")
    print(f"rows: {len(df):,} (dropped {dropped} with NaN labels)")
    print(f"date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"tickers ({df['ticker'].nunique()}): {sorted(df['ticker'].unique().tolist())}")
    print(f"feature cols ({len(feature_cols)}): {feature_cols}")
    if extra_cols:
        na_frac = df[extra_cols].isna().mean().sort_values(ascending=False)
        print(
            f"merged from technical_dataset ({len(extra_cols)}): {extra_cols}\n"
            f"  NaN fraction (max={na_frac.max():.4f}, min={na_frac.min():.4f})"
        )
    up = df["label_updown_1d"]
    ex = df["label_excess_5d"]
    print(
        "label_updown_1d: "
        f"pos={up.mean():.4f} ({int(up.sum())}/{len(up)}), "
        f"neg={(1 - up.mean()):.4f}"
    )
    print(
        "label_excess_5d: "
        f"mean={ex.mean():.6f}, std={ex.std():.6f}, "
        f"min={ex.min():.6f}, max={ex.max():.6f}"
    )
    print("label_multi (from labeled_dataset):")
    print(df["label_multi"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()
