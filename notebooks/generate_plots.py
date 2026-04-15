"""
generate_plots.py
-----------------
Generates three publication-ready figures from saved result JSON files.

Plots
-----
1. ablation_f1.png       — Feature-set ablation: Test Macro F1 for
                           XGBoost, LSTM, and Transformer side by side
2. equity_curves.png     — Portfolio value over time for all strategies
3. confidence_sharpe.png — Confidence threshold vs Sharpe ratio for
                           LSTM and Transformer

Output
------
  results/plots/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "sans-serif",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "figure.dpi"       : 150,
})

RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_XGB   = "#4C72B0"
C_LSTM  = "#DD8452"
C_TFM   = "#55A868"
C_BH    = "#808080"
C_MOM   = "#9467BD"
C_LR    = "#8C564B"

LABEL_SETS = ["TA only\n(10)", "TA+Sent\n(23)", "TA+Fund\n(25)", "TA+Fund+Sent\n(38)"]
SET_KEYS   = ["ta_only", "ta_sent", "ta_fund", "ta_fund_sent"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE-SET ABLATION — Test Macro F1
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_f1():
    # Load results
    with open(RESULTS_DIR / "ablation_results.json")             as f: xgb_data  = json.load(f)
    with open(RESULTS_DIR / "lstm_ablation_results.json")        as f: lstm_data = json.load(f)
    with open(RESULTS_DIR / "transformer_ablation_results.json") as f: tfm_data  = json.load(f)

    xgb_map  = {r["feature_set"]: r["test"]["macro_f1"]
                for r in xgb_data["stage2"]["results"]}
    lstm_map = {k: v["test_macro_f1"]
                for k, v in lstm_data["results"].items()}
    tfm_map  = {r["feature_set"]: r["test"]["macro_f1"]
                for r in tfm_data["results"]}

    xgb_f1  = [xgb_map.get(k, float("nan"))  for k in SET_KEYS]
    lstm_f1 = [lstm_map.get(k, float("nan")) for k in SET_KEYS]
    tfm_f1  = [tfm_map.get(k, float("nan"))  for k in SET_KEYS]

    x     = np.arange(len(SET_KEYS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width, xgb_f1,  width, label="XGBoost",     color=C_XGB,  alpha=0.85)
    b2 = ax.bar(x,         lstm_f1, width, label="LSTM",         color=C_LSTM, alpha=0.85)
    b3 = ax.bar(x + width, tfm_f1,  width, label="Transformer",  color=C_TFM,  alpha=0.85)

    # Value labels on bars
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_SETS)
    ax.set_ylabel("Test Macro F1")
    ax.set_title("Feature-Set Ablation — Test Macro F1\n(higher is better)", fontweight="bold")
    ax.legend(framealpha=0.5)
    ax.set_ylim(0, max(max(xgb_f1), max(lstm_f1), max(tfm_f1)) + 0.06)

    plt.tight_layout()
    out = PLOTS_DIR / "ablation_f1.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. EQUITY CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_equity_curves():
    with open(RESULTS_DIR / "trading_results.json") as f:
        data = json.load(f)

    results = data["results"]

    # Strategies to plot and their display names / colours
    strategies = {
        "buy_hold"          : ("Buy & Hold",            C_BH,   "--",  2.0),
        "momentum"          : ("Momentum",               C_MOM,  ":",   1.5),
        "xgboost"           : ("XGBoost (38 feat)",      C_XGB,  "-.",  1.5),
        "logistic_regression": ("Logistic Reg.",         C_LR,   "-.",  1.5),
        "lstm_conf_0.50"    : ("LSTM conf≥0.50",         C_LSTM, "-",   2.0),
        "tfm_conf_0.60"     : ("Transformer conf≥0.60",  C_TFM,  "-",   2.0),
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    for key, (label, color, ls, lw) in strategies.items():
        if key not in results:
            continue
        dv = results[key].get("daily_values", {})
        if not dv:
            continue
        series = pd.Series(dv)
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()
        # Normalise to 100 at start
        norm = series / series.iloc[0] * 100
        ax.plot(norm.index, norm.values, label=label, color=color,
                linestyle=ls, linewidth=lw)

    ax.axhline(100, color="black", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("Portfolio Value (indexed to 100)")
    ax.set_title("Portfolio Equity Curves — Test Period (Jul 2023 – Dec 2024)",
                 fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.6, fontsize=9)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b '%y"))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    out = PLOTS_DIR / "equity_curves.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONFIDENCE THRESHOLD vs SHARPE
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence_sharpe():
    with open(RESULTS_DIR / "trading_results.json") as f:
        data = json.load(f)

    results = data["results"]
    thresholds = [0.38, 0.42, 0.46, 0.50, 0.60]

    lstm_sharpe = []
    tfm_sharpe  = []
    lstm_trades = []
    tfm_trades  = []

    for t in thresholds:
        lk = f"lstm_conf_{t:.2f}"
        tk = f"tfm_conf_{t:.2f}"
        lstm_sharpe.append(results[lk]["portfolio"]["sharpe_ratio"] if lk in results else float("nan"))
        tfm_sharpe.append( results[tk]["portfolio"]["sharpe_ratio"] if tk in results else float("nan"))
        lstm_trades.append(results[lk]["portfolio"]["n_sells"]      if lk in results else 0)
        tfm_trades.append( results[tk]["portfolio"]["n_sells"]      if tk in results else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # ── Sharpe vs threshold ───────────────────────────────────────────────────
    ax1.plot(thresholds, lstm_sharpe, "o-", color=C_LSTM, linewidth=2,
             markersize=7, label="LSTM (ta_sent)")
    ax1.plot(thresholds, tfm_sharpe,  "s-", color=C_TFM,  linewidth=2,
             markersize=7, label="Transformer (ta_only)")

    # Annotate best points
    best_l = thresholds[int(np.nanargmax(lstm_sharpe))]
    best_t = thresholds[int(np.nanargmax(tfm_sharpe))]
    ax1.axvline(best_l, color=C_LSTM, linestyle="--", alpha=0.4)
    ax1.axvline(best_t, color=C_TFM,  linestyle="--", alpha=0.4)

    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Annualised Sharpe Ratio")
    ax1.set_title("Confidence Threshold vs Sharpe Ratio", fontweight="bold")
    ax1.legend(framealpha=0.6)
    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── Number of trades vs threshold ─────────────────────────────────────────
    ax2.plot(thresholds, lstm_trades, "o-", color=C_LSTM, linewidth=2,
             markersize=7, label="LSTM (ta_sent)")
    ax2.plot(thresholds, tfm_trades,  "s-", color=C_TFM,  linewidth=2,
             markersize=7, label="Transformer (ta_only)")

    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("Number of Trades")
    ax2.set_title("Confidence Threshold vs Trade Count", fontweight="bold")
    ax2.legend(framealpha=0.6)
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    out = PLOTS_DIR / "confidence_sharpe.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating plots...")
    print()

    print("1. Feature-set ablation F1 chart")
    plot_ablation_f1()

    print("2. Equity curves")
    plot_equity_curves()

    print("3. Confidence threshold vs Sharpe")
    plot_confidence_sharpe()

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
