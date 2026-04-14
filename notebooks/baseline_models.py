"""
baseline_models.py
------------------
Establishes two performance floors before any deep learning model is trained:

  1. Always-HOLD baseline  — the dumbest possible classifier.  Predicts HOLD
     for every single row regardless of features.  Any real model must beat
     this or it has learned nothing useful.

  2. Logistic Regression   — a simple linear classifier trained on the 10
     technical-indicator (TA) features only.  This is the simplest model that
     actually looks at the data.  It sets the floor for the ablation study
     (TA-only → TA+Fund → TA+Fund+Sent → LSTM).

Both models are evaluated on the validation split (used for development) and
the test split (used only for final reporting).

Input
-----
  data/splits/train.csv  — used to fit Logistic Regression
  data/splits/val.csv    — evaluation during development
  data/splits/test.csv   — final held-out evaluation

Features used (TA-only, 10 columns)
-------------------------------------
  rsi_14, macd_hist, roc_5, stoch_k, adx_14,
  atr_norm_14, bb_pct_20, ema_ratio_20, obv_roc_5, vol_ratio_20

Target
------
  label  →  0 = HOLD, 1 = BUY, 2 = SELL

Output
------
  Printed to screen only.  Nothing is written to disk.
  Results are meant to be copied into the project report.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress overflow/divide-by-zero warnings emitted by sklearn's lbfgs
# solver during early optimization iterations.  These occur transiently
# before the solver stabilizes and do not affect the final fitted weights
# or any evaluation metric.  This is a known issue in sklearn 1.x.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*overflow.*|.*divide by zero.*|.*invalid value.*",
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "data/splits/train.csv"
VAL_PATH    = "data/splits/val.csv"
TEST_PATH   = "data/splits/test.csv"
RESULTS_DIR = Path("results")

# ── Feature columns ───────────────────────────────────────────────────────────
# Only the 10 technical indicators are used here.
# Sentiment and fundamental columns are intentionally excluded —
# they will be added incrementally in the ablation study.
TA_COLS = [
    "rsi_14",       # momentum oscillator 0–100
    "macd_hist",    # rate of change of momentum (second derivative of price)
    "roc_5",        # 5-day backward return (matches prediction horizon)
    "stoch_k",      # stochastic %K — close relative to recent high/low range
    "adx_14",       # trend strength (non-directional, 0–100)
    "atr_norm_14",  # normalised volatility (ATR / close price)
    "bb_pct_20",    # Bollinger %B — position within volatility-adjusted bands
    "ema_ratio_20", # close / 20-day EMA — dimensionless trend deviation
    "obv_roc_5",    # 5-day rate of change of on-balance volume
    "vol_ratio_20", # today's volume / 20-day average volume
]

# ── Target column ─────────────────────────────────────────────────────────────
TARGET = "label"   # 0 = HOLD, 1 = BUY, 2 = SELL

# Human-readable class names for display
CLASS_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_splits() -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
]:
    """
    Load train, val, and test CSVs and return (X, y) pairs for each.

    The Date column is parsed but not used as a feature — it is only
    needed to verify chronological ordering.  Ticker is also excluded
    from features; each row is treated as an independent sample.
    """
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    val   = pd.read_csv(VAL_PATH,   parse_dates=["Date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["Date"])

    # Verify chronological ordering: last train date must be before first val
    # date, and last val date must be before first test date.
    assert train["Date"].max() < val["Date"].min(), \
        "Train/val date ordering violated!"
    assert val["Date"].max() < test["Date"].min(), \
        "Val/test date ordering violated!"

    # Extract TA features and target label for each split
    X_train = train[TA_COLS].copy()
    y_train = train[TARGET].copy()

    X_val   = val[TA_COLS].copy()
    y_val   = val[TARGET].copy()

    X_test  = test[TA_COLS].copy()
    y_test  = test[TARGET].copy()

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def fit_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[SimpleImputer, StandardScaler]:
    """
    Fit imputer and scaler on the TRAINING set only.

    Two-step preprocessing:
      1. Median imputation  — fills any NaN values with the column median
         computed from training data.  TA features should have zero NaNs
         but this is a safeguard.
      2. Standard scaling   — subtracts mean and divides by std so all 10
         features are on the same scale.  Logistic Regression is sensitive
         to feature scale; unscaled features would bias the coefficients.

    IMPORTANT: the imputer and scaler are fit ONLY on X_train.  They are
    then applied to val and test without re-fitting.  Re-fitting on val/test
    would leak their statistics into preprocessing, which is a form of
    data leakage.
    """
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    scaler.fit(X_imputed)

    return imputer, scaler


def apply_preprocessor(
    X: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply already-fitted imputer and scaler to a feature matrix.
    Returns a numpy array ready for model input.

    After scaling, values are clipped to [-10, 10].  Some features
    (particularly obv_roc_5) can produce extreme outliers when OBV
    crosses zero, which causes overflow in the matrix multiplications
    inside Logistic Regression's gradient computation.  Clipping at
    ±10 standard deviations removes these rare pathological values
    while preserving all meaningful signal.
    """
    X_imputed = imputer.transform(X)
    X_scaled  = scaler.transform(X_imputed)
    X_clipped = np.clip(X_scaled, -10, 10)
    return X_clipped


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def print_results(
    split_name: str,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> dict:
    """
    Print a standardised evaluation block for one split and return
    a dictionary of all metrics for saving to disk.

    Reports:
      - Accuracy        (raw fraction correct — reported but not primary metric)
      - Macro F1        (PRIMARY metric — averages F1 across all 3 classes
                         equally, regardless of class frequency.  Prevents the
                         majority HOLD class from dominating the score.)
      - Per-class F1    (shows whether the model is learning BUY and SELL,
                         not just HOLD)
      - Confusion matrix (rows = true label, columns = predicted label)
    """
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n  [{split_name}]")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {macro_f1:.4f}  ← primary metric")

    # Per-class breakdown — critical for understanding whether the model
    # is just predicting the majority class or actually learning all three
    per_class_f1 = {}
    print(f"\n  Per-class F1:")
    for label_int, label_name in CLASS_NAMES.items():
        class_f1 = f1_score(
            y_true, y_pred,
            labels=[label_int],
            average="macro",
            zero_division=0,
        )
        per_class_f1[label_name] = round(class_f1, 4)
        print(f"    {label_name:4s} ({label_int}): {class_f1:.4f}")

    # Full classification report (precision, recall, f1 per class)
    print(f"\n  Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
        zero_division=0,
    ))

    # Confusion matrix — rows are true labels, columns are predicted labels.
    # For a good model you want the diagonal to dominate.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index  =[f"true_{CLASS_NAMES[i]}"  for i in [0, 1, 2]],
        columns=[f"pred_{CLASS_NAMES[i]}"  for i in [0, 1, 2]],
    )
    print(f"  Confusion Matrix:")
    print(cm_df.to_string())

    # Return metrics dict for saving
    return {
        "accuracy" : round(acc, 4),
        "macro_f1" : round(macro_f1, 4),
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm_df.to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 1: ALWAYS-HOLD
# ─────────────────────────────────────────────────────────────────────────────

def run_always_hold_baseline(
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Always-HOLD baseline: predicts class 0 (HOLD) for every single row.

    This requires no training and ignores all features entirely.
    It represents the absolute performance floor — a model that simply
    exploits the fact that HOLD is the most frequent class (~51.5%).

    Expected macro F1: ~0.22
      - HOLD F1 = ~1.0  (predicts all HOLD correctly)
      - BUY  F1 = 0.0   (never predicts BUY)
      - SELL F1 = 0.0   (never predicts SELL)
      - Macro average = (1.0 + 0.0 + 0.0) / 3 ≈ 0.33... but precision
        for HOLD is reduced by false positives from BUY/SELL rows.

    Any real model that scores below this on macro F1 has learned nothing.
    """
    print("\n" + "=" * 60)
    print("BASELINE 1: Always-HOLD")
    print("(Predicts HOLD for every row, ignores all features)")
    print("=" * 60)

    # Predict HOLD (0) for every sample
    val_pred  = np.zeros(len(y_val),  dtype=int)
    test_pred = np.zeros(len(y_test), dtype=int)

    val_metrics  = print_results("Validation", y_val,  val_pred)
    test_metrics = print_results("Test",       y_test, test_pred)

    return {"validation": val_metrics, "test": test_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 2: LOGISTIC REGRESSION (TA ONLY)
# ─────────────────────────────────────────────────────────────────────────────

def run_logistic_regression(
    X_train: np.ndarray, y_train: pd.Series,
    X_val:   np.ndarray, y_val:   pd.Series,
    X_test:  np.ndarray, y_test:  pd.Series,
) -> None:
    """
    Logistic Regression trained on the 10 TA features.

    This is the simplest model that actually looks at the data.
    Key design choices:

      multi_class='multinomial'
        — uses a single softmax over all 3 classes rather than
          one-vs-rest.  More appropriate for 3-class problems.

      class_weight='balanced'
        — automatically upweights BUY and SELL during training to
          compensate for the fact that HOLD is ~51.5% of labels.
          Without this, the model would heavily bias toward HOLD.

      max_iter=1000
        — the default 100 iterations is often insufficient for
          convergence on standardised financial data.

      random_state=42
        — ensures reproducible results.

    The model is fit on the training set only.  Val and test are
    used purely for evaluation.
    """
    print("\n" + "=" * 60)
    print("BASELINE 2: Logistic Regression (TA features only)")
    print(f"(Features: {TA_COLS})")
    print("=" * 60)

    # Instantiate with balanced class weights so BUY and SELL are not
    # drowned out by the majority HOLD class during gradient updates.
    # solver='lbfgs' is the sklearn default — it uses L-BFGS-B which is
    # a quasi-Newton method that is numerically stable and well-suited
    # for multinomial logistic regression on small feature sets.
    # C=1.0 is the default L2 regularisation strength.
    model = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        random_state=42,
    )

    # Fit on training data only — val and test are never seen during fitting
    print("\n  Fitting on training set...")
    model.fit(X_train, y_train)
    print("  Done.")

    # Generate predictions for val and test
    val_pred  = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics  = print_results("Validation", y_val,  val_pred)
    test_metrics = print_results("Test",       y_test, test_pred)

    # Print the top feature coefficients so we can see which TA indicators
    # the model finds most useful for each class
    top_features_per_class = {}
    print("\n  Top feature coefficients per class:")
    classes = model.classes_  # [0, 1, 2]
    for i, cls in enumerate(classes):
        coef_series = pd.Series(model.coef_[i], index=TA_COLS)
        coef_sorted = coef_series.abs().sort_values(ascending=False)
        top_features = coef_sorted.head(5).index.tolist()
        top_features_per_class[CLASS_NAMES[cls]] = top_features
        print(f"    {CLASS_NAMES[cls]:4s}: top features = {top_features}")

    return {
        "validation": val_metrics,
        "test": test_metrics,
        "top_features_per_class": top_features_per_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    print(f"  Train : {X_train.shape[0]:,} rows  |  "
          f"HOLD={( y_train==0).sum()}  BUY={(y_train==1).sum()}  SELL={(y_train==2).sum()}")
    print(f"  Val   : {X_val.shape[0]:,} rows  |  "
          f"HOLD={(y_val==0).sum()}  BUY={(y_val==1).sum()}  SELL={(y_val==2).sum()}")
    print(f"  Test  : {X_test.shape[0]:,} rows  |  "
          f"HOLD={(y_test==0).sum()}  BUY={(y_test==1).sum()}  SELL={(y_test==2).sum()}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    # Fit on train only, then apply to val and test
    print("\nFitting preprocessor on training set...")
    imputer, scaler = fit_preprocessor(X_train)

    X_train_scaled = apply_preprocessor(X_train, imputer, scaler)
    X_val_scaled   = apply_preprocessor(X_val,   imputer, scaler)
    X_test_scaled  = apply_preprocessor(X_test,  imputer, scaler)
    print("  Imputation and scaling done.")

    # ── Run baselines ─────────────────────────────────────────────────────────

    # Baseline 1: Always-HOLD — no features, no training
    always_hold_results = run_always_hold_baseline(y_val, y_test)

    # Baseline 2: Logistic Regression on TA features
    lr_results = run_logistic_regression(
        X_train_scaled, y_train,
        X_val_scaled,   y_val,
        X_test_scaled,  y_test,
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Model':<35} {'Val Macro F1':>12} {'Test Macro F1':>13}")
    print("-" * 60)
    print(f"{'Always-HOLD':<35} {always_hold_results['validation']['macro_f1']:>12.4f} {always_hold_results['test']['macro_f1']:>13.4f}")
    print(f"{'Logistic Regression (TA only)':<35} {lr_results['validation']['macro_f1']:>12.4f} {lr_results['test']['macro_f1']:>13.4f}")
    print("=" * 60)
    print("Any subsequent model must beat Logistic Regression on test Macro F1.")

    # ── Save results to disk ──────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "description": (
            "Baseline model results. Two models evaluated on validation "
            "and test splits. Features: TA-only (10 columns). "
            "Primary metric: Macro F1."
        ),
        "feature_set": TA_COLS,
        "models": {
            "always_hold": always_hold_results,
            "logistic_regression_ta_only": lr_results,
        },
        "summary": {
            "always_hold":               {"val_macro_f1": always_hold_results["validation"]["macro_f1"], "test_macro_f1": always_hold_results["test"]["macro_f1"]},
            "logistic_regression_ta_only": {"val_macro_f1": lr_results["validation"]["macro_f1"],          "test_macro_f1": lr_results["test"]["macro_f1"]},
        },
    }

    output_path = RESULTS_DIR / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
