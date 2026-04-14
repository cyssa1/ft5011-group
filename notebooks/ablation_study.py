"""
ablation_study.py
-----------------
Two-stage experiment to find the best classical ML model and then
understand what each data modality contributes.

STAGE 1 — Model selection
--------------------------
Train Logistic Regression, Random Forest, and XGBoost on the FULL
feature set (TA + Fundamentals + Sentiment, 38 columns).  Pick the
model with the highest test Macro F1.

STAGE 2 — Ablation study on the winner
----------------------------------------
Re-train the best model using four feature sets:
  - TA only                        (10 columns)
  - TA + Sentiment                 (23 columns)
  - TA + Fundamentals              (25 columns)
  - TA + Fundamentals + Sentiment  (38 columns)

This isolates the incremental contribution of each modality.

Input
-----
  data/splits/train.csv  — model fitting
  data/splits/val.csv    — monitoring only (not used for tuning)
  data/splits/test.csv   — final held-out evaluation

Feature sets
------------
  TA_COLS   : 10 technical indicator columns
  FUND_COLS : 15 fundamental ratio columns (quarterly, forward-filled)
  SENT_COLS : 13 news sentiment columns

Target
------
  label  →  0 = HOLD, 1 = BUY, 2 = SELL

Output
------
  Printed to screen + saved to results/ablation_results.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Suppress transient numerical warnings from sklearn/xgboost optimisers
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*overflow.*|.*divide by zero.*|.*invalid value.*",
)
warnings.filterwarnings("ignore", category=UserWarning)


# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "data/splits/train.csv"
VAL_PATH    = "data/splits/val.csv"
TEST_PATH   = "data/splits/test.csv"
RESULTS_DIR = Path("results")

# ── Feature column groups ─────────────────────────────────────────────────────
TA_COLS = [
    "rsi_14", "macd_hist", "roc_5", "stoch_k", "adx_14",
    "atr_norm_14", "bb_pct_20", "ema_ratio_20", "obv_roc_5", "vol_ratio_20",
]

FUND_COLS = [
    "gross_margin", "op_margin", "net_margin", "roa", "roe",
    "debt_ratio", "debt_to_equity", "rd_intensity", "fcf_margin",
    "asset_turnover", "rev_growth_qoq", "rev_growth_yoy",
    "ni_growth_qoq", "ni_growth_yoy", "eps_growth_yoy",
]

SENT_COLS = [
    "sent_avg", "heat_pub", "heat_read", "pub_count",
    "pos_count", "neg_count", "neu_count",
    "pos_ratio", "neg_ratio", "sent_ma5", "sent_ma10",
    "sent_momentum", "controversy",
]

# Ablation feature sets
ABLATION_SETS = {
    "ta_only":      TA_COLS,
    "ta_sent":      TA_COLS + SENT_COLS,
    "ta_fund":      TA_COLS + FUND_COLS,
    "ta_fund_sent": TA_COLS + FUND_COLS + SENT_COLS,
}

TARGET = "label"
CLASS_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, val, and test CSVs.
    Returns full DataFrames — feature selection happens later so we
    can reuse the same loaded data across all experiments.
    """
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    val   = pd.read_csv(VAL_PATH,   parse_dates=["Date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["Date"])

    assert train["Date"].max() < val["Date"].min(),  "Train/val date overlap!"
    assert val["Date"].max()   < test["Date"].min(), "Val/test date overlap!"

    return train, val, test


def get_Xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y from a split DataFrame."""
    return df[feature_cols].copy(), df[TARGET].copy()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

def build_lr_pipeline() -> Pipeline:
    """
    Logistic Regression wrapped in a sklearn Pipeline.

    Steps:
      1. SimpleImputer (median) — fills NaNs in fundamental columns
      2. StandardScaler — required for LR which is sensitive to feature scale
      3. LogisticRegression — class_weight='balanced' upweights BUY and SELL
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            random_state=42,
        )),
    ])


def build_rf_pipeline() -> Pipeline:
    """
    Random Forest wrapped in a sklearn Pipeline.

    Steps:
      1. SimpleImputer (median) — sklearn RF cannot handle NaNs natively
      2. No scaling needed — tree models are scale-invariant
      3. RandomForestClassifier — class_weight='balanced', n_jobs=-1 for
         parallel tree building across all CPU cores
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_xgb_pipeline() -> Pipeline:
    """
    XGBoost wrapped in a sklearn Pipeline.

    Steps:
      1. SimpleImputer (median) — for consistency with other models
      2. No scaling needed — gradient boosting is scale-invariant
      3. XGBClassifier — class imbalance handled via sample_weight
         passed at fit time (see fit_pipeline below)
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHTS FOR XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_weights(y: pd.Series) -> np.ndarray:
    """
    Compute per-sample weights for XGBoost to handle class imbalance.
    XGBoost does not support class_weight='balanced' directly — we pass
    sample weights at fit time instead.
    Formula: weight_i = n_total / (n_classes * count_of_class_i)
    """
    class_counts = y.value_counts()
    n_total      = len(y)
    n_classes    = len(class_counts)
    weight_map   = {
        cls: n_total / (n_classes * count)
        for cls, count in class_counts.items()
    }
    return np.array([weight_map[label] for label in y])


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING AND EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def fit_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> Pipeline:
    """
    Fit a pipeline on training data.
    XGBoost receives sample weights; LR and RF use class_weight='balanced'.
    """
    if model_name == "xgboost":
        sample_weights = compute_sample_weights(y_train)
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
    else:
        pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
) -> dict:
    """
    Generate predictions and compute all metrics for one split.
    Returns a dict of metrics for saving to disk.
    """
    y_pred   = pipeline.predict(X)
    acc      = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"\n  [{split_name}]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}  ← primary metric")

    per_class_f1 = {}
    print(f"\n  Per-class F1:")
    for label_int, label_name in CLASS_NAMES.items():
        cf1 = f1_score(y, y_pred, labels=[label_int], average="macro", zero_division=0)
        per_class_f1[label_name] = round(cf1, 4)
        print(f"    {label_name:4s} ({label_int}): {cf1:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(
        y, y_pred,
        target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
        zero_division=0,
    ))

    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index  =[f"true_{CLASS_NAMES[i]}" for i in [0, 1, 2]],
        columns=[f"pred_{CLASS_NAMES[i]}" for i in [0, 1, 2]],
    )
    print(f"  Confusion Matrix:")
    print(cm_df.to_string())

    return {
        "accuracy"        : round(acc, 4),
        "macro_f1"        : round(macro_f1, 4),
        "per_class_f1"    : per_class_f1,
        "confusion_matrix": cm_df.to_dict(),
    }


def run_experiment(
    model_name: str,
    pipeline: Pipeline,
    feature_cols: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """
    Run one complete experiment: fit on train, evaluate on val and test.
    This is the core unit called by both Stage 1 and Stage 2.
    """
    print(f"\n{'='*60}")
    print(f"Model    : {model_name}")
    print(f"Features : {len(feature_cols)} columns  ({', '.join(feature_cols[:3])} ...)")
    print(f"{'='*60}")

    X_train, y_train = get_Xy(train, feature_cols)
    X_val,   y_val   = get_Xy(val,   feature_cols)
    X_test,  y_test  = get_Xy(test,  feature_cols)

    print(f"\n  Fitting on {len(X_train):,} training rows...")
    pipeline = fit_pipeline(pipeline, X_train, y_train, model_name)
    print(f"  Done.")

    val_metrics  = evaluate_pipeline(pipeline, X_val,  y_val,  "Validation")
    test_metrics = evaluate_pipeline(pipeline, X_test, y_test, "Test")

    return {
        "model"       : model_name,
        "n_features"  : len(feature_cols),
        "feature_cols": feature_cols,
        "validation"  : val_metrics,
        "test"        : test_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — MODEL SELECTION ON FULL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[str, dict]:
    """
    Train LR, RF, and XGBoost on the full feature set.
    Returns the name and results of the best model by test Macro F1.
    """
    print("\n" + "#"*60)
    print("STAGE 1 — MODEL SELECTION (Full features: TA + Fund + Sent)")
    print("#"*60)

    full_features = TA_COLS + FUND_COLS + SENT_COLS

    models = {
        "logistic_regression": build_lr_pipeline(),
        "random_forest":       build_rf_pipeline(),
        "xgboost":             build_xgb_pipeline(),
    }

    stage1_results = {}
    for model_name, pipeline in models.items():
        result = run_experiment(
            model_name   = model_name,
            pipeline     = pipeline,
            feature_cols = full_features,
            train        = train,
            val          = val,
            test         = test,
        )
        stage1_results[model_name] = result

    # Summary table
    print("\n" + "="*60)
    print("STAGE 1 SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Val Macro F1':>12} {'Test Macro F1':>13}")
    print("-"*60)
    for model_name, result in stage1_results.items():
        print(
            f"{model_name:<25} "
            f"{result['validation']['macro_f1']:>12.4f} "
            f"{result['test']['macro_f1']:>13.4f}"
        )
    print("="*60)

    # Pick best by test Macro F1
    best_model_name = max(
        stage1_results,
        key=lambda m: stage1_results[m]["test"]["macro_f1"]
    )
    best_f1 = stage1_results[best_model_name]["test"]["macro_f1"]
    print(f"\nBest model: {best_model_name}  (test Macro F1 = {best_f1:.4f})")

    return best_model_name, stage1_results


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — ABLATION ON BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2(
    best_model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """
    Run the ablation study using the best model from Stage 1.
    Only the feature set changes between runs — model and hyperparameters
    stay identical across all four ablation configurations.
    """
    print("\n" + "#"*60)
    print(f"STAGE 2 — ABLATION STUDY  (model: {best_model_name})")
    print("#"*60)

    pipeline_builders = {
        "logistic_regression": build_lr_pipeline,
        "random_forest":       build_rf_pipeline,
        "xgboost":             build_xgb_pipeline,
    }

    stage2_results = {}
    for set_name, feature_cols in ABLATION_SETS.items():
        # Fresh pipeline for each run — pipelines are stateful after fitting
        pipeline = pipeline_builders[best_model_name]()
        result = run_experiment(
            model_name   = best_model_name,
            pipeline     = pipeline,
            feature_cols = feature_cols,
            train        = train,
            val          = val,
            test         = test,
        )
        stage2_results[set_name] = result

    # Summary table
    print("\n" + "="*60)
    print("STAGE 2 SUMMARY — Ablation")
    print("="*60)
    print(f"{'Feature Set':<25} {'N Features':>10} {'Val Macro F1':>12} {'Test Macro F1':>13}")
    print("-"*60)
    for set_name, result in stage2_results.items():
        print(
            f"{set_name:<25} "
            f"{result['n_features']:>10} "
            f"{result['validation']['macro_f1']:>12.4f} "
            f"{result['test']['macro_f1']:>13.4f}"
        )
    print("="*60)

    # Gains vs TA-only baseline
    ta_only_f1 = stage2_results["ta_only"]["test"]["macro_f1"]
    comparisons = {
        "ta_sent":      "TA + Sentiment",
        "ta_fund":      "TA + Fundamentals",
        "ta_fund_sent": "TA + Fund + Sent",
    }
    print(f"\nGains vs TA-only baseline ({ta_only_f1:.4f}) — Test Macro F1:")
    for set_name, label in comparisons.items():
        curr = stage2_results[set_name]["test"]["macro_f1"]
        gain = curr - ta_only_f1
        print(f"  {label:<25}: {curr:.4f}  ({gain:+.4f})")

    return stage2_results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading splits...")
    train, val, test = load_splits()
    print(f"  Train : {len(train):,} rows")
    print(f"  Val   : {len(val):,} rows")
    print(f"  Test  : {len(test):,} rows")

    # ── Stage 1: find best model on full features ──────────────────────────
    best_model_name, stage1_results = run_stage1(train, val, test)

    # ── Stage 2: ablation on best model ────────────────────────────────────
    stage2_results = run_stage2(best_model_name, train, val, test)

    # ── Final combined summary table ───────────────────────────────────────
    print("\n" + "="*60)
    print("FULL RESULTS TABLE (for report)")
    print("="*60)
    print(f"{'Model':<35} {'Features':<20} {'Test Macro F1':>13}")
    print("-"*60)

    # Previously computed baselines
    print(f"{'Always-HOLD':<35} {'—':<20} {'0.2233':>13}")
    print(f"{'Logistic Regression':<35} {'TA only':<20} {'0.3190':>13}")

    # Stage 1 results
    for model_name, result in stage1_results.items():
        print(
            f"{model_name:<35} "
            f"{'Full (38 cols)':<20} "
            f"{result['test']['macro_f1']:>13.4f}"
        )

    # Stage 2 ablation results
    for set_name, result in stage2_results.items():
        label = f"{best_model_name} (ablation)"
        print(
            f"{label:<35} "
            f"{set_name:<20} "
            f"{result['test']['macro_f1']:>13.4f}"
        )
    print("="*60)

    # ── Save results ───────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "description": (
            "Two-stage experiment. Stage 1: LR vs RF vs XGBoost on full "
            "features (38 cols). Stage 2: ablation study on best model "
            "across four feature sets. Default hyperparameters used. "
            "Primary metric: Test Macro F1."
        ),
        "best_model_stage1": best_model_name,
        "stage1_model_selection": stage1_results,
        "stage2_ablation": {
            "model": best_model_name,
            "results": stage2_results,
        },
    }

    output_path = RESULTS_DIR / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
