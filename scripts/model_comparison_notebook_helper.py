from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT / "src" / "models") not in sys.path:
    sys.path.append(str(ROOT / "src" / "models"))

import model_training as mt


def load_experiment_data() -> dict[str, Any]:
    """Load the dataset once and prepare the shared train/val/test objects."""
    df = mt.load_data()
    X, y, dates, tickers = mt.prepare_features(df)
    (
        X_train,
        y_train,
        dates_train,
        tickers_train,
        X_val,
        y_val,
        dates_val,
        tickers_val,
        X_test,
        y_test,
        dates_test,
        tickers_test,
    ) = mt.split_data(X, y, dates, tickers)

    return {
        "df": df,
        "X": X,
        "y": y,
        "dates": dates,
        "tickers": tickers,
        "X_train": X_train,
        "y_train": y_train,
        "dates_train": dates_train,
        "tickers_train": tickers_train,
        "X_val": X_val,
        "y_val": y_val,
        "dates_val": dates_val,
        "tickers_val": tickers_val,
        "X_test": X_test,
        "y_test": y_test,
        "dates_test": dates_test,
        "tickers_test": tickers_test,
    }


def prepare_sequence_bundle(data_bundle: dict[str, Any]) -> dict[str, Any]:
    """Prepare one shared sequence bundle for all sequence models."""
    return mt.prepare_sequence_data(
        X_train=data_bundle["X_train"],
        y_train=data_bundle["y_train"],
        dates_train=data_bundle["dates_train"],
        tickers_train=data_bundle["tickers_train"],
        X_val=data_bundle["X_val"],
        y_val=data_bundle["y_val"],
        dates_val=data_bundle["dates_val"],
        tickers_val=data_bundle["tickers_val"],
        X_test=data_bundle["X_test"],
        y_test=data_bundle["y_test"],
        dates_test=data_bundle["dates_test"],
        tickers_test=data_bundle["tickers_test"],
        sequence_length=mt.DEFAULT_SEQUENCE_LENGTH,
    )


def build_data_overview(data_bundle: dict[str, Any]) -> pd.DataFrame:
    """Return a compact table describing the data splits."""
    rows = []
    for split_name in ("train", "validation", "test"):
        X_split = data_bundle[f"X_{'val' if split_name == 'validation' else split_name}"]
        y_split = data_bundle[f"y_{'val' if split_name == 'validation' else split_name}"]
        dates_split = data_bundle[f"dates_{'val' if split_name == 'validation' else split_name}"]
        rows.append(
            {
                "split": split_name,
                "rows": len(X_split),
                "features": X_split.shape[1],
                "start": pd.Timestamp(dates_split.min()).date(),
                "end": pd.Timestamp(dates_split.max()).date(),
                "label_dist": y_split.value_counts().sort_index().to_dict(),
            }
        )
    return pd.DataFrame(rows)


def apply_training_overrides(model_name: str, training_overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Temporarily override model hyperparameters for notebook runs."""
    saved_values: dict[str, Any] = {}
    override_values = training_overrides.get(model_name, {})

    if model_name == "xgboost":
        for key, value in override_values.items():
            saved_values[key] = mt.XGB_PARAMS[key]
            mt.XGB_PARAMS[key] = value
        return saved_values

    model_params = mt.MODEL_CONFIGS[model_name]["params"]
    for key, value in override_values.items():
        saved_values[key] = model_params[key]
        model_params[key] = value
    return saved_values


def restore_training_overrides(model_name: str, saved_values: dict[str, Any]) -> None:
    """Restore model hyperparameters after a notebook run."""
    if model_name == "xgboost":
        for key, value in saved_values.items():
            mt.XGB_PARAMS[key] = value
        return

    model_params = mt.MODEL_CONFIGS[model_name]["params"]
    for key, value in saved_values.items():
        model_params[key] = value


def collect_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Collect the same headline metrics used in the training script."""
    try:
        auc = roc_auc_score(y_true, y_score, labels=labels, multi_class="ovr", average="macro")
    except ValueError:
        auc = np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def evaluate_model_bundle(
    model_bundle: dict[str, Any],
    data_bundle: dict[str, Any],
    sequence_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one trained model bundle and return structured metrics."""
    family = model_bundle["family"]
    label_encoder = model_bundle["label_encoder"]
    labels = label_encoder.classes_

    if family == "xgboost":
        model = model_bundle["model"]
        results = {}
        for split_name, X_split, y_split in [
            ("validation", data_bundle["X_val"], data_bundle["y_val"]),
            ("test", data_bundle["X_test"], data_bundle["y_test"]),
        ]:
            y_score = model.predict_proba(X_split)
            y_pred = label_encoder.inverse_transform(model.predict(X_split).astype(int))
            results[split_name] = collect_metrics(y_split.to_numpy(), y_pred, y_score, labels)
        return results

    model_name = model_bundle["model_name"]
    device = model_bundle["device"]

    if model_name == "lstm_ic":
        dataloaders = mt.build_day_grouped_dataloaders(sequence_bundle)
        label_values = torch.tensor(labels, dtype=torch.float32, device=device)

        def run_eval(split_key: str) -> dict[str, Any]:
            metrics = mt.run_lstm_ic_epoch(
                model=model_bundle["model"],
                dataloader=dataloaders[split_key],
                device=device,
                label_values=label_values,
            )
            y_true = label_encoder.inverse_transform(metrics["y_true"].astype(int))
            y_pred = label_encoder.inverse_transform(metrics["y_pred"].astype(int))
            return collect_metrics(y_true, y_pred, metrics["y_score"], labels)

    elif model_name == "lstm_attention":
        dataloaders = mt.build_day_grouped_dataloaders(sequence_bundle)

        def run_eval(split_key: str) -> dict[str, Any]:
            metrics = mt.run_attention_epoch(
                model=model_bundle["model"],
                dataloader=dataloaders[split_key],
                criterion=model_bundle["criterion"],
                device=device,
            )
            y_true = label_encoder.inverse_transform(metrics["y_true"].astype(int))
            y_pred = label_encoder.inverse_transform(metrics["y_pred"].astype(int))
            return collect_metrics(y_true, y_pred, metrics["y_score"], labels)

    else:
        dataloaders = mt.build_sequence_dataloaders(
            sequence_bundle,
            batch_size=mt.MODEL_CONFIGS[model_name]["params"]["batch_size"],
        )

        def run_eval(split_key: str) -> dict[str, Any]:
            metrics = mt.run_sequence_epoch(
                model=model_bundle["model"],
                dataloader=dataloaders[split_key],
                criterion=model_bundle["criterion"],
                device=device,
            )
            y_true = label_encoder.inverse_transform(metrics["y_true"].astype(int))
            y_pred = label_encoder.inverse_transform(metrics["y_pred"].astype(int))
            return collect_metrics(y_true, y_pred, metrics["y_score"], labels)

    return {
        "validation": run_eval("validation"),
        "test": run_eval("test"),
    }


def run_model_comparison(
    models_to_run: list[str],
    training_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Train and evaluate a list of models, returning summary and artifacts."""
    training_overrides = training_overrides or {}
    data_bundle = load_experiment_data()
    sequence_bundle = None
    if any(model_name != "xgboost" for model_name in models_to_run):
        sequence_bundle = prepare_sequence_bundle(data_bundle)

    summary_records: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    for model_name in models_to_run:
        saved_values = apply_training_overrides(model_name, training_overrides)
        try:
            model_bundle = mt.train_model(
                data_bundle["X_train"],
                data_bundle["y_train"],
                model_name=model_name,
                sequence_bundle=sequence_bundle,
            )
            evaluation = evaluate_model_bundle(
                model_bundle=model_bundle,
                data_bundle=data_bundle,
                sequence_bundle=sequence_bundle,
            )
        finally:
            restore_training_overrides(model_name, saved_values)

        artifacts[model_name] = {
            "model_bundle": model_bundle,
            "evaluation": evaluation,
        }
        if "history" in model_bundle:
            artifacts[model_name]["history"] = pd.DataFrame(model_bundle["history"])

        summary_records.append(
            {
                "model": model_name,
                "val_accuracy": evaluation["validation"]["accuracy"],
                "val_macro_f1": evaluation["validation"]["macro_f1"],
                "val_weighted_f1": evaluation["validation"]["weighted_f1"],
                "val_auc": evaluation["validation"]["auc"],
                "test_accuracy": evaluation["test"]["accuracy"],
                "test_macro_f1": evaluation["test"]["macro_f1"],
                "test_weighted_f1": evaluation["test"]["weighted_f1"],
                "test_auc": evaluation["test"]["auc"],
            }
        )

    summary_df = pd.DataFrame(summary_records).reset_index(drop=True)
    return summary_df, artifacts, data_bundle, sequence_bundle


def plot_metric_bars(summary_df: pd.DataFrame) -> None:
    """Plot side-by-side bars for Macro F1 and AUC."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    summary_df.plot(
        x="model",
        y=["val_macro_f1", "test_macro_f1"],
        kind="bar",
        ax=axes[0],
        title="Macro F1 Comparison",
    )
    axes[0].set_ylabel("Macro F1")
    axes[0].set_ylim(bottom=0.20)
    axes[0].tick_params(axis="x", rotation=30)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    summary_df.plot(
        x="model",
        y=["val_auc", "test_auc"],
        kind="bar",
        ax=axes[1],
        title="AUC Comparison",
    )
    axes[1].set_ylabel("ROC AUC (macro OVR)")
    axes[1].set_ylim(bottom=0.40)
    axes[1].tick_params(axis="x", rotation=30)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(artifacts: dict[str, Any], split_name: str = "test") -> None:
    """Plot confusion matrices for all evaluated models."""
    model_names = list(artifacts.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(4.8 * len(model_names), 4))
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        cm = artifacts[model_name]["evaluation"][split_name]["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{model_name} | {split_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        ax.set_xticklabels([-1, 0, 1])
        ax.set_yticklabels([-1, 0, 1])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()


def plot_training_curves(artifacts: dict[str, Any]) -> None:
    """Plot training curves for sequence models that expose epoch history."""
    sequence_model_names = [name for name in artifacts if "history" in artifacts[name]]
    if not sequence_model_names:
        print("No sequence-model training history available.")
        return

    fig, axes = plt.subplots(len(sequence_model_names), 2, figsize=(12, 4 * len(sequence_model_names)))
    if len(sequence_model_names) == 1:
        axes = np.array([axes])

    for row_idx, model_name in enumerate(sequence_model_names):
        history_df = artifacts[model_name]["history"]

        axes[row_idx, 0].plot(history_df["epoch"], history_df["train_macro_f1"], label="train")
        axes[row_idx, 0].plot(history_df["epoch"], history_df["val_macro_f1"], label="validation")
        axes[row_idx, 0].set_title(f"{model_name} | Macro F1")
        axes[row_idx, 0].set_xlabel("Epoch")
        axes[row_idx, 0].set_ylabel("Macro F1")
        axes[row_idx, 0].legend()

        axes[row_idx, 1].plot(history_df["epoch"], history_df["train_loss"], label="train")
        axes[row_idx, 1].plot(history_df["epoch"], history_df["val_loss"], label="validation")
        axes[row_idx, 1].set_title(f"{model_name} | Loss")
        axes[row_idx, 1].set_xlabel("Epoch")
        axes[row_idx, 1].set_ylabel("Loss")
        axes[row_idx, 1].legend()

    plt.tight_layout()
    plt.show()
