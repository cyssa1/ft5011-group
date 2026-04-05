from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from xgboost import XGBClassifier


DATA_PATH = Path("data/labeled_dataset.csv")
DATE_COLUMN = "date"
TARGET_COLUMN = "label"
NON_FEATURE_COLUMNS = [DATE_COLUMN, "ticker_id", TARGET_COLUMN]
LOAD_DROP_COLUMNS = ["fwd_return_5d"]

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "family": "sklearn",
        "estimator": LogisticRegression,
        "params": {
            "multi_class": "multinomial",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        },
    },
    "random_forest": {
        "family": "sklearn",
        "estimator": RandomForestClassifier,
        "params": {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
    },
    "xgboost": {
        "family": "xgboost",
        "estimator": XGBClassifier,
        "params": {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "random_state": 42,
        },
    },
    "lstm": {
        "family": "pytorch",
        "params": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "ticker_embedding_dim": 8,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 100,
            "weight_decay": 1e-4,
            "neutral_class_weight": 0.8,
            "signal_class_boost": 1.35,
            "early_stopping_patience": 10,
            "early_stopping_min_delta": 0.001,
        },
    },
    "tcn": {
        "family": "deep_learning_placeholder",
        "params": {
            "num_channels": [32, 64],
            "kernel_size": 3,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 20,
        },
    },
}

DEFAULT_MODEL_NAME = "logistic_regression"
DEFAULT_SEQUENCE_LENGTH = 50


def select_model_name(model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Return the model name to use in the current run."""
    return model_name


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build a shared preprocessing pipeline for tabular models."""
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in X_train.columns if col not in numeric_columns]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers)


def build_model(model_name: str, X_train: pd.DataFrame) -> Pipeline:
    """Create a model pipeline from a model name and predefined config."""
    config = MODEL_CONFIGS[model_name]
    family = config["family"]

    if family in {"pytorch", "deep_learning_placeholder"}:
        raise NotImplementedError(
            f"Model '{model_name}' is handled by a separate deep learning pipeline."
        )

    estimator_cls = config["estimator"]
    preprocessor = build_preprocessor(X_train)
    estimator = estimator_cls(**config["params"])

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def load_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=[DATE_COLUMN])
    columns_to_drop = [col for col in LOAD_DROP_COLUMNS if col in df.columns]
    if columns_to_drop:
        print(f"Dropping columns during load: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    df = df.sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    return df


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Separate features, labels, and dates.

    Missing values are intentionally handled inside the training pipeline so that
    imputation statistics are learned only from the training period.
    """
    dates = df[DATE_COLUMN].copy()
    tickers = df["ticker"].copy()
    y = df[TARGET_COLUMN].copy()

    columns_to_drop = [col for col in NON_FEATURE_COLUMNS if col in df.columns]
    X = df.drop(columns=columns_to_drop).copy()

    bool_columns = X.select_dtypes(include=["bool"]).columns
    if len(bool_columns) > 0:
        X[bool_columns] = X[bool_columns].astype(int)

    return X, y, dates, tickers


def split_data(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series, tickers: pd.Series
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Split data by unique dates so one trading day stays in only one split."""
    sorted_index = dates.sort_values().index
    X = X.loc[sorted_index].reset_index(drop=True)
    y = y.loc[sorted_index].reset_index(drop=True)
    dates = dates.loc[sorted_index].reset_index(drop=True)
    tickers = tickers.loc[sorted_index].reset_index(drop=True)

    unique_dates = pd.Series(dates.unique()).sort_values().reset_index(drop=True)
    n_dates = len(unique_dates)
    train_end = int(n_dates * 0.70)
    val_end = int(n_dates * 0.85)

    train_dates = set(unique_dates.iloc[:train_end])
    val_dates = set(unique_dates.iloc[train_end:val_end])
    test_dates = set(unique_dates.iloc[val_end:])

    train_mask = dates.isin(train_dates)
    val_mask = dates.isin(val_dates)
    test_mask = dates.isin(test_dates)

    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    tickers_train = tickers.loc[train_mask].copy()

    X_val = X.loc[val_mask].copy()
    y_val = y.loc[val_mask].copy()
    tickers_val = tickers.loc[val_mask].copy()

    X_test = X.loc[test_mask].copy()
    y_test = y.loc[test_mask].copy()
    tickers_test = tickers.loc[test_mask].copy()

    return (
        X_train,
        y_train,
        tickers_train,
        X_val,
        y_val,
        tickers_val,
        X_test,
        y_test,
        tickers_test,
    )



def prepare_sequence_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tickers_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tickers_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tickers_test: pd.Series,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> dict[str, Any]:
    """Prepare sequence-ready train, validation, and test datasets."""
    sequence_preprocessor = fit_sequence_preprocessor(X_train)

    X_train_scaled = transform_sequence_features(X_train, sequence_preprocessor)
    X_val_scaled = transform_sequence_features(X_val, sequence_preprocessor)
    X_test_scaled = transform_sequence_features(X_test, sequence_preprocessor)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    ticker_encoder = LabelEncoder()
    train_ticker_ids = ticker_encoder.fit_transform(tickers_train.astype(str))
    val_ticker_ids = ticker_encoder.transform(tickers_val.astype(str))
    test_ticker_ids = ticker_encoder.transform(tickers_test.astype(str))

    train_sequences = build_grouped_sequence_windows(
        X_values=X_train_scaled,
        y_values=y_train_encoded,
        tickers=tickers_train,
        ticker_ids=train_ticker_ids,
        sequence_length=sequence_length,
    )

    train_history = build_ticker_histories(
        X_values=X_train_scaled,
        y_values=y_train_encoded,
        tickers=tickers_train,
        sequence_length=sequence_length,
    )
    val_sequences = build_grouped_sequence_windows(
        X_values=X_val_scaled,
        y_values=y_val_encoded,
        tickers=tickers_val,
        ticker_ids=val_ticker_ids,
        sequence_length=sequence_length,
        history_by_ticker=train_history,
    )

    prior_X = np.concatenate([X_train_scaled, X_val_scaled], axis=0)
    prior_y = np.concatenate([y_train_encoded, y_val_encoded], axis=0)
    prior_tickers = pd.concat([tickers_train, tickers_val], ignore_index=True)
    prior_history = build_ticker_histories(
        X_values=prior_X,
        y_values=prior_y,
        tickers=prior_tickers,
        sequence_length=sequence_length,
    )
    test_sequences = build_grouped_sequence_windows(
        X_values=X_test_scaled,
        y_values=y_test_encoded,
        tickers=tickers_test,
        ticker_ids=test_ticker_ids,
        sequence_length=sequence_length,
        history_by_ticker=prior_history,
    )

    return {
        "train": train_sequences,
        "validation": val_sequences,
        "test": test_sequences,
        "feature_names": sequence_preprocessor["feature_names"],
        "num_features": len(sequence_preprocessor["feature_names"]),
        "sequence_length": sequence_length,
        "label_encoder": label_encoder,
        "ticker_encoder": ticker_encoder,
        "num_tickers": len(ticker_encoder.classes_),
        "imputer": sequence_preprocessor["imputer"],
        "scaler": sequence_preprocessor["scaler"],
    }


def fit_sequence_preprocessor(X_train: pd.DataFrame) -> dict[str, Any]:
    """Fit numeric-only preprocessing for sequence models on the training split."""
    feature_names = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X_train[feature_names].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_numeric)

    scaler = StandardScaler()
    scaler.fit(X_imputed)

    return {
        "feature_names": feature_names,
        "imputer": imputer,
        "scaler": scaler,
    }


def transform_sequence_features(
    X: pd.DataFrame, sequence_preprocessor: dict[str, Any]
) -> np.ndarray:
    """Apply training-fitted numeric preprocessing to one split."""
    feature_names = sequence_preprocessor["feature_names"]
    imputer = sequence_preprocessor["imputer"]
    scaler = sequence_preprocessor["scaler"]

    X_numeric = X[feature_names].replace([np.inf, -np.inf], np.nan)
    X_imputed = imputer.transform(X_numeric)
    X_scaled = scaler.transform(X_imputed)
    return X_scaled.astype(np.float32)


def build_sequence_windows(
    X_values: np.ndarray,
    y_values: np.ndarray,
    sequence_length: int,
    target_start_index: int = 0,
) -> dict[str, np.ndarray]:
    """Convert row-wise tabular data into rolling windows for sequence models."""
    start_target = max(sequence_length, target_start_index)

    sequences = []
    targets = []

    for target_idx in range(start_target, len(X_values)):
        start_idx = target_idx - sequence_length
        sequences.append(X_values[start_idx:target_idx])
        targets.append(y_values[target_idx])

    if sequences:
        X_sequence = np.asarray(sequences, dtype=np.float32)
        y_sequence = np.asarray(targets, dtype=np.int64)
    else:
        num_features = X_values.shape[1]
        X_sequence = np.empty((0, sequence_length, num_features), dtype=np.float32)
        y_sequence = np.empty((0,), dtype=np.int64)

    return {"X": X_sequence, "y": y_sequence}


def build_ticker_histories(
    X_values: np.ndarray,
    y_values: np.ndarray,
    tickers: pd.Series,
    sequence_length: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Keep the most recent rows for each ticker as sequence history."""
    ticker_history = {}
    ticker_series = tickers.reset_index(drop=True)

    for ticker in ticker_series.unique():
        ticker_mask = (ticker_series == ticker).to_numpy()
        ticker_history[ticker] = {
            "X": X_values[ticker_mask][-sequence_length:],
            "y": y_values[ticker_mask][-sequence_length:],
        }

    return ticker_history


def build_grouped_sequence_windows(
    X_values: np.ndarray,
    y_values: np.ndarray,
    tickers: pd.Series,
    ticker_ids: np.ndarray,
    sequence_length: int,
    history_by_ticker: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """Build sequence windows independently for each ticker."""
    grouped_sequences = []
    grouped_targets = []
    grouped_ticker_ids = []
    ticker_series = tickers.reset_index(drop=True)
    history_by_ticker = history_by_ticker or {}

    for ticker in ticker_series.unique():
        ticker_mask = (ticker_series == ticker).to_numpy()
        current_X = X_values[ticker_mask]
        current_y = y_values[ticker_mask]
        current_ticker_ids = ticker_ids[ticker_mask]

        history = history_by_ticker.get(ticker)
        if history is None:
            history_X = np.empty((0, X_values.shape[1]), dtype=np.float32)
            history_y = np.empty((0,), dtype=np.int64)
        else:
            history_X = history["X"]
            history_y = history["y"]

        combined_X = np.concatenate([history_X, current_X], axis=0)
        combined_y = np.concatenate([history_y, current_y], axis=0)
        ticker_windows = build_sequence_windows(
            X_values=combined_X,
            y_values=combined_y,
            sequence_length=sequence_length,
            target_start_index=len(history_X),
        )

        if len(ticker_windows["X"]) > 0:
            grouped_sequences.append(ticker_windows["X"])
            grouped_targets.append(ticker_windows["y"])
            grouped_ticker_ids.append(
                np.full(len(ticker_windows["y"]), current_ticker_ids[0], dtype=np.int64)
            )

    if grouped_sequences:
        X_sequence = np.concatenate(grouped_sequences, axis=0)
        y_sequence = np.concatenate(grouped_targets, axis=0)
        ticker_sequence_ids = np.concatenate(grouped_ticker_ids, axis=0)
    else:
        num_features = X_values.shape[1]
        X_sequence = np.empty((0, sequence_length, num_features), dtype=np.float32)
        y_sequence = np.empty((0,), dtype=np.int64)
        ticker_sequence_ids = np.empty((0,), dtype=np.int64)

    return {"X": X_sequence, "y": y_sequence, "ticker_ids": ticker_sequence_ids}


def describe_sequence_data(sequence_bundle: dict[str, Any]) -> None:
    """Print a compact summary of prepared neural-network sequence data."""
    print("Prepared neural-network sequence data:")
    print(f"- train X shape: {sequence_bundle['train']['X'].shape}")
    print(f"- train y shape: {sequence_bundle['train']['y'].shape}")
    print(f"- validation X shape: {sequence_bundle['validation']['X'].shape}")
    print(f"- validation y shape: {sequence_bundle['validation']['y'].shape}")
    print(f"- test X shape: {sequence_bundle['test']['X'].shape}")
    print(f"- test y shape: {sequence_bundle['test']['y'].shape}")
    print(f"- sequence length: {sequence_bundle['sequence_length']}")
    print(f"- number of features: {sequence_bundle['num_features']}")
    print(f"- number of tickers: {sequence_bundle['num_tickers']}")
    print(f"- encoded classes: {list(sequence_bundle['label_encoder'].classes_)}")


class SequenceDataset(Dataset):
    """Simple dataset wrapper for prebuilt sequence arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray, ticker_ids: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.ticker_ids = torch.tensor(ticker_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index], self.ticker_ids[index]


class LSTMClassifier(nn.Module):
    """Basic LSTM classifier for multi-class sequence prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        num_tickers: int,
        ticker_embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.ticker_embedding = nn.Embedding(num_tickers, ticker_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + ticker_embedding_dim, num_classes)

    def forward(self, X: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.lstm(X)
        last_hidden = hidden_state[-1]
        ticker_embedding = self.ticker_embedding(ticker_ids)
        combined_hidden = torch.cat([last_hidden, ticker_embedding], dim=1)
        combined_hidden = self.dropout(combined_hidden)
        return self.classifier(combined_hidden)


def get_torch_device() -> torch.device:
    """Choose the best available device for PyTorch training."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_sequence_dataloaders(
    sequence_bundle: dict[str, Any], batch_size: int
) -> dict[str, DataLoader]:
    """Create DataLoaders for train, validation, and test sequence splits."""
    train_dataset = SequenceDataset(
        sequence_bundle["train"]["X"],
        sequence_bundle["train"]["y"],
        sequence_bundle["train"]["ticker_ids"],
    )
    val_dataset = SequenceDataset(
        sequence_bundle["validation"]["X"],
        sequence_bundle["validation"]["y"],
        sequence_bundle["validation"]["ticker_ids"],
    )
    test_dataset = SequenceDataset(
        sequence_bundle["test"]["X"],
        sequence_bundle["test"]["y"],
        sequence_bundle["test"]["ticker_ids"],
    )

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
        "validation": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def run_lstm_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Run one training or evaluation epoch."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    for X_batch, y_batch, ticker_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        ticker_batch = ticker_batch.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(X_batch, ticker_batch)
            loss = criterion(logits, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)
        all_targets.append(y_batch.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())
        all_probabilities.append(probabilities.detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    y_score = np.concatenate(all_probabilities)

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def build_lstm_class_weights(
    sequence_bundle: dict[str, Any], model_name: str, device: torch.device
) -> torch.Tensor:
    """Build class weights that downweight 0 and upweight -1 / 1."""
    config = MODEL_CONFIGS[model_name]["params"]
    y_train = sequence_bundle["train"]["y"]
    num_classes = len(sequence_bundle["label_encoder"].classes_)
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = len(y_train) / (num_classes * class_counts)

    class_labels = sequence_bundle["label_encoder"].classes_
    for class_index, class_label in enumerate(class_labels):
        if class_label == 0:
            class_weights[class_index] *= config["neutral_class_weight"]
        else:
            class_weights[class_index] *= config["signal_class_boost"]

    class_weights = class_weights / class_weights.mean()
    print("LSTM class weights:")
    for class_label, weight in zip(class_labels, class_weights):
        print(f"- class {class_label}: {weight:.4f}")

    return torch.tensor(class_weights, dtype=torch.float32, device=device)


def train_lstm_model(sequence_bundle: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Train an LSTM model on the prepared sequence datasets."""
    if sequence_bundle is None:
        raise ValueError("sequence_bundle is required for LSTM training.")

    config = MODEL_CONFIGS[model_name]["params"]
    device = get_torch_device()
    dataloaders = build_sequence_dataloaders(
        sequence_bundle=sequence_bundle, batch_size=config["batch_size"]
    )

    model = LSTMClassifier(
        input_size=sequence_bundle["num_features"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=len(sequence_bundle["label_encoder"].classes_),
        num_tickers=sequence_bundle["num_tickers"],
        ticker_embedding_dim=config["ticker_embedding_dim"],
        dropout=config["dropout"],
    ).to(device)

    class_weights = build_lstm_class_weights(
        sequence_bundle=sequence_bundle,
        model_name=model_name,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_f1 = -np.inf
    epochs_without_improvement = 0
    history = []

    print(f"Training on device: {device}")

    epoch_progress = tqdm(
        range(1, config["epochs"] + 1),
        desc=f"Training {model_name}",
        unit="epoch",
    )

    for epoch in epoch_progress:
        train_metrics = run_lstm_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_lstm_epoch(
            model=model,
            dataloader=dataloaders["validation"],
            criterion=criterion,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        epoch_progress.set_postfix(
            train_loss=f"{train_metrics['loss']:.4f}",
            train_f1=f"{train_metrics['macro_f1']:.4f}",
            val_loss=f"{val_metrics['loss']:.4f}",
            val_f1=f"{val_metrics['macro_f1']:.4f}",
        )

        if val_metrics["macro_f1"] >= best_val_f1:
            if val_metrics["macro_f1"] > best_val_f1 + config["early_stopping_min_delta"]:
                best_val_f1 = val_metrics["macro_f1"]
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["early_stopping_patience"]:
            print(
                f"Early stopping triggered after epoch {epoch}. "
                f"Best validation macro F1: {best_val_f1:.4f}"
            )
            break

    model.load_state_dict(best_state_dict)

    return {
        "family": "pytorch",
        "model_name": model_name,
        "model": model,
        "device": device,
        "criterion": criterion,
        "history": history,
        "label_encoder": sequence_bundle["label_encoder"],
    }


def decode_sequence_labels(
    encoded_labels: np.ndarray, label_encoder: LabelEncoder
) -> np.ndarray:
    """Map encoded class indices back to the original labels."""
    return label_encoder.inverse_transform(encoded_labels.astype(int))


def print_classification_metrics(
    split_name: str,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    labels: np.ndarray | list[Any] | None = None,
) -> None:
    """Print classification metrics with consistent formatting."""
    if labels is None:
        labels = sorted(pd.Index(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])).unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    print(f"\n{split_name} Results")
    print("-" * (len(split_name) + 8))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    if y_score is not None:
        auc = roc_auc_score(
            y_true,
            y_score,
            labels=labels,
            multi_class="ovr",
            average="macro",
        )
        print(f"ROC AUC (macro, OVR): {auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm_df)


def print_logistic_regression_weights(model_bundle: dict[str, Any], top_n: int = 10) -> None:
    """Print the largest absolute logistic-regression coefficients for each class."""
    if model_bundle["model_name"] != "logistic_regression":
        return

    model = model_bundle["model"]
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    print("\nTop Logistic Regression Weights")
    print("------------------------------")

    for class_index, class_label in enumerate(classifier.classes_):
        class_weights = pd.Series(classifier.coef_[class_index], index=feature_names)
        class_weights = class_weights.sort_values(key=lambda values: values.abs(), ascending=False)
        print(f"\nClass {class_label}:")
        print(class_weights.head(top_n).to_string())


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = DEFAULT_MODEL_NAME,
    sequence_bundle: dict[str, Any] | None = None,
) -> Any:
    """Train a selected model using the appropriate family-specific pipeline."""
    family = MODEL_CONFIGS[model_name]["family"]

    if family == "sklearn":
        model = build_model(model_name=model_name, X_train=X_train)
        model.fit(X_train, y_train)
        return {"family": family, "model_name": model_name, "model": model}

    if family == "xgboost":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        model = build_model(model_name=model_name, X_train=X_train)
        model.fit(X_train, y_train_encoded)
        return {
            "family": family,
            "model_name": model_name,
            "model": model,
            "label_encoder": label_encoder,
        }

    if family == "pytorch":
        return train_lstm_model(sequence_bundle=sequence_bundle, model_name=model_name)

    raise NotImplementedError(f"Model family '{family}' is not implemented yet.")


def evaluate_model(
    model_bundle: dict[str, Any],
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    sequence_bundle: dict[str, Any] | None = None,
) -> None:
    family = model_bundle["family"]

    if family == "sklearn":
        model = model_bundle["model"]
        val_predictions = model.predict(X_val)
        test_predictions = model.predict(X_test)
        val_probabilities = model.predict_proba(X_val)
        test_probabilities = model.predict_proba(X_test)
        labels = model.named_steps["classifier"].classes_
        print_classification_metrics(
            "Validation",
            y_val,
            val_predictions,
            y_score=val_probabilities,
            labels=labels,
        )
        print_classification_metrics(
            "Test",
            y_test,
            test_predictions,
            y_score=test_probabilities,
            labels=labels,
        )
        return

    if family == "xgboost":
        model = model_bundle["model"]
        label_encoder = model_bundle["label_encoder"]
        val_predictions_encoded = model.predict(X_val)
        test_predictions_encoded = model.predict(X_test)
        val_probabilities = model.predict_proba(X_val)
        test_probabilities = model.predict_proba(X_test)

        val_predictions = label_encoder.inverse_transform(val_predictions_encoded.astype(int))
        test_predictions = label_encoder.inverse_transform(test_predictions_encoded.astype(int))
        labels = label_encoder.classes_

        print_classification_metrics(
            "Validation",
            y_val,
            val_predictions,
            y_score=val_probabilities,
            labels=labels,
        )
        print_classification_metrics(
            "Test",
            y_test,
            test_predictions,
            y_score=test_probabilities,
            labels=labels,
        )
        return

    if family == "pytorch":
        model = model_bundle["model"]
        criterion = model_bundle["criterion"]
        device = model_bundle["device"]
        label_encoder = model_bundle["label_encoder"]
        dataloaders = build_sequence_dataloaders(
            sequence_bundle=sequence_bundle,
            batch_size=MODEL_CONFIGS[model_bundle["model_name"]]["params"]["batch_size"],
        )

        val_metrics = run_lstm_epoch(
            model=model,
            dataloader=dataloaders["validation"],
            criterion=criterion,
            device=device,
        )
        test_metrics = run_lstm_epoch(
            model=model,
            dataloader=dataloaders["test"],
            criterion=criterion,
            device=device,
        )

        val_true = decode_sequence_labels(val_metrics["y_true"], label_encoder)
        val_pred = decode_sequence_labels(val_metrics["y_pred"], label_encoder)
        test_true = decode_sequence_labels(test_metrics["y_true"], label_encoder)
        test_pred = decode_sequence_labels(test_metrics["y_pred"], label_encoder)
        labels = label_encoder.classes_

        print_classification_metrics(
            "Validation",
            val_true,
            val_pred,
            y_score=val_metrics["y_score"],
            labels=labels,
        )
        print_classification_metrics(
            "Test",
            test_true,
            test_pred,
            y_score=test_metrics["y_score"],
            labels=labels,
        )
        return

    raise NotImplementedError(f"Evaluation for model family '{family}' is not implemented.")


def main() -> None:
    selected_model = select_model_name(input())  # Change this to select a different model

    print("Loading dataset...")
    df = load_data()
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

    print("Preparing features...")
    X, y, dates, tickers = prepare_features(df)
    print(f"Features prepared: {X.shape[0]} rows, {X.shape[1]} feature columns")

    print("Splitting data by time...")
    (
        X_train,
        y_train,
        tickers_train,
        X_val,
        y_val,
        tickers_val,
        X_test,
        y_test,
        tickers_test,
    ) = split_data(X, y, dates, tickers)
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    print("Preparing sequence datasets...")
    sequence_bundle = prepare_sequence_data(
        X_train=X_train,
        y_train=y_train,
        tickers_train=tickers_train,
        X_val=X_val,
        y_val=y_val,
        tickers_val=tickers_val,
        X_test=X_test,
        y_test=y_test,
        tickers_test=tickers_test,
        sequence_length=DEFAULT_SEQUENCE_LENGTH,
    )
    describe_sequence_data(sequence_bundle)

    print("Available models:")
    for model_name in MODEL_CONFIGS:
        print(f"- {model_name}")
    print(f"Default sequence length for neural-network preparation: {DEFAULT_SEQUENCE_LENGTH}")

    print(f"Training model: {selected_model}")
    model = train_model(
        X_train,
        y_train,
        model_name=selected_model,
        sequence_bundle=sequence_bundle,
    )

    print_logistic_regression_weights(model)

    print("Evaluating model...")
    evaluate_model(
        model,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        sequence_bundle=sequence_bundle,
    )


if __name__ == "__main__":
    main()
