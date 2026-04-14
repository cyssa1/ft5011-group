from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from xgboost import XGBClassifier


DATA_PATH = Path("data/final_dataset.csv")
DATE_COLUMN = "date"
TICKER_COLUMN = "ticker"
TARGET_COLUMN = "label_multi"
ALL_LABEL_COLUMNS = ["label_updown_1d", "label_multi", "label_excess_5d"]
NON_FEATURE_COLUMNS = [DATE_COLUMN, TICKER_COLUMN, *ALL_LABEL_COLUMNS]
LOAD_DROP_COLUMNS = ["fwd_ret_5d", "fwd_return_5d", "signal"]

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "xgboost": {"family": "xgboost"},
    "cnn": {
        "family": "pytorch",
        "params": {
            "conv_channels": [32, 64],
            "kernel_size": 3,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 30,
            "weight_decay": 1e-4,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 0.001,
        },
    },
    "lstm": {
        "family": "pytorch",
        "params": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 30,
            "weight_decay": 1e-4,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 0.001,
        },
    },
}

DEFAULT_MODEL_NAME = "cnn"
DEFAULT_SEQUENCE_LENGTH = 20


def load_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset, normalize columns, and drop leakage columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    columns_to_drop = [c for c in LOAD_DROP_COLUMNS if c in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df.sort_values([DATE_COLUMN, TICKER_COLUMN], ascending=True).reset_index(drop=True)


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into numeric technical features, multiclass target, dates, tickers."""
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    dates = df[DATE_COLUMN].copy()
    tickers = df[TICKER_COLUMN].copy()
    y = df[TARGET_COLUMN].copy()

    columns_to_drop = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    X = df.drop(columns=columns_to_drop).select_dtypes(include=[np.number]).copy()

    return X, y, dates, tickers


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    tickers: pd.Series,
) -> tuple[
    pd.DataFrame, pd.Series, pd.Series,
    pd.DataFrame, pd.Series, pd.Series,
    pd.DataFrame, pd.Series, pd.Series,
]:
    """Split by unique dates so one trading day only appears in one split."""
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

    return (
        X.loc[train_mask].copy(), y.loc[train_mask].copy(), tickers.loc[train_mask].copy(),
        X.loc[val_mask].copy(), y.loc[val_mask].copy(), tickers.loc[val_mask].copy(),
        X.loc[test_mask].copy(), y.loc[test_mask].copy(), tickers.loc[test_mask].copy(),
    )


def build_tabular_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """XGBoost pipeline with median-impute + standard-scale on numeric features."""
    numeric_columns = X_train.columns.tolist()
    return Pipeline(steps=[
        ("preprocessor", ColumnTransformer(transformers=[(
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_columns,
        )])),
        ("classifier", XGBClassifier(**XGB_PARAMS)),
    ])


def build_sequence_windows(
    X_values: np.ndarray,
    y_values: np.ndarray,
    sequence_length: int,
    target_start_index: int = 0,
) -> dict[str, np.ndarray]:
    """Convert row-wise data into rolling windows; drop windows with any NaN."""
    start_target = max(sequence_length, target_start_index)
    sequences: list[np.ndarray] = []
    targets: list[int] = []

    for target_index in range(start_target, len(X_values)):
        window = X_values[target_index - sequence_length:target_index]
        if np.isnan(window).any():
            continue
        sequences.append(window)
        targets.append(y_values[target_index])

    num_features = X_values.shape[1]
    if sequences:
        return {
            "X": np.asarray(sequences, dtype=np.float32),
            "y": np.asarray(targets, dtype=np.int64),
        }
    return {
        "X": np.empty((0, sequence_length, num_features), dtype=np.float32),
        "y": np.empty((0,), dtype=np.int64),
    }


def build_grouped_sequence_windows(
    X_values: np.ndarray,
    y_values: np.ndarray,
    tickers: pd.Series,
    sequence_length: int,
    history_by_ticker: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """Build sequence windows independently for each ticker."""
    grouped_sequences: list[np.ndarray] = []
    grouped_targets: list[np.ndarray] = []
    ticker_series = tickers.reset_index(drop=True)
    history_by_ticker = history_by_ticker or {}

    for ticker in ticker_series.unique():
        ticker_mask = (ticker_series == ticker).to_numpy()
        current_X = X_values[ticker_mask]
        current_y = y_values[ticker_mask]

        history = history_by_ticker.get(ticker)
        if history is None:
            history_X = np.empty((0, X_values.shape[1]), dtype=np.float32)
            history_y = np.empty((0,), dtype=np.int64)
        else:
            history_X = history["X"]
            history_y = history["y"]

        ticker_windows = build_sequence_windows(
            X_values=np.concatenate([history_X, current_X], axis=0),
            y_values=np.concatenate([history_y, current_y], axis=0),
            sequence_length=sequence_length,
            target_start_index=len(history_X),
        )

        if len(ticker_windows["X"]) > 0:
            grouped_sequences.append(ticker_windows["X"])
            grouped_targets.append(ticker_windows["y"])

    num_features = X_values.shape[1]
    if grouped_sequences:
        return {
            "X": np.concatenate(grouped_sequences, axis=0),
            "y": np.concatenate(grouped_targets, axis=0),
        }
    return {
        "X": np.empty((0, sequence_length, num_features), dtype=np.float32),
        "y": np.empty((0,), dtype=np.int64),
    }


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
    """Fit scaler on train, scale all splits, encode labels, build per-ticker windows."""
    feature_names = X_train.columns.tolist()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    def _scale(X: pd.DataFrame, fit: bool) -> np.ndarray:
        X_num = X[feature_names].replace([np.inf, -np.inf], np.nan)
        X_imp = imputer.fit_transform(X_num) if fit else imputer.transform(X_num)
        X_scl = scaler.fit_transform(X_imp) if fit else scaler.transform(X_imp)
        return X_scl.astype(np.float32)

    X_train_scaled = _scale(X_train, fit=True)
    X_val_scaled = _scale(X_val, fit=False)
    X_test_scaled = _scale(X_test, fit=False)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train).astype(np.int64)
    y_val_enc = label_encoder.transform(y_val).astype(np.int64)
    y_test_enc = label_encoder.transform(y_test).astype(np.int64)

    def _tail_history(
        X_arr: np.ndarray, y_arr: np.ndarray, ticker_series: pd.Series,
    ) -> dict[str, dict[str, np.ndarray]]:
        series = ticker_series.reset_index(drop=True)
        return {
            ticker: {
                "X": X_arr[(series == ticker).to_numpy()][-sequence_length:],
                "y": y_arr[(series == ticker).to_numpy()][-sequence_length:],
            }
            for ticker in series.unique()
        }

    train_sequences = build_grouped_sequence_windows(
        X_values=X_train_scaled, y_values=y_train_enc,
        tickers=tickers_train, sequence_length=sequence_length,
    )
    val_sequences = build_grouped_sequence_windows(
        X_values=X_val_scaled, y_values=y_val_enc,
        tickers=tickers_val, sequence_length=sequence_length,
        history_by_ticker=_tail_history(X_train_scaled, y_train_enc, tickers_train),
    )
    test_sequences = build_grouped_sequence_windows(
        X_values=X_test_scaled, y_values=y_test_enc,
        tickers=tickers_test, sequence_length=sequence_length,
        history_by_ticker=_tail_history(
            np.concatenate([X_train_scaled, X_val_scaled], axis=0),
            np.concatenate([y_train_enc, y_val_enc], axis=0),
            pd.concat([tickers_train, tickers_val], ignore_index=True),
        ),
    )

    return {
        "train": train_sequences,
        "validation": val_sequences,
        "test": test_sequences,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "sequence_length": sequence_length,
        "label_encoder": label_encoder,
        "imputer": imputer,
        "scaler": scaler,
    }


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


class CNNClassifier(nn.Module):
    """1D CNN over technical-factor sequences."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        conv_channels: list[int],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(input_size, conv_channels[0], kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels[0]),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels[1]),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(conv_channels[1], num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.transpose(1, 2)
        X = self.features(X)
        X = X.squeeze(-1)
        X = self.dropout(X)
        return self.classifier(X)


class LSTMClassifier(nn.Module):
    """LSTM over technical-factor sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
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
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.lstm(X)
        last_hidden = hidden_state[-1]
        last_hidden = self.dropout(last_hidden)
        return self.classifier(last_hidden)


def build_sequence_dataloaders(
    sequence_bundle: dict[str, Any],
    batch_size: int,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            SequenceDataset(sequence_bundle[split]["X"], sequence_bundle[split]["y"]),
            batch_size=batch_size,
            shuffle=False,
        )
        for split in ("train", "validation", "test")
    }


def run_sequence_epoch(
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
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(y_batch.detach().cpu().numpy())
        all_predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        all_probabilities.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    y_score = np.concatenate(all_probabilities)

    try:
        auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = DEFAULT_MODEL_NAME,
    sequence_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the selected model (xgboost / cnn / lstm) for multiclass label_multi."""
    family = MODEL_CONFIGS[model_name]["family"]

    if family == "xgboost":
        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)
        model = build_tabular_pipeline(X_train)
        model.fit(X_train, y_train_enc)
        return {
            "family": family,
            "model_name": model_name,
            "model": model,
            "label_encoder": label_encoder,
        }

    if family == "pytorch":
        config = MODEL_CONFIGS[model_name]["params"]
        device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        dataloaders = build_sequence_dataloaders(sequence_bundle, batch_size=config["batch_size"])

        num_features = sequence_bundle["num_features"]
        num_classes = len(sequence_bundle["label_encoder"].classes_)

        if model_name == "cnn":
            model = CNNClassifier(
                input_size=num_features, num_classes=num_classes,
                conv_channels=config["conv_channels"],
                kernel_size=config["kernel_size"], dropout=config["dropout"],
            ).to(device)
        elif model_name == "lstm":
            model = LSTMClassifier(
                input_size=num_features, hidden_size=config["hidden_size"],
                num_layers=config["num_layers"], num_classes=num_classes,
                dropout=config["dropout"],
            ).to(device)
        else:
            raise NotImplementedError(f"Sequence model '{model_name}' is not implemented.")

        y_train_seq = sequence_bundle["train"]["y"]
        class_counts = np.bincount(y_train_seq, minlength=num_classes).astype(np.float32)
        class_weights = len(y_train_seq) / (num_classes * class_counts)
        class_weights = class_weights / class_weights.mean()
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        best_score = -np.inf
        best_state_dict = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        history: list[dict[str, Any]] = []

        epoch_progress = tqdm(
            range(1, config["epochs"] + 1),
            desc=f"Training {model_name}",
            unit="epoch",
            disable=True,
        )

        for epoch in epoch_progress:
            train_metrics = run_sequence_epoch(
                model=model, dataloader=dataloaders["train"],
                criterion=criterion, device=device, optimizer=optimizer,
            )
            val_metrics = run_sequence_epoch(
                model=model, dataloader=dataloaders["validation"],
                criterion=criterion, device=device,
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_auc": train_metrics["auc"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_auc": val_metrics["auc"],
            })
            current_score = val_metrics["macro_f1"]
            if current_score > best_score + config["early_stopping_min_delta"]:
                best_score = current_score
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config["early_stopping_patience"]:
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

    raise NotImplementedError(f"Model family '{family}' is not implemented.")


def print_classification_metrics(
    split_name: str,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    labels: np.ndarray | list[Any],
) -> None:
    """Print compact final metrics for one split."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    try:
        auc = roc_auc_score(
            y_true, np.asarray(y_score), labels=labels,
            multi_class="ovr", average="macro",
        )
        auc_line = f"ROC AUC (macro, OVR): {auc:.4f}"
    except ValueError:
        auc_line = "ROC AUC (macro, OVR): nan"

    print(f"\n{split_name} Results")
    print("-" * (len(split_name) + 8))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(auc_line)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm_df)


def print_split_summary(
    split_name: str,
    X_split: pd.DataFrame,
    y_split: pd.Series,
    split_dates: pd.Series,
) -> None:
    """Print a compact summary for one split."""
    print(
        f"{split_name}: rows={len(X_split)}, features={X_split.shape[1]}, "
        f"range={pd.Timestamp(split_dates.min()).date()} -> {pd.Timestamp(split_dates.max()).date()}, "
        f"labels={y_split.value_counts(dropna=False).sort_index().to_dict()}"
    )


def print_run_header(model_name: str, df: pd.DataFrame, X: pd.DataFrame) -> None:
    """Print a clear header for one model run."""
    line = "=" * 88
    print()
    print(line)
    print(f"RUN START | model={model_name} | target={TARGET_COLUMN} | dataset={DATA_PATH}")
    print(line)
    print(f"Loaded data: rows={len(df)}, columns={len(df.columns)}, feature_count={X.shape[1]}")


def print_run_footer(model_name: str) -> None:
    """Print a clear footer for one model run."""
    line = "=" * 88
    print(line)
    print(f"RUN END | model={model_name}")
    print(line)


def evaluate_model(
    model_bundle: dict[str, Any],
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    sequence_bundle: dict[str, Any] | None = None,
) -> None:
    """Evaluate the model on validation and test splits."""
    family = model_bundle["family"]
    label_encoder: LabelEncoder = model_bundle["label_encoder"]
    labels = label_encoder.classes_

    if family == "xgboost":
        model = model_bundle["model"]
        for split_name, X_split, y_split in (
            ("Validation", X_val, y_val),
            ("Test", X_test, y_test),
        ):
            y_score = model.predict_proba(X_split)
            y_pred = label_encoder.inverse_transform(model.predict(X_split).astype(int))
            print_classification_metrics(split_name, y_split, y_pred, y_score=y_score, labels=labels)
        return

    if family == "pytorch":
        dataloaders = build_sequence_dataloaders(
            sequence_bundle=sequence_bundle,
            batch_size=MODEL_CONFIGS[model_bundle["model_name"]]["params"]["batch_size"],
        )
        for split_name, split_key in (("Validation", "validation"), ("Test", "test")):
            metrics = run_sequence_epoch(
                model=model_bundle["model"],
                dataloader=dataloaders[split_key],
                criterion=model_bundle["criterion"],
                device=model_bundle["device"],
            )
            y_true = label_encoder.inverse_transform(metrics["y_true"].astype(int))
            y_pred = label_encoder.inverse_transform(metrics["y_pred"].astype(int))
            print_classification_metrics(
                split_name, y_true, y_pred, y_score=metrics["y_score"], labels=labels,
            )
        return

    raise NotImplementedError(f"Evaluation for model family '{family}' is not implemented.")


def main(model_name: str = DEFAULT_MODEL_NAME) -> None:
    """Run the multiclass training pipeline on the technical dataset."""
    df = load_data()
    X, y, dates, tickers = prepare_features(df)
    (
        X_train, y_train, tickers_train,
        X_val, y_val, tickers_val,
        X_test, y_test, tickers_test,
    ) = split_data(X, y, dates, tickers)

    print_run_header(model_name=model_name, df=df, X=X)
    print("Split Summary")
    print("-" * 88)
    print_split_summary("Train", X_train, y_train, dates.loc[X_train.index])
    print_split_summary("Validation", X_val, y_val, dates.loc[X_val.index])
    print_split_summary("Test", X_test, y_test, dates.loc[X_test.index])

    sequence_bundle = None
    if MODEL_CONFIGS[model_name]["family"] == "pytorch":
        print("-" * 88)
        print("Sequence Summary")
        print("-" * 88)
        sequence_bundle = prepare_sequence_data(
            X_train=X_train, y_train=y_train, tickers_train=tickers_train,
            X_val=X_val, y_val=y_val, tickers_val=tickers_val,
            X_test=X_test, y_test=y_test, tickers_test=tickers_test,
            sequence_length=DEFAULT_SEQUENCE_LENGTH,
        )
        print(
            "Sequence data: "
            f"train={sequence_bundle['train']['X'].shape}, "
            f"validation={sequence_bundle['validation']['X'].shape}, "
            f"test={sequence_bundle['test']['X'].shape}, "
            f"T={sequence_bundle['sequence_length']}"
        )

    print("-" * 88)
    print("Final Metrics")
    print("-" * 88)
    model = train_model(
        X_train, y_train,
        model_name=model_name, sequence_bundle=sequence_bundle,
    )

    evaluate_model(
        model,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        sequence_bundle=sequence_bundle,
    )
    print_run_footer(model_name=model_name)


if __name__ == "__main__":
    main()
