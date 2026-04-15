"""
lstm_model.py
-------------
LSTM classifier for BUY / HOLD / SELL prediction on US tech stocks.

Follows the same data pipeline as ablation_study.py:
  - same pre-built train / val / test splits (data/splits/)
  - same 38-feature set: TA (10) + Fundamentals (15) + Sentiment (13)
  - same target column: label  (0=HOLD, 1=BUY, 2=SELL)
  - same evaluation metric: macro-averaged F1

What changes vs the classical models:
  - Each prediction uses a rolling window of features (sequence input)
  - Sequences are built per-ticker so AAPL history never bleeds into MSFT
  - The val / test lookback naturally reaches back into the previous split

Experiments run (picks best by val Macro F1):
  1. Baseline LSTM        — seq=20, hidden=64,  dropout=0.3, no attention
  2. LSTM + Attention     — seq=20, hidden=64,  dropout=0.4, soft attention
  3. LSTM + Attention     — seq=30, hidden=64,  dropout=0.4, longer lookback
  4. LSTM + Attention     — seq=20, hidden=128, dropout=0.4, wider model

All variants use:
  - Gradient clipping (max_norm=1.0) — stabilises LSTM training
  - CrossEntropyLoss with inverse-frequency class weights
  - Adam + ReduceLROnPlateau scheduler
  - Early stopping on val Macro F1 (patience=10)

Output
------
  Printed to screen + saved to results/lstm_results.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "data/splits/train.csv"
VAL_PATH    = "data/splits/val.csv"
TEST_PATH   = "data/splits/test.csv"
RESULTS_DIR = Path("results")

# ── Feature column groups (identical to ablation_study.py) ───────────────────
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

FEATURE_COLS = TA_COLS + FUND_COLS + SENT_COLS   # 38 features
TARGET       = "label"
CLASS_NAMES  = ["HOLD", "BUY", "SELL"]           # indices 0, 1, 2

# ── Shared training constants ─────────────────────────────────────────────────
SEQ_LEN    = 20        # default rolling-window length (days)
LR         = 5e-4      # Adam learning rate  (lowered: 1e-3 was overshooting)
WD         = 1e-4      # Adam weight decay (L2 regularisation)
BATCH      = 256       # mini-batch size
MAX_EPOCHS = 100       # hard cap on training epochs (increased from 50)
PATIENCE   = 15        # early stopping patience (val macro F1)
GRAD_CLIP  = 1.0       # gradient clipping max norm (stabilises LSTM)
SEED       = 42

# ── Experiment grid ───────────────────────────────────────────────────────────
# Each dict defines one model configuration.
# Best by val Macro F1 is selected and evaluated on test.
EXPERIMENTS = [
    {
        "name":           "lstm_baseline",
        "seq_len":        20,
        "hidden":         64,
        "n_layers":       2,
        "dropout":        0.3,
        "attention":      False,
        "bidirectional":  False,
    },
    {
        "name":           "lstm_attention_seq20",
        "seq_len":        20,
        "hidden":         64,
        "n_layers":       2,
        "dropout":        0.4,
        "attention":      True,
        "bidirectional":  False,
    },
    {
        "name":           "lstm_attention_seq30",
        "seq_len":        30,
        "hidden":         64,
        "n_layers":       2,
        "dropout":        0.4,
        "attention":      True,
        "bidirectional":  False,
    },
    {
        "name":           "lstm_attention_wide",
        "seq_len":        20,
        "hidden":         128,
        "n_layers":       2,
        "dropout":        0.4,
        "attention":      True,
        "bidirectional":  False,
    },
    # New: bidirectional LSTM — within-window look-back is fine (no future leakage)
    {
        "name":           "lstm_bidir_attention",
        "seq_len":        20,
        "hidden":         64,
        "n_layers":       2,
        "dropout":        0.4,
        "attention":      True,
        "bidirectional":  True,
    },
    {
        "name":           "lstm_bidir_wide",
        "seq_len":        30,
        "hidden":         128,
        "n_layers":       2,
        "dropout":        0.4,
        "attention":      True,
        "bidirectional":  True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the pre-built train / val / test CSVs and verify no date overlap."""
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    val   = pd.read_csv(VAL_PATH,   parse_dates=["Date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["Date"])

    assert train["Date"].max() < val["Date"].min(),  "Train/val date overlap!"
    assert val["Date"].max()   < test["Date"].min(), "Val/test date overlap!"

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING  (fit on train only, apply to all splits)
# ─────────────────────────────────────────────────────────────────────────────

def fit_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[SimpleImputer, StandardScaler]:
    """
    Fit median imputer and standard scaler on the training set.
    Returns fitted objects to be applied to val and test.
    """
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_train)
    scaler  = StandardScaler()
    scaler.fit(X_imp)
    return imputer, scaler


def apply_preprocessor(
    X: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Impute → scale → clip to [-10, 10].
    Clipping prevents extreme outlier values (e.g. from fundamental ratios
    during the GFC) from destabilising gradient updates.
    """
    X_imp    = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return np.clip(X_scaled, -10, 10).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEQUENCE BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    seq_len: int = SEQ_LEN,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Convert flat (row-per-day) data into rolling windows per ticker.

    For each ticker and each target day t, the input window is the scaled
    feature matrix for days [t-seq_len, t-1] and the label is the class
    for day t.

    Val and test windows can look back into the previous split's rows,
    which is correct — at prediction time you have the full prior history.
    We achieve this by concatenating all splits per ticker before windowing
    and then routing each window to its split by the target day's date.

    Rows containing NaN after scaling (shouldn't happen after imputation,
    but guards against edge cases) are silently dropped.
    """
    # Combine all splits into one frame, keeping track of which split each
    # date belongs to.
    train_dates = set(train["Date"].dt.date)
    val_dates   = set(val["Date"].dt.date)
    test_dates  = set(test["Date"].dt.date)

    full = pd.concat([train, val, test], ignore_index=True)
    full = full.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Scale the full feature matrix using train-fitted objects
    X_all     = apply_preprocessor(full[FEATURE_COLS], imputer, scaler)
    y_all     = full[TARGET].values.astype(np.int64)
    dates_all = full["Date"].dt.date.values
    tickers   = full["Ticker"].values

    result: dict[str, dict[str, list]] = {
        split: {"X": [], "y": []}
        for split in ("train", "val", "test")
    }

    for ticker in np.unique(tickers):
        # Extract this ticker's rows in chronological order
        mask    = tickers == ticker
        X_t     = X_all[mask]
        y_t     = y_all[mask]
        dates_t = dates_all[mask]

        # Slide a window of length seq_len over the ticker's time series
        for i in range(seq_len, len(X_t)):
            window = X_t[i - seq_len : i]          # shape: (seq_len, n_features)

            # Drop windows with any NaN (warm-up edge cases)
            if np.isnan(window).any():
                continue

            target_date = dates_t[i]

            if target_date in train_dates:
                result["train"]["X"].append(window)
                result["train"]["y"].append(y_t[i])
            elif target_date in val_dates:
                result["val"]["X"].append(window)
                result["val"]["y"].append(y_t[i])
            elif target_date in test_dates:
                result["test"]["X"].append(window)
                result["test"]["y"].append(y_t[i])

    # Convert lists to numpy arrays
    return {
        split: {
            "X": np.array(data["X"], dtype=np.float32),
            "y": np.array(data["y"], dtype=np.int64),
        }
        for split, data in result.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. PYTORCH DATASET & DATALOADER
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Simple Dataset wrapper around (X, y) numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_dataloaders(
    sequences: dict[str, dict[str, np.ndarray]],
    batch_size: int = BATCH,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            SequenceDataset(sequences[split]["X"], sequences[split]["y"]),
            batch_size=batch_size,
            shuffle=(split == "train"),   # shuffle training only
        )
        for split in ("train", "val", "test")
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. LSTM MODELS
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    2-layer stacked LSTM — baseline, uses only the last hidden state.

    Improvements over v1:
      - LayerNorm after LSTM output stabilises gradient flow
      - 2-layer MLP classifier head (hidden → hidden//2 → n_classes)
        gives more representational capacity before the softmax

    Input shape : (batch, seq_len, n_features)
    Output shape: (batch, n_classes)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int   = 64,
        num_layers: int    = 2,
        num_classes: int   = 3,
        dropout: float     = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        lstm_dropout   = dropout if num_layers > 1 else 0.0
        self.bidir     = bidirectional
        self.lstm      = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=lstm_dropout, bidirectional=bidirectional,
        )
        out_size        = hidden_size * 2 if bidirectional else hidden_size
        self.norm       = nn.LayerNorm(out_size)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(out_size, out_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        # For bidirectional: concat last forward and last backward hidden states
        if self.bidir:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        return self.classifier(self.dropout(self.norm(last_hidden)))


class LSTMWithAttention(nn.Module):
    """
    2-layer stacked LSTM with additive soft attention over all timesteps.

    Instead of discarding all hidden states except the last, attention
    computes a weighted sum over the full sequence of hidden states.
    The weights are learned — the model can focus on the most informative
    days within the window (e.g. a day with an unusual volume spike or a
    sharp sentiment shift) rather than always relying on the most recent day.

    Attention mechanism:
      score_t  = tanh(W * h_t)          (learned projection)
      alpha_t  = softmax(score_t)        (normalised importance weights)
      context  = sum_t(alpha_t * h_t)    (weighted summary of the window)

    Improvements over v1:
      - LayerNorm on the context vector before classification
      - 2-layer MLP classifier head for more representational capacity
      - Optional bidirectional LSTM (safe within a historical window)

    Input shape : (batch, seq_len, n_features)
    Output shape: (batch, n_classes)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int    = 64,
        num_layers: int     = 2,
        num_classes: int    = 3,
        dropout: float      = 0.4,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.bidir   = bidirectional
        self.lstm    = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=lstm_dropout, bidirectional=bidirectional,
        )
        out_size        = hidden_size * 2 if bidirectional else hidden_size
        # Attention: project each hidden state to a scalar score
        self.attention  = nn.Linear(out_size, 1)
        self.norm       = nn.LayerNorm(out_size)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(out_size, out_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # all_hidden: (batch, seq_len, out_size)
        all_hidden, _ = self.lstm(x)

        # Compute attention scores and normalise across timesteps
        scores  = self.attention(torch.tanh(all_hidden))   # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)             # (batch, seq_len, 1)

        # Weighted sum over timesteps → context vector
        context = (weights * all_hidden).sum(dim=1)        # (batch, out_size)
        context = self.dropout(self.norm(context))
        return self.classifier(context)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """
    Inverse-frequency class weights so the loss penalises SELL and BUY
    mistakes more heavily than HOLD (which is ~51% of labels).

    weight_c = total_samples / (n_classes * count_c)
    """
    counts  = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = GRAD_CLIP,
) -> dict[str, float]:
    """
    Run one pass over the dataloader.
    If optimizer is provided, runs in training mode with backprop.
    Gradient clipping prevents exploding gradients in LSTM training.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss  = 0.0
    all_targets: list[np.ndarray] = []
    all_preds:   list[np.ndarray] = []

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            all_targets.append(y_batch.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    n      = len(y_true)

    return {
        "loss":     total_loss / n,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. FULL TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, n_features: int) -> nn.Module:
    """Instantiate the correct model class from a config dict."""
    bidir = cfg.get("bidirectional", False)
    if cfg["attention"]:
        return LSTMWithAttention(
            n_features=n_features,
            hidden_size=cfg["hidden"],
            num_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
            bidirectional=bidir,
        )
    return LSTMClassifier(
        n_features=n_features,
        hidden_size=cfg["hidden"],
        num_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        bidirectional=bidir,
    )


def train_lstm(
    sequences: dict[str, dict[str, np.ndarray]],
    cfg: dict,
    device: torch.device,
) -> tuple[nn.Module, float]:
    """
    Train one LSTM configuration with early stopping on val Macro F1.
    Returns (best_model, best_val_f1).
    """
    torch.manual_seed(SEED)

    n_features = sequences["train"]["X"].shape[2]
    loaders    = make_dataloaders(sequences)
    model      = build_model(cfg, n_features).to(device)

    class_weights = compute_class_weights(sequences["train"]["y"]).to(device)
    # label_smoothing=0.1 softens targets → reduces overconfident wrong predictions
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    # CosineAnnealingWarmRestarts cycles LR smoothly; works better than
    # ReduceLROnPlateau when the loss landscape is noisy (financial data).
    # T_0=10: first restart after 10 epochs; T_mult=2: each cycle doubles.
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    best_val_f1    = -1.0
    best_state     = None
    no_improvement = 0

    print(f"\n  {'Epoch':>5}  {'TrainLoss':>9}  {'TrainF1':>7}  "
          f"{'ValLoss':>7}  {'ValF1':>6}  {'LR':>8}")
    print("  " + "-" * 55)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_m = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_m   = run_epoch(model, loaders["val"],   criterion, device)

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  {epoch:>5}  {train_m['loss']:>9.4f}  {train_m['macro_f1']:>7.4f}  "
            f"{val_m['loss']:>7.4f}  {val_m['macro_f1']:>6.4f}  {current_lr:>8.2e}"
        )

        if val_m["macro_f1"] > best_val_f1 + 1e-4:
            best_val_f1    = val_m["macro_f1"]
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val Macro F1 = {best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_val_f1


# ─────────────────────────────────────────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    sequences: dict[str, dict[str, np.ndarray]],
    device: torch.device,
    split: str,
) -> dict:
    """
    Run model on one split, print a full classification report,
    and return a results dict for JSON serialisation.
    """
    model.eval()
    loader = DataLoader(
        SequenceDataset(sequences[split]["X"], sequences[split]["y"]),
        batch_size=BATCH,
        shuffle=False,
    )

    all_targets: list[np.ndarray] = []
    all_preds:   list[np.ndarray] = []
    all_probs:   list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_targets.append(y_batch.numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_probs.append(probs)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None,    zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    cm        = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df     = pd.DataFrame(
        cm,
        index   =[f"true_{c}" for c in CLASS_NAMES],
        columns =[f"pred_{c}" for c in CLASS_NAMES],
    )

    label_name = split.capitalize()
    print(f"\n  [{label_name}]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}  ← primary metric")
    print(f"  ROC AUC  : {roc_auc:.4f}  (OvR macro)")
    print(f"\n  Per-class F1:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:<6} ({i}): {per_class[i]:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    print(f"  Confusion Matrix:")
    print(cm_df.to_string())

    return {
        "accuracy":    round(acc, 4),
        "macro_f1":    round(macro_f1, 4),
        "roc_auc":     round(roc_auc, 4),
        "per_class_f1": {
            CLASS_NAMES[i]: round(float(per_class[i]), 4)
            for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device : {device}")

    # ── Load splits ───────────────────────────────────────────────────────────
    print("\nLoading splits...")
    train, val, test = load_splits()
    print(f"  Train : {len(train):,} rows  "
          f"({train['Date'].min().date()} → {train['Date'].max().date()})")
    print(f"  Val   : {len(val):,} rows  "
          f"({val['Date'].min().date()} → {val['Date'].max().date()})")
    print(f"  Test  : {len(test):,} rows  "
          f"({test['Date'].min().date()} → {test['Date'].max().date()})")
    print(f"\n  Features : {len(FEATURE_COLS)}  "
          f"(TA={len(TA_COLS)}, Fund={len(FUND_COLS)}, Sent={len(SENT_COLS)})")

    # ── Fit preprocessor on train only ───────────────────────────────────────
    print("\nFitting preprocessor on train...")
    imputer, scaler = fit_preprocessor(train[FEATURE_COLS])

    # ── Pre-build sequences for each unique seq_len in experiments ────────────
    # We cache them so tickers with seq_len=20 are not rebuilt multiple times.
    unique_seq_lens = list({cfg["seq_len"] for cfg in EXPERIMENTS})
    seq_cache: dict[int, dict] = {}
    for sl in unique_seq_lens:
        print(f"\nBuilding {sl}-day rolling sequences per ticker...")
        seq_cache[sl] = build_sequences(train, val, test, imputer, scaler, seq_len=sl)
        for split, data in seq_cache[sl].items():
            uniq, cnts = np.unique(data["y"], return_counts=True)
            print(f"  {split:<5} : {data['X'].shape}  "
                  f"labels={dict(zip(uniq.tolist(), cnts.tolist()))}")

    # ── Run all experiments ───────────────────────────────────────────────────
    all_results: list[dict] = []

    for cfg in EXPERIMENTS:
        print("\n" + "=" * 65)
        print(f"EXPERIMENT : {cfg['name']}")
        print(f"  seq_len={cfg['seq_len']}, hidden={cfg['hidden']}, "
              f"layers={cfg['n_layers']}, dropout={cfg['dropout']}, "
              f"attention={cfg['attention']}")
        print("=" * 65)

        sequences          = seq_cache[cfg["seq_len"]]
        model, best_val_f1 = train_lstm(sequences, cfg, device)

        print(f"\n  Evaluating best checkpoint (val Macro F1 = {best_val_f1:.4f})...")
        val_results  = evaluate(model, sequences, device, "val")
        test_results = evaluate(model, sequences, device, "test")

        all_results.append({
            "name":         cfg["name"],
            "config":       cfg,
            "val_macro_f1": val_results["macro_f1"],
            "test_macro_f1":test_results["macro_f1"],
            "validation":   val_results,
            "test":         test_results,
        })

    # ── Experiment summary ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EXPERIMENT SUMMARY")
    print("=" * 65)
    print(f"  {'Experiment':<30}  {'Val F1':>7}  {'Test F1':>8}  {'Val AUC':>8}  {'Test AUC':>9}")
    print("  " + "-" * 68)
    for r in all_results:
        print(f"  {r['name']:<30}  {r['val_macro_f1']:>7.4f}  "
              f"{r['test_macro_f1']:>8.4f}  "
              f"{r['validation']['roc_auc']:>8.4f}  "
              f"{r['test']['roc_auc']:>9.4f}")

    best = max(all_results, key=lambda r: r["val_macro_f1"])
    print(f"\n  Best by val Macro F1 : {best['name']}")
    print(f"    Val  Macro F1 : {best['val_macro_f1']:.4f}  |  Val  ROC AUC : {best['validation']['roc_auc']:.4f}")
    print(f"    Test Macro F1 : {best['test_macro_f1']:.4f}  |  Test ROC AUC : {best['test']['roc_auc']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "model":        "lstm",
        "features":     "ta_fund_sent",
        "n_features":   len(FEATURE_COLS),
        "experiments":  all_results,
        "best": {
            "name":         best["name"],
            "config":       best["config"],
            "val_macro_f1": best["val_macro_f1"],
            "test_macro_f1":best["test_macro_f1"],
            "validation":   best["validation"],
            "test":         best["test"],
        },
    }

    out_path = RESULTS_DIR / "lstm_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
