"""
lstm_ablation.py
----------------
Feature-set ablation study for the LSTM model.

Mirrors the XGBoost ablation in ablation_study.py but uses the LSTM
baseline configuration (seq=20, hidden=64, 2-layer, no attention).

The same four feature sets are tested:
  ta_only       10 features  — technical indicators only
  ta_sent       23 features  — TA + sentiment
  ta_fund       25 features  — TA + fundamentals
  ta_fund_sent  38 features  — TA + fundamentals + sentiment  (full)

This isolates the incremental contribution of each data modality
for the deep learning model specifically, as distinct from the
classical-model ablation already run with XGBoost.

Architecture (fixed across all runs)
--------------------------------------
  LSTM: 2-layer, hidden=64, dropout=0.3
  Head: Linear(64→32) → ReLU → Dropout → Linear(32→3)
  Norm: LayerNorm after LSTM output
  Loss: CrossEntropyLoss with inverse-frequency class weights + label_smoothing=0.1
  Optimiser: Adam (lr=5e-4, weight_decay=1e-4)
  Scheduler: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
  Early stopping: patience=15 on val Macro F1
  Max epochs: 100

Input / Output
--------------
  data/splits/train.csv  — fit preprocessor + train model
  data/splits/val.csv    — early stopping signal
  data/splits/test.csv   — final held-out evaluation

  results/lstm_ablation_results.json
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

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "data/splits/train.csv"
VAL_PATH    = "data/splits/val.csv"
TEST_PATH   = "data/splits/test.csv"
RESULTS_DIR = Path("results")

# ── Feature column groups ──────────────────────────────────────────────────────
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

ABLATION_SETS = {
    "ta_only"     : TA_COLS,
    "ta_sent"     : TA_COLS + SENT_COLS,
    "ta_fund"     : TA_COLS + FUND_COLS,
    "ta_fund_sent": TA_COLS + FUND_COLS + SENT_COLS,
}

TARGET      = "label"
CLASS_NAMES = ["HOLD", "BUY", "SELL"]

# ── Training constants ─────────────────────────────────────────────────────────
SEQ_LEN    = 20
LR         = 5e-4
WD         = 1e-4
BATCH      = 256
MAX_EPOCHS = 100
PATIENCE   = 15
GRAD_CLIP  = 1.0
SEED       = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_splits():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    val   = pd.read_csv(VAL_PATH,   parse_dates=["Date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["Date"])
    assert train["Date"].max() < val["Date"].min()
    assert val["Date"].max()   < test["Date"].min()
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def fit_preprocessor(X_train):
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    scl.fit(imp.fit_transform(X_train))
    return imp, scl


def preprocess(X, imp, scl):
    return np.clip(scl.transform(imp.transform(X)), -10, 10).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEQUENCE BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(train, val, test, feature_cols, imp, scl, seq_len=SEQ_LEN):
    """
    Build per-ticker rolling windows.
    Val/test windows look back into the previous split — correct behaviour
    since at prediction time the full prior history is available.
    """
    train_dates = set(train["Date"].dt.date)
    val_dates   = set(val["Date"].dt.date)
    test_dates  = set(test["Date"].dt.date)

    full    = pd.concat([train, val, test], ignore_index=True)
    full    = full.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    X_all   = preprocess(full[feature_cols], imp, scl)
    y_all   = full[TARGET].values.astype(np.int64)
    d_all   = full["Date"].dt.date.values
    t_all   = full["Ticker"].values

    res = {s: {"X": [], "y": []} for s in ("train", "val", "test")}

    for ticker in np.unique(t_all):
        mask = t_all == ticker
        Xt, yt, dt = X_all[mask], y_all[mask], d_all[mask]
        for i in range(seq_len, len(Xt)):
            w = Xt[i - seq_len: i]
            if np.isnan(w).any():
                continue
            d = dt[i]
            if   d in train_dates: split = "train"
            elif d in val_dates:   split = "val"
            elif d in test_dates:  split = "test"
            else: continue
            res[split]["X"].append(w)
            res[split]["y"].append(yt[i])

    return {
        s: {"X": np.array(v["X"], dtype=np.float32),
            "y": np.array(v["y"], dtype=np.int64)}
        for s, v in res.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_loader(seqs, split, shuffle=False):
    return DataLoader(
        SequenceDataset(seqs[split]["X"], seqs[split]["y"]),
        batch_size=BATCH, shuffle=shuffle,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    2-layer LSTM with LayerNorm + 2-layer MLP head.
    n_features varies per ablation run; everything else is fixed.
    """
    def __init__(self, n_features, hidden=64, n_layers=2,
                 n_classes=3, dropout=0.3):
        super().__init__()
        lstm_drop  = dropout if n_layers > 1 else 0.0
        self.lstm  = nn.LSTM(n_features, hidden, n_layers,
                             batch_first=True, dropout=lstm_drop)
        self.norm  = nn.LayerNorm(hidden)
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(self.drop(self.norm(h[-1])))


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y, device):
    counts  = np.bincount(y, minlength=3).astype(np.float32)
    weights = len(y) / (3 * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_y, all_pred, all_prob = 0.0, [], [], []

    with torch.set_grad_enabled(training):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits  = model(Xb)
            if criterion is not None:
                loss = criterion(logits, yb)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                total_loss += loss.item() * len(yb)
            all_y.append(yb.cpu().numpy())
            all_pred.append(logits.argmax(dim=1).cpu().numpy())
            all_prob.append(torch.softmax(logits, dim=1).cpu().detach().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return {
        "loss"    : total_loss / len(y_true),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true"  : y_true,
        "y_pred"  : y_pred,
        "y_prob"  : y_prob,
    }


def train_model(seqs, device):
    torch.manual_seed(SEED)
    n_feat  = seqs["train"]["X"].shape[2]
    model   = LSTMClassifier(n_feat).to(device)
    weights = compute_class_weights(seqs["train"]["y"], device)
    crit    = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch     = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6)

    train_loader = make_loader(seqs, "train", shuffle=True)
    val_loader   = make_loader(seqs, "val")

    best_val_f1, best_state, no_imp = -1.0, None, 0

    print(f"  {'Epoch':>5}  {'TrainLoss':>9}  {'TrainF1':>7}  "
          f"{'ValLoss':>7}  {'ValF1':>6}  {'LR':>8}")
    print("  " + "-" * 55)

    for epoch in range(1, MAX_EPOCHS + 1):
        tr = run_epoch(model, train_loader, crit, device, opt)
        vl = run_epoch(model, val_loader,   crit, device)
        sch.step(epoch)
        lr = opt.param_groups[0]["lr"]

        print(f"  {epoch:>5}  {tr['loss']:>9.4f}  {tr['macro_f1']:>7.4f}  "
              f"{vl['loss']:>7.4f}  {vl['macro_f1']:>6.4f}  {lr:>8.2e}")

        if vl["macro_f1"] > best_val_f1 + 1e-4:
            best_val_f1 = vl["macro_f1"]
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp      = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val Macro F1 = {best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_val_f1


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, seqs, device, split):
    m      = run_epoch(model, make_loader(seqs, split), None, device)
    y_true = m["y_true"]
    y_pred = m["y_pred"]
    y_prob = m["y_prob"]

    acc       = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None,    zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    cm        = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    label = split.capitalize()
    print(f"\n  [{label}]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro F1 : {macro_f1:.4f}  ← primary metric")
    print(f"  ROC AUC  : {roc_auc:.4f}  (OvR macro)")
    print(f"\n  Per-class F1:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:<6} ({i}): {per_class[i]:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES, zero_division=0))
    cm_df = pd.DataFrame(cm,
        index  =[f"true_{c}" for c in CLASS_NAMES],
        columns=[f"pred_{c}" for c in CLASS_NAMES])
    print("  Confusion Matrix:")
    print(cm_df.to_string())

    return {
        "accuracy"    : round(acc, 4),
        "macro_f1"    : round(macro_f1, 4),
        "roc_auc"     : round(roc_auc, 4),
        "per_class_f1": {CLASS_NAMES[i]: round(float(per_class[i]), 4)
                         for i in range(3)},
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available()         else
              torch.device("cpu"))
    print(f"Device : {device}")

    print("\nLoading splits...")
    train, val, test = load_splits()
    print(f"  Train : {len(train):,}  Val : {len(val):,}  Test : {len(test):,}")

    all_results = {}

    for set_name, feature_cols in ABLATION_SETS.items():
        n_feat = len(feature_cols)
        print("\n" + "=" * 65)
        print(f"ABLATION: {set_name}  ({n_feat} features)")
        print("=" * 65)

        # Fit preprocessor on train features only
        imp, scl = fit_preprocessor(train[feature_cols])

        # Build sequences
        print(f"\nBuilding {SEQ_LEN}-day sequences...")
        seqs = build_sequences(train, val, test, feature_cols, imp, scl)
        for split, data in seqs.items():
            uniq, cnts = np.unique(data["y"], return_counts=True)
            print(f"  {split:<5}: {data['X'].shape}  "
                  f"labels={dict(zip(uniq.tolist(), cnts.tolist()))}")

        # Train
        print(f"\nTraining LSTM (n_features={n_feat})...")
        model, best_val_f1 = train_model(seqs, device)

        # Evaluate
        print(f"\nEvaluating best checkpoint (val Macro F1 = {best_val_f1:.4f})...")
        val_res  = evaluate(model, seqs, device, "val")
        test_res = evaluate(model, seqs, device, "test")

        all_results[set_name] = {
            "n_features"  : n_feat,
            "feature_cols": feature_cols,
            "val_macro_f1": val_res["macro_f1"],
            "val_roc_auc" : val_res["roc_auc"],
            "test_macro_f1": test_res["macro_f1"],
            "test_roc_auc" : test_res["roc_auc"],
            "validation"  : val_res,
            "test"        : test_res,
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("LSTM ABLATION SUMMARY")
    print("=" * 65)
    print(f"  {'Feature Set':<20} {'N Feat':>6} {'Val F1':>8} {'Test F1':>8} "
          f"{'Val AUC':>8} {'Test AUC':>9}")
    print("  " + "-" * 63)

    ta_only_f1 = all_results["ta_only"]["test_macro_f1"]
    for set_name, r in all_results.items():
        print(f"  {set_name:<20} {r['n_features']:>6} "
              f"{r['val_macro_f1']:>8.4f} {r['test_macro_f1']:>8.4f} "
              f"{r['val_roc_auc']:>8.4f} {r['test_roc_auc']:>9.4f}")

    print(f"\n  Gains vs TA-only ({ta_only_f1:.4f}) — Test Macro F1:")
    for set_name, r in all_results.items():
        if set_name == "ta_only":
            continue
        gain = r["test_macro_f1"] - ta_only_f1
        print(f"    {set_name:<20}: {r['test_macro_f1']:.4f}  ({gain:+.4f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "description": (
            "LSTM feature-set ablation. Same architecture and hyperparameters "
            "across all runs (seq=20, hidden=64, 2-layer, no attention). "
            "Only the input feature set changes. Mirrors XGBoost ablation in "
            "ablation_results.json. Primary metric: Test Macro F1."
        ),
        "model_config": {
            "seq_len": SEQ_LEN, "hidden": 64, "n_layers": 2,
            "dropout": 0.3, "attention": False,
            "lr": LR, "weight_decay": WD, "batch_size": BATCH,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "label_smoothing": 0.1,
            "scheduler": "CosineAnnealingWarmRestarts(T_0=10, T_mult=2)",
        },
        "results": all_results,
    }
    out_path = RESULTS_DIR / "lstm_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
