"""
transformer_ablation.py
-----------------------
Feature-set ablation study for a Transformer encoder model.

Mirrors lstm_ablation.py exactly — same splits, same four feature sets,
same preprocessing, same training hyperparameters — so results are
directly comparable.

Architecture
------------
  Input projection : Linear(n_features → d_model=64)
  Positional embed : Learnable, shape (1, seq_len, d_model)
  CLS token        : Learnable, prepended before the sequence
  Encoder          : 2-layer TransformerEncoderLayer
                       d_model=64, nhead=4, dim_ff=256
                       Pre-LN (norm_first=True), dropout=0.1
  Head             : LayerNorm → Linear(64→32) → ReLU → Dropout → Linear(32→3)
  Output           : Raw logits (3 classes: HOLD / BUY / SELL)

Why CLS token?
  Appending a learnable [CLS] token at position 0 lets the model aggregate
  sequence information through attention without any fixed pooling choice.
  The CLS output is used as the sequence representation.

Training (identical to lstm_ablation.py)
-----------------------------------------
  Loss      : CrossEntropyLoss + inverse-frequency class weights + label_smoothing=0.1
  Optimiser : AdamW (lr=5e-4, weight_decay=1e-4)
  Scheduler : CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6)
  Clip      : grad norm ≤ 1.0
  Early stop: patience=15 on val Macro F1
  Max epochs: 100

Feature sets (same as lstm_ablation.py)
-----------------------------------------
  ta_only       10 features
  ta_sent       23 features  ← expected best (matches LSTM ablation finding)
  ta_fund       25 features
  ta_fund_sent  38 features

Output
------
  results/transformer_ablation_results.json
"""

from __future__ import annotations

import json
import math
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
from tqdm import tqdm

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

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN    = 20
D_MODEL    = 64
N_HEAD     = 4        # D_MODEL must be divisible by N_HEAD
N_LAYERS   = 2
DIM_FF     = 256      # 4 × D_MODEL — standard Transformer ratio
DROPOUT    = 0.1      # Transformers use lower dropout than LSTMs
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
    assert train["Date"].max() < val["Date"].min(), "Train/val date overlap"
    assert val["Date"].max()   < test["Date"].min(), "Val/test date overlap"
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
    Per-ticker rolling windows.  Val/test windows may look back into the
    previous split's raw data — correct at inference time.
    """
    train_dates = set(train["Date"].dt.date)
    val_dates   = set(val["Date"].dt.date)
    test_dates  = set(test["Date"].dt.date)

    full  = pd.concat([train, val, test], ignore_index=True)
    full  = full.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    X_all = preprocess(full[feature_cols], imp, scl)
    y_all = full[TARGET].values.astype(np.int64)
    d_all = full["Date"].dt.date.values
    t_all = full["Ticker"].values

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
            else:                  continue
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
    def __len__(self):              return len(self.X)
    def __getitem__(self, i):       return self.X[i], self.y[i]


def make_loader(seqs, split, shuffle=False):
    return DataLoader(
        SequenceDataset(seqs[split]["X"], seqs[split]["y"]),
        batch_size=BATCH, shuffle=shuffle,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TransformerClassifier(nn.Module):
    """
    Transformer encoder for sequence classification.

    Pipeline
    --------
    1. Project each time step : Linear(n_features → d_model)
    2. Add learnable positional embedding (shape: seq_len × d_model)
    3. Prepend a learnable [CLS] token
    4. Pass through N TransformerEncoderLayers (Pre-LN, multi-head attention)
    5. Take the CLS output → LayerNorm → MLP head → logits

    Pre-LayerNorm (norm_first=True) stabilises training at higher learning
    rates compared to the original post-LN Transformer.
    """

    def __init__(
        self,
        n_features: int,
        seq_len:    int = SEQ_LEN,
        d_model:    int = D_MODEL,
        nhead:      int = N_HEAD,
        n_layers:   int = N_LAYERS,
        dim_ff:     int = DIM_FF,
        dropout:    float = DROPOUT,
        n_classes:  int = 3,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Linear(n_features, d_model)

        # ── Learnable positional embedding (seq_len positions only, not CLS) ─
        self.pos_embed  = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Learnable CLS token ───────────────────────────────────────────────
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_ff,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Classification head ───────────────────────────────────────────────
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, n_features)
        returns : (batch, n_classes)  — raw logits
        """
        B = x.size(0)

        # Project features and add positional encoding
        x = self.input_proj(x) + self.pos_embed          # (B, seq_len, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                  # (B, seq_len+1, d_model)

        # Transformer encoder
        x   = self.transformer(x)                         # (B, seq_len+1, d_model)

        # CLS output → head
        cls_out = self.norm(x[:, 0])                      # (B, d_model)
        return self.head(cls_out)                          # (B, n_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
            Xb, yb  = Xb.to(device), yb.to(device)
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
    n      = len(y_true)
    return {
        "loss"    : total_loss / n if total_loss > 0 else 0.0,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true"  : y_true,
        "y_pred"  : y_pred,
        "y_prob"  : y_prob,
    }


def train_model(seqs, device, n_features):
    torch.manual_seed(SEED)
    model   = TransformerClassifier(n_features).to(device)
    weights = compute_class_weights(seqs["train"]["y"], device)
    crit    = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch     = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                  opt, T_0=10, T_mult=2, eta_min=1e-6)

    train_loader = make_loader(seqs, "train", shuffle=True)
    val_loader   = make_loader(seqs, "val")

    best_val_f1, best_state, no_imp = -1.0, None, 0

    pbar = tqdm(range(1, MAX_EPOCHS + 1), desc="  Training", unit="epoch",
                ncols=90, leave=True)

    for epoch in pbar:
        tr = run_epoch(model, train_loader, crit, device, opt)
        vl = run_epoch(model, val_loader,   crit, device)
        sch.step(epoch)
        lr = opt.param_groups[0]["lr"]

        pbar.set_postfix({
            "tr_loss": f"{tr['loss']:.4f}",
            "tr_f1"  : f"{tr['macro_f1']:.4f}",
            "val_f1" : f"{vl['macro_f1']:.4f}",
            "best"   : f"{best_val_f1:.4f}",
            "lr"     : f"{lr:.1e}",
        })

        if vl["macro_f1"] > best_val_f1 + 1e-4:
            best_val_f1 = vl["macro_f1"]
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp      = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                pbar.set_description(f"  Early stop (epoch {epoch})")
                pbar.close()
                break

    model.load_state_dict(best_state)
    return model, best_val_f1


# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, seqs, device, split):
    m         = run_epoch(model, make_loader(seqs, split), None, device)
    y_true    = m["y_true"]
    y_pred    = m["y_pred"]
    y_prob    = m["y_prob"]

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
        "accuracy"    : round(float(acc), 4),
        "macro_f1"    : round(float(macro_f1), 4),
        "roc_auc"     : round(float(roc_auc), 4),
        "per_class_f1": {CLASS_NAMES[i]: round(float(per_class[i]), 4)
                         for i in range(3)},
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(device):
    """
    Verify the model runs correctly before committing to full training.
    Checks forward pass shapes and that the output is valid logits.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)

    for n_feat in [10, 23, 25, 38]:
        model  = TransformerClassifier(n_feat).to(device)
        params = model.count_parameters()
        x      = torch.randn(4, SEQ_LEN, n_feat).to(device)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (4, 3), f"Wrong output shape: {out.shape}"
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

        probs = torch.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4).to(device), atol=1e-5), \
            "Softmax probs don't sum to 1"

        print(f"  n_features={n_feat:2d}  output={tuple(out.shape)}  "
              f"params={params:,}  logit_range=[{out.min():.3f}, {out.max():.3f}]  OK")

    print("\n  All checks passed.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available()          else
              torch.device("cpu"))
    print(f"Device : {device}")

    # ── Sanity check before any training ─────────────────────────────────────
    sanity_check(device)

    # ── Load splits ───────────────────────────────────────────────────────────
    print("Loading splits...")
    train, val, test = load_splits()
    print(f"  Train : {len(train):,}  Val : {len(val):,}  Test : {len(test):,}")

    all_results = []

    # ── Ablation loop ─────────────────────────────────────────────────────────
    for set_name, feature_cols in ABLATION_SETS.items():
        n_feat = len(feature_cols)
        print("\n" + "=" * 60)
        print(f"ABLATION: {set_name}  ({n_feat} features)")
        print("=" * 60)

        imp, scl = fit_preprocessor(train[feature_cols])
        seqs     = build_sequences(train, val, test, feature_cols, imp, scl)

        tr_labels = dict(zip(*np.unique(seqs["train"]["y"], return_counts=True)))
        vl_labels = dict(zip(*np.unique(seqs["val"]["y"],   return_counts=True)))
        ts_labels = dict(zip(*np.unique(seqs["test"]["y"],  return_counts=True)))
        print(f"  Sequences — train:{seqs['train']['X'].shape}  "
              f"val:{seqs['val']['X'].shape}  test:{seqs['test']['X'].shape}")
        print(f"  Train labels : {tr_labels}")
        print(f"  Val labels   : {vl_labels}")
        print(f"  Test labels  : {ts_labels}")

        model = TransformerClassifier(n_feat).to(device)
        print(f"  Parameters   : {model.count_parameters():,}")

        print(f"\n  Training Transformer ({n_feat} features)...")
        model, best_val_f1 = train_model(seqs, device, n_feat)
        print(f"\n  Best val Macro F1 = {best_val_f1:.4f}")

        print("\n  Evaluating best checkpoint...")
        val_res  = evaluate(model, seqs, device, "val")
        test_res = evaluate(model, seqs, device, "test")

        all_results.append({
            "feature_set" : set_name,
            "n_features"  : n_feat,
            "best_val_f1" : round(best_val_f1, 4),
            "val"         : val_res,
            "test"        : test_res,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRANSFORMER ABLATION SUMMARY")
    print("=" * 60)
    print(f"  {'Feature Set':<16} {'N Feat':>6}  {'Val F1':>7}  {'Test F1':>7}  "
          f"{'Val AUC':>7}  {'Test AUC':>8}")
    print("  " + "-" * 62)
    baseline_f1 = None
    for r in all_results:
        if baseline_f1 is None:
            baseline_f1 = r["test"]["macro_f1"]
        print(f"  {r['feature_set']:<16} {r['n_features']:>6}  "
              f"{r['val']['macro_f1']:>7.4f}  {r['test']['macro_f1']:>7.4f}  "
              f"{r['val']['roc_auc']:>7.4f}  {r['test']['roc_auc']:>8.4f}")

    print(f"\n  Gains vs ta_only ({baseline_f1:.4f}) — Test Macro F1:")
    for r in all_results[1:]:
        delta = r["test"]["macro_f1"] - baseline_f1
        print(f"    {r['feature_set']:<16}: {r['test']['macro_f1']:.4f}  ({delta:+.4f})")

    # ── Load LSTM results for side-by-side comparison ─────────────────────────
    lstm_path = RESULTS_DIR / "lstm_ablation_results.json"
    if lstm_path.exists():
        with open(lstm_path) as f:
            lstm_data = json.load(f)
        # results is a dict keyed by feature_set name
        lstm_map = lstm_data["results"]

        print("\n" + "=" * 60)
        print("TRANSFORMER vs LSTM — Test Macro F1")
        print("=" * 60)
        print(f"  {'Feature Set':<16}  {'Transformer':>12}  {'LSTM':>8}  {'Delta':>7}")
        print("  " + "-" * 50)
        for r in all_results:
            fs   = r["feature_set"]
            t_f1 = r["test"]["macro_f1"]
            l_f1 = lstm_map[fs]["test_macro_f1"] if fs in lstm_map else float("nan")
            delta = t_f1 - l_f1
            print(f"  {fs:<16}  {t_f1:>12.4f}  {l_f1:>8.4f}  {delta:>+7.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "model"     : "transformer",
        "architecture": {
            "d_model"   : D_MODEL,
            "nhead"     : N_HEAD,
            "n_layers"  : N_LAYERS,
            "dim_ff"    : DIM_FF,
            "dropout"   : DROPOUT,
            "seq_len"   : SEQ_LEN,
            "pos_embed" : "learnable",
            "cls_token" : True,
            "norm_first": True,
        },
        "training"  : {
            "lr": LR, "weight_decay": WD, "batch": BATCH,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingWarmRestarts(T_0=10,T_mult=2)",
            "loss": "CrossEntropyLoss(class_weights+label_smoothing=0.1)",
        },
        "results"   : all_results,
        "best"      : max(all_results, key=lambda r: r["val"]["macro_f1"]),
    }

    out_path = RESULTS_DIR / "transformer_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
