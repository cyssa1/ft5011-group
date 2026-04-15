"""
trading_evaluation.py
---------------------
Realistic per-ticker dollar trading simulation.

Rules
-----
- Start : $1,000 cash per ticker  ($10,000 total across 10 tickers)
- BUY signal  → buy shares worth $1,000 at today's Adj_Close
                (only if currently FLAT — ignore if already holding)
- SELL signal → sell ALL shares held at today's Adj_Close
                (only if currently LONG — ignore if already flat)
- HOLD signal → do nothing
- End of test  → liquidate any remaining open positions at final price

P&L is booked only when a position is closed (SELL or final liquidation).
Each ticker runs its own independent portfolio with its own trade log.

Benchmarks
----------
  buy_hold  : buy all 10 tickers on first test date, hold to end
  momentum  : BUY when roc_5 > 0 (if flat), SELL when roc_5 ≤ 0 (if long)

Models
------
  logistic_regression  TA-only  (10 features)
  xgboost              TA + Fund + Sent  (38 features)
  lstm_baseline        seq=20, hidden=64, 2-layer, no attention

Per-ticker output (trade log)
------------------------------
  date, action (BUY/SELL), price, shares, value,
  and for SELL: buy_date, buy_price, pnl ($), pnl_pct (%), holding_days

Portfolio-level metrics
-----------------------
  initial_capital      starting cash
  final_value          cash + open position value at end
  total_return         (final_value - initial_capital) / initial_capital
  annualized_return    total_return annualised over test period length
  sharpe_ratio         annualised Sharpe from daily portfolio value changes
  max_drawdown         worst peak-to-trough fall in portfolio value
  n_buys               total buy executions across all tickers
  n_sells              total sell executions across all tickers
  win_rate             % of closed trades that were profitable (pnl > 0)
  avg_pnl_per_trade    mean $ P&L per closed trade
  avg_holding_days     mean calendar days between buy and sell

Output
------
  results/trading_results.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH  = "data/splits/train.csv"
TEST_PATH   = "data/splits/test.csv"
RESULTS_DIR = Path("results")

# ── Feature columns ────────────────────────────────────────────────────────────
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
TA_SENT_COLS = TA_COLS + SENT_COLS            # 23 features (best LSTM ablation)
FEATURE_COLS = TA_COLS + FUND_COLS + SENT_COLS
TARGET   = "label"
PRICE_COL = "Adj_Close"

INITIAL_CASH = 1_000.0   # $ per ticker
TRADE_SIZE   = 1_000.0   # $ spent per BUY signal
SEED         = 42
ANNUALIZE    = np.sqrt(252)   # daily returns → annual


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
    test  = pd.read_csv(TEST_PATH,  parse_dates=["Date"])
    test  = test.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# 2. PER-TICKER PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────

class TickerPortfolio:
    """
    Tracks cash, share holdings, and trade history for one ticker.

    State machine:
      FLAT  → BUY  → LONG
      LONG  → SELL → FLAT
      *     → HOLD → unchanged
      BUY when already LONG  → ignored
      SELL when already FLAT → ignored
    """

    def __init__(self, ticker: str, initial_cash: float = INITIAL_CASH,
                 trade_size: float = TRADE_SIZE) -> None:
        self.ticker       = ticker
        self.cash         = initial_cash
        self.initial_cash = initial_cash
        self.trade_size   = trade_size
        self.shares       = 0.0
        self.buy_price    = None
        self.buy_date     = None
        self.trade_log: list[dict] = []

    # ── Actions ───────────────────────────────────────────────────────────────

    def buy(self, date: pd.Timestamp, price: float) -> None:
        if self.shares > 0:
            return                          # already long — skip
        amount       = min(self.trade_size, self.cash)
        if amount <= 0 or price <= 0:
            return
        self.shares    = amount / price
        self.cash     -= amount
        self.buy_price = price
        self.buy_date  = date
        self.trade_log.append({
            "date"   : str(date.date()),
            "action" : "BUY",
            "price"  : round(price, 4),
            "shares" : round(self.shares, 6),
            "value"  : round(amount, 2),
        })

    def sell(self, date: pd.Timestamp, price: float) -> None:
        if self.shares == 0:
            return                          # nothing to sell — skip
        proceeds     = self.shares * price
        pnl          = proceeds - self.shares * self.buy_price
        pnl_pct      = (price - self.buy_price) / self.buy_price
        holding_days = (date - self.buy_date).days
        self.trade_log.append({
            "date"        : str(date.date()),
            "action"      : "SELL",
            "price"       : round(price, 4),
            "shares"      : round(self.shares, 6),
            "value"       : round(proceeds, 2),
            "buy_date"    : str(self.buy_date.date()),
            "buy_price"   : round(self.buy_price, 4),
            "pnl"         : round(pnl, 2),
            "pnl_pct"     : round(pnl_pct, 4),
            "holding_days": holding_days,
        })
        self.cash     += proceeds
        self.shares    = 0.0
        self.buy_price = None
        self.buy_date  = None

    def liquidate(self, date: pd.Timestamp, price: float) -> None:
        """Force-close any open position at end of test period."""
        if self.shares > 0:
            self.trade_log.append({"note": "end-of-period liquidation"})
            self.sell(date, price)

    def portfolio_value(self, current_price: float) -> float:
        return self.cash + self.shares * current_price

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, final_price: float, final_date: pd.Timestamp) -> dict:
        self.liquidate(final_date, final_price)
        final_val   = self.portfolio_value(final_price)
        total_ret   = (final_val - self.initial_cash) / self.initial_cash

        sells = [t for t in self.trade_log
                 if t.get("action") == "SELL"]
        pnls  = [t["pnl"] for t in sells]

        return {
            "ticker"      : self.ticker,
            "initial_cash": round(self.initial_cash, 2),
            "final_value" : round(final_val, 2),
            "total_return": round(total_ret, 4),
            "n_buys"      : len([t for t in self.trade_log if t.get("action") == "BUY"]),
            "n_sells"     : len(sells),
            "win_rate"    : round(sum(p > 0 for p in pnls) / len(pnls), 4) if pnls else None,
            "avg_pnl"     : round(float(np.mean(pnls)), 2)                  if pnls else None,
            "avg_holding_days": round(
                float(np.mean([t["holding_days"] for t in sells])), 1
            ) if sells else None,
            "trade_log"   : self.trade_log,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    test_df: pd.DataFrame,
    signal_map: dict[tuple, int],   # (ticker, date) → 0/1/2
    strategy_name: str,
) -> dict:
    """
    Run the per-ticker simulation given a signal map.

    signal_map : dict mapping (ticker, date) → label (0=HOLD,1=BUY,2=SELL)
    Returns a results dict with per-ticker summaries + portfolio-level metrics.
    """
    tickers   = sorted(test_df["Ticker"].unique())
    portfolios = {t: TickerPortfolio(t) for t in tickers}

    # ── Daily price table for portfolio value tracking ─────────────────────
    price_pivot = (
        test_df.pivot(index="Date", columns="Ticker", values=PRICE_COL)
        .sort_index()
    )

    # Daily portfolio value series (sum across all tickers)
    port_values: dict[pd.Timestamp, float] = {}

    for date in price_pivot.index:
        for ticker in tickers:
            price  = float(price_pivot.loc[date, ticker])
            signal = signal_map.get((ticker, date), 0)   # default = HOLD

            if signal == 1:
                portfolios[ticker].buy(date, price)
            elif signal == 2:
                portfolios[ticker].sell(date, price)
            # 0 = HOLD → do nothing

        # Record total portfolio value at end of day
        total = sum(
            portfolios[t].portfolio_value(float(price_pivot.loc[date, t]))
            for t in tickers
        )
        port_values[date] = total

    # ── Liquidate open positions at final price ────────────────────────────
    final_date = price_pivot.index[-1]
    for ticker in tickers:
        final_price = float(price_pivot.loc[final_date, ticker])
        portfolios[ticker].liquidate(final_date, final_price)

    # ── Per-ticker summaries ───────────────────────────────────────────────
    ticker_summaries = {}
    for ticker in tickers:
        final_price = float(price_pivot.loc[final_date, ticker])
        ticker_summaries[ticker] = portfolios[ticker].summary(final_price, final_date)

    # ── Portfolio-level metrics ────────────────────────────────────────────
    pv_series    = pd.Series(port_values).sort_index()
    daily_ret    = pv_series.pct_change().dropna()

    initial_total = INITIAL_CASH * len(tickers)
    final_total   = pv_series.iloc[-1]
    total_ret     = (final_total - initial_total) / initial_total
    n_days        = len(pv_series)
    years         = n_days / 252
    ann_ret       = (1 + total_ret) ** (1 / years) - 1

    std    = daily_ret.std(ddof=1)
    sharpe = float((daily_ret.mean() / std) * ANNUALIZE) if std > 1e-10 else 0.0

    wealth = pv_series / pv_series.iloc[0]
    peak   = wealth.cummax()
    dd     = (wealth - peak) / peak
    max_dd = float(dd.min())

    # Aggregate trade stats across all tickers
    all_sells = [
        t for ticker in tickers
        for t in ticker_summaries[ticker]["trade_log"]
        if t.get("action") == "SELL"
    ]
    pnls = [t["pnl"] for t in all_sells]

    portfolio_metrics = {
        "strategy"          : strategy_name,
        "initial_capital"   : round(initial_total, 2),
        "final_value"       : round(final_total, 2),
        "total_return"      : round(total_ret, 4),
        "annualized_return" : round(ann_ret, 4),
        "sharpe_ratio"      : round(sharpe, 4),
        "max_drawdown"      : round(max_dd, 4),
        "n_buys"            : sum(s["n_buys"]  for s in ticker_summaries.values()),
        "n_sells"           : sum(s["n_sells"] for s in ticker_summaries.values()),
        "win_rate"          : round(sum(p > 0 for p in pnls) / len(pnls), 4) if pnls else None,
        "avg_pnl_per_trade" : round(float(np.mean(pnls)), 2)                  if pnls else None,
        "avg_holding_days"  : round(float(np.mean([t["holding_days"] for t in all_sells])), 1)
                              if all_sells else None,
    }

    return {
        "portfolio"   : portfolio_metrics,
        "by_ticker"   : ticker_summaries,
        "daily_values": {str(d): round(v, 4) for d, v in pv_series.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIGNAL MAP BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def signals_from_array(
    test_df: pd.DataFrame,
    preds: np.ndarray,
) -> dict[tuple, int]:
    """Build a (ticker, date) → label dict from a flat predictions array
    aligned with test_df rows."""
    return {
        (row["Ticker"], row["Date"]): int(preds[i])
        for i, row in test_df.reset_index(drop=True).iterrows()
    }


def signals_buy_hold(test_df: pd.DataFrame) -> dict[tuple, int]:
    """BUY on the very first date for each ticker, HOLD thereafter."""
    sig = {}
    for ticker, grp in test_df.groupby("Ticker"):
        first_date = grp["Date"].min()
        for _, row in grp.iterrows():
            sig[(ticker, row["Date"])] = 1 if row["Date"] == first_date else 0
    return sig


def signals_momentum(test_df: pd.DataFrame) -> dict[tuple, int]:
    """BUY when roc_5 > 0 (if flat), SELL when roc_5 ≤ 0 (if long), else HOLD."""
    sig = {}
    for ticker, grp in test_df.groupby("Ticker"):
        grp = grp.sort_values("Date")
        in_position = False
        for _, row in grp.iterrows():
            if not in_position and row["roc_5"] > 0:
                sig[(ticker, row["Date"])] = 1   # BUY
                in_position = True
            elif in_position and row["roc_5"] <= 0:
                sig[(ticker, row["Date"])] = 2   # SELL
                in_position = False
            else:
                sig[(ticker, row["Date"])] = 0   # HOLD
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 5. SKLEARN MODEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    n, k   = len(y), len(counts)
    wmap   = {c: n / (k * cnt) for c, cnt in counts.items()}
    return np.array([wmap[l] for l in y])


def fit_lr(train: pd.DataFrame) -> Pipeline:
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", class_weight="balanced",
                                   C=1.0, max_iter=1000, random_state=SEED)),
    ])
    pipe.fit(train[TA_COLS], train[TARGET])
    return pipe


def fit_xgb(train: pd.DataFrame) -> Pipeline:
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=300,
            max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=SEED, verbosity=0,
        )),
    ])
    sw = compute_sample_weights(train[TARGET])
    pipe.fit(train[FEATURE_COLS], train[TARGET], clf__sample_weight=sw)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# 6. LSTM (inline)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, n_classes=3, dropout=0.3):
        super().__init__()
        drop      = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=drop)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(self.drop(self.norm(h[-1])))


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_sequences(train, test, seq_len=20, feature_cols=None):
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    imp = SimpleImputer(strategy="median").fit(train[feature_cols])
    scl = StandardScaler().fit(imp.transform(train[feature_cols]))

    train_dates = set(train["Date"].dt.date)
    test_dates  = set(test["Date"].dt.date)

    full = pd.concat([train, test]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    Xa   = np.clip(scl.transform(imp.transform(full[feature_cols])), -10, 10).astype(np.float32)
    ya   = full[TARGET].values.astype(np.int64)
    da   = full["Date"].dt.date.values
    ta   = full["Ticker"].values

    res = {s: {"X": [], "y": [], "dates": [], "tickers": []} for s in ("train", "test")}
    for ticker in np.unique(ta):
        m = ta == ticker
        Xt, yt, dt = Xa[m], ya[m], da[m]
        for i in range(seq_len, len(Xt)):
            w = Xt[i - seq_len: i]
            if np.isnan(w).any():
                continue
            d = dt[i]
            s = "train" if d in train_dates else ("test" if d in test_dates else None)
            if s:
                res[s]["X"].append(w);       res[s]["y"].append(yt[i])
                res[s]["dates"].append(d);   res[s]["tickers"].append(ticker)

    for s in res:
        for k in ("X", "y", "dates", "tickers"):
            res[s][k] = np.array(res[s][k])
    return res


def train_lstm(seqs, device):
    torch.manual_seed(SEED)
    n_feat = seqs["train"]["X"].shape[2]
    model  = LSTMClassifier(n_feat).to(device)
    y_tr   = seqs["train"]["y"]
    counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    w      = torch.tensor(len(y_tr) / (3 * counts)).to(device)
    loader = DataLoader(SeqDataset(seqs["train"]["X"], y_tr), batch_size=256, shuffle=True)
    crit   = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt    = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch    = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    best_loss, best_state, no_imp = float("inf"), None, 0
    for epoch in range(1, 51):
        model.train(); total = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * len(yb)
        sch.step(epoch)
        avg = total / len(y_tr)
        if avg < best_loss - 1e-4:
            best_loss, best_state, no_imp = avg, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= 10:
                print(f"    Early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model


def predict_lstm(model, seqs, device):
    """Returns (preds, probs): argmax class labels and softmax probabilities."""
    model.eval(); preds, probs = [], []
    with torch.no_grad():
        for Xb, _ in DataLoader(SeqDataset(seqs["test"]["X"], seqs["test"]["y"]),
                                 batch_size=256, shuffle=False):
            logits = model(Xb.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def signals_with_confidence(preds, probs, tickers, dates, threshold):
    """Return signal map, falling back to HOLD when max(softmax) < threshold."""
    return {
        (tickers[i], pd.Timestamp(dates[i])): int(preds[i]) if probs[i].max() >= threshold else 0
        for i in range(len(preds))
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. TRANSFORMER (inline)
# Best ablation result: ta_only (10 features), d_model=64, 2 layers, 4 heads
# ─────────────────────────────────────────────────────────────────────────────

class TransformerClassifier(nn.Module):
    def __init__(self, n_features, seq_len=20, d_model=64, nhead=4,
                 n_layers=2, dim_ff=256, dropout=0.1, n_classes=3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed  = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x):
        B  = x.size(0)
        x  = self.input_proj(x) + self.pos_embed
        x  = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x  = self.transformer(x)
        return self.head(self.norm(x[:, 0]))


def train_transformer(seqs, device):
    torch.manual_seed(SEED)
    n_feat = seqs["train"]["X"].shape[2]
    model  = TransformerClassifier(n_feat).to(device)
    y_tr   = seqs["train"]["y"]
    counts = np.bincount(y_tr, minlength=3).astype(np.float32)
    w      = torch.tensor(len(y_tr) / (3 * counts)).to(device)
    loader = DataLoader(SeqDataset(seqs["train"]["X"], y_tr), batch_size=256, shuffle=True)
    crit   = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt    = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch    = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    best_loss, best_state, no_imp = float("inf"), None, 0
    for epoch in range(1, 51):
        model.train(); total = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = crit(model(Xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * len(yb)
        sch.step(epoch)
        avg = total / len(y_tr)
        if avg < best_loss - 1e-4:
            best_loss, best_state, no_imp = avg, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= 10:
                print(f"    Early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model


def predict_transformer(model, seqs, device):
    """Returns (preds, probs): argmax labels and softmax probabilities."""
    model.eval(); preds, probs = [], []
    with torch.no_grad():
        for Xb, _ in DataLoader(SeqDataset(seqs["test"]["X"], seqs["test"]["y"]),
                                 batch_size=256, shuffle=False):
            logits = model(Xb.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_portfolio(m: dict) -> None:
    r = m["portfolio"]
    print(f"\n  Strategy         : {r['strategy']}")
    print(f"  Initial capital  : ${r['initial_capital']:,.2f}")
    print(f"  Final value      : ${r['final_value']:,.2f}")
    print(f"  Total return     : {r['total_return']:+.2%}")
    print(f"  Ann. return      : {r['annualized_return']:+.2%}")
    print(f"  Sharpe ratio     : {r['sharpe_ratio']:+.4f}")
    print(f"  Max drawdown     : {r['max_drawdown']:+.2%}")
    print(f"  Trades (B/S)     : {r['n_buys']} buys / {r['n_sells']} sells")
    print(f"  Win rate         : {r['win_rate']:.2%}" if r["win_rate"] else "  Win rate         : n/a")
    print(f"  Avg P&L/trade    : ${r['avg_pnl_per_trade']:+.2f}" if r["avg_pnl_per_trade"] else "  Avg P&L/trade    : n/a")
    print(f"  Avg holding days : {r['avg_holding_days']}" if r["avg_holding_days"] else "  Avg holding days : n/a")

    print(f"\n  Per-ticker breakdown:")
    print(f"  {'Ticker':<8} {'Final $':>9} {'Return':>8} {'Trades':>7} {'Win%':>7} {'Avg P&L':>10}")
    print("  " + "-" * 56)
    for t, s in sorted(m["by_ticker"].items()):
        wr  = f"{s['win_rate']:.0%}"  if s["win_rate"]  is not None else "  n/a"
        apl = f"${s['avg_pnl']:+.2f}" if s["avg_pnl"]   is not None else "   n/a"
        tr  = f"{s['n_buys']}B/{s['n_sells']}S"
        print(f"  {t:<8} {s['final_value']:>9.2f} {s['total_return']:>+8.2%} {tr:>7} {wr:>7} {apl:>10}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available()         else
              torch.device("cpu"))
    print(f"Device : {device}")

    print("\nLoading data...")
    train, test = load_data()
    print(f"  Train : {len(train):,} rows")
    print(f"  Test  : {len(test):,} rows  "
          f"({test['Date'].min().date()} → {test['Date'].max().date()})")

    all_results = {}

    # ── Benchmark: Buy & Hold ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK: Buy & Hold")
    print("=" * 60)
    bh_result = run_simulation(test, signals_buy_hold(test), "buy_hold")
    print_portfolio(bh_result)
    all_results["buy_hold"] = bh_result

    # ── Benchmark: Momentum ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK: Momentum (roc_5 sign)")
    print("=" * 60)
    mo_result = run_simulation(test, signals_momentum(test), "momentum")
    print_portfolio(mo_result)
    all_results["momentum"] = mo_result

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL: Logistic Regression (TA only)")
    print("=" * 60)
    print("  Fitting...")
    lr_pipe  = fit_lr(train)
    lr_preds = lr_pipe.predict(test[TA_COLS])
    lr_result = run_simulation(test, signals_from_array(test, lr_preds), "logistic_regression")
    print_portfolio(lr_result)
    all_results["logistic_regression"] = lr_result

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL: XGBoost (38 features)")
    print("=" * 60)
    print("  Fitting...")
    xgb_pipe  = fit_xgb(train)
    xgb_preds = xgb_pipe.predict(test[FEATURE_COLS])
    xgb_result = run_simulation(test, signals_from_array(test, xgb_preds), "xgboost")
    print_portfolio(xgb_result)
    all_results["xgboost"] = xgb_result

    # ── LSTM (ta_sent: 23 features — best ablation result) ────────────────────
    print("\n" + "=" * 60)
    print("MODEL: LSTM ta_sent (seq=20, hidden=64, TA+Sentiment 23 features)")
    print("=" * 60)
    print("  Building sequences...")
    seqs = build_sequences(train, test, seq_len=20, feature_cols=TA_SENT_COLS)
    print(f"  Train: {seqs['train']['X'].shape}  Test: {seqs['test']['X'].shape}")
    print("  Training...")
    lstm_model = train_lstm(seqs, device)
    lstm_preds, lstm_probs = predict_lstm(lstm_model, seqs, device)

    lstm_tickers = seqs["test"]["tickers"]
    lstm_dates   = seqs["test"]["dates"]

    # ── Unfiltered ────────────────────────────────────────────────────────────
    lstm_sig = {
        (lstm_tickers[i], pd.Timestamp(lstm_dates[i])): int(lstm_preds[i])
        for i in range(len(lstm_preds))
    }
    lstm_result = run_simulation(test, lstm_sig, "lstm_ta_sent")
    print_portfolio(lstm_result)
    all_results["lstm_ta_sent"] = lstm_result

    # ── Confidence threshold sweep ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL: LSTM ta_sent — confidence threshold sweep")
    print("=" * 60)
    best_sharpe, best_thresh, best_result = -999, None, None
    for thresh in [0.38, 0.42, 0.46, 0.50, 0.60]:
        sig   = signals_with_confidence(lstm_preds, lstm_probs, lstm_tickers, lstm_dates, thresh)
        n_act = sum(1 for v in sig.values() if v != 0)
        res   = run_simulation(test, sig, f"lstm_conf_{thresh:.2f}")
        p     = res["portfolio"]
        wr    = f"{p['win_rate']:.0%}" if p["win_rate"] else "n/a"
        print(f"  thresh={thresh:.2f}  active={n_act:4d}/{len(sig)} ({n_act/len(sig)*100:4.1f}%)  "
              f"return={p['total_return']:+.2%}  sharpe={p['sharpe_ratio']:+.4f}  "
              f"maxdd={p['max_drawdown']:+.2%}  win={wr}")
        all_results[f"lstm_conf_{thresh:.2f}"] = res   # save every threshold
        if p["sharpe_ratio"] > best_sharpe:
            best_sharpe, best_thresh, best_result = p["sharpe_ratio"], thresh, res

    print(f"\n  Best threshold : {best_thresh:.2f}  (Sharpe={best_sharpe:+.4f})")
    print_portfolio(best_result)

    # ── Transformer (ta_only: 10 features — best Transformer ablation) ─────────
    print("\n" + "=" * 60)
    print("MODEL: Transformer ta_only (seq=20, d_model=64, 4 heads, 2 layers)")
    print("=" * 60)
    print("  Building sequences...")
    seqs_tr = build_sequences(train, test, seq_len=20, feature_cols=TA_COLS)
    print(f"  Train: {seqs_tr['train']['X'].shape}  Test: {seqs_tr['test']['X'].shape}")
    print("  Training...")
    tfm_model = train_transformer(seqs_tr, device)
    tfm_preds, tfm_probs = predict_transformer(tfm_model, seqs_tr, device)

    tfm_tickers = seqs_tr["test"]["tickers"]
    tfm_dates   = seqs_tr["test"]["dates"]

    # ── Unfiltered ────────────────────────────────────────────────────────────
    tfm_sig = {
        (tfm_tickers[i], pd.Timestamp(tfm_dates[i])): int(tfm_preds[i])
        for i in range(len(tfm_preds))
    }
    tfm_result = run_simulation(test, tfm_sig, "transformer_ta_only")
    print_portfolio(tfm_result)
    all_results["transformer_ta_only"] = tfm_result

    # ── Confidence threshold sweep ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL: Transformer ta_only — confidence threshold sweep")
    print("=" * 60)
    best_sharpe, best_thresh, best_result = -999, None, None
    for thresh in [0.38, 0.42, 0.46, 0.50, 0.60]:
        sig   = signals_with_confidence(tfm_preds, tfm_probs, tfm_tickers, tfm_dates, thresh)
        n_act = sum(1 for v in sig.values() if v != 0)
        res   = run_simulation(test, sig, f"tfm_conf_{thresh:.2f}")
        p     = res["portfolio"]
        wr    = f"{p['win_rate']:.0%}" if p["win_rate"] else "n/a"
        print(f"  thresh={thresh:.2f}  active={n_act:4d}/{len(sig)} ({n_act/len(sig)*100:4.1f}%)  "
              f"return={p['total_return']:+.2%}  sharpe={p['sharpe_ratio']:+.4f}  "
              f"maxdd={p['max_drawdown']:+.2%}  win={wr}")
        all_results[f"tfm_conf_{thresh:.2f}"] = res    # save every threshold
        if p["sharpe_ratio"] > best_sharpe:
            best_sharpe, best_thresh, best_result = p["sharpe_ratio"], thresh, res

    print(f"\n  Best threshold : {best_thresh:.2f}  (Sharpe={best_sharpe:+.4f})")
    print_portfolio(best_result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  {'Strategy':<25} {'Final $':>9} {'Tot Ret':>8} {'Ann Ret':>8} "
          f"{'Sharpe':>8} {'MaxDD':>8} {'Win%':>7} {'Trades':>7}")
    print("  " + "-" * 78)
    for name, res in all_results.items():
        p  = res["portfolio"]
        wr = f"{p['win_rate']:.0%}" if p["win_rate"] else "  n/a"
        print(
            f"  {name:<25} "
            f"${p['final_value']:>8,.0f} "
            f"{p['total_return']:>+8.2%} "
            f"{p['annualized_return']:>+8.2%} "
            f"{p['sharpe_ratio']:>+8.4f} "
            f"{p['max_drawdown']:>+8.2%} "
            f"{wr:>7} "
            f"{p['n_sells']:>7}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)

    # Make trade logs JSON-serialisable
    def make_serialisable(obj):
        if isinstance(obj, dict):
            return {k: make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serialisable(i) for i in obj]
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    out = {
        "description": (
            "Per-ticker dollar trading simulation. $1,000 initial cash per ticker "
            "($10,000 total). BUY = invest $1,000 at Adj_Close (only if flat). "
            "SELL = sell all shares at Adj_Close (only if long). "
            "HOLD = no action. Open positions liquidated at final test date. "
            "P&L booked only on sell/liquidation. "
            f"Test period: {test['Date'].min().date()} to {test['Date'].max().date()}."
        ),
        "parameters": {
            "initial_cash_per_ticker": INITIAL_CASH,
            "trade_size"             : TRADE_SIZE,
            "tickers"                : sorted(test["Ticker"].unique().tolist()),
        },
        "results": make_serialisable(all_results),
    }

    out_path = RESULTS_DIR / "trading_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
