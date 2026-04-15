from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import model_training as mt


RESULTS_DIR = Path("results")
TRADING_DAYS_PER_YEAR = 252


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a saved model on its test split.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to one saved .pkl model artifact from run_models.py",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forward return horizon in trading days, default=1",
    )
    return parser.parse_args()


def compute_forward_returns(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create forward compounded returns from the daily return column."""
    if "ret_1d" not in df.columns:
        raise KeyError("Backtest requires a 'ret_1d' column in the dataset.")

    returns_df = df[[mt.TICKER_COLUMN, mt.DATE_COLUMN, "ret_1d"]].copy()
    returns_df[mt.DATE_COLUMN] = pd.to_datetime(returns_df[mt.DATE_COLUMN])
    returns_df = returns_df.sort_values([mt.TICKER_COLUMN, mt.DATE_COLUMN]).reset_index(drop=True)

    forward_column = f"forward_return_{horizon}d"
    forward_returns = np.full(len(returns_df), np.nan, dtype=np.float64)

    for _, index_values in returns_df.groupby(mt.TICKER_COLUMN).groups.items():
        idx = np.asarray(list(index_values), dtype=int)
        daily_returns = returns_df.loc[idx, "ret_1d"].astype(float).to_numpy()
        for start in range(len(daily_returns) - horizon):
            future_slice = daily_returns[start + 1:start + 1 + horizon]
            forward_returns[idx[start]] = np.prod(1.0 + future_slice) - 1.0

    returns_df[forward_column] = forward_returns
    return returns_df


def attach_realized_returns(
    prediction_df: pd.DataFrame,
    artifact: dict[str, Any],
    horizon: int,
) -> pd.DataFrame:
    """Merge row-level predictions with realized forward returns."""
    dataset_path = mt.resolve_dataset_path(artifact["data_set"])
    base_df = mt.load_data(file_path=dataset_path)
    returns_df = compute_forward_returns(base_df, horizon=horizon)
    merged = prediction_df.copy()
    merged[mt.DATE_COLUMN] = pd.to_datetime(merged[mt.DATE_COLUMN])
    merged = merged.merge(
        returns_df,
        on=[mt.TICKER_COLUMN, mt.DATE_COLUMN],
        how="left",
    )
    return merged.dropna(subset=[f"forward_return_{horizon}d"]).reset_index(drop=True)


def build_strategy_frame(prediction_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Turn test predictions into an equal-weight long/short daily strategy."""
    forward_column = f"forward_return_{horizon}d"
    strategy_df = prediction_df.copy()
    strategy_df["position"] = np.where(
        strategy_df["y_pred"] == 1,
        1.0,
        np.where(strategy_df["y_pred"] == -1, -1.0, 0.0),
    )

    daily_rows: list[dict[str, Any]] = []
    for date, day_df in strategy_df.groupby(mt.DATE_COLUMN):
        long_bucket = day_df[day_df["position"] > 0]
        short_bucket = day_df[day_df["position"] < 0]

        long_return = long_bucket[forward_column].mean() if len(long_bucket) > 0 else 0.0
        short_return = -short_bucket[forward_column].mean() if len(short_bucket) > 0 else 0.0

        active_sides = int(len(long_bucket) > 0) + int(len(short_bucket) > 0)
        if active_sides == 0:
            strategy_return = 0.0
        elif active_sides == 2:
            strategy_return = 0.5 * (long_return + short_return)
        else:
            strategy_return = long_return + short_return

        daily_rows.append(
            {
                "date": pd.Timestamp(date),
                "strategy_return": strategy_return,
                "long_count": int(len(long_bucket)),
                "short_count": int(len(short_bucket)),
                "avg_signal_score": float(day_df.get("signal_score", pd.Series([0.0])).mean()),
            }
        )

    daily_df = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    daily_df["equity_curve"] = (1.0 + daily_df["strategy_return"]).cumprod()
    daily_df["drawdown"] = daily_df["equity_curve"] / daily_df["equity_curve"].cummax() - 1.0
    return daily_df


def build_inventory_strategy_frame(
    prediction_df: pd.DataFrame,
    horizon: int,
    unit_size: float = 1.0,
    max_position_units: float = 3.0,
) -> pd.DataFrame:
    """Stateful strategy where buy/sell change inventory by fixed units."""
    forward_column = f"forward_return_{horizon}d"
    strategy_df = prediction_df.copy()
    strategy_df = strategy_df.sort_values([mt.TICKER_COLUMN, mt.DATE_COLUMN]).reset_index(drop=True)

    position_units = np.zeros(len(strategy_df), dtype=np.float64)
    scaled_position = np.zeros(len(strategy_df), dtype=np.float64)

    for _, index_values in strategy_df.groupby(mt.TICKER_COLUMN).groups.items():
        idx = np.asarray(list(index_values), dtype=int)
        current_position = 0.0

        for row_index in idx:
            signal = strategy_df.at[row_index, "y_pred"]
            if signal == 1:
                current_position = min(current_position + unit_size, max_position_units)
            elif signal == -1:
                current_position = max(current_position - unit_size, -max_position_units)

            position_units[row_index] = current_position
            scaled_position[row_index] = current_position / max_position_units

    strategy_df["position_units"] = position_units
    strategy_df["position_weight"] = scaled_position
    strategy_df["stock_return_contribution"] = (
        strategy_df["position_weight"] * strategy_df[forward_column]
    )

    daily_df = (
        strategy_df.groupby(mt.DATE_COLUMN, as_index=False)
        .agg(
            strategy_return=("stock_return_contribution", "mean"),
            long_count=("position_units", lambda s: int((s > 0).sum())),
            short_count=("position_units", lambda s: int((s < 0).sum())),
            avg_signal_score=("signal_score", "mean"),
            avg_gross_exposure=("position_weight", lambda s: float(np.abs(s).mean())),
            avg_net_exposure=("position_weight", "mean"),
        )
        .rename(columns={mt.DATE_COLUMN: "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily_df["equity_curve"] = (1.0 + daily_df["strategy_return"]).cumprod()
    daily_df["drawdown"] = daily_df["equity_curve"] / daily_df["equity_curve"].cummax() - 1.0
    return daily_df


def compute_strategy_metrics(daily_df: pd.DataFrame) -> dict[str, Any]:
    """Compute headline strategy metrics from the daily strategy returns."""
    returns = daily_df["strategy_return"].astype(float)
    num_days = len(returns)

    if num_days == 0:
        raise ValueError("No test observations remained after merging realized returns.")

    total_return = float(daily_df["equity_curve"].iloc[-1] - 1.0)
    avg_daily_return = float(returns.mean())
    daily_vol = float(returns.std(ddof=0))
    annualized_return = float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / num_days) - 1.0)
    annualized_volatility = float(daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe_ratio = float(
        avg_daily_return / daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    ) if daily_vol > 0 else float("nan")

    return {
        "num_days": num_days,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": float(daily_df["drawdown"].min()),
        "win_rate": float((returns > 0).mean()),
        "avg_daily_return": avg_daily_return,
        "avg_long_count": float(daily_df["long_count"].mean()),
        "avg_short_count": float(daily_df["short_count"].mean()),
    }


def build_equal_weight_baseline(
    artifact: dict[str, Any],
    horizon: int = 1,
) -> dict[str, Any]:
    """Build an equal-weight benchmark that holds all stocks in the test split."""
    dataset_path = mt.resolve_dataset_path(artifact["data_set"])
    data_bundle = mt.load_data_bundle(dataset_path)
    base_df = mt.load_data(file_path=dataset_path)
    returns_df = compute_forward_returns(base_df, horizon=horizon)

    test_dates = pd.to_datetime(data_bundle["dates_test"]).drop_duplicates().sort_values()
    forward_column = f"forward_return_{horizon}d"

    benchmark_rows = (
        returns_df[returns_df[mt.DATE_COLUMN].isin(test_dates)]
        .dropna(subset=[forward_column])
        .groupby(mt.DATE_COLUMN, as_index=False)
        .agg(
            strategy_return=(forward_column, "mean"),
            long_count=(mt.TICKER_COLUMN, "count"),
        )
    )

    benchmark_rows["short_count"] = 0
    benchmark_rows["avg_signal_score"] = 0.0
    benchmark_rows = benchmark_rows.sort_values(mt.DATE_COLUMN).reset_index(drop=True)
    benchmark_rows["equity_curve"] = (1.0 + benchmark_rows["strategy_return"]).cumprod()
    benchmark_rows["drawdown"] = (
        benchmark_rows["equity_curve"] / benchmark_rows["equity_curve"].cummax() - 1.0
    )

    daily_df = benchmark_rows.rename(columns={mt.DATE_COLUMN: "date"})
    metrics = compute_strategy_metrics(daily_df)
    return {
        "artifact": artifact,
        "prediction_df": None,
        "daily_df": daily_df,
        "metrics": metrics,
    }


def run_backtest_from_model_path(
    model_path: Path | str,
    horizon: int = 1,
) -> dict[str, Any]:
    """Run the full backtest workflow and return in-memory objects for notebooks."""
    model_path = Path(model_path)
    artifact = mt.load_model_artifact(model_path)
    prediction_df = mt.predict_test_split_from_artifact(artifact)
    prediction_df = attach_realized_returns(prediction_df, artifact, horizon=horizon)
    daily_df = build_strategy_frame(prediction_df, horizon=horizon)
    metrics = compute_strategy_metrics(daily_df)
    return {
        "model_path": model_path,
        "artifact": artifact,
        "prediction_df": prediction_df,
        "daily_df": daily_df,
        "metrics": metrics,
    }


def run_inventory_backtest_from_model_path(
    model_path: Path | str,
    horizon: int = 1,
    unit_size: float = 1.0,
    max_position_units: float = 3.0,
) -> dict[str, Any]:
    """Run a stateful inventory-style backtest from one saved model artifact."""
    model_path = Path(model_path)
    artifact = mt.load_model_artifact(model_path)
    prediction_df = mt.predict_test_split_from_artifact(artifact)
    prediction_df = attach_realized_returns(prediction_df, artifact, horizon=horizon)
    daily_df = build_inventory_strategy_frame(
        prediction_df,
        horizon=horizon,
        unit_size=unit_size,
        max_position_units=max_position_units,
    )
    metrics = compute_strategy_metrics(daily_df)
    metrics["unit_size"] = unit_size
    metrics["max_position_units"] = max_position_units
    return {
        "model_path": model_path,
        "artifact": artifact,
        "prediction_df": prediction_df,
        "daily_df": daily_df,
        "metrics": metrics,
    }


def build_backtest_summary(backtest_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a compact comparison table from several backtest results."""
    rows: list[dict[str, Any]] = []
    for model_name, result in backtest_results.items():
        row = {"model": model_name}
        row.update(result["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_backtest_equity_curves(backtest_results: dict[str, dict[str, Any]]) -> None:
    """Plot cumulative equity curves for several backtests."""
    plt.figure(figsize=(10, 5))
    for model_name, result in backtest_results.items():
        daily_df = result["daily_df"]
        plt.plot(daily_df["date"], daily_df["equity_curve"], label=model_name, linewidth=2)
    plt.title("Test-Set Strategy Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_backtest_metric_bars(summary_df: pd.DataFrame) -> None:
    """Plot headline backtest metrics for visual comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    summary_df.plot(
        x="model",
        y=["annualized_return", "sharpe_ratio"],
        kind="bar",
        ax=axes[0],
        title="Return and Sharpe",
    )
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Value")
    axes[0].grid(axis="y", alpha=0.3)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    summary_df.plot(
        x="model",
        y=["max_drawdown", "win_rate"],
        kind="bar",
        ax=axes[1],
        title="Drawdown and Win Rate",
    )
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("Value")
    axes[1].grid(axis="y", alpha=0.3)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.show()


def save_backtest_outputs(
    model_path: Path,
    prediction_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    metrics: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Persist backtest tables and metrics."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = model_path.stem

    predictions_path = RESULTS_DIR / f"{stem}_test_predictions_{timestamp}.csv"
    daily_path = RESULTS_DIR / f"{stem}_strategy_daily_{timestamp}.csv"
    metrics_path = RESULTS_DIR / f"{stem}_strategy_metrics_{timestamp}.json"

    prediction_df.to_csv(predictions_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)

    return predictions_path, daily_path, metrics_path


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    artifact = mt.load_model_artifact(model_path)
    prediction_df = mt.predict_test_split_from_artifact(artifact)
    prediction_df = attach_realized_returns(prediction_df, artifact, horizon=args.horizon)
    daily_df = build_strategy_frame(prediction_df, horizon=args.horizon)
    metrics = compute_strategy_metrics(daily_df)
    predictions_path, daily_path, metrics_path = save_backtest_outputs(
        model_path=model_path,
        prediction_df=prediction_df,
        daily_df=daily_df,
        metrics=metrics,
    )

    print(f"Model artifact: {model_path}")
    print(f"Dataset: {artifact['data_set']}")
    print(f"Backtest horizon: {args.horizon}d")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved daily strategy returns to {daily_path}")
    print(f"Saved strategy metrics to {metrics_path}")


if __name__ == "__main__":
    main()
