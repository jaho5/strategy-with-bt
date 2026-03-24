"""Generate equity curve plots for winning strategies and save as PNGs.

Runs the 4 winning configurations plus an equal-weight benchmark,
then produces four publication-quality charts:

  1. Equity curves (all overlaid, log scale)
  2. Drawdown chart (underwater equity for InvVol Top-4)
  3. Rolling 252-day Sharpe for top 3 strategies
  4. Monthly returns heatmap for InvVol Ensemble (1.5x leverage)

Usage:
    uv run python -m src.utils.plot_equity
"""

from __future__ import annotations

import importlib
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.backtest.engine import BacktestEngine
from src.data.downloader import SECTOR_ETFS, download_etf_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_START = "2010-01-01"
DATA_END = "2025-12-31"
TRAIN_FRACTION = 0.70
INV_VOL_LOOKBACK = 63
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

# 5 component strategies for the ensemble (ordered by prior performance)
ENSEMBLE_STRATEGIES: List[Tuple[str, str, str, dict]] = [
    (
        "Entropy Regularized",
        "src.strategies.entropy_regularized",
        "EntropyRegularizedStrategy",
        {},
    ),
    (
        "GARCH Vol",
        "src.strategies.garch_vol",
        "GarchVolStrategy",
        {},
    ),
    (
        "HMM Regime",
        "src.strategies.hmm_regime",
        "HMMRegimeStrategy",
        {},
    ),
    (
        "Spectral Momentum",
        "src.strategies.spectral_momentum",
        "SpectralMomentumStrategy",
        {},
    ),
    (
        "Bayesian Changepoint",
        "src.strategies.bayesian_changepoint",
        "BayesianChangepointStrategy",
        {},
    ),
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _extract_close_prices(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Extract close prices from multi-level yfinance DataFrame."""
    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Close" in raw_data.columns.get_level_values(0):
            close = raw_data["Close"]
        else:
            first_level = raw_data.columns.get_level_values(0)[0]
            close = raw_data[first_level]
    else:
        close = raw_data
    close = close.dropna(how="all").ffill().bfill()
    return close


def _split_data(
    data: pd.DataFrame, train_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test by row count."""
    n = len(data)
    split_idx = int(n * train_frac)
    return data.iloc[:split_idx], data.iloc[split_idx:]


# ---------------------------------------------------------------------------
# Signal -> Portfolio return conversion
# ---------------------------------------------------------------------------

def _signals_to_portfolio_return(
    signals: pd.DataFrame, prices: pd.DataFrame,
) -> pd.Series:
    """Convert a multi-asset signal DataFrame to daily portfolio returns."""
    common_idx = signals.index.intersection(prices.index)
    if len(common_idx) == 0:
        common_idx = signals.index

    price_tickers = list(prices.columns)

    direct_match = [c for c in signals.columns if c in price_tickers]
    signal_weight_match = {
        t: f"{t}_signal" for t in price_tickers
        if f"{t}_signal" in signals.columns
    }
    has_single = "signal" in signals.columns

    returns = prices.reindex(common_idx).ffill().bfill().pct_change().fillna(0.0)

    if direct_match:
        sig_df = signals[direct_match].reindex(common_idx).fillna(0.0)
        port_ret = (sig_df.shift(1).fillna(0.0) * returns[direct_match]).sum(axis=1)

    elif signal_weight_match:
        sig_df = pd.DataFrame(index=common_idx)
        for ticker, sig_col in signal_weight_match.items():
            wgt_col = f"{ticker}_weight"
            if wgt_col in signals.columns:
                sig_df[ticker] = (
                    signals[sig_col].reindex(common_idx).fillna(0.0)
                    * signals[wgt_col].reindex(common_idx).fillna(0.0)
                )
            else:
                sig_df[ticker] = signals[sig_col].reindex(common_idx).fillna(0.0)

        tickers_used = list(sig_df.columns)
        port_ret = (sig_df.shift(1).fillna(0.0) * returns[tickers_used]).sum(axis=1)

    elif has_single:
        sig_arr = signals["signal"].reindex(common_idx).fillna(0.0)
        wgt_arr = signals.get("weight", pd.Series(1.0, index=common_idx))
        wgt_arr = wgt_arr.reindex(common_idx).fillna(1.0)
        composite = sig_arr * wgt_arr
        avg_return = returns.mean(axis=1)
        port_ret = composite.shift(1).fillna(0.0) * avg_return

    else:
        sig_arr = signals.reindex(common_idx).fillna(0.0).select_dtypes(
            include="number"
        ).mean(axis=1)
        avg_return = returns.mean(axis=1)
        port_ret = sig_arr.shift(1).fillna(0.0) * avg_return

    return port_ret


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _load_strategy(
    module_path: str, class_name: str, kwargs: dict,
) -> Any:
    """Import and instantiate a single strategy."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def _run_strategy_oos(
    name: str,
    strategy: Any,
    close_prices: pd.DataFrame,
) -> Optional[pd.Series]:
    """Fit on training data, generate signals on test data, return OOS daily returns."""
    train_data, test_data = _split_data(close_prices, TRAIN_FRACTION)

    logger.info("  [%s] Fitting on training data...", name)
    strategy.fit(train_data)

    logger.info("  [%s] Generating signals on test data (OOS)...", name)
    signals = strategy.generate_signals(test_data)

    port_ret = _signals_to_portfolio_return(signals, test_data)

    cum_pnl = (np.prod(1.0 + port_ret.values) - 1.0) * 100
    logger.info(
        "  [%s] OOS return series: %d bars, cumulative=%.2f%%",
        name, len(port_ret), cum_pnl,
    )
    return port_ret


# ---------------------------------------------------------------------------
# Inverse-volatility weighted ensemble
# ---------------------------------------------------------------------------

def _inverse_vol_combine(
    strategy_returns: Dict[str, pd.Series],
    lookback: int = INV_VOL_LOOKBACK,
) -> pd.Series:
    """Inverse-volatility weighted combination of strategy return series."""
    ret_df = pd.DataFrame(strategy_returns).fillna(0.0)
    names = list(ret_df.columns)
    K = len(names)
    T = len(ret_df)

    combined = pd.Series(0.0, index=ret_df.index)

    for t in range(T):
        if t < lookback:
            w = np.full(K, 1.0 / K)
        else:
            window = ret_df.iloc[t - lookback:t]
            vol = window.std().values
            vol = np.where(vol < 1e-10, 1e-10, vol)
            inv_vol = 1.0 / vol
            w = inv_vol / inv_vol.sum()

        combined.iloc[t] = float(ret_df.iloc[t].values @ w)

    return combined


# ---------------------------------------------------------------------------
# Build equity curves from return series
# ---------------------------------------------------------------------------

def _returns_to_equity(returns: pd.Series, initial: float = 100.0) -> pd.Series:
    """Convert daily returns to an equity curve starting at *initial*."""
    return initial * (1.0 + returns).cumprod()


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute the drawdown series (negative values, 0 = at peak)."""
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    return dd


def _rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling annualised Sharpe ratio."""
    rolling_mean = returns.rolling(window=window, min_periods=window).mean()
    rolling_std = returns.rolling(window=window, min_periods=window).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return (rolling_mean / rolling_std) * np.sqrt(252)


# ---------------------------------------------------------------------------
# Run all 4 strategies + benchmark
# ---------------------------------------------------------------------------

def _run_all_strategies(
    close_prices: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """Run the 4 winning configurations and the equal-weight benchmark.

    Returns a dict mapping strategy label to OOS daily return series.
    """
    _, test_data = _split_data(close_prices, TRAIN_FRACTION)
    results: Dict[str, pd.Series] = {}

    # ------------------------------------------------------------------
    # 1. Run all 5 component strategies for ensemble construction
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Running 5 component strategies for ensemble construction")
    logger.info("=" * 70)

    component_returns: Dict[str, pd.Series] = {}
    component_order: List[str] = []

    for display_name, module_path, class_name, kwargs in ENSEMBLE_STRATEGIES:
        try:
            strategy = _load_strategy(module_path, class_name, kwargs)
            port_ret = _run_strategy_oos(display_name, strategy, close_prices)
            if port_ret is not None and len(port_ret) > 0:
                component_returns[display_name] = port_ret
                component_order.append(display_name)
        except Exception as exc:
            logger.warning("  Strategy '%s' FAILED: %s", display_name, exc)

    if len(component_returns) < 2:
        logger.error("Fewer than 2 component strategies succeeded.")
        return results

    # Align all return series to a common index
    ret_df = pd.DataFrame(component_returns).sort_index()
    common_idx = ret_df.dropna(how="all").index
    component_returns = {
        name: series.reindex(common_idx).fillna(0.0)
        for name, series in component_returns.items()
    }

    logger.info(
        "Common OOS period: %s to %s (%d bars)",
        common_idx[0].date(), common_idx[-1].date(), len(common_idx),
    )

    # ------------------------------------------------------------------
    # 2. InvVol Ensemble Top-4 (1x leverage)
    # ------------------------------------------------------------------
    available = [n for n in component_order if n in component_returns]
    top4_names = available[:4]
    if len(top4_names) >= 2:
        top4_rets = {n: component_returns[n] for n in top4_names}
        invvol_top4 = _inverse_vol_combine(top4_rets)
        results["InvVol Ensemble Top-4 (1x)"] = invvol_top4
        logger.info(
            "InvVol Top-4 cumulative PnL: %.2f%%",
            (np.prod(1.0 + invvol_top4.values) - 1.0) * 100,
        )

    # ------------------------------------------------------------------
    # 3. InvVol Ensemble (1.5x leverage) -- all 5 components
    # ------------------------------------------------------------------
    if len(component_returns) >= 2:
        invvol_base = _inverse_vol_combine(component_returns)
        invvol_15x = invvol_base * 1.5
        results["InvVol Ensemble (1.5x)"] = invvol_15x
        logger.info(
            "InvVol 1.5x cumulative PnL: %.2f%%",
            (np.prod(1.0 + invvol_15x.values) - 1.0) * 100,
        )

    # ------------------------------------------------------------------
    # 4. GARCH Vol standalone (EGARCH, target_vol=0.25, adaptive_blend)
    # ------------------------------------------------------------------
    logger.info("Running GARCH Vol standalone...")
    try:
        from src.strategies.garch_vol import GarchVolConfig, GarchVolStrategy

        cfg = GarchVolConfig(
            garch_model="EGARCH",
            target_vol=0.25,
            adaptive_blend=True,
        )
        garch_strat = GarchVolStrategy(config=cfg)
        garch_ret = _run_strategy_oos("GARCH Vol", garch_strat, close_prices)
        if garch_ret is not None and len(garch_ret) > 0:
            garch_ret = garch_ret.reindex(common_idx).fillna(0.0)
            results["GARCH Vol"] = garch_ret
    except Exception as exc:
        logger.warning("GARCH Vol standalone failed: %s", exc)

    # ------------------------------------------------------------------
    # 5. Equal-weight benchmark (buy-and-hold all 18 ETFs)
    # ------------------------------------------------------------------
    logger.info("Computing equal-weight benchmark (buy-and-hold 18 ETFs)...")
    test_returns = test_data.pct_change().fillna(0.0)
    benchmark_ret = test_returns.mean(axis=1)
    benchmark_ret = benchmark_ret.reindex(common_idx).fillna(0.0)
    results["Equal-Weight Benchmark"] = benchmark_ret
    logger.info(
        "Benchmark cumulative PnL: %.2f%%",
        (np.prod(1.0 + benchmark_ret.values) - 1.0) * 100,
    )

    return results


# ---------------------------------------------------------------------------
# Plot 1: Equity Curves
# ---------------------------------------------------------------------------

def plot_equity_curves(
    return_series: Dict[str, pd.Series],
    save_path: Path,
) -> None:
    """Plot all strategy equity curves overlaid on a log-scale y-axis."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {
        "InvVol Ensemble Top-4 (1x)": "#1f77b4",
        "InvVol Ensemble (1.5x)": "#ff7f0e",
        "GARCH Vol": "#2ca02c",
        "Equal-Weight Benchmark": "#7f7f7f",
    }

    for label, rets in return_series.items():
        equity = _returns_to_equity(rets, initial=100.0)
        final_pnl = (equity.iloc[-1] / 100.0 - 1.0) * 100
        color = colors.get(label, None)
        ax.plot(
            equity.index, equity.values,
            label=f"{label} ({final_pnl:+.1f}%)",
            color=color,
            linewidth=1.5 if "Benchmark" not in label else 1.2,
            linestyle="-" if "Benchmark" not in label else "--",
        )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_title("Equity Curves: Winning Strategies vs Benchmark", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (log scale, start=100)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved equity curves to %s", save_path)


# ---------------------------------------------------------------------------
# Plot 2: Drawdown Chart (InvVol Top-4)
# ---------------------------------------------------------------------------

def plot_drawdown(
    returns: pd.Series,
    label: str,
    save_path: Path,
) -> None:
    """Plot underwater equity (drawdown) for a single strategy."""
    equity = _returns_to_equity(returns, initial=100.0)
    dd = _drawdown_series(equity)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5)
    ax.plot(dd.index, dd.values, color="#d62728", linewidth=0.8)
    ax.set_title(f"Drawdown Chart: {label}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(top=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved drawdown chart to %s", save_path)


# ---------------------------------------------------------------------------
# Plot 3: Rolling Sharpe
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(
    return_series: Dict[str, pd.Series],
    save_path: Path,
    window: int = 252,
) -> None:
    """Plot 252-day rolling Sharpe for the top 3 strategies."""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {
        "InvVol Ensemble Top-4 (1x)": "#1f77b4",
        "InvVol Ensemble (1.5x)": "#ff7f0e",
        "GARCH Vol": "#2ca02c",
    }

    for label, rets in return_series.items():
        if label == "Equal-Weight Benchmark":
            continue
        rs = _rolling_sharpe(rets, window=window)
        color = colors.get(label, None)
        ax.plot(
            rs.index, rs.values,
            label=label,
            color=color,
            linewidth=1.3,
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(y=1, color="green", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Sharpe = 1")
    ax.set_title(f"Rolling {window}-Day Sharpe Ratio: Top 3 Strategies", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio (annualised)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved rolling Sharpe chart to %s", save_path)


# ---------------------------------------------------------------------------
# Plot 4: Monthly Returns Heatmap (InvVol Ensemble)
# ---------------------------------------------------------------------------

def plot_monthly_heatmap(
    returns: pd.Series,
    label: str,
    save_path: Path,
) -> None:
    """Plot a monthly returns heatmap (year x month) for a strategy."""
    # Aggregate daily returns to monthly
    monthly = (1.0 + returns).resample("ME").prod() - 1.0

    # Build a year x month pivot table
    monthly_df = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    pivot = monthly_df.pivot_table(
        index="Year", columns="Month", values="Return", aggfunc="sum",
    )
    # Label months
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    pivot.columns = [month_labels[m - 1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))

    # Custom diverging colormap: red for negative, green for positive
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)
    vmax = max(abs(pivot.max().max()), abs(pivot.min().min()))
    vmax = max(vmax, 0.01)  # avoid zero range

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Monthly Return"},
    )
    ax.set_title(f"Monthly Returns Heatmap: {label}", fontsize=14)
    ax.set_ylabel("Year")
    ax.set_xlabel("Month")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved monthly heatmap to %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: download data, run strategies, generate all plots."""

    # ---- Download data ----
    logger.info("Downloading ETF data from %s to %s...", DATA_START, DATA_END)
    raw_data = download_etf_data(start=DATA_START, end=DATA_END)
    close_prices = _extract_close_prices(raw_data)
    logger.info(
        "Close prices: %d rows x %d tickers (%s to %s)",
        len(close_prices), len(close_prices.columns),
        close_prices.index[0].date(), close_prices.index[-1].date(),
    )

    # ---- Run all strategies ----
    return_series = _run_all_strategies(close_prices)

    if not return_series:
        logger.error("No strategies produced returns. Cannot generate plots.")
        return

    # ---- Ensure output directory exists ----
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Plot 1: Equity Curves ----
    logger.info("Generating equity curves plot...")
    plot_equity_curves(
        return_series,
        save_path=REPORTS_DIR / "equity_curves.png",
    )

    # ---- Plot 2: Drawdown Chart (InvVol Top-4) ----
    if "InvVol Ensemble Top-4 (1x)" in return_series:
        logger.info("Generating drawdown chart...")
        plot_drawdown(
            return_series["InvVol Ensemble Top-4 (1x)"],
            label="InvVol Ensemble Top-4 (1x)",
            save_path=REPORTS_DIR / "drawdown_chart.png",
        )
    else:
        logger.warning("InvVol Top-4 not available; skipping drawdown chart.")

    # ---- Plot 3: Rolling Sharpe (top 3 strategies) ----
    logger.info("Generating rolling Sharpe chart...")
    plot_rolling_sharpe(
        return_series,
        save_path=REPORTS_DIR / "rolling_sharpe.png",
    )

    # ---- Plot 4: Monthly Returns Heatmap (InvVol Ensemble 1.5x) ----
    if "InvVol Ensemble (1.5x)" in return_series:
        logger.info("Generating monthly returns heatmap...")
        plot_monthly_heatmap(
            return_series["InvVol Ensemble (1.5x)"],
            label="InvVol Ensemble (1.5x leverage)",
            save_path=REPORTS_DIR / "monthly_returns.png",
        )
    else:
        logger.warning("InvVol 1.5x not available; skipping monthly heatmap.")

    logger.info("All plots saved to %s", REPORTS_DIR)


if __name__ == "__main__":
    main()
