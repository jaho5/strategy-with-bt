"""Generate correlation heatmap and dendrogram of top-10 strategy returns.

Runs the top 10 strategies on test data, computes pairwise Pearson
correlations of their daily return series, and produces:

1. A seaborn heatmap saved to ``reports/strategy_correlations.png``
2. A scipy hierarchical-clustering dendrogram saved to
   ``reports/strategy_dendrogram.png``
3. The correlation matrix printed to stdout.

Usage:
    uv run python -m src.utils.plot_correlations
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from src.backtest.engine import BacktestEngine
from src.data.downloader import download_etf_data

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
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

# Top 10 strategies: (display_name, module_path, class_name)
TOP_10_STRATEGIES: List[Tuple[str, str, str]] = [
    ("Entropy Regularized", "src.strategies.entropy_regularized", "EntropyRegularizedStrategy"),
    ("GARCH Vol", "src.strategies.garch_vol", "GarchVolStrategy"),
    ("HMM Regime", "src.strategies.hmm_regime", "HMMRegimeStrategy"),
    ("Spectral Momentum", "src.strategies.spectral_momentum", "SpectralMomentumStrategy"),
    ("Bayesian Changepoint", "src.strategies.bayesian_changepoint", "BayesianChangepointStrategy"),
    ("Mean Field Game", "src.strategies.mean_field", "MeanFieldStrategy"),
    ("Concentration Bounds", "src.strategies.concentration_bounds", "ConcentrationBoundsStrategy"),
    ("Stochastic Control (Conservative)", "src.strategies.stochastic_control", "ConservativeStochasticControlStrategy"),
    ("Kelly Growth Optimal", "src.strategies.kelly_growth", "KellyGrowthStrategy"),
    ("Momentum Crash Hedge", "src.strategies.momentum_crash_hedge", "MomentumCrashHedgeStrategy"),
]


# ---------------------------------------------------------------------------
# Data helpers (reused from src.main)
# ---------------------------------------------------------------------------

def _extract_close_prices(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Extract close prices from the multi-level yfinance DataFrame."""
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


def _extract_ohlcv_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw yfinance MultiIndex DataFrame for strategy consumption."""
    if not isinstance(raw_data.columns, pd.MultiIndex):
        return raw_data.dropna(how="all").ffill().bfill()
    return raw_data.dropna(how="all").ffill().bfill()


def _accepts_kwargs(method) -> bool:
    """Return True if *method* has a ``**kwargs`` parameter."""
    try:
        sig = inspect.signature(method)
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
    except (ValueError, TypeError):
        return False


def _split_data(
    data: pd.DataFrame, train_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and testing sets by row count."""
    n = len(data)
    split_idx = int(n * train_frac)
    return data.iloc[:split_idx], data.iloc[split_idx:]


def _signals_to_portfolio(
    signals: pd.DataFrame, prices: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a multi-asset signal DataFrame to portfolio signal and price series.

    Mirrors the implementation in ``src.main`` so that return streams are
    consistent with the main backtest pipeline.
    """
    common_idx = signals.index.intersection(prices.index)
    if len(common_idx) == 0:
        common_idx = signals.index

    sig_cols: Dict[str, str] = {}
    price_tickers = list(prices.columns)

    # Pattern 1: columns match price tickers directly
    direct_match = [c for c in signals.columns if c in price_tickers]
    if direct_match:
        for c in direct_match:
            sig_cols[c] = c

    # Pattern 2: '{ticker}_signal' columns
    if not sig_cols:
        for ticker in price_tickers:
            scol = f"{ticker}_signal"
            if scol in signals.columns:
                sig_cols[ticker] = scol

    # Pattern 3: single 'signal' column
    if not sig_cols and "signal" in signals.columns:
        sig_arr = signals["signal"].reindex(common_idx).fillna(0.0).values
        price_arr = prices.reindex(common_idx).ffill().bfill().mean(axis=1).values
        if price_arr[0] != 0:
            price_arr = price_arr / price_arr[0] * 100.0
        return sig_arr, price_arr

    # Pattern 4: no match -- average all numeric signal columns
    if not sig_cols:
        sig_arr = signals.reindex(common_idx).fillna(0.0).select_dtypes(include="number").mean(axis=1).values
        price_arr = prices.reindex(common_idx).ffill().bfill().mean(axis=1).values
        if len(price_arr) > 0 and price_arr[0] != 0:
            price_arr = price_arr / price_arr[0] * 100.0
        return sig_arr, price_arr

    # Build aligned signal and price DataFrames
    tickers_used = list(sig_cols.keys())
    sig_df = pd.DataFrame(index=common_idx)
    for ticker in tickers_used:
        sig_df[ticker] = signals[sig_cols[ticker]].reindex(common_idx).fillna(0.0)

    price_aligned = prices[tickers_used].reindex(common_idx).ffill().bfill()
    returns = price_aligned.pct_change().fillna(0.0)

    net_exposure = sig_df.sum(axis=1)
    gross_exposure = sig_df.abs().sum(axis=1)
    mean_gross = gross_exposure.mean()
    mean_abs_net = net_exposure.abs().mean()
    is_long_short = (mean_gross > 1e-8) and (mean_abs_net / mean_gross < 0.3)

    if is_long_short:
        lagged_signals = sig_df.shift(1).fillna(0.0)
        portfolio_returns = (lagged_signals * returns).sum(axis=1)
        portfolio_price = 100.0 * (1.0 + portfolio_returns).cumprod()
        portfolio_signal = np.ones(len(common_idx))
    else:
        portfolio_signal = sig_df.mean(axis=1).values
        portfolio_returns = returns.mean(axis=1)
        portfolio_price = 100.0 * (1.0 + portfolio_returns).cumprod()

    return portfolio_signal, portfolio_price.values


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _load_top10_strategies() -> List[Tuple[str, Any]]:
    """Import and instantiate the top-10 strategies."""
    strategies: List[Tuple[str, Any]] = []
    for display_name, module_path, class_name in TOP_10_STRATEGIES:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls()
            strategies.append((display_name, instance))
            logger.info("Loaded strategy: %s", display_name)
        except Exception as exc:
            logger.warning(
                "Could not load strategy '%s' (%s.%s): %s",
                display_name, module_path, class_name, exc,
            )
    return strategies


def _run_strategy_on_test(
    name: str,
    strategy: Any,
    close_prices: pd.DataFrame,
    ohlcv_data: pd.DataFrame,
    engine: BacktestEngine,
) -> pd.Series | None:
    """Fit a strategy on train data, generate signals on test data, backtest,
    and return the daily return series (or None on failure).
    """
    pass_ohlcv_fit = _accepts_kwargs(strategy.fit)
    pass_ohlcv_sig = _accepts_kwargs(strategy.generate_signals)

    train_data, test_data = _split_data(close_prices, TRAIN_FRACTION)
    ohlcv_train, ohlcv_test = _split_data(ohlcv_data, TRAIN_FRACTION)

    logger.info("  Fitting '%s' on training data...", name)
    if pass_ohlcv_fit:
        strategy.fit(train_data, ohlcv_data=ohlcv_train)
    else:
        strategy.fit(train_data)

    logger.info("  Generating signals on test data...")
    if pass_ohlcv_sig:
        signals = strategy.generate_signals(test_data, ohlcv_data=ohlcv_test)
    else:
        signals = strategy.generate_signals(test_data)

    portfolio_signal, portfolio_price = _signals_to_portfolio(signals, test_data)
    min_len = min(len(portfolio_signal), len(portfolio_price))
    portfolio_signal = portfolio_signal[:min_len]
    portfolio_price = portfolio_price[:min_len]

    if min_len < 2:
        logger.warning("  '%s': not enough data points after signal generation.", name)
        return None

    bt_result = engine.run(portfolio_signal, portfolio_price)
    # Return daily returns as a pandas Series indexed by the test dates
    test_dates = test_data.index[:min_len]
    return pd.Series(bt_result.returns, index=test_dates, name=name)


def _collect_strategy_returns(
    close_prices: pd.DataFrame,
    ohlcv_data: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """Run all top-10 strategies and collect their daily return series."""
    strategies = _load_top10_strategies()
    if not strategies:
        logger.error("No strategies could be loaded.")
        sys.exit(1)

    engine = BacktestEngine()
    returns_dict: Dict[str, pd.Series] = {}

    for name, strategy in strategies:
        logger.info("=" * 60)
        logger.info("Running strategy: %s", name)
        logger.info("=" * 60)
        try:
            ret = _run_strategy_on_test(name, strategy, close_prices, ohlcv_data, engine)
            if ret is not None:
                returns_dict[name] = ret
                logger.info("  '%s' returned %d daily observations.", name, len(ret))
        except Exception as exc:
            logger.error("  Strategy '%s' failed: %s", name, exc)

    return returns_dict


def _plot_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
    """Create and save the correlation heatmap."""
    n = len(corr)
    fig_size = max(8, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title("Strategy Return Correlations", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap saved to %s", output_path)


def _plot_dendrogram(corr: pd.DataFrame, output_path: Path) -> None:
    """Create and save the hierarchical-clustering dendrogram."""
    # Distance = 1 - |correlation|
    dist = 1.0 - np.abs(corr.values)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    dist = np.clip(dist, 0.0, 2.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")

    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(
        Z,
        labels=list(corr.columns),
        leaf_rotation=35,
        leaf_font_size=9,
        ax=ax,
        color_threshold=0.7 * max(Z[:, 2]),
    )
    ax.set_title("Strategy Hierarchical Clustering (by Correlation Distance)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Distance (1 - |correlation|)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Dendrogram saved to %s", output_path)


def main() -> None:
    """Entry point."""
    logger.info("Downloading ETF data from %s to %s...", DATA_START, DATA_END)
    raw_data = download_etf_data(start=DATA_START, end=DATA_END)
    logger.info("Downloaded data shape: %s", raw_data.shape)

    close_prices = _extract_close_prices(raw_data)
    ohlcv_data = _extract_ohlcv_data(raw_data)
    logger.info(
        "Close prices: %d rows x %d tickers (%s to %s)",
        len(close_prices),
        len(close_prices.columns),
        close_prices.index[0].date(),
        close_prices.index[-1].date(),
    )

    # Step 1: Run top-10 strategies on test data
    returns_dict = _collect_strategy_returns(close_prices, ohlcv_data)

    if len(returns_dict) < 2:
        logger.error("Need at least 2 strategies to compute correlations. Exiting.")
        sys.exit(1)

    # Align return series into a single DataFrame
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.sort_index().ffill().fillna(0.0)

    # Step 2: Compute pairwise Pearson correlations
    corr = returns_df.corr(method="pearson")

    # Ensure output directory exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 3: Heatmap
    heatmap_path = REPORTS_DIR / "strategy_correlations.png"
    _plot_heatmap(corr, heatmap_path)

    # Step 4: Dendrogram
    dendro_path = REPORTS_DIR / "strategy_dendrogram.png"
    _plot_dendrogram(corr, dendro_path)

    # Step 5: Print correlation matrix to stdout
    print("\n" + "=" * 80)
    print("STRATEGY RETURN CORRELATIONS (Pearson)")
    print("=" * 80)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print(corr)
    print("=" * 80)
    print(f"\nHeatmap saved to:     {heatmap_path}")
    print(f"Dendrogram saved to:  {dendro_path}")


if __name__ == "__main__":
    main()
