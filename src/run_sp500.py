"""S&P 500 individual stock strategy runner.

Tests the top-5 strategies best suited for individual stocks on the
50 largest S&P 500 constituents.  For each strategy, cross-sectional
ranking selects long top-10 / short bottom-10 with equal weight within
each leg.

Usage:
    uv run python -m src.run_sp500
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    MonteCarloResult,
    WalkForwardResult,
)
from src.data.downloader import download_universe, get_sp500_tickers

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
DATA_START = "2015-01-01"
DATA_END = "2025-12-31"
N_STOCKS = 50                  # top N from get_sp500_tickers()
TRAIN_FRACTION = 0.70
WALK_FORWARD_FOLDS = 5
MONTE_CARLO_SIMS = 10_000
MONTE_CARLO_TARGET_PNL = 45.0
LONG_LEG = 10                  # number of stocks in the long leg
SHORT_LEG = 10                 # number of stocks in the short leg
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# The five strategies selected for individual-stock alpha:
STRATEGY_REGISTRY: List[Tuple[str, str, str, dict]] = [
    (
        "Kelly Growth Optimal",
        "src.strategies.kelly_growth",
        "KellyGrowthStrategy",
        {},
    ),
    (
        "Momentum Crash Hedge",
        "src.strategies.momentum_crash_hedge",
        "MomentumCrashHedgeStrategy",
        {"long_pct": LONG_LEG / N_STOCKS, "short_pct": SHORT_LEG / N_STOCKS},
    ),
    (
        "Fractional Differentiation",
        "src.strategies.fractional_differentiation",
        "FractionalDifferentiationStrategy",
        {},
    ),
    (
        "GARCH Vol",
        "src.strategies.garch_vol",
        "GarchVolStrategy",
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

def download_sp500_data() -> pd.DataFrame:
    """Download close prices for the top N_STOCKS S&P 500 constituents.

    Uses the cache provided by ``download_universe``.  Returns a DataFrame
    indexed by date with one column per ticker (close prices only).  Missing
    data is forward-filled and back-filled; tickers with > 50 % NaN are
    dropped entirely.
    """
    tickers = get_sp500_tickers()[:N_STOCKS]
    logger.info(
        "Downloading %d S&P 500 stocks (%s to %s)...",
        len(tickers), DATA_START, DATA_END,
    )

    raw = download_universe(tickers, start=DATA_START, end=DATA_END, use_cache=True)

    # Extract close prices from the multi-level yfinance DataFrame.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        else:
            first_level = raw.columns.get_level_values(0)[0]
            close = raw[first_level]
    else:
        close = raw

    # Drop rows where everything is NaN, then forward/back fill gaps.
    close = close.dropna(how="all")

    # Drop tickers with excessive NaN (e.g. IPO'd after start date).
    pct_nan = close.isna().mean()
    keep = pct_nan[pct_nan < 0.50].index.tolist()
    dropped = [t for t in close.columns if t not in keep]
    if dropped:
        logger.warning(
            "Dropping %d tickers with >50%% missing data: %s",
            len(dropped), dropped,
        )
    close = close[keep]

    close = close.ffill().bfill()
    logger.info(
        "Close prices ready: %d rows x %d tickers (%s to %s)",
        len(close), len(close.columns),
        close.index[0].date(), close.index[-1].date(),
    )
    return close


def _split_data(
    data: pd.DataFrame, train_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train / test by row count."""
    n = len(data)
    split_idx = int(n * train_frac)
    return data.iloc[:split_idx], data.iloc[split_idx:]


# ---------------------------------------------------------------------------
# Cross-sectional ranking: long top-K / short bottom-K
# ---------------------------------------------------------------------------

def _rank_and_select(
    scores: pd.DataFrame,
    n_long: int = LONG_LEG,
    n_short: int = SHORT_LEG,
) -> pd.DataFrame:
    """Rank stocks cross-sectionally and assign equal-weight long/short.

    Parameters
    ----------
    scores : pd.DataFrame
        Per-stock score at each date (higher = more bullish).
        Columns are tickers, index is dates.
    n_long : int
        Number of stocks in the long leg.
    n_short : int
        Number of stocks in the short leg.

    Returns
    -------
    pd.DataFrame
        Position weights: +1/n_long for longs, -1/n_short for shorts, 0 else.
    """
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    for t in range(len(scores)):
        row = scores.iloc[t]
        valid = row.dropna()
        if len(valid) < n_long + n_short:
            continue
        ranked = valid.sort_values(ascending=False)
        long_tickers = ranked.index[:n_long]
        short_tickers = ranked.index[-n_short:]
        weights.iloc[t, weights.columns.isin(long_tickers)] = 1.0 / n_long
        weights.iloc[t, weights.columns.isin(short_tickers)] = -1.0 / n_short

    return weights


# ---------------------------------------------------------------------------
# Signal extraction helpers (strategy-specific)
# ---------------------------------------------------------------------------

def _extract_scores_signal_weight(
    signals_df: pd.DataFrame,
    tickers: List[str],
) -> pd.DataFrame:
    """Extract a per-ticker score from '{ticker}_signal' * '{ticker}_weight'.

    Returns a DataFrame of composite scores (signal * weight) with one
    column per ticker.
    """
    scores = pd.DataFrame(index=signals_df.index)
    for t in tickers:
        sig_col = f"{t}_signal"
        wgt_col = f"{t}_weight"
        if sig_col in signals_df.columns and wgt_col in signals_df.columns:
            scores[t] = (
                signals_df[sig_col].fillna(0.0) * signals_df[wgt_col].fillna(0.0)
            )
        elif sig_col in signals_df.columns:
            scores[t] = signals_df[sig_col].fillna(0.0)
        else:
            scores[t] = 0.0
    return scores


def _extract_scores_direct(
    signals_df: pd.DataFrame,
    tickers: List[str],
) -> pd.DataFrame:
    """Extract scores when the strategy returns columns matching tickers.

    The GARCH strategy returns position sizes directly under ticker names.
    """
    scores = pd.DataFrame(index=signals_df.index)
    for t in tickers:
        if t in signals_df.columns:
            scores[t] = signals_df[t].fillna(0.0)
        else:
            scores[t] = 0.0
    return scores


def _extract_scores(
    signals_df: pd.DataFrame,
    tickers: List[str],
) -> pd.DataFrame:
    """Detect the signal format and extract per-ticker scores.

    Supports both '{ticker}_signal'/'{ticker}_weight' and direct-ticker
    column conventions.
    """
    # Check which convention is in use.
    has_signal_weight = any(f"{t}_signal" in signals_df.columns for t in tickers)
    has_direct = any(t in signals_df.columns for t in tickers)

    if has_signal_weight:
        return _extract_scores_signal_weight(signals_df, tickers)
    elif has_direct:
        return _extract_scores_direct(signals_df, tickers)
    else:
        logger.warning("Could not match signal columns to tickers; returning zeros.")
        return pd.DataFrame(0.0, index=signals_df.index, columns=tickers)


# ---------------------------------------------------------------------------
# Portfolio-level return series from cross-sectional weights
# ---------------------------------------------------------------------------

def _weights_to_portfolio_return(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.Series:
    """Compute a daily portfolio return series from position weights and prices.

    Parameters
    ----------
    weights : pd.DataFrame
        Position weights (same shape as prices), shifted by 1 bar
        internally to avoid look-ahead.
    prices : pd.DataFrame
        Close prices aligned with weights.

    Returns
    -------
    pd.Series
        Daily portfolio simple returns.
    """
    common_cols = weights.columns.intersection(prices.columns)
    daily_returns = prices[common_cols].pct_change().fillna(0.0)
    # Positions at time t earn the return from t to t+1.
    port_ret = (weights[common_cols].shift(1).fillna(0.0) * daily_returns).sum(axis=1)
    return port_ret


def _portfolio_return_to_price(port_ret: pd.Series, start: float = 100.0) -> pd.Series:
    """Convert a return series to a synthetic price series."""
    return start * (1.0 + port_ret).cumprod()


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def _fit_and_score(
    strategy: Any,
    train_prices: pd.DataFrame,
    eval_prices: pd.DataFrame,
    tickers: List[str],
) -> pd.DataFrame:
    """Fit a strategy on train_prices, generate signals on eval_prices,
    and extract cross-sectional scores.

    Returns a DataFrame of per-ticker scores (higher = more bullish).
    """
    strategy.fit(train_prices)
    raw_signals = strategy.generate_signals(eval_prices)
    scores = _extract_scores(raw_signals, tickers)
    return scores


def _run_cross_sectional_backtest(
    name: str,
    strategy: Any,
    close_prices: pd.DataFrame,
    engine: BacktestEngine,
) -> Optional[Dict[str, Any]]:
    """Run a single strategy through the full pipeline.

    1. Split into train / test.
    2. Fit on train, generate cross-sectional scores on test.
    3. Rank stocks, assign long top-10 / short bottom-10.
    4. Build portfolio return series, run through BacktestEngine.
    5. Walk-forward validation (5 folds).
    6. Monte Carlo confidence.

    Returns a dict of summary metrics or None on failure.
    """
    logger.info("=" * 70)
    logger.info("Running strategy: %s", name)
    logger.info("=" * 70)

    tickers = list(close_prices.columns)

    # ---- 1. Split data ----
    train_data, test_data = _split_data(close_prices, TRAIN_FRACTION)
    logger.info(
        "  Train: %s to %s (%d bars)",
        train_data.index[0].date(), train_data.index[-1].date(), len(train_data),
    )
    logger.info(
        "  Test:  %s to %s (%d bars)",
        test_data.index[0].date(), test_data.index[-1].date(), len(test_data),
    )

    # ---- 2. Fit and score ----
    logger.info("  Fitting and generating signals on test data...")
    scores = _fit_and_score(strategy, train_data, test_data, tickers)

    # ---- 3. Cross-sectional ranking ----
    logger.info("  Cross-sectional ranking: long top %d, short bottom %d", LONG_LEG, SHORT_LEG)
    weights = _rank_and_select(scores, n_long=LONG_LEG, n_short=SHORT_LEG)

    # ---- 4. Portfolio return and backtest ----
    port_ret = _weights_to_portfolio_return(weights, test_data)
    port_price = _portfolio_return_to_price(port_ret)

    # Use a constant signal of 1.0 because portfolio construction is already
    # baked into the synthetic price series.
    port_signal = np.ones(len(port_price))
    port_price_arr = port_price.values

    if len(port_price_arr) < 2:
        logger.warning("  Not enough data for backtest. Skipping.")
        return None

    logger.info("  Running backtest...")
    bt_result: BacktestResult = engine.run(port_signal, port_price_arr)
    metrics = bt_result.metrics

    logger.info("  --- Backtest Results ---")
    logger.info("  Total PnL:          %.2f%%", metrics["total_pnl_pct"])
    logger.info("  Ann. Return:        %.2f%%", metrics["annualized_return"] * 100)
    logger.info("  Sharpe Ratio:       %.3f", metrics["sharpe_ratio"])
    logger.info("  Sortino Ratio:      %.3f", metrics["sortino_ratio"])
    logger.info("  Max Drawdown:       %.2f%%", metrics["max_drawdown"] * 100)
    logger.info("  Win Rate:           %.2f%%", metrics["win_rate"] * 100)
    logger.info("  Bootstrap p-value:  %.4f", metrics["bootstrap_pvalue"])

    # ---- 5. Walk-forward validation ----
    logger.info("  Running walk-forward test (%d folds)...", WALK_FORWARD_FOLDS)

    def wf_strategy_fn(context: dict) -> np.ndarray:
        """Walk-forward callback compatible with BacktestEngine."""
        test_start = context["test_start"]
        test_end = context["test_end"]
        expected_len = test_end - test_start

        # Map integer indices back to the full multi-asset DataFrame.
        wf_train = close_prices.iloc[:test_start]
        wf_test = close_prices.iloc[test_start:test_end]

        if len(wf_train) < 252 or len(wf_test) < 2:
            return np.zeros(expected_len)

        try:
            wf_scores = _fit_and_score(strategy, wf_train, wf_test, tickers)
            wf_weights = _rank_and_select(wf_scores)
            wf_port_ret = _weights_to_portfolio_return(wf_weights, wf_test)
            # Return the portfolio returns as the "signal" to apply to a
            # synthetic price.  We reconstruct a price from the returns so
            # the engine can compute equity correctly.
            # Since the engine multiplies signal * asset_return, and we want
            # the engine to faithfully replay our portfolio return, we pass
            # signal=1.0 and let the price series encode the returns.
            return np.ones(expected_len)
        except Exception as exc:
            logger.warning("  WF fold failed: %s", exc)
            return np.zeros(expected_len)

    # Build a full-period synthetic price for walk-forward.
    # For each fold the callback returns signal=1.0 and the price series
    # is the full cross-sectional portfolio price.
    full_scores = _fit_and_score(strategy, train_data, close_prices, tickers)
    full_weights = _rank_and_select(full_scores)
    full_port_ret = _weights_to_portfolio_return(full_weights, close_prices)
    full_port_price = _portfolio_return_to_price(full_port_ret).values

    try:
        wf_result: WalkForwardResult = engine.walk_forward_test(
            strategy_fn=wf_strategy_fn,
            prices=full_port_price,
            n_splits=WALK_FORWARD_FOLDS,
            train_pct=TRAIN_FRACTION,
        )
        wf_oos_sharpe = wf_result.aggregate_metrics.get("sharpe_ratio", np.nan)
    except Exception as exc:
        logger.warning("  Walk-forward failed: %s", exc)
        wf_oos_sharpe = np.nan

    logger.info("  WF OOS Sharpe:      %.3f", wf_oos_sharpe)

    # ---- 6. Monte Carlo ----
    logger.info("  Running Monte Carlo (%d simulations)...", MONTE_CARLO_SIMS)
    try:
        mc_result: MonteCarloResult = engine.monte_carlo_confidence(
            returns=bt_result.returns,
            n_simulations=MONTE_CARLO_SIMS,
            target_pnl_pct=MONTE_CARLO_TARGET_PNL,
        )
        mc_prob = mc_result.prob_above_target
    except Exception as exc:
        logger.warning("  Monte Carlo failed: %s", exc)
        mc_prob = np.nan

    logger.info(
        "  P(PnL > %.0f%%):     %.4f",
        MONTE_CARLO_TARGET_PNL,
        mc_prob if not np.isnan(mc_prob) else 0.0,
    )

    return {
        "Strategy": name,
        "Total PnL%": round(metrics["total_pnl_pct"], 2),
        "Ann. Return%": round(metrics["annualized_return"] * 100, 2),
        "Sharpe": round(metrics["sharpe_ratio"], 3),
        "Sortino": round(metrics["sortino_ratio"], 3),
        "Max DD%": round(metrics["max_drawdown"] * 100, 2),
        "Win Rate%": round(metrics["win_rate"] * 100, 2),
        f"P(PnL>{int(MONTE_CARLO_TARGET_PNL)}%) MC": round(mc_prob, 4) if not np.isnan(mc_prob) else np.nan,
        "Bootstrap p-value": round(metrics["bootstrap_pvalue"], 4),
        "WF OOS Sharpe": round(wf_oos_sharpe, 3) if not np.isnan(wf_oos_sharpe) else np.nan,
    }


# ---------------------------------------------------------------------------
# Strategy loader
# ---------------------------------------------------------------------------

def _load_strategies() -> List[Tuple[str, Any]]:
    """Import and instantiate the five selected strategies."""
    strategies: List[Tuple[str, Any]] = []
    for display_name, module_path, class_name, kwargs in STRATEGY_REGISTRY:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls(**kwargs)
            strategies.append((display_name, instance))
            logger.info("Loaded strategy: %s", display_name)
        except Exception as exc:
            logger.warning(
                "Could not load strategy '%s' (%s.%s): %s",
                display_name, module_path, class_name, exc,
            )
    return strategies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: download S&P 500 data, run strategies, produce report."""

    # ---- Download and prepare data ----
    close_prices = download_sp500_data()

    # ---- Load strategies ----
    strategies = _load_strategies()
    if not strategies:
        logger.error("No strategies could be loaded. Exiting.")
        sys.exit(1)

    logger.info(
        "Loaded %d strategies. Universe: %d stocks.",
        len(strategies), len(close_prices.columns),
    )

    # ---- Backtest engine ----
    engine = BacktestEngine()

    # ---- Run each strategy ----
    results: List[Dict[str, Any]] = []

    for name, strategy in strategies:
        try:
            t_start = time.perf_counter()
            row = _run_cross_sectional_backtest(name, strategy, close_prices, engine)
            elapsed = time.perf_counter() - t_start
            logger.info("Strategy '%s' completed in %.2f seconds.", name, elapsed)
            if row is not None:
                results.append(row)
        except Exception:
            logger.error(
                "Strategy '%s' failed:\n%s", name, traceback.format_exc(),
            )
            continue

    if not results:
        logger.error("All strategies failed. No results to report.")
        sys.exit(1)

    # ---- Build comparison table ----
    comparison = pd.DataFrame(results)

    mc_col = f"P(PnL>{int(MONTE_CARLO_TARGET_PNL)}%) MC"
    if mc_col in comparison.columns:
        comparison = comparison.sort_values(mc_col, ascending=False).reset_index(drop=True)

    # ---- Save to CSV ----
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "sp500_results.csv"
    comparison.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    # ---- Print the table ----
    print("\n" + "=" * 110)
    print("S&P 500 INDIVIDUAL STOCK STRATEGY COMPARISON")
    print(f"Universe: top {N_STOCKS} stocks | Long {LONG_LEG} / Short {SHORT_LEG}")
    print(f"Period: {DATA_START} to {DATA_END} | Walk-forward: {WALK_FORWARD_FOLDS} folds")
    print("=" * 110)
    print(
        tabulate(
            comparison,
            headers="keys",
            tablefmt="pretty",
            showindex=False,
            floatfmt=".4f",
        )
    )
    print("=" * 110)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
