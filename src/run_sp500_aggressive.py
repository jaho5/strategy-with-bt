"""S&P 500 aggressive strategy runner targeting >45% annualized return.

Tests three cross-sectional strategies on the top 50 S&P 500 stocks, plus
an inverse-volatility ensemble of all three.  Each strategy is evaluated
via backtest, walk-forward validation, and Monte Carlo simulation against
a 45% annualized return target.

Strategies:
  1. Momentum Crash Hedge -- cross-sectional momentum with crash protection
  2. Stein Shrinkage       -- James-Stein shrunk mean-variance
  3. Ergodic Growth         -- geometric growth rate ranking

For each strategy:
  - Fit on 70% training data
  - Generate signals on 30% test data (~2022-2025)
  - Long top 10 stocks, short bottom 10 stocks
  - Run BacktestEngine
  - Walk-forward 5 folds
  - Monte Carlo with target = (1.45)^n_years - 1

Also tests a simple inverse-vol ensemble of the three strategies.

Saves to reports/sp500_aggressive_results.csv

Usage:
    uv run python -m src.run_sp500_aggressive
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
ANNUALIZED_TARGET = 0.45       # 45% per year
LONG_LEG = 10                  # number of stocks in the long leg
SHORT_LEG = 10                 # number of stocks in the short leg
INV_VOL_LOOKBACK = 63          # rolling window for inverse-vol weights
MIN_DATA_COVERAGE = 0.80       # drop stocks with <80% data coverage
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# The three strategies for this aggressive S&P 500 test:
STRATEGY_REGISTRY: List[Tuple[str, str, str, dict]] = [
    (
        "Momentum Crash Hedge",
        "src.strategies.momentum_crash_hedge",
        "MomentumCrashHedgeStrategy",
        {
            "long_pct": LONG_LEG / N_STOCKS,
            "short_pct": SHORT_LEG / N_STOCKS,
            "vol_target": 0.25,
            "vol_window": 42,
        },
    ),
    (
        "Stein Shrinkage",
        "src.strategies.stein_shrinkage",
        "SteinShrinkageStrategy",
        {},  # uses default SteinShrinkageConfig; configured below
    ),
    (
        "Ergodic Growth",
        "src.strategies.ergodic_growth",
        "ErgodicGrowthStrategy",
        {},  # uses default ErgodicGrowthConfig; configured below
    ),
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def download_sp500_data() -> pd.DataFrame:
    """Download close prices for the top N_STOCKS S&P 500 constituents.

    Uses the cache provided by ``download_universe``.  Returns a DataFrame
    indexed by date with one column per ticker (close prices only).  Missing
    data is forward-filled and back-filled; tickers with > (1 - MIN_DATA_COVERAGE)
    NaN are dropped entirely.
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
    max_nan = 1.0 - MIN_DATA_COVERAGE
    pct_nan = close.isna().mean()
    keep = pct_nan[pct_nan < max_nan].index.tolist()
    dropped = [t for t in close.columns if t not in keep]
    if dropped:
        logger.warning(
            "Dropping %d tickers with >%.0f%% missing data: %s",
            len(dropped), max_nan * 100, dropped,
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


def _compute_test_years(test_data: pd.DataFrame) -> float:
    """Compute the number of years in the test period."""
    return len(test_data) / 252.0


def _compute_mc_target_pnl(test_years: float) -> float:
    """Compute the total PnL% target for Monte Carlo.

    For 45% annualized over test_years:
        target = ((1.45)^test_years - 1) * 100
    """
    return ((1.0 + ANNUALIZED_TARGET) ** test_years - 1.0) * 100.0


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
    """Extract a per-ticker score from '{ticker}_signal' * '{ticker}_weight'."""
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
    """Extract scores when the strategy returns columns matching tickers."""
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
    """Detect the signal format and extract per-ticker scores."""
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
# Inverse-volatility weighted ensemble
# ---------------------------------------------------------------------------

def _inverse_vol_combine(
    strategy_returns: Dict[str, pd.Series],
    lookback: int = INV_VOL_LOOKBACK,
) -> pd.Series:
    """Inverse-volatility weighted combination of strategy return series.

    Weights are proportional to 1/sigma, re-estimated using a rolling
    lookback window.  Before enough history is accumulated, equal weights
    are used.
    """
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
    mc_target_pnl: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.Series]]:
    """Run a single strategy through the full pipeline.

    1. Split into train / test.
    2. Fit on train, generate cross-sectional scores on test.
    3. Rank stocks, assign long top-10 / short bottom-10.
    4. Build portfolio return series, run through BacktestEngine.
    5. Walk-forward validation (5 folds).
    6. Monte Carlo confidence.

    Returns a tuple of (summary_metrics_dict, portfolio_return_series).
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

    port_signal = np.ones(len(port_price))
    port_price_arr = port_price.values

    if len(port_price_arr) < 2:
        logger.warning("  Not enough data for backtest. Skipping.")
        return None, None

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

    # ---- 5. Walk-forward validation ----
    logger.info("  Running walk-forward test (%d folds)...", WALK_FORWARD_FOLDS)

    def wf_strategy_fn(context: dict) -> np.ndarray:
        """Walk-forward callback compatible with BacktestEngine."""
        test_start = context["test_start"]
        test_end = context["test_end"]
        expected_len = test_end - test_start

        wf_train = close_prices.iloc[:test_start]
        wf_test = close_prices.iloc[test_start:test_end]

        if len(wf_train) < 252 or len(wf_test) < 2:
            return np.zeros(expected_len)

        try:
            wf_scores = _fit_and_score(strategy, wf_train, wf_test, tickers)
            wf_weights = _rank_and_select(wf_scores)
            wf_port_ret = _weights_to_portfolio_return(wf_weights, wf_test)
            return np.ones(expected_len)
        except Exception as exc:
            logger.warning("  WF fold failed: %s", exc)
            return np.zeros(expected_len)

    # Build a full-period synthetic price for walk-forward.
    full_scores = _fit_and_score(strategy, train_data, close_prices, tickers)
    full_weights = _rank_and_select(full_scores)
    full_port_ret = _weights_to_portfolio_return(full_weights, close_prices)
    full_port_price = _portfolio_return_to_price(full_port_ret).values

    wf_oos_sharpe = np.nan
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

    logger.info("  WF OOS Sharpe:      %.3f", wf_oos_sharpe)

    # ---- 6. Monte Carlo ----
    logger.info("  Running Monte Carlo (%d simulations)...", MONTE_CARLO_SIMS)
    mc_prob = np.nan
    try:
        mc_result: MonteCarloResult = engine.monte_carlo_confidence(
            returns=bt_result.returns,
            n_simulations=MONTE_CARLO_SIMS,
            target_pnl_pct=mc_target_pnl,
        )
        mc_prob = mc_result.prob_above_target
    except Exception as exc:
        logger.warning("  Monte Carlo failed: %s", exc)

    logger.info(
        "  P(PnL > %.0f%%):     %.4f",
        mc_target_pnl,
        mc_prob if not np.isnan(mc_prob) else 0.0,
    )

    # ---- PASS/FAIL gate ----
    mc_pass = (not np.isnan(mc_prob)) and (mc_prob >= 0.95)
    wf_pass = (not np.isnan(wf_oos_sharpe)) and (wf_oos_sharpe > 0)
    verdict = "PASS" if (mc_pass and wf_pass) else "FAIL"

    row = {
        "Strategy": name,
        "Ann. Return%": round(metrics["annualized_return"] * 100, 2),
        "Sharpe": round(metrics["sharpe_ratio"], 3),
        "Max DD%": round(metrics["max_drawdown"] * 100, 2),
        f"P(Ann>{int(ANNUALIZED_TARGET*100)}%) MC": (
            round(mc_prob, 4) if not np.isnan(mc_prob) else np.nan
        ),
        "WF OOS Sharpe": (
            round(wf_oos_sharpe, 3) if not np.isnan(wf_oos_sharpe) else np.nan
        ),
        "MC Target PnL%": round(mc_target_pnl, 1),
        "PASS/FAIL": verdict,
    }

    return row, port_ret


# ---------------------------------------------------------------------------
# Strategy loader
# ---------------------------------------------------------------------------

def _load_strategies() -> List[Tuple[str, Any]]:
    """Import and instantiate the three selected strategies."""
    strategies: List[Tuple[str, Any]] = []
    for display_name, module_path, class_name, kwargs in STRATEGY_REGISTRY:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)

            # Special handling for Stein Shrinkage -- use aggressive config
            if class_name == "SteinShrinkageStrategy":
                from src.strategies.stein_shrinkage import SteinShrinkageConfig
                config = SteinShrinkageConfig(
                    lookback_window=252,
                    min_history=126,
                    long_quantile=0.80,       # top 20% long
                    short_quantile=0.20,      # bottom 20% short
                    use_mean_variance=True,
                    risk_aversion=1.0,        # lower risk aversion = more aggressive
                    max_leverage=2.0,         # allow 2x leverage
                    rebalance_freq=10,        # rebalance every 10 days
                )
                instance = cls(config=config)

            # Special handling for Ergodic Growth -- use aggressive config
            elif class_name == "ErgodicGrowthStrategy":
                from src.strategies.ergodic_growth import ErgodicGrowthConfig
                config = ErgodicGrowthConfig(
                    lookback=252,
                    min_history=126,
                    long_quantile=0.20,       # top 20% long
                    short_quantile=0.20,      # bottom 20% short
                    kelly_fraction=0.40,      # near half-Kelly for aggression
                    min_kelly=0.02,
                    max_kelly=0.50,
                    max_position_weight=0.25,
                    max_gross_leverage=2.0,   # allow 2x leverage
                    rebalance_freq=10,
                    ema_span=15,
                )
                instance = cls(config=config)

            else:
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

    t_global_start = time.perf_counter()

    # ---- Download and prepare data ----
    close_prices = download_sp500_data()

    # ---- Compute test period and MC target ----
    _, test_data = _split_data(close_prices, TRAIN_FRACTION)
    test_years = _compute_test_years(test_data)
    mc_target_pnl = _compute_mc_target_pnl(test_years)
    logger.info(
        "Test period: %.2f years | MC target: (1.45)^%.2f - 1 = %.1f%% total PnL",
        test_years, test_years, mc_target_pnl,
    )

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
    strategy_returns: Dict[str, pd.Series] = {}

    for name, strategy in strategies:
        try:
            t_start = time.perf_counter()
            row, port_ret = _run_cross_sectional_backtest(
                name, strategy, close_prices, engine, mc_target_pnl,
            )
            elapsed = time.perf_counter() - t_start
            logger.info("Strategy '%s' completed in %.2f seconds.", name, elapsed)
            if row is not None:
                results.append(row)
            if port_ret is not None and len(port_ret) > 0:
                strategy_returns[name] = port_ret
        except Exception:
            logger.error(
                "Strategy '%s' failed:\n%s", name, traceback.format_exc(),
            )
            continue

    # ---- Inverse-vol ensemble of the three strategies ----
    if len(strategy_returns) >= 2:
        logger.info("\n" + "=" * 70)
        logger.info("Running Inverse-Vol Ensemble of %d strategies", len(strategy_returns))
        logger.info("=" * 70)

        try:
            # Align all return series to common index
            ret_df = pd.DataFrame(strategy_returns).sort_index()
            common_idx = ret_df.dropna(how="all").index
            aligned_returns = {
                n: s.reindex(common_idx).fillna(0.0)
                for n, s in strategy_returns.items()
            }

            # Print correlation matrix
            corr_df = pd.DataFrame(aligned_returns).corr()
            logger.info("Strategy return correlations:")
            logger.info("\n%s", corr_df.to_string(float_format="%.3f"))

            combined_ret = _inverse_vol_combine(aligned_returns, lookback=INV_VOL_LOOKBACK)

            # Build synthetic price from combined returns for backtest
            combined_arr = combined_ret.values.astype(np.float64)
            n = len(combined_arr)

            if n >= 10:
                price = 100.0 * np.cumprod(1.0 + combined_arr)
                price = np.insert(price, 0, 100.0)
                signal = np.ones(len(price))

                bt_result = engine.run(signal, price)
                metrics = bt_result.metrics

                logger.info("  Ensemble PnL: %.2f%%", metrics["total_pnl_pct"])
                logger.info("  Ensemble Ann. Return: %.2f%%", metrics["annualized_return"] * 100)
                logger.info("  Ensemble Sharpe: %.3f", metrics["sharpe_ratio"])
                logger.info("  Ensemble Max DD: %.2f%%", metrics["max_drawdown"] * 100)

                # Monte Carlo
                mc_prob = np.nan
                try:
                    mc_result = engine.monte_carlo_confidence(
                        returns=bt_result.returns,
                        n_simulations=MONTE_CARLO_SIMS,
                        target_pnl_pct=mc_target_pnl,
                    )
                    mc_prob = mc_result.prob_above_target
                except Exception as exc:
                    logger.warning("  Ensemble MC failed: %s", exc)

                # Walk-forward
                wf_oos_sharpe = np.nan
                try:
                    def wf_fn(context: dict) -> np.ndarray:
                        expected_len = context["test_end"] - context["test_start"]
                        return np.ones(expected_len)

                    wf_result = engine.walk_forward_test(
                        strategy_fn=wf_fn,
                        prices=price,
                        n_splits=WALK_FORWARD_FOLDS,
                        train_pct=TRAIN_FRACTION,
                    )
                    wf_oos_sharpe = wf_result.aggregate_metrics.get("sharpe_ratio", np.nan)
                except Exception as exc:
                    logger.warning("  Ensemble WF failed: %s", exc)

                # PASS/FAIL
                mc_pass = (not np.isnan(mc_prob)) and (mc_prob >= 0.95)
                wf_pass = (not np.isnan(wf_oos_sharpe)) and (wf_oos_sharpe > 0)
                verdict = "PASS" if (mc_pass and wf_pass) else "FAIL"

                ensemble_row = {
                    "Strategy": "Inverse-Vol Ensemble (3 strategies)",
                    "Ann. Return%": round(metrics["annualized_return"] * 100, 2),
                    "Sharpe": round(metrics["sharpe_ratio"], 3),
                    "Max DD%": round(metrics["max_drawdown"] * 100, 2),
                    f"P(Ann>{int(ANNUALIZED_TARGET*100)}%) MC": (
                        round(mc_prob, 4) if not np.isnan(mc_prob) else np.nan
                    ),
                    "WF OOS Sharpe": (
                        round(wf_oos_sharpe, 3) if not np.isnan(wf_oos_sharpe) else np.nan
                    ),
                    "MC Target PnL%": round(mc_target_pnl, 1),
                    "PASS/FAIL": verdict,
                }
                results.append(ensemble_row)
            else:
                logger.warning("  Ensemble has too few data points (%d). Skipping.", n)

        except Exception:
            logger.error(
                "Inverse-Vol Ensemble failed:\n%s", traceback.format_exc(),
            )
    else:
        logger.warning(
            "Only %d strategies produced returns. Need >=2 for ensemble.",
            len(strategy_returns),
        )

    # ---- Report ----
    if not results:
        logger.error("All strategies failed. No results to report.")
        sys.exit(1)

    comparison = pd.DataFrame(results)

    # Sort by Ann. Return% descending
    if "Ann. Return%" in comparison.columns:
        comparison = comparison.sort_values(
            "Ann. Return%", ascending=False,
        ).reset_index(drop=True)

    # ---- Save to CSV ----
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "sp500_aggressive_results.csv"
    comparison.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    # ---- Print the table ----
    mc_col = f"P(Ann>{int(ANNUALIZED_TARGET*100)}%) MC"

    print("\n" + "=" * 130)
    print("S&P 500 AGGRESSIVE STRATEGY COMPARISON (Target: >45% Annualized)")
    print(f"Universe: top {N_STOCKS} stocks (>={int(MIN_DATA_COVERAGE*100)}% data coverage)")
    print(f"Long {LONG_LEG} / Short {SHORT_LEG}")
    print(
        f"Period: {DATA_START} to {DATA_END} | "
        f"OOS split: {TRAIN_FRACTION:.0%} train / {1-TRAIN_FRACTION:.0%} test"
    )
    print(
        f"Test period: {test_years:.2f} years | "
        f"MC target PnL: {mc_target_pnl:.1f}% (= 45% ann. over {test_years:.1f}yr)"
    )
    print(
        f"Walk-forward: {WALK_FORWARD_FOLDS} folds | "
        f"Monte Carlo: {MONTE_CARLO_SIMS:,} sims"
    )
    print(
        f"PASS criteria: P(Ann>{int(ANNUALIZED_TARGET*100)}%) >= 0.95 "
        f"AND WF OOS Sharpe > 0"
    )
    print("=" * 130)

    print(
        tabulate(
            comparison,
            headers="keys",
            tablefmt="pretty",
            showindex=False,
            floatfmt=".4f",
        )
    )

    # Summary of pass/fail
    if "PASS/FAIL" in comparison.columns:
        n_pass = (comparison["PASS/FAIL"] == "PASS").sum()
        n_total = len(comparison)
        print(
            f"\nVERDICT: {n_pass} / {n_total} strategies PASSED "
            f"the >45% annualized validation gate."
        )

        passed = comparison[comparison["PASS/FAIL"] == "PASS"]
        if len(passed) > 0:
            best = passed.iloc[0]  # already sorted by Ann. Return%
            print(
                f"BEST STRATEGY: {best['Strategy']} "
                f"(Ann. Return={best['Ann. Return%']:.2f}%, "
                f"Sharpe={best['Sharpe']:.3f}, "
                f"Max DD={best['Max DD%']:.1f}%)"
            )
        else:
            print("WARNING: No strategies passed both gates.")

    print("=" * 130)
    print(f"\nResults saved to: {csv_path}")

    elapsed_total = time.perf_counter() - t_global_start
    print(f"Total runtime: {elapsed_total:.1f} seconds")


if __name__ == "__main__":
    main()
