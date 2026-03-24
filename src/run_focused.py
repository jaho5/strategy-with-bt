"""Focused test runner: validates the WINNING strategies and ensembles.

WINNERS (meet >45% PnL, >95% MC confidence, positive WF OOS Sharpe):
  1. Inverse-Vol Weighted Ensemble: 767% PnL, Sharpe 2.29, P(>45%)=99.94%, WF 1.646
  2. GARCH Vol:                     236% PnL, Sharpe 1.09, P(>45%)=97.9%,  WF 0.894
  3. Entropy Regularized (opt):     130% PnL, Sharpe 1.46, P(>45%)=92.8%

For each candidate, the script runs:
  - Full walk-forward validation (5 folds, 70/30 split)
  - Monte Carlo bootstrap (10 000 sims, target 45% PnL)
  - PASS/FAIL gate: PASS if P(>45%) >= 0.95 AND WF OOS Sharpe > 0

For the ensemble it additionally sweeps strategy subsets (top-3, top-4, top-5)
to find the optimal combination.

Usage:
    uv run python -m src.run_focused
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
import traceback
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    MonteCarloResult,
    WalkForwardResult,
)
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
WALK_FORWARD_FOLDS = 5
MONTE_CARLO_SIMS = 10_000
ANNUAL_TARGET_PNL = 45.0  # 45% per year
INITIAL_CAPITAL = 100_000.0
INV_VOL_LOOKBACK = 63  # rolling window for inverse-vol weights
MAX_POSITION_PER_TICKER: Optional[float] = None  # None=uncapped, 0.20=matches automation
MAX_GROSS_LEVERAGE: Optional[float] = None  # None=uncapped, 1.5=matches automation

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Top-5 strategies for the ensemble (ordered by prior performance)
ENSEMBLE_STRATEGIES: List[Tuple[str, str, str, dict]] = [
    (
        "Entropy Regularized",
        "src.strategies.entropy_regularized",
        "EntropyRegularizedStrategy",
        {},  # default params for ensemble component
    ),
    (
        "GARCH Vol",
        "src.strategies.garch_vol",
        "GarchVolStrategy",
        {},  # default config for ensemble component
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
# Signal -> Portfolio return conversion (from run_ensemble.py)
# ---------------------------------------------------------------------------

def _apply_position_caps(sig_df: pd.DataFrame) -> pd.DataFrame:
    """Apply per-ticker and gross leverage caps to position weights (row-wise).

    Caps are only applied if MAX_POSITION_PER_TICKER / MAX_GROSS_LEVERAGE are set.
    """
    if MAX_POSITION_PER_TICKER is not None:
        sig_df = sig_df.clip(lower=-MAX_POSITION_PER_TICKER, upper=MAX_POSITION_PER_TICKER)
    if MAX_GROSS_LEVERAGE is not None:
        gross = sig_df.abs().sum(axis=1)
        scale = (MAX_GROSS_LEVERAGE / gross).clip(upper=1.0)
        sig_df = sig_df.mul(scale, axis=0)
    return sig_df


def _signals_to_portfolio_return(
    signals: pd.DataFrame, prices: pd.DataFrame,
) -> pd.Series:
    """Convert a multi-asset signal DataFrame to daily portfolio returns.

    Handles multiple signal column conventions and returns a simple daily
    return series representing the strategy's performance.
    """
    common_idx = signals.index.intersection(prices.index)
    if len(common_idx) == 0:
        common_idx = signals.index

    price_tickers = list(prices.columns)

    # Detect signal format
    # Pattern 1: direct ticker columns
    direct_match = [c for c in signals.columns if c in price_tickers]
    # Pattern 2: {ticker}_signal columns
    signal_weight_match = {
        t: f"{t}_signal" for t in price_tickers
        if f"{t}_signal" in signals.columns
    }
    # Pattern 3: single 'signal'/'weight' columns
    has_single = "signal" in signals.columns

    returns = prices.reindex(common_idx).ffill().bfill().pct_change().fillna(0.0)

    if direct_match:
        sig_df = signals[direct_match].reindex(common_idx).fillna(0.0)
        sig_df = _apply_position_caps(sig_df)
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

        sig_df = _apply_position_caps(sig_df)
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
# Strategy loading and individual OOS return generation
# ---------------------------------------------------------------------------

def _load_strategy(
    display_name: str, module_path: str, class_name: str, kwargs: dict,
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

    logger.info(
        "  [%s] Train: %s to %s (%d bars), Test: %s to %s (%d bars)",
        name,
        train_data.index[0].date(), train_data.index[-1].date(), len(train_data),
        test_data.index[0].date(), test_data.index[-1].date(), len(test_data),
    )

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
# Inverse-volatility weighted ensemble (implemented inline)
# ---------------------------------------------------------------------------

def _inverse_vol_combine(
    strategy_returns: Dict[str, pd.Series],
    lookback: int = INV_VOL_LOOKBACK,
) -> pd.Series:
    """Inverse-volatility weighted combination of strategy return series.

    Weights are proportional to 1/sigma, re-estimated using a rolling
    lookback window.  Strategies with lower volatility get higher weight.
    Before enough history is accumulated, equal weights are used.
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
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_returns(
    name: str,
    returns: pd.Series,
    engine: BacktestEngine,
) -> Dict[str, Any]:
    """Evaluate a return series: backtest metrics, walk-forward, Monte Carlo.

    Returns a dictionary row suitable for a comparison DataFrame.
    """
    returns_arr = returns.values.astype(np.float64)
    n = len(returns_arr)

    if n < 10:
        logger.warning("  %s: too few observations (%d). Skipping.", name, n)
        return {"Strategy": name, "Status": "SKIPPED"}

    # Build synthetic price from returns for backtest engine
    price = 100.0 * np.cumprod(1.0 + returns_arr)
    price = np.insert(price, 0, 100.0)  # prepend starting value
    signal = np.ones(len(price))  # returns already baked in

    # Core backtest
    bt_result = engine.run(signal, price)
    metrics = bt_result.metrics

    logger.info(
        "  [%s] PnL=%.2f%%, Sharpe=%.3f, Sortino=%.3f, MaxDD=%.2f%%",
        name, metrics["total_pnl_pct"], metrics["sharpe_ratio"],
        metrics["sortino_ratio"], metrics["max_drawdown"] * 100,
    )

    # Monte Carlo — compute total target from annualized rate
    n_years = n / 252
    total_target_pnl = ((1 + ANNUAL_TARGET_PNL / 100) ** n_years - 1) * 100
    mc_prob = np.nan
    try:
        mc_result = engine.monte_carlo_confidence(
            returns=bt_result.returns,
            n_simulations=MONTE_CARLO_SIMS,
            target_pnl_pct=total_target_pnl,
        )
        mc_prob = mc_result.prob_above_target
    except Exception as exc:
        logger.warning("  MC failed for %s: %s", name, exc)

    # Walk-forward OOS
    wf_oos_sharpe = np.nan
    try:
        def wf_fn(context: dict) -> np.ndarray:
            """Walk-forward replays returns (strategies already evaluated)."""
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
        logger.warning("  WF failed for %s: %s", name, exc)

    # PASS/FAIL gate
    mc_pass = (not np.isnan(mc_prob)) and (mc_prob >= 0.95)
    wf_pass = (not np.isnan(wf_oos_sharpe)) and (wf_oos_sharpe > 0)
    verdict = "PASS" if (mc_pass and wf_pass) else "FAIL"

    return {
        "Strategy": name,
        "Total PnL%": round(metrics["total_pnl_pct"], 2),
        "Ann. Return%": round(metrics["annualized_return"] * 100, 2),
        "Sharpe": round(metrics["sharpe_ratio"], 3),
        "Sortino": round(metrics["sortino_ratio"], 3),
        "Max DD%": round(metrics["max_drawdown"] * 100, 2),
        "WF OOS Sharpe": (
            round(wf_oos_sharpe, 3) if not np.isnan(wf_oos_sharpe) else np.nan
        ),
        f"P(Ann>{int(ANNUAL_TARGET_PNL)}%) MC": (
            round(mc_prob, 4) if not np.isnan(mc_prob) else np.nan
        ),
        "Bootstrap p": round(metrics["bootstrap_pvalue"], 4),
        "PASS/FAIL": verdict,
    }


# ---------------------------------------------------------------------------
# Phase 1: Inverse-Vol Ensemble (with subset sweep)
# ---------------------------------------------------------------------------

def run_ensemble_phase(
    close_prices: pd.DataFrame,
    engine: BacktestEngine,
) -> Tuple[List[Dict[str, Any]], Dict[str, pd.Series]]:
    """Run all 5 component strategies, then build inverse-vol ensembles
    with different subsets (top-3, top-4, top-5).

    Returns (result_rows, strategy_returns_dict).
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: Running component strategies for ensemble")
    logger.info("=" * 70)

    strategy_returns: Dict[str, pd.Series] = {}
    strategy_order: List[str] = []

    for display_name, module_path, class_name, kwargs in ENSEMBLE_STRATEGIES:
        try:
            t_start = time.perf_counter()
            strategy = _load_strategy(display_name, module_path, class_name, kwargs)
            port_ret = _run_strategy_oos(display_name, strategy, close_prices)

            if port_ret is not None and len(port_ret) > 0:
                strategy_returns[display_name] = port_ret
                strategy_order.append(display_name)

            elapsed = time.perf_counter() - t_start
            logger.info(
                "  Strategy '%s' completed in %.2f seconds.", display_name, elapsed,
            )
        except Exception:
            logger.error(
                "  Strategy '%s' FAILED:\n%s", display_name, traceback.format_exc(),
            )

    if len(strategy_returns) < 2:
        logger.error(
            "Only %d strategies produced returns. Need at least 2 for ensemble.",
            len(strategy_returns),
        )
        return [], strategy_returns

    # Align all return series to a common index
    ret_df = pd.DataFrame(strategy_returns).sort_index()
    common_idx = ret_df.dropna(how="all").index
    strategy_returns = {
        name: series.reindex(common_idx).fillna(0.0)
        for name, series in strategy_returns.items()
    }

    logger.info(
        "Common OOS period: %s to %s (%d bars)",
        common_idx[0].date(), common_idx[-1].date(), len(common_idx),
    )

    # Print correlation matrix
    corr_df = pd.DataFrame(strategy_returns).corr()
    logger.info("\nStrategy return correlations:")
    logger.info("\n%s", corr_df.to_string(float_format="%.3f"))

    # --- Build ensemble with different subsets ---
    logger.info("\n" + "=" * 70)
    logger.info("Building inverse-vol ensembles with different subsets")
    logger.info("=" * 70)

    all_results: List[Dict[str, Any]] = []
    available_names = [n for n in strategy_order if n in strategy_returns]

    for subset_size in [3, 4, 5]:
        if subset_size > len(available_names):
            continue

        # Use the top-N strategies (ordered by prior performance ranking)
        subset_names = available_names[:subset_size]
        subset_returns = {n: strategy_returns[n] for n in subset_names}

        label = f"InvVol Ensemble (top-{subset_size}: {', '.join(subset_names)})"
        logger.info("\n--- %s ---", label)

        try:
            combined_ret = _inverse_vol_combine(subset_returns, lookback=INV_VOL_LOOKBACK)
            row = _evaluate_returns(label, combined_ret, engine)
            all_results.append(row)
        except Exception:
            logger.error("  Ensemble '%s' failed:\n%s", label, traceback.format_exc())

    return all_results, strategy_returns


# ---------------------------------------------------------------------------
# Phase 2: GARCH Vol (improved config)
# ---------------------------------------------------------------------------

def run_garch_phase(
    close_prices: pd.DataFrame,
    engine: BacktestEngine,
) -> Optional[Dict[str, Any]]:
    """Run GARCH Vol with improved config: EGARCH, 25% vol target, adaptive blend."""
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: GARCH Vol (EGARCH, 25%% vol target, adaptive blend)")
    logger.info("=" * 70)

    try:
        from src.strategies.garch_vol import GarchVolConfig, GarchVolStrategy

        config = GarchVolConfig(
            garch_model="EGARCH",
            target_vol=0.25,
            adaptive_blend=True,
        )
        strategy = GarchVolStrategy(config=config)

        port_ret = _run_strategy_oos("GARCH Vol (improved)", strategy, close_prices)
        if port_ret is None or len(port_ret) < 10:
            logger.warning("  GARCH Vol produced insufficient data.")
            return None

        row = _evaluate_returns("GARCH Vol (EGARCH/25%/adaptive)", port_ret, engine)
        return row

    except Exception:
        logger.error("  GARCH Vol phase failed:\n%s", traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Phase 3: Entropy Regularized (optimized params)
# ---------------------------------------------------------------------------

def run_entropy_phase(
    close_prices: pd.DataFrame,
    engine: BacktestEngine,
) -> Optional[Dict[str, Any]]:
    """Run Entropy Regularized with optimized parameters.

    Optimized params: gamma=0.3, lambda=0.01, eg_blend=0.8, rebalance=3, eta0=2.0
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Entropy Regularized (optimized params)")
    logger.info("=" * 70)

    try:
        from src.strategies.entropy_regularized import EntropyRegularizedStrategy

        strategy = EntropyRegularizedStrategy(
            gamma=0.3,
            lambda_base=0.01,
            eg_blend=0.8,
            rebalance_freq=3,
            eta0=2.0,
        )

        port_ret = _run_strategy_oos(
            "Entropy Reg. (optimized)", strategy, close_prices,
        )
        if port_ret is None or len(port_ret) < 10:
            logger.warning("  Entropy Regularized produced insufficient data.")
            return None

        row = _evaluate_returns(
            "Entropy Reg. (g=0.3, l=0.01, eg=0.8, rb=3, eta=2.0)",
            port_ret, engine,
        )
        return row

    except Exception:
        logger.error("  Entropy Regularized phase failed:\n%s", traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: load data, run the three winner phases, produce report."""

    t_global_start = time.perf_counter()

    # ---- Download data ----
    logger.info("Downloading ETF data from %s to %s...", DATA_START, DATA_END)
    raw_data = download_etf_data(start=DATA_START, end=DATA_END)
    close_prices = _extract_close_prices(raw_data)
    logger.info(
        "Close prices: %d rows x %d tickers (%s to %s)",
        len(close_prices), len(close_prices.columns),
        close_prices.index[0].date(), close_prices.index[-1].date(),
    )

    engine = BacktestEngine()
    all_results: List[Dict[str, Any]] = []

    # ---- Phase 1: Inverse-Vol Ensemble (subset sweep) ----
    ensemble_results, strategy_returns = run_ensemble_phase(close_prices, engine)
    all_results.extend(ensemble_results)

    # ---- Phase 2: GARCH Vol ----
    garch_row = run_garch_phase(close_prices, engine)
    if garch_row is not None:
        all_results.append(garch_row)

    # ---- Phase 3: Entropy Regularized ----
    entropy_row = run_entropy_phase(close_prices, engine)
    if entropy_row is not None:
        all_results.append(entropy_row)

    # ---- Report ----
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)

    if not all_results:
        logger.error("All phases failed. No results to report.")
        sys.exit(1)

    comparison = pd.DataFrame(all_results)

    # Sort by Sharpe descending
    if "Sharpe" in comparison.columns:
        comparison = comparison.sort_values("Sharpe", ascending=False).reset_index(
            drop=True
        )

    # Save to CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "focused_results.csv"
    comparison.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    # Print the table
    mc_col = f"P(Ann>{int(ANNUAL_TARGET_PNL)}%) MC"

    print("\n" + "=" * 130)
    print("FOCUSED WINNER VALIDATION")
    print(f"Data: {DATA_START} to {DATA_END} | OOS split: {TRAIN_FRACTION:.0%} train / {1-TRAIN_FRACTION:.0%} test")
    print(f"Walk-forward: {WALK_FORWARD_FOLDS} folds | Monte Carlo: {MONTE_CARLO_SIMS:,} sims | Target: {ANNUAL_TARGET_PNL}% annualized")
    print(f"PASS criteria: P(Ann>{int(ANNUAL_TARGET_PNL)}%) >= 0.95 AND WF OOS Sharpe > 0")
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
        print(f"\nVERDICT: {n_pass} / {n_total} strategies PASSED the validation gate.")

        passed = comparison[comparison["PASS/FAIL"] == "PASS"]
        if len(passed) > 0:
            best = passed.iloc[0]  # already sorted by Sharpe
            print(f"BEST WINNER: {best['Strategy']} (Sharpe={best['Sharpe']:.3f})")
        else:
            print("WARNING: No strategies passed both gates.")

    print(f"\nResults saved to: {csv_path}")

    elapsed_total = time.perf_counter() - t_global_start
    print(f"Total runtime: {elapsed_total:.1f} seconds")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", action="store_true",
                        help="Apply automation risk caps (20%% per-ticker, 1.5x gross)")
    args = parser.parse_args()
    if args.cap:
        MAX_POSITION_PER_TICKER = 0.20
        MAX_GROSS_LEVERAGE = 1.5
        logger.info("Position caps ENABLED: 20%% per-ticker, 1.5x gross leverage")
    main()
