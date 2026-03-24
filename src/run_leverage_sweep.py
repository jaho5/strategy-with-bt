"""Leverage sweep on the winning Inverse-Vol Ensemble.

Sweeps leverage multipliers on the 5-strategy Inverse-Vol Ensemble to find
the optimal leverage point that achieves P(Ann return > 45%) >= 95% while
minimising max drawdown.

The key insight: if the base ensemble has a high Sharpe ratio, we can lever
up or down to hit any return target.  Lower leverage reduces drawdown at the
cost of lower expected returns; higher leverage amplifies both.

Leverage multipliers tested:
    0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 1.75x, 2.0x, 2.5x, 3.0x

For each leverage level:
    - Annualized return
    - Sharpe ratio (roughly constant across leverage levels)
    - Max drawdown (increases with leverage)
    - Monte Carlo P(Ann return > 45%)  -- target = (1.45)^n_years - 1
    - Walk-forward OOS Sharpe

Optimal leverage selection:
    - Must achieve P(Ann > 45%) >= 95%
    - Minimise Max DD subject to the above constraint

Saves results to: reports/leverage_sweep.csv

Usage:
    uv run python -m src.run_leverage_sweep
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

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
WALK_FORWARD_FOLDS = 5
MONTE_CARLO_SIMS = 10_000
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Annual return target for the probability gate
ANNUAL_RETURN_TARGET = 0.45  # 45%

# Minimum MC probability to pass
MC_PROB_THRESHOLD = 0.95

# Leverage multipliers to sweep
LEVERAGE_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

# Inverse-vol lookback
INVERSE_VOL_LOOKBACK = 63

# Top-5 strategies to ensemble
STRATEGY_REGISTRY: List[Tuple[str, str, str, dict]] = [
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

_TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Data helpers (mirrored from run_ensemble.py)
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
# Signal -> Portfolio return conversion (mirrored from run_ensemble.py)
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
# Strategy loading and return generation
# ---------------------------------------------------------------------------

def _load_strategy(
    display_name: str, module_path: str, class_name: str, kwargs: dict,
) -> Any:
    """Import and instantiate a single strategy."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def _run_strategy_full_period(
    name: str,
    strategy: Any,
    close_prices: pd.DataFrame,
) -> Optional[pd.Series]:
    """Fit on train, generate OOS signals, return daily portfolio return."""
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

    logger.info(
        "  [%s] OOS return series: %d bars, cumulative=%.2f%%",
        name, len(port_ret),
        (np.prod(1.0 + port_ret.values) - 1.0) * 100,
    )
    return port_ret


# ---------------------------------------------------------------------------
# Inverse-vol combination (the winning method from run_ensemble.py)
# ---------------------------------------------------------------------------

def _inverse_vol_combine(
    strategy_returns: Dict[str, pd.Series],
    lookback: int = 63,
) -> pd.Series:
    """Inverse-volatility weighted combination.

    Weights are proportional to 1/sigma, re-estimated using a rolling
    lookback window. Strategies with lower volatility get higher weight.
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
# Leverage application
# ---------------------------------------------------------------------------

def _apply_leverage(
    base_returns: pd.Series,
    leverage: float,
) -> pd.Series:
    """Apply a constant leverage multiplier to a daily return series.

    Leveraged return on day t = leverage * base_return_t

    This models daily rebalanced leverage (like a leveraged ETF or a
    futures-based overlay).  The compounding effect means that
    annualized return != leverage * base annualized return, and max DD
    scales more than linearly with leverage.
    """
    return base_returns * leverage


# ---------------------------------------------------------------------------
# Monte Carlo for annualized return target
# ---------------------------------------------------------------------------

def _monte_carlo_ann_return(
    returns: np.ndarray,
    n_simulations: int = 10_000,
    ann_return_target: float = 0.45,
    rng_seed: int = 42,
) -> Tuple[float, float, float]:
    """Block-bootstrap Monte Carlo for P(annualized return > target).

    Uses the same block-bootstrap approach as BacktestEngine.monte_carlo_confidence
    but evaluates against an annualized return target rather than a cumulative
    PnL percentage.

    The target for the full OOS period is: (1 + ann_return_target)^n_years - 1

    Returns:
        (prob_above_target, mean_ann_return, median_ann_return)
    """
    # Strip leading zeros
    if len(returns) > 0 and returns[0] == 0.0:
        returns = returns[1:]

    n = len(returns)
    if n < 2:
        return np.nan, np.nan, np.nan

    n_years = n / _TRADING_DAYS_PER_YEAR
    # The cumulative return needed to achieve the annualized target
    cum_target = (1.0 + ann_return_target) ** n_years - 1.0

    rng = np.random.default_rng(rng_seed)
    block_size = max(1, int(np.sqrt(n)))
    n_blocks = int(np.ceil(n / block_size))

    terminal_cum_returns = np.empty(n_simulations)
    ann_returns = np.empty(n_simulations)

    for sim in range(n_simulations):
        starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        blocks = [returns[s: s + block_size] for s in starts]
        path = np.concatenate(blocks)[:n]
        cum = np.prod(1.0 + path) - 1.0
        terminal_cum_returns[sim] = cum
        # Annualized return
        total_growth = 1.0 + cum
        if total_growth > 0 and n_years > 0:
            ann_returns[sim] = total_growth ** (1.0 / n_years) - 1.0
        else:
            ann_returns[sim] = -1.0

    prob_above = float(np.mean(terminal_cum_returns >= cum_target))
    mean_ann = float(np.mean(ann_returns))
    median_ann = float(np.median(ann_returns))

    return prob_above, mean_ann, median_ann


# ---------------------------------------------------------------------------
# Walk-forward Sharpe
# ---------------------------------------------------------------------------

def _walk_forward_sharpe(
    returns: np.ndarray,
    engine: BacktestEngine,
    n_splits: int = 5,
    train_pct: float = 0.70,
) -> float:
    """Compute walk-forward OOS Sharpe on a return series.

    Builds a synthetic price from the returns and runs the backtest engine's
    walk-forward test with a pass-through signal.
    """
    price = 100.0 * np.cumprod(1.0 + returns)
    price = np.insert(price, 0, 100.0)

    def wf_fn(context: dict) -> np.ndarray:
        expected_len = context["test_end"] - context["test_start"]
        return np.ones(expected_len)

    try:
        wf_result = engine.walk_forward_test(
            strategy_fn=wf_fn,
            prices=price,
            n_splits=n_splits,
            train_pct=train_pct,
        )
        return wf_result.aggregate_metrics.get("sharpe_ratio", np.nan)
    except Exception as exc:
        logger.warning("  Walk-forward failed: %s", exc)
        return np.nan


# ---------------------------------------------------------------------------
# Evaluate a single leverage level
# ---------------------------------------------------------------------------

def _evaluate_leverage(
    leverage: float,
    base_returns: pd.Series,
    engine: BacktestEngine,
) -> Dict[str, Any]:
    """Evaluate a leverage level: return, Sharpe, DD, MC prob, WF Sharpe."""
    lev_returns = _apply_leverage(base_returns, leverage)
    lev_arr = lev_returns.values.astype(np.float64)
    n = len(lev_arr)

    if n < 10:
        return {
            "Leverage": leverage,
            "Status": "SKIPPED",
        }

    # --- Core metrics ---
    cum = np.prod(1.0 + lev_arr) - 1.0
    n_years = n / _TRADING_DAYS_PER_YEAR
    if n_years > 0 and (1.0 + cum) > 0:
        ann_return = (1.0 + cum) ** (1.0 / n_years) - 1.0
    else:
        ann_return = 0.0

    # Sharpe
    mean_ret = np.mean(lev_arr)
    std_ret = np.std(lev_arr, ddof=1) if n > 1 else np.nan
    if std_ret and std_ret > 0:
        sharpe = (mean_ret / std_ret) * np.sqrt(_TRADING_DAYS_PER_YEAR)
    else:
        sharpe = np.nan

    # Max drawdown
    equity = np.cumprod(1.0 + lev_arr)
    running_max = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / running_max
    max_dd = float(np.max(drawdown)) if n > 0 else 0.0

    # Sortino
    downside = lev_arr.copy()
    downside[downside > 0] = 0.0
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std > 0:
        sortino = (mean_ret / downside_std) * np.sqrt(_TRADING_DAYS_PER_YEAR)
    else:
        sortino = np.nan

    # Calmar
    calmar = ann_return / max_dd if max_dd > 0 else np.nan

    logger.info(
        "  [%.2fx] Ann=%.1f%%, Sharpe=%.3f, MaxDD=%.1f%%",
        leverage, ann_return * 100, sharpe, max_dd * 100,
    )

    # --- Monte Carlo P(Ann return > 45%) ---
    mc_prob, mc_mean_ann, mc_median_ann = _monte_carlo_ann_return(
        lev_arr,
        n_simulations=MONTE_CARLO_SIMS,
        ann_return_target=ANNUAL_RETURN_TARGET,
    )
    logger.info(
        "    MC P(Ann>%.0f%%) = %.4f  (mean ann=%.1f%%, median ann=%.1f%%)",
        ANNUAL_RETURN_TARGET * 100, mc_prob,
        mc_mean_ann * 100, mc_median_ann * 100,
    )

    # --- Walk-forward Sharpe ---
    wf_sharpe = _walk_forward_sharpe(
        lev_arr, engine,
        n_splits=WALK_FORWARD_FOLDS,
        train_pct=TRAIN_FRACTION,
    )
    logger.info("    WF OOS Sharpe = %.3f", wf_sharpe)

    # --- Pass/fail ---
    mc_pass = (not np.isnan(mc_prob)) and (mc_prob >= MC_PROB_THRESHOLD)

    return {
        "Leverage": leverage,
        "Ann. Return%": round(ann_return * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Max DD%": round(max_dd * 100, 2),
        "Calmar": round(calmar, 3) if not np.isnan(calmar) else np.nan,
        "MC P(Ann>45%)": round(mc_prob, 4) if not np.isnan(mc_prob) else np.nan,
        "MC Mean Ann%": round(mc_mean_ann * 100, 2) if not np.isnan(mc_mean_ann) else np.nan,
        "MC Median Ann%": round(mc_median_ann * 100, 2) if not np.isnan(mc_median_ann) else np.nan,
        "WF OOS Sharpe": round(wf_sharpe, 3) if not np.isnan(wf_sharpe) else np.nan,
        "P(Ann>45%)>=95%": "YES" if mc_pass else "NO",
    }


# ---------------------------------------------------------------------------
# Find optimal leverage
# ---------------------------------------------------------------------------

def _find_optimal_leverage(
    results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the optimal leverage: must pass MC gate, minimise Max DD.

    Selection criteria:
    1. Must achieve P(Ann > 45%) >= 95%
    2. Among those that pass, pick the one with lowest Max DD
       (this is the "efficient" leverage point)
    """
    passing = [
        r for r in results
        if r.get("P(Ann>45%)>=95%") == "YES"
    ]

    if not passing:
        logger.warning("No leverage level achieves P(Ann>45%%) >= 95%%.")
        return None

    # Sort by Max DD ascending -- lowest DD wins
    passing.sort(key=lambda r: r.get("Max DD%", float("inf")))
    optimal = passing[0]

    logger.info(
        "OPTIMAL LEVERAGE: %.2fx  (Ann=%.1f%%, MaxDD=%.1f%%, MC P=%.4f)",
        optimal["Leverage"],
        optimal["Ann. Return%"],
        optimal["Max DD%"],
        optimal["MC P(Ann>45%)"],
    )

    return optimal


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: run strategies, build inverse-vol ensemble, sweep leverage."""

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

    # ---- Phase 1: Run each strategy individually ----
    logger.info("=" * 70)
    logger.info("PHASE 1: Running 5 component strategies (walk-forward OOS)")
    logger.info("=" * 70)

    strategy_returns: Dict[str, pd.Series] = {}

    for display_name, module_path, class_name, kwargs in STRATEGY_REGISTRY:
        try:
            t_start = time.perf_counter()
            strategy = _load_strategy(display_name, module_path, class_name, kwargs)

            port_ret = _run_strategy_full_period(
                display_name, strategy, close_prices,
            )
            if port_ret is not None and len(port_ret) > 0:
                strategy_returns[display_name] = port_ret

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
        sys.exit(1)

    logger.info(
        "\n%d / %d strategies produced return series.",
        len(strategy_returns), len(STRATEGY_REGISTRY),
    )

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

    # ---- Phase 2: Build inverse-vol ensemble (base, 1x leverage) ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Building Inverse-Vol Ensemble (base returns)")
    logger.info("=" * 70)

    base_returns = _inverse_vol_combine(
        strategy_returns, lookback=INVERSE_VOL_LOOKBACK,
    )

    # Report base stats
    base_arr = base_returns.values.astype(np.float64)
    n = len(base_arr)
    n_years = n / _TRADING_DAYS_PER_YEAR
    cum = np.prod(1.0 + base_arr) - 1.0
    if n_years > 0 and (1.0 + cum) > 0:
        base_ann = (1.0 + cum) ** (1.0 / n_years) - 1.0
    else:
        base_ann = 0.0
    base_std = np.std(base_arr, ddof=1)
    base_sharpe = (np.mean(base_arr) / base_std * np.sqrt(_TRADING_DAYS_PER_YEAR)
                   if base_std > 0 else np.nan)
    base_equity = np.cumprod(1.0 + base_arr)
    base_peak = np.maximum.accumulate(base_equity)
    base_dd = float(np.max(1.0 - base_equity / base_peak))

    logger.info(
        "Base Inverse-Vol Ensemble: Ann=%.1f%%, Sharpe=%.3f, MaxDD=%.1f%%",
        base_ann * 100, base_sharpe, base_dd * 100,
    )
    logger.info(
        "OOS period: %.1f years (%d trading days)",
        n_years, n,
    )

    # ---- Phase 3: Leverage sweep ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Leverage sweep (%d levels)", len(LEVERAGE_MULTIPLIERS))
    logger.info("=" * 70)

    engine = BacktestEngine()
    all_results: List[Dict[str, Any]] = []

    for leverage in LEVERAGE_MULTIPLIERS:
        logger.info("\n--- Leverage %.2fx ---", leverage)
        try:
            row = _evaluate_leverage(leverage, base_returns, engine)
            all_results.append(row)
        except Exception:
            logger.error(
                "  Leverage %.2fx FAILED:\n%s", leverage, traceback.format_exc(),
            )

    if not all_results:
        logger.error("All leverage levels failed. No results to report.")
        sys.exit(1)

    # ---- Phase 4: Find optimal leverage ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: Optimal leverage selection")
    logger.info("=" * 70)

    optimal = _find_optimal_leverage(all_results)

    # ---- Phase 5: Report ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: Results")
    logger.info("=" * 70)

    results_df = pd.DataFrame(all_results)

    # Save to CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "leverage_sweep.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    # Print table
    print("\n" + "=" * 130)
    print("LEVERAGE SWEEP: Inverse-Vol Ensemble")
    print(f"Data: {DATA_START} to {DATA_END} | OOS period: {TRAIN_FRACTION:.0%} train / {1 - TRAIN_FRACTION:.0%} test")
    print(f"Base ensemble: Ann={base_ann * 100:.1f}%, Sharpe={base_sharpe:.3f}, MaxDD={base_dd * 100:.1f}%")
    print(f"Target: P(Ann return > {ANNUAL_RETURN_TARGET * 100:.0f}%) >= {MC_PROB_THRESHOLD * 100:.0f}%")
    print(f"Monte Carlo: {MONTE_CARLO_SIMS:,} simulations | Walk-forward: {WALK_FORWARD_FOLDS} folds")
    print("=" * 130)

    print("\n--- LEVERAGE SWEEP RESULTS ---")
    print(
        tabulate(
            results_df,
            headers="keys",
            tablefmt="pretty",
            showindex=False,
            floatfmt=".4f",
        )
    )

    # Highlight optimal
    print("\n" + "=" * 130)
    if optimal is not None:
        print(
            f"OPTIMAL LEVERAGE: {optimal['Leverage']:.2f}x"
            f"  |  Ann. Return = {optimal['Ann. Return%']:.1f}%"
            f"  |  Sharpe = {optimal['Sharpe']:.3f}"
            f"  |  Max DD = {optimal['Max DD%']:.1f}%"
            f"  |  MC P(Ann>45%) = {optimal['MC P(Ann>45%)']:.4f}"
        )
        print()
        print("Rationale: lowest Max DD among leverage levels that achieve")
        print(f"P(Ann return > {ANNUAL_RETURN_TARGET * 100:.0f}%) >= {MC_PROB_THRESHOLD * 100:.0f}%.")
        print()
        print("Key insight: Sharpe ratio is roughly constant across leverage levels.")
        print(f"  At 1.0x leverage: {base_ann * 100:.1f}% annual, {base_dd * 100:.1f}% max DD")
        if optimal["Leverage"] != 1.0:
            print(
                f"  At {optimal['Leverage']:.2f}x leverage: "
                f"{optimal['Ann. Return%']:.1f}% annual, "
                f"{optimal['Max DD%']:.1f}% max DD"
            )
        print(
            "  Leverage scales returns linearly (pre-compounding) but "
            "drawdowns scale more than linearly."
        )
    else:
        print("NO LEVERAGE LEVEL achieves P(Ann>45%) >= 95%.")
        print("Consider:")
        print("  - The base ensemble may need higher Sharpe to reliably hit 45% annual")
        print("  - Try higher leverage multipliers (> 3x)")
        print("  - Or relax the MC probability threshold")
    print("=" * 130)

    print(f"\nResults saved to: {csv_path}")
    elapsed_total = time.perf_counter() - t_global_start
    print(f"Total runtime: {elapsed_total:.1f} seconds")


if __name__ == "__main__":
    main()
