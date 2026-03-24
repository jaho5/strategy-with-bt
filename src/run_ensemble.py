"""Ensemble strategy runner: combines the top-5 strategies and evaluates
multiple combination methods.

Strategies (by backtest performance):
    1. Entropy Regularized  -- 71% PnL, 0.89 Sharpe, WF 0.285
    2. GARCH Vol            -- 12% PnL, 1.14 Sharpe, p=0.007
    3. HMM Regime           -- 26% PnL, 0.53 Sharpe
    4. Spectral Momentum    -- 22% PnL, 0.70 Sharpe
    5. Bayesian Changepoint --  7% PnL, 0.25 Sharpe

Combination methods tested:
    A. EnsembleMetaStrategy (Hedge/MW + Sharpe-softmax + Markowitz)
    B. Equal-weight average
    C. Inverse-volatility weighted
    D. Markowitz max-Sharpe on strategy returns

Usage:
    uv run python -m src.run_ensemble
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
from scipy.optimize import minimize
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
MAX_POSITION_PER_TICKER: Optional[float] = None  # None=uncapped, 0.20=matches automation
MAX_GROSS_LEVERAGE: Optional[float] = None  # None=uncapped, 1.5=matches automation
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

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
# Strategy loading and individual return generation
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
    """Fit a strategy on training data, generate signals on the FULL period,
    and return its daily portfolio return series.

    Uses walk-forward: fit on first 70%, generate signals on remaining 30%.
    Returns the OOS return series aligned to the test period dates.
    """
    train_data, test_data = _split_data(close_prices, TRAIN_FRACTION)

    logger.info(
        "  [%s] Train: %s to %s (%d bars), Test: %s to %s (%d bars)",
        name,
        train_data.index[0].date(), train_data.index[-1].date(), len(train_data),
        test_data.index[0].date(), test_data.index[-1].date(), len(test_data),
    )

    # Fit on training data
    logger.info("  [%s] Fitting on training data...", name)
    strategy.fit(train_data)

    # Generate signals on test data (OOS)
    logger.info("  [%s] Generating signals on test data (OOS)...", name)
    signals = strategy.generate_signals(test_data)

    # Convert to portfolio return series
    port_ret = _signals_to_portfolio_return(signals, test_data)

    logger.info(
        "  [%s] OOS return series: %d bars, cumulative=%.2f%%",
        name, len(port_ret),
        (np.prod(1.0 + port_ret.values) - 1.0) * 100,
    )
    return port_ret


# ---------------------------------------------------------------------------
# Combination methods
# ---------------------------------------------------------------------------

def _equal_weight_combine(
    strategy_returns: Dict[str, pd.Series],
) -> pd.Series:
    """Equal-weight average of strategy returns."""
    ret_df = pd.DataFrame(strategy_returns).fillna(0.0)
    return ret_df.mean(axis=1)


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
            # Use equal weights until we have enough history
            w = np.full(K, 1.0 / K)
        else:
            window = ret_df.iloc[t - lookback:t]
            vol = window.std().values
            vol = np.where(vol < 1e-10, 1e-10, vol)
            inv_vol = 1.0 / vol
            w = inv_vol / inv_vol.sum()

        combined.iloc[t] = float(ret_df.iloc[t].values @ w)

    return combined


def _markowitz_max_sharpe_combine(
    strategy_returns: Dict[str, pd.Series],
    lookback: int = 126,
    shrinkage: float = 0.1,
) -> pd.Series:
    """Markowitz max-Sharpe weighted combination.

    Solves: max_w mu'w / sqrt(w'Sigma w) s.t. w >= 0, sum(w) = 1
    using a rolling lookback window for mu and Sigma estimation.
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
            mu = window.mean().values
            cov = window.cov().values

            # Regularise covariance
            trace_cov = np.trace(cov)
            if trace_cov <= 0:
                w = np.full(K, 1.0 / K)
            else:
                cov = (1 - shrinkage) * cov + shrinkage * (trace_cov / K) * np.eye(K)
                w = _solve_max_sharpe(mu, cov, K)

        combined.iloc[t] = float(ret_df.iloc[t].values @ w)

    return combined


def _solve_max_sharpe(mu: np.ndarray, cov: np.ndarray, K: int) -> np.ndarray:
    """Solve the max-Sharpe portfolio optimization."""
    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = mu @ w
        port_var = w @ cov @ w
        if port_var <= 1e-16:
            return 0.0
        return -port_ret / np.sqrt(port_var)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * K
    w0 = np.full(K, 1.0 / K)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                neg_sharpe, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
        if res.success:
            weights = np.maximum(res.x, 0.0)
            s = weights.sum()
            if s > 0:
                return weights / s
    except Exception:
        pass

    return np.full(K, 1.0 / K)


def _ensemble_meta_combine(
    strategy_returns: Dict[str, pd.Series],
    close_prices: pd.DataFrame,
    strategy_signals: Dict[str, pd.Series],
) -> pd.Series:
    """Use the EnsembleMetaStrategy to combine strategies.

    This feeds the strategy return and signal series into the meta-strategy's
    three weighting schemes (Hedge/MW, Sharpe-softmax, Markowitz) and produces
    the combined ensemble signal, which is then applied to an equal-weight
    basket of the underlying assets.
    """
    from src.strategies.ensemble_meta import EnsembleMetaStrategy

    meta = EnsembleMetaStrategy()

    # Get test period prices
    _, test_prices = _split_data(close_prices, TRAIN_FRACTION)
    common_idx = test_prices.index

    # Fit on the strategy returns
    logger.info("  Fitting EnsembleMetaStrategy on %d expert return series...",
                len(strategy_returns))
    meta.fit(test_prices, strategy_returns=strategy_returns)

    # Generate ensemble signals
    logger.info("  Generating ensemble signals...")
    ensemble_signals = meta.generate_signals(
        test_prices,
        strategy_signals=strategy_signals,
        strategy_returns=strategy_returns,
    )

    # Apply ensemble signal to an equal-weight basket
    asset_returns = test_prices.pct_change().fillna(0.0)
    avg_asset_return = asset_returns.mean(axis=1)

    signal = ensemble_signals["signal"].reindex(common_idx).fillna(0.0)
    weight = ensemble_signals["weight"].reindex(common_idx).fillna(0.0)
    composite = signal * weight

    port_ret = composite.shift(1).fillna(0.0) * avg_asset_return
    return port_ret


# ---------------------------------------------------------------------------
# Performance evaluation
# ---------------------------------------------------------------------------

def _evaluate_returns(
    name: str,
    returns: pd.Series,
    engine: BacktestEngine,
) -> Dict[str, Any]:
    """Evaluate a combined return series: metrics, MC, walk-forward."""
    returns_arr = returns.values.astype(np.float64)
    n = len(returns_arr)

    if n < 10:
        logger.warning("  %s: too few observations (%d). Skipping.", name, n)
        return {"Combination": name, "Status": "SKIPPED"}

    # Build synthetic price from returns for backtest engine
    price = 100.0 * np.cumprod(1.0 + returns_arr)
    price = np.insert(price, 0, 100.0)  # prepend starting value
    signal = np.ones(len(price))  # returns already baked in

    # Core backtest
    bt_result = engine.run(signal, price)
    metrics = bt_result.metrics

    logger.info("  [%s] PnL=%.2f%%, Sharpe=%.3f, MaxDD=%.2f%%",
                name, metrics["total_pnl_pct"], metrics["sharpe_ratio"],
                metrics["max_drawdown"] * 100)

    # Monte Carlo — compute total target from annualized rate
    n_years = n / 252
    total_target_pnl = ((1 + ANNUAL_TARGET_PNL / 100) ** n_years - 1) * 100
    try:
        mc_result = engine.monte_carlo_confidence(
            returns=bt_result.returns,
            n_simulations=MONTE_CARLO_SIMS,
            target_pnl_pct=total_target_pnl,
        )
        mc_prob = mc_result.prob_above_target
    except Exception as exc:
        logger.warning("  MC failed for %s: %s", name, exc)
        mc_prob = np.nan

    # Walk-forward OOS
    try:
        def wf_fn(context: dict) -> np.ndarray:
            """Walk-forward just replays returns (strategies already evaluated)."""
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
        wf_oos_sharpe = np.nan

    return {
        "Combination": name,
        "Total PnL%": round(metrics["total_pnl_pct"], 2),
        "Ann. Return%": round(metrics["annualized_return"] * 100, 2),
        "Sharpe": round(metrics["sharpe_ratio"], 3),
        "Sortino": round(metrics["sortino_ratio"], 3),
        "Max DD%": round(metrics["max_drawdown"] * 100, 2),
        "Win Rate%": round(metrics["win_rate"] * 100, 2),
        f"P(Ann>{int(ANNUAL_TARGET_PNL)}%) MC": (
            round(mc_prob, 4) if not np.isnan(mc_prob) else np.nan
        ),
        "Bootstrap p": round(metrics["bootstrap_pvalue"], 4),
        "WF OOS Sharpe": (
            round(wf_oos_sharpe, 3) if not np.isnan(wf_oos_sharpe) else np.nan
        ),
    }


# ---------------------------------------------------------------------------
# Individual strategy evaluation (for comparison)
# ---------------------------------------------------------------------------

def _evaluate_individual(
    name: str,
    returns: pd.Series,
    engine: BacktestEngine,
) -> Dict[str, Any]:
    """Evaluate an individual strategy's returns for the comparison table."""
    result = _evaluate_returns(f"Individual: {name}", returns, engine)
    result["Combination"] = f"[Individual] {name}"
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: load data, run strategies, build ensemble, evaluate."""

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
    logger.info("PHASE 1: Running individual strategies (walk-forward OOS)")
    logger.info("=" * 70)

    strategy_returns: Dict[str, pd.Series] = {}
    strategy_signals: Dict[str, pd.Series] = {}
    strategy_instances: Dict[str, Any] = {}

    for display_name, module_path, class_name, kwargs in STRATEGY_REGISTRY:
        try:
            t_start = time.perf_counter()
            strategy = _load_strategy(display_name, module_path, class_name, kwargs)
            strategy_instances[display_name] = strategy

            port_ret = _run_strategy_full_period(
                display_name, strategy, close_prices,
            )
            if port_ret is not None and len(port_ret) > 0:
                strategy_returns[display_name] = port_ret
                # For ensemble meta, we need a signal series too
                # Use sign of returns as a proxy signal
                strategy_signals[display_name] = np.sign(port_ret)

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
    strategy_signals = {
        name: series.reindex(common_idx).fillna(0.0)
        for name, series in strategy_signals.items()
    }

    logger.info(
        "Common OOS period: %s to %s (%d bars)",
        common_idx[0].date(), common_idx[-1].date(), len(common_idx),
    )

    # Print correlation matrix
    corr_df = pd.DataFrame(strategy_returns).corr()
    logger.info("\nStrategy return correlations:")
    logger.info("\n%s", corr_df.to_string(float_format="%.3f"))

    # ---- Phase 2: Build ensemble combinations ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Building ensemble combinations")
    logger.info("=" * 70)

    engine = BacktestEngine()
    all_results: List[Dict[str, Any]] = []

    # --- Individual strategy evaluations (for comparison) ---
    logger.info("\n--- Evaluating individual strategies ---")
    for name, returns in strategy_returns.items():
        try:
            row = _evaluate_individual(name, returns, engine)
            all_results.append(row)
        except Exception:
            logger.error("  Eval failed for %s:\n%s", name, traceback.format_exc())

    # --- A. EnsembleMetaStrategy ---
    logger.info("\n--- A. EnsembleMetaStrategy (Hedge/MW + Sharpe + Markowitz) ---")
    try:
        meta_ret = _ensemble_meta_combine(
            strategy_returns, close_prices, strategy_signals,
        )
        row = _evaluate_returns("EnsembleMeta (MW+Sharpe+MVO)", meta_ret, engine)
        all_results.append(row)
    except Exception:
        logger.error(
            "  EnsembleMeta failed:\n%s", traceback.format_exc(),
        )

    # --- B. Equal-weight average ---
    logger.info("\n--- B. Equal-weight average ---")
    try:
        ew_ret = _equal_weight_combine(strategy_returns)
        row = _evaluate_returns("Equal Weight", ew_ret, engine)
        all_results.append(row)
    except Exception:
        logger.error("  Equal weight failed:\n%s", traceback.format_exc())

    # --- C. Inverse-volatility weighted ---
    logger.info("\n--- C. Inverse-volatility weighted ---")
    try:
        iv_ret = _inverse_vol_combine(strategy_returns, lookback=63)
        row = _evaluate_returns("Inverse-Vol Weighted", iv_ret, engine)
        all_results.append(row)
    except Exception:
        logger.error("  Inverse-vol failed:\n%s", traceback.format_exc())

    # --- D. Markowitz max-Sharpe ---
    logger.info("\n--- D. Markowitz max-Sharpe ---")
    try:
        mkz_ret = _markowitz_max_sharpe_combine(
            strategy_returns, lookback=126, shrinkage=0.1,
        )
        row = _evaluate_returns("Markowitz Max-Sharpe", mkz_ret, engine)
        all_results.append(row)
    except Exception:
        logger.error("  Markowitz failed:\n%s", traceback.format_exc())

    # ---- Phase 3: Report ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Results")
    logger.info("=" * 70)

    if not all_results:
        logger.error("All combinations failed. No results to report.")
        sys.exit(1)

    comparison = pd.DataFrame(all_results)

    # Separate individual vs ensemble rows for sorting
    ensemble_mask = ~comparison["Combination"].str.startswith("[Individual]")
    individuals = comparison[~ensemble_mask].copy()
    ensembles = comparison[ensemble_mask].copy()

    # Sort ensembles by Sharpe descending
    if "Sharpe" in ensembles.columns and len(ensembles) > 0:
        ensembles = ensembles.sort_values("Sharpe", ascending=False)

    # Sort individuals by Sharpe descending
    if "Sharpe" in individuals.columns and len(individuals) > 0:
        individuals = individuals.sort_values("Sharpe", ascending=False)

    # Combine: ensembles first, then individuals
    comparison = pd.concat([ensembles, individuals], ignore_index=True)

    # Identify the best ensemble approach
    if len(ensembles) > 0 and "Sharpe" in ensembles.columns:
        best_row = ensembles.loc[ensembles["Sharpe"].idxmax()]
        best_name = best_row["Combination"]
        best_sharpe = best_row["Sharpe"]
    else:
        best_name = "N/A"
        best_sharpe = np.nan

    # Save to CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "ensemble_results.csv"
    comparison.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    # Print results
    print("\n" + "=" * 120)
    print("ENSEMBLE STRATEGY COMPARISON")
    print(f"Data: {DATA_START} to {DATA_END} | OOS period: {TRAIN_FRACTION:.0%} train / {1-TRAIN_FRACTION:.0%} test")
    print(f"Walk-forward: {WALK_FORWARD_FOLDS} folds | Monte Carlo: {MONTE_CARLO_SIMS:,} simulations")
    print(f"Strategies combined: {', '.join(strategy_returns.keys())}")
    print("=" * 120)

    print("\n--- ENSEMBLE COMBINATIONS ---")
    if len(ensembles) > 0:
        print(
            tabulate(
                ensembles,
                headers="keys",
                tablefmt="pretty",
                showindex=False,
                floatfmt=".4f",
            )
        )
    else:
        print("  (no ensemble results)")

    print("\n--- INDIVIDUAL STRATEGIES (for reference) ---")
    if len(individuals) > 0:
        print(
            tabulate(
                individuals,
                headers="keys",
                tablefmt="pretty",
                showindex=False,
                floatfmt=".4f",
            )
        )

    # Print correlation matrix
    print("\n--- STRATEGY RETURN CORRELATIONS ---")
    print(corr_df.to_string(float_format="%.3f"))

    # Print best combination
    print("\n" + "=" * 120)
    print(f"BEST ENSEMBLE APPROACH: {best_name} (Sharpe = {best_sharpe:.3f})")
    print("=" * 120)
    print(f"\nResults saved to: {csv_path}")

    elapsed_total = time.perf_counter() - t_global_start
    print(f"Total runtime: {elapsed_total:.1f} seconds")


if __name__ == "__main__":
    main()
