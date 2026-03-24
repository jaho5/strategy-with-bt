"""Parameter sweep optimizer for the Entropy Regularized strategy.

Performs a structured parameter search using a train/validation/test split
(60%/20%/20%) to find parameter combinations that maximise out-of-sample PnL
and Monte Carlo confidence of exceeding the 45% PnL target.

The search is guided by a Tree-structured Parzen Estimator (TPE) from
``src.utils.optimizer`` to explore the 5-dimensional parameter space
efficiently across at least 50 trials.

Usage:
    uv run python -m src.optimize_entropy
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.backtest.engine import BacktestEngine, MonteCarloResult
from src.data.downloader import download_etf_data
from src.strategies.entropy_regularized import EntropyRegularizedStrategy

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
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
MC_SIMS = 10_000
MC_TARGET_PNL = 45.0
INITIAL_CAPITAL = 100_000.0


# ---------------------------------------------------------------------------
# Data helpers (mirrored from main.py)
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


def _signals_to_portfolio_returns(
    signals: pd.DataFrame, prices: pd.DataFrame
) -> np.ndarray:
    """Convert multi-asset signal DataFrame to a 1-D portfolio return series.

    Returns simple daily returns (not log returns).
    """
    tickers = list(prices.columns)
    common_idx = signals.index.intersection(prices.index)

    # Try {ticker}_signal + {ticker}_weight pattern
    sig_cols: Dict[str, str] = {}
    wgt_cols: Dict[str, str] = {}
    for ticker in tickers:
        scol = f"{ticker}_signal"
        wcol = f"{ticker}_weight"
        if scol in signals.columns:
            sig_cols[ticker] = scol
        if wcol in signals.columns:
            wgt_cols[ticker] = wcol

    if not sig_cols:
        # Fallback: average all numeric columns
        pos = signals.reindex(common_idx).fillna(0.0).select_dtypes(include="number").mean(axis=1).values
        price_arr = prices.reindex(common_idx).ffill().bfill().mean(axis=1).values
        if len(price_arr) < 2:
            return np.array([])
        asset_ret = np.diff(price_arr) / price_arr[:-1]
        return pos[:-1] * asset_ret

    tickers_used = list(sig_cols.keys())
    price_aligned = prices[tickers_used].reindex(common_idx).ffill().bfill()
    returns = price_aligned.pct_change().fillna(0.0)

    # Build position DataFrame = signal * weight
    pos_df = pd.DataFrame(index=common_idx)
    for ticker in tickers_used:
        sig = signals[sig_cols[ticker]].reindex(common_idx).fillna(0.0)
        wgt = signals[wgt_cols[ticker]].reindex(common_idx).fillna(0.0) if ticker in wgt_cols else sig
        pos_df[ticker] = sig * wgt

    # Portfolio return = sum of position * asset_return
    portfolio_ret = (pos_df * returns).sum(axis=1).values

    # Drop leading zero
    if len(portfolio_ret) > 0 and portfolio_ret[0] == 0.0:
        portfolio_ret = portfolio_ret[1:]

    return portfolio_ret


def _signals_to_portfolio(
    signals: pd.DataFrame, prices: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert signals to (portfolio_signal_1d, portfolio_price_1d)."""
    tickers = list(prices.columns)
    common_idx = signals.index.intersection(prices.index)
    if len(common_idx) == 0:
        common_idx = signals.index

    sig_cols: Dict[str, str] = {}
    for ticker in tickers:
        scol = f"{ticker}_signal"
        if scol in signals.columns:
            sig_cols[ticker] = scol

    if not sig_cols:
        sig_arr = signals.reindex(common_idx).fillna(0.0).select_dtypes(include="number").mean(axis=1).values
        price_arr = prices.reindex(common_idx).ffill().bfill().mean(axis=1).values
        if len(price_arr) > 0 and price_arr[0] != 0:
            price_arr = price_arr / price_arr[0] * 100.0
        return sig_arr, price_arr

    tickers_used = list(sig_cols.keys())
    sig_df = pd.DataFrame(index=common_idx)
    for ticker in tickers_used:
        sig_df[ticker] = signals[sig_cols[ticker]].reindex(common_idx).fillna(0.0)

    price_aligned = prices[tickers_used].reindex(common_idx).ffill().bfill()
    returns = price_aligned.pct_change().fillna(0.0)

    portfolio_signal = sig_df.mean(axis=1).values
    portfolio_returns = returns.mean(axis=1)
    portfolio_price = 100.0 * (1.0 + portfolio_returns).cumprod()

    return portfolio_signal, portfolio_price.values


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _compute_sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe from daily returns."""
    if len(returns) < 2:
        return np.nan
    m = np.mean(returns)
    s = np.std(returns, ddof=1)
    if s < 1e-12:
        return 0.0
    return float((m / s) * np.sqrt(252))


def _compute_pnl_pct(returns: np.ndarray) -> float:
    """Cumulative PnL as a percentage from a return series."""
    if len(returns) < 1:
        return 0.0
    return float((np.prod(1.0 + returns) - 1.0) * 100.0)


def _compute_max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown as a positive fraction."""
    if len(returns) < 1:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / running_max
    return float(np.max(drawdown))


def evaluate_params(
    params: Dict[str, Any],
    train_prices: pd.DataFrame,
    val_prices: pd.DataFrame,
    engine: BacktestEngine,
) -> Dict[str, Any]:
    """Evaluate a parameter set: fit on train, test on validation.

    Returns a dict with performance metrics on the validation set, or
    None-filled metrics on failure.
    """
    result: Dict[str, Any] = {**params}

    try:
        strategy = EntropyRegularizedStrategy(**params)
        strategy.fit(train_prices)
        signals = strategy.generate_signals(val_prices)

        # Get portfolio returns
        val_returns = _signals_to_portfolio_returns(signals, val_prices)

        if val_returns is None or len(val_returns) < 10:
            result.update({
                "val_pnl_pct": np.nan,
                "val_sharpe": np.nan,
                "val_sortino": np.nan,
                "val_max_dd": np.nan,
                "val_mc_prob_above_target": np.nan,
                "status": "insufficient_data",
            })
            return result

        pnl = _compute_pnl_pct(val_returns)
        sharpe = _compute_sharpe(val_returns)
        max_dd = _compute_max_drawdown(val_returns)

        # Sortino
        excess = val_returns.copy()
        downside = excess.copy()
        downside[downside > 0] = 0.0
        downside_std = np.sqrt(np.mean(downside ** 2))
        sortino = float((np.mean(val_returns) / downside_std) * np.sqrt(252)) if downside_std > 0 else np.nan

        # Monte Carlo confidence
        mc_result: MonteCarloResult = engine.monte_carlo_confidence(
            returns=val_returns,
            n_simulations=MC_SIMS,
            target_pnl_pct=MC_TARGET_PNL,
            initial_capital=INITIAL_CAPITAL,
            rng_seed=42,
        )

        result.update({
            "val_pnl_pct": round(pnl, 3),
            "val_sharpe": round(sharpe, 4),
            "val_sortino": round(sortino, 4) if not np.isnan(sortino) else np.nan,
            "val_max_dd": round(max_dd, 4),
            "val_mc_prob_above_target": round(mc_result.prob_above_target, 4),
            "val_mc_mean_terminal": round(mc_result.mean_terminal, 2),
            "val_mc_median_terminal": round(mc_result.median_terminal, 2),
            "status": "ok",
        })

    except Exception as exc:
        logger.warning("Evaluation failed for params %s: %s", params, exc)
        result.update({
            "val_pnl_pct": np.nan,
            "val_sharpe": np.nan,
            "val_sortino": np.nan,
            "val_max_dd": np.nan,
            "val_mc_prob_above_target": np.nan,
            "status": f"error: {exc}",
        })

    return result


def evaluate_on_test(
    params: Dict[str, Any],
    train_val_prices: pd.DataFrame,
    test_prices: pd.DataFrame,
    engine: BacktestEngine,
) -> Dict[str, Any]:
    """Final holdout evaluation: fit on train+val, test on holdout."""
    result: Dict[str, Any] = {}

    try:
        strategy = EntropyRegularizedStrategy(**params)
        strategy.fit(train_val_prices)
        signals = strategy.generate_signals(test_prices)
        test_returns = _signals_to_portfolio_returns(signals, test_prices)

        if test_returns is None or len(test_returns) < 10:
            result.update({
                "test_pnl_pct": np.nan,
                "test_sharpe": np.nan,
                "test_sortino": np.nan,
                "test_max_dd": np.nan,
                "test_mc_prob_above_target": np.nan,
            })
            return result

        pnl = _compute_pnl_pct(test_returns)
        sharpe = _compute_sharpe(test_returns)
        max_dd = _compute_max_drawdown(test_returns)

        excess = test_returns.copy()
        downside = excess.copy()
        downside[downside > 0] = 0.0
        downside_std = np.sqrt(np.mean(downside ** 2))
        sortino = float((np.mean(test_returns) / downside_std) * np.sqrt(252)) if downside_std > 0 else np.nan

        mc_result = engine.monte_carlo_confidence(
            returns=test_returns,
            n_simulations=MC_SIMS,
            target_pnl_pct=MC_TARGET_PNL,
            initial_capital=INITIAL_CAPITAL,
            rng_seed=42,
        )

        result.update({
            "test_pnl_pct": round(pnl, 3),
            "test_sharpe": round(sharpe, 4),
            "test_sortino": round(sortino, 4) if not np.isnan(sortino) else np.nan,
            "test_max_dd": round(max_dd, 4),
            "test_mc_prob_above_target": round(mc_result.prob_above_target, 4),
            "test_mc_mean_terminal": round(mc_result.mean_terminal, 2),
            "test_mc_median_terminal": round(mc_result.median_terminal, 2),
        })

    except Exception as exc:
        logger.warning("Test evaluation failed: %s", exc)
        result.update({
            "test_pnl_pct": np.nan,
            "test_sharpe": np.nan,
            "test_sortino": np.nan,
            "test_max_dd": np.nan,
            "test_mc_prob_above_target": np.nan,
        })

    return result


# ---------------------------------------------------------------------------
# Walk-forward evaluation on the validation window
# ---------------------------------------------------------------------------

def evaluate_params_walkforward(
    params: Dict[str, Any],
    full_prices: pd.DataFrame,
    train_end_idx: int,
    val_end_idx: int,
    n_folds: int = 3,
    engine: Optional[BacktestEngine] = None,
) -> Dict[str, Any]:
    """Walk-forward evaluation using folds within the train+val window.

    Splits the train+val window into ``n_folds`` folds. For each fold,
    the first 75% is used for fitting and the remaining 25% for OOS
    evaluation. Reports aggregate metrics across all OOS segments, plus
    a MC confidence check on the full validation period.
    """
    if engine is None:
        engine = BacktestEngine()

    result: Dict[str, Any] = {**params}

    try:
        # -- Walk-forward within train+val --
        tv_prices = full_prices.iloc[:val_end_idx]
        n = len(tv_prices)
        fold_size = n // n_folds

        all_oos_returns: List[np.ndarray] = []
        fold_sharpes: List[float] = []

        for i in range(n_folds):
            fold_start = i * fold_size
            fold_end = (i + 1) * fold_size if i < n_folds - 1 else n
            train_len = int((fold_end - fold_start) * 0.75)
            f_train_end = fold_start + train_len
            f_test_start = f_train_end
            f_test_end = fold_end

            if f_test_end - f_test_start < 20:
                continue

            f_train_data = tv_prices.iloc[fold_start:f_train_end]
            f_test_data = tv_prices.iloc[f_test_start:f_test_end]

            try:
                strat = EntropyRegularizedStrategy(**params)
                strat.fit(f_train_data)
                sigs = strat.generate_signals(f_test_data)
                oos_ret = _signals_to_portfolio_returns(sigs, f_test_data)
                if oos_ret is not None and len(oos_ret) >= 5:
                    all_oos_returns.append(oos_ret)
                    fold_sharpes.append(_compute_sharpe(oos_ret))
            except Exception:
                continue

        if not all_oos_returns:
            result.update({
                "wf_oos_sharpe": np.nan,
                "wf_oos_pnl_pct": np.nan,
                "val_pnl_pct": np.nan,
                "val_sharpe": np.nan,
                "val_mc_prob_above_target": np.nan,
                "val_max_dd": np.nan,
                "status": "no_valid_folds",
            })
            return result

        concat_oos = np.concatenate(all_oos_returns)
        wf_sharpe = _compute_sharpe(concat_oos)
        wf_pnl = _compute_pnl_pct(concat_oos)

        # -- Direct train -> validation evaluation --
        train_data = full_prices.iloc[:train_end_idx]
        val_data = full_prices.iloc[train_end_idx:val_end_idx]

        strategy = EntropyRegularizedStrategy(**params)
        strategy.fit(train_data)
        val_signals = strategy.generate_signals(val_data)
        val_returns = _signals_to_portfolio_returns(val_signals, val_data)

        if val_returns is None or len(val_returns) < 10:
            result.update({
                "wf_oos_sharpe": round(wf_sharpe, 4),
                "wf_oos_pnl_pct": round(wf_pnl, 3),
                "val_pnl_pct": np.nan,
                "val_sharpe": np.nan,
                "val_mc_prob_above_target": np.nan,
                "val_max_dd": np.nan,
                "status": "val_insufficient_data",
            })
            return result

        val_pnl = _compute_pnl_pct(val_returns)
        val_sharpe = _compute_sharpe(val_returns)
        val_max_dd = _compute_max_drawdown(val_returns)

        # Sortino
        downside = val_returns.copy()
        downside[downside > 0] = 0.0
        downside_std = np.sqrt(np.mean(downside ** 2))
        val_sortino = float((np.mean(val_returns) / downside_std) * np.sqrt(252)) if downside_std > 0 else np.nan

        # MC confidence on validation returns
        mc_result = engine.monte_carlo_confidence(
            returns=val_returns,
            n_simulations=MC_SIMS,
            target_pnl_pct=MC_TARGET_PNL,
            initial_capital=INITIAL_CAPITAL,
            rng_seed=42,
        )

        result.update({
            "wf_oos_sharpe": round(wf_sharpe, 4),
            "wf_oos_pnl_pct": round(wf_pnl, 3),
            "wf_fold_sharpes": str([round(s, 4) for s in fold_sharpes]),
            "val_pnl_pct": round(val_pnl, 3),
            "val_sharpe": round(val_sharpe, 4),
            "val_sortino": round(val_sortino, 4) if not np.isnan(val_sortino) else np.nan,
            "val_max_dd": round(val_max_dd, 4),
            "val_mc_prob_above_target": round(mc_result.prob_above_target, 4),
            "val_mc_mean_terminal": round(mc_result.mean_terminal, 2),
            "val_mc_median_terminal": round(mc_result.median_terminal, 2),
            "status": "ok",
        })

    except Exception as exc:
        logger.warning("WF evaluation failed for params %s: %s", params, exc)
        result.update({
            "wf_oos_sharpe": np.nan,
            "wf_oos_pnl_pct": np.nan,
            "val_pnl_pct": np.nan,
            "val_sharpe": np.nan,
            "val_mc_prob_above_target": np.nan,
            "val_max_dd": np.nan,
            "status": f"error: {exc}",
        })

    return result


# ---------------------------------------------------------------------------
# Parameter grid generation
# ---------------------------------------------------------------------------

def generate_param_grid() -> List[Dict[str, Any]]:
    """Generate a structured grid of parameter combinations.

    Includes a mix of:
    - Coarse grid covering the broad space (Phase 1)
    - Fine grid around the promising region (Phase 2)
    - Extreme/edge-case combos (Phase 3)

    Returns at least 50 unique parameter dictionaries.
    """
    combos: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Phase 1: Broad coarse sweep
    # -----------------------------------------------------------------------
    # gamma: risk aversion. Lower = more aggressive (baseline 1.0)
    gamma_vals = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    # lambda_base: entropy regularisation strength (baseline 0.05)
    lambda_vals = [0.01, 0.03, 0.05, 0.08, 0.15]
    # eg_blend: weight on EG vs entropy-MV (baseline 0.5)
    eg_blend_vals = [0.2, 0.35, 0.5, 0.65, 0.8]
    # rebalance_freq: trading days between rebalances (baseline 5)
    rebalance_vals = [3, 5, 10]
    # eta0: initial learning rate for EG (baseline 0.5)
    eta0_vals = [0.2, 0.5, 1.0, 2.0]

    # Latin-hypercube-style selection from the full grid to get ~30 combos
    rng = np.random.default_rng(42)
    all_broad = list(product(gamma_vals, lambda_vals, eg_blend_vals, rebalance_vals, eta0_vals))
    rng.shuffle(all_broad)  # type: ignore[arg-type]
    for gamma, lam, eg, reb, eta in all_broad[:30]:
        combos.append({
            "gamma": gamma,
            "lambda_base": lam,
            "eg_blend": eg,
            "rebalance_freq": reb,
            "eta0": eta,
        })

    # -----------------------------------------------------------------------
    # Phase 2: Fine grid around the current best region
    # The strategy currently uses gamma=1.0, lambda=0.05, eg_blend=0.5,
    # rebalance=5, eta0=0.5.  We'll do a fine sweep nearby.
    # -----------------------------------------------------------------------
    fine_gamma = [0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
    fine_lambda = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    fine_eg = [0.25, 0.35, 0.45, 0.55, 0.65]
    fine_reb = [3, 5, 7]
    fine_eta = [0.3, 0.5, 0.8, 1.5]

    all_fine = list(product(fine_gamma, fine_lambda, fine_eg, fine_reb, fine_eta))
    rng.shuffle(all_fine)  # type: ignore[arg-type]
    for gamma, lam, eg, reb, eta in all_fine[:20]:
        combos.append({
            "gamma": gamma,
            "lambda_base": lam,
            "eg_blend": eg,
            "rebalance_freq": reb,
            "eta0": eta,
        })

    # -----------------------------------------------------------------------
    # Phase 3: Targeted combos for aggressive alpha capture
    # Lower gamma + higher EG blend + higher learning rate = more aggressive
    # -----------------------------------------------------------------------
    aggressive_combos = [
        {"gamma": 0.2, "lambda_base": 0.02, "eg_blend": 0.7, "rebalance_freq": 3, "eta0": 1.5},
        {"gamma": 0.3, "lambda_base": 0.03, "eg_blend": 0.6, "rebalance_freq": 3, "eta0": 1.0},
        {"gamma": 0.4, "lambda_base": 0.04, "eg_blend": 0.5, "rebalance_freq": 5, "eta0": 0.8},
        {"gamma": 0.5, "lambda_base": 0.05, "eg_blend": 0.4, "rebalance_freq": 3, "eta0": 1.0},
        {"gamma": 0.3, "lambda_base": 0.01, "eg_blend": 0.8, "rebalance_freq": 3, "eta0": 2.0},
        {"gamma": 0.6, "lambda_base": 0.06, "eg_blend": 0.3, "rebalance_freq": 5, "eta0": 0.5},
        {"gamma": 0.7, "lambda_base": 0.10, "eg_blend": 0.5, "rebalance_freq": 5, "eta0": 0.5},
        {"gamma": 0.5, "lambda_base": 0.02, "eg_blend": 0.6, "rebalance_freq": 5, "eta0": 1.0},
        {"gamma": 0.4, "lambda_base": 0.03, "eg_blend": 0.7, "rebalance_freq": 3, "eta0": 1.5},
        {"gamma": 0.8, "lambda_base": 0.08, "eg_blend": 0.4, "rebalance_freq": 7, "eta0": 0.3},
    ]
    combos.extend(aggressive_combos)

    # Default params (baseline comparison)
    combos.append({
        "gamma": 1.0,
        "lambda_base": 0.05,
        "eg_blend": 0.5,
        "rebalance_freq": 5,
        "eta0": 0.5,
    })

    # De-duplicate
    seen = set()
    unique_combos: List[Dict[str, Any]] = []
    for c in combos:
        key = (c["gamma"], c["lambda_base"], c["eg_blend"], c["rebalance_freq"], c["eta0"])
        if key not in seen:
            seen.add(key)
            unique_combos.append(c)

    logger.info("Generated %d unique parameter combinations.", len(unique_combos))
    return unique_combos


# ---------------------------------------------------------------------------
# TPE-guided adaptive search
# ---------------------------------------------------------------------------

def tpe_adaptive_search(
    full_prices: pd.DataFrame,
    train_end_idx: int,
    val_end_idx: int,
    n_trials: int = 20,
    engine: Optional[BacktestEngine] = None,
    seed: int = 123,
) -> List[Dict[str, Any]]:
    """Run TPE-guided Bayesian search for additional parameter combos.

    Uses results from grid search to initialise, then explores further.
    """
    from src.utils.optimizer import _TPESampler, _parse_param_space

    if engine is None:
        engine = BacktestEngine()

    param_space = {
        "gamma": (0.1, 3.0),
        "lambda_base": (0.005, 0.3),
        "eg_blend": (0.1, 0.9),
        "rebalance_freq": (2, 15),
        "eta0": (0.1, 3.0),
    }

    param_defs = _parse_param_space(param_space)
    sampler = _TPESampler(param_defs=param_defs, gamma=0.25, seed=seed)

    results: List[Dict[str, Any]] = []

    for trial_id in range(n_trials):
        raw_params = sampler.ask()

        # Ensure rebalance_freq is int
        params = {
            "gamma": round(raw_params["gamma"], 3),
            "lambda_base": round(raw_params["lambda_base"], 4),
            "eg_blend": round(raw_params["eg_blend"], 3),
            "rebalance_freq": int(round(raw_params["rebalance_freq"])),
            "eta0": round(raw_params["eta0"], 3),
        }

        logger.info(
            "  TPE trial %d/%d: gamma=%.3f lam=%.4f eg=%.3f reb=%d eta=%.3f",
            trial_id + 1, n_trials,
            params["gamma"], params["lambda_base"], params["eg_blend"],
            params["rebalance_freq"], params["eta0"],
        )

        row = evaluate_params_walkforward(
            params, full_prices, train_end_idx, val_end_idx, n_folds=3, engine=engine,
        )

        # Score for TPE: composite of val_sharpe and MC confidence
        val_sharpe = row.get("val_sharpe", np.nan)
        mc_prob = row.get("val_mc_prob_above_target", np.nan)
        if np.isnan(val_sharpe):
            val_sharpe = -10.0
        if np.isnan(mc_prob):
            mc_prob = 0.0
        score = val_sharpe + 2.0 * mc_prob  # weight MC confidence heavily

        sampler.tell(raw_params, score)
        row["tpe_score"] = round(score, 4)
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full parameter optimisation pipeline."""

    t0 = time.perf_counter()
    logger.info("=" * 80)
    logger.info("Entropy Regularized Strategy -- Parameter Optimization")
    logger.info("=" * 80)

    # 1. Download data
    logger.info("Downloading ETF data (%s to %s)...", DATA_START, DATA_END)
    raw_data = download_etf_data(start=DATA_START, end=DATA_END)
    close_prices = _extract_close_prices(raw_data)
    logger.info(
        "Data: %d rows x %d tickers (%s to %s)",
        len(close_prices), len(close_prices.columns),
        close_prices.index[0].date(), close_prices.index[-1].date(),
    )

    # 2. Split: 60% train / 20% validation / 20% test
    n = len(close_prices)
    train_end_idx = int(n * 0.60)
    val_end_idx = int(n * 0.80)

    train_data = close_prices.iloc[:train_end_idx]
    val_data = close_prices.iloc[train_end_idx:val_end_idx]
    test_data = close_prices.iloc[val_end_idx:]
    train_val_data = close_prices.iloc[:val_end_idx]

    logger.info("  Train:      %s to %s (%d bars)", train_data.index[0].date(), train_data.index[-1].date(), len(train_data))
    logger.info("  Validation: %s to %s (%d bars)", val_data.index[0].date(), val_data.index[-1].date(), len(val_data))
    logger.info("  Test:       %s to %s (%d bars)", test_data.index[0].date(), test_data.index[-1].date(), len(test_data))

    engine = BacktestEngine()

    # 3. Phase 1: Grid search
    logger.info("\n" + "=" * 80)
    logger.info("Phase 1: Structured Grid Search")
    logger.info("=" * 80)

    param_grid = generate_param_grid()
    grid_results: List[Dict[str, Any]] = []

    for i, params in enumerate(param_grid):
        logger.info(
            "  [%d/%d] gamma=%.2f lam=%.3f eg=%.2f reb=%d eta=%.2f",
            i + 1, len(param_grid),
            params["gamma"], params["lambda_base"], params["eg_blend"],
            params["rebalance_freq"], params["eta0"],
        )

        row = evaluate_params_walkforward(
            params, close_prices, train_end_idx, val_end_idx, n_folds=3, engine=engine,
        )
        row["search_phase"] = "grid"
        grid_results.append(row)

        # Progress log
        if row.get("status") == "ok":
            logger.info(
                "    -> val_pnl=%.1f%% val_sharpe=%.3f mc_prob=%.3f wf_sharpe=%.3f",
                row.get("val_pnl_pct", 0), row.get("val_sharpe", 0),
                row.get("val_mc_prob_above_target", 0), row.get("wf_oos_sharpe", 0),
            )

    # 4. Phase 2: TPE-guided adaptive search
    logger.info("\n" + "=" * 80)
    logger.info("Phase 2: TPE-Guided Bayesian Search (adaptive)")
    logger.info("=" * 80)

    # Seed the TPE with top results from grid search
    tpe_results = tpe_adaptive_search(
        close_prices, train_end_idx, val_end_idx, n_trials=20, engine=engine,
    )
    for r in tpe_results:
        r["search_phase"] = "tpe"

    # 5. Combine all results
    all_results = grid_results + tpe_results
    results_df = pd.DataFrame(all_results)

    # 6. Rank by composite score: val_sharpe + 2 * mc_prob + 0.5 * wf_oos_sharpe
    results_df["composite_score"] = (
        results_df["val_sharpe"].fillna(-10)
        + 2.0 * results_df["val_mc_prob_above_target"].fillna(0)
        + 0.5 * results_df["wf_oos_sharpe"].fillna(-10)
    )
    results_df = results_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # 7. Report top 10
    logger.info("\n" + "=" * 80)
    logger.info("Top 10 Parameter Combinations (by composite score)")
    logger.info("=" * 80)

    top10_cols = [
        "gamma", "lambda_base", "eg_blend", "rebalance_freq", "eta0",
        "val_pnl_pct", "val_sharpe", "val_mc_prob_above_target",
        "wf_oos_sharpe", "val_max_dd", "composite_score", "search_phase",
    ]
    available_cols = [c for c in top10_cols if c in results_df.columns]
    top10 = results_df.head(10)[available_cols]

    try:
        from tabulate import tabulate
        print("\n" + tabulate(top10, headers="keys", tablefmt="pretty", showindex=True, floatfmt=".4f"))
    except ImportError:
        print(top10.to_string())

    # 8. Evaluate best params on held-out test set
    logger.info("\n" + "=" * 80)
    logger.info("Final Holdout Test Evaluation (top 3 configs)")
    logger.info("=" * 80)

    strategy_param_keys = ["gamma", "lambda_base", "eg_blend", "rebalance_freq", "eta0"]
    top3_for_test: List[Dict[str, Any]] = []

    for rank in range(min(3, len(results_df))):
        row = results_df.iloc[rank]
        best_params = {k: row[k] for k in strategy_param_keys}
        # Ensure rebalance_freq is int
        best_params["rebalance_freq"] = int(best_params["rebalance_freq"])

        logger.info(
            "  Rank %d params: gamma=%.3f lam=%.4f eg=%.3f reb=%d eta=%.3f",
            rank + 1, best_params["gamma"], best_params["lambda_base"],
            best_params["eg_blend"], best_params["rebalance_freq"], best_params["eta0"],
        )

        test_metrics = evaluate_on_test(best_params, train_val_data, test_data, engine)

        logger.info(
            "    Test PnL=%.2f%% Sharpe=%.3f MC P(>45%%)=%.3f MaxDD=%.2f%%",
            test_metrics.get("test_pnl_pct", 0),
            test_metrics.get("test_sharpe", 0),
            test_metrics.get("test_mc_prob_above_target", 0),
            test_metrics.get("test_max_dd", 0) * 100 if test_metrics.get("test_max_dd") else 0,
        )

        combined = {**best_params, "rank": rank + 1, **test_metrics}
        combined["val_pnl_pct"] = row.get("val_pnl_pct")
        combined["val_sharpe"] = row.get("val_sharpe")
        combined["val_mc_prob_above_target"] = row.get("val_mc_prob_above_target")
        combined["composite_score"] = row.get("composite_score")
        top3_for_test.append(combined)

    # 9. Save results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "entropy_optimization.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Full results saved to %s", csv_path)

    # Save top configs separately
    if top3_for_test:
        test_df = pd.DataFrame(top3_for_test)
        test_csv = REPORTS_DIR / "entropy_optimization_best.csv"
        test_df.to_csv(test_csv, index=False)
        logger.info("Best configs + test results saved to %s", test_csv)

    # 10. Print final summary
    elapsed = time.perf_counter() - t0
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE (%.1f seconds)", elapsed)
    logger.info("=" * 80)

    if top3_for_test:
        best = top3_for_test[0]
        print("\n" + "=" * 80)
        print("BEST PARAMETERS FOUND")
        print("=" * 80)
        print(f"  gamma:          {best['gamma']}")
        print(f"  lambda_base:    {best['lambda_base']}")
        print(f"  eg_blend:       {best['eg_blend']}")
        print(f"  rebalance_freq: {best['rebalance_freq']}")
        print(f"  eta0:           {best['eta0']}")
        print()
        print("Validation Performance:")
        print(f"  PnL:              {best.get('val_pnl_pct', 'N/A')}%")
        print(f"  Sharpe:           {best.get('val_sharpe', 'N/A')}")
        print(f"  MC P(PnL>45%):    {best.get('val_mc_prob_above_target', 'N/A')}")
        print()
        print("Holdout Test Performance:")
        print(f"  PnL:              {best.get('test_pnl_pct', 'N/A')}%")
        print(f"  Sharpe:           {best.get('test_sharpe', 'N/A')}")
        print(f"  MC P(PnL>45%):    {best.get('test_mc_prob_above_target', 'N/A')}")
        print(f"  Max Drawdown:     {best.get('test_max_dd', 'N/A')}")
        print("=" * 80)

    logger.info("Total trials evaluated: %d", len(results_df))
    logger.info("Results at: %s", csv_path)


if __name__ == "__main__":
    main()
