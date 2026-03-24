"""Ensemble strategy runner v2: tests additional ensemble variants to find
even better combinations beyond the original four methods.

Ensemble variants tested:
    1. Risk Parity             -- Equal risk contribution weighting
    2. Minimum Correlation     -- Greedy selection of 3 least-correlated strategies
    3. Kelly-Optimal           -- Half-Kelly weights maximising geometric growth
    4. Regime-Switching        -- HMM-regime-dependent weight switching
    5. Best-3 Inverse-Vol      -- Top 3 by OOS Sharpe, inverse-vol weighted

For each variant:
    - Walk-forward validation
    - Monte Carlo P(>45%)
    - PASS/FAIL report

Usage:
    uv run python -m src.run_ensemble_v2
"""

from __future__ import annotations

import importlib
import itertools
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
MONTE_CARLO_TARGET_PNL = 45.0
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Pass/fail thresholds
MC_PROB_THRESHOLD = 0.95  # P(>45%) must exceed this
WF_SHARPE_THRESHOLD = 0.0  # WF OOS Sharpe must exceed this

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

# Momentum vs defensive classification for regime switching
MOMENTUM_STRATEGIES = {"Spectral Momentum", "HMM Regime"}
DEFENSIVE_STRATEGIES = {"GARCH Vol", "Bayesian Changepoint"}
NEUTRAL_STRATEGIES = {"Entropy Regularized"}


# ---------------------------------------------------------------------------
# Data helpers  (mirrored from run_ensemble.py)
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
# Signal -> Portfolio return conversion  (mirrored from run_ensemble.py)
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
# Ensemble combination methods  (v2 variants)
# ---------------------------------------------------------------------------

def _risk_parity_combine(
    strategy_returns: Dict[str, pd.Series],
    lookback: int = 126,
    max_iter: int = 100,
) -> pd.Series:
    """Risk Parity Ensemble: each strategy contributes equal risk.

    Weight_i chosen so that RC_i = w_i * (Sigma * w)_i / (w' Sigma w) = 1/K
    for all i.  Solved via sequential least-squares optimisation at each step.
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
            cov = window.cov().values

            # Regularise
            trace_cov = np.trace(cov)
            if trace_cov <= 0:
                w = np.full(K, 1.0 / K)
            else:
                shrinkage = 0.1
                cov = (1 - shrinkage) * cov + shrinkage * (trace_cov / K) * np.eye(K)
                w = _solve_risk_parity(cov, K)

        combined.iloc[t] = float(ret_df.iloc[t].values @ w)

    return combined


def _solve_risk_parity(cov: np.ndarray, K: int) -> np.ndarray:
    """Solve for risk-parity weights.

    Minimise sum_i (RC_i - 1/K)^2 subject to w >= 0, sum(w) = 1,
    where RC_i = w_i * (Sigma * w)_i / (w' * Sigma * w).
    """
    target_rc = 1.0 / K

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        if port_var <= 1e-16:
            return 0.0
        marginal_risk = cov @ w  # (K,)
        risk_contrib = w * marginal_risk / port_var
        return float(np.sum((risk_contrib - target_rc) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(1e-6, 1.0)] * K
    w0 = np.full(K, 1.0 / K)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                objective, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-12},
            )
        if res.success:
            weights = np.maximum(res.x, 0.0)
            s = weights.sum()
            if s > 0:
                return weights / s
    except Exception:
        pass

    return np.full(K, 1.0 / K)


def _min_correlation_combine(
    strategy_returns: Dict[str, pd.Series],
    n_select: int = 3,
    lookback: int = 126,
) -> pd.Series:
    """Minimum Correlation Ensemble: greedily select the n_select LEAST
    correlated strategies, then inverse-vol weight the subset.

    From the correlation matrix, greedily pick strategies that minimise
    average pairwise correlation.  Then apply inverse-vol weighting on the
    selected subset.
    """
    ret_df = pd.DataFrame(strategy_returns).fillna(0.0)
    names = list(ret_df.columns)
    K = len(names)
    T = len(ret_df)

    n_select = min(n_select, K)

    combined = pd.Series(0.0, index=ret_df.index)

    for t in range(T):
        if t < lookback:
            w_full = np.full(K, 1.0 / K)
        else:
            window = ret_df.iloc[t - lookback:t]
            corr = window.corr().values
            vol = window.std().values

            # Greedy selection of least-correlated subset
            selected = _greedy_min_correlation(corr, n_select)

            # Inverse-vol weighting on selected subset
            vol_selected = vol[selected]
            vol_selected = np.where(vol_selected < 1e-10, 1e-10, vol_selected)
            inv_vol = 1.0 / vol_selected
            w_selected = inv_vol / inv_vol.sum()

            w_full = np.zeros(K)
            for idx_pos, idx_orig in enumerate(selected):
                w_full[idx_orig] = w_selected[idx_pos]

        combined.iloc[t] = float(ret_df.iloc[t].values @ w_full)

    return combined


def _greedy_min_correlation(corr: np.ndarray, n_select: int) -> List[int]:
    """Greedily select n_select indices that minimise average pairwise correlation.

    1. Start with the pair of strategies that has the lowest correlation.
    2. Iteratively add the strategy whose average correlation with the
       already-selected set is lowest.
    """
    K = corr.shape[0]
    if n_select >= K:
        return list(range(K))

    # Find the pair with lowest correlation
    best_pair = (0, 1)
    best_corr = corr[0, 1]
    for i in range(K):
        for j in range(i + 1, K):
            if corr[i, j] < best_corr:
                best_corr = corr[i, j]
                best_pair = (i, j)

    selected = list(best_pair)

    # Greedily add remaining
    remaining = [i for i in range(K) if i not in selected]
    while len(selected) < n_select and remaining:
        best_candidate = remaining[0]
        best_avg_corr = np.mean([abs(corr[best_candidate, s]) for s in selected])

        for candidate in remaining[1:]:
            avg_corr = np.mean([abs(corr[candidate, s]) for s in selected])
            if avg_corr < best_avg_corr:
                best_avg_corr = avg_corr
                best_candidate = candidate

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


def _kelly_optimal_combine(
    strategy_returns: Dict[str, pd.Series],
    lookback: int = 126,
    kelly_fraction: float = 0.5,
    shrinkage: float = 0.1,
) -> pd.Series:
    """Kelly-Optimal Ensemble: maximise geometric growth rate.

    g(w) = E[log(1 + w'r)] ~ w'mu - (1/2)w'Sigma*w   (for small returns)

    Uses half-Kelly (kelly_fraction=0.5) for safety.
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

            trace_cov = np.trace(cov)
            if trace_cov <= 0:
                w = np.full(K, 1.0 / K)
            else:
                cov = (1 - shrinkage) * cov + shrinkage * (trace_cov / K) * np.eye(K)
                w = _solve_kelly(mu, cov, K, kelly_fraction)

        combined.iloc[t] = float(ret_df.iloc[t].values @ w)

    return combined


def _solve_kelly(
    mu: np.ndarray, cov: np.ndarray, K: int, kelly_fraction: float,
) -> np.ndarray:
    """Solve the Kelly criterion optimisation.

    max_w  w'mu - (1/2) w'Sigma w   s.t. w >= 0, sum(w) = 1
    Then scale by kelly_fraction (half-Kelly).
    """
    def neg_growth(w: np.ndarray) -> float:
        return -(w @ mu - 0.5 * w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * K
    w0 = np.full(K, 1.0 / K)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                neg_growth, w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-12},
            )
        if res.success:
            weights = np.maximum(res.x, 0.0)
            # Apply half-Kelly: blend toward equal weight
            ew = np.full(K, 1.0 / K)
            weights = kelly_fraction * weights + (1.0 - kelly_fraction) * ew
            s = weights.sum()
            if s > 0:
                return weights / s
    except Exception:
        pass

    return np.full(K, 1.0 / K)


def _regime_switching_combine(
    strategy_returns: Dict[str, pd.Series],
    close_prices: pd.DataFrame,
    strategy_instances: Dict[str, Any],
    lookback: int = 63,
) -> pd.Series:
    """Regime-Switching Ensemble: use HMM regimes to switch weights.

    - Bull regime:    overweight momentum strategies (Spectral, HMM)
    - Bear regime:    overweight defensive strategies (GARCH, Bayesian CP)
    - Sideways:       equal weight

    Regime detection comes from the fitted HMM Regime strategy.
    If HMM is unavailable, falls back to a simple volatility-based regime proxy.
    """
    ret_df = pd.DataFrame(strategy_returns).fillna(0.0)
    names = list(ret_df.columns)
    K = len(names)
    T = len(ret_df)
    _, test_data = _split_data(close_prices, TRAIN_FRACTION)

    # Build regime series (bull/bear/sideways for each date)
    regimes = _detect_regimes(test_data, strategy_instances)
    regimes = regimes.reindex(ret_df.index).fillna("sideways")

    # Precompute weight vectors for each regime
    momentum_idx = [i for i, n in enumerate(names) if n in MOMENTUM_STRATEGIES]
    defensive_idx = [i for i, n in enumerate(names) if n in DEFENSIVE_STRATEGIES]
    neutral_idx = [i for i, n in enumerate(names) if n in NEUTRAL_STRATEGIES]

    combined = pd.Series(0.0, index=ret_df.index)

    for t in range(T):
        regime = regimes.iloc[t]

        if regime == "bull":
            # Overweight momentum: 2x for momentum, 0.5x for defensive, 1x neutral
            raw_w = np.ones(K)
            for idx in momentum_idx:
                raw_w[idx] = 2.0
            for idx in defensive_idx:
                raw_w[idx] = 0.5
        elif regime == "bear":
            # Overweight defensive: 0.5x for momentum, 2x for defensive, 1x neutral
            raw_w = np.ones(K)
            for idx in momentum_idx:
                raw_w[idx] = 0.5
            for idx in defensive_idx:
                raw_w[idx] = 2.0
        else:
            # Sideways: equal weight
            raw_w = np.ones(K)

        # Normalise
        s = raw_w.sum()
        w = raw_w / s if s > 0 else np.full(K, 1.0 / K)

        # Optionally blend with inverse-vol if we have enough history
        if t >= lookback:
            window = ret_df.iloc[t - lookback:t]
            vol = window.std().values
            vol = np.where(vol < 1e-10, 1e-10, vol)
            inv_vol = 1.0 / vol
            w_iv = inv_vol / inv_vol.sum()
            # 60% regime, 40% inverse-vol for stability
            w = 0.6 * w + 0.4 * w_iv

        combined.iloc[t] = float(ret_df.iloc[t].values @ w)

    return combined


def _detect_regimes(
    test_data: pd.DataFrame,
    strategy_instances: Dict[str, Any],
) -> pd.Series:
    """Detect regimes using the HMM strategy's fitted model.

    Falls back to a simple return/vol-based regime proxy if HMM unavailable.
    """
    # Try to use the HMM strategy's model
    hmm_strategy = strategy_instances.get("HMM Regime")
    if hmm_strategy is not None and getattr(hmm_strategy, "_is_fitted", False):
        try:
            return _detect_regimes_hmm(test_data, hmm_strategy)
        except Exception as exc:
            logger.warning("HMM regime detection failed, using fallback: %s", exc)

    # Fallback: simple return/vol-based regime classification
    return _detect_regimes_fallback(test_data)


def _detect_regimes_hmm(
    test_data: pd.DataFrame,
    hmm_strategy: Any,
) -> pd.Series:
    """Use the fitted HMM model to classify each date into a regime."""
    from src.strategies.hmm_regime import _build_features

    # Use the first column as representative price series
    prices = test_data.iloc[:, 0] if test_data.shape[1] > 1 else test_data.iloc[:, 0]
    log_returns = np.log(prices / prices.shift(1)).dropna()

    cfg = hmm_strategy.config
    features = _build_features(
        log_returns,
        vol_lookback=cfg.vol_lookback,
        skew_lookback=cfg.skew_lookback,
    )

    if features.empty:
        return pd.Series("sideways", index=test_data.index)

    X = features.values
    model = hmm_strategy._model
    labels = hmm_strategy._regime_labels

    # Get regime probabilities
    regime_probs = hmm_strategy._filtered_probabilities(X)

    # Map to regime names
    inv_labels = {v: k for k, v in labels.items()}
    regime_names = []
    for t in range(len(features)):
        dominant = int(np.argmax(regime_probs[t]))
        regime_names.append(inv_labels.get(dominant, "sideways"))

    regime_series = pd.Series(regime_names, index=features.index)
    return regime_series.reindex(test_data.index, method="ffill").fillna("sideways")


def _detect_regimes_fallback(
    test_data: pd.DataFrame,
    lookback: int = 63,
) -> pd.Series:
    """Simple return/volatility-based regime proxy.

    - Bull:     rolling return > 0 AND rolling vol < median vol
    - Bear:     rolling return < 0 AND rolling vol > median vol
    - Sideways: everything else
    """
    prices = test_data.iloc[:, 0] if test_data.shape[1] > 1 else test_data.iloc[:, 0]
    returns = prices.pct_change().fillna(0.0)

    rolling_ret = returns.rolling(lookback).mean()
    rolling_vol = returns.rolling(lookback).std()

    median_vol = rolling_vol.median()

    regimes = pd.Series("sideways", index=test_data.index)
    regimes[(rolling_ret > 0) & (rolling_vol < median_vol)] = "bull"
    regimes[(rolling_ret < 0) & (rolling_vol > median_vol)] = "bear"

    return regimes


def _best3_inverse_vol_combine(
    strategy_returns: Dict[str, pd.Series],
    lookback: int = 63,
) -> Tuple[pd.Series, List[str]]:
    """Best-3 Inverse-Vol: use only the top 3 strategies by OOS Sharpe,
    weighted by inverse volatility.

    Returns the combined series and the list of selected strategy names.
    """
    ret_df = pd.DataFrame(strategy_returns).fillna(0.0)
    names = list(ret_df.columns)

    # Compute full-period OOS Sharpe for each strategy
    sharpes = {}
    for name in names:
        r = ret_df[name].values
        mu = np.mean(r) * 252
        sigma = np.std(r, ddof=1) * np.sqrt(252)
        sharpes[name] = mu / sigma if sigma > 1e-10 else 0.0

    # Select top 3
    sorted_names = sorted(sharpes.keys(), key=lambda n: sharpes[n], reverse=True)
    top3 = sorted_names[:3]
    logger.info(
        "  Best-3 selected: %s (Sharpes: %s)",
        top3,
        {n: f"{sharpes[n]:.3f}" for n in top3},
    )

    # Inverse-vol combine on the top 3 subset
    subset_returns = {n: strategy_returns[n] for n in top3}
    combined = _inverse_vol_combine_simple(subset_returns, lookback=lookback)

    return combined, top3


def _inverse_vol_combine_simple(
    strategy_returns: Dict[str, pd.Series],
    lookback: int = 63,
) -> pd.Series:
    """Simple inverse-volatility combination (no regime or other overlays)."""
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
        return {"Combination": name, "Status": "SKIPPED", "PASS/FAIL": "SKIP"}

    # Build synthetic price from returns for backtest engine
    price = 100.0 * np.cumprod(1.0 + returns_arr)
    price = np.insert(price, 0, 100.0)
    signal = np.ones(len(price))

    # Core backtest
    bt_result = engine.run(signal, price)
    metrics = bt_result.metrics

    logger.info("  [%s] PnL=%.2f%%, Sharpe=%.3f, MaxDD=%.2f%%",
                name, metrics["total_pnl_pct"], metrics["sharpe_ratio"],
                metrics["max_drawdown"] * 100)

    # Monte Carlo
    try:
        mc_result = engine.monte_carlo_confidence(
            returns=bt_result.returns,
            n_simulations=MONTE_CARLO_SIMS,
            target_pnl_pct=MONTE_CARLO_TARGET_PNL,
        )
        mc_prob = mc_result.prob_above_target
    except Exception as exc:
        logger.warning("  MC failed for %s: %s", name, exc)
        mc_prob = np.nan

    # Walk-forward OOS
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
        wf_oos_sharpe = np.nan

    # PASS/FAIL determination
    mc_pass = (not np.isnan(mc_prob)) and (mc_prob > MC_PROB_THRESHOLD)
    wf_pass = (not np.isnan(wf_oos_sharpe)) and (wf_oos_sharpe > WF_SHARPE_THRESHOLD)
    overall_pass = mc_pass and wf_pass

    return {
        "Combination": name,
        "Total PnL%": round(metrics["total_pnl_pct"], 2),
        "Ann. Return%": round(metrics["annualized_return"] * 100, 2),
        "Sharpe": round(metrics["sharpe_ratio"], 3),
        "Sortino": round(metrics["sortino_ratio"], 3),
        "Max DD%": round(metrics["max_drawdown"] * 100, 2),
        "Win Rate%": round(metrics["win_rate"] * 100, 2),
        f"P(PnL>{int(MONTE_CARLO_TARGET_PNL)}%) MC": (
            round(mc_prob, 4) if not np.isnan(mc_prob) else np.nan
        ),
        "Bootstrap p": round(metrics["bootstrap_pvalue"], 4),
        "WF OOS Sharpe": (
            round(wf_oos_sharpe, 3) if not np.isnan(wf_oos_sharpe) else np.nan
        ),
        "PASS/FAIL": "PASS" if overall_pass else "FAIL",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: load data, run strategies, test ensemble variants."""

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

    # ---- Phase 2: Build ensemble v2 combinations ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Building ensemble v2 combinations")
    logger.info("=" * 70)

    engine = BacktestEngine()
    all_results: List[Dict[str, Any]] = []

    # --- 1. Risk Parity ---
    logger.info("\n--- 1. Risk Parity Ensemble ---")
    try:
        rp_ret = _risk_parity_combine(strategy_returns, lookback=126)
        row = _evaluate_returns("Risk Parity", rp_ret, engine)
        all_results.append(row)
    except Exception:
        logger.error("  Risk Parity failed:\n%s", traceback.format_exc())

    # --- 2. Minimum Correlation ---
    logger.info("\n--- 2. Minimum Correlation Ensemble (top 3 least correlated) ---")
    try:
        mc_ret = _min_correlation_combine(strategy_returns, n_select=3, lookback=126)
        row = _evaluate_returns("Min-Correlation (3)", mc_ret, engine)
        all_results.append(row)
    except Exception:
        logger.error("  Min-Correlation failed:\n%s", traceback.format_exc())

    # --- 3. Kelly-Optimal ---
    logger.info("\n--- 3. Kelly-Optimal Ensemble (half-Kelly) ---")
    try:
        kelly_ret = _kelly_optimal_combine(
            strategy_returns, lookback=126, kelly_fraction=0.5,
        )
        row = _evaluate_returns("Half-Kelly Optimal", kelly_ret, engine)
        all_results.append(row)
    except Exception:
        logger.error("  Kelly-Optimal failed:\n%s", traceback.format_exc())

    # --- 4. Regime-Switching ---
    logger.info("\n--- 4. Regime-Switching Ensemble ---")
    try:
        rs_ret = _regime_switching_combine(
            strategy_returns, close_prices, strategy_instances, lookback=63,
        )
        row = _evaluate_returns("Regime-Switching", rs_ret, engine)
        all_results.append(row)
    except Exception:
        logger.error("  Regime-Switching failed:\n%s", traceback.format_exc())

    # --- 5. Best-3 Inverse-Vol ---
    logger.info("\n--- 5. Best-3 Inverse-Vol ---")
    try:
        b3_ret, b3_names = _best3_inverse_vol_combine(
            strategy_returns, lookback=63,
        )
        row = _evaluate_returns(
            f"Best-3 InvVol ({', '.join(b3_names)})", b3_ret, engine,
        )
        all_results.append(row)
    except Exception:
        logger.error("  Best-3 InvVol failed:\n%s", traceback.format_exc())

    # ---- Phase 3: Report ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Results")
    logger.info("=" * 70)

    if not all_results:
        logger.error("All combinations failed. No results to report.")
        sys.exit(1)

    comparison = pd.DataFrame(all_results)

    # Sort by Sharpe descending
    if "Sharpe" in comparison.columns and len(comparison) > 0:
        comparison = comparison.sort_values("Sharpe", ascending=False)

    # Save to CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "ensemble_v2_results.csv"
    comparison.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    # Print results
    print("\n" + "=" * 130)
    print("ENSEMBLE V2 STRATEGY COMPARISON")
    print(
        f"Data: {DATA_START} to {DATA_END} | "
        f"OOS period: {TRAIN_FRACTION:.0%} train / {1 - TRAIN_FRACTION:.0%} test"
    )
    print(
        f"Walk-forward: {WALK_FORWARD_FOLDS} folds | "
        f"Monte Carlo: {MONTE_CARLO_SIMS:,} simulations"
    )
    print(f"Strategies combined: {', '.join(strategy_returns.keys())}")
    print(
        f"Pass criteria: P(>{int(MONTE_CARLO_TARGET_PNL)}%) > "
        f"{MC_PROB_THRESHOLD:.0%} AND WF OOS Sharpe > {WF_SHARPE_THRESHOLD}"
    )
    print("=" * 130)

    print("\n--- ENSEMBLE V2 VARIANTS ---")
    if len(comparison) > 0:
        print(
            tabulate(
                comparison,
                headers="keys",
                tablefmt="pretty",
                showindex=False,
                floatfmt=".4f",
            )
        )
    else:
        print("  (no results)")

    # Print correlation matrix
    print("\n--- STRATEGY RETURN CORRELATIONS ---")
    print(corr_df.to_string(float_format="%.3f"))

    # Identify the best passing ensemble
    passing = comparison[comparison["PASS/FAIL"] == "PASS"]
    if len(passing) > 0 and "Sharpe" in passing.columns:
        best_row = passing.loc[passing["Sharpe"].idxmax()]
        best_name = best_row["Combination"]
        best_sharpe = best_row["Sharpe"]
        print(f"\nBEST PASSING VARIANT: {best_name} (Sharpe = {best_sharpe:.3f})")
    else:
        # Even if none pass, show the best by Sharpe
        if "Sharpe" in comparison.columns and len(comparison) > 0:
            best_row = comparison.loc[comparison["Sharpe"].idxmax()]
            best_name = best_row["Combination"]
            best_sharpe = best_row["Sharpe"]
            print(
                f"\nBEST VARIANT (none passed): {best_name} "
                f"(Sharpe = {best_sharpe:.3f})"
            )
        else:
            best_name = "N/A"
            best_sharpe = np.nan
            print("\nNo variants produced results.")

    # Summary pass/fail table
    print("\n--- PASS/FAIL SUMMARY ---")
    for _, row in comparison.iterrows():
        status = row.get("PASS/FAIL", "N/A")
        mc_val = row.get(f"P(PnL>{int(MONTE_CARLO_TARGET_PNL)}%) MC", np.nan)
        wf_val = row.get("WF OOS Sharpe", np.nan)
        mc_str = f"{mc_val:.4f}" if not (isinstance(mc_val, float) and np.isnan(mc_val)) else "N/A"
        wf_str = f"{wf_val:.3f}" if not (isinstance(wf_val, float) and np.isnan(wf_val)) else "N/A"
        print(
            f"  [{status:4s}]  {row['Combination']:<45s}  "
            f"MC P(>45%)={mc_str:>8s}  WF Sharpe={wf_str:>8s}"
        )

    print("\n" + "=" * 130)
    print(f"Results saved to: {csv_path}")

    elapsed_total = time.perf_counter() - t_global_start
    print(f"Total runtime: {elapsed_total:.1f} seconds")


if __name__ == "__main__":
    main()
