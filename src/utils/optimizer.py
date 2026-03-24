"""Bayesian parameter optimizer for trading strategies.

Uses a from-scratch Tree-structured Parzen Estimator (TPE) with walk-forward
validation as the objective function.  Includes Deflated Sharpe Ratio
(Harvey & Liu 2015) to guard against overfitting from multiple testing.

Dependencies: numpy, scipy only (no optuna or other optimization libraries).

Usage
-----
>>> from src.utils.optimizer import StrategyOptimizer
>>> optimizer = StrategyOptimizer()
>>> result = optimizer.optimize(
...     strategy_class=MyStrategy,
...     param_space={
...         'threshold': (0.5, 3.0),         # continuous
...         'window': (20, 252),              # discrete (int bounds)
...         'method': ['momentum', 'mean_reversion'],  # categorical
...     },
...     prices=close_prices_df,
...     n_trials=100,
... )
>>> print(result.best_params)
>>> print(result.deflated_sharpe)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParamDef:
    """Internal representation of a single parameter dimension."""

    name: str
    kind: str  # "continuous", "discrete", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None


@dataclass
class TrialResult:
    """Stores the outcome of a single optimization trial."""

    trial_id: int
    params: Dict[str, Any]
    oos_sharpe: float  # raw out-of-sample Sharpe from walk-forward
    fold_sharpes: List[float] = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Final output of the optimization run."""

    best_params: Dict[str, Any]
    best_oos_sharpe: float
    deflated_sharpe: float
    all_trials: List[TrialResult]
    oos_sharpe_distribution: np.ndarray  # array of all OOS Sharpe values


# ---------------------------------------------------------------------------
# Parameter space parsing
# ---------------------------------------------------------------------------

def _parse_param_space(param_space: Dict[str, Any]) -> List[ParamDef]:
    """Convert a user-friendly param_space dict into internal ParamDef list.

    Rules:
    - ``list`` value  -> categorical
    - ``tuple(int, int)`` where both bounds are int -> discrete
    - ``tuple(float, float)`` -> continuous
    """
    defs: List[ParamDef] = []
    for name, spec in param_space.items():
        if isinstance(spec, list):
            defs.append(ParamDef(name=name, kind="categorical", choices=spec))
        elif isinstance(spec, tuple) and len(spec) == 2:
            lo, hi = spec
            if isinstance(lo, int) and isinstance(hi, int):
                defs.append(ParamDef(name=name, kind="discrete", low=float(lo), high=float(hi)))
            else:
                defs.append(ParamDef(name=name, kind="continuous", low=float(lo), high=float(hi)))
        else:
            raise ValueError(
                f"Parameter '{name}': expected a list (categorical) or a 2-tuple "
                f"(continuous/discrete range), got {type(spec).__name__}"
            )
    return defs


# ---------------------------------------------------------------------------
# TPE sampler (from scratch)
# ---------------------------------------------------------------------------

class _TPESampler:
    """Tree-structured Parzen Estimator for Bayesian hyperparameter optimization.

    Splits observed trials into "good" (top quantile) and "bad" (rest),
    fits kernel density estimates to each group, and proposes candidates
    that maximize l(x) / g(x) -- the ratio of the good-model density to
    the bad-model density.

    Reference: Bergstra et al., "Algorithms for Hyper-Parameter Optimization",
    NeurIPS 2011.
    """

    def __init__(
        self,
        param_defs: List[ParamDef],
        gamma: float = 0.25,
        n_candidates: int = 24,
        seed: int = 42,
    ) -> None:
        self.param_defs = param_defs
        self.gamma = gamma  # quantile split for good vs bad
        self.n_candidates = n_candidates
        self.rng = np.random.default_rng(seed)

        # Storage
        self._params_history: List[Dict[str, Any]] = []
        self._scores: List[float] = []

    @property
    def n_observed(self) -> int:
        return len(self._scores)

    def tell(self, params: Dict[str, Any], score: float) -> None:
        """Record an observation (higher score = better)."""
        self._params_history.append(params)
        self._scores.append(score)

    def ask(self) -> Dict[str, Any]:
        """Suggest the next parameter configuration to evaluate.

        For the first ``n_startup`` calls (< 2 / gamma observations),
        returns a uniform random sample.  After that, uses TPE.
        """
        n_startup = max(int(2.0 / self.gamma), 10)
        if self.n_observed < n_startup:
            return self._random_sample()

        # Split into good (top gamma quantile) and bad
        scores = np.array(self._scores)
        n_good = max(1, int(np.ceil(self.gamma * len(scores))))
        sorted_idx = np.argsort(-scores)  # descending
        good_idx = sorted_idx[:n_good]
        bad_idx = sorted_idx[n_good:]

        # Generate candidates and pick the one with highest l(x)/g(x)
        best_candidate = None
        best_ratio = -np.inf

        for _ in range(self.n_candidates):
            candidate = self._sample_from_good(good_idx)
            log_l = self._log_density(candidate, good_idx)
            log_g = self._log_density(candidate, bad_idx)
            ratio = log_l - log_g
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = candidate

        return best_candidate  # type: ignore[return-value]

    def _random_sample(self) -> Dict[str, Any]:
        """Uniform random sample across the parameter space."""
        params: Dict[str, Any] = {}
        for pd_ in self.param_defs:
            if pd_.kind == "continuous":
                params[pd_.name] = float(self.rng.uniform(pd_.low, pd_.high))
            elif pd_.kind == "discrete":
                params[pd_.name] = int(self.rng.integers(int(pd_.low), int(pd_.high) + 1))
            elif pd_.kind == "categorical":
                params[pd_.name] = self.rng.choice(pd_.choices)
            else:
                raise ValueError(f"Unknown param kind: {pd_.kind}")
        return params

    def _sample_from_good(self, good_idx: np.ndarray) -> Dict[str, Any]:
        """Draw a sample from the "good" distribution (l(x)).

        For continuous/discrete params: pick a random good observation, add
        Gaussian noise with bandwidth chosen by Silverman's rule.
        For categorical: sample proportional to category frequency in good set.
        """
        params: Dict[str, Any] = {}
        for pd_ in self.param_defs:
            good_vals = [self._params_history[i][pd_.name] for i in good_idx]

            if pd_.kind == "continuous":
                base = self.rng.choice(good_vals)
                bw = self._silverman_bandwidth(
                    np.array([float(v) for v in good_vals])
                )
                sample = self.rng.normal(base, bw)
                sample = float(np.clip(sample, pd_.low, pd_.high))
                params[pd_.name] = sample

            elif pd_.kind == "discrete":
                base = self.rng.choice(good_vals)
                bw = max(
                    1.0,
                    self._silverman_bandwidth(
                        np.array([float(v) for v in good_vals])
                    ),
                )
                sample = self.rng.normal(float(base), bw)
                sample = int(np.clip(np.round(sample), pd_.low, pd_.high))
                params[pd_.name] = sample

            elif pd_.kind == "categorical":
                # Weighted by frequency in good set, with Laplace smoothing
                counts: Dict[Any, float] = {}
                for c in pd_.choices:
                    counts[c] = 1.0  # Laplace prior
                for v in good_vals:
                    counts[v] = counts.get(v, 1.0) + 1.0
                total = sum(counts.values())
                probs = [counts[c] / total for c in pd_.choices]
                idx = self.rng.choice(len(pd_.choices), p=probs)
                params[pd_.name] = pd_.choices[idx]

        return params

    def _log_density(self, candidate: Dict[str, Any], indices: np.ndarray) -> float:
        """Estimate log-density of candidate under KDE built from indexed observations.

        Uses a Parzen window (Gaussian kernel for continuous/discrete,
        categorical kernel for categorical params).
        """
        if len(indices) == 0:
            return -1e10

        log_dens = 0.0
        for pd_ in self.param_defs:
            vals = [self._params_history[i][pd_.name] for i in indices]
            x = candidate[pd_.name]

            if pd_.kind in ("continuous", "discrete"):
                arr = np.array([float(v) for v in vals])
                bw = self._silverman_bandwidth(arr)
                if bw < 1e-12:
                    bw = 1.0
                # Gaussian KDE: average of individual kernels
                diffs = (float(x) - arr) / bw
                # log of mean of exp(-0.5 * diffs^2)
                log_kernels = -0.5 * diffs ** 2 - np.log(bw) - 0.5 * np.log(2 * np.pi)
                # log-sum-exp trick
                max_lk = np.max(log_kernels)
                log_dens += max_lk + np.log(np.mean(np.exp(log_kernels - max_lk)))

            elif pd_.kind == "categorical":
                # Categorical kernel with Laplace smoothing
                n_cats = len(pd_.choices)
                match_count = sum(1 for v in vals if v == x)
                # Laplace-smoothed probability
                prob = (match_count + 1.0) / (len(vals) + n_cats)
                log_dens += np.log(prob)

        return log_dens

    @staticmethod
    def _silverman_bandwidth(arr: np.ndarray) -> float:
        """Silverman's rule of thumb for KDE bandwidth."""
        n = len(arr)
        if n < 2:
            return 1.0
        std = np.std(arr, ddof=1)
        iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        spread = min(std, iqr / 1.34) if iqr > 0 else std
        if spread < 1e-12:
            spread = 1.0
        return spread * (n ** (-1.0 / 5.0))


# ---------------------------------------------------------------------------
# Walk-forward objective
# ---------------------------------------------------------------------------

def _walk_forward_sharpe(
    strategy_class: Type,
    params: Dict[str, Any],
    prices: "pd.DataFrame",
    n_splits: int = 5,
    train_pct: float = 0.70,
) -> Tuple[float, List[float]]:
    """Evaluate a parameter set using walk-forward validation.

    Returns the aggregate OOS Sharpe ratio and per-fold Sharpe values.
    This is the objective function for the optimizer -- it measures
    performance on truly out-of-sample data via rolling re-fits.

    Parameters
    ----------
    strategy_class : Type
        The strategy class (subclass of Strategy). Will be instantiated
        with ``**params`` for each fold.
    params : dict
        Parameter dictionary to pass to the strategy constructor.
    prices : pd.DataFrame
        Full price DataFrame (datetime index, one column per ticker).
    n_splits : int
        Number of walk-forward folds.
    train_pct : float
        Fraction of each fold used for training.

    Returns
    -------
    (aggregate_sharpe, fold_sharpes)
    """
    import pandas as pd

    n = len(prices)
    fold_size = n // n_splits
    if fold_size < 10:
        return np.nan, []

    all_oos_returns: List[np.ndarray] = []
    fold_sharpes: List[float] = []

    for i in range(n_splits):
        fold_start = i * fold_size
        fold_end = (i + 1) * fold_size if i < n_splits - 1 else n

        train_len = int((fold_end - fold_start) * train_pct)
        train_end = fold_start + train_len
        test_start = train_end
        test_end = fold_end

        if test_end - test_start < 5:
            continue

        train_data = prices.iloc[fold_start:train_end]
        test_data = prices.iloc[test_start:test_end]

        try:
            strategy = strategy_class(**params)
            strategy.fit(train_data)
            signals = strategy.generate_signals(test_data)
        except Exception as exc:
            logger.debug("Fold %d failed for params %s: %s", i, params, exc)
            continue

        # Extract portfolio returns from signals
        oos_returns = _signals_to_returns(signals, test_data)
        if oos_returns is None or len(oos_returns) < 2:
            continue

        fold_sharpe = _compute_sharpe(oos_returns)
        fold_sharpes.append(fold_sharpe)
        all_oos_returns.append(oos_returns)

    if not all_oos_returns:
        return np.nan, []

    concat_returns = np.concatenate(all_oos_returns)
    aggregate_sharpe = _compute_sharpe(concat_returns)

    return aggregate_sharpe, fold_sharpes


def _signals_to_returns(signals: "pd.DataFrame", prices: "pd.DataFrame") -> Optional[np.ndarray]:
    """Convert strategy signals + prices into a 1-D portfolio return series.

    Handles the same signal column conventions as ``main.py``:
    - Direct ticker column matches
    - ``{ticker}_signal`` / ``{ticker}_weight`` pattern
    - Single ``signal`` / ``weight`` pattern
    """
    import pandas as pd

    price_tickers = list(prices.columns)
    common_idx = signals.index.intersection(prices.index)
    if len(common_idx) < 2:
        return None

    # Try to identify signal columns
    sig_cols: Dict[str, str] = {}

    # Pattern 1: columns match price tickers directly
    direct = [c for c in signals.columns if c in price_tickers]
    if direct:
        for c in direct:
            sig_cols[c] = c

    # Pattern 2: '{ticker}_signal' columns
    if not sig_cols:
        for ticker in price_tickers:
            scol = f"{ticker}_signal"
            if scol in signals.columns:
                sig_cols[ticker] = scol

    # Pattern 3: single 'signal' + 'weight' column
    if not sig_cols and "signal" in signals.columns:
        weight = signals.get("weight", pd.Series(1.0, index=signals.index))
        pos = (signals["signal"] * weight).reindex(common_idx).fillna(0.0).values
        price_arr = prices.reindex(common_idx).ffill().bfill().mean(axis=1).values
        if len(price_arr) < 2:
            return None
        asset_ret = np.diff(price_arr) / price_arr[:-1]
        portfolio_ret = pos[:-1] * asset_ret
        return portfolio_ret

    # Pattern 4: average all numeric columns as positions
    if not sig_cols:
        pos = signals.reindex(common_idx).fillna(0.0).select_dtypes(include="number").mean(axis=1).values
        price_arr = prices.reindex(common_idx).ffill().bfill().mean(axis=1).values
        if len(price_arr) < 2:
            return None
        asset_ret = np.diff(price_arr) / price_arr[:-1]
        portfolio_ret = pos[:-1] * asset_ret
        return portfolio_ret

    # Multi-asset: compute weighted portfolio return
    tickers_used = list(sig_cols.keys())
    sig_df = pd.DataFrame(index=common_idx)
    for ticker in tickers_used:
        sig_df[ticker] = signals[sig_cols[ticker]].reindex(common_idx).fillna(0.0)

    price_aligned = prices[tickers_used].reindex(common_idx).ffill().bfill()
    returns = price_aligned.pct_change().fillna(0.0)

    # Portfolio return: sum of signal * asset return
    portfolio_ret = (sig_df * returns).sum(axis=1).values
    # Drop leading zero
    if len(portfolio_ret) > 0 and portfolio_ret[0] == 0.0:
        portfolio_ret = portfolio_ret[1:]

    return portfolio_ret if len(portfolio_ret) >= 2 else None


def _compute_sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe ratio from a daily return series."""
    if len(returns) < 2:
        return np.nan
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r < 1e-12:
        return 0.0
    return float((mean_r / std_r) * np.sqrt(_TRADING_DAYS_PER_YEAR))


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def _deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis_excess: float = 0.0,
) -> float:
    """Compute the Deflated Sharpe Ratio (Harvey & Liu 2015).

    The DSR adjusts for the multiple-testing bias that arises when many
    parameter configurations (trials) are evaluated.  Under the null of
    zero true Sharpe, the expected maximum Sharpe across K independent
    trials is approximately:

        E[max SR] ~ sqrt(2 * log(K)) - (log(pi) + log(log(K))) / (2 * sqrt(2 * log(K)))

    The DSR is the probability that the observed Sharpe exceeds this
    expected maximum, taking into account non-normality of returns
    (skewness and excess kurtosis).

    Parameters
    ----------
    sharpe : float
        Observed (raw) annualized Sharpe ratio.
    n_obs : int
        Number of return observations used to compute the Sharpe.
    n_trials : int
        Total number of parameter configurations tried.
    skewness : float
        Sample skewness of the return series.
    kurtosis_excess : float
        Sample excess kurtosis (kurtosis - 3) of the return series.

    Returns
    -------
    float
        Deflated Sharpe Ratio -- probability in [0, 1] that the observed
        Sharpe is genuine (i.e. not a product of multiple testing).
        Higher is better; values above 0.95 are considered robust.
    """
    if n_trials < 1:
        n_trials = 1
    if n_obs < 3:
        return 0.0

    # Expected maximum Sharpe under the null (Euler-Mascheroni approximation)
    log_k = np.log(max(n_trials, 1))
    if log_k < 1e-10:
        # Only one trial -- no multiple-testing penalty
        sharpe_0 = 0.0
    else:
        sqrt_2logk = np.sqrt(2.0 * log_k)
        sharpe_0 = sqrt_2logk - (np.log(np.pi) + np.log(log_k)) / (2.0 * sqrt_2logk)

    # Variance of the Sharpe ratio estimator (Lo 2002), incorporating
    # higher moments (non-normality adjustment)
    sr = sharpe / np.sqrt(_TRADING_DAYS_PER_YEAR)  # de-annualise
    sr0 = sharpe_0 / np.sqrt(_TRADING_DAYS_PER_YEAR)

    var_sr = (
        (1.0 - skewness * sr + ((kurtosis_excess) / 4.0) * sr ** 2) / n_obs
    )

    if var_sr <= 0:
        var_sr = 1.0 / n_obs

    # Test statistic: how many standard deviations is the observed Sharpe
    # above the expected maximum?
    z = (sr - sr0) / np.sqrt(var_sr)

    # p-value from standard normal CDF
    dsr = float(sp_stats.norm.cdf(z))

    return dsr


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class StrategyOptimizer:
    """Bayesian optimizer for trading strategy parameters.

    Uses a Tree-structured Parzen Estimator (TPE) to explore the parameter
    space and walk-forward validation (out-of-sample Sharpe ratio) as the
    objective function.  Reports both raw and Deflated Sharpe Ratios to
    flag potential overfitting.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward folds for the objective function.
    train_pct : float
        Fraction of each fold used for training.
    gamma : float
        TPE quantile threshold separating "good" from "bad" trials.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_pct: float = 0.70,
        gamma: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.gamma = gamma
        self.seed = seed

    def optimize(
        self,
        strategy_class: Type,
        param_space: Dict[str, Any],
        prices: "pd.DataFrame",
        n_trials: int = 100,
    ) -> OptimizationResult:
        """Find the best strategy parameters via Bayesian optimization.

        Parameters
        ----------
        strategy_class : Type
            A ``Strategy`` subclass.  Must accept the parameter-space keys
            as keyword arguments in its constructor.
        param_space : dict
            Parameter search space.  Supported value formats:

            - ``(lo, hi)`` with both ``float`` -> continuous range
            - ``(lo, hi)`` with both ``int`` -> discrete (integer) range
            - ``[val1, val2, ...]`` -> categorical choices

        prices : pd.DataFrame
            Full historical price data (datetime index, one column per
            ticker).
        n_trials : int
            Number of parameter configurations to evaluate.

        Returns
        -------
        OptimizationResult
            Contains best parameters, raw and deflated Sharpe ratios,
            full trial history, and the OOS Sharpe distribution.
        """
        import pandas as pd

        param_defs = _parse_param_space(param_space)
        sampler = _TPESampler(
            param_defs=param_defs,
            gamma=self.gamma,
            seed=self.seed,
        )

        trials: List[TrialResult] = []
        best_sharpe = -np.inf
        best_params: Dict[str, Any] = {}

        logger.info(
            "Starting optimization: %d trials, %d WF folds, %d params",
            n_trials, self.n_splits, len(param_defs),
        )

        for trial_id in range(n_trials):
            params = sampler.ask()

            # Evaluate via walk-forward validation
            oos_sharpe, fold_sharpes = _walk_forward_sharpe(
                strategy_class=strategy_class,
                params=params,
                prices=prices,
                n_splits=self.n_splits,
                train_pct=self.train_pct,
            )

            # Handle NaN: treat as a very bad result
            if np.isnan(oos_sharpe):
                oos_sharpe = -10.0

            sampler.tell(params, oos_sharpe)

            trial = TrialResult(
                trial_id=trial_id,
                params=params,
                oos_sharpe=oos_sharpe,
                fold_sharpes=fold_sharpes,
            )
            trials.append(trial)

            if oos_sharpe > best_sharpe:
                best_sharpe = oos_sharpe
                best_params = params.copy()

            if (trial_id + 1) % max(1, n_trials // 10) == 0:
                logger.info(
                    "  Trial %d/%d  |  current best OOS Sharpe: %.4f",
                    trial_id + 1, n_trials, best_sharpe,
                )

        # Collect OOS Sharpe distribution
        oos_sharpes = np.array([t.oos_sharpe for t in trials])

        # Compute return statistics for DSR from the best trial's folds
        # Re-run the best params to get the actual return series for moments
        _, best_fold_sharpes = _walk_forward_sharpe(
            strategy_class=strategy_class,
            params=best_params,
            prices=prices,
            n_splits=self.n_splits,
            train_pct=self.train_pct,
        )

        # Estimate moments from the OOS return data of the best config
        # Use a full walk-forward run to get concatenated returns
        n_obs = _estimate_oos_n_obs(prices, self.n_splits, self.train_pct)
        skew, kurt_excess = _estimate_return_moments(
            strategy_class, best_params, prices, self.n_splits, self.train_pct
        )

        deflated_sharpe = _deflated_sharpe_ratio(
            sharpe=best_sharpe,
            n_obs=n_obs,
            n_trials=n_trials,
            skewness=skew,
            kurtosis_excess=kurt_excess,
        )

        logger.info("Optimization complete.")
        logger.info("  Best OOS Sharpe (raw):      %.4f", best_sharpe)
        logger.info("  Deflated Sharpe Ratio:       %.4f", deflated_sharpe)
        logger.info("  Best params: %s", best_params)
        logger.info(
            "  OOS Sharpe distribution: mean=%.4f, std=%.4f, median=%.4f",
            np.nanmean(oos_sharpes),
            np.nanstd(oos_sharpes),
            np.nanmedian(oos_sharpes),
        )

        return OptimizationResult(
            best_params=best_params,
            best_oos_sharpe=best_sharpe,
            deflated_sharpe=deflated_sharpe,
            all_trials=trials,
            oos_sharpe_distribution=oos_sharpes,
        )


# ---------------------------------------------------------------------------
# Helpers for DSR moment estimation
# ---------------------------------------------------------------------------

def _estimate_oos_n_obs(
    prices: "pd.DataFrame",
    n_splits: int,
    train_pct: float,
) -> int:
    """Estimate total number of OOS return observations across all folds."""
    n = len(prices)
    fold_size = n // n_splits
    total_oos = 0
    for i in range(n_splits):
        fold_end = (i + 1) * fold_size if i < n_splits - 1 else n
        fold_start = i * fold_size
        train_len = int((fold_end - fold_start) * train_pct)
        test_len = (fold_end - fold_start) - train_len
        total_oos += max(0, test_len - 1)  # -1 for diff
    return max(total_oos, 1)


def _estimate_return_moments(
    strategy_class: Type,
    params: Dict[str, Any],
    prices: "pd.DataFrame",
    n_splits: int,
    train_pct: float,
) -> Tuple[float, float]:
    """Run walk-forward with the best params and compute skewness and excess kurtosis
    of the concatenated OOS returns."""
    import pandas as pd

    n = len(prices)
    fold_size = n // n_splits
    all_oos: List[np.ndarray] = []

    for i in range(n_splits):
        fold_start = i * fold_size
        fold_end = (i + 1) * fold_size if i < n_splits - 1 else n

        train_len = int((fold_end - fold_start) * train_pct)
        train_end = fold_start + train_len
        test_start = train_end
        test_end = fold_end

        if test_end - test_start < 5:
            continue

        train_data = prices.iloc[fold_start:train_end]
        test_data = prices.iloc[test_start:test_end]

        try:
            strategy = strategy_class(**params)
            strategy.fit(train_data)
            signals = strategy.generate_signals(test_data)
            oos_ret = _signals_to_returns(signals, test_data)
            if oos_ret is not None and len(oos_ret) >= 2:
                all_oos.append(oos_ret)
        except Exception:
            continue

    if not all_oos:
        return 0.0, 0.0

    concat = np.concatenate(all_oos)
    if len(concat) < 4:
        return 0.0, 0.0

    skew = float(sp_stats.skew(concat, bias=False))
    kurt = float(sp_stats.kurtosis(concat, bias=False))  # excess kurtosis

    return skew, kurt
