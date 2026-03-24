"""
Fractional Differentiation Strategy
====================================

Strategy based on fractional differentiation for optimal memory preservation
in financial time series, following Marcos Lopez de Prado's *Advances in
Financial Machine Learning* (Chapter 5).

Mathematical foundation
-----------------------
The fractional differentiation operator is defined as:

    (1 - B)^d = sum_{k=0}^{inf} (-1)^k * C(d, k) * B^k

where B is the backshift operator, 0 < d < 1, and

    C(d, k) = Gamma(d + 1) / (Gamma(k + 1) * Gamma(d - k + 1))

Integer differentiation (d = 1, i.e. standard returns) removes ALL memory
from a price series.  Fractional differentiation with d < 1 achieves
stationarity while preserving long-range dependence -- the minimum amount
of differencing needed to pass the ADF test yields a series that is both
stationary and maximally informative.

Key insight: the optimal d is the *smallest* value such that the
fractionally-differenced series is stationary (ADF p < 0.05).  This d
encodes the market's memory structure and can itself serve as a regime
indicator.

References
----------
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. Chapter 5: Fractionally Differentiated Features.
- Hosking, J. R. M. (1981). "Fractional differencing". Biometrika, 68(1).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import gamma
from statsmodels.tsa.stattools import adfuller

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FracDiffConfig:
    """Hyper-parameters for the fractional differentiation strategy."""

    # -- Optimal d search --
    d_min: float = 0.0
    d_max: float = 1.0
    d_step: float = 0.05
    adf_pvalue_threshold: float = 0.05

    # -- Fixed-width window frac-diff --
    # A threshold of 1e-3 keeps filter widths under ~75 observations,
    # which is practical for typical test windows (500-1000 rows).
    # The original 1e-5 produced filter widths of 3000-4000, causing
    # the frac-diff series to be entirely NaN on short test data.
    weight_threshold: float = 1e-3  # drop weights below this magnitude

    # -- Signal generation --
    z_lookback: int = 63          # rolling window for z-score mean/std
    momentum_lookback: int = 21   # momentum window on frac-diff series
    d_regime_threshold: float = 0.4  # d < threshold => mean-reversion, else momentum
    entry_z: float = 1.0         # |z| must exceed this for a signal
    exit_z: float = 0.3          # |z| below this => flatten

    # -- Adaptive re-estimation --
    refit_period: int = 63       # re-estimate d every N trading days (quarterly)

    # -- Position sizing --
    max_weight: float = 1.0      # maximum position weight


# ---------------------------------------------------------------------------
# Core fractional differentiation routines
# ---------------------------------------------------------------------------

def _frac_diff_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Compute the fractional differentiation filter weights.

    The weights are the coefficients in the binomial expansion of (1 - B)^d:

        w_k = (-1)^k * C(d, k) = prod_{i=1}^{k} (d - i + 1) / i  ,  w_0 = 1

    Weights are accumulated until |w_k| < threshold, which determines the
    effective window width.

    Parameters
    ----------
    d : float
        Fractional differencing order, typically in (0, 1).
    threshold : float
        Minimum absolute weight to retain.

    Returns
    -------
    np.ndarray
        Weight vector w[0], w[1], ..., w[K] where |w[K]| >= threshold.
    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
        # Safety: cap at 10000 to prevent runaway loops
        if k > 10000:
            break
    return np.array(weights, dtype=np.float64)


def frac_diff_fixed_width(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """Apply fixed-width fractional differentiation to a series.

    Uses a finite window determined by the weight threshold.  This is the
    practical implementation from Lopez de Prado that avoids the infinite
    lookback of the exact fractional difference.

    The fractionally-differenced value at time t is:

        x^(d)_t = sum_{k=0}^{K} w_k * x_{t-k}

    where K is the window length determined by the weight threshold.

    Parameters
    ----------
    series : pd.Series
        Raw price series (log-prices are recommended but not required).
    d : float
        Fractional differencing order.
    threshold : float
        Minimum absolute weight; determines window width.

    Returns
    -------
    pd.Series
        Fractionally-differenced series.  The first (K-1) values are NaN
        because the filter needs K observations.
    """
    weights = _frac_diff_weights(d, threshold)
    width = len(weights)

    values = series.values.astype(np.float64)
    n = len(values)
    result = np.full(n, np.nan)

    # Apply the convolution filter
    for t in range(width - 1, n):
        result[t] = np.dot(weights, values[t - width + 1 : t + 1][::-1])

    return pd.Series(result, index=series.index, name=series.name)


def find_optimal_d(
    series: pd.Series,
    d_min: float = 0.0,
    d_max: float = 1.0,
    d_step: float = 0.05,
    adf_pvalue_threshold: float = 0.05,
    weight_threshold: float = 1e-5,
) -> Tuple[float, float]:
    """Find the minimum fractional differencing order d that makes the
    series stationary (ADF test p-value < threshold).

    Searches d in [d_min, d_max] with the given step size, returning the
    smallest d for which the ADF null hypothesis of a unit root is rejected.

    Parameters
    ----------
    series : pd.Series
        Raw price (or log-price) series.
    d_min, d_max, d_step : float
        Grid-search bounds and resolution.
    adf_pvalue_threshold : float
        Significance level for the ADF test.
    weight_threshold : float
        Threshold for frac-diff weight truncation.

    Returns
    -------
    tuple[float, float]
        (optimal_d, adf_pvalue) -- the minimum d achieving stationarity
        and the corresponding ADF p-value.  If no d in the range yields
        stationarity, returns (d_max, last_pvalue).
    """
    d_values = np.arange(d_min, d_max + d_step / 2, d_step)

    best_d = d_max
    best_pvalue = 1.0

    for d in d_values:
        if d == 0.0:
            # d=0 is no differentiation; skip since raw prices are
            # almost certainly non-stationary
            continue

        frac_series = frac_diff_fixed_width(series, d, threshold=weight_threshold)
        frac_clean = frac_series.dropna()

        if len(frac_clean) < 20:
            # Not enough data points for a meaningful ADF test
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_stat, adf_pvalue, *_ = adfuller(frac_clean, maxlag=1)
        except Exception:
            continue

        if adf_pvalue < adf_pvalue_threshold:
            best_d = d
            best_pvalue = adf_pvalue
            break  # We want the minimum d, so stop at first success

        best_pvalue = adf_pvalue

    return float(best_d), float(best_pvalue)


def _frac_diff_weights_gamma(d: float, k: int) -> float:
    """Compute the k-th binomial coefficient C(d, k) using the Gamma function.

    C(d, k) = Gamma(d + 1) / (Gamma(k + 1) * Gamma(d - k + 1))

    This is used for analytical/verification purposes.  The recursive
    formulation in ``_frac_diff_weights`` is numerically more stable and
    faster for generating the full weight vector.

    Parameters
    ----------
    d : float
        Fractional differencing order.
    k : int
        Weight index (k = 0, 1, 2, ...).

    Returns
    -------
    float
        The weight w_k = (-1)^k * C(d, k).
    """
    coeff = gamma(d + 1) / (gamma(k + 1) * gamma(d - k + 1))
    return ((-1) ** k) * coeff


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

class FractionalDifferentiationStrategy(Strategy):
    """Trading strategy based on fractional differentiation for optimal
    memory preservation in financial time series.

    The strategy exploits the insight that integer differentiation (d=1,
    standard log-returns) discards all long-range memory, while the raw
    price series (d=0) is non-stationary.  By finding the minimum d in
    (0, 1) that achieves stationarity, we retain the maximum amount of
    predictive memory.

    Workflow
    --------
    1. **fit(prices)** -- For each asset, find the optimal fractional
       differencing order d* via grid search + ADF testing.  Store d*
       and pre-compute the frac-diff series for signal generation.

    2. **generate_signals(prices)** -- Compute the frac-diff series at d*,
       then apply a regime-dependent signal:
       - If d* < 0.4 (strong memory): mean-reversion on frac-diff z-score
       - If d* >= 0.4 (weak memory):  momentum on frac-diff series
       Signal strength is scaled by |z_t| and clipped to [-1, 1].

    3. **Adaptive d** -- Re-estimates d* every 63 trading days.  Changes
       in d* over time indicate evolving market memory structure.
    """

    def __init__(self, config: Optional[FracDiffConfig] = None) -> None:
        super().__init__(
            name="FractionalDifferentiation",
            description=(
                "Fractional differentiation strategy for optimal memory "
                "preservation (Lopez de Prado, AFML Ch.5)"
            ),
        )
        self.config = config or FracDiffConfig()

        # Per-asset fitted state
        self._optimal_d: Dict[str, float] = {}
        self._adf_pvalues: Dict[str, float] = {}
        self._d_history: Dict[str, List[Tuple[Any, float]]] = {}
        self._last_fit_idx: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    #  Fit (abstract method implementation)                                #
    # ------------------------------------------------------------------ #

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "FractionalDifferentiationStrategy":
        """Calibrate the strategy by finding optimal d for each asset.

        For each column in ``prices``, searches for the minimum fractional
        differencing order d such that the frac-diff series is stationary
        (ADF p < 0.05).

        Parameters
        ----------
        prices : pd.DataFrame
            Price data indexed by datetime, one column per asset.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        cfg = self.config

        for ticker in prices.columns:
            series = prices[ticker].dropna()
            if len(series) < 50:
                logger.warning(
                    "Ticker %s has only %d observations; skipping d estimation.",
                    ticker,
                    len(series),
                )
                continue

            # Use log-prices for frac-diff (more numerically stable)
            log_series = np.log(series)

            d_star, adf_p = find_optimal_d(
                log_series,
                d_min=cfg.d_min,
                d_max=cfg.d_max,
                d_step=cfg.d_step,
                adf_pvalue_threshold=cfg.adf_pvalue_threshold,
                weight_threshold=cfg.weight_threshold,
            )

            self._optimal_d[ticker] = d_star
            self._adf_pvalues[ticker] = adf_p

            # Track d history for regime analysis
            timestamp = series.index[-1] if hasattr(series.index, '__len__') else None
            if ticker not in self._d_history:
                self._d_history[ticker] = []
            self._d_history[ticker].append((timestamp, d_star))

            logger.info(
                "Ticker %s: optimal d=%.3f (ADF p=%.4f, window=%d weights)",
                ticker,
                d_star,
                adf_p,
                len(_frac_diff_weights(d_star, cfg.weight_threshold)),
            )

        self.parameters = {
            "optimal_d": dict(self._optimal_d),
            "adf_pvalues": dict(self._adf_pvalues),
        }
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  Signal generation (abstract method implementation)                  #
    # ------------------------------------------------------------------ #

    def generate_signals(self, prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate trading signals from fractionally-differenced features.

        For each asset:

        1. Compute the frac-diff series x^(d*) at the fitted optimal d*.
        2. Compute rolling z-score: z_t = (x^(d*)_t - mean) / std.
        3. Apply regime-dependent signal logic:
           - d* < 0.4 (mean-reversion): buy when z < -entry_z, sell when z > entry_z
           - d* >= 0.4 (momentum): buy when frac-diff momentum > 0 and z > entry_z,
             sell when frac-diff momentum < 0 and z < -entry_z
        4. Scale weight by min(|z_t| / entry_z, 1.0).

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV or adjusted-close prices indexed by datetime.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``{ticker}_signal`` and ``{ticker}_weight``
            columns for each asset.
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        cfg = self.config
        n = len(prices)
        result_frames: List[pd.DataFrame] = []

        for ticker in prices.columns:
            if ticker not in self._optimal_d:
                logger.warning(
                    "No fitted d for ticker %s; emitting flat signals.", ticker
                )
                sig = pd.DataFrame(
                    {f"{ticker}_signal": 0, f"{ticker}_weight": 0.0},
                    index=prices.index,
                )
                result_frames.append(sig)
                continue

            d_star = self._optimal_d[ticker]
            series = prices[ticker].copy()

            # Handle NaN forward fill for continuity, then backfill leading NaN
            series = series.ffill().bfill()

            log_series = np.log(series)

            # -- Compute frac-diff series --
            frac_series = frac_diff_fixed_width(
                log_series, d_star, threshold=cfg.weight_threshold
            )

            # -- Z-score of frac-diff level --
            frac_mean = frac_series.rolling(
                window=cfg.z_lookback, min_periods=max(cfg.z_lookback // 2, 1)
            ).mean()
            frac_std = frac_series.rolling(
                window=cfg.z_lookback, min_periods=max(cfg.z_lookback // 2, 1)
            ).std()

            # Prevent division by zero
            frac_std = frac_std.replace(0, np.nan)
            z_score = (frac_series - frac_mean) / frac_std

            # -- Momentum on frac-diff series --
            frac_momentum = frac_series.diff(cfg.momentum_lookback)

            # -- Signal generation with hysteresis --
            signals = np.zeros(n, dtype=np.float64)
            weights = np.zeros(n, dtype=np.float64)
            current_pos = 0.0

            is_mean_reversion = d_star < cfg.d_regime_threshold

            for t in range(n):
                z_t = z_score.iloc[t]
                mom_t = frac_momentum.iloc[t]

                if np.isnan(z_t):
                    signals[t] = current_pos
                    weights[t] = 0.0
                    continue

                abs_z = abs(z_t)

                if is_mean_reversion:
                    # -- Mean-reversion regime (strong memory, low d) --
                    # Buy when z is very negative (price below frac-diff mean)
                    # Sell when z is very positive (price above frac-diff mean)
                    if current_pos == 0.0:
                        if z_t < -cfg.entry_z:
                            current_pos = 1.0
                        elif z_t > cfg.entry_z:
                            current_pos = -1.0
                    else:
                        if abs_z < cfg.exit_z:
                            current_pos = 0.0
                        elif current_pos > 0 and z_t > cfg.entry_z:
                            # Z flipped -- reverse position
                            current_pos = -1.0
                        elif current_pos < 0 and z_t < -cfg.entry_z:
                            current_pos = 1.0
                else:
                    # -- Momentum regime (weak memory, higher d) --
                    # Follow the frac-diff momentum direction when z confirms
                    if np.isnan(mom_t):
                        signals[t] = current_pos
                        weights[t] = 0.0
                        continue

                    if current_pos == 0.0:
                        if mom_t > 0 and z_t > cfg.entry_z:
                            current_pos = 1.0
                        elif mom_t < 0 and z_t < -cfg.entry_z:
                            current_pos = -1.0
                    else:
                        if abs_z < cfg.exit_z:
                            current_pos = 0.0
                        # Exit if momentum reverses
                        elif current_pos > 0 and mom_t < 0:
                            current_pos = 0.0
                        elif current_pos < 0 and mom_t > 0:
                            current_pos = 0.0

                signals[t] = current_pos
                # Weight scaled by signal strength (how far z is from mean)
                weights[t] = min(abs_z / cfg.entry_z, 1.0) * cfg.max_weight if current_pos != 0.0 else 0.0

            sig_df = pd.DataFrame(
                {
                    f"{ticker}_signal": signals,
                    f"{ticker}_weight": weights,
                },
                index=prices.index,
            )
            result_frames.append(sig_df)

        if not result_frames:
            return pd.DataFrame(index=prices.index)

        return pd.concat(result_frames, axis=1)

    # ------------------------------------------------------------------ #
    #  Adaptive re-estimation                                              #
    # ------------------------------------------------------------------ #

    def generate_signals_adaptive(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate signals with periodic re-estimation of optimal d.

        Re-estimates d* every ``config.refit_period`` trading days
        (default 63 = quarterly).  This captures evolving market memory
        structure: an increasing d* over time suggests the market is
        becoming more efficient (less predictable memory).

        Parameters
        ----------
        prices : pd.DataFrame
            Full price history.

        Returns
        -------
        pd.DataFrame
            Concatenated signals covering the full price range.
        """
        cfg = self.config
        n = len(prices)

        # Need a minimum burn-in for initial fit
        min_burn_in = max(cfg.z_lookback * 2, 126)
        if n <= min_burn_in:
            logger.warning(
                "Price history (%d rows) too short for adaptive signals; "
                "need at least %d. Falling back to static fit.",
                n,
                min_burn_in,
            )
            self.fit(prices)
            return self.generate_signals(prices)

        all_signals: List[pd.DataFrame] = []
        t = min_burn_in

        while t < n:
            end = min(t + cfg.refit_period, n)

            # Fit on history up to current point
            train_data = prices.iloc[:t]
            signal_data = prices.iloc[t:end]

            self.fit(train_data)
            sigs = self.generate_signals(signal_data)
            all_signals.append(sigs)

            # Log d evolution
            for ticker in self._optimal_d:
                d_val = self._optimal_d[ticker]
                logger.info(
                    "Adaptive refit at index %d: %s d=%.3f (%s regime)",
                    t,
                    ticker,
                    d_val,
                    "mean-reversion" if d_val < cfg.d_regime_threshold else "momentum",
                )

            t = end

        if not all_signals:
            return pd.DataFrame(index=prices.index)

        return pd.concat(all_signals)

    # ------------------------------------------------------------------ #
    #  Diagnostic / analysis helpers                                       #
    # ------------------------------------------------------------------ #

    def get_d_history(self, ticker: str) -> pd.DataFrame:
        """Return the history of estimated d values for a given ticker.

        Useful for analysing how the market's memory structure evolves
        over time.  An increasing d trend suggests the market is becoming
        more efficient; a decreasing d suggests growing long-range
        dependence (potentially more predictable).

        Parameters
        ----------
        ticker : str
            Asset ticker symbol.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp``, ``d``.
        """
        if ticker not in self._d_history:
            return pd.DataFrame(columns=["timestamp", "d"])

        records = self._d_history[ticker]
        return pd.DataFrame(records, columns=["timestamp", "d"])

    def compute_fracdiff_series(
        self, prices: pd.DataFrame, ticker: str
    ) -> pd.Series:
        """Compute the frac-diff series for a single ticker at its fitted d*.

        Convenience method for analysis and plotting.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        ticker : str
            Column name in prices.

        Returns
        -------
        pd.Series
            Fractionally-differenced log-price series.
        """
        self.ensure_fitted()
        if ticker not in self._optimal_d:
            raise ValueError(f"No fitted d for ticker '{ticker}'.")

        log_series = np.log(prices[ticker].dropna())
        return frac_diff_fixed_width(
            log_series,
            self._optimal_d[ticker],
            threshold=self.config.weight_threshold,
        )

    def d_regime_summary(self) -> pd.DataFrame:
        """Return a summary of the current regime classification for all assets.

        Returns
        -------
        pd.DataFrame
            One row per asset with columns: ``ticker``, ``d``,
            ``adf_pvalue``, ``regime``, ``filter_width``.
        """
        self.ensure_fitted()
        rows = []
        cfg = self.config
        for ticker in self._optimal_d:
            d = self._optimal_d[ticker]
            regime = "mean_reversion" if d < cfg.d_regime_threshold else "momentum"
            width = len(_frac_diff_weights(d, cfg.weight_threshold))
            rows.append(
                {
                    "ticker": ticker,
                    "d": d,
                    "adf_pvalue": self._adf_pvalues.get(ticker, np.nan),
                    "regime": regime,
                    "filter_width": width,
                }
            )
        return pd.DataFrame(rows)
