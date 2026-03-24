"""
Benford's Law Deviation Strategy
=================================

Statistical forensics applied to financial markets: detects unusual price
activity by measuring deviations from the expected leading-digit distribution.

Mathematical foundation
-----------------------
Benford's Law states that for naturally occurring numerical data spanning
several orders of magnitude, the probability of leading digit *d* is:

    P(d) = log10(1 + 1/d),   d = 1, 2, ..., 9

Expected distribution:
    P(1) = 30.1%,  P(2) = 17.6%,  P(3) = 12.5%,  P(4) = 9.7%,
    P(5) = 7.9%,   P(6) = 6.7%,   P(7) = 5.8%,   P(8) = 5.1%,  P(9) = 4.6%

Financial prices generally conform to Benford's Law.  Deviations indicate:
  - Market manipulation or herding behaviour
  - Bubble formation (prices cluster around round numbers)
  - Artificial price support / resistance levels

Detection statistics:
  - Chi-squared: chi2 = sum (O_d - E_d)^2 / E_d, with 8 degrees of freedom
  - Kolmogorov-Smirnov: max |F_obs(d) - F_benford(d)| for mantissa uniformity
  - Mean Absolute Deviation (MAD) from Benford proportions

Second-digit analysis:
    P(d2 = d | d1) = sum_{d1=1}^{9} log10(1 + 1/(10*d1 + d))
    for d = 0, 1, ..., 9

Mantissa analysis:
    For naturally occurring data, the mantissa of log10(x) should be
    uniformly distributed on [0, 1).  Non-uniformity reveals manipulation
    or structural anomalies.

Strategy logic
--------------
1. **Benford Conformity Score**: rolling chi-squared statistic of first-
   significant-digit distribution vs Benford expectation, computed over
   a 63-day window on prices and volumes.

2. **Second-Digit & Mantissa Analysis**: second-digit Benford test and
   Kolmogorov-Smirnov test for mantissa uniformity provide supplementary
   anomaly detection.

3. **Trading Signal**: direction is inferred from *which* digits are
   over-represented.  Excess low digits (1, 2, 3) relative to Benford
   suggests accumulation (bullish); excess high digits (8, 9) suggests
   distribution (bearish).

4. **Cross-Sectional**: assets are ranked by Benford conformity.  High
   chi-squared (anomalous) assets receive stronger signals; conforming
   assets are left to other strategies.

References
----------
- Benford, F. (1938). "The law of anomalous numbers." Proceedings of the
  American Philosophical Society, 78(4), 551-572.
- Nigrini, M. J. (2012). "Benford's Law: Applications for Forensic
  Accounting, Auditing, and Fraud Detection." Wiley.
- Ley, E. (1996). "On the peculiar distribution of the U.S. stock indexes'
  digits." The American Statistician, 50(4), 311-313.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benford's Law constants
# ---------------------------------------------------------------------------

# First-digit Benford probabilities: P(d) = log10(1 + 1/d) for d = 1..9
BENFORD_FIRST_DIGIT: np.ndarray = np.array(
    [np.log10(1.0 + 1.0 / d) for d in range(1, 10)]
)  # shape (9,), sums to 1.0

# Second-digit Benford probabilities: P(d2 = k) for k = 0..9
# P(d2 = k) = sum_{d1=1}^{9} log10(1 + 1/(10*d1 + k))
BENFORD_SECOND_DIGIT: np.ndarray = np.array(
    [
        sum(np.log10(1.0 + 1.0 / (10 * d1 + k)) for d1 in range(1, 10))
        for k in range(10)
    ]
)  # shape (10,), sums to 1.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BenfordConfig:
    """Tuneable hyper-parameters for the Benford's Law strategy."""

    # --- Window parameters ---
    # Rolling window (trading days) over which digit distributions are measured.
    window: int = 63
    # Minimum number of valid observations within the window to compute stats.
    min_obs: int = 30

    # --- Chi-squared thresholds ---
    # chi2 with 8 df: p = 0.05 -> 15.51, p = 0.01 -> 20.09, p = 0.001 -> 26.12
    # We use a softer threshold for signal activation (anomaly detection).
    chi2_signal_threshold: float = 15.51  # ~p < 0.05 for 8 df
    chi2_strong_threshold: float = 26.12  # ~p < 0.001 for 8 df

    # --- Mantissa KS test ---
    # KS statistic threshold for mantissa non-uniformity.
    ks_threshold: float = 0.10

    # --- Signal construction ---
    # Weight assigned to volume-based Benford deviation (vs price-based).
    volume_weight: float = 0.3
    # Maximum absolute signal weight.
    max_signal_weight: float = 1.0
    # Smoothing span (EMA) for raw anomaly scores before signal generation.
    smooth_span: int = 5

    # --- Digit direction mapping ---
    # Digits considered "low" (accumulation indicator when over-represented).
    low_digits: Tuple[int, ...] = (1, 2, 3)
    # Digits considered "high" (distribution indicator when over-represented).
    high_digits: Tuple[int, ...] = (8, 9)

    # --- Cross-sectional ---
    # If True, rank assets by conformity and scale signals accordingly.
    cross_sectional_ranking: bool = True

    # --- Reversion signal ---
    # Bars of declining chi2 after anomaly to trigger reversion signal.
    reversion_lookback: int = 10


# ---------------------------------------------------------------------------
# Digit extraction utilities
# ---------------------------------------------------------------------------


def _first_significant_digit(x: np.ndarray) -> np.ndarray:
    """Extract the first significant (non-zero) digit of each element.

    Parameters
    ----------
    x : np.ndarray
        Array of positive numbers.

    Returns
    -------
    np.ndarray
        Integer array with values in {1, 2, ..., 9}.  Elements where the
        digit cannot be determined (zero, NaN, Inf) are set to 0.
    """
    result = np.zeros(len(x), dtype=np.int64)
    valid = np.isfinite(x) & (x > 0)
    if not valid.any():
        return result
    # First significant digit = floor(10^(frac(log10(|x|))))
    log_vals = np.log10(x[valid])
    mantissa = log_vals - np.floor(log_vals)
    digits = np.floor(10.0 ** mantissa).astype(np.int64)
    # Clamp to [1, 9] (numerical edge cases can produce 0 or 10)
    digits = np.clip(digits, 1, 9)
    result[valid] = digits
    return result


def _second_significant_digit(x: np.ndarray) -> np.ndarray:
    """Extract the second significant digit of each element.

    Parameters
    ----------
    x : np.ndarray
        Array of positive numbers.

    Returns
    -------
    np.ndarray
        Integer array with values in {0, 1, ..., 9}.  Invalid entries get -1.
    """
    result = np.full(len(x), -1, dtype=np.int64)
    valid = np.isfinite(x) & (x > 0)
    if not valid.any():
        return result
    log_vals = np.log10(x[valid])
    mantissa = log_vals - np.floor(log_vals)
    # First two significant digits: floor(10^(mantissa + 1))
    two_digits = np.floor(10.0 ** (mantissa + 1.0)).astype(np.int64)
    # Second digit is the units digit of two_digits
    second = two_digits % 10
    # Clamp to [0, 9]
    second = np.clip(second, 0, 9)
    result[valid] = second
    return result


def _log_mantissa(x: np.ndarray) -> np.ndarray:
    """Extract the mantissa of log10(|x|) for each element.

    Parameters
    ----------
    x : np.ndarray
        Array of positive numbers.

    Returns
    -------
    np.ndarray
        Mantissa values in [0, 1).  Invalid entries are NaN.
    """
    result = np.full(len(x), np.nan)
    valid = np.isfinite(x) & (x > 0)
    if not valid.any():
        return result
    log_vals = np.log10(x[valid])
    result[valid] = log_vals - np.floor(log_vals)
    return result


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def _chi_squared_vs_benford(
    digits: np.ndarray,
    expected_probs: np.ndarray = BENFORD_FIRST_DIGIT,
    n_categories: int = 9,
    digit_offset: int = 1,
) -> float:
    """Pearson chi-squared statistic for observed digits vs Benford.

    Parameters
    ----------
    digits : np.ndarray
        Integer array of observed digits.
    expected_probs : np.ndarray
        Expected probability for each digit category.
    n_categories : int
        Number of categories (9 for first digit, 10 for second).
    digit_offset : int
        Minimum digit value (1 for first-digit, 0 for second-digit).

    Returns
    -------
    float
        Chi-squared statistic.  Returns NaN if no valid observations.
    """
    valid = (digits >= digit_offset) & (digits < digit_offset + n_categories)
    obs = digits[valid]
    n = len(obs)
    if n < 10:
        return np.nan

    # Count occurrences of each digit
    observed_counts = np.zeros(n_categories, dtype=np.float64)
    for i in range(n_categories):
        observed_counts[i] = np.sum(obs == (i + digit_offset))

    expected_counts = expected_probs * n

    # Avoid division by zero for categories with zero expected count
    mask = expected_counts > 0
    chi2 = np.sum(
        (observed_counts[mask] - expected_counts[mask]) ** 2
        / expected_counts[mask]
    )
    return float(chi2)


def _digit_direction_score(
    digits: np.ndarray,
    low_digits: Tuple[int, ...] = (1, 2, 3),
    high_digits: Tuple[int, ...] = (8, 9),
) -> float:
    """Compute a directional score based on which digits are over-represented.

    Returns a value in [-1, +1]:
      - Positive: excess low digits (accumulation → bullish)
      - Negative: excess high digits (distribution → bearish)

    The score is the difference between the observed excess of low digits
    and the observed excess of high digits, normalised.
    """
    valid = (digits >= 1) & (digits <= 9)
    obs = digits[valid]
    n = len(obs)
    if n < 10:
        return 0.0

    # Observed proportions
    obs_prop = np.zeros(9)
    for i in range(9):
        obs_prop[i] = np.sum(obs == (i + 1)) / n

    # Excess over Benford for low and high digit groups
    low_excess = sum(
        obs_prop[d - 1] - BENFORD_FIRST_DIGIT[d - 1] for d in low_digits
    )
    high_excess = sum(
        obs_prop[d - 1] - BENFORD_FIRST_DIGIT[d - 1] for d in high_digits
    )

    # Directional score: positive if low digits are over-represented
    raw = low_excess - high_excess

    # Normalise to [-1, 1] using tanh scaling
    # The maximum possible excess is bounded, but tanh provides smooth clipping
    return float(np.tanh(raw * 10.0))


def _ks_mantissa_uniformity(mantissa: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic testing mantissa uniformity on [0, 1).

    Parameters
    ----------
    mantissa : np.ndarray
        Mantissa values (should be in [0, 1) for valid entries).

    Returns
    -------
    float
        KS statistic.  Returns NaN if insufficient data.
    """
    clean = mantissa[np.isfinite(mantissa)]
    if len(clean) < 10:
        return np.nan
    # Test against Uniform(0, 1)
    ks_stat, _ = sp_stats.kstest(clean, "uniform", args=(0.0, 1.0))
    return float(ks_stat)


def _mean_absolute_deviation_benford(digits: np.ndarray) -> float:
    """Mean Absolute Deviation of observed first-digit proportions from Benford.

    Nigrini's MAD conformity test:
      - MAD <= 0.006: close conformity
      - 0.006 < MAD <= 0.012: acceptable conformity
      - 0.012 < MAD <= 0.015: marginally acceptable
      - MAD > 0.015: non-conformity

    Parameters
    ----------
    digits : np.ndarray
        Integer array of first significant digits (values 1-9).

    Returns
    -------
    float
        MAD value.  Returns NaN if insufficient data.
    """
    valid = (digits >= 1) & (digits <= 9)
    obs = digits[valid]
    n = len(obs)
    if n < 10:
        return np.nan

    obs_prop = np.zeros(9)
    for i in range(9):
        obs_prop[i] = np.sum(obs == (i + 1)) / n

    mad = np.mean(np.abs(obs_prop - BENFORD_FIRST_DIGIT))
    return float(mad)


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


class BenfordsLawStrategy(Strategy):
    """Benford's Law deviation strategy for detecting unusual price activity.

    Monitors rolling first-digit, second-digit, and mantissa distributions
    of prices and volumes.  Deviations from theoretical Benford / uniform
    expectations are scored as anomalies, with direction inferred from which
    end of the digit spectrum is over-represented.

    Parameters
    ----------
    config : BenfordConfig | None
        Strategy hyper-parameters.  Defaults are used when *None*.
    """

    def __init__(self, config: BenfordConfig | None = None) -> None:
        self.config = config or BenfordConfig()
        super().__init__(
            name="BenfordsLaw",
            description=(
                "Benford's Law deviation detector: identifies unusual price "
                "activity through leading-digit distribution analysis, "
                "second-digit tests, and mantissa uniformity (KS test).  "
                "Directional signals are derived from digit-cluster asymmetry."
            ),
        )
        # Learned during fit
        self._baseline_chi2_price: float = 0.0
        self._baseline_chi2_volume: float = 0.0
        self._baseline_ks: float = 0.0
        self._baseline_mad: float = 0.0

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> "BenfordsLawStrategy":
        """Calibrate baseline Benford statistics from training data.

        Computes the "normal" chi-squared, KS, and MAD levels on historical
        prices so that anomaly detection during signal generation is
        calibrated relative to the asset's typical conformity.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (OHLCV or adjusted close).  If the DataFrame contains
            a ``Volume`` column it is used for volume-based Benford tests.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        # Use first price column (handle both single and multi-column)
        price_col = self._select_price_column(prices)
        price_series = prices[price_col].dropna()

        if len(price_series) < self.config.min_obs:
            raise ValueError(
                f"Insufficient data for fitting: need >= {self.config.min_obs} "
                f"observations, got {len(price_series)}."
            )

        # Baseline Benford statistics on the full training set
        price_vals = price_series.values.astype(np.float64)
        price_vals = price_vals[price_vals > 0]

        first_digits = _first_significant_digit(price_vals)
        self._baseline_chi2_price = _chi_squared_vs_benford(first_digits)
        self._baseline_mad = _mean_absolute_deviation_benford(first_digits)
        self._baseline_ks = _ks_mantissa_uniformity(
            _log_mantissa(price_vals)
        )

        # Volume baseline (if available)
        vol_col = self._find_volume_column(prices)
        if vol_col is not None:
            vol_vals = prices[vol_col].dropna().values.astype(np.float64)
            vol_vals = vol_vals[vol_vals > 0]
            if len(vol_vals) >= self.config.min_obs:
                vol_digits = _first_significant_digit(vol_vals)
                self._baseline_chi2_volume = _chi_squared_vs_benford(
                    vol_digits
                )
            else:
                self._baseline_chi2_volume = 0.0
        else:
            self._baseline_chi2_volume = 0.0

        self.parameters = {
            "baseline_chi2_price": self._baseline_chi2_price,
            "baseline_chi2_volume": self._baseline_chi2_volume,
            "baseline_ks": self._baseline_ks,
            "baseline_mad": self._baseline_mad,
        }
        self._fitted = True

        logger.info(
            "BenfordsLaw fitted: chi2_price=%.3f, chi2_volume=%.3f, "
            "ks=%.4f, mad=%.4f",
            self._baseline_chi2_price,
            self._baseline_chi2_volume,
            self._baseline_ks,
            self._baseline_mad,
        )
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals based on Benford's Law deviations.

        For each asset (column) in *prices*, the method:
        1. Computes rolling first-digit chi-squared statistics.
        2. Performs second-digit and mantissa uniformity tests.
        3. Determines signal direction from digit-cluster asymmetry.
        4. Optionally applies cross-sectional ranking.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data indexed by datetime.  If a ``Volume`` or
            ``{ticker}_volume`` column exists, it is used for the
            volume-based Benford component.

        Returns
        -------
        pd.DataFrame
            Columns ``signal`` and ``weight`` (single-asset) or
            ``{ticker}_signal`` and ``{ticker}_weight`` per asset.
        """
        self.ensure_fitted()

        # Identify tradeable price columns (exclude volume)
        price_cols = self._get_price_columns(prices)
        single_asset = len(price_cols) == 1

        result = pd.DataFrame(index=prices.index)
        anomaly_scores: Dict[str, pd.Series] = {}

        for col in price_cols:
            series = prices[col].dropna()
            if len(series) < self.config.min_obs:
                logger.warning(
                    "Skipping %s -- insufficient data (%d rows).", col, len(series)
                )
                sig = pd.Series(0.0, index=series.index)
                wgt = pd.Series(0.0, index=series.index)
                anom = pd.Series(0.0, index=series.index)
            else:
                vol_col = self._find_volume_column_for_ticker(prices, col)
                vol_series = (
                    prices[vol_col].dropna() if vol_col is not None else None
                )
                sig, wgt, anom = self._generate_single_asset(
                    series, vol_series
                )

            ticker = col
            anomaly_scores[ticker] = anom.reindex(prices.index, fill_value=0.0)

            if single_asset:
                result["signal"] = sig.reindex(prices.index, fill_value=0.0)
                result["weight"] = wgt.reindex(prices.index, fill_value=0.0)
            else:
                result[f"{ticker}_signal"] = sig.reindex(
                    prices.index, fill_value=0.0
                )
                result[f"{ticker}_weight"] = wgt.reindex(
                    prices.index, fill_value=0.0
                )

        # Cross-sectional ranking adjustment (multi-asset only)
        if not single_asset and self.config.cross_sectional_ranking:
            result = self._apply_cross_sectional_ranking(
                result, anomaly_scores, price_cols
            )

        return result

    # ------------------------------------------------------------------
    # Internal: single-asset signal generation
    # ------------------------------------------------------------------

    def _generate_single_asset(
        self,
        price_series: pd.Series,
        volume_series: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute Benford-based signals for a single asset.

        Returns
        -------
        signal : pd.Series
            Directional signal in [-1, +1].
        weight : pd.Series
            Position-sizing weight in [0, 1].
        anomaly : pd.Series
            Raw anomaly score (for cross-sectional ranking).
        """
        cfg = self.config
        n = len(price_series)
        idx = price_series.index

        signals = np.zeros(n, dtype=np.float64)
        weights = np.zeros(n, dtype=np.float64)
        anomaly_raw = np.zeros(n, dtype=np.float64)

        price_vals = price_series.values.astype(np.float64)

        # Pre-extract volume values if available
        vol_vals: Optional[np.ndarray] = None
        vol_idx: Optional[pd.Index] = None
        if volume_series is not None and len(volume_series) >= cfg.min_obs:
            vol_vals = volume_series.values.astype(np.float64)
            vol_idx = volume_series.index

        for t in range(cfg.window, n):
            # ---- Window of prices ----
            window_prices = price_vals[t - cfg.window + 1 : t + 1]
            pos_mask = (window_prices > 0) & np.isfinite(window_prices)
            window_pos = window_prices[pos_mask]

            if len(window_pos) < cfg.min_obs:
                continue

            # ---- First-digit chi-squared (prices) ----
            first_digits = _first_significant_digit(window_pos)
            chi2_price = _chi_squared_vs_benford(first_digits)
            if np.isnan(chi2_price):
                continue

            # ---- Second-digit chi-squared (prices) ----
            second_digits = _second_significant_digit(window_pos)
            chi2_second = _chi_squared_vs_benford(
                second_digits,
                expected_probs=BENFORD_SECOND_DIGIT,
                n_categories=10,
                digit_offset=0,
            )

            # ---- Mantissa KS test (prices) ----
            mantissa = _log_mantissa(window_pos)
            ks_stat = _ks_mantissa_uniformity(mantissa)

            # ---- MAD (prices) ----
            mad = _mean_absolute_deviation_benford(first_digits)

            # ---- Volume chi-squared ----
            chi2_volume = 0.0
            if vol_vals is not None and vol_idx is not None:
                # Align volume window with price window dates
                window_start_date = idx[t - cfg.window + 1]
                window_end_date = idx[t]
                vol_mask = (vol_idx >= window_start_date) & (
                    vol_idx <= window_end_date
                )
                window_vol = vol_vals[vol_mask.values] if hasattr(vol_mask, 'values') else vol_vals[vol_mask]
                window_vol = window_vol[
                    (window_vol > 0) & np.isfinite(window_vol)
                ]
                if len(window_vol) >= cfg.min_obs:
                    vol_digits = _first_significant_digit(window_vol)
                    chi2_volume = _chi_squared_vs_benford(vol_digits)
                    if np.isnan(chi2_volume):
                        chi2_volume = 0.0

            # ---- Composite anomaly score ----
            # Normalise chi2 by critical value for interpretability
            # chi2 / threshold gives a ratio: >1 means significant deviation
            price_anomaly = chi2_price / cfg.chi2_signal_threshold
            volume_anomaly = chi2_volume / cfg.chi2_signal_threshold

            # Second-digit anomaly (9 df for second digit)
            second_digit_anomaly = (
                chi2_second / 16.92 if not np.isnan(chi2_second) else 0.0
            )  # 16.92 = chi2 critical value at p=0.05 for 9 df

            # KS anomaly
            ks_anomaly = (
                ks_stat / cfg.ks_threshold if not np.isnan(ks_stat) else 0.0
            )

            # Weighted composite
            composite = (
                0.40 * price_anomaly
                + cfg.volume_weight * volume_anomaly
                + 0.15 * second_digit_anomaly
                + 0.15 * ks_anomaly
            )
            anomaly_raw[t] = composite

            # ---- Direction from digit asymmetry ----
            direction = _digit_direction_score(
                first_digits,
                low_digits=cfg.low_digits,
                high_digits=cfg.high_digits,
            )

            # ---- Signal and weight ----
            if composite > 1.0:
                # Anomaly detected: signal based on direction
                signals[t] = np.sign(direction) if abs(direction) > 0.05 else 0.0
                # Weight scales with anomaly strength, capped
                raw_weight = min(
                    (composite - 1.0) / (
                        cfg.chi2_strong_threshold / cfg.chi2_signal_threshold - 1.0
                    ),
                    1.0,
                )
                # Also scale by direction confidence
                raw_weight *= min(abs(direction), 1.0)
                weights[t] = min(raw_weight, cfg.max_signal_weight)
            else:
                signals[t] = 0.0
                weights[t] = 0.0

        # ---- Smooth signals ----
        signal_series = pd.Series(signals, index=idx)
        weight_series = pd.Series(weights, index=idx)
        anomaly_series = pd.Series(anomaly_raw, index=idx)

        if cfg.smooth_span > 1:
            # Smooth weights to reduce whipsaw
            weight_series = self.exponential_smooth(weight_series, span=cfg.smooth_span)
            weight_series = weight_series.clip(0.0, cfg.max_signal_weight)
            # Smooth anomaly score for reversion detection
            anomaly_series = self.exponential_smooth(
                anomaly_series, span=cfg.smooth_span
            )

        # ---- Reversion signal: declining anomaly after spike ----
        if cfg.reversion_lookback > 0:
            signal_series, weight_series = self._add_reversion_signals(
                signal_series, weight_series, anomaly_series
            )

        return signal_series, weight_series, anomaly_series

    def _add_reversion_signals(
        self,
        signals: pd.Series,
        weights: pd.Series,
        anomaly: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Add reversion signals when anomaly is declining from a high.

        When the anomaly score was recently above threshold but is now
        declining, this indicates the unusual activity is ending and the
        asset may revert to normal behavior.  We generate a weak contrary
        signal in this case.
        """
        cfg = self.config
        sig_arr = signals.values.copy()
        wgt_arr = weights.values.copy()
        anom_arr = anomaly.values

        lookback = cfg.reversion_lookback
        for t in range(lookback, len(anom_arr)):
            if sig_arr[t] != 0.0:
                # Already have a signal; skip reversion
                continue

            window = anom_arr[t - lookback : t + 1]
            peak = np.max(window[:-1])  # peak in prior window
            current = window[-1]

            # Reversion condition: peak was anomalous, current is declining
            if peak > 1.5 and current < peak * 0.6 and current < 1.0:
                # Find the direction of the original anomaly (from the peak)
                peak_idx = t - lookback + np.argmax(window[:-1])
                original_direction = sig_arr[peak_idx] if peak_idx < len(sig_arr) else 0.0

                if original_direction != 0.0:
                    # Reversion: opposite direction, weak weight
                    sig_arr[t] = -original_direction
                    wgt_arr[t] = 0.3 * min(
                        (peak - current) / peak, cfg.max_signal_weight
                    )

        return (
            pd.Series(sig_arr, index=signals.index),
            pd.Series(wgt_arr, index=weights.index),
        )

    # ------------------------------------------------------------------
    # Cross-sectional ranking
    # ------------------------------------------------------------------

    def _apply_cross_sectional_ranking(
        self,
        result: pd.DataFrame,
        anomaly_scores: Dict[str, pd.Series],
        price_cols: list,
    ) -> pd.DataFrame:
        """Scale signal weights by cross-sectional anomaly rank.

        Assets with higher anomaly scores (less Benford-conforming) receive
        amplified weights; conforming assets are dampened.  This focuses
        capital on the assets where the Benford signal has the most
        information content.
        """
        if len(price_cols) <= 1:
            return result

        # Build anomaly DataFrame
        anom_df = pd.DataFrame(anomaly_scores)

        for date in result.index:
            row_scores = anom_df.loc[date]
            valid = row_scores[row_scores != 0.0]
            if len(valid) < 2:
                continue

            # Rank: higher anomaly gets higher rank
            ranks = valid.rank(method="average")
            n = len(ranks)
            # Normalise ranks to [0.5, 1.5] range for scaling
            rank_scale = 0.5 + (ranks - 1.0) / max(n - 1, 1)

            for ticker in valid.index:
                wgt_col = f"{ticker}_weight"
                if wgt_col in result.columns:
                    result.loc[date, wgt_col] *= rank_scale[ticker]
                    # Re-clip after scaling
                    result.loc[date, wgt_col] = min(
                        float(result.loc[date, wgt_col]),
                        self.config.max_signal_weight,
                    )

        return result

    # ------------------------------------------------------------------
    # Column identification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_price_column(prices: pd.DataFrame) -> str:
        """Select the primary price column from a DataFrame."""
        candidates = ["Close", "close", "Adj Close", "adj_close", "AdjClose"]
        for c in candidates:
            if c in prices.columns:
                return c
        return prices.columns[0]

    @staticmethod
    def _find_volume_column(prices: pd.DataFrame) -> Optional[str]:
        """Find a volume column if present."""
        candidates = ["Volume", "volume", "vol", "Vol"]
        for c in candidates:
            if c in prices.columns:
                return c
        return None

    @staticmethod
    def _find_volume_column_for_ticker(
        prices: pd.DataFrame, ticker: str
    ) -> Optional[str]:
        """Find a volume column associated with a specific ticker."""
        candidates = [
            f"{ticker}_volume",
            f"{ticker}_Volume",
            "Volume",
            "volume",
        ]
        for c in candidates:
            if c in prices.columns:
                return c
        return None

    @staticmethod
    def _get_price_columns(prices: pd.DataFrame) -> list:
        """Get all price columns (excluding volume-like columns)."""
        exclude_patterns = {"volume", "vol", "Volume", "Vol"}
        cols = []
        for c in prices.columns:
            # Skip columns that look like volume
            lower = c.lower()
            if any(
                lower == pat.lower() or lower.endswith(f"_{pat.lower()}")
                for pat in exclude_patterns
            ):
                continue
            cols.append(c)
        return cols if cols else list(prices.columns)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def compute_diagnostics(
        self, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute rolling Benford diagnostics for analysis and plotting.

        Returns a DataFrame with columns:
          - ``chi2_first``: rolling first-digit chi-squared
          - ``chi2_second``: rolling second-digit chi-squared
          - ``ks_mantissa``: rolling mantissa KS statistic
          - ``mad``: rolling Mean Absolute Deviation from Benford
          - ``direction``: digit-direction score
          - ``anomaly``: composite anomaly score

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (single column or first column is used).

        Returns
        -------
        pd.DataFrame
            Diagnostics indexed by date.
        """
        self.ensure_fitted()
        cfg = self.config

        col = self._select_price_column(prices)
        series = prices[col].dropna()
        vals = series.values.astype(np.float64)
        n = len(vals)

        records = []
        for t in range(n):
            if t < cfg.window:
                records.append({
                    "chi2_first": np.nan,
                    "chi2_second": np.nan,
                    "ks_mantissa": np.nan,
                    "mad": np.nan,
                    "direction": np.nan,
                    "anomaly": np.nan,
                })
                continue

            window = vals[t - cfg.window + 1 : t + 1]
            pos = window[(window > 0) & np.isfinite(window)]

            if len(pos) < cfg.min_obs:
                records.append({
                    "chi2_first": np.nan,
                    "chi2_second": np.nan,
                    "ks_mantissa": np.nan,
                    "mad": np.nan,
                    "direction": np.nan,
                    "anomaly": np.nan,
                })
                continue

            fd = _first_significant_digit(pos)
            sd = _second_significant_digit(pos)
            mant = _log_mantissa(pos)

            chi2_1 = _chi_squared_vs_benford(fd)
            chi2_2 = _chi_squared_vs_benford(
                sd, BENFORD_SECOND_DIGIT, 10, 0
            )
            ks = _ks_mantissa_uniformity(mant)
            mad = _mean_absolute_deviation_benford(fd)
            direction = _digit_direction_score(fd)

            price_anom = chi2_1 / cfg.chi2_signal_threshold if not np.isnan(chi2_1) else 0.0
            second_anom = chi2_2 / 16.92 if not np.isnan(chi2_2) else 0.0
            ks_anom = ks / cfg.ks_threshold if not np.isnan(ks) else 0.0
            composite = 0.40 * price_anom + 0.15 * second_anom + 0.15 * ks_anom

            records.append({
                "chi2_first": chi2_1,
                "chi2_second": chi2_2,
                "ks_mantissa": ks,
                "mad": mad,
                "direction": direction,
                "anomaly": composite,
            })

        return pd.DataFrame(records, index=series.index)
