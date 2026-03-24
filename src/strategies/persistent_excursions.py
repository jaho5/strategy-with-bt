"""Persistent excursions strategy based on record theory and excursion analysis.

Uses graduate-level extreme value theory (record counting) and excursion
theory (connected intervals above/below a level) to detect whether an
asset is in a persistent (trending) or anti-persistent (mean-reverting)
regime, and trades accordingly.

Mathematical foundation
-----------------------
Record theory:
    In an i.i.d. sequence X_1, ..., X_n the probability that X_n is a
    record (i.e. X_n > max(X_1, ..., X_{n-1})) is exactly 1/n.  The
    expected number of records in the first n observations is the
    n-th harmonic number:

        H_n = sum_{k=1}^{n} 1/k  ~  ln(n) + gamma   (gamma ~ 0.5772)

    For dependent sequences (e.g. financial returns), deviations of the
    observed record count from H_n reveal persistence structure.

Excursion theory:
    An excursion is a maximal connected interval where a stochastic
    process stays strictly on one side of a reference level.  For
    standard Brownian motion the fraction of time spent above zero over
    [0, T] follows the arcsine distribution:

        P(fraction <= x) = (2/pi) * arcsin(sqrt(x)),   x in [0, 1]

    Financial price paths are *not* Brownian; departures from the
    arcsine law indicate exploitable persistence or anti-persistence.

Signal construction
-------------------
1.  Record excess  = (observed records - H_n) / sqrt(Var_records)
    where Var_records = H_n - H_n^{(2)} (second-order harmonic).
    Positive excess => momentum; negative => mean-reversion.

2.  Excursion persistence = empirical mean excursion length vs null.
    Under Brownian motion the expected excursion length in a window of
    size n scales as sqrt(n) (arcsine law consequence).  Observed /
    expected ratio > 1 => persistence; < 1 => anti-persistence.

3.  Combined signal: weighted average of z-scored record excess and
    excursion ratio, mapped through tanh to [-1, 1].
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from src.strategies.base import Strategy

# Euler-Mascheroni constant
_EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Record-theory primitives
# ---------------------------------------------------------------------------

def _harmonic(n: int) -> float:
    """Compute the n-th harmonic number H_n = sum_{k=1}^{n} 1/k."""
    if n <= 0:
        return 0.0
    return np.sum(1.0 / np.arange(1, n + 1))


def _harmonic_second(n: int) -> float:
    """Compute the n-th second-order harmonic H_n^{(2)} = sum_{k=1}^{n} 1/k^2."""
    if n <= 0:
        return 0.0
    return np.sum(1.0 / np.arange(1, n + 1) ** 2)


def _count_records(x: np.ndarray) -> tuple[int, int]:
    """Count upper and lower records in sequence *x*.

    A value x[i] is an upper record if x[i] > max(x[0..i-1]).
    A value x[i] is a lower record if x[i] < min(x[0..i-1]).
    The first element is always counted as both an upper and lower record.

    Parameters
    ----------
    x : 1-D array of floats

    Returns
    -------
    n_upper, n_lower : int
    """
    n = len(x)
    if n == 0:
        return 0, 0

    n_upper = 1
    n_lower = 1
    running_max = x[0]
    running_min = x[0]

    for i in range(1, n):
        if x[i] > running_max:
            n_upper += 1
            running_max = x[i]
        if x[i] < running_min:
            n_lower += 1
            running_min = x[i]

    return n_upper, n_lower


def _rolling_record_excess(
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling z-scored record excess for upper and lower records.

    For each position t >= window, we count records in values[t-window:t]
    and compare to the null expectation H_window.

    Returns
    -------
    upper_z, lower_z : 1-D arrays of length len(values)
        Z-scored excess record counts (NaN where window is incomplete).
    """
    n = len(values)
    upper_z = np.full(n, np.nan)
    lower_z = np.full(n, np.nan)

    h_n = _harmonic(window)
    h2_n = _harmonic_second(window)
    # Variance of record count for i.i.d. sequence:
    # Var(R_n) = H_n - H_n^{(2)}  (well-known result)
    var_records = h_n - h2_n
    std_records = np.sqrt(max(var_records, 1e-12))

    for t in range(window, n):
        seg = values[t - window: t]
        if np.any(np.isnan(seg)):
            continue
        n_up, n_lo = _count_records(seg)
        upper_z[t] = (n_up - h_n) / std_records
        lower_z[t] = (n_lo - h_n) / std_records

    return upper_z, lower_z


# ---------------------------------------------------------------------------
# Excursion-theory primitives
# ---------------------------------------------------------------------------

def _excursion_lengths(x: np.ndarray, level: float = 0.0) -> tuple[list[int], list[int]]:
    """Compute excursion lengths above and below *level*.

    An excursion above (below) is a maximal contiguous block of indices
    where x[i] > level (x[i] < level).  Ties (x[i] == level) break the
    excursion.

    Returns
    -------
    above_lengths, below_lengths : lists of int
    """
    above: list[int] = []
    below: list[int] = []
    n = len(x)
    if n == 0:
        return above, below

    current_len = 0
    current_side = 0  # +1 above, -1 below, 0 on level

    for i in range(n):
        if x[i] > level:
            side = 1
        elif x[i] < level:
            side = -1
        else:
            side = 0

        if side == current_side and side != 0:
            current_len += 1
        else:
            # Close previous excursion
            if current_len > 0:
                if current_side == 1:
                    above.append(current_len)
                elif current_side == -1:
                    below.append(current_len)
            current_side = side
            current_len = 1 if side != 0 else 0

    # Close final excursion
    if current_len > 0:
        if current_side == 1:
            above.append(current_len)
        elif current_side == -1:
            below.append(current_len)

    return above, below


def _arcsine_expected_mean_excursion(n: int) -> float:
    """Expected mean excursion length under the arcsine law (Brownian null).

    For a discrete random walk of length n, the expected number of
    sign changes is ~ sqrt(2n/pi), giving an expected mean excursion
    length of ~ n / sqrt(2n/pi) = sqrt(pi*n/2).

    This serves as the null-model baseline for comparison.
    """
    if n <= 1:
        return 1.0
    return np.sqrt(np.pi * n / 2.0)


def _rolling_excursion_ratio(
    values: np.ndarray,
    window: int,
    ma_window: int,
) -> np.ndarray:
    """Rolling ratio of observed mean excursion length to null expectation.

    The reference level for excursions is the rolling simple moving
    average over *ma_window* bars (the "center" the process deviates from).

    Returns
    -------
    ratio : 1-D array of length len(values)
        Observed / expected mean excursion length.  Values > 1 indicate
        persistent (trending) behavior; < 1 indicates anti-persistence.
        NaN where the window is incomplete or no excursions found.
    """
    n = len(values)
    ratio = np.full(n, np.nan)
    null_mean = _arcsine_expected_mean_excursion(window)

    # Precompute rolling MA for the reference level
    ma = pd.Series(values).rolling(ma_window, min_periods=ma_window).mean().values

    for t in range(max(window, ma_window), n):
        seg = values[t - window: t]
        ref = ma[t]
        if np.isnan(ref) or np.any(np.isnan(seg)):
            continue

        deviations = seg - ref
        above_lens, below_lens = _excursion_lengths(deviations, level=0.0)
        all_lens = above_lens + below_lens
        if len(all_lens) == 0:
            continue

        observed_mean = np.mean(all_lens)
        ratio[t] = observed_mean / null_mean

    return ratio


# ---------------------------------------------------------------------------
# Arcsine distribution test
# ---------------------------------------------------------------------------

def _arcsine_departure(
    values: np.ndarray,
    window: int,
    ma_window: int,
) -> np.ndarray:
    """Rolling departure of fraction-of-time-above-mean from arcsine CDF.

    Under the Brownian null, the fraction of time a path spends above
    its starting value follows the arcsine distribution on [0,1], which
    is Beta(1/2, 1/2).

    We compute the empirical fraction of time above the rolling MA in
    each window and measure its departure from the arcsine median (0.5)
    weighted by how unlikely it is under the null.

    Returns
    -------
    departure : 1-D array, positive means more time above than expected
                (bullish persistence), negative means less.
    """
    n = len(values)
    departure = np.full(n, np.nan)

    ma = pd.Series(values).rolling(ma_window, min_periods=ma_window).mean().values

    for t in range(max(window, ma_window), n):
        seg = values[t - window: t]
        ref = ma[t]
        if np.isnan(ref) or np.any(np.isnan(seg)):
            continue

        frac_above = np.mean(seg > ref)
        # CDF of the arcsine distribution (Beta(0.5, 0.5)) at frac_above
        # tells us how extreme this observation is.
        p = beta_dist.cdf(frac_above, 0.5, 0.5)
        # Map to signed z-like score: 0.5 -> 0, near 0 or 1 -> large magnitude
        departure[t] = 2.0 * (p - 0.5)

    return departure


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _tanh_normalise(values: np.ndarray) -> np.ndarray:
    """Map unbounded values to (-1, 1) via tanh after robust z-scoring."""
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return values.copy()
    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    scale = 1.4826 * mad if mad > 1e-12 else 1.0  # MAD -> std-equivalent
    z = (values - med) / scale
    return np.tanh(z)


def _ewma(x: np.ndarray, span: int) -> np.ndarray:
    """Exponentially weighted moving average (pure numpy)."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(x)
    out[0] = x[0] if np.isfinite(x[0]) else 0.0
    for i in range(1, len(x)):
        if np.isfinite(x[i]):
            out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


# ===========================================================================
# Strategy class
# ===========================================================================

class PersistentExcursionsStrategy(Strategy):
    """Trading strategy based on record theory and excursion analysis.

    Detects persistence (trending) vs. anti-persistence (mean-reverting)
    regimes by comparing observed record counts and excursion lengths
    to their null-model expectations, then trades momentum or contrarian
    accordingly.

    Parameters
    ----------
    record_window : int
        Rolling window for counting records.  Default 126 (~6 months).
    excursion_window : int
        Rolling window for excursion length analysis.  Default 126.
    ma_window : int
        Moving average window used as the reference level for excursions.
        Default 50.
    signal_smooth_span : int
        EMA span for smoothing the composite signal.  Default 10.
    persistence_threshold : float
        Minimum absolute composite signal to take a position (dead zone).
        Default 0.15.
    record_weight : float
        Weight for the record-excess component in the composite signal.
        Default 0.4.
    excursion_weight : float
        Weight for the excursion-ratio component.  Default 0.3.
    arcsine_weight : float
        Weight for the arcsine-departure component.  Default 0.3.
    """

    def __init__(
        self,
        record_window: int = 126,
        excursion_window: int = 126,
        ma_window: int = 50,
        signal_smooth_span: int = 10,
        persistence_threshold: float = 0.15,
        record_weight: float = 0.4,
        excursion_weight: float = 0.3,
        arcsine_weight: float = 0.3,
    ) -> None:
        super().__init__(
            name="PersistentExcursions",
            description=(
                "Record-theory and excursion-analysis strategy that detects "
                "persistence vs. anti-persistence regimes and trades momentum "
                "or mean-reversion accordingly."
            ),
        )
        self.record_window = record_window
        self.excursion_window = excursion_window
        self.ma_window = ma_window
        self.signal_smooth_span = signal_smooth_span
        self.persistence_threshold = persistence_threshold
        self.record_weight = record_weight
        self.excursion_weight = excursion_weight
        self.arcsine_weight = arcsine_weight

        # Learned during fit()
        self._calibrated_weights: Optional[Dict[str, float]] = None

    # -----------------------------------------------------------------
    # Internal signal builders (per-asset)
    # -----------------------------------------------------------------

    def _record_signal(self, prices: np.ndarray) -> np.ndarray:
        """Compute record-excess signal from price levels.

        Excess upper records => strong upward momentum (bullish).
        Excess lower records => strong downward momentum (bearish).
        The net signal is (upper_z - lower_z): positive when upward
        records dominate, negative when downward records dominate.
        Deficit of both upper and lower records indicates mean-reversion.
        """
        upper_z, lower_z = _rolling_record_excess(prices, self.record_window)

        # Net record signal:
        # - If upper records excess >> 0 and lower records deficit => bullish trend
        # - If lower records excess >> 0 and upper records deficit => bearish trend
        # - If both deficit => mean-reverting (signal near 0)
        # We use (upper_z - lower_z) as the directional signal and
        # (upper_z + lower_z) as the persistence magnitude.
        net_direction = upper_z - lower_z
        persistence = upper_z + lower_z

        # Weight the direction by the absolute persistence level:
        # When persistence is high (both record types in excess), trends
        # are strong; when low/negative, contrarian signals dominate.
        signal = np.where(
            np.isfinite(net_direction) & np.isfinite(persistence),
            net_direction * np.clip(1.0 + 0.5 * persistence, 0.3, 3.0),
            np.nan,
        )
        return signal

    def _excursion_signal(self, prices: np.ndarray) -> np.ndarray:
        """Compute excursion-persistence signal.

        Ratio > 1 => excursions last longer than Brownian null => trending.
        Ratio < 1 => excursions are cut short => mean-reverting.

        We combine the ratio (magnitude of persistence) with the
        direction inferred from whether the current price is above or
        below its moving average.
        """
        ratio = _rolling_excursion_ratio(
            prices, self.excursion_window, self.ma_window
        )

        # Direction: current price vs its MA
        ma = pd.Series(prices).rolling(
            self.ma_window, min_periods=self.ma_window
        ).mean().values
        direction = np.where(
            np.isfinite(ma),
            np.sign(prices - ma),
            0.0,
        )

        # Persistence score: log(ratio) is zero-centered
        # Positive log-ratio + above MA => long; below MA => short
        # Negative log-ratio => contrarian: flip the direction
        log_ratio = np.where(
            np.isfinite(ratio) & (ratio > 0),
            np.log(ratio),
            np.nan,
        )

        signal = np.where(
            np.isfinite(log_ratio),
            log_ratio * direction,
            np.nan,
        )
        return signal

    def _arcsine_signal(self, prices: np.ndarray) -> np.ndarray:
        """Arcsine-departure signal.

        Measures how much the fraction-of-time-above-mean departs from
        the arcsine distribution.  Positive departure = bullish persistence,
        negative = bearish persistence.
        """
        return _arcsine_departure(
            prices, self.excursion_window, self.ma_window
        )

    def _composite_signal(self, prices: np.ndarray) -> np.ndarray:
        """Build the combined signal for a single asset.

        Steps:
        1. Compute three sub-signals (record, excursion, arcsine).
        2. Normalise each via tanh to [-1, 1].
        3. Weighted combination.
        4. Smooth with EMA.
        """
        w = self._calibrated_weights or {
            "record": self.record_weight,
            "excursion": self.excursion_weight,
            "arcsine": self.arcsine_weight,
        }

        sig_record = self._record_signal(prices)
        sig_excursion = self._excursion_signal(prices)
        sig_arcsine = self._arcsine_signal(prices)

        # Tanh-normalise each to [-1, 1]
        norm_record = _tanh_normalise(sig_record)
        norm_excursion = _tanh_normalise(sig_excursion)
        norm_arcsine = _tanh_normalise(sig_arcsine)

        # Weighted combination (handle NaN propagation)
        composite = np.full(len(prices), np.nan)
        for i in range(len(prices)):
            parts = []
            weights = []
            for val, wt_key in [
                (norm_record[i], "record"),
                (norm_excursion[i], "excursion"),
                (norm_arcsine[i], "arcsine"),
            ]:
                if np.isfinite(val):
                    parts.append(val * w[wt_key])
                    weights.append(w[wt_key])
            if weights:
                composite[i] = np.sum(parts) / np.sum(weights)

        # Smooth
        finite_mask = np.isfinite(composite)
        if np.any(finite_mask):
            first_valid = np.argmax(finite_mask)
            smoothed = composite.copy()
            smoothed[first_valid:] = _ewma(
                composite[first_valid:], self.signal_smooth_span
            )
            return smoothed

        return composite

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "PersistentExcursionsStrategy":
        """Calibrate component weights using historical predictive power.

        For each asset, we compute the rolling rank correlation (Spearman)
        between each sub-signal and 1-day forward returns, then set
        combination weights proportional to mean |correlation|.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data.  Columns are tickers, index is datetime.

        Returns
        -------
        self
        """
        from scipy.stats import spearmanr

        ic_window = max(self.record_window, self.excursion_window) + 60

        accum: Dict[str, list[float]] = {
            "record": [],
            "excursion": [],
            "arcsine": [],
        }

        for col in prices.columns:
            p = prices[col].dropna().values.astype(np.float64)
            if len(p) < ic_window + 60:
                continue

            fwd_ret = np.empty(len(p))
            fwd_ret[:-1] = p[1:] / p[:-1] - 1.0
            fwd_ret[-1] = np.nan

            sig_r = self._record_signal(p)
            sig_e = self._excursion_signal(p)
            sig_a = self._arcsine_signal(p)

            for name, sig in [
                ("record", sig_r),
                ("excursion", sig_e),
                ("arcsine", sig_a),
            ]:
                # Rolling rank IC over the latter portion of the data
                valid = np.isfinite(sig) & np.isfinite(fwd_ret)
                if valid.sum() < 30:
                    continue
                sig_valid = sig[valid]
                ret_valid = fwd_ret[valid]
                corr, _ = spearmanr(sig_valid, ret_valid)
                if np.isfinite(corr):
                    accum[name].append(abs(corr))

        # Derive weights proportional to mean |IC|
        raw: Dict[str, float] = {}
        for name, vals in accum.items():
            raw[name] = float(np.mean(vals)) if vals else 1.0 / 3

        total = sum(raw.values())
        if total < 1e-12:
            self._calibrated_weights = {k: 1.0 / 3 for k in raw}
        else:
            self._calibrated_weights = {k: v / total for k, v in raw.items()}

        self.parameters = {
            "component_weights": self._calibrated_weights.copy(),
        }
        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals from price data.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data with columns as tickers and DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Columns: ``{ticker}_signal`` and ``{ticker}_weight`` for each
            ticker.  Signals are in {-1, 0, +1}; weights are in [0, 1]
            representing confidence / position size derived from departure
            magnitude.
        """
        result = pd.DataFrame(index=prices.index)

        min_bars = max(self.record_window, self.excursion_window, self.ma_window)

        for col in prices.columns:
            series = prices[col].dropna()
            p = series.values.astype(np.float64)

            sig_col = f"{col}_signal" if len(prices.columns) > 1 else "signal"
            wt_col = f"{col}_weight" if len(prices.columns) > 1 else "weight"

            if len(p) < min_bars + 10:
                result[sig_col] = 0.0
                result[wt_col] = 0.0
                continue

            composite = self._composite_signal(p)

            # Discretise into {-1, 0, +1}
            signal = np.where(
                composite > self.persistence_threshold,
                1.0,
                np.where(composite < -self.persistence_threshold, -1.0, 0.0),
            )

            # Weight = magnitude of composite (clipped to [0, 1])
            # Larger departures from the null model => higher confidence
            weight = np.clip(np.abs(composite), 0.0, 1.0)
            weight = np.where(np.isfinite(weight), weight, 0.0)

            # Zero weight where signal is flat
            weight = np.where(signal != 0.0, weight, 0.0)

            # Align back to the full index (series may have been dropna'd)
            sig_series = pd.Series(signal, index=series.index)
            wt_series = pd.Series(weight, index=series.index)

            result[sig_col] = sig_series.reindex(prices.index, fill_value=0.0)
            result[wt_col] = wt_series.reindex(prices.index, fill_value=0.0)

        return result
