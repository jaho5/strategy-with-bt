"""Martingale Difference Hypothesis Strategy
=============================================

Exploits statistically significant departures from the martingale difference
sequence (MDS) property predicted by the Efficient Market Hypothesis.

Mathematical foundation
-----------------------
Under the EMH, returns form a martingale difference sequence:

    E[r_t | F_{t-1}] = 0

The primary diagnostic is the *Variance Ratio* (Lo & MacKinlay 1988):

    VR(q) = Var(r_t(q)) / (q * Var(r_t))

where r_t(q) = r_t + r_{t-1} + ... + r_{t-q+1} is the q-period return.

Under a random walk:  VR(q) = 1 for all q.
    VR(q) > 1  =>  positive autocorrelation (momentum)
    VR(q) < 1  =>  negative autocorrelation (mean-reversion)

The strategy uses the Lo-MacKinlay heteroskedasticity-robust z-statistic:

    z(q) = (VR(q) - 1) / sqrt(Var_hat(VR(q)))

Under H0:  z ~ N(0, 1).

Adaptive regime detection
-------------------------
Rolling VR analysis across multiple horizons (q = 2, 5, 10, 20) classifies
the local return-generating process into one of three regimes:

*   **Momentum**  (z > +1.96):  trend-follow (sign of recent return).
*   **Mean-reversion** (z < -1.96):  contrarian (opposite of recent return).
*   **Efficient** (|z| <= 1.96):  no trade.

A multi-horizon composite signal is constructed by weighting each q
proportionally to its |z|, producing a single scalar in [-1, +1].

Position sizing is proportional to |VR(q) - 1|, the departure magnitude,
with a stability discount when VR estimates are noisy across rolling windows.

References
----------
- Lo, A. W. & MacKinlay, A. C. (1988). "Stock Market Prices Do Not Follow
  Random Walks: Evidence from a Simple Specification Test". Review of
  Financial Studies, 1(1), 41-66.
- Choi, I. (2009). "Testing the Random Walk Hypothesis for Real Exchange
  Rates". Journal of Applied Econometrics, 14(3), 293-308.
- Escanciano, J. C. & Velasco, C. (2006). "Generalized Spectral Tests for
  the Martingale Difference Hypothesis". Journal of Econometrics, 134(1),
  151-185.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MartingaleDifferenceConfig:
    """Hyper-parameters for the Martingale Difference strategy."""

    # Variance-ratio horizons
    vr_horizons: List[int] = field(default_factory=lambda: [2, 5, 10, 20])

    # Rolling estimation window (trading days)
    rolling_window: int = 252

    # Significance level for the z-test (two-sided)
    z_critical: float = 1.96

    # Momentum signal: lookback for return sign determination
    momentum_lookback: int = 20

    # Mean-reversion signal: lookback for z-score normalisation
    mr_lookback: int = 20

    # Position sizing
    max_weight: float = 1.0
    min_weight: float = 0.05

    # Stability discount: if the standard deviation of VR across
    # sub-windows exceeds this fraction of |VR - 1|, halve the weight.
    stability_threshold: float = 0.5

    # Signal smoothing span (EMA)
    smoothing_span: int = 5

    # Minimum number of observations required before generating signals
    min_observations: int = 60


# ---------------------------------------------------------------------------
# Variance Ratio computation (Lo-MacKinlay, heteroskedasticity-robust)
# ---------------------------------------------------------------------------

def _variance_ratio(
    returns: np.ndarray, q: int
) -> Tuple[float, float, float]:
    """Compute the variance ratio VR(q) and the heteroskedasticity-robust
    z-statistic of Lo & MacKinlay (1988).

    Parameters
    ----------
    returns : 1-D array
        Log returns (length T).
    q : int
        Aggregation horizon (q >= 2).

    Returns
    -------
    vr : float
        Variance ratio VR(q).
    z_stat : float
        Heteroskedasticity-robust z-statistic.
    p_value : float
        Two-sided p-value under H0: VR(q) = 1.
    """
    T = len(returns)
    if T < q + 1:
        return np.nan, np.nan, np.nan

    # Single-period variance (unbiased)
    mu = np.mean(returns)
    sigma2_1 = np.sum((returns - mu) ** 2) / (T - 1)

    if sigma2_1 < 1e-15:
        return np.nan, np.nan, np.nan

    # q-period overlapping returns
    r_q = np.array([
        np.sum(returns[t - q + 1 : t + 1])
        for t in range(q - 1, T)
    ])
    nq = len(r_q)
    sigma2_q = np.sum((r_q - q * mu) ** 2) / (nq - 1)

    vr = sigma2_q / (q * sigma2_1)

    # ------------------------------------------------------------------
    # Heteroskedasticity-robust variance of VR(q) -- Lo & MacKinlay 1988
    # ------------------------------------------------------------------
    # delta_j = sum_{t=j+1}^{T} (r_t - mu)^2 * (r_{t-j} - mu)^2
    #           / [sum_{t=1}^{T} (r_t - mu)^2]^2
    #
    # Var(VR) = sum_{j=1}^{q-1} [2(q-j)/q]^2 * delta_j
    # ------------------------------------------------------------------
    centered = returns - mu
    sum_sq = np.sum(centered ** 2)

    if sum_sq < 1e-15:
        return vr, np.nan, np.nan

    var_vr = 0.0
    for j in range(1, q):
        # delta_j: heteroskedasticity-robust kernel
        prod = centered[j:] ** 2 * centered[:-j] ** 2
        delta_j = np.sum(prod) / (sum_sq ** 2)
        weight = (2.0 * (q - j) / q) ** 2
        var_vr += weight * delta_j

    if var_vr <= 0:
        return vr, np.nan, np.nan

    z_stat = (vr - 1.0) / np.sqrt(var_vr)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(z_stat)))

    return vr, z_stat, p_value


def _rolling_variance_ratios(
    returns: np.ndarray,
    q: int,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling VR(q), z-statistics, and p-values.

    Parameters
    ----------
    returns : 1-D array of log returns (length T).
    q : int
        Variance ratio horizon.
    window : int
        Rolling window size.

    Returns
    -------
    vr_series, z_series, p_series : arrays of length T (NaN-padded).
    """
    T = len(returns)
    vr_out = np.full(T, np.nan)
    z_out = np.full(T, np.nan)
    p_out = np.full(T, np.nan)

    for t in range(window - 1, T):
        r_win = returns[t - window + 1 : t + 1]
        vr, z, p = _variance_ratio(r_win, q)
        vr_out[t] = vr
        z_out[t] = z
        p_out[t] = p

    return vr_out, z_out, p_out


# ---------------------------------------------------------------------------
# Automatic Variance Ratio (Choi 2009-inspired data-adaptive lag selection)
# ---------------------------------------------------------------------------

def _automatic_variance_ratio(
    returns: np.ndarray,
    max_q: int = 30,
) -> Tuple[int, float, float]:
    """Select the optimal horizon q* that maximises |z(q)| and return
    the corresponding VR and z-statistic.

    This is a simplified version of Choi (2009)'s automatic variance ratio
    that scans q = 2..max_q and picks the horizon with the strongest
    departure from the random-walk null.

    Parameters
    ----------
    returns : 1-D array of log returns.
    max_q : int
        Maximum horizon to search.

    Returns
    -------
    q_star : int
        Optimal horizon.
    vr_star : float
        VR(q*).
    z_star : float
        z(q*).
    """
    T = len(returns)
    best_q = 2
    best_vr = np.nan
    best_z = 0.0

    for q in range(2, min(max_q + 1, T // 2)):
        vr, z, _ = _variance_ratio(returns, q)
        if np.isfinite(z) and np.abs(z) > np.abs(best_z):
            best_q = q
            best_vr = vr
            best_z = z

    return best_q, best_vr, best_z


# ---------------------------------------------------------------------------
# Generalized Spectral Test (Escanciano & Velasco 2006, simplified)
# ---------------------------------------------------------------------------

def _generalised_spectral_test(
    returns: np.ndarray,
    max_lag: int = 20,
    n_freq: int = 50,
) -> float:
    """Simplified generalized spectral (GS) test statistic for nonlinear
    serial dependence.

    Computes the integrated squared difference between the conditional and
    marginal characteristic functions across frequencies, using the sample
    covariance of exp(i*w*r_t) and exp(i*w*r_{t-j}).

    A large value indicates nonlinear predictability not captured by
    variance-ratio (linear) tests.

    Parameters
    ----------
    returns : 1-D array of log returns.
    max_lag : int
        Maximum lag order.
    n_freq : int
        Number of frequency grid points in [0, pi].

    Returns
    -------
    gs_stat : float
        Test statistic (larger = more nonlinear dependence).
    """
    T = len(returns)
    if T < max_lag + 10:
        return 0.0

    freqs = np.linspace(0.01, np.pi, n_freq)
    stat = 0.0

    for j in range(1, max_lag + 1):
        for w in freqs:
            # exp(i * w * r_t)  and  exp(i * w * r_{t-j})
            e1 = np.exp(1j * w * returns[j:])
            e2 = np.exp(1j * w * returns[:-j])

            # Sample cross-covariance of the complex exponentials
            cov = np.mean(e1 * np.conj(e2)) - np.mean(e1) * np.mean(np.conj(e2))
            stat += np.abs(cov) ** 2

    # Normalise by number of terms
    stat /= (max_lag * n_freq)
    return float(stat)


# ---------------------------------------------------------------------------
# Multi-horizon composite signal
# ---------------------------------------------------------------------------

def _composite_signal(
    z_stats: Dict[int, float],
    vr_values: Dict[int, float],
    z_critical: float,
) -> Tuple[float, float]:
    """Compute a multi-horizon composite signal from VR z-statistics.

    The signal is a weighted average of per-horizon directional signals,
    where the weight for horizon q is |z(q)| / sum(|z|).

    Parameters
    ----------
    z_stats : {q: z(q)} for each horizon.
    vr_values : {q: VR(q)} for each horizon.
    z_critical : critical value for significance.

    Returns
    -------
    signal : float in [-1, +1].
        > 0 indicates momentum (trend-follow), < 0 mean-reversion.
    confidence : float in [0, 1].
        Strength / confidence of the combined signal.
    """
    total_abs_z = sum(np.abs(z) for z in z_stats.values() if np.isfinite(z))

    if total_abs_z < 1e-10:
        return 0.0, 0.0

    weighted_signal = 0.0
    weighted_departure = 0.0

    for q, z in z_stats.items():
        if not np.isfinite(z):
            continue

        abs_z = np.abs(z)
        w = abs_z / total_abs_z

        # Per-horizon regime direction
        if z > z_critical:
            direction = 1.0       # momentum
        elif z < -z_critical:
            direction = -1.0      # mean-reversion
        else:
            direction = 0.0       # efficient -- no signal

        weighted_signal += w * direction

        # Departure magnitude for sizing
        vr = vr_values.get(q, 1.0)
        if np.isfinite(vr):
            weighted_departure += w * np.abs(vr - 1.0)

    # Clip to [-1, 1]
    signal = float(np.clip(weighted_signal, -1.0, 1.0))
    confidence = float(np.clip(weighted_departure, 0.0, 1.0))

    return signal, confidence


# ===========================================================================
# Strategy class
# ===========================================================================

class MartingaleDifferenceStrategy(Strategy):
    """Trading strategy based on testing the Martingale Difference Hypothesis.

    Workflow
    --------
    1. **fit(prices)** -- run rolling VR analysis and automatic VR lag
       selection on the training set, calibrate stability parameters and
       the generalized spectral nonlinearity flag.

    2. **generate_signals(prices)** -- for each asset and each date with
       sufficient history, compute rolling VR(q) for q in {2, 5, 10, 20},
       classify the local regime (momentum / mean-reversion / efficient),
       and emit weighted position signals.
    """

    def __init__(self, config: Optional[MartingaleDifferenceConfig] = None) -> None:
        super().__init__(
            name="Martingale Difference",
            description=(
                "Exploits departures from the martingale difference hypothesis "
                "via multi-horizon variance ratio analysis."
            ),
        )
        self.config = config or MartingaleDifferenceConfig()

        # Learned parameters (populated by fit)
        self._vr_stability: Dict[str, float] = {}
        self._auto_vr_q: Dict[str, int] = {}
        self._gs_nonlinear: Dict[str, bool] = {}

    # -----------------------------------------------------------------
    # Internal: per-asset signal construction
    # -----------------------------------------------------------------

    def _compute_asset_signal(
        self,
        log_returns: np.ndarray,
        prices: np.ndarray,
        stability: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute signal and weight arrays for a single asset.

        Parameters
        ----------
        log_returns : 1-D array of log returns (length T).
        prices : 1-D array of prices (length T+1, aligned so that
                 log_returns[t] = log(prices[t+1]/prices[t])).
        stability : float in [0, 1].  1 = fully stable VR estimates.

        Returns
        -------
        signal : 1-D array (length T+1) in {-1, 0, +1}.
        weight : 1-D array (length T+1) in [0, 1].
        """
        cfg = self.config
        T = len(log_returns)
        n_prices = len(prices)
        signal = np.zeros(n_prices)
        weight = np.zeros(n_prices)

        if T < cfg.min_observations:
            return signal, weight

        # Pre-compute rolling VR for each horizon
        vr_arrays: Dict[int, np.ndarray] = {}
        z_arrays: Dict[int, np.ndarray] = {}

        for q in cfg.vr_horizons:
            vr_arr, z_arr, _ = _rolling_variance_ratios(
                log_returns, q, cfg.rolling_window
            )
            vr_arrays[q] = vr_arr
            z_arrays[q] = z_arr

        # Generate signal for each date
        for t in range(cfg.rolling_window - 1, T):
            # Collect z-stats and VR values at this date
            z_dict: Dict[int, float] = {}
            vr_dict: Dict[int, float] = {}

            for q in cfg.vr_horizons:
                z_val = z_arrays[q][t]
                vr_val = vr_arrays[q][t]
                if np.isfinite(z_val) and np.isfinite(vr_val):
                    z_dict[q] = z_val
                    vr_dict[q] = vr_val

            if not z_dict:
                continue

            # Multi-horizon composite
            composite, departure = _composite_signal(
                z_dict, vr_dict, cfg.z_critical
            )

            if composite == 0.0:
                continue

            # Determine directional signal from recent returns
            # (prices array is 1 element longer than log_returns)
            price_idx = t + 1  # index into prices array
            lookback_start = max(0, t - cfg.momentum_lookback + 1)
            recent_return = np.sum(log_returns[lookback_start : t + 1])

            if composite > 0:
                # Momentum regime: follow the trend
                direction = np.sign(recent_return)
            else:
                # Mean-reversion regime: fade the move
                direction = -np.sign(recent_return)

            if direction == 0.0:
                continue

            # Position weight: departure from RW scaled by stability
            raw_weight = departure * stability
            raw_weight = float(np.clip(raw_weight, cfg.min_weight, cfg.max_weight))

            signal[price_idx] = direction
            weight[price_idx] = raw_weight

        return signal, weight

    def _estimate_stability(
        self, log_returns: np.ndarray
    ) -> float:
        """Estimate VR stability by comparing VR estimates across
        non-overlapping sub-windows.

        Returns a value in [0, 1] where 1 means highly stable.
        """
        cfg = self.config
        T = len(log_returns)
        n_sub = max(2, T // cfg.rolling_window)
        sub_len = T // n_sub

        if sub_len < cfg.min_observations:
            return 0.5  # not enough data; use neutral

        vr_sub_estimates: List[List[float]] = [[] for _ in cfg.vr_horizons]

        for i in range(n_sub):
            start = i * sub_len
            end = start + sub_len
            r_sub = log_returns[start:end]

            for j, q in enumerate(cfg.vr_horizons):
                vr, _, _ = _variance_ratio(r_sub, q)
                if np.isfinite(vr):
                    vr_sub_estimates[j].append(vr)

        # Stability = 1 / (1 + mean_cv)  where cv = std / |mean_departure|
        cvs = []
        for estimates in vr_sub_estimates:
            if len(estimates) < 2:
                continue
            arr = np.array(estimates)
            departure = np.abs(arr - 1.0)
            mean_dep = np.mean(departure)
            std_dep = np.std(departure, ddof=1)
            if mean_dep > 1e-10:
                cvs.append(std_dep / mean_dep)

        if not cvs:
            return 0.5

        mean_cv = np.mean(cvs)
        stability = 1.0 / (1.0 + mean_cv)
        return float(np.clip(stability, 0.1, 1.0))

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "MartingaleDifferenceStrategy":
        """Calibrate strategy parameters on training data.

        For each asset, estimates:
        1. VR stability across sub-windows.
        2. Automatic variance-ratio optimal lag (Choi 2009 inspired).
        3. Generalized spectral test for nonlinear dependence.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (columns = tickers, index = DatetimeIndex).

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        cfg = self.config
        self._vr_stability = {}
        self._auto_vr_q = {}
        self._gs_nonlinear = {}

        for col in prices.columns:
            p = prices[col].dropna().values.astype(np.float64)
            if len(p) < cfg.min_observations + 1:
                continue

            # Guard against non-positive prices
            if np.any(p <= 0):
                p = p[p > 0]
                if len(p) < cfg.min_observations + 1:
                    continue

            log_ret = np.diff(np.log(p))

            # 1. Stability
            self._vr_stability[col] = self._estimate_stability(log_ret)

            # 2. Automatic VR lag selection
            q_star, vr_star, z_star = _automatic_variance_ratio(log_ret)
            self._auto_vr_q[col] = q_star

            # 3. Generalized spectral test for nonlinear dependence
            gs_stat = _generalised_spectral_test(log_ret, max_lag=10, n_freq=30)
            # Rough threshold: if GS statistic exceeds 2x the baseline
            # computed under independence (shuffled returns), flag nonlinear.
            rng = np.random.RandomState(42)
            baseline_stats = []
            for _ in range(20):
                shuffled = rng.permutation(log_ret)
                baseline_stats.append(
                    _generalised_spectral_test(shuffled, max_lag=10, n_freq=30)
                )
            gs_baseline = np.mean(baseline_stats)
            self._gs_nonlinear[col] = bool(gs_stat > 2.0 * gs_baseline)

            logger.info(
                "Asset %s: stability=%.3f, auto_q=%d (VR=%.4f, z=%.3f), "
                "GS_nonlinear=%s",
                col,
                self._vr_stability[col],
                q_star,
                vr_star,
                z_star,
                self._gs_nonlinear[col],
            )

        self.parameters = {
            "vr_stability": dict(self._vr_stability),
            "auto_vr_q": dict(self._auto_vr_q),
            "gs_nonlinear": dict(self._gs_nonlinear),
        }
        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals from price data.

        For each asset, computes rolling variance ratios at multiple
        horizons, classifies the local regime, and emits directional
        signals with confidence-based position weights.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (columns = tickers, index = DatetimeIndex).

        Returns
        -------
        pd.DataFrame
            Columns: ``{ticker}_signal`` and ``{ticker}_weight`` for each
            ticker.  Signal in {-1, 0, +1}, weight in [0, 1].
        """
        self.validate_prices(prices)
        cfg = self.config

        result_cols: Dict[str, np.ndarray] = {}

        for col in prices.columns:
            p = prices[col].values.astype(np.float64)

            # Forward-fill NaN prices for continuity
            mask_nan = np.isnan(p)
            if mask_nan.all():
                result_cols[f"{col}_signal"] = np.zeros(len(prices))
                result_cols[f"{col}_weight"] = np.zeros(len(prices))
                continue

            # Forward-fill
            if mask_nan.any():
                p_filled = p.copy()
                for i in range(1, len(p_filled)):
                    if np.isnan(p_filled[i]):
                        p_filled[i] = p_filled[i - 1]
                # Back-fill the leading NaNs
                first_valid = np.argmax(~mask_nan)
                p_filled[:first_valid] = p_filled[first_valid]
                p = p_filled

            # Guard against non-positive prices
            p[p <= 0] = np.nan
            valid_mask = ~np.isnan(p)
            if valid_mask.sum() < cfg.min_observations + 1:
                result_cols[f"{col}_signal"] = np.zeros(len(prices))
                result_cols[f"{col}_weight"] = np.zeros(len(prices))
                continue

            log_ret = np.diff(np.log(p))

            # Retrieve stability from fit, or compute on-the-fly
            stability = self._vr_stability.get(col, 0.5)

            # If the GS test flagged nonlinear dependence, boost weight
            # slightly (the market is more predictable than linear VR alone
            # suggests).
            if self._gs_nonlinear.get(col, False):
                stability = min(1.0, stability * 1.25)

            sig, wt = self._compute_asset_signal(log_ret, p, stability)

            # Smooth the raw signal via EMA to reduce whipsaws
            sig_series = pd.Series(sig)
            sig_smooth = sig_series.ewm(span=cfg.smoothing_span, adjust=False).mean()
            # Re-discretise after smoothing
            sig_discrete = np.where(
                sig_smooth > 0.3, 1.0,
                np.where(sig_smooth < -0.3, -1.0, 0.0),
            )

            # Smooth weights similarly
            wt_series = pd.Series(wt)
            wt_smooth = wt_series.ewm(span=cfg.smoothing_span, adjust=False).mean().values
            wt_smooth = np.clip(wt_smooth, 0.0, cfg.max_weight)

            result_cols[f"{col}_signal"] = sig_discrete
            result_cols[f"{col}_weight"] = wt_smooth

        return pd.DataFrame(result_cols, index=prices.index)

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def variance_ratio_report(
        self,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate a diagnostic report of variance ratio statistics.

        Returns a DataFrame with one row per (asset, horizon) showing:
        VR(q), z-statistic, p-value, and the implied regime.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.

        Returns
        -------
        pd.DataFrame
            Diagnostic summary.
        """
        cfg = self.config
        rows = []

        for col in prices.columns:
            p = prices[col].dropna().values.astype(np.float64)
            if len(p) < cfg.min_observations + 1:
                continue

            p = p[p > 0]
            if len(p) < cfg.min_observations + 1:
                continue

            log_ret = np.diff(np.log(p))

            for q in cfg.vr_horizons:
                vr, z, pval = _variance_ratio(log_ret, q)

                if np.isfinite(z):
                    if z > cfg.z_critical:
                        regime = "momentum"
                    elif z < -cfg.z_critical:
                        regime = "mean-reversion"
                    else:
                        regime = "efficient"
                else:
                    regime = "insufficient data"

                rows.append({
                    "asset": col,
                    "q": q,
                    "VR(q)": vr,
                    "z_stat": z,
                    "p_value": pval,
                    "regime": regime,
                })

            # Also add the automatic VR
            q_star, vr_star, z_star = _automatic_variance_ratio(log_ret)
            pval_star = 2.0 * (1.0 - norm.cdf(np.abs(z_star))) if np.isfinite(z_star) else np.nan
            rows.append({
                "asset": col,
                "q": f"{q_star} (auto)",
                "VR(q)": vr_star,
                "z_stat": z_star,
                "p_value": pval_star,
                "regime": (
                    "momentum" if z_star > cfg.z_critical
                    else "mean-reversion" if z_star < -cfg.z_critical
                    else "efficient"
                ) if np.isfinite(z_star) else "insufficient data",
            })

        return pd.DataFrame(rows)
