"""Proper scoring rules strategy for probabilistic forecast evaluation.

Uses proper scoring rules from decision theory to evaluate probabilistic
forecasts of asset returns, then converts forecast quality and calibration
diagnostics into trading signals.

Mathematical foundation
-----------------------
A scoring rule S(P, x) evaluates a probabilistic forecast P against a
realised outcome x.  A scoring rule is **proper** if the expected score
is uniquely maximised when P equals the true data-generating distribution:

    E_Q[S(P, X)] <= E_Q[S(Q, X)]   for all P, with equality iff P = Q.

Key scoring rules implemented here:

1. **Log score** (Shannon, 1948):
       S_log(P, x) = log p(x)
   where p is the density of P.  Uniquely proper; corresponds to
   maximising expected log-likelihood (information-theoretic optimality).

2. **Continuous Ranked Probability Score** (Matheson & Winkler, 1976):
       CRPS(F, x) = integral_R (F(y) - 1{y >= x})^2 dy
   For a Gaussian forecast N(mu, sigma^2):
       CRPS = sigma [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]
   where z = (x - mu) / sigma.  Decomposition:
       CRPS = Reliability - Resolution + Uncertainty
   Resolution measures the conditional variance vs marginal variance.

3. **Brier score** (Brier, 1950):
       S_brier(p, x) = -(p - x)^2
   for binary events (e.g., positive vs negative returns).

Strategy logic
--------------
1. **Forecast calibration tracking**: EWMA Gaussian forecasts with
   time-varying mu (mean) and sigma (volatility); scored by CRPS and
   log score against realised returns.

2. **Forecast quality as signal**: Good scores indicate a predictable
   market (trade aggressively); poor scores indicate surprise (reduce
   exposure).

3. **PIT (Probability Integral Transform) analysis**: If the forecast
   CDF F_hat is calibrated, u_t = F_hat(r_t) should be Uniform(0,1).
   Non-uniformity (tested via Kolmogorov-Smirnov) reveals systematic
   bias.  PIT clustered near 0 => forecasts too optimistic => bearish;
   PIT clustered near 1 => too pessimistic => bullish.

4. **CRPS resolution component**: High resolution (conditional
   distributions differ meaningfully from the marginal) indicates
   predictability; track it and trade more aggressively when high.

References
----------
*   Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. JASA 102(477), 359-378.
*   Matheson, J. E. & Winkler, R. L. (1976). Scoring rules for continuous
    probability distributions. Management Science 22(10), 1087-1096.
*   Brier, G. W. (1950). Verification of forecasts expressed in terms
    of probability. Monthly Weather Review 78(1), 1-3.
*   Dawid, A. P. (1984). Statistical theory: The prequential approach.
    JRSS-A 147(2), 278-292.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, kstest

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring rule computations (pure numpy/scipy)
# ---------------------------------------------------------------------------

def _gaussian_log_score(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Log score for Gaussian forecast: S_log = log p(x | mu, sigma).

    Parameters
    ----------
    x : observed values
    mu : forecast means
    sigma : forecast standard deviations (must be > 0)

    Returns
    -------
    Log-density values.  More negative = worse forecast.
    """
    sigma_safe = np.maximum(sigma, 1e-15)
    return norm.logpdf(x, loc=mu, scale=sigma_safe)


def _gaussian_crps(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for Gaussian forecast N(mu, sigma^2).

    CRPS(F, x) = sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]

    where z = (x - mu) / sigma, Phi is the standard normal CDF, and
    phi is the standard normal PDF.

    Lower CRPS = better forecast (it is a negatively oriented score in
    our sign convention: we negate so that higher = better).

    Parameters
    ----------
    x : observed values
    mu : forecast means
    sigma : forecast standard deviations

    Returns
    -------
    CRPS values (non-negative; lower is better).
    """
    sigma_safe = np.maximum(sigma, 1e-15)
    z = (x - mu) / sigma_safe
    crps = sigma_safe * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return crps


def _brier_score(p: np.ndarray, x_binary: np.ndarray) -> np.ndarray:
    """Brier score: S_brier = -(p - x)^2.

    Parameters
    ----------
    p : predicted probability of x = 1 (positive return)
    x_binary : binary outcomes (1 if positive return, 0 otherwise)

    Returns
    -------
    Brier scores (higher = better, in [-1, 0]).
    """
    return -(p - x_binary) ** 2


def _pit_values(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Probability Integral Transform: u_t = F_hat(r_t).

    If the forecast CDF F_hat is correctly calibrated, the PIT values
    u_t should be i.i.d. Uniform(0, 1).

    Parameters
    ----------
    x : observed values
    mu : forecast means
    sigma : forecast standard deviations

    Returns
    -------
    PIT values in [0, 1].
    """
    sigma_safe = np.maximum(sigma, 1e-15)
    return norm.cdf(x, loc=mu, scale=sigma_safe)


def _ks_uniformity_test(pit_values: np.ndarray) -> float:
    """Kolmogorov-Smirnov test for uniformity of PIT values.

    Returns the KS statistic (distance from uniform CDF).
    Higher values indicate worse calibration.

    Parameters
    ----------
    pit_values : array of PIT values to test

    Returns
    -------
    KS statistic in [0, 1].
    """
    valid = pit_values[~np.isnan(pit_values)]
    if len(valid) < 10:
        return 0.5  # insufficient data, return neutral
    # Clip to (0, 1) to avoid edge issues with the CDF
    valid = np.clip(valid, 1e-10, 1.0 - 1e-10)
    stat, _ = kstest(valid, "uniform")
    return float(stat)


def _crps_decomposition(
    crps_values: np.ndarray,
    forecast_sigma: np.ndarray,
    marginal_sigma: float,
) -> Dict[str, float]:
    """Approximate CRPS decomposition into reliability, resolution, uncertainty.

    The CRPS can be decomposed as:
        CRPS = Reliability - Resolution + Uncertainty

    We approximate:
    - Uncertainty: CRPS of the marginal (climatological) distribution
    - Resolution: how much the conditional forecast sigma varies from marginal
    - Reliability: residual (total CRPS - resolution + uncertainty)

    Parameters
    ----------
    crps_values : array of CRPS scores
    forecast_sigma : array of forecast standard deviations
    marginal_sigma : unconditional standard deviation

    Returns
    -------
    Dictionary with keys 'reliability', 'resolution', 'uncertainty'.
    """
    # Uncertainty: CRPS of marginal Gaussian N(0, marginal_sigma^2)
    # For a Gaussian with zero mean, CRPS = sigma / sqrt(pi)
    uncertainty = marginal_sigma / np.sqrt(np.pi)

    # Resolution: related to the variance of forecast sigma relative to marginal
    # High resolution means forecast sigma varies substantially
    sigma_variance = np.var(forecast_sigma)
    marginal_var = marginal_sigma ** 2
    resolution = sigma_variance / (marginal_var + 1e-15)

    # Reliability: residual
    mean_crps = np.mean(crps_values)
    reliability = mean_crps + resolution - uncertainty

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
    }


# ---------------------------------------------------------------------------
# EWMA forecast model
# ---------------------------------------------------------------------------

def _ewma_forecast(
    returns: np.ndarray,
    mu_span: int,
    sigma_span: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate EWMA Gaussian forecasts with time-varying mu and sigma.

    Uses exponentially weighted moving average for the conditional mean
    and exponentially weighted moving variance for the conditional
    standard deviation.

    Parameters
    ----------
    returns : (T,) array of log returns
    mu_span : EMA span for the conditional mean
    sigma_span : EMA span for the conditional variance

    Returns
    -------
    mu_forecast : (T,) one-step-ahead forecast means
    sigma_forecast : (T,) one-step-ahead forecast standard deviations
    """
    T = len(returns)
    mu_forecast = np.full(T, np.nan)
    sigma_forecast = np.full(T, np.nan)

    alpha_mu = 2.0 / (mu_span + 1.0)
    alpha_sigma = 2.0 / (sigma_span + 1.0)

    # Initialise with first observation
    ewma_mu = returns[0]
    ewma_var = 0.0  # will be updated from second observation

    for t in range(1, T):
        # The forecast for time t is the EWMA computed from data up to t-1
        mu_forecast[t] = ewma_mu
        sigma_forecast[t] = np.sqrt(max(ewma_var, 1e-15))

        # Update EWMA statistics with observation at time t
        ewma_mu = alpha_mu * returns[t] + (1.0 - alpha_mu) * ewma_mu
        deviation_sq = (returns[t] - ewma_mu) ** 2
        ewma_var = alpha_sigma * deviation_sq + (1.0 - alpha_sigma) * ewma_var

    return mu_forecast, sigma_forecast


# ===========================================================================
# Strategy class
# ===========================================================================

class ScoringRulesStrategy(Strategy):
    """Proper scoring rules strategy for probabilistic forecasting.

    Generates EWMA-Gaussian probabilistic forecasts of returns, evaluates
    them with proper scoring rules (CRPS, log score, Brier), and converts
    forecast quality and calibration diagnostics into trading signals.

    The signal combines four components:

    1. **Score quality signal**: EWMA of forecast scores relative to a
       baseline (unconditional Gaussian).  Good scores => predictable
       market => higher position weight.

    2. **Score improvement signal**: Rate of change in cumulative score.
       Improving scores => regime becoming predictable => increase.

    3. **PIT bias signal**: Directional signal from PIT non-uniformity.
       PIT < 0.5 on average => forecasts too optimistic => bearish.
       PIT > 0.5 on average => forecasts too pessimistic => bullish.

    4. **Resolution signal**: CRPS resolution component tracking.
       High resolution => conditional forecasts differ from marginal
       => the market is more predictable => trade more aggressively.

    Parameters
    ----------
    mu_span : int
        EWMA span for conditional mean estimation.  Default 20.
    sigma_span : int
        EWMA span for conditional variance estimation.  Default 60.
    score_span : int
        EWMA span for smoothing cumulative scores.  Default 63.
    pit_window : int
        Rolling window for PIT uniformity testing.  Default 126.
    resolution_window : int
        Rolling window for CRPS resolution estimation.  Default 252.
    min_history : int
        Minimum observations before generating non-trivial signals.
        Default 63.
    score_quality_weight : float
        Weight of the score quality component in signal blending.
        Default 0.3.
    score_improvement_weight : float
        Weight of the score improvement component.  Default 0.2.
    pit_bias_weight : float
        Weight of the PIT bias directional component.  Default 0.3.
    resolution_weight : float
        Weight of the resolution component for position sizing.
        Default 0.2.
    max_weight : float
        Maximum position weight (before risk limits).  Default 1.0.
    """

    def __init__(
        self,
        mu_span: int = 20,
        sigma_span: int = 60,
        score_span: int = 63,
        pit_window: int = 126,
        resolution_window: int = 252,
        min_history: int = 63,
        score_quality_weight: float = 0.3,
        score_improvement_weight: float = 0.2,
        pit_bias_weight: float = 0.3,
        resolution_weight: float = 0.2,
        max_weight: float = 1.0,
    ) -> None:
        super().__init__(
            name="ScoringRules",
            description=(
                "Proper scoring rules strategy: evaluates probabilistic "
                "forecasts via CRPS, log score, and PIT analysis, converting "
                "forecast quality and calibration into trading signals."
            ),
        )
        self.mu_span = mu_span
        self.sigma_span = sigma_span
        self.score_span = score_span
        self.pit_window = pit_window
        self.resolution_window = resolution_window
        self.min_history = min_history
        self.score_quality_weight = score_quality_weight
        self.score_improvement_weight = score_improvement_weight
        self.pit_bias_weight = pit_bias_weight
        self.resolution_weight = resolution_weight
        self.max_weight = max_weight

        # Populated by fit()
        self._marginal_mu: Optional[float] = None
        self._marginal_sigma: Optional[float] = None

        # Diagnostics populated by generate_signals()
        self._crps_series: Optional[Dict[str, pd.Series]] = None
        self._log_score_series: Optional[Dict[str, pd.Series]] = None
        self._pit_series: Optional[Dict[str, pd.Series]] = None
        self._ks_stats: Optional[Dict[str, pd.Series]] = None
        self._crps_decomposition: Optional[Dict[str, Dict[str, float]]] = None

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "ScoringRulesStrategy":
        """Calibrate marginal distribution parameters from training data.

        Computes the unconditional mean and standard deviation of log
        returns, which serve as the baseline (climatological) forecast
        for score comparisons.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (DatetimeIndex, columns = tickers).

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        returns = np.log(prices / prices.shift(1)).dropna()

        # Marginal (unconditional) distribution statistics across all assets
        all_returns = returns.values.flatten()
        all_returns = all_returns[~np.isnan(all_returns)]
        self._marginal_mu = float(np.mean(all_returns))
        self._marginal_sigma = float(np.std(all_returns))

        self.parameters = {
            "mu_span": self.mu_span,
            "sigma_span": self.sigma_span,
            "score_span": self.score_span,
            "pit_window": self.pit_window,
            "resolution_window": self.resolution_window,
            "marginal_mu": self._marginal_mu,
            "marginal_sigma": self._marginal_sigma,
        }

        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals from proper scoring rule analysis.

        For each asset:
        1. Produce EWMA-Gaussian one-step-ahead forecasts.
        2. Score each forecast against the realised return using CRPS,
           log score, and Brier score.
        3. Compute PIT values and test uniformity via rolling KS test.
        4. Estimate CRPS resolution in a rolling window.
        5. Blend the four signal components into direction and weight.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (DatetimeIndex, columns = tickers).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` (+1/-1/0) and ``{ticker}_weight``
            for each asset.
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        returns = np.log(prices / prices.shift(1))
        tickers = list(prices.columns)
        n_dates = len(prices)

        signals_df = pd.DataFrame(index=prices.index)

        # Diagnostic storage
        self._crps_series = {}
        self._log_score_series = {}
        self._pit_series = {}
        self._ks_stats = {}
        self._crps_decomposition = {}

        for ticker in tickers:
            ret = returns[ticker].values.astype(np.float64)

            # ----------------------------------------------------------
            # Step 1: Generate EWMA-Gaussian forecasts
            # ----------------------------------------------------------
            mu_fc, sigma_fc = _ewma_forecast(ret, self.mu_span, self.sigma_span)

            # ----------------------------------------------------------
            # Step 2: Score forecasts against realised returns
            # ----------------------------------------------------------
            # Mask: valid where we have both a forecast and a return
            valid = ~np.isnan(mu_fc) & ~np.isnan(sigma_fc) & ~np.isnan(ret)

            crps_vals = np.full(n_dates, np.nan)
            log_score_vals = np.full(n_dates, np.nan)
            brier_vals = np.full(n_dates, np.nan)
            pit_vals = np.full(n_dates, np.nan)

            crps_vals[valid] = _gaussian_crps(
                ret[valid], mu_fc[valid], sigma_fc[valid]
            )
            log_score_vals[valid] = _gaussian_log_score(
                ret[valid], mu_fc[valid], sigma_fc[valid]
            )

            # Brier score: probability of positive return
            prob_positive = np.full(n_dates, np.nan)
            prob_positive[valid] = 1.0 - norm.cdf(
                0.0, loc=mu_fc[valid], scale=np.maximum(sigma_fc[valid], 1e-15)
            )
            binary_outcome = np.full(n_dates, np.nan)
            binary_outcome[valid] = (ret[valid] > 0).astype(float)
            brier_vals[valid] = _brier_score(
                prob_positive[valid], binary_outcome[valid]
            )

            # PIT values
            pit_vals[valid] = _pit_values(
                ret[valid], mu_fc[valid], sigma_fc[valid]
            )

            # Store diagnostics
            self._crps_series[ticker] = pd.Series(
                crps_vals, index=prices.index, name=f"{ticker}_crps"
            )
            self._log_score_series[ticker] = pd.Series(
                log_score_vals, index=prices.index, name=f"{ticker}_log_score"
            )
            self._pit_series[ticker] = pd.Series(
                pit_vals, index=prices.index, name=f"{ticker}_pit"
            )

            # ----------------------------------------------------------
            # Step 3: Compute signal components
            # ----------------------------------------------------------

            # --- Component 1: Score quality signal ---
            # Compare CRPS to the marginal (climatological) baseline.
            # Baseline CRPS for N(marginal_mu, marginal_sigma^2):
            baseline_crps = np.full(n_dates, np.nan)
            baseline_crps[valid] = _gaussian_crps(
                ret[valid],
                np.full(int(valid.sum()), self._marginal_mu),
                np.full(int(valid.sum()), max(self._marginal_sigma, 1e-15)),
            )

            # Skill score: 1 - CRPS / CRPS_ref  (higher = better)
            # Positive skill => our forecast beats climatology
            crps_skill = np.full(n_dates, np.nan)
            denom = np.abs(baseline_crps[valid]) + 1e-15
            crps_skill[valid] = 1.0 - crps_vals[valid] / denom

            # EWMA smooth the skill score
            skill_series = pd.Series(crps_skill, index=prices.index)
            skill_smooth = skill_series.ewm(
                span=self.score_span, min_periods=self.min_history
            ).mean()

            # Map skill to [0, 1] for position sizing
            # Positive skill => more aggressive, negative => less
            score_quality = np.clip(skill_smooth.values, -1.0, 1.0)
            # Rescale from [-1, 1] to [0, 1]
            score_quality_weight = (score_quality + 1.0) / 2.0

            # --- Component 2: Score improvement signal ---
            # Rate of change in cumulative log score
            cum_log_score = pd.Series(log_score_vals, index=prices.index)
            cum_log_score = cum_log_score.cumsum()
            # Smoothed rate of change (first difference of EWMA)
            cum_smooth = cum_log_score.ewm(
                span=self.score_span, min_periods=self.min_history
            ).mean()
            score_roc = cum_smooth.diff()
            # Normalise to [-1, 1] via tanh
            score_roc_norm = np.tanh(
                score_roc.values / (np.nanstd(score_roc.values) + 1e-15)
            )
            # Rescale to [0, 1]
            score_improvement = (score_roc_norm + 1.0) / 2.0

            # --- Component 3: PIT bias signal ---
            # Rolling mean of PIT values: <0.5 => forecasts too optimistic
            # (actual returns are lower than predicted) => bearish
            # >0.5 => too pessimistic => bullish
            pit_series = pd.Series(pit_vals, index=prices.index)
            pit_rolling_mean = pit_series.rolling(
                window=self.pit_window, min_periods=self.min_history
            ).mean()

            # Rolling KS test for uniformity
            ks_stats = np.full(n_dates, np.nan)
            for t in range(self.min_history, n_dates):
                start_idx = max(0, t - self.pit_window)
                window_pit = pit_vals[start_idx : t + 1]
                ks_stats[t] = _ks_uniformity_test(window_pit)

            self._ks_stats[ticker] = pd.Series(
                ks_stats, index=prices.index, name=f"{ticker}_ks_stat"
            )

            # PIT directional signal: deviation from 0.5
            # Strong deviation + high KS stat => stronger signal
            pit_deviation = pit_rolling_mean.values - 0.5
            # Scale by KS statistic (higher KS => more confident in bias)
            pit_direction = np.where(
                ~np.isnan(ks_stats),
                np.sign(pit_deviation) * np.minimum(np.abs(pit_deviation) * 2.0, 1.0),
                0.0,
            )

            # --- Component 4: Resolution signal ---
            # Rolling CRPS decomposition for position sizing
            resolution_signal = np.full(n_dates, np.nan)
            for t in range(self.min_history, n_dates):
                start_idx = max(0, t - self.resolution_window)
                window_crps = crps_vals[start_idx : t + 1]
                window_sigma = sigma_fc[start_idx : t + 1]

                # Filter NaNs
                mask = ~np.isnan(window_crps) & ~np.isnan(window_sigma)
                if mask.sum() < 20:
                    resolution_signal[t] = 0.0
                    continue

                decomp = _crps_decomposition(
                    window_crps[mask],
                    window_sigma[mask],
                    max(self._marginal_sigma, 1e-15),
                )
                resolution_signal[t] = decomp["resolution"]

            # Store final decomposition
            if n_dates > self.min_history:
                final_start = max(0, n_dates - self.resolution_window)
                final_crps = crps_vals[final_start:]
                final_sigma = sigma_fc[final_start:]
                fmask = ~np.isnan(final_crps) & ~np.isnan(final_sigma)
                if fmask.sum() >= 20:
                    self._crps_decomposition[ticker] = _crps_decomposition(
                        final_crps[fmask],
                        final_sigma[fmask],
                        max(self._marginal_sigma, 1e-15),
                    )

            # Normalise resolution to [0, 1] via empirical rank
            res_series = pd.Series(resolution_signal, index=prices.index)
            res_smooth = res_series.ewm(
                span=self.score_span, min_periods=self.min_history
            ).mean()
            # Use tanh to map to [0, 1]; higher resolution => higher weight
            res_vals = res_smooth.values
            res_std = np.nanstd(res_vals)
            resolution_weight = np.where(
                ~np.isnan(res_vals),
                (np.tanh(res_vals / (res_std + 1e-15)) + 1.0) / 2.0,
                0.5,
            )

            # ----------------------------------------------------------
            # Step 4: Combine components into signal and weight
            # ----------------------------------------------------------

            # Direction: primarily from PIT bias signal
            # PIT bias > 0 => market stronger than forecast => long
            # PIT bias < 0 => market weaker than forecast => short
            raw_direction = pit_direction

            # Also incorporate score improvement direction
            improvement_direction = np.where(
                score_roc_norm > 0.1, 1.0,
                np.where(score_roc_norm < -0.1, -1.0, 0.0)
            )

            # Blend directions
            combined_direction = (
                self.pit_bias_weight * raw_direction
                + self.score_improvement_weight * improvement_direction
            )
            # Normalise by total weight of directional components
            dir_total = self.pit_bias_weight + self.score_improvement_weight
            if dir_total > 0:
                combined_direction /= dir_total

            # Discrete signal: -1, 0, +1
            signal = np.where(
                combined_direction > 0.1, 1,
                np.where(combined_direction < -0.1, -1, 0)
            )

            # Weight: from score quality and resolution
            combined_weight = (
                self.score_quality_weight * score_quality_weight
                + self.resolution_weight * resolution_weight
            )
            # Normalise by total weight of sizing components
            weight_total = self.score_quality_weight + self.resolution_weight
            if weight_total > 0:
                combined_weight /= weight_total

            # Apply max weight cap
            combined_weight = np.clip(combined_weight, 0.0, self.max_weight)

            # Zero weight during burn-in
            combined_weight[:self.min_history] = 0.0
            signal[:self.min_history] = 0

            # Handle any remaining NaNs
            signal = np.nan_to_num(signal, nan=0.0).astype(int)
            combined_weight = np.nan_to_num(combined_weight, nan=0.0)

            # ----------------------------------------------------------
            # Step 5: Store in output DataFrame
            # ----------------------------------------------------------
            signals_df[f"{ticker}_signal"] = signal
            signals_df[f"{ticker}_weight"] = combined_weight

        # Handle single-ticker shorthand
        if len(tickers) == 1:
            t = tickers[0]
            signals_df["signal"] = signals_df[f"{t}_signal"]
            signals_df["weight"] = signals_df[f"{t}_weight"]

        return signals_df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_crps_series(self) -> Optional[Dict[str, pd.Series]]:
        """Return per-ticker CRPS time series from the last signal run.

        Lower CRPS indicates a better probabilistic forecast.
        """
        return self._crps_series

    def get_log_score_series(self) -> Optional[Dict[str, pd.Series]]:
        """Return per-ticker log score time series.

        Higher (less negative) log score indicates a better forecast.
        """
        return self._log_score_series

    def get_pit_series(self) -> Optional[Dict[str, pd.Series]]:
        """Return per-ticker PIT value time series.

        For a calibrated forecast, PIT values should be Uniform(0, 1).
        """
        return self._pit_series

    def get_ks_stats(self) -> Optional[Dict[str, pd.Series]]:
        """Return per-ticker rolling KS uniformity test statistics.

        Higher KS statistic indicates greater departure from uniformity,
        suggesting the forecast is miscalibrated.
        """
        return self._ks_stats

    def get_crps_decomposition(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Return per-ticker CRPS decomposition from the final window.

        Returns dictionary per ticker with keys:
        - 'reliability': calibration error (lower is better)
        - 'resolution': conditional variance vs marginal (higher is better)
        - 'uncertainty': irreducible uncertainty
        """
        return self._crps_decomposition
