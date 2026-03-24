"""James-Stein shrinkage estimation portfolio strategy.

Exploits Stein's paradox — the maximum-likelihood estimator of a
multivariate normal mean is *inadmissible* for dimension p >= 3 — to build
portfolios whose expected-return estimates dominate the naive sample mean
uniformly in mean-squared error.

Mathematical foundation
-----------------------
James-Stein estimator for μ ∈ R^p (p >= 3) observed via X ~ N(μ, σ²I):

    μ̂_JS = μ_target + (1 - c / ||X - μ_target||²) · (X - μ_target)

where c = (p - 2) · σ² / T is the shrinkage numerator.

The positive-part variant clamps the shrinkage factor to [0, 1]:

    μ̂_JS+ = μ_target + max(0, 1 - c / ||X - μ_target||²) · (X - μ_target)

This dominates the MLE (X itself) for *any* true μ, a result that
Charles Stein called "an embarrassment to the field" of classical
statistics.

Strategy
--------
1. **Multi-target shrinkage** — compute JS-shrunk means toward three
   structurally motivated targets (grand mean, zero, market-cap proxy)
   and combine them by inverse expected loss.

2. **Shrinkage portfolio construction** — feed JS-shrunk expected
   returns into a mean-variance optimiser with Ledoit-Wolf shrunk
   covariance.  Estimation error is attacked on *both* the mean and
   covariance sides.

3. **Dynamic shrinkage intensity** — the shrinkage factor
   c = (p-2)·σ²/T adapts naturally: more observations → less
   shrinkage; higher volatility → more shrinkage.

4. **Long/short signal** — rank assets by JS-shrunk expected return,
   go long the top quartile and short the bottom quartile, with
   position sizes proportional to rank distance from the median.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SteinShrinkageConfig:
    """Tuneable parameters for the James-Stein shrinkage strategy."""

    # Estimation windows
    lookback_window: int = 252         # rolling window for sample moments
    min_history: int = 126             # minimum observations before trading

    # Shrinkage targets to use (any subset of {"grand_mean", "zero", "mktcap"})
    shrinkage_targets: List[str] = field(
        default_factory=lambda: ["grand_mean", "zero", "mktcap"]
    )

    # Portfolio construction
    long_quantile: float = 0.75        # go long above this percentile
    short_quantile: float = 0.25       # go short below this percentile
    use_mean_variance: bool = True     # use MV optimisation (vs. rank-based)
    risk_aversion: float = 2.0         # γ in MV: w* = (1/γ) Σ^{-1} μ

    # Risk
    max_leverage: float = 1.5          # gross leverage cap
    rebalance_freq: int = 21           # trading days between rebalances

    # Annualisation
    annualisation_factor: int = 252    # trading days per year


# ---------------------------------------------------------------------------
# James-Stein shrinkage helpers
# ---------------------------------------------------------------------------

def _james_stein_shrink(
    sample_mean: np.ndarray,
    target: np.ndarray,
    sample_variance: float,
    n_obs: int,
    positive_part: bool = True,
) -> Tuple[np.ndarray, float]:
    """Apply the James-Stein (positive-part) estimator.

    Parameters
    ----------
    sample_mean : (p,) array
        Sample mean vector X̄.
    target : (p,) array
        Shrinkage target μ_target.
    sample_variance : float
        Pooled variance estimate σ² (scalar, as in the spherical model).
    n_obs : int
        Number of observations T used to compute sample_mean.
    positive_part : bool
        If True, clamp the shrinkage multiplier to [0, 1] (positive-part JS).

    Returns
    -------
    mu_js : (p,) array
        James-Stein shrunk estimate.
    shrinkage_intensity : float
        The realised shrinkage factor in [0, 1] (1 = fully shrunk to target).
    """
    p = len(sample_mean)
    if p < 3:
        warnings.warn(
            "James-Stein shrinkage requires p >= 3 to dominate the MLE. "
            f"Got p = {p}; returning the sample mean unchanged.",
            stacklevel=2,
        )
        return sample_mean.copy(), 0.0

    delta = sample_mean - target
    delta_norm_sq = np.dot(delta, delta)

    # c = (p - 2) * σ² / T
    c = (p - 2) * sample_variance / n_obs

    if delta_norm_sq < 1e-15:
        # Sample mean is essentially at the target — full shrinkage
        return target.copy(), 1.0

    shrinkage_factor = c / delta_norm_sq  # this is the amount to shrink *by*

    if positive_part:
        shrinkage_factor = min(shrinkage_factor, 1.0)
        shrinkage_factor = max(shrinkage_factor, 0.0)

    mu_js = target + (1.0 - shrinkage_factor) * delta
    return mu_js, float(shrinkage_factor)


def _compute_shrinkage_targets(
    sample_mean: np.ndarray,
    n_assets: int,
    market_cap_weights: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Construct the canonical shrinkage target vectors.

    Parameters
    ----------
    sample_mean : (p,) array
        Sample mean returns.
    n_assets : int
        Number of assets p.
    market_cap_weights : (p,) array or None
        If provided, used to construct a market-cap-weighted return target.

    Returns
    -------
    dict
        Mapping from target name to target vector.
    """
    targets: Dict[str, np.ndarray] = {}

    # Target 1: Grand mean — shrink all means toward their cross-sectional average
    grand_mean_val = np.mean(sample_mean)
    targets["grand_mean"] = np.full(n_assets, grand_mean_val)

    # Target 2: Zero — shrink toward no expected return (efficient-market prior)
    targets["zero"] = np.zeros(n_assets)

    # Target 3: Market-cap weighted returns — shrink toward the
    # market-cap-weighted average return (CAPM-like prior)
    if market_cap_weights is not None:
        mktcap_return = np.dot(market_cap_weights, sample_mean)
        targets["mktcap"] = np.full(n_assets, mktcap_return)
    else:
        # Fallback: equal-weighted grand mean (same as target 1)
        targets["mktcap"] = targets["grand_mean"].copy()

    return targets


def _combine_shrinkage_estimates(
    estimates: List[np.ndarray],
    intensities: List[float],
    sample_mean: np.ndarray,
    sample_variance: float,
    n_obs: int,
) -> np.ndarray:
    """Combine multiple JS-shrunk estimates by inverse expected loss.

    Each JS estimator has an approximate risk (expected MSE) of:
        R(μ̂_JS) ≈ p·σ²/T - c²/||X - target||²

    We weight each estimate inversely proportional to its estimated risk.

    Parameters
    ----------
    estimates : list of (p,) arrays
        JS-shrunk mean estimates from different targets.
    intensities : list of float
        Shrinkage intensities for each estimate.
    sample_mean : (p,) array
        Original sample mean.
    sample_variance : float
        Pooled variance.
    n_obs : int
        Number of observations.

    Returns
    -------
    combined : (p,) array
        Optimally combined JS estimate.
    """
    if len(estimates) == 0:
        return sample_mean.copy()
    if len(estimates) == 1:
        return estimates[0]

    p = len(sample_mean)

    # Baseline risk of the MLE
    mle_risk = p * sample_variance / n_obs

    weights = []
    for est, intensity in zip(estimates, intensities):
        # Approximate risk reduction from shrinkage
        # Risk ≈ mle_risk * (1 - intensity) for the positive-part estimator
        risk = mle_risk * max(1.0 - intensity, 1e-8)
        # Inverse risk weighting
        weights.append(1.0 / max(risk, 1e-15))

    weights_arr = np.array(weights)
    weights_arr /= weights_arr.sum()

    combined = np.zeros(p)
    for est, w in zip(estimates, weights_arr):
        combined += w * est

    return combined


# ---------------------------------------------------------------------------
# Mean-variance optimisation
# ---------------------------------------------------------------------------

def _mean_variance_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 2.0,
) -> np.ndarray:
    """Unconstrained mean-variance optimal weights.

        w* = (1/γ) Σ^{-1} μ

    Uses pseudo-inverse for numerical stability.

    Parameters
    ----------
    mu : (p,) array
        Expected return vector.
    cov : (p, p) array
        Covariance matrix.
    risk_aversion : float
        Risk-aversion parameter γ.

    Returns
    -------
    weights : (p,) array
        Raw MV-optimal weights (not yet normalised).
    """
    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return np.full(len(mu), 1.0 / len(mu))

    w = (1.0 / risk_aversion) * cov_inv @ mu
    return w


# ===========================================================================
# Strategy class
# ===========================================================================

class SteinShrinkageStrategy(Strategy):
    """Portfolio strategy based on James-Stein shrinkage estimation.

    Applies the James-Stein positive-part estimator to expected returns,
    shrinking toward multiple structurally motivated targets to dominate
    the naive sample mean in MSE.  Combines shrunk returns with a
    Ledoit-Wolf shrunk covariance in a mean-variance framework.

    Parameters
    ----------
    config : SteinShrinkageConfig, optional
        Strategy configuration.  Uses defaults if not supplied.
    """

    def __init__(self, config: Optional[SteinShrinkageConfig] = None) -> None:
        self.cfg = config or SteinShrinkageConfig()

        super().__init__(
            name="SteinShrinkage",
            description=(
                "James-Stein shrinkage estimation strategy: exploits the "
                "inadmissibility of the MLE for p >= 3 to construct "
                "portfolios with estimation-error-robust expected returns."
            ),
        )

        # State populated during fit / generate_signals
        self._asset_names: Optional[pd.Index] = None
        self._shrunk_mu: Optional[np.ndarray] = None
        self._shrunk_cov: Optional[np.ndarray] = None
        self._shrinkage_intensities: Dict[str, float] = {}
        self._combined_intensity: float = 0.0
        self._pooled_variance: float = 0.0
        self._portfolio_weights: Optional[np.ndarray] = None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _estimate_pooled_variance(self, returns: pd.DataFrame) -> float:
        """Compute the pooled (cross-sectional average) variance of returns.

        This is the σ² in the James-Stein formula, corresponding to the
        spherical covariance model X ~ N(μ, σ²I).

        Parameters
        ----------
        returns : pd.DataFrame
            Period returns, shape (T, p).

        Returns
        -------
        float
            Pooled variance estimate.
        """
        # Average variance across assets
        variances = returns.var(axis=0).values
        pooled = np.mean(variances)
        return max(float(pooled), 1e-15)

    def _estimate_market_cap_weights(
        self, prices: pd.DataFrame
    ) -> np.ndarray:
        """Proxy market-cap weights from price levels.

        In the absence of actual market-cap data, we use the last
        observed price as a rough proxy (higher-priced stocks tend to
        have larger market caps, though this is imperfect).

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.

        Returns
        -------
        weights : (p,) array
            Normalised proxy weights summing to 1.
        """
        last_prices = prices.iloc[-1].values.astype(float)
        last_prices = np.nan_to_num(last_prices, nan=0.0)
        last_prices = np.maximum(last_prices, 0.0)
        total = last_prices.sum()
        if total < 1e-15:
            return np.full(len(last_prices), 1.0 / len(last_prices))
        return last_prices / total

    def _compute_shrunk_moments(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Compute JS-shrunk expected returns and LW-shrunk covariance.

        This is the core estimation engine that attacks estimation error
        on *both* the mean vector (James-Stein) and the covariance
        matrix (Ledoit-Wolf).

        Parameters
        ----------
        returns : pd.DataFrame
            Period returns, shape (T, p).
        prices : pd.DataFrame
            Price data for market-cap proxy.

        Returns
        -------
        mu_js : (p,) array
            Combined James-Stein shrunk expected return vector.
        cov_lw : (p, p) array
            Ledoit-Wolf shrunk covariance matrix.
        intensities : dict
            Shrinkage intensity per target.
        """
        n_obs, p = returns.shape
        sample_mean = returns.mean(axis=0).values
        pooled_var = self._estimate_pooled_variance(returns)

        # Market-cap proxy weights for the mktcap target
        mktcap_weights = self._estimate_market_cap_weights(prices)

        # Build shrinkage targets
        all_targets = _compute_shrinkage_targets(
            sample_mean, p, market_cap_weights=mktcap_weights
        )

        # Apply JS shrinkage to each configured target
        estimates: List[np.ndarray] = []
        intensities_list: List[float] = []
        intensities_dict: Dict[str, float] = {}

        for target_name in self.cfg.shrinkage_targets:
            if target_name not in all_targets:
                logger.warning(
                    "Unknown shrinkage target '%s'; skipping.", target_name
                )
                continue

            target_vec = all_targets[target_name]
            mu_js, intensity = _james_stein_shrink(
                sample_mean, target_vec, pooled_var, n_obs,
                positive_part=True,
            )
            estimates.append(mu_js)
            intensities_list.append(intensity)
            intensities_dict[target_name] = intensity

            logger.debug(
                "JS shrinkage toward '%s': intensity=%.4f",
                target_name, intensity,
            )

        # Combine estimates by inverse expected loss
        mu_combined = _combine_shrinkage_estimates(
            estimates, intensities_list, sample_mean, pooled_var, n_obs
        )

        # Ledoit-Wolf covariance shrinkage
        try:
            lw = LedoitWolf().fit(returns.values)
            cov_lw = lw.covariance_
        except Exception as exc:
            logger.warning(
                "Ledoit-Wolf estimation failed (%s); falling back to "
                "sample covariance.", exc,
            )
            cov_lw = returns.cov().values

        return mu_combined, cov_lw, intensities_dict

    def _rank_based_weights(
        self, mu: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate long/short signals and weights from ranked JS returns.

        Ranks assets by JS-shrunk expected return, goes long the top
        quartile, short the bottom quartile, and sizes positions by
        normalised rank distance from the median.

        Parameters
        ----------
        mu : (p,) array
            JS-shrunk expected return vector.

        Returns
        -------
        signals : (p,) array
            Signal in {-1, 0, +1}.
        weights : (p,) array
            Position-sizing weights in [0, 1].
        """
        p = len(mu)
        ranks = np.argsort(np.argsort(mu)).astype(float)  # 0-indexed ranks
        # Normalise ranks to [0, 1]
        normalised_ranks = ranks / max(p - 1, 1)

        signals = np.zeros(p)
        weights = np.zeros(p)

        long_mask = normalised_ranks >= self.cfg.long_quantile
        short_mask = normalised_ranks <= self.cfg.short_quantile

        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        # Weight by distance from median rank (0.5)
        distances = np.abs(normalised_ranks - 0.5)
        # Only weight active positions
        active_mask = long_mask | short_mask
        if active_mask.any():
            active_distances = distances[active_mask]
            total = active_distances.sum()
            if total > 1e-15:
                weights[active_mask] = distances[active_mask] / total
            else:
                n_active = active_mask.sum()
                weights[active_mask] = 1.0 / n_active

        return signals, weights

    def _mv_based_weights(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate long/short signals and weights from MV optimisation.

        Computes unconstrained MV-optimal weights w* = (1/γ)Σ^{-1}μ,
        then normalises and caps leverage.

        Parameters
        ----------
        mu : (p,) array
            JS-shrunk expected return vector.
        cov : (p, p) array
            LW-shrunk covariance matrix.

        Returns
        -------
        signals : (p,) array
            Signal in {-1, 0, +1}.
        weights : (p,) array
            Position-sizing weights in [0, 1].
        """
        raw_weights = _mean_variance_weights(mu, cov, self.cfg.risk_aversion)

        # Cap leverage
        gross = np.abs(raw_weights).sum()
        if gross > self.cfg.max_leverage:
            raw_weights *= self.cfg.max_leverage / gross

        signals = np.sign(raw_weights)

        # Weights as fraction of gross exposure
        abs_weights = np.abs(raw_weights)
        total = abs_weights.sum()
        if total > 1e-15:
            weights = abs_weights / total
        else:
            weights = np.full(len(mu), 1.0 / len(mu))

        return signals, weights

    # -----------------------------------------------------------------
    # Strategy interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "SteinShrinkageStrategy":
        """Calibrate the James-Stein shrinkage model on historical prices.

        Computes sample moments from log-returns, applies multi-target
        JS shrinkage to the mean vector, and estimates a Ledoit-Wolf
        shrunk covariance matrix.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data.  Columns are asset tickers; index is
            a DatetimeIndex.

        Returns
        -------
        self
        """
        self.validate_prices(prices)
        self._asset_names = prices.columns

        n_obs = min(len(prices), self.cfg.lookback_window)
        recent_prices = prices.iloc[-n_obs:]

        # Log returns
        log_returns = np.log(recent_prices / recent_prices.shift(1)).dropna()
        n_assets = log_returns.shape[1]
        n_periods = log_returns.shape[0]

        if n_periods < self.cfg.min_history:
            warnings.warn(
                f"Insufficient history for JS shrinkage: {n_periods} "
                f"observations for {n_assets} assets "
                f"(need >= {self.cfg.min_history}).",
                stacklevel=2,
            )
            self._fitted = True
            return self

        if n_assets < 3:
            warnings.warn(
                "James-Stein shrinkage requires p >= 3 to dominate the MLE. "
                f"Got {n_assets} assets; shrinkage will not improve on MLE.",
                stacklevel=2,
            )

        # Core computation: shrunk moments
        mu_js, cov_lw, intensities = self._compute_shrunk_moments(
            log_returns, recent_prices,
        )

        self._shrunk_mu = mu_js
        self._shrunk_cov = cov_lw
        self._shrinkage_intensities = intensities
        self._pooled_variance = self._estimate_pooled_variance(log_returns)

        # Compute combined intensity (weighted average)
        if intensities:
            self._combined_intensity = np.mean(list(intensities.values()))
        else:
            self._combined_intensity = 0.0

        # Compute portfolio weights
        if self.cfg.use_mean_variance:
            signals, weights = self._mv_based_weights(mu_js, cov_lw)
        else:
            signals, weights = self._rank_based_weights(mu_js)

        self._portfolio_weights = weights * signals  # signed weights

        # Store parameters for inspection
        self.parameters = {
            "shrinkage_intensities": dict(intensities),
            "combined_intensity": self._combined_intensity,
            "pooled_variance": self._pooled_variance,
            "n_assets": n_assets,
            "n_observations": n_periods,
            "js_numerator_c": (n_assets - 2) * self._pooled_variance / n_periods,
        }

        logger.info(
            "Stein shrinkage fit complete: p=%d, T=%d, "
            "intensities=%s, pooled_var=%.6f",
            n_assets, n_periods,
            {k: f"{v:.4f}" for k, v in intensities.items()},
            self._pooled_variance,
        )

        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate per-asset trading signals using JS-shrunk expected returns.

        On each rebalance date, re-estimates the JS-shrunk mean vector
        and Ledoit-Wolf covariance, then either ranks stocks by shrunk
        return (quartile long/short) or runs MV optimisation.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (columns = tickers, index = DatetimeIndex).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for
            each asset.
        """
        self.ensure_fitted()

        asset_names = prices.columns
        n_assets = len(asset_names)
        n_rows = len(prices)

        # Prepare output DataFrame
        output_cols = []
        for name in asset_names:
            output_cols.extend([f"{name}_signal", f"{name}_weight"])
        signals_df = pd.DataFrame(0.0, index=prices.index, columns=output_cols)

        # Log returns
        log_returns = np.log(prices / prices.shift(1))

        # Rolling signal generation with periodic rebalancing
        min_lookback = max(self.cfg.min_history, self.cfg.lookback_window)
        last_rebalance = -self.cfg.rebalance_freq  # force on first eligible date

        cached_signals = np.zeros(n_assets)
        cached_weights = np.zeros(n_assets)

        for t in range(min_lookback, n_rows):
            if (t - last_rebalance) >= self.cfg.rebalance_freq:
                # Extract rolling window
                start = max(0, t - self.cfg.lookback_window)
                window_returns = log_returns.iloc[start:t].dropna()
                window_prices = prices.iloc[start:t]

                if len(window_returns) < self.cfg.min_history:
                    continue

                n_window_assets = window_returns.shape[1]

                if n_window_assets < 3:
                    # Cannot apply JS shrinkage; fall back to equal weight
                    cached_signals = np.ones(n_assets)
                    cached_weights = np.full(n_assets, 1.0 / n_assets)
                    last_rebalance = t
                    continue

                # Re-estimate shrunk moments
                try:
                    mu_js, cov_lw, intensities = self._compute_shrunk_moments(
                        window_returns, window_prices,
                    )
                except Exception as exc:
                    logger.warning(
                        "Shrinkage estimation failed at t=%d (%s); "
                        "keeping previous signals.", t, exc,
                    )
                    continue

                # Generate signals and weights
                if self.cfg.use_mean_variance:
                    sigs, wts = self._mv_based_weights(mu_js, cov_lw)
                else:
                    sigs, wts = self._rank_based_weights(mu_js)

                cached_signals = sigs
                cached_weights = wts
                last_rebalance = t

                self._shrinkage_intensities = intensities

            # Write signals for this date
            for j, name in enumerate(asset_names):
                signals_df.iloc[
                    t, signals_df.columns.get_loc(f"{name}_signal")
                ] = cached_signals[j]
                signals_df.iloc[
                    t, signals_df.columns.get_loc(f"{name}_weight")
                ] = cached_weights[j]

        return signals_df

    # -----------------------------------------------------------------
    # Diagnostic methods
    # -----------------------------------------------------------------

    def get_shrunk_expected_returns(self) -> Optional[pd.Series]:
        """Return the JS-shrunk expected return vector from the last fit.

        Returns
        -------
        pd.Series or None
            JS-shrunk annualised expected returns indexed by asset name.
        """
        if self._shrunk_mu is None or self._asset_names is None:
            return None
        annualised = self._shrunk_mu * self.cfg.annualisation_factor
        return pd.Series(
            annualised, index=self._asset_names, name="js_expected_return"
        )

    def get_shrinkage_intensities(self) -> Dict[str, float]:
        """Return the shrinkage intensity per target from the last fit.

        A value of 0.0 means no shrinkage (MLE dominates); 1.0 means
        full shrinkage to the target.

        Returns
        -------
        dict
            Mapping from target name to shrinkage intensity.
        """
        return dict(self._shrinkage_intensities)

    def get_shrunk_covariance(self) -> Optional[pd.DataFrame]:
        """Return the Ledoit-Wolf shrunk covariance matrix.

        Returns
        -------
        pd.DataFrame or None
        """
        if self._shrunk_cov is None or self._asset_names is None:
            return None
        return pd.DataFrame(
            self._shrunk_cov,
            index=self._asset_names,
            columns=self._asset_names,
        )

    def get_portfolio_weights(self) -> Optional[pd.Series]:
        """Return signed portfolio weights from the last fit.

        Returns
        -------
        pd.Series or None
            Signed weights (positive = long, negative = short).
        """
        if self._portfolio_weights is None or self._asset_names is None:
            return None
        return pd.Series(
            self._portfolio_weights,
            index=self._asset_names,
            name="stein_weight",
        )

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        targets = ", ".join(self.cfg.shrinkage_targets)
        intensity_info = ""
        if self._fitted and self._shrinkage_intensities:
            avg = np.mean(list(self._shrinkage_intensities.values()))
            intensity_info = f", avg_intensity={avg:.3f}"
        return (
            f"SteinShrinkageStrategy({fitted_tag}, "
            f"targets=[{targets}]{intensity_info})"
        )
