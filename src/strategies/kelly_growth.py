"""Growth-optimal (Kelly) portfolio strategy.

Maximises the expected logarithmic growth rate of wealth — the criterion
derived from information theory and stochastic optimisation that guarantees
the fastest asymptotic capital accumulation among all essentially
different strategies.

Mathematical foundation
-----------------------
The Kelly criterion maximises E[log(W_T)].  For N assets with jointly
Gaussian returns characterised by mean vector mu and covariance matrix
Sigma, the full-Kelly allocation is:

    f* = Sigma^{-1} mu                            (unconstrained)

Fractional Kelly scales this by alpha in (0, 1):

    f = alpha * Sigma^{-1} mu

In continuous time the geometric growth rate of a portfolio with weight
vector w is:

    g(w) = w' mu  -  (1/2) w' Sigma w

which is exactly the quantity Kelly maximises (the second term is the
"volatility drag" that makes geometric mean < arithmetic mean).

With finite estimation horizon T and N assets, the *estimation-adjusted*
Kelly (accounting for in-sample overfitting of the estimated Sharpe
ratio) is:

    f* = ((T - N - 2) / (T (1 + mu_hat' Sigma_hat^{-1} mu_hat)))
         * Sigma_hat^{-1} mu_hat

This naturally shrinks toward zero when the sample is small relative to
the number of assets, providing built-in regularisation.

Strategy
--------
1.  **Parameter estimation** -- rolling 252-day window for mu_hat and
    Sigma_hat.  Mean returns are shrunk toward zero via a James-Stein
    estimator; the covariance matrix is shrunk via the Ledoit-Wolf
    estimator (sklearn).

2.  **Fractional Kelly with adaptive fraction** -- base fraction
    alpha = 0.25 (quarter Kelly).  Adjusted upward when the
    signal-to-noise ratio is favourable (high T/N) and downward in
    high-volatility regimes:

        alpha_t = 0.25 * min(1, T / (5*N)) * (sigma_target / sigma_realized)

3.  **Growth-rate monitoring** -- tracks the realised geometric growth
    rate and compares it to the theoretical optimum.  Persistent
    under-performance triggers a drawdown in the Kelly fraction
    (signal decay detection).

4.  **Constraints** -- max 20 % per asset, max 1.5x gross leverage,
    optional long-only mode, and rebalancing only when drift exceeds 3 %.

References
----------
*   Kelly, J. L. (1956). A new interpretation of information rate.
    Bell System Technical Journal 35(4).
*   Thorp, E. O. (2006). The Kelly criterion in blackjack, sports
    betting, and the stock market.  Handbook of Asset and Liability
    Management.
*   MacLean, Thorp & Ziemba (2011). The Kelly Capital Growth Investment
    Criterion.  World Scientific.
*   Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for
    large-dimensional covariance matrices.  J. Multivariate Analysis.
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
class KellyGrowthConfig:
    """All tuneable parameters for the Kelly growth strategy."""

    # Estimation windows
    lookback: int = 252                 # rolling window for mu, Sigma estimation
    min_history: int = 252              # minimum observations before trading

    # Fractional Kelly
    base_fraction: float = 0.25        # quarter-Kelly baseline
    sigma_target: float = 0.15         # annualised vol target for adaptive fraction
    min_fraction: float = 0.05         # floor on the Kelly fraction
    max_fraction: float = 0.50         # ceiling on the Kelly fraction

    # Constraints
    max_position_weight: float = 0.20  # max absolute weight per asset
    max_gross_leverage: float = 1.50   # max sum of |w_i|
    long_only: bool = False            # if True, clamp negative weights to zero
    drift_threshold: float = 0.03      # rebalance when L1 drift exceeds this

    # Rebalancing
    rebalance_freq: int = 21           # maximum days between rebalances

    # Growth-rate monitoring
    growth_monitor_window: int = 63    # window for comparing realised vs theoretical g
    decay_threshold: float = 0.50      # if realised/theoretical < this, reduce fraction
    decay_multiplier: float = 0.50     # multiply fraction by this when decay detected


# ---------------------------------------------------------------------------
# James-Stein shrinkage for mean returns
# ---------------------------------------------------------------------------

def _james_stein_shrink_mean(
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    T: int,
) -> np.ndarray:
    """Shrink the sample mean toward zero using the James-Stein estimator.

    The James-Stein estimator dominates the MLE (sample mean) in terms
    of total squared error for dimension N >= 3.  It shrinks toward a
    common target (here zero, appropriate for excess returns):

        mu_JS = (1 - c) * mu_hat

    where c = (N - 2) / (T * mu_hat' Sigma_hat^{-1} mu_hat).

    Parameters
    ----------
    mu_hat : (N,) array
        Sample mean return vector.
    Sigma_hat : (N, N) array
        Covariance matrix estimate (used for Mahalanobis norm).
    T : int
        Number of observations in the estimation window.

    Returns
    -------
    mu_shrunk : (N,) array
        James-Stein shrunk mean estimate.
    """
    N = len(mu_hat)
    if N < 3:
        # James-Stein inadmissibility result requires N >= 3
        return mu_hat.copy()

    try:
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    except np.linalg.LinAlgError:
        return mu_hat.copy()

    quad_form = float(mu_hat @ Sigma_inv @ mu_hat)
    if quad_form < 1e-15:
        # mu is essentially zero already
        return mu_hat.copy()

    shrinkage = (N - 2) / (T * quad_form)
    # Clamp to [0, 1] — positive-part James-Stein
    shrinkage = min(max(shrinkage, 0.0), 1.0)

    mu_shrunk = (1.0 - shrinkage) * mu_hat
    logger.debug(
        "James-Stein shrinkage factor: %.4f (N=%d, T=%d, quad_form=%.6f)",
        shrinkage, N, T, quad_form,
    )
    return mu_shrunk


# ---------------------------------------------------------------------------
# Estimation-adjusted Kelly weights
# ---------------------------------------------------------------------------

def _estimation_adjusted_kelly(
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    T: int,
) -> np.ndarray:
    """Compute the estimation-adjusted full-Kelly weights.

    Accounts for the finite-sample bias in the estimated in-sample
    Sharpe ratio.  The adjustment factor (T - N - 2) / (T (1 + SR^2))
    shrinks the allocation toward zero when the estimation window is
    short relative to the number of assets.

        f* = ((T - N - 2) / (T * (1 + mu_hat' Sigma_hat^{-1} mu_hat)))
             * Sigma_hat^{-1} mu_hat

    Parameters
    ----------
    mu_hat : (N,) array
        (Possibly shrunk) mean return estimates.
    Sigma_hat : (N, N) array
        Covariance estimate.
    T : int
        Effective sample size.

    Returns
    -------
    f : (N,) array
        Full-Kelly weight vector (before fractional scaling).
    """
    N = len(mu_hat)

    try:
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    except np.linalg.LinAlgError:
        logger.warning(
            "Covariance inversion failed; returning zero weights."
        )
        return np.zeros(N)

    raw_kelly = Sigma_inv @ mu_hat  # Sigma^{-1} mu

    # Estimation adjustment factor
    quad_form = float(mu_hat @ Sigma_inv @ mu_hat)
    numerator = max(T - N - 2, 0)
    denominator = T * (1.0 + quad_form)

    if denominator < 1e-15:
        return np.zeros(N)

    adjustment = numerator / denominator
    f = adjustment * raw_kelly

    logger.debug(
        "Estimation-adjusted Kelly: adjustment=%.4f (T=%d, N=%d, SR^2=%.4f)",
        adjustment, T, N, quad_form,
    )
    return f


# ---------------------------------------------------------------------------
# Adaptive Kelly fraction
# ---------------------------------------------------------------------------

def _adaptive_kelly_fraction(
    base_fraction: float,
    T: int,
    N: int,
    sigma_target: float,
    sigma_realized: float,
    min_fraction: float,
    max_fraction: float,
) -> float:
    """Compute the adaptive Kelly fraction.

        alpha_t = base * min(1, T / (5*N)) * (sigma_target / sigma_realized)

    The first factor rewards having more data per asset (high T/N ratio),
    the second moderates aggressiveness when realised volatility is above
    the target.

    Parameters
    ----------
    base_fraction : float
        Baseline Kelly fraction (e.g. 0.25 = quarter Kelly).
    T : int
        Effective sample size.
    N : int
        Number of assets.
    sigma_target : float
        Annualised vol target.
    sigma_realized : float
        Annualised realised portfolio volatility.
    min_fraction : float
        Floor for the Kelly fraction.
    max_fraction : float
        Ceiling for the Kelly fraction.

    Returns
    -------
    alpha : float
        Adaptive Kelly fraction in [min_fraction, max_fraction].
    """
    # Data-sufficiency adjustment: ramps from 0 to 1 as T grows from 0 to 5*N
    data_factor = min(1.0, T / (5 * N)) if N > 0 else 1.0

    # Volatility adjustment: scale down when realised vol exceeds target
    if sigma_realized > 1e-10:
        vol_factor = sigma_target / sigma_realized
    else:
        vol_factor = 1.0

    alpha = base_fraction * data_factor * vol_factor
    alpha = float(np.clip(alpha, min_fraction, max_fraction))

    return alpha


# ---------------------------------------------------------------------------
# Geometric growth rate computation
# ---------------------------------------------------------------------------

def _theoretical_growth_rate(
    mu: np.ndarray,
    Sigma: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute the theoretical (continuous-time) geometric growth rate.

        g(w) = w' mu  -  (1/2) w' Sigma w

    Parameters
    ----------
    mu : (N,) array
        Expected return vector (per period).
    Sigma : (N, N) array
        Covariance matrix (per period).
    weights : (N,) array
        Portfolio weight vector.

    Returns
    -------
    g : float
        Geometric growth rate per period.
    """
    port_return = float(weights @ mu)
    port_var = float(weights @ Sigma @ weights)
    return port_return - 0.5 * port_var


def _realised_geometric_growth_rate(
    log_returns: np.ndarray,
) -> float:
    """Compute realised geometric growth rate from a series of log returns.

    Simply the mean log return (since log returns are additive).

    Parameters
    ----------
    log_returns : (T,) array
        Portfolio log returns.

    Returns
    -------
    g : float
        Mean log return per period.
    """
    if len(log_returns) == 0:
        return 0.0
    return float(np.mean(log_returns))


# ---------------------------------------------------------------------------
# Constraint application
# ---------------------------------------------------------------------------

def _apply_constraints(
    weights: np.ndarray,
    max_position: float,
    max_leverage: float,
    long_only: bool,
) -> np.ndarray:
    """Apply position-level and portfolio-level constraints.

    Parameters
    ----------
    weights : (N,) array
        Raw portfolio weights.
    max_position : float
        Maximum absolute weight per asset.
    max_leverage : float
        Maximum gross leverage (sum of |w_i|).
    long_only : bool
        If True, set negative weights to zero.

    Returns
    -------
    constrained : (N,) array
        Weight vector after constraints.
    """
    w = weights.copy()

    # Long-only constraint
    if long_only:
        w = np.maximum(w, 0.0)

    # Per-asset cap
    w = np.clip(w, -max_position, max_position)

    # Gross leverage cap: scale proportionally if needed
    gross = np.abs(w).sum()
    if gross > max_leverage and gross > 1e-12:
        w *= max_leverage / gross

    return w


# ===========================================================================
# Strategy class
# ===========================================================================

class KellyGrowthStrategy(Strategy):
    """Growth-optimal (Kelly criterion) portfolio strategy.

    Implements the full estimation-adjusted, fractional, and adaptive
    Kelly allocation pipeline:

    1.  Estimate mu and Sigma with James-Stein and Ledoit-Wolf shrinkage.
    2.  Compute the estimation-adjusted Kelly weights.
    3.  Scale by an adaptive Kelly fraction that responds to data
        quality (T/N ratio) and realised volatility.
    4.  Monitor the geometric growth rate and reduce exposure when
        realised growth persistently under-performs theoretical.
    5.  Apply position-level and leverage constraints; rebalance only
        when portfolio drift exceeds a threshold.

    Parameters
    ----------
    config : KellyGrowthConfig, optional
        Strategy configuration.  Uses sensible defaults if not provided.
    """

    def __init__(self, config: Optional[KellyGrowthConfig] = None) -> None:
        self.cfg = config or KellyGrowthConfig()

        super().__init__(
            name="KellyGrowth",
            description=(
                "Growth-optimal (Kelly criterion) portfolio strategy with "
                "estimation-adjusted weights, Ledoit-Wolf covariance shrinkage, "
                "James-Stein mean shrinkage, and adaptive Kelly fraction."
            ),
        )

        # State populated during fit / generate_signals
        self._mu_hat: Optional[np.ndarray] = None
        self._Sigma_hat: Optional[np.ndarray] = None
        self._asset_names: Optional[pd.Index] = None

        # Diagnostics populated during generate_signals
        self._kelly_fractions: Optional[pd.Series] = None
        self._realised_growth: Optional[pd.Series] = None
        self._theoretical_growth: Optional[pd.Series] = None
        self._weight_history: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal: covariance estimation via Ledoit-Wolf
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_covariance_lw(returns: np.ndarray) -> np.ndarray:
        """Estimate the covariance matrix using the Ledoit-Wolf shrinkage.

        Parameters
        ----------
        returns : (T, N) array
            Return observations.

        Returns
        -------
        Sigma : (N, N) array
            Ledoit-Wolf shrunk covariance matrix (guaranteed PSD).
        """
        lw = LedoitWolf()
        lw.fit(returns)
        return lw.covariance_

    @staticmethod
    def _ensure_psd(Sigma: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
        """Ensure a matrix is symmetric positive semi-definite.

        Clips any eigenvalue below min_eigenvalue and reconstructs.
        """
        Sigma = 0.5 * (Sigma + Sigma.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(Sigma)
            eigvals = np.maximum(eigvals, min_eigenvalue)
            Sigma = (eigvecs * eigvals[np.newaxis, :]) @ eigvecs.T
            Sigma = 0.5 * (Sigma + Sigma.T)
        except np.linalg.LinAlgError:
            Sigma += min_eigenvalue * np.eye(Sigma.shape[0])
        return Sigma

    # ------------------------------------------------------------------
    # Internal: realised portfolio volatility
    # ------------------------------------------------------------------

    @staticmethod
    def _annualised_vol(log_returns: np.ndarray) -> float:
        """Annualised volatility from a series of daily log returns."""
        if len(log_returns) < 2:
            return 0.0
        return float(np.std(log_returns, ddof=1) * np.sqrt(252))

    # ------------------------------------------------------------------
    # Strategy interface: fit
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "KellyGrowthStrategy":
        """Calibrate initial parameter estimates on historical prices.

        Computes sample mean and Ledoit-Wolf covariance over the trailing
        lookback window, and stores them for use in ``generate_signals``.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (DatetimeIndex, columns = tickers).

        Returns
        -------
        self
        """
        self.validate_prices(prices)
        self._asset_names = prices.columns

        log_returns = np.log(prices / prices.shift(1)).dropna()
        tail = log_returns.iloc[-self.cfg.lookback:]

        if len(tail) < max(self.cfg.min_history // 2, 30):
            warnings.warn(
                f"Insufficient history for Kelly estimation: "
                f"{len(tail)} observations (need at least "
                f"{self.cfg.min_history // 2}).",
                stacklevel=2,
            )

        returns_array = tail.values.astype(np.float64)

        # Replace any remaining NaN with 0 for estimation
        returns_array = np.nan_to_num(returns_array, nan=0.0)

        # Ledoit-Wolf covariance
        self._Sigma_hat = self._estimate_covariance_lw(returns_array)
        self._Sigma_hat = self._ensure_psd(self._Sigma_hat)

        # James-Stein shrunk mean
        mu_raw = np.mean(returns_array, axis=0)
        self._mu_hat = _james_stein_shrink_mean(
            mu_raw, self._Sigma_hat, T=len(returns_array),
        )

        self.parameters = {
            "lookback": self.cfg.lookback,
            "base_fraction": self.cfg.base_fraction,
            "sigma_target": self.cfg.sigma_target,
            "max_position_weight": self.cfg.max_position_weight,
            "max_gross_leverage": self.cfg.max_gross_leverage,
            "long_only": self.cfg.long_only,
            "n_assets": len(self._asset_names),
            "estimation_window": len(tail),
        }

        self._fitted = True
        logger.info(
            "KellyGrowth fit complete: %d assets, %d-day estimation window, "
            "Ledoit-Wolf covariance, James-Stein mean shrinkage.",
            len(self._asset_names), len(tail),
        )
        return self

    # ------------------------------------------------------------------
    # Strategy interface: generate_signals
    # ------------------------------------------------------------------

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate Kelly-optimal portfolio signals.

        Walks forward through the price data, re-estimating parameters
        at each rebalance point and computing the estimation-adjusted,
        fractionally-scaled, adaptively-modulated Kelly weights.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (DatetimeIndex, columns = tickers).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for
            each asset.  Signal encodes direction (+1 / -1 / 0); weight
            encodes the absolute allocation.
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        n_assets = prices.shape[1]
        n_dates = len(prices)
        tickers = list(prices.columns)
        log_returns = np.log(prices / prices.shift(1))

        # Output storage
        output_cols: List[str] = []
        for t in tickers:
            output_cols.extend([f"{t}_signal", f"{t}_weight"])
        signals_df = pd.DataFrame(0.0, index=prices.index, columns=output_cols)

        # Diagnostics storage
        kelly_fractions = np.zeros(n_dates)
        realised_g = np.full(n_dates, np.nan)
        theoretical_g = np.full(n_dates, np.nan)
        all_weights = np.zeros((n_dates, n_assets))

        # Portfolio tracking
        current_weights = np.zeros(n_assets)
        portfolio_log_returns: List[float] = []
        last_rebalance_idx = -self.cfg.rebalance_freq  # force first rebalance
        decay_active = False

        for t in range(n_dates):
            r_t = log_returns.iloc[t].values.astype(np.float64)
            r_t = np.nan_to_num(r_t, nan=0.0)

            # Track portfolio return
            port_log_ret = float(current_weights @ r_t)
            portfolio_log_returns.append(port_log_ret)

            if t < self.cfg.min_history:
                # Not enough history — stay flat
                all_weights[t] = current_weights
                kelly_fractions[t] = 0.0
                continue

            # --- Check if rebalancing is needed ---
            need_rebalance = False

            # Time-based trigger
            if (t - last_rebalance_idx) >= self.cfg.rebalance_freq:
                need_rebalance = True

            # Drift-based trigger: check if weights have drifted too far
            if not need_rebalance and np.abs(current_weights).sum() > 1e-10:
                # Estimate drifted weights from returns since last rebalance
                # (approximate: assume small returns)
                drift = np.sum(np.abs(
                    current_weights * (np.exp(r_t) - 1.0)
                ))
                if drift > self.cfg.drift_threshold:
                    need_rebalance = True

            if need_rebalance:
                # --- Re-estimate parameters ---
                start = max(0, t - self.cfg.lookback)
                window_returns = log_returns.iloc[start:t].dropna()
                T_eff = len(window_returns)

                if T_eff < max(n_assets + 5, 30):
                    # Insufficient data — hold current weights
                    all_weights[t] = current_weights
                    kelly_fractions[t] = kelly_fractions[max(0, t - 1)]
                    continue

                returns_array = window_returns.values.astype(np.float64)
                returns_array = np.nan_to_num(returns_array, nan=0.0)

                # Ledoit-Wolf covariance
                Sigma_hat = self._estimate_covariance_lw(returns_array)
                Sigma_hat = self._ensure_psd(Sigma_hat)

                # James-Stein mean shrinkage
                mu_raw = np.mean(returns_array, axis=0)
                mu_hat = _james_stein_shrink_mean(mu_raw, Sigma_hat, T=T_eff)

                # --- Estimation-adjusted Kelly weights ---
                full_kelly = _estimation_adjusted_kelly(mu_hat, Sigma_hat, T=T_eff)

                # --- Adaptive Kelly fraction ---
                # Realised portfolio volatility from recent returns
                recent_port_rets = np.array(
                    portfolio_log_returns[max(0, len(portfolio_log_returns) - 63):]
                )
                sigma_realised = self._annualised_vol(recent_port_rets)
                if sigma_realised < 1e-10:
                    sigma_realised = self.cfg.sigma_target  # fallback

                alpha = _adaptive_kelly_fraction(
                    base_fraction=self.cfg.base_fraction,
                    T=T_eff,
                    N=n_assets,
                    sigma_target=self.cfg.sigma_target,
                    sigma_realized=sigma_realised,
                    min_fraction=self.cfg.min_fraction,
                    max_fraction=self.cfg.max_fraction,
                )

                # --- Growth rate monitoring ---
                if len(portfolio_log_returns) >= self.cfg.growth_monitor_window:
                    recent_log_rets = np.array(
                        portfolio_log_returns[-self.cfg.growth_monitor_window:]
                    )
                    g_realised = _realised_geometric_growth_rate(recent_log_rets)
                    g_theoretical = _theoretical_growth_rate(
                        mu_hat, Sigma_hat, current_weights,
                    )

                    realised_g[t] = g_realised
                    theoretical_g[t] = g_theoretical

                    # Decay detection: if realised growth is persistently
                    # below theoretical, the signal has decayed
                    if g_theoretical > 1e-10:
                        ratio = g_realised / g_theoretical
                        if ratio < self.cfg.decay_threshold:
                            if not decay_active:
                                logger.info(
                                    "Growth-rate decay detected at index %d: "
                                    "realised/theoretical = %.4f < %.4f. "
                                    "Reducing Kelly fraction.",
                                    t, ratio, self.cfg.decay_threshold,
                                )
                            decay_active = True
                        else:
                            decay_active = False

                if decay_active:
                    alpha *= self.cfg.decay_multiplier

                kelly_fractions[t] = alpha

                # --- Scale and constrain ---
                target_weights = alpha * full_kelly

                target_weights = _apply_constraints(
                    target_weights,
                    max_position=self.cfg.max_position_weight,
                    max_leverage=self.cfg.max_gross_leverage,
                    long_only=self.cfg.long_only,
                )

                current_weights = target_weights
                last_rebalance_idx = t

                # Update stored estimates for diagnostics
                self._mu_hat = mu_hat
                self._Sigma_hat = Sigma_hat

            else:
                kelly_fractions[t] = kelly_fractions[max(0, t - 1)]

            all_weights[t] = current_weights

        # --- Build output DataFrame ---
        for i, ticker in enumerate(tickers):
            w = all_weights[:, i]
            signals_df[f"{ticker}_signal"] = np.sign(w)
            signals_df[f"{ticker}_weight"] = np.abs(w)

        # --- Store diagnostics ---
        self._kelly_fractions = pd.Series(
            kelly_fractions, index=prices.index, name="kelly_fraction",
        )
        self._realised_growth = pd.Series(
            realised_g, index=prices.index, name="realised_growth_rate",
        )
        self._theoretical_growth = pd.Series(
            theoretical_g, index=prices.index, name="theoretical_growth_rate",
        )
        self._weight_history = pd.DataFrame(
            all_weights, index=prices.index, columns=tickers,
        )

        return signals_df

    # ------------------------------------------------------------------
    # Diagnostic methods
    # ------------------------------------------------------------------

    def get_kelly_fractions(self) -> Optional[pd.Series]:
        """Return the adaptive Kelly fraction time series.

        Returns
        -------
        pd.Series or None
            Kelly fraction alpha_t at each date, or None if
            ``generate_signals`` has not been called.
        """
        return self._kelly_fractions

    def get_growth_rates(self) -> Optional[pd.DataFrame]:
        """Return realised and theoretical growth rates side-by-side.

        The theoretical growth rate is g(w) = w'mu - (1/2) w'Sigma w
        evaluated at the current estimates.  The realised growth rate
        is the rolling mean of portfolio log returns.

        Returns
        -------
        pd.DataFrame or None
            Two columns: 'realised' and 'theoretical', indexed by date.
        """
        if self._realised_growth is None or self._theoretical_growth is None:
            return None
        return pd.DataFrame({
            "realised": self._realised_growth,
            "theoretical": self._theoretical_growth,
        })

    def get_weight_history(self) -> Optional[pd.DataFrame]:
        """Return the full weight history as a DataFrame.

        Returns
        -------
        pd.DataFrame or None
            Columns are asset tickers; rows are dates.
        """
        return self._weight_history

    def get_current_estimates(self) -> Optional[Dict[str, Any]]:
        """Return the current mu_hat and Sigma_hat estimates.

        Returns
        -------
        dict or None
            Keys: 'mu_hat' (pd.Series), 'Sigma_hat' (pd.DataFrame).
        """
        if self._mu_hat is None or self._Sigma_hat is None:
            return None
        if self._asset_names is None:
            return None
        return {
            "mu_hat": pd.Series(
                self._mu_hat, index=self._asset_names, name="mu_hat",
            ),
            "Sigma_hat": pd.DataFrame(
                self._Sigma_hat,
                index=self._asset_names,
                columns=self._asset_names,
            ),
        }

    def get_growth_efficiency(self) -> Optional[pd.Series]:
        """Return the ratio of realised to theoretical growth rate.

        Values near 1.0 indicate the strategy is capturing most of the
        available growth.  Values persistently below the decay_threshold
        suggest signal degradation.

        Returns
        -------
        pd.Series or None
            Ratio series, or None if not yet computed.
        """
        if self._realised_growth is None or self._theoretical_growth is None:
            return None
        # Avoid division by zero
        theoretical = self._theoretical_growth.replace(0.0, np.nan)
        ratio = self._realised_growth / theoretical
        ratio.name = "growth_efficiency"
        return ratio

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        n_assets = len(self._asset_names) if self._asset_names is not None else 0
        mode = "long-only" if self.cfg.long_only else "long-short"
        return (
            f"KellyGrowthStrategy({fitted_tag}, {n_assets} assets, "
            f"{mode}, alpha_base={self.cfg.base_fraction})"
        )
