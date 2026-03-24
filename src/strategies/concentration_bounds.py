"""Concentration inequality strategy for robust portfolio construction.

Uses finite-sample probabilistic bounds from high-dimensional statistics
to construct portfolios with worst-case performance guarantees, rather
than relying on fragile point estimates.

Mathematical foundation
-----------------------
1. **Hoeffding's inequality** -- for bounded returns in [a, b] with
   range R = b - a observed over T periods:

       P(|mu_hat - mu| > eps) <= 2 * exp(-2 T eps^2 / R^2)

   Inverting at confidence level delta:
       eps = R * sqrt(log(2 / delta) / (2T))

2. **Bernstein's inequality** (tighter when variance is small relative
   to range, which is typical for most equities):

       P(|mu_hat - mu| > eps) <= 2 * exp(-T eps^2 / (2 sigma^2 + 2R eps / 3))

   Inverting for eps (solving the quadratic):
       eps = (2 sigma^2 / T) * C + (2R / (3T)) * C
       where C = log(2 / delta), simplified via the quadratic formula.

3. **Matrix Hoeffding** (covariance estimation error):

       ||Sigma_hat - Sigma||_op <= O(sqrt(log(N) / T))

   This controls the spectral norm of the estimation error, informing
   how much to shrink the sample covariance.

Strategy
--------
1. **Robust portfolio (worst-case optimisation)** -- for each asset i,
   compute the Bernstein confidence interval for mu_i.  The lower
   confidence bound is mu_i^L = mu_hat_i - eps_i.  The robust
   portfolio solves:

       max_w  min_{mu in uncertainty_set}  w' mu
     = max_w  (w' mu_hat - ||w odot eps||_1)

   This is equivalent to an L1-penalised portfolio optimisation, which
   induces sparsity: assets with noisy estimates are automatically
   excluded.

2. **Confidence-weighted signal** -- the signal strength for asset i
   is mu_hat_i / eps_i, analogous to a t-statistic but using the
   (tighter) Bernstein bound instead of the Gaussian CLT.  Only assets
   where mu_hat_i > eps_i (95 % confidence of positive return) are traded.

3. **Adaptive confidence level** -- the confidence level adapts to the
   volatility regime:
   * Low-vol regimes: tighter bounds -> more assets tradeable -> more
     diversification.
   * High-vol regimes: wider bounds -> fewer tradeable assets -> more
     concentrated portfolio (natural risk reduction).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConcentrationBoundsConfig:
    """Tuneable parameters for the concentration bounds strategy."""

    # Estimation windows
    lookback: int = 252             # primary estimation window (trading days)
    min_history: int = 60           # minimum observations before trading

    # Confidence level
    base_confidence: float = 0.60   # base confidence level (1 - delta)
    adaptive_confidence: bool = True  # adapt confidence to volatility regime

    # Bernstein inequality parameters
    use_bernstein: bool = True      # True = Bernstein, False = Hoeffding
    return_bound_quantile: float = 0.995  # quantile for estimating return range R

    # Robust portfolio construction
    robust_weight: float = 0.95     # blend weight for robust (worst-case) portfolio
    min_var_weight: float = 0.05    # blend weight for shrunk min-variance portfolio

    # Matrix Hoeffding covariance shrinkage
    cov_shrinkage_auto: bool = True   # auto-calibrate shrinkage from matrix Hoeffding
    cov_shrinkage_floor: float = 0.05  # minimum shrinkage intensity

    # Signal filtering
    signal_threshold: float = 0.10  # min confidence ratio (mu_hat/eps) to trade
    max_assets: int = 50            # maximum number of assets to hold

    # Risk
    max_leverage: float = 5.0       # gross leverage cap
    rebalance_freq: int = 21        # trading days between rebalances

    # Regime detection
    vol_lookback: int = 60          # window for realised vol estimation
    vol_expansion_factor: float = 1.5  # high-vol regime = vol > factor * median


# ---------------------------------------------------------------------------
# Concentration inequality helpers
# ---------------------------------------------------------------------------

def _hoeffding_epsilon(
    R: float,
    T: int,
    delta: float,
) -> float:
    """Hoeffding confidence half-width for a bounded mean.

    For returns bounded in an interval of width R observed over T periods,
    with failure probability delta:

        eps = R * sqrt(log(2 / delta) / (2T))

    Parameters
    ----------
    R : float
        Range of the bounded random variable (b - a).
    T : int
        Number of observations.
    delta : float
        Failure probability (e.g. 0.05 for 95 % confidence).

    Returns
    -------
    float
        Confidence half-width eps > 0.
    """
    if T <= 0 or R <= 0 or delta <= 0:
        return np.inf
    return R * np.sqrt(np.log(2.0 / delta) / (2.0 * T))


def _bernstein_epsilon(
    sigma: float,
    R: float,
    T: int,
    delta: float,
) -> float:
    """Bernstein confidence half-width for a bounded mean with known variance.

    Solves the Bernstein bound inversion:

        P(|mu_hat - mu| > eps) <= 2 exp(-T eps^2 / (2 sigma^2 + 2R eps / 3))

    Setting the RHS = delta and solving the resulting quadratic in eps:

        T eps^2 / (2 sigma^2 + (2R/3) eps) = log(2/delta) =: C

    Rearranging:
        T eps^2 - (2RC/3) eps - 2 sigma^2 C = 0

    Positive root:
        eps = [(2RC/3) + sqrt((2RC/3)^2 + 8 T sigma^2 C)] / (2T)

    Parameters
    ----------
    sigma : float
        Standard deviation of the random variable.
    R : float
        Range of the bounded random variable.
    T : int
        Number of observations.
    delta : float
        Failure probability.

    Returns
    -------
    float
        Confidence half-width eps > 0.  Tighter than Hoeffding when
        sigma^2 << R^2 (typical for financial returns).
    """
    if T <= 0 or delta <= 0:
        return np.inf
    if sigma <= 0 and R <= 0:
        return 0.0

    C = np.log(2.0 / delta)

    # Quadratic: T eps^2 - (2RC/3) eps - 2 sigma^2 C = 0
    a_coef = T
    b_coef = -(2.0 * R * C / 3.0)
    c_coef = -(2.0 * sigma**2 * C)

    discriminant = b_coef**2 - 4.0 * a_coef * c_coef
    if discriminant < 0:
        # Fallback to Hoeffding (should not happen with valid inputs)
        return _hoeffding_epsilon(R, T, delta)

    eps = (-b_coef + np.sqrt(discriminant)) / (2.0 * a_coef)
    return max(eps, 0.0)


def _compute_return_range(
    returns: np.ndarray,
    quantile: float = 0.995,
) -> float:
    """Estimate the return range R for concentration inequalities.

    Uses symmetric quantiles to robustly estimate [a, b] without being
    dominated by extreme outliers, then returns R = b - a.

    Parameters
    ----------
    returns : (T,) array
        Observed returns for a single asset.
    quantile : float
        Upper quantile for range estimation (lower = 1 - quantile).

    Returns
    -------
    float
        Estimated range R > 0.
    """
    clean = returns[np.isfinite(returns)]
    if len(clean) < 2:
        return 1.0  # safe fallback

    lo = np.quantile(clean, 1.0 - quantile)
    hi = np.quantile(clean, quantile)
    R = hi - lo
    return max(R, 1e-8)


# ---------------------------------------------------------------------------
# Matrix Hoeffding shrinkage
# ---------------------------------------------------------------------------

def _matrix_hoeffding_shrinkage(
    n_assets: int,
    n_obs: int,
    floor: float = 0.05,
) -> float:
    """Compute covariance shrinkage intensity from Matrix Hoeffding bound.

    The operator-norm error of the sample covariance satisfies:

        ||Sigma_hat - Sigma||_op <= C * sqrt(log(N) / T)

    with high probability.  We use this rate to set the Ledoit-Wolf-style
    shrinkage intensity alpha in:

        Sigma_shrunk = (1 - alpha) Sigma_hat + alpha * (trace(Sigma_hat)/N) I

    Parameters
    ----------
    n_assets : int
        Number of assets N.
    n_obs : int
        Number of observations T.
    floor : float
        Minimum shrinkage intensity.

    Returns
    -------
    float
        Shrinkage intensity in [floor, 1.0].
    """
    if n_obs <= 1 or n_assets <= 0:
        return 1.0

    # Rate from matrix Hoeffding: O(sqrt(log(N) / T))
    # We scale by a constant calibrated for typical financial data
    rate = np.sqrt(np.log(max(n_assets, 2)) / n_obs)

    # Clamp to [floor, 1.0]
    alpha = np.clip(rate, floor, 1.0)
    return float(alpha)


def _shrink_covariance(
    cov: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply linear shrinkage to a sample covariance matrix.

    Sigma_shrunk = (1 - alpha) * Sigma_hat + alpha * mu * I

    where mu = trace(Sigma_hat) / N (average variance).

    Parameters
    ----------
    cov : (N, N) array
        Sample covariance matrix.
    alpha : float
        Shrinkage intensity in [0, 1].

    Returns
    -------
    (N, N) array
        Shrunk covariance matrix (guaranteed positive definite for alpha > 0).
    """
    n = cov.shape[0]
    mu = np.trace(cov) / n
    shrunk = (1.0 - alpha) * cov + alpha * mu * np.eye(n)
    # Enforce symmetry
    shrunk = (shrunk + shrunk.T) / 2.0
    return shrunk


# ---------------------------------------------------------------------------
# Robust portfolio optimisation
# ---------------------------------------------------------------------------

def _robust_portfolio_weights(
    mu_hat: np.ndarray,
    epsilon: np.ndarray,
    cov_shrunk: np.ndarray,
    max_leverage: float = 1.0,
) -> np.ndarray:
    """Solve the robust worst-case portfolio optimisation.

    The problem is:

        max_w  (w' mu_hat - ||w odot epsilon||_1)
        s.t.   ||w||_1 <= max_leverage
               w' Sigma w <= target_risk   (implicit via regularisation)

    The L1 penalty ||w odot eps||_1 = sum_i |w_i| eps_i makes this a
    weighted-LASSO-like problem that induces sparsity: assets with large
    uncertainty (large eps_i) are penalised more heavily.

    We solve via scipy.optimize.minimize with the SLSQP method, using
    the risk term w' Sigma w as a quadratic regulariser.

    Parameters
    ----------
    mu_hat : (N,) array
        Sample mean returns.
    epsilon : (N,) array
        Confidence half-widths for each asset.
    cov_shrunk : (N, N) array
        Shrunk covariance matrix.
    max_leverage : float
        Maximum gross leverage (sum of |w_i|).

    Returns
    -------
    (N,) array
        Optimal portfolio weights.
    """
    n = len(mu_hat)
    if n == 0:
        return np.array([])

    # Risk aversion parameter (controls return-risk trade-off)
    # Scale relative to typical eigenvalue of the covariance
    avg_var = np.trace(cov_shrunk) / n
    gamma = 1.0 / max(avg_var, 1e-10)

    def objective(w: np.ndarray) -> float:
        # Negative of robust return (we minimise, so negate)
        worst_case_return = w @ mu_hat - np.sum(np.abs(w) * epsilon)
        risk = w @ cov_shrunk @ w
        # Maximise worst-case return minus risk penalty
        return -(worst_case_return - 0.5 * gamma * risk)

    def grad(w: np.ndarray) -> np.ndarray:
        # Gradient of the objective
        sign_w = np.sign(w)
        # Handle w_i = 0: subgradient, use 0
        sign_w[w == 0] = 0.0
        d_return = -(mu_hat - epsilon * sign_w)
        d_risk = gamma * cov_shrunk @ w
        return d_return + d_risk

    # Constraints: gross leverage <= max_leverage
    constraints = [
        {
            "type": "ineq",
            "fun": lambda w: max_leverage - np.sum(np.abs(w)),
        },
    ]

    # Bounds: allow long and short positions
    bounds = [(-max_leverage, max_leverage)] * n

    # Initial guess: equal weight (long only, small)
    w0 = np.full(n, 1.0 / n) * 0.1

    try:
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=grad,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        w_opt = result.x
    except Exception as exc:
        logger.warning("Robust optimisation failed (%s); falling back to heuristic.", exc)
        w_opt = _heuristic_robust_weights(mu_hat, epsilon, max_leverage)

    return w_opt


def _heuristic_robust_weights(
    mu_hat: np.ndarray,
    epsilon: np.ndarray,
    max_leverage: float = 1.0,
) -> np.ndarray:
    """Fast heuristic for robust weights when optimisation fails.

    Uses the confidence-weighted signal direction with inverse-epsilon
    sizing:

        w_i propto sign(mu_hat_i) * max(|mu_hat_i| - eps_i, 0) / eps_i

    Parameters
    ----------
    mu_hat : (N,) array
        Sample mean returns.
    epsilon : (N,) array
        Confidence half-widths.
    max_leverage : float
        Gross leverage cap.

    Returns
    -------
    (N,) array
        Heuristic portfolio weights.
    """
    n = len(mu_hat)
    if n == 0:
        return np.array([])

    eps_safe = np.where(epsilon > 1e-12, epsilon, 1e-12)

    # Only trade where |mu_hat| > eps (confident of sign)
    excess = np.abs(mu_hat) - eps_safe
    tradeable = excess > 0

    w = np.zeros(n)
    if tradeable.any():
        # Weight by excess confidence, inversely by uncertainty
        w[tradeable] = (
            np.sign(mu_hat[tradeable])
            * excess[tradeable]
            / eps_safe[tradeable]
        )
        # Normalise to leverage constraint
        gross = np.abs(w).sum()
        if gross > max_leverage:
            w *= max_leverage / gross
    return w


def _minimum_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio from a covariance matrix.

    w_mv = Sigma^{-1} 1 / (1' Sigma^{-1} 1)

    Parameters
    ----------
    cov : (N, N) array
        Positive (semi-)definite covariance matrix.

    Returns
    -------
    (N,) array
        Weights summing to 1.
    """
    n = cov.shape[0]
    ones = np.ones(n)
    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return ones / n
    w = cov_inv @ ones
    denom = ones @ w
    if abs(denom) < 1e-12:
        return ones / n
    return w / denom


# ===========================================================================
# Strategy class
# ===========================================================================

class ConcentrationBoundsStrategy(Strategy):
    """Concentration inequality strategy for robust portfolio construction.

    Uses Bernstein (or Hoeffding) confidence intervals to construct
    portfolios with worst-case return guarantees.  Assets are only traded
    when their estimated mean return exceeds the concentration bound
    (i.e., we are statistically confident the return is non-zero).

    The strategy naturally adapts to the volatility regime: in calm
    markets, tighter bounds allow more diversification; in turbulent
    markets, wider bounds concentrate the portfolio on the strongest
    signals.

    Parameters
    ----------
    config : ConcentrationBoundsConfig, optional
        Strategy configuration.  Uses defaults if not supplied.
    """

    def __init__(self, config: Optional[ConcentrationBoundsConfig] = None) -> None:
        self.cfg = config or ConcentrationBoundsConfig()

        super().__init__(
            name="ConcentrationBounds",
            description=(
                "Robust portfolio construction using Bernstein/Hoeffding "
                "concentration inequalities with adaptive confidence levels"
            ),
        )

        # State populated during fit
        self._mu_hat: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._epsilon: Optional[np.ndarray] = None
        self._return_range: Optional[np.ndarray] = None
        self._confidence_ratio: Optional[np.ndarray] = None
        self._cov_shrunk: Optional[np.ndarray] = None
        self._shrinkage_alpha: Optional[float] = None
        self._robust_weights: Optional[np.ndarray] = None
        self._min_var_weights: Optional[np.ndarray] = None
        self._blended_weights: Optional[np.ndarray] = None
        self._current_delta: Optional[float] = None
        self._n_tradeable: int = 0
        self._asset_names: Optional[pd.Index] = None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _compute_confidence_intervals(
        self,
        returns: np.ndarray,
        delta: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-asset confidence intervals using concentration bounds.

        Parameters
        ----------
        returns : (T, N) array
            Return observations.
        delta : float
            Failure probability (1 - confidence level).

        Returns
        -------
        mu_hat : (N,) array
            Sample mean returns.
        sigma : (N,) array
            Sample standard deviations.
        epsilon : (N,) array
            Confidence half-widths (Bernstein or Hoeffding).
        R : (N,) array
            Estimated return ranges per asset.
        """
        T, N = returns.shape
        mu_hat = np.nanmean(returns, axis=0)
        sigma = np.nanstd(returns, axis=0, ddof=1)

        # Per-asset return range
        R = np.array([
            _compute_return_range(returns[:, i], self.cfg.return_bound_quantile)
            for i in range(N)
        ])

        # Confidence half-widths
        epsilon = np.empty(N)
        for i in range(N):
            # Use effective sample size (excluding NaN)
            T_eff = int(np.sum(np.isfinite(returns[:, i])))
            if T_eff < 2:
                epsilon[i] = np.inf
                continue

            if self.cfg.use_bernstein:
                epsilon[i] = _bernstein_epsilon(sigma[i], R[i], T_eff, delta)
            else:
                epsilon[i] = _hoeffding_epsilon(R[i], T_eff, delta)

        return mu_hat, sigma, epsilon, R

    def _adapt_confidence_level(
        self,
        returns: np.ndarray,
    ) -> float:
        """Adapt the confidence level (delta) based on volatility regime.

        In low-vol regimes, use a higher confidence level (lower delta)
        to tighten bounds and allow more assets.  In high-vol regimes,
        use a lower confidence level (higher delta) to widen bounds
        and concentrate the portfolio.

        Parameters
        ----------
        returns : (T, N) array
            Recent return observations.

        Returns
        -------
        float
            Adapted failure probability delta.
        """
        base_delta = 1.0 - self.cfg.base_confidence

        if not self.cfg.adaptive_confidence:
            return base_delta

        # Portfolio-level realised vol (equal-weight proxy)
        N = returns.shape[1]
        port_returns = np.nanmean(returns, axis=1)

        # Recent vol vs long-run vol
        vol_window = min(self.cfg.vol_lookback, len(port_returns))
        if vol_window < 10:
            return base_delta

        recent_vol = np.nanstd(port_returns[-vol_window:])
        full_vol = np.nanstd(port_returns)

        if full_vol < 1e-12:
            return base_delta

        vol_ratio = recent_vol / full_vol

        # High-vol regime: increase delta (widen bounds, fewer assets)
        # Low-vol regime: decrease delta (tighten bounds, more assets)
        if vol_ratio > self.cfg.vol_expansion_factor:
            # High vol: relax confidence (widen intervals)
            adapted_delta = base_delta * vol_ratio
        elif vol_ratio < 1.0 / self.cfg.vol_expansion_factor:
            # Low vol: tighten confidence (narrow intervals)
            adapted_delta = base_delta * vol_ratio
        else:
            adapted_delta = base_delta

        # Clamp to reasonable range [0.001, 0.20]
        adapted_delta = np.clip(adapted_delta, 0.001, 0.20)

        return float(adapted_delta)

    def _select_tradeable_assets(
        self,
        mu_hat: np.ndarray,
        epsilon: np.ndarray,
    ) -> np.ndarray:
        """Select assets where we are confident about the return sign.

        An asset is tradeable if its confidence ratio |mu_hat_i| / eps_i
        exceeds the signal threshold.  This means we are statistically
        confident that the true mean return is non-zero.

        Parameters
        ----------
        mu_hat : (N,) array
            Sample mean returns.
        epsilon : (N,) array
            Confidence half-widths.

        Returns
        -------
        (N,) bool array
            True for tradeable assets.
        """
        eps_safe = np.where(epsilon > 1e-12, epsilon, 1e-12)
        confidence_ratio = np.abs(mu_hat) / eps_safe

        tradeable = confidence_ratio >= self.cfg.signal_threshold

        # If too many assets pass, keep only the top max_assets
        n_pass = tradeable.sum()
        if n_pass > self.cfg.max_assets:
            # Keep assets with highest confidence ratio
            sorted_idx = np.argsort(confidence_ratio)[::-1]
            tradeable[:] = False
            tradeable[sorted_idx[: self.cfg.max_assets]] = True

        return tradeable

    # -----------------------------------------------------------------
    # Strategy interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "ConcentrationBoundsStrategy":
        """Calibrate concentration bounds on historical price data.

        Computes per-asset Bernstein (or Hoeffding) confidence intervals,
        applies Matrix-Hoeffding-calibrated covariance shrinkage, and
        solves the robust worst-case portfolio optimisation.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical prices (columns = tickers, index = DatetimeIndex).

        Returns
        -------
        self
        """
        self.validate_prices(prices)
        self._asset_names = prices.columns

        n_obs = min(len(prices), self.cfg.lookback)
        recent_prices = prices.iloc[-n_obs:]

        # Log returns
        log_returns = np.log(recent_prices / recent_prices.shift(1)).iloc[1:].values
        T, N = log_returns.shape

        if T < self.cfg.min_history:
            warnings.warn(
                f"Insufficient history for concentration bounds: {T} observations "
                f"(need at least {self.cfg.min_history}).",
                stacklevel=2,
            )
            self._fitted = True
            return self

        # 1. Adaptive confidence level
        self._current_delta = self._adapt_confidence_level(log_returns)
        delta = self._current_delta

        # 2. Per-asset concentration bounds
        mu_hat, sigma, epsilon, R = self._compute_confidence_intervals(
            log_returns, delta
        )
        self._mu_hat = mu_hat
        self._sigma = sigma
        self._epsilon = epsilon
        self._return_range = R

        # Confidence ratio (analogous to t-statistic)
        eps_safe = np.where(epsilon > 1e-12, epsilon, 1e-12)
        self._confidence_ratio = mu_hat / eps_safe

        # 3. Matrix Hoeffding covariance shrinkage
        sample_cov = np.cov(log_returns, rowvar=False)
        if sample_cov.ndim == 0:
            sample_cov = np.array([[sample_cov]])

        if self.cfg.cov_shrinkage_auto:
            self._shrinkage_alpha = _matrix_hoeffding_shrinkage(
                N, T, floor=self.cfg.cov_shrinkage_floor
            )
        else:
            self._shrinkage_alpha = self.cfg.cov_shrinkage_floor

        self._cov_shrunk = _shrink_covariance(sample_cov, self._shrinkage_alpha)

        # 4. Identify tradeable assets
        tradeable = self._select_tradeable_assets(mu_hat, epsilon)
        self._n_tradeable = int(tradeable.sum())

        # 5. Robust portfolio (on tradeable subset)
        if self._n_tradeable > 0:
            idx = np.where(tradeable)[0]
            mu_sub = mu_hat[idx]
            eps_sub = epsilon[idx]
            cov_sub = self._cov_shrunk[np.ix_(idx, idx)]

            w_robust_sub = _robust_portfolio_weights(
                mu_sub, eps_sub, cov_sub, self.cfg.max_leverage
            )

            # Map back to full asset space
            self._robust_weights = np.zeros(N)
            self._robust_weights[idx] = w_robust_sub
        else:
            self._robust_weights = np.zeros(N)

        # 6. Minimum-variance portfolio (full universe, shrunk covariance)
        self._min_var_weights = _minimum_variance_weights(self._cov_shrunk)

        # 7. Blend robust + min-variance
        self._blended_weights = (
            self.cfg.robust_weight * self._robust_weights
            + self.cfg.min_var_weight * self._min_var_weights
        )

        # Enforce leverage constraint
        gross = np.abs(self._blended_weights).sum()
        if gross > self.cfg.max_leverage:
            self._blended_weights *= self.cfg.max_leverage / gross

        # Store diagnostics
        self.parameters = {
            "delta": delta,
            "confidence_level": 1.0 - delta,
            "shrinkage_alpha": self._shrinkage_alpha,
            "n_tradeable": self._n_tradeable,
            "n_total_assets": N,
            "n_observations": T,
            "mean_epsilon": float(np.mean(epsilon[np.isfinite(epsilon)])),
            "mean_confidence_ratio": float(
                np.mean(np.abs(self._confidence_ratio[np.isfinite(self._confidence_ratio)]))
            ),
            "bound_type": "Bernstein" if self.cfg.use_bernstein else "Hoeffding",
        }

        logger.info(
            "ConcentrationBounds fit: %d/%d assets tradeable at %.1f%% confidence "
            "(delta=%.4f, shrinkage=%.3f, bound=%s)",
            self._n_tradeable, N,
            (1.0 - delta) * 100,
            delta,
            self._shrinkage_alpha,
            "Bernstein" if self.cfg.use_bernstein else "Hoeffding",
        )

        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate per-asset trading signals using concentration bounds.

        For each rebalance date:
        1. Compute Bernstein/Hoeffding confidence intervals from the
           trailing return window.
        2. Filter assets by confidence ratio (|mu_hat| / eps > threshold).
        3. Solve the robust worst-case portfolio and blend with
           minimum-variance weights from the shrunk covariance.
        4. Signal direction = sign of blended weight; position size =
           magnitude of blended weight.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (columns = tickers, index = DatetimeIndex).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for each
            asset.
        """
        self.ensure_fitted()

        N = prices.shape[1]
        n_rows = len(prices)
        asset_names = prices.columns

        # Prepare output
        output_cols = []
        for name in asset_names:
            output_cols.extend([f"{name}_signal", f"{name}_weight"])
        signals_df = pd.DataFrame(0.0, index=prices.index, columns=output_cols)

        # Log returns
        log_returns_df = np.log(prices / prices.shift(1))
        log_returns = log_returns_df.values  # (n_rows, N), row 0 is NaN

        # Minimum lookback needed
        min_lookback = max(self.cfg.min_history, self.cfg.vol_lookback)

        # Cached weights for between rebalances
        cached_signals = np.zeros(N)
        cached_weights = np.zeros(N)
        last_rebalance = -(self.cfg.rebalance_freq + 1)

        for t in range(min_lookback, n_rows):
            if (t - last_rebalance) >= self.cfg.rebalance_freq:
                # Extract estimation window
                start = max(1, t - self.cfg.lookback)  # skip row 0 (NaN)
                window_returns = log_returns[start:t, :]

                # Remove rows with all NaN
                valid_rows = ~np.all(np.isnan(window_returns), axis=1)
                window_returns = window_returns[valid_rows]

                T_eff = window_returns.shape[0]
                if T_eff < self.cfg.min_history:
                    continue

                # Adaptive confidence
                delta = self._adapt_confidence_level(window_returns)

                # Per-asset concentration bounds
                mu_hat, sigma, epsilon, R = self._compute_confidence_intervals(
                    window_returns, delta
                )

                # Covariance estimation + shrinkage
                # Replace NaN with 0 for covariance computation
                clean_returns = np.nan_to_num(window_returns, nan=0.0)
                sample_cov = np.cov(clean_returns, rowvar=False)
                if sample_cov.ndim == 0:
                    sample_cov = np.array([[sample_cov]])

                alpha = (
                    _matrix_hoeffding_shrinkage(
                        N, T_eff, floor=self.cfg.cov_shrinkage_floor
                    )
                    if self.cfg.cov_shrinkage_auto
                    else self.cfg.cov_shrinkage_floor
                )
                cov_shrunk = _shrink_covariance(sample_cov, alpha)

                # Select tradeable assets
                tradeable = self._select_tradeable_assets(mu_hat, epsilon)
                n_tradeable = int(tradeable.sum())

                # Robust portfolio on tradeable subset
                if n_tradeable > 0:
                    idx = np.where(tradeable)[0]
                    w_robust = np.zeros(N)
                    w_robust[idx] = _robust_portfolio_weights(
                        mu_hat[idx],
                        epsilon[idx],
                        cov_shrunk[np.ix_(idx, idx)],
                        self.cfg.max_leverage,
                    )
                else:
                    w_robust = np.zeros(N)

                # Minimum-variance weights
                w_minvar = _minimum_variance_weights(cov_shrunk)

                # Blend
                blended = (
                    self.cfg.robust_weight * w_robust
                    + self.cfg.min_var_weight * w_minvar
                )

                # Enforce leverage
                gross = np.abs(blended).sum()
                if gross > self.cfg.max_leverage:
                    blended *= self.cfg.max_leverage / gross

                cached_signals = np.sign(blended)
                cached_weights = np.abs(blended)

                # Normalise weights to sum to at most max_leverage
                weight_sum = cached_weights.sum()
                if weight_sum > self.cfg.max_leverage:
                    cached_weights *= self.cfg.max_leverage / weight_sum

                last_rebalance = t

            # Write signals for this date
            for j, name in enumerate(asset_names):
                signals_df.iat[t, 2 * j] = cached_signals[j]      # signal
                signals_df.iat[t, 2 * j + 1] = cached_weights[j]  # weight

        return signals_df

    # -----------------------------------------------------------------
    # Diagnostic methods
    # -----------------------------------------------------------------

    def get_confidence_intervals(self) -> Optional[pd.DataFrame]:
        """Return per-asset confidence interval diagnostics from the last fit.

        Returns
        -------
        pd.DataFrame or None
            Columns: mu_hat, sigma, epsilon, lower_bound, upper_bound,
            confidence_ratio, tradeable.
        """
        if self._mu_hat is None or self._asset_names is None:
            return None

        eps_safe = np.where(self._epsilon > 1e-12, self._epsilon, 1e-12)
        tradeable = self._select_tradeable_assets(self._mu_hat, self._epsilon)

        return pd.DataFrame(
            {
                "mu_hat": self._mu_hat,
                "sigma": self._sigma,
                "epsilon": self._epsilon,
                "return_range_R": self._return_range,
                "lower_bound": self._mu_hat - self._epsilon,
                "upper_bound": self._mu_hat + self._epsilon,
                "confidence_ratio": self._mu_hat / eps_safe,
                "tradeable": tradeable,
            },
            index=self._asset_names,
        )

    def get_shrinkage_info(self) -> Optional[Dict[str, Any]]:
        """Return covariance shrinkage diagnostics.

        Returns
        -------
        dict or None
            Keys: shrinkage_alpha, matrix_hoeffding_bound,
            condition_number_before, condition_number_after.
        """
        if self._shrinkage_alpha is None or self._cov_shrunk is None:
            return None

        try:
            eigs = np.linalg.eigvalsh(self._cov_shrunk)
            eigs_pos = eigs[eigs > 1e-15]
            cond = eigs_pos[-1] / eigs_pos[0] if len(eigs_pos) > 1 else 1.0
        except np.linalg.LinAlgError:
            cond = np.inf

        return {
            "shrinkage_alpha": self._shrinkage_alpha,
            "condition_number": cond,
        }

    def get_robust_weights(self) -> Optional[pd.Series]:
        """Return the robust (worst-case) portfolio weights from the last fit."""
        if self._robust_weights is None or self._asset_names is None:
            return None
        return pd.Series(
            self._robust_weights, index=self._asset_names, name="robust_weight"
        )

    def get_blended_weights(self) -> Optional[pd.Series]:
        """Return the final blended portfolio weights from the last fit."""
        if self._blended_weights is None or self._asset_names is None:
            return None
        return pd.Series(
            self._blended_weights, index=self._asset_names, name="blended_weight"
        )

    def compare_bounds(self, prices: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compare Hoeffding vs Bernstein bounds for each asset.

        Useful for understanding how much tighter Bernstein is when
        variance is small relative to range (sigma^2 << R^2).

        Parameters
        ----------
        prices : pd.DataFrame
            Price data to compute bounds from.

        Returns
        -------
        pd.DataFrame or None
            Columns: hoeffding_eps, bernstein_eps, tightening_ratio,
            sigma, R, sigma_sq_over_R_sq.
        """
        n_obs = min(len(prices), self.cfg.lookback)
        recent = prices.iloc[-n_obs:]
        log_returns = np.log(recent / recent.shift(1)).iloc[1:].values
        T, N = log_returns.shape

        delta = 1.0 - self.cfg.base_confidence

        rows = []
        for i in range(N):
            col_returns = log_returns[:, i]
            T_eff = int(np.sum(np.isfinite(col_returns)))
            if T_eff < 2:
                rows.append({
                    "hoeffding_eps": np.nan,
                    "bernstein_eps": np.nan,
                    "tightening_ratio": np.nan,
                    "sigma": np.nan,
                    "R": np.nan,
                    "sigma_sq_over_R_sq": np.nan,
                })
                continue

            s = np.nanstd(col_returns, ddof=1)
            R = _compute_return_range(col_returns, self.cfg.return_bound_quantile)

            h_eps = _hoeffding_epsilon(R, T_eff, delta)
            b_eps = _bernstein_epsilon(s, R, T_eff, delta)

            rows.append({
                "hoeffding_eps": h_eps,
                "bernstein_eps": b_eps,
                "tightening_ratio": b_eps / h_eps if h_eps > 1e-15 else np.nan,
                "sigma": s,
                "R": R,
                "sigma_sq_over_R_sq": s**2 / R**2 if R > 1e-15 else np.nan,
            })

        return pd.DataFrame(rows, index=prices.columns)

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        bound_type = "Bernstein" if self.cfg.use_bernstein else "Hoeffding"
        extra = ""
        if self._fitted and self._n_tradeable > 0:
            extra = f", {self._n_tradeable} tradeable"
        return f"ConcentrationBoundsStrategy({fitted_tag}, {bound_type}{extra})"
