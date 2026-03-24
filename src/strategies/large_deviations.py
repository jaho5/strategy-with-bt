"""Large Deviation Principles strategy for tail-probability anomaly detection.

Uses graduate-level probability theory (Cramér's theorem, Legendre-Fenchel
transforms) to identify market inefficiencies in the tails of return
distributions.

Mathematical foundation
-----------------------
**Cramér's theorem** states that for i.i.d. random variables X_1, ..., X_n
with mean mu, the probability that the sample mean deviates from mu decays
exponentially:

    P(S_n / n > x) ~ exp(-n * I(x))   as n -> inf

where I(x) is the *rate function* (or Cramér function), defined as the
Legendre-Fenchel transform of the cumulant generating function (CGF):

    I(x) = sup_theta { theta * x - Lambda(theta) }

with Lambda(theta) = log M(theta) = log E[exp(theta * X)] being the CGF
and M(theta) the moment generating function (MGF).

**Key insight**: The rate function I(x) governs how "difficult" it is for
the empirical average to reach level x.  If empirical tail probabilities
deviate significantly from the LDP prediction exp(-n * I(x)), the market
is mispricing tail risk.

Strategy
--------
1. **Rate function estimation** -- estimate the MGF from data, compute the
   CGF Lambda(theta), and numerically evaluate the Legendre-Fenchel
   transform to obtain I(x).  A polynomial fit to the CGF enables fast,
   stable Legendre transforms.

2. **Tail probability anomaly** -- compare empirical tail frequencies with
   the LDP prediction.  Large positive anomaly (empirical >> predicted)
   indicates underpriced tail risk; large negative anomaly indicates an
   unusually calm market.

3. **Trading signal** -- synthesise signals from:
   - Rate function curvature at the current return level (convexity = regime)
   - Tail anomaly direction and magnitude
   - Rate function shape changes over time (regime shift detection)

4. **Risk management** -- use I(x) to set dynamic stop-losses and position
   sizes.  Given a drawdown threshold d, the exceedance probability is
   approximately exp(-I(d) * T), allowing principled risk budgeting.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LargeDeviationsConfig:
    """Tuneable parameters for the large deviations strategy."""

    # Estimation windows
    lookback: int = 252             # primary estimation window (trading days)
    min_history: int = 60           # minimum observations before trading

    # MGF / CGF estimation
    theta_grid_points: int = 200    # number of grid points for theta
    theta_max: float = 5.0          # max |theta| for MGF estimation
    cgf_poly_degree: int = 6        # degree of polynomial CGF approximation

    # Rate function evaluation
    rate_grid_points: int = 100     # grid resolution for I(x)

    # Tail anomaly detection
    tail_quantile: float = 0.05    # quantile defining "tail" (5th/95th)
    anomaly_threshold: float = 1.5  # log-ratio threshold for anomaly signal
    rolling_tail_window: int = 63   # window for rolling tail frequency

    # Signal construction
    momentum_rate_threshold: float = 0.1   # I(x) below this -> easy to reach
    curvature_lookback: int = 63           # window for rate function shape change
    signal_ema_span: int = 10              # EMA smoothing for final signal

    # Position sizing
    max_leverage: float = 1.0       # gross leverage cap
    base_weight: float = 0.5        # base position weight
    fat_tail_shrink: float = 0.3    # shrink factor when tails are fat

    # Risk management via LDP
    stop_loss_prob: float = 0.01    # target exceedance probability for stop
    rebalance_freq: int = 21        # trading days between rebalances


# ---------------------------------------------------------------------------
# Moment generating function / cumulant generating function helpers
# ---------------------------------------------------------------------------

def _estimate_mgf(
    returns: np.ndarray,
    theta_grid: np.ndarray,
) -> np.ndarray:
    """Estimate the moment generating function M(theta) = E[exp(theta * r)].

    Uses the empirical average as an unbiased estimator of the expectation.

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Return observations.
    theta_grid : np.ndarray, shape (K,)
        Grid of theta values at which to evaluate the MGF.

    Returns
    -------
    np.ndarray, shape (K,)
        Estimated M(theta) for each theta in the grid.
    """
    # returns: (T,), theta_grid: (K,) -> outer product: (K, T)
    # M(theta) = (1/T) * sum_t exp(theta * r_t)
    exponents = np.outer(theta_grid, returns)  # (K, T)

    # Numerical stability: subtract row-wise max before exp
    row_max = exponents.max(axis=1, keepdims=True)
    mgf = np.exp(row_max.ravel()) * np.mean(np.exp(exponents - row_max), axis=1)

    return mgf


def _estimate_cgf(
    returns: np.ndarray,
    theta_grid: np.ndarray,
) -> np.ndarray:
    """Estimate the cumulant generating function Lambda(theta) = log M(theta).

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
        Return observations.
    theta_grid : np.ndarray, shape (K,)
        Grid of theta values.

    Returns
    -------
    np.ndarray, shape (K,)
        Estimated Lambda(theta).
    """
    mgf = _estimate_mgf(returns, theta_grid)
    # Clamp to avoid log(0) or log(negative) from numerical noise
    mgf = np.maximum(mgf, 1e-300)
    return np.log(mgf)


def _fit_cgf_polynomial(
    theta_grid: np.ndarray,
    cgf_values: np.ndarray,
    degree: int,
) -> np.poly1d:
    """Fit a polynomial approximation to the CGF.

    The polynomial Lambda_hat(theta) enables fast, smooth evaluation
    of the CGF and its derivative (needed for the Legendre transform).

    Parameters
    ----------
    theta_grid : np.ndarray
        Theta values.
    cgf_values : np.ndarray
        Corresponding CGF values.
    degree : int
        Polynomial degree (typically 4-8 for financial returns).

    Returns
    -------
    np.poly1d
        Fitted polynomial.
    """
    # Remove any NaN/inf values
    mask = np.isfinite(cgf_values)
    if mask.sum() < degree + 1:
        warnings.warn(
            "Too few finite CGF values for polynomial fit; "
            "falling back to quadratic.",
            stacklevel=2,
        )
        degree = min(2, mask.sum() - 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.RankWarning)
        coeffs = np.polyfit(theta_grid[mask], cgf_values[mask], degree)

    return np.poly1d(coeffs)


# ---------------------------------------------------------------------------
# Rate function (Legendre-Fenchel transform)
# ---------------------------------------------------------------------------

def _legendre_fenchel_transform(
    cgf_poly: np.poly1d,
    x: float,
    theta_bounds: Tuple[float, float] = (-10.0, 10.0),
) -> float:
    """Compute the rate function I(x) = sup_theta { theta*x - Lambda(theta) }.

    Uses numerical optimisation (bounded scalar minimisation of the negative
    objective) to evaluate the Legendre-Fenchel transform at a single point.

    Parameters
    ----------
    cgf_poly : np.poly1d
        Polynomial approximation of the CGF Lambda(theta).
    x : float
        Point at which to evaluate the rate function.
    theta_bounds : tuple
        Search bounds for theta.

    Returns
    -------
    float
        Rate function value I(x) >= 0.
    """
    # Minimise -f(theta) = -(theta*x - Lambda(theta))
    def neg_objective(theta: float) -> float:
        return -(theta * x - cgf_poly(theta))

    result = minimize_scalar(
        neg_objective,
        bounds=theta_bounds,
        method="bounded",
    )

    # I(x) = -result.fun, and by convexity I(x) >= 0
    rate = max(-result.fun, 0.0)
    return rate


def _compute_rate_function(
    cgf_poly: np.poly1d,
    x_grid: np.ndarray,
    theta_bounds: Tuple[float, float] = (-10.0, 10.0),
) -> np.ndarray:
    """Compute the rate function I(x) on a grid of x values.

    Parameters
    ----------
    cgf_poly : np.poly1d
        Polynomial CGF approximation.
    x_grid : np.ndarray, shape (M,)
        Grid of x values (return levels).
    theta_bounds : tuple
        Search bounds for the optimisation.

    Returns
    -------
    np.ndarray, shape (M,)
        Rate function values I(x) for each x in the grid.
    """
    rate_values = np.array([
        _legendre_fenchel_transform(cgf_poly, x, theta_bounds)
        for x in x_grid
    ])
    return rate_values


# ---------------------------------------------------------------------------
# Tail anomaly detection
# ---------------------------------------------------------------------------

def _compute_tail_anomaly(
    returns: np.ndarray,
    cgf_poly: np.poly1d,
    tail_quantile: float,
    n_effective: int,
    theta_bounds: Tuple[float, float] = (-10.0, 10.0),
) -> Tuple[float, float]:
    """Compare empirical tail probabilities with LDP predictions.

    For both the upper and lower tail, compute:
        anomaly = log(P_empirical) - log(P_ldp)
    where P_ldp = exp(-n * I(x_threshold)).

    Positive anomaly: empirical tails fatter than LDP predicts (tail risk
    underpriced).  Negative anomaly: thinner tails (calm market).

    Parameters
    ----------
    returns : np.ndarray
        Return observations.
    cgf_poly : np.poly1d
        Polynomial CGF approximation.
    tail_quantile : float
        Quantile defining the tail (e.g., 0.05 for 5th percentile).
    n_effective : int
        Effective sample size for LDP scaling.
    theta_bounds : tuple
        Search bounds for rate function computation.

    Returns
    -------
    tuple of (upper_anomaly, lower_anomaly)
        Log-ratio anomaly scores for upper and lower tails.
    """
    T = len(returns)
    if T < 10:
        return 0.0, 0.0

    mean_r = np.mean(returns)

    # Upper tail
    upper_threshold = np.quantile(returns, 1.0 - tail_quantile)
    p_emp_upper = np.mean(returns > upper_threshold)

    rate_upper = _legendre_fenchel_transform(
        cgf_poly, upper_threshold, theta_bounds,
    )
    # LDP prediction: P(r > x) ~ exp(-n * I(x))
    log_p_ldp_upper = -n_effective * rate_upper
    log_p_emp_upper = np.log(max(p_emp_upper, 1e-10))
    upper_anomaly = log_p_emp_upper - log_p_ldp_upper

    # Lower tail
    lower_threshold = np.quantile(returns, tail_quantile)
    p_emp_lower = np.mean(returns < lower_threshold)

    # For the lower tail, use rate function of negative deviation
    rate_lower = _legendre_fenchel_transform(
        cgf_poly, lower_threshold, theta_bounds,
    )
    log_p_ldp_lower = -n_effective * rate_lower
    log_p_emp_lower = np.log(max(p_emp_lower, 1e-10))
    lower_anomaly = log_p_emp_lower - log_p_ldp_lower

    return upper_anomaly, lower_anomaly


# ---------------------------------------------------------------------------
# Rate function curvature (regime detection)
# ---------------------------------------------------------------------------

def _rate_function_curvature(
    cgf_poly: np.poly1d,
    x: float,
    dx: float = 1e-4,
    theta_bounds: Tuple[float, float] = (-10.0, 10.0),
) -> float:
    """Estimate the second derivative of I(x) at a point (curvature).

    High curvature at x means the probability drops off sharply near x
    (thin tails / low risk).  Low curvature means probabilities change
    slowly (fat tails / high risk).

    Uses finite differences: I''(x) ~ [I(x+dx) - 2I(x) + I(x-dx)] / dx^2.

    Parameters
    ----------
    cgf_poly : np.poly1d
        Polynomial CGF approximation.
    x : float
        Point at which to evaluate curvature.
    dx : float
        Finite difference step size.
    theta_bounds : tuple
        Search bounds.

    Returns
    -------
    float
        Estimated curvature I''(x).
    """
    I_plus = _legendre_fenchel_transform(cgf_poly, x + dx, theta_bounds)
    I_center = _legendre_fenchel_transform(cgf_poly, x, theta_bounds)
    I_minus = _legendre_fenchel_transform(cgf_poly, x - dx, theta_bounds)

    curvature = (I_plus - 2.0 * I_center + I_minus) / (dx * dx)
    return curvature


# ---------------------------------------------------------------------------
# LDP-based stop loss
# ---------------------------------------------------------------------------

def _ldp_stop_loss_level(
    cgf_poly: np.poly1d,
    target_prob: float,
    horizon: int,
    mean_return: float,
    std_return: float,
    theta_bounds: Tuple[float, float] = (-10.0, 10.0),
) -> float:
    """Compute the stop-loss level d such that P(drawdown > d) ~ target_prob.

    From LDP: P(drawdown > d) ~ exp(-I(d) * T).
    Inverting: I(d) = -log(target_prob) / T.
    Then find d such that I(d) = target_rate.

    Parameters
    ----------
    cgf_poly : np.poly1d
        Polynomial CGF approximation.
    target_prob : float
        Target exceedance probability for the stop (e.g. 0.01).
    horizon : int
        Time horizon T in periods.
    mean_return : float
        Mean return (for grid centering).
    std_return : float
        Std of returns (for grid scaling).
    theta_bounds : tuple
        Search bounds.

    Returns
    -------
    float
        Stop-loss level d (as a negative return, i.e. d < 0 for losses).
    """
    if target_prob <= 0 or target_prob >= 1 or horizon <= 0:
        return -3.0 * std_return  # fallback

    target_rate = -np.log(target_prob) / horizon

    # Search for d < mean where I(d) = target_rate
    # I(x) is convex with minimum at x = mean, so for d < mean, I is decreasing
    # We search on the negative side
    search_min = mean_return - 6.0 * std_return
    search_max = mean_return

    def objective(d: float) -> float:
        rate = _legendre_fenchel_transform(cgf_poly, d, theta_bounds)
        return (rate - target_rate) ** 2

    result = minimize_scalar(
        objective,
        bounds=(search_min, search_max),
        method="bounded",
    )

    return result.x


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

class LargeDeviationsStrategy(Strategy):
    """Trading strategy based on Large Deviation Principles.

    Detects tail-probability anomalies by comparing empirical tail
    frequencies with predictions from Cramér's theorem.  Uses the
    rate function I(x) for dynamic risk management and position sizing.

    Parameters
    ----------
    config : LargeDeviationsConfig, optional
        Strategy configuration.  Uses defaults if not provided.
    """

    def __init__(
        self,
        config: Optional[LargeDeviationsConfig] = None,
    ) -> None:
        self.config = config or LargeDeviationsConfig()
        super().__init__(
            name="LargeDeviations",
            description=(
                "Tail-probability anomaly detection via Cramér's theorem "
                "and Legendre-Fenchel rate functions"
            ),
        )

        # Fitted state
        self._cgf_poly: Optional[np.poly1d] = None
        self._mean_return: float = 0.0
        self._std_return: float = 1.0
        self._rate_at_mean: float = 0.0
        self._theta_bounds: Tuple[float, float] = (
            -self.config.theta_max,
            self.config.theta_max,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "LargeDeviationsStrategy":
        """Calibrate the rate function from historical price data.

        Estimates the CGF from log-returns, fits a polynomial
        approximation, and pre-computes baseline rate function values.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data indexed by datetime.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        if prices.ndim == 2:
            series = prices.iloc[:, 0]
        else:
            series = prices

        log_returns = np.log(series / series.shift(1)).dropna().values

        if len(log_returns) < self.config.min_history:
            raise ValueError(
                f"Need at least {self.config.min_history} observations, "
                f"got {len(log_returns)}."
            )

        self._mean_return = float(np.mean(log_returns))
        self._std_return = float(np.std(log_returns))
        if self._std_return < 1e-10:
            self._std_return = 1e-10

        # Estimate CGF on theta grid
        theta_grid = np.linspace(
            -self.config.theta_max,
            self.config.theta_max,
            self.config.theta_grid_points,
        )
        cgf_values = _estimate_cgf(log_returns, theta_grid)

        # Polynomial fit
        self._cgf_poly = _fit_cgf_polynomial(
            theta_grid, cgf_values, self.config.cgf_poly_degree,
        )

        # Baseline: rate function at the mean (should be ~0)
        self._rate_at_mean = _legendre_fenchel_transform(
            self._cgf_poly, self._mean_return, self._theta_bounds,
        )

        # Store parameters for inspection
        self.parameters = {
            "mean_return": self._mean_return,
            "std_return": self._std_return,
            "cgf_poly_coefficients": self._cgf_poly.coefficients.tolist(),
            "rate_at_mean": self._rate_at_mean,
            "n_observations": len(log_returns),
        }

        self._fitted = True
        logger.info(
            "LargeDeviations fitted: mean=%.6f, std=%.6f, "
            "rate_at_mean=%.6f, n=%d",
            self._mean_return,
            self._std_return,
            self._rate_at_mean,
            len(log_returns),
        )

        return self

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals from LDP tail anomaly analysis.

        For each date (after the warm-up period), the strategy:
        1. Estimates the CGF from the trailing window of returns.
        2. Computes upper/lower tail anomalies vs LDP predictions.
        3. Evaluates the rate function at the current return level.
        4. Detects rate-function shape changes (regime shifts).
        5. Combines into a directional signal with LDP-based sizing.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data indexed by datetime.

        Returns
        -------
        pd.DataFrame
            Columns: ``signal`` (-1, 0, 1) and ``weight`` in [0, 1].
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        if prices.ndim == 2:
            series = prices.iloc[:, 0]
        else:
            series = prices

        log_returns = np.log(series / series.shift(1))

        n = len(series)
        signals = np.zeros(n)
        weights = np.zeros(n)

        cfg = self.config
        lookback = cfg.lookback
        min_hist = cfg.min_history
        theta_grid = np.linspace(
            -cfg.theta_max, cfg.theta_max, cfg.theta_grid_points,
        )

        # Track previous rate functions for shape-change detection
        prev_rate_values: Optional[np.ndarray] = None
        prev_curvature: Optional[float] = None

        for t in range(min_hist, n):
            start = max(0, t - lookback)
            window_returns = log_returns.iloc[start:t].dropna().values

            if len(window_returns) < min_hist:
                continue

            # --- Step 1: Estimate CGF and rate function for this window ---
            cgf_values = _estimate_cgf(window_returns, theta_grid)
            cgf_poly = _fit_cgf_polynomial(
                theta_grid, cgf_values, cfg.cgf_poly_degree,
            )

            mean_r = np.mean(window_returns)
            std_r = np.std(window_returns)
            if std_r < 1e-10:
                continue

            # --- Step 2: Tail anomaly ---
            n_effective = max(1, len(window_returns) // cfg.rolling_tail_window)
            upper_anomaly, lower_anomaly = _compute_tail_anomaly(
                window_returns,
                cgf_poly,
                cfg.tail_quantile,
                n_effective,
                self._theta_bounds,
            )

            # --- Step 3: Rate function at current return level ---
            current_return = log_returns.iloc[t]
            if np.isnan(current_return):
                continue

            rate_at_current = _legendre_fenchel_transform(
                cgf_poly, current_return, self._theta_bounds,
            )

            # --- Step 4: Curvature and shape change ---
            curvature = _rate_function_curvature(
                cgf_poly, mean_r, dx=std_r * 0.1, theta_bounds=self._theta_bounds,
            )

            # Rate function on a small grid for shape comparison
            x_grid = np.linspace(
                mean_r - 3 * std_r, mean_r + 3 * std_r, 20,
            )
            rate_values = _compute_rate_function(
                cgf_poly, x_grid, self._theta_bounds,
            )

            # Shape change: L2 distance between current and previous rate func
            shape_change = 0.0
            if prev_rate_values is not None and len(prev_rate_values) == len(rate_values):
                # Normalise by the sum to measure relative shape change
                norm_prev = prev_rate_values / (np.sum(prev_rate_values) + 1e-10)
                norm_curr = rate_values / (np.sum(rate_values) + 1e-10)
                shape_change = np.sqrt(np.mean((norm_curr - norm_prev) ** 2))

            prev_rate_values = rate_values.copy()

            # Curvature change
            curvature_change = 0.0
            if prev_curvature is not None and prev_curvature != 0:
                curvature_change = (curvature - prev_curvature) / (
                    abs(prev_curvature) + 1e-10
                )
            prev_curvature = curvature

            # --- Step 5: Combine into signal ---
            signal_score = 0.0

            # Component 1: Tail anomaly signal
            # Upper anomaly > 0 means fatter right tail than expected -> bullish
            # Lower anomaly > 0 means fatter left tail than expected -> bearish
            tail_signal = 0.0
            if abs(upper_anomaly) > cfg.anomaly_threshold:
                tail_signal += np.sign(upper_anomaly) * 0.4
            if abs(lower_anomaly) > cfg.anomaly_threshold:
                tail_signal -= np.sign(lower_anomaly) * 0.4

            signal_score += tail_signal

            # Component 2: Momentum via rate function
            # If rate is high at current level but price persists there,
            # that's unusual persistence -> momentum signal
            if rate_at_current > cfg.momentum_rate_threshold:
                # High rate = statistically unlikely level
                # If current return is positive, momentum is bullish
                momentum_signal = np.sign(current_return - mean_r) * 0.3
                signal_score += momentum_signal

            # Component 3: Regime shift via shape change
            if shape_change > 0.1:
                # Significant shape change detected
                # If curvature is decreasing (tails getting fatter), reduce
                if curvature_change < -0.1:
                    signal_score *= 0.5  # dampen signal during regime shift
                # Shape change itself is not directional, but large changes
                # suggest caution

            # --- Discretise signal ---
            if signal_score > 0.2:
                signals[t] = 1.0
            elif signal_score < -0.2:
                signals[t] = -1.0
            else:
                signals[t] = 0.0

            # --- Position sizing ---
            # Base weight, adjusted by tail fatness
            weight = cfg.base_weight

            # If tails are fatter than expected (either side), shrink position
            tail_fatness = max(upper_anomaly, lower_anomaly, 0.0)
            if tail_fatness > cfg.anomaly_threshold:
                shrink = cfg.fat_tail_shrink * min(
                    tail_fatness / (cfg.anomaly_threshold * 3), 1.0,
                )
                weight *= (1.0 - shrink)

            # Curvature-based adjustment: low curvature = fat tails = less weight
            if curvature > 0:
                # Higher curvature = thinner tails = more confident
                curvature_factor = min(curvature / (1.0 / (std_r ** 2) + 1e-10), 2.0)
                weight *= min(0.5 + 0.5 * curvature_factor, 1.0)

            # LDP-based stop loss level
            stop_level = _ldp_stop_loss_level(
                cgf_poly,
                cfg.stop_loss_prob,
                horizon=cfg.rebalance_freq,
                mean_return=mean_r,
                std_return=std_r,
                theta_bounds=self._theta_bounds,
            )

            # If current drawdown approaches stop level, reduce weight
            if current_return < stop_level:
                weight *= 0.25

            weights[t] = np.clip(weight, 0.0, 1.0)

        # --- Smooth signals ---
        signal_series = pd.Series(signals, index=series.index)
        weight_series = pd.Series(weights, index=series.index)

        # Smooth weights (not signals, to preserve discrete {-1,0,1})
        weight_series = self.exponential_smooth(weight_series, span=cfg.signal_ema_span)
        weight_series = weight_series.clip(0.0, 1.0)

        # Enforce leverage cap
        weight_series = weight_series.clip(upper=cfg.max_leverage)

        result = pd.DataFrame({
            "signal": signal_series,
            "weight": weight_series,
        }, index=series.index)

        return result

    # ------------------------------------------------------------------
    # Diagnostic methods
    # ------------------------------------------------------------------

    def compute_rate_function_grid(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        n_points: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the rate function I(x) on a grid for visualisation.

        Parameters
        ----------
        x_min, x_max : float, optional
            Grid bounds.  Defaults to mean +/- 4 std.
        n_points : int
            Number of grid points.

        Returns
        -------
        tuple of (x_grid, rate_values)
            Arrays of x values and corresponding I(x) values.
        """
        self.ensure_fitted()

        if x_min is None:
            x_min = self._mean_return - 4.0 * self._std_return
        if x_max is None:
            x_max = self._mean_return + 4.0 * self._std_return

        x_grid = np.linspace(x_min, x_max, n_points)
        rate_values = _compute_rate_function(
            self._cgf_poly, x_grid, self._theta_bounds,
        )

        return x_grid, rate_values

    def tail_exceedance_probability(
        self,
        level: float,
        horizon: int = 1,
    ) -> float:
        """Estimate tail exceedance probability using LDP.

        P(average return over horizon > level) ~ exp(-horizon * I(level))

        Parameters
        ----------
        level : float
            Return level.
        horizon : int
            Number of periods.

        Returns
        -------
        float
            Estimated exceedance probability.
        """
        self.ensure_fitted()

        rate = _legendre_fenchel_transform(
            self._cgf_poly, level, self._theta_bounds,
        )
        log_prob = -horizon * rate
        # Clamp to [0, 1]
        prob = np.clip(np.exp(log_prob), 0.0, 1.0)
        return float(prob)
