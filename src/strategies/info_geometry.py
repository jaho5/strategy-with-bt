"""Information-geometric trading strategy.

Uses Fisher information, KL divergence, and Fisher-Rao geodesic distance on
the Gaussian statistical manifold for regime detection and position sizing.

Mathematical background
-----------------------
We model daily returns as draws from a time-varying Gaussian
r_t ~ N(mu_t, sigma_t^2).  The pair (mu, sigma) parameterises a point on the
*Gaussian statistical manifold*, a 2-D Riemannian manifold whose metric is
the Fisher information matrix

    I(mu, sigma) = diag(1/sigma^2,  2/sigma^2)

(using the (mu, sigma) parameterisation -- note sigma, not sigma^2).

The geodesic (Fisher-Rao) distance between two Gaussians N(mu1, sigma1^2) and
N(mu2, sigma2^2) on this manifold is

    d_FR = sqrt(2) * arccosh(1 + (mu1 - mu2)^2 / (2 * sigma1 * sigma2)
                              + (sigma1 - sigma2)^2 / (2 * sigma1 * sigma2))

    (Atkinson & Mitchell 1981, simplified for the univariate case.)

We also track the forward and backward KL divergences between a *recent*
window and a *historical* window to detect distributional shifts.

Trading rules
~~~~~~~~~~~~~
1. Large Fisher-Rao distance from the historical baseline signals a regime
   change -- we reduce position size.
2. Favourable KL shift (rising mean, stable vol) → long; unfavourable → short.
3. Position magnitude is scaled by an *information quality* factor derived from
   det(I(theta)).  High Fisher information → parameters are well-estimated →
   we take larger positions.
4. A natural-gradient step is used each bar to smooth position adjustments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.base import Strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ewm_mean_var(
    returns: np.ndarray, halflife: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exponentially weighted mean and variance (element-wise).

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
    halflife : int

    Returns
    -------
    mu : np.ndarray, shape (T,)
    var : np.ndarray, shape (T,)
    """
    alpha = 1 - np.exp(-np.log(2) / halflife)
    mu = np.empty_like(returns, dtype=np.float64)
    var = np.empty_like(returns, dtype=np.float64)
    mu[0] = returns[0]
    var[0] = 0.0
    for t in range(1, len(returns)):
        diff = returns[t] - mu[t - 1]
        mu[t] = mu[t - 1] + alpha * diff
        var[t] = (1 - alpha) * (var[t - 1] + alpha * diff * diff)
    return mu, var


def _kl_divergence_gaussian(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> float:
    r"""KL(N(mu1,sigma1^2) || N(mu2,sigma2^2)).

    D_KL = log(sigma2/sigma1) + (sigma1^2 + (mu1 - mu2)^2) / (2*sigma2^2) - 0.5
    """
    if sigma1 <= 0 or sigma2 <= 0:
        return 0.0
    return (
        np.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
        - 0.5
    )


def _fisher_rao_distance(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> float:
    """Geodesic distance on the univariate Gaussian statistical manifold.

    Uses the (mu, sigma) parameterisation.  The Fisher-Rao metric for the
    univariate Gaussian family is

        ds^2 = (1/sigma^2) dmu^2 + (2/sigma^2) dsigma^2

    The closed-form geodesic distance (Atkinson & Mitchell 1981) is

        d = sqrt(2) * arccosh(1 + delta)

    where delta = [(mu1-mu2)^2 + 2*(sigma1-sigma2)^2] / (2*sigma1*sigma2).
    """
    if sigma1 <= 0 or sigma2 <= 0:
        return 0.0
    delta = ((mu1 - mu2) ** 2 + 2 * (sigma1 - sigma2) ** 2) / (
        2 * sigma1 * sigma2
    )
    # arccosh(1 + delta) = log(1 + delta + sqrt(delta*(delta+2)))
    # numerically more stable than np.arccosh for small delta
    arg = 1.0 + delta
    if arg < 1.0:
        return 0.0
    return float(np.sqrt(2) * np.arccosh(arg))


def _fisher_info_det_gaussian(sigma: float) -> float:
    """Determinant of the Fisher information matrix for N(mu, sigma^2).

    I(mu, sigma) = diag(1/sigma^2, 2/sigma^2)  →  det = 2 / sigma^4
    """
    if sigma <= 0:
        return 0.0
    return 2.0 / sigma**4


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class InformationGeometryStrategy(Strategy):
    """Trading strategy grounded in information geometry.

    Parameters
    ----------
    halflife : int
        Half-life (in trading days) for the exponentially weighted moment
        estimates of the return distribution.  Default 21 (~1 month).
    historical_window : int
        Number of bars used to define the *historical baseline* distribution.
        Default 63 (~3 months).
    regime_threshold : float
        Fisher-Rao distance above which we declare a regime change and scale
        down positions.  Default 1.0.
    learning_rate : float
        Step size for the natural-gradient position update.  Default 0.1.
    max_position : float
        Maximum absolute position size (signal clamp).  Default 1.0.
    """

    def __init__(
        self,
        halflife: int = 21,
        historical_window: int = 63,
        regime_threshold: float = 1.0,
        learning_rate: float = 0.1,
        max_position: float = 1.0,
    ) -> None:
        self.halflife = halflife
        self.historical_window = historical_window
        self.regime_threshold = regime_threshold
        self.learning_rate = learning_rate
        self.max_position = max_position

        # Populated by fit()
        self._baseline_mu: dict[str, float] = {}
        self._baseline_sigma: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "InformationGeometryStrategy":
        """Calibrate baseline distributional parameters from historical data.

        Computes the mean and standard deviation of daily returns over the
        last ``self.historical_window`` bars for every column in *data*.
        These serve as the *historical baseline* on the statistical manifold.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price data (DatetimeIndex, columns = tickers).

        Returns
        -------
        self
        """
        returns = data.pct_change().dropna()
        tail = returns.iloc[-self.historical_window :]
        for col in returns.columns:
            self._baseline_mu[col] = float(tail[col].mean())
            self._baseline_sigma[col] = float(tail[col].std(ddof=1))
            # Guard against zero vol
            if self._baseline_sigma[col] < 1e-10:
                self._baseline_sigma[col] = 1e-10
        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate position signals via information-geometric analysis.

        For each asset and each bar the method:

        1. Maintains exponentially weighted estimates of (mu_t, sigma_t).
        2. Computes the Fisher-Rao geodesic distance from the fitted
           baseline to the current point on the Gaussian manifold.
        3. Computes forward and backward KL divergences between recent and
           baseline distributions.
        4. Derives a raw directional signal from the KL asymmetry and the
           sign of the mean shift.
        5. Scales the signal by an *information quality* factor proportional
           to sqrt(det(I(theta_t))) normalised by the baseline, and damped
           when the Fisher-Rao distance exceeds the regime threshold.
        6. Smooths position changes via a natural-gradient update step.

        Parameters
        ----------
        data : pd.DataFrame
            Price data (DatetimeIndex, columns = tickers).

        Returns
        -------
        pd.DataFrame
            Position signals in [-max_position, max_position].
        """
        returns = data.pct_change()
        signals = pd.DataFrame(
            np.zeros_like(returns.values),
            index=returns.index,
            columns=returns.columns,
        )

        for col in returns.columns:
            ret = returns[col].values.astype(np.float64)
            sig = self._generate_signal_for_asset(col, ret)
            signals[col] = sig

        return signals

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_signal_for_asset(
        self, col: str, returns: np.ndarray
    ) -> np.ndarray:
        """Compute the full signal series for a single asset."""
        T = len(returns)
        signal = np.zeros(T, dtype=np.float64)

        # Fallback baseline if fit() was not called for this asset
        base_mu = self._baseline_mu.get(col, 0.0)
        base_sigma = self._baseline_sigma.get(col, 1e-4)

        # EWM moments ------------------------------------------------
        # Skip first element (NaN from pct_change)
        clean = np.copy(returns)
        clean[np.isnan(clean)] = 0.0
        ewm_mu, ewm_var = _ewm_mean_var(clean, self.halflife)
        ewm_sigma = np.sqrt(np.maximum(ewm_var, 1e-20))

        # Baseline Fisher info determinant (for normalisation)
        base_info_det = _fisher_info_det_gaussian(base_sigma)
        if base_info_det < 1e-30:
            base_info_det = 1e-30

        position = 0.0

        # Need a short burn-in so the EWM estimates are meaningful
        burn_in = max(self.halflife, 2)

        for t in range(burn_in, T):
            mu_t = ewm_mu[t]
            sigma_t = ewm_sigma[t]
            if sigma_t < 1e-10:
                sigma_t = 1e-10

            # 1. Fisher-Rao distance from baseline -------------------
            fr_dist = _fisher_rao_distance(mu_t, sigma_t, base_mu, base_sigma)

            # 2. KL divergences --------------------------------------
            kl_forward = _kl_divergence_gaussian(
                mu_t, sigma_t, base_mu, base_sigma
            )  # D_KL(recent || historical)
            kl_backward = _kl_divergence_gaussian(
                base_mu, base_sigma, mu_t, sigma_t
            )  # D_KL(historical || recent)
            kl_asymmetry = abs(kl_forward - kl_backward)

            # 3. Raw directional signal ------------------------------
            # Positive mean shift → bullish; negative → bearish.
            # Scale by the average KL magnitude so that larger
            # distributional shifts produce stronger signals.
            mean_shift = mu_t - base_mu
            kl_magnitude = 0.5 * (kl_forward + kl_backward)
            # tanh to bound in (-1, 1)
            raw_signal = float(
                np.tanh(mean_shift / (sigma_t + 1e-10))
                * np.tanh(kl_magnitude)
            )

            # 4. Regime-change damping -------------------------------
            if fr_dist > self.regime_threshold:
                # Exponential decay beyond threshold
                regime_scale = float(
                    np.exp(-(fr_dist - self.regime_threshold))
                )
            else:
                regime_scale = 1.0

            # 5. Information-quality sizing ---------------------------
            current_info_det = _fisher_info_det_gaussian(sigma_t)
            info_quality = float(
                np.sqrt(current_info_det / base_info_det)
            )
            # Clamp so it doesn't blow up when sigma_t << base_sigma
            info_quality = min(info_quality, 2.0)

            # 6. Target position -------------------------------------
            target = raw_signal * regime_scale * info_quality

            # 7. Natural-gradient update -----------------------------
            # The natural gradient replaces the Euclidean step
            #   delta_pos = eta * grad
            # with
            #   delta_pos = eta * I^{-1} * grad
            # For a scalar position, I^{-1} ~ sigma_t^2 (inverse Fisher
            # information in the mean direction).  We normalise by
            # base_sigma^2 to keep the step size unit-free.
            natural_lr = self.learning_rate * (sigma_t / base_sigma) ** 2
            natural_lr = min(natural_lr, 1.0)  # cap at full adjustment

            position = position + natural_lr * (target - position)

            # Clamp
            position = float(
                np.clip(position, -self.max_position, self.max_position)
            )
            signal[t] = position

        return signal
