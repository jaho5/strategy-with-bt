"""JKO-inspired Wasserstein gradient flow portfolio rebalancing strategy.

Treats portfolio rebalancing as a discretised Wasserstein gradient flow
(Jordan-Kinderlehrer-Otto scheme).  At each rebalance step we minimise
a free energy functional that balances expected return, risk, diversification,
and rebalancing cost:

    F(w) = -mu'w + (gamma/2) w'Sigma w - lambda H(w) + (1/(2 tau)) ||w - w_prev||^2

where:
*   mu        -- estimated expected returns
*   Sigma     -- estimated covariance matrix
*   H(w)      -- Shannon entropy (diversification bonus)
*   tau       -- JKO time step (controls rebalancing speed)
*   w_prev    -- portfolio weights from the previous step

The quadratic proximal term (1/(2 tau)) ||w - w_prev||^2 is the discrete
analogue of the Wasserstein-2 distance in the JKO scheme and penalises
large weight changes, naturally controlling turnover.

References
----------
*   Jordan, Kinderlehrer, Otto (1998). The variational formulation of
    the Fokker-Planck equation. SIAM J. Math. Anal.
*   Ambrosio, Gigli, Savare (2008). Gradient flows in metric spaces and
    in the space of probability measures.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(w: np.ndarray) -> float:
    """H(w) = -sum w_i log(w_i), with 0 log 0 := 0."""
    w_safe = np.clip(w, 1e-15, None)
    return -float(np.sum(w_safe * np.log(w_safe)))


def _shrink_covariance(S: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Ledoit-Wolf-style constant-correlation shrinkage.

    Blends the sample covariance *S* toward a scaled identity matrix:
        Sigma_shrunk = (1 - shrinkage) * S + shrinkage * (trace(S)/n) * I
    """
    n = S.shape[0]
    mu = np.trace(S) / n
    return (1.0 - shrinkage) * S + shrinkage * mu * np.eye(n)


def _solve_jko_step(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    gamma: float,
    lam: float,
    tau: float,
) -> np.ndarray:
    """Solve one JKO proximal step on the probability simplex.

    Minimise:
        F(w) = -mu'w + (gamma/2) w'Sigma w - lam H(w) + (1/(2 tau)) ||w - w_prev||^2

    subject to  w >= 0,  sum(w) = 1.

    Parameters
    ----------
    mu : (N,) expected returns.
    Sigma : (N, N) covariance matrix.
    w_prev : (N,) previous portfolio weights.
    gamma : risk-aversion parameter.
    lam : entropy regularisation strength.
    tau : JKO time-step (larger = slower rebalancing).

    Returns
    -------
    w : (N,) optimal weights on the simplex.
    """
    n = len(mu)
    if n == 0:
        return np.array([])

    inv_tau = 1.0 / max(tau, 1e-12)

    def objective(w: np.ndarray) -> float:
        ret = mu @ w
        risk = 0.5 * gamma * (w @ Sigma @ w)
        entropy = _shannon_entropy(w)
        proximal = 0.5 * inv_tau * np.sum((w - w_prev) ** 2)
        return -ret + risk - lam * entropy + proximal

    def gradient(w: np.ndarray) -> np.ndarray:
        grad_ret = -mu
        grad_risk = gamma * Sigma @ w
        w_safe = np.clip(w, 1e-15, None)
        grad_entropy = -(1.0 + np.log(w_safe))  # -dH/dw
        grad_prox = inv_tau * (w - w_prev)
        return grad_ret + grad_risk - lam * (-grad_entropy) + grad_prox

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    eps = 1e-10
    bounds = [(eps, 1.0)] * n
    w0 = w_prev.copy()

    result = minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    w_opt = result.x
    # Project back to simplex (numerical safety)
    w_opt = np.maximum(w_opt, 0.0)
    total = w_opt.sum()
    if total > 0:
        w_opt /= total
    else:
        w_opt = np.full(n, 1.0 / n)

    return w_opt


# ===========================================================================
# Strategy class
# ===========================================================================

class WassersteinGradientStrategy(Strategy):
    """JKO-inspired Wasserstein gradient flow portfolio rebalancing.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient for the quadratic risk term.  Default 1.0.
    lam : float
        Entropy regularisation strength.  Larger values push the portfolio
        toward equal-weight (maximum entropy).  Default 0.1.
    tau : float
        JKO time-step controlling rebalancing speed.  Larger values penalise
        weight changes more heavily, producing smoother rebalancing paths.
        Default 5.0.
    lookback : int
        Rolling window (trading days) for estimating mu and Sigma.
        Default 60.
    rebalance_freq : int
        Rebalance every *rebalance_freq* trading days.  Default 5 (weekly).
    shrinkage : float
        Covariance shrinkage intensity in [0, 1].  Default 0.1.
    min_history : int
        Minimum number of return observations required before optimising.
        Default 30.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        lam: float = 0.1,
        tau: float = 5.0,
        lookback: int = 60,
        rebalance_freq: int = 5,
        shrinkage: float = 0.1,
        min_history: int = 30,
    ) -> None:
        super().__init__(
            name="WassersteinGradient",
            description=(
                "JKO-inspired Wasserstein gradient flow portfolio rebalancing "
                "with entropy regularisation and proximal turnover control."
            ),
        )
        self.gamma = gamma
        self.lam = lam
        self.tau = tau
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.shrinkage = shrinkage
        self.min_history = min_history

        # Learned parameters (populated by fit)
        self._mu_hat: Optional[np.ndarray] = None
        self._Sigma_hat: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "WassersteinGradientStrategy":
        """Calibrate expected returns and covariance from historical prices.

        Parameters
        ----------
        prices : pd.DataFrame
            Close prices, columns = tickers, index = dates.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        # Forward-fill then drop any remaining NaN rows
        prices_clean = prices.ffill().dropna()
        if len(prices_clean) < self.min_history:
            warnings.warn(
                f"Only {len(prices_clean)} clean observations; need at least "
                f"{self.min_history}. Strategy will use equal weights.",
                stacklevel=2,
            )
            n = prices.shape[1]
            self._mu_hat = np.zeros(n)
            self._Sigma_hat = np.eye(n)
            self._fitted = True
            return self

        log_returns = np.log(prices_clean / prices_clean.shift(1)).dropna()
        window = log_returns.iloc[-self.lookback:]

        self._mu_hat = window.mean().values.astype(np.float64)
        S = window.cov().values.astype(np.float64)
        self._Sigma_hat = _shrink_covariance(S, self.shrinkage)

        self.parameters = {
            "mu_hat": self._mu_hat.tolist(),
            "Sigma_hat_trace": float(np.trace(self._Sigma_hat)),
        }

        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate portfolio allocation signals via JKO gradient flow.

        Parameters
        ----------
        prices : pd.DataFrame
            Close prices, columns = tickers, index = dates.

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` (+1, long-only) and
            ``{ticker}_weight`` (allocation fraction) for each ticker.
        """
        self.validate_prices(prices)

        tickers = list(prices.columns)
        n_assets = len(tickers)
        n_dates = len(prices)

        # Forward-fill for clean return computation
        prices_clean = prices.ffill()
        log_returns = np.log(prices_clean / prices_clean.shift(1))
        # Replace NaN / inf with 0
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Initialise weights to equal-weight
        w_prev = np.full(n_assets, 1.0 / n_assets)
        all_weights = np.zeros((n_dates, n_assets))

        for t in range(n_dates):
            if t < self.min_history:
                all_weights[t] = w_prev
                continue

            # Rebalance only at specified frequency
            if t % self.rebalance_freq != 0:
                all_weights[t] = w_prev
                continue

            # Rolling estimates
            start = max(0, t - self.lookback)
            window_ret = log_returns.iloc[start:t]

            if len(window_ret) < self.min_history:
                all_weights[t] = w_prev
                continue

            mu_t = window_ret.mean().values.astype(np.float64)
            S_t = window_ret.cov().values.astype(np.float64)

            # Handle degenerate covariance
            if np.any(np.isnan(S_t)) or np.any(np.isinf(S_t)):
                all_weights[t] = w_prev
                continue

            Sigma_t = _shrink_covariance(S_t, self.shrinkage)

            # Solve JKO proximal step
            w_new = _solve_jko_step(
                mu=mu_t,
                Sigma=Sigma_t,
                w_prev=w_prev,
                gamma=self.gamma,
                lam=self.lam,
                tau=self.tau,
            )

            all_weights[t] = w_new
            w_prev = w_new

        # Build output DataFrame
        signals_df = pd.DataFrame(index=prices.index)
        for i, ticker in enumerate(tickers):
            signals_df[f"{ticker}_signal"] = 1  # long-only
            signals_df[f"{ticker}_weight"] = all_weights[:, i]

        return signals_df
