"""Entropy-regularized portfolio optimization strategy.

Combines online portfolio selection (exponentiated gradient) with
entropy-regularized mean-variance optimization to produce robust,
diversified, long-only portfolios with theoretical no-regret guarantees.

Mathematical foundation
-----------------------
We maximise the entropy-regularized objective:

    max_w  { mu'w  -  (gamma/2) w'Sigma w  +  lambda H(w) }

subject to  w >= 0,  sum(w) = 1   (probability simplex)

where H(w) = -sum_i w_i log(w_i) is the Shannon entropy.  The entropy
term penalises concentration and makes the solution robust to estimation
error in mu and Sigma.

The online component uses the *exponentiated gradient* (EG) algorithm
from online learning theory:

    w_{t+1,i}  =  w_{t,i} exp(eta r_{t,i})  /  Z_t

where Z_t is the normalisation constant.  EG is the optimal algorithm
for portfolio selection on the simplex, achieving cumulative regret

    R_T  <=  O(sqrt(T log N))

against the best *fixed* portfolio chosen in hindsight.

The final portfolio blends the EG adaptive signal with the batch
entropy-regularised mean-variance solution, rebalancing weekly.

References
----------
*   Cover, T. (1991). Universal portfolios. Mathematical Finance 1(1).
*   Helmbold et al. (1998). On-line portfolio selection using
    multiplicative updates. Mathematical Finance 8(4).
*   Boyd & Vandenberghe (2004). Convex Optimization, Ch. 7.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: Shannon entropy and gradient
# ---------------------------------------------------------------------------

def _shannon_entropy(w: np.ndarray) -> float:
    """Compute H(w) = -sum_i w_i log(w_i), with 0 log 0 := 0."""
    w_safe = np.clip(w, 1e-15, None)
    return -float(np.sum(w_safe * np.log(w_safe)))


def _entropy_gradient(w: np.ndarray) -> np.ndarray:
    """Gradient of H(w) w.r.t. w:  dH/dw_i = -(1 + log(w_i))."""
    w_safe = np.clip(w, 1e-15, None)
    return -(1.0 + np.log(w_safe))


# ---------------------------------------------------------------------------
# Helper: project onto the probability simplex
# ---------------------------------------------------------------------------

def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex {w >= 0, sum(w) = 1}.

    Uses the O(n log n) algorithm from Duchi et al. (2008),
    'Efficient Projections onto the L1-Ball for Learning in High Dimensions'.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0]
    if len(rho) == 0:
        # Fallback: uniform
        return np.full(n, 1.0 / n)
    rho_max = rho[-1]
    theta = cssv[rho_max] / (rho_max + 1.0)
    return np.maximum(v - theta, 0.0)


# ---------------------------------------------------------------------------
# Core: entropy-regularised mean-variance solver
# ---------------------------------------------------------------------------

def _solve_entropy_regularized_mv(
    mu: np.ndarray,
    Sigma: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Solve: max_w { mu'w - (gamma/2) w'Sigma w + lam H(w) }  s.t. simplex.

    Parameters
    ----------
    mu : (N,) expected returns.
    Sigma : (N, N) covariance matrix.
    gamma : risk aversion parameter.
    lam : entropy regularisation strength.

    Returns
    -------
    w : (N,) optimal portfolio weights on the simplex.
    """
    n = len(mu)
    if n == 0:
        return np.array([])

    # We minimise the negative of the objective.
    def neg_objective(w: np.ndarray) -> float:
        port_return = mu @ w
        port_risk = 0.5 * gamma * w @ Sigma @ w
        entropy = _shannon_entropy(w)
        return -(port_return - port_risk + lam * entropy)

    def neg_gradient(w: np.ndarray) -> np.ndarray:
        grad_return = mu
        grad_risk = gamma * Sigma @ w
        grad_entropy = _entropy_gradient(w)
        # Objective gradient: mu - gamma Sigma w + lam * (-1 - log w)
        # Negative for minimisation:
        return -(grad_return - grad_risk + lam * (-grad_entropy))

    # Constraints: sum(w) = 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Bounds: w_i >= small epsilon (strict positivity for log)
    eps = 1e-10
    bounds = [(eps, 1.0)] * n

    # Initial point: uniform
    w0 = np.full(n, 1.0 / n)

    result = minimize(
        neg_objective,
        w0,
        jac=neg_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if result.success:
        w = np.maximum(result.x, 0.0)
        w /= w.sum()
        return w

    # Fallback: try without gradient (more robust for ill-conditioned cases)
    result2 = minimize(
        neg_objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if result2.success:
        w = np.maximum(result2.x, 0.0)
        w /= w.sum()
        return w

    logger.warning(
        "Entropy-regularised optimisation did not converge: %s. "
        "Falling back to uniform weights.",
        result2.message,
    )
    return np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# Core: adaptive lambda from covariance condition number
# ---------------------------------------------------------------------------

def _adaptive_lambda(
    Sigma: np.ndarray,
    lam_base: float,
    lam_min: float = 0.001,
    lam_max: float = 1.0,
) -> float:
    """Set entropy regularisation strength adaptively.

    When the covariance matrix is ill-conditioned (high condition number),
    the mean-variance solution is unstable -- so we increase lambda to
    pull weights towards uniform.  When Sigma is well-conditioned, we
    use a smaller lambda to allow concentration on high-Sharpe assets.

    lambda = lam_base * log(1 + kappa(Sigma))

    where kappa is the condition number (ratio of largest to smallest
    eigenvalue).
    """
    try:
        eigvals = np.linalg.eigvalsh(Sigma)
        eigvals = eigvals[eigvals > 0]
        if len(eigvals) < 2:
            return lam_base
        kappa = eigvals[-1] / eigvals[0]
        lam = lam_base * np.log1p(kappa)
    except np.linalg.LinAlgError:
        lam = lam_base

    return float(np.clip(lam, lam_min, lam_max))


# ---------------------------------------------------------------------------
# Core: exponentiated gradient update
# ---------------------------------------------------------------------------

def _eg_update(
    w: np.ndarray,
    returns: np.ndarray,
    eta: float,
) -> np.ndarray:
    """One step of the exponentiated gradient algorithm.

    w_{t+1,i} = w_{t,i} exp(eta * r_{t,i}) / Z

    Parameters
    ----------
    w : (N,) current portfolio weights (on simplex).
    returns : (N,) most recent period returns for each asset.
    eta : learning rate.

    Returns
    -------
    w_new : (N,) updated weights (on simplex).
    """
    # Replace NaN returns with 0 (no update for missing assets)
    r = np.nan_to_num(returns, nan=0.0)

    # Multiplicative update
    log_w = np.log(np.clip(w, 1e-15, None)) + eta * r
    # Subtract max for numerical stability (log-sum-exp trick)
    log_w -= log_w.max()
    w_new = np.exp(log_w)

    total = w_new.sum()
    if total > 0:
        w_new /= total
    else:
        w_new = np.full_like(w, 1.0 / len(w))

    return w_new


def _adagrad_lr(
    eta0: float,
    sum_sq_returns: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """AdaGrad-style adaptive learning rate: eta_t = eta0 / sqrt(sum r^2).

    Uses the average accumulated squared return across assets for a
    single scalar learning rate.
    """
    avg_sum_sq = np.mean(sum_sq_returns)
    return eta0 / (np.sqrt(avg_sum_sq) + epsilon)


# ===========================================================================
# Strategy class
# ===========================================================================

class EntropyRegularizedStrategy(Strategy):
    """Entropy-regularised portfolio optimisation strategy.

    Blends two complementary approaches:

    1. **Exponentiated Gradient (EG)** -- an online, no-regret algorithm
       that adapts weights multiplicatively based on realised returns.
       Provides adaptive momentum and guarantees sub-linear cumulative
       regret O(sqrt(T log N)).

    2. **Entropy-regularised mean-variance** -- a batch convex
       optimisation that maximises expected return minus risk, regularised
       by Shannon entropy to prevent over-concentration.  The entropy
       penalty lambda is set adaptively from the condition number of the
       covariance matrix.

    The final portfolio is a convex combination of the two, rebalanced
    at a configurable frequency.

    Parameters
    ----------
    gamma : float
        Risk aversion coefficient for the mean-variance term.
        Higher values penalise variance more.  Default 1.0.
    lambda_base : float
        Base entropy regularisation strength.  Adaptively scaled by
        the covariance condition number.  Default 0.05.
    eta0 : float
        Initial learning rate for the EG algorithm.  Adapted via
        AdaGrad.  Default 0.5.
    eg_blend : float
        Weight on the EG portfolio in the final blend.  The
        entropy-regularised MV portfolio receives weight (1 - eg_blend).
        Default 0.5.
    lookback : int
        Number of trading days used to estimate mu and Sigma for the
        batch optimisation.  Default 63 (~3 months).
    rebalance_freq : int
        Rebalance every *rebalance_freq* trading days.  Default 5
        (weekly).
    min_history : int
        Minimum number of bars required before generating non-trivial
        signals.  Default 63.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        lambda_base: float = 0.05,
        eta0: float = 0.5,
        eg_blend: float = 0.5,
        lookback: int = 63,
        rebalance_freq: int = 5,
        min_history: int = 63,
    ) -> None:
        super().__init__(
            name="EntropyRegularized",
            description=(
                "Entropy-regularised portfolio optimisation blending "
                "exponentiated gradient (online) with entropy-penalised "
                "mean-variance (batch)."
            ),
        )
        self.gamma = gamma
        self.lambda_base = lambda_base
        self.eta0 = eta0
        self.eg_blend = eg_blend
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.min_history = min_history

        # Populated by fit()
        self._mu: Optional[np.ndarray] = None
        self._Sigma: Optional[np.ndarray] = None
        self._asset_names: Optional[list] = None

        # Regret tracking
        self._cumulative_regret: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "EntropyRegularizedStrategy":
        """Calibrate baseline estimates of expected returns and covariance.

        Stores the sample mean and covariance of daily log returns over the
        last ``self.lookback`` bars.  These serve as the initial estimates
        for the entropy-regularised optimisation (they are refreshed on
        each rebalance date in ``generate_signals``).

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
        tail = returns.iloc[-self.lookback :]

        self._asset_names = list(prices.columns)
        self._mu = tail.mean().values.astype(np.float64)
        self._Sigma = tail.cov().values.astype(np.float64)

        # Regularise Sigma to ensure positive-definiteness
        self._Sigma = self._regularise_covariance(self._Sigma)

        self.parameters = {
            "gamma": self.gamma,
            "lambda_base": self.lambda_base,
            "eta0": self.eta0,
            "eg_blend": self.eg_blend,
            "lookback": self.lookback,
            "rebalance_freq": self.rebalance_freq,
        }

        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate portfolio allocation signals.

        For each rebalance date:
        1. Update EG weights from realised returns (online).
        2. Re-estimate mu, Sigma and solve entropy-regularised MV (batch).
        3. Blend the two portfolios.
        4. Track cumulative regret against the best hindsight portfolio.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (DatetimeIndex, columns = tickers).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for each
            asset.  Signals are always +1 (long-only); weights encode
            the portfolio allocation.
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        returns = np.log(prices / prices.shift(1))
        n_assets = len(prices.columns)
        n_dates = len(prices)
        tickers = list(prices.columns)

        # Initialise EG weights to uniform
        eg_weights = np.full(n_assets, 1.0 / n_assets)

        # AdaGrad accumulator for learning rate adaptation
        sum_sq_returns = np.full(n_assets, 1e-8)

        # Storage for blended weights at each time step
        all_weights = np.zeros((n_dates, n_assets))

        # Track portfolio log-returns for regret computation
        portfolio_log_returns = np.zeros(n_dates)
        asset_cum_log_returns = np.zeros((n_dates, n_assets))

        # Current blended weights (held between rebalances)
        current_weights = np.full(n_assets, 1.0 / n_assets)

        for t in range(n_dates):
            r_t = returns.iloc[t].values.astype(np.float64)
            r_t = np.nan_to_num(r_t, nan=0.0)

            if t < self.min_history:
                # Not enough history -- stay uniform
                all_weights[t] = current_weights
                portfolio_log_returns[t] = current_weights @ r_t
                if t > 0:
                    asset_cum_log_returns[t] = (
                        asset_cum_log_returns[t - 1] + r_t
                    )
                else:
                    asset_cum_log_returns[t] = r_t
                continue

            # --- Online: EG update ---
            # Adapt learning rate via AdaGrad
            sum_sq_returns += r_t ** 2
            eta = _adagrad_lr(self.eta0, sum_sq_returns)
            eg_weights = _eg_update(eg_weights, r_t, eta)

            # --- Batch: entropy-regularised MV (at rebalance dates) ---
            if t % self.rebalance_freq == 0:
                # Estimate mu and Sigma from trailing window
                start = max(0, t - self.lookback)
                window_returns = returns.iloc[start:t].dropna()

                if len(window_returns) >= 20:
                    mu_hat = window_returns.mean().values.astype(np.float64)
                    Sigma_hat = window_returns.cov().values.astype(np.float64)
                    Sigma_hat = self._regularise_covariance(Sigma_hat)

                    # Adaptive lambda from condition number
                    lam = _adaptive_lambda(Sigma_hat, self.lambda_base)

                    # Solve entropy-regularised MV
                    mv_weights = _solve_entropy_regularized_mv(
                        mu_hat, Sigma_hat, self.gamma, lam
                    )
                else:
                    mv_weights = np.full(n_assets, 1.0 / n_assets)

                # Blend EG and MV portfolios
                current_weights = (
                    self.eg_blend * eg_weights
                    + (1.0 - self.eg_blend) * mv_weights
                )
                # Re-normalise (convex combination is already normalised,
                # but guard against numerical drift)
                current_weights = np.maximum(current_weights, 0.0)
                total = current_weights.sum()
                if total > 0:
                    current_weights /= total
                else:
                    current_weights = np.full(n_assets, 1.0 / n_assets)

            all_weights[t] = current_weights

            # Track returns for regret computation
            portfolio_log_returns[t] = current_weights @ r_t
            if t > 0:
                asset_cum_log_returns[t] = (
                    asset_cum_log_returns[t - 1] + r_t
                )
            else:
                asset_cum_log_returns[t] = r_t

        # --- Compute cumulative regret ---
        cum_portfolio_return = np.cumsum(portfolio_log_returns)
        best_asset_cum_return = np.max(asset_cum_log_returns, axis=1)
        cumulative_regret = best_asset_cum_return - cum_portfolio_return
        self._cumulative_regret = pd.Series(
            cumulative_regret,
            index=prices.index,
            name="cumulative_regret",
        )

        # --- Build output DataFrame ---
        signals_df = pd.DataFrame(index=prices.index)
        for i, ticker in enumerate(tickers):
            # Long-only strategy: signal is always +1, weight encodes allocation
            signals_df[f"{ticker}_signal"] = 1
            signals_df[f"{ticker}_weight"] = all_weights[:, i]

        return signals_df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_regret(self) -> Optional[pd.Series]:
        """Return the cumulative regret series from the last signal generation.

        Regret is defined as:
            R_T = max_i { sum_{t=1}^T r_{t,i} } - sum_{t=1}^T w_t' r_t

        i.e. the difference between the cumulative log-return of the best
        single asset in hindsight and the cumulative log-return of the
        strategy's portfolio.

        The EG algorithm guarantees R_T <= O(sqrt(T log N)).

        Returns
        -------
        pd.Series or None
            Cumulative regret indexed by date, or None if
            ``generate_signals`` has not been called.
        """
        return self._cumulative_regret

    def get_regret_bound(self, T: int, N: int) -> float:
        """Theoretical upper bound on cumulative regret for EG.

        R_T <= sqrt(2 T ln(N)) / eta

        For the AdaGrad-adapted learning rate the effective bound is
        approximately sqrt(T log N).

        Parameters
        ----------
        T : int
            Number of time steps (trading days).
        N : int
            Number of assets.

        Returns
        -------
        float
            Theoretical regret bound.
        """
        return float(np.sqrt(2.0 * T * np.log(N)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regularise_covariance(
        Sigma: np.ndarray,
        min_eigenvalue: float = 1e-6,
    ) -> np.ndarray:
        """Ensure Sigma is symmetric positive-definite.

        Adds a small multiple of the identity if the minimum eigenvalue
        is below the threshold.  Also symmetrises to fix any numerical
        asymmetry.
        """
        Sigma = 0.5 * (Sigma + Sigma.T)  # enforce symmetry
        try:
            eigvals = np.linalg.eigvalsh(Sigma)
            min_eig = eigvals[0]
            if min_eig < min_eigenvalue:
                Sigma += (min_eigenvalue - min_eig + 1e-8) * np.eye(len(Sigma))
        except np.linalg.LinAlgError:
            Sigma += min_eigenvalue * np.eye(len(Sigma))
        return Sigma
