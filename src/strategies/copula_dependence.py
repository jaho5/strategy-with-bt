"""Copula-based tail dependence strategy.

Uses empirical copula methods and rank-based dependence measures to detect
changes in tail dependence structure across a multi-asset portfolio, then
adjusts exposure based on systemic risk signals.

Mathematical foundation
-----------------------
Sklar's Theorem decomposes any joint CDF into marginals and a copula:

    F(x, y) = C(F_X(x), F_Y(y))

where C : [0,1]^2 -> [0,1] is the copula function.

Tail dependence coefficients measure the probability of joint extreme moves:

    lambda_L = lim_{u -> 0}  P(U_2 < u | U_1 < u)  = lim_{u -> 0}  C(u, u) / u
    lambda_U = lim_{u -> 1}  2 - (1 - C(u, u)) / (1 - u)

Empirical estimation uses rank-transformed data (pseudo-observations) and
a threshold approach:

    lambda_hat_L(q) = #{(U_i < q) and (V_i < q)} / #{U_i < q}
    lambda_hat_U(q) = #{(U_i > 1-q) and (V_i > 1-q)} / #{U_i > 1-q}

for a small threshold q (e.g. 0.05).

Strategy
--------
1. **Tail Dependence Estimation** -- For each asset pair, compute empirical
   lower and upper tail dependence coefficients from rank-transformed
   returns over rolling windows.

2. **Contagion Signal** -- Rising lower tail dependence signals increasing
   systemic risk (crisis contagion).  When the z-score of the change in
   average lower tail dependence is positive and large, reduce exposure.
   When it peaks and reverts, increase exposure for the recovery.

3. **Diversification Signal** -- Average pairwise tail dependence across
   the portfolio measures how much diversification benefit remains under
   stress.  Low tail dependence = genuine diversification -> full weight.
   High tail dependence = spurious diversification that evaporates in
   crisis -> reduce weight.

4. **Asymmetry Signal** -- The difference lambda_U - lambda_L measures
   directional tail risk asymmetry.  Positive asymmetry (upper > lower)
   suggests upside co-movement dominates -> mildly bullish.

5. **Implementation** -- All computations use rank-based methods (no
   parametric copula fitting needed).  Kendall's tau and Spearman's rho
   serve as concordance diagnostics.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CopulaDependenceConfig:
    """All tuneable knobs for the copula tail dependence strategy."""

    # Rolling window for tail dependence estimation
    rolling_window: int = 126          # ~6 months of trading days
    min_history: int = 150             # minimum observations before trading

    # Tail dependence threshold
    tail_quantile: float = 0.05        # threshold q for tail dep. estimation
    tail_quantile_upper: float = 0.05  # threshold for upper tail (symmetric default)

    # Z-score parameters for contagion signal
    zscore_window: int = 252           # lookback for z-score normalisation
    contagion_threshold: float = 1.0   # z-score threshold for risk reduction
    recovery_threshold: float = -0.5   # z-score threshold for recovery signal

    # Diversification signal
    high_tail_dep_threshold: float = 0.4   # above this, diversification is poor
    low_tail_dep_threshold: float = 0.15   # below this, diversification is good

    # Signal blending weights
    contagion_weight: float = 0.50     # weight for contagion signal
    diversification_weight: float = 0.30  # weight for diversification signal
    asymmetry_weight: float = 0.20     # weight for asymmetry signal

    # Smoothing
    ema_span: int = 10                 # EMA span for signal smoothing

    # Risk / rebalancing
    max_leverage: float = 1.0          # gross leverage cap
    rebalance_freq: int = 21           # trading days between recalculations


# ---------------------------------------------------------------------------
# Rank transformation utilities
# ---------------------------------------------------------------------------

def _pseudo_observations(data: np.ndarray) -> np.ndarray:
    """Transform data columns to pseudo-observations (empirical CDF values).

    For each column, replace values with their scaled ranks:

        U_i = R_i / (n + 1)

    where R_i is the rank and n is the sample size.  The (n+1) denominator
    ensures U_i in (0, 1), avoiding boundary issues.

    Parameters
    ----------
    data : (T, N) array
        Raw return data with T observations and N assets.

    Returns
    -------
    pseudo_obs : (T, N) array
        Rank-transformed pseudo-observations in (0, 1).
    """
    n = data.shape[0]
    # scipy.stats.rankdata ranks from 1..n, average ties
    pseudo_obs = np.empty_like(data, dtype=np.float64)
    for j in range(data.shape[1]):
        pseudo_obs[:, j] = scipy_stats.rankdata(data[:, j]) / (n + 1)
    return pseudo_obs


# ---------------------------------------------------------------------------
# Tail dependence estimation
# ---------------------------------------------------------------------------

def _empirical_lower_tail_dependence(
    u: np.ndarray,
    v: np.ndarray,
    q: float,
) -> float:
    """Estimate the lower tail dependence coefficient.

    lambda_hat_L(q) = P(V < q | U < q)
                    = #{(U_i < q) and (V_i < q)} / #{U_i < q}

    Parameters
    ----------
    u, v : (T,) arrays
        Pseudo-observations (uniform marginals) for two assets.
    q : float
        Tail threshold, typically 0.05.

    Returns
    -------
    float
        Estimated lower tail dependence in [0, 1].  Returns 0 if there
        are no observations below the threshold.
    """
    mask_u = u < q
    count_u = mask_u.sum()
    if count_u == 0:
        return 0.0
    joint_count = ((u < q) & (v < q)).sum()
    return float(joint_count) / float(count_u)


def _empirical_upper_tail_dependence(
    u: np.ndarray,
    v: np.ndarray,
    q: float,
) -> float:
    """Estimate the upper tail dependence coefficient.

    lambda_hat_U(q) = P(V > 1-q | U > 1-q)
                    = #{(U_i > 1-q) and (V_i > 1-q)} / #{U_i > 1-q}

    Parameters
    ----------
    u, v : (T,) arrays
        Pseudo-observations for two assets.
    q : float
        Tail threshold, typically 0.05.

    Returns
    -------
    float
        Estimated upper tail dependence in [0, 1].
    """
    threshold = 1.0 - q
    mask_u = u > threshold
    count_u = mask_u.sum()
    if count_u == 0:
        return 0.0
    joint_count = ((u > threshold) & (v > threshold)).sum()
    return float(joint_count) / float(count_u)


def _pairwise_tail_dependence(
    pseudo_obs: np.ndarray,
    q_lower: float,
    q_upper: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise lower and upper tail dependence matrices.

    Parameters
    ----------
    pseudo_obs : (T, N) array
        Rank-transformed data.
    q_lower : float
        Threshold for lower tail.
    q_upper : float
        Threshold for upper tail.

    Returns
    -------
    lambda_L : (N, N) array
        Lower tail dependence matrix.
    lambda_U : (N, N) array
        Upper tail dependence matrix.
    """
    n_assets = pseudo_obs.shape[1]
    lambda_L = np.zeros((n_assets, n_assets))
    lambda_U = np.zeros((n_assets, n_assets))

    for i in range(n_assets):
        lambda_L[i, i] = 1.0
        lambda_U[i, i] = 1.0
        for j in range(i + 1, n_assets):
            lam_l = _empirical_lower_tail_dependence(
                pseudo_obs[:, i], pseudo_obs[:, j], q_lower,
            )
            lam_u = _empirical_upper_tail_dependence(
                pseudo_obs[:, i], pseudo_obs[:, j], q_upper,
            )
            lambda_L[i, j] = lam_l
            lambda_L[j, i] = lam_l
            lambda_U[i, j] = lam_u
            lambda_U[j, i] = lam_u

    return lambda_L, lambda_U


def _average_off_diagonal(matrix: np.ndarray) -> float:
    """Compute the average of the off-diagonal elements of a square matrix.

    Parameters
    ----------
    matrix : (N, N) array
        Square matrix.

    Returns
    -------
    float
        Mean of the upper-triangular off-diagonal entries.
    """
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(matrix[mask].mean())


# ---------------------------------------------------------------------------
# Concordance measures
# ---------------------------------------------------------------------------

def _kendall_tau_matrix(data: np.ndarray) -> np.ndarray:
    """Compute the pairwise Kendall's tau matrix.

    Uses scipy.stats.kendalltau for each pair.

    Parameters
    ----------
    data : (T, N) array
        Raw return data.

    Returns
    -------
    tau : (N, N) array
        Kendall's tau correlation matrix.
    """
    n_assets = data.shape[1]
    tau = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            t, _ = scipy_stats.kendalltau(data[:, i], data[:, j])
            if np.isnan(t):
                t = 0.0
            tau[i, j] = t
            tau[j, i] = t
    return tau


def _spearman_rho_matrix(data: np.ndarray) -> np.ndarray:
    """Compute the pairwise Spearman's rho matrix.

    Uses scipy.stats.spearmanr for each pair.

    Parameters
    ----------
    data : (T, N) array
        Raw return data.

    Returns
    -------
    rho : (N, N) array
        Spearman rank correlation matrix.
    """
    n_assets = data.shape[1]
    rho = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            r, _ = scipy_stats.spearmanr(data[:, i], data[:, j])
            if np.isnan(r):
                r = 0.0
            rho[i, j] = r
            rho[j, i] = r
    return rho


# ---------------------------------------------------------------------------
# Copula parameter inversion (for diagnostics, not used in signal generation)
# ---------------------------------------------------------------------------

def _kendall_tau_to_clayton_theta(tau: float) -> float:
    """Convert Kendall's tau to Clayton copula parameter.

    For the Clayton copula: tau = theta / (theta + 2)
    => theta = 2 * tau / (1 - tau)

    Valid only for tau > 0 (positive dependence).

    Parameters
    ----------
    tau : float
        Kendall's tau in (0, 1).

    Returns
    -------
    float
        Clayton theta parameter.  Returns 0.0 if tau <= 0.
    """
    if tau <= 0.0 or tau >= 1.0:
        return 0.0
    return 2.0 * tau / (1.0 - tau)


def _kendall_tau_to_gumbel_theta(tau: float) -> float:
    """Convert Kendall's tau to Gumbel copula parameter.

    For the Gumbel copula: tau = 1 - 1/theta
    => theta = 1 / (1 - tau)

    Valid for tau in [0, 1), theta >= 1.

    Parameters
    ----------
    tau : float
        Kendall's tau in [0, 1).

    Returns
    -------
    float
        Gumbel theta parameter.  Returns 1.0 (independence) if tau <= 0.
    """
    if tau <= 0.0:
        return 1.0
    if tau >= 1.0:
        return np.inf
    return 1.0 / (1.0 - tau)


def _clayton_lower_tail_dependence(theta: float) -> float:
    """Theoretical lower tail dependence for the Clayton copula.

    lambda_L = 2^{-1/theta}

    Parameters
    ----------
    theta : float
        Clayton parameter (> 0).

    Returns
    -------
    float
        Lower tail dependence coefficient.
    """
    if theta <= 0:
        return 0.0
    return 2.0 ** (-1.0 / theta)


def _gumbel_upper_tail_dependence(theta: float) -> float:
    """Theoretical upper tail dependence for the Gumbel copula.

    lambda_U = 2 - 2^{1/theta}

    Parameters
    ----------
    theta : float
        Gumbel parameter (>= 1).

    Returns
    -------
    float
        Upper tail dependence coefficient.
    """
    if theta <= 1.0:
        return 0.0
    return 2.0 - 2.0 ** (1.0 / theta)


# ===========================================================================
# Strategy class
# ===========================================================================

class CopulaDependenceStrategy(Strategy):
    """Copula-based tail dependence strategy.

    Estimates time-varying tail dependence across asset pairs using
    empirical copula methods (rank-transformed pseudo-observations) and
    generates trading signals based on:

    * **Contagion signal**: z-score of changes in average lower tail
      dependence.  Rising lower tail dependence indicates increasing
      systemic risk.
    * **Diversification signal**: overall level of tail dependence.
      Low tail dependence means genuine diversification benefit persists
      under stress; high tail dependence means it evaporates.
    * **Asymmetry signal**: lambda_U - lambda_L captures directional
      asymmetry in tail risk.

    All computations are non-parametric (rank-based empirical copula).
    Kendall's tau and Spearman's rho are computed for diagnostic purposes
    and parametric copula parameter inversion.

    Parameters
    ----------
    config : CopulaDependenceConfig, optional
        Strategy configuration.  Uses defaults if not supplied.
    """

    def __init__(self, config: Optional[CopulaDependenceConfig] = None) -> None:
        self.cfg = config or CopulaDependenceConfig()

        super().__init__(
            name="CopulaDependence",
            description=(
                "Copula-based tail dependence strategy that monitors "
                "systemic risk through empirical lower/upper tail dependence "
                "coefficients and adjusts portfolio exposure accordingly."
            ),
        )

        # State populated during fit / generate_signals
        self._lambda_L: Optional[np.ndarray] = None
        self._lambda_U: Optional[np.ndarray] = None
        self._kendall_tau: Optional[np.ndarray] = None
        self._spearman_rho: Optional[np.ndarray] = None
        self._asset_names: Optional[pd.Index] = None

        # Time-series of aggregate tail dependence for z-scoring
        self._avg_lower_td_history: List[float] = []
        self._avg_upper_td_history: List[float] = []

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _compute_tail_dependence(
        self,
        returns: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute pairwise tail dependence from returns.

        Parameters
        ----------
        returns : (T, N) array
            Log returns.

        Returns
        -------
        lambda_L, lambda_U : (N, N) arrays
            Lower and upper tail dependence matrices.
        """
        pseudo_obs = _pseudo_observations(returns)
        lambda_L, lambda_U = _pairwise_tail_dependence(
            pseudo_obs,
            q_lower=self.cfg.tail_quantile,
            q_upper=self.cfg.tail_quantile_upper,
        )
        return lambda_L, lambda_U

    def _contagion_signal(
        self,
        avg_lower_td_history: np.ndarray,
    ) -> float:
        """Compute the contagion-based signal from the history of average
        lower tail dependence.

        Uses a z-score of the most recent change in average lower tail
        dependence.  Positive z-score (rising tail dependence) signals
        increasing systemic risk -> reduce exposure (signal = -1).
        Negative z-score (declining tail dependence) signals recovery
        -> increase exposure (signal = +1).

        Parameters
        ----------
        avg_lower_td_history : (K,) array
            Time-series of average lower tail dependence values.

        Returns
        -------
        float
            Signal in [-1, +1], continuously valued.
        """
        if len(avg_lower_td_history) < 3:
            return 0.0

        # Changes in tail dependence
        changes = np.diff(avg_lower_td_history)
        if len(changes) < 2:
            return 0.0

        # Z-score of the latest change
        mu = np.mean(changes)
        sigma = np.std(changes, ddof=1)
        if sigma < 1e-12:
            return 0.0

        z = (changes[-1] - mu) / sigma

        # Map z-score to signal
        if z > self.cfg.contagion_threshold:
            # Rising tail dependence: systemic risk increasing
            # Signal strength proportional to z-score magnitude, capped at -1
            return max(-1.0, -z / (2.0 * self.cfg.contagion_threshold))
        elif z < self.cfg.recovery_threshold:
            # Declining tail dependence: recovery
            return min(1.0, -z / (2.0 * abs(self.cfg.recovery_threshold)))
        else:
            # Neutral zone: linear interpolation
            mid = (self.cfg.contagion_threshold + self.cfg.recovery_threshold) / 2.0
            span = (self.cfg.contagion_threshold - self.cfg.recovery_threshold) / 2.0
            if span < 1e-12:
                return 0.0
            return -(z - mid) / span * 0.5

    def _diversification_signal(
        self,
        avg_lower_td: float,
        avg_upper_td: float,
    ) -> float:
        """Compute the diversification quality signal.

        Low average tail dependence means diversification is genuine
        (holds under stress) -> signal = +1.
        High average tail dependence means diversification is illusory
        -> signal = -1.

        Parameters
        ----------
        avg_lower_td : float
            Average pairwise lower tail dependence.
        avg_upper_td : float
            Average pairwise upper tail dependence.

        Returns
        -------
        float
            Signal in [-1, +1].
        """
        # Use the maximum of lower and upper tail dependence as the
        # conservative measure of tail co-movement
        max_td = max(avg_lower_td, avg_upper_td)

        if max_td <= self.cfg.low_tail_dep_threshold:
            # Good diversification
            return 1.0
        elif max_td >= self.cfg.high_tail_dep_threshold:
            # Poor diversification
            return -1.0
        else:
            # Linear interpolation between thresholds
            span = self.cfg.high_tail_dep_threshold - self.cfg.low_tail_dep_threshold
            if span < 1e-12:
                return 0.0
            frac = (max_td - self.cfg.low_tail_dep_threshold) / span
            return 1.0 - 2.0 * frac

    def _asymmetry_signal(
        self,
        avg_lower_td: float,
        avg_upper_td: float,
    ) -> float:
        """Compute the tail asymmetry signal.

        Asymmetry = lambda_U - lambda_L.
        Positive asymmetry (upper tail dependence > lower) means assets
        tend to boom together more than they crash together -> mildly
        bullish.  Negative asymmetry means crash contagion dominates
        -> bearish.

        Parameters
        ----------
        avg_lower_td : float
            Average pairwise lower tail dependence.
        avg_upper_td : float
            Average pairwise upper tail dependence.

        Returns
        -------
        float
            Signal in [-1, +1].
        """
        asymmetry = avg_upper_td - avg_lower_td

        # Normalise: typical asymmetry range is roughly [-0.3, 0.3]
        # Map linearly to [-1, 1]
        normalised = np.clip(asymmetry / 0.3, -1.0, 1.0)
        return float(normalised)

    def _compute_per_asset_weights(
        self,
        returns: np.ndarray,
        lambda_L: np.ndarray,
        lambda_U: np.ndarray,
    ) -> np.ndarray:
        """Compute per-asset position weights based on tail dependence.

        Assets with lower average pairwise tail dependence contribute
        more diversification and receive higher weight.  This implements
        an inverse-tail-dependence weighting scheme.

        Parameters
        ----------
        returns : (T, N) array
            Recent log returns.
        lambda_L : (N, N) array
            Lower tail dependence matrix.
        lambda_U : (N, N) array
            Upper tail dependence matrix.

        Returns
        -------
        weights : (N,) array
            Per-asset weights, summing to 1.
        """
        n_assets = returns.shape[1]
        if n_assets < 2:
            return np.ones(1)

        # Per-asset average tail dependence (average of the row,
        # excluding the diagonal)
        combined_td = np.maximum(lambda_L, lambda_U)
        avg_td_per_asset = np.zeros(n_assets)
        for i in range(n_assets):
            others = np.concatenate([combined_td[i, :i], combined_td[i, i + 1:]])
            avg_td_per_asset[i] = others.mean() if len(others) > 0 else 0.0

        # Inverse tail dependence weighting: lower tail dep -> higher weight
        # Add a floor to avoid division by zero
        inv_td = 1.0 / (avg_td_per_asset + 0.01)

        # Also incorporate inverse volatility for risk management
        asset_vols = np.std(returns, axis=0, ddof=1)
        asset_vols = np.where(asset_vols > 1e-12, asset_vols, 1e-12)
        inv_vol = 1.0 / asset_vols

        # Blend: 60% inverse tail dependence, 40% inverse volatility
        raw_weights = 0.6 * (inv_td / inv_td.sum()) + 0.4 * (inv_vol / inv_vol.sum())

        # Normalise
        total = raw_weights.sum()
        if total > 1e-12:
            raw_weights /= total

        return raw_weights

    # -----------------------------------------------------------------
    # Strategy interface
    # -----------------------------------------------------------------

    def fit(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> "CopulaDependenceStrategy":
        """Calibrate the copula tail dependence model on historical data.

        Computes initial tail dependence matrices, Kendall's tau, and
        Spearman's rho from the training data.  Populates the tail
        dependence history for z-score normalisation.

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

        # Log returns
        log_returns = np.log(prices / prices.shift(1)).iloc[1:].values
        n_obs, n_assets = log_returns.shape

        if n_obs < self.cfg.min_history:
            warnings.warn(
                f"Insufficient history for copula estimation: {n_obs} "
                f"observations (need at least {self.cfg.min_history}).",
                stacklevel=2,
            )
            self._fitted = True
            return self

        # Handle any NaN in returns (forward fill marginals via zero replacement)
        log_returns = np.nan_to_num(log_returns, nan=0.0)

        # Compute tail dependence on the full training set
        self._lambda_L, self._lambda_U = self._compute_tail_dependence(log_returns)

        # Concordance measures (diagnostic)
        self._kendall_tau = _kendall_tau_matrix(log_returns)
        self._spearman_rho = _spearman_rho_matrix(log_returns)

        # Build tail dependence history using rolling windows over
        # the training data for z-score normalisation
        self._avg_lower_td_history = []
        self._avg_upper_td_history = []
        window = self.cfg.rolling_window
        step = max(1, self.cfg.rebalance_freq)

        for t in range(window, n_obs, step):
            window_data = log_returns[t - window:t]
            lam_L, lam_U = self._compute_tail_dependence(window_data)
            self._avg_lower_td_history.append(_average_off_diagonal(lam_L))
            self._avg_upper_td_history.append(_average_off_diagonal(lam_U))

        # Store parameters for inspection
        avg_lower = _average_off_diagonal(self._lambda_L)
        avg_upper = _average_off_diagonal(self._lambda_U)
        avg_tau = _average_off_diagonal(self._kendall_tau)

        # Parametric copula diagnostics via inversion
        clayton_theta = _kendall_tau_to_clayton_theta(max(avg_tau, 0.0))
        gumbel_theta = _kendall_tau_to_gumbel_theta(max(avg_tau, 0.0))

        self.parameters = {
            "avg_lower_tail_dependence": avg_lower,
            "avg_upper_tail_dependence": avg_upper,
            "tail_asymmetry": avg_upper - avg_lower,
            "avg_kendall_tau": avg_tau,
            "avg_spearman_rho": _average_off_diagonal(self._spearman_rho),
            "clayton_theta_from_tau": clayton_theta,
            "gumbel_theta_from_tau": gumbel_theta,
            "clayton_theoretical_lambda_L": _clayton_lower_tail_dependence(clayton_theta),
            "gumbel_theoretical_lambda_U": _gumbel_upper_tail_dependence(gumbel_theta),
            "n_history_points": len(self._avg_lower_td_history),
        }

        logger.info(
            "Copula fit complete: avg lambda_L=%.4f, avg lambda_U=%.4f, "
            "asymmetry=%.4f, avg Kendall tau=%.4f, %d history points",
            avg_lower, avg_upper, avg_upper - avg_lower, avg_tau,
            len(self._avg_lower_td_history),
        )

        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate per-asset trading signals from copula tail dependence.

        Combines three sub-signals (contagion, diversification, asymmetry)
        into a composite exposure signal, then allocates across assets
        using inverse-tail-dependence weighting.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (columns = tickers, index = DatetimeIndex).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for each
            asset.  Signal is in {-1, 0, +1}; weight is the position size.
        """
        self.ensure_fitted()

        n_rows = len(prices)
        n_assets = prices.shape[1]
        asset_names = prices.columns

        # Prepare output DataFrame
        output_cols = []
        for name in asset_names:
            output_cols.extend([f"{name}_signal", f"{name}_weight"])
        signals_df = pd.DataFrame(0.0, index=prices.index, columns=output_cols)

        # Log returns
        log_returns_df = np.log(prices / prices.shift(1))
        log_returns_all = log_returns_df.values
        # Replace NaN with 0 for the first row and any gaps
        log_returns_all = np.nan_to_num(log_returns_all, nan=0.0)

        # Copy tail dependence history from fit (will be extended in-sample)
        lower_td_history = list(self._avg_lower_td_history)
        upper_td_history = list(self._avg_upper_td_history)

        window = self.cfg.rolling_window
        min_lookback = max(window, self.cfg.min_history)

        # Cached signals
        cached_signals = np.zeros(n_assets)
        cached_weights = np.zeros(n_assets)
        last_rebalance = -self.cfg.rebalance_freq  # force first computation

        for t in range(min_lookback, n_rows):
            if (t - last_rebalance) < self.cfg.rebalance_freq:
                # Use cached values
                for j, name in enumerate(asset_names):
                    sig_col = f"{name}_signal"
                    wgt_col = f"{name}_weight"
                    signals_df.iloc[t, signals_df.columns.get_loc(sig_col)] = cached_signals[j]
                    signals_df.iloc[t, signals_df.columns.get_loc(wgt_col)] = cached_weights[j]
                continue

            # Extract window of returns
            window_returns = log_returns_all[t - window:t]

            # 1. Compute tail dependence for current window
            lambda_L, lambda_U = self._compute_tail_dependence(window_returns)

            avg_lower_td = _average_off_diagonal(lambda_L)
            avg_upper_td = _average_off_diagonal(lambda_U)

            lower_td_history.append(avg_lower_td)
            upper_td_history.append(avg_upper_td)

            # 2. Contagion signal (from z-score of tail dep. changes)
            # Use the most recent zscore_window observations
            zscore_lookback = min(
                len(lower_td_history),
                self.cfg.zscore_window // self.cfg.rebalance_freq + 1,
            )
            recent_history = np.array(lower_td_history[-zscore_lookback:])
            contagion_sig = self._contagion_signal(recent_history)

            # 3. Diversification signal
            divers_sig = self._diversification_signal(avg_lower_td, avg_upper_td)

            # 4. Asymmetry signal
            asym_sig = self._asymmetry_signal(avg_lower_td, avg_upper_td)

            # 5. Composite signal (weighted blend)
            composite = (
                self.cfg.contagion_weight * contagion_sig
                + self.cfg.diversification_weight * divers_sig
                + self.cfg.asymmetry_weight * asym_sig
            )

            # 6. Per-asset weights via inverse tail dependence
            asset_weights = self._compute_per_asset_weights(
                window_returns, lambda_L, lambda_U,
            )

            # 7. Apply composite signal direction to weights
            # All assets share the same directional view (systemic signal)
            # but differ in sizing
            direction = np.sign(composite) if abs(composite) > 0.05 else 0.0
            magnitude = min(abs(composite), 1.0)

            raw_weights = asset_weights * magnitude

            # Enforce leverage limit
            gross = raw_weights.sum()
            if gross > self.cfg.max_leverage:
                raw_weights *= self.cfg.max_leverage / gross

            cached_signals = np.full(n_assets, direction)
            cached_weights = raw_weights

            last_rebalance = t

            # Write signals
            for j, name in enumerate(asset_names):
                sig_col = f"{name}_signal"
                wgt_col = f"{name}_weight"
                signals_df.iloc[t, signals_df.columns.get_loc(sig_col)] = cached_signals[j]
                signals_df.iloc[t, signals_df.columns.get_loc(wgt_col)] = cached_weights[j]

        # Apply EMA smoothing to weight columns for signal stability
        for name in asset_names:
            wgt_col = f"{name}_weight"
            signals_df[wgt_col] = self.exponential_smooth(
                signals_df[wgt_col], span=self.cfg.ema_span,
            )

        return signals_df

    # -----------------------------------------------------------------
    # Diagnostic methods
    # -----------------------------------------------------------------

    def get_tail_dependence_matrices(
        self,
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Return the lower and upper tail dependence matrices from the
        last fit as labelled DataFrames.

        Returns
        -------
        (lambda_L_df, lambda_U_df) or None if not fitted.
        """
        if (
            self._lambda_L is None
            or self._lambda_U is None
            or self._asset_names is None
        ):
            return None
        lambda_L_df = pd.DataFrame(
            self._lambda_L, index=self._asset_names, columns=self._asset_names,
        )
        lambda_U_df = pd.DataFrame(
            self._lambda_U, index=self._asset_names, columns=self._asset_names,
        )
        return lambda_L_df, lambda_U_df

    def get_concordance_measures(
        self,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Return Kendall's tau and Spearman's rho matrices from the last fit.

        Returns
        -------
        dict with keys 'kendall_tau' and 'spearman_rho', each a DataFrame,
        or None if not fitted.
        """
        if (
            self._kendall_tau is None
            or self._spearman_rho is None
            or self._asset_names is None
        ):
            return None
        return {
            "kendall_tau": pd.DataFrame(
                self._kendall_tau,
                index=self._asset_names,
                columns=self._asset_names,
            ),
            "spearman_rho": pd.DataFrame(
                self._spearman_rho,
                index=self._asset_names,
                columns=self._asset_names,
            ),
        }

    def get_tail_dependence_history(self) -> Optional[pd.DataFrame]:
        """Return the time-series of average tail dependence used for
        z-score computation during fit.

        Returns
        -------
        pd.DataFrame with columns 'avg_lower_td' and 'avg_upper_td',
        or None if not fitted.
        """
        if not self._avg_lower_td_history:
            return None
        return pd.DataFrame({
            "avg_lower_td": self._avg_lower_td_history,
            "avg_upper_td": self._avg_upper_td_history,
        })

    def get_copula_parameters(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Return parametric copula parameters derived via Kendall's tau
        inversion for each asset pair.

        Computes Clayton theta (lower tail model) and Gumbel theta
        (upper tail model) from pairwise Kendall's tau, along with
        the implied theoretical tail dependence coefficients.

        Returns
        -------
        dict with keys 'clayton_theta', 'gumbel_theta',
        'clayton_lambda_L', 'gumbel_lambda_U', each a DataFrame,
        or None if not fitted.
        """
        if self._kendall_tau is None or self._asset_names is None:
            return None

        n = self._kendall_tau.shape[0]
        clayton_theta = np.zeros((n, n))
        gumbel_theta = np.ones((n, n))
        clayton_lam_L = np.zeros((n, n))
        gumbel_lam_U = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                tau = self._kendall_tau[i, j]
                ct = _kendall_tau_to_clayton_theta(max(tau, 0.0))
                gt = _kendall_tau_to_gumbel_theta(max(tau, 0.0))
                clayton_theta[i, j] = ct
                gumbel_theta[i, j] = gt
                clayton_lam_L[i, j] = _clayton_lower_tail_dependence(ct)
                gumbel_lam_U[i, j] = _gumbel_upper_tail_dependence(gt)

        names = self._asset_names
        return {
            "clayton_theta": pd.DataFrame(
                clayton_theta, index=names, columns=names,
            ),
            "gumbel_theta": pd.DataFrame(
                gumbel_theta, index=names, columns=names,
            ),
            "clayton_lambda_L": pd.DataFrame(
                clayton_lam_L, index=names, columns=names,
            ),
            "gumbel_lambda_U": pd.DataFrame(
                gumbel_lam_U, index=names, columns=names,
            ),
        }

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        extra = ""
        if self._fitted and self._lambda_L is not None:
            n = self._lambda_L.shape[0]
            avg_l = _average_off_diagonal(self._lambda_L)
            avg_u = _average_off_diagonal(self._lambda_U)
            extra = f", {n} assets, avg_lambda_L={avg_l:.3f}, avg_lambda_U={avg_u:.3f}"
        return f"CopulaDependenceStrategy({fitted_tag}{extra})"
