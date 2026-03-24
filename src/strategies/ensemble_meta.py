"""Meta-strategy: advanced ensemble combination of individual strategy experts.

Combines up to K individual strategies using three principled aggregation
methods drawn from online learning theory and portfolio optimisation, then
averages the three weighting schemes into a single robust meta-signal.

Mathematical foundation
-----------------------
1.  **Hedge / Multiplicative Weights (MW)** -- online convex optimisation
    with expert advice.  Exponentially weighted average forecaster
    (Vovk-Azoury-Warmuth family):

        w_{t+1,k} = w_{t,k} * exp(eta * r_{k,t}) / Z_t

    where Z_t is the normalisation constant and
    eta = sqrt(2 * log(K) / T) yields the minimax-optimal regret bound

        Regret(T) <= sqrt(T * log(K) / 2).

2.  **Performance-weighted softmax** -- rolling Sharpe ratios mapped
    through a temperature-annealed softmax:

        w_k  proportional to  exp(sharpe_k / tau_t)

    with tau_t = tau_0 * decay^t  (annealing from exploration to
    exploitation).

3.  **Correlation-aware Markowitz** -- mean-variance optimisation over the
    strategy return streams:

        max_w  (mu' w) / sqrt(w' Sigma w)
        s.t.   w >= 0,  sum(w) = 1

    solved via constrained quadratic programming (scipy.optimize).

The final signal at each time step is the equal-weight average of the
positions produced by the three weighting schemes, providing robustness
against any single combination method failing.
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
# Constants
# ---------------------------------------------------------------------------

_SHARPE_WINDOW: int = 63          # Rolling Sharpe window (~ 1 quarter)
_ANNUALISATION: float = np.sqrt(252)
_INITIAL_TEMPERATURE: float = 1.0
_TEMPERATURE_DECAY: float = 0.995  # Per-period decay for annealing
_MIN_TEMPERATURE: float = 0.05
_MIN_HISTORY: int = 10             # Minimum observations before we trust MW


# ---------------------------------------------------------------------------
# Ensemble Meta-Strategy
# ---------------------------------------------------------------------------

class EnsembleMetaStrategy(Strategy):
    """Meta-strategy that aggregates K expert strategies via three
    complementary weighting schemes and averages the result.

    Parameters
    ----------
    sharpe_window : int
        Lookback for rolling Sharpe ratio estimation (default 63).
    initial_temperature : float
        Starting temperature for softmax annealing (default 1.0).
    temperature_decay : float
        Multiplicative decay applied to temperature each period (default 0.995).
    min_temperature : float
        Floor for the annealed temperature (default 0.05).
    markowitz_lookback : int
        Lookback for the covariance matrix used in the Markowitz weighting
        scheme (default 63).
    risk_limits
        Optional :class:`RiskLimits` forwarded to the base class.
    """

    def __init__(
        self,
        sharpe_window: int = _SHARPE_WINDOW,
        initial_temperature: float = _INITIAL_TEMPERATURE,
        temperature_decay: float = _TEMPERATURE_DECAY,
        min_temperature: float = _MIN_TEMPERATURE,
        markowitz_lookback: int = 63,
        risk_limits=None,
    ) -> None:
        super().__init__(
            name="EnsembleMeta",
            description=(
                "Meta-strategy combining K expert strategies via "
                "Hedge/MW, Sharpe-softmax, and Markowitz-MVO weighting"
            ),
            risk_limits=risk_limits,
        )
        self.parameters.update(
            sharpe_window=sharpe_window,
            initial_temperature=initial_temperature,
            temperature_decay=temperature_decay,
            min_temperature=min_temperature,
            markowitz_lookback=markowitz_lookback,
        )

        # Learned / calibrated state (populated by fit)
        self._expert_names: list[str] = []
        self._K: int = 0                   # Number of experts
        self._eta: float = 0.0             # Learning rate for MW
        self._mw_weights: np.ndarray = np.array([])  # Current MW weights
        self._cumulative_returns: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Fitting / calibration
    # ------------------------------------------------------------------

    def fit(
        self,
        prices: pd.DataFrame,
        *,
        strategy_returns: Optional[Dict[str, pd.Series]] = None,
        **kwargs: Any,
    ) -> "EnsembleMetaStrategy":
        """Calibrate the ensemble on historical strategy return streams.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (forwarded for API compatibility; not directly used
            by the meta-strategy).
        strategy_returns : dict[str, pd.Series]
            Mapping from strategy name to its daily return series.  All
            series must share a common DatetimeIndex (or at least overlap
            substantially).

        Returns
        -------
        self
        """
        if strategy_returns is None:
            strategy_returns = kwargs.get("strategy_returns", None)
        if strategy_returns is None or len(strategy_returns) == 0:
            raise ValueError(
                "EnsembleMetaStrategy.fit() requires 'strategy_returns': "
                "a dict mapping strategy names to their return series."
            )

        self._expert_names = sorted(strategy_returns.keys())
        self._K = len(self._expert_names)

        # Align return streams into a single DataFrame
        returns_df = pd.DataFrame(
            {name: strategy_returns[name] for name in self._expert_names}
        ).sort_index()
        returns_df = returns_df.fillna(0.0)

        T = len(returns_df)

        # Optimal learning rate for Hedge with T-step horizon
        if T > 0 and self._K > 1:
            self._eta = np.sqrt(2.0 * np.log(self._K) / max(T, 1))
        else:
            self._eta = 0.1  # Fallback

        # Run the MW algorithm over the training data to obtain initial
        # weights (warm start for live trading)
        self._mw_weights = self._run_multiplicative_weights(returns_df)

        # Store cumulative returns for potential downstream diagnostics
        self._cumulative_returns = (1.0 + returns_df).cumprod()

        self.parameters["eta"] = float(self._eta)
        self.parameters["K"] = self._K
        self.parameters["expert_names"] = list(self._expert_names)
        self._fitted = True
        logger.info(
            "EnsembleMeta fitted on %d experts over %d periods (eta=%.6f).",
            self._K, T, self._eta,
        )
        return self

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        prices: pd.DataFrame,
        *,
        strategy_signals: Optional[Dict[str, pd.Series]] = None,
        strategy_returns: Optional[Dict[str, pd.Series]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Produce ensemble signals from individual strategy signals.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (used to define the output index when strategy
            signals do not cover the full range).
        strategy_signals : dict[str, pd.Series]
            Per-strategy signal series (values in {-1, 0, 1} or continuous).
        strategy_returns : dict[str, pd.Series]
            Per-strategy return series used for adaptive weighting.
            If ``None``, equal weights are used for MW and Sharpe methods.

        Returns
        -------
        pd.DataFrame
            Columns ``signal`` and ``weight``.
        """
        self.ensure_fitted()

        if strategy_signals is None:
            strategy_signals = kwargs.get("strategy_signals", None)
        if strategy_signals is None or len(strategy_signals) == 0:
            raise ValueError(
                "generate_signals() requires 'strategy_signals': "
                "a dict mapping strategy names to signal series."
            )
        if strategy_returns is None:
            strategy_returns = kwargs.get("strategy_returns", None)

        # Build aligned DataFrames ------------------------------------------
        sig_df = pd.DataFrame(strategy_signals).sort_index().fillna(0.0)

        if strategy_returns is not None and len(strategy_returns) > 0:
            ret_df = pd.DataFrame(strategy_returns).sort_index().fillna(0.0)
        else:
            ret_df = None

        # Ensure we only work with experts known at fit time
        known = [c for c in self._expert_names if c in sig_df.columns]
        if len(known) == 0:
            warnings.warn(
                "No fitted expert names found in strategy_signals. "
                "Returning zero signal.",
                stacklevel=2,
            )
            out = pd.DataFrame(
                {"signal": 0.0, "weight": 0.0}, index=sig_df.index
            )
            return out

        sig_df = sig_df[known]
        if ret_df is not None:
            ret_cols = [c for c in known if c in ret_df.columns]
            ret_df = ret_df[ret_cols] if ret_cols else None

        K = len(known)

        # --- 1. Multiplicative Weights (Hedge) ----------------------------
        if ret_df is not None and len(ret_df) >= _MIN_HISTORY:
            mw_w = self._run_multiplicative_weights(ret_df)
        else:
            mw_w = np.full(K, 1.0 / K)

        mw_signal = sig_df.values @ mw_w  # (T,)

        # --- 2. Sharpe-softmax weighting ----------------------------------
        if ret_df is not None and len(ret_df) >= _MIN_HISTORY:
            sharpe_w = self._sharpe_softmax_weights(ret_df)
        else:
            sharpe_w = np.full(K, 1.0 / K)

        sharpe_signal = sig_df.values @ sharpe_w  # (T,)

        # --- 3. Markowitz correlation-aware weighting ---------------------
        if ret_df is not None and len(ret_df) >= _MIN_HISTORY:
            mvo_w = self._markowitz_weights(ret_df)
        else:
            mvo_w = np.full(K, 1.0 / K)

        mvo_signal = sig_df.values @ mvo_w  # (T,)

        # --- Combine the three schemes ------------------------------------
        combined = (mw_signal + sharpe_signal + mvo_signal) / 3.0

        # Convert continuous combined signal to discrete {-1, 0, +1} and a
        # confidence weight in [0, 1].
        signal = np.sign(combined)
        weight = np.clip(np.abs(combined), 0.0, 1.0)

        out = pd.DataFrame(
            {"signal": signal, "weight": weight},
            index=sig_df.index,
        )
        return out

    # ------------------------------------------------------------------
    # Weighting scheme 1: Multiplicative Weights / Hedge
    # ------------------------------------------------------------------

    def _run_multiplicative_weights(
        self, returns_df: pd.DataFrame
    ) -> np.ndarray:
        """Run the Hedge algorithm over ``returns_df`` and return final
        normalised weight vector.

        The learning rate ``eta`` is set at fit time for the minimax-optimal
        regret bound::

            eta = sqrt(2 * log(K) / T)

        yielding  Regret(T) <= sqrt(T * log(K) / 2).
        """
        K = returns_df.shape[1]
        T = returns_df.shape[0]

        if K == 0:
            return np.array([])

        eta = np.sqrt(2.0 * np.log(max(K, 2)) / max(T, 1))

        # Initialise uniform weights
        log_w = np.zeros(K)  # Work in log-space for numerical stability

        ret_matrix = returns_df.values  # (T, K)

        for t in range(T):
            r_t = ret_matrix[t]  # (K,)
            # Multiplicative update in log-space:
            # log w_{t+1,k} = log w_{t,k} + eta * r_{k,t}
            log_w += eta * r_t
            # Normalise (subtract log Z_t = logsumexp)
            log_w -= _logsumexp(log_w)

        # Convert back to probability simplex
        weights = np.exp(log_w)
        weights /= weights.sum()  # Guard against floating-point drift
        return weights

    # ------------------------------------------------------------------
    # Weighting scheme 2: Rolling Sharpe softmax with temperature annealing
    # ------------------------------------------------------------------

    def _sharpe_softmax_weights(
        self, returns_df: pd.DataFrame
    ) -> np.ndarray:
        """Compute softmax weights from rolling Sharpe ratios with
        temperature annealing.

        Temperature schedule::

            tau_t = max(tau_0 * decay^t, tau_min)

        At high tau (early), weights are nearly uniform (exploration);
        as tau decreases, weights concentrate on the best-performing
        expert (exploitation).
        """
        window = self.parameters["sharpe_window"]
        tau_0 = self.parameters["initial_temperature"]
        decay = self.parameters["temperature_decay"]
        tau_min = self.parameters["min_temperature"]

        T = len(returns_df)
        K = returns_df.shape[1]

        if K == 0:
            return np.array([])

        # Compute rolling Sharpe ratios
        rolling_mean = returns_df.rolling(window, min_periods=max(window // 2, 2)).mean()
        rolling_std = returns_df.rolling(window, min_periods=max(window // 2, 2)).std()
        rolling_std = rolling_std.replace(0.0, np.nan)
        rolling_sharpe = (rolling_mean / rolling_std) * _ANNUALISATION
        rolling_sharpe = rolling_sharpe.fillna(0.0)

        # Use the most recent Sharpe ratios for the final weight
        # (with annealed temperature reflecting how far into the series we are)
        sharpe_last = rolling_sharpe.iloc[-1].values  # (K,)
        tau_t = max(tau_0 * (decay ** T), tau_min)

        # Numerically stable softmax
        scaled = sharpe_last / tau_t
        weights = _softmax(scaled)
        return weights

    # ------------------------------------------------------------------
    # Weighting scheme 3: Markowitz mean-variance (max Sharpe)
    # ------------------------------------------------------------------

    def _markowitz_weights(
        self, returns_df: pd.DataFrame
    ) -> np.ndarray:
        """Solve the constrained max-Sharpe portfolio over expert return
        streams:

            max_w  mu' w / sqrt(w' Sigma w)
            s.t.   w >= 0,  sum(w) = 1

        Uses ``scipy.optimize.minimize`` with SLSQP.  Falls back to
        equal-weight if optimisation fails or if the covariance matrix is
        degenerate.
        """
        lookback = self.parameters["markowitz_lookback"]
        K = returns_df.shape[1]

        if K == 0:
            return np.array([])
        if K == 1:
            return np.array([1.0])

        # Use the trailing window for estimation
        tail = returns_df.iloc[-lookback:]
        mu = tail.mean().values  # (K,)
        cov = tail.cov().values  # (K, K)

        # Regularise covariance (Ledoit-Wolf-style shrinkage lite)
        trace_cov = np.trace(cov)
        if trace_cov <= 0:
            return np.full(K, 1.0 / K)
        shrinkage = 0.1
        cov = (1 - shrinkage) * cov + shrinkage * (trace_cov / K) * np.eye(K)

        # Objective: minimise negative Sharpe ratio
        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = mu @ w
            port_var = w @ cov @ w
            if port_var <= 1e-16:
                return 0.0  # Degenerate -- no risk
            return -port_ret / np.sqrt(port_var)

        # Gradient for SLSQP
        def neg_sharpe_jac(w: np.ndarray) -> np.ndarray:
            port_ret = mu @ w
            port_var = w @ cov @ w
            port_vol = np.sqrt(max(port_var, 1e-16))
            d_ret = mu
            d_var = 2.0 * cov @ w
            # d/dw  [-ret/vol] = -mu/vol + ret / (2 * vol^3) * 2 * Sigma w
            return -d_ret / port_vol + (port_ret / (port_vol ** 3)) * (cov @ w)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, 1.0)] * K
        w0 = np.full(K, 1.0 / K)

        try:
            res = minimize(
                neg_sharpe,
                w0,
                jac=neg_sharpe_jac,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if res.success:
                weights = res.x
                weights = np.maximum(weights, 0.0)
                s = weights.sum()
                if s > 0:
                    weights /= s
                else:
                    weights = np.full(K, 1.0 / K)
            else:
                logger.debug(
                    "Markowitz optimisation did not converge: %s. "
                    "Falling back to equal weight.",
                    res.message,
                )
                weights = np.full(K, 1.0 / K)
        except Exception:
            logger.warning(
                "Markowitz optimisation failed; falling back to equal weight.",
                exc_info=True,
            )
            weights = np.full(K, 1.0 / K)

        return weights


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------

def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    a_max = a.max()
    if not np.isfinite(a_max):
        return a_max
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - x.max()
    e = np.exp(x_shifted)
    s = e.sum()
    if s == 0:
        return np.full_like(x, 1.0 / len(x))
    return e / s
