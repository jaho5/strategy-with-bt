"""Semigroup decay autocorrelation trading strategy.

Models the autocorrelation function of log-returns as a sum of decaying
exponentials (a *contraction semigroup*):

    rho(tau) = sum_{k=1}^{K} c_k exp(-lambda_k tau)

and extracts the characteristic decay eigenvalues {lambda_k} using Prony's
method (eigenvalue decomposition of a Hankel matrix built from the sample
autocorrelation).

Trading logic
-------------
*   Small lambda (slow decay) -> strong autocorrelation -> trending regime
    -> momentum signal (trade in the direction of recent returns).
*   Large lambda (fast decay) -> weak autocorrelation -> mean-reverting
    regime -> contrarian signal (trade against recent returns).

The dominant eigenvalue (largest |c_k|) determines which regime the asset
is in.  A blending weight between momentum and mean-reversion is computed
from the ratio of slow-decay vs fast-decay energy.

References
----------
*   Prony, G. R. de (1795). Essai experimental et analytique.
*   Hildebrand (1956). Introduction to Numerical Analysis, ch. 9.
*   Hauer, Demeure, Scharf (1990). Initial results in Prony analysis of
    power system response signals. IEEE Trans. Power Systems.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prony's method for exponential decomposition
# ---------------------------------------------------------------------------

def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute the sample autocorrelation of *x* for lags 0..max_lag.

    Returns an array of length max_lag + 1 where index k is rho(k).
    """
    n = len(x)
    x_centered = x - x.mean()
    var = np.dot(x_centered, x_centered) / n
    if var < 1e-15:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag >= n:
            break
        acf[lag] = np.dot(x_centered[: n - lag], x_centered[lag:]) / (n * var)
    return acf


def _prony_decomposition(
    acf: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract exponential decay eigenvalues and amplitudes via Prony's method.

    Given autocorrelation values rho(0), rho(1), ..., rho(2K-1) where K is
    *n_components*, build the Hankel matrix and solve the generalised
    eigenvalue problem to recover {lambda_k, c_k}.

    Parameters
    ----------
    acf : 1-D array of autocorrelation values (length >= 2 * n_components).
    n_components : number of exponential components K to extract.

    Returns
    -------
    lambdas : (K,) array of decay rates (positive = decaying).
    amplitudes : (K,) array of corresponding amplitudes c_k.
    """
    K = n_components
    L = len(acf)

    # Need at least 2K values
    if L < 2 * K:
        K = L // 2
    if K < 1:
        return np.array([1.0]), np.array([1.0])

    # Build Hankel matrix H of size K x K from acf[0..2K-2]
    H = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            idx = i + j
            if idx < L:
                H[i, j] = acf[idx]

    # Build shifted Hankel H1 from acf[1..2K-1]
    H1 = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            idx = i + j + 1
            if idx < L:
                H1[i, j] = acf[idx]

    # Eigenvalues of H^{-1} H1 give the z_k = exp(-lambda_k)
    try:
        # Use pseudo-inverse for robustness
        H_pinv = np.linalg.pinv(H, rcond=1e-8)
        A = H_pinv @ H1

        eigenvalues = np.linalg.eigvals(A)
        # Keep only real parts (small imaginary parts are numerical noise)
        z_k = np.real(eigenvalues)
    except np.linalg.LinAlgError:
        return np.array([1.0]), np.array([1.0])

    # Convert z_k = exp(-lambda_k) -> lambda_k = -log(z_k)
    # Clip to avoid log of non-positive values
    z_k = np.clip(z_k, 1e-10, 1.0 - 1e-10)
    lambdas = -np.log(z_k)

    # Recover amplitudes c_k via Vandermonde system:
    #   acf[m] = sum_k c_k z_k^m  for m = 0, ..., K-1
    V = np.vander(z_k, N=K, increasing=True).T  # shape (K, K)
    try:
        amplitudes = np.linalg.lstsq(V.T, acf[:K], rcond=None)[0]
    except np.linalg.LinAlgError:
        amplitudes = np.ones(K) / K

    return np.real(lambdas), np.real(amplitudes)


# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------

def _regime_blend(
    lambdas: np.ndarray,
    amplitudes: np.ndarray,
    threshold: float,
) -> float:
    """Compute a regime blend score in [-1, +1].

    Returns a value where:
    *   +1 = purely trending (momentum)
    *   -1 = purely mean-reverting (contrarian)
    *    0 = ambiguous / mixed

    The score is the energy-weighted average of sign(threshold - lambda_k).
    """
    abs_amp = np.abs(amplitudes)
    total_energy = abs_amp.sum()
    if total_energy < 1e-15:
        return 0.0

    weights = abs_amp / total_energy
    # Positive contribution when lambda < threshold (slow decay = trending)
    # Negative contribution when lambda > threshold (fast decay = reverting)
    scores = np.where(lambdas < threshold, 1.0, -1.0)
    blend = float(np.dot(weights, scores))
    return np.clip(blend, -1.0, 1.0)


def _momentum_signal(returns: np.ndarray, lookback: int) -> float:
    """Simple momentum: sign of cumulative return over lookback."""
    if len(returns) < lookback:
        return 0.0
    cum = returns[-lookback:].sum()
    return float(np.sign(cum))


def _mean_reversion_signal(returns: np.ndarray, lookback: int) -> float:
    """Mean-reversion: negative sign of z-score of cumulative return."""
    if len(returns) < lookback:
        return 0.0
    cum = returns[-lookback:].sum()
    std = returns[-lookback:].std()
    if std < 1e-12:
        return 0.0
    z = cum / std
    # Contrarian: short when z is high, long when z is low
    return -float(np.sign(z))


# ===========================================================================
# Strategy class
# ===========================================================================

class SemigroupDecayStrategy(Strategy):
    """Semigroup exponential-decay autocorrelation strategy.

    Parameters
    ----------
    n_components : int
        Number of exponential components K to extract via Prony's method.
        Default 3.
    acf_max_lag : int
        Maximum lag for autocorrelation estimation.  Default 40.
    lookback : int
        Rolling window (trading days) for computing autocorrelation and
        return statistics.  Default 120.
    signal_lookback : int
        Lookback for the momentum / mean-reversion sub-signals.  Default 20.
    decay_threshold : float
        Lambda threshold separating "slow" (trending) from "fast"
        (mean-reverting) decay.  Default 0.3.
    rebalance_freq : int
        Rebalance every N trading days.  Default 5.
    min_history : int
        Minimum observations before generating non-zero signals.  Default 60.
    """

    def __init__(
        self,
        n_components: int = 3,
        acf_max_lag: int = 40,
        lookback: int = 120,
        signal_lookback: int = 20,
        decay_threshold: float = 0.3,
        rebalance_freq: int = 5,
        min_history: int = 60,
    ) -> None:
        super().__init__(
            name="SemigroupDecay",
            description=(
                "Autocorrelation semigroup decay strategy using Prony's method "
                "to classify trending vs mean-reverting regimes."
            ),
        )
        self.n_components = n_components
        self.acf_max_lag = acf_max_lag
        self.lookback = lookback
        self.signal_lookback = signal_lookback
        self.decay_threshold = decay_threshold
        self.rebalance_freq = rebalance_freq
        self.min_history = min_history

        # Fitted parameters (per-ticker dominant eigenvalues)
        self._fitted_eigenvalues: Optional[Dict[str, np.ndarray]] = None
        self._fitted_amplitudes: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def fit(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> "SemigroupDecayStrategy":
        """Fit Prony decomposition on the training window.

        Computes the sample autocorrelation of each asset's log-returns
        and extracts the exponential decay eigenvalues.

        Parameters
        ----------
        prices : pd.DataFrame
            Close prices, columns = tickers, index = dates.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        prices_clean = prices.ffill().dropna()
        if len(prices_clean) < self.min_history:
            warnings.warn(
                f"Only {len(prices_clean)} clean observations; need at least "
                f"{self.min_history}. Prony decomposition will be deferred.",
                stacklevel=2,
            )
            self._fitted = True
            self._fitted_eigenvalues = {}
            self._fitted_amplitudes = {}
            return self

        log_ret = np.log(prices_clean / prices_clean.shift(1)).dropna()

        self._fitted_eigenvalues = {}
        self._fitted_amplitudes = {}

        for col in log_ret.columns:
            r = log_ret[col].values.astype(np.float64)
            r = r[np.isfinite(r)]
            if len(r) < 2 * self.n_components:
                continue

            acf = _autocorrelation(r, self.acf_max_lag)
            lambdas, amps = _prony_decomposition(acf, self.n_components)
            self._fitted_eigenvalues[col] = lambdas
            self._fitted_amplitudes[col] = amps

        self.parameters = {
            "n_assets_fitted": len(self._fitted_eigenvalues),
            "decay_threshold": self.decay_threshold,
        }
        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate directional signals based on autocorrelation regime.

        Parameters
        ----------
        prices : pd.DataFrame
            Close prices, columns = tickers, index = dates.

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` (in {-1, 0, +1}) and
            ``{ticker}_weight`` for each ticker.
        """
        self.validate_prices(prices)

        tickers = list(prices.columns)
        n_assets = len(tickers)
        n_dates = len(prices)

        prices_clean = prices.ffill()
        log_returns = np.log(prices_clean / prices_clean.shift(1))
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        all_signals = np.zeros((n_dates, n_assets))
        all_weights = np.zeros((n_dates, n_assets))

        for j, ticker in enumerate(tickers):
            r = log_returns[ticker].values.astype(np.float64)
            prev_signal = 0.0

            for t in range(n_dates):
                if t < self.min_history:
                    all_signals[t, j] = 0.0
                    all_weights[t, j] = 0.0
                    continue

                # Only recompute at rebalance frequency
                if t % self.rebalance_freq != 0:
                    all_signals[t, j] = prev_signal
                    all_weights[t, j] = all_weights[max(0, t - 1), j]
                    continue

                # Rolling autocorrelation and Prony decomposition
                start = max(0, t - self.lookback)
                window = r[start:t]
                valid = window[np.isfinite(window)]
                if len(valid) < 2 * self.n_components:
                    all_signals[t, j] = 0.0
                    all_weights[t, j] = 0.0
                    prev_signal = 0.0
                    continue

                acf = _autocorrelation(valid, self.acf_max_lag)
                lambdas, amps = _prony_decomposition(acf, self.n_components)

                # Compute regime blend
                blend = _regime_blend(lambdas, amps, self.decay_threshold)

                # Sub-signals
                mom = _momentum_signal(valid, self.signal_lookback)
                mr = _mean_reversion_signal(valid, self.signal_lookback)

                # Blend: positive blend -> more momentum, negative -> more MR
                alpha = (blend + 1.0) / 2.0  # map [-1,1] to [0,1]
                raw_signal = alpha * mom + (1.0 - alpha) * mr

                # Discretise
                if raw_signal > 0.2:
                    signal = 1.0
                elif raw_signal < -0.2:
                    signal = -1.0
                else:
                    signal = 0.0

                # Weight based on confidence (how extreme the blend is)
                confidence = abs(blend)
                weight = np.clip(confidence, 0.1, 1.0)

                all_signals[t, j] = signal
                all_weights[t, j] = weight
                prev_signal = signal

        # Build output
        signals_df = pd.DataFrame(index=prices.index)
        for j, ticker in enumerate(tickers):
            signals_df[f"{ticker}_signal"] = all_signals[:, j]
            signals_df[f"{ticker}_weight"] = all_weights[:, j]

        return signals_df
