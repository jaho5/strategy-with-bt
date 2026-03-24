"""Szego prediction-theoretic trading strategy.

Uses the classical result from Szego's theorem on Toeplitz determinants
to assess the theoretical predictability of asset returns, then builds
an AR-based optimal linear predictor whose aggressiveness is scaled by
the prediction efficiency.

Mathematical foundation
-----------------------
For a stationary process with spectral density f(omega), Szego's theorem
gives the geometric mean of the spectrum:

    G = exp( (1/2pi) int_0^{2pi} log f(omega) d omega )

The *prediction efficiency* is:

    eta = 1 - G / A

where A = (1/2pi) int_0^{2pi} f(omega) d omega is the arithmetic mean
(total power).  By the AM-GM inequality G <= A, so eta in [0, 1].

*   eta ~ 1: the spectrum has deep nulls -> the process is highly
    predictable -> trade aggressively.
*   eta ~ 0: the spectrum is flat (white noise) -> no predictability
    -> stay flat.

The AR coefficients are estimated via Burg's method (maximum-entropy
spectral estimation), which is numerically stable and well-suited to
short time series.

References
----------
*   Szego, G. (1920). Beitrage zur Theorie der Toeplitzschen Formen.
*   Burg, J. P. (1975). Maximum entropy spectral analysis. PhD thesis,
    Stanford University.
*   Haykin, S. (2001). Adaptive Filter Theory, 4th ed., ch. 7-9.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Burg's method for AR parameter estimation
# ---------------------------------------------------------------------------

def _burg_ar(x: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Estimate AR coefficients using Burg's method.

    Parameters
    ----------
    x : 1-D array of observations (length N).
    order : AR model order p.

    Returns
    -------
    a : (p,) AR coefficients [a_1, ..., a_p] such that
        x[n] = sum_{k=1}^{p} a_k x[n-k] + e[n].
    sigma2 : estimated innovation variance.
    """
    n = len(x)
    if order < 1 or n < order + 1:
        return np.zeros(max(order, 1)), float(np.var(x)) if len(x) > 0 else 1.0

    # Initialise forward/backward prediction errors
    ef = x[1:].copy().astype(np.float64)
    eb = x[:-1].copy().astype(np.float64)

    # Initial error power
    sigma2 = float(np.dot(x, x) / n)

    a = np.zeros(order, dtype=np.float64)

    for k in range(order):
        # Reflection coefficient
        num = -2.0 * np.dot(ef, eb)
        den = np.dot(ef, ef) + np.dot(eb, eb)
        if abs(den) < 1e-15:
            break
        kk = num / den

        # Update AR coefficients (Levinson recursion)
        if k == 0:
            a[0] = kk
        else:
            a_prev = a[:k].copy()
            for j in range(k):
                a[j] = a_prev[j] + kk * a_prev[k - 1 - j]
            a[k] = kk

        # Update error power
        sigma2 *= 1.0 - kk * kk
        if sigma2 < 1e-15:
            sigma2 = 1e-15

        # Update forward/backward errors
        ef_new = ef[1:] + kk * eb[1:]
        eb_new = eb[:-1] + kk * ef[:-1]

        if len(ef_new) == 0:
            break
        ef = ef_new
        eb = eb_new

    # Convention: x[n] = a[0]*x[n-1] + a[1]*x[n-2] + ...
    # Burg gives negative reflection coefficients in the prediction-error
    # filter; the AR coefficients are already in the right sign.
    return a, sigma2


# ---------------------------------------------------------------------------
# Spectral analysis helpers
# ---------------------------------------------------------------------------

def _ar_spectrum(
    a: np.ndarray, sigma2: float, n_freqs: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the power spectral density from AR coefficients.

    Parameters
    ----------
    a : (p,) AR coefficients.
    sigma2 : innovation variance.
    n_freqs : number of frequency bins in [0, pi].

    Returns
    -------
    freqs : (n_freqs,) angular frequencies in [0, pi].
    psd : (n_freqs,) power spectral density values.
    """
    freqs = np.linspace(0, np.pi, n_freqs, endpoint=False)
    p = len(a)
    if p == 0:
        return freqs, np.full(n_freqs, sigma2 / (2 * np.pi))

    psd = np.zeros(n_freqs)
    for i, omega in enumerate(freqs):
        # A(z) = 1 - sum_{k=1}^{p} a_k z^{-k}  evaluated at z = e^{j omega}
        z = np.exp(-1j * omega * np.arange(1, p + 1))
        A = 1.0 - np.dot(a, z)
        psd[i] = sigma2 / (2 * np.pi * max(abs(A) ** 2, 1e-15))

    return freqs, psd


def _prediction_efficiency(psd: np.ndarray) -> float:
    """Compute prediction efficiency eta = 1 - G/A from a PSD.

    G = geometric mean of PSD = exp(mean(log(PSD)))
    A = arithmetic mean of PSD

    Returns eta in [0, 1].  Returns 0 if computation fails.
    """
    psd_pos = psd[psd > 0]
    if len(psd_pos) == 0:
        return 0.0

    log_psd = np.log(psd_pos)
    G = np.exp(np.mean(log_psd))
    A = np.mean(psd_pos)

    if A < 1e-15:
        return 0.0

    eta = 1.0 - G / A
    return float(np.clip(eta, 0.0, 1.0))


def _ar_predict(x: np.ndarray, a: np.ndarray, steps: int = 1) -> float:
    """One-step (or multi-step) AR prediction.

    Parameters
    ----------
    x : recent observations [..., x_{t-p+1}, ..., x_{t}].
    a : AR coefficients [a_1, ..., a_p].
    steps : number of steps ahead to predict.

    Returns
    -------
    Predicted value at t + steps.
    """
    p = len(a)
    if len(x) < p or p == 0:
        return 0.0

    # Extend with predictions for multi-step
    buf = list(x[-p:])
    for _ in range(steps):
        pred = sum(a[k] * buf[-(k + 1)] for k in range(p))
        buf.append(pred)

    return float(buf[-1])


# ===========================================================================
# Strategy class
# ===========================================================================

class SzegoPredictionStrategy(Strategy):
    """Szego prediction-theoretic AR trading strategy.

    Parameters
    ----------
    ar_order : int
        Order of the AR model.  Default 10.
    lookback : int
        Rolling window (trading days) for AR estimation.  Default 120.
    eta_threshold : float
        Minimum prediction efficiency below which the strategy stays flat.
        Default 0.1.
    aggressiveness : float
        Scaling factor for how strongly eta amplifies signals.  Default 2.0.
    rebalance_freq : int
        Rebalance every N trading days.  Default 1 (daily).
    min_history : int
        Minimum observations before generating signals.  Default 30.
    n_freqs : int
        Number of frequency bins for spectral estimation.  Default 512.
    """

    def __init__(
        self,
        ar_order: int = 10,
        lookback: int = 120,
        eta_threshold: float = 0.1,
        aggressiveness: float = 2.0,
        rebalance_freq: int = 1,
        min_history: int = 30,
        n_freqs: int = 512,
    ) -> None:
        super().__init__(
            name="SzegoPrediction",
            description=(
                "Szego prediction-theoretic strategy using Burg's AR model "
                "with prediction efficiency for adaptive position sizing."
            ),
        )
        self.ar_order = ar_order
        self.lookback = lookback
        self.eta_threshold = eta_threshold
        self.aggressiveness = aggressiveness
        self.rebalance_freq = rebalance_freq
        self.min_history = min_history
        self.n_freqs = n_freqs

        # Learned parameters
        self._ar_coeffs: Optional[Dict[str, np.ndarray]] = None
        self._efficiencies: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def fit(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> "SzegoPredictionStrategy":
        """Fit AR models and compute prediction efficiencies.

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
                f"{self.min_history}.",
                stacklevel=2,
            )
            self._ar_coeffs = {}
            self._efficiencies = {}
            self._fitted = True
            return self

        log_ret = np.log(prices_clean / prices_clean.shift(1)).dropna()

        self._ar_coeffs = {}
        self._efficiencies = {}

        for col in log_ret.columns:
            r = log_ret[col].values.astype(np.float64)
            r = r[np.isfinite(r)]
            if len(r) < self.ar_order + 1:
                continue

            a, sigma2 = _burg_ar(r, self.ar_order)
            _, psd = _ar_spectrum(a, sigma2, self.n_freqs)
            eta = _prediction_efficiency(psd)

            self._ar_coeffs[col] = a
            self._efficiencies[col] = eta

        self.parameters = {
            "n_assets_fitted": len(self._ar_coeffs),
            "mean_efficiency": float(
                np.mean(list(self._efficiencies.values()))
                if self._efficiencies
                else 0.0
            ),
        }
        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate directional signals from AR predictions scaled by eta.

        Parameters
        ----------
        prices : pd.DataFrame
            Close prices, columns = tickers, index = dates.

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` (in {-1, 0, +1}) and
            ``{ticker}_weight`` (scaled by prediction efficiency).
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
            prev_weight = 0.0

            for t in range(n_dates):
                if t < self.min_history:
                    continue

                # Rebalance check
                if t % self.rebalance_freq != 0:
                    all_signals[t, j] = prev_signal
                    all_weights[t, j] = prev_weight
                    continue

                # Rolling window
                start = max(0, t - self.lookback)
                window = r[start:t]
                valid = window[np.isfinite(window)]

                if len(valid) < self.ar_order + 1:
                    all_signals[t, j] = 0.0
                    all_weights[t, j] = 0.0
                    prev_signal = 0.0
                    prev_weight = 0.0
                    continue

                # Fit AR model on rolling window
                a, sigma2 = _burg_ar(valid, self.ar_order)

                # Compute prediction efficiency
                _, psd = _ar_spectrum(a, sigma2, self.n_freqs)
                eta = _prediction_efficiency(psd)

                # If predictability is too low, stay flat
                if eta < self.eta_threshold:
                    all_signals[t, j] = 0.0
                    all_weights[t, j] = 0.0
                    prev_signal = 0.0
                    prev_weight = 0.0
                    continue

                # AR prediction: expected next-period return
                predicted_return = _ar_predict(valid, a, steps=1)

                # Direction
                if predicted_return > 0:
                    signal = 1.0
                elif predicted_return < 0:
                    signal = -1.0
                else:
                    signal = 0.0

                # Weight scaled by prediction efficiency
                # Higher eta -> more confident -> larger weight
                weight = np.clip(
                    eta * self.aggressiveness, 0.0, 1.0
                )

                all_signals[t, j] = signal
                all_weights[t, j] = weight
                prev_signal = signal
                prev_weight = weight

        # Build output
        signals_df = pd.DataFrame(index=prices.index)
        for j, ticker in enumerate(tickers):
            signals_df[f"{ticker}_signal"] = all_signals[:, j]
            signals_df[f"{ticker}_weight"] = all_weights[:, j]

        return signals_df
