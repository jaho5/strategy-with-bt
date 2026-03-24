"""Renyi entropy and divergence strategy for market regime detection.

Uses multi-order Renyi entropy to characterise the return distribution
and detect regime changes, then maps regimes to trading signals
(momentum, mean-reversion, or risk-off).

Mathematical foundation
-----------------------
Renyi entropy of order alpha (alpha > 0, alpha != 1):

    H_alpha(X) = (1 / (1 - alpha)) * log( sum_i  p_i^alpha )

Special cases:
    alpha -> 1 :  Shannon entropy  H(X) = -sum_i p_i log(p_i)
    alpha = 2  :  collision entropy H_2 = -log(sum_i p_i^2)
                  (related to the Herfindahl-Hirschman index)
    alpha -> inf: min-entropy  H_inf = -log(max_i p_i)

Renyi divergence of order alpha between distributions P and Q:

    D_alpha(P || Q) = (1 / (alpha - 1)) * log( sum_i  p_i^alpha  q_i^{1 - alpha} )

Strategy mechanics
------------------
1. **Return distribution entropy tracking**: discretise rolling-window
   returns into histogram bins and compute H_alpha for
   alpha in {0.5, 1, 2, inf} on a 63-day window.

2. **Multi-order entropy profile**: the shape of H_alpha(alpha) reveals
   distributional structure.  H_0.5 is sensitive to rare events
   (heavy tails); H_2 is sensitive to modal structure.  The
   "tail-heaviness spread" H_0.5 - H_2 tracks tail risk.

3. **Regime detection via Renyi divergence**: compute
   D_alpha(P_recent || P_historical) for multiple alpha.
   Large divergence signals a regime change.  Low-alpha divergence
   catches tail changes; high-alpha divergence catches mode changes.

4. **Trading rules**:
   - Low entropy + declining   -> trend forming   -> momentum signal
   - High entropy + stable     -> normal market    -> mean-revert signal
   - Entropy spike             -> regime change    -> reduce exposure
   - Tail-heaviness increasing -> hedge tail risk  -> scale down

References
----------
*   Renyi, A. (1961). On measures of entropy and information.
    Proc. Fourth Berkeley Symposium on Mathematics, Statistics and
    Probability, 1, 547-561.
*   van Erven, T. & Harremoes, P. (2014). Renyi divergence and
    Kullback-Leibler divergence. IEEE Trans. Information Theory 60(7).
*   Granger & Lin (1994). Using the mutual information coefficient to
    identify lags in nonlinear models. J. Time Series Analysis 15(4).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: Renyi entropy computation
# ---------------------------------------------------------------------------

def _discretise_returns(
    returns: np.ndarray,
    n_bins: int = 20,
    bin_edges: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretise continuous returns into a probability histogram.

    Parameters
    ----------
    returns : (N,) array of return observations.
    n_bins : number of histogram bins.
    bin_edges : if provided, use these edges instead of computing from data.

    Returns
    -------
    probs : (n_bins,) probability mass in each bin.
    edges : (n_bins + 1,) bin edge values.
    """
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        probs = np.full(n_bins, 1.0 / n_bins)
        if bin_edges is None:
            bin_edges = np.linspace(-0.1, 0.1, n_bins + 1)
        return probs, bin_edges

    if bin_edges is not None:
        counts, _ = np.histogram(returns, bins=bin_edges)
    else:
        counts, bin_edges = np.histogram(returns, bins=n_bins)

    total = counts.sum()
    if total == 0:
        probs = np.full(len(counts), 1.0 / len(counts))
    else:
        # Laplace smoothing to avoid zero probabilities
        probs = (counts + 1e-10) / (total + 1e-10 * len(counts))
        probs /= probs.sum()

    return probs, bin_edges


def _renyi_entropy(probs: np.ndarray, alpha: float) -> float:
    """Compute Renyi entropy H_alpha of a discrete probability distribution.

    Parameters
    ----------
    probs : (K,) probability vector (must sum to 1, entries > 0).
    alpha : order parameter.  Must be > 0.
        alpha = 1   -> Shannon entropy (via L'Hopital / limit)
        alpha = inf -> min-entropy

    Returns
    -------
    float : H_alpha in nats (natural log).
    """
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-15, None)
    p /= p.sum()  # re-normalise after clipping

    if np.isinf(alpha):
        # Min-entropy: H_inf = -log(max p_i)
        return -np.log(np.max(p))

    if np.abs(alpha - 1.0) < 1e-10:
        # Shannon entropy: H_1 = -sum p_i log(p_i)
        return -float(np.sum(p * np.log(p)))

    if alpha == 0.0:
        # Hartley entropy: H_0 = log(|support|)
        return np.log(np.sum(p > 1e-14))

    # General case: H_alpha = (1 / (1 - alpha)) * log(sum p_i^alpha)
    power_sum = np.sum(p ** alpha)
    if power_sum <= 0:
        return 0.0
    return float((1.0 / (1.0 - alpha)) * np.log(power_sum))


def _renyi_divergence(
    p: np.ndarray,
    q: np.ndarray,
    alpha: float,
) -> float:
    """Compute Renyi divergence D_alpha(P || Q).

    Parameters
    ----------
    p, q : (K,) probability vectors.
    alpha : order parameter (> 0, != 1).
        For alpha = 1 we return the KL divergence as the limit.

    Returns
    -------
    float : D_alpha(P || Q) in nats.  Non-negative; 0 iff P == Q.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 1e-15, None)
    q = np.clip(q, 1e-15, None)
    p /= p.sum()
    q /= q.sum()

    if np.abs(alpha - 1.0) < 1e-10:
        # KL divergence: D_1(P || Q) = sum p_i log(p_i / q_i)
        return float(np.sum(p * np.log(p / q)))

    if np.isinf(alpha):
        # D_inf(P || Q) = log(max_i  p_i / q_i)
        return float(np.log(np.max(p / q)))

    # General case:
    # D_alpha(P || Q) = (1 / (alpha - 1)) * log(sum p_i^alpha * q_i^{1 - alpha})
    integrand = np.sum((p ** alpha) * (q ** (1.0 - alpha)))
    if integrand <= 0:
        return 0.0
    return float((1.0 / (alpha - 1.0)) * np.log(integrand))


# ---------------------------------------------------------------------------
# Core: multi-order entropy profile
# ---------------------------------------------------------------------------

def _entropy_profile(
    probs: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """Compute H_alpha for a vector of alpha values.

    Parameters
    ----------
    probs : (K,) probability distribution.
    alphas : (M,) array of alpha orders.

    Returns
    -------
    (M,) array of H_alpha values.
    """
    return np.array([_renyi_entropy(probs, a) for a in alphas])


def _tail_heaviness_spread(
    probs: np.ndarray,
    alpha_low: float = 0.5,
    alpha_high: float = 2.0,
) -> float:
    """Compute the tail-heaviness indicator H_{alpha_low} - H_{alpha_high}.

    For a distribution with heavy tails, H_0.5 >> H_2 because low-order
    Renyi entropy up-weights rare events.  For a peaked/concentrated
    distribution, H_2 is small but H_0.5 may still be moderate.

    A large positive spread indicates heavy tails / fat-tail risk.
    """
    h_low = _renyi_entropy(probs, alpha_low)
    h_high = _renyi_entropy(probs, alpha_high)
    return h_low - h_high


# ---------------------------------------------------------------------------
# Core: regime classifier from entropy features
# ---------------------------------------------------------------------------

def _classify_entropy_regime(
    entropy_current: float,
    entropy_ma: float,
    entropy_std: float,
    entropy_slope: float,
    divergence: float,
    divergence_threshold: float,
    tail_spread: float,
    tail_spread_ma: float,
    tail_spread_std: float,
) -> tuple[int, float]:
    """Classify market regime from entropy features and return (signal, weight).

    Returns
    -------
    signal : int
        +1 (momentum / long), -1 (mean-revert / short bias), 0 (flat / risk-off).
    weight : float
        Conviction in [0, 1].
    """
    # Normalised entropy level (z-score relative to rolling stats)
    if entropy_std > 1e-10:
        z_entropy = (entropy_current - entropy_ma) / entropy_std
    else:
        z_entropy = 0.0

    # Normalised tail spread
    if tail_spread_std > 1e-10:
        z_tail = (tail_spread - tail_spread_ma) / tail_spread_std
    else:
        z_tail = 0.0

    # Rule 1: Entropy spike (large divergence) -> regime change -> risk-off
    if divergence > divergence_threshold:
        weight = np.clip(1.0 - divergence / (2.0 * divergence_threshold), 0.05, 0.3)
        return 0, weight

    # Rule 2: Tail heaviness increasing sharply -> hedge -> reduce position
    if z_tail > 1.5:
        # Tail risk is elevated; scale down
        weight = np.clip(0.5 - 0.1 * z_tail, 0.1, 0.5)
        return 0, weight

    # Rule 3: Low entropy + declining -> concentrated/trending -> momentum
    if z_entropy < -0.5 and entropy_slope < 0:
        # Strength of conviction based on how far below mean
        weight = np.clip(0.5 + 0.15 * abs(z_entropy), 0.3, 1.0)
        return 1, weight

    # Rule 4: High entropy + stable -> diffuse/normal -> mean-revert
    if z_entropy > 0.5 and abs(entropy_slope) < entropy_std * 0.5:
        weight = np.clip(0.5 + 0.1 * z_entropy, 0.3, 0.8)
        return -1, weight

    # Default: moderate conviction long (mild momentum bias)
    weight = 0.3
    return 1, weight


# ===========================================================================
# Strategy class
# ===========================================================================

class RenyiEntropyStrategy(Strategy):
    """Renyi entropy and divergence strategy for market regime detection.

    Uses multi-order Renyi entropy computed on rolling return distributions
    to detect trending, mean-reverting, and regime-change states, then
    maps each state to an appropriate trading posture.

    Parameters
    ----------
    window : int
        Rolling window (trading days) for return distribution estimation.
        Default 63 (~1 quarter).
    n_bins : int
        Number of histogram bins for discretising returns.
        Default 20.
    alphas : tuple[float, ...]
        Renyi entropy orders to compute.
        Default (0.5, 1.0, 2.0, inf).
    divergence_window : int
        Length of the "recent" window for divergence computation.
        Default 21 (~1 month).
    divergence_threshold : float
        Multiplier on the rolling MAD of divergence for adaptive
        regime-change detection.  Divergence above
        median + threshold * MAD triggers risk-off.
        Default 3.0.
    entropy_ma_window : int
        Window for the moving average / std of entropy (used to
        z-score the current level).
        Default 126 (~6 months).
    signal_smoothing_span : int
        EMA span for smoothing the raw signal.
        Default 5.
    min_history : int
        Minimum bars before producing non-trivial signals.
        Default 126.
    """

    def __init__(
        self,
        window: int = 63,
        n_bins: int = 20,
        alphas: tuple[float, ...] = (0.5, 1.0, 2.0, np.inf),
        divergence_window: int = 21,
        divergence_threshold: float = 3.0,
        entropy_ma_window: int = 126,
        signal_smoothing_span: int = 5,
        min_history: int = 126,
    ) -> None:
        super().__init__(
            name="RenyiEntropy",
            description=(
                "Market regime detection via multi-order Renyi entropy "
                "and divergence on rolling return distributions.  Maps "
                "entropy regimes to momentum, mean-reversion, or risk-off "
                "signals."
            ),
        )
        self.window = window
        self.n_bins = n_bins
        self.alphas = np.array(alphas, dtype=np.float64)
        self.divergence_window = divergence_window
        self.divergence_threshold = divergence_threshold
        self.entropy_ma_window = entropy_ma_window
        self.signal_smoothing_span = signal_smoothing_span
        self.min_history = min_history

        # Populated by fit()
        self._historical_bin_edges: Optional[np.ndarray] = None
        self._historical_probs: Optional[np.ndarray] = None
        self._baseline_entropy_profile: Optional[np.ndarray] = None

        # Diagnostic series populated by generate_signals()
        self._entropy_series: Optional[Dict[float, pd.Series]] = None
        self._divergence_series: Optional[Dict[float, pd.Series]] = None
        self._tail_spread_series: Optional[pd.Series] = None
        self._regime_series: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> "RenyiEntropyStrategy":
        """Calibrate historical return distribution and entropy baseline.

        Computes the full-sample return histogram and entropy profile
        which serve as the reference distribution for divergence
        calculations and the baseline for entropy z-scoring.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (DatetimeIndex, columns = tickers).
            If multiple columns, uses the first column.

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

        # Build the reference histogram using the full training set
        self._historical_probs, self._historical_bin_edges = _discretise_returns(
            log_returns, n_bins=self.n_bins
        )

        # Baseline entropy profile
        self._baseline_entropy_profile = _entropy_profile(
            self._historical_probs, self.alphas
        )

        self.parameters = {
            "window": self.window,
            "n_bins": self.n_bins,
            "alphas": list(self.alphas),
            "divergence_window": self.divergence_window,
            "divergence_threshold": self.divergence_threshold,
            "entropy_ma_window": self.entropy_ma_window,
            "baseline_entropy_profile": {
                float(a): float(h)
                for a, h in zip(self.alphas, self._baseline_entropy_profile)
            },
        }

        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals based on Renyi entropy regime detection.

        For each date (after sufficient history):
        1. Compute the return distribution over the trailing ``window``.
        2. Compute H_alpha for each alpha in ``self.alphas``.
        3. Compute Renyi divergence between the recent window and the
           full trailing window.
        4. Compute the tail-heaviness spread H_0.5 - H_2.
        5. Classify the regime and emit a signal + weight.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (DatetimeIndex, columns = tickers).
            Uses the first column for single-instrument analysis.

        Returns
        -------
        pd.DataFrame
            Columns ``signal`` in {-1, 0, 1} and ``weight`` in [0, 1].
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        if prices.ndim == 2:
            series = prices.iloc[:, 0]
        else:
            series = prices

        log_returns = np.log(series / series.shift(1))
        n = len(prices)

        # Pre-allocate output arrays
        signals = np.zeros(n, dtype=np.int64)
        weights = np.zeros(n, dtype=np.float64)

        # Diagnostic storage
        entropy_dict: Dict[float, np.ndarray] = {
            float(a): np.full(n, np.nan) for a in self.alphas
        }
        divergence_dict: Dict[float, np.ndarray] = {
            float(a): np.full(n, np.nan) for a in self.alphas
        }
        tail_spread_arr = np.full(n, np.nan)

        # Determine alpha indices for tail spread
        has_alpha_half = 0.5 in self.alphas
        has_alpha_two = 2.0 in self.alphas

        for t in range(n):
            if t < self.min_history:
                continue

            # --- 1. Full-window return distribution ---
            window_start = max(0, t - self.window + 1)
            window_returns = log_returns.iloc[window_start : t + 1].values
            probs_full, edges_full = _discretise_returns(
                window_returns, n_bins=self.n_bins
            )

            # --- 2. Multi-order entropy ---
            for a in self.alphas:
                h = _renyi_entropy(probs_full, a)
                entropy_dict[float(a)][t] = h

            # --- 3. Renyi divergence: recent vs full window ---
            # Use fewer bins for divergence to reduce noise from sparse
            # recent histograms (divergence_window << window).
            div_n_bins = max(8, self.n_bins // 2)
            recent_start = max(0, t - self.divergence_window + 1)
            recent_returns = log_returns.iloc[recent_start : t + 1].values

            # Build both histograms on the same bin edges derived from
            # the full window, but with fewer bins for stability.
            _, div_edges = _discretise_returns(
                window_returns, n_bins=div_n_bins
            )
            probs_recent_div, _ = _discretise_returns(
                recent_returns, n_bins=div_n_bins, bin_edges=div_edges
            )
            probs_full_div, _ = _discretise_returns(
                window_returns, n_bins=div_n_bins, bin_edges=div_edges
            )

            for a in self.alphas:
                d = _renyi_divergence(probs_recent_div, probs_full_div, a)
                divergence_dict[float(a)][t] = d

            # --- 4. Tail-heaviness spread ---
            if has_alpha_half and has_alpha_two:
                ts = _tail_heaviness_spread(probs_full, 0.5, 2.0)
            else:
                # Fallback: use first and last alpha
                ts = _tail_heaviness_spread(
                    probs_full, float(self.alphas[0]), float(self.alphas[-1])
                )
            tail_spread_arr[t] = ts

        # --- Compute rolling statistics for z-scoring ---
        # Use Shannon entropy (alpha=1) as the primary regime indicator
        primary_alpha = 1.0
        if primary_alpha not in [float(a) for a in self.alphas]:
            primary_alpha = float(self.alphas[0])

        entropy_series = pd.Series(entropy_dict[primary_alpha], index=prices.index)
        entropy_ma = entropy_series.rolling(
            self.entropy_ma_window, min_periods=self.min_history
        ).mean()
        entropy_std = entropy_series.rolling(
            self.entropy_ma_window, min_periods=self.min_history
        ).std()

        # Entropy slope: regression slope over a short window
        slope_window = 21
        entropy_slope = entropy_series.rolling(slope_window, min_periods=10).apply(
            lambda x: _linear_slope(x.values), raw=False
        )

        tail_spread_series = pd.Series(tail_spread_arr, index=prices.index)
        tail_spread_ma = tail_spread_series.rolling(
            self.entropy_ma_window, min_periods=self.min_history
        ).mean()
        tail_spread_std = tail_spread_series.rolling(
            self.entropy_ma_window, min_periods=self.min_history
        ).std()

        # Use the KL divergence (alpha=1) as the primary regime-change
        # detector since it is the most well-behaved and interpretable.
        # Fall back to average across alphas if alpha=1 is not available.
        if 1.0 in divergence_dict:
            avg_divergence = divergence_dict[1.0]
        else:
            avg_divergence = np.nanmean(
                np.column_stack(
                    [divergence_dict[float(a)] for a in self.alphas
                     if not np.isinf(a)]
                ),
                axis=1,
            )

        # Adaptive divergence threshold: use rolling median + k * MAD
        # so the threshold adjusts to the baseline divergence level.
        div_series = pd.Series(avg_divergence, index=prices.index)
        div_median = div_series.rolling(
            self.entropy_ma_window, min_periods=self.min_history
        ).median()
        div_mad = (div_series - div_median).abs().rolling(
            self.entropy_ma_window, min_periods=self.min_history
        ).median()

        # --- Classify regime and assign signals ---
        for t in range(n):
            if t < self.min_history:
                continue

            e_curr = entropy_series.iloc[t]
            e_ma = entropy_ma.iloc[t]
            e_std = entropy_std.iloc[t]
            e_slope = entropy_slope.iloc[t]
            div_val = avg_divergence[t]
            ts_val = tail_spread_arr[t]
            ts_ma = tail_spread_ma.iloc[t]
            ts_std = tail_spread_std.iloc[t]

            # Adaptive threshold: median + divergence_threshold_factor * MAD
            # The self.divergence_threshold acts as a multiplier on MAD
            d_med = div_median.iloc[t]
            d_mad = div_mad.iloc[t]
            if not np.isnan(d_med) and not np.isnan(d_mad) and d_mad > 1e-10:
                adaptive_div_thresh = d_med + self.divergence_threshold * d_mad
            else:
                adaptive_div_thresh = self.divergence_threshold

            if np.isnan(e_curr) or np.isnan(e_ma) or np.isnan(e_std):
                continue

            e_slope_safe = e_slope if not np.isnan(e_slope) else 0.0
            div_safe = div_val if not np.isnan(div_val) else 0.0
            ts_ma_safe = ts_ma if not np.isnan(ts_ma) else ts_val
            ts_std_safe = ts_std if not np.isnan(ts_std) else 1e-10

            sig, wgt = _classify_entropy_regime(
                entropy_current=e_curr,
                entropy_ma=e_ma,
                entropy_std=e_std,
                entropy_slope=e_slope_safe,
                divergence=div_safe,
                divergence_threshold=adaptive_div_thresh,
                tail_spread=ts_val,
                tail_spread_ma=ts_ma_safe,
                tail_spread_std=ts_std_safe,
            )
            signals[t] = sig
            weights[t] = wgt

        # --- Smooth signals ---
        raw_signals = pd.Series(signals, index=prices.index, dtype=np.float64)
        smoothed = self.exponential_smooth(raw_signals, span=self.signal_smoothing_span)
        # Re-discretise: threshold at +/-0.3 for signal, keep weight continuous
        final_signals = np.where(
            smoothed > 0.3, 1, np.where(smoothed < -0.3, -1, 0)
        )

        raw_weights = pd.Series(weights, index=prices.index)
        smoothed_weights = self.exponential_smooth(
            raw_weights, span=self.signal_smoothing_span
        )
        final_weights = np.clip(smoothed_weights.values, 0.0, 1.0)

        # --- Store diagnostics ---
        self._entropy_series = {
            a: pd.Series(entropy_dict[a], index=prices.index, name=f"H_{a}")
            for a in entropy_dict
        }
        self._divergence_series = {
            a: pd.Series(divergence_dict[a], index=prices.index, name=f"D_{a}")
            for a in divergence_dict
        }
        self._tail_spread_series = tail_spread_series
        self._regime_series = pd.Series(
            final_signals, index=prices.index, name="regime_signal"
        )

        # --- Build output DataFrame ---
        result = pd.DataFrame(index=prices.index)
        result["signal"] = final_signals.astype(int)
        result["weight"] = final_weights

        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_entropy_series(self) -> Optional[Dict[float, pd.Series]]:
        """Return the rolling Renyi entropy series for each alpha.

        Returns
        -------
        dict[float, pd.Series] or None
            Keyed by alpha value.  None if ``generate_signals`` has not
            been called.
        """
        return self._entropy_series

    def get_divergence_series(self) -> Optional[Dict[float, pd.Series]]:
        """Return the rolling Renyi divergence series for each alpha.

        Returns
        -------
        dict[float, pd.Series] or None
            Keyed by alpha value.  None if ``generate_signals`` has not
            been called.
        """
        return self._divergence_series

    def get_tail_spread(self) -> Optional[pd.Series]:
        """Return the tail-heaviness spread H_0.5 - H_2 series.

        A large positive value indicates fat tails in the recent return
        distribution.

        Returns
        -------
        pd.Series or None
        """
        return self._tail_spread_series

    def get_regime_series(self) -> Optional[pd.Series]:
        """Return the classified regime signal series.

        Values: +1 (momentum), -1 (mean-revert), 0 (risk-off).

        Returns
        -------
        pd.Series or None
        """
        return self._regime_series

    def get_entropy_profile_snapshot(
        self, prices: pd.DataFrame, as_of: int = -1
    ) -> Dict[str, Any]:
        """Compute a point-in-time entropy profile for diagnostics.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        as_of : int
            Index position for the snapshot (default: last bar).

        Returns
        -------
        dict with keys:
            'alphas': list of alpha values
            'entropies': list of H_alpha values
            'tail_spread': float
            'divergence_vs_historical': list of D_alpha values
        """
        self.ensure_fitted()

        if prices.ndim == 2:
            series = prices.iloc[:, 0]
        else:
            series = prices

        log_returns = np.log(series / series.shift(1)).dropna().values
        window_returns = log_returns[max(0, as_of - self.window + 1) : as_of + 1]

        probs, edges = _discretise_returns(window_returns, n_bins=self.n_bins)
        profile = _entropy_profile(probs, self.alphas)

        # Divergence against historical baseline
        divs = []
        for a in self.alphas:
            d = _renyi_divergence(probs, self._historical_probs, a)
            divs.append(d)

        ts = _tail_heaviness_spread(probs, 0.5, 2.0)

        return {
            "alphas": [float(a) for a in self.alphas],
            "entropies": [float(h) for h in profile],
            "tail_spread": float(ts),
            "divergence_vs_historical": [float(d) for d in divs],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear_slope(y: np.ndarray) -> float:
    """OLS slope of y against integer indices 0, 1, ..., len(y)-1.

    Parameters
    ----------
    y : (N,) array.

    Returns
    -------
    float : slope coefficient.
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 2 or np.any(np.isnan(y)):
        return np.nan
    x = np.arange(len(y), dtype=np.float64)
    x_dm = x - x.mean()
    denom = x_dm @ x_dm
    if denom == 0:
        return 0.0
    return float(x_dm @ (y - y.mean()) / denom)
