"""Topological Data Analysis (TDA) strategy for crash detection.

Uses persistent-homology-inspired techniques to detect regime changes
in financial time series without requiring a full TDA library.

Mathematical foundation
-----------------------
Persistent homology tracks topological features (connected components,
loops, voids) across scales in a filtration of simplicial complexes.

For financial time series:
*   Embed returns in delay-coordinate space (Takens' embedding theorem):
        x_t = (r_t, r_{t-tau}, ..., r_{t-(d-1)*tau})
*   Build Vietoris-Rips complex at varying scale epsilon
*   Track birth/death of topological features
*   Persistence diagram: {(birth_i, death_i)} for each feature
*   Features with long persistence = real structure; short = noise

Implementation (without full TDA library)
------------------------------------------
We approximate topological summaries via the spectrum of the Gram
matrix of the delay-embedded point cloud:

1.  **Spectral gap** (lambda_1 - lambda_2) / lambda_1 measures
    clustering / connectedness (H0 proxy).
2.  **Spectral entropy** H = -sum p_i log(p_i), where p_i = lambda_i / sum(lambda_j),
    measures geometric complexity.
    *   High entropy -> uniform eigenvalue spread -> normal market
    *   Low entropy -> one dominant mode -> trend / bubble / pre-crash
3.  **Wasserstein proxy**: rolling change in the sorted eigenvalue
    distribution approximates Wasserstein distance between persistence
    diagrams across time windows.

Signal logic
------------
*   Normal regime (high entropy): use momentum signal (20-day return sign)
*   Stressed regime (low entropy): reduce or reverse positions
*   Transition detection: large entropy change -> go flat temporarily
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core TDA-inspired building blocks
# ---------------------------------------------------------------------------

def _delay_embedding(
    returns: np.ndarray,
    delay: int = 5,
    dimension: int = 3,
) -> np.ndarray:
    """Construct a delay-coordinate embedding of the return series.

    Given a 1-D time series r_0, r_1, ..., r_{N-1}, produces vectors:

        x_t = (r_t, r_{t-tau}, r_{t-2*tau}, ..., r_{t-(d-1)*tau})

    for t = (d-1)*tau, ..., N-1.

    Parameters
    ----------
    returns : 1-D array of log returns.
    delay : int
        Delay parameter tau (default 5 trading days).
    dimension : int
        Embedding dimension d (default 3 -> point cloud in R^3).

    Returns
    -------
    np.ndarray of shape (n_points, dimension) -- the embedded point cloud.
    """
    n = len(returns)
    start = (dimension - 1) * delay
    if start >= n:
        return np.empty((0, dimension))

    n_points = n - start
    embedded = np.empty((n_points, dimension))
    for k in range(dimension):
        offset = start - k * delay
        embedded[:, k] = returns[offset : offset + n_points]
    return embedded


def _gram_matrix(points: np.ndarray) -> np.ndarray:
    """Compute the Gram (inner product) matrix of a point cloud.

    G_{ij} = <x_i, x_j>

    The eigenvalues of G encode geometric structure: distances,
    curvature, and topological complexity of the underlying space.
    """
    return points @ points.T


def _pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute the pairwise Euclidean distance matrix.

    D_{ij} = ||x_i - x_j||_2

    Used for H0 (connected components) analysis at varying scales.
    """
    # Using the identity ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    sq_norms = np.sum(points ** 2, axis=1)
    D_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (points @ points.T)
    # Numerical safety: clamp negatives from floating-point errors
    np.maximum(D_sq, 0.0, out=D_sq)
    return np.sqrt(D_sq)


def _spectral_summary(
    points: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Compute topological summary statistics from the point cloud spectrum.

    Parameters
    ----------
    points : (n_points, dimension) array -- the delay-embedded cloud.

    Returns
    -------
    spectral_gap : float
        (lambda_1 - lambda_2) / lambda_1, measuring dominant-mode
        concentration.  Values near 1 -> single dominant cluster;
        near 0 -> uniform structure.
    spectral_entropy : float
        Shannon entropy of the normalised eigenvalue distribution.
        High -> uniform (normal market); low -> concentrated
        (trend / bubble / pre-crash).
    eigenvalues : 1-D array
        Sorted (descending) eigenvalues of the Gram matrix.
    """
    G = _gram_matrix(points)
    # Eigenvalues of the symmetric Gram matrix
    eigvals = np.linalg.eigvalsh(G)
    # Sort descending
    eigvals = eigvals[::-1]
    # Clamp negatives (numerical noise)
    eigvals = np.maximum(eigvals, 0.0)

    total = eigvals.sum()
    if total < 1e-15:
        return 0.0, 0.0, eigvals

    # Spectral gap
    lambda_1 = eigvals[0]
    lambda_2 = eigvals[1] if len(eigvals) > 1 else 0.0
    spectral_gap = (lambda_1 - lambda_2) / lambda_1 if lambda_1 > 1e-15 else 0.0

    # Spectral entropy: H = -sum p_i * log(p_i)
    p = eigvals / total
    # Filter out zeros to avoid log(0)
    p_pos = p[p > 1e-15]
    spectral_entropy = -np.sum(p_pos * np.log(p_pos))

    return spectral_gap, spectral_entropy, eigvals


def _h0_components(
    points: np.ndarray,
    n_thresholds: int = 20,
) -> np.ndarray:
    """Approximate H0 (connected components) persistence via distance thresholds.

    Sweeps through distance thresholds from 0 to max(D) and counts the
    number of connected components at each scale (using union-find).
    Returns the persistence profile: number of components at each threshold.

    This approximates the Betti-0 curve of the Vietoris-Rips filtration.
    """
    D = _pairwise_distance_matrix(points)
    n = len(points)
    max_dist = D.max()
    if max_dist < 1e-15 or n < 2:
        return np.ones(n_thresholds)

    thresholds = np.linspace(0.0, max_dist, n_thresholds)
    component_counts = np.empty(n_thresholds)

    for ti, eps in enumerate(thresholds):
        # Union-find to count connected components at scale eps
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] <= eps:
                    union(i, j)

        n_components = len(set(find(i) for i in range(n)))
        component_counts[ti] = n_components

    return component_counts


def _topological_complexity(
    points: np.ndarray,
    n_thresholds: int = 20,
) -> float:
    """Approximate total persistence (topological complexity) from H0.

    Total persistence = sum over all features of (death - birth).
    For H0 features, birth = 0 and death = the threshold at which two
    components merge.  We approximate this by integrating the Betti-0
    curve (number of components vs. threshold).

    Intuitively, this measures how much topological structure exists
    across scales.
    """
    h0 = _h0_components(points, n_thresholds)
    # Total persistence ~ area under the Betti curve
    # (the integral of #components over the filtration parameter)
    # Normalise by number of points so it's scale-invariant
    n = len(points)
    if n < 2:
        return 0.0
    # np.trapezoid replaces the deprecated np.trapz in NumPy >= 2.0
    _integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(_integrate(h0 / n, dx=1.0 / n_thresholds))


def _wasserstein_proxy(
    eigvals_prev: np.ndarray,
    eigvals_curr: np.ndarray,
) -> float:
    """Approximate 1-Wasserstein distance between persistence diagrams.

    We use the sorted normalised eigenvalue distributions as surrogates
    for persistence diagrams.  The 1-Wasserstein distance between two
    1-D distributions is simply the L1 distance between their quantile
    functions, which for discrete distributions of the same size is:

        W_1 = sum |F^{-1}_P(i/n) - F^{-1}_Q(i/n)|

    We normalise eigenvalues to sum to 1 (probability distributions)
    and pad the shorter vector with zeros.
    """
    # Normalise
    s1 = eigvals_prev.sum()
    s2 = eigvals_curr.sum()
    p = eigvals_prev / s1 if s1 > 1e-15 else eigvals_prev
    q = eigvals_curr / s2 if s2 > 1e-15 else eigvals_curr

    # Pad to equal length
    max_len = max(len(p), len(q))
    p_padded = np.zeros(max_len)
    q_padded = np.zeros(max_len)
    p_padded[: len(p)] = np.sort(p)[::-1]
    q_padded[: len(q)] = np.sort(q)[::-1]

    return float(np.sum(np.abs(p_padded - q_padded)))


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class TopologicalStrategy(Strategy):
    """Crash detection strategy based on Topological Data Analysis.

    Uses delay-coordinate embedding and spectral analysis of the
    resulting point cloud to detect regime changes that precede crashes.

    Parameters
    ----------
    embedding_delay : int
        Delay parameter tau for Takens embedding (default 5).
    embedding_dimension : int
        Dimension d of the embedding space (default 3).
    rolling_window : int
        Number of days in the rolling analysis window (default 60).
    momentum_lookback : int
        Lookback for the momentum signal in normal regime (default 20).
    entropy_smooth_span : int
        EMA span for smoothing the spectral entropy series (default 10).
    entropy_threshold_low : float
        Percentile (of historical entropy) below which the market is
        classified as stressed (default 25.0).
    transition_threshold : float
        Minimum absolute change in smoothed entropy (as a fraction of
        its rolling std) to trigger a transition signal (default 2.0).
    transition_cooldown : int
        Number of days to remain flat after a transition is detected
        (default 5).
    stressed_signal : float
        Position weight in stressed regime.  Negative means short
        (default -0.5 for a defensive tilt).
    n_thresholds : int
        Number of distance thresholds for H0 persistence (default 15).
    complexity_weight : float
        Blending weight for topological complexity in regime score
        (default 0.3).
    """

    def __init__(
        self,
        embedding_delay: int = 5,
        embedding_dimension: int = 3,
        rolling_window: int = 30,
        momentum_lookback: int = 20,
        entropy_smooth_span: int = 10,
        entropy_threshold_low: float = 25.0,
        transition_threshold: float = 2.0,
        transition_cooldown: int = 5,
        stressed_signal: float = -0.5,
        n_thresholds: int = 15,
        complexity_weight: float = 0.3,
    ) -> None:
        super().__init__(
            name="TopologicalCrashDetection",
            description=(
                "TDA-inspired crash detection using persistent-homology "
                "proxies (spectral entropy, Betti-0 complexity, Wasserstein "
                "distance) on delay-embedded return point clouds."
            ),
        )
        self.embedding_delay = embedding_delay
        self.embedding_dimension = embedding_dimension
        self.rolling_window = rolling_window
        self.momentum_lookback = momentum_lookback
        self.entropy_smooth_span = entropy_smooth_span
        self.entropy_threshold_low = entropy_threshold_low
        self.transition_threshold = transition_threshold
        self.transition_cooldown = transition_cooldown
        self.stressed_signal = stressed_signal
        self.n_thresholds = n_thresholds
        self.complexity_weight = complexity_weight

        # Populated during fit()
        self.parameters: Dict[str, Any] = {}

    # -----------------------------------------------------------------
    # Minimum data requirements
    # -----------------------------------------------------------------

    @property
    def _min_history(self) -> int:
        """Minimum number of observations needed before producing signals."""
        embedding_overhead = (self.embedding_dimension - 1) * self.embedding_delay
        return self.rolling_window + embedding_overhead + 1

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _compute_rolling_topology(
        self,
        log_returns: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute rolling topological features from log returns.

        Returns arrays aligned with the input (NaN-padded at the start):
        - spectral_gap: rolling spectral gap series
        - spectral_entropy: rolling spectral entropy series
        - complexity: rolling topological complexity series
        - wasserstein: rolling Wasserstein-proxy distance series
        """
        n = len(log_returns)
        spectral_gap = np.full(n, np.nan)
        spectral_entropy = np.full(n, np.nan)
        complexity = np.full(n, np.nan)
        wasserstein = np.full(n, np.nan)

        prev_eigvals: Optional[np.ndarray] = None
        embed_overhead = (self.embedding_dimension - 1) * self.embedding_delay
        min_start = embed_overhead + self.rolling_window

        for t in range(min_start, n):
            window_returns = log_returns[t - self.rolling_window : t]
            points = _delay_embedding(
                window_returns,
                delay=self.embedding_delay,
                dimension=self.embedding_dimension,
            )
            if len(points) < 3:
                continue

            gap, entropy, eigvals = _spectral_summary(points)
            spectral_gap[t] = gap
            spectral_entropy[t] = entropy

            # Topological complexity (H0 persistence integral)
            cpx = _topological_complexity(points, n_thresholds=self.n_thresholds)
            complexity[t] = cpx

            # Wasserstein proxy between successive windows
            if prev_eigvals is not None:
                w_dist = _wasserstein_proxy(prev_eigvals, eigvals)
                wasserstein[t] = w_dist
            prev_eigvals = eigvals.copy()

        return spectral_gap, spectral_entropy, complexity, wasserstein

    def _classify_regime(
        self,
        spectral_entropy: np.ndarray,
        complexity: np.ndarray,
        wasserstein: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify each time step into normal / stressed / transition.

        Returns
        -------
        regime : array of int
            0 = normal, 1 = stressed, 2 = transition (flat)
        regime_score : array of float
            Continuous regime score in [0, 1]; 0 = fully normal, 1 = fully stressed.
        """
        n = len(spectral_entropy)
        regime = np.zeros(n, dtype=int)
        regime_score = np.full(n, np.nan)

        # Smooth entropy for cleaner regime classification
        entropy_series = pd.Series(spectral_entropy)
        smoothed_entropy = entropy_series.ewm(
            span=self.entropy_smooth_span, min_periods=1, ignore_na=True
        ).mean().values

        # Rolling statistics of entropy for adaptive thresholding
        entropy_df = pd.Series(smoothed_entropy)
        rolling_median = entropy_df.expanding(min_periods=20).median().values
        rolling_std = entropy_df.expanding(min_periods=20).std().values

        # Rolling percentile rank of current entropy
        entropy_pctile = entropy_df.expanding(min_periods=20).apply(
            lambda x: (x.iloc[-1] <= x).mean() * 100.0 if len(x) > 0 else 50.0,
            raw=False,
        ).values

        # Smooth complexity for regime scoring
        complexity_series = pd.Series(complexity)
        smoothed_complexity = complexity_series.ewm(
            span=self.entropy_smooth_span, min_periods=1, ignore_na=True
        ).mean().values

        # Normalise complexity to [0, 1] via expanding rank
        complexity_pctile = complexity_series.expanding(min_periods=20).apply(
            lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else 0.5,
            raw=False,
        ).values

        # Entropy change for transition detection
        entropy_change = np.abs(np.diff(smoothed_entropy, prepend=smoothed_entropy[0]))

        cooldown_remaining = 0

        for t in range(n):
            if np.isnan(spectral_entropy[t]):
                regime[t] = 0  # default normal when no data
                regime_score[t] = 0.0
                continue

            # Regime score: blend of entropy percentile and complexity percentile
            # Low entropy percentile -> stressed; high complexity -> stressed
            e_score = 1.0 - entropy_pctile[t] / 100.0 if not np.isnan(entropy_pctile[t]) else 0.0
            c_score = complexity_pctile[t] if not np.isnan(complexity_pctile[t]) else 0.0
            score = (1.0 - self.complexity_weight) * e_score + self.complexity_weight * c_score
            regime_score[t] = np.clip(score, 0.0, 1.0)

            # Transition detection: large entropy change relative to its std
            if not np.isnan(rolling_std[t]) and rolling_std[t] > 1e-10:
                normalised_change = entropy_change[t] / rolling_std[t]
                if normalised_change > self.transition_threshold:
                    cooldown_remaining = self.transition_cooldown
                    regime[t] = 2  # transition
                    continue

            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                regime[t] = 2  # still in cooldown
                continue

            # Stressed vs normal based on entropy percentile
            if entropy_pctile[t] < self.entropy_threshold_low:
                regime[t] = 1  # stressed
            else:
                regime[t] = 0  # normal

        return regime, regime_score

    def _momentum_signal(self, prices: np.ndarray) -> np.ndarray:
        """Simple momentum signal: sign of trailing return over lookback."""
        n = len(prices)
        signal = np.full(n, np.nan)
        for t in range(self.momentum_lookback, n):
            ret = (prices[t] - prices[t - self.momentum_lookback]) / prices[t - self.momentum_lookback]
            signal[t] = np.sign(ret)
        return signal

    def _generate_signals_single(
        self,
        prices: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate signal and weight arrays for a single asset.

        Returns
        -------
        signal : array of float in {-1, 0, +1}
        weight : array of float in [0, 1]
        """
        price_vals = prices.values.astype(np.float64)
        n = len(price_vals)
        signal = np.zeros(n)
        weight = np.zeros(n)

        if n < self._min_history:
            return signal, weight

        # Log returns
        log_returns = np.diff(np.log(price_vals))

        # Compute rolling topological features
        gap, entropy, complexity, wass = self._compute_rolling_topology(log_returns)

        # Classify regime
        regime, regime_score = self._classify_regime(entropy, complexity, wass)

        # Momentum signal on price levels (for normal-regime signal)
        mom = self._momentum_signal(price_vals)

        # Map regimes to positions
        # log_returns is 1 shorter than prices, so topological features
        # are indexed into log_returns (offset by 1 relative to prices)
        for t in range(1, n):
            rt = t - 1  # index into log_returns / topological arrays
            if rt >= len(regime):
                break

            r = regime[rt]
            rs = regime_score[rt] if not np.isnan(regime_score[rt]) else 0.0

            if r == 2:
                # Transition: go flat
                signal[t] = 0.0
                weight[t] = 0.0
            elif r == 1:
                # Stressed regime: defensive position
                # Use the regime score to scale the stressed signal
                signal[t] = np.sign(self.stressed_signal)
                weight[t] = abs(self.stressed_signal) * rs
            else:
                # Normal regime: follow momentum
                if not np.isnan(mom[t]):
                    signal[t] = mom[t]
                    # Weight inversely proportional to regime score
                    # (less conviction as we approach stressed territory)
                    weight[t] = np.clip(1.0 - rs, 0.1, 1.0)
                else:
                    signal[t] = 0.0
                    weight[t] = 0.0

        return signal, weight

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "TopologicalStrategy":
        """Calibrate strategy parameters on historical data.

        Estimates the entropy threshold adaptively from training data
        by computing the full topological feature history and storing
        key distribution statistics.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (columns = tickers, index = dates).

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        entropy_values: list[float] = []
        complexity_values: list[float] = []

        for col in prices.columns:
            series = prices[col].dropna()
            if len(series) < self._min_history:
                logger.warning(
                    "Ticker %s has insufficient history (%d < %d); skipping.",
                    col, len(series), self._min_history,
                )
                continue

            price_vals = series.values.astype(np.float64)
            log_returns = np.diff(np.log(price_vals))

            _, entropy, complexity, _ = self._compute_rolling_topology(log_returns)

            valid_entropy = entropy[np.isfinite(entropy)]
            valid_complexity = complexity[np.isfinite(complexity)]

            if len(valid_entropy) > 0:
                entropy_values.extend(valid_entropy.tolist())
            if len(valid_complexity) > 0:
                complexity_values.extend(valid_complexity.tolist())

        # Store learned distribution statistics
        if entropy_values:
            ent_arr = np.array(entropy_values)
            self.parameters["entropy_median"] = float(np.median(ent_arr))
            self.parameters["entropy_std"] = float(np.std(ent_arr))
            self.parameters["entropy_q25"] = float(np.percentile(ent_arr, 25))
            self.parameters["entropy_q75"] = float(np.percentile(ent_arr, 75))
            # Adaptively set threshold from training data
            self.entropy_threshold_low = float(
                np.percentile(
                    ent_arr,
                    self.entropy_threshold_low,  # use initial value as percentile
                )
            )
            # After fit, store as absolute threshold in parameters
            self.parameters["entropy_threshold_fitted"] = self.entropy_threshold_low
            logger.info(
                "Fitted entropy threshold: %.4f (median=%.4f, std=%.4f)",
                self.entropy_threshold_low,
                self.parameters["entropy_median"],
                self.parameters["entropy_std"],
            )

        if complexity_values:
            cpx_arr = np.array(complexity_values)
            self.parameters["complexity_median"] = float(np.median(cpx_arr))
            self.parameters["complexity_std"] = float(np.std(cpx_arr))

        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals from price data.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (columns = tickers, index = dates).  If a single
            column, output uses the ``signal`` / ``weight`` convention;
            if multiple columns, uses ``{ticker}_signal`` / ``{ticker}_weight``.

        Returns
        -------
        pd.DataFrame
            Signal and weight columns for each asset.
        """
        self.validate_prices(prices)

        result = pd.DataFrame(index=prices.index)

        if len(prices.columns) == 1:
            col = prices.columns[0]
            series = prices[col].dropna()
            sig, wgt = self._generate_signals_single(series)
            # Reindex to full price index (in case dropna removed rows)
            full_sig = np.zeros(len(prices))
            full_wgt = np.zeros(len(prices))
            mask = prices.index.isin(series.index)
            full_sig[mask] = sig
            full_wgt[mask] = wgt
            result["signal"] = full_sig
            result["weight"] = full_wgt
        else:
            for col in prices.columns:
                series = prices[col].dropna()
                sig, wgt = self._generate_signals_single(series)
                full_sig = np.zeros(len(prices))
                full_wgt = np.zeros(len(prices))
                mask = prices.index.isin(series.index)
                full_sig[mask] = sig
                full_wgt[mask] = wgt
                result[f"{col}_signal"] = full_sig
                result[f"{col}_weight"] = full_wgt

        return result
