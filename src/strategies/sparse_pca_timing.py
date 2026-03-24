"""Sparse PCA factor timing strategy.

Applies Sparse PCA (Zou, Hastie, Tibshirani 2006) to extract interpretable,
sparse factors from a rolling window of asset returns, then times those
factors using momentum / mean-reversion signals weighted by eigenvalue
magnitude, with a factor rotation stability filter.

Mathematical foundation
-----------------------
Standard PCA maximises explained variance but yields dense loadings that
are difficult to interpret:

    V = argmax_v  v' Sigma v   s.t.  ||v|| = 1

Sparse PCA adds L1 (lasso) and L2 (ridge) penalties to the dictionary
learning formulation:

    min_{A,B}  sum_i ||x_i - A B' x_i||^2  +  lambda_2 ||B||^2_F
                                              +  sum_j lambda_1 ||b_j||_1
    s.t.  A' A = I

This produces sparse loading vectors b_j, each loading on only a few
assets, yielding interpretable factors (e.g. a "tech factor" that loads
exclusively on AAPL, MSFT, GOOG).

Strategy
--------
1. **Sparse factor extraction** -- Rolling 252-day window of log returns
   fed into sklearn SparsePCA to extract k=5 sparse factors.

2. **Factor timing** -- For each factor, compute 20-day momentum:
   * Strongest factor (largest explained variance): trend-follow.
   * Weaker factors: mean-revert (they capture transient dislocations).
   * All signals weighted by eigenvalue (explained variance) magnitude.

3. **Factor rotation detection** -- Track cosine similarity between
   successive loading matrices.  When cos(theta) < 0.7 the factor
   structure has rotated significantly and we reduce exposure to that
   factor until the structure restabilises.

4. **Long-short construction** -- For each factor signal, go long assets
   with positive loadings and short assets with negative loadings, scaled
   by loading magnitude.  Combine across factors.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SparsePCATimingConfig:
    """Tuneable parameters for the Sparse PCA factor timing strategy."""

    # Rolling estimation window
    lookback_window: int = 252          # trading days for return matrix
    min_history: int = 252              # minimum observations before trading

    # Sparse PCA
    n_components: int = 5               # number of sparse factors to extract
    alpha: float = 1.0                  # L1 penalty (sparsity)
    ridge_alpha: float = 0.01           # L2 penalty (stability)
    max_iter: int = 500                 # SparsePCA solver iterations
    random_state: int = 42              # reproducibility

    # Factor timing
    momentum_window: int = 20           # lookback for factor momentum signal
    trend_follow_top_k: int = 1         # top-k factors by variance: trend-follow
    mean_revert_zscore_window: int = 60 # z-score window for mean-reversion factors
    mean_revert_threshold: float = 1.5  # z-score threshold for mean-reversion entry

    # Factor rotation / stability
    stability_threshold: float = 0.7    # cos(theta) floor to trust a factor
    stability_decay: float = 0.5        # weight multiplier for unstable factors

    # Portfolio construction
    max_leverage: float = 1.5           # gross leverage cap
    rebalance_freq: int = 21            # trading days between recalculations


# ---------------------------------------------------------------------------
# Sparse PCA helpers
# ---------------------------------------------------------------------------

def _fit_sparse_pca(
    returns: np.ndarray,
    cfg: SparsePCATimingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Sparse PCA on a (T, N) return matrix.

    Parameters
    ----------
    returns : (T, N) array
        Centred (demeaned) log returns.
    cfg : SparsePCATimingConfig
        Configuration with SparsePCA hyper-parameters.

    Returns
    -------
    components : (k, N) array
        Sparse loading vectors (each row is a factor's loadings).
    transformed : (T, k) array
        Factor scores (projections of returns onto sparse components).
    explained_variances : (k,) array
        Variance of each factor score (proxy for eigenvalue magnitude).
    """
    n_samples, n_features = returns.shape
    k = min(cfg.n_components, n_features, n_samples)

    spca = SparsePCA(
        n_components=k,
        alpha=cfg.alpha,
        ridge_alpha=cfg.ridge_alpha,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
    )

    transformed = spca.fit_transform(returns)          # (T, k)
    components = spca.components_                       # (k, N)

    # Explained variance per component (empirical variance of scores)
    explained_variances = np.var(transformed, axis=0)   # (k,)

    # Sort by explained variance descending
    order = np.argsort(explained_variances)[::-1]
    components = components[order]
    transformed = transformed[:, order]
    explained_variances = explained_variances[order]

    return components, transformed, explained_variances


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, safe for zero-norm vectors.

    Parameters
    ----------
    a, b : 1-D arrays of equal length.

    Returns
    -------
    float in [-1, 1].  Returns 0.0 if either vector is zero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _match_factors(
    prev_components: np.ndarray,
    curr_components: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match current factors to previous factors by maximum cosine similarity.

    Because Sparse PCA does not guarantee a consistent ordering or sign
    across successive fits, we use a greedy matching that pairs each
    previous factor to its most similar current factor (by |cos(theta)|).

    Parameters
    ----------
    prev_components : (k_prev, N) array
    curr_components : (k_curr, N) array

    Returns
    -------
    reorder : (k_curr,) int array
        Permutation indices so that curr_components[reorder] best aligns
        with prev_components.
    similarities : (k_curr,) float array
        Cosine similarities after alignment (with sign correction).
    """
    k_prev = prev_components.shape[0]
    k_curr = curr_components.shape[0]
    k = min(k_prev, k_curr)

    # Compute full |cosine similarity| matrix
    sim_matrix = np.zeros((k_prev, k_curr))
    for i in range(k_prev):
        for j in range(k_curr):
            sim_matrix[i, j] = abs(_cosine_similarity(
                prev_components[i], curr_components[j]
            ))

    # Greedy assignment: for each previous factor, pick the most similar
    # unassigned current factor.
    used_curr = set()
    reorder = np.arange(k_curr)
    similarities = np.zeros(k_curr)

    for i in range(k):
        # Mask already-assigned columns
        row = sim_matrix[i].copy()
        for used in used_curr:
            row[used] = -1.0
        best_j = int(np.argmax(row))
        used_curr.add(best_j)
        reorder[i] = best_j

        # Sign correction: align signs so that cos > 0
        raw_cos = _cosine_similarity(
            prev_components[i], curr_components[best_j]
        )
        similarities[i] = abs(raw_cos)

    # Unmatched current factors (if k_curr > k_prev) keep original order
    remaining = [j for j in range(k_curr) if j not in used_curr]
    for idx, j in enumerate(remaining):
        reorder[k + idx] = j
        similarities[k + idx] = 0.0  # no predecessor -> unstable

    return reorder, similarities


# ---------------------------------------------------------------------------
# Factor signal construction
# ---------------------------------------------------------------------------

def _factor_momentum(
    factor_returns: np.ndarray,
    window: int,
) -> float:
    """Compute cumulative return (momentum) of a factor over a trailing window.

    Parameters
    ----------
    factor_returns : 1-D array of factor return time series.
    window : int, lookback in trading days.

    Returns
    -------
    float : cumulative return over the trailing window.
    """
    if len(factor_returns) < window:
        return 0.0
    return float(np.sum(factor_returns[-window:]))


def _factor_zscore(
    factor_returns: np.ndarray,
    window: int,
) -> float:
    """Rolling z-score of cumulative factor returns for mean-reversion.

    Parameters
    ----------
    factor_returns : 1-D array of factor return time series.
    window : int, lookback for z-score computation.

    Returns
    -------
    float : z-score of the trailing cumulative return.
    """
    if len(factor_returns) < window:
        return 0.0
    recent = factor_returns[-window:]
    cum_ret = np.sum(recent)
    mu = np.mean(recent) * window
    std = np.std(recent, ddof=1) * np.sqrt(window)
    if std < 1e-12:
        return 0.0
    return float((cum_ret - mu) / std)


# ===========================================================================
# Strategy class
# ===========================================================================

class SparsePCATimingStrategy(Strategy):
    """Sparse PCA factor timing strategy.

    Extracts interpretable sparse factors from a rolling return matrix via
    Sparse PCA, then generates long-short signals by trend-following the
    dominant factor and mean-reverting weaker factors, subject to a cosine-
    similarity stability filter that detects factor rotation.

    Parameters
    ----------
    config : SparsePCATimingConfig, optional
        Strategy configuration.  Uses defaults if not supplied.
    """

    def __init__(self, config: Optional[SparsePCATimingConfig] = None) -> None:
        self.cfg = config or SparsePCATimingConfig()

        super().__init__(
            name="SparsePCATiming",
            description=(
                "Sparse PCA interpretable factor extraction with momentum / "
                "mean-reversion factor timing and rotation stability filter."
            ),
        )

        # State populated during fit / generate_signals
        self._components: Optional[np.ndarray] = None       # (k, N)
        self._explained_var: Optional[np.ndarray] = None     # (k,)
        self._prev_components: Optional[np.ndarray] = None   # for rotation detection
        self._factor_stabilities: Optional[np.ndarray] = None  # cos(theta) per factor
        self._asset_names: Optional[pd.Index] = None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _compute_factor_returns(
        self,
        returns: np.ndarray,
        components: np.ndarray,
    ) -> np.ndarray:
        """Project asset returns onto sparse factor loadings.

        Parameters
        ----------
        returns : (T, N) array of asset log returns.
        components : (k, N) array of sparse loadings.

        Returns
        -------
        (T, k) array of factor returns.
        """
        # Normalise each loading vector to unit L1 norm for interpretability
        norms = np.abs(components).sum(axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normed = components / norms
        return returns @ normed.T  # (T, k)

    def _generate_asset_weights(
        self,
        returns: np.ndarray,
        components: np.ndarray,
        explained_var: np.ndarray,
        stabilities: np.ndarray,
    ) -> np.ndarray:
        """Combine factor signals into per-asset position weights.

        Parameters
        ----------
        returns : (T, N) array of recent asset log returns.
        components : (k, N) array of sparse loadings.
        explained_var : (k,) array of factor variances (eigenvalue proxy).
        stabilities : (k,) array of cosine similarities (rotation metric).

        Returns
        -------
        weights : (N,) array of signed asset weights.
        """
        k, n_assets = components.shape
        if k == 0:
            return np.zeros(n_assets)

        factor_returns = self._compute_factor_returns(returns, components)

        # Eigenvalue-based importance weights (normalised)
        total_var = explained_var.sum()
        if total_var < 1e-12:
            var_weights = np.ones(k) / k
        else:
            var_weights = explained_var / total_var

        combined = np.zeros(n_assets)

        for i in range(k):
            fr = factor_returns[:, i]

            # Stability gate: scale down unstable factors
            stability = stabilities[i] if i < len(stabilities) else 0.0
            if stability < self.cfg.stability_threshold:
                stability_scale = self.cfg.stability_decay
            else:
                stability_scale = 1.0

            # Factor signal: trend-follow top-k, mean-revert the rest
            if i < self.cfg.trend_follow_top_k:
                # Trend-follow: positive momentum -> long the factor
                mom = _factor_momentum(fr, self.cfg.momentum_window)
                signal = np.sign(mom) if abs(mom) > 1e-10 else 0.0
            else:
                # Mean-revert: fade extreme z-scores
                z = _factor_zscore(fr, self.cfg.mean_revert_zscore_window)
                if z > self.cfg.mean_revert_threshold:
                    signal = -1.0
                elif z < -self.cfg.mean_revert_threshold:
                    signal = 1.0
                else:
                    signal = 0.0

            # Contribution to asset weights:
            #   signal * variance_weight * stability_scale * loading_vector
            # Normalise loading vector to unit L1 norm
            loading = components[i].copy()
            l1 = np.abs(loading).sum()
            if l1 > 1e-12:
                loading /= l1

            combined += signal * var_weights[i] * stability_scale * loading

        return combined

    def _update_stability(
        self,
        new_components: np.ndarray,
    ) -> np.ndarray:
        """Compute factor stability via cosine similarity to previous loadings.

        Also updates the internal previous-components state.

        Parameters
        ----------
        new_components : (k, N) array of current sparse loadings.

        Returns
        -------
        stabilities : (k,) array of cosine similarities in [0, 1].
        """
        k = new_components.shape[0]

        if self._prev_components is None:
            # First fit: no predecessor, assume fully stable
            self._prev_components = new_components.copy()
            return np.ones(k)

        # Match factors between previous and current via greedy assignment
        reorder, similarities = _match_factors(
            self._prev_components, new_components
        )

        # Reorder current components to align with previous
        aligned = new_components[reorder]

        # Sign-correct: flip loading sign if cos < 0
        for i in range(min(k, self._prev_components.shape[0])):
            raw_cos = _cosine_similarity(
                self._prev_components[i], aligned[i]
            )
            if raw_cos < 0:
                aligned[i] *= -1.0

        self._prev_components = aligned.copy()
        self._factor_stabilities = similarities

        return similarities

    # -----------------------------------------------------------------
    # Strategy interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "SparsePCATimingStrategy":
        """Fit Sparse PCA on historical price data.

        Computes log returns over the configured lookback window and
        extracts sparse factors.  Also initialises the factor stability
        tracking state.

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

        n_obs = min(len(prices), self.cfg.lookback_window)
        recent_prices = prices.iloc[-n_obs:]

        # Log returns
        log_returns = np.log(recent_prices / recent_prices.shift(1)).dropna()
        n_periods = log_returns.shape[0]
        n_assets = log_returns.shape[1]

        if n_periods < self.cfg.min_history // 2:
            warnings.warn(
                f"Insufficient history for Sparse PCA: {n_periods} observations "
                f"for {n_assets} assets (need >= {self.cfg.min_history // 2}).",
                stacklevel=2,
            )
            self._fitted = True
            return self

        # Demean returns (Sparse PCA assumes centred data)
        returns_arr = log_returns.values.copy()
        returns_arr -= returns_arr.mean(axis=0, keepdims=True)

        # Fit Sparse PCA
        try:
            components, transformed, explained_var = _fit_sparse_pca(
                returns_arr, self.cfg
            )
        except Exception as exc:
            logger.warning("SparsePCA fit failed: %s. Strategy will be inert.", exc)
            self._components = np.zeros((0, n_assets))
            self._explained_var = np.zeros(0)
            self._factor_stabilities = np.zeros(0)
            self._fitted = True
            return self

        self._components = components
        self._explained_var = explained_var

        # Stability tracking
        stabilities = self._update_stability(components)
        self._factor_stabilities = stabilities

        # Store parameters for inspection
        k = components.shape[0]
        sparsity_ratios = [
            float((np.abs(components[i]) < 1e-6).sum() / n_assets)
            for i in range(k)
        ]
        self.parameters = {
            "n_factors": k,
            "explained_variances": explained_var.tolist(),
            "sparsity_ratios": sparsity_ratios,
            "factor_stabilities": stabilities.tolist(),
            "n_assets": n_assets,
            "n_observations": n_periods,
        }

        logger.info(
            "SparsePCA fit complete: %d factors, explained variances=%s, "
            "mean sparsity=%.2f",
            k,
            np.round(explained_var, 4).tolist(),
            float(np.mean(sparsity_ratios)),
        )

        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate per-asset trading signals from sparse factor timing.

        For each rebalance date:
        1. Re-fit Sparse PCA on the trailing lookback window.
        2. Compute factor stability (cosine similarity to previous loadings).
        3. Generate momentum / mean-reversion signals per factor.
        4. Map factor signals back to asset-level long-short weights.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (columns = tickers, index = DatetimeIndex).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for each
            asset.
        """
        self.ensure_fitted()

        n_assets = prices.shape[1]
        n_rows = len(prices)
        asset_names = prices.columns

        # Prepare output DataFrame
        output_cols: List[str] = []
        for name in asset_names:
            output_cols.extend([f"{name}_signal", f"{name}_weight"])
        signals_df = pd.DataFrame(0.0, index=prices.index, columns=output_cols)

        # Log returns
        log_returns = np.log(prices / prices.shift(1))

        # Minimum lookback: need at least lookback_window for Sparse PCA
        # plus momentum / z-score window for signals
        min_start = max(
            self.cfg.lookback_window,
            self.cfg.min_history,
        )

        # Edge case: no components fitted
        if self._components is None or len(self._components) == 0:
            logger.warning(
                "No sparse factors available; returning zero signals."
            )
            return signals_df

        last_rebalance = -self.cfg.rebalance_freq  # force first computation
        cached_signals = np.zeros(n_assets)
        cached_weights = np.zeros(n_assets)

        for t in range(min_start, n_rows):
            if (t - last_rebalance) >= self.cfg.rebalance_freq:
                # Extract window of returns
                start = max(0, t - self.cfg.lookback_window)
                window_returns = log_returns.iloc[start:t].dropna()

                if len(window_returns) < self.cfg.min_history // 2:
                    continue

                returns_arr = window_returns.values.copy()
                returns_arr -= returns_arr.mean(axis=0, keepdims=True)

                # Re-fit Sparse PCA
                try:
                    components, transformed, explained_var = _fit_sparse_pca(
                        returns_arr, self.cfg
                    )
                except Exception as exc:
                    logger.debug("SparsePCA refit failed at t=%d: %s", t, exc)
                    continue

                # Update stability tracking
                stabilities = self._update_stability(components)
                self._components = components
                self._explained_var = explained_var

                # Generate asset weights from factor signals
                # Use a slightly longer window for signal computation
                signal_start = max(
                    0, t - max(
                        self.cfg.momentum_window,
                        self.cfg.mean_revert_zscore_window,
                    )
                )
                signal_returns = log_returns.iloc[signal_start:t].dropna()

                if len(signal_returns) < self.cfg.momentum_window:
                    continue

                raw_weights = self._generate_asset_weights(
                    signal_returns.values,
                    components,
                    explained_var,
                    stabilities,
                )

                # Enforce leverage constraint
                gross = np.abs(raw_weights).sum()
                if gross > self.cfg.max_leverage:
                    raw_weights *= self.cfg.max_leverage / gross

                cached_signals = np.sign(raw_weights)
                cached_weights = np.abs(raw_weights)

                # Normalise weights to sum to at most 1
                weight_sum = cached_weights.sum()
                if weight_sum > 1.0:
                    cached_weights /= weight_sum

                last_rebalance = t

            # Write signals for this date
            for j, name in enumerate(asset_names):
                signals_df.iloc[
                    t, signals_df.columns.get_loc(f"{name}_signal")
                ] = cached_signals[j]
                signals_df.iloc[
                    t, signals_df.columns.get_loc(f"{name}_weight")
                ] = cached_weights[j]

        return signals_df

    # -----------------------------------------------------------------
    # Diagnostic methods
    # -----------------------------------------------------------------

    def get_sparse_components(self) -> Optional[pd.DataFrame]:
        """Return the sparse loading matrix as a labelled DataFrame.

        Rows are factors (ordered by descending explained variance),
        columns are asset names.  Zero entries indicate the factor does
        not load on that asset.
        """
        if self._components is None or self._asset_names is None:
            return None
        if len(self._components) == 0:
            return pd.DataFrame(columns=self._asset_names)

        index = [
            f"factor_{i} (var={self._explained_var[i]:.4f})"
            for i in range(len(self._explained_var))
        ]
        return pd.DataFrame(
            self._components,
            index=index,
            columns=self._asset_names,
        )

    def get_factor_sparsity(self) -> Optional[pd.Series]:
        """Return the sparsity ratio (fraction of zero loadings) per factor.

        A sparsity ratio of 0.8 means 80% of asset loadings are zero for
        that factor, indicating high interpretability.
        """
        if self._components is None:
            return None
        k, n = self._components.shape
        ratios = [
            float((np.abs(self._components[i]) < 1e-6).sum() / n)
            for i in range(k)
        ]
        return pd.Series(
            ratios,
            index=[f"factor_{i}" for i in range(k)],
            name="sparsity_ratio",
        )

    def get_factor_stabilities(self) -> Optional[pd.Series]:
        """Return the cosine similarity stability metric per factor.

        Values close to 1.0 indicate the factor structure is stable
        (loadings have not rotated).  Values below the threshold
        (default 0.7) trigger position reduction.
        """
        if self._factor_stabilities is None:
            return None
        k = len(self._factor_stabilities)
        return pd.Series(
            self._factor_stabilities,
            index=[f"factor_{i}" for i in range(k)],
            name="stability_cos_theta",
        )

    def get_explained_variances(self) -> Optional[pd.Series]:
        """Return explained variance (eigenvalue proxy) per factor."""
        if self._explained_var is None:
            return None
        k = len(self._explained_var)
        return pd.Series(
            self._explained_var,
            index=[f"factor_{i}" for i in range(k)],
            name="explained_variance",
        )

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        n_factors = (
            self._components.shape[0]
            if self._components is not None and len(self._components) > 0
            else 0
        )
        factor_info = f", {n_factors} factors" if self._fitted else ""
        return f"SparsePCATimingStrategy({fitted_tag}{factor_info})"
