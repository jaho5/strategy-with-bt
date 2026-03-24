"""Random Matrix Theory eigenportfolio strategy.

Uses Marchenko-Pastur spectral analysis to separate signal from noise in
sample correlation matrices, then constructs eigenportfolios from the
cleaned covariance structure.

Mathematical foundation
-----------------------
Marchenko-Pastur distribution for the eigenvalues of a sample correlation
matrix of N assets observed over T periods with ratio q = N/T:

    f(lambda) = (T/N) / (2 pi sigma^2) * sqrt((lambda_+ - lambda)(lambda - lambda_-)) / lambda

where the bulk edges are:

    lambda_+/- = sigma^2 * (1 +/- sqrt(N/T))^2

Eigenvalues that fall within [lambda_-, lambda_+] are consistent with
noise (random correlations).  Eigenvalues exceeding lambda_+ carry
genuine signal: the first corresponds to the market factor, subsequent
ones capture sector / style factors.

Strategy
--------
1. **Covariance cleaning** -- replace noise eigenvalues with their
   average to suppress estimation error while preserving the signal
   subspace.  Reconstruct a positive semi-definite cleaned correlation
   matrix.

2. **Eigenportfolio construction** -- the eigenvectors associated with
   signal eigenvalues define orthogonal systematic factor portfolios.

3. **Trading signals** --
   * Market eigenportfolio: trend-follow (20-day momentum > 0 -> long).
   * Sector/style eigenportfolios: mean-revert (|z-score| > 2 -> fade).
   * Weight each factor signal by its eigenvalue magnitude.

4. **Minimum-variance overlay** -- compute the global minimum-variance
   portfolio from the cleaned covariance to blend alpha signals with
   risk management.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RMTEigenportfolioConfig:
    """All tuneable knobs for the RMT eigenportfolio strategy."""

    # Correlation estimation
    corr_window: int = 252          # rolling window for sample correlation
    min_history: int = 252          # minimum observations before trading

    # Marchenko-Pastur
    mp_sigma: float = 1.0           # noise variance (1.0 for standardised returns)

    # Eigenportfolio signals
    momentum_window: int = 20       # lookback for eigenportfolio momentum
    zscore_window: int = 60         # lookback for z-score of eigenportfolio returns
    market_trend_threshold: float = 0.0   # momentum threshold for market factor
    sector_zscore_threshold: float = 2.0  # z-score threshold for mean-reversion

    # Minimum variance
    min_var_weight: float = 0.30    # blend weight for minimum-variance portfolio
    factor_weight: float = 0.70     # blend weight for factor signals

    # Risk
    max_leverage: float = 1.5       # gross leverage cap
    rebalance_freq: int = 21        # trading days between full recalculations


# ---------------------------------------------------------------------------
# Marchenko-Pastur helpers
# ---------------------------------------------------------------------------

def _marchenko_pastur_bounds(
    sigma_sq: float,
    q: float,
) -> Tuple[float, float]:
    """Compute Marchenko-Pastur bulk-edge eigenvalues.

    Parameters
    ----------
    sigma_sq : float
        Noise variance (typically 1.0 for a correlation matrix of
        standardised returns).
    q : float
        Ratio N / T (number of assets / number of observations).

    Returns
    -------
    lambda_minus, lambda_plus : float
        Lower and upper edges of the MP distribution support.
    """
    sqrt_q = np.sqrt(q)
    lambda_plus = sigma_sq * (1.0 + sqrt_q) ** 2
    lambda_minus = sigma_sq * (1.0 - sqrt_q) ** 2
    return float(lambda_minus), float(lambda_plus)


def _marchenko_pastur_pdf(
    x: np.ndarray,
    sigma_sq: float,
    q: float,
) -> np.ndarray:
    """Evaluate the Marchenko-Pastur density at points *x*.

    Parameters
    ----------
    x : array-like
        Points at which to evaluate the density.
    sigma_sq : float
        Noise variance.
    q : float
        Ratio N / T.

    Returns
    -------
    density : np.ndarray
        MP probability density values.  Zero outside the support.
    """
    lam_m, lam_p = _marchenko_pastur_bounds(sigma_sq, q)
    x = np.asarray(x, dtype=np.float64)
    density = np.zeros_like(x)
    mask = (x > lam_m) & (x < lam_p) & (x > 0)
    density[mask] = (
        (1.0 / (2.0 * np.pi * sigma_sq * q))
        * np.sqrt((lam_p - x[mask]) * (x[mask] - lam_m))
        / x[mask]
    )
    return density


# ---------------------------------------------------------------------------
# Covariance cleaning
# ---------------------------------------------------------------------------

def _clean_correlation_matrix(
    corr: np.ndarray,
    n_assets: int,
    n_obs: int,
    sigma_sq: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Clean a sample correlation matrix using RMT.

    Steps
    -----
    1. Eigendecompose ``corr = Q diag(lambda) Q'``.
    2. Compute Marchenko-Pastur bounds for the noise band.
    3. Replace eigenvalues within the MP bulk with their average
       (shrinkage towards the noise mean).
    4. Rescale so the trace equals N (preserve the correlation-matrix
       convention that diag = 1 on average).
    5. Reconstruct ``C_clean = Q diag(lambda_clean) Q'``.

    Parameters
    ----------
    corr : (N, N) array
        Sample correlation matrix (symmetric, PSD).
    n_assets : int
        Number of assets N.
    n_obs : int
        Number of return observations T.
    sigma_sq : float
        Noise variance for MP formula.

    Returns
    -------
    corr_clean : (N, N) array
        Cleaned correlation matrix.
    eigenvalues : (N,) array
        Original eigenvalues (ascending order).
    eigenvectors : (N, N) array
        Eigenvectors as columns (ascending eigenvalue order).
    n_signal : int
        Number of eigenvalues above the MP upper bound (signal count).
    """
    q = n_assets / n_obs
    lam_m, lam_p = _marchenko_pastur_bounds(sigma_sq, q)

    # Eigendecompose (eigh returns ascending eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Identify noise vs signal eigenvalues
    noise_mask = eigenvalues <= lam_p
    signal_mask = ~noise_mask
    n_signal = int(signal_mask.sum())

    # Clean: replace noise eigenvalues with their average
    cleaned_eigenvalues = eigenvalues.copy()
    if noise_mask.any():
        noise_mean = eigenvalues[noise_mask].mean()
        # Ensure the replacement is positive (numerical safety)
        noise_mean = max(noise_mean, 1e-10)
        cleaned_eigenvalues[noise_mask] = noise_mean

    # Rescale to preserve trace = N (correlation matrix convention)
    current_trace = cleaned_eigenvalues.sum()
    if current_trace > 0:
        cleaned_eigenvalues *= n_assets / current_trace

    # Reconstruct cleaned correlation
    corr_clean = (
        eigenvectors * cleaned_eigenvalues[np.newaxis, :]
    ) @ eigenvectors.T

    # Force exact symmetry (numerical insurance)
    corr_clean = (corr_clean + corr_clean.T) / 2.0

    # Clamp diagonal to 1.0 (correlation matrix)
    np.fill_diagonal(corr_clean, 1.0)

    logger.debug(
        "RMT cleaning: N=%d, T=%d, q=%.3f, MP bounds=[%.4f, %.4f], "
        "signal eigenvalues=%d",
        n_assets, n_obs, q, lam_m, lam_p, n_signal,
    )

    return corr_clean, eigenvalues, eigenvectors, n_signal


# ---------------------------------------------------------------------------
# Eigenportfolio construction
# ---------------------------------------------------------------------------

def _build_eigenportfolios(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    n_signal: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract signal eigenportfolios from the cleaned eigendecomposition.

    An eigenportfolio is the eigenvector (interpreted as portfolio weights)
    associated with a signal eigenvalue.  We normalise each eigenvector
    so that its weights sum to 1 in absolute value.

    Parameters
    ----------
    eigenvectors : (N, N) array
        Columns are eigenvectors, ordered by ascending eigenvalue.
    eigenvalues : (N,) array
        Corresponding eigenvalues (ascending).
    n_signal : int
        Number of signal eigenvalues (those above MP upper bound).

    Returns
    -------
    portfolios : (n_signal, N) array
        Each row is an eigenportfolio weight vector.  Row 0 corresponds
        to the *largest* eigenvalue (market factor).
    signal_eigenvalues : (n_signal,) array
        The signal eigenvalues in descending order.
    """
    if n_signal == 0:
        return np.empty((0, eigenvectors.shape[0])), np.empty(0)

    # Signal eigenvectors are the last n_signal columns (ascending order)
    # Reverse to get descending eigenvalue order
    signal_vecs = eigenvectors[:, -n_signal:][:, ::-1]  # (N, n_signal)
    signal_evals = eigenvalues[-n_signal:][::-1]          # (n_signal,)

    portfolios = signal_vecs.T.copy()  # (n_signal, N)

    # Normalise so |weights| sum to 1
    for i in range(n_signal):
        abs_sum = np.abs(portfolios[i]).sum()
        if abs_sum > 1e-12:
            portfolios[i] /= abs_sum

    return portfolios, signal_evals


# ---------------------------------------------------------------------------
# Minimum-variance portfolio
# ---------------------------------------------------------------------------

def _minimum_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Global minimum-variance portfolio weights from a covariance matrix.

    w_mv = C^{-1} 1 / (1' C^{-1} 1)

    Uses the pseudo-inverse for numerical stability.

    Parameters
    ----------
    cov : (N, N) array
        Positive semi-definite covariance matrix.

    Returns
    -------
    weights : (N,) array
        Minimum-variance weights summing to 1.
    """
    n = cov.shape[0]
    ones = np.ones(n)

    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        # Fallback to equal weight
        return ones / n

    w = cov_inv @ ones
    denom = ones @ w
    if abs(denom) < 1e-12:
        return ones / n

    w /= denom
    return w


# ===========================================================================
# Strategy class
# ===========================================================================

class RMTEigenportfolioStrategy(Strategy):
    """Random Matrix Theory eigenportfolio strategy.

    Cleans the sample correlation matrix using the Marchenko-Pastur law
    to separate signal from noise, constructs eigenportfolios from the
    signal subspace, and generates trading signals by trend-following the
    market factor and mean-reverting sector/style factors.

    Parameters
    ----------
    config : RMTEigenportfolioConfig, optional
        Strategy configuration.  Uses defaults if not supplied.
    """

    def __init__(self, config: Optional[RMTEigenportfolioConfig] = None) -> None:
        self.cfg = config or RMTEigenportfolioConfig()

        # State populated during fit / generate_signals
        self._corr_clean: Optional[np.ndarray] = None
        self._eigenvalues: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None
        self._n_signal: int = 0
        self._eigenportfolios: Optional[np.ndarray] = None
        self._signal_eigenvalues: Optional[np.ndarray] = None
        self._min_var_weights: Optional[np.ndarray] = None
        self._asset_names: Optional[pd.Index] = None
        self._mp_bounds: Optional[Tuple[float, float]] = None
        self._fitted: bool = False

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _compute_sample_correlation(
        self,
        returns: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the sample correlation matrix and per-asset volatilities.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily log-returns, shape (T, N).

        Returns
        -------
        corr : (N, N) array
            Sample correlation matrix.
        vols : (N,) array
            Per-asset annualised volatilities.
        """
        corr_df = returns.corr()
        # Handle any NaN in correlation (can happen with constant columns)
        corr = corr_df.values.copy()
        np.fill_diagonal(corr, 1.0)
        # Replace NaN off-diagonal entries with 0 (no correlation)
        corr = np.nan_to_num(corr, nan=0.0)
        # Force symmetry
        corr = (corr + corr.T) / 2.0

        vols = returns.std().values * np.sqrt(252)
        return corr, vols

    def _eigenportfolio_returns(
        self,
        returns: pd.DataFrame,
        portfolios: np.ndarray,
    ) -> pd.DataFrame:
        """Compute time-series returns of eigenportfolios.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns, shape (T, N).
        portfolios : (K, N) array
            Eigenportfolio weight vectors.

        Returns
        -------
        pd.DataFrame
            Eigenportfolio returns, shape (T, K).
        """
        port_returns = returns.values @ portfolios.T  # (T, K)
        columns = [f"eigen_{i}" for i in range(portfolios.shape[0])]
        return pd.DataFrame(port_returns, index=returns.index, columns=columns)

    def _market_factor_signal(
        self,
        eigen_returns: pd.Series,
    ) -> float:
        """Trend-following signal for the market eigenportfolio.

        Computes the momentum (cumulative return) over the momentum
        window.  Signal is +1 if momentum > threshold, -1 otherwise.

        Parameters
        ----------
        eigen_returns : pd.Series
            Time-series of the market eigenportfolio returns.

        Returns
        -------
        float
            Signal in {-1, +1}.
        """
        window = self.cfg.momentum_window
        if len(eigen_returns) < window:
            return 0.0

        momentum = eigen_returns.iloc[-window:].sum()
        if momentum > self.cfg.market_trend_threshold:
            return 1.0
        else:
            return -1.0

    def _sector_factor_signal(
        self,
        eigen_returns: pd.Series,
    ) -> float:
        """Mean-reversion signal for a sector/style eigenportfolio.

        Computes a rolling z-score of the cumulative eigenportfolio
        return.  If the z-score exceeds the threshold in magnitude,
        fade the move.

        Parameters
        ----------
        eigen_returns : pd.Series
            Time-series of a sector eigenportfolio's returns.

        Returns
        -------
        float
            Signal in {-1, 0, +1}.
        """
        window = self.cfg.zscore_window
        if len(eigen_returns) < window:
            return 0.0

        recent = eigen_returns.iloc[-window:]
        cum_ret = recent.sum()
        mu = recent.mean() * window
        std = recent.std() * np.sqrt(window)

        if std < 1e-12:
            return 0.0

        z = (cum_ret - mu) / std
        threshold = self.cfg.sector_zscore_threshold

        if z > threshold:
            return -1.0   # mean-revert: short the over-extended factor
        elif z < -threshold:
            return 1.0    # mean-revert: long the under-extended factor
        else:
            return 0.0

    def _generate_factor_weights(
        self,
        returns: pd.DataFrame,
    ) -> np.ndarray:
        """Combine eigenportfolio signals into asset-level weights.

        For each signal eigenportfolio:
        * Market factor (index 0): trend-follow.
        * Other factors (index 1..K-1): mean-revert.
        * Weight by eigenvalue magnitude / sum of signal eigenvalues.

        Parameters
        ----------
        returns : pd.DataFrame
            Recent asset returns, shape (T, N).

        Returns
        -------
        weights : (N,) array
            Combined asset-level signal weights.
        """
        n_assets = returns.shape[1]
        if self._eigenportfolios is None or self._n_signal == 0:
            return np.zeros(n_assets)

        # Compute eigenportfolio return time-series
        ep_returns = self._eigenportfolio_returns(returns, self._eigenportfolios)

        # Eigenvalue-based importance weights (normalised)
        eval_weights = self._signal_eigenvalues / self._signal_eigenvalues.sum()

        combined_weights = np.zeros(n_assets)

        for i in range(self._n_signal):
            ep_ret = ep_returns.iloc[:, i]

            if i == 0:
                # Market factor: trend-follow
                signal = self._market_factor_signal(ep_ret)
            else:
                # Sector/style factor: mean-revert
                signal = self._sector_factor_signal(ep_ret)

            # Contribution = signal * eigenvalue_weight * eigenportfolio_weights
            combined_weights += (
                signal * eval_weights[i] * self._eigenportfolios[i]
            )

        return combined_weights

    # -----------------------------------------------------------------
    # Strategy interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "RMTEigenportfolioStrategy":
        """Calibrate the RMT eigenportfolio model on historical price data.

        Computes the sample correlation matrix over the configured window,
        cleans it via the Marchenko-Pastur law, extracts eigenportfolios,
        and computes global minimum-variance weights.

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

        n_obs = min(len(prices), self.cfg.corr_window)
        recent_prices = prices.iloc[-n_obs:]

        # Log returns
        log_returns = np.log(recent_prices / recent_prices.shift(1)).dropna()
        n_assets = log_returns.shape[1]
        n_periods = log_returns.shape[0]

        if n_periods < self.cfg.min_history // 2:
            warnings.warn(
                f"Insufficient history for RMT cleaning: {n_periods} observations "
                f"for {n_assets} assets (need at least {self.cfg.min_history // 2}).",
                stacklevel=2,
            )
            self._fitted = True
            return self

        # 1. Sample correlation matrix
        corr, vols = self._compute_sample_correlation(log_returns)

        # 2. RMT cleaning
        q = n_assets / n_periods
        self._mp_bounds = _marchenko_pastur_bounds(self.cfg.mp_sigma, q)

        corr_clean, eigenvalues, eigenvectors, n_signal = _clean_correlation_matrix(
            corr, n_assets, n_periods, sigma_sq=self.cfg.mp_sigma,
        )

        self._corr_clean = corr_clean
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._n_signal = n_signal

        # 3. Eigenportfolios
        self._eigenportfolios, self._signal_eigenvalues = _build_eigenportfolios(
            eigenvectors, eigenvalues, n_signal,
        )

        # 4. Minimum-variance portfolio from cleaned covariance
        # Convert cleaned correlation back to covariance: Σ = diag(σ) C diag(σ)
        vols_safe = np.where(vols > 1e-12, vols, 1e-12)
        cov_clean = corr_clean * np.outer(vols_safe, vols_safe)
        self._min_var_weights = _minimum_variance_weights(cov_clean)

        # Store parameters for inspection
        self.parameters = {
            "mp_lambda_minus": self._mp_bounds[0],
            "mp_lambda_plus": self._mp_bounds[1],
            "n_signal_eigenvalues": n_signal,
            "signal_eigenvalues": self._signal_eigenvalues.tolist()
            if self._signal_eigenvalues is not None and len(self._signal_eigenvalues) > 0
            else [],
            "q_ratio": q,
        }

        logger.info(
            "RMT fit complete: %d signal eigenvalues out of %d total "
            "(MP upper bound=%.4f, largest eigenvalue=%.4f)",
            n_signal,
            n_assets,
            self._mp_bounds[1],
            eigenvalues[-1] if len(eigenvalues) > 0 else 0.0,
        )

        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate per-asset trading signals.

        Blends eigenportfolio factor signals (trend-following on the
        market factor, mean-reverting on sector factors) with a global
        minimum-variance portfolio derived from the RMT-cleaned
        covariance matrix.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (columns = tickers, index = DatetimeIndex).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for each
            asset.  Signal is in {-1, 0, +1}; weight is the position
            size.
        """
        self.ensure_fitted()

        n_assets = prices.shape[1]
        n_rows = len(prices)
        asset_names = prices.columns

        # Prepare output DataFrame
        output_cols = []
        for name in asset_names:
            output_cols.extend([f"{name}_signal", f"{name}_weight"])
        signals_df = pd.DataFrame(0.0, index=prices.index, columns=output_cols)

        # Log returns for signal computation
        log_returns = np.log(prices / prices.shift(1))

        # Edge case: no signal eigenvalues found
        if self._n_signal == 0 or self._eigenportfolios is None:
            logger.warning(
                "No signal eigenvalues detected; all structure is noise. "
                "Falling back to equal-weight minimum-variance."
            )
            # Use minimum-variance weights only
            if self._min_var_weights is not None:
                for j, name in enumerate(asset_names):
                    w = self._min_var_weights[j] if j < len(self._min_var_weights) else 0.0
                    signals_df[f"{name}_signal"] = np.sign(w)
                    signals_df[f"{name}_weight"] = abs(w)
            return signals_df

        # Rolling signal generation
        min_lookback = max(
            self.cfg.momentum_window,
            self.cfg.zscore_window,
            self.cfg.corr_window,
        )

        last_rebalance = -self.cfg.rebalance_freq  # force computation on first date
        cached_factor_weights = np.zeros(n_assets)
        cached_signals = np.zeros(n_assets)
        cached_weights = np.zeros(n_assets)

        for t in range(min_lookback, n_rows):
            # Rebalance periodically for computational efficiency
            if (t - last_rebalance) >= self.cfg.rebalance_freq:
                window_returns = log_returns.iloc[max(0, t - self.cfg.corr_window):t].dropna()

                if len(window_returns) < self.cfg.min_history // 2:
                    continue

                # Re-estimate correlation and clean it
                corr, vols = self._compute_sample_correlation(window_returns)
                n_obs_window = len(window_returns)
                n_assets_window = window_returns.shape[1]

                corr_clean, evals, evecs, n_sig = _clean_correlation_matrix(
                    corr, n_assets_window, n_obs_window,
                    sigma_sq=self.cfg.mp_sigma,
                )

                # Update eigenportfolios
                eigenportfolios, signal_evals = _build_eigenportfolios(
                    evecs, evals, n_sig,
                )

                self._eigenportfolios = eigenportfolios
                self._signal_eigenvalues = signal_evals
                self._n_signal = n_sig
                self._corr_clean = corr_clean

                # Update minimum-variance weights
                vols_safe = np.where(vols > 1e-12, vols, 1e-12)
                cov_clean = corr_clean * np.outer(vols_safe, vols_safe)
                self._min_var_weights = _minimum_variance_weights(cov_clean)

                # Factor signals from eigenportfolios
                lookback_returns = log_returns.iloc[
                    max(0, t - self.cfg.zscore_window):t
                ].dropna()

                if self._n_signal > 0 and len(lookback_returns) >= self.cfg.momentum_window:
                    cached_factor_weights = self._generate_factor_weights(
                        lookback_returns
                    )
                else:
                    cached_factor_weights = np.zeros(n_assets)

                # Blend factor signals with minimum-variance weights
                blended = (
                    self.cfg.factor_weight * cached_factor_weights
                    + self.cfg.min_var_weight * self._min_var_weights
                )

                # Normalise to respect leverage constraint
                gross = np.abs(blended).sum()
                if gross > self.cfg.max_leverage:
                    blended *= self.cfg.max_leverage / gross

                cached_signals = np.sign(blended)
                cached_weights = np.abs(blended)

                # Normalise weights to sum to at most 1
                weight_sum = cached_weights.sum()
                if weight_sum > 1.0:
                    cached_weights /= weight_sum

                last_rebalance = t

            # Write signals for this date
            for j, name in enumerate(asset_names):
                signals_df.iloc[t, signals_df.columns.get_loc(f"{name}_signal")] = (
                    cached_signals[j]
                )
                signals_df.iloc[t, signals_df.columns.get_loc(f"{name}_weight")] = (
                    cached_weights[j]
                )

        return signals_df

    # -----------------------------------------------------------------
    # Diagnostic methods
    # -----------------------------------------------------------------

    def get_mp_bounds(self) -> Optional[Tuple[float, float]]:
        """Return the Marchenko-Pastur bounds from the last fit.

        Returns
        -------
        (lambda_minus, lambda_plus) or None if not yet fitted.
        """
        return self._mp_bounds

    def get_eigenvalue_spectrum(self) -> Optional[np.ndarray]:
        """Return the full eigenvalue spectrum from the last fit.

        Eigenvalues are in ascending order.  Compare against the MP
        upper bound to identify signal eigenvalues.
        """
        return self._eigenvalues

    def get_signal_eigenportfolios(self) -> Optional[pd.DataFrame]:
        """Return the signal eigenportfolios as a labelled DataFrame.

        Rows are eigenportfolios (ordered by descending eigenvalue),
        columns are asset names.
        """
        if self._eigenportfolios is None or self._asset_names is None:
            return None
        if len(self._eigenportfolios) == 0:
            return pd.DataFrame(columns=self._asset_names)

        index = [
            f"factor_{i} (lambda={self._signal_eigenvalues[i]:.3f})"
            for i in range(len(self._signal_eigenvalues))
        ]
        return pd.DataFrame(
            self._eigenportfolios,
            index=index,
            columns=self._asset_names,
        )

    def get_cleaned_correlation(self) -> Optional[pd.DataFrame]:
        """Return the RMT-cleaned correlation matrix as a labelled DataFrame."""
        if self._corr_clean is None or self._asset_names is None:
            return None
        return pd.DataFrame(
            self._corr_clean,
            index=self._asset_names,
            columns=self._asset_names,
        )

    def get_minimum_variance_weights(self) -> Optional[pd.Series]:
        """Return the minimum-variance portfolio weights."""
        if self._min_var_weights is None or self._asset_names is None:
            return None
        return pd.Series(self._min_var_weights, index=self._asset_names, name="min_var_weight")

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        signal_info = f", {self._n_signal} signal factors" if self._fitted else ""
        return f"RMTEigenportfolioStrategy({fitted_tag}{signal_info})"
