"""
Sparse Mean-Reverting Portfolio Strategy
========================================

Constructs maximally mean-reverting portfolios under L1/L2 sparsity
constraints using elastic-net regularisation.  The key insight is that
LASSO-type penalties select the *minimal subset* of assets whose linear
combination exhibits the strongest mean reversion -- a compressed-sensing
view of statistical arbitrage.

Mathematical foundation
-----------------------
We seek portfolio weights w that solve:

    min_w  ||Delta(w'p) - theta * (mu - w'p)||^2
           + lambda_1 ||w||_1  +  lambda_2 ||w||_2^2

where:
    Delta       -- first-difference operator
    theta       -- mean-reversion speed (to be maximised)
    mu          -- long-run equilibrium level
    lambda_1    -- L1 (lasso) penalty promoting sparsity
    lambda_2    -- L2 (ridge) penalty for numerical stability
    p           -- (T x N) matrix of asset prices

In practice we reformulate this as an elastic-net regression:

    p_{t+1} = A * p_t + epsilon_t

so that the coefficients of the fitted model encode the portfolio
weights of the mean-reverting linear combination.  Sparsity from the
L1 penalty selects the relevant assets automatically, while time-series
cross-validation (TimeSeriesSplit) chooses the regularisation strength.

The sparse portfolio value V_t = w' p_t is then z-scored and traded
with standard mean-reversion entry/exit rules.

This is intimately related to the Portmanteau statistic and Box-Pierce
test: a strongly mean-reverting portfolio will have large negative
first-lag autocorrelation, which these tests detect.

Strategy
--------
1. Fit an elastic-net regression of p_{t+1} on p_t with
   TimeSeriesSplit cross-validation to select (alpha, l1_ratio).
2. Extract sparse portfolio weights from the fitted coefficients.
3. Compute the portfolio value V_t = w' p_t.
4. Z-score V_t over a rolling 63-day window.
5. Trade: long when z < -1.5, short when z > 1.5, flat when |z| < 0.5.
6. Refit every 63 trading days (quarterly) to adapt to regime changes.

References
----------
- d'Aspremont, A. (2011). "Identifying small mean-reverting portfolios".
  Quantitative Finance, 11(3), 351-364.
- Tibshirani, R. (1996). "Regression Shrinkage and Selection via the
  Lasso". JRSS-B, 58(1), 267-288.
- Zou, H. & Hastie, T. (2005). "Regularization and variable selection
  via the elastic net". JRSS-B, 67(2), 301-320.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SparseMeanReversionConfig:
    """Hyper-parameters for the sparse mean-reversion strategy."""

    # -- Elastic-net regularisation --
    alpha_grid: List[float] = field(
        default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]
    )
    l1_ratio_grid: List[float] = field(
        default_factory=lambda: [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    )
    n_cv_splits: int = 5  # number of TimeSeriesSplit folds

    # -- Portfolio construction --
    lookback: int = 252  # calibration window (trading days)
    refit_freq: int = 63  # refit every 63 days (quarterly)
    min_nonzero_assets: int = 2  # require at least 2 assets in portfolio
    max_nonzero_assets: int = 20  # cap on portfolio cardinality
    weight_normalisation: str = "l1"  # "l1" or "l2" normalisation of weights

    # -- Signal generation --
    zscore_window: int = 63  # rolling window for z-score
    entry_z: float = 1.5  # |z| > entry_z => open position
    exit_z: float = 0.5  # |z| < exit_z => close position
    stop_z: float = 4.0  # |z| > stop_z => stop-loss exit

    # -- Predictability matrix approach --
    use_predictability_matrix: bool = False  # use Gamma = E[Delta p * p'] method
    budget: float = 1.0  # L1 budget constraint ||w||_1 <= budget

    # -- Risk --
    max_leverage: float = 1.0


# ---------------------------------------------------------------------------
# Elastic-net cross-validation helpers
# ---------------------------------------------------------------------------

def _cross_validate_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    alpha_grid: List[float],
    l1_ratio_grid: List[float],
    n_splits: int = 5,
) -> Tuple[float, float, float]:
    """Select optimal (alpha, l1_ratio) via time-series cross-validation.

    Uses ``TimeSeriesSplit`` to respect temporal ordering -- no look-ahead
    bias, unlike standard k-fold.

    Parameters
    ----------
    X : (T, N) array
        Feature matrix (lagged prices / returns).
    y : (T,) array
        Target vector (next-period portfolio value or price).
    alpha_grid : list of float
        Candidate regularisation strengths.
    l1_ratio_grid : list of float
        Candidate L1/L2 mixing parameters (1.0 = pure lasso).
    n_splits : int
        Number of time-series CV folds.

    Returns
    -------
    best_alpha : float
        Optimal regularisation strength.
    best_l1_ratio : float
        Optimal L1/L2 mixing parameter.
    best_score : float
        Best (negative) mean-squared-error from CV.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_score = -np.inf
    best_alpha = alpha_grid[0]
    best_l1_ratio = l1_ratio_grid[0]

    for alpha in alpha_grid:
        for l1_ratio in l1_ratio_grid:
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=5000,
                    tol=1e-5,
                    fit_intercept=True,
                    warm_start=False,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                # Negative MSE (higher is better)
                mse = -np.mean((y_val - y_pred) ** 2)
                scores.append(mse)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
                best_l1_ratio = l1_ratio

    logger.debug(
        "CV selected alpha=%.5f, l1_ratio=%.2f (neg MSE=%.6f)",
        best_alpha, best_l1_ratio, best_score,
    )

    return best_alpha, best_l1_ratio, best_score


# ---------------------------------------------------------------------------
# Predictability matrix approach
# ---------------------------------------------------------------------------

def _compute_predictability_matrix(prices: np.ndarray) -> np.ndarray:
    """Compute the cross-autocovariance (predictability) matrix.

    Gamma = (1/T) * sum_t  Delta_p_t * p_t'

    where Delta_p_t = p_{t+1} - p_t is the first difference.

    A large |w' Gamma w| / (w' Sigma w) ratio indicates a portfolio w
    with strong predictability (mean-reversion or momentum).

    Parameters
    ----------
    prices : (T, N) array
        Price matrix.

    Returns
    -------
    Gamma : (N, N) array
        Predictability (cross-autocovariance) matrix.
    """
    dp = np.diff(prices, axis=0)  # (T-1, N)
    p_lag = prices[:-1]  # (T-1, N)
    T = dp.shape[0]
    gamma = (dp.T @ p_lag) / T  # (N, N)
    return gamma


def _solve_sparse_predictability(
    prices: np.ndarray,
    alpha: float,
    l1_ratio: float,
    budget: float = 1.0,
) -> np.ndarray:
    """Find sparse mean-reverting portfolio via predictability maximisation.

    We reformulate the generalised eigenvalue problem:

        max_w  |w' Gamma w| / (w' Sigma w)   s.t. ||w||_1 <= budget

    as an elastic-net regression.  The first-difference Delta(w'p) is
    regressed on the portfolio level w'p using coordinate descent,
    promoting sparsity in w.

    Parameters
    ----------
    prices : (T, N) array
        Price matrix.
    alpha : float
        Regularisation strength.
    l1_ratio : float
        L1/L2 mixing parameter.
    budget : float
        L1 norm budget for the weights.

    Returns
    -------
    weights : (N,) array
        Sparse portfolio weights (normalised so ||w||_1 = budget).
    """
    T, N = prices.shape
    dp = np.diff(prices, axis=0)  # (T-1, N)
    p_lag = prices[:-1]  # (T-1, N)

    # Standardise features for stable elastic-net fitting
    scaler = StandardScaler()
    p_lag_scaled = scaler.fit_transform(p_lag)

    # For each asset j, regress dp_j on p_lag to find cross-predictability.
    # Then aggregate to find the most predictable linear combination.
    # We use the "average coefficient" approach: fit N regressions,
    # then find the principal sparse direction.

    # Alternative simpler approach: regress the mean price change
    # (equal-weight portfolio change) on lagged prices to seed,
    # then refine.  More directly: use the first principal component
    # of Gamma as the target direction.

    gamma = _compute_predictability_matrix(prices)

    # Eigendecompose Gamma to find the direction of maximal predictability
    eigenvalues, eigenvectors = np.linalg.eigh(gamma + gamma.T)

    # The eigenvector with the largest-magnitude eigenvalue gives the
    # direction of maximal predictability (mean-reversion if negative)
    idx_max = np.argmax(np.abs(eigenvalues))
    target_direction = eigenvectors[:, idx_max]

    # Now find a sparse approximation to this direction via elastic-net:
    # Regress target_direction components on the identity (variable selection)
    # This is equivalent to soft-thresholding the eigenvector.
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=5000,
        fit_intercept=False,
    )

    # Use p_lag to predict dp in the target direction
    y = dp @ target_direction  # (T-1,) -- portfolio returns in target direction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(p_lag_scaled, y)

    weights = model.coef_.copy()

    # Rescale weights back to original scale
    scale = scaler.scale_
    scale[scale < 1e-12] = 1e-12
    weights = weights / scale

    # Normalise
    w_abs_sum = np.abs(weights).sum()
    if w_abs_sum > 1e-12:
        weights = weights * (budget / w_abs_sum)

    return weights


# ---------------------------------------------------------------------------
# Regression-based portfolio construction
# ---------------------------------------------------------------------------

def _fit_regression_weights(
    prices: np.ndarray,
    alpha: float,
    l1_ratio: float,
) -> Tuple[np.ndarray, float]:
    """Fit sparse mean-reverting weights via elastic-net regression.

    Regresses p_{t+1} on p_t column-by-column, then extracts the
    dominant sparse weight vector from the coefficient matrix.

    More precisely, for each asset j we fit:

        p_{j, t+1} = sum_i  A_{ji} * p_{i, t} + intercept_j + eps

    The matrix A encodes the cross-predictability structure.  The
    eigenvector of (A - I) with the most negative eigenvalue
    corresponds to the fastest mean-reverting portfolio.

    Parameters
    ----------
    prices : (T, N) array
        Price matrix.
    alpha : float
        Elastic-net regularisation strength.
    l1_ratio : float
        L1/L2 mixing ratio.

    Returns
    -------
    weights : (N,) array
        Sparse portfolio weights.
    mean_reversion_speed : float
        Estimated mean-reversion speed (most negative eigenvalue of A - I).
    """
    T, N = prices.shape
    p_lag = prices[:-1]  # (T-1, N)
    p_lead = prices[1:]  # (T-1, N)

    # Standardise for numerical stability
    scaler = StandardScaler()
    p_lag_scaled = scaler.fit_transform(p_lag)

    # Fit N separate elastic-net regressions to build the transition matrix A
    A = np.zeros((N, N))
    intercepts = np.zeros(N)

    for j in range(N):
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=5000,
            tol=1e-5,
            fit_intercept=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(p_lag_scaled, p_lead[:, j])

        # Rescale coefficients back to original space
        scale = scaler.scale_.copy()
        scale[scale < 1e-12] = 1e-12
        A[j, :] = model.coef_ / scale
        intercepts[j] = model.intercept_ - (model.coef_ / scale) @ scaler.mean_

    # The mean-reverting portfolio corresponds to the eigenvector of
    # (A - I) with the most negative eigenvalue.
    # If A*p_t ~ p_{t+1}, then (A - I)*p_t ~ Delta_p_t.
    # Eigenvalue lambda of (A-I) measures reversion speed:
    # lambda < 0 => mean-reverting, lambda ~ 0 => random walk.
    M = A - np.eye(N)
    eigenvalues, eigenvectors = np.linalg.eig(M)

    # Take real parts (M may not be symmetric)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Select the eigenvector with the most negative eigenvalue
    # (fastest mean reversion)
    idx_min = np.argmin(eigenvalues)
    theta = -eigenvalues[idx_min]  # positive => mean-reverting
    weights = eigenvectors[:, idx_min]

    # Apply sparsity: zero out small weights (those below 5% of max)
    max_abs = np.abs(weights).max()
    if max_abs > 1e-12:
        threshold = 0.05 * max_abs
        weights[np.abs(weights) < threshold] = 0.0

    # Normalise so ||w||_1 = 1
    w_abs_sum = np.abs(weights).sum()
    if w_abs_sum > 1e-12:
        weights /= w_abs_sum

    logger.debug(
        "Regression weights: theta=%.4f, n_nonzero=%d/%d, "
        "eigenvalue=%.6f",
        theta, np.count_nonzero(weights), N, eigenvalues[idx_min],
    )

    return weights, theta


# ===========================================================================
# Strategy class
# ===========================================================================

class SparseMeanReversionStrategy(Strategy):
    """Sparse mean-reverting portfolio strategy using elastic-net.

    Finds the sparsest linear combination of assets that exhibits
    maximal mean reversion, then trades the resulting synthetic spread
    with z-score-based entry/exit rules.

    The strategy supports two portfolio construction methods:

    1. **Regression-based** (default): fits p_{t+1} = A * p_t via
       elastic-net, extracts the fastest mean-reverting eigenvector of
       (A - I), and sparsifies the resulting weight vector.

    2. **Predictability-matrix** (``use_predictability_matrix=True``):
       computes Gamma = E[Delta p * p'] and finds a sparse approximation
       to the eigenvector of maximal predictability.

    Both methods use TimeSeriesSplit cross-validation to select the
    regularisation parameters (alpha, l1_ratio).

    Parameters
    ----------
    config : SparseMeanReversionConfig, optional
        Strategy configuration.  Uses defaults if not supplied.
    """

    def __init__(
        self,
        config: Optional[SparseMeanReversionConfig] = None,
    ) -> None:
        cfg = config or SparseMeanReversionConfig()
        super().__init__(
            name="SparseMeanReversion",
            description=(
                "Elastic-net sparse mean-reverting portfolio strategy. "
                "Finds the sparsest linear combination of assets with "
                "maximal mean reversion and trades the z-scored spread."
            ),
        )
        self.cfg = cfg

        # Fitted state
        self._weights: Optional[np.ndarray] = None  # sparse portfolio weights
        self._asset_names: Optional[pd.Index] = None
        self._theta: float = 0.0  # mean-reversion speed
        self._mu: float = 0.0  # long-run mean of portfolio value
        self._sigma: float = 1.0  # std of portfolio value
        self._best_alpha: float = 0.01
        self._best_l1_ratio: float = 0.9
        self._last_refit_idx: int = -1

    # -----------------------------------------------------------------
    # Cross-validation for hyperparameter selection
    # -----------------------------------------------------------------

    def _select_hyperparameters(
        self, prices: np.ndarray
    ) -> Tuple[float, float]:
        """Choose elastic-net (alpha, l1_ratio) via time-series CV.

        Constructs a regression problem p_{t+1}^{(0)} = f(p_t) for the
        first asset as a representative target, then cross-validates
        the regularisation parameters.

        Parameters
        ----------
        prices : (T, N) array
            Price matrix.

        Returns
        -------
        alpha, l1_ratio : float
            Selected hyperparameters.
        """
        T, N = prices.shape

        # Use the mean price as the target (represents an equal-weight portfolio)
        p_lag = prices[:-1]
        y = prices[1:].mean(axis=1)

        # Standardise features
        scaler = StandardScaler()
        X = scaler.fit_transform(p_lag)

        # Adjust number of CV splits if data is limited
        n_splits = min(self.cfg.n_cv_splits, max(2, T // 50))

        alpha, l1_ratio, _ = _cross_validate_elastic_net(
            X, y,
            alpha_grid=self.cfg.alpha_grid,
            l1_ratio_grid=self.cfg.l1_ratio_grid,
            n_splits=n_splits,
        )

        return alpha, l1_ratio

    # -----------------------------------------------------------------
    # Portfolio construction
    # -----------------------------------------------------------------

    def _construct_portfolio(
        self, prices: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Build the sparse mean-reverting portfolio.

        Parameters
        ----------
        prices : (T, N) array
            Price matrix for the calibration window.

        Returns
        -------
        weights : (N,) array
            Sparse portfolio weights.
        theta : float
            Estimated mean-reversion speed.
        """
        T, N = prices.shape

        if N < 2:
            raise ValueError(
                "Need at least 2 assets for sparse portfolio construction."
            )

        # Step 1: select regularisation parameters via CV
        alpha, l1_ratio = self._select_hyperparameters(prices)
        self._best_alpha = alpha
        self._best_l1_ratio = l1_ratio

        # Step 2: construct the sparse portfolio
        if self.cfg.use_predictability_matrix:
            weights = _solve_sparse_predictability(
                prices, alpha, l1_ratio, budget=self.cfg.budget,
            )
            # Estimate theta from the resulting portfolio
            port_val = prices @ weights
            if len(port_val) > 2:
                dp = np.diff(port_val)
                p_lag_port = port_val[:-1]
                p_lag_dm = p_lag_port - p_lag_port.mean()
                denom = np.sum(p_lag_dm ** 2)
                if denom > 1e-12:
                    theta = -np.sum(dp * p_lag_dm) / denom
                else:
                    theta = 0.0
            else:
                theta = 0.0
        else:
            weights, theta = _fit_regression_weights(
                prices, alpha, l1_ratio,
            )

        # Step 3: enforce cardinality constraints
        n_nonzero = np.count_nonzero(weights)

        if n_nonzero > self.cfg.max_nonzero_assets:
            # Keep only the top max_nonzero_assets by absolute weight
            abs_w = np.abs(weights)
            threshold = np.sort(abs_w)[-self.cfg.max_nonzero_assets]
            weights[abs_w < threshold] = 0.0

        if n_nonzero < self.cfg.min_nonzero_assets and N >= self.cfg.min_nonzero_assets:
            logger.warning(
                "Only %d non-zero weights (min=%d). Relaxing sparsity "
                "by halving alpha.",
                n_nonzero, self.cfg.min_nonzero_assets,
            )
            # Retry with reduced regularisation
            relaxed_alpha = alpha / 2.0
            if self.cfg.use_predictability_matrix:
                weights = _solve_sparse_predictability(
                    prices, relaxed_alpha, l1_ratio,
                    budget=self.cfg.budget,
                )
                port_val = prices @ weights
                if len(port_val) > 2:
                    dp = np.diff(port_val)
                    p_lag_port = port_val[:-1]
                    p_lag_dm = p_lag_port - p_lag_port.mean()
                    denom = np.sum(p_lag_dm ** 2)
                    theta = -np.sum(dp * p_lag_dm) / denom if denom > 1e-12 else 0.0
                else:
                    theta = 0.0
            else:
                weights, theta = _fit_regression_weights(
                    prices, relaxed_alpha, l1_ratio,
                )

        # Final normalisation
        if self.cfg.weight_normalisation == "l2":
            w_norm = np.linalg.norm(weights)
            if w_norm > 1e-12:
                weights /= w_norm
        else:
            w_abs_sum = np.abs(weights).sum()
            if w_abs_sum > 1e-12:
                weights /= w_abs_sum

        return weights, max(theta, 0.0)

    # -----------------------------------------------------------------
    # Z-score computation
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_zscore(
        portfolio_value: pd.Series,
        window: int,
    ) -> pd.Series:
        """Compute the rolling z-score of the portfolio value.

        z_t = (V_t - mu_V) / sigma_V

        where mu_V and sigma_V are estimated over a rolling window.

        Parameters
        ----------
        portfolio_value : pd.Series
            Time series of portfolio value V_t = w' p_t.
        window : int
            Rolling window for mean / std estimation.

        Returns
        -------
        pd.Series
            Z-scored portfolio value.
        """
        rolling_mean = portfolio_value.rolling(window=window, min_periods=max(window // 2, 2)).mean()
        rolling_std = portfolio_value.rolling(window=window, min_periods=max(window // 2, 2)).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0.0, np.nan)

        zscore = (portfolio_value - rolling_mean) / rolling_std
        return zscore

    # -----------------------------------------------------------------
    # Strategy interface: fit
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "SparseMeanReversionStrategy":
        """Fit the sparse mean-reversion strategy on historical prices.

        1. Selects elastic-net hyperparameters via time-series CV.
        2. Constructs the sparse mean-reverting portfolio.
        3. Estimates the long-run mean and volatility of the portfolio value.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data.  Columns are tickers, index is
            a DatetimeIndex.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        # Use the calibration window
        lookback = self.cfg.lookback
        fit_prices = prices.iloc[-lookback:] if len(prices) > lookback else prices
        fit_prices = fit_prices.dropna(axis=1, how="any")

        if fit_prices.shape[1] < 2:
            warnings.warn(
                "Fewer than 2 assets with complete data in the calibration "
                "window; cannot construct sparse portfolio.",
                stacklevel=2,
            )
            self._fitted = True
            return self

        if fit_prices.shape[0] < 30:
            warnings.warn(
                f"Only {fit_prices.shape[0]} observations in calibration "
                f"window (need >= 30 for reliable estimation).",
                stacklevel=2,
            )
            self._fitted = True
            return self

        self._asset_names = fit_prices.columns
        price_matrix = fit_prices.values.astype(np.float64)

        # Construct the sparse portfolio
        self._weights, self._theta = self._construct_portfolio(price_matrix)

        # Compute portfolio value statistics
        port_val = price_matrix @ self._weights
        self._mu = float(np.mean(port_val))
        self._sigma = float(np.std(port_val, ddof=1))
        if self._sigma < 1e-12:
            self._sigma = 1.0

        # Store parameters for inspection
        n_nonzero = int(np.count_nonzero(self._weights))
        nonzero_idx = np.nonzero(self._weights)[0]
        nonzero_assets = {
            str(self._asset_names[i]): float(self._weights[i])
            for i in nonzero_idx
        }

        self.parameters = {
            "alpha": self._best_alpha,
            "l1_ratio": self._best_l1_ratio,
            "theta": self._theta,
            "mu": self._mu,
            "sigma": self._sigma,
            "n_nonzero_assets": n_nonzero,
            "nonzero_assets": nonzero_assets,
            "half_life": np.log(2) / self._theta if self._theta > 1e-8 else np.inf,
        }

        logger.info(
            "SparseMeanReversion fit: alpha=%.5f, l1_ratio=%.2f, "
            "theta=%.4f, half_life=%.1f days, %d/%d non-zero assets",
            self._best_alpha,
            self._best_l1_ratio,
            self._theta,
            self.parameters["half_life"],
            n_nonzero,
            fit_prices.shape[1],
        )

        self._fitted = True
        return self

    # -----------------------------------------------------------------
    # Strategy interface: generate_signals
    # -----------------------------------------------------------------

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals from the sparse mean-reverting portfolio.

        Computes the portfolio value V_t = w' p_t, z-scores it over a
        rolling window, and applies entry/exit rules with hysteresis:

        * Long  when z < -entry_z  (portfolio is cheap relative to mean)
        * Short when z >  entry_z  (portfolio is rich relative to mean)
        * Flat  when |z| < exit_z  (near equilibrium)
        * Stop  when |z| > stop_z  (extreme deviation -- risk control)

        Individual asset signals are derived from the portfolio-level
        signal scaled by each asset's weight in the sparse portfolio.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.  Columns are tickers, index is DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for each
            asset.  Signal is in {-1, 0, +1}; weight is the position size.
        """
        self.ensure_fitted()

        asset_names = prices.columns
        n_rows = len(prices)
        n_assets = len(asset_names)

        # Prepare output DataFrame
        output_cols: List[str] = []
        for name in asset_names:
            output_cols.extend([f"{name}_signal", f"{name}_weight"])
        signals_df = pd.DataFrame(0.0, index=prices.index, columns=output_cols)

        # Edge case: no weights fitted
        if self._weights is None or self._asset_names is None:
            logger.warning("No sparse portfolio weights available; returning flat signals.")
            return signals_df

        # Map fitted weights to the current price columns
        weights = np.zeros(n_assets)
        for i, name in enumerate(asset_names):
            if name in self._asset_names:
                j = self._asset_names.get_loc(name)
                weights[i] = self._weights[j]

        if np.abs(weights).sum() < 1e-12:
            logger.warning(
                "All portfolio weights are zero after mapping to current "
                "asset universe; returning flat signals."
            )
            return signals_df

        # Compute portfolio value: V_t = w' p_t
        price_matrix = prices.values.astype(np.float64)
        portfolio_value = pd.Series(
            price_matrix @ weights,
            index=prices.index,
            name="portfolio_value",
        )

        # Z-score the portfolio value
        zscore = self._compute_zscore(portfolio_value, self.cfg.zscore_window)

        # Signal generation with hysteresis
        signal = np.zeros(n_rows, dtype=np.float64)
        current_pos = 0.0

        for t in range(n_rows):
            z_t = zscore.iloc[t]

            if np.isnan(z_t):
                signal[t] = current_pos
                continue

            abs_z = abs(z_t)

            if current_pos == 0.0:
                # No position -- check for entry
                if z_t < -self.cfg.entry_z:
                    current_pos = 1.0  # long (portfolio is cheap)
                elif z_t > self.cfg.entry_z:
                    current_pos = -1.0  # short (portfolio is rich)
            else:
                # Holding a position -- check for exit or stop
                if abs_z < self.cfg.exit_z:
                    current_pos = 0.0  # mean-reversion exit
                elif abs_z > self.cfg.stop_z:
                    current_pos = 0.0  # stop-loss exit

            signal[t] = current_pos

        # Map portfolio-level signal to per-asset signals and weights
        # The direction for each asset depends on the sign of its weight
        # in the sparse portfolio.
        abs_weights = np.abs(weights)
        weight_sum = abs_weights.sum()
        if weight_sum > 1e-12:
            normalised_abs_weights = abs_weights / weight_sum
        else:
            normalised_abs_weights = abs_weights

        # Enforce leverage constraint
        if weight_sum > self.cfg.max_leverage:
            normalised_abs_weights *= self.cfg.max_leverage / weight_sum

        weight_signs = np.sign(weights)

        for j, name in enumerate(asset_names):
            if abs_weights[j] < 1e-12:
                continue

            # Asset signal = portfolio_signal * sign(weight_j)
            # If portfolio is "long" (signal=+1) and weight_j > 0, go long asset j.
            # If portfolio is "long" (signal=+1) and weight_j < 0, go short asset j.
            asset_signal = signal * weight_signs[j]
            signals_df[f"{name}_signal"] = asset_signal
            signals_df[f"{name}_weight"] = normalised_abs_weights[j]

        return signals_df

    # -----------------------------------------------------------------
    # Rolling refit
    # -----------------------------------------------------------------

    def generate_signals_rolling(
        self,
        prices: pd.DataFrame,
        refit_freq: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate signals with periodic rolling refitting.

        Every ``refit_freq`` trading days the model is refit on the
        trailing calibration window and signals are generated for the
        next block.

        Parameters
        ----------
        prices : pd.DataFrame
            Full price history.
        refit_freq : int, optional
            Days between refits.  Defaults to ``self.cfg.refit_freq``.

        Returns
        -------
        pd.DataFrame
            Concatenated signals covering the full price range.
        """
        refit_freq = refit_freq or self.cfg.refit_freq
        lookback = self.cfg.lookback
        all_signals: List[pd.DataFrame] = []

        t = lookback
        while t < len(prices):
            end = min(t + refit_freq, len(prices))
            train = prices.iloc[max(0, t - lookback):t]
            test = prices.iloc[t:end]

            self.fit(train)
            sigs = self.generate_signals(test)
            all_signals.append(sigs)

            t = end

        if not all_signals:
            # Not enough data for even one calibration window
            output_cols: List[str] = []
            for name in prices.columns:
                output_cols.extend([f"{name}_signal", f"{name}_weight"])
            return pd.DataFrame(0.0, index=prices.index, columns=output_cols)

        return pd.concat(all_signals)

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def get_portfolio_weights(self) -> Optional[pd.Series]:
        """Return the sparse portfolio weights as a labelled Series.

        Non-zero entries correspond to the assets selected by the
        elastic-net procedure.

        Returns
        -------
        pd.Series or None
            Weights indexed by asset name, or None if not fitted.
        """
        if self._weights is None or self._asset_names is None:
            return None
        return pd.Series(
            self._weights,
            index=self._asset_names,
            name="sparse_weight",
        )

    def get_nonzero_assets(self) -> Optional[pd.Series]:
        """Return only the non-zero portfolio weights.

        Convenience method that filters ``get_portfolio_weights`` to
        the active (selected) assets.

        Returns
        -------
        pd.Series or None
            Non-zero weights, sorted by absolute magnitude (descending).
        """
        w = self.get_portfolio_weights()
        if w is None:
            return None
        nonzero = w[w.abs() > 1e-12]
        return nonzero.reindex(nonzero.abs().sort_values(ascending=False).index)

    def get_mean_reversion_speed(self) -> float:
        """Return the estimated mean-reversion speed theta.

        The half-life of mean reversion is ln(2) / theta trading days.
        """
        return self._theta

    def get_half_life(self) -> float:
        """Return the estimated half-life of mean reversion in trading days."""
        if self._theta > 1e-8:
            return np.log(2) / self._theta
        return np.inf

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        if self._fitted and self._weights is not None:
            n_nz = int(np.count_nonzero(self._weights))
            hl = self.get_half_life()
            return (
                f"SparseMeanReversionStrategy({fitted_tag}, "
                f"{n_nz} active assets, half_life={hl:.1f}d)"
            )
        return f"SparseMeanReversionStrategy({fitted_tag})"
