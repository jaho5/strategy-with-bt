"""Reproducing Kernel Hilbert Space (RKHS) regression strategy.

Applies kernel ridge regression with a Gaussian RBF kernel to predict
next-day returns from a nonlinear embedding of lagged features.

Mathematical foundation
-----------------------
RKHS:  A Hilbert space H of functions where point evaluation is a
continuous linear functional:  |f(x)| <= C ||f||_H  for all f in H.

Representer Theorem:  The solution to

    min_{f in H}  sum_i L(y_i, f(x_i)) + lambda ||f||^2_H

has the finite-dimensional form  f*(x) = sum_i alpha_i K(x, x_i).

For kernel ridge regression the closed-form solution is:

    alpha = (K + lambda I)^{-1} y

with Gram matrix  K_{ij} = k(x_i, x_j)  and Gaussian RBF kernel

    k(x, y) = exp(-||x - y||^2 / (2 sigma^2)).

The kernel bandwidth sigma is set via the median heuristic (median of
pairwise distances), and the regularisation lambda is chosen by
minimising the leave-one-out cross-validation error, which has a
well-known closed form for KRR:

    LOO-MSE = (1/n) sum_i (alpha_i / G_{ii})^2

where G = (K + lambda I)^{-1}.

The model is retrained every 21 trading days on a trailing 252-day
window, producing strictly out-of-sample predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _build_features(prices: pd.Series, volumes: Optional[pd.Series] = None) -> pd.DataFrame:
    """Construct the feature matrix x_t in R^10 from price (and optional volume) data.

    Features (10-dimensional):
        0-4: Lagged returns r_{t-1}, ..., r_{t-5}
        5:   Rolling 20-day volatility
        6:   Rolling 5-day momentum  (cumulative return over 5 days)
        7:   Rolling 20-day momentum
        8:   Rolling 60-day momentum
        9:   Volume ratio (current / 20-day average), or 1.0 if unavailable

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by datetime.
    volumes : pd.Series, optional
        Volume series aligned to *prices*.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per date and 10 columns.
    """
    log_ret = np.log(prices / prices.shift(1))

    features: Dict[str, pd.Series] = {}

    # Lagged returns
    for lag in range(1, 6):
        features[f"lag_{lag}"] = log_ret.shift(lag)

    # Rolling volatility (20-day)
    features["vol_20"] = log_ret.rolling(20, min_periods=15).std()

    # Rolling momentum (cumulative log-return)
    for window in (5, 20, 60):
        features[f"mom_{window}"] = log_ret.rolling(window, min_periods=max(window // 2, 3)).sum()

    # Volume ratio
    if volumes is not None and not volumes.isna().all():
        vol_ma = volumes.rolling(20, min_periods=10).mean()
        vol_ratio = volumes / vol_ma
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        features["vol_ratio"] = vol_ratio
    else:
        features["vol_ratio"] = pd.Series(1.0, index=prices.index)

    df = pd.DataFrame(features, index=prices.index)
    return df


def _build_target(prices: pd.Series) -> pd.Series:
    """Next-day log return (the prediction target).

    y_t = log(p_{t+1} / p_t), aligned so that features at time t
    predict y_t (which is realised at t+1).
    """
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.shift(-1)


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF kernel matrix.

    K_{ij} = exp(-||x_i - y_j||^2 / (2 sigma^2))

    Parameters
    ----------
    X : (n, d) array
    Y : (m, d) array
    sigma : kernel bandwidth

    Returns
    -------
    K : (n, m) kernel matrix
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x . y
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)  # (m, 1)
    sq_dists = X_sq + Y_sq.T - 2.0 * X @ Y.T      # (n, m)
    # Numerical guard: distances can go slightly negative due to float arithmetic
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def _median_heuristic(X: np.ndarray) -> float:
    """Median heuristic for kernel bandwidth selection.

    sigma = median of pairwise Euclidean distances (excluding zero self-distances).
    When the dataset is large, subsample for efficiency.

    Parameters
    ----------
    X : (n, d) feature matrix

    Returns
    -------
    sigma : float > 0
    """
    n = X.shape[0]
    max_subsample = 500
    if n > max_subsample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_subsample, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    X_sq = np.sum(X_sub ** 2, axis=1, keepdims=True)
    sq_dists = X_sq + X_sq.T - 2.0 * X_sub @ X_sub.T
    sq_dists = np.maximum(sq_dists, 0.0)

    # Extract upper-triangle (exclude diagonal zeros)
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    if len(upper) == 0 or np.all(upper == 0):
        return 1.0

    median_dist = np.sqrt(np.median(upper))
    return max(median_dist, 1e-8)


# ---------------------------------------------------------------------------
# Kernel Ridge Regression
# ---------------------------------------------------------------------------

class KernelRidgeRegressor:
    """Kernel Ridge Regression with Gaussian RBF kernel.

    Solves:  alpha = (K + lambda I)^{-1} y
    Predicts: y_new = k(X_new, X_train) @ alpha

    The regularisation parameter lambda is selected from a grid by
    minimising the closed-form leave-one-out cross-validation MSE.
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        lam: Optional[float] = None,
        lambda_grid: Optional[List[float]] = None,
    ) -> None:
        self.sigma = sigma
        self.lam = lam
        self.lambda_grid = lambda_grid or [
            1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0, 5.0,
        ]
        # Fitted state
        self.X_train_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        self.sigma_: Optional[float] = None
        self.lam_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidgeRegressor":
        """Fit kernel ridge regression.

        Parameters
        ----------
        X : (n, d) training features (assumed already standardised)
        y : (n,) training targets

        Returns
        -------
        self
        """
        n = X.shape[0]

        # --- Kernel bandwidth ---
        sigma = self.sigma if self.sigma is not None else _median_heuristic(X)
        self.sigma_ = sigma

        K = _rbf_kernel(X, X, sigma)  # (n, n) Gram matrix

        # --- Lambda selection via LOO-CV (closed form for KRR) ---
        if self.lam is not None:
            best_lam = self.lam
        else:
            best_lam = self.lambda_grid[0]
            best_loo_mse = np.inf

            for lam_candidate in self.lambda_grid:
                try:
                    G = np.linalg.inv(K + lam_candidate * np.eye(n))
                except np.linalg.LinAlgError:
                    continue

                alpha_candidate = G @ y
                # LOO-MSE = (1/n) sum_i (alpha_i / G_{ii})^2
                diag_G = np.diag(G)
                # Guard against zero diagonal entries
                diag_G_safe = np.where(np.abs(diag_G) < 1e-15, 1e-15, diag_G)
                loo_residuals = alpha_candidate / diag_G_safe
                loo_mse = np.mean(loo_residuals ** 2)

                if loo_mse < best_loo_mse:
                    best_loo_mse = loo_mse
                    best_lam = lam_candidate

        self.lam_ = best_lam

        # --- Final fit with chosen lambda ---
        try:
            G = np.linalg.inv(K + best_lam * np.eye(n))
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            G = np.linalg.pinv(K + best_lam * np.eye(n))

        self.alpha_ = G @ y
        self.X_train_ = X.copy()

        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict targets for new feature vectors.

        y_hat = K(X_new, X_train) @ alpha

        Parameters
        ----------
        X_new : (m, d) test features

        Returns
        -------
        y_hat : (m,) predictions
        """
        if self.X_train_ is None or self.alpha_ is None:
            raise RuntimeError("KernelRidgeRegressor has not been fitted.")
        K_new = _rbf_kernel(X_new, self.X_train_, self.sigma_)
        return K_new @ self.alpha_


# ---------------------------------------------------------------------------
# Feature standardisation (rolling, to prevent look-ahead)
# ---------------------------------------------------------------------------

def _standardise_features(
    features: np.ndarray,
    train_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score features using statistics computed only on the training portion.

    Parameters
    ----------
    features : (T, d) full feature matrix
    train_end : index marking the end of the training window (exclusive)

    Returns
    -------
    features_std : (T, d) standardised features
    mean : (d,) training mean
    std : (d,) training std
    """
    train_feats = features[:train_end]
    mean = np.nanmean(train_feats, axis=0)
    std = np.nanstd(train_feats, axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    features_std = (features - mean) / std
    return features_std, mean, std


# ===========================================================================
# Strategy class
# ===========================================================================

class RKHSRegressionStrategy(Strategy):
    """RKHS (Kernel Ridge Regression) return-prediction strategy.

    Uses a Gaussian RBF kernel to learn a nonlinear mapping from a
    10-dimensional feature vector (lagged returns, volatility, momentum,
    volume ratio) to next-day returns.

    Parameters
    ----------
    train_window : int
        Number of trailing trading days used for each training window.
        Default 252 (one year).
    retrain_interval : int
        Number of trading days between model retrains.  Default 21
        (approximately one month).
    signal_threshold : float
        Minimum absolute predicted return to generate a nonzero signal.
        The raw prediction is scaled to [-1, 1] relative to this value.
        Default 0.001 (10 bps).
    sigma : float or None
        Fixed kernel bandwidth.  If *None* (default), the median
        heuristic is used at each retraining.
    lam : float or None
        Fixed regularisation parameter.  If *None* (default), lambda is
        selected via leave-one-out CV at each retraining.
    signal_smoothing_span : int
        EMA span applied to the raw signal for noise reduction.
        Default 5.
    """

    def __init__(
        self,
        train_window: int = 120,
        retrain_interval: int = 63,
        signal_threshold: float = 0.001,
        sigma: Optional[float] = None,
        lam: Optional[float] = None,
        signal_smoothing_span: int = 5,
    ) -> None:
        super().__init__(
            name="RKHSRegression",
            description=(
                "Nonlinear return prediction via kernel ridge regression "
                "in a Reproducing Kernel Hilbert Space (Gaussian RBF kernel)."
            ),
        )
        self.train_window = train_window
        self.retrain_interval = retrain_interval
        self.signal_threshold = signal_threshold
        self.sigma = sigma
        self.lam = lam
        self.signal_smoothing_span = signal_smoothing_span

        # Trained model state per ticker (populated by fit / generate_signals)
        self._models: Dict[str, KernelRidgeRegressor] = {}

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _valid_mask(features: pd.DataFrame, target: pd.Series) -> pd.Series:
        """Boolean mask: True where both features and target are finite."""
        feat_ok = features.notna().all(axis=1)
        targ_ok = target.notna()
        return feat_ok & targ_ok

    def _predict_single_asset(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Run rolling RKHS regression for a single asset.

        Returns a Series of predicted next-day returns, NaN where the
        model has insufficient data or has not yet been trained.
        """
        features_df = _build_features(prices, volumes)
        target = _build_target(prices)
        valid = self._valid_mask(features_df, target)

        n = len(prices)
        predictions = np.full(n, np.nan)

        # Minimum data requirement: train_window valid rows
        min_start = self.train_window

        # Determine retraining dates
        features_arr = features_df.values.astype(np.float64)
        target_arr = target.values.astype(np.float64)
        valid_arr = valid.values

        current_model: Optional[KernelRidgeRegressor] = None
        current_mean: Optional[np.ndarray] = None
        current_std: Optional[np.ndarray] = None
        last_train_idx: int = -self.retrain_interval  # force first train

        for t in range(min_start, n):
            days_since_train = t - last_train_idx

            # --- Retrain if needed ---
            if days_since_train >= self.retrain_interval or current_model is None:
                # Training window: [t - train_window, t)
                train_start = max(0, t - self.train_window)
                train_slice = slice(train_start, t)

                train_valid = valid_arr[train_slice]
                if np.sum(train_valid) < 30:
                    # Not enough valid data to train
                    continue

                train_features_raw = features_arr[train_slice][train_valid]
                train_targets = target_arr[train_slice][train_valid]

                # Standardise using only training data
                mean = np.nanmean(train_features_raw, axis=0)
                std = np.nanstd(train_features_raw, axis=0)
                std = np.where(std < 1e-12, 1.0, std)

                train_features = (train_features_raw - mean) / std

                # Replace any remaining NaN with 0 (should be rare after filtering)
                train_features = np.nan_to_num(train_features, nan=0.0)
                train_targets = np.nan_to_num(train_targets, nan=0.0)

                model = KernelRidgeRegressor(sigma=self.sigma, lam=self.lam)
                try:
                    model.fit(train_features, train_targets)
                except Exception as e:
                    logger.warning("KRR fit failed at index %d: %s", t, e)
                    continue

                current_model = model
                current_mean = mean
                current_std = std
                last_train_idx = t

            # --- Predict at time t ---
            if current_model is not None and current_mean is not None:
                feat_t = features_arr[t]
                if np.any(np.isnan(feat_t)):
                    continue

                feat_t_std = (feat_t - current_mean) / current_std
                feat_t_std = np.nan_to_num(feat_t_std, nan=0.0).reshape(1, -1)

                try:
                    y_hat = current_model.predict(feat_t_std)[0]
                    predictions[t] = y_hat
                except Exception as e:
                    logger.warning("KRR predict failed at index %d: %s", t, e)

        return pd.Series(predictions, index=prices.index, name=prices.name)

    def _prediction_to_signal(self, predictions: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Convert raw predicted returns into signal and weight.

        Signal generation rule:
            signal = sign(y_hat) * min(|y_hat| / threshold, 1.0)
        then smoothed with an EMA.

        Returns
        -------
        signal : pd.Series in [-1, 1]
        weight : pd.Series in [0, 1]
        """
        y_hat = predictions.values.copy()
        threshold = self.signal_threshold

        # Raw signal: sign(y_hat) * min(|y_hat| / threshold, 1.0)
        raw_signal = np.where(
            np.isnan(y_hat),
            np.nan,
            np.sign(y_hat) * np.minimum(np.abs(y_hat) / threshold, 1.0),
        )

        signal_series = pd.Series(raw_signal, index=predictions.index)

        # Smooth with EMA
        if self.signal_smoothing_span > 1:
            signal_series = self.exponential_smooth(signal_series, span=self.signal_smoothing_span)

        # Clip to [-1, 1]
        signal_series = signal_series.clip(-1.0, 1.0)

        # Decompose into direction {-1, 0, 1} and weight [0, 1]
        direction = np.sign(signal_series.values)
        weight = np.abs(signal_series.values)

        # Zero out very small weights (noise reduction)
        direction = np.where(weight < 0.05, 0.0, direction)
        weight = np.where(weight < 0.05, 0.0, weight)

        return (
            pd.Series(direction, index=predictions.index),
            pd.Series(weight, index=predictions.index),
        )

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "RKHSRegressionStrategy":
        """Fit the RKHS regression model on historical price data.

        Trains one KRR model per ticker on the most recent ``train_window``
        rows.  The fitted models are stored for use by ``generate_signals``.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data.  Columns are tickers, index is
            DatetimeIndex.  If a 'volume' column or DataFrame is passed
            in kwargs, it is used for the volume-ratio feature.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        volumes_df: Optional[pd.DataFrame] = kwargs.get("volumes", None)

        for col in prices.columns:
            series = prices[col].dropna()
            if len(series) < self.train_window + 60:
                logger.info(
                    "Skipping %s: insufficient data (%d rows, need %d).",
                    col, len(series), self.train_window + 60,
                )
                continue

            vol_series = None
            if volumes_df is not None and col in volumes_df.columns:
                vol_series = volumes_df[col].reindex(series.index)

            features_df = _build_features(series, vol_series)
            target = _build_target(series)
            valid = self._valid_mask(features_df, target)

            # Use last train_window points for final model
            train_slice = slice(-self.train_window, None)
            train_valid = valid.iloc[train_slice].values
            if np.sum(train_valid) < 30:
                logger.info("Skipping %s: too few valid rows (%d).", col, np.sum(train_valid))
                continue

            train_features_raw = features_df.iloc[train_slice].values[train_valid]
            train_targets = target.iloc[train_slice].values[train_valid]

            mean = np.nanmean(train_features_raw, axis=0)
            std = np.nanstd(train_features_raw, axis=0)
            std = np.where(std < 1e-12, 1.0, std)

            train_features = (train_features_raw - mean) / std
            train_features = np.nan_to_num(train_features, nan=0.0)
            train_targets = np.nan_to_num(train_targets, nan=0.0)

            model = KernelRidgeRegressor(sigma=self.sigma, lam=self.lam)
            try:
                model.fit(train_features, train_targets)
                self._models[col] = model
                self.parameters[f"{col}_sigma"] = model.sigma_
                self.parameters[f"{col}_lambda"] = model.lam_
                self.parameters[f"{col}_n_support"] = len(train_targets)
                self.parameters[f"{col}_feat_mean"] = mean.tolist()
                self.parameters[f"{col}_feat_std"] = std.tolist()
            except Exception as e:
                logger.warning("Failed to fit KRR for %s: %s", col, e)

        self._fitted = True
        return self

    def generate_signals(self, prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate trading signals via rolling RKHS regression.

        For each ticker the model is retrained every ``retrain_interval``
        days on a trailing ``train_window``-day window, producing strictly
        out-of-sample return predictions that are converted to position
        signals.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (columns = tickers, index = DatetimeIndex).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for each
            ticker, or ``signal`` and ``weight`` for single-ticker data.
        """
        self.validate_prices(prices)

        volumes_df: Optional[pd.DataFrame] = kwargs.get("volumes", None)
        result = pd.DataFrame(index=prices.index)

        single_ticker = len(prices.columns) == 1

        for col in prices.columns:
            series = prices[col].dropna()
            if len(series) < self.train_window + 60:
                if single_ticker:
                    result["signal"] = 0.0
                    result["weight"] = 0.0
                else:
                    result[f"{col}_signal"] = 0.0
                    result[f"{col}_weight"] = 0.0
                continue

            vol_series = None
            if volumes_df is not None and col in volumes_df.columns:
                vol_series = volumes_df[col].reindex(series.index)

            predictions = self._predict_single_asset(series, vol_series)

            # Reindex predictions to full price index
            predictions = predictions.reindex(prices.index)

            signal, weight = self._prediction_to_signal(predictions)

            if single_ticker:
                result["signal"] = signal.values
                result["weight"] = weight.values
            else:
                result[f"{col}_signal"] = signal.reindex(prices.index, fill_value=0.0).values
                result[f"{col}_weight"] = weight.reindex(prices.index, fill_value=0.0).values

        # Fill remaining NaN with 0 (flat position)
        result = result.fillna(0.0)
        return result
