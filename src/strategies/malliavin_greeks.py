"""Malliavin calculus-inspired non-parametric sensitivity estimation strategy.

Applies the "Greeks without differentiation" philosophy from Malliavin
calculus to empirical return series, using kernel regression gradients
to estimate sensitivities of future returns to conditioning variables.

Mathematical foundation
-----------------------
Malliavin derivative D_t F:  measures the sensitivity of a random variable
F to perturbation of the Brownian motion at time t.

For a European call the Malliavin-based delta is:

    Delta = E[1_{S_T > K} * W_T / (S_0 * sigma * T)]

This avoids finite-difference bumping entirely and instead re-weights
the payoff using the Malliavin weight (the *score function* of the path
measure).  The key insight transferred to empirical trading is:

    We can estimate sensitivities of portfolio value to market movements
    without parametric models, using only the return series.

Simplified implementation
-------------------------
1. **Non-parametric sensitivity estimation** via Nadaraya-Watson kernel
   regression and its analytic gradient:

       f_hat(x)   = sum K_h(x - x_i) y_i / sum K_h(x - x_i)
       f_hat'(x)  = sum K'_h(x - x_i) (y_i - f_hat(x)) / sum K_h(x - x_i)

   where K is a Gaussian kernel and K' its derivative.

2. **Sensitivity-based trading**:
   - Positive gradient  -> positive feedback (momentum regime)
   - Negative gradient  -> negative feedback (mean-reversion regime)
   - Large |gradient|   -> strong signal
   - Small |gradient|   -> weak signal -> flat

3. **Higher-order sensitivities** (gamma):
   - Second derivative of the conditional mean estimates convexity.
   - Positive gamma + positive delta: accelerating momentum -> increase
   - Positive gamma + negative delta: decelerating reversal -> prepare to flip

4. **Conditioning variables** (feature vector):
   - Lagged return  r_{t-1}
   - Rolling volatility (20-day)
   - Volume change  (ratio of current volume to 20-day average)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian kernel and its derivatives
# ---------------------------------------------------------------------------

def _gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Standard Gaussian kernel: K(u) = (1/sqrt(2*pi)) * exp(-u^2/2)."""
    return np.exp(-0.5 * u ** 2) / np.sqrt(2.0 * np.pi)


def _gaussian_kernel_derivative(u: np.ndarray) -> np.ndarray:
    """Derivative of the Gaussian kernel: K'(u) = -u * K(u)."""
    return -u * _gaussian_kernel(u)


def _gaussian_kernel_second_derivative(u: np.ndarray) -> np.ndarray:
    """Second derivative of the Gaussian kernel: K''(u) = (u^2 - 1) * K(u)."""
    return (u ** 2 - 1.0) * _gaussian_kernel(u)


# ---------------------------------------------------------------------------
# Multivariate Nadaraya-Watson estimator with gradient
# ---------------------------------------------------------------------------

def _product_kernel_weights(
    x_query: np.ndarray,
    X_data: np.ndarray,
    bandwidths: np.ndarray,
) -> np.ndarray:
    """Compute product-kernel weights for a single query point.

    Uses a product of univariate Gaussian kernels (one per dimension):

        K_H(x - x_i) = prod_j (1/h_j) K((x_j - x_{ij}) / h_j)

    Parameters
    ----------
    x_query : (d,)   query point
    X_data  : (n, d) data points
    bandwidths : (d,) per-dimension bandwidths

    Returns
    -------
    weights : (n,) unnormalised kernel weights
    """
    # (n, d) scaled differences
    u = (x_query - X_data) / bandwidths  # broadcasting: (d,) / (d,)
    # Product kernel: product over dimensions of K(u_j) / h_j
    log_weights = np.sum(-0.5 * u ** 2, axis=1) - np.sum(np.log(bandwidths))
    # Subtract max for numerical stability before exp
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return weights


def _nadaraya_watson_with_gradient(
    x_query: np.ndarray,
    X_data: np.ndarray,
    y_data: np.ndarray,
    bandwidths: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Nadaraya-Watson regression value, gradient, and second derivative at a point.

    For multivariate x, the gradient with respect to dimension j is:

        df/dx_j = sum_i [K'_j / h_j * prod_{k!=j} K_k] * (y_i - f_hat)
                  / sum_i [prod_k K_k]

    where K_j = K((x_j - x_{ij}) / h_j) and K'_j = K'((x_j - x_{ij}) / h_j).

    Similarly for the second derivative (gamma) w.r.t. dimension j.

    Parameters
    ----------
    x_query    : (d,)   query point
    X_data     : (n, d) training inputs
    y_data     : (n,)   training targets
    bandwidths : (d,)   per-dimension bandwidths

    Returns
    -------
    f_hat      : float       conditional mean estimate
    gradient   : (d,) array  partial derivatives df/dx_j
    gamma      : (d,) array  second partial derivatives d^2f/dx_j^2
    """
    n, d = X_data.shape

    # Scaled differences: (n, d)
    u = (x_query - X_data) / bandwidths

    # Per-dimension kernel values: (n, d)
    K_vals = _gaussian_kernel(u)

    # Product kernel weights (unnormalised): (n,)
    # W_i = prod_j K(u_{ij}) / h_j
    # For numerical stability, compute in log-space
    weights = _product_kernel_weights(x_query, X_data, bandwidths)

    sum_w = np.sum(weights)
    if sum_w < 1e-300:
        # No data near this query point
        return 0.0, np.zeros(d), np.zeros(d)

    # Nadaraya-Watson estimate: f_hat = sum(w_i * y_i) / sum(w_i)
    f_hat = np.dot(weights, y_data) / sum_w

    # Residuals: (n,)
    residuals = y_data - f_hat

    # Gradient w.r.t. each dimension j
    # df/dx_j = sum_i w_i * (-u_{ij} / h_j) * (y_i - f_hat) / sum_i w_i
    #
    # This comes from differentiating the product kernel:
    # d/dx_j [prod_k K(u_k)] = prod_k K(u_k) * (-u_j / h_j)
    #                         = w_i * (-u_j / h_j)
    gradient = np.zeros(d)
    gamma = np.zeros(d)

    for j in range(d):
        # First derivative contribution: -u_{ij} / h_j
        deriv_factor = -u[:, j] / bandwidths[j]  # (n,)
        gradient[j] = np.dot(weights * deriv_factor, residuals) / sum_w

        # Second derivative contribution for gamma:
        # d^2/dx_j^2 [prod_k K(u_k)] = prod_k K(u_k) * [(u_j/h_j)^2 - 1/h_j^2]
        #                              = w_i * [(u_j^2 - 1) / h_j^2]
        second_deriv_factor = (u[:, j] ** 2 - 1.0) / (bandwidths[j] ** 2)  # (n,)

        # Gamma also has a correction term from differentiating f_hat in residuals
        # Full formula:
        # d^2f/dx_j^2 = [sum w_i * second_deriv * (y_i - f) - 2 * sum w_i * deriv * df/dx_j] / sum w_i
        # Simplified (dropping the correction for the cross-term which is small):
        gamma[j] = (
            np.dot(weights * second_deriv_factor, residuals) / sum_w
            - 2.0 * gradient[j] * np.dot(weights * deriv_factor, np.ones(n)) / sum_w
        )

    return f_hat, gradient, gamma


# ---------------------------------------------------------------------------
# Bandwidth selection: Silverman's rule of thumb (per dimension)
# ---------------------------------------------------------------------------

def _silverman_bandwidth(X: np.ndarray) -> np.ndarray:
    """Silverman's rule of thumb for multivariate kernel density estimation.

    For each dimension j:
        h_j = 1.06 * sigma_j * n^{-1/5}

    where sigma_j is the standard deviation of dimension j.

    Parameters
    ----------
    X : (n, d) data matrix

    Returns
    -------
    bandwidths : (d,) per-dimension bandwidths
    """
    n, d = X.shape
    stds = np.std(X, axis=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    # Silverman factor for Gaussian kernel
    h = 1.06 * stds * (n ** (-1.0 / 5.0))
    # Ensure minimum bandwidth
    h = np.maximum(h, 1e-8)
    return h


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _build_conditioning_features(
    prices: pd.Series,
    volumes: Optional[pd.Series] = None,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Construct the conditioning variable vector x_t.

    Features (3-dimensional):
        0: Lagged return r_{t-1}
        1: Rolling volatility (vol_window-day)
        2: Volume change ratio (current / 20-day average), or 1.0 if unavailable

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by datetime.
    volumes : pd.Series, optional
        Volume series aligned to *prices*.
    vol_window : int
        Window for rolling volatility calculation.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per date and 3 columns.
    """
    log_ret = np.log(prices / prices.shift(1))

    features: Dict[str, pd.Series] = {}

    # Lagged return
    features["lag_return"] = log_ret.shift(1)

    # Rolling volatility
    features["rolling_vol"] = log_ret.rolling(vol_window, min_periods=max(vol_window // 2, 5)).std()

    # Volume change ratio
    if volumes is not None and not volumes.isna().all():
        vol_ma = volumes.rolling(vol_window, min_periods=max(vol_window // 2, 5)).mean()
        vol_ratio = volumes / vol_ma
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        features["volume_change"] = vol_ratio
    else:
        features["volume_change"] = pd.Series(1.0, index=prices.index)

    df = pd.DataFrame(features, index=prices.index)
    return df


def _build_target(prices: pd.Series) -> pd.Series:
    """Next-period log return (the prediction target).

    y_t = log(p_{t+1} / p_t), aligned so that features at time t
    predict y_t (which is realised at t+1).
    """
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.shift(-1)


# ===========================================================================
# Strategy class
# ===========================================================================

class MalliavinGreeksStrategy(Strategy):
    """Malliavin calculus-inspired non-parametric sensitivity strategy.

    Uses Nadaraya-Watson kernel regression with a Gaussian kernel to
    estimate the conditional mean of next-period returns given a low-
    dimensional conditioning vector, then computes the analytic gradient
    (delta) and second derivative (gamma) of that conditional mean to
    determine trading signals.

    Trading logic:
    - **Delta** (first derivative w.r.t. lagged return):
        - Positive delta -> momentum regime  -> go long
        - Negative delta -> mean-reversion   -> go short
    - **Gamma** (second derivative w.r.t. lagged return):
        - Positive gamma amplifies the delta signal (accelerating)
        - Negative gamma dampens the delta signal (decelerating)
    - **Signal strength** scales with |delta|, capped at 1.

    Parameters
    ----------
    train_window : int
        Number of trailing trading days for the kernel regression
        training set.  Default 252.
    vol_window : int
        Rolling window for volatility feature.  Default 20.
    bandwidth_multiplier : float
        Multiplier applied to Silverman bandwidths (controls smoothing).
        Values > 1 give smoother estimates; < 1 give more responsive
        estimates.  Default 1.0.
    delta_threshold : float
        Minimum absolute delta to generate a nonzero signal.
        Default 0.5 (normalised units after z-scoring features).
    gamma_weight : float
        How much the gamma (convexity) signal scales the delta signal.
        The effective signal is: delta * (1 + gamma_weight * sign(gamma)).
        Default 0.3.
    signal_smoothing_span : int
        EMA span for smoothing the raw signal.  Default 5.
    retrain_interval : int
        Days between bandwidth recalculations.  Default 21.
    min_train_obs : int
        Minimum number of valid observations required to estimate
        sensitivities.  Default 60.
    """

    def __init__(
        self,
        train_window: int = 252,
        vol_window: int = 20,
        bandwidth_multiplier: float = 1.0,
        delta_threshold: float = 0.5,
        gamma_weight: float = 0.3,
        signal_smoothing_span: int = 5,
        retrain_interval: int = 21,
        min_train_obs: int = 60,
    ) -> None:
        super().__init__(
            name="MalliavinGreeks",
            description=(
                "Non-parametric sensitivity estimation inspired by Malliavin "
                "calculus.  Estimates delta (gradient) and gamma (convexity) "
                "of the conditional return function via Nadaraya-Watson kernel "
                "regression, and trades on the sign and magnitude of these "
                "sensitivities."
            ),
        )
        self.train_window = train_window
        self.vol_window = vol_window
        self.bandwidth_multiplier = bandwidth_multiplier
        self.delta_threshold = delta_threshold
        self.gamma_weight = gamma_weight
        self.signal_smoothing_span = signal_smoothing_span
        self.retrain_interval = retrain_interval
        self.min_train_obs = min_train_obs

        # Fitted state (per ticker)
        self._bandwidths: Dict[str, np.ndarray] = {}
        self._feat_mean: Dict[str, np.ndarray] = {}
        self._feat_std: Dict[str, np.ndarray] = {}

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _valid_mask(features: pd.DataFrame, target: pd.Series) -> pd.Series:
        """Boolean mask: True where both features and target are finite."""
        feat_ok = features.notna().all(axis=1)
        targ_ok = target.notna()
        return feat_ok & targ_ok

    def _estimate_sensitivities_single(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Estimate delta and gamma for a single asset via rolling NW regression.

        Returns
        -------
        f_hat_series : pd.Series
            Conditional mean estimates at each time step.
        delta_series : pd.Series
            First derivative w.r.t. the first conditioning variable
            (lagged return), i.e. the "delta" sensitivity.
        gamma_series : pd.Series
            Second derivative w.r.t. the first conditioning variable,
            i.e. the "gamma" (convexity) sensitivity.
        """
        features_df = _build_conditioning_features(prices, volumes, self.vol_window)
        target = _build_target(prices)
        valid = self._valid_mask(features_df, target)

        n = len(prices)
        d = features_df.shape[1]

        f_hat_arr = np.full(n, np.nan)
        delta_arr = np.full(n, np.nan)
        gamma_arr = np.full(n, np.nan)

        features_arr = features_df.values.astype(np.float64)
        target_arr = target.values.astype(np.float64)
        valid_arr = valid.values

        min_start = max(self.min_train_obs, self.vol_window + 5)

        current_bandwidths: Optional[np.ndarray] = None
        current_mean: Optional[np.ndarray] = None
        current_std: Optional[np.ndarray] = None
        last_bw_idx: int = -self.retrain_interval

        for t in range(min_start, n):
            # Determine training window: [t - train_window, t)
            train_start = max(0, t - self.train_window)
            train_slice = slice(train_start, t)

            train_valid = valid_arr[train_slice]
            n_valid = int(np.sum(train_valid))

            if n_valid < self.min_train_obs:
                continue

            train_features_raw = features_arr[train_slice][train_valid]
            train_targets = target_arr[train_slice][train_valid]

            # Recalculate bandwidths and standardisation periodically
            days_since_bw = t - last_bw_idx
            if days_since_bw >= self.retrain_interval or current_bandwidths is None:
                mean = np.nanmean(train_features_raw, axis=0)
                std = np.nanstd(train_features_raw, axis=0)
                std = np.where(std < 1e-12, 1.0, std)

                # Standardise training data for bandwidth estimation
                train_std = (train_features_raw - mean) / std
                train_std = np.nan_to_num(train_std, nan=0.0)

                bandwidths = _silverman_bandwidth(train_std) * self.bandwidth_multiplier
                current_bandwidths = bandwidths
                current_mean = mean
                current_std = std
                last_bw_idx = t

            # Standardise training features with current statistics
            train_features_std = (train_features_raw - current_mean) / current_std
            train_features_std = np.nan_to_num(train_features_std, nan=0.0)
            train_targets_clean = np.nan_to_num(train_targets, nan=0.0)

            # Current query point
            feat_t = features_arr[t]
            if np.any(np.isnan(feat_t)):
                continue

            x_query = (feat_t - current_mean) / current_std
            x_query = np.nan_to_num(x_query, nan=0.0)

            try:
                f_hat, gradient, gamma = _nadaraya_watson_with_gradient(
                    x_query, train_features_std, train_targets_clean,
                    current_bandwidths,
                )
                f_hat_arr[t] = f_hat
                # Delta = gradient w.r.t. first feature (lagged return)
                delta_arr[t] = gradient[0]
                # Gamma = second derivative w.r.t. first feature
                gamma_arr[t] = gamma[0]
            except Exception as e:
                logger.warning("NW estimation failed at index %d: %s", t, e)

        return (
            pd.Series(f_hat_arr, index=prices.index, name="f_hat"),
            pd.Series(delta_arr, index=prices.index, name="delta"),
            pd.Series(gamma_arr, index=prices.index, name="gamma"),
        )

    def _sensitivities_to_signal(
        self,
        delta: pd.Series,
        gamma: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """Convert delta and gamma into a trading signal and weight.

        Signal logic:
        1. Raw signal = sign(delta) * min(|delta| / delta_threshold, 1.0)
        2. Gamma adjustment: multiply by (1 + gamma_weight * sign(gamma * delta))
           - When gamma and delta have the same sign, the trend is
             *accelerating* -> amplify
           - When they have opposite signs, the trend is
             *decelerating* -> dampen
        3. Smooth with EMA and clip to [-1, 1].
        4. Decompose into direction {-1, 0, 1} and weight [0, 1].

        Returns
        -------
        signal : pd.Series  (values in {-1, 0, 1})
        weight : pd.Series  (values in [0, 1])
        """
        delta_vals = delta.values.copy()
        gamma_vals = gamma.values.copy()

        # Step 1: raw signal from delta
        with np.errstate(invalid="ignore"):
            raw_signal = np.where(
                np.isnan(delta_vals),
                np.nan,
                np.sign(delta_vals)
                * np.minimum(np.abs(delta_vals) / self.delta_threshold, 1.0),
            )

        # Step 2: gamma adjustment
        with np.errstate(invalid="ignore"):
            gamma_sign = np.where(
                np.isnan(gamma_vals) | np.isnan(delta_vals),
                0.0,
                np.sign(gamma_vals * delta_vals),
            )
            gamma_adjustment = 1.0 + self.gamma_weight * gamma_sign
            raw_signal = np.where(
                np.isnan(raw_signal),
                np.nan,
                raw_signal * gamma_adjustment,
            )

        signal_series = pd.Series(raw_signal, index=delta.index)

        # Step 3: smooth with EMA
        if self.signal_smoothing_span > 1:
            signal_series = self.exponential_smooth(
                signal_series, span=self.signal_smoothing_span,
            )

        # Clip to [-1, 1]
        signal_series = signal_series.clip(-1.0, 1.0)

        # Step 4: decompose into direction and weight
        direction = np.sign(signal_series.values)
        weight = np.abs(signal_series.values)

        # Zero out very small weights (noise reduction)
        direction = np.where(weight < 0.05, 0.0, direction)
        weight = np.where(weight < 0.05, 0.0, weight)

        return (
            pd.Series(direction, index=delta.index),
            pd.Series(weight, index=delta.index),
        )

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "MalliavinGreeksStrategy":
        """Fit the Malliavin sensitivity estimator on historical data.

        Pre-computes bandwidth parameters and feature standardisation
        statistics for each ticker using the most recent ``train_window``
        rows of data.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data.  Columns are tickers, index is
            DatetimeIndex.  If a 'volumes' DataFrame is passed in kwargs,
            it is used for the volume-change feature.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        volumes_df: Optional[pd.DataFrame] = kwargs.get("volumes", None)

        for col in prices.columns:
            series = prices[col].dropna()
            if len(series) < self.train_window:
                logger.info(
                    "Skipping %s: insufficient data (%d rows, need %d).",
                    col, len(series), self.train_window,
                )
                continue

            vol_series = None
            if volumes_df is not None and col in volumes_df.columns:
                vol_series = volumes_df[col].reindex(series.index)

            features_df = _build_conditioning_features(series, vol_series, self.vol_window)
            target = _build_target(series)
            valid = self._valid_mask(features_df, target)

            # Use last train_window points for calibration
            train_slice = slice(-self.train_window, None)
            train_valid = valid.iloc[train_slice].values

            n_valid = int(np.sum(train_valid))
            if n_valid < self.min_train_obs:
                logger.info(
                    "Skipping %s: too few valid rows (%d, need %d).",
                    col, n_valid, self.min_train_obs,
                )
                continue

            train_features_raw = features_df.iloc[train_slice].values[train_valid]

            # Compute and store standardisation statistics
            mean = np.nanmean(train_features_raw, axis=0)
            std = np.nanstd(train_features_raw, axis=0)
            std = np.where(std < 1e-12, 1.0, std)

            train_features_std = (train_features_raw - mean) / std
            train_features_std = np.nan_to_num(train_features_std, nan=0.0)

            # Compute and store bandwidths
            bandwidths = _silverman_bandwidth(train_features_std) * self.bandwidth_multiplier

            self._bandwidths[col] = bandwidths
            self._feat_mean[col] = mean
            self._feat_std[col] = std

            self.parameters[f"{col}_bandwidths"] = bandwidths.tolist()
            self.parameters[f"{col}_feat_mean"] = mean.tolist()
            self.parameters[f"{col}_feat_std"] = std.tolist()
            self.parameters[f"{col}_n_train"] = n_valid

        self._fitted = True
        return self

    def generate_signals(self, prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate trading signals via Malliavin-inspired sensitivity estimation.

        For each ticker, estimates the delta (gradient) and gamma (second
        derivative) of the conditional return function using rolling
        Nadaraya-Watson kernel regression, then converts these
        sensitivities into position signals.

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
            if len(series) < self.train_window:
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

            _, delta, gamma = self._estimate_sensitivities_single(series, vol_series)

            # Reindex to full price index
            delta = delta.reindex(prices.index)
            gamma = gamma.reindex(prices.index)

            signal, weight = self._sensitivities_to_signal(delta, gamma)

            if single_ticker:
                result["signal"] = signal.values
                result["weight"] = weight.values
            else:
                result[f"{col}_signal"] = signal.reindex(
                    prices.index, fill_value=0.0,
                ).values
                result[f"{col}_weight"] = weight.reindex(
                    prices.index, fill_value=0.0,
                ).values

        # Fill remaining NaN with 0 (flat position)
        result = result.fillna(0.0)
        return result
