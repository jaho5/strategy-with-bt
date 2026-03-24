"""Rough volatility strategy based on Hurst exponent estimation.

Mathematical foundation
-----------------------
Rough volatility: log-volatility follows fractional Brownian motion (fBm) with
Hurst exponent H ~ 0.1, far below the Brownian benchmark of H = 0.5.

fBm covariance structure:
    E[B^H_s B^H_t] = 1/2 (|s|^{2H} + |t|^{2H} - |s-t|^{2H})

Hurst exponent regimes:
    H < 0.5 : rough / anti-persistent (mean-reverting increments)
    H = 0.5 : standard Brownian motion (random walk)
    H > 0.5 : smooth / persistent (trending increments)

Key insight from Gatheral, Jaisson & Rosenbaum (2018): volatility is *rough*
(H_vol ~ 0.1), while price returns can exhibit any H.  Estimating H on the
return series gives a directional regime signal; estimating H on log realized
volatility gives a secondary vol-timing signal.

Estimation methods
------------------
1. Detrended Fluctuation Analysis (DFA) -- primary estimator
   - Divide cumulative deviation profile into non-overlapping windows of size n
   - Detrend each window via least-squares linear fit
   - Compute root-mean-square fluctuation F(n)
   - H is the slope of log F(n) vs log n

2. Rescaled Range (R/S) analysis -- confirmation estimator
   - For each window size n, compute R(n)/S(n) where R is the range of
     cumulative deviations and S is the standard deviation
   - R/S ~ c * n^H  =>  H = slope of log(R/S) vs log(n)

3. Variogram method for log-volatility roughness
   - E[|log sigma_{t+Delta} - log sigma_t|^q] ~ Delta^{qH}
   - Used to estimate H_vol from realized volatility series

Strategy signals
----------------
1. Estimate H_returns on a rolling 126-day window using DFA (primary) with
   R/S confirmation.
2. Adaptive regime selection:
   - H > 0.55: trending => momentum signal (sign of 12-day return)
   - H < 0.45: mean-reverting => mean-reversion signal (z-score of 20-day MA)
   - 0.45 <= H <= 0.55: random walk => flat / minimal position
3. Roughness of volatility (H_vol) as secondary signal:
   - H_vol < 0.2: vol is anti-persistent => expect vol mean-reversion after
     spikes => favorable for long positions
4. Position sizing proportional to |H - 0.5| (signal strength).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RoughVolConfig:
    """Tuneable parameters for the rough volatility strategy."""

    # Hurst estimation
    hurst_window: int = 126             # rolling window for H estimation (~ 6 months)
    dfa_min_box: int = 4                # smallest DFA box size
    dfa_max_box_frac: float = 0.25      # largest box = frac * window
    dfa_n_boxes: int = 20               # number of log-spaced box sizes
    rs_min_box: int = 10                # smallest R/S box size
    rs_n_boxes: int = 15                # number of log-spaced box sizes for R/S

    # Regime thresholds
    trending_threshold: float = 0.55    # H > this => trending
    mean_revert_threshold: float = 0.45 # H < this => mean-reverting
    dfa_rs_blend: float = 0.7           # weight on DFA vs R/S (0.7 DFA + 0.3 R/S)

    # Momentum sub-signal
    momentum_lookback: int = 12         # return lookback for momentum signal

    # Mean-reversion sub-signal
    mr_ma_window: int = 20              # MA window for mean-reversion z-score
    mr_zscore_window: int = 60          # lookback for z-score normalisation

    # Volatility roughness
    rv_window: int = 20                 # realized vol window (trading days)
    vol_hurst_window: int = 252         # window for estimating H_vol
    vol_rough_threshold: float = 0.2    # H_vol < this => rough (anti-persistent)
    vol_signal_weight: float = 0.2      # weight of vol roughness signal in final blend

    # Position sizing
    max_position: float = 1.0           # maximum absolute position weight
    min_signal_strength: float = 0.02   # |H - 0.5| below this => go flat

    # Smoothing
    signal_ema_span: int = 5            # EMA span for final signal smoothing

    # Re-estimation frequency (days) -- skip H estimation on interim days
    refit_frequency: int = 5


# ---------------------------------------------------------------------------
# Pure numpy/scipy Hurst estimators
# ---------------------------------------------------------------------------

def _dfa(series: np.ndarray, min_box: int = 4, max_box: int | None = None,
         n_boxes: int = 20) -> float:
    """Detrended Fluctuation Analysis for Hurst exponent estimation.

    Parameters
    ----------
    series : 1-D array
        Time series of returns or increments.
    min_box : int
        Minimum box (window) size.
    max_box : int or None
        Maximum box size.  Defaults to len(series) // 4.
    n_boxes : int
        Number of logarithmically spaced box sizes.

    Returns
    -------
    float
        Estimated Hurst exponent H.

    Notes
    -----
    DFA procedure:
    1. Compute the cumulative deviation profile Y(k) = sum_{i=1}^{k} (x_i - <x>)
    2. For each box size n, divide Y into non-overlapping segments
    3. In each segment, fit a linear trend and compute residuals
    4. F(n) = sqrt(mean of squared residuals across all segments)
    5. H = slope of log(F) vs log(n)
    """
    n = len(series)
    if n < 2 * min_box:
        return 0.5  # insufficient data

    if max_box is None:
        max_box = n // 4
    max_box = max(max_box, min_box + 1)

    # Step 1: cumulative deviation profile
    mean_x = np.mean(series)
    profile = np.cumsum(series - mean_x)

    # Step 2: generate log-spaced box sizes
    box_sizes = np.unique(
        np.logspace(np.log10(min_box), np.log10(max_box), num=n_boxes).astype(int)
    )
    box_sizes = box_sizes[box_sizes >= min_box]
    if len(box_sizes) < 3:
        return 0.5

    fluctuations = np.empty(len(box_sizes))

    for idx, box in enumerate(box_sizes):
        n_segments = n // box
        if n_segments < 1:
            fluctuations[idx] = np.nan
            continue

        # Collect detrended residual variance from each segment
        rms_list = []
        t = np.arange(box, dtype=float)

        for seg in range(n_segments):
            start = seg * box
            end = start + box
            segment = profile[start:end]

            # Linear detrending via least-squares: y = a + b*t
            # Using numpy for speed (no scipy overhead per segment)
            t_mean = t.mean()
            seg_mean = segment.mean()
            t_centered = t - t_mean
            s_centered = segment - seg_mean
            denom = np.dot(t_centered, t_centered)
            if denom == 0:
                continue
            slope = np.dot(t_centered, s_centered) / denom
            intercept = seg_mean - slope * t_mean
            trend = intercept + slope * t
            residuals = segment - trend
            rms_list.append(np.mean(residuals ** 2))

        if len(rms_list) == 0:
            fluctuations[idx] = np.nan
        else:
            fluctuations[idx] = np.sqrt(np.mean(rms_list))

    # Remove NaNs
    valid = ~np.isnan(fluctuations)
    if valid.sum() < 3:
        return 0.5

    log_n = np.log(box_sizes[valid].astype(float))
    log_f = np.log(fluctuations[valid])

    # Step 5: linear regression
    slope, _, r_value, _, _ = linregress(log_n, log_f)

    # Sanity check: R^2 should be reasonable; if not, return 0.5
    if r_value ** 2 < 0.8:
        return 0.5

    # Clamp to [0, 1] range
    return float(np.clip(slope, 0.01, 0.99))


def _rescaled_range(series: np.ndarray, min_box: int = 10,
                    n_boxes: int = 15) -> float:
    """Rescaled Range (R/S) analysis for Hurst exponent estimation.

    Parameters
    ----------
    series : 1-D array
        Time series of returns or increments.
    min_box : int
        Minimum window size.
    n_boxes : int
        Number of logarithmically spaced window sizes.

    Returns
    -------
    float
        Estimated Hurst exponent H.

    Notes
    -----
    For each window size n:
        1. Divide series into non-overlapping blocks of length n
        2. For each block, compute:
           - mean-adjusted cumulative sum Z(k) = sum_{i=1}^{k} (x_i - x_bar)
           - R(n) = max(Z) - min(Z)
           - S(n) = std(block)
           - R/S ratio for the block
        3. Average R/S across blocks
    H = slope of log(R/S) vs log(n)
    """
    n = len(series)
    if n < 2 * min_box:
        return 0.5

    max_box = n // 2
    max_box = max(max_box, min_box + 1)

    box_sizes = np.unique(
        np.logspace(np.log10(min_box), np.log10(max_box), num=n_boxes).astype(int)
    )
    box_sizes = box_sizes[box_sizes >= min_box]
    if len(box_sizes) < 3:
        return 0.5

    rs_values = np.empty(len(box_sizes))

    for idx, box in enumerate(box_sizes):
        n_segments = n // box
        if n_segments < 1:
            rs_values[idx] = np.nan
            continue

        rs_list = []
        for seg in range(n_segments):
            start = seg * box
            end = start + box
            block = series[start:end]

            block_mean = np.mean(block)
            block_std = np.std(block, ddof=1)
            if block_std < 1e-15:
                continue

            # Cumulative deviation from mean
            cumdev = np.cumsum(block - block_mean)
            r = np.max(cumdev) - np.min(cumdev)
            rs_list.append(r / block_std)

        if len(rs_list) == 0:
            rs_values[idx] = np.nan
        else:
            rs_values[idx] = np.mean(rs_list)

    valid = ~np.isnan(rs_values)
    if valid.sum() < 3:
        return 0.5

    log_n = np.log(box_sizes[valid].astype(float))
    log_rs = np.log(rs_values[valid])

    slope, _, r_value, _, _ = linregress(log_n, log_rs)

    if r_value ** 2 < 0.7:
        return 0.5

    return float(np.clip(slope, 0.01, 0.99))


def _variogram_hurst(log_vol: np.ndarray, max_lag: int = 30, q: float = 2.0) -> float:
    """Variogram-based Hurst exponent for log-volatility roughness.

    Estimates H from the power-law scaling:
        E[|log sigma_{t+Delta} - log sigma_t|^q] ~ Delta^{qH}

    Parameters
    ----------
    log_vol : 1-D array
        Log realized volatility series.
    max_lag : int
        Maximum lag Delta to consider.
    q : float
        Moment order (default 2 = second moment / variogram).

    Returns
    -------
    float
        Estimated H_vol.
    """
    n = len(log_vol)
    if n < max_lag + 10:
        return 0.5

    lags = np.arange(1, max_lag + 1)
    moments = np.empty(len(lags))

    for i, lag in enumerate(lags):
        diffs = log_vol[lag:] - log_vol[:-lag]
        # Remove NaNs
        diffs = diffs[~np.isnan(diffs)]
        if len(diffs) < 10:
            moments[i] = np.nan
            continue
        moments[i] = np.mean(np.abs(diffs) ** q)

    valid = ~np.isnan(moments) & (moments > 0)
    if valid.sum() < 3:
        return 0.5

    log_lag = np.log(lags[valid].astype(float))
    log_mom = np.log(moments[valid])

    slope, _, r_value, _, _ = linregress(log_lag, log_mom)

    # slope = q * H  =>  H = slope / q
    h = slope / q

    if r_value ** 2 < 0.7:
        return 0.5

    return float(np.clip(h, 0.01, 0.99))


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

class RoughVolatilityStrategy(Strategy):
    """Trading strategy based on rough volatility theory and Hurst exponent.

    Uses Detrended Fluctuation Analysis (DFA) and Rescaled Range (R/S) analysis
    to estimate the Hurst exponent of return series, then adaptively selects
    between momentum and mean-reversion regimes.  A secondary signal from the
    roughness of log realized volatility provides vol-timing information.

    Parameters
    ----------
    config : RoughVolConfig, optional
        Strategy configuration.  Defaults are used if not supplied.
    """

    def __init__(self, config: Optional[RoughVolConfig] = None) -> None:
        cfg = config or RoughVolConfig()
        super().__init__(
            name="RoughVolatility",
            description=(
                "Adaptive momentum/mean-reversion strategy driven by Hurst "
                "exponent estimation via DFA and R/S analysis, with volatility "
                "roughness timing from the variogram method."
            ),
        )
        self.cfg = cfg

        # Internal state populated by fit / generate_signals
        self._hurst_history: Dict[str, pd.Series] = {}
        self._hvol_history: Dict[str, pd.Series] = {}
        self.parameters: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Hurst estimation wrappers
    # ------------------------------------------------------------------

    def _estimate_hurst_returns(self, returns: np.ndarray) -> Tuple[float, float]:
        """Estimate Hurst exponent from return series using DFA + R/S.

        Returns
        -------
        (h_dfa, h_rs) : tuple of float
            Hurst estimates from DFA and R/S respectively.
        """
        max_box_dfa = max(
            int(len(returns) * self.cfg.dfa_max_box_frac),
            self.cfg.dfa_min_box + 2,
        )
        h_dfa = _dfa(
            returns,
            min_box=self.cfg.dfa_min_box,
            max_box=max_box_dfa,
            n_boxes=self.cfg.dfa_n_boxes,
        )
        h_rs = _rescaled_range(
            returns,
            min_box=self.cfg.rs_min_box,
            n_boxes=self.cfg.rs_n_boxes,
        )
        return h_dfa, h_rs

    def _blended_hurst(self, returns: np.ndarray) -> float:
        """Blended Hurst estimate (DFA-weighted + R/S confirmation)."""
        h_dfa, h_rs = self._estimate_hurst_returns(returns)
        w = self.cfg.dfa_rs_blend
        return w * h_dfa + (1.0 - w) * h_rs

    def _estimate_hurst_vol(self, log_rv: np.ndarray) -> float:
        """Estimate Hurst exponent of log realized volatility (roughness).

        Uses the variogram method which is most appropriate for
        fractional processes with small H.
        """
        return _variogram_hurst(log_rv, max_lag=min(30, len(log_rv) // 4))

    # ------------------------------------------------------------------
    # Sub-signal generators
    # ------------------------------------------------------------------

    @staticmethod
    def _momentum_signal(returns: pd.Series, lookback: int = 12) -> pd.Series:
        """Momentum signal: sign of cumulative return over lookback period.

        Returns values in {-1, 0, +1}.
        """
        cum_ret = returns.rolling(window=lookback, min_periods=lookback).sum()
        signal = np.sign(cum_ret)
        return signal.fillna(0.0)

    @staticmethod
    def _mean_reversion_signal(
        prices: pd.Series,
        ma_window: int = 20,
        zscore_window: int = 60,
    ) -> pd.Series:
        """Mean-reversion signal: z-score of price relative to its MA.

        z_t = (P_t - MA_t) / rolling_std(P - MA)

        Returns signal in [-1, +1] (negative z => buy, positive z => sell,
        i.e., contrarian).
        """
        ma = prices.rolling(window=ma_window, min_periods=ma_window).mean()
        deviation = prices - ma
        dev_std = deviation.rolling(
            window=zscore_window, min_periods=zscore_window // 2
        ).std()
        dev_std = dev_std.replace(0, np.nan)

        z = deviation / dev_std
        # Mean-reversion: negative z => expect reversion up => buy (+1)
        signal = -z.clip(-3.0, 3.0) / 3.0  # normalise to [-1, 1]
        return signal.fillna(0.0)

    def _vol_roughness_signal(
        self,
        log_rv: pd.Series,
        h_vol: float,
    ) -> pd.Series:
        """Secondary signal from volatility roughness.

        When H_vol is very small (< vol_rough_threshold), vol is
        anti-persistent and tends to mean-revert after spikes.  If current
        vol is elevated relative to its median, we expect vol compression
        (favorable for long positions).

        Returns signal in [-1, +1].
        """
        if h_vol >= self.cfg.vol_rough_threshold:
            # Vol is not rough enough for this signal to be informative
            return pd.Series(0.0, index=log_rv.index)

        # Roughness strength: how far below the threshold
        roughness_strength = (self.cfg.vol_rough_threshold - h_vol) / self.cfg.vol_rough_threshold

        # Detect elevated vol (above rolling median => expect compression)
        rv_median = log_rv.rolling(
            window=self.cfg.vol_hurst_window,
            min_periods=self.cfg.vol_hurst_window // 2,
        ).median()
        rv_std = log_rv.rolling(
            window=self.cfg.vol_hurst_window,
            min_periods=self.cfg.vol_hurst_window // 2,
        ).std()
        rv_std = rv_std.replace(0, np.nan)

        z_vol = (log_rv - rv_median) / rv_std

        # High vol (positive z) + rough vol => expect compression => long signal
        # Low vol (negative z) + rough vol => expect expansion => caution
        signal = z_vol.clip(-2.0, 2.0) / 2.0 * roughness_strength
        return signal.fillna(0.0)

    # ------------------------------------------------------------------
    # Rolling Hurst computation
    # ------------------------------------------------------------------

    def _rolling_hurst(
        self,
        returns: pd.Series,
        window: int,
        refit_freq: int = 5,
    ) -> pd.Series:
        """Compute rolling blended Hurst exponent.

        For computational efficiency, H is only re-estimated every
        ``refit_freq`` days and forward-filled in between.
        """
        n = len(returns)
        h_series = pd.Series(np.nan, index=returns.index)

        ret_values = returns.values
        last_h = np.nan

        for i in range(window, n):
            if (i - window) % refit_freq == 0:
                window_data = ret_values[i - window: i]
                # Skip if too many NaNs
                valid_data = window_data[~np.isnan(window_data)]
                if len(valid_data) < window // 2:
                    last_h = 0.5
                else:
                    last_h = self._blended_hurst(valid_data)
            h_series.iloc[i] = last_h

        return h_series

    def _rolling_hurst_vol(
        self,
        log_rv: pd.Series,
        window: int,
        refit_freq: int = 21,
    ) -> pd.Series:
        """Compute rolling Hurst exponent of log realized volatility.

        Uses the variogram method.  Re-estimated less frequently because
        the vol roughness parameter is more stable.
        """
        n = len(log_rv)
        h_series = pd.Series(np.nan, index=log_rv.index)

        rv_values = log_rv.values
        last_h = np.nan

        for i in range(window, n):
            if (i - window) % refit_freq == 0:
                window_data = rv_values[i - window: i]
                valid_data = window_data[~np.isnan(window_data)]
                if len(valid_data) < window // 2:
                    last_h = 0.5
                else:
                    last_h = self._estimate_hurst_vol(valid_data)
            h_series.iloc[i] = last_h

        return h_series

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "RoughVolatilityStrategy":
        """Calibrate strategy on historical price data.

        Computes baseline Hurst exponents and stores calibration parameters.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data.  Columns are tickers; index is DatetimeIndex.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        log_returns = np.log(prices / prices.shift(1)).dropna()

        calibrated_params: Dict[str, Dict[str, float]] = {}

        for col in prices.columns:
            rets = log_returns[col].dropna()
            if len(rets) < self.cfg.hurst_window:
                warnings.warn(
                    f"Insufficient data for {col}: {len(rets)} rows < "
                    f"hurst_window {self.cfg.hurst_window}. Skipping.",
                    stacklevel=2,
                )
                continue

            # Calibration: estimate H on the full training set
            ret_values = rets.values
            h_dfa, h_rs = self._estimate_hurst_returns(ret_values)
            h_blend = self.cfg.dfa_rs_blend * h_dfa + (1.0 - self.cfg.dfa_rs_blend) * h_rs

            # Realized volatility and its roughness
            rv = rets.rolling(window=self.cfg.rv_window, min_periods=self.cfg.rv_window).std()
            log_rv = np.log(rv.dropna().clip(lower=1e-10))
            h_vol = self._estimate_hurst_vol(log_rv.values) if len(log_rv) > 60 else 0.5

            calibrated_params[col] = {
                "h_dfa": h_dfa,
                "h_rs": h_rs,
                "h_blend": h_blend,
                "h_vol": h_vol,
            }

            logger.info(
                "Calibrated %s: H_dfa=%.3f, H_rs=%.3f, H_blend=%.3f, H_vol=%.3f",
                col, h_dfa, h_rs, h_blend, h_vol,
            )

        self.parameters = {"calibration": calibrated_params}
        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals based on Hurst exponent regimes.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (may extend beyond the fit window).

        Returns
        -------
        pd.DataFrame
            Columns: ``signal`` and ``weight`` for single-asset, or
            ``{ticker}_signal`` and ``{ticker}_weight`` for multi-asset.
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        log_returns = np.log(prices / prices.shift(1))
        n_cols = len(prices.columns)
        is_single = n_cols == 1

        results = pd.DataFrame(index=prices.index)

        for col in prices.columns:
            rets = log_returns[col]
            price_col = prices[col]

            # ---- Rolling Hurst on returns ----
            h_returns = self._rolling_hurst(
                rets,
                window=self.cfg.hurst_window,
                refit_freq=self.cfg.refit_frequency,
            )
            self._hurst_history[col] = h_returns

            # ---- Rolling Hurst on log realized vol ----
            rv = rets.rolling(
                window=self.cfg.rv_window, min_periods=self.cfg.rv_window
            ).std()
            log_rv = np.log(rv.clip(lower=1e-10))

            h_vol = self._rolling_hurst_vol(
                log_rv,
                window=self.cfg.vol_hurst_window,
                refit_freq=self.cfg.refit_frequency * 4,  # vol H is more stable
            )
            self._hvol_history[col] = h_vol

            # ---- Sub-signals ----
            sig_mom = self._momentum_signal(rets, self.cfg.momentum_lookback)
            sig_mr = self._mean_reversion_signal(
                price_col, self.cfg.mr_ma_window, self.cfg.mr_zscore_window
            )

            # ---- Regime-adaptive signal construction ----
            signal = pd.Series(0.0, index=prices.index)
            weight = pd.Series(0.0, index=prices.index)

            for i in range(len(prices)):
                h = h_returns.iloc[i]
                hv = h_vol.iloc[i]

                if np.isnan(h):
                    continue

                # Signal strength = distance from random walk
                strength = abs(h - 0.5)

                if strength < self.cfg.min_signal_strength:
                    # Too close to random walk => stay flat
                    signal.iloc[i] = 0.0
                    weight.iloc[i] = 0.0
                    continue

                # ---- Primary directional signal ----
                if h > self.cfg.trending_threshold:
                    # Trending regime: use momentum
                    primary_signal = sig_mom.iloc[i]
                elif h < self.cfg.mean_revert_threshold:
                    # Mean-reverting regime: use mean-reversion
                    primary_signal = sig_mr.iloc[i]
                else:
                    # Transition zone: blend with reduced size
                    # Linear interpolation between regimes
                    mid = 0.5
                    if h >= mid:
                        # Leaning trending
                        blend_w = (h - mid) / (self.cfg.trending_threshold - mid)
                        primary_signal = blend_w * sig_mom.iloc[i]
                    else:
                        # Leaning mean-reverting
                        blend_w = (mid - h) / (mid - self.cfg.mean_revert_threshold)
                        primary_signal = blend_w * sig_mr.iloc[i]

                # ---- Secondary vol roughness signal ----
                if not np.isnan(hv):
                    vol_sig = 0.0
                    if hv < self.cfg.vol_rough_threshold:
                        # Vol is rough => expect mean-reversion of vol
                        # Check if vol is elevated (above median)
                        if not np.isnan(log_rv.iloc[i]):
                            lookback_end = max(0, i - self.cfg.rv_window)
                            recent_lrv = log_rv.iloc[lookback_end:i + 1].dropna()
                            if len(recent_lrv) > 5:
                                vol_z = (
                                    (log_rv.iloc[i] - recent_lrv.median())
                                    / (recent_lrv.std() + 1e-10)
                                )
                                # High vol + rough => expect compression => long tilt
                                roughness_factor = (
                                    (self.cfg.vol_rough_threshold - hv)
                                    / self.cfg.vol_rough_threshold
                                )
                                vol_sig = float(
                                    np.clip(vol_z, -2.0, 2.0) / 2.0
                                    * roughness_factor
                                )
                else:
                    vol_sig = 0.0

                # ---- Combine ----
                w_primary = 1.0 - self.cfg.vol_signal_weight
                w_vol = self.cfg.vol_signal_weight
                combined = w_primary * primary_signal + w_vol * vol_sig

                # ---- Position sizing proportional to |H - 0.5| ----
                # Scale: |H - 0.5| ranges from 0 to ~0.5, normalise to [0, 1]
                size_factor = min(strength / 0.5, 1.0)

                signal.iloc[i] = float(np.sign(combined)) if abs(combined) > 1e-8 else 0.0
                weight.iloc[i] = float(
                    np.clip(abs(combined) * size_factor, 0.0, self.cfg.max_position)
                )

            # ---- Smooth the signal ----
            signal = self.exponential_smooth(signal, span=self.cfg.signal_ema_span)
            weight = self.exponential_smooth(weight, span=self.cfg.signal_ema_span)

            # Discretize signal back to {-1, 0, 1} after smoothing
            signal = signal.apply(
                lambda x: 1.0 if x > 0.3 else (-1.0 if x < -0.3 else 0.0)
            )
            weight = weight.clip(0.0, self.cfg.max_position)

            # ---- Store results ----
            if is_single:
                results["signal"] = signal
                results["weight"] = weight
            else:
                results[f"{col}_signal"] = signal
                results[f"{col}_weight"] = weight

        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_hurst_history(self, ticker: Optional[str] = None) -> pd.Series:
        """Return the rolling Hurst exponent series for a given ticker.

        Useful for visualisation and debugging.
        """
        if ticker is None:
            if len(self._hurst_history) == 1:
                return next(iter(self._hurst_history.values()))
            raise ValueError(
                "Multiple tickers available; specify one: "
                f"{list(self._hurst_history.keys())}"
            )
        if ticker not in self._hurst_history:
            raise KeyError(f"No Hurst history for ticker '{ticker}'.")
        return self._hurst_history[ticker]

    def get_vol_hurst_history(self, ticker: Optional[str] = None) -> pd.Series:
        """Return the rolling volatility Hurst exponent series."""
        if ticker is None:
            if len(self._hvol_history) == 1:
                return next(iter(self._hvol_history.values()))
            raise ValueError(
                "Multiple tickers available; specify one: "
                f"{list(self._hvol_history.keys())}"
            )
        if ticker not in self._hvol_history:
            raise KeyError(f"No vol-Hurst history for ticker '{ticker}'.")
        return self._hvol_history[ticker]

    def get_regime_labels(self, ticker: Optional[str] = None) -> pd.Series:
        """Return a categorical regime label series based on the Hurst exponent.

        Labels: 'trending', 'mean_reverting', 'random_walk'.
        """
        h = self.get_hurst_history(ticker)

        def _label(val: float) -> str:
            if np.isnan(val):
                return "unknown"
            if val > self.cfg.trending_threshold:
                return "trending"
            if val < self.cfg.mean_revert_threshold:
                return "mean_reverting"
            return "random_walk"

        return h.apply(_label)
