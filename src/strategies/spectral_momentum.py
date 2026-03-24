"""Spectral analysis momentum strategy.

Uses Fourier decomposition, wavelet multi-resolution analysis, and
Hilbert-transform phase estimation to extract clean momentum signals
from noisy price series.

Mathematical foundation
-----------------------
*   DFT:  X[k] = sum_{n=0}^{N-1} x[n] e^{-j 2 pi k n / N}
*   DWT:  Daubechies-4 wavelet, 5-level decomposition
*   Hilbert transform -> analytic signal -> instantaneous phase / amplitude

The final position signal is a predictive-power-weighted combination of
three sub-signals (spectral trend, wavelet momentum, phase timing).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.fft import fft, ifft
from scipy.signal import hilbert

from src.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Try to use PyWavelets; fall back to a manual Daubechies-4 implementation.
# ---------------------------------------------------------------------------
try:
    import pywt

    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


# ---- Manual Daubechies-4 DWT (used only when pywt is not installed) ------

# Daubechies-4 scaling (low-pass) coefficients
_DB4_LO: np.ndarray = np.array(
    [
        -0.010597401784997278,
        0.032883011666982945,
        0.030841381835986965,
        -0.18703481171888114,
        -0.027983769416983849,
        0.6308807679295904,
        0.7148465705525415,
        0.23037781330885523,
    ]
)
# High-pass coefficients derived via the QMF relation: h[n] = (-1)^n g[N-1-n]
_DB4_HI: np.ndarray = _DB4_LO[::-1].copy()
_DB4_HI[1::2] *= -1


def _manual_dwt_single(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One level of Daubechies-4 discrete wavelet transform (periodic mode).

    Parameters
    ----------
    signal : 1-D array
        Input signal (length should ideally be even).

    Returns
    -------
    approx, detail : 1-D arrays
        Approximation and detail coefficients (each half the input length).
    """
    n = len(signal)
    # Periodic extension
    padded = np.concatenate([signal, signal[: len(_DB4_LO) - 1]])
    half = n // 2
    approx = np.zeros(half)
    detail = np.zeros(half)
    for i in range(half):
        idx = 2 * i
        seg = padded[idx : idx + len(_DB4_LO)]
        if len(seg) < len(_DB4_LO):
            seg = np.concatenate([seg, padded[: len(_DB4_LO) - len(seg)]])
        approx[i] = np.dot(seg, _DB4_LO)
        detail[i] = np.dot(seg, _DB4_HI)
    return approx, detail


def _manual_wavedec(signal: np.ndarray, level: int) -> List[np.ndarray]:
    """Multi-level Daubechies-4 DWT decomposition (periodic boundary).

    Returns a list ``[cA_n, cD_n, cD_{n-1}, ..., cD_1]`` matching the
    convention used by ``pywt.wavedec``.
    """
    coeffs: list[np.ndarray] = []
    a = signal.copy()
    for _ in range(level):
        # Ensure even length by zero-padding if necessary
        if len(a) % 2 != 0:
            a = np.append(a, 0.0)
        a, d = _manual_dwt_single(a)
        coeffs.append(d)
    coeffs.append(a)
    coeffs.reverse()  # [cA_n, cD_n, ..., cD_1]
    return coeffs


def _manual_idwt_single(
    approx: np.ndarray, detail: np.ndarray, output_len: int
) -> np.ndarray:
    """Inverse single-level Daubechies-4 DWT (periodic mode).

    Parameters
    ----------
    approx, detail : 1-D arrays of length N/2
    output_len : desired output length N

    Returns
    -------
    Reconstructed signal of length *output_len*.
    """
    half = len(approx)
    n = 2 * half
    # Reconstruction filters (time-reversed analysis filters)
    lo_r = _DB4_LO[::-1]
    hi_r = _DB4_HI[::-1]
    filt_len = len(lo_r)

    out = np.zeros(n)
    for i in range(half):
        for j in range(filt_len):
            idx = (2 * i + j) % n
            out[idx] += approx[i] * lo_r[j] + detail[i] * hi_r[j]
    return out[:output_len]


def _manual_waverec(coeffs: List[np.ndarray], target_len: int) -> np.ndarray:
    """Reconstruct signal from wavelet coefficients (periodic mode).

    Parameters
    ----------
    coeffs : list of arrays ``[cA_n, cD_n, ..., cD_1]``
    target_len : length of the original signal to truncate to.
    """
    a = coeffs[0]
    for d in coeffs[1:]:
        out_len = 2 * len(a)
        # Match detail length (may differ by 1 due to odd-length padding)
        if len(a) != len(d):
            min_len = min(len(a), len(d))
            a = a[:min_len]
            d = d[:min_len]
            out_len = 2 * min_len
        a = _manual_idwt_single(a, d, out_len)
    return a[:target_len]


# ---------------------------------------------------------------------------
# Wavelet convenience wrappers
# ---------------------------------------------------------------------------

def _wavedec(signal: np.ndarray, level: int) -> List[np.ndarray]:
    if _HAS_PYWT:
        return pywt.wavedec(signal, "db4", mode="periodization", level=level)
    return _manual_wavedec(signal, level)


def _waverec(coeffs: List[np.ndarray], target_len: int) -> np.ndarray:
    if _HAS_PYWT:
        rec = pywt.waverec(coeffs, "db4", mode="periodization")
        return rec[:target_len]
    return _manual_waverec(coeffs, target_len)


# ---------------------------------------------------------------------------
# Core signal-processing building blocks
# ---------------------------------------------------------------------------

def _dominant_frequencies(
    log_prices: np.ndarray,
    n_top: int = 3,
    power_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the DFT, mask of dominant frequency bins, and spectrum.

    Parameters
    ----------
    log_prices : 1-D array of log prices (length N).
    n_top : number of dominant frequency components to keep.
    power_threshold : minimum fraction of total power for a component
        to be considered signal rather than noise.

    Returns
    -------
    X : full complex DFT of *log_prices*.
    mask : boolean array (length N) – True for bins retained.
    power_spectrum : |X|^2, one-sided.
    """
    n = len(log_prices)
    X = fft(log_prices)
    power = np.abs(X) ** 2

    # Only look at positive frequencies (indices 1..N//2); DC kept always.
    total_power = power[1 : n // 2].sum()
    if total_power == 0:
        mask = np.zeros(n, dtype=bool)
        mask[0] = True
        return X, mask, power

    # Rank by power (excluding DC at index 0)
    pos_indices = np.arange(1, n // 2)
    ranked = pos_indices[np.argsort(power[pos_indices])[::-1]]

    mask = np.zeros(n, dtype=bool)
    mask[0] = True  # always keep DC
    kept = 0
    for idx in ranked:
        if kept >= n_top:
            break
        if power[idx] / total_power >= power_threshold:
            mask[idx] = True
            mask[n - idx] = True  # mirror for real-valued reconstruction
            kept += 1

    return X, mask, power


def _reconstruct_clean_trend(log_prices: np.ndarray, **kwargs) -> np.ndarray:
    """Reconstruct a noise-free trend via spectral filtering."""
    X, mask, _ = _dominant_frequencies(log_prices, **kwargs)
    X_clean = np.where(mask, X, 0.0)
    return np.real(ifft(X_clean))


def _wavelet_momentum(
    returns: np.ndarray, level: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose *returns* with DWT and return short-term / long-term signals.

    Returns
    -------
    short_mom : signal from levels 3-4 (5-20 day cycles)
    long_mom  : signal from level 5+ approximation (60+ day cycles)
    """
    n = len(returns)
    if n < 2:
        # Not enough data for any wavelet decomposition
        return np.zeros(n), np.zeros(n)
    # Ensure we have enough data for the requested decomposition depth
    min_len = 2 ** level
    if n < min_len:
        level = max(1, int(np.log2(n)))

    coeffs = _wavedec(returns, level=level)
    # coeffs layout: [cA_level, cD_level, cD_{level-1}, ..., cD_1]

    # ---- Long-term trend (approximation + deepest detail) ----
    long_coeffs = [coeffs[0]] + [
        np.zeros_like(c) for c in coeffs[1:]
    ]
    # Also add level 5 detail (index 1 when level==5) if it exists
    if level >= 5:
        long_coeffs[1] = coeffs[1].copy()
    long_mom = _waverec(long_coeffs, n)

    # ---- Short-term momentum (levels 3-4 detail) ----
    short_coeffs = [np.zeros_like(coeffs[0])] + [
        np.zeros_like(c) for c in coeffs[1:]
    ]
    # Detail at decomposition level l is stored at index (level - l + 1)
    # for the convention [cA_L, cD_L, cD_{L-1}, ..., cD_1].
    # Level 3 detail -> index level-3+1 = level-2
    # Level 4 detail -> index level-4+1 = level-3
    for target_level in (3, 4):
        idx = level - target_level + 1
        if 0 < idx < len(coeffs):
            short_coeffs[idx] = coeffs[idx].copy()
    short_mom = _waverec(short_coeffs, n)

    return short_mom, long_mom


def _hilbert_phase_amplitude(
    signal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic signal via Hilbert transform.

    Returns
    -------
    phase : instantaneous phase in radians
    amplitude : instantaneous amplitude (envelope)
    dphase : rate of change of phase (instantaneous angular frequency)
    """
    analytic = hilbert(signal)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    dphase = np.gradient(phase)
    return phase, amplitude, dphase


# ---------------------------------------------------------------------------
# Rolling information coefficient (rank IC) for adaptive weighting
# ---------------------------------------------------------------------------

def _rolling_rank_ic(
    signal: np.ndarray, forward_returns: np.ndarray, window: int
) -> np.ndarray:
    """Compute rolling Spearman rank IC between *signal* and *forward_returns*.

    Returns an array of the same length as *signal* with NaN where the
    window is incomplete.
    """
    from scipy.stats import spearmanr

    n = len(signal)
    ic = np.full(n, np.nan)
    for i in range(window, n):
        s = signal[i - window : i]
        r = forward_returns[i - window : i]
        valid = ~(np.isnan(s) | np.isnan(r))
        if valid.sum() >= 20:
            corr, _ = spearmanr(s[valid], r[valid])
            ic[i] = corr
    return ic


# ===========================================================================
# Strategy class
# ===========================================================================

class SpectralMomentumStrategy(Strategy):
    """Spectral-analysis momentum strategy.

    Combines three orthogonal signal-processing views of price dynamics:

    1. **Spectral trend** -- dominant Fourier components of log prices.
    2. **Wavelet momentum** -- multi-resolution DWT decomposition of returns.
    3. **Phase timing** -- Hilbert-transform instantaneous phase / amplitude.

    Parameters
    ----------
    spectral_window : int
        Rolling window (trading days) for the DFT analysis.  Default 252.
    n_dominant : int
        Number of top frequency components to retain.  Default 3.
    power_threshold : float
        Minimum fraction of total spectral power for a component to be
        considered signal.  Default 0.05.
    wavelet_level : int
        Number of DWT decomposition levels.  Default 5.
    ic_window : int
        Rolling window for computing rank-IC weights.  Default 126 (~6 mo).
    momentum_lookback : int
        Lookback for computing wavelet-scale momentum scores.  Default 20.
    amplitude_ma : int
        Moving-average window for smoothing amplitude trend.  Default 10.
    vol_target : float
        Annualised volatility target for position sizing.  Default 0.40.
    vol_lookback : int
        Rolling window for realised volatility estimation.  Default 63.
    """

    def __init__(
        self,
        spectral_window: int = 252,
        n_dominant: int = 3,
        power_threshold: float = 0.05,
        wavelet_level: int = 5,
        ic_window: int = 126,
        momentum_lookback: int = 20,
        amplitude_ma: int = 10,
        vol_target: float = 0.60,
        vol_lookback: int = 63,
    ) -> None:
        super().__init__(
            name="SpectralMomentum",
            description="Spectral-analysis momentum strategy combining Fourier, "
            "wavelet, and Hilbert-transform signals.",
        )
        self.spectral_window = spectral_window
        self.n_dominant = n_dominant
        self.power_threshold = power_threshold
        self.wavelet_level = wavelet_level
        self.ic_window = ic_window
        self.momentum_lookback = momentum_lookback
        self.amplitude_ma = amplitude_ma
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback

        self._component_weights: Optional[Dict[str, float]] = None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _spectral_trend_signal(self, prices: pd.Series) -> pd.Series:
        """Rolling spectral-trend signal for a single asset.

        For each date *t* with at least ``spectral_window`` history we:
        1. Take the trailing window of log-prices.
        2. Extract dominant DFT components.
        3. Reconstruct the clean trend.
        4. The signal is the slope of the clean trend over the last bar
           (first difference of the reconstructed series at the window end).
        """
        price_vals = prices.values.astype(np.float64).copy()
        # Guard against non-positive prices that would produce -inf / NaN
        price_vals[price_vals <= 0] = np.nan
        log_p = np.log(price_vals)
        n = len(log_p)
        w = self.spectral_window
        signal = np.full(n, np.nan)

        for t in range(w, n):
            window_lp = log_p[t - w : t]
            if np.any(np.isnan(window_lp)):
                continue
            # Demean for better spectral estimation
            mean_lp = window_lp.mean()
            clean = _reconstruct_clean_trend(
                window_lp - mean_lp,
                n_top=self.n_dominant,
                power_threshold=self.power_threshold,
            )
            # Slope: last vs second-to-last value of reconstructed trend
            signal[t] = clean[-1] - clean[-2]

        return pd.Series(signal, index=prices.index, name=prices.name)

    def _wavelet_momentum_signal(self, prices: pd.Series) -> pd.Series:
        """Combined wavelet momentum signal for a single asset.

        Short-term and long-term wavelet components are each turned into a
        rolling z-scored momentum score, then averaged.
        """
        price_vals = prices.values.astype(np.float64).copy()
        price_vals[price_vals <= 0] = np.nan
        log_prices = np.log(price_vals)
        log_ret = np.diff(log_prices)
        n = len(log_ret)
        lb = self.momentum_lookback

        if n < 2 ** self.wavelet_level:
            return pd.Series(
                np.full(len(prices), np.nan),
                index=prices.index,
                name=prices.name,
            )

        # Replace NaN returns with 0 for wavelet decomposition
        log_ret_clean = np.where(np.isnan(log_ret), 0.0, log_ret)
        signal = np.full(n, np.nan)

        short_mom, long_mom = _wavelet_momentum(
            log_ret_clean, level=self.wavelet_level
        )

        # Rolling z-score of the cumulative wavelet component
        for arr in (short_mom, long_mom):
            cum = np.cumsum(arr)
            # rolling mean / std
            _mean = pd.Series(cum).rolling(lb, min_periods=lb).mean().values.copy()
            _std = pd.Series(cum).rolling(lb, min_periods=lb).std().values.copy()
            _std[_std < 1e-12] = 1e-12
            z = (cum - _mean) / _std

            valid = ~np.isnan(z)
            if np.any(valid):
                signal = np.where(
                    np.isnan(signal),
                    np.where(valid, z, np.nan),
                    np.where(valid, signal + z, signal),
                )

        signal = signal / 2.0  # average of two components

        # Align: log_ret has length n-1 relative to prices
        out = np.full(len(prices), np.nan)
        out[1:] = signal
        return pd.Series(out, index=prices.index, name=prices.name)

    def _phase_timing_signal(self, prices: pd.Series) -> pd.Series:
        """Hilbert-transform phase-based timing signal for a single asset.

        Signal mapping:
        *  +1  when dphase > 0 **and** amplitude is rising
        *  -1  when dphase < 0 **and** amplitude is rising
        *   0  when amplitude is falling (trend weakening)
        """
        price_vals = prices.values.astype(np.float64).copy()
        price_vals[price_vals <= 0] = np.nan

        # Need at least 2 points for polyfit and Hilbert transform
        if len(price_vals) < 2 or np.all(np.isnan(price_vals)):
            return pd.Series(
                np.full(len(prices), np.nan),
                index=prices.index,
                name=prices.name,
            )

        # Forward-fill NaN for a usable log-price series
        pv_series = pd.Series(price_vals)
        pv_series = pv_series.ffill().bfill()
        log_p = np.log(pv_series.values)

        # Detrend with simple linear fit for cleaner analytic signal
        t_axis = np.arange(len(log_p), dtype=np.float64)
        slope, intercept = np.polyfit(t_axis, log_p, 1)
        detrended = log_p - (slope * t_axis + intercept)

        _, amplitude, dphase = _hilbert_phase_amplitude(detrended)

        # Smooth amplitude to determine trend in envelope
        amp_series = pd.Series(amplitude)
        amp_ma = amp_series.rolling(self.amplitude_ma, min_periods=1).mean().values.copy()
        amp_rising = np.gradient(amp_ma) > 0

        signal = np.where(
            amp_rising,
            np.where(dphase > 0, 1.0, -1.0),
            0.0,
        )
        return pd.Series(signal, index=prices.index, name=prices.name)

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def fit(self, data: pd.DataFrame, **kwargs) -> "SpectralMomentumStrategy":
        """Calibrate component weights from historical data.

        Uses rolling rank IC of each sub-signal against 1-day forward
        returns to set adaptive combination weights.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price data. Columns are tickers, index is
            DatetimeIndex.

        Returns
        -------
        self
        """
        weight_accum: Dict[str, list] = {
            "spectral": [],
            "wavelet": [],
            "phase": [],
        }

        for col in data.columns:
            prices = data[col].dropna()
            if len(prices) < self.spectral_window + self.ic_window:
                continue

            fwd_ret = prices.pct_change().shift(-1).values

            sig_spectral = self._spectral_trend_signal(prices).values
            sig_wavelet = self._wavelet_momentum_signal(prices).values
            sig_phase = self._phase_timing_signal(prices).values

            for name, sig in [
                ("spectral", sig_spectral),
                ("wavelet", sig_wavelet),
                ("phase", sig_phase),
            ]:
                ic = _rolling_rank_ic(sig, fwd_ret, self.ic_window)
                mean_ic = np.nanmean(np.abs(ic))
                if np.isfinite(mean_ic):
                    weight_accum[name].append(mean_ic)

        # Derive weights (proportional to mean |IC|, with a floor)
        raw: Dict[str, float] = {}
        for name, values in weight_accum.items():
            raw[name] = float(np.mean(values)) if values else 1.0 / 3

        total = sum(raw.values())
        if total < 1e-12:
            self._component_weights = {k: 1.0 / 3 for k in raw}
        else:
            self._component_weights = {k: v / total for k, v in raw.items()}

        self._fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate per-asset position signals with vol-targeting overlay.

        Parameters
        ----------
        data : pd.DataFrame
            Price data (columns = tickers, index = DatetimeIndex).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for each
            ticker.  Signal direction in {-1, 0, +1}; weight is vol-targeted
            position size.
        """
        if not self._fitted:
            # If generate_signals is called without prior fit, use equal
            # weights (graceful degradation).
            self._component_weights = {
                "spectral": 1.0 / 3,
                "wavelet": 1.0 / 3,
                "phase": 1.0 / 3,
            }

        w = self._component_weights

        # Compute asset-level rolling volatility for vol-targeting
        returns = data.pct_change()
        asset_vol = returns.rolling(
            self.vol_lookback, min_periods=self.vol_lookback // 2
        ).std() * np.sqrt(252)
        asset_vol = asset_vol.clip(lower=0.01)  # floor to avoid inf

        # Build output with _signal and _weight columns
        output_cols = []
        for col in data.columns:
            output_cols.extend([f"{col}_signal", f"{col}_weight"])
        signals_df = pd.DataFrame(0.0, index=data.index, columns=output_cols)

        for col in data.columns:
            prices = data[col].dropna()
            if len(prices) < self.spectral_window:
                continue

            s_spectral = self._spectral_trend_signal(prices)
            s_wavelet = self._wavelet_momentum_signal(prices)
            s_phase = self._phase_timing_signal(prices)

            # Normalise spectral & wavelet signals to [-1, 1] via tanh
            s_spectral_norm = _tanh_normalise(s_spectral)
            s_wavelet_norm = _tanh_normalise(s_wavelet)
            # Phase signal is already in {-1, 0, 1}

            composite = (
                w["spectral"] * s_spectral_norm
                + w["wavelet"] * s_wavelet_norm
                + w["phase"] * s_phase
            )

            # Discretise: long (+1) / short (-1) / flat (0)
            discretised = pd.Series(
                np.where(
                    composite > 0.15,
                    1.0,
                    np.where(composite < -0.15, -1.0, 0.0),
                ),
                index=composite.index,
            )
            discretised = discretised.reindex(data.index, fill_value=0.0)

            # Vol-targeting: scale position size to achieve target vol
            vol_scale = (self.vol_target / asset_vol[col]).clip(upper=8.0)
            weight = vol_scale * discretised.abs()
            weight = weight.clip(lower=0.0, upper=5.0)

            signals_df[f"{col}_signal"] = discretised
            signals_df[f"{col}_weight"] = weight

        return signals_df


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _tanh_normalise(s: pd.Series) -> pd.Series:
    """Map an unbounded signal to (-1, 1) via tanh after robust z-scoring."""
    values = s.values.astype(np.float64)
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return s
    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    mad = mad if mad > 1e-12 else 1.0
    z = (values - med) / (1.4826 * mad)  # 1.4826 scales MAD to std-equiv
    return pd.Series(np.tanh(z), index=s.index, name=s.name)
