"""Market microstructure strategy derived from daily OHLCV data.

Extracts latent microstructure signals from daily bars using techniques
from graduate-level market microstructure theory (Kyle 1985, Roll 1984,
Corwin & Schultz 2012, Amihud 2002).

Mathematical foundation
-----------------------
Even with daily data, microstructure information leaks through four channels:

1. **Amihud Illiquidity** (Amihud, 2002):
       ILLIQ_t = |r_t| / Volume_t
   Measures Kyle-style price impact -- dollars of price movement per unit
   of volume traded.

2. **Corwin-Schultz Spread Estimator** (Corwin & Schultz, 2012):
   Derives an effective bid-ask spread from daily high/low prices:
       beta  = E[ln(H_t/L_t)^2]
       gamma = [ln(H_{t,t+1} / L_{t,t+1})]^2
       alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
                 - sqrt(gamma / (3 - 2*sqrt(2)))
       Spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))

3. **Roll (1984) Spread**:
       S = 2 * sqrt(-cov(Delta_p_t, Delta_p_{t-1}))   if cov < 0
       S = 0                                           otherwise

4. **Kyle Lambda** (price impact coefficient):
   From the regression  |Delta_p| = lambda * sqrt(Volume) + epsilon,
   lambda captures the permanent price impact per unit of signed order flow.

Trading signals
---------------
*   **Liquidity momentum**: z-score of the 21-day change in rolling Amihud
    ILLIQ.  Decreasing ILLIQ (improving liquidity) is interpreted as
    institutional accumulation -> long; increasing ILLIQ -> short.

*   **Spread dynamics**: change in the Corwin-Schultz estimated spread.
    Narrowing spreads (improving market quality) -> long; widening -> short.

*   **Volume-price divergence**: OBV (on-balance volume) trend vs. price
    trend.  Price up + OBV down -> bearish divergence (short).
    Price down + OBV up -> accumulation (long).

Signals are combined with equal weight and rebalanced weekly.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Microstructure estimator functions
# ---------------------------------------------------------------------------

def _amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Rolling Amihud (2002) illiquidity ratio.

    ILLIQ_t = mean(|r_i| / Volume_i) over the trailing *window* days.

    A floor is placed on volume to avoid division by zero.

    Parameters
    ----------
    returns : pd.Series
        Log or arithmetic returns.
    volume : pd.Series
        Trading volume (shares or dollar volume).
    window : int
        Rolling window in trading days.

    Returns
    -------
    pd.Series
        Rolling ILLIQ ratio (higher = less liquid).
    """
    vol_safe = volume.replace(0, np.nan).ffill().clip(lower=1.0)
    daily_illiq = returns.abs() / vol_safe
    # Replace any inf that slipped through
    daily_illiq = daily_illiq.replace([np.inf, -np.inf], np.nan)
    return daily_illiq.rolling(window, min_periods=max(1, window // 2)).mean()


def _corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Corwin & Schultz (2012) high-low spread estimator.

    Estimates the effective bid-ask spread from consecutive daily
    high/low prices.

    Parameters
    ----------
    high, low : pd.Series
        Daily high and low prices.
    window : int
        Rolling window for averaging beta.

    Returns
    -------
    pd.Series
        Estimated proportional bid-ask spread.
    """
    # Single-day beta component: [ln(H/L)]^2
    log_hl = np.log(high / low)
    beta_daily = log_hl ** 2

    # Two-day high/low: max high over {t, t+1}, min low over {t, t+1}
    high_2d = high.rolling(2, min_periods=2).max()
    low_2d = low.rolling(2, min_periods=2).min()
    gamma = np.log(high_2d / low_2d) ** 2

    # Rolling average of beta (sum of two consecutive single-day betas)
    beta = (beta_daily + beta_daily.shift(1)).rolling(
        window, min_periods=max(1, window // 2)
    ).mean()

    # Corwin-Schultz alpha
    k = 3.0 - 2.0 * np.sqrt(2.0)  # 3 - 2*sqrt(2) ≈ 0.17157
    sqrt_2beta = np.sqrt((2.0 * beta).clip(lower=0.0))
    sqrt_beta = np.sqrt(beta.clip(lower=0.0))

    # Rolling average of gamma
    gamma_avg = gamma.rolling(window, min_periods=max(1, window // 2)).mean()

    alpha = (sqrt_2beta - sqrt_beta) / k - np.sqrt((gamma_avg / k).clip(lower=0.0))

    # Spread: 2*(exp(alpha) - 1) / (1 + exp(alpha))
    # Clip alpha to avoid numerical overflow
    alpha_clipped = alpha.clip(upper=5.0)
    exp_alpha = np.exp(alpha_clipped)
    spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)

    # Spread should be non-negative; negative values arise from noise
    spread = spread.clip(lower=0.0)
    return spread


def _roll_spread(
    prices: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Roll (1984) implied spread estimator.

    S = 2 * sqrt(-cov(Delta_p_t, Delta_p_{t-1}))  when cov < 0,
    S = 0 otherwise.

    Parameters
    ----------
    prices : pd.Series
        Close prices.
    window : int
        Rolling window for the autocovariance estimate.

    Returns
    -------
    pd.Series
        Estimated Roll spread.
    """
    dp = prices.diff()
    dp_lag = dp.shift(1)

    # Rolling covariance between consecutive price changes
    cov_series = dp.rolling(window, min_periods=max(1, window // 2)).cov(dp_lag)

    # Roll spread: 2*sqrt(-cov) when cov < 0, else 0
    neg_cov = (-cov_series).clip(lower=0.0)
    spread = 2.0 * np.sqrt(neg_cov)
    return spread


def _kyle_lambda(
    prices: pd.Series,
    volume: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Rolling Kyle lambda (price impact coefficient).

    Estimated from the regression:
        |Delta_p_t| = lambda * sqrt(Volume_t) + epsilon

    via rolling OLS (slope only).

    Parameters
    ----------
    prices : pd.Series
        Close prices.
    volume : pd.Series
        Trading volume.
    window : int
        Rolling window for the regression.

    Returns
    -------
    pd.Series
        Rolling Kyle lambda.
    """
    abs_dp = prices.diff().abs()
    sqrt_vol = np.sqrt(volume.clip(lower=1.0))

    # Rolling OLS: slope = cov(y, x) / var(x)
    cov_xy = abs_dp.rolling(window, min_periods=max(1, window // 2)).cov(sqrt_vol)
    var_x = sqrt_vol.rolling(window, min_periods=max(1, window // 2)).var()
    var_x_safe = var_x.replace(0, np.nan)

    kyle_lam = (cov_xy / var_x_safe).fillna(0.0)
    # Lambda should be non-negative (price impact is a cost)
    kyle_lam = kyle_lam.clip(lower=0.0)
    return kyle_lam


def _on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (OBV).

    OBV_t = OBV_{t-1} + sign(r_t) * Volume_t

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Trading volume.

    Returns
    -------
    pd.Series
        Cumulative OBV.
    """
    direction = np.sign(close.diff())
    # First value has no return; set direction to 0
    direction.iloc[0] = 0.0
    obv = (direction * volume).cumsum()
    return obv


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    mu = series.rolling(window, min_periods=max(1, window // 2)).mean()
    sigma = series.rolling(window, min_periods=max(1, window // 2)).std()
    sigma_safe = sigma.replace(0, np.nan)
    z = (series - mu) / sigma_safe
    return z.fillna(0.0)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope (trend) of a series against a linear time axis."""
    def _slope(y: np.ndarray) -> float:
        n = len(y)
        if n < 2 or np.any(np.isnan(y)):
            return np.nan
        x = np.arange(n, dtype=float)
        x_dm = x - x.mean()
        denom = x_dm @ x_dm
        if denom == 0:
            return 0.0
        return float(x_dm @ (y - y.mean())) / denom

    return series.rolling(window, min_periods=max(2, window // 2)).apply(
        _slope, raw=True
    )


# ---------------------------------------------------------------------------
# Helper: detect available OHLCV columns
# ---------------------------------------------------------------------------

def _detect_ohlcv_columns(
    df: pd.DataFrame,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Detect whether *df* has OHLCV columns or is a multi-ticker price frame.

    Returns
    -------
    is_ohlcv : bool
        True if the DataFrame contains OHLCV-style columns (Open, High,
        Low, Close, Volume) for a single ticker.
    open_col, high_col, low_col, close_col, volume_col : str | None
        Column names if found, else None.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    mapping = {}
    for canonical, alternatives in [
        ("open", ["open", "adj open", "adj_open"]),
        ("high", ["high", "adj high", "adj_high"]),
        ("low", ["low", "adj low", "adj_low"]),
        ("close", ["close", "adj close", "adj_close", "price"]),
        ("volume", ["volume", "vol"]),
    ]:
        for alt in alternatives:
            if alt in cols_lower:
                mapping[canonical] = cols_lower[alt]
                break

    has_close = "close" in mapping
    has_volume = "volume" in mapping
    has_hl = "high" in mapping and "low" in mapping
    is_ohlcv = has_close and has_volume

    return (
        is_ohlcv,
        mapping.get("open"),
        mapping.get("high"),
        mapping.get("low"),
        mapping.get("close"),
        mapping.get("volume"),
    )


# ---------------------------------------------------------------------------
# Multi-ticker OHLCV detection (e.g. MultiIndex columns)
# ---------------------------------------------------------------------------

def _is_multiindex_ohlcv(df: pd.DataFrame) -> bool:
    """Return True if *df* has MultiIndex columns with OHLCV fields."""
    if not isinstance(df.columns, pd.MultiIndex):
        return False
    # Check if any level contains OHLCV-like labels
    for level_values in [df.columns.get_level_values(i) for i in range(df.columns.nlevels)]:
        lower_vals = {str(v).lower() for v in level_values}
        if lower_vals & {"close", "volume", "high", "low"}:
            return True
    return False


# ===========================================================================
# Strategy class
# ===========================================================================

class MicrostructureStrategy(Strategy):
    """Market microstructure strategy using daily OHLCV data.

    Combines three microstructure signal families:

    1. **Liquidity momentum** -- z-scored change in Amihud ILLIQ ratio.
    2. **Spread dynamics** -- change in Corwin-Schultz estimated spread.
    3. **Volume-price divergence** -- OBV vs. price trend divergence.

    The strategy gracefully degrades when only Close+Volume columns are
    available (spread signals are omitted), and degrades further to a
    no-signal state when volume data is also missing.

    Parameters
    ----------
    illiq_window : int
        Rolling window for the Amihud ILLIQ ratio.  Default 21.
    spread_window : int
        Rolling window for the Corwin-Schultz spread estimator.  Default 21.
    zscore_window : int
        Window for z-scoring signal changes.  Default 63 (~ one quarter).
    trend_window : int
        Window for computing OBV and price trend slopes.  Default 21.
    rebalance_frequency : int
        Rebalance every N trading days (weekly = 5).  Default 5.
    signal_threshold : float
        Absolute composite z-score above which a position is taken.
        Default 0.5.
    signal_cap : float
        Cap on the absolute composite z-score for weight scaling.
        Default 3.0.
    """

    def __init__(
        self,
        illiq_window: int = 21,
        spread_window: int = 21,
        zscore_window: int = 63,
        trend_window: int = 21,
        rebalance_frequency: int = 5,
        signal_threshold: float = 0.5,
        signal_cap: float = 3.0,
    ) -> None:
        super().__init__(
            name="Microstructure",
            description=(
                "Market microstructure strategy combining Amihud illiquidity "
                "momentum, Corwin-Schultz spread dynamics, and OBV-price "
                "divergence signals from daily OHLCV data."
            ),
        )
        self.illiq_window = illiq_window
        self.spread_window = spread_window
        self.zscore_window = zscore_window
        self.trend_window = trend_window
        self.rebalance_frequency = rebalance_frequency
        self.signal_threshold = signal_threshold
        self.signal_cap = signal_cap

        # Populated by fit()
        self._signal_weights: Dict[str, float] = {
            "liquidity": 1.0 / 3,
            "spread": 1.0 / 3,
            "divergence": 1.0 / 3,
        }

    # -----------------------------------------------------------------
    # Single-asset signal generators
    # -----------------------------------------------------------------

    def _liquidity_momentum_signal(
        self,
        returns: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Liquidity momentum: z-scored change in Amihud ILLIQ.

        Decreasing ILLIQ (improving liquidity) -> positive signal (long).
        Increasing ILLIQ (deteriorating liquidity) -> negative signal (short).
        """
        illiq = _amihud_illiquidity(returns, volume, window=self.illiq_window)
        # Change in ILLIQ over the window
        illiq_change = illiq.diff(self.illiq_window)
        # Z-score the change; negate because *decreasing* ILLIQ is bullish
        z = _rolling_zscore(illiq_change, window=self.zscore_window)
        return -z  # negate: falling ILLIQ -> positive signal

    def _spread_dynamics_signal(
        self,
        high: pd.Series,
        low: pd.Series,
    ) -> pd.Series:
        """Spread dynamics: z-scored change in Corwin-Schultz spread.

        Narrowing spread -> positive signal (improving market quality).
        Widening spread -> negative signal.
        """
        spread = _corwin_schultz_spread(high, low, window=self.spread_window)
        spread_change = spread.diff(self.spread_window)
        z = _rolling_zscore(spread_change, window=self.zscore_window)
        return -z  # negate: falling spread -> positive signal

    def _volume_price_divergence_signal(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Volume-price divergence via OBV trend vs price trend.

        Price up + OBV down: distribution / bearish divergence -> short.
        Price down + OBV up: accumulation / bullish divergence -> long.

        The signal is the z-score of (OBV_slope_z - Price_slope_z).
        Divergence = OBV trend direction minus price trend direction.
        """
        obv = _on_balance_volume(close, volume)

        # Compute rolling slopes
        price_slope = _rolling_slope(close, window=self.trend_window)
        obv_slope = _rolling_slope(obv, window=self.trend_window)

        # Z-score the slopes for comparability
        price_slope_z = _rolling_zscore(price_slope, window=self.zscore_window)
        obv_slope_z = _rolling_zscore(obv_slope, window=self.zscore_window)

        # Divergence: OBV leading price
        # Positive when OBV trending up relative to price -> accumulation -> long
        # Negative when OBV trending down relative to price -> distribution -> short
        divergence = obv_slope_z - price_slope_z
        return divergence

    # -----------------------------------------------------------------
    # Core: compute composite signal for one asset
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Close-only proxy signal generators
    # -----------------------------------------------------------------

    def _volatility_regime_signal(self, close: pd.Series) -> pd.Series:
        """Volatility regime signal from close prices only.

        Uses change in realised volatility as a proxy for liquidity:
        declining volatility suggests improving market quality (long),
        rising volatility suggests deterioration (short).
        """
        returns = close.pct_change()
        realized_vol = returns.rolling(
            self.illiq_window, min_periods=max(1, self.illiq_window // 2)
        ).std()
        vol_change = realized_vol.diff(self.illiq_window)
        z = _rolling_zscore(vol_change, window=self.zscore_window)
        return -z  # falling vol -> positive signal

    def _roll_spread_signal(self, close: pd.Series) -> pd.Series:
        """Roll (1984) implied spread signal from close prices only.

        Narrowing Roll spread -> improving market quality -> long.
        Widening Roll spread -> deteriorating quality -> short.
        """
        spread = _roll_spread(close, window=self.spread_window)
        spread_change = spread.diff(self.spread_window)
        z = _rolling_zscore(spread_change, window=self.zscore_window)
        return -z  # falling spread -> positive signal

    def _momentum_divergence_signal(self, close: pd.Series) -> pd.Series:
        """Multi-timeframe momentum divergence from close prices only.

        Compares short-term (trend_window) vs. long-term (3x trend_window)
        momentum.  When short-term momentum leads long-term, it signals
        accumulation (long); when short-term lags, distribution (short).
        """
        short_mom = close.pct_change(self.trend_window)
        long_mom = close.pct_change(self.trend_window * 3)

        short_z = _rolling_zscore(short_mom, window=self.zscore_window)
        long_z = _rolling_zscore(long_mom, window=self.zscore_window)

        # Divergence: short-term leading long-term
        divergence = short_z - long_z
        return divergence

    def _compute_single_asset_signals(
        self,
        close: pd.Series,
        volume: Optional[pd.Series],
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Compute the composite microstructure signal for a single asset.

        Gracefully degrades:
        - Full OHLCV: all three signal components (Amihud, Corwin-Schultz, OBV).
        - Close + Volume only (no H/L): liquidity + divergence (no spread).
        - Close only (no volume): uses close-only proxies (volatility regime,
          Roll spread, momentum divergence).

        Parameters
        ----------
        close : pd.Series
            Close prices.
        volume : pd.Series or None
            Trading volume.
        high, low : pd.Series or None
            Daily high and low prices.

        Returns
        -------
        pd.Series
            Composite signal, same index as *close*.
        """
        signals: Dict[str, pd.Series] = {}
        weights: Dict[str, float] = {}

        returns = close.pct_change()

        has_volume = volume is not None and not volume.isna().all()
        has_hl = (high is not None and low is not None
                  and not high.isna().all() and not low.isna().all())

        # --- Liquidity momentum (requires volume) ---
        if has_volume:
            sig_liq = self._liquidity_momentum_signal(returns, volume)
            signals["liquidity"] = sig_liq
            weights["liquidity"] = self._signal_weights.get("liquidity", 1.0 / 3)

        # --- Spread dynamics (requires high & low) ---
        if has_hl:
            sig_spread = self._spread_dynamics_signal(high, low)
            signals["spread"] = sig_spread
            weights["spread"] = self._signal_weights.get("spread", 1.0 / 3)

        # --- Volume-price divergence (requires volume) ---
        if has_volume:
            sig_div = self._volume_price_divergence_signal(close, volume)
            signals["divergence"] = sig_div
            weights["divergence"] = self._signal_weights.get("divergence", 1.0 / 3)

        # --- Close-only fallback signals ---
        if not signals:
            logger.info(
                "No OHLCV data available; using close-only microstructure "
                "proxies (volatility regime, Roll spread, momentum divergence)."
            )
            sig_vol_regime = self._volatility_regime_signal(close)
            signals["vol_regime"] = sig_vol_regime
            weights["vol_regime"] = 1.0 / 3

            sig_roll = self._roll_spread_signal(close)
            signals["roll_spread"] = sig_roll
            weights["roll_spread"] = 1.0 / 3

            sig_mom_div = self._momentum_divergence_signal(close)
            signals["mom_divergence"] = sig_mom_div
            weights["mom_divergence"] = 1.0 / 3

        # Normalise weights to sum to 1
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        # Weighted combination
        composite = pd.Series(0.0, index=close.index)
        for name, sig in signals.items():
            # Clip individual signals to avoid outlier domination
            sig_clipped = sig.clip(lower=-self.signal_cap, upper=self.signal_cap)
            composite = composite + weights[name] * sig_clipped

        return composite

    # -----------------------------------------------------------------
    # Data unpacking helpers
    # -----------------------------------------------------------------

    def _unpack_single_ticker(
        self, prices: pd.DataFrame
    ) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """Unpack a single-ticker OHLCV DataFrame into component series."""
        is_ohlcv, _, high_col, low_col, close_col, vol_col = _detect_ohlcv_columns(prices)

        if close_col is not None:
            close = prices[close_col]
        else:
            # Fallback: use the first (or only) numeric column
            close = prices.iloc[:, 0]

        volume = prices[vol_col] if vol_col is not None else None
        high = prices[high_col] if high_col is not None else None
        low = prices[low_col] if low_col is not None else None

        return close, volume, high, low

    def _unpack_multi_ticker(
        self, prices: pd.DataFrame
    ) -> Dict[str, Dict[str, pd.Series]]:
        """Unpack a multi-ticker DataFrame.

        Handles two layouts:
        1. MultiIndex columns: (field, ticker) or (ticker, field).
        2. Flat columns where all columns are assumed to be Close prices
           for different tickers (no volume/OHLC available).

        Returns
        -------
        dict
            Mapping ticker -> {"close": Series, "volume": Series|None,
                               "high": Series|None, "low": Series|None}
        """
        result: Dict[str, Dict[str, pd.Series]] = {}

        if isinstance(prices.columns, pd.MultiIndex):
            # Determine which level contains field names (Open/High/Low/Close/Volume)
            field_level = None
            ticker_level = None
            for lvl in range(prices.columns.nlevels):
                vals_lower = {str(v).lower() for v in prices.columns.get_level_values(lvl)}
                if vals_lower & {"close", "volume", "high", "low", "open"}:
                    field_level = lvl
                    break

            if field_level is not None:
                ticker_level = 1 - field_level  # assumes 2-level MultiIndex
                tickers = prices.columns.get_level_values(ticker_level).unique()

                for ticker in tickers:
                    data: Dict[str, pd.Series] = {}
                    # Extract by ticker
                    if ticker_level == 0:
                        sub = prices[ticker]
                    else:
                        sub = prices.xs(ticker, level=ticker_level, axis=1)

                    sub_lower = {str(c).lower(): c for c in sub.columns}
                    data["close"] = sub[sub_lower["close"]] if "close" in sub_lower else sub.iloc[:, 0]

                    for field, aliases in [
                        ("volume", ["volume", "vol"]),
                        ("high", ["high"]),
                        ("low", ["low"]),
                    ]:
                        for alias in aliases:
                            if alias in sub_lower:
                                data[field] = sub[sub_lower[alias]]
                                break
                        else:
                            data[field] = None

                    result[str(ticker)] = data
            else:
                # MultiIndex but no recognisable field names -> treat each
                # top-level as a ticker with just Close prices.
                for col in prices.columns:
                    ticker = str(col)
                    result[ticker] = {
                        "close": prices[col],
                        "volume": None,
                        "high": None,
                        "low": None,
                    }
        else:
            # Flat columns: check if this is OHLCV for a single ticker
            is_ohlcv, _, _, _, _, _ = _detect_ohlcv_columns(prices)
            if is_ohlcv:
                # Single ticker with OHLCV columns
                close, volume, high, low = self._unpack_single_ticker(prices)
                result["asset"] = {
                    "close": close,
                    "volume": volume,
                    "high": high,
                    "low": low,
                }
            else:
                # Assume each column is a different ticker's Close price
                for col in prices.columns:
                    result[str(col)] = {
                        "close": prices[col],
                        "volume": None,
                        "high": None,
                        "low": None,
                    }

        return result

    # -----------------------------------------------------------------
    # Strategy interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "MicrostructureStrategy":
        """Calibrate signal combination weights from historical data.

        Uses in-sample signal-to-noise ratio (mean / std of each signal
        component) as a proxy for predictive value.  Components with
        higher SNR receive more weight.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical OHLCV or price-only data.
        ohlcv_data : pd.DataFrame, optional
            Full OHLCV MultiIndex DataFrame from yfinance.  When supplied,
            this is used instead of *prices* so that High, Low, and Volume
            columns are available for the spread and liquidity estimators.

        Returns
        -------
        self
        """
        # Prefer full OHLCV data when available
        data_for_unpack = kwargs.get("ohlcv_data", prices)
        self.validate_prices(prices)

        assets = self._unpack_multi_ticker(data_for_unpack)

        # Check whether we have any OHLCV data at all
        has_any_volume = False
        has_any_hl = False
        for components in assets.values():
            vol = components.get("volume")
            if vol is not None and not vol.isna().all():
                has_any_volume = True
            hi = components.get("high")
            lo = components.get("low")
            if (hi is not None and lo is not None
                    and not hi.isna().all() and not lo.isna().all()):
                has_any_hl = True

        if has_any_volume or has_any_hl:
            # Full or partial OHLCV available -- calibrate on OHLCV signals
            snr_accum: Dict[str, list] = {
                "liquidity": [],
                "spread": [],
                "divergence": [],
            }

            for ticker, components in assets.items():
                close = components["close"].dropna()
                volume = components.get("volume")
                high = components.get("high")
                low = components.get("low")

                if len(close) < self.zscore_window + self.illiq_window:
                    continue

                returns = close.pct_change()

                if volume is not None and not volume.isna().all():
                    volume = volume.reindex(close.index)
                    sig = self._liquidity_momentum_signal(returns, volume)
                    valid = sig.dropna()
                    if len(valid) > 0:
                        snr = abs(valid.mean()) / max(valid.std(), 1e-12)
                        snr_accum["liquidity"].append(snr)

                if (high is not None and low is not None
                        and not high.isna().all() and not low.isna().all()):
                    high_aligned = high.reindex(close.index)
                    low_aligned = low.reindex(close.index)
                    sig = self._spread_dynamics_signal(high_aligned, low_aligned)
                    valid = sig.dropna()
                    if len(valid) > 0:
                        snr = abs(valid.mean()) / max(valid.std(), 1e-12)
                        snr_accum["spread"].append(snr)

                if volume is not None and not volume.isna().all():
                    volume = volume.reindex(close.index)
                    sig = self._volume_price_divergence_signal(close, volume)
                    valid = sig.dropna()
                    if len(valid) > 0:
                        snr = abs(valid.mean()) / max(valid.std(), 1e-12)
                        snr_accum["divergence"].append(snr)

            raw_weights: Dict[str, float] = {}
            for name, values in snr_accum.items():
                raw_weights[name] = float(np.mean(values)) if values else 0.0

            total = sum(raw_weights.values())
            if total < 1e-12:
                self._signal_weights = {k: 1.0 / 3 for k in raw_weights}
            else:
                self._signal_weights = {k: v / total for k, v in raw_weights.items()}
        else:
            # Close-only mode -- calibrate weights for proxy signals
            snr_accum_proxy: Dict[str, list] = {
                "vol_regime": [],
                "roll_spread": [],
                "mom_divergence": [],
            }

            for ticker, components in assets.items():
                close = components["close"].dropna()
                if len(close) < self.zscore_window + self.illiq_window:
                    continue

                sig = self._volatility_regime_signal(close)
                valid = sig.dropna()
                if len(valid) > 0:
                    snr = abs(valid.mean()) / max(valid.std(), 1e-12)
                    snr_accum_proxy["vol_regime"].append(snr)

                sig = self._roll_spread_signal(close)
                valid = sig.dropna()
                if len(valid) > 0:
                    snr = abs(valid.mean()) / max(valid.std(), 1e-12)
                    snr_accum_proxy["roll_spread"].append(snr)

                sig = self._momentum_divergence_signal(close)
                valid = sig.dropna()
                if len(valid) > 0:
                    snr = abs(valid.mean()) / max(valid.std(), 1e-12)
                    snr_accum_proxy["mom_divergence"].append(snr)

            raw_weights = {}
            for name, values in snr_accum_proxy.items():
                raw_weights[name] = float(np.mean(values)) if values else 0.0

            total = sum(raw_weights.values())
            if total < 1e-12:
                self._signal_weights = {k: 1.0 / 3 for k in raw_weights}
            else:
                self._signal_weights = {k: v / total for k, v in raw_weights.items()}

        self.parameters = {
            "signal_weights": dict(self._signal_weights),
            "n_assets_calibrated": len(assets),
        }
        self._fitted = True
        logger.info(
            "Microstructure strategy fitted. Signal weights: %s",
            self._signal_weights,
        )
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate per-asset trading signals from OHLCV data.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV or price-only data.  Handles both single-ticker OHLCV
            DataFrames (columns: Open, High, Low, Close, Volume) and
            multi-ticker DataFrames (columns = tickers, or MultiIndex).
        ohlcv_data : pd.DataFrame, optional
            Full OHLCV MultiIndex DataFrame from yfinance.  When supplied,
            this is used instead of *prices* so that High, Low, and Volume
            columns are available for the spread and liquidity estimators.

        Returns
        -------
        pd.DataFrame
            Columns: ``<ticker>_signal`` and ``<ticker>_weight`` for each
            tradeable asset (or ``signal`` / ``weight`` for single-ticker).
            Signal values are in {-1, 0, +1}; weights are in [0, 1].
        """
        # Prefer full OHLCV data when available
        data_for_unpack = kwargs.get("ohlcv_data", prices)
        self.validate_prices(prices)

        if not self._fitted:
            warnings.warn(
                "MicrostructureStrategy has not been fitted; using equal "
                "signal weights.",
                stacklevel=2,
            )

        assets = self._unpack_multi_ticker(data_for_unpack)
        is_single_ticker = len(assets) == 1

        output = pd.DataFrame(index=prices.index)

        for ticker, components in assets.items():
            close = components["close"]
            volume = components.get("volume")
            high = components.get("high")
            low = components.get("low")

            composite = self._compute_single_asset_signals(
                close=close,
                volume=volume,
                high=high,
                low=low,
            )

            # --- Discretise into {-1, 0, +1} ---
            signal = pd.Series(0.0, index=prices.index)
            signal = signal.where(
                composite.abs() <= self.signal_threshold,
                np.sign(composite),
            )

            # --- Position weight proportional to signal strength ---
            abs_composite = composite.abs().clip(upper=self.signal_cap)
            weight = (abs_composite / self.signal_cap).clip(upper=1.0)
            # Zero weight where signal is flat
            weight = weight.where(signal != 0, 0.0)

            # --- Weekly rebalance: hold positions for rebalance_frequency days ---
            if self.rebalance_frequency > 1:
                signal = self._apply_rebalance_schedule(signal)
                weight = self._apply_rebalance_schedule(weight)

            # Store
            if is_single_ticker:
                output["signal"] = signal
                output["weight"] = weight
            else:
                output[f"{ticker}_signal"] = signal
                output[f"{ticker}_weight"] = weight

        return output

    # -----------------------------------------------------------------
    # Rebalancing helper
    # -----------------------------------------------------------------

    def _apply_rebalance_schedule(self, series: pd.Series) -> pd.Series:
        """Forward-fill signals between rebalance dates.

        Only updates positions every ``self.rebalance_frequency`` trading
        days; in between, the previous position is carried forward.
        """
        result = series.copy()
        last_rebal_idx = 0
        for i in range(len(result)):
            if i == 0:
                last_rebal_idx = 0
                continue
            if (i - last_rebal_idx) < self.rebalance_frequency:
                # Not a rebalance day -- carry forward
                result.iloc[i] = result.iloc[last_rebal_idx]
            else:
                last_rebal_idx = i
        return result
