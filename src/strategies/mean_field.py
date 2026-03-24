"""Mean Field Game strategy for modeling crowd behavior in financial markets.

Implements a simplified Mean Field Game (MFG) framework -- Lasry & Lions (2007)
-- to model the aggregate interaction of infinitely many rational agents and
exploit the resulting crowd dynamics.

Mathematical foundation
-----------------------
The full MFG system is a coupled forward-backward PDE:

    Forward Fokker-Planck (distribution of agents):
        dm/dt = (1/2) sigma^2 d^2m/dx^2 - d(alpha * m)/dx

    Backward Hamilton-Jacobi-Bellman (value function):
        -dV/dt = (1/2) sigma^2 d^2V/dx^2 + H(x, dV/dx, m)

In financial markets this captures the tension between two agent
populations: *trend-followers* (momentum crowd) and *mean-reverters*
(contrarian crowd).  Their relative mass determines the dominant regime
and, crucially, when that regime is fragile enough to exploit.

Simplified implementation (no PDE solver)
-----------------------------------------
Rather than solving the forward-backward system numerically, we estimate
the crowd composition and herding intensity from observable market
statistics, then take the *contrarian-to-crowd* position -- which
corresponds to the Nash equilibrium response in the MFG framework.

1. **Crowd behaviour proxy** -- rolling autocorrelation structure of
   returns at lags 1..5.  Positive weighted autocorrelation implies a
   momentum-dominated crowd; negative implies mean-reversion crowd.

2. **Herding index** -- rolling cross-sectional dispersion of returns.
   Low dispersion signals high herding (all agents doing the same thing),
   which is a fragile state.  We z-score dispersion against its own
   history to identify extremes.

3. **Contrarian-to-crowd signal** -- the optimal MFG response is always
   opposite to the crowd consensus:
   - When herding is extreme AND the crowd follows momentum: expect a
     momentum crash -> mean-revert.
   - When dispersion is high AND the crowd mean-reverts: expect trends
     to persist -> follow momentum.

4. **Position sizing** -- inverse-volatility scaling, with the signal
   strength modulated by how extreme the crowd/herding readings are.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crowd behaviour estimation
# ---------------------------------------------------------------------------

def _rolling_autocorrelation(
    returns: pd.Series,
    lag: int,
    window: int,
) -> pd.Series:
    """Rolling sample autocorrelation at a given lag.

    Uses the standard Pearson correlation between r_t and r_{t-lag}
    computed over a rolling window.
    """
    lagged = returns.shift(lag)
    rolling_corr = returns.rolling(window, min_periods=window // 2).corr(lagged)
    return rolling_corr


def _weighted_autocorrelation_profile(
    returns: pd.Series,
    max_lag: int = 5,
    window: int = 63,
) -> pd.Series:
    """Aggregate autocorrelation signal across lags 1..max_lag.

    Weights decay as 1/lag so that short-lag autocorrelations (more
    statistically stable, higher power) receive greater influence.

    Returns
    -------
    pd.Series
        Weighted average autocorrelation.  Positive values indicate a
        momentum-dominated crowd; negative values a mean-reversion crowd.
    """
    weights = np.array([1.0 / k for k in range(1, max_lag + 1)])
    weights /= weights.sum()

    ac_sum = pd.Series(0.0, index=returns.index)
    for lag_idx, lag in enumerate(range(1, max_lag + 1)):
        ac = _rolling_autocorrelation(returns, lag=lag, window=window)
        ac_sum = ac_sum + weights[lag_idx] * ac.fillna(0.0)

    return ac_sum


def _cross_sectional_dispersion(
    returns: pd.DataFrame,
    window: int = 21,
) -> pd.Series:
    """Rolling cross-sectional standard deviation of asset returns.

    At each date t, compute std across the N assets of their trailing
    ``window``-day cumulative returns.  Low dispersion = high herding.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, one column per asset.
    window : int
        Lookback for cumulative returns used in the cross-sectional
        dispersion calculation.

    Returns
    -------
    pd.Series
        Cross-sectional dispersion indexed by date.
    """
    # Use cumulative returns over the window for a more stable measure
    cum_returns = returns.rolling(window, min_periods=window // 2).sum()
    dispersion = cum_returns.std(axis=1)
    return dispersion


def _volume_trend_ratio(
    prices: pd.DataFrame,
    volume: Optional[pd.DataFrame],
    window: int = 21,
) -> Optional[pd.Series]:
    """Ratio of average volume on trend-continuation days to reversal days.

    A high ratio means most volume occurs when prices continue their
    recent direction -- proxy for momentum-crowd dominance.

    Returns None if volume data is not available.
    """
    if volume is None or volume.empty:
        return None

    returns = prices.pct_change()
    # Use first column as market proxy if multiple assets
    if returns.ndim == 2:
        mkt_ret = returns.mean(axis=1)
        mkt_vol = volume.mean(axis=1)
    else:
        mkt_ret = returns
        mkt_vol = volume

    # Prior-day direction
    prior_sign = np.sign(mkt_ret.shift(1))
    current_sign = np.sign(mkt_ret)

    # Trend continuation: same sign as yesterday
    trend_day = (prior_sign == current_sign) & (prior_sign != 0)
    reversal_day = (prior_sign != current_sign) & (prior_sign != 0) & (current_sign != 0)

    trend_volume = mkt_vol.where(trend_day).rolling(
        window, min_periods=window // 2
    ).mean()
    reversal_volume = mkt_vol.where(reversal_day).rolling(
        window, min_periods=window // 2
    ).mean()

    # Avoid division by zero
    reversal_volume = reversal_volume.replace(0, np.nan)
    ratio = trend_volume / reversal_volume
    return ratio


# ---------------------------------------------------------------------------
# Herding index
# ---------------------------------------------------------------------------

def _herding_index(
    dispersion: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """Z-score of cross-sectional dispersion (inverted).

    *Low* dispersion = *high* herding, so the herding index is the
    negative z-score of dispersion:

        herding_z = -(dispersion - mu) / sigma

    Positive values indicate above-average herding (fragile market).
    Negative values indicate above-average dispersion (diverse views).
    """
    mu = dispersion.rolling(lookback, min_periods=lookback // 2).mean()
    sigma = dispersion.rolling(lookback, min_periods=lookback // 2).std()
    sigma = sigma.replace(0, np.nan)
    herding_z = -(dispersion - mu) / sigma
    return herding_z


# ---------------------------------------------------------------------------
# Composite signal construction
# ---------------------------------------------------------------------------

def _crowd_regime_signal(
    autocorr_signal: pd.Series,
    herding_z: pd.Series,
    volume_ratio: Optional[pd.Series] = None,
    autocorr_threshold: float = 0.5,
    herding_threshold: float = 1.0,
) -> pd.Series:
    """Combine crowd-behaviour proxies into a single contrarian signal.

    Decision logic (the MFG Nash equilibrium response):

    1. **Strong momentum crowd + high herding** (autocorr_z > threshold
       AND herding_z > threshold):
       Signal = -1 (contrarian: mean-revert, expect momentum crash).

    2. **Strong MR crowd + high herding** (autocorr_z < -threshold AND
       herding_z > threshold):
       Signal = +1 (contrarian: follow momentum, expect trend persistence).

    3. **No herding extreme** (|herding_z| < threshold):
       Signal magnitude proportional to -autocorr_z (gentle lean against
       the crowd), scaled by herding intensity.

    4. **High dispersion** (herding_z < -threshold):
       Signal = 0 (diverse crowd = no exploitable consensus).

    The signal is continuous in [-1, 1].

    Parameters
    ----------
    autocorr_signal : pd.Series
        Weighted autocorrelation (positive = momentum crowd).
    herding_z : pd.Series
        Herding z-score (positive = high herding).
    volume_ratio : pd.Series or None
        Volume trend-to-reversal ratio (optional reinforcement).
    autocorr_threshold : float
        Z-score threshold to identify strong crowd regime.
    herding_threshold : float
        Z-score threshold to identify extreme herding/dispersion.

    Returns
    -------
    pd.Series
        Contrarian signal in [-1, 1].
    """
    # Z-score the autocorrelation signal for comparability
    ac_mu = autocorr_signal.rolling(252, min_periods=63).mean()
    ac_sigma = autocorr_signal.rolling(252, min_periods=63).std().replace(0, np.nan)
    ac_z = (autocorr_signal - ac_mu) / ac_sigma
    ac_z = ac_z.fillna(0.0)

    signal = pd.Series(0.0, index=autocorr_signal.index)

    # Core contrarian logic: lean against the crowd, scaled by herding
    # The stronger the herding, the more confident we are in the
    # contrarian bet.
    herding_weight = herding_z.clip(lower=0.0)  # only use positive herding
    herding_weight = herding_weight / (herding_weight + 1.0)  # squash to [0, 1)

    # Base signal: negative of crowd direction, modulated by herding
    signal = -ac_z * herding_weight

    # Strengthen when both are extreme
    extreme_mom_crowd = (ac_z > autocorr_threshold) & (herding_z > herding_threshold)
    extreme_mr_crowd = (ac_z < -autocorr_threshold) & (herding_z > herding_threshold)

    # Overwrite with stronger conviction on extremes
    signal = signal.where(
        ~extreme_mom_crowd,
        -herding_weight.clip(upper=1.0),  # strong mean-revert
    )
    signal = signal.where(
        ~extreme_mr_crowd,
        herding_weight.clip(upper=1.0),   # strong momentum
    )

    # Suppress signal when crowd is highly dispersed (no consensus)
    highly_dispersed = herding_z < -herding_threshold
    signal = signal.where(~highly_dispersed, 0.0)

    # Optional: reinforce with volume ratio
    if volume_ratio is not None:
        # High volume ratio confirms momentum crowd; low confirms MR crowd
        vol_mu = volume_ratio.rolling(252, min_periods=63).mean()
        vol_sigma = volume_ratio.rolling(252, min_periods=63).std().replace(0, np.nan)
        vol_z = ((volume_ratio - vol_mu) / vol_sigma).fillna(0.0)
        # Small reinforcement: nudge signal by up to 20%
        reinforcement = -vol_z * 0.2
        signal = signal + reinforcement

    # Clip to [-1, 1]
    signal = signal.clip(lower=-1.0, upper=1.0)

    return signal


# ===========================================================================
# Strategy class
# ===========================================================================

class MeanFieldStrategy(Strategy):
    """Mean Field Game strategy: trade against the crowd consensus.

    This strategy estimates the composition of the "crowd" -- whether
    market participants are predominantly following momentum or
    mean-reverting -- and takes the contrarian position, which
    corresponds to the Nash equilibrium response in the MFG framework.

    Parameters
    ----------
    autocorr_window : int
        Rolling window for autocorrelation estimation (default 63,
        ~3 months).
    max_lag : int
        Maximum lag for the autocorrelation profile (default 5).
    dispersion_window : int
        Rolling window for cross-sectional dispersion (default 21,
        ~1 month).
    herding_lookback : int
        Lookback for z-scoring the herding index (default 252,
        ~1 year).
    vol_target : float
        Annualised volatility target for position sizing (default 0.60).
    vol_lookback : int
        Rolling window for volatility estimation in position sizing
        (default 63).
    autocorr_threshold : float
        Z-score threshold to identify a strong crowd autocorrelation
        regime (default 0.5).
    herding_threshold : float
        Z-score threshold to identify extreme herding or dispersion
        (default 1.0).
    signal_smoothing_span : int
        EMA span for smoothing the final signal to reduce turnover
        (default 10).
    use_volume : bool
        Whether to incorporate volume data as an auxiliary crowd
        indicator (default True). Requires volume data passed via
        ``kwargs['volume']`` in ``generate_signals``.
    """

    def __init__(
        self,
        autocorr_window: int = 63,
        max_lag: int = 5,
        dispersion_window: int = 21,
        herding_lookback: int = 252,
        vol_target: float = 0.60,
        vol_lookback: int = 63,
        autocorr_threshold: float = 0.5,
        herding_threshold: float = 1.0,
        signal_smoothing_span: int = 10,
        use_volume: bool = True,
    ) -> None:
        super().__init__(
            name="MeanField",
            description=(
                "Mean Field Game strategy: estimates crowd composition "
                "(momentum vs mean-reversion) and herding intensity from "
                "autocorrelation structure and cross-sectional dispersion, "
                "then takes the Nash-equilibrium contrarian position."
            ),
        )
        self.autocorr_window = autocorr_window
        self.max_lag = max_lag
        self.dispersion_window = dispersion_window
        self.herding_lookback = herding_lookback
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.autocorr_threshold = autocorr_threshold
        self.herding_threshold = herding_threshold
        self.signal_smoothing_span = signal_smoothing_span
        self.use_volume = use_volume

        # Calibrated during fit()
        self._ac_baseline_mean: Optional[float] = None
        self._ac_baseline_std: Optional[float] = None
        self._herding_baseline_mean: Optional[float] = None
        self._herding_baseline_std: Optional[float] = None
        self._avg_dispersion: Optional[float] = None

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "MeanFieldStrategy":
        """Calibrate crowd behaviour baselines on historical data.

        Computes the long-run statistics (mean, std) of the autocorrelation
        profile and herding index so that out-of-sample z-scores are
        anchored to the training distribution.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (columns = tickers, index = dates).
        **kwargs
            volume : pd.DataFrame, optional
                Volume data aligned with prices.  Used to calibrate
                volume-ratio statistics if ``use_volume=True``.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        returns = prices.pct_change().dropna(how="all")

        # -- Autocorrelation baseline ------------------------------------
        # Use equal-weight market return for the aggregate autocorrelation
        mkt_returns = returns.mean(axis=1)
        ac_profile = _weighted_autocorrelation_profile(
            mkt_returns,
            max_lag=self.max_lag,
            window=self.autocorr_window,
        )
        ac_clean = ac_profile.dropna()

        if len(ac_clean) < self.autocorr_window:
            warnings.warn(
                f"Only {len(ac_clean)} valid autocorrelation observations; "
                "baseline statistics may be unreliable.",
                stacklevel=2,
            )

        self._ac_baseline_mean = float(ac_clean.mean()) if len(ac_clean) > 0 else 0.0
        self._ac_baseline_std = float(ac_clean.std()) if len(ac_clean) > 1 else 1.0
        if self._ac_baseline_std < 1e-12:
            self._ac_baseline_std = 1.0

        # -- Herding baseline --------------------------------------------
        dispersion = _cross_sectional_dispersion(
            returns, window=self.dispersion_window
        )
        herding_z = _herding_index(dispersion, lookback=self.herding_lookback)
        hz_clean = herding_z.dropna()

        self._herding_baseline_mean = float(hz_clean.mean()) if len(hz_clean) > 0 else 0.0
        self._herding_baseline_std = float(hz_clean.std()) if len(hz_clean) > 1 else 1.0
        if self._herding_baseline_std < 1e-12:
            self._herding_baseline_std = 1.0

        self._avg_dispersion = float(dispersion.mean()) if len(dispersion.dropna()) > 0 else 1.0

        # Store calibrated parameters
        self.parameters = {
            "ac_baseline_mean": self._ac_baseline_mean,
            "ac_baseline_std": self._ac_baseline_std,
            "herding_baseline_mean": self._herding_baseline_mean,
            "herding_baseline_std": self._herding_baseline_std,
            "avg_dispersion": self._avg_dispersion,
        }

        logger.info(
            "MeanField fitted: ac_baseline=%.4f +/- %.4f, "
            "herding_baseline=%.4f +/- %.4f, avg_disp=%.6f",
            self._ac_baseline_mean,
            self._ac_baseline_std,
            self._herding_baseline_mean,
            self._herding_baseline_std,
            self._avg_dispersion,
        )

        self._fitted = True
        return self

    # -----------------------------------------------------------------
    # Generate signals
    # -----------------------------------------------------------------

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate contrarian-to-crowd signals for each asset.

        Pipeline:
        1. Compute aggregate autocorrelation profile (crowd composition).
        2. Compute cross-sectional dispersion and herding index.
        3. Optionally compute volume trend-to-reversal ratio.
        4. Combine into a single contrarian signal via MFG logic.
        5. Smooth the signal to reduce turnover.
        6. Size positions via inverse-volatility scaling with vol target.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (columns = tickers, index = DatetimeIndex).
        **kwargs
            volume : pd.DataFrame, optional
                Volume data aligned with prices.

        Returns
        -------
        pd.DataFrame
            Columns: ``{ticker}_signal``, ``{ticker}_weight`` for each
            ticker, plus diagnostic columns ``crowd_autocorr``,
            ``herding_z``, and ``composite_signal``.
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        returns = prices.pct_change()
        volume = kwargs.get("volume", None)

        # -- Step 1: Crowd composition (autocorrelation) -----------------
        mkt_returns = returns.mean(axis=1)
        autocorr_signal = _weighted_autocorrelation_profile(
            mkt_returns,
            max_lag=self.max_lag,
            window=self.autocorr_window,
        )

        # -- Step 2: Herding index ---------------------------------------
        dispersion = _cross_sectional_dispersion(
            returns, window=self.dispersion_window
        )
        herding_z = _herding_index(dispersion, lookback=self.herding_lookback)

        # -- Step 3: Volume ratio (optional) -----------------------------
        volume_ratio: Optional[pd.Series] = None
        if self.use_volume and volume is not None:
            volume_ratio = _volume_trend_ratio(
                prices, volume, window=self.dispersion_window
            )

        # -- Step 4: Composite contrarian signal -------------------------
        composite = _crowd_regime_signal(
            autocorr_signal=autocorr_signal,
            herding_z=herding_z,
            volume_ratio=volume_ratio,
            autocorr_threshold=self.autocorr_threshold,
            herding_threshold=self.herding_threshold,
        )

        # -- Step 5: Smooth signal to reduce turnover --------------------
        smoothed = self.exponential_smooth(
            composite, span=self.signal_smoothing_span
        )

        # -- Step 6: Per-asset signals and inverse-vol weights -----------
        # The composite signal is market-level; we apply it to each asset
        # with asset-level volatility scaling.
        result = pd.DataFrame(index=prices.index)

        # Individual asset rolling volatility for weighting
        asset_vol = returns.rolling(
            self.vol_lookback, min_periods=self.vol_lookback // 2
        ).std() * np.sqrt(252)
        asset_vol = asset_vol.clip(lower=0.01)  # floor to avoid inf

        # Inverse-vol weights (normalised across assets)
        inv_vol = 1.0 / asset_vol
        inv_vol_sum = inv_vol.sum(axis=1)
        inv_vol_sum = inv_vol_sum.replace(0, np.nan)

        for col in prices.columns:
            # Signal direction: same composite signal for all assets
            # (market-level crowd dynamics), but sign can differ based
            # on individual asset's momentum character.
            asset_ac = _weighted_autocorrelation_profile(
                returns[col],
                max_lag=self.max_lag,
                window=self.autocorr_window,
            )
            # Blend: 70% market-level signal, 30% asset-specific
            asset_signal = 0.7 * smoothed + 0.3 * self.exponential_smooth(
                _crowd_regime_signal(
                    autocorr_signal=asset_ac,
                    herding_z=herding_z,
                    volume_ratio=volume_ratio,
                    autocorr_threshold=self.autocorr_threshold,
                    herding_threshold=self.herding_threshold,
                ),
                span=self.signal_smoothing_span,
            )
            asset_signal = asset_signal.clip(lower=-1.0, upper=1.0)

            # Discretise signal: sign for direction
            direction = np.sign(asset_signal)

            # Weight: inverse-vol share * vol-target scaling * signal magnitude
            iv_share = inv_vol[col] / inv_vol_sum  # fraction of risk budget
            vol_scale = (self.vol_target / asset_vol[col]).clip(upper=8.0)
            weight = iv_share * vol_scale * asset_signal.abs()
            weight = weight.clip(lower=0.0, upper=4.0)

            result[f"{col}_signal"] = direction
            result[f"{col}_weight"] = weight

        # -- Diagnostic columns ------------------------------------------
        result["crowd_autocorr"] = autocorr_signal
        result["herding_z"] = herding_z
        result["composite_signal"] = smoothed

        return result
