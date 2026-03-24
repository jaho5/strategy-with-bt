"""Leveraged multi-asset time-series momentum strategy.

Targets >45% annualized returns through volatility-targeted trend-following
across a diversified ETF universe, inspired by Moskowitz, Ooi & Pedersen (2012)
and the managed-futures / CTA industry.

Mathematical foundation
-----------------------
Time-series momentum signal for asset *i* at time *t*:

    signal_i(t) = sign(r_{i, t-252:t})

where r_{i, t-252:t} is the 12-month cumulative log-return.  Position sizing
uses volatility targeting:

    w_i(t) = sigma_target / (sigma_realized_i(t) * sqrt(252))

with sigma_realized estimated via a 63-day exponentially-weighted standard
deviation (EWM).  This ensures each asset contributes roughly equal risk.

Portfolio-level properties (under independence assumption):
    Sharpe_portfolio ~ S_asset * sqrt(N_assets)

With N=18 diversified ETFs, a per-asset Sharpe of 0.4--0.6, and a 20% vol
target per leg, the expected gross portfolio return is 40--60% annualized with
Sharpe ~1.5--2.5.

Crash protection
----------------
Two circuit breakers reduce exposure by 50% for 21 trading days:
1. **Drawdown breaker**: portfolio drawdown exceeds 15%.
2. **Correlation spike**: average pairwise rolling correlation > 0.6.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategies.base import RiskLimits, Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default ETF universe -- broad cross-asset coverage
# ---------------------------------------------------------------------------

DEFAULT_ETF_UNIVERSE: List[str] = [
    # US equity sectors
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    # International equity
    "EFA",   # EAFE (developed ex-US)
    "EEM",   # Emerging markets
    # Fixed income
    "TLT",   # 20+ Year Treasury
    "IEF",   # 7-10 Year Treasury
    "LQD",   # Investment-grade corporate
    "HYG",   # High-yield corporate
    # Commodities
    "GLD",   # Gold
    "SLV",   # Silver
    "USO",   # Oil
    "DBA",   # Agriculture
    # Real estate
    "VNQ",   # US REITs
    # Currency / inflation
    "UUP",   # US Dollar Index
    "TIP",   # TIPS (inflation-linked)
    # Volatility
    "XLU",   # Utilities (defensive proxy)
    "XLF",   # Financials
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ewm_volatility(
    returns: pd.Series,
    span: int = 63,
    min_periods: int = 21,
) -> pd.Series:
    """Exponentially-weighted annualized volatility.

    Parameters
    ----------
    returns : pd.Series
        Daily log-returns.
    span : int
        EWM span in trading days (default 63 ~ 3 months).
    min_periods : int
        Minimum observations before producing a value.

    Returns
    -------
    pd.Series
        Annualized volatility (daily EWM std * sqrt(252)).
    """
    ewm_var = returns.ewm(span=span, min_periods=min_periods).var()
    return np.sqrt(ewm_var * 252)


def _rolling_pairwise_correlation(
    returns: pd.DataFrame,
    window: int = 63,
    min_periods: int = 21,
) -> pd.Series:
    """Average pairwise rolling correlation across all columns.

    Uses the rolling correlation matrix and extracts the mean of the
    upper-triangular off-diagonal elements.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, one column per asset.
    window : int
        Rolling window for correlation estimation.
    min_periods : int
        Minimum observations.

    Returns
    -------
    pd.Series
        Average pairwise correlation indexed by date.
    """
    n_assets = returns.shape[1]
    if n_assets < 2:
        return pd.Series(0.0, index=returns.index)

    # Compute rolling correlation for each pair and average
    # For efficiency, use rolling covariance matrix approach
    avg_corr = pd.Series(np.nan, index=returns.index, dtype=float)

    # Rolling correlation via pandas built-in (works column-by-column)
    # We compute it in a vectorized way using the rolling window
    rolling_corr_sum = pd.Series(0.0, index=returns.index)
    n_pairs = 0

    cols = returns.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairwise = returns[cols[i]].rolling(
                window=window, min_periods=min_periods
            ).corr(returns[cols[j]])
            rolling_corr_sum = rolling_corr_sum.add(pairwise, fill_value=0.0)
            n_pairs += 1

    if n_pairs > 0:
        avg_corr = rolling_corr_sum / n_pairs

    return avg_corr


def _compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Compute drawdown series from an equity curve.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity (e.g. starting at 1.0).

    Returns
    -------
    pd.Series
        Drawdown as a non-positive fraction (e.g. -0.15 = 15% drawdown).
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown


# ===========================================================================
# Strategy class
# ===========================================================================

class LeveragedTrendStrategy(Strategy):
    """Leveraged multi-asset time-series momentum strategy.

    Implements the Moskowitz-Ooi-Pedersen (2012) time-series momentum
    framework with volatility targeting and crash protection, designed
    to achieve >45% annualized returns through cross-asset diversification
    and systematic leverage.

    Parameters
    ----------
    momentum_lookback : int
        Lookback window (trading days) for the trend signal.  Default 252
        (12 months), the canonical horizon from MOP (2012).
    vol_target : float
        Annualized volatility target per asset leg.  Default 0.20 (20%).
        Higher values produce more leverage and higher expected returns
        at the cost of larger drawdowns.
    vol_lookback_span : int
        EWM span for realized volatility estimation.  Default 63 (~3 months).
    max_gross_leverage : float
        Hard cap on total gross leverage (sum of absolute weights).
        Default 5.0.  Typical managed-futures portfolios run 3-4x.
    drawdown_threshold : float
        Portfolio drawdown level that triggers the crash-protection
        circuit breaker.  Default 0.15 (15%).
    correlation_threshold : float
        Average pairwise correlation level that triggers deleveraging.
        Default 0.60.
    crash_protection_days : int
        Number of trading days to maintain reduced exposure after a
        circuit breaker triggers.  Default 21 (~1 month).
    crash_reduction_factor : float
        Multiplicative reduction applied to all positions when a circuit
        breaker is active.  Default 0.50 (halve exposure).
    correlation_window : int
        Rolling window for computing average pairwise correlation.
        Default 63.
    signal_smoothing_span : int
        EWM span for smoothing the raw momentum signal to reduce
        turnover.  Default 5.  Set to 1 to disable.
    etf_universe : list[str] or None
        Ticker symbols to trade.  If None, uses the default 18-ETF
        cross-asset universe.
    """

    def __init__(
        self,
        momentum_lookback: int = 252,
        vol_target: float = 0.20,
        vol_lookback_span: int = 63,
        max_gross_leverage: float = 5.0,
        drawdown_threshold: float = 0.15,
        correlation_threshold: float = 0.60,
        crash_protection_days: int = 21,
        crash_reduction_factor: float = 0.50,
        correlation_window: int = 63,
        signal_smoothing_span: int = 5,
        etf_universe: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            name="LeveragedTrend",
            description=(
                "Leveraged multi-asset time-series momentum strategy with "
                "volatility targeting and crash protection.  Targets >45% "
                "annualized returns via cross-asset diversification."
            ),
            risk_limits=RiskLimits(
                max_position_weight=2.0,  # per-asset cap (vol-scaled)
                max_portfolio_leverage=max_gross_leverage,
            ),
        )

        # Signal parameters
        self.momentum_lookback = momentum_lookback
        self.vol_target = vol_target
        self.vol_lookback_span = vol_lookback_span
        self.max_gross_leverage = max_gross_leverage
        self.signal_smoothing_span = signal_smoothing_span

        # Crash protection parameters
        self.drawdown_threshold = drawdown_threshold
        self.correlation_threshold = correlation_threshold
        self.crash_protection_days = crash_protection_days
        self.crash_reduction_factor = crash_reduction_factor
        self.correlation_window = correlation_window

        # Universe
        self.etf_universe = etf_universe or DEFAULT_ETF_UNIVERSE

        # Fitted state
        self._asset_vol_floor: Dict[str, float] = {}
        self._asset_mean_vol: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal: signal computation
    # ------------------------------------------------------------------

    def _compute_momentum_signal(
        self, prices: pd.Series, lookback: int
    ) -> pd.Series:
        """Compute the time-series momentum signal for a single asset.

        signal(t) = sign(log(P_t / P_{t-lookback}))

        Parameters
        ----------
        prices : pd.Series
            Adjusted close prices.
        lookback : int
            Number of trading days for cumulative return.

        Returns
        -------
        pd.Series
            Values in {-1.0, 0.0, +1.0}.
        """
        log_ret_cum = np.log(prices / prices.shift(lookback))
        signal = np.sign(log_ret_cum)
        # Replace NaN (insufficient history) with 0 (flat)
        signal = signal.fillna(0.0)
        return signal

    def _compute_vol_scaled_weight(
        self,
        returns: pd.Series,
        asset: str,
    ) -> pd.Series:
        """Compute the volatility-targeting weight for a single asset.

        w(t) = sigma_target / sigma_realized(t)

        Clipped to [0.1, max_position_weight] to avoid degenerate positions
        from near-zero or exploding volatility estimates.

        Parameters
        ----------
        returns : pd.Series
            Daily log-returns.
        asset : str
            Asset name (used for looking up fitted vol floor).

        Returns
        -------
        pd.Series
            Position-sizing weight (positive, before directional signal).
        """
        realized_vol = _ewm_volatility(
            returns,
            span=self.vol_lookback_span,
            min_periods=max(21, self.vol_lookback_span // 3),
        )

        # Apply a vol floor from fitting to prevent blow-up in calm markets
        vol_floor = self._asset_vol_floor.get(asset, 0.02)
        realized_vol = realized_vol.clip(lower=vol_floor)

        weight = self.vol_target / realized_vol
        # Cap individual asset weight
        weight = weight.clip(upper=self.risk_limits.max_position_weight)
        return weight

    def _apply_crash_protection(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply drawdown and correlation circuit breakers.

        When triggered, all positions are reduced by ``crash_reduction_factor``
        for ``crash_protection_days`` trading days.

        Parameters
        ----------
        positions : pd.DataFrame
            Raw position weights (signal * vol-scaled weight) per asset.
        returns : pd.DataFrame
            Daily returns for available assets.

        Returns
        -------
        pd.DataFrame
            Adjusted positions with crash protection applied.
        """
        # --- 1. Build a synthetic portfolio equity curve ---
        # Use equal contribution from each active position
        portfolio_ret = (positions.shift(1) * returns).sum(axis=1)
        # Normalize by gross leverage to get a per-unit-leverage return
        gross_lev = positions.abs().sum(axis=1).replace(0, 1.0)
        portfolio_ret_norm = portfolio_ret / gross_lev.shift(1).replace(0, 1.0)
        equity = (1 + portfolio_ret_norm).cumprod()
        drawdown = _compute_drawdown(equity)

        # --- 2. Correlation circuit breaker ---
        avg_corr = _rolling_pairwise_correlation(
            returns,
            window=self.correlation_window,
            min_periods=max(21, self.correlation_window // 3),
        )

        # --- 3. Determine circuit-breaker trigger dates ---
        dd_triggered = drawdown < -self.drawdown_threshold
        corr_triggered = avg_corr > self.correlation_threshold

        # Either breaker fires
        any_trigger = dd_triggered | corr_triggered

        # Extend trigger window: once triggered, stay active for
        # crash_protection_days trading days
        protection_active = pd.Series(False, index=positions.index)
        cooldown_remaining = 0

        for i, dt in enumerate(positions.index):
            if any_trigger.iloc[i]:
                cooldown_remaining = self.crash_protection_days
            if cooldown_remaining > 0:
                protection_active.iloc[i] = True
                cooldown_remaining -= 1

        # --- 4. Apply reduction ---
        reduction = pd.Series(1.0, index=positions.index)
        reduction[protection_active] = self.crash_reduction_factor

        positions_adjusted = positions.mul(reduction, axis=0)

        n_protected = protection_active.sum()
        if n_protected > 0:
            logger.info(
                "Crash protection active for %d / %d trading days (%.1f%%).",
                n_protected,
                len(positions),
                100 * n_protected / len(positions),
            )

        return positions_adjusted

    # ------------------------------------------------------------------
    # Public interface: fit
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "LeveragedTrendStrategy":
        """Calibrate volatility floors and mean volatilities from training data.

        Learns per-asset realized volatility statistics so that the
        vol-targeting weights are well-behaved at inference time.  Also
        validates that the provided price data covers the required assets.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical adjusted close prices.  Columns are ticker symbols,
            index is DatetimeIndex.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        available = [c for c in prices.columns if c in self.etf_universe]
        if not available:
            # Fall back: use whatever columns are present
            available = prices.columns.tolist()
            logger.warning(
                "None of the default ETF universe found in price data. "
                "Using all %d available columns: %s",
                len(available),
                available[:5],
            )

        log_returns = np.log(prices[available] / prices[available].shift(1))

        for col in available:
            ret = log_returns[col].dropna()
            if len(ret) < 63:
                continue
            ann_vol = _ewm_volatility(ret, span=self.vol_lookback_span)
            valid_vol = ann_vol.dropna()
            if len(valid_vol) == 0:
                continue

            # Vol floor: 10th percentile of historical vol (prevents
            # weight blow-up during abnormally calm periods)
            self._asset_vol_floor[col] = float(
                max(valid_vol.quantile(0.10), 0.02)
            )
            self._asset_mean_vol[col] = float(valid_vol.mean())

        self.parameters = {
            "n_assets_fitted": len(self._asset_vol_floor),
            "mean_vol_floor": float(
                np.mean(list(self._asset_vol_floor.values()))
            ) if self._asset_vol_floor else 0.0,
            "universe": available,
        }

        self._fitted = True
        logger.info(
            "Fitted LeveragedTrend on %d assets. Mean vol floor: %.4f",
            len(self._asset_vol_floor),
            self.parameters["mean_vol_floor"],
        )
        return self

    # ------------------------------------------------------------------
    # Public interface: generate_signals
    # ------------------------------------------------------------------

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate leveraged trend-following signals for all available assets.

        For each asset in the universe (that is present in ``prices``):
        1. Compute the 12-month time-series momentum signal (+/-1).
        2. Compute the volatility-targeting weight.
        3. Combine: position = signal * weight.

        Then apply portfolio-level crash protection and leverage caps.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices.  Columns are ticker symbols, index is
            DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Columns are ``{ticker}_signal`` and ``{ticker}_weight`` for
            each traded asset, compatible with
            :meth:`Strategy.get_positions`.
        """
        self.validate_prices(prices)

        # Determine tradeable universe
        available = [c for c in prices.columns if c in self.etf_universe]
        if not available:
            available = prices.columns.tolist()

        log_returns = np.log(prices[available] / prices[available].shift(1))

        # --- Per-asset signal and weight computation ---
        signals_dict: Dict[str, pd.Series] = {}
        weights_dict: Dict[str, pd.Series] = {}
        raw_positions = pd.DataFrame(index=prices.index, dtype=float)

        for col in available:
            col_prices = prices[col].dropna()
            if len(col_prices) < self.momentum_lookback + 63:
                # Not enough history
                continue

            # 1. Momentum signal
            raw_signal = self._compute_momentum_signal(
                prices[col], self.momentum_lookback
            )

            # Smooth signal to reduce whipsaws at the zero crossing
            if self.signal_smoothing_span > 1:
                smoothed = self.exponential_smooth(
                    raw_signal, span=self.signal_smoothing_span
                )
                # Re-discretize after smoothing
                signal = pd.Series(
                    np.where(
                        smoothed > 0.1, 1.0,
                        np.where(smoothed < -0.1, -1.0, 0.0),
                    ),
                    index=raw_signal.index,
                )
            else:
                signal = raw_signal

            # 2. Volatility-scaled weight
            ret = log_returns[col]
            weight = self._compute_vol_scaled_weight(ret, col)

            # Store for output
            signals_dict[col] = signal
            weights_dict[col] = weight

            # Raw position = signal * weight
            raw_positions[col] = signal * weight

        if raw_positions.empty:
            logger.warning("No assets had sufficient history to generate signals.")
            return pd.DataFrame(index=prices.index)

        # Fill NaN with 0 (no position)
        raw_positions = raw_positions.fillna(0.0)

        # --- Portfolio-level gross leverage cap ---
        gross_leverage = raw_positions.abs().sum(axis=1)
        scale_factor = (self.max_gross_leverage / gross_leverage).clip(upper=1.0)
        raw_positions = raw_positions.mul(scale_factor, axis=0)

        # --- Crash protection ---
        active_returns = log_returns[raw_positions.columns].fillna(0.0)
        positions = self._apply_crash_protection(raw_positions, active_returns)

        # --- Build output DataFrame with _signal / _weight columns ---
        output = pd.DataFrame(index=prices.index)

        for col in positions.columns:
            # Signal: the directional component (+/-1 or 0)
            if col in signals_dict:
                output[f"{col}_signal"] = signals_dict[col].reindex(
                    prices.index, fill_value=0.0
                )
            else:
                output[f"{col}_signal"] = 0.0

            # Weight: the absolute magnitude of the final position
            # (after crash protection and leverage scaling)
            abs_position = positions[col].abs()
            output[f"{col}_weight"] = abs_position.reindex(
                prices.index, fill_value=0.0
            )

        return output
