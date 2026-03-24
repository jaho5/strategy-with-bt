"""Abstract base class for all trading strategies.

Provides a contract that every strategy must implement (generate_signals, fit)
along with production-grade utility methods for position sizing, signal
smoothing, regime detection, and risk management.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supporting data structures
# ---------------------------------------------------------------------------

class Regime(Enum):
    """Market regime labels."""
    BULL = auto()
    BEAR = auto()
    SIDEWAYS = auto()
    HIGH_VOL = auto()
    LOW_VOL = auto()


@dataclass
class RiskLimits:
    """Per-strategy risk-management guardrails.

    All thresholds are expressed as positive floats (e.g. 0.05 = 5 %).
    Set a field to ``None`` to disable that check.
    """
    max_position_weight: float = 1.0
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_portfolio_leverage: float = 1.0
    max_drawdown_pct: Optional[float] = None


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """Base class for all trading strategies.

    Subclasses **must** implement:
    * ``generate_signals`` -- produce a DataFrame of directional signals and
      position-sizing weights.
    * ``fit`` -- calibrate / train strategy parameters on historical data.

    The base class provides a library of composable helpers that subclasses
    can call from their implementations:
    * Position sizing  (equal-weight, inverse-vol / risk-parity, Kelly)
    * Signal smoothing (exponential, simple Kalman filter)
    * Regime detection (volatility-based, trend-based)
    * Risk management  (stop-loss, take-profit, leverage caps, drawdown
      circuit-breaker)
    """

    def __init__(
        self,
        name: str,
        description: str,
        risk_limits: Optional[RiskLimits] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters: Dict[str, Any] = {}
        self.risk_limits = risk_limits or RiskLimits()
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals from a price DataFrame.

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV or adjusted-close prices indexed by datetime with one
            column per ticker.

        Returns
        -------
        pd.DataFrame
            Must contain, at minimum, for each ticker:
            * ``signal``:  -1 (short), 0 (flat), or 1 (long)
            * ``weight``:  position-sizing weight in [0, 1]
        """
        ...

    @abstractmethod
    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "Strategy":
        """Fit / calibrate strategy parameters on training data.

        Implementations should set ``self._fitted = True`` before returning
        and should store learned parameters in ``self.parameters``.

        Returns ``self`` for chaining.
        """
        ...

    # ------------------------------------------------------------------
    # Signal -> Position conversion
    # ------------------------------------------------------------------

    def get_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert raw signals into risk-managed positions.

        Multiplies ``signal`` columns by their corresponding ``weight``
        columns, then applies the risk limits defined on this strategy.
        """
        signal_cols = [c for c in signals.columns if c.endswith("_signal") or c == "signal"]
        weight_cols = [c for c in signals.columns if c.endswith("_weight") or c == "weight"]

        if not signal_cols or not weight_cols:
            raise ValueError(
                "signals DataFrame must contain columns ending in "
                "'_signal'/'_weight' (or 'signal'/'weight')."
            )

        positions = pd.DataFrame(index=signals.index)

        if "signal" in signal_cols and "weight" in weight_cols:
            # Single-ticker shorthand
            positions["position"] = signals["signal"] * signals["weight"]
        else:
            for sc in signal_cols:
                ticker = sc.replace("_signal", "")
                wc = f"{ticker}_weight"
                if wc in signals.columns:
                    positions[ticker] = signals[sc] * signals[wc]

        # Enforce risk limits
        positions = self._apply_position_limits(positions)
        return positions

    # ------------------------------------------------------------------
    # Position-sizing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def equal_weight(n_assets: int) -> np.ndarray:
        """Return an equal-weight vector that sums to 1."""
        if n_assets <= 0:
            raise ValueError("n_assets must be positive")
        return np.full(n_assets, 1.0 / n_assets)

    @staticmethod
    def risk_parity_weights(
        returns: pd.DataFrame,
        lookback: int = 60,
        min_weight: float = 0.0,
    ) -> pd.Series:
        """Inverse-volatility (risk-parity) weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Period returns, one column per asset.
        lookback : int
            Rolling window for volatility estimation.
        min_weight : float
            Floor for any single weight (before re-normalisation).

        Returns
        -------
        pd.Series
            Weights indexed by asset name, summing to 1.
        """
        vol = returns.iloc[-lookback:].std()
        vol = vol.replace(0, np.nan)
        if vol.isna().all():
            warnings.warn("All volatilities are zero/NaN; falling back to equal weight.")
            return pd.Series(
                Strategy.equal_weight(len(returns.columns)),
                index=returns.columns,
            )
        inv_vol = 1.0 / vol
        inv_vol = inv_vol.fillna(0)
        weights = inv_vol / inv_vol.sum()
        weights = weights.clip(lower=min_weight)
        # Re-normalise after clipping
        total = weights.sum()
        if total > 0:
            weights /= total
        return weights

    @staticmethod
    def kelly_criterion_weights(
        returns: pd.DataFrame,
        lookback: int = 252,
        fraction: float = 0.5,
    ) -> pd.Series:
        """Half-Kelly (or fractional-Kelly) position sizing.

        Uses the simplified single-asset Kelly formula per column:
            f* = mu / sigma^2

        then scales by ``fraction`` (default 0.5 = half-Kelly) and
        normalises so that absolute weights sum to 1.

        Parameters
        ----------
        returns : pd.DataFrame
            Period returns, one column per asset.
        lookback : int
            Window for mean / variance estimation.
        fraction : float
            Kelly fraction (1.0 = full Kelly, 0.5 = half Kelly).
        """
        window = returns.iloc[-lookback:]
        mu = window.mean()
        var = window.var()
        var = var.replace(0, np.nan)

        kelly = (mu / var) * fraction
        kelly = kelly.fillna(0)

        # Normalise so |weights| sum to 1
        abs_sum = kelly.abs().sum()
        if abs_sum > 0:
            kelly /= abs_sum
        return kelly

    # ------------------------------------------------------------------
    # Signal-smoothing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def exponential_smooth(
        signal: pd.Series,
        span: int = 10,
    ) -> pd.Series:
        """Exponentially-weighted moving average smoothing.

        Parameters
        ----------
        signal : pd.Series
            Raw signal series.
        span : int
            EMA span (decay factor = 2 / (span + 1)).
        """
        return signal.ewm(span=span, adjust=False).mean()

    @staticmethod
    def kalman_smooth(
        signal: pd.Series,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-1,
    ) -> pd.Series:
        """Univariate Kalman filter for signal smoothing.

        Implements a minimal constant-velocity Kalman filter without
        external dependencies beyond NumPy.

        Parameters
        ----------
        signal : pd.Series
            Raw (noisy) signal.
        process_noise : float
            Variance of the process (state transition) noise.
        measurement_noise : float
            Variance of the measurement (observation) noise.

        Returns
        -------
        pd.Series
            Filtered signal, same index as input.
        """
        values = signal.values.astype(float)
        n = len(values)
        filtered = np.empty(n)

        # Initial state
        x_hat = values[0] if not np.isnan(values[0]) else 0.0
        p = 1.0

        Q = process_noise
        R = measurement_noise

        for i in range(n):
            # Prediction
            x_hat_prior = x_hat
            p_prior = p + Q

            # Update
            z = values[i]
            if np.isnan(z):
                # Missing observation -- skip update
                filtered[i] = x_hat_prior
                x_hat = x_hat_prior
                p = p_prior
                continue

            K = p_prior / (p_prior + R)  # Kalman gain
            x_hat = x_hat_prior + K * (z - x_hat_prior)
            p = (1 - K) * p_prior

            filtered[i] = x_hat

        return pd.Series(filtered, index=signal.index, name=signal.name)

    # ------------------------------------------------------------------
    # Regime-detection helper
    # ------------------------------------------------------------------

    @staticmethod
    def detect_regime(
        prices: pd.DataFrame,
        vol_lookback: int = 60,
        trend_lookback: int = 120,
        vol_threshold: float = 1.5,
    ) -> pd.Series:
        """Simple rule-based regime classifier.

        Classification rules (applied per row / date):
        1. Compute realised volatility over ``vol_lookback`` days and its
           long-run median.  If current vol > ``vol_threshold`` * median
           -> ``HIGH_VOL``.
        2. Else, fit a linear slope to log-prices over ``trend_lookback``
           days.  Positive slope -> ``BULL``; negative -> ``BEAR``.
        3. If slope is near zero (|slope| < 1 std of the slope series)
           -> ``SIDEWAYS``.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (typically adjusted close).  If multiple columns,
            the first column is used.
        vol_lookback : int
            Window for realised-vol calculation.
        trend_lookback : int
            Window for trend (slope) calculation.
        vol_threshold : float
            Multiplier over median vol to flag HIGH_VOL.

        Returns
        -------
        pd.Series
            ``Regime`` enum values indexed by date.
        """
        if prices.ndim == 2:
            series = prices.iloc[:, 0]
        else:
            series = prices

        log_ret = np.log(series / series.shift(1)).dropna()
        realised_vol = log_ret.rolling(vol_lookback).std()
        median_vol = realised_vol.expanding(min_periods=vol_lookback).median()

        # Trend: rolling OLS slope on log-prices
        log_prices = np.log(series)

        def _rolling_slope(window: pd.Series) -> float:
            y = window.values
            x = np.arange(len(y), dtype=float)
            if np.any(np.isnan(y)):
                return np.nan
            x_dm = x - x.mean()
            slope = (x_dm @ (y - y.mean())) / (x_dm @ x_dm)
            return slope

        slope = log_prices.rolling(trend_lookback).apply(_rolling_slope, raw=False)
        slope_std = slope.expanding(min_periods=trend_lookback).std()

        regimes = pd.Series(index=series.index, dtype=object)

        for dt in regimes.index:
            rv = realised_vol.get(dt)
            mv = median_vol.get(dt)
            s = slope.get(dt)
            ss = slope_std.get(dt)

            if rv is None or np.isnan(rv) or mv is None or np.isnan(mv):
                regimes[dt] = Regime.SIDEWAYS
                continue

            if rv > vol_threshold * mv:
                regimes[dt] = Regime.HIGH_VOL
                continue

            if s is None or np.isnan(s) or ss is None or np.isnan(ss) or ss == 0:
                regimes[dt] = Regime.SIDEWAYS
                continue

            if abs(s) < ss:
                regimes[dt] = Regime.SIDEWAYS
            elif s > 0:
                regimes[dt] = Regime.BULL
            else:
                regimes[dt] = Regime.BEAR

        return regimes

    # ------------------------------------------------------------------
    # Risk-management helpers
    # ------------------------------------------------------------------

    def _apply_position_limits(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Clip individual weights and enforce portfolio-level leverage."""
        rl = self.risk_limits

        # Per-asset cap
        positions = positions.clip(
            lower=-rl.max_position_weight,
            upper=rl.max_position_weight,
        )

        # Portfolio-level leverage cap: scale all positions proportionally
        gross = positions.abs().sum(axis=1)
        scale = (rl.max_portfolio_leverage / gross).clip(upper=1.0)
        positions = positions.mul(scale, axis=0)

        return positions

    def apply_stop_loss(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
        entry_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Zero out positions that have breached the stop-loss threshold.

        Parameters
        ----------
        positions : pd.DataFrame
            Current position weights (same shape as prices).
        prices : pd.DataFrame
            Current price levels.
        entry_prices : pd.DataFrame
            Price at which each position was entered.  NaN where flat.

        Returns
        -------
        pd.DataFrame
            Adjusted positions with stopped-out entries set to 0.
        """
        if self.risk_limits.stop_loss_pct is None:
            return positions

        pct_change = (prices - entry_prices) / entry_prices
        # For long positions a large negative move triggers stop; for short
        # positions a large positive move triggers stop.
        long_stop = (positions > 0) & (pct_change < -self.risk_limits.stop_loss_pct)
        short_stop = (positions < 0) & (pct_change > self.risk_limits.stop_loss_pct)
        stopped = long_stop | short_stop

        if stopped.any().any():
            n_stopped = stopped.sum().sum()
            logger.info("Stop-loss triggered for %d position(s).", n_stopped)

        positions = positions.where(~stopped, 0.0)
        return positions

    def apply_take_profit(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
        entry_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Zero out positions that have reached the take-profit threshold.

        Parameters mirror :meth:`apply_stop_loss`.
        """
        if self.risk_limits.take_profit_pct is None:
            return positions

        pct_change = (prices - entry_prices) / entry_prices
        long_tp = (positions > 0) & (pct_change > self.risk_limits.take_profit_pct)
        short_tp = (positions < 0) & (pct_change < -self.risk_limits.take_profit_pct)
        took_profit = long_tp | short_tp

        if took_profit.any().any():
            n_tp = took_profit.sum().sum()
            logger.info("Take-profit triggered for %d position(s).", n_tp)

        positions = positions.where(~took_profit, 0.0)
        return positions

    def check_drawdown_breaker(
        self,
        equity_curve: pd.Series,
    ) -> pd.Series:
        """Return a boolean mask that is ``True`` on dates where the
        strategy should be flat because the drawdown limit was breached.

        Parameters
        ----------
        equity_curve : pd.Series
            Cumulative equity indexed by date.

        Returns
        -------
        pd.Series[bool]
            ``True`` where max-drawdown limit is exceeded.
        """
        if self.risk_limits.max_drawdown_pct is None:
            return pd.Series(False, index=equity_curve.index)

        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        breached = drawdown < -self.risk_limits.max_drawdown_pct
        return breached

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_prices(prices: pd.DataFrame) -> None:
        """Raise if the price DataFrame is clearly invalid."""
        if prices.empty:
            raise ValueError("prices DataFrame is empty.")
        if not isinstance(prices.index, pd.DatetimeIndex):
            warnings.warn(
                "prices index is not a DatetimeIndex; some helpers may "
                "behave unexpectedly.",
                stacklevel=2,
            )
        pct_nan = prices.isna().mean().mean()
        if pct_nan > 0.5:
            raise ValueError(
                f"prices contain {pct_nan:.0%} NaN values -- too many to "
                "be usable."
            )

    def ensure_fitted(self) -> None:
        """Raise if ``fit`` has not been called."""
        if not self._fitted:
            raise RuntimeError(
                f"Strategy '{self.name}' has not been fitted. "
                "Call .fit() before generating signals."
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        return f"Strategy({self.name!r}, {fitted_tag})"

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
