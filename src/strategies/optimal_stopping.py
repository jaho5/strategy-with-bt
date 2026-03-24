"""
Optimal Stopping Strategy
=========================

Mathematical foundation
-----------------------
The optimal stopping problem seeks a stopping time tau that maximises
E[g(X_tau)] subject to the constraint that tau is adapted to the natural
filtration of the price process.

The key theoretical objects are:

1. **Snell envelope** (backward induction / dynamic programming):

       V_t = max(g(X_t), E[V_{t+1} | F_t])

   The optimal stopping rule is:

       tau* = inf{ t : V_t = g(X_t) }

   i.e., stop the first time the continuation value equals the exercise
   value.

2. **Secretary problem / 1/e rule** (for entry timing):

   Given n candidates observed sequentially with no recall, the strategy
   that maximises the probability of selecting the best candidate is:
   - Observe the first r = floor(n / e) candidates without selecting
     (the *exploration* phase).
   - Select the first subsequent candidate that exceeds the best seen
     so far (the *exploitation* phase).
   This achieves an asymptotic success probability of 1/e ~ 0.3679.

   For trading: over a rolling lookback window of n bars, the strategy
   identifies local extrema (price troughs for long entry) using this
   framework.

3. **CUSUM (Cumulative Sum) stopping rule** (for exit timing):

       S_t = max(0, S_{t-1} + (X_t - k))
       Stop when S_t > h

   where:
   - k = reference value (typically half the expected shift magnitude)
   - h = decision threshold (controls the average run length / false
     alarm rate)
   - X_t = observed return or negative return depending on direction

   The CUSUM detector is optimal (in the Lorden minimax sense) for
   detecting a shift in the mean of a sequence of independent
   observations.

4. **Shiryaev-Roberts statistic** (for regime change detection):

       R_t = (R_{t-1} + 1) * (f_1(X_t) / f_0(X_t))

   where f_0 is the density under the null (no change) and f_1 is the
   density under the alternative (changed regime).  The procedure stops
   when R_t exceeds a threshold A.

   The Shiryaev-Roberts procedure is asymptotically optimal for the
   integral criterion of Pollak (1985) and the multi-cyclic criterion
   of Shiryaev (1963).

Strategy logic
--------------
1. **Entry**: use the secretary problem / 1/e rule on a rolling window
   to identify locally optimal entry points (price troughs).

2. **Exit**: run a CUSUM detector on adverse returns.  When cumulative
   adverse drift exceeds a threshold, exit the position.

3. **Confirmation**: the Shiryaev-Roberts statistic provides a second
   exit signal based on likelihood ratio accumulation.

4. **Position sizing**: weight by the Snell envelope ratio
   (exercise value / continuation value), capped at 1.

References
----------
- Ferguson, T. S. (1989). "Who solved the secretary problem?"
  Statistical Science, 4(3), 282-289.
- Page, E. S. (1954). "Continuous Inspection Schemes." Biometrika,
  41(1/2), 100-115.
- Shiryaev, A. N. (1963). "On optimum methods in quickest detection
  problems." Theory of Probability & Its Applications, 8(1), 22-46.
- Roberts, S. W. (1966). "A Comparison of Some Control Chart
  Procedures." Technometrics, 8(3), 411-430.
- Chow, Y. S., Robbins, H., & Siegmund, D. (1971). "Great Expectations:
  The Theory of Optimal Stopping." Houghton Mifflin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OptimalStoppingConfig:
    """Tuneable hyper-parameters for the optimal stopping strategy."""

    # --- Secretary problem (entry) ---
    # Lookback window (n) for the rolling secretary problem.
    secretary_window: int = 40
    # Fraction of the window to use as the exploration phase.
    # Theory prescribes 1/e ~ 0.3679; can be adjusted.
    exploration_fraction: float = 1.0 / np.e

    # --- CUSUM detector (exit) ---
    # Reference value k: half the expected adverse shift in returns.
    # Typically calibrated from data in fit().
    cusum_k: float = 0.005
    # Decision threshold h: higher -> fewer false exits, slower detection.
    cusum_h: float = 0.08

    # --- Shiryaev-Roberts detector (exit confirmation) ---
    # Threshold A for the Shiryaev-Roberts statistic.
    sr_threshold: float = 50.0
    # Standard deviation ratio (sigma_1 / sigma_0) for the alternative
    # regime.  >1 means we detect transitions to higher-vol regimes.
    sr_vol_ratio: float = 1.5
    # Mean shift under the alternative hypothesis (in units of sigma_0).
    sr_mean_shift: float = -1.0  # negative = adverse for long positions

    # --- Cooldown ---
    # Number of bars to wait after exiting before re-entering.
    cooldown_bars: int = 10

    # --- Position sizing ---
    # Maximum signal weight.
    max_signal_weight: float = 1.0
    # Snell envelope lookback for continuation value estimation.
    snell_lookback: int = 20

    # --- Return computation ---
    use_log_returns: bool = True

    # --- Entry direction ---
    # If True, look for local minima (troughs) to enter long.
    # If False, look for local maxima (peaks) to enter short.
    entry_long: bool = True


# ---------------------------------------------------------------------------
# Secretary Problem Engine (Entry Timing)
# ---------------------------------------------------------------------------


class SecretaryProblemEngine:
    """Rolling secretary problem for identifying optimal entry points.

    Over a window of n observations, the 1/e rule is applied:
    - The first r = floor(n * exploration_fraction) observations are the
      exploration phase (observe only, do not act).
    - In the exploitation phase, signal entry the first time the price
      falls below the minimum (for long) or rises above the maximum
      (for short) of the exploration phase.

    The engine operates on a rolling basis: at each time step, the most
    recent n observations form the window.
    """

    def __init__(self, config: OptimalStoppingConfig) -> None:
        self.config = config
        self.n = config.secretary_window
        self.r = max(1, int(self.n * config.exploration_fraction))
        self.entry_long = config.entry_long

    def evaluate(self, prices: np.ndarray, idx: int) -> bool:
        """Evaluate whether the current bar is an optimal entry point.

        Parameters
        ----------
        prices : np.ndarray
            Full price array up to the current bar.
        idx : int
            Current index into prices.

        Returns
        -------
        bool
            True if the current bar triggers an entry signal.
        """
        if idx < self.n:
            return False

        window = prices[idx - self.n + 1: idx + 1]

        # Exploration phase: first r observations.
        exploration = window[:self.r]

        # Exploitation phase: remaining observations.
        exploitation = window[self.r:]

        if len(exploitation) == 0:
            return False

        if self.entry_long:
            # For long entry: we seek the minimum price (best buy).
            # Exploration benchmark: minimum price in exploration phase.
            benchmark = np.min(exploration)
            # In exploitation phase, take the first price that goes below
            # the exploration benchmark.  At bar idx, we check if the
            # current price (last element) is the first such price.
            for i, p in enumerate(exploitation):
                if p < benchmark:
                    # This is the first price below the benchmark.
                    # Signal entry only if this corresponds to the
                    # current bar (last element of exploitation).
                    return i == len(exploitation) - 1
            # No price beat the benchmark -> no entry in this window.
            return False
        else:
            # For short entry: we seek the maximum price (best sell).
            benchmark = np.max(exploration)
            for i, p in enumerate(exploitation):
                if p > benchmark:
                    return i == len(exploitation) - 1
            return False


# ---------------------------------------------------------------------------
# CUSUM Detector (Exit Timing)
# ---------------------------------------------------------------------------


class CUSUMDetector:
    """Cumulative Sum (CUSUM) change-point detector.

    Tracks cumulative adverse drift in returns and signals an exit when
    the CUSUM statistic exceeds a threshold.

    For a long position, adverse drift means negative returns:
        S_t^+ = max(0, S_{t-1}^+ + (-r_t - k))

    For a short position, adverse drift means positive returns:
        S_t^- = max(0, S_{t-1}^- + (r_t - k))

    where r_t is the return at time t, and k is the reference value.
    """

    def __init__(self, k: float, h: float) -> None:
        self.k = k
        self.h = h
        self.reset()

    def reset(self) -> None:
        """Reset the CUSUM statistic to zero."""
        self.S_pos: float = 0.0  # detects upward shift (adverse for short)
        self.S_neg: float = 0.0  # detects downward shift (adverse for long)

    def update(self, ret: float) -> tuple[bool, bool]:
        """Process a new return observation.

        Parameters
        ----------
        ret : float
            The return at the current time step.

        Returns
        -------
        tuple[bool, bool]
            (adverse_for_long, adverse_for_short):
            - adverse_for_long is True if cumulative negative drift
              exceeds threshold (exit long).
            - adverse_for_short is True if cumulative positive drift
              exceeds threshold (exit short).
        """
        # Detect downward shift (adverse for long positions).
        self.S_neg = max(0.0, self.S_neg + (-ret - self.k))
        # Detect upward shift (adverse for short positions).
        self.S_pos = max(0.0, self.S_pos + (ret - self.k))

        exit_long = self.S_neg > self.h
        exit_short = self.S_pos > self.h

        return exit_long, exit_short

    @property
    def statistic_long(self) -> float:
        """Current CUSUM statistic for long-adverse detection."""
        return self.S_neg

    @property
    def statistic_short(self) -> float:
        """Current CUSUM statistic for short-adverse detection."""
        return self.S_pos


# ---------------------------------------------------------------------------
# Shiryaev-Roberts Detector (Regime Change Confirmation)
# ---------------------------------------------------------------------------


class ShiryaevRobertsDetector:
    """Shiryaev-Roberts procedure for regime change detection.

    Computes the statistic:
        R_t = (R_{t-1} + 1) * (f_1(X_t) / f_0(X_t))

    where f_0 ~ N(mu_0, sigma_0^2) and f_1 ~ N(mu_1, sigma_1^2).

    The procedure declares a change when R_t > A.
    """

    def __init__(
        self,
        mu_0: float,
        sigma_0: float,
        mu_1: float,
        sigma_1: float,
        threshold: float,
    ) -> None:
        self.mu_0 = mu_0
        self.sigma_0 = max(sigma_0, 1e-10)
        self.mu_1 = mu_1
        self.sigma_1 = max(sigma_1, 1e-10)
        self.threshold = threshold
        self.R: float = 0.0

    def reset(self) -> None:
        """Reset the Shiryaev-Roberts statistic."""
        self.R = 0.0

    def update(self, x: float) -> bool:
        """Process a new observation and return whether to stop.

        Parameters
        ----------
        x : float
            New observation (typically a return).

        Returns
        -------
        bool
            True if the regime change threshold is exceeded.
        """
        # Log-likelihood ratio: log(f_1(x)) - log(f_0(x)).
        ll_0 = _normal_log_pdf(x, self.mu_0, self.sigma_0)
        ll_1 = _normal_log_pdf(x, self.mu_1, self.sigma_1)
        log_lr = ll_1 - ll_0

        # Clamp the log-likelihood ratio to prevent numerical overflow.
        log_lr = np.clip(log_lr, -20.0, 20.0)
        lr = np.exp(log_lr)

        # Shiryaev-Roberts recursion.
        self.R = (self.R + 1.0) * lr

        return self.R > self.threshold

    @property
    def statistic(self) -> float:
        """Current Shiryaev-Roberts statistic."""
        return self.R


# ---------------------------------------------------------------------------
# Snell Envelope Estimator (Position Sizing)
# ---------------------------------------------------------------------------


def estimate_snell_ratio(
    returns: np.ndarray,
    idx: int,
    lookback: int,
) -> float:
    """Estimate the Snell envelope ratio for position sizing.

    The Snell envelope V_t = max(g(X_t), E[V_{t+1} | F_t]) relates the
    exercise value to the continuation value.  We approximate the ratio
    g(X_t) / V_t using a rolling estimate of the reward-to-go.

    For trading, we interpret:
    - g(X_t) = current cumulative return since entry (exercise value:
      take profit now).
    - E[V_{t+1} | F_t] ~ mu * (T - t) where mu is the recent mean
      return (continuation value: expected future profit).

    The ratio approaches 1 when it is optimal to stop (exercise) and
    is < 1 when continuation is preferred.

    Parameters
    ----------
    returns : np.ndarray
        Array of returns.
    idx : int
        Current index.
    lookback : int
        Number of past observations for estimating continuation value.

    Returns
    -------
    float
        A weight in [0, 1] indicating conviction.  Higher values
        indicate the position is well-supported by the Snell envelope
        (continuation value exceeds exercise value).
    """
    if idx < lookback:
        return 0.5

    window = returns[idx - lookback + 1: idx + 1]
    mu = np.mean(window)
    sigma = np.std(window)

    if sigma < 1e-10:
        return 0.5

    # Sharpe-like measure of continuation value.
    # If recent returns have positive mean, continuation is attractive.
    sharpe = mu / sigma * np.sqrt(lookback)

    # Map to [0, 1] via sigmoid.
    weight = 1.0 / (1.0 + np.exp(-sharpe))

    return float(np.clip(weight, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class OptimalStoppingStrategy(Strategy):
    """Optimal stopping theory trading strategy.

    Combines the secretary problem (entry timing), CUSUM detector (exit
    timing), and Shiryaev-Roberts statistic (regime change confirmation)
    into a unified trading framework grounded in stochastic control and
    dynamic programming.

    Parameters
    ----------
    config : OptimalStoppingConfig | None
        Strategy hyper-parameters.  Defaults are used when *None*.
    """

    def __init__(self, config: OptimalStoppingConfig | None = None) -> None:
        self.config = config or OptimalStoppingConfig()
        super().__init__(
            name="OptimalStopping",
            description=(
                "Optimal stopping theory strategy combining the secretary "
                "problem (1/e rule for entry), CUSUM detector (Page 1954 "
                "for exit), and Shiryaev-Roberts statistic (regime change "
                "confirmation).  Grounded in Snell envelope / dynamic "
                "programming for position sizing."
            ),
        )
        # Learned during fit.
        self._global_mean: float = 0.0
        self._global_std: float = 1e-4
        self._cusum_k: float = self.config.cusum_k
        self._cusum_h: float = self.config.cusum_h

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> "OptimalStoppingStrategy":
        """Calibrate strategy parameters from training data.

        Estimates global return statistics used to:
        - Set the CUSUM reference value k to half the mean absolute
          return (the expected shift magnitude under the null).
        - Set Shiryaev-Roberts densities f_0 and f_1 from the
          empirical return distribution.
        - Calibrate the Snell envelope estimator.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (OHLCV or adjusted close).

        Returns
        -------
        self
        """
        self.validate_prices(prices)
        series = prices.iloc[:, 0] if prices.ndim == 2 else prices

        returns = self._compute_returns(series).dropna()

        if len(returns) < 30:
            raise ValueError(
                f"Insufficient data for fitting: need >= 30 observations, "
                f"got {len(returns)}."
            )

        self._global_mean = float(np.mean(returns.values))
        self._global_std = float(np.std(returns.values))

        # Calibrate CUSUM reference value: half the mean absolute return
        # acts as the "expected shift size / 2" in the CUSUM framework.
        mean_abs_ret = float(np.mean(np.abs(returns.values)))
        self._cusum_k = mean_abs_ret * 0.5
        # Keep user-specified h unless it is clearly too small.
        self._cusum_h = self.config.cusum_h

        self.parameters = {
            "global_mean": self._global_mean,
            "global_std": self._global_std,
            "cusum_k": self._cusum_k,
            "cusum_h": self._cusum_h,
            "secretary_window": self.config.secretary_window,
            "exploration_fraction": self.config.exploration_fraction,
        }
        self._fitted = True

        logger.info(
            "OptimalStopping fitted: mean=%.6f, std=%.6f, "
            "cusum_k=%.6f, cusum_h=%.6f",
            self._global_mean,
            self._global_std,
            self._cusum_k,
            self._cusum_h,
        )
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals via optimal stopping theory.

        For each asset (column) in *prices*, the method:
        1. Runs the secretary problem engine for entry timing.
        2. Runs CUSUM and Shiryaev-Roberts detectors for exit timing.
        3. Sizes positions using the Snell envelope ratio.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data indexed by datetime.

        Returns
        -------
        pd.DataFrame
            Columns ``signal`` and ``weight`` (single asset) or
            ``{ticker}_signal`` and ``{ticker}_weight`` per asset.
        """
        self.ensure_fitted()

        tickers = prices.columns.tolist()
        single_asset = len(tickers) == 1

        result = pd.DataFrame(index=prices.index)

        for ticker in tickers:
            series = prices[ticker].dropna()
            if len(series) < self.config.secretary_window + 10:
                logger.warning(
                    "Skipping %s -- insufficient data (%d rows).",
                    ticker,
                    len(series),
                )
                sig = pd.Series(0.0, index=series.index)
                wgt = pd.Series(0.0, index=series.index)
            else:
                sig, wgt = self._generate_single_asset(series)

            if single_asset:
                result["signal"] = sig.reindex(prices.index, fill_value=0.0)
                result["weight"] = wgt.reindex(prices.index, fill_value=0.0)
            else:
                result[f"{ticker}_signal"] = sig.reindex(
                    prices.index, fill_value=0.0
                )
                result[f"{ticker}_weight"] = wgt.reindex(
                    prices.index, fill_value=0.0
                )

        return result

    # ------------------------------------------------------------------
    # Internal: single-asset signal generation
    # ------------------------------------------------------------------

    def _generate_single_asset(
        self, series: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Run optimal stopping logic on a single price series.

        State machine:
        - FLAT: waiting for secretary problem to signal entry.
        - LONG: holding a long position, monitoring CUSUM / SR for exit.
        - COOLDOWN: waiting cooldown_bars after exit before re-entering.

        Returns
        -------
        signal : pd.Series
            Directional signal in {-1, 0, +1}.
        weight : pd.Series
            Position-sizing weight in [0, 1].
        """
        cfg = self.config
        price_arr = series.values.astype(np.float64)
        returns_series = self._compute_returns(series).dropna()
        returns_arr = returns_series.values.astype(np.float64)
        T = len(returns_arr)

        # Align price array to returns (returns start from index 1 of
        # the original price series, so prices for returns are shifted).
        # price_for_return[t] corresponds to returns_arr[t].
        price_for_return = price_arr[1: 1 + T]

        signals = np.zeros(T, dtype=np.float64)
        weights = np.zeros(T, dtype=np.float64)

        # --- Instantiate engines ---
        secretary = SecretaryProblemEngine(cfg)

        cusum = CUSUMDetector(k=self._cusum_k, h=self._cusum_h)

        sr_detector = ShiryaevRobertsDetector(
            mu_0=self._global_mean,
            sigma_0=self._global_std,
            mu_1=self._global_mean + cfg.sr_mean_shift * self._global_std,
            sigma_1=self._global_std * cfg.sr_vol_ratio,
            threshold=cfg.sr_threshold,
        )

        # --- State machine ---
        # States: 0 = FLAT, 1 = LONG, 2 = COOLDOWN
        STATE_FLAT = 0
        STATE_LONG = 1
        STATE_COOLDOWN = 2

        state = STATE_FLAT
        cooldown_remaining = 0

        for t in range(T):
            ret = returns_arr[t]

            if not np.isfinite(ret):
                continue

            if state == STATE_FLAT:
                # --- Entry logic: secretary problem ---
                entry_signal = secretary.evaluate(price_for_return, t)

                if entry_signal:
                    state = STATE_LONG
                    cusum.reset()
                    sr_detector.reset()

                    # Snell envelope ratio for initial sizing.
                    snell_w = estimate_snell_ratio(
                        returns_arr, t, cfg.snell_lookback
                    )

                    if cfg.entry_long:
                        signals[t] = 1.0
                    else:
                        signals[t] = -1.0
                    weights[t] = min(
                        snell_w, cfg.max_signal_weight
                    )
                else:
                    signals[t] = 0.0
                    weights[t] = 0.0

            elif state == STATE_LONG:
                # --- Exit logic: CUSUM + Shiryaev-Roberts ---
                exit_long, exit_short = cusum.update(ret)
                sr_alarm = sr_detector.update(ret)

                # Determine if we should exit.
                should_exit = False
                if cfg.entry_long and exit_long:
                    should_exit = True
                elif not cfg.entry_long and exit_short:
                    should_exit = True

                # Shiryaev-Roberts provides confirmatory exit signal.
                if sr_alarm:
                    should_exit = True

                if should_exit:
                    # Exit position.
                    state = STATE_COOLDOWN
                    cooldown_remaining = cfg.cooldown_bars
                    signals[t] = 0.0
                    weights[t] = 0.0
                else:
                    # Maintain position with Snell envelope sizing.
                    snell_w = estimate_snell_ratio(
                        returns_arr, t, cfg.snell_lookback
                    )

                    # Scale weight by CUSUM headroom (how far from
                    # threshold we are).  This implements gradual
                    # de-risking as the CUSUM approaches the threshold.
                    if cfg.entry_long:
                        cusum_ratio = cusum.statistic_long / self._cusum_h
                    else:
                        cusum_ratio = cusum.statistic_short / self._cusum_h

                    cusum_scale = max(0.0, 1.0 - cusum_ratio)

                    if cfg.entry_long:
                        signals[t] = 1.0
                    else:
                        signals[t] = -1.0

                    raw_weight = snell_w * cusum_scale
                    weights[t] = min(
                        raw_weight, cfg.max_signal_weight
                    )

            elif state == STATE_COOLDOWN:
                cooldown_remaining -= 1
                signals[t] = 0.0
                weights[t] = 0.0

                if cooldown_remaining <= 0:
                    state = STATE_FLAT

        signal_series = pd.Series(
            signals, index=returns_series.index, name=series.name
        )
        weight_series = pd.Series(
            weights, index=returns_series.index, name=series.name
        )

        return signal_series, weight_series

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_returns(self, series: pd.Series) -> pd.Series:
        """Compute period returns from a price series."""
        if self.config.use_log_returns:
            return np.log(series / series.shift(1))
        else:
            return series.pct_change()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def run_diagnostics(
        self, series: pd.Series
    ) -> pd.DataFrame:
        """Run all detectors and return a diagnostic DataFrame.

        Useful for analysis and plotting.  The returned DataFrame
        contains:
        - ``return``: computed return series.
        - ``secretary_entry``: True at bars where the secretary problem
          triggers an entry signal.
        - ``cusum_neg``: CUSUM statistic tracking adverse downward drift.
        - ``cusum_pos``: CUSUM statistic tracking adverse upward drift.
        - ``sr_statistic``: Shiryaev-Roberts statistic.
        - ``snell_ratio``: estimated Snell envelope ratio.
        - ``state``: position state (0=flat, 1=long, 2=cooldown).

        Parameters
        ----------
        series : pd.Series
            Price series (not returns).

        Returns
        -------
        pd.DataFrame
            Diagnostics indexed by the return dates.
        """
        self.ensure_fitted()
        cfg = self.config

        price_arr = series.values.astype(np.float64)
        returns_series = self._compute_returns(series).dropna()
        returns_arr = returns_series.values.astype(np.float64)
        T = len(returns_arr)
        price_for_return = price_arr[1: 1 + T]

        secretary = SecretaryProblemEngine(cfg)
        cusum = CUSUMDetector(k=self._cusum_k, h=self._cusum_h)
        sr_detector = ShiryaevRobertsDetector(
            mu_0=self._global_mean,
            sigma_0=self._global_std,
            mu_1=self._global_mean + cfg.sr_mean_shift * self._global_std,
            sigma_1=self._global_std * cfg.sr_vol_ratio,
            threshold=cfg.sr_threshold,
        )

        STATE_FLAT = 0
        STATE_LONG = 1
        STATE_COOLDOWN = 2
        state = STATE_FLAT
        cooldown_remaining = 0

        records = []

        for t in range(T):
            ret = returns_arr[t]
            record: dict[str, Any] = {"return": ret}

            if not np.isfinite(ret):
                record.update({
                    "secretary_entry": False,
                    "cusum_neg": np.nan,
                    "cusum_pos": np.nan,
                    "sr_statistic": np.nan,
                    "snell_ratio": np.nan,
                    "state": state,
                })
                records.append(record)
                continue

            entry_signal = secretary.evaluate(price_for_return, t)
            record["secretary_entry"] = entry_signal

            if state == STATE_FLAT:
                if entry_signal:
                    state = STATE_LONG
                    cusum.reset()
                    sr_detector.reset()
            elif state == STATE_LONG:
                exit_long, exit_short = cusum.update(ret)
                sr_alarm = sr_detector.update(ret)
                should_exit = False
                if cfg.entry_long and exit_long:
                    should_exit = True
                elif not cfg.entry_long and exit_short:
                    should_exit = True
                if sr_alarm:
                    should_exit = True
                if should_exit:
                    state = STATE_COOLDOWN
                    cooldown_remaining = cfg.cooldown_bars
            elif state == STATE_COOLDOWN:
                cooldown_remaining -= 1
                if cooldown_remaining <= 0:
                    state = STATE_FLAT

            record["cusum_neg"] = cusum.statistic_long
            record["cusum_pos"] = cusum.statistic_short
            record["sr_statistic"] = sr_detector.statistic
            record["snell_ratio"] = estimate_snell_ratio(
                returns_arr, t, cfg.snell_lookback
            )
            record["state"] = state

            records.append(record)

        return pd.DataFrame(records, index=returns_series.index)


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------


def _normal_log_pdf(x: float, mu: float, sigma: float) -> float:
    """Log probability density of the univariate normal distribution.

    Parameters
    ----------
    x : float
        Observation.
    mu : float
        Mean.
    sigma : float
        Standard deviation (must be > 0).

    Returns
    -------
    float
        log N(x; mu, sigma^2).
    """
    z = (x - mu) / sigma
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * z * z
