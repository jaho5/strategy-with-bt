"""Trading strategy based on Levy process jump detection and tail risk.

Mathematical foundation
-----------------------
Stock returns are NOT Gaussian -- they follow a Levy process with jumps.

Merton (1976) jump-diffusion model:

    dS/S = mu dt + sigma dW + J dN

where N is a Poisson process with intensity lambda, J ~ N(mu_J, sigma_J^2).

Lee-Mykland (2008) jump test statistic:

    L_t = |r_t| / sigma_hat_t   (BV-based)

compared against the extreme value distribution (Gumbel) for significance.

Strategy signals
----------------
1. **Jump detection** via bipower variation (BV) and realized variation (RV).
   Jump variation JV_t = max(RV_t - BV_t, 0) isolates the jump component.
   Individual returns are tested against a Gumbel-calibrated threshold.

2. **Jump-adjusted momentum**: decompose returns into continuous and jump
   components.  Continuous momentum (rolling sum of non-jump returns over
   252 days) is more persistent; jump returns are mean-reverting.

3. **Tail risk premium**: stocks with higher jump intensity lambda should
   command higher expected returns.  Go long high-lambda names.

4. **Composite signal**: 60% continuous momentum + 40% jump mean-reversion.
   After a detected negative jump, wait 3 days then go long (mean-reversion).
   After a detected positive jump, wait 3 days then go short.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.strategies.base import Strategy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LevyJumpConfig:
    """Tuneable parameters for the Levy jump detection strategy."""

    # Bipower / realized variation windows
    bv_window: int = 20              # rolling window for bipower variation
    rv_window: int = 20              # rolling window for realized variation

    # Jump test (Lee-Mykland / Gumbel threshold)
    jump_significance: float = 0.01  # p-value threshold for jump detection
    jump_test_window: int = 252      # calibration window for Gumbel constants

    # Momentum
    continuous_momentum_window: int = 252  # lookback for continuous momentum
    jump_momentum_window: int = 63         # lookback for jump momentum (unused in signal)

    # Jump mean-reversion
    jump_reversion_delay: int = 3    # wait N days after jump before entering
    jump_reversion_decay: int = 10   # signal decays over this many days

    # Jump intensity (tail risk premium)
    jump_intensity_window: int = 252  # lookback for estimating lambda

    # Signal blending
    continuous_momentum_weight: float = 0.60
    jump_reversion_weight: float = 0.40

    # Smoothing
    signal_ema_span: int = 5         # EMA span for final signal smoothing


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _bipower_variation(returns: pd.Series, window: int) -> pd.Series:
    """Rolling bipower variation (Barndorff-Nielsen & Shephard, 2004).

    BV_t = (pi/2) * (1/(n-1)) * sum_{i=2}^{n} |r_i| * |r_{i-1}|

    This estimates integrated variance without the jump component,
    providing a robust measure of continuous-path volatility.

    Parameters
    ----------
    returns : pd.Series
        Log-returns series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling bipower variation (daily variance scale).
    """
    abs_ret = returns.abs()
    # Product of consecutive absolute returns
    adjacent_product = abs_ret * abs_ret.shift(1)
    # Scale factor: pi/2 adjusts for E[|Z|] = sqrt(2/pi) when Z ~ N(0,1)
    scale = np.pi / 2.0
    bv = scale * adjacent_product.rolling(window=window, min_periods=window).mean()
    return bv


def _realized_variation(returns: pd.Series, window: int) -> pd.Series:
    """Rolling realized variation (sum of squared returns).

    RV_t = (1/n) * sum_{i=1}^{n} r_i^2

    Includes both continuous and jump components.

    Parameters
    ----------
    returns : pd.Series
        Log-returns series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling realized variation (daily variance scale).
    """
    return (returns ** 2).rolling(window=window, min_periods=window).mean()


def _gumbel_critical_value(n: int, significance: float) -> float:
    """Critical value from the Gumbel extreme-value distribution.

    For the Lee-Mykland (2008) jump test, the test statistic under H0
    (no jump) converges to a Gumbel distribution with location and scale
    parameters that depend on sample size n.

    Location: a_n = sqrt(2 * ln(n)) - (ln(pi) + ln(ln(n))) / (2 * sqrt(2 * ln(n)))
    Scale:    b_n = 1 / sqrt(2 * ln(n))

    Parameters
    ----------
    n : int
        Effective sample size (number of observations in calibration window).
    significance : float
        Desired significance level (e.g. 0.01 for 1%).

    Returns
    -------
    float
        Critical value C_n such that P(max|L_t| > C_n | H0) = significance.
    """
    if n < 2:
        return np.inf

    log_n = np.log(n)
    if log_n <= 0:
        return np.inf

    sqrt_2logn = np.sqrt(2.0 * log_n)

    # Centering constant
    a_n = sqrt_2logn - (np.log(np.pi) + np.log(log_n)) / (2.0 * sqrt_2logn)

    # Scaling constant
    b_n = 1.0 / sqrt_2logn

    # Gumbel quantile: P(X <= x) = exp(-exp(-(x - a_n)/b_n))
    # We want P(X > c) = significance => P(X <= c) = 1 - significance
    # => c = a_n - b_n * ln(-ln(1 - significance))
    c_n = a_n - b_n * np.log(-np.log(1.0 - significance))
    return c_n


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

class LevyJumpStrategy(Strategy):
    """Levy process jump detection and tail risk strategy.

    Combines jump-adjusted momentum (trend-following on the de-jumped
    return series) with jump mean-reversion (fading large detected jumps)
    to exploit the distinct dynamics of continuous and discontinuous
    return components.
    """

    def __init__(self, config: Optional[LevyJumpConfig] = None) -> None:
        super().__init__(
            name="LevyJump",
            description=(
                "Levy process jump detection with continuous momentum "
                "and jump mean-reversion"
            ),
        )
        self.cfg = config or LevyJumpConfig()

        # Populated during fit
        self._jump_intensity: Dict[str, float] = {}
        self._avg_jump_size: Dict[str, float] = {}
        self._continuous_vol: Dict[str, float] = {}

    # -----------------------------------------------------------------
    # Jump detection
    # -----------------------------------------------------------------

    def _detect_jumps(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """Detect jumps using the Lee-Mykland (2008) test.

        For each return r_t, the test statistic is:

            L_t = |r_t| / sqrt(BV_t / n)

        where BV_t is the bipower variation estimated over a local window.
        L_t is compared against a Gumbel-derived critical value.

        Parameters
        ----------
        returns : pd.Series
            Log-return series.

        Returns
        -------
        pd.Series
            Boolean series: True where a jump is detected.
        """
        window = self.cfg.bv_window
        bv = _bipower_variation(returns, window)

        # Instantaneous volatility estimate from BV (per-observation variance)
        # BV is already a mean, so the per-obs variance is BV itself
        sigma_hat = np.sqrt(bv.clip(lower=1e-20))

        # Test statistic: |r_t| / sigma_hat_t
        test_stat = returns.abs() / sigma_hat

        # Critical value from Gumbel distribution
        c_n = _gumbel_critical_value(
            self.cfg.jump_test_window,
            self.cfg.jump_significance,
        )

        is_jump = test_stat > c_n
        # Cannot detect jumps where BV is not yet available
        is_jump = is_jump.fillna(False)
        return is_jump

    def _decompose_returns(
        self,
        returns: pd.Series,
        is_jump: pd.Series,
    ) -> tuple:
        """Decompose returns into continuous and jump components.

        Parameters
        ----------
        returns : pd.Series
            Full log-return series.
        is_jump : pd.Series
            Boolean mask indicating detected jumps.

        Returns
        -------
        tuple of (pd.Series, pd.Series)
            (continuous_returns, jump_returns) -- each same index as input.
            Continuous returns are zero on jump days; jump returns are zero
            on non-jump days.
        """
        jump_returns = returns.where(is_jump, 0.0)
        continuous_returns = returns.where(~is_jump, 0.0)
        return continuous_returns, jump_returns

    # -----------------------------------------------------------------
    # Signal generators
    # -----------------------------------------------------------------

    def _continuous_momentum_signal(
        self,
        continuous_returns: pd.Series,
    ) -> pd.Series:
        """Trend-following signal on the de-jumped (continuous) return series.

        Computes the rolling sum of continuous returns over a lookback window,
        then normalises to [-1, +1] using a z-score approach.

        Parameters
        ----------
        continuous_returns : pd.Series
            Return series with jump components removed.

        Returns
        -------
        pd.Series
            Momentum signal in [-1, +1].
        """
        window = self.cfg.continuous_momentum_window

        # Cumulative continuous return over lookback
        cum_ret = continuous_returns.rolling(
            window=window, min_periods=window // 2,
        ).sum()

        # Z-score normalisation against expanding statistics
        cum_mean = cum_ret.expanding(min_periods=window // 2).mean()
        cum_std = cum_ret.expanding(min_periods=window // 2).std()
        cum_std = cum_std.replace(0, np.nan).ffill().fillna(1e-8)

        z = (cum_ret - cum_mean) / cum_std

        # Map to [-1, +1] via tanh-like clipping
        signal = z.clip(-3.0, 3.0) / 3.0
        return signal.fillna(0.0)

    def _jump_reversion_signal(
        self,
        returns: pd.Series,
        is_jump: pd.Series,
    ) -> pd.Series:
        """Mean-reversion signal triggered by detected jumps.

        After a negative jump: wait ``jump_reversion_delay`` days, then
        generate a +1 signal (expect bounce-back), decaying over
        ``jump_reversion_decay`` days.

        After a positive jump: wait ``jump_reversion_delay`` days, then
        generate a -1 signal (expect pull-back), decaying similarly.

        Parameters
        ----------
        returns : pd.Series
            Full return series (needed for jump sign).
        is_jump : pd.Series
            Boolean mask of detected jumps.

        Returns
        -------
        pd.Series
            Mean-reversion signal in [-1, +1].
        """
        delay = self.cfg.jump_reversion_delay
        decay = self.cfg.jump_reversion_decay

        signal = pd.Series(0.0, index=returns.index)
        idx = returns.index

        # Identify jump events with their signs
        jump_dates = idx[is_jump]
        jump_signs = returns.loc[jump_dates]  # positive or negative jump

        for jdate, jret in jump_signs.items():
            if np.isnan(jret) or jret == 0:
                continue

            # Direction to fade: opposite of jump
            fade_direction = -np.sign(jret)

            # Scale by magnitude (larger jumps get stronger reversion signal)
            # but cap at 1.0
            magnitude = min(abs(jret) / 0.03, 1.0)  # normalise by 3% move

            # Apply signal starting after delay, decaying linearly
            start_loc = idx.get_loc(jdate)
            for k in range(delay, delay + decay):
                target_loc = start_loc + k
                if target_loc >= len(idx):
                    break
                # Linear decay: full strength at delay, zero at delay + decay
                decay_factor = 1.0 - (k - delay) / decay
                contribution = fade_direction * magnitude * decay_factor
                signal.iloc[target_loc] += contribution

        # Clip aggregate signal to [-1, 1]
        signal = signal.clip(-1.0, 1.0)
        return signal

    def _jump_intensity_weight(
        self,
        is_jump: pd.Series,
    ) -> pd.Series:
        """Tail risk premium weight based on jump frequency.

        Higher jump intensity (lambda) implies higher tail risk and thus
        a higher expected risk premium.  This weight modulates position
        sizing: stocks with more frequent jumps get slightly larger
        allocations to capture the tail risk premium, subject to the
        risk management framework.

        Parameters
        ----------
        is_jump : pd.Series
            Boolean mask of detected jumps.

        Returns
        -------
        pd.Series
            Weight multiplier in [0.5, 1.5], where 1.0 is neutral.
        """
        window = self.cfg.jump_intensity_window

        # Rolling jump frequency (estimated lambda per day)
        jump_freq = is_jump.astype(float).rolling(
            window=window, min_periods=window // 2,
        ).mean()

        # Normalise: median frequency -> 1.0
        median_freq = jump_freq.expanding(min_periods=window // 2).median()
        median_freq = median_freq.replace(0, np.nan).ffill().fillna(1e-8)

        ratio = jump_freq / median_freq
        # Map to [0.5, 1.5]: higher lambda -> higher weight
        weight = ratio.clip(0.5, 1.5)
        return weight.fillna(1.0)

    def _relative_jump_intensity(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """Compute relative jump intensity JI_t = JV_t / RV_t.

        This ratio measures what fraction of total return variation
        is attributable to jumps.

        Parameters
        ----------
        returns : pd.Series
            Log-return series.

        Returns
        -------
        pd.Series
            Jump intensity ratio in [0, 1].
        """
        bv = _bipower_variation(returns, self.cfg.bv_window)
        rv = _realized_variation(returns, self.cfg.rv_window)

        # Jump variation: excess of RV over BV (floored at zero)
        jv = (rv - bv).clip(lower=0.0)

        # Relative jump intensity
        ji = jv / rv.replace(0, np.nan)
        return ji.fillna(0.0).clip(0.0, 1.0)

    # -----------------------------------------------------------------
    # Strategy interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "LevyJumpStrategy":
        """Calibrate jump parameters on historical price data.

        Estimates per-asset jump intensity (lambda), average jump size,
        and continuous volatility from the training sample.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data.  Columns are asset tickers; index is
            a DatetimeIndex.

        Returns
        -------
        self
        """
        self.validate_prices(prices)
        log_returns = np.log(prices / prices.shift(1)).dropna()

        for col in log_returns.columns:
            rets = log_returns[col].dropna()
            min_obs = max(self.cfg.bv_window * 2, 60)
            if len(rets) < min_obs:
                warnings.warn(
                    f"Insufficient data for {col}: {len(rets)} rows "
                    f"< minimum {min_obs}. Skipping.",
                    stacklevel=2,
                )
                continue

            # Detect jumps
            is_jump = self._detect_jumps(rets)
            continuous_rets, jump_rets = self._decompose_returns(rets, is_jump)

            # Estimate jump intensity (jumps per day)
            n_jumps = is_jump.sum()
            n_days = len(rets)
            self._jump_intensity[col] = n_jumps / n_days if n_days > 0 else 0.0

            # Average jump size (signed)
            jump_values = jump_rets[is_jump]
            if len(jump_values) > 0:
                self._avg_jump_size[col] = float(jump_values.mean())
            else:
                self._avg_jump_size[col] = 0.0

            # Continuous volatility (annualised)
            self._continuous_vol[col] = float(
                continuous_rets.std() * np.sqrt(252)
            )

        self.parameters = {
            "jump_intensity": dict(self._jump_intensity),
            "avg_jump_size": dict(self._avg_jump_size),
            "continuous_vol": dict(self._continuous_vol),
        }
        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals from price data.

        Signal construction:
        1. Detect jumps in each asset's return series.
        2. Decompose returns into continuous and jump components.
        3. Compute continuous momentum signal (trend-following on de-jumped
           series).
        4. Compute jump mean-reversion signal (fade detected jumps after
           a short delay).
        5. Blend: 60% continuous momentum + 40% jump mean-reversion.
        6. Modulate weight by tail risk premium (jump intensity).

        Parameters
        ----------
        prices : pd.DataFrame
            OHLCV or adjusted-close prices indexed by datetime with one
            column per ticker.

        Returns
        -------
        pd.DataFrame
            Contains ``{ticker}_signal`` and ``{ticker}_weight`` columns
            for each ticker, plus ``signal`` and ``weight`` if single-asset.
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        log_returns = np.log(prices / prices.shift(1))
        result = pd.DataFrame(index=prices.index)

        tickers = prices.columns.tolist()
        single_asset = len(tickers) == 1

        for col in tickers:
            rets = log_returns[col]

            # --- Jump detection ---
            is_jump = self._detect_jumps(rets)

            # --- Return decomposition ---
            continuous_rets, jump_rets = self._decompose_returns(rets, is_jump)

            # --- Component signals ---
            sig_continuous = self._continuous_momentum_signal(continuous_rets)
            sig_jump_revert = self._jump_reversion_signal(rets, is_jump)

            # --- Blend directional signal ---
            w_cont = self.cfg.continuous_momentum_weight
            w_jump = self.cfg.jump_reversion_weight
            total_w = w_cont + w_jump
            w_cont /= total_w
            w_jump /= total_w

            raw_signal = w_cont * sig_continuous + w_jump * sig_jump_revert

            # Smooth the composite signal
            smoothed_signal = self.exponential_smooth(
                raw_signal, span=self.cfg.signal_ema_span,
            )

            # Discretise to {-1, 0, +1} with a dead zone
            direction = pd.Series(0, index=prices.index, dtype=float)
            direction[smoothed_signal > 0.15] = 1.0
            direction[smoothed_signal < -0.15] = -1.0

            # --- Position weight from tail risk premium ---
            jump_weight = self._jump_intensity_weight(is_jump)

            # Base weight from signal magnitude (conviction)
            conviction = smoothed_signal.abs().clip(0.0, 1.0)
            weight = (conviction * jump_weight).clip(0.0, 1.0)

            # --- Store in result frame ---
            if single_asset:
                result["signal"] = direction
                result["weight"] = weight
            else:
                result[f"{col}_signal"] = direction
                result[f"{col}_weight"] = weight

        return result

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def jump_diagnostics(
        self, prices: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Return diagnostic DataFrames for each asset (for analysis/plotting).

        Includes: detected jumps, bipower variation, realized variation,
        relative jump intensity, continuous vs jump returns, and signals.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data.

        Returns
        -------
        dict
            Mapping from ticker to a DataFrame with diagnostic columns.
        """
        self.ensure_fitted()
        log_returns = np.log(prices / prices.shift(1))
        diagnostics: Dict[str, pd.DataFrame] = {}

        for col in prices.columns:
            rets = log_returns[col]
            is_jump = self._detect_jumps(rets)
            continuous_rets, jump_rets = self._decompose_returns(rets, is_jump)

            bv = _bipower_variation(rets, self.cfg.bv_window)
            rv = _realized_variation(rets, self.cfg.rv_window)
            jv = (rv - bv).clip(lower=0.0)
            ji = self._relative_jump_intensity(rets)

            sig_cont = self._continuous_momentum_signal(continuous_rets)
            sig_jump = self._jump_reversion_signal(rets, is_jump)

            diag = pd.DataFrame({
                "return": rets,
                "is_jump": is_jump,
                "continuous_return": continuous_rets,
                "jump_return": jump_rets,
                "bipower_variation": bv,
                "realized_variation": rv,
                "jump_variation": jv,
                "relative_jump_intensity": ji,
                "continuous_momentum_signal": sig_cont,
                "jump_reversion_signal": sig_jump,
                "jump_intensity_weight": self._jump_intensity_weight(is_jump),
            }, index=prices.index)

            diagnostics[col] = diag

        return diagnostics
