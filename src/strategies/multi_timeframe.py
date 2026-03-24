"""
Multi-Timeframe Momentum + Mean-Reversion Strategy
===================================================

Exploits the heterogeneous serial dependence of asset returns across time
horizons by combining three orthogonal signals:

    Signal_t = alpha_SR * MR(5d) + alpha_MOM * Mom(252d) + alpha_LTR * Rev(756d)

Mathematical foundation
-----------------------
Returns exhibit horizon-dependent autocorrelation structure:

1. **Short-term mean reversion** (1-5 days): negative first-order
   autocorrelation arising from market microstructure effects (bid-ask
   bounce, inventory management, short-term overreaction).  Modelled as a
   z-score of the 5-day return relative to a 20-day rolling distribution:

       z_t = (r_{t-5:t} - mu_{20}) / sigma_{20}

   Signal: fade extremes.  z > 2 => short;  z < -2 => long.

2. **Medium-term momentum** (1-12 months): positive autocorrelation
   documented extensively since Jegadeesh & Titman (1993).  The classic
   "12-1" momentum signal skips the most recent month to avoid the
   well-known short-term reversal contamination:

       Mom_t = P_t / P_{t-252} - 1   (skip last 21 days)

   Cross-sectional rank: long top 30 %, short bottom 30 %.

3. **Long-term reversal** (3-5 years): negative autocorrelation from
   overreaction, consistent with DeBondt & Thaler (1985):

       Rev_t = P_t / P_{t-756} - 1

   Fade the extreme winners/losers.

Signal combination rules:
    - When MR and MOM agree in sign: 2x conviction (double position size).
    - When they disagree: 0.5x conviction (half position size).

Position sizing:
    - Kelly-adjusted per signal: f* = Sharpe / sigma.
    - Per-asset cap at 30 %.
    - Volatility targeting at 25 % annualised on the combined portfolio.
    - Rebalancing enforced on weekly boundaries.

References
----------
- Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and
  Selling Losers". Journal of Finance, 48(1), 65-91.
- DeBondt, W. F. M. & Thaler, R. H. (1985). "Does the Stock Market
  Overreact?". Journal of Finance, 40(3), 793-805.
- Lo, A. W. & MacKinlay, A. C. (1990). "When Are Contrarian Profits Due
  to Stock Market Overreaction?". Review of Financial Studies, 3(2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252
ANNUALISATION_FACTOR = np.sqrt(TRADING_DAYS_PER_YEAR)


@dataclass
class MultiTimeframeConfig:
    """Hyper-parameters for the multi-timeframe strategy."""

    # -- Short-term mean reversion --
    mr_return_window: int = 5       # days for short-term return
    mr_rolling_window: int = 20     # days for rolling mean/std of short returns
    mr_entry_z: float = 2.0         # |z| threshold to trigger mean-reversion signal

    # -- Medium-term momentum --
    mom_lookback: int = 252          # 12-month return lookback
    mom_skip: int = 21               # skip most recent month (reversal avoidance)
    mom_long_quantile: float = 0.70  # top 30% go long
    mom_short_quantile: float = 0.30 # bottom 30% go short

    # -- Long-term reversal --
    ltr_lookback: int = 756          # ~3-year return lookback

    # -- Signal combination --
    alpha_sr: float = 0.40           # weight on short-term mean reversion
    alpha_mom: float = 0.40          # weight on medium-term momentum
    alpha_ltr: float = 0.20          # weight on long-term reversal

    # -- Position sizing / risk --
    vol_target: float = 0.25         # annualised volatility target
    vol_lookback: int = 60           # days for realised vol estimation
    kelly_fraction: float = 0.5      # half-Kelly
    max_weight_per_asset: float = 0.30  # 30% cap per asset
    rebalance_freq: int = 5          # rebalance every 5 trading days (weekly)

    # -- Conviction scaling --
    agree_multiplier: float = 2.0    # when MR and MOM signals agree
    disagree_multiplier: float = 0.5 # when they disagree


# ---------------------------------------------------------------------------
# Signal computation helpers
# ---------------------------------------------------------------------------

def _short_term_mean_reversion(
    prices: pd.DataFrame,
    return_window: int = 5,
    rolling_window: int = 20,
    entry_z: float = 2.0,
) -> pd.DataFrame:
    """Compute short-term mean-reversion z-scores for each asset.

    The signal is the z-score of the *return_window*-day return relative
    to a *rolling_window*-day expanding distribution of such returns.
    Extreme z-scores are faded: z > entry_z => -1 (short), z < -entry_z
    => +1 (long), otherwise linear interpolation in [-1, 1].

    Returns
    -------
    pd.DataFrame
        Continuous signal in [-1, 1], same shape as *prices*.
    """
    # Compute rolling short-term returns
    short_ret = prices.pct_change(return_window)

    # Rolling mean and std of the short-term return series
    rolling_mu = short_ret.rolling(window=rolling_window, min_periods=rolling_window).mean()
    rolling_sigma = short_ret.rolling(window=rolling_window, min_periods=rolling_window).std()

    # Guard against zero/tiny std
    rolling_sigma = rolling_sigma.clip(lower=1e-10)

    z_scores = (short_ret - rolling_mu) / rolling_sigma

    # Continuous signal: fade extremes, linear in the middle
    # z > entry_z  =>  signal = -1  (short the overreaction)
    # z < -entry_z =>  signal = +1  (buy the dip)
    # |z| < entry_z => linearly interpolate
    signal = -z_scores / entry_z
    signal = signal.clip(-1.0, 1.0)

    return signal


def _medium_term_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    long_quantile: float = 0.70,
    short_quantile: float = 0.30,
) -> pd.DataFrame:
    """Compute cross-sectional 12-1 momentum signal.

    The "12-1" formation skips the most recent *skip* days to avoid
    contamination from short-term reversal (Jegadeesh & Titman, 1993).

    Cross-sectional ranking:
        top quantile  => +1 (long)
        bottom quantile => -1 (short)
        middle         => 0 (flat)

    For a single-asset universe the ranking degenerates, so the signal
    falls back to the sign of raw momentum.

    Returns
    -------
    pd.DataFrame
        Signal in {-1, 0, +1}, same shape as *prices*.
    """
    n_assets = prices.shape[1]

    # Lagged price: price *skip* days ago
    prices_lagged = prices.shift(skip)
    # Price *lookback* days ago
    prices_formation = prices.shift(lookback)

    # 12-1 momentum return (skip the most recent month)
    mom_ret = (prices_lagged / prices_formation) - 1.0

    if n_assets < 3:
        # Not enough assets for meaningful cross-sectional ranking;
        # use the time-series sign of momentum instead.
        signal = np.sign(mom_ret)
        return signal.fillna(0.0)

    # Cross-sectional percentile rank each row (date)
    ranked = mom_ret.rank(axis=1, pct=True)

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[ranked >= long_quantile] = 1.0
    signal[ranked <= short_quantile] = -1.0

    # Where momentum return is NaN, set signal to 0
    signal[mom_ret.isna()] = 0.0

    return signal


def _long_term_reversal(
    prices: pd.DataFrame,
    lookback: int = 756,
    long_quantile: float = 0.70,
    short_quantile: float = 0.30,
) -> pd.DataFrame:
    """Compute long-term reversal signal (DeBondt & Thaler, 1985).

    3-5 year past winners are faded (shorted), past losers are bought.
    This is the mirror image of the momentum signal: high past returns
    produce a *negative* signal.

    Returns
    -------
    pd.DataFrame
        Signal in {-1, 0, +1}, same shape as *prices*.
    """
    n_assets = prices.shape[1]
    ltr_ret = (prices / prices.shift(lookback)) - 1.0

    if n_assets < 3:
        # Fade the sign of long-term return (time-series reversal)
        signal = -np.sign(ltr_ret)
        return signal.fillna(0.0)

    # Cross-sectional rank
    ranked = ltr_ret.rank(axis=1, pct=True)

    # Reversal: short past winners, long past losers
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[ranked >= long_quantile] = -1.0   # past winners => short
    signal[ranked <= short_quantile] = 1.0   # past losers  => long
    signal[ltr_ret.isna()] = 0.0

    return signal


def _conviction_scaling(
    mr_signal: pd.DataFrame,
    mom_signal: pd.DataFrame,
    agree_multiplier: float = 2.0,
    disagree_multiplier: float = 0.5,
) -> pd.DataFrame:
    """Compute conviction multiplier based on agreement between MR and MOM.

    Agreement is measured by the sign of the product of the two signals:
        same sign   => high conviction (agree_multiplier)
        opposite    => low conviction  (disagree_multiplier)
        one is zero => neutral (1.0)

    Returns
    -------
    pd.DataFrame
        Multiplier >= 0, same shape as inputs.
    """
    product = mr_signal * mom_signal

    conviction = pd.DataFrame(1.0, index=mr_signal.index, columns=mr_signal.columns)
    conviction[product > 0] = agree_multiplier    # signals agree
    conviction[product < 0] = disagree_multiplier  # signals disagree
    # product == 0 => one signal is flat => keep multiplier at 1.0

    return conviction


def _kelly_weight_per_asset(
    returns: pd.DataFrame,
    lookback: int = 252,
    fraction: float = 0.5,
) -> pd.Series:
    """Compute per-asset Kelly fraction: f* = fraction * (mu / sigma^2).

    Returns absolute Kelly weight per asset (clipped to [0.05, 1.0]).
    """
    window = returns.iloc[-lookback:]
    mu = window.mean()
    var = window.var()
    var = var.replace(0, np.nan)

    kelly = fraction * mu.abs() / var
    kelly = kelly.fillna(0.0).clip(0.05, 1.0)
    return kelly


def _vol_target_scalar(
    portfolio_returns: pd.Series,
    target_vol: float = 0.25,
    lookback: int = 60,
) -> float:
    """Compute a scalar to rescale the portfolio to the target volatility.

    vol_target_scalar = target_vol / (realised_vol * sqrt(252))

    Returns a scalar >= 0, capped at 3.0 to avoid extreme leverage.
    """
    if len(portfolio_returns) < lookback:
        return 1.0

    realised_vol = portfolio_returns.iloc[-lookback:].std()
    if realised_vol < 1e-10:
        return 1.0

    annualised_vol = realised_vol * ANNUALISATION_FACTOR
    scalar = target_vol / annualised_vol
    return float(np.clip(scalar, 0.1, 3.0))


# ===========================================================================
# Strategy class
# ===========================================================================

class MultiTimeframeStrategy(Strategy):
    """Multi-timeframe momentum + mean-reversion strategy.

    Combines three time-scale signals -- short-term mean reversion (1-5d),
    medium-term momentum (1-12mo), and long-term reversal (3-5yr) -- into
    a single, volatility-targeted, Kelly-sized portfolio.

    Parameters
    ----------
    config : MultiTimeframeConfig, optional
        Strategy hyper-parameters.  Uses defaults if not provided.
    """

    def __init__(self, config: Optional[MultiTimeframeConfig] = None) -> None:
        super().__init__(
            name="MultiTimeframe MR+Mom",
            description=(
                "Multi-timeframe strategy combining short-term mean reversion, "
                "medium-term momentum, and long-term reversal with "
                "Kelly sizing and volatility targeting."
            ),
        )
        self.config = config or MultiTimeframeConfig()
        self.parameters: Dict[str, Any] = {}

        # Learned during fit()
        self._alpha_sr: float = self.config.alpha_sr
        self._alpha_mom: float = self.config.alpha_mom
        self._alpha_ltr: float = self.config.alpha_ltr
        self._kelly_weights: Optional[pd.Series] = None
        self._vol_scalar: float = 1.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "MultiTimeframeStrategy":
        """Calibrate signal weights and position sizing from historical data.

        Calibration steps:
        1. Compute each sub-signal over the training period.
        2. Estimate the in-sample Sharpe ratio of each signal to derive
           adaptive alpha weights (proportional to |Sharpe|).
        3. Compute Kelly fractions per asset from trailing returns.
        4. Estimate the vol-targeting scalar.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (DatetimeIndex x tickers).

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        returns = prices.pct_change().dropna(how="all")

        # ---- Compute sub-signals on training data ----
        mr_sig = _short_term_mean_reversion(
            prices,
            return_window=self.config.mr_return_window,
            rolling_window=self.config.mr_rolling_window,
            entry_z=self.config.mr_entry_z,
        )
        mom_sig = _medium_term_momentum(
            prices,
            lookback=self.config.mom_lookback,
            skip=self.config.mom_skip,
            long_quantile=self.config.mom_long_quantile,
            short_quantile=self.config.mom_short_quantile,
        )
        ltr_sig = _long_term_reversal(
            prices,
            lookback=self.config.ltr_lookback,
        )

        # ---- In-sample Sharpe of each signal ----
        # Signal return at time t: signal_{t-1} * asset_return_t
        # (we shift signal by 1 to avoid look-ahead)
        sharpe_map: Dict[str, float] = {}
        for name, sig in [("sr", mr_sig), ("mom", mom_sig), ("ltr", ltr_sig)]:
            sig_ret = (sig.shift(1) * returns).dropna(how="all")
            port_ret = sig_ret.mean(axis=1)  # equal-weight across assets
            if len(port_ret) < 60 or port_ret.std() < 1e-10:
                sharpe_map[name] = 0.0
            else:
                sharpe_map[name] = float(
                    port_ret.mean() / port_ret.std() * ANNUALISATION_FACTOR
                )

        logger.info(
            "In-sample Sharpe ratios -- SR: %.3f, MOM: %.3f, LTR: %.3f",
            sharpe_map["sr"], sharpe_map["mom"], sharpe_map["ltr"],
        )

        # ---- Adaptive alpha weights (proportional to |Sharpe|) ----
        abs_sharpe = {k: abs(v) for k, v in sharpe_map.items()}
        total_sharpe = sum(abs_sharpe.values())

        if total_sharpe > 1e-8:
            self._alpha_sr = abs_sharpe["sr"] / total_sharpe
            self._alpha_mom = abs_sharpe["mom"] / total_sharpe
            self._alpha_ltr = abs_sharpe["ltr"] / total_sharpe
        else:
            # Fallback to config defaults when all Sharpes are ~0
            self._alpha_sr = self.config.alpha_sr
            self._alpha_mom = self.config.alpha_mom
            self._alpha_ltr = self.config.alpha_ltr

        logger.info(
            "Calibrated alpha weights -- SR: %.3f, MOM: %.3f, LTR: %.3f",
            self._alpha_sr, self._alpha_mom, self._alpha_ltr,
        )

        # ---- Kelly weights per asset ----
        self._kelly_weights = _kelly_weight_per_asset(
            returns,
            lookback=min(self.config.mom_lookback, len(returns)),
            fraction=self.config.kelly_fraction,
        )

        # ---- Vol-targeting scalar from combined signal ----
        combined = (
            self._alpha_sr * mr_sig
            + self._alpha_mom * mom_sig
            + self._alpha_ltr * ltr_sig
        )
        combined_ret = (combined.shift(1) * returns).mean(axis=1).dropna()
        self._vol_scalar = _vol_target_scalar(
            combined_ret,
            target_vol=self.config.vol_target,
            lookback=self.config.vol_lookback,
        )

        # ---- Store parameters ----
        self.parameters = {
            "alpha_sr": self._alpha_sr,
            "alpha_mom": self._alpha_mom,
            "alpha_ltr": self._alpha_ltr,
            "sharpe_sr": sharpe_map["sr"],
            "sharpe_mom": sharpe_map["mom"],
            "sharpe_ltr": sharpe_map["ltr"],
            "vol_scalar": self._vol_scalar,
        }

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Generate signals
    # ------------------------------------------------------------------

    def generate_signals(self, prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate trading signals from multi-timeframe analysis.

        Steps:
        1. Compute the three sub-signals (MR, MOM, LTR).
        2. Combine with calibrated alpha weights.
        3. Apply conviction scaling (agreement between MR and MOM).
        4. Apply Kelly position sizing per asset.
        5. Volatility-target the combined portfolio.
        6. Enforce weekly rebalancing and per-asset caps.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (DatetimeIndex x tickers).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for each
            ticker in the input.
        """
        self.validate_prices(prices)
        cfg = self.config

        if not self._fitted:
            logger.warning(
                "Strategy not fitted; using default alpha weights and "
                "uniform Kelly/vol scaling."
            )
            self._alpha_sr = cfg.alpha_sr
            self._alpha_mom = cfg.alpha_mom
            self._alpha_ltr = cfg.alpha_ltr
            self._kelly_weights = None
            self._vol_scalar = 1.0

        # ---- 1. Sub-signals ----
        mr_sig = _short_term_mean_reversion(
            prices,
            return_window=cfg.mr_return_window,
            rolling_window=cfg.mr_rolling_window,
            entry_z=cfg.mr_entry_z,
        )
        mom_sig = _medium_term_momentum(
            prices,
            lookback=cfg.mom_lookback,
            skip=cfg.mom_skip,
            long_quantile=cfg.mom_long_quantile,
            short_quantile=cfg.mom_short_quantile,
        )
        ltr_sig = _long_term_reversal(
            prices,
            lookback=cfg.ltr_lookback,
        )

        # ---- 2. Weighted combination ----
        combined = (
            self._alpha_sr * mr_sig
            + self._alpha_mom * mom_sig
            + self._alpha_ltr * ltr_sig
        )

        # ---- 3. Conviction scaling ----
        conviction = _conviction_scaling(
            mr_sig, mom_sig,
            agree_multiplier=cfg.agree_multiplier,
            disagree_multiplier=cfg.disagree_multiplier,
        )
        combined = combined * conviction

        # ---- 4. Kelly position sizing ----
        if self._kelly_weights is not None:
            # Broadcast per-asset Kelly weights across all dates
            for col in prices.columns:
                if col in self._kelly_weights.index:
                    combined[col] = combined[col] * self._kelly_weights[col]

        # ---- 5. Volatility targeting ----
        combined = combined * self._vol_scalar

        # ---- 6. Weekly rebalancing ----
        # Only allow signal changes on rebalancing dates
        rebal_mask = np.zeros(len(prices), dtype=bool)
        rebal_mask[0] = True  # always rebalance on first date
        for i in range(cfg.rebalance_freq, len(prices), cfg.rebalance_freq):
            rebal_mask[i] = True

        # Forward-fill the signal between rebalancing dates
        combined_rebal = combined.copy()
        for col in combined.columns:
            vals = combined[col].values.copy()
            for i in range(len(vals)):
                if not rebal_mask[i] and i > 0:
                    vals[i] = vals[i - 1]
            combined_rebal[col] = vals

        # ---- 7. Per-asset cap ----
        combined_rebal = combined_rebal.clip(
            lower=-cfg.max_weight_per_asset,
            upper=cfg.max_weight_per_asset,
        )

        # ---- Build output DataFrame ----
        # Discretise signal direction and extract weight magnitude
        output = pd.DataFrame(index=prices.index)

        for col in prices.columns:
            raw = combined_rebal[col].fillna(0.0)
            # signal: direction {-1, 0, +1}
            output[f"{col}_signal"] = np.sign(raw).astype(int)
            # weight: magnitude in [0, 1]
            output[f"{col}_weight"] = raw.abs().clip(upper=1.0)

        # Also provide convenience single-ticker columns when there is
        # exactly one asset.
        if len(prices.columns) == 1:
            col = prices.columns[0]
            output["signal"] = output[f"{col}_signal"]
            output["weight"] = output[f"{col}_weight"]

        return output

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def decompose_signals(
        self, prices: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Return the three raw sub-signals for diagnostic inspection.

        This is not part of the Strategy interface but is useful for
        research and attribution analysis.

        Returns
        -------
        dict
            Keys: ``"mean_reversion"``, ``"momentum"``, ``"long_term_reversal"``.
        """
        cfg = self.config
        return {
            "mean_reversion": _short_term_mean_reversion(
                prices,
                return_window=cfg.mr_return_window,
                rolling_window=cfg.mr_rolling_window,
                entry_z=cfg.mr_entry_z,
            ),
            "momentum": _medium_term_momentum(
                prices,
                lookback=cfg.mom_lookback,
                skip=cfg.mom_skip,
                long_quantile=cfg.mom_long_quantile,
                short_quantile=cfg.mom_short_quantile,
            ),
            "long_term_reversal": _long_term_reversal(
                prices,
                lookback=cfg.ltr_lookback,
            ),
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        return (
            f"MultiTimeframeStrategy("
            f"alpha_sr={self._alpha_sr:.2f}, "
            f"alpha_mom={self._alpha_mom:.2f}, "
            f"alpha_ltr={self._alpha_ltr:.2f}, "
            f"{fitted_tag})"
        )
