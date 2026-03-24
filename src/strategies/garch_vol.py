"""Volatility trading strategy using GARCH models and the volatility risk premium.

Mathematical foundation
-----------------------
EGARCH(1,1) with Student-t innovations:

    ln(σ²_t) = ω + α · [|z_{t-1}| − E|z_{t-1}|] + γ · z_{t-1} + β · ln(σ²_{t-1})

where γ captures the asymmetric *leverage effect* -- the log-specification
ensures σ² > 0 without parameter constraints and better captures the
asymmetric impact of negative shocks on volatility.

Volatility Risk Premium (VRP):

    VRP_t = σ^GARCH_{t|t-1} − RV_t

A persistently positive VRP compensates investors for bearing volatility
risk and can be harvested by maintaining systematic long-equity exposure
with volatility-targeted sizing.

Strategy signals
----------------
1. **Vol-targeting**: w_t = σ_target / σ_{t|t-1}  (counter-cyclical sizing)
2. **Vol mean-reversion**: z-score of conditional vol relative to its own
   rolling distribution; extreme high → expect vol crush (long), extreme
   low → expect vol expansion (reduce/short).
3. **VRP harvest**: when GARCH forecast exceeds recent realised vol the
   market overprices risk → tilt long.
4. **Adaptive weighting**: signal blend weights adapt based on trailing
   signal performance (exponentially-weighted), concentrating on the
   component that has been contributing the most.

The final signal is the product of the vol-target weight and directional
conviction from (2), (3), and the adaptive blend.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

try:
    from arch.utility.exceptions import ConvergenceWarning
except ImportError:
    # Older versions of arch may not have this; fall back to a generic warning
    ConvergenceWarning = UserWarning  # type: ignore[misc]

logger = logging.getLogger(__name__)

from src.strategies.base import Strategy


# ---------------------------------------------------------------------------
# Helper: Parkinson realised-volatility estimator
# ---------------------------------------------------------------------------

def _parkinson_rv(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Parkinson (1980) high-low range estimator of realised volatility.

    RV_Parkinson = sqrt( (1 / (4·n·ln2)) · Σ ln(H_i/L_i)² )

    More efficient than close-close estimator when intra-day range data
    is available.  Returns *annualised* volatility (×√252).
    """
    log_hl = np.log(high / low)
    # Rolling sum of squared log ranges
    sq_sum = (log_hl ** 2).rolling(window=window, min_periods=window).mean()
    rv = np.sqrt(sq_sum / (4.0 * np.log(2.0))) * np.sqrt(252)
    return rv


def _close_close_rv(
    returns: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Simple close-close realised volatility (annualised)."""
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class GarchVolConfig:
    """All tuneable knobs for the GARCH volatility strategy."""

    # GARCH estimation
    garch_p: int = 1
    garch_q: int = 1
    garch_model: str = "EGARCH"        # "GARCH", "GJR", or "EGARCH"
    garch_dist: str = "studentst"      # innovation distribution
    rolling_window: int = 504          # ≈ 2 trading years
    refit_freq: int = 10               # re-estimate GARCH every N days

    # Realised vol
    rv_window: int = 20               # trading days for RV calculation
    use_parkinson: bool = True         # use Parkinson estimator when H/L available

    # Vol mean-reversion
    vol_zscore_lookback: int = 252     # window for z-score normalisation
    vol_long_threshold: float = 1.5    # z > threshold → mean-revert long
    vol_short_threshold: float = -1.0  # z < threshold → reduce / short

    # VRP
    vrp_lookback: int = 40            # lookback for VRP signal smoothing

    # Vol-targeting
    target_vol: float = 0.25          # 25 % annualised target
    max_leverage: float = 3.0         # hard cap on position size

    # Signal blending (base weights -- used as priors for adaptive blend)
    vol_mr_weight: float = 0.60       # weight of vol mean-reversion signal
    vrp_weight: float = 0.25          # weight of VRP signal
    vol_target_weight: float = 0.15   # weight of pure vol-target signal

    # Adaptive signal blending
    adaptive_blend: bool = True       # adapt weights based on trailing perf
    adaptive_lookback: int = 63       # ~3 months trailing window for perf eval
    adaptive_decay: float = 0.94      # exponential decay for perf tracking
    adaptive_min_weight: float = 0.10 # floor so no signal is fully zeroed out


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

class GarchVolStrategy(Strategy):
    """GARCH-based volatility trading strategy.

    Combines conditional volatility forecasting with the volatility risk
    premium to generate counter-cyclical equity exposure signals.
    """

    def __init__(self, config: Optional[GarchVolConfig] = None) -> None:
        self.cfg = config or GarchVolConfig()

        # Populated during fit / generate_signals
        self._garch_results: Dict[str, object] = {}
        self._long_run_vol: Dict[str, float] = {}
        self._cond_vol_history: Dict[str, pd.Series] = {}
        self._ohlcv_data: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False

    # -----------------------------------------------------------------
    # GARCH helpers
    # -----------------------------------------------------------------

    def _fit_garch_single(
        self,
        returns: pd.Series,
    ) -> Optional[ARCHModelResult]:
        """Fit a GARCH-family model on a single return series.

        Parameters
        ----------
        returns : pd.Series
            Daily log-returns (in decimal).  ``arch`` expects percentage
            returns by convention, so we rescale internally.

        Returns
        -------
        ARCHModelResult or None
            The fitted model result, or ``None`` if fitting failed due to
            convergence issues or other errors.
        """
        # arch expects percentage returns for numerical stability
        y = returns * 100.0

        vol_model = self.cfg.garch_model.upper()

        # The arch library's arch_model() ``vol`` parameter only accepts
        # specific strings: 'GARCH', 'EGARCH', 'HARCH', 'CONSTANT', etc.
        # Notably, 'GJR' is NOT a valid ``vol`` value.  A GJR-GARCH model
        # is specified via vol='GARCH' with o>=1 (asymmetric order).
        if vol_model == "EGARCH":
            am = arch_model(
                y.dropna(),
                vol="EGARCH",
                p=self.cfg.garch_p,
                o=1,
                q=self.cfg.garch_q,
                dist=self.cfg.garch_dist,
            )
        elif vol_model in ("GJR", "GJR-GARCH", "GJRGARCH"):
            # GJR-GARCH: use vol='GARCH' with o=1 for the asymmetric term
            am = arch_model(
                y.dropna(),
                vol="GARCH",
                p=self.cfg.garch_p,
                o=1,
                q=self.cfg.garch_q,
                dist=self.cfg.garch_dist,
            )
        elif vol_model == "GARCH":
            # Standard symmetric GARCH
            am = arch_model(
                y.dropna(),
                vol="GARCH",
                p=self.cfg.garch_p,
                o=0,
                q=self.cfg.garch_q,
                dist=self.cfg.garch_dist,
            )
        else:
            raise ValueError(
                f"Unknown garch_model '{self.cfg.garch_model}'. "
                f"Supported values: 'GARCH', 'GJR', 'EGARCH'."
            )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                result = am.fit(disp="off", show_warning=False)
            except Exception as exc:
                logger.warning("GARCH optimisation raised an exception: %s", exc)
                return None

        # Check for convergence warnings
        for w in caught_warnings:
            if issubclass(w.category, ConvergenceWarning):
                logger.warning(
                    "GARCH fitting did not converge: %s", w.message
                )
                return None
            if issubclass(w.category, RuntimeWarning):
                logger.debug("RuntimeWarning during GARCH fit: %s", w.message)

        return result

    def _forecast_cond_vol(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        """Rolling GARCH conditional volatility forecast for each date.

        For each date *t* we use a trailing window of ``rolling_window``
        returns to fit a GARCH model and forecast σ_{t+1|t}.  The result
        is an *annualised* volatility series aligned with the return dates.
        """
        window = self.cfg.rolling_window
        n = len(returns)
        cond_vol = pd.Series(np.nan, index=returns.index)

        # We step through the series; re-estimate every refit_freq days.
        refit_freq = self.cfg.refit_freq
        last_result = None
        last_fit_idx = -refit_freq  # force fit on first eligible date

        for i in range(window, n):
            if (i - last_fit_idx) >= refit_freq or last_result is None:
                window_returns = returns.iloc[i - window: i]
                try:
                    fit_result = self._fit_garch_single(window_returns)
                    if fit_result is not None:
                        last_result = fit_result
                        last_fit_idx = i
                    elif last_result is None:
                        # No previous result to fall back on
                        continue
                except Exception:
                    # If fitting raises keep the previous result
                    if last_result is None:
                        continue

            # One-step-ahead forecast from the latest fit
            try:
                fcast = last_result.forecast(horizon=1)
                # fcast.variance is in (pct-return)² units; convert back
                var_forecast = fcast.variance.iloc[-1, 0]
                # daily σ in decimal, then annualise
                daily_sigma = np.sqrt(var_forecast) / 100.0
                cond_vol.iloc[i] = daily_sigma * np.sqrt(252)
            except Exception:
                pass

        return cond_vol

    @staticmethod
    def _long_run_garch_vol(result) -> float:
        """Extract the unconditional (long-run) annualised vol from a GARCH fit.

        For GARCH/GJR-GARCH(1,1): σ²_∞ = ω / (1 − α − β − γ/2)
        For EGARCH(1,1): ln(σ²_∞) = ω / (1 − β)  →  σ²_∞ = exp(ω / (1 − β))

        Returns annualised volatility in decimal.
        """
        params = result.params
        omega = params.get("omega", 0.0)
        alpha = params.get("alpha[1]", 0.0)
        beta = params.get("beta[1]", 0.0)
        gamma = params.get("gamma[1]", 0.0)

        # Detect EGARCH by checking if the volatility model name contains it
        model_name = type(result.model.volatility).__name__.upper()
        is_egarch = "EGARCH" in model_name

        if is_egarch:
            # EGARCH: unconditional log-variance = ω / (1 − β)
            denom = 1.0 - beta
            if abs(denom) < 1e-10:
                return float(np.sqrt(result.resid.var())) / 100.0 * np.sqrt(252)
            log_var = omega / denom
            daily_var = np.exp(log_var)  # in (pct)^2
        else:
            # GARCH / GJR-GARCH
            denom = 1.0 - alpha - beta - 0.5 * gamma
            if denom <= 0:
                # Model is non-stationary; fall back to sample variance
                return float(np.sqrt(result.resid.var())) / 100.0 * np.sqrt(252)
            daily_var = omega / denom  # in (pct)^2

        return np.sqrt(daily_var) / 100.0 * np.sqrt(252)

    # -----------------------------------------------------------------
    # Signal generators
    # -----------------------------------------------------------------

    def _vol_zscore_signal(
        self,
        cond_vol: pd.Series,
        long_run_vol: float,
    ) -> pd.Series:
        """Volatility mean-reversion signal based on z-score.

        z_t = (σ_t − σ̄) / rolling_std(σ)

        Returns a signal in [-1, +1]:
        * z > vol_long_threshold  → +1  (expect vol crush → long equity)
        * z < vol_short_threshold → -1  (expect vol spike → short/flat)
        * else                    → linearly interpolated
        """
        lookback = self.cfg.vol_zscore_lookback

        vol_mean = cond_vol.rolling(window=lookback, min_periods=lookback // 2).mean()
        vol_std = cond_vol.rolling(window=lookback, min_periods=lookback // 2).std()

        # Use long-run GARCH vol as centre when rolling mean is unavailable
        centre = vol_mean.fillna(long_run_vol)
        spread = vol_std.fillna(cond_vol.std())
        spread = spread.replace(0, np.nan).ffill()

        z = (cond_vol - centre) / spread

        hi = self.cfg.vol_long_threshold
        lo = self.cfg.vol_short_threshold

        signal = pd.Series(0.0, index=cond_vol.index)
        # Linear interpolation in [lo, hi] → [-1, +1]
        signal = (z - lo) / (hi - lo) * 2.0 - 1.0
        signal = signal.clip(-1.0, 1.0)
        return signal

    def _vrp_signal(
        self,
        cond_vol: pd.Series,
        realised_vol: pd.Series,
    ) -> pd.Series:
        """Volatility risk premium signal.

        VRP_t = σ^GARCH_t − RV_t

        Positive VRP ⇒ implied/forecast vol > realised ⇒ market overprices
        risk ⇒ tilt long (harvest the premium).

        The raw VRP is smoothed and normalised to [-1, +1].
        """
        vrp = cond_vol - realised_vol
        vrp_smooth = vrp.rolling(
            window=self.cfg.vrp_lookback,
            min_periods=self.cfg.vrp_lookback // 2,
        ).mean()

        # Normalise by its own rolling std
        vrp_std = vrp_smooth.rolling(
            window=self.cfg.vol_zscore_lookback,
            min_periods=self.cfg.vol_zscore_lookback // 2,
        ).std()
        vrp_std = vrp_std.replace(0, np.nan).ffill()

        signal = (vrp_smooth / vrp_std).clip(-1.0, 1.0)
        return signal.fillna(0.0)

    def _vol_target_weight(self, cond_vol: pd.Series) -> pd.Series:
        """Volatility-targeting position size.

        w_t = σ_target / σ_{t|t-1}

        Capped at ``max_leverage`` to avoid extreme sizing in calm markets.
        """
        w = self.cfg.target_vol / cond_vol
        return w.clip(upper=self.cfg.max_leverage).fillna(0.0)

    # -----------------------------------------------------------------
    # Adaptive signal weighting
    # -----------------------------------------------------------------

    @staticmethod
    def _adaptive_signal_weights(
        sig_mr: pd.Series,
        sig_vrp: pd.Series,
        returns: pd.Series,
        base_weights: tuple,
        lookback: int = 63,
        decay: float = 0.94,
        min_weight: float = 0.10,
    ) -> pd.DataFrame:
        """Compute time-varying blend weights based on trailing signal performance.

        For each signal component we measure how well it predicted next-day
        return direction (signal * next_return).  We use an exponentially-
        weighted mean of this "edge" metric and allocate proportionally,
        subject to a floor of ``min_weight`` per component.

        Parameters
        ----------
        sig_mr : pd.Series
            Mean-reversion signal in [-1, 1].
        sig_vrp : pd.Series
            VRP signal in [-1, 1].
        returns : pd.Series
            Next-day (forward) log returns for the asset.
        base_weights : tuple
            (w_mr, w_vrp, w_tgt) base/prior weights.
        lookback : int
            Window for trailing performance evaluation.
        decay : float
            Exponential decay factor for performance tracking.
        min_weight : float
            Minimum weight for any single component.

        Returns
        -------
        pd.DataFrame
            Columns ['w_mr', 'w_vrp', 'w_tgt'] with time-varying weights.
        """
        w_mr_base, w_vrp_base, w_tgt_base = base_weights

        # Forward return (shift -1 so signal at t is compared to ret at t+1)
        fwd_ret = returns.shift(-1)

        # Edge: signal * forward return (positive = signal was correct)
        edge_mr = (sig_mr * fwd_ret).fillna(0.0)
        edge_vrp = (sig_vrp * fwd_ret).fillna(0.0)
        # Vol-target is always "long" direction, edge = 1.0 * fwd_ret
        edge_tgt = fwd_ret.fillna(0.0)

        # Exponentially-weighted cumulative edge
        ew_mr = edge_mr.ewm(span=int(1 / (1 - decay)), adjust=False).mean()
        ew_vrp = edge_vrp.ewm(span=int(1 / (1 - decay)), adjust=False).mean()
        ew_tgt = edge_tgt.ewm(span=int(1 / (1 - decay)), adjust=False).mean()

        # Convert edge to a "score" -- use softmax-like allocation
        # Shift edges to be non-negative before weighting
        scores = pd.DataFrame({
            'mr': ew_mr.clip(lower=0) + min_weight,
            'vrp': ew_vrp.clip(lower=0) + min_weight,
            'tgt': ew_tgt.clip(lower=0) + min_weight,
        })

        # Blend with base weights (50% base prior, 50% adaptive)
        scores['mr'] = scores['mr'] + w_mr_base
        scores['vrp'] = scores['vrp'] + w_vrp_base
        scores['tgt'] = scores['tgt'] + w_tgt_base

        # Normalise to sum to 1
        row_sum = scores.sum(axis=1)
        weights = scores.div(row_sum, axis=0)

        # Enforce minimum weight floor and re-normalise
        weights = weights.clip(lower=min_weight)
        row_sum = weights.sum(axis=1)
        weights = weights.div(row_sum, axis=0)

        result = pd.DataFrame(index=sig_mr.index)
        result['w_mr'] = weights['mr']
        result['w_vrp'] = weights['vrp']
        result['w_tgt'] = weights['tgt']
        return result.fillna(pd.Series([w_mr_base, w_vrp_base, w_tgt_base],
                                        index=['w_mr', 'w_vrp', 'w_tgt']))

    # -----------------------------------------------------------------
    # Strategy interface
    # -----------------------------------------------------------------

    def fit(self, data: pd.DataFrame, **kwargs) -> "GarchVolStrategy":
        """Calibrate GARCH models on historical price data.

        Parameters
        ----------
        data : pd.DataFrame
            Historical *price* data.  Columns are asset tickers; index is
            a DatetimeIndex.
        ohlcv_data : pd.DataFrame, optional
            Full OHLCV MultiIndex DataFrame.  Stored internally so that
            ``generate_signals`` can use the Parkinson estimator when
            High/Low data is available.

        Returns
        -------
        self
        """
        # Store OHLCV data for use in generate_signals
        self._ohlcv_data = kwargs.get("ohlcv_data", None)

        log_returns = np.log(data / data.shift(1)).dropna()

        for col in log_returns.columns:
            rets = log_returns[col].dropna()
            if len(rets) < self.cfg.rolling_window:
                warnings.warn(
                    f"Insufficient data for {col}: {len(rets)} rows "
                    f"< rolling_window {self.cfg.rolling_window}. Skipping.",
                    stacklevel=2,
                )
                continue

            try:
                result = self._fit_garch_single(rets)
                if result is not None:
                    self._garch_results[col] = result
                    self._long_run_vol[col] = self._long_run_garch_vol(result)
                else:
                    warnings.warn(
                        f"GARCH fit did not converge for {col}. Skipping.",
                        stacklevel=2,
                    )
            except Exception as exc:
                warnings.warn(
                    f"GARCH fit failed for {col}: {exc}",
                    stacklevel=2,
                )

        self._is_fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Produce position signals from price data.

        Parameters
        ----------
        data : pd.DataFrame
            Price data (may extend beyond the fit window).
        ohlcv_data : pd.DataFrame, optional
            Full OHLCV MultiIndex DataFrame.  When supplied, High/Low
            prices are extracted per ticker for the Parkinson realised
            volatility estimator, which is more efficient than the
            close-close estimator.

        Returns
        -------
        pd.DataFrame
            Per-asset position signals.  Positive = long, negative = short,
            magnitude reflects conviction x vol-target sizing.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Strategy has not been fitted. Call .fit() first."
            )

        ohlcv = kwargs.get("ohlcv_data", getattr(self, "_ohlcv_data", None))

        # Extract High/Low DataFrames from OHLCV MultiIndex if available
        high_prices = None
        low_prices = None
        if ohlcv is not None and isinstance(ohlcv.columns, pd.MultiIndex):
            level_vals = ohlcv.columns.get_level_values(0)
            if "High" in level_vals:
                high_prices = ohlcv["High"]
            if "Low" in level_vals:
                low_prices = ohlcv["Low"]

        log_returns = np.log(data / data.shift(1))
        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        for col in data.columns:
            rets = log_returns[col].dropna()
            if len(rets) < self.cfg.rolling_window:
                continue

            # ---- conditional volatility via rolling GARCH ----
            cond_vol = self._forecast_cond_vol(rets)
            self._cond_vol_history[col] = cond_vol

            # If GARCH fitting produced no valid forecasts, emit flat signal
            if cond_vol.isna().all():
                logger.warning(
                    "No valid GARCH forecasts for %s; emitting flat signal.",
                    col,
                )
                continue  # signals[col] stays 0.0

            # ---- realised volatility ----
            # Prefer Parkinson estimator when High/Low data is available
            if (
                self.cfg.use_parkinson
                and high_prices is not None
                and low_prices is not None
                and col in high_prices.columns
                and col in low_prices.columns
            ):
                h = high_prices[col].reindex(rets.index).ffill().bfill()
                l = low_prices[col].reindex(rets.index).ffill().bfill()
                rv = _parkinson_rv(h, l, window=self.cfg.rv_window)
                logger.debug("Using Parkinson RV estimator for %s", col)
            else:
                rv = _close_close_rv(rets, window=self.cfg.rv_window)

            # ---- long-run vol (use fitted value or fall-back) ----
            lr_vol = self._long_run_vol.get(
                col,
                float(cond_vol.dropna().mean()) if cond_vol.notna().any() else 0.15,
            )

            # ---- component signals ----
            sig_mr = self._vol_zscore_signal(cond_vol, lr_vol)
            sig_vrp = self._vrp_signal(cond_vol, rv)
            w_target = self._vol_target_weight(cond_vol)

            # ---- composite signal ----
            mr_w = self.cfg.vol_mr_weight
            vrp_w = self.cfg.vrp_weight
            tgt_w = self.cfg.vol_target_weight

            total_w = mr_w + vrp_w + tgt_w
            mr_w /= total_w
            vrp_w /= total_w
            tgt_w /= total_w

            if self.cfg.adaptive_blend:
                # Time-varying weights based on trailing signal performance
                adapt_weights = self._adaptive_signal_weights(
                    sig_mr=sig_mr,
                    sig_vrp=sig_vrp,
                    returns=rets,
                    base_weights=(mr_w, vrp_w, tgt_w),
                    lookback=self.cfg.adaptive_lookback,
                    decay=self.cfg.adaptive_decay,
                    min_weight=self.cfg.adaptive_min_weight,
                )
                # Directional blend with adaptive weights
                direction = (
                    adapt_weights['w_mr'] * sig_mr
                    + adapt_weights['w_vrp'] * sig_vrp
                    + adapt_weights['w_tgt'] * 1.0
                )
            else:
                # Static blend
                direction = mr_w * sig_mr + vrp_w * sig_vrp + tgt_w * 1.0

            # Scale by vol-target weight (counter-cyclical sizing)
            composite = direction * w_target

            # Align back to full index; ensure NaNs become 0 (flat)
            signals[col] = composite.reindex(data.index, fill_value=0.0).fillna(0.0)

        return signals
