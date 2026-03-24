"""Ergodic growth strategy based on time-average vs ensemble-average returns.

Implements a trading strategy grounded in ergodicity economics (Ole Peters'
framework), which distinguishes between ensemble-average and time-average
growth rates — a distinction that classical expected-utility theory conflates.

Mathematical foundation
-----------------------
For multiplicative wealth dynamics W_{t+1} = W_t * (1 + r_t), the ensemble
average and time average diverge:

    Ensemble average:    E[W_T] = W_0 * (1 + mu)^T
    Time average:        W_T    = W_0 * exp(T * E[log(1 + r)])

The **ergodicity gap** quantifies the "variance tax" imposed by
multiplicative dynamics:

    Delta = mu - g = E[r] - E[log(1 + r)]  ~=  sigma^2 / 2

This follows from Jensen's inequality applied to the concave log function.
A strategy with high arithmetic mean mu but also high variance sigma^2 can
have a *negative* time-average growth rate — meaning it destroys wealth
almost surely despite looking profitable in expectation.

The growth-optimal (Kelly) leverage for a single asset is:

    f* = mu / sigma^2          (maximises geometric growth rate)

and the geometric growth rate under leverage f is:

    g(f) = f * mu  -  f^2 * sigma^2 / 2

which is maximised at f = f* with g(f*) = mu^2 / (2 * sigma^2).

Strategy
--------
1.  **Geometric growth rate estimation** — For each asset, compute the
    time-average growth rate g = mean(log(1 + r)), the arithmetic mean
    mu = mean(r), and the ergodicity gap Delta = mu - g.  Assets with
    large gaps pay a heavy variance tax and are penalised.

2.  **Growth-optimal allocation** — Rank assets by geometric growth
    rate g (NOT arithmetic mean mu).  Go long the top quintile, short
    the bottom quintile.  This is fundamentally different from momentum
    strategies that rank by mu.

3.  **Dynamic leverage via fractional Kelly** — For each asset, compute
    the optimal Kelly fraction f* = g / sigma^2.  Scale by a safety
    factor (0.25 by default = quarter-Kelly).  This naturally allocates
    more capital to low-variance, high-growth assets.

4.  **Variance drag monitor** — Track the portfolio-level variance tax
    sigma^2_p / 2.  When the variance tax exceeds the portfolio growth
    rate, the portfolio is on an almost-surely wealth-destroying path.
    Automatically reduce leverage (toward zero) when variance drag
    dominates.

References
----------
*   Peters, O. (2019). The ergodicity problem in economics. Nature
    Physics 15, 1216-1221.
*   Peters, O. & Gell-Mann, M. (2016). Evaluating gambles using
    dynamics. Chaos 26(2).
*   Kelly, J. L. (1956). A new interpretation of information rate.
    Bell System Technical Journal 35(4).
*   Peters, O. & Adamou, A. (2021). The ergodicity solution of the
    cooperation puzzle. Philosophical Transactions of the Royal
    Society A 380(2227).
*   Cover, T. M. & Thomas, J. A. (2006). Elements of Information
    Theory, Ch. 16 (Portfolio theory).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ErgodicGrowthConfig:
    """Tuneable parameters for the ergodic growth strategy."""

    # Estimation windows
    lookback: int = 252                 # rolling window for growth rate / variance
    min_history: int = 252              # minimum observations before trading

    # Portfolio construction
    long_quantile: float = 0.2         # top quantile of g to go long
    short_quantile: float = 0.2        # bottom quantile of g to go short
    long_only: bool = False            # if True, suppress short positions

    # Fractional Kelly
    kelly_fraction: float = 0.25       # quarter-Kelly default
    min_kelly: float = 0.01            # floor on per-asset Kelly weight
    max_kelly: float = 0.40            # ceiling on per-asset Kelly weight

    # Constraints
    max_position_weight: float = 0.20  # max absolute weight per asset
    max_gross_leverage: float = 1.50   # max sum of |w_i|

    # Variance drag safety
    drag_safety_multiplier: float = 0.5  # scale leverage by this when drag > growth
    drag_critical_ratio: float = 1.0     # ratio of var_drag / g that triggers cutback

    # Ergodicity gap filter
    max_ergodicity_gap: Optional[float] = None  # if set, exclude assets with gap > this

    # Rebalancing
    rebalance_freq: int = 21           # maximum days between rebalances

    # Smoothing
    ema_span: int = 21                 # EMA span for growth rate estimates


# ---------------------------------------------------------------------------
# Core computations (pure numpy)
# ---------------------------------------------------------------------------

def _compute_growth_metrics(
    log_returns: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute per-asset ergodic growth metrics from log returns.

    Parameters
    ----------
    log_returns : (T, N) array
        Log returns for N assets over T periods.  NaN-safe: columns with
        all NaN produce NaN metrics.

    Returns
    -------
    dict with keys:
        'geometric_growth'  : (N,) — time-average growth rate g = mean(log(1+r))
                                      Since input is already log returns, g = mean(lr).
        'arithmetic_mean'   : (N,) — ensemble-average mu = mean(exp(lr) - 1)
        'ergodicity_gap'    : (N,) — Delta = mu - g
        'variance'          : (N,) — variance of simple returns
        'variance_drag'     : (N,) — sigma^2 / 2
        'kelly_fraction'    : (N,) — f* = g / var(r), clipped to avoid blow-up
    """
    T, N = log_returns.shape

    # Time-average growth rate: mean of log returns
    # (log returns ARE log(1 + r) when computed as log(P_t / P_{t-1}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        g = np.nanmean(log_returns, axis=0)  # (N,)

    # Arithmetic mean of simple returns
    simple_returns = np.exp(log_returns) - 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu = np.nanmean(simple_returns, axis=0)  # (N,)

    # Ergodicity gap: Delta = mu - g  (~= sigma^2/2 by Jensen's inequality)
    gap = mu - g  # (N,)

    # Variance of simple returns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        var = np.nanvar(simple_returns, axis=0, ddof=1)  # (N,)

    # Variance drag: sigma^2 / 2
    var_drag = var / 2.0  # (N,)

    # Kelly fraction: f* = g / var
    # Guard against zero/negative variance
    var_safe = np.where(var > 1e-12, var, np.inf)
    kelly = g / var_safe  # (N,)

    return {
        "geometric_growth": g,
        "arithmetic_mean": mu,
        "ergodicity_gap": gap,
        "variance": var,
        "variance_drag": var_drag,
        "kelly_fraction": kelly,
    }


def _rank_assets_by_growth(
    g: np.ndarray,
    long_q: float,
    short_q: float,
    long_only: bool,
) -> np.ndarray:
    """Assign direction signals based on geometric growth rate ranking.

    Parameters
    ----------
    g : (N,) array
        Geometric growth rates for each asset.
    long_q : float
        Fraction of assets (top quantile) to go long.
    short_q : float
        Fraction of assets (bottom quantile) to go short.
    long_only : bool
        If True, no short positions are assigned.

    Returns
    -------
    signals : (N,) array
        +1 (long), -1 (short), or 0 (flat) for each asset.
    """
    N = len(g)
    signals = np.zeros(N)

    # Handle NaN growth rates: treat as flat
    valid_mask = ~np.isnan(g)
    n_valid = valid_mask.sum()

    if n_valid < 2:
        return signals

    # Rank valid assets (higher g = better rank)
    ranks = np.full(N, np.nan)
    valid_indices = np.where(valid_mask)[0]
    valid_g = g[valid_mask]
    order = np.argsort(valid_g)  # ascending: worst first
    for rank_pos, idx in enumerate(order):
        ranks[valid_indices[idx]] = rank_pos

    # Determine cutoffs
    n_long = max(1, int(np.ceil(n_valid * long_q)))
    n_short = max(1, int(np.ceil(n_valid * short_q)))

    long_threshold = n_valid - n_long  # ranks >= this are long
    short_threshold = n_short          # ranks < this are short

    for i in range(N):
        if np.isnan(ranks[i]):
            continue
        if ranks[i] >= long_threshold:
            signals[i] = 1.0
        elif ranks[i] < short_threshold and not long_only:
            signals[i] = -1.0

    return signals


def _compute_kelly_weights(
    signals: np.ndarray,
    kelly_fractions: np.ndarray,
    kelly_scale: float,
    min_kelly: float,
    max_kelly: float,
) -> np.ndarray:
    """Compute position weights using fractional Kelly sizing.

    Parameters
    ----------
    signals : (N,) array
        Direction signals (+1, -1, 0).
    kelly_fractions : (N,) array
        Raw Kelly fractions (f* = g / sigma^2) for each asset.
    kelly_scale : float
        Fraction of Kelly to use (e.g. 0.25 = quarter-Kelly).
    min_kelly : float
        Floor for absolute weight of any active position.
    max_kelly : float
        Ceiling for absolute weight of any active position.

    Returns
    -------
    weights : (N,) array
        Signed position weights (signal * |kelly_weight|).
    """
    active = signals != 0.0
    N = len(signals)
    weights = np.zeros(N)

    if not np.any(active):
        return weights

    # Scaled Kelly fractions — use absolute value (direction is in signals)
    raw_w = np.abs(kelly_fractions) * kelly_scale

    # Clamp per-asset
    raw_w = np.clip(raw_w, min_kelly, max_kelly)

    # Apply direction
    weights[active] = signals[active] * raw_w[active]

    return weights


def _apply_constraints(
    weights: np.ndarray,
    max_position: float,
    max_leverage: float,
) -> np.ndarray:
    """Enforce per-asset and portfolio-level constraints.

    Parameters
    ----------
    weights : (N,) array
        Raw position weights.
    max_position : float
        Maximum absolute weight per asset.
    max_leverage : float
        Maximum gross leverage (sum of |w_i|).

    Returns
    -------
    constrained : (N,) array
    """
    w = weights.copy()

    # Per-asset cap
    w = np.clip(w, -max_position, max_position)

    # Gross leverage cap
    gross = np.abs(w).sum()
    if gross > max_leverage and gross > 1e-12:
        w *= max_leverage / gross

    return w


def _variance_drag_leverage_adjustment(
    portfolio_growth_rate: float,
    portfolio_variance_drag: float,
    critical_ratio: float,
    safety_multiplier: float,
) -> float:
    """Compute a leverage scale factor based on the variance drag monitor.

    When the portfolio variance drag exceeds the growth rate (i.e., the
    portfolio is on a wealth-destroying trajectory), scale down leverage.

    Parameters
    ----------
    portfolio_growth_rate : float
        Current estimated portfolio geometric growth rate.
    portfolio_variance_drag : float
        Current portfolio variance drag = sigma_p^2 / 2.
    critical_ratio : float
        Ratio of drag/growth at which to trigger cutback.
    safety_multiplier : float
        Multiplicative factor to apply to weights when drag is excessive.

    Returns
    -------
    scale : float
        Leverage scale factor in (0, 1].  1.0 means no adjustment.
    """
    if portfolio_growth_rate <= 1e-12:
        # Growth rate is zero or negative; if drag is positive, cut back
        if portfolio_variance_drag > 1e-12:
            return safety_multiplier
        return 1.0

    ratio = portfolio_variance_drag / portfolio_growth_rate

    if ratio > critical_ratio:
        # Drag dominates: scale down proportionally, floored at safety_multiplier
        # Linear interpolation: at ratio = critical, scale = 1.0;
        # at ratio = 2 * critical, scale = safety_multiplier
        scale = 1.0 - (1.0 - safety_multiplier) * min(
            (ratio - critical_ratio) / max(critical_ratio, 1e-12), 1.0
        )
        return max(scale, safety_multiplier)

    return 1.0


def _portfolio_variance(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """Compute portfolio variance w' Sigma w.

    Parameters
    ----------
    weights : (N,) array
    cov_matrix : (N, N) array

    Returns
    -------
    float
    """
    return float(weights @ cov_matrix @ weights)


def _estimate_covariance_shrunk(
    returns: np.ndarray,
    shrinkage_target: str = "identity",
) -> np.ndarray:
    """Estimate covariance with simple shrinkage (pure numpy, no sklearn).

    Uses the Oracle Approximating Shrinkage (OAS) estimator toward a
    scaled identity target.

    Parameters
    ----------
    returns : (T, N) array
        Return observations (NaN-free).
    shrinkage_target : str
        Currently only 'identity' is supported.

    Returns
    -------
    Sigma : (N, N) array
        Shrunk covariance estimate.
    """
    T, N = returns.shape

    if T < 2:
        return np.eye(N)

    # Sample covariance
    sample_cov = np.cov(returns, rowvar=False, ddof=1)

    if N == 1:
        return sample_cov.reshape(1, 1)

    # Shrinkage target: scaled identity
    trace = np.trace(sample_cov)
    mu_target = trace / N
    target = mu_target * np.eye(N)

    # Ledoit-Wolf analytical shrinkage intensity
    # (simplified; see Ledoit & Wolf 2004, Lemma 3.1)
    delta = sample_cov - target
    delta_sq_sum = np.sum(delta ** 2)

    # Frobenius norm of centred outer products
    X = returns - returns.mean(axis=0)
    beta = 0.0
    for t in range(T):
        outer = np.outer(X[t], X[t]) - sample_cov
        beta += np.sum(outer ** 2)
    beta /= T ** 2

    if delta_sq_sum < 1e-15:
        alpha = 1.0
    else:
        alpha = min(beta / delta_sq_sum, 1.0)

    Sigma = alpha * target + (1.0 - alpha) * sample_cov

    # Ensure symmetry
    Sigma = 0.5 * (Sigma + Sigma.T)

    return Sigma


# ===========================================================================
# Strategy class
# ===========================================================================

class ErgodicGrowthStrategy(Strategy):
    """Ergodicity-economics portfolio strategy.

    Exploits the divergence between ensemble-average and time-average
    growth rates to construct a portfolio that maximises the geometric
    (time-average) growth rate of wealth:

    1.  Estimate per-asset geometric growth rates g = mean(log(1 + r)),
        arithmetic means mu = mean(r), and the ergodicity gap Delta.
    2.  Rank assets by g (not mu!) and go long the top quintile, short
        the bottom quintile.
    3.  Size positions using fractional Kelly: f = 0.25 * g / sigma^2.
    4.  Monitor portfolio-level variance drag; reduce leverage when
        the variance tax threatens to overwhelm the growth rate.

    Parameters
    ----------
    config : ErgodicGrowthConfig, optional
        Strategy configuration.  Uses sensible defaults if not provided.
    """

    def __init__(self, config: Optional[ErgodicGrowthConfig] = None) -> None:
        self.cfg = config or ErgodicGrowthConfig()

        super().__init__(
            name="ErgodicGrowth",
            description=(
                "Ergodicity-economics strategy that maximises the time-average "
                "(geometric) growth rate by ranking assets on g = E[log(1+r)] "
                "rather than mu = E[r], using fractional Kelly sizing, and "
                "monitoring the variance drag sigma^2/2."
            ),
        )

        # State populated during fit
        self._asset_names: Optional[pd.Index] = None
        self._growth_metrics: Optional[Dict[str, np.ndarray]] = None

        # Diagnostics populated during generate_signals
        self._weight_history: Optional[pd.DataFrame] = None
        self._growth_rate_history: Optional[pd.DataFrame] = None
        self._ergodicity_gap_history: Optional[pd.DataFrame] = None
        self._variance_drag_history: Optional[pd.Series] = None
        self._leverage_scale_history: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Strategy interface: fit
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "ErgodicGrowthStrategy":
        """Calibrate growth rate estimates on historical prices.

        Computes geometric growth rates, arithmetic means, ergodicity
        gaps, and Kelly fractions over the trailing lookback window.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (DatetimeIndex, columns = tickers).

        Returns
        -------
        self
        """
        self.validate_prices(prices)
        self._asset_names = prices.columns

        log_returns = np.log(prices / prices.shift(1)).dropna()
        tail = log_returns.iloc[-self.cfg.lookback :]

        if len(tail) < max(self.cfg.min_history // 2, 30):
            warnings.warn(
                f"Insufficient history for ergodic growth estimation: "
                f"{len(tail)} observations (need at least "
                f"{self.cfg.min_history // 2}).",
                stacklevel=2,
            )

        lr_array = tail.values.astype(np.float64)
        lr_array = np.nan_to_num(lr_array, nan=0.0)

        self._growth_metrics = _compute_growth_metrics(lr_array)

        self.parameters = {
            "lookback": self.cfg.lookback,
            "kelly_fraction": self.cfg.kelly_fraction,
            "long_quantile": self.cfg.long_quantile,
            "short_quantile": self.cfg.short_quantile,
            "long_only": self.cfg.long_only,
            "n_assets": len(self._asset_names),
            "estimation_window": len(tail),
        }

        self._fitted = True
        logger.info(
            "ErgodicGrowth fit complete: %d assets, %d-day estimation window.",
            len(self._asset_names),
            len(tail),
        )
        return self

    # ------------------------------------------------------------------
    # Strategy interface: generate_signals
    # ------------------------------------------------------------------

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate ergodic-growth portfolio signals.

        Walks forward through the price data, re-estimating growth
        metrics at each rebalance point and constructing positions
        that maximise the time-average growth rate.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (DatetimeIndex, columns = tickers).

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for
            each asset.
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        n_assets = prices.shape[1]
        n_dates = len(prices)
        tickers = list(prices.columns)
        log_returns = np.log(prices / prices.shift(1))

        # Output storage
        output_cols: List[str] = []
        for t in tickers:
            output_cols.extend([f"{t}_signal", f"{t}_weight"])
        signals_df = pd.DataFrame(0.0, index=prices.index, columns=output_cols)

        # Diagnostics storage
        all_weights = np.zeros((n_dates, n_assets))
        growth_g = np.full((n_dates, n_assets), np.nan)     # geometric growth per asset
        growth_mu = np.full((n_dates, n_assets), np.nan)    # arithmetic mean per asset
        ergo_gap = np.full((n_dates, n_assets), np.nan)     # ergodicity gap
        port_var_drag = np.full(n_dates, np.nan)
        leverage_scales = np.full(n_dates, np.nan)

        # Portfolio tracking
        current_weights = np.zeros(n_assets)
        current_cov = np.eye(n_assets)
        last_rebalance_idx = -self.cfg.rebalance_freq  # force first rebalance

        # EMA state for growth rate smoothing
        g_ema: Optional[np.ndarray] = None

        for t_idx in range(n_dates):
            if t_idx < self.cfg.min_history:
                # Not enough history — stay flat
                all_weights[t_idx] = 0.0
                continue

            # --- Check if rebalancing is needed ---
            if (t_idx - last_rebalance_idx) < self.cfg.rebalance_freq:
                # Hold current weights
                all_weights[t_idx] = current_weights
                continue

            # --- Re-estimate growth metrics ---
            start = max(0, t_idx - self.cfg.lookback)
            window_lr = log_returns.iloc[start:t_idx].values.astype(np.float64)
            window_lr = np.nan_to_num(window_lr, nan=0.0)

            T_eff = window_lr.shape[0]
            if T_eff < max(n_assets + 5, 30):
                all_weights[t_idx] = current_weights
                continue

            metrics = _compute_growth_metrics(window_lr)
            g = metrics["geometric_growth"]       # (N,)
            mu = metrics["arithmetic_mean"]       # (N,)
            gap = metrics["ergodicity_gap"]       # (N,)
            var = metrics["variance"]             # (N,)
            var_d = metrics["variance_drag"]      # (N,)
            kelly_f = metrics["kelly_fraction"]   # (N,)

            # Store diagnostics
            growth_g[t_idx] = g
            growth_mu[t_idx] = mu
            ergo_gap[t_idx] = gap

            # --- EMA smoothing of growth rate estimates ---
            ema_alpha = 2.0 / (self.cfg.ema_span + 1.0)
            if g_ema is None:
                g_ema = g.copy()
            else:
                g_ema = ema_alpha * g + (1.0 - ema_alpha) * g_ema

            g_smooth = g_ema.copy()

            # --- Ergodicity gap filter (optional) ---
            if self.cfg.max_ergodicity_gap is not None:
                excessive_gap = gap > self.cfg.max_ergodicity_gap
                g_smooth[excessive_gap] = np.nan  # exclude these from ranking
                logger.debug(
                    "Ergodicity gap filter: excluding %d assets with gap > %.4f",
                    excessive_gap.sum(),
                    self.cfg.max_ergodicity_gap,
                )

            # --- Rank assets by geometric growth rate ---
            signals = _rank_assets_by_growth(
                g_smooth,
                long_q=self.cfg.long_quantile,
                short_q=self.cfg.short_quantile,
                long_only=self.cfg.long_only,
            )

            # --- Kelly-based position sizing ---
            weights = _compute_kelly_weights(
                signals=signals,
                kelly_fractions=kelly_f,
                kelly_scale=self.cfg.kelly_fraction,
                min_kelly=self.cfg.min_kelly,
                max_kelly=self.cfg.max_kelly,
            )

            # --- Estimate portfolio-level covariance ---
            simple_returns = np.exp(window_lr) - 1.0
            current_cov = _estimate_covariance_shrunk(simple_returns)

            # --- Variance drag monitor ---
            port_var = _portfolio_variance(weights, current_cov)
            port_drag = port_var / 2.0
            port_g = float(weights @ g)  # portfolio geometric growth rate

            port_var_drag[t_idx] = port_drag

            # --- Dynamic leverage adjustment ---
            lev_scale = _variance_drag_leverage_adjustment(
                portfolio_growth_rate=port_g,
                portfolio_variance_drag=port_drag,
                critical_ratio=self.cfg.drag_critical_ratio,
                safety_multiplier=self.cfg.drag_safety_multiplier,
            )
            leverage_scales[t_idx] = lev_scale

            if lev_scale < 1.0:
                logger.debug(
                    "Variance drag adjustment at index %d: scale=%.4f "
                    "(port_g=%.6f, port_drag=%.6f)",
                    t_idx, lev_scale, port_g, port_drag,
                )
                weights *= lev_scale

            # --- Apply constraints ---
            weights = _apply_constraints(
                weights,
                max_position=self.cfg.max_position_weight,
                max_leverage=self.cfg.max_gross_leverage,
            )

            current_weights = weights
            last_rebalance_idx = t_idx
            all_weights[t_idx] = current_weights

        # --- Build output DataFrame ---
        for i, ticker in enumerate(tickers):
            w = all_weights[:, i]
            signals_df[f"{ticker}_signal"] = np.sign(w)
            signals_df[f"{ticker}_weight"] = np.abs(w)

        # --- Store diagnostics ---
        self._weight_history = pd.DataFrame(
            all_weights, index=prices.index, columns=tickers,
        )
        self._growth_rate_history = pd.DataFrame(
            np.column_stack([growth_g, growth_mu]),
            index=prices.index,
            columns=[f"{t}_g" for t in tickers] + [f"{t}_mu" for t in tickers],
        )
        self._ergodicity_gap_history = pd.DataFrame(
            ergo_gap, index=prices.index, columns=tickers,
        )
        self._variance_drag_history = pd.Series(
            port_var_drag, index=prices.index, name="portfolio_variance_drag",
        )
        self._leverage_scale_history = pd.Series(
            leverage_scales, index=prices.index, name="leverage_scale",
        )

        return signals_df

    # ------------------------------------------------------------------
    # Diagnostic methods
    # ------------------------------------------------------------------

    def get_weight_history(self) -> Optional[pd.DataFrame]:
        """Return the full weight history as a DataFrame.

        Returns
        -------
        pd.DataFrame or None
            Columns are asset tickers; rows are dates.
        """
        return self._weight_history

    def get_growth_rate_history(self) -> Optional[pd.DataFrame]:
        """Return per-asset geometric and arithmetic growth rates over time.

        Columns are ``{ticker}_g`` (geometric) and ``{ticker}_mu``
        (arithmetic) for each asset.

        Returns
        -------
        pd.DataFrame or None
        """
        return self._growth_rate_history

    def get_ergodicity_gaps(self) -> Optional[pd.DataFrame]:
        """Return the ergodicity gap Delta = mu - g for each asset over time.

        The gap approximates sigma^2 / 2 (Jensen's inequality).  Large
        gaps indicate assets paying a heavy variance tax.

        Returns
        -------
        pd.DataFrame or None
            Columns are tickers; values are the gap Delta.
        """
        return self._ergodicity_gap_history

    def get_variance_drag(self) -> Optional[pd.Series]:
        """Return the portfolio-level variance drag sigma_p^2 / 2 over time.

        When this exceeds the portfolio geometric growth rate, the
        portfolio is on a wealth-destroying path.

        Returns
        -------
        pd.Series or None
        """
        return self._variance_drag_history

    def get_leverage_scales(self) -> Optional[pd.Series]:
        """Return the leverage scale factor applied by the variance drag monitor.

        Values of 1.0 indicate no adjustment; values < 1.0 indicate
        the monitor reduced leverage to protect against variance drag.

        Returns
        -------
        pd.Series or None
        """
        return self._leverage_scale_history

    def get_current_metrics(self) -> Optional[Dict[str, pd.Series]]:
        """Return the most recently computed growth metrics as a dict of Series.

        Returns
        -------
        dict or None
            Keys: 'geometric_growth', 'arithmetic_mean', 'ergodicity_gap',
            'variance', 'variance_drag', 'kelly_fraction'.
        """
        if self._growth_metrics is None or self._asset_names is None:
            return None
        return {
            key: pd.Series(val, index=self._asset_names, name=key)
            for key, val in self._growth_metrics.items()
        }

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        n_assets = len(self._asset_names) if self._asset_names is not None else 0
        mode = "long-only" if self.cfg.long_only else "long-short"
        return (
            f"ErgodicGrowthStrategy({fitted_tag}, {n_assets} assets, "
            f"{mode}, kelly={self.cfg.kelly_fraction})"
        )
