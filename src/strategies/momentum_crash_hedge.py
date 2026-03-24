"""Momentum strategy with dynamic crash hedging.

Implements cross-sectional momentum (12-1 month) with a crash probability
model that dynamically adjusts exposure based on real-time risk indicators.

Mathematical foundation
-----------------------
Daniel & Moskowitz (2016) show that momentum crashes are predictable:
    - Momentum profits are larger when the market has recently been up.
    - Momentum crashes cluster in high-volatility environments.
    - Cross-sectional dispersion of individual momentums predicts both
      momentum profits and crash risk.

Dynamic position sizing with crash protection:

    w_t = (sigma_target / sigma_hat_{mom,t}) * sign(momentum_signal) * crash_adj_t

Crash probability estimation:
    P(crash_t) = 1 / (1 + exp(-z_t))

    z_t = beta_0 + beta_1 * drawdown_t
                 + beta_2 * dispersion_t
                 + beta_3 * recent_mom_return_t
                 + beta_4 * mom_vol_t

where the betas are calibrated via logistic regression during ``fit()``.

Position adjustment rules given crash_prob p:
    p <= 0.5  :  full momentum exposure
    0.5 < p <= 0.7  :  50 % of momentum exposure
    0.7 < p <= 0.9  :  reverse to short momentum (ride the crash)
    p > 0.9  :  go flat (maximum uncertainty)

Volatility scaling targets 15 % annualised:
    w_t = 0.15 / (sigma_{mom,t} * sqrt(252))
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
# Crash risk indicator computation
# ---------------------------------------------------------------------------

def _compute_individual_momentums(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """12-1 month momentum for each asset: cumulative return from
    *t - lookback* to *t - skip*, skipping the most recent month.

    Returns a DataFrame of momentum scores aligned to the price index.
    """
    # Avoid division by zero from NaN/zero prices
    lagged = prices.shift(skip)
    far_lagged = prices.shift(lookback)
    momentum = (lagged / far_lagged) - 1.0
    return momentum


def _market_drawdown(
    prices: pd.DataFrame,
    window: int = 252,
) -> pd.Series:
    """Drawdown of the equal-weight market proxy from its rolling
    252-day high.  Returns a Series of negative values (0 = at high)."""
    market = prices.mean(axis=1)
    rolling_high = market.rolling(window, min_periods=1).max()
    drawdown = (market - rolling_high) / rolling_high
    return drawdown


def _momentum_dispersion(
    individual_momentums: pd.DataFrame,
) -> pd.Series:
    """Cross-sectional standard deviation of individual asset momentums
    at each point in time."""
    return individual_momentums.std(axis=1)


def _recent_momentum_return(
    momentum_returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Cumulative return of the momentum strategy over the last
    *window* trading days."""
    return momentum_returns.rolling(window, min_periods=1).sum()


def _momentum_volatility(
    momentum_returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Rolling standard deviation of momentum strategy returns,
    annualised."""
    return momentum_returns.rolling(window, min_periods=1).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# Logistic regression (minimal, dependency-free implementation)
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Fit binary logistic regression via gradient descent.

    Parameters
    ----------
    X : (n_samples, n_features) -- should include intercept column.
    y : (n_samples,) binary labels in {0, 1}.
    lr : learning rate.
    max_iter : maximum gradient-descent iterations.
    tol : convergence tolerance on the norm of the gradient.

    Returns
    -------
    beta : (n_features,) coefficient vector.
    """
    n, k = X.shape
    beta = np.zeros(k)

    for _ in range(max_iter):
        p = _sigmoid(X @ beta)
        # Gradient of negative log-likelihood
        grad = X.T @ (p - y) / n
        # L2 regularisation (ridge, lambda=0.01) to stabilise
        grad[1:] += 0.01 * beta[1:]
        beta -= lr * grad
        if np.linalg.norm(grad) < tol:
            break

    return beta


# ---------------------------------------------------------------------------
# Momentum portfolio return construction
# ---------------------------------------------------------------------------

def _construct_momentum_portfolio_returns(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    long_pct: float = 0.3,
    short_pct: float = 0.3,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Build a long-short momentum portfolio and return its daily returns
    plus the weight matrix.

    Steps:
    1. Rank assets by 12-1 month momentum (skip most recent month).
    2. Long top ``long_pct`` fraction, short bottom ``short_pct`` fraction.
    3. Equal weight within each leg.

    Returns
    -------
    mom_returns : pd.Series
        Daily return of the long-short momentum portfolio.
    weights : pd.DataFrame
        Position weights (+1/n_long for longs, -1/n_short for shorts, 0 flat).
    """
    daily_returns = prices.pct_change()
    individual_mom = _compute_individual_momentums(prices, lookback, skip)

    n_assets = prices.shape[1]
    n_long = max(1, int(np.round(n_assets * long_pct)))
    n_short = max(1, int(np.round(n_assets * short_pct)))

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for t in range(lookback, len(prices)):
        row = individual_mom.iloc[t]
        valid = row.dropna()
        if len(valid) < 2:
            continue

        n_l = min(n_long, len(valid) // 2)
        n_s = min(n_short, len(valid) // 2)
        if n_l == 0 or n_s == 0:
            continue

        ranked = valid.sort_values(ascending=False)
        long_assets = ranked.index[:n_l]
        short_assets = ranked.index[-n_s:]

        # Use .loc with the actual index label to avoid chained indexing
        idx_label = weights.index[t]
        weights.loc[idx_label, long_assets] = 1.0 / n_l
        weights.loc[idx_label, short_assets] = -1.0 / n_s

    # Portfolio return: sum of weight_i * return_i
    mom_returns = (weights.shift(1) * daily_returns).sum(axis=1)
    return mom_returns, weights


# ---------------------------------------------------------------------------
# Crash label construction for logistic regression training
# ---------------------------------------------------------------------------

def _label_crash_periods(
    mom_returns: pd.Series,
    threshold_quantile: float = 0.05,
    window: int = 21,
) -> pd.Series:
    """Label periods as crash (1) or non-crash (0).

    A date is labelled as a crash precursor if the *forward* 21-day
    cumulative momentum return falls below the ``threshold_quantile``
    of its historical distribution.
    """
    fwd_cum = mom_returns.shift(-window).rolling(window, min_periods=1).sum()
    # Use expanding quantile so it is purely backward-looking
    expanding_thresh = fwd_cum.expanding(min_periods=252).quantile(threshold_quantile)
    labels = (fwd_cum < expanding_thresh).astype(float)
    return labels


# ===========================================================================
# Strategy class
# ===========================================================================

class MomentumCrashHedgeStrategy(Strategy):
    """Cross-sectional momentum with dynamic crash hedging.

    Parameters
    ----------
    momentum_lookback : int
        Number of trading days for momentum calculation (default 252,
        approximately 12 months).
    skip_period : int
        Number of recent days to skip in momentum calculation (default 21,
        approximately 1 month) -- Jegadeesh & Titman short-term reversal.
    long_pct : float
        Fraction of assets in the long leg (default 0.3 = top 30 %).
    short_pct : float
        Fraction of assets in the short leg (default 0.3 = bottom 30 %).
    vol_target : float
        Annualised volatility target for position sizing (default 0.15).
    vol_window : int
        Rolling window for estimating momentum strategy volatility
        (default 63, ~3 months).
    crash_lookback : int
        Rolling window for recent-momentum-return indicator (default 21).
    drawdown_window : int
        Window for computing the rolling high used in drawdown (default 252).
    crash_threshold_quantile : float
        Quantile of forward momentum returns below which a period is
        labelled as crash for logistic regression training (default 0.05).
    """

    def __init__(
        self,
        momentum_lookback: int = 252,
        skip_period: int = 21,
        long_pct: float = 0.3,
        short_pct: float = 0.3,
        vol_target: float = 0.15,
        vol_window: int = 63,
        crash_lookback: int = 21,
        drawdown_window: int = 252,
        crash_threshold_quantile: float = 0.05,
    ) -> None:
        super().__init__(
            name="MomentumCrashHedge",
            description=(
                "Cross-sectional momentum (12-1 month) with dynamic crash "
                "hedging via logistic crash probability model and volatility "
                "targeting."
            ),
        )
        self.momentum_lookback = momentum_lookback
        self.skip_period = skip_period
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.vol_target = vol_target
        self.vol_window = vol_window
        self.crash_lookback = crash_lookback
        self.drawdown_window = drawdown_window
        self.crash_threshold_quantile = crash_threshold_quantile

        # Learned parameters (set during fit)
        self._crash_betas: Optional[np.ndarray] = None
        self._indicator_means: Optional[np.ndarray] = None
        self._indicator_stds: Optional[np.ndarray] = None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _build_crash_indicators(
        self,
        prices: pd.DataFrame,
        mom_returns: pd.Series,
        individual_mom: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assemble the four crash risk indicator series into a DataFrame.

        Indicators (all computed from price data alone):
        1. Market drawdown from rolling high.
        2. Cross-sectional momentum dispersion.
        3. Recent cumulative momentum return (last ``crash_lookback`` days).
        4. Rolling volatility of momentum strategy returns.

        Returns a DataFrame with columns:
            drawdown, dispersion, recent_mom, mom_vol
        """
        drawdown = _market_drawdown(prices, window=self.drawdown_window)
        dispersion = _momentum_dispersion(individual_mom)
        recent_mom = _recent_momentum_return(mom_returns, window=self.crash_lookback)
        mom_vol = _momentum_volatility(mom_returns, window=self.crash_lookback)

        indicators = pd.DataFrame({
            "drawdown": drawdown,
            "dispersion": dispersion,
            "recent_mom": recent_mom,
            "mom_vol": mom_vol,
        }, index=prices.index)

        return indicators

    def _standardise_indicators(
        self,
        indicators: pd.DataFrame,
        fit: bool = False,
    ) -> np.ndarray:
        """Z-score the indicator matrix.

        During ``fit=True``, compute and store means/stds.
        During inference (``fit=False``), use stored statistics.

        Returns
        -------
        X : (n_samples, 5) array including intercept column.
        """
        values = indicators.values.astype(np.float64)

        if fit:
            self._indicator_means = np.nanmean(values, axis=0)
            self._indicator_stds = np.nanstd(values, axis=0)
            # Guard against zero std
            self._indicator_stds[self._indicator_stds < 1e-12] = 1.0

        means = self._indicator_means
        stds = self._indicator_stds

        z = (values - means) / stds
        # Replace any remaining NaN with 0 (neutral)
        z = np.nan_to_num(z, nan=0.0)

        # Prepend intercept column
        n = z.shape[0]
        X = np.column_stack([np.ones(n), z])
        return X

    def _crash_probability(self, X: np.ndarray) -> np.ndarray:
        """Predict crash probability from standardised indicator matrix.

        Parameters
        ----------
        X : (n_samples, 5) array with intercept.

        Returns
        -------
        prob : (n_samples,) crash probabilities in [0, 1].
        """
        return _sigmoid(X @ self._crash_betas)

    def _crash_adjustment(self, crash_prob: np.ndarray) -> np.ndarray:
        """Map crash probability to a position adjustment factor.

        Rules (from Daniel & Moskowitz-inspired dynamic hedging):
            p <= 0.5  ->  +1.0  (full momentum exposure)
            0.5 < p <= 0.7  ->  +0.5  (reduce to half)
            0.7 < p <= 0.9  ->  -1.0  (reverse: short momentum)
            p > 0.9  ->   0.0  (go flat)

        Returns
        -------
        adj : (n_samples,) adjustment multiplier.
        """
        adj = np.ones_like(crash_prob)
        adj = np.where(crash_prob > 0.9, 0.0, adj)
        adj = np.where((crash_prob > 0.7) & (crash_prob <= 0.9), -1.0, adj)
        adj = np.where((crash_prob > 0.5) & (crash_prob <= 0.7), 0.5, adj)
        return adj

    # -----------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "MomentumCrashHedgeStrategy":
        """Calibrate crash probability model on historical data.

        Steps:
        1. Construct cross-sectional momentum portfolio returns.
        2. Label crash / non-crash periods using forward returns.
        3. Build crash risk indicators (drawdown, dispersion, recent
           momentum return, momentum volatility).
        4. Fit logistic regression: P(crash) ~ indicators.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data.  Columns are tickers, index is
            DatetimeIndex.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        # 1. Momentum portfolio returns
        mom_returns, _ = _construct_momentum_portfolio_returns(
            prices,
            lookback=self.momentum_lookback,
            skip=self.skip_period,
            long_pct=self.long_pct,
            short_pct=self.short_pct,
        )

        # 2. Individual momentums for dispersion
        individual_mom = _compute_individual_momentums(
            prices, self.momentum_lookback, self.skip_period
        )

        # 3. Crash indicators
        indicators = self._build_crash_indicators(
            prices, mom_returns, individual_mom
        )

        # 4. Crash labels (forward-looking, for training only)
        crash_labels = _label_crash_periods(
            mom_returns,
            threshold_quantile=self.crash_threshold_quantile,
            window=self.crash_lookback,
        )

        # Align and drop NaN rows
        combined = pd.concat([indicators, crash_labels.rename("label")], axis=1)
        combined = combined.dropna()

        if len(combined) < 100:
            warnings.warn(
                f"Only {len(combined)} valid samples for crash model fitting; "
                "results may be unreliable. Consider providing more history.",
                stacklevel=2,
            )

        if len(combined) == 0:
            # Fallback: use neutral betas (all zero except small intercept)
            self._crash_betas = np.zeros(5)
            self._indicator_means = np.zeros(4)
            self._indicator_stds = np.ones(4)
            self._fitted = True
            logger.warning(
                "No valid training samples; crash model initialised with "
                "neutral coefficients."
            )
            return self

        X = self._standardise_indicators(
            combined[["drawdown", "dispersion", "recent_mom", "mom_vol"]],
            fit=True,
        )
        y = combined["label"].values.astype(np.float64)

        # Fit logistic regression
        self._crash_betas = _fit_logistic(X, y)

        self.parameters = {
            "crash_betas": self._crash_betas.tolist(),
            "indicator_means": self._indicator_means.tolist(),
            "indicator_stds": self._indicator_stds.tolist(),
        }

        logger.info(
            "Crash model fitted: betas=%s  (intercept=%.4f)",
            np.round(self._crash_betas[1:], 4),
            self._crash_betas[0],
        )

        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate momentum signals with dynamic crash hedging.

        For each date *t*:
        1. Rank assets by 12-1 month momentum.
        2. Assign long (top 30 %), short (bottom 30 %), flat (middle).
        3. Compute crash probability from risk indicators.
        4. Apply crash adjustment to position direction.
        5. Scale by volatility targeting (15 % annualised).

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (columns = tickers, index = DatetimeIndex).

        Returns
        -------
        pd.DataFrame
            Columns: ``{ticker}_signal`` and ``{ticker}_weight`` for each
            ticker, plus ``crash_prob`` diagnostic column.
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        # -- Momentum portfolio construction ----------------------------
        mom_returns, raw_weights = _construct_momentum_portfolio_returns(
            prices,
            lookback=self.momentum_lookback,
            skip=self.skip_period,
            long_pct=self.long_pct,
            short_pct=self.short_pct,
        )

        individual_mom = _compute_individual_momentums(
            prices, self.momentum_lookback, self.skip_period
        )

        # -- Crash indicators & probability -----------------------------
        indicators = self._build_crash_indicators(
            prices, mom_returns, individual_mom
        )
        X = self._standardise_indicators(indicators, fit=False)
        crash_prob = self._crash_probability(X)
        crash_adj = self._crash_adjustment(crash_prob)

        # -- Volatility scaling -----------------------------------------
        # Rolling vol of the momentum strategy returns
        rolling_vol = mom_returns.rolling(
            self.vol_window, min_periods=max(21, self.vol_window // 2)
        ).std() * np.sqrt(252)
        # Floor vol to avoid extreme leverage
        rolling_vol = rolling_vol.clip(lower=0.01)
        vol_scale = self.vol_target / rolling_vol
        # Cap leverage at 3x to prevent extreme positions
        vol_scale = vol_scale.clip(upper=3.0)

        # -- Assemble output signals ------------------------------------
        result = pd.DataFrame(index=prices.index)

        for col in prices.columns:
            # Signal: +1 (long), -1 (short), 0 (flat) from raw_weights
            signal = np.sign(raw_weights[col].values)
            # Apply crash adjustment (may flip sign or zero out)
            adjusted_signal = signal * crash_adj

            # Weight: vol-scaled absolute weight per asset
            base_weight = raw_weights[col].abs().values
            weight = base_weight * vol_scale.values
            # Clip individual weight to [0, 1]
            weight = np.clip(weight, 0.0, 1.0)

            result[f"{col}_signal"] = adjusted_signal
            result[f"{col}_weight"] = weight

        # Diagnostic columns
        result["crash_prob"] = crash_prob

        return result
