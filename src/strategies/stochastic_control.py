"""Portfolio allocation strategy based on stochastic optimal control (HJB).

Mathematical foundation
-----------------------
Merton's consumption-investment problem (extended):

    Maximize  E[ integral_0^T U(c_t) dt + B(W_T) ]

where U(x) = x^(1-gamma) / (1-gamma) is CRRA utility with risk aversion
parameter gamma.

Hamilton-Jacobi-Bellman equation:

    0 = max_w { V_t + (r + w'(mu - r)) W V_W + 1/2 w' Sigma w W^2 V_WW }

Optimal Merton fraction (closed-form solution):

    w* = (1/gamma) * Sigma^{-1} * (mu - r)

With time-varying parameters (regime-dependent estimation):

    w*_t = (1/gamma) * Sigma_t^{-1} * (mu_t - r_t)

Parameter estimation
--------------------
- Expected returns mu_t: shrinkage estimator (Ledoit-Wolf on means toward
  the grand mean across assets).
- Covariance Sigma_t: Ledoit-Wolf shrinkage toward constant-correlation
  target, via sklearn.covariance.LedoitWolf.
- Rolling window: 252 trading days.
- Risk-free rate: 3-month Treasury proxy or default 0.04 annualised.

Bayesian shrinkage (Black-Litterman inspired)
----------------------------------------------
- Prior: market-cap weights imply equilibrium returns
      pi = gamma * Sigma * w_mkt
- Without explicit investor views the posterior collapses to shrinkage
  toward the equilibrium:
      mu_BL = [(tau * Sigma)^{-1} + Sigma_sample^{-1}]^{-1}
              * [(tau * Sigma)^{-1} pi + Sigma_sample^{-1} mu_sample]
- tau controls confidence in the prior (default 0.05).

Dynamic rebalancing
-------------------
- Rebalance when max absolute weight drift > 5% from target.
- Or weekly (every 5 trading days), whichever comes first.
- Transaction cost penalty: proportional cost subtracted from excess
  returns in weight optimisation.

Risk controls (conservative variant)
-------------------------------------
- Drawdown circuit breaker: when realised drawdown exceeds a threshold,
  positions are scaled to a fraction (e.g. 50%) for a cooldown period.
- Per-asset weight cap prevents concentration risk.
- Higher gamma and lower max_leverage reduce tail risk.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StochasticControlConfig:
    """All tuneable parameters for the stochastic optimal control strategy."""

    # Risk aversion (CRRA gamma)
    gamma: float = 2.0

    # Rolling estimation window (trading days)
    rolling_window: int = 252

    # Risk-free rate (annualised, decimal)
    risk_free_rate: float = 0.04

    # Ridge regularisation added to covariance for numerical stability
    ridge_lambda: float = 1e-4

    # Leverage constraint: L1 norm of weights <= max_leverage
    max_leverage: float = 2.0

    # Black-Litterman prior confidence (tau)
    tau: float = 0.05

    # Rebalancing triggers
    max_drift_pct: float = 0.05          # max absolute weight drift
    rebalance_freq_days: int = 5         # weekly fallback

    # Transaction cost (proportional, one-way, in decimal)
    transaction_cost_bps: float = 10.0   # basis points

    # Mean shrinkage intensity toward grand mean (0 = no shrinkage, 1 = full)
    mean_shrinkage_intensity: float = 0.5

    # Minimum history required before generating signals
    min_history: int = 126               # ~6 months

    # Per-asset weight cap (absolute value); prevents concentration risk
    max_weight_per_asset: float = 1.0

    # Drawdown circuit breaker: if portfolio drawdown exceeds this level,
    # scale positions to ``drawdown_reduction_factor`` for ``drawdown_cooldown_days``.
    # Set to None to disable.
    drawdown_threshold: Optional[float] = None   # e.g. 0.20 = 20%
    drawdown_reduction_factor: float = 0.5       # scale positions to 50%
    drawdown_cooldown_days: int = 21             # hold reduced size for ~1 month


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------

class StochasticControlStrategy(Strategy):
    """Portfolio allocation via Merton optimal fractions with Bayesian
    shrinkage, Ledoit-Wolf covariance estimation, and dynamic rebalancing.

    This implements the closed-form HJB solution for CRRA utility under
    time-varying parameter estimates, combined with Black-Litterman-style
    shrinkage toward equilibrium returns.

    Parameters
    ----------
    config : StochasticControlConfig, optional
        Strategy configuration.  Uses defaults if not provided.
    market_cap_weights : dict[str, float], optional
        Prior market-capitalisation weights for Black-Litterman equilibrium.
        If ``None``, equal weights are used as the prior.
    """

    def __init__(
        self,
        config: Optional[StochasticControlConfig] = None,
        market_cap_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(
            name="StochasticControl",
            description=(
                "HJB-optimal Merton fractions with Bayesian shrinkage "
                "and Ledoit-Wolf covariance estimation"
            ),
        )
        self.cfg = config or StochasticControlConfig()
        self._market_cap_weights = market_cap_weights

        # Populated during fit
        self._mu_bl: Optional[np.ndarray] = None        # posterior expected returns
        self._sigma_shrunk: Optional[np.ndarray] = None  # shrunk covariance matrix
        self._tickers: list[str] = []
        self._current_weights: Optional[np.ndarray] = None
        self._last_rebalance_idx: int = -1

    # ------------------------------------------------------------------
    # Parameter estimation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_covariance_lw(returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance matrix using Ledoit-Wolf shrinkage.

        The Ledoit-Wolf estimator shrinks the sample covariance toward
        a structured target (constant-correlation model in sklearn's
        implementation), yielding a well-conditioned positive-definite
        matrix even when p ~ n.

        Parameters
        ----------
        returns : pd.DataFrame
            T x N matrix of asset returns.

        Returns
        -------
        np.ndarray
            N x N shrunk covariance matrix (daily scale).
        """
        lw = LedoitWolf(assume_centered=False)
        lw.fit(returns.values)
        return lw.covariance_

    @staticmethod
    def _shrink_means(
        sample_means: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Shrink sample means toward the grand (cross-sectional) mean.

        This is a James-Stein-style shrinkage: each asset's estimated
        mean return is pulled toward the average across all assets,
        reducing estimation error at the cost of bias.

        Parameters
        ----------
        sample_means : np.ndarray
            Vector of per-asset sample mean returns.
        intensity : float
            Shrinkage intensity in [0, 1].  0 = pure sample mean,
            1 = grand mean for all assets.

        Returns
        -------
        np.ndarray
            Shrunk mean return vector.
        """
        grand_mean = np.mean(sample_means)
        return (1.0 - intensity) * sample_means + intensity * grand_mean

    def _compute_equilibrium_returns(
        self,
        sigma: np.ndarray,
        w_mkt: np.ndarray,
    ) -> np.ndarray:
        """Compute implied equilibrium returns from market-cap weights.

        The Black-Litterman reverse-optimisation formula:
            pi = gamma * Sigma * w_mkt

        Parameters
        ----------
        sigma : np.ndarray
            N x N covariance matrix.
        w_mkt : np.ndarray
            N-vector of market-capitalisation weights (sum to 1).

        Returns
        -------
        np.ndarray
            N-vector of implied equilibrium excess returns.
        """
        return self.cfg.gamma * sigma @ w_mkt

    def _compute_bl_posterior(
        self,
        sigma: np.ndarray,
        pi: np.ndarray,
        mu_sample: np.ndarray,
    ) -> np.ndarray:
        """Compute Black-Litterman posterior mean (no explicit views).

        Without investor views (P, Q, Omega), the posterior simplifies to
        a weighted combination of the equilibrium prior and sample mean:

            mu_BL = [(tau Sigma)^{-1} + Sigma^{-1}]^{-1}
                    * [(tau Sigma)^{-1} pi + Sigma^{-1} mu_sample]

        This effectively shrinks the sample estimate toward equilibrium,
        with tau controlling the relative confidence.

        Parameters
        ----------
        sigma : np.ndarray
            N x N covariance matrix.
        pi : np.ndarray
            Implied equilibrium excess returns.
        mu_sample : np.ndarray
            Sample mean excess returns.

        Returns
        -------
        np.ndarray
            Posterior mean return vector.
        """
        tau = self.cfg.tau
        n = len(pi)

        tau_sigma = tau * sigma
        tau_sigma_inv = np.linalg.inv(
            tau_sigma + self.cfg.ridge_lambda * np.eye(n)
        )
        sigma_inv = np.linalg.inv(
            sigma + self.cfg.ridge_lambda * np.eye(n)
        )

        # Posterior precision and mean
        precision = tau_sigma_inv + sigma_inv
        posterior_cov = np.linalg.inv(precision)
        mu_bl = posterior_cov @ (tau_sigma_inv @ pi + sigma_inv @ mu_sample)

        return mu_bl

    def _compute_merton_weights(
        self,
        mu_excess: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        """Compute optimal Merton portfolio fractions.

        Closed-form HJB solution for CRRA utility:
            w* = (1/gamma) * Sigma^{-1} * (mu - r)

        The covariance inverse is regularised with a ridge penalty for
        numerical stability.

        Parameters
        ----------
        mu_excess : np.ndarray
            N-vector of expected excess returns (mu - r).
        sigma : np.ndarray
            N x N covariance matrix.

        Returns
        -------
        np.ndarray
            N-vector of optimal portfolio weights.
        """
        n = len(mu_excess)
        sigma_reg = sigma + self.cfg.ridge_lambda * np.eye(n)
        sigma_inv = np.linalg.inv(sigma_reg)
        raw_weights = (1.0 / self.cfg.gamma) * sigma_inv @ mu_excess
        return raw_weights

    @staticmethod
    def _apply_leverage_constraint(
        weights: np.ndarray,
        max_leverage: float,
    ) -> np.ndarray:
        """Enforce L1 leverage constraint: ||w||_1 <= max_leverage.

        If the L1 norm exceeds the limit, all weights are scaled down
        proportionally.

        Parameters
        ----------
        weights : np.ndarray
            Raw portfolio weights.
        max_leverage : float
            Maximum allowed L1 norm.

        Returns
        -------
        np.ndarray
            Constrained weights.
        """
        l1_norm = np.sum(np.abs(weights))
        if l1_norm > max_leverage:
            weights = weights * (max_leverage / l1_norm)
        return weights

    @staticmethod
    def _apply_per_asset_cap(
        weights: np.ndarray,
        max_weight: float,
    ) -> np.ndarray:
        """Clip each weight to [-max_weight, +max_weight].

        This prevents any single asset from dominating the portfolio,
        which is a primary driver of drawdown in concentrated portfolios.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights.
        max_weight : float
            Maximum absolute weight per asset.

        Returns
        -------
        np.ndarray
            Clipped weights.
        """
        return np.clip(weights, -max_weight, max_weight)

    def _adjust_for_transaction_costs(
        self,
        target_weights: np.ndarray,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """Adjust target weights for proportional transaction costs.

        The idea is to penalise trades: if the expected gain from
        rebalancing to the new target does not exceed the transaction
        cost of the trade, keep the current weight for that asset.

        For each asset i:
            trade_i = |w_target_i - w_current_i|
            cost_i  = trade_i * tc
        We only rebalance asset i if:
            |w_target_i - w_current_i| > 2 * tc
        (the factor 2 accounts for round-trip cost amortisation).

        Parameters
        ----------
        target_weights : np.ndarray
            Desired optimal weights.
        current_weights : np.ndarray
            Current portfolio weights.

        Returns
        -------
        np.ndarray
            Adjusted target weights (some assets may keep current weight).
        """
        tc = self.cfg.transaction_cost_bps / 10_000.0
        threshold = 2.0 * tc

        adjusted = target_weights.copy()
        trades = np.abs(target_weights - current_weights)

        # For small trades, keep the current weight
        small_trade_mask = trades < threshold
        adjusted[small_trade_mask] = current_weights[small_trade_mask]

        return adjusted

    def _should_rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        days_since_last: int,
    ) -> bool:
        """Determine whether to rebalance based on drift and time.

        Rebalance if:
        1. Maximum absolute weight drift exceeds ``max_drift_pct``, OR
        2. At least ``rebalance_freq_days`` trading days have elapsed.

        Parameters
        ----------
        current_weights : np.ndarray
            Current portfolio weights.
        target_weights : np.ndarray
            Newly computed optimal weights.
        days_since_last : int
            Trading days since last rebalance.

        Returns
        -------
        bool
        """
        max_drift = np.max(np.abs(current_weights - target_weights))
        if max_drift > self.cfg.max_drift_pct:
            return True
        if days_since_last >= self.cfg.rebalance_freq_days:
            return True
        return False

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "StochasticControlStrategy":
        """Calibrate strategy parameters on historical price data.

        Estimates the covariance matrix (Ledoit-Wolf), expected returns
        (shrinkage + Black-Litterman), and computes the initial Merton
        optimal weights.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical adjusted-close prices.  Columns are asset tickers,
            index is a DatetimeIndex.

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        self._tickers = list(prices.columns)
        n_assets = len(self._tickers)

        # Daily log-returns
        returns = np.log(prices / prices.shift(1)).dropna()

        if len(returns) < self.cfg.min_history:
            raise ValueError(
                f"Insufficient history: {len(returns)} rows < "
                f"min_history {self.cfg.min_history}."
            )

        # Use the most recent rolling_window for estimation
        window = min(self.cfg.rolling_window, len(returns))
        recent_returns = returns.iloc[-window:]

        # -- Covariance estimation (Ledoit-Wolf) --
        sigma_daily = self._estimate_covariance_lw(recent_returns)
        # Annualise: Sigma_annual = 252 * Sigma_daily
        sigma_annual = 252.0 * sigma_daily

        # -- Mean estimation (shrinkage toward grand mean) --
        mu_sample_daily = recent_returns.mean().values
        mu_sample_annual = 252.0 * mu_sample_daily  # annualise
        mu_shrunk = self._shrink_means(
            mu_sample_annual,
            self.cfg.mean_shrinkage_intensity,
        )

        # -- Market-cap weights for BL prior --
        if self._market_cap_weights is not None:
            w_mkt = np.array([
                self._market_cap_weights.get(t, 1.0 / n_assets)
                for t in self._tickers
            ])
            w_mkt = w_mkt / w_mkt.sum()  # normalise
        else:
            w_mkt = np.full(n_assets, 1.0 / n_assets)

        # -- Equilibrium returns (reverse optimisation) --
        pi = self._compute_equilibrium_returns(sigma_annual, w_mkt)

        # -- BL posterior (without views) --
        mu_excess_sample = mu_shrunk - self.cfg.risk_free_rate
        mu_bl = self._compute_bl_posterior(
            sigma_annual,
            pi,
            mu_excess_sample,
        )
        self._mu_bl = mu_bl
        self._sigma_shrunk = sigma_annual

        # -- Initial optimal weights --
        raw_weights = self._compute_merton_weights(mu_bl, sigma_annual)
        raw_weights = self._apply_per_asset_cap(
            raw_weights, self.cfg.max_weight_per_asset,
        )
        constrained = self._apply_leverage_constraint(
            raw_weights, self.cfg.max_leverage,
        )
        self._current_weights = constrained

        # Store fitted parameters for inspection
        self.parameters = {
            "mu_bl": mu_bl.tolist(),
            "sigma_annual_diag": np.diag(sigma_annual).tolist(),
            "initial_weights": constrained.tolist(),
            "equilibrium_returns": pi.tolist(),
            "tickers": self._tickers,
        }

        self._fitted = True
        logger.info(
            "StochasticControlStrategy fitted on %d assets, %d observations.",
            n_assets,
            len(recent_returns),
        )
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate allocation signals from price data.

        For each date in the price DataFrame (after sufficient history),
        the strategy:
        1. Estimates time-varying mu_t and Sigma_t on a rolling window.
        2. Computes optimal Merton weights with BL shrinkage.
        3. Enforces per-asset weight cap and leverage constraint.
        4. Decides whether to rebalance (drift or time trigger).
        5. Adjusts for transaction costs.
        6. Applies drawdown circuit breaker (if configured).

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (adjusted close).  Columns must match tickers
            from ``fit()``.

        Returns
        -------
        pd.DataFrame
            Per-asset columns with ``{ticker}_signal`` (direction: +1 long,
            -1 short, 0 flat) and ``{ticker}_weight`` (position size in
            [0, 1]).
        """
        if not self._fitted:
            self.fit(prices)

        self.validate_prices(prices)

        tickers = self._tickers
        n_assets = len(tickers)

        # Ensure price columns match fitted tickers
        available = [t for t in tickers if t in prices.columns]
        if not available:
            raise ValueError(
                "No fitted tickers found in prices columns. "
                f"Expected: {tickers}, got: {list(prices.columns)}."
            )

        # Work with the subset of tickers present in prices
        price_data = prices[available]
        returns = np.log(price_data / price_data.shift(1))

        # Output DataFrames
        signals_out = pd.DataFrame(index=prices.index)
        for t in available:
            signals_out[f"{t}_signal"] = 0.0
            signals_out[f"{t}_weight"] = 0.0

        # Track current portfolio weights for rebalancing logic
        current_weights = np.zeros(len(available))
        last_rebalance_idx = -self.cfg.rebalance_freq_days  # force first rebalance
        window = self.cfg.rolling_window

        # Map available tickers to their index in the available list
        ticker_to_idx = {t: i for i, t in enumerate(available)}

        # -- Drawdown circuit breaker state --
        # Track a synthetic equity curve from portfolio returns to detect
        # drawdowns *within* signal generation (not relying on external equity).
        equity = 1.0
        peak_equity = 1.0
        drawdown_cooldown_remaining = 0  # days left in reduced-position mode

        for i in range(window, len(prices)):
            # Rolling window of returns (drop NaN rows)
            ret_window = returns.iloc[max(0, i - window + 1): i + 1].dropna()

            if len(ret_window) < self.cfg.min_history:
                continue

            # -- Update synthetic equity for drawdown tracking --
            if i > window and np.any(current_weights != 0):
                # Portfolio return for this bar: sum of weight * asset return
                day_returns = returns.iloc[i].values
                if not np.any(np.isnan(day_returns)):
                    port_return = np.dot(current_weights, day_returns)
                    equity *= (1.0 + port_return)
                    if equity > peak_equity:
                        peak_equity = equity

            # -- Check drawdown circuit breaker --
            current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            if (
                self.cfg.drawdown_threshold is not None
                and current_drawdown > self.cfg.drawdown_threshold
                and drawdown_cooldown_remaining <= 0
            ):
                drawdown_cooldown_remaining = self.cfg.drawdown_cooldown_days
                logger.info(
                    "Drawdown circuit breaker triggered: %.1f%% drawdown at "
                    "index %d. Reducing positions for %d days.",
                    current_drawdown * 100.0,
                    i,
                    self.cfg.drawdown_cooldown_days,
                )

            if drawdown_cooldown_remaining > 0:
                drawdown_cooldown_remaining -= 1

            # -- Time-varying covariance (Ledoit-Wolf) --
            try:
                sigma_daily = self._estimate_covariance_lw(ret_window)
            except Exception:
                # Fall back to sample covariance with regularisation
                sigma_daily = ret_window.cov().values
                sigma_daily += self.cfg.ridge_lambda * np.eye(len(available))

            sigma_annual = 252.0 * sigma_daily

            # -- Time-varying mean (shrinkage) --
            mu_sample_daily = ret_window.mean().values
            mu_sample_annual = 252.0 * mu_sample_daily
            mu_shrunk = self._shrink_means(
                mu_sample_annual,
                self.cfg.mean_shrinkage_intensity,
            )

            # -- Market-cap prior --
            if self._market_cap_weights is not None:
                w_mkt = np.array([
                    self._market_cap_weights.get(t, 1.0 / len(available))
                    for t in available
                ])
                w_mkt = w_mkt / w_mkt.sum()
            else:
                w_mkt = np.full(len(available), 1.0 / len(available))

            # -- Equilibrium returns --
            pi = self._compute_equilibrium_returns(sigma_annual, w_mkt)

            # -- BL posterior --
            mu_excess = mu_shrunk - self.cfg.risk_free_rate
            mu_bl = self._compute_bl_posterior(sigma_annual, pi, mu_excess)

            # -- Optimal Merton weights --
            raw_weights = self._compute_merton_weights(mu_bl, sigma_annual)

            # -- Per-asset weight cap (before leverage constraint) --
            raw_weights = self._apply_per_asset_cap(
                raw_weights, self.cfg.max_weight_per_asset,
            )
            target_weights = self._apply_leverage_constraint(
                raw_weights, self.cfg.max_leverage,
            )

            # -- Apply drawdown reduction if in cooldown --
            if drawdown_cooldown_remaining > 0:
                target_weights = target_weights * self.cfg.drawdown_reduction_factor

            # -- Rebalancing decision --
            days_since = i - last_rebalance_idx
            if self._should_rebalance(current_weights, target_weights, days_since):
                # Adjust for transaction costs
                adjusted = self._adjust_for_transaction_costs(
                    target_weights, current_weights,
                )
                # Re-apply leverage constraint after TC adjustment
                adjusted = self._apply_leverage_constraint(
                    adjusted, self.cfg.max_leverage,
                )
                current_weights = adjusted
                last_rebalance_idx = i

            # -- Write signals --
            dt = prices.index[i]
            for t in available:
                idx = ticker_to_idx[t]
                w = current_weights[idx]
                signals_out.at[dt, f"{t}_signal"] = float(np.sign(w))
                signals_out.at[dt, f"{t}_weight"] = float(np.abs(w))

        return signals_out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_current_weights(self) -> Optional[Dict[str, float]]:
        """Return the most recent portfolio weights as a dict.

        Returns ``None`` if the strategy has not been fitted.
        """
        if self._current_weights is None:
            return None
        return dict(zip(self._tickers, self._current_weights.tolist()))

    def get_posterior_returns(self) -> Optional[Dict[str, float]]:
        """Return the Black-Litterman posterior expected returns.

        Returns ``None`` if the strategy has not been fitted.
        """
        if self._mu_bl is None:
            return None
        return dict(zip(self._tickers, self._mu_bl.tolist()))

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        return (
            f"StochasticControlStrategy(gamma={self.cfg.gamma}, "
            f"tau={self.cfg.tau}, {fitted_tag})"
        )


# ---------------------------------------------------------------------------
# Conservative variant with drawdown control
# ---------------------------------------------------------------------------

# Pre-built config targeting max drawdown < 25%, Sharpe > 1.3, PnL > 45%.
#
# Key changes vs. the default:
#   gamma:  2.0 -> 3.0    (higher risk aversion shrinks Merton fractions)
#   max_leverage: 2.0 -> 1.5  (tighter gross-exposure cap)
#   tau:  0.05 -> 0.025   (stronger BL prior: shrink more toward equilibrium;
#                           smaller tau => larger (tau*Sigma)^{-1} => heavier
#                           weight on equilibrium returns pi)
#   mean_shrinkage_intensity: 0.5 -> 0.65  (pull sample means harder toward
#                                            the grand mean, reducing outlier
#                                            alpha estimates)
#   max_weight_per_asset: 1.0 -> 0.40  (no single asset > 40% of capital)
#   drawdown_threshold: None -> 0.20  (circuit breaker at 20% drawdown)
#   drawdown_reduction_factor: 0.50   (halve positions during cooldown)
#   drawdown_cooldown_days: 21        (hold reduced size for ~1 month)
#   rebalance_freq_days: 5 -> 10      (less frequent rebalancing dampens
#                                       turnover-driven whipsaw)

CONSERVATIVE_CONFIG = StochasticControlConfig(
    gamma=3.0,
    rolling_window=252,
    risk_free_rate=0.04,
    ridge_lambda=1e-4,
    max_leverage=1.5,
    tau=0.025,
    max_drift_pct=0.05,
    rebalance_freq_days=10,
    transaction_cost_bps=10.0,
    mean_shrinkage_intensity=0.65,
    min_history=126,
    max_weight_per_asset=0.40,
    drawdown_threshold=0.20,
    drawdown_reduction_factor=0.50,
    drawdown_cooldown_days=21,
)


class ConservativeStochasticControlStrategy(StochasticControlStrategy):
    """Drawdown-controlled variant of the stochastic optimal control strategy.

    Targets:
    - Maximum drawdown < 25%  (via 20% circuit breaker + 1.5x leverage cap)
    - PnL > 45%  (still runs the full HJB optimisation, just with more
      conservative parameterisation)
    - Sharpe > 1.3  (lower variance from reduced leverage and drawdown
      control should improve risk-adjusted returns)

    Differences from the base ``StochasticControlStrategy``:

    1. **Higher risk aversion** (gamma = 3 vs 2): the Merton fraction
       ``w* = (1/gamma) Sigma^{-1} (mu - r)`` is 33% smaller, reducing
       exposure to estimation error in expected returns.

    2. **Tighter leverage** (1.5x vs 2.0x): caps gross exposure, the single
       most important lever for drawdown control.

    3. **Stronger BL shrinkage** (tau = 0.025 vs 0.05): the posterior mean
       is pulled harder toward the equilibrium prior, suppressing extreme
       positions driven by noisy sample means.

    4. **Per-asset cap** (40%): prevents concentrated bets that cause large
       drawdowns when a single asset reverses.

    5. **Drawdown circuit breaker** (20% threshold, 50% reduction, 21-day
       cooldown): when the strategy's synthetic equity drawdown exceeds 20%,
       all positions are halved for 21 trading days, giving the model time
       to re-calibrate without compounding losses.

    6. **Stronger mean shrinkage** (0.65 vs 0.50): pulls sample return
       estimates closer to the cross-sectional average, reducing the
       influence of in-sample outlier returns.
    """

    def __init__(
        self,
        config: Optional[StochasticControlConfig] = None,
        market_cap_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        # Allow overriding but default to the conservative config
        cfg = config or CONSERVATIVE_CONFIG
        super().__init__(config=cfg, market_cap_weights=market_cap_weights)
        self.name = "ConservativeStochasticControl"
        self.description = (
            "Drawdown-controlled HJB-optimal Merton fractions with "
            "circuit breaker, tighter leverage, and stronger BL shrinkage"
        )

    def __repr__(self) -> str:
        fitted_tag = "fitted" if self._fitted else "unfitted"
        return (
            f"ConservativeStochasticControlStrategy(gamma={self.cfg.gamma}, "
            f"tau={self.cfg.tau}, max_lev={self.cfg.max_leverage}, "
            f"dd_thresh={self.cfg.drawdown_threshold}, {fitted_tag})"
        )
