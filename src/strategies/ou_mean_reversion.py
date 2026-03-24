"""
Ornstein-Uhlenbeck Mean-Reversion Strategy
===========================================

Statistical arbitrage strategy grounded in the OU stochastic process:

    dX_t = theta * (mu - X_t) dt + sigma * dW_t

where:
    theta  -- mean-reversion speed (> 0 for stationarity)
    mu     -- long-run equilibrium level
    sigma  -- diffusion (volatility of the spread)

The strategy identifies cointegrated pairs, estimates OU parameters via
maximum-likelihood, and trades the normalised spread when it deviates
significantly from equilibrium.

References
----------
- Uhlenbeck, G. E. & Ornstein, L. S. (1930). "On the Theory of the
  Brownian Motion". Physical Review, 36(5), 823-841.
- Engle, R. F. & Granger, C. W. J. (1987). "Co-Integration and Error
  Correction: Representation, Estimation, and Testing". Econometrica.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class OUParams:
    """Estimated parameters of an Ornstein-Uhlenbeck process."""

    theta: float  # mean-reversion speed
    mu: float  # long-run mean
    sigma: float  # diffusion coefficient

    @property
    def half_life(self) -> float:
        """Half-life of mean reversion in discrete time steps.

        Derived from the continuous-time relation:
            half_life = ln(2) / theta
        """
        if self.theta <= 0:
            return np.inf
        return np.log(2) / self.theta


@dataclass
class PairInfo:
    """Metadata for a selected cointegrated pair."""

    asset_y: str
    asset_x: str
    beta: float  # hedge ratio from cointegration regression
    alpha: float  # intercept from cointegration regression
    coint_pvalue: float
    ou_params: OUParams
    adf_pvalue: float  # ADF test p-value on the spread


@dataclass
class OUStrategyConfig:
    """Hyper-parameters for the OU mean-reversion strategy."""

    # -- Pair selection --
    coint_pvalue_threshold: float = 0.05
    half_life_min: float = 5.0  # trading days
    half_life_max: float = 60.0  # trading days

    # -- Signal generation --
    entry_z: float = 2.0  # |z| > entry_z  => open position
    exit_z: float = 0.5  # |z| < exit_z   => close position
    stop_z: float = 4.0  # |z| > stop_z   => stop-loss exit

    # -- Estimation --
    lookback: int = 252  # rolling calibration window (trading days)
    adf_pvalue_threshold: float = 0.05

    # -- Position sizing --
    use_kelly: bool = True
    kelly_fraction: float = 0.5  # half-Kelly


# ---------------------------------------------------------------------------
# OU parameter estimation via MLE
# ---------------------------------------------------------------------------


def ou_mle(spread: np.ndarray, dt: float = 1.0) -> OUParams:
    """Estimate OU parameters (theta, mu, sigma) by maximum likelihood.

    For the discrete-time AR(1) representation:
        X_{t+1} = X_t + theta*(mu - X_t)*dt + sigma*sqrt(dt)*eps_t

    which gives the exact discretisation:
        X_{t+1} = a + b * X_t + eta_t
    where
        b   = exp(-theta * dt)
        a   = mu * (1 - b)
        var(eta) = sigma^2 / (2*theta) * (1 - b^2)

    We maximise the Gaussian log-likelihood of the transitions.

    Parameters
    ----------
    spread : np.ndarray
        Time series of spread values (length N >= 3).
    dt : float
        Time step between observations (default 1 day).

    Returns
    -------
    OUParams
        Fitted theta, mu, sigma.

    Raises
    ------
    ValueError
        If the spread is too short or optimisation fails.
    """
    x = np.asarray(spread, dtype=np.float64)
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 observations for OU MLE.")

    x_lag = x[:-1]
    x_lead = x[1:]

    # ------------------------------------------------------------------
    # Negative log-likelihood for the exact OU discretisation
    # params = [theta, mu, sigma]
    # ------------------------------------------------------------------
    def neg_log_likelihood(params: np.ndarray) -> float:
        theta_p, mu_p, sigma_p = params
        if theta_p <= 1e-8 or sigma_p <= 1e-8:
            return 1e12

        b = np.exp(-theta_p * dt)
        a = mu_p * (1.0 - b)
        var_eta = (sigma_p ** 2) / (2.0 * theta_p) * (1.0 - b ** 2)

        if var_eta <= 0:
            return 1e12

        residuals = x_lead - a - b * x_lag
        nll = 0.5 * (n - 1) * np.log(2.0 * np.pi * var_eta) + 0.5 * np.sum(
            residuals ** 2
        ) / var_eta
        return nll

    # ------------------------------------------------------------------
    # Initial guesses from OLS: X_{t+1} = a + b*X_t + eps
    # ------------------------------------------------------------------
    b_ols = np.cov(x_lead, x_lag)[0, 1] / np.var(x_lag, ddof=0)
    a_ols = np.mean(x_lead) - b_ols * np.mean(x_lag)

    if b_ols <= 0 or b_ols >= 1:
        # Fallback: assume moderate mean reversion
        theta0 = 0.05
        mu0 = np.mean(x)
    else:
        theta0 = -np.log(b_ols) / dt
        mu0 = a_ols / (1.0 - b_ols)

    residuals_init = x_lead - a_ols - b_ols * x_lag
    sigma0_sq = np.var(residuals_init, ddof=0)
    # Invert var(eta) = sigma^2/(2*theta)*(1-b^2) to get sigma
    b0 = np.exp(-theta0 * dt)
    denom = (1.0 - b0 ** 2) / (2.0 * theta0)
    sigma0 = np.sqrt(max(sigma0_sq / max(denom, 1e-12), 1e-8))

    x0 = np.array([max(theta0, 1e-4), mu0, max(sigma0, 1e-4)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            neg_log_likelihood,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10},
        )

    if not result.success:
        logger.warning("OU MLE did not converge: %s", result.message)

    theta_hat, mu_hat, sigma_hat = result.x
    theta_hat = max(theta_hat, 1e-8)
    sigma_hat = max(sigma_hat, 1e-8)

    return OUParams(theta=theta_hat, mu=mu_hat, sigma=sigma_hat)


# ---------------------------------------------------------------------------
# Strategy implementation
# ---------------------------------------------------------------------------


class OUMeanReversionStrategy(Strategy):
    """Pairs-trading strategy driven by OU mean-reversion dynamics.

    Workflow
    --------
    1. **fit(data)** -- screen all asset pairs for cointegration, estimate
       hedge ratios, fit OU parameters via MLE, and retain pairs whose
       spread half-life falls within [half_life_min, half_life_max].

    2. **generate_signals(data)** -- for each selected pair, compute the
       normalised z-score of the spread and emit position signals:
       +1 / -1 (entry), 0 (exit or stop-loss).  Signals are sized using
       (half-)Kelly criterion when enabled.
    """

    def __init__(self, config: Optional[OUStrategyConfig] = None) -> None:
        super().__init__(
            name="OU Mean Reversion",
            description="Pairs-trading strategy driven by OU mean-reversion dynamics.",
        )
        self.config = config or OUStrategyConfig()
        self.pairs: list[PairInfo] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    #  Pair selection helpers                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cointegration_regression(
        y: np.ndarray, x: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """Run OLS: Y = alpha + beta * X + eps.

        Returns (alpha, beta, residuals).
        """
        x_const = add_constant(x)
        model = OLS(y, x_const).fit()
        alpha, beta = model.params
        return alpha, beta, model.resid

    def _select_pairs(self, data: pd.DataFrame) -> list[PairInfo]:
        """Screen all (N choose 2) pairs for cointegration, fit OU, filter.

        Parameters
        ----------
        data : pd.DataFrame
            Price matrix (DatetimeIndex x tickers).

        Returns
        -------
        list[PairInfo]
            Pairs passing all filters, sorted by half-life (ascending).
        """
        tickers = list(data.columns)
        if len(tickers) < 2:
            raise ValueError("Need at least 2 assets for pair selection.")

        candidates: list[PairInfo] = []

        for t_y, t_x in combinations(tickers, 2):
            y = data[t_y].values
            x = data[t_x].values

            # -- Engle-Granger cointegration test --
            try:
                _, pvalue, _ = coint(y, x)
            except Exception:
                continue

            if pvalue > self.config.coint_pvalue_threshold:
                continue

            # -- Cointegration regression for hedge ratio --
            alpha, beta, residuals = self._cointegration_regression(y, x)

            # -- ADF test on the spread --
            try:
                adf_stat, adf_pvalue, *_ = adfuller(residuals, maxlag=1)
            except Exception:
                continue

            if adf_pvalue > self.config.adf_pvalue_threshold:
                continue

            # -- OU parameter estimation via MLE --
            try:
                ou = ou_mle(residuals)
            except (ValueError, RuntimeError):
                continue

            # -- Half-life filter --
            hl = ou.half_life
            if not (self.config.half_life_min <= hl <= self.config.half_life_max):
                continue

            pair = PairInfo(
                asset_y=t_y,
                asset_x=t_x,
                beta=beta,
                alpha=alpha,
                coint_pvalue=pvalue,
                ou_params=ou,
                adf_pvalue=adf_pvalue,
            )
            candidates.append(pair)
            logger.info(
                "Pair (%s, %s): coint p=%.4f, ADF p=%.4f, "
                "theta=%.4f, mu=%.4f, sigma=%.4f, half-life=%.1f days",
                t_y,
                t_x,
                pvalue,
                adf_pvalue,
                ou.theta,
                ou.mu,
                ou.sigma,
                hl,
            )

        # Sort by half-life (prefer faster mean reversion)
        candidates.sort(key=lambda p: p.ou_params.half_life)
        logger.info("Selected %d pairs from %d tickers.", len(candidates), len(tickers))
        return candidates

    # ------------------------------------------------------------------ #
    #  Kelly criterion                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _half_kelly_weight(returns: np.ndarray) -> float:
        """Compute half-Kelly fraction: f* = 0.5 * |mu_r| / sigma_r^2.

        For a mean-reversion strategy the sign of the expected return is
        already encoded in the position direction, so we use the absolute
        value of the mean to avoid zeroing out the weight when the
        in-sample mean happens to be negative over a short window.

        Parameters
        ----------
        returns : np.ndarray
            Historical strategy returns for the pair.

        Returns
        -------
        float
            Position sizing weight, clipped to [0.1, 1].  The floor of
            0.1 prevents a pair from being silenced entirely by noisy
            in-sample estimates.
        """
        mu_r = np.mean(returns)
        var_r = np.var(returns, ddof=1)
        if var_r < 1e-12:
            return 1.0
        kelly_full = abs(mu_r) / var_r
        return float(np.clip(0.5 * kelly_full, 0.1, 1.0))

    # ------------------------------------------------------------------ #
    #  Strategy interface                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, data: pd.DataFrame) -> "OUMeanReversionStrategy":
        """Calibrate the strategy: select pairs, estimate parameters.

        Parameters
        ----------
        data : pd.DataFrame
            Historical *price* data. Columns are tickers, rows are
            trading days (DatetimeIndex).

        Returns
        -------
        self
        """
        # Use the last `lookback` observations (or all if shorter)
        lookback = self.config.lookback
        fit_data = data.iloc[-lookback:] if len(data) > lookback else data

        # Drop columns with any NaN in the fit window
        fit_data = fit_data.dropna(axis=1, how="any")

        self.pairs = self._select_pairs(fit_data)
        self._is_fitted = True
        self._fitted = True

        if not self.pairs:
            logger.warning(
                "No qualifying pairs found.  "
                "Consider relaxing the cointegration or half-life thresholds."
            )

        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate position signals for every selected pair.

        For each pair (Y, X) the spread is:
            S_t = Y_t - beta * X_t - alpha

        The normalised z-score is:
            z_t = (S_t - mu) / sigma_eq

        where sigma_eq = sigma / sqrt(2 * theta) is the stationary std of
        the OU process.

        Signal rules (applied with hysteresis):
            * Entry long  spread:  z_t < -entry_z   =>  long Y, short X
            * Entry short spread:  z_t >  entry_z   =>  short Y, long X
            * Exit:                |z_t| < exit_z    =>  flatten
            * Stop-loss:           |z_t| > stop_z    =>  flatten

        The output DataFrame has one column per asset that participates
        in at least one pair.  When an asset appears in multiple pairs the
        signals are summed (portfolio-level aggregation).

        Parameters
        ----------
        data : pd.DataFrame
            Price data covering the signal-generation period.

        Returns
        -------
        pd.DataFrame
            Position signals indexed like *data*.
        """
        if not self._is_fitted:
            raise RuntimeError("Strategy must be fit before generating signals.")

        signals = pd.DataFrame(0.0, index=data.index, columns=data.columns)

        for pair in self.pairs:
            if pair.asset_y not in data.columns or pair.asset_x not in data.columns:
                logger.warning(
                    "Pair (%s, %s) assets missing from data; skipping.",
                    pair.asset_y,
                    pair.asset_x,
                )
                continue

            y = data[pair.asset_y].values.astype(np.float64)
            x = data[pair.asset_x].values.astype(np.float64)

            spread = y - pair.beta * x - pair.alpha

            # Stationary standard deviation of the OU process:
            #   Var_stationary = sigma^2 / (2 * theta)
            ou = pair.ou_params
            sigma_eq = ou.sigma / np.sqrt(2.0 * ou.theta)
            if sigma_eq < 1e-12:
                continue

            z = (spread - ou.mu) / sigma_eq

            # ---------------------------------------------------------
            # Vectorised signal generation with hysteresis
            # ---------------------------------------------------------
            n = len(z)
            pos = np.zeros(n, dtype=np.float64)  # position per unit
            current_pos = 0.0

            for t in range(n):
                zt = z[t]
                abs_zt = abs(zt)

                if current_pos == 0.0:
                    # No position -- check for entry
                    if zt < -self.config.entry_z:
                        current_pos = 1.0  # long spread (long Y, short X)
                    elif zt > self.config.entry_z:
                        current_pos = -1.0  # short spread (short Y, long X)
                else:
                    # Holding a position -- check for exit / stop
                    if abs_zt < self.config.exit_z:
                        current_pos = 0.0  # mean-reversion exit
                    elif abs_zt > self.config.stop_z:
                        current_pos = 0.0  # stop-loss exit

                pos[t] = current_pos

            # ---------------------------------------------------------
            # Kelly position sizing
            # ---------------------------------------------------------
            if self.config.use_kelly:
                # Estimate strategy returns from the spread changes
                spread_returns = np.diff(spread)
                # Align returns with position (position at t drives PnL at t+1)
                strat_returns = pos[:-1] * spread_returns
                weight = self._half_kelly_weight(strat_returns)
            else:
                weight = 1.0

            # ---------------------------------------------------------
            # Map pair-level position to per-asset signals
            # ---------------------------------------------------------
            # pos > 0  => long Y, short X  (long the spread)
            # pos < 0  => short Y, long X  (short the spread)
            signals[pair.asset_y] += pos * weight
            signals[pair.asset_x] -= pos * pair.beta * weight

        return signals

    # ------------------------------------------------------------------ #
    #  Rolling re-estimation                                               #
    # ------------------------------------------------------------------ #

    def refit_rolling(
        self, data: pd.DataFrame, refit_freq: int = 21
    ) -> pd.DataFrame:
        """Generate signals with periodic rolling re-estimation.

        Every *refit_freq* trading days the model is re-fit on the
        trailing *lookback* window and signals are generated for the
        next block.

        Parameters
        ----------
        data : pd.DataFrame
            Full price history.
        refit_freq : int
            Number of trading days between re-estimations (default 21,
            roughly one calendar month).

        Returns
        -------
        pd.DataFrame
            Concatenated signals covering the full *data* range.
        """
        lookback = self.config.lookback
        all_signals: list[pd.DataFrame] = []
        t = lookback

        while t < len(data):
            end = min(t + refit_freq, len(data))
            train = data.iloc[max(0, t - lookback) : t]
            test = data.iloc[t:end]

            self.fit(train)
            sigs = self.generate_signals(test)
            all_signals.append(sigs)

            t = end

        if not all_signals:
            return pd.DataFrame(0.0, index=data.index, columns=data.columns)

        return pd.concat(all_signals)
