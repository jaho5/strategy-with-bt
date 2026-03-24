"""
Bayesian Online Changepoint Detection Strategy
===============================================

Mathematical foundation
-----------------------
Adams & MacKay (2007) Bayesian Online Changepoint Detection (BOCPD).

The algorithm maintains a posterior distribution over the *run length*
r_t -- the number of time steps since the last changepoint.  At each new
observation x_t, the run length distribution is updated via:

    P(r_t | x_{1:t}) ~ P(x_t | r_t, x_{t-r_t:t-1}) * P(r_t | r_{t-1})

where:
    - P(x_t | r_t, ...) is the *predictive probability* under the
      sufficient statistics accumulated during the current run,
    - P(r_t | r_{t-1}) encodes the *hazard function* H(tau) = 1/lambda,
      which places a geometric prior on run length with mean lambda.

For Gaussian data with unknown mean and variance we use the
Normal-Inverse-Gamma (NIG) conjugate prior:

    mu | sigma^2 ~ N(mu_0, sigma^2 / kappa_0)
    sigma^2       ~ IG(alpha_0, beta_0)

The posterior predictive under NIG is a Student-t distribution:

    x_{n+1} | x_{1:n} ~ t_{2 alpha_n}(mu_n, beta_n (kappa_n + 1) / (alpha_n kappa_n))

Sufficient statistic updates (online, O(1) per step):
    kappa_n = kappa_0 + n
    mu_n    = (kappa_0 mu_0 + sum(x)) / kappa_n
    alpha_n = alpha_0 + n / 2
    beta_n  = beta_0 + 0.5 * (sum(x^2) - kappa_n mu_n^2 + kappa_0 mu_0^2)

Strategy logic
--------------
1. **Online changepoint detection**: maintain the full run length
   distribution.  A changepoint is detected when P(r_t = 0 | x_{1:t})
   exceeds a configurable threshold.

2. **Post-changepoint trading**: after a changepoint, wait for the new
   regime to establish (``regime_establish_bars`` observations), then
   estimate the new regime mean and variance.  Go long if the regime mean
   is positive, short if negative, with signal strength proportional to
   the t-statistic.

3. **Position sizing**: scale by (1 - changepoint_probability) so that
   exposure is reduced near changepoints.  Increase position as the run
   length grows and the regime becomes more established.

References
----------
- Adams, R. P. & MacKay, D. J. C. (2007). "Bayesian Online Changepoint
  Detection." arXiv:0710.3742.
- Murphy, K. P. (2007). "Conjugate Bayesian analysis of the Gaussian
  distribution." Technical report, UBC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln
from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BOCPDConfig:
    """Tuneable hyper-parameters for the BOCPD strategy."""

    # --- Hazard function ---
    # Geometric prior on run length: H(tau) = 1 / hazard_lambda.
    # hazard_lambda is the expected run length (mean time between
    # changepoints, in bars).
    hazard_lambda: float = 250.0

    # --- Normal-Inverse-Gamma prior ---
    # Prior on the mean: mu ~ N(mu0, sigma^2 / kappa0)
    mu0: float = 0.0
    kappa0: float = 1.0
    # Prior on the variance: sigma^2 ~ IG(alpha0, beta0)
    alpha0: float = 1.0
    beta0: float = 1e-4

    # --- Changepoint detection ---
    # Threshold on P(r_t = 0 | data) to declare a changepoint.
    changepoint_threshold: float = 0.3
    # Maximum run length to track (truncation for memory efficiency).
    max_run_length: int = 500

    # --- Signal generation ---
    # Number of bars to wait after a detected changepoint before trading.
    regime_establish_bars: int = 5
    # Minimum |t-statistic| of the regime mean to generate a signal.
    min_t_stat: float = 0.5
    # Maximum absolute signal weight (before vol-targeting).
    max_signal_weight: float = 1.0

    # --- Position sizing ---
    # Exponent controlling how position grows with run length:
    #   run_scale = min(1, (run_length / run_length_full_scale) ^ run_scale_power)
    run_length_full_scale: float = 50.0
    run_scale_power: float = 0.5

    # --- Return computation ---
    # Use log returns (True) or simple returns (False).
    use_log_returns: bool = True


# ---------------------------------------------------------------------------
# Normal-Inverse-Gamma sufficient statistics (vectorised over run lengths)
# ---------------------------------------------------------------------------


class NIGSufficientStats:
    """Maintains Normal-Inverse-Gamma sufficient statistics for all active
    run lengths simultaneously.

    For run length r, the sufficient statistics are:
        kappa[r], mu[r], alpha[r], beta[r]

    All arrays are 1-D with length equal to the current number of tracked
    run lengths (max_run_length + 1).
    """

    def __init__(self, config: BOCPDConfig, max_len: int) -> None:
        self.config = config
        self.max_len = max_len

        # Initialise arrays with the prior for run length 0.
        self.kappa = np.full(max_len, config.kappa0, dtype=np.float64)
        self.mu = np.full(max_len, config.mu0, dtype=np.float64)
        self.alpha = np.full(max_len, config.alpha0, dtype=np.float64)
        self.beta = np.full(max_len, config.beta0, dtype=np.float64)

    def reset_slot(self, idx: int) -> None:
        """Reset slot *idx* to the prior (for a new run)."""
        self.kappa[idx] = self.config.kappa0
        self.mu[idx] = self.config.mu0
        self.alpha[idx] = self.config.alpha0
        self.beta[idx] = self.config.beta0

    def predictive_log_prob(self, x: float, indices: np.ndarray) -> np.ndarray:
        """Compute log P(x | sufficient stats) for a subset of run lengths.

        The predictive distribution is Student-t:
            x ~ t_{2 alpha}(mu, beta (kappa + 1) / (alpha kappa))

        Parameters
        ----------
        x : float
            The new observation.
        indices : np.ndarray
            Integer indices into the stat arrays for active run lengths.

        Returns
        -------
        np.ndarray
            Log predictive probabilities, same length as *indices*.
        """
        kappa = self.kappa[indices]
        mu = self.mu[indices]
        alpha = self.alpha[indices]
        beta = self.beta[indices]

        # Degrees of freedom
        nu = 2.0 * alpha

        # Scale parameter of the predictive t-distribution
        # Var(x) = (beta / alpha) * (kappa + 1) / kappa * (nu / (nu - 2))
        # but we need the *scale* (not variance) for the density:
        scale_sq = beta * (kappa + 1.0) / (alpha * kappa)
        scale = np.sqrt(scale_sq)

        # Student-t log pdf:
        # log t_nu(x; mu, scale) = log Gamma((nu+1)/2) - log Gamma(nu/2)
        #   - 0.5 log(nu pi scale^2) - ((nu+1)/2) log(1 + ((x-mu)/scale)^2 / nu)
        z = (x - mu) / scale
        log_pdf = (
            gammaln(0.5 * (nu + 1.0))
            - gammaln(0.5 * nu)
            - 0.5 * np.log(nu * np.pi * scale_sq)
            - 0.5 * (nu + 1.0) * np.log1p(z * z / nu)
        )
        return log_pdf

    def update(self, x: float, indices: np.ndarray) -> None:
        """Update sufficient statistics for active run lengths after
        observing *x*.

        Online Bayesian update for NIG conjugate family:
            kappa_new = kappa + 1
            mu_new    = (kappa * mu + x) / kappa_new
            alpha_new = alpha + 0.5
            beta_new  = beta + 0.5 * kappa * (x - mu)^2 / kappa_new
        """
        kappa = self.kappa[indices]
        mu = self.mu[indices]

        kappa_new = kappa + 1.0
        mu_new = (kappa * mu + x) / kappa_new
        alpha_new = self.alpha[indices] + 0.5
        beta_new = self.beta[indices] + 0.5 * kappa * (x - mu) ** 2 / kappa_new

        self.kappa[indices] = kappa_new
        self.mu[indices] = mu_new
        self.alpha[indices] = alpha_new
        self.beta[indices] = beta_new


# ---------------------------------------------------------------------------
# Core BOCPD algorithm
# ---------------------------------------------------------------------------


class BOCPDEngine:
    """Bayesian Online Changepoint Detection engine.

    Implements the Adams & MacKay (2007) algorithm with NIG conjugate prior
    for Gaussian observations.  The engine processes one observation at a
    time and maintains the full (truncated) run length posterior.
    """

    def __init__(self, config: BOCPDConfig) -> None:
        self.config = config
        self.max_rl = config.max_run_length

        # Hazard: constant probability of changepoint at each step.
        self._H = 1.0 / config.hazard_lambda

        # Sufficient statistics for all possible run lengths.
        self._stats = NIGSufficientStats(config, self.max_rl + 1)

        # Run length log-probabilities (unnormalised).
        # _log_joint[r] = log P(r_t = r, x_{1:t}).
        # Only the first (_n_active) entries are meaningful.
        self._log_joint = np.full(self.max_rl + 1, -np.inf, dtype=np.float64)
        self._log_joint[0] = 0.0  # prior: r_0 = 0 with probability 1
        self._n_active = 1  # number of active run lengths

        # Posterior over run lengths (normalised), updated each step.
        self._run_length_post: np.ndarray = np.array([1.0])

        # Tracking quantities exposed to the strategy.
        self.changepoint_prob: float = 0.0
        self.map_run_length: int = 0
        self.regime_mean: float = 0.0
        self.regime_var: float = 1e-8
        self.regime_n: int = 0

    def update(self, x: float) -> None:
        """Process a single new observation and update the run length
        posterior.

        Parameters
        ----------
        x : float
            The new observation (typically a return).
        """
        n = self._n_active  # current number of active run lengths
        active = np.arange(n)

        # ------------------------------------------------------------------
        # Step 1: Evaluate predictive probabilities P(x_t | r_{t-1}, stats)
        # ------------------------------------------------------------------
        log_pred = self._stats.predictive_log_prob(x, active)

        # ------------------------------------------------------------------
        # Step 2: Compute growth probabilities (run length grows by 1)
        #   log P(r_t = r+1, x_{1:t}) = log P(x_t | r, stats) +
        #                                 log P(r_{t-1} = r, x_{1:t-1}) +
        #                                 log(1 - H)
        # ------------------------------------------------------------------
        log_growth = (
            log_pred + self._log_joint[:n] + np.log(1.0 - self._H)
        )

        # ------------------------------------------------------------------
        # Step 3: Compute changepoint probability (run length resets to 0)
        #   log P(r_t = 0, x_{1:t}) = logsumexp over r of
        #       [log P(x_t | r, stats) + log P(r_{t-1}=r, x_{1:t-1}) + log H]
        # ------------------------------------------------------------------
        log_cp_terms = log_pred + self._log_joint[:n] + np.log(self._H)
        log_cp = _logsumexp(log_cp_terms)

        # ------------------------------------------------------------------
        # Step 4: Update sufficient statistics *before* shifting arrays
        # ------------------------------------------------------------------
        self._stats.update(x, active)

        # ------------------------------------------------------------------
        # Step 5: Shift run length distribution and insert new run
        # ------------------------------------------------------------------
        new_n = min(n + 1, self.max_rl + 1)

        # Shift growth probs into positions 1..new_n-1
        if new_n <= self.max_rl:
            self._log_joint[1:new_n] = log_growth[:new_n - 1]
        else:
            # Truncation: merge the oldest run length probability into the
            # second-oldest to keep the array bounded.
            self._log_joint[1:new_n] = log_growth[:new_n - 1]
            # The last slot absorbs overflow from truncation (approximate).

        # Shift sufficient statistics: slot r+1 gets the stats that were in
        # slot r.  We shift from the end to avoid overwriting.
        if new_n > n:
            # Normal case: we have room to grow.
            for r in range(new_n - 1, 0, -1):
                self._stats.kappa[r] = self._stats.kappa[r - 1]
                self._stats.mu[r] = self._stats.mu[r - 1]
                self._stats.alpha[r] = self._stats.alpha[r - 1]
                self._stats.beta[r] = self._stats.beta[r - 1]
        else:
            # At max run length: shift within bounds.
            for r in range(new_n - 1, 0, -1):
                src = r - 1 if r - 1 < n else r
                self._stats.kappa[r] = self._stats.kappa[src]
                self._stats.mu[r] = self._stats.mu[src]
                self._stats.alpha[r] = self._stats.alpha[src]
                self._stats.beta[r] = self._stats.beta[src]

        # Slot 0 is the new run (reset to prior).
        self._log_joint[0] = log_cp
        self._stats.reset_slot(0)

        self._n_active = new_n

        # ------------------------------------------------------------------
        # Step 6: Normalise to get posterior P(r_t | x_{1:t})
        # ------------------------------------------------------------------
        log_evidence = _logsumexp(self._log_joint[:new_n])
        log_post = self._log_joint[:new_n] - log_evidence
        self._run_length_post = np.exp(log_post)

        # ------------------------------------------------------------------
        # Step 7: Extract summary quantities
        # ------------------------------------------------------------------
        self.changepoint_prob_r0 = float(self._run_length_post[0])
        self.map_run_length = int(np.argmax(self._run_length_post))

        # Effective changepoint probability: total posterior mass on short
        # run lengths (r <= regime_establish_bars).  This is more robust
        # than P(r=0) alone because after a true changepoint the mass
        # spreads across r=0,1,2,... as the new run grows.
        short_rl_cutoff = min(
            self.config.regime_establish_bars, new_n
        )
        self.changepoint_prob = float(
            self._run_length_post[:short_rl_cutoff].sum()
        )

        # Regime parameters from the MAP run length.
        map_r = self.map_run_length
        self.regime_mean = float(self._stats.mu[map_r])
        self.regime_var = float(
            self._stats.beta[map_r]
            / max(self._stats.alpha[map_r] - 1.0, 0.5)
        )
        self.regime_n = max(
            int(self._stats.kappa[map_r] - self.config.kappa0), 0
        )

    def expected_run_length(self) -> float:
        """Compute E[r_t | x_{1:t}] from the current posterior."""
        r_values = np.arange(self._n_active, dtype=np.float64)
        return float(np.dot(self._run_length_post, r_values))

    def regime_t_statistic(self) -> float:
        """Compute the t-statistic of the current regime mean estimate.

        Uses the posterior predictive:
            t = (mu_n - 0) / sqrt(beta_n / (alpha_n * kappa_n))

        with degrees of freedom 2 * alpha_n.
        """
        map_r = self.map_run_length
        kappa = self._stats.kappa[map_r]
        mu = self._stats.mu[map_r]
        alpha = self._stats.alpha[map_r]
        beta = self._stats.beta[map_r]

        se = np.sqrt(beta / (alpha * kappa))
        if se < 1e-15:
            return 0.0
        return float(mu / se)


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


class BayesianChangepointStrategy(Strategy):
    """Bayesian Online Changepoint Detection trading strategy.

    Parameters
    ----------
    config : BOCPDConfig | None
        Strategy hyper-parameters.  Defaults are used when *None*.
    """

    def __init__(self, config: BOCPDConfig | None = None) -> None:
        self.config = config or BOCPDConfig()
        super().__init__(
            name="BayesianChangepoint",
            description=(
                "Bayesian Online Changepoint Detection (Adams & MacKay 2007) "
                "with Normal-Inverse-Gamma conjugate prior.  Detects regime "
                "changes in return distributions and trades the new regime."
            ),
        )
        # Learned during fit.
        self._global_mean: float = 0.0
        self._global_var: float = 1e-4

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "BayesianChangepointStrategy":
        """Calibrate prior hyper-parameters from training data.

        The fit stage estimates global return statistics that inform the
        NIG prior:
        - mu0 is set to the sample mean of returns.
        - beta0 is set to match the sample variance so the prior is
          weakly informative but centred on the empirical scale.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (OHLCV or adjusted close).  For multi-column input
            the first column is used.

        Returns
        -------
        self
        """
        self.validate_prices(prices)
        series = prices.iloc[:, 0] if prices.ndim == 2 else prices

        returns = self._compute_returns(series)
        returns = returns.dropna()

        if len(returns) < 30:
            raise ValueError(
                f"Insufficient data for fitting: need >= 30 observations, "
                f"got {len(returns)}."
            )

        self._global_mean = float(returns.mean())
        self._global_var = float(returns.var())

        # Set informative prior based on training data.
        self.config.mu0 = self._global_mean
        # beta0: sets the prior expected variance.
        # E[sigma^2] = beta0 / (alpha0 - 1) for alpha0 > 1.
        # With alpha0 = 1 (default), use beta0 ~ sample_var * kappa0 / 2
        # so the prior predictive variance is on the right scale.
        self.config.beta0 = max(
            self._global_var * self.config.kappa0 * 0.5, 1e-8
        )

        self.parameters = {
            "global_mean": self._global_mean,
            "global_var": self._global_var,
            "mu0": self.config.mu0,
            "beta0": self.config.beta0,
        }
        self._fitted = True

        logger.info(
            "BOCPD fitted: global_mean=%.6f, global_var=%.6f, "
            "mu0=%.6f, beta0=%.6f",
            self._global_mean,
            self._global_var,
            self.config.mu0,
            self.config.beta0,
        )
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate trading signals via Bayesian Online Changepoint Detection.

        For each asset (column) in *prices*, the method:
        1. Computes returns.
        2. Runs the BOCPD algorithm sequentially.
        3. Detects changepoints and estimates new regime parameters.
        4. Generates directional signals scaled by regime confidence and
           run length.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data indexed by datetime.

        Returns
        -------
        pd.DataFrame
            Columns ``signal`` and ``weight`` (for single-asset input) or
            ``{ticker}_signal`` and ``{ticker}_weight`` per asset.
        """
        self.ensure_fitted()

        tickers = prices.columns.tolist()
        single_asset = len(tickers) == 1

        result = pd.DataFrame(index=prices.index)

        for ticker in tickers:
            series = prices[ticker].dropna()
            if len(series) < 30:
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
        """Run BOCPD on a single price series and return (signal, weight).

        Returns
        -------
        signal : pd.Series
            Directional signal in {-1, 0, +1}.
        weight : pd.Series
            Position-sizing weight in [0, 1].
        """
        cfg = self.config
        returns = self._compute_returns(series).dropna()
        T = len(returns)

        signals = np.zeros(T, dtype=np.float64)
        weights = np.zeros(T, dtype=np.float64)

        # Instantiate a fresh BOCPD engine for this series.
        engine = BOCPDEngine(cfg)

        # Track changepoint events for regime establishment logic.
        bars_since_cp = 0
        in_new_regime = False

        for t in range(T):
            x = returns.iloc[t]

            # Skip NaN / Inf observations.
            if not np.isfinite(x):
                continue

            engine.update(x)

            cp_prob = engine.changepoint_prob
            map_rl = engine.map_run_length

            # ----- Changepoint detection ---------------------------------
            if cp_prob > cfg.changepoint_threshold:
                in_new_regime = True
                bars_since_cp = 0
            else:
                bars_since_cp += 1

            # ----- Signal generation -------------------------------------
            if in_new_regime and bars_since_cp < cfg.regime_establish_bars:
                # Waiting for new regime to establish.
                signals[t] = 0.0
                weights[t] = 0.0
                continue

            if in_new_regime and bars_since_cp >= cfg.regime_establish_bars:
                # New regime established; start trading it.
                in_new_regime = False

            # Regime t-statistic.
            t_stat = engine.regime_t_statistic()

            if abs(t_stat) < cfg.min_t_stat:
                signals[t] = 0.0
                weights[t] = 0.0
                continue

            # Direction from sign of regime mean.
            signals[t] = 1.0 if engine.regime_mean > 0 else -1.0

            # ----- Position sizing ---------------------------------------
            # (a) Scale by (1 - cp_prob): reduce near changepoints.
            cp_scale = 1.0 - cp_prob

            # (b) Scale by run length maturity.
            run_ratio = min(
                float(map_rl) / cfg.run_length_full_scale, 1.0
            )
            run_scale = run_ratio ** cfg.run_scale_power

            # (c) Scale by |t-stat| (capped at reasonable level).
            t_scale = min(abs(t_stat) / 2.0, 1.0)

            raw_weight = cp_scale * run_scale * t_scale
            weights[t] = min(raw_weight, cfg.max_signal_weight)

        signal_series = pd.Series(
            signals, index=returns.index, name=series.name
        )
        weight_series = pd.Series(
            weights, index=returns.index, name=series.name
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

    def run_bocpd(
        self, series: pd.Series
    ) -> pd.DataFrame:
        """Run BOCPD and return a diagnostic DataFrame.

        Useful for analysis and plotting.  The returned DataFrame contains:
        - ``return``: the computed return series.
        - ``changepoint_prob``: total posterior mass on short run lengths.
        - ``changepoint_prob_r0``: P(r_t = 0 | data) at each step.
        - ``map_run_length``: MAP estimate of run length.
        - ``expected_run_length``: E[r_t | data].
        - ``regime_mean``: posterior mean of the current regime.
        - ``regime_var``: posterior variance of the current regime.
        - ``t_statistic``: regime t-statistic.

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
        returns = self._compute_returns(series).dropna()
        T = len(returns)

        engine = BOCPDEngine(cfg)

        records = []
        for t in range(T):
            x = returns.iloc[t]
            if not np.isfinite(x):
                records.append({
                    "return": x,
                    "changepoint_prob": np.nan,
                    "map_run_length": np.nan,
                    "expected_run_length": np.nan,
                    "regime_mean": np.nan,
                    "regime_var": np.nan,
                    "t_statistic": np.nan,
                })
                continue

            engine.update(x)
            records.append({
                "return": x,
                "changepoint_prob": engine.changepoint_prob,
                "changepoint_prob_r0": engine.changepoint_prob_r0,
                "map_run_length": engine.map_run_length,
                "expected_run_length": engine.expected_run_length(),
                "regime_mean": engine.regime_mean,
                "regime_var": engine.regime_var,
                "t_statistic": engine.regime_t_statistic(),
            })

        return pd.DataFrame(records, index=returns.index)


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------


def _logsumexp(log_x: np.ndarray) -> float:
    """Numerically stable log-sum-exp.

    Parameters
    ----------
    log_x : np.ndarray
        Array of log-values.

    Returns
    -------
    float
        log(sum(exp(log_x))).
    """
    if len(log_x) == 0:
        return -np.inf
    c = log_x.max()
    if not np.isfinite(c):
        return -np.inf
    return float(c + np.log(np.sum(np.exp(log_x - c))))
