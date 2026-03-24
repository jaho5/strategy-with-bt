"""Dynamic alpha estimation strategy using the Kalman Filter.

State-space model
-----------------
State vector:  x_t = [alpha_t, beta_t]'

    State equation (random walk):
        x_t = F * x_{t-1} + w_t,   w_t ~ N(0, Q)
        F = I_2  (identity -- both alpha and beta follow random walks)

    Observation equation:
        r_{stock,t} = H_t * x_t + v_t,   v_t ~ N(0, R)
        H_t = [1,  r_{market,t}]

Kalman recursion (predict / update):
    Predict:  x_{t|t-1} = F * x_{t-1|t-1}
              P_{t|t-1} = F * P_{t-1|t-1} * F' + Q
    Update:   S_t = H_t * P_{t|t-1} * H_t' + R
              K_t = P_{t|t-1} * H_t' * S_t^{-1}
              x_{t|t} = x_{t|t-1} + K_t * (y_t - H_t * x_{t|t-1})
              P_{t|t} = (I - K_t * H_t) * P_{t|t-1}

Trading rule (PRIOR-BASED -- no look-ahead):
    alpha z-score = alpha_{t|t-1} / sqrt(P_{t|t-1}[0,0])
    (Using the PRIOR / predicted state, NOT the posterior, to avoid
    using return_t information in the signal that trades on return_t.)

    Adaptive threshold: median(|z|) over a trailing window, floored at 0.5
    Long  if z > +threshold
    Short if z < -threshold
    Flat  otherwise

Position sizing is proportional to |z|, capped at 10 % per name,
normalised to target leverage 1.0, and beta-hedged to keep portfolio
beta within +/- 0.1 of zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class _KalmanState:
    """Per-asset Kalman filter state container.

    Stores the PRIOR (predicted) estimates at each step, which represent
    the forecast BEFORE seeing the observation at time t.  This avoids
    look-ahead bias: the signal at time t is based only on information
    available at time t-1.
    """

    kf: KalmanFilter
    # Prior (predicted) estimates -- available BEFORE seeing return_t
    prior_alphas: list = field(default_factory=list)
    prior_alpha_vars: list = field(default_factory=list)
    prior_betas: list = field(default_factory=list)


class KalmanAlphaStrategy(Strategy):
    """Market-neutral strategy that trades on time-varying Kalman alpha.

    Parameters
    ----------
    market_ticker : str
        Column name in *data* representing the broad market return
        (e.g. ``"SPY"``).  This column is used as the market factor in
        the observation equation and is excluded from the tradeable
        universe.
    z_entry : float
        Alpha z-score threshold for entering a position (long or short).
        If ``adaptive_threshold`` is True this serves as a floor.
    max_position : float
        Maximum absolute weight for a single name.
    target_leverage : float
        Target gross exposure after normalisation.
    beta_drift_limit : float
        Maximum tolerable absolute portfolio beta before rebalancing
        the hedge.
    q_alpha : float
        Process noise variance for the alpha state.
    q_beta : float
        Process noise variance for the beta state.
    obs_noise : float
        Observation noise variance R.
    adaptive_threshold : bool
        If True, compute the z-score entry threshold adaptively from
        the trailing distribution of absolute z-scores (median).  The
        ``z_entry`` parameter then acts as a floor.
    warmup : int
        Number of burn-in observations at the start of filtering during
        which no signals are produced (allows the filter to converge).
    """

    def __init__(
        self,
        market_ticker: str = "SPY",
        z_entry: float = 0.5,
        max_position: float = 0.10,
        target_leverage: float = 1.0,
        beta_drift_limit: float = 0.1,
        q_alpha: float = 1e-6,
        q_beta: float = 1e-5,
        obs_noise: float = 5e-4,
        adaptive_threshold: bool = True,
        warmup: int = 60,
    ) -> None:
        super().__init__(
            name="KalmanAlpha",
            description="Market-neutral strategy trading on time-varying Kalman alpha",
        )
        self.market_ticker = market_ticker
        self.z_entry = z_entry
        self.max_position = max_position
        self.target_leverage = target_leverage
        self.beta_drift_limit = beta_drift_limit
        self.q_alpha = q_alpha
        self.q_beta = q_beta
        self.obs_noise = obs_noise
        self.adaptive_threshold = adaptive_threshold
        self.warmup = warmup

        # Populated by fit()
        self._states: Dict[str, _KalmanState] = {}
        self._train_len: int = 0  # number of training-period return observations
        self._train_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_filter(self) -> KalmanFilter:
        """Instantiate a 2-state, 1-observation Kalman filter."""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition (identity -- random walk)
        kf.F = np.eye(2)

        # Process noise covariance
        kf.Q = np.array([
            [self.q_alpha, 0.0],
            [0.0, self.q_beta],
        ])

        # Measurement noise covariance (scalar observation)
        kf.R = np.array([[self.obs_noise]])

        # Initial state: zero alpha, unit beta
        kf.x = np.array([[0.0], [1.0]])

        # Initial state covariance -- scaled to be consistent with
        # daily-return magnitudes rather than a unit diffuse prior.
        kf.P = np.array([
            [1e-4, 0.0],
            [0.0, 1.0],
        ])

        # H is set dynamically each step (depends on market return)
        kf.H = np.array([[1.0, 0.0]])

        return kf

    def _run_filter(
        self,
        stock_returns: np.ndarray,
        market_returns: np.ndarray,
        kf: Optional[KalmanFilter] = None,
    ) -> _KalmanState:
        """Run the Kalman filter over a single stock's return series.

        Records the PRIOR (predicted) state at each step -- i.e. the
        state estimate BEFORE incorporating the observation at time t.
        This ensures no look-ahead bias: signal_t depends only on
        information from times <= t-1.

        Parameters
        ----------
        stock_returns : np.ndarray, shape (T,)
        market_returns : np.ndarray, shape (T,)
        kf : KalmanFilter, optional
            If provided, continue filtering from this filter's current
            state (used for out-of-sample continuation).  If None, a
            fresh filter is created.

        Returns
        -------
        _KalmanState with filled history arrays and the filter at its
        terminal state.
        """
        if kf is None:
            kf = self._make_filter()
        state = _KalmanState(kf=kf)

        for t in range(len(stock_returns)):
            r_market = market_returns[t]

            # Design matrix for this step: r_stock = 1*alpha + r_market*beta + eps
            kf.H = np.array([[1.0, r_market]])

            # Predict step: x_{t|t-1}, P_{t|t-1}
            kf.predict()

            # Record PRIOR estimates (before update) -- this is the
            # forecast based on information up to t-1 only.
            state.prior_alphas.append(float(kf.x[0, 0]))
            state.prior_alpha_vars.append(float(kf.P[0, 0]))
            state.prior_betas.append(float(kf.x[1, 0]))

            # Update with the observed stock return
            z = np.array([[stock_returns[t]]])
            kf.update(z)

        return state

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame, **kwargs: Any) -> "KalmanAlphaStrategy":
        """Calibrate per-asset Kalman filters on historical price data.

        Runs the filter forward over the training data and saves the
        terminal filter state for each asset.  This state is used as
        the starting point when ``generate_signals`` is called on
        out-of-sample data, ensuring NO information from the test
        period influences the training-period filter.

        Parameters
        ----------
        data : pd.DataFrame
            Historical **price** data.  Columns are asset tickers
            (must include ``self.market_ticker``), index is a
            DatetimeIndex.

        Returns
        -------
        self
        """
        if self.market_ticker not in data.columns:
            raise ValueError(
                f"Market ticker '{self.market_ticker}' not found in data columns."
            )

        self._train_data = data.copy()

        returns = data.pct_change().dropna()
        market_ret = returns[self.market_ticker].values
        self._train_len = len(returns)

        self._states = {}
        tickers = [c for c in returns.columns if c != self.market_ticker]

        for ticker in tickers:
            stock_ret = returns[ticker].values
            # Skip stocks that have all-NaN returns
            if np.all(np.isnan(stock_ret)):
                continue
            # Replace any remaining NaNs with 0 for filtering stability
            stock_ret = np.nan_to_num(stock_ret, nan=0.0)
            mkt = np.nan_to_num(market_ret, nan=0.0)
            self._states[ticker] = self._run_filter(stock_ret, mkt)

        self._fitted = True
        return self

    def generate_signals(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Generate position signals from price data.

        If the strategy has not been ``fit`` yet, ``fit`` is called
        automatically on the supplied *data*.

        For out-of-sample data, the Kalman filter CONTINUES from the
        saved terminal state of the training run.  No re-fitting or
        re-running on training data occurs.  Each test observation is
        processed sequentially (predict -> record prior -> update),
        and the signal at time t uses only the PRIOR state (information
        up to t-1), never the posterior (which would incorporate return_t).

        Parameters
        ----------
        data : pd.DataFrame
            Price data (may extend beyond the fit window).

        Returns
        -------
        pd.DataFrame
            Per-asset position weights.  The market ticker column is
            included and contains the market hedge weight.
        """
        if not self._fitted:
            self.fit(data)

        returns = data.pct_change().dropna()
        market_ret = returns[self.market_ticker].values
        tickers = [c for c in returns.columns if c != self.market_ticker]

        # Continue filtering from saved training state for each asset.
        # Deep-copy the filter so we don't mutate the training state
        # (allows generate_signals to be called multiple times).
        oos_states: Dict[str, _KalmanState] = {}
        for ticker in tickers:
            if ticker not in self._states:
                continue
            train_state = self._states[ticker]
            kf_copy = self._make_filter()
            # Restore terminal state from training
            kf_copy.x = train_state.kf.x.copy()
            kf_copy.P = train_state.kf.P.copy()

            stock_ret = returns[ticker].values
            if np.all(np.isnan(stock_ret)):
                continue
            stock_ret = np.nan_to_num(stock_ret, nan=0.0)
            mkt = np.nan_to_num(market_ret, nan=0.0)
            oos_states[ticker] = self._run_filter(stock_ret, mkt, kf=kf_copy)

        T_sig = len(returns)
        signal_data: Dict[str, np.ndarray] = {}

        for ticker in tickers:
            if ticker not in oos_states:
                signal_data[ticker] = np.zeros(T_sig)
                continue

            st = oos_states[ticker]
            alphas = np.array(st.prior_alphas)
            alpha_vars = np.array(st.prior_alpha_vars)

            # Alpha z-scores from PRIOR estimates
            alpha_std = np.sqrt(np.maximum(alpha_vars, 1e-14))
            z_full = alphas / alpha_std

            # Determine threshold using training-period z-scores for
            # the adaptive threshold calibration, then extend causally
            # into the OOS period.
            if self.adaptive_threshold:
                # Gather training-period z-scores for baseline
                train_st = self._states.get(ticker)
                if train_st is not None:
                    train_alphas = np.array(train_st.prior_alphas)
                    train_alpha_vars = np.array(train_st.prior_alpha_vars)
                    train_std = np.sqrt(np.maximum(train_alpha_vars, 1e-14))
                    train_z = train_alphas / train_std
                else:
                    train_z = np.array([])

                abs_z_all = np.concatenate([np.abs(train_z), np.abs(z_full)])
                window = 120
                thresholds = np.empty(T_sig)
                # Offset into the combined z-score history
                offset = len(train_z)
                for i in range(T_sig):
                    # Index into the combined array
                    idx = offset + i
                    start = max(0, idx - window + 1)
                    med = np.median(abs_z_all[start : idx + 1])
                    thresholds[i] = max(med, self.z_entry)
            else:
                thresholds = np.full(T_sig, self.z_entry)

            # No warmup needed for OOS -- the filter was already
            # warmed up during training.  But if training was very
            # short, apply a minimal guard.
            valid_sig = np.ones(T_sig, dtype=bool)
            if self._train_len < self.warmup:
                remaining_warmup = self.warmup - self._train_len
                valid_sig[:remaining_warmup] = False

            # Raw signal: proportional to z-score when beyond threshold
            raw = np.where(
                valid_sig & (z_full > thresholds),
                z_full,
                np.where(valid_sig & (z_full < -thresholds), z_full, 0.0),
            )

            signal_data[ticker] = raw

        # ----------------------------------------------------------
        # Diagnostic: check for look-ahead bias
        # ----------------------------------------------------------
        self._run_lookahead_diagnostic(signal_data, returns, tickers)

        # ----------------------------------------------------------
        # Build weight matrix row-by-row (each time step independently)
        # ----------------------------------------------------------
        sig_dates = returns.index
        weight_matrix = pd.DataFrame(
            0.0,
            index=sig_dates,
            columns=list(returns.columns),
        )

        for t_idx in range(T_sig):
            raw_weights: Dict[str, float] = {}
            asset_betas: Dict[str, float] = {}

            for ticker in tickers:
                if ticker not in oos_states:
                    continue
                sig = signal_data[ticker][t_idx]
                if sig == 0.0:
                    continue
                raw_weights[ticker] = sig
                asset_betas[ticker] = oos_states[ticker].prior_betas[t_idx]

            if not raw_weights:
                continue

            # ---- Position sizing ----
            total_abs = sum(abs(w) for w in raw_weights.values())
            if total_abs == 0.0:
                continue

            # Normalise to target leverage
            scale = self.target_leverage / total_abs
            weights = {tk: w * scale for tk, w in raw_weights.items()}

            # Cap individual positions
            capped = False
            for tk in list(weights):
                if abs(weights[tk]) > self.max_position:
                    weights[tk] = np.sign(weights[tk]) * self.max_position
                    capped = True

            # Re-normalise after capping if any were capped
            if capped:
                total_abs = sum(abs(w) for w in weights.values())
                if total_abs > 0:
                    scale = self.target_leverage / total_abs
                    weights = {tk: w * scale for tk, w in weights.items()}
                    # Re-cap (iterate once more for convergence)
                    for tk in list(weights):
                        if abs(weights[tk]) > self.max_position:
                            weights[tk] = (
                                np.sign(weights[tk]) * self.max_position
                            )

            # ---- Beta hedge (market-neutral) ----
            portfolio_beta = sum(
                weights.get(tk, 0.0) * asset_betas.get(tk, 0.0)
                for tk in weights
            )

            # Always hedge; add market position to neutralise beta
            if abs(portfolio_beta) > 0:
                weights[self.market_ticker] = -portfolio_beta

            # Write row
            for tk, w in weights.items():
                weight_matrix.at[sig_dates[t_idx], tk] = w

        return weight_matrix

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _run_lookahead_diagnostic(
        self,
        signal_data: Dict[str, np.ndarray],
        returns: pd.DataFrame,
        tickers: list,
    ) -> None:
        """Print diagnostic correlation between signal_t and return_t.

        If the correlation is high (> 0.5), the signal is likely using
        future information (look-ahead bias).  A properly causal signal
        (based on prior state) should have near-zero contemporaneous
        correlation with the return it is computed alongside.
        """
        correlations = []
        for ticker in tickers:
            if ticker not in signal_data:
                continue
            sig = signal_data[ticker]
            if ticker in returns.columns:
                ret = returns[ticker].values
                # Align lengths
                n = min(len(sig), len(ret))
                s = sig[:n]
                r = ret[:n]
                # Only compute on non-zero signal periods
                mask = s != 0.0
                if mask.sum() > 10:
                    corr = np.corrcoef(s[mask], r[mask])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

        if correlations:
            mean_corr = np.mean(correlations)
            max_corr = np.max(np.abs(correlations))
            logger.info(
                "[KalmanAlpha DIAGNOSTIC] signal_t vs return_t correlation: "
                "mean=%.4f, max|corr|=%.4f (n=%d assets)",
                mean_corr, max_corr, len(correlations),
            )
            if max_corr > 0.5:
                logger.warning(
                    "[KalmanAlpha DIAGNOSTIC] HIGH contemporaneous correlation "
                    "(%.4f) detected -- possible look-ahead bias!",
                    max_corr,
                )
            elif mean_corr > 0.3:
                logger.warning(
                    "[KalmanAlpha DIAGNOSTIC] Elevated mean contemporaneous "
                    "correlation (%.4f) -- review signal timing.",
                    mean_corr,
                )
            else:
                logger.info(
                    "[KalmanAlpha DIAGNOSTIC] Contemporaneous correlation is low "
                    "-- no look-ahead bias detected."
                )
