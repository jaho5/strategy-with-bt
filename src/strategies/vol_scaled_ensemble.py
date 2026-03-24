"""Inverse-Volatility Weighted Ensemble Strategy.

A self-contained, deployable strategy that combines 5 component strategies
using inverse-volatility weighting.  This is the winning ensemble from
strategy evaluation -- it encapsulates all components internally and
presents a single Strategy interface to main.py.

Mathematical foundation
-----------------------
The ensemble combines K strategies with inverse-volatility weights:

    w_i(t) = (1/sigma_i(t)) / sum_j (1/sigma_j(t))

where sigma_i(t) is the rolling realised volatility of strategy i's returns
over a lookback window.

The ensemble Sharpe ratio (for uncorrelated strategies):

    SR_ensemble = sqrt(sum SR_i^2) ~ sqrt(K) * avg(SR_i)

when cross-strategy correlations are low.

With a leverage multiplier L:
    - Return scales linearly:  r_L = L * r_base
    - Volatility scales linearly: sigma_L = L * sigma_base
    - Sharpe remains constant: SR_L = SR_base
    - Max drawdown scales approximately linearly.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)


class VolScaledEnsembleStrategy(Strategy):
    """Inverse-volatility weighted ensemble of 5 component strategies.

    This strategy internally instantiates, fits, and generates signals from:
      1. EntropyRegularizedStrategy (gamma=0.3, eg_blend=0.8, eta0=2.0, rebalance_freq=3)
      2. GarchVolStrategy (EGARCH, target_vol=0.25, adaptive_blend=True)
      3. HMMRegimeStrategy
      4. SpectralMomentumStrategy
      5. BayesianChangepointStrategy

    It converts each strategy's signals into portfolio returns, computes
    rolling inverse-volatility weights (63-day lookback), and combines
    them into a single ensemble signal.

    Parameters
    ----------
    leverage : float
        Leverage multiplier applied to the final ensemble signal.
        Default 1.0 (no leverage).  Can be increased to target a
        specific annual return level.
    vol_lookback : int
        Rolling window (trading days) for computing per-strategy
        realised volatility used in inverse-vol weighting.  Default 63.
    min_weight : float
        Minimum weight floor per component strategy (before
        renormalisation).  Prevents any single strategy from being
        completely zeroed out.  Default 0.05.
    """

    def __init__(
        self,
        leverage: float = 1.0,
        vol_lookback: int = 63,
        min_weight: float = 0.05,
    ) -> None:
        super().__init__(
            name="VolScaledEnsemble",
            description=(
                "Inverse-volatility weighted ensemble of 5 component "
                "strategies (Entropy-Regularized, GARCH Vol, HMM Regime, "
                "Spectral Momentum, Bayesian Changepoint)."
            ),
        )
        self.leverage = leverage
        self.vol_lookback = vol_lookback
        self.min_weight = min_weight

        # Component strategies (instantiated lazily in _build_components)
        self._components: List[tuple] = []  # [(name, strategy_instance), ...]
        self._build_components()

    # ------------------------------------------------------------------
    # Component instantiation
    # ------------------------------------------------------------------

    def _build_components(self) -> None:
        """Import and instantiate the 5 component strategies.

        Each import is wrapped in a try/except so that a missing optional
        dependency does not prevent the ensemble from being constructed --
        it will simply run with fewer components (with a warning).
        """
        self._components = []

        # 1. Entropy Regularized
        try:
            from src.strategies.entropy_regularized import (
                EntropyRegularizedStrategy,
            )

            self._components.append((
                "EntropyRegularized",
                EntropyRegularizedStrategy(
                    gamma=0.3,
                    eg_blend=0.8,
                    eta0=2.0,
                    rebalance_freq=3,
                ),
            ))
        except Exception as exc:
            logger.warning("Could not load EntropyRegularizedStrategy: %s", exc)

        # 2. GARCH Vol (EGARCH, target_vol=0.25, adaptive_blend=True)
        try:
            from src.strategies.garch_vol import (
                GarchVolConfig,
                GarchVolStrategy,
            )

            cfg = GarchVolConfig(
                garch_model="EGARCH",
                target_vol=0.25,
                adaptive_blend=True,
            )
            self._components.append((
                "GarchVol",
                GarchVolStrategy(config=cfg),
            ))
        except Exception as exc:
            logger.warning("Could not load GarchVolStrategy: %s", exc)

        # 3. HMM Regime
        try:
            from src.strategies.hmm_regime import HMMRegimeStrategy

            self._components.append((
                "HMMRegime",
                HMMRegimeStrategy(),
            ))
        except Exception as exc:
            logger.warning("Could not load HMMRegimeStrategy: %s", exc)

        # 4. Spectral Momentum
        try:
            from src.strategies.spectral_momentum import (
                SpectralMomentumStrategy,
            )

            self._components.append((
                "SpectralMomentum",
                SpectralMomentumStrategy(),
            ))
        except Exception as exc:
            logger.warning("Could not load SpectralMomentumStrategy: %s", exc)

        # 5. Bayesian Changepoint
        try:
            from src.strategies.bayesian_changepoint import (
                BayesianChangepointStrategy,
            )

            self._components.append((
                "BayesianChangepoint",
                BayesianChangepointStrategy(),
            ))
        except Exception as exc:
            logger.warning("Could not load BayesianChangepointStrategy: %s", exc)

        if not self._components:
            raise RuntimeError(
                "VolScaledEnsemble: no component strategies could be loaded."
            )

        logger.info(
            "VolScaledEnsemble: loaded %d component strategies: %s",
            len(self._components),
            [name for name, _ in self._components],
        )

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs: Any) -> "VolScaledEnsembleStrategy":
        """Fit all component strategies on training data.

        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (DatetimeIndex, columns = tickers).
        **kwargs
            Additional keyword arguments forwarded to component strategies
            that accept them (e.g. ``ohlcv_data``).

        Returns
        -------
        self
        """
        self.validate_prices(prices)

        for name, strat in self._components:
            try:
                logger.info("VolScaledEnsemble: fitting %s...", name)
                strat.fit(prices, **kwargs)
            except Exception as exc:
                logger.warning(
                    "VolScaledEnsemble: fitting %s failed: %s", name, exc
                )

        self.parameters = {
            "leverage": self.leverage,
            "vol_lookback": self.vol_lookback,
            "min_weight": self.min_weight,
            "n_components": len(self._components),
        }
        self._fitted = True
        return self

    def generate_signals(
        self, prices: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Generate ensemble signals by inverse-vol weighting of components.

        Steps:
        1. Generate signals from each component strategy.
        2. Convert each set of signals to per-day portfolio returns.
        3. Compute 63-day rolling inverse-vol weights across strategies.
        4. Combine into a single ensemble signal per ticker.
        5. Apply the leverage multiplier.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data (DatetimeIndex, columns = tickers).
        **kwargs
            Additional keyword arguments forwarded to component strategies.

        Returns
        -------
        pd.DataFrame
            Columns are ticker names; values are position weights
            (positive = long, negative = short).  Suitable for
            consumption by main.py's ``_signals_to_portfolio``.
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        tickers = list(prices.columns)
        asset_returns = np.log(prices / prices.shift(1)).fillna(0.0)

        # ------------------------------------------------------------------
        # Step 1 & 2: generate signals and derive portfolio returns for
        # each component strategy.
        # ------------------------------------------------------------------
        component_signals: Dict[str, pd.DataFrame] = {}
        component_returns: Dict[str, pd.Series] = {}

        for name, strat in self._components:
            try:
                signals = strat.generate_signals(prices, **kwargs)
            except Exception as exc:
                logger.warning(
                    "VolScaledEnsemble: %s generate_signals failed: %s",
                    name,
                    exc,
                )
                continue

            # Convert the strategy's signals into a normalised position
            # DataFrame aligned to prices.index with ticker columns.
            positions = self._normalise_signals(signals, prices)
            component_signals[name] = positions

            # Compute daily portfolio return for this strategy:
            #   r_portfolio(t) = sum_i position_i(t-1) * r_i(t)
            # The 1-bar lag (shift(1)) avoids look-ahead bias.
            lagged_pos = positions.shift(1).fillna(0.0)
            port_ret = (lagged_pos * asset_returns).sum(axis=1)
            component_returns[name] = port_ret

        if not component_returns:
            logger.warning(
                "VolScaledEnsemble: all component strategies failed. "
                "Returning flat signals."
            )
            return pd.DataFrame(0.0, index=prices.index, columns=tickers)

        # ------------------------------------------------------------------
        # Step 3: compute rolling inverse-vol weights across strategies.
        # ------------------------------------------------------------------
        ret_df = pd.DataFrame(component_returns)  # columns = strategy names
        inv_vol_weights = self._compute_inverse_vol_weights(ret_df)

        # ------------------------------------------------------------------
        # Step 4: combine component positions into ensemble position.
        # ------------------------------------------------------------------
        ensemble = pd.DataFrame(0.0, index=prices.index, columns=tickers)
        strat_names = list(component_signals.keys())

        for name in strat_names:
            # w_i(t) is a scalar weight for strategy i at time t
            w = inv_vol_weights[name]
            pos = component_signals[name]
            # Weighted contribution: w_i(t) * position_i(t, ticker)
            for ticker in tickers:
                if ticker in pos.columns:
                    ensemble[ticker] += w * pos[ticker]

        # ------------------------------------------------------------------
        # Step 5: apply leverage multiplier.
        # ------------------------------------------------------------------
        ensemble *= self.leverage

        return ensemble

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_signals(
        signals: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert a strategy's signal output into a position DataFrame
        with one column per ticker, aligned to the prices index.

        Handles the diverse signal conventions used by the 5 components:
          - Direct ticker columns (HMM, Spectral, GARCH)
          - ``{ticker}_signal`` / ``{ticker}_weight`` (Entropy, Bayesian)
          - ``signal`` / ``weight`` single-asset shorthand
        """
        tickers = list(prices.columns)
        result = pd.DataFrame(0.0, index=prices.index, columns=tickers)

        # Pattern 1: columns that match ticker names directly.
        direct = [c for c in signals.columns if c in tickers]
        if direct:
            for c in direct:
                result[c] = signals[c].reindex(prices.index, fill_value=0.0).fillna(0.0)
            return result

        # Pattern 2: {ticker}_signal and {ticker}_weight columns.
        has_signal_weight = False
        for ticker in tickers:
            scol = f"{ticker}_signal"
            wcol = f"{ticker}_weight"
            if scol in signals.columns and wcol in signals.columns:
                has_signal_weight = True
                sig = signals[scol].reindex(prices.index, fill_value=0.0).fillna(0.0)
                wgt = signals[wcol].reindex(prices.index, fill_value=0.0).fillna(0.0)
                result[ticker] = sig * wgt
            elif scol in signals.columns:
                has_signal_weight = True
                result[ticker] = (
                    signals[scol]
                    .reindex(prices.index, fill_value=0.0)
                    .fillna(0.0)
                )
        if has_signal_weight:
            return result

        # Pattern 3: single-asset 'signal' / 'weight' columns.
        if "signal" in signals.columns:
            sig = signals["signal"].reindex(prices.index, fill_value=0.0).fillna(0.0)
            if "weight" in signals.columns:
                wgt = signals["weight"].reindex(prices.index, fill_value=0.0).fillna(0.0)
                combined = sig * wgt
            else:
                combined = sig
            # Apply equally across tickers (equal-weight within the signal).
            n_tickers = len(tickers)
            for ticker in tickers:
                result[ticker] = combined / n_tickers
            return result

        # Pattern 4: fallback -- average all numeric columns.
        numeric = signals.select_dtypes(include="number")
        if not numeric.empty:
            avg = numeric.reindex(prices.index).fillna(0.0).mean(axis=1)
            n_tickers = len(tickers)
            for ticker in tickers:
                result[ticker] = avg / n_tickers

        return result

    def _compute_inverse_vol_weights(
        self,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute time-varying inverse-volatility weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily portfolio returns, one column per component strategy.

        Returns
        -------
        pd.DataFrame
            Same shape as *returns*.  Each row sums to 1.  Weights are
            floored at ``self.min_weight`` and renormalised.
        """
        n_strats = returns.shape[1]

        # Rolling realised volatility (annualised is not needed since we
        # only care about relative magnitudes for weighting).
        rolling_vol = returns.rolling(
            window=self.vol_lookback,
            min_periods=max(10, self.vol_lookback // 3),
        ).std()

        # Replace zero / NaN vols with NaN to avoid division issues.
        rolling_vol = rolling_vol.replace(0, np.nan)

        # Inverse volatility.
        inv_vol = 1.0 / rolling_vol

        # Where we have no vol estimate, fall back to equal weight.
        inv_vol = inv_vol.fillna(0.0)

        # Normalise row-wise.
        row_sum = inv_vol.sum(axis=1)
        # Avoid division by zero for early rows.
        row_sum = row_sum.replace(0, np.nan)
        weights = inv_vol.div(row_sum, axis=0)

        # For rows where all vols are NaN, use equal weight.
        equal_w = 1.0 / n_strats
        weights = weights.fillna(equal_w)

        # Apply minimum weight floor and renormalise.
        weights = weights.clip(lower=self.min_weight)
        row_sum = weights.sum(axis=1)
        weights = weights.div(row_sum, axis=0)

        return weights
