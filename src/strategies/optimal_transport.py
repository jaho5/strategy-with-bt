"""Cross-sectional momentum strategy based on Wasserstein distance.

Uses optimal transport theory to measure distributional shifts in asset
returns and construct a long-short portfolio from the cross-sectional
ranking of signed Earth Mover's Distances.

Mathematical foundation
-----------------------
Wasserstein-1 distance between probability measures P and Q on R:

    W_1(P, Q) = inf_{gamma in Gamma(P,Q)} E_{(x,y)~gamma}[|x - y|]

For univariate distributions this reduces to the L1 distance between CDFs:

    W_1 = int |F_P(x) - F_Q(x)| dx

The strategy exploits the fact that W_1 captures *all* moments of the
distributional shift (unlike simple mean-return momentum), while remaining
metrically well-behaved (it is a true distance on the space of probability
measures with finite first moment).

A distributional quality filter based on the 2-Wasserstein distance
eliminates stocks whose shift is purely variance-driven:

    W_2^2 = (mu_1 - mu_2)^2 + (sigma_1 - sigma_2)^2 + 2 sigma_1 sigma_2 (1 - rho)

where rho is the correlation between the two windows (set to 0 for
independent windows, giving the upper bound).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from src.strategies.base import Strategy


class OptimalTransportMomentum(Strategy):
    """Long-short momentum strategy ranked by signed Wasserstein distance.

    Parameters
    ----------
    recent_window : int
        Number of trading days for the recent return distribution
        (default 21, roughly one month).
    hist_start : int
        Start of the historical comparison window, measured in trading
        days back from the current date (default 252, one year).
    hist_end : int
        End of the historical comparison window (default 63, three months).
        The historical window spans ``[hist_start, hist_end)`` days back.
    rebalance_freq : int
        Rebalance every *rebalance_freq* trading days (default 5, weekly).
    turnover_limit : float
        Maximum fraction of the portfolio that may turn over at each
        rebalance (default 0.30).
    quintile_frac : float
        Fraction of the cross-section to include in each tail (default
        0.20, i.e. quintiles).
    quality_filter : bool
        If True (default), filter out stocks whose distributional shift
        is dominated by variance rather than mean.
    quality_threshold : float
        Minimum fraction of W2 shift attributable to the mean for the
        quality filter.  Default 0.0 (effectively disabled for ETF
        universes where variance shifts dominate due to small mean
        returns relative to volatility).
    min_assets_for_ranking : int
        Minimum number of valid assets needed to form a long-short
        portfolio.  Default 3 (suitable for small ETF universes).
    """

    def __init__(
        self,
        recent_window: int = 21,
        hist_start: int = 252,
        hist_end: int = 63,
        rebalance_freq: int = 5,
        turnover_limit: float = 0.30,
        quintile_frac: float = 0.20,
        quality_filter: bool = True,
        quality_threshold: float = 0.0,
        min_assets_for_ranking: int = 3,
    ) -> None:
        super().__init__(
            name="OptimalTransport",
            description=(
                "Cross-sectional momentum strategy ranked by signed "
                "Wasserstein distance (optimal transport) between recent "
                "and historical return distributions."
            ),
        )
        self.recent_window = recent_window
        self.hist_start = hist_start
        self.hist_end = hist_end
        self.rebalance_freq = rebalance_freq
        self.turnover_limit = turnover_limit
        self.quintile_frac = quintile_frac
        self.quality_filter = quality_filter
        self.quality_threshold = quality_threshold
        self.min_assets_for_ranking = min_assets_for_ranking

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame, **kwargs) -> "OptimalTransportMomentum":
        """Validate that sufficient history is available.

        No heavy calibration is required: the strategy is non-parametric
        and recomputes distributional distances on each rebalance date.
        """
        self.validate_prices(prices)
        min_rows = self.hist_start + self.recent_window
        if len(prices) < min_rows:
            raise ValueError(
                f"Need at least {min_rows} rows of price data, got {len(prices)}."
            )
        self._fitted = True
        return self

    def generate_signals(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Produce per-asset position signals.

        Returns
        -------
        pd.DataFrame
            Columns ``{ticker}_signal`` and ``{ticker}_weight`` for each
            ticker.  Signals are +1 (long), -1 (short), or 0 (flat).
            Weights reflect the equal-weight allocation within each leg.
        """
        self.ensure_fitted()
        self.validate_prices(prices)

        returns = prices.pct_change()
        # Internal weight DataFrame (raw position weights per ticker)
        raw_signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        prev_weights: pd.Series | None = None

        rebalance_dates = prices.index[self.hist_start :: self.rebalance_freq]

        for date in rebalance_dates:
            loc = prices.index.get_loc(date)

            computed_weights = self._compute_weights(returns, loc)
            if computed_weights is None:
                # Not enough valid stocks -- carry forward.
                if prev_weights is not None:
                    raw_signals.loc[date] = prev_weights
                continue

            # Apply turnover constraint relative to previous weights.
            if prev_weights is not None:
                computed_weights = self._apply_turnover_constraint(
                    prev_weights, computed_weights
                )

            raw_signals.loc[date] = computed_weights
            prev_weights = computed_weights

        # Forward-fill signals between rebalance dates (from hist_start on).
        # Only treat exact 0.0 rows (non-rebalance dates) as gaps.
        # A row is a non-rebalance gap if ALL columns are 0.
        mask = (raw_signals == 0.0).all(axis=1)
        raw_signals[mask] = np.nan
        raw_signals = raw_signals.ffill()
        raw_signals = raw_signals.fillna(0.0)

        # Convert to {ticker}_signal / {ticker}_weight format
        result = pd.DataFrame(index=prices.index)
        for col in prices.columns:
            w = raw_signals[col]
            result[f"{col}_signal"] = np.sign(w).astype(float)
            result[f"{col}_weight"] = w.abs()

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_weights(
        self, returns: pd.DataFrame, loc: int
    ) -> pd.Series | None:
        """Rank stocks by signed W_1 and return equal-weighted quintile weights."""
        recent_returns = returns.iloc[loc - self.recent_window + 1 : loc + 1]
        hist_returns = returns.iloc[loc - self.hist_start : loc - self.hist_end]

        signed_w1: dict[str, float] = {}
        mean_shift_ratio: dict[str, float] = {}

        for ticker in returns.columns:
            r_recent = recent_returns[ticker].dropna().values
            r_hist = hist_returns[ticker].dropna().values

            # Need a minimum number of observations for a meaningful CDF.
            if len(r_recent) < 10 or len(r_hist) < 20:
                continue

            # W_1 via scipy (exact 1-D optimal transport).
            w1 = wasserstein_distance(r_recent, r_hist)

            # Sign: direction of mean shift.
            mu_recent = r_recent.mean()
            mu_hist = r_hist.mean()
            sign = 1.0 if mu_recent >= mu_hist else -1.0

            signed_w1[ticker] = sign * w1

            # ---- Distributional quality filter (W_2 decomposition) ----
            if self.quality_filter:
                sigma_recent = r_recent.std(ddof=1)
                sigma_hist = r_hist.std(ddof=1)

                # W_2^2 upper bound (independent windows => rho = 0):
                #   W_2^2 = (mu_1 - mu_2)^2 + (sigma_1 - sigma_2)^2
                #           + 2 * sigma_1 * sigma_2 * (1 - rho)
                # With rho = 0 this simplifies to:
                #   (mu_1-mu_2)^2 + (sigma_1+sigma_2)^2 - 2*sigma_1*sigma_2
                #   ... but we keep the explicit formula for clarity.
                mean_shift_sq = (mu_recent - mu_hist) ** 2
                var_shift_sq = (sigma_recent - sigma_hist) ** 2
                cross_term = 2.0 * sigma_recent * sigma_hist  # (1 - 0)

                w2_sq = mean_shift_sq + var_shift_sq + cross_term
                # Fraction of W_2^2 attributable to mean shift.
                if w2_sq > 0:
                    mean_shift_ratio[ticker] = mean_shift_sq / w2_sq
                else:
                    mean_shift_ratio[ticker] = 0.0

        if len(signed_w1) < self.min_assets_for_ranking:
            return None

        scores = pd.Series(signed_w1)

        # Apply quality filter: drop stocks where mean shift is below
        # the quality threshold of the total W_2 shift.
        if self.quality_filter and mean_shift_ratio:
            quality = pd.Series(mean_shift_ratio)
            quality_mask = quality >= self.quality_threshold
            scores = scores[quality_mask]
            if len(scores) < self.min_assets_for_ranking:
                return None

        # Cross-sectional ranking into quintiles.
        n = len(scores)
        k_long = max(1, int(np.ceil(n * self.quintile_frac)))
        k_short = max(1, int(np.ceil(n * self.quintile_frac)))

        ranked = scores.sort_values()
        short_tickers = ranked.index[:k_short]
        long_tickers = ranked.index[-k_long:]

        weights = pd.Series(0.0, index=returns.columns)
        weights[long_tickers] = 0.5 / k_long
        weights[short_tickers] = -0.5 / k_short

        return weights

    def _apply_turnover_constraint(
        self, prev: pd.Series, target: pd.Series
    ) -> pd.Series:
        """Shrink *target* towards *prev* so turnover <= self.turnover_limit.

        Turnover is defined as 0.5 * sum(|w_new - w_old|), i.e. the
        fraction of gross exposure that changes hands.
        """
        diff = target - prev
        turnover = 0.5 * diff.abs().sum()

        if turnover <= self.turnover_limit:
            return target

        # Scale the change so that turnover equals the limit exactly.
        scale = self.turnover_limit / turnover
        constrained = prev + scale * diff
        return constrained
