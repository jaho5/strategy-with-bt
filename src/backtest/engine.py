"""Vectorized backtesting engine with walk-forward validation and Monte Carlo analysis.

Provides realistic strategy evaluation through:
- Vectorized PnL computation with configurable transaction costs
- Comprehensive risk-adjusted performance metrics
- Walk-forward (rolling out-of-sample) validation
- Block-bootstrap Monte Carlo confidence intervals
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


@dataclass(frozen=True)
class TransactionCosts:
    """Fee model applied on each trade.

    Attributes:
        slippage_bps: One-way slippage in basis points (applied on position changes).
        commission_bps: One-way commission in basis points (applied on position changes).
    """

    slippage_bps: float = 5.0
    commission_bps: float = 1.0

    @property
    def total_bps(self) -> float:
        return self.slippage_bps + self.commission_bps

    def cost_fraction(self) -> float:
        """Total one-way cost as a decimal fraction."""
        return self.total_bps / 10_000.0


@dataclass
class BacktestResult:
    """Container for a single backtest run's outputs.

    Attributes:
        equity_curve: Equity value at each bar.
        returns: Per-bar simple returns of the strategy equity curve.
        positions: Position series used during the run.
        metrics: Dictionary of computed performance / risk metrics.
        costs_paid: Cumulative transaction costs paid (in currency units).
    """

    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    metrics: Dict[str, Any] = field(default_factory=dict)
    costs_paid: float = 0.0


@dataclass
class WalkForwardResult:
    """Container for walk-forward analysis outputs.

    Attributes:
        fold_results: Per-fold ``BacktestResult`` on out-of-sample segments.
        fold_metrics: Per-fold metric dictionaries.
        aggregate_metrics: Metrics computed over the concatenated OOS returns.
        train_indices: List of (start, end) index tuples for each training window.
        test_indices: List of (start, end) index tuples for each testing window.
    """

    fold_results: List[BacktestResult]
    fold_metrics: List[Dict[str, Any]]
    aggregate_metrics: Dict[str, Any]
    train_indices: List[Tuple[int, int]]
    test_indices: List[Tuple[int, int]]


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo bootstrap analysis.

    Attributes:
        terminal_wealth: Array of simulated terminal wealth values.
        confidence_intervals: Dict mapping confidence level to (lower, upper) bounds.
        prob_above_target: Probability that terminal wealth exceeds the target PnL.
        mean_terminal: Mean terminal wealth across simulations.
        median_terminal: Median terminal wealth across simulations.
    """

    terminal_wealth: np.ndarray
    confidence_intervals: Dict[float, Tuple[float, float]]
    prob_above_target: float
    mean_terminal: float
    median_terminal: float


@dataclass
class DrawdownStats:
    """Drawdown statistics.

    Attributes:
        max_drawdown: Largest peak-to-trough decline (positive number, as a fraction).
        avg_drawdown: Average drawdown across all drawdown episodes.
        max_drawdown_duration: Duration in bars of the longest drawdown episode.
        avg_drawdown_duration: Average duration in bars across episodes.
        drawdown_series: Per-bar drawdown fraction from the running peak.
    """

    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    avg_drawdown_duration: float
    drawdown_series: np.ndarray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRADING_DAYS_PER_YEAR = 252


class BacktestEngine:
    """Vectorized backtesting engine.

    Parameters:
        costs: Transaction cost model.  Defaults to 5 bps slippage + 1 bp commission.
        risk_free_rate: Annualized risk-free rate used in Sharpe / Sortino calculations.
    """

    def __init__(
        self,
        costs: Optional[TransactionCosts] = None,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.costs = costs or TransactionCosts()
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Core backtest
    # ------------------------------------------------------------------

    def run(
        self,
        strategy_signals: np.ndarray,
        prices: np.ndarray,
        initial_capital: float = 100_000.0,
    ) -> BacktestResult:
        """Run a vectorized backtest.

        Parameters:
            strategy_signals: Array of target positions at each bar.  Values are
                interpreted as fractional allocation of capital (e.g. 1.0 = 100 %
                long, -0.5 = 50 % short, 0 = flat).  Length must equal ``len(prices)``.
            prices: Array of asset prices (e.g. close prices), one per bar.
            initial_capital: Starting equity in currency units.

        Returns:
            A ``BacktestResult`` with equity curve, returns, positions and metrics.

        Raises:
            ValueError: If input arrays have mismatched lengths or are too short.
        """
        strategy_signals = np.asarray(strategy_signals, dtype=np.float64)
        prices = np.asarray(prices, dtype=np.float64)

        if len(strategy_signals) != len(prices):
            raise ValueError(
                f"strategy_signals length ({len(strategy_signals)}) must equal "
                f"prices length ({len(prices)})"
            )
        if len(prices) < 2:
            raise ValueError("Need at least 2 price observations to run a backtest")

        # Asset returns (simple)
        asset_returns = np.diff(prices) / prices[:-1]  # length n-1

        # Positions are lagged by 1 bar: signal at bar t is executed at bar t+1.
        # So the return earned between bar t and t+1 uses the position from bar t.
        positions = strategy_signals[:-1]  # length n-1

        # --- Transaction costs ---
        # Position changes incur costs proportional to turnover.
        pos_changes = np.abs(np.diff(positions))  # length n-2
        # First bar has an implicit trade from 0 to positions[0].
        first_trade = np.abs(positions[0])
        turnover = np.empty(len(positions))
        turnover[0] = first_trade
        turnover[1:] = pos_changes

        cost_frac = self.costs.cost_fraction()
        per_bar_cost = turnover * cost_frac  # fraction of equity lost to costs

        # --- Strategy returns ---
        gross_returns = positions * asset_returns
        net_returns = gross_returns - per_bar_cost

        # --- Equity curve ---
        equity = np.empty(len(prices))
        equity[0] = initial_capital
        equity[1:] = initial_capital * np.cumprod(1.0 + net_returns)

        total_costs = float(np.sum(per_bar_cost * equity[:-1]))

        # Pad returns so length matches prices (return at index 0 is 0).
        full_returns = np.empty(len(prices))
        full_returns[0] = 0.0
        full_returns[1:] = net_returns

        full_positions = np.empty(len(prices))
        full_positions[:-1] = positions
        full_positions[-1] = strategy_signals[-1]

        metrics = self.calculate_metrics(full_returns)

        return BacktestResult(
            equity_curve=equity,
            returns=full_returns,
            positions=full_positions,
            metrics=metrics,
            costs_paid=total_costs,
        )

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def calculate_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive performance metrics from a return series.

        Parameters:
            returns: Per-bar simple returns (can include a leading zero).
            benchmark_returns: Optional benchmark return series of the same length,
                used for information ratio.  If *None*, information ratio is reported
                as ``NaN``.

        Returns:
            Dictionary with the following keys (among others):

            - ``total_pnl_pct``: Cumulative PnL as a percentage.
            - ``annualized_return``: CAGR approximation.
            - ``sharpe_ratio``: Annualized Sharpe ratio.
            - ``sortino_ratio``: Annualized Sortino ratio.
            - ``max_drawdown``: Maximum drawdown (positive fraction).
            - ``calmar_ratio``: Annualized return / max drawdown.
            - ``win_rate``: Fraction of bars with positive return.
            - ``profit_factor``: Sum of gains / abs(sum of losses).
            - ``ttest_pvalue``: p-value of one-sample t-test (H0: mean return = 0).
            - ``bootstrap_pvalue``: Bootstrap p-value for the mean return.
            - ``information_ratio``: Annualized information ratio vs benchmark.
        """
        returns = np.asarray(returns, dtype=np.float64)
        n = len(returns)

        # Cumulative
        cum = np.prod(1.0 + returns) - 1.0
        total_pnl_pct = cum * 100.0

        # Annualized return
        years = n / _TRADING_DAYS_PER_YEAR
        if years > 0 and (1.0 + cum) > 0:
            ann_return = (1.0 + cum) ** (1.0 / years) - 1.0
        else:
            ann_return = 0.0

        # Daily risk-free rate
        daily_rf = (1.0 + self.risk_free_rate) ** (1.0 / _TRADING_DAYS_PER_YEAR) - 1.0

        excess = returns - daily_rf
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1) if n > 1 else np.nan

        # Sharpe
        if std_excess and std_excess > 0:
            sharpe = (mean_excess / std_excess) * np.sqrt(_TRADING_DAYS_PER_YEAR)
        else:
            sharpe = np.nan

        # Sortino
        downside = excess.copy()
        downside[downside > 0] = 0.0
        downside_std = np.sqrt(np.mean(downside ** 2))
        if downside_std > 0:
            sortino = (mean_excess / downside_std) * np.sqrt(_TRADING_DAYS_PER_YEAR)
        else:
            sortino = np.nan

        # Drawdown
        dd_stats = self.compute_drawdown_stats(returns)

        # Calmar
        if dd_stats.max_drawdown > 0:
            calmar = ann_return / dd_stats.max_drawdown
        else:
            calmar = np.nan

        # Win rate / profit factor
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(gains) / n if n > 0 else 0.0
        sum_gains = float(np.sum(gains))
        sum_losses = float(np.abs(np.sum(losses)))
        profit_factor = sum_gains / sum_losses if sum_losses > 0 else np.inf

        # T-test
        if n > 1 and np.std(returns, ddof=1) > 0:
            t_stat, t_pval = sp_stats.ttest_1samp(returns, 0.0)
        else:
            t_stat, t_pval = np.nan, np.nan

        # Bootstrap p-value: fraction of bootstrap samples with mean <= 0
        bootstrap_pval = self._bootstrap_pvalue(returns)

        # Information ratio
        info_ratio = np.nan
        if benchmark_returns is not None:
            benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)
            active = returns - benchmark_returns
            te = np.std(active, ddof=1)
            if te > 0:
                info_ratio = (np.mean(active) / te) * np.sqrt(_TRADING_DAYS_PER_YEAR)

        return {
            "total_pnl_pct": total_pnl_pct,
            "annualized_return": ann_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": dd_stats.max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "ttest_statistic": float(t_stat) if not np.isnan(t_stat) else t_stat,
            "ttest_pvalue": float(t_pval) if not np.isnan(t_pval) else t_pval,
            "bootstrap_pvalue": bootstrap_pval,
            "information_ratio": info_ratio,
            "max_drawdown_duration": dd_stats.max_drawdown_duration,
            "avg_drawdown": dd_stats.avg_drawdown,
            "avg_drawdown_duration": dd_stats.avg_drawdown_duration,
            "n_bars": n,
        }

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward_test(
        self,
        strategy_fn: Callable[[np.ndarray], np.ndarray],
        prices: np.ndarray,
        n_splits: int = 5,
        train_pct: float = 0.7,
        initial_capital: float = 100_000.0,
    ) -> WalkForwardResult:
        """Walk-forward (rolling out-of-sample) validation.

        The price series is divided into ``n_splits`` consecutive folds.  For each
        fold the first ``train_pct`` fraction is used for training and the remainder
        for out-of-sample testing.  This is the *anchored* variant where each fold
        is an independent, non-overlapping window.

        Parameters:
            strategy_fn: A callable ``f(train_prices) -> signals`` that accepts a
                training price array and returns a signal array **of the same length
                as the full prices array** (or at least covering the test segment).
                In practice the function should fit / calibrate on the training
                window and then produce signals for the subsequent test window.

                More precisely, ``strategy_fn`` receives a dict with keys:

                - ``"train_prices"``: prices in the training window
                - ``"full_prices"``: the entire price array
                - ``"test_start"``: integer index where the test window begins
                - ``"test_end"``: integer index where the test window ends (exclusive)

                and must return an array of signals for the **test window only**
                (length = ``test_end - test_start``).
            prices: Full price array.
            n_splits: Number of non-overlapping folds.
            train_pct: Fraction of each fold used for training (0 < train_pct < 1).
            initial_capital: Starting capital for each fold's backtest.

        Returns:
            ``WalkForwardResult`` with per-fold and aggregate metrics.

        Raises:
            ValueError: If ``n_splits < 1`` or ``train_pct`` is out of range.
        """
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)

        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not 0 < train_pct < 1:
            raise ValueError("train_pct must be in (0, 1)")

        fold_size = n // n_splits
        if fold_size < 4:
            raise ValueError(
                f"Not enough data: {n} bars / {n_splits} splits = {fold_size} bars per fold"
            )

        train_indices: List[Tuple[int, int]] = []
        test_indices: List[Tuple[int, int]] = []
        fold_results: List[BacktestResult] = []
        fold_metrics: List[Dict[str, Any]] = []
        all_oos_returns: List[np.ndarray] = []

        for i in range(n_splits):
            fold_start = i * fold_size
            fold_end = (i + 1) * fold_size if i < n_splits - 1 else n

            train_len = int((fold_end - fold_start) * train_pct)
            train_start = fold_start
            train_end = fold_start + train_len
            test_start = train_end
            test_end = fold_end

            if test_end - test_start < 2:
                warnings.warn(
                    f"Fold {i}: test window too small ({test_end - test_start} bars), skipping."
                )
                continue

            train_indices.append((train_start, train_end))
            test_indices.append((test_start, test_end))

            # Call strategy function
            context = {
                "train_prices": prices[train_start:train_end],
                "full_prices": prices,
                "test_start": test_start,
                "test_end": test_end,
            }
            try:
                test_signals = np.asarray(strategy_fn(context), dtype=np.float64)
            except Exception as exc:
                warnings.warn(f"Fold {i}: strategy_fn raised {exc!r}, using flat signals.")
                test_signals = np.zeros(test_end - test_start)
            test_prices = prices[test_start:test_end]

            if len(test_signals) != len(test_prices):
                # Pad or truncate to match
                if len(test_signals) < len(test_prices):
                    test_signals = np.pad(test_signals, (0, len(test_prices) - len(test_signals)), constant_values=0.0)
                else:
                    test_signals = test_signals[:len(test_prices)]

            result = self.run(test_signals, test_prices, initial_capital=initial_capital)
            fold_results.append(result)
            fold_metrics.append(result.metrics)
            # Skip the leading zero return
            all_oos_returns.append(result.returns[1:])

        if not all_oos_returns:
            raise ValueError("No valid folds produced; data may be too short.")

        concat_returns = np.concatenate(all_oos_returns)
        aggregate_metrics = self.calculate_metrics(concat_returns)

        return WalkForwardResult(
            fold_results=fold_results,
            fold_metrics=fold_metrics,
            aggregate_metrics=aggregate_metrics,
            train_indices=train_indices,
            test_indices=test_indices,
        )

    # ------------------------------------------------------------------
    # Monte Carlo bootstrap
    # ------------------------------------------------------------------

    def monte_carlo_confidence(
        self,
        returns: np.ndarray,
        n_simulations: int = 10_000,
        confidence: float = 0.95,
        initial_capital: float = 100_000.0,
        target_pnl_pct: float = 45.0,
        rng_seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """Block-bootstrap Monte Carlo simulation for terminal wealth confidence intervals.

        Uses a block bootstrap to preserve serial correlation in returns.  Block
        size defaults to ``max(1, int(sqrt(n)))``.

        Parameters:
            returns: Observed strategy return series.
            n_simulations: Number of bootstrap paths to generate.
            confidence: Primary confidence level reported (also always reports
                90 %, 95 %, and 99 %).
            initial_capital: Starting equity for each simulated path.
            target_pnl_pct: Target cumulative PnL percentage; the method reports
                the probability of exceeding this threshold.
            rng_seed: Optional seed for reproducibility.

        Returns:
            ``MonteCarloResult`` with terminal wealth distribution and statistics.
        """
        returns = np.asarray(returns, dtype=np.float64)
        # Strip any leading zeros (e.g. from the backtest result padding)
        if len(returns) > 0 and returns[0] == 0.0:
            returns = returns[1:]

        n = len(returns)
        if n < 2:
            raise ValueError("Need at least 2 return observations for Monte Carlo.")

        rng = np.random.default_rng(rng_seed)
        block_size = max(1, int(np.sqrt(n)))

        # Pre-compute how many blocks we need to fill a path of length n
        n_blocks = int(np.ceil(n / block_size))

        terminal_wealth = np.empty(n_simulations)

        for sim in range(n_simulations):
            # Draw random block start indices
            starts = rng.integers(0, n - block_size + 1, size=n_blocks)
            # Build the bootstrapped return path
            blocks = [returns[s : s + block_size] for s in starts]
            path = np.concatenate(blocks)[:n]
            terminal_wealth[sim] = initial_capital * np.prod(1.0 + path)

        # Confidence intervals
        ci_levels = sorted(set([0.90, 0.95, 0.99, confidence]))
        confidence_intervals: Dict[float, Tuple[float, float]] = {}
        for level in ci_levels:
            alpha = 1.0 - level
            lo = float(np.percentile(terminal_wealth, 100 * alpha / 2))
            hi = float(np.percentile(terminal_wealth, 100 * (1 - alpha / 2)))
            confidence_intervals[level] = (lo, hi)

        # Probability of exceeding target PnL
        target_wealth = initial_capital * (1.0 + target_pnl_pct / 100.0)
        prob_above = float(np.mean(terminal_wealth >= target_wealth))

        return MonteCarloResult(
            terminal_wealth=terminal_wealth,
            confidence_intervals=confidence_intervals,
            prob_above_target=prob_above,
            mean_terminal=float(np.mean(terminal_wealth)),
            median_terminal=float(np.median(terminal_wealth)),
        )

    # ------------------------------------------------------------------
    # Drawdown analysis
    # ------------------------------------------------------------------

    def compute_drawdown_stats(self, returns: np.ndarray) -> DrawdownStats:
        """Compute drawdown statistics from a return series.

        Parameters:
            returns: Per-bar simple returns.

        Returns:
            ``DrawdownStats`` with max / average drawdown and duration information.
        """
        returns = np.asarray(returns, dtype=np.float64)
        n = len(returns)

        if n == 0:
            return DrawdownStats(
                max_drawdown=0.0,
                avg_drawdown=0.0,
                max_drawdown_duration=0,
                avg_drawdown_duration=0.0,
                drawdown_series=np.array([]),
            )

        # Build equity index (starting at 1.0)
        equity = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdown = 1.0 - equity / running_max  # positive values = drawdown

        max_dd = float(np.max(drawdown)) if n > 0 else 0.0

        # Identify drawdown episodes (contiguous regions where drawdown > 0)
        in_dd = drawdown > 0.0
        episode_depths: List[float] = []
        episode_durations: List[int] = []
        current_duration = 0
        current_peak_dd = 0.0

        for i in range(n):
            if in_dd[i]:
                current_duration += 1
                current_peak_dd = max(current_peak_dd, drawdown[i])
            else:
                if current_duration > 0:
                    episode_durations.append(current_duration)
                    episode_depths.append(current_peak_dd)
                current_duration = 0
                current_peak_dd = 0.0
        # Handle case where series ends in a drawdown
        if current_duration > 0:
            episode_durations.append(current_duration)
            episode_depths.append(current_peak_dd)

        if episode_durations:
            avg_dd = float(np.mean(episode_depths))
            max_dd_dur = int(np.max(episode_durations))
            avg_dd_dur = float(np.mean(episode_durations))
        else:
            avg_dd = 0.0
            max_dd_dur = 0
            avg_dd_dur = 0.0

        return DrawdownStats(
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_dd_dur,
            avg_drawdown_duration=avg_dd_dur,
            drawdown_series=drawdown,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bootstrap_pvalue(
        returns: np.ndarray,
        n_bootstrap: int = 5000,
        rng_seed: int = 42,
    ) -> float:
        """Compute a bootstrap p-value for the null hypothesis that mean return <= 0.

        Resamples the returns with replacement, computes the mean of each sample,
        and returns the fraction of bootstrap means that are <= 0.

        Parameters:
            returns: Observed return series.
            n_bootstrap: Number of bootstrap resamples.
            rng_seed: Seed for reproducibility.

        Returns:
            p-value (float in [0, 1]).
        """
        returns = np.asarray(returns, dtype=np.float64)
        n = len(returns)
        if n < 2:
            return np.nan

        rng = np.random.default_rng(rng_seed)
        observed_mean = np.mean(returns)

        # Center the returns under H0 (mean = 0)
        centered = returns - observed_mean
        boot_means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(centered, size=n, replace=True)
            boot_means[i] = np.mean(sample)

        # p-value: fraction of bootstrap means >= observed mean (one-sided)
        p_value = float(np.mean(boot_means >= observed_mean))
        return p_value
