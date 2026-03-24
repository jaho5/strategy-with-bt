"""Main orchestration script for running all strategies through backtesting.

Downloads ETF data, runs each strategy through the backtesting engine with
walk-forward validation and Monte Carlo analysis, then produces a comparison
report.

Usage:
    uv run python -m src.main
"""

from __future__ import annotations

import inspect
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.backtest.engine import BacktestEngine, BacktestResult, MonteCarloResult, WalkForwardResult
from src.data.downloader import download_etf_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_START = "2010-01-01"
DATA_END = "2025-12-31"
TRAIN_FRACTION = 0.70
WALK_FORWARD_FOLDS = 5
MONTE_CARLO_SIMS = 10_000
ANNUAL_TARGET_PNL = 45.0  # 45% per year
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
def _load_strategies() -> List[Tuple[str, Any]]:
    """Import and instantiate all strategies.

    Returns a list of (name, strategy_instance) tuples.  If a strategy
    module cannot be imported (e.g. not yet implemented), the error is
    logged and that strategy is skipped.
    """
    registry: List[Tuple[str, type, dict]] = [
        ("OU Mean Reversion", "src.strategies.ou_mean_reversion", "OUMeanReversionStrategy", {}),
        ("HMM Regime", "src.strategies.hmm_regime", "HMMRegimeStrategy", {}),
        ("Kalman Alpha", "src.strategies.kalman_alpha", "KalmanAlphaStrategy", {}),
        ("Spectral Momentum", "src.strategies.spectral_momentum", "SpectralMomentumStrategy", {}),
        ("GARCH Vol", "src.strategies.garch_vol", "GarchVolStrategy", {}),
        ("Optimal Transport", "src.strategies.optimal_transport", "OptimalTransportMomentum", {}),
        ("Info Geometry", "src.strategies.info_geometry", "InformationGeometryStrategy", {}),
        ("Stochastic Control", "src.strategies.stochastic_control", "StochasticControlStrategy", {}),
        ("Conservative Stochastic Control", "src.strategies.stochastic_control", "ConservativeStochasticControlStrategy", {}),
        ("RMT Eigenportfolio", "src.strategies.rmt_eigenportfolio", "RMTEigenportfolioStrategy", {}),
        ("Entropy Regularized", "src.strategies.entropy_regularized", "EntropyRegularizedStrategy", {}),
        ("Fractional Differentiation", "src.strategies.fractional_differentiation", "FractionalDifferentiationStrategy", {}),
        ("Lévy Jump", "src.strategies.levy_jump", "LevyJumpStrategy", {}),
        ("Topological TDA", "src.strategies.topological", "TopologicalStrategy", {}),
        ("Rough Volatility", "src.strategies.rough_volatility", "RoughVolatilityStrategy", {}),
        ("Bayesian Changepoint", "src.strategies.bayesian_changepoint", "BayesianChangepointStrategy", {}),
        ("Sparse Mean Reversion", "src.strategies.sparse_mean_reversion", "SparseMeanReversionStrategy", {}),
        ("Momentum Crash Hedge", "src.strategies.momentum_crash_hedge", "MomentumCrashHedgeStrategy", {}),
        ("Kelly Growth Optimal", "src.strategies.kelly_growth", "KellyGrowthStrategy", {}),
        ("Microstructure", "src.strategies.microstructure", "MicrostructureStrategy", {}),
        ("Copula Dependence", "src.strategies.copula_dependence", "CopulaDependenceStrategy", {}),
        ("Martingale Difference", "src.strategies.martingale_difference", "MartingaleDifferenceStrategy", {}),
        ("Max Entropy Spectrum", "src.strategies.max_entropy_spectrum", "MaxEntropySpectrumStrategy", {}),
        ("Stein Shrinkage", "src.strategies.stein_shrinkage", "SteinShrinkageStrategy", {}),
        ("Concentration Bounds", "src.strategies.concentration_bounds", "ConcentrationBoundsStrategy", {}),
        ("Mean Field Game", "src.strategies.mean_field", "MeanFieldStrategy", {}),
        ("RKHS Regression", "src.strategies.rkhs_regression", "RKHSRegressionStrategy", {}),
        ("Ergodic Growth", "src.strategies.ergodic_growth", "ErgodicGrowthStrategy", {}),
        ("Benford's Law", "src.strategies.benfords_law", "BenfordsLawStrategy", {}),
        ("Rényi Entropy", "src.strategies.renyi_entropy", "RenyiEntropyStrategy", {}),
        ("Persistent Excursions", "src.strategies.persistent_excursions", "PersistentExcursionsStrategy", {}),
        ("Ensemble Meta", "src.strategies.ensemble_meta", "EnsembleMetaStrategy", {}),
        ("Malliavin Greeks", "src.strategies.malliavin_greeks", "MalliavinGreeksStrategy", {}),
        ("Large Deviations", "src.strategies.large_deviations", "LargeDeviationsStrategy", {}),
        ("Scoring Rules", "src.strategies.scoring_rules", "ScoringRulesStrategy", {}),
        ("Optimal Stopping", "src.strategies.optimal_stopping", "OptimalStoppingStrategy", {}),
        ("Sparse PCA Timing", "src.strategies.sparse_pca_timing", "SparsePCATimingStrategy", {}),
        ("Wasserstein Gradient", "src.strategies.wasserstein_gradient", "WassersteinGradientStrategy", {}),
        ("Semigroup Decay", "src.strategies.semigroup_decay", "SemigroupDecayStrategy", {}),
        ("Szegő Prediction", "src.strategies.szego_prediction", "SzegoPredictionStrategy", {}),
        ("Vol-Scaled Ensemble", "src.strategies.vol_scaled_ensemble", "VolScaledEnsembleStrategy", {}),
        ("Multi-Timeframe", "src.strategies.multi_timeframe", "MultiTimeframeStrategy", {}),
        ("Leveraged Trend", "src.strategies.leveraged_trend", "LeveragedTrendStrategy", {}),
    ]

    strategies = []
    for display_name, module_path, class_name, kwargs in registry:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            instance = cls(**kwargs)
            strategies.append((display_name, instance))
            logger.info("Loaded strategy: %s", display_name)
        except Exception as exc:
            logger.warning(
                "Could not load strategy '%s' (%s.%s): %s",
                display_name, module_path, class_name, exc,
            )
    return strategies


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------
def _extract_close_prices(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Extract close prices from the multi-level yfinance DataFrame.

    Returns a DataFrame indexed by date with one column per ticker.
    """
    if isinstance(raw_data.columns, pd.MultiIndex):
        # yfinance multi-ticker format: (Price, Ticker)
        if "Close" in raw_data.columns.get_level_values(0):
            close = raw_data["Close"]
        else:
            # Fall back to first price level
            first_level = raw_data.columns.get_level_values(0)[0]
            close = raw_data[first_level]
    else:
        close = raw_data

    # Drop rows where all values are NaN, forward-fill remaining gaps
    close = close.dropna(how="all").ffill().bfill()
    return close


def _extract_ohlcv_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw yfinance MultiIndex DataFrame for strategy consumption.

    Returns the full OHLCV MultiIndex DataFrame (fields x tickers) with NaN
    rows dropped and remaining gaps forward/back-filled.  Strategies that
    understand MultiIndex OHLCV data (e.g. Microstructure, GARCH Vol with
    Parkinson estimator) can use this directly.
    """
    if not isinstance(raw_data.columns, pd.MultiIndex):
        # Single-ticker or already flat — return as-is after basic cleaning
        return raw_data.dropna(how="all").ffill().bfill()

    # Keep only rows that have at least some data
    cleaned = raw_data.dropna(how="all").ffill().bfill()
    return cleaned


def _accepts_kwargs(method) -> bool:
    """Return True if *method* has a ``**kwargs`` parameter."""
    try:
        sig = inspect.signature(method)
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
    except (ValueError, TypeError):
        return False


def _split_data(
    data: pd.DataFrame, train_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and testing sets by row count."""
    n = len(data)
    split_idx = int(n * train_frac)
    return data.iloc[:split_idx], data.iloc[split_idx:]


def _signals_to_portfolio(signals: pd.DataFrame, prices: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a multi-asset signal DataFrame to a single portfolio signal and price series.

    Handles multiple signal column conventions:
    - Direct ticker names matching prices columns
    - '{ticker}_signal' / '{ticker}_weight' pattern
    - 'signal' / 'weight' single-column pattern

    Returns (portfolio_signals, portfolio_prices) as 1-D numpy arrays of
    the same length.
    """
    common_idx = signals.index.intersection(prices.index)
    if len(common_idx) == 0:
        common_idx = signals.index

    # Try to extract signal columns matching price tickers
    sig_cols = {}
    price_tickers = list(prices.columns)

    # Pattern 1: columns match price tickers directly
    direct_match = [c for c in signals.columns if c in price_tickers]
    if direct_match:
        for c in direct_match:
            sig_cols[c] = c

    # Pattern 2: '{ticker}_signal' columns
    if not sig_cols:
        for ticker in price_tickers:
            scol = f"{ticker}_signal"
            if scol in signals.columns:
                sig_cols[ticker] = scol

    # Pattern 3: single 'signal' column (applied to equal-weight portfolio)
    if not sig_cols and "signal" in signals.columns:
        sig_arr = signals["signal"].reindex(common_idx).fillna(0.0).values
        price_arr = prices.reindex(common_idx).ffill().bfill().mean(axis=1).values
        # Normalise price to start at 100
        if price_arr[0] != 0:
            price_arr = price_arr / price_arr[0] * 100.0
        return sig_arr, price_arr

    # Pattern 4: no match — average all numeric signal columns
    if not sig_cols:
        sig_arr = signals.reindex(common_idx).fillna(0.0).select_dtypes(include="number").mean(axis=1).values
        price_arr = prices.reindex(common_idx).ffill().bfill().mean(axis=1).values
        if len(price_arr) > 0 and price_arr[0] != 0:
            price_arr = price_arr / price_arr[0] * 100.0
        return sig_arr, price_arr

    # Build aligned signal and price DataFrames
    tickers_used = list(sig_cols.keys())
    sig_df = pd.DataFrame(index=common_idx)
    for ticker in tickers_used:
        sig_df[ticker] = signals[sig_cols[ticker]].reindex(common_idx).fillna(0.0)

    price_aligned = prices[tickers_used].reindex(common_idx).ffill().bfill()

    # Compute per-asset returns for signal-weighted portfolio construction
    returns = price_aligned.pct_change().fillna(0.0)

    # Detect whether this is a long-short (dollar-neutral) strategy or a
    # directional strategy by comparing net vs gross exposure.
    net_exposure = sig_df.sum(axis=1)
    gross_exposure = sig_df.abs().sum(axis=1)

    # Ratio of |net| / gross.  For dollar-neutral strategies this is near 0;
    # for fully directional strategies it is near 1.
    mean_gross = gross_exposure.mean()
    mean_abs_net = net_exposure.abs().mean()
    is_long_short = (mean_gross > 1e-8) and (mean_abs_net / mean_gross < 0.3)

    if is_long_short:
        # Long-short / dollar-neutral strategy (e.g. pairs trading).
        # Averaging signals would cancel long vs short legs.  Instead,
        # compute the portfolio return directly:
        #   r_portfolio(t+1) = sum_i [ signal_i(t) * r_i(t+1) ]
        # The signal at time t is the position held DURING the next
        # period, so it earns return(t+1).  We shift the signal by 1
        # to implement this 1-bar execution lag, matching the convention
        # used by BacktestEngine.run().
        lagged_signals = sig_df.shift(1).fillna(0.0)
        portfolio_returns = (lagged_signals * returns).sum(axis=1)
        portfolio_price = 100.0 * (1.0 + portfolio_returns).cumprod()
        portfolio_signal = np.ones(len(common_idx))
    else:
        # Directional strategy: average position across assets gives the
        # overall directional bias.  Portfolio price is equal-weight basket.
        portfolio_signal = sig_df.mean(axis=1).values
        portfolio_returns = returns.mean(axis=1)
        portfolio_price = 100.0 * (1.0 + portfolio_returns).cumprod()

    return portfolio_signal, portfolio_price.values


def _align_signal_price(signal: np.ndarray, price: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure signal and price arrays have the same length."""
    min_len = min(len(signal), len(price))
    return signal[:min_len], price[:min_len]


# ---------------------------------------------------------------------------
# Walk-forward wrapper
# ---------------------------------------------------------------------------
def _make_wf_strategy_fn(
    strategy,
    close_prices: pd.DataFrame,
    ohlcv_data: Optional[pd.DataFrame] = None,
):
    """Create a walk-forward callback compatible with BacktestEngine.walk_forward_test.

    The engine passes a dict with train_prices (1-D), full_prices (1-D),
    test_start, test_end.  We map that back to the multi-asset DataFrame
    to call strategy.fit / generate_signals and then collapse to a 1-D
    portfolio signal.

    When *ohlcv_data* is provided it is sliced in parallel with
    *close_prices* and passed as the ``ohlcv_data`` keyword argument to
    strategies whose ``fit`` / ``generate_signals`` accept ``**kwargs``.
    """
    pass_ohlcv_fit = ohlcv_data is not None and _accepts_kwargs(strategy.fit)
    pass_ohlcv_sig = ohlcv_data is not None and _accepts_kwargs(strategy.generate_signals)

    def strategy_fn(context: dict) -> np.ndarray:
        test_start = context["test_start"]
        test_end = context["test_end"]
        expected_len = test_end - test_start

        # Slice the multi-asset DataFrame to match the walk-forward windows
        train_data = close_prices.iloc[:test_start]
        test_data = close_prices.iloc[test_start:test_end]

        # Re-fit on training data, optionally passing full OHLCV
        if pass_ohlcv_fit:
            strategy.fit(train_data, ohlcv_data=ohlcv_data.iloc[:test_start])
        else:
            strategy.fit(train_data)

        # Generate signals on test data, optionally passing full OHLCV
        if pass_ohlcv_sig:
            signals = strategy.generate_signals(
                test_data, ohlcv_data=ohlcv_data.iloc[test_start:test_end],
            )
        else:
            signals = strategy.generate_signals(test_data)

        # Collapse to portfolio-level signal
        portfolio_signal, _ = _signals_to_portfolio(signals, test_data)

        # Ensure signal length matches expected window
        if len(portfolio_signal) < expected_len:
            portfolio_signal = np.pad(portfolio_signal, (0, expected_len - len(portfolio_signal)), constant_values=0.0)
        elif len(portfolio_signal) > expected_len:
            portfolio_signal = portfolio_signal[:expected_len]

        return portfolio_signal

    return strategy_fn


# ---------------------------------------------------------------------------
# Single-strategy runner
# ---------------------------------------------------------------------------
def run_single_strategy(
    name: str,
    strategy,
    close_prices: pd.DataFrame,
    engine: BacktestEngine,
    ohlcv_data: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    """Run a single strategy through the full pipeline.

    Parameters
    ----------
    name : str
        Human-readable strategy name.
    strategy : Strategy
        Strategy instance with ``fit`` and ``generate_signals`` methods.
    close_prices : pd.DataFrame
        Close-only price DataFrame (tickers x dates).
    engine : BacktestEngine
        Backtesting engine.
    ohlcv_data : pd.DataFrame, optional
        Full OHLCV MultiIndex DataFrame.  When provided and the strategy's
        ``fit`` / ``generate_signals`` accept ``**kwargs``, this is passed
        as the ``ohlcv_data`` keyword argument so that strategies can use
        Open, High, Low, Volume in addition to Close.

    Returns a dict of results or None on failure.
    """
    logger.info("=" * 70)
    logger.info("Running strategy: %s", name)
    logger.info("=" * 70)

    # Detect whether the strategy methods accept **kwargs for OHLCV pass-through
    pass_ohlcv_fit = ohlcv_data is not None and _accepts_kwargs(strategy.fit)
    pass_ohlcv_sig = ohlcv_data is not None and _accepts_kwargs(strategy.generate_signals)

    # 1. Split data
    train_data, test_data = _split_data(close_prices, TRAIN_FRACTION)
    if ohlcv_data is not None:
        ohlcv_train, ohlcv_test = _split_data(ohlcv_data, TRAIN_FRACTION)
    else:
        ohlcv_train = ohlcv_test = None

    logger.info(
        "  Train: %s to %s (%d bars)",
        train_data.index[0].date(), train_data.index[-1].date(), len(train_data),
    )
    logger.info(
        "  Test:  %s to %s (%d bars)",
        test_data.index[0].date(), test_data.index[-1].date(), len(test_data),
    )

    # 2. Fit (pass full OHLCV when the strategy can accept it)
    logger.info("  Fitting strategy on training data...")
    if pass_ohlcv_fit:
        strategy.fit(train_data, ohlcv_data=ohlcv_train)
    else:
        strategy.fit(train_data)

    # 3. Generate signals on test data (pass full OHLCV when accepted)
    logger.info("  Generating signals on test data...")
    if pass_ohlcv_sig:
        signals = strategy.generate_signals(test_data, ohlcv_data=ohlcv_test)
    else:
        signals = strategy.generate_signals(test_data)

    # 4. Convert to portfolio-level signal and price
    portfolio_signal, portfolio_price = _signals_to_portfolio(signals, test_data)
    portfolio_signal, portfolio_price = _align_signal_price(portfolio_signal, portfolio_price)

    if len(portfolio_signal) < 2 or len(portfolio_price) < 2:
        logger.warning("  Not enough data points after signal generation. Skipping.")
        return None

    # 5. Run backtest
    logger.info("  Running backtest...")
    bt_result: BacktestResult = engine.run(portfolio_signal, portfolio_price)
    metrics = bt_result.metrics

    # 6. Print individual results
    logger.info("  --- Backtest Results ---")
    logger.info("  Total PnL:          %.2f%%", metrics["total_pnl_pct"])
    logger.info("  Ann. Return:        %.2f%%", metrics["annualized_return"] * 100)
    logger.info("  Sharpe Ratio:       %.3f", metrics["sharpe_ratio"])
    logger.info("  Sortino Ratio:      %.3f", metrics["sortino_ratio"])
    logger.info("  Max Drawdown:       %.2f%%", metrics["max_drawdown"] * 100)
    logger.info("  Win Rate:           %.2f%%", metrics["win_rate"] * 100)
    logger.info("  Bootstrap p-value:  %.4f", metrics["bootstrap_pvalue"])

    # 7. Walk-forward test
    logger.info("  Running walk-forward test (%d folds)...", WALK_FORWARD_FOLDS)
    wf_strategy_fn = _make_wf_strategy_fn(strategy, close_prices, ohlcv_data=ohlcv_data)

    # Build the 1-D price series for walk-forward (equal-weight portfolio)
    _, full_portfolio_price = _signals_to_portfolio(
        pd.DataFrame(0.0, index=close_prices.index, columns=close_prices.columns),
        close_prices,
    )

    wf_result: WalkForwardResult = engine.walk_forward_test(
        strategy_fn=wf_strategy_fn,
        prices=full_portfolio_price,
        n_splits=WALK_FORWARD_FOLDS,
        train_pct=TRAIN_FRACTION,
    )
    wf_oos_sharpe = wf_result.aggregate_metrics.get("sharpe_ratio", np.nan)
    logger.info("  WF OOS Sharpe:      %.3f", wf_oos_sharpe)

    # 8. Monte Carlo confidence
    n_years = len(test_data) / 252
    total_target_pnl = ((1 + ANNUAL_TARGET_PNL / 100) ** n_years - 1) * 100
    logger.info("  Running Monte Carlo (%d simulations, %.0f%% ann -> %.0f%% total over %.1f yr)...",
                MONTE_CARLO_SIMS, ANNUAL_TARGET_PNL, total_target_pnl, n_years)
    mc_result: MonteCarloResult = engine.monte_carlo_confidence(
        returns=bt_result.returns,
        n_simulations=MONTE_CARLO_SIMS,
        target_pnl_pct=total_target_pnl,
    )
    logger.info(
        "  P(Ann > %.0f%%):     %.4f",
        ANNUAL_TARGET_PNL,
        mc_result.prob_above_target,
    )

    return {
        "Strategy": name,
        "Total PnL%": round(metrics["total_pnl_pct"], 2),
        "Ann. Return%": round(metrics["annualized_return"] * 100, 2),
        "Sharpe": round(metrics["sharpe_ratio"], 3),
        "Sortino": round(metrics["sortino_ratio"], 3),
        "Max DD%": round(metrics["max_drawdown"] * 100, 2),
        "Win Rate%": round(metrics["win_rate"] * 100, 2),
        f"P(Ann>{int(ANNUAL_TARGET_PNL)}%) MC": round(mc_result.prob_above_target, 4),
        "Bootstrap p-value": round(metrics["bootstrap_pvalue"], 4),
        "WF OOS Sharpe": round(wf_oos_sharpe, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point: download data, run all strategies, produce comparison."""

    logger.info("Downloading ETF data from %s to %s...", DATA_START, DATA_END)
    raw_data = download_etf_data(start=DATA_START, end=DATA_END)
    logger.info("Downloaded data shape: %s", raw_data.shape)

    close_prices = _extract_close_prices(raw_data)
    ohlcv_data = _extract_ohlcv_data(raw_data)
    logger.info(
        "Close prices: %d rows x %d tickers (%s to %s)",
        len(close_prices),
        len(close_prices.columns),
        close_prices.index[0].date(),
        close_prices.index[-1].date(),
    )
    logger.info(
        "OHLCV data: %d rows x %d columns (full MultiIndex)",
        len(ohlcv_data),
        len(ohlcv_data.columns),
    )

    # Load strategies
    strategies = _load_strategies()
    if not strategies:
        logger.error("No strategies could be loaded. Exiting.")
        sys.exit(1)

    logger.info("Loaded %d strategies.", len(strategies))

    # Backtest engine
    engine = BacktestEngine()

    # Run each strategy
    results: List[Dict[str, Any]] = []

    for name, strategy in strategies:
        try:
            t_start = time.perf_counter()
            row = run_single_strategy(name, strategy, close_prices, engine, ohlcv_data=ohlcv_data)
            elapsed = time.perf_counter() - t_start
            logger.info("Strategy '%s' completed in %.2f seconds.", name, elapsed)
            if row is not None:
                results.append(row)
        except Exception:
            logger.error(
                "Strategy '%s' failed with an error:\n%s",
                name,
                traceback.format_exc(),
            )
            continue

    if not results:
        logger.error("All strategies failed. No results to report.")
        sys.exit(1)

    # Build comparison table
    comparison = pd.DataFrame(results)

    # Sort by P(Ann>45%) MC descending
    mc_col = f"P(Ann>{int(ANNUAL_TARGET_PNL)}%) MC"
    comparison = comparison.sort_values(mc_col, ascending=False).reset_index(drop=True)

    # Save to CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "strategy_comparison.csv"
    comparison.to_csv(csv_path, index=False)
    logger.info("Comparison saved to %s", csv_path)

    # Print the table
    print("\n" + "=" * 100)
    print(f"STRATEGY COMPARISON (sorted by P(Ann>{int(ANNUAL_TARGET_PNL)}%) MC descending)")
    print("=" * 100)
    print(
        tabulate(
            comparison,
            headers="keys",
            tablefmt="pretty",
            showindex=False,
            floatfmt=".4f",
        )
    )
    print("=" * 100)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
