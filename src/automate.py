"""Daily trading signal automation for the three winning strategies.

Downloads the latest market data and runs:
  1. GARCH Vol (EGARCH, 25% vol target, adaptive blend)
  2. Entropy Regularized (gamma=0.3, lambda=0.01, eg_blend=0.8,
     rebalance=3, eta0=2.0)
  3. Inverse-Vol Weighted Ensemble of 5 strategies

Target: 45% annualized return for the combined ensemble.

Outputs live trading signals, position sizes, and risk checks.

Usage:
    uv run python -m src.automate
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.data.downloader import download_etf_data, SECTOR_ETFS

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
TRAINING_DAYS = 1260  # ~5 trading years -- GARCH Vol needs 504 bars for warm-up
ROLLING_SHARPE_WINDOW = 63  # ~3 months for Sharpe monitoring
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Risk limits
MAX_POSITION_PER_TICKER = 0.20  # 20%
MAX_GROSS_LEVERAGE = 1.0
MIN_ROLLING_SHARPE = 0.3
ANNUAL_RETURN_TARGET = 0.45  # 45% annualized
TRAILING_RETURN_WINDOW = 252  # ~1 trading year

# Calendar buffer: fetch extra days to ensure 504 trading days after holidays
CALENDAR_BUFFER_DAYS = 60

# The 5 strategies used in the inverse-vol ensemble
ENSEMBLE_STRATEGY_REGISTRY: List[Tuple[str, str, str, dict]] = [
    (
        "Entropy Regularized",
        "src.strategies.entropy_regularized",
        "EntropyRegularizedStrategy",
        {"gamma": 0.3, "lambda_base": 0.01, "eg_blend": 0.8,
         "rebalance_freq": 3, "eta0": 2.0},
    ),
    (
        "GARCH Vol",
        "src.strategies.garch_vol",
        "GarchVolStrategy",
        {},
    ),
    (
        "HMM Regime",
        "src.strategies.hmm_regime",
        "HMMRegimeStrategy",
        {},
    ),
    (
        "Spectral Momentum",
        "src.strategies.spectral_momentum",
        "SpectralMomentumStrategy",
        {},
    ),
    (
        "Bayesian Changepoint",
        "src.strategies.bayesian_changepoint",
        "BayesianChangepointStrategy",
        {},
    ),
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _download_latest_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download the last ~504 trading days of ETF data.

    Returns
    -------
    close_prices : pd.DataFrame
        Close prices indexed by date, columns = tickers.
    ohlcv_data : pd.DataFrame
        Full OHLCV MultiIndex DataFrame for strategies needing H/L data.
    """
    today = datetime.now()
    # Fetch enough calendar days to cover 504 trading days + buffer
    start_date = today - timedelta(days=TRAINING_DAYS + CALENDAR_BUFFER_DAYS + 365)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    logger.info("Downloading ETF data from %s to %s...", start_str, end_str)
    raw_data = download_etf_data(start=start_str, end=end_str, use_cache=False)

    # Extract close prices
    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Close" in raw_data.columns.get_level_values(0):
            close_prices = raw_data["Close"]
        else:
            first_level = raw_data.columns.get_level_values(0)[0]
            close_prices = raw_data[first_level]
    else:
        close_prices = raw_data

    close_prices = close_prices.dropna(how="all").ffill().bfill()

    # Clean OHLCV data
    ohlcv_data = raw_data.dropna(how="all").ffill().bfill()

    # Trim to last TRAINING_DAYS trading days
    if len(close_prices) > TRAINING_DAYS:
        close_prices = close_prices.iloc[-TRAINING_DAYS:]
        ohlcv_data = ohlcv_data.loc[close_prices.index[0]:]

    logger.info(
        "Data ready: %d rows x %d tickers (%s to %s)",
        len(close_prices), len(close_prices.columns),
        close_prices.index[0].date(), close_prices.index[-1].date(),
    )
    return close_prices, ohlcv_data


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def _load_strategy(
    module_path: str, class_name: str, kwargs: dict,
) -> Any:
    """Import and instantiate a single strategy."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def _signals_to_portfolio_return(
    signals: pd.DataFrame, prices: pd.DataFrame,
) -> pd.Series:
    """Convert a multi-asset signal DataFrame to daily portfolio returns.

    Handles the multiple signal column conventions used across strategies.
    """
    common_idx = signals.index.intersection(prices.index)
    if len(common_idx) == 0:
        common_idx = signals.index

    price_tickers = list(prices.columns)
    returns = prices.reindex(common_idx).ffill().bfill().pct_change().fillna(0.0)

    # Pattern 1: direct ticker columns
    direct_match = [c for c in signals.columns if c in price_tickers]
    # Pattern 2: {ticker}_signal columns
    signal_weight_match = {
        t: f"{t}_signal" for t in price_tickers
        if f"{t}_signal" in signals.columns
    }
    # Pattern 3: single 'signal'/'weight' columns
    has_single = "signal" in signals.columns

    if direct_match:
        sig_df = signals[direct_match].reindex(common_idx).fillna(0.0)
        port_ret = (sig_df.shift(1).fillna(0.0) * returns[direct_match]).sum(axis=1)
    elif signal_weight_match:
        sig_df = pd.DataFrame(index=common_idx)
        for ticker, sig_col in signal_weight_match.items():
            wgt_col = f"{ticker}_weight"
            if wgt_col in signals.columns:
                sig_df[ticker] = (
                    signals[sig_col].reindex(common_idx).fillna(0.0)
                    * signals[wgt_col].reindex(common_idx).fillna(0.0)
                )
            else:
                sig_df[ticker] = signals[sig_col].reindex(common_idx).fillna(0.0)
        tickers_used = list(sig_df.columns)
        port_ret = (sig_df.shift(1).fillna(0.0) * returns[tickers_used]).sum(axis=1)
    elif has_single:
        sig_arr = signals["signal"].reindex(common_idx).fillna(0.0)
        wgt_arr = signals.get("weight", pd.Series(1.0, index=common_idx))
        wgt_arr = wgt_arr.reindex(common_idx).fillna(1.0)
        composite = sig_arr * wgt_arr
        avg_return = returns.mean(axis=1)
        port_ret = composite.shift(1).fillna(0.0) * avg_return
    else:
        sig_arr = (
            signals.reindex(common_idx).fillna(0.0)
            .select_dtypes(include="number").mean(axis=1)
        )
        avg_return = returns.mean(axis=1)
        port_ret = sig_arr.shift(1).fillna(0.0) * avg_return

    return port_ret


def _extract_latest_positions(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Extract the most recent position from a signal DataFrame.

    Returns a dict mapping ticker -> {signal, weight, direction}.
    """
    price_tickers = list(prices.columns)
    positions: Dict[str, Dict[str, Any]] = {}

    # Detect signal format and extract last row
    direct_match = [c for c in signals.columns if c in price_tickers]
    signal_weight_match = {
        t: f"{t}_signal" for t in price_tickers
        if f"{t}_signal" in signals.columns
    }

    if direct_match:
        last = signals[direct_match].iloc[-1]
        for ticker in direct_match:
            val = float(last[ticker])
            positions[ticker] = {
                "raw_weight": val,
                "direction": "LONG" if val > 0.01 else ("SHORT" if val < -0.01 else "FLAT"),
                "abs_weight": abs(val),
            }
    elif signal_weight_match:
        for ticker, sig_col in signal_weight_match.items():
            wgt_col = f"{ticker}_weight"
            sig_val = float(signals[sig_col].iloc[-1])
            wgt_val = float(signals[wgt_col].iloc[-1]) if wgt_col in signals.columns else 1.0
            composite = sig_val * wgt_val
            positions[ticker] = {
                "raw_weight": composite,
                "direction": "LONG" if composite > 0.01 else ("SHORT" if composite < -0.01 else "FLAT"),
                "abs_weight": abs(composite),
            }
    elif "signal" in signals.columns:
        sig_val = float(signals["signal"].iloc[-1])
        wgt_val = float(signals.get("weight", pd.Series(1.0)).iloc[-1])
        composite = sig_val * wgt_val
        # Apply uniformly to all tickers
        for ticker in price_tickers:
            positions[ticker] = {
                "raw_weight": composite / len(price_tickers),
                "direction": "LONG" if composite > 0.01 else ("SHORT" if composite < -0.01 else "FLAT"),
                "abs_weight": abs(composite) / len(price_tickers),
            }

    return positions


def _run_garch_vol(
    close_prices: pd.DataFrame, ohlcv_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Run GARCH Vol strategy (EGARCH, 25% vol target, adaptive blend).

    Returns (signals_df, portfolio_returns).
    """
    from src.strategies.garch_vol import GarchVolStrategy, GarchVolConfig

    config = GarchVolConfig(
        garch_model="EGARCH",
        target_vol=0.25,
        adaptive_blend=True,
    )
    strategy = GarchVolStrategy(config=config)

    logger.info("  Fitting GARCH Vol strategy...")
    strategy.fit(close_prices, ohlcv_data=ohlcv_data)

    logger.info("  Generating GARCH Vol signals...")
    signals = strategy.generate_signals(close_prices, ohlcv_data=ohlcv_data)

    port_ret = _signals_to_portfolio_return(signals, close_prices)
    return signals, port_ret


def _run_entropy_regularized(
    close_prices: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Run Entropy Regularized strategy with optimized parameters.

    Parameters from optimization:
      gamma=0.3, lambda_base=0.01, eg_blend=0.8, rebalance_freq=3, eta0=2.0

    Returns (signals_df, portfolio_returns).
    """
    from src.strategies.entropy_regularized import EntropyRegularizedStrategy

    strategy = EntropyRegularizedStrategy(
        gamma=0.3,
        lambda_base=0.01,
        eg_blend=0.8,
        rebalance_freq=3,
        eta0=2.0,
    )

    logger.info("  Fitting Entropy Regularized strategy...")
    strategy.fit(close_prices)

    logger.info("  Generating Entropy Regularized signals...")
    signals = strategy.generate_signals(close_prices)

    port_ret = _signals_to_portfolio_return(signals, close_prices)
    return signals, port_ret


def _extract_position_timeseries(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Extract a full timeseries of per-ticker positions from a signal DataFrame.

    Handles the multiple signal column conventions used across strategies
    and returns a DataFrame with ticker columns and position values over time.
    """
    price_tickers = list(prices.columns)
    common_idx = signals.index.intersection(prices.index)
    if len(common_idx) == 0:
        common_idx = signals.index

    positions = pd.DataFrame(0.0, index=common_idx, columns=price_tickers)

    # Pattern 1: direct ticker columns (GARCH Vol, HMM, Spectral Momentum)
    direct_match = [c for c in signals.columns if c in price_tickers]
    # Pattern 2: {ticker}_signal columns (Entropy Reg, Bayesian Changepoint)
    signal_weight_match = {
        t: f"{t}_signal" for t in price_tickers
        if f"{t}_signal" in signals.columns
    }
    # Pattern 3: single 'signal'/'weight' columns
    has_single = "signal" in signals.columns

    if direct_match:
        for ticker in direct_match:
            positions[ticker] = signals[ticker].reindex(common_idx).fillna(0.0)
    elif signal_weight_match:
        for ticker, sig_col in signal_weight_match.items():
            wgt_col = f"{ticker}_weight"
            sig_vals = signals[sig_col].reindex(common_idx).fillna(0.0)
            if wgt_col in signals.columns:
                wgt_vals = signals[wgt_col].reindex(common_idx).fillna(0.0)
                positions[ticker] = sig_vals * wgt_vals
            else:
                positions[ticker] = sig_vals
    elif has_single:
        sig_arr = signals["signal"].reindex(common_idx).fillna(0.0)
        wgt_arr = signals.get("weight", pd.Series(1.0, index=common_idx))
        wgt_arr = wgt_arr.reindex(common_idx).fillna(1.0)
        composite = sig_arr * wgt_arr
        for ticker in price_tickers:
            positions[ticker] = composite / len(price_tickers)

    return positions


def _run_inverse_vol_ensemble(
    close_prices: pd.DataFrame,
    ohlcv_data: pd.DataFrame,
    lookback: int = 63,
    fit_fraction: float = 0.5,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, pd.Series]]:
    """Run the Inverse-Vol Weighted Ensemble of 5 strategies.

    The dataset is split into a *training* period (first ``fit_fraction`` of
    rows, ~250 days) used to calibrate each component strategy, and an
    *out-of-sample* (OOS) period (remaining ~250 days) on which signals are
    generated.  This ensures that:

    1. Component signals are genuinely out-of-sample and past any warm-up.
    2. Strategy returns (position_t * asset_return_{t+1}) are meaningful
       for computing rolling inverse-vol weights.
    3. The ensemble positions at the end of the series are actively non-zero.

    Returns (combined_signals_df, combined_portfolio_returns,
             individual_strategy_returns).
    """
    n_total = len(close_prices)
    split_idx = int(n_total * fit_fraction)
    # Ensure at least ``lookback`` OOS rows for meaningful inv-vol weights
    split_idx = min(split_idx, n_total - lookback - 1)
    split_idx = max(split_idx, lookback)  # and enough training data

    train_prices = close_prices.iloc[:split_idx]
    oos_prices = close_prices.iloc[split_idx:]

    # Build matching OHLCV slice for the training period
    train_ohlcv = ohlcv_data.loc[train_prices.index[0]:train_prices.index[-1]]

    logger.info(
        "  [Ensemble] Data split: train %d rows (%s..%s), "
        "OOS %d rows (%s..%s)",
        len(train_prices),
        train_prices.index[0].date(), train_prices.index[-1].date(),
        len(oos_prices),
        oos_prices.index[0].date(), oos_prices.index[-1].date(),
    )

    # Asset returns on the OOS period (for strategy return computation)
    asset_returns = oos_prices.pct_change().fillna(0.0)

    strategy_returns: Dict[str, pd.Series] = {}
    strategy_signals_map: Dict[str, pd.DataFrame] = {}
    strategy_positions_map: Dict[str, pd.DataFrame] = {}

    for display_name, module_path, class_name, kwargs in ENSEMBLE_STRATEGY_REGISTRY:
        try:
            t0 = time.perf_counter()
            strategy = _load_strategy(module_path, class_name, kwargs)

            # ---- FIT on training period only ----
            logger.info("  [Ensemble] Fitting %s on training data...", display_name)
            if hasattr(strategy, 'fit'):
                try:
                    strategy.fit(train_prices, ohlcv_data=train_ohlcv)
                except TypeError:
                    strategy.fit(train_prices)

            # ---- GENERATE SIGNALS on full data, then slice to OOS ----
            # Many strategies use rolling windows that look back into the
            # training period, so we pass the full price history and then
            # trim the signals to the OOS period.
            logger.info("  [Ensemble] Generating signals for %s...", display_name)
            try:
                sigs = strategy.generate_signals(close_prices, ohlcv_data=ohlcv_data)
            except TypeError:
                sigs = strategy.generate_signals(close_prices)

            # Trim signals to OOS period
            oos_sigs = sigs.loc[sigs.index.isin(oos_prices.index)]
            if oos_sigs.empty:
                # Fallback: select the last len(oos_prices) rows
                oos_sigs = sigs.iloc[-len(oos_prices):]
            strategy_signals_map[display_name] = oos_sigs

            # Extract per-ticker positions on the OOS period
            pos_ts = _extract_position_timeseries(oos_sigs, oos_prices)
            strategy_positions_map[display_name] = pos_ts

            # Compute strategy returns on OOS:
            # strategy_return_t = position_{t-1} * asset_return_t
            # Shift positions by 1 to avoid look-ahead bias.
            pos_shifted = pos_ts.shift(1).fillna(0.0)
            aligned_asset_ret = asset_returns.reindex(pos_ts.index).fillna(0.0)
            strat_ret = (pos_shifted * aligned_asset_ret).sum(axis=1)
            strategy_returns[display_name] = strat_ret

            elapsed = time.perf_counter() - t0

            # Diagnostics: check for non-trivial signals
            n_active_days = int((pos_ts.abs().sum(axis=1) > 0.005).sum())
            last_gross = float(pos_ts.iloc[-1].abs().sum()) if len(pos_ts) > 0 else 0.0
            cum_pnl = (np.prod(1.0 + strat_ret.values) - 1.0) * 100
            logger.info(
                "  [Ensemble] %s done in %.1fs -- OOS cum PnL: %.2f%%, "
                "active days: %d/%d, last gross exp: %.3f",
                display_name, elapsed, cum_pnl,
                n_active_days, len(pos_ts), last_gross,
            )

            if n_active_days < 5:
                logger.warning(
                    "  [Ensemble] %s produced only %d active days out of %d "
                    "OOS days -- signals may be near-zero.",
                    display_name, n_active_days, len(pos_ts),
                )

        except Exception:
            logger.error(
                "  [Ensemble] %s FAILED:\n%s",
                display_name, traceback.format_exc(),
            )

    if len(strategy_returns) < 2:
        raise RuntimeError(
            f"Only {len(strategy_returns)} strategies succeeded; "
            "need at least 2 for ensemble."
        )

    # ---- Inverse-volatility weighting on strategy RETURNS ----
    ret_df = pd.DataFrame(strategy_returns).fillna(0.0)
    names = list(ret_df.columns)
    K = len(names)
    T = len(ret_df)

    # Pre-compute rolling strategy weights for every timestep
    # shape: (T, K) -- each row is the inv-vol weight vector at time t
    strategy_weight_ts = np.empty((T, K))

    for t in range(T):
        if t < lookback:
            # During warm-up, use equal weights
            strategy_weight_ts[t] = 1.0 / K
        else:
            window = ret_df.iloc[t - lookback:t]
            vol = window.std().values
            # Only use inverse-vol for strategies with meaningful volatility;
            # a vol near zero means the strategy was inactive, so assign
            # it a small but non-dominant weight.
            has_activity = vol > 1e-8
            if has_activity.any():
                # For inactive strategies, assign them the max observed vol
                # so they get the smallest inv-vol weight, not infinity.
                max_vol = vol[has_activity].max()
                vol = np.where(has_activity, vol, max_vol * 10.0)
                inv_vol = 1.0 / vol
                strategy_weight_ts[t] = inv_vol / inv_vol.sum()
            else:
                strategy_weight_ts[t] = 1.0 / K

    # Combined portfolio return (weighted sum of component returns)
    combined = pd.Series(0.0, index=ret_df.index)
    for t in range(T):
        combined.iloc[t] = float(ret_df.iloc[t].values @ strategy_weight_ts[t])

    # ---- Build combined position timeseries ----
    # For each date, blend the component per-ticker positions using
    # the inverse-vol weights computed from strategy returns.
    price_tickers = list(close_prices.columns)
    # The combined signals span the OOS period
    combined_oos = pd.DataFrame(0.0, index=oos_prices.index, columns=price_tickers)

    # Align all position timeseries to the OOS index
    aligned_positions: Dict[str, pd.DataFrame] = {}
    for name in names:
        if name in strategy_positions_map:
            pos = strategy_positions_map[name]
            aligned_positions[name] = pos.reindex(oos_prices.index).fillna(0.0)

    # Vectorised blending: for each strategy, multiply its position
    # timeseries by its time-varying weight and accumulate.
    weight_df = pd.DataFrame(
        strategy_weight_ts, index=ret_df.index, columns=names,
    ).reindex(oos_prices.index).ffill().bfill().fillna(1.0 / K)

    for i, name in enumerate(names):
        if name in aligned_positions:
            w_series = weight_df[name]
            combined_oos += aligned_positions[name].mul(w_series, axis=0)

    # ---- Extend to full close_prices index ----
    # The returned DataFrame must cover the full close_prices index for
    # callers like _extract_latest_positions and _build_daily_signals_df.
    # The training period gets zeros; only OOS rows carry active positions.
    full_combined = pd.DataFrame(0.0, index=close_prices.index, columns=price_tickers)
    full_combined.loc[oos_prices.index] = combined_oos.values

    # Log diagnostics about the final ensemble positions
    last_row = full_combined.iloc[-1]
    n_active = int((last_row.abs() > 0.005).sum())
    gross_exp = float(last_row.abs().sum())
    logger.info(
        "  [Ensemble] Final positions: %d active tickers, gross exposure %.3f, "
        "weight vector: %s",
        n_active, gross_exp,
        {name: f"{weight_df[name].iloc[-1]:.3f}" for name in names},
    )

    if n_active == 0:
        logger.warning(
            "  [Ensemble] WARNING: zero active positions in the ensemble! "
            "Check component strategy signals."
        )
        # Log per-strategy last-row positions for debugging
        for name in names:
            if name in aligned_positions:
                pos_last = aligned_positions[name].iloc[-1]
                n_nz = int((pos_last.abs() > 0.005).sum())
                logger.warning(
                    "    %s: %d non-zero tickers, gross=%.4f",
                    name, n_nz, float(pos_last.abs().sum()),
                )

    return full_combined, combined, strategy_returns


# ---------------------------------------------------------------------------
# Risk checks
# ---------------------------------------------------------------------------

def _apply_risk_limits(
    positions: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Apply risk guardrails: per-ticker cap and gross leverage cap.

    Modifies position weights in-place and returns the adjusted dict.
    """
    alerts: List[str] = []

    # 1. Cap individual positions at MAX_POSITION_PER_TICKER
    for ticker, pos in positions.items():
        if pos["abs_weight"] > MAX_POSITION_PER_TICKER:
            alerts.append(
                f"  RISK: {ticker} weight {pos['abs_weight']:.1%} exceeds "
                f"{MAX_POSITION_PER_TICKER:.0%} cap -> clamped."
            )
            scale = MAX_POSITION_PER_TICKER / pos["abs_weight"]
            pos["raw_weight"] *= scale
            pos["abs_weight"] = MAX_POSITION_PER_TICKER

    # 2. Cap gross leverage
    gross = sum(p["abs_weight"] for p in positions.values())
    if gross > MAX_GROSS_LEVERAGE:
        alerts.append(
            f"  RISK: Gross leverage {gross:.2f}x exceeds "
            f"{MAX_GROSS_LEVERAGE:.1f}x cap -> scaling down."
        )
        scale = MAX_GROSS_LEVERAGE / gross
        for pos in positions.values():
            pos["raw_weight"] *= scale
            pos["abs_weight"] *= scale

    for msg in alerts:
        logger.warning(msg)

    return positions


def _check_rolling_sharpe(
    strategy_name: str,
    returns: pd.Series,
    window: int = ROLLING_SHARPE_WINDOW,
) -> Optional[str]:
    """Check if the strategy's rolling Sharpe has dropped below threshold.

    Returns an alert string if so, else None.
    """
    if len(returns) < window:
        return None

    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_std = rolling_std.replace(0, np.nan)
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

    latest_sharpe = rolling_sharpe.iloc[-1]
    if pd.isna(latest_sharpe):
        return None

    if latest_sharpe < MIN_ROLLING_SHARPE:
        return (
            f"  ALERT: {strategy_name} rolling {window}d Sharpe = "
            f"{latest_sharpe:.3f} < {MIN_ROLLING_SHARPE:.1f} threshold"
        )
    return None


def _check_trailing_annual_return(
    strategy_name: str,
    returns: pd.Series,
    window: int = TRAILING_RETURN_WINDOW,
) -> Optional[str]:
    """Warn if trailing 252-day return falls below the 45% annualized target."""
    if len(returns) < window:
        return None

    trailing = returns.iloc[-window:]
    cum_return = float(np.prod(1.0 + trailing.values) - 1.0)
    # Annualize (window is already ~1 year, but be precise)
    ann_return = (1.0 + cum_return) ** (252 / window) - 1.0

    if ann_return < ANNUAL_RETURN_TARGET:
        return (
            f"  ALERT: {strategy_name} trailing {window}d annualized return = "
            f"{ann_return:.1%} < {ANNUAL_RETURN_TARGET:.0%} target"
        )
    return None


def _compute_trailing_annual_return(
    returns: pd.Series,
    window: int = TRAILING_RETURN_WINDOW,
) -> Optional[float]:
    """Compute the trailing 252-day annualized return, or None if insufficient data."""
    if len(returns) < window:
        return None
    trailing = returns.iloc[-window:]
    cum_return = float(np.prod(1.0 + trailing.values) - 1.0)
    return (1.0 + cum_return) ** (252 / window) - 1.0


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def _build_daily_signals_df(
    today_str: str,
    strategy_results: List[Tuple[str, pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """Build the daily_signals.csv DataFrame.

    Each row: date, ticker, signal, weight, strategy
    """
    rows: List[Dict[str, Any]] = []

    for strategy_name, signals, prices in strategy_results:
        positions = _extract_latest_positions(signals, prices)
        positions = _apply_risk_limits(positions)

        for ticker, pos in positions.items():
            rows.append({
                "date": today_str,
                "ticker": ticker,
                "signal": pos["direction"],
                "weight": round(pos["raw_weight"], 6),
                "strategy": strategy_name,
            })

    return pd.DataFrame(rows)


def _build_portfolio_positions_df(
    today_str: str,
    all_positions: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Build the portfolio_positions.csv DataFrame.

    Aggregates across all strategies into final target weights.
    Each row: date, ticker, target_weight, current_action
    """
    rows: List[Dict[str, Any]] = []

    for ticker, pos in sorted(all_positions.items()):
        action = pos["direction"]
        if pos["abs_weight"] < 0.005:
            action = "FLAT"

        rows.append({
            "date": today_str,
            "ticker": ticker,
            "target_weight": round(pos["raw_weight"], 6),
            "current_action": action,
        })

    return pd.DataFrame(rows)


def _compute_combined_positions(
    strategy_results: List[Tuple[str, pd.DataFrame, pd.DataFrame]],
) -> Dict[str, Dict[str, Any]]:
    """Compute the inverse-vol weighted combined position across strategies.

    Weights each strategy equally for the combined signal (the three
    winning strategies serve as the final ensemble).
    """
    # Collect per-strategy positions
    all_strategy_positions: List[Dict[str, Dict[str, Any]]] = []
    strategy_weights = []

    for strategy_name, signals, prices in strategy_results:
        positions = _extract_latest_positions(signals, prices)
        all_strategy_positions.append(positions)
        # Equal weight across the 3 winning strategies
        strategy_weights.append(1.0 / len(strategy_results))

    # Aggregate: weighted average of positions
    all_tickers = set()
    for pos_dict in all_strategy_positions:
        all_tickers.update(pos_dict.keys())

    combined: Dict[str, Dict[str, Any]] = {}
    for ticker in sorted(all_tickers):
        weighted_sum = 0.0
        for i, pos_dict in enumerate(all_strategy_positions):
            if ticker in pos_dict:
                weighted_sum += strategy_weights[i] * pos_dict[ticker]["raw_weight"]

        combined[ticker] = {
            "raw_weight": weighted_sum,
            "direction": "LONG" if weighted_sum > 0.01 else ("SHORT" if weighted_sum < -0.01 else "FLAT"),
            "abs_weight": abs(weighted_sum),
        }

    # Uncapped = raw weights normalized to 100% gross (no per-ticker cap)
    import copy
    uncapped = copy.deepcopy(combined)
    gross = sum(p["abs_weight"] for p in uncapped.values())
    if gross > 0:
        for pos in uncapped.values():
            pos["raw_weight"] /= gross
            pos["abs_weight"] /= gross

    # Capped = per-ticker cap + gross leverage cap (proportional scale-down)
    combined = _apply_risk_limits(combined)

    return combined, uncapped


def _print_summary(
    strategy_results: List[Tuple[str, pd.DataFrame, pd.DataFrame, pd.Series]],
    combined_positions: Dict[str, Dict[str, Any]],
    alerts: List[str],
) -> None:
    """Print a human-readable summary to stdout."""
    today_str = datetime.now().strftime("%Y-%m-%d")

    print("\n" + "=" * 80)
    print(f"  DAILY TRADING SIGNALS -- {today_str}")
    print("=" * 80)

    # Per-strategy summary
    for strategy_name, signals, prices, returns in strategy_results:
        positions = _extract_latest_positions(signals, prices)
        cum_pnl = (np.prod(1.0 + returns.values) - 1.0) * 100
        trailing_ann = _compute_trailing_annual_return(returns)
        trailing_str = f", Trailing 252d Ann: {trailing_ann:.1%}" if trailing_ann is not None else ""

        print(f"\n--- {strategy_name} (Cumulative PnL: {cum_pnl:+.2f}%{trailing_str}) ---")
        print(f"  {'Ticker':<8} {'Signal':<8} {'Weight':>10}  {'Confidence':>12}")
        print(f"  {'-'*8} {'-'*8} {'-'*10}  {'-'*12}")

        for ticker in sorted(positions.keys()):
            pos = positions[ticker]
            confidence = min(abs(pos["raw_weight"]) / MAX_POSITION_PER_TICKER, 1.0)
            conf_label = (
                "HIGH" if confidence > 0.7
                else "MEDIUM" if confidence > 0.3
                else "LOW"
            )
            print(
                f"  {ticker:<8} {pos['direction']:<8} "
                f"{pos['raw_weight']:>+10.4f}  "
                f"{conf_label:>12} ({confidence:.0%})"
            )

    # Combined signal
    print(f"\n{'=' * 80}")
    print("  COMBINED PORTFOLIO (equal-weight ensemble of 3 strategies)")
    print(f"{'=' * 80}")
    print(f"  {'Ticker':<8} {'Action':<8} {'Target Wt':>10}  {'Confidence':>12}")
    print(f"  {'-'*8} {'-'*8} {'-'*10}  {'-'*12}")

    gross_leverage = 0.0
    n_long = 0
    n_short = 0
    n_flat = 0

    for ticker in sorted(combined_positions.keys()):
        pos = combined_positions[ticker]
        gross_leverage += pos["abs_weight"]
        confidence = min(pos["abs_weight"] / MAX_POSITION_PER_TICKER, 1.0)
        conf_label = (
            "HIGH" if confidence > 0.7
            else "MEDIUM" if confidence > 0.3
            else "LOW"
        )

        if pos["direction"] == "LONG":
            n_long += 1
        elif pos["direction"] == "SHORT":
            n_short += 1
        else:
            n_flat += 1

        if pos["abs_weight"] >= 0.005:  # Only print non-trivial positions
            print(
                f"  {ticker:<8} {pos['direction']:<8} "
                f"{pos['raw_weight']:>+10.4f}  "
                f"{conf_label:>12} ({confidence:.0%})"
            )

    print(f"\n  Summary: {n_long} LONG, {n_short} SHORT, {n_flat} FLAT")
    print(f"  Gross Leverage: {gross_leverage:.3f}x (limit: {MAX_GROSS_LEVERAGE:.1f}x)")
    net_exposure = sum(p["raw_weight"] for p in combined_positions.values())
    print(f"  Net Exposure:   {net_exposure:+.3f}")

    # Risk alerts
    if alerts:
        print(f"\n{'=' * 80}")
        print("  RISK ALERTS")
        print(f"{'=' * 80}")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print(f"\n  No risk alerts.")

    print(f"\n{'=' * 80}")
    print(f"  Reports saved to: {REPORTS_DIR}/")
    print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: download data, run winning strategies, output signals."""

    from src.position_manager import PositionManager

    # ---- Early exit: --status prints current positions and exits ----
    if '--status' in sys.argv:
        pm = PositionManager()
        pm.print_current_positions()
        sys.exit(0)

    t_start = time.perf_counter()
    today_str = datetime.now().strftime("%Y-%m-%d")

    # ---- Step 1: Download latest market data ----
    logger.info("Step 1: Downloading latest market data...")
    try:
        close_prices, ohlcv_data = _download_latest_data()
    except Exception:
        logger.error("Failed to download market data:\n%s", traceback.format_exc())
        sys.exit(1)

    # ---- Step 2: Run the 3 winning strategies ----
    logger.info("Step 2: Running winning strategies...")

    strategy_results: List[Tuple[str, pd.DataFrame, pd.DataFrame, pd.Series]] = []
    alerts: List[str] = []

    # Strategy 1: GARCH Vol
    logger.info("--- Strategy 1: GARCH Vol (EGARCH, 25%% vol target, adaptive blend) ---")
    try:
        garch_signals, garch_returns = _run_garch_vol(close_prices, ohlcv_data)
        strategy_results.append(("GARCH Vol", garch_signals, close_prices, garch_returns))
        alert = _check_rolling_sharpe("GARCH Vol", garch_returns)
        if alert:
            alerts.append(alert)
        logger.info("  GARCH Vol: done.")
    except Exception:
        logger.error("  GARCH Vol FAILED:\n%s", traceback.format_exc())

    # Strategy 2: Entropy Regularized
    logger.info("--- Strategy 2: Entropy Regularized (optimized params) ---")
    try:
        entropy_signals, entropy_returns = _run_entropy_regularized(close_prices)
        strategy_results.append(("Entropy Regularized", entropy_signals, close_prices, entropy_returns))
        alert = _check_rolling_sharpe("Entropy Regularized", entropy_returns)
        if alert:
            alerts.append(alert)
        logger.info("  Entropy Regularized: done.")
    except Exception:
        logger.error("  Entropy Regularized FAILED:\n%s", traceback.format_exc())

    # Strategy 3: Inverse-Vol Weighted Ensemble
    logger.info("--- Strategy 3: Inverse-Vol Weighted Ensemble (5 strategies) ---")
    try:
        ensemble_signals, ensemble_returns, individual_returns = (
            _run_inverse_vol_ensemble(close_prices, ohlcv_data)
        )
        strategy_results.append(("InvVol Ensemble", ensemble_signals, close_prices, ensemble_returns))
        alert = _check_rolling_sharpe("InvVol Ensemble", ensemble_returns)
        if alert:
            alerts.append(alert)

        # Check trailing 252-day return vs 45% annualized target
        alert = _check_trailing_annual_return("InvVol Ensemble", ensemble_returns)
        if alert:
            alerts.append(alert)

        # Also check individual ensemble component Sharpes
        for comp_name, comp_returns in individual_returns.items():
            alert = _check_rolling_sharpe(f"[Ensemble/{comp_name}]", comp_returns)
            if alert:
                alerts.append(alert)

        logger.info("  Inverse-Vol Ensemble: done.")
    except Exception:
        logger.error("  Inverse-Vol Ensemble FAILED:\n%s", traceback.format_exc())

    if not strategy_results:
        logger.error("All strategies failed. Cannot produce signals.")
        sys.exit(1)

    # ---- Step 3: Compute combined signal ----
    logger.info("Step 3: Computing combined portfolio positions...")

    # Strip returns from tuples for the builder functions
    results_for_signals = [
        (name, sigs, prices)
        for name, sigs, prices, _ in strategy_results
    ]
    combined_positions, uncapped_positions = _compute_combined_positions(results_for_signals)

    # ---- Step 4: Save outputs ----
    logger.info("Step 4: Saving reports...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # uncapped_positions.csv (raw weights before risk limits)
    uncapped_df = _build_portfolio_positions_df(today_str, uncapped_positions)
    uncapped_path = REPORTS_DIR / "uncapped_positions.csv"
    if uncapped_path.exists():
        existing = pd.read_csv(uncapped_path)
        existing = existing[existing["date"] != today_str]
        uncapped_df = pd.concat([existing, uncapped_df], ignore_index=True)
    uncapped_df.to_csv(uncapped_path, index=False)
    logger.info("  Saved %s (%d rows)", uncapped_path, len(uncapped_df))

    # uncapped per-strategy signals (raw weights normalized to 100% per strategy)
    uncapped_signals_rows: List[Dict[str, Any]] = []
    for strategy_name, signals, prices in results_for_signals:
        positions = _extract_latest_positions(signals, prices)
        strat_gross = sum(abs(p["raw_weight"]) for p in positions.values())
        for ticker, pos in positions.items():
            w = pos["raw_weight"] / strat_gross if strat_gross > 0 else 0.0
            uncapped_signals_rows.append({
                "date": today_str,
                "ticker": ticker,
                "signal": pos["direction"],
                "weight": round(w, 6),
                "strategy": strategy_name,
            })
    uncapped_signals_df = pd.DataFrame(uncapped_signals_rows)
    uncapped_signals_path = REPORTS_DIR / "uncapped_signals.csv"
    if uncapped_signals_path.exists():
        existing = pd.read_csv(uncapped_signals_path)
        existing = existing[existing["date"] != today_str]
        uncapped_signals_df = pd.concat([existing, uncapped_signals_df], ignore_index=True)
    uncapped_signals_df.to_csv(uncapped_signals_path, index=False)
    logger.info("  Saved %s (%d rows)", uncapped_signals_path, len(uncapped_signals_df))

    # daily_signals.csv
    daily_signals_df = _build_daily_signals_df(today_str, results_for_signals)
    daily_signals_path = REPORTS_DIR / "daily_signals.csv"
    # Append if file exists and today's date is not already present
    if daily_signals_path.exists():
        existing = pd.read_csv(daily_signals_path)
        if today_str not in existing["date"].values:
            daily_signals_df = pd.concat([existing, daily_signals_df], ignore_index=True)
        else:
            # Replace today's entries
            existing = existing[existing["date"] != today_str]
            daily_signals_df = pd.concat([existing, daily_signals_df], ignore_index=True)
    daily_signals_df.to_csv(daily_signals_path, index=False)
    logger.info("  Saved %s (%d rows)", daily_signals_path, len(daily_signals_df))

    # portfolio_positions.csv
    positions_df = _build_portfolio_positions_df(today_str, combined_positions)
    positions_path = REPORTS_DIR / "portfolio_positions.csv"
    if positions_path.exists():
        existing = pd.read_csv(positions_path)
        if today_str not in existing["date"].values:
            positions_df = pd.concat([existing, positions_df], ignore_index=True)
        else:
            existing = existing[existing["date"] != today_str]
            positions_df = pd.concat([existing, positions_df], ignore_index=True)
    positions_df.to_csv(positions_path, index=False)
    logger.info("  Saved %s (%d rows)", positions_path, len(positions_df))

    # ---- Step 5: Print human-readable summary ----
    _print_summary(strategy_results, combined_positions, alerts)

    # ---- Step 6: Position management integration ----
    logger.info("Step 6: Position management...")

    pm = PositionManager()
    current_positions = pm.get_current_positions()

    # Convert combined_positions dict to the format expected by PositionManager:
    #   {ticker: (direction_str, signed_weight)}
    combined_signals_dict: Dict[str, Tuple[str, float]] = {}
    for ticker, pos in combined_positions.items():
        if pos["abs_weight"] >= 0.005:  # Only include non-trivial positions
            combined_signals_dict[ticker] = (pos["direction"], pos["raw_weight"])

    # Extract latest prices from close_prices DataFrame
    latest_prices: Dict[str, float] = {}
    for ticker in close_prices.columns:
        price_val = close_prices[ticker].dropna().iloc[-1]
        latest_prices[ticker] = float(price_val)

    # Generate trade orders
    orders = pm.generate_trade_orders(
        new_signals=combined_signals_dict,
        current_prices=latest_prices,
    )

    # Print execution plan
    pm.print_execution_plan(orders, portfolio_value=100_000)

    # Save updated target state (actual execution happens externally)
    pm.save_target_state(orders)

    # If --execute flag is passed, mark orders as executed and update state
    if '--execute' in sys.argv:
        pm.mark_executed(orders, execution_prices=latest_prices)
        logger.info("Positions updated after execution.")

    elapsed = time.perf_counter() - t_start
    logger.info("Automation complete in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
