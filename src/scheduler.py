"""Daily scheduling, health monitoring, and performance tracking for the
strategy automation pipeline.

Runs the full pipeline (data download -> strategy execution -> order
generation -> alerting -> logging), monitors strategy health over time,
and tracks live performance against backtest expectations.

State is persisted to ``reports/scheduler_state.json`` so it survives
restarts.

Setup (cron):
    # Run at 4:30 PM ET every weekday, after market close:
    # 30 16 * * 1-5 cd /home/jasonho/proj/strategy-with-bt && uv run python -m src.scheduler

Usage:
    uv run python -m src.scheduler            # full daily run
    uv run python -m src.scheduler --health   # health check only
    uv run python -m src.scheduler --perf     # performance report only
    uv run python -m src.scheduler --setup    # print cron setup instructions
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
SCHEDULER_STATE_FILE = REPORTS_DIR / "scheduler_state.json"
PERFORMANCE_STATE_FILE = REPORTS_DIR / "performance_state.json"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
ROLLING_SHARPE_WINDOW = 63  # ~3 months
MIN_ROLLING_SHARPE = 0.3
MAX_DRAWDOWN_LIMIT = 0.25  # 25%
SIGNAL_STALENESS_DAYS = 20
CORRELATION_THRESHOLD = 0.85  # ensemble component correlation cap
ANNUAL_RETURN_TARGET = 0.45  # 45% annualized
MIN_ACCEPTABLE_RETURN = 0.30  # 30% -- 1 std below target
TRADING_DAYS_PER_YEAR = 252


# ============================================================================
# Alert
# ============================================================================

@dataclass
class Alert:
    """A single health/performance alert."""

    severity: str  # INFO, WARNING, CRITICAL
    strategy: str
    message: str
    timestamp: str
    metric_value: float
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Alert:
        return cls(**d)


def format_alerts(alerts: List[Alert]) -> str:
    """Format alerts for stdout / email / slack.

    Returns a human-readable multi-line string.  Each alert is prefixed
    with a severity tag so that downstream consumers (email subject lines,
    Slack colour coding) can act on it.
    """
    if not alerts:
        return "No alerts."

    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    sorted_alerts = sorted(alerts, key=lambda a: severity_order.get(a.severity, 9))

    lines: List[str] = []
    lines.append(f"{'=' * 72}")
    lines.append(f"  STRATEGY ALERTS  ({len(alerts)} total)")
    lines.append(f"{'=' * 72}")

    for a in sorted_alerts:
        tag = f"[{a.severity}]"
        lines.append(
            f"  {tag:<12} {a.strategy:<25} "
            f"metric={a.metric_value:+.4f}  threshold={a.threshold:+.4f}"
        )
        lines.append(f"               {a.message}")
        lines.append("")

    lines.append(f"{'=' * 72}")
    return "\n".join(lines)


# ============================================================================
# StrategyHealthMonitor
# ============================================================================

class StrategyHealthMonitor:
    """Track strategy health metrics over time and raise alerts."""

    def __init__(
        self,
        sharpe_window: int = ROLLING_SHARPE_WINDOW,
        min_sharpe: float = MIN_ROLLING_SHARPE,
        max_drawdown: float = MAX_DRAWDOWN_LIMIT,
        staleness_days: int = SIGNAL_STALENESS_DAYS,
        correlation_cap: float = CORRELATION_THRESHOLD,
    ) -> None:
        self.sharpe_window = sharpe_window
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.staleness_days = staleness_days
        self.correlation_cap = correlation_cap

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def check_health(
        self,
        strategy_returns: Dict[str, pd.Series],
        backtest_expectations: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[Alert]:
        """Run all health checks and return a list of alerts.

        Parameters
        ----------
        strategy_returns
            Mapping of strategy name to daily return series.
        backtest_expectations
            Optional mapping of strategy name to expected metrics
            (``ann_return``, ``sharpe``, ``max_dd``).
        """
        now_str = datetime.now(timezone.utc).isoformat()
        alerts: List[Alert] = []

        for name, returns in strategy_returns.items():
            if returns is None or len(returns) < 2:
                continue
            alerts.extend(self._check_rolling_sharpe(name, returns, now_str))
            alerts.extend(self._check_drawdown(name, returns, now_str))
            alerts.extend(self._check_signal_staleness(name, returns, now_str))

            if backtest_expectations and name in backtest_expectations:
                alerts.extend(
                    self._check_return_deviation(
                        name, returns, backtest_expectations[name], now_str,
                    )
                )

        # Cross-strategy correlation check
        if len(strategy_returns) >= 2:
            alerts.extend(
                self._check_correlation_breakdown(strategy_returns, now_str)
            )

        return alerts

    # ------------------------------------------------------------------ #
    # Individual checks
    # ------------------------------------------------------------------ #

    def _check_rolling_sharpe(
        self, name: str, returns: pd.Series, ts: str,
    ) -> List[Alert]:
        alerts: List[Alert] = []
        if len(returns) < self.sharpe_window:
            return alerts

        rolling_mean = returns.rolling(self.sharpe_window).mean()
        rolling_std = returns.rolling(self.sharpe_window).std().replace(0, np.nan)
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS_PER_YEAR)

        latest = rolling_sharpe.iloc[-1]
        if pd.isna(latest):
            return alerts

        if latest < 0:
            alerts.append(Alert(
                severity="CRITICAL",
                strategy=name,
                message=(
                    f"Rolling {self.sharpe_window}d Sharpe is negative "
                    f"({latest:.3f}). Strategy may be broken."
                ),
                timestamp=ts,
                metric_value=float(latest),
                threshold=0.0,
            ))
        elif latest < self.min_sharpe:
            alerts.append(Alert(
                severity="WARNING",
                strategy=name,
                message=(
                    f"Rolling {self.sharpe_window}d Sharpe ({latest:.3f}) "
                    f"below threshold ({self.min_sharpe:.1f})."
                ),
                timestamp=ts,
                metric_value=float(latest),
                threshold=self.min_sharpe,
            ))
        return alerts

    def _check_drawdown(
        self, name: str, returns: pd.Series, ts: str,
    ) -> List[Alert]:
        alerts: List[Alert] = []
        equity = (1 + returns).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        current_dd = float(drawdown.iloc[-1])

        if current_dd < -self.max_drawdown:
            sev = "CRITICAL" if current_dd < -self.max_drawdown * 1.5 else "WARNING"
            alerts.append(Alert(
                severity=sev,
                strategy=name,
                message=(
                    f"Current drawdown {current_dd:.1%} exceeds "
                    f"limit ({-self.max_drawdown:.0%})."
                ),
                timestamp=ts,
                metric_value=current_dd,
                threshold=-self.max_drawdown,
            ))
        return alerts

    def _check_signal_staleness(
        self, name: str, returns: pd.Series, ts: str,
    ) -> List[Alert]:
        """Flag if returns have been ~zero (no signal change) for too long."""
        alerts: List[Alert] = []
        if len(returns) < self.staleness_days:
            return alerts

        tail = returns.iloc[-self.staleness_days:]
        # "stale" means the absolute daily return is negligible every day
        # This indicates the signal hasn't changed (position is flat or
        # the strategy has stopped producing new signals).
        near_zero = (tail.abs() < 1e-8).all()
        if near_zero:
            alerts.append(Alert(
                severity="WARNING",
                strategy=name,
                message=(
                    f"No meaningful return for {self.staleness_days} consecutive "
                    f"days. Signal may be stale."
                ),
                timestamp=ts,
                metric_value=float(self.staleness_days),
                threshold=float(self.staleness_days),
            ))
        return alerts

    def _check_correlation_breakdown(
        self,
        strategy_returns: Dict[str, pd.Series],
        ts: str,
    ) -> List[Alert]:
        """Warn when ensemble components become too correlated."""
        alerts: List[Alert] = []
        ret_df = pd.DataFrame(strategy_returns).dropna()
        if len(ret_df) < self.sharpe_window:
            return alerts

        # Use the last sharpe_window days for correlation
        window = ret_df.iloc[-self.sharpe_window:]
        corr = window.corr()

        names = list(corr.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rho = corr.iloc[i, j]
                if abs(rho) > self.correlation_cap:
                    alerts.append(Alert(
                        severity="WARNING",
                        strategy=f"{names[i]} / {names[j]}",
                        message=(
                            f"Rolling correlation ({rho:.3f}) exceeds "
                            f"threshold ({self.correlation_cap:.2f}). "
                            f"Ensemble diversification is degraded."
                        ),
                        timestamp=ts,
                        metric_value=float(rho),
                        threshold=self.correlation_cap,
                    ))
        return alerts

    def _check_return_deviation(
        self,
        name: str,
        returns: pd.Series,
        expectations: Dict[str, float],
        ts: str,
    ) -> List[Alert]:
        """Compare live performance against backtest expectations."""
        alerts: List[Alert] = []
        if len(returns) < TRADING_DAYS_PER_YEAR // 2:
            return alerts

        # Annualized return
        cum = float(np.prod(1 + returns.values) - 1)
        n_years = len(returns) / TRADING_DAYS_PER_YEAR
        if n_years > 0:
            ann_ret = (1 + cum) ** (1 / n_years) - 1
        else:
            return alerts

        expected_ann = expectations.get("ann_return", ANNUAL_RETURN_TARGET)
        # Flag if more than 50% below expectation
        if ann_ret < expected_ann * 0.5:
            alerts.append(Alert(
                severity="CRITICAL",
                strategy=name,
                message=(
                    f"Annualized return ({ann_ret:.1%}) is more than 50% "
                    f"below backtest expectation ({expected_ann:.1%})."
                ),
                timestamp=ts,
                metric_value=ann_ret,
                threshold=expected_ann * 0.5,
            ))
        elif ann_ret < expected_ann * 0.75:
            alerts.append(Alert(
                severity="WARNING",
                strategy=name,
                message=(
                    f"Annualized return ({ann_ret:.1%}) is below 75% of "
                    f"backtest expectation ({expected_ann:.1%})."
                ),
                timestamp=ts,
                metric_value=ann_ret,
                threshold=expected_ann * 0.75,
            ))

        return alerts


# ============================================================================
# PerformanceTracker
# ============================================================================

class PerformanceTracker:
    """Track live performance versus backtest expectations.

    State is persisted to a JSON file so metrics accumulate across runs.
    """

    def __init__(
        self,
        state_file: str | Path = PERFORMANCE_STATE_FILE,
    ) -> None:
        self._state_file = Path(state_file)
        self._state: Dict[str, Any] = self._load_state()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load_state(self) -> Dict[str, Any]:
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                logger.info(
                    "Loaded performance state: %d daily returns tracked.",
                    len(data.get("daily_returns", {})),
                )
                return data
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Corrupt performance state (%s). Starting fresh.", exc,
                )
        return {
            "daily_returns": {},  # date_str -> float
            "metadata": {},
        }

    def _save_state(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._state_file.write_text(json.dumps(self._state, indent=2) + "\n")
        logger.info("Performance state saved to %s", self._state_file)

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(self, date: str | datetime, portfolio_return: float) -> None:
        """Record today's portfolio return.

        Parameters
        ----------
        date
            Date string (``YYYY-MM-DD``) or datetime object.
        portfolio_return
            Simple daily return (e.g. 0.003 for +0.3%).
        """
        if isinstance(date, datetime):
            date_str = date.strftime("%Y-%m-%d")
        else:
            date_str = str(date)

        self._state["daily_returns"][date_str] = float(portfolio_return)
        self._save_state()
        logger.info(
            "Recorded return for %s: %+.4f%%",
            date_str,
            portfolio_return * 100,
        )

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def _returns_series(self) -> pd.Series:
        """Build a sorted pd.Series of daily returns."""
        dr = self._state.get("daily_returns", {})
        if not dr:
            return pd.Series(dtype=float)
        s = pd.Series(dr, dtype=float)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    def get_metrics(self) -> Dict[str, Any]:
        """Compute and return current performance metrics.

        Returns
        -------
        dict with keys:
            n_days, cumulative_return, annualized_return, sharpe,
            sortino, max_drawdown, win_rate, best_day, worst_day,
            backtest_target, on_track
        """
        returns = self._returns_series()
        n = len(returns)
        if n == 0:
            return {"n_days": 0, "message": "No returns recorded yet."}

        cum = float(np.prod(1 + returns.values) - 1)
        n_years = n / TRADING_DAYS_PER_YEAR

        if n_years > 0:
            ann_ret = (1 + cum) ** (1 / n_years) - 1
        else:
            ann_ret = 0.0

        mean_daily = returns.mean()
        std_daily = returns.std()
        if std_daily > 0:
            sharpe = float(mean_daily / std_daily * np.sqrt(TRADING_DAYS_PER_YEAR))
        else:
            sharpe = 0.0

        downside = returns[returns < 0].std()
        if downside > 0:
            sortino = float(mean_daily / downside * np.sqrt(TRADING_DAYS_PER_YEAR))
        else:
            sortino = 0.0

        equity = (1 + returns).cumprod()
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max
        max_dd = float(dd.min())

        win_rate = float((returns > 0).mean())

        on_track = ann_ret > MIN_ACCEPTABLE_RETURN

        return {
            "n_days": n,
            "cumulative_return": round(cum, 6),
            "annualized_return": round(ann_ret, 4),
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "best_day": round(float(returns.max()), 6),
            "worst_day": round(float(returns.min()), 6),
            "backtest_target": ANNUAL_RETURN_TARGET,
            "min_acceptable": MIN_ACCEPTABLE_RETURN,
            "on_track": on_track,
        }

    def is_on_track(self) -> bool:
        """Is the strategy performing within acceptable bounds?

        Returns True if annualized return exceeds 30% (allowing for
        some underperformance relative to the 45% backtest target).
        """
        metrics = self.get_metrics()
        if metrics.get("n_days", 0) == 0:
            return True  # No data yet -- assume OK
        return metrics.get("annualized_return", 0) > MIN_ACCEPTABLE_RETURN

    def print_report(self) -> None:
        """Print a formatted performance report to stdout."""
        metrics = self.get_metrics()

        print(f"\n{'=' * 72}")
        print("  LIVE PERFORMANCE TRACKER")
        print(f"{'=' * 72}")

        if metrics.get("n_days", 0) == 0:
            print("  No returns recorded yet.")
            print(f"{'=' * 72}\n")
            return

        print(f"  Trading days tracked:  {metrics['n_days']}")
        print(f"  Cumulative return:     {metrics['cumulative_return']:+.2%}")
        print(f"  Annualized return:     {metrics['annualized_return']:+.2%}")
        print(f"  Sharpe ratio:          {metrics['sharpe']:.3f}")
        print(f"  Sortino ratio:         {metrics['sortino']:.3f}")
        print(f"  Max drawdown:          {metrics['max_drawdown']:.2%}")
        print(f"  Win rate:              {metrics['win_rate']:.1%}")
        print(f"  Best day:              {metrics['best_day']:+.4%}")
        print(f"  Worst day:             {metrics['worst_day']:+.4%}")
        print()
        print(f"  Backtest target:       {metrics['backtest_target']:.0%} annualized")
        print(f"  Minimum acceptable:    {metrics['min_acceptable']:.0%} annualized")

        status = "ON TRACK" if metrics["on_track"] else "BELOW TARGET"
        print(f"  Status:                {status}")
        print(f"{'=' * 72}\n")


# ============================================================================
# Scheduler State
# ============================================================================

class _SchedulerState:
    """Persist run metadata across invocations."""

    def __init__(self, state_file: Path = SCHEDULER_STATE_FILE) -> None:
        self._file = state_file
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text())
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt scheduler state. Starting fresh.")
        return {"runs": [], "last_alerts": []}

    def save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._data, indent=2) + "\n")

    def record_run(
        self,
        success: bool,
        elapsed_sec: float,
        n_alerts: int,
        alerts: List[Alert],
    ) -> None:
        run_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "elapsed_sec": round(elapsed_sec, 2),
            "n_alerts": n_alerts,
        }
        self._data.setdefault("runs", []).append(run_entry)
        # Keep last 90 runs (~ 3 months of weekdays)
        self._data["runs"] = self._data["runs"][-90:]
        self._data["last_alerts"] = [a.to_dict() for a in alerts]
        self.save()

    @property
    def last_run(self) -> Optional[Dict[str, Any]]:
        runs = self._data.get("runs", [])
        return runs[-1] if runs else None

    @property
    def total_runs(self) -> int:
        return len(self._data.get("runs", []))

    @property
    def success_rate(self) -> float:
        runs = self._data.get("runs", [])
        if not runs:
            return 0.0
        return sum(1 for r in runs if r.get("success")) / len(runs)


# ============================================================================
# Daily Run Pipeline
# ============================================================================

def daily_run() -> None:
    """Run the full automation pipeline.

    Designed to be called by cron or systemd timer.

    Steps:
        1. Download latest data
        2. Run strategies (via ``src.automate``)
        3. Generate trade orders (via ``src.position_manager``)
        4. Check strategy health and send alerts
        5. Update performance tracker
        6. Log everything to scheduler state
    """
    state = _SchedulerState()
    perf_tracker = PerformanceTracker()
    health_monitor = StrategyHealthMonitor()

    t_start = time.perf_counter()
    today_str = datetime.now().strftime("%Y-%m-%d")
    all_alerts: List[Alert] = []
    success = False

    logger.info("=" * 72)
    logger.info("  SCHEDULER: daily_run starting  (%s)", today_str)
    logger.info("=" * 72)

    try:
        # ---- Step 1: Download latest market data -----------------------
        logger.info("Step 1/5: Downloading latest market data...")
        from src.automate import _download_latest_data
        close_prices, ohlcv_data = _download_latest_data()

        # ---- Step 2: Run strategies ------------------------------------
        logger.info("Step 2/5: Running strategies...")
        from src.automate import (
            _run_garch_vol,
            _run_entropy_regularized,
            _run_inverse_vol_ensemble,
            _extract_latest_positions,
            _apply_risk_limits,
            _compute_combined_positions,
            _build_daily_signals_df,
            _build_portfolio_positions_df,
            _print_summary,
        )

        strategy_results = []  # (name, signals, prices, returns)
        strategy_return_map: Dict[str, pd.Series] = {}

        # GARCH Vol
        try:
            logger.info("  Running GARCH Vol...")
            garch_sigs, garch_rets = _run_garch_vol(close_prices, ohlcv_data)
            strategy_results.append(("GARCH Vol", garch_sigs, close_prices, garch_rets))
            strategy_return_map["GARCH Vol"] = garch_rets
        except Exception:
            logger.error("  GARCH Vol FAILED:\n%s", traceback.format_exc())

        # Entropy Regularized
        try:
            logger.info("  Running Entropy Regularized...")
            ent_sigs, ent_rets = _run_entropy_regularized(close_prices)
            strategy_results.append(("Entropy Regularized", ent_sigs, close_prices, ent_rets))
            strategy_return_map["Entropy Regularized"] = ent_rets
        except Exception:
            logger.error("  Entropy Regularized FAILED:\n%s", traceback.format_exc())

        # Inverse-Vol Ensemble
        try:
            logger.info("  Running Inverse-Vol Ensemble...")
            ens_sigs, ens_rets, ind_rets = _run_inverse_vol_ensemble(close_prices, ohlcv_data)
            strategy_results.append(("InvVol Ensemble", ens_sigs, close_prices, ens_rets))
            strategy_return_map["InvVol Ensemble"] = ens_rets
            for comp_name, comp_rets in ind_rets.items():
                strategy_return_map[f"Ensemble/{comp_name}"] = comp_rets
        except Exception:
            logger.error("  Inverse-Vol Ensemble FAILED:\n%s", traceback.format_exc())

        if not strategy_results:
            raise RuntimeError("All strategies failed. Cannot produce signals.")

        # ---- Step 3: Generate trade orders -----------------------------
        logger.info("Step 3/5: Generating trade orders...")
        from src.position_manager import PositionManager

        results_for_signals = [
            (name, sigs, prices)
            for name, sigs, prices, _ in strategy_results
        ]
        combined_positions, _ = _compute_combined_positions(results_for_signals)

        # Build new_signals for PositionManager
        pm = PositionManager()
        new_signals = {}
        current_prices = {}
        for ticker, pos in combined_positions.items():
            direction = pos["direction"]
            weight = pos["raw_weight"]
            new_signals[ticker] = (direction, weight)
            # Use the latest close price
            if ticker in close_prices.columns:
                current_prices[ticker] = float(close_prices[ticker].iloc[-1])

        orders = pm.generate_trade_orders(new_signals, current_prices)
        if orders:
            pm.save_target_state(orders)
            pm.print_execution_plan(orders)
        else:
            logger.info("  No trades needed -- portfolio is on target.")

        # Save CSV outputs
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        daily_signals_df = _build_daily_signals_df(today_str, results_for_signals)
        daily_signals_path = REPORTS_DIR / "daily_signals.csv"
        if daily_signals_path.exists():
            existing = pd.read_csv(daily_signals_path)
            existing = existing[existing["date"] != today_str]
            daily_signals_df = pd.concat([existing, daily_signals_df], ignore_index=True)
        daily_signals_df.to_csv(daily_signals_path, index=False)

        positions_df = _build_portfolio_positions_df(today_str, combined_positions)
        positions_path = REPORTS_DIR / "portfolio_positions.csv"
        if positions_path.exists():
            existing = pd.read_csv(positions_path)
            existing = existing[existing["date"] != today_str]
            positions_df = pd.concat([existing, positions_df], ignore_index=True)
        positions_df.to_csv(positions_path, index=False)

        # ---- Step 4: Health checks and alerts --------------------------
        logger.info("Step 4/5: Checking strategy health...")
        health_alerts = health_monitor.check_health(strategy_return_map)
        all_alerts.extend(health_alerts)

        if all_alerts:
            alert_text = format_alerts(all_alerts)
            print(alert_text)
        else:
            logger.info("  All health checks passed.")

        # ---- Step 5: Update performance tracker ------------------------
        logger.info("Step 5/5: Updating performance tracker...")

        # Compute today's blended portfolio return from the ensemble
        # Use the last strategy result's returns for the most recent day
        if strategy_results:
            # Average across all strategies for the combined return
            all_rets = [r for _, _, _, r in strategy_results]
            combined_daily = pd.DataFrame({f"s{i}": r for i, r in enumerate(all_rets)})
            combined_daily = combined_daily.fillna(0).mean(axis=1)
            if len(combined_daily) > 0:
                today_return = float(combined_daily.iloc[-1])
                perf_tracker.update(today_str, today_return)

        # Print summary
        _print_summary(strategy_results, combined_positions, [a.message for a in all_alerts])
        perf_tracker.print_report()

        success = True

    except Exception:
        logger.error("SCHEDULER: daily_run FAILED:\n%s", traceback.format_exc())
        all_alerts.append(Alert(
            severity="CRITICAL",
            strategy="SCHEDULER",
            message=f"Pipeline failed: {traceback.format_exc(limit=3)}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            metric_value=0.0,
            threshold=0.0,
        ))

    # ---- Record run metadata -------------------------------------------
    elapsed = time.perf_counter() - t_start
    state.record_run(
        success=success,
        elapsed_sec=elapsed,
        n_alerts=len(all_alerts),
        alerts=all_alerts,
    )

    logger.info(
        "SCHEDULER: daily_run %s in %.1f seconds (%d alerts).",
        "SUCCEEDED" if success else "FAILED",
        elapsed,
        len(all_alerts),
    )

    # Print final status
    print(f"\n{'=' * 72}")
    print(f"  SCHEDULER STATUS")
    print(f"{'=' * 72}")
    print(f"  Run completed:   {'YES' if success else 'FAILED'}")
    print(f"  Elapsed:         {elapsed:.1f}s")
    print(f"  Alerts:          {len(all_alerts)}")
    print(f"  Total runs:      {state.total_runs}")
    print(f"  Success rate:    {state.success_rate:.0%}")
    last = state.last_run
    if last:
        print(f"  Last run:        {last['timestamp']}")
    print(f"  State file:      {SCHEDULER_STATE_FILE}")
    print(f"  Perf state file: {PERFORMANCE_STATE_FILE}")
    print(f"{'=' * 72}\n")


# ============================================================================
# Setup Instructions
# ============================================================================

def print_setup_instructions() -> None:
    """Print cron / systemd setup instructions."""
    print(f"""
{'=' * 72}
  SCHEDULER SETUP INSTRUCTIONS
{'=' * 72}

  Option 1: cron (recommended)
  ----------------------------
  Add to crontab with: crontab -e

  # Run at 4:30 PM ET every weekday, after market close:
  30 16 * * 1-5 cd /home/jasonho/proj/strategy-with-bt && uv run python -m src.scheduler >> reports/scheduler.log 2>&1

  # Verify cron is running:
  crontab -l


  Option 2: systemd timer
  -----------------------
  Create /etc/systemd/system/strategy-scheduler.service:

    [Unit]
    Description=Strategy Automation Scheduler
    After=network.target

    [Service]
    Type=oneshot
    User=jasonho
    WorkingDirectory=/home/jasonho/proj/strategy-with-bt
    ExecStart=/home/jasonho/.local/bin/uv run python -m src.scheduler
    StandardOutput=append:/home/jasonho/proj/strategy-with-bt/reports/scheduler.log
    StandardError=append:/home/jasonho/proj/strategy-with-bt/reports/scheduler.log

  Create /etc/systemd/system/strategy-scheduler.timer:

    [Unit]
    Description=Run strategy scheduler weekdays at 4:30 PM ET

    [Timer]
    OnCalendar=Mon..Fri 16:30 America/New_York
    Persistent=true

    [Install]
    WantedBy=timers.target

  Then:
    sudo systemctl daemon-reload
    sudo systemctl enable --now strategy-scheduler.timer


  Option 3: manual run
  --------------------
  uv run python -m src.scheduler            # full daily run
  uv run python -m src.scheduler --health   # health check only
  uv run python -m src.scheduler --perf     # performance report only
  uv run python -m src.scheduler --setup    # show this message

{'=' * 72}
""")


# ============================================================================
# CLI entry point
# ============================================================================

def _cli_main() -> None:
    """Parse command-line flags and dispatch."""
    args = sys.argv[1:]

    if "--setup" in args:
        print_setup_instructions()
        return

    if "--health" in args:
        # Health-check-only mode: load last recorded returns and check
        logger.info("Running health check only...")
        perf = PerformanceTracker()
        returns = perf._returns_series()
        if len(returns) == 0:
            print("No performance data available yet. Run a full daily_run first.")
            return
        monitor = StrategyHealthMonitor()
        alerts = monitor.check_health({"Portfolio": returns})
        print(format_alerts(alerts))
        return

    if "--perf" in args:
        # Performance report only
        perf = PerformanceTracker()
        perf.print_report()
        return

    # Default: full daily run
    daily_run()


if __name__ == "__main__":
    _cli_main()
