"""Daily autonomous trading pipeline.

Generates signals, then executes trades via Schwab.

Usage:
    uv run python -m src.run_daily            # Full pipeline: signals + execute
    uv run python -m src.run_daily --dry-run   # Signals + simulated execution
    uv run python -m src.run_daily --skip-signals  # Execute existing pending orders only
    uv run python -m src.run_daily --signals-only  # Generate signals only (no execution)

Cron (9 AM ET weekdays):
    0 9 * * 1-5 cd /home/jasonho/proj/strategy-with-bt && uv run python -m src.run_daily >> reports/daily_run.log 2>&1
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = PROJECT_ROOT / "reports" / "positions_state.json"


def _is_trading_day() -> bool:
    """Check if today is a US market trading day (weekday, not a major holiday)."""
    today = datetime.now()
    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Major US market holidays (month, day) — approximate, doesn't handle
    # observed dates or early closes, but good enough to skip obvious days.
    y = today.year
    fixed_holidays = {
        (1, 1),   # New Year's
        (7, 4),   # Independence Day
        (12, 25), # Christmas
    }
    if (today.month, today.day) in fixed_holidays:
        return False

    return True


def _run_signals() -> None:
    """Run the signal generation pipeline (src.automate.main)."""
    logger.info("=" * 60)
    logger.info("STEP 1: Generating trading signals")
    logger.info("=" * 60)

    from src.automate import main as automate_main
    automate_main()


def _execute_orders(dry_run: bool = False) -> None:
    """Execute pending orders via Schwab."""
    from src.schwab_executor import SchwabExecutor, _load_pending_orders, _log_results
    from src.position_manager import PositionManager, TradeOrder

    logger.info("=" * 60)
    logger.info("STEP 2: Executing trades via Schwab%s", " [DRY RUN]" if dry_run else "")
    logger.info("=" * 60)

    orders = _load_pending_orders()
    if not orders:
        logger.info("No pending orders. Nothing to execute.")
        return

    logger.info("Found %d pending orders.", len(orders))

    executor = SchwabExecutor(dry_run=dry_run)
    executor.authenticate()
    portfolio_value = executor.get_portfolio_value()
    logger.info("Portfolio value: $%.2f", portfolio_value)

    results = executor.execute_orders(orders, portfolio_value)
    _log_results(results)

    # Update position state with filled orders
    filled = [r for r in results if r.status in ("FILLED", "DRY_RUN")]
    if not filled:
        logger.warning("No orders were filled. Position state unchanged.")
        return

    pm = PositionManager(portfolio_value=portfolio_value)

    # Reconstruct TradeOrder objects from the original pending orders
    filled_tickers = {r.ticker for r in filled}
    fill_prices: Dict[str, float] = {}
    executed_trade_orders: List[TradeOrder] = []

    for order_dict in orders:
        ticker = order_dict["ticker"]
        if ticker not in filled_tickers:
            continue

        trade_order = TradeOrder(
            ticker=ticker,
            action=order_dict["action"],
            quantity_pct=order_dict.get("quantity_pct", 0.0),
            reason=order_dict.get("reason", ""),
            urgency=order_dict.get("urgency", "NEXT_OPEN"),
            strategy=order_dict.get("strategy", "ensemble"),
            current_weight=order_dict.get("current_weight", 0.0),
            target_weight=order_dict.get("target_weight", 0.0),
            delta_weight=order_dict.get("delta_weight", 0.0),
            current_direction=order_dict.get("current_direction", "FLAT"),
            target_direction=order_dict.get("target_direction", "FLAT"),
            estimated_notional=order_dict.get("estimated_notional"),
            price=order_dict.get("price"),
        )
        executed_trade_orders.append(trade_order)

        # Get fill price from execution results
        result = next(r for r in filled if r.ticker == ticker)
        if result.fill_price:
            fill_prices[ticker] = result.fill_price

    pm.update_positions(executed_trade_orders, execution_prices=fill_prices)
    logger.info("Position state updated with %d filled orders.", len(executed_trade_orders))


def main() -> None:
    t_start = time.perf_counter()
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 60)
    logger.info("DAILY TRADING PIPELINE — %s", today)
    logger.info("=" * 60)

    dry_run = "--dry-run" in sys.argv
    skip_signals = "--skip-signals" in sys.argv
    signals_only = "--signals-only" in sys.argv
    force = "--force" in sys.argv

    if not force and not _is_trading_day():
        logger.info("Not a trading day. Use --force to override. Exiting.")
        return

    try:
        if not skip_signals:
            _run_signals()

        if not signals_only:
            _execute_orders(dry_run=dry_run)

    except Exception:
        logger.exception("Pipeline failed!")
        sys.exit(1)

    elapsed = time.perf_counter() - t_start
    logger.info("Pipeline complete in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
