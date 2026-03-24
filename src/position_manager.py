"""Position management system for live trading automation.

Handles the full lifecycle of position management: state persistence,
signal-to-order translation, risk checks, rebalancing, and execution
plan generation.

State is persisted as JSON in ``reports/positions_state.json`` so it
survives restarts.

Exported API (for use from automate.py):
    generate_trade_orders(new_signals, current_prices, ...) -> List[TradeOrder]
    update_positions(executed_orders, execution_prices)      -> dict
    get_current_positions()                                  -> dict
    print_execution_plan(orders, portfolio_value)             -> None

Usage (standalone):
    uv run python -m src.position_manager --status
    uv run python -m src.position_manager --clear
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STATE_DIR = Path(__file__).resolve().parents[1] / "reports"
POSITIONS_STATE_FILE = STATE_DIR / "positions_state.json"
TRADE_LOG_FILE = STATE_DIR / "trade_log.jsonl"

# ---------------------------------------------------------------------------
# Risk / threshold constants
# ---------------------------------------------------------------------------
# Minimum absolute weight change to trigger a rebalance trade
REBALANCE_THRESHOLD = 0.05  # 5% of portfolio
# Minimum position weight to consider "open"
MIN_POSITION_WEIGHT = 0.005
# Minimum trade size worth executing (avoids noise)
MIN_TRADE_THRESHOLD = 0.005  # 0.5% of portfolio

# Risk limits
MAX_POSITION_PER_TICKER = 0.20   # 20% per name
MAX_GROSS_LEVERAGE = 1.50         # 1.5x
MAX_SINGLE_DAY_TURNOVER = 0.30   # 30% of portfolio
WHIPSAW_PROTECTION_DAYS = 3       # no reversal within N days of entry
DRAWDOWN_CIRCUIT_BREAKER = 0.20   # 20% portfolio drawdown
DRAWDOWN_REDUCTION_FACTOR = 0.50  # reduce all positions to 50%

# Transaction cost estimate (basis points)
TRANSACTION_COST_BPS = 6  # 6 bps round-trip


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    """Full state for a single position in a ticker."""

    ticker: str
    current_position: str  # LONG / SHORT / FLAT
    current_weight: float  # signed: +long, -short, range roughly -1 to 1
    entry_date: Optional[str] = None  # ISO date when position was opened
    entry_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None  # in dollar terms or pct
    last_signal_date: Optional[str] = None  # ISO date of latest signal update
    strategy: Optional[str] = None  # which strategy generated this position

    @property
    def abs_weight(self) -> float:
        return abs(self.current_weight)

    @property
    def direction(self) -> str:
        """Convenience alias -- same as current_position."""
        return self.current_position

    def days_held(self, as_of: Optional[str] = None) -> Optional[int]:
        """Number of calendar days since entry.  Returns None if no entry_date."""
        if not self.entry_date:
            return None
        entry = datetime.fromisoformat(self.entry_date)
        ref = datetime.fromisoformat(as_of) if as_of else datetime.now()
        return (ref - entry).days

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "current_position": self.current_position,
            "current_weight": self.current_weight,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "unrealized_pnl": self.unrealized_pnl,
            "last_signal_date": self.last_signal_date,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PositionState":
        return cls(
            ticker=d["ticker"],
            current_position=d.get("current_position", "FLAT"),
            current_weight=d.get("current_weight", 0.0),
            entry_date=d.get("entry_date"),
            entry_price=d.get("entry_price"),
            unrealized_pnl=d.get("unrealized_pnl"),
            last_signal_date=d.get("last_signal_date"),
            strategy=d.get("strategy"),
        )

    @classmethod
    def flat(cls, ticker: str) -> "PositionState":
        """Create a FLAT (no-position) state for *ticker*."""
        return cls(ticker=ticker, current_position="FLAT", current_weight=0.0)


@dataclass
class TradeOrder:
    """A single trade order describing how to move from current to target."""

    ticker: str
    action: str  # BUY, SELL, SELL_SHORT, BUY_TO_COVER, NO_ACTION
    quantity_pct: float  # absolute % of portfolio to trade
    reason: str  # human-readable, e.g. "Open long", "Rebalance +3%"
    urgency: str  # IMMEDIATE / NEXT_OPEN / END_OF_DAY
    strategy: str  # which strategy generated this

    # Extra context (populated by generate_trade_orders)
    current_weight: float = 0.0
    target_weight: float = 0.0
    delta_weight: float = 0.0
    current_direction: str = "FLAT"
    target_direction: str = "FLAT"
    estimated_notional: Optional[float] = None
    price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "action": self.action,
            "quantity_pct": self.quantity_pct,
            "reason": self.reason,
            "urgency": self.urgency,
            "strategy": self.strategy,
            "current_weight": self.current_weight,
            "target_weight": self.target_weight,
            "delta_weight": self.delta_weight,
            "current_direction": self.current_direction,
            "target_direction": self.target_direction,
            "estimated_notional": self.estimated_notional,
            "price": self.price,
        }


# ---------------------------------------------------------------------------
# Position transition logic
# ---------------------------------------------------------------------------

class PositionTransition:
    """Determines what action to take given current position and new signal.

    Transition table
    ----------------
    FLAT  -> LONG signal:   OPEN LONG  (BUY)
    FLAT  -> SHORT signal:  OPEN SHORT (SELL_SHORT)
    FLAT  -> FLAT signal:   NO ACTION

    LONG  -> LONG signal:   HOLD (or ADJUST if weight changed > threshold)
    LONG  -> FLAT signal:   CLOSE LONG (SELL)
    LONG  -> SHORT signal:  REVERSE   (SELL current + SELL_SHORT new = 2x trade)

    SHORT -> SHORT signal:  HOLD (or ADJUST if weight changed > threshold)
    SHORT -> FLAT signal:   CLOSE SHORT (BUY_TO_COVER)
    SHORT -> LONG signal:   REVERSE   (BUY_TO_COVER current + BUY new = 2x trade)
    """

    @staticmethod
    def classify(
        cur_dir: str,
        tgt_dir: str,
        cur_weight: float,
        tgt_weight: float,
    ) -> Tuple[str, str, str]:
        """Return (action, reason, urgency).

        *action* is one of: BUY, SELL, SELL_SHORT, BUY_TO_COVER, NO_ACTION.
        *reason* is a human-readable explanation.
        *urgency* is IMMEDIATE, NEXT_OPEN, or END_OF_DAY.
        """
        delta = tgt_weight - cur_weight
        abs_delta = abs(delta)

        # --- FLAT -> X ---
        if cur_dir == "FLAT":
            if tgt_dir == "LONG":
                return "BUY", "Open long from Flat", "NEXT_OPEN"
            elif tgt_dir == "SHORT":
                return "SELL_SHORT", "Open short from Flat", "NEXT_OPEN"
            else:
                return "NO_ACTION", "Remain Flat", "END_OF_DAY"

        # --- LONG -> X ---
        if cur_dir == "LONG":
            if tgt_dir == "LONG":
                if abs_delta >= REBALANCE_THRESHOLD:
                    direction = "increase" if delta > 0 else "decrease"
                    return (
                        "BUY" if delta > 0 else "SELL",
                        f"Rebalance long ({direction} "
                        f"{abs(cur_weight)*100:.1f}% -> {abs(tgt_weight)*100:.1f}%)",
                        "NEXT_OPEN",
                    )
                return "NO_ACTION", "Hold long, weight change < 5%", "END_OF_DAY"
            elif tgt_dir == "FLAT":
                return "SELL", "Close long, signal turned Flat", "NEXT_OPEN"
            elif tgt_dir == "SHORT":
                return "SELL", "Reverse: close long + open short", "IMMEDIATE"

        # --- SHORT -> X ---
        if cur_dir == "SHORT":
            if tgt_dir == "SHORT":
                if abs_delta >= REBALANCE_THRESHOLD:
                    direction = "increase" if delta < 0 else "decrease"
                    return (
                        "SELL_SHORT" if delta < 0 else "BUY_TO_COVER",
                        f"Rebalance short ({direction} "
                        f"{abs(cur_weight)*100:.1f}% -> {abs(tgt_weight)*100:.1f}%)",
                        "NEXT_OPEN",
                    )
                return "NO_ACTION", "Hold short, weight change < 5%", "END_OF_DAY"
            elif tgt_dir == "FLAT":
                return "BUY_TO_COVER", "Close short, signal turned Flat", "NEXT_OPEN"
            elif tgt_dir == "LONG":
                return "BUY_TO_COVER", "Reverse: cover short + open long", "IMMEDIATE"

        # Fallback
        return "NO_ACTION", "No action required", "END_OF_DAY"


# ---------------------------------------------------------------------------
# Risk checks
# ---------------------------------------------------------------------------

class RiskChecker:
    """Pre-execution risk checks.  All methods return (passed: bool, msg: str)."""

    def __init__(
        self,
        positions: Dict[str, PositionState],
        portfolio_value: float = 100_000,
        portfolio_high_water_mark: Optional[float] = None,
    ) -> None:
        self._positions = positions
        self._portfolio_value = portfolio_value
        self._hwm = portfolio_high_water_mark or portfolio_value

    # -- Individual checks ---------------------------------------------------

    def check_max_position(
        self, ticker: str, target_weight: float,
    ) -> Tuple[bool, str]:
        if abs(target_weight) > MAX_POSITION_PER_TICKER:
            return False, (
                f"{ticker}: target weight {abs(target_weight)*100:.1f}% exceeds "
                f"{MAX_POSITION_PER_TICKER*100:.0f}% per-ticker limit"
            )
        return True, ""

    def check_gross_leverage(
        self, proposed_weights: Dict[str, float],
    ) -> Tuple[bool, str]:
        gross = sum(abs(w) for w in proposed_weights.values())
        if gross > MAX_GROSS_LEVERAGE:
            return False, (
                f"Gross leverage {gross:.2f}x exceeds "
                f"{MAX_GROSS_LEVERAGE:.1f}x limit"
            )
        return True, ""

    def check_single_day_turnover(
        self, orders: List[TradeOrder],
    ) -> Tuple[bool, str]:
        total_turnover = sum(o.quantity_pct for o in orders if o.action != "NO_ACTION")
        if total_turnover > MAX_SINGLE_DAY_TURNOVER:
            return False, (
                f"Single-day turnover {total_turnover*100:.1f}% exceeds "
                f"{MAX_SINGLE_DAY_TURNOVER*100:.0f}% limit"
            )
        return True, ""

    def check_whipsaw_protection(
        self, ticker: str, target_direction: str, today: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Block reversal if the current position was entered < N days ago."""
        pos = self._positions.get(ticker)
        if pos is None or pos.current_position == "FLAT":
            return True, ""

        # Only applies to actual reversals (LONG->SHORT or SHORT->LONG)
        is_reversal = (
            (pos.current_position == "LONG" and target_direction == "SHORT")
            or (pos.current_position == "SHORT" and target_direction == "LONG")
        )
        if not is_reversal:
            return True, ""

        days = pos.days_held(as_of=today)
        if days is not None and days < WHIPSAW_PROTECTION_DAYS:
            return False, (
                f"{ticker}: reversal blocked -- position only {days}d old "
                f"(minimum {WHIPSAW_PROTECTION_DAYS}d)"
            )
        return True, ""

    def check_drawdown_circuit_breaker(
        self, current_portfolio_value: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """If portfolio is in deep drawdown, return (triggered, scale_factor).

        When triggered, all target weights should be multiplied by the returned
        scale factor (e.g. 0.5 to cut exposure in half).
        """
        pv = current_portfolio_value or self._portfolio_value
        if self._hwm <= 0:
            return False, 1.0
        dd = 1.0 - (pv / self._hwm)
        if dd >= DRAWDOWN_CIRCUIT_BREAKER:
            return True, DRAWDOWN_REDUCTION_FACTOR
        return False, 1.0

    # -- Aggregate check -----------------------------------------------------

    def run_all_checks(
        self,
        orders: List[TradeOrder],
        proposed_weights: Dict[str, float],
        today: Optional[str] = None,
    ) -> List[str]:
        """Run every risk check and return a list of warning/block messages."""
        messages: List[str] = []

        # Per-ticker checks
        for ticker, weight in proposed_weights.items():
            ok, msg = self.check_max_position(ticker, weight)
            if not ok:
                messages.append(f"RISK BLOCK: {msg}")

        # Gross leverage
        ok, msg = self.check_gross_leverage(proposed_weights)
        if not ok:
            messages.append(f"RISK BLOCK: {msg}")

        # Turnover
        ok, msg = self.check_single_day_turnover(orders)
        if not ok:
            messages.append(f"RISK WARN: {msg}")

        # Whipsaw per order
        for order in orders:
            ok, msg = self.check_whipsaw_protection(
                order.ticker, order.target_direction, today=today,
            )
            if not ok:
                messages.append(f"RISK BLOCK: {msg}")

        return messages


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------

class PositionManager:
    """Manages the full lifecycle of portfolio positions.

    Responsibilities
    ----------------
    - Load / save position state from disk (``positions_state.json``)
    - Compare current positions to new target signals
    - Generate trade orders with proper handling of reversals, rebalancing,
      whipsaw protection, and risk limits
    - Provide a human-readable execution plan
    - Log trades for audit (``trade_log.jsonl``)
    """

    def __init__(
        self,
        state_file: Optional[Path] = None,
        portfolio_value: float = 100_000,
    ) -> None:
        self._state_file = state_file or POSITIONS_STATE_FILE
        self._positions: Dict[str, PositionState] = {}
        self._metadata: Dict[str, Any] = {}
        self._portfolio_value = portfolio_value
        self._load_state()

    # =====================================================================
    # State persistence
    # =====================================================================

    def _load_state(self) -> None:
        """Load position state from disk."""
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                self._metadata = data.get("metadata", {})
                for ticker, pos_dict in data.get("positions", {}).items():
                    self._positions[ticker] = PositionState.from_dict(pos_dict)
                logger.info(
                    "Loaded %d positions from %s (last updated: %s)",
                    len(self._positions),
                    self._state_file,
                    self._metadata.get("last_updated", "unknown"),
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Failed to load position state: %s.  Starting fresh.", exc,
                )
                self._positions = {}
                self._metadata = {}
        else:
            logger.info("No existing position state found.  Starting fresh (all FLAT).")

    def _save_state(self) -> None:
        """Persist position state to disk."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._metadata["last_updated"] = datetime.now().isoformat()
        data = {
            "metadata": self._metadata,
            "positions": {
                ticker: pos.to_dict()
                for ticker, pos in self._positions.items()
            },
        }
        self._state_file.write_text(json.dumps(data, indent=2) + "\n")
        logger.info("Position state saved to %s", self._state_file)

    def _append_trade_log(self, orders: List[TradeOrder]) -> None:
        """Append executed orders to the trade log (JSONL format)."""
        log_file = TRADE_LOG_FILE
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as fh:
            for order in orders:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    **order.to_dict(),
                }
                fh.write(json.dumps(entry) + "\n")

    # =====================================================================
    # Public accessors
    # =====================================================================

    def get_current_positions(self) -> Dict[str, PositionState]:
        """Return a copy of current positions (keyed by ticker)."""
        return dict(self._positions)

    def get_position(self, ticker: str) -> Optional[PositionState]:
        """Get a single position, or None if not held."""
        return self._positions.get(ticker)

    # =====================================================================
    # Core: generate trade orders
    # =====================================================================

    def generate_trade_orders(
        self,
        new_signals: Dict[str, Tuple[str, float]],
        current_prices: Dict[str, float],
        portfolio_value: Optional[float] = None,
        strategy: str = "ensemble",
        today: Optional[str] = None,
    ) -> List[TradeOrder]:
        """Compare current positions to target signals and generate orders.

        Parameters
        ----------
        new_signals : dict
            ``{ticker: (direction_str, target_weight)}`` where direction is
            ``"LONG"`` / ``"SHORT"`` / ``"FLAT"`` and target_weight is signed
            (positive for long, negative for short).
        current_prices : dict
            ``{ticker: latest_price}`` for notional estimation and P&L.
        portfolio_value : float, optional
            Total portfolio value.  Defaults to the value set at init.
        strategy : str
            Label of the strategy producing these signals.
        today : str, optional
            ISO date string for "today".  Defaults to now.

        Returns
        -------
        list of TradeOrder
        """
        pv = portfolio_value or self._portfolio_value
        today_str = today or datetime.now().strftime("%Y-%m-%d")

        # --- Update unrealized P&L on existing positions ---
        self._update_unrealized_pnl(current_prices, pv)

        # --- Drawdown circuit breaker ---
        hwm = self._metadata.get("high_water_mark", pv)
        rc = RiskChecker(self._positions, pv, hwm)
        dd_triggered, dd_scale = rc.check_drawdown_circuit_breaker(pv)
        if dd_triggered:
            logger.warning(
                "DRAWDOWN CIRCUIT BREAKER triggered (DD > %d%%).  "
                "Reducing all target weights to %.0f%%.",
                int(DRAWDOWN_CIRCUIT_BREAKER * 100),
                dd_scale * 100,
            )

        # --- Build raw orders ---
        orders: List[TradeOrder] = []
        all_tickers = sorted(set(self._positions.keys()) | set(new_signals.keys()))
        proposed_weights: Dict[str, float] = {}

        for ticker in all_tickers:
            # Current state
            cur = self._positions.get(ticker)
            cur_weight = cur.current_weight if cur else 0.0
            cur_dir = cur.current_position if cur else "FLAT"

            # Target state
            if ticker in new_signals:
                tgt_dir, tgt_weight = new_signals[ticker]
            else:
                # Ticker not in new signals -> close it
                tgt_dir = "FLAT"
                tgt_weight = 0.0

            # Apply drawdown scale to target
            if dd_triggered and tgt_dir != "FLAT":
                tgt_weight *= dd_scale

            # Normalise direction from weight when caller provides only weight
            if tgt_dir == "FLAT" and abs(tgt_weight) >= MIN_POSITION_WEIGHT:
                tgt_dir = "LONG" if tgt_weight > 0 else "SHORT"
            if abs(tgt_weight) < MIN_POSITION_WEIGHT:
                tgt_dir = "FLAT"
                tgt_weight = 0.0

            proposed_weights[ticker] = tgt_weight

            # Classify action
            action, reason, urgency = PositionTransition.classify(
                cur_dir, tgt_dir, cur_weight, tgt_weight,
            )

            if action == "NO_ACTION":
                continue

            delta = tgt_weight - cur_weight
            quantity_pct = abs(delta)

            # For reversals the quantity is the full round-trip
            if "Reverse" in reason:
                quantity_pct = abs(cur_weight) + abs(tgt_weight)

            # Skip trivially small trades
            if quantity_pct < MIN_TRADE_THRESHOLD:
                continue

            price = current_prices.get(ticker)
            notional = quantity_pct * pv

            orders.append(TradeOrder(
                ticker=ticker,
                action=action,
                quantity_pct=round(quantity_pct, 6),
                reason=reason,
                urgency=urgency,
                strategy=strategy,
                current_weight=round(cur_weight, 6),
                target_weight=round(tgt_weight, 6),
                delta_weight=round(delta, 6),
                current_direction=cur_dir,
                target_direction=tgt_dir,
                estimated_notional=round(notional, 2) if notional else None,
                price=round(price, 4) if price else None,
            ))

        # --- Risk checks ---
        risk_checker = RiskChecker(self._positions, pv, hwm)
        risk_msgs = risk_checker.run_all_checks(orders, proposed_weights, today=today_str)

        # Clamp positions that exceed per-ticker limit
        orders = self._apply_position_caps(orders, pv)

        # Clamp gross leverage
        orders = self._apply_leverage_cap(orders, proposed_weights, pv)

        # Attach risk messages to metadata so they can be shown in the plan
        self._metadata["last_risk_messages"] = risk_msgs

        return orders

    # =====================================================================
    # Post-execution state update
    # =====================================================================

    def update_positions(
        self,
        executed_orders: List[TradeOrder],
        execution_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, PositionState]:
        """Mark *executed_orders* as filled and update internal state.

        Parameters
        ----------
        executed_orders : list of TradeOrder
        execution_prices : dict, optional
            ``{ticker: fill_price}``.  Falls back to each order's ``.price``.

        Returns
        -------
        dict of updated PositionState objects (keyed by ticker).
        """
        now_str = datetime.now().isoformat()
        today_str = datetime.now().strftime("%Y-%m-%d")

        for order in executed_orders:
            ticker = order.ticker
            fill_price = (
                (execution_prices or {}).get(ticker) or order.price
            )

            if order.target_direction == "FLAT" or abs(order.target_weight) < MIN_POSITION_WEIGHT:
                # Position closed
                if ticker in self._positions:
                    del self._positions[ticker]
            else:
                existing = self._positions.get(ticker)
                # If direction stays the same, keep the original entry info
                if existing and existing.current_position == order.target_direction:
                    existing.current_weight = order.target_weight
                    existing.last_signal_date = today_str
                    if fill_price is not None:
                        # Weighted average entry price for scaling
                        existing.entry_price = fill_price
                else:
                    # New position or reversal -> fresh entry
                    self._positions[ticker] = PositionState(
                        ticker=ticker,
                        current_position=order.target_direction,
                        current_weight=order.target_weight,
                        entry_date=today_str,
                        entry_price=fill_price,
                        unrealized_pnl=0.0,
                        last_signal_date=today_str,
                        strategy=order.strategy,
                    )

        # Update high water mark
        hwm = self._metadata.get("high_water_mark", self._portfolio_value)
        if self._portfolio_value > hwm:
            self._metadata["high_water_mark"] = self._portfolio_value

        # Clear pending
        self._metadata.pop("pending_orders", None)
        self._metadata.pop("pending_since", None)
        self._save_state()
        self._append_trade_log(executed_orders)
        logger.info("Marked %d orders as executed.  Positions updated.", len(executed_orders))

        return dict(self._positions)

    # =====================================================================
    # Display
    # =====================================================================

    def print_current_positions(self) -> None:
        """Print a formatted table of current positions."""
        positions = self.get_current_positions()

        print("\n" + "=" * 74)
        print("  CURRENT POSITIONS")
        print("=" * 74)

        if not positions:
            print("  (no open positions)")
            print("=" * 74 + "\n")
            return

        print(
            f"  {'Ticker':<8} {'Dir':<7} {'Weight':>9}  "
            f"{'Entry':>9}  {'Unreal P&L':>11}  {'Days':>5}  {'Signal Date'}"
        )
        print(
            f"  {'-'*8} {'-'*7} {'-'*9}  "
            f"{'-'*9}  {'-'*11}  {'-'*5}  {'-'*12}"
        )

        gross = 0.0
        net = 0.0
        for ticker in sorted(positions.keys()):
            pos = positions[ticker]
            if pos.abs_weight < MIN_POSITION_WEIGHT:
                continue
            gross += pos.abs_weight
            net += pos.current_weight
            entry_str = f"${pos.entry_price:.2f}" if pos.entry_price else "N/A"
            pnl_str = f"${pos.unrealized_pnl:+,.0f}" if pos.unrealized_pnl is not None else "N/A"
            days = pos.days_held()
            days_str = str(days) if days is not None else "N/A"
            sig_str = pos.last_signal_date or "N/A"
            print(
                f"  {pos.ticker:<8} {pos.current_position:<7} "
                f"{pos.current_weight:>+9.4f}  "
                f"{entry_str:>9}  {pnl_str:>11}  {days_str:>5}  {sig_str}"
            )

        print(f"\n  Gross Exposure: {gross:.3f}x")
        print(f"  Net Exposure:   {net:+.3f}")

        pending = self._metadata.get("pending_orders")
        if pending:
            since = self._metadata.get("pending_since", "unknown")
            print(f"\n  WARNING: {len(pending)} pending orders since {since}")

        risk_msgs = self._metadata.get("last_risk_messages", [])
        if risk_msgs:
            print(f"\n  RISK MESSAGES:")
            for msg in risk_msgs:
                print(f"    {msg}")

        print("=" * 74 + "\n")

    def print_execution_plan(
        self,
        orders: List[TradeOrder],
        portfolio_value: Optional[float] = None,
    ) -> None:
        """Print a clear, actionable execution plan."""
        pv = portfolio_value or self._portfolio_value
        today_str = datetime.now().strftime("%Y-%m-%d")

        print()
        print("=" * 74)
        print(f"  EXECUTION PLAN for {today_str}")
        print(f"  Portfolio Value: ${pv:,.0f} (estimated)")
        print("=" * 74)

        if not orders:
            print("  (no trades required -- portfolio is on target)")
            self._print_unchanged_positions({})
            print("=" * 74 + "\n")
            return

        # --- Trades to execute ---
        print("\n  TRADES TO EXECUTE:")
        total_turnover_pct = 0.0
        total_turnover_usd = 0.0
        action_counts: Dict[str, int] = {}

        for i, o in enumerate(orders, 1):
            notional = o.estimated_notional or (o.quantity_pct * pv)
            sign = "+" if o.delta_weight > 0 else "-"
            pct_display = abs(o.delta_weight) * 100
            price_str = f"@ ${o.price:.2f}" if o.price else ""

            # Build a concise action label
            action_label = o.action.replace("_", " ")
            print(
                f"    {i:>2}. {action_label:<14} {o.ticker:<6} "
                f"{sign}{pct_display:.1f}% (${notional:,.0f})  "
                f"[{o.reason}] {price_str}"
            )

            total_turnover_pct += o.quantity_pct
            total_turnover_usd += notional
            action_counts[o.action] = action_counts.get(o.action, 0) + 1

        # --- Positions unchanged ---
        traded_tickers = {o.ticker for o in orders}
        self._print_unchanged_positions(traded_tickers)

        # --- Estimated costs ---
        tx_cost = total_turnover_usd * (TRANSACTION_COST_BPS / 10_000)
        print(f"\n  ESTIMATED COSTS:")
        print(f"    Total turnover: {total_turnover_pct*100:.1f}% (${total_turnover_usd:,.0f})")
        print(f"    Transaction costs: ~${tx_cost:,.0f} ({TRANSACTION_COST_BPS} bps)")

        # --- Risk summary ---
        # Compute post-trade state
        proposed: Dict[str, float] = {}
        for ticker, pos in self._positions.items():
            proposed[ticker] = pos.current_weight
        for o in orders:
            proposed[o.ticker] = o.target_weight

        gross = sum(abs(w) for w in proposed.values())
        net = sum(w for w in proposed.values())
        largest_ticker = max(proposed, key=lambda t: abs(proposed[t])) if proposed else "N/A"
        largest_weight = abs(proposed.get(largest_ticker, 0))

        print(f"\n  RISK SUMMARY:")
        print(f"    Gross leverage after trades: {gross:.2f}x (limit: {MAX_GROSS_LEVERAGE:.1f}x)")
        print(f"    Largest position: {largest_ticker} {largest_weight*100:.1f}% (limit: {MAX_POSITION_PER_TICKER*100:.0f}%)")
        print(f"    Net exposure: {net:+.2f} ({'long-biased' if net > 0.05 else 'short-biased' if net < -0.05 else 'neutral'})")
        print(f"    Single-day turnover: {total_turnover_pct*100:.1f}% (limit: {MAX_SINGLE_DAY_TURNOVER*100:.0f}%)")

        # Show risk alerts
        risk_msgs = self._metadata.get("last_risk_messages", [])
        if risk_msgs:
            print(f"\n  RISK ALERTS:")
            for msg in risk_msgs:
                print(f"    {msg}")

        print("=" * 74 + "\n")

    def _print_unchanged_positions(self, traded_tickers: set) -> None:
        """Print positions that are being held without change."""
        unchanged = [
            pos for ticker, pos in sorted(self._positions.items())
            if ticker not in traded_tickers and pos.abs_weight >= MIN_POSITION_WEIGHT
        ]
        if unchanged:
            print("\n  POSITIONS UNCHANGED:")
            for pos in unchanged:
                print(
                    f"    - {pos.ticker}: {pos.current_position} "
                    f"{pos.abs_weight*100:.1f}% (holding, signal confirmed)"
                )

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _update_unrealized_pnl(
        self, current_prices: Dict[str, float], portfolio_value: float,
    ) -> None:
        """Recompute unrealized P&L for all open positions."""
        for ticker, pos in self._positions.items():
            price = current_prices.get(ticker)
            if price is None or pos.entry_price is None:
                continue
            # P&L as % of entry, scaled by position weight and portfolio value
            if pos.current_position == "LONG":
                pnl_pct = (price - pos.entry_price) / pos.entry_price
            elif pos.current_position == "SHORT":
                pnl_pct = (pos.entry_price - price) / pos.entry_price
            else:
                pnl_pct = 0.0
            pos.unrealized_pnl = round(pnl_pct * pos.abs_weight * portfolio_value, 2)

    def _apply_position_caps(
        self, orders: List[TradeOrder], portfolio_value: float,
    ) -> List[TradeOrder]:
        """Clamp any order whose target weight exceeds the per-ticker cap."""
        capped: List[TradeOrder] = []
        for o in orders:
            if abs(o.target_weight) > MAX_POSITION_PER_TICKER:
                sign = 1 if o.target_weight > 0 else -1
                capped_weight = sign * MAX_POSITION_PER_TICKER
                delta = capped_weight - o.current_weight
                logger.warning(
                    "Capping %s target from %.1f%% to %.1f%%",
                    o.ticker,
                    o.target_weight * 100,
                    capped_weight * 100,
                )
                capped.append(TradeOrder(
                    ticker=o.ticker,
                    action=o.action,
                    quantity_pct=round(abs(delta), 6),
                    reason=o.reason + f" [capped to {MAX_POSITION_PER_TICKER*100:.0f}%]",
                    urgency=o.urgency,
                    strategy=o.strategy,
                    current_weight=o.current_weight,
                    target_weight=round(capped_weight, 6),
                    delta_weight=round(delta, 6),
                    current_direction=o.current_direction,
                    target_direction=o.target_direction,
                    estimated_notional=round(abs(delta) * portfolio_value, 2),
                    price=o.price,
                ))
            else:
                capped.append(o)
        return capped

    def _apply_leverage_cap(
        self,
        orders: List[TradeOrder],
        proposed_weights: Dict[str, float],
        portfolio_value: float,
    ) -> List[TradeOrder]:
        """Scale down all orders proportionally if gross leverage exceeds limit."""
        # Build post-trade weight map
        post_trade: Dict[str, float] = {}
        for ticker, pos in self._positions.items():
            post_trade[ticker] = pos.current_weight
        for o in orders:
            post_trade[o.ticker] = o.target_weight

        gross = sum(abs(w) for w in post_trade.values())
        if gross <= MAX_GROSS_LEVERAGE:
            return orders

        scale = MAX_GROSS_LEVERAGE / gross
        logger.warning(
            "Gross leverage %.2fx exceeds %.1fx limit.  Scaling all targets by %.2f.",
            gross, MAX_GROSS_LEVERAGE, scale,
        )

        scaled: List[TradeOrder] = []
        for o in orders:
            new_target = o.target_weight * scale
            new_delta = new_target - o.current_weight
            if abs(new_delta) < MIN_TRADE_THRESHOLD:
                continue
            scaled.append(TradeOrder(
                ticker=o.ticker,
                action=o.action,
                quantity_pct=round(abs(new_delta), 6),
                reason=o.reason + f" [leverage-scaled x{scale:.2f}]",
                urgency=o.urgency,
                strategy=o.strategy,
                current_weight=o.current_weight,
                target_weight=round(new_target, 6),
                delta_weight=round(new_delta, 6),
                current_direction=o.current_direction,
                target_direction=o.target_direction,
                estimated_notional=round(abs(new_delta) * portfolio_value, 2),
                price=o.price,
            ))
        return scaled

    # =====================================================================
    # Convenience: save pending orders
    # =====================================================================

    def save_target_state(self, orders: List[TradeOrder]) -> None:
        """Persist the list of pending orders (pre-execution) so we can
        detect stale orders on the next run.
        """
        self._metadata["pending_orders"] = [o.to_dict() for o in orders]
        self._metadata["pending_since"] = datetime.now().isoformat()
        self._save_state()
        logger.info("Saved %d pending orders to state.", len(orders))


# ---------------------------------------------------------------------------
# Module-level convenience functions (for import from automate.py)
# ---------------------------------------------------------------------------

_default_pm: Optional[PositionManager] = None


def _get_pm() -> PositionManager:
    global _default_pm
    if _default_pm is None:
        _default_pm = PositionManager()
    return _default_pm


def generate_trade_orders(
    new_signals: Dict[str, Tuple[str, float]],
    current_prices: Dict[str, float],
    portfolio_value: float = 100_000,
    strategy: str = "ensemble",
) -> List[TradeOrder]:
    """Module-level convenience wrapper around PositionManager.generate_trade_orders."""
    return _get_pm().generate_trade_orders(
        new_signals, current_prices,
        portfolio_value=portfolio_value, strategy=strategy,
    )


def update_positions(
    executed_orders: List[TradeOrder],
    execution_prices: Optional[Dict[str, float]] = None,
) -> Dict[str, PositionState]:
    """Module-level convenience wrapper around PositionManager.update_positions."""
    return _get_pm().update_positions(executed_orders, execution_prices)


def get_current_positions() -> Dict[str, PositionState]:
    """Module-level convenience wrapper around PositionManager.get_current_positions."""
    return _get_pm().get_current_positions()


def print_execution_plan(
    orders: List[TradeOrder],
    portfolio_value: float = 100_000,
) -> None:
    """Module-level convenience wrapper around PositionManager.print_execution_plan."""
    _get_pm().print_execution_plan(orders, portfolio_value=portfolio_value)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """Simple CLI for checking / managing positions."""
    import sys

    pm = PositionManager()

    if "--status" in sys.argv:
        pm.print_current_positions()
    elif "--clear" in sys.argv:
        pm._positions.clear()
        pm._metadata.pop("pending_orders", None)
        pm._metadata.pop("pending_since", None)
        pm._metadata.pop("last_risk_messages", None)
        pm._save_state()
        print("All positions cleared.")
    elif "--demo" in sys.argv:
        # Quick demo: generate orders from fake signals
        demo_signals: Dict[str, Tuple[str, float]] = {
            "SPY": ("LONG", 0.052),
            "TLT": ("FLAT", 0.0),
            "GLD": ("LONG", 0.112),
            "XLF": ("LONG", 0.100),
            "XLE": ("SHORT", -0.045),
        }
        demo_prices = {
            "SPY": 520.00, "TLT": 98.50, "GLD": 245.00,
            "XLF": 42.30, "XLE": 88.10,
        }
        orders = pm.generate_trade_orders(
            demo_signals, demo_prices, portfolio_value=100_000, strategy="demo",
        )
        pm.print_execution_plan(orders, portfolio_value=100_000)
    else:
        pm.print_current_positions()


if __name__ == "__main__":
    _cli_main()
