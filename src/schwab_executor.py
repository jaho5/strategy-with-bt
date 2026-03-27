"""Execute trade orders against a real Charles Schwab brokerage account.

Uses schwab-py to authenticate, fetch account data, and place market orders.

Usage:
    # One-time auth setup (interactive, needs browser):
    uv run python -m src.schwab_executor --setup

    # Dry-run to verify orders without executing:
    uv run python -m src.schwab_executor --dry-run

    # Execute pending orders from positions_state.json:
    uv run python -m src.schwab_executor
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = PROJECT_ROOT / "reports" / "positions_state.json"
EXECUTION_LOG_DIR = PROJECT_ROOT / "reports"

load_dotenv(PROJECT_ROOT / ".env")


def _get_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Missing environment variable: {key}. See .env.example")
    return val


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    ticker: str
    action: str
    requested_shares: int
    status: str  # FILLED, REJECTED, SKIPPED, ERROR, DRY_RUN
    fill_price: Optional[float] = None
    order_id: Optional[str] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Schwab Executor
# ---------------------------------------------------------------------------

class SchwabExecutor:
    """Connects to Schwab API, reads account, and executes trade orders."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._client = None
        self._account_hash: Optional[str] = None

    # -- Authentication ------------------------------------------------------

    def authenticate(self):
        """Load token and create an authenticated Schwab client."""
        import schwab.auth

        token_path = os.environ.get("SCHWAB_TOKEN_PATH", "schwab_token.json")
        # Resolve relative to project root
        if not os.path.isabs(token_path):
            token_path = str(PROJECT_ROOT / token_path)

        api_key = _get_env("SCHWAB_API_KEY")
        app_secret = _get_env("SCHWAB_APP_SECRET")

        if not Path(token_path).exists():
            raise RuntimeError(
                f"Token file not found: {token_path}\n"
                "Run: uv run python -m src.schwab_executor --setup"
            )

        self._client = schwab.auth.client_from_token_file(
            token_path, api_key, app_secret,
        )
        logger.info("Schwab API authenticated successfully.")
        return self._client

    @staticmethod
    def interactive_setup() -> None:
        """One-time interactive OAuth setup. Requires a browser."""
        import schwab.auth

        api_key = _get_env("SCHWAB_API_KEY")
        app_secret = _get_env("SCHWAB_APP_SECRET")
        callback_url = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1")
        token_path = os.environ.get("SCHWAB_TOKEN_PATH", "schwab_token.json")
        if not os.path.isabs(token_path):
            token_path = str(PROJECT_ROOT / token_path)

        print("=" * 60)
        print("Schwab OAuth Setup")
        print("=" * 60)
        print(f"API Key:      {api_key[:8]}...")
        print(f"Callback URL: {callback_url}")
        print(f"Token file:   {token_path}")
        print()
        print("A browser window will open. Log in to Schwab and authorize.")
        print("After authorization, you'll be redirected. Paste the full")
        print("redirect URL below when prompted.")
        print("=" * 60)

        client = schwab.auth.client_from_manual_flow(
            api_key, app_secret, callback_url, token_path,
        )

        # Verify it works
        resp = client.get_account_numbers()
        accounts = resp.json()
        print(f"\nSuccess! Found {len(accounts)} account(s):")
        for acc in accounts:
            if isinstance(acc, dict):
                print(f"  - {acc.get('accountNumber', 'N/A')}")
            else:
                print(f"  - {acc}")
        print(f"\nToken saved to: {token_path}")

    # -- Account data --------------------------------------------------------

    def _ensure_account_hash(self) -> str:
        if self._account_hash:
            return self._account_hash
        resp = self._client.get_account_numbers()
        resp.raise_for_status()
        accounts = resp.json()
        if not accounts:
            raise RuntimeError("No accounts found in Schwab API response.")
        first = accounts[0]
        if isinstance(first, dict):
            self._account_hash = first["hashValue"]
            logger.info("Using account: %s", first.get("accountNumber", "N/A"))
        else:
            # API returned flat list of account hashes
            self._account_hash = str(first)
            logger.info("Using account hash: %s", self._account_hash)
        return self._account_hash

    def get_portfolio_value(self) -> float:
        """Fetch total liquidation value of the account."""
        from schwab.client import Client

        account_hash = self._ensure_account_hash()
        resp = self._client.get_account(account_hash, fields=Client.Account.Fields.POSITIONS)
        resp.raise_for_status()
        data = resp.json()

        balances = data.get("securitiesAccount", {}).get("currentBalances", {})
        value = balances.get("liquidationValue", 0.0)
        if value <= 0:
            # Fallback: try totalCash + longMarketValue
            cash = balances.get("totalCash", balances.get("cashBalance", 0.0))
            long_val = balances.get("longMarketValue", 0.0)
            short_val = balances.get("shortMarketValue", 0.0)
            value = cash + long_val - abs(short_val)

        logger.info("Portfolio value: $%.2f", value)
        return float(value)

    def get_current_holdings(self) -> Dict[str, int]:
        """Fetch current share holdings from the account. {ticker: signed_shares}"""
        from schwab.client import Client

        account_hash = self._ensure_account_hash()
        resp = self._client.get_account(account_hash, fields=Client.Account.Fields.POSITIONS)
        resp.raise_for_status()
        data = resp.json()

        positions = data.get("securitiesAccount", {}).get("positions", [])
        holdings: Dict[str, int] = {}
        for pos in positions:
            instrument = pos.get("instrument", {})
            symbol = instrument.get("symbol", "")
            if instrument.get("assetType") == "EQUITY" and symbol:
                qty = int(pos.get("longQuantity", 0)) - int(pos.get("shortQuantity", 0))
                if qty != 0:
                    holdings[symbol] = qty
        return holdings

    def get_quotes(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch latest quotes for a list of tickers. Returns {ticker: price}."""
        if not tickers:
            return {}
        resp = self._client.get_quotes(tickers)
        resp.raise_for_status()
        data = resp.json()
        prices: Dict[str, float] = {}
        for ticker in tickers:
            quote_data = data.get(ticker, {})
            ref = quote_data.get("reference", {})
            quote = quote_data.get("quote", {})
            # Prefer lastPrice, fall back to closePrice
            price = quote.get("lastPrice") or quote.get("closePrice") or ref.get("lastPrice", 0.0)
            if price and price > 0:
                prices[ticker] = float(price)
        return prices

    # -- Order execution -----------------------------------------------------

    def execute_orders(
        self,
        orders: List[Dict[str, Any]],
        portfolio_value: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """Execute a list of trade order dicts from positions_state.json.

        Each order dict has: ticker, action, delta_weight, target_weight, price, etc.
        """
        from schwab.orders.equities import (
            equity_buy_market,
            equity_sell_market,
            equity_sell_short_market,
            equity_buy_to_cover_market,
        )

        if portfolio_value is None:
            portfolio_value = self.get_portfolio_value()

        account_hash = self._ensure_account_hash()

        # Get live quotes for accurate share calculation
        tickers = [o["ticker"] for o in orders]
        live_prices = self.get_quotes(tickers)

        results: List[ExecutionResult] = []

        for order in orders:
            ticker = order["ticker"]
            action = order["action"]
            delta_weight = abs(order.get("delta_weight", 0.0))

            if action == "NO_ACTION":
                continue

            # Use live price, fall back to stale price from signal generation
            price = live_prices.get(ticker, order.get("price", 0.0))
            if not price or price <= 0:
                results.append(ExecutionResult(
                    ticker=ticker, action=action, requested_shares=0,
                    status="ERROR", error_message="Could not get price",
                ))
                continue

            shares = math.floor(delta_weight * portfolio_value / price)
            if shares <= 0:
                logger.info("Skipping %s %s: calculated 0 shares (too small)", action, ticker)
                results.append(ExecutionResult(
                    ticker=ticker, action=action, requested_shares=0,
                    status="SKIPPED", error_message="Order too small for 1 share",
                ))
                continue

            # Build the order spec
            order_builders = {
                "BUY": equity_buy_market,
                "SELL": equity_sell_market,
                "SELL_SHORT": equity_sell_short_market,
                "BUY_TO_COVER": equity_buy_to_cover_market,
            }
            builder_fn = order_builders.get(action)
            if not builder_fn:
                results.append(ExecutionResult(
                    ticker=ticker, action=action, requested_shares=shares,
                    status="ERROR", error_message=f"Unknown action: {action}",
                ))
                continue

            order_spec = builder_fn(ticker, shares)

            logger.info(
                "%s %s %d shares @ ~$%.2f ($%.0f notional)%s",
                action, ticker, shares, price, shares * price,
                " [DRY RUN]" if self.dry_run else "",
            )

            if self.dry_run:
                results.append(ExecutionResult(
                    ticker=ticker, action=action, requested_shares=shares,
                    status="DRY_RUN", fill_price=price,
                ))
                continue

            # Place the order
            try:
                from schwab.utils import Utils

                resp = self._client.place_order(account_hash, order_spec)
                if resp.status_code == 201:
                    order_id = Utils(self._client, account_hash).extract_order_id(resp)

                    # Wait briefly and check fill
                    fill_price = self._poll_fill(account_hash, order_id, price)

                    results.append(ExecutionResult(
                        ticker=ticker, action=action, requested_shares=shares,
                        status="FILLED", fill_price=fill_price, order_id=order_id,
                    ))
                    logger.info("  -> FILLED %s %s %d shares (order %s)", action, ticker, shares, order_id)
                else:
                    error_text = resp.text[:200] if resp.text else f"HTTP {resp.status_code}"
                    results.append(ExecutionResult(
                        ticker=ticker, action=action, requested_shares=shares,
                        status="REJECTED", error_message=error_text,
                    ))
                    logger.warning("  -> REJECTED %s %s: %s", action, ticker, error_text)

            except Exception as e:
                results.append(ExecutionResult(
                    ticker=ticker, action=action, requested_shares=shares,
                    status="ERROR", error_message=str(e),
                ))
                logger.error("  -> ERROR %s %s: %s", action, ticker, e)

        return results

    def _poll_fill(
        self, account_hash: str, order_id: Optional[str], fallback_price: float,
    ) -> float:
        """Poll order status briefly to get actual fill price."""
        if not order_id:
            return fallback_price

        for _ in range(6):  # up to 3 seconds
            time.sleep(0.5)
            try:
                resp = self._client.get_order(order_id, account_hash)
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "")
                if status == "FILLED":
                    activities = data.get("orderActivityCollection", [])
                    for activity in activities:
                        legs = activity.get("executionLegs", [])
                        if legs:
                            return float(legs[0].get("price", fallback_price))
                    return fallback_price
                if status in ("CANCELED", "REJECTED", "EXPIRED"):
                    return fallback_price
            except Exception:
                pass

        return fallback_price


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _load_pending_orders() -> List[Dict[str, Any]]:
    """Load pending orders from positions_state.json."""
    if not STATE_FILE.exists():
        logger.error("No state file: %s", STATE_FILE)
        return []
    data = json.loads(STATE_FILE.read_text())
    orders = data.get("metadata", {}).get("pending_orders", [])
    if not orders:
        logger.info("No pending orders found.")
    return orders


def _log_results(results: List[ExecutionResult]) -> None:
    """Print and log execution results."""
    print(f"\n{'='*70}")
    print(f"{'EXECUTION RESULTS':^70}")
    print(f"{'='*70}")
    print(f"{'Ticker':<8} {'Action':<16} {'Shares':>8} {'Status':<10} {'Price':>10} {'Error'}")
    print(f"{'-'*70}")
    for r in results:
        price_str = f"${r.fill_price:.2f}" if r.fill_price else ""
        err_str = r.error_message or ""
        print(f"{r.ticker:<8} {r.action:<16} {r.requested_shares:>8} {r.status:<10} {price_str:>10} {err_str}")

    filled = [r for r in results if r.status == "FILLED"]
    skipped = [r for r in results if r.status in ("SKIPPED", "REJECTED")]
    errors = [r for r in results if r.status == "ERROR"]
    dry = [r for r in results if r.status == "DRY_RUN"]
    print(f"\nFilled: {len(filled)}  Skipped: {len(skipped)}  Errors: {len(errors)}  Dry-run: {len(dry)}")
    print(f"{'='*70}\n")

    # Save to log file
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = EXECUTION_LOG_DIR / f"execution_{today}.log"
    with open(log_path, "a") as fh:
        fh.write(f"\n--- {datetime.now().isoformat()} ---\n")
        for r in results:
            fh.write(f"{r.ticker} {r.action} {r.requested_shares}sh {r.status}")
            if r.fill_price:
                fh.write(f" @${r.fill_price:.2f}")
            if r.error_message:
                fh.write(f" [{r.error_message}]")
            fh.write("\n")


def main() -> None:
    if "--setup" in sys.argv:
        SchwabExecutor.interactive_setup()
        return

    dry_run = "--dry-run" in sys.argv

    executor = SchwabExecutor(dry_run=dry_run)
    executor.authenticate()

    orders = _load_pending_orders()
    if not orders:
        print("No pending orders to execute.")
        return

    portfolio_value = executor.get_portfolio_value()
    print(f"Portfolio value: ${portfolio_value:,.2f}")
    print(f"Pending orders: {len(orders)}")
    if dry_run:
        print("[DRY RUN MODE - no orders will be placed]")

    results = executor.execute_orders(orders, portfolio_value)
    _log_results(results)


if __name__ == "__main__":
    main()
