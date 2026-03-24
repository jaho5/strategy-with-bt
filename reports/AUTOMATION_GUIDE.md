# Automation & Position Management Guide

## Overview

The system automates the full lifecycle from signal generation to position management:

```
Market Close (4 PM ET)
    ↓
Download latest prices (yfinance)
    ↓
Run 5 component strategies
    ↓
Combine into InvVol Ensemble
    ↓
Load current positions (positions_state.json)
    ↓
Compare target vs current → Generate trade orders
    ↓
Risk checks (leverage, concentration, turnover, whipsaw)
    ↓
Print execution plan
    ↓
[Manual or automated execution via broker API]
    ↓
Mark orders as executed → Update state
```

## Daily Workflow

### 1. Generate Signals (automated)
```bash
uv run python -m src.automate
```
This produces:
- Target positions for each ETF
- Trade orders (what to buy/sell)
- Risk alerts

### 2. Review Execution Plan
The system prints a clear plan:
```
=== EXECUTION PLAN for 2026-03-23 ===

TRADES TO EXECUTE:
  1. BUY    GLD   +11.2% ($11,200)  [New long from Flat]
  2. SELL   TLT   -3.1% ($3,100)    [Close long, signal turned Flat]
  3. ADJUST XLF   +1.8% ($1,800)    [Increase long 8.2% → 10.0%]

POSITIONS UNCHANGED:
  - SLV: LONG 6.3% (holding, signal confirmed)
  - USO: LONG 2.6% (holding, signal confirmed)

RISK SUMMARY:
  - Gross leverage: 0.85x (limit: 1.5x)
  - Largest position: GLD 11.2% (limit: 20%)
```

### 3. Execute Trades (manual or automated)
After reviewing, execute and confirm:
```bash
uv run python -m src.automate --execute
```

### 4. Check Status Anytime
```bash
uv run python -m src.automate --status
```

## Position Transition Logic

| Current | New Signal | Action |
|---------|-----------|--------|
| FLAT | LONG | OPEN LONG (buy) |
| FLAT | SHORT | OPEN SHORT (sell short) |
| FLAT | FLAT | NO ACTION |
| LONG | LONG (same weight) | HOLD |
| LONG | LONG (different weight) | ADJUST (if change > 5%) |
| LONG | FLAT | CLOSE (sell) |
| LONG | SHORT | REVERSE (sell + short) |
| SHORT | SHORT (same weight) | HOLD |
| SHORT | FLAT | CLOSE (buy to cover) |
| SHORT | LONG | REVERSE (cover + buy) |

## Pre-existing Position Handling

### First Run
- System assumes all positions are FLAT
- All signals generate OPEN orders

### Subsequent Runs
- Loads state from `reports/positions_state.json`
- Compares new signals with stored positions
- Only generates orders for CHANGES

### Manual Intervention
If you manually adjust a position outside the system:
1. Edit `reports/positions_state.json` to reflect actual positions
2. Next run will diff against the updated state
3. Or use `--reset` to clear state and start fresh

### Position State File Format
```json
{
  "last_updated": "2026-03-23T16:30:00",
  "portfolio_value": 100000,
  "positions": {
    "GLD": {
      "direction": "LONG",
      "weight": 0.112,
      "entry_date": "2026-03-20",
      "entry_price": 185.42,
      "shares": 60
    },
    "SPY": {
      "direction": "FLAT",
      "weight": 0.0,
      "entry_date": null,
      "entry_price": null,
      "shares": 0
    }
  }
}
```

## Risk Controls

### Hard Limits (never exceeded)
- Max position per ticker: **20%**
- Max gross leverage: **1.5x**
- Max single-day turnover: **30%**

### Soft Limits (trigger alerts)
- Rolling 63-day Sharpe < 0.3 → WARNING
- Portfolio drawdown > 15% → reduce positions 50%
- Portfolio drawdown > 20% → flatten all positions
- No signal change for > 20 days → check for staleness

### Whipsaw Protection
- No reversals within 3 days of entry
- Weight changes < 5% are ignored (reduces turnover)
- Rebalancing only on strategy rebalance dates

## Scheduling

### Cron Setup (Linux)
```bash
# Run at 4:30 PM ET (after market close), Monday-Friday
30 16 * * 1-5 cd /home/jasonho/proj/strategy-with-bt && uv run python -m src.scheduler >> reports/scheduler.log 2>&1
```

### Systemd Timer (alternative)
```ini
[Unit]
Description=Strategy signal generation

[Timer]
OnCalendar=Mon..Fri 16:30 America/New_York
Persistent=true

[Install]
WantedBy=timers.target
```

## Strategy Health Monitoring

The scheduler tracks:
1. **Rolling Sharpe** — per strategy and ensemble
2. **Drawdown** — alerts if approaching limits
3. **Correlation drift** — if strategies become more correlated, diversification benefit degrades
4. **Return vs expectations** — compares live return to backtest expectations
5. **Signal staleness** — detects if a strategy stops generating new signals

## Broker Integration Points

The system generates trade orders but does NOT execute them directly. Integration options:

### Interactive Brokers (IBKR)
```python
# Example: convert TradeOrder to IBKR API call
from ib_insync import IB, Stock, MarketOrder

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

for order in trade_orders:
    contract = Stock(order.ticker, 'SMART', 'USD')
    ib_order = MarketOrder(
        'BUY' if order.action in ('BUY', 'BUY_TO_COVER') else 'SELL',
        abs(order.shares)
    )
    ib.placeOrder(contract, ib_order)
```

### Alpaca (paper trading)
```python
import alpaca_trade_api as tradeapi

api = tradeapi.REST(key_id, secret_key, base_url)

for order in trade_orders:
    api.submit_order(
        symbol=order.ticker,
        qty=abs(order.shares),
        side='buy' if order.action in ('BUY', 'BUY_TO_COVER') else 'sell',
        type='market',
        time_in_force='day'
    )
```

## Recovery Procedures

### Strategy Underperformance
If the ensemble's trailing 252-day annualized return drops below 30%:
1. Check individual component health
2. If one component is dragging: temporarily remove it from ensemble
3. Re-run leverage sweep to find new optimal leverage
4. Consider adding new uncorrelated strategies

### Data Issues
If yfinance download fails:
1. System uses cached data (Parquet files in .cache/)
2. Signals will be stale but positions won't change
3. Alert is generated

### State Corruption
If positions_state.json is corrupted:
1. Use `--reset` flag to clear state
2. Manually enter current positions
3. Or let the system assume FLAT and regenerate all orders
