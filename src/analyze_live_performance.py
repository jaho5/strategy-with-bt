"""
Analyze how the strategy has actually performed since the first live signal
file was generated (2026-03-25), using the daily portfolio_positions_YYYY-MM-DD.csv
snapshots as the authoritative record of held weights.

Methodology:
    * Each portfolio_positions_<d>.csv records weights *as of the close of d*,
      to be traded at the next open. We therefore apply weights from file dated
      d to the close-to-close return from d -> next trading day.
    * Returns use yfinance auto-adjusted closes (dividends reinvested, splits).
    * We report gross return (no friction) and a simple after-cost variant that
      charges 1 bp per unit of turnover per side.
"""

from __future__ import annotations

import glob
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.downloader import download_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-6s  %(message)s")
log = logging.getLogger("live-perf")

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"


def load_position_snapshots() -> pd.DataFrame:
    """Return weights DataFrame indexed by date, one column per ticker."""
    rows = []
    for path in sorted(glob.glob(str(REPORTS / "portfolio_positions_*.csv"))):
        m = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", path)
        if not m:
            continue
        d = pd.Timestamp(m.group(1))
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({"date": d, "ticker": r["ticker"], "w": float(r["target_weight"])})
    long = pd.DataFrame(rows)
    wide = long.pivot(index="date", columns="ticker", values="w").fillna(0.0)
    wide = wide.sort_index()
    return wide


def main() -> None:
    weights = load_position_snapshots()
    log.info("Loaded %d snapshots over %s → %s; universe=%d",
             len(weights), weights.index.min().date(), weights.index.max().date(),
             weights.shape[1])

    tickers = list(weights.columns)
    start = (weights.index.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prices = download_universe(tickers, start, end, use_cache=False)

    closes = prices["Close"][tickers].dropna(how="all").sort_index()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    rets = closes.pct_change().dropna(how="all")

    # Align weights to trading days; weights are "as of close of d", applied to
    # return from close(d) to close(d+1).
    w = weights.reindex(rets.index, method="ffill").fillna(0.0)
    w_lag = w.shift(1).fillna(0.0)

    # Only include trading days from the first position snapshot onwards.
    first = weights.index.min()
    mask = rets.index >= first
    rets = rets.loc[mask]
    w_lag = w_lag.loc[mask]

    # Gross daily portfolio return.
    port_ret = (w_lag * rets).sum(axis=1)
    port_ret = port_ret.dropna()

    # Turnover (L1 change) for a simple transaction cost estimate.
    w_change = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost_bps = 1.0  # 1 bp per dollar traded per side
    daily_cost = (cost_bps / 1e4) * w_change.reindex(port_ret.index).fillna(0.0)
    port_ret_net = port_ret - daily_cost

    equity_gross = (1.0 + port_ret).cumprod()
    equity_net = (1.0 + port_ret_net).cumprod()

    # Benchmark: SPY buy-and-hold over the same window.
    spy_ret = rets["SPY"].loc[port_ret.index]
    spy_equity = (1.0 + spy_ret).cumprod()

    days = len(port_ret)
    ann_factor = 252.0 / max(days, 1)

    def stats(r: pd.Series, eq: pd.Series) -> dict:
        total = float(eq.iloc[-1] - 1.0)
        ann = float((1.0 + total) ** ann_factor - 1.0)
        vol = float(r.std() * np.sqrt(252))
        sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
        dd = float((eq / eq.cummax() - 1.0).min())
        return {"total_return": total, "annualized": ann, "vol": vol,
                "sharpe": sharpe, "max_drawdown": dd}

    summary = {
        "period_start": str(port_ret.index.min().date()),
        "period_end": str(port_ret.index.max().date()),
        "trading_days": int(days),
        "calendar_days": int((port_ret.index.max() - port_ret.index.min()).days),
        "strategy_gross": stats(port_ret, equity_gross),
        "strategy_net_1bp": stats(port_ret_net, equity_net),
        "spy_buy_hold": stats(spy_ret, spy_equity),
    }
    print(json.dumps(summary, indent=2))

    # Daily table
    table = pd.DataFrame({
        "port_gross_ret": port_ret,
        "port_net_ret": port_ret_net,
        "spy_ret": spy_ret,
        "equity_gross": equity_gross,
        "equity_net": equity_net,
        "spy_equity": spy_equity,
        "turnover": w_change.reindex(port_ret.index).fillna(0.0),
    })
    out = REPORTS / "live_performance.csv"
    table.to_csv(out, float_format="%.6f")
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
