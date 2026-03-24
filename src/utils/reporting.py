"""Comprehensive reporting module for strategy backtesting results.

Generates detailed markdown reports, comparison tables, and matplotlib
visualisations for individual strategies and cross-strategy comparisons.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend -- must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from ..backtest.engine import BacktestResult, MonteCarloResult, WalkForwardResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRADING_DAYS_PER_YEAR = 252
_PNL_TARGET_PCT = 45.0
_CONFIDENCE_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any, fmt: str = ".2f") -> str:
    """Format a numeric value, returning 'N/A' for NaN / None."""
    if value is None:
        return "N/A"
    try:
        if np.isnan(value) or np.isinf(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return f"{value:{fmt}}"


def _pct(value: Any) -> str:
    """Format a value as a percentage string (e.g. 12.34%)."""
    if value is None:
        return "N/A"
    try:
        if np.isnan(value) or np.isinf(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return f"{value * 100:.2f}%"


def _ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. Individual strategy report
# ---------------------------------------------------------------------------


def generate_strategy_report(
    strategy_name: str,
    backtest_result: BacktestResult,
    walk_forward_result: Optional[WalkForwardResult],
    monte_carlo_result: Optional[MonteCarloResult],
    output_dir: str | Path,
    *,
    strategy_description: str = "",
    math_foundation: str = "",
) -> Path:
    """Generate a detailed markdown report for a single strategy.

    Parameters
    ----------
    strategy_name : str
        Human-readable strategy name.
    backtest_result : BacktestResult
        Full-sample backtest output.
    walk_forward_result : WalkForwardResult or None
        Walk-forward validation output.  Section is skipped when *None*.
    monte_carlo_result : MonteCarloResult or None
        Monte Carlo bootstrap output.  Section is skipped when *None*.
    output_dir : str or Path
        Directory where the report markdown file is written.
    strategy_description : str, optional
        One-paragraph strategy description.
    math_foundation : str, optional
        Mathematical formulation / derivation notes.

    Returns
    -------
    Path
        Absolute path to the generated markdown file.
    """
    out = _ensure_dir(output_dir)
    safe_name = strategy_name.lower().replace(" ", "_").replace("/", "_")
    report_path = out / f"{safe_name}_report.md"

    lines: list[str] = []

    def _h1(text: str) -> None:
        lines.append(f"# {text}\n")

    def _h2(text: str) -> None:
        lines.append(f"## {text}\n")

    def _h3(text: str) -> None:
        lines.append(f"### {text}\n")

    def _p(text: str) -> None:
        lines.append(f"{text}\n")

    def _table(headers: list[str], rows: list[list[str]]) -> None:
        lines.append(tabulate(rows, headers=headers, tablefmt="pipe"))
        lines.append("")

    # -- Header --
    _h1(f"Strategy Report: {strategy_name}")
    _p(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    _p("---")

    # -- A. Strategy Overview --
    _h2("Strategy Overview")
    _p(f"**Name:** {strategy_name}")
    if strategy_description:
        _p(f"**Description:** {strategy_description}")
    if math_foundation:
        _h3("Mathematical Foundation")
        _p(math_foundation)

    # -- B. Performance Summary --
    m = backtest_result.metrics
    _h2("Performance Summary")
    perf_rows = [
        ["Total PnL", f"{_fmt(m.get('total_pnl_pct'))}%"],
        ["Annualised Return", _pct(m.get("annualized_return"))],
        ["Sharpe Ratio", _fmt(m.get("sharpe_ratio"))],
        ["Sortino Ratio", _fmt(m.get("sortino_ratio"))],
        ["Calmar Ratio", _fmt(m.get("calmar_ratio"))],
        ["Max Drawdown", _pct(m.get("max_drawdown"))],
        ["Win Rate", _pct(m.get("win_rate"))],
        ["Profit Factor", _fmt(m.get("profit_factor"))],
        ["Information Ratio", _fmt(m.get("information_ratio"))],
        ["t-test p-value", _fmt(m.get("ttest_pvalue"), ".4f")],
        ["Bootstrap p-value", _fmt(m.get("bootstrap_pvalue"), ".4f")],
        ["Number of Bars", str(m.get("n_bars", "N/A"))],
        ["Transaction Costs Paid", _fmt(backtest_result.costs_paid, ",.2f")],
    ]
    _table(["Metric", "Value"], perf_rows)

    # -- C. Walk-Forward Analysis --
    if walk_forward_result is not None:
        _h2("Walk-Forward Analysis")
        wf_headers = [
            "Fold",
            "PnL%",
            "Sharpe",
            "Sortino",
            "MaxDD%",
            "Win Rate",
            "Profit Factor",
        ]
        wf_rows = []
        for i, fm in enumerate(walk_forward_result.fold_metrics):
            wf_rows.append([
                str(i + 1),
                _fmt(fm.get("total_pnl_pct")),
                _fmt(fm.get("sharpe_ratio")),
                _fmt(fm.get("sortino_ratio")),
                _fmt(fm.get("max_drawdown", 0) * 100),
                _pct(fm.get("win_rate")),
                _fmt(fm.get("profit_factor")),
            ])
        _table(wf_headers, wf_rows)

        # Aggregate OOS metrics
        agg = walk_forward_result.aggregate_metrics
        _h3("Aggregate Out-of-Sample Metrics")
        agg_rows = [
            ["OOS Total PnL", f"{_fmt(agg.get('total_pnl_pct'))}%"],
            ["OOS Sharpe", _fmt(agg.get("sharpe_ratio"))],
            ["OOS Sortino", _fmt(agg.get("sortino_ratio"))],
            ["OOS Max Drawdown", _pct(agg.get("max_drawdown"))],
        ]
        _table(["Metric", "Value"], agg_rows)

    # -- D. Monte Carlo Confidence --
    if monte_carlo_result is not None:
        _h2("Monte Carlo Confidence Analysis")
        _p(
            f"Simulations: **{len(monte_carlo_result.terminal_wealth):,}** | "
            f"Mean terminal wealth: **{monte_carlo_result.mean_terminal:,.2f}** | "
            f"Median terminal wealth: **{monte_carlo_result.median_terminal:,.2f}**"
        )

        ci_headers = ["Confidence Level", "Lower Bound", "Upper Bound"]
        ci_rows = []
        for level in sorted(monte_carlo_result.confidence_intervals.keys()):
            lo, hi = monte_carlo_result.confidence_intervals[level]
            ci_rows.append([
                f"{level * 100:.0f}%",
                f"{lo:,.2f}",
                f"{hi:,.2f}",
            ])
        _table(ci_headers, ci_rows)

        _p(
            f"**Probability of achieving >{_PNL_TARGET_PCT:.0f}% PnL:** "
            f"{monte_carlo_result.prob_above_target * 100:.2f}%"
        )
        meets = monte_carlo_result.prob_above_target >= _CONFIDENCE_THRESHOLD
        _p(
            f"**Meets >{_PNL_TARGET_PCT:.0f}% PnL with "
            f">{_CONFIDENCE_THRESHOLD * 100:.0f}% confidence:** "
            f"{'YES' if meets else 'NO'}"
        )

    # -- E. Risk Analysis --
    _h2("Risk Analysis")

    _h3("Drawdown Statistics")
    dd_rows = [
        ["Max Drawdown", _pct(m.get("max_drawdown"))],
        ["Avg Drawdown", _pct(m.get("avg_drawdown"))],
        [
            "Max Drawdown Duration (bars)",
            str(m.get("max_drawdown_duration", "N/A")),
        ],
        [
            "Avg Drawdown Duration (bars)",
            _fmt(m.get("avg_drawdown_duration")),
        ],
    ]
    _table(["Metric", "Value"], dd_rows)

    _h3("Tail Risk Metrics")
    # Compute tail-risk metrics from the return series
    returns = backtest_result.returns
    if len(returns) > 0:
        var_95 = float(np.percentile(returns, 5))
        var_99 = float(np.percentile(returns, 1))
        tail_mask_95 = returns[returns <= var_95]
        cvar_95 = float(np.mean(tail_mask_95)) if len(tail_mask_95) > 0 else var_95
        tail_mask_99 = returns[returns <= var_99]
        cvar_99 = float(np.mean(tail_mask_99)) if len(tail_mask_99) > 0 else var_99

        # Skewness and kurtosis
        from scipy import stats as sp_stats

        skew = float(sp_stats.skew(returns))
        kurt = float(sp_stats.kurtosis(returns))

        # Worst single-day return
        worst_day = float(np.min(returns))
        best_day = float(np.max(returns))

        tail_rows = [
            ["VaR (95%)", _pct(abs(var_95))],
            ["CVaR / ES (95%)", _pct(abs(cvar_95))],
            ["VaR (99%)", _pct(abs(var_99))],
            ["CVaR / ES (99%)", _pct(abs(cvar_99))],
            ["Skewness", _fmt(skew)],
            ["Excess Kurtosis", _fmt(kurt)],
            ["Worst Single-Day Return", _pct(worst_day)],
            ["Best Single-Day Return", _pct(best_day)],
        ]
        _table(["Metric", "Value"], tail_rows)
    else:
        _p("*No return data available for tail-risk analysis.*")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    return report_path.resolve()


# ---------------------------------------------------------------------------
# 2. Comparison report across strategies
# ---------------------------------------------------------------------------


def generate_comparison_report(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Generate a master comparison report across all strategies.

    Parameters
    ----------
    all_results : dict
        Mapping of ``strategy_name`` to a dict with keys:

        - ``"backtest"`` : ``BacktestResult``
        - ``"walk_forward"`` : ``WalkForwardResult`` (optional)
        - ``"monte_carlo"`` : ``MonteCarloResult`` (optional)

    output_dir : str or Path
        Directory where the report markdown file is written.

    Returns
    -------
    Path
        Absolute path to the generated markdown file.
    """
    out = _ensure_dir(output_dir)
    report_path = out / "strategy_comparison_report.md"

    lines: list[str] = []

    lines.append("# Strategy Comparison Report\n")
    lines.append(
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    )
    lines.append("---\n")

    # -- Ranking table --
    lines.append("## Strategy Rankings\n")

    headers = [
        "Rank",
        "Strategy",
        "PnL%",
        "Sharpe",
        "Sortino",
        "MaxDD%",
        "Calmar",
        "Win Rate",
        "P(>45%)",
        "Meets Threshold",
    ]

    # Collect rows with sort key (Sharpe)
    row_data: list[Tuple[float, list[str]]] = []
    for name, res in all_results.items():
        bt: BacktestResult = res["backtest"]
        mc: Optional[MonteCarloResult] = res.get("monte_carlo")
        m = bt.metrics

        sharpe = m.get("sharpe_ratio", float("-inf"))
        if sharpe is None or (isinstance(sharpe, float) and np.isnan(sharpe)):
            sharpe_sort = float("-inf")
        else:
            sharpe_sort = sharpe

        prob = mc.prob_above_target if mc is not None else None
        meets = (
            "YES"
            if (prob is not None and prob >= _CONFIDENCE_THRESHOLD)
            else "NO"
        )

        row = [
            "",  # rank placeholder
            name,
            _fmt(m.get("total_pnl_pct")),
            _fmt(m.get("sharpe_ratio")),
            _fmt(m.get("sortino_ratio")),
            _fmt(m.get("max_drawdown", 0) * 100),
            _fmt(m.get("calmar_ratio")),
            _pct(m.get("win_rate")),
            f"{prob * 100:.2f}%" if prob is not None else "N/A",
            meets,
        ]
        row_data.append((sharpe_sort, row))

    # Sort descending by Sharpe
    row_data.sort(key=lambda x: x[0], reverse=True)
    rows = []
    for rank, (_, row) in enumerate(row_data, 1):
        row[0] = str(rank)
        rows.append(row)

    lines.append(tabulate(rows, headers=headers, tablefmt="pipe"))
    lines.append("")

    # -- Strategies meeting threshold --
    lines.append("## Strategies Meeting Threshold\n")
    lines.append(
        f"**Criterion:** >{_PNL_TARGET_PCT:.0f}% cumulative PnL with "
        f">{_CONFIDENCE_THRESHOLD * 100:.0f}% Monte Carlo confidence\n"
    )

    qualifying = [r for r in rows if r[-1] == "YES"]
    if qualifying:
        for r in qualifying:
            lines.append(
                f"- **{r[1]}** -- PnL: {r[2]}%, Sharpe: {r[3]}, P(>45%): {r[8]}"
            )
        lines.append("")
    else:
        lines.append(
            "*No strategies meet the target threshold with the required confidence.*\n"
        )

    # -- Walk-forward consistency --
    lines.append("## Walk-Forward Consistency\n")
    wf_headers = ["Strategy", "OOS PnL%", "OOS Sharpe", "OOS MaxDD%", "Folds"]
    wf_rows = []
    for name, res in all_results.items():
        wf: Optional[WalkForwardResult] = res.get("walk_forward")
        if wf is None:
            continue
        agg = wf.aggregate_metrics
        wf_rows.append([
            name,
            _fmt(agg.get("total_pnl_pct")),
            _fmt(agg.get("sharpe_ratio")),
            _fmt(agg.get("max_drawdown", 0) * 100),
            str(len(wf.fold_metrics)),
        ])
    if wf_rows:
        lines.append(tabulate(wf_rows, headers=wf_headers, tablefmt="pipe"))
        lines.append("")
    else:
        lines.append("*No walk-forward results available.*\n")

    # -- Recommendations --
    lines.append("## Recommendations\n")
    if qualifying:
        best = qualifying[0]
        lines.append(
            f"**Top pick:** {best[1]} (Sharpe {best[3]}, PnL {best[2]}%)\n"
        )
        lines.append("Key observations:\n")
        for r in qualifying:
            lines.append(
                f"- {r[1]}: Achieves >{_PNL_TARGET_PCT:.0f}% PnL target "
                f"with {r[8]} probability. Max drawdown: {r[5]}%."
            )
        lines.append("")
    else:
        lines.append(
            "No strategy currently meets the confidence threshold. Consider:\n"
        )
        lines.append("- Extending the backtest period for more data.")
        lines.append("- Relaxing the confidence threshold.")
        lines.append(
            "- Combining strategies via ensemble to improve robustness.\n"
        )

    # -- Disclaimer --
    lines.append("---\n")
    lines.append(
        "*Past performance is not indicative of future results. All metrics are "
        "based on historical backtests with simulated transaction costs.*\n"
    )

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    return report_path.resolve()


# ---------------------------------------------------------------------------
# 3. Console summary table
# ---------------------------------------------------------------------------


def print_summary_table(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Pretty-print a comparison table to stdout using tabulate.

    Parameters
    ----------
    all_results : dict
        Same format as :func:`generate_comparison_report`.
    """
    headers = [
        "Strategy",
        "PnL%",
        "Sharpe",
        "MaxDD%",
        "P(>45%)",
        "Confidence",
    ]

    rows: list[list[str]] = []
    for name, res in all_results.items():
        bt: BacktestResult = res["backtest"]
        mc: Optional[MonteCarloResult] = res.get("monte_carlo")
        m = bt.metrics

        prob = mc.prob_above_target if mc is not None else None
        meets = (
            "PASS"
            if (prob is not None and prob >= _CONFIDENCE_THRESHOLD)
            else "FAIL"
        )

        rows.append([
            name,
            _fmt(m.get("total_pnl_pct")),
            _fmt(m.get("sharpe_ratio")),
            _fmt(m.get("max_drawdown", 0) * 100),
            f"{prob * 100:.2f}%" if prob is not None else "N/A",
            meets,
        ])

    # Sort by Sharpe descending
    def _sharpe_key(row: list[str]) -> float:
        try:
            return float(row[2])
        except (ValueError, TypeError):
            return float("-inf")

    rows.sort(key=_sharpe_key, reverse=True)

    print("\n" + "=" * 72)
    print("  STRATEGY COMPARISON SUMMARY")
    print("=" * 72)
    print(tabulate(rows, headers=headers, tablefmt="pretty"))
    print(
        f"\nThreshold: >{_PNL_TARGET_PCT:.0f}% PnL with "
        f">{_CONFIDENCE_THRESHOLD * 100:.0f}% confidence"
    )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# 4. Equity-curve and risk charts
# ---------------------------------------------------------------------------


def generate_equity_curves(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str | Path,
    *,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 150,
) -> List[Path]:
    """Generate matplotlib charts and save as PNG files.

    Creates four chart files:
    a. Equity curves for all strategies overlaid.
    b. Drawdown chart for each strategy.
    c. Rolling Sharpe ratio (252-day) for each strategy.
    d. Monte Carlo terminal wealth distribution histogram.

    Parameters
    ----------
    all_results : dict
        Same format as :func:`generate_comparison_report`.
    output_dir : str or Path
        Directory where PNG files are saved.
    figsize : tuple
        Figure size in inches (width, height).
    dpi : int
        Resolution for saved images.

    Returns
    -------
    list[Path]
        Paths to all generated PNG files.
    """
    out = _ensure_dir(output_dir)
    saved: list[Path] = []

    names = list(all_results.keys())
    n_strategies = len(names)

    if n_strategies == 0:
        return saved

    # Colour cycle
    cmap = plt.cm.get_cmap("tab10", max(n_strategies, 1))
    colours = [cmap(i) for i in range(n_strategies)]

    # ---- (a) Equity curves overlaid ----
    fig, ax = plt.subplots(figsize=figsize)
    for idx, name in enumerate(names):
        bt: BacktestResult = all_results[name]["backtest"]
        eq = bt.equity_curve
        ax.plot(
            np.arange(len(eq)),
            eq,
            label=name,
            color=colours[idx],
            linewidth=1.2,
        )
    ax.set_title("Equity Curves -- All Strategies", fontsize=14)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "equity_curves.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    saved.append(path.resolve())

    # ---- (b) Drawdown chart for each strategy ----
    n_cols = min(n_strategies, 3)
    n_rows = (n_strategies + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize[0], 4 * n_rows),
        squeeze=False,
    )
    for idx, name in enumerate(names):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        bt = all_results[name]["backtest"]
        returns = bt.returns

        # Compute drawdown series
        equity = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        drawdown_pct = drawdown * 100

        ax.fill_between(
            np.arange(len(drawdown_pct)),
            drawdown_pct,
            0,
            color=colours[idx],
            alpha=0.5,
        )
        ax.set_title(f"Drawdown: {name}", fontsize=10)
        ax.set_xlabel("Bar")
        ax.set_ylabel("Drawdown %")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_strategies, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].set_visible(False)

    fig.suptitle("Drawdown Charts", fontsize=14, y=1.02)
    fig.tight_layout()
    path = out / "drawdown_charts.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(path.resolve())

    # ---- (c) Rolling Sharpe ratio (252-day) ----
    fig, ax = plt.subplots(figsize=figsize)
    window = _TRADING_DAYS_PER_YEAR
    for idx, name in enumerate(names):
        bt = all_results[name]["backtest"]
        returns = bt.returns
        if len(returns) < window:
            continue

        # Compute rolling Sharpe
        rolling_mean = np.convolve(returns, np.ones(window) / window, mode="valid")
        rolling_var = np.array([
            np.var(returns[i : i + window], ddof=1) for i in range(len(returns) - window + 1)
        ])
        rolling_std = np.sqrt(rolling_var)
        with np.errstate(divide="ignore", invalid="ignore"):
            rolling_sharpe = np.where(
                rolling_std > 0,
                (rolling_mean / rolling_std) * np.sqrt(window),
                0.0,
            )

        x_axis = np.arange(window - 1, len(returns))
        ax.plot(
            x_axis,
            rolling_sharpe,
            label=name,
            color=colours[idx],
            linewidth=1.0,
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Rolling Sharpe Ratio ({window}-day)", fontsize=14)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "rolling_sharpe.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    saved.append(path.resolve())

    # ---- (d) Monte Carlo terminal wealth distribution ----
    mc_strategies = [
        (name, all_results[name]["monte_carlo"])
        for name in names
        if all_results[name].get("monte_carlo") is not None
    ]
    if mc_strategies:
        n_mc = len(mc_strategies)
        n_mc_cols = min(n_mc, 3)
        n_mc_rows = (n_mc + n_mc_cols - 1) // n_mc_cols
        fig, axes = plt.subplots(
            n_mc_rows,
            n_mc_cols,
            figsize=(figsize[0], 4 * n_mc_rows),
            squeeze=False,
        )
        for idx, (name, mc) in enumerate(mc_strategies):
            r = idx // n_mc_cols
            c = idx % n_mc_cols
            ax = axes[r][c]

            tw = mc.terminal_wealth
            ax.hist(
                tw,
                bins=80,
                color=colours[names.index(name)],
                alpha=0.7,
                edgecolor="white",
                linewidth=0.3,
            )

            # Vertical lines for mean, median, and CI bounds
            ax.axvline(
                mc.mean_terminal,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"Mean: {mc.mean_terminal:,.0f}",
            )
            ax.axvline(
                mc.median_terminal,
                color="orange",
                linestyle="-.",
                linewidth=1.2,
                label=f"Median: {mc.median_terminal:,.0f}",
            )

            # 95% CI bounds
            if 0.95 in mc.confidence_intervals:
                lo, hi = mc.confidence_intervals[0.95]
                ax.axvline(
                    lo,
                    color="grey",
                    linestyle=":",
                    linewidth=1.0,
                    label=f"95% CI: [{lo:,.0f}, {hi:,.0f}]",
                )
                ax.axvline(hi, color="grey", linestyle=":", linewidth=1.0)

            ax.set_title(f"Terminal Wealth: {name}", fontsize=10)
            ax.set_xlabel("Terminal Wealth")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_mc, n_mc_rows * n_mc_cols):
            r = idx // n_mc_cols
            c = idx % n_mc_cols
            axes[r][c].set_visible(False)

        fig.suptitle(
            "Monte Carlo Terminal Wealth Distributions", fontsize=14, y=1.02
        )
        fig.tight_layout()
        path = out / "monte_carlo_distributions.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path.resolve())

    return saved
