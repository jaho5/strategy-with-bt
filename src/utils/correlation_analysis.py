"""Correlation analysis module for strategy return streams.

Analyses pairwise correlations between strategy returns to identify the
best ensemble combinations, clusters strategies by similarity, evaluates
regime-conditional performance, and generates a comprehensive markdown
report.

Key capabilities:
- Pearson, Spearman, and tail-dependence correlation matrices
- Greedy forward selection of an optimal ensemble maximising Sharpe
- Hierarchical clustering of strategies by correlation distance
- Regime-conditional (bull/bear/high-vol/low-vol) performance analysis
- Automated markdown report generation
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from src.utils.reporting import _ensure_dir, _fmt, _pct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TRADING_DAYS_PER_YEAR = 252
_TAIL_QUANTILE = 0.10  # Bottom/top 10% for tail correlation


# ---------------------------------------------------------------------------
# 1. Pairwise correlation matrices
# ---------------------------------------------------------------------------


def compute_strategy_correlations(
    strategy_returns_dict: Dict[str, pd.Series],
) -> Dict[str, pd.DataFrame]:
    """Compute pairwise Pearson, Spearman, and tail correlations.

    Parameters
    ----------
    strategy_returns_dict : dict[str, pd.Series]
        Mapping of strategy name to its daily return series.  Series are
        aligned by index (union with NaN-fill then forward-fill).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``"pearson"``, ``"spearman"``, ``"tail_lower"``,
        ``"tail_upper"``.  Each value is a square correlation matrix
        with strategy names as both row and column labels.
    """
    if len(strategy_returns_dict) < 2:
        raise ValueError(
            "Need at least 2 strategies to compute correlations, "
            f"got {len(strategy_returns_dict)}."
        )

    returns_df = _align_returns(strategy_returns_dict)
    names = list(returns_df.columns)
    n = len(names)

    # -- Pearson --
    pearson = returns_df.corr(method="pearson")

    # -- Spearman --
    spearman = returns_df.corr(method="spearman")

    # -- Tail correlations (lower and upper) --
    tail_lower = pd.DataFrame(
        np.eye(n), index=names, columns=names, dtype=np.float64
    )
    tail_upper = pd.DataFrame(
        np.eye(n), index=names, columns=names, dtype=np.float64
    )

    for i in range(n):
        for j in range(i + 1, n):
            r_i = returns_df.iloc[:, i].values
            r_j = returns_df.iloc[:, j].values

            tl = _tail_correlation(r_i, r_j, quantile=_TAIL_QUANTILE, tail="lower")
            tu = _tail_correlation(r_i, r_j, quantile=_TAIL_QUANTILE, tail="upper")

            tail_lower.iloc[i, j] = tl
            tail_lower.iloc[j, i] = tl
            tail_upper.iloc[i, j] = tu
            tail_upper.iloc[j, i] = tu

    return {
        "pearson": pearson,
        "spearman": spearman,
        "tail_lower": tail_lower,
        "tail_upper": tail_upper,
    }


# ---------------------------------------------------------------------------
# 2. Optimal ensemble selection
# ---------------------------------------------------------------------------


def find_optimal_ensemble(
    strategy_returns_dict: Dict[str, pd.Series],
    n_select: int = 5,
) -> Dict[str, Any]:
    """Select the best N strategies maximising ensemble Sharpe ratio.

    Uses greedy forward selection: at each step the strategy whose addition
    most improves the portfolio Sharpe ratio (via Markowitz max-Sharpe
    weights) is added.  After selection, optimal weights are computed over
    the final set.

    Parameters
    ----------
    strategy_returns_dict : dict[str, pd.Series]
        Mapping of strategy name to its daily return series.
    n_select : int
        Number of strategies to select (capped at the number available).

    Returns
    -------
    dict
        Keys:

        - ``"selected_strategies"`` : list[str] -- names in selection order.
        - ``"optimal_weights"``     : dict[str, float] -- Markowitz weights.
        - ``"ensemble_sharpe"``     : float -- annualised Sharpe of the
          optimally-weighted ensemble.
        - ``"selection_path"``      : list[dict] -- per-step details
          (strategy added, Sharpe after addition).
    """
    returns_df = _align_returns(strategy_returns_dict)
    all_names = list(returns_df.columns)
    n_select = min(n_select, len(all_names))

    selected: List[str] = []
    remaining = set(all_names)
    selection_path: List[Dict[str, Any]] = []

    for step in range(n_select):
        best_candidate: Optional[str] = None
        best_sharpe = -np.inf

        for candidate in remaining:
            trial = selected + [candidate]
            trial_returns = returns_df[trial]
            weights = _markowitz_max_sharpe(trial_returns)
            portfolio_returns = trial_returns.values @ weights
            sharpe = _annualised_sharpe(portfolio_returns)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.discard(best_candidate)
        selection_path.append({
            "step": step + 1,
            "added": best_candidate,
            "ensemble_sharpe": best_sharpe,
        })
        logger.info(
            "Ensemble step %d: added '%s' (Sharpe=%.4f)",
            step + 1, best_candidate, best_sharpe,
        )

    # Final optimal weights over selected set
    final_returns = returns_df[selected]
    final_weights = _markowitz_max_sharpe(final_returns)
    final_portfolio = final_returns.values @ final_weights
    ensemble_sharpe = _annualised_sharpe(final_portfolio)

    weight_dict = {name: float(w) for name, w in zip(selected, final_weights)}

    return {
        "selected_strategies": selected,
        "optimal_weights": weight_dict,
        "ensemble_sharpe": ensemble_sharpe,
        "selection_path": selection_path,
    }


# ---------------------------------------------------------------------------
# 3. Strategy clustering
# ---------------------------------------------------------------------------


def strategy_clustering(
    strategy_returns_dict: Dict[str, pd.Series],
    max_clusters: int = 6,
) -> Dict[str, Any]:
    """Hierarchical clustering of strategies by correlation distance.

    Uses ``1 - |Pearson correlation|`` as the distance metric and
    Ward's linkage.  The number of clusters is chosen automatically
    (up to ``max_clusters``) by maximising the gap in the merge
    distances, or by the caller via ``max_clusters``.

    Parameters
    ----------
    strategy_returns_dict : dict[str, pd.Series]
        Mapping of strategy name to its daily return series.
    max_clusters : int
        Upper bound on the number of clusters.

    Returns
    -------
    dict
        Keys:

        - ``"labels"``          : dict[str, int] -- strategy -> cluster id.
        - ``"linkage_matrix"``  : np.ndarray -- linkage matrix (for
          ``scipy.cluster.hierarchy.dendrogram``).
        - ``"clusters"``        : dict[int, list[str]] -- cluster id to
          member strategies.
        - ``"cluster_descriptions"`` : dict[int, str] -- heuristic labels
          (e.g. "momentum-like", "mean-reversion-like").
    """
    returns_df = _align_returns(strategy_returns_dict)
    names = list(returns_df.columns)
    n = len(names)

    if n < 2:
        return {
            "labels": {names[0]: 0} if names else {},
            "linkage_matrix": np.array([]),
            "clusters": {0: names},
            "cluster_descriptions": {0: "singleton"},
        }

    # Correlation-based distance matrix
    corr = returns_df.corr(method="pearson").values
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)

    # Ensure symmetry and non-negativity
    dist = (dist + dist.T) / 2.0
    dist = np.clip(dist, 0.0, 2.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")

    # Determine optimal number of clusters via gap heuristic
    n_clusters = _optimal_n_clusters(Z, max_k=min(max_clusters, n))

    cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")

    labels = {name: int(cid) for name, cid in zip(names, cluster_ids)}
    clusters: Dict[int, List[str]] = {}
    for name, cid in labels.items():
        clusters.setdefault(cid, []).append(name)

    # Heuristic cluster descriptions
    descriptions = _describe_clusters(clusters, returns_df)

    return {
        "labels": labels,
        "linkage_matrix": Z,
        "clusters": clusters,
        "cluster_descriptions": descriptions,
    }


# ---------------------------------------------------------------------------
# 4. Regime-conditional performance
# ---------------------------------------------------------------------------


def regime_conditional_performance(
    strategy_returns: Dict[str, pd.Series],
    market_returns: pd.Series,
) -> Dict[str, Any]:
    """Evaluate each strategy's performance across market regimes.

    Regimes are defined as:
    - **Bull**: market return > 0 (rolling 63-day mean)
    - **Bear**: market return <= 0 (rolling 63-day mean)
    - **High-vol**: rolling 21-day volatility above its median
    - **Low-vol**: rolling 21-day volatility at or below its median

    Parameters
    ----------
    strategy_returns : dict[str, pd.Series]
        Mapping of strategy name to daily return series.
    market_returns : pd.Series
        Daily return series of the market benchmark (e.g. SPY).

    Returns
    -------
    dict
        Keys:

        - ``"regime_labels"``       : pd.DataFrame -- per-date regime flags.
        - ``"performance_by_regime"``: dict[str, pd.DataFrame] -- per-strategy
          metrics for each regime.
        - ``"all_weather"``         : list[str] -- strategies with positive
          Sharpe in every regime.
        - ``"regime_specialists"``  : dict[str, list[str]] -- regime name to
          strategies that perform best in that regime.
    """
    returns_df = _align_returns(strategy_returns)
    market = market_returns.reindex(returns_df.index).fillna(0.0)

    # -- Regime identification --
    rolling_mean_63 = market.rolling(63, min_periods=21).mean()
    rolling_vol_21 = market.rolling(21, min_periods=10).std() * np.sqrt(_TRADING_DAYS_PER_YEAR)

    vol_median = rolling_vol_21.median()

    regimes = pd.DataFrame(index=returns_df.index)
    regimes["bull"] = (rolling_mean_63 > 0).astype(int)
    regimes["bear"] = (rolling_mean_63 <= 0).astype(int)
    regimes["high_vol"] = (rolling_vol_21 > vol_median).astype(int)
    regimes["low_vol"] = (rolling_vol_21 <= vol_median).astype(int)

    # Drop rows before enough history exists for regime detection
    valid_mask = rolling_mean_63.notna() & rolling_vol_21.notna()
    regimes = regimes.loc[valid_mask]
    returns_sub = returns_df.loc[valid_mask]

    regime_names = ["bull", "bear", "high_vol", "low_vol"]
    strategy_names = list(returns_df.columns)

    # -- Per-strategy, per-regime metrics --
    performance: Dict[str, pd.DataFrame] = {}

    for sname in strategy_names:
        rows = []
        for rname in regime_names:
            mask = regimes[rname].astype(bool)
            r = returns_sub.loc[mask, sname].values
            metrics = _quick_metrics(r)
            metrics["regime"] = rname
            metrics["n_days"] = int(mask.sum())
            rows.append(metrics)
        performance[sname] = pd.DataFrame(rows).set_index("regime")

    # -- All-weather identification --
    all_weather: List[str] = []
    for sname in strategy_names:
        perf = performance[sname]
        if (perf["sharpe"] > 0).all():
            all_weather.append(sname)

    # -- Regime specialists (best Sharpe in each regime) --
    specialists: Dict[str, List[str]] = {}
    for rname in regime_names:
        sharpe_scores = {
            sname: performance[sname].loc[rname, "sharpe"]
            for sname in strategy_names
        }
        if sharpe_scores:
            best_sharpe = max(sharpe_scores.values())
            threshold = best_sharpe * 0.8  # within 80% of the best
            specialists[rname] = [
                s for s, v in sharpe_scores.items()
                if v >= threshold and v > 0
            ]

    return {
        "regime_labels": regimes,
        "performance_by_regime": performance,
        "all_weather": all_weather,
        "regime_specialists": specialists,
    }


# ---------------------------------------------------------------------------
# 5. Correlation report generation
# ---------------------------------------------------------------------------


def generate_correlation_report(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Generate a markdown report covering correlation, clusters, and ensemble.

    Parameters
    ----------
    all_results : dict
        Mapping of ``strategy_name`` to a dict with at least a
        ``"backtest"`` key containing a ``BacktestResult`` (which has
        a ``.returns`` attribute).
    output_dir : str or Path
        Directory where the report file is written.

    Returns
    -------
    Path
        Absolute path to the generated report file.
    """
    out = _ensure_dir(output_dir)
    report_path = out / "correlation_analysis.md"

    # -- Extract strategy returns --
    strategy_returns: Dict[str, pd.Series] = {}
    for name, res in all_results.items():
        bt = res["backtest"]
        returns = bt.returns
        # Convert to pandas Series; use integer index if no DatetimeIndex
        if isinstance(returns, pd.Series):
            strategy_returns[name] = returns
        else:
            strategy_returns[name] = pd.Series(
                returns, name=name, dtype=np.float64
            )

    if len(strategy_returns) < 2:
        lines = [
            "# Correlation Analysis Report\n",
            "*Not enough strategies (need >= 2) to perform correlation analysis.*\n",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path.resolve()

    lines: List[str] = []

    def _h1(text: str) -> None:
        lines.append(f"# {text}\n")

    def _h2(text: str) -> None:
        lines.append(f"## {text}\n")

    def _h3(text: str) -> None:
        lines.append(f"### {text}\n")

    def _p(text: str) -> None:
        lines.append(f"{text}\n")

    # -- Header --
    _h1("Correlation Analysis Report")
    _p(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    _p(f"*Strategies analysed: {len(strategy_returns)}*")
    _p("---")

    # -- A. Correlation matrices --
    _h2("Pairwise Correlations")

    try:
        corr_matrices = compute_strategy_correlations(strategy_returns)

        _h3("Pearson Correlation")
        _p(_dataframe_to_md_table(corr_matrices["pearson"]))

        _h3("Spearman Rank Correlation")
        _p(_dataframe_to_md_table(corr_matrices["spearman"]))

        _h3("Lower-Tail Correlation (worst 10%)")
        _p(_dataframe_to_md_table(corr_matrices["tail_lower"]))

        _h3("Upper-Tail Correlation (best 10%)")
        _p(_dataframe_to_md_table(corr_matrices["tail_upper"]))

        # Summary statistics
        pearson_vals = corr_matrices["pearson"].values
        mask = np.triu(np.ones_like(pearson_vals, dtype=bool), k=1)
        off_diag = pearson_vals[mask]
        _p(
            f"**Average pairwise Pearson correlation:** {np.mean(off_diag):.4f}  \n"
            f"**Min:** {np.min(off_diag):.4f} | **Max:** {np.max(off_diag):.4f} | "
            f"**Median:** {np.median(off_diag):.4f}"
        )
    except Exception as exc:
        _p(f"*Correlation computation failed: {exc}*")

    # -- B. Strategy clustering --
    _h2("Strategy Clustering")

    try:
        cluster_result = strategy_clustering(strategy_returns)
        clusters = cluster_result["clusters"]
        descriptions = cluster_result["cluster_descriptions"]

        for cid in sorted(clusters.keys()):
            members = clusters[cid]
            desc = descriptions.get(cid, "")
            _h3(f"Cluster {cid}: {desc}")
            for m in members:
                _p(f"- {m}")
    except Exception as exc:
        _p(f"*Clustering failed: {exc}*")

    # -- C. Optimal ensemble --
    _h2("Optimal Ensemble Selection")

    try:
        ensemble = find_optimal_ensemble(strategy_returns, n_select=min(5, len(strategy_returns)))

        _p(f"**Ensemble Sharpe ratio:** {ensemble['ensemble_sharpe']:.4f}")

        _h3("Selected Strategies and Weights")
        header = "| Strategy | Weight |"
        sep = "|---|---|"
        lines.append(header)
        lines.append(sep)
        for sname in ensemble["selected_strategies"]:
            w = ensemble["optimal_weights"][sname]
            lines.append(f"| {sname} | {w:.4f} |")
        lines.append("")

        _h3("Selection Path")
        header = "| Step | Added | Ensemble Sharpe |"
        sep = "|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for sp in ensemble["selection_path"]:
            lines.append(
                f"| {sp['step']} | {sp['added']} | {sp['ensemble_sharpe']:.4f} |"
            )
        lines.append("")
    except Exception as exc:
        _p(f"*Ensemble selection failed: {exc}*")

    # -- D. Regime-conditional performance --
    _h2("Regime-Conditional Performance")

    # Try to identify market returns from the first strategy's returns as a
    # fallback (equal-weight of all strategies).  Ideally the caller would
    # provide market returns, but for the report we approximate.
    try:
        returns_df = _align_returns(strategy_returns)
        market_proxy = returns_df.mean(axis=1)
        market_proxy.name = "market_proxy"

        regime_result = regime_conditional_performance(
            strategy_returns, market_proxy
        )

        for regime in ["bull", "bear", "high_vol", "low_vol"]:
            _h3(f"Regime: {regime.replace('_', ' ').title()}")
            header = "| Strategy | Sharpe | Ann. Return | Max DD | Win Rate | N Days |"
            sep = "|---|---|---|---|---|---|"
            lines.append(header)
            lines.append(sep)
            for sname in sorted(strategy_returns.keys()):
                perf = regime_result["performance_by_regime"][sname]
                if regime in perf.index:
                    row = perf.loc[regime]
                    lines.append(
                        f"| {sname} "
                        f"| {row['sharpe']:.4f} "
                        f"| {_pct(row['ann_return'])} "
                        f"| {_pct(row['max_dd'])} "
                        f"| {_pct(row['win_rate'])} "
                        f"| {row['n_days']} |"
                    )
            lines.append("")

        if regime_result["all_weather"]:
            _h3("All-Weather Strategies")
            _p(
                "Strategies with positive Sharpe ratio in every regime:"
            )
            for s in regime_result["all_weather"]:
                _p(f"- **{s}**")
        else:
            _p("*No strategy achieved positive Sharpe in all regimes.*")

        if regime_result["regime_specialists"]:
            _h3("Regime Specialists")
            for regime, specialists in regime_result["regime_specialists"].items():
                if specialists:
                    _p(
                        f"**{regime.replace('_', ' ').title()}:** "
                        + ", ".join(specialists)
                    )
    except Exception as exc:
        _p(f"*Regime analysis failed: {exc}*")

    # -- Footer --
    lines.append("---\n")
    _p(
        "*This report analyses historical correlations and does not guarantee "
        "future diversification benefits. Strategy correlations can change "
        "significantly during market stress.*"
    )

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Correlation report saved to %s", report_path.resolve())
    return report_path.resolve()


# ===========================================================================
# Internal helpers
# ===========================================================================


def _align_returns(
    strategy_returns_dict: Dict[str, pd.Series],
) -> pd.DataFrame:
    """Align multiple return series into a single DataFrame.

    Performs an outer join on indices, forward-fills, then zero-fills any
    remaining NaNs.
    """
    returns_df = pd.DataFrame(strategy_returns_dict)
    returns_df = returns_df.sort_index().ffill().fillna(0.0)
    return returns_df


def _tail_correlation(
    x: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.10,
    tail: str = "lower",
) -> float:
    """Compute tail correlation between two return series.

    For the lower tail, selects observations where *both* series are below
    their respective ``quantile``-th percentile.  For the upper tail,
    selects observations where both are above the ``(1-quantile)``-th
    percentile.  Returns the Pearson correlation on the selected subset.

    Returns 0.0 if too few observations fall in the tail.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if tail == "lower":
        threshold_x = np.percentile(x, quantile * 100)
        threshold_y = np.percentile(y, quantile * 100)
        mask = (x <= threshold_x) | (y <= threshold_y)
    else:  # upper
        threshold_x = np.percentile(x, (1 - quantile) * 100)
        threshold_y = np.percentile(y, (1 - quantile) * 100)
        mask = (x >= threshold_x) | (y >= threshold_y)

    x_tail = x[mask]
    y_tail = y[mask]

    if len(x_tail) < 5:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = sp_stats.pearsonr(x_tail, y_tail)

    return float(corr) if np.isfinite(corr) else 0.0


def _annualised_sharpe(returns: np.ndarray) -> float:
    """Compute annualised Sharpe ratio from daily returns."""
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float((mu / sigma) * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _markowitz_max_sharpe(returns_df: pd.DataFrame) -> np.ndarray:
    """Solve max-Sharpe Markowitz optimisation with long-only constraints.

    Returns weight vector summing to 1.  Falls back to equal weight on
    failure.
    """
    if isinstance(returns_df, pd.DataFrame):
        mu = returns_df.mean().values
        cov = returns_df.cov().values
    else:
        returns_arr = np.asarray(returns_df, dtype=np.float64)
        mu = np.mean(returns_arr, axis=0)
        cov = np.cov(returns_arr, rowvar=False)

    K = len(mu)
    if K == 0:
        return np.array([])
    if K == 1:
        return np.array([1.0])

    # Regularise covariance (shrinkage toward diagonal)
    trace_cov = np.trace(cov)
    if trace_cov <= 0:
        return np.full(K, 1.0 / K)
    shrinkage = 0.1
    cov = (1 - shrinkage) * cov + shrinkage * (trace_cov / K) * np.eye(K)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = mu @ w
        port_var = w @ cov @ w
        if port_var <= 1e-16:
            return 0.0
        return -port_ret / np.sqrt(port_var)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * K
    w0 = np.full(K, 1.0 / K)

    try:
        res = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if res.success:
            weights = np.maximum(res.x, 0.0)
            s = weights.sum()
            if s > 0:
                weights /= s
            else:
                weights = np.full(K, 1.0 / K)
        else:
            weights = np.full(K, 1.0 / K)
    except Exception:
        logger.warning(
            "Markowitz optimisation failed; falling back to equal weight.",
            exc_info=True,
        )
        weights = np.full(K, 1.0 / K)

    return weights


def _optimal_n_clusters(Z: np.ndarray, max_k: int = 6) -> int:
    """Choose the number of clusters from a linkage matrix via the gap
    heuristic (largest relative jump in merge distances).

    Parameters
    ----------
    Z : np.ndarray
        Linkage matrix from ``scipy.cluster.hierarchy.linkage``.
    max_k : int
        Maximum number of clusters to consider.

    Returns
    -------
    int
        Optimal number of clusters (at least 2).
    """
    if len(Z) == 0:
        return 1

    # Merge distances are in column 2 of the linkage matrix
    merge_dists = Z[:, 2]

    if len(merge_dists) < 2:
        return min(2, max_k)

    # Look at the last (max_k - 1) merges (going from many clusters to few)
    # The largest gap corresponds to the best number of clusters.
    n_merges = len(merge_dists)
    max_check = min(max_k, n_merges)

    gaps = np.diff(merge_dists[-(max_check):])

    if len(gaps) == 0:
        return min(2, max_k)

    # The gap at index i (from the tail) corresponds to going from
    # (max_check - i) clusters to (max_check - i - 1) clusters.
    best_gap_idx = int(np.argmax(gaps))

    # Number of clusters = number of items at that gap from the end
    n_clusters = max_check - best_gap_idx

    return max(2, min(n_clusters, max_k))


def _describe_clusters(
    clusters: Dict[int, List[str]],
    returns_df: pd.DataFrame,
) -> Dict[int, str]:
    """Generate heuristic descriptions for each cluster based on return
    characteristics.

    Uses autocorrelation, skewness, and volatility to label clusters as
    momentum-like, mean-reversion-like, volatility-based, etc.
    """
    descriptions: Dict[int, str] = {}

    for cid, members in clusters.items():
        if not members:
            descriptions[cid] = "empty"
            continue

        # Aggregate characteristics across cluster members
        autocorrs = []
        skews = []
        vols = []

        for m in members:
            if m not in returns_df.columns:
                continue
            r = returns_df[m].dropna().values
            if len(r) < 20:
                continue

            # Lag-1 autocorrelation
            if len(r) > 1:
                ac = float(np.corrcoef(r[:-1], r[1:])[0, 1])
                if np.isfinite(ac):
                    autocorrs.append(ac)

            # Skewness
            sk = float(sp_stats.skew(r))
            if np.isfinite(sk):
                skews.append(sk)

            # Annualised volatility
            vol = float(np.std(r, ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
            if np.isfinite(vol):
                vols.append(vol)

        avg_autocorr = np.mean(autocorrs) if autocorrs else 0.0
        avg_skew = np.mean(skews) if skews else 0.0
        avg_vol = np.mean(vols) if vols else 0.0

        # Classification heuristic
        label_parts = []
        if avg_autocorr > 0.05:
            label_parts.append("momentum-like")
        elif avg_autocorr < -0.05:
            label_parts.append("mean-reversion-like")

        if avg_vol > 0.20:
            label_parts.append("high-volatility")
        elif avg_vol < 0.05:
            label_parts.append("low-volatility")

        if avg_skew > 0.5:
            label_parts.append("positive-skew")
        elif avg_skew < -0.5:
            label_parts.append("negative-skew")

        if not label_parts:
            label_parts.append("mixed")

        descriptions[cid] = ", ".join(label_parts)

    return descriptions


def _quick_metrics(returns: np.ndarray) -> Dict[str, float]:
    """Compute a compact set of performance metrics for a return array."""
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)

    if n < 2:
        return {
            "sharpe": 0.0,
            "ann_return": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
        }

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    sharpe = (mu / sigma * np.sqrt(_TRADING_DAYS_PER_YEAR)) if sigma > 1e-12 else 0.0

    cum = np.prod(1.0 + returns) - 1.0
    years = n / _TRADING_DAYS_PER_YEAR
    ann_return = ((1.0 + cum) ** (1.0 / years) - 1.0) if years > 0 and (1.0 + cum) > 0 else 0.0

    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = 1.0 - equity / running_max
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    win_rate = float(np.mean(returns > 0))

    return {
        "sharpe": float(sharpe),
        "ann_return": float(ann_return),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
    }


def _dataframe_to_md_table(df: pd.DataFrame, precision: int = 3) -> str:
    """Convert a DataFrame to a markdown table string.

    Formats floating-point values to ``precision`` decimal places.
    """
    names = list(df.columns)
    header = "| |" + "|".join(f" {n} " for n in names) + "|"
    sep = "|---|" + "|".join("---" for _ in names) + "|"

    rows = [header, sep]
    for idx in df.index:
        vals = df.loc[idx]
        formatted = [f" {v:.{precision}f} " if isinstance(v, (float, np.floating)) else f" {v} " for v in vals]
        rows.append(f"| {idx} |" + "|".join(formatted) + "|")

    return "\n".join(rows)
