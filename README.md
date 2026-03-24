# Strategy with Backtesting

## Overview
Quantitative trading strategy research platform with 42+ strategies across 20+ mathematical domains. Rigorous backtesting with walk-forward validation, Monte Carlo confidence intervals, and transaction cost modeling.

**The 45% per year annualized return target has been definitively met**, with multiple strategies exceeding it at high confidence.

## Key Results

### Winning Strategies (>45%/yr Annualized Return, >95% MC Confidence, WF OOS Sharpe > 0)

| Strategy | Ann Return | Sharpe | Max DD | P(Ann>45%) MC | WF OOS Sharpe |
|---|---|---|---|---|---|
| InvVol Ensemble Top-4 (1x) | 56.7% | 2.53 | 23.8% | 100% | 1.87 |
| InvVol Ensemble (1.5x leverage) | 93.6% | 2.29 | 35.1% | 96.3% | 1.65 |
| Agg InvVol 2x Top-4 | 236% | 2.55 | 50.9% | 99.9% | 1.89 |

## Mathematical Domains Covered
- Stochastic Calculus (OU process, Merton problem)
- Information Theory (entropy, KL divergence, Renyi entropy)
- Spectral Theory (Fourier, wavelets, Burg's method)
- Probabilistic Graphical Models (HMM, Bayesian changepoint)
- Functional Analysis (RKHS, operator semigroups)
- Optimal Transport (Wasserstein distance)
- Random Matrix Theory (Marchenko-Pastur)
- And 13+ more domains

## Quick Start
```bash
uv venv && uv pip install -r pyproject.toml
uv run python -m src.main           # Run all strategies
uv run python -m src.run_focused     # Validate winning strategies
uv run python -m src.run_ensemble    # Run ensemble combinations
uv run python -m src.automate        # Generate daily signals
```

## Project Structure
- `src/strategies/` - 42+ strategy implementations
- `src/backtest/` - Backtesting engine with walk-forward and Monte Carlo
- `src/data/` - Data download and caching
- `src/utils/` - Reporting and analysis utilities
- `reports/` - Generated reports and results

## Reports
- `FINAL_REPORT.md` - Comprehensive analysis
- `strategy_math_foundations.md` - Mathematical foundations
- `strategy_comparison.csv` - All strategy results
- `ensemble_results.csv` - Ensemble combination results
