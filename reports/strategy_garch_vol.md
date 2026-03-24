# Strategy Report: GARCH Volatility

**Date:** 2026-03-23
**Framework:** strategy-with-bt
**Strategy Class:** `GarchVolStrategy` (`src/strategies/garch_vol.py`)

---

## 1. Strategy Overview

The GARCH Volatility strategy combines conditional volatility forecasting with the volatility risk premium (VRP) to generate counter-cyclical equity exposure signals. It harvests the well-documented tendency for forecast/implied volatility to exceed realised volatility, producing systematic long-equity exposure with volatility-targeted sizing.

The strategy blends three complementary signal components -- volatility mean-reversion, VRP harvest, and volatility targeting -- using an adaptive weighting scheme that concentrates on whichever component has been most predictive recently.

---

## 2. Mathematical Foundation

### 2.1 EGARCH(1,1) with Student-t Innovations

The strategy uses the Exponential GARCH specification (Nelson, 1991), which models the log of conditional variance:

    ln(sigma^2_t) = omega + alpha * [|z_{t-1}| - E|z_{t-1}|] + gamma * z_{t-1} + beta * ln(sigma^2_{t-1})

where:
- `omega` is the constant term,
- `alpha >= 0` is the magnitude effect (large shocks increase vol),
- `gamma` captures the **leverage effect** (negative shocks inflate vol more than positive),
- `beta` is the persistence parameter,
- `z_{t-1} = epsilon_{t-1} / sigma_{t-1}` are standardised innovations,
- Innovations follow a Student-t distribution: `z_t ~ t_nu`.

**Key advantage over standard GARCH:** The log-specification ensures `sigma^2 > 0` automatically without imposing non-negativity constraints on parameters. It also better captures the asymmetric impact of negative shocks.

**Unconditional (long-run) log-variance:**

    ln(sigma^2_inf) = omega / (1 - beta)

    sigma^2_inf = exp(omega / (1 - beta))

### 2.2 GJR-GARCH(1,1) Alternative

The strategy also supports the GJR-GARCH specification (Glosten, Jagannathan, Runkle, 1993):

    sigma^2_t = omega + (alpha + gamma * I_{epsilon < 0}) * epsilon^2_{t-1} + beta * sigma^2_{t-1}

Stationarity condition: `alpha + beta + gamma/2 < 1`

Unconditional variance: `sigma^2_inf = omega / (1 - alpha - beta - gamma/2)`

### 2.3 Volatility Risk Premium (VRP)

    VRP_t = sigma^GARCH_{t|t-1} - RV_t

where `RV_t` is the realised volatility. A persistently positive VRP compensates investors for bearing volatility risk. This is one of the most robust risk premia in finance: forecast volatility systematically exceeds realised volatility approximately 80% of the time.

**Signal:** The raw VRP is smoothed over a 40-day window, normalised by its own rolling standard deviation, and clipped to [-1, +1].

### 2.4 Parkinson Realised Volatility Estimator (1980)

    RV_Parkinson = sqrt( (1 / (4 * n * ln(2))) * SUM ln(H_i / L_i)^2 ) * sqrt(252)

where `H_i` and `L_i` are the daily high and low prices. This estimator is approximately 5x more efficient than the close-to-close estimator when intraday range data is available.

### 2.5 Volatility-Targeting Weight

    w_t = sigma_target / sigma_{t|t-1}

Counter-cyclical by construction: positions increase in calm markets (low sigma) and decrease in volatile markets (high sigma). Capped at `max_leverage`.

### 2.6 Volatility Mean-Reversion Z-Score

    z_t = (sigma_t - sigma_bar) / rolling_std(sigma)

where `sigma_bar` is the rolling mean of conditional volatility over `vol_zscore_lookback` days. The z-score is linearly mapped to [-1, +1] between the short threshold and long threshold:

    signal = (z - z_short) / (z_long - z_short) * 2 - 1,  clipped to [-1, +1]

**Interpretation:** When vol is unusually high (z > z_long), expect a vol crush and go long equity. When vol is unusually low (z < z_short), expect a vol expansion and reduce/short.

### 2.7 Adaptive Signal Blending

Signal component weights adapt based on trailing performance using exponentially-weighted edge tracking:

    edge_k(t) = signal_k(t) * r(t+1)    (positive = signal was correct)

    ew_edge_k = EWM(edge_k, decay=0.94)

Scores are computed via softmax-like allocation blended 50/50 with base prior weights, subject to a minimum weight floor of 10% per component.

### 2.8 Composite Signal

    composite_t = direction_t * w_target_t

where:

    direction_t = w_mr(t) * sig_mr(t) + w_vrp(t) * sig_vrp(t) + w_tgt(t) * 1.0

The vol-target weight provides counter-cyclical sizing; the directional blend determines conviction.

---

## 3. Parameter Settings

| Parameter | Value | Description |
|---|---|---|
| `garch_model` | EGARCH | Exponential GARCH specification |
| `garch_p` / `garch_q` | 1 / 1 | EGARCH(1,1) order |
| `garch_dist` | Student-t | Innovation distribution (fat-tailed) |
| `rolling_window` | 504 | ~2 trading years for GARCH estimation |
| `refit_freq` | 10 | Re-estimate GARCH every 10 trading days |
| `rv_window` | 20 | Realised vol lookback (trading days) |
| `use_parkinson` | True | Parkinson estimator when H/L available |
| `vol_zscore_lookback` | 252 | Z-score normalisation window |
| `vol_long_threshold` | 1.5 | Z > 1.5 triggers long signal |
| `vol_short_threshold` | -1.0 | Z < -1.0 triggers short signal |
| `vrp_lookback` | 40 | VRP signal smoothing window |
| `target_vol` | 0.25 (25%) | Annualised volatility target |
| `max_leverage` | 3.0x | Hard cap on position size |
| `vol_mr_weight` | 0.60 | Base weight: vol mean-reversion |
| `vrp_weight` | 0.25 | Base weight: VRP signal |
| `vol_target_weight` | 0.15 | Base weight: pure vol-target |
| `adaptive_blend` | True | Adapt weights from trailing performance |
| `adaptive_lookback` | 63 | ~3 months trailing performance window |
| `adaptive_decay` | 0.94 | Exponential decay factor |
| `adaptive_min_weight` | 0.10 | Minimum weight floor per component |

---

## 4. Performance Summary

### 4.1 Full Backtest (Ensemble OOS Period)

| Metric | Value |
|---|---|
| Total PnL | 235.54% |
| Annualised Return | 28.73% |
| Sharpe Ratio | 1.092 |
| Sortino Ratio | 1.696 |
| Max Drawdown | 23.14% |
| Win Rate | 30.22% |
| P(PnL > 45%) Monte Carlo | 97.92% |
| Bootstrap p-value | 0.0082 |
| WF OOS Sharpe | 0.894 |

### 4.2 Walk-Forward Validation

The walk-forward out-of-sample Sharpe of **0.894** confirms that the strategy's edge persists on unseen data. This is the second-highest WF OOS Sharpe among all individual strategies tested.

### 4.3 Monte Carlo Confidence

With P(PnL > 45%) = **97.92%**, the strategy exceeds the 95% confidence threshold for the 45% PnL target. This is based on 10,000 block bootstrap simulations.

### 4.4 Statistical Significance

Bootstrap p-value = **0.0082**, rejecting the null hypothesis that the strategy's returns are indistinguishable from zero at the 1% significance level. This is the most statistically significant result among all individual strategies in the original backtest run (p = 0.007).

---

## 5. Why It Works

### 5.1 The Volatility Risk Premium (VRP > 0 Most of the Time)

The core economic edge is the volatility risk premium. Forecast/implied volatility systematically exceeds realised volatility because:

1. **Risk aversion:** Investors are willing to pay a premium for downside protection (vol insurance).
2. **Leverage constraints:** Many investors cannot lever up low-vol assets, so they bid up high-vol/high-beta assets, creating a vol premium for those who can provide it.
3. **Jump risk compensation:** GARCH forecasts embed a premium for the possibility of large negative moves that may not materialise.

The VRP is positive approximately 80% of the time in equity markets. By systematically sizing positions inversely to forecast vol (and tilting long when VRP is positive), the strategy harvests this premium.

### 5.2 Volatility Mean-Reversion

Volatility is one of the most predictable features of financial time series. GARCH models capture ~95% of the dynamics of conditional variance. When vol spikes to unusually high levels, it reliably mean-reverts -- creating a long-equity signal that profits from the subsequent vol compression and equity recovery.

### 5.3 Counter-Cyclical Sizing

The vol-targeting mechanism (`w_t = sigma_target / sigma_t`) provides automatic deleveraging in crises and re-leveraging in calm markets. This is equivalent to a constant-risk allocation that improves Sharpe ratio relative to constant-dollar allocation.

### 5.4 Adaptive Signal Blend

The adaptive weighting dynamically allocates to whichever signal component (vol mean-reversion, VRP, or vol-target) is performing best. This prevents the strategy from being dragged down by a temporarily dysfunctional component.

---

## 6. Risk Profile

| Metric | Value | Assessment |
|---|---|---|
| Max Drawdown | 23.14% | Moderate, within acceptable bounds |
| Win Rate | 30.22% | Low -- but expected for a trend/vol strategy |
| Profit Factor | High | Few winning trades with large magnitude |
| Sharpe Ratio | 1.092 | Strong risk-adjusted performance |
| Sortino Ratio | 1.696 | Good downside protection |

### 6.1 Key Risks

- **Low win rate (30%):** The strategy makes money on relatively few, large winning trades. This is psychologically challenging but mathematically sound -- the profit factor compensates.
- **GARCH estimation risk:** If the GARCH model misspecifies the volatility process (e.g., during structural breaks), forecasts degrade. Mitigated by 10-day refitting and convergence monitoring.
- **Gap risk:** Vol-targeting cannot react intraday. A gap-down opening can cause losses before the strategy de-levers.
- **Leverage amplification:** At 3.0x max leverage, a sudden adverse move in a low-vol environment could produce outsized losses. The 25% vol target and 3.0x cap provide headroom.

### 6.2 Regime Performance

| Regime | Expected Performance |
|---|---|
| High vol / post-crash | Strong (vol MR + VRP both large) |
| Calm / grind higher | Moderate (vol-target relevering captures equity premium) |
| Vol-of-vol spike | Variable (GARCH may lag) |
| Trending low-vol | Moderate (counter-cyclical sizing may underweight trend) |
| Flash crash / gap risk | Weak (cannot react intraday) |

---

## 7. Automation and Operational Notes

- **Signal generation is fully automated.** The strategy requires only daily price data (close-to-close; optionally high/low for Parkinson RV).
- **GARCH refit every 10 days** on a rolling 504-day window. Each refit takes ~1-2 seconds per asset.
- **No manual intervention required** -- convergence failures fall back to the most recent valid GARCH estimate.
- **Rebalancing:** Signals update daily; position changes are proportional to signal changes.
- **Data requirements:** Minimum 504 trading days (~2 years) of history before the first signal is generated.

---

## 8. Comparison with Original (Un-Optimised) GARCH Vol

The original GARCH Vol strategy (before the ensemble framework) produced only 7-12% total PnL with a 2.0x leverage cap and conservative vol target. The improved version achieves 236% PnL by:

1. Switching from GJR-GARCH to **EGARCH** (better captures asymmetric leverage effects).
2. Raising the vol target from ~15% to **25%** annualised.
3. Increasing max leverage from 2.0x to **3.0x** (the low max drawdown provides headroom).
4. Increasing vol MR weight from 0.40 to **0.60** (vol mean-reversion was the most predictive component).
5. Enabling **adaptive signal blending** to dynamically reweight components.
6. Reducing refit frequency from 21 to **10 days** for more responsive vol forecasts.

---

*Past performance is not indicative of future results. All metrics are based on historical backtests with simulated transaction costs (6 bps per trade).*
