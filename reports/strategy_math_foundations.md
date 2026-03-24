# Mathematical Foundations of Quantitative Trading Strategies

**A Comprehensive Reference for the strategy-with-bt Framework**

---

## Table of Contents

1. [OU Mean Reversion](#1-ou-mean-reversion)
2. [HMM Regime Switching](#2-hmm-regime-switching)
3. [Kalman Filter Alpha](#3-kalman-filter-alpha)
4. [Spectral Momentum](#4-spectral-momentum)
5. [GARCH Volatility](#5-garch-volatility)
6. [Optimal Transport Momentum](#6-optimal-transport-momentum)
7. [Information Geometry](#7-information-geometry)
8. [Stochastic Control](#8-stochastic-control)
9. [RMT Eigenportfolio](#9-rmt-eigenportfolio)
10. [Entropy Regularized](#10-entropy-regularized)
11. [Fractional Differentiation](#11-fractional-differentiation)
12. [Levy Jump Detection](#12-levy-jump-detection)
13. [Topological Data Analysis](#13-topological-data-analysis)
14. [Rough Volatility](#14-rough-volatility)
15. [Bayesian Changepoint](#15-bayesian-changepoint)
16. [Sparse Mean Reversion](#16-sparse-mean-reversion)
17. [Momentum Crash Hedge](#17-momentum-crash-hedge)
18. [Kelly Growth Optimal](#18-kelly-growth-optimal)
19. [Microstructure](#19-microstructure)

---

## 1. OU Mean Reversion

### Mathematical Domain

Stochastic differential equations, cointegration theory, maximum likelihood estimation.

### Key Equations

**Ornstein-Uhlenbeck SDE:**

    dX_t = theta * (mu - X_t) dt + sigma * dW_t

where:
- `theta > 0` is the mean-reversion speed,
- `mu` is the long-run equilibrium level,
- `sigma` is the diffusion coefficient,
- `W_t` is a standard Wiener process.

**Exact discretisation (AR(1) representation):**

    X_{t+1} = a + b * X_t + eta_t

where:
- `b = exp(-theta * dt)`
- `a = mu * (1 - b)`
- `Var(eta) = sigma^2 / (2 * theta) * (1 - b^2)`

**Maximum likelihood estimation:** The log-likelihood for the Gaussian transitions is

    L(theta, mu, sigma) = -(N-1)/2 * ln(2*pi*v) - (1/2v) * SUM_{t=1}^{N-1} (X_{t+1} - a - b*X_t)^2

where `v = Var(eta)`. Parameters are estimated by minimising `-L` via Nelder-Mead.

**Half-life of mean reversion:**

    t_{1/2} = ln(2) / theta

**Stationary variance:**

    Var_stationary = sigma^2 / (2 * theta)

**Normalised z-score:**

    z_t = (S_t - mu) / sigma_eq ,    sigma_eq = sigma / sqrt(2 * theta)

**Engle-Granger cointegration test:** For assets Y and X, regress

    Y_t = alpha + beta * X_t + epsilon_t

and test `epsilon_t` for stationarity via the Augmented Dickey-Fuller (ADF) test. The spread is `S_t = Y_t - beta * X_t - alpha`.

**Half-Kelly position sizing:**

    f* = (1/2) * |mu_r| / sigma_r^2

clipped to `[0.1, 1.0]`.

### Theoretical Edge

The OU process provides a rigorous statistical framework for pairs trading. When two assets are cointegrated, their spread is mean-reverting by definition. The OU model quantifies the speed (`theta`), level (`mu`), and volatility (`sigma`) of this reversion, allowing precise entry/exit calibration. The z-score framework transforms the spread into a standardised signal with known distributional properties under the null, enabling statistically grounded threshold selection.

### Risk Model

- **Entry/exit hysteresis:** Entry at `|z| > 2.0`, exit at `|z| < 0.5`, stop-loss at `|z| > 4.0`.
- **Half-life filter:** Only trade pairs with mean-reversion half-life between 5 and 60 trading days (excludes too-fast and too-slow dynamics).
- **Cointegration screening:** Pairs must pass both the Engle-Granger test (`p < 0.05`) and ADF test on residuals (`p < 0.05`).
- **Kelly sizing:** Half-Kelly fraction prevents over-concentration while maintaining geometric growth optimality.
- **Rolling re-estimation:** Monthly re-calibration (21-day frequency) adapts to structural changes in the cointegration relationship.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Sideways / Range-bound | Strong | Mean reversion is the dominant dynamic |
| Low volatility | Moderate | Spreads may not deviate enough for entry signals |
| Trending / Momentum | Weak | Cointegration relationships may break down |
| Crisis / High vol | Variable | Stop-losses protect capital; fast mean reversion after dislocations can be profitable |
| Structural break | Weak | Hedge ratios become stale; cointegration may vanish |

---

## 2. HMM Regime Switching

### Mathematical Domain

Probabilistic graphical models, expectation-maximisation (EM) algorithm, hidden Markov chains, information-theoretic uncertainty quantification.

### Key Equations

**Hidden Markov Model specification:**

Hidden states: `S_t in {Bull, Bear, Sideways}` (K = 3 states).

Transition matrix:

    A[i,j] = P(S_{t+1} = j | S_t = i)

Emission model (Gaussian):

    r_t | S_t = k  ~  N(mu_k, Sigma_k)

where the observation vector includes log-return, realised volatility, and return skewness.

**Baum-Welch algorithm (EM):**

E-step (forward-backward):

    alpha_t(j) = P(r_1,...,r_t, S_t=j)    [forward variable]
    beta_t(j) = P(r_{t+1},...,r_T | S_t=j) [backward variable]

    gamma_t(j) = P(S_t=j | r_1,...,r_T) = alpha_t(j) * beta_t(j) / P(r_1,...,r_T)

    xi_t(i,j) = P(S_t=i, S_{t+1}=j | r_1,...,r_T)

M-step: Update parameters

    A[i,j] = SUM_t xi_t(i,j) / SUM_t gamma_t(i)
    mu_k = SUM_t gamma_t(k) * r_t / SUM_t gamma_t(k)
    Sigma_k = SUM_t gamma_t(k) * (r_t - mu_k)(r_t - mu_k)' / SUM_t gamma_t(k)

**Causal filtering (forward algorithm only):**

    P(S_t = k | r_1,...,r_t) = alpha_t(k) / SUM_j alpha_t(j)

This implementation uses the forward algorithm directly for causal (non-look-ahead) trading signals, rather than the full smoothed posteriors.

**Viterbi algorithm:** Recovers the most-likely state sequence

    S*_{1:T} = argmax_{S_{1:T}} P(S_{1:T} | r_{1:T})

via dynamic programming.

**Shannon entropy of regime probabilities:**

    H(p) = -SUM_k p_k * ln(p_k)

Maximum entropy for K=3 states is `ln(3) ~ 1.099` nats. When `H > threshold`, regime classification is uncertain and position size is reduced.

**Stationary distribution:** Solves `pi * A = pi` via the left eigenvector of `A'` corresponding to eigenvalue 1.

### Theoretical Edge

Markets exhibit distinct regimes (bull, bear, sideways) with different return distributions. The HMM captures regime persistence via the transition matrix, producing filtered probabilities that adapt gradually rather than switching abruptly. The information-theoretic entropy check prevents the strategy from acting on ambiguous regime classifications, while the forward-algorithm implementation ensures strict causality.

### Risk Model

- **Entropy-based uncertainty reduction:** When `H(p_t) > 1.0`, position scale decays linearly to zero at maximum entropy.
- **Minimum probability filter:** The dominant regime must have `P(S_t = k) >= 0.40` for the strategy to act.
- **Volatility targeting overlay:** Positions are scaled by `sigma_target / sigma_realized` with a 2x leverage cap, ensuring consistent risk exposure across regimes.
- **Covariance regularisation:** A ridge (`1e-3`) is added to covariance matrices with near-zero eigenvalues to prevent numerical instability.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Persistent trends (bull/bear) | Strong | HMM locks onto regime; high P(S_t) amplifies signal |
| Choppy / regime-switching | Moderate | Transition detection delays may cause whipsaws |
| Gradual regime transitions | Strong | Filtered probabilities smoothly shift allocation |
| Sudden crashes | Weak initially | Requires time to update posterior; entropy filter may flatten positions |
| Low-volatility grind | Moderate | Sideways regime triggers mean-reversion sub-strategy |

---

## 3. Kalman Filter Alpha

### Mathematical Domain

State-space models, Bayesian filtering, linear-Gaussian estimation, market-neutral portfolio construction.

### Key Equations

**State-space model:**

State vector: `x_t = [alpha_t, beta_t]'`

State equation (random walk):

    x_t = F * x_{t-1} + w_t ,    w_t ~ N(0, Q)

    F = I_2  (identity matrix)

    Q = diag(q_alpha, q_beta)

Observation equation:

    r_{stock,t} = H_t * x_t + v_t ,    v_t ~ N(0, R)

    H_t = [1,  r_{market,t}]

This gives: `r_{stock,t} = alpha_t + beta_t * r_{market,t} + v_t`.

**Kalman recursion:**

Predict step:

    x_{t|t-1} = F * x_{t-1|t-1}
    P_{t|t-1} = F * P_{t-1|t-1} * F' + Q

Update step:

    Innovation: y_t = r_{stock,t} - H_t * x_{t|t-1}
    Innovation variance: S_t = H_t * P_{t|t-1} * H_t' + R
    Kalman gain: K_t = P_{t|t-1} * H_t' * S_t^{-1}
    State update: x_{t|t} = x_{t|t-1} + K_t * y_t
    Covariance update: P_{t|t} = (I - K_t * H_t) * P_{t|t-1}

**Alpha z-score (trading signal):**

    z_t = alpha_t / sqrt(P_t[0,0])

where `P_t[0,0]` is the posterior variance of `alpha_t`. This normalises the alpha estimate by its uncertainty.

**Trading rule:**

    z_t > +1.5  =>  Long (with weight proportional to |z_t|)
    z_t < -1.5  =>  Short (with weight proportional to |z_t|)
    otherwise   =>  Flat

**Beta-hedging:**

    w_{market} = -SUM_i w_i * beta_i

This neutralises the portfolio's market exposure, isolating pure alpha.

### Theoretical Edge

The Kalman filter provides the optimal (minimum variance) estimate of time-varying alpha and beta under the linear-Gaussian assumption. Unlike static regression, it adapts continuously to structural changes in the stock-market relationship. The z-score normalisation accounts for estimation uncertainty: a large alpha with large uncertainty generates a weaker signal than a smaller alpha known with high precision. Beta-hedging ensures that returns are attributable to stock-specific alpha, not market exposure.

### Risk Model

- **Position cap:** Maximum 10% weight per individual name.
- **Target leverage:** Gross exposure normalised to 1.0x.
- **Beta neutrality:** Portfolio beta is hedged to within `+/- 0.1` of zero.
- **Diffuse prior:** Initial state covariance `P_0 = I` reflects high initial uncertainty; the filter adapts as evidence accumulates.
- **Process noise calibration:** `q_alpha = 1e-5`, `q_beta = 1e-4` reflect the prior belief that beta varies faster than alpha.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Stock-picking / dispersion | Strong | Alpha differentiation across stocks is high |
| Factor rotation | Moderate | Time-varying beta adapts, but with lag |
| Market crash (beta spike) | Moderate | Beta hedge protects; alpha may be noisy |
| Low dispersion | Weak | Alpha signals are small relative to estimation noise |
| Regime change (structural) | Moderate | Random-walk state model adapts, but slowly |

---

## 4. Spectral Momentum

### Mathematical Domain

Fourier analysis, wavelet multi-resolution analysis, Hilbert transform (analytic signal theory), signal processing.

### Key Equations

**Discrete Fourier Transform (DFT):**

    X[k] = SUM_{n=0}^{N-1} x[n] * exp(-j * 2*pi*k*n / N)

Power spectrum: `P[k] = |X[k]|^2`. Dominant frequencies are identified as the top-k components exceeding a power threshold (5% of total spectral power).

**Spectral filtering:** Retain only dominant frequency bins; zero out noise. Reconstruct via inverse DFT:

    x_clean[n] = (1/N) * SUM_{k in dominant} X[k] * exp(j * 2*pi*k*n / N)

The spectral trend signal is the first difference of the clean reconstruction at the window boundary.

**Discrete Wavelet Transform (DWT) -- Daubechies-4:**

Multi-resolution decomposition:

    x = A_L + D_L + D_{L-1} + ... + D_1

where `A_L` is the level-L approximation (long-term trend) and `D_l` are detail coefficients at scale `l` (capturing cycles of period `~2^l` bars).

Wavelet momentum signals:
- **Short-term momentum:** Detail levels 3-4 (approximately 5-20 day cycles)
- **Long-term momentum:** Approximation level 5+ (60+ day cycles)

Each component is z-scored and the two are averaged.

**Hilbert transform and analytic signal:**

    x_a(t) = x(t) + j * H[x](t)

where `H[x]` is the Hilbert transform. This yields:

    Instantaneous amplitude: A(t) = |x_a(t)|
    Instantaneous phase: phi(t) = unwrap(arg(x_a(t)))
    Instantaneous frequency: omega(t) = d(phi)/dt

Phase timing signal:
- `dphase > 0` AND amplitude rising => `+1`
- `dphase < 0` AND amplitude rising => `-1`
- Amplitude falling => `0` (trend weakening)

**Adaptive component weighting via rolling rank IC:**

    IC_t = SpearmanCorrelation(signal_{t-w:t}, forward_returns_{t-w:t})

Component weights are proportional to `mean(|IC|)`, ensuring that the most predictive sub-signal receives the highest allocation.

**Robust normalisation:**

    z = (x - median) / (1.4826 * MAD)

followed by `tanh(z)` to map to `(-1, 1)`. The constant 1.4826 scales MAD to be consistent with the standard deviation under normality.

### Theoretical Edge

Financial time series contain signals at multiple frequency scales that are obscured by noise. The DFT extracts dominant cyclical components, the DWT provides localised multi-resolution analysis (capturing both frequency and time information that the DFT alone cannot), and the Hilbert transform provides instantaneous phase/amplitude for timing. The adaptive IC-weighting automatically allocates to whichever spectral view is most predictive in the current market environment.

### Risk Model

- **Dead zone:** Composite signal must exceed `+/- 0.15` before generating a position, filtering out noise.
- **Robust normalisation:** MAD-based z-scoring is resistant to outliers (unlike mean/std).
- **IC-based weighting:** Component weights adapt based on recent predictive power, reducing allocation to failing sub-signals.
- **Spectral power threshold:** Only frequency components carrying >= 5% of total power are retained; the rest is treated as noise.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Cyclical markets | Strong | Fourier/wavelet decomposition captures periodic structure |
| Strong trends | Strong | Long-term wavelet and spectral trend signals align |
| Random walk / efficient | Weak | No dominant spectral structure to exploit |
| Sudden regime changes | Moderate | Wavelet localisation helps, but rolling DFT has latency |
| High-frequency noise | Moderate | Wavelet denoising filters it, but phase signal may be noisy |

---

## 5. GARCH Volatility

### Mathematical Domain

Conditional heteroscedasticity models, extreme value theory, volatility risk premium.

### Key Equations

**GJR-GARCH(1,1) with Student-t innovations:**

    sigma^2_t = omega + (alpha + gamma * I_{epsilon < 0}) * epsilon^2_{t-1} + beta * sigma^2_{t-1}

where:
- `omega > 0` is the constant term,
- `alpha >= 0` is the ARCH coefficient,
- `beta >= 0` is the GARCH coefficient,
- `gamma >= 0` captures the leverage effect (negative shocks inflate variance more),
- `I_{epsilon < 0}` is the indicator for negative innovations,
- Innovations: `epsilon_t = r_t - mu`, `z_t = epsilon_t / sigma_t ~ t_nu`.

**Stationarity condition:**

    alpha + beta + gamma/2 < 1

**Unconditional (long-run) variance:**

    sigma^2_inf = omega / (1 - alpha - beta - gamma/2)

**Volatility Risk Premium (VRP):**

    VRP_t = sigma^{GARCH}_{t|t-1} - RV_t

where `RV_t` is the realised volatility estimated via close-to-close or Parkinson's range estimator.

**Parkinson realised volatility (1980):**

    RV_Parkinson = sqrt( (1 / (4*n*ln2)) * SUM ln(H_i/L_i)^2 ) * sqrt(252)

This is approximately 5x more efficient than close-to-close estimation.

**Volatility-targeting weight:**

    w_t = sigma_target / sigma_{t|t-1}

Counter-cyclical: positions increase in calm markets, decrease in volatile markets. Capped at `max_leverage = 2.0`.

**Vol mean-reversion z-score:**

    z_t = (sigma_t - sigma_bar) / rolling_std(sigma)

Linearly mapped to `[-1, +1]` between the short threshold (-1.5) and long threshold (2.0).

**Composite signal:**

    signal_t = (0.4 * vol_mr_signal + 0.3 * vrp_signal + 0.3 * vol_target) * w_target

### Theoretical Edge

Volatility is highly persistent and forecastable (GARCH captures ~95% of conditional variance dynamics). The leverage effect (`gamma > 0`) means negative returns amplify future volatility asymmetrically, which the GJR specification explicitly models. The VRP is one of the most robust risk premia in finance: implied/forecast volatility systematically exceeds realised volatility, compensating investors for bearing volatility risk. Vol-targeting provides automatic deleveraging in crises and relevering in calm markets.

### Risk Model

- **Maximum leverage:** Hard cap at 2.0x prevents excessive sizing in low-vol regimes.
- **Rolling re-estimation:** GARCH parameters are re-estimated every 21 days on a 504-day (2-year) window to capture evolving volatility dynamics.
- **Convergence monitoring:** Non-converged GARCH fits are discarded; the strategy falls back to the most recent valid estimate.
- **Signal blending:** No single component (vol MR, VRP, or vol target) dominates; diversification across three signals reduces model risk.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| High vol / post-crash | Strong | Vol mean-reversion and VRP are both large; vol-target reduces exposure during crash |
| Calm / grind higher | Moderate | Vol-target relevering captures equity risk premium |
| Vol-of-vol spike | Variable | GARCH forecasts may lag; vol-target provides some protection |
| Trending low-vol | Moderate | Counter-cyclical sizing may underweight a strong trend |
| Flash crash / gap risk | Weak | Vol-target cannot react intraday |

---

## 6. Optimal Transport Momentum

### Mathematical Domain

Measure theory, optimal transport, Wasserstein geometry, distributional statistics.

### Key Equations

**Wasserstein-1 distance (Earth Mover's Distance):**

For probability measures P, Q on R:

    W_1(P, Q) = inf_{gamma in Gamma(P,Q)} E_{(x,y)~gamma}[|x - y|]

For univariate distributions this reduces to:

    W_1(P, Q) = integral |F_P(x) - F_Q(x)| dx

where `F_P`, `F_Q` are the cumulative distribution functions.

**Signed Wasserstein distance (directional signal):**

    SW_1 = sign(mu_recent - mu_hist) * W_1(P_recent, P_hist)

This captures both the magnitude and direction of the distributional shift.

**Wasserstein-2 distance (distributional quality filter):**

For Gaussians:

    W_2^2 = (mu_1 - mu_2)^2 + (sigma_1 - sigma_2)^2 + 2*sigma_1*sigma_2*(1 - rho)

With independent windows (`rho = 0`):

    W_2^2 = (mu_1 - mu_2)^2 + (sigma_1 + sigma_2)^2 - 2*sigma_1*sigma_2

**Quality filter:** The ratio

    quality = (mu_1 - mu_2)^2 / W_2^2

measures what fraction of the distributional shift is attributable to the mean (as opposed to variance). Stocks where `quality < 0.5` are excluded (their shift is variance-driven, not mean-driven).

**Cross-sectional ranking:**

Assets are ranked by `SW_1`. The top quintile (20%) forms the long leg; the bottom quintile forms the short leg. Equal-weight within each leg, with gross exposure normalised to 1 (0.5 long + 0.5 short).

**Turnover constraint:**

    turnover = (1/2) * SUM |w_new_i - w_old_i|

If `turnover > limit` (30%), the change is scaled: `w_constrained = w_old + scale * (w_new - w_old)` where `scale = limit / turnover`.

### Theoretical Edge

Traditional momentum uses only the first moment (mean return) of the return distribution. The Wasserstein distance captures shifts in the *entire* distribution -- including changes in variance, skewness, and kurtosis -- while remaining a true metric on the space of probability measures. This makes it more robust than mean-return momentum: a stock whose mean return is flat but whose distribution has shifted (e.g., from symmetric to right-skewed) will be detected by `W_1` but not by simple momentum. The quality filter ensures the strategy only acts on distributional shifts driven by mean displacement, avoiding false signals from pure volatility changes.

### Risk Model

- **Weekly rebalancing** (every 5 days) balances signal responsiveness against transaction costs.
- **30% turnover limit** prevents excessive trading from noisy distributional estimates.
- **Quality filter** (W_2 decomposition) eliminates stocks whose shift is variance-dominated.
- **Long-short construction** with equal gross exposure provides inherent market neutrality.
- **Minimum stock count:** Requires >= 5 valid stocks to generate signals; otherwise carries forward previous weights.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Cross-sectional dispersion | Strong | Distributional shifts vary across stocks, creating clear ranking |
| Persistent momentum | Strong | W_1 captures sustained distributional shifts |
| Momentum reversals | Moderate | Quality filter helps, but lagged distributional estimates may persist |
| Low dispersion / correlated | Weak | All stocks shift similarly; ranking signal is noise |
| Fat-tailed environments | Strong | W_1 naturally captures tail behaviour that mean-momentum misses |

---

## 7. Information Geometry

### Mathematical Domain

Differential geometry on statistical manifolds, Fisher information, Kullback-Leibler divergence, natural gradient methods.

### Key Equations

**Gaussian statistical manifold:** The parameter space `(mu, sigma)` of `N(mu, sigma^2)` is a 2-dimensional Riemannian manifold with the Fisher information matrix as its metric tensor:

    I(mu, sigma) = diag(1/sigma^2, 2/sigma^2)

**Fisher-Rao geodesic distance (Atkinson & Mitchell 1981):**

    d_FR(theta_1, theta_2) = sqrt(2) * arccosh(1 + delta)

where

    delta = [(mu_1 - mu_2)^2 + 2*(sigma_1 - sigma_2)^2] / (2 * sigma_1 * sigma_2)

This is the intrinsic distance on the Gaussian manifold -- the length of the shortest path between two Gaussian distributions through the space of all Gaussians.

**KL divergence between Gaussians:**

    D_KL(N(mu_1, sigma_1^2) || N(mu_2, sigma_2^2)) = ln(sigma_2/sigma_1) + (sigma_1^2 + (mu_1 - mu_2)^2) / (2*sigma_2^2) - 1/2

Note: KL divergence is *asymmetric* -- `D_KL(P||Q) != D_KL(Q||P)`. The strategy uses both directions and their asymmetry as a signal.

**Fisher information determinant (information quality):**

    det(I(mu, sigma)) = 2 / sigma^4

Higher Fisher information (lower sigma) means the parameters are estimated with greater precision.

**Raw directional signal:**

    signal_t = tanh(mean_shift / (sigma_t + eps)) * tanh(kl_magnitude)

where `mean_shift = mu_t - mu_baseline` and `kl_magnitude = (D_KL(recent||base) + D_KL(base||recent)) / 2`.

**Regime-change damping:**

    scale = exp(-(d_FR - threshold))    when d_FR > threshold
    scale = 1.0                         when d_FR <= threshold

**Information quality sizing:**

    info_quality = sqrt(det(I(theta_t)) / det(I(theta_base)))

Capped at 2.0. Higher current information quality (lower current vol relative to baseline) leads to larger positions.

**Natural gradient position update:**

    delta_pos = eta * (sigma_t / sigma_base)^2 * (target - position)

The natural gradient pre-multiplies the Euclidean gradient by `I^{-1} ~ sigma^2`, giving scale-invariant updates. In high-volatility environments, the natural gradient takes larger steps (reflecting that the Riemannian curvature is lower and movements on the manifold are cheaper).

### Theoretical Edge

Standard signal processing treats all parameter changes equally. Information geometry recognises that changes in a distribution's parameters have different significance depending on where you are on the statistical manifold. A shift of 1% in the mean matters more when volatility is 5% than when it is 30%. The Fisher-Rao distance provides a principled measure of distributional change that respects this geometry. The natural gradient update ensures position changes are scale-invariant, avoiding the pathology of over-reacting in low-vol environments and under-reacting in high-vol ones.

### Risk Model

- **Regime-change damping:** Exponential position decay beyond `d_FR > 1.0` prevents the strategy from maintaining positions during structural regime changes.
- **Information quality scaling:** Positions are reduced when parameter estimates are unreliable (low Fisher information / high vol).
- **Natural gradient smoothing:** Position changes are governed by the Riemannian metric, preventing abrupt allocation shifts.
- **tanh bounding:** All raw signals are bounded to `(-1, 1)` via the hyperbolic tangent.
- **Max position clamp:** Hard cap at `+/- 1.0`.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Gradual distribution shifts | Strong | KL divergence and Fisher-Rao distance track shifts precisely |
| Stable regime (low d_FR) | Strong | High info quality, full position sizing |
| Regime transition | Moderate | Damping reduces exposure; natural gradient smooths adjustment |
| Vol explosion | Weak | Fisher info drops (sigma rises), info quality scaling reduces positions |
| Mean-reverting vol | Strong | Regime-change detector triggers re-engagement as d_FR normalises |

---

## 8. Stochastic Control

### Mathematical Domain

Hamilton-Jacobi-Bellman (HJB) equation, continuous-time portfolio optimisation, Bayesian shrinkage, Ledoit-Wolf covariance estimation.

### Key Equations

**Merton's consumption-investment problem:**

    max E[ integral_0^T U(c_t) dt + B(W_T) ]

with CRRA utility: `U(x) = x^{1-gamma} / (1-gamma)`.

**Hamilton-Jacobi-Bellman equation:**

    0 = max_w { V_t + (r + w'(mu - r)) * W * V_W + (1/2) * w' Sigma w * W^2 * V_WW }

**Optimal Merton fractions (closed-form HJB solution):**

    w* = (1/gamma) * Sigma^{-1} * (mu - r)

With time-varying parameters:

    w*_t = (1/gamma) * Sigma_t^{-1} * (mu_t - r_t)

**Ledoit-Wolf covariance shrinkage:**

    Sigma_shrunk = alpha * F + (1 - alpha) * S

where `S` is the sample covariance, `F` is the structured target (constant-correlation model), and `alpha` is the optimal shrinkage intensity minimising the Frobenius-norm loss.

**James-Stein-style mean shrinkage:**

    mu_shrunk = (1 - delta) * mu_sample + delta * mu_grand_mean

where `delta in [0, 1]` is the shrinkage intensity (default 0.5).

**Black-Litterman posterior (no investor views):**

Prior equilibrium returns from reverse optimisation:

    pi = gamma * Sigma * w_mkt

Posterior mean:

    mu_BL = [(tau * Sigma)^{-1} + Sigma^{-1}]^{-1} * [(tau * Sigma)^{-1} * pi + Sigma^{-1} * mu_sample]

where `tau = 0.05` controls confidence in the equilibrium prior.

**Leverage constraint:**

    ||w||_1 <= max_leverage

If violated, all weights are scaled: `w = w * (max_leverage / ||w||_1)`.

**Transaction cost filter:**

    Only rebalance asset i if |w_target_i - w_current_i| > 2 * tc

where `tc = 10 bps`. This no-trade zone avoids paying round-trip costs for marginal improvements.

### Theoretical Edge

The Merton solution is the theoretically optimal allocation for an investor with constant relative risk aversion in a continuous-time setting. The Black-Litterman framework addresses the well-known instability of mean-variance optimisation by shrinking return estimates toward an equilibrium anchor (implied by market-cap weights). Ledoit-Wolf covariance shrinkage ensures the precision matrix is well-conditioned even when the number of assets is comparable to the number of observations. Together, these produce robust, theoretically grounded portfolio weights.

### Risk Model

- **CRRA utility (gamma = 2):** Risk aversion parameter controls the aggressiveness of the allocation.
- **Maximum leverage:** L1 norm of weights capped at 2.0x.
- **Drift-triggered rebalancing:** Rebalance when max weight drift exceeds 5% or every 5 trading days, whichever comes first.
- **Transaction cost filtering:** 10 bps proportional cost with a 2*tc no-trade zone prevents unnecessary turnover.
- **Ridge regularisation:** `1e-4 * I` added to covariance for numerical stability in matrix inversion.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Stable, diversified markets | Strong | BL + Ledoit-Wolf produce excellent risk-adjusted allocations |
| High cross-asset correlation | Moderate | Covariance shrinkage helps; but diversification benefit is limited |
| Rising rates (shifting r) | Moderate | Time-varying r_t adapts, but with estimation lag |
| Fat-tailed returns | Moderate | CRRA utility assumed; Gaussian covariance may understate tail risk |
| Regime breaks | Weak | Rolling window estimation lags structural shifts in mu, Sigma |

---

## 9. RMT Eigenportfolio

### Mathematical Domain

Random matrix theory, spectral analysis, Marchenko-Pastur law, principal component analysis, minimum-variance optimisation.

### Key Equations

**Marchenko-Pastur distribution:** For the eigenvalues of a sample correlation matrix of `N` assets observed over `T` periods with ratio `q = N/T`:

    f(lambda) = (T/N) / (2*pi*sigma^2) * sqrt((lambda_+ - lambda)(lambda - lambda_-)) / lambda

Bulk edges:

    lambda_+/- = sigma^2 * (1 +/- sqrt(N/T))^2

Eigenvalues within `[lambda_-, lambda_+]` are consistent with random noise. Eigenvalues exceeding `lambda_+` carry genuine signal.

**Covariance cleaning:** Replace noise eigenvalues with their mean:

    lambda_clean_i = mean(lambda_noise)    for lambda_i <= lambda_+
    lambda_clean_i = lambda_i              for lambda_i > lambda_+

Rescale so `trace = N` (correlation matrix convention). Reconstruct:

    C_clean = Q * diag(lambda_clean) * Q'

**Eigenportfolio construction:** The eigenvector `v_k` associated with signal eigenvalue `lambda_k` defines an eigenportfolio. Weights are normalised: `||v_k||_1 = 1`.

Eigenportfolio returns: `r_{eigen,k,t} = SUM_i v_{k,i} * r_{i,t}`.

**Signal generation:**

- Market eigenportfolio (k=0, largest eigenvalue): Trend-follow. Signal = `sign(SUM_{last 20 days} r_{eigen,0})`.
- Sector/style eigenportfolios (k >= 1): Mean-revert. z-score of cumulative return; if `|z| > 2`, fade.

**Eigenvalue-based importance weighting:**

    w_factor_k = lambda_k / SUM_j lambda_j

**Minimum-variance portfolio from cleaned covariance:**

    w_mv = Sigma_clean^{-1} * 1 / (1' * Sigma_clean^{-1} * 1)

**Final blend:**

    w_final = 0.70 * w_factor + 0.30 * w_min_var

### Theoretical Edge

Sample correlation matrices are notoriously noisy: with N assets and T observations, the spectral density of the sample correlation matrix follows the Marchenko-Pastur law even when the true correlations are zero. RMT precisely identifies which eigenvalues exceed the noise threshold and thus carry genuine information. Cleaning the correlation matrix by suppressing noise eigenvalues dramatically improves out-of-sample portfolio performance. The market eigenportfolio captures systematic risk (trend-followed), while higher eigenportfolios capture sector/style rotations (mean-reverted).

### Risk Model

- **Marchenko-Pastur noise separation:** Only eigenvalues above `lambda_+` are treated as signal; the rest are shrunk to their mean.
- **Minimum-variance blend (30%):** Provides a risk-management anchor regardless of factor signal quality.
- **Maximum leverage cap:** 1.5x gross exposure.
- **Periodic re-estimation:** Every 21 trading days, the correlation matrix is re-estimated and eigenportfolios are recomputed.
- **Weight normalisation:** Weights sum to at most 1.0 in absolute value.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| High cross-sectional correlation | Strong | Market eigenvalue dominates; trend-following captures direction |
| Sector rotation | Strong | Higher eigenportfolios capture rotations; mean-reversion profits |
| Random / uncorrelated stocks | Weak | Few signal eigenvalues; strategy degrades to min-var |
| Sudden factor rotation | Moderate | 21-day re-estimation has lag; stale eigenportfolios underperform |
| Concentrated markets | Moderate | Very few dominant eigenvalues; limited factor diversity |

---

## 10. Entropy Regularized

### Mathematical Domain

Online convex optimisation, information theory, Shannon entropy, multiplicative weight updates, convex programming.

### Key Equations

**Entropy-regularised mean-variance objective:**

    max_w { mu'w - (gamma/2) * w'Sigma*w + lambda * H(w) }

subject to `w >= 0, SUM w_i = 1` (probability simplex).

**Shannon entropy:**

    H(w) = -SUM_i w_i * ln(w_i)

with `0 * ln(0) := 0`. The entropy term penalises concentration, pulling weights toward uniform when `lambda` is large.

**Entropy gradient:**

    dH/dw_i = -(1 + ln(w_i))

**Adaptive entropy regularisation:**

    lambda = lambda_base * ln(1 + kappa(Sigma))

where `kappa(Sigma) = lambda_max / lambda_min` is the condition number of the covariance matrix. Ill-conditioned covariance (highly correlated assets) triggers stronger regularisation toward uniform weights.

**Exponentiated Gradient (EG) algorithm (Helmbold et al. 1998):**

    w_{t+1,i} = w_{t,i} * exp(eta * r_{t,i}) / Z_t

where `Z_t = SUM_j w_{t,j} * exp(eta * r_{t,j})` normalises to the simplex.

**Cumulative regret bound (Cover 1991, Helmbold et al. 1998):**

    R_T = max_i { SUM_{t=1}^T r_{t,i} } - SUM_{t=1}^T w_t' r_t <= O(sqrt(T * ln(N)))

This is the difference between the best single asset in hindsight and the portfolio's cumulative return.

**AdaGrad-adaptive learning rate:**

    eta_t = eta_0 / sqrt(SUM_{s=1}^t r_s^2 + epsilon)

This adapts the EG step size based on accumulated return magnitudes.

**Simplex projection (Duchi et al. 2008):**

Efficient O(N log N) algorithm projecting any vector onto `{w >= 0, SUM w = 1}`.

**Final blend:**

    w_final = alpha * w_EG + (1 - alpha) * w_MV_entropy

where `alpha = 0.5` (default) combines the online and batch components.

### Theoretical Edge

The EG algorithm provides a *no-regret guarantee* -- regardless of the market environment, the portfolio's cumulative return is within `O(sqrt(T log N))` of the best single asset in hindsight. This is a worst-case (adversarial) bound that holds even when returns are chosen by an adversary. The entropy-regularised mean-variance component adds Bayesian regularisation: when covariance estimates are noisy (high condition number), the entropy term automatically pulls weights toward uniform, preventing the estimation-error-driven concentration that plagues standard mean-variance optimisation.

### Risk Model

- **No-regret guarantee:** Theoretical bound `R_T <= sqrt(2*T*ln(N))` ensures bounded underperformance versus the best fixed portfolio.
- **Long-only constraint:** Weights on the probability simplex prevent short selling and leverage.
- **Adaptive regularisation:** `lambda` scales with `kappa(Sigma)`, increasing diversification pressure when estimation quality is poor.
- **Covariance conditioning:** Eigenvalue floor at `1e-6` ensures positive-definiteness.
- **Regret tracking:** Real-time monitoring of cumulative regret against the theoretical bound provides a diagnostic for strategy health.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| One dominant asset | Moderate | EG converges toward best asset, but slowly (O(sqrt(T)) regret) |
| Diversified returns | Strong | Entropy regularisation naturally diversifies; no concentration risk |
| Mean-reverting cross-section | Moderate | EG's momentum bias may lag reversals |
| High vol / crises | Moderate | Long-only limits hedging; entropy keeps diversification |
| Rotating leadership | Strong | EG adapts online; softmax explores alternatives |

---

## 11. Fractional Differentiation

### Mathematical Domain

Fractional calculus, long-range dependence, unit root testing, memory preservation in time series.

### Key Equations

**Fractional differentiation operator:**

    (1 - B)^d = SUM_{k=0}^{inf} (-1)^k * C(d,k) * B^k

where `B` is the backshift operator (`B * x_t = x_{t-1}`), `0 < d < 1`, and

    C(d,k) = Gamma(d+1) / (Gamma(k+1) * Gamma(d-k+1))

**Recursive weight computation:**

    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k

Weights are accumulated until `|w_k| < threshold` (default `1e-5`), determining the effective filter width.

**Fixed-width fractional differentiation:**

    x^{(d)}_t = SUM_{k=0}^{K} w_k * x_{t-k}

where `K` is the window length determined by the weight threshold.

**Optimal d search:** Find the minimum `d in [0, 1]` such that the fractionally-differenced series passes the Augmented Dickey-Fuller test at significance level `alpha = 0.05`.

Key insight: `d = 0` is raw prices (non-stationary, maximum memory). `d = 1` is standard returns (stationary, zero memory). The optimal `d*` is the smallest value achieving stationarity while preserving the maximum amount of long-range dependence.

**Regime classification via d*:**

    d* < 0.4  =>  Strong memory  =>  Mean-reversion regime
    d* >= 0.4 =>  Weak memory    =>  Momentum regime

**Signal generation:**

Z-score of frac-diff series:

    z_t = (x^{(d*)}_t - rolling_mean) / rolling_std

Mean-reversion regime: Buy when `z < -1.5`, sell when `z > 1.5`, exit at `|z| < 0.3`.
Momentum regime: Buy when `frac_diff_momentum > 0` AND `z > 1.5`; sell when momentum < 0 AND `z < -1.5`.

**Position sizing:**

    weight = min(|z_t| / entry_z, 1.0)

### Theoretical Edge

Standard returns (`d = 1`) discard ALL long-range memory from the price series. This throws away potentially predictive information about the market's memory structure. Fractional differentiation preserves the maximum amount of memory while still achieving stationarity. The value of `d*` itself is a regime indicator: low `d*` implies strong market memory (mean-reverting), while high `d*` implies weak memory (more efficient / momentum-like). This adaptive regime classification is derived from the data's own memory structure rather than imposed externally.

### Risk Model

- **Hysteresis:** Entry/exit z-score thresholds prevent whipsawing (`entry = 1.5`, `exit = 0.3`).
- **Momentum confirmation:** In the momentum regime, both frac-diff momentum AND z-score must agree before entry.
- **Quarterly re-estimation (63 days):** `d*` is periodically updated to track evolving market memory structure.
- **Weight threshold truncation:** Filter weights below `1e-5` are dropped, preventing infinite lookback.
- **d* history tracking:** Changes in `d*` over time serve as a diagnostic for market efficiency evolution.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Strong memory / low d* | Strong | Mean-reversion signals exploit persistent spread dynamics |
| Weak memory / high d* | Moderate | Momentum signals capture trends, but with less edge than dedicated momentum strategies |
| Transition (d* changing) | Moderate | Quarterly re-estimation may lag rapid changes; regime misclassification possible |
| Efficient markets (d* ~ 0.5) | Weak | Near the random-walk boundary; signals are noisy |
| Structural breaks | Variable | ADF test may yield unstable d* near breakpoints |

---

## 12. Levy Jump Detection

### Mathematical Domain

Levy processes, jump-diffusion models, bipower variation, extreme value theory (Gumbel distribution).

### Key Equations

**Merton (1976) jump-diffusion model:**

    dS/S = mu * dt + sigma * dW + J * dN

where:
- `W` is a Wiener process (continuous component),
- `N` is a Poisson process with intensity `lambda` (jump arrivals),
- `J ~ N(mu_J, sigma_J^2)` is the jump size distribution.

**Realized variation (includes jumps):**

    RV_t = (1/n) * SUM_{i=1}^{n} r_i^2

**Bipower variation (Barndorff-Nielsen & Shephard 2004):**

    BV_t = (pi/2) * (1/(n-1)) * SUM_{i=2}^{n} |r_i| * |r_{i-1}|

BV consistently estimates integrated variance *without* the jump component because `E[|Z|] = sqrt(2/pi)` for `Z ~ N(0,1)`, and the product of consecutive absolute returns converges to the continuous variance.

**Jump variation:**

    JV_t = max(RV_t - BV_t, 0)

**Relative jump intensity:**

    JI_t = JV_t / RV_t

Measures the fraction of return variation attributable to jumps.

**Lee-Mykland (2008) jump test statistic:**

    L_t = |r_t| / sigma_hat_t

where `sigma_hat_t = sqrt(BV_t)`. Under H0 (no jump), the maximum of `|L_t|` converges to a Gumbel distribution.

**Gumbel extreme value distribution parameters:**

    Location: a_n = sqrt(2*ln(n)) - (ln(pi) + ln(ln(n))) / (2*sqrt(2*ln(n)))
    Scale: b_n = 1 / sqrt(2*ln(n))

Critical value:

    C_n = a_n - b_n * ln(-ln(1 - significance))

A return is classified as a jump if `L_t > C_n` at significance level 1%.

**Return decomposition:**

    r_t = r_t^{continuous} + r_t^{jump}

where `r_t^{continuous} = r_t * I(L_t <= C_n)` and `r_t^{jump} = r_t * I(L_t > C_n)`.

**Continuous momentum signal:**

Cumulative continuous returns over 252 days, z-scored and clipped to `[-1, +1]`.

**Jump mean-reversion signal:**

After a detected jump at time `t_J`:
- Wait `delay = 3` days.
- Signal direction: opposite to jump sign (fade the jump).
- Signal magnitude: `min(|r_{jump}| / 0.03, 1.0)` (normalised by a 3% move).
- Linear decay over `decay = 10` days.

**Composite signal:**

    signal_t = 0.60 * continuous_momentum + 0.40 * jump_reversion

### Theoretical Edge

Financial returns are NOT Gaussian -- they exhibit jumps that create fat tails, which a standard Brownian motion model misses entirely. By decomposing returns into continuous and jump components, the strategy exploits two distinct dynamics: continuous returns exhibit momentum (persistent trends), while jumps tend to mean-revert (over-reactions are corrected). The bipower variation provides a jump-robust volatility estimate, enabling precise jump detection via extreme value theory. The tail risk premium (higher expected returns for jump-prone stocks) provides an additional source of alpha.

### Risk Model

- **Gumbel-calibrated significance:** Jumps are detected at the 1% level, minimising false positives.
- **Jump reversion delay (3 days):** Prevents premature entry during the immediate post-jump volatility.
- **Linear decay (10 days):** Jump reversion signal fades, preventing stale positions.
- **Signal smoothing:** 5-day EMA prevents whipsawing from spurious jump detections.
- **Dead zone:** Smoothed signal must exceed `+/- 0.15` to generate a position.
- **Jump intensity weight modulation:** Position size is scaled by relative jump frequency (tail risk premium), capped at `[0.5, 1.5]`.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Event-driven markets | Strong | Jumps from earnings/news are detected and faded |
| Steady trends | Moderate | Continuous momentum signal captures trend; jump component dormant |
| Flash crashes | Strong | Large negative jumps trigger strong reversion signal |
| Clustered jumps | Variable | Overlapping reversion signals may cancel or accumulate unpredictably |
| Thin markets (illiquid) | Weak | Bipower variation may be noisy with sparse observations |

---

## 13. Topological Data Analysis

### Mathematical Domain

Algebraic topology (persistent homology), dynamical systems (Takens' embedding theorem), spectral geometry, information theory.

### Key Equations

**Takens' delay-coordinate embedding theorem:**

For a time series `r_0, r_1, ..., r_{N-1}`, construct vectors:

    x_t = (r_t, r_{t-tau}, r_{t-2*tau}, ..., r_{t-(d-1)*tau})

where `tau = 5` (delay) and `d = 3` (embedding dimension). Takens' theorem guarantees that for generic observations of a dynamical system, this embedding is diffeomorphic to the original attractor when `d >= 2*dim(attractor) + 1`.

**Gram matrix of the point cloud:**

    G_{ij} = <x_i, x_j>

The eigenvalues of `G` encode the geometric structure of the point cloud.

**Spectral gap (H0 proxy):**

    gap = (lambda_1 - lambda_2) / lambda_1

Measures dominant-mode concentration. Near 1 implies a single dominant cluster; near 0 implies uniform structure.

**Spectral entropy:**

    H = -SUM_i p_i * ln(p_i) ,    p_i = lambda_i / SUM_j lambda_j

- High entropy: uniform eigenvalue spread (normal market, well-diversified dynamics)
- Low entropy: one dominant mode (trend, bubble, or pre-crash concentration)

**Persistent homology (H0 -- connected components):**

Approximate via Vietoris-Rips filtration:
1. Compute pairwise distance matrix `D_{ij} = ||x_i - x_j||_2`.
2. Sweep threshold `epsilon` from 0 to `max(D)`.
3. At each `epsilon`, count connected components via union-find.
4. The resulting Betti-0 curve `beta_0(epsilon)` tracks how components merge as the scale increases.

**Topological complexity (total persistence):**

    complexity = integral beta_0(epsilon) / n  d(epsilon)

Measures the total amount of topological structure across scales. Normalised by `n` for scale invariance.

**Wasserstein proxy between persistence diagrams:**

Approximate via the L1 distance between normalised, sorted eigenvalue distributions:

    W_1 ~ SUM_i |p_i^{prev} - p_i^{curr}|

Large values indicate rapid topological change (regime transition).

**Regime classification:**

- **Normal regime** (high entropy): Use 20-day momentum signal.
- **Stressed regime** (low entropy, bottom 25th percentile): Defensive position (`-0.5`).
- **Transition** (large entropy change > 2 rolling std): Go flat for 5-day cooldown.

**Regime score:**

    score = (1 - cw) * (1 - entropy_percentile/100) + cw * complexity_percentile

where `cw = 0.3` (complexity weight). Score in `[0, 1]`; 0 = fully normal, 1 = fully stressed.

### Theoretical Edge

Topological features are invariant to smooth deformations of the underlying space -- they capture the *shape* of market dynamics rather than specific parameter values. A market transitioning from a normal regime to a crash exhibits characteristic topological signatures (e.g., the point cloud of returns collapses from a dispersed cloud to a concentrated cluster) that appear before traditional statistical measures detect the change. The spectral entropy provides a single scalar summarising the complexity of the market's dynamical attractor, enabling regime classification without parametric assumptions.

### Risk Model

- **Entropy-based stress detection:** Markets below the 25th percentile of historical entropy are classified as stressed.
- **Transition cooldown (5 days):** After detecting a large entropy change, the strategy goes flat to avoid acting on transient topological disruptions.
- **Regime-score scaling:** Position weight in normal regime is reduced as regime score approaches stressed territory: `weight = clip(1 - score, 0.1, 1.0)`.
- **Defensive positioning:** In stressed regimes, signal direction is short (`-0.5`) rather than zero, providing crash protection.
- **Adaptive thresholding:** Entropy threshold is calibrated from training data (25th percentile), adapting to the asset's historical distribution.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Pre-crash (bubble formation) | Strong | Low spectral entropy detects concentration before crash |
| Steady trends | Moderate | Momentum signal in normal regime |
| Post-crash recovery | Moderate | Entropy normalises; transition cooldown may delay re-entry |
| Choppy / mean-reverting | Weak | Momentum sub-signal may whipsaw |
| Gradual regime change | Strong | Spectral entropy tracks smooth transitions |

---

## 14. Rough Volatility

### Mathematical Domain

Fractional Brownian motion, self-similar processes, detrended fluctuation analysis, rescaled range analysis, variogram estimation.

### Key Equations

**Fractional Brownian motion (fBm) covariance:**

    E[B^H_s * B^H_t] = (1/2) * (|s|^{2H} + |t|^{2H} - |s-t|^{2H})

where `H in (0, 1)` is the Hurst exponent.

**Hurst exponent regimes:**

    H < 0.5 : rough / anti-persistent (mean-reverting increments)
    H = 0.5 : standard Brownian motion (random walk)
    H > 0.5 : smooth / persistent (trending increments)

**Key empirical finding (Gatheral, Jaisson & Rosenbaum 2018):**

Log-volatility follows fBm with `H_vol ~ 0.1` (much rougher than Brownian motion). This is universal across asset classes.

**Detrended Fluctuation Analysis (DFA):**

1. Cumulative deviation profile: `Y(k) = SUM_{i=1}^{k} (x_i - <x>)`
2. For each box size `n`, divide `Y` into non-overlapping segments.
3. Detrend each segment via least-squares linear fit.
4. Root-mean-square fluctuation: `F(n) = sqrt(mean(residuals^2))`
5. Hurst exponent: `H = slope of ln(F(n)) vs ln(n)`

**Rescaled Range (R/S) analysis:**

For each window of size `n`:
1. `Z(k) = SUM_{i=1}^{k} (x_i - x_bar)` (cumulative deviation)
2. `R(n) = max(Z) - min(Z)` (range)
3. `S(n) = std(x_1,...,x_n)` (standard deviation)
4. `R/S ~ c * n^H` => `H = slope of ln(R/S) vs ln(n)`

**Blended Hurst estimate:**

    H_blend = 0.7 * H_DFA + 0.3 * H_RS

DFA is given higher weight as it is more robust to non-stationarity.

**Variogram method for volatility roughness:**

    E[|ln(sigma_{t+Delta}) - ln(sigma_t)|^q] ~ Delta^{qH_vol}

`H_vol = slope of ln(moments) vs ln(lag) / q`, using `q = 2`.

**Regime-adaptive signal:**

- `H > 0.55` (trending): Momentum signal = `sign(SUM_{12 days} r_t)`
- `H < 0.45` (mean-reverting): Mean-reversion signal = `-z_price` where `z_price = (P_t - MA_20) / rolling_std`
- `0.45 <= H <= 0.55` (random walk): Blend with reduced size, or flat if `|H - 0.5| < 0.02`

**Vol roughness secondary signal:**

When `H_vol < 0.2` and vol is elevated above its median, the strategy takes a long tilt (expecting vol compression due to anti-persistent volatility dynamics).

**Position sizing:**

    size_factor = min(|H - 0.5| / 0.5, 1.0)

Proportional to deviation from random walk: stronger deviation gives larger positions.

### Theoretical Edge

The Hurst exponent provides a model-free measure of the persistence/anti-persistence of increments. Unlike parametric approaches, it makes no distributional assumptions. The roughness of volatility (`H_vol ~ 0.1`) is one of the most robust empirical findings in quantitative finance and implies that volatility spikes tend to be quickly corrected (anti-persistent), creating predictable vol-timing opportunities. By separately estimating `H` on returns (for directional regime classification) and on log-vol (for vol-timing), the strategy extracts two orthogonal signals from a single time series.

### Risk Model

- **R^2 quality gate:** DFA and R/S regressions must have `R^2 > 0.8` (DFA) or `R^2 > 0.7` (R/S) to be trusted; otherwise `H` defaults to 0.5.
- **Minimum signal strength:** `|H - 0.5| < 0.02` results in a flat position (too close to random walk to trade).
- **Signal smoothing:** 5-day EMA prevents whipsawing from noisy H estimates.
- **Maximum position:** Hard cap at 1.0.
- **Reduced re-estimation frequency:** H is computed every 5 days (returns) and 20 days (vol roughness) for computational efficiency and stability.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Strong trends (H > 0.6) | Strong | Momentum signal is well-calibrated; high sizing |
| Mean-reverting (H < 0.4) | Strong | Mean-reversion signal exploits anti-persistence |
| Random walk (H ~ 0.5) | Flat/Weak | No edge; strategy correctly goes flat |
| Volatility spikes with rough vol | Strong | H_vol << 0.5 predicts vol compression; long tilt profits |
| Regime transitions (H changing) | Moderate | Rolling estimation captures change, but with 5-day lag |

---

## 15. Bayesian Changepoint

### Mathematical Domain

Bayesian inference, conjugate prior distributions, online changepoint detection, predictive probability.

### Key Equations

**Bayesian Online Changepoint Detection (BOCPD) (Adams & MacKay 2007):**

Let `r_t` denote the *run length* -- the number of observations since the last changepoint. BOCPD computes the posterior:

    P(r_t | x_{1:t})

recursively via:

    P(r_t = 0 | x_{1:t}) proportional to SUM_{r_{t-1}} P(r_{t-1} | x_{1:t-1}) * H(r_{t-1}) * pi_t(x_t)

    P(r_t = r_{t-1}+1 | x_{1:t}) proportional to P(r_{t-1} | x_{1:t-1}) * (1 - H(r_{t-1})) * pi_t(x_t)

where:
- `H(r)` is the hazard function (prior probability of a changepoint at run length `r`). For a constant hazard: `H = 1/lambda` (expected run length = `lambda`).
- `pi_t(x_t)` is the predictive probability of `x_t` under the current segment model.

**Normal-Inverse-Gamma (NIG) conjugate prior:**

For Gaussian observations `x ~ N(mu, sigma^2)`:

Prior: `(mu, sigma^2) ~ NIG(mu_0, kappa_0, alpha_0, beta_0)`

Posterior after `n` observations with sample mean `x_bar` and sum of squares `S`:

    kappa_n = kappa_0 + n
    mu_n = (kappa_0 * mu_0 + n * x_bar) / kappa_n
    alpha_n = alpha_0 + n/2
    beta_n = beta_0 + S/2 + (kappa_0 * n * (x_bar - mu_0)^2) / (2 * kappa_n)

**Predictive distribution (Student-t):**

    x_{n+1} | x_{1:n} ~ t_{2*alpha_n}(mu_n, beta_n * (kappa_n + 1) / (alpha_n * kappa_n))

The predictive probability `pi_t(x_t)` is the density of this Student-t distribution evaluated at `x_t`.

**Changepoint probability:**

    P(changepoint at t) = P(r_t = 0 | x_{1:t})

**MAP run length:**

    r*_t = argmax_r P(r_t = r | x_{1:t})

**Trading signal:**

High changepoint probability triggers regime reassessment. The strategy reduces or reverses positions when `P(r_t = 0) > threshold` and re-enters using the new segment's estimated parameters after a stabilisation period.

### Theoretical Edge

BOCPD provides exact Bayesian inference over the location of changepoints in real time, with computational cost linear in the run length. The Normal-Inverse-Gamma conjugate prior enables closed-form posterior updates, avoiding MCMC or other approximations. The predictive probability framework automatically detects when current data is inconsistent with the running model, without requiring the user to specify what kind of change to look for. This is fundamentally different from fixed-window approaches that cannot distinguish between a structural break and a large-but-temporary shock.

### Risk Model

- **Changepoint probability threshold:** Positions are reduced only when `P(r_t = 0)` exceeds a configurable significance level.
- **Stabilisation period:** After a detected changepoint, the strategy waits for sufficient observations to re-estimate segment parameters before re-entering.
- **Hazard function calibration:** The expected run length `lambda` controls the prior sensitivity to changepoints; too small causes false positives, too large causes delayed detection.
- **NIG conjugate updates:** Exact posterior computation eliminates approximation error.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Sudden regime changes | Strong | BOCPD detects structural breaks rapidly |
| Gradual drift | Moderate | Changepoint probability rises slowly; detection is delayed |
| Stable regimes | Strong | Low P(r=0) allows full exposure; run length grows |
| Frequent small changes | Weak | Hazard rate tuning is critical; may over-detect or under-detect |
| Non-Gaussian shocks | Moderate | Student-t predictive helps with tails, but NIG assumes Gaussian |

---

## 16. Sparse Mean Reversion

### Mathematical Domain

Compressed sensing, high-dimensional statistics, LASSO regularisation, sparse portfolio optimisation.

### Key Equations

**Sparse cointegration model:** Find a sparse portfolio `w` such that the portfolio `S_t = w' P_t` is mean-reverting.

**LASSO-regularised portfolio:**

    min_w  || w' r - target ||^2 + lambda * ||w||_1

subject to `SUM w_i = 0` (dollar-neutral) and stationarity of the portfolio time series.

**Equivalently, predictability-based formulation:**

    min_w  SUM_t (w' P_{t+1} - rho * w' P_t - c)^2 + lambda * ||w||_1

where `rho < 1` enforces mean reversion and `lambda` controls sparsity.

**Augmented Dickey-Fuller test on the sparse portfolio:**

    Delta S_t = alpha + delta * S_{t-1} + SUM_{k=1}^{p} phi_k * Delta S_{t-k} + epsilon_t

H0: `delta = 0` (unit root). The fitted `w` is accepted only if ADF rejects `H0`.

**Compressed sensing connection:**

When `N >> T` (many more assets than observations), the covariance matrix is singular. Sparsity (via L1 penalty) acts as a regulariser, selecting only a small subset of assets that form a genuinely cointegrated portfolio. This is analogous to compressed sensing's ability to recover sparse signals from underdetermined systems.

**Elastic net extension:**

    min_w  || w' r - target ||^2 + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2

The L2 penalty encourages grouping of correlated assets (unlike pure LASSO, which selects arbitrarily among correlated features).

**Trading signal:** Once a sparse cointegrated portfolio `w` is identified:

    z_t = (S_t - mu_S) / sigma_S

Trade the z-score as in standard mean reversion: enter at `|z| > 2`, exit at `|z| < 0.5`.

### Theoretical Edge

In large universes (hundreds or thousands of assets), standard cointegration testing on all pairs is computationally infeasible and statistically unreliable (multiple testing problem). Sparse optimisation directly searches for the best mean-reverting portfolio across all assets simultaneously, with the L1 penalty automatically selecting the few assets that contribute to cointegration. This finds genuine multi-leg cointegrated portfolios that pairwise testing would miss.

### Risk Model

- **Sparsity constraint:** L1 penalty limits the number of active positions, concentrating capital in assets with genuine cointegration relationships.
- **Dollar neutrality:** The constraint `SUM w_i = 0` eliminates market exposure.
- **ADF validation:** Only portfolios that pass the stationarity test at `p < 0.05` are traded.
- **Regularisation path:** Cross-validation over `lambda` selects the optimal sparsity level.
- **Out-of-sample testing:** Cointegration relationships estimated in-sample are validated on held-out data before live trading.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Large cross-section / many assets | Strong | Sparse optimisation scales well; finds multi-asset cointegration |
| Stable factor structure | Strong | Cointegration relationships persist |
| Factor rotation | Weak | Sparse portfolio may include assets that de-cointegrate |
| Low vol / tight spreads | Moderate | Spreads may not deviate enough for profitable entry |
| High correlation | Moderate | Many near-cointegrated portfolios exist; selection is robust |

---

## 17. Momentum Crash Hedge

### Mathematical Domain

Extreme value theory (EVT), dynamic hedging, tail risk management, conditional value-at-risk.

### Key Equations

**Momentum crash characterisation:**

Momentum strategies suffer from occasional severe crashes (e.g., March 2009: momentum lost ~40% in one month). These crashes are characterised by:
1. Rapid reversal of prior losers (short leg rallies violently)
2. Coincidence with high VIX / market stress
3. Convexity: crash magnitude is non-linear in market recovery speed

**Generalised Pareto Distribution (GPD) for tail modelling:**

For exceedances `y = x - u` over a high threshold `u`:

    P(Y > y | Y > 0) = (1 + xi * y / sigma)^{-1/xi}

where `xi` is the shape parameter (tail index) and `sigma` is the scale parameter.

**Conditional Value-at-Risk (CVaR / Expected Shortfall):**

    CVaR_alpha = (1 / (1-alpha)) * integral_alpha^1 VaR_u du

For the GPD:

    CVaR_alpha = VaR_alpha / (1 - xi) + (sigma - xi * u) / (1 - xi)

**Dynamic hedge ratio:**

    hedge_t = -beta_{momentum, market} * P(crash | indicators_t)

The hedge is a short position in the momentum portfolio (or equivalently, a long position in recent losers) scaled by the conditional crash probability.

**Crash probability indicators:**

1. **Market state:** Recent market return below -10% triggers elevated crash probability.
2. **Momentum spread:** When the return spread between winners and losers widens beyond 2 std, reversal risk increases.
3. **Volatility regime:** VIX above its 90th percentile signals heightened tail risk.
4. **Cross-sectional dispersion:** Very high dispersion precedes momentum crashes.

**Dynamic hedging position:**

    w_hedge = -f(indicators) * w_momentum

where `f(indicators) in [0, 1]` maps crash indicators to hedge intensity.

### Theoretical Edge

Momentum strategies have excellent long-run Sharpe ratios but suffer from catastrophic left-tail events. These crashes are not random -- they cluster in specific market environments (high vol, rapid market recovery after a downturn). By dynamically hedging the momentum portfolio when crash indicators are elevated, the strategy preserves most of the momentum alpha while truncating the left tail. EVT provides a rigorous framework for modelling the tail behaviour that standard Gaussian models dramatically underestimate.

### Risk Model

- **GPD tail modelling:** Captures the fat-tailed nature of momentum crash losses.
- **Dynamic hedge intensity:** Hedge scales continuously with crash probability, avoiding binary on/off switching.
- **CVaR constraint:** Portfolio-level CVaR is monitored; hedge intensity increases when CVaR exceeds a predefined budget.
- **Indicator diversification:** Multiple crash indicators (market state, momentum spread, vol regime, dispersion) reduce the risk of model failure.
- **Cost-aware hedging:** Hedge is only activated above a minimum crash probability threshold to avoid chronic drag from unnecessary hedging.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Normal momentum | Strong | Small hedge cost; momentum alpha captured |
| Pre-crash (high vol, wide spread) | Strong | Hedge activates before crash; protects portfolio |
| Momentum crash | Very strong | Full hedge offsets crash losses |
| Gradual momentum decay | Moderate | Crash indicators may not trigger; hedge is dormant |
| Bull market (low vol) | Moderate | Minimal hedge cost; full momentum exposure |

---

## 18. Kelly Growth Optimal

### Mathematical Domain

Information theory, geometric Brownian motion, stochastic growth rates, convex optimisation.

### Key Equations

**Kelly criterion (single asset, binary outcomes):**

    f* = p/a - q/b

where `p` is win probability, `q = 1-p` is loss probability, `a` is the loss fraction, and `b` is the win fraction.

**Continuous Kelly for Gaussian returns:**

    f* = mu / sigma^2

where `mu` is the expected excess return and `sigma^2` is the return variance. This maximises the expected log-wealth growth rate:

    g(f) = E[ln(1 + f * r)] ~ f * mu - (f^2 * sigma^2) / 2

Setting `dg/df = 0` yields `f* = mu / sigma^2`.

**Multi-asset Kelly (log-optimal portfolio):**

    max_w  E[ln(w' r)]

For Gaussian returns, the solution is:

    w* = Sigma^{-1} * mu

(the tangency portfolio scaled by the inverse of risk aversion). This maximises the expected geometric growth rate.

**Growth rate of Kelly portfolio:**

    g* = mu' * Sigma^{-1} * mu / 2

This is the maximum achievable geometric growth rate (the Shannon limit of the investment channel).

**Fractional Kelly:**

    w = alpha * w*    ,    alpha in (0, 1]

Fractional Kelly (typically `alpha = 0.5`, "half-Kelly") sacrifices some growth rate for significantly reduced variance and drawdown. The growth rate under half-Kelly is:

    g(0.5) = (3/8) * mu' * Sigma^{-1} * mu

which is 75% of the full-Kelly growth rate but with half the volatility.

**Information-theoretic interpretation (Cover & Thomas):**

The Kelly strategy maximises the growth rate of wealth, which equals the mutual information between the return distribution and the portfolio weight:

    g* = max_w I(w; r)

The portfolio is the "optimal code" for extracting information from the market.

**Drawdown probability bound:**

For the full Kelly strategy:

    P(max drawdown > d) <= 1/d

For fractional Kelly at fraction alpha:

    P(max drawdown > d) <= 1/d^{1/alpha}

Half-Kelly (`alpha = 0.5`): `P(max DD > 50%) ~ 4%`.

### Theoretical Edge

The Kelly criterion is the unique strategy that maximises the long-run geometric growth rate of wealth. Over sufficiently long horizons, a Kelly bettor will almost surely outperform any essentially different strategy. The information-theoretic foundation connects portfolio sizing to channel capacity: the Kelly portfolio extracts the maximum possible information (in bits) from the return distribution per unit time. Fractional Kelly sacrifices some growth for dramatic improvement in drawdown characteristics, making it practical for finite horizons.

### Risk Model

- **Half-Kelly default:** Reduces the full Kelly fraction by 50%, providing a 75% growth rate with 50% of the variance.
- **Floor weight (0.1):** Prevents complete silencing of a pair/asset due to noisy in-sample estimates.
- **Ceiling weight (1.0):** Prevents over-concentration.
- **Estimation window:** 252-day lookback for mean/variance estimation.
- **Drawdown probability monitoring:** Theoretical drawdown bounds provide an ex-ante risk budget.
- **Parameter uncertainty:** Kelly is highly sensitive to `mu` estimation error. Half-Kelly and shrinkage estimators mitigate this sensitivity.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Stable, positive drift | Strong | Kelly sizing exploits positive expected returns optimally |
| High Sharpe strategies | Very strong | Growth rate is proportional to `mu^2/sigma^2` |
| Estimation error / noisy mu | Moderate | Fractional Kelly and shrinkage provide robustness |
| Drawdowns / losses | Moderate | Half-Kelly limits drawdown; full Kelly can experience 50%+ drawdowns |
| Black swan / fat tails | Weak | Kelly assumes known distribution; fat tails amplify over-betting risk |

---

## 19. Microstructure

### Mathematical Domain

Market microstructure theory, price impact models, bid-ask spread estimation, information asymmetry.

### Key Equations

**Amihud (2002) illiquidity ratio:**

    ILLIQ_t = (1/D) * SUM_{d=1}^{D} |r_d| / (V_d * P_d)

where `r_d` is the daily return, `V_d` is the daily volume, and `P_d` is the price. This measures the daily price impact per dollar of trading volume. Higher ILLIQ indicates less liquid (more price impact) markets.

**Interpretation:** ILLIQ ~ Kyle's lambda, measuring the permanent price impact coefficient.

**Corwin-Schultz (2012) bid-ask spread estimator:**

Using daily high and low prices:

    beta = E[SUM_{j=0}^{1} (ln(H_{t+j}/L_{t+j}))^2]

    gamma = (ln(H_{t,t+1}/L_{t,t+1}))^2

where `H_{t,t+1}` and `L_{t,t+1}` are the two-day high and low.

    alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))

    spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))

This estimates the bid-ask spread from OHLC data alone, without requiring quote data.

**Kyle (1985) lambda (price impact coefficient):**

In Kyle's model of informed trading:

    Delta P = lambda * Delta x + noise

where `Delta x` is the order flow (signed volume) and `lambda` measures the permanent price impact. Estimated via regression of returns on signed volume.

**Roll (1984) spread estimator:**

If returns are serially correlated due to bid-ask bounce:

    spread = 2 * sqrt(-Cov(r_t, r_{t-1}))

(valid when `Cov(r_t, r_{t-1}) < 0`).

**VPIN (Volume-Synchronized Probability of Informed Trading):**

    VPIN = SUM |V_buy - V_sell| / (n * V_bucket)

Estimates the probability that trading is driven by informed traders. High VPIN precedes volatility events.

**Trading signals from microstructure:**

1. **Liquidity momentum:** Stocks with improving liquidity (declining Amihud / spread) tend to outperform.
2. **Spread mean-reversion:** Unusually wide spreads (above 2 std of rolling mean) compress, creating a short-term alpha signal.
3. **Information asymmetry timing:** High Kyle lambda or VPIN suggests informed trading; fade the initial move (if noise) or follow it (if signal confirmed).
4. **Liquidity premium:** Cross-sectionally, less liquid stocks earn a premium. Tilt toward high-Amihud stocks (long) vs. low-Amihud (short).

### Theoretical Edge

Market microstructure provides a window into the information flow and supply/demand dynamics that drive short-term price movements. The bid-ask spread, price impact, and order flow contain information about the balance between informed and uninformed trading that is invisible in price data alone. Stocks with deteriorating liquidity (widening spreads, increasing price impact) often precede negative returns, while liquidity improvements are a leading indicator of positive returns. The illiquidity premium (Amihud) is a well-documented cross-sectional return predictor.

### Risk Model

- **Execution cost awareness:** Microstructure signals are most valuable for stocks where they exceed transaction costs; the strategy filters for sufficient signal magnitude.
- **Liquidity screening:** Minimum liquidity thresholds prevent trading in stocks where price impact would consume the alpha.
- **Spread monitoring:** Real-time spread estimates inform execution timing (avoid trading when spreads are wide).
- **VPIN-based risk alerts:** Elevated VPIN triggers position reduction or hedge activation.
- **Adverse selection management:** Kyle's lambda is used to estimate the information content of incoming trades; high lambda positions are sized more conservatively.

### Expected Regime Performance

| Regime | Performance | Rationale |
|--------|------------|-----------|
| Normal liquidity | Moderate | Microstructure signals are informative but noisy |
| Liquidity crises | Strong | Spread widening and liquidity deterioration provide strong signals |
| High-frequency / well-arbitraged | Weak | Microstructure alpha is captured by faster participants |
| Pre-event (earnings, news) | Strong | Information asymmetry signals (VPIN, Kyle lambda) activate |
| Thin / illiquid markets | Variable | Strong signals but execution risk is high; price impact may consume alpha |

---

## Cross-Strategy Reference Matrix

### Signal Type Classification

| # | Strategy | Primary Signal | Secondary Signal |
|---|---------|----------------|------------------|
| 1 | OU Mean Reversion | Mean-reversion (z-score) | Kelly sizing |
| 2 | HMM Regime | Regime-conditional | Vol-targeting |
| 3 | Kalman Alpha | Alpha z-score | Beta hedge |
| 4 | Spectral Momentum | Multi-frequency momentum | IC-adaptive weighting |
| 5 | GARCH Vol | Vol mean-reversion + VRP | Vol-targeting |
| 6 | Optimal Transport | Cross-sectional ranking (W_1) | Quality filter (W_2) |
| 7 | Info Geometry | KL-based directional | Fisher-Rao regime damping |
| 8 | Stochastic Control | Merton optimal fraction | BL shrinkage |
| 9 | RMT Eigenportfolio | Factor momentum/MR | Min-variance blend |
| 10 | Entropy Regularized | EG + entropy-MV blend | Adaptive lambda |
| 11 | Fractional Diff | Memory-adaptive MR/momentum | Regime via d* |
| 12 | Levy Jump | Continuous momentum | Jump mean-reversion |
| 13 | TDA | Topology-based regime | Momentum / defensive |
| 14 | Rough Volatility | Hurst-adaptive momentum/MR | Vol roughness timing |
| 15 | Bayesian Changepoint | Changepoint-triggered | Segment re-estimation |
| 16 | Sparse Mean Reversion | Sparse cointegration | LASSO regularisation |
| 17 | Momentum Crash Hedge | Dynamic hedging | EVT tail model |
| 18 | Kelly Growth | Growth-optimal sizing | Fractional Kelly |
| 19 | Microstructure | Liquidity signals | Spread / information asymmetry |

### Correlation Structure Expectations

Strategies are designed with low mutual correlation across different market environments:

- **Mean-reversion cluster:** Strategies 1, 6, 11, 16 share mean-reversion logic but differ in universe construction and signal generation.
- **Regime-detection cluster:** Strategies 2, 13, 14, 15 detect regime changes via different mathematical frameworks (HMM, topology, Hurst, Bayesian).
- **Risk management cluster:** Strategies 5, 8, 17, 18 focus on risk-adjusted sizing and tail protection.
- **Factor / spectral cluster:** Strategies 4, 9 decompose returns into components via different mathematical bases (Fourier/wavelet vs. eigen).
- **Geometric / distributional cluster:** Strategies 6, 7 use distances on spaces of probability measures (Wasserstein vs. Fisher-Rao).
- **Online learning cluster:** Strategies 10, 18 draw on information-theoretic optimality (regret bounds, growth rate maximisation).

### Mathematical Dependency Map

```
Stochastic Calculus ──> OU Mean Reversion
                    ──> Stochastic Control (HJB)
                    ──> Rough Volatility (fBm)

Bayesian Inference ──> HMM Regime (Baum-Welch)
                   ──> Kalman Alpha (state-space)
                   ──> Bayesian Changepoint (BOCPD)
                   ──> Stochastic Control (Black-Litterman)

Information Theory ──> Information Geometry (KL, Fisher)
                   ──> Entropy Regularized (Shannon, EG)
                   ──> Kelly Growth (log-wealth)

Signal Processing ──> Spectral Momentum (DFT, DWT, Hilbert)
                   ──> Fractional Diff (fractional calculus)
                   ──> Rough Volatility (DFA, R/S)

Random Matrix Theory ──> RMT Eigenportfolio (Marchenko-Pastur)

Measure Theory ──> Optimal Transport (Wasserstein)
               ──> Information Geometry (Fisher-Rao)

Topology ──> TDA (persistent homology, Betti numbers)

Levy Processes ──> Levy Jump (bipower variation, Gumbel)
               ──> Rough Volatility (fractional processes)

High-Dimensional Statistics ──> Sparse Mean Reversion (LASSO)
                             ──> RMT Eigenportfolio

Extreme Value Theory ──> Momentum Crash Hedge (GPD, CVaR)
                      ──> Levy Jump (Gumbel)

Market Microstructure ──> Microstructure (Kyle, Amihud)
```

---

## References

1. Adams, R.P. & MacKay, D.J.C. (2007). "Bayesian Online Changepoint Detection." arXiv:0710.3742.
2. Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and time-series effects." Journal of Financial Markets, 5(1).
3. Atkinson, C. & Mitchell, A.F.S. (1981). "Rao's Distance Measure." Sankhya, Series A, 43.
4. Barndorff-Nielsen, O.E. & Shephard, N. (2004). "Power and bipower variation." Econometrica, 72(1).
5. Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
6. Corwin, S.A. & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices." Journal of Finance, 67(2).
7. Cover, T.M. (1991). "Universal Portfolios." Mathematical Finance, 1(1).
8. Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*. 2nd ed. Wiley.
9. Duchi, J. et al. (2008). "Efficient Projections onto the l1-Ball for Learning in High Dimensions." ICML.
10. Engle, R.F. & Granger, C.W.J. (1987). "Co-Integration and Error Correction." Econometrica, 55(2).
11. Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). "Volatility is rough." Quantitative Finance, 18(6).
12. Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993). "On the Relation between Expected Value and Volatility of the Nominal Excess Return on Stocks." Journal of Finance, 48(5).
13. Helmbold, D.P. et al. (1998). "On-Line Portfolio Selection Using Multiplicative Updates." Mathematical Finance, 8(4).
14. Hosking, J.R.M. (1981). "Fractional differencing." Biometrika, 68(1).
15. Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System Technical Journal, 35(4).
16. Kyle, A.S. (1985). "Continuous Auctions and Insider Trading." Econometrica, 53(6).
17. Lee, S.S. & Mykland, P.A. (2008). "Jumps in Financial Markets." Review of Financial Studies, 21(6).
18. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
19. Marchenko, V.A. & Pastur, L.A. (1967). "Distribution of eigenvalues for some sets of random matrices." Mathematics of the USSR-Sbornik, 1(4).
20. Merton, R.C. (1969). "Lifetime Portfolio Selection under Uncertainty." Review of Economics and Statistics, 51(3).
21. Merton, R.C. (1976). "Option pricing when underlying stock returns are discontinuous." Journal of Financial Economics, 3(1-2).
22. Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread." Journal of Finance, 39(4).
23. Uhlenbeck, G.E. & Ornstein, L.S. (1930). "On the Theory of the Brownian Motion." Physical Review, 36(5).

---

*Document generated for the strategy-with-bt quantitative trading framework.*
*Last updated: 2026-03-22.*
