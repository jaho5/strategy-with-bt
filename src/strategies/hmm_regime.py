"""
Hidden Markov Model regime-switching strategy.

Mathematical foundation
-----------------------
Hidden states : S_t in {Bull, Bear, Sideways}
Emission model: r_t | S_t ~ N(mu_{S_t}, sigma^2_{S_t})
Transition matrix: A[i,j] = P(S_{t+1}=j | S_t=i)

Parameter estimation uses the Baum-Welch algorithm (EM) via
``hmmlearn.hmm.GaussianHMM``.  The Viterbi algorithm recovers the most-likely
state sequence for diagnostics.

Strategy logic
--------------
* **Bull** regime (highest mean return)  -> go long
* **Bear** regime (lowest mean return)   -> go short / flat
* **Sideways** regime                    -> mean-reversion sub-strategy

Position sizing is scaled by the filtered regime probability and further
adjusted by a volatility-targeting overlay (annualised target = 15 %).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import entropy as scipy_entropy

from src.strategies.base import Strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class HMMRegimeConfig:
    """All tuneable hyper-parameters for the HMM regime strategy."""

    # HMM specification
    n_regimes: int = 3
    covariance_type: str = "full"
    n_em_iterations: int = 200
    em_tolerance: float = 1e-4
    random_state: int = 42

    # Rolling window for model fitting (trading days)
    rolling_window: int = 504  # ~2 years

    # Feature engineering
    vol_lookback: int = 20  # realised vol window (days)
    skew_lookback: int = 60  # return skewness window (days)

    # Signal generation
    short_allowed: bool = True  # set False to make Bear -> flat instead of short
    mr_lookback: int = 20  # mean-reversion z-score lookback for sideways regime

    # Position sizing
    entropy_threshold: float = 1.0  # nats; max entropy for 3 states ~ ln(3) ~ 1.099
    min_probability: float = 0.40  # minimum regime prob to act on signal

    # Volatility targeting
    target_annual_vol: float = 0.15
    realised_vol_lookback: int = 20  # for the vol-targeting overlay
    annualisation_factor: float = 252.0


# ---------------------------------------------------------------------------
# Helper: build feature matrix from log-returns
# ---------------------------------------------------------------------------


def _build_features(
    log_returns: pd.Series,
    *,
    vol_lookback: int = 20,
    skew_lookback: int = 60,
) -> pd.DataFrame:
    """Construct the observation matrix for the HMM.

    Features (per asset):
        1. Log return
        2. Realised volatility  (rolling std of log returns, *vol_lookback* days)
        3. Return skewness      (rolling skewness, *skew_lookback* days)

    Returns a DataFrame aligned to the original index with NaN rows dropped.
    """
    rvol = log_returns.rolling(vol_lookback).std()
    rskew = log_returns.rolling(skew_lookback).skew()

    features = pd.DataFrame(
        {
            "log_return": log_returns,
            "realised_vol": rvol,
            "return_skew": rskew,
        }
    )
    features.dropna(inplace=True)
    return features


# ---------------------------------------------------------------------------
# Core: label regimes after fitting
# ---------------------------------------------------------------------------


def _label_regimes(model: GaussianHMM) -> dict[str, int]:
    """Assign semantic labels to HMM states based on estimated means.

    The state with the highest mean return is labelled *Bull*, the lowest
    *Bear*, and the remaining state *Sideways*.

    Returns
    -------
    dict
        Mapping ``{"bull": int, "bear": int, "sideways": int}``.
    """
    # model.means_ has shape (n_components, n_features).
    # The first feature column is the log return.
    mean_returns = model.means_[:, 0]
    sorted_indices = np.argsort(mean_returns)

    labels: dict[str, int] = {
        "bear": int(sorted_indices[0]),
        "sideways": int(sorted_indices[1]),
        "bull": int(sorted_indices[2]),
    }
    return labels


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


class HMMRegimeStrategy(Strategy):
    """Hidden Markov Model regime-switching strategy.

    Parameters
    ----------
    config : HMMRegimeConfig | None
        Strategy hyper-parameters.  Defaults are used when *None*.
    """

    def __init__(self, config: HMMRegimeConfig | None = None) -> None:
        self.config = config or HMMRegimeConfig()

        # Populated by ``fit``
        self._model: GaussianHMM | None = None
        self._regime_labels: dict[str, int] = {}
        self._is_fitted: bool = False
        self._fit_features: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "HMMRegimeStrategy":
        """Fit a 3-state Gaussian HMM on the most recent rolling window.

        Parameters
        ----------
        data : pd.DataFrame
            Price data.  For a single asset pass a DataFrame with one column;
            for multiple assets each column is a ticker.  The index must be a
            DatetimeIndex.

        Returns
        -------
        self
        """
        cfg = self.config
        prices = self._to_series(data)
        log_returns = np.log(prices / prices.shift(1)).dropna()

        features = _build_features(
            log_returns,
            vol_lookback=cfg.vol_lookback,
            skew_lookback=cfg.skew_lookback,
        )

        # Use only the trailing rolling window for calibration
        if len(features) > cfg.rolling_window:
            features = features.iloc[-cfg.rolling_window:]

        if len(features) < cfg.n_regimes * 10:
            raise ValueError(
                f"Insufficient data for HMM fitting: need at least "
                f"{cfg.n_regimes * 10} observations, got {len(features)}."
            )

        X = features.values  # shape (T, n_features)

        model = GaussianHMM(
            n_components=cfg.n_regimes,
            covariance_type=cfg.covariance_type,
            n_iter=cfg.n_em_iterations,
            tol=cfg.em_tolerance,
            random_state=cfg.random_state,
            min_covar=1e-3,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                model.fit(X)

            # Covariance regularisation: ensure covars_ are positive-definite
            _regularize_covars(model)

        except Exception:
            logger.warning(
                "HMM fitting failed – strategy will produce flat signals.",
                exc_info=True,
            )
            self._model = None
            self._regime_labels = {}
            self._fit_features = features
            self._is_fitted = False
            return self

        self._model = model
        self._regime_labels = _label_regimes(model)
        self._fit_features = features
        self._is_fitted = True

        logger.info(
            "HMM fitted – regime means (log-return): Bull=%.5f, Bear=%.5f, "
            "Sideways=%.5f",
            model.means_[self._regime_labels["bull"], 0],
            model.means_[self._regime_labels["bear"], 0],
            model.means_[self._regime_labels["sideways"], 0],
        )

        return self

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate position signals for each asset in *data*.

        For every date the method:
        1. Computes filtered regime probabilities P(S_t=k | r_1,...,r_t).
        2. Determines the most-probable regime.
        3. Maps the regime to a raw directional signal (+1 / -1 / MR).
        4. Scales the signal by the regime probability.
        5. Reduces exposure when regime uncertainty is high (entropy check).
        6. Applies a volatility-targeting overlay.

        Parameters
        ----------
        data : pd.DataFrame
            Price data with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Position signals with the same index as *data* and one column per
            asset.  Values represent desired portfolio weight.
        """
        cfg = self.config
        tickers = data.columns.tolist()
        signals = pd.DataFrame(index=data.index, columns=tickers, dtype=float)
        signals[:] = 0.0

        if not self._is_fitted:
            logger.warning(
                "generate_signals called but model is not fitted – "
                "returning flat (zero) signals."
            )
            return signals

        for ticker in tickers:
            prices = data[ticker].dropna()
            if len(prices) < cfg.skew_lookback + 2:
                logger.warning(
                    "Skipping %s – insufficient history (%d rows).",
                    ticker,
                    len(prices),
                )
                continue

            ticker_signals = self._generate_single_asset_signals(prices)
            # Align back to global index
            signals.loc[ticker_signals.index, ticker] = ticker_signals.values

        return signals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_series(data: pd.DataFrame) -> pd.Series:
        """Collapse a (possibly multi-column) price DataFrame to a Series.

        When multiple columns are present the first one is used (the strategy
        is calibrated on a single representative asset or index).
        """
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        return data.iloc[:, 0]

    def _generate_single_asset_signals(self, prices: pd.Series) -> pd.Series:
        """Produce position weights for one asset."""
        cfg = self.config
        model = self._model
        labels = self._regime_labels

        log_returns = np.log(prices / prices.shift(1)).dropna()
        features = _build_features(
            log_returns,
            vol_lookback=cfg.vol_lookback,
            skew_lookback=cfg.skew_lookback,
        )
        if features.empty:
            return pd.Series(0.0, index=prices.index, dtype=float)

        X = features.values

        # --- Filtered probabilities via forward algorithm -----------------
        # hmmlearn's predict_proba gives P(S_t | r_1,...,r_T) (smoothed).
        # For a *causal* trading signal we want filtered probabilities.
        # We approximate filtered probs by running predict_proba on data up
        # to time t in a rolling fashion.  For computational efficiency we
        # use the full smoothed posteriors as a pragmatic approximation and
        # note that in live trading one would implement the forward pass
        # incrementally.
        regime_probs = self._filtered_probabilities(X)  # (T, n_regimes)

        raw_signal = np.zeros(len(features))
        position_scale = np.ones(len(features))

        bull_idx = labels["bull"]
        bear_idx = labels["bear"]
        sideways_idx = labels["sideways"]

        # --- Mean-reversion z-score for sideways regime -------------------
        rolling_mean = log_returns.rolling(cfg.mr_lookback).mean()
        rolling_std = log_returns.rolling(cfg.mr_lookback).std()
        mr_zscore = (log_returns - rolling_mean) / rolling_std.replace(0, np.nan)
        mr_zscore = mr_zscore.reindex(features.index).fillna(0.0).values

        for t in range(len(features)):
            probs = regime_probs[t]
            dominant_regime = int(np.argmax(probs))
            prob_dominant = probs[dominant_regime]

            # --- Directional signal from regime --------------------------
            if dominant_regime == bull_idx:
                raw_signal[t] = 1.0
            elif dominant_regime == bear_idx:
                raw_signal[t] = -1.0 if cfg.short_allowed else 0.0
            else:
                # Sideways: mean-reversion sub-strategy
                # Fade extreme z-scores
                z = mr_zscore[t]
                raw_signal[t] = float(np.clip(-z, -1.0, 1.0))

            # --- Scale by regime probability -----------------------------
            raw_signal[t] *= prob_dominant

            # --- Entropy-based uncertainty reduction ----------------------
            h = float(scipy_entropy(probs))  # natural log (nats)
            if h > cfg.entropy_threshold:
                # Linear reduction: at max entropy the scale -> 0
                max_h = np.log(cfg.n_regimes)
                scale = max(0.0, 1.0 - (h - cfg.entropy_threshold) / (max_h - cfg.entropy_threshold))
                position_scale[t] = scale

            # --- Minimum-probability filter ------------------------------
            if prob_dominant < cfg.min_probability:
                position_scale[t] = 0.0

        # --- Volatility targeting overlay ---------------------------------
        realised_vol = (
            log_returns
            .rolling(cfg.realised_vol_lookback)
            .std()
            .reindex(features.index)
            .bfill()
        )
        annualised_vol = realised_vol * np.sqrt(cfg.annualisation_factor)
        vol_scale = cfg.target_annual_vol / annualised_vol.replace(0, np.nan)
        vol_scale = vol_scale.fillna(1.0).clip(upper=2.0).values  # cap leverage at 2x

        final_signal = raw_signal * position_scale * vol_scale

        return pd.Series(final_signal, index=features.index, name=prices.name)

    def _filtered_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Compute (approximate) filtered state probabilities.

        For causal correctness we implement the forward algorithm directly,
        producing P(S_t | r_1, ..., r_t) at each time step rather than the
        smoothed posteriors returned by ``hmmlearn``'s ``predict_proba``.

        Parameters
        ----------
        X : np.ndarray, shape (T, n_features)
            Observation sequence.

        Returns
        -------
        np.ndarray, shape (T, n_components)
            Filtered probabilities (each row sums to 1).
        """
        model = self._model
        n_states = model.n_components
        T = X.shape[0]

        try:
            # Log start probabilities
            log_startprob = np.log(model.startprob_ + 1e-300)

            # Log transition matrix
            log_transmat = np.log(model.transmat_ + 1e-300)

            # Compute log emission probabilities for all observations
            # hmmlearn exposes _compute_log_likelihood for this purpose
            log_lik = model._compute_log_likelihood(X)  # (T, n_states)

            filtered = np.zeros((T, n_states))

            # t = 0: alpha_0(j) = pi_j * b_j(x_0)
            log_alpha = log_startprob + log_lik[0]
            log_norm = _logsumexp(log_alpha)
            filtered[0] = np.exp(log_alpha - log_norm)

            for t in range(1, T):
                # Prediction step: sum_i alpha_{t-1}(i) * a_{i,j}
                log_alpha_pred = np.zeros(n_states)
                for j in range(n_states):
                    log_alpha_pred[j] = _logsumexp(log_alpha + log_transmat[:, j])

                # Update step: alpha_t(j) = b_j(x_t) * pred_j
                log_alpha = log_alpha_pred + log_lik[t]
                log_norm = _logsumexp(log_alpha)
                filtered[t] = np.exp(log_alpha - log_norm)

            return filtered

        except Exception:
            logger.warning(
                "Forward algorithm via _compute_log_likelihood failed; "
                "falling back to smoothed predict_proba.",
                exc_info=True,
            )
            try:
                return model.predict_proba(X)
            except Exception:
                logger.warning(
                    "predict_proba also failed; returning uniform probabilities.",
                    exc_info=True,
                )
                return np.full((T, n_states), 1.0 / n_states)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def decode(self, data: pd.DataFrame) -> pd.Series:
        """Run the Viterbi algorithm and return the most-likely state sequence.

        Parameters
        ----------
        data : pd.DataFrame
            Price data for a single asset.

        Returns
        -------
        pd.Series
            Integer-coded state sequence aligned to the valid feature dates.
        """
        if not self._is_fitted:
            raise RuntimeError("Strategy has not been fitted.")

        cfg = self.config
        prices = self._to_series(data)
        log_returns = np.log(prices / prices.shift(1)).dropna()
        features = _build_features(
            log_returns,
            vol_lookback=cfg.vol_lookback,
            skew_lookback=cfg.skew_lookback,
        )
        X = features.values
        _, state_seq = self._model.decode(X, algorithm="viterbi")
        return pd.Series(state_seq, index=features.index, name="regime")

    def regime_summary(self) -> dict[str, Any]:
        """Return a summary of estimated regime parameters.

        Returns
        -------
        dict
            Includes means, covariances, transition matrix, and regime labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Strategy has not been fitted.")

        model = self._model
        labels = self._regime_labels
        inv_labels = {v: k for k, v in labels.items()}

        summary: dict[str, Any] = {
            "transition_matrix": pd.DataFrame(
                model.transmat_,
                index=[inv_labels[i] for i in range(model.n_components)],
                columns=[inv_labels[i] for i in range(model.n_components)],
            ),
            "regime_labels": labels,
        }

        for name, idx in labels.items():
            summary[f"{name}_mean_return"] = float(model.means_[idx, 0])
            summary[f"{name}_vol"] = float(np.sqrt(model.covars_[idx][0, 0]))

        summary["stationary_distribution"] = _stationary_distribution(
            model.transmat_
        )

        return summary


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------


def _regularize_covars(
    model: GaussianHMM,
    ridge: float = 1e-3,
) -> None:
    """Ensure all covariance matrices in the fitted model are positive-definite.

    After EM convergence on a small training window the estimated covariance
    matrices may be near-singular.  This function checks the minimum eigenvalue
    of each component's covariance and, if it is below *ridge*, adds a diagonal
    ridge so that the matrix is safely positive-definite.

    Modifies ``model.covars_`` in-place.
    """
    cov_type = model.covariance_type
    covars = model.covars_

    if cov_type == "full":
        # covars_ shape: (n_components, n_features, n_features)
        for k in range(model.n_components):
            C = covars[k]
            eigvals = np.linalg.eigvalsh(C)
            min_eig = float(eigvals.min())
            if min_eig < ridge:
                correction = ridge - min_eig
                covars[k] += correction * np.eye(C.shape[0])
                logger.debug(
                    "Regularised covariance for state %d: added ridge %.2e "
                    "(min eigenvalue was %.2e).",
                    k,
                    correction,
                    min_eig,
                )
        model.covars_ = covars

    elif cov_type == "diag":
        # covars_ shape: (n_components, n_features)
        covars = np.maximum(covars, ridge)
        model.covars_ = covars

    elif cov_type == "spherical":
        # covars_ shape: (n_components,)
        covars = np.maximum(covars, ridge)
        model.covars_ = covars

    elif cov_type == "tied":
        # covars_ shape: (n_features, n_features)
        eigvals = np.linalg.eigvalsh(covars)
        min_eig = float(eigvals.min())
        if min_eig < ridge:
            correction = ridge - min_eig
            covars += correction * np.eye(covars.shape[0])
            model.covars_ = covars
            logger.debug(
                "Regularised tied covariance: added ridge %.2e "
                "(min eigenvalue was %.2e).",
                correction,
                min_eig,
            )


def _logsumexp(log_x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = log_x.max()
    return float(c + np.log(np.sum(np.exp(log_x - c))))


def _stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of a transition matrix.

    Solves  pi @ A = pi  subject to  sum(pi) = 1  via the left-eigenvector
    corresponding to eigenvalue 1.
    """
    eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
    # Find the eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()
    return pi
