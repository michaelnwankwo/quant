"""Three-state Gaussian-HMM Regime Switchboard.

The switchboard is the *routing brain* of the system: it maps the market's
current statistical character onto one of three canonical regimes, and the
strategy layer activates/deactivates itself accordingly.

Feature vector
--------------
``X_t = [r_t, ATR_t, sigma_t]^T`` where

* ``r_t``     - equal-weighted cross-sectional log return (market composite),
* ``ATR_t``   - equal-weighted ATR expressed as a fraction of price, so gold and
  FX crosses are dimensionally comparable,
* ``sigma_t`` - equal-weighted annualised realised volatility (N = 20).

Canonical regime mapping
------------------------
``hmmlearn`` emits arbitrary hidden-state ids, so after every fit the states are
re-labelled onto a stable, economically meaningful ordering:

=====  ==============================  =========================================
State  Label                           Identification rule
=====  ==============================  =========================================
0      Low-Volatility / Range-Bound    Lowest return variance.
1      High-Momentum / Strong Trend    Mid-to-high variance, highest directional
                                       persistence (|mean| / sd) among the
                                       remaining states.
2      High-Volatility / Market Shock  Highest variance combined with the
                                       fattest tails (excess kurtosis).
=====  ==============================  =========================================

Look-ahead control
------------------
:class:`CausalRegimeStreamer` produces the regime path used by the backtester.
At every bar ``t`` it fits (or reuses) a model on a trailing window that ends at
``t`` and decodes the state at ``t``.  No future observation ever enters a
fit or a decode.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

# --- Gaussian-HMM backend ----------------------------------------------- #
# ``hmmlearn`` is preferred when importable, but it compiles a Cython
# extension and publishes Windows wheels only for CPython 3.8-3.13
# (win_amd64).  On newer interpreters pip falls back to the source tarball
# and the install fails without the MSVC toolchain, so we fall back to the
# pure NumPy/SciPy implementation in ``_hmm_fallback`` (same API surface).
try:  # pragma: no cover - depends on what the environment has installed
    from hmmlearn.hmm import GaussianHMM

    HMM_BACKEND: str = "hmmlearn"
except ImportError:  # pragma: no cover - exercised on wheel-less interpreters
    from quant_system.models._hmm_fallback import GaussianHMM

    HMM_BACKEND = "builtin-numpy"
# ------------------------------------------------------------------------ #

from quant_system.config import settings as cfg
from quant_system.data.preprocessing import (
    build_feature_frame,
    standardize_features,
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeStreamResult:
    """Output of the causal streaming regime pass.

    Attributes:
        states: Integer regime series (``0``/``1``/``2``) aligned to the input
            feature index. Bars before ``min_train`` are filled with
            ``STATE_RANGE_BOUND`` and flagged in ``warmup_mask``.
        probabilities: DataFrame with columns ``0``, ``1``, ``2`` holding the
            canonical state probability distribution at each bar.
        warmup_mask: Boolean Series; ``True`` where the label is a warm-up
            placeholder rather than a model output.
        refit_index: Timestamps at which the model was refitted.
        state_statistics: Per-regime descriptive statistics of the final fit.
    """

    states: pd.Series
    probabilities: pd.DataFrame
    warmup_mask: pd.Series
    refit_index: List[pd.Timestamp] = field(default_factory=list)
    state_statistics: Optional[pd.DataFrame] = None

    def as_frame(self) -> pd.DataFrame:
        """Return states and probabilities as a single tidy DataFrame.

        Returns:
            DataFrame with columns ``state``, ``p_state_0``, ``p_state_1``,
            ``p_state_2``, ``warmup``.
        """
        out = pd.DataFrame(
            {
                "state": self.states.astype(int),
                "p_state_0": self.probabilities[cfg.STATE_RANGE_BOUND],
                "p_state_1": self.probabilities[cfg.STATE_TREND],
                "p_state_2": self.probabilities[cfg.STATE_SHOCK],
                "warmup": self.warmup_mask,
            }
        )
        return out


class HMMSwitchboard:
    """Three-state Gaussian HMM with canonical regime re-labelling.

    Attributes:
        config: HMM hyper-parameters.
        model_: Fitted Gaussian-HMM model (``None`` before fit). The
            concrete class depends on :data:`HMM_BACKEND`.
        scaler_: Fitted :class:`StandardScaler` for the feature matrix.
        mapping_: Array where ``mapping_[hidden_id] = canonical_state_id``.
        state_statistics_: Descriptive statistics per canonical state.
        last_probabilities_: Latest canonical probability vector.
    """

    def __init__(self, config: Optional[cfg.HMMConfig] = None) -> None:
        """Initialise the switchboard.

        Args:
            config: HMM configuration; defaults to ``settings.hmm``.
        """
        self.config: cfg.HMMConfig = config or cfg.DEFAULT_SETTINGS.hmm
        self.model_: Optional[GaussianHMM] = None
        self.scaler_ = None
        self.mapping_: Optional[np.ndarray] = None
        self.state_statistics_: Optional[pd.DataFrame] = None
        self.last_probabilities_: Optional[np.ndarray] = None
        self.features_: Optional[pd.DataFrame] = None
        self.log_likelihood_: float = float("-inf")

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, data: pd.DataFrame) -> "HMMSwitchboard":
        """Fit the HMM on a historical window.

        Args:
            data: Either a feature frame with the columns named in
                ``config.feature_columns`` (``ret``, ``atr``, ``sigma``) or a raw
                OHLCV frame with ``open/high/low/close/volume`` columns, in which
                case the features are derived internally.

        Returns:
            ``self`` (enables fluent chaining).

        Raises:
            ValueError: If the feature frame is empty or shorter than
                ``n_components``.
        """
        features = self._extract_features(data)
        if features.empty or len(features) < self.config.n_components:
            raise ValueError(
                "Insufficient data to fit the HMM "
                f"(got {len(features)} rows, need >= {self.config.n_components})."
            )

        scaled, scaler = standardize_features(features)
        values = scaled.to_numpy(dtype=float)

        best_model: Optional[GaussianHMM] = None
        best_score = float("-inf")
        for attempt in range(max(1, self.config.n_fits)):
            model = GaussianHMM(
                n_components=self.config.n_components,
                covariance_type=self.config.covariance_type,
                n_iter=self.config.n_iter,
                random_state=self.config.random_state + attempt,
            )
            try:
                model.fit(values)
                score = float(model.score(values))
            except Exception as exc:  # pragma: no cover - numerical edge cases
                logger.debug("HMM fit attempt %d failed: %s", attempt, exc)
                continue
            if np.isfinite(score) and score > best_score:
                best_score, best_model = score, model

        if best_model is None:
            raise RuntimeError("GaussianHMM failed to converge on the supplied window.")

        hidden = best_model.predict(values)
        self.model_ = best_model
        self.scaler_ = scaler
        self.features_ = features
        self.log_likelihood_ = best_score
        self.mapping_ = self._derive_mapping(features, hidden)
        self.state_statistics_ = self._summarise_states(features, hidden, self.mapping_)
        self.last_probabilities_ = self._canonical_probabilities(
            best_model.predict_proba(values)[-1], self.mapping_
        )
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict_state(self, latest_data: pd.DataFrame) -> int:
        """Predict the currently active canonical regime.

        Args:
            latest_data: Feature frame (or raw OHLCV frame) covering the recent
                past. Only the trailing window is used, but passing more history
                improves the Viterbi decode.

        Returns:
            Canonical regime id in ``{0, 1, 2}``.

        Raises:
            RuntimeError: If the switchboard has not been fitted.
        """
        if self.model_ is None or self.scaler_ is None or self.mapping_ is None:
            raise RuntimeError("HMMSwitchboard must be fitted before prediction.")
        features = self._extract_features(latest_data)
        values = self.scaler_.transform(features.to_numpy(dtype=float))
        hidden = self.model_.predict(values)
        state = int(self.mapping_[int(hidden[-1])])
        self.last_probabilities_ = self._canonical_probabilities(
            self.model_.predict_proba(values)[-1], self.mapping_
        )
        return state

    def predict_path(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Decode the canonical regime path for a whole feature frame.

        Args:
            data: Feature frame or raw OHLCV frame.

        Returns:
            Tuple of ``(states, probabilities)`` where ``states`` is an integer
            Series and ``probabilities`` is a DataFrame with columns ``0``/``1``/``2``.
        """
        if self.model_ is None or self.scaler_ is None or self.mapping_ is None:
            raise RuntimeError("HMMSwitchboard must be fitted before prediction.")
        features = self._extract_features(data)
        values = self.scaler_.transform(features.to_numpy(dtype=float))
        hidden = self.model_.predict(values)
        states = pd.Series(
            self.mapping_[hidden], index=features.index, name="state", dtype=int
        )
        probs = self.model_.predict_proba(values)
        canonical = np.zeros_like(probs)
        for hidden_id, canonical_id in enumerate(self.mapping_):
            canonical[:, canonical_id] = probs[:, hidden_id]
        probabilities = pd.DataFrame(
            canonical, index=features.index, columns=[0, 1, 2]
        )
        return states, probabilities

    def get_state_probabilities(self) -> np.ndarray:
        """Return the canonical state probability distribution.

        Returns:
            Array of shape ``(3,)`` ordered by canonical state id
            ``[P(state 0), P(state 1), P(state 2)]``.

        Raises:
            RuntimeError: If called before any fit/prediction.
        """
        if self.last_probabilities_ is None:
            raise RuntimeError("No state probabilities available; fit or predict first.")
        return self.last_probabilities_.copy()

    def get_state_statistics(self) -> pd.DataFrame:
        """Return descriptive statistics for each canonical state.

        Returns:
            DataFrame indexed by canonical state id with columns ``count``,
            ``mean_return``, ``variance``, ``volatility_annual``, ``kurtosis``
            and ``label``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.state_statistics_ is None:
            raise RuntimeError("Model is not fitted.")
        return self.state_statistics_.copy()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Coerce input into the canonical feature frame.

        Args:
            data: Feature frame or raw OHLCV frame.

        Returns:
            DataFrame restricted to ``config.feature_columns``.

        Raises:
            ValueError: If required columns are missing.
        """
        if data is None or len(data) == 0:
            raise ValueError("Empty data passed to HMMSwitchboard.")
        columns = set(data.columns)
        if set(self.config.feature_columns).issubset(columns):
            frame = data.loc[:, list(self.config.feature_columns)]
        elif {"open", "high", "low", "close"}.issubset(columns):
            frame = build_feature_frame(
                data,
                atr_period=self.config.atr_period,
                vol_window=self.config.vol_window,
            ).loc[:, list(self.config.feature_columns)]
        else:
            raise ValueError(
                "Input must expose either the HMM feature columns "
                f"{self.config.feature_columns} or OHLCV columns."
            )
        return frame.dropna().astype(float)

    @staticmethod
    def _canonical_probabilities(
        probabilities: np.ndarray, mapping: np.ndarray
    ) -> np.ndarray:
        """Reorder raw hidden-state probabilities into canonical order.

        Args:
            probabilities: Raw probability vector from hmmlearn.
            mapping: Array mapping hidden id -> canonical id.

        Returns:
            Canonical probability vector of length ``3``.
        """
        out = np.zeros(cfg.N_REGIMES, dtype=float)
        for hidden_id, canonical_id in enumerate(mapping):
            out[int(canonical_id)] = float(probabilities[hidden_id])
        total = out.sum()
        if total > 0:
            out = out / total
        return out

    def _derive_mapping(self, features: pd.DataFrame, hidden: np.ndarray) -> np.ndarray:
        """Map arbitrary hidden ids onto canonical regime ids.

        Args:
            features: Feature frame used for the fit.
            hidden: Hidden-state sequence produced by Viterbi decoding.

        Returns:
            Array of length ``n_components`` with ``mapping_[hidden] = canonical``.
        """
        n_states = int(self.config.n_components)
        returns = features["ret"].to_numpy(dtype=float)
        stats: List[Dict[str, float]] = []
        for state_id in range(n_states):
            mask = hidden == state_id
            sample = returns[mask]
            count = int(sample.size)
            if count < 2:
                stats.append(
                    {
                        "count": count,
                        "mean": 0.0,
                        "variance": float("inf") if count == 0 else 0.0,
                        "kurtosis": 0.0,
                        "persistence": 0.0,
                    }
                )
                continue
            variance = float(np.var(sample, ddof=1))
            mean = float(np.mean(sample))
            excess_kurtosis = (
                float(kurtosis(sample, fisher=True)) if count >= 4 else 0.0
            )
            sd = float(np.sqrt(max(variance, 1e-18)))
            stats.append(
                {
                    "count": count,
                    "mean": mean,
                    "variance": variance,
                    "kurtosis": excess_kurtosis,
                    "persistence": abs(mean) / sd,
                }
            )

        order = np.argsort([s["variance"] for s in stats])
        mapping = np.full(n_states, cfg.STATE_TREND, dtype=int)
        # State 0: lowest variance.
        mapping[order[0]] = cfg.STATE_RANGE_BOUND
        rest = list(order[1:])
        if not rest:
            return mapping
        if len(rest) == 1:
            mapping[rest[0]] = cfg.STATE_SHOCK
            return mapping

        # Highest-variance state is the shock candidate; the runner-up is the
        # trend candidate.  Confirm the split with tail behaviour: a shock state
        # is fatter-tailed, a trend state is more directionally persistent.
        shock_candidate = rest[-1]
        trend_candidate = rest[-2]
        if stats[shock_candidate]["kurtosis"] >= stats[trend_candidate]["kurtosis"]:
            mapping[shock_candidate] = cfg.STATE_SHOCK
            mapping[trend_candidate] = cfg.STATE_TREND
        else:
            # The high-variance state is *less* fat-tailed than the mid state:
            # assign the trend label to whichever is more persistent.
            if (
                stats[shock_candidate]["persistence"]
                >= stats[trend_candidate]["persistence"]
            ):
                mapping[shock_candidate] = cfg.STATE_TREND
                mapping[trend_candidate] = cfg.STATE_SHOCK
            else:
                mapping[shock_candidate] = cfg.STATE_SHOCK
                mapping[trend_candidate] = cfg.STATE_TREND
        for leftover in rest[:-2]:
            mapping[leftover] = cfg.STATE_TREND
        return mapping

    def _summarise_states(
        self,
        features: pd.DataFrame,
        hidden: np.ndarray,
        mapping: np.ndarray,
    ) -> pd.DataFrame:
        """Build the canonical per-state statistics table.

        Args:
            features: Feature frame used for the fit.
            hidden: Hidden-state sequence.
            mapping: Hidden -> canonical mapping.

        Returns:
            DataFrame indexed by canonical state id.
        """
        rows: List[Dict[str, float]] = []
        returns = features["ret"].to_numpy(dtype=float)
        sigma = features["sigma"].to_numpy(dtype=float)
        atr = features["atr"].to_numpy(dtype=float)
        for canonical in range(cfg.N_REGIMES):
            hidden_ids = np.where(mapping == canonical)[0]
            mask = np.isin(hidden, hidden_ids)
            sample = returns[mask]
            count = int(sample.size)
            variance = float(np.var(sample, ddof=1)) if count >= 2 else float("nan")
            rows.append(
                {
                    "state": canonical,
                    "label": cfg.REGIME_LABELS.get(canonical, "unknown"),
                    "count": count,
                    "share": count / max(len(hidden), 1),
                    "mean_return": float(np.mean(sample)) if count else float("nan"),
                    "variance": variance,
                    "volatility_annual": float(np.sqrt(variance * 252.0))
                    if np.isfinite(variance)
                    else float("nan"),
                    "kurtosis": float(kurtosis(sample, fisher=True))
                    if count >= 4
                    else float("nan"),
                    "mean_sigma": float(np.mean(sigma[mask])) if count else float("nan"),
                    "mean_atr_pct": float(np.mean(atr[mask])) if count else float("nan"),
                }
            )
        frame = pd.DataFrame(rows).set_index("state")
        return frame.sort_index()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> None:
        """Pickle the fitted switchboard to disk.

        Args:
            path: Destination file path.
        """
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Union[str, Path]) -> "HMMSwitchboard":
        """Load a pickled switchboard.

        Args:
            path: Source file path.

        Returns:
            The deserialised switchboard.
        """
        with Path(path).open("rb") as handle:
            return pickle.load(handle)  # type: ignore[no-any-return]


class CausalRegimeStreamer:
    """Produces a strictly causal regime path for backtesting / live use.

    The streamer walks forward bar by bar.  At each bar ``t``:

    1. A trailing window ``[t - train_window + 1, t]`` is sliced from the feature
       frame (never extending past ``t``).
    2. The HMM is refitted every ``refit_every`` bars, otherwise the most recent
       model is reused.
    3. Viterbi decoding yields the state at ``t``; probabilities are recorded too.

    Bars before ``min_train`` are labelled ``STATE_RANGE_BOUND`` and flagged as
    warm-up so downstream code can exclude them from statistics.

    Attributes:
        config: HMM configuration.
        train_window: Trailing bars used per fit.
        refit_every: Refit cadence in bars.
        min_train: Bars required before the first real label.
        verbose: Emit progress logging.
    """

    def __init__(
        self,
        config: Optional[cfg.HMMConfig] = None,
        train_window: Optional[int] = None,
        refit_every: Optional[int] = None,
        min_train: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Initialise the streamer.

        Args:
            config: HMM configuration; defaults to ``settings.hmm``.
            train_window: Override for the rolling training window.
            refit_every: Override for the refit cadence.
            min_train: Override for the warm-up length.
            verbose: Enable progress logging.
        """
        self.config: cfg.HMMConfig = config or cfg.DEFAULT_SETTINGS.hmm
        self.train_window: int = int(train_window or self.config.train_window)
        self.refit_every: int = int(refit_every or self.config.refit_every)
        self.min_train: int = int(min_train or self.config.min_train)
        self.verbose: bool = verbose
        self.switchboard: HMMSwitchboard = HMMSwitchboard(self.config)

    def run(self, market_features: pd.DataFrame) -> RegimeStreamResult:
        """Compute the causal regime path.

        Args:
            market_features: Feature frame with ``ret``, ``atr``, ``sigma``
                columns indexed by timestamp.

        Returns:
            A :class:`RegimeStreamResult`.

        Raises:
            ValueError: If the feature frame is empty.
        """
        if market_features is None or market_features.empty:
            raise ValueError("Cannot stream regimes from an empty feature frame.")
        features = market_features.loc[:, list(self.config.feature_columns)].dropna()
        n = len(features)
        warmup_bars = min(self.min_train, n)

        states = np.full(n, cfg.STATE_RANGE_BOUND, dtype=int)
        probs = np.zeros((n, cfg.N_REGIMES), dtype=float)
        probs[:, cfg.STATE_RANGE_BOUND] = 1.0
        warmup = np.ones(n, dtype=bool)
        refit_index: List[pd.Timestamp] = []

        fitted = False
        start = warmup_bars - 1
        for t in range(start, n):
            window_start = max(0, t - self.train_window + 1)
            window = features.iloc[window_start : t + 1]

            needs_fit = (not fitted) or ((t - start) % max(1, self.refit_every) == 0)
            if needs_fit:
                try:
                    self.switchboard.fit(window)
                    fitted = True
                    refit_index.append(pd.Timestamp(features.index[t]))
                    if self.verbose:
                        logger.info(
                            "Refit HMM at %s (window=%d bars, ll=%.2f)",
                            features.index[t],
                            len(window),
                            self.switchboard.log_likelihood_,
                        )
                except Exception as exc:
                    logger.debug("HMM refit failed at bar %d: %s", t, exc)
                    if not fitted:
                        continue
            if not fitted:
                continue

            try:
                state = self.switchboard.predict_state(window)
                probabilities = self.switchboard.get_state_probabilities()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("HMM predict failed at bar %d: %s", t, exc)
                continue

            states[t] = state
            probs[t, :] = probabilities
            warmup[t] = False

        index = features.index
        return RegimeStreamResult(
            states=pd.Series(states, index=index, name="state"),
            probabilities=pd.DataFrame(
                probs, index=index, columns=[0, 1, 2]
            ),
            warmup_mask=pd.Series(warmup, index=index, name="warmup"),
            refit_index=refit_index,
            state_statistics=self.switchboard.state_statistics_,
        )

    def align_to(
        self, result: RegimeStreamResult, target_index: pd.DatetimeIndex
    ) -> RegimeStreamResult:
        """Reindex a stream result onto another calendar (forward-filled).

        Args:
            result: Result produced by :meth:`run`.
            target_index: Destination index.

        Returns:
            New :class:`RegimeStreamResult` aligned to ``target_index``.
        """
        states = (
            result.states.reindex(target_index)
            .ffill()
            .infer_objects(copy=False)
            .fillna(cfg.STATE_RANGE_BOUND)
        )
        probs = result.probabilities.reindex(target_index).ffill().fillna(0.0)
        warmup = (
            result.warmup_mask.reindex(target_index, fill_value=True)
            .infer_objects(copy=False)
            .astype(bool)
        )
        return RegimeStreamResult(
            states=states.astype(int),
            probabilities=probs,
            warmup_mask=warmup,
            refit_index=result.refit_index,
            state_statistics=result.state_statistics,
        )


def regime_distribution(states: pd.Series) -> pd.Series:
    """Compute the empirical share of bars spent in each regime.

    Args:
        states: Integer regime series.

    Returns:
        Series indexed by regime id with the fraction of bars, plus a
        ``"transitions"`` entry counting regime switches.
    """
    counts = states.value_counts(normalize=True).sort_index()
    counts.name = "share"
    transitions = int((states.diff().fillna(0) != 0).sum())
    counts.loc["transitions"] = transitions
    return counts


__all__: List[str] = [
    "HMMSwitchboard",
    "CausalRegimeStreamer",
    "RegimeStreamResult",
    "regime_distribution",
]
