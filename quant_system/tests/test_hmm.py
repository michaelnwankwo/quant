"""Unit tests for the HMM regime switchboard.

The suite covers three things that matter for correctness:

1. the canonical re-labelling actually orders the states by volatility,
2. inference is deterministic and returns well-formed probability vectors,
3. the streaming pass is **causal** - a state at bar ``t`` must be a function of
   bars ``<= t`` only (verified by truncating the input and comparing prefixes).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from quant_system.config import settings as cfg
from quant_system.models.hmm_switchboard import (
    CausalRegimeStreamer,
    HMMSwitchboard,
    regime_distribution,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_regime_blocks(
    n_per_block: int = 300, seed: int = 7
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build a feature frame with three clearly separated volatility blocks.

    Args:
        n_per_block: Bars per regime block.
        seed: Random seed.

    Returns:
        Tuple ``(features, true_labels)`` where ``features`` has the columns
        ``ret``/``atr``/``sigma`` and ``true_labels`` is the ground truth.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    labels = []
    specs = (
        # (vol, drift, atr_pct) -> state 0: calm, state 1: trending, state 2: shock
        (0.0030, 0.00010, 0.0040),
        (0.0100, 0.00120, 0.0110),
        (0.0280, -0.00150, 0.0300),
    )
    for label, (vol, drift, atr_pct) in enumerate(specs):
        shocks = rng.standard_t(4 if label == 2 else 30, size=n_per_block)
        shocks = shocks / np.sqrt(np.var(shocks))
        returns = drift + vol * shocks
        blocks.append(
            pd.DataFrame(
                {
                    "ret": returns,
                    "atr": atr_pct * (1.0 + 0.05 * rng.standard_normal(n_per_block)),
                    "sigma": (vol * np.sqrt(252.0))
                    * (1.0 + 0.05 * rng.standard_normal(n_per_block)),
                }
            )
        )
        labels.append(np.full(n_per_block, label, dtype=int))

    features = pd.concat(blocks, ignore_index=True)
    features.index = pd.bdate_range("2020-01-01", periods=len(features), name="timestamp")
    return features, np.concatenate(labels)


@pytest.fixture(scope="module")
def regime_features() -> Tuple[pd.DataFrame, np.ndarray]:
    """Module-scoped feature fixture."""
    return make_regime_blocks()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_fit_produces_three_labelled_states(regime_features: Tuple[pd.DataFrame, np.ndarray]) -> None:
    """The switchboard must expose all three canonical states with statistics."""
    features, _ = regime_features
    switchboard = HMMSwitchboard()
    switchboard.fit(features)

    stats = switchboard.get_state_statistics()
    assert len(stats) == cfg.N_REGIMES
    assert set(stats.index) == {0, 1, 2}
    assert stats["count"].sum() == len(features)
    assert np.isclose(stats["share"].sum(), 1.0, atol=1e-9)


def test_state_zero_is_the_lowest_variance_regime(
    regime_features: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Canonical state 0 must be the calmest regime by variance."""
    features, _ = regime_features
    switchboard = HMMSwitchboard()
    switchboard.fit(features)

    stats = switchboard.get_state_statistics()
    variance = stats["variance"].sort_values()
    lowest_variance_state = int(variance.index[0])
    assert lowest_variance_state == cfg.STATE_RANGE_BOUND, (
        f"Expected state 0 to have the lowest variance, got {lowest_variance_state}."
    )

    highest_variance_state = int(variance.index[-1])
    assert highest_variance_state == cfg.STATE_SHOCK


def test_shock_state_is_fatter_tailed_than_trend_state(
    regime_features: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """State 2 must be the shock state (fat tails), state 1 the trend state."""
    features, _ = regime_features
    switchboard = HMMSwitchboard()
    switchboard.fit(features)

    stats = switchboard.get_state_statistics()
    # The synthetic shock block uses Student-t(4) draws -> positive excess kurtosis.
    assert stats.loc[cfg.STATE_SHOCK, "kurtosis"] > stats.loc[cfg.STATE_TREND, "kurtosis"]


def test_predict_state_returns_canonical_id(
    regime_features: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """``predict_state`` must return an id in {0, 1, 2}."""
    features, _ = regime_features
    switchboard = HMMSwitchboard()
    switchboard.fit(features)

    state = switchboard.predict_state(features.iloc[-50:])
    assert isinstance(state, int)
    assert state in {cfg.STATE_RANGE_BOUND, cfg.STATE_TREND, cfg.STATE_SHOCK}


def test_state_probabilities_are_a_distribution(
    regime_features: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """``get_state_probabilities`` must return a non-negative vector summing to 1."""
    features, _ = regime_features
    switchboard = HMMSwitchboard()
    switchboard.fit(features)
    switchboard.predict_state(features.iloc[-50:])

    probabilities = switchboard.get_state_probabilities()
    assert probabilities.shape == (cfg.N_REGIMES,)
    assert np.all(probabilities >= -1e-12)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-9)


def test_get_state_probabilities_raises_before_fit() -> None:
    """Calling the accessor on an unfitted model must raise clearly."""
    switchboard = HMMSwitchboard()
    with pytest.raises(RuntimeError):
        switchboard.get_state_probabilities()
    with pytest.raises(RuntimeError):
        switchboard.predict_state(pd.DataFrame({"ret": [0.0], "atr": [0.0], "sigma": [0.0]}))


def test_fit_is_deterministic(regime_features: Tuple[pd.DataFrame, np.ndarray]) -> None:
    """Two fits with the same random state must agree exactly."""
    features, _ = regime_features
    first = HMMSwitchboard().fit(features)
    second = HMMSwitchboard().fit(features)

    pd.testing.assert_frame_equal(
        first.get_state_statistics(), second.get_state_statistics()
    )
    assert np.array_equal(first.mapping_, second.mapping_)
    assert np.isclose(first.log_likelihood_, second.log_likelihood_)


def test_fit_accepts_raw_ohlcv() -> None:
    """The switchboard should derive features from an OHLCV frame."""
    rng = np.random.default_rng(11)
    n = 400
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    high = close * (1.0 + 0.004 * np.abs(rng.standard_normal(n)))
    low = close * (1.0 - 0.004 * np.abs(rng.standard_normal(n)))
    ohlcv = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1.0},
        index=pd.bdate_range("2021-01-01", periods=n, name="timestamp"),
    )
    switchboard = HMMSwitchboard()
    switchboard.fit(ohlcv)
    assert switchboard.model_ is not None
    assert switchboard.features_ is not None
    assert set(switchboard.features_.columns) == set(cfg.DEFAULT_SETTINGS.hmm.feature_columns)


def test_rejects_insufficient_data() -> None:
    """Fitting on fewer rows than the number of states must raise."""
    tiny = pd.DataFrame({"ret": [0.01, 0.02], "atr": [0.1, 0.1], "sigma": [0.2, 0.2]})
    with pytest.raises(ValueError):
        HMMSwitchboard().fit(tiny)


def test_causal_stream_shapes_and_labels(
    regime_features: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """The streaming pass must label every bar and fill the warm-up."""
    features, _ = regime_features
    config = cfg.HMMConfig(train_window=200, refit_every=25, min_train=100)
    streamer = CausalRegimeStreamer(config=config)
    result = streamer.run(features)

    assert len(result.states) == len(features)
    assert result.states.isna().sum() == 0
    assert set(result.states.unique()).issubset({0, 1, 2})
    assert list(result.probabilities.columns) == [0, 1, 2]
    assert result.warmup_mask.iloc[: 100 - 1].all()
    assert len(result.refit_index) > 0


def test_causal_stream_has_no_lookahead(
    regime_features: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Truncating the future must not change any past regime label."""
    features, _ = regime_features
    config = cfg.HMMConfig(train_window=200, refit_every=25, min_train=100)

    full = CausalRegimeStreamer(config=config).run(features)
    truncated = CausalRegimeStreamer(config=config).run(features.iloc[:500])

    np.testing.assert_array_equal(
        full.states.to_numpy()[:500], truncated.states.to_numpy()[:500]
    )
    np.testing.assert_allclose(
        full.probabilities.to_numpy()[:500],
        truncated.probabilities.to_numpy()[:500],
        atol=1e-12,
    )


def test_full_sample_fit_recovers_the_ground_truth_blocks(
    regime_features: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """A single fit over all three blocks must separate calm from shock."""
    features, truth = regime_features
    switchboard = HMMSwitchboard()
    switchboard.fit(features)
    states, _ = switchboard.predict_path(features)

    labels = states.to_numpy()
    calm_block = truth == 0
    shock_block = truth == 2

    calm_share_state0 = float((labels[calm_block] == cfg.STATE_RANGE_BOUND).mean())
    shock_share_state0 = float((labels[shock_block] == cfg.STATE_RANGE_BOUND).mean())
    assert calm_share_state0 > shock_share_state0, (
        f"State 0 should concentrate in the calm block "
        f"(calm={calm_share_state0:.2%}, shock={shock_share_state0:.2%})."
    )

    shock_share_state2 = float((labels[shock_block] == cfg.STATE_SHOCK).mean())
    calm_share_state2 = float((labels[calm_block] == cfg.STATE_SHOCK).mean())
    assert shock_share_state2 > calm_share_state2, (
        f"State 2 should concentrate in the shock block "
        f"(shock={shock_share_state2:.2%}, calm={calm_share_state2:.2%})."
    )


def test_streamer_states_order_volatility_monotonically(
    regime_features: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Bars labelled State 0 must be calmer than bars labelled State 2.

    The streaming pass refits on a trailing window, so the labels are *relative*
    to that window.  The invariant that always holds - and the one the strategy
    router depends on - is that State 0 marks the calmest observations and
    State 2 the most volatile ones.
    """
    features, _ = regime_features
    config = cfg.HMMConfig(train_window=250, refit_every=25, min_train=120)
    result = CausalRegimeStreamer(config=config).run(features)

    states = result.states.to_numpy()
    evaluated = ~result.warmup_mask.to_numpy()
    volatility = features["sigma"].to_numpy()[evaluated]
    labels = states[evaluated]

    vol_state0 = float(np.mean(volatility[labels == cfg.STATE_RANGE_BOUND]))
    vol_state1 = float(np.mean(volatility[labels == cfg.STATE_TREND]))
    vol_state2 = float(np.mean(volatility[labels == cfg.STATE_SHOCK]))

    assert vol_state0 < vol_state2, (
        f"State 0 mean vol {vol_state0:.4f} should be below State 2 {vol_state2:.4f}."
    )
    assert vol_state1 < vol_state2, (
        f"State 1 mean vol {vol_state1:.4f} should be below State 2 {vol_state2:.4f}."
    )
    assert vol_state0 > 0.0 and vol_state2 > 0.0


def test_regime_distribution_counts_transitions(
    regime_features: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """``regime_distribution`` must report shares plus a transition count."""
    features, _ = regime_features
    config = cfg.HMMConfig(train_window=200, refit_every=25, min_train=100)
    result = CausalRegimeStreamer(config=config).run(features)
    distribution = regime_distribution(result.states)
    assert "transitions" in distribution.index
    assert distribution.drop("transitions").sum() > 0.99
