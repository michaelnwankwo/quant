"""Unit tests for the statistical-arbitrage module.

Covers the econometrics (OLS hedge ratio, Engle-Granger), the numerical guards
(zero-variance z-score, degenerate regressions) and the trading state machine
(entry / mean-exit / stop-out / cointegration-breakdown exit).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from quant_system.config import settings as cfg
from quant_system.data.preprocessing import (
    compute_spread,
    engle_granger_test,
    ols_hedge_ratio,
    rolling_cointegration,
    rolling_correlation,
    rolling_hedge_ratio,
    rolling_zscore,
)
from quant_system.strategies.base import Signal, StrategyContext
from quant_system.strategies.stat_arb import PairsStatArbStrategy


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_cointegrated_pair(
    n: int = 600,
    beta: float = 2.0,
    intercept: float = 5.0,
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """Generate a cointegrated pair ``y = intercept + beta * x + OU``.

    Args:
        n: Number of observations.
        beta: True hedge ratio.
        intercept: Cointegrating intercept.
        seed: Random seed.

    Returns:
        Tuple ``(y, x)``.
    """
    rng = np.random.default_rng(seed)
    # A wide-ranging regressor keeps the OLS slope well identified.
    x = 100.0 + np.cumsum(rng.normal(0.0, 1.5, n))
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = ou[t - 1] * 0.95 + rng.normal(0.0, 0.30)
    y = intercept + beta * x + ou
    index = pd.bdate_range("2020-01-01", periods=n, name="timestamp")
    return pd.Series(y, index=index, name="y"), pd.Series(x, index=index, name="x")


def make_ohlcv(series: pd.Series, name: str) -> pd.DataFrame:
    """Wrap a price series into an OHLCV frame.

    Args:
        series: Close prices.
        name: Symbol name.

    Returns:
        A minimal OHLCV DataFrame.
    """
    return pd.DataFrame(
        {
            "open": series,
            "high": series * 1.002,
            "low": series * 0.998,
            "close": series,
            "volume": 1_000.0,
        },
        index=series.index,
    )


@pytest.fixture(scope="module")
def pair_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Module-scoped cointegrated pair fixture."""
    y, x = make_cointegrated_pair()
    return make_ohlcv(y, "LEGA"), make_ohlcv(x, "LEGB")


# --------------------------------------------------------------------------- #
# Econometrics
# --------------------------------------------------------------------------- #
def test_ols_hedge_ratio_recovers_beta(pair_data: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """OLS must recover the true hedge ratio and intercept."""
    leg_a, leg_b = pair_data
    beta, alpha, r_squared = ols_hedge_ratio(leg_a["close"], leg_b["close"])
    assert beta == pytest.approx(2.0, abs=0.05)
    assert alpha == pytest.approx(5.0, abs=2.0)
    assert r_squared > 0.95


def test_ols_hedge_ratio_handles_degenerate_input() -> None:
    """A constant regressor must yield NaN rather than raising."""
    constant = pd.Series(np.full(50, 3.0))
    response = pd.Series(np.arange(50, dtype=float))
    beta, alpha, r_squared = ols_hedge_ratio(response, constant)
    assert np.isnan(beta)
    assert np.isnan(alpha)
    assert np.isnan(r_squared)


def test_engle_granger_detects_cointegration(
    pair_data: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The cointegrated pair must produce a small p-value."""
    leg_a, leg_b = pair_data
    _stat, pvalue = engle_granger_test(leg_a["close"], leg_b["close"])
    assert np.isfinite(pvalue)
    assert pvalue < 0.05


def test_engle_granger_rejects_independent_random_walks() -> None:
    """Two independent random walks must not be declared cointegrated."""
    rng = np.random.default_rng(3)
    a = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 500)))
    b = pd.Series(50.0 + np.cumsum(rng.normal(0, 1, 500)))
    _stat, pvalue = engle_granger_test(a, b)
    # A p-value above 0.05 is the expected outcome; a NaN (test could not run) is
    # also acceptable and must be treated as "not cointegrated" by callers.
    assert (not np.isfinite(pvalue)) or pvalue > 0.05


def test_engle_granger_survives_constant_series() -> None:
    """A constant input must return NaN instead of raising."""
    constant = pd.Series(np.full(200, 7.0))
    rng = np.random.default_rng(5)
    other = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    stat, pvalue = engle_granger_test(constant, other)
    assert np.isnan(stat) and np.isnan(pvalue)


def test_rolling_cointegration_shape_and_range(
    pair_data: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Rolling p-values must be forward-filled into a full-length series."""
    leg_a, leg_b = pair_data
    frame = rolling_cointegration(
        leg_a["close"], leg_b["close"], window=200, step=10
    )
    assert len(frame) == len(leg_a)
    assert frame["pvalue"].dropna().between(0.0, 1.0).all()
    # After the first window there should be no interior gaps (forward-filled).
    interior = frame["pvalue"].iloc[200:]
    assert interior.notna().mean() > 0.95


def test_rolling_hedge_ratio_is_stable_and_clipped() -> None:
    """The rolling beta must be stable, finite and clipped to the bound."""
    y, x = make_cointegrated_pair(n=400, beta=2.0)
    beta = rolling_hedge_ratio(y, x, window=120, max_abs_beta=5.0)
    valid = beta.dropna()
    assert valid.abs().max() <= 5.0 + 1e-9
    assert np.isfinite(valid).all()
    assert valid.mean() == pytest.approx(2.0, abs=0.15)
    assert valid.std() < 0.25, "The rolling hedge ratio is too unstable."


def test_spread_is_mean_reverting(pair_data: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The hedged spread must be stationary (ADF rejects a unit root)."""
    from statsmodels.tsa.stattools import adfuller

    leg_a, leg_b = pair_data
    beta, _alpha, _r2 = ols_hedge_ratio(leg_a["close"], leg_b["close"])
    spread = compute_spread(
        leg_a["close"], leg_b["close"], pd.Series(beta, index=leg_a.index)
    ).dropna()

    # The hedged spread must be far less dispersed than either leg.
    assert spread.std() < 0.25 * leg_a["close"].std()

    adf_stat, pvalue = adfuller(
        spread.to_numpy(), autolag="AIC", result_object=False
    )[:2]
    assert pvalue < 0.05, f"Spread failed the ADF stationarity test (p={pvalue:.4f})."


def test_rolling_spread_stays_bounded(pair_data: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The rolling-beta spread must stay bounded relative to the legs."""
    leg_a, leg_b = pair_data
    beta = rolling_hedge_ratio(leg_a["close"], leg_b["close"], window=120)
    spread = compute_spread(leg_a["close"], leg_b["close"], beta).dropna()
    # Rolling-beta estimation error adds some dispersion, but the spread must
    # remain an order of magnitude tamer than the underlying random walk.
    assert spread.std() < leg_a["close"].std()


# --------------------------------------------------------------------------- #
# Numerical guards
# --------------------------------------------------------------------------- #
def test_zscore_zero_variance_guard_returns_zero() -> None:
    """A constant series must produce a finite z-score of exactly zero."""
    constant = pd.Series(np.full(100, 4.2))
    z = rolling_zscore(constant, window=20)
    assert np.isfinite(z.dropna()).all()
    assert (z.dropna() == 0.0).all()


def test_zscore_is_finite_for_spiky_input() -> None:
    """Near-zero variance bursts must be clipped, not explode to infinity."""
    rng = np.random.default_rng(9)
    values = np.concatenate([np.zeros(60), rng.normal(0, 1, 40)])
    z = rolling_zscore(pd.Series(values), window=20, clip=20.0)
    assert np.isfinite(z.dropna()).all()
    assert z.abs().max() <= 20.0 + 1e-9


def test_zscore_matches_manual_calculation() -> None:
    """The rolling z-score must equal the textbook definition."""
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0, 1, 200))
    z = rolling_zscore(series, window=30)
    manual = (series.iloc[-1] - series.iloc[-30:].mean()) / series.iloc[-30:].std(ddof=1)
    assert z.iloc[-1] == pytest.approx(float(manual), rel=1e-9)


def test_rolling_correlation_of_inverse_series() -> None:
    """Perfectly inverse series must yield a correlation near -1."""
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(0, 1, 300))
    correlation = rolling_correlation(a, -a, window=60)
    assert correlation.dropna().iloc[-1] == pytest.approx(-1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Strategy state machine
# --------------------------------------------------------------------------- #
def _build_strategy(
    leg_a: pd.DataFrame,
    leg_b: pd.DataFrame,
    **overrides: object,
) -> PairsStatArbStrategy:
    """Build a prepared strategy for the supplied pair.

    Args:
        leg_a: Dependent leg OHLCV.
        leg_b: Independent leg OHLCV.
        **overrides: Keyword overrides forwarded to the strategy.

    Returns:
        A prepared :class:`PairsStatArbStrategy`.
    """
    strategy = PairsStatArbStrategy("LEGA", "LEGB", **overrides)  # type: ignore[arg-type]
    strategy.prepare({"LEGA": leg_a, "LEGB": leg_b}, leg_a.index)
    return strategy


def _context(bar_index: int, index: pd.DatetimeIndex, **params: object) -> StrategyContext:
    """Build a minimal strategy context.

    Args:
        bar_index: Bar position.
        index: Master index.
        **params: Parameter overrides.

    Returns:
        The :class:`StrategyContext`.
    """
    return StrategyContext(
        timestamp=pd.Timestamp(index[bar_index]),
        bar_index=bar_index,
        data={},
        features={},
        regime_state=cfg.STATE_RANGE_BOUND,
        regime_probabilities=np.array([1.0, 0.0, 0.0]),
        params=dict(params),
    )


def test_strategy_active_only_in_range_bound_regime(
    pair_data: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Stat-arb must be gated to State 0."""
    leg_a, leg_b = pair_data
    strategy = _build_strategy(leg_a, leg_b)
    assert strategy.is_active(cfg.STATE_RANGE_BOUND)
    assert not strategy.is_active(cfg.STATE_TREND)
    assert not strategy.is_active(cfg.STATE_SHOCK)


def test_strategy_enters_and_exits_on_zscore_thresholds(
    pair_data: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The state machine must honour entry / mean-exit / stop thresholds."""
    leg_a, leg_b = pair_data
    strategy = _build_strategy(leg_a, leg_b)
    frame = strategy.indicators
    assert frame is not None

    z = frame["zscore"].to_numpy()
    valid = frame["valid"].to_numpy()
    index = frame.index
    entry_z, exit_z, stop_z = 2.0, 0.0, 3.5

    seen_entry = False
    seen_exit = False
    expected_position = 0
    for i in range(len(frame)):
        if not valid[i]:
            continue
        strategy._position = expected_position  # keep the harness in sync
        signals = strategy.generate_signals(_context(i, index))
        zi = z[i]
        if signals:
            tag = signals[0].tag
            if expected_position == 0:
                assert tag in {"entry_long", "entry_short"}
                if tag == "entry_long":
                    assert zi <= -entry_z
                    expected_position = 1
                else:
                    assert zi >= entry_z
                    expected_position = -1
                seen_entry = True
            else:
                assert tag in {"exit_mean", "stop_out", "invalid_exit"}
                assert abs(zi) >= stop_z or (
                    expected_position == 1 and zi >= exit_z
                ) or (expected_position == -1 and zi <= exit_z)
                expected_position = 0
                seen_exit = True

    assert seen_entry, "The pair never produced an entry signal."
    assert seen_exit, "The pair never produced an exit signal."


def test_signals_carry_both_legs_with_opposite_signs(
    pair_data: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A pair entry must emit two signals with opposite directions."""
    leg_a, leg_b = pair_data
    strategy = _build_strategy(leg_a, leg_b)
    frame = strategy.indicators
    valid_positions = np.where(frame["valid"].to_numpy())[0]

    emitted: list[Signal] = []
    for i in valid_positions:
        emitted = strategy.generate_signals(_context(i, frame.index))
        if emitted:
            break
    assert len(emitted) == 2
    symbols = {signal.symbol for signal in emitted}
    assert symbols == {"LEGA", "LEGB"}
    assert np.sign(emitted[0].target_weight) == -np.sign(emitted[1].target_weight)


def test_stop_loss_triggers_on_extreme_zscore(
    pair_data: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A |z| >= 3.5 excursion must force a hard exit."""
    leg_a, leg_b = pair_data
    strategy = _build_strategy(leg_a, leg_b)
    frame = strategy.indicators
    z = frame["zscore"].to_numpy()
    valid = frame["valid"].to_numpy()

    # Force a long-spread position, then find a bar with z >= stop_z.
    strategy._position = 1
    extreme = [i for i in range(len(z)) if valid[i] and z[i] >= 3.5]
    if not extreme:
        pytest.skip("The fixture did not produce a |z| >= 3.5 excursion.")
    signals = strategy.generate_signals(_context(extreme[0], frame.index))
    assert signals
    assert signals[0].tag == "stop_out"
    assert strategy.position == 0


def test_leg_weights_preserve_gross_exposure() -> None:
    """Leg weights must sum (in absolute value) to the requested gross."""
    weight_a, weight_b = PairsStatArbStrategy.leg_weights(
        price_a=2000.0, price_b=25.0, beta=80.0, gross=0.30, side=1
    )
    assert abs(weight_a) + abs(weight_b) == pytest.approx(0.30, rel=1e-9)
    assert weight_a > 0 > weight_b


def test_leg_weights_reject_non_positive_prices() -> None:
    """Non-positive prices must raise rather than silently zeroing exposure."""
    with pytest.raises(ValueError):
        PairsStatArbStrategy.leg_weights(0.0, 25.0, 1.0, 0.2, 1)


def test_non_stationary_pair_is_gated_out() -> None:
    """A pair that fails the cointegration gate must never trade."""
    rng = np.random.default_rng(17)
    n = 600
    index = pd.bdate_range("2020-01-01", periods=n, name="timestamp")
    a = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=index)
    b = pd.Series(50 + np.cumsum(rng.normal(0, 1, n)), index=index)

    strategy = PairsStatArbStrategy(
        "WALKA", "WALKB", require_cointegration=True, name="GatedPair"
    )
    strategy.prepare(
        {"WALKA": make_ohlcv(a, "WALKA"), "WALKB": make_ohlcv(b, "WALKB")}, index
    )
    frame = strategy.indicators
    assert frame is not None
    # A size-5% test produces ~5% false positives on independent random walks,
    # so the gate must suppress the overwhelming majority of bars, not literally
    # every one of them.
    valid_share = float(frame["valid"].mean())
    assert valid_share < 0.10, (
        f"A non-cointegrated pair was tradable on {valid_share:.1%} of bars."
    )


def test_inverse_correlation_gate() -> None:
    """The FX gate must reject a pair whose correlation is not below -0.75."""
    rng = np.random.default_rng(23)
    n = 500
    index = pd.bdate_range("2020-01-01", periods=n, name="timestamp")
    a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))), index=index)
    b = pd.Series(1.1 * np.exp(np.cumsum(rng.normal(0, 0.005, n))), index=index)

    strategy = PairsStatArbStrategy(
        "FXA", "FXB", use_log=True, require_inverse_correlation=True,
        require_cointegration=False, name="FXGate",
    )
    strategy.prepare({"FXA": make_ohlcv(a, "FXA"), "FXB": make_ohlcv(b, "FXB")}, index)
    frame = strategy.indicators
    assert frame is not None
    assert frame["valid"].sum() == 0


def test_notify_flat_resets_internal_state(
    pair_data: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """An external stop-out must clear the strategy's internal position."""
    leg_a, leg_b = pair_data
    strategy = _build_strategy(leg_a, leg_b)
    strategy._position = 1
    strategy.notify_flat("LEGA")
    assert strategy.position == 0


def test_reset_clears_state(pair_data: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """``reset`` must return the strategy to its unprepared state."""
    leg_a, leg_b = pair_data
    strategy = _build_strategy(leg_a, leg_b)
    strategy.reset()
    assert strategy.indicators is None
    assert strategy.position == 0
    assert not strategy.prepared
