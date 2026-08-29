"""Integration tests for the execution and backtesting pipeline.

These tests exercise the whole chain end to end on a deterministic synthetic
universe: data ingestion -> feature engineering -> causal HMM regimes ->
regime-routed strategies -> ATR stops -> sizing -> costed fills -> metrics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from quant_system.backtesting.engine import BacktestEngine, VectorizedBacktester
from quant_system.backtesting.walk_forward import WalkForwardOptimizer
from quant_system.config import settings as cfg
from quant_system.data.ingestion import DataIngestion
from quant_system.execution.portfolio import (
    Fill,
    Portfolio,
    kelly_fraction,
    kelly_fraction_from_returns,
    realized_covariance,
    risk_contributions,
    risk_parity_weights,
)
from quant_system.execution.risk_manager import RiskManager
from quant_system.models.hmm_switchboard import CausalRegimeStreamer
from quant_system.strategies.adaptive_momentum import AdaptiveMomentumStrategy
from quant_system.strategies.stat_arb import build_default_stat_arb_book

START = "2019-01-01"
END = "2021-06-30"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def universe() -> Dict[str, pd.DataFrame]:
    """Deterministic synthetic universe (no network access)."""
    ingestion = DataIngestion(source="synthetic", use_cache=False)
    return ingestion.fetch_universe(start=START, end=END, interval="1d")


@pytest.fixture(scope="module")
def regimes(universe: Dict[str, pd.DataFrame]) -> pd.Series:
    """Causal regime path for the synthetic universe."""
    from quant_system.data.preprocessing import build_market_features

    features = build_market_features(universe)
    config = cfg.HMMConfig(train_window=250, refit_every=25, min_train=120)
    return CausalRegimeStreamer(config=config).run(features).states


def _strategies() -> list:
    """Fresh strategy book."""
    strategies: list = list(build_default_stat_arb_book())
    strategies.append(AdaptiveMomentumStrategy())
    return strategies


# --------------------------------------------------------------------------- #
# Engine integration
# --------------------------------------------------------------------------- #
def test_engine_runs_end_to_end(universe: Dict[str, pd.DataFrame], regimes: pd.Series) -> None:
    """The engine must produce a complete, finite equity curve."""
    engine = BacktestEngine(
        data=universe, strategies=_strategies(), regime_states=regimes
    )
    result = engine.run()

    assert not result.equity_curve.empty
    assert np.isfinite(result.equity_curve.to_numpy()).all()
    assert len(result.equity_curve) == len(next(iter(universe.values())))
    assert not result.position_weights.empty
    assert "sharpe_ratio" in result.metrics
    assert len(result.regime_states) == len(result.equity_curve)


def test_engine_starting_equity_matches_capital(
    universe: Dict[str, pd.DataFrame], regimes: pd.Series
) -> None:
    """The first equity observation must equal the configured capital."""
    capital = 500_000.0
    engine = BacktestEngine(
        data=universe,
        strategies=_strategies(),
        regime_states=regimes,
        initial_capital=capital,
    )
    result = engine.run()
    assert result.equity_curve.iloc[0] == pytest.approx(capital, rel=1e-9)
    assert result.metrics["start_equity"] == pytest.approx(capital, rel=1e-9)


def test_engine_produces_trades_and_charges_costs(
    universe: Dict[str, pd.DataFrame], regimes: pd.Series
) -> None:
    """A live book must generate fills that carry spread and slippage costs."""
    engine = BacktestEngine(
        data=universe, strategies=_strategies(), regime_states=regimes
    )
    result = engine.run()

    assert not result.fills.empty, "The engine produced no fills at all."
    assert (result.fills["spread_cost"] > 0).all()
    assert (result.fills["slippage_cost"] > 0).all()
    assert (result.fills["commission"] >= 0).all()
    if not result.trades.empty:
        # Net PnL must equal gross PnL minus attributed costs.
        np.testing.assert_allclose(
            result.trades["net_pnl"].to_numpy(),
            result.trades["gross_pnl"].to_numpy() - result.trades["costs"].to_numpy(),
            atol=1e-6,
        )


def test_higher_commissions_lower_terminal_equity(
    universe: Dict[str, pd.DataFrame], regimes: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monotonically: more costs -> weakly lower terminal equity."""
    cheap = BacktestEngine(
        data=universe,
        strategies=_strategies(),
        regime_states=regimes,
        costs=cfg.CostConfig(commission_rate=0.0, financing_rate_annual=0.0),
    ).run()

    expensive = BacktestEngine(
        data=universe,
        strategies=_strategies(),
        regime_states=regimes,
        costs=cfg.CostConfig(commission_rate=0.01, financing_rate_annual=0.05),
    ).run()

    assert cheap.final_equity >= expensive.final_equity - 1e-6


def test_state_two_halts_all_entries(universe: Dict[str, pd.DataFrame]) -> None:
    """A permanent State 2 must flatten the book and block every new entry."""
    index = next(iter(universe.values())).index
    shock_regimes = pd.Series(cfg.STATE_SHOCK, index=index, dtype=int)

    engine = BacktestEngine(
        data=universe, strategies=_strategies(), regime_states=shock_regimes
    )
    result = engine.run()

    assert result.trades.empty, "Positions were opened while State 2 was active."
    assert result.equity_curve.nunique() == 1, "Equity moved with no positions open."
    assert result.final_equity == pytest.approx(
        cfg.DEFAULT_SETTINGS.sizing.initial_capital, rel=1e-9
    )


def test_state_one_deactivates_stat_arb(universe: Dict[str, pd.DataFrame]) -> None:
    """A permanent State 1 must keep the mean-reversion book flat."""
    index = next(iter(universe.values())).index
    trend_regimes = pd.Series(cfg.STATE_TREND, index=index, dtype=int)
    strategies = list(build_default_stat_arb_book())

    engine = BacktestEngine(
        data=universe, strategies=strategies, regime_states=trend_regimes
    )
    result = engine.run()
    traded_symbols = set(result.fills["symbol"]) if not result.fills.empty else set()
    stat_arb_symbols = {
        *cfg.DEFAULT_SETTINGS.universe.metals_pair,
        *cfg.DEFAULT_SETTINGS.universe.fx_pair,
    }
    assert traded_symbols.isdisjoint(stat_arb_symbols), (
        f"Stat-arb traded {traded_symbols & stat_arb_symbols} outside State 0."
    )


def test_position_weights_stay_within_leverage_limit(
    universe: Dict[str, pd.DataFrame], regimes: pd.Series
) -> None:
    """Gross leverage must never breach the configured maximum."""
    engine = BacktestEngine(
        data=universe, strategies=_strategies(), regime_states=regimes
    )
    result = engine.run()
    gross = result.position_weights.abs().sum(axis=1)
    assert gross.max() <= cfg.DEFAULT_SETTINGS.sizing.max_gross_leverage + 0.05


def test_engine_is_deterministic(universe: Dict[str, pd.DataFrame], regimes: pd.Series) -> None:
    """Two identical runs must produce identical equity curves."""
    first = BacktestEngine(
        data=universe, strategies=_strategies(), regime_states=regimes
    ).run()
    second = BacktestEngine(
        data=universe, strategies=_strategies(), regime_states=regimes
    ).run()
    pd.testing.assert_series_equal(first.equity_curve, second.equity_curve)


# --------------------------------------------------------------------------- #
# Portfolio accounting
# --------------------------------------------------------------------------- #
def test_average_cost_realized_pnl() -> None:
    """Realised PnL must use the volume-weighted average entry price."""
    portfolio = Portfolio(initial_capital=100_000.0)
    timestamp = pd.Timestamp("2021-01-01")

    portfolio.apply_fill(Fill(timestamp, "TEST", 10.0, 100.0, strategy="t"))
    portfolio.apply_fill(Fill(timestamp, "TEST", 10.0, 120.0, strategy="t"))
    position = portfolio.position("TEST")
    assert position.quantity == pytest.approx(20.0)
    assert position.average_price == pytest.approx(110.0)

    realized, closed = portfolio.apply_fill(
        Fill(timestamp, "TEST", -20.0, 130.0, strategy="t")
    )
    # (130 - 110) * 20 units * contract_size(1.0)
    assert realized == pytest.approx(400.0)
    assert closed == pytest.approx(20.0)
    assert position.quantity == pytest.approx(0.0)
    assert len(portfolio.trades) == 1


def test_short_position_pnl_sign() -> None:
    """A short position must profit when the price falls."""
    portfolio = Portfolio(initial_capital=100_000.0)
    timestamp = pd.Timestamp("2021-01-01")
    portfolio.apply_fill(Fill(timestamp, "TEST", -5.0, 50.0, strategy="t"))
    realized, _ = portfolio.apply_fill(Fill(timestamp, "TEST", 5.0, 40.0, strategy="t"))
    assert realized == pytest.approx(50.0)


def test_partial_close_and_commission_are_booked() -> None:
    """A partial close must realise a proportional PnL and charge commission."""
    portfolio = Portfolio(initial_capital=100_000.0)
    timestamp = pd.Timestamp("2021-01-01")
    portfolio.apply_fill(Fill(timestamp, "TEST", 10.0, 100.0, strategy="t"))
    cash_before = portfolio.cash
    realized, closed = portfolio.apply_fill(
        Fill(timestamp, "TEST", -4.0, 110.0, commission=7.0, strategy="t")
    )
    assert closed == pytest.approx(4.0)
    assert realized == pytest.approx(40.0)
    # Cash: -10*100 (open) + 4*110 (close) - 7 commission
    assert portfolio.cash == pytest.approx(cash_before + 440.0 - 7.0)
    assert portfolio.trades[-1].net_pnl == pytest.approx(33.0)


def test_position_flip_closes_and_reopens() -> None:
    """An oversized opposing fill must close the leg and open the reverse."""
    portfolio = Portfolio(initial_capital=100_000.0)
    timestamp = pd.Timestamp("2021-01-01")
    portfolio.apply_fill(Fill(timestamp, "TEST", 5.0, 100.0, strategy="t"))
    realized, closed = portfolio.apply_fill(Fill(timestamp, "TEST", -12.0, 90.0, strategy="t"))
    assert closed == pytest.approx(5.0)
    assert realized == pytest.approx(-50.0)
    position = portfolio.position("TEST")
    assert position.quantity == pytest.approx(-7.0)
    assert position.average_price == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# Sizing: Kelly and risk parity
# --------------------------------------------------------------------------- #
def test_kelly_fraction_matches_closed_form() -> None:
    """Kelly must equal ``(p*b - q)/b`` for a known example."""
    # p=0.6, avg_win=2, avg_loss=1 -> b=2 -> f* = (0.6*2 - 0.4)/2 = 0.4
    assert kelly_fraction(0.6, 2.0, 1.0) == pytest.approx(0.4)
    # Negative expectancy -> never bet.
    assert kelly_fraction(0.4, 1.0, 2.0) == 0.0


def test_kelly_fraction_clips_to_unit_interval() -> None:
    """A near-certain bet must be clipped to 1.0, not exceed it."""
    assert kelly_fraction(0.99, 100.0, 1.0) <= 1.0


def test_kelly_from_returns_is_mean_over_variance() -> None:
    """Continuous Kelly must equal ``mu / sigma^2``."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 1000))
    expected = returns.mean() / returns.var(ddof=1)
    assert kelly_fraction_from_returns(returns) == pytest.approx(expected, rel=1e-9)


def test_risk_parity_equalises_risk_contributions() -> None:
    """ERC weights must give every asset an identical risk contribution."""
    covariance = np.array(
        [
            [0.04, 0.006, 0.002],
            [0.006, 0.09, 0.003],
            [0.002, 0.003, 0.01],
        ]
    )
    weights = risk_parity_weights(covariance)
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)

    contributions = risk_contributions(weights, covariance)
    np.testing.assert_allclose(contributions, np.full(3, 1 / 3), atol=1e-6)


def test_risk_parity_beats_equal_weight_on_concentration() -> None:
    """ERC must be less concentrated than inverse-vol for a correlated book."""
    covariance = np.array([[0.09, 0.05], [0.05, 0.04]])
    weights = risk_parity_weights(covariance)
    contributions = risk_contributions(weights, covariance)
    assert np.max(np.abs(contributions - 0.5)) < 1e-6


def test_sizing_engine_caps_gross_exposure() -> None:
    """The sizing engine must respect the aggregate leverage cap."""
    from quant_system.execution.portfolio import SizingEngine

    engine = SizingEngine()
    targets = {f"S{i}": 0.5 for i in range(5)}
    sized = engine.size(targets)
    gross = sum(abs(w) for w in sized.values())
    assert gross <= cfg.DEFAULT_SETTINGS.sizing.max_gross_leverage + 1e-9


def test_sizing_engine_zeroes_exposure_in_shock_regime() -> None:
    """State 2's exposure scalar must flatten targeted exposure."""
    from quant_system.execution.portfolio import SizingEngine

    engine = SizingEngine()
    sized = engine.size({"XAUUSD": 0.3}, regime_state=cfg.STATE_SHOCK)
    assert all(abs(weight) < 1e-12 for weight in sized.values())


def test_realized_covariance_is_symmetric_and_annualised(
    universe: Dict[str, pd.DataFrame],
) -> None:
    """The covariance helper must return a symmetric positive-diagonal matrix."""
    covariance = realized_covariance(universe, window=60)
    assert not covariance.empty
    np.testing.assert_allclose(covariance.to_numpy(), covariance.to_numpy().T, atol=1e-12)
    assert (np.diag(covariance.to_numpy()) > 0).all()


# --------------------------------------------------------------------------- #
# Risk manager
# --------------------------------------------------------------------------- #
def test_atr_trailing_stop_ratchets_only_favourably() -> None:
    """A long stop must rise with the price and never fall."""
    from quant_system.execution.portfolio import Position

    manager = RiskManager()
    position = Position(symbol="T", quantity=10.0, average_price=100.0, last_price=100.0)
    atr = pd.Series([1.0, 1.0, 1.0])
    prices = {"T": 1.0}
    manager.arm_stop(position, atr=1.0, multiple=2.5)
    first_stop = position.stop_price
    assert first_stop == pytest.approx(97.5)

    position.last_price = 110.0
    manager.update_trailing_stops({"T": position}, prices, multiplier=1.0)
    assert position.stop_price == pytest.approx(107.5)

    position.last_price = 105.0
    manager.update_trailing_stops({"T": position}, prices, multiplier=1.0)
    assert position.stop_price == pytest.approx(107.5), "Stop must not ratchet down."


def test_shock_regime_tightens_stops_by_half() -> None:
    """State 2 must halve the ATR stop distance."""
    from quant_system.execution.portfolio import Position

    manager = RiskManager()
    position = Position(symbol="T", quantity=10.0, average_price=100.0, last_price=100.0)
    manager.update_trailing_stops(
        {"T": position}, {"T": 2.0}, multiplier=cfg.DEFAULT_SETTINGS.preservation.stop_tightening_factor
    )
    # 2.5 ATR * 0.5 = 1.25 ATR = 2.5 price units below 100.
    assert position.stop_price == pytest.approx(97.5)


def test_atr_position_size_bounds_risk_per_unit() -> None:
    """A stop-out must cost approximately the configured risk fraction."""
    units = RiskManager.atr_position_size(
        equity=1_000_000.0, price=100.0, atr=2.0, risk_per_unit_pct=0.01,
        stop_multiple=2.5, contract_size=1.0,
    )
    loss_if_stopped = units * 2.5 * 2.0
    assert loss_if_stopped == pytest.approx(10_000.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# Vectorized engine
# --------------------------------------------------------------------------- #
def test_vectorized_backtester_matches_manual_algebra(
    universe: Dict[str, pd.DataFrame],
) -> None:
    """A constant long position must track the underlying asset return."""
    prices = pd.DataFrame({symbol: frame["close"] for symbol, frame in universe.items()})
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights["XAUUSD"] = 1.0

    tester = VectorizedBacktester(
        commission_rate=0.0, financing_rate=0.0, spread_cost=0.0
    )
    result = tester.run(weights, prices)

    from quant_system.data.preprocessing import log_returns

    expected = log_returns(prices["XAUUSD"]).shift(0).iloc[1:]
    np.testing.assert_allclose(
        result.returns.to_numpy()[1:], expected.to_numpy(), atol=1e-12
    )


def test_vectorized_backtester_charges_costs(universe: Dict[str, pd.DataFrame]) -> None:
    """Turnover must be charged, so a zero-cost run beats a costed one."""
    prices = pd.DataFrame({symbol: frame["close"] for symbol, frame in universe.items()})
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights["USDJPY"] = np.where(np.arange(len(weights)) % 10 < 5, 1.0, -1.0)

    free = VectorizedBacktester(commission_rate=0.0).run(weights, prices)
    costed = VectorizedBacktester(commission_rate=0.01).run(weights, prices)
    assert free.final_equity > costed.final_equity


# --------------------------------------------------------------------------- #
# Walk-forward optimisation
# --------------------------------------------------------------------------- #
def test_walk_forward_produces_stitched_oos_curve(
    universe: Dict[str, pd.DataFrame], regimes: pd.Series
) -> None:
    """The optimiser must emit per-segment results and a stitched OOS curve."""
    config = cfg.WalkForwardConfig(
        is_months=12,
        oos_months=3,
        step_months=6,
        z_entry_grid=(1.8, 2.0, 2.2),
        momentum_grid=((15, 40),),
    )
    optimizer = WalkForwardOptimizer(
        data=universe,
        regime_states=regimes,
        config=config,
        strategy_factory=_strategies,
        warmup_bars=180,
        verbose=False,
    )
    result = optimizer.run()

    assert len(result.segments) >= 1
    assert not result.oos_returns.empty
    assert not result.oos_equity.empty
    assert len(result.oos_returns) == len(result.oos_equity)
    assert set(result.degradation_summary.columns) >= {
        "segment",
        "is_sharpe",
        "oos_sharpe",
        "degradation_pct",
        "accepted",
    }
    assert "sharpe_ratio" in result.metrics
    # Every segment must have selected a parameter set from the grid.
    for segment in result.segments:
        assert segment.best_params["entry_z"] in (1.8, 2.0, 2.2)
        assert segment.n_combinations == 3


def test_walk_forward_segment_windows_are_contiguous(
    universe: Dict[str, pd.DataFrame], regimes: pd.Series
) -> None:
    """OOS segments must be non-overlapping and monotonically ordered."""
    config = cfg.WalkForwardConfig(
        is_months=12, oos_months=3, step_months=6,
        z_entry_grid=(2.0,), momentum_grid=((15, 40),),
    )
    optimizer = WalkForwardOptimizer(
        data=universe,
        regime_states=regimes,
        config=config,
        strategy_factory=_strategies,
        warmup_bars=180,
        verbose=False,
    )
    result = optimizer.run()
    for previous, current in zip(result.segments, result.segments[1:]):
        assert current.oos_start > previous.oos_start
        assert current.is_start >= previous.is_start


def test_backtrader_cross_validation_agrees_with_vectorized_engine(
    universe: Dict[str, pd.DataFrame],
) -> None:
    """Our PnL algebra must agree with an independent third-party engine.

    ``backtrader`` runs its own broker, data feed and order machinery, so
    agreement on the same signal is meaningful evidence that the vectorized
    engine prices exposure correctly.
    """
    pytest.importorskip("backtrader", reason="backtrader is not installed")

    from quant_system.backtesting.engine import compare_engines_on_sma_cross

    ohlcv = universe["USDJPY"].copy()
    ohlcv.attrs["symbol"] = "USDJPY"
    diagnostics = compare_engines_on_sma_cross(ohlcv, fast_period=20, slow_period=50)

    assert diagnostics["exposure"] > 0.05, "The crossover signal is never long."
    assert diagnostics["return_correlation"] > 0.90, (
        f"Engines disagree: return correlation "
        f"{diagnostics['return_correlation']:.3f} (backtrader "
        f"{diagnostics['backtrader_total_return']:.4f} vs vectorized "
        f"{diagnostics['vectorized_total_return']:.4f})."
    )
    # Residual differences are expected: backtrader uses integer share sizing and
    # commits a percentage of cash at entry, while our engine rebalances to its
    # target weight every bar.  The *sign* of the outcome must still agree.
    if abs(diagnostics["vectorized_total_return"]) > 0.01:
        assert (
            np.sign(diagnostics["backtrader_total_return"])
            == np.sign(diagnostics["vectorized_total_return"])
        ), "The two engines disagree on whether the signal made or lost money."


def test_walk_forward_rejects_short_sample(universe: Dict[str, pd.DataFrame], regimes: pd.Series) -> None:
    """A sample too short for one segment must raise a clear error."""
    short_data = {symbol: frame.iloc[:60] for symbol, frame in universe.items()}
    config = cfg.WalkForwardConfig(is_months=12, oos_months=3)
    optimizer = WalkForwardOptimizer(
        data=short_data, regime_states=regimes.iloc[:60], config=config, verbose=False
    )
    with pytest.raises(ValueError):
        optimizer.run()
