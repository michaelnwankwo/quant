"""Backtesting engines: event-driven (primary) and vectorized (fast sweep).

Event-driven engine
-------------------
The engine replays the aligned bar calendar one timestamp at a time and keeps a
strict ordering of operations so that no decision can observe the future:

.. code-block:: text

    bar i:
      1. fill orders queued at the close of bar i-1, at bar i's OPEN
      2. intrabar ATR stop surveillance against bar i's HIGH/LOW
         (a gap through the stop fills at bar i's OPEN)
      3. mark to market at bar i's CLOSE
      4. ratchet ATR trailing stops (multiple scaled by the active regime)
      5. portfolio drawdown circuit breaker
      6. capital-preservation overlay (halt / tighten / de-risk)
      7. strategy signals computed from data up to bar i's CLOSE
      8. sizing -> orders queued for bar i+1's OPEN
      9. record equity at bar i's CLOSE

Signals are therefore always generated on a close and executed on the *next*
open, which is both realistic and free of look-ahead bias.

Cost model
----------
Each fill pays, explicitly:

* half the quoted spread (``AssetSpec.spread_pips * pip_size / 2``),
* adverse slippage (``AssetSpec.slippage_pips * pip_size``),
* commission (``CostConfig.commission_rate`` on notional + a fixed ticket fee).

Costs are booked as a cash debit *and* attributed to the trade record, so
``net_pnl = gross_pnl - costs`` holds exactly.

Vectorized engine
-----------------
:class:`VectorizedBacktester` takes a precomputed frame of target weights and
produces the same PnL algebra in closed form.  It is ~100x faster and is used for
sanity checks and rapid parameter exploration, while the event-driven engine
remains the source of truth (it models stops, de-risking and partial fills).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quant_system.analytics import metrics as ametrics
from quant_system.config import settings as cfg
from quant_system.data.preprocessing import BARS_PER_YEAR, FeatureBuilder, log_returns
from quant_system.execution.portfolio import (
    Fill,
    Portfolio,
    Position,
    SizingEngine,
    realized_covariance,
)
from quant_system.execution.risk_manager import RiskManager
from quant_system.models.hmm_switchboard import CausalRegimeStreamer, RegimeStreamResult
from quant_system.strategies.base import BaseStrategy, PositionSnapshot, Signal, StrategyContext
from quant_system.strategies.risk_preservation import (
    PreservationAction,
    RiskPreservationStrategy,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    """Everything a backtest produces.

    Attributes:
        equity_curve: Equity indexed by bar timestamp.
        returns: Periodic returns of the equity curve.
        trades: Trade log DataFrame.
        fills: Fill log DataFrame.
        regime_states: Regime id per bar.
        regime_probabilities: Canonical state probabilities per bar.
        position_weights: Signed exposure per symbol per bar.
        metrics: Headline performance dictionary.
        regime_metrics: Per-regime performance breakdown.
        events: Chronological list of notable events (stops, de-risking, halts).
    """

    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    fills: pd.DataFrame
    regime_states: pd.Series
    regime_probabilities: pd.DataFrame
    position_weights: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)
    regime_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    events: List[Dict[str, object]] = field(default_factory=list)

    def summary(self) -> str:
        """Render the headline metrics as text.

        Returns:
            A formatted multi-line summary.
        """
        return ametrics.format_summary(self.metrics)

    @property
    def final_equity(self) -> float:
        """Terminal equity."""
        return float(self.equity_curve.iloc[-1]) if not self.equity_curve.empty else 0.0


@dataclass(frozen=True)
class _PendingOrder:
    """An order queued for execution on the next bar's open.

    Attributes:
        symbol: Instrument symbol.
        quantity: Signed order quantity.
        tag: Reason code.
        strategy: Strategy attribution.
    """

    symbol: str
    quantity: float
    tag: str
    strategy: str


# --------------------------------------------------------------------------- #
# Event-driven engine
# --------------------------------------------------------------------------- #
class BacktestEngine:
    """Event-driven backtester with fractional sizing, costs and regime routing.

    Attributes:
        data: Mapping of symbol -> aligned OHLCV frame.
        strategies: Alpha strategies to run.
        regime_states: Causal regime series (or a :class:`RegimeStreamResult`).
        portfolio: Portfolio instance used for accounting.
        risk_manager: Stop and exposure-limit manager.
        sizing_engine: Kelly / risk-parity sizing engine.
        preservation: Capital-preservation overlay (State 2).
        costs: Execution cost model.
    """

    def __init__(
        self,
        data: Mapping[str, pd.DataFrame],
        strategies: Sequence[BaseStrategy],
        regime_states: "pd.Series | RegimeStreamResult",
        regime_probabilities: Optional[pd.DataFrame] = None,
        portfolio: Optional[Portfolio] = None,
        risk_manager: Optional[RiskManager] = None,
        sizing_engine: Optional[SizingEngine] = None,
        preservation: Optional[RiskPreservationStrategy] = None,
        costs: Optional[cfg.CostConfig] = None,
        config: Optional[cfg.BacktestConfig] = None,
        initial_capital: Optional[float] = None,
        covariance: Optional[pd.DataFrame] = None,
        fill_at: Optional[str] = None,
        params: Optional[Dict[str, object]] = None,
        strategies_prepared: bool = False,
        compute_metrics: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialise the engine.

        Args:
            data: Mapping of symbol -> OHLCV frame (must share one index).
            strategies: Strategies to run.
            regime_states: Regime series or streaming result aligned to the data.
            regime_probabilities: Optional probability frame.
            portfolio: Portfolio instance; a new one is created if omitted.
            risk_manager: Risk manager; a default one is created if omitted.
            sizing_engine: Sizing engine; a default one is created if omitted.
            preservation: Preservation overlay; a default one is created if omitted.
            costs: Cost model.
            config: Backtest configuration.
            initial_capital: Overrides the sizing config's initial capital.
            covariance: Pre-computed annualised covariance used for vol targeting
                and risk parity. When omitted it is estimated from the first
                ``covariance_calibration_bars`` bars (a causal calibration
                window, never the full sample).
            fill_at: ``"next_open"`` (default) or ``"close"``.
            params: Parameter overrides forwarded to ``StrategyContext.params``.
            strategies_prepared: When ``True`` the engine will *not* call
                :meth:`BaseStrategy.prepare` (the caller has already done it).
                Used by the walk-forward optimiser to reuse one indicator
                preparation across an entire parameter sweep.
            compute_metrics: When ``False`` the result skips the (comparatively
                expensive) performance-metric and regime-breakdown computation.
                Used by the walk-forward optimiser's in-sample sweep, which only
                needs the objective.
            verbose: Emit per-bar logging.
        """
        self.config: cfg.BacktestConfig = config or cfg.DEFAULT_SETTINGS.backtest
        self.costs: cfg.CostConfig = costs or cfg.DEFAULT_SETTINGS.costs
        if not data:
            raise ValueError("BacktestEngine requires at least one symbol.")
        self.strategies: List[BaseStrategy] = list(strategies)
        self.verbose: bool = verbose
        self._index = self._build_index(dict(data))
        # Reindex every symbol onto the master calendar so bar ``i`` means the
        # same instant for all instruments.
        self.data: Dict[str, pd.DataFrame] = {
            symbol: frame.reindex(self._index).ffill() for symbol, frame in data.items()
        }

        if isinstance(regime_states, RegimeStreamResult):
            self.regime_states: pd.Series = regime_states.states
            self.regime_probabilities: pd.DataFrame = (
                regime_probabilities
                if regime_probabilities is not None
                else regime_states.probabilities
            )
        else:
            self.regime_states = pd.Series(regime_states)
            self.regime_probabilities = (
                regime_probabilities
                if regime_probabilities is not None
                else pd.DataFrame()
            )

        self.portfolio: Portfolio = portfolio or Portfolio(
            initial_capital=initial_capital or cfg.DEFAULT_SETTINGS.sizing.initial_capital
        )
        self.risk_manager: RiskManager = risk_manager or RiskManager()
        self.sizing_engine: SizingEngine = sizing_engine or SizingEngine()
        self.preservation: RiskPreservationStrategy = preservation or RiskPreservationStrategy(
            symbols=list(self.data.keys())
        )
        self.fill_at: str = fill_at or self.config.fill_at
        self.covariance: Optional[pd.DataFrame] = covariance
        self.covariance_calibration_bars: int = 252
        #: Parameter overrides injected into every :class:`StrategyContext`.
        self.params: Dict[str, object] = dict(params or {})
        self.strategies_prepared: bool = bool(strategies_prepared)
        self.compute_metrics: bool = bool(compute_metrics)

        # Runtime state
        self._features: Dict[str, pd.DataFrame] = {}
        self._atr: Dict[str, pd.Series] = {}
        self._pending: List[_PendingOrder] = []
        self._raw_targets: Dict[str, float] = {}
        self._sent_targets: Dict[str, float] = {}
        self._suppressed_until: Dict[str, int] = {}
        # Cumulative count of new positions refused by the daily trade cap.
        self._blocked_by_limit: int = 0
        self._events: List[Dict[str, object]] = []
        self._weights_history: Dict[pd.Timestamp, Dict[str, float]] = {}
        self._latest_stop_multiple: Dict[str, Optional[float]] = {}
        self._pending_stop_multiplier: float = 1.0
        self._trade_frame_cache: Optional[pd.DataFrame] = None
        self._trade_frame_count: int = -1

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_index(data: Mapping[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Build the master bar index (intersection of all symbol calendars).

        Args:
            data: Mapping of symbol -> OHLCV frame.

        Returns:
            The aligned DatetimeIndex.

        Raises:
            ValueError: If the calendars do not overlap.
        """
        index: Optional[pd.DatetimeIndex] = None
        for frame in data.values():
            index = frame.index if index is None else index.intersection(frame.index)
        if index is None or len(index) == 0:
            raise ValueError("Symbol calendars have no overlapping timestamps.")
        return pd.DatetimeIndex(sorted(index))

    def _prepare(self) -> None:
        """Precompute features/ATR and prepare every strategy."""
        builder = FeatureBuilder(interval=self.config.interval)
        for symbol, frame in self.data.items():
            features = builder.symbol_features(symbol, frame)
            self._features[symbol] = features
            self._atr[symbol] = features["atr_raw"].reindex(self._index).ffill()

        # Causal covariance calibration from the head of the sample only.
        if self.covariance is None:
            window = min(self.covariance_calibration_bars, max(30, len(self._index) // 5))
            head = {s: f.iloc[:window] for s, f in self.data.items()}
            self.covariance = realized_covariance(
                head,
                window=cfg.DEFAULT_SETTINGS.sizing.cov_window,
                interval=self.config.interval,
            )
            if self.covariance.empty:
                self.covariance = None

        if not self.strategies_prepared:
            for strategy in self.strategies:
                strategy.reset()
                strategy.prepare(self.data, self._index)
        self.preservation.prepare(self.data, self._index)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> BacktestResult:
        """Execute the backtest.

        Returns:
            A :class:`BacktestResult`.
        """
        self._prepare()
        index = self._index
        n = len(index)

        for i in range(n):
            timestamp = pd.Timestamp(index[i])
            bar = {symbol: frame.iloc[i] for symbol, frame in self.data.items()}

            # 1. Fill orders queued on the previous bar ------------------ #
            self._execute_pending(i, timestamp)

            # 2. Intrabar stop surveillance ------------------------------ #
            self._check_stops(i, timestamp, bar)

            # 3. Mark to market at the close ----------------------------- #
            closes = {symbol: float(row["close"]) for symbol, row in bar.items()}
            self.portfolio.update_prices(closes)
            self.portfolio.accrue_financing(
                self.costs.financing_rate_annual,
                BARS_PER_YEAR.get(self.config.interval, 252.0),
            )

            # 4. Ratchet trailing stops ---------------------------------- #
            state = self._regime_at(i)
            stop_multiplier = self._pending_stop_multiplier
            self._update_stops(i, stop_multiplier)

            # 5. Drawdown circuit breaker -------------------------------- #
            breached, drawdown = self.risk_manager.drawdown_breach(self.portfolio)
            if breached and not self._halted:
                self._halt(i, timestamp, drawdown)

            # 6. Preservation overlay ------------------------------------ #
            action = self._evaluate_preservation(i, timestamp)
            self._pending_stop_multiplier = action.stop_multiplier

            # 7. Signals -------------------------------------------------- #
            self._collect_signals(i, timestamp, state, action)

            # 8. Record equity ------------------------------------------- #
            self.portfolio.record_equity(timestamp)
            self._weights_history[timestamp] = {
                symbol: position.market_value / max(self.portfolio.equity, 1e-12)
                for symbol, position in self.portfolio.positions.items()
            }

        return self._build_result()

    # ------------------------------------------------------------------ #
    # Bar-level steps
    # ------------------------------------------------------------------ #
    @property
    def _halted(self) -> bool:
        """Whether the drawdown circuit breaker has fired."""
        return self.risk_manager.halted

    @property
    def blocked_trades(self) -> int:
        """New positions refused by the daily trade-frequency cap."""
        return int(self._blocked_by_limit)

    @property
    def demo_mode(self) -> bool:
        """Whether the portfolio is running in unlimited demo mode."""
        return bool(self.portfolio.demo_mode)

    def _regime_at(self, i: int) -> int:
        """Return the regime id at bar ``i``.

        Args:
            i: Bar position.

        Returns:
            Canonical regime id (defaults to ``STATE_RANGE_BOUND``).
        """
        if i < len(self.regime_states):
            value = self.regime_states.iloc[i]
            if np.isfinite(value):
                return int(value)
        return int(cfg.STATE_RANGE_BOUND)

    def _probabilities_at(self, i: int) -> np.ndarray:
        """Return the canonical probability vector at bar ``i``.

        Args:
            i: Bar position.

        Returns:
            Array ``[p0, p1, p2]``.
        """
        if self.regime_probabilities is not None and i < len(self.regime_probabilities):
            row = self.regime_probabilities.iloc[i].to_numpy(dtype=float)
            if row.size == cfg.N_REGIMES and np.isfinite(row).all():
                total = row.sum()
                return row / total if total > 0 else row
        return np.zeros(cfg.N_REGIMES)

    def _execute_pending(self, i: int, timestamp: pd.Timestamp) -> None:
        """Fill every queued order at the current bar's reference price.

        Args:
            i: Bar position.
            timestamp: Bar timestamp.
        """
        if not self._pending:
            return
        for order in self._pending:
            frame = self.data.get(order.symbol)
            if frame is None:
                continue
            reference = (
                float(frame.iloc[i]["open"])
                if self.fill_at == "next_open"
                else float(frame.iloc[i]["close"])
            )
            self._fill(order.symbol, order.quantity, reference, timestamp, order.strategy, order.tag)
        self._pending = []

    def _check_stops(
        self,
        i: int,
        timestamp: pd.Timestamp,
        bar: Mapping[str, pd.Series],
    ) -> None:
        """Close positions whose ATR stop was breached intrabar.

        Args:
            i: Bar position.
            timestamp: Bar timestamp.
            bar: Mapping of symbol -> current bar row.
        """
        opens = {symbol: float(row["open"]) for symbol, row in bar.items()}
        highs = {symbol: float(row["high"]) for symbol, row in bar.items()}
        lows = {symbol: float(row["low"]) for symbol, row in bar.items()}
        triggers = self.risk_manager.check_stops(self.portfolio.positions, opens, highs, lows)
        for trigger in triggers:
            position = self.portfolio.positions.get(trigger.symbol)
            if position is None or position.direction == 0:
                continue
            quantity = -position.quantity
            self._fill(
                trigger.symbol,
                quantity,
                trigger.fill_price,
                timestamp,
                position.strategy or "risk",
                trigger.reason,
            )
            self._raw_targets[trigger.symbol] = 0.0
            self._sent_targets[trigger.symbol] = 0.0
            for strategy in self.strategies:
                if trigger.symbol in strategy.symbols:
                    strategy.notify_flat(trigger.symbol)
            self._events.append(
                {
                    "timestamp": timestamp,
                    "event": "stop_out",
                    "symbol": trigger.symbol,
                    "detail": f"stop={trigger.stop_price:.5f} fill={trigger.fill_price:.5f}",
                }
            )

    def _update_stops(self, i: int, multiplier: float) -> None:
        """Ratchet trailing stops, honouring per-signal stop preferences.

        Args:
            i: Bar position.
            multiplier: Regime adjustment applied to the ATR multiple.
        """
        atr_now = {
            symbol: (
                float(series.iloc[i])
                if i < len(series) and np.isfinite(series.iloc[i])
                else float("nan")
            )
            for symbol, series in self._atr.items()
        }
        managed: Dict[str, Position] = {}
        for symbol, position in self.portfolio.positions.items():
            if position.direction == 0:
                continue
            # Strategies that exit on a statistical trigger (e.g. a spread
            # z-score) opt out of ATR stops by passing stop_atr_multiple=None.
            if position.metadata.get("stop_atr_multiple", None) is None:
                position.stop_price = None
                continue
            managed[symbol] = position
        self.risk_manager.update_trailing_stops(managed, atr_now, multiplier=multiplier)

    def _evaluate_preservation(self, i: int, timestamp: pd.Timestamp) -> PreservationAction:
        """Run the capital-preservation overlay for the current bar.

        Args:
            i: Bar position.
            timestamp: Bar timestamp.

        Returns:
            The :class:`PreservationAction` (also applied to queued orders).
        """
        context = StrategyContext(
            timestamp=timestamp,
            bar_index=i,
            data=self.data,
            features=self._features,
            regime_state=self._regime_at(i),
            regime_probabilities=self._probabilities_at(i),
            positions=self._position_snapshots(),
            equity=self.portfolio.equity,
            params=self.params,
        )
        action = self.preservation.evaluate(context)

        if action.liquidation:
            orders = self.risk_manager.liquidation_orders(self.portfolio, action.liquidation)
            for symbol, quantity in orders.items():
                self._pending.append(
                    _PendingOrder(symbol, quantity, "de_risk", "RiskPreservation")
                )
                position = self.portfolio.positions.get(symbol)
                remaining_weight = 0.0
                if position is not None:
                    equity = max(self.portfolio.equity, 1e-12)
                    remaining_weight = (position.quantity + quantity) * position.last_price * position.contract_size / equity
                self._raw_targets[symbol] = remaining_weight
                self._sent_targets[symbol] = remaining_weight
                self._suppressed_until[symbol] = i + int(
                    cfg.DEFAULT_SETTINGS.preservation.cooldown_bars
                )
                for strategy in self.strategies:
                    if symbol in strategy.symbols:
                        strategy.notify_flat(symbol)
            self._events.append(
                {
                    "timestamp": timestamp,
                    "event": "de_risk",
                    "symbol": ",".join(sorted(action.liquidation)),
                    "detail": f"{cfg.DEFAULT_SETTINGS.preservation.de_risk_fraction:.0%} liquidation",
                }
            )
        if action.halt_entries and self.verbose:
            logger.debug("Entries halted at %s (%s)", timestamp, action.reason)
        return action

    def _collect_signals(
        self,
        i: int,
        timestamp: pd.Timestamp,
        state: int,
        action: PreservationAction,
    ) -> None:
        """Gather strategy signals, size them and queue the resulting orders.

        Args:
            i: Bar position.
            timestamp: Bar timestamp.
            state: Active regime id.
            action: Preservation action governing whether entries are allowed.
        """
        context = StrategyContext(
            timestamp=timestamp,
            bar_index=i,
            data=self.data,
            features=self._features,
            regime_state=state,
            regime_probabilities=self._probabilities_at(i),
            positions=self._position_snapshots(),
            equity=self.portfolio.equity,
            params=self.params,
        )

        changed: Dict[str, float] = {}
        for strategy in self.strategies:
            if i < strategy.required_history:
                continue
            if strategy.is_active(state):
                signals = strategy.generate_signals(context)
            else:
                signals = strategy.flat_signals(context)
            for signal in signals:
                changed[signal.symbol] = float(signal.target_weight)
                self._latest_stop_multiple[signal.symbol] = signal.stop_atr_multiple
                if signal.tag:
                    self._events.append(
                        {
                            "timestamp": timestamp,
                            "event": "signal",
                            "symbol": signal.symbol,
                            "detail": f"{strategy.name}:{signal.tag}",
                        }
                    )
        # Size only when the strategy layer actually changed something.  Between
        # signal events the book is allowed to drift with the market, which is
        # both realistic (no needless turnover) and ~10x faster.
        if not changed:
            return

        self._raw_targets.update(changed)

        # --- Kelly estimate from realised trades ------------------------ #
        # The *raw* Kelly fraction is passed through; the fractional-Kelly
        # haircut and the hard cap live inside SizingEngine.
        kelly: Optional[float] = None
        trade_frame = self._cached_trade_frame()
        if len(trade_frame) >= cfg.DEFAULT_SETTINGS.sizing.kelly_min_trades:
            kelly = self._kelly_from_trades(trade_frame)

        sized = self.sizing_engine.size(
            self._raw_targets,
            covariance=self.covariance,
            regime_state=state,
            kelly=kelly,
        )

        # --- Diff against the last sent targets and queue orders -------- #
        band = cfg.DEFAULT_SETTINGS.risk.rebalance_band
        current_weights = self.portfolio.weights()
        for symbol in sorted(set(sized) | set(self._sent_targets)):
            target = float(sized.get(symbol, 0.0))
            sent = float(self._sent_targets.get(symbol, 0.0))
            if abs(target - sent) <= band:
                continue
            if symbol in self._suppressed_until and i <= self._suppressed_until[symbol]:
                continue
            # Entries are halted while State 2 (or its cool-down) is active.
            current = float(current_weights.get(symbol, 0.0))
            increases_exposure = abs(target) > abs(current) + band
            if action.halt_entries and increases_exposure:
                continue

            price = float(self.data[symbol].iloc[i]["close"])
            if price <= 0:
                continue
            try:
                quantity = self.portfolio.target_quantity(symbol, target, price)
            except ValueError:
                continue
            notional = abs(quantity) * price * self.portfolio.position(symbol).contract_size
            if notional < cfg.DEFAULT_SETTINGS.risk.min_order_notional:
                continue
            strategy_name = next(
                (s.name for s in self.strategies if symbol in s.symbols), "engine"
            )

            # --- Daily trade-frequency gate (lifted in demo mode) ------- #
            # Only *openings* (flat -> non-flat) are throttled; adding to or
            # trimming an existing position is always allowed, as are all
            # risk-reducing stops and de-risk flows below.
            open_position = self.portfolio.position(symbol)
            opens_new = (
                abs(open_position.quantity) < 1e-12
                and abs(open_position.quantity + quantity) > 1e-12
            )
            if opens_new:
                allowed, gate_reason = self.portfolio.can_open_trade(timestamp)
                if not allowed:
                    self._blocked_by_limit += 1
                    self._events.append(
                        {
                            "timestamp": timestamp,
                            "event": "trade_blocked",
                            "symbol": symbol,
                            "detail": f"{strategy_name}:{gate_reason}",
                            "target_weight": target,
                        }
                    )
                    continue

            self._pending.append(
                _PendingOrder(symbol, float(quantity), "rebalance", strategy_name)
            )
            self._sent_targets[symbol] = target

    def _cached_trade_frame(self) -> pd.DataFrame:
        """Return the trade log, rebuilding it only when a trade has closed.

        Serialising the trade list to a DataFrame on every bar is O(trades) per
        bar; caching it keyed on the trade count makes the Kelly estimate
        effectively free.

        Returns:
            The cached trade-log DataFrame.
        """
        count = len(self.portfolio.trades)
        if self._trade_frame_cache is None or count != self._trade_frame_count:
            self._trade_frame_cache = self.portfolio.trade_frame()
            self._trade_frame_count = count
        return self._trade_frame_cache

    @staticmethod
    def _kelly_from_trades(trade_frame: pd.DataFrame) -> float:
        """Estimate the Kelly fraction from a trade log.

        Args:
            trade_frame: Trade log DataFrame.

        Returns:
            The Kelly fraction in ``[0, 1]``.
        """
        from quant_system.execution.portfolio import kelly_fraction_from_trades

        return kelly_fraction_from_trades(trade_frame)


    def _halt(self, i: int, timestamp: pd.Timestamp, drawdown: float) -> None:
        """Flatten the whole book after a drawdown breach.

        Args:
            i: Bar position.
            timestamp: Bar timestamp.
            drawdown: Realised drawdown that triggered the halt.
        """
        logger.warning("Drawdown halt triggered at %s (%.2f%%)", timestamp, drawdown * 100.0)
        fractions = {
            symbol: 1.0
            for symbol, position in self.portfolio.positions.items()
            if position.direction != 0
        }
        orders = self.risk_manager.liquidation_orders(self.portfolio, fractions)
        for symbol, quantity in orders.items():
            self._pending.append(_PendingOrder(symbol, quantity, "drawdown_halt", "risk"))
        self._raw_targets = {symbol: 0.0 for symbol in self._raw_targets}
        self._sent_targets = {symbol: 0.0 for symbol in self._sent_targets}
        for strategy in self.strategies:
            for symbol in strategy.symbols:
                strategy.notify_flat(symbol)
        self._events.append(
            {
                "timestamp": timestamp,
                "event": "drawdown_halt",
                "symbol": ",".join(sorted(fractions)),
                "detail": f"{drawdown:.2%} drawdown",
            }
        )

    # ------------------------------------------------------------------ #
    # Execution helpers
    # ------------------------------------------------------------------ #
    def _fill(
        self,
        symbol: str,
        quantity: float,
        reference_price: float,
        timestamp: pd.Timestamp,
        strategy: str,
        tag: str,
    ) -> None:
        """Create and apply a costed fill.

        Args:
            symbol: Instrument symbol.
            quantity: Signed units (positive = buy).
            reference_price: Mid price the costs are applied around.
            timestamp: Bar timestamp.
            strategy: Strategy attribution.
            tag: Reason code.
        """
        if abs(quantity) < 1e-12 or reference_price <= 0:
            return
        try:
            spec = cfg.DEFAULT_SETTINGS.universe.spec(symbol)
        except KeyError:
            spec = None
        spread_pips = spec.spread_pips if spec else 1.5
        slippage_pips = spec.slippage_pips if spec else 0.2
        pip_size = spec.pip_size if spec else 0.0001

        half_spread = 0.5 * spread_pips * pip_size
        slippage = slippage_pips * pip_size
        contract_size = spec.contract_size if spec else 1.0
        commission = (
            abs(quantity) * reference_price * contract_size * self.costs.commission_rate
            + self.costs.commission_fixed
        )
        fill = Fill(
            timestamp=timestamp,
            symbol=symbol,
            quantity=float(quantity),
            price=float(reference_price),
            spread_cost=half_spread,
            slippage_cost=slippage,
            commission=float(commission),
            strategy=strategy,
            tag=tag,
        )
        self.portfolio.apply_fill(fill)
        position = self.portfolio.position(symbol)
        position.metadata["stop_atr_multiple"] = self._latest_stop_multiple.get(symbol)

    def _position_snapshots(self) -> Dict[str, PositionSnapshot]:
        """Build position snapshots for the strategy context.

        Returns:
            Mapping of symbol -> :class:`PositionSnapshot`.
        """
        return {
            symbol: self.portfolio.snapshot(symbol)  # type: ignore[dict-item]
            for symbol, position in self.portfolio.positions.items()
            if abs(position.quantity) > 1e-12
        }

    # ------------------------------------------------------------------ #
    # Result assembly
    # ------------------------------------------------------------------ #
    def _build_result(self) -> BacktestResult:
        """Assemble the final :class:`BacktestResult` with metrics.

        Returns:
            The completed result object.
        """
        equity = self.portfolio.equity_curve
        returns = self.portfolio.returns()
        trades = self.portfolio.trade_frame()
        fills = self.portfolio.fill_frame()
        weights = (
            pd.DataFrame(self._weights_history).T.sort_index()
            if self._weights_history
            else pd.DataFrame()
        )
        if not weights.empty:
            weights = weights.reindex(columns=sorted(self.data.keys())).fillna(0.0)

        if not self.compute_metrics:
            return BacktestResult(
                equity_curve=equity,
                returns=returns,
                trades=trades,
                fills=fills,
                regime_states=self.regime_states.reindex(equity.index).ffill()
                .fillna(cfg.STATE_RANGE_BOUND)
                .astype(int),
                regime_probabilities=pd.DataFrame(),
                position_weights=weights,
            )

        aligned_states = (
            self.regime_states.reindex(equity.index).ffill().fillna(cfg.STATE_RANGE_BOUND)
        )
        summary = ametrics.performance_summary(equity, trades)
        regime_frame = ametrics.regime_breakdown(
            returns, aligned_states.reindex(returns.index).ffill(), trades
        )
        return BacktestResult(
            equity_curve=equity,
            returns=returns,
            trades=trades,
            fills=fills,
            regime_states=aligned_states.astype(int),
            regime_probabilities=(
                self.regime_probabilities.reindex(equity.index).ffill()
                if not self.regime_probabilities.empty
                else pd.DataFrame()
            ),
            position_weights=weights,
            metrics=summary,
            regime_metrics=regime_frame,
            events=self._events,
        )


# --------------------------------------------------------------------------- #
# Vectorized engine
# --------------------------------------------------------------------------- #
class VectorizedBacktester:
    """Fast closed-form backtester for precomputed target-weight frames.

    Given a frame of *target weights* ``W`` (symbols x bars) and the
    corresponding price frame, the strategy return is::

        r_t = sum_s W_{s, t-1} * ret_{s, t} - turnover_t * cost_rate

    where ``W`` is lagged by one bar (decide on the close, earn from the next
    bar) and turnover is the summed absolute weight change.

    Attributes:
        commission_rate: Proportional cost per unit of turnover.
        financing_rate: Annualised financing charged on short exposure.
        spread_cost: Half-spread cost per unit of turnover, as a price fraction.
    """

    def __init__(
        self,
        commission_rate: Optional[float] = None,
        financing_rate: Optional[float] = None,
        spread_cost: Optional[float] = None,
        periods_per_year: float = float(cfg.TRADING_DAYS_PER_YEAR),
    ) -> None:
        """Initialise the vectorized backtester.

        Args:
            commission_rate: Cost per unit of turnover; defaults to
                ``settings.costs.commission_rate``.
            financing_rate: Annualised financing on shorts.
            spread_cost: Half-spread cost per unit of turnover. Defaults to the
                universe-average half-spread.
            periods_per_year: Periods per year for annualisation.
        """
        self.commission_rate: float = float(
            commission_rate if commission_rate is not None
            else cfg.DEFAULT_SETTINGS.costs.commission_rate
        )
        self.financing_rate: float = float(
            financing_rate if financing_rate is not None
            else cfg.DEFAULT_SETTINGS.costs.financing_rate_annual
        )
        self.spread_cost: Optional[float] = (
            None if spread_cost is None else float(spread_cost)
        )
        self.periods_per_year: float = float(periods_per_year)

    def run(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        initial_capital: float = cfg.DEFAULT_SETTINGS.sizing.initial_capital,
    ) -> BacktestResult:
        """Run the vectorized backtest.

        Args:
            weights: Target weights (rows = timestamps, columns = symbols).
            prices: Close prices with the same index/columns.
            initial_capital: Starting equity.

        Returns:
            A :class:`BacktestResult` (no trade-level records; the vectorized
            path models aggregate PnL and costs only).

        Raises:
            ValueError: If the inputs do not share an index.
        """
        aligned_weights = weights.reindex(prices.index).ffill().fillna(0.0)
        returns = pd.DataFrame(
            {symbol: log_returns(prices[symbol].astype(float)) for symbol in prices.columns}
        )
        asset_returns = returns.reindex(columns=aligned_weights.columns).fillna(0.0)

        lagged = aligned_weights.shift(1).fillna(0.0)
        gross = (lagged * asset_returns).sum(axis=1)

        turnover = lagged.diff().abs().fillna(lagged.abs()).sum(axis=1)
        unit_spread_cost = (
            self.spread_cost
            if self.spread_cost is not None
            else self._average_spread_cost(prices.columns)
        )
        spread_cost = turnover * unit_spread_cost
        commission = turnover * self.commission_rate
        short_exposure = lagged.clip(upper=0.0).abs().sum(axis=1)
        financing = short_exposure * self.financing_rate / self.periods_per_year

        net = gross - spread_cost - commission - financing
        equity = initial_capital * (1.0 + net).cumprod()
        equity.iloc[0] = initial_capital

        summary = ametrics.performance_summary(equity, None)
        summary["turnover_total"] = float(turnover.sum())
        summary["avg_turnover"] = float(turnover.mean())
        summary["total_costs_pct"] = float((spread_cost + commission + financing).sum()) * 100.0

        return BacktestResult(
            equity_curve=equity,
            returns=net,
            trades=pd.DataFrame(),
            fills=pd.DataFrame(),
            regime_states=pd.Series(dtype=int),
            regime_probabilities=pd.DataFrame(),
            position_weights=aligned_weights,
            metrics=summary,
            regime_metrics=pd.DataFrame(),
        )

    @staticmethod
    def _average_spread_cost(symbols: Sequence[str]) -> float:
        """Mean round-trip spread cost across the traded symbols.

        Args:
            symbols: Traded symbols.

        Returns:
            Average half-spread cost expressed as a price fraction.
        """
        costs: List[float] = []
        for symbol in symbols:
            try:
                spec = cfg.DEFAULT_SETTINGS.universe.spec(symbol)
            except KeyError:
                continue
            reference = spec.base_price if spec.base_price > 0 else 1.0
            costs.append(0.5 * spec.spread_pips * spec.pip_size / reference)
        return float(np.mean(costs)) if costs else 0.0


def cross_validate_with_backtrader(
    ohlcv: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    initial_capital: float = cfg.DEFAULT_SETTINGS.sizing.initial_capital,
    percent: int = 90,
) -> pd.DataFrame:
    """Run an independent SMA-crossover backtest in **backtrader**.

    ``backtrader`` is a mandated dependency and is used here for what it is good
    at: an *independent* second opinion.  Because it is a third-party engine with
    its own broker, data feed and order machinery, agreement between it and
    :class:`VectorizedBacktester` on the same signal is meaningful evidence that
    our PnL algebra is correct - a shared bug is far less likely than a shared
    result.

    Args:
        ohlcv: OHLCV frame for a single symbol.
        fast_period: Fast SMA lookback.
        slow_period: Slow SMA lookback.
        initial_capital: Starting cash.
        percent: Percentage of available cash committed on each entry. Using a
            percent-of-equity sizer (rather than a fixed stake) keeps backtrader's
            exposure profile comparable to :class:`VectorizedBacktester`, which
            rebalances to its target weight on every bar.

    Returns:
        DataFrame with ``equity`` indexed by timestamp, and ``returns``.

    Raises:
        ImportError: If ``backtrader`` is not installed.
    """
    try:
        import backtrader as bt  # noqa: PLC0415 - mandated optional dependency
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "backtrader is required for cross-validation; install it with "
            "`pip install backtrader`."
        ) from exc

    class _ValueAnalyzer(bt.Analyzer):
        """Records broker equity on every bar."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialise the recorder.

            Args:
                *args: Forwarded to the backtrader base class.
                **kwargs: Forwarded (``cerebro`` injects ``name``).
            """
            # backtrader's metaclass injects ``name``; the base Analyzer has no
            # argument-taking __init__, so the kwargs are simply absorbed.
            del args, kwargs
            self.equity: List[float] = []
            self.timestamps: List[pd.Timestamp] = []

        def next(self) -> None:
            """Append the current account value."""
            self.equity.append(float(self.strategy.broker.getvalue()))
            self.timestamps.append(pd.Timestamp(self.strategy.data.datetime.datetime(0)))

    class _SmaCross(bt.SignalStrategy):
        """Long/flat on a fast-over-slow SMA crossover."""

        def __init__(self) -> None:
            """Attach the crossover signal."""
            fast = bt.ind.SMA(self.data.close, period=fast_period)
            slow = bt.ind.SMA(self.data.close, period=slow_period)
            self.signal_add(bt.SIGNAL_LONG, bt.ind.CrossOver(fast, slow))

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(_SmaCross)
    cerebro.adddata(bt.feeds.PandasData(dataname=ohlcv.astype(float)))
    cerebro.broker.setcash(float(initial_capital))
    cerebro.broker.setcommission(commission=0.0)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=int(percent))
    cerebro.addanalyzer(_ValueAnalyzer, _name="value")

    strategy = cerebro.run()[0]
    analyzer = strategy.analyzers.value
    equity = pd.Series(analyzer.equity, index=pd.DatetimeIndex(analyzer.timestamps), name="equity")
    frame = equity.to_frame()
    frame["returns"] = frame["equity"].pct_change().fillna(0.0)
    return frame


def compare_engines_on_sma_cross(
    ohlcv: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    initial_capital: float = cfg.DEFAULT_SETTINGS.sizing.initial_capital,
) -> Dict[str, float]:
    """Cross-check our vectorized engine against backtrader on one signal.

    Both engines trade the same long/flat SMA-crossover signal: we build the
    target weights explicitly (``1.0`` when fast > slow, else ``0.0``) and price
    them with :class:`VectorizedBacktester` at zero cost; backtrader runs its own
    broker.  The reported diagnostics are the terminal returns and the
    correlation of the two return streams.

    Args:
        ohlcv: OHLCV frame for a single symbol.
        fast_period: Fast SMA lookback.
        slow_period: Slow SMA lookback.
        initial_capital: Starting capital.

    Returns:
        Dictionary with ``backtrader_total_return``, ``vectorized_total_return``,
        ``return_correlation`` and ``exposure``.
    """
    reference = cross_validate_with_backtrader(
        ohlcv, fast_period, slow_period, initial_capital
    )

    close = ohlcv["close"].astype(float)
    fast = close.rolling(window=fast_period, min_periods=fast_period).mean()
    slow = close.rolling(window=slow_period, min_periods=slow_period).mean()
    signal = (fast > slow).astype(float).where(slow.notna(), 0.0)

    weights = signal.to_frame(name=ohlcv.attrs.get("symbol", "ASSET"))
    prices = close.to_frame(name=ohlcv.attrs.get("symbol", "ASSET"))
    ours = VectorizedBacktester(
        commission_rate=0.0, financing_rate=0.0, spread_cost=0.0
    ).run(weights, prices, initial_capital=initial_capital)

    bt_returns = reference["returns"].reindex(ours.returns.index).fillna(0.0)
    correlation = float(np.corrcoef(bt_returns.to_numpy(), ours.returns.to_numpy())[0, 1])
    return {
        "backtrader_total_return": float(
            reference["equity"].iloc[-1] / reference["equity"].iloc[0] - 1.0
        ),
        "vectorized_total_return": float(ours.metrics.get("total_return_pct", 0.0) / 100.0),
        "return_correlation": correlation,
        "exposure": float(signal.mean()),
    }


def build_default_regimes(
    data: Mapping[str, pd.DataFrame],
    config: Optional[cfg.HMMConfig] = None,
    interval: str = "1d",
    verbose: bool = False,
) -> RegimeStreamResult:
    """Compute the causal regime path for a universe.

    Args:
        data: Mapping of symbol -> OHLCV frame.
        config: HMM configuration.
        interval: Bar interval (used for volatility annualisation).
        verbose: Streamer progress logging.

    Returns:
        The :class:`RegimeStreamResult`.
    """
    builder = FeatureBuilder(config=config, interval=interval)
    features = builder.market_features(data)
    streamer = CausalRegimeStreamer(config=config, verbose=verbose)
    return streamer.run(features)


__all__: List[str] = [
    "BacktestResult",
    "BacktestEngine",
    "VectorizedBacktester",
    "build_default_regimes",
    "cross_validate_with_backtrader",
    "compare_engines_on_sma_cross",
]
