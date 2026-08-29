"""Abstract strategy interface shared by every alpha module.

Design contract
---------------
* A strategy is a **pure function of past and present data**.  Implementations
  receive the *complete* OHLCV/feature frames plus the current integer bar
  position and must only read ``.iloc[: bar_index + 1]`` (or the precomputed
  indicator row at ``bar_index``).  Anything else introduces look-ahead bias.
* Expensive indicator work happens once in :meth:`BaseStrategy.prepare`; the
  per-bar :meth:`BaseStrategy.generate_signals` is expected to be O(1).
* A strategy declares the regime ids in which it is allowed to trade (``active_states``).
  The engine asks inactive strategies for flattening signals so positions do not
  linger once their regime has passed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, FrozenSet, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__: List[str] = [
    "PositionSnapshot",
    "Signal",
    "StrategyContext",
    "BaseStrategy",
]


@dataclass(frozen=True)
class PositionSnapshot:
    """Immutable view of an open position handed to strategies.

    Attributes:
        symbol: Instrument symbol.
        quantity: Signed units held (negative = short).
        average_price: Volume-weighted average entry price.
        last_price: Latest mark price.
        unrealized_pnl: Mark-to-market PnL in quote currency.
        strategy: Name of the strategy that opened the position.
        stop_price: Currently armed stop price (``None`` if unarmed).
    """

    symbol: str
    quantity: float
    average_price: float
    last_price: float
    unrealized_pnl: float = 0.0
    strategy: str = ""
    stop_price: Optional[float] = None


@dataclass(frozen=True)
class Signal:
    """A target-exposure instruction emitted by a strategy.

    Attributes:
        symbol: Instrument to trade.
        direction: ``+1`` long, ``-1`` short, ``0`` flat.
        target_weight: Signed target exposure as a fraction of equity
            (e.g. ``-0.15`` = 15% of equity short).  Fractional values are the
            norm; the sizing layer may scale them.
        strategy: Emitting strategy name.
        timestamp: Bar timestamp the signal refers to.
        stop_atr_multiple: ATR multiple for the trailing stop; ``None`` disables
            the ATR stop for this signal (used by mean-reversion strategies that
            exit on a statistical trigger instead).
        tag: Free-form reason code for the trade log.
        metadata: Diagnostics (z-score, hedge ratio, indicator values, ...).
    """

    symbol: str
    direction: int
    target_weight: float
    strategy: str
    timestamp: Optional[pd.Timestamp] = None
    stop_atr_multiple: Optional[float] = None
    tag: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalise the signal.

        Raises:
            ValueError: If ``direction`` is outside ``{-1, 0, 1}`` or the sign of
                ``target_weight`` disagrees with ``direction``.
        """
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"direction must be -1, 0 or 1; got {self.direction}.")
        if self.direction == 0 and abs(self.target_weight) > 1e-12:
            raise ValueError("A flat signal must carry a zero target weight.")
        if self.direction != 0 and np.sign(self.target_weight) != np.sign(self.direction):
            raise ValueError(
                "target_weight sign must match direction "
                f"(direction={self.direction}, weight={self.target_weight})."
            )

    @property
    def is_flat(self) -> bool:
        """Whether the signal instructs the book to be flat in ``symbol``."""
        return self.direction == 0

    @classmethod
    def flat(
        cls,
        symbol: str,
        strategy: str,
        timestamp: Optional[pd.Timestamp] = None,
        tag: str = "flatten",
    ) -> "Signal":
        """Build a flattening signal.

        Args:
            symbol: Instrument to flatten.
            strategy: Emitting strategy name.
            timestamp: Bar timestamp.
            tag: Reason code.

        Returns:
            A zero-weight :class:`Signal`.
        """
        return cls(
            symbol=symbol,
            direction=0,
            target_weight=0.0,
            strategy=strategy,
            timestamp=timestamp,
            tag=tag,
        )


@dataclass
class StrategyContext:
    """Everything a strategy is allowed to observe at a given bar.

    Attributes:
        timestamp: Current bar timestamp.
        bar_index: Integer position of the current bar in the engine's index.
            Strategies must only read rows at positions ``<= bar_index``.
        data: Mapping of symbol -> complete OHLCV frame.
        features: Mapping of symbol -> complete feature frame (``ret``/``atr``/``sigma``).
        regime_state: Active canonical HMM regime id.
        regime_probabilities: Canonical probability vector ``[p0, p1, p2]``.
        positions: Mapping of symbol -> current position snapshot.
        equity: Current account equity.
        params: Ad-hoc parameter overrides supplied by the optimiser.
    """

    timestamp: pd.Timestamp
    bar_index: int
    data: Mapping[str, pd.DataFrame]
    features: Mapping[str, pd.DataFrame]
    regime_state: int
    regime_probabilities: np.ndarray
    positions: Mapping[str, PositionSnapshot] = field(default_factory=dict)
    equity: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    def param(self, key: str, default: Any) -> Any:
        """Read a parameter override.

        Args:
            key: Parameter name.
            default: Value returned when the key is absent.

        Returns:
            The override if present, else ``default``.
        """
        return self.params.get(key, default)


class BaseStrategy(ABC):
    """Abstract base class for all strategies.

    Attributes:
        name: Human-readable strategy name used in logs and trade tags.
        symbols: Instruments the strategy trades.
        active_states: Regime ids in which the strategy may open positions.
        required_history: Minimum bars of history required before signals are
            emitted (the engine suppresses signals before this point).
    """

    name: ClassVar[str] = "BaseStrategy"
    active_states: ClassVar[FrozenSet[int]] = frozenset({0, 1, 2})
    required_history: ClassVar[int] = 1
    symbols: tuple[str, ...]

    def __init__(
        self,
        symbols: Sequence[str],
        active_states: Optional[FrozenSet[int]] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialise the strategy.

        Args:
            symbols: Instruments traded by this strategy.
            active_states: Regimes in which entries are allowed; defaults to all.
            name: Override for the class-level ``name``.
        """
        self.symbols: tuple[str, ...] = tuple(symbols)
        if active_states is not None:
            self.active_states = frozenset(active_states)
        if name is not None:
            self.name = name
        self.prepared: bool = False
        self._index: Optional[pd.DatetimeIndex] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def prepare(
        self,
        data: Mapping[str, pd.DataFrame],
        index: pd.DatetimeIndex,
    ) -> None:
        """Precompute indicators once for the whole run.

        The default implementation only records the engine index.  Subclasses
        override it (calling ``super().prepare``) to build indicator frames that
        are reindexed onto ``index``.

        Args:
            data: Mapping of symbol -> OHLCV frame.
            index: The engine's master bar index.
        """
        self._index = pd.DatetimeIndex(index)
        self.prepared = True

    def reset(self) -> None:
        """Clear all internal state so the strategy can be re-run."""
        self.prepared = False
        self._index = None

    def reset_runtime_state(self) -> None:
        """Clear only the *mutable trading state*, keeping prepared indicators.

        This lets a parameter sweep reuse the expensive indicator preparation
        (rolling regressions, cointegration tests, dynamic EMAs) while still
        replaying the state machine from a clean slate for every parameter set.
        The default implementation is a no-op for stateless strategies.
        """

    @abstractmethod
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """Emit the desired target exposures for the current bar.

        Args:
            context: Current-bar market/account context.

        Returns:
            List of :class:`Signal` objects. An empty list means "no change".
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def is_active(self, regime_state: int) -> bool:
        """Whether the strategy may open positions in ``regime_state``.

        Args:
            regime_state: Canonical regime id.

        Returns:
            ``True`` if entries are permitted.
        """
        return regime_state in self.active_states

    def notify_flat(self, symbol: str) -> None:
        """Inform the strategy that a position was closed outside its control.

        Called by the engine after an ATR stop-out, a risk-preservation
        liquidation or a drawdown circuit-breaker flattening.  Strategies that
        keep internal state machines must reset the relevant state here,
        otherwise they would believe a position is still open.

        Args:
            symbol: Instrument that was flattened.
        """

    def flat_signals(self, context: StrategyContext) -> List[Signal]:
        """Emit flattening signals for every symbol with an open position.

        Args:
            context: Current-bar context.

        Returns:
            List of flat signals (empty if the strategy holds nothing).
        """
        return [
            Signal.flat(symbol, self.name, context.timestamp, tag="regime_exit")
            for symbol in self.symbols
            if symbol in context.positions and abs(context.positions[symbol].quantity) > 1e-12
        ]

    @property
    def index(self) -> Optional[pd.DatetimeIndex]:
        """The engine index the strategy was prepared against."""
        return self._index

    def __repr__(self) -> str:
        """Return a concise developer representation."""
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"symbols={self.symbols}, active_states={sorted(self.active_states)})"
        )
