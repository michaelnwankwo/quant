"""Capital-preservation engine - active in **State 2** (market shock).

The overlay does not generate alpha; it *governs* the rest of the book.  The
moment the HMM switchboard emits State 2 the overlay:

1. **Halts all new entry orders** across every strategy (only risk-reducing
   orders are allowed through).
2. **Tightens ATR trailing stops by 50 %** - e.g. a ``2.5 x ATR`` stop becomes
   ``1.25 x ATR`` - so survivors are taken out quickly if the shock persists.
3. **Partially liquidates open positions** (configurable, default 60 %, inside
   the specification's 50 %-70 % band) on the transition into the state, so the
   book de-risks in a single decisive step rather than bleeding out.

A configurable cool-down keeps entries halted for a few bars after the regime
reverts, which prevents re-entering into the tail of a volatility burst.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from quant_system.config import settings as cfg
from quant_system.strategies.base import BaseStrategy, Signal, StrategyContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreservationAction:
    """Instruction bundle produced by the preservation overlay.

    Attributes:
        halt_entries: When ``True`` the engine must drop every risk-increasing
            order for the current bar.
        stop_multiplier: Factor applied to ATR trailing-stop distances
            (``0.5`` = stops tightened by 50 %).
        liquidation: Mapping of symbol -> fraction of the position to close
            (``0.6`` = liquidate 60 % of the holding).
        reason: Human-readable explanation for the audit log.
        regime_state: Regime id the action was derived from.
        is_transition: ``True`` when the bar marks the entry into State 2.
    """

    halt_entries: bool = False
    stop_multiplier: float = 1.0
    liquidation: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    regime_state: int = cfg.STATE_RANGE_BOUND
    is_transition: bool = False

    @property
    def is_active(self) -> bool:
        """Whether the overlay is constraining the book at all."""
        return self.halt_entries or self.stop_multiplier < 1.0 or bool(self.liquidation)


class RiskPreservationStrategy(BaseStrategy):
    """Regime-conditional capital-preservation overlay.

    Attributes:
        config: Preservation hyper-parameters.
        last_action: The most recent action emitted (useful for logging/tests).
    """

    name: str = "RiskPreservation"
    active_states: frozenset[int] = frozenset({cfg.STATE_SHOCK})

    def __init__(
        self,
        symbols: Optional[Sequence[str]] = None,
        config: Optional[cfg.RiskPreservationConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialise the overlay.

        Args:
            symbols: Instruments the overlay may act on; defaults to the universe.
            config: Preservation hyper-parameters.
            name: Strategy name override.
        """
        super().__init__(
            symbols=list(symbols) if symbols else list(cfg.DEFAULT_SETTINGS.symbols),
            name=name,
        )
        self.config: cfg.RiskPreservationConfig = (
            config or cfg.DEFAULT_SETTINGS.preservation
        )
        self._previous_state: Optional[int] = None
        self._cooldown_remaining: int = 0
        self.last_action: PreservationAction = PreservationAction(reason="initialised")

    # ------------------------------------------------------------------ #
    # Overlay evaluation (primary API used by the engine)
    # ------------------------------------------------------------------ #
    def evaluate(self, context: StrategyContext) -> PreservationAction:
        """Derive the preservation action for the current bar.

        Args:
            context: Current-bar context (regime, positions, equity).

        Returns:
            The :class:`PreservationAction` to apply.
        """
        state = int(context.regime_state)
        is_transition = state == cfg.STATE_SHOCK and self._previous_state != cfg.STATE_SHOCK
        in_cooldown = self._cooldown_remaining > 0

        if state == cfg.STATE_SHOCK:
            liquidation: Dict[str, float] = {}
            if is_transition and self.config.de_risk_fraction > 0:
                liquidation = {
                    symbol: float(self.config.de_risk_fraction)
                    for symbol, snapshot in context.positions.items()
                    if abs(snapshot.quantity) > 1e-12
                }
                if liquidation:
                    logger.info(
                        "State 2 detected at %s - de-risking %.0f%% of %d position(s).",
                        context.timestamp,
                        100.0 * self.config.de_risk_fraction,
                        len(liquidation),
                    )
            self._cooldown_remaining = int(self.config.cooldown_bars)
            action = PreservationAction(
                halt_entries=bool(self.config.halt_entries),
                stop_multiplier=float(self.config.stop_tightening_factor),
                liquidation=liquidation,
                reason="regime_state_2_shock",
                regime_state=state,
                is_transition=is_transition,
            )
        elif in_cooldown:
            self._cooldown_remaining -= 1
            action = PreservationAction(
                halt_entries=bool(self.config.halt_entries),
                stop_multiplier=float(self.config.stop_tightening_factor),
                liquidation={},
                reason=f"post_shock_cooldown({self._cooldown_remaining} bars left)",
                regime_state=state,
                is_transition=False,
            )
        else:
            action = PreservationAction(
                halt_entries=False,
                stop_multiplier=1.0,
                liquidation={},
                reason="normal_conditions",
                regime_state=state,
                is_transition=False,
            )

        self._previous_state = state
        self.last_action = action
        return action

    # ------------------------------------------------------------------ #
    # Signal interface (alternative wiring: de-risk via target weights)
    # ------------------------------------------------------------------ #
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """Emit scaled-down target weights while State 2 is active.

        This is the *declarative* counterpart of :meth:`evaluate`: instead of
        partial-fill orders the strategy simply re-states each target weight
        reduced by ``de_risk_fraction``.  The engine uses :meth:`evaluate` (which
        produces explicit liquidation fractions) so that the reduction is
        executed once, on the transition bar.

        Args:
            context: Current-bar context.

        Returns:
            List of :class:`Signal` objects, or an empty list.
        """
        action = self.evaluate(context)
        if not action.halt_entries or context.equity <= 0:
            return []
        signals: List[Signal] = []
        keep = 1.0 - self.config.de_risk_fraction
        for symbol, snapshot in context.positions.items():
            if abs(snapshot.quantity) <= 1e-12:
                continue
            weight = self._current_weight(symbol, snapshot, context)
            reduced = weight * keep
            if abs(reduced - weight) < cfg.DEFAULT_SETTINGS.risk.rebalance_band:
                continue
            signals.append(
                Signal(
                    symbol=symbol,
                    direction=int(np.sign(reduced)) if abs(reduced) > 1e-12 else 0,
                    target_weight=reduced,
                    strategy=self.name,
                    timestamp=context.timestamp,
                    stop_atr_multiple=None,
                    tag="de_risk",
                    metadata={"action": action.reason, "keep_fraction": keep},
                )
            )
        return signals

    @staticmethod
    def _current_weight(
        symbol: str, snapshot: object, context: StrategyContext
    ) -> float:
        """Convert a position snapshot into a weight of equity.

        Args:
            symbol: Instrument symbol.
            snapshot: :class:`~quant_system.strategies.base.PositionSnapshot`.
            context: Current-bar context supplying equity.

        Returns:
            Signed exposure as a fraction of equity.
        """
        quantity = float(getattr(snapshot, "quantity", 0.0))
        price = float(getattr(snapshot, "last_price", 0.0))
        try:
            contract_size = cfg.DEFAULT_SETTINGS.universe.spec(symbol).contract_size
        except KeyError:
            contract_size = 1.0
        if context.equity <= 0:
            return 0.0
        return quantity * price * contract_size / context.equity

    def reset(self) -> None:
        """Clear the overlay's regime memory."""
        super().reset()
        self._previous_state = None
        self._cooldown_remaining = 0
        self.last_action = PreservationAction(reason="reset")


__all__: List[str] = ["PreservationAction", "RiskPreservationStrategy"]
