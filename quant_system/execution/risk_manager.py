"""Dynamic position sizing, ATR trailing stops and portfolio-level circuit breakers.

The risk manager is deliberately *stateless with respect to alpha*: it never
decides direction, only how much of a signal to fund and when an existing
position must be closed.

Responsibilities
----------------
* **ATR trailing stops** - ``stop = close -/+ k * ATR`` ratcheted in the
  favourable direction only; ``k`` is ``2.5`` by default and multiplied by
  ``shock_stop_multiplier`` (0.5) while the HMM is in State 2, which is the
  mechanical implementation of "tighten stops by 50 %".
* **Intrabar stop surveillance** - a stop is evaluated against the bar's
  ``high``/``low`` and filled at ``min(open, stop)`` / ``max(open, stop)`` so
  overnight gaps are handled conservatively.
* **ATR unit sizing** - ``units = (equity * risk_pct) / (k * ATR * contract_size)``
  so a stopped-out position loses approximately ``risk_pct`` of equity.
* **Drawdown circuit breaker** - flattens the book if portfolio drawdown exceeds
  ``max_drawdown_halt``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from quant_system.config import settings as cfg
from quant_system.execution.portfolio import Portfolio, Position

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StopUpdate:
    """Record of a trailing-stop adjustment.

    Attributes:
        symbol: Instrument symbol.
        old_stop: Previous stop price (``None`` if the stop was just armed).
        new_stop: New stop price.
        atr: ATR used for the calculation.
        multiple: Effective ATR multiple after regime adjustment.
    """

    symbol: str
    old_stop: Optional[float]
    new_stop: Optional[float]
    atr: float
    multiple: float


@dataclass(frozen=True)
class StopTrigger:
    """A stop that has been breached and must be executed.

    Attributes:
        symbol: Instrument symbol.
        direction: Direction of the position being closed (``+1``/``-1``).
        stop_price: The stop level that was breached.
        fill_price: Conservative fill price (gap-aware).
        reason: ``"atr_trailing_stop"``.
    """

    symbol: str
    direction: int
    stop_price: float
    fill_price: float
    reason: str = "atr_trailing_stop"


class RiskManager:
    """Executes stop management and exposure limits for the book.

    Attributes:
        config: Risk configuration.
        atr_period: Wilder ATR lookback used for stop distances.
        halted: Set when the drawdown circuit breaker has fired.
    """

    def __init__(
        self,
        config: Optional[cfg.RiskConfig] = None,
        atr_period: Optional[int] = None,
    ) -> None:
        """Initialise the risk manager.

        Args:
            config: Risk configuration; defaults to ``settings.risk``.
            atr_period: ATR lookback; defaults to ``settings.hmm.atr_period``.
        """
        self.config: cfg.RiskConfig = config or cfg.DEFAULT_SETTINGS.risk
        self.atr_period: int = int(atr_period or cfg.DEFAULT_SETTINGS.hmm.atr_period)
        self.halted: bool = False
        self.halt_reason: str = ""

    # ------------------------------------------------------------------ #
    # Stop management
    # ------------------------------------------------------------------ #
    def arm_stop(
        self,
        position: Position,
        atr: float,
        multiple: Optional[float] = None,
        reference_price: Optional[float] = None,
    ) -> None:
        """Arm an initial ATR stop for a freshly opened position.

        Args:
            position: The position to protect.
            atr: Current ATR value.
            multiple: ATR multiple; defaults to ``config.atr_stop_multiple``.
            reference_price: Price the stop is measured from; defaults to the
                position's last mark.
        """
        if position.direction == 0 or not np.isfinite(atr) or atr <= 0:
            return
        multiple = float(multiple or self.config.atr_stop_multiple)
        price = float(reference_price if reference_price else position.last_price)
        if position.direction > 0:
            position.stop_price = price - multiple * atr
        else:
            position.stop_price = price + multiple * atr
        position.stop_atr_multiple = multiple

    def update_trailing_stops(
        self,
        positions: Mapping[str, Position],
        atr_by_symbol: Mapping[str, float],
        multiplier: float = 1.0,
    ) -> List[StopUpdate]:
        """Ratchet ATR trailing stops in the favourable direction only.

        Args:
            positions: Open positions keyed by symbol.
            atr_by_symbol: Current ATR per symbol.
            multiplier: Regime adjustment (``0.5`` in State 2 = 50 % tighter).

        Returns:
            List of :class:`StopUpdate` records describing every change.
        """
        updates: List[StopUpdate] = []
        for symbol, position in positions.items():
            if position.direction == 0:
                continue
            atr = float(atr_by_symbol.get(symbol, float("nan")))
            if not np.isfinite(atr) or atr <= 0:
                continue
            multiple = float(position.stop_atr_multiple or self.config.atr_stop_multiple)
            effective = multiple * float(multiplier)
            price = position.last_price
            if position.direction > 0:
                candidate = price - effective * atr
                new_stop = (
                    max(position.stop_price, candidate)
                    if position.stop_price is not None
                    else candidate
                )
            else:
                candidate = price + effective * atr
                new_stop = (
                    min(position.stop_price, candidate)
                    if position.stop_price is not None
                    else candidate
                )
            if position.stop_price is None or abs(new_stop - position.stop_price) > 1e-12:
                updates.append(
                    StopUpdate(
                        symbol=symbol,
                        old_stop=position.stop_price,
                        new_stop=float(new_stop),
                        atr=atr,
                        multiple=effective,
                    )
                )
                position.stop_price = float(new_stop)
                position.stop_atr_multiple = multiple
        return updates

    def check_stops(
        self,
        positions: Mapping[str, Position],
        open_prices: Mapping[str, float],
        high_prices: Mapping[str, float],
        low_prices: Mapping[str, float],
    ) -> List[StopTrigger]:
        """Detect intrabar stop breaches.

        Args:
            positions: Open positions keyed by symbol.
            open_prices: Bar open per symbol.
            high_prices: Bar high per symbol.
            low_prices: Bar low per symbol.

        Returns:
            List of :class:`StopTrigger` for every breached stop.
        """
        triggers: List[StopTrigger] = []
        for symbol, position in positions.items():
            if position.direction == 0 or position.stop_price is None:
                continue
            low = float(low_prices.get(symbol, np.nan))
            high = float(high_prices.get(symbol, np.nan))
            open_ = float(open_prices.get(symbol, np.nan))
            if not np.isfinite(low) or not np.isfinite(high):
                continue
            stop = float(position.stop_price)
            if position.direction > 0 and low <= stop:
                fill = min(open_, stop) if np.isfinite(open_) else stop
                triggers.append(
                    StopTrigger(symbol, 1, stop, float(fill))
                )
            elif position.direction < 0 and high >= stop:
                fill = max(open_, stop) if np.isfinite(open_) else stop
                triggers.append(
                    StopTrigger(symbol, -1, stop, float(fill))
                )
        return triggers

    # ------------------------------------------------------------------ #
    # Sizing primitives
    # ------------------------------------------------------------------ #
    @staticmethod
    def atr_position_size(
        equity: float,
        price: float,
        atr: float,
        risk_per_unit_pct: float,
        stop_multiple: float,
        contract_size: float = 1.0,
    ) -> float:
        """Units such that a full stop-out costs ``risk_per_unit_pct`` of equity.

        ``units = (equity * risk_pct) / (stop_multiple * ATR * contract_size)``

        Args:
            equity: Account equity.
            price: Instrument price (used only for a sanity bound).
            atr: Current ATR.
            risk_per_unit_pct: Fraction of equity to risk per unit.
            stop_multiple: ATR multiple of the stop distance.
            contract_size: Notional multiplier.

        Returns:
            Absolute number of units (always non-negative).

        Raises:
            ValueError: If ``atr`` or ``price`` is non-positive.
        """
        if atr <= 0 or price <= 0:
            raise ValueError("atr and price must be positive.")
        risk_capital = equity * risk_per_unit_pct
        risk_per_unit = stop_multiple * atr * contract_size
        if risk_per_unit <= 0:
            return 0.0
        units = risk_capital / risk_per_unit
        # Never risk more than 100% of equity via a single instrument.
        max_units = (equity * 0.5) / max(price * contract_size, 1e-12)
        return float(max(0.0, min(units, max_units)))

    def drawdown_breach(self, portfolio: Portfolio) -> Tuple[bool, float]:
        """Check the portfolio drawdown circuit breaker.

        Args:
            portfolio: The portfolio to inspect.

        Returns:
            Tuple ``(breached, drawdown)``.
        """
        drawdown = portfolio.drawdown
        threshold = self.config.max_drawdown_halt
        if threshold <= 0:
            return False, drawdown
        breached = drawdown >= threshold
        if breached and not self.halted:
            self.halted = True
            self.halt_reason = (
                f"drawdown {drawdown:.2%} exceeded limit {threshold:.2%}"
            )
            logger.warning("Risk manager halted the book: %s", self.halt_reason)
        return breached, drawdown

    def liquidation_orders(
        self,
        portfolio: Portfolio,
        fraction_by_symbol: Mapping[str, float],
    ) -> Dict[str, float]:
        """Convert liquidation fractions into order quantities.

        Args:
            portfolio: The portfolio holding the positions.
            fraction_by_symbol: Mapping of symbol -> fraction to close (0..1).

        Returns:
            Mapping of symbol -> signed order quantity (opposite of the position).
        """
        orders: Dict[str, float] = {}
        for symbol, fraction in fraction_by_symbol.items():
            position = portfolio.positions.get(symbol)
            if position is None or position.direction == 0:
                continue
            fraction = float(np.clip(fraction, 0.0, 1.0))
            if fraction <= 0:
                continue
            quantity = -(position.quantity * fraction)
            if abs(quantity) > 1e-12:
                orders[symbol] = float(quantity)
        return orders

    def enforce_exposure_limits(
        self,
        weights: Mapping[str, float],
    ) -> Dict[str, float]:
        """Clip weights to the configured per-symbol and aggregate caps.

        Args:
            weights: Signed target weights.

        Returns:
            Clipped weights.
        """
        capped = {
            symbol: float(
                np.clip(weight, -self.config.max_symbol_weight, self.config.max_symbol_weight)
            )
            for symbol, weight in weights.items()
        }
        gross = sum(abs(w) for w in capped.values())
        max_gross = cfg.DEFAULT_SETTINGS.sizing.max_gross_leverage
        if gross > max_gross and gross > 0:
            scale = max_gross / gross
            capped = {symbol: weight * scale for symbol, weight in capped.items()}
        return capped

    def reset(self) -> None:
        """Clear the circuit-breaker state."""
        self.halted = False
        self.halt_reason = ""


__all__: List[str] = [
    "StopUpdate",
    "StopTrigger",
    "RiskManager",
]
