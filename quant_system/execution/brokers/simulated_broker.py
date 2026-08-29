"""In-process simulated broker.

The adapter wraps a :class:`~quant_system.execution.portfolio.Portfolio` and
applies the same spread / slippage / commission model as the backtester, so a
"live" dry run produces PnL that is directly comparable to the simulation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional

import pandas as pd

from quant_system.config import settings as cfg
from quant_system.execution.brokers.base import (
    BrokerAccount,
    BrokerBase,
    Order,
    OrderReport,
    OrderSide,
    OrderStatus,
    PositionReport,
)
from quant_system.execution.portfolio import Fill, Portfolio

logger = logging.getLogger(__name__)


class SimulatedBroker(BrokerBase):
    """Fills orders instantly against supplied reference prices.

    Attributes:
        portfolio: The portfolio that records fills and PnL.
        prices: Latest reference prices (mid) keyed by symbol.
    """

    name: str = "simulated"

    def __init__(
        self,
        portfolio: Optional[Portfolio] = None,
        prices: Optional[Mapping[str, float]] = None,
        config: Optional[cfg.BrokerConfig] = None,
    ) -> None:
        """Initialise the simulated broker.

        Args:
            portfolio: Portfolio to book fills into; a new one is created if omitted.
            prices: Initial reference prices.
            config: Broker configuration.
        """
        super().__init__(config or cfg.DEFAULT_SETTINGS.broker)
        self.portfolio: Portfolio = portfolio or Portfolio()
        self.prices: Dict[str, float] = dict(prices or {})
        self.reports: List[OrderReport] = []

    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Mark the session as connected."""
        self._connected = True
        logger.info("Simulated broker connected.")

    def disconnect(self) -> None:
        """Mark the session as disconnected."""
        self._connected = False

    def update_prices(self, prices: Mapping[str, float]) -> None:
        """Refresh the reference prices used to fill subsequent orders.

        Args:
            prices: Mapping of symbol -> mid price.
        """
        self.prices.update({k: float(v) for k, v in prices.items()})

    # ------------------------------------------------------------------ #
    def submit_order(self, order: Order) -> OrderReport:
        """Fill an order immediately at mid +/- half-spread plus slippage.

        Args:
            order: The order to fill.

        Returns:
            The resulting :class:`OrderReport`.
        """
        self._require_connection()
        price = self.prices.get(order.symbol)
        if price is None or price <= 0:
            return OrderReport(
                client_order_id=order.client_order_id,
                status=OrderStatus.REJECTED,
                message=f"No reference price for {order.symbol}.",
            )

        spec = self._spec(order.symbol)
        half_spread = 0.5 * spec.spread_pips * spec.pip_size
        slippage = spec.slippage_pips * spec.pip_size
        # The reference (mid) price is booked as the fill price; spread and
        # slippage are carried as explicit costs so they are not double-counted
        # in ``Portfolio.apply_fill``.
        fill_price = float(price)

        signed_qty = (
            order.quantity if order.side == OrderSide.BUY else -order.quantity
        )
        commission = abs(order.quantity) * fill_price * spec.contract_size * cfg.DEFAULT_SETTINGS.costs.commission_rate
        fill = Fill(
            timestamp=pd.Timestamp.utcnow().tz_localize(None),
            symbol=order.symbol,
            quantity=signed_qty,
            price=float(fill_price),
            spread_cost=half_spread,
            slippage_cost=slippage,
            commission=commission,
            strategy=order.strategy,
            tag=order.tag,
        )
        self.portfolio.apply_fill(fill)
        report = OrderReport(
            client_order_id=order.client_order_id,
            broker_order_id=f"SIM-{order.client_order_id}",
            status=OrderStatus.FILLED,
            filled_quantity=float(order.quantity),
            average_fill_price=float(fill_price),
            message="simulated fill",
        )
        self.reports.append(report)
        return report

    def cancel_order(self, client_order_id: str) -> OrderReport:
        """Cancel (no-op: simulated orders fill synchronously).

        Args:
            client_order_id: The order id.

        Returns:
            A cancelled :class:`OrderReport`.
        """
        return OrderReport(
            client_order_id=client_order_id,
            status=OrderStatus.CANCELLED,
            message="simulated orders fill immediately; nothing to cancel",
        )

    def get_positions(self) -> List[PositionReport]:
        """Return the portfolio's open positions.

        Returns:
            List of :class:`PositionReport`.
        """
        return [
            PositionReport(
                symbol=symbol,
                quantity=position.quantity,
                average_price=position.average_price,
                last_price=position.last_price,
                unrealized_pnl=position.unrealized_pnl,
                strategy=position.strategy,
            )
            for symbol, position in self.portfolio.positions.items()
            if abs(position.quantity) > 1e-12
        ]

    def get_account(self) -> BrokerAccount:
        """Return the portfolio's account snapshot.

        Returns:
            The :class:`BrokerAccount`.
        """
        equity = self.portfolio.equity
        gross = self.portfolio.gross_exposure()
        return BrokerAccount(
            equity=equity,
            cash=self.portfolio.cash,
            margin_used=gross,
            margin_available=equity - gross,
        )

    @staticmethod
    def _spec(symbol: str) -> cfg.AssetSpec:
        """Resolve the asset spec, defaulting to a generic FX contract.

        Args:
            symbol: Instrument symbol.

        Returns:
            The :class:`~quant_system.config.settings.AssetSpec`.
        """
        try:
            return cfg.DEFAULT_SETTINGS.universe.spec(symbol)
        except KeyError:
            return cfg.AssetSpec(
                symbol=symbol,
                yf_symbols=(symbol,),
                mt5_symbol=symbol,
                pip_size=0.0001,
                contract_size=1.0,
                asset_class="fx",
                spread_pips=1.5,
                slippage_pips=0.2,
                vol_scale=1.0,
                base_price=100.0,
            )


__all__: List[str] = ["SimulatedBroker"]
