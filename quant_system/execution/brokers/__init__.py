"""Broker adapters: a transport-agnostic interface plus MT5, FIX and simulated implementations.

The live layer mirrors the backtesting API (orders in, fills out) so the exact
same :class:`~quant_system.execution.brokers.router.OrderRouter` code path can be
exercised in simulation and then pointed at a broker by swapping the adapter.
"""

from __future__ import annotations

from quant_system.execution.brokers.base import (
    BrokerAccount,
    BrokerBase,
    BrokerError,
    BrokerNotConnectedError,
    Order,
    OrderRejectedError,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionReport,
)
from quant_system.execution.brokers.fix_broker import FIXBroker, FIXMessage
from quant_system.execution.brokers.mt5_broker import MT5Broker, MT5UnavailableError
from quant_system.execution.brokers.router import OrderRouter
from quant_system.execution.brokers.simulated_broker import SimulatedBroker

__all__: list[str] = [
    "BrokerAccount",
    "BrokerBase",
    "BrokerError",
    "BrokerNotConnectedError",
    "Order",
    "OrderRejectedError",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PositionReport",
    "SimulatedBroker",
    "MT5Broker",
    "MT5UnavailableError",
    "FIXBroker",
    "FIXMessage",
    "OrderRouter",
]
