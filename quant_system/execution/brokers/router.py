"""Order router: turns target weights into broker orders and manages their lifecycle.

The router is the boundary between the *declarative* strategy layer ("I want
+12 % of equity in XAUUSD") and the *imperative* broker layer ("buy 0.58 lots of
XAUUSD.m IOC").  It handles:

* delta computation against current positions,
* dust filtering (``min_order_notional``),
* token-bucket rate limiting,
* bounded retries with exponential backoff,
* idempotency via a stable ``ClOrdID`` derived from the rebalance batch.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from quant_system.config import settings as cfg
from quant_system.execution.brokers.base import (
    BrokerBase,
    Order,
    OrderReport,
    OrderSide,
    OrderStatus,
)
from quant_system.execution.portfolio import Portfolio

logger = logging.getLogger(__name__)


@dataclass
class RoutingStats:
    """Counters for a routing session.

    Attributes:
        submitted: Orders sent.
        filled: Orders that reached ``FILLED``.
        rejected: Orders rejected by the venue.
        skipped: Deltas filtered out as dust.
        retries: Retry attempts made.
    """

    submitted: int = 0
    filled: int = 0
    rejected: int = 0
    skipped: int = 0
    retries: int = 0

    def as_dict(self) -> Dict[str, int]:
        """Return the counters as a plain dictionary."""
        return {
            "submitted": self.submitted,
            "filled": self.filled,
            "rejected": self.rejected,
            "skipped": self.skipped,
            "retries": self.retries,
        }


class OrderRouter:
    """Translates target weights into rate-limited broker orders.

    Attributes:
        broker: The destination broker adapter.
        config: Broker configuration (rate limits, retries).
        stats: Cumulative :class:`RoutingStats`.
    """

    def __init__(
        self,
        broker: BrokerBase,
        config: Optional[cfg.BrokerConfig] = None,
        min_order_notional: Optional[float] = None,
    ) -> None:
        """Initialise the router.

        Args:
            broker: Destination broker adapter.
            config: Broker configuration.
            min_order_notional: Dust threshold; defaults to
                ``settings.risk.min_order_notional``.
        """
        self.broker: BrokerBase = broker
        self.config: cfg.BrokerConfig = config or cfg.DEFAULT_SETTINGS.broker
        self.min_order_notional: float = float(
            min_order_notional if min_order_notional is not None
            else cfg.DEFAULT_SETTINGS.risk.min_order_notional
        )
        self.stats: RoutingStats = RoutingStats()
        self._tokens: float = float(self.config.max_orders_per_second)
        self._last_refill: float = time.monotonic()
        self._history: List[OrderReport] = []

    # ------------------------------------------------------------------ #
    # Order construction
    # ------------------------------------------------------------------ #
    def build_orders(
        self,
        portfolio: Portfolio,
        target_weights: Mapping[str, float],
        prices: Mapping[str, float],
        batch_id: Optional[str] = None,
        strategy: str = "",
        tag: str = "",
    ) -> List[Order]:
        """Convert target weights into delta orders.

        Args:
            portfolio: Portfolio holding the current positions.
            target_weights: Signed target exposure per symbol.
            prices: Reference price per symbol.
            batch_id: Optional id used to make ``ClOrdID``s reproducible.
            strategy: Strategy attribution.
            tag: Reason code.

        Returns:
            List of :class:`Order` objects (dust-filtered).
        """
        batch = batch_id or uuid.uuid4().hex[:8]
        orders: List[Order] = []
        symbols = set(target_weights) | {
            symbol for symbol, position in portfolio.positions.items()
            if abs(position.quantity) > 1e-12
        }
        for index, symbol in enumerate(sorted(symbols)):
            price = float(prices.get(symbol, 0.0))
            if price <= 0:
                continue
            target = float(target_weights.get(symbol, 0.0))
            delta_units = portfolio.target_quantity(symbol, target, price)
            if abs(delta_units) < 1e-12:
                continue
            contract_size = portfolio.position(symbol).contract_size
            notional = abs(delta_units) * price * contract_size
            if notional < self.min_order_notional:
                self.stats.skipped += 1
                continue
            orders.append(
                Order(
                    symbol=symbol,
                    side=OrderSide.BUY if delta_units > 0 else OrderSide.SELL,
                    quantity=abs(delta_units),
                    client_order_id=f"{batch}-{index:02d}",
                    strategy=strategy,
                    tag=tag,
                )
            )
        return orders

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def send(
        self,
        orders: Sequence[Order],
        max_retries: Optional[int] = None,
    ) -> List[OrderReport]:
        """Send orders with rate limiting and bounded retries.

        Args:
            orders: Orders to send.
            max_retries: Override for ``config.max_retries``.

        Returns:
            The list of :class:`OrderReport` outcomes, in input order.
        """
        retries = int(max_retries if max_retries is not None else self.config.max_retries)
        reports: List[OrderReport] = []
        for order in orders:
            report = self._send_with_retry(order, retries)
            reports.append(report)
            self._history.append(report)
            if report.status == OrderStatus.FILLED:
                self.stats.filled += 1
            elif report.status == OrderStatus.REJECTED:
                self.stats.rejected += 1
        return reports

    def _send_with_retry(self, order: Order, retries: int) -> OrderReport:
        """Send one order, retrying transient rejections.

        Args:
            order: The order to send.
            retries: Maximum retry attempts.

        Returns:
            The final :class:`OrderReport`.
        """
        attempt = 0
        report = OrderReport(client_order_id=order.client_order_id)
        while attempt <= retries:
            self._acquire_token()
            self.stats.submitted += 1
            try:
                report = self.broker.submit_order(order)
            except Exception as exc:  # transport-level failure
                logger.error("Broker error submitting %s: %s", order.client_order_id, exc)
                report = OrderReport(
                    client_order_id=order.client_order_id,
                    status=OrderStatus.REJECTED,
                    message=str(exc),
                )
            if report.status not in (OrderStatus.REJECTED,):
                return report
            attempt += 1
            if attempt > retries:
                break
            self.stats.retries += 1
            backoff = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Order %s rejected (%s); retry %d/%d in %.2fs",
                order.client_order_id,
                report.message,
                attempt,
                retries,
                backoff,
            )
            time.sleep(backoff)
        return report

    def execute_targets(
        self,
        portfolio: Portfolio,
        target_weights: Mapping[str, float],
        prices: Mapping[str, float],
        strategy: str = "",
        tag: str = "rebalance",
    ) -> Tuple[List[Order], List[OrderReport]]:
        """Build *and* send the orders that move the book to ``target_weights``.

        Args:
            portfolio: Portfolio holding current positions.
            target_weights: Signed target exposure per symbol.
            prices: Reference prices.
            strategy: Strategy attribution.
            tag: Reason code.

        Returns:
            Tuple ``(orders, reports)``.
        """
        orders = self.build_orders(
            portfolio, target_weights, prices, strategy=strategy, tag=tag
        )
        reports = self.send(orders)
        return orders, reports

    def flatten(
        self,
        portfolio: Portfolio,
        prices: Mapping[str, float],
        tag: str = "flatten",
    ) -> List[OrderReport]:
        """Close every open position.

        Args:
            portfolio: Portfolio holding current positions.
            prices: Reference prices.
            tag: Reason code.

        Returns:
            The list of :class:`OrderReport` outcomes.
        """
        return self.execute_targets(portfolio, {}, prices, strategy="risk", tag=tag)[1]

    # ------------------------------------------------------------------ #
    # Rate limiting
    # ------------------------------------------------------------------ #
    def _acquire_token(self) -> None:
        """Block until the token bucket allows another order."""
        rate = max(float(self.config.max_orders_per_second), 0.1)
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(rate, self._tokens + elapsed * rate)
        self._last_refill = now
        if self._tokens < 1.0:
            sleep_for = (1.0 - self._tokens) / rate
            time.sleep(sleep_for)
            self._tokens = 0.0
            self._last_refill = time.monotonic()
        else:
            self._tokens -= 1.0

    # ------------------------------------------------------------------ #
    @property
    def history(self) -> List[OrderReport]:
        """Every order report produced in this session."""
        return list(self._history)

    def reset_stats(self) -> None:
        """Clear the routing counters and history."""
        self.stats = RoutingStats()
        self._history = []


__all__: List[str] = ["OrderRouter", "RoutingStats"]
