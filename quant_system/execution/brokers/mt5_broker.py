"""MetaTrader 5 execution adapter.

MetaTrader5 ships as a **Windows-only** Python package that drives the desktop
terminal over IPC, so this adapter degrades gracefully: importing it on Linux or
without the terminal installed raises :class:`MT5UnavailableError`, never an
``ImportError``, and the rest of the system keeps running.

Symbol resolution
-----------------
Broker symbol naming varies by venue (``XAUUSD``, ``XAUUSD.m``, ``GOLD``...).
:meth:`MT5Broker._resolve_symbol` tries the configured name, then a list of
common variants, then a scan of the terminal's symbol table.

Filling modes
-------------
MT5 rejects orders whose ``type_filling`` is unsupported by the symbol.  The
adapter tries ``ORDER_FILLING_IOC`` -> ``ORDER_FILLING_FOK`` ->
``ORDER_FILLING_RETURN`` in sequence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from quant_system.config import settings as cfg
from quant_system.execution.brokers.base import (
    BrokerAccount,
    BrokerBase,
    BrokerError,
    Order,
    OrderReport,
    OrderSide,
    OrderStatus,
    PositionReport,
)

logger = logging.getLogger(__name__)


class MT5UnavailableError(BrokerError):
    """Raised when the MetaTrader 5 terminal or package is not reachable."""


class MT5Broker(BrokerBase):
    """Executes orders through a local MetaTrader 5 terminal.

    Attributes:
        mt5: The imported ``MetaTrader5`` module (``None`` until connected).
        symbol_map: Cache of canonical symbol -> broker symbol.
    """

    name: str = "mt5"

    #: Common broker naming variants tried after the configured symbol.
    SYMBOL_VARIANTS: Sequence[str] = ("", ".m", ".a", ".pro", ".raw", "_m", "#", ".i")

    def __init__(self, config: Optional[cfg.BrokerConfig] = None) -> None:
        """Initialise the adapter.

        Args:
            config: Broker configuration.
        """
        super().__init__(config or cfg.DEFAULT_SETTINGS.broker)
        self.mt5: Any = None
        self.symbol_map: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Import MetaTrader5, initialise the terminal and log in.

        Raises:
            MT5UnavailableError: If the package or terminal is unavailable, or if
                ``initialize()`` / ``login()`` fails.
        """
        try:
            import MetaTrader5 as mt5  # noqa: PLC0415 - optional, Windows-only
        except Exception as exc:  # pragma: no cover - platform dependent
            raise MT5UnavailableError(
                "MetaTrader5 is unavailable on this host "
                "(the package is Windows-only and requires the MT5 terminal)."
            ) from exc

        self.mt5 = mt5
        broker_config: cfg.BrokerConfig = self.config
        kwargs: Dict[str, Any] = {}
        if broker_config.mt5_terminal_path:
            kwargs["path"] = broker_config.mt5_terminal_path

        if not mt5.initialize(**kwargs):
            raise MT5UnavailableError(f"MT5 initialize() failed: {mt5.last_error()}")

        if broker_config.mt5_login is not None:
            authorised = mt5.login(
                int(broker_config.mt5_login),
                password=broker_config.mt5_password or "",
                server=broker_config.mt5_server or "",
            )
            if not authorised:
                mt5.shutdown()
                raise MT5UnavailableError(f"MT5 login failed: {mt5.last_error()}")

        self._connected = True
        account = mt5.account_info()
        logger.info(
            "MT5 connected: account=%s server=%s balance=%.2f",
            getattr(account, "login", "?"),
            getattr(account, "server", "?"),
            getattr(account, "balance", float("nan")),
        )

    def disconnect(self) -> None:
        """Shut the terminal connection down."""
        if self.mt5 is not None and self._connected:
            self.mt5.shutdown()
        self._connected = False

    # ------------------------------------------------------------------ #
    def _resolve_symbol(self, symbol: str) -> str:
        """Map a canonical symbol onto the broker's naming convention.

        Args:
            symbol: Canonical symbol (e.g. ``"XAUUSD"``).

        Returns:
            The broker symbol.

        Raises:
            MT5UnavailableError: If no matching symbol is found.
        """
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        mt5 = self.mt5
        try:
            configured = cfg.DEFAULT_SETTINGS.universe.spec(symbol).mt5_symbol
        except KeyError:
            configured = symbol

        candidates = [configured] + [f"{configured}{suffix}" for suffix in self.SYMBOL_VARIANTS if suffix]
        for candidate in candidates:
            info = mt5.symbol_info(candidate)
            if info is not None:
                mt5.symbol_select(candidate, True)
                self.symbol_map[symbol] = candidate
                return candidate

        # Last resort: scan the whole symbol table.
        all_symbols = mt5.symbols_get()
        if all_symbols:
            names = [s.name for s in all_symbols]
            for name in names:
                if name.upper().startswith(configured.upper()):
                    mt5.symbol_select(name, True)
                    self.symbol_map[symbol] = name
                    return name
        raise MT5UnavailableError(f"No MT5 symbol matching {symbol!r} (tried {candidates}).")

    @staticmethod
    def _filling_modes(mt5: Any) -> List[int]:
        """Ordered filling modes to attempt.

        Args:
            mt5: The MetaTrader5 module.

        Returns:
            List of ``ORDER_FILLING_*`` constants.
        """
        modes: List[int] = []
        for attr in ("ORDER_FILLING_IOC", "ORDER_FILLING_FOK", "ORDER_FILLING_RETURN"):
            value = getattr(mt5, attr, None)
            if value is not None:
                modes.append(value)
        return modes

    # ------------------------------------------------------------------ #
    def submit_order(self, order: Order) -> OrderReport:
        """Send a market or limit order through the terminal.

        Args:
            order: The order to send.

        Returns:
            The :class:`OrderReport` produced from MT5's ``retcode``.
        """
        self._require_connection()
        mt5 = self.mt5
        broker_symbol = self._resolve_symbol(order.symbol)
        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None:
            return OrderReport(
                client_order_id=order.client_order_id,
                status=OrderStatus.REJECTED,
                message=f"No tick for {broker_symbol}.",
            )

        if order.side == OrderSide.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)

        if order.order_type.value == "limit":
            order_type = mt5.ORDER_TYPE_BUY_LIMIT if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL_LIMIT
            price = float(order.price or price)

        broker_config: cfg.BrokerConfig = self.config
        request: Dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL
            if order.order_type.value == "market"
            else mt5.TRADE_ACTION_PENDING,
            "symbol": broker_symbol,
            "volume": float(order.quantity),
            "type": order_type,
            "price": price,
            "deviation": int(broker_config.mt5_deviation_points),
            "magic": int(broker_config.mt5_magic),
            "comment": f"{order.strategy}:{order.tag}"[:31],
            "type_time": mt5.ORDER_TIME_GTC,
        }
        last_result = None
        for filling_mode in self._filling_modes(mt5):
            request["type_filling"] = filling_mode
            result = mt5.order_send(request)
            last_result = result
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                return OrderReport(
                    client_order_id=order.client_order_id,
                    broker_order_id=str(result.order),
                    status=OrderStatus.FILLED,
                    filled_quantity=float(result.volume),
                    average_fill_price=float(result.price),
                    message="MT5 deal executed",
                    raw={"retcode": result.retcode, "comment": result.comment},
                )
        retcode = getattr(last_result, "retcode", None)
        comment = getattr(last_result, "comment", "unknown failure")
        logger.error("MT5 order rejected for %s: retcode=%s %s", order.symbol, retcode, comment)
        return OrderReport(
            client_order_id=order.client_order_id,
            status=OrderStatus.REJECTED,
            message=f"MT5 retcode={retcode}: {comment}",
        )

    def cancel_order(self, client_order_id: str) -> OrderReport:
        """Cancel a working order by ticket.

        Args:
            client_order_id: Broker ticket (or the stored client id).

        Returns:
            The resulting :class:`OrderReport`.
        """
        self._require_connection()
        mt5 = self.mt5
        try:
            ticket = int(client_order_id)
        except (TypeError, ValueError):
            return OrderReport(
                client_order_id=client_order_id,
                status=OrderStatus.REJECTED,
                message="MT5 cancellation requires the numeric ticket.",
            )
        result = mt5.order_send(
            {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
        )
        done = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        return OrderReport(
            client_order_id=client_order_id,
            status=OrderStatus.CANCELLED if done else OrderStatus.REJECTED,
            message=getattr(result, "comment", ""),
        )

    def close_position(self, symbol: str, fraction: float = 1.0) -> OrderReport:
        """Close (or partially close) a position.

        Args:
            symbol: Canonical symbol.
            fraction: Fraction of the position to close, in ``(0, 1]``.

        Returns:
            The resulting :class:`OrderReport`.
        """
        self._require_connection()
        mt5 = self.mt5
        broker_symbol = self._resolve_symbol(symbol)
        positions = mt5.positions_get(symbol=broker_symbol)
        if not positions:
            return OrderReport(
                client_order_id=symbol,
                status=OrderStatus.REJECTED,
                message=f"No open MT5 position for {broker_symbol}.",
            )
        position = positions[0]
        volume = round(float(position.volume) * float(fraction), 2)
        if volume <= 0:
            return OrderReport(
                client_order_id=symbol,
                status=OrderStatus.REJECTED,
                message="Computed close volume is zero.",
            )
        side = OrderSide.SELL if position.type == mt5.ORDER_TYPE_BUY else OrderSide.BUY
        return self.submit_order(
            Order(
                symbol=symbol,
                side=side,
                quantity=volume,
                order_type="market",
                strategy="risk",
                tag="close_position",
            )
        )

    def get_positions(self) -> List[PositionReport]:
        """Return every open MT5 position.

        Returns:
            List of :class:`PositionReport`.
        """
        self._require_connection()
        mt5 = self.mt5
        raw_positions = mt5.positions_get() or []
        reports: List[PositionReport] = []
        for position in raw_positions:
            direction = 1 if position.type == mt5.ORDER_TYPE_BUY else -1
            reports.append(
                PositionReport(
                    symbol=str(position.symbol),
                    quantity=direction * float(position.volume),
                    average_price=float(position.price_open),
                    last_price=float(position.price_current),
                    unrealized_pnl=float(position.profit),
                )
            )
        return reports

    def get_account(self) -> BrokerAccount:
        """Return the MT5 account snapshot.

        Returns:
            The :class:`BrokerAccount`.
        """
        self._require_connection()
        info = self.mt5.account_info()
        return BrokerAccount(
            equity=float(info.equity),
            cash=float(info.balance),
            margin_used=float(info.margin),
            margin_available=float(info.margin_free),
            currency=str(getattr(info, "currency", "USD")),
        )

    def get_bars(self, symbol: str, timeframe: str = "1d", count: int = 500) -> pd.DataFrame:
        """Fetch recent bars directly from the terminal.

        Args:
            symbol: Canonical symbol.
            timeframe: ``"1d"``, ``"1h"``, ``"15m"`` or ``"5m"``.
            count: Number of bars.

        Returns:
            OHLCV DataFrame indexed by timestamp.
        """
        self._require_connection()
        mt5 = self.mt5
        mapping = {
            "1d": mt5.TIMEFRAME_D1,
            "1h": mt5.TIMEFRAME_H1,
            "15m": mt5.TIMEFRAME_M15,
            "5m": mt5.TIMEFRAME_M5,
        }
        broker_symbol = self._resolve_symbol(symbol)
        rates = mt5.copy_rates_from_pos(broker_symbol, mapping.get(timeframe, mt5.TIMEFRAME_D1), 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s")
        frame = frame.rename(columns={"tick_volume": "volume"}).set_index("time")
        frame.index.name = "timestamp"
        return frame[["open", "high", "low", "close", "volume"]].astype(float)


__all__: List[str] = ["MT5Broker", "MT5UnavailableError"]
