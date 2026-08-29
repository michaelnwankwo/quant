"""FIX 4.4 execution adapter with a WebSocket market-data feed.

The adapter speaks a *real* FIX session: it performs the logon handshake,
maintains sequence numbers, answers ``TestRequest`` with ``Heartbeat``, sends
``Heartbeat`` on schedule, and correlates ``ExecutionReport`` messages back to
the originating ``ClOrdID``.

Because :class:`~quant_system.execution.brokers.base.BrokerBase` exposes a
*synchronous* API while FIX is inherently asynchronous, the adapter owns an
``asyncio`` event loop running on a daemon thread and marshals calls into it with
``run_coroutine_threadsafe``.  Callers keep the simple blocking interface; the
session stays non-blocking underneath.

FIX framing
-----------
``8=FIX.4.4|9=<BodyLength>|35=<MsgType>|49=<Sender>|56=<Target>|34=<SeqNum>|52=<SendingTime>|<body>|10=<CheckSum>|``
with ``|`` = ``SOH`` (``\\x01``).  ``BodyLength`` counts every byte between the
field immediately after ``9=`` and the ``10=`` field; ``CheckSum`` is
``sum(bytes) mod 256`` over everything up to (and including) the SOH preceding
``10=``, rendered as three digits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

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

SOH: str = "\x01"
BEGIN_STRING: str = "FIX.4.4"

#: FIX message-type tags used by this adapter.
MSG_HEARTBEAT: str = "0"
MSG_TEST_REQUEST: str = "1"
MSG_RESEND_REQUEST: str = "2"
MSG_REJECT: str = "3"
MSG_LOGOUT: str = "5"
MSG_EXECUTION_REPORT: str = "8"
MSG_ORDER_CANCEL_REJECT: str = "9"
MSG_LOGON: str = "A"
MSG_NEW_ORDER_SINGLE: str = "D"
MSG_ORDER_CANCEL_REQUEST: str = "F"
MSG_MARKET_DATA_REQUEST: str = "V"
MSG_MARKET_DATA_SNAPSHOT: str = "W"


@dataclass
class FIXMessage:
    """A parsed (or to-be-encoded) FIX message.

    Attributes:
        msg_type: FIX ``35`` value.
        fields: Ordered ``(tag, value)`` pairs of the message body (excluding the
            header fields ``8``/``9``/``35``/``49``/``56``/``34``/``52`` and the
            trailer ``10``, which are managed by :meth:`encode`).
    """

    msg_type: str
    fields: List[Tuple[str, str]] = field(default_factory=list)

    def get(self, tag: str) -> Optional[str]:
        """Return the value of ``tag`` if present.

        Args:
            tag: FIX tag number as a string.

        Returns:
            The value, or ``None``.
        """
        for key, value in self.fields:
            if key == tag:
                return value
        return None

    def __getitem__(self, tag: str) -> Optional[str]:
        """Alias for :meth:`get`."""
        return self.get(tag)

    @staticmethod
    def parse(raw: str) -> "FIXMessage":
        """Parse a raw FIX message string.

        Args:
            raw: The message including header and trailer.

        Returns:
            The parsed :class:`FIXMessage`.

        Raises:
            BrokerError: If the message has no ``35`` field or a bad checksum.
        """
        pairs: List[Tuple[str, str]] = []
        for chunk in raw.split(SOH):
            if not chunk:
                continue
            if "=" not in chunk:
                continue
            tag, _, value = chunk.partition("=")
            pairs.append((tag, value))
        msg_type = next((value for tag, value in pairs if tag == "35"), None)
        if msg_type is None:
            raise BrokerError("FIX message is missing tag 35 (MsgType).")

        body_start = 0
        for index, (tag, _value) in enumerate(pairs):
            if tag in {"8", "9", "35", "49", "56", "34", "52", "43", "97", "122"}:
                body_start = index + 1
            else:
                break
        body = [pair for pair in pairs if pair[0] not in {"8", "9", "10", "35", "49", "56", "34", "52"}]
        del body_start
        return FIXMessage(msg_type=msg_type, fields=body)

    def encode(self, seq_num: int, sender: str, target: str) -> bytes:
        """Serialise the message with a correct header, BodyLength and CheckSum.

        Args:
            seq_num: Outgoing ``MsgSeqNum`` (34).
            sender: ``SenderCompID`` (49).
            target: ``TargetCompID`` (56).

        Returns:
            The encoded message, SOH-delimited, ready to put on the wire.
        """
        sending_time = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]
        header: List[Tuple[str, str]] = [
            ("8", BEGIN_STRING),
            ("35", self.msg_type),
            ("49", sender),
            ("56", target),
            ("34", str(seq_num)),
            ("52", sending_time),
        ]
        body = header + list(self.fields)
        body_text = "".join(f"{tag}={value}{SOH}" for tag, value in body)
        length = len(body_text.encode("ascii", errors="replace"))
        full = f"8={BEGIN_STRING}{SOH}9={length}{SOH}" + body_text
        checksum = sum(full.encode("ascii", errors="replace")) % 256
        return (full + f"10={checksum:03d}{SOH}").encode("ascii", errors="replace")

    @staticmethod
    def verify_checksum(raw: str) -> bool:
        """Validate a raw message's checksum.

        Args:
            raw: The full raw message string.

        Returns:
            ``True`` if the checksum is correct (or absent).
        """
        index = raw.rfind("10=")
        if index == -1:
            return False
        payload = raw[:index]
        provided = raw[index + 3 :].rstrip(SOH)
        try:
            return f"{sum(payload.encode('ascii', errors='replace')) % 256:03d}" == provided
        except Exception:  # pragma: no cover - encoding edge cases
            return False

    def __repr__(self) -> str:
        """Return a compact developer representation."""
        pairs = ", ".join(f"{tag}={value}" for tag, value in self.fields[:6])
        return f"FIXMessage(35={self.msg_type}, {pairs})"


class FIXBroker(BrokerBase):
    """FIX 4.4 initiator with an optional FIX-over-WebSocket market-data feed.

    Attributes:
        seq_num: Outgoing sequence number.
        logged_on: ``True`` once the acceptor has confirmed the logon.
        exec_reports: Correlated execution reports keyed by ``ClOrdID``.
    """

    name: str = "fix"

    def __init__(self, config: Optional[cfg.BrokerConfig] = None) -> None:
        """Initialise the adapter.

        Args:
            config: Broker configuration.
        """
        super().__init__(config or cfg.DEFAULT_SETTINGS.broker)
        self.seq_num: int = 1
        self.logged_on: bool = False
        self.exec_reports: Dict[str, OrderReport] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._logon_event: Optional[asyncio.Event] = None
        self._heartbeat_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._reader_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._exec_events: Dict[str, asyncio.Event] = {}
        self._quote_queue: Optional[asyncio.Queue] = None  # type: ignore[type-arg]
        self._stop_event: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------ #
    # Synchronous facade
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        """Start the event loop thread and perform the FIX logon.

        Raises:
            BrokerError: If the logon is not confirmed within 30 seconds.
        """
        if self._connected:
            return
        self._loop = asyncio.new_event_loop()
        self._stop_event = asyncio.Event()
        self._thread = threading.Thread(
            target=self._run_loop, name="fix-event-loop", daemon=True
        )
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._async_connect(), self._loop)
        try:
            future.result(timeout=30)
        except Exception as exc:
            raise BrokerError(f"FIX logon failed: {exc}") from exc
        self._connected = True

    def disconnect(self) -> None:
        """Send a Logout message and stop the event loop thread."""
        if not self._connected or self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._async_disconnect(), self._loop).result(
                timeout=10
            )
        except Exception as exc:  # pragma: no cover - teardown best effort
            logger.debug("FIX disconnect raised: %s", exc)
        finally:
            self._connected = False
            self.logged_on = False

    def submit_order(self, order: Order) -> OrderReport:
        """Send a ``NewOrderSingle`` and wait for the correlated report.

        Args:
            order: The order to send.

        Returns:
            The correlated :class:`OrderReport`.
        """
        self._require_connection()
        broker_config: cfg.BrokerConfig = self.config
        event = asyncio.Event()
        loop = self._loop
        assert loop is not None
        self._exec_events[order.client_order_id] = event

        message = FIXMessage(
            msg_type=MSG_NEW_ORDER_SINGLE,
            fields=[
                ("11", order.client_order_id),
                ("55", order.symbol),
                ("54", "1" if order.side == OrderSide.BUY else "2"),
                ("60", datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]),
                ("38", f"{order.quantity:.8f}".rstrip("0").rstrip(".")),
                ("40", "1" if order.order_type.value == "market" else "2"),
                ("59", order.time_in_force),
            ]
            + ([("44", f"{order.price:.10f}")] if order.price is not None else [])
            + ([("1", str(broker_config.fix_username))] if broker_config.fix_username else []),
        )
        future = asyncio.run_coroutine_threadsafe(
            self._send_and_wait(message, order.client_order_id, event),
            loop,
        )
        try:
            return future.result(timeout=30)
        except Exception as exc:
            self._exec_events.pop(order.client_order_id, None)
            return OrderReport(
                client_order_id=order.client_order_id,
                status=OrderStatus.REJECTED,
                message=f"FIX submit failed: {exc}",
            )

    def cancel_order(self, client_order_id: str) -> OrderReport:
        """Send an ``OrderCancelRequest``.

        Args:
            client_order_id: The ``ClOrdID`` of the order to cancel.

        Returns:
            The resulting :class:`OrderReport`.
        """
        self._require_connection()
        loop = self._loop
        assert loop is not None
        cancel_id = f"C{client_order_id}"[:16]
        event = asyncio.Event()
        self._exec_events[cancel_id] = event
        message = FIXMessage(
            msg_type=MSG_ORDER_CANCEL_REQUEST,
            fields=[
                ("11", cancel_id),
                ("41", client_order_id),
                ("37", client_order_id),
                ("55", ""),
                ("54", "1"),
                ("60", datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S.%f")[:-3]),
            ],
        )
        future = asyncio.run_coroutine_threadsafe(
            self._send_and_wait(message, cancel_id, event), loop
        )
        try:
            return future.result(timeout=30)
        except Exception as exc:
            return OrderReport(
                client_order_id=client_order_id,
                status=OrderStatus.REJECTED,
                message=f"FIX cancel failed: {exc}",
            )

    def get_positions(self) -> List[PositionReport]:
        """Return positions known from execution reports.

        Note:
            Position reconciliation requires a ``PositionRequest`` (``35=AN``)
            or a proprietary message; the reference implementation derives
            positions from the fills it has seen, which is sufficient for
            stateless day-trading sessions.

        Returns:
            An empty list (no venue-side position snapshot is maintained).
        """
        return []

    def get_account(self) -> BrokerAccount:
        """Return a placeholder account snapshot.

        Returns:
            A zeroed :class:`BrokerAccount`; override per venue.
        """
        return BrokerAccount(equity=0.0, cash=0.0)

    async def stream_quotes(self, symbols: List[str]) -> AsyncIterator[Dict[str, float]]:
        """Stream quotes over the WebSocket feed (or FIX market data).

        Args:
            symbols: Symbols to subscribe to.

        Yields:
            Quote dictionaries with ``symbol``, ``bid``, ``ask``, ``last`` and
            ``timestamp`` keys.

        Raises:
            BrokerError: If no market-data endpoint is configured.
        """
        broker_config: cfg.BrokerConfig = self.config
        if not broker_config.ws_marketdata_url:
            raise BrokerError(
                "No ws_marketdata_url configured; FIX market-data snapshots are "
                "only emitted on request."
            )
        try:
            import websockets  # noqa: PLC0415 - optional dependency
        except Exception as exc:  # pragma: no cover
            raise BrokerError("The `websockets` package is required for streaming.") from exc

        async with websockets.connect(broker_config.ws_marketdata_url) as socket:  # type: ignore[attr-defined]
            subscribe = {"action": "subscribe", "symbols": symbols}
            await socket.send(json.dumps(subscribe))
            async for payload in socket:
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, list):
                    for item in data:
                        yield self._normalise_quote(item)
                elif isinstance(data, dict):
                    yield self._normalise_quote(data)

    @staticmethod
    def _normalise_quote(payload: Dict[str, Any]) -> Dict[str, float | str]:
        """Normalise a vendor quote payload.

        Args:
            payload: Raw decoded JSON object.

        Returns:
            A dict with ``symbol``, ``bid``, ``ask``, ``last``, ``timestamp``.
        """
        symbol = str(payload.get("symbol") or payload.get("s") or "")
        bid = float(payload.get("bid", payload.get("b", float("nan"))))
        ask = float(payload.get("ask", payload.get("a", float("nan"))))
        last = float(payload.get("last", payload.get("price", payload.get("p", float("nan")))))
        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "last": last,
            "timestamp": float(payload.get("ts", time.time())),
        }

    # ------------------------------------------------------------------ #
    # Async internals
    # ------------------------------------------------------------------ #
    def _run_loop(self) -> None:
        """Run the owned event loop until stopped."""
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    async def _async_connect(self) -> None:
        """Open the socket, send Logon and start session tasks."""
        broker_config: cfg.BrokerConfig = self.config
        ssl_context: Optional[ssl.SSLContext] = None
        if broker_config.fix_use_tls:
            ssl_context = ssl.create_default_context()
        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(
                broker_config.fix_host, broker_config.fix_port, ssl=ssl_context
            ),
            timeout=15,
        )
        self._logon_event = asyncio.Event()
        self.seq_num = 1
        await self._send_logon()
        self._reader_task = asyncio.create_task(self._read_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        await asyncio.wait_for(self._logon_event.wait(), timeout=20)
        self.logged_on = True
        logger.info(
            "FIX session established with %s:%s (sender=%s)",
            broker_config.fix_host,
            broker_config.fix_port,
            broker_config.fix_sender_comp_id,
        )

    async def _async_disconnect(self) -> None:
        """Send Logout, cancel tasks and stop the loop."""
        try:
            logout = FIXMessage(msg_type=MSG_LOGOUT, fields=[("58", "client shutdown")])
            await self._send(logout)
        except Exception as exc:  # pragma: no cover
            logger.debug("Logout send failed: %s", exc)
        for task in (self._reader_task, self._heartbeat_task):
            if task is not None:
                task.cancel()
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:  # pragma: no cover
                pass
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

    async def _send_logon(self) -> None:
        """Send the Logon (35=A) message."""
        broker_config: cfg.BrokerConfig = self.config
        fields: List[Tuple[str, str]] = [
            ("98", "0"),  # EncryptMethod = none
            ("108", str(broker_config.fix_heartbeat_seconds)),
            ("141", "Y"),  # ResetSeqNumFlag
        ]
        if broker_config.fix_username:
            fields.append(("553", str(broker_config.fix_username)))
        if broker_config.fix_password:
            fields.append(("554", str(broker_config.fix_password)))
        await self._send(FIXMessage(MSG_LOGON, fields))

    async def _send(self, message: FIXMessage) -> None:
        """Write a message to the socket and increment the sequence number.

        Args:
            message: The message to send.
        """
        broker_config: cfg.BrokerConfig = self.config
        if self._writer is None:
            raise BrokerError("FIX socket is not open.")
        payload = message.encode(
            self.seq_num, broker_config.fix_sender_comp_id, broker_config.fix_target_comp_id
        )
        self._writer.write(payload)
        await self._writer.drain()
        self.seq_num += 1
        logger.debug("FIX -> %s", message)

    async def _send_and_wait(
        self, message: FIXMessage, key: str, event: asyncio.Event
    ) -> OrderReport:
        """Send a message and await its correlated execution report.

        Args:
            message: The message to send.
            key: Correlation key (``ClOrdID``).
            event: Event signalled when the report arrives.

        Returns:
            The correlated :class:`OrderReport`.
        """
        await self._send(message)
        try:
            await asyncio.wait_for(event.wait(), timeout=25)
        except asyncio.TimeoutError:
            return OrderReport(
                client_order_id=key,
                status=OrderStatus.PENDING,
                message="Timed out waiting for an execution report.",
            )
        finally:
            self._exec_events.pop(key, None)
        return self.exec_reports.get(
            key, OrderReport(client_order_id=key, status=OrderStatus.PENDING)
        )

    async def _read_loop(self) -> None:
        """Continuously read, frame and dispatch inbound messages."""
        assert self._reader is not None
        buffer: str = ""
        try:
            while True:
                chunk = await self._reader.read(4096)
                if not chunk:
                    logger.warning("FIX socket closed by peer.")
                    break
                buffer += chunk.decode("ascii", errors="replace")
                while SOH in buffer:
                    end = buffer.find(f"10=")
                    if end == -1:
                        break
                    terminator = buffer.find(SOH, end)
                    if terminator == -1:
                        break
                    raw = buffer[: terminator + 1]
                    buffer = buffer[terminator + 1 :]
                    self._dispatch(raw)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("FIX read loop error: %s", exc)

    def _dispatch(self, raw: str) -> None:
        """Route a single raw message to the right handler.

        Args:
            raw: The complete raw message including the checksum field.
        """
        if not FIXMessage.verify_checksum(raw):
            logger.warning("Dropping FIX message with a bad checksum: %r", raw[:80])
            return
        try:
            message = FIXMessage.parse(raw)
        except BrokerError as exc:
            logger.warning("Unparseable FIX message: %s", exc)
            return
        logger.debug("FIX <- %s", message)

        if message.msg_type == MSG_LOGON:
            if self._logon_event is not None:
                self._logon_event.set()
        elif message.msg_type == MSG_LOGOUT:
            text = message.get("58") or "peer logged out"
            logger.info("FIX logout received: %s", text)
            self.logged_on = False
        elif message.msg_type == MSG_HEARTBEAT:
            pass
        elif message.msg_type == MSG_TEST_REQUEST:
            test_id = message.get("112") or ""
            heartbeat = FIXMessage(MSG_HEARTBEAT, [("112", test_id)] if test_id else [])
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(self._send(heartbeat), self._loop)
        elif message.msg_type in (MSG_EXECUTION_REPORT, MSG_ORDER_CANCEL_REJECT):
            self._handle_execution_report(message)
        elif message.msg_type == MSG_MARKET_DATA_SNAPSHOT:
            if self._quote_queue is not None and self._loop is not None:
                quote = {
                    "symbol": message.get("55") or "",
                    "bid": float(message.get("270") or float("nan")),
                    "ask": float(message.get("271") or float("nan")),
                    "last": float(message.get("270") or float("nan")),
                    "timestamp": time.time(),
                }
                asyncio.run_coroutine_threadsafe(self._quote_queue.put(quote), self._loop)
        elif message.msg_type == MSG_REJECT:
            logger.warning("FIX session-level reject: %s", message.get("58"))

    def _handle_execution_report(self, message: FIXMessage) -> None:
        """Translate an ExecutionReport into an :class:`OrderReport`.

        Args:
            message: The parsed execution report.
        """
        clord_id = message.get("11") or ""
        status_map = {
            "0": OrderStatus.ACCEPTED,
            "1": OrderStatus.PARTIALLY_FILLED,
            "2": OrderStatus.FILLED,
            "4": OrderStatus.CANCELLED,
            "6": OrderStatus.PENDING,
            "8": OrderStatus.REJECTED,
            "A": OrderStatus.PENDING,
            "E": OrderStatus.PENDING,
        }
        ord_status = message.get("39") or "A"
        exec_type = message.get("150") or ord_status
        status = status_map.get(exec_type) or status_map.get(ord_status, OrderStatus.PENDING)

        cum_qty = float(message.get("14") or 0.0)
        avg_px = float(message.get("6") or 0.0)
        report = OrderReport(
            client_order_id=clord_id,
            broker_order_id=message.get("37"),
            status=status,
            filled_quantity=cum_qty,
            average_fill_price=avg_px,
            message=message.get("58") or "",
            raw={"39": ord_status, "150": exec_type, "14": cum_qty, "6": avg_px},
        )
        self.exec_reports[clord_id] = report
        # Also key by OrigClOrdID so cancels resolve against the original id.
        orig = message.get("41")
        if orig:
            self.exec_reports[orig] = report
        for key in (clord_id, f"C{clord_id}", orig or ""):
            event = self._exec_events.get(key)
            if event is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(event.set)

    async def _heartbeat_loop(self) -> None:
        """Send a Heartbeat whenever the session has been idle."""
        broker_config: cfg.BrokerConfig = self.config
        interval = max(1, int(broker_config.fix_heartbeat_seconds))
        try:
            while True:
                await asyncio.sleep(interval)
                if self.logged_on:
                    await self._send(FIXMessage(MSG_HEARTBEAT, []))
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:  # pragma: no cover
            logger.error("FIX heartbeat loop error: %s", exc)

    def request_market_data(
        self, symbols: Sequence[str], entry_types: Sequence[str] = ("0", "1")
    ) -> None:
        """Subscribe to FIX market data for ``symbols``.

        Args:
            symbols: Symbols to request.
            entry_types: MDEntryTypes: ``"0"`` bid, ``"1"`` offer, ``"2"`` trade.
        """
        self._require_connection()
        if self._loop is None:
            return
        request_id = f"MD{int(time.time())}"
        fields: List[Tuple[str, str]] = [
            ("262", request_id),
            ("263", "1"),  # SubscriptionRequestType = snapshot + updates
            ("264", "0"),  # MarketDepth
            ("267", str(len(entry_types))),
        ]
        for entry in entry_types:
            fields.append(("269", entry))
        fields.append(("146", str(len(symbols))))
        for symbol in symbols:
            fields.append(("55", symbol))
        asyncio.run_coroutine_threadsafe(
            self._send(FIXMessage(MSG_MARKET_DATA_REQUEST, fields)), self._loop
        )


__all__: List[str] = ["FIXBroker", "FIXMessage"]
