"""Trade notification engine: desktop toasts and spoken trade alerts.

Design notes
------------
**Event filtering.** The engine listens *only* for confirmed fills.  Raw tick
updates, routine HMM-state polls, working/unfilled limit orders, cancellations and
rejections are counted and dropped.  Two entry points exist:

* :meth:`NotifierEngine.notify_fill` — accepts an
  :class:`~quant_system.execution.brokers.base.OrderReport` and fires only when
  ``status == OrderStatus.FILLED`` (``PARTIALLY_FILLED`` is opt-in).
* :meth:`NotifierEngine.notify_mt5_result` — accepts a raw MetaTrader 5
  ``OrderSendResult`` and fires only when ``retcode == mt5.TRADE_RETCODE_DONE``.

**Concurrency.** Speech runs on a single dedicated daemon thread
(:class:`VoiceWorker`) fed by a :class:`queue.Queue`, so ``pyttsx3``'s blocking
``runAndWait()`` can never stall the asyncio execution loop.  Desktop toasts run
on their own GUI thread (Qt/tk require widgets to be constructed in the thread
that owns the event loop).  Both are fail-soft: if a backend cannot initialise
(headless container, missing audio device, no display) the engine degrades to the
next backend and records why, rather than raising into the trading loop.

**Backends.**

``pyqt``   PyQt6 frameless always-on-top toast, fading, auto-closing.
``tk``     tkinter fallback with the same queue-driven architecture.
``none``   Records events in memory without touching the windowing system — used
           for deterministic tests and as the last-resort fallback.

On a headless host PyQt6 is forced onto the ``offscreen`` Qt platform plugin so
the widget tree still builds and can be rendered to a PNG via
:meth:`PyQtToastBackend.capture`, which is what the verification suite uses to
prove the UI path.
"""

from __future__ import annotations

import logging
import os
import contextlib
import io
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from quant_system.config import settings as cfg

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Enums & events
# --------------------------------------------------------------------------- #
class VoiceMode(str, Enum):
    """How the voice worker renders speech.

    Attributes:
        SPEAK: Play through the default audio device (live desktop use).
        FILE: Synthesise to a WAV file (headless / verification use).
        OFF: Accept and discard messages.
    """

    SPEAK = "speak"
    FILE = "file"
    OFF = "off"


class ToastBackend(str, Enum):
    """Available toast rendering backends.

    Attributes:
        AUTO: Try PyQt6, then tkinter, then the null recorder.
        PYQT: PyQt6 only.
        TK: tkinter only.
        NONE: In-memory recorder (no windowing system).
    """

    AUTO = "auto"
    PYQT = "pyqt"
    TK = "tk"
    NONE = "none"


class FilterReason(str, Enum):
    """Why an inbound message was not announced."""

    DISABLED = "notifier_disabled"
    NOT_FILLED = "not_a_confirmed_fill"
    ZERO_QUANTITY = "zero_filled_quantity"
    DUPLICATE = "duplicate_order_id"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    UNKNOWN_SYMBOL = "unknown_symbol"


def _utc_now() -> pd.Timestamp:
    """Return the current UTC time as a timezone-naive timestamp.

    Returns:
        The current timestamp.
    """
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


#: Human-readable instrument names used by the speech synthesiser.
SPOKEN_SYMBOLS: Dict[str, str] = {
    "XAUUSD": "Gold",
    "XAGUSD": "Silver",
    "EURUSD": "Euro Dollar",
    "GBPUSD": "Cable",
    "USDCHF": "Dollar Swiss",
    "USDJPY": "Dollar Yen",
    "AUDUSD": "Aussie Dollar",
    "USDCAD": "Dollar Cad",
    "NZDUSD": "Kiwi Dollar",
}

#: Short regime names used in the spoken alert.
SPOKEN_REGIMES: Dict[int, str] = {
    cfg.STATE_RANGE_BOUND: "State 0 Range",
    cfg.STATE_TREND: "State 1 Momentum",
    cfg.STATE_SHOCK: "State 2 Shock",
}


def price_precision(value: float) -> int:
    """Return the conventional decimal precision for a price level.

    Args:
        value: Price or quantity.

    Returns:
        ``2`` for metals/futures-style levels (>= 100), ``4`` for FX majors
        (1-100), ``5`` for sub-unit rates (< 1).
    """
    magnitude = abs(value)
    if magnitude >= 100:
        return 2
    return 4 if magnitude >= 1 else 5


def format_price(value: Optional[float]) -> str:
    """Format a price for display with thousands separators.

    Args:
        value: Price, or ``None``.

    Returns:
        The formatted string, or ``"-"`` when ``value`` is ``None``.
    """
    if value is None:
        return "-"
    return f"{value:,.{price_precision(value)}f}"


def speak_number(value: float, decimals: Optional[int] = None) -> str:
    """Render a number in a natural, trader-friendly spoken form.

    Large numbers use two decimals (``2650.50`` -> ``"2650 point 50"``); FX-rate
    style numbers use pip pairs so ``1.0850`` reads as ``"1 point 08 50"`` rather
    than an unwieldy digit string.

    Args:
        value: The number to render.
        decimals: Explicit precision. Defaults to :func:`price_precision`, but
            pass ``2`` for lot sizes so ``0.50`` reads as ``"0 point 50"``.

    Returns:
        The spoken form.

    Raises:
        ValueError: If ``value`` is not finite.
    """
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        raise ValueError("speak_number requires a finite value.")
    precision = int(decimals) if decimals is not None else price_precision(value)
    text = f"{value:.{precision}f}".replace(",", "")
    whole, _, frac = text.partition(".")
    if not frac:
        return whole
    if abs(value) >= 100:
        return f"{whole} point {frac}"
    # FX rates are spoken in pip pairs.  Drop trailing zeros (they carry no
    # information) and pad back to an even length so pairs line up: 0.88200 ->
    # "88 20", 1.0850 -> "08 50".
    frac = frac.rstrip("0")
    if not frac:
        return whole
    if len(frac) % 2:
        frac += "0"
    pairs = " ".join(frac[i : i + 2] for i in range(0, len(frac), 2))
    return f"{whole} point {pairs}"


def spoken_symbol(symbol: str) -> str:
    """Map a broker symbol onto a spoken instrument name.

    Args:
        symbol: Canonical symbol.

    Returns:
        A name suitable for text-to-speech.
    """
    clean = symbol.upper().replace("/", "").replace("_", "")
    if clean in SPOKEN_SYMBOLS:
        return SPOKEN_SYMBOLS[clean]
    return " ".join(clean)


@dataclass(frozen=True)
class TradeEvent:
    """A confirmed trade execution worth announcing.

    Attributes:
        action: ``"BUY"``, ``"SELL"`` or ``"CLOSE"``.
        symbol: Instrument symbol.
        volume: Filled units (always positive).
        price: Average fill price.
        stop_loss: Stop-loss price if known.
        take_profit: Take-profit price if known.
        regime_state: Active HMM regime at execution time.
        strategy: Originating strategy name.
        order_id: Client or broker order id (used for de-duplication).
        realised_pnl: Realised PnL when the fill closed a position.
        timestamp: Event timestamp.
    """

    action: str
    symbol: str
    volume: float
    price: float
    regime_state: int = cfg.STATE_RANGE_BOUND
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = ""
    order_id: str = ""
    realised_pnl: Optional[float] = None
    timestamp: pd.Timestamp = field(default_factory=_utc_now)

    # ------------------------------------------------------------------ #
    @property
    def regime_label(self) -> str:
        """Short spoken regime label, e.g. ``"State 1 Momentum"``."""
        return SPOKEN_REGIMES.get(int(self.regime_state), f"State {self.regime_state}")

    @property
    def is_closing(self) -> bool:
        """Whether the fill closed (part of) a position."""
        return self.action == "CLOSE" or self.realised_pnl is not None

    def toast_title(self) -> str:
        """Return the headline line used by the desktop toast.

        Returns:
            A string such as ``"[BUY] XAUUSD"``.
        """
        return f"[{self.action}] {self.symbol}"

    def toast_body(self) -> str:
        """Return the detail line used by the desktop toast.

        Returns:
            ``"Volume | Price | SL | TP | Regime"`` pipe-delimited.
        """
        parts: List[str] = [
            f"{self.volume:,.2f}",
            format_price(self.price),
            f"SL {format_price(self.stop_loss)}",
            f"TP {format_price(self.take_profit)}",
            self.regime_label,
        ]
        return " | ".join(parts)

    def speech_text(self) -> str:
        """Return the sentence read aloud by the voice worker.

        Returns:
            A concise spoken summary, e.g.
            ``"Bought 0.50 lots of Gold at 2650 point 50 under State 1 Momentum"``.
        """
        verb = {"BUY": "Bought", "SELL": "Sold", "CLOSE": "Closed"}.get(
            self.action.upper(), self.action.title()
        )
        phrase = (
            f"{verb} {speak_number(self.volume, decimals=2)} lots of "
            f"{spoken_symbol(self.symbol)} "
            f"at {speak_number(self.price)} under {self.regime_label}"
        )
        if self.stop_loss is not None:
            phrase += f", stop {speak_number(self.stop_loss)}"
        if self.realised_pnl is not None:
            direction = "profit" if self.realised_pnl >= 0 else "loss"
            phrase += f", realised {direction} {speak_number(abs(self.realised_pnl))}"
        return phrase + "."

    def as_dict(self) -> Dict[str, Any]:
        """Serialise the event for logging and reports."""
        return {
            "timestamp": str(self.timestamp),
            "action": self.action,
            "symbol": self.symbol,
            "volume": self.volume,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "regime_state": int(self.regime_state),
            "regime_label": self.regime_label,
            "strategy": self.strategy,
            "order_id": self.order_id,
            "realised_pnl": self.realised_pnl,
            "speech": self.speech_text(),
        }


# --------------------------------------------------------------------------- #
# Voice worker
# --------------------------------------------------------------------------- #
class VoiceWorker(threading.Thread):
    """Background ``pyttsx3`` worker fed by a :class:`queue.Queue`.

    The ``pyttsx3`` engine is created *inside* the worker thread (it binds to the
    thread that drives its run loop) and reused for the worker's lifetime, which
    avoids the per-call initialisation cost and the thread-affinity pitfalls.

    Attributes:
        mode: :class:`VoiceMode` controlling how speech is rendered.
        spoken: Number of utterances successfully rendered.
        failed: Number of utterances that raised.
        skipped: Number of messages dropped (engine unavailable or mode ``off``).
    """

    def __init__(
        self,
        mode: VoiceMode | str = VoiceMode.SPEAK,
        rate: int = 170,
        volume: float = 1.0,
        output_dir: Optional[Path] = None,
        queue_maxsize: int = 256,
        name: str = "voice-worker",
    ) -> None:
        """Initialise the worker.

        Args:
            mode: Rendering mode.
            rate: Words per minute.
            volume: Master volume in ``[0, 1]``.
            output_dir: Destination directory for :attr:`VoiceMode.FILE`.
            queue_maxsize: Backlog bound; full queues drop the newest message.
            name: Thread name.
        """
        super().__init__(name=name, daemon=True)
        self.mode: VoiceMode = VoiceMode(mode)
        self.rate: int = int(rate)
        self.volume: float = float(volume)
        self.output_dir: Path = Path(output_dir or cfg.VOICE_DIR)
        self._queue: "queue.Queue[Optional[Tuple[str, str]]]" = queue.Queue(
            maxsize=max(1, int(queue_maxsize))
        )
        self._ready = threading.Event()
        self._engine: Any = None
        self.spoken: int = 0
        self.failed: int = 0
        self.skipped: int = 0
        self.dropped: int = 0
        self.available: bool = False
        self.last_error: Optional[str] = None
        self.last_file: Optional[Path] = None
        self._files: List[Path] = []
        self._counter: int = 0

    # ------------------------------------------------------------------ #
    @property
    def files(self) -> List[Path]:
        """WAV files produced in :attr:`VoiceMode.FILE` mode."""
        return list(self._files)

    def submit(self, text: str, key: str = "") -> bool:
        """Queue an utterance (never blocks the caller).

        Args:
            text: The sentence to speak.
            key: Optional identifier used for output filenames.

        Returns:
            ``True`` if the message was queued, ``False`` if the queue was full.
        """
        try:
            self._queue.put_nowait((text, key))
            return True
        except queue.Full:
            self.dropped += 1
            logger.warning("Voice queue full; dropped alert: %s", text[:60])
            return False

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """Block until the worker has finished initialising its engine.

        Args:
            timeout: Seconds to wait.

        Returns:
            ``True`` if the worker signalled readiness.
        """
        return bool(self._ready.wait(timeout))

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and join it.

        Args:
            timeout: Join timeout in seconds.
        """
        if self.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:  # pragma: no cover - only under a burst
                pass
            self.join(timeout=timeout)

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Worker main loop: initialise the engine, then drain the queue."""
        self._initialise_engine()
        self._ready.set()
        while True:
            item = self._queue.get()
            if item is None:
                break
            text, key = item
            self._render(text, key)
        self._shutdown_engine()

    def _initialise_engine(self) -> None:
        """Create and configure the ``pyttsx3`` engine.

        Failures are recorded rather than raised: a missing audio device or a
        headless host must not break the trading loop.
        """
        if self.mode is VoiceMode.OFF:
            self.available = False
            self.last_error = "voice mode is 'off'"
            return
        try:
            import pyttsx3  # noqa: PLC0415 - optional dependency

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", max(0.0, min(1.0, self.volume)))
            self._engine = engine
            self.available = True
            logger.info("Voice worker ready (pyttsx3, mode=%s).", self.mode.value)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("pyttsx3 unavailable; voice alerts disabled (%s).", exc)

    def _shutdown_engine(self) -> None:
        """Best-effort teardown of the speech engine."""
        stop = getattr(self._engine, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception as exc:  # pragma: no cover
                logger.debug("pyttsx3 stop() failed: %s", exc)
        self._engine = None

    def _render(self, text: str, key: str) -> None:
        """Render one utterance.

        Args:
            text: The sentence to render.
            key: Identifier used for the output filename.
        """
        if not self.available or self._engine is None:
            self.skipped += 1
            return
        try:
            # The bundled espeak driver print()s "Audio saved to ..." straight to
            # stdout; swallow it so the trading log (and the test output) stays
            # clean. The WAV path is reported via :attr:`last_file` instead.
            with contextlib.redirect_stdout(io.StringIO()):
                if self.mode is VoiceMode.FILE:
                    self._render_to_file(text, key)
                else:
                    self._engine.say(text)
                    self._engine.runAndWait()
            self.spoken += 1
        except Exception as exc:  # pragma: no cover - driver dependent
            self.failed += 1
            self.last_error = str(exc)
            logger.warning("Speech synthesis failed for %r: %s", text[:40], exc)

    def _render_to_file(self, text: str, key: str) -> None:
        """Synthesise ``text`` to a WAV file.

        Args:
            text: The sentence to render.
            key: Identifier used for the output filename.

        Raises:
            RuntimeError: If the output directory cannot be created.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._counter += 1
        slug = re.sub(r"[^A-Za-z0-9]+", "_", key or "alert").strip("_").lower()[:40]
        path = self.output_dir / f"{self._counter:03d}_{slug or 'alert'}.wav"
        self._engine.save_to_file(text, str(path))
        self._engine.runAndWait()
        if path.exists() and path.stat().st_size > 0:
            self.last_file = path
            self._files.append(path)


# --------------------------------------------------------------------------- #
# Toast backends
# --------------------------------------------------------------------------- #
@dataclass
class ToastRecord:
    """A toast rendered by the null backend (used for tests).

    Attributes:
        event: The trade event that was announced.
        rendered_at: Monotonic timestamp of the render.
    """

    event: TradeEvent
    rendered_at: float = field(default_factory=time.monotonic)


class NullToastBackend:
    """Records toasts in memory without touching the windowing system.

    This is the deterministic backend used by the verification suite and the
    last-resort fallback when neither Qt nor tkinter can initialise.
    """

    name: str = "none"

    def __init__(self, config: Optional[cfg.NotifierConfig] = None) -> None:
        """Initialise the recorder.

        Args:
            config: Notifier configuration.
        """
        self.config = config or cfg.DEFAULT_SETTINGS.notifier
        self.records: List[ToastRecord] = []
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Start the backend.

        Returns:
            Always ``True``.
        """
        return True

    def submit(self, event: TradeEvent) -> bool:
        """Record an event.

        Args:
            event: The trade event.

        Returns:
            Always ``True``.
        """
        with self._lock:
            self.records.append(ToastRecord(event))
        return True

    def stop(self) -> None:
        """No-op."""

    @property
    def delivered(self) -> int:
        """Number of toasts recorded."""
        with self._lock:
            return len(self.records)

    @property
    def last_error(self) -> Optional[str]:
        """Never fails."""
        return None


class PyQtToastBackend:
    """Frameless, always-on-top, auto-closing PyQt6 toast.

    Qt requires widgets to be created in the thread that owns the
    ``QApplication``, so the backend starts a dedicated daemon thread, builds the
    application there and drains a :class:`queue.Queue` on a ``QTimer``.  Toasts
    stack bottom-right and fade in/out.

    On a headless host the Qt platform plugin is forced to ``offscreen`` so the
    widget tree still builds; :meth:`capture` then renders a toast to a PNG, which
    is how the verification suite proves the UI path without a display.
    """

    name: str = "pyqt"

    def __init__(self, config: Optional[cfg.NotifierConfig] = None) -> None:
        """Initialise the backend.

        Args:
            config: Notifier configuration.
        """
        self.config = config or cfg.DEFAULT_SETTINGS.notifier
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=64)
        self._thread: Optional[threading.Thread] = None
        self._app: Any = None
        self._timer: Any = None
        self._ready = threading.Event()
        self._live: List[Any] = []
        self._delivered = 0
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _has_display() -> bool:
        """Whether a windowing display appears to be available.

        Returns:
            ``True`` if ``DISPLAY`` or ``WAYLAND_DISPLAY`` is set (non-macOS).
        """
        if os.name == "nt":
            return True
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    @staticmethod
    def _prepare_platform() -> None:
        """Force the ``offscreen`` Qt plugin when no display is present."""
        if not PyQtToastBackend._has_display():
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")

    def start(self) -> bool:
        """Start the Qt event loop on a background thread.

        Returns:
            ``True`` if the application started successfully.
        """
        try:
            from PyQt6 import QtWidgets  # noqa: PLC0415, F401
        except Exception as exc:  # pragma: no cover - optional dependency
            self._error = f"PyQt6 unavailable: {exc}"
            logger.info("PyQt6 toast backend unavailable (%s).", exc)
            return False

        self._prepare_platform()
        self._thread = threading.Thread(
            target=self._run, name="toast-qt", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=15.0):
            self._error = "Qt application failed to start within 15 s"
            return False
        return self._error is None

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        """Thread entry point: build the QApplication and poll the queue."""
        try:
            from PyQt6 import QtCore, QtWidgets

            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            self._app = app
            timer = QtCore.QTimer()
            timer.timeout.connect(self._drain)
            timer.start(80)
            self._timer = timer
            self._ready.set()
            app.exec()
            self._app = None
            self._timer = None
        except Exception as exc:  # pragma: no cover - platform dependent
            self._error = f"{type(exc).__name__}: {exc}"
            logger.warning("Qt toast backend failed: %s", exc)
            self._ready.set()

    def _drain(self) -> None:
        """Process queued items on the Qt thread (invoked by the timer).

        Everything is wrapped defensively: an exception escaping a Qt slot
        terminates the process, and a cosmetic toast must never be able to take
        the trading loop down with it.
        """
        try:
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item is None:
                        self._destroy_live()
                        if self._app is not None:
                            self._app.quit()
                        return
                    if isinstance(item, tuple) and len(item) == 3:
                        event, path, done = item
                        try:
                            self._render_capture(event, path)
                        finally:
                            done.set()
                    else:
                        self._render_toast(item)
            except queue.Empty:
                pass
            self._prune()
        except Exception as exc:  # pragma: no cover - defensive
            self._error = f"{type(exc).__name__}: {exc}"
            logger.warning("Qt toast backend error; disabling toasts: %s", exc)

    def _prune(self) -> None:
        """Drop references to toasts that have been closed."""
        self._live = [w for w in self._live if w is not None and w.isVisible()]

    def _destroy_live(self) -> None:
        """Close and release every live toast (Qt thread only).

        Widgets must not outlive the ``QApplication``: destroying a ``QWidget``
        after the application object has been torn down is a hard crash, so the
        live list is emptied *before* the event loop is quit.
        """
        for widget in self._live:
            try:
                widget.close()
                widget.deleteLater()
            except Exception:  # pragma: no cover - defensive
                pass
        self._live.clear()

    def _position(self, index: int, height: int, width: int, screen: Any) -> Tuple[int, int]:
        """Compute the stacked position for toast ``index``.

        Args:
            index: Stack position (0 = closest to the corner).
            height: Toast height in pixels.
            width: Toast width in pixels.
            screen: The ``QScreen`` to anchor against.

        Returns:
            Tuple ``(x, y)`` in screen coordinates.
        """
        margin = 18
        gap = 10
        if screen is None:
            return margin, margin
        area = screen.availableGeometry()
        x = area.right() - width - margin
        y = area.bottom() - height - margin - index * (height + gap)
        return x, max(y, area.top() + margin)

    def _build_toast(self, event: TradeEvent, animate: bool = True) -> Any:
        """Construct the toast widget (must run on the Qt thread).

        Args:
            event: The trade event to display.
            animate: Run the fade-in animation. Disable for synchronous
                captures, otherwise the grab happens while opacity is still 0.

        Returns:
            The configured ``QWidget``.
        """
        from PyQt6 import QtCore, QtWidgets

        accent = {
            "BUY": "#16a34a",
            "SELL": "#dc2626",
            "CLOSE": "#2563eb",
        }.get(event.action.upper(), "#475569")

        window = QtWidgets.QWidget()
        window.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        window.setWindowTitle("quant_system trade alert")

        outer = QtWidgets.QVBoxLayout(window)
        outer.setContentsMargins(0, 0, 0, 0)

        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card.setStyleSheet(
            f"""
            #card {{
                background: #0f172a;
                border: 1px solid #1e293b;
                border-left: 6px solid {accent};
                border-radius: 10px;
            }}
            QLabel {{ color: #e2e8f0; background: transparent; }}
            """
        )
        outer.addWidget(card)

        row = QtWidgets.QHBoxLayout(card)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(12)

        badge = QtWidgets.QLabel(event.action.upper())
        badge.setStyleSheet(
            f"color:#ffffff; background:{accent}; padding:3px 8px;"
            "border-radius:4px; font-weight:700;"
        )
        row.addWidget(badge, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        texts = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel(f"<b>{event.symbol}</b>")
        title.setStyleSheet("font-size:14px;")
        body = QtWidgets.QLabel(event.toast_body())
        body.setStyleSheet("font-size:11px; color:#94a3b8;")
        texts.addWidget(title)
        texts.addWidget(body)
        row.addLayout(texts, stretch=1)

        close_btn = QtWidgets.QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "QPushButton{color:#64748b;background:transparent;border:none;font-size:15px;}"
            "QPushButton:hover{color:#e2e8f0;}"
        )
        close_btn.clicked.connect(window.close)
        row.addWidget(close_btn, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

        window.adjustSize()
        window.setFixedWidth(max(window.width(), 360))

        if animate:
            effect = QtWidgets.QGraphicsOpacityEffect(window)
            window.setGraphicsEffect(effect)
            animation = QtCore.QPropertyAnimation(effect, b"opacity")
            animation.setDuration(220)
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
            animation.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
            window._fade = animation  # keep a reference alive
        return window

    def _render_toast(self, event: TradeEvent) -> None:
        """Show a toast for ``event`` (Qt thread only).

        Args:
            event: The trade event.
        """
        from PyQt6 import QtCore

        if len(self._live) >= max(1, self.config.max_concurrent_toasts):
            oldest = self._live.pop(0)
            try:
                oldest.close()
            except Exception:  # pragma: no cover
                pass
        toast = self._build_toast(event)
        screen = self._app.primaryScreen() if self._app is not None else None
        x, y = self._position(len(self._live), toast.height(), toast.width(), screen)
        toast.move(x, y)
        toast.show()
        self._live.append(toast)
        self._delivered += 1
        QtCore.QTimer.singleShot(int(self.config.toast_duration_ms), toast.close)

    def _render_capture(self, event: TradeEvent, path: Path) -> None:
        """Render a toast offscreen and save it as a PNG (Qt thread only).

        Args:
            event: The trade event to render.
            path: Destination PNG path.
        """
        toast = self._build_toast(event, animate=False)
        toast.show()
        if self._app is not None:
            self._app.processEvents()
        from PyQt6 import QtGui

        # ``grab()`` captures the composited surface, which stays fully
        # transparent under the offscreen plugin; ``render()`` forces an
        # explicit paint of the widget and its children onto a filled backdrop.
        pixmap = QtGui.QPixmap(toast.size())
        pixmap.fill(QtGui.QColor("#94a3b8"))
        toast.render(pixmap)
        path.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(str(path))
        self._delivered += 1
        toast.close()
        toast.deleteLater()

    # ------------------------------------------------------------------ #
    def submit(self, event: TradeEvent) -> bool:
        """Queue a toast.

        Args:
            event: The trade event.

        Returns:
            ``True`` if queued.
        """
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:  # pragma: no cover - only under a burst
            return False

    def capture(self, event: TradeEvent, path: Path, timeout: float = 15.0) -> bool:
        """Render one toast to a PNG synchronously (used by verification).

        Args:
            event: The trade event to render.
            path: Destination PNG path.
            timeout: Seconds to wait for the Qt thread.

        Returns:
            ``True`` if the file was written.
        """
        if self._error is not None:
            return False
        done = threading.Event()
        try:
            self._queue.put_nowait((event, Path(path), done))
        except queue.Full:  # pragma: no cover
            return False
        if not done.wait(timeout=timeout):
            return False
        try:
            result = Path(path)
            return result.exists() and result.stat().st_size > 0
        except OSError:  # pragma: no cover - filesystem dependent
            return False

    def stop(self) -> None:
        """Stop the Qt event loop and release every widget reference."""
        if self._thread is not None and self._thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:  # pragma: no cover
                pass
            self._thread.join(timeout=5.0)
        self._live.clear()

    @property
    def delivered(self) -> int:
        """Number of toasts rendered."""
        return self._delivered

    @property
    def last_error(self) -> Optional[str]:
        """The last initialisation/render error, if any."""
        return self._error


class TkToastBackend:
    """tkinter fallback toast with the same queue-driven architecture."""

    name: str = "tk"

    def __init__(self, config: Optional[cfg.NotifierConfig] = None) -> None:
        """Initialise the backend.

        Args:
            config: Notifier configuration.
        """
        self.config = config or cfg.DEFAULT_SETTINGS.notifier
        self._queue: "queue.Queue[Optional[TradeEvent]]" = queue.Queue(maxsize=64)
        self._thread: Optional[threading.Thread] = None
        self._root: Any = None
        self._ready = threading.Event()
        self._delivered = 0
        self._error: Optional[str] = None

    def start(self) -> bool:
        """Start the tkinter loop on a background thread.

        Returns:
            ``True`` if tkinter initialised.
        """
        try:
            import tkinter  # noqa: PLC0415, F401
        except Exception as exc:  # pragma: no cover
            self._error = f"tkinter unavailable: {exc}"
            return False
        self._thread = threading.Thread(target=self._run, name="toast-tk", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self._error = self._error or "tkinter root failed to start"
            return False
        return self._error is None

    def _run(self) -> None:
        """Thread entry point: build the Tk root and poll the queue."""
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            self._root = root
            self._ready.set()
            self._poll()
            root.mainloop()
        except Exception as exc:  # pragma: no cover - headless hosts
            self._error = f"{type(exc).__name__}: {exc}"
            logger.info("tkinter toast backend unavailable (%s).", exc)
            self._ready.set()

    def _poll(self) -> None:
        """Drain the queue and reschedule (Tk thread only)."""
        try:
            while True:
                item = self._queue.get_nowait()
                if item is None:
                    if self._root is not None:
                        self._root.quit()
                    return
                self._render(item)
        except queue.Empty:
            pass
        if self._root is not None:
            self._root.after(120, self._poll)

    def _render(self, event: TradeEvent) -> None:
        """Build and show one Tk toast (Tk thread only).

        Args:
            event: The trade event.
        """
        import tkinter as tk

        accent = {"BUY": "#16a34a", "SELL": "#dc2626", "CLOSE": "#2563eb"}.get(
            event.action.upper(), "#475569"
        )
        window = tk.Toplevel(self._root)
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.configure(bg="#0f172a")
        frame = tk.Frame(window, bg="#0f172a", highlightthickness=1,
                         highlightbackground="#1e293b")
        frame.pack(fill="both", expand=True, padx=1, pady=1)
        bar = tk.Frame(frame, bg=accent, width=6)
        bar.pack(side="left", fill="y")
        text_frame = tk.Frame(frame, bg="#0f172a")
        text_frame.pack(side="left", fill="both", expand=True, padx=10, pady=8)
        tk.Label(text_frame, text=f"{event.action.upper()}  {event.symbol}",
                 fg="#e2e8f0", bg="#0f172a",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(text_frame, text=event.toast_body(), fg="#94a3b8", bg="#0f172a",
                 font=("Segoe UI", 9)).pack(anchor="w")
        window.update_idletasks()
        width, height = window.winfo_width(), window.winfo_height()
        x = window.winfo_screenwidth() - width - 18
        y = window.winfo_screenheight() - height - 18 - (self._delivered % 3) * (height + 10)
        window.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        window.after(int(self.config.toast_duration_ms), window.destroy)
        self._delivered += 1

    def submit(self, event: TradeEvent) -> bool:
        """Queue a toast.

        Args:
            event: The trade event.

        Returns:
            ``True`` if queued.
        """
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:  # pragma: no cover
            return False

    def stop(self) -> None:
        """Stop the Tk loop."""
        if self._thread is not None and self._thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:  # pragma: no cover
                pass
            self._thread.join(timeout=5.0)

    @property
    def delivered(self) -> int:
        """Number of toasts rendered."""
        return self._delivered

    @property
    def last_error(self) -> Optional[str]:
        """The last error, if any."""
        return self._error


def resolve_toast_backend(
    backend: ToastBackend | str = ToastBackend.AUTO,
    config: Optional[cfg.NotifierConfig] = None,
) -> Tuple[Any, str]:
    """Select and start a toast backend with graceful degradation.

    Args:
        backend: Requested backend.
        config: Notifier configuration.

    Returns:
        Tuple ``(backend_instance, resolution_note)``. The instance is always
        usable: the null recorder is returned if nothing else starts.
    """
    config = config or cfg.DEFAULT_SETTINGS.notifier
    requested = ToastBackend(backend)
    candidates: Sequence[ToastBackend]
    if requested is ToastBackend.AUTO:
        candidates = (ToastBackend.PYQT, ToastBackend.TK, ToastBackend.NONE)
    else:
        candidates = (requested, ToastBackend.NONE)

    notes: List[str] = []
    for candidate in candidates:
        if candidate is ToastBackend.NONE:
            instance: Any = NullToastBackend(config)
            instance.start()
            notes.append("null recorder (headless)")
            return instance, " -> ".join(notes)
        klass = PyQtToastBackend if candidate is ToastBackend.PYQT else TkToastBackend
        instance = klass(config)
        try:
            if instance.start():
                return instance, f"{candidate.value} started"
            notes.append(f"{candidate.value}: {instance.last_error}")
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"{candidate.value}: {exc}")
    fallback = NullToastBackend(config)
    fallback.start()
    return fallback, " -> ".join(notes) or "no backend"


# --------------------------------------------------------------------------- #
# Notifier engine
# --------------------------------------------------------------------------- #
class NotifierEngine:
    """Thread-safe facade that turns confirmed fills into toasts and speech.

    Attributes:
        config: Notifier configuration.
        toast_backend: The resolved toast backend instance.
        voice: The :class:`VoiceWorker` (started lazily when voice is enabled).
    """

    def __init__(
        self,
        config: Optional[cfg.NotifierConfig] = None,
        regime_provider: Optional[Callable[[], int]] = None,
    ) -> None:
        """Initialise the engine.

        Args:
            config: Notifier configuration; defaults to ``settings.notifier``.
            regime_provider: Optional callable returning the current HMM regime,
                used when a caller does not pass one explicitly.
        """
        self.config: cfg.NotifierConfig = config or cfg.DEFAULT_SETTINGS.notifier
        self.regime_provider = regime_provider
        self.toast_backend: Any = NullToastBackend(self.config)
        self.backend_note: str = "not started"
        self.voice: Optional[VoiceWorker] = None
        self._lock = threading.RLock()
        self._events: List[TradeEvent] = []
        self._seen: Dict[str, float] = {}
        self._filtered: Dict[str, int] = {reason.value: 0 for reason in FilterReason}
        self._started = False
        #: Snapshot of the last voice worker, retained after :meth:`stop`.
        self._voice_summary: Dict[str, Any] = {"enabled": False}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        """Start the toast backend and (if enabled) the voice worker.

        Returns:
            ``True`` if the engine is running.
        """
        with self._lock:
            if self._started:
                return True
            if self.config.toast_enabled:
                self.toast_backend, self.backend_note = resolve_toast_backend(
                    self.config.toast_backend, self.config
                )
            else:
                self.toast_backend = NullToastBackend(self.config)
                self.toast_backend.start()
                self.backend_note = "toasts disabled by configuration"

            if self.config.voice_enabled:
                self.voice = VoiceWorker(
                    mode=self.config.voice_mode,
                    rate=self.config.voice_rate,
                    volume=self.config.voice_volume,
                    output_dir=self.config.voice_output_dir,
                    queue_maxsize=self.config.queue_maxsize,
                )
                self.voice.start()
                self.voice.wait_until_ready(timeout=15.0)

            self._started = True
            logger.info(
                "Notifier started (toast=%s voice=%s)", self.toast_backend.name,
                "on" if self.voice is not None else "off",
            )
            return True

    def stop(self) -> None:
        """Stop the voice worker and the toast backend."""
        with self._lock:
            if not self._started:
                return
            # Graceful drain: the toast backend renders asynchronously (Qt/Tk
            # poll their queue on a timer), so give it a bounded window to show
            # everything that was submitted before tearing the thread down.
            deadline = time.monotonic() + 2.0
            while (
                len(self._events) > self.toast_backend.delivered
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)

            if self.voice is not None:
                worker = self.voice
                # Drain first: stop() joins the worker, so the counters are only
                # final *after* it returns. Snapshotting beforehand reports zeros.
                thread_started = worker.is_alive()
                worker.stop(timeout=5.0)
                self._voice_summary = {
                    "enabled": True,
                    "mode": worker.mode.value,
                    "available": worker.available,
                    "thread_started": thread_started,
                    "thread_stopped": not worker.is_alive(),
                    "queue_depth": int(worker._queue.qsize()),
                    "spoken": int(worker.spoken),
                    "failed": int(worker.failed),
                    "skipped": int(worker.skipped),
                    "dropped": int(worker.dropped),
                    "last_error": worker.last_error,
                    "last_file": str(worker.last_file) if worker.last_file else None,
                    "files": [str(path) for path in worker.files],
                }
                self.voice = None
            try:
                self.toast_backend.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Toast backend stop failed: %s", exc)
            self._started = False

    def __enter__(self) -> "NotifierEngine":
        """Start the engine on context entry."""
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Stop the engine on context exit."""
        self.stop()

    # ------------------------------------------------------------------ #
    # Event intake
    # ------------------------------------------------------------------ #
    @staticmethod
    def is_trade_fill(report: Any, allow_partial: bool = False) -> bool:
        """Whether an :class:`OrderReport` represents a confirmed fill.

        Rejects pending/accepted/working orders (e.g. unfilled limit orders),
        cancellations, rejections and zero-quantity reports.

        Args:
            report: The order report to test.
            allow_partial: Also accept ``PARTIALLY_FILLED``.

        Returns:
            ``True`` when the report should be announced.
        """
        from quant_system.execution.brokers.base import OrderStatus

        status = getattr(report, "status", None)
        if status is None:
            return False
        if isinstance(status, str):
            try:
                status = OrderStatus(status)
            except ValueError:
                return False
        if status == OrderStatus.FILLED:
            pass
        elif status == OrderStatus.PARTIALLY_FILLED:
            if not allow_partial:
                return False
        else:
            return False
        filled = getattr(report, "filled_quantity", 0.0) or 0.0
        return float(filled) > 0.0

    def notify_fill(
        self,
        report: Any,
        order: Any = None,
        *,
        regime_state: Optional[int] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        realised_pnl: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> bool:
        """Announce a confirmed fill.

        Args:
            report: The broker's :class:`OrderReport`.
            order: The originating :class:`Order` (gives side/strategy/SL/TP).
            regime_state: Active HMM regime; falls back to ``regime_provider``.
            stop_loss: Stop-loss price for the toast.
            take_profit: Take-profit price for the toast.
            realised_pnl: Realised PnL if the fill closed a position.
            timestamp: Event timestamp.

        Returns:
            ``True`` if an alert was dispatched.
        """
        with self._lock:
            if not self.config.enabled:
                return self._reject(FilterReason.DISABLED)
            if not self.is_trade_fill(report, self.config.notify_on_partial):
                return self._reject(FilterReason.NOT_FILLED)

            event = self._build_event(
                report=report,
                order=order,
                regime_state=regime_state,
                stop_loss=stop_loss,
                take_profit=take_profit,
                realised_pnl=realised_pnl,
                timestamp=timestamp,
            )
            if event is None:
                return self._reject(FilterReason.ZERO_QUANTITY)
            if self.config.dedupe and not self._register(event.order_id):
                return self._reject(FilterReason.DUPLICATE)

            self._events.append(event)
            logger.info("TRADE ALERT | %s | %s", event.toast_title(), event.toast_body())
            try:
                self.toast_backend.submit(event)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Toast dispatch failed: %s", exc)
            if self.voice is not None:
                self.voice.submit(event.speech_text(), key=f"{event.action}_{event.symbol}")
            return True

    def notify_mt5_result(
        self,
        order: Any,
        result: Any,
        mt5_module: Any = None,
        **kwargs: Any,
    ) -> bool:
        """Announce a MetaTrader 5 execution result.

        Fires only when ``result.retcode == mt5.TRADE_RETCODE_DONE``; every other
        retcode (requote, invalid volume, market closed, ...) is filtered out.

        Args:
            order: The originating :class:`Order`.
            result: The MT5 ``OrderSendResult`` named tuple.
            mt5_module: The imported ``MetaTrader5`` module (imported on demand
                when omitted, only to read ``TRADE_RETCODE_DONE``).
            **kwargs: Forwarded to :meth:`notify_fill`.

        Returns:
            ``True`` if an alert was dispatched.
        """
        if mt5_module is None:
            try:
                import MetaTrader5 as mt5_module  # noqa: PLC0415 - optional
            except Exception:
                mt5_module = None
        retcode = getattr(result, "retcode", None)
        done_code = getattr(mt5_module, "TRADE_RETCODE_DONE", 10009)
        if retcode is None or retcode != done_code:
            with self._lock:
                return self._reject(FilterReason.NOT_FILLED)

        from quant_system.execution.brokers.base import OrderReport, OrderStatus

        side = str(getattr(order, "side", "buy")).lower()
        report = OrderReport(
            client_order_id=str(getattr(order, "client_order_id", "") or ""),
            broker_order_id=str(getattr(result, "order", "") or ""),
            status=OrderStatus.FILLED,
            filled_quantity=float(getattr(result, "volume", 0.0) or 0.0),
            average_fill_price=float(getattr(result, "price", 0.0) or 0.0),
            message="MT5 TRADE_RETCODE_DONE",
        )
        del side
        return self.notify_fill(report, order, **kwargs)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _reject(self, reason: FilterReason) -> bool:
        """Record a filtered message.

        Args:
            reason: Why the message was dropped.

        Returns:
            Always ``False``.
        """
        self._filtered[reason.value] = self._filtered.get(reason.value, 0) + 1
        return False

    def _register(self, order_id: str) -> bool:
        """De-duplicate on order id.

        Args:
            order_id: The order identifier.

        Returns:
            ``True`` if the id is new.
        """
        if not order_id:
            return True
        if order_id in self._seen:
            return False
        self._seen[order_id] = time.time()
        if len(self._seen) > 4096:  # bound memory on long sessions
            for stale in sorted(self._seen, key=self._seen.get)[:1024]:  # type: ignore[arg-type]
                self._seen.pop(stale, None)
        return True

    def _build_event(
        self,
        report: Any,
        order: Any,
        regime_state: Optional[int],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        realised_pnl: Optional[float],
        timestamp: Optional[pd.Timestamp],
    ) -> Optional[TradeEvent]:
        """Assemble a :class:`TradeEvent` from a fill report and its order.

        Args:
            report: The order report.
            order: The originating order (may be ``None``).
            regime_state: Explicit regime override.
            stop_loss: Stop-loss price override.
            take_profit: Take-profit price override.
            realised_pnl: Realised PnL override.
            timestamp: Event timestamp override.

        Returns:
            The event, or ``None`` when the fill carries no quantity.
        """
        from quant_system.execution.brokers.base import OrderSide

        volume = float(getattr(report, "filled_quantity", 0.0) or 0.0)
        price = float(getattr(report, "average_fill_price", 0.0) or 0.0)
        if volume <= 0.0:
            return None

        side = getattr(order, "side", None)
        if side is None:
            action = "CLOSE"
        elif OrderSide(side) == OrderSide.BUY:
            action = "BUY"
        else:
            action = "SELL"

        if regime_state is None and self.regime_provider is not None:
            try:
                regime_state = int(self.regime_provider())
            except Exception:  # pragma: no cover - defensive
                regime_state = None

        symbol = str(getattr(order, "symbol", "") or getattr(report, "symbol", "") or "?")
        return TradeEvent(
            action=action,
            symbol=symbol,
            volume=volume,
            price=price,
            regime_state=int(regime_state if regime_state is not None else cfg.STATE_RANGE_BOUND),
            stop_loss=stop_loss if stop_loss is not None else getattr(order, "stop_loss", None),
            take_profit=take_profit if take_profit is not None else getattr(order, "take_profit", None),
            strategy=str(getattr(order, "strategy", "") or ""),
            order_id=str(
                getattr(report, "broker_order_id", None)
                or getattr(report, "client_order_id", "")
                or ""
            ),
            realised_pnl=realised_pnl,
            timestamp=_utc_now() if timestamp is None else pd.Timestamp(timestamp),
        )

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    @property
    def events(self) -> List[TradeEvent]:
        """Every announced :class:`TradeEvent`, oldest first."""
        with self._lock:
            return list(self._events)

    @property
    def filtered_counts(self) -> Dict[str, int]:
        """Counts of dropped messages keyed by :class:`FilterReason`."""
        with self._lock:
            return dict(self._filtered)

    def stats(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the notifier.

        Returns:
            Dictionary with backend info, announced/filtered counts and voice
            worker counters.
        """
        with self._lock:
            voice_stats: Dict[str, Any] = (
                {
                    "enabled": True,
                    "mode": self.voice.mode.value,
                    "available": self.voice.available,
                    "spoken": self.voice.spoken,
                    "failed": self.voice.failed,
                    "skipped": self.voice.skipped,
                    "dropped": self.voice.dropped,
                    "last_error": self.voice.last_error,
                    "last_file": str(self.voice.last_file)
                    if self.voice.last_file
                    else None,
                    "files": [str(path) for path in self.voice.files],
                }
                if self.voice is not None
                else dict(self._voice_summary)
            )
            if False:
                voice_stats.update(
                    {
                        "mode": self.voice.mode.value,
                        "available": self.voice.available,
                        "spoken": self.voice.spoken,
                        "failed": self.voice.failed,
                        "skipped": self.voice.skipped,
                        "dropped": self.voice.dropped,
                        "last_error": self.voice.last_error,
                        "last_file": str(self.voice.last_file)
                        if self.voice.last_file
                        else None,
                    }
                )
            return {
                "started": self._started,
                "toast_backend": self.toast_backend.name,
                "backend_note": self.backend_note,
                "toasts_delivered": self.toast_backend.delivered,
                "announced": len(self._events),
                "filtered": self.filtered_counts,
                "voice": voice_stats,
            }


__all__: List[str] = [
    "VoiceMode",
    "ToastBackend",
    "FilterReason",
    "TradeEvent",
    "VoiceWorker",
    "NullToastBackend",
    "PyQtToastBackend",
    "TkToastBackend",
    "NotifierEngine",
    "resolve_toast_backend",
    "speak_number",
    "spoken_symbol",
    "SPOKEN_SYMBOLS",
    "SPOKEN_REGIMES",
]
