"""Orchestrator and execution entrypoint for the quant_system framework.

Usage examples
--------------
Full backtest on real data (falls back to synthetic if the network is down)::

    python main.py --mode backtest --source auto --start 2016-01-01 --end 2025-06-30

Walk-forward optimisation with the out-of-sample equity curve stitched::

    python main.py --mode walk-forward --start 2015-01-01 --end 2025-06-30 --plot

Offline, fully reproducible synthetic run (used by CI)::

    python main.py --mode backtest --source synthetic --no-cache

Paper-trade through the simulated broker, or go live via MT5 / FIX::

    python main.py --mode live --broker simulated
    python main.py --mode live --broker mt5
    python main.py --mode live --broker fix
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import logging.handlers
import signal
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Make the package importable when this file is executed directly.
if __package__ in (None, ""):  # pragma: no cover - script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "quant_system"

from quant_system.analytics import metrics as ametrics  # noqa: E402
from quant_system.utils.notifier import (  # noqa: E402
    NotifierEngine,
    ToastBackend,
    VoiceMode,
)

from quant_system.backtesting.engine import (  # noqa: E402
    BacktestEngine,
    BacktestResult,
    build_default_regimes,
)
from quant_system.backtesting.walk_forward import (  # noqa: E402
    WalkForwardOptimizer,
    WalkForwardResult,
)
from quant_system.config import settings as cfg  # noqa: E402
from quant_system.data.ingestion import DataIngestion  # noqa: E402
from quant_system.execution.brokers.base import (  # noqa: E402
    BrokerBase,
    Order,
    OrderReport,
    OrderSide,
)
from quant_system.execution.brokers.mt5_broker import (  # noqa: E402
    MT5UnavailableError,
)
from quant_system.execution.brokers.router import OrderRouter  # noqa: E402
from quant_system.execution.portfolio import Portfolio, SizingEngine  # noqa: E402
from quant_system.models.hmm_switchboard import (  # noqa: E402
    CausalRegimeStreamer,
    RegimeStreamResult,
)
from quant_system.strategies.adaptive_momentum import AdaptiveMomentumStrategy  # noqa: E402
from quant_system.strategies.base import BaseStrategy, StrategyContext  # noqa: E402
from quant_system.strategies.risk_preservation import RiskPreservationStrategy  # noqa: E402
from quant_system.strategies.stat_arb import build_default_stat_arb_book  # noqa: E402

LOGGER = logging.getLogger("quant_system")


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
@dataclass
class PipelineOutput:
    """Artifacts produced by a pipeline run.

    Attributes:
        report_dir: Directory the artifacts were written to.
        files: Mapping of logical name -> written path.
        equity: Equity curve (backtest or stitched OOS).
        metrics: Headline metrics.
        regime_metrics: Regime-conditional breakdown.
    """

    report_dir: Path
    files: Dict[str, Path]
    equity: pd.Series
    metrics: Dict[str, float]
    regime_metrics: pd.DataFrame


LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def configure_logging(
    verbose: bool = False,
    log_dir: Optional[Path] = None,
    enable_file_log: bool = True,
) -> Optional[Path]:
    """Configure the root logger for console **and** disk.

    A rotating file handler is attached so an unattended live session leaves a
    durable audit trail that survives a terminal disconnect.

    Args:
        verbose: Enable DEBUG-level logging for the framework packages.
        log_dir: Directory for the log file; defaults to ``settings.LOGS_DIR``.
        enable_file_log: Attach the rotating file handler.

    Returns:
        The path of the log file, or ``None`` when file logging is disabled.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_path: Optional[Path] = None

    if enable_file_log:
        directory = Path(log_dir) if log_dir else cfg.LOGS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "quant_system.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=20 * 1024 * 1024,  # 20 MiB per file
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S", handlers=handlers, force=True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("hmmlearn").setLevel(logging.WARNING)
    return log_path


def _ensure_reports_dir(path: Optional[Path] = None) -> Path:
    """Create (and return) the reports directory.

    Args:
        path: Optional override; defaults to ``settings.REPORTS_DIR``.

    Returns:
        The reports directory path.
    """
    directory = Path(path) if path else cfg.REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_text(path: Path, content: str) -> Path:
    """Write a text artifact.

    Args:
        path: Destination path.
        content: File content.

    Returns:
        The destination path.
    """
    path.write_text(content, encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    return path


def plot_equity(
    equity: pd.Series,
    states: Optional[pd.Series],
    destination: Path,
    title: str = "Equity curve",
) -> Optional[Path]:
    """Render an equity curve with regime shading.

    Args:
        equity: Equity curve.
        states: Optional regime series used for background shading.
        destination: Output image path.
        title: Plot title.

    Returns:
        The path written, or ``None`` when matplotlib is unavailable.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.warning("matplotlib unavailable; skipping the equity plot.")
        return None

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[3, 1])
    axes[0].plot(equity.index, equity.values, color="#1f77b4", linewidth=1.4, label="Equity")
    axes[0].set_title(title)
    axes[0].set_ylabel("Equity")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left")

    drawdown = ametrics.drawdown_series(equity) * 100.0
    axes[1].fill_between(drawdown.index, drawdown.values, color="#d62728", alpha=0.5)
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(alpha=0.3)

    if states is not None and not states.empty:
        aligned = states.reindex(equity.index).ffill()
        colours = {0: "#2ca02c", 1: "#1f77b4", 2: "#d62728"}
        previous: Optional[int] = None
        start: Optional[pd.Timestamp] = None
        for timestamp, value in aligned.items():
            value = int(value)
            if previous is None:
                previous, start = value, timestamp
                continue
            if value != previous:
                axes[0].axvspan(
                    start, timestamp, color=colours.get(previous, "#7f7f7f"), alpha=0.10
                )
                previous, start = value, timestamp
        if start is not None:
            axes[0].axvspan(
                start, aligned.index[-1], color=colours.get(previous, "#7f7f7f"), alpha=0.10
            )

    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    LOGGER.info("Wrote %s", destination)
    return destination


# --------------------------------------------------------------------------- #
# Strategy book
# --------------------------------------------------------------------------- #
def build_strategy_book(
    coint_method: Optional[str] = None,
) -> List[BaseStrategy]:
    """Build the standard regime-routed strategy book.

    Args:
        coint_method: Overrides
            :attr:`~quant_system.config.settings.StatArbConfig.coint_method`
            (``"coint"``, ``"adf_spread"`` or ``"either"``).

    Returns:
        ``[MetalsStatArb, FXStatArb, AdaptiveMomentum]``.

    Raises:
        ValueError: If ``coint_method`` is not one of the supported options.
    """
    valid_methods = {"coint", "adf_spread", "either"}
    from dataclasses import replace

    if coint_method is not None and coint_method not in valid_methods:
        raise ValueError(
            f"coint_method must be one of {sorted(valid_methods)}; got {coint_method!r}."
        )
    stat_arb_config = cfg.DEFAULT_SETTINGS.stat_arb
    if coint_method is not None:
        stat_arb_config = replace(stat_arb_config, coint_method=coint_method)

    strategies: List[BaseStrategy] = list(
        build_default_stat_arb_book(config=stat_arb_config)
    )
    strategies.append(AdaptiveMomentumStrategy())
    return strategies


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def load_data(
    source: str,
    start: str,
    end: str,
    interval: str,
    use_cache: bool,
) -> Tuple[Dict[str, pd.DataFrame], DataIngestion]:
    """Load the aligned universe.

    Args:
        source: ``"auto"``, ``"yfinance"`` or ``"synthetic"``.
        start: Inclusive start date.
        end: Inclusive end date.
        interval: Bar interval.
        use_cache: Enable the pickle cache.

    Returns:
        Tuple ``(data, ingestion)``.
    """
    ingestion = DataIngestion(source=source, use_cache=use_cache)
    data = ingestion.fetch_universe(start=start, end=end, interval=interval)
    LOGGER.info(
        "Loaded %d symbols x %d bars from %s",
        len(data),
        len(next(iter(data.values()))),
        ingestion.sources_used or {"n/a"},
    )
    return data, ingestion


def run_backtest(
    data: Dict[str, pd.DataFrame],
    settings_obj: Optional[cfg.Settings] = None,
    strategies: Optional[List[BaseStrategy]] = None,
    regime_result: Optional[RegimeStreamResult] = None,
    initial_capital: Optional[float] = None,
    coint_method: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[BacktestResult, RegimeStreamResult]:
    """Run the full backtest pipeline.

    Args:
        data: Aligned OHLCV universe.
        settings_obj: Settings override.
        strategies: Strategy book; defaults to :func:`build_strategy_book`.
        regime_result: Precomputed regimes; computed on the fly if omitted.
        initial_capital: Starting equity.
        coint_method: Overrides the stat-arb stationarity gate.
        verbose: Verbose logging.

    Returns:
        Tuple ``(result, regime_result)``.
    """
    settings_obj = settings_obj or cfg.DEFAULT_SETTINGS
    if regime_result is None:
        LOGGER.info("Streaming HMM regimes over %d bars...", len(next(iter(data.values()))))
        regime_result = build_default_regimes(
            data, config=settings_obj.hmm, interval=settings_obj.backtest.interval,
            verbose=verbose,
        )
    distribution = regime_result.states.value_counts(normalize=True).sort_index()
    LOGGER.info(
        "Regime distribution: %s",
        {cfg.REGIME_LABELS.get(int(k), k): f"{v:.1%}" for k, v in distribution.items()},
    )

    engine = BacktestEngine(
        data=data,
        strategies=list(strategies) if strategies else build_strategy_book(coint_method),
        regime_states=regime_result,
        initial_capital=initial_capital,
        verbose=verbose,
    )
    result = engine.run()
    return result, regime_result


def report_backtest(
    result: BacktestResult,
    regime_result: RegimeStreamResult,
    report_dir: Path,
    prefix: str = "backtest",
    make_plot: bool = True,
) -> PipelineOutput:
    """Persist backtest artifacts.

    Args:
        result: Backtest result.
        regime_result: Regime stream used by the run.
        report_dir: Destination directory.
        prefix: Filename prefix.
        make_plot: Render the equity-curve PNG.

    Returns:
        The :class:`PipelineOutput`.
    """
    files: Dict[str, Path] = {}
    files["equity"] = result.equity_curve.rename("equity").to_frame().to_csv(
        report_dir / f"{prefix}_equity.csv"
    ) and (report_dir / f"{prefix}_equity.csv")
    trades_path = report_dir / f"{prefix}_trades.csv"
    result.trades.to_csv(trades_path, index=False)
    files["trades"] = trades_path

    if not result.position_weights.empty:
        weights_path = report_dir / f"{prefix}_weights.csv"
        result.position_weights.to_csv(weights_path)
        files["weights"] = weights_path

    if not result.regime_metrics.empty:
        regime_path = report_dir / f"{prefix}_regime_metrics.csv"
        result.regime_metrics.to_csv(regime_path)
        files["regime_metrics"] = regime_path

    summary_lines = [
        "=" * 78,
        "BACKTEST PERFORMANCE SUMMARY",
        "=" * 78,
        ametrics.format_summary(result.metrics),
        "",
        "REGIME-CONDITIONAL BREAKDOWN",
        "-" * 78,
        result.regime_metrics.to_string()
        if not result.regime_metrics.empty
        else "no regime data",
        "",
        "REGIME TRANSITION MATRIX",
        "-" * 78,
        ametrics.regime_transitions(regime_result.states).to_string(),
        "",
        f"Events logged: {len(result.events)}",
        "=" * 78,
    ]
    files["summary"] = _write_text(
        report_dir / f"{prefix}_summary.txt", "\n".join(summary_lines)
    )

    if make_plot and not result.equity_curve.empty:
        plot_path = plot_equity(
            result.equity_curve,
            result.regime_states,
            cfg.FIGURES_DIR / f"{prefix}_equity.png",
            title=f"{prefix.title()} equity curve",
        )
        if plot_path:
            files["plot"] = plot_path

    return PipelineOutput(
        report_dir=report_dir,
        files=files,
        equity=result.equity_curve,
        metrics=result.metrics,
        regime_metrics=result.regime_metrics,
    )


def run_walk_forward(
    data: Dict[str, pd.DataFrame],
    regime_states: pd.Series,
    regime_probabilities: Optional[pd.DataFrame],
    settings_obj: Optional[cfg.Settings] = None,
    warmup_bars: int = 252,
    initial_capital: Optional[float] = None,
    coint_method: Optional[str] = None,
    grid: str = "full",
    verbose: bool = True,
) -> WalkForwardResult:
    """Run the walk-forward optimisation study.

    Args:
        data: Aligned OHLCV universe.
        regime_states: Causal regime series.
        regime_probabilities: Regime probability frame.
        settings_obj: Settings override.
        warmup_bars: Indicator warm-up per window.
        initial_capital: Starting equity per engine run.
        coint_method: Overrides the stat-arb stationarity gate.
        verbose: Progress logging.

    Returns:
        The :class:`WalkForwardResult`.
    """
    settings_obj = settings_obj or cfg.DEFAULT_SETTINGS
    if grid == "fast":
        from dataclasses import replace

        settings_obj = replace(
            settings_obj,
            walk_forward=replace(
                settings_obj.walk_forward,
                z_entry_grid=(1.6, 1.8, 2.0, 2.2, 2.4),
                momentum_grid=((15, 40), (20, 50)),
            ),
        )
    optimizer = WalkForwardOptimizer(
        data=data,
        regime_states=regime_states,
        regime_probabilities=regime_probabilities,
        config=settings_obj.walk_forward,
        strategy_factory=lambda: build_strategy_book(coint_method),
        warmup_bars=warmup_bars,
        initial_capital=initial_capital,
        verbose=verbose,
    )
    return optimizer.run()


# --------------------------------------------------------------------------- #
# Live trading
# --------------------------------------------------------------------------- #
def build_broker(mode: str, portfolio: Portfolio) -> BrokerBase:
    """Instantiate a broker adapter.

    Args:
        mode: ``"simulated"``, ``"mt5"`` or ``"fix"``.
        portfolio: Portfolio used by the simulated adapter.

    Returns:
        A :class:`BrokerBase` instance.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    from quant_system.execution.brokers.fix_broker import FIXBroker  # noqa: PLC0415
    from quant_system.execution.brokers.mt5_broker import MT5Broker  # noqa: PLC0415
    from quant_system.execution.brokers.simulated_broker import SimulatedBroker  # noqa: PLC0415

    if mode == "simulated":
        return SimulatedBroker(portfolio=portfolio)
    if mode == "mt5":
        return MT5Broker()
    if mode == "fix":
        return FIXBroker()
    raise ValueError(f"Unknown broker mode {mode!r}; use simulated, mt5 or fix.")
def _announce_fills(
    notifier: "NotifierEngine",
    orders: Sequence["Order"],
    reports: Sequence["OrderReport"],
    regime_state: int,
    portfolio: "Portfolio",
) -> None:
    """Forward confirmed fills to the notifier (toast + voice).

    Only fills that the notifier's own gate accepts are announced, so pending
    acknowledgements, partials, cancellations and rejections never reach the
    desktop or the speaker.  Realised P&L is attached when the fill closed (or
    reduced) a position, which makes the spoken alert report the round trip.

    Args:
        notifier: The running notifier engine.
        orders: Orders submitted this cycle, aligned with ``reports``.
        reports: Broker reports, one per order.
        regime_state: Active HMM regime id for the bar.
        portfolio: Portfolio used to attribute realised P&L to closing fills.
    """
    known_trades = len(portfolio.trades)
    for order, report in zip(orders, reports):
        realised: Optional[float] = None
        if len(portfolio.trades) > known_trades:
            realised = float(portfolio.trades[-1].net_pnl)
            known_trades = len(portfolio.trades)
        metadata = getattr(order, "metadata", None) or {}
        notifier.notify_fill(
            order=order,
            report=report,
            regime_state=regime_state,
            stop_loss=_to_float(metadata.get("stop_loss")),
            take_profit=_to_float(metadata.get("take_profit")),
            realised_pnl=realised,
        )


def _attach_stops(
    orders: Sequence["Order"],
    data: Dict[str, pd.DataFrame],
    bar_index: int,
    stop_multiples: Dict[str, Optional[float]],
    regime_state: int,
    risk_config: cfg.RiskConfig,
) -> None:
    """Annotate orders with their ATR stop so the alert can display it.

    The stop distance is ``multiple x ATR(14)``, tightened by
    :attr:`~quant_system.config.settings.RiskConfig.shock_stop_multiplier`
    while State 2 (risk preservation) is active.

    Args:
        orders: Orders to annotate (mutated in place).
        data: Symbol -> OHLCV frames.
        bar_index: Bar position used to read the current ATR.
        stop_multiples: Per-symbol ATR multiple supplied by the strategy layer.
        regime_state: Active HMM regime id.
        risk_config: Risk configuration supplying the default multiple.
    """
    from quant_system.data.preprocessing import wilder_atr

    atr_cache: Dict[str, float] = {}
    for order in orders:
        frame = data.get(order.symbol)
        if frame is None:
            continue
        atr = atr_cache.get(order.symbol)
        if atr is None:
            series = wilder_atr(frame["high"], frame["low"], frame["close"], period=14)
            atr = float(series.iloc[bar_index]) if bar_index < len(series) else float("nan")
            atr_cache[order.symbol] = atr
        if not np.isfinite(atr) or atr <= 0:
            continue
        multiple = float(stop_multiples.get(order.symbol) or risk_config.atr_stop_multiple)
        if regime_state == cfg.STATE_SHOCK:
            multiple *= risk_config.shock_stop_multiplier
        price = float(order.price or frame.iloc[bar_index]["close"])
        if order.price is None:
            order.price = price
        distance = multiple * atr
        order.metadata["stop_loss"] = (
            price - distance if order.side is OrderSide.BUY else price + distance
        )
        order.metadata["stop_atr_multiple"] = multiple


def _to_float(value: Any) -> Optional[float]:
    """Coerce ``value`` to ``float``, mapping ``None``/garbage to ``None``.

    Args:
        value: Candidate value from broker metadata.

    Returns:
        The float, or ``None`` when it cannot be interpreted as a number.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


class LiveSessionController:
    """Cooperative shutdown controller for an unattended live session.

    Three independent stop mechanisms are supported, all of which let the
    running cycle finish (so an order is never abandoned half-routed):

    1. **SIGINT / SIGTERM** — the first signal requests a graceful stop; a second
       one aborts immediately (without flattening, because there is no time).
    2. **Halt file** — creating ``reports/HALT`` stops the loop at the next cycle
       boundary. This is the remote/ops-friendly kill switch: it works from an
       SSH session that never touched the bot's terminal.
    3. **Iteration bound** — ``--max-iterations`` (unchanged).

    Attributes:
        halt_file: Optional sentinel path checked every cycle.
        signal_count: Number of stop signals received so far.
    """

    def __init__(self, halt_file: Optional[Path] = None) -> None:
        """Initialise the controller.

        Args:
            halt_file: Sentinel file whose existence requests a shutdown.
        """
        self.halt_file: Optional[Path] = Path(halt_file) if halt_file else None
        self.signal_count: int = 0
        self._previous: Dict[int, Any] = {}

    # ---------------------------------------------------------------- #
    def install(self) -> None:
        """Install the SIGINT/SIGTERM handlers (main thread only)."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except ValueError:  # pragma: no cover - non-main thread
                LOGGER.debug("Could not install handler for %s.", sig)

    def restore(self) -> None:
        """Restore the handlers that were in place before :meth:`install`."""
        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, TypeError):  # pragma: no cover
                pass

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Record a shutdown request, aborting on the second signal.

        Args:
            signum: Signal number.
            frame: Interrupted stack frame (unused).
        """
        del frame
        self.signal_count += 1
        try:
            name = signal.Signals(signum).name
        except ValueError:  # pragma: no cover
            name = str(signum)
        if self.signal_count >= 2:
            LOGGER.critical(
                "Second %s received - aborting immediately. Positions are LEFT OPEN; "
                "flatten them manually.",
                name,
            )
            raise SystemExit(130)
        LOGGER.warning(
            "%s received - finishing the current cycle, then shutting down cleanly. "
            "Send it again to abort right now.",
            name,
        )

    # ---------------------------------------------------------------- #
    @property
    def requested(self) -> bool:
        """Whether a shutdown has been requested by any mechanism."""
        if self.signal_count:
            return True
        return self.halt_file is not None and self.halt_file.exists()

    @property
    def reason(self) -> str:
        """Short code describing which mechanism requested the shutdown."""
        if self.signal_count:
            return "signal"
        if self.halt_file is not None and self.halt_file.exists():
            return "halt_file"
        return ""

    def sleep(self, seconds: float) -> None:
        """Interruptible sleep used between polling cycles.

        Args:
            seconds: Nominal sleep duration.
        """
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            if self.requested:
                return
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))


def run_live(
    source: str,
    broker_mode: str,
    start: str,
    end: str,
    interval: str,
    max_iterations: int = 5,
    poll_seconds: float = 5.0,
    initial_capital: Optional[float] = None,
    demo_config: Optional[cfg.DemoConfig] = None,
    risk_config: Optional[cfg.RiskConfig] = None,
    notifier: Optional["NotifierEngine"] = None,
    flatten_on_exit: bool = False,
    halt_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the live/paper trading loop.

    The loop reuses the *exact* backtest machinery: fresh data -> causal regime
    decode -> strategy signals -> sizing -> broker orders via the router.

    Args:
        source: Data source for the decision engine.
        broker_mode: ``"simulated"``, ``"mt5"`` or ``"fix"``.
        start: History start for indicator warm-up.
        end: History end.
        interval: Bar interval.
        max_iterations: Number of rebalance cycles before exiting. Values ``<= 0``
            run indefinitely until a shutdown request arrives.
        poll_seconds: Sleep between cycles.
        initial_capital: Starting equity.
        demo_config: Demo-mode settings governing the daily trade cap.
        risk_config: Risk settings including ``max_daily_trades``.
        notifier: Optional fill notifier; fires a toast + voice alert for every
            confirmed fill.
        flatten_on_exit: Close every open position when the loop stops because
            of a shutdown request (signal or halt file).
        halt_file: Sentinel file whose existence stops the loop cleanly.

    Returns:
        A dictionary with the router statistics and the final portfolio state.
    """
    data, _ = load_data(source, start, end, interval, use_cache=False)
    index = next(iter(data.values())).index
    regime_stream = build_default_regimes(data)
    # The regime stream starts after the feature warm-up, so align it onto the
    # full price calendar before indexing by bar position.
    regimes = CausalRegimeStreamer(
        config=cfg.DEFAULT_SETTINGS.hmm
    ).align_to(regime_stream, index)
    strategies = build_strategy_book()
    portfolio = Portfolio(
        initial_capital=initial_capital or cfg.DEFAULT_SETTINGS.sizing.initial_capital,
        risk_config=risk_config,
        demo_config=demo_config,
    )
    broker = build_broker(broker_mode, portfolio)
    router = OrderRouter(broker)
    sizing = SizingEngine()
    preservation = RiskPreservationStrategy(symbols=list(data.keys()))

    strategies_prepared = False
    stats: Dict[str, Any] = {}
    closes: Dict[str, float] = {}
    controller = LiveSessionController(halt_file=halt_file)

    if controller.halt_file is not None and controller.halt_file.exists():
        LOGGER.error(
            "Halt file %s exists at start-up. Remove it before launching: "
            "rm %s",
            controller.halt_file,
            controller.halt_file,
        )
        raise SystemExit(2)

    controller.install()
    try:
        broker.connect()
        account = broker.get_account()
        LOGGER.info("Broker %s connected (equity=%.2f)", broker.name, account.equity)
        if account.equity > 0:
            portfolio.cash = account.equity

        # ``max_iterations <= 0`` means "run until stopped" (signal or halt file).
        unbounded = max_iterations is None or int(max_iterations) <= 0
        if unbounded:
            LOGGER.info(
                "Running unbounded - stop with Ctrl+C or by creating %s.",
                controller.halt_file or "the halt file",
            )
        for cycle in itertools.count() if unbounded else range(int(max_iterations)):
            if controller.requested:
                LOGGER.warning(
                    "Shutdown requested (%s) - stopping before cycle %d.",
                    controller.reason,
                    cycle + 1,
                )
                break
            bar_index = len(index) - 1
            timestamp = pd.Timestamp(index[bar_index])
            if not strategies_prepared:
                for strategy in strategies:
                    strategy.prepare(data, index)
                preservation.prepare(data, index)
                strategies_prepared = True

            state = int(regimes.states.iloc[bar_index])
            probabilities = (
                regimes.probabilities.iloc[bar_index].to_numpy(dtype=float)
                if not regimes.probabilities.empty
                else np.zeros(cfg.N_REGIMES)
            )
            closes = {
                symbol: float(frame.iloc[bar_index]["close"])
                for symbol, frame in data.items()
            }
            # Adapters that fill against a reference price (the simulated broker,
            # paper-trading gateways) need the current marks before routing.
            price_feed = getattr(broker, "update_prices", None)
            if callable(price_feed):
                price_feed(closes)
            portfolio.update_prices(closes)

            context = StrategyContext(
                timestamp=timestamp,
                bar_index=bar_index,
                data=data,
                features={},
                regime_state=state,
                regime_probabilities=probabilities,
                positions={
                    symbol: portfolio.snapshot(symbol)  # type: ignore[dict-item]
                    for symbol, position in portfolio.positions.items()
                    if abs(position.quantity) > 1e-12
                },
                equity=portfolio.equity,
            )
            action = preservation.evaluate(context)

            raw_targets: Dict[str, float] = {}
            stop_multiples: Dict[str, Optional[float]] = {}
            for strategy in strategies:
                signals = (
                    strategy.generate_signals(context)
                    if strategy.is_active(state) and not action.halt_entries
                    else strategy.flat_signals(context)
                )
                for signal in signals:
                    raw_targets[signal.symbol] = float(signal.target_weight)
                    stop_multiples[signal.symbol] = signal.stop_atr_multiple

            if raw_targets and not action.halt_entries:
                sized = sizing.size(raw_targets, regime_state=state)
                prices = {
                    symbol: float(frame.iloc[bar_index]["close"])
                    for symbol, frame in data.items()
                }
                orders, reports = router.execute_targets(portfolio, sized, prices)
                if notifier is not None:
                    _attach_stops(
                        orders,
                        data=data,
                        bar_index=bar_index,
                        stop_multiples=stop_multiples,
                        regime_state=state,
                        risk_config=risk_config,
                    )
                    _announce_fills(notifier, orders, reports, state, portfolio)
                LOGGER.info(
                    "Cycle %d | regime=%d | orders=%d filled=%d",
                    cycle + 1,
                    state,
                    len(orders),
                    sum(1 for r in reports if r.status.value == "filled"),
                )
            else:
                LOGGER.info(
                    "Cycle %d | regime=%d | no orders (%s)",
                    cycle + 1,
                    state,
                    action.reason,
                )
            still_running = unbounded or cycle < int(max_iterations) - 1
            if still_running and not controller.requested:
                controller.sleep(poll_seconds)
    finally:
        if controller.requested and flatten_on_exit:
            open_now = [
                symbol
                for symbol, position in portfolio.positions.items()
                if abs(position.quantity) > 1e-12
            ]
            if open_now:
                LOGGER.warning(
                    "Shutdown requested (%s) - flattening %d position(s): %s",
                    controller.reason,
                    len(open_now),
                    open_now,
                )
                try:
                    flatten_reports = router.flatten(portfolio, closes)
                    stats["flatten"] = {
                        "requested": open_now,
                        "statuses": [r.status.value for r in flatten_reports],
                    }
                    LOGGER.warning(
                        "Flatten result: %s",
                        [r.status.value for r in flatten_reports],
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.error("Emergency flatten FAILED: %s", exc)
                    stats["flatten_error"] = str(exc)
            else:
                LOGGER.info("Shutdown requested (%s) - no open positions.", controller.reason)
        controller.restore()
        broker.disconnect()
        stats["shutdown_reason"] = controller.reason or "completed"

    stats = router.stats.as_dict()
    stats["final_equity"] = portfolio.equity
    stats["open_positions"] = len(
        [p for p in portfolio.positions.values() if abs(p.quantity) > 1e-12]
    )
    stats["demo_mode"] = portfolio.demo_mode
    stats["daily_trade_limit"] = portfolio.daily_trade_limit
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="quant_system",
        description="HMM Regime Switchboard quantitative trading & backtesting framework.",
    )
    parser.add_argument(
        "--mode",
        choices=("backtest", "walk-forward", "both", "live", "verify"),
        default="backtest",
        help=(
            "Pipeline to execute (default: backtest). 'verify' runs the "
            "end-to-end system verification suite."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("auto", "yfinance", "synthetic"),
        default=cfg.DEFAULT_SETTINGS.backtest.data_source,
        help="Market data source (default: auto -> yfinance with synthetic fallback).",
    )
    parser.add_argument("--start", default=cfg.DEFAULT_SETTINGS.backtest.start)
    parser.add_argument("--end", default=cfg.DEFAULT_SETTINGS.backtest.end)
    parser.add_argument("--interval", default=cfg.DEFAULT_SETTINGS.backtest.interval)
    parser.add_argument("--capital", type=float, default=None, help="Initial capital.")
    parser.add_argument(
        "--broker",
        choices=("simulated", "mt5", "fix"),
        default="simulated",
        help="Broker adapter used by --mode live.",
    )
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=252,
        help="Indicator warm-up bars prepended to every walk-forward window.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help=(
            "Rebalance cycles for --mode live (safety bound). "
            "Use 0 to run until stopped (Ctrl+C or the halt file)."
        ),
    )
    parser.add_argument(
        "--poll-seconds", type=float, default=5.0, help="Sleep between live cycles."
    )
    parser.add_argument(
        "--wfo-grid",
        choices=("full", "fast"),
        default="full",
        help=(
            "'full' sweeps the specification grid (11 Z-thresholds x 4 momentum "
            "lookbacks = 44 combinations per segment); 'fast' sweeps a 5 x 2 "
            "subset for quick iteration."
        ),
    )
    parser.add_argument(
        "--coint-method",
        choices=("coint", "adf_spread", "either"),
        default=cfg.DEFAULT_SETTINGS.stat_arb.coint_method,
        help=(
            "Stationarity gate for the stat-arb book. 'coint' is the literal "
            "Engle-Granger test on the price series; 'adf_spread' ADF-tests the "
            "traded rolling-beta spread; 'either' passes when either rejects."
        ),
    )
    parser.add_argument(
        "--demo",
        dest="demo",
        action="store_true",
        default=None,
        help="Force unlimited demo mode (lifts the daily trade-frequency cap).",
    )
    parser.add_argument(
        "--no-demo",
        dest="demo",
        action="store_false",
        help="Force live risk limits (enforce --max-daily-trades).",
    )
    parser.add_argument(
        "--max-daily-trades",
        type=int,
        default=cfg.DEFAULT_SETTINGS.risk.max_daily_trades,
        help=(
            "Maximum new positions per calendar day in live mode "
            "(ignored in demo mode; 0 disables the cap)."
        ),
    )
    parser.add_argument(
        "--enable-voice",
        action="store_true",
        help="Speak every confirmed fill with pyttsx3 on a background thread.",
    )
    parser.add_argument(
        "--voice-mode",
        choices=("speak", "file", "off"),
        default=cfg.DEFAULT_SETTINGS.notifier.voice_mode,
        help=(
            "'speak' plays audio through the default device; 'file' writes WAV "
            "clips to reports/voice (headless-safe); 'off' disables synthesis."
        ),
    )
    parser.add_argument(
        "--toast-backend",
        choices=("auto", "pyqt", "tk", "none"),
        default=cfg.DEFAULT_SETTINGS.notifier.toast_backend,
        help="Desktop toast backend (default: auto -> PyQt6 -> tkinter -> none).",
    )
    parser.add_argument(
        "--no-toast", action="store_true", help="Disable desktop toasts entirely."
    )
    parser.add_argument(
        "--flatten-on-exit",
        action="store_true",
        help="Close every open position when the loop stops via Ctrl+C or the halt file.",
    )
    parser.add_argument(
        "--halt-file",
        default=str(cfg.HALT_FILE),
        help=(
            "Sentinel file whose existence stops the live loop cleanly "
            "(default: reports/HALT). Set to '' to disable."
        ),
    )
    parser.add_argument(
        "--no-file-log",
        action="store_true",
        help="Keep logging on the console only (skip reports/logs/quant_system.log).",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Bypass the on-disk data cache."
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip equity-curve rendering.")
    parser.add_argument("--out", default=None, help="Report directory override.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser.parse_args(argv)


def _run_verification(report_dir: Path) -> int:
    """Execute the end-to-end verification suite and persist its report.

    Args:
        report_dir: Directory that receives ``verification_report.txt``.

    Returns:
        ``0`` when every check passes, ``1`` otherwise.
    """
    # tests/ is not a package (it holds the pytest modules), so load the file
    # directly rather than importing it as ``quant_system.tests.verify_system``.
    import importlib.util

    verify_path = Path(__file__).resolve().parent / "tests" / "verify_system.py"
    spec = importlib.util.spec_from_file_location("verify_system", verify_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        LOGGER.error("Could not locate tests/verify_system.py at %s", verify_path)
        return 1
    module = importlib.util.module_from_spec(spec)
    # ``dataclasses`` resolves ``cls.__module__`` through ``sys.modules``, so the
    # module must be registered *before* its body executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    SystemVerifier = module.SystemVerifier

    LOGGER.info("Running end-to-end system verification ...")
    result = SystemVerifier().run()
    text = result.render()
    print(text)
    _write_text(report_dir / "verification_report.txt", text)
    LOGGER.info("Verification report written to %s", report_dir / "verification_report.txt")
    return 0 if result.passed else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entrypoint.

    Args:
        argv: Argument vector.

    Returns:
        Process exit code (``0`` on success).
    """
    args = parse_args(argv)
    report_dir = _ensure_reports_dir(Path(args.out) if args.out else None)
    log_path = configure_logging(
        args.verbose,
        log_dir=report_dir / "logs",
        enable_file_log=not args.no_file_log,
    )

    # --- Demo-mode / notifier configuration ---------------------------- #
    demo_enabled = (
        bool(cfg.DEFAULT_SETTINGS.demo.enabled) if args.demo is None else bool(args.demo)
    )
    demo_config = replace(cfg.DEFAULT_SETTINGS.demo, enabled=demo_enabled)
    risk_config = replace(
        cfg.DEFAULT_SETTINGS.risk, max_daily_trades=int(args.max_daily_trades)
    )
    notifier_config = cfg.DEFAULT_SETTINGS.notifier
    if args.enable_voice:
        notifier_config = notifier_config.with_voice(True, args.voice_mode)
    if args.no_toast:
        notifier_config = replace(notifier_config, toast_enabled=False)
    if args.toast_backend != cfg.DEFAULT_SETTINGS.notifier.toast_backend:
        notifier_config = replace(notifier_config, toast_backend=args.toast_backend)

    if log_path is not None:
        LOGGER.info("Logging to %s", log_path)
    if args.mode == "verify":
        return _run_verification(report_dir)

    LOGGER.info(
        "quant_system v%s | mode=%s source=%s %s..%s",
        cfg.__version__ if hasattr(cfg, "__version__") else "1.0.0",
        args.mode,
        args.source,
        args.start,
        args.end,
    )

    if args.mode == "live":
        if args.broker == "mt5" and sys.platform != "win32":
            LOGGER.error(
                "The MetaTrader 5 adapter requires Windows and a running MT5 "
                "terminal (host is %s). Run 'python scripts/preflight.py' for the "
                "full environment report, or use --broker simulated / --broker fix.",
                sys.platform,
            )
            if log_path is not None:
                LOGGER.error("See %s for details.", log_path)
            return 2
        LOGGER.info(
            "Trade governance: mode=%s daily_trade_limit=%s",
            "DEMO (unlimited)" if demo_config.enabled else "LIVE",
            "unlimited"
            if demo_config.enabled and demo_config.unlimited_trades
            else risk_config.max_daily_trades,
        )
        try:
            with NotifierEngine(config=notifier_config) as notifier:
                if notifier_config.enabled:
                    LOGGER.info(
                        "Notifier: toasts=%s (%s) voice=%s (%s)",
                        notifier_config.toast_enabled,
                        notifier.backend_note,
                        notifier_config.voice_enabled,
                        notifier_config.voice_mode,
                    )
                stats = run_live(
                    source=args.source,
                    broker_mode=args.broker,
                    start=args.start,
                    end=args.end,
                    interval=args.interval,
                    max_iterations=args.max_iterations,
                    poll_seconds=args.poll_seconds,
                    initial_capital=args.capital,
                    demo_config=demo_config,
                    risk_config=risk_config,
                    notifier=notifier if notifier_config.enabled else None,
                    flatten_on_exit=args.flatten_on_exit,
                    halt_file=Path(args.halt_file) if args.halt_file else None,
                )
            # Read the counters *after* stop() so the drained voice/toast totals
            # are final rather than mid-flight.
            if notifier_config.enabled:
                stats["notifier"] = notifier.stats()
            LOGGER.info("Live session finished: %s", json.dumps(stats, indent=2, default=str))
            return 0
        except MT5UnavailableError as exc:
            LOGGER.error(
                "MetaTrader 5 is not reachable: %s\n"
                "  Check: terminal running, logged in, Algo Trading enabled (Ctrl+E),\n"
                "         symbols present in Market Watch, mt5_symbol set in config/settings.py.",
                exc,
            )
            return 2

    data, ingestion = load_data(
        source=args.source,
        start=args.start,
        end=args.end,
        interval=args.interval,
        use_cache=not args.no_cache,
    )
    _write_text(
        report_dir / "data_sources.json",
        json.dumps(ingestion.sources_used, indent=2, default=str),
    )

    result, regime_result = run_backtest(
        data,
        initial_capital=args.capital,
        coint_method=args.coint_method,
        verbose=args.verbose,
    )
    LOGGER.info("\n%s", ametrics.format_summary(result.metrics))
    output = report_backtest(
        result, regime_result, report_dir, make_plot=not args.no_plot
    )

    if args.mode in ("walk-forward", "both"):
        wf_result = run_walk_forward(
            data=data,
            regime_states=result.regime_states,
            regime_probabilities=result.regime_probabilities
            if not result.regime_probabilities.empty
            else None,
            warmup_bars=args.warmup_bars,
            initial_capital=args.capital,
            coint_method=args.coint_method,
            grid=args.wfo_grid,
            verbose=True,
        )
        LOGGER.info("\n%s", wf_result.report())
        _write_text(report_dir / "walk_forward_report.txt", wf_result.report())
        wf_result.oos_equity.to_frame().to_csv(report_dir / "walk_forward_oos_equity.csv")
        wf_result.degradation_summary.to_csv(
            report_dir / "walk_forward_degradation.csv", index=False
        )
        wf_result.parameter_history.to_csv(
            report_dir / "walk_forward_parameters.csv", index=False
        )
        if not args.no_plot and not wf_result.oos_equity.empty:
            plot_equity(
                wf_result.oos_equity,
                result.regime_states,
                cfg.FIGURES_DIR / "walk_forward_oos_equity.png",
                title="Walk-forward stitched out-of-sample equity",
            )

    LOGGER.info("Artifacts written to %s", output.report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
