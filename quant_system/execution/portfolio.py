"""Position tracking, PnL accounting and advanced position sizing.

Accounting model
----------------
Positions are held in *units* of the base asset and converted to notional with
``AssetSpec.contract_size``.  Average-cost accounting is used: increasing a
position re-weights the average price; reducing it realises
``(fill - average) * closed_units * contract_size``, and a flip closes the old leg
and opens a new one in a single fill.

Sizing model
------------
:class:`SizingEngine` composes four mechanisms, in this order:

1. **Regime scaling** - :class:`~quant_system.config.settings.RiskConfig.regime_exposure`
   (State 2 contributes a zero multiplier, which is what actually flattens the
   book together with the preservation overlay).
2. **Volatility targeting** - the whole book is scaled so that the ex-ante
   portfolio volatility ``sqrt(w' Sigma w)`` matches ``target_volatility``.
3. **Kelly sizing** - the Kelly optimum (from realised trade statistics or from
   the return stream) caps the *aggregate* gross exposure.  A fractional Kelly
   (default half) is used because the inputs are estimates.
4. **Risk parity / ERC** - weights are re-distributed across the active
   instruments so each contributes an equal share of portfolio variance, then
   blended with the raw signal weights by ``risk_parity_blend``.

Finally the weights are clipped per-symbol and in aggregate
(``max_symbol_weight`` / ``max_gross_leverage``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quant_system.config import settings as cfg

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Fill:
    """An executed (or simulated) fill.

    Attributes:
        timestamp: Execution timestamp.
        symbol: Instrument symbol.
        quantity: Signed units filled (negative = sell).
        price: Fill price before costs.
        spread_cost: Half-spread cost per unit, in price terms.
        slippage_cost: Slippage per unit, in price terms.
        commission: Total commission charged (quote currency).
        strategy: Strategy that requested the fill.
        tag: Reason code.
    """

    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission: float = 0.0
    strategy: str = ""
    tag: str = ""

    @property
    def notional(self) -> float:
        """Gross traded notional in quote currency."""
        return abs(self.quantity) * self.price * self._contract_size()

    def _contract_size(self) -> float:
        """Contract size for the fill's symbol (1.0 for unknown symbols)."""
        try:
            return cfg.DEFAULT_SETTINGS.universe.spec(self.symbol).contract_size
        except KeyError:
            return 1.0

    @property
    def total_cost(self) -> float:
        """All-in execution cost in quote currency."""
        return (
            abs(self.quantity) * (self.spread_cost + self.slippage_cost) * self._contract_size()
            + self.commission
        )


@dataclass
class Trade:
    """A completed round trip (or partial close) used for expectancy stats.

    Attributes:
        symbol: Instrument symbol.
        strategy: Strategy name.
        entry_timestamp: Opening timestamp.
        exit_timestamp: Closing timestamp.
        direction: ``+1`` long, ``-1`` short.
        entry_price: Average entry price.
        exit_price: Exit price.
        quantity: Units closed.
        gross_pnl: PnL before costs.
        costs: All-in costs attributed to the trade.
        net_pnl: ``gross_pnl - costs``.
        return_pct: Net PnL divided by the entry notional.
        exit_reason: Tag explaining the exit.
    """

    symbol: str
    strategy: str
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
    exit_reason: str = ""


@dataclass
class Position:
    """An open position with average-cost accounting.

    Attributes:
        symbol: Instrument symbol.
        quantity: Signed units held.
        average_price: Average entry price.
        last_price: Latest mark.
        realized_pnl: Cumulative realised PnL (quote currency).
        strategy: Strategy that opened the position.
        opened_at: Opening timestamp.
        stop_price: Armed stop price (``None`` when unarmed).
        stop_atr_multiple: ATR multiple the stop is based on.
        metadata: Free-form diagnostics (z-score, regime, ...).
    """

    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    last_price: float = 0.0
    realized_pnl: float = 0.0
    strategy: str = ""
    opened_at: Optional[pd.Timestamp] = None
    stop_price: Optional[float] = None
    stop_atr_multiple: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def contract_size(self) -> float:
        """Notional multiplier for the instrument."""
        try:
            return cfg.DEFAULT_SETTINGS.universe.spec(self.symbol).contract_size
        except KeyError:
            return 1.0

    @property
    def market_value(self) -> float:
        """Signed mark-to-market value in quote currency."""
        return self.quantity * self.last_price * self.contract_size

    @property
    def unrealized_pnl(self) -> float:
        """Unrealised PnL in quote currency."""
        if abs(self.quantity) < 1e-12:
            return 0.0
        return (self.last_price - self.average_price) * self.quantity * self.contract_size

    @property
    def notional(self) -> float:
        """Absolute exposure in quote currency."""
        return abs(self.market_value)

    @property
    def direction(self) -> int:
        """``+1`` long, ``-1`` short, ``0`` flat."""
        if self.quantity > 1e-12:
            return 1
        if self.quantity < -1e-12:
            return -1
        return 0

    def mark(self, price: float) -> None:
        """Update the mark price.

        Args:
            price: New mark price (must be positive).
        """
        if price > 0:
            self.last_price = float(price)

    def apply_fill(
        self,
        quantity: float,
        price: float,
        timestamp: Optional[pd.Timestamp] = None,
        strategy: str = "",
    ) -> Tuple[float, float]:
        """Apply a fill and return the realised PnL it generated.

        Args:
            quantity: Signed units to trade (negative = sell).
            price: Fill price.
            timestamp: Fill timestamp (used when opening a new position).
            strategy: Strategy name recorded when the position is opened.

        Returns:
            Tuple ``(realized_pnl, closed_units)`` for this fill.
        """
        contract_size = self.contract_size
        current = self.quantity
        delta = float(quantity)
        if abs(delta) < 1e-15:
            return 0.0, 0.0

        realized = 0.0
        closed_units = 0.0

        same_direction = (current == 0.0) or (np.sign(current) == np.sign(delta))
        if same_direction:
            new_qty = current + delta
            total_abs = abs(current) + abs(delta)
            self.average_price = (
                (self.average_price * abs(current) + price * abs(delta)) / total_abs
                if total_abs > 0
                else price
            )
            self.quantity = new_qty
            if abs(current) < 1e-12:
                self.opened_at = timestamp
                self.strategy = strategy
        else:
            closed_units = min(abs(delta), abs(current))
            realized = (price - self.average_price) * np.sign(current) * closed_units * contract_size
            remaining = current + delta
            if abs(remaining) < 1e-12:
                self.quantity = 0.0
                self.average_price = 0.0
                self.stop_price = None
                self.opened_at = None
            elif np.sign(remaining) == np.sign(current):
                self.quantity = remaining
            else:
                # Position flipped: close the old leg, open a new one at `price`.
                self.quantity = remaining
                self.average_price = price
                self.opened_at = timestamp
                self.strategy = strategy
        self.realized_pnl += realized
        self.last_price = price if price > 0 else self.last_price
        return float(realized), float(closed_units)


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
class Portfolio:
    """Tracks cash, positions, PnL and the equity curve.

    Attributes:
        initial_capital: Starting equity.
        cash: Cash balance (quote currency).
        positions: Mapping of symbol -> :class:`Position`.
        equity_curve: Recorded equity indexed by timestamp.
        trades: Completed round trips.
        fills: Every individual fill.
    """

    def __init__(
        self,
        initial_capital: float = cfg.DEFAULT_SETTINGS.sizing.initial_capital,
        sizing_config: Optional[cfg.SizingConfig] = None,
        risk_config: Optional[cfg.RiskConfig] = None,
        demo_config: Optional[cfg.DemoConfig] = None,
    ) -> None:
        """Initialise the portfolio.

        Args:
            initial_capital: Starting equity in quote currency.
            sizing_config: Sizing configuration; defaults to ``settings.sizing``.
            risk_config: Risk configuration; defaults to ``settings.risk``.
            demo_config: Demo-mode configuration; defaults to ``settings.demo``.
        """
        self.config: cfg.SizingConfig = sizing_config or cfg.DEFAULT_SETTINGS.sizing
        self.risk_config: cfg.RiskConfig = risk_config or cfg.DEFAULT_SETTINGS.risk
        self.demo_config: cfg.DemoConfig = demo_config or cfg.DEFAULT_SETTINGS.demo
        self.initial_capital: float = float(initial_capital)
        self.cash: float = float(initial_capital)
        self.positions: Dict[str, Position] = {}
        self.fills: List[Fill] = []
        self.trades: List[Trade] = []
        self._equity_curve: Dict[pd.Timestamp, float] = {}
        self._pending_costs: Dict[str, float] = {}
        self._peak_equity: float = float(initial_capital)
        # New positions opened per calendar day (the demo-mode throttled metric).
        self._daily_open_counts: Dict[pd.Timestamp, int] = {}

    # ------------------------------------------------------------------ #
    # Valuation
    # ------------------------------------------------------------------ #
    @property
    def equity(self) -> float:
        """Current account equity (cash + mark-to-market)."""
        return self.cash + sum(position.market_value for position in self.positions.values())

    @property
    def peak_equity(self) -> float:
        """Highest equity observed so far."""
        return self._peak_equity

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak, as a positive fraction."""
        equity = self.equity
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - equity / self._peak_equity)

    @property
    def equity_curve(self) -> pd.Series:
        """Recorded equity curve as a Series indexed by timestamp."""
        if not self._equity_curve:
            return pd.Series(dtype=float, name="equity")
        series = pd.Series(self._equity_curve, name="equity").sort_index()
        return series

    def returns(self) -> pd.Series:
        """Simple period returns of the equity curve."""
        curve = self.equity_curve
        if curve.empty:
            return pd.Series(dtype=float, name="return")
        return curve.pct_change().dropna().rename("return")

    def update_prices(
        self, prices: Mapping[str, float], timestamp: Optional[pd.Timestamp] = None
    ) -> None:
        """Mark every position to the supplied prices.

        Args:
            prices: Mapping of symbol -> mark price.
            timestamp: Optional timestamp at which to record equity.
        """
        for symbol, position in self.positions.items():
            price = prices.get(symbol)
            if price is not None and price > 0:
                position.mark(float(price))
        if timestamp is not None:
            self.record_equity(timestamp)

    def record_equity(self, timestamp: pd.Timestamp) -> None:
        """Append the current equity to the curve.

        Args:
            timestamp: Bar timestamp.
        """
        equity = self.equity
        self._equity_curve[pd.Timestamp(timestamp)] = equity
        self._peak_equity = max(self._peak_equity, equity)

    # ------------------------------------------------------------------ #
    # Exposure
    # ------------------------------------------------------------------ #
    def gross_exposure(self) -> float:
        """Sum of absolute position notional, in quote currency."""
        return float(sum(position.notional for position in self.positions.values()))

    def net_exposure(self) -> float:
        """Signed position notional, in quote currency."""
        return float(sum(position.market_value for position in self.positions.values()))

    def leverage(self) -> float:
        """Gross exposure divided by equity."""
        equity = self.equity
        return self.gross_exposure() / equity if equity > 0 else 0.0

    def weights(self) -> Dict[str, float]:
        """Signed exposure of each position as a fraction of equity."""
        equity = self.equity
        if equity <= 0:
            return {}
        return {
            symbol: position.market_value / equity
            for symbol, position in self.positions.items()
            if abs(position.quantity) > 1e-12
        }

    # ------------------------------------------------------------------ #
    # Order sizing & fills
    # ------------------------------------------------------------------ #
    def position(self, symbol: str) -> Position:
        """Return (creating if needed) the position for ``symbol``.

        Args:
            symbol: Instrument symbol.

        Returns:
            The :class:`Position` object.
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def target_quantity(self, symbol: str, target_weight: float, price: float) -> float:
        """Convert a target weight into an order quantity delta.

        Args:
            symbol: Instrument symbol.
            target_weight: Signed target exposure as a fraction of equity.
            price: Reference price used for the conversion.

        Returns:
            Signed units to buy (positive) or sell (negative).

        Raises:
            ValueError: If ``price`` is non-positive.
        """
        if price <= 0:
            raise ValueError("Target quantity requires a positive price.")
        contract_size = self.position(symbol).contract_size
        target_notional = target_weight * self.equity
        target_units = target_notional / (price * contract_size)
        return float(target_units - self.position(symbol).quantity)

    def apply_fill(self, fill: Fill) -> Tuple[float, float]:
        """Apply a fill to cash, positions and the trade log.

        Args:
            fill: The fill to apply.

        Returns:
            Tuple ``(realized_pnl, closed_units)``.
        """
        position = self.position(fill.symbol)
        entry_price = position.average_price if abs(position.quantity) > 1e-12 else fill.price
        entry_time = position.opened_at
        direction_before = position.direction
        had_position = abs(position.quantity) > 1e-12

        realized, closed_units = position.apply_fill(
            fill.quantity, fill.price, timestamp=fill.timestamp, strategy=fill.strategy
        )
        if not had_position and abs(position.quantity) > 1e-12:
            self._register_open(fill.timestamp)

        contract_size = position.contract_size
        cash_delta = -fill.quantity * fill.price * contract_size
        cost = fill.total_cost
        self.cash += cash_delta - cost
        self.fills.append(fill)

        if closed_units > 1e-12 and direction_before != 0:
            gross = realized
            # Attribute costs proportionally to the closed fraction.
            attributed_cost = cost
            net = gross - attributed_cost
            entry_notional = abs(entry_price * closed_units * contract_size)
            self.trades.append(
                Trade(
                    symbol=fill.symbol,
                    strategy=fill.strategy or position.strategy,
                    entry_timestamp=entry_time or fill.timestamp,
                    exit_timestamp=fill.timestamp,
                    direction=direction_before,
                    entry_price=float(entry_price),
                    exit_price=float(fill.price),
                    quantity=float(closed_units),
                    gross_pnl=float(gross),
                    costs=float(attributed_cost),
                    net_pnl=float(net),
                    return_pct=float(net / entry_notional) if entry_notional > 0 else 0.0,
                    exit_reason=fill.tag,
                )
            )
        elif fill.strategy and abs(position.quantity) > 1e-12 and position.strategy == "":
            position.strategy = fill.strategy
        return realized, closed_units

    def accrue_financing(self, annual_rate: float, bars_per_year: float) -> float:
        """Charge financing on short notional and credit/debit cash.

        Args:
            annual_rate: Annualised financing rate.
            bars_per_year: Number of bars per year for the interval.

        Returns:
            The financing charge applied (positive = cost).
        """
        if annual_rate <= 0 or bars_per_year <= 0:
            return 0.0
        short_notional = sum(
            position.notional for position in self.positions.values() if position.quantity < 0
        )
        charge = short_notional * annual_rate / bars_per_year
        if charge > 0:
            self.cash -= charge
        return float(charge)

    # ------------------------------------------------------------------ #
    # Trade-frequency governance (demo mode)
    # ------------------------------------------------------------------ #
    @property
    def demo_mode(self) -> bool:
        """Whether demo (unlimited paper-trading) mode is enabled."""
        return bool(self.demo_config.enabled and cfg.DEMO_MODE)

    @property
    def unlimited_demo_trades(self) -> bool:
        """Whether the daily trade-frequency cap is lifted."""
        return bool(
            self.demo_mode
            and self.demo_config.unlimited_trades
            and cfg.UNLIMITED_DEMO_TRADES
        )

    @property
    def daily_trade_limit(self) -> int:
        """Maximum new positions per calendar day (``-1`` = unlimited)."""
        return -1 if self.unlimited_demo_trades else int(self.risk_config.max_daily_trades)

    def trades_opened_on(self, timestamp: pd.Timestamp) -> int:
        """Number of new positions opened on the day of ``timestamp``.

        Args:
            timestamp: Any timestamp within the day of interest.

        Returns:
            The count of position openings for that calendar day.
        """
        return int(self._daily_open_counts.get(pd.Timestamp(timestamp).normalize(), 0))

    def can_open_trade(
        self, timestamp: Optional[pd.Timestamp] = None
    ) -> Tuple[bool, str]:
        """Whether a *new* position may be opened right now.

        The throttle counts position **openings** (flat -> non-flat), not
        individual fills, so adding to an existing position or closing one is
        never blocked.  In demo mode with
        :attr:`~quant_system.config.settings.DemoConfig.unlimited_trades` the cap
        is lifted entirely.

        Args:
            timestamp: Timestamp to evaluate; defaults to "now" (UTC).

        Returns:
            Tuple ``(allowed, reason)`` where ``reason`` is a short code suitable
            for the audit log.
        """
        if self.unlimited_demo_trades:
            return True, "demo_unlimited"
        limit = int(self.risk_config.max_daily_trades)
        if limit <= 0:
            return True, "no_limit"
        stamp = (
            pd.Timestamp(timestamp).normalize()
            if timestamp is not None
            else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        )
        used = self._daily_open_counts.get(stamp, 0)
        if used >= limit:
            return False, f"daily_limit_{used}/{limit}"
        return True, f"ok_{used}/{limit}"

    def _register_open(self, timestamp: pd.Timestamp) -> None:
        """Count a new position opening against the current day.

        Args:
            timestamp: Fill timestamp.
        """
        if timestamp is None:
            return
        day = pd.Timestamp(timestamp).normalize()
        self._daily_open_counts[day] = self._daily_open_counts.get(day, 0) + 1

    @property
    def daily_open_counts(self) -> Dict[pd.Timestamp, int]:
        """Read-only view of the per-day position-opening counters."""
        return dict(self._daily_open_counts)

    # ------------------------------------------------------------------ #
    # Reporting helpers
    # ------------------------------------------------------------------ #
    def trade_frame(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame.

        Returns:
            DataFrame of :class:`Trade` records (empty frame when no trades).
        """
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "strategy",
                    "entry_timestamp",
                    "exit_timestamp",
                    "direction",
                    "entry_price",
                    "exit_price",
                    "quantity",
                    "gross_pnl",
                    "costs",
                    "net_pnl",
                    "return_pct",
                    "exit_reason",
                ]
            )
        frame = pd.DataFrame([trade.__dict__ for trade in self.trades])
        frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"])
        frame["exit_timestamp"] = pd.to_datetime(frame["exit_timestamp"])
        return frame.sort_values("exit_timestamp").reset_index(drop=True)

    def fill_frame(self) -> pd.DataFrame:
        """Return the fill log as a DataFrame.

        Returns:
            DataFrame of :class:`Fill` records.
        """
        if not self.fills:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "quantity",
                    "price",
                    "spread_cost",
                    "slippage_cost",
                    "commission",
                    "strategy",
                    "tag",
                ]
            )
        frame = pd.DataFrame([fill.__dict__ for fill in self.fills])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame

    def snapshot(self, symbol: str) -> object:
        """Build an immutable snapshot for strategy consumption.

        Args:
            symbol: Instrument symbol.

        Returns:
            A :class:`~quant_system.strategies.base.PositionSnapshot`.
        """
        from quant_system.strategies.base import PositionSnapshot  # local: avoids a cycle

        position = self.positions.get(symbol)
        if position is None or abs(position.quantity) < 1e-12:
            return PositionSnapshot(
                symbol=symbol,
                quantity=0.0,
                average_price=0.0,
                last_price=0.0,
                unrealized_pnl=0.0,
            )
        return PositionSnapshot(
            symbol=symbol,
            quantity=position.quantity,
            average_price=position.average_price,
            last_price=position.last_price,
            unrealized_pnl=position.unrealized_pnl,
            strategy=position.strategy,
            stop_price=position.stop_price,
        )

    def reset(self) -> None:
        """Return the portfolio to its initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.fills = []
        self.trades = []
        self._equity_curve = {}
        self._peak_equity = self.initial_capital
        self._daily_open_counts = {}


# --------------------------------------------------------------------------- #
# Kelly criterion
# --------------------------------------------------------------------------- #
def kelly_fraction(win_rate: float, average_win: float, average_loss: float) -> float:
    """Kelly-optimal fraction of capital for a discrete win/loss bet.

    Implements ``f* = (p * b - q) / b`` with ``b = average_win / average_loss``,
    ``p = win_rate`` and ``q = 1 - p``.

    Args:
        win_rate: Probability of a winning trade, in ``[0, 1]``.
        average_win: Mean PnL of winning trades (positive).
        average_loss: Mean loss of losing trades, expressed as a positive number.

    Returns:
        The Kelly fraction, clipped to ``[0, 1]``. Returns ``0.0`` for a
        non-positive expectancy (never bet on a losing system).
    """
    if not np.isfinite(win_rate) or win_rate < 0.0 or win_rate > 1.0:
        return 0.0
    if average_loss <= 0 or average_win <= 0:
        return 0.0
    b = average_win / average_loss
    q = 1.0 - win_rate
    f_star = (win_rate * b - q) / b
    return float(np.clip(f_star, 0.0, 1.0))


def kelly_fraction_from_trades(trades: pd.DataFrame) -> float:
    """Estimate the Kelly fraction from a trade log.

    Args:
        trades: DataFrame with a ``net_pnl`` column (see
            :meth:`Portfolio.trade_frame`).

    Returns:
        The Kelly fraction in ``[0, 1]``; ``0.0`` when there is insufficient or
        degenerate data.
    """
    if trades is None or trades.empty or "net_pnl" not in trades.columns:
        return 0.0
    wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"]
    losses = trades.loc[trades["net_pnl"] < 0, "net_pnl"]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    win_rate = len(wins) / len(trades)
    return kelly_fraction(win_rate, float(wins.mean()), float(-losses.mean()))


def kelly_fraction_from_returns(returns: pd.Series) -> float:
    """Continuous Kelly fraction ``f* = mu / sigma^2`` for a return stream.

    Args:
        returns: Periodic returns of the *sized* strategy.

    Returns:
        The Kelly fraction (may exceed 1; callers should cap it).
    """
    if returns is None or len(returns) < 2:
        return 0.0
    series = pd.Series(returns).dropna()
    if len(series) < 2:
        return 0.0
    variance = float(series.var(ddof=1))
    if variance <= 1e-18:
        return 0.0
    return float(series.mean() / variance)


# --------------------------------------------------------------------------- #
# Risk parity / ERC
# --------------------------------------------------------------------------- #
def inverse_volatility_weights(covariance: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Inverse-volatility (naive risk parity) weights.

    Args:
        covariance: Covariance matrix.

    Returns:
        Weight vector summing to 1. Falls back to equal weights when any
        variance is non-positive.
    """
    cov = np.asarray(covariance, dtype=float)
    n = cov.shape[0]
    if n == 0:
        return np.zeros(0)
    variances = np.diag(cov)
    if np.any(variances <= 0) or not np.all(np.isfinite(variances)):
        return np.full(n, 1.0 / n)
    inverse_vol = 1.0 / np.sqrt(variances)
    total = inverse_vol.sum()
    return inverse_vol / total if total > 0 else np.full(n, 1.0 / n)


def risk_parity_weights(
    covariance: pd.DataFrame | np.ndarray,
    budget: Optional[Sequence[float]] = None,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> np.ndarray:
    """Equal-risk-contribution (true risk parity) weights.

    Solves ``C y = b / y`` element-wise for ``y > 0`` using the cyclical
    coordinate-descent scheme of Spinu / Griveau-Billion et al., then normalises
    ``w = y / sum(y)``.  Each asset's risk contribution
    ``w_i * (C w)_i / (w' C w)`` then equals its budget ``b_i``.

    Args:
        covariance: Positive semi-definite covariance matrix.
        budget: Target risk contributions summing to 1; defaults to equal risk.
        max_iter: Maximum coordinate-descent sweeps.
        tol: Convergence tolerance on the maximum absolute step.

    Returns:
        Non-negative weight vector summing to 1.

    Raises:
        ValueError: If ``covariance`` is not square.
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    n = cov.shape[0]
    if n == 0:
        return np.zeros(0)
    if budget is None:
        b = np.full(n, 1.0 / n)
    else:
        b = np.asarray(budget, dtype=float)
        total = b.sum()
        b = b / total if total > 0 else np.full(n, 1.0 / n)

    if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) <= 0):
        return inverse_volatility_weights(cov)

    # Symmetrise and nudge the diagonal to guarantee positive definiteness.
    cov = (cov + cov.T) / 2.0
    diag = np.diag(cov).copy()
    if np.any(diag <= 0):
        return inverse_volatility_weights(cov)
    y = 1.0 / np.sqrt(diag)
    y = y / max(y.sum(), 1e-18)

    for _ in range(max_iter):
        y_old = y.copy()
        for i in range(n):
            others = cov[i, :] @ y - cov[i, i] * y[i]
            discriminant = others**2 + 4.0 * cov[i, i] * b[i]
            y[i] = (-others + math.sqrt(max(discriminant, 0.0))) / (2.0 * cov[i, i])
        if np.max(np.abs(y - y_old)) < tol:
            break
    total = y.sum()
    if total <= 0 or not np.all(np.isfinite(y)):
        return inverse_volatility_weights(cov)
    weights = y / total
    return np.clip(weights, 0.0, None) / np.clip(weights, 0.0, None).sum()


def risk_contributions(weights: np.ndarray, covariance: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Compute each asset's fractional contribution to portfolio variance.

    Args:
        weights: Portfolio weights.
        covariance: Covariance matrix.

    Returns:
        Vector of risk contributions summing to 1.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    variance = float(w @ cov @ w)
    if variance <= 1e-18:
        return np.zeros_like(w)
    marginal = cov @ w
    return (w * marginal) / variance


# --------------------------------------------------------------------------- #
# Composed sizing engine
# --------------------------------------------------------------------------- #
class SizingEngine:
    """Turns raw strategy weights into risk-managed, funded target weights.

    Attributes:
        config: Sizing configuration.
        risk_config: Risk configuration (regime exposure scalars).
    """

    def __init__(
        self,
        config: Optional[cfg.SizingConfig] = None,
        risk_config: Optional[cfg.RiskConfig] = None,
    ) -> None:
        """Initialise the engine.

        Args:
            config: Sizing configuration.
            risk_config: Risk configuration.
        """
        self.config: cfg.SizingConfig = config or cfg.DEFAULT_SETTINGS.sizing
        self.risk_config: cfg.RiskConfig = risk_config or cfg.DEFAULT_SETTINGS.risk
        self.last_diagnostics: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def size(
        self,
        target_weights: Mapping[str, float],
        covariance: Optional[pd.DataFrame] = None,
        regime_state: int = cfg.STATE_RANGE_BOUND,
        kelly: Optional[float] = None,
    ) -> Dict[str, float]:
        """Apply the full sizing pipeline.

        Args:
            target_weights: Raw signed target weights from the strategies.
            covariance: Annualised covariance matrix of instrument returns. When
                supplied it drives volatility targeting and risk parity.
            regime_state: Active regime id (selects the exposure scalar).
            kelly: Pre-computed Kelly fraction; when omitted it is derived from
                ``kelly`` diagnostics set previously (or ignored).

        Returns:
            Mapping of symbol -> final signed target weight.
        """
        weights = {
            symbol: float(weight)
            for symbol, weight in target_weights.items()
            if np.isfinite(weight) and abs(weight) > 1e-12
        }
        diagnostics: Dict[str, float] = {"raw_gross": self._gross(weights)}
        if not weights:
            self.last_diagnostics = diagnostics
            return {}

        # 1. Regime exposure scalar -------------------------------------- #
        scalar = float(self.risk_config.regime_exposure.get(int(regime_state), 1.0))
        weights = {symbol: weight * scalar for symbol, weight in weights.items()}
        diagnostics["regime_scalar"] = scalar

        # 2. Volatility targeting ---------------------------------------- #
        if covariance is not None and self.config.target_volatility > 0:
            weights, vol_scale = self._volatility_target(weights, covariance)
            diagnostics["vol_scale"] = vol_scale

        # 3. Kelly cap on aggregate gross -------------------------------- #
        if kelly is not None and np.isfinite(kelly) and kelly > 0:
            weights, kelly_scale = self._apply_kelly(weights, kelly)
            diagnostics["kelly_raw"] = float(kelly)
            diagnostics["kelly_scale"] = kelly_scale
            diagnostics["kelly_target_gross"] = float(
                min(self.config.kelly_fraction * kelly, self.config.kelly_cap)
            )

        # 4. Risk parity redistribution ---------------------------------- #
        if (
            covariance is not None
            and self.config.risk_parity_blend > 0
            and len(weights) > 1
        ):
            weights = self._risk_parity_blend(weights, covariance)

        # 5. Hard caps & dust removal ------------------------------------ #
        weights = self._apply_caps(weights)
        diagnostics["final_gross"] = self._gross(weights)
        diagnostics["final_leverage"] = diagnostics["final_gross"]
        self.last_diagnostics = diagnostics
        return weights

    # ------------------------------------------------------------------ #
    def _volatility_target(
        self, weights: Mapping[str, float], covariance: pd.DataFrame
    ) -> Tuple[Dict[str, float], float]:
        """Scale the book toward the target portfolio volatility.

        Args:
            weights: Current weights.
            covariance: Annualised covariance matrix.

        Returns:
            Tuple ``(scaled_weights, scale_applied)``.
        """
        symbols = [s for s in weights if s in covariance.columns]
        if not symbols:
            return dict(weights), 1.0
        cov = covariance.loc[symbols, symbols].to_numpy(dtype=float)
        if not np.all(np.isfinite(cov)):
            return dict(weights), 1.0
        w = np.array([weights[s] for s in symbols], dtype=float)
        variance = float(w @ cov @ w)
        if variance <= 1e-18:
            return dict(weights), 1.0
        estimated_vol = math.sqrt(variance)
        scale = self.config.target_volatility / estimated_vol
        scale = float(np.clip(scale, 0.0, 3.0))
        scaled = {symbol: weight * scale for symbol, weight in weights.items()}
        return scaled, scale

    def _apply_kelly(
        self, weights: Mapping[str, float], kelly: float
    ) -> Tuple[Dict[str, float], float]:
        """Cap aggregate gross exposure at the fractional-Kelly level.

        Args:
            weights: Current weights.
            kelly: Raw Kelly fraction.

        Returns:
            Tuple ``(weights, scale_applied)``.
        """
        target_gross = min(self.config.kelly_fraction * float(kelly), self.config.kelly_cap)
        gross = self._gross(weights)
        if gross <= 1e-12 or target_gross <= 0:
            return dict(weights), 0.0
        if gross <= target_gross:
            return dict(weights), 1.0
        scale = target_gross / gross
        return {symbol: weight * scale for symbol, weight in weights.items()}, float(scale)

    def _risk_parity_blend(
        self, weights: Mapping[str, float], covariance: pd.DataFrame
    ) -> Dict[str, float]:
        """Blend raw weights with equal-risk-contribution weights.

        Args:
            weights: Current weights.
            covariance: Annualised covariance matrix.

        Returns:
            Blended weights preserving the original signs and gross exposure.
        """
        symbols = [s for s in weights if s in covariance.columns]
        if len(symbols) < 2:
            return dict(weights)
        cov = covariance.loc[symbols, symbols]
        cov_values = cov.to_numpy(dtype=float)
        if not np.all(np.isfinite(cov_values)):
            return dict(weights)
        try:
            rp = risk_parity_weights(
                cov_values,
                max_iter=self.config.risk_parity_max_iter,
                tol=self.config.risk_parity_tol,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Risk parity solver failed: %s", exc)
            return dict(weights)

        gross = self._gross({s: weights[s] for s in symbols})
        signs = np.sign([weights[s] for s in symbols])
        blended: Dict[str, float] = dict(weights)
        blend = float(np.clip(self.config.risk_parity_blend, 0.0, 1.0))
        for index, symbol in enumerate(symbols):
            raw_abs = abs(weights[symbol])
            rp_abs = rp[index] * gross
            magnitude = (1.0 - blend) * raw_abs + blend * rp_abs
            blended[symbol] = float(signs[index] * magnitude)
        return blended

    def _apply_caps(self, weights: Mapping[str, float]) -> Dict[str, float]:
        """Clip per-symbol and aggregate exposure, dropping dust.

        Args:
            weights: Current weights.

        Returns:
            Capped weights.
        """
        capped = {
            symbol: float(np.clip(weight, -self.config.max_symbol_weight, self.config.max_symbol_weight))
            for symbol, weight in weights.items()
        }
        gross = self._gross(capped)
        if gross > self.config.max_gross_leverage and gross > 0:
            scale = self.config.max_gross_leverage / gross
            capped = {symbol: weight * scale for symbol, weight in capped.items()}
        band = self.risk_config.rebalance_band
        return {
            symbol: (0.0 if abs(weight) < band else weight)
            for symbol, weight in capped.items()
        }

    @staticmethod
    def _gross(weights: Mapping[str, float]) -> float:
        """Sum of absolute weights.

        Args:
            weights: Weight mapping.

        Returns:
            Gross exposure as a fraction of equity.
        """
        return float(sum(abs(w) for w in weights.values()))


def realized_covariance(
    data: Mapping[str, pd.DataFrame],
    window: int = 60,
    interval: str = "1d",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Rolling annualised covariance matrix of log returns.

    Args:
        data: Mapping of symbol -> OHLCV frame.
        window: Rolling window in bars.
        interval: Bar interval (used for annualisation).
        min_periods: Minimum observations; defaults to ``window``.

    Returns:
        The most recent annualised covariance DataFrame (symbols x symbols).
        An empty DataFrame is returned when there is insufficient history.
    """
    from quant_system.data.preprocessing import log_returns  # local: keeps import light

    min_periods = min_periods or window
    returns = pd.DataFrame(
        {symbol: log_returns(frame["close"].astype(float)) for symbol, frame in data.items()}
    ).dropna()
    if len(returns) < min_periods:
        return pd.DataFrame()
    factor = float(cfg.TRADING_DAYS_PER_YEAR if interval == "1d" else 252.0 * 24.0)
    cov = returns.iloc[-window:].cov() * factor
    return cov


__all__: List[str] = [
    "Fill",
    "Trade",
    "Position",
    "Portfolio",
    "kelly_fraction",
    "kelly_fraction_from_trades",
    "kelly_fraction_from_returns",
    "inverse_volatility_weights",
    "risk_parity_weights",
    "risk_contributions",
    "SizingEngine",
    "realized_covariance",
]
