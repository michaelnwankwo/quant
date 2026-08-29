"""Volatility-adjusted trend following - active in **State 1**.

Every lookback in this strategy is *dynamic*: the EMA/VWAP periods shrink when
volatility expands (so the trend follower reacts faster in fast markets) and
stretch when volatility contracts (so it stops whipsawing in quiet markets).

``EMA_Period_t = round( Base_Period * ATR_baseline / ATR_t )``

where ``ATR_baseline`` is the long-run mean ATR (default 100-bar rolling mean,
falling back to an expanding mean early in the sample).  Because the period
varies bar by bar, the EMA is computed with the time-varying recursion
``EMA_t = alpha_t * P_t + (1 - alpha_t) * EMA_{t-1}`` with
``alpha_t = 2 / (period_t + 1)`` (see
:func:`quant_system.data.preprocessing.dynamic_ema`).

Entry signals
-------------
* Long  : ``P > VWAP_dyn`` **and** ``EMA_dyn(20) > EMA_dyn(50)`` **and** ``RSI(14) > 55``
* Short : ``P < VWAP_dyn`` **and** ``EMA_dyn(20) < EMA_dyn(50)`` **and** ``RSI(14) < 45``

Exit signals
------------
* ATR trailing stop at ``2.5 x ATR`` (armed in the engine via
  ``Signal.stop_atr_multiple``), or
* dynamic trend inversion: ``EMA_fast`` crosses ``EMA_slow``, or price crosses
  back through the dynamic VWAP.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from quant_system.config import settings as cfg
from quant_system.data.preprocessing import (
    dynamic_ema,
    rolling_vwap,
    rsi,
    wilder_atr,
)
from quant_system.strategies.base import BaseStrategy, Signal, StrategyContext

logger = logging.getLogger(__name__)


class AdaptiveMomentumStrategy(BaseStrategy):
    """Volatility-scaled trend follower for USDJPY and XAUUSD.

    Attributes:
        config: Momentum hyper-parameters.
        gross_weight: Target gross exposure per instrument when a trend is live.
        indicators: Mapping of symbol -> prepared indicator frame.
    """

    name: str = "AdaptiveMomentum"
    active_states: frozenset[int] = frozenset({cfg.STATE_TREND})

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        config: Optional[cfg.MomentumConfig] = None,
        gross_weight: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialise the momentum strategy.

        Args:
            symbols: Instruments to trade; defaults to the configured momentum book.
            config: Momentum hyper-parameters.
            gross_weight: Override for the per-instrument target weight.
            name: Strategy name override.
        """
        universe_cfg = cfg.DEFAULT_SETTINGS.universe
        symbols = list(symbols) if symbols else list(universe_cfg.momentum_symbols)
        super().__init__(symbols=symbols, name=name)
        self.config: cfg.MomentumConfig = config or cfg.DEFAULT_SETTINGS.momentum
        self.gross_weight: float = (
            float(gross_weight) if gross_weight is not None else self.config.gross_weight
        )
        self.indicators: Dict[str, pd.DataFrame] = {}
        self._positions: Dict[str, int] = {symbol: 0 for symbol in symbols}
        self.required_history: int = (
            self.config.slow_base_period + self.config.atr_baseline_window
        )

    # ------------------------------------------------------------------ #
    # Preparation
    # ------------------------------------------------------------------ #
    def prepare(
        self,
        data: Mapping[str, pd.DataFrame],
        index: pd.DatetimeIndex,
    ) -> None:
        """Precompute dynamic indicators for every symbol.

        Args:
            data: Mapping of symbol -> OHLCV frame.
            index: Engine master index the indicators are reindexed onto.

        Raises:
            KeyError: If a traded symbol is missing from ``data``.
        """
        for symbol in self.symbols:
            if symbol not in data:
                raise KeyError(f"Missing OHLCV data for momentum symbol {symbol!r}.")
            self.indicators[symbol] = self._build_indicator_frame(data[symbol], index)
        super().prepare(data, index)

    def _build_indicator_frame(
        self, ohlcv: pd.DataFrame, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Build the dynamic indicator frame for one symbol.

        Args:
            ohlcv: OHLCV frame for the symbol.
            index: Engine master index.

        Returns:
            DataFrame with ``close``, ``atr``, ``atr_baseline``, ``period_fast``,
            ``period_slow``, ``ema_fast``, ``ema_slow``, ``vwap``, ``rsi`` and
            ``ready`` columns, reindexed onto ``index``.
        """
        close = ohlcv["close"].astype(float)
        atr = wilder_atr(ohlcv["high"], ohlcv["low"], close, period=cfg.DEFAULT_SETTINGS.hmm.atr_period)
        baseline = (
            atr.rolling(window=self.config.atr_baseline_window, min_periods=1).mean()
        )
        # Expanding-mean fallback for the first ``atr_baseline_window`` bars.
        baseline = baseline.fillna(atr.expanding(min_periods=1).mean())
        ratio = (baseline / atr.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        period_fast = self._scaled_period(ratio, self.config.base_period)
        period_slow = self._scaled_period(ratio, self.config.slow_base_period)
        period_vwap = self._scaled_period(ratio, self.config.vwap_base_period)

        frame = pd.DataFrame(
            {
                "close": close,
                "atr": atr,
                "atr_baseline": baseline,
                "period_fast": period_fast,
                "period_slow": period_slow,
                "ema_fast": dynamic_ema(close, period_fast),
                "ema_slow": dynamic_ema(close, period_slow),
                "vwap": rolling_vwap(ohlcv, period_vwap),
                "rsi": rsi(close, period=self.config.rsi_period),
            }
        )
        frame["ready"] = (
            frame[["close", "atr", "ema_fast", "ema_slow", "vwap", "rsi"]].notna().all(axis=1)
        )
        return frame.reindex(index).ffill()

    def _scaled_period(self, ratio: pd.Series, base_period: int) -> pd.Series:
        """Scale a base lookback by the ATR ratio.

        Args:
            ratio: Series ``ATR_baseline / ATR_t``.
            base_period: Lookback at neutral volatility.

        Returns:
            Integer-valued period series clipped to
            ``[ema_min_period, ema_max_period]``.
        """
        scaled = (float(base_period) * ratio).round()
        return scaled.clip(
            lower=float(self.config.ema_min_period), upper=float(self.config.ema_max_period)
        )

    # ------------------------------------------------------------------ #
    # Signals
    # ------------------------------------------------------------------ #
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """Emit trend signals for every traded symbol.

        Args:
            context: Current-bar context.

        Returns:
            List of :class:`Signal` objects (only for symbols whose state changed).
        """
        signals: List[Signal] = []
        for symbol in self.symbols:
            frame = self.indicators.get(symbol)
            if frame is None:
                continue
            if context.bar_index >= len(frame):
                continue
            row = frame.iloc[context.bar_index]
            if not bool(row["ready"]):
                continue

            signal = self._evaluate_symbol(symbol, row, context)
            if signal is not None:
                signals.append(signal)
        return signals

    def _evaluate_symbol(
        self,
        symbol: str,
        row: pd.Series,
        context: StrategyContext,
    ) -> Optional[Signal]:
        """Apply the entry/exit state machine for one symbol.

        Args:
            symbol: Instrument symbol.
            row: Indicator row for the current bar.
            context: Current-bar context.

        Returns:
            A :class:`Signal` when the desired state changes, else ``None``.
        """
        close = float(row["close"])
        vwap = float(row["vwap"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        momentum = float(row["rsi"])
        atr = float(row["atr"])

        rsi_long = float(context.param("rsi_long", self.config.rsi_long_threshold))
        rsi_short = float(context.param("rsi_short", self.config.rsi_short_threshold))
        gross = float(context.param("gross_weight", self.gross_weight))
        fast_base = int(context.param("fast_base", self.config.base_period))
        slow_base = int(context.param("slow_base", self.config.slow_base_period))
        if fast_base != self.config.base_period or slow_base != self.config.slow_base_period:
            # The optimiser is sweeping lookbacks: the prepared frame used the
            # configured bases, so recompute this bar's EMAs on the fly.
            ema_fast, ema_slow = self._recomputed_emas(symbol, context, fast_base, slow_base)

        trend_up = ema_fast > ema_slow
        trend_down = ema_fast < ema_slow
        above_vwap = close > vwap
        below_vwap = close < vwap

        long_entry = above_vwap and trend_up and momentum > rsi_long
        short_entry = below_vwap and trend_down and momentum < rsi_short

        current = self._positions.get(symbol, 0)
        new_position = current
        tag = "hold"

        if current == 0:
            if long_entry:
                new_position, tag = 1, "entry_long"
            elif short_entry:
                new_position, tag = -1, "entry_short"
        elif current == 1:
            # Trend inversion or price lost the dynamic VWAP -> exit.
            if trend_down or not above_vwap:
                new_position, tag = 0, "trend_inversion"
        elif current == -1:
            if trend_up or not below_vwap:
                new_position, tag = 0, "trend_inversion"

        if new_position == current:
            return None

        self._positions[symbol] = new_position
        return Signal(
            symbol=symbol,
            direction=new_position,
            target_weight=new_position * gross,
            strategy=self.name,
            timestamp=context.timestamp,
            stop_atr_multiple=self.config.atr_stop_multiple,
            tag=tag,
            metadata={
                "close": close,
                "vwap": vwap,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "rsi": momentum,
                "atr": atr,
                "fast_base": fast_base,
                "slow_base": slow_base,
            },
        )

    def _recomputed_emas(
        self,
        symbol: str,
        context: StrategyContext,
        fast_base: int,
        slow_base: int,
    ) -> tuple[float, float]:
        """Recompute dynamic EMAs for parameter sweeps.

        Args:
            symbol: Instrument symbol.
            context: Current-bar context (gives access to the OHLCV history).
            fast_base: Fast base lookback.
            slow_base: Slow base lookback.

        Returns:
            Tuple ``(ema_fast, ema_slow)`` evaluated at the current bar.
        """
        ohlcv = context.data.get(symbol)
        if ohlcv is None:
            return float("nan"), float("nan")
        sliced = ohlcv.iloc[: context.bar_index + 1]
        frame = self.indicators.get(symbol)
        if frame is None:
            return float("nan"), float("nan")
        ratio = frame["atr_baseline"].iloc[: context.bar_index + 1] / frame["atr"].iloc[
            : context.bar_index + 1
        ]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
        period_fast = self._scaled_period(ratio, fast_base)
        period_slow = self._scaled_period(ratio, slow_base)
        ema_fast = dynamic_ema(sliced["close"], period_fast).iloc[-1]
        ema_slow = dynamic_ema(sliced["close"], period_slow).iloc[-1]
        return float(ema_fast), float(ema_slow)

    def notify_flat(self, symbol: str) -> None:
        """Reset the directional state for a stopped-out symbol.

        Args:
            symbol: The instrument that was flattened.
        """
        if symbol in self._positions:
            self._positions[symbol] = 0

    def reset_runtime_state(self) -> None:
        """Keep the prepared indicator frames; clear directional state only."""
        self._positions = {symbol: 0 for symbol in self.symbols}

    def reset(self) -> None:
        """Clear prepared indicators and internal position state."""
        super().reset()
        self.indicators = {}
        self._positions = {symbol: 0 for symbol in self.symbols}

    @property
    def positions(self) -> Dict[str, int]:
        """Current internal directional state per symbol."""
        return dict(self._positions)


__all__: List[str] = ["AdaptiveMomentumStrategy"]
