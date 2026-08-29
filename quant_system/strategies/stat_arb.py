"""Multi-asset statistical arbitrage (pairs trading) - active in **State 0**.

Two books are implemented:

``MetalsStatArb`` (XAUUSD / XAGUSD)
    Engle-Granger two-step cointegration on rolling windows, a rolling OLS hedge
    ratio ``beta`` and the spread ``S_t = P_XAU - beta * P_XAG``.

``FXStatArb`` (EURUSD / USDCHF)
    Log-price spread with a rolling OLS hedge ratio (the regression discovers
    ``beta ~ -1``), gated on the pair's rolling return correlation being below
    ``-0.75`` so the book only trades a genuinely inverted relationship.

Trading rules
-------------
* Long spread  : entry ``Z_t < -entry_z``  (``-2.0``), exit ``Z_t >= 0.0``
* Short spread : entry ``Z_t > +entry_z``  (``+2.0``), exit ``Z_t <= 0.0``
* Stop loss    : hard exit on ``|Z_t| >= stop_z`` (``3.5``) - a breakdown of the
  cointegrating relationship, not a normal adverse excursion.

Risk budgeting
--------------
The two legs are allocated so that the portfolio holds the spread itself: with
``k`` units of leg A and ``-k * beta`` units of leg B, the instantaneous PnL is
``k * (dP_A - beta * dP_B) = k * dS``.  Gross exposure is split proportional to
each leg's notional contribution so the deployed capital equals ``gross_weight``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from quant_system.config import settings as cfg
from quant_system.data.preprocessing import (
    compute_spread,
    log_returns,
    rolling_adf_pvalue,
    rolling_cointegration,
    rolling_correlation,
    rolling_hedge_ratio,
    rolling_zscore,
)
from quant_system.strategies.base import BaseStrategy, Signal, StrategyContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairSignal:
    """Decomposed view of a pair trade, attached to each leg's metadata.

    Attributes:
        pair_id: Unique pair identifier (``"XAUUSD_XAGUSD"``).
        side: ``+1`` long spread, ``-1`` short spread, ``0`` flat.
        zscore: Spread z-score that triggered the signal.
        beta: Hedge ratio applied to leg B.
        weight_a: Signed target weight for leg A.
        weight_b: Signed target weight for leg B.
        coint_pvalue: Latest rolling Engle-Granger p-value (``NaN`` if unused).
        correlation: Latest rolling return correlation (``NaN`` if unused).
        tag: Reason code (``entry_long``, ``exit_mean``, ``stop_out``, ...).
    """

    pair_id: str
    side: int
    zscore: float
    beta: float
    weight_a: float
    weight_b: float
    coint_pvalue: float
    adf_pvalue: float
    correlation: float
    tag: str

    def as_dict(self) -> Dict[str, float | int | str]:
        """Serialise to a flat dictionary for signal metadata."""
        return {
            "pair_id": self.pair_id,
            "side": self.side,
            "zscore": self.zscore,
            "beta": self.beta,
            "weight_a": self.weight_a,
            "weight_b": self.weight_b,
            "coint_pvalue": self.coint_pvalue,
            "adf_pvalue": self.adf_pvalue,
            "correlation": self.correlation,
            "tag": self.tag,
        }


class PairsStatArbStrategy(BaseStrategy):
    """Rolling cointegration pairs trading, active in the range-bound regime.

    Attributes:
        leg_a: First (dependent) leg, e.g. ``"XAUUSD"``.
        leg_b: Second (independent) leg, e.g. ``"XAGUSD"``.
        config: Statistical-arbitrage hyper-parameters.
        use_log: Regress/spread on log prices instead of raw prices.
        require_cointegration: Gate entries on the rolling Engle-Granger p-value.
        require_inverse_correlation: Gate entries on ``corr(A, B) < threshold``.
        gross_weight: Total gross exposure deployed when the pair is live.
    """

    name: str = "StatArb"
    active_states: frozenset[int] = frozenset({cfg.STATE_RANGE_BOUND})

    def __init__(
        self,
        leg_a: str,
        leg_b: str,
        config: Optional[cfg.StatArbConfig] = None,
        use_log: bool = False,
        require_cointegration: bool = True,
        require_inverse_correlation: bool = False,
        gross_weight: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialise the pairs strategy.

        Args:
            leg_a: Dependent leg symbol.
            leg_b: Independent leg symbol.
            config: Stat-arb hyper-parameters; defaults to ``settings.stat_arb``.
            use_log: Use log prices for the regression and the spread.
            require_cointegration: Require a significant Engle-Granger test.
            require_inverse_correlation: Require a strongly negative correlation.
            gross_weight: Override the configured gross exposure.
            name: Strategy name; defaults to ``f"{leg_a}_{leg_b}_StatArb"``.
        """
        super().__init__(
            symbols=(leg_a, leg_b),
            name=name or f"{leg_a}_{leg_b}_StatArb",
        )
        self.leg_a: str = leg_a
        self.leg_b: str = leg_b
        self.config: cfg.StatArbConfig = config or cfg.DEFAULT_SETTINGS.stat_arb
        self.use_log: bool = use_log
        self.require_cointegration: bool = require_cointegration
        self.require_inverse_correlation: bool = require_inverse_correlation
        self.gross_weight: float = (
            float(gross_weight) if gross_weight is not None else self.config.gross_weight
        )
        self.pair_id: str = f"{leg_a}_{leg_b}"

        # Runtime state
        self._frame: Optional[pd.DataFrame] = None
        self._position: int = 0  # -1 short spread, 0 flat, +1 long spread
        self._entry_z: float = 0.0
        self.required_history: int = (
            max(self.config.window, self.config.zscore_window) + self.config.coint_window
        )

    # ------------------------------------------------------------------ #
    # Preparation
    # ------------------------------------------------------------------ #
    def prepare(
        self,
        data: Mapping[str, pd.DataFrame],
        index: pd.DatetimeIndex,
    ) -> None:
        """Precompute the pair's indicator frame.

        Args:
            data: Mapping of symbol -> OHLCV frame. Both legs must be present.
            index: Engine master index the indicators are reindexed onto.

        Raises:
            KeyError: If a leg is missing from ``data``.
        """
        for leg in (self.leg_a, self.leg_b):
            if leg not in data:
                raise KeyError(f"Missing OHLCV data for pair leg {leg!r}.")

        ohlcv_a = data[self.leg_a]
        ohlcv_b = data[self.leg_b]
        series_a = self._transform(ohlcv_a["close"])
        series_b = self._transform(ohlcv_b["close"])

        frame = pd.DataFrame({"a": series_a, "b": series_b}).dropna()

        beta = rolling_hedge_ratio(frame["a"], frame["b"], window=self.config.window)
        # Scale-relative, causal cap: the bound at bar t uses only prices up to t.
        price_ratio = (frame["a"] / frame["b"]).abs()
        dynamic_cap = (
            self.config.max_abs_beta_ratio * price_ratio.replace([np.inf, -np.inf], np.nan)
        )
        beta = beta.clip(lower=-dynamic_cap, upper=dynamic_cap)
        beta = beta.clip(lower=-self.config.max_abs_beta, upper=self.config.max_abs_beta)
        spread = compute_spread(frame["a"], frame["b"], beta)
        zscore = rolling_zscore(spread, window=self.config.zscore_window)

        frame = pd.DataFrame(
            {
                "price_a": ohlcv_a["close"].reindex(frame.index).ffill(),
                "price_b": ohlcv_b["close"].reindex(frame.index).ffill(),
                "a": frame["a"],
                "b": frame["b"],
                "beta": beta,
                "spread": spread,
                "zscore": zscore,
            }
        )

        if self.require_cointegration:
            # Engle-Granger step 1+2 on the *same* series the spread is built
            # from, so the gate validates the relationship actually being traded.
            coint = rolling_cointegration(
                series_a,
                series_b,
                window=self.config.coint_window,
                step=self.config.coint_step,
            ).reindex(frame.index)
            frame["coint_pvalue"] = coint["pvalue"].ffill()
            # Engle-Granger step 2 applied directly to the traded (rolling-beta)
            # spread residual - higher power than testing the raw price levels.
            frame["adf_pvalue"] = rolling_adf_pvalue(
                spread,
                window=self.config.coint_window,
                step=self.config.coint_step,
            ).reindex(frame.index)
        else:
            frame["coint_pvalue"] = np.nan
            frame["adf_pvalue"] = np.nan

        if self.require_inverse_correlation:
            frame["correlation"] = rolling_correlation(
                log_returns(ohlcv_a["close"]),
                log_returns(ohlcv_b["close"]),
                window=self.config.correlation_window,
            ).reindex(frame.index)
        else:
            frame["correlation"] = np.nan

        frame["valid"] = self._validity_mask(frame)
        # Reindex onto the engine calendar; forward-fill keeps the transform causal.
        self._frame = frame.reindex(index).ffill()
        self._frame["valid"] = self._frame["valid"].fillna(False).astype(bool)
        super().prepare(data, index)

    def _transform(self, close: pd.Series) -> pd.Series:
        """Apply the log/level transform configured for this pair.

        Args:
            close: Raw close series.

        Returns:
            Log prices when ``use_log`` is set, else raw prices.

        Raises:
            ValueError: If prices are non-positive while ``use_log`` is set.
        """
        if not self.use_log:
            return close.astype(float)
        if (close <= 0).any():
            raise ValueError("Log-price spreads require strictly positive prices.")
        return np.log(close.astype(float))

    def _validity_mask(self, frame: pd.DataFrame) -> pd.Series:
        """Build the boolean mask of tradable bars.

        Args:
            frame: Indicator frame.

        Returns:
            Boolean Series; ``True`` where all entry gates pass.
        """
        valid = frame["zscore"].notna() & frame["beta"].notna()
        if self.require_cointegration:
            threshold = self.config.coint_pvalue
            method = self.config.coint_method
            if method == "coint":
                valid &= frame["coint_pvalue"] < threshold
            elif method == "adf_spread":
                valid &= frame["adf_pvalue"] < threshold
            elif method == "either":
                valid &= (frame["coint_pvalue"] < threshold) | (
                    frame["adf_pvalue"] < threshold
                )
            else:
                raise ValueError(
                    f"Unknown coint_method {method!r}; "
                    "use 'coint', 'adf_spread' or 'either'."
                )
        if self.require_inverse_correlation:
            valid &= frame["correlation"] < self.config.max_inverse_correlation
        return valid.fillna(False)

    # ------------------------------------------------------------------ #
    # Signals
    # ------------------------------------------------------------------ #
    def generate_signals(self, context: StrategyContext) -> List[Signal]:
        """Emit pair signals for the current bar.

        Args:
            context: Current-bar context.

        Returns:
            Empty list when nothing changes, otherwise one :class:`Signal` per leg.
        """
        if self._frame is None:
            return []
        row = self._row_at(context.bar_index)
        if row is None:
            return []

        z = float(row["zscore"]) if np.isfinite(row["zscore"]) else 0.0
        valid = bool(row["valid"])
        entry_z = float(context.param("entry_z", self.config.entry_z))
        exit_z = float(context.param("exit_z", self.config.exit_z))
        stop_z = float(context.param("stop_z", self.config.stop_z))

        current = self._position
        new_position = current
        tag = "hold"

        # --- stop loss takes precedence over everything ------------------ #
        if current != 0 and abs(z) >= stop_z:
            new_position = 0
            tag = "stop_out"
        elif not valid and current != 0:
            # Relationship broke down (failed cointegration / correlation drift).
            new_position = 0
            tag = "invalid_exit"
        elif current == 0 and valid:
            if z <= -entry_z:
                new_position = 1
                tag = "entry_long"
            elif z >= entry_z:
                new_position = -1
                tag = "entry_short"
        elif current == 1 and z >= exit_z:
            new_position = 0
            tag = "exit_mean"
        elif current == -1 and z <= exit_z:
            new_position = 0
            tag = "exit_mean"

        if new_position == current:
            return []

        self._position = new_position
        pair_signal = self._build_pair_signal(row, new_position, z, tag, context)
        logger.debug(
            "%s %s: z=%.2f side=%d tag=%s", self.name, context.timestamp, z, new_position, tag
        )
        return [
            Signal(
                symbol=self.leg_a,
                direction=int(np.sign(pair_signal.weight_a)),
                target_weight=pair_signal.weight_a,
                strategy=self.name,
                timestamp=context.timestamp,
                stop_atr_multiple=None,  # statistical exit, not an ATR stop
                tag=tag,
                metadata={**pair_signal.as_dict(), "leg": "a"},
            ),
            Signal(
                symbol=self.leg_b,
                direction=int(np.sign(pair_signal.weight_b)),
                target_weight=pair_signal.weight_b,
                strategy=self.name,
                timestamp=context.timestamp,
                stop_atr_multiple=None,
                tag=tag,
                metadata={**pair_signal.as_dict(), "leg": "b"},
            ),
        ]

    def notify_flat(self, symbol: str) -> None:
        """Reset the pair state when either leg is closed externally.

        Args:
            symbol: The leg that was flattened.
        """
        if symbol in (self.leg_a, self.leg_b):
            self._position = 0

    def reset_runtime_state(self) -> None:
        """Keep the prepared indicator frame; clear the pair position only."""
        self._position = 0

    def reset(self) -> None:
        """Clear prepared indicators and the internal position state."""
        super().reset()
        self._frame = None
        self._position = 0
        self._entry_z = 0.0

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _row_at(self, bar_index: int) -> Optional[pd.Series]:
        """Fetch the indicator row for a bar, guarding warm-up.

        Args:
            bar_index: Position in the engine index.

        Returns:
            The indicator row, or ``None`` during warm-up / out of range.
        """
        frame = self._frame
        if frame is None or bar_index < 0 or bar_index >= len(frame):
            return None
        row = frame.iloc[bar_index]
        if not np.isfinite(row.get("spread", np.nan)):
            return None
        return row

    def _build_pair_signal(
        self,
        row: pd.Series,
        side: int,
        z: float,
        tag: str,
        context: StrategyContext,
    ) -> PairSignal:
        """Compute the pair decomposition and the two leg weights.

        Args:
            row: Indicator row for the current bar.
            side: ``+1`` long spread, ``-1`` short spread, ``0`` flat.
            z: Spread z-score.
            tag: Reason code.
            context: Current-bar context (used for parameter overrides).

        Returns:
            The populated :class:`PairSignal`.
        """
        beta = float(row["beta"]) if np.isfinite(row["beta"]) else 1.0
        beta = float(np.clip(beta, -self.config.max_abs_beta, self.config.max_abs_beta))
        price_a = float(row["price_a"])
        price_b = float(row["price_b"])
        gross = float(context.param("gross_weight", self.gross_weight))

        if side == 0:
            return PairSignal(
                pair_id=self.pair_id,
                side=0,
                zscore=z,
                beta=beta,
                weight_a=0.0,
                weight_b=0.0,
                coint_pvalue=float(row.get("coint_pvalue", np.nan)),
                adf_pvalue=float(row.get("adf_pvalue", np.nan)),
                correlation=float(row.get("correlation", np.nan)),
                tag=tag,
            )

        weight_a, weight_b = self.leg_weights(price_a, price_b, beta, gross, side)
        return PairSignal(
            pair_id=self.pair_id,
            side=side,
            zscore=z,
            beta=beta,
            weight_a=weight_a,
            weight_b=weight_b,
            coint_pvalue=float(row.get("coint_pvalue", np.nan)),
            adf_pvalue=float(row.get("adf_pvalue", np.nan)),
            correlation=float(row.get("correlation", np.nan)),
            tag=tag,
        )

    @staticmethod
    def leg_weights(
        price_a: float,
        price_b: float,
        beta: float,
        gross: float,
        side: int,
    ) -> Tuple[float, float]:
        """Split gross exposure across the two legs to hold the spread.

        With ``k`` units of A and ``-k * beta`` units of B the PnL is
        ``k * dS``; ``k`` is chosen so ``|w_a| + |w_b| == gross``::

            k      = gross * equity / (P_a + |beta| * P_b)
            w_a    =  k * P_a / equity        * side
            w_b    = -k * beta * P_b / equity * side

        Args:
            price_a: Price of leg A.
            price_b: Price of leg B.
            beta: Hedge ratio.
            gross: Desired gross exposure as a fraction of equity.
            side: ``+1`` long spread, ``-1`` short spread.

        Returns:
            Tuple ``(weight_a, weight_b)`` of signed exposure fractions.

        Raises:
            ValueError: If prices are non-positive.
        """
        if price_a <= 0 or price_b <= 0:
            raise ValueError("Leg weights require strictly positive prices.")
        denominator = price_a + abs(beta) * price_b
        if denominator <= 0:
            return 0.0, 0.0
        w_a = side * gross * price_a / denominator
        w_b = -side * gross * beta * price_b / denominator
        return float(w_a), float(w_b)

    @property
    def position(self) -> int:
        """Current internal pair position (``-1``, ``0`` or ``+1``)."""
        return self._position

    @property
    def indicators(self) -> Optional[pd.DataFrame]:
        """The prepared indicator frame (for analytics and tests)."""
        return self._frame


def build_default_stat_arb_book(
    config: Optional[cfg.StatArbConfig] = None,
) -> List[PairsStatArbStrategy]:
    """Construct the standard stat-arb book described in the specification.

    Args:
        config: Shared hyper-parameters; defaults to ``settings.stat_arb``.

    Returns:
        List with the XAU/XAG cointegration pair and the EURUSD/USDCHF pair.
    """
    config = config or cfg.DEFAULT_SETTINGS.stat_arb
    metals_pair = cfg.DEFAULT_SETTINGS.universe.metals_pair
    fx_pair = cfg.DEFAULT_SETTINGS.universe.fx_pair
    return [
        PairsStatArbStrategy(
            leg_a=metals_pair[0],
            leg_b=metals_pair[1],
            config=config,
            use_log=False,
            require_cointegration=True,
            require_inverse_correlation=False,
            name="MetalsStatArb",
        ),
        PairsStatArbStrategy(
            leg_a=fx_pair[0],
            leg_b=fx_pair[1],
            config=config,
            use_log=True,
            require_cointegration=False,
            require_inverse_correlation=True,
            name="FXStatArb",
        ),
    ]


__all__: List[str] = [
    "PairSignal",
    "PairsStatArbStrategy",
    "build_default_stat_arb_book",
]
