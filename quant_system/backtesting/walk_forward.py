"""Walk-forward optimisation with out-of-sample validation.

Method
------
The sample is split into rolling (or anchored) windows:

.. code-block:: text

    |<------ IS 12m ------>|<-- OOS 3m -->|
                           |<------ IS 12m ------>|<-- OOS 3m -->|

For every segment:

1. Sweep the parameter grid on the **in-sample** window only, using the exact
   event-driven engine that will trade the parameters.  (Optimising with the
   production engine - rather than a faster surrogate - removes the risk of the
   optimiser exploiting a modelling shortcut.)
2. Select the parameter set maximising the configured objective (Sharpe).
3. Re-run the engine on the **out-of-sample** window with those frozen
   parameters.
4. Stitch the OOS return streams of all segments into one continuous
   out-of-sample equity curve - the only curve that is free of parameter
   selection bias.

Curve-fitting guard
-------------------
``degradation = (IS_objective - OOS_objective) / |IS_objective|`` is reported per
segment.  A segment is *accepted* when degradation stays below
``max_degradation_pct`` (15 % by default); a systematic pattern of large
degradations is the standard fingerprint of an over-fitted parameter set.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quant_system.analytics import metrics as ametrics
from quant_system.backtesting.engine import BacktestEngine, BacktestResult
from quant_system.config import settings as cfg
from quant_system.strategies.adaptive_momentum import AdaptiveMomentumStrategy
from quant_system.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

StrategyFactory = Callable[[], List[BaseStrategy]]


@dataclass
class SegmentResult:
    """Outcome of one in-sample / out-of-sample segment.

    Attributes:
        segment_id: Zero-based segment index.
        is_start / is_end: In-sample date bounds (end exclusive).
        oos_start / oos_end: Out-of-sample date bounds (end exclusive).
        best_params: Parameters selected on the IS window.
        is_metrics: Performance metrics on the IS window.
        oos_metrics: Performance metrics on the OOS window.
        is_objective: Objective value achieved in-sample.
        oos_objective: Objective value achieved out-of-sample.
        degradation_pct: Relative degradation IS -> OOS.
        accepted: ``True`` when degradation is within tolerance.
        oos_returns: Out-of-sample returns for this segment.
        oos_equity: Out-of-sample equity for this segment (standalone).
        n_combinations: Number of parameter sets evaluated.
    """

    segment_id: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    best_params: Dict[str, Any]
    is_metrics: Dict[str, float]
    oos_metrics: Dict[str, float]
    is_objective: float
    oos_objective: float
    degradation_pct: float
    accepted: bool
    oos_returns: pd.Series
    oos_equity: pd.Series
    n_combinations: int = 0


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward output.

    Attributes:
        segments: Per-segment results.
        oos_returns: Stitched out-of-sample return stream.
        oos_equity: Stitched out-of-sample equity curve.
        metrics: Performance metrics of the stitched OOS curve.
        regime_metrics: Regime-conditional breakdown of the stitched OOS curve.
        degradation_summary: Per-segment degradation table.
        parameter_history: Parameter sets selected per segment.
    """

    segments: List[SegmentResult] = field(default_factory=list)
    oos_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    oos_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: Dict[str, float] = field(default_factory=dict)
    regime_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    degradation_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    parameter_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of segments whose degradation stayed within tolerance."""
        if not self.segments:
            return 0.0
        return float(np.mean([segment.accepted for segment in self.segments]))

    @property
    def mean_degradation_pct(self) -> float:
        """Average IS -> OOS degradation across segments."""
        if not self.segments:
            return 0.0
        return float(np.mean([segment.degradation_pct for segment in self.segments]))

    def report(self) -> str:
        """Render a human-readable walk-forward report.

        Returns:
            A multi-line summary string.
        """
        lines: List[str] = []
        lines.append("=" * 78)
        lines.append("WALK-FORWARD OPTIMISATION REPORT")
        lines.append("=" * 78)
        lines.append(f"Segments              : {len(self.segments)}")
        lines.append(f"Acceptance rate       : {self.acceptance_rate:.1%}")
        lines.append(f"Mean degradation      : {self.mean_degradation_pct:.2f}%")
        lines.append("")
        lines.append("Stitched out-of-sample performance:")
        lines.append("-" * 78)
        lines.append(ametrics.format_summary(self.metrics))
        lines.append("")
        lines.append("Per-segment detail:")
        lines.append("-" * 78)
        if not self.degradation_summary.empty:
            lines.append(self.degradation_summary.to_string())
        lines.append("=" * 78)
        return "\n".join(lines)


class WalkForwardOptimizer:
    """Rolling in-sample / out-of-sample optimiser.

    Attributes:
        data: Full aligned OHLCV universe.
        regime_states: Causal regime series aligned to ``data``.
        regime_probabilities: Optional regime probability frame.
        config: Walk-forward configuration.
        strategy_factory: Callable returning a fresh strategy list per run.
        warmup_bars: Indicator warm-up prepended to every window.
    """

    def __init__(
        self,
        data: Mapping[str, pd.DataFrame],
        regime_states: pd.Series,
        regime_probabilities: Optional[pd.DataFrame] = None,
        config: Optional[cfg.WalkForwardConfig] = None,
        strategy_factory: Optional[StrategyFactory] = None,
        warmup_bars: int = 252,
        initial_capital: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        """Initialise the optimiser.

        Args:
            data: Mapping of symbol -> aligned OHLCV frame.
            regime_states: Regime series aligned to the data index.
            regime_probabilities: Optional probability frame.
            config: Walk-forward configuration.
            strategy_factory: Factory producing fresh strategies; defaults to the
                standard three-strategy book (metals + FX stat arb, momentum).
            warmup_bars: Extra bars prepended to every window so indicators are
                fully initialised (these bars are never scored).
            initial_capital: Starting equity per engine run.
            verbose: Emit per-segment progress logging.
        """
        self.data: Dict[str, pd.DataFrame] = {k: v for k, v in data.items()}
        if not self.data:
            raise ValueError("WalkForwardOptimizer requires at least one symbol.")
        self.regime_states: pd.Series = pd.Series(regime_states)
        self.regime_probabilities: Optional[pd.DataFrame] = regime_probabilities
        self.config: cfg.WalkForwardConfig = config or cfg.DEFAULT_SETTINGS.walk_forward
        self.strategy_factory: StrategyFactory = (
            strategy_factory or self._default_strategy_factory
        )
        self.warmup_bars: int = int(warmup_bars)
        self.initial_capital: float = float(
            initial_capital or cfg.DEFAULT_SETTINGS.sizing.initial_capital
        )
        self.verbose: bool = verbose
        self._index: pd.DatetimeIndex = self._build_index()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> WalkForwardResult:
        """Execute the full walk-forward study.

        Returns:
            A :class:`WalkForwardResult`.

        Raises:
            ValueError: If the sample is too short for even one segment.
        """
        segments = self._segment_bounds()
        if not segments:
            raise ValueError(
                "Sample is too short for a single in-sample/out-of-sample segment; "
                f"need at least {self.config.min_bars + self.config.oos_months * 21} bars."
            )

        param_grid = self.build_parameter_grid()
        results: List[SegmentResult] = []

        for segment_id, (is_start, is_end, oos_start, oos_end) in enumerate(segments):
            if self.verbose:
                logger.info(
                    "Segment %d/%d  IS %s..%s  OOS %s..%s",
                    segment_id + 1,
                    len(segments),
                    is_start.date(),
                    is_end.date(),
                    oos_start.date(),
                    oos_end.date(),
                )

            best_params, is_result, is_objective = self._optimise(
                is_start, is_end, param_grid
            )
            oos_result = self._evaluate(oos_start, oos_end, best_params)
            oos_objective = self._objective(oos_result)

            is_metrics = ametrics.performance_summary(
                is_result.equity_curve, is_result.trades
            )
            oos_metrics = ametrics.performance_summary(
                oos_result.equity_curve, oos_result.trades
            )
            degradation = self._degradation(is_objective, oos_objective)
            accepted = (
                degradation < self.config.max_degradation_pct
                and oos_objective >= self.config.min_oos_sharpe
                if self.config.objective == "sharpe"
                else degradation < self.config.max_degradation_pct
            )

            oos_returns = self._slice_returns(oos_result, oos_start, oos_end)
            results.append(
                SegmentResult(
                    segment_id=segment_id,
                    is_start=is_start,
                    is_end=is_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    best_params=best_params,
                    is_metrics=is_metrics,
                    oos_metrics=oos_metrics,
                    is_objective=float(is_objective),
                    oos_objective=float(oos_objective),
                    degradation_pct=float(degradation),
                    accepted=bool(accepted),
                    oos_returns=oos_returns,
                    oos_equity=self.initial_capital * (1.0 + oos_returns).cumprod(),
                    n_combinations=len(param_grid),
                )
            )
            if self.verbose:
                logger.info(
                    "  best=%s  IS %.3f -> OOS %.3f  (degradation %.1f%%, %s)",
                    best_params,
                    is_objective,
                    oos_objective,
                    degradation,
                    "accepted" if accepted else "REJECTED",
                )

        return self._assemble(results)

    # ------------------------------------------------------------------ #
    # Parameter grid
    # ------------------------------------------------------------------ #
    def build_parameter_grid(self) -> List[Dict[str, Any]]:
        """Build the cross-product parameter grid.

        Returns:
            List of parameter dictionaries. The stat-arb Z-score entry threshold
            (``1.5`` .. ``2.5`` step ``0.1``) is crossed with the momentum
            lookback pairs.
        """
        return [
            {
                "entry_z": float(z_entry),
                "fast_base": int(fast_base),
                "slow_base": int(slow_base),
            }
            for z_entry, (fast_base, slow_base) in itertools.product(
                self.config.z_entry_grid, self.config.momentum_grid
            )
        ]

    @staticmethod
    def _group_grid_by_structural_parameters(
        param_grid: Sequence[Dict[str, Any]],
    ) -> "OrderedDict[Tuple[int, int], List[float]]":
        """Group the grid by the parameters that change indicator preparation.

        Only the momentum ``fast_base``/``slow_base`` lookbacks change the
        indicators; the stat-arb ``entry_z`` threshold is a pure runtime
        parameter.  Grouping lets the optimiser prepare the indicators once per
        structural configuration and replay the whole Z-threshold sweep on top of
        it, which removes ~90 % of the compute.

        Args:
            param_grid: Flat parameter grid.

        Returns:
            Ordered mapping ``(fast_base, slow_base) -> [entry_z, ...]``.
        """
        from collections import OrderedDict

        grouped: "OrderedDict[Tuple[int, int], List[float]]" = OrderedDict()
        for params in param_grid:
            key = (int(params["fast_base"]), int(params["slow_base"]))
            grouped.setdefault(key, []).append(float(params["entry_z"]))
        return grouped

    # ------------------------------------------------------------------ #
    # Segmentation
    # ------------------------------------------------------------------ #
    def _build_index(self) -> pd.DatetimeIndex:
        """Build the master index from the symbol calendars.

        Returns:
            The aligned DatetimeIndex.
        """
        index = None
        for frame in self.data.values():
            index = frame.index if index is None else index.intersection(frame.index)
        if index is None:
            raise ValueError("No overlapping timestamps across symbols.")
        return pd.DatetimeIndex(sorted(index))

    def _segment_bounds(self) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Compute the (IS start, IS end, OOS start, OOS end) tuples.

        Returns:
            List of four-tuples; IS/OOS ends are exclusive.
        """
        index = self._index
        if len(index) < self.config.min_bars:
            return []
        is_bars = max(int(self.config.is_months * 21), self.config.min_bars)
        oos_bars = max(int(self.config.oos_months * 21), 21)
        step_bars = max(int(self.config.step_months * 21), 21)

        bounds: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        start = 0 if self.config.anchored is False else 0
        while True:
            is_start_pos = start
            is_end_pos = is_start_pos + is_bars
            oos_start_pos = is_end_pos
            oos_end_pos = oos_start_pos + oos_bars
            if oos_end_pos > len(index):
                break
            bounds.append(
                (
                    pd.Timestamp(index[is_start_pos]),
                    pd.Timestamp(index[is_end_pos]),
                    pd.Timestamp(index[oos_start_pos]),
                    pd.Timestamp(index[min(oos_end_pos, len(index) - 1)]),
                )
            )
            if self.config.anchored:
                # Anchored: the IS start stays fixed and the window grows.
                is_bars += step_bars
            else:
                start += step_bars
        return bounds

    # ------------------------------------------------------------------ #
    # Engine runs
    # ------------------------------------------------------------------ #
    def _optimise(
        self,
        is_start: pd.Timestamp,
        is_end: pd.Timestamp,
        param_grid: Sequence[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], BacktestResult, float]:
        """Sweep the grid on the in-sample window.

        The sweep is nested: the outer loop walks the *structural* parameters
        (momentum lookbacks) and prepares the indicators once per configuration;
        the inner loop replays the Z-threshold sweep on the already-prepared
        strategies, resetting only their mutable trading state.

        Args:
            is_start: In-sample start (inclusive).
            is_end: In-sample end (exclusive).
            param_grid: Parameter combinations to evaluate.

        Returns:
            Tuple ``(best_params, best_result, best_objective)``.
        """
        best_params: Dict[str, Any] = dict(param_grid[0]) if param_grid else {}
        best_result: Optional[BacktestResult] = None
        best_objective = float("-inf")

        grouped = self._group_grid_by_structural_parameters(param_grid)
        for (fast_base, slow_base), z_values in grouped.items():
            structural = {"fast_base": fast_base, "slow_base": slow_base}
            strategies, data_slice, regime_slice, prob_slice, slice_index = self._prepare_window(
                is_start, is_end, structural
            )
            for z_entry in z_values:
                for strategy in strategies:
                    strategy.reset_runtime_state()
                params = {**structural, "entry_z": z_entry}
                engine = BacktestEngine(
                    data=data_slice,
                    strategies=strategies,
                    regime_states=regime_slice,
                    regime_probabilities=prob_slice,
                    initial_capital=self.initial_capital,
                    params=params,
                    strategies_prepared=True,
                    compute_metrics=False,
                )
                result = engine.run()
                objective = self._objective(result)
                if np.isfinite(objective) and objective > best_objective:
                    best_objective = objective
                    best_params = dict(params)
                    best_result = result
            del slice_index

        if best_result is None:
            best_result = self._run_window(is_start, is_end, {})
            best_objective = self._objective(best_result)
        return best_params, best_result, float(best_objective)

    def _evaluate(
        self,
        oos_start: pd.Timestamp,
        oos_end: pd.Timestamp,
        params: Mapping[str, Any],
    ) -> BacktestResult:
        """Run the engine on the out-of-sample window with frozen parameters.

        Args:
            oos_start: OOS start (inclusive).
            oos_end: OOS end (exclusive).
            params: Parameters selected in-sample.

        Returns:
            The :class:`BacktestResult`.
        """
        return self._run_window(oos_start, oos_end, dict(params))

    def _slice_window(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> Tuple[Dict[str, pd.DataFrame], pd.Series, Optional[pd.DataFrame], pd.DatetimeIndex]:
        """Slice the universe (with warm-up) for a window.

        Args:
            start: Window start (inclusive).
            end: Window end (exclusive).

        Returns:
            Tuple ``(data_slice, regime_slice, prob_slice, slice_index)``.
        """
        start_pos = int(self._index.searchsorted(pd.Timestamp(start)))
        end_pos = int(self._index.searchsorted(pd.Timestamp(end), side="right"))
        warmup_start = max(0, start_pos - self.warmup_bars)
        slice_index = self._index[warmup_start:end_pos]
        data_slice = {
            symbol: frame.reindex(slice_index).ffill() for symbol, frame in self.data.items()
        }
        regime_slice = self.regime_states.reindex(slice_index).ffill().fillna(
            cfg.STATE_RANGE_BOUND
        )
        prob_slice = (
            self.regime_probabilities.reindex(slice_index).ffill()
            if self.regime_probabilities is not None
            else None
        )
        return data_slice, regime_slice, prob_slice, slice_index

    def _prepare_window(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        params: Mapping[str, Any],
    ) -> Tuple[
        List[BaseStrategy],
        Dict[str, pd.DataFrame],
        pd.Series,
        Optional[pd.DataFrame],
        pd.DatetimeIndex,
    ]:
        """Slice a window and prepare a fresh strategy book for it.

        Args:
            start: Window start (inclusive).
            end: Window end (exclusive).
            params: Parameter overrides (only *structural* ones affect preparation).

        Returns:
            Tuple ``(strategies, data_slice, regime_slice, prob_slice, slice_index)``.
        """
        data_slice, regime_slice, prob_slice, slice_index = self._slice_window(start, end)
        strategies = self.strategy_factory()
        apply_parameters(strategies, params)
        for strategy in strategies:
            strategy.reset()
            strategy.prepare(data_slice, slice_index)
        return strategies, data_slice, regime_slice, prob_slice, slice_index

    def _run_window(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        params: Mapping[str, Any],
    ) -> BacktestResult:
        """Slice the data (with warm-up) and run one engine pass.

        Args:
            start: Window start (inclusive).
            end: Window end (exclusive).
            params: Runtime parameter overrides.

        Returns:
            The :class:`BacktestResult`.
        """
        data_slice, regime_slice, prob_slice, _ = self._slice_window(start, end)
        strategies = self.strategy_factory()
        apply_parameters(strategies, params)
        engine = BacktestEngine(
            data=data_slice,
            strategies=strategies,
            regime_states=regime_slice,
            regime_probabilities=prob_slice,
            initial_capital=self.initial_capital,
            params=dict(params),
        )
        return engine.run()

    # ------------------------------------------------------------------ #
    # Scoring helpers
    # ------------------------------------------------------------------ #
    def _objective(self, result: BacktestResult) -> float:
        """Evaluate the optimisation objective on a result.

        Args:
            result: Backtest result to score.

        Returns:
            The objective value (``-inf`` when undefined).
        """
        returns = result.returns
        if returns is None or len(returns) < 5:
            return float("-inf")
        if self.config.objective == "sharpe":
            value = ametrics.sharpe_ratio(returns)
        elif self.config.objective == "sortino":
            value = ametrics.sortino_ratio(returns)
        elif self.config.objective == "calmar":
            value = ametrics.calmar_ratio(result.equity_curve)
        elif self.config.objective == "net_profit":
            value = float(result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1.0)
        else:
            raise ValueError(f"Unknown objective {self.config.objective!r}.")
        return float(value) if np.isfinite(value) else float("-inf")

    @staticmethod
    def _degradation(is_objective: float, oos_objective: float) -> float:
        """Relative performance degradation between IS and OOS.

        Args:
            is_objective: In-sample objective.
            oos_objective: Out-of-sample objective.

        Returns:
            Degradation in percent. For a positive IS objective this is
            ``(IS - OOS) / IS * 100``; a negative OOS result on a positive IS
            result therefore yields a value above 100 %. When the IS objective is
            non-positive the metric is reported as ``0.0`` (there is no edge to
            degrade) and the segment is judged on the OOS floor alone.
        """
        if not np.isfinite(is_objective) or not np.isfinite(oos_objective):
            return 100.0
        if is_objective <= 0:
            return 0.0
        return float((is_objective - oos_objective) / abs(is_objective) * 100.0)

    @staticmethod
    def _slice_returns(
        result: BacktestResult, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """Extract the returns belonging to a window (excluding warm-up).

        Args:
            result: Backtest result.
            start: Window start (inclusive).
            end: Window end (exclusive).

        Returns:
            The sliced return series.
        """
        returns = result.returns
        if returns is None or returns.empty:
            return pd.Series(dtype=float)
        mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
        return returns.loc[mask]

    # ------------------------------------------------------------------ #
    # Assembly
    # ------------------------------------------------------------------ #
    def _assemble(self, segments: List[SegmentResult]) -> WalkForwardResult:
        """Stitch OOS streams and build the summary tables.

        Args:
            segments: Per-segment results.

        Returns:
            The aggregated :class:`WalkForwardResult`.
        """
        if not segments:
            return WalkForwardResult()

        oos_returns = pd.concat([s.oos_returns for s in segments if not s.oos_returns.empty])
        oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")].sort_index()
        oos_returns.name = "oos_return"
        oos_equity = self.initial_capital * (1.0 + oos_returns.fillna(0.0)).cumprod()
        oos_equity.name = "oos_equity"

        # Trade-level statistics for the stitched curve are unavailable (each
        # segment runs its own book), so the summary is return-based.
        metrics = ametrics.performance_summary(oos_equity, None)

        degradation_rows: List[Dict[str, Any]] = []
        param_rows: List[Dict[str, Any]] = []
        for segment in segments:
            degradation_rows.append(
                {
                    "segment": segment.segment_id,
                    "is_start": segment.is_start.date(),
                    "is_end": segment.is_end.date(),
                    "oos_start": segment.oos_start.date(),
                    "oos_end": segment.oos_end.date(),
                    "is_sharpe": round(segment.is_objective, 3),
                    "oos_sharpe": round(segment.oos_objective, 3),
                    "degradation_pct": round(segment.degradation_pct, 2),
                    "accepted": segment.accepted,
                    "combinations": segment.n_combinations,
                }
            )
            row: Dict[str, Any] = {"segment": segment.segment_id}
            row.update(segment.best_params)
            param_rows.append(row)

        return WalkForwardResult(
            segments=segments,
            oos_returns=oos_returns,
            oos_equity=oos_equity,
            metrics=metrics,
            regime_metrics=pd.DataFrame(),
            degradation_summary=pd.DataFrame(degradation_rows),
            parameter_history=pd.DataFrame(param_rows),
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_strategy_factory() -> List[BaseStrategy]:
        """Build the standard three-strategy book.

        Returns:
            Fresh strategy instances.
        """
        from quant_system.strategies.stat_arb import build_default_stat_arb_book

        strategies: List[BaseStrategy] = list(build_default_stat_arb_book())
        strategies.append(AdaptiveMomentumStrategy())
        return strategies


# --------------------------------------------------------------------------- #
# Parameter application
# --------------------------------------------------------------------------- #
def apply_parameters(
    strategies: Sequence[BaseStrategy], params: Mapping[str, Any]
) -> None:
    """Push optimiser parameters into strategy instances.

    Two mechanisms are used, matching how each strategy consumes them:

    * **Runtime context parameters** (``entry_z``, ``exit_z``, ``stop_z``,
      ``gross_weight``) are read from ``StrategyContext.params`` at every bar, so
      they need no re-preparation.
    * **Structural parameters** (momentum ``fast_base`` / ``slow_base``) change
      the indicators themselves, so the strategy's frozen config is replaced
      *before* :meth:`BaseStrategy.prepare` runs.

    Args:
        strategies: Strategies to configure.
        params: Parameter dictionary.
    """
    if not params:
        return
    for strategy in strategies:
        if isinstance(strategy, AdaptiveMomentumStrategy):
            from dataclasses import replace

            fast = int(params.get("fast_base", strategy.config.base_period))
            slow = int(params.get("slow_base", strategy.config.slow_base_period))
            strategy.config = replace(
                strategy.config,
                base_period=fast,
                slow_base_period=slow,
                vwap_base_period=fast,
            )


__all__: List[str] = [
    "SegmentResult",
    "WalkForwardResult",
    "WalkForwardOptimizer",
    "apply_parameters",
]
