"""End-to-end system verification suite.

This module walks the *whole* production pipeline in order and asserts that each
stage behaves as specified.  It is deliberately written so it can be consumed
three ways::

    python tests/verify_system.py                 # CLI report, exit code = failures
    python main.py --mode verify                  # routed through the CLI
    pytest tests/verify_system.py -q              # one test per stage

Stages
------
1. **Data Ingestion & Preprocessing** — fetch the universe (network first,
   synthetic fallback), then validate the three mandatory feature transforms:
   log returns, 14-period Wilder ATR, annualised 20-day rolling volatility and
   ``StandardScaler`` standardisation.
2. **HMM Switchboard Decoding** — fit the three-state Gaussian HMM, confirm the
   canonical re-labelling orders the states by variance, and verify the causal
   decoder produces a dense, in-range state path with rows summing to one.
3. **Strategy Signal Generation & Sizing** — assert regime routing (only the
   stat-arb book trades in State 0, only momentum in State 1, nothing opens in
   State 2), signal finiteness, and the Kelly / risk-parity / ATR sizing maths.
4. **MT5 Broker Connection / Simulated Fallback** — attempt a real MT5 terminal
   connection and, when it is unavailable (the usual case off a trading desk),
   transparently fall back to :class:`SimulatedBroker`; then verify the router's
   order lifecycle end to end.
5. **Trade Notifier Dispatch & Audio Queue Execution** — confirm that *only*
   confirmed fills raise a toast + voice alert, that pending/partial/rejected
   reports are filtered out, that the audio worker synthesises its WAV on a
   background thread without blocking the asyncio loop, and that the demo-mode
   daily trade cap is lifted while sizing and stops stay intact.

Every stage executes inside :func:`warnings.catch_warnings` with
``simplefilter("always")``; any warning emitted by the pipeline is recorded as a
*failed* check, which enforces the "0 errors, 0 warnings" acceptance criterion.

Author: quant_system
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# tests/verify_system.py -> parents[0]=tests, [1]=quant_system, [2]=repo root.
# The repo root (not the package dir) must be importable for `quant_system.*`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from quant_system.config import settings as cfg  # noqa: E402

# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
STAGES: Tuple[Tuple[str, str], ...] = (
    ("ingestion", "Data Ingestion & Preprocessing"),
    ("regime", "HMM Switchboard Decoding"),
    ("signals", "Strategy Signal Generation & Sizing"),
    ("broker", "MT5 Connection / Simulated Fallback"),
    ("notifier", "Trade Notifier Dispatch & Audio Queue"),
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single assertion inside a stage.

    Attributes:
        stage: Stage key (see :data:`STAGES`).
        name: Short human-readable check name.
        passed: Whether the check passed.
        detail: Observed values / failure reason.
        elapsed_ms: Wall-clock duration of the check.
    """

    stage: str
    name: str
    passed: bool
    detail: str = ""
    elapsed_ms: float = 0.0


@dataclass
class VerificationReport:
    """Aggregate outcome of a full verification run.

    Attributes:
        results: Every executed check, in execution order.
        elapsed_s: Total wall-clock duration in seconds.
        data_source: Ingestion backend that actually supplied the bars.
        artifacts: Paths written during the run (WAV clips, toast preview, ...).
    """

    results: List[CheckResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    data_source: str = "unknown"
    artifacts: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    @property
    def passed(self) -> bool:
        """Whether every check passed."""
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> List[CheckResult]:
        """The subset of checks that failed."""
        return [r for r in self.results if not r.passed]

    def for_stage(self, stage: str) -> List[CheckResult]:
        """Return the checks belonging to ``stage``.

        Args:
            stage: Stage key.

        Returns:
            The matching checks (possibly empty).
        """
        return [r for r in self.results if r.stage == stage]

    # ---------------------------------------------------------------- #
    def render(self) -> str:
        """Render the report as a fixed-width console table.

        Returns:
            The formatted multi-line report.
        """
        lines: List[str] = []
        lines.append("=" * 78)
        lines.append("  quant_system :: END-TO-END SYSTEM VERIFICATION")
        lines.append("=" * 78)
        for key, title in STAGES:
            checks = self.for_stage(key)
            if not checks:
                continue
            stage_ok = all(c.passed for c in checks)
            flag = "PASS" if stage_ok else "FAIL"
            lines.append("")
            lines.append(f"[{flag}]  {title}")
            lines.append("-" * 78)
            for check in checks:
                mark = "  ok  " if check.passed else " FAIL "
                timing = f"{check.elapsed_ms:7.1f} ms"
                lines.append(f"  {mark} {timing}  {check.name}")
                for chunk in _wrap(check.detail, 62):
                    lines.append(f"                        {chunk}")
        lines.append("")
        lines.append("=" * 78)
        total = len(self.results)
        failed = len(self.failures)
        lines.append(
            f"  RESULT: {total - failed}/{total} checks passed "
            f"in {self.elapsed_s:.2f}s   (data source: {self.data_source})"
        )
        if self.artifacts:
            for artifact in self.artifacts:
                lines.append(f"  artifact: {artifact}")
        lines.append("=" * 78)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - trivial delegation
        """Render the report."""
        return self.render()


def _wrap(text: str, width: int) -> List[str]:
    """Soft-wrap ``text`` for the console table.

    Args:
        text: Text to wrap.
        width: Maximum line width.

    Returns:
        List of wrapped lines (empty when ``text`` is empty).
    """
    if not text:
        return []
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #
class SystemVerifier:
    """Runs the five verification stages and collects :class:`CheckResult` rows.

    Attributes:
        start: Inclusive start date for the ingested history.
        end: Inclusive end date.
        interval: Bar interval.
        source: Ingestion backend selector (``"auto"`` tries the network first).
        use_cache: Whether the on-disk pickle cache may be used.
        voice_enabled: Whether stage 5 synthesises real audio.
        report: The report populated by :meth:`run`.
    """

    def __init__(
        self,
        start: str = "2018-01-01",
        end: str = "2024-12-31",
        interval: str = "1d",
        source: str = "auto",
        use_cache: bool = True,
        voice_enabled: bool = True,
    ) -> None:
        """Initialise the verifier.

        Args:
            start: Inclusive start date.
            end: Inclusive end date.
            interval: Bar interval.
            source: Ingestion backend selector.
            use_cache: Allow the on-disk data cache.
            voice_enabled: Synthesise real audio in stage 5.
        """
        self.start = start
        self.end = end
        self.interval = interval
        self.source = source
        self.use_cache = use_cache
        self.voice_enabled = voice_enabled
        self.report = VerificationReport()

    # ------------------------------------------------------------------ #
    # Driver
    # ------------------------------------------------------------------ #
    def run(self) -> VerificationReport:
        """Execute every stage in order.

        Returns:
            The populated :class:`VerificationReport`.
        """
        self.report = VerificationReport()
        started = time.perf_counter()
        # Shared state handed from stage to stage, mirroring the real pipeline.
        shared: Dict[str, Any] = {}
        runners = (
            self._stage_ingestion,
            self._stage_regime,
            self._stage_signals,
            self._stage_broker,
            self._stage_notifier,
        )
        for runner in runners:
            key = runner.__name__.replace("_stage_", "")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    runner(shared)
                except Exception as exc:  # pragma: no cover - defensive
                    self._record(
                        key,
                        f"{key} stage raised",
                        False,
                        f"{type(exc).__name__}: {exc}",
                    )
            for warning in caught:
                self._record(
                    key,
                    f"no warning: {warning.category.__name__}",
                    False,
                    f"{warning.filename}:{warning.lineno} {warning.message}",
                )
        self.report.elapsed_s = time.perf_counter() - started
        return self.report

    # ------------------------------------------------------------------ #
    def _record(self, stage: str, name: str, passed: bool, detail: str = "") -> None:
        """Append a check result.

        Args:
            stage: Stage key.
            name: Check name.
            passed: Outcome.
            detail: Observed values / failure reason.
        """
        self.report.results.append(
            CheckResult(stage=stage, name=name, passed=bool(passed), detail=detail)
        )

    def _check(
        self, stage: str, name: str, condition: bool, detail: str = ""
    ) -> bool:
        """Assert ``condition`` and record the outcome.

        Args:
            stage: Stage key.
            name: Check name.
            condition: The assertion result.
            detail: Observed values.

        Returns:
            ``condition`` (so callers can short-circuit).
        """
        self._record(stage, name, condition, detail)
        return bool(condition)

    # ------------------------------------------------------------------ #
    # Stage 1 — Data ingestion & preprocessing
    # ------------------------------------------------------------------ #
    def _stage_ingestion(self, shared: Dict[str, Any]) -> None:
        """Verify ingestion and the three mandatory feature transforms.

        Args:
            shared: Cross-stage state; receives ``data`` and ``features``.
        """
        stage = "ingestion"
        from quant_system.data.ingestion import DataIngestion, align_universe
        from quant_system.data.preprocessing import (
            build_feature_frame,
            log_returns,
            realized_volatility,
            standardize_features,
            wilder_atr,
        )

        # --- Ingestion ------------------------------------------------- #
        ingestion = DataIngestion(source=self.source, use_cache=self.use_cache)
        data = ingestion.fetch_universe(
            start=self.start, end=self.end, interval=self.interval
        )
        sources = ingestion.sources_used
        self.report.data_source = ",".join(sorted(set(sources.values()))) or "unknown"

        self._check(
            stage,
            "universe fetched (non-empty)",
            bool(data),
            f"{len(data)} symbols: {sorted(data)}",
        )
        if not data:
            return
        data = align_universe(data, method="intersection")
        shared["data"] = data

        lengths = {sym: len(frame) for sym, frame in data.items()}
        self._check(
            stage,
            "all symbols share one aligned index",
            len(set(lengths.values())) == 1,
            f"bars per symbol: {lengths}",
        )

        indexes = [frame.index for frame in data.values()]
        common = indexes[0]
        self._check(
            stage,
            "index identical across symbols",
            all(idx.equals(common) for idx in indexes),
            f"{len(common)} bars {common[0].date()} .. {common[-1].date()}",
        )
        self._check(
            stage,
            "index strictly increasing",
            bool(common.is_monotonic_increasing and common.is_unique),
            f"monotonic={common.is_monotonic_increasing} unique={common.is_unique}",
        )

        schema_ok = all(
            {"open", "high", "low", "close", "volume"}.issubset(frame.columns)
            for frame in data.values()
        )
        self._check(stage, "OHLCV schema complete", schema_ok)

        # --- Log returns ----------------------------------------------- #
        close = data["XAUUSD"]["close"] if "XAUUSD" in data else next(iter(data.values()))["close"]
        returns = log_returns(close)
        self._check(
            stage,
            "log returns: length preserved, first NaN only",
            len(returns) == len(close) and returns.isna().sum() <= 1,
            f"n={len(returns)} nan={int(returns.isna().sum())}",
        )
        manual = np.log(close / close.shift(1)).iloc[1:]
        self._check(
            stage,
            "log returns match ln(P_t / P_t-1)",
            bool(np.allclose(returns.iloc[1:].to_numpy(), manual.to_numpy(), atol=1e-12)),
            f"max abs diff={np.nanmax(np.abs(returns.iloc[1:].to_numpy() - manual.to_numpy())):.3e}",
        )
        self._check(
            stage,
            "log returns finite (no inf)",
            bool(np.isfinite(returns.dropna().to_numpy()).all()),
        )

        # --- Wilder ATR (14) ------------------------------------------- #
        ohlcv = data["XAUUSD"] if "XAUUSD" in data else next(iter(data.values()))
        atr = wilder_atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], period=14)
        self._check(
            stage,
            "Wilder ATR(14): positive and finite after warm-up",
            bool((atr.iloc[14:] > 0).all() and np.isfinite(atr.iloc[14:].to_numpy()).all()),
            f"min={atr.iloc[14:].min():.4f} max={atr.iloc[14:].max():.4f}",
        )
        # Independent Wilder recursion: TR -> SMA seed -> ATR_t = (13*ATR_t-1 + TR_t)/14
        prev_close = ohlcv["close"].shift(1)
        true_range = pd.concat(
            [
                ohlcv["high"] - ohlcv["low"],
                (ohlcv["high"] - prev_close).abs(),
                (ohlcv["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        # ``wilder_atr`` seeds with nanmean(tr[:14]); tr[0] is NaN because the
        # first bar has no previous close, so the seed averages TR[1..13].
        expected = np.full(len(true_range), np.nan)
        seed = float(np.nanmean(true_range.to_numpy(dtype=float)[:14]))
        expected[13] = seed
        prev = seed
        for pos in range(14, len(true_range)):
            prev = (prev * 13 + float(true_range.iloc[pos])) / 14.0
            expected[pos] = prev
        expected_series = pd.Series(expected, index=true_range.index)
        both = pd.concat([atr, expected_series], axis=1).dropna()
        self._check(
            stage,
            "Wilder ATR matches the (13*A_prev + TR)/14 recursion",
            bool(np.allclose(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy(), rtol=1e-9)),
            f"compared {len(both)} bars, max rel diff="
            f"{np.max(np.abs(both.iloc[:, 0].to_numpy() - both.iloc[:, 1].to_numpy()) / np.maximum(both.iloc[:, 1].to_numpy(), 1e-12)):.3e}",
        )

        # --- Annualised rolling volatility (N=20) ---------------------- #
        vol = realized_volatility(returns, window=20, annualize=True, interval="1d")
        naive = returns.rolling(20).std(ddof=1) * np.sqrt(252.0)
        both_vol = pd.concat([vol, naive], axis=1).dropna()
        self._check(
            stage,
            "annualised vol == sample std (ddof=1) x sqrt(252)",
            bool(np.allclose(both_vol.iloc[:, 0].to_numpy(), both_vol.iloc[:, 1].to_numpy(), rtol=1e-9)),
            f"compared {len(both_vol)} bars; last={vol.dropna().iloc[-1]:.4f}",
        )
        self._check(
            stage,
            "annualised vol strictly positive",
            bool((vol.dropna() > 0).all()),
            f"min={vol.dropna().min():.6f}",
        )

        # --- Feature frame & standardisation --------------------------- #
        frame = build_feature_frame(ohlcv, atr_period=14, vol_window=20, interval="1d")
        self._check(
            stage,
            "feature frame exposes X_t = [ret, atr, sigma]",
            {"ret", "atr", "sigma"}.issubset(frame.columns),
            f"columns={list(frame.columns)}",
        )
        shared["features"] = frame

        model_frame = frame[list(cfg.DEFAULT_SETTINGS.hmm.feature_columns)].dropna()
        self._check(
            stage,
            "HMM matrix has 3 columns matching settings.feature_columns",
            model_frame.shape[1] == 3,
            f"{list(cfg.DEFAULT_SETTINGS.hmm.feature_columns)} -> {model_frame.shape}",
        )

        scaled, scaler = standardize_features(model_frame)
        means = scaled.mean().to_numpy()
        stds = scaled.std(ddof=0).to_numpy()
        self._check(
            stage,
            "StandardScaler: column means ~ 0",
            bool(np.allclose(means, 0.0, atol=1e-8)),
            f"means={np.round(means, 10).tolist()}",
        )
        self._check(
            stage,
            "StandardScaler: column std ~ 1",
            bool(np.allclose(stds, 1.0, atol=1e-8)),
            f"stds={np.round(stds, 10).tolist()}",
        )
        shared["scaler"] = scaler

    # ------------------------------------------------------------------ #
    # Stage 2 — HMM switchboard decoding
    # ------------------------------------------------------------------ #
    def _stage_regime(self, shared: Dict[str, Any]) -> None:
        """Fit and decode the HMM switchboard.

        Args:
            shared: Cross-stage state; reads ``features``, writes ``regimes``.
        """
        stage = "regime"
        from quant_system.models.hmm_switchboard import (
            HMMSwitchboard,
            regime_distribution,
        )

        features = shared.get("features")
        if features is None:
            self._record(stage, "feature frame available", False, "stage 1 produced no features")
            return

        matrix = features[list(cfg.DEFAULT_SETTINGS.hmm.feature_columns)].dropna()
        switchboard = HMMSwitchboard(config=cfg.DEFAULT_SETTINGS.hmm)
        switchboard.fit(matrix)
        shared["switchboard"] = switchboard

        self._check(
            stage,
            "HMM fitted with 3 components, full covariance",
            switchboard.model_ is not None
            and int(switchboard.model_.n_components) == 3
            and switchboard.model_.covariance_type == "full",
            f"n_components={switchboard.model_.n_components} "
            f"covariance_type={switchboard.model_.covariance_type} "
            f"n_iter={switchboard.model_.n_iter} random_state={switchboard.model_.random_state}",
        )
        self._check(
            stage,
            "startprob_ / transmat_ are proper distributions",
            bool(
                np.isclose(switchboard.model_.startprob_.sum(), 1.0)
                and np.allclose(switchboard.model_.transmat_.sum(axis=1), 1.0, atol=1e-6)
            ),
            f"startprob sum={switchboard.model_.startprob_.sum():.10f} "
            f"transmat row sums={np.round(switchboard.model_.transmat_.sum(axis=1), 8).tolist()}",
        )
        self._check(
            stage,
            "means_ shape is (3, 3)",
            switchboard.model_.means_.shape == (3, 3),
            f"shape={switchboard.model_.means_.shape}",
        )

        # Canonical ordering: state 0 = lowest variance, state 2 = highest.
        stats = switchboard.state_statistics_
        self._check(
            stage,
            "canonical mapping defined for all 3 states",
            switchboard.mapping_ is not None and sorted(switchboard.mapping_.tolist()) == [0, 1, 2],
            f"mapping_={None if switchboard.mapping_ is None else switchboard.mapping_.tolist()}",
        )
        if stats is not None and "mean_sigma" in getattr(stats, "columns", []):
            sigmas = [float(stats.loc[idx, "mean_sigma"]) for idx in sorted(stats.index)]
            self._check(
                stage,
                "state 0 = lowest variance, state 2 = highest",
                sigmas[0] < sigmas[1] < sigmas[2],
                f"sigma by state = {[round(s, 5) for s in sigmas]}",
            )
        else:  # pragma: no cover - defensive
            self._record(stage, "state statistics expose mean_sigma", False, f"columns={stats}")

        # --- Viterbi decoding over the full history -------------------- #
        hidden = switchboard.model_.predict(
            switchboard.scaler_.transform(matrix.to_numpy(dtype=float))
        )
        canonical = np.asarray([switchboard.mapping_[h] for h in hidden], dtype=int)
        shared["states"] = pd.Series(canonical, index=matrix.index)

        self._check(
            stage,
            "decoded states cover every bar (no gaps)",
            len(canonical) == len(matrix),
            f"{len(canonical)} decoded / {len(matrix)} bars",
        )
        self._check(
            stage,
            "decoded states within {0, 1, 2}",
            bool(set(np.unique(canonical)).issubset({0, 1, 2})),
            f"observed={sorted(set(canonical.tolist()))}",
        )

        # --- Point prediction & probabilities -------------------------- #
        last_state = switchboard.predict_state(matrix)
        self._check(
            stage,
            "predict_state returns a canonical id",
            int(last_state) in (0, 1, 2),
            f"state={last_state}",
        )
        probabilities = switchboard.get_state_probabilities()
        self._check(
            stage,
            "get_state_probabilities has shape (3,) and sums to 1",
            probabilities.shape == (3,) and np.isclose(probabilities.sum(), 1.0, atol=1e-6),
            f"P={np.round(probabilities, 6).tolist()} sum={probabilities.sum():.10f}",
        )
        self._check(
            stage,
            "probabilities are non-negative",
            bool((probabilities >= -1e-9).all()),
            f"min={probabilities.min():.3e}",
        )

        distribution = regime_distribution(pd.Series(canonical, index=matrix.index))
        shares = {
            int(key): round(float(value) * 100, 2)
            for key, value in distribution.items()
            if str(key) != "transitions"
        }
        self._check(
            stage,
            "all three regimes are visited",
            len(shares) == 3 and all(v > 0 for v in shares.values()),
            f"{shares} (% of bars), "
            f"transitions={int(distribution.get('transitions', 0))}",
        )

        # --- Causal (walk-forward) decoder ----------------------------- #
        from quant_system.models.hmm_switchboard import CausalRegimeStreamer

        data = shared.get("data")
        if data is not None:
            from quant_system.backtesting.engine import build_default_regimes

            index = next(iter(data.values())).index
            stream_result = build_default_regimes(data, config=cfg.DEFAULT_SETTINGS.hmm)
            stream = CausalRegimeStreamer(config=cfg.DEFAULT_SETTINGS.hmm)
            aligned = stream.align_to(stream_result, index)
            shared["regimes"] = aligned
            dense = aligned.states.dropna()
            self._check(
                stage,
                "causal streamer emits a dense state path",
                len(dense) > 0,
                f"{len(dense)}/{len(index)} bars decoded",
            )
            self._check(
                stage,
                "causal states are canonical ids",
                bool(set(np.unique(dense.to_numpy())).issubset({0, 1, 2})),
                f"observed={sorted(set(int(v) for v in dense.to_numpy()))}",
            )
            probs = aligned.probabilities
            if not probs.empty:
                # align_to() pads the front of the frame: the streamer needs a
                # warm-up window before it has a fitted model, so the leading rows
                # are legitimately all-zero. Everything after that prefix must be a
                # proper distribution.
                row_sums = probs.sum(axis=1).to_numpy()
                zero_rows = np.flatnonzero(row_sums == 0.0)
                prefixed = bool(
                    np.array_equal(zero_rows, np.arange(zero_rows.size))
                )
                decoded = row_sums[zero_rows.size :]
                self._check(
                    stage,
                    "causal probabilities sum to 1 after the warm-up prefix",
                    prefixed
                    and decoded.size > 0
                    and bool(np.allclose(decoded, 1.0, atol=1e-6)),
                    f"{decoded.size} decoded rows sum to 1 "
                    f"[{decoded.min():.8f}, {decoded.max():.8f}]; "
                    f"{zero_rows.size} leading warm-up rows (contiguous={prefixed})",
                )

    # ------------------------------------------------------------------ #
    # Stage 3 — Strategy signal generation & sizing
    # ------------------------------------------------------------------ #
    def _stage_signals(self, shared: Dict[str, Any]) -> None:
        """Verify regime routing, signal validity and the sizing maths.

        Args:
            shared: Cross-stage state; reads ``data``, ``features``, ``regimes``.
        """
        stage = "signals"
        from quant_system.execution.portfolio import (
            Portfolio,
            SizingEngine,
            inverse_volatility_weights,
            kelly_fraction,
            risk_contributions,
            risk_parity_weights,
        )
        from quant_system.execution.risk_manager import RiskManager
        from quant_system.strategies.adaptive_momentum import AdaptiveMomentumStrategy
        from quant_system.strategies.base import StrategyContext
        from quant_system.strategies.stat_arb import build_default_stat_arb_book

        data = shared.get("data")
        regimes = shared.get("regimes")
        if data is None:
            self._record(stage, "market data available", False, "stage 1 produced no data")
            return

        index = next(iter(data.values())).index
        stat_arb = build_default_stat_arb_book()
        momentum = AdaptiveMomentumStrategy(symbols=list(data.keys()))
        book = list(stat_arb) + [momentum]
        shared["strategies"] = book

        self._check(
            stage,
            "strategy book built",
            bool(book),
            f"{[s.name for s in book]}",
        )

        # --- Regime routing -------------------------------------------- #
        routing = {}
        for state in (0, 1, 2):
            routing[state] = [s.name for s in book if s.is_active(state)]
        self._check(
            stage,
            "State 0 activates only the stat-arb book",
            all("StatArb" in name or "arb" in name.lower() for name in routing[0]) and bool(routing[0]),
            f"{routing[0]}",
        )
        self._check(
            stage,
            "State 1 activates only the momentum strategy",
            routing[1] == [momentum.name],
            f"{routing[1]}",
        )
        self._check(
            stage,
            "State 2 activates no entry strategy",
            routing[2] == [],
            f"{routing[2]}",
        )

        # --- Signals on the final bar ---------------------------------- #
        states = (
            regimes.states
            if regimes is not None
            else pd.Series(1, index=index, dtype=int)
        )
        last = len(index) - 1
        state_now = int(states.iloc[last]) if not pd.isna(states.iloc[last]) else 1
        portfolio = Portfolio(
            initial_capital=cfg.DEFAULT_SETTINGS.sizing.initial_capital,
            demo_config=cfg.DEFAULT_SETTINGS.demo,
        )
        context = StrategyContext(
            timestamp=pd.Timestamp(index[last]),
            bar_index=last,
            data=data,
            features={},
            regime_state=state_now,
            regime_probabilities=np.full(cfg.N_REGIMES, 1.0 / cfg.N_REGIMES),
            positions={},
            equity=portfolio.equity,
        )
        shared["context"] = context

        raw_targets: Dict[str, float] = {}
        for strategy in book:
            signals = (
                strategy.generate_signals(context)
                if strategy.is_active(state_now)
                else strategy.flat_signals(context)
            )
            for signal in signals:
                raw_targets[signal.symbol] = float(signal.target_weight)
        self._check(
            stage,
            "all emitted signals are finite",
            all(np.isfinite(v) for v in raw_targets.values()),
            f"{ {k: round(v, 4) for k, v in raw_targets.items()} }",
        )
        self._check(
            stage,
            "targets are bounded weights",
            all(abs(v) <= 1.0 + 1e-9 for v in raw_targets.values()),
            f"max |w| = {max((abs(v) for v in raw_targets.values()), default=0.0):.4f}",
        )
        inactive = [s for s in book if not s.is_active(state_now)]
        flat_ok = all(
            all(sig.target_weight == 0.0 for sig in s.flat_signals(context))
            for s in inactive
        )
        self._check(
            stage,
            "inactive strategies emit flat (0.0) targets",
            flat_ok,
            f"state={state_now}, inactive={[s.name for s in inactive]}",
        )

        # --- Kelly / risk parity / inverse-vol maths ------------------- #
        wins = 0.6
        avg_win = 150.0
        avg_loss = 100.0
        kelly = kelly_fraction(wins, avg_win, avg_loss)
        expected_kelly = wins - (1.0 - wins) / (avg_win / avg_loss)
        self._check(
            stage,
            "Kelly fraction == p - (1-p)/(W/L)",
            abs(kelly - expected_kelly) < 1e-12,
            f"kelly={kelly:.10f} expected={expected_kelly:.10f}",
        )

        covariance = pd.DataFrame(
            np.diag([0.04, 0.09, 0.16]),
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        inverse_vol = inverse_volatility_weights(covariance)
        self._check(
            stage,
            "inverse-volatility weights sum to 1",
            abs(float(np.sum(inverse_vol)) - 1.0) < 1e-9,
            f"w={np.round(inverse_vol, 6).tolist()} for vols 0.20/0.30/0.40",
        )
        parity = risk_parity_weights(covariance)
        contributions = risk_contributions(parity, covariance)
        spread = float(np.max(contributions) - np.min(contributions))
        self._check(
            stage,
            "risk-parity equalises risk contributions",
            spread < 1e-9 and abs(float(np.sum(parity)) - 1.0) < 1e-9,
            f"w={np.round(parity, 6).tolist()} rc spread={spread:.3e}",
        )

        # --- ATR stop sizing ------------------------------------------- #
        risk_manager = RiskManager()
        atr_value = 12.5
        # (a) Uncapped branch: price is low enough that the 50%-notional sanity
        #     bound does not bind, so the raw ATR formula must hold exactly.
        size = risk_manager.atr_position_size(
            equity=100_000.0,
            price=100.0,
            atr=atr_value,
            risk_per_unit_pct=0.01,
            stop_multiple=2.5,
            contract_size=1.0,
        )
        expected_size = (100_000.0 * 0.01) / (2.5 * atr_value)
        self._check(
            stage,
            "ATR sizing == (equity x risk%) / (ATR x stop multiple)",
            abs(size - expected_size) < 1e-9,
            f"units={size:.8f} expected={expected_size:.8f} "
            f"(risk 1,000 of 100k equity @ 2.5 x 12.5 ATR)",
        )
        # (b) Capped branch: the 50%-of-equity notional bound must win.
        capped_size = risk_manager.atr_position_size(
            equity=100_000.0,
            price=2_500.0,
            atr=atr_value,
            risk_per_unit_pct=0.01,
            stop_multiple=2.5,
            contract_size=1.0,
        )
        self._check(
            stage,
            "ATR sizing honours the 50%-notional sanity cap",
            abs(capped_size - 20.0) < 1e-9 and capped_size < expected_size,
            f"units={capped_size:.8f} <= 50,000/2,500 = 20.0 (raw {expected_size:.1f})",
        )

        # --- End-to-end sizing through SizingEngine -------------------- #
        engine = SizingEngine()
        sized = engine.size({"XAUUSD": 0.5, "EURUSD": -0.3}, regime_state=1)
        self._check(
            stage,
            "SizingEngine returns finite, sign-preserving weights",
            all(np.isfinite(v) for v in sized.values())
            and np.sign(sized["XAUUSD"]) == np.sign(0.5)
            and np.sign(sized["EURUSD"]) == np.sign(-0.3),
            f"{ {k: round(v, 6) for k, v in sized.items()} }",
        )
        shared["portfolio"] = portfolio

    # ------------------------------------------------------------------ #
    # Stage 4 — MT5 broker connection / simulated fallback
    # ------------------------------------------------------------------ #
    def _stage_broker(self, shared: Dict[str, Any]) -> None:
        """Attempt a live MT5 connection, else fall back to the simulated broker.

        Args:
            shared: Cross-stage state; writes ``broker`` and ``router``.
        """
        stage = "broker"
        from quant_system.execution.brokers.base import (
            Order,
            OrderSide,
            OrderStatus,
            OrderType,
        )
        from quant_system.execution.brokers.mt5_broker import MT5Broker
        from quant_system.execution.brokers.router import OrderRouter
        from quant_system.execution.brokers.simulated_broker import SimulatedBroker
        from quant_system.execution.portfolio import Portfolio

        portfolio = Portfolio(
            initial_capital=cfg.DEFAULT_SETTINGS.sizing.initial_capital,
            demo_config=cfg.DEFAULT_SETTINGS.demo,
        )
        shared["broker_portfolio"] = portfolio

        # --- Try the real terminal first ------------------------------- #
        mt5_note = "not attempted"
        broker: Any = None
        try:
            candidate = MT5Broker()
            connected = bool(candidate.connect())
            mt5_note = f"connect()={connected}"
            if connected:
                broker = candidate
        except Exception as exc:  # pragma: no cover - depends on the host
            mt5_note = f"{type(exc).__name__}: {exc}"

        if broker is None:
            broker = SimulatedBroker(portfolio=portfolio)
            self._record(
                stage,
                "MT5 terminal unreachable -> simulated fallback engaged",
                True,
                f"mt5: {mt5_note}; using '{broker.name}' adapter",
            )
        else:  # pragma: no cover - only on a host with MT5 installed
            self._record(stage, "MT5 terminal connected", True, mt5_note)
        shared["broker"] = broker

        broker.connect()
        connected_flag = bool(getattr(broker, "is_connected", True))
        self._check(
            stage,
            f"broker '{broker.name}' connected",
            connected_flag,
            f"is_connected={connected_flag}",
        )
        account = broker.get_account()
        self._check(
            stage,
            "account query returns a non-negative balance/equity",
            account is not None and account.equity >= 0,
            f"equity={account.equity:,.2f}" if account is not None else "no account",
        )

        # --- Order lifecycle ------------------------------------------- #
        prices = {"XAUUSD": 2_650.0, "EURUSD": 1.0850}
        broker.update_prices(prices) if hasattr(broker, "update_prices") else None
        router = OrderRouter(broker)
        shared["router"] = router

        orders, reports = router.execute_targets(
            portfolio, {"XAUUSD": 0.10, "EURUSD": -0.05}, prices
        )
        self._check(
            stage,
            "router builds one order per target",
            len(orders) == 2 and len(orders) == len(reports),
            f"{len(orders)} orders / {len(reports)} reports",
        )
        by_symbol = {o.symbol: o for o in orders}
        self._check(
            stage,
            "order sides follow the sign of the target",
            by_symbol.get("XAUUSD") is not None
            and by_symbol["XAUUSD"].side is OrderSide.BUY
            and by_symbol.get("EURUSD") is not None
            and by_symbol["EURUSD"].side is OrderSide.SELL,
            f"{[(o.symbol, o.side.value) for o in orders]}",
        )
        filled = [r for r in reports if r.status is OrderStatus.FILLED]
        self._check(
            stage,
            "submitted orders reach FILLED (or are cleanly rejected)",
            all(r.status is OrderStatus.FILLED for r in reports),
            f"statuses={[r.status.value for r in reports]}, filled={len(filled)}",
        )
        self._check(
            stage,
            "fills report a positive price and quantity",
            all(r.average_fill_price > 0 and r.filled_quantity > 0 for r in filled),
            f"{[(r.broker_order_id, round(r.average_fill_price, 4), round(r.filled_quantity, 4)) for r in filled]}",
        )
        self._check(
            stage,
            "fills are booked into the portfolio",
            len(portfolio.fills) >= len(filled),
            f"portfolio fills={len(portfolio.fills)}",
        )

        # A deliberately invalid order must be refused, not silently filled.
        try:
            bad_order = Order(
                symbol="XAUUSD",
                side=OrderSide.BUY,
                quantity=0.0,
                order_type=OrderType.MARKET,
            )
            self._check(
                stage,
                "invalid (zero-quantity) order is rejected by validation",
                False,
                f"Order constructed with quantity=0: {bad_order}",
            )
        except ValueError as exc:
            self._check(
                stage,
                "invalid (zero-quantity) order is rejected by validation",
                True,
                str(exc),
            )

        # --- Demo-mode daily cap --------------------------------------- #
        unlimited = replace(cfg.DEFAULT_SETTINGS.demo, enabled=True, unlimited_trades=True)
        demo_book = Portfolio(initial_capital=100_000.0, demo_config=unlimited)
        capped = replace(cfg.DEFAULT_SETTINGS.demo, enabled=False)
        capped_book = Portfolio(
            initial_capital=100_000.0,
            demo_config=capped,
            risk_config=replace(cfg.DEFAULT_SETTINGS.risk, max_daily_trades=3),
        )
        stamp = pd.Timestamp(index_last := pd.Timestamp("2024-05-01 10:00"))
        self._check(
            stage,
            "demo mode reports an unlimited daily cap",
            demo_book.daily_trade_limit == -1 and demo_book.unlimited_demo_trades,
            f"limit={demo_book.daily_trade_limit} unlimited={demo_book.unlimited_demo_trades}",
        )
        self._check(
            stage,
            "live mode enforces max_daily_trades",
            capped_book.daily_trade_limit == 3,
            f"limit={capped_book.daily_trade_limit}",
        )
        from quant_system.execution.portfolio import Fill

        demo_results = []
        for symbol in ("XAUUSD", "XAGUSD", "EURUSD", "USDCHF", "USDJPY"):
            allowed, _ = demo_book.can_open_trade(stamp)
            demo_results.append(allowed)
            if allowed:
                demo_book.apply_fill(Fill(stamp, symbol, 1.0, 100.0))
        capped_results = []
        for symbol in ("XAUUSD", "XAGUSD", "EURUSD", "USDCHF", "USDJPY"):
            allowed, _ = capped_book.can_open_trade(stamp)
            capped_results.append(allowed)
            if allowed:
                capped_book.apply_fill(Fill(stamp, symbol, 1.0, 100.0))
        self._check(
            stage,
            "demo mode opens 5/5 positions past a 3/day cap",
            all(demo_results),
            f"allowed={demo_results}",
        )
        self._check(
            stage,
            "live mode blocks positions 4 and 5 at a 3/day cap",
            capped_results == [True, True, True, False, False],
            f"allowed={capped_results}",
        )

    # ------------------------------------------------------------------ #
    # Stage 5 — Trade notifier dispatch & audio queue
    # ------------------------------------------------------------------ #
    def _stage_notifier(self, shared: Dict[str, Any]) -> None:
        """Verify fill-only gating, toast delivery and the audio worker thread.

        Args:
            shared: Cross-stage state (unused; the stage is self-contained).
        """
        stage = "notifier"
        from quant_system.execution.brokers.base import (
            Order,
            OrderReport,
            OrderSide,
            OrderStatus,
            OrderType,
        )
        from quant_system.utils.notifier import (
            NotifierEngine,
            resolve_toast_backend,
        )

        workdir = Path(tempfile.mkdtemp(prefix="quant_verify_"))
        notifier_config = replace(
            cfg.DEFAULT_SETTINGS.notifier,
            enabled=True,
            toast_enabled=True,
            toast_backend="none",
            toast_duration_ms=600,
            voice_enabled=bool(self.voice_enabled),
            voice_mode="file",
            voice_output_dir=workdir,
            queue_maxsize=64,
            dedupe=True,
        )
        notifier = NotifierEngine(config=notifier_config, regime_provider=lambda: 1)
        started = notifier.start()
        self._check(stage, "notifier engine started", bool(started))

        def make_order(symbol: str, side: OrderSide, volume: float) -> Order:
            """Build a market order for ``symbol``.

            Args:
                symbol: Instrument symbol.
                side: Buy or sell.
                volume: Absolute units.

            Returns:
                The constructed :class:`Order`.
            """
            return Order(
                symbol=symbol,
                side=side,
                quantity=volume,
                order_type=OrderType.MARKET,
                price=2_650.50,
                strategy="AdaptiveMomentum",
                tag="verify",
            )

        # --- Fill gating ----------------------------------------------- #
        statuses = (
            (OrderStatus.PENDING, False, "unfilled confirmation"),
            (OrderStatus.ACCEPTED, False, "acknowledgement only"),
            (OrderStatus.PARTIALLY_FILLED, False, "partial fill"),
            (OrderStatus.REJECTED, False, "venue rejection"),
            (OrderStatus.CANCELLED, False, "cancellation"),
            (OrderStatus.FILLED, True, "confirmed fill"),
        )
        gate_ok = True
        gate_detail: List[str] = []
        for status, expected, label in statuses:
            report = OrderReport(
                client_order_id=f"verify-{status.value}",
                broker_order_id=f"brk-{status.value}",
                status=status,
                filled_quantity=0.5,
                average_fill_price=2_650.50,
            )
            result = notifier.is_trade_fill(report)
            gate_ok = gate_ok and result == expected
            gate_detail.append(f"{status.value}={result}")
        self._check(
            stage,
            "only FILLED reports pass the fill gate",
            gate_ok,
            "; ".join(gate_detail),
        )

        # --- Dispatch a real fill -------------------------------------- #
        buy_order = make_order("XAUUSD", OrderSide.BUY, 0.50)
        buy_report = OrderReport(
            client_order_id=buy_order.client_order_id,
            broker_order_id="brk-fill-1",
            status=OrderStatus.FILLED,
            filled_quantity=0.50,
            average_fill_price=2_650.50,
        )
        dispatched = notifier.notify_fill(
            buy_report,
            buy_order,
            regime_state=1,
            stop_loss=2_640.00,
            take_profit=2_680.00,
        )
        self._check(stage, "confirmed fill is announced", bool(dispatched))

        # Duplicate suppression for the same broker_order_id.
        duplicate = notifier.notify_fill(
            buy_report, buy_order, regime_state=1, stop_loss=2_640.00
        )
        self._check(
            stage, "duplicate broker_order_id is suppressed", not bool(duplicate)
        )

        # Non-fills must not be announced even through the public API.
        pending_report = OrderReport(
            client_order_id="verify-pending-2",
            broker_order_id="brk-pending-2",
            status=OrderStatus.PENDING,
            filled_quantity=0.0,
        )
        pending_announced = notifier.notify_fill(
            pending_report, make_order("EURUSD", OrderSide.BUY, 1.25), regime_state=0
        )
        self._check(
            stage, "pending report is not announced", not bool(pending_announced)
        )

        events = notifier.events
        self._check(
            stage,
            "event log holds exactly the announced fills",
            len(events) == 1,
            f"{len(events)} event(s): {[e.action + ' ' + e.symbol for e in events]}",
        )
        if events:
            event = events[0]
            title = event.toast_title()
            body = event.toast_body()
            speech = event.speech_text()
            self._check(
                stage,
                "toast payload carries [ACTION] Symbol | Vol | Price | SL | TP | State",
                "XAUUSD" in title
                and "BUY" in title.upper()
                and "2,650.50" in body
                and "2,640.00" in body
                and "2,680.00" in body
                and "State 1" in body,
                f"title='{title}' | body='{body}'",
            )
            self._check(
                stage,
                "speech text is concise and speaks prices digit-wise",
                "Bought" in speech
                and "Gold" in speech
                and "2650 point 50" in speech
                and "State 1" in speech,
                f"speech='{speech}'",
            )

        # --- Non-blocking dispatch ------------------------------------- #
        latencies: List[float] = []
        for i in range(12):
            order = make_order("USDJPY", OrderSide.SELL, 1.0 + i / 100.0)
            report = OrderReport(
                client_order_id=order.client_order_id,
                broker_order_id=f"brk-burst-{i}",
                status=OrderStatus.FILLED,
                filled_quantity=order.quantity,
                average_fill_price=148.50 + i / 10.0,
            )
            tick = time.perf_counter()
            notifier.notify_fill(report, order, regime_state=1)
            latencies.append((time.perf_counter() - tick) * 1000.0)
        worst = max(latencies) if latencies else 0.0
        self._check(
            stage,
            "notify_fill never blocks the caller (< 25 ms worst case)",
            worst < 25.0,
            f"12 dispatches, worst={worst:.3f} ms, mean={np.mean(latencies):.3f} ms",
        )

        # --- Audio queue execution ------------------------------------- #
        notifier.stop()
        stats = notifier.stats()
        voice_stats = stats.get("voice", {})
        if self.voice_enabled:
            self._check(
                stage,
                "voice worker ran on a dedicated background thread",
                voice_stats.get("thread_started") is True
                and voice_stats.get("thread_stopped") is True
                and voice_stats.get("queue_depth") == 0,
                f"started={voice_stats.get('thread_started')} "
                f"stopped={voice_stats.get('thread_stopped')} "
                f"queue_depth={voice_stats.get('queue_depth')} "
                f"mode={voice_stats.get('mode')}",
            )
            self._check(
                stage,
                "every queued utterance was synthesised, none dropped",
                int(voice_stats.get("spoken", -1)) == 13
                and int(voice_stats.get("failed", -1)) == 0
                and int(voice_stats.get("dropped", -1)) == 0
                and len(voice_stats.get("files", [])) == 13,
                f"spoken={voice_stats.get('spoken')} failed={voice_stats.get('failed')} "
                f"skipped={voice_stats.get('skipped')} dropped={voice_stats.get('dropped')} "
                f"files={len(voice_stats.get('files', []))}",
            )
            wave_files = sorted(workdir.glob("*.wav"))
            sizes = [f.stat().st_size for f in wave_files]
            self._check(
                stage,
                "audio worker wrote one non-empty WAV per utterance",
                len(wave_files) == 13 and all(size > 1_000 for size in sizes),
                f"{len(wave_files)} WAVs, min={min(sizes) if sizes else 0} bytes, "
                f"total={sum(sizes) if sizes else 0} bytes",
            )
            if wave_files:
                header = wave_files[0].read_bytes()[:4]
                self._check(
                    stage,
                    "WAV clips carry a valid RIFF header",
                    header == b"RIFF",
                    f"{wave_files[0].name} magic={header!r}",
                )
                for path in wave_files[:3]:
                    self.report.artifacts.append(str(path))
        else:  # pragma: no cover - only when voice is disabled
            self._record(stage, "voice synthesis skipped (disabled)", True, "")

        all_events = notifier.events
        self._check(
            stage,
            "notifier statistics reflect delivered vs filtered counts",
            int(stats.get("toasts_delivered", 0)) == len(all_events) == 13
            and int(stats.get("announced", 0)) == len(all_events),
            f"delivered={stats.get('toasts_delivered')} "
            f"announced={stats.get('announced')} events={len(all_events)}",
        )
        self._check(
            stage,
            "filtered counter records non-fills and the duplicate",
            int(stats.get("filtered", {}).get("not_a_confirmed_fill", 0)) >= 1
            and int(stats.get("filtered", {}).get("duplicate_order_id", 0)) == 1,
            f"filtered={stats.get('filtered')}",
        )
        self._check(
            stage,
            "stop() is idempotent and shuts the engine down cleanly",
            notifier.stop() is None and notifier.stop() is None,
            "second and third stop() calls returned without error",
        )

        # --- Toast backend resolution (headless-safe) ------------------ #
        for requested in ("none", "tk", "pyqt"):
            backend, note = resolve_toast_backend(requested, notifier_config)
            self._check(
                stage,
                f"toast backend '{requested}' resolves without raising",
                backend is not None,
                f"backend={type(backend).__name__} note={note}",
            )
            try:
                backend.stop()
            except Exception as exc:  # pragma: no cover - teardown best effort
                self._record(
                    stage,
                    f"toast backend '{requested}' stops cleanly",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )

        shutil.rmtree(workdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Pytest entry points
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def verification() -> VerificationReport:
    """Run the full verification once per module and share the report.

    Returns:
        The :class:`VerificationReport` produced by :class:`SystemVerifier`.
    """
    return SystemVerifier().run()


@pytest.mark.parametrize("stage,title", STAGES)
def test_stage_passes(verification: VerificationReport, stage: str, title: str) -> None:
    """Assert that every check in a stage passed.

    Args:
        verification: Shared verification report.
        stage: Stage key.
        title: Human-readable stage title.
    """
    checks = verification.for_stage(stage)
    assert checks, f"stage '{stage}' produced no checks"
    failed = [c for c in checks if not c.passed]
    detail = "\n".join(f"  - {c.name}: {c.detail}" for c in failed)
    assert not failed, f"{title} failed {len(failed)}/{len(checks)} checks:\n{detail}"


def test_no_warnings_emitted(verification: VerificationReport) -> None:
    """Assert the pipeline emitted no warnings at all.

    Args:
        verification: Shared verification report.
    """
    noisy = [c for c in verification.results if c.name.startswith("no warning:")]
    detail = "\n".join(f"  - {c.detail}" for c in noisy)
    assert not noisy, f"{len(noisy)} warning(s) emitted:\n{detail}"


def test_full_pipeline_passes(verification: VerificationReport) -> None:
    """Assert the aggregate result and print the console table on failure.

    Args:
        verification: Shared verification report.
    """
    if not verification.passed:
        pytest.fail(
            f"{len(verification.failures)} check(s) failed:\n{verification.render()}"
        )
    print("\n" + verification.render())


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the verification suite from the command line.

    Args:
        argv: Optional argument vector (currently unused; reserved for flags).

    Returns:
        ``0`` when every check passes, ``1`` otherwise.
    """
    del argv  # reserved for future flags
    report = SystemVerifier().run()
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
