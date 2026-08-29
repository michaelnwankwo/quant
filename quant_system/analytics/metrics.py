"""Performance metrics, expectancy and regime-conditional attribution.

All ratios are computed with a risk-free rate of **0 %** as specified.  Return
inputs are interpreted as *periodic* simple returns; annualisation uses
``periods_per_year`` (252 for daily bars).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from quant_system.config import settings as cfg

logger = logging.getLogger(__name__)

EPS: float = 1e-12


# --------------------------------------------------------------------------- #
# Core ratios
# --------------------------------------------------------------------------- #
def annualisation_factor(periods_per_year: float = float(cfg.TRADING_DAYS_PER_YEAR)) -> float:
    """Return the annualisation factor used by the ratio functions.

    Args:
        periods_per_year: Number of periods per year.

    Returns:
        The factor (default ``252``).
    """
    return float(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = cfg.RISK_FREE_RATE,
    periods_per_year: float = float(cfg.TRADING_DAYS_PER_YEAR),
) -> float:
    """Annualised Sharpe ratio with ``Rf = 0``.

    ``S = (mean(r) - rf_period) / std(r) * sqrt(P)``

    Args:
        returns: Periodic returns.
        risk_free_rate: Annualised risk-free rate.
        periods_per_year: Periods per year.

    Returns:
        The Sharpe ratio (``0.0`` when volatility is degenerate).
    """
    series = pd.Series(returns).dropna()
    if len(series) < 2:
        return 0.0
    std = float(series.std(ddof=1))
    if std <= EPS:
        return 0.0
    excess = float(series.mean()) - risk_free_rate / periods_per_year
    return float(excess / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = cfg.RISK_FREE_RATE,
    periods_per_year: float = float(cfg.TRADING_DAYS_PER_YEAR),
) -> float:
    """Annualised Sortino ratio (downside deviation in the denominator).

    Args:
        returns: Periodic returns.
        risk_free_rate: Annualised risk-free rate.
        periods_per_year: Periods per year.

    Returns:
        The Sortino ratio (``0.0`` when downside deviation is degenerate).
    """
    series = pd.Series(returns).dropna()
    if len(series) < 2:
        return 0.0
    target = risk_free_rate / periods_per_year
    downside = series[series < target] - target
    if len(downside) == 0:
        return 0.0
    downside_deviation = float(np.sqrt((downside**2).mean()))
    if downside_deviation <= EPS:
        return 0.0
    excess = float(series.mean()) - target
    return float(excess / downside_deviation * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running equity peak.

    Args:
        equity: Equity curve.

    Returns:
        Drawdown series in ``[0, 1]`` (positive values).
    """
    curve = pd.Series(equity).dropna()
    if curve.empty:
        return pd.Series(dtype=float, name="drawdown")
    running_max = curve.cummax()
    return (1.0 - curve / running_max.replace(0.0, np.nan)).fillna(0.0).rename("drawdown")


@dataclass(frozen=True)
class DrawdownInfo:
    """Peak-to-trough analysis of an equity curve.

    Attributes:
        max_drawdown: Maximum peak-to-trough decline as a positive fraction.
        peak_timestamp: Timestamp of the equity peak preceding the trough.
        trough_timestamp: Timestamp of the equity trough.
        recovery_timestamp: Timestamp at which the peak was recovered
            (``None`` if not recovered).
        duration_bars: Bars from peak to trough.
        recovery_bars: Bars from trough to recovery (``None`` if unrecovered).
        total_drawdown_bars: Bars spent below the running peak.
    """

    max_drawdown: float
    peak_timestamp: Optional[pd.Timestamp]
    trough_timestamp: Optional[pd.Timestamp]
    recovery_timestamp: Optional[pd.Timestamp]
    duration_bars: int
    recovery_bars: Optional[int]
    total_drawdown_bars: int


def max_drawdown(equity: pd.Series) -> DrawdownInfo:
    """Compute peak-to-trough drawdown, duration and recovery.

    Args:
        equity: Equity curve indexed by timestamp.

    Returns:
        A :class:`DrawdownInfo` record.
    """
    curve = pd.Series(equity).dropna()
    if curve.empty:
        return DrawdownInfo(0.0, None, None, None, 0, None, 0)
    values = curve.to_numpy(dtype=float)
    running_max = np.maximum.accumulate(values)
    drawdowns = 1.0 - values / np.where(running_max > 0, running_max, np.nan)
    drawdowns = np.nan_to_num(drawdowns, nan=0.0)

    trough_index = int(np.argmax(drawdowns))
    max_dd = float(drawdowns[trough_index])
    if max_dd <= 0:
        return DrawdownInfo(0.0, None, None, None, 0, None, 0)

    peak_index = int(np.argmax(values[: trough_index + 1])) if trough_index > 0 else 0
    peak_value = values[peak_index]
    recovery_index: Optional[int] = None
    for index in range(trough_index, len(values)):
        if values[index] >= peak_value:
            recovery_index = index
            break

    index = curve.index
    return DrawdownInfo(
        max_drawdown=max_dd,
        peak_timestamp=pd.Timestamp(index[peak_index]),
        trough_timestamp=pd.Timestamp(index[trough_index]),
        recovery_timestamp=(
            pd.Timestamp(index[recovery_index]) if recovery_index is not None else None
        ),
        duration_bars=int(trough_index - peak_index),
        recovery_bars=(
            int(recovery_index - trough_index) if recovery_index is not None else None
        ),
        total_drawdown_bars=int((drawdowns > 0).sum()),
    )


def calmar_ratio(
    equity: pd.Series,
    periods_per_year: float = float(cfg.TRADING_DAYS_PER_YEAR),
) -> float:
    """Annualised return divided by the maximum drawdown.

    Args:
        equity: Equity curve.
        periods_per_year: Periods per year.

    Returns:
        The Calmar ratio (``0.0`` when the drawdown is zero).
    """
    return float(cagr(equity, periods_per_year) / max_drawdown(equity).max_drawdown) if (
        max_drawdown(equity).max_drawdown > EPS
    ) else 0.0


def cagr(
    equity: pd.Series,
    periods_per_year: float = float(cfg.TRADING_DAYS_PER_YEAR),
) -> float:
    """Compound annual growth rate of an equity curve.

    Args:
        equity: Equity curve.
        periods_per_year: Periods per year.

    Returns:
        The annualised compound growth rate.
    """
    curve = pd.Series(equity).dropna()
    if len(curve) < 2 or curve.iloc[0] <= 0:
        return 0.0
    years = (len(curve) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((curve.iloc[-1] / curve.iloc[0]) ** (1.0 / years) - 1.0)


def annualised_volatility(
    returns: pd.Series,
    periods_per_year: float = float(cfg.TRADING_DAYS_PER_YEAR),
) -> float:
    """Annualised standard deviation of returns.

    Args:
        returns: Periodic returns.
        periods_per_year: Periods per year.

    Returns:
        Annualised volatility.
    """
    series = pd.Series(returns).dropna()
    if len(series) < 2:
        return 0.0
    return float(series.std(ddof=1) * np.sqrt(periods_per_year))


def value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """Historical Value at Risk at the given tail probability.

    Args:
        returns: Periodic returns.
        level: Tail probability (``0.05`` = 95 % VaR).

    Returns:
        The VaR as a positive loss fraction.
    """
    series = pd.Series(returns).dropna()
    if series.empty:
        return 0.0
    return float(-np.quantile(series.to_numpy(dtype=float), level))


def conditional_value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """Historical Conditional VaR (expected shortfall).

    Args:
        returns: Periodic returns.
        level: Tail probability.

    Returns:
        The CVaR as a positive loss fraction.
    """
    series = pd.Series(returns).dropna()
    if series.empty:
        return 0.0
    values = series.to_numpy(dtype=float)
    threshold = np.quantile(values, level)
    tail = values[values <= threshold]
    if tail.size == 0:
        return 0.0
    return float(-tail.mean())


# --------------------------------------------------------------------------- #
# Trade statistics & expectancy
# --------------------------------------------------------------------------- #
def trade_statistics(trades: pd.DataFrame) -> Dict[str, float]:
    """Win/loss statistics and expectancy from a trade log.

    Expectancy follows the specification:
    ``E = (WinRate * AvgWin) - (LossRate * AvgLoss)``

    Args:
        trades: DataFrame produced by :meth:`Portfolio.trade_frame` with a
            ``net_pnl`` column.

    Returns:
        Dictionary with ``num_trades``, ``win_rate``, ``avg_win``, ``avg_loss``,
        ``expectancy``, ``expectancy_pct``, ``profit_factor``, ``largest_win``,
        ``largest_loss``, ``payoff_ratio``, ``avg_bars_held``.
    """
    empty = {
        "num_trades": 0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "expectancy": 0.0,
        "expectancy_pct": 0.0,
        "profit_factor": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "payoff_ratio": 0.0,
        "avg_bars_held": 0.0,
    }
    if trades is None or trades.empty or "net_pnl" not in trades.columns:
        return empty

    pnl = trades["net_pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    n = len(pnl)
    win_rate = len(wins) / n if n else 0.0
    loss_rate = 1.0 - win_rate
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > EPS else float("inf") if gross_profit > 0 else 0.0

    expectancy_pct = (
        float(trades["return_pct"].astype(float).mean())
        if "return_pct" in trades.columns
        else 0.0
    )

    avg_bars = 0.0
    if {"entry_timestamp", "exit_timestamp"}.issubset(trades.columns):
        held = (
            pd.to_datetime(trades["exit_timestamp"]) - pd.to_datetime(trades["entry_timestamp"])
        ).dt.total_seconds()
        avg_bars = float((held / 86400.0).mean()) if len(held) else 0.0

    return {
        "num_trades": int(n),
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": float(expectancy),
        "expectancy_pct": float(expectancy_pct),
        "profit_factor": float(profit_factor),
        "largest_win": float(pnl.max()) if n else 0.0,
        "largest_loss": float(pnl.min()) if n else 0.0,
        "payoff_ratio": float(avg_win / avg_loss) if avg_loss > EPS else 0.0,
        "avg_bars_held": avg_bars,
    }


def expectancy_from_returns(returns: pd.Series) -> float:
    """Expectancy per period expressed as an average return.

    Args:
        returns: Periodic returns.

    Returns:
        The mean periodic return.
    """
    series = pd.Series(returns).dropna()
    return float(series.mean()) if len(series) else 0.0


# --------------------------------------------------------------------------- #
# Aggregated summary
# --------------------------------------------------------------------------- #
def performance_summary(
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    periods_per_year: float = float(cfg.TRADING_DAYS_PER_YEAR),
) -> Dict[str, float]:
    """Compute the full performance dashboard.

    Args:
        equity: Equity curve.
        trades: Optional trade log for expectancy statistics.
        periods_per_year: Periods per year.

    Returns:
        Dictionary of headline metrics (Sharpe, Sortino, Calmar, MaxDD, CAGR,
        volatility, VaR, CVaR, hit rate, trade statistics, final equity).
    """
    curve = pd.Series(equity).dropna()
    if curve.empty:
        return {}
    returns = curve.pct_change().dropna()
    dd_info = max_drawdown(curve)

    summary: Dict[str, float] = {
        "start_equity": float(curve.iloc[0]),
        "final_equity": float(curve.iloc[-1]),
        "total_return_pct": float(curve.iloc[-1] / curve.iloc[0] - 1.0) * 100.0,
        "cagr_pct": cagr(curve, periods_per_year) * 100.0,
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(returns, periods_per_year=periods_per_year),
        "calmar_ratio": calmar_ratio(curve, periods_per_year),
        "max_drawdown_pct": dd_info.max_drawdown * 100.0,
        "max_drawdown_duration_bars": float(dd_info.duration_bars),
        "recovery_bars": float(dd_info.recovery_bars or 0),
        "drawdown_bars": float(dd_info.total_drawdown_bars),
        "annualised_volatility_pct": annualised_volatility(returns, periods_per_year) * 100.0,
        "var_95_pct": value_at_risk(returns, 0.05) * 100.0,
        "cvar_95_pct": conditional_value_at_risk(returns, 0.05) * 100.0,
        "hit_rate_pct": float((returns > 0).mean() * 100.0) if len(returns) else 0.0,
        # A regime-gated book is flat on most days; the active hit rate reports
        # the win rate conditional on actually having exposure.
        "active_hit_rate_pct": (
            float((returns[returns != 0] > 0).mean() * 100.0)
            if len(returns) and (returns != 0).any()
            else 0.0
        ),
        "exposure_pct": float((returns != 0).mean() * 100.0) if len(returns) else 0.0,
        "best_period_pct": float(returns.max() * 100.0) if len(returns) else 0.0,
        "worst_period_pct": float(returns.min() * 100.0) if len(returns) else 0.0,
        "num_periods": float(len(returns)),
    }
    if dd_info.peak_timestamp is not None:
        summary["max_dd_peak"] = dd_info.peak_timestamp  # type: ignore[assignment]
        summary["max_dd_trough"] = dd_info.trough_timestamp  # type: ignore[assignment]
    summary.update(trade_statistics(trades) if trades is not None else {})
    return summary


# --------------------------------------------------------------------------- #
# Regime attribution
# --------------------------------------------------------------------------- #
def regime_breakdown(
    returns: pd.Series,
    states: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    periods_per_year: float = float(cfg.TRADING_DAYS_PER_YEAR),
) -> pd.DataFrame:
    """Performance statistics conditioned on the HMM regime.

    Args:
        returns: Strategy returns indexed by timestamp.
        states: Regime ids aligned to ``returns`` (``0``/``1``/``2``).
        trades: Optional trade log; when supplied, trades are attributed to the
            regime active on their exit timestamp.
        periods_per_year: Periods per year.

    Returns:
        DataFrame indexed by regime id with per-regime counts, Sharpe, Sortino,
        expectancy, max drawdown, hit rate and trade statistics.
    """
    aligned_returns = pd.Series(returns).dropna()
    # Leading bars belong to the HMM warm-up, where no label exists yet: fall
    # back to the benign range-bound state rather than propagating NaN.
    aligned_states = (
        pd.Series(states)
        .reindex(aligned_returns.index)
        .ffill()
        .fillna(cfg.STATE_RANGE_BOUND)
    )
    if aligned_returns.empty:
        return pd.DataFrame()

    rows: List[Dict[str, float]] = []
    combined = pd.concat([aligned_returns.rename("ret"), aligned_states.rename("state")], axis=1)
    combined["state"] = combined["state"].astype(int)

    for state_id in sorted(combined["state"].unique()):
        subset = combined.loc[combined["state"] == state_id, "ret"]
        if subset.empty:
            continue
        equity = (1.0 + subset).cumprod()
        dd_info = max_drawdown(equity)
        regime_trades = None
        if trades is not None and not trades.empty and "exit_timestamp" in trades.columns:
            exit_states = (
                pd.Series(states)
                .reindex(pd.to_datetime(trades["exit_timestamp"]))
                .ffill()
                .fillna(cfg.STATE_RANGE_BOUND)
                .to_numpy()
            )
            mask = exit_states == state_id
            regime_trades = trades.loc[mask]
        stats = trade_statistics(regime_trades) if regime_trades is not None else {}
        rows.append(
            {
                "state": int(state_id),
                "label": cfg.REGIME_LABELS.get(int(state_id), "unknown"),
                "bars": float(len(subset)),
                "share_pct": 100.0 * len(subset) / len(combined),
                "total_return_pct": float(equity.iloc[-1] - 1.0) * 100.0,
                "mean_return_bps": float(subset.mean()) * 10_000.0,
                "sharpe_ratio": sharpe_ratio(subset, periods_per_year=periods_per_year),
                "sortino_ratio": sortino_ratio(subset, periods_per_year=periods_per_year),
                "max_drawdown_pct": dd_info.max_drawdown * 100.0,
                "hit_rate_pct": float((subset > 0).mean()) * 100.0,
                "expectancy": float(stats.get("expectancy", float(subset.mean()))),
                "num_trades": float(stats.get("num_trades", 0)),
                "win_rate_pct": float(stats.get("win_rate", 0.0)) * 100.0,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.set_index("state").sort_index() if not frame.empty else pd.DataFrame()


def regime_transitions(states: pd.Series) -> pd.DataFrame:
    """Empirical regime transition matrix.

    Args:
        states: Regime series.

    Returns:
        DataFrame where ``[i, j]`` is ``P(state_{t+1} = j | state_t = i)``.
    """
    series = pd.Series(states).dropna().astype(int)
    if len(series) < 2:
        return pd.DataFrame()
    pairs = pd.DataFrame({"from": series[:-1].to_numpy(), "to": series[1:].to_numpy()})
    matrix = pd.crosstab(pairs["from"], pairs["to"], normalize="index")
    return matrix.reindex(index=range(cfg.N_REGIMES), columns=range(cfg.N_REGIMES)).fillna(0.0)


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Month-by-month return table (rows = years, columns = months).

    Args:
        equity: Equity curve.

    Returns:
        Pivot table of monthly returns.
    """
    curve = pd.Series(equity).dropna()
    if curve.empty:
        return pd.DataFrame()
    monthly = curve.resample("ME").last().pct_change().dropna()
    frame = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
    return frame.pivot(index="year", columns="month", values="ret")


def format_summary(summary: Mapping[str, float]) -> str:
    """Render a performance summary as an aligned text block.

    Args:
        summary: Mapping produced by :func:`performance_summary`.

    Returns:
        A multi-line, human-readable report.
    """
    lines: List[str] = []
    for key, value in summary.items():
        if isinstance(value, pd.Timestamp):
            lines.append(f"{key:>32s} : {value:%Y-%m-%d}")
        elif isinstance(value, float):
            if np.isinf(value):
                lines.append(f"{key:>32s} : inf")
            else:
                lines.append(f"{key:>32s} : {value:,.4f}")
        else:
            lines.append(f"{key:>32s} : {value}")
    return "\n".join(lines)


__all__: List[str] = [
    "EPS",
    "annualisation_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "drawdown_series",
    "DrawdownInfo",
    "max_drawdown",
    "calmar_ratio",
    "cagr",
    "annualised_volatility",
    "value_at_risk",
    "conditional_value_at_risk",
    "trade_statistics",
    "expectancy_from_returns",
    "performance_summary",
    "regime_breakdown",
    "regime_transitions",
    "monthly_returns",
    "format_summary",
]
