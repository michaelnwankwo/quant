"""Feature engineering and preprocessing utilities.

All transforms in this module are **causal by construction**: every rolling
window uses data up to and including the current bar only, and no transform
peeks at future observations.  Where a statistic is consumed by a strategy as a
*decision input* (as opposed to a label), callers should apply
:func:`lag_features` to shift it by one bar.

Mathematical definitions
------------------------
Log returns
    ``r_t = ln(P_t / P_{t-1})``

Average True Range (Wilder, 14)
    ``TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)``
    seeded with a simple mean over the first ``period`` observations, then
    smoothed with Wilder's RMA: ``ATR_t = (ATR_{t-1} * (p - 1) + TR_t) / p``.

Historical volatility
    ``sigma_t = sqrt(252) * sqrt( 1/(N-1) * sum_i (r_i - rbar)^2 )`` with ``N = 20``.
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from quant_system.config import settings as cfg

logger = logging.getLogger(__name__)

EPS: float = 1e-12

#: Bars per year, keyed by interval, used to annualise volatility.
BARS_PER_YEAR: Dict[str, float] = {
    "1d": 252.0,
    "1h": 252.0 * 24.0,
    "15m": 252.0 * 24.0 * 4.0,
    "5m": 252.0 * 24.0 * 12.0,
    "1m": 252.0 * 24.0 * 60.0,
}


# --------------------------------------------------------------------------- #
# Basic transforms
# --------------------------------------------------------------------------- #
def log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns ``r_t = ln(P_t / P_{t-1})``.

    Args:
        prices: Positive price series.

    Returns:
        Log-return series whose first element is ``NaN``.

    Raises:
        ValueError: If any price is non-positive (log undefined).
    """
    if (prices <= 0).any():
        raise ValueError("Log returns require strictly positive prices.")
    return np.log(prices / prices.shift(1)).rename("ret")


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Compute the true range series.

    Args:
        high: Bar highs.
        low: Bar lows.
        close: Bar closes.

    Returns:
        True-range series (first element equals ``high - low``).
    """
    prev_close = close.shift(1)
    ranges = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1).fillna(high - low).rename("true_range")


def wilder_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Wilder's Average True Range with an SMA seed.

    Args:
        high: Bar highs.
        low: Bar lows.
        close: Bar closes.
        period: Smoothing period (default 14).

    Returns:
        ATR series; the first ``period - 1`` entries are ``NaN``.

    Raises:
        ValueError: If ``period`` is not a positive integer.
    """
    if period <= 0:
        raise ValueError("period must be a positive integer.")
    tr = true_range(high, low, close).to_numpy(dtype=float)
    n = tr.size
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return pd.Series(out, index=close.index, name="atr")
    seed = np.nanmean(tr[:period])
    if not np.isfinite(seed):
        return pd.Series(out, index=close.index, name="atr")
    out[period - 1] = seed
    for i in range(period, n):
        prev = out[i - 1]
        out[i] = (prev * (period - 1) + tr[i]) / period
    return pd.Series(out, index=close.index, name="atr")


def realized_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    interval: str = "1d",
) -> pd.Series:
    """Rolling annualised historical volatility.

    Implements ``sigma_t = sqrt(A) * sqrt(1/(N-1) * sum_i (r_i - rbar)^2)``
    where ``A`` is the annualisation factor (252 for daily bars).

    Args:
        returns: Return series (log returns).
        window: Rolling window ``N``.
        annualize: Multiply by ``sqrt(bars_per_year)``.
        interval: Bar interval used to look up the annualisation factor.

    Returns:
        Rolling volatility series.
    """
    rolling_std = returns.rolling(window=window, min_periods=window).std(ddof=1)
    if annualize:
        factor = float(np.sqrt(BARS_PER_YEAR.get(interval, 252.0)))
        rolling_std = rolling_std * factor
    return rolling_std.rename("sigma")


# --------------------------------------------------------------------------- #
# Feature frames
# --------------------------------------------------------------------------- #
def build_feature_frame(
    ohlcv: pd.DataFrame,
    atr_period: int = 14,
    vol_window: int = 20,
    interval: str = "1d",
    normalize_atr: bool = True,
) -> pd.DataFrame:
    """Build the per-symbol HMM feature frame ``[ret, atr, sigma]``.

    Args:
        ohlcv: DataFrame with ``open/high/low/close/volume`` columns.
        atr_period: Wilder ATR lookback.
        vol_window: Realised-volatility lookback.
        interval: Bar interval used for annualisation.
        normalize_atr: Divide ATR by close so the feature is dimensionless and
            comparable across assets and price levels.

    Returns:
        DataFrame with columns ``ret``, ``atr``, ``sigma`` (plus ``atr_raw``)
        and NaNs dropped.
    """
    close = ohlcv["close"].astype(float)
    ret = log_returns(close)
    atr_raw = wilder_atr(ohlcv["high"], ohlcv["low"], close, period=atr_period)
    sigma = realized_volatility(ret, window=vol_window, interval=interval)
    atr = (atr_raw / close) if normalize_atr else atr_raw
    frame = pd.DataFrame({"ret": ret, "atr": atr, "sigma": sigma, "atr_raw": atr_raw})
    return frame.dropna()


def build_market_features(
    data: Mapping[str, pd.DataFrame],
    atr_period: int = 14,
    vol_window: int = 20,
    interval: str = "1d",
) -> pd.DataFrame:
    """Build the cross-sectional *market composite* feature frame.

    The switchboard is a single market-wide model, so the three features are
    aggregated across the universe:

    ``ret``
        Equal-weighted mean of per-symbol log returns.
    ``atr``
        Equal-weighted mean of per-symbol ATR expressed as a fraction of price
        (dimensionless, so gold and FX crosses are directly comparable).
    ``sigma``
        Equal-weighted mean of per-symbol annualised realised volatility.

    Args:
        data: Mapping of symbol -> OHLCV frame.
        atr_period: Wilder ATR lookback.
        vol_window: Realised-volatility lookback.
        interval: Bar interval used for annualisation.

    Returns:
        DataFrame with columns ``ret``, ``atr``, ``sigma``.

    Raises:
        ValueError: If ``data`` is empty.
    """
    if not data:
        raise ValueError("Cannot build market features from an empty universe.")
    rets: Dict[str, pd.Series] = {}
    atrs: Dict[str, pd.Series] = {}
    sigmas: Dict[str, pd.Series] = {}
    for symbol, ohlcv in data.items():
        feats = build_feature_frame(
            ohlcv, atr_period=atr_period, vol_window=vol_window, interval=interval
        )
        rets[symbol] = feats["ret"]
        atrs[symbol] = feats["atr"]
        sigmas[symbol] = feats["sigma"]
    frame = pd.DataFrame(
        {
            "ret": pd.DataFrame(rets).mean(axis=1),
            "atr": pd.DataFrame(atrs).mean(axis=1),
            "sigma": pd.DataFrame(sigmas).mean(axis=1),
        }
    )
    return frame.dropna()


def standardize_features(
    features: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardise a feature frame with :class:`StandardScaler`.

    Args:
        features: Frame whose columns are the model features.
        scaler: Pre-fitted scaler; a new one is fitted when ``None``.

    Returns:
        Tuple of ``(scaled_frame, fitted_scaler)``. The returned frame preserves
        the original index and column labels.

    Raises:
        ValueError: If ``features`` is empty or contains non-finite values.
    """
    if features.empty:
        raise ValueError("Cannot standardise an empty feature frame.")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Feature frame contains non-finite values.")
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(values)
    scaled = scaler.transform(values)
    return (
        pd.DataFrame(scaled, index=features.index, columns=features.columns),
        scaler,
    )


def lag_features(features: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Shift a feature frame forward in time to eliminate look-ahead.

    Args:
        features: Feature frame computed from contemporaneous prices.
        periods: Number of bars to shift.

    Returns:
        Lagged feature frame (first ``periods`` rows become ``NaN``).
    """
    return features.shift(periods)


# --------------------------------------------------------------------------- #
# Spreads, z-scores, cointegration
# --------------------------------------------------------------------------- #
def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
    clip: Optional[float] = 20.0,
) -> pd.Series:
    """Rolling z-score with a zero-variance guard.

    Args:
        series: Input series (typically a spread).
        window: Rolling window.
        min_periods: Minimum observations; defaults to ``window``.
        clip: Absolute cap applied to the output to neutralise degenerate
            spikes produced by an (almost) zero rolling standard deviation.
            Pass ``None`` to disable clipping.

    Returns:
        Z-score series. Where the rolling standard deviation is below
        ``EPS`` the z-score is defined as ``0.0`` (the spread is exactly at its
        rolling mean) rather than ``inf``/``NaN``.

    Notes:
        A zero (or numerically negligible) rolling variance usually indicates a
        flat/patched price series or a perfectly hedged spread; returning 0.0
        keeps downstream threshold logic finite and safe.
    """
    min_periods = min_periods or window
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std(ddof=1)
    safe_std = std.where(std.abs() > EPS, other=np.nan)
    z = (series - mean) / safe_std
    z = z.where(std.abs() > EPS, other=0.0)
    if clip is not None:
        z = z.clip(lower=-clip, upper=clip)
    return z.rename("zscore")


def ols_hedge_ratio(
    y: Union[pd.Series, np.ndarray],
    x: Union[pd.Series, np.ndarray],
) -> Tuple[float, float, float]:
    """Static OLS hedge ratio for ``y = alpha + beta * x + eps``.

    Args:
        y: Dependent series (e.g. XAUUSD price).
        x: Independent series (e.g. XAGUSD price).

    Returns:
        Tuple ``(beta, alpha, r_squared)``. ``(nan, nan, nan)`` is returned when
        the regression is degenerate (zero-variance regressor or too few points).
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    x_arr = np.asarray(x, dtype=float).ravel()
    mask = np.isfinite(y_arr) & np.isfinite(x_arr)
    y_arr, x_arr = y_arr[mask], x_arr[mask]
    if y_arr.size < 3 or np.ptp(x_arr) < EPS:
        return float("nan"), float("nan"), float("nan")
    x_mean = x_arr.mean()
    y_mean = y_arr.mean()
    sxx = float(np.sum((x_arr - x_mean) ** 2))
    if sxx < EPS:
        return float("nan"), float("nan"), float("nan")
    beta = float(np.sum((x_arr - x_mean) * (y_arr - y_mean)) / sxx)
    alpha = float(y_mean - beta * x_mean)
    resid = y_arr - (alpha + beta * x_arr)
    ss_tot = float(np.sum((y_arr - y_mean) ** 2))
    ss_res = float(np.sum(resid**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > EPS else float("nan")
    return beta, alpha, r_squared


def rolling_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int,
    max_abs_beta: Optional[float] = None,
) -> pd.Series:
    """Rolling OLS hedge ratio ``beta_t`` for the spread ``y - beta * x``.

    Computed in closed form as ``Cov(y, x) / Var(x)`` over the rolling window,
    which is the exact OLS slope for a simple regression with intercept.

    Args:
        y: Dependent price series.
        x: Independent price series.
        window: Rolling regression window.
        max_abs_beta: Optional absolute cap. ``None`` (the default) leaves the
            estimate unclipped - callers that need a bound should prefer a
            *scale-relative* cap (see
            :attr:`~quant_system.config.settings.StatArbConfig.max_abs_beta_ratio`)
            because the economically correct beta spans orders of magnitude
            across asset pairs (XAU/XAG ~ 85, EURUSD/USDCHF ~ -1).

    Returns:
        Rolling beta series (forward-filled for stability).
    """
    cov = y.rolling(window=window, min_periods=window).cov(x)
    var = x.rolling(window=window, min_periods=window).var(ddof=1)
    beta = cov / var.where(var.abs() > EPS, other=np.nan)
    beta = beta.replace([np.inf, -np.inf], np.nan)
    if max_abs_beta is not None:
        beta = beta.clip(lower=-abs(max_abs_beta), upper=abs(max_abs_beta))
    return beta.ffill().rename("beta")


def compute_spread(a: pd.Series, b: pd.Series, beta: pd.Series) -> pd.Series:
    """Compute the cointegrating spread ``S_t = A_t - beta_t * B_t``.

    Args:
        a: First leg (e.g. XAUUSD).
        b: Second leg (e.g. XAGUSD).
        beta: Hedge-ratio series aligned to the price index.

    Returns:
        Spread series.
    """
    return (a - beta * b).rename("spread")


def engle_granger_test(
    a: pd.Series,
    b: pd.Series,
    trend: str = "c",
    maxlag: Optional[int] = None,
) -> Tuple[float, float]:
    """Two-step Engle-Granger cointegration test on a price pair.

    Args:
        a: First price series.
        b: Second price series.
        trend: Deterministic term in the cointegrating regression and ADF test
            (``"c"`` constant, ``"n"`` none, ``"ct"`` constant + trend).
        maxlag: ADF maximum lag; ``None`` lets statsmodels choose.

    Returns:
        Tuple ``(t_statistic, p_value)``. ``(nan, nan)`` when the test cannot be
        computed (short sample, constant series, numerical failure) - callers
        must treat this as *not cointegrated* rather than raising.
    """
    from statsmodels.tsa.stattools import coint  # noqa: PLC0415 - heavy import

    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 30:
        return float("nan"), float("nan")
    x1 = joined.iloc[:, 0].to_numpy(dtype=float)
    x2 = joined.iloc[:, 1].to_numpy(dtype=float)
    if np.ptp(x1) < EPS or np.ptp(x2) < EPS:
        return float("nan"), float("nan")
    try:
        stat, pvalue, _ = coint(x1, x2, trend=trend, maxlag=maxlag, autolag="AIC")
    except Exception as exc:  # broad: statsmodels raises heterogeneous errors
        logger.debug("Engle-Granger test failed: %s", exc)
        return float("nan"), float("nan")
    return float(stat), float(pvalue)


def rolling_cointegration(
    a: pd.Series,
    b: pd.Series,
    window: int = 120,
    step: int = 5,
    trend: str = "c",
    pvalue_floor: float = 1.0,
) -> pd.DataFrame:
    """Rolling Engle-Granger cointegration diagnostics for a pair.

    The test is evaluated every ``step`` bars and forward-filled, which keeps
    the O(n * window) cost tractable while preserving a causal signal.

    Args:
        a: First price series.
        b: Second price series.
        window: Rolling test window.
        step: Evaluate every ``step`` bars, forward-fill in between.
        trend: Deterministic term passed to the cointegrating regression.
        pvalue_floor: Upper bound applied to the p-value (numerical guard).

    Returns:
        DataFrame with columns ``tstat`` and ``pvalue`` aligned to the union
        index of the inputs.
    """
    joined = pd.concat([a, b], axis=1).dropna()
    n = len(joined)
    tstat = np.full(n, np.nan)
    pvalue = np.full(n, np.nan)
    if n < window:
        return pd.DataFrame({"tstat": tstat, "pvalue": pvalue}, index=joined.index)

    x1 = joined.iloc[:, 0].to_numpy(dtype=float)
    x2 = joined.iloc[:, 1].to_numpy(dtype=float)
    for end in range(window, n + 1, step):
        slice_a = x1[end - window : end]
        slice_b = x2[end - window : end]
        stat, pval = engle_granger_test(
            pd.Series(slice_a), pd.Series(slice_b), trend=trend
        )
        tstat[end - 1] = stat
        pvalue[end - 1] = min(pval, pvalue_floor) if np.isfinite(pval) else np.nan
    frame = pd.DataFrame({"tstat": tstat, "pvalue": pvalue}, index=joined.index)
    return frame.ffill()


def rolling_adf_pvalue(
    series: pd.Series,
    window: int = 120,
    step: int = 5,
    regression: str = "c",
) -> pd.Series:
    """Rolling Augmented Dickey-Fuller p-value (stationarity test).

    This is Engle-Granger *step 2* applied to an already-constructed residual
    series (e.g. a rolling-beta spread): reject the null of a unit root and the
    residual is stationary, i.e. the pair is cointegrated for that hedge ratio.

    Args:
        series: Residual / spread series to test.
        window: Rolling test window.
        step: Evaluate every ``step`` bars and forward-fill in between.
        regression: ADF deterministic term (``"c"`` constant, ``"n"`` none).

    Returns:
        Rolling p-value series in ``[0, 1]``.

    Raises:
        ValueError: If ``window`` exceeds the length of ``series`` (returns an
            all-NaN series rather than raising when the series is very short).
    """
    from statsmodels.tsa.stattools import adfuller  # noqa: PLC0415 - heavy import

    clean = pd.Series(series).dropna()
    n = len(clean)
    pvalues = np.full(n, np.nan)
    if n < max(window, 12):
        return pd.Series(np.nan, index=pd.Series(series).index, name="adf_pvalue")

    values = clean.to_numpy(dtype=float)
    for end in range(window, n + 1, step):
        sample = values[end - window : end]
        if np.ptp(sample) < EPS:  # constant window -> no information
            continue
        try:
            pvalues[end - 1] = float(
                adfuller(sample, regression=regression, autolag="AIC", result_object=False)[1]
            )
        except Exception as exc:  # broad: statsmodels raises heterogeneous errors
            logger.debug("Rolling ADF failed at bar %d: %s", end, exc)
    out = pd.Series(pvalues, index=clean.index, name="adf_pvalue")
    return out.reindex(pd.Series(series).index).ffill()


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Pearson correlation of two return series.

    Args:
        a: First series.
        b: Second series.
        window: Rolling window.

    Returns:
        Rolling correlation in ``[-1, 1]``; ``NaN`` where undefined.
    """
    return a.rolling(window=window, min_periods=window).corr(b).rename("correlation")


# --------------------------------------------------------------------------- #
# Momentum indicator primitives
# --------------------------------------------------------------------------- #
def dynamic_ema(close: pd.Series, periods: pd.Series) -> pd.Series:
    """Exponential moving average with a time-varying smoothing period.

    The standard EMA recursion is used with a *bar-dependent* decay:
    ``alpha_t = 2 / (period_t + 1)``,
    ``EMA_t = alpha_t * P_t + (1 - alpha_t) * EMA_{t-1}``.

    Args:
        close: Price series.
        periods: Integer (or float) series of EMA periods, one per bar.

    Returns:
        Dynamic-EMA series.
    """
    prices = close.to_numpy(dtype=float)
    pers = periods.reindex(close.index).ffill().to_numpy(dtype=float)
    n = prices.size
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return pd.Series(out, index=close.index, name="dynamic_ema")
    prev = prices[0]
    out[0] = prev
    for i in range(1, n):
        period = pers[i]
        if not np.isfinite(period) or period <= 1.0:
            period = 2.0
        alpha = 2.0 / (period + 1.0)
        prev = alpha * prices[i] + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=close.index, name="dynamic_ema")


def rolling_vwap(
    ohlcv: pd.DataFrame,
    periods: pd.Series,
    volume_column: str = "volume",
) -> pd.Series:
    """Volume-weighted average price with a time-varying lookback.

    Implemented with prefix sums so each bar costs O(1) despite the variable
    window.  When volume is unavailable (or identically zero) the statistic
    degenerates gracefully to a rolling mean of the typical price.

    Args:
        ohlcv: Frame with ``high``, ``low``, ``close`` and volume columns.
        periods: Integer series of VWAP lookbacks, one per bar.
        volume_column: Name of the volume column.

    Returns:
        Dynamic VWAP series.
    """
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    typical = (high + low + close) / 3.0

    volume = ohlcv[volume_column].astype(float) if volume_column in ohlcv else None
    if volume is None or (volume.fillna(0.0) <= 0).all():
        volume = pd.Series(1.0, index=ohlcv.index)

    tv = (typical * volume).to_numpy(dtype=float)
    vol = volume.to_numpy(dtype=float)
    cum_tv = np.concatenate(([0.0], np.nancumsum(tv)))
    cum_vol = np.concatenate(([0.0], np.nancumsum(vol)))

    pers = periods.reindex(ohlcv.index).ffill().to_numpy(dtype=float)
    n = len(ohlcv)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        period = pers[i]
        if not np.isfinite(period) or period < 1:
            period = 1.0
        w = int(min(max(period, 1.0), i + 1))
        start = i + 1 - w
        vol_sum = cum_vol[i + 1] - cum_vol[start]
        if vol_sum <= EPS:
            out[i] = typical.iloc[max(0, start) : i + 1].mean()
        else:
            out[i] = (cum_tv[i + 1] - cum_tv[start]) / vol_sum
    return pd.Series(out, index=ohlcv.index, name="vwap")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Args:
        close: Price series.
        period: Lookback (default 14).

    Returns:
        RSI series in ``[0, 100]``.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 -> all gains -> RSI == 100; avg_gain == 0 -> RSI == 0
    out = out.where(avg_loss > EPS, other=100.0)
    out = out.where(avg_gain > EPS, other=0.0)
    return out.fillna(50.0).rename("rsi")


# --------------------------------------------------------------------------- #
# Orchestrating helper
# --------------------------------------------------------------------------- #
class FeatureBuilder:
    """Builds and caches per-symbol feature frames for a whole universe.

    Attributes:
        config: HMM configuration supplying lookbacks and feature columns.
        interval: Bar interval used for annualisation.
    """

    def __init__(self, config: Optional[cfg.HMMConfig] = None, interval: str = "1d") -> None:
        """Initialise the builder.

        Args:
            config: HMM configuration; defaults to ``settings.hmm``.
            interval: Bar interval.
        """
        self.config: cfg.HMMConfig = config or cfg.DEFAULT_SETTINGS.hmm
        self.interval: str = interval
        self._cache: Dict[str, pd.DataFrame] = {}

    @property
    def feature_columns(self) -> Sequence[str]:
        """Names of the model feature columns."""
        return self.config.feature_columns

    def symbol_features(self, symbol: str, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return (and cache) the feature frame for one symbol.

        Args:
            symbol: Canonical symbol.
            ohlcv: OHLCV frame.

        Returns:
            Feature frame with ``ret``, ``atr``, ``sigma`` and ``atr_raw``.
        """
        if symbol not in self._cache:
            self._cache[symbol] = build_feature_frame(
                ohlcv,
                atr_period=self.config.atr_period,
                vol_window=self.config.vol_window,
                interval=self.interval,
            )
        return self._cache[symbol]

    def market_features(self, data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Return the cross-sectional composite feature frame.

        Args:
            data: Mapping of symbol -> OHLCV frame.

        Returns:
            Composite feature frame with ``ret``, ``atr``, ``sigma``.
        """
        return build_market_features(
            data,
            atr_period=self.config.atr_period,
            vol_window=self.config.vol_window,
            interval=self.interval,
        )


__all__: List[str] = [
    "EPS",
    "BARS_PER_YEAR",
    "log_returns",
    "true_range",
    "wilder_atr",
    "realized_volatility",
    "build_feature_frame",
    "build_market_features",
    "standardize_features",
    "lag_features",
    "rolling_zscore",
    "ols_hedge_ratio",
    "rolling_hedge_ratio",
    "compute_spread",
    "engle_granger_test",
    "rolling_cointegration",
    "rolling_correlation",
    "dynamic_ema",
    "rolling_vwap",
    "rsi",
    "FeatureBuilder",
]
