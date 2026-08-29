"""Market-data ingestion.

The ingestion layer is transport-agnostic: it exposes one public entry point,
:meth:`DataIngestion.fetch_universe`, which returns a mapping of canonical
symbol -> OHLCV :class:`pandas.DataFrame`.  Supported backends:

``yfinance``
    Default source for daily and intraday bars of FX crosses, metals and
    futures.  Each :class:`~quant_system.config.settings.AssetSpec` carries an
    ordered ticker *fallback chain* so that a renamed or delisted venue ticker
    degrades gracefully instead of failing the run.

``synthetic``
    Deterministic regime-switching GBM generator used when the network is not
    reachable (sandboxes, CI, offline development).  Prices reproduce the
    volatility character of each asset via ``AssetSpec.vol_scale``.

``mt5`` / ``ccxt``
    Live/broker and exchange adapters (optional dependencies).

Caching
-------
Downloaded frames are pickled under ``settings.DATA_CACHE_DIR``; the *source* is
encoded in the filename so a cached synthetic series is never silently mistaken
for real market data.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quant_system.config import settings as cfg

logger = logging.getLogger(__name__)

OHLCV_COLUMNS: List[str] = ["open", "high", "low", "close", "volume"]

#: Regime parameters for the synthetic generator: (daily vol, daily drift).
_SYNTHETIC_REGIMES: Sequence[tuple[float, float]] = (
    (0.0045, 0.00020),  # State 0 - calm, mildly positive drift
    (0.0095, 0.00075),  # State 1 - trending, higher vol
    (0.0260, -0.00180),  # State 2 - shock, fat tails, negative drift
)

#: Persistent Markov chain over the three synthetic regimes.
_SYNTHETIC_TRANSITIONS: np.ndarray = np.array(
    [
        [0.975, 0.022, 0.003],
        [0.030, 0.955, 0.015],
        [0.080, 0.120, 0.800],
    ]
)


def _hash_key(*parts: object) -> str:
    """Build a short deterministic hash string from ``parts``.

    Args:
        *parts: Hashable components of the cache key.

    Returns:
        12-character hexadecimal digest.
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


class SyntheticDataGenerator:
    """Deterministic three-regime synthetic OHLCV generator.

    The generator simulates a Markov-switching geometric Brownian motion with
    occasional jump events, then derives an intrabar ``high``/``low`` range from
    the realised volatility of each bar.  Output is fully reproducible for a
    given ``(symbol, start, end, interval, seed)`` tuple, which makes it safe
    for unit tests and offline walk-forward runs.

    Attributes:
        seed: Master seed for all stochastic draws.
    """

    def __init__(self, seed: int = cfg.SYNTHETIC_SEED) -> None:
        """Initialise the generator.

        Args:
            seed: Master random seed.
        """
        self.seed: int = int(seed)

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def make_index(start: str, end: str, interval: str) -> pd.DatetimeIndex:
        """Build the bar index for the requested interval.

        Args:
            start: Inclusive start date (``YYYY-MM-DD``).
            end: Inclusive end date (``YYYY-MM-DD``).
            interval: ``"1d"``, ``"1h"`` or ``"15m"``.

        Returns:
            Timezone-naive :class:`pandas.DatetimeIndex`.

        Raises:
            ValueError: If ``interval`` is not supported.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if interval == "1d":
            return pd.bdate_range(start_ts, end_ts, name="timestamp")
        freq_map: Dict[str, str] = {"1h": "1h", "15m": "15min", "5m": "5min"}
        if interval not in freq_map:
            raise ValueError(f"Unsupported interval {interval!r}; use one of {sorted(freq_map)}")
        # Business-hour-ish grid: 24h crypto/FX style continuous session.
        return pd.date_range(start_ts, end_ts, freq=freq_map[interval], name="timestamp")

    # ------------------------------------------------------------------ #
    # Core generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Generate a synthetic OHLCV frame for one symbol.

        Args:
            symbol: Canonical system symbol.
            start: Inclusive start date.
            end: Inclusive end date.
            interval: Bar interval.

        Returns:
            DataFrame indexed by timestamp with lowercase OHLCV columns.
        """
        spec = self._spec_for(symbol)
        index = self.make_index(start, end, interval)
        n: int = len(index)
        if n == 0:
            return pd.DataFrame(columns=OHLCV_COLUMNS, index=index)

        # Per-symbol sub-seed keeps the universe jointly reproducible but not
        # pathologically correlated.
        sub_seed: int = (self.seed + int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)) % (2**31)
        rng = np.random.default_rng(sub_seed)

        bars_per_day: float = 1.0 if interval == "1d" else (24.0 if interval == "1h" else 96.0)
        scale: float = float(np.sqrt(1.0 / bars_per_day))

        # --- Markov regime path ----------------------------------------- #
        regimes = self._simulate_regimes(rng, n)

        vol = np.array([_SYNTHETIC_REGIMES[r][0] for r in regimes]) * spec.vol_scale * scale
        drift = np.array([_SYNTHETIC_REGIMES[r][1] for r in regimes]) / bars_per_day

        # --- Return path: GBM + Student-t fat tails + Poisson jumps ----- #
        dof: int = 5
        shocks = rng.standard_t(dof, size=n) / np.sqrt(dof / (dof - 2.0))
        jumps = (rng.random(n) < 0.004) * rng.normal(0.0, 3.0, size=n) * vol
        log_ret = drift + vol * shocks + jumps

        close = spec.base_price * np.exp(np.cumsum(log_ret))

        # --- Intrabar range from realised volatility -------------------- #
        open_ = np.empty(n, dtype=float)
        open_[0] = close[0] * (1.0 - 0.25 * vol[0])
        open_[1:] = close[:-1]

        span = np.abs(vol) * (0.55 + 0.85 * rng.random(n))
        high = np.maximum(open_, close) * (1.0 + span)
        low = np.minimum(open_, close) * (1.0 - span)
        # Guarantee the OHLC ordering invariants hold.
        high = np.maximum.reduce([high, open_, close])
        low = np.minimum.reduce([low, open_, close])
        low = np.maximum(low, 1e-8)

        volume = (1.0 + 4.0 * rng.random(n)) * 1_000.0 * (1.0 + 6.0 * np.abs(log_ret) / vol.mean())

        frame = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "synthetic_regime": regimes.astype(int),
            },
            index=index,
        )
        frame.attrs["source"] = "synthetic"
        frame.attrs["symbol"] = symbol
        return frame

    def generate_universe(
        self,
        symbols: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Generate a *jointly* consistent synthetic universe.

        Unlike :meth:`generate` (which draws each symbol independently) this
        method imposes the economic structure the strategies depend on:

        * a single shared Markov regime path, so shocks hit the whole book
          simultaneously (this is what the HMM is supposed to detect);
        * **XAUUSD / XAGUSD are cointegrated by construction** - gold is
          generated as ``beta0 * silver + OU`` so the Engle-Granger test and the
          mean-reverting spread z-score behave like the real metals complex;
        * **EURUSD / USDCHF are inversely correlated by construction** - the
          Swissie is generated as ``exp(c - gamma * ln(EURUSD) + OU)`` which
          yields a rolling return correlation of roughly ``-0.8``;
        * every other symbol follows its own regime-switching GBM.

        Args:
            symbols: Symbols to generate, in any order (dependencies are
                resolved internally).
            start: Inclusive start date.
            end: Inclusive end date.
            interval: Bar interval.

        Returns:
            Mapping of symbol -> OHLCV DataFrame on a shared index.
        """
        symbols = list(symbols)
        index = self.make_index(start, end, interval)
        n: int = len(index)
        if n == 0:
            return {s: pd.DataFrame(columns=OHLCV_COLUMNS, index=index) for s in symbols}

        rng = np.random.default_rng(self.seed)
        regimes = self._simulate_regimes(rng, n)
        bars_per_day = 1.0 if interval == "1d" else (24.0 if interval == "1h" else 96.0)
        scale = float(np.sqrt(1.0 / bars_per_day))

        def _regime_paths(symbol: str) -> Tuple[np.ndarray, np.ndarray]:
            spec = self._spec_for(symbol)
            vol = np.array([_SYNTHETIC_REGIMES[r][0] for r in regimes]) * spec.vol_scale * scale
            drift = np.array([_SYNTHETIC_REGIMES[r][1] for r in regimes]) / bars_per_day
            return vol, drift

        def _walk(symbol: str, rng_: np.random.Generator) -> np.ndarray:
            """Return a log-price path for an independent symbol."""
            vol, drift = _regime_paths(symbol)
            spec = self._spec_for(symbol)
            shocks = rng_.standard_t(5, size=n) / np.sqrt(5.0 / 3.0)
            jumps = (rng_.random(n) < 0.004) * rng_.normal(0.0, 3.0, size=n) * vol
            log_ret = drift + vol * shocks + jumps
            return np.log(spec.base_price) + np.cumsum(log_ret)

        def _ou(theta: float, sigma: float, rng_: np.random.Generator) -> np.ndarray:
            """Ornstein-Uhlenbeck path (zero long-run mean)."""
            eps = rng_.standard_normal(n)
            out = np.zeros(n)
            for t in range(1, n):
                out[t] = out[t - 1] * (1.0 - theta) + sigma * eps[t]
            return out

        log_prices: Dict[str, np.ndarray] = {}
        # --- dependency-ordered generation -------------------------------- #
        ordered: List[str] = []
        for anchor in ("XAGUSD", "EURUSD"):
            if anchor in symbols and anchor not in ordered:
                ordered.append(anchor)
        for symbol in symbols:
            if symbol not in ordered:
                ordered.append(symbol)

        for symbol in ordered:
            if symbol == "XAUUSD" and "XAGUSD" in symbols:
                silver = np.exp(log_prices["XAGUSD"])
                spread = _ou(theta=0.05, sigma=18.0, rng_=rng)  # sd ~ 57
                beta0 = (
                    self._spec_for("XAUUSD").base_price / self._spec_for("XAGUSD").base_price
                )
                gold = beta0 * silver + spread
                log_prices[symbol] = np.log(np.maximum(gold, 1e-8))
            elif symbol == "USDCHF" and "EURUSD" in symbols:
                eur = log_prices["EURUSD"]
                gamma = 0.92
                base_eur = self._spec_for("EURUSD").base_price
                base_chf = self._spec_for("USDCHF").base_price
                intercept = np.log(base_chf) + gamma * np.log(base_eur)
                resid = _ou(theta=0.05, sigma=0.0011, rng_=rng)  # sd ~ 0.0035
                log_prices[symbol] = intercept - gamma * eur + resid
            else:
                log_prices[symbol] = _walk(symbol, rng)

        frames: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            close = np.exp(log_prices[symbol])
            spec = self._spec_for(symbol)
            vol, _ = _regime_paths(symbol)

            open_ = np.empty(n)
            open_[0] = close[0] * (1.0 - 0.25 * vol[0])
            open_[1:] = close[:-1]
            span = np.abs(vol) * (0.55 + 0.85 * rng.random(n))
            high = np.maximum.reduce([np.maximum(open_, close) * (1.0 + span), open_, close])
            low = np.minimum.reduce([np.minimum(open_, close) * (1.0 - span), open_, close])
            low = np.maximum(low, 1e-8)
            volume = (1.0 + 4.0 * rng.random(n)) * 1_000.0

            frame = pd.DataFrame(
                {
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "synthetic_regime": regimes.astype(int),
                },
                index=index,
            )
            frame.attrs["source"] = "synthetic"
            frame.attrs["symbol"] = symbol
            frames[symbol] = frame
        return frames

    @staticmethod
    def _spec_for(symbol: str) -> cfg.AssetSpec:
        """Resolve the :class:`AssetSpec` for ``symbol``, with a safe default.

        Args:
            symbol: Canonical system symbol.

        Returns:
            The matching asset spec, or a generic spec for unknown symbols.
        """
        try:
            return cfg.DEFAULT_SETTINGS.universe.spec(symbol)
        except KeyError:
            logger.warning("Unknown symbol %r - using generic synthetic spec.", symbol)
            return cfg.AssetSpec(
                symbol=symbol,
                yf_symbols=(symbol,),
                mt5_symbol=symbol,
                pip_size=0.0001,
                contract_size=1.0,
                asset_class="fx",
                spread_pips=1.5,
                slippage_pips=0.2,
                vol_scale=1.0,
                base_price=100.0,
            )

    @staticmethod
    def _simulate_regimes(rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw a Markov-chain regime path.

        Args:
            rng: Seeded NumPy generator.
            n: Number of bars.

        Returns:
            Integer array of regime ids in ``{0, 1, 2}``.
        """
        cumulative = _SYNTHETIC_TRANSITIONS.cumsum(axis=1)
        regimes = np.empty(n, dtype=np.int64)
        regimes[0] = 0
        draws = rng.random(n)
        for t in range(1, n):
            regimes[t] = int(np.searchsorted(cumulative[regimes[t - 1]], draws[t]))
        return regimes


class DataIngestion:
    """Fetches, caches and normalises OHLCV market data for the universe.

    Attributes:
        source: Requested backend: ``"auto"``, ``"yfinance"``, ``"synthetic"``,
            ``"mt5"`` or ``"ccxt"``.
        cache_dir: Directory used for the pickle cache.
        use_cache: Whether to read/write the on-disk cache.
        generator: Synthetic fallback generator instance.
    """

    def __init__(
        self,
        source: str = "auto",
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        seed: int = cfg.SYNTHETIC_SEED,
    ) -> None:
        """Initialise the ingestion service.

        Args:
            source: Backend selector (see class docstring).
            cache_dir: Cache directory; defaults to ``settings.DATA_CACHE_DIR``.
            use_cache: Enable the on-disk pickle cache.
            seed: Seed for the synthetic fallback generator.
        """
        self.source: str = source.lower()
        self.cache_dir: Path = Path(cache_dir) if cache_dir else cfg.DATA_CACHE_DIR
        self.use_cache: bool = use_cache
        self.generator: SyntheticDataGenerator = SyntheticDataGenerator(seed=seed)
        self._sources_used: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def sources_used(self) -> Dict[str, str]:
        """Mapping of symbol -> backend that actually produced its data."""
        return dict(self._sources_used)

    def fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch a single symbol's OHLCV series.

        Args:
            symbol: Canonical system symbol (e.g. ``"XAUUSD"``).
            start: Inclusive start date.
            end: Inclusive end date.
            interval: Bar interval.

        Returns:
            Normalised DataFrame with lowercase ``open/high/low/close/volume``
            columns and a timezone-naive ``DatetimeIndex`` named ``timestamp``.
        """
        cached = self._read_cache(symbol, start, end, interval)
        if cached is not None:
            self._sources_used[symbol] = cached.attrs.get("source", "cache")
            return cached

        frame: Optional[pd.DataFrame] = None
        if self.source in {"auto", "yfinance"}:
            frame = self._fetch_yfinance(symbol, start, end, interval)
        if frame is None and self.source in {"auto", "mt5"}:
            frame = self._fetch_mt5(symbol, start, end, interval)
        if frame is None and self.source in {"auto", "ccxt"}:
            frame = self._fetch_ccxt(symbol, start, end, interval)
        if frame is None and self.source == "mt5":
            frame = self._fetch_mt5(symbol, start, end, interval)
        if frame is None and self.source == "ccxt":
            frame = self._fetch_ccxt(symbol, start, end, interval)

        if frame is None or frame.empty:
            if self.source == "yfinance":
                logger.warning(
                    "yfinance returned no data for %s; falling back to synthetic.", symbol
                )
            frame = self.generator.generate(symbol, start, end, interval)
            frame.attrs["source"] = "synthetic"
        frame = self._normalise(frame, symbol)
        self._sources_used[symbol] = str(frame.attrs.get("source", "unknown"))
        self._write_cache(frame, symbol, start, end, interval)
        return frame

    def fetch_universe(
        self,
        symbols: Optional[Iterable[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        align: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch every symbol in the universe and optionally align them.

        Args:
            symbols: Symbols to fetch; defaults to the configured universe.
            start: Inclusive start date; defaults to ``settings.backtest.start``.
            end: Inclusive end date; defaults to ``settings.backtest.end``.
            interval: Bar interval.
            align: If ``True``, intersect all indices onto a common calendar and
                forward-fill gaps so the backtester sees a consistent grid.

        Returns:
            Mapping of symbol -> normalised OHLCV DataFrame.
        """
        symbols = tuple(symbols) if symbols is not None else cfg.DEFAULT_SETTINGS.symbols
        start = start or cfg.DEFAULT_SETTINGS.backtest.start
        end = end or cfg.DEFAULT_SETTINGS.backtest.end

        if self.source == "synthetic":
            frames = self.generator.generate_universe(list(symbols), start, end, interval)
            for symbol, frame in frames.items():
                self._sources_used[symbol] = "synthetic"
            return align_universe(frames) if align else frames

        data: Dict[str, pd.DataFrame] = {
            symbol: self.fetch(symbol, start, end, interval) for symbol in symbols
        }
        data = {sym: df for sym, df in data.items() if not df.empty}
        sources = {str(df.attrs.get("source", "unknown")) for df in data.values()}
        if sources == {"synthetic"} and len(data) > 1:
            # Every symbol fell back: regenerate jointly so the pairs keep their
            # cointegrating / inverse-correlation structure.
            logger.info(
                "All %d symbols fell back to synthetic data; regenerating a "
                "coherent universe (shared regimes + pair structure).",
                len(data),
            )
            data = self.generator.generate_universe(list(data.keys()), start, end, interval)
            for symbol, frame in data.items():
                self._sources_used[symbol] = "synthetic"
        if not data:
            raise RuntimeError("No market data could be loaded for any symbol.")
        if align:
            data = align_universe(data)
        return data

    # ------------------------------------------------------------------ #
    # Backends
    # ------------------------------------------------------------------ #
    def _fetch_yfinance(
        self, symbol: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Download bars from Yahoo Finance using the ticker fallback chain.

        Args:
            symbol: Canonical system symbol.
            start: Inclusive start date.
            end: Inclusive end date.
            interval: Bar interval.

        Returns:
            Raw OHLCV DataFrame, or ``None`` if every attempt failed.
        """
        try:
            import yfinance as yf  # noqa: PLC0415 - optional dependency
        except Exception:  # pragma: no cover - environment dependent
            logger.debug("yfinance is not installed; skipping this backend.")
            return None

        tickers: tuple[str, ...]
        try:
            tickers = cfg.DEFAULT_SETTINGS.universe.spec(symbol).yf_symbols
        except KeyError:
            tickers = (symbol,)

        for ticker in tickers:
            frame = self._download_one(yf, ticker, start, end, interval)
            if frame is not None and not frame.empty:
                frame.attrs["source"] = "yfinance"
                frame.attrs["ticker"] = ticker
                logger.info("Loaded %s from yfinance ticker %s (%d bars)", symbol, ticker, len(frame))
                return frame
        return None

    @staticmethod
    def _download_one(
        yf: object, ticker: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Execute a single yfinance download attempt.

        Args:
            yf: Imported ``yfinance`` module.
            ticker: Venue ticker.
            start: Inclusive start date.
            end: Inclusive end date.
            interval: Bar interval.

        Returns:
            Downloaded DataFrame or ``None`` on failure.
        """
        try:
            frame = yf.download(  # type: ignore[attr-defined]
                tickers=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
                multi_level_index=False,
            )
        except TypeError:
            # Older yfinance releases do not accept ``multi_level_index``.
            try:
                frame = yf.download(  # type: ignore[attr-defined]
                    tickers=ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                logger.debug("yfinance download failed for %s: %s", ticker, exc)
                return None
        except Exception as exc:  # pragma: no cover - network dependent
            logger.debug("yfinance download failed for %s: %s", ticker, exc)
            return None
        if frame is None:
            return None
        return frame

    def _fetch_mt5(
        self, symbol: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch bars from a locally installed MetaTrader 5 terminal.

        Args:
            symbol: Canonical system symbol.
            start: Inclusive start date.
            end: Inclusive end date (ignored when ``count`` is inferred).
            interval: Bar interval mapped onto an MT5 timeframe.

        Returns:
            OHLCV DataFrame, or ``None`` if MT5 is unavailable.
        """
        try:
            import MetaTrader5 as mt5  # noqa: PLC0415 - optional, Windows-only
        except Exception:
            logger.debug("MetaTrader5 package unavailable; skipping MT5 backend.")
            return None

        timeframe_map: Dict[str, int] = {
            "1d": mt5.TIMEFRAME_D1,
            "1h": mt5.TIMEFRAME_H1,
            "15m": mt5.TIMEFRAME_M15,
            "5m": mt5.TIMEFRAME_M5,
        }
        if interval not in timeframe_map:
            return None
        if not mt5.initialize():
            logger.warning("MT5 initialize() failed: %s", mt5.last_error())
            return None
        try:
            broker_symbol = cfg.DEFAULT_SETTINGS.universe.spec(symbol).mt5_symbol
            rates = mt5.copy_rates_range(
                broker_symbol,
                timeframe_map[interval],
                pd.Timestamp(start).to_pydatetime(),
                pd.Timestamp(end).to_pydatetime(),
            )
            if rates is None or len(rates) == 0:
                return None
            frame = pd.DataFrame(rates)
            frame["time"] = pd.to_datetime(frame["time"], unit="s")
            frame = frame.set_index("time").rename(
                columns={
                    "tick_volume": "volume",
                    "real_volume": "real_volume",
                }
            )
            frame.index.name = "timestamp"
            return frame
        finally:
            mt5.shutdown()

    @staticmethod
    def _fetch_ccxt(
        symbol: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Fetch bars from a CCXT-supported exchange.

        Args:
            symbol: Canonical system symbol.
            start: Inclusive start date.
            end: Inclusive end date.
            interval: Bar interval.

        Returns:
            OHLCV DataFrame, or ``None`` if CCXT is unavailable.
        """
        try:
            import ccxt  # noqa: F401,PLC0415 - optional dependency
        except Exception:
            logger.debug("ccxt unavailable; skipping CCXT backend.")
            return None
        logger.warning(
            "CCXT backend is declared but not configured for %s "
            "(supply an exchange id and credentials in BrokerConfig).",
            symbol,
        )
        return None

    # ------------------------------------------------------------------ #
    # Normalisation & cache
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Coerce a vendor frame into the canonical OHLCV schema.

        Args:
            frame: Raw vendor DataFrame.
            symbol: Canonical symbol (recorded in ``attrs``).

        Returns:
            DataFrame with lowercase OHLCV columns, sorted unique index.
        """
        out = frame.copy()
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [str(col[0]).lower() for col in out.columns]
        else:
            out.columns = [str(col).lower() for col in out.columns]
        out = out.loc[:, ~out.columns.duplicated()]

        for column in OHLCV_COLUMNS:
            if column not in out.columns:
                if column == "volume":
                    out[column] = 0.0
                else:
                    out[column] = np.nan
        # Back-fill missing OHLC from close before dropping NaNs.
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        for column in ("open", "high", "low"):
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(out["close"])
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)

        out = out[OHLCV_COLUMNS].astype(float)
        out.index = pd.to_datetime(out.index, utc=False)
        out.index.name = "timestamp"
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out = out.dropna(subset=["close"])
        out = out[out["close"] > 0]
        out.attrs["symbol"] = symbol
        return out

    def _cache_stem(self, symbol: str, start: str, end: str, interval: str) -> str:
        """Return the cache filename stem shared by all sources.

        Args:
            symbol: Canonical symbol.
            start: Start date.
            end: End date.
            interval: Bar interval.

        Returns:
            The filename prefix (without the source suffix or extension).
        """
        key = _hash_key(symbol, start, end, interval)
        safe = "".join(ch if ch.isalnum() else "_" for ch in symbol)
        return f"{safe}_{start}_{end}_{interval}_{key}"

    def _cache_path(
        self, symbol: str, start: str, end: str, interval: str, source: str
    ) -> Path:
        """Return the cache file path for a request and a resolved source.

        Args:
            symbol: Canonical symbol.
            start: Start date.
            end: End date.
            interval: Bar interval.
            source: Backend that produced the data.

        Returns:
            Path to the pickle file.
        """
        return self.cache_dir / f"{self._cache_stem(symbol, start, end, interval)}_{source}.pkl"

    def _read_cache(
        self, symbol: str, start: str, end: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Read a cached frame, always preferring real market data.

        The source is encoded in the filename so a cached *synthetic* fallback can
        never shadow a genuinely downloaded series: candidates are inspected and
        any non-synthetic entry wins.

        Args:
            symbol: Canonical symbol.
            start: Start date.
            end: End date.
            interval: Bar interval.

        Returns:
            Cached DataFrame, or ``None``.
        """
        if not self.use_cache:
            return None
        pattern = f"{self._cache_stem(symbol, start, end, interval)}_*.pkl"
        candidates = sorted(self.cache_dir.glob(pattern)) if self.cache_dir.exists() else []
        if not candidates:
            return None

        loaded: List[Tuple[str, pd.DataFrame]] = []
        for path in candidates:
            try:
                with path.open("rb") as handle:
                    frame = pickle.load(handle)
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    loaded.append((str(frame.attrs.get("source", "unknown")), frame))
            except Exception as exc:  # pragma: no cover - corrupt cache
                logger.debug("Cache read failed for %s (%s); refetching.", path.name, exc)
        if not loaded:
            return None
        for source, frame in loaded:
            if source != "synthetic":
                return frame
        return loaded[0][1]

    def _write_cache(
        self,
        frame: pd.DataFrame,
        symbol: str,
        start: str,
        end: str,
        interval: str,
    ) -> None:
        """Persist a frame to the pickle cache, keyed by its resolved source.

        Args:
            frame: Frame to persist.
            symbol: Canonical symbol.
            start: Start date.
            end: End date.
            interval: Bar interval.
        """
        if not self.use_cache:
            return
        source = str(frame.attrs.get("source", "unknown"))
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with self._cache_path(symbol, start, end, interval, source).open("wb") as handle:
                pickle.dump(frame, handle)
        except Exception as exc:  # pragma: no cover - filesystem dependent
            logger.debug("Cache write failed for %s: %s", symbol, exc)


def align_universe(
    data: Mapping[str, pd.DataFrame], method: str = "intersection"
) -> Dict[str, pd.DataFrame]:
    """Align a symbol->frame mapping onto a common index.

    Args:
        data: Mapping of symbol to OHLCV frame.
        method: ``"intersection"`` (default) keeps only timestamps present in
            every frame; ``"union"`` keeps all timestamps and forward-fills
            missing prices.

    Returns:
        New mapping with aligned indices.

    Raises:
        ValueError: If ``data`` is empty or ``method`` is unknown.
    """
    if not data:
        raise ValueError("Cannot align an empty universe.")
    indices = [frame.index for frame in data.values()]
    if method == "intersection":
        common = indices[0]
        for index in indices[1:]:
            common = common.intersection(index)
    elif method == "union":
        common = indices[0]
        for index in indices[1:]:
            common = common.union(index)
        common = common.sort_values()
    else:
        raise ValueError(f"Unknown alignment method {method!r}")

    aligned: Dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        reindexed = frame.reindex(common).sort_index()
        if method == "union":
            reindexed = reindexed.ffill()
            reindexed["volume"] = reindexed["volume"].fillna(0.0)
        reindexed.attrs.update(frame.attrs)
        aligned[symbol] = reindexed
    return aligned


def load_universe(
    source: str = "auto",
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    symbols: Optional[Sequence[str]] = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Convenience helper that builds an ingestion service and fetches data.

    Args:
        source: Backend selector.
        start: Inclusive start date.
        end: Inclusive end date.
        interval: Bar interval.
        symbols: Optional symbol subset.
        use_cache: Enable the pickle cache.

    Returns:
        Mapping of symbol -> aligned OHLCV DataFrame.
    """
    ingestion = DataIngestion(source=source, use_cache=use_cache)
    return ingestion.fetch_universe(symbols=symbols, start=start, end=end, interval=interval)


__all__: List[str] = [
    "OHLCV_COLUMNS",
    "SyntheticDataGenerator",
    "DataIngestion",
    "align_universe",
    "load_universe",
]
