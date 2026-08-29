"""System constants, asset universe and hyper-parameters.

This module is the *single source of truth* for every tunable in the framework.
Nothing here performs I/O or imports heavy scientific packages, so it is safe
(and cheap) to import from anywhere in the system.

Conventions
-----------
* All prices are in quote currency; quantities are in *units* of the base asset
  and are multiplied by :attr:`AssetSpec.contract_size` to obtain notional.
* Time is always a timezone-naive ``DatetimeIndex`` in UTC.
* HMM regime identifiers follow the canonical mapping produced by
  :class:`quant_system.models.hmm_switchboard.HMMSwitchboard`::

      STATE_RANGE_BOUND = 0   low volatility, mean-reverting
      STATE_TREND       = 1   directional momentum / trending
      STATE_SHOCK       = 2   high volatility, market stress
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Final, FrozenSet, List, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_CACHE_DIR: Final[Path] = PROJECT_ROOT / "data_cache"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
#: Directory used by the notifier's file-based speech synthesis.
VOICE_DIR: Final[Path] = REPORTS_DIR / "voice"
#: Directory holding the rotating runtime log for unattended live sessions.
LOGS_DIR: Final[Path] = REPORTS_DIR / "logs"
#: Sentinel file whose presence halts the live loop and (optionally) flattens.
HALT_FILE: Final[Path] = REPORTS_DIR / "HALT"

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #
TRADING_DAYS_PER_YEAR: Final[int] = 252
RISK_FREE_RATE: Final[float] = 0.0  # Sharpe/Sortino computed with Rf = 0%

N_REGIMES: Final[int] = 3

STATE_RANGE_BOUND: Final[int] = 0
STATE_TREND: Final[int] = 1
STATE_SHOCK: Final[int] = 2

REGIME_LABELS: Final[Dict[int, str]] = {
    STATE_RANGE_BOUND: "Low-Volatility / Range-Bound",
    STATE_TREND: "High-Momentum / Strong Trend",
    STATE_SHOCK: "High-Volatility / Market Shock",
}


# --------------------------------------------------------------------------- #
# Asset universe
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AssetSpec:
    """Static description of a tradable instrument.

    Attributes:
        symbol: Canonical system symbol (e.g. ``"XAUUSD"``).
        yf_symbols: Ordered yfinance ticker fallback chain. The first ticker that
            returns a non-empty frame wins, which makes ingestion resilient to
            venue/renaming changes.
        mt5_symbol: MetaTrader 5 broker symbol (empty string if unavailable).
        pip_size: Price increment corresponding to one pip.
        contract_size: Notional multiplier for one unit of quantity.
        asset_class: One of ``{"metal", "fx"}``.
        spread_pips: Typical round-turn-quoted spread, in pips.
        slippage_pips: Adverse fill slippage per side, in pips.
        vol_scale: Multiplier applied to the synthetic data generator's base
            volatility so that fallback data preserves each asset's character.
        base_price: Anchor price used by the synthetic data generator.
    """

    symbol: str
    yf_symbols: Tuple[str, ...]
    mt5_symbol: str
    pip_size: float
    contract_size: float
    asset_class: str
    spread_pips: float
    slippage_pips: float
    vol_scale: float
    base_price: float


@dataclass(frozen=True)
class UniverseConfig:
    """Definition of the traded universe and of the stat-arb pair book."""

    assets: Tuple[AssetSpec, ...] = (
        AssetSpec(
            symbol="XAUUSD",
            yf_symbols=("XAUUSD=X", "GC=F"),
            mt5_symbol="XAUUSD",
            pip_size=0.01,
            contract_size=100.0,  # 100 troy oz per standard lot
            asset_class="metal",
            spread_pips=3.0,
            slippage_pips=0.5,
            vol_scale=1.00,
            base_price=2050.00,
        ),
        AssetSpec(
            symbol="XAGUSD",
            yf_symbols=("XAGUSD=X", "SI=F"),
            mt5_symbol="XAGUSD",
            pip_size=0.001,
            contract_size=5000.0,  # 5,000 troy oz per standard lot
            asset_class="metal",
            spread_pips=4.0,
            slippage_pips=0.5,
            vol_scale=1.45,
            base_price=24.50,
        ),
        AssetSpec(
            symbol="EURUSD",
            yf_symbols=("EURUSD=X",),
            mt5_symbol="EURUSD",
            pip_size=0.0001,
            contract_size=100_000.0,
            asset_class="fx",
            spread_pips=1.2,
            slippage_pips=0.2,
            vol_scale=0.42,
            base_price=1.0850,
        ),
        AssetSpec(
            symbol="USDCHF",
            yf_symbols=("USDCHF=X",),
            mt5_symbol="USDCHF",
            pip_size=0.0001,
            contract_size=100_000.0,
            asset_class="fx",
            spread_pips=1.5,
            slippage_pips=0.2,
            vol_scale=0.45,
            base_price=0.8820,
        ),
        AssetSpec(
            symbol="USDJPY",
            yf_symbols=("USDJPY=X",),
            mt5_symbol="USDJPY",
            pip_size=0.01,
            contract_size=100_000.0,
            asset_class="fx",
            spread_pips=1.3,
            slippage_pips=0.2,
            vol_scale=0.50,
            base_price=148.50,
        ),
    )

    #: Cointegrated precious-metals pair (Engle-Granger 2-step).
    metals_pair: Tuple[str, str] = ("XAUUSD", "XAGUSD")
    #: Inversely-correlated FX pair (log-price spread).
    fx_pair: Tuple[str, str] = ("EURUSD", "USDCHF")
    #: Momentum book.
    momentum_symbols: Tuple[str, str] = ("USDJPY", "XAUUSD")

    @property
    def symbols(self) -> Tuple[str, ...]:
        """All canonical symbols in the universe, in declaration order."""
        return tuple(spec.symbol for spec in self.assets)

    def spec(self, symbol: str) -> AssetSpec:
        """Return the :class:`AssetSpec` for ``symbol``.

        Args:
            symbol: Canonical symbol, e.g. ``"XAUUSD"``.

        Raises:
            KeyError: If ``symbol`` is not part of the configured universe.
        """
        for spec in self.assets:
            if spec.symbol == symbol:
                return spec
        raise KeyError(f"Symbol {symbol!r} is not in the configured universe.")


# --------------------------------------------------------------------------- #
# HMM Regime Switchboard
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HMMConfig:
    """Hyper-parameters for the Gaussian-HMM regime switchboard."""

    n_components: int = 3
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42
    #: Feature vector X_t = [r_t, ATR_t, sigma_t]^T
    feature_columns: Tuple[str, ...] = ("ret", "atr", "sigma")
    #: Annualisation window for realised volatility (N = 20).
    vol_window: int = 20
    #: Wilder ATR lookback.
    atr_period: int = 14
    #: Rolling bars used to fit the HMM inside the causal streaming pass.
    train_window: int = 504  # ~2 years of daily bars
    #: Refit cadence (bars) for the causal streaming pass.
    refit_every: int = 21  # ~1 month
    #: Minimum bars required before the first regime label is emitted.
    min_train: int = 252
    #: Probability floor below which a regime label is considered unreliable.
    min_state_confidence: float = 0.0
    n_fits: int = 1  # random restarts; best log-likelihood wins


# --------------------------------------------------------------------------- #
# Strategy: statistical arbitrage (active in State 0)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StatArbConfig:
    """Hyper-parameters for the pairs-trading strategies."""

    #: Rolling OLS window used to estimate the hedge ratio.  Longer windows give
    #: a more stable beta; measured on XAU/XAG (2016-2024) a 90-bar window is the
    #: sweet spot between stability and responsiveness.
    window: int = 90
    #: Rolling window for the spread mean / standard deviation used by the
    #: z-score.  Deliberately shorter than :attr:`window` so the z-score reacts
    #: faster than the hedge ratio it is computed from.
    zscore_window: int = 60
    #: Window for the rolling Engle-Granger cointegration / ADF tests.
    coint_window: int = 100
    #: Rolling window for the FX pair's inverse-correlation gate.
    correlation_window: int = 90
    #: Cointegration / ADF tests are evaluated every ``coint_step`` bars and
    #: forward-filled in between.  A gate does not need bar-by-bar resolution,
    #: and stepping is what keeps the walk-forward sweep tractable.
    coint_step: int = 10
    #: Significance level below which the pair is declared cointegrated.
    coint_pvalue: float = 0.05
    #: Which stationarity test gates entries:
    #: ``"coint"``       - :func:`statsmodels.tsa.stattools.coint` on the two
    #:                     price series (the literal Engle-Granger two-step test).
    #: ``"adf_spread"``  - ADF test on the residuals of the *rolling-beta* spread,
    #:                     i.e. Engle-Granger step 2 applied to the exact series
    #:                     being traded.  Materially higher power on real data
    #:                     (measured: 18.4 % of bars vs 5.7 % for ``"coint"`` on
    #:                     XAU/XAG, 2016-2024) because the hedge ratio is
    #:                     re-estimated rather than assumed constant.
    #: ``"either"``      - Pass when *either* test rejects (default).
    coint_method: str = "either"
    #: Minimum |correlation| required for the FX pair (spec: r < -0.75).
    max_inverse_correlation: float = -0.75
    #: Z-score entry / exit / stop thresholds.
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 3.5
    #: Gross exposure deployed when a pair signal is live (split across legs).
    gross_weight: float = 0.30
    #: Hedge-ratio cap expressed as a multiple of the *price-scale ratio*
    #: ``mean(P_a) / mean(P_b)``.  A scale-relative bound is essential for pairs
    #: with very different price levels (XAUUSD ~ 2,600 vs XAGUSD ~ 30, i.e. a
    #: true beta near 85) while still catching degenerate OLS blow-ups.
    max_abs_beta_ratio: float = 10.0
    #: Absolute backstop applied on top of the scale-relative cap.
    max_abs_beta: float = 1.0e6


# --------------------------------------------------------------------------- #
# Strategy: adaptive momentum (active in State 1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MomentumConfig:
    """Hyper-parameters for the volatility-adjusted trend follower."""

    base_period: int = 20  # fast dynamic EMA base lookback
    slow_base_period: int = 50  # slow dynamic EMA base lookback
    vwap_base_period: int = 20
    rsi_period: int = 14
    rsi_long_threshold: float = 55.0
    rsi_short_threshold: float = 45.0
    #: ATR baseline used to rescale lookbacks: period_t = base * ATR_base / ATR_t
    atr_baseline_window: int = 100
    ema_min_period: int = 5
    ema_max_period: int = 200
    #: ATR trailing-stop multiple.
    atr_stop_multiple: float = 2.5
    #: Target gross weight per momentum instrument.
    gross_weight: float = 0.20


# --------------------------------------------------------------------------- #
# Capital preservation overlay (active in State 2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskPreservationConfig:
    """Behaviour of the capital-preservation engine when State 2 is detected."""

    #: Multiply every ATR trailing-stop distance by this factor (spec: 50%).
    stop_tightening_factor: float = 0.50
    #: Fraction of each open position liquidated on entering State 2.
    de_risk_fraction: float = 0.60  # spec: 50%-70%
    #: Halt all new entry orders while State 2 is active.
    halt_entries: bool = True
    #: Number of bars to keep entries halted after leaving State 2 (cool-down).
    cooldown_bars: int = 5


# --------------------------------------------------------------------------- #
# Portfolio / position sizing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SizingConfig:
    """Kelly, risk-parity and volatility-targeting controls."""

    initial_capital: float = 1_000_000.0
    #: Maximum gross leverage (sum of |weights|).
    max_gross_leverage: float = 1.50
    #: Per-instrument weight cap as a fraction of equity.
    max_symbol_weight: float = 0.40
    #: Annualised portfolio volatility target (0 disables vol targeting).
    target_volatility: float = 0.10
    #: Fractional Kelly applied to the raw Kelly optimum (0.5 = half Kelly).
    kelly_fraction: float = 0.50
    #: Hard cap on the Kelly-derived weight (guards fat-tailed estimates).
    kelly_cap: float = 0.40
    #: Minimum trades before a Kelly estimate is trusted.
    kelly_min_trades: int = 20
    #: Blend between risk-parity weights and raw signal weights (0..1).
    risk_parity_blend: float = 0.50
    #: Fraction of equity risked per ATR-based unit.
    risk_per_unit_pct: float = 0.01
    #: Rolling window for realised covariance used by the risk-parity solver.
    cov_window: int = 60
    #: Risk-parity solver iterations / tolerance.  The cyclical-coordinate-descent
    #: scheme converges in a handful of sweeps for typical 2-10 asset books.
    risk_parity_max_iter: int = 100
    risk_parity_tol: float = 1e-9


@dataclass(frozen=True)
class RiskConfig:
    """Risk-manager controls (stops, drawdown circuit breaker, exposure caps)."""

    atr_stop_multiple: float = 2.5
    #: Stop distance multiplier applied while State 2 is active.
    shock_stop_multiplier: float = 0.5
    #: Portfolio drawdown at which all positions are flattened (0 disables).
    max_drawdown_halt: float = 0.25
    #: Per-regime exposure scalars applied to strategy target weights.
    regime_exposure: Dict[int, float] = field(
        default_factory=lambda: {0: 1.00, 1: 1.00, 2: 0.00}
    )
    #: Maximum number of *new* positions that may be opened per calendar day.
    #: This is the frequency limit that demo mode bypasses - it is a throttle on
    #: order-flow churn, not a risk limit, so Kelly / risk-parity sizing and the
    #: ATR stop logic remain fully active either way.
    max_daily_trades: int = 20
    #: Minimum notional for an order to be sent (avoids dust).
    min_order_notional: float = 1_000.0
    #: Rebalance band: skip trades smaller than this fraction of equity.
    rebalance_band: float = 0.005


# --------------------------------------------------------------------------- #
# Execution costs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CostConfig:
    """Spread, slippage and commission model."""

    #: Extra spread (in pips) charged on top of each asset's quoted spread.
    spread_pips_override: float | None = None
    #: Commission as a fraction of traded notional (per side).
    commission_rate: float = 2.0e-5  # 0.002% ~ $2 per $100k
    #: Fixed commission per order (quote currency).
    commission_fixed: float = 0.0
    #: Borrow/financing cost, annualised, applied to short notional.
    financing_rate_annual: float = 0.02
    #: Fill model: ``"next_open"`` (default, no look-ahead) or ``"close"``.
    fill_at: str = "next_open"


# --------------------------------------------------------------------------- #
# Backtesting & walk-forward optimisation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BacktestConfig:
    """Backtest engine configuration."""

    start: str = "2015-01-01"
    end: str = "2025-12-31"
    interval: str = "1d"
    data_source: str = "auto"  # {"auto", "yfinance", "synthetic", "mt5", "ccxt"}
    fill_at: str = "next_open"
    allow_fractional: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward optimisation configuration."""

    is_months: int = 12
    oos_months: int = 3
    step_months: int = 3
    anchored: bool = False
    min_bars: int = 252
    #: Z-score entry thresholds swept on in-sample data (spec: 1.5 -> 2.5 @ 0.1).
    z_entry_grid: Tuple[float, ...] = tuple(round(1.5 + 0.1 * k, 2) for k in range(11))
    #: Momentum lookback grid (fast EMA base, slow EMA base).
    momentum_grid: Tuple[Tuple[int, int], ...] = ((10, 30), (15, 40), (20, 50), (25, 60))
    #: Objective maximised on in-sample data.
    objective: str = "sharpe"
    #: Maximum tolerated performance degradation IS -> OOS (spec: 15%).
    max_degradation_pct: float = 15.0
    #: Minimum OOS Sharpe for a segment's parameters to be accepted.
    min_oos_sharpe: float = -1.0


# --------------------------------------------------------------------------- #
# Live execution (MT5 / FIX)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BrokerConfig:
    """Live-execution configuration for the MT5 and FIX adapters."""

    mode: str = "simulated"  # {"simulated", "mt5", "fix", "none"}
    #: MetaTrader 5 terminal credentials (never hard-code in production).
    mt5_login: int | None = None
    mt5_password: str | None = None
    mt5_server: str | None = None
    mt5_terminal_path: str | None = None
    mt5_magic: int = 990_101
    mt5_deviation_points: int = 20
    #: FIX 4.4 session parameters.
    fix_host: str = "127.0.0.1"
    fix_port: int = 9_878
    fix_sender_comp_id: str = "QUANTSYS"
    fix_target_comp_id: str = "BROKER"
    fix_username: str | None = None
    fix_password: str | None = None
    fix_heartbeat_seconds: int = 30
    fix_use_tls: bool = False
    #: FIX-over-WebSocket market-data endpoint (optional).
    ws_marketdata_url: str | None = None
    #: Maximum orders per second sent to the broker (rate limiting).
    max_orders_per_second: float = 5.0
    #: Order retry policy.
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5


# --------------------------------------------------------------------------- #
# Composed settings object
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Demo (unlimited paper-trading) mode
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DemoConfig:
    """Controls the unlimited demo / paper-trading mode.

    Demo mode exists so a strategy can be exercised at full order flow without
    an artificial throttle.  Critically it removes **only the trade-frequency
    cap**: every genuine risk control - Kelly sizing, risk-parity allocation,
    ATR trailing stops, per-symbol and aggregate exposure caps, the drawdown
    circuit breaker and the State-2 preservation overlay - stays fully active.
    """

    #: Master switch for demo mode.
    enabled: bool = True
    #: Remove the ``RiskConfig.max_daily_trades`` ceiling while demo is on.
    unlimited_trades: bool = True
    #: Banner text surfaced in reports so a demo run is never mistaken for live.
    label: str = "UNLIMITED DEMO"


#: Module-level switches, mirroring the spec, for callers that do not thread a
#: :class:`Settings` object through.
DEMO_MODE: Final[bool] = True
UNLIMITED_DEMO_TRADES: Final[bool] = True


# --------------------------------------------------------------------------- #
# Trade notifier (desktop toast + voice)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NotifierConfig:
    """Configuration for :class:`quant_system.utils.notifier.NotifierEngine`.

    Attributes:
        enabled: Master switch for the notifier.
        toast_enabled: Show desktop toast popups.
        toast_backend: ``"auto"`` (PyQt6 -> tkinter -> null), ``"pyqt"``,
            ``"tk"`` or ``"none"``.
        toast_duration_ms: Auto-close delay for each toast.
        max_concurrent_toasts: Toasts kept on screen at once.
        voice_enabled: Speak trade summaries (opt-in via ``--enable-voice``).
        voice_mode: ``"speak"`` (live audio), ``"file"`` (synthesise to WAV, used
            for headless verification) or ``"off"``.
        voice_rate: Words per minute.
        voice_volume: Master volume in ``[0, 1]``.
        voice_output_dir: Where ``voice_mode="file"`` writes WAV files.
        notify_on_partial: Emit an alert on ``PARTIALLY_FILLED`` reports too.
        queue_maxsize: Backlog bound for the speech queue (drops when full).
        dedupe: Suppress duplicate alerts for the same order id.
    """

    enabled: bool = True
    toast_enabled: bool = True
    toast_backend: str = "auto"
    toast_duration_ms: int = 6_000
    max_concurrent_toasts: int = 3
    voice_enabled: bool = False
    voice_mode: str = "speak"
    voice_rate: int = 170
    voice_volume: float = 1.0
    voice_output_dir: Path = VOICE_DIR
    notify_on_partial: bool = False
    queue_maxsize: int = 256
    dedupe: bool = True

    def with_voice(self, enabled: bool = True, mode: Optional[str] = None) -> "NotifierConfig":
        """Return a copy with the voice settings changed.

        Args:
            enabled: Whether speech is on.
            mode: Optional override for ``voice_mode``.

        Returns:
            A new :class:`NotifierConfig`.
        """
        from dataclasses import replace

        return replace(
            self,
            voice_enabled=enabled,
            voice_mode=mode if mode is not None else self.voice_mode,
        )


@dataclass(frozen=True)
class Settings:
    """Aggregate settings container threaded through the whole system."""

    universe: UniverseConfig = field(default_factory=UniverseConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    stat_arb: StatArbConfig = field(default_factory=StatArbConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    preservation: RiskPreservationConfig = field(default_factory=RiskPreservationConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    notifier: NotifierConfig = field(default_factory=NotifierConfig)

    @property
    def symbols(self) -> Tuple[str, ...]:
        """Canonical symbols of the configured universe."""
        return self.universe.symbols

    @property
    def is_demo(self) -> bool:
        """Whether demo (unlimited paper-trading) mode is active."""
        return bool(self.demo.enabled)

    @property
    def unlimited_demo_trades(self) -> bool:
        """Whether the daily trade-frequency cap is lifted."""
        return bool(self.demo.enabled and self.demo.unlimited_trades and UNLIMITED_DEMO_TRADES)


#: Module-level default settings instance.
DEFAULT_SETTINGS: Final[Settings] = Settings()

#: Symbols whose quotes are expressed with non-USD conventions, used by the
#: synthetic data generator to keep fallback prices realistic.
SYNTHETIC_SEED: Final[int] = 20_240_101

#: Regime-dependent strategy routing table: regime id -> strategy names.
REGIME_ROUTING: Final[Dict[int, Tuple[str, ...]]] = {
    STATE_RANGE_BOUND: ("MetalsStatArb", "FXStatArb"),
    STATE_TREND: ("AdaptiveMomentum",),
    STATE_SHOCK: ("RiskPreservation",),
}

__all__: List[str] = [
    "PROJECT_ROOT",
    "DATA_CACHE_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "VOICE_DIR",
    "LOGS_DIR",
    "HALT_FILE",
    "TRADING_DAYS_PER_YEAR",
    "RISK_FREE_RATE",
    "N_REGIMES",
    "STATE_RANGE_BOUND",
    "STATE_TREND",
    "STATE_SHOCK",
    "REGIME_LABELS",
    "REGIME_ROUTING",
    "SYNTHETIC_SEED",
    "AssetSpec",
    "UniverseConfig",
    "HMMConfig",
    "StatArbConfig",
    "MomentumConfig",
    "RiskPreservationConfig",
    "SizingConfig",
    "RiskConfig",
    "CostConfig",
    "BacktestConfig",
    "WalkForwardConfig",
    "BrokerConfig",
    "DemoConfig",
    "NotifierConfig",
    "Settings",
    "DEFAULT_SETTINGS",
    "DEMO_MODE",
    "UNLIMITED_DEMO_TRADES",
    "VOICE_DIR",
]
