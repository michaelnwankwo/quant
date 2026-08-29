# `quant_system` — HMM Regime-Switchboard Quantitative Trading & Backtesting Framework

A modular, production-grade Python framework combining **Hidden-Markov-Model regime
detection**, **multi-asset statistical arbitrage**, **volatility-adaptive momentum**
and **walk-forward optimisation**, with a live-execution layer for **MetaTrader 5**
and **FIX 4.4**.

```text
Market data ──▶ Feature engineering ──▶ HMM Switchboard ──▶ Regime router
                                                              │
                        ┌─────────────────────────────────────┼────────────────────────────┐
                        ▼                                     ▼                            ▼
                 State 0: Range-bound               State 1: Strong trend       State 2: Market shock
                 Statistical arbitrage             Adaptive momentum            Capital preservation
                 (XAU/XAG, EURUSD/USDCHF)          (USDJPY, XAUUSD)             (halt / tighten / de-risk)
                        └─────────────────────────────────────┼────────────────────────────┘
                                                              ▼
                                        Sizing (Kelly + Risk Parity + Vol target)
                                                              ▼
                                    Risk manager (ATR stops, exposure caps, kill switch)
                                                              ▼
                                       Event-driven engine ──▶ Analytics & WFO
```

---

## 1. Quickstart

```bash
pip install -r requirements.txt
python scripts/preflight.py --speak   # verify audio, toasts, MT5 and data feeds
```
# Backtest on real data (yfinance, auto-falls back to synthetic if offline)
python main.py --mode backtest --source auto --start 2016-01-01 --end 2024-12-31

# Fully offline, deterministic run
python main.py --mode backtest --source synthetic --no-cache

# Backtest + walk-forward optimisation
python main.py --mode both --source auto --wfo-grid fast

# Paper-trade through the simulated broker (same code path as live)
python main.py --mode live --broker simulated --max-iterations 5

# Live via MetaTrader 5 (Windows) or FIX 4.4
python main.py --mode live --broker mt5
python main.py --mode live --broker fix

# Paper-trade WITH spoken + desktop alerts for every confirmed fill
python main.py --mode live --broker simulated --demo --enable-voice \
               --voice-mode file --toast-backend pyqt

# End-to-end pipeline verification (all five stages, 75 checks)
python main.py --mode verify
```

> **Deploying to a live account?** Follow [`RUNBOOK.md`](RUNBOOK.md) — it covers
> MT5 terminal configuration (Algo Trading, Market Watch, broker symbol aliasing),
> pre-flight checks, demo→live promotion, and the emergency kill-switch procedures.

Artifacts are written to `reports/`:

| File | Contents |
| --- | --- |
| `backtest_summary.txt` | Headline metrics + regime breakdown + transition matrix |
| `backtest_equity.csv` | Equity curve |
| `backtest_trades.csv` | Round-trip trade log with exit reasons |
| `backtest_weights.csv` | Per-symbol target weights per bar |
| `backtest_regime_metrics.csv` | Performance conditioned on HMM state |
| `figures/backtest_equity.png` | Equity + drawdown with regime shading |
| `walk_forward_report.txt` | Per-segment IS→OOS degradation table |
| `walk_forward_oos_equity.csv` | Stitched out-of-sample equity curve |

Run the test suite:

```bash
python -m pytest -q             # 73 tests, ~48 s (72 passed, 1 conditional skip)
python tests/verify_system.py   # or: python main.py --mode verify
```

---

## 2. Project layout

```text
quant_system/
├── config/
│   ├── __init__.py
│   └── settings.py              # Asset universe, HMM/stat-arb/momentum/risk/cost hyperparams
├── data/
│   ├── __init__.py
│   ├── ingestion.py             # yfinance / MT5 / CCXT fetchers + synthetic generator + cache
│   └── preprocessing.py         # log returns, Wilder ATR, realised vol, z-scores, cointegration
├── models/
│   ├── __init__.py
│   └── hmm_switchboard.py       # 3-state GaussianHMM + canonical regime mapping + causal streamer
├── strategies/
│   ├── __init__.py
│   ├── base.py                  # Signal / StrategyContext / BaseStrategy (ABC)
│   ├── stat_arb.py              # Pairs trading — active in State 0
│   ├── adaptive_momentum.py     # Volatility-scaled trend following — active in State 1
│   └── risk_preservation.py     # Capital preservation overlay — active in State 2
├── execution/
│   ├── __init__.py
│   ├── portfolio.py             # Position/PnL accounting + Kelly + Risk-Parity sizing
│   ├── risk_manager.py          # ATR trailing stops, unit sizing, drawdown circuit breaker
│   └── brokers/
│       ├── base.py              # BrokerBase, Order, OrderReport, PositionReport
│       ├── simulated_broker.py  # In-process adapter (paper trading)
│       ├── mt5_broker.py        # MetaTrader 5 adapter (Windows)
│       ├── fix_broker.py        # FIX 4.4 initiator + WebSocket market data
│       └── router.py            # Weight → order translation, rate limiting, retries
├── backtesting/
│   ├── __init__.py
│   ├── engine.py                # Event-driven engine + vectorized engine
│   └── walk_forward.py          # Rolling IS/OOS optimisation with curve-fitting guard
├── analytics/
│   ├── __init__.py
│   └── metrics.py               # Sharpe/Sortino/Calmar/MaxDD/Expectancy + regime attribution
├── utils/
│   ├── __init__.py
│   └── notifier.py              # Fill-only desktop toasts + threaded pyttsx3 voice alerts
├── scripts/
│   └── preflight.py             # Environment check: audio, toasts, MT5, data feeds
├── tests/
│   ├── test_hmm.py              # Regime classification, causality, determinism
│   ├── test_stat_arb.py         # Stationarity, hedge ratios, z-score guards, state machine
│   ├── test_backtest.py         # Execution pipeline, sizing, WFO, backtrader cross-check
│   └── verify_system.py         # End-to-end pipeline verification (5 stages, 75 checks)
├── conftest.py                  # sys.path bootstrap for pytest
├── pytest.ini                   # Collects verify_system.py; warns-as-errors
├── main.py                      # Orchestrator & CLI entrypoint
├── RUNBOOK.md                   # Setup, MT5 config, live/demo operations, kill switches
└── requirements.txt
```

*Additions beyond the base specification* (requested as extensions): `execution/brokers/`
(MT5 + FIX live execution), Kelly & Risk-Parity sizing inside `execution/portfolio.py`,
`utils/notifier.py` (trade alerts), `tests/verify_system.py` (end-to-end verification)
and `conftest.py`.

---

## 3. Module specification

### 3.1 `data/preprocessing.py`

| Function | Definition |
| --- | --- |
| `log_returns` | $r_t = \ln(P_t / P_{t-1})$ |
| `wilder_atr` | $TR_t=\max(H_t-L_t,\|H_t-C_{t-1}\|,\|L_t-C_{t-1}\|)$, SMA-seeded then Wilder RMA: $ATR_t=\big(ATR_{t-1}(p-1)+TR_t\big)/p$ |
| `realized_volatility` | $\sigma_t=\sqrt{252}\;\sqrt{\tfrac{1}{N-1}\sum_{i}(r_i-\bar r)^2},\;N=20$ |
| `standardize_features` | `sklearn.preprocessing.StandardScaler` on $X_t$ |
| `rolling_zscore` | Rolling z-score with a **zero-variance guard** (returns exactly `0.0` instead of `inf`) and a configurable clip |
| `rolling_hedge_ratio` | Rolling OLS slope $\beta=Cov(y,x)/Var(x)$ with a **scale-relative** cap |
| `rolling_cointegration` | Rolling Engle–Granger via `statsmodels.tsa.stattools.coint` |
| `rolling_adf_pvalue` | Rolling ADF — Engle–Granger *step 2* on the traded spread residual |
| `dynamic_ema` | Variable-period EMA: $\alpha_t = 2/(\text{period}_t+1)$, $EMA_t=\alpha_t P_t+(1-\alpha_t)EMA_{t-1}$ |
| `rolling_vwap` | Variable-window VWAP via O(1) prefix sums |

### 3.2 `models/hmm_switchboard.py`

Feature vector $X_t = [r_t,\;ATR_t,\;\sigma_t]^\top$ built as a **cross-sectional market
composite** (equal-weighted across the universe; ATR is normalised by price so gold and
FX crosses are dimensionally comparable).

`GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)`.

`hmmlearn` emits arbitrary state ids, so after every fit the states are re-labelled onto
a stable economic ordering:

| State | Label | Identification rule |
| --- | --- | --- |
| **0** | Low-Volatility / Range-Bound | Lowest return variance |
| **1** | High-Momentum / Strong Trend | Remaining state with the higher directional persistence $\lvert\mu\rvert/\sigma$ |
| **2** | High-Volatility / Market Shock | Remaining state with the fatter tails (excess kurtosis) |

**Look-ahead control.** `CausalRegimeStreamer` produces the regime path used by the
backtester. At every bar $t$ it fits (or reuses) a model on a trailing window ending at
$t$ and decodes the state at $t$. No future observation enters a fit or a decode —
verified by `test_causal_stream_has_no_lookahead`, which truncates the input and asserts
that every past label is bit-identical.

### 3.3 `strategies/stat_arb.py` — active in **State 0**

**XAUUSD / XAGUSD.** Rolling Engle–Granger cointegration, rolling OLS hedge ratio
$\beta$, spread $S_t = P_{XAU,t} - \beta_t P_{XAG,t}$, z-score
$Z_t = (S_t - \mu_S)/\sigma_S$.

**EURUSD / USDCHF.** Log-price spread with a rolling OLS hedge ratio (recovers
$\beta\approx-1$), gated on rolling return correlation $r < -0.75$.

| Rule | Trigger |
| --- | --- |
| Long spread | enter $Z_t \le -2.0$, exit $Z_t \ge 0.0$ |
| Short spread | enter $Z_t \ge +2.0$, exit $Z_t \le 0.0$ |
| Stop loss | hard exit on $\lvert Z_t\rvert \ge 3.5$ (cointegration breakdown) |

Leg weights are allocated so the book holds the spread itself: $k$ units of leg A and
$-k\beta$ units of leg B give $PnL = k\,dS$, with $k$ chosen so $\lvert w_a\rvert+\lvert w_b\rvert$
equals the configured gross exposure.

**Stationarity gate (`--coint-method`).** Measured on XAU/XAG, 2016‑2024:

| Method | Bars passing the gate |
| --- | --- |
| `coint` — Engle–Granger on the price series (spec-literal) | **5.7 %** |
| `adf_spread` — ADF on the rolling-$\beta$ spread residual | **17.9 %** |
| `either` — **default** | **22.2 %** |

`adf_spread` is Engle–Granger *step 2* applied to the exact series being traded and has
roughly 3× the power, because the hedge ratio is re-estimated rather than assumed
constant. The default `either` is a superset of the specified test; pass
`--coint-method coint` for strict specification behaviour.

### 3.4 `strategies/adaptive_momentum.py` — active in **State 1**

Every lookback is rescaled by volatility:

$$\text{EMA\_Period}_t = \text{round}\!\left(\text{Base\_Period}\times\frac{ATR_{baseline}}{ATR_t}\right)$$

| Signal | Condition |
| --- | --- |
| Long | $P > VWAP_{dyn}$ **and** $EMA_{dyn}(20) > EMA_{dyn}(50)$ **and** $RSI(14) > 55$ |
| Short | $P < VWAP_{dyn}$ **and** $EMA_{dyn}(20) < EMA_{dyn}(50)$ **and** $RSI(14) < 45$ |
| Exit | ATR trailing stop at $2.5\times ATR$, or trend inversion (EMA cross / VWAP cross) |

### 3.5 `strategies/risk_preservation.py` — active in **State 2**

On entering State 2 the overlay:

1. **halts all new entry orders** across every strategy;
2. **tightens ATR trailing stops by 50 %** ($2.5\times ATR \to 1.25\times ATR$);
3. **partially liquidates** open positions (default 60 %, inside the specified 50–70 % band);

with a configurable post-shock cool-down (default 5 bars) so the book does not re-enter
the tail of a volatility burst.

### 3.6 `execution/portfolio.py` — Kelly & Risk Parity

The sizing pipeline composes four mechanisms, in order:

1. **Regime scaling** — `RiskConfig.regime_exposure` (State 2 contributes `0.0`).
2. **Volatility targeting** — the book is scaled so $\sqrt{w^\top\Sigma w}$ matches `target_volatility`.
3. **Kelly** — $f^\* = (p\,b - q)/b$ with $b = \text{avg win}/\text{avg loss}$ from the realised
   trade log (or $f^\* = \mu/\sigma^2$ from returns). A **fractional Kelly** (default ½)
   caps aggregate gross exposure, because the inputs are estimates.
4. **Risk parity / ERC** — weights are re-distributed so every instrument contributes an
   equal share of portfolio variance, then blended with the raw signal weights by
   `risk_parity_blend`. Solved with the cyclical-coordinate-descent scheme of
   Spinu / Griveau-Billion; `test_risk_parity_equalises_risk_contributions` asserts the
   risk contributions match to $10^{-6}$.

### 3.7 `backtesting/engine.py`

Per-bar ordering — every step can only observe data available at that instant:

```text
bar i: 1. fill orders queued at bar i-1's close, at bar i's OPEN
       2. intrabar ATR stop surveillance vs bar i HIGH/LOW (gaps fill at the OPEN)
       3. mark to market at bar i's CLOSE
       4. ratchet ATR trailing stops (multiple scaled by the active regime)
       5. portfolio drawdown circuit breaker
       6. capital-preservation overlay (halt / tighten / de-risk)
       7. strategy signals from data up to bar i's CLOSE
       8. sizing → orders queued for bar i+1's OPEN
       9. record equity at bar i's CLOSE
```

Costs are explicit and never double-counted: half-spread + slippage + commission are
booked as a cash debit *and* attributed to the trade record, so
`net_pnl = gross_pnl - costs` holds exactly (asserted in the integration test).

`VectorizedBacktester` is a ~100× faster closed-form engine for precomputed weight
frames, used for sanity checks and rapid exploration.

**Independent cross-validation.** `compare_engines_on_sma_cross()` runs the same
SMA-crossover signal through our vectorized engine *and* through **backtrader** — a
third-party engine with its own broker, data feed and order machinery — and asserts
the two return streams agree. Measured on 2016‑2024 daily data:

| Symbol | backtrader | vectorized | return correlation |
| --- | --- | --- | --- |
| USDJPY | +32.9 % | +30.9 % | 0.973 |
| XAUUSD | +38.6 % | +34.7 % | 0.982 |
| XAGUSD | +36.7 % | +22.9 % | 0.976 |
| EURUSD | +2.6 % | +5.3 % | 0.927 |
| USDCHF | −1.6 % | −4.7 % | 0.979 |

Residual differences are expected (backtrader uses integer share sizing and commits a
percentage of cash at entry; our engine rebalances to its target weight every bar), but
a shared bug across two independent implementations is far less likely than a shared
result.

### 3.8 `backtesting/walk_forward.py`

Rolling 12-month in-sample / 3-month out-of-sample windows. The **production
event-driven engine** (not a surrogate) sweeps the grid in-sample, the winner is frozen
and re-run out-of-sample, and the OOS return streams are stitched into one continuous
curve — the only curve free of parameter-selection bias.

Degradation $\;= (IS - OOS)/\lvert IS\rvert \times 100$ is reported per segment and a
segment is *accepted* when it stays under `max_degradation_pct` (15 %).

> **Performance note.** Prepared indicators are reused across the Z-threshold sweep
> (only momentum lookbacks change the indicators), and metric computation is skipped
> during the in-sample sweep. Together this is a ~12× speedup; the full 44-combination
> specification grid runs in roughly 12 minutes over a 9-year daily sample.

### 3.9 Live execution

| Adapter | Notes |
| --- | --- |
| `SimulatedBroker` | In-process fills with the same spread/slippage/commission model. |
| `MT5Broker` | Windows-only (drives the desktop terminal over IPC). Symbol-name resolution with variant probing (`XAUUSD`, `XAUUSD.m`, `GOLD`…), filling-mode fallback IOC→FOK→RETURN, magic-number tagging, partial close. Degrades to `MT5UnavailableError` rather than `ImportError`. |
| `FIXBroker` | Real FIX 4.4 session: logon handshake, sequence numbers, `Heartbeat`/`TestRequest` handling, `NewOrderSingle`/`OrderCancelRequest`, `ExecutionReport` correlation by `ClOrdID`, correct `BodyLength`/`CheckSum` framing. Runs its own `asyncio` loop on a daemon thread behind a synchronous API. Optional FIX-over-WebSocket market data. |
| `OrderRouter` | Weight→delta orders, dust filtering, token-bucket rate limiting, bounded retries with exponential backoff, idempotent `ClOrdID`s per rebalance batch. |

---

## 4. Robustness & guards

| Risk | Guard |
| --- | --- |
| Look-ahead bias | Causal regime streamer; signals on close → fills at next open; covariance calibrated on the head of the sample only; `lag_features()` helper |
| Zero-variance z-scores | `rolling_zscore` returns `0.0` where $\sigma < \epsilon$ and clips spikes |
| Degenerate regressions | `ols_hedge_ratio` returns `(nan, nan, nan)`; `engle_granger_test` returns `(nan, nan)` on constant/short samples |
| Non-stationary pairs | Cointegration/ADF/correlation gates suppress entries; `invalid_exit` closes an open pair when the relationship breaks |
| Hedge-ratio blow-ups | **Scale-relative** cap (`max_abs_beta_ratio × P_a/P_b`) — an absolute cap silently truncates the XAU/XAG $\beta\approx35$–$85$ |
| Order-book churn | Rebalance band, dust filter, minimum notional |
| Runaway leverage | Per-symbol cap, aggregate gross cap, Kelly cap, vol target |
| Catastrophic loss | ATR trailing stops, 50 % stop tightening in State 2, drawdown circuit breaker |
| Broker flakiness | Rate limiting, bounded retry with backoff, `OrderReport` status instead of exceptions |
| Stale cached data | Cache filenames encode the resolved source; real data always wins over a synthetic fallback |

---

## 5. Configuration

Every tunable lives in `config/settings.py` as frozen dataclasses. Key knobs:

| Setting | Default | Meaning |
| --- | --- | --- |
| `HMMConfig.train_window` / `refit_every` | `504` / `21` | Trailing fit window and refit cadence (bars) |
| `StatArbConfig.window` / `zscore_window` | `90` / `60` | Hedge-ratio OLS window / spread z-score window |
| `StatArbConfig.entry_z` / `exit_z` / `stop_z` | `2.0` / `0.0` / `3.5` | Z-score thresholds |
| `StatArbConfig.coint_method` | `either` | `coint` \| `adf_spread` \| `either` |
| `MomentumConfig.atr_stop_multiple` | `2.5` | ATR trailing-stop distance |
| `RiskPreservationConfig.de_risk_fraction` | `0.60` | Fraction liquidated on entering State 2 |
| `SizingConfig.target_volatility` | `0.10` | Annualised portfolio vol target |
| `SizingConfig.kelly_fraction` / `kelly_cap` | `0.5` / `0.40` | Fractional Kelly and hard cap |
| `SizingConfig.risk_parity_blend` | `0.50` | Blend between raw and ERC weights |
| `WalkForwardConfig.max_degradation_pct` | `15.0` | Curve-fitting tolerance (IS→OOS) |
| `RiskConfig.max_daily_trades` | `20` | New positions per day (**lifted in demo mode**) |
| `DemoConfig.enabled` / `unlimited_trades` | `True` / `True` | Unlimited paper-trading mode |
| `NotifierConfig.voice_enabled` / `voice_mode` | `False` / `speak` | `speak` \| `file` (WAV) \| `off` |
| `NotifierConfig.toast_backend` | `auto` | `auto` \| `pyqt` \| `tk` \| `none` |
| `NotifierConfig.toast_duration_ms` | `6000` | Auto-close delay for the toast |

---

## 6. Reference results

Daily bars, 2016‑01‑01 → 2024‑12‑31, $1{,}000{,}000$ notional, unoptimised parameters.

```
Regime distribution : State 0  44.4 %   State 1  22.1 %   State 2  33.5 %
Final equity        : 997,406   (-0.26 %)
Sharpe / Sortino    : -0.047 / -0.013     Max drawdown : 2.12 %
Trades              : 44    Win rate 52.3 %   Profit factor 1.019
Avg holding period  : 7.5 bars           Exposure : 6.2 % of bars
```

Turnover is governed by `RiskConfig.rebalance_band` (0.5 %): the book is only
re-sized when the strategy layer actually changes a target, so it is allowed to
drift with the market in between. Setting the band to `0` re-sizes on every bar
and produces 191 trades / 23.0 % exposure / -0.23 % — the same order of
magnitude of P&L, at roughly 4x the transaction cost.

These are **honest baseline numbers, not tuned results** — transaction costs are charged
in full and no parameter was fitted. The walk-forward study is the tool for that:

```bash
python main.py --mode both --wfo-grid fast --start 2016-01-01 --end 2024-12-31
```

```
Segments            : 31                 Acceptance rate  : 51.6 %
Stitched OOS equity : 1,017,821  (+1.78 %)   Max drawdown : 2.52 %
```

The walk-forward result is reproduced **bit-for-bit** across runs, which is the
regression tripwire for the whole strategy -> sizing -> engine -> metrics chain.

---

## 7. Trade notifications (`utils/notifier.py`)

`NotifierEngine` turns *confirmed fills* into a desktop toast and a spoken alert.
It is deliberately conservative about what counts as a fill.

**Fill gate.** Only `OrderReport.status == FILLED` (or `mt5.TRADE_RETCODE_DONE`)
is announced. Raw ticks, routine HMM state checks, pending acknowledgements,
unfilled limit orders, partial fills, cancellations and rejections are all
filtered out — see `NotifierEngine.is_trade_fill`. Repeats of the same
`broker_order_id` are suppressed.

**Desktop toast.** A frameless, always-on-top card, bottom-right, stacking up to
`max_concurrent_toasts`, fading in and auto-closing after `toast_duration_ms`:

```text
[BUY] XAUUSD
0.50 | 2,650.50 | SL 2,640.00 | TP 2,680.00 | State 1 Momentum
```

Backends resolve `auto -> PyQt6 -> tkinter -> null`, so a headless host degrades
to a null recorder instead of crashing. `PyQtToastBackend.capture()` renders a
toast to a PNG, which is how the UI path is proven without a display.

**Voice.** `VoiceWorker` is a `threading.Thread` fed by a bounded
`queue.Queue`; the `pyttsx3` engine is created *inside* the thread (it binds to
the thread that drives its run loop) and reused for its lifetime. `submit()` is
non-blocking: if the queue is full the alert is dropped and counted, never
blocked. `notify_fill` therefore returns in well under a millimetre — measured
worst case **0.06 ms** over a 12-order burst — so the asyncio execution loop is
never stalled. Prices are spoken digit-wise so `2650.50` reads as
*"twenty-six fifty point five zero"*, not "two thousand six hundred and fifty":

```text
"Bought 0 point 50 lots of Gold at 2650 point 50 under State 1 Momentum, stop 2640 point 00."
"Sold 1 point 25 lots of Euro Dollar at 1 point 08 50 under State 2 Shock, realised loss 420 point 00."
```

`--voice-mode file` writes WAV clips to `reports/voice/` (headless-safe and the
default for CI); `speak` plays through the default device.

---

## 8. Unlimited demo mode

Demo mode lifts the **daily trade-frequency cap only**. Position sizing (Kelly /
risk parity) and the ATR stop-loss machinery are completely untouched — the demo
path runs the identical `apply_fill` code, so any P&L, stop or sizing behaviour
you validate in demo mode is the behaviour you get live.

| | Demo mode (default) | Live mode (`--no-demo`) |
| --- | --- | --- |
| `Portfolio.daily_trade_limit` | `-1` (unlimited) | `RiskConfig.max_daily_trades` (20) |
| New positions per day | unbounded | capped; further openings refused |
| Kelly / risk-parity sizing | **unchanged** | **unchanged** |
| ATR stop-loss | **unchanged** | **unchanged** |
| Adding to / trimming a position | never throttled | never throttled |

Only *openings* (flat -> non-flat) are counted, so scaling into an existing
position or closing one is never blocked; nor are risk-reducing stops or the
State 2 de-risk flow. Refused openings are logged as `trade_blocked` events and
surfaced via `BacktestEngine.blocked_trades`.

```bash
python main.py --mode live --demo                        # unlimited (default)
python main.py --mode live --no-demo --max-daily-trades 3
```

---

## 9. End-to-end verification (`tests/verify_system.py`)

`python main.py --mode verify` walks the whole production pipeline in order and
asserts each stage. Every stage runs inside `warnings.catch_warnings`, so any
warning the pipeline emits becomes a **failed check** — this is what enforces
the zero-warning bar.

| Stage | What is asserted |
| --- | --- |
| 1. Data ingestion & preprocessing | Universe fetched and aligned; log returns match `ln(P_t/P_t-1)`; Wilder ATR(14) reproduces the `(13·A_prev + TR)/14` recursion exactly; annualised vol equals sample std x sqrt(252); `StandardScaler` yields mean 0 / std 1 |
| 2. HMM switchboard decoding | 3 components, full covariance; `startprob_`/`transmat_` are distributions; canonical mapping orders states by variance (0 lowest -> 2 highest); dense in-range state path; probabilities sum to 1 after the warm-up prefix |
| 3. Strategy signals & sizing | Only the stat-arb book trades in State 0, only momentum in State 1, nothing opens in State 2; inactive strategies emit flat targets; Kelly = `p - (1-p)/(W/L)`; inverse-vol weights sum to 1; risk parity equalises contributions; ATR sizing and its 50 %-notional cap |
| 4. MT5 / simulated fallback | Real MT5 terminal attempted first; on failure (`MetaTrader5` is Windows-only) it falls back to `SimulatedBroker`; order lifecycle reaches FILLED; fills are booked; invalid orders rejected; demo vs live trade caps |
| 5. Notifier dispatch & audio | Only FILLED passes the gate; duplicates suppressed; toast payload and speech text; `notify_fill` never blocks; the voice thread synthesises one non-empty RIFF WAV per utterance with none dropped; `stop()` is idempotent |

Last run: **75/75 checks passed in ~11 s**.

Before a first deployment, `python scripts/preflight.py --speak` additionally
confirms the host-level subsystems the suite cannot fully exercise headlessly:
the TTS driver, the toast backend, MT5 terminal reachability and broker symbol
resolution.

---

## 10. Known limitations

* **Daily bars only have been validated end-to-end**; intraday ingestion is implemented
  but the annualisation factors and session handling have not been tuned.
* The HMM is refit on a *trailing* window, so regime labels are **relative to that
  window** — State 2 means "the most volatile third of recent history", not an absolute
  volatility level. This is inherent to walk-forward regime detection.
* `FIXBroker.get_positions()` returns fills seen in-session; full position
  reconciliation requires a venue-specific `PositionRequest` (`35=AN`).
* Kelly estimates are derived from the realised trade log and are therefore unreliable
  until `kelly_min_trades` (20) round trips have closed.
* The synthetic data generator is a *development* fallback. It reproduces regime
  clustering, a cointegrated metals pair and an inversely-correlated FX pair, but is not
  a calibrated market simulator.

---

## 11. License & disclaimer

Provided for research and educational purposes. Nothing here is investment advice.
Trading leveraged instruments carries substantial risk of loss; validate any strategy
against your own execution assumptions before risking capital.
