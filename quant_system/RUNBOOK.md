# `quant_system` — Operational Runbook & Setup Guide

**Audience:** the operator deploying this framework against a MetaTrader 5 account.
**Scope:** environment setup, MT5 terminal configuration, pre-flight verification,
unlimited demo (paper) trading, promotion to real capital, and daily monitoring.

> **Read this first.** `MetaTrader5` is a **Windows-only** package that drives the MT5
> desktop terminal over IPC. *Everything except live MT5 execution* — backtesting,
> walk-forward optimisation, the verification suite, the simulated broker, toasts and
> voice — runs unchanged on Linux and macOS. Plan for a **Windows host** for live
> trading and a **Linux/macOS host** for research.

**Terminology used below**

| Term | Meaning |
| --- | --- |
| **Demo mode** | `DEMO_MODE = True`. Unlimited new positions per day. Sizing, stops and P&L are **identical** to live. |
| **Live mode** | `DEMO_MODE = False`. `RiskConfig.max_daily_trades` (default 20) caps new positions per day. |
| **Paper broker** | `--broker simulated` — in-process fills, no venue. Same code path as live. |
| **Halt file** | `reports/HALT`. Creating it stops the live loop cleanly at the next cycle. |

---

## SECTION 1 — ENVIRONMENT & DEPENDENCY INITIALIZATION

### 1.1 Create and activate a clean virtual environment

**Windows (PowerShell)** — this is the host you need for live MT5:

```powershell
cd C:\path\to\quant_system
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Linux / macOS (bash):**

```bash
cd ~/quant_system
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

You should see `(venv)` in your prompt. Confirm the interpreter is the venv's:

```bash
python -c "import sys; print(sys.prefix); print(sys.executable)"
```

### 1.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs the mandatory stack (`numpy`, `pandas`, `scipy`, `statsmodels`,
`scikit-learn`, `hmmlearn`, `backtrader`, `yfinance`, `pytest`) plus the notification
stack (`pyttsx3`, `PyQt6`).

`MetaTrader5` is declared with an environment marker and installs **only on Windows**:

```
MetaTrader5>=5.0.45 ; platform_system == "Windows"
```

Install it explicitly if the marker was skipped:

```powershell
# Windows only
pip install MetaTrader5
```

### 1.3 Install OS-level system dependencies

The Python wheels for TTS and Qt still need native libraries.

**Windows** — nothing to install. `pyttsx3` uses the built-in **SAPI5** voice engine
and PyQt6 ships its own Qt DLLs.

**Linux (Debian/Ubuntu)** — required for voice alerts and desktop toasts:

```bash
# Speech synthesis backend used by pyttsx3
sudo apt-get update
sudo apt-get install -y espeak-ng libespeak1

# Playback (optional; 'file' mode writes WAVs and does not need it)
sudo apt-get install -y alsa-utils

# Qt/xcb runtime required by PyQt6 to create a window
sudo apt-get install -y \
  libxkbcommon0 libxkbcommon-x11-0 libdbus-1-3 libegl1 libfontconfig1 \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
  libsm6 libice6 libglib2.0-0t64 libxcb-xfixes0 libxcb-sync1 libxcb-render0 \
  libxcb-util1

# Headless servers: a virtual display so toasts can be rendered/tested
sudo apt-get install -y xvfb
```

**macOS:**

```bash
brew install espeak-ng
# Voice alerts use the `say` backend; toasts use PyQt6 (works natively on Aqua)
```

### 1.4 Verify the installation (audio + popup drivers)

A purpose-built checker ships with the project. Run it **immediately after install**:

```bash
python scripts/preflight.py            # human-readable
python scripts/preflight.py --json     # machine-readable
python scripts/preflight.py --speak    # also plays a test utterance aloud
```

Verified output on a headless Linux research host:

```
==============================================================================
  quant_system :: ENVIRONMENT PRE-FLIGHT
==============================================================================
  host    : Linux 6.1.158+ | Python 3.13.14 | display=no

  [ OK ] CORE: numpy 2.3.5, pandas 2.2.3, scipy 1.17.1, statsmodels 0.15.0,
         sklearn 1.6.1, hmmlearn 0.3.3, yfinance 1.7.0, matplotlib 3.10.9,
         backtrader 1.9.78.123
  [ OK ] AUDIO: 141 voices, WAV render 87614 bytes
  [ OK ] TOASTS: auto -> PyQtToastBackend (pyqt started)
            backends: {'pyqt': {'class': 'PyQtToastBackend', 'note': 'pyqt started'},
                      'tk': {'class': 'NullToastBackend', ... },
                      'none': {'class': 'NullToastBackend', ...}}
  [DEGR] MT5: not applicable - MetaTrader5 is Windows-only and this host is linux.
            hint: Use --broker simulated or --broker fix on this host.
  [ OK ] DATA: 5 symbols x 124 bars via ['yfinance']
            symbols: ['EURUSD', 'USDCHF', 'USDJPY', 'XAGUSD', 'XAUUSD']

==============================================================================
  READY: required subsystems (core, data) passed
==============================================================================
```

How to read it:

| Tag | Meaning |
| --- | --- |
| `[ OK ]` | Subsystem fully working. |
| `[DEGR]` | Working via a fallback, or not applicable on this host. Investigate the `hint`. |
| `[FAIL]` | Broken. Fix before trading. |

Exit code is `0` when the **required** subsystems (`CORE`, `DATA`) pass. `AUDIO`,
`TOASTS` and `MT5` degrade gracefully and never hard-fail the check — but a `[FAIL]`
on `AUDIO` means your voice alerts will be silent.

**Target state for a live Windows host:** all five checks `[ OK ]`, with `MT5`
reporting `trade_allowed: True`.

**Manual spot-checks** (if you prefer not to use the script):

```bash
# Audio: must print a voice count > 0
python -c "import pyttsx3; e=pyttsx3.init(); print('voices:', len(e.getProperty('voices')))"

# Audio: must write a non-empty WAV (runAndWait() is mandatory to flush it)
python -c "
import pyttsx3, pathlib
e = pyttsx3.init(); p = pathlib.Path('reports/voice/_test.wav')
p.parent.mkdir(parents=True, exist_ok=True)
e.save_to_file('audio check', str(p)); e.runAndWait()
print('wav bytes:', p.stat().st_size)"

# Toasts: must print 'PyQtToastBackend' (or Null* on a headless host)
python -c "
from quant_system.config import settings as cfg
from quant_system.utils.notifier import resolve_toast_backend
b, note = resolve_toast_backend('auto', cfg.DEFAULT_SETTINGS.notifier)
print(type(b).__name__, '|', note); b.stop()"
```

> **Headless hosts.** With no display, the toast backend falls back to a
> null recorder — alerts are logged but not shown. To exercise the real widget
> on a server, start a virtual display:
> ```bash
> Xvfb :99 -screen 0 1280x1024x24 &
> export DISPLAY=:99
> ```

---

## SECTION 2 — METATRADER 5 TERMINAL CONFIGURATION

> Windows only. The framework drives the **desktop terminal** via IPC, so the
> terminal must be running **and logged in** before you launch the bot.

### 2.1 Enable Algo Trading

1. Open the **MetaTrader 5** terminal and log in to your account.
2. **Toolbar method** — click the **"Algo Trading"** button in the main toolbar
   (it turns **green** when armed). Keyboard shortcut: **`Ctrl + E`**.
   - If you do not see it: **View → Toolbars → Standard**, or right-click the
     toolbar area and tick `Algo Trading`.
3. **Per-terminal setting** — press **`Ctrl + O`** (or **Tools → Options**), open the
   **Expert Advisors** tab, and tick:
   - ☑ **"Allow automated trading"**
   - ☑ **"Allow DLL imports"** (required: the `MetaTrader5` Python package is loaded
     as a native module)
   - ☑ **"Allow automated trading for all symbols"** *(optional but recommended —
     otherwise each new symbol needs re-approval)*
4. Click **OK**. The `Algo Trading` button must now be green and the status bar
   (bottom-right) must show a green **algo-trading** icon.

**Verify from Python** (the terminal must already be open):

```powershell
python -c "
import MetaTrader5 as mt5
print('init:', mt5.initialize())
ti = mt5.terminal_info()
print('trade_allowed:', ti.trade_allowed)
print('connected   :', ti.connected)
mt5.shutdown()"
```

`trade_allowed: True` is the requirement. `False` means step 2/3 is incomplete.

### 2.2 Add the traded symbols to Market Watch

The adapter can only trade symbols that are **visible** in Market Watch.

1. **`Ctrl + M`** (or **View → Market Watch**) to show the window.
2. **Right-click inside Market Watch → Symbols** (or **`Ctrl + U`**).
3. Expand the groups and **double-click** each required symbol — it turns
   **yellow/highlighted** when active:

   | Group | Symbol |
   | --- | --- |
   | Metals / Spot | `XAUUSD` (gold) |
   | Metals / Spot | `XAGUSD` (silver) |
   | Forex Majors | `EURUSD` |
   | Forex Majors | `USDCHF` |
   | Forex Majors | `USDJPY` |

4. If a symbol is missing from the list, in the *Symbols* dialog enable
   **"Show all symbols"** and search again — many brokers hide them until requested.
5. Close the dialog. Confirm all five appear in Market Watch with live bid/ask ticks.

### 2.3 Broker symbol aliasing (`XAUUSD.m`, `GOLD`, …)

Brokers rename instruments. `XAUUSD` may be `XAUUSD.m`, `XAUUSD.pro`, `GOLD`,
`XAUUSD#`, etc. There are **three layers** of handling, in order.

**Layer 1 — automatic variant fallback (no config change needed).**
`MT5Broker._resolve_symbol` tries, in order:

```python
SYMBOL_VARIANTS = ("", ".m", ".a", ".pro", ".raw", "_m", "#", ".i")
```

then, if all fail, **scans the terminal's full symbol table** for a name containing
your symbol. Most brokers need no configuration at all.

**Layer 2 — explicit mapping in `config/settings.py` (recommended).**
Set `mt5_symbol` to your broker's exact name. Edit the `UniverseConfig.assets` tuple:

```python
# config/settings.py  ->  class UniverseConfig
AssetSpec(
    symbol="XAUUSD",              # canonical name used everywhere internally
    yf_symbols=("XAUUSD=X", "GC=F"),
    mt5_symbol="XAUUSD.m",        # <-- your broker's name (was "XAUUSD")
    pip_size=0.01,
    contract_size=100.0,          # 100 troy oz per standard lot
    asset_class="metal",
    spread_pips=3.0,
    slippage_pips=0.5,
    vol_scale=1.00,
    base_price=2050.00,
),
```

Apply the same edit to each renamed instrument, e.g.:

| Canonical | Example broker names |
| --- | --- |
| `XAUUSD` | `XAUUSD.m`, `XAUUSD.pro`, `GOLD`, `XAUUSD#` |
| `XAGUSD` | `XAGUSD.m`, `XAGUSD.pro`, `SILVER` |
| `EURUSD` | `EURUSD.m`, `EURUSD.pro`, `EURUSD.raw` |
| `USDCHF` | `USDCHF.m`, `USDCHF.pro` |
| `USDJPY` | `USDJPY.m`, `USDJPY.pro` |

> **Important:** if your broker quotes gold on a **different contract size**
> (e.g. `GOLD` = 100 oz vs a mini = 10 oz), also update `contract_size` and
> `pip_size` — position sizing and stop distances depend on them.

**Layer 3 — verify.** On Windows with the terminal open:

```powershell
python -c "
import MetaTrader5 as mt5
from quant_system.config import settings as cfg
from quant_system.execution.brokers.mt5_broker import MT5Broker
mt5.initialize()
b = MT5Broker()
for spec in cfg.DEFAULT_SETTINGS.universe.assets:
    try:
        print(f'{spec.symbol:8s} -> {b._resolve_symbol(spec.symbol)}')
    except Exception as exc:
        print(f'{spec.symbol:8s} -> UNRESOLVED: {exc}')
mt5.shutdown()"
```

Every line must print a real broker symbol. `UNRESOLVED` means the symbol is
missing from Market Watch (Section 2.2) or `mt5_symbol` is wrong (Layer 2).

### 2.4 Account credentials

Set them in `config/settings.py` → `BrokerConfig`, or pass via environment variables
on the command line to keep secrets out of source control:

```python
# config/settings.py  ->  class BrokerConfig
mt5_login: Optional[int] = 12345678
mt5_password: Optional[str] = "your-password"
mt5_server: Optional[str] = "YourBroker-Demo"
mt5_terminal_path: Optional[str] = r"C:\Program Files\MetaTrader 5\terminal64.exe"
mt5_magic: int = 990101          # identifies this bot's orders on the account
mt5_deviation_points: int = 20   # max slippage from the requested price
```

> **Security:** prefer environment variables or a secrets manager over committing
> credentials. Anyone with the login can trade your account.

---

## SECTION 3 — SYSTEM PRE-FLIGHT VERIFICATION

Run **both** of these before placing a single order. Together they take ~60 seconds.

### 3.1 Unit and integration test suite

```bash
python -m pytest tests/ -q
```

Expected (this is the current, verified result):

```
...........................................................s............ [ 98%]
.                                                                        [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_stat_arb.py:370: The fixture did not produce a |z| >= 3.5 excursion.
72 passed, 1 skipped in 46.99s
```

- **72 passed** — HMM regime classification and causality, stat-arb stationarity and
  z-score state machine, execution pipeline, sizing, walk-forward, and the
  backtrader cross-validation.
- **1 skipped** — a conditional test that needs a `|z| >= 3.5` stop-loss excursion
  the fixture did not produce. Benign.
- **Zero warnings.** `pytest.ini` sets `filterwarnings = error`, so any warning
  becomes a failure. If you see `FAILED`, the build is not clean — do not deploy.

### 3.2 End-to-end pipeline verification

```bash
python tests/verify_system.py          # standalone
python main.py --mode verify           # via the CLI (also writes reports/verification_report.txt)
```

Expected tail:

```
==============================================================================
  RESULT: 75/75 checks passed in 11.33s   (data source: yfinance)
  artifact: /tmp/quant_verify_xxxxx/001_buy_xauusd.wav
==============================================================================
```

The five stages and what each proves:

| Stage | Checks | What it proves |
| --- | --- | --- |
| **1. Data Ingestion & Preprocessing** | 16 | Universe fetched/aligned; log returns; Wilder ATR(14) matches the `(13·A_prev + TR)/14` recursion exactly; annualised vol = sample std × √252; `StandardScaler` → mean 0 / std 1 |
| **2. HMM Switchboard Decoding** | 15 | 3 components, full covariance; `startprob_`/`transmat_` are distributions; states ordered by variance (0 lowest → 2 highest); dense in-range state path; probabilities sum to 1 |
| **3. Strategy Signal Generation & Sizing** | 15 | Correct regime routing; inactive strategies flat; Kelly = `p − (1−p)/(W/L)`; risk parity equalises contributions; ATR sizing + 50 % notional cap |
| **4. MT5 Connection / Simulated Fallback** | 16 | Real MT5 attempted first; clean fallback to `SimulatedBroker`; orders reach `FILLED`; fills booked; invalid orders rejected; **demo vs live trade caps** |
| **5. Trade Notifier Dispatch & Audio Queue** | 13 | Only `FILLED` passes the gate; duplicates suppressed; toast payload + speech text correct; `notify_fill` never blocks; **one non-empty RIFF WAV per utterance, none dropped** |

### 3.3 What "100 % operational" looks like

Sign off **only** when all of these are true:

| Subsystem | Green signal |
| --- | --- |
| **HMM models** | Stage 2 fully `[PASS]`; regime distribution printed, e.g. `State 0 44.4 % State 1 22.1 % State 2 33.5 %` |
| **Strategy switchboard** | Stage 3 `[PASS]`: State 0 → stat-arb only, State 1 → momentum only, State 2 → nothing; `75/75` overall |
| **MT5 connection** | Stage 4 logs `MT5 terminal connected`, **or** on a non-Windows host `MT5 terminal unreachable -> simulated fallback engaged` (expected, not a failure). On your Windows host it must say `connected`. |
| **Voice engine** | Stage 5 `[PASS]` with `spoken=13 failed=0 dropped=0 files=13` and each WAV > 1 kB with a `RIFF` header. Confirm audibly with `python scripts/preflight.py --speak`. |
| **Popup windows** | `PyQtToastBackend` selected in the preflight `TOASTS` line. On a headless host this is `NullToastBackend` — acceptable for research, **not** for an attended live desk. |
| **Zero errors** | `72 passed, 1 skipped` **and** `75/75` with no `FAILED` and no warnings. |

---

## SECTION 4 — RUNNING IN UNLIMITED DEMO / PAPER-TRADING MODE

### 4.1 Configure demo mode

`config/settings.py`:

```python
#: Module-level master switches (mirrored in DemoConfig below).
DEMO_MODE: bool = True
UNLIMITED_DEMO_TRADES: bool = True
```

and the frozen config block:

```python
@dataclass(frozen=True)
class DemoConfig:
    enabled: bool = True            # demo (paper) mode on
    unlimited_trades: bool = True   # lift the daily trade-frequency cap
    label: str = "UNLIMITED DEMO"
```

You can also toggle it per-run with **`--demo` / `--no-demo`** without editing files.

**What demo mode changes — and what it does not:**

| | Demo (default) | Live (`--no-demo`) |
| --- | --- | --- |
| `Portfolio.daily_trade_limit` | `-1` (unlimited) | `RiskConfig.max_daily_trades` (20) |
| New positions per day | unbounded | capped; further openings refused |
| **Kelly / risk-parity sizing** | **unchanged** | **unchanged** |
| **ATR stop-loss** | **unchanged** | **unchanged** |
| Adding to / trimming a position | never throttled | never throttled |

Only *openings* (flat → non-flat) are counted, so scaling into an existing position
or closing one is never blocked. Refused openings are logged as `trade_blocked`.

### 4.2 Generate the baseline equity reports

**Historical backtest:**

```bash
python main.py --mode backtest --source auto --start 2020-01-01 --end 2025-12-31
```

Verified output:

```
04:53:23 | INFO | quant_system | Loaded 5 symbols x 1508 bars from
                  {'XAUUSD': 'yfinance', 'XAGUSD': 'yfinance', 'EURUSD': 'yfinance',
                   'USDCHF': 'yfinance', 'USDJPY': 'yfinance'}
...
                    final_equity : 964,679.7096
                total_return_pct : -3.5320
                    sharpe_ratio : -0.3886
                   sortino_ratio : -0.1933
                    calmar_ratio : -0.1122
                max_drawdown_pct : 5.3446
                    exposure_pct : 28.7326
                      num_trades : 80
```

**Backtest + walk-forward optimisation:**

```bash
python main.py --mode both --source auto --wfo-grid fast
```

Verified output:

```
                    final_equity : 1,004,420.6608     # full-period backtest
                total_return_pct : 0.4421
                    sharpe_ratio : 0.0743
                max_drawdown_pct : 1.0877

Segments              : 39
Acceptance rate       : 38.5%
Mean degradation      : -22.58%

                    final_equity : 1,022,284.3721     # stitched out-of-sample
                total_return_pct : 2.3043
```

`reports/` now contains `backtest_summary.txt`, `backtest_equity.csv`,
`backtest_trades.csv`, `backtest_weights.csv`, `backtest_regime_metrics.csv`,
`walk_forward_report.txt`, `walk_forward_oos_equity.csv`,
`walk_forward_degradation.csv`, `walk_forward_parameters.csv` and
`figures/*.png`.

> **Note on `yfinance` 404s.** You may see
> `HTTP Error 404 ... Quote not found for symbol: XAUUSD=X`. This is **benign**:
> the ingestion fallbacks `XAUUSD=X → GC=F` and `XAGUSD=X → SI=F` resolve it. Only
> a message saying all tickers failed is a real problem.

### 4.3 Launch unlimited demo live mode with popups and voice

**Windows, MT5 demo account (the real target):**

```powershell
python main.py --mode live --broker mt5 --demo --enable-voice
```

**With explicit backend and WAV capture** (recommended for a first run — proves the
audio path even if speakers are muted):

```powershell
python main.py --mode live --broker mt5 --demo --enable-voice `
               --voice-mode file --toast-backend pyqt
```

**Linux/macOS (no MT5) — the identical strategy code path against the paper broker:**

```bash
python main.py --mode live --broker simulated --demo --enable-voice \
               --voice-mode file --toast-backend pyqt \
               --max-iterations 50 --poll-seconds 60
```

Useful companion flags:

| Flag | Purpose |
| --- | --- |
| `--enable-voice` | Turn on speech (off by default). |
| `--voice-mode {speak,file,off}` | `file` writes WAVs to `reports/voice/`; `speak` plays aloud. |
| `--toast-backend {auto,pyqt,tk,none}` | Force a toast backend. |
| `--no-toast` / `--no-plot` | Disable toasts / chart rendering. |
| `--max-iterations N` | Rebalance cycles before exiting. **`0` = run until stopped** (Ctrl+C / halt file). |
| `--poll-seconds S` | Sleep between cycles. |
| `--flatten-on-exit` | Flatten open positions on Ctrl+C / halt file (see §5.3). |
| `-v` | Debug logging. |

### 4.4 What you see and hear when a signal fires

Every rebalance cycle logs one line:

```
Cycle 12 | regime=1 | orders=1 filled=1
Cycle 13 | regime=1 | no orders (normal_conditions)
Cycle 14 | regime=2 | no orders (regime_state_2_shock)
```

`no orders (regime_state_2_shock)` is **correct behaviour** — State 2 halts all new
entries by design.

**On a confirmed fill**, three things happen at once:

1. **Log line:**
   ```
   04:19:14 | INFO | quant_system.utils.notifier | TRADE ALERT | [SELL] XAUUSD |
   0.17 | 11,582.42 | SL 12,075.28 | TP - | State 1 Momentum
   ```

2. **Desktop toast** — bottom-right, auto-closing after 6 s:
   ```
   ┌──────────────────────────────────────────────┐
   │ [SELL] XAUUSD                                │
   │ 0.17 | 11,582.42 | SL 12,075.28 | TP - |     │
   │ State 1 Momentum                             │
   └──────────────────────────────────────────────┘
   ```
   Format: `[ACTION] Symbol` then `Volume | Price | SL | TP | Active Regime State`.
   `TP -` is normal — the momentum strategy exits on a 2.5×ATR trailing stop, not a
   fixed target.

3. **Spoken alert** (prices read digit-wise so they are unambiguous):
   > *"Sold 0 point 17 lots of Gold at 11582 point 42 under State 1 Momentum,
   > stop 12075 point 28."*

   A closing fill adds the round trip:
   > *"Sold 1 point 25 lots of Euro Dollar at 1 point 08 50 under State 2 Shock,
   > realised loss 420 point 00."*

**Nothing fires for** pending acknowledgements, unfilled limit orders, partial fills,
cancellations, rejections, raw ticks, or routine HMM state checks. A repeat of the
same `broker_order_id` is suppressed as a duplicate.

---

## SECTION 5 — PROMOTING TO REAL CAPITAL (LIVE TRADING)

> **Safety first.** Complete at least one full demo session (Section 4) with alerts
> verified end to end before touching real capital.

### 5.1 Pre-promotion checklist

| # | Item | Command / action |
| --- | --- | --- |
| 1 | All tests green | `python -m pytest tests/ -q` → `72 passed, 1 skipped` |
| 2 | Pipeline verified | `python main.py --mode verify` → `75/75` |
| 3 | MT5 connected & `trade_allowed` | `python scripts/preflight.py` → `MT5 [ OK ]` |
| 4 | All symbols resolve | §2.3 Layer 3 script → no `UNRESOLVED` |
| 5 | **Account is the one you intend** | Check `login` + `server` in the preflight `terminal:` line |
| 6 | Backtest + WFO reviewed | §4.2 results understood and accepted |
| 7 | Alerts confirmed | Heard a spoken alert and seen a toast in demo |

### 5.2 Switch off demo mode

`config/settings.py`:

```python
DEMO_MODE: bool = False             # was True
UNLIMITED_DEMO_TRADES: bool = False # was True
```

and the frozen config:

```python
@dataclass(frozen=True)
class DemoConfig:
    enabled: bool = False           # live risk limits now enforced
    unlimited_trades: bool = False
    label: str = "LIVE"
```

Confirm before launching:

```bash
python -c "
from quant_system.config import settings as cfg
from quant_system.execution.portfolio import Portfolio
p = Portfolio(demo_config=cfg.DEFAULT_SETTINGS.demo)
print('demo_mode    :', p.demo_mode)
print('daily limit  :', p.daily_trade_limit)"
```

Expected: `demo_mode : False` and `daily limit : 20` (or your configured cap).

### 5.3 Launch command with explicit risk caps

```powershell
python main.py --mode live --broker mt5 --no-demo `
               --max-daily-trades 5 `
               --capital 25000 `
               --max-iterations 0 `
               --poll-seconds 300 `
               --enable-voice --voice-mode speak `
               --flatten-on-exit
```

| Flag | Effect |
| --- | --- |
| `--no-demo` | Enforce `max_daily_trades`. |
| `--max-daily-trades 5` | Cap new positions at 5/day (Layers of protection below). |
| `--capital 25000` | Starting equity for sizing (defaults to broker equity when omitted). |
| `--poll-seconds 300` | Rebalance every 5 minutes. |
| `--flatten-on-exit` | **Flatten on Ctrl+C or halt file** — see §5.4. |

**Defence-in-depth risk limits** (independent of the CLI):

| Layer | Setting | Default | Behaviour |
| --- | --- | --- | --- |
| Daily frequency | `RiskConfig.max_daily_trades` | 20 | Refuses new positions past the cap |
| Per-unit risk | `SizingConfig.risk_per_unit_pct` | 1 % | Equity at risk per unit |
| Stop distance | `RiskConfig.atr_stop_multiple` | 2.5 ×ATR | Trailing stop |
| Shock tightening | `RiskConfig.shock_stop_multiplier` | 0.5 | Stops halve in State 2 |
| Regime exposure | `RiskConfig.regime_exposure` | `{0:1.0, 1:1.0, 2:0.0}` | No exposure in State 2 |
| **Drawdown breaker** | `RiskConfig.max_drawdown_halt` | 25 % | **Auto-flattens the entire book** |

The drawdown circuit breaker is automatic: on breach it logs
`Drawdown halt triggered at <ts> (<dd>%)` and flattens everything.

### 5.4 Emergency circuit breakers (kill switch)

There are **four** independent ways to stop the bot. Use the first that fits.

---

**① Ctrl+C — graceful stop (attended terminal)**

Press `Ctrl+C` **once**. The current cycle finishes, then:

```
04:51:54 | WARNING | quant_system | SIGINT received - finishing the current cycle,
                     then shutting down cleanly. Send it again to abort right now.
04:51:54 | WARNING | quant_system | Shutdown requested (signal) - stopping before cycle 102.
04:51:54 | INFO    | quant_system | Shutdown requested (signal) - no open positions.
```

With `--flatten-on-exit` and open positions, it closes them first:

```
04:50:00 | WARNING | quant_system | Shutdown requested (halt_file) - stopping before cycle 14.
04:50:00 | WARNING | quant_system | Shutdown requested (halt_file) - flattening 1 position(s): ['XAUUSD']
04:50:00 | WARNING | quant_system | Flatten result: ['filled']
```

Exit code `0`. The inter-cycle sleep is re-checked every 250 ms, so a graceful
stop lands almost immediately even with `--poll-seconds 300`.

Press `Ctrl+C` **twice** to abort immediately:

```
CRITICAL | Second SIGINT received - aborting immediately. Positions are LEFT OPEN;
           flatten them manually.
```

Exit code `130`. **Positions are intentionally left open** — there is no time to
close them safely.

> **Timing caveat.** Two `Ctrl+C` presses delivered within a fraction of a second
> are merged into a *single* signal by the OS/Python, so you get the graceful stop
> instead of the abort — usually the better outcome. To force the abort, pause
> about a second between presses. If the process ignores signals entirely it is
> stuck in a native call; go to ③ (`SIGKILL`) and flatten manually.

---

**② Halt file — remote / unattended stop (recommended for production)**

Create the sentinel file; the bot stops at the next cycle boundary and exits `0`.
This works from **any shell**, including an SSH session that never touched the
bot's terminal:

```bash
touch reports/HALT      # Linux/macOS
```

```powershell
New-Item -ItemType File reports\HALT     # Windows
```

Same output as ① (with `halt_file` as the reason). With `--flatten-on-exit` it
flattens first.

> **Clean up:** the bot refuses to start while the file exists:
> ```
> ERROR | Halt file reports/HALT exists at start-up. Remove it before launching:
>         rm reports/HALT
> ```
> Remove it before restarting: `rm reports/HALT` (or `Remove-Item reports\HALT`).

---

**③ Kill the process — last resort**

```bash
# Linux/macOS
pkill -INT -f "main.py --mode live"    # graceful (same as one Ctrl+C)
sleep 10
pkill -TERM -f "main.py --mode live"   # SIGTERM (also graceful)
sleep 10
pkill -9   -f "main.py --mode live"    # SIGKILL - cannot flatten, positions stay open
```

```powershell
# Windows
Stop-Process -Name python -Force
```

`SIGKILL` / `-Force` cannot run cleanup. **Positions stay open** — go to the terminal
immediately.

---

**④ Manual MT5 emergency flatten**

If the bot is dead but positions are open, close them in the terminal:

1. **MT5 → Terminal window (`Ctrl + T`) → Trade tab.**
2. **Right-click any position → "Close All"** (or `Close All by Symbol`).
3. Confirm the **Exposure** column is empty and the Trade tab shows no rows.
4. **Disable Algo Trading** (`Ctrl + E`) so nothing re-opens — the button turns grey.
5. Optionally close every chart's EA and **Tools → Options → Expert Advisors →
   untick "Allow automated trading"**.
6. Only then diagnose. **Do not restart the bot until the cause is understood.**

---

## SECTION 6 — DAILY MAINTENANCE & LOG MONITORING

### 6.1 Where everything is written

```
quant_system/
└── reports/
    ├── logs/
    │   └── quant_system.log        ← rotating runtime log (20 MiB x 5 files)
    ├── verification_report.txt     ← 'python main.py --mode verify' output
    ├── data_sources.json           ← which feed supplied each symbol
    │
    │   # -- backtest artifacts --
    ├── backtest_summary.txt        ← full metrics summary
    ├── backtest_equity.csv         ← equity curve
    ├── backtest_trades.csv         ← trade log (entry/exit/PnL/reason)
    ├── backtest_weights.csv        ← per-symbol target weights over time
    ├── backtest_regime_metrics.csv ← performance split by HMM regime
    │
    │   # -- walk-forward artifacts --
    ├── walk_forward_report.txt
    ├── walk_forward_oos_equity.csv ← stitched out-of-sample equity curve
    ├── walk_forward_degradation.csv
    ├── walk_forward_parameters.csv
    │
    ├── voice/                      ← WAV alerts (001_buy_xauusd.wav, ...)
    │
    └── figures/
        ├── backtest_equity.png
        ├── walk_forward_oos_equity.png
        ├── toast_buy_xauusd.png
        └── toast_sell_usdjpy.png
```

**Runtime log.** Every session appends to `reports/logs/quant_system.log`, formatted
`YYYY-MM-DD HH:MM:SS | LEVEL | logger | message`. It rotates at 20 MiB with 5
backups, so it survives unattended runs and terminal disconnects. Disable with
`--no-file-log`.

Tail it live:

```bash
tail -f reports/logs/quant_system.log                 # Linux/macOS
Get-Content reports\logs\quant_system.log -Wait       # Windows PowerShell
```

### 6.2 Daily health check

```bash
# 1. Any errors or rejections in the last session?
grep -E "ERROR|CRITICAL|Traceback" reports/logs/quant_system.log | tail -20

# 2. Trade alerts only (what the bot actually did)
grep "TRADE ALERT" reports/logs/quant_system.log | tail -20

# 3. Cycle summary — confirms the loop was alive
grep "Cycle " reports/logs/quant_system.log | tail -5

# 4. Regime path — is the HMM still decoding sensibly?
grep "regime=" reports/logs/quant_system.log | awk '{print $NF}' | sort | uniq -c
```

### 6.3 Detecting connectivity drops and order rejections

**MT5 order rejections.** The adapter logs every non-`TRADE_RETCODE_DONE` result at
`ERROR` with the numeric retcode and the venue comment:

```
ERROR | quant_system.execution.brokers.mt5_broker | MT5 order rejected for XAUUSD:
retcode=10027 TRADE_RETCODE_AUTOTRADING_DISABLED
```

Watch for them with:

```bash
grep -E "order rejected|retcode=" reports/logs/quant_system.log | tail -20
```

Common MT5 retcodes:

| Retcode | Name | Cause | Fix |
| --- | --- | --- | --- |
| `10004` | `REQUOTE` | Price moved | Automatic (retried) |
| `10006` | `REJECT` | Generic refusal | Check volume/stops |
| `10013` | `INVALID_REQUEST` | Malformed order | Check symbol/filling mode |
| `10014` | `INVALID_VOLUME` | Lot size/step | Check `contract_size` and min lot |
| `10016` | `INVALID_PRICE` | Bad SL/TP/stops | Check stop distance vs broker freeze level |
| `10018` | `MARKET_CLOSED` | Outside session | Adjust schedule |
| `10019` | `NO_MONEY` | Insufficient margin | Reduce sizing / deposit |
| **10027** | **`AUTOTRADING_DISABLED`** | **Algo Trading off** | **`Ctrl+E` in MT5 (§2.1)** |
| `10030` | `UNSUPPORTED_FILLING` | Filling mode | Automatic (adapter retries modes) |
| `10031` | `NO_CONNECTION` | Terminal ↔ server link down | Check terminal connection |

**Connectivity drops.** Symptoms and checks:

```bash
# Broker disconnects / reconnects
grep -iE "disconnect|connect|reconnect" reports/logs/quant_system.log | tail -10

# Stalls: are cycle timestamps still advancing?
grep "Cycle " reports/logs/quant_system.log | tail -3

# Data feed degraded to synthetic? (network outage)
cat reports/data_sources.json
#   all "yfinance" -> healthy |  any "synthetic" -> network problem, prices simulated
```

**Silence is the main symptom.** If no `Cycle` lines have appeared for longer than
`--poll-seconds`, the process is likely stuck or dead:

```bash
ps aux | grep "[m]ain.py"        # Linux/macOS
Get-Process python               # Windows
```

### 6.4 Routine maintenance schedule

| Frequency | Task |
| --- | --- |
| **Every session** | Run `python main.py --mode verify`; confirm `75/75` before launching. |
| **Daily** | Grep the log for `ERROR`/`rejected`; confirm `data_sources.json` shows real feeds; check equity vs expectation. |
| **Weekly** | Re-run the backtest over the latest window; compare to the WFO baseline for drift. |
| **Weekly** | Refresh the data cache (delete `data_cache/` or run with `--no-cache`). |
| **Monthly** | Re-run walk-forward (`--mode both --wfo-grid fast`); investigate acceptance-rate drift. |
| **Monthly** | Prune `reports/voice/` and rotated logs if disk is a concern. |
| **After any dependency upgrade** | Re-run `python -m pytest tests/ -q` **and** `python main.py --mode verify` before deploying. |

### 6.5 Troubleshooting quick reference

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `MT5UnavailableError` on Linux/macOS | Expected — `MetaTrader5` is Windows-only | Use `--broker simulated` or `--broker fix`; trade MT5 from Windows |
| `trade_allowed: False` | Algo Trading disabled | `Ctrl+E` in MT5; check **Options → Expert Advisors** |
| `No MT5 symbol matching 'XAUUSD'` | Symbol absent from Market Watch, or wrong `mt5_symbol` | §2.2 add to Market Watch; §2.3 set `mt5_symbol` |
| `TRADE_RETCODE_MARKET_CLOSED` | Outside trading session | Adjust your run schedule |
| No toasts appear | Headless host, or `--no-toast` | Start a display / `Xvfb :99 & export DISPLAY=:99`; use `--toast-backend pyqt` |
| Toasts appear but silent | Voice disabled or no `espeak-ng` | `--enable-voice`; `sudo apt-get install -y espeak-ng` |
| `WAV` files empty / missing | `runAndWait()` not called (driver flushes on the terminated event) | Use `--voice-mode file` via the CLI; do not call `pyttsx3` directly |
| No trades for hours | Regime 2 active — entries halted **by design** | Check `regime=` in the log; this is correct behaviour |
| Bot won't start | Stale halt file | `rm reports/HALT` |
| `error: Microsoft Visual C++ 14.0 or greater is required` while building `hmmlearn` / `Failed building wheel for hmmlearn` | No prebuilt wheel for your Python version/arch — hmmlearn ships `win_amd64` wheels for **CPython 3.8–3.13 only** (no cp314+, no win32, no arm64), so pip tried to compile `_hmmc` | Use **64-bit Python 3.13** (or 3.12) and recreate the venv; or install the MSVC Build Tools; or use conda. Confirm instantly with `pip install --only-binary=:all: hmmlearn` |
| Backtest trade count changed | `RiskConfig.rebalance_band` controls turnover | Expected; band `0` → ~191 trades, `0.005` (default) → ~44 |

---

## Appendix — Command quick reference

```bash
# --- environment ---
python -m venv venv && source venv/bin/activate       # (Windows: .\venv\Scripts\Activate.ps1)
pip install -r requirements.txt
python scripts/preflight.py --speak

# --- verification ---
python -m pytest tests/ -q
python main.py --mode verify

# --- research ---
python main.py --mode backtest --source auto --start 2020-01-01 --end 2025-12-31
python main.py --mode both --source auto --wfo-grid fast

# --- demo / paper ---
python main.py --mode live --broker mt5 --demo --enable-voice        # Windows
python main.py --mode live --broker simulated --demo --enable-voice  # any OS

# --- live ---
python main.py --mode live --broker mt5 --no-demo --max-daily-trades 5 \
               --capital 25000 --enable-voice --flatten-on-exit

# --- emergency ---
touch reports/HALT          # graceful stop (+flatten with --flatten-on-exit)
Ctrl+C                      # graceful stop (twice = abort now)
rm reports/HALT             # required before restarting
```

---

*Provided for research and educational purposes. Nothing here is investment advice.
Trading leveraged instruments carries substantial risk of loss; validate any
configuration against your own execution assumptions before risking capital.*
