# `quant_system` — VS Code Operational Runbook (Windows)

**Audience:** the operator running this framework from Visual Studio Code on Windows.
**Companion documents:** [`RUNBOOK.md`](RUNBOOK.md) (platform-agnostic operations,
MT5 reference, kill-switch semantics) · [`README.md`](README.md) (architecture & spec).

> **Scope note.** Sections marked ✅ were executed and verified while writing this
> document. Sections marked ⚠️ are Windows/VS Code/MT5 procedures that **could not be
> executed** on the Linux authoring host — they are written from the official
> tooling documentation and the verified CLI surface, and you should confirm them
> once on your own machine. Everything else (JSON validity, every launch-profile
> argument vector, the verification and demo pipelines) was run for real.

---

## SECTION 1 — ENVIRONMENT & VS CODE PREPARATION

### 1.1 Install VS Code extensions

**UI path:** `Ctrl + Shift + X` (Extensions) → paste the ID into the search box →
**Install**.

**CLI alternative** (PowerShell, one line each):

```powershell
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.debugpy
code --install-extension ms-vscode.powershell
code --install-extension ms-python.black-formatter
```

| # | Extension | ID | Why |
| --- | --- | --- | --- |
| 1 | **Python** | `ms-python.python` | Language support, interpreter management, test discovery |
| 2 | **Pylance** | `ms-python.vscode-pylance` | IntelliSense + type checking (now bundled with #1) |
| 3 | **Python Debugger** | `ms-python.debugpy` | The `debugpy` debug adapter (also bundled with #1) |
| 4 | **PowerShell** | `ms-vscode.powershell` | Integrated PowerShell terminal + `.ps1` editing |
| 5 | Black Formatter | `ms-python.black-formatter` | `formatOnSave` |
| 6 | isort | `ms-python.isort` | Import ordering |
| 7 | Flake8 | `ms-python.flake8` | Linting |
| 8 | autoDocstring | `njpwerner.autodocstring` | Generates the Google-style docstrings used throughout (`Ctrl+Shift+2`) |
| 9 | Even Better TOML | `tamasfe.even-better-toml` | `pyproject.toml` |
| 10 | YAML | `redhat.vscode-yaml` | CI / config schemas |
| 11 | Markdown All in One | `yzhang.markdown-all-in-one` | Reading these runbooks |
| 12 | markdownlint | `davidanson.vscode-markdownlint` | Docs linting |
| 13 | Code Spell Checker | `streetsidesoftware.code-spell-checker` | Comments & docs |
| 14 | GitLens | `eamodio.gitlens` | Inline blame/history |

All 14 are declared in **`.vscode/extensions.json`**, so VS Code prompts
*"Do you want to install the recommended extensions for this workspace?"* on first
open. Click **Install All**.

### 1.2 Set PowerShell as the default integrated terminal ⚠️

1. Open the Command Palette: **`Ctrl + Shift + P`**
2. Type `Terminal: Select Default Profile` → **Enter**
3. Choose **`PowerShell`**

This writes `"terminal.integrated.defaultProfile.windows": "PowerShell"` — already
set in the shipped `.vscode/settings.json`, so you can skip the UI if you use the
provided workspace config.

Open a terminal any time with **`Ctrl + ``** (backtick) or
`Ctrl + Shift + P` → `Terminal: Create New Terminal`.

**Why PowerShell and not Command Prompt:** the venv activation script is
`.\venv\Scripts\Activate.ps1`, and the HALT-file / log-tailing snippets below are
PowerShell cmdlets.

### 1.3 Unblock activation scripts (execution policy) ⚠️

If you see:

```
.\venv\Scripts\Activate.ps1 : File ... cannot be loaded because running scripts is
disabled on this system.
```

run, once per user account:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Confirm with `Y`. Verify:

```powershell
Get-ExecutionPolicy -List
```

`CurrentUser` should read `RemoteSigned`. This scope does **not** require
Administrator and does not touch the machine-wide policy.

### 1.4 Create and activate the virtual environment

Open the project folder first:

```powershell
cd C:\path\to\quant_system
code .
```

In the VS Code integrated terminal (**`Ctrl + ``**):

```powershell
# Create (use the py launcher if 'python' is not on PATH)
python -m venv venv
#   or, if you have the Python Launcher but no PATH entry:
py -3.13 -m venv venv

# Activate
.\venv\Scripts\Activate.ps1
```

The prompt becomes `(venv) PS C:\path\to\quant_system>`.

> **If activation still fails:** `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> (§1.3), or use the Command Prompt profile with `.\venv\Scripts\activate.bat`.

Confirm you are inside the venv:

```powershell
python -c "import sys; print(sys.prefix)"
Get-Command python | Select-Object Source
```

`sys.prefix` must end in `venv`, and `Source` must be `...\quant_system\venv\Scripts\python.exe`.

### 1.5 Select the interpreter in VS Code ⚠️

1. **`Ctrl + Shift + P`** → `Python: Select Interpreter` → **Enter**
2. Pick **`.\venv\Scripts\python.exe`** (labelled *Python 3.x.x ('venv': venv)*)
   - If it is not listed: **Enter interpreter path…** → **Find…** → browse to
     `C:\path\to\quant_system\venv\Scripts\python.exe`
3. Confirm in the **status bar** (bottom-left/right) — it should read
   `Python 3.x.x ('venv': venv)`.

This writes `python.defaultInterpreterPath` into `.vscode/settings.json`. The
shipped config already sets it to `${workspaceFolder}\venv\Scripts\python.exe`.

### 1.6 Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

On Windows this **automatically installs `MetaTrader5`**, because `requirements.txt`
declares it with an environment marker:

```
MetaTrader5>=5.0.45 ; platform_system == "Windows"
```

#### If hmmlearn fails to build (the MSVC error)

This is the one install that can fail on Windows:

```
building 'hmmlearn._hmmc' extension
error: Microsoft Visual C++ 14.0 or greater is required.
ERROR: Failed building wheel for hmmlearn
```

**Cause.** `hmmlearn` compiles a Cython extension and publishes prebuilt 64-bit
Windows wheels **only for CPython 3.8 – 3.13** (`cp38`…`cp313`, `win_amd64`). There
is no `cp314+` wheel, and no 32-bit or ARM64 wheel. On any other interpreter pip
falls back to the source tarball and tries to compile it.

**Diagnose in one command** (fails instantly instead of compiling):

```powershell
python -c "import sys,struct; print(sys.version); print(struct.calcsize('P')*8,'bit')"
python -m pip install --only-binary=:all: hmmlearn
```

**Fix — recommended (no toolchain, ~2 minutes):** use **64-bit Python 3.13**.

```powershell
# 1. Install Python 3.13 (64-bit) from python.org, tick "Add python.exe to PATH"
# 2. Rebuild the venv
cd C:\path\to\quant_system
Remove-Item -Recurse -Force venv
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Fix — if you must stay on Python 3.14+** (installs ~2 GB of build tools):

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools"
# close and reopen the terminal, then:
pip install -r requirements.txt
```

**Fix — conda alternative** (ships prebuilt binaries):

```powershell
conda create -n quant python=3.13
conda activate quant
conda install -c conda-forge hmmlearn
pip install -r requirements.txt
```

> Only `hmmlearn` should fail this way. If `numpy` or `scipy` also fail to build,
> the problem is broader than the wheel matrix — install the MSVC Build Tools.



```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

On Windows this **automatically installs `MetaTrader5`**, because `requirements.txt`
declares it with an environment marker:

```
MetaTrader5>=5.0.45 ; platform_system == "Windows"
```

Confirm:

```powershell
pip list | Select-String -Pattern "MetaTrader5|pyttsx3|PyQt6|hmmlearn|statsmodels|backtrader"
```

> **No OS-level packages are needed on Windows.** `pyttsx3` uses the built-in SAPI5
> voice engine and PyQt6 ships its Qt DLLs — unlike Linux, which needs
> `espeak-ng` and the `libxcb-*` set (see `RUNBOOK.md` §1.3).

---

## SECTION 2 — METATRADER 5 & OS DEPENDENCY PRE-FLIGHT (WINDOWS)

> `MetaTrader5` is a **Windows-only** package that drives the MT5 desktop terminal
> over IPC. The terminal must be running **and logged in** before you launch the bot.

### 2.1 Install, launch and log in ⚠️

**Install the Python package** (already done by `requirements.txt` on Windows;
otherwise):

```powershell
.\venv\Scripts\Activate.ps1
pip install MetaTrader5
```

**Install the desktop terminal** (skip if already installed): download from your
broker or metaquotes.net, run the installer, accept the defaults.

**Launch and log in:**

1. Start **MetaTrader 5** from the Start menu.
2. `File → Login to Trade Account` (or click the account name in the Navigator).
3. Enter **login**, **password**, and select the **server** — for your first run
   choose the **demo** server.
4. Click **Login**. Confirm in the bottom-right status bar that the connection
   indicator shows a **green bar** and a ping time (e.g. `1234 / 45 ms`).

### 2.2 Configure MT5 for automated trading ⚠️

1. Press **`Ctrl + O`** (or `Tools → Options`) → **Expert Advisors** tab.
2. Tick:
   - ☑ **Allow automated trading**
   - ☑ **Allow DLL imports** ← *required: the `MetaTrader5` Python package loads as a native module*
   - ☑ **Allow automated trading for all symbols** *(recommended)*
3. Click **OK**.
4. Enable the runtime switch: click the **Algo Trading** toolbar button, or press
   **`Ctrl + E`**. It turns **green**, and the status bar shows the algo-trading icon.

Verify from the integrated terminal:

```powershell
.\venv\Scripts\Activate.ps1
python -c "import MetaTrader5 as mt5; mt5.initialize(); ti=mt5.terminal_info(); print('trade_allowed:', ti.trade_allowed); print('connected:', ti.connected); mt5.shutdown()"
```

**Required:** `trade_allowed: True`. If `False`, revisit step 4 (`Ctrl + E`).

### 2.3 Add the traded symbols to Market Watch ⚠️

1. **`Ctrl + M`** (or `View → Market Watch`).
2. Right-click inside the window → **Symbols** (or **`Ctrl + U`**).
3. Expand the groups and **double-click** each symbol — it highlights when active:

   | Group | Symbol |
   | --- | --- |
   | Metals | `XAUUSD` |
   | Metals | `XAGUSD` |
   | Forex Majors | `EURUSD` |
   | Forex Majors | `USDCHF` |
   | Forex Majors | `USDJPY` |

4. If a symbol is absent, tick **Show all symbols** in the same dialog and search again.
5. Close the dialog; all five must show live bid/ask ticks.

**Broker aliasing.** If your broker renames instruments (`XAUUSD.m`, `GOLD`,
`XAUUSD.pro`), the adapter auto-resolves via
`("", ".m", ".a", ".pro", ".raw", "_m", "#", ".i")` and then a full symbol-table
scan. To pin it explicitly, edit `mt5_symbol` in `config/settings.py`. Full details
and a resolution script: `RUNBOOK.md` §2.3.

### 2.4 Run the integrated pre-flight check ✅

```powershell
python scripts/preflight.py --speak
```

Or launch **Profile 6** from the Run & Debug panel (`Ctrl + Shift + D`).

Verified output (Linux authoring host; on Windows the `MT5` line becomes `[ OK ]`):

```
==============================================================================
  quant_system :: ENVIRONMENT PRE-FLIGHT
==============================================================================
  host    : Linux 6.1.158+ | Python 3.13.14 | display=no

  [ OK ] CORE: numpy 2.3.5, pandas 2.2.3, scipy 1.17.1, statsmodels 0.15.0,
         sklearn 1.6.1, hmmlearn 0.3.3, yfinance 1.7.0, matplotlib 3.10.9,
         backtrader 1.9.78.123
  [ OK ] AUDIO: 141 voices, WAV render 87614 bytes, speak=ok
  [ OK ] TOASTS: auto -> PyQtToastBackend (pyqt started)
  [DEGR] MT5: not applicable - MetaTrader5 is Windows-only and this host is linux.
  [ OK ] DATA: 5 symbols x 124 bars via ['yfinance']

==============================================================================
  READY: required subsystems (core, data) passed
==============================================================================
```

**Green criteria per subsystem:**

| Check | Green means | If not green |
| --- | --- | --- |
| **CORE** | Every scientific package imports and reports a version. No `missing:` entry. | Re-run `pip install -r requirements.txt` |
| **AUDIO** | Voice count > 0, `WAV render` > 1 kB, and `speak=ok` with `--speak`. | Windows: check SAPI5 voices in *Settings → Time & language → Speech*. Linux: `sudo apt-get install -y espeak-ng` |
| **TOASTS** | `auto -> PyQtToastBackend`. | Install PyQt6; log in to a graphical session. `NullToastBackend` = headless (alerts logged, not shown) — **not acceptable for an attended desk** |
| **MT5** | `[ OK ]` with `trade_allowed: True`, and the correct `login@server`. | Terminal running? Logged in? `Ctrl + E`? See §2.2 |
| **DATA** | 5 symbols fetched. `sources` all `yfinance`. | `synthetic` means the network is down — prices are simulated, do not trade |

Exit code is `0` when **CORE** and **DATA** pass; the other three degrade gracefully.
A `[FAIL]` on **AUDIO** means your voice alerts will be silent.

> **Note:** the `AUDIO` check renders a real WAV *and* (with `--speak`) plays one
> utterance, so it proves the driver end to end — not just that the library imports.
> The speak test is bounded by a 20 s watchdog on a daemon thread, so a missing
> audio device can never hang the check.

---

## SECTION 3 — VS CODE WORKSPACE & LAUNCH CONFIGURATIONS

Both files ship with the project. Open them via `Ctrl + P` → type the filename.

### 3.1 `.vscode/settings.json` ✅ (validated)

```jsonc
{
    // Paths use DOUBLE backslashes: this is JSON, not PowerShell.
    "python.defaultInterpreterPath": "${workspaceFolder}\\venv\\Scripts\\python.exe",

    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.extraPaths": ["${workspaceFolder}"],
    "python.analysis.autoSearchPaths": true,
    "python.analysis.diagnosticMode": "workspace",

    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": ["tests"],

    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": { "source.organizeImports": "explicit" },
        "editor.rulers": [100]
    },

    // PowerShell default terminal — required for .\venv\Scripts\Activate.ps1
    // and for Ctrl+C to reach the trading process.
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "terminal.integrated.cwd": "${workspaceFolder}",

    // Hide generated noise from the explorer AND the file watcher.
    "files.exclude": {
        "**/__pycache__": true, "**/*.pyc": true,
        "**/.pytest_cache": true, "**/.mypy_cache": true,
        "**/.ruff_cache": true,   "venv/**": true
    },
    "files.watcherExclude": {
        "**/data_cache/**": true, "**/reports/**": true,
        "**/__pycache__/**": true, "**/venv/**": true
    },

    // Keep MT5 credentials out of source control.
    "python.envFile": "${workspaceFolder}\\.env",

    "debugpy.debugJustMyCode": true
}
```

The shipped file also configures the `PowerShell` / `Command Prompt` / `Git Bash`
terminal profiles, Black/isort/Flake8 arguments (`--line-length 100` to match the
codebase), editor hygiene, and `search.exclude`.

### 3.2 `.vscode/launch.json` ✅ (validated)

```jsonc
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "1. Verify Pipeline (--mode verify)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}\\main.py",
            "args": ["--mode", "verify"],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": { "PYTHONPATH": "${workspaceFolder}" }
        },
        {
            "name": "2. Backtest & WFO (--mode both)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}\\main.py",
            "args": ["--mode", "both", "--source", "auto", "--wfo-grid", "fast"],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": { "PYTHONPATH": "${workspaceFolder}" }
        },
        {
            "name": "3. DEMO: MT5 Paper Trading (--broker mt5 --demo --enable-voice)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}\\main.py",
            "args": [
                "--mode", "live", "--broker", "mt5",
                "--demo", "--enable-voice",
                "--max-iterations", "0", "--poll-seconds", "60"
            ],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": { "PYTHONPATH": "${workspaceFolder}" }
        },
        {
            "name": "4. ⚠ LIVE REAL MONEY (--no-demo --flatten-on-exit)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}\\main.py",
            "args": [
                "--mode", "live", "--broker", "mt5",
                "--no-demo", "--max-daily-trades", "5",
                "--capital", "25000",
                "--max-iterations", "0", "--poll-seconds", "300",
                "--enable-voice", "--flatten-on-exit"
            ],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": { "PYTHONPATH": "${workspaceFolder}" }
        }
    ]
}
```

The shipped file additionally includes **Profile 5** (simulated broker — no MT5
needed), **Profile 6** (pre-flight), **Profile 7** (pytest under the debugger) and
**Profile 8** (attach to a running session on port 5678).

#### Two settings that are not optional

**① `"console": "integratedTerminal"`**

The bot installs `SIGINT`/`SIGTERM` handlers. If you launch into the **Debug
Console** (`"internalConsole"`), `Ctrl + C` is swallowed and cannot reach the
process — you lose the graceful stop and the flatten-on-exit path. Every profile
above uses `integratedTerminal`.

**② `--max-iterations 0`**

The CLI default is **5 cycles**, after which the session exits by itself — useless
for a trading session. `--max-iterations 0` means *run until stopped*. Every
long-running profile sets it explicitly.

| Profile | Mode | Broker | Runs forever? | Real money? |
| --- | --- | --- | --- | --- |
| 1. Verify Pipeline | `verify` | — | No (~60 s) | No |
| 2. Backtest & WFO | `both` | — | No (~3-5 min) | No |
| 3. DEMO | `live` | `mt5` | Yes | **No** (demo account) |
| 4. LIVE | `live` | `mt5` | Yes | **YES** |
| 5. DEMO (simulated) | `live` | `simulated` | Yes | No |
| 6. Pre-flight | — | — | No | No |
| 7. pytest | — | — | No | No |
| 8. Attach | — | — | n/a | depends |

### 3.3 `.vscode/tasks.json` ✅ (validated)

Ten tasks, runnable via `Ctrl + Shift + P` → `Tasks: Run Task`:

| Task | Command |
| --- | --- |
| `preflight: environment check` | `python scripts/preflight.py --speak` |
| `pytest: full suite` (default test task) | `python -m pytest -q` |
| `verify: end-to-end pipeline (75 checks)` | `python main.py --mode verify` |
| `backtest: 2016-2024` | `python main.py --mode backtest --source auto --start 2016-01-01 --end 2024-12-31` |
| `walk-forward: backtest + WFO (fast grid)` | `python main.py --mode both --source auto --wfo-grid fast` |
| `HALT: create sentinel (stop the bot)` | `New-Item -ItemType File -Force -Path reports\HALT` |
| `HALT: remove sentinel (allow restart)` | `Remove-Item -Force -Path reports\HALT` |
| `logs: tail quant_system.log` | `Get-Content reports\logs\quant_system.log -Wait -Tail 40` |
| `logs: show errors & rejections` | `Select-String ... -Pattern "ERROR\|CRITICAL\|rejected\|retcode="` |
| `reports: open backtest summary` | `Get-Content reports\backtest_summary.txt` |

> **Prefer tasks over `F5` for long sessions.** Tasks run in the terminal *without*
> the debugger attached, so a stray breakpoint can never freeze a live position.

---

## SECTION 4 — RUNNING & DEBUGGING MODES IN VS CODE

Open the panel with **`Ctrl + Shift + D`**, pick the profile from the dropdown,
then:

| Action | Shortcut | Use for |
| --- | --- | --- |
| **Start Debugging** | `F5` | Breakpoints, stepping, inspecting frames |
| **Run Without Debugging** | `Ctrl + F5` | **Long/live sessions** — no debugger overhead, no accidental pause |
| **Stop** | `Shift + F5` | Terminate the debug session |
| **Restart** | `Ctrl + Shift + F5` | Re-run the active profile |
| Toggle breakpoint | `F9` | |

### 4.1 Mode 1 — Pipeline verification ✅

**Profile 1** → `F5` (or `Ctrl + F5`).

Runs the five-stage end-to-end check. Verified output in the **integrated terminal**:

```
[PASS]  Data Ingestion & Preprocessing
[PASS]  HMM Switchboard Decoding
[PASS]  Strategy Signal Generation & Sizing
[PASS]  MT5 Connection / Simulated Fallback
[PASS]  Trade Notifier Dispatch & Audio Queue
==============================================================================
  RESULT: 75/75 checks passed in 10.65s   (data source: yfinance)
==============================================================================
```

**Zero errors / zero warnings criteria:**

- Every stage header reads `[PASS]` (not `[FAIL]`).
- The final line is `75/75 checks passed`.
- **No** `no warning: ...` entries. The suite runs each stage inside
  `warnings.catch_warnings` and converts any warning into a failed check, so a
  warning can never hide.
- On your Windows host, stage 4 must log `MT5 terminal connected`. On a non-Windows
  host it logs `MT5 terminal unreachable -> simulated fallback engaged` — expected,
  and it still passes.

Then run the unit suite (`Ctrl + Shift + P` → `Tasks: Run Task` → `pytest: full suite`,
or **Profile 7**):

```
72 passed, 1 skipped in 47.32s
```

The single skip is conditional (`test_stat_arb.py:370` needs a `|z| >= 3.5`
excursion the fixture did not produce) and is benign.

> Run **both** before any trading session. Together they take ~60 s.

### 4.2 Mode 2 — Backtest & walk-forward optimisation ✅

**Profile 2** → `Ctrl + F5`. Takes ~3-5 minutes; writes `reports/`.

```
                    final_equity : 1,004,420.6608
                total_return_pct : 0.4421
                    sharpe_ratio : 0.0743
                max_drawdown_pct : 1.0877

Segments              : 39
Acceptance rate       : 38.5%
Mean degradation      : -22.58%

                    final_equity : 1,022,284.3721     # stitched out-of-sample
                total_return_pct : 2.3043
```

`reports/` then holds `backtest_summary.txt`, `backtest_equity.csv`,
`backtest_trades.csv`, `backtest_weights.csv`, `backtest_regime_metrics.csv`,
`walk_forward_report.txt`, `walk_forward_oos_equity.csv`,
`walk_forward_degradation.csv`, `walk_forward_parameters.csv`, `figures/*.png`.

### 4.3 Mode 3 — Demo trading (paper) ✅

**Profile 3** (MT5 demo account) or **Profile 5** (simulated broker, no MT5 needed).

Verified startup sequence (Profile 5):

```
Trade governance: mode=DEMO (unlimited) daily_trade_limit=unlimited
Notifier: toasts=True (pyqt started) voice=True (file)
Loaded 5 symbols x 2870 bars from {...}
Broker simulated connected (equity=1000000.00)
Running unbounded - stop with Ctrl+C or by creating ...\reports\HALT.
Cycle 1 | regime=2 | no orders (regime_state_2_shock)
Cycle 2 | regime=2 | no orders (regime_state_2_shock)
```

**Inspecting live HMM regime logs in VS Code:**

- The `Cycle N | regime=R | ...` line is printed every poll. Filter the terminal
  with **`Ctrl + F`** (find in terminal) and search `regime=`.
- For a persistent, scrollable trail, run the **`logs: tail quant_system.log`
  task** in a second terminal — see §6.1.
- `no orders (regime_state_2_shock)` is **correct**: State 2 halts entries by design.

**Observing alerts on a fill** — three things fire at once:

1. **Terminal / log:**
   ```
   TRADE ALERT | [SELL] XAUUSD | 0.17 | 11,582.42 | SL 12,075.28 | TP - | State 1 Momentum
   ```
2. **Windows toast** — bottom-right, auto-closing after 6 s:
   ```
   [SELL] XAUUSD
   0.17 | 11,582.42 | SL 12,075.28 | TP - | State 1 Momentum
   ```
3. **Spoken alert** (SAPI5 on Windows), prices read digit-wise:
   > *"Sold 0 point 17 lots of Gold at 11582 point 42 under State 1 Momentum,
   > stop 12075 point 28."*

`TP -` is normal: momentum exits on a 2.5×ATR trailing stop, not a fixed target.

**Debugging tip:** set a breakpoint in `utils/notifier.py` →
`NotifierEngine.notify_fill` and run **Profile 5** with `F5`. The debugger pauses
on the first fill so you can inspect the `TradeEvent`, the toast payload and the
speech text. **Never do this in Profile 4** — pausing a live bot mid-cycle leaves
orders half-routed.

**Demo mode semantics:** unlimited new positions per day; **Kelly / risk-parity
sizing and the ATR stops are identical to live.** Only *openings* (flat → non-flat)
are counted, so scaling into or trimming an existing position is never throttled.

### 4.4 Mode 4 — Live trading: mandatory pre-checks ⚠️

**Profile 4 is the only profile that risks real money.** Before pressing `F5` or
`Ctrl + F5`, confirm every line:

| # | Check | How |
| --- | --- | --- |
| 1 | Tests green | Task `pytest: full suite` → `72 passed, 1 skipped` |
| 2 | Pipeline verified | Profile 1 → `75/75` |
| 3 | Pre-flight green on **MT5** | Task `preflight: environment check` → `MT5 [ OK ]`, `trade_allowed: True` |
| 4 | **Correct account** | Check `login` / `server` in the pre-flight `terminal:` line. Demo vs real! |
| 5 | Symbols resolve | No `UNRESOLVED` — see `RUNBOOK.md` §2.3 |
| 6 | Backtest + WFO reviewed | §4.2 numbers understood and accepted |
| 7 | Alerts proven in demo | You have *heard* a spoken alert and *seen* a toast (§4.3) |
| 8 | `DEMO_MODE = False` | `config/settings.py`, or rely on `--no-demo` in the profile |
| 9 | `--flatten-on-exit` present | It is, in Profile 4 |
| 10 | You can stop it | You know §5 — read it before starting |

Then press **`Ctrl + F5`** (Run Without Debugging), **not** `F5`.

Profile 4 applies five independent risk layers:

| Layer | Value |
| --- | --- |
| Daily trade cap | `--max-daily-trades 5` |
| Per-unit risk | 1 % of equity (`SizingConfig.risk_per_unit_pct`) |
| Stop distance | 2.5 × ATR |
| Shock tightening | Stops halve in State 2 |
| Regime exposure | Zero exposure in State 2 |
| **Drawdown breaker** | Auto-flatten at 25 % drawdown |

---

## SECTION 5 — SHORTCUTS, CONTROL & EMERGENCY HALT PROCEDURES

### 5.1 Keyboard reference

| Shortcut | Action |
| --- | --- |
| `Ctrl + Shift + D` | Run & Debug panel |
| `F5` / `Ctrl + F5` | Start Debugging / Run Without Debugging |
| `Shift + F5` | Stop the debug session |
| `Ctrl + Shift + F5` | Restart |
| `F9` | Toggle breakpoint |
| `Ctrl + Shift + P` | Command Palette |
| `` Ctrl + ` `` | Toggle integrated terminal |
| `Ctrl + Shift + ` `` ` `` | New terminal |
| `Ctrl + C` | **Send SIGINT to the running process** (see §5.2) |
| `Ctrl + Shift + X` | Extensions |
| `Ctrl + Shift + E` | Explorer |
| `Ctrl + Shift + F` | Search across files |
| `Ctrl + Shift + M` | Problems panel |
| `Ctrl + P` | Quick file open |
| `Ctrl + B` | Toggle sidebar |

### 5.2 Graceful stop — `Ctrl + C` ✅

Click the **integrated terminal** (where the bot is running) and press
**`Ctrl + C`** once:

```
WARNING | SIGINT received - finishing the current cycle, then shutting down cleanly.
          Send it again to abort right now.
WARNING | Shutdown requested (signal) - stopping before cycle 102.
INFO    | Shutdown requested (signal) - no open positions.
```

With `--flatten-on-exit` (Profile 4) and open positions, it closes them first:

```
WARNING | Shutdown requested (halt_file) - flattening 1 position(s): ['XAUUSD']
WARNING | Flatten result: ['filled']
```

Exit code `0`. The inter-cycle sleep is re-checked every 250 ms, so the stop lands
almost immediately even with `--poll-seconds 300`.

Pressing `Ctrl + C` **twice** aborts at once:

```
CRITICAL | Second SIGINT received - aborting immediately. Positions are LEFT OPEN;
           flatten them manually.
```

Exit code `130`; positions are intentionally left open.

> **Timing caveat:** two presses within a fraction of a second are merged into one
> signal by the OS/Python, so you get the graceful stop instead of the abort —
> usually the better outcome. To force the abort, pause ~1 s between presses.

### 5.3 Sentinel halt file ✅

From **any** PowerShell terminal in VS Code — including one that never launched the
bot:

```powershell
New-Item -ItemType File reports\HALT
```

Or use the task: `Ctrl + Shift + P` → `Tasks: Run Task` → **`HALT: create sentinel
(stop the bot)`**.

The bot stops at the next cycle boundary and exits `0`; with `--flatten-on-exit` it
flattens first. This is the recommended production stop because it works over
Remote Desktop / SSH and from automation.

> **Cleanup — the bot refuses to start while the file exists:**
> ```
> ERROR | Halt file reports\HALT exists at start-up. Remove it before launching
> ```
> ```powershell
> Remove-Item reports\HALT      # or run the "HALT: remove sentinel" task
> ```

### 5.4 Emergency disconnect ⚠️

Use when the process is unresponsive, VS Code is hung, or you need everything flat
*now*.

**Step 1 — kill the VS Code terminal session (least forceful first):**

1. Click the terminal panel, press **`Ctrl + C`** → wait 10 s.
2. If still running, click the **trash / kill** icon in the terminal panel
   (`Terminal: Kill the Active Terminal Instance` via `Ctrl + Shift + P`) → wait 10 s.
3. If still running, close VS Code entirely.
4. Last resort, from a separate PowerShell window:
   ```powershell
   Get-Process python | Stop-Process -Force
   ```
   `Stop-Process -Force` is `SIGKILL` — **no cleanup runs, positions stay open.**

**Step 2 — disable Algo Trading in MT5:**

1. Switch to the MT5 window.
2. Press **`Ctrl + E`** — the Algo Trading button turns **grey**.
   *Nothing can now open a new position.*
3. Belt and braces: `Ctrl + O` → **Expert Advisors** → untick **Allow automated
   trading** → OK.

**Step 3 — manually flatten open positions:**

1. **`Ctrl + T`** (Terminal window) → **Trade** tab.
2. Right-click **any** position → **Close All** (or *Close All by Symbol*).
3. Confirm the Trade tab is empty and **Exposure** shows nothing.
4. Only then diagnose. **Do not restart the bot until the cause is understood.**

---

## SECTION 6 — IN-EDITOR LOG & ARTIFACT MONITORING

### 6.1 Tail the live log inside VS Code

**Option A — dedicated terminal** (recommended). Open a second terminal
(`Ctrl + Shift + ` `` ` ``) and run:

```powershell
Get-Content reports\logs\quant_system.log -Wait -Tail 40
```

This streams new lines as they are written. Run it side-by-side with the bot:

- `Ctrl + \` splits the editor; drag the terminal into the right-hand group.
- Or `Ctrl + Shift + P` → `Tasks: Run Task` → **`logs: tail quant_system.log`**.

The log rotates at 20 MiB with 5 backups, so it survives long unattended runs and
terminal disconnects.

**Option B — open the file.** `Ctrl + P` → `quant_system.log`. VS Code will show a
*reload* prompt when it changes on disk.

**Useful filters** (PowerShell in the integrated terminal):

```powershell
# Errors, rejections and tracebacks
Select-String -Path reports\logs\quant_system.log -Pattern "ERROR|CRITICAL|Traceback" |
    Select-Object -Last 20

# What the bot actually did
Select-String -Path reports\logs\quant_system.log -Pattern "TRADE ALERT" |
    Select-Object -Last 20

# Is the loop alive? (timestamps must advance)
Select-String -Path reports\logs\quant_system.log -Pattern "Cycle " |
    Select-Object -Last 5

# Regime distribution over the session
Select-String -Path reports\logs\quant_system.log -Pattern "regime=" |
    ForEach-Object { ($_ -split "regime=")[1].Substring(0,1) } | Group-Object |
    Select-Object Name, Count

# MT5 rejections with retcodes
Select-String -Path reports\logs\quant_system.log -Pattern "order rejected|retcode=" |
    Select-Object -Last 20
```

Or run the **`logs: show errors & rejections`** task.

Common MT5 retcodes: `10027 AUTOTRADING_DISABLED` (press `Ctrl + E`),
`10018 MARKET_CLOSED`, `10019 NO_MONEY`, `10016 INVALID_PRICE`,
`10014 INVALID_VOLUME`, `10031 NO_CONNECTION`. Full table: `RUNBOOK.md` §6.3.

**Confirm the data feed is real, not synthetic:**

```powershell
Get-Content reports\data_sources.json
```

All five symbols should read `"yfinance"`. Any `"synthetic"` means the network was
down and prices were simulated.

### 6.2 Review reports and equity graphs in the Explorer

**`Ctrl + Shift + E`** → expand `reports/`:

| File | What it is | How to view |
| --- | --- | --- |
| `backtest_summary.txt` | Full metrics: Sharpe, Sortino, Calmar, MaxDD, expectancy, exposure | Click to open, or run the **`reports: open backtest summary`** task |
| `backtest_equity.csv` | Equity curve | Open as CSV; or the **`Excel Viewer`** extension |
| `backtest_trades.csv` | Trade log: entry/exit, PnL, exit reason | Open as CSV |
| `backtest_weights.csv` | Per-symbol target weights over time | Open as CSV |
| `backtest_regime_metrics.csv` | Performance split by HMM regime | Open as CSV |
| `walk_forward_report.txt` | Segment table + acceptance rate | Click to open |
| `walk_forward_oos_equity.csv` | Stitched out-of-sample equity | Open as CSV |
| `figures\backtest_equity.png` | Equity curve chart | **Click → VS Code image preview** |
| `figures\walk_forward_oos_equity.png` | OOS equity chart | Click to preview |
| `figures\toast_buy_xauusd.png` | Sample toast render | Click to preview |
| `voice\*.wav` | Spoken alerts (`--voice-mode file`) | Double-click → plays in your default player |
| `verification_report.txt` | Last `--mode verify` output | Click to open |
| `logs\quant_system.log` | Rotating runtime log | See §6.1 |

**Side-by-side review tip:** open `backtest_summary.txt`, then
`Ctrl + P` → `backtest_equity.png`, right-click the tab → **Split Right**
(or `Ctrl + \`). Metrics on the left, chart on the right.

**Markdown runbooks:** open `RUNBOOK.md` or this file and press
`Ctrl + Shift + V` for the rendered preview, or `Ctrl + K V` for side-by-side.

---

## Appendix A — What was verified vs. what was not

| Item | Status |
| --- | --- |
| `.vscode/settings.json`, `launch.json`, `extensions.json`, `tasks.json` parse as valid JSONC | ✅ validated |
| All 8 launch profiles' argument vectors accepted by the real `argparse` parser | ✅ validated |
| Profiles 1, 2, 5, 6, 7 executed end to end | ✅ `75/75`, WFO numbers, demo session, `speak=ok`, `72 passed` |
| `--max-iterations 0` runs unbounded and stops via HALT | ✅ validated (exit 0) |
| `preflight.py --speak` completes instead of hanging | ✅ bug found and fixed (20 s daemon-thread watchdog) |
| `console: "integratedTerminal"` is required for Ctrl+C | ✅ reasoned from the SIGINT handler; ⚠️ not executed in VS Code |
| Extension IDs (`ms-python.python`, `ms-python.vscode-pylance`, `ms-python.debugpy`, `ms-vscode.powershell`) | ✅ confirmed against Microsoft docs |
| `"type": "debugpy"` (not the deprecated `"python"`) | ✅ confirmed against Microsoft docs |
| PowerShell commands, execution policy, `Ctrl+O` / `Ctrl+M` / `Ctrl+E` MT5 steps | ⚠️ **not executed** — Linux authoring host |

## Appendix B — First-run checklist

```powershell
cd C:\path\to\quant_system
code .                                          # opens with the shipped .vscode config
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned    # once per user
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt                 # installs MetaTrader5 on Windows
python scripts/preflight.py --speak             # fix anything not [ OK ]
python -m pytest -q                             # 72 passed, 1 skipped
python main.py --mode verify                    # 75/75
```

Then `Ctrl + Shift + D` → **Profile 3 (DEMO)** → `Ctrl + F5`.

---

*Provided for research and educational purposes. Nothing here is investment advice.
Trading leveraged instruments carries substantial risk of loss; validate any
configuration against your own execution assumptions before risking capital.*
