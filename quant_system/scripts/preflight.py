"""Environment pre-flight check for ``quant_system``.

Verifies that every optional subsystem the live trader depends on is actually
usable on *this* host, before you ever connect capital:

1. Core numerical / econometrics imports.
2. **Audio** — ``pyttsx3`` initialisation, voice count, and a real WAV render
   (proves the espeak-ng / SAPI5 backend works headlessly).
3. **Desktop toasts** — resolves each backend (PyQt6, tkinter, null) and reports
   which one would actually be used.
4. **MetaTrader 5** — whether the terminal + ``MetaTrader5`` package are
   reachable (Windows-only; a clean "unavailable" is a normal result elsewhere).
5. **Market data** — whether yfinance can reach the network, and whether the
   synthetic fallback engages cleanly.

Run it immediately after ``pip install -r requirements.txt``::

    python scripts/preflight.py                 # human-readable report
    python scripts/preflight.py --json           # machine-readable
    python scripts/preflight.py --speak          # also plays a test utterance

Exit codes:
    0  every *required* subsystem is healthy (audio/toast/MT5 degrade gracefully)
    1  a required subsystem is broken (core imports or data ingestion)

Author: quant_system
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repository importable when run as a plain script.
# scripts/preflight.py -> parents[0]=scripts, [1]=quant_system, [2]=repo root.
# The repo *parent* must be on sys.path for `quant_system.*` to resolve.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ANSI colours (disabled when not a TTY).
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
_GREEN = "\033[32m" if _USE_COLOR else ""
_RED = "\033[31m" if _USE_COLOR else ""
_YELLOW = "\033[33m" if _USE_COLOR else ""
_BOLD = "\033[1m" if _USE_COLOR else ""
_OFF = "\033[0m" if _USE_COLOR else ""


def _mark(ok: bool, degraded: bool = False) -> str:
    """Return a coloured status tag.

    Args:
        ok: Whether the check passed.
        degraded: Passed, but via a fallback.

    Returns:
        A short status string.
    """
    if ok and not degraded:
        return f"{_GREEN}[ OK ]{_OFF}"
    if ok:
        return f"{_YELLOW}[DEGR]{_OFF}"
    return f"{_RED}[FAIL]{_OFF}"


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def check_core() -> Dict[str, Any]:
    """Verify the mandatory scientific stack imports.

    Returns:
        A result dictionary with ``ok``, ``detail`` and per-package versions.
    """
    versions: Dict[str, str] = {}
    required = (
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "sklearn",
        "hmmlearn",
        "yfinance",
        "matplotlib",
    )
    missing: List[str] = []
    for name in required:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - install-time issue
            missing.append(f"{name}: {exc}")
    try:
        import backtrader  # noqa: F401

        versions["backtrader"] = getattr(backtrader, "__version__", "unknown")
    except Exception as exc:
        missing.append(f"backtrader: {exc}")

    return {
        "ok": not missing,
        "degraded": False,
        "versions": versions,
        "missing": missing,
        "detail": ", ".join(f"{k} {v}" for k, v in versions.items()) if not missing else "; ".join(missing),
    }


def _speak_once(timeout: float = 20.0) -> tuple:
    """Speak a test utterance on a daemon thread with a hard timeout.

    Some drivers (notably espeak on a host with no audio device) never return
    from ``runAndWait()``. Running it on a daemon thread and joining with a
    timeout guarantees the pre-flight can never hang, and a daemon thread cannot
    block interpreter shutdown either.

    Args:
        timeout: Seconds to wait for the utterance to complete.

    Returns:
        Tuple ``(spoken, error)``.
    """
    box: Dict[str, Any] = {}

    def work() -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            try:
                engine.say("Quant system voice alerts are operational.")
                engine.runAndWait()
                box["r"] = (True, None)
            finally:
                try:
                    engine.stop()
                except Exception:  # pragma: no cover
                    pass
        except Exception as exc:  # pragma: no cover - driver dependent
            box["r"] = (False, str(exc))

    thread = threading.Thread(target=work, name="preflight-speak", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return False, f"still speaking after {timeout:.0f} s (no audio device?)"
    return box.get("r", (False, "no result"))


def check_audio(speak: bool = False) -> Dict[str, Any]:
    """Verify text-to-speech, including a real WAV render.

    The WAV render uses its own engine and the *speak* test a second, fresh one:
    mixing ``save_to_file`` and ``say`` on a single espeak engine leaves a pending
    save callback that fires after the temp directory is gone and can wedge the
    driver.

    Args:
        speak: Also play a short utterance through the default device.

    Returns:
        A result dictionary.
    """
    result: Dict[str, Any] = {"ok": False, "degraded": False, "detail": ""}
    try:
        import pyttsx3
    except Exception as exc:
        result["detail"] = f"pyttsx3 not importable: {exc}"
        result["hint"] = "pip install pyttsx3  (Linux also needs: sudo apt-get install -y espeak-ng)"
        return result

    try:
        engine = pyttsx3.init()
    except Exception as exc:
        result["detail"] = f"pyttsx3.init() failed: {exc}"
        result["hint"] = (
            "Linux: sudo apt-get install -y espeak-ng libespeak1\n"
            "Windows: no driver needed (SAPI5 is built in)"
        )
        return result

    try:
        voices = engine.getProperty("voices")
    except Exception:  # pragma: no cover - driver dependent
        voices = []
    result["voices"] = len(voices)

    # A real render: the WAV is only flushed on the terminated-event callback,
    # so runAndWait() is mandatory after save_to_file().  Write to a persistent
    # path (not a temp dir) so a late callback still has a valid target.
    out_path = Path(tempfile.gettempdir()) / f"qs_preflight_{__import__('os').getpid()}.wav"
    try:
        engine.save_to_file("quant system audio check complete.", str(out_path))
        engine.runAndWait()
        size = out_path.stat().st_size if out_path.exists() else 0
        result["wav_bytes"] = size
        result["wav_ok"] = size > 1000
    except Exception as exc:
        result["detail"] = f"WAV render failed: {exc}"
        result["wav_ok"] = False
        return result
    finally:
        try:
            engine.stop()
        except Exception:  # pragma: no cover
            pass
        out_path.unlink(missing_ok=True)

    if speak:
        spoken, error = _speak_once(timeout=20.0)
        result["spoken"] = bool(spoken)
        if error:
            result["speak_error"] = error
            result["hint"] = (
                "No output device, or the driver did not return. WAV synthesis "
                "works, so --voice-mode file will still produce audio files."
            )

    result["ok"] = bool(result["wav_ok"])
    result["detail"] = (
        f"{len(voices)} voices, WAV render {result['wav_bytes']} bytes"
        + (f", speak={'ok' if result.get('spoken') else 'FAILED'}" if speak else "")
        if result["ok"]
        else "WAV render produced an empty file"
    )
    return result


def check_toasts() -> Dict[str, Any]:
    """Resolve every desktop-toast backend and report which one wins.

    Returns:
        A result dictionary.
    """
    result: Dict[str, Any] = {"ok": False, "degraded": False, "backends": {}}
    try:
        from quant_system.config import settings as cfg
        from quant_system.utils.notifier import resolve_toast_backend
    except Exception as exc:
        result["detail"] = f"notifier import failed: {exc}"
        return result

    for requested in ("pyqt", "tk", "none"):
        try:
            backend, note = resolve_toast_backend(requested, cfg.DEFAULT_SETTINGS.notifier)
            result["backends"][requested] = {
                "class": type(backend).__name__,
                "note": note,
            }
            try:
                backend.stop()
            except Exception:  # pragma: no cover
                pass
        except Exception as exc:  # pragma: no cover
            result["backends"][requested] = {"class": "error", "note": str(exc)}

    try:
        auto_backend, auto_note = resolve_toast_backend("auto", cfg.DEFAULT_SETTINGS.notifier)
        result["auto"] = {"class": type(auto_backend).__name__, "note": auto_note}
        try:
            auto_backend.stop()
        except Exception:  # pragma: no cover
            pass
    except Exception as exc:  # pragma: no cover
        result["auto"] = {"class": "error", "note": str(exc)}

    chosen = str(result.get("auto", {}).get("class", ""))
    result["ok"] = True  # the null recorder always works, so this never hard-fails
    result["degraded"] = chosen == "NullToastBackend"
    result["detail"] = f"auto -> {chosen} ({result.get('auto', {}).get('note', '')})"
    if result["degraded"]:
        result["hint"] = (
            "No display detected. Toasts will be recorded, not shown. "
            "On a desktop either log in to a graphical session or run under Xvfb: "
            "Xvfb :99 & export DISPLAY=:99"
        )
    return result


def check_mt5() -> Dict[str, Any]:
    """Check whether the MetaTrader 5 terminal is reachable.

    Returns:
        A result dictionary.
    """
    result: Dict[str, Any] = {"ok": True, "degraded": True, "detail": ""}
    if sys.platform != "win32":
        result["detail"] = (
            f"not applicable - MetaTrader5 is Windows-only and this host is "
            f"{sys.platform}. Live MT5 execution requires Windows."
        )
        result["hint"] = "Use --broker simulated or --broker fix on this host."
        return result

    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except Exception as exc:
        result["ok"] = False
        result["degraded"] = False
        result["detail"] = f"MetaTrader5 package not installed: {exc}"
        result["hint"] = "pip install MetaTrader5"
        return result

    try:
        initialised = bool(mt5.initialize())
    except Exception as exc:
        result["detail"] = f"mt5.initialize() raised: {exc}"
        result["hint"] = "Start the MT5 terminal and log in to your account first."
        return result

    if not initialised:
        result["detail"] = f"mt5.initialize() returned False: {mt5.last_error()}"
        result["hint"] = (
            "Open the MT5 terminal, log in, and enable Algo Trading "
            "(toolbar button or Ctrl+E)."
        )
        return result

    try:
        info = mt5.terminal_info()
        account = mt5.account_info()
        result["terminal"] = {
            "company": getattr(info, "company", "?"),
            "build": getattr(info, "build", "?"),
            "trade_allowed": bool(getattr(info, "trade_allowed", False)),
            "login": getattr(account, "login", "?"),
            "server": getattr(account, "server", "?"),
            "balance": float(getattr(account, "balance", 0.0) or 0.0),
        }
        result["ok"] = True
        result["degraded"] = not bool(getattr(info, "trade_allowed", False))
        result["detail"] = (
            f"connected: {result['terminal']['company']} build "
            f"{result['terminal']['build']}, login {result['terminal']['login']}"
            f"@{result['terminal']['server']}"
        )
        if result["degraded"]:
            result["hint"] = (
                "Terminal is connected but trade_allowed is False. Enable "
                "'Algo Trading' in the MT5 toolbar (Ctrl+E)."
            )
    finally:
        try:
            mt5.shutdown()
        except Exception:  # pragma: no cover
            pass
    return result


def check_symbols() -> Dict[str, Any]:
    """Verify every configured MT5 symbol resolves on the terminal.

    Returns:
        A result dictionary; skipped (ok=True) when MT5 is unavailable.
    """
    result: Dict[str, Any] = {"ok": True, "degraded": False, "symbols": {}, "detail": "skipped (no MT5)"}
    if sys.platform != "win32":
        return result
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]

        from quant_system.config import settings as cfg
        from quant_system.execution.brokers.mt5_broker import MT5Broker

        if not mt5.initialize():
            result["detail"] = "MT5 not initialised"
            return result
        try:
            broker = MT5Broker()
            for spec in cfg.DEFAULT_SETTINGS.universe.assets:
                try:
                    resolved = broker._resolve_symbol(spec.symbol)
                    result["symbols"][spec.symbol] = resolved
                except Exception as exc:
                    result["symbols"][spec.symbol] = f"UNRESOLVED: {exc}"
            unresolved = [k for k, v in result["symbols"].items() if str(v).startswith("UNRESOLVED")]
            result["ok"] = not unresolved
            result["degraded"] = any(
                str(v) != spec.mt5_symbol
                for spec, v in zip(cfg.DEFAULT_SETTINGS.universe.assets, result["symbols"].values())
            )
            result["detail"] = (
                "all symbols resolved"
                if not unresolved
                else f"unresolved: {unresolved} - set mt5_symbol in config/settings.py"
            )
        finally:
            mt5.shutdown()
    except Exception as exc:  # pragma: no cover
        result["detail"] = f"symbol check failed: {exc}"
    return result


def check_data() -> Dict[str, Any]:
    """Check market-data reachability and the synthetic fallback.

    Returns:
        A result dictionary.
    """
    result: Dict[str, Any] = {"ok": False, "degraded": False, "detail": ""}
    try:
        from quant_system.data.ingestion import DataIngestion
    except Exception as exc:
        result["detail"] = f"ingestion import failed: {exc}"
        return result
    try:
        ingestion = DataIngestion(source="auto", use_cache=False)
        data = ingestion.fetch_universe(start="2024-01-01", end="2024-06-30", interval="1d")
    except Exception as exc:
        result["detail"] = f"fetch failed: {exc}"
        return result

    if not data:
        result["detail"] = "no data returned"
        return result

    sources = ingestion.sources_used
    result["symbols"] = sorted(data)
    result["bars"] = len(next(iter(data.values())))
    result["sources"] = sources
    using_synthetic = any(v == "synthetic" for v in sources.values())
    result["ok"] = True
    result["degraded"] = using_synthetic
    result["detail"] = (
        f"{len(data)} symbols x {result['bars']} bars via {sorted(set(sources.values()))}"
    )
    if using_synthetic:
        result["hint"] = (
            "Network unreachable - the synthetic generator is in use. Live and "
            "walk-forward runs are still valid, but prices are simulated."
        )
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_preflight(speak: bool = False) -> Dict[str, Any]:
    """Execute every check.

    Args:
        speak: Also play a test utterance.

    Returns:
        The aggregated report.
    """
    checks = {
        "core": check_core(),
        "audio": check_audio(speak=speak),
        "toasts": check_toasts(),
        "mt5": check_mt5(),
        "data": check_data(),
    }
    if checks["mt5"]["ok"] and sys.platform == "win32":
        checks["symbols"] = check_symbols()

    required_ok = all(checks[key]["ok"] for key in ("core", "data"))
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
            "display": bool(
                __import__("os").environ.get("DISPLAY")
                or __import__("os").environ.get("WAYLAND_DISPLAY")
                or sys.platform in ("win32", "darwin")
            ),
        },
        "checks": checks,
        "healthy": required_ok,
    }


def render(report: Dict[str, Any]) -> str:
    """Render the report as a console table.

    Args:
        report: Output of :func:`run_preflight`.

    Returns:
        The formatted multi-line report.
    """
    plat = report["platform"]
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("  quant_system :: ENVIRONMENT PRE-FLIGHT")
    lines.append("=" * 78)
    lines.append(
        f"  host    : {plat['system']} {plat['release']} | Python {plat['python']} "
        f"| display={'yes' if plat['display'] else 'no'}"
    )
    lines.append("")
    for name, check in report["checks"].items():
        ok = bool(check.get("ok"))
        degraded = bool(check.get("degraded"))
        lines.append(f"  {_mark(ok, degraded)} {_BOLD}{name.upper()}{_OFF}: {check.get('detail', '')}")
        for key in ("hint", "missing", "backends", "auto", "symbols", "sources", "terminal"):
            value = check.get(key)
            if value:
                lines.append(f"            {key}: {value}")
    lines.append("")
    lines.append("=" * 78)
    verdict = "READY" if report["healthy"] else "NOT READY"
    lines.append(f"  {verdict}: required subsystems (core, data) "
                 f"{'passed' if report['healthy'] else 'FAILED'}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Optional argument vector.

    Returns:
        ``0`` when the required subsystems are healthy.
    """
    parser = argparse.ArgumentParser(description="quant_system environment pre-flight check.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--speak", action="store_true", help="Also play a short test utterance aloud."
    )
    args = parser.parse_args(argv)

    report = run_preflight(speak=args.speak)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render(report))
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
