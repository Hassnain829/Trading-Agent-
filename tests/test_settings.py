"""Dashboard settings: validation, live apply, persistence, reset, daily profit target, fixed-lot sizing."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

tmp = Path(tempfile.mkdtemp())
config.RISK_STATE_FILE, config.SHADOW_FILE, config.NEWS_CACHE_FILE = tmp / "risk_state.json", tmp / "shadow.json", tmp / "news.json"
for attr, name in (("MEMORY_FILE", "memory.json"), ("RULES_FILE", "new_rules.json"), ("SETTINGS_FILE", "settings.json")):
    setattr(config, attr, tmp / name)
config.SYMBOLS_DEMO = config.SYMBOLS = ["EURUSDm", "GBPUSDm"]  # the stale Exness names from .env
config.SYMBOLS_LIVE = ["EURUSD"]

import MetaTrader5 as mt5
import data_engine
import execution
import main
import settings_store
from fastapi import HTTPException
from pydantic import ValidationError

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


BROKER = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AMD"]
data_engine.broker_symbols = lambda: [{"name": s, "description": s, "path": "x"} for s in BROKER]
data_engine.get_account_snapshot = lambda: {"login": 1, "server": "MetaQuotes-Demo", "account_mode": "DEMO", "equity": 100.0}

s0 = main.api_get_settings()
check("GET /api/settings: current + .env defaults + bounds", s0["settings"]["symbols_demo"] == ["EURUSDm", "GBPUSDm"]
      and s0["defaults"]["max_open_positions"] == config.MAX_OPEN_POSITIONS and "risk_percent" in s0["bounds"]
      and not any(s0["overridden"].values()))

try:
    main.api_save_settings(main.SettingsUpdate(symbols_demo=["EURUSDm", "NOTREAL"]))
    check("Unknown symbol rejected", False)
except HTTPException as exc:
    check("Unknown symbol rejected with the bad names", exc.status_code == 422 and "NOTREAL" in exc.detail, exc.detail)

for bad in ({"risk_percent": 9}, {"max_open_positions": -1}, {"confidence_threshold": 20}, {"sizing_mode": "YOLO"},
            {"symbols_demo": []}, {"max_daily_profit_percent": 500}):
    try:
        main.SettingsUpdate(**bad)
        check(f"Out-of-range rejected {bad}", False)
    except ValidationError:
        check(f"Out-of-range rejected {bad}", True)

res = main.api_save_settings(main.SettingsUpdate(
    symbols_demo=["eurusd", "GBPUSD", "EURUSD", "xauusd"], max_open_positions=3, max_daily_loss_percent=4,
    max_daily_profit_percent=2.5, confidence_threshold=70, risk_percent=0.753, scan_interval_seconds=300))
check("Demo list saved de-duplicated; traded symbols resolved to broker names", res["settings"]["symbols_demo"] == ["EURUSD", "GBPUSD", "XAUUSD"]
      and res["active_symbols"] == ["EURUSD", "GBPUSD", "XAUUSD"], res["active_symbols"])
check("Applied live to config", config.SYMBOLS == ["EURUSD", "GBPUSD", "XAUUSD"] and config.MAX_OPEN_POSITIONS == 3
      and config.CONFIDENCE_THRESHOLD == 70 and config.MAX_DAILY_PROFIT_PERCENT == 2.5)
check("Engine state synced (risk rounded, interval)", main.bot_state["risk_percent"] == 0.75 and main.bot_state["interval"] == 300)
check("Status payload reflects new limits", main._status_payload()["guardrails"]["max_open_positions"] == 3
      and main._status_payload()["symbols"] == ["EURUSD", "GBPUSD", "XAUUSD"])
saved = json.loads(config.SETTINGS_FILE.read_text())
check("Persisted to settings.json", saved["symbols_demo"] == ["EURUSD", "GBPUSD", "XAUUSD"] and saved["max_open_positions"] == 3)
check("Overridden flags shown", res["overridden"]["symbols_demo"] and res["overridden"]["confidence_threshold"])

# risk/interval changed from the header / risk panel also persist
asyncio.run(main.api_control(main.ControlRequest(risk_percent=1.25, interval=60)))
saved = json.loads(config.SETTINGS_FILE.read_text())
check("Risk panel + interval changes persist", saved["risk_percent"] == 1.25 and saved["scan_interval_seconds"] == 60)

# simulate a restart: .env defaults back in config, then load settings.json
config.SYMBOLS_DEMO, config.MAX_OPEN_POSITIONS, config.DEFAULT_RISK_PERCENT = ["EURUSDm", "GBPUSDm"], 5, 1.0
settings_store.load_saved()
check("Restart: saved settings override .env", config.SYMBOLS_DEMO == ["EURUSD", "GBPUSD", "XAUUSD"]
      and config.MAX_OPEN_POSITIONS == 3 and config.DEFAULT_RISK_PERCENT == 1.25)

# ---------------------------------------------------------------- daily profit target
main.bot_state.update(day_key=None, day_start_equity=None, equity=100.0)
main._update_daily_risk(100.0)
main._update_daily_risk(101.0)
check("Below target: entries allowed", not main.bot_state["profit_target_hit"] and main.bot_state["day_pnl_pct"] == 1.0)
main._update_daily_risk(102.6)
check("Profit target 2.5% reached -> entries paused", main.bot_state["profit_target_hit"])
main.bot_state["equity"] = 102.6
main.api_save_settings(main.SettingsUpdate(max_daily_profit_percent=5))
check("Raising the target re-evaluates and resumes entries", not main.bot_state["profit_target_hit"])
main._update_daily_risk(95.5)
check("Daily loss limit 4% trips the breaker", main.bot_state["circuit_breaker"] and main.bot_state["daily_drawdown_pct"] > 4)


async def blocked_run():
    main.bot_state["is_running"] = True
    market = {"tradeable": True, "market_idle": False, "equity": 100, "digits": 5, "h1_data": {}, "daily_data": {}}
    data_engine.fetch_multi_timeframe_data = lambda s: market
    data_engine.fetch_correlated_asset_prices = lambda *a, **k: {}
    main.ai_brain.get_ai_decision = lambda m, s: {"signal": "BUY", "confidence_score": 80, "logic": "x", "applied_rules": [],
                                                  "timestamp": "t"}
    sent = []
    main.execution.get_open_positions = lambda *a: sent.append(a) or []
    await main.process_symbol("EURUSD")
    return sent


check("Breaker blocks the trade before any position check", asyncio.run(blocked_run()) == [])

# ---------------------------------------------------------------- reset
res = main.api_reset_settings()
check("Reset to .env", config.SYMBOLS_DEMO == settings_store.ENV_DEFAULTS["symbols_demo"] and not any(res["overridden"].values())
      and json.loads(config.SETTINGS_FILE.read_text()) == {})

# ---------------------------------------------------------------- fixed-lot sizing
info = SimpleNamespace(digits=5, point=0.00001, trade_tick_size=0.00001, trade_tick_value=1.0, trade_tick_value_loss=1.0,
                       trade_contract_size=100000.0, volume_min=0.01, volume_max=100.0, volume_step=0.01)
mt5.symbol_info = lambda s: info
try:  # 0.03 lots x 20 pips = $6 = 6% of a $100 account
    execution.fixed_position_size("EURUSD", 1.09800, 0.03, 100.0, entry_price=1.10000)
    check("Fixed lot risking > 5% refused", False)
except execution.TradeExecutionError as exc:
    check("Fixed lot risking > 5% refused", "hard cap" in str(exc), str(exc)[:90])
vol = execution.fixed_position_size("EURUSD", 1.09800, 0.037, 1000.0, entry_price=1.10000)
check("Fixed lot snapped to the volume step (0.037 -> 0.03) on a larger account", vol == 0.03, vol)

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
