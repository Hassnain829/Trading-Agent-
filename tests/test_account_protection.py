"""Account protection: equity peak, half-risk throttle and total-drawdown kill-switch (live engine)."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

tmp = Path(tempfile.mkdtemp())
for attr, name in (("MEMORY_FILE", "memory.json"), ("RULES_FILE", "new_rules.json"), ("SETTINGS_FILE", "settings.json"),
                   ("RISK_STATE_FILE", "risk_state.json"), ("SHADOW_FILE", "shadow.json"), ("NEWS_CACHE_FILE", "news.json"),
                   ("AUDIT_LOCK_FILE", ".audit.lock"), ("LOCK_FILE", ".server.lock"),
                   ("JOURNAL_DIR", "journal")):
    setattr(config, attr, tmp / name)
config.AGENT_DIR = config.SHADOW_FILE.parent / "agent"  # the learning agent's files stay in the temp folder too
config.EXPLORE_FILE = config.AGENT_DIR / "explore_shadows.json"
config.DEEPSEEK_API_KEY = "test"
config.MAX_TOTAL_DRAWDOWN_PERCENT, config.DRAWDOWN_THROTTLE_PERCENT, config.KILL_SWITCH_CLOSE_POSITIONS = 10.0, 5.0, True
config.MAX_DAILY_LOSS_PERCENT = config.MAX_DAILY_PROFIT_PERCENT = 0.0  # keep the daily limits out of the way

import data_engine
import execution
import main

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ============================================================ live kill-switch and throttle
main.bot_state.update(circuit_breaker=False, profit_target_hit=False)
main._update_daily_risk(1000.0, 9)
main._update_daily_risk(1100.0, 9)
check("Peak follows new equity highs", main.bot_state["peak_equity"] == 1100.0 and not main.bot_state["risk_throttled"])
main._update_daily_risk(1040.0, 9)
check("-5.5% from the peak: half risk, no kill-switch", main.bot_state["risk_throttled"] and not main.bot_state["kill_switch"],
      main.bot_state["total_drawdown_pct"])
main._update_daily_risk(985.0, 9)
check("-10.5% from the peak: kill-switch trips", main.bot_state["kill_switch"] and main.bot_state["kill_switch_at"])
main.bot_state.update(guard_login=None, peak_equity=None, kill_switch=False, kill_switch_at=None, day_key=None,
                      day_login=None, day_start_equity=None)  # "restart"
main._peak_saved.update(login=None, peak=None)
main._update_daily_risk(1000.0, 9)
check("After a restart: kill-switch and peak restored", main.bot_state["kill_switch"] and main.bot_state["peak_equity"] == 1100.0)
real_key = data_engine.trading_day_key
data_engine.trading_day_key = lambda now=None: "2099-01-01"
main._update_daily_risk(1000.0, 9)
check("A new trading day does not clear the kill-switch", main.bot_state["kill_switch"] and main.bot_state["peak_equity"] == 1100.0)
data_engine.trading_day_key = real_key
main._update_daily_risk(500.0, 10)
check("Another login has its own peak", main.bot_state["peak_equity"] == 500.0 and not main.bot_state["kill_switch"])
saved = json.loads(config.RISK_STATE_FILE.read_text())
check("Saved per login next to the daily state", set(saved) == {"9", "10"} and saved["9"]["kill_switch"] is True
      and saved["9"]["peak_equity"] == 1100.0 and saved["10"]["peak_equity"] == 500.0, saved)
main._update_daily_risk(985.0, 9)

entry = {}
main.bot_state["is_running"] = True
asyncio.run(main._act_on_decision("EURUSD", {}, {"signal": "BUY", "strategy": "SCALP"}, entry))
check("Kill-switch blocks new entries", entry.get("action") == "blocked: drawdown kill-switch", entry)

closed = []
data_engine.get_account_snapshot = lambda: {"login": 9, "equity": 985.0, "balance": 985.0, "profit": 0.0}
execution.get_open_positions = lambda symbol=None: [{"ticket": 1, "symbol": "EURUSD", "side": "BUY", "profit": -5.0, "managed": True},
                                                    {"ticket": 2, "symbol": "USDJPY", "side": "SELL", "profit": 1.0, "managed": False}]
execution.close_position = lambda ticket: closed.append(ticket) or {"success": True, "ticket": ticket}
main._record_close_execution = lambda result, reason: closed.append(reason)
main._scalp_time_stops = lambda: None
main._watchdog_tick(False)
check("Kill-switch closes the engine's positions only", closed == [1, "KILL_SWITCH"] and main.bot_state["kill_switch_flattened"],
      closed)
main._watchdog_tick(False)
check("...once", closed == [1, "KILL_SWITCH"])

main.bot_state["equity"] = 985.0
main._reset_kill_switch()
saved = json.loads(config.RISK_STATE_FILE.read_text())
check("Reset: new peak at current equity, kill-switch cleared and saved",
      not main.bot_state["kill_switch"] and main.bot_state["peak_equity"] == 985.0
      and saved["9"]["kill_switch"] is False and saved["9"]["peak_equity"] == 985.0, saved["9"])
entry = {}
execution.get_open_positions = lambda symbol=None: []
config.MAX_OPEN_POSITIONS = 0
try:
    asyncio.run(main._act_on_decision("EURUSD", {}, {"signal": "BUY", "strategy": "SCALP"}, entry))
except Exception:
    pass  # past the kill-switch the fake decision is incomplete; only the gate matters here
check("After the reset entries are no longer blocked by it", entry.get("action") != "blocked: drawdown kill-switch", entry)
config.MAX_TOTAL_DRAWDOWN_PERCENT = config.DRAWDOWN_THROTTLE_PERCENT = 0.0
main._update_daily_risk(100.0, 9)
check("0 turns the protection off", not main.bot_state["kill_switch"] and not main.bot_state["risk_throttled"])

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
