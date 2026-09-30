"""Intraday strategy (break and retest) and running it next to the scalper: rules, position policy, risk, time stops."""
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
config.AGENT_ENABLED, config.AGENT_EXPLORE = True, False  # no rewards yet: the AI decides
config.AGENT_SHADOW_UNTIL_LEARNED = False  # this suite tests the order path (shadow-only learning: test_agent)
config.MAX_OPEN_POSITIONS = 0
config.LOSS_COOLDOWN_MINUTES = 0
config.MAX_TOTAL_DRAWDOWN_PERCENT = config.DRAWDOWN_THROTTLE_PERCENT = 0.0
config.SCALP_SESSION_START_LONDON, config.SCALP_SESSION_END_NEW_YORK = 7, 16

import ai_brain
import data_engine
import execution
import intraday
import journal
import main
import scalper

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ============================================================ 1. the day walk on constructed charts
# Wednesday 14 Jan 2026 (winter): server clock = New York + 7h; the London open (07:00 London = 07:00 UTC)
# is 09:00 server time. The trading day starts at 00:00 server time.
DAY = int(pd.Timestamp("2026-01-14 00:00").value // 10**9)


def frame(times, o, h, l, c, spread=2):
    return pd.DataFrame({"time": np.asarray(times, dtype="int64"), "open": o, "high": h, "low": l, "close": c,
                         "tick_volume": 100, "spread": spread})


def build(m15_path):
    """m15_path: list of (open, high, low, close) for the day's M15 bars from 00:00 server time."""
    n = len(m15_path)
    t15 = DAY + 900 * np.arange(n)
    o, h, l, c = (np.array([bar[k] for bar in m15_path]) for k in range(4))
    # 150 flat days before, yesterday's range 1.0950 - 1.1050
    d_times = DAY - 86400 * np.arange(150, 0, -1)
    d1 = frame(d_times, 1.1, 1.105, 1.095, 1.1)
    prev15 = DAY - 900 * np.arange(120, 0, -1)  # yesterday's M15 bars, flat, so ATR has history
    m15 = pd.concat([frame(prev15, 1.1, 1.1004, 1.0996, 1.1), frame(t15, o, h, l, c)], ignore_index=True)
    t5 = DAY - 300 * 400 + 300 * np.arange(400 + 3 * n)
    m5 = frame(t5, 1.1, 1.1002, 1.0998, 1.1)
    h1 = frame(DAY - 3600 * np.arange(300, 0, -1), 1.1, 1.1005, 1.0995, 1.1)
    return intraday.prepare(scalper.prepare({"M5": m5, "M15": m15, "H1": h1, "D1": d1}), new_york_plus_7=True)


def asian():  # 36 quiet bars 00:00-09:00 server time: Asian range 1.0990 - 1.1010
    return [(1.1000, 1.1010, 1.0990, 1.1000)] * 36


# London: break above 1.1010, pull back to it, then a bullish close above it
clean = asian() + [
    (1.1000, 1.1006, 1.0998, 1.1004),   # 09:00-09:15 inside the range
    (1.1004, 1.1022, 1.1003, 1.1018),   # break: closes above the Asian high 1.1010
    (1.1018, 1.1020, 1.1011, 1.1013),   # retest: low touches the level
    (1.1013, 1.1026, 1.1012, 1.1024),   # bullish close above the level -> entry
    (1.1024, 1.1030, 1.1020, 1.1028),
]
p = build(clean)
q = p.a["M15"]
k_entry = int(np.searchsorted(q["time"], DAY + 900 * 39))
day = intraday._today(p, k_entry)
walked = intraday._walk_day(p, day)
entries = [e for e in walked["entries"] if e["level_name"] == "ASIA_HIGH"]
check("Clean break and retest of the Asian high -> one BUY entry on the confirming candle",
      len(entries) == 1 and entries[0]["side"] == "BUY" and entries[0]["k"] == k_entry, walked["entries"])
e = entries[0]
atr = q["atr14"][k_entry]
check("Structural stop just beyond the retest low; target 2R",
      abs((q["close"][k_entry] - e["sl_distance"]) - (min(q["low"][k_entry - 2:k_entry + 1]) - 0.1 * atr)) < 1e-9
      and abs(e["tp_distance"] - 2 * e["sl_distance"]) < 1e-12, e)
check("Yesterday's high/low are levels too, not broken here", walked["levels"]["PDH"]["state"] == "WAIT_BREAK"
      and walked["levels"]["PDH"]["level"] == 1.105)
live = intraday.evaluate(p, k_entry)
check("Live evaluation on the entry candle -> SETUP with the level in the reason",
      live["stage"] == "SETUP" and live["setup"]["strategy"] == "INTRADAY" and "Asian high" in live["setup"]["reason"],
      live["reason"])
later = intraday.evaluate(p, k_entry + 1)
check("Next candle: no second entry on the same level", later["setup"] is None and later["stage"] == "I_ENTERED",
      later["stage"])
check("24/7 (default): the Asian hours are not off-session", config.TRADE_ALL_HOURS
      and intraday.evaluate(p, int(np.searchsorted(q["time"], DAY + 900 * 20)))["stage"] != "OFF_SESSION")
config.TRADE_ALL_HOURS = False
windowed = build(clean)
check("Trading-hours window mode: before the London open is off session",
      intraday.evaluate(windowed, int(np.searchsorted(windowed.a["M15"]["time"], DAY + 900 * 20)))["stage"] == "OFF_SESSION")
config.TRADE_ALL_HOURS = True

# Yesterday's high breaks in the middle of the Asian session (03:00 server time)
asia_break = asian()[:9] + [
    (1.1000, 1.1046, 1.0998, 1.1045),   # rallies up to yesterday's high
    (1.1045, 1.1062, 1.1044, 1.1058),   # breaks yesterday's high 1.1050
    (1.1058, 1.1060, 1.1050, 1.1053),   # retest touches it
    (1.1053, 1.1070, 1.1052, 1.1066),   # bullish close above -> entry
] + [(1.1066, 1.1070, 1.1060, 1.1065)] * 30
pa = build(asia_break)
k_asia = int(np.searchsorted(pa.a["M15"]["time"], DAY + 900 * 12))
hit = intraday.evaluate(pa, k_asia)
check("24/7: yesterday's high broken and retested during the Asian session -> BUY entry at 03:00",
      hit["setup"] is not None and hit["setup"]["level_name"] == "PDH" and hit["setup"]["side"] == "BUY", hit["reason"])
walked_asia = intraday._walk_day(pa, intraday._today(pa, len(pa.a["M15"]["close"]) - 1))
check("...while the Asian range itself is only traded after the London open (it is still forming)",
      all(e["level_name"] == "PDH" for e in walked_asia["entries"]), walked_asia["entries"])
config.TRADE_ALL_HOURS = False
pw = build(asia_break)
check("Window mode: the same Asian-session break is not traded",
      intraday.evaluate(pw, int(np.searchsorted(pw.a["M15"]["time"], DAY + 900 * 12)))["setup"] is None)
config.TRADE_ALL_HOURS = True

failed = asian() + [
    (1.1000, 1.1006, 1.0998, 1.1004),
    (1.1004, 1.1022, 1.1003, 1.1018),   # break
    (1.1018, 1.1019, 1.0990, 1.0992),   # closes back well below the level -> failed break
    (1.0992, 1.1016, 1.0991, 1.1015),
]
p = build(failed)
w = intraday._walk_day(p, intraday._today(p, len(p.a["M15"]["close"]) - 1))
check("A break that falls back through the level fails: no trade", not w["entries"]
      and w["levels"]["ASIA_HIGH"]["state"] == "FAILED", w["levels"]["ASIA_HIGH"])

wide = asian() + [
    (1.1000, 1.1006, 1.0998, 1.1004),
    (1.1004, 1.1022, 1.1003, 1.1018),   # break
    (1.1018, 1.1019, 1.0950, 1.1012),   # a deep spike (closes back above): the retest low is far away
    (1.1012, 1.1026, 1.1011, 1.1024),   # bullish close -> the structural stop would be > 3 ATR
]
p = build(wide)
w = intraday._walk_day(p, intraday._today(p, len(p.a["M15"]["close"]) - 1))
check("Structural stop too wide -> the trade is REJECTED, the stop is never moved inside the swing",
      not w["entries"] and w["levels"]["ASIA_HIGH"]["state"] == "REJECTED", w["levels"]["ASIA_HIGH"])

near = asian() + [
    (1.1000, 1.1006, 1.0998, 1.1004),
    (1.1004, 1.1022, 1.1003, 1.1018),   # break
    (1.1018, 1.1020, 1.1014, 1.1016),   # comes back to 4 pips above the level: not a strict touch (0.1 ATR)
    (1.1016, 1.1026, 1.1015, 1.1024),   # bullish close
]
p = build(near)
k_last = len(p.a["M15"]["close"]) - 1
strict_near, relaxed_near = intraday.evaluate(p, k_last), intraday.evaluate(p, k_last, relaxed=True)
check("Near miss (retest stopped short of the level): no strict setup, but a relaxed one for exploration",
      strict_near["setup"] is None and relaxed_near["setup"] is not None and relaxed_near["setup"]["side"] == "BUY"
      and relaxed_near["setup"]["bars_since_break"] == 2, (strict_near["stage"], relaxed_near["stage"]))

# ============================================================ 2. position policy between strategies
open_positions = []
executed = []
execution.get_open_positions = lambda symbol=None: [x for x in open_positions if symbol in (None, x["symbol"])]


def fake_execute(symbol, side, sl, tp, risk, comment=None, **kw):
    executed.append({"symbol": symbol, "side": side, "risk": risk, "comment": comment,
                     "currency_ignore": kw.get("currency_ignore")})
    raise execution.TradeExecutionError("test stop")  # everything before the order passed


execution.execute_trade = fake_execute
main.bot_state.update(is_running=True, kill_switch=False, circuit_breaker=False, profit_target_hit=False,
                      risk_percent=2.0, risk_throttled=False)


def act(strategy, side):
    entry = {}
    decision = {"signal": side, "strategy": strategy, "stop_loss": 1.09, "take_profit": 1.12, "sl_distance": 0.001,
                "tp_distance": 0.002, "entry_reference": 1.1, "atr_reference": 0.001, "confidence_score": 80,
                "threshold": 60}
    asyncio.run(main._act_on_decision("EURUSD", {}, decision, entry))
    return entry.get("action")


def position(ticket, side, tag):
    return {"ticket": ticket, "symbol": "EURUSD", "side": side, "volume": 0.1, "profit": 0.0, "managed": True,
            "comment": f"{config.ORDER_COMMENT}{tag}", "time": data_engine.utc_now_iso()}


open_positions[:] = [position(1, "BUY", "-S")]
data_engine.hedging_account = lambda: True
check("Hedging account: scalp BUY open -> an intraday SELL on the same pair still trades (independent strategies)",
      act("INTRADAY", "SELL") == "rejected: test stop" and executed[-1]["side"] == "SELL", executed[-1:])
check("...and its currency cap ignores the scalp's positions (own budget per strategy)",
      executed[-1]["currency_ignore"] == [f"{config.ORDER_COMMENT}-S"], executed[-1])
data_engine.hedging_account = lambda: False
check("Netting account: the opposite side is refused (it would net the scalp off)",
      act("INTRADAY", "SELL") == "blocked: opposite trade of another strategy (netting account)")
data_engine.hedging_account = lambda: True
executed.clear()
check("...an intraday BUY on the same pair is allowed (same direction)", act("INTRADAY", "BUY") == "rejected: test stop"
      and executed and executed[-1]["comment"].endswith("-I"), executed[-1:])
open_positions.append(position(2, "BUY", "-I"))
check("Each strategy holds one position per pair: a second intraday BUY is a duplicate",
      act("INTRADAY", "BUY") == "blocked: same direction already open")
open_positions[:] = [{**position(3, "SELL", ""), "managed": False}]
check("A manual trade on the pair: same direction is still a duplicate", act("SCALP", "SELL") == "blocked: same direction already open")
config.MAX_OPEN_POSITIONS = 1
open_positions[:] = [{**position(4, "BUY", "-S"), "symbol": "GBPUSD"}]
check("Position limit per strategy: one scalp open elsewhere blocks a second scalp ...",
      act("SCALP", "BUY") == "blocked: max open positions")
check("... but not an intraday trade", act("INTRADAY", "BUY") == "rejected: test stop")
config.MAX_OPEN_POSITIONS = 0

# ============================================================ 3. risk per strategy
open_positions[:] = []
executed.clear()
config.INTRADAY_RISK_PERCENT = 0.0
act("INTRADAY", "BUY")
check("Intraday risk 0 -> uses the scalp risk (2%)", executed[-1]["risk"] == 2.0, executed[-1])
config.INTRADAY_RISK_PERCENT = 0.5
act("INTRADAY", "BUY")
act("SCALP", "BUY")
check("Intraday risk set to 0.5% -> intraday 0.5%, scalp stays 2%, orders tagged -I / -S",
      executed[-2]["risk"] == 0.5 and executed[-1]["risk"] == 2.0 and executed[-2]["comment"].endswith("-I")
      and executed[-1]["comment"].endswith("-S"), executed[-2:])

# ============================================================ 4. time stops per strategy
now = int(time.time())
config.MEMORY_FILE.write_text(json.dumps([
    {"id": "a", "ticket": 11, "status": "CONFIRMED", "strategy": "SCALP", "time_stop_minutes": 60, "symbol": "EURUSD",
     "side": "BUY", "timestamp": data_engine.utc_now_iso()},
    {"id": "b", "ticket": 12, "status": "CONFIRMED", "strategy": "INTRADAY", "time_stop_minutes": 360, "symbol": "GBPUSD",
     "side": "BUY", "timestamp": data_engine.utc_now_iso()},
    {"id": "c", "ticket": 13, "status": "CONFIRMED", "strategy": "INTRADAY", "time_stop_minutes": 360, "symbol": "USDJPY",
     "side": "BUY", "timestamp": data_engine.utc_now_iso()},
]))
main._history_cache["stamp"] = None
iso = lambda minutes: pd.Timestamp(now - minutes * 60, unit="s", tz="UTC").isoformat()
open_positions[:] = [{**position(11, "BUY", "-S"), "symbol": "EURUSD", "time": iso(61)},
                     {**position(12, "BUY", "-I"), "symbol": "GBPUSD", "time": iso(120)},
                     {**position(13, "BUY", "-I"), "symbol": "USDJPY", "time": iso(365)}]
closed = []
execution.close_position = lambda ticket: closed.append(ticket) or {"success": True, "ticket": ticket}
main._record_close_execution = lambda result, reason, market=None: None
data_engine.server_now_epoch = lambda symbol: float(now)
main._scalp_time_stops()
check("Time stops per strategy: scalp after 60 min, intraday after 6 h (a 2 h intraday trade stays open)",
      sorted(closed) == [11, 13], closed)

# ============================================================ 5. the intraday pipeline end to end
open_positions[:] = []
executed.clear()
config.SCALP_ENABLED, config.INTRADAY_ENABLED = False, True
setup = {**entries[0], "strategy": "INTRADAY", "trend": "UP", "risk_reward": 2.0, "entry_ref": 1.1024,
         "reason": "BUY the Asian high break and retest at 1.101: broke, came back to the level and closed beyond it again"}
fake_result = {"stage": "SETUP", "setup": setup, "reason": setup["reason"], "levels": walked["levels"],
               "recent": {"closes": [1.1], "highs": [1.1], "lows": [1.1], "rsi14": [50]},
               "features": {"feature_version": scalper.FEATURE_VERSION, "side": "BUY"}}
main._evaluate_intraday = lambda symbol, spread: fake_result
data_engine.last_closed_bar_time = lambda symbol, tf: 123456
scalper.in_session = lambda now=None: True
market = {"symbol": "EURUSD", "bid": 1.1024, "ask": 1.1025, "mid": 1.10245, "spread": 10, "spread_price": 0.0001,
          "digits": 5, "point": 0.00001, "tick_size": 0.00001, "equity": 1000.0, "balance": 1000.0, "account_mode": "DEMO",
          "broker": "X", "tick_epoch": now, "day_change_pct": 0.1, "h1_data": {"atr14": 0.001}, "daily_data": {}}


async def fake_prepare(symbol):
    return dict(market)


main._prepare_market = fake_prepare
data_engine.fetch_correlated_asset_prices = lambda *a, **k: {}
answers = []


def fake_chat(messages, **kw):
    answers.append(messages[0]["content"])
    return {"content": json.dumps({"decision": "CONFIRM", "confidence_score": 80, "logic": "clean retest"})}


ai_brain.deepseek_chat = fake_chat
outcome = asyncio.run(main.process_symbol("EURUSD"))
last = list(journal.iter_entries(1))[-1]
check("Intraday setup -> AI asked with the INTRADAY reviewer prompt", answers and "INTRADAY SETUP" in answers[-1]
      and "break and retest" in answers[-1], answers[-1][:120] if answers else None)
check("...confirmed -> order sent with the intraday tag and risk (0.5%)", executed and executed[-1]["comment"].endswith("-I")
      and executed[-1]["risk"] == 0.5, executed[-1:])
check("...journaled as an intraday evaluation with its level and the AI answer",
      last["kind"] == "intraday_eval" and last["strategy"] == "INTRADAY" and last["setup"]["level_name"] == "ASIA_HIGH"
      and last["ai"]["base_confidence"] == 80, {k: last.get(k) for k in ("kind", "strategy", "action")})
check("Dashboard view per pair for the intraday strategy", main.bot_state["intraday"]["EURUSD"]["stage"] == "AI_CONFIRMED")
shadow = json.loads(config.SHADOW_FILE.read_text())[-1]
check("The order was refused (test stop): the TAKE is still followed as a shadow trade, so its reward is learned",
      last["action"] == "rejected: test stop" and shadow["blocked_by"] == ["SAFETY"] and shadow["strategy"] == "INTRADAY"
      and shadow["agent"]["action"] == "blocked" and shadow["agent"]["context"]["ai"]["asked"]
      and last.get("shadow_id") == shadow["id"], (last.get("action"), shadow.get("blocked_by")))
check("Same M15 bar -> not evaluated again", asyncio.run(main.process_intraday("EURUSD")) == "unchanged")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
