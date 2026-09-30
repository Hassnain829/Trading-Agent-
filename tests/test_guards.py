"""Overextension guard, weekend guard and USD direction, including a replay of the real USDCAD trade."""
import asyncio
import copy
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "usdcad_trade_2026-09-25.json"
real = json.loads(FIXTURE.read_text(encoding="utf-8"))  # the live USDCAD trade the guards were built around

tmp = Path(tempfile.mkdtemp())
config.RISK_STATE_FILE, config.SHADOW_FILE, config.NEWS_CACHE_FILE = tmp / "risk_state.json", tmp / "shadow.json", tmp / "news.json"
config.AGENT_DIR = config.SHADOW_FILE.parent / "agent"  # the learning agent's files stay in the temp folder too
config.EXPLORE_FILE = config.AGENT_DIR / "explore_shadows.json"
config.TRADING_MODE = "DEMO"  # these checks place (stubbed) orders
for attr, name in (("MEMORY_FILE", "memory.json"), ("RULES_FILE", "new_rules.json"), ("SETTINGS_FILE", "settings.json")):
    setattr(config, attr, tmp / name)
config.CONFIDENCE_THRESHOLD = 65
config.OVEREXTENSION_GUARD = True

import ai_brain
import data_engine
import main

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- USD direction basics
check("fx_pair handles broker spellings", ai_brain.fx_pair("USDCADm") == ("USD", "CAD") and ai_brain.fx_pair("EURUSD.r") == ("EUR", "USD"))
check("fx_pair ignores metals/crypto", ai_brain.fx_pair("XAUUSD") is None and ai_brain.fx_pair("BTCUSDm") is None)
check("USD exposure", ai_brain.usd_exposure("USDCAD", "BUY") == "LONG_USD" and ai_brain.usd_exposure("EURUSD", "BUY") == "SHORT_USD"
      and ai_brain.usd_exposure("EURUSD", "SELL") == "LONG_USD" and ai_brain.usd_exposure("XAUUSD", "BUY") is None)
weak = {"symbol": "EURUSD", "day_change_pct": 0.2, "correlated_prices": {
    "GBPUSD": {"day_change_pct": 0.3}, "USDJPY": {"day_change_pct": -0.8}, "AUDUSD": {"day_change_pct": 0.1},
    "XAUUSD": {"day_change_pct": 1.5}}}
u = ai_brain.usd_direction(weak)
check("EURUSD/GBPUSD/AUDUSD up + USDJPY down = USD WEAKENING (gold ignored)", u["label"] == "WEAKENING"
      and u["pairs"] == 4 and u["agreement"] == 1.0 and u["usd_change_pct"] < -0.3, u)
check("Too few pairs -> UNKNOWN", ai_brain.usd_direction({"correlated_prices": {"EURUSD": {"day_change_pct": 0.1}}})["label"] == "UNKNOWN")

# ---------------------------------------------------------------- replay of the real USDCAD trade
if real is None:
    check("Real USDCAD record found in memory.json", False)
else:
    ctx = real["market_context"]
    market = {"symbol": "USDCAD", "bid": 1.41447, "ask": 1.41461, "mid": 1.41454, "spread": 14, "spread_price": 0.00014,
              "digits": 5, "point": 0.00001, "tick_size": 0.00001, "equity": 100.0, "currency": "USD",
              "h1_data": dict(ctx["h1"], structure=ctx["h1"].get("structure") or {}),
              "daily_data": dict(ctx["d1"], structure=ctx["d1"].get("structure") or {}),
              "correlated_prices": ctx["correlated_prices"]}
    guards = ai_brain.protective_guards(market, "USDCAD", "BUY")
    ids = {g["id"]: g["points"] for g in guards}
    print("   guards on the real trade:", ids)
    check("Real trade: H1 overextension (11.3 ATR) flagged", ids.get("G-OVEREXT-H1") == 15)
    check("Real trade: long USD against a weakening USD flagged", "G-USD" in ids)
    check("Real trade: combined penalty capped at 30", sum(ids.values()) == 30)
    payload = {"signal": "BUY", "base_confidence": 70, "confidence_score": 70, "triggered_rule_ids": [],
               "sl_atr_multiple": 1.5, "tp_atr_multiple": 3.0, "logic": real["logic"]}
    d = ai_brain._validate_decision(payload, market, "USDCAD", [], {"applied_rules": []})
    check("Real trade now becomes HOLD (70 -> 40 < 65)", d["signal"] == "HOLD" and d["confidence_score"] == 40
          and d["raw_signal"] == "BUY" and "[GUARDS -30" in d["logic"], d["logic"][:120])
    check("Guards shown as applied rules; USD reading attached", any(r["id"] == "G-USD" for r in d["applied_rules"])
          and d["usd_direction"]["label"] == "WEAKENING")
    prompt = ai_brain.build_system_prompt(market, "USDCAD", [])
    check("Prompt tells the AI the USD direction and that BUY USDCAD fights it",
          "USD DIRECTION (computed" in prompt and "USD WEAKENING" in prompt and "BUY USDCAD = LONG USD: AGAINST" in prompt)
    check("Prompt tells the AI not to double-deduct overextension", "do NOT deduct for them" in prompt)
    config.OVEREXTENSION_GUARD = False
    ids_off = {g["id"] for g in ai_brain.protective_guards(market, "USDCAD", "BUY")}
    check("Overextension guard can be switched off (USD guard stays)", ids_off == {"G-USD"}, ids_off)
    config.OVEREXTENSION_GUARD = True

sell = {"symbol": "EURUSD", "h1_data": {"ema_distance_atr": -7.0, "rsi14": 22.0}, "daily_data": {"ema_distance_atr": -2.0, "rsi14": 40.0},
        "correlated_prices": {}}
check("SELL mirrored: stretched below EMA + low RSI penalised",
      {g["id"]: g["points"] for g in ai_brain.protective_guards(sell, "EURUSD", "SELL")} == {"G-OVEREXT-H1": 10, "G-RSI-H1": 5})
check("Trend-aligned, not stretched: no guards",
      ai_brain.protective_guards({"symbol": "EURUSD", "h1_data": {"ema_distance_atr": 1.5, "rsi14": 58},
                                  "daily_data": {"ema_distance_atr": 1.0, "rsi14": 55}, "correlated_prices": {}}, "EURUSD", "BUY") == [])

# ---------------------------------------------------------------- weekend clock (DST aware)
clock = data_engine.fx_week_clock
fri_summer = clock(datetime(2026, 9, 25, 20, 29, tzinfo=timezone.utc))
check("Fri 20:29 UTC in summer (16:29 NY) -> 31 min to close", not fri_summer["closed"] and fri_summer["minutes_to_close"] == 31.0, fri_summer)
fri_winter = clock(datetime(2026, 1, 9, 21, 30, tzinfo=timezone.utc))
check("Fri 21:30 UTC in winter (16:30 NY) -> 30 min to close", fri_winter["minutes_to_close"] == 30.0, fri_winter)
check("Saturday -> closed", clock(datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc))["closed"])
check("Sunday 20:00 UTC (16:00 NY) -> still closed", clock(datetime(2026, 9, 27, 20, 0, tzinfo=timezone.utc))["closed"])
sun_open = clock(datetime(2026, 9, 27, 22, 0, tzinfo=timezone.utc))
check("Sunday 22:00 UTC (18:00 NY) -> open, ~5 days to close", not sun_open["closed"] and 5 * 1440 - 120 <= sun_open["minutes_to_close"] <= 5 * 1440, sun_open)
wed = clock(datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc))
check("Midweek -> open, days to close", not wed["closed"] and wed["minutes_to_close"] > 2 * 1440)

# ---------------------------------------------------------------- weekend entry gate + auto-close
real_clock = data_engine.fx_week_clock
data_engine.fx_week_clock = lambda now=None: real_clock(datetime(2026, 9, 25, 20, 29, tzinfo=timezone.utc))
config.WEEKEND_ENTRY_CUTOFF_HOURS = 3
block = main._weekend_entry_block("USDCAD", "Forex\\Majors\\USDCAD")
check("The real trade's time is now inside the entry pause", block and "paused 3h before the Friday close" in block, block)
check("Crypto exempt", main._weekend_entry_block("BTCUSD", "Crypto\\BTCUSD") is None)
config.WEEKEND_ENTRY_CUTOFF_HOURS = 0
check("Cutoff 0 = off", main._weekend_entry_block("USDCAD", "Forex\\USDCAD") is None)
config.WEEKEND_ENTRY_CUTOFF_HOURS = 3


async def gated_scan():
    main.bot_state["is_running"] = True
    data_engine.fetch_multi_timeframe_data = lambda s: {"tradeable": True, "market_idle": False, "path": "Forex\\USDCAD"}
    asked = []
    main.ai_brain.get_ai_decision = lambda m, s: asked.append(s)
    await main.process_symbol("USDCAD")
    return asked


check("Paused symbols are not even sent to the AI (saves quota)", asyncio.run(gated_scan()) == [])

positions = [{"ticket": 1, "symbol": "USDCAD", "side": "BUY", "volume": 0.01, "profit": -0.2, "managed": True},
             {"ticket": 2, "symbol": "EURUSD", "side": "SELL", "volume": 0.01, "profit": 0.5, "managed": False},
             {"ticket": 3, "symbol": "BTCUSD", "side": "BUY", "volume": 0.01, "profit": 1.0, "managed": True}]
closed, recorded = [], []
main.execution.get_open_positions = lambda symbol=None: positions
main.execution.close_position = lambda t: closed.append(t) or {"success": True, "ticket": t, "symbol": "USDCAD"}
main._record_close_execution = lambda result, reason, market=None: recorded.append(reason)
data_engine.trades_weekends = lambda s, path=None: s.startswith("BTC")
config.WEEKEND_CLOSE = False
check("Auto-close off by default: nothing closed", main._weekend_close_positions() == 0 and not closed)
config.WEEKEND_CLOSE = True
data_engine.fx_week_clock = lambda now=None: real_clock(datetime(2026, 9, 25, 19, 0, tzinfo=timezone.utc))
check("2h before close: not yet", main._weekend_close_positions() == 0 and not closed)
data_engine.fx_week_clock = lambda now=None: real_clock(datetime(2026, 9, 25, 20, 29, tzinfo=timezone.utc))
n = main._weekend_close_positions()
check("31 min before close: only the bot's forex trade is closed (manual + crypto kept)", n == 1 and closed == [1]
      and recorded == ["WEEKEND_CLOSE"], (closed, recorded))
data_engine.fx_week_clock = real_clock

# ---------------------------------------------------------------- settings
data_engine.get_account_snapshot = lambda: None
res = main.api_save_settings(main.SettingsUpdate(overextension_guard=False, weekend_entry_cutoff_hours=4.5, weekend_close=True))
check("Guard settings saved + applied", res["settings"]["overextension_guard"] is False and config.WEEKEND_ENTRY_CUTOFF_HOURS == 4.5
      and config.WEEKEND_CLOSE is True and json.loads(config.SETTINGS_FILE.read_text())["weekend_entry_cutoff_hours"] == 4.5)
g = main._status_payload()["guardrails"]
check("Status payload exposes guard settings + week clock", g["weekend_close"] is True and "fx_week" in main._status_payload())

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
