"""Verify the ATR stop engine in ai_brain and fill-time re-anchoring in execution."""
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
config.AGENT_DIR = config.SHADOW_FILE.parent / "agent"  # the learning agent's files stay in the temp folder too
config.EXPLORE_FILE = config.AGENT_DIR / "explore_shadows.json"
config.TRADING_MODE = "DEMO"  # these checks place (stubbed) orders
config.MEMORY_FILE, config.RULES_FILE = tmp / "memory.json", tmp / "new_rules.json"

import MetaTrader5 as mt5
import ai_brain
import execution

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


structure = {"low": 1.09700, "high": 1.10300, "closes": [], "pattern": "BULLISH_HH_HL"}
market = {
    "symbol": "EURUSDm", "bid": 1.10000, "ask": 1.10010, "mid": 1.10005, "spread": 10, "spread_price": 0.00010,
    "digits": 5, "point": 0.00001, "tick_size": 0.00001, "equity": 1000.0, "currency": "USD",
    "h1_data": {"atr14": 0.00200, "atr_ratio": 1.1, "structure": structure},
    "daily_data": {"atr14": 0.00900, "structure": {}}, "correlated_prices": {},
}


def decide(payload, mkt=None):
    return ai_brain._validate_decision(payload, mkt or market, "EURUSDm", [], {"applied_rules": []})


base = {"signal": "BUY", "base_confidence": 80, "confidence_score": 80, "triggered_rule_ids": [], "logic": "trend"}

r = decide({**base, "sl_atr_multiple": 1.5, "tp_atr_multiple": 3.0})
check("BUY exact SL = ask - 1.5 x ATR", r["stop_loss"] == 1.09710, r["stop_loss"])
check("BUY exact TP = ask + 3.0 x ATR", r["take_profit"] == 1.10610, r["take_profit"])
check("Distances, R:R and source", r["sl_distance"] == 0.003 and r["tp_distance"] == 0.006
      and r["risk_reward"] == 2.0 and r["stops_source"] == "AI_ATR" and r["signal"] == "BUY")

r = decide({**base, "signal": "SELL", "sl_atr_multiple": 2.0, "tp_atr_multiple": 4.0})
check("SELL exact SL = bid + 2 x ATR / TP = bid - 4 x ATR", r["stop_loss"] == 1.10400 and r["take_profit"] == 1.09200,
      f"{r['stop_loss']} / {r['take_profit']}")

r = decide({**base, "sl_atr_multiple": 0.4, "tp_atr_multiple": 9})
check("Multiples clamped to bounds (0.4 -> 1.0, 9 -> 6.0)", r["sl_atr_multiple"] == 1.0 and r["tp_atr_multiple"] == 6.0
      and r["stops_source"] == "AI_ATR_ADJUSTED", str(r["stop_notes"]))

r = decide({**base, "sl_atr_multiple": 2.5, "tp_atr_multiple": 2.0})
check("TP raised to keep R:R >= 1.5", r["tp_atr_multiple"] == 3.75 and r["risk_reward"] == 1.5, str(r["tp_atr_multiple"]))

r = decide(base)
check("Missing multiples -> default 1.5x / 3.0x", r["sl_atr_multiple"] == 1.5 and r["tp_atr_multiple"] == 3.0
      and r["stops_source"] == "DEFAULT_ATR")

r = decide({**base, "stop_loss": 1.09810, "take_profit": 1.10410})
check("Quoted prices converted to multiples (1.0x / 2.0x)", r["sl_atr_multiple"] == 1.0 and r["tp_atr_multiple"] == 2.0
      and r["stop_loss"] == 1.09810, f"{r['sl_atr_multiple']} / {r['tp_atr_multiple']}")

r = decide({**base, "stop_loss": 1.10500, "take_profit": 1.09000})
check("Wrong-side quoted prices ignored -> defaults", r["stops_source"] == "DEFAULT_ATR")

xau = {**market, "bid": 2650.12, "ask": 2650.37, "digits": 2, "point": 0.01, "tick_size": 0.05, "spread_price": 0.25,
       "h1_data": {"atr14": 4.337, "structure": {"low": 2640.0, "high": 2660.0}}}
r = decide({**base, "sl_atr_multiple": 1.5, "tp_atr_multiple": 3.0}, xau)
on_grid = lambda p: abs(round(p / 0.05) * 0.05 - p) < 1e-9
check("Prices snapped to 0.05 tick grid", on_grid(r["stop_loss"]) and on_grid(r["take_profit"]),
      f"{r['stop_loss']} / {r['take_profit']}")

wide = {**market, "spread_price": 0.0012}
r = decide({**base, "sl_atr_multiple": 1.0, "tp_atr_multiple": 2.0}, wide)
check("Spread > 25% of stop -> HOLD with [COSTS]", r["signal"] == "HOLD" and "[COSTS]" in r["logic"])

r = decide({**base, "confidence_score": 50, "base_confidence": 50, "sl_atr_multiple": 2, "tp_atr_multiple": 4})
check("Threshold HOLD still shows hypothetical ATR stops", r["signal"] == "HOLD" and r["raw_signal"] == "BUY"
      and r["stop_loss"] == 1.09610)

r = decide({"signal": "HOLD", "confidence_score": 20, "logic": "flat"})
check("Pure HOLD has no stops", r["stop_loss"] is None and r["sl_atr_multiple"] is None)

noatr = {**market, "h1_data": {"structure": {}}, "daily_data": {}}
r = decide({**base, "sl_atr_multiple": 1.5, "tp_atr_multiple": 3}, noatr)
check("No ATR -> forced HOLD", r["signal"] == "HOLD" and "[STOPS]" in r["logic"])

prompt = ai_brain.build_system_prompt(market, "EURUSDm", [])
check("Prompt: ATR STOP ENGINE + bounds formatted", "ATR STOP ENGINE" in prompt and "between 1.0 and 3.0" in prompt
      and "between 1.5 and 6.0" in prompt and "{" + "sl_min}" not in prompt)
check("Prompt: ATR geometry with swing distance", "ATR STOP GEOMETRY (H1 ATR14 = 0.002000 = 200.0 points)" in prompt
      and "swing low 1.09700 is 1.55 ATR below" in prompt, prompt[prompt.find("ATR STOP GEOMETRY"):][:300])
check("Prompt: JSON asks for multiples, not prices", '"sl_atr_multiple"' in prompt and '"stop_loss": <float>' not in prompt)

# ---------------------------------------------------------------- execution re-anchoring
info = SimpleNamespace(digits=5, point=0.00001, trade_tick_size=0.00001, trade_tick_value=1.0, trade_tick_value_loss=1.0,
                       trade_contract_size=100000.0, volume_min=0.01, volume_max=100.0, volume_step=0.01,
                       trade_stops_level=0, filling_mode=2)
sent = []
mt5.symbol_info = lambda s: info
mt5.symbol_info_tick = lambda s: SimpleNamespace(bid=1.10100, ask=1.10110)  # price moved +10 pips since decision
mt5.account_info = lambda: SimpleNamespace(equity=10_000.0, margin_free=9_000.0)
mt5.order_calc_margin = lambda *a: 100.0
mt5.order_send = lambda req: (sent.append(dict(req)) or SimpleNamespace(
    retcode=mt5.TRADE_RETCODE_DONE, deal=1, order=2, price=req["price"], volume=req["volume"], comment="ok"))
mt5.history_deals_get = lambda **kw: (SimpleNamespace(position_id=2),)

d = decide({**base, "sl_atr_multiple": 1.5, "tp_atr_multiple": 3.0})
res = execution.execute_trade("EURUSDm", "BUY", d["stop_loss"], d["take_profit"], 1.0,
                              sl_distance=d["sl_distance"], tp_distance=d["tp_distance"])
check("Re-anchored SL = live ask - 1.5 x ATR", res["stop_loss"] == 1.09810, res["stop_loss"])
check("Re-anchored TP = live ask + 3.0 x ATR", res["take_profit"] == 1.10710, res["take_profit"])
check("Size uses the exact ATR stop (1% of 10k / 30 pips = 0.33)", res["volume"] == 0.33, res["volume"])

res = execution.execute_trade("EURUSDm", "BUY", d["stop_loss"], d["take_profit"], 1.0)
check("Without distances the given prices are used as-is", res["stop_loss"] == 1.09710)

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
