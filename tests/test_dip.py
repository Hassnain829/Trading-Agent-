"""H1 dip strategy (dip.py): setups in both trend directions, the EMA5 exit, shadow resolution, the AI path, the agent."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config

tmp = Path(tempfile.mkdtemp())
for attr, name in (("MEMORY_FILE", "memory.json"), ("RULES_FILE", "new_rules.json"), ("SETTINGS_FILE", "settings.json"),
                   ("RISK_STATE_FILE", "risk_state.json"), ("SHADOW_FILE", "shadow.json"), ("NEWS_CACHE_FILE", "news.json"),
                   ("AUDIT_LOCK_FILE", ".audit.lock"), ("LOCK_FILE", ".server.lock"), ("JOURNAL_DIR", "journal")):
    setattr(config, attr, tmp / name)
config.AGENT_DIR = config.SHADOW_FILE.parent / "agent"
config.EXPLORE_FILE = config.AGENT_DIR / "explore_shadows.json"
config.DEEPSEEK_API_KEY = "test"
config.DIP_ENABLED, config.DIP_RSI_LOW, config.DIP_STOP_ATR, config.DIP_TIME_STOP_MINUTES = True, 5.0, 2.5, 2880
config.AGENT_CLUSTER_WEIGHTING, config.AGENT_HISTORY_WEIGHT = True, 0.3

import ai_brain
import dip
import main
import shadow_store
from ml import agent

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


END = 1_790_000_000 - 1_790_000_000 % 3600  # an H1 boundary (server epoch)


def frame(closes, start, step):
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({"time": start + step * np.arange(len(closes)), "open": opens,
                         "high": np.maximum(opens, closes) + 0.0002, "low": np.minimum(opens, closes) - 0.0002,
                         "close": closes, "tick_volume": 100, "spread": 10})


def frames(h1_closes, d1_slope=0.002):
    d1 = 0.90 + d1_slope * np.arange(300) if d1_slope >= 0 else 1.50 + d1_slope * np.arange(300)
    last = h1_closes[-1]
    return {"D1": frame(d1, END - 86400 * 301, 86400), "H1": frame(h1_closes, END - 3600 * len(h1_closes), 3600),
            "M15": frame(np.full(300, last), END - 900 * 300, 900), "M5": frame(np.full(300, last), END - 300 * 300, 300)}


# ============================================================ 1. setups
calm = [1.20 + 0.0003 * np.sin(k / 3) for k in range(297)]
dip_up = calm + [calm[-1] - 0.004, calm[-1] - 0.008, calm[-1] - 0.012]  # three sharp H1 drops: RSI(2) near 0
p = dip.prepare(frames(dip_up))
r = dip.evaluate(p, spread_price=0.0001)
s = r["setup"] or {}
check("D1 uptrend + H1 RSI(2) dip -> BUY setup with the EMA5 exit rule",
      r["stage"] == "SETUP" and s.get("side") == "BUY" and s.get("exit_rule") == dip.EXIT_RULE
      and s.get("strategy") == "DIP", r["reason"])
check("Stop = DIP_STOP_ATR x H1 ATR, entry at the ask, far safety target",
      abs(s["sl_distance"] - 2.5 * s["atr"]) < 1e-12 and abs(s["entry_ref"] - (dip_up[-1] + 0.0001)) < 1e-12
      and abs(s["tp_distance"] - config.DIP_TARGET_R * s["sl_distance"]) < 1e-12)
rally_down = [2.0 - c + 1.2 for c in dip_up]  # mirror: a sharp rally
r = dip.evaluate(dip.prepare(frames(rally_down, d1_slope=-0.002)))
check("D1 downtrend + H1 RSI(2) rally -> SELL setup", (r["setup"] or {}).get("side") == "SELL", r["reason"])
r = dip.evaluate(dip.prepare(frames(dip_up, d1_slope=-0.002)))
check("A dip against a D1 downtrend is not bought", r["setup"] is None and r["stage"] == "NO_DIP", r["reason"])
r = dip.evaluate(dip.prepare(frames(calm + [calm[-1] + 0.0005 * k for k in (1, 2, 3)])))
check("No RSI(2) extreme -> waiting (NO_DIP)", r["setup"] is None and r["stage"] == "NO_DIP", r["reason"])
f = dip.features(p, side="BUY")
check("Agent features come from the M5 bar that closed with the H1 bar", f.get("side") == "BUY" and "d1_adx" in f)

# ============================================================ 2. the EMA5 exit
times = END + 3600 * np.arange(6, dtype=float)
closes = np.array([1.1000, 1.1010, 1.1030, 1.1040, 1.1050, 1.1060])
ema5 = np.array([1.1020, 1.1020, 1.1025, 1.1030, 1.1040, 1.1050])
highs, lows = closes + 0.0005, closes - 0.0005
out = dip.exit_outcome(times, highs, lows, closes, ema5, END, "BUY", 1.1000, 1.0975, 1.1075, 0.0, 48 * 3600)
check("BUY: the first H1 close at/above EMA5 exits there (WIN, R = move / stop)",
      out is not None and out[0] == "WIN" and out[2] == 2 and abs(out[1] - 0.03 / 0.025) < 1e-3, out)
out = dip.exit_outcome(times, highs, lows, closes, ema5, END + 7, "BUY", 1.1000, 1.0975, 1.1075, 0.0, 48 * 3600)
check("Live entry seconds after the signal bar's close still uses the next bar", out is not None and out[2] == 2, out)
crash = lows.copy()
crash[1] = 1.0960
out = dip.exit_outcome(times, highs, crash, closes, ema5, END, "BUY", 1.1000, 1.0975, 1.1075, 0.0, 48 * 3600)
check("Stop touched before the EMA5 close -> LOSS -1R", out is not None and out[:2] == ("LOSS", -1.0), out)
flat = np.full(6, 1.0990)
out = dip.exit_outcome(times, flat + 0.0003, flat - 0.0003, flat, np.full(6, 1.1100), END, "BUY", 1.1000, 1.0975,
                       1.1075, 0.0, 3 * 3600)
check("Time stop: closed at the last bar inside the window (TIMEOUT)", out is not None and out[0] == "TIMEOUT"
      and out[2] == 2 and abs(out[1] - (-0.0010 / 0.0025)) < 1e-3, out)
out = dip.exit_outcome(times[:2], flat[:2] + 0.0003, flat[:2] - 0.0003, flat[:2], np.full(2, 1.1100), END, "BUY",
                       1.1000, 1.0975, 1.1075, 0.0, 48 * 3600)
check("Still running (no exit, time stop not reached) -> None", out is None)
out = dip.exit_outcome(times, 2.2 - lows, 2.2 - highs, 2.2 - closes, 2.2 - ema5, END, "SELL", 1.1000, 1.1025, 1.0925,
                       0.0001, 48 * 3600)
check("SELL mirror: an H1 close at/below EMA5 exits, paying the spread", out is not None and out[2] == 2
      and abs(out[1] - (1.1000 - (2.2 - 1.1030 + 0.0001)) / 0.0025) < 1e-3, out)

# ============================================================ 3. shadow trades
path = [1.1050] * 6 + [1.1000, 1.0995, 1.0990, 1.1060]  # the dip, two lower closes, then the snap-back
bars = np.zeros(len(path), dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8")])
bars["time"] = END - 3600 * len(path) + 3600 * np.arange(len(path))
bars["close"] = path
bars["high"], bars["low"] = bars["close"] + 0.0003, bars["close"] - 0.0003
shadow = {"server_epoch": int(bars["time"][6]) + 5, "side": "BUY", "entry_price": 1.1000, "stop_loss": 1.0975,
          "take_profit": 1.1075, "spread_price": 0.0, "time_stop_minutes": 2880, "exit_rule": dip.EXIT_RULE}
outcome = shadow_store._rule_outcome(shadow, bars)
check("Shadow with the EMA5 rule is resolved on H1 bars (exits on the snap-back close)",
      outcome is not None and outcome[0] == "WIN" and abs(outcome[1] - 0.006 / 0.0025) < 1e-3, outcome)
market = {"ask": 1.1001, "bid": 1.1000, "digits": 5, "tick_size": 0.00001, "point": 0.00001, "spread_price": 0.0001,
          "tick_epoch": END + 5, "account_mode": "DEMO", "broker": "test"}
setup = {**dip.evaluate(p, spread_price=0.0001)["setup"]}
recorded = main._shadow_skipped_setup("EURUSD", market, {"features": {}}, setup, ["LIMIT"])
check("A skipped dip setup is followed with its exit rule and the 48h time stop",
      recorded is not None and recorded["exit_rule"] == dip.EXIT_RULE and recorded["time_stop_minutes"] == 2880
      and recorded["strategy"] == "DIP", recorded and {k: recorded.get(k) for k in ("exit_rule", "time_stop_minutes")})

# ============================================================ 4. the AI's review and the agent
ai_brain.rule_penalties = lambda rules, mkt, side: []
ai_brain.protective_guards = lambda mkt, symbol, side: []
d = {"signal": "HOLD", "raw_signal": "BUY", "confidence_score": 0, "base_confidence": 0, "logic": "", "blocked_by": [],
     "applied_rules": [], "strategy": "DIP", "error": None}
d = ai_brain._scalp_decision({"decision": "VETO", "confidence_score": 20, "logic": "news"}, market, "EURUSD", [], setup, d)
check("AI decision on a dip: H1 ATR, 48h time stop, exit rule carried, veto shadow-followed",
      d["atr_source"] == "H1 ATR14" and d["time_stop_minutes"] == 2880 and d["exit_rule"] == dip.EXIT_RULE
      and d["blocked_by"] == ["AI-VETO"] and d["strategy"] == "DIP")
check("Dip strategy is wired in: order tag, risk fallback, trade limit, its own learning agent",
      main.ORDER_TAGS["DIP"] == "-D" and "DIP" in main.RULE_STRATEGIES and main._strategy_risk("DIP") > 0
      and "DIP" in agent.STRATEGIES and "DIP" in main._rule_strategies())
ctx = agent.make_context("DIP", f, "BUY", setup)
rows = [{"id": f"h{i}", "source": "history", "action": "history", "strategy": "DIP", "symbol": "EURUSD", "side": "BUY",
         "opened_at": f"2026-01-{1 + i % 28:02d}T{i % 24:02d}:00:00+00:00",
         "resolved_at": f"2026-01-{1 + i % 28:02d}T{i % 24:02d}:30:00+00:00", "reward": 0.5 if i % 3 else -1.0,
         "context": ctx} for i in range(60)]
fitted = agent.fit("DIP", rows)
check("The DIP agent learns from history rows (counted apart, not as real setups)",
      fitted["history_rewards"] == 60 and fitted["real_rewards"] == 0 and fitted.get("mu"), fitted["rewards"])

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
