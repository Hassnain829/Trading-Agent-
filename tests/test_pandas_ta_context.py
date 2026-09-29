"""Verify the pandas_ta indicator engine and system-prompt context injection."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder

import numpy as np
import pandas as pd

import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

tmp = Path(tempfile.mkdtemp())
config.RISK_STATE_FILE, config.SHADOW_FILE, config.NEWS_CACHE_FILE = tmp / "risk_state.json", tmp / "shadow.json", tmp / "news.json"
config.MEMORY_FILE = tmp / "memory.json"
config.RULES_FILE = tmp / "new_rules.json"

import ai_brain
import data_engine

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


def make_frame(n, seed=7, start=0):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.001, n))
    frame = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "open": np.r_[close[0], close[:-1]],
        "high": close + rng.uniform(0.0002, 0.0015, n),
        "low": close - rng.uniform(0.0002, 0.0015, n),
        "close": close,
        "tick_volume": rng.integers(500, 3000, n).astype(float),
    })
    frame.index = range(start, start + n)
    return frame


# ---------------------------------------------------------------- indicators vs independent references
df = make_frame(250)
ind = data_engine.calculate_indicators(df, digits=5)
close, high, low, vol = (df[c].to_numpy() for c in ("close", "high", "low", "tick_volume"))

check("Engine is pandas_ta", ind["engine"].startswith("pandas_ta "), ind["engine"])

alpha = 2 / 201
ema = close[:200].mean()
for v in close[200:]:
    ema = alpha * v + (1 - alpha) * ema
check("EMA200 == SMA-seeded reference", abs(ind["ema200"] - round(ema, 5)) < 1e-9, f"{ind['ema200']} vs {ema:.5f}")

d = np.diff(close)
g, l = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
ag, al = g[0], l[0]
for i in range(1, len(d)):
    ag += (g[i] - ag) / 14
    al += (l[i] - al) / 14
rsi_ref = 100 - 100 / (1 + ag / al)
check("RSI14 == Wilder reference", abs(ind["rsi14"] - rsi_ref) < 0.01, f"{ind['rsi14']} vs {rsi_ref:.2f}")

prev = np.r_[np.nan, close[:-1]]
tr = np.nanmax(np.vstack([high - low, np.abs(high - prev), np.abs(low - prev)]), axis=0)
atr = tr[:14].mean()
for v in tr[14:]:
    atr += (v - atr) / 14
check("ATR14 == Wilder true-range reference", abs(ind["atr14"] - atr) < 1e-6, f"{ind['atr14']} vs {atr:.7f}")

relv = vol[-1] / vol[-20:].mean()
check("rel_volume == volume / SMA20", abs(ind["rel_volume"] - relv) < 1e-3, f"{ind['rel_volume']} vs {relv:.3f}")
check("volume_ma20 == SMA20", abs(ind["volume_ma20"] - vol[-20:].mean()) < 0.1)

traj = ind["trajectory"]
check("Trajectory has 5 bars per series", all(len(traj[k]) == 5 for k in ("rsi14", "rel_volume", "atr14")))
check("Trajectory ends at the latest value", traj["rsi14"][-1] == ind["rsi14"] and traj["rel_volume"][-1] == ind["rel_volume"])
check("Derived metrics present", all(ind[k] is not None for k in ("atr_ratio", "atr_pct", "ema_distance_atr", "ema_slope")))
check("Output is JSON-serializable", bool(json.dumps(ind)))

shifted = data_engine.calculate_indicators(make_frame(250, start=500), digits=5)
check("Non-zero-based index gives identical results", shifted["ema200"] == ind["ema200"] and shifted["atr14"] == ind["atr14"])

short = data_engine.calculate_indicators(make_frame(120), digits=5)
check("Short history: EMA200 None, RSI/ATR/rel_volume still valid",
      short["ema200"] is None and short["price_vs_ema"] is None and short["rsi14"] is not None
      and short["atr14"] is not None and short["rel_volume"] is not None, str(short["ema200"]))

flat = make_frame(250)
flat[["open", "high", "low", "close"]] = 1.0
fi = data_engine.calculate_indicators(flat, digits=5)
check("Flat market: no crash, RSI None, JSON-safe", fi["rsi14"] is None and bool(json.dumps(fi)))

try:
    data_engine.calculate_indicators(make_frame(10), digits=5)
    check("Too few bars rejected", False)
except data_engine.DataEngineError:
    check("Too few bars rejected", True)

# ---------------------------------------------------------------- system prompt injection
d1 = data_engine.calculate_indicators(make_frame(250, seed=11), digits=5)
market = {
    "symbol": "EURUSDm", "bid": 1.10000, "ask": 1.10010, "mid": 1.10005, "spread": 10, "spread_price": 0.0001,
    "digits": 5, "equity": 1000.0, "currency": "USD", "h1_data": ind, "daily_data": d1,
    "correlated_prices": {"USDJPYm": {"bid": 150.123, "day_change_pct": 0.412}},
}
config.RULES_FILE.write_text(json.dumps({"rules": [
    {"id": "R-ABC123", "affected_symbol": "ALL", "setup": "BUY when h1.rel_volume < 0.8", "side": "BUY",
     "conditions": [{"metric": "h1.rel_volume", "op": "<", "value": 0.8}],
     "confidence_reduction_points": 20, "sample_size": 3, "evidence": "3 losses", "status": "ACTIVE"},
    {"id": "R-GBP999", "affected_symbol": "GBPUSDm", "setup": "SELL when d1.rsi14 < 30", "side": "SELL",
     "conditions": [{"metric": "d1.rsi14", "op": "<", "value": 30}],
     "confidence_reduction_points": 25, "sample_size": 2, "evidence": "2 losses", "status": "ACTIVE"},
]}))

captured = {}


def fake_chat(messages, **kwargs):
    captured["messages"] = messages
    return {"content": json.dumps({"signal": "HOLD", "base_confidence": 40, "confidence_score": 40,
                                   "triggered_rule_ids": [], "stop_loss": 0, "take_profit": 0, "logic": "no edge"})}


config.DEEPSEEK_API_KEY = "test"
ai_brain.deepseek_chat = fake_chat
decision = ai_brain.get_ai_decision(market, "EURUSDm")
system, user = captured["messages"][0]["content"], captured["messages"][1]["content"]

check("Decision still produced", decision["signal"] == "HOLD" and decision["error"] is None)
check("System prompt starts with the static mandate", system.startswith("You are the Head of Systematic Trading"))
check("System prompt carries DEEP MARKET CONTEXT", "=== DEEP MARKET CONTEXT: EURUSDm ===" in system
      and "=== END DEEP MARKET CONTEXT ===" in system)
check("pandas_ta engine named in context", ind["engine"] in system)
for label, value in (("EMA200", f"{ind['ema200']:.5f}"), ("RSI14", f"{ind['rsi14']:.2f}"),
                     ("ATR14", f"{ind['atr14']:.6f}"), ("rel_volume", f"{ind['rel_volume']:.2f}"),
                     ("D1 EMA200", f"{d1['ema200']:.5f}")):
    check(f"{label} value injected into system prompt", value in system, value)
check("Trajectory injected", "trajectory, last 5 closed bars" in system)
check("Cross-asset injected", "USDJPYm: bid 150.12300 | day change 0.412%" in system)
check("Learned rule for ALL injected, other-symbol rule excluded", "R-ABC123" in system and "R-GBP999" not in system)
check("Threshold formatted", "Signals below 65" in system and "{threshold}" not in system)
check("User turn is a short request without market data", len(user) < 160 and "EMA200" not in user and "json" in user.lower(), user)
check("Context comes after the static prefix", system.index("OUTPUT") < system.index("=== DEEP MARKET CONTEXT"))

print("\n----- system prompt context section -----")
print(system[system.index("=== DEEP MARKET CONTEXT"):])
print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
