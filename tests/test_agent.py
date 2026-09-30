"""The learning agent: state, rewards from trades/shadows/explorations, Thompson-sampling decisions, the scorecard."""
import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config

tmp = Path(tempfile.mkdtemp())
for attr, name in (("MEMORY_FILE", "memory.json"), ("RULES_FILE", "new_rules.json"), ("SETTINGS_FILE", "settings.json"),
                   ("RISK_STATE_FILE", "risk_state.json"), ("SHADOW_FILE", "shadow.json"), ("NEWS_CACHE_FILE", "news.json"),
                   ("AUDIT_LOCK_FILE", ".audit.lock"), ("LOCK_FILE", ".server.lock"), ("JOURNAL_DIR", "journal")):
    setattr(config, attr, tmp / name)
config.AGENT_DIR = config.SHADOW_FILE.parent / "agent"  # the learning agent's files stay in the temp folder too
config.EXPLORE_FILE = config.AGENT_DIR / "explore_shadows.json"
config.SCALP_RSI_PULLBACK = 40.0  # the synthetic charts are built for RSI 40, whatever the user's .env says
config.DEEPSEEK_API_KEY = "test"
config.AGENT_ENABLED, config.AGENT_EXPLORE, config.AGENT_MIN_REWARDS, config.AGENT_HALF_LIFE_DAYS = True, True, 30, 30.0
config.AGENT_SHADOW_UNTIL_LEARNED = False  # switched on for its own checks below
config.MAX_OPEN_POSITIONS = config.LOSS_COOLDOWN_MINUTES = 0
config.MAX_TOTAL_DRAWDOWN_PERCENT = config.DRAWDOWN_THROTTLE_PERCENT = 0.0
config.CONFIDENCE_THRESHOLD, config.CALIBRATE_THRESHOLD = 65, False

import MetaTrader5 as mt5

import ai_brain
import data_engine
import execution
import journal
import main
import scalper
import shadow_store
from ml import agent

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


NOW = datetime.now(timezone.utc)
iso = lambda dt: dt.isoformat(timespec="seconds")
rng = np.random.default_rng(7)


def features(rsi=45.0, room=2.0):
    return {"m5_rsi": rsi, "m5_room_atr": room, "d1_adx": 25.0, "spread_atr": 0.05, "server_hour": 14.0,
            "feature_version": scalper.FEATURE_VERSION, "side": "BUY"}


def ai(confirmed, confidence=75):
    return {"blocked_by": [] if confirmed else ["AI-VETO"], "base_confidence": confidence, "applied_rules": []}


# ============================================================ 1. state
ctx = agent.make_context("SCALP", features(), "BUY", {"sl_atr": 1.4, "room_r": 2.0}, ai(True, 80))
x = agent.vector("SCALP", ctx)
k = {name: i for i, name in enumerate(agent.keys("SCALP"))}
check("State vector: market features, AI answer (asked, confirmed, score), side and setup geometry",
      len(x) == len(agent.keys("SCALP")) and x[k["ai_asked"]] == 1 and x[k["ai_confirmed"]] == 1
      and abs(x[k["ai_score"]] - 0.6) < 1e-9 and x[k["is_buy"]] == 1 and x[k["sl_atr"]] == 1.4 and x[k["m5_rsi"]] == 45.0)
veto = agent.vector("SCALP", agent.make_context("SCALP", features(), "SELL", None, ai(False, 30)))
check("A veto is asked-but-not-confirmed; unknown values stay NaN (standardised to the average later)",
      veto[k["ai_asked"]] == 1 and veto[k["ai_confirmed"]] == 0 and veto[k["is_buy"]] == 0 and np.isnan(veto[k["sl_atr"]]))
failed_ai = agent.vector("SCALP", agent.make_context("SCALP", features(), "BUY", None, {"error": "timeout"}))
check("No AI answer (error): ai_asked = 0", failed_ai[k["ai_asked"]] == 0 and failed_ai[k["ai_score"]] == 0)
check("Intraday state adds the level type and bars since the break",
      agent.keys("INTRADAY")[-2:] == ("level_asia", "bars_since_break")
      and agent.vector("INTRADAY", agent.make_context("INTRADAY", features(), "BUY",
                                                       {"level_name": "ASIA_HIGH", "bars_since_break": 3}))[-2:].tolist() == [1.0, 3.0])

# ============================================================ 2. warm-up: the AI decides, the agent learns
verdict = agent.decide("SCALP", ctx)
check("No rewards yet -> warm-up: take is None (the AI decides)", verdict["take"] is None and verdict["phase"] == "warmup"
      and "0/30 rewards" in verdict["note"], verdict)


# ============================================================ 3. learning from rewards
def row(i, confirmed, reward, days_ago=1.0, strategy="SCALP", action="taken"):
    when = NOW - timedelta(days=days_ago, minutes=i)
    return {"id": f"t{strategy}{i}", "source": "trade", "strategy": strategy, "symbol": "EURUSD", "side": "BUY",
            "opened_at": iso(when - timedelta(hours=1)), "resolved_at": iso(when), "reward": reward, "action": action,
            "decided_by": "ai", "context": agent.make_context(strategy, features(45 + rng.normal(0, 3)), "BUY",
                                                             {"sl_atr": 1.5}, ai(confirmed, 80 if confirmed else 30))}


# the market pays AI-confirmed setups (+1.5R, 70% of the time) and punishes vetoed ones (-1R, 75%)
rows = []
for i in range(240):
    confirmed = i % 2 == 0
    win = rng.random() < (0.7 if confirmed else 0.25)
    rows.append(row(i, confirmed, 1.5 if win else -1.0, days_ago=rng.uniform(0, 20),
                    action="taken" if confirmed else "skipped"))
config.AGENT_DIR.mkdir(parents=True, exist_ok=True)
(config.AGENT_DIR / "experience.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
agent._cache.clear()
good = agent.make_context("SCALP", features(), "BUY", {"sl_atr": 1.5}, ai(True, 80))
bad = agent.make_context("SCALP", features(), "BUY", {"sl_atr": 1.5}, ai(False, 30))
pg, pb = agent.predict("SCALP", good), agent.predict("SCALP", bad)
check("Rewards learned: a confirmed setup expects a profit, a vetoed one a loss", pg["expected_r"] > 0.3 > -0.3 > pb["expected_r"]
      and pg["p_positive"] > 0.9 and pb["p_positive"] < 0.1, (pg, pb))
takes_good = sum(agent.decide("SCALP", good, np.random.default_rng(s))["take"] for s in range(200))
takes_bad = sum(agent.decide("SCALP", bad, np.random.default_rng(s))["take"] for s in range(200))
check("Thompson sampling: takes the paying setup almost always, the losing one almost never",
      takes_good >= 190 and takes_bad <= 10, (takes_good, takes_bad))
check("After warm-up the phase is 'learning' and the note explains the decision",
      agent.decide("SCALP", good)["phase"] == "learning" and "expects +" in agent.decide("SCALP", good)["note"])
status = agent.status("SCALP")
check("Status: rewards, phase and the features that matter most (the AI's answer here)",
      status["rewards"] == 240 and status["phase"] == "learning"
      and {e["feature"] for e in status["effects"][:3]} & {"ai_confirmed", "ai_score"}, status["effects"][:3])

# recency: the market changed - confirmed setups paid a year ago, lose now
old = [row(i, True, 1.5, days_ago=300 + i / 10) for i in range(120)]
new = [row(1000 + i, True, -1.0 if i % 4 else 1.5, days_ago=i / 10) for i in range(60)]
model = agent.fit("SCALP", old + new)
check("Recency weights (half-life 30 days): last month's losses outweigh last year's wins",
      model["mean_reward"] < 0 and agent.predict("SCALP", good, model)["expected_r"] < 0, model["mean_reward"])
explore_only = [{**r, "action": "explore"} for r in rows]
check("Learned only with real setups too: 240 exploration rewards alone are not enough (needs 10 of 30 real)",
      not agent.learned(agent.fit("SCALP", explore_only)) and agent.learned(agent.fit("SCALP", rows))
      and agent.min_real_rewards() == 10)
check("Strategies learn separately: the intraday agent has seen none of the scalp rewards",
      agent.fit("INTRADAY", rows)["rewards"] == 0 and agent.predict("INTRADAY", good) is None)
config.AGENT_ENABLED = False
check("Agent switched off -> the AI decides (take None)", agent.decide("SCALP", good)["take"] is None)
config.AGENT_ENABLED = True

# ============================================================ 4. scorecard
card = agent.scorecard("SCALP", 28)
g = card["groups"]
check("Scorecard: taken vs skipped vs every setup, and the edge of the picks",
      g["taken"]["n"] == 120 and g["skipped"]["n"] == 120 and g["all"]["n"] == 240
      and g["picks"]["avg_r"] > g["all"]["avg_r"] > g["skipped"]["avg_r"] and card["edge_r"] > 0, card["edge_r"])
check("Scorecard: AI-confirmed vs AI-vetoed results, and a cumulative curve per day",
      g["ai_confirmed"]["avg_r"] > 0 > g["ai_vetoed"]["avg_r"] and card["curve"]
      and card["curve"][-1]["all"] == round(sum(r["reward"] for r in rows), 2), card["curve"][-1])
check("Scorecard window: 7 days holds fewer rewards than 28", agent.scorecard("SCALP", 7)["groups"]["all"]["n"] < 240)

# ============================================================ 5. sync: rewards from trades, shadows and explorations
(config.AGENT_DIR / "experience.jsonl").unlink()
agent._cache.clear()
trade_ctx = agent.make_context("SCALP", features(), "BUY", {"sl_atr": 1.2}, ai(True))
config.MEMORY_FILE.write_text(json.dumps([
    {"id": "m1", "ticket": 501, "status": "CLOSED", "symbol": "EURUSD", "side": "BUY", "entry_price": 1.1000,
     "stop_loss": 1.0990, "exit_price": 1.1015, "timestamp": iso(NOW - timedelta(hours=2)),
     "exit_time_utc": iso(NOW - timedelta(hours=1)), "outcome": "WIN", "exit_reason": "TP",
     "agent": {"context": trade_ctx, "decided_by": "agent", "action": "taken"}},
    {"id": "m2", "ticket": 502, "status": "CONFIRMED", "symbol": "EURUSD", "side": "BUY", "entry_price": 1.1,
     "stop_loss": 1.099, "agent": {"context": trade_ctx}},  # still open: no reward yet
    {"id": "m3", "ticket": 503, "status": "CLOSED", "symbol": "EURUSD", "side": "SELL", "entry_price": 1.1,
     "stop_loss": 1.101, "exit_price": 1.1005},  # an old trade without the agent's state: not usable
]))
shadow_ctx = agent.make_context("INTRADAY", features(), "SELL", {"level_name": "PDL"}, ai(False, 20))
config.SHADOW_FILE.write_text(json.dumps([
    {"id": "sh1", "status": "LOSS", "r_multiple": -1.0, "symbol": "GBPUSD", "side": "SELL", "strategy": "INTRADAY",
     "created_at": iso(NOW - timedelta(hours=3)), "resolved_at": iso(NOW - timedelta(hours=2)), "blocked_by": ["AI-VETO", "AGENT-SKIP"],
     "agent": {"context": shadow_ctx, "decided_by": "agent", "action": "skipped"}},
    {"id": "sh2", "status": "OPEN", "symbol": "GBPUSD", "side": "SELL", "strategy": "INTRADAY", "agent": {"context": shadow_ctx}},
    {"id": "sh3", "status": "WIN", "r_multiple": 1.5, "symbol": "USDJPY", "side": "BUY", "strategy": "SCALP",
     "created_at": iso(NOW - timedelta(hours=5)), "resolved_at": iso(NOW - timedelta(hours=4)), "blocked_by": ["AI-VETO"],
     "base_confidence": 40, "features": features()},  # recorded before the agent existed: rebuilt from its fields
]))
config.EXPLORE_FILE.write_text(json.dumps([
    {"id": "ex1", "status": "TIMEOUT", "r_multiple": 0.3, "symbol": "EURUSD", "side": "BUY", "strategy": "SCALP",
     "created_at": iso(NOW - timedelta(hours=2)), "resolved_at": iso(NOW - timedelta(hours=1)), "blocked_by": ["EXPLORE"],
     "agent": {"context": agent.make_context("SCALP", features(), "BUY", None, explore=True), "action": "explore",
               "decided_by": "explore"}},
]))
added = agent.sync()
exp = {r["id"]: r for r in agent.load_experience()}
check("sync: a closed trade, a resolved shadow, a pre-agent shadow and an exploration -> 4 rewards",
      added == 4 and set(exp) == {"trade:501", "shadow:sh1", "shadow:sh3", "explore:ex1"}, sorted(exp))
check("Trade reward from its prices: +15 pips on a 10-pip stop = +1.5R, action taken",
      abs(exp["trade:501"]["reward"] - 1.5) < 1e-9 and exp["trade:501"]["action"] == "taken"
      and exp["trade:501"]["decided_by"] == "agent")
check("Shadow reward keeps the decision (skipped by the agent) and strategy; the old shadow is rebuilt as an AI veto",
      exp["shadow:sh1"]["action"] == "skipped" and exp["shadow:sh1"]["strategy"] == "INTRADAY"
      and exp["shadow:sh3"]["context"]["ai"]["asked"] and not exp["shadow:sh3"]["context"]["ai"]["confirmed"])
check("Exploration reward tagged explore", exp["explore:ex1"]["action"] == "explore" and exp["explore:ex1"]["context"]["explore"])
check("sync is idempotent", agent.sync() == 0 and len(agent.load_experience()) == 4)

# ============================================================ 6. shadow store: exploration file, one fetch per symbol
config.SHADOW_FILE.write_text("[]")
config.EXPLORE_FILE.write_text("[]")
market = {"tick_epoch": 1_800_000_000, "spread_price": 0.0001, "account_mode": "DEMO", "broker": "X", "h1_data": {},
          "daily_data": {}}
base = {"raw_signal": "BUY", "stop_loss": 1.0990, "take_profit": 1.1015, "entry_reference": 1.1000, "risk_reward": 1.5,
        "strategy": "SCALP", "time_stop_minutes": 60}
explore_ctx = agent.make_context("SCALP", features(), "BUY", None, explore=True)
shadow_store.record_blocked("EURUSD", market, {**base, "blocked_by": ["EXPLORE"],
                                              "agent": {"context": explore_ctx, "action": "explore"}})
shadow_store.record_blocked("EURUSD", market, {**base, "blocked_by": ["AI-VETO"], "agent": {"context": ctx, "action": "skipped"}})
shadow_store.record_blocked("GBPUSD", market, {**base, "blocked_by": ["EXPLORE"],
                                              "agent": {"context": explore_ctx, "action": "explore"}})
explore_rows, real_rows = shadow_store.load_shadows(config.EXPLORE_FILE), shadow_store.load_shadows()
check("Exploration trades go to their own file; a real skipped setup to the shadow file (no cross-dedup)",
      len(explore_rows) == 2 and len(real_rows) == 1 and real_rows[0]["blocked_by"] == ["AI-VETO"]
      and explore_rows[0]["agent"]["action"] == "explore")
check("Exploration trades never feed the auditor's shadow statistics", "EXPLORE" not in shadow_store.stats_by_blocker())
fetches = []
bars = np.array([(1_800_000_000 + 60 * k, 1.1, 1.1020, 1.0999, 1.1018) for k in range(1, 10)],
                dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8")])
for s in explore_rows:
    s["created_at"] = iso(NOW - timedelta(minutes=10))
config.EXPLORE_FILE.write_text(json.dumps(explore_rows))
shadow_store.mt5.copy_rates_from_pos = lambda symbol, tf, start, count: fetches.append(symbol) or bars
resolved = shadow_store.resolve_open_shadows(config.EXPLORE_FILE)
check("Exploration trades resolve like shadows (TP hit -> WIN +1.5R), one M1 fetch per symbol",
      resolved == 2 and sorted(fetches) == ["EURUSD", "GBPUSD"]
      and all(s["status"] == "WIN" and s["r_multiple"] == 1.5 for s in shadow_store.load_shadows(config.EXPLORE_FILE)),
      fetches)

# ============================================================ 7. the agent's decision applied to the AI's answer
def ai_decision(verdict_word, confidence, spread=0.0001):
    setup = {"side": "BUY", "sl_distance": 0.0010, "tp_distance": 0.0015, "sl_atr": 1.2, "risk_reward": 1.5, "atr": 0.0008,
             "reason": "BUY pullback scalp", "trend": "UP", "bar_time": 1}
    mkt = {"digits": 5, "ask": 1.1001, "bid": 1.1000, "spread_price": spread, "tick_size": 0.00001}
    d = {"signal": "HOLD", "raw_signal": "BUY", "confidence_score": 0, "base_confidence": 0, "logic": "", "blocked_by": [],
         "applied_rules": [], "strategy": "SCALP", "error": None}
    ai_brain.rule_penalties = lambda rules, market, side: []
    ai_brain.protective_guards = lambda market, symbol, side: []
    return ai_brain._scalp_decision({"decision": verdict_word, "confidence_score": confidence, "logic": "x"}, mkt,
                                    "EURUSD", [], setup, d)


d = main._apply_agent(ai_decision("VETO", 30), {"take": True, "phase": "learning", "note": "expects +0.30R"}, ctx)
check("Agent TAKE overrules an AI veto: order side set, veto recorded as overruled, stage AGENT_TAKE",
      d["signal"] == "BUY" and d["blocked_by"] == [] and d["overruled"] == ["AI-VETO"]
      and d["agent"]["decided_by"] == "agent" and d["agent"]["action"] == "taken" and main._decision_stage(d) == "AGENT_TAKE"
      and d["logic"].startswith("[AGENT TAKE, overrules the AI's veto"), d["logic"][:80])
d = main._apply_agent(ai_decision("CONFIRM", 85), {"take": False, "phase": "learning", "note": "expects -0.20R"}, ctx)
check("Agent SKIP of an AI-confirmed setup: HOLD, AGENT-SKIP (so it is shadow-followed), stage AGENT_SKIP",
      d["signal"] == "HOLD" and d["blocked_by"] == ["AGENT-SKIP"] and d["agent"]["action"] == "skipped"
      and main._decision_stage(d) == "AGENT_SKIP")
d = main._apply_agent(ai_decision("CONFIRM", 85, spread=0.0005), {"take": True, "phase": "learning", "note": ""}, ctx)
check("Hard limit: spread too wide for the stop -> never taken, not even by the agent",
      d["signal"] == "HOLD" and d["agent"]["decided_by"] == "costs" and d["cost_block"])
d = main._apply_agent(ai_decision("CONFIRM", 85), {"take": None, "phase": "warmup", "note": ""}, ctx)
check("Warm-up: the AI's CONFIRM stands (decided by the AI)", d["signal"] == "BUY" and d["agent"]["decided_by"] == "ai"
      and main._decision_stage(d) == "AI_CONFIRMED")
d = main._apply_agent(ai_decision("CONFIRM", 50), {"take": None, "phase": "warmup", "note": ""}, ctx)
check("Warm-up: an AI 'yes' below the threshold is a HOLD that is still shadow-followed (THRESHOLD)",
      d["signal"] == "HOLD" and d["blocked_by"] == ["THRESHOLD"])
config.AGENT_SHADOW_UNTIL_LEARNED = True
d = main._apply_agent(ai_decision("CONFIRM", 85), {"take": None, "phase": "warmup", "note": "learning (3/30 rewards)"}, ctx)
check("Shadow-only while learning: an AI-confirmed setup becomes a shadow trade (LEARNING), not an order",
      d["signal"] == "HOLD" and d["blocked_by"] == ["LEARNING"] and d["agent"]["action"] == "blocked"
      and main._decision_stage(d) == "LEARNING" and d["logic"].startswith("[LEARNING"), d["blocked_by"])
d = main._apply_agent(ai_decision("CONFIRM", 85), {"take": True, "phase": "learning", "note": "expects +0.2R"}, ctx)
check("...and once learned the agent's TAKE is a real order", d["signal"] == "BUY" and not d["blocked_by"])
config.AGENT_ENABLED = False
d = main._apply_agent(ai_decision("CONFIRM", 85), {"take": None, "phase": "off", "note": ""}, ctx)
check("...agent switched off: the AI's CONFIRM trades as before", d["signal"] == "BUY")
config.AGENT_ENABLED = True
config.AGENT_SHADOW_UNTIL_LEARNED = False
config.DEEPSEEK_API_KEY = ""
no_ai = ai_brain.get_scalp_decision({"digits": 5, "ask": 1.1001, "bid": 1.1, "spread_price": 0.0001, "mid": 1.1},
                                    "EURUSD", {"side": "BUY", "sl_distance": 0.001, "tp_distance": 0.0015, "sl_atr": 1.2,
                                               "risk_reward": 1.5, "atr": 0.0008, "reason": "r", "trend": "UP",
                                               "bar_time": 1}, {})
config.DEEPSEEK_API_KEY = "test"
check("No AI answer: HOLD with exact levels and AI-ERROR, so the agent can still decide and the idea is followed",
      no_ai["signal"] == "HOLD" and no_ai["stop_loss"] and no_ai["blocked_by"] == ["AI-ERROR"] and no_ai["error"])

# ============================================================ 8. near-miss setups (exploration) in the scalper rules
def frame(closes, t0, step):
    closes = np.asarray(closes, float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({"time": (t0 + step * np.arange(len(closes))).astype("int64"), "open": opens,
                         "high": np.maximum(opens, closes) + 0.0001, "low": np.minimum(opens, closes) - 0.0001,
                         "close": closes, "tick_volume": 100, "spread": 10})


END = int(datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc).timestamp())
m5 = list(1.10 + np.cumsum([0.0006 if k % 2 == 0 else -0.0001 for k in range(300)]))
m5 = m5 + [m5[-1] - 0.0018 * k for k in range(1, 4)]
m5 = m5 + [m5[-1] + 0.0012]
against = {"M5": frame(m5, END - 300 * len(m5), 300), "M15": frame(1.05 - 0.0005 * np.arange(300), END - 900 * 300, 900),
           "H1": frame(1.00 + 0.0003 * np.arange(300), END - 3600 * 300, 3600),
           "D1": frame(0.90 + 0.002 * np.arange(300), END - 86400 * 301, 86400)}
config.SCALP_STRICT_GUARD, config.SCALP_MEDIUM_TREND = False, True
prepared = scalper.prepare(against)
strict, relaxed = scalper.evaluate(prepared, spread_price=0.0001), scalper.evaluate(prepared, spread_price=0.0001, relaxed=True)
check("Scalper near miss (M15 momentum still against the trend): no real setup, a relaxed one to explore",
      strict["setup"] is None and strict["stage"] == "M15_AGAINST" and relaxed["setup"] and relaxed["setup"]["side"] == "BUY",
      (strict["stage"], relaxed["stage"]))

# ============================================================ 9. the live flow with a learned agent
open_positions, executed = [], []
execution.get_open_positions = lambda symbol=None: [p for p in open_positions if symbol in (None, p["symbol"])]


def fake_execute(symbol, side, sl, tp, risk, comment=None, **kw):
    executed.append({"symbol": symbol, "side": side, "comment": comment})
    return {"side": side, "position_ticket": 9000 + len(executed), "deal": 1, "order": 1, "price": 1.1001, "stop_loss": sl,
            "take_profit": tp, "volume": 0.01, "digits": 5, "execution": {}, "equity_at_entry": 1000.0}


execution.execute_trade = fake_execute
main._capture_context = lambda symbol, market=None: None
main._refresh_account_state = lambda: None
main.bot_state.update(is_running=True, kill_switch=False, circuit_breaker=False, profit_target_hit=False,
                      risk_percent=1.0, risk_throttled=False)
setup = {"side": "BUY", "sl_distance": 0.0010, "tp_distance": 0.0015, "sl_atr": 1.2, "risk_reward": 1.5, "atr": 0.0008,
         "reason": "BUY pullback scalp: test", "trend": "UP", "bar_time": 1, "room_r": 2.0, "m5_rsi": 45.0,
         "m5_rsi_extreme": 35.0, "entry_ref": 1.1001, "m5_ema20": 1.1, "m5_ema50": 1.0995, "m15_close": 1.1,
         "m15_ema50": 1.099, "d1_close": 1.1, "d1_ema200": 1.05}
result = {"stage": "SETUP", "setup": setup, "reason": setup["reason"], "trend": "UP", "rsi": 45.0,
          "recent": {"closes": [1.1], "highs": [1.1], "lows": [1.1], "rsi14": [50]},
          "features": features(), "guards": []}
market = {"symbol": "EURUSD", "bid": 1.1000, "ask": 1.1001, "mid": 1.10005, "spread": 10, "spread_price": 0.0001,
          "digits": 5, "point": 0.00001, "tick_size": 0.00001, "equity": 1000.0, "balance": 1000.0, "account_mode": "DEMO",
          "broker": "X", "server": "Demo", "account_login": 5, "tick_epoch": int(time.time()), "day_change_pct": 0.1,
          "h1_data": {"atr14": 0.001}, "daily_data": {}}
main._evaluate_scalp = lambda symbol, spread: dict(result)
bar = {"t": 1000}
data_engine.last_closed_bar_time = lambda symbol, tf: bar["t"]
scalper.in_session = lambda now=None: True


async def fake_prepare(symbol):
    return dict(market)


main._prepare_market = fake_prepare
data_engine.fetch_correlated_asset_prices = lambda *a, **k: {}
answer = {"decision": "VETO", "confidence_score": 30, "logic": "target runs into resistance"}
ai_brain.deepseek_chat = lambda messages, **kw: {"content": json.dumps(answer)}
config.SCALP_ENABLED, config.INTRADAY_ENABLED = True, False
agent.decide = lambda strategy, ctx, rng=None: {"take": True, "phase": "learning", "expected_r": 0.25, "p_positive": 0.8,
                                                "rewards": 120, "note": "TAKE: expects +0.25R"}
asyncio.run(main.process_symbol("EURUSD"))
line = list(journal.iter_entries(1))[-1]
record = json.loads(config.MEMORY_FILE.read_text())[-1]
check("Learned agent takes an AI-vetoed setup on demo: order sent, journal shows the AI veto it overruled",
      executed and executed[-1]["comment"].endswith("-S") and line["action"] == "filled"
      and line["ai"]["overruled"] == ["AI-VETO"] and line["agent"]["decided_by"] == "agent"
      and main.bot_state["decisions"]["EURUSD"]["stage"] == "AGENT_TAKE", line.get("action"))
check("The trade record keeps the agent's state (action taken) so its reward is learned when it closes",
      record["agent"]["action"] == "taken" and record["agent"]["context"]["ai"]["asked"]
      and record["agent"]["context"]["strategy"] == "SCALP")

bar["t"] += 300
config.SHADOW_FILE.write_text("[]")
answer.update(decision="CONFIRM", confidence_score=85)
agent.decide = lambda strategy, ctx, rng=None: {"take": False, "phase": "learning", "expected_r": -0.2, "p_positive": 0.2,
                                                "rewards": 120, "note": "SKIP: expects -0.20R"}
n = len(executed)
asyncio.run(main.process_symbol("EURUSD"))
line = list(journal.iter_entries(1))[-1]
shadow = shadow_store.load_shadows()[-1]
check("Learned agent skips an AI-confirmed setup: no order, followed as a shadow (AGENT-SKIP, action skipped)",
      len(executed) == n and line["action"] == "hold" and shadow["blocked_by"] == ["AGENT-SKIP"]
      and shadow["agent"]["action"] == "skipped" and line.get("shadow_id") == shadow["id"], shadow.get("blocked_by"))

bar["t"] += 300
config.SHADOW_FILE.write_text("[]")
config.AGENT_SHADOW_UNTIL_LEARNED = True
agent.decide = lambda strategy, ctx, rng=None: {"take": None, "phase": "warmup", "rewards": 12, "note": "learning (12/30 rewards)"}
n = len(executed)
asyncio.run(main.process_symbol("EURUSD"))
line = list(journal.iter_entries(1))[-1]
shadow = shadow_store.load_shadows()[-1]
check("Live: while learning, an AI-confirmed scalp is NOT ordered; it is followed as a LEARNING shadow trade",
      len(executed) == n and shadow["blocked_by"] == ["LEARNING"] and shadow["agent"]["action"] == "blocked"
      and main.bot_state["decisions"]["EURUSD"]["stage"] == "LEARNING" and line.get("shadow_id") == shadow["id"])
config.AGENT_SHADOW_UNTIL_LEARNED = False

bar["t"] += 300
near_setup = {**setup, "strategy": "SCALP"}
main._evaluate_scalp = lambda symbol, spread: {**result, "stage": "M15_AGAINST", "setup": None, "reason": "M15 against",
                                               "explore": {"setup": near_setup, "features": features()}}
calls = []
ai_brain.deepseek_chat = lambda messages, **kw: calls.append(1) or {"content": json.dumps(answer)}
before_real = len(shadow_store.load_shadows())
config.EXPLORE_FILE.write_text("[]")
out = asyncio.run(main.process_symbol("EURUSD"))
line = list(journal.iter_entries(1))[-1]
explored = shadow_store.load_shadows(config.EXPLORE_FILE)
check("No real setup, but a near miss -> a virtual exploration trade: no AI call, no order, own file",
      out == "no_setup" and not calls and len(executed) == n and len(explored) == 1
      and explored[0]["blocked_by"] == ["EXPLORE"] and explored[0]["agent"]["context"]["explore"]
      and len(shadow_store.load_shadows()) == before_real and line.get("explore_id") == explored[0]["id"])

# ============================================================ 10. dashboard + settings
payload = main._status_payload()
check("Status payload: the agent per strategy with its 7-day scorecard, no ML/backtest leftovers",
      set(payload["agent"]["strategies"]) == {"SCALP", "INTRADAY"} and "week" in payload["agent"]["strategies"]["SCALP"]
      and "ml" not in payload and "backtest" not in payload and "agent_enabled" in payload["guardrails"])
card = main.api_scorecard(days=0)
check("GET /api/scorecard: every strategy's rewards (days=0 = all)", card["days"] is None
      and set(card["strategies"]) == {"SCALP", "INTRADAY"} and "groups" in card["strategies"]["SCALP"])
data_engine.get_account_snapshot = lambda: None
res = main.api_save_settings(main.SettingsUpdate(agent_enabled=False, agent_explore=False))
check("Agent and exploration switchable from Settings", config.AGENT_ENABLED is False and config.AGENT_EXPLORE is False
      and res["settings"]["agent_enabled"] is False)

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
