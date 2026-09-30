"""Daily auditor: numeric rules verified by code, evidence-based penalties, schedule, locking, CLI,
expiry/retirement lifecycle, and decision-side enforcement without the model's cooperation."""
import json
import random
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

tmp = Path(tempfile.mkdtemp())
config.RISK_STATE_FILE, config.SHADOW_FILE, config.NEWS_CACHE_FILE = tmp / "risk_state.json", tmp / "shadow.json", tmp / "news.json"
config.AGENT_DIR = config.SHADOW_FILE.parent / "agent"  # the learning agent's files stay in the temp folder too
config.EXPLORE_FILE = config.AGENT_DIR / "explore_shadows.json"
config.TRADING_MODE = "DEMO"  # these checks place (stubbed) orders
config.MEMORY_FILE, config.RULES_FILE = tmp / "memory.json", tmp / "new_rules.json"
config.LOCK_FILE, config.AUDIT_LOCK_FILE = tmp / ".server.lock", tmp / ".audit.lock"
config.SHADOW_FILE = tmp / "shadow.json"
config.DEEPSEEK_API_KEY = "test-key"
config.SYMBOLS = ["EURUSDm", "GBPUSDm", "USDJPYm", "XAUUSDm", "BTCUSDm"]  # independent of the real .env
config.CONFIDENCE_THRESHOLD, config.CALIBRATE_THRESHOLD = 65, False

import ai_brain
import auditor

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


SYMS = ["EURUSDm", "GBPUSDm", "USDJPYm", "XAUUSDm", "BTCUSDm"]
random.seed(11)
start = datetime(2026, 9, 1, tzinfo=timezone.utc)


def trade(i, symbol, side, pnl, jpy_corr, jpy_day, relv, atr_ratio):
    ts = start + timedelta(hours=6 * i)
    cross = {s: {"bid": 1.0, "day_change_pct": round(random.uniform(-0.3, 0.3), 3),
                 "corr_h1": round(random.uniform(-0.3, 0.3), 3)} for s in SYMS if s != symbol}
    if "USDJPYm" in cross:
        cross["USDJPYm"].update(day_change_pct=jpy_day, corr_h1=jpy_corr)
    return {"id": f"rec{i:03d}", "status": "CLOSED", "timestamp": ts.isoformat(), "symbol": symbol, "side": side,
            "ticket": 1000 + i, "realized_pnl": pnl, "outcome": "WIN" if pnl > 0 else "LOSS",
            "exit_time": (ts + timedelta(hours=2)).isoformat(), "reconciled_at": (ts + timedelta(hours=2)).isoformat(),
            "exit_reason": "TAKE_PROFIT" if pnl > 0 else "STOP_LOSS", "holding_minutes": 90, "sl_atr_multiple": 1.5,
            "market_context": {
                "h1": {"rel_volume": relv, "atr_ratio": atr_ratio, "atr_pct": 0.15, "rsi14": 55},
                "d1": {"rel_volume": 1.0, "atr_ratio": 1.0}, "day_change_pct": 0.1,
                "correlated_prices": cross}}


records = []
planted = {5, 14, 23, 31, 40, 52}  # EURUSDm BUY losses into USDJPY rally, thin volume, expanding ATR
for i in range(60):
    if i in planted:
        records.append(trade(i, "EURUSDm", "BUY", -round(random.uniform(40, 90), 2), -0.72, 0.45, 0.62, 1.48))
    elif i % 3 == 0:
        records.append(trade(i, random.choice(SYMS[1:]), random.choice(["BUY", "SELL"]), -round(random.uniform(20, 60), 2),
                             0.1, 0.0, 1.1, 1.0))
    else:
        records.append(trade(i, random.choice(SYMS), random.choice(["BUY", "SELL"]), round(random.uniform(30, 120), 2),
                             -0.6 if i % 7 == 0 else 0.1, 0.1, 1.2, 0.95))
config.MEMORY_FILE.write_text(json.dumps(records))

REAL = [{"metric": "cross.USDJPYm.corr_h1", "op": "<=", "value": -0.5},
        {"metric": "cross.USDJPYm.day_change_pct", "op": ">", "value": 0.3},
        {"metric": "h1.rel_volume", "op": "<", "value": 0.8},
        {"metric": "h1.atr_ratio", "op": ">", "value": 1.3}]
calls = []


def fake_deepseek(messages, **kwargs):
    calls.append(json.loads(messages[1]["content"].split("\n", 1)[1]))
    rule = {"affected_symbol": "EURUSD", "side": "BUY", "conditions": REAL, "confidence_reduction_points": 45,
            "evidence": "EURUSD longs into USDJPY strength", "existing_rule_id": None}
    return {"content": json.dumps({"summary": "USDJPY-led USD strength sinks thin-volume EURUSD longs.", "rules": [
        rule,
        # metric the engine cannot check at decision time -> no volume condition left -> discarded
        {**rule, "conditions": [REAL[0], {"metric": "m1.rel_volume_5m", "op": ">", "value": 1.8}, REAL[3]]},
        # missing volume condition
        {**rule, "conditions": [REAL[0], REAL[3]]},
        # pattern as common among winners as losers (broad thresholds on GBPUSD)
        {"affected_symbol": "GBPUSDm", "side": "ANY", "evidence": "x", "conditions": [
            {"metric": "cross.BTCUSDm.corr_h1", "op": "<", "value": 0.5},
            {"metric": "h1.rel_volume", "op": "<", "value": 2.0}, {"metric": "h1.atr_ratio", "op": ">", "value": 0.5}]},
        # the model claims a pattern its own numbers do not support (nothing matches)
        {**rule, "conditions": [{"metric": "cross.USDJPYm.day_change_pct", "op": ">", "value": 5.0}, REAL[2], REAL[3]]},
        # "ALL" backed by one symbol -> narrowed to EURUSDm
        {**rule, "affected_symbol": "ALL", "conditions": [{"metric": "cross.USDJPYm.corr_h1", "op": "<=", "value": -0.6},
                                                          REAL[1], REAL[2], REAL[3]]},
    ]})}


auditor.deepseek_chat = fake_deepseek

# ---------------------------------------------------------------- first audit
res = auditor.run_audit(force=False, reconcile=False)
check("Audit completed", res["status"] == "completed", res.get("message"))
brief = calls[0]
check("Exactly the last 50 closed trades analysed", brief["window"]["closed_trades"] == 50
      and len(brief["losing_trades"]) + len(brief["winning_trades"]) == 50)
lt = brief["losing_trades"][0]
check("Trade brief uses the exact rule metric names", {"h1.rel_volume", "h1.atr_ratio", "cross.USDJPYm.corr_h1"}
      <= set(lt["metrics"]) or {"h1.rel_volume", "h1.atr_ratio"} <= set(lt["metrics"]), sorted(lt["metrics"])[:8])
check("Losses labelled L*, wins W*", lt["id"].startswith("L") and brief["winning_trades"][0]["id"].startswith("W"))
check("Prompt tells the model the engine verifies by code", "evaluates your conditions on EVERY trade"
      in auditor.auditor_system_prompt())
check("6 proposals -> 2 verified by code", res["rules_proposed"] == 6 and res["rules_added"] == 2, res["message"])

rules = auditor.load_rules_document()["rules"]
main_rule = next(r for r in rules if r["conditions"] == REAL)
check("Rule stores machine-checkable conditions", main_rule["correlated_symbol"] == "USDJPYm"
      and main_rule["pattern_type"] == "CROSS_ASSET" and len(main_rule["conditions"]) == 4)
check("Sample size counted by code (planted losses in window)", main_rule["sample_size"] == 5
      and main_rule["winning_matches"] == 0, (main_rule["sample_size"], main_rule["winning_matches"]))
check("Penalty set from evidence, not the model's 45 (5 losses, 100% -> 21)", main_rule["confidence_reduction_points"] == 21,
      main_rule["confidence_reduction_points"])
check("Setup text composed from the conditions", main_rule["setup"] == "BUY EURUSDm when cross.USDJPYm.corr_h1 <= -0.5 AND "
      "cross.USDJPYm.day_change_pct > 0.3 AND h1.rel_volume < 0.8 AND h1.atr_ratio > 1.3", main_rule["setup"])
check("Loss refs point at real memory records", set(main_rule["loss_trade_refs"]) <= {f"rec{i:03d}" for i in planted})
expires = datetime.fromisoformat(main_rule["expires_at"])
check("Rule expires in RULE_TTL_DAYS", timedelta(days=config.RULE_TTL_DAYS - 1) < expires - datetime.now(timezone.utc)
      <= timedelta(days=config.RULE_TTL_DAYS))
narrowed = [r for r in rules if r is not main_rule]
check("'ALL' rule backed by one symbol narrowed to EURUSDm", len(narrowed) == 1 and narrowed[0]["affected_symbol"] == "EURUSDm")

# ---------------------------------------------------------------- once-a-day schedule
due = auditor.audit_due()
check("Right after an audit: not due for 24h", not due["due"] and "next daily audit" in due["reason"], due["reason"])
check("Scheduled run inside the window -> not_due, no API call",
      auditor.run_audit(force=False, reconcile=False)["status"] == "not_due" and len(calls) == 1)
doc = auditor.load_rules_document()
doc["last_run_at"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
auditor.save_rules_document(doc)
check("25h later: due", auditor.audit_due()["due"])
r2 = auditor.run_audit(force=False, reconcile=False)
check("Due but no new closed trades -> no_new_trades without calling the model", r2["status"] == "no_new_trades" and len(calls) == 1)
check("...and that run restarts the 24h clock", not auditor.audit_due()["due"])

doc = auditor.load_rules_document()
doc["last_run_at"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
auditor.save_rules_document(doc)
new = trade(60, "EURUSDm", "BUY", -70.0, -0.7, 0.5, 0.6, 1.5)
new["reconciled_at"] = datetime.now(timezone.utc).isoformat()
records.append(new)
config.MEMORY_FILE.write_text(json.dumps(records))
r3 = auditor.run_audit(force=False, reconcile=False)
check("New closed trade + day elapsed -> audits again, reconfirms instead of duplicating",
      r3["status"] == "completed" and r3["rules_updated"] >= 1 and r3["rules_added"] == 0
      and len(auditor.load_rules_document()["rules"]) == 2, r3.get("message"))

# ---------------------------------------------------------------- locking + CLI
with auditor._process_lock() as got:
    busy = auditor.run_audit(force=True, reconcile=False)
check("Another process holding the audit lock -> busy", got and busy["status"] == "busy")
check("CLI inside the daily window exits 0 without auditing", auditor.main(["--no-mt5"]) == 0 and len(calls) == 2)
check("CLI --force runs an audit", auditor.main(["--force", "--no-mt5"]) == 0 and len(calls) == 3)

# ---------------------------------------------------------------- decision side: enforced by code
market = {"symbol": "EURUSDm", "bid": 1.1, "ask": 1.1001, "mid": 1.10005, "spread": 10, "spread_price": 0.0001,
          "digits": 5, "point": 0.00001, "equity": 1000, "currency": "USD",
          "h1_data": {"atr14": 0.002, "rel_volume": 0.6, "atr_ratio": 1.5, "structure": {}},
          "daily_data": {"structure": {}},
          "correlated_prices": {"USDJPYm": {"bid": 148.2, "day_change_pct": 0.41, "corr_h1": -0.71}}}
active = ai_brain.load_learned_rules("EURUSDm")
prompt = ai_brain.build_system_prompt(market, "EURUSDm", active)
check("Decision prompt shows corr_h1 of correlated assets", "USDJPYm: bid 148.20000 | day change 0.410% | corr_h1 -0.71" in prompt)
check("Decision prompt lists the rule and that it matches BUY now", main_rule["id"] in prompt and "matches now: BUY" in prompt)
check("GBPUSDm does not receive the EURUSDm rule", not ai_brain.load_learned_rules("GBPUSDm"))
answer = {"signal": "BUY", "confidence_score": 80, "sl_atr_multiple": 1.5, "tp_atr_multiple": 3.0, "logic": "x"}
d = ai_brain._validate_decision(dict(answer), market, "EURUSDm", active, {})
check("Model says nothing about rules: engine still applies both, capped at 30 (80 -> 50 HOLD)",
      d["confidence_score"] == 50 and d["signal"] == "HOLD" and len(d["applied_rules"]) == 2
      and sum(r["points"] for r in d["applied_rules"]) == config.RULE_TOTAL_PENALTY_CAP, d["applied_rules"])
check("Blocked by rules -> recorded as a shadow candidate", sorted(d["blocked_by"]) == sorted(r["id"] for r in active))
quiet = dict(market, h1_data=dict(market["h1_data"], rel_volume=1.4))
d = ai_brain._validate_decision(dict(answer), quiet, "EURUSDm", active, {})
check("Conditions not met (normal volume) -> no penalty, trade allowed", d["confidence_score"] == 80 and d["signal"] == "BUY")
d = ai_brain._validate_decision(dict(answer, signal="SELL"), market, "EURUSDm", active, {})
check("BUY-only rule does not touch a SELL", not [r for r in d["applied_rules"] if r.get("kind") == "rule"])

# ---------------------------------------------------------------- lifecycle
doc = auditor.load_rules_document()
doc["rules"].append({"id": "R-LEGACY", "status": "ACTIVE", "affected_symbol": "GBPUSDm", "setup": "free text"})
doc["rules"][0]["expires_at"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
auditor.save_rules_document(doc)
changed = {r["id"]: r["status"] for r in auditor.maintain_rules()}
check("Expired rule -> EXPIRED, free-text rule -> LEGACY", changed.get(doc["rules"][0]["id"]) == "EXPIRED"
      and changed.get("R-LEGACY") == "LEGACY", changed)
try:
    auditor.set_rule_status("R-LEGACY", "ACTIVE")
    check("LEGACY rule cannot be re-activated", False)
except ValueError:
    check("LEGACY rule cannot be re-activated", True)
auditor.set_rule_status(doc["rules"][0]["id"], "ACTIVE")
check("Operator re-activation of an expired rule restarts its lifetime",
      not ai_brain.rule_expired(auditor.load_rules_document()["rules"][0]))

rule_id = auditor.load_rules_document()["rules"][1]["id"]
shadows = [{"id": f"s{i}", "status": "WIN" if i < 4 else "LOSS", "r_multiple": 2.0 if i < 4 else -1.0,
            "blocked_by": [rule_id], "symbol": "EURUSDm", "side": "BUY"} for i in range(6)]
config.SHADOW_FILE.write_text(json.dumps(shadows))
changed = {r["id"]: r for r in auditor.maintain_rules()}
check("Rule whose blocked trades would have won (+1.0R avg over 6) is RETIRED",
      rule_id in changed and changed[rule_id]["status"] == "RETIRED" and "shadow" in changed[rule_id]["retired_reason"],
      {k: v.get("retired_reason") for k, v in changed.items()})

# evidence flips: the planted setups now win -> the audit retires the rule it can no longer support
flipped = [dict(r, realized_pnl=abs(r["realized_pnl"]), outcome="WIN") if r["id"] in {f"rec{i:03d}" for i in planted} | {"rec060"}
           else r for r in records]
config.MEMORY_FILE.write_text(json.dumps(flipped))
r4 = auditor.run_audit(force=True, reconcile=False)
first = auditor.load_rules_document()["rules"][0]
check("Latest trades no longer support the rule -> retired by the audit", first["status"] == "RETIRED"
      and "no longer support" in first.get("retired_reason", "") and r4["rules_retired"] >= 1, first.get("retired_reason"))

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
