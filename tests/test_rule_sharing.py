"""Demo -> live: rule sharing by pair (any broker or same broker), symbol localisation, DEMO/LIVE trade separation."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

tmp = Path(tempfile.mkdtemp())
for attr, name in (("MEMORY_FILE", "memory.json"), ("RULES_FILE", "new_rules.json"), ("SETTINGS_FILE", "settings.json"),
                   ("AUDIT_LOCK_FILE", ".audit.lock"), ("LOCK_FILE", ".server.lock"),
                   ("SHADOW_FILE", "shadow.json"), ("RISK_STATE_FILE", "risk_state.json"), ("NEWS_CACHE_FILE", "news.json")):
    setattr(config, attr, tmp / name)
config.AGENT_DIR = config.SHADOW_FILE.parent / "agent"  # the learning agent's files stay in the temp folder too
config.EXPLORE_FILE = config.AGENT_DIR / "explore_shadows.json"
config.DEEPSEEK_API_KEY = "test"

import ai_brain
import auditor
import data_engine
import main
import memory_store

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


def use_account(mode, broker, server, symbols):
    config.ACTIVE_ACCOUNT = {"mode": mode, "broker": broker, "server": server}
    config.SYMBOLS = symbols


DEMO_MQ = ("DEMO", "MetaQuotes Ltd.", "MetaQuotes-Demo", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
LIVE_EXNESS = ("LIVE", "Exness Technologies Ltd", "Exness-MT5Real8", ["EURUSDm", "GBPUSDm", "USDJPYm", "XAUUSDm"])
DEMO_EXNESS = ("DEMO", "Exness Technologies Ltd", "Exness-MT5Trial7", ["EURUSDm", "GBPUSDm", "USDJPYm", "XAUUSDm"])

CONDITIONS = [{"metric": "cross.USDJPY.corr_h1", "op": "<=", "value": -0.5},
              {"metric": "cross.USDJPY.day_change_pct", "op": ">", "value": 0.3},
              {"metric": "h1.rel_volume", "op": "<", "value": 0.8}, {"metric": "h1.atr_ratio", "op": ">", "value": 1.3}]
rule = {"id": "R-DEMO01", "status": "ACTIVE", "affected_symbol": "EURUSD", "side": "BUY", "correlated_symbol": "USDJPY",
        "conditions": CONDITIONS, "setup": "BUY EURUSD when ...",
        "confidence_reduction_points": 25, "sample_size": 4, "evidence": "4 EURUSD longs lost while USDJPY rallied",
        "learned_modes": ["DEMO"], "learned_brokers": ["MetaQuotes Ltd."], "learned_servers": ["MetaQuotes-Demo"]}
legacy = {"id": "R-OLD001", "status": "ACTIVE", "affected_symbol": "GBPUSD", "setup": "SELL GBPUSD when h1.rel_volume < 0.7",
          "confidence_reduction_points": 15, "sample_size": 2, "evidence": "old"}
nobroker = dict(rule, id="R-NOBRK1", affected_symbol="GBPUSD", learned_brokers=[], learned_servers=[])
config.RULES_FILE.write_text(json.dumps({"rules": [rule, legacy, nobroker]}))

# ---------------------------------------------------------------- rule sharing
config.RULE_SCOPE = "PAIR"
use_account(*LIVE_EXNESS)
ids = [r["id"] for r in ai_brain.load_learned_rules("EURUSDm")]
check("PAIR: MetaQuotes-demo EURUSD rule applies to Exness live EURUSDm", ids == ["R-DEMO01"], ids)
check("PAIR: not applied to other pairs", ai_brain.load_learned_rules("XAUUSDm") == [])
market = {"symbol": "EURUSDm", "bid": 1.1, "ask": 1.1001, "mid": 1.10005, "spread": 10, "spread_price": 0.0001, "digits": 5,
          "point": 0.00001, "equity": 1000, "currency": "USD",
          "h1_data": {"atr14": 0.002, "rel_volume": 0.6, "atr_ratio": 1.5, "structure": {}},
          "daily_data": {"structure": {}}, "correlated_prices": {"USDJPYm": {"bid": 148.2, "day_change_pct": 0.41, "corr_h1": -0.7}}}
prompt = ai_brain.build_system_prompt(market, "EURUSDm", ai_brain.load_learned_rules("EURUSDm"))
rules_part = prompt[prompt.index("LEARNED RISK RULES (checked"):]
check("Rule shown to the AI in this broker's names", "BUY EURUSDm" in rules_part
      and "cross.USDJPYm.corr_h1 <= -0.5" in rules_part and "cross.USDJPY." not in rules_part, rules_part[:300])
check("Field names untouched by localisation", "h1.rel_volume < 0.8" in rules_part and "day_change_pct" in rules_part)
check("Engine tells the AI the rule matches a BUY right now (checked by code)", "matches now: BUY" in rules_part, rules_part[:300])
d = ai_brain._validate_decision({"signal": "BUY", "confidence_score": 80, "sl_atr_multiple": 1.5, "tp_atr_multiple": 3},
                                market, "EURUSDm", ai_brain.load_learned_rules("EURUSDm"), {})
check("Cross-broker rule applied by code on EURUSDm (80 -> 55 HOLD)", d["confidence_score"] == 55 and d["signal"] == "HOLD"
      and [r["id"] for r in d["applied_rules"]] == ["R-DEMO01"], (d["confidence_score"], d["applied_rules"]))

config.RULE_SCOPE = "BROKER"
check("BROKER: MetaQuotes rule NOT applied at Exness live", ai_brain.load_learned_rules("EURUSDm") == [])
check("BROKER: rule with no broker recorded still applies; free-text legacy rule never does",
      [r["id"] for r in ai_brain.load_learned_rules("GBPUSDm")] == ["R-NOBRK1"])
use_account(*DEMO_MQ)
check("BROKER: applies at the broker it was learned at", [r["id"] for r in ai_brain.load_learned_rules("EURUSD")] == ["R-DEMO01"])
exrule = dict(rule, id="R-EXN001", learned_brokers=["Exness Technologies Ltd"], learned_servers=["Exness-MT5Trial7"])
config.RULES_FILE.write_text(json.dumps({"rules": [rule, exrule, legacy, nobroker]}))
use_account(*LIVE_EXNESS)
check("BROKER: Exness demo rule applies on Exness live", [r["id"] for r in ai_brain.load_learned_rules("EURUSDm")] == ["R-EXN001"])
config.RULE_SCOPE = "PAIR"

main._refresh_learning_state()
views = {r["id"]: r for r in main.bot_state["learned_rules"]}
check("Dashboard: rules applying here flagged", views["R-DEMO01"]["applies_here"] and views["R-EXN001"]["applies_here"])
config.RULE_SCOPE = "BROKER"
main._refresh_learning_state()
views = {r["id"]: r for r in main.bot_state["learned_rules"]}
check("Dashboard: BROKER scope explains why a rule is off here", not views["R-DEMO01"]["applies_here"]
      and "another broker" in views["R-DEMO01"]["applies_note"], views["R-DEMO01"]["applies_note"])
config.RULE_SCOPE = "PAIR"

# ---------------------------------------------------------------- DEMO/LIVE tagging and separation
def trade(i, mode, broker, server, symbol, pnl, login):
    return {"id": f"t{i}", "status": "CLOSED", "timestamp": f"2026-09-{10 + i:02d}T10:00:00+00:00", "symbol": symbol,
            "side": "BUY", "ticket": 100 + i, "realized_pnl": pnl, "outcome": "WIN" if pnl > 0 else "LOSS",
            "reconciled_at": f"2026-09-{10 + i:02d}T12:00:00+00:00", "exit_time": f"2026-09-{10 + i:02d}T12:00:00+00:00",
            "account_login": login, "account_mode": mode, "broker": broker, "server": server,
            "market_context": {"correlated_prices": {"USDJPY" if mode == "DEMO" else "USDJPYm": {"day_change_pct": 0.4, "corr_h1": -0.7}},
                               "h1": {"rel_volume": 0.6 if pnl < 0 else 1.3, "atr_ratio": 1.5}}}


records = [trade(i, "DEMO", "MetaQuotes Ltd.", "MetaQuotes-Demo", "EURUSD", -30 if i % 2 else 40, 111) for i in range(6)]
records += [trade(10 + i, "LIVE", "Exness Technologies Ltd", "Exness-MT5Real8", "EURUSDm", -20 if i < 3 else 25, 222) for i in range(4)]
untagged = {"id": "old1", "status": "CLOSED", "timestamp": "2026-09-01T10:00:00+00:00", "symbol": "EURUSD", "side": "SELL",
            "ticket": 99, "realized_pnl": 10.0, "outcome": "WIN", "account_login": 111}
config.MEMORY_FILE.write_text(json.dumps([untagged] + records))
check("Earlier untagged trades get tagged for their login", memory_store.tag_untagged_trades(111, "DEMO", "MetaQuotes Ltd.", "MetaQuotes-Demo") == 1
      and memory_store.load_trade_memory()[0]["account_mode"] == "DEMO")
check("Tagging is idempotent", memory_store.tag_untagged_trades(111, "DEMO", "MetaQuotes Ltd.", "MetaQuotes-Demo") == 0)

main._history_cache["stamp"] = None
cache = main._history_and_stats()
by = cache["stats_by_mode"]
check("Stats split by account type", by["DEMO"]["closed"] == 7 and by["LIVE"]["closed"] == 4 and by["ALL"]["closed"] == 11,
      {k: v["closed"] for k, v in by.items()})
check("LIVE stats are live-only", by["LIVE"]["wins"] == 1 and by["LIVE"]["realized_pnl"] == -35.0, by["LIVE"])
check("Ledger rows carry the account type", {r["account_mode"] for r in cache["ledger"]} == {"DEMO", "LIVE"})
main.bot_state["account_mode"] = "LIVE"
data_engine.get_account_snapshot = lambda: None
main._refresh_account_state()
check("Headline stats follow the logged-in type", main.bot_state["stats_scope"] == "LIVE" and main.bot_state["stats"]["closed"] == 4)

# auditor learns from the logged-in type only and records where rules came from
config.RULES_FILE.write_text(json.dumps({"rules": []}))
seen = {}


def fake_chat(messages, **kw):
    brief = json.loads(messages[1]["content"].split("\n", 1)[1])
    seen["brief"] = brief
    loss_ids = [t["id"] for t in brief["losing_trades"]]
    return {"content": json.dumps({"rules": [{
        "affected_symbol": brief["losing_trades"][0]["symbol"], "side": "BUY", "conditions": CONDITIONS,
        "evidence": "x", "existing_rule_id": None}], "summary": "s"})}


auditor.deepseek_chat = fake_chat
use_account(*LIVE_EXNESS)
res = auditor.run_audit(force=True, reconcile=False)
check("LIVE audit analyses only live trades", seen["brief"]["window"]["account_type"] == "LIVE"
      and seen["brief"]["window"]["closed_trades"] == 4 and res["status"] == "completed", res.get("message"))
r1 = auditor.load_rules_document()["rules"][0]
check("New rule records where it was learned", r1["learned_modes"] == ["LIVE"] and r1["learned_brokers"] == ["Exness Technologies Ltd"])

use_account(*DEMO_MQ)
config.RULES_FILE.write_text(json.dumps({"rules": [dict(r1, setup=r1["setup"].replace("EURUSDm", "EURUSD"),
                                                         affected_symbol="EURUSD")]}))
res = auditor.run_audit(force=True, reconcile=False)
r2 = auditor.load_rules_document()["rules"]
check("DEMO audit analyses only demo trades", seen["brief"]["window"]["account_type"] == "DEMO"
      and seen["brief"]["window"]["closed_trades"] == 7)
check("Reconfirmed on another account: provenance merged, not duplicated", len(r2) == 1
      and r2[0]["learned_modes"] == ["DEMO", "LIVE"] and len(r2[0]["learned_brokers"]) == 2, r2[0].get("learned_modes"))

# settings
data_engine.get_account_snapshot = lambda: None
res = main.api_save_settings(main.SettingsUpdate(rule_scope="BROKER"))
check("Rule sharing saved from settings", config.RULE_SCOPE == "BROKER" and res["settings"]["rule_scope"] == "BROKER"
      and json.loads(config.SETTINGS_FILE.read_text())["rule_scope"] == "BROKER")

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
