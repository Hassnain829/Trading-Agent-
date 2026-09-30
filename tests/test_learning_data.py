"""Learning data: shadow trades in the auditor and calibration, side-signed features, the decision journal."""
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
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
config.SCALP_RSI_PULLBACK = 40.0  # the synthetic charts are built for RSI 40, whatever the user's .env says
config.DEEPSEEK_API_KEY = "test"

import auditor
import calibration
import journal
import scalper
import shadow_store

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ============================================================ synthetic 1-year uptrend (for the features)
def frame(closes, t0, step, spread=3):
    closes = np.asarray(closes, float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({"time": (t0 + step * np.arange(len(closes))).astype("int64"), "open": opens,
                         "high": np.maximum(opens, closes) + 0.0001, "low": np.minimum(opens, closes) - 0.0001,
                         "close": closes, "tick_volume": 100, "spread": spread})


END = int(pd.Timestamp("2026-07-01 13:00").value // 10**9)  # 13:00 server time = 10:00 UTC, London session
steps = [0.0006 if k % 2 == 0 else -0.0001 for k in range(300)]
m5_close = list(1.10 + np.cumsum(steps))
m5_close += [m5_close[-1] - 0.0018 * k for k in range(1, 4)]
m5_close += [m5_close[-1] + 0.0012] + [m5_close[-1] + 0.0012 + 0.0008 * k for k in range(1, 15)]
synthetic = {"M5": frame(m5_close, END - 300 * len(m5_close), 300), "M15": frame(1.05 + 0.0005 * np.arange(300), END - 900 * 300, 900),
             "H1": frame(1.00 + 0.0003 * np.arange(300), END - 3600 * 300, 3600),
             "D1": frame(0.90 + 0.002 * np.arange(300), END - 86400 * 301, 86400)}

# ============================================================ 1. shadow trades feed the learning loops
config.SHADOW_WEIGHT = 0.5
now = datetime.now(timezone.utc)
shadows = []
for k in range(8):
    shadows.append({"id": f"s{k}", "status": "LOSS" if k < 6 else "WIN", "r_multiple": -1.0 if k < 6 else 1.5,
                    "symbol": "EURUSD", "side": "BUY", "blocked_by": ["AI-VETO"], "confidence_score": 70,
                    "base_confidence": 70, "account_mode": "DEMO", "created_at": (now - timedelta(hours=k + 2)).isoformat(),
                    "resolved_at": (now - timedelta(hours=k + 1)).isoformat(),
                    "market_context": {"h1": {"rel_volume": 0.6, "atr_ratio": 1.5}, "d1": {},
                                       "correlated_prices": {"USDJPY": {"day_change_pct": 0.5, "corr_h1": -0.7}}}})
shadows.append({"id": "open1", "status": "OPEN", "symbol": "EURUSD", "side": "BUY", "blocked_by": ["G-USD"]})
config.SHADOW_FILE.write_text(json.dumps(shadows))
rows = shadow_store.learning_records("DEMO")
check("learning_records: resolved shadows only, shaped like closed trades with R and weight",
      len(rows) == 8 and rows[0]["source"] == "shadow" and rows[0]["weight"] == 0.5
      and {r["outcome"] for r in rows} == {"WIN", "LOSS"}, len(rows))
check("learning_records filters by account type", shadow_store.learning_records("LIVE") == [])

cal = calibration.build_report([], "DEMO", rows)
check("Calibration counts shadow trades (weighted) in its buckets",
      cal["shadow_trades"] == 8 and cal["closed_trades"] == 0 and cal["buckets"][0]["trades"] == 8
      and cal["buckets"][0]["weighted"] == 4.0, cal["buckets"])

config.ACTIVE_ACCOUNT = {"mode": "DEMO", "broker": None, "server": None}
window = auditor._learning_window("DEMO")
check("Auditor window = real closed trades + resolved shadow trades", len(window) == 8
      and all(t["source"] == "shadow" for t in window))
conditions = [{"metric": "cross.USDJPY.day_change_pct", "op": ">", "value": 0.3},
              {"metric": "h1.rel_volume", "op": "<", "value": 0.8}, {"metric": "h1.atr_ratio", "op": ">", "value": 1.3}]
ev = auditor.evaluate_rule("EURUSD", "BUY", conditions, window)
check("Rule evidence is weighted: 6 shadow losses count as 3, 2 wins as 1; net in R",
      ev["loss_weight"] == 3.0 and ev["win_weight"] == 1.0 and ev["net_r"] == round(0.5 * (-6 + 3.0), 3), ev["net_r"])
due = auditor.audit_due()
check("Audit schedule counts shadow trades (a bot with few real trades can still learn)",
      due["shadow_trades"] == 8 and due["closed_trades"] == 0 and due["due"], due)
raw_rule = {"affected_symbol": "EURUSD", "side": "BUY", "conditions": conditions, "evidence": "x"}
accepted = auditor._validate_rule(raw_rule, ["EURUSD", "USDJPY"], window, 0.5)
check("A rule backed by shadow evidence is accepted, with its shadow share recorded",
      accepted is not None and accepted["shadow_losses"] == 6 and accepted["weighted_losses"] == 3.0
      and accepted["matched_net_r"] < 0, accepted and accepted.get("weighted_losses"))
config.SHADOW_WEIGHT = 0.0
check("SHADOW_WEIGHT=0 turns shadow learning off", auditor._learning_window("DEMO") == [])
config.SHADOW_WEIGHT = 0.5

# ============================================================ 2. features: complete and side-signed
p = scalper.prepare(synthetic)
i = len(m5_close) - 15
buy, sell = scalper.features(p, i, "BUY", 0.00003), scalper.features(p, i, "SELL", 0.00003)
numeric = {k: v for k, v in buy.items() if k not in ("side", "feature_version")}
check("~30 features, all finite on a warmed-up chart", len(numeric) >= 28 and all(v is not None for v in numeric.values()),
      [k for k, v in numeric.items() if v is None])
check("Directional features flip sign for a SELL; non-directional ones do not",
      buy["d1_ema200_dist_atr"] == -sell["d1_ema200_dist_atr"] and buy["m5_atr_pct"] == sell["m5_atr_pct"]
      and abs((buy["m5_rsi"] - 50) + (sell["m5_rsi"] - 50)) < 1e-6)

# ============================================================ 3. journal module
journal.record({"kind": "test", "stage": "NO_PULLBACK", "action": "no setup"})
journal.record({"kind": "test", "stage": "SETUP", "action": "filled", "value": np.float64(1.5)})
entries = list(journal.iter_entries(1))
summary = journal.summary(1)
check("Journal: lines appended and read back (numpy values serialised)", len(entries) == 2 and entries[1]["value"] == 1.5)
check("Journal summary counts stages and actions", summary["stages"] == {"NO_PULLBACK": 1, "SETUP": 1}
      and summary["actions"] == {"no setup": 1, "filled": 1} and summary["days_on_disk"] == 1, summary)
config.JOURNAL_ENABLED = False
journal.record({"kind": "test"})
check("JOURNAL_ENABLED=False writes nothing", len(list(journal.iter_entries(1))) == 2)

# ============================================================ 4. many pairs: short prompts, USD read from the majors
import ai_brain
import data_engine
many = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD", "EURJPY", "GBPJPY", "AUDJPY",
        "CADJPY", "EURGBP", "EURCHF", "USDTRY", "USDZAR", "USDMXN", "EURNOK", "GBPNZD", "AUDNZD", "PLNJPY", "SEKNOK"]
rel = data_engine.related_symbols("EURJPY", many)
check("AI context for EURJPY: pairs sharing EUR or JPY first, then USD majors; no unrelated exotics; capped",
      rel[:3] == ["EURUSD", "USDJPY", "GBPJPY"] and "EURNOK" in rel and "PLNJPY" in rel and "USDTRY" not in rel
      and "SEKNOK" not in rel and len(rel) <= data_engine.RELATED_LIMIT, rel)
usd = ai_brain.usd_direction({"symbol": "EURUSD", "day_change_pct": -0.5, "correlated_prices": {
      "GBPUSD": {"day_change_pct": -0.4}, "USDJPY": {"day_change_pct": 0.3}, "USDTRY": {"day_change_pct": -3.0},
      "USDZAR": {"day_change_pct": -2.0}}})
check("USD direction counts only the major USD pairs (exotics trend on their own)",
      usd["pairs"] == 3 and "USDTRY" not in usd["moves"] and usd["label"] == "STRENGTHENING", usd)

# ============================================================ 5. sessions: which pairs the dashboard shows as active
import sessions
utc = lambda text: datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
check("Open sessions by the clock (DST-aware): Sydney+Tokyo, London, London+New York, New York; none at the weekend",
      sessions.open_sessions(utc("2026-10-01T03:00")) == ["Sydney", "Tokyo"]
      and sessions.open_sessions(utc("2026-10-01T10:00")) == ["London"]
      and sessions.open_sessions(utc("2026-10-01T14:00")) == ["London", "New York"]
      and sessions.open_sessions(utc("2026-10-01T20:00")) == ["New York"]
      and sessions.open_sessions(utc("2026-10-03T12:00")) == [])
check("Home sessions: EURJPY -> London + Tokyo; gold -> London + New York; USDTRY -> New York + London",
      sessions.home_sessions("EURJPY") == ["London", "Tokyo"] and sessions.home_sessions("XAUUSDm") == ["London", "New York"]
      and sessions.home_sessions("USDTRY") == ["New York", "London"])
st = sessions.pair_status(["AUDUSD", "EURGBP", "USDJPY", "EURJPY"],
                          {"AUDUSD": 1000.0, "EURGBP": 1000.0, "USDJPY": 1000.0, "EURJPY": 1000.0 - 3600},
                          utc("2026-10-01T03:00"))
check("Tokyo morning: AUDUSD and USDJPY active, EURGBP closed, EURJPY not quoting for an hour -> not active",
      st["pairs"]["AUDUSD"]["active"] and st["pairs"]["USDJPY"]["active"] and not st["pairs"]["EURGBP"]["active"]
      and not st["pairs"]["EURJPY"]["quoting"] and not st["pairs"]["EURJPY"]["active"], st["pairs"])

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
