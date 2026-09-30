"""FXCM + HistData history (merge, time zones, spreads, bars, backtest source) and ML upkeep (drift, retrain)."""
import gzip
import io
import json
import pickle
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

tmp = Path(tempfile.mkdtemp())
for attr, name in (("MEMORY_FILE", "memory.json"), ("RULES_FILE", "new_rules.json"), ("SETTINGS_FILE", "settings.json"),
                   ("RISK_STATE_FILE", "risk_state.json"), ("SHADOW_FILE", "shadow.json"), ("NEWS_CACHE_FILE", "news.json"),
                   ("AUDIT_LOCK_FILE", ".audit.lock"), ("LOCK_FILE", ".server.lock"), ("BACKTEST_DIR", "backtests"),
                   ("JOURNAL_DIR", "journal"), ("ML_DIR", "ml")):
    setattr(config, attr, tmp / name)
config.DEEPSEEK_API_KEY = "test"

import backtest
import fxhistory
import journal
import scalper
from ml import model as ml_model
from ml import monitor

fxhistory.DATA_DIR = tmp / "history"
fxhistory.RAW_DIR, fxhistory.BARS_DIR = fxhistory.DATA_DIR / "raw", fxhistory.DATA_DIR / "bars"
failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ============================================================ 1. raw files -> merged minutes
def fxcm_week(code, year, week, start, minutes, bid=1.10000, spread=0.00008):
    rows = ["DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose"]
    for k in range(minutes):
        t = start + timedelta(minutes=k)
        b = bid + k * 1e-5
        rows.append(f"{t:%m/%d/%Y %H:%M:%S}.000,{b},{b + 2e-5},{b - 2e-5},{b},{b + spread},{b + spread + 2e-5},{b + spread - 2e-5},{b + spread}")
    path = fxhistory._fxcm_file(code, year, week)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress("\n".join(rows).encode()))


def histdata_year(code, year, rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"DAT_ASCII_{code}_M1_{year}.csv", "\n".join(rows))
    path = fxhistory._histdata_file(code, year, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


monday = datetime(2024, 3, 4, 8, 0)  # UTC; New York is on winter time (EST) until 10 March
fxcm_week("EURUSD", 2024, 10, monday, 10)
histdata_year("EURUSD", 2024, [
    "20240304 030000;1.2;1.2;1.2;1.2;0",   # 08:00 UTC Monday: FXCM has this day -> ignored
    "20240305 030000;1.3;1.3;1.3;1.3;0",   # 08:00 UTC Tuesday: FXCM has no Tuesday -> fills
    "20240305 030100;1.3;1.3;1.3;1.3;0",
    "20240717 030000;1.4;1.4;1.4;1.4;0",   # summer: New York is on EDT (UTC-4)
])
m1, info = fxhistory.merged_m1("EURUSD")
check("FXCM minutes kept, HistData only on days FXCM lacks", info["fxcm_minutes"] == 10 and info["histdata_minutes"] == 3
      and not (m1["close"] == 1.2).any(), info)
tuesday = m1[m1.index.normalize() == pd.Timestamp("2024-03-05", tz="UTC")]
summer = m1[m1["close"] == 1.4]
check("HistData New York local time -> UTC: +5h in winter, +4h in summer (daylight saving)",
      str(tuesday.index[0]) == "2024-03-05 08:00:00+00:00" and str(summer.index[0]) == "2024-07-17 07:00:00+00:00",
      (tuesday.index[0], summer.index[0]))
check("FXCM spread = ask - bid; filled minutes get FXCM's median for that hour",
      abs(m1["spread"].iloc[0] - 0.00008) < 1e-9 and abs(tuesday["spread"].iloc[0] - 0.00008) < 1e-9)
histdata_year("XAUUSD", 2024, ["20240304 030000;2100.0;2100.5;2099.5;2100.2;0"])
_, gold = fxhistory.merged_m1("XAUUSD")
check("Gold (no FXCM feed): fixed default spread", gold["fxcm_minutes"] == 0 and gold["median_spread_points"] == 25, gold)

# ============================================================ 2. bars on the server clock + backtest source
built = fxhistory.build(["EURUSDm", "XAUUSD"], progress=None)
m5 = fxhistory.load_bars("EURUSDm", "M5")
first = pd.Timestamp(int(m5["time"].iloc[0]), unit="s")
check("Bars on the broker clock (New York + 7h): 08:00 UTC in winter -> 10:00", str(first) == "2024-03-04 10:00:00", first)
check("M5 bars: bid OHLC, spread in points, no volume", list(m5.columns) == ["time", "open", "high", "low", "close", "tick_volume", "spread"]
      and int(m5["spread"].iloc[0]) == 8 and int(m5["tick_volume"].sum()) == 0, m5.head(2).to_dict("records"))
check("available() and info(): the March -> July jump counts as one gap", fxhistory.available("EURUSD")
      and fxhistory.info("EURUSD")["gaps_over_4_days"] == 1, fxhistory.info("EURUSD"))
check("Backtest knows the source", "history" in backtest.SOURCES and backtest.default_days("history") == 1250
      and backtest.default_days("mt5") == 330)
check("best_source prefers the 5-year history when every pair has it",
      backtest.best_source(["EURUSD", "XAUUSD"]) == "history" and backtest.best_source(["EURUSD", "GBPUSD"]) == "mt5")
try:
    backtest.load_history("EURUSD", 5, source="history")
    refused = ""
except ValueError as exc:
    refused = str(exc)
check("A 2-day history is refused by the coverage check (not tested silently)", "EMA200" in refused, refused)

# ============================================================ 3. drift pause
trained_at = "2026-09-01T00:00:00+00:00"
model_file = config.ML_DIR / ml_model.MODEL_FILE
config.ML_DIR.mkdir(parents=True, exist_ok=True)


class Always:
    def predict_proba(self, X):
        import numpy as np
        return np.tile([0.3, 0.7], (len(X), 1))


with model_file.open("wb") as handle:
    pickle.dump({"approved": True, "model": Always(), "model_name": "logit", "threshold": 0.45, "trained_at": trained_at,
                 "features": ["m5_rsi", "is_buy"], "feature_version": scalper.FEATURE_VERSION}, handle)
config.ML_FILTER = True
feats = {"feature_version": scalper.FEATURE_VERSION, "m5_rsi": 35.0}
check("Approved model acts before any live trades", ml_model.evaluate(feats, "BUY") is not None)

records, day = [], datetime(2026, 9, 10, tzinfo=timezone.utc)
for k in range(25):
    won = k < 5  # 20% live wins against 70% predicted
    records.append({"ticket": 900 + k, "status": "CLOSED", "realized_pnl": 10.0 if won else -10.0, "risk_percent": 1.0,
                    "market_context": {"equity": 1000.0}})
    journal.record({"symbol": "EURUSD", "ticket": 900 + k, "ml": {"p_win": 0.7, "threshold": 0.45, "take": True, "model": "logit"}})
config.MEMORY_FILE.write_text(json.dumps(records))
report = monitor.check_drift(trained_at, force=True)
check("Drift detected: 20% won vs 70% predicted over 25 trades", report["drift"] and report["trades"] == 25
      and report["actual_win_rate"] == 0.2, report)
check("A drifted model is paused (plain rules decide)", ml_model.evaluate(feats, "BUY") is None
      and not ml_model.status()["active"] and "live win rate 20%" in ml_model.status()["drift"])
check("Trades decided before the model was trained are not counted",
      monitor.drift_report("2099-01-01T00:00:00+00:00")["trades"] == 0)
check("A new model starts with a clean record", monitor.drifted("2026-10-01T00:00:00+00:00") is None)
few = monitor.drift_report(trained_at)
config.ML_DRIFT_MIN_TRADES = 50
check("No verdict before ML_DRIFT_MIN_TRADES", not monitor.drift_report(trained_at)["drift"])
config.ML_DRIFT_MIN_TRADES = 20

# ============================================================ 4. retrain schedule
config.ML_AUTO_RETRAIN_DAYS = 7
old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
fresh = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
monitor._save({"last_attempt_at": None})
check("Retrain due after 7 days (or with no model), not before", monitor.retrain_due(old) and monitor.retrain_due(None)
      and not monitor.retrain_due(fresh))
monitor.record_attempt({"accepted": None, "reason": "no model accepted"})
check("A failed attempt is not repeated for 12 hours", not monitor.retrain_due(old)
      and ml_model.status()["last_attempt"]["reason"] == "no model accepted")
config.ML_AUTO_RETRAIN_DAYS = 0
check("0 turns automatic retraining off", not monitor.retrain_due(None) and monitor.next_retrain(old) is None)

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
