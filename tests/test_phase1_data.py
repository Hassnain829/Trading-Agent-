"""Phase 1: Dukascopy data pipeline, Dukascopy backtests, shadow trades in the learning loops, features, journal."""
import json
import lzma
import struct
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

tmp = Path(tempfile.mkdtemp())
for attr, name in (("MEMORY_FILE", "memory.json"), ("RULES_FILE", "new_rules.json"), ("SETTINGS_FILE", "settings.json"),
                   ("RISK_STATE_FILE", "risk_state.json"), ("SHADOW_FILE", "shadow.json"), ("NEWS_CACHE_FILE", "news.json"),
                   ("AUDIT_LOCK_FILE", ".audit.lock"), ("LOCK_FILE", ".server.lock"), ("BACKTEST_DIR", "backtests"),
                   ("JOURNAL_DIR", "journal")):
    setattr(config, attr, tmp / name)
config.DEEPSEEK_API_KEY = "test"

import auditor
import backtest
import calibration
import dukascopy
import journal
import scalper
import shadow_store

dukascopy.DATA_DIR = tmp / "dukascopy"
dukascopy.RAW_DIR, dukascopy.BARS_DIR = dukascopy.DATA_DIR / "raw", dukascopy.DATA_DIR / "bars"
failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ============================================================ 1. Dukascopy file format + instruments
check("Instrument codes from broker names", dukascopy.instrument("EURUSDm") == "EURUSD"
      and dukascopy.instrument("XAUUSD.r") == "XAUUSD")
check("Price scale / point: FX 1e5 & 0.00001, JPY 1e3 & 0.001, gold 1e3 & 0.01",
      dukascopy.price_scale("EURUSD") == 1e5 and dukascopy.price_scale("USDJPY") == 1e3 and dukascopy.price_scale("XAUUSD") == 1e3
      and dukascopy.point_size("EURUSD") == 0.00001 and dukascopy.point_size("USDJPY") == 0.001 and dukascopy.point_size("XAUUSD") == 0.01)
raw = b"".join(struct.pack(">5if", 60 * m, 114000 + m, 114002 + m, 113990 + m, 114010 + m, 1.5) for m in range(3))
rows = dukascopy._decode(lzma.compress(raw, format=lzma.FORMAT_ALONE), 1e5)
check("Decode LZMA-alone M1 candles into real prices", rows.shape == (3, 6) and abs(rows[1][1] - 1.14001) < 1e-9
      and rows[2][0] == 120 and rows[0][5] == 1.5, rows[:2].tolist())
check("Empty file (market shut) -> no rows", len(dukascopy._decode(b"", 1e5)) == 0)
days = dukascopy.trading_days(1, end=date(2026, 9, 30))
check("Trading days: no Saturdays, newest first, about 1 year", days[0] == date(2026, 9, 30)
      and all(d.weekday() != 5 for d in days) and 300 < len(days) < 320)

# ============================================================ 2. time alignment (UTC -> broker server clock)
summer = pd.DatetimeIndex([datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)])
winter = pd.DatetimeIndex([datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)])
check("Server clock = New York + 7h (summer: 12:00 UTC -> 15:00; winter: 12:00 UTC -> 14:00)",
      dukascopy.to_server_time(summer)[0].hour == 15 and dukascopy.to_server_time(winter)[0].hour == 14)
roundtrip = pd.Timestamp(dukascopy.to_server_time(summer)[0]).value // 10**9
check("...and the backtest converts it back to the same UTC time",
      str(__import__("data_engine").server_epochs_to_utc([roundtrip], True)[0]) == "2026-07-01 12:00:00+00:00")

# ============================================================ 3. build bars from cached days
def put_day(code, day, side, rows):
    path = dukascopy._day_file(code, day, side)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(rows, dtype=float).reshape(-1, 6))


day = date(2026, 7, 1)  # Wednesday
minutes = range(20 * 60, 22 * 60)  # 20:00-22:00 UTC = 16:00-18:00 New York: spans the 17:00 NY rollover
bid = [[m * 60, 1.10000 + m * 1e-6, 1.10001 + m * 1e-6, 1.09995 + m * 1e-6, 1.10005 + m * 1e-6, 2.0] for m in minutes]
ask = [[r[0], r[1] + 0.00003, r[2] + 0.00003, r[3] + 0.00003, r[4] + 0.00003, 2.0] for r in bid]
put_day("EURUSD", day, "BID", bid)
put_day("EURUSD", day, "ASK", ask)
put_day("EURUSD", date(2026, 7, 2), "BID", [])  # an empty day must be tolerated
put_day("EURUSD", date(2026, 7, 2), "ASK", [])
built = dukascopy.build(["EURUSD"])
m5, d1 = dukascopy.load_bars("EURUSD", "M5"), dukascopy.load_bars("EURUSD", "D1")
check("Build: 2 hours of M1 -> 24 M5 bars; spread 3 points from ask - bid", len(m5) == 24
      and set(m5["spread"]) == {3} and list(m5.columns) == ["time", "open", "high", "low", "close", "tick_volume", "spread"],
      built)
d1_times = [pd.Timestamp(t, unit="s") for t in d1["time"]]
check("D1 bars split at the 17:00 New York rollover (00:00 server time), like the broker's",
      len(d1) == 2 and [t.strftime("%Y-%m-%d %H:%M") for t in d1_times] == ["2026-07-01 00:00", "2026-07-02 00:00"],
      d1_times)
check("available() / load_bars roundtrip", dukascopy.available("EURUSDm") and len(dukascopy.load_bars("EURUSDm", "H1")) == 2)

# ============================================================ 4. backtest on Dukascopy bars (synthetic 1-year trend)
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
for tf, bars in synthetic.items():
    bars.to_pickle(dukascopy.BARS_DIR / f"GBPUSD_{tf}.pkl.gz", compression="gzip")
loaded = backtest.load_history("GBPUSD", 2, source="dukascopy")
check("load_history(source='dukascopy') reads the built bars", len(loaded["M5"]) == len(m5_close) and set(loaded) == {"M5", "M15", "H1", "D1"})
spec = backtest.default_spec("USDJPY", 150.0)
check("Contract specs without MT5: USDJPY point 0.001, 1 point = 100k x 0.001 / price",
      spec["point"] == 0.001 and abs(spec["tick_value"] - 100_000 * 0.001 / 150) < 1e-9 and spec["digits"] == 3)
config.SCALP_STRICT_GUARD = config.SCALP_MEDIUM_TREND = False
backtest.data_engine.mt5.symbol_info = lambda s: None  # no MT5: default specs
report = backtest.run_backtest(["GBPUSD"], 2, 2.0, 100.0, source="dukascopy")
check("run_backtest on Dukascopy history: trades found, source recorded", report["summary"].get("trades", 0) >= 1
      and report["params"]["source"] == "dukascopy", report["summary"])

# ============================================================ 5. shadow trades feed the learning loops
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

# ============================================================ 6. features: complete and side-signed
p = scalper.prepare(synthetic)
i = len(m5_close) - 15
buy, sell = scalper.features(p, i, "BUY", 0.00003), scalper.features(p, i, "SELL", 0.00003)
numeric = {k: v for k, v in buy.items() if k not in ("side", "feature_version")}
check("~30 features, all finite on a warmed-up chart", len(numeric) >= 28 and all(v is not None for v in numeric.values()),
      [k for k, v in numeric.items() if v is None])
check("Directional features flip sign for a SELL; non-directional ones do not",
      buy["d1_ema200_dist_atr"] == -sell["d1_ema200_dist_atr"] and buy["m5_atr_pct"] == sell["m5_atr_pct"]
      and abs((buy["m5_rsi"] - 50) + (sell["m5_rsi"] - 50)) < 1e-6)

# ============================================================ 7. journal module
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

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
