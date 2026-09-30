"""Phase 3: ML dataset rows, purged walk-forward, acceptance rule, trade replay, training output, live filter."""
import json
import pickle
import sys
import tempfile
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
                   ("JOURNAL_DIR", "journal"), ("ML_DIR", "ml")):
    setattr(config, attr, tmp / name)
config.DEEPSEEK_API_KEY = "test"
config.LOSS_COOLDOWN_MINUTES, config.SCALP_MAX_TRADES_PER_SYMBOL, config.SHADOW_WEIGHT = 30, 4, 0.5

import journal
import scalper
from ml import dataset, train, validate

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ============================================================ 1. feature rows
feats = {"feature_version": scalper.FEATURE_VERSION, "m5_rsi": 35.0, "d1_adx": 22.0}
row = dataset.feature_row(feats, "SELL")
check("Feature row: all model inputs, side as is_buy", row is not None and set(row) == set(dataset.FEATURES)
      and row["is_buy"] == 0.0 and row["m5_rsi"] == 35.0 and row["weekday"] is None)
check("Other feature versions are refused", dataset.feature_row({**feats, "feature_version": 99}, "BUY") is None
      and dataset.feature_row(None, "BUY") is None)


# ============================================================ 2. synthetic datasets
def synthetic(signal: bool, months: int = 24, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01 08:00", tz="UTC")
    rows = []
    for k in range(months * 60):  # ~60 setups a month, two symbols, 12 hours apart
        entry = start + pd.Timedelta(hours=12 * k)
        x = rng.normal()
        win = rng.random() < (1 / (1 + np.exp(-3 * x)) if signal else 0.45)
        r = 1.5 if win else -1.0
        features = {name: rng.normal() for name in dataset.FEATURES}
        features["m5_rsi_extreme"] = x
        rows.append({"source": "backtest", "symbol": "EURUSD" if k % 2 else "GBPUSD", "side": "BUY",
                     "entry_time": entry, "exit_time": entry + pd.Timedelta(minutes=30),
                     "exit_time_stress": entry + pd.Timedelta(minutes=30), "day": f"{entry:%Y-%m-%d}",
                     "r": r, "r_stress": r - 0.1, "win": int(win), "exit_reason": "TP" if win else "SL", "weight": 1.0,
                     **features})
    return dataset._frame(rows)


good, noise = synthetic(True), synthetic(False)
seen = []
real_fit, real_predict = validate.fit, validate.predict
validate.fit = lambda name, rows: seen.append(("train", rows["exit_time"].max())) or real_fit(name, rows)
validate.predict = lambda model, rows: seen.append(("test", rows["entry_time"].min())) or real_predict(model, rows)
report = validate.walk_forward(good, trials=24)
validate.fit, validate.predict = real_fit, real_predict
pairs = [(seen[k][1], seen[k + 1][1]) for k in range(len(seen) - 1) if seen[k][0] == "train" and seen[k + 1][0] == "test"]
check("Purged: every training label closed a day before its test window",
      pairs and all(train_end < test_start - validate.EMBARGO for train_end, test_start in pairs), pairs[:2])
span = lambda w: (w[1] - w[0]).days
check("Two years: 1-month test windows after 150 training rows", report["folds"][0]["test"] == "2024-04-01..2024-05-01"
      and len(report["folds"]) == 21, (report["folds"][0], len(report["folds"])))
long_windows = validate.folds(synthetic(True, months=36), 150)
check("900+ days: 3-month test windows", long_windows and all(89 <= span(w) <= 92 for w in long_windows), long_windows[:2])
check("A real signal is learned (AUC > 0.7) and ACCEPTED",
      report["accepted"] in ("lgbm", "logit") and report["quality"]["logit"]["auc"] > 0.7, report["reason"])
chosen = report["results"][report["accepted"]]
check("...its trades beat no model at +1 pip", chosen["expectancy_r"] > report["results"]["rules"]["expectancy_r"] + 0.1,
      report["results"])
check("A linear signal: the simpler logistic model is preferred unless LightGBM is clearly better",
      report["accepted"] == "logit" or report["results"]["lgbm"]["expectancy_r"]
      >= report["results"]["logit"]["expectancy_r"] + config.ML_MIN_GAIN_R, report["results"])
rejected = validate.walk_forward(noise, trials=24)
check("Pure noise is REJECTED", rejected["accepted"] is None and "no model accepted" in rejected["reason"],
      rejected["reason"])
check("Short history: not enough rows -> no verdict", validate.walk_forward(good.head(100), 24)["accepted"] is None)
check("Break-even probability from average win and loss", abs(validate.breakeven_probability(
      pd.DataFrame({"r_stress": [1.4, 1.4, -1.1, -1.1]})) - 1.1 / 2.5) < 1e-9)

# ============================================================ 3. trade rules replayed
t = lambda h, m=0: pd.Timestamp("2026-03-02 08:00", tz="UTC") + pd.Timedelta(hours=h, minutes=m)
replay = pd.DataFrame([
    {"symbol": "EURUSD", "side": "BUY", "entry_time": t(0), "exit_time_stress": t(0, 50), "day": "d1", "r_stress": -1.0},
    {"symbol": "EURUSD", "side": "BUY", "entry_time": t(0, 20), "exit_time_stress": t(1), "day": "d1", "r_stress": 1.5},
    {"symbol": "EURUSD", "side": "BUY", "entry_time": t(1), "exit_time_stress": t(1, 30), "day": "d1", "r_stress": 1.5},
    {"symbol": "EURUSD", "side": "SELL", "entry_time": t(1), "exit_time_stress": t(1, 30), "day": "d1", "r_stress": 1.5},
    {"symbol": "EURUSD", "side": "BUY", "entry_time": t(2), "exit_time_stress": t(2, 10), "day": "d1", "r_stress": 1.5},
    {"symbol": "GBPUSD", "side": "BUY", "entry_time": t(0, 20), "exit_time_stress": t(1), "day": "d1", "r_stress": 1.5},
])
trades = validate.select_trades(replay)
check("Replay: overlap skipped, loss cooldown per side, other symbols independent",
      [(x["symbol"], x["side"], x["entry_time"][11:16]) for x in trades]
      == [("EURUSD", "BUY", "08:00"), ("GBPUSD", "BUY", "08:20"), ("EURUSD", "SELL", "09:00"), ("EURUSD", "BUY", "10:00")],
      trades)
config.SCALP_MAX_TRADES_PER_SYMBOL = 1
check("Replay: max trades per symbol per day", len([x for x in validate.select_trades(replay) if x["symbol"] == "EURUSD"]) == 1)
config.SCALP_MAX_TRADES_PER_SYMBOL = 4

# ============================================================ 4. live and shadow rows
config.MEMORY_FILE.write_text(json.dumps([{"ticket": 77, "symbol": "EURUSD", "realized_pnl": 3.0, "risk_percent": 1.0,
                                           "market_context": {"equity": 200.0}, "exit_time_utc": "2026-09-29T10:40:00+00:00",
                                           "exit_reason": "TP"}]))
journal.record({"symbol": "EURUSD", "stage": "SETUP", "setup": {"side": "BUY"}, "features": feats, "ticket": 77,
                "action": "filled"})
journal.record({"symbol": "EURUSD", "stage": "SETUP", "setup": {"side": "BUY"}, "features": feats, "ticket": 78})  # still open
config.SHADOW_FILE.write_text(json.dumps([
    {"id": "s1", "symbol": "GBPUSD", "side": "SELL", "status": "LOSS", "r_multiple": -1.0, "features": feats,
     "created_at": "2026-09-29T09:00:00+00:00", "resolved_at": "2026-09-29T09:40:00+00:00", "blocked_by": ["AI-VETO"]},
    {"id": "s2", "symbol": "GBPUSD", "side": "SELL", "status": "OPEN", "features": feats},
    {"id": "s3", "symbol": "USDJPY", "side": "BUY", "status": "WIN", "r_multiple": 1.5, "features": {"h1": {}}},
]))
live = dataset.live_rows()
check("Live fill joined by ticket: R from P/L / (equity x risk%)", len(live[live["source"] == "live"]) == 1
      and abs(live[live["source"] == "live"]["r"].iloc[0] - 1.5) < 1e-9, live[["source", "r"]].to_dict("records"))
check("Resolved scalp shadows join at SHADOW_WEIGHT; open and swing shadows do not",
      live[live["source"] == "shadow"]["weight"].tolist() == [0.5], live[["source", "weight"]].to_dict("records"))

# ============================================================ 5. training output
dataset.build = lambda symbols, days, source, progress=None: good
report = train.train(["EURUSD", "GBPUSD"], 500, "history")
with (config.ML_DIR / train.MODEL_FILE).open("rb") as handle:
    saved = pickle.load(handle)
check("Accepted model saved as APPROVED, refit on all rows, with its threshold",
      saved["approved"] and saved["model"] is not None and 0 < saved["threshold"] < 1
      and saved["features"] == dataset.FEATURES and (config.ML_DIR / report["file"]).exists(), saved["reason"])
check("Every run counts as a trial", train.trials() == config.BACKTEST_TRIALS + 2)

# ============================================================ 6. live filter (ml.model)
from ml import model as ml_model
row_feats = {**{name: float(good[name].iloc[0]) for name in dataset.FEATURES if name != "is_buy"},
             "feature_version": scalper.FEATURE_VERSION}
config.ML_FILTER = False
check("Switch off -> the filter does not apply", ml_model.evaluate(row_feats, "BUY") is None
      and not ml_model.status()["active"] and ml_model.status()["approved"])
config.ML_FILTER = True
strong, weak = dict(row_feats, m5_rsi_extreme=2.5), dict(row_feats, m5_rsi_extreme=-2.5)
high, low = ml_model.evaluate(strong, "BUY"), ml_model.evaluate(weak, "BUY")
check("Approved model: high-signal setup taken, low-signal skipped", high["take"] and not low["take"]
      and high["p_win"] > high["threshold"] > low["p_win"] and ml_model.status()["active"], (high, low))
check("Missing features are tolerated (NaN), other feature versions ignored",
      ml_model.evaluate({"feature_version": scalper.FEATURE_VERSION}, "SELL") is not None
      and ml_model.evaluate({"feature_version": 99}, "BUY") is None)
(config.ML_DIR / ml_model.MODEL_FILE).write_bytes(b"not a pickle")
check("A broken model file -> plain rules, no crash", ml_model.evaluate(strong, "BUY") is None
      and not ml_model.status()["trained"])
dataset.build = lambda symbols, days, source, progress=None: noise
train.train(["EURUSD", "GBPUSD"], 500, "history")
with (config.ML_DIR / train.MODEL_FILE).open("rb") as handle:
    saved = pickle.load(handle)
check("A rejected run replaces the model with an UNAPPROVED one (never used live)",
      not saved["approved"] and saved["model"] is None, saved["reason"])
check("...and the live filter stops using it at once", ml_model.evaluate(strong, "BUY") is None
      and ml_model.status()["trained"] and not ml_model.status()["active"])

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
