"""
Purged walk-forward validation of the setup filter (Phase 3.2 / 3.3).

Folds: expanding training window, then the next test window (3 months on multi-year data, 1 month on
short data). Training rows whose outcome was still open near the test start are purged, plus a 1-day
embargo, so no label overlaps the test period. Live and shadow rows join training folds that come after
them; scoring uses the backtest rows only (the same rules replayed on history).

Two models: LightGBM (calibrated) and a logistic-regression baseline. A setup is taken when its
predicted P(win) makes the expected R positive given the training fold's average win and loss.
The chosen setups are replayed through the live trade rules (one trade at a time per symbol, loss
cooldown, max per day) with every spread 1 pip wider. A model is ACCEPTED only when its trades beat
both "no model" and the other model by ML_MIN_GAIN_R per trade and pass the Deflated Sharpe test.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import validation
from ml.dataset import FEATURES

EMBARGO = pd.Timedelta(days=1)
MODELS = ("lgbm", "logit")


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def make_model(name: str):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.3, max_iter=2000))
    from lightgbm import LGBMClassifier
    booster = LGBMClassifier(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                             random_state=7, verbose=-1)
    return CalibratedClassifierCV(booster, method="sigmoid", cv=3)


def fit(name: str, rows: pd.DataFrame):
    model = make_model(name)
    X, y, w = rows[FEATURES].to_numpy(float), rows["win"].to_numpy(int), rows["weight"].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "logit":
            model.fit(X, y, logisticregression__sample_weight=w)
        else:
            model.fit(X, y, sample_weight=w)
    return model


def predict(model, rows: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict_proba(rows[FEATURES].to_numpy(float))[:, 1]


def breakeven_probability(rows: pd.DataFrame, column: str = "r_stress") -> float:
    """P(win) at which the expected R is zero, from the average win and loss (after costs)."""
    r = rows[column].dropna()
    wins, losses = r[r > 0], -r[r <= 0]
    if not len(wins) or not len(losses):
        return 0.5
    return float(losses.mean() / (wins.mean() + losses.mean()))


# -----------------------------------------------------------------------------
# Trade rules replayed on chosen setups
# -----------------------------------------------------------------------------
def select_trades(rows: pd.DataFrame, r_col: str = "r_stress", exit_col: str = "exit_time_stress") -> List[Dict[str, Any]]:
    """The live rules on a set of setups: one trade at a time per symbol, loss cooldown, max per day."""
    cooldown = pd.Timedelta(minutes=config.LOSS_COOLDOWN_MINUTES)
    trades: List[Dict[str, Any]] = []
    usable = rows.dropna(subset=[r_col, exit_col]).sort_values("entry_time", kind="stable")
    for symbol, group in usable.groupby("symbol", sort=False):
        busy_until, last_loss, per_day = None, {"BUY": None, "SELL": None}, {}
        for row in group.itertuples(index=False):
            entry, exit_ = row.entry_time, getattr(row, exit_col)
            if busy_until is not None and entry <= busy_until:
                continue
            if per_day.get(row.day, 0) >= config.SCALP_MAX_TRADES_PER_SYMBOL:
                continue
            if last_loss[row.side] is not None and entry - last_loss[row.side] < cooldown:
                continue
            r = float(getattr(row, r_col))
            trades.append({"symbol": symbol, "side": row.side, "entry_time": entry.isoformat(), "r": r})
            busy_until = exit_
            per_day[row.day] = per_day.get(row.day, 0) + 1
            if r < 0:
                last_loss[row.side] = exit_
    trades.sort(key=lambda t: t["entry_time"])
    return trades


def trade_stats(trades: List[Dict[str, Any]], trials: int) -> Dict[str, Any]:
    rs = [t["r"] for t in trades]
    if not rs:
        return {"trades": 0, "expectancy_r": None, "net_r": 0.0, "win_rate": None, "dsr": None}
    sig = validation.significance(rs, trials)
    return {"trades": len(rs), "expectancy_r": round(float(np.mean(rs)), 3), "net_r": round(float(np.sum(rs)), 2),
            "win_rate": round(float(np.mean(np.array(rs) > 0)) * 100, 1), "dsr": sig.get("dsr"),
            "t_stat": sig.get("t_stat")}


# -----------------------------------------------------------------------------
# Walk-forward
# -----------------------------------------------------------------------------
def folds(history: pd.DataFrame, min_train: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """(test start, test end) windows over the backtest rows, after enough training rows."""
    if history.empty:
        return []
    first, last = history["entry_time"].min(), history["entry_time"].max()
    months = 3 if (last - first) >= pd.Timedelta(days=900) else 1
    start = first.normalize().replace(day=1)
    windows = []
    while start <= last:
        end = start + pd.DateOffset(months=months)
        if (history["entry_time"] < start).sum() >= min_train and ((history["entry_time"] >= start)
                                                                     & (history["entry_time"] < end)).any():
            windows.append((start, end))
        start = end
    return windows


def walk_forward(data: pd.DataFrame, trials: int, min_train: int = 150) -> Dict[str, Any]:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    history = data[data["source"] == "backtest"]
    windows = folds(history, min_train)
    scored: List[pd.DataFrame] = []
    fold_log = []
    for start, end in windows:
        train = data[data["exit_time"] < start - EMBARGO]  # purge: outcome known before the test (+ embargo)
        test = history[(history["entry_time"] >= start) & (history["entry_time"] < end)].copy()
        if train["win"].nunique() < 2:
            continue
        threshold = breakeven_probability(train)
        test["threshold"] = threshold
        for name in MODELS:
            test[f"p_{name}"] = predict(fit(name, train), test)
        scored.append(test)
        fold_log.append({"test": f"{start:%Y-%m-%d}..{end:%Y-%m-%d}", "train_rows": int(len(train)),
                         "test_rows": int(len(test)), "threshold": round(threshold, 3),
                         **{f"taken_{n}": int((test[f"p_{n}"] >= threshold).sum()) for n in MODELS}})
    if not scored:
        return {"accepted": None, "reason": f"not enough history for a walk-forward ({len(history)} setups, "
                                            f"{min_train} needed before the first test month)", "folds": []}
    oos = pd.concat(scored)
    results = {"rules": trade_stats(select_trades(oos), trials)}
    quality = {}
    for name in MODELS:
        chosen = oos[oos[f"p_{name}"] >= oos["threshold"]]
        results[name] = trade_stats(select_trades(chosen), trials)
        try:
            auc = roc_auc_score(oos["win"], oos[f"p_{name}"])
        except ValueError:
            auc = None
        quality[name] = {"auc": round(float(auc), 3) if auc is not None else None,
                         "brier": round(float(brier_score_loss(oos["win"], oos[f"p_{name}"])), 4),
                         "calibration": calibration_table(oos[f"p_{name}"], oos["win"])}
    verdict = accept(results)
    return {**verdict, "results": results, "quality": quality, "folds": fold_log,
            "oos_setups": int(len(oos)), "base_win_rate": round(float(oos["win"].mean()) * 100, 1),
            "trials": trials}


def calibration_table(p: pd.Series, win: pd.Series, bins: int = 5) -> List[Dict[str, Any]]:
    """Predicted vs actual win rate in probability quantiles (a calibrated model matches)."""
    frame = pd.DataFrame({"p": p.to_numpy(), "win": win.to_numpy()})
    try:
        frame["bin"] = pd.qcut(frame["p"], bins, duplicates="drop")
    except ValueError:
        return []
    return [{"predicted": round(float(g["p"].mean()), 3), "actual": round(float(g["win"].mean()), 3), "n": int(len(g))}
            for _, g in frame.groupby("bin", observed=True)]


def accept(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    The plan's acceptance rule, out of sample and at +1 pip: a model must beat "no model" by ML_MIN_GAIN_R
    per trade, stay positive and pass the Deflated Sharpe test. LightGBM must also beat the logistic
    baseline by ML_MIN_GAIN_R; when both pass otherwise, the simpler logistic model is preferred.
    """
    rules = results["rules"].get("expectancy_r")
    problems: Dict[str, List[str]] = {}
    for name in MODELS:
        mine = results[name]
        found = []
        if not mine["trades"] or mine["trades"] < 30:
            found.append(f"only {mine['trades']} trades")
        else:
            if mine["expectancy_r"] <= 0:
                found.append("negative after +1 pip")
            if rules is None or mine["expectancy_r"] < rules + config.ML_MIN_GAIN_R:
                found.append(f"{mine['expectancy_r']}R does not beat no model ({rules}R) by {config.ML_MIN_GAIN_R}R")
            if (mine["dsr"] or 0) < config.ML_MIN_DSR:
                found.append(f"Deflated Sharpe {mine['dsr']} < {config.ML_MIN_DSR} (could be luck)")
        problems[name] = found
    logit, lgbm = results["logit"], results["lgbm"]
    if not problems["lgbm"] and (problems["logit"]
                                 or lgbm["expectancy_r"] >= logit["expectancy_r"] + config.ML_MIN_GAIN_R):
        return {"accepted": "lgbm", "reason": "LightGBM beats no model and the logistic baseline out of sample at +1 pip"}
    if not problems["logit"]:
        return {"accepted": "logit", "reason": "logistic regression beats no model out of sample at +1 pip"
                                               + ("" if problems["lgbm"] else "; LightGBM was not clearly better")}
    return {"accepted": None, "reason": "no model accepted: " + " | ".join(
        f"{name}: {'; '.join(found)}" for name, found in problems.items())}
