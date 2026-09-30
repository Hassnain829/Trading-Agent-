"""
ML upkeep (Phase 3.5): scheduled retraining and a live drift check.

* Retraining: every ML_AUTO_RETRAIN_DAYS the engine rebuilds the dataset (best downloaded history
  + every live trade + every resolved shadow trade), re-runs the purged walk-forward, and replaces
  the model. A model that fails the acceptance test is saved as NOT approved, so the filter stops
  acting until a later retrain passes. Runs only outside trading hours (see main._ml_maintenance).
* Drift: for trades the current model let through, compare its predicted win rate with the live
  result. When the live win rate is clearly lower (more than two standard errors, after
  ML_DRIFT_MIN_TRADES trades), the model is paused (ml.model.usable -> False) until it is retrained.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import calibration
import config
import journal
import memory_store

STATE_NAME = "monitor.json"
CHECK_EVERY = timedelta(minutes=30)
_cache: Dict[str, Any] = {"stamp": None, "state": {}}


def _path():
    return config.ML_DIR / STATE_NAME


def state() -> Dict[str, Any]:
    """Saved monitor state (cached on the file's modification time)."""
    try:
        stamp = os.stat(_path()).st_mtime_ns
    except OSError:
        return {}
    if stamp != _cache["stamp"]:
        try:
            _cache.update(stamp=stamp, state=json.loads(_path().read_text(encoding="utf-8")))
        except (OSError, ValueError):
            _cache.update(stamp=stamp, state={})
    return _cache["state"]


def _save(update: Dict[str, Any]) -> None:
    config.ML_DIR.mkdir(parents=True, exist_ok=True)
    data = {**state(), **update}
    tmp = _path().with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, _path())
    _cache.update(stamp=os.stat(_path()).st_mtime_ns, state=data)  # two saves can share one timestamp


def _parse(value: Any) -> Optional[datetime]:
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


# -----------------------------------------------------------------------------
# Drift
# -----------------------------------------------------------------------------
def drift_report(model_trained_at: Optional[str], journal_days: int = 400) -> Dict[str, Any]:
    """Predicted vs actual win rate of the closed trades the current model approved."""
    since = _parse(model_trained_at)
    closed = {}
    for record in memory_store.load_trade_memory():
        r = calibration._r_multiple(record)
        if record.get("ticket") and r is not None and record.get("status") == "CLOSED":
            closed[int(record["ticket"])] = r
    predicted, wins = [], []
    for entry in journal.iter_entries(journal_days):
        ml, ticket = entry.get("ml") or {}, entry.get("ticket")
        if not ml.get("take") or not ticket or int(ticket) not in closed:
            continue
        if since and (_parse(entry.get("ts")) or since) < since:
            continue  # decided by an earlier model
        predicted.append(float(ml["p_win"]))
        wins.append(closed[int(ticket)] > 0)
    n = len(predicted)
    if not n:
        return {"trades": 0, "drift": False}
    p, actual = sum(predicted) / n, sum(wins) / n
    se = math.sqrt(max(p * (1 - p), 1e-6) / n)
    drift = n >= config.ML_DRIFT_MIN_TRADES and actual < p - 2 * se
    return {"trades": n, "predicted_win_rate": round(p, 3), "actual_win_rate": round(actual, 3),
            "margin": round(2 * se, 3), "drift": drift,
            "reason": (f"live win rate {actual:.0%} vs {p:.0%} predicted over {n} trades" if drift else None)}


def check_drift(model_trained_at: Optional[str], force: bool = False) -> Dict[str, Any]:
    """Refresh the drift verdict at most every CHECK_EVERY; logs nothing itself."""
    current = state().get("drift") or {}
    last = _parse(current.get("checked_at"))
    if not force and last and current.get("model_trained_at") == model_trained_at \
            and datetime.now(timezone.utc) - last < CHECK_EVERY:
        return current
    report = {**drift_report(model_trained_at), "model_trained_at": model_trained_at,
              "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _save({"drift": report})
    return report


def drifted(model_trained_at: Optional[str]) -> Optional[str]:
    """The drift reason if the model trained at ``model_trained_at`` is paused, else None."""
    report = state().get("drift") or {}
    if report.get("drift") and report.get("model_trained_at") == model_trained_at:
        return report.get("reason") or "live results drifted from the model"
    return None


# -----------------------------------------------------------------------------
# Retraining schedule
# -----------------------------------------------------------------------------
def retrain_due(model_trained_at: Optional[str]) -> bool:
    days = config.ML_AUTO_RETRAIN_DAYS
    if days <= 0:
        return False
    now = datetime.now(timezone.utc)
    last_attempt = _parse(state().get("last_attempt_at"))
    if last_attempt and now - last_attempt < timedelta(hours=12):
        return False  # a failed or rejected attempt is not repeated all night
    trained = _parse(model_trained_at)
    return trained is None or now - trained >= timedelta(days=days)


def record_attempt(result: Dict[str, Any]) -> None:
    _save({"last_attempt_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "last_result": result})


def next_retrain(model_trained_at: Optional[str]) -> Optional[str]:
    days = config.ML_AUTO_RETRAIN_DAYS
    if days <= 0:
        return None
    trained = _parse(model_trained_at)
    return (trained + timedelta(days=days)).isoformat(timespec="seconds") if trained else "at the next quiet period"
