"""
Live setup filter (Phase 3.4).

When ML_FILTER is on AND data/ml/model.pkl holds a model the walk-forward APPROVED for the current
feature version, every scalp setup gets a predicted P(win); setups below the model's break-even
probability are skipped before the AI is asked (and followed as shadow trades, blocker ML-FILTER).
In every other case (switch off, no model, unapproved model, any error) the plain rules run as before.
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np

import config
import scalper
from ml import monitor
from ml.dataset import feature_row

logger = logging.getLogger("hedgefund.ml")
MODEL_FILE = "model.pkl"
_cache: Dict[str, Any] = {"stamp": None, "saved": None}


def load() -> Optional[Dict[str, Any]]:
    """The saved training result (cached on the file's modification time); None if there is none."""
    path = config.ML_DIR / MODEL_FILE
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        _cache.update(stamp=None, saved=None)
        return None
    if stamp != _cache["stamp"]:
        try:
            with path.open("rb") as handle:
                saved = pickle.load(handle)
        except Exception as exc:  # a half-written or incompatible file must not stop trading
            logger.warning("[ML] Could not load %s: %s; trading on the plain rules", path.name, exc)
            saved = None
        _cache.update(stamp=stamp, saved=saved if isinstance(saved, dict) else None)
    return _cache["saved"]


def usable(saved: Optional[Dict[str, Any]]) -> bool:
    return bool(saved and saved.get("approved") and saved.get("model") is not None
                and saved.get("feature_version") == scalper.FEATURE_VERSION
                and not monitor.drifted(saved.get("trained_at")))


def status() -> Dict[str, Any]:
    saved = load() or {}
    drift = (monitor.state().get("drift") or {}) if saved else {}
    return {
        "drift": monitor.drifted(saved.get("trained_at")) if saved else None,
        "live_check": {k: drift.get(k) for k in ("trades", "predicted_win_rate", "actual_win_rate")}
        if drift.get("model_trained_at") == saved.get("trained_at") else None,
        "next_retrain": monitor.next_retrain(saved.get("trained_at")),
        "last_attempt": monitor.state().get("last_result"),
        "enabled": config.ML_FILTER, "active": config.ML_FILTER and usable(saved), "trained": bool(saved),
        "approved": bool(saved.get("approved")), "model": saved.get("model_name"), "trained_at": saved.get("trained_at"),
        "reason": saved.get("reason"), "threshold": saved.get("threshold"), "expected": saved.get("expected"),
        "data_key": saved.get("data_key"), "validation_file": saved.get("validation_file"),
        "feature_version_ok": saved.get("feature_version") in (None, scalper.FEATURE_VERSION),
    }


def evaluate(features: Optional[Dict[str, Any]], side: str) -> Optional[Dict[str, Any]]:
    """{p_win, threshold, take, model} for a live setup, or None when the filter does not apply."""
    if not config.ML_FILTER:
        return None
    saved = load()
    if not usable(saved):
        return None
    row = feature_row(features or {}, side)
    if row is None:
        return None
    try:
        values = np.array([[np.nan if row.get(name) is None else float(row[name]) for name in saved["features"]]])
        p_win = float(saved["model"].predict_proba(values)[:, 1][0])
    except Exception as exc:
        logger.warning("[ML] Prediction failed (%s); this setup goes to the plain rules", exc)
        return None
    threshold = float(saved["threshold"])
    return {"p_win": round(p_win, 3), "threshold": round(threshold, 3), "take": p_win >= threshold,
            "model": saved.get("model_name")}
