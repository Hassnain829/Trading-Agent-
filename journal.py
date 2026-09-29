"""
Decision journal: one JSON line for every evaluation the engine makes.

Every closed M5 bar the scalper checks, and every AI decision, is written to
data/journal/YYYY-MM-DD.jsonl (UTC date) with the full feature set, the stage it stopped
at, the AI's answer and what happened next (no setup, skipped, vetoed, blocked, filled).
Trades and shadow trades link back by ticket / shadow id, so the learning code can join
every decision with its outcome. Writing never raises: a full disk must not stop trading.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

import config

logger = logging.getLogger("hedgefund.journal")
_LOCK = threading.Lock()


def journal_dir() -> Path:
    return config.JOURNAL_DIR


def _default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def record(entry: Dict[str, Any]) -> None:
    """Append one decision (blocking, thread-safe)."""
    if not config.JOURNAL_ENABLED:
        return
    now = datetime.now(timezone.utc)
    line = {"ts": now.isoformat(timespec="seconds"), **entry}
    path = journal_dir() / f"{now:%Y-%m-%d}.jsonl"
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False, default=_default) + "\n")
    except OSError as exc:
        logger.warning("[LEARNING] Decision journal write failed: %s", exc)


def iter_entries(days: int = 7) -> Iterator[Dict[str, Any]]:
    """Entries of the last ``days`` UTC days, oldest first (unreadable lines are skipped)."""
    today = datetime.now(timezone.utc).date()
    for offset in range(days - 1, -1, -1):
        path = journal_dir() / f"{today - timedelta(days=offset):%Y-%m-%d}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def summary(days: int = 1) -> Dict[str, Any]:
    """Counts by stage and action, for the dashboard."""
    stages: Dict[str, int] = {}
    actions: Dict[str, int] = {}
    total = 0
    for entry in iter_entries(days):
        total += 1
        stages[entry.get("stage") or "?"] = stages.get(entry.get("stage") or "?", 0) + 1
        if entry.get("action"):
            key = str(entry["action"]).split(":")[0]
            actions[key] = actions.get(key, 0) + 1
    files = sorted(journal_dir().glob("*.jsonl")) if journal_dir().exists() else []
    return {"entries": total, "stages": stages, "actions": actions, "days_on_disk": len(files),
            "size_mb": round(sum(f.stat().st_size for f in files) / 1e6, 2)}


def recent(limit: int = 50, days: int = 2) -> List[Dict[str, Any]]:
    entries = list(iter_entries(days))
    return entries[-limit:]
