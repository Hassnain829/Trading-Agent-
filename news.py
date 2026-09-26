"""
Economic calendar guard.

Downloads this week's calendar (Forex Factory's public JSON feed by default)
every few hours and blocks new entries on a symbol from NEWS_BLOCK_BEFORE_MINUTES
before until NEWS_BLOCK_AFTER_MINUTES after any high-impact event for one of its
currencies (EURUSD: EUR and USD news; XAUUSD and US indices: USD news).

If the calendar cannot be downloaded the guard stays inactive and says so on the
dashboard; the last good copy is kept on disk and reused.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

import config
from data_engine import news_currencies, utc_now_iso
from memory_store import read_json_file, write_json_atomic

logger = logging.getLogger("hedgefund.news")

_state: Dict[str, Any] = {"events": None, "fetched_at": None, "error": None, "checked": 0.0}
RETRY_AFTER_FAILURE_SECONDS = 900


def _parse(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _high_impact(raw: Any) -> List[Dict[str, Any]]:
    events = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or str(item.get("impact", "")).lower() != "high":
            continue
        when = _parse(item.get("date"))
        if when is None:
            continue
        events.append({"title": str(item.get("title") or "")[:120], "currency": str(item.get("country") or "").upper(),
                       "time": when.isoformat(timespec="minutes")})
    events.sort(key=lambda event: event["time"])
    return events


def _load_cache() -> None:
    if _state["events"] is not None:
        return
    cached = read_json_file(config.NEWS_CACHE_FILE, dict, quarantine_corrupt=False)
    if isinstance(cached, dict) and isinstance(cached.get("events"), list):
        _state.update(events=cached["events"], fetched_at=cached.get("fetched_at"))


def refresh_if_stale(force: bool = False) -> bool:
    """Download the calendar when the copy is older than NEWS_REFRESH_HOURS (blocking). True when fresh data arrived."""
    if not config.NEWS_GUARD and not force:
        return False
    _load_cache()
    fetched = _parse(_state["fetched_at"])
    age = (datetime.now(timezone.utc) - fetched) if fetched else None
    if not force and age is not None and age < timedelta(hours=config.NEWS_REFRESH_HOURS):
        return False
    if not force and _state["error"] and time.monotonic() - _state["checked"] < RETRY_AFTER_FAILURE_SECONDS:
        return False
    _state["checked"] = time.monotonic()
    try:
        response = requests.get(config.NEWS_CALENDAR_URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        events = _high_impact(response.json())
    except (requests.RequestException, ValueError) as exc:
        _state["error"] = f"calendar download failed: {exc}"[:200]
        logger.warning("[NEWS] %s%s", _state["error"],
                       "; using the last saved calendar" if _state["events"] else "; news guard inactive")
        return False
    _state.update(events=events, fetched_at=utc_now_iso(), error=None)
    try:
        write_json_atomic(config.NEWS_CACHE_FILE, {"fetched_at": _state["fetched_at"], "events": events})
    except OSError as exc:
        logger.debug("[NEWS] Could not cache the calendar: %s", exc)
    logger.info("[NEWS] Economic calendar updated: %d high-impact events this week", len(events))
    return True


def _events() -> List[Dict[str, Any]]:
    _load_cache()
    return list(_state["events"] or [])


def blocking_event(symbol: str, now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """The high-impact event that blocks new entries on ``symbol`` right now, if any."""
    if not config.NEWS_GUARD:
        return None
    now = now or datetime.now(timezone.utc)
    currencies = set(news_currencies(symbol))
    before = timedelta(minutes=config.NEWS_BLOCK_BEFORE_MINUTES)
    after = timedelta(minutes=config.NEWS_BLOCK_AFTER_MINUTES)
    for event in _events():
        when = _parse(event["time"])
        if when and event["currency"] in currencies and when - before <= now <= when + after:
            return event
    return None


def upcoming(symbol: str, hours: float = 24.0, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """High-impact events for the symbol's currencies in the next ``hours``."""
    now = now or datetime.now(timezone.utc)
    currencies = set(news_currencies(symbol))
    horizon = now + timedelta(hours=hours)
    result = []
    for event in _events():
        when = _parse(event["time"])
        if when and event["currency"] in currencies and now - timedelta(minutes=config.NEWS_BLOCK_AFTER_MINUTES) <= when <= horizon:
            result.append(event)
    return result


def status(now: Optional[datetime] = None) -> Dict[str, Any]:
    """For the dashboard: whether the guard has data, and the next high-impact events."""
    now = now or datetime.now(timezone.utc)
    events = [e for e in _events() if (_parse(e["time"]) or now) >= now - timedelta(minutes=config.NEWS_BLOCK_AFTER_MINUTES)]
    return {"enabled": config.NEWS_GUARD, "fetched_at": _state["fetched_at"], "error": _state["error"],
            "has_data": _state["events"] is not None, "next_events": events[:6],
            "block_before_minutes": config.NEWS_BLOCK_BEFORE_MINUTES,
            "block_after_minutes": config.NEWS_BLOCK_AFTER_MINUTES}
