"""
Autonomous AI Hedge Fund: FastAPI server, dashboard API and 24/7 trading engine.

Run with:  .venv\\Scripts\\python.exe main.py
Dashboard: http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Literal, Optional, Set

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import MetaTrader5 as mt5

import ai_brain
import auditor
import calibration
import config
import data_engine
import intraday
import execution
import journal
import memory_store
import news
import rule_engine
import scalper
import sessions
import settings_store
import shadow_store
from ml import agent


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
class _RingBufferHandler(logging.Handler):
    """Keeps the most recent log lines for the dashboard console."""

    def __init__(self, capacity: int = 300) -> None:
        super().__init__(level=logging.INFO)
        self.records: Deque[Dict[str, Any]] = deque(maxlen=capacity)
        self.sequence = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sequence += 1
            self.records.append({
                "seq": self.sequence,
                "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="seconds"),
                "level": record.levelname,
                "message": record.getMessage(),
            })
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


LOG_BUFFER = _RingBufferHandler()


def _configure_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    root.addHandler(console)
    root.addHandler(LOG_BUFFER)
    for noisy in ("uvicorn.access", "urllib3", "httpx", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger("hedgefund.main")


# -----------------------------------------------------------------------------
# Global engine state
# -----------------------------------------------------------------------------
bot_state: Dict[str, Any] = {
    "is_running": False,
    "interval": config.SCAN_INTERVAL_SECONDS,
    "risk_percent": config.DEFAULT_RISK_PERCENT,
    "equity": 0.0,
    "last_logic": "",
    "last_confidence": 0,
    "last_signal": "HOLD",
    "last_entry_price": None,
    "last_stop_loss": None,
    "last_take_profit": None,
    "trade_history": [],
    "open_positions": [],
    "learned_rules": [],
    "last_audit_at": None,
    # Extended telemetry for the dashboard.
    "balance": 0.0,
    "floating_pnl": 0.0,
    "currency": "",
    "account": {},
    "mt5_connected": False,
    "last_symbol": None,
    "last_raw_signal": "HOLD",
    "last_base_confidence": 0,
    "last_applied_rules": [],
    "last_stops_source": None,
    "last_sl_atr_multiple": None,
    "last_tp_atr_multiple": None,
    "last_sl_distance": None,
    "last_tp_distance": None,
    "last_risk_reward": None,
    "last_atr": None,
    "last_atr_source": None,
    "last_stop_notes": [],
    "last_digits": None,
    "last_usd_direction": None,
    "last_decision_at": None,
    "decisions": {},
    "scan_count": 0,
    "current_symbol": None,
    "last_scan_at": None,
    "last_scan_duration": None,
    "next_scan_at": None,
    "audit_in_progress": False,
    "last_audit_result": None,
    "day_key": None,
    "day_start_equity": None,
    "daily_drawdown_pct": 0.0,
    "day_pnl_pct": 0.0,
    "circuit_breaker": False,
    "profit_target_hit": False,
    "last_error": None,
    "stats": {},
    "fills_version": "empty",
    "account_mode": None,
    "symbol_map": {},
    "unresolved_symbols": [],
    "day_login": None,
    "breaker_flattened": False,
    # Account protection across days: equity peak, total drawdown kill-switch and risk throttle.
    "guard_login": None,
    "peak_equity": None,
    "total_drawdown_pct": 0.0,
    "kill_switch": False,
    "kill_switch_at": None,
    "kill_switch_flattened": False,
    "risk_throttled": False,
    "open_risk": {},
    "unchanged_symbols": [],
    "started_at": data_engine.utc_now_iso(),
}

templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))
_background_tasks: Set[asyncio.Task] = set()
_server: Optional[uvicorn.Server] = None
_history_cache: Dict[str, Any] = {"stamp": None, "version": "empty", "ledger": [], "history": [], "stats": {},
                                  "last_execution": None}
LEDGER_LIMIT = 1000
HOUSEKEEPING_SECONDS = 60


# -----------------------------------------------------------------------------
# State refresh helpers (blocking: call from a worker thread)
# -----------------------------------------------------------------------------
_RISK_STATE_KEYS = ("day_key", "day_start_equity", "circuit_breaker", "profit_target_hit", "breaker_flattened")
# Saved in the same per-login entry but never reset by a new trading day.
_ACCOUNT_GUARD_KEYS = ("peak_equity", "kill_switch", "kill_switch_at", "kill_switch_flattened")
_peak_saved: Dict[str, Any] = {"login": None, "peak": None}


def _load_risk_state(login: int) -> Optional[Dict[str, Any]]:
    """Today's saved daily-risk state for this account (survives restarts), or None."""
    document = memory_store.read_json_file(config.RISK_STATE_FILE, dict, quarantine_corrupt=False)
    state = document.get(str(login)) if isinstance(document, dict) else None
    if isinstance(state, dict) and state.get("day_key") == data_engine.trading_day_key():
        return state
    return None


def _save_risk_state(login: int) -> None:
    try:
        with memory_store.file_lock(config.RISK_STATE_FILE):
            document = memory_store.read_json_file(config.RISK_STATE_FILE, dict, quarantine_corrupt=False)
            document = document if isinstance(document, dict) else {}
            entry = {**{key: bot_state.get(key) for key in _RISK_STATE_KEYS}, "saved_at": data_engine.utc_now_iso()}
            if bot_state.get("guard_login") == login:
                entry.update({key: bot_state.get(key) for key in _ACCOUNT_GUARD_KEYS})
                _peak_saved.update(login=login, peak=bot_state.get("peak_equity"))
            else:  # keep what is on disk for this login
                previous = document.get(str(login)) if isinstance(document.get(str(login)), dict) else {}
                entry.update({key: previous.get(key) for key in _ACCOUNT_GUARD_KEYS if key in previous})
            document[str(login)] = entry
            memory_store.write_json_atomic(config.RISK_STATE_FILE, document)
    except OSError as exc:
        logger.error("[SYSTEM] Could not save the daily risk state: %s", exc)


def _load_account_guard(login: int) -> Dict[str, Any]:
    document = memory_store.read_json_file(config.RISK_STATE_FILE, dict, quarantine_corrupt=False)
    state = document.get(str(login)) if isinstance(document, dict) else None
    return {key: state.get(key) for key in _ACCOUNT_GUARD_KEYS} if isinstance(state, dict) else {}


def _update_total_drawdown(equity: float, login: int) -> bool:
    """
    Drawdown from the account's equity peak (kept on disk across days and restarts). Beyond
    DRAWDOWN_THROTTLE_PERCENT new trades use THROTTLE_RISK_FACTOR x risk; at MAX_TOTAL_DRAWDOWN_PERCENT the
    kill-switch stops new entries (and closes the engine's positions) until it is reset from the dashboard.
    Returns True when the state should be saved.
    """
    save = False
    if bot_state.get("guard_login") != login:
        saved = _load_account_guard(login) if login else {}
        bot_state.update(guard_login=login, peak_equity=float(saved.get("peak_equity") or equity),
                         kill_switch=bool(saved.get("kill_switch")), kill_switch_at=saved.get("kill_switch_at"),
                         kill_switch_flattened=bool(saved.get("kill_switch_flattened")))
        _peak_saved.update(login=login, peak=saved.get("peak_equity"))
        save = not saved.get("peak_equity")
        if bot_state["kill_switch"]:
            logger.warning("[SYSTEM] TOTAL DRAWDOWN KILL-SWITCH still active since %s (peak equity %.2f): no new "
                           "entries until it is reset in Settings", bot_state["kill_switch_at"], bot_state["peak_equity"])
    peak = float(bot_state["peak_equity"] or equity)
    if equity > peak:
        peak = bot_state["peak_equity"] = float(equity)
        save = save or not _peak_saved["peak"] or peak >= float(_peak_saved["peak"]) * 1.001  # not on every tick
    drawdown = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
    bot_state["total_drawdown_pct"] = round(drawdown, 3)
    throttle = config.DRAWDOWN_THROTTLE_PERCENT
    throttled = throttle > 0 and drawdown >= throttle
    if throttled != bot_state["risk_throttled"]:
        logger.warning("[SYSTEM] Risk throttle %s: equity %.2f is %.2f%% below its peak %.2f (throttle at %.2f%%)",
                       f"ON, new trades use {config.THROTTLE_RISK_FACTOR:g}x risk" if throttled else "OFF",
                       equity, drawdown, peak, throttle)
    bot_state["risk_throttled"] = throttled
    limit = config.MAX_TOTAL_DRAWDOWN_PERCENT
    if limit > 0 and drawdown >= limit and not bot_state["kill_switch"]:
        bot_state.update(kill_switch=True, kill_switch_at=data_engine.utc_now_iso(), kill_switch_flattened=False)
        save = True
        logger.warning("[SYSTEM] TOTAL DRAWDOWN KILL-SWITCH TRIPPED: equity %.2f is %.2f%% below its peak %.2f "
                       "(limit %.2f%%). New entries halted until reset in Settings%s.", equity, drawdown, peak, limit,
                       "; the engine's positions will be closed" if config.KILL_SWITCH_CLOSE_POSITIONS else "")
    return save


def _reset_kill_switch() -> Dict[str, Any]:
    """Start a new peak at the current equity and clear the kill-switch (dashboard action)."""
    equity = float(bot_state.get("equity") or 0.0)
    if equity <= 0:
        raise HTTPException(status_code=409, detail="account equity unknown; connect MT5 first")
    was = bot_state.get("kill_switch")
    bot_state.update(peak_equity=equity, kill_switch=False, kill_switch_at=None, kill_switch_flattened=False,
                     total_drawdown_pct=0.0, risk_throttled=False)
    login = int(bot_state.get("guard_login") or 0)
    if login:
        _save_risk_state(login)
    logger.warning("[SYSTEM] Account protection reset from the dashboard: new equity peak %.2f%s", equity,
                   " (kill-switch cleared)" if was else "")
    return {"peak_equity": equity, "kill_switch": False}


def _update_daily_risk(equity: float, login: Optional[int] = None) -> None:
    """
    Daily loss/profit limits (_update_day_limits) and the total-drawdown protection (_update_total_drawdown).
    The guard state is saved after the daily state switched to this login, never before.
    """
    login = int(login or bot_state.get("day_login") or (bot_state.get("account") or {}).get("login") or 0)
    guard_changed = _update_total_drawdown(equity, login)
    _update_day_limits(equity, login)
    if guard_changed and login:
        _save_risk_state(login)


def _update_day_limits(equity: float, login: int) -> None:
    """
    Daily loss/profit limits against the equity at the start of the trading day (17:00 New York).
    The start equity and a tripped breaker are saved to disk, so a restart cannot reset them.
    """
    today = data_engine.trading_day_key()
    if bot_state["day_key"] != today or not bot_state["day_start_equity"] or bot_state.get("day_login") != login:
        saved = _load_risk_state(login) if login else None
        if saved:
            bot_state.update({key: saved.get(key) for key in _RISK_STATE_KEYS}, day_login=login)
            logger.info("[SYSTEM] Daily risk state restored for trading day %s: start equity %.2f%s", today,
                        float(saved["day_start_equity"]), " | CIRCUIT BREAKER ACTIVE" if saved.get("circuit_breaker")
                        else "")
        else:
            bot_state.update(day_key=today, day_login=login, day_start_equity=equity, circuit_breaker=False,
                             profit_target_hit=False, breaker_flattened=False, daily_drawdown_pct=0.0, day_pnl_pct=0.0)
            if login:
                _save_risk_state(login)
            return
    start = float(bot_state["day_start_equity"])
    day_pnl = (equity - start) / start * 100.0 if start > 0 else 0.0
    drawdown = max(0.0, -day_pnl)
    bot_state["daily_drawdown_pct"] = round(drawdown, 3)
    bot_state["day_pnl_pct"] = round(day_pnl, 3)
    changed = False
    limit = config.MAX_DAILY_LOSS_PERCENT
    if limit > 0 and drawdown >= limit and not bot_state["circuit_breaker"]:
        bot_state["circuit_breaker"] = changed = True
        logger.warning("[SYSTEM] DAILY LOSS CIRCUIT BREAKER TRIPPED: equity %.2f is %.2f%% below the trading day's "
                       "start %.2f (limit %.2f%%). New entries halted until 17:00 New York%s.",
                       equity, drawdown, start, limit,
                       "; the engine's positions will be closed" if config.DAILY_LOSS_CLOSE_POSITIONS else "")
    target = config.MAX_DAILY_PROFIT_PERCENT
    if target > 0 and day_pnl >= target and not bot_state["profit_target_hit"]:
        bot_state["profit_target_hit"] = changed = True
        logger.warning("[SYSTEM] DAILY PROFIT TARGET REACHED: equity %.2f is %.2f%% above the trading day's start "
                       "%.2f (target %.2f%%). New entries paused until 17:00 New York.", equity, day_pnl, start, target)
    if changed and login:
        _save_risk_state(login)


def _reevaluate_daily_limits() -> None:
    """Re-check the daily loss/profit limits after they were changed from the dashboard."""
    bot_state["circuit_breaker"] = False
    bot_state["profit_target_hit"] = False
    bot_state["breaker_flattened"] = False
    if bot_state["equity"] and bot_state["day_start_equity"]:
        _update_daily_risk(float(bot_state["equity"]))
    if bot_state.get("day_login"):
        _save_risk_state(int(bot_state["day_login"]))


def _setting_conflicts() -> List[str]:
    """Combinations of settings that quietly stop the engine from trading, in plain words."""
    risk = float(bot_state.get("risk_percent") or config.DEFAULT_RISK_PERCENT)
    notes = []
    cap = config.MAX_CURRENCY_RISK_PERCENT
    if 0 < cap < risk * 2:
        notes.append(f"risk {risk:g}% per trade with a {cap:g}% per-currency cap allows only one open trade per "
                     f"currency direction (e.g. one long-USD trade); raise the cap to {risk * 2:g}% for two")
    limit = config.MAX_DAILY_LOSS_PERCENT
    if limit > 0 and risk >= limit:
        notes.append(f"risk {risk:g}% per trade is at or above the {limit:g}% daily loss limit: the daily risk "
                     f"budget refuses every trade")
    elif limit > 0 and limit / risk < 2:
        notes.append(f"the {limit:g}% daily loss limit is less than two {risk:g}% trades: one loss ends the day")
    last_exit = config.SCALP_SESSION_END_NEW_YORK + config.SCALP_TIME_STOP_MINUTES / 60.0
    if config.STRATEGY_MODE == "SCALP" and not config.TRADE_ALL_HOURS and last_exit > 17:
        notes.append(f"scalps opened until {config.SCALP_SESSION_END_NEW_YORK}:00 New York with a "
                     f"{config.SCALP_TIME_STOP_MINUTES}-minute time stop can still be open at the 17:00 New York "
                     f"rollover, when spreads blow out; end entries by {int(17 - config.SCALP_TIME_STOP_MINUTES / 60)}:00")
    if config.STRATEGY_MODE == "SCALP" and config.LOSS_COOLDOWN_MINUTES > 60:
        notes.append(f"a {config.LOSS_COOLDOWN_MINUTES}-minute loss cooldown is long for scalping; 30 is typical")
    return notes


def _daily_risk_budget() -> Optional[float]:
    """% of equity that may still be put at risk today: loss limit minus today's drawdown (None = unlimited)."""
    if not config.DAILY_LOSS_RISK_BUDGET or config.MAX_DAILY_LOSS_PERCENT <= 0:
        return None
    return max(0.0, config.MAX_DAILY_LOSS_PERCENT - float(bot_state.get("daily_drawdown_pct") or 0.0))


_LEDGER_FIELDS = (
    "id", "timestamp", "symbol", "side", "entry_price", "stop_loss", "take_profit", "volume", "risk_percent",
    "deal", "order", "ticket", "status", "outcome", "realized_pnl", "exit_price", "exit_time", "exit_reason",
    "holding_minutes", "confidence_score", "digits", "stops_source", "sl_atr_multiple", "tp_atr_multiple",
    "atr_reference", "account_mode", "broker", "server", "exit_time_utc", "reconciled_at", "strategy",
    "time_stop_minutes",
)


def _ledger_row(record: Dict[str, Any]) -> Dict[str, Any]:
    row = {key: record.get(key) for key in _LEDGER_FIELDS}
    equity = (record.get("market_context") or {}).get("equity")
    risk = record.get("risk_percent")
    row["risk_amount"] = round(float(equity) * float(risk) / 100.0, 2) if equity and risk else None
    row["slippage_points"] = (record.get("execution") or {}).get("slippage_points")
    row["close_reason"] = (record.get("close_execution") or {}).get("reason")
    return row


def _execution_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """The execution-moment context of one fill, shaped for the dashboard."""
    context = record.get("market_context") or {}
    return {
        **{key: record.get(key) for key in ("id", "symbol", "side", "timestamp", "entry_price", "digits", "deal",
                                            "ticket")},
        "lots": record.get("volume"),
        "execution": record.get("execution") or {},
        "captured_at": context.get("captured_at"),
        "context_source": context.get("context_source"),
        "quote": {key: context.get(key) for key in ("bid", "ask", "spread_points")},
        "volume": context.get("volume") or {},
        "volatility": context.get("volatility") or {},
        "correlated": context.get("correlated_prices") or {},
    }


def _history_and_stats() -> Dict[str, Any]:
    """Dashboard ledger + performance stats, cached on memory.json's mtime."""
    try:
        stat = os.stat(config.MEMORY_FILE)
        stamp = (stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        stamp = None
    if stamp is not None and stamp == _history_cache["stamp"]:
        return _history_cache

    records = memory_store.load_trade_memory()
    ledger = [_ledger_row(record) for record in reversed(records[-LEDGER_LIMIT:])]
    latest_with_context = next((r for r in reversed(records) if (r.get("market_context") or {}).get("volume")), None)
    stats_by_mode = {
        "ALL": _performance(records),
        "DEMO": _performance([r for r in records if r.get("account_mode") == "DEMO"]),
        "LIVE": _performance([r for r in records if r.get("account_mode") == "LIVE"]),
    }
    _history_cache.update(
        records=records,
        stamp=stamp,
        version=f"{stamp[0]}-{stamp[1]}" if stamp else "empty",
        ledger=ledger,
        history=ledger[:60],
        last_execution=_execution_summary(latest_with_context) if latest_with_context else None,
        stats=stats_by_mode["ALL"],
        stats_by_mode=stats_by_mode,
    )
    return _history_cache


def _performance(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    closed = [r for r in records if r.get("status") == "CLOSED"]
    wins = sum(1 for r in closed if r.get("outcome") == "WIN")
    losses = sum(1 for r in closed if r.get("outcome") == "LOSS")
    return {
        "fills": len(records),
        "open": sum(1 for r in records if r.get("status") == "CONFIRMED"),
        "closed": len(closed),
        "wins": wins,
        "losses": losses,
        "breakeven": len(closed) - wins - losses,
        "win_rate": round(wins / len(closed) * 100.0, 1) if closed else None,
        "realized_pnl": round(sum(float(r.get("realized_pnl") or 0.0) for r in closed), 2),
    }


def _rule_for_dashboard(rule: Dict[str, Any]) -> Dict[str, Any]:
    """A rule plus whether it applies to the account in use (and why not)."""
    view = dict(rule)
    if rule.get("conditions"):
        view["conditions_text"] = rule_engine.describe(rule["conditions"])
    if rule.get("status") != "ACTIVE":
        view.update(applies_here=False, applies_note=rule.get("retired_reason") or "")
    elif any(ai_brain.rule_applies(rule, symbol) for symbol in config.SYMBOLS):
        view.update(applies_here=True, applies_note="applies to this account")
    elif str(rule.get("affected_symbol", "")).upper() == "ALL" or any(
            data_engine.symbols_match(rule.get("affected_symbol", ""), s) for s in config.SYMBOLS):
        view.update(applies_here=False, applies_note="learned at another broker (rule sharing: same broker only)")
    else:
        view.update(applies_here=False, applies_note="pair not traded on this account")
    return view


_shadow_cache: Dict[str, Any] = {"stamp": None, "rows": {}}


def _shadow_learning(mode: Optional[str]) -> List[Dict[str, Any]]:
    """Resolved shadow trades for calibration, cached on the shadow file's timestamp."""
    try:
        stamp = os.stat(config.SHADOW_FILE).st_mtime_ns
    except OSError:
        return []
    if stamp != _shadow_cache["stamp"]:
        _shadow_cache.update(stamp=stamp, rows={})
    if mode not in _shadow_cache["rows"]:
        _shadow_cache["rows"][mode] = shadow_store.learning_records(mode, limit=500)
    return _shadow_cache["rows"][mode]


def _refresh_learning_state() -> None:
    document = auditor.load_rules_document()
    shadow_stats = shadow_store.stats_by_blocker()
    bot_state["learned_rules"] = [{**_rule_for_dashboard(rule), "shadow": shadow_stats.get(str(rule.get("id")))}
                                  for rule in document["rules"]]
    bot_state["guard_shadow_stats"] = {key: value for key, value in shadow_stats.items()
                                       if key.startswith("G-") or key == "AI-VETO"}
    bot_state["last_audit_at"] = document.get("last_audit_at")
    bot_state["audit_summary"] = document.get("audit_summary")


def _refresh_account_state() -> None:
    account = data_engine.get_account_snapshot()
    if account is not None:
        bot_state.update(
            equity=account["equity"],
            balance=account["balance"],
            floating_pnl=account["profit"],
            currency=account["currency"],
            mt5_connected=True,
            account={k: account[k] for k in ("login", "server", "company", "name", "leverage", "margin",
                                             "margin_free", "margin_level", "trade_allowed", "account_mode")},
        )
        _update_daily_risk(account["equity"], account["login"])
        try:
            _refresh_symbol_universe()  # notices account switches even while the engine is stopped
        except Exception as exc:
            logger.debug("[SYSTEM] Symbol list refresh failed: %s", exc)
        try:
            bot_state["open_positions"] = execution.get_open_positions()
            bot_state["open_risk"] = execution.open_risk()
        except Exception as exc:
            logger.debug("[SYSTEM] Position refresh failed: %s", exc)
    else:
        bot_state["mt5_connected"] = False
    cache = _history_and_stats()
    bot_state["trade_history"] = cache["history"]
    mode = bot_state.get("account_mode")
    bot_state["stats_scope"] = mode or "ALL"  # headline stats follow the logged-in account type
    bot_state["stats"] = cache["stats_by_mode"].get(mode or "ALL", cache["stats"])
    bot_state["stats_by_mode"] = cache["stats_by_mode"]
    bot_state["fills_version"] = cache["version"]
    bot_state["last_execution"] = cache.get("last_execution")
    calibration.refresh(cache.get("records") or [], mode, _shadow_learning(mode))
    _refresh_learning_state()


def _status_payload() -> Dict[str, Any]:
    payload = dict(bot_state)
    payload["decisions"] = dict(bot_state["decisions"])  # the trading loop mutates this dict
    payload["logs"] = list(LOG_BUFFER.records)[-120:]
    payload["server_time"] = data_engine.utc_now_iso()
    payload["guardrails"] = {
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "effective_threshold": calibration.effective_threshold(),
        "calibrate_threshold": config.CALIBRATE_THRESHOLD,
        "max_currency_risk_percent": config.MAX_CURRENCY_RISK_PERCENT,
        "daily_loss_risk_budget": config.DAILY_LOSS_RISK_BUDGET,
        "daily_loss_close_positions": config.DAILY_LOSS_CLOSE_POSITIONS,
        "risk_budget_left_percent": _daily_risk_budget(),
        "loss_cooldown_minutes": config.LOSS_COOLDOWN_MINUTES,
        "ai_new_bar_only": config.AI_NEW_BAR_ONLY,
        "reversal_extra_confidence": config.REVERSAL_EXTRA_CONFIDENCE,
        "max_entry_drift_atr": config.MAX_ENTRY_DRIFT_ATR,
        "news_guard": config.NEWS_GUARD,
        "rule_ttl_days": config.RULE_TTL_DAYS,
        "rule_total_penalty_cap": config.RULE_TOTAL_PENALTY_CAP,
        "max_risk_percent": config.MAX_RISK_PERCENT,
        "max_open_positions": config.MAX_OPEN_POSITIONS,
        "max_daily_loss_percent": config.MAX_DAILY_LOSS_PERCENT,
        "max_daily_profit_percent": config.MAX_DAILY_PROFIT_PERCENT,
        "max_total_drawdown_percent": config.MAX_TOTAL_DRAWDOWN_PERCENT,
        "drawdown_throttle_percent": config.DRAWDOWN_THROTTLE_PERCENT,
        "throttle_risk_factor": config.THROTTLE_RISK_FACTOR,
        "kill_switch_close_positions": config.KILL_SWITCH_CLOSE_POSITIONS,
        "agent_enabled": config.AGENT_ENABLED,
        "agent_explore": config.AGENT_EXPLORE,
        "agent_shadow_until_learned": config.AGENT_SHADOW_UNTIL_LEARNED,
        "rule_scope": config.RULE_SCOPE,
        "overextension_guard": config.OVEREXTENSION_GUARD,
        "weekend_entry_cutoff_hours": config.WEEKEND_ENTRY_CUTOFF_HOURS,
        "weekend_close": config.WEEKEND_CLOSE,
        "weekend_close_minutes": config.WEEKEND_CLOSE_MINUTES,
        "guard_max_penalty": config.GUARD_MAX_PENALTY,
        "sizing_mode": config.POSITION_SIZING_MODE,
        "fixed_lot": config.FIXED_LOT,
        "sl_atr_multiplier": config.SL_ATR_MULTIPLIER,
        "tp_atr_multiplier": config.TP_ATR_MULTIPLIER,
        "sl_atr_min": config.SL_ATR_MIN,
        "sl_atr_max": config.SL_ATR_MAX,
        "tp_atr_min": config.TP_ATR_MIN,
        "tp_atr_max": config.TP_ATR_MAX,
        "min_reward_risk": config.MIN_REWARD_RISK,
        "max_spread_to_stop": config.MAX_SPREAD_TO_STOP,
        "magic_number": config.MAGIC_NUMBER,
        "audit_interval_hours": config.AUDIT_INTERVAL_HOURS,
        "audit_lookback_trades": config.AUDIT_LOOKBACK_TRADES,
        "min_lot_risk_tolerance": config.MIN_LOT_RISK_TOLERANCE,
    }
    payload["symbols"] = config.SYMBOLS
    payload["ai_configured"] = bool(config.DEEPSEEK_API_KEY)
    payload["model"] = config.DEEPSEEK_MODEL
    payload["fallback_models"] = config.LLM_FALLBACK_MODELS
    payload["fx_week"] = data_engine.fx_week_clock()
    payload["news"] = news.status()
    payload["calibration"] = calibration.report()
    opens = scalper.next_session_open()
    payload["strategy"] = {
        "mode": config.STRATEGY_MODE,
        "session_open": scalper.in_session(),
        "session_note": scalper.session_note(),
        "trade_all_hours": config.TRADE_ALL_HOURS,
        "next_open_utc": opens.isoformat() if opens else None,
        "setting_warnings": _setting_conflicts(),
        "time_stop_minutes": config.SCALP_TIME_STOP_MINUTES,
        "max_trades_per_symbol": config.SCALP_MAX_TRADES_PER_SYMBOL,
        "strict_guard": config.SCALP_STRICT_GUARD,
        "medium_trend": config.SCALP_MEDIUM_TREND,
        "reward_risk": config.SCALP_REWARD_RISK,
        "scalp_enabled": config.SCALP_ENABLED,
        "intraday_enabled": config.INTRADAY_ENABLED,
        "intraday": {"time_stop_minutes": config.INTRADAY_TIME_STOP_MINUTES, "reward_risk": config.INTRADAY_REWARD_RISK,
                     "max_trades_per_symbol": config.INTRADAY_MAX_TRADES_PER_SYMBOL,
                     "risk_percent": _strategy_risk("INTRADAY")},
        "hedging": (bot_state.get("account") or {}).get("hedging"),
    }
    payload["intraday"] = dict(bot_state.get("intraday") or {})
    payload["learning_data"] = _learning_data_status()
    payload["agent"] = _agent_status()
    payload["sessions"] = _session_status()
    return payload


_session_cache: Dict[str, Any] = {"key": None, "at": 0.0, "value": None}
SESSION_CACHE_SECONDS = 30


def _session_status() -> Dict[str, Any]:
    """Open sessions and which pairs are active in them (tick times re-read every 30 seconds)."""
    key = tuple(config.SYMBOLS)
    if key != _session_cache["key"] or time.monotonic() - _session_cache["at"] > SESSION_CACHE_SECONDS:
        ticks = data_engine.last_tick_times(config.SYMBOLS) if bot_state.get("mt5_connected") else {}
        _session_cache.update(key=key, at=time.monotonic(), value=sessions.pair_status(list(config.SYMBOLS), ticks))
    return _session_cache["value"]


_agent_cache: Dict[str, Any] = {"key": None, "value": None}


def _rule_strategies() -> List[str]:
    """The rules strategies that are switched on (each has its own learning agent)."""
    return [s for s, on in (("SCALP", config.SCALP_ENABLED), ("INTRADAY", config.INTRADAY_ENABLED)) if on]


def _agent_status() -> Dict[str, Any]:
    """The learning agent per strategy + a 7-day scorecard, cached until new rewards arrive (or a minute passes)."""
    target = config.AGENT_DIR / "experience.jsonl"
    try:
        stamp = os.stat(target).st_mtime_ns
    except OSError:
        stamp = None
    key = (stamp, config.AGENT_ENABLED, config.AGENT_EXPLORE, config.AGENT_MIN_REWARDS, config.AGENT_SHADOW_UNTIL_LEARNED,
           int(time.time() // 60))
    if key != _agent_cache["key"]:
        strategies = {}
        for strategy in agent.STRATEGIES:
            try:
                card = agent.scorecard(strategy, 7)
                strategies[strategy] = {**agent.status(strategy), "week": card["groups"], "edge_r": card["edge_r"]}
            except Exception as exc:  # a broken experience line must never break the dashboard
                strategies[strategy] = {"strategy": strategy, "phase": "error", "error": str(exc)}
        _agent_cache.update(key=key, value={"enabled": config.AGENT_ENABLED, "explore": config.AGENT_EXPLORE,
                                            "shadow_until_learned": config.AGENT_SHADOW_UNTIL_LEARNED,
                                            "min_rewards": config.AGENT_MIN_REWARDS,
                                            "half_life_days": config.AGENT_HALF_LIFE_DAYS,
                                            "active": _rule_strategies(), "strategies": strategies})
    return _agent_cache["value"]


_learning_cache: Dict[str, Any] = {"key": None, "value": None}


def _learning_data_status() -> Dict[str, Any]:
    """Decision journal, shadow and exploration trades, for the dashboard (cached on file times)."""
    today = config.JOURNAL_DIR / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    stamps = []
    for path in (today, config.SHADOW_FILE, config.EXPLORE_FILE):
        try:
            stamps.append(os.stat(path).st_mtime_ns)
        except OSError:
            stamps.append(None)
    key = (str(today), *stamps)
    if key != _learning_cache["key"]:
        shadows = shadow_store.load_shadows()
        resolved = [s for s in shadows if s.get("status") in ("WIN", "LOSS", "TIMEOUT")]
        explore = shadow_store.load_shadows(config.EXPLORE_FILE)
        _learning_cache.update(key=key, value={
            "journal_today": journal.summary(1),
            "shadow_total": len(shadows), "shadow_resolved": len(resolved),
            "shadow_weight": config.SHADOW_WEIGHT,
            "explore_total": len(explore),
            "explore_open": sum(1 for s in explore if s.get("status") == "OPEN"),
        })
    return _learning_cache["value"]


# -----------------------------------------------------------------------------
# Trading engine
# -----------------------------------------------------------------------------
def _record_decision(symbol: str, market: Dict[str, Any], decision: Dict[str, Any], view: str = "decisions") -> None:
    bot_state.update(
        last_strategy=decision.get("strategy") or "SWING",
        last_symbol=symbol,
        last_signal=decision["signal"],
        last_raw_signal=decision.get("raw_signal", decision["signal"]),
        last_confidence=decision["confidence_score"],
        last_base_confidence=decision.get("base_confidence", decision["confidence_score"]),
        last_logic=decision["logic"],
        last_entry_price=decision.get("entry_reference"),
        last_stop_loss=decision.get("stop_loss"),
        last_take_profit=decision.get("take_profit"),
        last_applied_rules=decision.get("applied_rules", []),
        last_stops_source=decision.get("stops_source"),
        last_sl_atr_multiple=decision.get("sl_atr_multiple"),
        last_tp_atr_multiple=decision.get("tp_atr_multiple"),
        last_sl_distance=decision.get("sl_distance"),
        last_tp_distance=decision.get("tp_distance"),
        last_risk_reward=decision.get("risk_reward"),
        last_atr=decision.get("atr_reference"),
        last_atr_source=decision.get("atr_source"),
        last_stop_notes=decision.get("stop_notes", []),
        last_digits=market.get("digits"),
        last_decision_at=decision.get("timestamp"),
        last_usd_direction=decision.get("usd_direction"),
        last_threshold=decision.get("threshold"),
        last_blocked_by=decision.get("blocked_by") or [],
        last_agent=_agent_summary(decision),
        equity=market.get("equity", bot_state["equity"]),
    )
    bot_state.setdefault(view, {})[symbol] = {
        "signal": decision["signal"],
        "raw_signal": decision.get("raw_signal"),
        "confidence": decision["confidence_score"],
        "penalty": sum(rule.get("points", 0) for rule in decision.get("applied_rules", [])),
        "time": decision.get("timestamp"),
        "error": decision.get("error"),
        "digits": market.get("digits"),
        "strategy": decision.get("strategy") or "SWING",
        "note": (decision.get("setup") or {}).get("reason"),
        "agent": _agent_summary(decision),
    }


def _agent_summary(decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The learning agent's verdict on a decision, without its (large) state snapshot."""
    info = decision.get("agent")
    return {k: v for k, v in info.items() if k != "context"} if info else None


def _timeframe_context(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: data.get(key) for key in (
        "last_bar_time", "close", "ema200", "price_vs_ema", "ema_slope", "ema_distance_atr", "rsi14",
        "atr14", "atr_pct", "atr_ratio", "tick_volume", "volume_ma20", "rel_volume", "structure")}


def _closed_bar_context(market: Dict[str, Any]) -> Dict[str, Any]:
    """Volume/volatility from the decision snapshot; used only if the live capture fails."""
    def pick(data: Dict[str, Any], keys: tuple) -> Dict[str, Any]:
        return {key: data.get(key) for key in keys}

    h1, d1 = market.get("h1_data") or {}, market.get("daily_data") or {}
    volume_keys, volatility_keys = ("tick_volume", "volume_ma20", "rel_volume"), ("atr14", "atr_pct", "atr_ratio")
    return {
        "volume": {"h1": pick(h1, volume_keys), "d1": pick(d1, volume_keys)},
        "volatility": {"h1": pick(h1, volatility_keys), "d1": pick(d1, volatility_keys)},
    }


def _capture_context(symbol: str, market: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Execution-moment market context (blocking). Never raises: a failed capture must not lose the fill."""
    try:
        if market is None:
            market = data_engine.fetch_multi_timeframe_data(symbol)
        return data_engine.capture_execution_context(symbol, config.SYMBOLS, market)
    except Exception as exc:
        logger.warning("[LEARNING] %s execution context capture failed (%s); using decision-time context",
                       symbol, exc)
        return None


def _describe_context(symbol: str, context: Dict[str, Any]) -> str:
    volume, volatility = context.get("volume") or {}, context.get("volatility") or {}
    h1v, m1v = volume.get("h1") or {}, volume.get("m1") or {}
    h1x, m1x = volatility.get("h1") or {}, volatility.get("m1") or {}
    correlated = context.get("correlated_prices") or context.get("correlated_assets") or {}
    strongest = sorted(((s, q.get("corr_h1")) for s, q in correlated.items() if q.get("corr_h1") is not None),
                       key=lambda item: -abs(item[1]))[:2]
    return (f"{symbol} @ {context.get('captured_at')}: H1 rel vol {h1v.get('rel_volume')} "
            f"(forming pace {h1v.get('projected_rel_volume')}) | 5m rel vol {m1v.get('rel_volume_5m')} | "
            f"ATR {h1x.get('atr_pct')}% (ratio {h1x.get('atr_ratio')}) | 1h realized vol "
            f"{m1x.get('realized_vol_1h_pct')}% | {len(correlated)} correlated assets"
            + (f" (strongest {', '.join(f'{s} {c:+.2f}' for s, c in strongest)})" if strongest else ""))


def _build_trade_record(symbol: str, market: Dict[str, Any], decision: Dict[str, Any], trade: Dict[str, Any],
                        risk_percent: float, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Memory record for a fill; market_context describes the exact moment of execution."""
    captured = context or {}
    quote = captured.get("quote") or {}
    fallback = _closed_bar_context(market)
    return {
        "status": "CONFIRMED",
        "timestamp": data_engine.utc_now_iso(),
        "account_login": market.get("account_login"),
        "account_mode": market.get("account_mode"),
        "broker": market.get("broker"),
        "server": market.get("server"),
        "symbol": symbol,
        "side": trade["side"],
        "ticket": trade["position_ticket"],
        "deal": trade["deal"],
        "order": trade["order"],
        "entry_price": trade["price"],
        "stop_loss": trade["stop_loss"],
        "take_profit": trade["take_profit"],
        "volume": trade["volume"],
        "risk_percent": risk_percent,
        "digits": trade["digits"],
        "confidence_score": decision["confidence_score"],
        "base_confidence": decision.get("base_confidence"),
        "threshold": decision.get("threshold"),
        "strategy": decision.get("strategy") or "SWING",
        "agent": {**decision["agent"], "action": "taken"} if decision.get("agent") else None,
        "setup": decision.get("setup"),
        "time_stop_minutes": decision.get("time_stop_minutes"),
        "applied_rules": decision.get("applied_rules", []),
        "stops_source": decision.get("stops_source"),
        "sl_atr_multiple": decision.get("sl_atr_multiple"),
        "tp_atr_multiple": decision.get("tp_atr_multiple"),
        "atr_reference": decision.get("atr_reference"),
        "risk_reward": decision.get("risk_reward"),
        "logic": decision["logic"],
        "execution": trade.get("execution") or {},
        "market_context": {
            "captured_at": captured.get("captured_at") or decision.get("timestamp"),
            "context_source": "EXECUTION" if captured else "DECISION",
            "capture_ms": captured.get("capture_ms"),
            "bid": quote.get("bid", market["bid"]),
            "ask": quote.get("ask", market["ask"]),
            "spread_points": quote.get("spread_points", market["spread"]),
            "equity": trade.get("equity_at_entry", market["equity"]),
            "balance": market["balance"],
            "digits": market["digits"],
            "day_change_pct": market.get("day_change_pct"),
            "volume": captured.get("volume") or fallback["volume"],
            "volatility": captured.get("volatility") or fallback["volatility"],
            "correlated_prices": captured.get("correlated_assets") or market.get("correlated_prices", {}),
            "usd_direction": decision.get("usd_direction") or ai_brain.usd_direction(market),
            "h1": _timeframe_context(market["h1_data"]),
            "d1": _timeframe_context(market["daily_data"]),
            "decision_quote": {"at": decision.get("timestamp"), "bid": market["bid"], "ask": market["ask"],
                               "spread_points": market["spread"]},
        },
    }


def _record_close_execution(result: Dict[str, Any], reason: str, market: Optional[Dict[str, Any]] = None) -> None:
    """Attach a close fill and its execution-moment context to the trade that opened it (blocking)."""
    context = _capture_context(result["symbol"], market)
    block = {
        "reason": reason,
        "deal": result.get("deal"),
        "order": result.get("order"),
        "price": result.get("price"),
        "volume": result.get("volume"),
        "profit_estimate": result.get("profit_estimate"),
        **(result.get("execution") or {}),
        "market_context": None if context is None else {
            "captured_at": context["captured_at"],
            "quote": context["quote"],
            "volume": context["volume"],
            "volatility": context["volatility"],
            "correlated_prices": context["correlated_assets"],
        },
    }
    try:
        attached = memory_store.attach_close_execution(result["ticket"], block)
    except OSError as exc:
        logger.error("[LEARNING] Could not record %s close of ticket %s: %s", reason, result["ticket"], exc)
        return
    if attached:
        logger.info("[LEARNING] %s close of ticket %s recorded with execution context%s", reason, result["ticket"],
                    f": {_describe_context(result['symbol'], block['market_context'])}" if context else "")
    else:
        logger.info("[LEARNING] Ticket %s was not opened by this engine; its %s close is not in memory",
                    result["ticket"], reason)


def _weekend_entry_block(symbol: str, path: Optional[str] = None) -> Optional[str]:
    """Why new entries are paused for the weekend, or None. Crypto trades through the weekend."""
    cutoff_hours = config.WEEKEND_ENTRY_CUTOFF_HOURS
    if cutoff_hours <= 0 or data_engine.trades_weekends(symbol, path):
        return None
    clock = data_engine.fx_week_clock()
    if clock["closed"]:
        return "forex is closed for the weekend"
    if clock["minutes_to_close"] <= cutoff_hours * 60:
        minutes = int(clock["minutes_to_close"])
        return (f"new entries paused {cutoff_hours:g}h before the Friday close "
                f"({minutes // 60}h {minutes % 60:02d}m left)")
    return None


def _weekend_close_positions() -> int:
    """Close the engine's own non-crypto positions shortly before the Friday close (blocking)."""
    if not config.WEEKEND_CLOSE:
        return 0
    clock = data_engine.fx_week_clock()
    if clock["closed"] or clock["minutes_to_close"] > config.WEEKEND_CLOSE_MINUTES:
        return 0
    closed = 0
    for position in execution.get_open_positions():
        if not position["managed"] or data_engine.trades_weekends(position["symbol"]):
            continue
        logger.info("[TRADE] Weekend close: %s %s ticket %s (%.2f lots, floating %.2f), %d min before the close",
                    position["symbol"], position["side"], position["ticket"], position["volume"], position["profit"],
                    int(clock["minutes_to_close"]))
        result = execution.close_position(position["ticket"])
        if result.get("success"):
            _record_close_execution(result, "WEEKEND_CLOSE")
            closed += 1
        else:
            logger.error("[TRADE] Weekend close of ticket %s failed: %s", position["ticket"], result.get("error"))
    return closed


def _flatten_managed(label: str, reason: str, flag: str) -> int:
    """Close the engine's own positions after a protection tripped (blocking). Manual trades stay."""
    positions = [p for p in execution.get_open_positions() if p["managed"]]
    closed = failed = 0
    for position in positions:
        logger.warning("[SYSTEM] %s: closing %s %s ticket %s (floating %.2f)", label,
                       position["symbol"], position["side"], position["ticket"], position["profit"])
        result = execution.close_position(position["ticket"])
        if result.get("success"):
            _record_close_execution(result, reason)
            closed += 1
        elif "not found" in str(result.get("error", "")):
            continue  # already closed by its stop
        else:
            failed += 1
            logger.error("[SYSTEM] %s close of ticket %s failed: %s", label, position["ticket"], result.get("error"))
    if not failed:
        bot_state[flag] = True
        login = bot_state.get("day_login") or bot_state.get("guard_login")
        if login:
            _save_risk_state(int(login))
    return closed


def _flatten_for_daily_loss() -> int:
    return _flatten_managed("Daily loss limit", "DAILY_LOSS_LIMIT", "breaker_flattened")


def _watchdog_tick(check_weekend: bool) -> None:
    """Equity-based protections between (and during) scans; the AI can take minutes per symbol (blocking)."""
    account = data_engine.get_account_snapshot()
    if account is None:
        return
    bot_state.update(equity=account["equity"], balance=account["balance"], floating_pnl=account["profit"])
    _update_daily_risk(account["equity"], account["login"])
    if not bot_state["is_running"]:
        return
    if bot_state["circuit_breaker"] and config.DAILY_LOSS_CLOSE_POSITIONS and not bot_state["breaker_flattened"]:
        _flatten_for_daily_loss()
    if bot_state["kill_switch"] and config.KILL_SWITCH_CLOSE_POSITIONS and not bot_state["kill_switch_flattened"]:
        _flatten_managed("Total drawdown kill-switch", "KILL_SWITCH", "kill_switch_flattened")
    _scalp_time_stops()
    if check_weekend:
        _weekend_close_positions()


async def risk_watchdog() -> None:
    last_weekend_check = 0.0
    while True:
        try:
            if bot_state["mt5_connected"]:
                check_weekend = config.WEEKEND_CLOSE and time.monotonic() - last_weekend_check >= 60
                if check_weekend:
                    last_weekend_check = time.monotonic()
                await asyncio.to_thread(_watchdog_tick, check_weekend)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("[SYSTEM] Risk watchdog tick failed: %s", exc)
        await asyncio.sleep(config.RISK_WATCHDOG_SECONDS)


# symbol -> the H1 bar and price the AI last judged, so the same closed bar is not re-asked every scan
_last_evaluation: Dict[str, Dict[str, Any]] = {}


def _evaluation_unchanged(symbol: str, market: Dict[str, Any]) -> bool:
    """True when the AI already judged this closed H1 bar and price has not moved enough to change the picture."""
    if not config.AI_NEW_BAR_ONLY:
        return False
    last = _last_evaluation.get(symbol)
    h1 = market.get("h1_data") or {}
    if (not last or last["bar"] != h1.get("last_bar_time") or last["login"] != market.get("account_login")
            or not h1.get("atr14")):
        return False
    return abs(float(market["mid"]) - last["mid"]) < config.AI_REEVALUATE_ATR_MOVE * float(h1["atr14"])


def _remember_evaluation(symbol: str, market: Dict[str, Any]) -> None:
    _last_evaluation[symbol] = {"bar": (market.get("h1_data") or {}).get("last_bar_time"),
                                "mid": float(market["mid"]), "login": market.get("account_login")}


def _news_entry_block(symbol: str) -> Optional[str]:
    event = news.blocking_event(symbol)
    if event is None:
        return None
    return (f"high-impact news {event['currency']} '{event['title']}' at {event['time']} UTC "
            f"(no entries {config.NEWS_BLOCK_BEFORE_MINUTES} min before to {config.NEWS_BLOCK_AFTER_MINUTES} min after)")


def _parse_utc(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _loss_cooldown_block(symbol: str, side: str, strategy: Optional[str] = None) -> Optional[str]:
    """No new trade in the same direction right after a losing one on the same symbol (of the same strategy)."""
    minutes = config.LOSS_COOLDOWN_MINUTES
    if minutes <= 0:
        return None
    now = datetime.now(timezone.utc)
    for row in _history_and_stats()["ledger"]:
        if row.get("status") != "CLOSED" or row.get("outcome") != "LOSS" or row.get("side") != side:
            continue
        if strategy in RULE_STRATEGIES and (row.get("strategy") or "SWING") != strategy:
            continue
        if not data_engine.symbols_match(str(row.get("symbol") or ""), symbol):
            continue
        closed_at = _parse_utc(row.get("exit_time_utc") or row.get("reconciled_at"))
        if closed_at is None:
            continue
        elapsed = (now - closed_at).total_seconds() / 60.0
        if 0 <= elapsed < minutes:
            return (f"loss cooldown: a {side} {row.get('symbol')} lost {row.get('realized_pnl')} {int(elapsed)} min ago "
                    f"(no same-direction re-entry for {minutes} min)")
    return None


async def _prepare_market(symbol: str) -> Optional[Dict[str, Any]]:
    """Live snapshot for a symbol, or None when it must not be traded right now (logged)."""
    try:
        market = await asyncio.to_thread(data_engine.fetch_multi_timeframe_data, symbol)
    except data_engine.DataEngineError as exc:
        logger.warning("[SYSTEM] %s skipped: %s", symbol, exc)
        return None
    if not market["tradeable"]:
        logger.info("[SYSTEM] %s skipped: trading disabled/close-only at the broker", symbol)
        return None
    if market["market_idle"]:
        logger.info("[SYSTEM] %s skipped: no new ticks for %ds (market closed)", symbol, config.MARKET_IDLE_SECONDS)
        return None
    weekend_block = _weekend_entry_block(symbol, market.get("path"))
    if weekend_block:
        logger.info("[SYSTEM] %s skipped: %s", symbol, weekend_block)
        return None
    news_block = _news_entry_block(symbol)
    if news_block:
        logger.info("[SYSTEM] %s skipped: %s", symbol, news_block)
        return None
    return market


def _log_decision(symbol: str, decision: Dict[str, Any]) -> None:
    penalty = sum(rule.get("points", 0) for rule in decision.get("applied_rules", []))
    geometry = ""
    if decision.get("sl_atr_multiple"):
        geometry = (f" | SL {decision['stop_loss']} ({decision['sl_atr_multiple']}x {decision.get('atr_source') or 'ATR'}) "
                    f"TP {decision['take_profit']} ({decision['tp_atr_multiple']}x) R:R {decision['risk_reward']}")
    logger.info("[AI] %s%s -> %s | confidence %d%s%s | %s", symbol,
                {"SCALP": " scalp", "INTRADAY": " intraday"}.get(decision.get("strategy"), ""), decision["signal"],
                decision["confidence_score"], f" (penalties -{penalty})" if penalty else "",
                geometry, decision["logic"][:240])


async def process_symbol(symbol: str) -> str:
    """
    One symbol's cycle. Returns 'unchanged' (nothing new to judge), 'off_session', 'no_setup',
    'skipped' or 'evaluated'.
    """
    if config.STRATEGY_MODE == "SCALP":
        outcomes = []
        if config.SCALP_ENABLED:
            outcomes.append(await process_scalp(symbol))
        if config.INTRADAY_ENABLED:
            outcomes.append(await process_intraday(symbol))
        for outcome in ("evaluated", "skipped", "no_setup", "error", "unchanged", "off_session"):
            if outcome in outcomes:
                return outcome  # the most informative result, for the scan summary
        return "unchanged"
    market = await _prepare_market(symbol)
    if market is None:
        return "skipped"
    if _evaluation_unchanged(symbol, market):
        return "unchanged"

    market["correlated_prices"] = await asyncio.to_thread(
        data_engine.fetch_correlated_asset_prices, symbol,
        data_engine.related_symbols(symbol, config.SYMBOLS), True)
    decision = await asyncio.to_thread(ai_brain.get_ai_decision, market, symbol)
    _record_decision(symbol, market, decision)
    if not decision.get("error"):
        _remember_evaluation(symbol, market)
    _log_decision(symbol, decision)
    entry = {"kind": "swing_eval", "symbol": symbol, "account_mode": market.get("account_mode"),
             "broker": market.get("broker"), "stage": "AI_DECISION", "spread_points": market.get("spread"),
             "bid": market.get("bid"), "ask": market.get("ask"),
             "features": {"h1": {k: (market.get("h1_data") or {}).get(k) for k in _SWING_FEATURES},
                          "d1": {k: (market.get("daily_data") or {}).get(k) for k in _SWING_FEATURES}},
             "ai": _journal_ai(decision)}
    if decision.get("blocked_by"):
        with contextlib.suppress(Exception):
            shadow = await asyncio.to_thread(shadow_store.record_blocked, symbol, market, decision)
            entry["shadow_id"] = (shadow or {}).get("id")
    outcome = await _act_on_decision(symbol, market, decision, entry)
    await asyncio.to_thread(journal.record, entry)
    return outcome


_SWING_FEATURES = ("ema_distance_atr", "rsi14", "atr_pct", "atr_ratio", "rel_volume", "price_vs_ema", "ema_slope")


# symbol -> server time of the last M5 bar the scalper evaluated (each closed bar is judged once)
_last_scalp_bar: Dict[str, int] = {}


RULE_STRATEGIES = ("SCALP", "INTRADAY")
ORDER_TAGS = {"SCALP": "-S", "INTRADAY": "-I"}


def _strategy_risk(strategy: str) -> float:
    """Risk per trade (% of equity) for a strategy; intraday falls back to the scalp risk when not set."""
    base = float(bot_state.get("risk_percent") or config.DEFAULT_RISK_PERCENT)
    if strategy == "INTRADAY" and config.INTRADAY_RISK_PERCENT > 0:
        return float(config.INTRADAY_RISK_PERCENT)
    return base


def _position_strategy(position: Dict[str, Any]) -> str:
    """SCALP / INTRADAY / SWING for the engine's positions (ledger first, then the order comment), MANUAL otherwise."""
    if not position.get("managed"):
        return "MANUAL"
    ticket = position.get("ticket")
    for row in _history_and_stats()["ledger"]:
        if row.get("ticket") == ticket and row.get("strategy"):
            return str(row["strategy"])
    comment = str(position.get("comment") or "")
    for strategy, tag in ORDER_TAGS.items():
        if comment.endswith(tag):
            return strategy
    return "SWING"


def _scalp_count_today(symbol: str, strategy: str = "SCALP") -> int:
    today = data_engine.trading_day_key()
    count = 0
    for row in _history_and_stats()["ledger"]:
        if row.get("strategy") != strategy or not data_engine.symbols_match(str(row.get("symbol") or ""), symbol):
            continue
        opened = _parse_utc(row.get("timestamp"))
        if opened is not None and data_engine.trading_day_key(opened) == today:
            count += 1
    return count


def _scalp_limit_block(symbol: str, strategy: str = "SCALP") -> Optional[str]:
    limit = config.INTRADAY_MAX_TRADES_PER_SYMBOL if strategy == "INTRADAY" else config.SCALP_MAX_TRADES_PER_SYMBOL
    count = _scalp_count_today(symbol, strategy)
    if count >= limit:
        return f"{count} {strategy.lower()} trades today (max {limit} per symbol per trading day)"
    return None


def _evaluate_scalp(symbol: str, spread_price: float) -> Dict[str, Any]:
    """Fetch M5/M15/H1/D1 bars and run the scalper rules on the latest closed M5 bar (blocking)."""
    prepared = scalper.prepare(scalper.fetch_live_frames(symbol))
    result = scalper.evaluate(prepared, spread_price=spread_price)
    result["recent"] = scalper.recent_m5(prepared)
    side = ((result.get("setup") or result.get("blocked_setup") or {}).get("side")
            or {"UP": "BUY", "DOWN": "SELL"}.get(result.get("trend") or ""))
    result["features"] = scalper.features(prepared, side=side, spread_price=spread_price)
    if config.AGENT_EXPLORE and result["setup"] is None and not result.get("blocked_setup"):
        near = scalper.evaluate(prepared, spread_price=spread_price, relaxed=True)
        if near.get("setup"):
            result["explore"] = {"setup": {**near["setup"], "strategy": "SCALP"},
                                 "features": scalper.features(prepared, side=near["setup"]["side"],
                                                              spread_price=spread_price)}
    return result


def _shadow_skipped_setup(symbol: str, market: Dict[str, Any], result: Dict[str, Any],
                          blocked: Optional[Dict[str, Any]] = None,
                          blocked_by: Optional[List[str]] = None,
                          agent_info: Optional[Dict[str, Any]] = None,
                          features: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """A setup not sent to the AI (strict guard, limits, exploration) is followed on price data like any skipped trade."""
    blocked = blocked or result["blocked_setup"]
    side = blocked["side"]
    digits = int(market.get("digits") or 5)
    tick = float(market.get("tick_size") or market.get("point") or 10 ** -digits)
    entry = float(market["ask"] if side == "BUY" else market["bid"])
    direction = 1.0 if side == "BUY" else -1.0
    decision = {
        "raw_signal": side, "blocked_by": blocked_by or [g["id"] for g in result.get("guards") or []],
        "features": features if features is not None else result.get("features"),
        "stop_loss": data_engine.round_to_tick(entry - direction * blocked["sl_distance"], tick, digits),
        "take_profit": data_engine.round_to_tick(entry + direction * blocked["tp_distance"], tick, digits),
        "entry_reference": round(entry, digits), "risk_reward": blocked["risk_reward"],
        "strategy": blocked.get("strategy") or "SCALP",
        "time_stop_minutes": (config.INTRADAY_TIME_STOP_MINUTES if blocked.get("strategy") == "INTRADAY"
                              else config.SCALP_TIME_STOP_MINUTES),
        "base_confidence": None, "confidence_score": None, "agent": agent_info,
    }
    with contextlib.suppress(Exception):
        return shadow_store.record_blocked(symbol, market, decision)
    return None


def _explore(symbol: str, market: Dict[str, Any], result: Dict[str, Any], strategy: str) -> Optional[str]:
    """Follow this bar's near-miss setup as a virtual exploration trade (never a real order, no AI call)."""
    near = result.get("explore")
    if not near:
        return None
    setup = near["setup"]
    spread = float(market.get("spread_price") or 0.0)
    if spread > config.MAX_SPREAD_TO_STOP * float(setup["sl_distance"]):
        return None  # a real trade would be refused for its cost: nothing worth learning from
    ctx = agent.make_context(strategy, near["features"], setup["side"], setup, explore=True)
    shadow = _shadow_skipped_setup(symbol, market, result, setup, ["EXPLORE"],
                                   {"context": ctx, "action": "explore", "decided_by": "explore"}, near["features"])
    return (shadow or {}).get("id")


def _apply_agent(decision: Dict[str, Any], verdict: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    The learning agent's TAKE / SKIP becomes the decision (the AI's answer is one of its inputs). While it
    warms up, or when switched off, the AI's confirm/veto stands. Hard limits (spread too wide for the stop,
    no executable levels) can never be overruled. Every setup not taken ends up with a blocked_by reason,
    so it is followed as a shadow trade and its reward is learned too.
    """
    side = decision.get("raw_signal")
    ai_take = decision["signal"] != "HOLD"
    blocked = list(decision.get("blocked_by") or [])
    take = verdict.get("take")
    decided_by = "agent" if take is not None else "ai"
    if decision.get("cost_block") or not decision.get("stop_loss"):
        take, decided_by = False, "costs"
    elif take is None:
        take = ai_take
    decision["agent"] = {**{k: verdict.get(k) for k in ("phase", "expected_r", "p_positive", "sampled_r",
                                                          "uncertainty_r", "rewards", "note")},
                         "context": ctx, "decided_by": decided_by, "action": "taken" if take else "skipped"}
    if decided_by == "costs":
        return decision
    if decided_by == "agent":
        if take and not ai_take:
            why = ("the AI's veto" if "AI-VETO" in blocked else "a missing AI answer" if decision.get("error")
                   else "the confidence threshold")
            decision.update(signal=side, blocked_by=[], overruled=blocked or ["THRESHOLD"])
            decision["logic"] = f"[AGENT TAKE, overrules {why}: {verdict.get('note')}] {decision['logic']}"
        elif take:
            decision["logic"] = f"[AGENT TAKE: {verdict.get('note')}] {decision['logic']}"
        else:
            decision.update(signal="HOLD", blocked_by=blocked + ["AGENT-SKIP"])
            decision["logic"] = f"[AGENT SKIP: {verdict.get('note')}] {decision['logic']}"
    if decision["signal"] == "HOLD" and not decision.get("blocked_by"):
        decision["blocked_by"] = ["THRESHOLD"]  # AI agreed below the threshold: still followed as a shadow
    if (decision["signal"] != "HOLD" and verdict.get("phase") == "warmup" and config.AGENT_ENABLED
            and config.AGENT_SHADOW_UNTIL_LEARNED):
        # Still learning: the trade the AI wants is followed as a shadow trade; demo orders start once learned.
        decision.update(signal="HOLD", blocked_by=["LEARNING"])
        decision["agent"]["action"] = "blocked"
        decision["logic"] = f"[LEARNING: shadow trade only, {verdict.get('note')}] {decision['logic']}"
    return decision


def _decision_stage(decision: Dict[str, Any]) -> str:
    agent_info = decision.get("agent") or {}
    blocked = decision.get("blocked_by") or []
    if decision["signal"] != "HOLD":
        return "AGENT_TAKE" if agent_info.get("decided_by") == "agent" else "AI_CONFIRMED"
    if "AGENT-SKIP" in blocked:
        return "AGENT_SKIP"
    if "LEARNING" in blocked:
        return "LEARNING"
    if "AI-VETO" in blocked:
        return "AI_VETO"
    return "AI_ERROR" if decision.get("error") else "BLOCKED"


async def _handle_setup(strategy: str, symbol: str, market: Dict[str, Any], result: Dict[str, Any],
                        setup: Dict[str, Any], entry: Dict[str, Any], set_view) -> str:
    """A rules setup (either strategy): limits, the AI's review, the learning agent's decision, the order."""
    kind = "intraday" if strategy == "INTRADAY" else "scalp"
    side = setup["side"]
    features = result.get("features")
    # Checks that would make the AI's answer irrelevant come first (no API call wasted).
    blocked = _scalp_limit_block(symbol, strategy) or _loss_cooldown_block(symbol, side, strategy)
    if blocked:
        logger.info("[SYSTEM] %s %s %s setup skipped: %s", symbol, side, kind, blocked)
        ctx = agent.make_context(strategy, features, side, setup)
        shadow = await asyncio.to_thread(_shadow_skipped_setup, symbol, market, result, setup, ["LIMIT"],
                                         {"context": ctx, "action": "blocked", "decided_by": "limit"})
        entry.update(action=f"skipped: {blocked}", shadow_id=(shadow or {}).get("id"))
        await asyncio.to_thread(journal.record, entry)
        set_view("SKIPPED", blocked, None)
        return "skipped"
    logger.info("[SYSTEM] %s %s setup found: %s", symbol, kind, setup["reason"])
    market["correlated_prices"] = await asyncio.to_thread(
        data_engine.fetch_correlated_asset_prices, symbol,
        data_engine.related_symbols(symbol, config.SYMBOLS), True)
    decision = await asyncio.to_thread(ai_brain.get_scalp_decision, market, symbol, setup, result["recent"])
    decision["features"] = features
    ctx = agent.make_context(strategy, features, side, setup, decision)
    try:
        verdict = await asyncio.to_thread(agent.decide, strategy, ctx)
    except Exception as exc:  # the agent must never stop trading: fall back to the AI's answer
        logger.warning("[AGENT] %s decision failed (%s); the AI decides", kind, exc)
        verdict = {"take": None, "phase": "error", "note": f"agent error: {exc}"}
    _apply_agent(decision, verdict, ctx)
    _record_decision(symbol, market, decision, view="intraday_decisions" if strategy == "INTRADAY" else "decisions")
    set_view(_decision_stage(decision), decision.get("logic"), decision)
    _log_decision(symbol, decision)
    entry["ai"] = _journal_ai(decision)
    entry["agent"] = _agent_summary(decision)
    if decision.get("blocked_by"):
        with contextlib.suppress(Exception):
            shadow = await asyncio.to_thread(shadow_store.record_blocked, symbol, market, decision)
            entry["shadow_id"] = (shadow or {}).get("id")
    outcome = await _act_on_decision(symbol, market, decision, entry)
    action = str(entry.get("action") or "")
    if decision["signal"] != "HOLD" and action.startswith(("blocked", "rejected")):
        # TAKE, but a safety limit stopped the order: follow it anyway so its reward is learned.
        stopped = {**decision, "blocked_by": ["SAFETY"], "agent": {**decision["agent"], "action": "blocked"}}
        with contextlib.suppress(Exception):
            shadow = await asyncio.to_thread(shadow_store.record_blocked, symbol, market, stopped)
            entry["shadow_id"] = (shadow or {}).get("id")
    await asyncio.to_thread(journal.record, entry)
    return outcome


def _journal_base(symbol: str, market: Dict[str, Any], result: Dict[str, Any], bar: int) -> Dict[str, Any]:
    """The part of a journal line every scalp evaluation shares."""
    setup = result.get("setup") or result.get("blocked_setup") or {}
    return {
        "kind": "scalp_eval", "strategy": "SCALP", "symbol": symbol, "bar_time": bar,
        "account_mode": market.get("account_mode"),
        "broker": market.get("broker"), "stage": result.get("stage"), "reason": result.get("reason"),
        "trend": result.get("trend"), "rsi": result.get("rsi"), "spread_points": market.get("spread"),
        "bid": market.get("bid"), "ask": market.get("ask"), "features": result.get("features"),
        "setup": {k: setup.get(k) for k in ("side", "sl_distance", "tp_distance", "risk_reward", "sl_atr", "room_r",
                                               "m5_rsi_extreme")} if setup else None,
        "guards": [g["id"] for g in result.get("guards") or []],
        "settings": {"strict_guard": config.SCALP_STRICT_GUARD, "medium_trend": config.SCALP_MEDIUM_TREND,
                     "threshold": calibration.effective_threshold(), "risk_percent": bot_state.get("risk_percent")},
    }


async def process_scalp(symbol: str) -> str:
    """Scalp strategy: rules find an M5 pullback in the D1 trend; the AI reviews it; the learning agent decides."""
    if not scalper.in_session():
        return "off_session"
    bar = await asyncio.to_thread(data_engine.last_closed_bar_time, symbol, mt5.TIMEFRAME_M5)
    if bar is None:
        return "skipped"
    if _last_scalp_bar.get(symbol) == bar:
        return "unchanged"
    market = await _prepare_market(symbol)
    if market is None:
        return "skipped"
    try:
        result = await asyncio.to_thread(_evaluate_scalp, symbol, float(market.get("spread_price") or 0.0))
    except data_engine.DataEngineError as exc:
        logger.warning("[SYSTEM] %s scalp data unavailable: %s", symbol, exc)
        return "skipped"
    _last_scalp_bar[symbol] = bar  # judged once per closed M5 bar, whatever the outcome
    bot_state["scalp_last_check_at"] = data_engine.utc_now_iso()
    scan_view = {"trend": result.get("trend"), "stage": result.get("stage"), "rsi": result.get("rsi"),
                 "checked_at": data_engine.utc_now_iso()}
    setup = result["setup"]
    entry = _journal_base(symbol, market, result, bar)
    if setup is None and result.get("blocked_setup"):
        stretched = result["blocked_setup"]
        ctx = agent.make_context("SCALP", result.get("features"), stretched["side"], stretched)
        shadow = _shadow_skipped_setup(symbol, market, result,
                                       agent_info={"context": ctx, "action": "skipped", "decided_by": "guard"})
        entry.update(action="skipped: stretched", shadow_id=(shadow or {}).get("id"))
    if setup is None:
        explored = await asyncio.to_thread(_explore, symbol, market, result, "SCALP")
        if explored:
            entry["explore_id"] = explored
        entry.setdefault("action", "no setup")
        await asyncio.to_thread(journal.record, entry)
        bot_state["decisions"][symbol] = {
            "signal": "HOLD", "raw_signal": "HOLD", "confidence": 0, "penalty": 0, "time": data_engine.utc_now_iso(),
            "error": None, "digits": market.get("digits"), "note": result["reason"], "strategy": "SCALP", **scan_view}
        bot_state.setdefault("scalp_reasons", {})[symbol] = result["reason"]
        return "no_setup"

    def set_view(stage: str, note: Optional[str], decision: Optional[Dict[str, Any]]) -> None:
        if decision is None:
            bot_state["decisions"][symbol] = {
                "signal": "HOLD", "raw_signal": setup["side"], "confidence": 0, "penalty": 0,
                "time": data_engine.utc_now_iso(), "error": None, "digits": market.get("digits"), "note": note,
                "strategy": "SCALP", **scan_view, "stage": stage}
        else:
            bot_state["decisions"][symbol].update(scan_view, stage=stage)

    return await _handle_setup("SCALP", symbol, market, result, setup, entry, set_view)


# symbol -> server time of the last M15 bar the intraday strategy evaluated
_last_intraday_bar: Dict[str, int] = {}


def _evaluate_intraday(symbol: str, spread_price: float) -> Dict[str, Any]:
    """Fetch the bars and run the intraday rules on the latest closed M15 bar (blocking)."""
    prepared = intraday.prepare(scalper.prepare(scalper.fetch_live_frames(symbol)))
    result = intraday.evaluate(prepared, spread_price=spread_price)
    result["recent"] = scalper.recent_m5(prepared)
    side = (result.get("setup") or {}).get("side")
    result["features"] = scalper.features(prepared, side=side, spread_price=spread_price)
    if config.AGENT_EXPLORE and result["setup"] is None and result.get("stage") not in ("OFF_SESSION", "WARMUP"):
        near = intraday.evaluate(prepared, spread_price=spread_price, relaxed=True)
        if near.get("setup"):
            result["explore"] = {"setup": near["setup"],
                                 "features": scalper.features(prepared, side=near["setup"]["side"],
                                                              spread_price=spread_price)}
    return result


def _intraday_view(result: Dict[str, Any], stage: Optional[str] = None, note: Optional[str] = None,
                   decision: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    levels = {name: {"level": st.get("level"), "side": st.get("side"), "state": st.get("state")}
              for name, st in (result.get("levels") or {}).items()}
    return {"stage": stage or result.get("stage"), "note": note or result.get("reason"), "levels": levels,
            "checked_at": data_engine.utc_now_iso(), "strategy": "INTRADAY",
            "agent": _agent_summary(decision) if decision else None}


async def process_intraday(symbol: str) -> str:
    """Intraday strategy: M15 break and retest of key levels; the AI reviews it; the learning agent decides."""
    if not scalper.in_session():
        return "off_session"
    bar = await asyncio.to_thread(data_engine.last_closed_bar_time, symbol, mt5.TIMEFRAME_M15)
    if bar is None:
        return "skipped"
    if _last_intraday_bar.get(symbol) == bar:
        return "unchanged"
    market = await _prepare_market(symbol)
    if market is None:
        return "skipped"
    try:
        result = await asyncio.to_thread(_evaluate_intraday, symbol, float(market.get("spread_price") or 0.0))
    except data_engine.DataEngineError as exc:
        logger.warning("[SYSTEM] %s intraday data unavailable: %s", symbol, exc)
        return "skipped"
    _last_intraday_bar[symbol] = bar  # judged once per closed M15 bar
    views = bot_state.setdefault("intraday", {})
    setup = result["setup"]
    entry = {"kind": "intraday_eval", "strategy": "INTRADAY", "symbol": symbol, "bar_time": bar,
             "account_mode": market.get("account_mode"), "broker": market.get("broker"), "stage": result.get("stage"),
             "reason": result.get("reason"), "spread_points": market.get("spread"), "bid": market.get("bid"),
             "ask": market.get("ask"), "features": result.get("features"),
             "levels": {k: {"level": v.get("level"), "state": v.get("state")} for k, v in (result.get("levels") or {}).items()},
             "setup": {k: setup.get(k) for k in ("side", "level_name", "level", "sl_distance", "tp_distance", "sl_atr",
                                                 "risk_reward", "bars_since_break")} if setup else None,
             "settings": {"threshold": calibration.effective_threshold(), "risk_percent": _strategy_risk("INTRADAY")}}
    if setup is None:
        explored = await asyncio.to_thread(_explore, symbol, market, result, "INTRADAY")
        if explored:
            entry["explore_id"] = explored
        entry["action"] = "no setup"
        await asyncio.to_thread(journal.record, entry)
        views[symbol] = _intraday_view(result)
        return "no_setup"

    def set_view(stage: str, note: Optional[str], decision: Optional[Dict[str, Any]]) -> None:
        views[symbol] = _intraday_view(result, stage, note, decision)

    return await _handle_setup("INTRADAY", symbol, market, result, setup, entry, set_view)


def _journal_ai(decision: Dict[str, Any]) -> Dict[str, Any]:
    return {"signal": decision.get("signal"), "raw_signal": decision.get("raw_signal"),
            "base_confidence": decision.get("base_confidence"), "confidence": decision.get("confidence_score"),
            "threshold": decision.get("threshold"), "blocked_by": decision.get("blocked_by") or [],
            "penalties": [{"id": r.get("id"), "points": r.get("points")} for r in decision.get("applied_rules") or []],
            "logic": (decision.get("logic") or "")[:400], "latency_ms": decision.get("latency_ms"),
            "error": decision.get("error"), "model": config.DEEPSEEK_MODEL,
            "overruled": decision.get("overruled")}


def _scalp_time_stops() -> int:
    """Close the engine's scalp and intraday trades once open longer than their strategy's time stop (blocking)."""
    defaults = {"SCALP": config.SCALP_TIME_STOP_MINUTES, "INTRADAY": config.INTRADAY_TIME_STOP_MINUTES}
    limits = {int(row["ticket"]): int(row.get("time_stop_minutes") or defaults[row["strategy"]])
              for row in _history_and_stats()["ledger"]
              if row.get("strategy") in defaults and row.get("status") == "CONFIRMED" and row.get("ticket")}
    if not limits:
        return 0
    closed = 0
    for position in execution.get_open_positions():
        if not position["managed"] or int(position["ticket"]) not in limits:
            continue
        limit = limits[int(position["ticket"])] * 60
        opened = _parse_utc(position.get("time"))
        now_server = data_engine.server_now_epoch(position["symbol"])
        if opened is None or now_server is None:
            continue
        age = now_server - opened.timestamp()  # both in broker server time
        if age < limit:
            continue
        logger.info("[TRADE] Time stop: %s %s ticket %s open %d min (limit %d), floating %.2f",
                    position["symbol"], position["side"], position["ticket"], int(age // 60),
                    limit // 60, position["profit"])
        result = execution.close_position(position["ticket"])
        if result.get("success"):
            _record_close_execution(result, "TIME_STOP")
            closed += 1
        elif "not found" not in str(result.get("error", "")):
            logger.error("[TRADE] Time-stop close of ticket %s failed: %s", position["ticket"], result.get("error"))
    return closed


async def _act_on_decision(symbol: str, market: Dict[str, Any], decision: Dict[str, Any],
                           journal_entry: Optional[Dict[str, Any]] = None) -> str:
    """Every portfolio and account check, then the order and its memory record."""
    def done(action: str) -> str:
        if journal_entry is not None:
            journal_entry["action"] = action
        return "evaluated"

    signal = decision["signal"]
    strategy = decision.get("strategy") or "SWING"
    scalp = strategy in RULE_STRATEGIES  # rules-found setups: drift-checked, tagged orders
    if signal == "HOLD":
        return done("hold")
    if not bot_state["is_running"]:
        logger.info("[SYSTEM] Engine stopped before %s %s could execute", signal, symbol)
        return done("blocked: engine stopped")
    if bot_state["kill_switch"]:
        logger.warning("[SYSTEM] %s %s blocked: total drawdown kill-switch is active (reset it in Settings)",
                       signal, symbol)
        return done("blocked: drawdown kill-switch")
    if bot_state["circuit_breaker"]:
        logger.warning("[SYSTEM] %s %s blocked: daily loss circuit breaker is active", signal, symbol)
        return done("blocked: daily loss limit")
    if bot_state["profit_target_hit"]:
        logger.info("[SYSTEM] %s %s blocked: daily profit target reached, new entries paused", signal, symbol)
        return done("blocked: daily profit target")
    cooldown = _loss_cooldown_block(symbol, signal, strategy)
    if cooldown:
        logger.info("[SYSTEM] %s %s blocked: %s", signal, symbol, cooldown)
        return done("blocked: loss cooldown")

    positions = await asyncio.to_thread(execution.get_open_positions, symbol)
    if scalp:
        # The strategies trade independently: each holds at most one position per pair and ignores the
        # other's. Opposite sides on one pair need a hedging account (a netting account would net them off).
        others = [p for p in positions if p["managed"] and _position_strategy(p) != strategy]
        against = [p for p in others if p["side"] != signal]
        if against and not await asyncio.to_thread(data_engine.hedging_account):
            logger.info("[SYSTEM] %s %s %s blocked: the %s strategy holds the opposite side (ticket %s) and this "
                        "netting account cannot hold both", strategy.lower(), signal, symbol,
                        _position_strategy(against[0]).lower(), against[0]["ticket"])
            return done("blocked: opposite trade of another strategy (netting account)")
        positions = [p for p in positions if not p["managed"] or _position_strategy(p) == strategy]
    same_side = [p for p in positions if p["side"] == signal]
    if same_side:
        logger.info("[DUPLICATE PREVENTED] %s %s already open (ticket %s, %.2f lots); no new order",
                    symbol, signal, same_side[0]["ticket"], same_side[0]["volume"])
        return done("blocked: same direction already open")

    opposite = [p for p in positions if p["side"] != signal]
    if opposite:
        unmanaged = [p for p in opposite if not p["managed"]]
        if unmanaged:
            logger.warning("[REVERSAL] %s %s skipped: opposite position(s) %s were not opened by this engine "
                           "and are left untouched", symbol, signal, [p["ticket"] for p in unmanaged])
            return done("blocked: manual opposite position")
        required = int(decision.get("threshold") or config.CONFIDENCE_THRESHOLD) + config.REVERSAL_EXTRA_CONFIDENCE
        if decision["confidence_score"] < required:
            logger.info("[REVERSAL] %s %s not flipped: confidence %d < %d needed to close the open %s "
                        "(threshold + %d)", symbol, signal, decision["confidence_score"], required,
                        opposite[0]["side"], config.REVERSAL_EXTRA_CONFIDENCE)
            return done("blocked: reversal needs more confidence")

    if config.MAX_OPEN_POSITIONS > 0:
        all_positions = await asyncio.to_thread(execution.get_open_positions)
        if scalp:  # each rules strategy has its own position limit
            all_positions = [p for p in all_positions if p["managed"] and _position_strategy(p) == strategy]
        if len(all_positions) - len(opposite) >= config.MAX_OPEN_POSITIONS:
            logger.warning("[SYSTEM] %s %s skipped: %d open %spositions (max %d)", signal, symbol, len(all_positions),
                           f"{strategy.lower()} " if scalp else "", config.MAX_OPEN_POSITIONS)
            return done("blocked: max open positions")

    risk_percent = _strategy_risk(strategy) if scalp else float(bot_state["risk_percent"])
    if bot_state["risk_throttled"]:
        risk_percent *= config.THROTTLE_RISK_FACTOR
        logger.info("[SYSTEM] %s %s: risk throttled to %.2f%% (equity %.2f%% below its peak)", signal, symbol,
                    risk_percent, bot_state["total_drawdown_pct"])
    order_args = dict(sl_distance=decision.get("sl_distance"), tp_distance=decision.get("tp_distance"),
                      entry_reference=decision.get("entry_reference"), atr_reference=decision.get("atr_reference"),
                      max_total_open_risk=_daily_risk_budget(),
                      max_drift_atr=config.SCALP_MAX_DRIFT_ATR if scalp else None,
                      # each rules strategy has its own per-currency budget; the daily loss budget is shared
                      currency_ignore=[f"{config.ORDER_COMMENT}{tag}" for s, tag in ORDER_TAGS.items()
                                       if scalp and s != strategy])

    if opposite:
        # Check the replacement first: never close a position for a trade that would then be refused.
        try:
            await asyncio.to_thread(execution.preview_trade, symbol, signal, decision["stop_loss"],
                                    decision["take_profit"], risk_percent,
                                    exclude_tickets=[p["ticket"] for p in opposite], **order_args)
        except execution.TradeExecutionError as exc:
            logger.warning("[REVERSAL] %s %s would be refused (%s); the open %s is kept", symbol, signal, exc,
                           opposite[0]["side"])
            bot_state["last_error"] = f"{symbol} {signal}: {exc}"
            return done(f"blocked: {exc}")
        for position in opposite:
            logger.info("[REVERSAL] %s flipped to %s: closing %s ticket %s (%.2f lots, floating %.2f)",
                        symbol, signal, position["side"], position["ticket"], position["volume"], position["profit"])
            result = await asyncio.to_thread(execution.close_position, position["ticket"])
            if not result.get("success"):
                logger.error("[REVERSAL] Could not close ticket %s (%s); new %s aborted",
                             position["ticket"], result.get("error"), signal)
                return done("blocked: reversal close failed")
            await asyncio.to_thread(_record_close_execution, result, "REVERSAL", market)
        remaining = await asyncio.to_thread(execution.get_open_positions, symbol)
        if any(p["side"] != signal for p in remaining):
            logger.error("[REVERSAL] %s opposite exposure still open after close; new %s aborted", symbol, signal)
            return done("blocked: opposite position still open")

    try:
        trade = await asyncio.to_thread(execution.execute_trade, symbol, signal, decision["stop_loss"],
                                        decision["take_profit"], risk_percent,
                                        comment=f"{config.ORDER_COMMENT}{ORDER_TAGS[strategy]}" if scalp else None,
                                        **order_args)
    except execution.TradeExecutionError as exc:
        bot_state["last_error"] = f"{symbol} {signal}: {exc}"
        logger.error("[TRADE] %s %s not executed: %s", symbol, signal, exc)
        return done(f"rejected: {exc}")

    if trade.get("risk_percent_actual") is not None:
        risk_percent = trade["risk_percent_actual"]  # record what the stop actually risks (min lot, fixed lot)
    quality = trade.get("execution") or {}
    logger.info("[TRADE] FILLED %s %s %.2f lots (%s sizing) @ %s | SL %s | TP %s | deal %s | position %s | "
                "risk %.2f%% | slippage %s pts in %s ms", signal, symbol, trade["volume"], trade.get("sizing_mode"),
                trade["price"], trade["stop_loss"], trade["take_profit"], trade["deal"], trade["position_ticket"],
                risk_percent, quality.get("slippage_points"), quality.get("execution_ms"))

    # Snapshot the market at the moment of the fill, then log the trade to memory.json.
    context = await asyncio.to_thread(_capture_context, symbol, market)
    record = _build_trade_record(symbol, market, decision, trade, risk_percent, context)
    record["sizing_mode"] = trade.get("sizing_mode")
    try:
        await asyncio.to_thread(memory_store.append_trade_memory, record)
    except Exception as exc:  # append_trade_memory journals I/O failures itself; this is a last resort
        logger.exception("[LEARNING] Trade %s filled but could not be logged: %s", trade["deal"], exc)
    else:
        logger.info("[LEARNING] Execution context: %s", _describe_context(symbol, record["market_context"]))
    if journal_entry is not None:
        journal_entry.update(ticket=trade["position_ticket"], fill_price=trade["price"], lots=trade["volume"],
                             risk_percent=risk_percent)
    await asyncio.to_thread(_refresh_account_state)
    return done("filled")


async def _run_audit_async(force: bool) -> Dict[str, Any]:
    bot_state["audit_in_progress"] = True
    try:
        result = await asyncio.to_thread(auditor.run_audit, force)
    finally:
        bot_state["audit_in_progress"] = False
    if result.get("status") != "busy":
        bot_state["last_audit_result"] = {**result, "at": data_engine.utc_now_iso()}
    await asyncio.to_thread(_refresh_learning_state)
    return result


def _reconcile() -> None:
    try:
        newly_closed = memory_store.reconcile_closed_trades()
        if newly_closed:
            logger.info("[LEARNING] %d newly closed trade(s) reconciled into memory.json", newly_closed)
    except Exception as exc:
        logger.warning("[LEARNING] Reconciliation failed: %s", exc)


def _rule_housekeeping() -> None:
    """Follow blocked (shadow) trades and expire/retire rules they or time no longer support."""
    try:
        shadow_store.resolve_open_shadows()
        if auditor.maintain_rules():
            _refresh_learning_state()
    except Exception as exc:
        logger.warning("[LEARNING] Rule housekeeping failed: %s", exc)


async def _learning_cycle() -> None:
    """Reconcile closed trades, follow shadow trades, run the auditor once its daily window is reached."""
    await asyncio.to_thread(_reconcile)
    await asyncio.to_thread(_rule_housekeeping)
    due = await asyncio.to_thread(auditor.audit_due)
    if due["due"]:
        logger.info("[LEARNING] Daily audit due: %s", due["reason"])
        await _run_audit_async(force=False)
    await asyncio.to_thread(_agent_upkeep)


def _agent_upkeep() -> None:
    """Resolve the exploration trades and hand every new reward (trades, shadows, explorations) to the agent."""
    try:
        shadow_store.resolve_open_shadows(config.EXPLORE_FILE)
        agent.sync()
    except Exception as exc:
        logger.warning("[AGENT] Learning update failed: %s", exc)


OFF_SESSION_REMINDER_SECONDS = 1800


def _mark_off_session() -> None:
    """Scalp mode outside the trading window: say so on the dashboard and, every 30 minutes, in the log."""
    note = scalper.session_note()
    for symbol in config.SYMBOLS:
        previous = bot_state["decisions"].get(symbol) or {}
        bot_state["decisions"][symbol] = {**previous, "signal": "HOLD", "raw_signal": "HOLD", "confidence": 0,
                                          "penalty": 0, "note": f"off session: {note}", "strategy": "SCALP",
                                          "stage": "OFF_SESSION"}
    if time.monotonic() - float(bot_state.get("_off_session_logged") or 0) >= OFF_SESSION_REMINDER_SECONDS:
        bot_state["_off_session_logged"] = time.monotonic()
        logger.info("[SYSTEM] Waiting for the trading hours: %s. The engine keeps watching positions, "
                    "news and the daily limits meanwhile.", note)


async def run_scan_cycle() -> bool:
    """One scan. Returns False when it was a quiet off-session pass (nothing to log per scan)."""
    bot_state["scan_count"] += 1
    scan_number = bot_state["scan_count"]
    connected = await asyncio.to_thread(data_engine.ensure_connection)
    bot_state["mt5_connected"] = connected
    if not connected:
        logger.warning("[SYSTEM] Scan #%d skipped: MT5 is not connected", scan_number)
        return True
    await asyncio.to_thread(_refresh_symbol_universe)  # follows account switches in the terminal
    await asyncio.to_thread(_reconcile)  # a stop-out since the last scan must count for the loss cooldown
    with contextlib.suppress(Exception):
        await asyncio.to_thread(news.refresh_if_stale)
    await asyncio.to_thread(_weekend_close_positions)
    await asyncio.to_thread(_refresh_account_state)
    if config.STRATEGY_MODE == "SCALP" and not scalper.in_session():
        _log_scan_outcomes({"off_session": list(config.SYMBOLS)})  # logs the OPEN -> CLOSED change once
        _mark_off_session()
        await _learning_cycle()
        bot_state["last_scan_at"] = data_engine.utc_now_iso()
        return False
    bot_state["_off_session_logged"] = 0
    if config.STRATEGY_MODE != "SCALP":  # scalp mode logs one summary line per M5 bar instead
        logger.info("[SYSTEM] Scan #%d started | %d symbols | risk %.2f%% | equity %.2f %s", scan_number,
                    len(config.SYMBOLS), bot_state["risk_percent"], bot_state["equity"], bot_state["currency"])

    outcomes: Dict[str, List[str]] = {}
    for symbol in list(config.SYMBOLS):
        if not bot_state["is_running"]:
            logger.info("[SYSTEM] Engine stopped mid-scan; remaining symbols skipped")
            break
        if not await asyncio.to_thread(data_engine.ensure_connection):
            # The terminal restarted or lost its link while the AI was thinking: retry next scan.
            logger.warning("[SYSTEM] MT5 connection lost mid-scan; remaining symbols wait for the next scan")
            bot_state["mt5_connected"] = False
            break
        bot_state["current_symbol"] = symbol
        try:
            outcome = await process_symbol(symbol)
        except Exception as exc:
            logger.exception("[SYSTEM] %s processing failed: %s", symbol, exc)
            outcome = "error"
        outcomes.setdefault(outcome, []).append(symbol)
    bot_state["current_symbol"] = None
    bot_state["unchanged_symbols"] = outcomes.get("unchanged", [])
    _log_scan_outcomes(outcomes)

    await _learning_cycle()
    await asyncio.to_thread(_refresh_account_state)
    bot_state["last_scan_at"] = data_engine.utc_now_iso()
    # Scalp scans between M5 bars only confirm nothing new closed: no per-scan log lines for those.
    return not (config.STRATEGY_MODE == "SCALP" and set(outcomes) <= {"unchanged"})


def _log_scan_outcomes(outcomes: Dict[str, List[str]]) -> None:
    """One summary line per event instead of the same line every 30 seconds."""
    if config.STRATEGY_MODE == "SCALP":
        session_open = scalper.in_session()
        if bot_state.get("scalp_session_open") != session_open:
            bot_state["scalp_session_open"] = session_open
            logger.info("[SYSTEM] Trading hours %s: %s", "OPEN" if session_open else "CLOSED", scalper.session_note())
        no_setup = outcomes.get("no_setup", [])
        checked = sum(len(v) for k, v in outcomes.items() if k != "unchanged")
        if checked:
            now = datetime.now(timezone.utc)
            next_check = (now + timedelta(minutes=5 - now.minute % 5)).replace(second=0, microsecond=0)
            bot_state["scalp_next_check_at"] = next_check.isoformat()
            reasons = bot_state.get("scalp_reasons") or {}
            logger.info("[SYSTEM] Scan #%d: M5 bar checked on %d symbol(s), %d without a setup; next check %s UTC "
                        "(%s your time)%s", bot_state["scan_count"], checked, len(no_setup),
                        f"{next_check:%H:%M}", f"{next_check.astimezone():%H:%M}",
                        "".join(f"\n    {s}: {reasons.get(s, '')}" for s in no_setup))
        return
    unchanged = outcomes.get("unchanged", [])
    if unchanged and unchanged != bot_state.get("_last_unchanged_logged"):
        logger.info("[SYSTEM] %d symbol(s) not re-sent to the AI (same closed H1 bar, price within %.1f ATR): %s",
                    len(unchanged), config.AI_REEVALUATE_ATR_MOVE, ", ".join(unchanged))
    bot_state["_last_unchanged_logged"] = unchanged


async def trading_loop() -> None:
    logger.info("[SYSTEM] Trading loop online (engine %s)", "RUNNING" if bot_state["is_running"] else "STANDBY")
    last_housekeeping = time.monotonic()
    while True:
        try:
            if not bot_state["is_running"]:
                bot_state["next_scan_at"] = None
                # Keep learning from trades that close while the engine is paused.
                if time.monotonic() - last_housekeeping >= HOUSEKEEPING_SECONDS:
                    last_housekeeping = time.monotonic()
                    if bot_state["mt5_connected"]:
                        await _learning_cycle()
                await asyncio.sleep(1.0)
                continue

            started = time.monotonic()
            active = await run_scan_cycle()
            finished = time.monotonic()
            last_housekeeping = finished
            bot_state["last_scan_duration"] = round(finished - started, 1)
            if active:
                logger.info("[SYSTEM] Scan #%d finished in %.1fs; next scan in %ds",
                            bot_state["scan_count"], finished - started, bot_state["interval"])

            # Sleep for the configured interval, re-reading it every second so
            # interval changes and Stop requests from the dashboard apply immediately.
            # (Weekend and daily-loss closes run in risk_watchdog, independently of this loop.)
            while bot_state["is_running"]:
                remaining = finished + float(bot_state["interval"]) - time.monotonic()
                if remaining <= 0:
                    break
                bot_state["next_scan_at"] = (datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat(
                    timespec="seconds")
                await asyncio.sleep(min(1.0, remaining))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("[SYSTEM] Trading loop error: %s", exc)
            await asyncio.sleep(5.0)


def _spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


# -----------------------------------------------------------------------------
# Application lifecycle
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    for warning in config.CONFIG_WARNINGS:
        logger.warning("[SYSTEM] Config: %s", warning)
    settings_store.load_saved()
    bot_state.update(risk_percent=config.DEFAULT_RISK_PERCENT, interval=config.SCAN_INTERVAL_SECONDS)
    logger.info("[SYSTEM] Autonomous AI Hedge Fund booting | %s strategy | %d symbols configured | threshold %d | "
                "risk %.2f%% | interval %ds | model %s", config.STRATEGY_MODE, len(config.SYMBOLS),
                config.CONFIDENCE_THRESHOLD, config.DEFAULT_RISK_PERCENT, config.SCAN_INTERVAL_SECONDS,
                config.DEEPSEEK_MODEL)
    for warning in _setting_conflicts():
        logger.warning("[SYSTEM] Settings: %s", warning)
    if config.STRATEGY_MODE == "SCALP":
        logger.info("[SYSTEM] Trading hours: %s", scalper.session_note())
    if not config.DEEPSEEK_API_KEY:
        logger.warning("[SYSTEM] DEEPSEEK_API_KEY is missing: every decision will be HOLD until it is set in .env")

    try:
        await asyncio.to_thread(memory_store.flush_pending_journal)
    except OSError as exc:
        logger.error("[LEARNING] Journaled fills could not be merged into memory.json yet: %s", exc)

    connected = await asyncio.to_thread(data_engine.initialize_mt5)
    bot_state["mt5_connected"] = connected
    if connected:
        await asyncio.to_thread(_refresh_symbol_universe, True)
        try:
            reconciled = await asyncio.to_thread(memory_store.reconcile_closed_trades)
            logger.info("[LEARNING] Startup reconciliation: %d trade(s) closed while offline", reconciled)
        except Exception as exc:
            logger.warning("[LEARNING] Startup reconciliation failed: %s", exc)
        await asyncio.to_thread(_refresh_account_state)
    else:
        logger.warning("[SYSTEM] Starting without MT5; the engine retries the connection on every scan")

    await asyncio.to_thread(_rule_housekeeping)  # expire / retire / mark legacy rules before showing them
    await asyncio.to_thread(_refresh_learning_state)
    active = [rule for rule in bot_state["learned_rules"] if rule.get("status") == "ACTIVE"]
    logger.info("[LEARNING] %d active learned rule(s) loaded from %s", len(active), config.RULES_FILE.name)

    due = await asyncio.to_thread(auditor.audit_due)
    if due["due"]:
        logger.info("[AUDITOR] Startup audit due: %s", due["reason"])
        _spawn(_run_audit_async(force=False))
    else:
        logger.info("[AUDITOR] No startup audit needed: %s", due.get("reason"))

    if config.AUTO_START_ENGINE:
        bot_state["is_running"] = True
        logger.info("[SYSTEM] AUTO_START_ENGINE=true: engine started automatically")

    _spawn(asyncio.to_thread(news.refresh_if_stale))
    loop_task = asyncio.create_task(trading_loop(), name="trading_loop")
    watchdog_task = asyncio.create_task(risk_watchdog(), name="risk_watchdog")
    logger.info("[SYSTEM] Dashboard ready at http://%s:%d", config.HOST, config.PORT)
    try:
        yield
    finally:
        bot_state["is_running"] = False
        for task in (loop_task, watchdog_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for task in list(_background_tasks):
            task.cancel()
        await asyncio.to_thread(data_engine.shutdown_mt5)
        logger.info("[SYSTEM] Shutdown complete")


app = FastAPI(title="Autonomous AI Hedge Fund", version="1.0.0", lifespan=lifespan)


# -----------------------------------------------------------------------------
# API models
# -----------------------------------------------------------------------------
class ControlRequest(BaseModel):
    action: Optional[Literal["start", "stop"]] = None
    interval: Optional[int] = Field(default=None, ge=5, le=86_400)
    risk_percent: Optional[float] = Field(default=None, gt=0, le=config.MAX_RISK_PERCENT)


class CloseRequest(BaseModel):
    ticket: int = Field(gt=0)


class AuditRequest(BaseModel):
    force: bool = True


class SettingsUpdate(BaseModel):
    """Fields left out (or null) keep their current value."""
    risk_percent: Optional[float] = Field(default=None, ge=0.05, le=config.MAX_RISK_PERCENT)
    sizing_mode: Optional[Literal["RISK", "FIXED"]] = None
    fixed_lot: Optional[float] = Field(default=None, ge=0.01, le=100.0)
    max_open_positions: Optional[int] = Field(default=None, ge=0, le=50)
    max_daily_loss_percent: Optional[float] = Field(default=None, ge=0.0, le=50.0)
    max_daily_profit_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    confidence_threshold: Optional[int] = Field(default=None, ge=50, le=95)
    scan_interval_seconds: Optional[int] = Field(default=None, ge=5, le=86_400)
    symbols_demo: Optional[List[str]] = Field(default=None, min_length=1, max_length=config.MAX_SYMBOLS)
    symbols_live: Optional[List[str]] = Field(default=None, min_length=1, max_length=config.MAX_SYMBOLS)
    rule_scope: Optional[Literal["PAIR", "BROKER"]] = None
    overextension_guard: Optional[bool] = None
    weekend_entry_cutoff_hours: Optional[float] = Field(default=None, ge=0.0, le=48.0)
    weekend_close: Optional[bool] = None
    max_currency_risk_percent: Optional[float] = Field(default=None, ge=0.0, le=50.0)
    daily_loss_close_positions: Optional[bool] = None
    loss_cooldown_minutes: Optional[int] = Field(default=None, ge=0, le=10_080)
    ai_new_bar_only: Optional[bool] = None
    news_guard: Optional[bool] = None
    calibrate_threshold: Optional[bool] = None
    strategy_mode: Optional[Literal["SCALP", "SWING"]] = None
    scalp_time_stop_minutes: Optional[int] = Field(default=None, ge=5, le=1440)
    scalp_max_trades_per_symbol: Optional[int] = Field(default=None, ge=1, le=50)
    trade_all_hours: Optional[bool] = None
    scalp_session_start_london: Optional[int] = Field(default=None, ge=0, le=23)
    scalp_session_end_new_york: Optional[int] = Field(default=None, ge=1, le=17)
    scalp_strict_guard: Optional[bool] = None
    scalp_medium_trend: Optional[bool] = None
    max_total_drawdown_percent: Optional[float] = Field(default=None, ge=0.0, le=90.0)
    drawdown_throttle_percent: Optional[float] = Field(default=None, ge=0.0, le=90.0)
    kill_switch_close_positions: Optional[bool] = None
    agent_enabled: Optional[bool] = None
    agent_explore: Optional[bool] = None
    agent_shadow_until_learned: Optional[bool] = None
    agent_min_rewards: Optional[int] = Field(default=None, ge=5, le=10_000)
    scalp_rsi_pullback: Optional[float] = Field(default=None, ge=30.0, le=48.0)
    scalp_enabled: Optional[bool] = None
    intraday_enabled: Optional[bool] = None
    intraday_risk_percent: Optional[float] = Field(default=None, ge=0.0, le=config.MAX_RISK_PERCENT)


SETTINGS_BOUNDS = {
    "risk_percent": [0.05, config.MAX_RISK_PERCENT], "fixed_lot": [0.01, 100.0], "max_open_positions": [0, 50],
    "max_daily_loss_percent": [0.0, 50.0], "max_daily_profit_percent": [0.0, 100.0],
    "confidence_threshold": [50, 95], "scan_interval_seconds": [5, 86_400],
    "symbols_demo": [1, config.MAX_SYMBOLS], "symbols_live": [1, config.MAX_SYMBOLS],
    "weekend_entry_cutoff_hours": [0.0, 48.0], "max_currency_risk_percent": [0.0, 50.0],
    "loss_cooldown_minutes": [0, 10_080], "scalp_time_stop_minutes": [5, 1440], "scalp_max_trades_per_symbol": [1, 50],
    "scalp_session_start_london": [0, 23], "scalp_session_end_new_york": [1, 17],
    "max_total_drawdown_percent": [0.0, 90.0], "drawdown_throttle_percent": [0.0, 90.0],
    "intraday_risk_percent": [0.0, config.MAX_RISK_PERCENT],
    "agent_min_rewards": [5, 10_000], "scalp_rsi_pullback": [30.0, 48.0],
}
_universe: Dict[str, Any] = {"key": None, "at": 0.0, "offered": []}
UNIVERSE_TTL_SECONDS = 600


def _offered_symbol_names(force: bool = False) -> Optional[List[str]]:
    """Broker symbol names for the logged-in account, cached for 10 minutes per account."""
    account = data_engine.get_account_snapshot()
    if account is None:
        return None
    key = (account["login"], account["server"])
    if force or key != _universe["key"] or time.monotonic() - _universe["at"] > UNIVERSE_TTL_SECONDS:
        offered = data_engine.broker_symbols()
        if offered is None:
            return None
        _universe.update(key=key, at=time.monotonic(), offered=[item["name"] for item in offered])
    return _universe["offered"]


def _refresh_symbol_universe(force: bool = False) -> None:
    """Pick the DEMO or LIVE list for the logged-in account and resolve it to the broker's symbol names."""
    account = data_engine.get_account_snapshot()
    if account is None:
        return
    config.ACTIVE_ACCOUNT = {"mode": account["account_mode"], "broker": account.get("company"),
                             "server": account.get("server")}
    login_key = (account["login"], account.get("server"))
    if _universe.get("tagged_for") != login_key:
        _universe["tagged_for"] = login_key
        try:
            memory_store.tag_untagged_trades(account["login"], account["account_mode"], account.get("company"),
                                             account.get("server"))
        except OSError as exc:
            logger.warning("[LEARNING] Could not tag earlier trades with the account type: %s", exc)
    offered = _offered_symbol_names(force)
    if offered is None:
        return
    mode = account["account_mode"]
    requested = config.SYMBOLS_LIVE if mode == "LIVE" else config.SYMBOLS_DEMO
    resolved, mapping, unresolved = data_engine.resolve_symbols(requested, offered)
    changed = resolved != config.SYMBOLS or mode != bot_state.get("account_mode") or unresolved != bot_state.get(
        "unresolved_symbols")
    config.SYMBOLS = resolved
    bot_state.update(account_mode=mode, symbol_map=mapping, unresolved_symbols=unresolved)
    if changed:
        renamed = [f"{k}->{v}" for k, v in mapping.items() if k != v]
        logger.info("[SYSTEM] %s account on %s: trading the %s list (%d symbols)%s", mode, account["server"], mode,
                    len(resolved), f" | matched {', '.join(renamed)}" if renamed else "")
        if unresolved:
            logger.warning("[SYSTEM] Not offered by %s and skipped: %s", account["server"], ", ".join(unresolved))
        if not resolved:
            logger.warning("[SYSTEM] None of the %s symbols exist on this broker: nothing will be traded", mode)


def _settings_payload(warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    account = data_engine.get_account_snapshot() or {}
    return {
        "settings": settings_store.current(),
        "defaults": settings_store.ENV_DEFAULTS,
        "overridden": settings_store.overridden(),
        "bounds": SETTINGS_BOUNDS,
        "account": {"mode": account.get("account_mode"), "login": account.get("login"),
                    "server": account.get("server"), "connected": bool(account)},
        "active_symbols": list(config.SYMBOLS),
        "symbol_map": bot_state.get("symbol_map") or {},
        "unresolved_symbols": bot_state.get("unresolved_symbols") or [],
        "day": {key: bot_state.get(key) for key in ("day_start_equity", "day_pnl_pct", "daily_drawdown_pct",
                                                    "circuit_breaker", "profit_target_hit")},
        "account_guard": {key: bot_state.get(key) for key in ("peak_equity", "total_drawdown_pct", "kill_switch",
                                                              "kill_switch_at", "risk_throttled")},
        "warnings": warnings or [],
    }


def _clean_symbol_list(requested: List[str]) -> List[str]:
    """Trim and de-duplicate, ignoring case (first spelling wins)."""
    unique: Dict[str, str] = {}
    for symbol in requested:
        name = (symbol or "").strip()
        if name and name.upper() not in unique:
            unique[name.upper()] = name
    return list(unique.values())


def _check_symbol_list(key: str, symbols: List[str], warnings: List[str]) -> List[str]:
    """
    For the connected account's list: reject symbols its broker does not offer and use the
    broker's exact spelling for case-only differences. The other list is kept as entered.
    """
    list_mode = "LIVE" if key == "symbols_live" else "DEMO"
    account = data_engine.get_account_snapshot()
    offered = _offered_symbol_names()
    if account is None or offered is None:
        warnings.append(f"MT5 is offline: the {list_mode} list will be matched to the broker when it connects")
        return symbols
    if account["account_mode"] != list_mode:
        warnings.append(f"The {list_mode} list will be matched to the broker's names when a {list_mode} account "
                        f"is logged in (connected now: {account['account_mode']})")
        return symbols
    _, _, unresolved = data_engine.resolve_symbols(symbols, offered)
    if unresolved:
        raise HTTPException(status_code=422, detail=f"Not offered by {account['server']} ({list_mode} list): "
                                                    f"{', '.join(unresolved)}")
    exact = {name.upper(): name for name in offered}
    return [exact.get(symbol.upper(), symbol) for symbol in symbols]


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    bootstrap = {
        "symbols": config.SYMBOLS,
        "interval": bot_state["interval"],
        "risk_percent": bot_state["risk_percent"],
        "max_risk_percent": config.MAX_RISK_PERCENT,
        "confidence_threshold": calibration.effective_threshold(),
    }
    return templates.TemplateResponse(request, "index.html", {"bootstrap": bootstrap})


@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    try:
        _refresh_account_state()
    except Exception as exc:
        logger.debug("[SYSTEM] Status refresh failed: %s", exc)
    return _status_payload()


@app.post("/api/control")
async def api_control(body: ControlRequest) -> Dict[str, Any]:
    changes: List[str] = []
    persist: Dict[str, Any] = {}
    if body.interval is not None and body.interval != bot_state["interval"]:
        bot_state["interval"] = persist["scan_interval_seconds"] = body.interval
        changes.append(f"interval={body.interval}s")
    if body.risk_percent is not None:
        risk = round(float(body.risk_percent), 2)
        if risk != bot_state["risk_percent"]:
            bot_state["risk_percent"] = persist["risk_percent"] = risk
            changes.append(f"risk={risk}%")
    if persist:
        await asyncio.to_thread(settings_store.save, persist)
    if body.action == "start" and not bot_state["is_running"]:
        bot_state["is_running"] = True
        changes.append("engine=RUNNING")
    elif body.action == "stop" and bot_state["is_running"]:
        bot_state["is_running"] = False
        changes.append("engine=STOPPED")
    if changes:
        logger.info("[SYSTEM] Control update: %s", ", ".join(changes))
    return {
        "success": True,
        "changes": changes,
        "is_running": bot_state["is_running"],
        "interval": bot_state["interval"],
        "risk_percent": bot_state["risk_percent"],
    }


@app.post("/api/close")
def api_close(body: CloseRequest) -> Dict[str, Any]:
    logger.info("[TRADE] Manual close requested from dashboard for ticket %s", body.ticket)
    result = execution.close_position(body.ticket)
    if not result.get("success") and "not found" in str(result.get("error", "")):
        raise HTTPException(status_code=404, detail=result["error"])
    if result.get("success"):
        _record_close_execution(result, "MANUAL_DASHBOARD")
    with contextlib.suppress(Exception):
        _refresh_account_state()
    return result


@app.post("/api/audit")
async def api_audit(body: Optional[AuditRequest] = None) -> Dict[str, Any]:
    force = body.force if body is not None else True
    logger.info("[AUDITOR] Audit requested from dashboard (force=%s)", force)
    return await _run_audit_async(force)


@app.get("/api/scorecard")
def api_scorecard(days: int = Query(default=28, ge=0, le=3650)) -> Dict[str, Any]:
    """Rewards and penalties per strategy from demo/live and shadow trades (days=0: everything)."""
    return {"days": days or None, "agent": _agent_status(),
            "strategies": {s: agent.scorecard(s, days or None) for s in agent.STRATEGIES}}


@app.get("/api/rules")
def api_rules() -> Dict[str, Any]:
    return auditor.load_rules_document()


@app.get("/api/fills")
def api_fills(limit: int = Query(default=500, ge=1, le=LEDGER_LIMIT)) -> Dict[str, Any]:
    """Confirmed fills ledger (newest first) for the dashboard grid."""
    cache = _history_and_stats()
    return {"version": cache["version"], "rows": cache["ledger"][:limit], "stats": cache["stats"],
            "currency": bot_state.get("currency", "")}


@app.get("/api/settings")
def api_get_settings() -> Dict[str, Any]:
    return _settings_payload()


@app.post("/api/settings")
def api_save_settings(body: SettingsUpdate) -> Dict[str, Any]:
    changes = body.model_dump(exclude_none=True)
    warnings: List[str] = []
    for key in settings_store.LIST_SETTINGS:
        if key in changes:
            changes[key] = _clean_symbol_list(changes[key])
            if not changes[key]:
                raise HTTPException(status_code=422, detail="Each symbol list needs at least one symbol")
            changes[key] = _check_symbol_list(key, changes[key], warnings)
    for key in ("risk_percent", "fixed_lot", "max_daily_loss_percent", "max_daily_profit_percent",
                "weekend_entry_cutoff_hours", "max_currency_risk_percent"):
        if key in changes:
            changes[key] = round(float(changes[key]), 2)

    before = settings_store.current()
    changed = {key: value for key, value in changes.items() if before.get(key) != value}
    if changed:
        settings_store.save(changed)
        bot_state.update(risk_percent=config.DEFAULT_RISK_PERCENT, interval=config.SCAN_INTERVAL_SECONDS)
        if {"max_daily_loss_percent", "max_daily_profit_percent"} & changed.keys():
            _reevaluate_daily_limits()
        if set(settings_store.LIST_SETTINGS) & changed.keys():
            _refresh_symbol_universe()
        if "news_guard" in changed and config.NEWS_GUARD:
            news.refresh_if_stale()
        if {"ai_new_bar_only", "strategy_mode", "scalp_enabled", "intraday_enabled", "trade_all_hours"} & changed.keys():
            _last_evaluation.clear()
            _last_scalp_bar.clear()
            _last_intraday_bar.clear()
        _refresh_learning_state()  # rule applicability depends on symbols and rule sharing
        logger.info("[SYSTEM] Settings updated from dashboard: %s",
                    ", ".join(f"{key}={value}" for key, value in changed.items()))
        for conflict in _setting_conflicts():
            logger.warning("[SYSTEM] Settings: %s", conflict)
            warnings.append(conflict)
    return _settings_payload(warnings)


@app.post("/api/settings/reset")
def api_reset_settings() -> Dict[str, Any]:
    settings_store.reset()
    bot_state.update(risk_percent=config.DEFAULT_RISK_PERCENT, interval=config.SCAN_INTERVAL_SECONDS)
    _reevaluate_daily_limits()
    _refresh_symbol_universe()
    _refresh_learning_state()
    return _settings_payload()


@app.post("/api/kill-switch/reset")
def api_reset_kill_switch() -> Dict[str, Any]:
    """Clear the total-drawdown kill-switch and measure drawdown from today's equity (after a review)."""
    result = _reset_kill_switch()
    return {**result, **_settings_payload()}


@app.get("/api/symbols")
def api_symbols() -> Dict[str, Any]:
    """Symbols the connected broker offers (for the settings picker) and the account type."""
    offered = data_engine.broker_symbols()
    account = data_engine.get_account_snapshot()
    return {"connected": offered is not None, "account_mode": account["account_mode"] if account else None,
            "server": account["server"] if account else None, "symbols": (offered or [])[:5000]}


@app.post("/api/rules/{rule_id}/toggle")
def api_toggle_rule(rule_id: str) -> Dict[str, Any]:
    document = auditor.load_rules_document()
    rule = next((r for r in document["rules"] if str(r.get("id", "")).upper() == rule_id.upper()), None)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"rule {rule_id} not found")
    new_status = "DISABLED" if rule.get("status") == "ACTIVE" else "ACTIVE"
    try:
        updated = auditor.set_rule_status(rule_id, new_status)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    _refresh_learning_state()
    return {"success": True, "rule": updated}


@app.post("/api/shutdown")
async def api_shutdown(request: Request) -> Dict[str, Any]:
    """Graceful stop used by stop_background.ps1 (loopback callers only)."""
    client = request.client.host if request.client else ""
    if client not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="shutdown is only accepted from localhost")
    bot_state["is_running"] = False
    logger.info("[SYSTEM] Graceful shutdown requested")
    if _server is not None:
        _server.should_exit = True
    return {"success": True}


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
# run_server.bat restarts the server after a crash but not after this exit code.
EXIT_ALREADY_RUNNING = 3
_instance_lock_handle = None


def _acquire_instance_lock() -> bool:
    """Hold an OS file lock so two engines can never trade the same account from this folder."""
    global _instance_lock_handle
    try:
        handle = open(config.LOCK_FILE, "a+")
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    _instance_lock_handle = handle
    return True


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def main() -> None:
    global _server
    if not _acquire_instance_lock():
        logger.error("[SYSTEM] Another instance is already running from %s; exiting", config.BASE_DIR)
        sys.exit(EXIT_ALREADY_RUNNING)
    if not _port_available(config.HOST, config.PORT):
        logger.error("[SYSTEM] %s:%d is already in use; stop the other server first", config.HOST, config.PORT)
        sys.exit(EXIT_ALREADY_RUNNING)
    server_config = uvicorn.Config(app, host=config.HOST, port=config.PORT, log_config=None,
                                   access_log=False, log_level="info")
    _server = uvicorn.Server(server_config)
    _server.run()


if __name__ == "__main__":
    main()
