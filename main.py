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
import backtest
import calibration
import config
import data_engine
import execution
import memory_store
import news
import rule_engine
import scalper
import settings_store
import shadow_store


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
            document[str(login)] = {**{key: bot_state.get(key) for key in _RISK_STATE_KEYS},
                                    "saved_at": data_engine.utc_now_iso()}
            memory_store.write_json_atomic(config.RISK_STATE_FILE, document)
    except OSError as exc:
        logger.error("[SYSTEM] Could not save the daily risk state: %s", exc)


def _update_daily_risk(equity: float, login: Optional[int] = None) -> None:
    """
    Daily loss/profit limits against the equity at the start of the trading day (17:00 New York).
    The start equity and a tripped breaker are saved to disk, so a restart cannot reset them.
    """
    login = int(login or bot_state.get("day_login") or (bot_state.get("account") or {}).get("login") or 0)
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
    if config.STRATEGY_MODE == "SCALP" and last_exit > 17:
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
    calibration.refresh(cache.get("records") or [], mode)
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
        "next_open_utc": opens.isoformat() if opens else None,
        "setting_warnings": _setting_conflicts(),
        "time_stop_minutes": config.SCALP_TIME_STOP_MINUTES,
        "max_trades_per_symbol": config.SCALP_MAX_TRADES_PER_SYMBOL,
        "strict_guard": config.SCALP_STRICT_GUARD,
        "medium_trend": config.SCALP_MEDIUM_TREND,
        "reward_risk": config.SCALP_REWARD_RISK,
    }
    payload["backtest"] = dict(bot_state.get("backtest") or {})
    return payload


# -----------------------------------------------------------------------------
# Trading engine
# -----------------------------------------------------------------------------
def _record_decision(symbol: str, market: Dict[str, Any], decision: Dict[str, Any]) -> None:
    bot_state.update(
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
        equity=market.get("equity", bot_state["equity"]),
    )
    bot_state["decisions"][symbol] = {
        "signal": decision["signal"],
        "raw_signal": decision.get("raw_signal"),
        "confidence": decision["confidence_score"],
        "penalty": sum(rule.get("points", 0) for rule in decision.get("applied_rules", [])),
        "time": decision.get("timestamp"),
        "error": decision.get("error"),
        "digits": market.get("digits"),
        "strategy": decision.get("strategy") or "SWING",
        "note": (decision.get("setup") or {}).get("reason"),
    }


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


def _flatten_for_daily_loss() -> int:
    """Close the engine's own positions after the daily loss limit tripped (blocking). Manual trades stay."""
    positions = [p for p in execution.get_open_positions() if p["managed"]]
    closed = failed = 0
    for position in positions:
        logger.warning("[SYSTEM] Daily loss limit: closing %s %s ticket %s (floating %.2f)",
                       position["symbol"], position["side"], position["ticket"], position["profit"])
        result = execution.close_position(position["ticket"])
        if result.get("success"):
            _record_close_execution(result, "DAILY_LOSS_LIMIT")
            closed += 1
        elif "not found" in str(result.get("error", "")):
            continue  # already closed by its stop
        else:
            failed += 1
            logger.error("[SYSTEM] Daily loss close of ticket %s failed: %s", position["ticket"], result.get("error"))
    if not failed:
        bot_state["breaker_flattened"] = True
        if bot_state.get("day_login"):
            _save_risk_state(int(bot_state["day_login"]))
    return closed


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


def _loss_cooldown_block(symbol: str, side: str) -> Optional[str]:
    """No new trade in the same direction right after a losing one on the same symbol."""
    minutes = config.LOSS_COOLDOWN_MINUTES
    if minutes <= 0:
        return None
    now = datetime.now(timezone.utc)
    for row in _history_and_stats()["ledger"]:
        if row.get("status") != "CLOSED" or row.get("outcome") != "LOSS" or row.get("side") != side:
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
                " scalp" if decision.get("strategy") == "SCALP" else "", decision["signal"],
                decision["confidence_score"], f" (penalties -{penalty})" if penalty else "",
                geometry, decision["logic"][:240])


async def process_symbol(symbol: str) -> str:
    """
    One symbol's cycle. Returns 'unchanged' (nothing new to judge), 'off_session', 'no_setup',
    'skipped' or 'evaluated'.
    """
    if config.STRATEGY_MODE == "SCALP":
        return await process_scalp(symbol)
    market = await _prepare_market(symbol)
    if market is None:
        return "skipped"
    if _evaluation_unchanged(symbol, market):
        return "unchanged"

    market["correlated_prices"] = await asyncio.to_thread(
        data_engine.fetch_correlated_asset_prices, symbol, config.SYMBOLS, True)
    decision = await asyncio.to_thread(ai_brain.get_ai_decision, market, symbol)
    _record_decision(symbol, market, decision)
    if not decision.get("error"):
        _remember_evaluation(symbol, market)
    _log_decision(symbol, decision)
    if decision.get("blocked_by"):
        with contextlib.suppress(Exception):
            await asyncio.to_thread(shadow_store.record_blocked, symbol, market, decision)
    return await _act_on_decision(symbol, market, decision)


# symbol -> server time of the last M5 bar the scalper evaluated (each closed bar is judged once)
_last_scalp_bar: Dict[str, int] = {}


def _scalp_count_today(symbol: str) -> int:
    today = data_engine.trading_day_key()
    count = 0
    for row in _history_and_stats()["ledger"]:
        if row.get("strategy") != "SCALP" or not data_engine.symbols_match(str(row.get("symbol") or ""), symbol):
            continue
        opened = _parse_utc(row.get("timestamp"))
        if opened is not None and data_engine.trading_day_key(opened) == today:
            count += 1
    return count


def _scalp_limit_block(symbol: str) -> Optional[str]:
    count = _scalp_count_today(symbol)
    if count >= config.SCALP_MAX_TRADES_PER_SYMBOL:
        return f"{count} scalps today (max {config.SCALP_MAX_TRADES_PER_SYMBOL} per symbol per trading day)"
    return None


def _evaluate_scalp(symbol: str, spread_price: float) -> Dict[str, Any]:
    """Fetch M5/M15/H1/D1 bars and run the scalper rules on the latest closed M5 bar (blocking)."""
    prepared = scalper.prepare(scalper.fetch_live_frames(symbol))
    result = scalper.evaluate(prepared, spread_price=spread_price)
    result["recent"] = scalper.recent_m5(prepared)
    return result


def _shadow_skipped_setup(symbol: str, market: Dict[str, Any], result: Dict[str, Any]) -> None:
    """A setup the strict guard skipped is followed on price data like any other blocked trade."""
    blocked = result["blocked_setup"]
    side = blocked["side"]
    digits = int(market.get("digits") or 5)
    tick = float(market.get("tick_size") or market.get("point") or 10 ** -digits)
    entry = float(market["ask"] if side == "BUY" else market["bid"])
    direction = 1.0 if side == "BUY" else -1.0
    decision = {
        "raw_signal": side, "blocked_by": [g["id"] for g in result.get("guards") or []],
        "stop_loss": data_engine.round_to_tick(entry - direction * blocked["sl_distance"], tick, digits),
        "take_profit": data_engine.round_to_tick(entry + direction * blocked["tp_distance"], tick, digits),
        "entry_reference": round(entry, digits), "risk_reward": blocked["risk_reward"], "strategy": "SCALP",
        "time_stop_minutes": config.SCALP_TIME_STOP_MINUTES, "base_confidence": None, "confidence_score": None,
    }
    with contextlib.suppress(Exception):
        shadow_store.record_blocked(symbol, market, decision)


async def process_scalp(symbol: str) -> str:
    """Scalp mode: rules look for an M5 pullback setup in the D1 trend; the AI confirms or vetoes it."""
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
    if setup is None and result.get("blocked_setup"):
        _shadow_skipped_setup(symbol, market, result)
    if setup is None:
        bot_state["decisions"][symbol] = {
            "signal": "HOLD", "raw_signal": "HOLD", "confidence": 0, "penalty": 0, "time": data_engine.utc_now_iso(),
            "error": None, "digits": market.get("digits"), "note": result["reason"], "strategy": "SCALP", **scan_view}
        bot_state.setdefault("scalp_reasons", {})[symbol] = result["reason"]
        return "no_setup"

    # Checks that would make the AI's answer irrelevant come first (no API call wasted).
    blocked = _scalp_limit_block(symbol) or _loss_cooldown_block(symbol, setup["side"])
    if blocked:
        logger.info("[SYSTEM] %s %s scalp setup skipped: %s", symbol, setup["side"], blocked)
        return "skipped"
    logger.info("[SYSTEM] %s setup found: %s", symbol, setup["reason"])
    market["correlated_prices"] = await asyncio.to_thread(
        data_engine.fetch_correlated_asset_prices, symbol, config.SYMBOLS, True)
    decision = await asyncio.to_thread(ai_brain.get_scalp_decision, market, symbol, setup, result["recent"])
    _record_decision(symbol, market, decision)
    bot_state["decisions"][symbol].update(scan_view, stage="AI_CONFIRMED" if decision["signal"] != "HOLD"
                                          else "AI_VETO" if "AI-VETO" in (decision.get("blocked_by") or [])
                                          else "BLOCKED" if not decision.get("error") else "AI_ERROR")
    _log_decision(symbol, decision)
    if decision.get("blocked_by"):
        with contextlib.suppress(Exception):
            await asyncio.to_thread(shadow_store.record_blocked, symbol, market, decision)
    return await _act_on_decision(symbol, market, decision)


def _scalp_time_stops() -> int:
    """Close the engine's scalps that have been open longer than SCALP_TIME_STOP_MINUTES (blocking)."""
    scalp_tickets = {int(row["ticket"]) for row in _history_and_stats()["ledger"]
                     if row.get("strategy") == "SCALP" and row.get("status") == "CONFIRMED" and row.get("ticket")}
    if not scalp_tickets:
        return 0
    closed = 0
    limit = config.SCALP_TIME_STOP_MINUTES * 60
    for position in execution.get_open_positions():
        if not position["managed"] or int(position["ticket"]) not in scalp_tickets:
            continue
        opened = _parse_utc(position.get("time"))
        now_server = data_engine.server_now_epoch(position["symbol"])
        if opened is None or now_server is None:
            continue
        age = now_server - opened.timestamp()  # both in broker server time
        if age < limit:
            continue
        logger.info("[TRADE] Time stop: %s %s ticket %s open %d min (limit %d), floating %.2f",
                    position["symbol"], position["side"], position["ticket"], int(age // 60),
                    config.SCALP_TIME_STOP_MINUTES, position["profit"])
        result = execution.close_position(position["ticket"])
        if result.get("success"):
            _record_close_execution(result, "TIME_STOP")
            closed += 1
        elif "not found" not in str(result.get("error", "")):
            logger.error("[TRADE] Time-stop close of ticket %s failed: %s", position["ticket"], result.get("error"))
    return closed


async def _act_on_decision(symbol: str, market: Dict[str, Any], decision: Dict[str, Any]) -> str:
    """Every portfolio and account check, then the order and its memory record."""
    signal = decision["signal"]
    scalp = decision.get("strategy") == "SCALP"
    if signal == "HOLD":
        return "evaluated"
    if not bot_state["is_running"]:
        logger.info("[SYSTEM] Engine stopped before %s %s could execute", signal, symbol)
        return "evaluated"
    if bot_state["circuit_breaker"]:
        logger.warning("[SYSTEM] %s %s blocked: daily loss circuit breaker is active", signal, symbol)
        return "evaluated"
    if bot_state["profit_target_hit"]:
        logger.info("[SYSTEM] %s %s blocked: daily profit target reached, new entries paused", signal, symbol)
        return "evaluated"
    cooldown = _loss_cooldown_block(symbol, signal)
    if cooldown:
        logger.info("[SYSTEM] %s %s blocked: %s", signal, symbol, cooldown)
        return "evaluated"

    positions = await asyncio.to_thread(execution.get_open_positions, symbol)
    same_side = [p for p in positions if p["side"] == signal]
    if same_side:
        logger.info("[DUPLICATE PREVENTED] %s %s already open (ticket %s, %.2f lots); no new order",
                    symbol, signal, same_side[0]["ticket"], same_side[0]["volume"])
        return "evaluated"

    opposite = [p for p in positions if p["side"] != signal]
    if opposite:
        unmanaged = [p for p in opposite if not p["managed"]]
        if unmanaged:
            logger.warning("[REVERSAL] %s %s skipped: opposite position(s) %s were not opened by this engine "
                           "and are left untouched", symbol, signal, [p["ticket"] for p in unmanaged])
            return "evaluated"
        required = int(decision.get("threshold") or config.CONFIDENCE_THRESHOLD) + config.REVERSAL_EXTRA_CONFIDENCE
        if decision["confidence_score"] < required:
            logger.info("[REVERSAL] %s %s not flipped: confidence %d < %d needed to close the open %s "
                        "(threshold + %d)", symbol, signal, decision["confidence_score"], required,
                        opposite[0]["side"], config.REVERSAL_EXTRA_CONFIDENCE)
            return "evaluated"

    if config.MAX_OPEN_POSITIONS > 0:
        all_positions = await asyncio.to_thread(execution.get_open_positions)
        if len(all_positions) - len(opposite) >= config.MAX_OPEN_POSITIONS:
            logger.warning("[SYSTEM] %s %s skipped: %d open positions (max %d)",
                           signal, symbol, len(all_positions), config.MAX_OPEN_POSITIONS)
            return "evaluated"

    risk_percent = float(bot_state["risk_percent"])
    order_args = dict(sl_distance=decision.get("sl_distance"), tp_distance=decision.get("tp_distance"),
                      entry_reference=decision.get("entry_reference"), atr_reference=decision.get("atr_reference"),
                      max_total_open_risk=_daily_risk_budget(),
                      max_drift_atr=config.SCALP_MAX_DRIFT_ATR if scalp else None)

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
            return "evaluated"
        for position in opposite:
            logger.info("[REVERSAL] %s flipped to %s: closing %s ticket %s (%.2f lots, floating %.2f)",
                        symbol, signal, position["side"], position["ticket"], position["volume"], position["profit"])
            result = await asyncio.to_thread(execution.close_position, position["ticket"])
            if not result.get("success"):
                logger.error("[REVERSAL] Could not close ticket %s (%s); new %s aborted",
                             position["ticket"], result.get("error"), signal)
                return "evaluated"
            await asyncio.to_thread(_record_close_execution, result, "REVERSAL", market)
        remaining = await asyncio.to_thread(execution.get_open_positions, symbol)
        if any(p["side"] != signal for p in remaining):
            logger.error("[REVERSAL] %s opposite exposure still open after close; new %s aborted", symbol, signal)
            return "evaluated"

    try:
        trade = await asyncio.to_thread(execution.execute_trade, symbol, signal, decision["stop_loss"],
                                        decision["take_profit"], risk_percent,
                                        comment=f"{config.ORDER_COMMENT}-S" if scalp else None, **order_args)
    except execution.TradeExecutionError as exc:
        bot_state["last_error"] = f"{symbol} {signal}: {exc}"
        logger.error("[TRADE] %s %s not executed: %s", symbol, signal, exc)
        return "evaluated"

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
    await asyncio.to_thread(_refresh_account_state)
    return "evaluated"


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
        logger.info("[SYSTEM] Waiting for the scalping session: %s. The engine keeps watching positions, "
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
            logger.info("[SYSTEM] Scalp session %s: %s", "OPEN" if session_open else "CLOSED", scalper.session_note())
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
        logger.info("[SYSTEM] Scalping session: %s", scalper.session_note())
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
    scalp_session_start_london: Optional[int] = Field(default=None, ge=0, le=23)
    scalp_session_end_new_york: Optional[int] = Field(default=None, ge=1, le=17)
    scalp_strict_guard: Optional[bool] = None
    scalp_medium_trend: Optional[bool] = None


SETTINGS_BOUNDS = {
    "risk_percent": [0.05, config.MAX_RISK_PERCENT], "fixed_lot": [0.01, 100.0], "max_open_positions": [0, 50],
    "max_daily_loss_percent": [0.0, 50.0], "max_daily_profit_percent": [0.0, 100.0],
    "confidence_threshold": [50, 95], "scan_interval_seconds": [5, 86_400],
    "symbols_demo": [1, config.MAX_SYMBOLS], "symbols_live": [1, config.MAX_SYMBOLS],
    "weekend_entry_cutoff_hours": [0.0, 48.0], "max_currency_risk_percent": [0.0, 50.0],
    "loss_cooldown_minutes": [0, 10_080], "scalp_time_stop_minutes": [5, 1440], "scalp_max_trades_per_symbol": [1, 50],
    "scalp_session_start_london": [0, 23], "scalp_session_end_new_york": [1, 17],
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


class BacktestRequest(BaseModel):
    days: int = Field(default=60, ge=5, le=365)
    extra_spread_pips: float = Field(default=0.0, ge=0.0, le=20.0)


async def _run_backtest_async(days: int, extra_spread_pips: float = 0.0) -> None:
    symbols = list(config.SYMBOLS)
    state = {"running": True, "progress": "starting", "started_at": data_engine.utc_now_iso(), "days": days,
             "symbols": symbols, "error": None}
    bot_state["backtest"] = state
    logger.info("[BACKTEST] Scalp backtest started: %d days, %d symbols", days, len(symbols))
    try:
        report = await asyncio.to_thread(
            backtest.run_backtest, symbols, days, float(bot_state["risk_percent"]),
            float(bot_state.get("equity") or 100.0), lambda message: state.update(progress=message),
            extra_spread_pips)
    except Exception as exc:
        logger.exception("[BACKTEST] Backtest failed: %s", exc)
        state.update(running=False, error=str(exc), progress="failed")
        return
    summary = report["summary"]
    state.update(running=False, progress="done", file=report.get("file"), verdict=report["verdict"],
                 finished_at=data_engine.utc_now_iso())
    logger.info("[BACKTEST] Done: %s trades, win rate %s%%, expectancy %sR, balance %s -> %s | %s",
                summary.get("trades"), summary.get("win_rate"), summary.get("expectancy_r"),
                report["params"]["start_balance"], report["account"]["final_balance"], report["verdict"])


@app.post("/api/backtest")
async def api_run_backtest(body: Optional[BacktestRequest] = None) -> Dict[str, Any]:
    if (bot_state.get("backtest") or {}).get("running"):
        raise HTTPException(status_code=409, detail="a backtest is already running")
    if not bot_state.get("mt5_connected"):
        raise HTTPException(status_code=503, detail="MT5 is not connected; the backtest needs its price history")
    request = body or BacktestRequest()
    _spawn(_run_backtest_async(request.days, request.extra_spread_pips))
    return {"started": True}


@app.get("/api/backtest")
def api_backtest() -> Dict[str, Any]:
    report = backtest.latest_report()
    if report is not None:
        report = {key: value for key, value in report.items() if key != "trades"}
        report["recent_trades"] = (backtest.latest_report() or {}).get("trades", [])[-40:]
    return {"state": bot_state.get("backtest") or {}, "report": report}


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
        if {"ai_new_bar_only", "strategy_mode"} & changed.keys():
            _last_evaluation.clear()
            _last_scalp_bar.clear()
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
