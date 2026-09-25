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
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import ai_brain
import auditor
import config
import data_engine
import execution
import memory_store


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
    "circuit_breaker": False,
    "last_error": None,
    "started_at": data_engine.utc_now_iso(),
}

templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))
_background_tasks: Set[asyncio.Task] = set()
_server: Optional[uvicorn.Server] = None
_history_cache: Dict[str, Any] = {"stamp": None, "history": [], "stats": {}}
HOUSEKEEPING_SECONDS = 60


# -----------------------------------------------------------------------------
# State refresh helpers (blocking: call from a worker thread)
# -----------------------------------------------------------------------------
def _update_daily_risk(equity: float) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if bot_state["day_key"] != today or not bot_state["day_start_equity"]:
        bot_state.update(day_key=today, day_start_equity=equity, circuit_breaker=False, daily_drawdown_pct=0.0)
        return
    start = float(bot_state["day_start_equity"])
    drawdown = max(0.0, (start - equity) / start * 100.0) if start > 0 else 0.0
    bot_state["daily_drawdown_pct"] = round(drawdown, 3)
    limit = config.MAX_DAILY_LOSS_PERCENT
    if limit > 0 and drawdown >= limit and not bot_state["circuit_breaker"]:
        bot_state["circuit_breaker"] = True
        logger.warning("[SYSTEM] DAILY LOSS CIRCUIT BREAKER TRIPPED: equity %.2f is %.2f%% below today's "
                       "open %.2f (limit %.2f%%). New entries halted until the next UTC day.",
                       equity, drawdown, start, limit)


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
    closed = [r for r in records if r.get("status") == "CLOSED"]
    wins = sum(1 for r in closed if r.get("outcome") == "WIN")
    losses = sum(1 for r in closed if r.get("outcome") == "LOSS")
    history = [
        {key: record.get(key) for key in (
            "id", "timestamp", "symbol", "side", "entry_price", "stop_loss", "take_profit", "volume",
            "risk_percent", "deal", "order", "ticket", "status", "outcome", "realized_pnl", "exit_price",
            "exit_time", "exit_reason", "confidence_score", "digits")}
        for record in reversed(records[-60:])
    ]
    _history_cache.update(
        stamp=stamp,
        history=history,
        stats={
            "fills": len(records),
            "open": sum(1 for r in records if r.get("status") == "CONFIRMED"),
            "closed": len(closed),
            "wins": wins,
            "losses": losses,
            "breakeven": len(closed) - wins - losses,
            "win_rate": round(wins / len(closed) * 100.0, 1) if closed else None,
            "realized_pnl": round(sum(float(r.get("realized_pnl") or 0.0) for r in closed), 2),
        },
    )
    return _history_cache


def _refresh_learning_state() -> None:
    document = auditor.load_rules_document()
    bot_state["learned_rules"] = document["rules"]
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
            account={k: account[k] for k in ("login", "server", "company", "name", "leverage",
                                             "margin", "margin_free", "margin_level", "trade_allowed")},
        )
        _update_daily_risk(account["equity"])
        try:
            bot_state["open_positions"] = execution.get_open_positions()
        except Exception as exc:
            logger.debug("[SYSTEM] Position refresh failed: %s", exc)
    else:
        bot_state["mt5_connected"] = False
    cache = _history_and_stats()
    bot_state["trade_history"] = cache["history"]
    bot_state["stats"] = cache["stats"]
    _refresh_learning_state()


def _status_payload() -> Dict[str, Any]:
    payload = dict(bot_state)
    payload["decisions"] = dict(bot_state["decisions"])  # the trading loop mutates this dict
    payload["logs"] = list(LOG_BUFFER.records)[-120:]
    payload["server_time"] = data_engine.utc_now_iso()
    payload["guardrails"] = {
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "max_risk_percent": config.MAX_RISK_PERCENT,
        "max_open_positions": config.MAX_OPEN_POSITIONS,
        "max_daily_loss_percent": config.MAX_DAILY_LOSS_PERCENT,
        "sl_atr_multiplier": config.SL_ATR_MULTIPLIER,
        "tp_atr_multiplier": config.TP_ATR_MULTIPLIER,
        "magic_number": config.MAGIC_NUMBER,
        "audit_cooldown_hours": config.AUDIT_COOLDOWN_HOURS,
        "audit_trade_threshold": config.AUDIT_TRADE_THRESHOLD,
        "min_lot_risk_tolerance": config.MIN_LOT_RISK_TOLERANCE,
    }
    payload["symbols"] = config.SYMBOLS
    payload["ai_configured"] = bool(config.DEEPSEEK_API_KEY)
    payload["model"] = config.DEEPSEEK_MODEL
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
        last_decision_at=decision.get("timestamp"),
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
    }


def _build_trade_record(symbol: str, market: Dict[str, Any], decision: Dict[str, Any],
                        trade: Dict[str, Any], risk_percent: float) -> Dict[str, Any]:
    def timeframe_context(data: Dict[str, Any]) -> Dict[str, Any]:
        return {key: data.get(key) for key in (
            "last_bar_time", "close", "ema200", "price_vs_ema", "ema_slope", "ema_distance_atr", "rsi14",
            "atr14", "atr_pct", "atr_ratio", "tick_volume", "volume_ma20", "rel_volume", "structure")}

    return {
        "status": "CONFIRMED",
        "timestamp": data_engine.utc_now_iso(),
        "account_login": market.get("account_login"),
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
        "applied_rules": decision.get("applied_rules", []),
        "stops_source": decision.get("stops_source"),
        "logic": decision["logic"],
        "market_context": {
            "bid": market["bid"],
            "ask": market["ask"],
            "spread_points": market["spread"],
            "equity": trade.get("equity_at_entry", market["equity"]),
            "balance": market["balance"],
            "digits": market["digits"],
            "h1": timeframe_context(market["h1_data"]),
            "d1": timeframe_context(market["daily_data"]),
            "correlated_prices": market.get("correlated_prices", {}),
        },
    }


async def process_symbol(symbol: str) -> None:
    try:
        market = await asyncio.to_thread(data_engine.fetch_multi_timeframe_data, symbol)
    except data_engine.DataEngineError as exc:
        logger.warning("[SYSTEM] %s skipped: %s", symbol, exc)
        return
    if not market["tradeable"]:
        logger.info("[SYSTEM] %s skipped: trading disabled/close-only at the broker", symbol)
        return
    if market["market_idle"]:
        logger.info("[SYSTEM] %s skipped: no new ticks for %ds (market closed)", symbol, config.MARKET_IDLE_SECONDS)
        return

    market["correlated_prices"] = await asyncio.to_thread(
        data_engine.fetch_correlated_asset_prices, symbol, config.SYMBOLS)
    decision = await asyncio.to_thread(ai_brain.get_ai_decision, market, symbol)
    _record_decision(symbol, market, decision)

    penalty = sum(rule.get("points", 0) for rule in decision.get("applied_rules", []))
    logger.info("[AI] %s -> %s | confidence %d%s | SL %s TP %s | %s", symbol, decision["signal"],
                decision["confidence_score"], f" (rules -{penalty})" if penalty else "",
                decision.get("stop_loss"), decision.get("take_profit"), decision["logic"][:220])

    signal = decision["signal"]
    if signal == "HOLD":
        return
    if not bot_state["is_running"]:
        logger.info("[SYSTEM] Engine stopped before %s %s could execute", signal, symbol)
        return
    if bot_state["circuit_breaker"]:
        logger.warning("[SYSTEM] %s %s blocked: daily loss circuit breaker is active", signal, symbol)
        return

    positions = await asyncio.to_thread(execution.get_open_positions, symbol)
    same_side = [p for p in positions if p["side"] == signal]
    if same_side:
        logger.info("[DUPLICATE PREVENTED] %s %s already open (ticket %s, %.2f lots); no new order",
                    symbol, signal, same_side[0]["ticket"], same_side[0]["volume"])
        return

    opposite = [p for p in positions if p["side"] != signal]
    if opposite:
        unmanaged = [p for p in opposite if not p["managed"]]
        if unmanaged:
            logger.warning("[REVERSAL] %s %s skipped: opposite position(s) %s were not opened by this engine "
                           "and are left untouched", symbol, signal, [p["ticket"] for p in unmanaged])
            return
        for position in opposite:
            logger.info("[REVERSAL] %s flipped to %s: closing %s ticket %s (%.2f lots, floating %.2f)",
                        symbol, signal, position["side"], position["ticket"], position["volume"], position["profit"])
            result = await asyncio.to_thread(execution.close_position, position["ticket"])
            if not result.get("success"):
                logger.error("[REVERSAL] Could not close ticket %s (%s); new %s aborted",
                             position["ticket"], result.get("error"), signal)
                return
        remaining = await asyncio.to_thread(execution.get_open_positions, symbol)
        if any(p["side"] != signal for p in remaining):
            logger.error("[REVERSAL] %s opposite exposure still open after close; new %s aborted", symbol, signal)
            return

    if config.MAX_OPEN_POSITIONS > 0:
        all_positions = await asyncio.to_thread(execution.get_open_positions)
        if len(all_positions) >= config.MAX_OPEN_POSITIONS:
            logger.warning("[SYSTEM] %s %s skipped: %d open positions (max %d)",
                           signal, symbol, len(all_positions), config.MAX_OPEN_POSITIONS)
            return

    risk_percent = float(bot_state["risk_percent"])
    try:
        trade = await asyncio.to_thread(execution.execute_trade, symbol, signal, decision["stop_loss"],
                                        decision["take_profit"], risk_percent)
    except execution.TradeExecutionError as exc:
        bot_state["last_error"] = f"{symbol} {signal}: {exc}"
        logger.error("[TRADE] %s %s not executed: %s", symbol, signal, exc)
        return

    logger.info("[TRADE] FILLED %s %s %.2f lots @ %s | SL %s | TP %s | deal %s | position %s | risk %.2f%%",
                signal, symbol, trade["volume"], trade["price"], trade["stop_loss"], trade["take_profit"],
                trade["deal"], trade["position_ticket"], risk_percent)
    record = _build_trade_record(symbol, market, decision, trade, risk_percent)
    try:
        await asyncio.to_thread(memory_store.append_trade_memory, record)
    except Exception as exc:
        logger.error("[LEARNING] Trade filled but memory write failed: %s", exc)
    await asyncio.to_thread(_refresh_account_state)


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


async def _learning_cycle() -> None:
    """Reconcile closed trades and audit when warranted."""
    try:
        newly_closed = await asyncio.to_thread(memory_store.reconcile_closed_trades)
    except Exception as exc:
        logger.warning("[LEARNING] Reconciliation failed: %s", exc)
        return
    if newly_closed:
        logger.info("[LEARNING] %d newly closed trade(s) reconciled; forcing an audit", newly_closed)
        await _run_audit_async(force=True)
        return
    due = await asyncio.to_thread(auditor.audit_due)
    if due["due"]:
        logger.info("[LEARNING] Audit due: %s", due["reason"])
        await _run_audit_async(force=due["force"])


async def run_scan_cycle() -> None:
    bot_state["scan_count"] += 1
    scan_number = bot_state["scan_count"]
    connected = await asyncio.to_thread(data_engine.ensure_connection)
    bot_state["mt5_connected"] = connected
    if not connected:
        logger.warning("[SYSTEM] Scan #%d skipped: MT5 is not connected", scan_number)
        return
    await asyncio.to_thread(_refresh_account_state)
    logger.info("[SYSTEM] Scan #%d started | %d symbols | risk %.2f%% | equity %.2f %s",
                scan_number, len(config.SYMBOLS), bot_state["risk_percent"], bot_state["equity"], bot_state["currency"])

    for symbol in list(config.SYMBOLS):
        if not bot_state["is_running"]:
            logger.info("[SYSTEM] Engine stopped mid-scan; remaining symbols skipped")
            break
        bot_state["current_symbol"] = symbol
        try:
            await process_symbol(symbol)
        except Exception as exc:
            logger.exception("[SYSTEM] %s processing failed: %s", symbol, exc)
    bot_state["current_symbol"] = None

    await _learning_cycle()
    await asyncio.to_thread(_refresh_account_state)
    bot_state["last_scan_at"] = data_engine.utc_now_iso()


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
            await run_scan_cycle()
            finished = time.monotonic()
            last_housekeeping = finished
            bot_state["last_scan_duration"] = round(finished - started, 1)
            logger.info("[SYSTEM] Scan #%d finished in %.1fs; next scan in %ds",
                        bot_state["scan_count"], finished - started, bot_state["interval"])

            # Sleep for the configured interval, re-reading it every second so
            # interval changes and Stop requests from the dashboard apply immediately.
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
    logger.info("[SYSTEM] Autonomous AI Hedge Fund booting | %d symbols | threshold %d | risk %.2f%% | "
                "interval %ds | model %s", len(config.SYMBOLS), config.CONFIDENCE_THRESHOLD,
                config.DEFAULT_RISK_PERCENT, config.SCAN_INTERVAL_SECONDS, config.DEEPSEEK_MODEL)
    if not config.DEEPSEEK_API_KEY:
        logger.warning("[SYSTEM] DEEPSEEK_API_KEY is missing: every decision will be HOLD until it is set in .env")

    connected = await asyncio.to_thread(data_engine.initialize_mt5)
    bot_state["mt5_connected"] = connected
    if connected:
        try:
            reconciled = await asyncio.to_thread(memory_store.reconcile_closed_trades)
            logger.info("[LEARNING] Startup reconciliation: %d trade(s) closed while offline", reconciled)
        except Exception as exc:
            logger.warning("[LEARNING] Startup reconciliation failed: %s", exc)
        await asyncio.to_thread(_refresh_account_state)
    else:
        logger.warning("[SYSTEM] Starting without MT5; the engine retries the connection on every scan")

    await asyncio.to_thread(_refresh_learning_state)
    active = [rule for rule in bot_state["learned_rules"] if rule.get("status") == "ACTIVE"]
    logger.info("[LEARNING] %d active learned rule(s) loaded from %s", len(active), config.RULES_FILE.name)

    due = await asyncio.to_thread(auditor.audit_due)
    if due["due"]:
        logger.info("[AUDITOR] Startup audit due: %s", due["reason"])
        _spawn(_run_audit_async(force=due["force"]))
    else:
        logger.info("[AUDITOR] No startup audit needed: %s", due.get("reason"))

    if config.AUTO_START_ENGINE:
        bot_state["is_running"] = True
        logger.info("[SYSTEM] AUTO_START_ENGINE=true: engine started automatically")

    loop_task = asyncio.create_task(trading_loop(), name="trading_loop")
    logger.info("[SYSTEM] Dashboard ready at http://%s:%d", config.HOST, config.PORT)
    try:
        yield
    finally:
        bot_state["is_running"] = False
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task
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
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
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
    if body.interval is not None and body.interval != bot_state["interval"]:
        bot_state["interval"] = body.interval
        changes.append(f"interval={body.interval}s")
    if body.risk_percent is not None:
        risk = round(float(body.risk_percent), 2)
        if risk != bot_state["risk_percent"]:
            bot_state["risk_percent"] = risk
            changes.append(f"risk={risk}%")
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
    with contextlib.suppress(Exception):
        _refresh_account_state()
    return result


@app.post("/api/audit")
async def api_audit(body: Optional[AuditRequest] = None) -> Dict[str, Any]:
    force = body.force if body is not None else True
    logger.info("[AUDITOR] Audit requested from dashboard (force=%s)", force)
    return await _run_audit_async(force)


@app.get("/api/rules")
def api_rules() -> Dict[str, Any]:
    return auditor.load_rules_document()


@app.post("/api/rules/{rule_id}/toggle")
def api_toggle_rule(rule_id: str) -> Dict[str, Any]:
    document = auditor.load_rules_document()
    rule = next((r for r in document["rules"] if str(r.get("id", "")).upper() == rule_id.upper()), None)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"rule {rule_id} not found")
    new_status = "DISABLED" if rule.get("status") == "ACTIVE" else "ACTIVE"
    updated = auditor.set_rule_status(rule_id, new_status)
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
