"""
Shadow trades: the trades that learned rules, guards, the AI or the learning agent did not take.

When penalties turn a signal that would have traded into a HOLD, the trade that
was not taken is recorded here with its exact entry, stop and target. It is then
followed on M1 price data until the stop or the target is hit. This shows whether
a blocking rule actually avoids losses; a rule whose blocked trades would have
made money is retired (see auditor.maintain_rules). Without this, a rule that
stops a pair from trading could never be proven wrong.

Every resolved shadow trade is also a reward for the learning agent (ml/agent.py). Its
virtual exploration trades (near-miss setups) live in a separate file, config.EXPLORE_FILE,
so they never feed the auditor or the calibration.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import MetaTrader5 as mt5

import config
from data_engine import MT5_LOCK, utc_now_iso
from memory_store import file_lock, read_json_file, write_json_atomic

logger = logging.getLogger("hedgefund.shadow")

MAX_SHADOWS = 2000
DUPLICATE_WINDOW_MINUTES = 60
EXPLORE_DUPLICATE_MINUTES = 20  # exploration: the same pair and side may be followed again after this long


def tradeable_cost(spread: Any, entry: Any, stop: Any) -> bool:
    """False when the spread is more than MAX_SPREAD_TO_STOP of the stop distance (the engine's own hard limit)."""
    try:
        distance = abs(float(entry) - float(stop))
        return distance > 0 and float(spread or 0.0) <= config.MAX_SPREAD_TO_STOP * distance
    except (TypeError, ValueError):
        return True  # unknown: do not block


def load_shadows(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    data = read_json_file(path or config.SHADOW_FILE, list)
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _parse(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def record_blocked(symbol: str, market: Dict[str, Any], decision: Dict[str, Any],
                   path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Store a trade that was not taken. Returns the entry, or None when not recordable or a duplicate.
    Exploration setups (decision["agent"]["context"]["explore"]) go to config.EXPLORE_FILE.
    """
    agent = decision.get("agent") or None
    if path is None:
        path = config.EXPLORE_FILE if ((agent or {}).get("context") or {}).get("explore") else config.SHADOW_FILE
    side = decision.get("raw_signal")
    blocked_by = decision.get("blocked_by") or []
    if side not in ("BUY", "SELL") or not blocked_by or not decision.get("stop_loss") or not decision.get("take_profit"):
        return None
    if not tradeable_cost(market.get("spread_price"), decision.get("entry_reference"), decision.get("stop_loss")):
        logger.debug("[LEARNING] %s %s not followed: spread too large for the stop", side, symbol)
        return None  # a real order would be refused for its cost: its result would only teach "spreads lose"
    now = datetime.now(timezone.utc)
    entry = {
        "id": uuid.uuid4().hex[:12],
        "status": "OPEN",
        "created_at": now.isoformat(timespec="seconds"),
        "server_epoch": market.get("tick_epoch"),
        "symbol": symbol,
        "side": side,
        "entry_price": decision.get("entry_reference"),
        "stop_loss": decision["stop_loss"],
        "take_profit": decision["take_profit"],
        "risk_reward": decision.get("risk_reward"),
        "spread_price": market.get("spread_price") or 0.0,
        "blocked_by": list(blocked_by),
        "base_confidence": decision.get("base_confidence"),
        "confidence_score": decision.get("confidence_score"),
        "account_mode": market.get("account_mode"),
        "broker": market.get("broker"),
        "strategy": decision.get("strategy") or "SWING",
        "time_stop_minutes": decision.get("time_stop_minutes"),
        # the same market snapshot a real trade stores, so the auditor can test rule conditions on it
        "market_context": market_snapshot(market),
        "features": decision.get("features"),
        "agent": agent,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        shadows = load_shadows(path)
        window = timedelta(minutes=int(decision.get("time_stop_minutes") or DUPLICATE_WINDOW_MINUTES))
        if path == config.EXPLORE_FILE:
            window = min(window, timedelta(minutes=EXPLORE_DUPLICATE_MINUTES))
        for other in reversed(shadows[-200:]):
            created = _parse(other.get("created_at"))
            if (other.get("symbol") == symbol and other.get("side") == side and other.get("status") == "OPEN"
                    and (other.get("strategy") or "SWING") == entry["strategy"] and created and now - created < window):
                return None  # the same blocked idea is already being followed
        shadows.append(entry)
        write_json_atomic(path, shadows[-MAX_SHADOWS:])
    if path == config.EXPLORE_FILE:
        logger.debug("[AGENT] Exploration trade recorded: %s %s %s (SL %s / TP %s)", entry["strategy"].lower(),
                     side, symbol, entry["stop_loss"], entry["take_profit"])
    else:
        logger.info("[LEARNING] Shadow trade recorded: %s %s blocked by %s (SL %s / TP %s)",
                    side, symbol, ", ".join(blocked_by), entry["stop_loss"], entry["take_profit"])
    return entry


_TF_KEYS = ("ema_distance_atr", "rsi14", "atr_pct", "atr_ratio", "rel_volume", "tick_volume", "volume_ma20")


def market_snapshot(market: Dict[str, Any]) -> Dict[str, Any]:
    """The rule-relevant market state (same fields as a trade record's market_context)."""
    from ai_brain import usd_direction  # local import: ai_brain does not import this module
    correlated = market.get("correlated_prices") or {}
    return {
        "h1": {k: (market.get("h1_data") or {}).get(k) for k in _TF_KEYS},
        "d1": {k: (market.get("daily_data") or {}).get(k) for k in _TF_KEYS},
        "day_change_pct": market.get("day_change_pct"),
        "correlated_prices": {s: {"day_change_pct": q.get("day_change_pct"), "corr_h1": q.get("corr_h1")}
                              for s, q in correlated.items()},
        "usd_direction": usd_direction(market) if correlated else None,
        "equity": market.get("equity"),
    }


def learning_records(mode: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Resolved shadow trades shaped like closed trade records, for the auditor and calibration.
    Each carries its result in R and a weight (config.SHADOW_WEIGHT) below a real trade's 1.0.
    """
    rows = []
    for shadow in load_shadows():
        if shadow.get("status") not in ("WIN", "LOSS", "TIMEOUT") or shadow.get("r_multiple") is None:
            continue
        if mode and shadow.get("account_mode") and shadow["account_mode"] != mode:
            continue
        r = float(shadow["r_multiple"])
        rows.append({
            "id": f"shadow:{shadow['id']}", "source": "shadow", "status": "CLOSED",
            "symbol": shadow.get("symbol"), "side": shadow.get("side"),
            "outcome": "WIN" if r > 0 else "LOSS" if r < 0 else "BREAKEVEN", "r": r, "realized_pnl": None,
            "weight": config.SHADOW_WEIGHT, "blocked_by": shadow.get("blocked_by") or [],
            "confidence_score": shadow.get("confidence_score"), "base_confidence": shadow.get("base_confidence"),
            "account_mode": shadow.get("account_mode"), "broker": shadow.get("broker"),
            "timestamp": shadow.get("created_at"), "reconciled_at": shadow.get("resolved_at"),
            "exit_time": shadow.get("resolved_at"), "exit_reason": shadow.get("status"),
            "market_context": shadow.get("market_context") or {},
        })
    rows.sort(key=lambda row: str(row.get("reconciled_at") or ""))
    return rows[-limit:]


def _outcome(shadow: Dict[str, Any], bars: Any) -> Optional[tuple]:
    """
    (status, R) from M1 bars (bid prices) after the entry; a bar that touches both counts as LOSS.
    With a time stop, an unresolved trade is closed at the last bar inside the window (TIMEOUT).
    """
    start = float(shadow.get("server_epoch") or 0)
    spread = float(shadow.get("spread_price") or 0.0)
    sl, tp = float(shadow["stop_loss"]), float(shadow["take_profit"])
    entry = float(shadow.get("entry_price") or 0.0)
    time_stop = float(shadow.get("time_stop_minutes") or 0) * 60
    first = start - (start % 60) + 60 if start else 0  # only whole minutes after the decision
    last_bar = None
    for bar in bars:
        bar_time = float(bar["time"])
        if bar_time < first:
            continue
        if time_stop and bar_time >= first + time_stop:
            break
        last_bar = bar
        high, low = float(bar["high"]), float(bar["low"])
        if shadow["side"] == "BUY":  # a long exits at the bid
            hit_sl, hit_tp = low <= sl, high >= tp
        else:  # a short exits at the ask = bid + spread
            hit_sl, hit_tp = high + spread >= sl, low + spread <= tp
        if hit_sl:
            return "LOSS", -1.0
        if hit_tp:
            return "WIN", float(shadow.get("risk_reward") or 1.0)
    window_over = time_stop and bars is not None and len(bars) and float(bars[-1]["time"]) >= first + time_stop
    if window_over and last_bar is not None and entry and abs(entry - sl) > 0:
        close = float(last_bar["close"]) + (spread if shadow["side"] == "SELL" else 0.0)
        moved = (close - entry) if shadow["side"] == "BUY" else (entry - close)
        return "TIMEOUT", round(moved / abs(entry - sl), 3)
    return None


def resolve_open_shadows(path: Optional[Path] = None) -> int:
    """Follow OPEN shadow trades on M1 data; mark WIN/LOSS, or EXPIRED after SHADOW_MAX_DAYS. Returns the count."""
    path = path or config.SHADOW_FILE
    shadows = load_shadows(path)
    open_ones = [s for s in shadows if s.get("status") == "OPEN"]
    if not open_ones:
        return 0
    now = datetime.now(timezone.utc)
    results: Dict[str, Dict[str, Any]] = {}
    ages: Dict[str, float] = {}
    for shadow in open_ones:
        created = _parse(shadow.get("created_at")) or now
        ages[shadow["id"]] = (now - created).total_seconds() / 60.0
    # One M1 fetch per symbol, long enough for its oldest open shadow.
    needed: Dict[str, int] = {}
    for shadow in open_ones:
        if ages[shadow["id"]] >= 1:
            bars = int(min(ages[shadow["id"]] + 5, config.SHADOW_MAX_DAYS * 1440 + 5))
            needed[shadow["symbol"]] = max(needed.get(shadow["symbol"], 0), bars)
    rates: Dict[str, Any] = {}
    for symbol, count in needed.items():
        with MT5_LOCK:
            try:
                rates[symbol] = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
            except Exception:
                rates[symbol] = None
    for shadow in open_ones:
        age_minutes = ages[shadow["id"]]
        if age_minutes < 1:
            continue
        bars = rates.get(shadow["symbol"])
        outcome = _outcome(shadow, bars) if bars is not None and shadow.get("server_epoch") else None
        if outcome:
            results[shadow["id"]] = {"status": outcome[0], "r_multiple": outcome[1], "resolved_at": utc_now_iso()}
        elif age_minutes > config.SHADOW_MAX_DAYS * 1440:
            results[shadow["id"]] = {"status": "EXPIRED", "r_multiple": 0.0, "resolved_at": utc_now_iso()}
    if not results:
        return 0
    with file_lock(path):
        shadows = load_shadows(path)
        for shadow in shadows:
            if shadow.get("id") in results and shadow.get("status") == "OPEN":
                shadow.update(results[shadow["id"]])
        write_json_atomic(path, shadows)
    if path == config.EXPLORE_FILE:
        return len(results)
    for shadow in shadows:
        if shadow.get("id") in results:
            logger.info("[LEARNING] Shadow %s %s (blocked by %s) -> %s", shadow["side"], shadow["symbol"],
                        ", ".join(shadow.get("blocked_by") or []), shadow["status"])
    return len(results)


def stats_by_blocker(shadows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """Per rule/guard id: how the trades it blocked would have done."""
    stats: Dict[str, Dict[str, Any]] = {}
    for shadow in shadows if shadows is not None else load_shadows():
        for blocker in shadow.get("blocked_by") or []:
            entry = stats.setdefault(blocker, {"blocked": 0, "open": 0, "wins": 0, "losses": 0, "timeouts": 0,
                                               "r_total": 0.0})
            entry["blocked"] += 1
            status = shadow.get("status")
            if status == "OPEN":
                entry["open"] += 1
            elif status in ("WIN", "LOSS", "TIMEOUT"):
                entry[{"WIN": "wins", "LOSS": "losses", "TIMEOUT": "timeouts"}[status]] += 1
                entry["r_total"] += float(shadow.get("r_multiple") or 0.0)
    for entry in stats.values():
        resolved = entry["wins"] + entry["losses"] + entry["timeouts"]
        entry["resolved"] = resolved
        entry["expectancy_r"] = round(entry["r_total"] / resolved, 2) if resolved else None
        entry["r_total"] = round(entry["r_total"], 2)
    return stats
