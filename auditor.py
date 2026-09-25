"""
Self-learning risk auditor.

Reviews recently closed trades, asks DeepSeek (as Chief Risk Officer and
quantitative auditor) to find recurring loss patterns, and persists them in
new_rules.json as confidence penalties that ai_brain injects into every
future decision for the affected symbols.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import config
from ai_brain import DeepSeekError, deepseek_chat, parse_json_payload
from data_engine import utc_now_iso
from memory_store import load_trade_memory, read_json_file, reconcile_closed_trades, write_json_atomic

logger = logging.getLogger("hedgefund.auditor")

_AUDIT_LOCK = threading.Lock()

AUDITOR_SYSTEM_PROMPT = """You are the Chief Risk Officer and Quantitative Auditor of a systematic hedge fund.
You review the fund's closed trades and write binding risk rules that cut confidence on setups that
keep losing money. You are skeptical, evidence-driven and allergic to curve-fitting.

Hunt for RECURRING loss patterns, in particular:
1. Low-volume breakouts/breakdowns: entries taken while h1.rel_volume (or d1.rel_volume) was weak.
2. Overextended ATR / volatility traps: high atr_ratio, large |ema_distance_atr|, RSI extremes,
   entries after the move already happened.
3. Adverse cross-asset correlation: e.g. buying EURUSD/GBPUSD/AUDUSD while USD strengthened across
   USDJPY/USDCHF/USDCAD, or gold/BTC trades fighting the prevailing risk tone.
4. Counter-trend entries against the D1 EMA200 bias, or poor timing (stops hit quickly).

RULE REQUIREMENTS
- Each rule must be supported by at least 2 losing trades; sample_size = number of losing trades matching.
- Compare against the winning trades provided: if the same condition is just as common among winners,
  it is not a loss pattern; do not emit it.
- "setup" must be a precise, machine-checkable condition written with the field names the trading desk
  sees: side (BUY/SELL), h1.rel_volume, d1.rel_volume, h1.atr_ratio, h1.atr_pct, h1.rsi14, d1.rsi14,
  h1.ema_distance_atr, d1.price_vs_ema, d1.ema_slope, h1/d1 structure pattern, and cross-asset
  day_change_pct of named symbols. Include numeric thresholds.
- confidence_reduction_points: integer 15-30, scaled by evidence strength and loss severity.
- affected_symbol: an exact symbol from the trade list, or "ALL" if the pattern spans 2+ symbols.
- If a rule restates an existing active rule, set existing_rule_id to that id so it is updated
  instead of duplicated; otherwise existing_rule_id is null.
- Return {"rules": []} when no robust pattern exists. Fewer strong rules beat many weak ones (max 5).

OUTPUT: one raw JSON object only, no markdown:
{"rules": [{"affected_symbol": "EURUSDm" | "ALL",
            "setup": "BUY when h1.rel_volume < 0.8 and h1.atr_ratio > 1.4",
            "confidence_reduction_points": 20,
            "sample_size": 3,
            "evidence": "3 of 4 EURUSDm losses (-41.20 total) were ...",
            "existing_rule_id": null}],
 "summary": "<= 60 words on the dominant failure modes"}"""


# -----------------------------------------------------------------------------
# Rules document
# -----------------------------------------------------------------------------
def _default_rules_document() -> Dict[str, Any]:
    return {"rules": [], "last_audit_at": None, "trades_analyzed": 0}


def load_rules_document() -> Dict[str, Any]:
    document = read_json_file(config.RULES_FILE, _default_rules_document)
    if not isinstance(document, dict) or not isinstance(document.get("rules"), list):
        return _default_rules_document()
    document["rules"] = [rule for rule in document["rules"] if isinstance(rule, dict)]
    document.setdefault("last_audit_at", None)
    document.setdefault("trades_analyzed", 0)
    return document


def save_rules_document(doc: Dict[str, Any]) -> None:
    write_json_atomic(config.RULES_FILE, doc)


def get_active_rules(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    rules = [rule for rule in load_rules_document()["rules"]
             if str(rule.get("status", "ACTIVE")).upper() == "ACTIVE"]
    if symbol is None:
        return rules
    wanted = {symbol.upper(), "ALL"}
    return [rule for rule in rules if str(rule.get("affected_symbol", "")).upper() in wanted]


def set_rule_status(rule_id: str, status: str) -> Optional[Dict[str, Any]]:
    """Manually activate/disable a rule from the dashboard. Returns the updated rule."""
    status = status.upper()
    if status not in ("ACTIVE", "DISABLED"):
        raise ValueError("status must be ACTIVE or DISABLED")
    document = load_rules_document()
    for rule in document["rules"]:
        if str(rule.get("id", "")).upper() == rule_id.upper():
            rule["status"] = status
            rule["updated_at"] = utc_now_iso()
            save_rules_document(document)
            logger.info("[LEARNING] Rule %s set to %s by operator", rule.get("id"), status)
            return rule
    return None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _closed_trades() -> List[Dict[str, Any]]:
    closed = [
        record for record in load_trade_memory()
        if record.get("status") == "CLOSED" and record.get("realized_pnl") is not None
    ]
    closed.sort(key=lambda r: str(r.get("exit_time") or r.get("reconciled_at") or r.get("timestamp") or ""))
    return closed


def audit_due() -> Dict[str, Any]:
    """Decide whether an audit should run now and whether it may bypass the cooldown."""
    document = load_rules_document()
    closed = _closed_trades()
    status: Dict[str, Any] = {"due": False, "force": False, "closed_trades": len(closed), "new_closed_trades": 0}
    if len(closed) < config.AUDIT_MIN_TRADES:
        status["reason"] = f"{len(closed)} closed trades (< {config.AUDIT_MIN_TRADES} required)"
        return status

    last_run = _parse_iso(document.get("last_run_at") or document.get("last_audit_at"))
    new_closed = [
        record for record in closed
        if last_run is None or (_parse_iso(record.get("reconciled_at")) or datetime.min.replace(tzinfo=timezone.utc)) > last_run
    ]
    status["new_closed_trades"] = len(new_closed)
    if not new_closed:
        status["reason"] = "no closed trades since the last audit"
        return status
    if len(new_closed) >= config.AUDIT_TRADE_THRESHOLD:
        status.update(due=True, force=True,
                      reason=f"{len(new_closed)} new closed trades >= threshold {config.AUDIT_TRADE_THRESHOLD}")
        return status

    last_audit = _parse_iso(document.get("last_audit_at"))
    if last_audit is None or datetime.now(timezone.utc) - last_audit >= timedelta(hours=config.AUDIT_COOLDOWN_HOURS):
        status.update(due=True, reason=f"{len(new_closed)} new closed trade(s) and cooldown elapsed")
    else:
        next_at = last_audit + timedelta(hours=config.AUDIT_COOLDOWN_HOURS)
        status["reason"] = f"cooldown active until {next_at.isoformat(timespec='minutes')}"
    return status


def _context_view(record: Dict[str, Any]) -> Dict[str, Any]:
    context = record.get("market_context") or {}

    def timeframe(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        data = data or {}
        structure = data.get("structure") or {}
        return {
            "rel_volume": data.get("rel_volume"),
            "tick_volume": data.get("tick_volume"),
            "atr14": data.get("atr14"),
            "atr_pct": data.get("atr_pct"),
            "atr_ratio": data.get("atr_ratio"),
            "rsi14": data.get("rsi14"),
            "ema_distance_atr": data.get("ema_distance_atr"),
            "price_vs_ema": data.get("price_vs_ema"),
            "ema_slope": data.get("ema_slope"),
            "structure": structure.get("pattern"),
        }

    return {
        "h1": timeframe(context.get("h1")),
        "d1": timeframe(context.get("d1")),
        "cross_asset": context.get("correlated_prices") or {},
    }


def _trade_view(record: Dict[str, Any], detailed: bool) -> Dict[str, Any]:
    view = {
        "symbol": record.get("symbol"),
        "side": record.get("side"),
        "entry": record.get("entry_price"),
        "exit": record.get("exit_price"),
        "sl": record.get("stop_loss"),
        "tp": record.get("take_profit"),
        "pnl": record.get("realized_pnl"),
        "exit_reason": record.get("exit_reason"),
        "held_min": record.get("holding_minutes"),
        "confidence": record.get("confidence_score"),
    }
    if detailed:
        view.update(_context_view(record))
    else:
        context = _context_view(record)
        view.update({
            "h1_rel_volume": context["h1"]["rel_volume"],
            "h1_atr_ratio": context["h1"]["atr_ratio"],
            "h1_rsi14": context["h1"]["rsi14"],
            "h1_ema_distance_atr": context["h1"]["ema_distance_atr"],
            "d1_price_vs_ema": context["d1"]["price_vs_ema"],
        })
    return view


def _known_symbols(trades: Iterable[Dict[str, Any]]) -> List[str]:
    symbols = {str(t.get("symbol")) for t in trades if t.get("symbol")}
    symbols.update(config.SYMBOLS)
    return sorted(symbols)


def _normalize_symbol(raw: Any, known: List[str]) -> Optional[str]:
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.upper() in ("ALL", "*", "ANY", "ALL_SYMBOLS"):
        return "ALL"
    for symbol in known:
        if symbol.upper() == candidate.upper():
            return symbol
    # Tolerate broker suffix differences such as "EURUSD" vs "EURUSDm".
    matches = [s for s in known if s.upper().startswith(candidate.upper()) or candidate.upper().startswith(s.upper())]
    return matches[0] if len(matches) == 1 else None


def _validate_rule(raw: Any, known: List[str], losses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    setup = re.sub(r"\s+", " ", str(raw.get("setup") or "")).strip()
    if len(setup) < 10:
        return None
    symbol = _normalize_symbol(raw.get("affected_symbol"), known)
    if symbol is None:
        logger.warning("[AUDITOR] Discarded rule for unknown symbol %r", raw.get("affected_symbol"))
        return None

    try:
        points = int(round(float(raw.get("confidence_reduction_points", 20))))
    except (TypeError, ValueError):
        points = 20
    points = max(config.RULE_MIN_PENALTY, min(config.RULE_MAX_PENALTY, points))

    try:
        sample = int(raw.get("sample_size") or 0)
    except (TypeError, ValueError):
        sample = 0
    available = len(losses) if symbol == "ALL" else sum(1 for t in losses if t.get("symbol") == symbol)
    sample = min(sample, available)
    if sample < config.RULE_MIN_SAMPLE_SIZE:
        logger.info("[AUDITOR] Discarded weak rule (sample %d < %d): %s",
                    sample, config.RULE_MIN_SAMPLE_SIZE, setup[:90])
        return None

    existing = raw.get("existing_rule_id")
    return {
        "affected_symbol": symbol,
        "setup": setup[:400],
        "confidence_reduction_points": points,
        "sample_size": sample,
        "evidence": re.sub(r"\s+", " ", str(raw.get("evidence") or "")).strip()[:600],
        "existing_rule_id": str(existing).strip() if existing else None,
    }


def _rule_key(rule: Dict[str, Any]) -> str:
    text = f"{str(rule.get('affected_symbol', '')).upper()}|{re.sub(r'[^a-z0-9.<>=]+', '', str(rule.get('setup', '')).lower())}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _merge_rules(document: Dict[str, Any], proposals: List[Dict[str, Any]], now: str) -> Dict[str, List[Dict[str, Any]]]:
    rules: List[Dict[str, Any]] = document["rules"]
    by_id = {str(rule.get("id", "")).upper(): rule for rule in rules}
    by_key = {_rule_key(rule): rule for rule in rules}
    added: List[Dict[str, Any]] = []
    updated: List[Dict[str, Any]] = []

    for proposal in proposals:
        target = by_id.get(str(proposal.get("existing_rule_id") or "").upper()) or by_key.get(_rule_key(proposal))
        fields = {k: proposal[k] for k in ("affected_symbol", "setup", "confidence_reduction_points",
                                          "sample_size", "evidence")}
        if target is not None:
            target.update(fields)
            target["updated_at"] = now
            target["times_confirmed"] = int(target.get("times_confirmed", 1)) + 1
            if target.get("status") == "RETIRED":
                target["status"] = "ACTIVE"  # the operator's DISABLED choice is always respected
            by_key[_rule_key(target)] = target
            updated.append(target)
            continue
        rule = {
            "id": f"R-{uuid.uuid4().hex[:6].upper()}",
            **fields,
            "status": "ACTIVE",
            "created_at": now,
            "updated_at": now,
            "times_confirmed": 1,
        }
        rules.append(rule)
        by_id[rule["id"]] = rule
        by_key[_rule_key(rule)] = rule
        added.append(rule)

    active = sorted((rule for rule in rules if rule.get("status") == "ACTIVE"),
                    key=lambda rule: str(rule.get("updated_at") or ""))
    retired = active[: max(0, len(active) - config.MAX_ACTIVE_RULES)]
    for rule in retired:
        rule["status"] = "RETIRED"
        rule["retired_at"] = now
    return {"added": added, "updated": updated, "retired": retired}


def _record_run(document: Dict[str, Any], now: str, status: str, **extra: Any) -> None:
    document["last_run_at"] = now
    document["last_run_status"] = status
    history = list(document.get("audit_history") or [])
    history.append({"at": now, "status": status, **extra})
    document["audit_history"] = history[-20:]


# -----------------------------------------------------------------------------
# Audit
# -----------------------------------------------------------------------------
def run_audit(force: bool = False) -> Dict[str, Any]:
    """Run one audit pass. Concurrent calls return {"status": "busy"} immediately."""
    if not _AUDIT_LOCK.acquire(blocking=False):
        return {"status": "busy", "message": "An audit is already in progress."}
    try:
        return _run_audit_locked(force)
    except Exception as exc:  # never let the auditor take the trading loop down
        logger.exception("[AUDITOR] Audit crashed: %s", exc)
        return {"status": "error", "message": f"audit crashed: {exc}"}
    finally:
        _AUDIT_LOCK.release()


def _run_audit_locked(force: bool) -> Dict[str, Any]:
    try:
        reconciled = reconcile_closed_trades()
    except Exception as exc:
        logger.warning("[AUDITOR] Reconciliation before audit failed: %s", exc)
        reconciled = 0

    document = load_rules_document()
    window = _closed_trades()[-config.AUDIT_LOOKBACK_TRADES:]
    base = {"reconciled": reconciled, "trades_analyzed": len(window), "last_audit_at": document.get("last_audit_at")}

    if len(window) < config.AUDIT_MIN_TRADES:
        message = f"{len(window)} closed trade(s) with realized P/L; at least {config.AUDIT_MIN_TRADES} required."
        logger.info("[AUDITOR] Audit skipped: %s", message)
        return {**base, "status": "insufficient_trades", "message": message}

    last_audit = _parse_iso(document.get("last_audit_at"))
    if not force and last_audit is not None:
        next_at = last_audit + timedelta(hours=config.AUDIT_COOLDOWN_HOURS)
        if datetime.now(timezone.utc) < next_at:
            message = f"Cooldown active until {next_at.isoformat(timespec='minutes')}."
            logger.info("[AUDITOR] Audit skipped: %s", message)
            return {**base, "status": "cooldown", "message": message, "next_audit_at": next_at.isoformat()}

    now = utc_now_iso()
    losses = [trade for trade in window if trade.get("outcome") == "LOSS"]
    wins = [trade for trade in window if trade.get("outcome") == "WIN"]
    if not losses:
        _record_run(document, now, "no_losses", trades=len(window))
        save_rules_document(document)
        message = f"No losing trades among the last {len(window)} closed trades; no new penalties."
        logger.info("[AUDITOR] %s", message)
        return {**base, "status": "no_losses", "message": message}

    if not config.DEEPSEEK_API_KEY:
        return {**base, "status": "error", "message": "DEEPSEEK_API_KEY is not configured."}

    active_rules = [
        {"id": r.get("id"), "affected_symbol": r.get("affected_symbol"), "setup": r.get("setup"),
         "confidence_reduction_points": r.get("confidence_reduction_points")}
        for r in document["rules"] if r.get("status") == "ACTIVE"
    ]
    total_loss = sum(float(t.get("realized_pnl") or 0.0) for t in losses)
    brief = {
        "window": {"closed_trades": len(window), "wins": len(wins), "losses": len(losses),
                   "net_pnl": round(sum(float(t.get("realized_pnl") or 0.0) for t in window), 2),
                   "loss_total": round(total_loss, 2)},
        "losing_trades": [_trade_view(t, detailed=True) for t in losses],
        "winning_trades_for_contrast": [_trade_view(t, detailed=False) for t in wins],
        "existing_active_rules": active_rules,
    }
    messages = [
        {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
        {"role": "user", "content": "Audit these trades and return the JSON rule set.\n"
                                    + json.dumps(brief, separators=(",", ":"), default=str)},
    ]

    logger.info("[AUDITOR] Auditing %d closed trades (%d losses, %.2f) against %d active rule(s)%s",
                len(window), len(losses), total_loss, len(active_rules), " [forced]" if force else "")
    try:
        response = deepseek_chat(messages, temperature=config.AI_TEMPERATURE, max_tokens=1800, json_mode=True)
        parsed = parse_json_payload(response["content"])
    except (DeepSeekError, ValueError) as exc:
        logger.error("[AUDITOR] DeepSeek audit failed: %s", exc)
        _record_run(document, now, "error", error=str(exc)[:200])
        save_rules_document(document)
        return {**base, "status": "error", "message": f"DeepSeek audit failed: {exc}"}

    raw_rules = parsed.get("rules", []) if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else []
    summary = str(parsed.get("summary", "")).strip()[:600] if isinstance(parsed, dict) else ""
    known = _known_symbols(window)
    proposals = [rule for rule in (_validate_rule(r, known, losses) for r in raw_rules or []) if rule]
    merged = _merge_rules(document, proposals, now)

    document["last_audit_at"] = now
    document["trades_analyzed"] = len(window)
    document["losses_analyzed"] = len(losses)
    document["audit_summary"] = summary
    document["audits_completed"] = int(document.get("audits_completed", 0)) + 1
    _record_run(document, now, "completed", trades=len(window), losses=len(losses),
                added=len(merged["added"]), updated=len(merged["updated"]))
    save_rules_document(document)

    for rule in merged["added"]:
        logger.info("[AUDITOR] NEW RULE %s | %s | -%d | n=%d | %s", rule["id"], rule["affected_symbol"],
                    rule["confidence_reduction_points"], rule["sample_size"], rule["setup"])
    for rule in merged["updated"]:
        logger.info("[AUDITOR] RECONFIRMED %s | %s | -%d | n=%d", rule["id"], rule["affected_symbol"],
                    rule["confidence_reduction_points"], rule["sample_size"])
    for rule in merged["retired"]:
        logger.info("[AUDITOR] RETIRED %s (active rule cap %d)", rule.get("id"), config.MAX_ACTIVE_RULES)

    active_count = sum(1 for rule in document["rules"] if rule.get("status") == "ACTIVE")
    logger.info("[LEARNING] Audit complete: %d added, %d reconfirmed, %d active rules. %s",
                len(merged["added"]), len(merged["updated"]), active_count, summary)
    return {
        **base,
        "status": "completed",
        "last_audit_at": now,
        "losses_analyzed": len(losses),
        "rules_added": len(merged["added"]),
        "rules_updated": len(merged["updated"]),
        "rules_retired": len(merged["retired"]),
        "active_rules": active_count,
        "summary": summary,
        "new_rules": merged["added"],
        "message": (f"Audited {len(window)} trades ({len(losses)} losses): {len(merged['added'])} new, "
                    f"{len(merged['updated'])} reconfirmed, {active_count} active."),
    }
