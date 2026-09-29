"""
Daily self-learning risk auditor.

Once a day it reads the last 50 closed trades from memory.json, sends them to
DeepSeek and asks for repeated LOSS patterns in which a specific cross-asset
correlation, combined with a volume condition and a volatility condition, cost
the fund money more than once. Each such pattern becomes a rule in
new_rules.json that cuts the trading AI's confidence for that exact setup.

The AI only proposes rules as numeric conditions; the code evaluates them on
every trade in the window to decide whether they hold, sets the penalty from the
evidence, expires rules after RULE_TTL_DAYS unless reconfirmed, and retires rules
the latest trades or their blocked "shadow" trades no longer support.

It runs inside the trading server automatically (every AUDIT_INTERVAL_HOURS,
24 by default) and can also be run on its own, e.g. from Windows Task Scheduler:

    .venv\\Scripts\\python.exe auditor.py            # run if today's audit is due
    .venv\\Scripts\\python.exe auditor.py --force    # run now regardless of schedule
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

import config
import rule_engine
from ai_brain import DeepSeekError, deepseek_chat, parse_json_payload, rule_applies, rule_expired
from data_engine import currency_legs, symbols_match, utc_now_iso
from memory_store import load_trade_memory, read_json_file, reconcile_closed_trades, write_json_atomic
from shadow_store import learning_records, stats_by_blocker
from calibration import _r_multiple

logger = logging.getLogger("hedgefund.auditor")

_AUDIT_LOCK = threading.Lock()
_OK_STATUSES = {"completed", "no_losses", "no_new_trades", "not_due", "insufficient_trades"}

def auditor_system_prompt() -> str:
    return """You are the Chief Risk Officer and Quantitative Auditor of a systematic hedge fund.
Once a day you review the fund's last closed trades and propose risk rules that cut the trading AI's
confidence on setups that keep losing money. You are skeptical, evidence-driven and allergic to
curve-fitting.

YOUR TASK
Find REPEATED patterns in the LOSING trades where a cross-asset condition, together with a volume
condition and a volatility condition, preceded the loss - and that the WINNING trades do not share.
Example: buying EURUSDm while USDJPYm was up > 0.3% on the day, on H1 rel_volume < 0.8, with
h1.atr_ratio > 1.3.

DATA
Each trade has an id (L1, L2 ... losses; W1, W2 ... wins; SL1/SW1 ... are SHADOW trades: setups the bot
rejected and followed on price data, which count half), its symbol, side, P/L or R result and "metrics": the
numeric market state when it was opened. Metric names (use them exactly):
  volume:      h1.rel_volume, d1.rel_volume
  volatility:  h1.atr_ratio, h1.atr_pct, d1.atr_ratio, d1.atr_pct
  trend:       h1.rsi14, d1.rsi14, h1.ema_distance_atr, d1.ema_distance_atr, day_change_pct
  cross-asset: usd.change_pct (today's USD move, + = stronger), cross.<SYMBOL>.day_change_pct,
               cross.<SYMBOL>.corr_h1 (H1 return correlation of <SYMBOL> with the traded symbol)

RULE REQUIREMENTS
- conditions: 3 to 5 objects {"metric": <name above>, "op": "<" | "<=" | ">" | ">=", "value": <number>}.
  At least one cross-asset condition, one volume condition and one volatility condition. All must
  hold at once for the rule to match.
- The engine evaluates your conditions on EVERY trade in the window itself: it counts the matching
  losses and wins, and discards a rule with fewer than """ + str(config.RULE_MIN_SAMPLE_SIZE) + """ matching losses, with losses
  under """ + f"{config.RULE_MIN_LOSS_RATE:.0%}" + """ of its matches, or with a matching loss rate no worse than the window's. Pick
  thresholds that separate the losers from the winners; the engine sets the penalty from the evidence.
- side: BUY, SELL or ANY. affected_symbol: an exact traded symbol, or "ALL" when the matching losses
  span two or more symbols.
- If a rule restates an existing active rule, set existing_rule_id to that id; otherwise null.
- Return {"rules": []} when no pattern meets the bar. Fewer strong rules beat many weak ones (max 5).

OUTPUT: one raw JSON object only, no markdown:
{"rules": [{"affected_symbol": "EURUSDm",
            "side": "BUY",
            "conditions": [{"metric": "cross.USDJPYm.day_change_pct", "op": ">", "value": 0.3},
                           {"metric": "h1.rel_volume", "op": "<", "value": 0.8},
                           {"metric": "h1.atr_ratio", "op": ">", "value": 1.3}],
            "evidence": "EURUSDm longs lost while USDJPY rallied on thin volume into expanding ATR",
            "existing_rule_id": null}],
 "summary": "<= 60 words on the dominant cross-asset failure modes"}"""


# -----------------------------------------------------------------------------
# Rules document (shared with main.py and ai_brain.py)
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
    return [rule for rule in rules if rule_applies(rule, symbol)]


def set_rule_status(rule_id: str, status: str) -> Optional[Dict[str, Any]]:
    """Manually activate/disable a rule from the dashboard. Returns the updated rule."""
    status = status.upper()
    if status not in ("ACTIVE", "DISABLED"):
        raise ValueError("status must be ACTIVE or DISABLED")
    document = load_rules_document()
    for rule in document["rules"]:
        if str(rule.get("id", "")).upper() == rule_id.upper():
            if status == "ACTIVE" and not rule.get("conditions"):
                raise ValueError(f"rule {rule.get('id')} has no machine-checkable conditions and cannot be activated")
            rule["status"] = status
            rule["updated_at"] = utc_now_iso()
            if status == "ACTIVE" and rule_expired(rule):
                rule["expires_at"] = _expiry(utc_now_iso())  # reactivating restarts its lifetime
            save_rules_document(document)
            logger.info("[LEARNING] Rule %s set to %s by operator", rule.get("id"), status)
            return rule
    return None


def _expiry(now_iso: str) -> str:
    start = _parse_iso(now_iso) or datetime.now(timezone.utc)
    return (start + timedelta(days=config.RULE_TTL_DAYS)).isoformat(timespec="seconds")


def maintain_rules() -> List[Dict[str, Any]]:
    """
    Housekeeping that runs every scan, independent of the daily audit:
      * rules without machine-checkable conditions (older versions) become LEGACY and stop applying;
      * rules past expires_at become EXPIRED;
      * rules whose blocked (shadow) trades would have made money overall are RETIRED.
    Returns the rules that changed.
    """
    document = load_rules_document()
    shadow_stats = stats_by_blocker()
    now = utc_now_iso()
    changed: List[Dict[str, Any]] = []
    for rule in document["rules"]:
        if rule.get("status") != "ACTIVE":
            continue
        reason = None
        if not rule.get("conditions"):
            rule["status"], reason = "LEGACY", "free-text rule from an older version; its conditions cannot be checked"
        elif rule_expired(rule):
            rule["status"], reason = "EXPIRED", f"not reconfirmed by an audit within {config.RULE_TTL_DAYS} days"
        else:
            stats = shadow_stats.get(str(rule.get("id")))
            if (stats and stats["resolved"] >= config.RULE_SHADOW_MIN_RESOLVED
                    and (stats["expectancy_r"] or 0) > 0):
                rule["status"] = "RETIRED"
                reason = (f"shadow: the {stats['resolved']} trades it blocked would have made "
                          f"{stats['expectancy_r']:+.2f}R on average ({stats['wins']} won, {stats['losses']} lost)")
        if reason:
            rule.update(retired_at=now, retired_reason=reason, updated_at=now)
            changed.append(rule)
            logger.warning("[LEARNING] Rule %s -> %s: %s", rule.get("id"), rule["status"], reason)
    if changed:
        save_rules_document(document)
    return changed


# -----------------------------------------------------------------------------
# Schedule
# -----------------------------------------------------------------------------
def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _audit_mode() -> Optional[str]:
    """Account type whose trades are audited: the logged-in one, else the most recent trade's."""
    mode = config.ACTIVE_ACCOUNT.get("mode")
    if mode:
        return mode
    tagged = [r.get("account_mode") for r in load_trade_memory() if r.get("account_mode")]
    return tagged[-1] if tagged else None


def _closed_trades(mode: Optional[str] = None) -> List[Dict[str, Any]]:
    """Closed trades with realized P/L, oldest first; only ``mode`` (DEMO/LIVE) trades when given."""
    closed = [
        record for record in load_trade_memory()
        if record.get("status") == "CLOSED" and record.get("realized_pnl") is not None
        and (mode is None or record.get("account_mode") == mode)
    ]
    closed.sort(key=lambda r: str(r.get("exit_time") or r.get("reconciled_at") or r.get("timestamp") or ""))
    return closed


def _learning_window(mode: Optional[str]) -> List[Dict[str, Any]]:
    """
    The last AUDIT_LOOKBACK_TRADES real closed trades (weight 1) plus up to as many resolved shadow
    trades (rejected setups followed on price data, weight config.SHADOW_WEIGHT), all with their result in R.
    """
    real = []
    for trade in _closed_trades(mode)[-config.AUDIT_LOOKBACK_TRADES:]:
        real.append({**trade, "source": "trade", "weight": 1.0, "r": _r_multiple(trade)})
    shadows = learning_records(mode, limit=config.AUDIT_LOOKBACK_TRADES) if config.SHADOW_WEIGHT > 0 else []
    window = real + shadows
    window.sort(key=lambda t: str(t.get("exit_time") or t.get("reconciled_at") or t.get("timestamp") or ""))
    return window


def _weight(trades: Iterable[Dict[str, Any]]) -> float:
    return round(sum(float(t.get("weight", 1.0)) for t in trades), 3)


def _next_audit_at(document: Dict[str, Any]) -> Optional[datetime]:
    last_run = _parse_iso(document.get("last_run_at") or document.get("last_audit_at"))
    return last_run + timedelta(hours=config.AUDIT_INTERVAL_HOURS) if last_run else None


def audit_due() -> Dict[str, Any]:
    """Whether today's audit should run now (once per AUDIT_INTERVAL_HOURS)."""
    document = load_rules_document()
    mode = _audit_mode()
    closed = _learning_window(mode)
    real = sum(1 for t in closed if t.get("source") != "shadow")
    status: Dict[str, Any] = {"due": False, "force": False, "closed_trades": real,
                              "shadow_trades": len(closed) - real, "account_mode": mode}
    if len(closed) < config.AUDIT_MIN_TRADES:
        label = f"{mode} " if mode else ""
        status["reason"] = (f"{real} closed {label}trades + {len(closed) - real} shadow trades "
                            f"(< {config.AUDIT_MIN_TRADES} required)")
        return status
    next_at = _next_audit_at(document)
    if next_at and datetime.now(timezone.utc) < next_at:
        status["reason"] = f"next daily audit at {next_at.isoformat(timespec='minutes')}"
        status["next_audit_at"] = next_at.isoformat()
        return status
    status.update(due=True, reason="daily audit window reached")
    return status


# -----------------------------------------------------------------------------
# Trade brief for DeepSeek
# -----------------------------------------------------------------------------
def _trade_brief(record: Dict[str, Any], label: str) -> Dict[str, Any]:
    """One trade with the numeric market state at its fill, named exactly as rule conditions name it."""
    metrics = rule_engine.flatten(rule_engine.snapshot_from_record(record))
    return {
        "id": label,
        "symbol": record.get("symbol"),
        "side": record.get("side"),
        "pnl": record.get("realized_pnl"),
        "r": record.get("r"),
        "source": record.get("source", "trade"),
        "exit_reason": record.get("exit_reason"),
        "held_min": record.get("holding_minutes"),
        "sl_atr_multiple": record.get("sl_atr_multiple"),
        "metrics": {key: value for key, value in metrics.items() if value is not None},
    }


# -----------------------------------------------------------------------------
# Rule validation and merging
# -----------------------------------------------------------------------------
def _known_symbols(trades: Iterable[Dict[str, Any]]) -> List[str]:
    symbols = {str(t.get("symbol")) for t in trades if t.get("symbol")}
    symbols.update(config.SYMBOLS)
    for trade in trades:
        symbols.update(((trade.get("market_context") or {}).get("correlated_prices") or {}).keys())
    return sorted(symbols)


def _normalize_symbol(raw: Any, known: List[str], allow_all: bool = True) -> Optional[str]:
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if allow_all and candidate.upper() in ("ALL", "*", "ANY", "ALL_SYMBOLS"):
        return "ALL"
    for symbol in known:
        if symbol.upper() == candidate.upper():
            return symbol
    # Tolerate broker suffix differences such as "EURUSD" vs "EURUSDm".
    matches = [s for s in known if symbols_match(s, candidate)]
    return matches[0] if len(matches) == 1 else None


def _clean_text(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def evaluate_rule(symbol: str, side: str, conditions: List[Dict[str, Any]],
                  trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Which trades a rule matches, decided by code from each trade's stored market state."""
    matched = [
        trade for trade in trades
        if (symbol == "ALL" or symbols_match(str(trade.get("symbol") or ""), symbol))
        and (side == "ANY" or trade.get("side") == side)
        and rule_engine.evaluate(conditions, rule_engine.snapshot_from_record(trade)) is True
    ]
    losses = [t for t in matched if t.get("outcome") == "LOSS"]
    wins = [t for t in matched if t.get("outcome") == "WIN"]
    loss_w, win_w = _weight(losses), _weight(wins)
    decided = loss_w + win_w
    net_r = sum(float(t.get("weight", 1.0)) * float(t["r"] if t.get("r") is not None else _r_multiple(t) or 0.0)
                for t in matched)
    return {
        "matched": matched, "losses": losses, "wins": wins, "loss_weight": loss_w, "win_weight": win_w,
        "loss_rate": loss_w / decided if decided else None,
        "net_r": round(net_r, 3),  # weighted: a shadow trade counts SHADOW_WEIGHT of a real one
        "net_pnl": sum(float(t.get("realized_pnl") or 0.0) for t in matched),
    }


def _penalty_points(losses: int, loss_rate: float) -> int:
    """Penalty from the evidence: more matching losses and a higher loss rate -> a bigger cut."""
    size = min(1.0, (losses - config.RULE_MIN_SAMPLE_SIZE + 1) / 4.0)
    purity = max(0.0, min(1.0, (loss_rate - 0.5) / 0.5))
    points = config.RULE_MIN_PENALTY + (config.RULE_MAX_PENALTY - config.RULE_MIN_PENALTY) * size * purity
    return int(round(max(config.RULE_MIN_PENALTY, min(config.RULE_MAX_PENALTY, points))))


def _validate_rule(raw: Any, known: List[str], window: List[Dict[str, Any]],
                   baseline_loss_rate: float) -> Optional[Dict[str, Any]]:
    """
    Accept a proposed rule only if its numeric conditions, evaluated by code on every trade in the
    window, match enough losses, mostly losses, and a worse loss rate than the window as a whole.
    """
    if not isinstance(raw, dict):
        return None
    conditions = rule_engine.normalize_conditions(raw.get("conditions"), known)
    kinds = rule_engine.categories(conditions)
    required = {rule_engine.CATEGORY_CROSS, rule_engine.CATEGORY_VOLUME, rule_engine.CATEGORY_VOLATILITY}
    if not required <= kinds:
        logger.info("[AUDITOR] Discarded rule: needs cross-asset, volume and volatility conditions, got %s (%s)",
                    sorted(kinds) or "none", raw.get("conditions"))
        return None

    symbol = _normalize_symbol(raw.get("affected_symbol"), known)
    if symbol is None:
        logger.info("[AUDITOR] Discarded rule with unknown symbol %r", raw.get("affected_symbol"))
        return None
    side = str(raw.get("side") or "ANY").strip().upper()
    side = side if side in ("BUY", "SELL") else "ANY"
    setup = f"{'ANY side' if side == 'ANY' else side} {symbol} when {rule_engine.describe(conditions)}"

    result = evaluate_rule(symbol, side, conditions, window)
    losses, wins, loss_rate = result["losses"], result["wins"], result["loss_rate"]
    if result["loss_weight"] < config.RULE_MIN_SAMPLE_SIZE:
        logger.info("[AUDITOR] Discarded rule: matches %.1f weighted loss(es), need %d | %s",
                    result["loss_weight"], config.RULE_MIN_SAMPLE_SIZE, setup)
        return None
    if loss_rate is None or loss_rate < config.RULE_MIN_LOSS_RATE or loss_rate <= baseline_loss_rate:
        logger.info("[AUDITOR] Discarded rule: %.1f losses vs %.1f wins (loss rate %.0f%%, window %.0f%%) | %s",
                    result["loss_weight"], result["win_weight"], (loss_rate or 0) * 100, baseline_loss_rate * 100, setup)
        return None
    if result["net_r"] >= 0:
        logger.info("[AUDITOR] Discarded rule: its matching trades made money overall (%+.2fR) | %s",
                    result["net_r"], setup)
        return None
    loss_symbols = {record.get("symbol") for record in losses}
    if symbol == "ALL" and len(loss_symbols) == 1:
        symbol = loss_symbols.pop()  # an "ALL" rule backed by one symbol only applies to that symbol
        setup = f"{'ANY side' if side == 'ANY' else side} {symbol} when {rule_engine.describe(conditions)}"

    existing = raw.get("existing_rule_id")
    return {
        "pattern_type": "CROSS_ASSET",
        "affected_symbol": symbol,
        "side": side,
        "correlated_symbol": rule_engine.correlated_symbol(conditions),
        "conditions": conditions,
        "setup": setup[:480],
        "confidence_reduction_points": _penalty_points(len(losses), loss_rate),
        "sample_size": len(losses),
        "winning_matches": len(wins),
        "shadow_losses": sum(1 for t in losses if t.get("source") == "shadow"),
        "weighted_losses": result["loss_weight"],
        "matched_net_r": result["net_r"],
        "loss_rate": round(loss_rate, 3),
        "baseline_loss_rate": round(baseline_loss_rate, 3),
        "loss_total": round(sum(float(r.get("realized_pnl") or 0.0) for r in losses), 2),
        "matched_net_pnl": round(result["net_pnl"], 2),
        "loss_trade_refs": [str(r.get("id") or r.get("ticket")) for r in losses],
        # Where the evidence came from (demo/live, broker), for rule sharing across accounts.
        "learned_modes": sorted({str(r.get("account_mode")) for r in losses if r.get("account_mode")}),
        "learned_brokers": sorted({str(r.get("broker")) for r in losses if r.get("broker")}),
        "learned_servers": sorted({str(r.get("server")) for r in losses if r.get("server")}),
        "evidence": _clean_text(raw.get("evidence"), 600),
        "existing_rule_id": str(existing).strip() if existing else None,
    }


def _revalidate_active_rules(document: Dict[str, Any], window: List[Dict[str, Any]], now: str) -> List[Dict[str, Any]]:
    """Retire active rules that the latest trades no longer support (checked by code, every audit)."""
    retired = []
    for rule in document["rules"]:
        if rule.get("status") != "ACTIVE" or not rule.get("conditions"):
            continue
        result = evaluate_rule(str(rule.get("affected_symbol")), str(rule.get("side") or "ANY"),
                               rule["conditions"], window)
        decided = result["loss_weight"] + result["win_weight"]
        if decided >= config.RULE_MIN_SAMPLE_SIZE and (result["net_r"] >= 0 or (result["loss_rate"] or 0) < 0.5):
            rule.update(status="RETIRED", retired_at=now, updated_at=now,
                        retired_reason=(f"the last {len(window)} trades no longer support it: {len(result['losses'])} "
                                        f"losses vs {len(result['wins'])} wins, net {result['net_r']:+.2f}R"))
            retired.append(rule)
    return retired


def _canonical_symbol(symbol: str) -> str:
    """Broker-neutral name (EURUSDm / EURUSD.r -> EURUSD) so one pattern learned at two brokers is one rule."""
    legs = currency_legs(symbol)
    return "".join(legs) if legs else re.sub(r"[^A-Z0-9]", "", str(symbol).upper())


def _rule_key(rule: Dict[str, Any]) -> str:
    parts = []
    for condition in rule.get("conditions") or []:
        metric = condition["metric"]
        if metric.startswith("cross."):
            _, symbol, field = metric.split(".", 2)
            metric = f"cross.{_canonical_symbol(symbol)}.{field}"
        parts.append(f"{metric}{condition['op']}{condition['value']}")
    setup = "|".join(sorted(parts)) or re.sub(r"[^a-z0-9.<>=+-]+", "", str(rule.get("setup", "")).lower())
    key = f"{_canonical_symbol(str(rule.get('affected_symbol', '')))}|{str(rule.get('side') or 'ANY').upper()}|{setup}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


_RULE_FIELDS = ("pattern_type", "affected_symbol", "side", "correlated_symbol", "conditions", "setup",
                "confidence_reduction_points", "sample_size", "winning_matches", "loss_rate", "baseline_loss_rate",
                "loss_total", "matched_net_pnl", "loss_trade_refs", "evidence")
_PROVENANCE_FIELDS = ("learned_modes", "learned_brokers", "learned_servers")


def _merge_rules(document: Dict[str, Any], proposals: List[Dict[str, Any]], now: str) -> Dict[str, List[Dict[str, Any]]]:
    rules: List[Dict[str, Any]] = document["rules"]
    by_id = {str(rule.get("id", "")).upper(): rule for rule in rules}
    by_key = {_rule_key(rule): rule for rule in rules}
    added: List[Dict[str, Any]] = []
    updated: List[Dict[str, Any]] = []

    for proposal in proposals:
        target = by_id.get(str(proposal.get("existing_rule_id") or "").upper()) or by_key.get(_rule_key(proposal))
        fields = {key: proposal[key] for key in _RULE_FIELDS}
        if target is not None:
            shadow_retired = str(target.get("retired_reason") or "").startswith("shadow")
            target.update(fields)
            for key in _PROVENANCE_FIELDS:  # a rule reconfirmed elsewhere is known at both places
                target[key] = sorted(set(target.get(key) or []) | set(proposal.get(key) or []))
            target["updated_at"] = now
            target["expires_at"] = _expiry(now)
            target["times_confirmed"] = int(target.get("times_confirmed", 1)) + 1
            # The operator's DISABLED choice is always respected; a rule whose blocked trades proved it
            # wrong is not revived by the same losing trades.
            if target.get("status") in ("RETIRED", "EXPIRED", "LEGACY") and not shadow_retired:
                target["status"] = "ACTIVE"
                target.pop("retired_reason", None)
            by_key[_rule_key(target)] = target
            updated.append(target)
            continue
        rule = {"id": f"R-{uuid.uuid4().hex[:6].upper()}", **fields,
                **{key: proposal.get(key) or [] for key in _PROVENANCE_FIELDS},
                "status": "ACTIVE", "created_at": now, "updated_at": now, "expires_at": _expiry(now),
                "times_confirmed": 1}
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
        rule["retired_reason"] = f"active rule cap ({config.MAX_ACTIVE_RULES}) reached; oldest retired"
    return {"added": added, "updated": updated, "retired": retired}


def _record_run(document: Dict[str, Any], now: str, status: str, **extra: Any) -> None:
    document["last_run_at"] = now
    document["last_run_status"] = status
    history = list(document.get("audit_history") or [])
    history.append({"at": now, "status": status, **extra})
    document["audit_history"] = history[-30:]


# -----------------------------------------------------------------------------
# Audit
# -----------------------------------------------------------------------------
@contextlib.contextmanager
def _process_lock() -> Iterator[bool]:
    """Cross-process lock so the server and a scheduled `python auditor.py` never audit at once."""
    handle = open(config.AUDIT_LOCK_FILE, "a+")
    try:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def run_audit(force: bool = False, reconcile: bool = True) -> Dict[str, Any]:
    """
    Run one audit. Without ``force`` it only proceeds when the daily window has
    been reached. Concurrent calls (same or another process) return "busy".
    """
    if not _AUDIT_LOCK.acquire(blocking=False):
        return {"status": "busy", "message": "An audit is already in progress."}
    try:
        with _process_lock() as acquired:
            if not acquired:
                return {"status": "busy", "message": "An audit is already running in another process."}
            return _run_audit_locked(force, reconcile)
    except Exception as exc:  # never let the auditor take the trading loop down
        logger.exception("[AUDITOR] Audit crashed: %s", exc)
        return {"status": "error", "message": f"audit crashed: {exc}"}
    finally:
        _AUDIT_LOCK.release()


def _run_audit_locked(force: bool, reconcile: bool) -> Dict[str, Any]:
    reconciled = 0
    if reconcile:
        try:
            reconciled = reconcile_closed_trades()
        except Exception as exc:
            logger.warning("[AUDITOR] Reconciliation before audit failed: %s", exc)

    try:
        maintain_rules()
    except Exception as exc:
        logger.warning("[AUDITOR] Rule maintenance failed: %s", exc)
    document = load_rules_document()
    mode = _audit_mode()
    window = _learning_window(mode)
    shadow_count = sum(1 for t in window if t.get("source") == "shadow")
    base = {"reconciled": reconciled, "trades_analyzed": len(window) - shadow_count,
            "shadow_trades_analyzed": shadow_count, "account_mode": mode,
            "last_audit_at": document.get("last_audit_at")}

    if len(window) < config.AUDIT_MIN_TRADES:
        message = f"{len(window)} closed trade(s) with realized P/L; at least {config.AUDIT_MIN_TRADES} required."
        logger.info("[AUDITOR] Audit skipped: %s", message)
        return {**base, "status": "insufficient_trades", "message": message}

    next_at = _next_audit_at(document)
    if not force and next_at and datetime.now(timezone.utc) < next_at:
        message = f"Today's audit already ran; next at {next_at.isoformat(timespec='minutes')}."
        logger.info("[AUDITOR] Audit skipped: %s", message)
        return {**base, "status": "not_due", "message": message, "next_audit_at": next_at.isoformat()}

    now = utc_now_iso()
    last_audit = _parse_iso(document.get("last_audit_at"))
    fresh = [t for t in window if last_audit is None
             or (_parse_iso(t.get("reconciled_at")) or datetime.min.replace(tzinfo=timezone.utc)) > last_audit]
    if not force and not fresh:
        _record_run(document, now, "no_new_trades", trades=len(window))
        save_rules_document(document)
        message = "No trades closed since the last audit; rules unchanged."
        logger.info("[AUDITOR] %s", message)
        return {**base, "status": "no_new_trades", "message": message}

    losses = [trade for trade in window if trade.get("outcome") == "LOSS"]
    wins = [trade for trade in window if trade.get("outcome") == "WIN"]
    revalidated = _revalidate_active_rules(document, window, now)
    for rule in revalidated:
        logger.warning("[AUDITOR] RETIRED %s: %s", rule.get("id"), rule.get("retired_reason"))
    if not losses:
        _record_run(document, now, "no_losses", trades=len(window), retired=len(revalidated))
        save_rules_document(document)
        message = f"No losing trades among the last {len(window)} closed trades; no new penalties."
        logger.info("[AUDITOR] %s", message)
        return {**base, "status": "no_losses", "message": message}

    if not config.DEEPSEEK_API_KEY:
        if revalidated:
            save_rules_document(document)
        return {**base, "status": "error", "message": "DEEPSEEK_API_KEY is not configured."}

    def labelled(trades: List[Dict[str, Any]], prefix: str) -> Dict[str, Dict[str, Any]]:
        real = [t for t in trades if t.get("source") != "shadow"]
        shadow = [t for t in trades if t.get("source") == "shadow"]
        return {**{f"{prefix}{i}": t for i, t in enumerate(real, start=1)},
                **{f"S{prefix}{i}": t for i, t in enumerate(shadow, start=1)}}

    loss_map, win_map = labelled(losses, "L"), labelled(wins, "W")
    total_loss = sum(float(t.get("realized_pnl") or 0.0) for t in losses)
    active_rules = [{"id": r.get("id"), "affected_symbol": r.get("affected_symbol"), "side": r.get("side"),
                     "conditions": r.get("conditions"),
                     "confidence_reduction_points": r.get("confidence_reduction_points")}
                    for r in document["rules"] if r.get("status") == "ACTIVE"]
    brief = {
        "window": {"account_type": mode, "closed_trades": len(window) - shadow_count, "shadow_trades": shadow_count,
                   "wins": len(wins), "losses": len(losses),
                   "net_pnl": round(sum(float(t.get("realized_pnl") or 0.0) for t in window), 2),
                   "loss_total": round(total_loss, 2)},
        "losing_trades": [_trade_brief(trade, label) for label, trade in loss_map.items()],
        "winning_trades": [_trade_brief(trade, label) for label, trade in win_map.items()],
        "existing_active_rules": active_rules,
    }
    messages = [
        {"role": "system", "content": auditor_system_prompt()},
        {"role": "user", "content": "Audit these trades and return the JSON rule set.\n"
                                    + json.dumps(brief, separators=(",", ":"), default=str)},
    ]

    logger.info("[AUDITOR] Daily audit: %d closed trades (%d losses, %.2f) sent to DeepSeek%s",
                len(window), len(losses), total_loss, " [forced]" if force else "")
    try:
        response = deepseek_chat(messages, temperature=config.AI_TEMPERATURE,
                                 max_tokens=max(config.LLM_MAX_TOKENS, 4096), json_mode=True)
        parsed = parse_json_payload(response["content"])
    except (DeepSeekError, ValueError) as exc:
        logger.error("[AUDITOR] DeepSeek audit failed: %s", exc)
        _record_run(document, now, "error", error=str(exc)[:200])
        save_rules_document(document)
        return {**base, "status": "error", "message": f"DeepSeek audit failed: {exc}"}

    raw_rules = parsed.get("rules", []) if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else []
    summary = _clean_text(parsed.get("summary"), 600) if isinstance(parsed, dict) else ""
    known = _known_symbols(window)
    decided = _weight(losses) + _weight(wins)
    baseline_loss_rate = _weight(losses) / decided if decided else 0.0
    proposals = [rule for rule in (_validate_rule(r, known, window, baseline_loss_rate) for r in raw_rules or [])
                 if rule]
    merged = _merge_rules(document, proposals, now)

    document["last_audit_at"] = now
    document["trades_analyzed"] = len(window)
    document["losses_analyzed"] = len(losses)
    document["audit_summary"] = summary
    document["audits_completed"] = int(document.get("audits_completed", 0)) + 1
    _record_run(document, now, "completed", trades=len(window), losses=len(losses), proposed=len(raw_rules or []),
                accepted=len(proposals), added=len(merged["added"]), updated=len(merged["updated"]),
                retired=len(merged["retired"]) + len(revalidated))
    save_rules_document(document)

    for rule in merged["added"]:
        logger.info("[AUDITOR] NEW RULE %s | -%d | n=%d losses (%.2f) | %s", rule["id"],
                    rule["confidence_reduction_points"], rule["sample_size"], rule["loss_total"], rule["setup"])
    for rule in merged["updated"]:
        logger.info("[AUDITOR] RECONFIRMED %s | -%d | n=%d | %s", rule["id"], rule["confidence_reduction_points"],
                    rule["sample_size"], rule["setup"])
    for rule in merged["retired"]:
        logger.info("[AUDITOR] RETIRED %s (%s)", rule.get("id"), rule.get("retired_reason"))

    active_count = sum(1 for rule in document["rules"] if rule.get("status") == "ACTIVE")
    logger.info("[LEARNING] Audit complete: %d proposed, %d verified, %d added, %d reconfirmed, %d active. %s",
                len(raw_rules or []), len(proposals), len(merged["added"]), len(merged["updated"]), active_count, summary)
    return {
        **base,
        "status": "completed",
        "last_audit_at": now,
        "losses_analyzed": len(losses),
        "rules_proposed": len(raw_rules or []),
        "rules_added": len(merged["added"]),
        "rules_updated": len(merged["updated"]),
        "rules_retired": len(merged["retired"]) + len(revalidated),
        "active_rules": active_count,
        "summary": summary,
        "new_rules": merged["added"],
        "message": (f"Audited {len(window)} trades ({len(losses)} losses): {len(proposals)} of "
                    f"{len(raw_rules or [])} proposed rules verified, {len(merged['added'])} new, "
                    f"{len(merged['updated'])} reconfirmed, {active_count} active."),
    }


# -----------------------------------------------------------------------------
# Command line (for Windows Task Scheduler or manual runs)
# -----------------------------------------------------------------------------
def _server_running() -> bool:
    """True when the trading server holds its instance lock."""
    try:
        handle = open(config.LOCK_FILE, "a+")
    except OSError:
        return False
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        handle.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Daily cross-asset loss-pattern audit (writes new_rules.json).")
    parser.add_argument("--force", action="store_true", help="run now even if today's audit already ran")
    parser.add_argument("--no-mt5", action="store_true",
                        help="skip MT5 reconciliation and audit the trades already in memory.json")
    args = parser.parse_args(argv)

    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                            format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for warning in config.CONFIG_WARNINGS:
        logger.warning("[SYSTEM] Config: %s", warning)

    # memory.json is only written by one process: when the server is up it reconciles itself.
    mt5_ready = False
    if _server_running():
        logger.info("[AUDITOR] Trading server is running; auditing memory.json without reconciling")
    elif not args.no_mt5:
        import data_engine
        mt5_ready = data_engine.initialize_mt5()
        if mt5_ready:
            account = data_engine.get_account_snapshot() or {}
            config.ACTIVE_ACCOUNT = {"mode": account.get("account_mode"), "broker": account.get("company"),
                                     "server": account.get("server")}
        else:
            logger.warning("[AUDITOR] MT5 unavailable; auditing the trades already in memory.json")
    try:
        if not args.force:
            due = audit_due()
            if not due["due"]:
                logger.info("[AUDITOR] Nothing to do: %s", due["reason"])
                return 0
        logger.info("[AUDITOR] Auditing %s trades", _audit_mode() or "all")
        result = run_audit(force=args.force, reconcile=mt5_ready)
    finally:
        if mt5_ready:
            import data_engine
            data_engine.shutdown_mt5()

    logger.info("[AUDITOR] Result: %s - %s", result.get("status"), result.get("message", ""))
    for rule in result.get("new_rules", []):
        print(json.dumps({key: rule[key] for key in ("id", "setup", "confidence_reduction_points", "sample_size")},
                         ensure_ascii=False))
    return 0 if result.get("status") in _OK_STATUSES else 1


if __name__ == "__main__":
    sys.exit(main())
