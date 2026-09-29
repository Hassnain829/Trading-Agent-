"""
Confidence calibration: does the AI's confidence score actually predict results?

Closed trades are grouped by the confidence they were executed at and scored in
R (realized P/L divided by the amount risked at entry). The effective threshold
steps up past the lowest 5-point confidence bands that lost money on their own
(each needs CALIBRATION_MIN_TRADES / 2 trades), and further if everything at or
above that level still lost money over CALIBRATION_MIN_TRADES trades (capped at
CALIBRATION_MAX_THRESHOLD). It is never lowered below CONFIDENCE_THRESHOLD.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger("hedgefund.calibration")

BUCKET = 5
_state: Dict[str, Any] = {"threshold": None, "report": None}


def _r_multiple(record: Dict[str, Any]) -> Optional[float]:
    pnl = record.get("realized_pnl")
    if pnl is None:
        return None
    equity = (record.get("market_context") or {}).get("equity")
    risk_percent = record.get("risk_percent")
    try:
        risk = float(equity) * float(risk_percent) / 100.0
    except (TypeError, ValueError):
        risk = 0.0
    if risk > 0:
        return float(pnl) / risk
    return 1.0 if float(pnl) > 0 else -1.0 if float(pnl) < 0 else 0.0


def build_report(records: List[Dict[str, Any]], mode: Optional[str] = None,
                 shadows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Win rate and expectancy (R) per confidence bucket, plus the threshold the data supports.
    Resolved shadow trades (rejected setups) count with their weight (config.SHADOW_WEIGHT).
    """
    trades = []  # (confidence, R, weight)
    for record in records:
        if record.get("status") != "CLOSED" or (mode and record.get("account_mode") != mode):
            continue
        r_value = _r_multiple(record)
        confidence = record.get("confidence_score")
        if r_value is None or confidence is None:
            continue
        trades.append((int(confidence), r_value, 1.0))
    shadow_count = 0
    for shadow in shadows or []:
        if shadow.get("confidence_score") is None or shadow.get("r") is None:
            continue  # skipped before the AI was asked: no confidence to calibrate
        trades.append((int(shadow["confidence_score"]), float(shadow["r"]), float(shadow.get("weight") or 0.5)))
        shadow_count += 1

    def weight(rows):
        return sum(w for _, _, w in rows)

    def mean_r(rows):
        total = weight(rows)
        return sum(r * w for _, r, w in rows) / total if total else 0.0

    buckets: Dict[int, Dict[str, Any]] = {}
    for confidence, r_value, w in trades:
        bucket = buckets.setdefault(confidence // BUCKET * BUCKET, {"trades": 0.0, "wins": 0.0, "r_total": 0.0, "n": 0})
        bucket["trades"] += w
        bucket["wins"] += w * int(r_value > 0)
        bucket["r_total"] += w * r_value
        bucket["n"] += 1
    rows = [{"from": low, "to": low + BUCKET - 1, "trades": b["n"], "weighted": round(b["trades"], 1),
             "win_rate": round(b["wins"] / b["trades"] * 100.0, 1) if b["trades"] else None,
             "expectancy_r": round(b["r_total"] / b["trades"], 2) if b["trades"] else None}
            for low, b in sorted(buckets.items())]

    configured = config.CONFIDENCE_THRESHOLD
    cap = config.CALIBRATION_MAX_THRESHOLD
    threshold, reasons = configured, []

    # 1. Step up past the lowest confidence bands that have proven to lose money on their own.
    band_min = max(3, config.CALIBRATION_MIN_TRADES // 2)
    while threshold < cap:
        band = [t for t in trades if threshold <= t[0] < threshold + BUCKET]
        if weight(band) < band_min or mean_r(band) >= 0:
            break
        reasons.append(f"{threshold}-{threshold + BUCKET - 1} lost {mean_r(band):+.2f}R over {weight(band):g} trades")
        threshold = min(cap, threshold + BUCKET)

    # 2. Everything at or above the level must also be profitable overall.
    at_or_above = [t for t in trades if t[0] >= threshold]
    if weight(at_or_above) >= config.CALIBRATION_MIN_TRADES and mean_r(at_or_above) <= 0:
        expectancy = mean_r(at_or_above)
        reasons.append(f">= {threshold} lost {expectancy:+.2f}R over {weight(at_or_above):g} trades")
        raised = cap
        for level in range(threshold + BUCKET, cap + 1, BUCKET):
            subset = [t for t in trades if t[0] >= level]
            if weight(subset) < config.CALIBRATION_MIN_TRADES:
                break
            if mean_r(subset) > 0:
                raised = level
                break
        threshold = raised

    if reasons:
        reason = f"raised to {threshold}: " + "; ".join(reasons)
    elif weight(at_or_above) >= config.CALIBRATION_MIN_TRADES:
        reason = f"trades at >= {threshold} made {mean_r(at_or_above):+.2f}R on average"
    else:
        reason = "not enough closed trades yet"
    return {"configured_threshold": configured, "suggested_threshold": threshold, "reason": reason,
            "closed_trades": len(trades) - shadow_count, "shadow_trades": shadow_count,
            "min_trades": config.CALIBRATION_MIN_TRADES, "buckets": rows,
            "account_mode": mode}


def refresh(records: List[Dict[str, Any]], mode: Optional[str] = None,
            shadows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    report = build_report(records, mode, shadows)
    previous = _state["threshold"]
    _state.update(threshold=report["suggested_threshold"], report=report)
    if previous is not None and previous != report["suggested_threshold"] and config.CALIBRATE_THRESHOLD:
        logger.warning("[LEARNING] Confidence threshold calibrated %s -> %s (%s)",
                       previous, report["suggested_threshold"], report["reason"])
    return report


def effective_threshold() -> int:
    """The threshold decisions must reach: the configured one, raised by calibration when enabled."""
    configured = config.CONFIDENCE_THRESHOLD
    if not config.CALIBRATE_THRESHOLD or _state["threshold"] is None:
        return configured
    return max(configured, int(_state["threshold"]))


def report() -> Optional[Dict[str, Any]]:
    return _state["report"]
