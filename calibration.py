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


def build_report(records: List[Dict[str, Any]], mode: Optional[str] = None) -> Dict[str, Any]:
    """Win rate and expectancy (R) per confidence bucket, plus the threshold the data supports."""
    trades = []
    for record in records:
        if record.get("status") != "CLOSED" or (mode and record.get("account_mode") != mode):
            continue
        r_value = _r_multiple(record)
        confidence = record.get("confidence_score")
        if r_value is None or confidence is None:
            continue
        trades.append((int(confidence), r_value))

    buckets: Dict[int, Dict[str, Any]] = {}
    for confidence, r_value in trades:
        bucket = buckets.setdefault(confidence // BUCKET * BUCKET, {"trades": 0, "wins": 0, "r_total": 0.0})
        bucket["trades"] += 1
        bucket["wins"] += int(r_value > 0)
        bucket["r_total"] += r_value
    rows = [{"from": low, "to": low + BUCKET - 1, "trades": b["trades"],
             "win_rate": round(b["wins"] / b["trades"] * 100.0, 1),
             "expectancy_r": round(b["r_total"] / b["trades"], 2)} for low, b in sorted(buckets.items())]

    configured = config.CONFIDENCE_THRESHOLD
    cap = config.CALIBRATION_MAX_THRESHOLD
    threshold, reasons = configured, []

    # 1. Step up past the lowest confidence bands that have proven to lose money on their own.
    band_min = max(3, config.CALIBRATION_MIN_TRADES // 2)
    while threshold < cap:
        band = [r for c, r in trades if threshold <= c < threshold + BUCKET]
        if len(band) < band_min or sum(band) / len(band) >= 0:
            break
        reasons.append(f"{threshold}-{threshold + BUCKET - 1} lost {sum(band) / len(band):+.2f}R over {len(band)} trades")
        threshold = min(cap, threshold + BUCKET)

    # 2. Everything at or above the level must also be profitable overall.
    at_or_above = [r for c, r in trades if c >= threshold]
    if len(at_or_above) >= config.CALIBRATION_MIN_TRADES and sum(at_or_above) / len(at_or_above) <= 0:
        expectancy = sum(at_or_above) / len(at_or_above)
        reasons.append(f">= {threshold} lost {expectancy:+.2f}R over {len(at_or_above)} trades")
        raised = cap
        for level in range(threshold + BUCKET, cap + 1, BUCKET):
            subset = [r for c, r in trades if c >= level]
            if len(subset) < config.CALIBRATION_MIN_TRADES:
                break
            if sum(subset) / len(subset) > 0:
                raised = level
                break
        threshold = raised

    if reasons:
        reason = f"raised to {threshold}: " + "; ".join(reasons)
    elif len(at_or_above) >= config.CALIBRATION_MIN_TRADES:
        reason = f"trades at >= {threshold} made {sum(at_or_above) / len(at_or_above):+.2f}R on average"
    else:
        reason = "not enough closed trades yet"
    return {"configured_threshold": configured, "suggested_threshold": threshold, "reason": reason,
            "closed_trades": len(trades), "min_trades": config.CALIBRATION_MIN_TRADES, "buckets": rows,
            "account_mode": mode}


def refresh(records: List[Dict[str, Any]], mode: Optional[str] = None) -> Dict[str, Any]:
    report = build_report(records, mode)
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
