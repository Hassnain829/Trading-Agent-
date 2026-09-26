"""
Machine-checkable conditions for learned rules.

A learned rule is a list of numeric conditions, all of which must hold:

    {"metric": "h1.rel_volume", "op": "<", "value": 0.8}
    {"metric": "h1.atr_ratio", "op": ">", "value": 1.3}
    {"metric": "cross.USDJPY.day_change_pct", "op": ">", "value": 0.3}

The same evaluation runs on a closed trade's stored context (the auditor counts
the matching wins and losses itself) and on the live market snapshot (the
decision engine applies the penalty itself). The AI proposes rules; it never
decides whether one matches.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional

from data_engine import symbols_match

_TIMEFRAME_FIELDS = ("rel_volume", "atr_ratio", "atr_pct", "rsi14", "ema_distance_atr")
_CROSS_FIELDS = ("day_change_pct", "corr_h1")
OPERATORS = {"<": float.__lt__, "<=": float.__le__, ">": float.__gt__, ">=": float.__ge__}
MAX_CONDITIONS = 5

CATEGORY_VOLUME = "volume"
CATEGORY_VOLATILITY = "volatility"
CATEGORY_CROSS = "cross_asset"
CATEGORY_TREND = "trend"

# Metrics that can be checked both on a stored trade and on the live market.
FIXED_METRICS = {
    **{f"{tf}.rel_volume": CATEGORY_VOLUME for tf in ("h1", "d1")},
    **{f"{tf}.{field}": CATEGORY_VOLATILITY for tf in ("h1", "d1") for field in ("atr_ratio", "atr_pct")},
    **{f"{tf}.{field}": CATEGORY_TREND for tf in ("h1", "d1") for field in ("rsi14", "ema_distance_atr")},
    "day_change_pct": CATEGORY_TREND,
    "usd.change_pct": CATEGORY_CROSS,
}


def metric_vocabulary() -> str:
    """The metric names, for the auditor's prompt."""
    return (", ".join(sorted(FIXED_METRICS)) + ", cross.<SYMBOL>.day_change_pct, cross.<SYMBOL>.corr_h1")


# -----------------------------------------------------------------------------
# Snapshots: the same shape from a stored trade and from the live market
# -----------------------------------------------------------------------------
def snapshot_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    context = record.get("market_context") or {}
    return {
        "h1": context.get("h1") or {},
        "d1": context.get("d1") or {},
        "day_change_pct": context.get("day_change_pct"),
        "cross": context.get("correlated_prices") or {},
        "usd": context.get("usd_direction") or {},
    }


def snapshot_from_market(market: Dict[str, Any], usd: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "h1": market.get("h1_data") or {},
        "d1": market.get("daily_data") or {},
        "day_change_pct": market.get("day_change_pct"),
        "cross": market.get("correlated_prices") or {},
        "usd": usd or {},
    }


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def metric_value(metric: str, snapshot: Dict[str, Any]) -> Optional[float]:
    if metric.startswith("cross."):
        _, symbol, field = metric.split(".", 2)
        cross = snapshot.get("cross") or {}
        quote = cross.get(symbol) or next((q for s, q in cross.items() if symbols_match(s, symbol)), None)
        return _number((quote or {}).get(field))
    if metric == "usd.change_pct":
        return _number((snapshot.get("usd") or {}).get("usd_change_pct"))
    if metric == "day_change_pct":
        return _number(snapshot.get("day_change_pct"))
    timeframe, field = metric.split(".", 1)
    return _number((snapshot.get(timeframe) or {}).get(field))


def flatten(snapshot: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Every metric value in a snapshot (for the auditor's trade brief)."""
    values = {metric: metric_value(metric, snapshot) for metric in FIXED_METRICS}
    for symbol in (snapshot.get("cross") or {}):
        for field in _CROSS_FIELDS:
            values[f"cross.{symbol}.{field}"] = metric_value(f"cross.{symbol}.{field}", snapshot)
    return {key: (round(value, 4) if value is not None else None) for key, value in values.items()}


# -----------------------------------------------------------------------------
# Conditions
# -----------------------------------------------------------------------------
def category(metric: str) -> str:
    return CATEGORY_CROSS if metric.startswith("cross.") else FIXED_METRICS.get(metric, CATEGORY_TREND)


def normalize_condition(raw: Any, known_symbols: Iterable[str]) -> Optional[Dict[str, Any]]:
    """A clean condition, or None when the metric/operator/value is not usable."""
    if not isinstance(raw, dict):
        return None
    metric = re.sub(r"\s+", "", str(raw.get("metric") or ""))
    op = str(raw.get("op") or "").strip()
    value = _number(raw.get("value"))
    if op not in OPERATORS or value is None or not metric:
        return None
    if metric.lower().startswith("cross."):
        parts = metric.split(".")
        if len(parts) != 3 or parts[2].lower() not in _CROSS_FIELDS:
            return None
        known = [s for s in known_symbols if symbols_match(s, parts[1])]
        if not known:
            return None
        metric = f"cross.{known[0]}.{parts[2].lower()}"
        if parts[2].lower() == "corr_h1" and not -1.0 <= value <= 1.0:
            return None
    else:
        metric = metric.lower()
        if metric not in FIXED_METRICS:
            return None
    return {"metric": metric, "op": op, "value": round(value, 4)}


def normalize_conditions(raw: Any, known_symbols: Iterable[str]) -> List[Dict[str, Any]]:
    known = list(known_symbols)
    conditions: List[Dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        condition = normalize_condition(item, known)
        if condition and condition not in conditions:
            conditions.append(condition)
    return conditions[:MAX_CONDITIONS]


def categories(conditions: Iterable[Dict[str, Any]]) -> set:
    return {category(condition["metric"]) for condition in conditions}


def evaluate(conditions: List[Dict[str, Any]], snapshot: Dict[str, Any]) -> Optional[bool]:
    """True when every condition holds, False when one fails, None when data for one is missing."""
    if not conditions:
        return None
    unknown = False
    for condition in conditions:
        value = metric_value(condition["metric"], snapshot)
        if value is None:
            unknown = True
            continue
        if not OPERATORS[condition["op"]](value, float(condition["value"])):
            return False
    return None if unknown else True


def describe(conditions: Iterable[Dict[str, Any]]) -> str:
    return " AND ".join(f"{c['metric']} {c['op']} {c['value']:g}" for c in conditions)


def correlated_symbol(conditions: Iterable[Dict[str, Any]]) -> Optional[str]:
    for condition in conditions:
        if condition["metric"].startswith("cross."):
            return condition["metric"].split(".")[1]
        if condition["metric"] == "usd.change_pct":
            return "USD"
    return None
