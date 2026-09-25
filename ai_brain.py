"""
DeepSeek-powered decision engine.

Builds a multi-timeframe market brief for one symbol, injects the auditor's
active learned rules as mandatory confidence penalties, and converts the
model's JSON answer into a validated, risk-checked trading decision.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

import config
from data_engine import utc_now_iso
from memory_store import read_json_file

logger = logging.getLogger("hedgefund.ai")

_session = requests.Session()

SYSTEM_PROMPT = """You are the Head of Systematic Trading at a risk-managed multi-strategy hedge fund.
You allocate real capital under a strict institutional mandate. Capital preservation comes first:
HOLD is the default and a trade must be earned by aligned evidence.

DECISION FRAMEWORK
1. Daily (D1) sets the bias: price vs EMA200, EMA slope, RSI regime, 10-bar structure.
2. H1 times the entry: it must agree with the D1 bias. Counter-trend trades against D1 need
   RSI extremes plus a confirmed structure break, and their confidence is capped at 70.
3. Participation: H1 rel_volume < 0.8 means a weak, low-conviction move; reduce confidence.
4. Volatility: atr_ratio > 1.5 signals a volatility expansion/trap risk; |ema_distance_atr| > 3
   or RSI > 75 (BUY) / < 25 (SELL) means overextension; reduce confidence.
5. Costs: if the spread exceeds 20% of H1 ATR the edge is gone; HOLD.
6. Cross-asset context: check that correlated instruments are not moving against the trade
   (e.g. broad USD strength when buying EURUSD).

RISK GEOMETRY
- BUY:  stop_loss < ask < take_profit.   SELL: take_profit < bid < stop_loss.
- Stop between 1.0x and 2.5x H1 ATR from entry, beyond obvious structure.
- Take profit at least 1.5x the stop distance.

ACTIVE RISK AUDIT RULES
The fund's auditor derived these penalties from our own losing trades. They are MANDATORY.
For every rule whose condition matches the current setup and the direction you would trade,
subtract its penalty points from your confidence, list its id in "triggered_rule_ids",
and name it in "logic". Never ignore a matching rule.

CONFIDENCE CALIBRATION
0-49 no edge | 50-64 weak | 65-79 tradeable | 80-100 exceptional (rare).
Signals below {threshold} are not executed, so do not inflate scores.

OUTPUT
Respond with one raw JSON object only (no markdown, no prose outside JSON):
{{"signal": "BUY" | "SELL" | "HOLD",
  "base_confidence": <integer 0-100, before learned-rule penalties>,
  "triggered_rule_ids": [<ids of matching rules, [] if none>],
  "confidence_score": <integer 0-100, after learned-rule penalties>,
  "stop_loss": <float>,
  "take_profit": <float>,
  "logic": "<= 70 words: key indicator evidence, cross-asset read, and every learned-rule deduction"}}
For HOLD, give the stops you would use for the direction you leaned towards (or 0)."""


class DeepSeekError(RuntimeError):
    """Raised when the DeepSeek API cannot produce a usable completion."""


# -----------------------------------------------------------------------------
# DeepSeek transport (shared with the auditor)
# -----------------------------------------------------------------------------
def deepseek_chat(messages: List[Dict[str, str]], temperature: float = config.AI_TEMPERATURE,
                  max_tokens: int = 800, json_mode: bool = True) -> Dict[str, Any]:
    """POST /chat/completions with retries on network errors, 429 and 5xx."""
    if not config.DEEPSEEK_API_KEY:
        raise DeepSeekError("DEEPSEEK_API_KEY is not configured")

    url = f"{config.DEEPSEEK_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": config.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_error: Optional[DeepSeekError] = None
    for attempt in range(1, config.DEEPSEEK_MAX_RETRIES + 1):
        try:
            response = _session.post(url, json=payload, headers=headers,
                                     timeout=(10, config.DEEPSEEK_TIMEOUT_SECONDS))
        except requests.RequestException as exc:
            last_error = DeepSeekError(f"network error: {exc}")
        else:
            if response.status_code == 200:
                try:
                    body = response.json()
                    content = body["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    last_error = DeepSeekError(f"malformed response: {exc}")
                else:
                    if content and content.strip():
                        return {"content": content, "usage": body.get("usage") or {},
                                "model": body.get("model", config.DEEPSEEK_MODEL)}
                    last_error = DeepSeekError("empty completion")
            elif response.status_code in (408, 429, 500, 502, 503, 504):
                last_error = DeepSeekError(f"HTTP {response.status_code}: {response.text[:200]}")
            else:
                # 400 bad request, 401 bad key, 402 no balance: retrying will not help.
                raise DeepSeekError(f"HTTP {response.status_code}: {response.text[:300]}")
        if attempt < config.DEEPSEEK_MAX_RETRIES:
            delay = min(2 ** attempt, 10)
            logger.warning("[AI] DeepSeek attempt %d/%d failed (%s); retrying in %ds",
                           attempt, config.DEEPSEEK_MAX_RETRIES, last_error, delay)
            time.sleep(delay)
    raise last_error or DeepSeekError("DeepSeek request failed")


def parse_json_payload(text: str) -> Any:
    """Parse model output that should be JSON, tolerating code fences and stray prose."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char in "{[":
            try:
                value, _ = decoder.raw_decode(cleaned[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in model response")


# -----------------------------------------------------------------------------
# Learned rules
# -----------------------------------------------------------------------------
def load_learned_rules(symbol: str) -> List[Dict[str, Any]]:
    """Active auditor rules that target this symbol or ALL symbols."""
    document = read_json_file(config.RULES_FILE, lambda: {"rules": []}, quarantine_corrupt=False)
    rules = document.get("rules", []) if isinstance(document, dict) else []
    wanted = {symbol.upper(), "ALL"}
    return [
        rule for rule in rules
        if isinstance(rule, dict)
        and str(rule.get("status", "ACTIVE")).upper() == "ACTIVE"
        and str(rule.get("affected_symbol", "")).upper() in wanted
    ]


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _timeframe_block(label: str, data: Dict[str, Any], digits: int) -> str:
    structure = data.get("structure") or {}
    closes = ", ".join(_fmt(c, digits) for c in structure.get("closes", []))
    return (
        f"{label} (last closed bar {data.get('last_bar_time', 'n/a')}):\n"
        f"  close {_fmt(data.get('close'), digits)} | EMA200 {_fmt(data.get('ema200'), digits)} "
        f"({data.get('price_vs_ema') or 'n/a'}, slope {data.get('ema_slope') or 'n/a'}, "
        f"distance {_fmt(data.get('ema_distance_atr'))} ATR)\n"
        f"  RSI14 {_fmt(data.get('rsi14'))} | ATR14 {_fmt(data.get('atr14'), digits + 1)} "
        f"({_fmt(data.get('atr_pct'), 3)}% of price, atr_ratio {_fmt(data.get('atr_ratio'))} vs 50-bar mean)\n"
        f"  rel_volume {_fmt(data.get('rel_volume'))} (tick volume {data.get('tick_volume', 'n/a')} "
        f"vs 20-bar mean {_fmt(data.get('volume_ma20'), 0)})\n"
        f"  10-bar structure {structure.get('pattern', 'n/a')}, bias {structure.get('bias', 'n/a')}, "
        f"change {_fmt(structure.get('change_pct'), 2)}%, range {_fmt(structure.get('low'), digits)}"
        f"-{_fmt(structure.get('high'), digits)}, HH {structure.get('higher_highs', 'n/a')} / "
        f"LL {structure.get('lower_lows', 'n/a')}\n"
        f"  closes: {closes}"
    )


def _rules_block(rules: List[Dict[str, Any]]) -> str:
    if not rules:
        return "None: no learned penalties currently apply to this symbol."
    lines = []
    for rule in rules:
        lines.append(
            f"- [{rule.get('id', '?')}] target {rule.get('affected_symbol')} | penalty "
            f"-{rule.get('confidence_reduction_points')} | condition: {rule.get('setup')} | "
            f"evidence (n={rule.get('sample_size')}): {rule.get('evidence')}"
        )
    return "\n".join(lines)


def _build_market_prompt(market: Dict[str, Any], symbol: str, rules: List[Dict[str, Any]]) -> str:
    digits = int(market.get("digits", 5))
    h1 = market.get("h1_data") or {}
    d1 = market.get("daily_data") or {}
    spread_price = market.get("spread_price") or 0.0
    spread_vs_atr = (spread_price / h1["atr14"] * 100.0) if h1.get("atr14") else None

    correlated = market.get("correlated_prices") or {}
    cross_lines = [
        f"  {sym}: bid {_fmt(q.get('bid'), 5)} | day change {_fmt(q.get('day_change_pct'), 3)}%"
        for sym, q in correlated.items()
    ]
    cross_asset = "\n".join(cross_lines) if cross_lines else "  unavailable"

    return (
        f"SYMBOL: {symbol} (digits {digits})\n"
        f"LIVE QUOTE: bid {_fmt(market.get('bid'), digits)} | ask {_fmt(market.get('ask'), digits)} | "
        f"spread {market.get('spread')} points ({_fmt(spread_price, digits)}, "
        f"{_fmt(spread_vs_atr, 1)}% of H1 ATR)\n"
        f"ACCOUNT: equity {_fmt(market.get('equity'))} {market.get('currency', '')}\n\n"
        f"{_timeframe_block('DAILY (D1)', d1, digits)}\n\n"
        f"{_timeframe_block('HOURLY (H1)', h1, digits)}\n\n"
        f"CROSS-ASSET SNAPSHOT (change since today's D1 open):\n{cross_asset}\n\n"
        f"ACTIVE RISK AUDIT RULES (mandatory penalties when the condition matches):\n"
        f"{_rules_block(rules)}\n\n"
        f"Return the JSON decision object now."
    )


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def _to_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _stops_problem(side: str, price: float, stop_loss: Optional[float], take_profit: Optional[float],
                   atr: Optional[float], spread: float) -> Optional[str]:
    """Describe why stops are unusable, or None when they are valid."""
    if stop_loss is None or take_profit is None:
        return "missing stop_loss/take_profit"
    if side == "BUY" and not (stop_loss < price < take_profit):
        return f"BUY geometry violated (SL {stop_loss} < price {price} < TP {take_profit} required)"
    if side == "SELL" and not (take_profit < price < stop_loss):
        return f"SELL geometry violated (TP {take_profit} < price {price} < SL {stop_loss} required)"
    risk = abs(price - stop_loss)
    if atr:
        if risk < 0.3 * atr:
            return f"stop {risk:.6g} sits inside noise (< 0.3x H1 ATR)"
        if risk > 10 * atr:
            return f"stop {risk:.6g} is unrealistically wide (> 10x H1 ATR)"
    if spread and risk < 2 * spread:
        return "stop closer than 2x the spread"
    return None


def _reference_atr(market: Dict[str, Any]) -> Optional[float]:
    h1_atr = (market.get("h1_data") or {}).get("atr14")
    if h1_atr:
        return float(h1_atr)
    d1_atr = (market.get("daily_data") or {}).get("atr14")
    return float(d1_atr) * 0.25 if d1_atr else None


def _atr_stops(side: str, price: float, atr: float, digits: int) -> Tuple[float, float]:
    if side == "BUY":
        return (round(price - config.SL_ATR_MULTIPLIER * atr, digits),
                round(price + config.TP_ATR_MULTIPLIER * atr, digits))
    return (round(price + config.SL_ATR_MULTIPLIER * atr, digits),
            round(price - config.TP_ATR_MULTIPLIER * atr, digits))


def _validate_decision(payload: Dict[str, Any], market: Dict[str, Any], symbol: str,
                       rules: List[Dict[str, Any]], decision: Dict[str, Any]) -> Dict[str, Any]:
    digits = int(market.get("digits", 5))
    signal = str(payload.get("signal", "HOLD")).strip().upper()
    if signal not in ("BUY", "SELL", "HOLD"):
        signal = "HOLD"

    confidence = max(0, min(100, _to_int(payload.get("confidence_score"), 0)))
    has_base = "base_confidence" in payload
    base_confidence = max(0, min(100, _to_int(payload.get("base_confidence"), confidence)))

    # Enforce learned penalties deterministically for every rule the model flagged.
    rules_by_id = {str(rule.get("id", "")).upper(): rule for rule in rules if rule.get("id")}
    triggered: List[Dict[str, Any]] = []
    for rule_id in payload.get("triggered_rule_ids") or []:
        rule = rules_by_id.get(str(rule_id).strip().upper())
        if rule is not None and rule not in triggered:
            triggered.append(rule)
    penalty = sum(_to_int(rule.get("confidence_reduction_points"), 0) for rule in triggered)
    if triggered and has_base:
        enforced = max(0, base_confidence - penalty)
        if enforced < confidence:
            logger.info("[LEARNING] %s: enforcing learned penalties -%d (model %d -> %d)",
                        symbol, penalty, confidence, enforced)
            confidence = enforced

    logic = re.sub(r"\s+", " ", str(payload.get("logic", ""))).strip()[:1200] or "No rationale supplied."
    raw_signal = signal
    if signal != "HOLD" and confidence < config.CONFIDENCE_THRESHOLD:
        logic = (f"[THRESHOLD] {signal} conviction {confidence} < {config.CONFIDENCE_THRESHOLD}: "
                 f"forced HOLD. {logic}")
        signal = "HOLD"

    stop_loss = _to_float(payload.get("stop_loss"))
    take_profit = _to_float(payload.get("take_profit"))
    stops_source = "MODEL"
    direction = raw_signal if raw_signal in ("BUY", "SELL") else None
    price = float(market["ask"] if direction == "BUY" else market["bid"]) if direction else float(market["mid"])
    atr = _reference_atr(market)

    if direction:
        problem = _stops_problem(direction, price, stop_loss, take_profit, atr,
                                 float(market.get("spread_price") or 0.0))
        if problem:
            if atr:
                stop_loss, take_profit = _atr_stops(direction, price, atr, digits)
                stops_source = "ATR_FALLBACK"
                logger.info("[AI] %s: model stops rejected (%s); ATR fallback SL %s / TP %s",
                            symbol, problem, stop_loss, take_profit)
            else:
                logic = f"[STOPS] {problem} and no ATR available: forced HOLD. {logic}"
                signal, stop_loss, take_profit, stops_source = "HOLD", None, None, None
    else:
        stop_loss = take_profit = None
        stops_source = None

    decision.update({
        "signal": signal,
        "raw_signal": raw_signal,
        "confidence_score": confidence,
        "base_confidence": base_confidence if has_base else confidence,
        "stop_loss": round(stop_loss, digits) if stop_loss else None,
        "take_profit": round(take_profit, digits) if take_profit else None,
        "entry_reference": round(price, digits),
        "stops_source": stops_source,
        "atr_reference": round(atr, digits + 1) if atr else None,
        "logic": logic,
        "applied_rules": [
            {"id": rule.get("id"), "points": _to_int(rule.get("confidence_reduction_points"), 0),
             "setup": rule.get("setup")}
            for rule in triggered
        ],
    })
    return decision


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
def get_ai_decision(market_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """Ask DeepSeek for a trade decision; any failure degrades safely to HOLD."""
    rules = load_learned_rules(symbol)
    decision: Dict[str, Any] = {
        "symbol": symbol,
        "timestamp": utc_now_iso(),
        "signal": "HOLD",
        "raw_signal": "HOLD",
        "confidence_score": 0,
        "base_confidence": 0,
        "stop_loss": None,
        "take_profit": None,
        "entry_reference": market_data.get("mid"),
        "stops_source": None,
        "atr_reference": None,
        "logic": "",
        "applied_rules": [],
        "active_rules_count": len(rules),
        "latency_ms": None,
        "error": None,
    }

    if not config.DEEPSEEK_API_KEY:
        decision["logic"] = "DeepSeek API key not configured; engine holds all positions flat."
        decision["error"] = "missing_api_key"
        return decision

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(threshold=config.CONFIDENCE_THRESHOLD)},
        {"role": "user", "content": _build_market_prompt(market_data, symbol, rules)},
    ]
    started = time.perf_counter()
    try:
        response = deepseek_chat(messages, temperature=config.AI_TEMPERATURE, max_tokens=700, json_mode=True)
        payload = parse_json_payload(response["content"])
        if not isinstance(payload, dict):
            raise ValueError("model returned JSON that is not an object")
    except (DeepSeekError, ValueError) as exc:
        logger.error("[AI] %s decision failed: %s", symbol, exc)
        decision["logic"] = f"AI decision unavailable ({exc}); defaulting to HOLD."
        decision["error"] = str(exc)
        return decision
    finally:
        decision["latency_ms"] = int((time.perf_counter() - started) * 1000)

    return _validate_decision(payload, market_data, symbol, rules, decision)
