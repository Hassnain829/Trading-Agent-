"""
DeepSeek-powered decision engine.

Builds a deep multi-timeframe market context for one symbol (pandas_ta
indicators and trajectories, ATR stop geometry, price structure, cross-asset
moves and the auditor's active learned rules), injects it into the system
prompt, and converts the model's JSON answer into a validated decision. The
model chooses stop/target ATR multiples; the engine computes the exact prices.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

import config
from data_engine import round_to_tick, utc_now_iso
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

ATR STOP ENGINE
You do not quote stop prices. You choose volatility multiples of H1 ATR14 and the execution
engine converts them into exact prices from the live quote (see ATR STOP GEOMETRY below):
  BUY : stop_loss = ask - sl_atr_multiple * ATR    take_profit = ask + tp_atr_multiple * ATR
  SELL: stop_loss = bid + sl_atr_multiple * ATR    take_profit = bid - tp_atr_multiple * ATR
- sl_atr_multiple between {sl_min} and {sl_max}. Put the stop beyond the nearest 10-bar H1 swing
  when that swing is within reach (its distance in ATR is given). Use 1.0-1.5 in calm, orderly
  trends (atr_ratio < 1.0); 2.0-{sl_max} when atr_ratio > 1.3 or structure is expanding/noisy.
- tp_atr_multiple between {tp_min} and {tp_max}, and at least {min_rr} x sl_atr_multiple. Scale it
  with trend strength and the D1 ATR room; do not target beyond realistic reach.
- Wider stops automatically shrink position size (fixed % risk), so never widen a stop just to
  avoid being stopped out; widen it only when volatility demands it.

ACTIVE RISK AUDIT RULES
The ACTIVE RISK AUDIT RULES in the market context below were derived by the fund's auditor from
our own losing trades. They are MANDATORY. For every rule whose condition matches the current
setup and the direction you would trade, subtract its penalty points from your confidence, list
its id in "triggered_rule_ids", and name it in "logic". Never ignore a matching rule.

CONFIDENCE CALIBRATION
0-49 no edge | 50-64 weak | 65-79 tradeable | 80-100 exceptional (rare).
Signals below {threshold} are not executed, so do not inflate scores.

OUTPUT
Respond with one raw JSON object only (no markdown, no prose outside JSON):
{{"signal": "BUY" | "SELL" | "HOLD",
  "base_confidence": <integer 0-100, before learned-rule penalties>,
  "triggered_rule_ids": [<ids of matching rules, [] if none>],
  "confidence_score": <integer 0-100, after learned-rule penalties>,
  "sl_atr_multiple": <number {sl_min}-{sl_max}>,
  "tp_atr_multiple": <number {tp_min}-{tp_max}>,
  "logic": "<= 70 words: indicator evidence, why these ATR multiples, cross-asset read, every learned-rule deduction"}}
For HOLD, give the multiples you would use for the direction you leaned towards.
Base every number in your answer on the DEEP MARKET CONTEXT below; it is the only market data."""


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


def _series(values: Any, digits: int) -> str:
    return ", ".join(_fmt(v, digits) for v in (values or []))


def _timeframe_block(label: str, data: Dict[str, Any], digits: int) -> str:
    structure = data.get("structure") or {}
    trajectory = data.get("trajectory") or {}
    closes = _series(structure.get("closes"), digits)
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
        f"  closes: {closes}\n"
        f"  trajectory, last {len(trajectory.get('rsi14') or [])} closed bars (oldest -> newest): "
        f"RSI14 [{_series(trajectory.get('rsi14'), 1)}] | "
        f"rel_volume [{_series(trajectory.get('rel_volume'), 2)}] | "
        f"ATR14 [{_series(trajectory.get('atr14'), digits + 1)}]"
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


def _atr_geometry_block(market: Dict[str, Any], digits: int) -> str:
    """ATR in price and points, distance ladder, and the nearest swings expressed in ATR."""
    atr, source = _reference_atr(market)
    if not atr:
        return "ATR STOP GEOMETRY: unavailable (no ATR); the engine will HOLD any trade signal."
    h1 = market.get("h1_data") or {}
    d1 = market.get("daily_data") or {}
    structure = h1.get("structure") or {}
    point = float(market.get("point") or 10 ** -digits)
    ask, bid = float(market["ask"]), float(market["bid"])
    ladder = " | ".join(f"{m:.1f}x = {_fmt(m * atr, digits)}" for m in (1.0, 1.5, 2.0, 2.5, 3.0))
    lines = [
        f"ATR STOP GEOMETRY ({source} = {_fmt(atr, digits + 1)} = {atr / point:.1f} points):",
        f"  distance ladder: {ladder}",
    ]
    if d1.get("atr14"):
        lines.append(f"  D1 ATR14 = {float(d1['atr14']) / atr:.2f} x H1 ATR (daily room for targets)")
    if structure.get("low") is not None:
        lines.append(f"  BUY from ask {_fmt(ask, digits)}: 10-bar H1 swing low {_fmt(structure['low'], digits)} "
                     f"is {(ask - float(structure['low'])) / atr:.2f} ATR below")
    if structure.get("high") is not None:
        lines.append(f"  SELL from bid {_fmt(bid, digits)}: 10-bar H1 swing high {_fmt(structure['high'], digits)} "
                     f"is {(float(structure['high']) - bid) / atr:.2f} ATR above")
    spread = float(market.get("spread_price") or 0.0)
    lines.append(f"  spread = {spread / atr * 100:.1f}% of 1x ATR; trades are refused when the spread exceeds "
                 f"{config.MAX_SPREAD_TO_STOP * 100:.0f}% of the stop distance")
    return "\n".join(lines)


def _market_context_block(market: Dict[str, Any], symbol: str, rules: List[Dict[str, Any]]) -> str:
    """The per-call DEEP MARKET CONTEXT section appended to the system prompt."""
    digits = int(market.get("digits", 5))
    h1 = market.get("h1_data") or {}
    d1 = market.get("daily_data") or {}
    spread_price = market.get("spread_price") or 0.0
    spread_vs_atr = (spread_price / h1["atr14"] * 100.0) if h1.get("atr14") else None
    engine = h1.get("engine") or d1.get("engine") or "pandas_ta"

    correlated = market.get("correlated_prices") or {}
    cross_lines = [
        f"  {sym}: bid {_fmt(q.get('bid'), 5)} | day change {_fmt(q.get('day_change_pct'), 3)}%"
        for sym, q in correlated.items()
    ]
    cross_asset = "\n".join(cross_lines) if cross_lines else "  unavailable"

    return (
        f"=== DEEP MARKET CONTEXT: {symbol} ===\n"
        f"Indicators: {engine} on closed bars. EMA200 is SMA-seeded; RSI14 and ATR14 use Wilder "
        f"smoothing; rel_volume = tick volume / SMA20 of tick volume; atr_ratio = ATR14 / SMA50 of "
        f"ATR14; ema_distance_atr = (close - EMA200) / ATR14.\n\n"
        f"SYMBOL: {symbol} (digits {digits})\n"
        f"LIVE QUOTE: bid {_fmt(market.get('bid'), digits)} | ask {_fmt(market.get('ask'), digits)} | "
        f"spread {market.get('spread')} points ({_fmt(spread_price, digits)}, "
        f"{_fmt(spread_vs_atr, 1)}% of H1 ATR)\n"
        f"ACCOUNT: equity {_fmt(market.get('equity'))} {market.get('currency', '')}\n\n"
        f"{_timeframe_block('DAILY (D1)', d1, digits)}\n\n"
        f"{_timeframe_block('HOURLY (H1)', h1, digits)}\n\n"
        f"{_atr_geometry_block(market, digits)}\n\n"
        f"CROSS-ASSET SNAPSHOT (change since today's D1 open):\n{cross_asset}\n\n"
        f"ACTIVE RISK AUDIT RULES (mandatory penalties when the condition matches):\n"
        f"{_rules_block(rules)}\n"
        f"=== END DEEP MARKET CONTEXT ==="
    )


def build_system_prompt(market: Dict[str, Any], symbol: str, rules: List[Dict[str, Any]]) -> str:
    """Static mandate first (a stable prefix for DeepSeek's context cache), then the live context."""
    mandate = SYSTEM_PROMPT.format(
        threshold=config.CONFIDENCE_THRESHOLD,
        sl_min=config.SL_ATR_MIN, sl_max=config.SL_ATR_MAX,
        tp_min=config.TP_ATR_MIN, tp_max=config.TP_ATR_MAX,
        min_rr=config.MIN_REWARD_RISK,
    )
    return mandate + "\n\n" + _market_context_block(market, symbol, rules)


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


# -----------------------------------------------------------------------------
# ATR stop engine
# -----------------------------------------------------------------------------
def _reference_atr(market: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """H1 ATR14 drives stops; a quarter of D1 ATR14 stands in when H1 is unavailable."""
    h1_atr = (market.get("h1_data") or {}).get("atr14")
    if h1_atr:
        return float(h1_atr), "H1 ATR14"
    d1_atr = (market.get("daily_data") or {}).get("atr14")
    if d1_atr:
        return float(d1_atr) * 0.25, "D1 ATR14 / 4"
    return None, "unavailable"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolve_atr_multiples(payload: Dict[str, Any], side: str, entry: float,
                           atr: float) -> Tuple[float, float, str, List[str]]:
    """The model's stop/target ATR multiples, bounded and R:R-checked: (sl, tp, source, notes)."""
    notes: List[str] = []
    sl = _to_float(payload.get("sl_atr_multiple"))
    tp = _to_float(payload.get("tp_atr_multiple"))

    # Tolerate a model that quoted prices instead of multiples by converting them.
    if sl is None:
        price = _to_float(payload.get("stop_loss"))
        if price and (price < entry if side == "BUY" else price > entry):
            sl = abs(entry - price) / atr
            notes.append("SL multiple derived from a quoted price")
    if tp is None:
        price = _to_float(payload.get("take_profit"))
        if price and (price > entry if side == "BUY" else price < entry):
            tp = abs(price - entry) / atr
            notes.append("TP multiple derived from a quoted price")

    from_model = int(sl is not None) + int(tp is not None)
    if sl is None:
        sl = config.SL_ATR_MULTIPLIER
        notes.append(f"no SL multiple, default {sl:.2f}x")
    if tp is None:
        tp = config.TP_ATR_MULTIPLIER
        notes.append(f"no TP multiple, default {tp:.2f}x")

    bounded_sl = _clamp(sl, config.SL_ATR_MIN, config.SL_ATR_MAX)
    if abs(bounded_sl - sl) > 1e-9:
        notes.append(f"SL {sl:.2f}x clamped to {bounded_sl:.2f}x")
    bounded_tp = _clamp(tp, config.TP_ATR_MIN, config.TP_ATR_MAX)
    if abs(bounded_tp - tp) > 1e-9:
        notes.append(f"TP {tp:.2f}x clamped to {bounded_tp:.2f}x")
    floor_tp = bounded_sl * config.MIN_REWARD_RISK
    if bounded_tp < floor_tp - 1e-9:
        notes.append(f"TP raised {bounded_tp:.2f}x -> {floor_tp:.2f}x for R:R >= {config.MIN_REWARD_RISK}")
        bounded_tp = min(floor_tp, config.TP_ATR_MAX)

    if from_model == 0:
        source = "DEFAULT_ATR"
    elif from_model == 2 and not notes:
        source = "AI_ATR"
    else:
        source = "AI_ATR_ADJUSTED"
    return round(bounded_sl, 2), round(bounded_tp, 2), source, notes


def compute_atr_stops(side: str, entry: float, atr: float, sl_multiple: float, tp_multiple: float,
                      digits: int, tick_size: float) -> Dict[str, float]:
    """Exact SL/TP prices: entry -/+ multiple x ATR, snapped to the symbol's tick grid."""
    direction = 1.0 if side == "BUY" else -1.0
    sl_distance = sl_multiple * atr
    tp_distance = tp_multiple * atr
    return {
        "stop_loss": round_to_tick(entry - direction * sl_distance, tick_size, digits),
        "take_profit": round_to_tick(entry + direction * tp_distance, tick_size, digits),
        "sl_distance": round(sl_distance, digits + 3),
        "tp_distance": round(tp_distance, digits + 3),
        "risk_reward": round(tp_multiple / sl_multiple, 2),
    }


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

    # Exact stops from ATR: the model picks multiples, the engine does the arithmetic.
    direction = raw_signal if raw_signal in ("BUY", "SELL") else None
    entry = float(market["ask"] if direction == "BUY" else market["bid"]) if direction else float(market["mid"])
    atr, atr_source = _reference_atr(market)
    tick_size = float(market.get("tick_size") or market.get("point") or 10 ** -digits)
    stops: Dict[str, Any] = {"stop_loss": None, "take_profit": None, "sl_atr_multiple": None,
                             "tp_atr_multiple": None, "sl_distance": None, "tp_distance": None,
                             "risk_reward": None, "stops_source": None, "stop_notes": []}
    if direction and not atr:
        if signal != "HOLD":
            logic = f"[STOPS] no ATR available to place stops: forced HOLD. {logic}"
        signal = "HOLD"
    elif direction:
        sl_multiple, tp_multiple, source, notes = _resolve_atr_multiples(payload, direction, entry, atr)
        levels = compute_atr_stops(direction, entry, atr, sl_multiple, tp_multiple, digits, tick_size)
        stops.update(levels, sl_atr_multiple=sl_multiple, tp_atr_multiple=tp_multiple,
                     stops_source=source, stop_notes=notes)
        if notes:
            logger.info("[AI] %s ATR stops adjusted: %s", symbol, "; ".join(notes))
        spread = float(market.get("spread_price") or 0.0)
        if signal != "HOLD" and spread > config.MAX_SPREAD_TO_STOP * levels["sl_distance"]:
            logic = (f"[COSTS] spread is {spread / levels['sl_distance']:.0%} of the {sl_multiple}x ATR stop "
                     f"(limit {config.MAX_SPREAD_TO_STOP:.0%}): forced HOLD. {logic}")
            signal = "HOLD"

    decision.update({
        "signal": signal,
        "raw_signal": raw_signal,
        "confidence_score": confidence,
        "base_confidence": base_confidence if has_base else confidence,
        **stops,
        "entry_reference": round(entry, digits),
        "atr_reference": round(atr, digits + 1) if atr else None,
        "atr_source": atr_source,
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
        "sl_atr_multiple": None,
        "tp_atr_multiple": None,
        "sl_distance": None,
        "tp_distance": None,
        "risk_reward": None,
        "entry_reference": market_data.get("mid"),
        "stops_source": None,
        "stop_notes": [],
        "atr_reference": None,
        "atr_source": None,
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
        {"role": "system", "content": build_system_prompt(market_data, symbol, rules)},
        {"role": "user", "content": f"Evaluate {symbol} using the DEEP MARKET CONTEXT in your instructions "
                                    f"and return the JSON decision object now."},
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
