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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

import calibration
import config
import news
import rule_engine
from data_engine import news_currencies, round_to_tick, symbols_match, utc_now_iso
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
4. Volatility: atr_ratio > 1.5 signals a volatility expansion/trap risk; reduce confidence.
   Overextension (price many ATRs from EMA200, RSI extremes) and trading against the broad USD
   move are penalised automatically by the engine after you answer: do NOT deduct for them
   yourself, but do mention them in logic when present.
5. Costs: if the spread exceeds 20% of H1 ATR the edge is gone; HOLD.
6. Cross-asset context: read the USD DIRECTION section, which is computed for you. A pair quoted
   XXXUSD (EURUSD, GBPUSD, AUDUSD) rising means USD WEAKNESS; a USDXXX pair (USDJPY, USDCAD,
   USDCHF) rising means USD STRENGTH. Never contradict the computed USD direction.

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

LEARNED RISK RULES AND NEWS
The LEARNED RISK RULES in the market context were derived by the fund's auditor from our own losing
trades. The engine checks their numeric conditions and applies their penalties itself after you answer,
so do NOT deduct for them; the context tells you which ones currently match each direction, so weigh
the underlying risk and mention matching rules in logic. The same goes for the HIGH-IMPACT NEWS list:
new entries are blocked around those events, and a trade must be able to survive the next one.

CONFIDENCE CALIBRATION
0-49 no edge | 50-64 weak | 65-79 tradeable | 80-100 exceptional (rare).
Signals below {threshold} are not executed, so do not inflate scores.

OUTPUT
Respond with one raw JSON object only (no markdown, no prose outside JSON):
{{"signal": "BUY" | "SELL" | "HOLD",
  "confidence_score": <integer 0-100, your own conviction; the engine applies rule and guard penalties>,
  "sl_atr_multiple": <number {sl_min}-{sl_max}>,
  "tp_atr_multiple": <number {tp_min}-{tp_max}>,
  "logic": "<= 70 words: indicator evidence, why these ATR multiples, cross-asset read, matching rules or news"}}
For HOLD, give the multiples you would use for the direction you leaned towards.
Base every number in your answer on the DEEP MARKET CONTEXT below; it is the only market data."""


class DeepSeekError(RuntimeError):
    """Raised when the DeepSeek API cannot produce a usable completion."""


# -----------------------------------------------------------------------------
# DeepSeek transport (shared with the auditor)
# -----------------------------------------------------------------------------
class _AccountError(DeepSeekError):
    """The key/account was rejected (401/402/403): no other model will work either."""


_RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504)
# Upper bound for one decision across retries and fallback models, so a slow provider cannot stall a scan.
LLM_CALL_BUDGET_SECONDS = 300


def _reasoning_options() -> Dict[str, Any]:
    """Request fields that switch off the thinking phase of reasoning models (LLM_REASONING=off)."""
    if config.LLM_REASONING != "off":
        return {}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _message_text(message: Dict[str, Any]) -> str:
    """The answer text; some reasoning models leave content empty and put everything in reasoning."""
    content = message.get("content") or ""
    if content.strip():
        return content
    return message.get("reasoning_content") or message.get("reasoning") or ""


def _chat_with_model(model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int,
                     json_mode: bool, deadline: float) -> Dict[str, Any]:
    """One model with retries. Raises DeepSeekError when this model cannot produce a usable answer."""
    url = f"{config.DEEPSEEK_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    use_json_mode = json_mode
    last_error: Optional[DeepSeekError] = None
    for attempt in range(1, config.DEEPSEEK_MAX_RETRIES + 1):
        remaining = deadline - time.monotonic()
        if remaining < 15:
            raise last_error or DeepSeekError("time budget for this decision used up")
        payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                   "max_tokens": max_tokens, "stream": False, **_reasoning_options()}
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = _session.post(url, json=payload, headers=headers,
                                     timeout=(10, min(config.DEEPSEEK_TIMEOUT_SECONDS, remaining)))
        except requests.RequestException as exc:
            last_error = DeepSeekError(f"network error: {exc}")
        else:
            status = response.status_code
            if status in (401, 402, 403):
                raise _AccountError(f"HTTP {status}: {response.text[:300]}")
            if status == 404:
                raise DeepSeekError("model not available for this key (HTTP 404)")
            if status == 400 and use_json_mode:
                use_json_mode = False  # this model rejects JSON mode: ask again without it
                last_error = DeepSeekError(f"HTTP 400 with JSON mode: {response.text[:160]}")
                logger.info("[AI] %s rejected JSON mode; retrying without it", model)
                continue
            if status == 200:
                try:
                    body = response.json()
                    choice = body["choices"][0]
                    text = _message_text(choice["message"])
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    last_error = DeepSeekError(f"malformed response: {exc}")
                else:
                    try:
                        if json_mode:
                            parse_json_payload(text)  # only accept answers that contain the JSON we need
                        return {"content": text, "usage": body.get("usage") or {}, "model": body.get("model", model)}
                    except ValueError:
                        if choice.get("finish_reason") == "length":
                            last_error = DeepSeekError(f"ran out of tokens ({max_tokens}) before answering; "
                                                       f"raise LLM_MAX_TOKENS or set LLM_REASONING=off")
                        else:
                            last_error = DeepSeekError("answer contained no JSON")
                        if use_json_mode:
                            use_json_mode = False  # some models answer empty in JSON mode
                            continue
            elif status in _RETRYABLE_STATUS:
                last_error = DeepSeekError(f"HTTP {status}: {response.text[:160]}")
            else:
                raise DeepSeekError(f"HTTP {status}: {response.text[:300]}")
        if attempt < config.DEEPSEEK_MAX_RETRIES:
            delay = min(2 ** attempt, 10)
            logger.warning("[AI] %s attempt %d/%d failed (%s); retrying in %ds",
                           model, attempt, config.DEEPSEEK_MAX_RETRIES, last_error, delay)
            time.sleep(delay)
    raise last_error or DeepSeekError("request failed")


def deepseek_chat(messages: List[Dict[str, str]], temperature: float = config.AI_TEMPERATURE,
                  max_tokens: Optional[int] = None, json_mode: bool = True) -> Dict[str, Any]:
    """
    POST /chat/completions to the configured model, then to each LLM_FALLBACK_MODELS entry,
    returning the first usable answer. Works with any OpenAI-compatible endpoint.
    """
    if not config.DEEPSEEK_API_KEY:
        raise DeepSeekError("LLM_API_KEY (or DEEPSEEK_API_KEY) is not configured")
    deadline = time.monotonic() + LLM_CALL_BUDGET_SECONDS
    models = [config.DEEPSEEK_MODEL, *config.LLM_FALLBACK_MODELS]
    failures: List[str] = []
    for index, model in enumerate(models):
        try:
            result = _chat_with_model(model, messages, temperature, max_tokens or config.LLM_MAX_TOKENS,
                                      json_mode, deadline)
        except _AccountError:
            raise
        except DeepSeekError as exc:
            failures.append(f"{model}: {exc}")
            if index + 1 < len(models):
                logger.warning("[AI] %s failed (%s); trying fallback %s", model, exc, models[index + 1])
            continue
        if index:
            logger.info("[AI] Answered by fallback model %s", model)
        return result
    raise DeepSeekError(" | ".join(failures))


def parse_json_payload(text: str) -> Any:
    """
    Parse model output that should be JSON, tolerating code fences, <think> blocks and prose.
    When several JSON values appear, the last object is the answer (reasoning text comes first).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    found: List[Any] = []
    index = 0
    while index < len(cleaned):
        if cleaned[index] in "{[":
            try:
                value, end = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                index += 1
                continue
            found.append(value)
            index += end
        else:
            index += 1
    objects = [value for value in found if isinstance(value, dict) and value]
    if objects:
        return objects[-1]
    if found:
        return found[-1]
    raise ValueError("no JSON object found in model response")


# -----------------------------------------------------------------------------
# Learned rules
# -----------------------------------------------------------------------------
def rule_expired(rule: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    expires = rule.get("expires_at")
    if not expires:
        return False
    try:
        when = datetime.fromisoformat(str(expires))
    except ValueError:
        return False
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return when <= (now or datetime.now(timezone.utc))


def rule_applies(rule: Dict[str, Any], symbol: str, account: Optional[Dict[str, Any]] = None) -> bool:
    """
    Whether an active rule applies to ``symbol`` on the account in use.

    Rules match by pair, so one learned on EURUSD (e.g. on a demo) also covers EURUSDm,
    EURUSD.r... elsewhere. With RULE_SCOPE=BROKER a rule only applies at the broker(s)
    it was learned at; rules from before brokers were recorded apply everywhere.
    Only rules with machine-checkable conditions that have not expired apply.
    """
    if str(rule.get("status", "ACTIVE")).upper() != "ACTIVE" or not rule.get("conditions") or rule_expired(rule):
        return False
    target = str(rule.get("affected_symbol", ""))
    if target.upper() != "ALL" and not symbols_match(target, symbol):
        return False
    if config.RULE_SCOPE == "BROKER":
        broker = (account or config.ACTIVE_ACCOUNT).get("broker")
        learned_at = rule.get("learned_brokers") or []
        if broker and learned_at and broker not in learned_at:
            return False
    return True


def load_learned_rules(symbol: str) -> List[Dict[str, Any]]:
    """Active auditor rules that apply to this symbol (or ALL symbols) on the current account."""
    document = read_json_file(config.RULES_FILE, lambda: {"rules": []}, quarantine_corrupt=False)
    rules = document.get("rules", []) if isinstance(document, dict) else []
    return [rule for rule in rules if isinstance(rule, dict) and rule_applies(rule, symbol)]


def matching_rules(rules: List[Dict[str, Any]], market: Dict[str, Any], side: str) -> List[Dict[str, Any]]:
    """Rules whose side and every numeric condition hold on the live market (evaluated by code)."""
    if side not in ("BUY", "SELL"):
        return []
    snapshot = rule_engine.snapshot_from_market(market, usd_direction(market))
    return [rule for rule in rules
            if str(rule.get("side") or "ANY").upper() in ("ANY", side)
            and rule_engine.evaluate(rule.get("conditions") or [], snapshot) is True]


def localize_conditions(conditions: List[Dict[str, Any]], names: List[str]) -> List[Dict[str, Any]]:
    """Conditions with cross-asset symbols in this broker's spelling (cross.USDJPY.* -> cross.USDJPYm.*)."""
    localized = []
    for condition in conditions:
        metric = condition["metric"]
        if metric.startswith("cross."):
            _, symbol, field = metric.split(".", 2)
            symbol = next((name for name in names if symbols_match(name, symbol)), symbol)
            metric = f"cross.{symbol}.{field}"
        localized.append({**condition, "metric": metric})
    return localized


def rule_penalties(rules: List[Dict[str, Any]], market: Dict[str, Any], side: str) -> List[Dict[str, Any]]:
    """The learned-rule penalties for a trade, capped at RULE_TOTAL_PENALTY_CAP in total."""
    budget = config.RULE_TOTAL_PENALTY_CAP
    applied = []
    for rule in sorted(matching_rules(rules, market, side),
                       key=lambda r: -int(r.get("confidence_reduction_points") or 0)):
        points = min(int(rule.get("confidence_reduction_points") or 0), budget)
        if points <= 0:
            continue
        budget -= points
        names = [market.get("symbol") or "", *config.SYMBOLS]
        applied.append({"id": rule.get("id"), "points": points, "kind": "rule",
                        "setup": rule_engine.describe(localize_conditions(rule.get("conditions") or [], names))})
    return applied


_SYMBOL_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9._+#]{4,}")


def localize_symbols(text: str, names: List[str]) -> str:
    """Rewrite symbol names in rule text to this broker's spelling (USDJPY -> USDJPYm) for the AI."""
    def swap(match: "re.Match[str]") -> str:
        token = match.group(0)
        for name in names:
            if token.upper() != name.upper() and symbols_match(token, name):
                return name
        return token
    return _SYMBOL_TOKEN.sub(swap, text or "")


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


def _rules_block(rules: List[Dict[str, Any]], symbol: str, market: Optional[Dict[str, Any]] = None) -> str:
    if not rules:
        return "None: no learned penalties currently apply to this symbol."
    names = list(dict.fromkeys([symbol, *config.SYMBOLS]))
    matches = {side: {r.get("id") for r in matching_rules(rules, market or {}, side)} for side in ("BUY", "SELL")}
    lines = []
    for rule in rules:
        side = str(rule.get("side") or "ANY").upper()
        now = [s for s in ("BUY", "SELL") if rule.get("id") in matches[s]]
        lines.append(
            f"- [{rule.get('id', '?')}] {side} {symbol} | penalty -{rule.get('confidence_reduction_points')} | "
            f"conditions: {rule_engine.describe(localize_conditions(rule.get('conditions') or [], names))} | "
            f"evidence: {rule.get('sample_size')} losses vs {rule.get('winning_matches', 0)} wins | "
            f"matches now: {', '.join(now) if now else 'no'}"
        )
    return "\n".join(lines)


def _news_block(symbol: str) -> str:
    if not config.NEWS_GUARD:
        return "HIGH-IMPACT NEWS: news guard off."
    state = news.status()
    if not state["has_data"]:
        return "HIGH-IMPACT NEWS: calendar unavailable."
    events = news.upcoming(symbol, hours=24)
    if not events:
        return f"HIGH-IMPACT NEWS (next 24h, {', '.join(news_currencies(symbol))}): none."
    listed = "; ".join(f"{e['time']} {e['currency']} {e['title']}" for e in events[:6])
    return f"HIGH-IMPACT NEWS (next 24h, UTC): {listed}"


# -----------------------------------------------------------------------------
# USD direction and protection guards (deterministic, applied by the engine)
# -----------------------------------------------------------------------------
_FX_CODES = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK", "SGD", "HKD", "MXN",
             "ZAR", "TRY", "PLN", "CNH", "HUF", "CZK"}
USD_TREND_THRESHOLD_PCT = 0.10
USD_AGREEMENT_FOR_GUARD = 0.75


def fx_pair(symbol: str) -> Optional[Tuple[str, str]]:
    """('USD', 'CAD') for USDCAD / USDCADm / USDCAD.r; None for metals, crypto and indices."""
    letters = re.sub(r"[^A-Z]", "", (symbol or "").upper())
    base, quote = letters[:3], letters[3:6]
    return (base, quote) if base in _FX_CODES and quote in _FX_CODES else None


_USD_DIRECTION_CODES = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"}


def usd_direction(market: Dict[str, Any]) -> Dict[str, Any]:
    """
    Today's broad USD move from every USD forex pair available: XXXUSD falling or USDXXX
    rising counts as USD strength. Returns the average move, its label and how many pairs agree.
    """
    quotes = dict(market.get("correlated_prices") or {})
    if market.get("symbol") and market.get("day_change_pct") is not None:
        quotes[market["symbol"]] = {"day_change_pct": market["day_change_pct"]}
    moves: Dict[str, float] = {}
    for symbol, quote in quotes.items():
        pair = fx_pair(symbol)
        change = quote.get("day_change_pct")
        if not pair or "USD" not in pair or change is None:
            continue
        if not set(pair) <= _USD_DIRECTION_CODES:  # exotics (USDTRY, USDZAR...) trend on their own
            continue
        moves[symbol] = float(change) if pair[0] == "USD" else -float(change)
    if len(moves) < 3:
        return {"label": "UNKNOWN", "usd_change_pct": None, "pairs": len(moves), "agreement": None, "moves": moves}
    average = sum(moves.values()) / len(moves)
    agreeing = sum(1 for value in moves.values() if (value > 0) == (average > 0) and value != 0)
    label = ("STRENGTHENING" if average >= USD_TREND_THRESHOLD_PCT
             else "WEAKENING" if average <= -USD_TREND_THRESHOLD_PCT else "MIXED")
    return {"label": label, "usd_change_pct": round(average, 3), "pairs": len(moves),
            "agreement": round(agreeing / len(moves), 2), "moves": {k: round(v, 3) for k, v in moves.items()}}


def usd_exposure(symbol: str, side: str) -> Optional[str]:
    """'LONG_USD' / 'SHORT_USD' for a USD forex trade, else None."""
    pair = fx_pair(symbol)
    if not pair or "USD" not in pair or side not in ("BUY", "SELL"):
        return None
    return "LONG_USD" if (pair[0] == "USD") == (side == "BUY") else "SHORT_USD"


def protective_guards(market: Dict[str, Any], symbol: str, side: str) -> List[Dict[str, Any]]:
    """Confidence penalties the engine applies itself, whatever the model concluded."""
    guards: List[Dict[str, Any]] = []
    direction = 1.0 if side == "BUY" else -1.0
    if config.OVEREXTENSION_GUARD:
        h1 = market.get("h1_data") or {}
        d1 = market.get("daily_data") or {}
        h1_stretch = float(h1.get("ema_distance_atr") or 0.0) * direction
        points = 15 if h1_stretch > 10 else 10 if h1_stretch > 6 else 5 if h1_stretch > 3 else 0
        if points:
            guards.append({"id": "G-OVEREXT-H1", "points": points,
                           "setup": f"price {h1_stretch:.1f} H1 ATRs beyond the H1 EMA200 in the trade direction"})
        d1_stretch = float(d1.get("ema_distance_atr") or 0.0) * direction
        points = 10 if d1_stretch > 5 else 5 if d1_stretch > 3 else 0
        if points:
            guards.append({"id": "G-OVEREXT-D1", "points": points,
                           "setup": f"price {d1_stretch:.1f} D1 ATRs beyond the D1 EMA200 in the trade direction"})
        for label, rsi in (("D1", d1.get("rsi14")), ("H1", h1.get("rsi14"))):
            if rsi is None:
                continue
            stretch = float(rsi) - 50.0 if side == "BUY" else 50.0 - float(rsi)  # 25 == RSI 75 for a BUY
            if label == "D1":  # D1 RSI > 75 (< 25 for SELL): -10, > 70 (< 30): -5
                points = 10 if stretch > 25 else 5 if stretch > 20 else 0
            else:  # H1 RSI > 75 (< 25 for SELL): -5
                points = 5 if stretch > 25 else 0
            if points:
                guards.append({"id": f"G-RSI-{label}", "points": points,
                               "setup": f"{label} RSI {float(rsi):.1f} already stretched in the trade direction"})
    usd = usd_direction(market)
    exposure = usd_exposure(symbol, side)
    if (exposure and usd["label"] in ("STRENGTHENING", "WEAKENING")
            and (usd["agreement"] or 0) >= USD_AGREEMENT_FOR_GUARD
            and ((exposure == "LONG_USD") != (usd["label"] == "STRENGTHENING"))):
        guards.append({"id": "G-USD", "points": 10,
                       "setup": f"{side} {symbol} is {exposure.replace('_', ' ').lower()} while USD is "
                                f"{usd['label'].lower()} ({usd['usd_change_pct']:+.2f}% avg, "
                                f"{usd['agreement']:.0%} of {usd['pairs']} pairs agree)"})

    budget = config.GUARD_MAX_PENALTY  # keep the combined guard penalty bounded
    for guard in guards:
        guard["points"] = min(guard["points"], budget)
        budget -= guard["points"]
    return [guard for guard in guards if guard["points"] > 0]


def _usd_block(market: Dict[str, Any], symbol: str) -> str:
    usd = usd_direction(market)
    if usd["label"] == "UNKNOWN":
        return "USD DIRECTION: not enough USD pairs to measure today."
    moves = ", ".join(f"{s} {v:+.2f}%" for s, v in usd["moves"].items())
    lines = [f"USD DIRECTION (computed, today, positive = USD stronger): USD {usd['label']} "
             f"({usd['usd_change_pct']:+.3f}% average across {usd['pairs']} pairs, {usd['agreement']:.0%} agree)",
             f"  per pair USD move: {moves}"]
    for side in ("BUY", "SELL"):
        exposure = usd_exposure(symbol, side)
        if exposure:
            with_or_against = ("WITH" if (exposure == "LONG_USD") == (usd["label"] == "STRENGTHENING") else "AGAINST") \
                if usd["label"] != "MIXED" else "NEUTRAL TO"
            lines.append(f"  {side} {symbol} = {exposure.replace('_', ' ')}: {with_or_against} today's USD direction")
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
        + (f" | corr_h1 {q['corr_h1']:+.2f}" if q.get("corr_h1") is not None else "")
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
        f"{_usd_block(market, symbol)}\n\n"
        f"CROSS-ASSET SNAPSHOT (change since today's D1 open; corr_h1 = correlation of H1 returns with "
        f"{symbol} over the last 50 bars):\n{cross_asset}\n\n"
        f"{_news_block(symbol)}\n\n"
        f"LEARNED RISK RULES (checked and applied by the engine after you answer):\n"
        f"{_rules_block(rules, symbol, market)}\n"
        f"=== END DEEP MARKET CONTEXT ==="
    )


def build_system_prompt(market: Dict[str, Any], symbol: str, rules: List[Dict[str, Any]]) -> str:
    """Static mandate first (a stable prefix for DeepSeek's context cache), then the live context."""
    mandate = SYSTEM_PROMPT.format(
        threshold=calibration.effective_threshold(),
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

    # The model's own conviction; every penalty below is applied by the engine, never left to the model.
    base_confidence = max(0, min(100, _to_int(payload.get("confidence_score"), 0)))
    confidence = base_confidence
    logic = re.sub(r"\s+", " ", str(payload.get("logic", ""))).strip()[:1200] or "No rationale supplied."
    raw_signal = signal
    threshold = calibration.effective_threshold()

    # Learned rules: their numeric conditions are evaluated on the live market by code.
    triggered = rule_penalties(rules, market, signal) if signal in ("BUY", "SELL") else []
    if triggered:
        rule_points = sum(rule["points"] for rule in triggered)
        logger.info("[LEARNING] %s %s learned rules -%d: %s (confidence %d -> %d)", symbol, signal, rule_points,
                    ", ".join(str(rule["id"]) for rule in triggered), confidence, max(0, confidence - rule_points))
        confidence = max(0, confidence - rule_points)
        logic = f"[RULES -{rule_points}: " + "; ".join(f"{r['id']} {r['setup']}" for r in triggered) + f"] {logic}"

    # Engine guards: overextension and trading against the broad USD move, whatever the model said.
    guards = protective_guards(market, symbol, signal) if signal in ("BUY", "SELL") else []
    for guard in guards:
        guard.setdefault("kind", "guard")
    if guards:
        guard_points = sum(guard["points"] for guard in guards)
        logger.info("[AI] %s %s guards -%d: %s (confidence %d -> %d)", symbol, signal, guard_points,
                    ", ".join(guard["id"] for guard in guards), confidence, max(0, confidence - guard_points))
        confidence = max(0, confidence - guard_points)
        logic = (f"[GUARDS -{guard_points}: " + "; ".join(f"{g['id']} {g['setup']}" for g in guards) + f"] {logic}")

    # A trade the model wanted at a tradeable conviction but the penalties stopped: follow it as a shadow trade.
    blocked_by: List[str] = []
    if signal != "HOLD" and confidence < threshold:
        if base_confidence >= threshold:
            blocked_by = [str(item["id"]) for item in (*triggered, *guards)]
        logic = (f"[THRESHOLD] {signal} conviction {confidence} < {threshold}: "
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
        if spread > config.MAX_SPREAD_TO_STOP * levels["sl_distance"]:
            blocked_by = []  # costs would have stopped this trade anyway
            if signal != "HOLD":
                logic = (f"[COSTS] spread is {spread / levels['sl_distance']:.0%} of the {sl_multiple}x ATR stop "
                         f"(limit {config.MAX_SPREAD_TO_STOP:.0%}): forced HOLD. {logic}")
                signal = "HOLD"
    else:
        blocked_by = []

    decision.update({
        "signal": signal,
        "raw_signal": raw_signal,
        "confidence_score": confidence,
        "base_confidence": base_confidence,
        "threshold": threshold,
        **stops,
        "entry_reference": round(entry, digits),
        "atr_reference": round(atr, digits + 1) if atr else None,
        "atr_source": atr_source,
        "logic": logic,
        "applied_rules": triggered + guards,
        "blocked_by": blocked_by,
        "usd_direction": usd_direction(market),
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
        "blocked_by": [],
        "threshold": calibration.effective_threshold(),
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
        response = deepseek_chat(messages, temperature=config.AI_TEMPERATURE, json_mode=True)
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


# -----------------------------------------------------------------------------
# Scalping: the rules find the setup, the AI confirms or vetoes it
# -----------------------------------------------------------------------------
SCALP_PROMPT = """You are the scalping desk reviewer at a risk-managed fund.
A deterministic rules engine has found the {side} setup below on {symbol}: an M5 pullback scalp in the
direction of the D1 trend, with M15 momentum agreeing. It closes at its stop, its target, or after
{time_stop} minutes. Your job is to CONFIRM or VETO this one trade and rate your conviction.

VETO when any of these hold:
- the target runs straight into a nearby H1/D1 swing level (see the H1 10-bar range and D1 structure);
- the pullback looks like a reversal (M5 lows/highs breaking against the trend, H1 structure turning);
- volatility is disorderly (H1 atr_ratio > 1.8, or the last M5 ranges are huge versus M5 ATR);
- the spread is a large share of the stop, or high-impact news for these currencies is close;
- the computed USD direction or the cross-asset moves clearly fight the trade.
CONFIRM when the trend is clean and the pullback orderly. Do not invent reasons; if nothing on the list
applies, confirm. The engine applies learned-rule and guard penalties itself after you answer and
executes only if the final confidence reaches {threshold}; score honestly (50 = coin flip).

OUTPUT one raw JSON object only, no markdown:
{{"decision": "CONFIRM" | "VETO", "confidence_score": <integer 0-100>, "logic": "<= 50 words"}}"""


REVERSION_PROMPT = """You are the scalping desk reviewer at a risk-managed fund.
A deterministic rules engine has found the {side} setup below on {symbol}: an M5 MEAN-REVERSION scalp in a
ranging market (weak H1 trend): price closed outside its Bollinger band with an RSI extreme and is now back
inside, targeting the band's middle. It closes at its stop, its target, or after {time_stop} minutes. Your job
is to CONFIRM or VETO this one trade and rate your conviction.

VETO when any of these hold:
- a real trend or breakout is starting (H1 structure breaking, a news move, expanding ranges and volume);
- high-impact news for these currencies is close, or the move came from news;
- the spread is a large share of the stop, or the target (the band's middle) is too close to pay the costs;
- volatility is disorderly (H1 atr_ratio > 1.8, or the last M5 ranges are huge versus M5 ATR).
CONFIRM when the market is clearly ranging and the stretch looks like noise. Do not invent reasons; if nothing
on the list applies, confirm. The engine applies learned-rule and guard penalties itself after you answer and
executes only if the final confidence reaches {threshold}; score honestly (50 = coin flip).

OUTPUT one raw JSON object only, no markdown:
{{"decision": "CONFIRM" | "VETO", "confidence_score": <integer 0-100>, "logic": "<= 50 words"}}"""


INTRADAY_PROMPT = """You are the intraday desk reviewer at a risk-managed fund.
A deterministic rules engine has found the {side} setup below on {symbol}: a break and retest of a key
level (yesterday's high/low or the Asian-session range) on the M15 chart. It closes at its stop, its target,
or after {time_stop} minutes. Your job is to CONFIRM or VETO this one trade and rate your conviction.

VETO when any of these hold:
- the target runs straight into a nearby H1/D1 swing level or another key level;
- the break looks like a false breakout (a long wick through the level, closes straddling it, momentum fading);
- volatility is disorderly (H1 atr_ratio > 1.8) or high-impact news for these currencies is close;
- the spread is a large share of the stop;
- the computed USD direction or the cross-asset moves clearly fight the trade.
CONFIRM when the break was clean and the retest held. Do not invent reasons; if nothing on the list
applies, confirm. The engine applies learned-rule and guard penalties itself after you answer and
executes only if the final confidence reaches {threshold}; score honestly (50 = coin flip).

OUTPUT one raw JSON object only, no markdown:
{{"decision": "CONFIRM" | "VETO", "confidence_score": <integer 0-100>, "logic": "<= 50 words"}}"""


DIP_PROMPT = """You are the swing desk reviewer at a risk-managed fund.
A deterministic rules engine has found the {side} setup below on {symbol}: a sharp H1 pullback (RSI(2) extreme)
against an established D1 trend, bought (sold) for a snap-back. It exits on the first H1 close back across the
H1 EMA5, at its stop, or after {time_stop} minutes. Your job is to CONFIRM or VETO this one trade.

VETO when any of these hold:
- the pullback is news-driven or high-impact news for these currencies is close (a shock, not noise);
- the D1 trend is breaking (price closing through major D1 structure, the move is a reversal, not a dip);
- volatility is disorderly (H1 atr_ratio > 2, or the last H1 ranges are huge versus H1 ATR);
- the spread is a large share of the stop.
CONFIRM when it looks like an ordinary pullback in a healthy trend. Do not invent reasons; if nothing on the
list applies, confirm. Score honestly (50 = coin flip); the engine applies its own penalties and limits.

OUTPUT one raw JSON object only, no markdown:
{{"decision": "CONFIRM" | "VETO", "confidence_score": <integer 0-100>, "logic": "<= 50 words"}}"""


def _strategy(setup: Dict[str, Any]) -> str:
    return str(setup.get("strategy") or "SCALP").upper()


def _time_stop(setup: Dict[str, Any]) -> int:
    return {"INTRADAY": config.INTRADAY_TIME_STOP_MINUTES,
            "DIP": config.DIP_TIME_STOP_MINUTES}.get(_strategy(setup), config.SCALP_TIME_STOP_MINUTES)


def _dip_block(symbol: str, setup: Dict[str, Any], market: Dict[str, Any]) -> str:
    digits = int(market.get("digits", 5))
    point = float(market.get("point") or 10 ** -digits)
    spread = float(market.get("spread_price") or 0.0)
    return (
        f"DIP SETUP ({setup['side']} {symbol}): {setup['reason']}\n"
        f"  entry ~{_fmt(setup['entry_ref'], digits)} | stop {setup['sl_distance'] / point:.0f} points "
        f"({setup['sl_atr']}x H1 ATR) | exit: H1 close back across EMA5 {_fmt(setup['h1_ema5'], digits)} "
        f"(safety target {setup['risk_reward']}R) | time stop {_time_stop(setup)} min\n"
        f"  H1 ATR14 {setup['atr'] / point:.1f} points | spread {spread / point:.0f} points = "
        f"{spread / setup['sl_distance'] * 100:.0f}% of the stop | "
        f"D1 close {_fmt(setup['d1_close'], digits)} vs D1 EMA200 {_fmt(setup['d1_ema200'], digits)}"
    )


def _intraday_block(symbol: str, setup: Dict[str, Any], market: Dict[str, Any]) -> str:
    digits = int(market.get("digits", 5))
    point = float(market.get("point") or 10 ** -digits)
    spread = float(market.get("spread_price") or 0.0)
    return (
        f"INTRADAY SETUP ({setup['side']} {symbol}): {setup['reason']}\n"
        f"  level {setup['level_name']} {_fmt(setup['level'], digits)} | entry ~{_fmt(setup['entry_ref'], digits)} | "
        f"stop {setup['sl_distance'] / point:.0f} points ({setup['sl_atr']}x M15 ATR, beyond the retest) | "
        f"target {setup['tp_distance'] / point:.0f} points (R:R {setup['risk_reward']}) | time stop {_time_stop(setup)} min\n"
        f"  M15 ATR14 {setup['atr'] / point:.1f} points | spread {spread / point:.0f} points = "
        f"{spread / setup['sl_distance'] * 100:.0f}% of the stop"
    )


def _scalp_block(symbol: str, setup: Dict[str, Any], recent: Dict[str, Any], market: Dict[str, Any]) -> str:
    digits = int(market.get("digits", 5))
    point = float(market.get("point") or 10 ** -digits)
    atr = setup["atr"]
    spread = float(market.get("spread_price") or 0.0)
    return (
        f"SCALP SETUP ({setup['side']} {symbol}): {setup['reason']}\n"
        f"  entry ~{_fmt(setup['entry_ref'], digits)} | stop {setup['sl_distance'] / point:.0f} points "
        f"({setup['sl_atr']}x M5 ATR) | target {setup['tp_distance'] / point:.0f} points "
        f"(R:R {setup['risk_reward']}) | time stop {config.SCALP_TIME_STOP_MINUTES} min\n"
        f"  M5 ATR14 {atr / point:.1f} points | spread {spread / point:.0f} points = "
        f"{spread / setup['sl_distance'] * 100:.0f}% of the stop\n"
        f"  M5 EMA20 {_fmt(setup['m5_ema20'], digits)} | M5 EMA50 {_fmt(setup['m5_ema50'], digits)} | "
        f"M15 close {_fmt(setup['m15_close'], digits)} vs M15 EMA50 {_fmt(setup['m15_ema50'], digits)} | "
        f"D1 close {_fmt(setup['d1_close'], digits)} vs D1 EMA200 {_fmt(setup['d1_ema200'], digits)}\n"
        f"  last {len(recent['closes'])} M5 bars (oldest -> newest): closes [{_series(recent['closes'], digits)}] | "
        f"highs [{_series(recent['highs'], digits)}] | lows [{_series(recent['lows'], digits)}] | "
        f"RSI14 [{_series(recent['rsi14'], 1)}]"
    )


def build_scalp_prompt(market: Dict[str, Any], symbol: str, rules: List[Dict[str, Any]], setup: Dict[str, Any],
                       recent: Dict[str, Any]) -> str:
    digits = int(market.get("digits", 5))
    strategy = _strategy(setup)
    intraday = strategy == "INTRADAY"
    template = (INTRADAY_PROMPT if intraday else DIP_PROMPT if strategy == "DIP"
                else REVERSION_PROMPT if setup.get("kind") == "REVERSION" else SCALP_PROMPT)
    mandate = template.format(
        side=setup["side"], symbol=symbol, time_stop=_time_stop(setup), threshold=calibration.effective_threshold())
    correlated = market.get("correlated_prices") or {}
    cross = "\n".join(f"  {s}: day change {_fmt(q.get('day_change_pct'), 3)}%"
                      + (f" | corr_h1 {q['corr_h1']:+.2f}" if q.get("corr_h1") is not None else "")
                      for s, q in correlated.items()) or "  unavailable"
    return (
        f"{mandate}\n\n=== CONTEXT: {symbol} ===\n"
        f"{_intraday_block(symbol, setup, market) if intraday else _dip_block(symbol, setup, market) if strategy == 'DIP' else _scalp_block(symbol, setup, recent, market)}\n\n"
        f"{_timeframe_block('DAILY (D1)', market.get('daily_data') or {}, digits)}\n\n"
        f"{_timeframe_block('HOURLY (H1)', market.get('h1_data') or {}, digits)}\n\n"
        f"{_usd_block(market, symbol)}\n\nCROSS-ASSET (change since today's D1 open):\n{cross}\n\n"
        f"{_news_block(symbol)}\n\n"
        f"LEARNED RISK RULES (checked and applied by the engine after you answer):\n"
        f"{_rules_block(rules, symbol, market)}\n=== END CONTEXT ==="
    )


def _scalp_decision(payload: Dict[str, Any], market: Dict[str, Any], symbol: str, rules: List[Dict[str, Any]],
                    setup: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the AI's verdict on a rules-found setup into an executable (or HOLD) decision."""
    digits = int(market.get("digits", 5))
    side = setup["side"]
    verdict = str(payload.get("decision", "VETO")).strip().upper()
    base_confidence = max(0, min(100, _to_int(payload.get("confidence_score"), 0)))
    confidence = base_confidence
    threshold = calibration.effective_threshold()
    logic = re.sub(r"\s+", " ", str(payload.get("logic", ""))).strip()[:600] or "No rationale supplied."
    triggered: List[Dict[str, Any]] = []
    guards: List[Dict[str, Any]] = []
    blocked_by: List[str] = []
    signal = side

    if verdict != "CONFIRM":
        signal = "HOLD"
        blocked_by = ["AI-VETO"]  # followed as a shadow trade: shows whether the AI's vetoes help
        logic = f"[AI VETO] {logic}"
    else:
        triggered = rule_penalties(rules, market, side)
        guards = protective_guards(market, symbol, side)
        for guard in guards:
            guard.setdefault("kind", "guard")
        penalties = triggered + guards
        if penalties:
            points = sum(item["points"] for item in penalties)
            confidence = max(0, confidence - points)
            logic = f"[PENALTIES -{points}: " + "; ".join(f"{p['id']} {p['setup']}" for p in penalties) + f"] {logic}"
        if confidence < threshold:
            if base_confidence >= threshold:
                blocked_by = [str(item["id"]) for item in penalties]
            logic = f"[THRESHOLD] {side} conviction {confidence} < {threshold}: forced HOLD. {logic}"
            signal = "HOLD"

    spread = float(market.get("spread_price") or 0.0)
    cost_block = spread > config.MAX_SPREAD_TO_STOP * setup["sl_distance"]
    if cost_block:  # a hard limit: neither the AI nor the learning agent can take this trade
        blocked_by = []
        if signal != "HOLD":
            logic = (f"[COSTS] spread is {spread / setup['sl_distance']:.0%} of the stop "
                     f"(limit {config.MAX_SPREAD_TO_STOP:.0%}): forced HOLD. {logic}")
            signal = "HOLD"

    entry = float(market["ask"] if side == "BUY" else market["bid"])
    direction = 1.0 if side == "BUY" else -1.0
    tick_size = float(market.get("tick_size") or market.get("point") or 10 ** -digits)
    decision.update({
        "signal": signal,
        "raw_signal": side,
        "confidence_score": confidence,
        "base_confidence": base_confidence,
        "threshold": threshold,
        "stop_loss": round_to_tick(entry - direction * setup["sl_distance"], tick_size, digits),
        "take_profit": round_to_tick(entry + direction * setup["tp_distance"], tick_size, digits),
        "sl_distance": round(setup["sl_distance"], digits + 3),
        "tp_distance": round(setup["tp_distance"], digits + 3),
        "sl_atr_multiple": setup["sl_atr"],
        "tp_atr_multiple": round(setup["sl_atr"] * setup["risk_reward"], 2),
        "risk_reward": setup["risk_reward"],
        "entry_reference": round(entry, digits),
        "atr_reference": round(setup["atr"], digits + 1),
        "atr_source": {"INTRADAY": "M15 ATR14", "DIP": "H1 ATR14"}.get(_strategy(setup), "M5 ATR14"),
        "exit_rule": setup.get("exit_rule"),
        "stops_source": f"{_strategy(setup)}_RULES",
        "stop_notes": [],
        "logic": f"{setup['reason']}. {logic}",
        "applied_rules": triggered + guards,
        "blocked_by": blocked_by,
        "cost_block": cost_block,
        "usd_direction": usd_direction(market),
        "strategy": _strategy(setup),
        "time_stop_minutes": _time_stop(setup),
        "setup": {k: setup[k] for k in ("side", "trend", "kind", "bar_time", "sl_atr", "m5_rsi", "m5_rsi_extreme",
                                        "reason", "level_name", "level") if k in setup},
    })
    return decision


def get_scalp_decision(market_data: Dict[str, Any], symbol: str, setup: Dict[str, Any],
                       recent: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the AI to confirm or veto a rules-found scalp; any failure degrades safely to HOLD."""
    rules = load_learned_rules(symbol)
    decision: Dict[str, Any] = {
        "symbol": symbol, "timestamp": utc_now_iso(), "signal": "HOLD", "raw_signal": setup["side"],
        "confidence_score": 0, "base_confidence": 0, "stop_loss": None, "take_profit": None,
        "sl_atr_multiple": None, "tp_atr_multiple": None, "sl_distance": None, "tp_distance": None,
        "risk_reward": None, "entry_reference": market_data.get("mid"), "stops_source": None, "stop_notes": [],
        "atr_reference": None, "atr_source": None, "logic": "", "applied_rules": [], "blocked_by": [],
        "threshold": calibration.effective_threshold(), "active_rules_count": len(rules), "latency_ms": None,
        "error": None, "strategy": _strategy(setup), "time_stop_minutes": _time_stop(setup),
    }
    kind = {"INTRADAY": "intraday trade", "DIP": "H1 dip trade"}.get(_strategy(setup), "scalp")
    if not config.DEEPSEEK_API_KEY:
        return _without_ai(market_data, symbol, rules, setup, decision, "missing_api_key",
                           "AI key not configured; the AI's answer is missing, so HOLD unless the learning agent takes it")
    messages = [
        {"role": "system", "content": build_scalp_prompt(market_data, symbol, rules, setup, recent)},
        {"role": "user", "content": f"Confirm or veto the {setup['side']} {symbol} {kind}. Return the JSON now."},
    ]
    started = time.perf_counter()
    try:
        response = deepseek_chat(messages, temperature=config.AI_TEMPERATURE, max_tokens=1024, json_mode=True)
        payload = parse_json_payload(response["content"])
        if not isinstance(payload, dict):
            raise ValueError("model returned JSON that is not an object")
    except (DeepSeekError, ValueError) as exc:
        logger.error("[AI] %s %s review failed: %s", symbol, kind, exc)
        decision["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return _without_ai(market_data, symbol, rules, setup, decision, str(exc),
                           f"AI review unavailable ({exc}); HOLD unless the learning agent takes it")
    decision["latency_ms"] = int((time.perf_counter() - started) * 1000)
    return _scalp_decision(payload, market_data, symbol, rules, setup, decision)


def _without_ai(market: Dict[str, Any], symbol: str, rules: List[Dict[str, Any]], setup: Dict[str, Any],
                decision: Dict[str, Any], error: str, note: str) -> Dict[str, Any]:
    """HOLD with the setup's exact levels, so the learning agent can still decide and the idea is shadow-followed."""
    decision = _scalp_decision({"decision": "VETO", "confidence_score": 0, "logic": note}, market, symbol, rules,
                               setup, decision)
    decision.update(error=error, logic=f"{setup['reason']}. {note}.", base_confidence=0, confidence_score=0,
                    blocked_by=[] if decision.get("cost_block") else ["AI-ERROR"])
    return decision
