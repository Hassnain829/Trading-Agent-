"""
Intraday strategy: break and retest of key levels (M15), held for hours.

    levels     yesterday's high / low (the last closed D1 bar) and today's Asian range (the trading
               day's start until the London open); highs are traded upward, lows downward
    break      an M15 candle closes beyond the level, after the London open, having been inside it
    retest     within INTRADAY_RETEST_BARS candles price comes back to the level (a touch within
               0.1 M15 ATR); a close back through by more than 0.25 ATR means the break failed
    entry      the first candle after the touch that closes beyond the level in the break direction
    stop       beyond the retest's extreme (+0.1 ATR): the structural stop is never moved; a trade is
               rejected instead when that stop is too tight for the costs or wider than 3 M15 ATR
    exit       INTRADAY_REWARD_RISK x the stop, or after INTRADAY_TIME_STOP_MINUTES

evaluate(relaxed=True) runs the same day walk with looser near-miss limits; the learning agent follows
those setups as virtual exploration trades (never real orders).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import config
import scalper

_LONDON = scalper._LONDON
MIN_ASIAN_BARS = 12        # at least 3 hours of Asian trading before its range counts
ASIA_END_LONDON = 7        # the Asian range: the trading day's bars before 07:00 London
TOUCH_ATR, FAIL_ATR, STOP_PAD_ATR = 0.10, 0.25, 0.10
MIN_STOP_ATR, MAX_STOP_ATR, MIN_STOP_SPREADS = 0.3, 3.0, 4.0
# Near-miss limits for exploration: a looser touch, a deeper failure, twice the retest time, wider stops.
STRICT = {"touch": TOUCH_ATR, "fail": FAIL_ATR, "retest_factor": 1.0, "min_stop": MIN_STOP_ATR,
          "max_stop": MAX_STOP_ATR, "min_spreads": MIN_STOP_SPREADS}
RELAXED = {"touch": 0.25, "fail": 0.5, "retest_factor": 2.0, "min_stop": 0.2, "max_stop": 4.0, "min_spreads": 3.0}
LEVELS = (("PDH", "BUY"), ("PDL", "SELL"), ("ASIA_HIGH", "BUY"), ("ASIA_LOW", "SELL"))
LEVEL_NAMES = {"PDH": "yesterday's high", "PDL": "yesterday's low", "ASIA_HIGH": "the Asian high", "ASIA_LOW": "the Asian low"}


# -----------------------------------------------------------------------------
# Preparation
# -----------------------------------------------------------------------------
def prepare(p: scalper.Prepared, new_york_plus_7: bool = False) -> scalper.Prepared:
    """Add what the intraday rules need to a scalper.Prepared: M15 ATR, UTC times, server days."""
    import data_engine
    q = p.a["M15"]
    if "atr14" not in q:
        f = p.frames["M15"]
        high, low, close = f["high"].astype(float), f["low"].astype(float), f["close"].astype(float)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        q["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
        utc = data_engine.server_epochs_to_utc(q["close_time"].astype("int64"), new_york_plus_7)
        london = pd.DatetimeIndex(utc).tz_convert(_LONDON)
        q["_utc"] = utc
        q["_london_hour"] = london.hour.to_numpy()
        q["_weekday"] = london.weekday.to_numpy()
        q["_in_session"] = scalper.session_mask(pd.DatetimeIndex(utc))
        # the Asian range ends at the London open, whatever hours the bot trades
        new_york = pd.DatetimeIndex(utc).tz_convert(scalper._NEW_YORK)
        q["_asia_over"] = np.asarray((london.hour >= ASIA_END_LONDON) & (new_york.hour < 17), dtype=bool)
        q["_server_day"] = ((q["close_time"].astype("int64") - 1) // 86400).astype("int64")
    return p


# -----------------------------------------------------------------------------
# The day walk
# -----------------------------------------------------------------------------
def _levels(p: scalper.Prepared, day: np.ndarray) -> Dict[str, float]:
    """Yesterday's high/low and the Asian range for the bars of one server day (indices ``day``)."""
    q = p.a["M15"]
    levels: Dict[str, float] = {}
    d = p.last_closed("D1", q["close_time"][day[0]] - 1)
    if d is not None:
        levels["PDH"], levels["PDL"] = float(p.a["D1"]["high"][d]), float(p.a["D1"]["low"][d])
    # The Asian range: every bar of the trading day before the London open.
    after = [pos for pos, j in enumerate(day) if q["_asia_over"][j]]
    asian = day[:after[0]] if after else []
    if len(asian) >= MIN_ASIAN_BARS:
        levels["ASIA_HIGH"] = float(np.max(q["high"][asian]))
        levels["ASIA_LOW"] = float(np.min(q["low"][asian]))
    return levels


def _walk_day(p: scalper.Prepared, day: np.ndarray, spreads: Optional[np.ndarray] = None,
              relaxed: bool = False) -> Dict[str, Any]:
    """
    Walk one day's M15 bars (up to the last index in ``day``) and return every level's state and every
    entry signal: {"levels": {name: {...}}, "entries": [ {k, side, level, sl_distance, ...} ]}.
    ``spreads`` are M15 spreads in price (None = 0). ``relaxed`` = the near-miss limits.
    """
    lim = RELAXED if relaxed else STRICT
    q = p.a["M15"]
    levels = _levels(p, day)
    asia_end = next((pos for pos, j in enumerate(day) if q["_asia_over"][j]), len(day))
    states: Dict[str, Dict[str, Any]] = {}
    entries: List[Dict[str, Any]] = []
    for name, side in LEVELS:
        if name not in levels:
            continue
        level, sign = levels[name], 1.0 if side == "BUY" else -1.0
        st: Dict[str, Any] = {"level": level, "side": side, "state": "WAIT_BREAK"}
        broke = touched = None
        low_since = high_since = None
        for pos, j in enumerate(day):
            if not q["_in_session"][j] or pos == 0:
                continue
            if name.startswith("ASIA") and pos < asia_end:  # the range is complete only at the London open
                continue
            close, prev = q["close"][j], q["close"][day[pos - 1]]
            atr = q["atr14"][j]
            if not np.isfinite(atr) or atr <= 0:
                continue
            beyond = sign * (close - level) > 0
            if broke is None:
                if beyond and sign * (prev - level) <= 0:
                    broke, st = j, {**st, "state": "BROKEN", "break_at": int(j)}
                    low_since, high_since = q["low"][j], q["high"][j]
                continue
            if st["state"] in ("FAILED", "ENTERED", "EXPIRED", "REJECTED"):
                continue
            low_since, high_since = min(low_since, q["low"][j]), max(high_since, q["high"][j])
            if sign * (level - close) > lim["fail"] * atr:
                st = {**st, "state": "FAILED"}
                continue
            if j - broke > config.INTRADAY_RETEST_BARS * lim["retest_factor"]:
                st = {**st, "state": "EXPIRED"}
                continue
            touch = ((q["low"][j] <= level + lim["touch"] * atr) if side == "BUY"
                     else (q["high"][j] >= level - lim["touch"] * atr))
            if touch and touched is None:
                touched, st = j, {**st, "state": "RETEST"}
            if touched is not None and beyond and sign * (close - q["open"][j]) > 0:
                spread = float(spreads[j]) if spreads is not None else 0.0
                extreme = low_since if side == "BUY" else high_since
                stop = extreme - STOP_PAD_ATR * atr if side == "BUY" else extreme + STOP_PAD_ATR * atr
                sl = (close + spread - stop) if side == "BUY" else (stop + spread - close)
                if sl < max(lim["min_stop"] * atr, lim["min_spreads"] * spread) or sl > lim["max_stop"] * atr:
                    st = {**st, "state": "REJECTED", "why": f"structural stop {sl / atr:.1f} ATR is outside "
                          f"{lim['min_stop']}-{lim['max_stop']} ATR (or under {lim['min_spreads']:g} spreads)"}
                    continue
                entries.append({"k": int(j), "side": side, "level_name": name, "level": level,
                                "sl_distance": float(sl), "tp_distance": float(sl * config.INTRADAY_REWARD_RISK),
                                "atr": float(atr), "sl_atr": round(float(sl / atr), 2),
                                "bars_since_break": int(j - broke)})
                st = {**st, "state": "ENTERED", "entry_at": int(j)}
        states[name] = st
    return {"levels": states, "entries": entries}


def _today(p: scalper.Prepared, k: int) -> np.ndarray:
    q = p.a["M15"]
    day = q["_server_day"][k]
    first = k
    while first > 0 and q["_server_day"][first - 1] == day:
        first -= 1
    return np.arange(first, k + 1)


# -----------------------------------------------------------------------------
# Live evaluation (latest closed M15 bar)
# -----------------------------------------------------------------------------
STAGE_ORDER = ("ENTERED", "RETEST", "BROKEN", "WAIT_BREAK", "FAILED", "EXPIRED", "REJECTED")


def evaluate(p: scalper.Prepared, k: Optional[int] = None, spread_price: float = 0.0,
             relaxed: bool = False) -> Dict[str, Any]:
    """
    The intraday signal on M15 bar ``k`` (default: the latest closed one): {"stage", "setup" | None,
    "reason", "levels"}. A setup is returned only when bar ``k`` itself completes a retest entry.
    ``relaxed`` = the near-miss limits (exploration).
    """
    q = p.a["M15"]
    k = len(q["close"]) - 1 if k is None else k
    if "atr14" not in q:
        prepare(p)
    if not q["_in_session"][k]:
        return {"stage": "OFF_SESSION", "setup": None, "reason": "outside the intraday entry window", "levels": {}}
    day = _today(p, k)
    spreads = np.full(len(q["close"]), spread_price)
    walked = _walk_day(p, day, spreads, relaxed)
    levels = walked["levels"]
    if not levels:
        return {"stage": "WARMUP", "setup": None, "reason": "levels not known yet (need yesterday's bar and the Asian range)",
                "levels": {}}
    hit = [e for e in walked["entries"] if e["k"] == k]
    if hit:
        e = hit[0]
        setup = {**e, "strategy": "INTRADAY", "trend": "UP" if e["side"] == "BUY" else "DOWN",
                 "risk_reward": config.INTRADAY_REWARD_RISK, "entry_ref": float(q["close"][k]),
                 "reason": f"{e['side']} {LEVEL_NAMES[e['level_name']]} break and retest at {e['level']:.5g}: "
                           f"broke, came back to the level and closed beyond it again"}
        return {"stage": "SETUP", "setup": setup, "reason": setup["reason"], "levels": levels}
    best = min(levels.items(), key=lambda kv: STAGE_ORDER.index(kv[1]["state"]) if kv[1]["state"] in STAGE_ORDER else 9)
    name, st = best
    reasons = {
        "ENTERED": f"{LEVEL_NAMES[name]} already gave its signal today (one per level per day)",
        "RETEST": f"{LEVEL_NAMES[name]} ({st['level']:.5g}) broke and is being retested: waiting for a candle closing beyond it",
        "BROKEN": f"{LEVEL_NAMES[name]} ({st['level']:.5g}) broke: waiting for price to come back and retest it",
        "WAIT_BREAK": "waiting for a break of " + ", ".join(f"{LEVEL_NAMES[n]} {s['level']:.5g}" for n, s in levels.items()),
        "FAILED": f"the break of {LEVEL_NAMES[name]} failed (price fell back through it)",
        "EXPIRED": f"{LEVEL_NAMES[name]} broke but was not retested in time",
        "REJECTED": f"{LEVEL_NAMES[name]} retest found, but {st.get('why', 'the stop did not fit')}",
    }
    return {"stage": f"I_{st['state']}", "setup": None, "reason": reasons.get(st["state"], st["state"]), "levels": levels}
