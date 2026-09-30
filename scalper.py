"""
M5 pullback scalper: the entry rules of the scalping strategy.

    D1   trend      close above a rising EMA200 -> longs only; below a falling one -> shorts only
    M15  momentum   close on the trend side of a sloping EMA50
    M5   trigger    RSI14 pulled back below SCALP_RSI_PULLBACK (above 100 - it for shorts) within the last
                    SCALP_PULLBACK_BARS bars and has now turned back, on a candle closing in the trend
                    direction, with price still on the trend side of the M5 EMA50
    stop            beyond the last SCALP_SWING_BARS-bar swing (+0.1 ATR), clamped to 1-2 x M5 ATR14
    target          stop x SCALP_REWARD_RISK; the engine closes the trade after SCALP_TIME_STOP_MINUTES
    session         24/7 (TRADE_ALL_HOURS), or a London-open to New York window, DST-aware

Indicators are computed once per frame (vectorised). evaluate(relaxed=True) gives the near-miss
setups the learning agent follows as virtual exploration trades.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta

import config

TF_SECONDS = {"M5": 300, "M15": 900, "H1": 3600, "D1": 86400}
TIMEFRAMES = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}
LIVE_BARS = {"M5": 300, "M15": 300, "H1": 300, "D1": 260}
_LONDON = ZoneInfo("Europe/London")
_NEW_YORK = ZoneInfo("America/New_York")


# -----------------------------------------------------------------------------
# Session
# -----------------------------------------------------------------------------
def in_session(now: Optional[datetime] = None) -> bool:
    """
    May new trades open now? 24/7 (TRADE_ALL_HOURS): always; a closed market simply has no new candles.
    Otherwise weekdays from the London open (London time) until SCALP_SESSION_END_NEW_YORK (NY time).
    """
    if config.TRADE_ALL_HOURS:
        return True
    now = now or datetime.now(timezone.utc)
    london, new_york = now.astimezone(_LONDON), now.astimezone(_NEW_YORK)
    if london.weekday() >= 5 or new_york.weekday() >= 5:
        return False
    return london.hour >= config.SCALP_SESSION_START_LONDON and new_york.hour < config.SCALP_SESSION_END_NEW_YORK


def session_mask(utc_times: pd.DatetimeIndex) -> np.ndarray:
    """in_session for many UTC timestamps at once."""
    if config.TRADE_ALL_HOURS:
        return np.ones(len(utc_times), dtype=bool)
    london, new_york = utc_times.tz_convert(_LONDON), utc_times.tz_convert(_NEW_YORK)
    mask = ((london.weekday < 5) & (new_york.weekday < 5) & (london.hour >= config.SCALP_SESSION_START_LONDON)
            & (new_york.hour < config.SCALP_SESSION_END_NEW_YORK))
    return np.asarray(mask, dtype=bool)


def next_session_open(now: Optional[datetime] = None) -> Optional[datetime]:
    """When the scalping window next opens (UTC); ``now`` itself when it is open. None if never within a week."""
    now = (now or datetime.now(timezone.utc)).replace(second=0, microsecond=0)
    if in_session(now):
        return now
    probe = now + timedelta(minutes=5 - now.minute % 5)
    for _ in range(7 * 24 * 12):  # 5-minute steps; the window opens on the hour
        if in_session(probe):
            return probe
        probe += timedelta(minutes=5)
    return None


def session_note(now: Optional[datetime] = None) -> str:
    """The bot's entry window (not the forex market's hours), with this computer's local time for clarity."""
    if config.TRADE_ALL_HOURS:
        return "24/7: new trades in every session (Asia, London, New York) whenever the market is open"
    window = (f"new scalps from London {config.SCALP_SESSION_START_LONDON:02d}:00 to New York "
              f"{config.SCALP_SESSION_END_NEW_YORK:02d}:00")
    if in_session(now):
        return f"{window} (open now)"
    opens = next_session_open(now)
    if opens is None:
        return f"{window} (entry window closed)"
    minutes = int((opens - (now or datetime.now(timezone.utc))).total_seconds() // 60)
    return (f"{window} (entry window closed; the forex market may still be open. Next entries "
            f"{opens:%a %H:%M} UTC = {opens.astimezone():%H:%M} your time, in {minutes // 60}h {minutes % 60:02d}m)")


# -----------------------------------------------------------------------------
# Indicators
# -----------------------------------------------------------------------------
def _col(result: Optional[pd.Series], index: pd.Index) -> pd.Series:
    return result if result is not None else pd.Series(np.nan, index=index)


class Prepared:
    """Indicator columns per timeframe plus numpy views for fast evaluation."""

    def __init__(self, frames: Dict[str, pd.DataFrame]):
        self.frames: Dict[str, pd.DataFrame] = {}
        self.a: Dict[str, Dict[str, np.ndarray]] = {}
        for tf, raw in frames.items():
            f = raw.reset_index(drop=True).copy()
            close, high, low = f["close"].astype(float), f["high"].astype(float), f["low"].astype(float)
            idx = f.index
            f["close_time"] = f["time"].astype("int64") + TF_SECONDS[tf]
            if tf in ("D1", "H1"):
                f["ema200"] = _col(ta.ema(close, length=200, talib=False), idx)
                f["ema200_slope"] = f["ema200"] - f["ema200"].shift(5)
                f["atr14"] = _col(ta.atr(high, low, close, length=14, talib=False), idx)
                f["rsi14"] = _col(ta.rsi(close, length=14, talib=False), idx)
                f["ema_distance_atr"] = (close - f["ema200"]) / f["atr14"]
                f["atr_ratio"] = f["atr14"] / f["atr14"].rolling(50).mean()
                if tf == "D1":  # medium-term trend and trend strength (optional filters)
                    f["ema20"] = _col(ta.ema(close, length=20, talib=False), idx)
                    f["ema50"] = _col(ta.ema(close, length=50, talib=False), idx)
                    adx = ta.adx(high, low, close, length=14, talib=False)
                    f["adx14"] = adx["ADX_14"] if adx is not None and "ADX_14" in adx else np.nan
            elif tf == "M15":
                f["ema50"] = _col(ta.ema(close, length=50, talib=False), idx)
                f["ema50_slope"] = f["ema50"] - f["ema50"].shift(3)
            elif tf == "M5":
                f["ema20"] = _col(ta.ema(close, length=20, talib=False), idx)
                f["ema50"] = _col(ta.ema(close, length=50, talib=False), idx)
                f["rsi14"] = _col(ta.rsi(close, length=14, talib=False), idx)
                f["atr14"] = _col(ta.atr(high, low, close, length=14, talib=False), idx)
                f["atr_ratio"] = f["atr14"] / f["atr14"].rolling(50).mean()
                volume = f["tick_volume"].astype(float) if "tick_volume" in f else pd.Series(np.nan, index=idx)
                f["rel_volume"] = volume / volume.rolling(20).mean().replace(0, np.nan)
                f["swing_low"] = low.rolling(config.SCALP_SWING_BARS).min()
                f["swing_high"] = high.rolling(config.SCALP_SWING_BARS).max()
            self.frames[tf] = f
            self.a[tf] = {column: f[column].to_numpy(dtype=float) if column != "time" else f[column].to_numpy()
                          for column in f.columns}

    def last_closed(self, tf: str, t: float) -> Optional[int]:
        """Index of the last bar of ``tf`` that had closed at server time ``t``."""
        index = int(np.searchsorted(self.a[tf]["close_time"], t, side="right")) - 1
        return index if index >= 0 else None


def prepare(frames: Dict[str, pd.DataFrame]) -> Prepared:
    return Prepared(frames)


def fetch_live_frames(symbol: str) -> Dict[str, pd.DataFrame]:
    """Closed bars for every timeframe the rules use (blocking)."""
    from data_engine import fetch_bars
    return {tf: fetch_bars(symbol, TIMEFRAMES[tf], LIVE_BARS[tf]) for tf in TF_SECONDS}


# -----------------------------------------------------------------------------
# Setup detection
# -----------------------------------------------------------------------------
def _nan(value: float) -> bool:
    return value is None or not np.isfinite(value)


# Near-miss rules for the learning agent's virtual exploration trades (never real orders): the RSI dip
# may stop this many points short, and M15 momentum, the medium trend, the room and the stretch checks
# are not required.
EXPLORE_RSI_SLACK = 7.0


def evaluate(p: Prepared, i: Optional[int] = None, spread_price: float = 0.0, relaxed: bool = False) -> Dict[str, Any]:
    """The setup on M5 bar ``i`` plus the stage it stopped at and the M5 RSI (see _evaluate)."""
    result = _evaluate(p, i, spread_price, relaxed)
    m5_rsi = p.a["M5"]["rsi14"]
    index = len(m5_rsi) - 1 if i is None else i
    result["rsi"] = None if index < 0 or _nan(m5_rsi[index]) else round(float(m5_rsi[index]), 1)
    return result


def _evaluate(p: Prepared, i: Optional[int] = None, spread_price: float = 0.0, relaxed: bool = False) -> Dict[str, Any]:
    """
    The setup on M5 bar ``i`` (default: the latest closed bar). Returns {"setup": dict | None,
    "reason": why not / what was found, "trend": "UP" | "DOWN" | None}. ``relaxed`` = the near-miss rules.
    """
    m5 = p.a["M5"]
    i = len(m5["close"]) - 1 if i is None else i
    lookback = config.SCALP_PULLBACK_BARS
    if i < 60:
        return {"stage": "WARMUP", "setup": None, "reason": "not enough M5 history", "trend": None}
    t = m5["close_time"][i]

    d = p.last_closed("D1", t)
    if d is None or _nan(p.a["D1"]["ema200"][d]) or _nan(p.a["D1"]["ema200_slope"][d]):
        return {"stage": "WARMUP", "setup": None, "reason": "not enough D1 history for EMA200", "trend": None}
    d_close, d_ema, d_slope = p.a["D1"]["close"][d], p.a["D1"]["ema200"][d], p.a["D1"]["ema200_slope"][d]
    trend = "UP" if d_close > d_ema and d_slope > 0 else "DOWN" if d_close < d_ema and d_slope < 0 else None
    if trend is None:
        return {"stage": "NO_TREND", "setup": None, "reason": "D1 trend unclear (price and EMA200 slope disagree): no scalps in either "
                "direction until it clears", "trend": None}
    if config.SCALP_MEDIUM_TREND and not relaxed:
        e20, e50 = p.a["D1"]["ema20"][d], p.a["D1"]["ema50"][d]
        agrees = (d_close > e50 and e20 > e50) if trend == "UP" else (d_close < e50 and e20 < e50)
        if not agrees:
            return {"stage": "MEDIUM_AGAINST", "setup": None, "reason": f"D1 {trend.lower()}trend on EMA200, but the medium-term "
                    f"trend (EMA20/50) disagrees: no scalps until they line up", "trend": trend}
    if config.SCALP_MIN_ADX > 0 and not relaxed:
        adx = p.a["D1"]["adx14"][d]
        if _nan(adx) or adx < config.SCALP_MIN_ADX:
            return {"stage": "WEAK_TREND", "setup": None, "reason": f"D1 {trend.lower()}trend too weak (ADX "
                    f"{0 if _nan(adx) else adx:.0f} < {config.SCALP_MIN_ADX:g})", "trend": trend}

    q = p.last_closed("M15", t)
    if q is None or _nan(p.a["M15"]["ema50_slope"][q]):
        return {"stage": "WARMUP", "setup": None, "reason": "not enough M15 history", "trend": trend}
    q_close, q_ema, q_slope = p.a["M15"]["close"][q], p.a["M15"]["ema50"][q], p.a["M15"]["ema50_slope"][q]
    if not relaxed and ((trend == "UP" and not (q_close > q_ema and q_slope > 0))
                        or (trend == "DOWN" and not (q_close < q_ema and q_slope < 0))):
        return {"stage": "M15_AGAINST", "setup": None, "reason": f"D1 {trend.lower()}trend, but M15 momentum is against it: waiting for "
                f"M15 to turn back {'up' if trend == 'UP' else 'down'}", "trend": trend}

    close, open_, rsi, atr = m5["close"][i], m5["open"][i], m5["rsi14"][i], m5["atr14"][i]
    ema50, prev_rsi = m5["ema50"][i], m5["rsi14"][i - 1]
    window = m5["rsi14"][i - lookback:i]
    if _nan(rsi) or _nan(atr) or _nan(ema50) or atr <= 0 or not np.isfinite(window).all():
        return {"stage": "WARMUP", "setup": None, "reason": "M5 indicators warming up", "trend": trend}
    level = config.SCALP_RSI_PULLBACK
    dip = min(50.0, level + EXPLORE_RSI_SLACK) if relaxed else level  # how far RSI must have pulled back
    if trend == "UP":
        pulled_back = window.min() < dip
        turned = rsi >= level and rsi > prev_rsi and close > open_ and close > ema50
    else:
        pulled_back = window.max() > 100 - dip
        turned = rsi <= 100 - level and rsi < prev_rsi and close < open_ and close < ema50
    if pulled_back and config.SCALP_PULLBACK_TO_EMA and not relaxed:  # the dip must reach value (the M5 EMA20), not just cool RSI
        lows, highs, ema20s = m5["low"][i - lookback:i + 1], m5["high"][i - lookback:i + 1], m5["ema20"][i - lookback:i + 1]
        pulled_back = bool(np.nanmin(lows - ema20s) <= 0) if trend == "UP" else bool(np.nanmax(highs - ema20s) >= 0)
    if turned and config.SCALP_TRIGGER == "BREAK" and not relaxed:  # price action confirms: close beyond the previous candle
        turned = close > m5["high"][i - 1] if trend == "UP" else close < m5["low"][i - 1]
    if not pulled_back:
        wanted = (f"a dip: M5 RSI below {level:.0f}" if trend == "UP" else f"a bounce: M5 RSI above {100 - level:.0f}")
        if config.SCALP_PULLBACK_TO_EMA:
            wanted += " that reaches the M5 EMA20"
        return {"stage": "NO_PULLBACK", "setup": None, "reason": f"D1 {trend.lower()}trend, no M5 pullback yet: waiting for {wanted} "
                f"(now {rsi:.0f})", "trend": trend}
    if not turned:
        extreme = window.min() if trend == "UP" else window.max()
        wanted = (f"RSI back above {level:.0f} on a bullish M5 candle above EMA50" if trend == "UP"
                  else f"RSI back below {100 - level:.0f} on a bearish M5 candle below EMA50")
        if config.SCALP_TRIGGER == "BREAK":
            wanted += (" closing above the previous candle's high" if trend == "UP"
                       else " closing below the previous candle's low")
        return {"stage": "NO_TURN", "setup": None, "reason": f"D1 {trend.lower()}trend, M5 pulled back (RSI {extreme:.0f}): waiting "
                f"for the turn, {wanted} (now {rsi:.0f})", "trend": trend}

    side = "BUY" if trend == "UP" else "SELL"
    if side == "BUY":
        entry = close + spread_price  # a long fills at the ask
        raw = entry - m5["swing_low"][i] + 0.1 * atr
    else:
        entry = close  # a short fills at the bid; its stop triggers on the ask
        raw = m5["swing_high"][i] + spread_price - entry + 0.1 * atr
    sl_distance = float(min(max(raw, config.SCALP_SL_ATR_MIN * atr), config.SCALP_SL_ATR_MAX * atr))
    tp_distance = sl_distance * config.SCALP_REWARD_RISK
    # Room to the recent swing high (buy) / low (sell): the level the trade has to get through.
    recent = slice(max(0, i - config.SCALP_ROOM_BARS), i)
    room = (float(np.max(m5["high"][recent])) - entry) if side == "BUY" else (entry - float(np.min(m5["low"][recent])))
    room_r = room / sl_distance if sl_distance > 0 else 0.0
    if config.SCALP_ROOM_MIN_R > 0 and room_r < config.SCALP_ROOM_MIN_R and not relaxed:
        return {"stage": "NO_ROOM", "setup": None, "reason": f"D1 {trend.lower()}trend pullback turned, but the recent swing "
                f"{'high' if side == 'BUY' else 'low'} is only {max(room_r, 0):.1f}R away (need {config.SCALP_ROOM_MIN_R:g}R)",
                "trend": trend}
    if config.SCALP_TARGET == "STRUCTURE":
        tp_distance = min(sl_distance * config.SCALP_REWARD_RISK, room - 0.1 * atr)
        if tp_distance < config.SCALP_MIN_TARGET_R * sl_distance:
            return {"stage": "NO_ROOM", "setup": None, "reason": f"D1 {trend.lower()}trend pullback turned, but the target "
                    f"before the recent swing is under {config.SCALP_MIN_TARGET_R:g}R", "trend": trend}
    extreme = window.min() if side == "BUY" else window.max()
    stretched = []
    if config.SCALP_STRICT_GUARD and not relaxed:
        from ai_brain import protective_guards  # the same H1/D1 overextension checks the AI decision applies
        stretched = [g for g in protective_guards(guard_context(p, t), "", side) if g["id"] != "G-USD"]
    setup = {
        "side": side,
        "trend": trend,
        "bar_time": int(m5["time"][i]),
        "entry_ref": float(entry),
        "sl_distance": sl_distance,
        "tp_distance": float(tp_distance),
        "risk_reward": round(float(tp_distance / sl_distance), 2),
        "room_r": round(float(room_r), 2),
        "atr": float(atr),
        "sl_atr": round(float(sl_distance / atr), 2),
        "m5_rsi": round(float(rsi), 1),
        "m5_rsi_extreme": round(float(extreme), 1),
        "m5_ema20": float(m5["ema20"][i]),
        "m5_ema50": float(ema50),
        "m15_close": float(q_close),
        "m15_ema50": float(q_ema),
        "d1_close": float(d_close),
        "d1_ema200": float(d_ema),
        "reason": (f"{side} pullback scalp: D1 {trend.lower()}trend, M15 aligned, M5 RSI "
                   f"{'dipped to' if side == 'BUY' else 'rose to'} {extreme:.0f} and turned at {rsi:.0f}"),
    }
    if stretched:
        ids = ", ".join(g["id"] for g in stretched)
        return {"stage": "STRETCHED", "setup": None, "blocked_setup": setup, "guards": stretched, "trend": trend,
                "reason": f"D1 {trend.lower()}trend pullback turned, but price is already stretched ({ids}): skipped "
                          f"(followed as a shadow trade so the agent learns whether that is right)"}
    return {"stage": "SETUP", "setup": setup, "reason": setup["reason"], "trend": trend}


def guard_context(p: Prepared, t: float) -> Dict[str, Any]:
    """H1/D1 values the overextension guard reads, as of server time ``t``."""
    context: Dict[str, Any] = {"h1_data": {}, "daily_data": {}}
    for tf, key in (("H1", "h1_data"), ("D1", "daily_data")):
        index = p.last_closed(tf, t)
        if index is not None:
            context[key] = {"ema_distance_atr": p.a[tf]["ema_distance_atr"][index], "rsi14": p.a[tf]["rsi14"][index]}
            context[key] = {k: (None if _nan(v) else float(v)) for k, v in context[key].items()}
    return context


FEATURE_VERSION = 1


def features(p: Prepared, i: Optional[int] = None, side: Optional[str] = None,
             spread_price: float = 0.0) -> Dict[str, Optional[float]]:
    """
    The numeric market state at M5 bar ``i``, for the decision journal and the ML model.

    Directional values are signed so that positive = in the trade's favour (for a SELL, a
    stretch below the EMA is positive), which lets one model learn both sides. Without a side
    they are raw (positive = above). Every value uses only bars closed at bar ``i``.
    """
    m5 = p.a["M5"]
    i = len(m5["close"]) - 1 if i is None else i
    t = m5["close_time"][i]
    sign = -1.0 if side == "SELL" else 1.0
    out: Dict[str, Optional[float]] = {"feature_version": FEATURE_VERSION}

    def put(name: str, value: Any, signed: bool = False) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")
        out[name] = None if not np.isfinite(number) else round(number * (sign if signed else 1.0), 5)

    d = p.last_closed("D1", t)
    if d is not None:
        a = p.a["D1"]
        close = a["close"][d]
        put("d1_ema200_dist_atr", a["ema_distance_atr"][d], True)
        put("d1_ema200_slope_atr", a["ema200_slope"][d] / a["atr14"][d] if a["atr14"][d] else np.nan, True)
        put("d1_ema20_vs_50_atr", (a["ema20"][d] - a["ema50"][d]) / a["atr14"][d] if a["atr14"][d] else np.nan, True)
        put("d1_close_vs_ema50_atr", (close - a["ema50"][d]) / a["atr14"][d] if a["atr14"][d] else np.nan, True)
        put("d1_rsi", (a["rsi14"][d] - 50.0) * sign + 50.0)
        put("d1_adx", a["adx14"][d])
        put("d1_atr_ratio", a["atr_ratio"][d])
        put("d1_change_20d_pct", (close / a["close"][d - 20] - 1) * 100 if d >= 20 else np.nan, True)
    h = p.last_closed("H1", t)
    if h is not None:
        a = p.a["H1"]
        put("h1_ema200_dist_atr", a["ema_distance_atr"][h], True)
        put("h1_rsi", (a["rsi14"][h] - 50.0) * sign + 50.0)
        put("h1_atr_ratio", a["atr_ratio"][h])
        put("h1_change_24h_pct", (a["close"][h] / a["close"][h - 24] - 1) * 100 if h >= 24 else np.nan, True)
    q = p.last_closed("M15", t)
    if q is not None:
        a = p.a["M15"]
        put("m15_close_vs_ema50_pct", (a["close"][q] / a["ema50"][q] - 1) * 100, True)
        put("m15_ema50_slope_pct", a["ema50_slope"][q] / a["ema50"][q] * 100, True)

    atr, close = m5["atr14"][i], m5["close"][i]
    lookback = config.SCALP_PULLBACK_BARS
    window = m5["rsi14"][max(0, i - lookback):i]
    put("m5_rsi", (m5["rsi14"][i] - 50.0) * sign + 50.0)
    put("m5_rsi_extreme", ((np.nanmin(window) if sign > 0 else np.nanmax(window)) - 50.0) * sign + 50.0
        if len(window) else np.nan)
    put("m5_pullback_bars", float(np.sum((window < config.SCALP_RSI_PULLBACK) if sign > 0
                                         else (window > 100 - config.SCALP_RSI_PULLBACK))) if len(window) else np.nan)
    put("m5_atr_pct", atr / close * 100 if close else np.nan)
    put("m5_atr_ratio", m5["atr_ratio"][i])
    put("m5_rel_volume", m5["rel_volume"][i])
    put("m5_ema20_vs_50_atr", (m5["ema20"][i] - m5["ema50"][i]) / atr if atr else np.nan, True)
    put("m5_close_vs_ema50_atr", (close - m5["ema50"][i]) / atr if atr else np.nan, True)
    rng = m5["high"][i] - m5["low"][i]
    put("m5_body_ratio", abs(close - m5["open"][i]) / rng if rng else 0.0)
    put("m5_close_location", ((close - m5["low"][i]) / rng if rng else 0.5) if sign > 0
        else ((m5["high"][i] - close) / rng if rng else 0.5))
    recent = slice(max(0, i - config.SCALP_ROOM_BARS), i + 1)
    swing_high, swing_low = float(np.max(m5["high"][recent])), float(np.min(m5["low"][recent]))
    span = swing_high - swing_low
    put("m5_retrace_pct", ((swing_high - close) / span if sign > 0 else (close - swing_low) / span) * 100 if span else np.nan)
    put("m5_room_atr", ((swing_high - close) if sign > 0 else (close - swing_low)) / atr if atr else np.nan)
    stop = (close - m5["swing_low"][i]) if sign > 0 else (m5["swing_high"][i] - close)
    put("m5_swing_stop_atr", stop / atr if atr else np.nan)
    put("spread_atr", spread_price / atr if atr else np.nan)

    when = pd.Timestamp(int(t), unit="s")  # broker server time (New York + 7h)
    put("server_hour", when.hour + when.minute / 60)
    put("weekday", when.weekday())
    out["side"] = side
    return out


def recent_m5(p: Prepared, bars: int = 8) -> Dict[str, Any]:
    """The last few M5 closes/RSI values for the AI's context."""
    m5 = p.a["M5"]
    return {"closes": [float(v) for v in m5["close"][-bars:]],
            "rsi14": [None if _nan(v) else round(float(v), 1) for v in m5["rsi14"][-bars:]],
            "highs": [float(v) for v in m5["high"][-bars:]], "lows": [float(v) for v in m5["low"][-bars:]]}
