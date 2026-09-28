"""
M5 pullback scalper: the entry rules shared by the live engine and the backtest.

    D1   trend      close above a rising EMA200 -> longs only; below a falling one -> shorts only
    M15  momentum   close on the trend side of a sloping EMA50
    M5   trigger    RSI14 pulled back below SCALP_RSI_PULLBACK (above 100 - it for shorts) within the last
                    SCALP_PULLBACK_BARS bars and has now turned back, on a candle closing in the trend
                    direction, with price still on the trend side of the M5 EMA50
    stop            beyond the last SCALP_SWING_BARS-bar swing (+0.1 ATR), clamped to 1-2 x M5 ATR14
    target          stop x SCALP_REWARD_RISK; the engine closes the trade after SCALP_TIME_STOP_MINUTES
    session         London open (07:00 London) to midday New York (12:00 NY), DST-aware

Indicators are computed once per frame (vectorised), so the backtest can evaluate every
historical M5 bar with exactly the code the live engine runs on the latest closed bar.
"""
from __future__ import annotations

from datetime import datetime, timezone
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
    """Weekdays from the London open (London time) until midday New York (NY time)."""
    now = now or datetime.now(timezone.utc)
    london, new_york = now.astimezone(_LONDON), now.astimezone(_NEW_YORK)
    if london.weekday() >= 5 or new_york.weekday() >= 5:
        return False
    return london.hour >= config.SCALP_SESSION_START_LONDON and new_york.hour < config.SCALP_SESSION_END_NEW_YORK


def session_mask(utc_times: pd.DatetimeIndex) -> np.ndarray:
    """in_session for many UTC timestamps at once (backtest)."""
    london, new_york = utc_times.tz_convert(_LONDON), utc_times.tz_convert(_NEW_YORK)
    mask = ((london.weekday < 5) & (new_york.weekday < 5) & (london.hour >= config.SCALP_SESSION_START_LONDON)
            & (new_york.hour < config.SCALP_SESSION_END_NEW_YORK))
    return np.asarray(mask, dtype=bool)


def session_note(now: Optional[datetime] = None) -> str:
    return (f"London {config.SCALP_SESSION_START_LONDON:02d}:00 to New York {config.SCALP_SESSION_END_NEW_YORK:02d}:00 "
            f"({'open now' if in_session(now) else 'closed now'})")


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
            elif tf == "M15":
                f["ema50"] = _col(ta.ema(close, length=50, talib=False), idx)
                f["ema50_slope"] = f["ema50"] - f["ema50"].shift(3)
            elif tf == "M5":
                f["ema20"] = _col(ta.ema(close, length=20, talib=False), idx)
                f["ema50"] = _col(ta.ema(close, length=50, talib=False), idx)
                f["rsi14"] = _col(ta.rsi(close, length=14, talib=False), idx)
                f["atr14"] = _col(ta.atr(high, low, close, length=14, talib=False), idx)
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


def evaluate(p: Prepared, i: Optional[int] = None, spread_price: float = 0.0) -> Dict[str, Any]:
    """
    The setup on M5 bar ``i`` (default: the latest closed bar). Returns {"setup": dict | None,
    "reason": why not / what was found, "trend": "UP" | "DOWN" | None}.
    """
    m5 = p.a["M5"]
    i = len(m5["close"]) - 1 if i is None else i
    lookback = config.SCALP_PULLBACK_BARS
    if i < 60:
        return {"setup": None, "reason": "not enough M5 history", "trend": None}
    t = m5["close_time"][i]

    d = p.last_closed("D1", t)
    if d is None or _nan(p.a["D1"]["ema200"][d]) or _nan(p.a["D1"]["ema200_slope"][d]):
        return {"setup": None, "reason": "not enough D1 history for EMA200", "trend": None}
    d_close, d_ema, d_slope = p.a["D1"]["close"][d], p.a["D1"]["ema200"][d], p.a["D1"]["ema200_slope"][d]
    trend = "UP" if d_close > d_ema and d_slope > 0 else "DOWN" if d_close < d_ema and d_slope < 0 else None
    if trend is None:
        return {"setup": None, "reason": "D1 trend unclear (price and EMA200 slope disagree)", "trend": None}

    q = p.last_closed("M15", t)
    if q is None or _nan(p.a["M15"]["ema50_slope"][q]):
        return {"setup": None, "reason": "not enough M15 history", "trend": trend}
    q_close, q_ema, q_slope = p.a["M15"]["close"][q], p.a["M15"]["ema50"][q], p.a["M15"]["ema50_slope"][q]
    if (trend == "UP" and not (q_close > q_ema and q_slope > 0)) or (trend == "DOWN" and not (q_close < q_ema and q_slope < 0)):
        return {"setup": None, "reason": f"M15 momentum not with the D1 {trend.lower()}trend", "trend": trend}

    close, open_, rsi, atr = m5["close"][i], m5["open"][i], m5["rsi14"][i], m5["atr14"][i]
    ema50, prev_rsi = m5["ema50"][i], m5["rsi14"][i - 1]
    window = m5["rsi14"][i - lookback:i]
    if _nan(rsi) or _nan(atr) or _nan(ema50) or atr <= 0 or not np.isfinite(window).all():
        return {"setup": None, "reason": "M5 indicators warming up", "trend": trend}
    level = config.SCALP_RSI_PULLBACK
    if trend == "UP":
        pulled_back = window.min() < level
        turned = rsi >= level and rsi > prev_rsi and close > open_ and close > ema50
    else:
        pulled_back = window.max() > 100 - level
        turned = rsi <= 100 - level and rsi < prev_rsi and close < open_ and close < ema50
    if not pulled_back:
        return {"setup": None, "reason": f"no M5 pullback yet (RSI {rsi:.0f})", "trend": trend}
    if not turned:
        return {"setup": None, "reason": f"M5 pullback not turned yet (RSI {rsi:.0f})", "trend": trend}

    side = "BUY" if trend == "UP" else "SELL"
    if side == "BUY":
        entry = close + spread_price  # a long fills at the ask
        raw = entry - m5["swing_low"][i] + 0.1 * atr
    else:
        entry = close  # a short fills at the bid; its stop triggers on the ask
        raw = m5["swing_high"][i] + spread_price - entry + 0.1 * atr
    sl_distance = float(min(max(raw, config.SCALP_SL_ATR_MIN * atr), config.SCALP_SL_ATR_MAX * atr))
    tp_distance = sl_distance * config.SCALP_REWARD_RISK
    extreme = window.min() if side == "BUY" else window.max()
    setup = {
        "side": side,
        "trend": trend,
        "bar_time": int(m5["time"][i]),
        "entry_ref": float(entry),
        "sl_distance": sl_distance,
        "tp_distance": float(tp_distance),
        "risk_reward": config.SCALP_REWARD_RISK,
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
    return {"setup": setup, "reason": setup["reason"], "trend": trend}


def guard_context(p: Prepared, t: float) -> Dict[str, Any]:
    """H1/D1 values the overextension guard reads, as of server time ``t`` (backtest)."""
    context: Dict[str, Any] = {"h1_data": {}, "daily_data": {}}
    for tf, key in (("H1", "h1_data"), ("D1", "daily_data")):
        index = p.last_closed(tf, t)
        if index is not None:
            context[key] = {"ema_distance_atr": p.a[tf]["ema_distance_atr"][index], "rsi14": p.a[tf]["rsi14"][index]}
            context[key] = {k: (None if _nan(v) else float(v)) for k, v in context[key].items()}
    return context


def recent_m5(p: Prepared, bars: int = 8) -> Dict[str, Any]:
    """The last few M5 closes/RSI values for the AI's context."""
    m5 = p.a["M5"]
    return {"closes": [float(v) for v in m5["close"][-bars:]],
            "rsi14": [None if _nan(v) else round(float(v), 1) for v in m5["rsi14"][-bars:]],
            "highs": [float(v) for v in m5["high"][-bars:]], "lows": [float(v) for v in m5["low"][-bars:]]}
