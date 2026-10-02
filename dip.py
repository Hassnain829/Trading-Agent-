"""
H1 dip strategy: mean reversion inside the D1 trend (an RSI(2) pullback, Connors-style).

    trend   D1 close above a rising EMA200 -> buy dips only; below a falling one -> sell rallies only
    entry   the last closed H1 bar's RSI(2) is below DIP_RSI_LOW (above 100 - it for a sell)
    stop    DIP_STOP_ATR x H1 ATR14 from the entry, plus a far safety target at DIP_TARGET_R
    exit    the first H1 close back across the H1 EMA5 (at or above it for a buy), the stop, or the time stop

On 10 years of history (18 pairs) about 64% of these trades won and the result was roughly break-even after
spread and swap; the learning agent's job is to find the conditions where it pays. exit_outcome() is the one
exit rule used by shadow trades, the history replay and the engine's open positions.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

import config
import scalper

EXIT_RULE = "EMA5_H1"
H1_SECONDS = 3600


def add_indicators(p: scalper.Prepared) -> scalper.Prepared:
    """Adds the H1 EMA5 and RSI(2) to prepared scalper frames."""
    frame = p.frames["H1"]
    close = frame["close"].astype(float)
    for column, series in (("ema5", ta.ema(close, length=5, talib=False)), ("rsi2", ta.rsi(close, length=2, talib=False))):
        frame[column] = series if series is not None else np.nan
        p.a["H1"][column] = frame[column].to_numpy(dtype=float)
    return p


def prepare(frames: Dict[str, pd.DataFrame]) -> scalper.Prepared:
    return add_indicators(scalper.prepare(frames))


def evaluate(p: scalper.Prepared, h: Optional[int] = None, spread_price: float = 0.0) -> Dict[str, Any]:
    """The dip setup on H1 bar ``h`` (default: the latest closed one): {"stage", "setup", "reason", "trend"}."""
    a = p.a["H1"]
    h = len(a["close"]) - 1 if h is None else h
    if h < 210:
        return {"stage": "WARMUP", "setup": None, "reason": "not enough H1 history", "trend": None}
    t = a["close_time"][h]
    d = p.last_closed("D1", t)
    if d is None or scalper._nan(p.a["D1"]["ema200"][d]) or scalper._nan(p.a["D1"]["ema200_slope"][d]):
        return {"stage": "WARMUP", "setup": None, "reason": "not enough D1 history for EMA200", "trend": None}
    d_close, d_ema, d_slope = p.a["D1"]["close"][d], p.a["D1"]["ema200"][d], p.a["D1"]["ema200_slope"][d]
    trend = "UP" if d_close > d_ema and d_slope > 0 else "DOWN" if d_close < d_ema and d_slope < 0 else None
    if trend is None:
        return {"stage": "NO_TREND", "setup": None, "trend": None,
                "reason": "D1 trend unclear (price and EMA200 slope disagree): no dips to trade"}
    rsi2, atr, close, ema5 = a["rsi2"][h], a["atr14"][h], a["close"][h], a["ema5"][h]
    if scalper._nan(rsi2) or scalper._nan(atr) or scalper._nan(ema5) or atr <= 0:
        return {"stage": "WARMUP", "setup": None, "reason": "H1 indicators warming up", "trend": trend}
    low = config.DIP_RSI_LOW
    if trend == "UP" and rsi2 < low:
        side = "BUY"
    elif trend == "DOWN" and rsi2 > 100 - low:
        side = "SELL"
    else:
        wanted = f"H1 RSI(2) below {low:g}" if trend == "UP" else f"H1 RSI(2) above {100 - low:g}"
        return {"stage": "NO_DIP", "setup": None, "trend": trend,
                "reason": f"D1 {trend.lower()}trend, waiting for a {'dip' if trend == 'UP' else 'rally'}: {wanted} (now {rsi2:.0f})"}
    entry = close + spread_price if side == "BUY" else close  # a long fills at the ask, a short at the bid
    sl_distance = float(config.DIP_STOP_ATR * atr)
    setup = {
        "side": side, "trend": trend, "strategy": "DIP", "kind": "DIP", "exit_rule": EXIT_RULE,
        "bar_time": int(a["time"][h]), "entry_ref": float(entry),
        "sl_distance": sl_distance, "tp_distance": sl_distance * config.DIP_TARGET_R,
        "risk_reward": float(config.DIP_TARGET_R), "atr": float(atr), "sl_atr": float(config.DIP_STOP_ATR),
        "h1_rsi2": round(float(rsi2), 1), "h1_ema5": float(ema5),
        "d1_close": float(d_close), "d1_ema200": float(d_ema),
        "reason": (f"{side} H1 {'dip' if side == 'BUY' else 'rally'} in the D1 {trend.lower()}trend: RSI(2) "
                   f"{rsi2:.0f}, exit on an H1 close back {'above' if side == 'BUY' else 'below'} EMA5"),
    }
    return {"stage": "SETUP", "setup": setup, "reason": setup["reason"], "trend": trend}


def features(p: scalper.Prepared, h: Optional[int] = None, side: Optional[str] = None,
             spread_price: float = 0.0) -> Dict[str, Optional[float]]:
    """The agent's market state at H1 bar ``h``: scalper.features on the M5 bar that closed with it."""
    a = p.a["H1"]
    h = len(a["close"]) - 1 if h is None else h
    i = p.last_closed("M5", a["close_time"][h])
    return scalper.features(p, i, side=side, spread_price=spread_price) if i is not None else {}


def exit_outcome(times: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, ema5: np.ndarray,
                 opened: float, side: str, entry: float, stop: float, target: Optional[float], spread: float,
                 time_stop_seconds: float) -> Optional[tuple]:
    """
    (status, R, bar index) of a dip trade on CLOSED H1 bars (bid prices; ``times`` = bar open times, server
    epoch) opened at server time ``opened``. Stop first when a bar touches both. An H1 close back across EMA5
    exits at that close: WIN when it is a profit, LOSS otherwise. None while the trade is still open.
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    last = None
    for j in range(len(closes)):
        if times[j] + H1_SECONDS <= opened:
            continue  # closed before the entry (a live entry comes a few seconds after the signal bar's close)
        if times[j] >= opened + time_stop_seconds:
            if last is None:
                return None
            close = closes[last] + (spread if side == "SELL" else 0.0)
            moved = (close - entry) if side == "BUY" else (entry - close)
            return "TIMEOUT", round(moved / risk, 3), last
        last = j
        high, low, close = highs[j], lows[j], closes[j]
        if side == "BUY":
            hit_sl, hit_tp = low <= stop, target is not None and high >= target
        else:  # a short exits at the ask = bid + spread
            hit_sl, hit_tp = high + spread >= stop, target is not None and low + spread <= target
        if hit_sl:
            return "LOSS", -1.0, j
        if hit_tp:
            return "WIN", round(abs(target - entry) / risk, 3), j
        if np.isfinite(ema5[j]) and ((side == "BUY" and close >= ema5[j]) or (side == "SELL" and close <= ema5[j])):
            price = close + (spread if side == "SELL" else 0.0)
            r = round(((price - entry) if side == "BUY" else (entry - price)) / risk, 3)
            return ("WIN" if r > 0 else "LOSS"), r, j
    return None


def h1_ema5(closes: np.ndarray) -> np.ndarray:
    series = ta.ema(pd.Series(closes, dtype=float), length=5, talib=False)
    return series.to_numpy(dtype=float) if series is not None else np.full(len(closes), np.nan)
