"""
Backtest of the M5 pullback scalper on the broker's own history.

The same scalper.evaluate() the live engine runs is applied to every closed M5 bar
inside the session. A setup is entered at the next bar's open (longs at the ask =
bid + that bar's recorded spread), then followed bar by bar until the stop, the
target or the time stop (a bar touching both counts as a stop). Trades are then
sized like the real account would size them: % risk of the running balance,
the broker's lot step and minimum lot, and the same min-lot tolerance.

Costs: the recorded spread (+ an optional stress), slippage on entries and stop /
time exits (not on take-profit limits) and a commission per lot, converted to R.

Every report also carries the professional checks from validation.py (month-by-month
stability, Monte Carlo drawdowns, Deflated Sharpe Ratio after the number of variants
tried), the exit variants (FIXED, BREAKEVEN, PARTIAL, TRAIL) with a walk-forward choice
between them, a +1 pip cost stress, and the account run with and without the
total-drawdown kill-switch and risk throttle.

What it does not simulate: the AI confirm/veto step (it cannot be replayed
honestly), news blocks, and cross-symbol risk caps. The overextension guard and
the USD-direction guard are measured, not applied: results are split into setups
each guard would have penalised and the rest.

    .venv\\Scripts\\python.exe backtest.py --days 60
    .venv\\Scripts\\python.exe backtest.py --days 90 --symbols EURUSD,GBPUSD --risk 2 --balance 100
    .venv\\Scripts\\python.exe backtest.py --source history --days 1250 --slippage 2 --commission 7
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import ai_brain
import config
import data_engine
import fxhistory
import scalper
import validation

logger = logging.getLogger("hedgefund.backtest")

M5_PER_DAY = 288
IN_SAMPLE_SHARE = 0.7
EXIT_MODES = ("FIXED", "BREAKEVEN", "PARTIAL", "TRAIL")
STRESS_PIPS = 1.0  # the "still positive with +1 pip of extra cost" test


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_history(symbol: str, days: int, source: str = "mt5") -> Dict[str, Any]:
    """
    Closed bars for every timeframe, with enough warm-up for EMA200 on D1 and H1 (blocking).
    source "mt5" reads the broker's history (about 16 months); "history" reads the downloaded
    FXCM + HistData bid/ask history (fxhistory.py, 5+ years).
    """
    counts = {"M5": days * M5_PER_DAY + 400, "M15": days * 96 + 400, "H1": days * 24 + 400, "D1": days + 300}
    if source in DOWNLOADED:
        store = DOWNLOADED[source]
        if not store.available(symbol):
            store.build([symbol])
        frames = {tf: store.load_bars(symbol, tf).iloc[-count:].reset_index(drop=True) for tf, count in counts.items()}
        check_coverage(symbol, frames, days)
        return frames
    return {tf: data_engine.fetch_bars(symbol, scalper.TIMEFRAMES[tf], count) for tf, count in counts.items()}


DOWNLOADED = {"history": fxhistory}  # downloaded sources, on the New York + 7h server clock
SOURCES = ("mt5", *DOWNLOADED)


def best_source(symbols: List[str]) -> str:
    """The longest history available for every symbol: the downloaded FXCM + HistData history, else the broker's."""
    for name, store in DOWNLOADED.items():
        if symbols and all(store.available(s) for s in symbols):
            return name
    return "mt5"


def default_days(source: str) -> int:
    return 1250 if source in DOWNLOADED else 330


D1_WARMUP = 210  # D1 bars the EMA200 trend (+ its 5-day slope) needs before the first setup
MAX_GAP_DAYS = 4  # longer holes than a long weekend mean missing downloads


def check_coverage(symbol: str, frames: Dict[str, Any], days: int) -> int:
    """
    Refuse downloaded history the scalper cannot use honestly: too short for the D1 EMA200 warm-up, or
    with holes. Returns the number of trading days that can actually produce setups.
    """
    d1, m5 = frames["D1"], frames["M5"]
    tradable = len(d1) - D1_WARMUP
    if tradable < 20:
        raise ValueError(f"only {len(d1)} daily bars of {symbol} downloaded; the D1 EMA200 trend needs "
                         f"{D1_WARMUP}+ (download more history: python fxhistory.py all --years 5)")
    gaps = np.diff(m5["time"].to_numpy(dtype="int64")) / 86400.0
    worst = float(gaps.max()) if len(gaps) else 0.0
    if worst > MAX_GAP_DAYS:
        at = datetime.fromtimestamp(int(m5["time"].iloc[int(gaps.argmax())]), timezone.utc)
        raise ValueError(f"{symbol} history has holes (a {worst:.0f}-day gap after {at:%Y-%m-%d}): the download is "
                         f"incomplete (download it again: python fxhistory.py all --years 5)")
    return min(days, tradable)


def default_spec(symbol: str, last_price: float) -> Dict[str, float]:
    """Contract specs when MT5 is not connected: standard 100k FX lots, 100 oz gold."""
    code = fxhistory.instrument(symbol)
    point = fxhistory.point_size(code)
    if code.startswith(("XAU", "XAG")):
        tick_value = 100 * point if code.startswith("XAU") else 5000 * point
    elif code.endswith("USD"):
        tick_value = 100_000 * point
    else:  # USD-based pairs: a point is worth 100k x point in the quote currency, converted back to USD
        tick_value = 100_000 * point / last_price if last_price else 1.0
    return {"point": point, "digits": 3 if point == 0.001 else 2 if point == 0.01 else 5, "tick_size": point,
            "tick_value": tick_value, "volume_min": 0.01, "volume_step": 0.01}


# -----------------------------------------------------------------------------
# Simulation (per symbol, in R)
# -----------------------------------------------------------------------------
def scan_setups(prepared: scalper.Prepared, point: float, days: int, extra_spread_points: float = 0.0,
                new_york_plus_7: bool = False) -> Dict[str, Any]:
    """Every in-session M5 bar the scalper finds a setup on (evaluate is pure, so this runs once per symbol)."""
    m5 = prepared.a["M5"]
    n = len(m5["close"])
    first = max(60, n - days * M5_PER_DAY)
    utc_close = data_engine.server_epochs_to_utc(m5["close_time"].astype("int64"), new_york_plus_7)
    in_session = scalper.session_mask(utc_close)
    day_keys = [data_engine.trading_day_key(t.to_pydatetime()) if t is not None and t == t else "" for t in utc_close]
    spreads = (m5["spread"] + extra_spread_points) * point  # extra = stress test for wider live spreads
    setups: Dict[int, Dict[str, Any]] = {}
    for i in range(first, n - 1):
        if in_session[i]:
            setup = scalper.evaluate(prepared, i, spread_price=float(spreads[i]))["setup"]
            if setup is not None:
                setups[i] = setup
    return {"setups": setups, "utc_close": utc_close, "day_keys": day_keys, "spreads": spreads, "n": n}


def _exit(m5: Dict[str, np.ndarray], spreads: np.ndarray, side: str, entry: float, sl_d: float, tp_d: float,
          j0: int, j_last: int, mode: str, slip: float) -> tuple:
    """
    Follow a trade bar by bar. Returns (R before commission, exit reason, exit bar).
    Within a bar the stop is checked before the target (conservative); stop moves take effect from the
    next bar. FIXED: stop / target / time. BREAKEVEN: stop to entry once +1R was reached. PARTIAL: half
    closed at +1R, stop to entry for the rest. TRAIL: no target; from +1R the stop trails 1R behind the best price.
    """
    buy = side == "BUY"
    sign = 1.0 if buy else -1.0
    stop = entry - sign * sl_d
    target = entry + sign * tp_d if mode != "TRAIL" else None
    one_r = entry + sign * sl_d
    banked, size, armed = 0.0, 1.0, False  # R already realised, open fraction, +1R reached

    def r_at(price: float) -> float:
        return sign * (price - entry) / sl_d

    for j in range(j0, j_last + 1):
        high, low, spread_j = m5["high"][j], m5["low"][j], float(spreads[j])
        best, worst = (high, low) if buy else (low + spread_j, high + spread_j)  # longs exit at the bid, shorts at the ask
        if (worst <= stop) if buy else (worst >= stop):
            reason = "SL" if r_at(stop) < 0 else ("BE" if abs(r_at(stop)) < 1e-9 else "TRAIL")
            return banked + size * r_at(stop - sign * slip), reason, j
        if mode == "PARTIAL" and not armed and ((best >= one_r) if buy else (best <= one_r)):
            banked, size = 0.5 * 1.0, 0.5
        if target is not None and ((best >= target) if buy else (best <= target)):
            return banked + size * r_at(target), "TP", j
        if (best >= one_r) if buy else (best <= one_r):
            armed = True
        if armed and mode in ("BREAKEVEN", "PARTIAL"):
            stop = entry
        elif armed and mode == "TRAIL":
            stop = max(stop, best - sl_d) if buy else min(stop, best + sl_d)
    close = m5["close"][j_last] + (float(spreads[j_last]) if not buy else 0.0)
    return banked + size * r_at(close - sign * slip), "TIME", j_last


def fill_setup(m5: Dict[str, np.ndarray], spreads: np.ndarray, i: int, setup: Dict[str, Any], time_stop_bars: int,
               exit_mode: str = "FIXED", slip: float = 0.0) -> Optional[tuple]:
    """
    The trade a setup on bar ``i`` becomes: (entry, entry spread, R before commission, exit reason, exit bar),
    or None when the live engine would refuse it for its spread. Shared by the backtest and the ML labels.
    """
    # Enter at the next bar's open; re-anchor the stop/target distances there, like the live engine.
    j0 = i + 1
    if j0 >= len(m5["close"]):
        return None
    entry_spread = float(spreads[j0])
    # The live engine refuses a trade when the spread is too large a share of the stop, both when it
    # decides and when it sends the order (rollover and news spikes).
    limit = config.MAX_SPREAD_TO_STOP * setup["sl_distance"]
    if float(spreads[i]) > limit or entry_spread > limit:
        return None
    side = setup["side"]
    entry = m5["open"][j0] + entry_spread + slip if side == "BUY" else m5["open"][j0] - slip
    r_multiple, exit_reason, j_exit = _exit(m5, spreads, side, entry, setup["sl_distance"], setup["tp_distance"], j0,
                                            min(len(m5["close"]) - 1, j0 + time_stop_bars - 1), exit_mode, slip)
    return entry, entry_spread, float(r_multiple), exit_reason, j_exit


def simulate_symbol(symbol: str, prepared: scalper.Prepared, point: float, days: int,
                    time_stop_bars: int, cooldown_bars: int, max_per_day: int,
                    extra_spread_points: float = 0.0, new_york_plus_7: bool = False, exit_mode: str = "FIXED",
                    slippage_points: float = 0.0, commission_per_lot: float = 0.0,
                    value_per_price: Optional[float] = None, scan: Optional[Dict[str, Any]] = None,
                    stress_points: float = 0.0, market_at: Optional[Callable[[str, float], Dict[str, Any]]] = None,
                    with_guards: bool = True) -> List[Dict[str, Any]]:
    """
    Trades of one symbol in R. ``scan`` reuses scan_setups() output (exit variants and cost stress re-run
    only this cheap part); ``stress_points`` widens every spread after the setups were found.
    ``market_at(symbol, t)`` adds the other pairs' day moves so the USD guard can be measured.
    """
    m5 = prepared.a["M5"]
    scan = scan or scan_setups(prepared, point, days, extra_spread_points, new_york_plus_7)
    n, utc_close, day_keys = scan["n"], scan["utc_close"], scan["day_keys"]
    spreads = scan["spreads"] + stress_points * point
    slip = slippage_points * point

    trades: List[Dict[str, Any]] = []
    busy_until = -1
    last_loss_bar = {"BUY": -10**9, "SELL": -10**9}
    per_day: Dict[str, int] = {}
    for i in sorted(scan["setups"]):
        setup = scan["setups"][i]
        if i <= busy_until or per_day.get(day_keys[i], 0) >= max_per_day:
            continue
        side = setup["side"]
        if i - last_loss_bar[side] < cooldown_bars:
            continue

        filled = fill_setup(m5, spreads, i, setup, time_stop_bars, exit_mode, slip)
        if filled is None:
            continue
        entry, entry_spread, r_multiple, exit_reason, j_exit = filled
        sl_d = setup["sl_distance"]
        commission_r = commission_per_lot / (sl_d * value_per_price) if commission_per_lot and value_per_price else 0.0
        r_multiple -= commission_r

        trade = {
            "symbol": symbol, "side": side, "entry_time": utc_close[i].isoformat(),
            "exit_time": utc_close[j_exit].isoformat(), "day": day_keys[i], "entry": round(entry, 6),
            "sl_distance": sl_d, "exit_reason": exit_reason,
            "r": round(float(r_multiple), 3), "spread_points": int(m5["spread"][i + 1]),
            "spread_share_of_stop": round(entry_spread / sl_d, 3), "commission_r": round(commission_r, 4),
        }
        if with_guards:
            t = m5["close_time"][i]
            context = scalper.guard_context(prepared, t)
            if market_at is not None:
                context.update(market_at(symbol, t))
            guards = ai_brain.protective_guards(context, symbol, side)
            trade["guard_points"] = sum(g["points"] for g in guards if g["id"] != "G-USD")
            trade["guards"] = [g["id"] for g in guards]
        trades.append(trade)
        per_day[day_keys[i]] = per_day.get(day_keys[i], 0) + 1
        busy_until = j_exit
        if r_multiple < 0:
            last_loss_bar[side] = j_exit
    return trades


def day_moves(frames: Dict[str, Dict[str, Any]]) -> Callable[[str, float], Dict[str, Any]]:
    """
    ``market_at(symbol, t)``: every pair's change since its server-day open, as of server time ``t``
    (the same numbers the live USD-direction guard reads). All sources share one server clock.
    """
    table = {}
    for symbol, tf in frames.items():
        m5 = tf["M5"]
        close_time = m5["time"].astype("int64").to_numpy() + 300
        day = (close_time - 1) // 86400
        opens = m5.groupby(day)["open"].transform("first").to_numpy(dtype=float)
        change = (m5["close"].to_numpy(dtype=float) - opens) / opens * 100.0
        table[symbol] = (close_time, day, change)

    def market_at(symbol: str, t: float) -> Dict[str, Any]:
        quotes = {}
        for other, (times, days, change) in table.items():
            k = int(np.searchsorted(times, t, side="right")) - 1
            if k >= 0 and days[k] == (int(t) - 1) // 86400:
                quotes[other] = {"day_change_pct": float(change[k])}
        own = quotes.pop(symbol, {}).get("day_change_pct")
        return {"symbol": symbol, "day_change_pct": own, "correlated_prices": quotes}
    return market_at


# -----------------------------------------------------------------------------
# Sizing like the real account
# -----------------------------------------------------------------------------
def size_trades(trades: List[Dict[str, Any]], specs: Dict[str, Dict[str, float]], balance: float,
                risk_percent: float, kill_switch_pct: float = 0.0, throttle_pct: float = 0.0) -> Dict[str, Any]:
    """
    Apply % risk sizing with lot steps/minimums to the trades in time order; realise P/L at exit.
    ``throttle_pct``: trade at THROTTLE_RISK_FACTOR x risk while the balance is that far below its peak.
    ``kill_switch_pct``: no new trades once the balance is that far below its peak (like the live engine,
    which then waits for a manual reset). 0 turns either off.
    """
    events = sorted(trades, key=lambda t: t["entry_time"])
    open_trades: List[Dict[str, Any]] = []
    curve, skipped = [], 0
    peak = balance
    max_dd = 0.0
    halted_at, halted_skipped, throttled = None, 0, 0

    def realise(until: Optional[str]) -> None:
        nonlocal balance, peak, max_dd
        for trade in sorted([t for t in open_trades if until is None or t["exit_time"] <= until],
                            key=lambda t: t["exit_time"]):
            balance += trade["pnl"]
            peak = max(peak, balance)
            max_dd = max(max_dd, (peak - balance) / peak * 100.0 if peak > 0 else 0.0)
            curve.append({"t": trade["exit_time"], "balance": round(balance, 2)})
            open_trades.remove(trade)

    for trade in events:
        realise(trade["entry_time"])
        drawdown = (peak - balance) / peak * 100.0 if peak > 0 else 0.0
        if halted_at is None and kill_switch_pct > 0 and drawdown >= kill_switch_pct:
            halted_at = trade["entry_time"]
        if halted_at is not None:
            trade["lots"], trade["pnl"] = 0.0, 0.0
            halted_skipped += 1
            continue
        spec = specs[trade["symbol"]]
        loss_per_lot = trade["sl_distance"] / spec["tick_size"] * spec["tick_value"]
        budget = balance * risk_percent / 100.0
        if throttle_pct > 0 and drawdown >= throttle_pct:
            budget *= config.THROTTLE_RISK_FACTOR
            throttled += 1
        if loss_per_lot <= 0 or budget <= 0:
            trade["lots"], trade["pnl"] = 0.0, 0.0
            skipped += 1
            continue
        lots = math.floor(budget / loss_per_lot / spec["volume_step"] + 1e-9) * spec["volume_step"]
        if lots < spec["volume_min"]:
            if spec["volume_min"] * loss_per_lot <= budget * config.MIN_LOT_RISK_TOLERANCE:
                lots = spec["volume_min"]
            else:
                trade["lots"], trade["pnl"] = 0.0, 0.0
                trade["skipped"] = "minimum lot risks too much"
                skipped += 1
                continue
        trade["lots"] = round(lots, 2)
        trade["pnl"] = round(trade["r"] * lots * loss_per_lot, 2)
        open_trades.append(trade)
    realise(None)
    return {"final_balance": round(balance, 2), "max_drawdown_pct": round(max_dd, 2), "skipped_min_lot": skipped,
            "kill_switch_at": halted_at, "skipped_after_kill_switch": halted_skipped, "throttled_trades": throttled,
            "equity_curve": curve}


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
def stats(trades: List[Dict[str, Any]], days: Optional[float] = None) -> Dict[str, Any]:
    rs = np.array([t["r"] for t in trades], dtype=float)
    if not len(rs):
        return {"trades": 0}
    wins, losses = rs[rs > 0], rs[rs < 0]
    cumulative = np.cumsum(rs)
    drawdown = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:] - cumulative
    exits: Dict[str, int] = {}
    for trade in trades:
        exits[trade["exit_reason"]] = exits.get(trade["exit_reason"], 0) + 1
    result = {
        "trades": int(len(rs)),
        "win_rate": round(float(len(wins) / len(rs) * 100.0), 1),
        "avg_win_r": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss_r": round(float(losses.mean()), 2) if len(losses) else None,
        "expectancy_r": round(float(rs.mean()), 3),
        "net_r": round(float(rs.sum()), 2),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() else None,
        "max_drawdown_r": round(float(drawdown.max()), 2),
        "exits": exits,
    }
    if days:
        result["trades_per_day"] = round(len(rs) / days, 2)
    return result


def verdict(summary: Dict[str, Any], out_of_sample: Dict[str, Any], checks: Optional[Dict[str, Any]] = None,
            stress: Optional[Dict[str, Any]] = None) -> str:
    return validation.verdict(summary, out_of_sample, checks or {}, stress)


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
def _spec(symbol: str, frames: Dict[str, Any], downloaded: bool) -> Optional[Dict[str, float]]:
    with data_engine.MT5_LOCK:
        info = data_engine.mt5.symbol_info(symbol)
    if info is not None:
        return {"point": float(info.point), "digits": int(info.digits),
                "tick_size": float(info.trade_tick_size) or float(info.point),
                "tick_value": float(getattr(info, "trade_tick_value_loss", 0) or info.trade_tick_value),
                "volume_min": float(info.volume_min), "volume_step": float(info.volume_step) or 0.01}
    if downloaded:
        return default_spec(symbol, float(frames["M5"]["close"].iloc[-1]))
    return None


def run_backtest(symbols: List[str], days: int, risk_percent: float, balance: float,
                 progress: Optional[Callable[[str], None]] = None,
                 extra_spread_points: float = 0.0, source: str = "mt5", exit_mode: str = "FIXED",
                 slippage_points: Optional[float] = None, commission_per_lot: Optional[float] = None,
                 trials: Optional[int] = None, save: bool = True) -> Dict[str, Any]:
    """
    ``extra_spread_points`` and ``slippage_points`` are given in pips (x10 points on 3/5-digit symbols);
    ``commission_per_lot`` is a round trip in account currency.
    """
    started = datetime.now(timezone.utc)
    slippage_pips = config.BACKTEST_SLIPPAGE_POINTS / 10 if slippage_points is None else slippage_points
    commission = config.BACKTEST_COMMISSION_PER_LOT if commission_per_lot is None else commission_per_lot
    time_stop_bars = max(1, config.SCALP_TIME_STOP_MINUTES // 5)
    cooldown_bars = config.LOSS_COOLDOWN_MINUTES // 5
    specs: Dict[str, Dict[str, float]] = {}
    per_symbol: Dict[str, Any] = {}
    period = {"from": None, "to": None}
    ny7 = source in DOWNLOADED  # downloaded bars are rebuilt on the New York + 7h server clock
    if not ny7:
        data_engine.server_utc_offset_seconds(symbols)

    loaded: Dict[str, Dict[str, Any]] = {}
    for number, symbol in enumerate(symbols, start=1):
        if progress:
            progress(f"{symbol} ({number}/{len(symbols)}): loading {source} history")
        try:
            frames = load_history(symbol, days, source)
            spec = _spec(symbol, frames, ny7)
        except Exception as exc:
            per_symbol[symbol] = {"error": str(exc)}
            continue
        if spec is None:
            per_symbol[symbol] = {"error": "symbol_info unavailable"}
            continue
        specs[symbol] = spec
        loaded[symbol] = frames
        if ny7 and progress and check_coverage(symbol, frames, days) < days:
            progress(f"{symbol}: only {check_coverage(symbol, frames, days)} of {days} days can trade "
                     f"(the first {D1_WARMUP} daily bars warm up the EMA200 trend)")
    if not loaded:
        problems = "; ".join(f"{s}: {v['error']}" for s, v in per_symbol.items())
        raise RuntimeError(f"no usable price data (source: {source}), nothing was tested ({problems})")
    market_at = day_moves(loaded)

    variants: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in EXIT_MODES}
    stressed: List[Dict[str, Any]] = []
    for number, (symbol, frames) in enumerate(loaded.items(), start=1):
        if progress:
            progress(f"{symbol} ({number}/{len(loaded)}): simulating")
        spec = specs[symbol]
        pip = 10 if spec["digits"] in (3, 5) else 1  # pips -> points
        prepared = scalper.prepare(frames)
        scan = scan_setups(prepared, spec["point"], days, extra_spread_points * pip, ny7)
        common = dict(slippage_points=slippage_pips * pip, commission_per_lot=commission,
                      value_per_price=spec["tick_value"] / spec["tick_size"], scan=scan)
        args = (symbol, prepared, spec["point"], days, time_stop_bars, cooldown_bars, config.SCALP_MAX_TRADES_PER_SYMBOL)
        for mode in EXIT_MODES:
            variants[mode].extend(simulate_symbol(*args, exit_mode=mode, market_at=market_at,
                                                  with_guards=mode == exit_mode, **common))
        stressed.extend(simulate_symbol(*args, exit_mode=exit_mode, stress_points=STRESS_PIPS * pip,
                                        with_guards=False, **common))
        m5_times = data_engine.server_epochs_to_utc(
            prepared.a["M5"]["close_time"][-days * M5_PER_DAY:].astype("int64"), ny7)
        span_from, span_to = m5_times[0].isoformat(), m5_times[-1].isoformat()
        period["from"] = min(filter(None, [period["from"], span_from]))
        period["to"] = max(filter(None, [period["to"], span_to]))
        per_symbol[symbol] = stats([t for t in variants[exit_mode] if t["symbol"] == symbol], days)

    for trades in (*variants.values(), stressed):
        trades.sort(key=lambda t: t["entry_time"])
    all_trades = variants[exit_mode]
    split = int(len(all_trades) * IN_SAMPLE_SHARE)
    kill, throttle = config.MAX_TOTAL_DRAWDOWN_PERCENT, config.DRAWDOWN_THROTTLE_PERCENT
    sizing = size_trades([dict(t) for t in all_trades], specs, balance, risk_percent, kill, throttle)
    unprotected = size_trades([dict(t) for t in all_trades], specs, balance, risk_percent)
    summary = stats(all_trades, days)
    out_of_sample = stats(all_trades[split:])
    checks = validation.report(all_trades, risk_percent, trials)
    stress = stats(stressed)
    train_months, test_months = validation.window_plan(all_trades)
    exit_variants = {mode: stats(trades) for mode, trades in variants.items()}
    report = {
        "generated_at": started.isoformat(timespec="seconds"),
        "duration_seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "strategy": "SCALP M5 pullback in D1 trend",
        "params": {"days": days, "symbols": symbols, "risk_percent": risk_percent, "start_balance": balance,
                   "reward_risk": config.SCALP_REWARD_RISK, "time_stop_minutes": config.SCALP_TIME_STOP_MINUTES,
                   "rsi_pullback": config.SCALP_RSI_PULLBACK, "max_trades_per_symbol": config.SCALP_MAX_TRADES_PER_SYMBOL,
                   "strict_guard": config.SCALP_STRICT_GUARD, "medium_trend": config.SCALP_MEDIUM_TREND,
                   "min_adx": config.SCALP_MIN_ADX, "trigger": config.SCALP_TRIGGER, "room_min_r": config.SCALP_ROOM_MIN_R,
                   "target": config.SCALP_TARGET, "exit_mode": exit_mode,
                   "loss_cooldown_minutes": config.LOSS_COOLDOWN_MINUTES, "extra_spread_pips": extra_spread_points,
                   "slippage_pips": slippage_pips, "commission_per_lot": commission,
                   "kill_switch_pct": kill, "throttle_pct": throttle,
                   "source": source,
                   "session": scalper.session_note().split(" (")[0]},
        "period": period,
        "summary": summary,
        "in_sample": stats(all_trades[:split]),
        "out_of_sample": out_of_sample,
        "validation": {**checks,
                       "walk_forward": validation.walk_forward({exit_mode: all_trades}, train_months, test_months),
                       "cost_stress": {"extra_pips": STRESS_PIPS, **stress,
                                       "note": "same setups, every spread widened (approximation)"}},
        "exit_variants": {"stats": exit_variants,
                          "walk_forward": validation.walk_forward(variants, train_months, test_months,
                                                                  baseline=exit_mode)},
        "account": {key: sizing[key] for key in ("final_balance", "max_drawdown_pct", "skipped_min_lot", "kill_switch_at",
                                                 "skipped_after_kill_switch", "throttled_trades")},
        "account_unprotected": {key: unprotected[key] for key in ("final_balance", "max_drawdown_pct")},
        "by_symbol": per_symbol,
        "guards": {"not_penalised": stats([t for t in all_trades if not t["guard_points"]]),
                   "penalised": stats([t for t in all_trades if t["guard_points"]])},
        "usd_guard": {"with_usd_flow": stats([t for t in all_trades if "G-USD" not in t["guards"]]),
                      "against_usd_flow": stats([t for t in all_trades if "G-USD" in t["guards"]])},
        "by_hour_utc": {f"{hour:02d}": stats([t for t in all_trades if int(t["entry_time"][11:13]) == hour])
                        for hour in sorted({int(t["entry_time"][11:13]) for t in all_trades})},
        "verdict": verdict(summary, out_of_sample, checks, stress),
        "not_simulated": ["AI confirm/veto", "news guard", "per-currency and daily-loss caps",
                          "USD-direction and overextension guards are measured (split), not applied"],
        "equity_curve": sizing["equity_curve"][-400:],
        "trades": all_trades[-300:],
    }
    if save:
        config.BACKTEST_DIR.mkdir(exist_ok=True)
        path = config.BACKTEST_DIR / f"scalp-{started:%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["file"] = path.name
    report["all_trades"] = all_trades  # in memory only (research loop); the saved file keeps the last 300
    return report


def latest_report() -> Optional[Dict[str, Any]]:
    if not config.BACKTEST_DIR.exists():
        return None
    files = sorted(config.BACKTEST_DIR.glob("scalp-*.json"))
    if not files:
        return None
    try:
        report = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    report["file"] = files[-1].name
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest the M5 pullback scalper on MT5 or downloaded history.")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--risk", type=float, default=None, help="risk %% per trade (default: DEFAULT_RISK_PERCENT)")
    parser.add_argument("--balance", type=float, default=None, help="start balance (default: the account equity)")
    parser.add_argument("--source", choices=SOURCES, default="mt5",
                        help="price history: the broker's (mt5) or the downloaded FXCM + HistData bid/ask data")
    parser.add_argument("--extra-spread", type=float, default=0.0,
                        help="add this many pips to every recorded spread (stress test for a live account's costs)")
    parser.add_argument("--slippage", type=float, default=None,
                        help="pips of slippage on entries and stop/time exits (default: BACKTEST_SLIPPAGE_POINTS)")
    parser.add_argument("--commission", type=float, default=None,
                        help="round-trip commission per lot (default: BACKTEST_COMMISSION_PER_LOT)")
    parser.add_argument("--exit", choices=EXIT_MODES, default="FIXED", help="exit variant for the main result")
    parser.add_argument("--trials", type=int, default=None, help="variants tried so far (default: BACKTEST_TRIALS)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", stream=sys.stdout)
    import settings_store
    settings_store.load_saved()
    if not data_engine.initialize_mt5():
        print("MT5 is not available")
        return 1
    try:
        account = data_engine.get_account_snapshot() or {}
        requested = [s for s in args.symbols.split(",") if s.strip()] or (
            config.SYMBOLS_LIVE if account.get("account_mode") == "LIVE" else config.SYMBOLS_DEMO)
        offered = [item["name"] for item in data_engine.broker_symbols() or []]
        symbols, _, missing = data_engine.resolve_symbols(requested, offered)
        if missing:
            print(f"Not offered and skipped: {', '.join(missing)}")
        report = run_backtest(symbols, args.days, args.risk or config.DEFAULT_RISK_PERCENT,
                              args.balance or float(account.get("equity") or 100.0), progress=print,
                              extra_spread_points=args.extra_spread, source=args.source, exit_mode=args.exit,
                              slippage_points=args.slippage, commission_per_lot=args.commission, trials=args.trials)
    except RuntimeError as exc:
        print(f"BACKTEST NOT RUN: {exc}")
        return 1
    finally:
        data_engine.shutdown_mt5()
    s, a = report["summary"], report["account"]
    v = report["validation"]
    print(json.dumps({"period": report["period"], "summary": s, "out_of_sample": report["out_of_sample"],
                      "account": a, "account_unprotected": report["account_unprotected"],
                      "significance": v["significance"], "monte_carlo": v["monte_carlo"],
                      "walk_forward": {k: v["walk_forward"][k] for k in ("oos", "positive_windows_share", "windows_count")},
                      "cost_stress": v["cost_stress"],
                      "exit_variants": report["exit_variants"]["stats"],
                      "exit_walk_forward": {k: report["exit_variants"]["walk_forward"][k] for k in ("oos", "picks")},
                      "guards": report["guards"], "usd_guard": report["usd_guard"],
                      "by_symbol": report["by_symbol"]}, indent=2, default=str))
    print("VERDICT:", report["verdict"], "| saved", report["file"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
