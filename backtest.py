"""
Backtest of the M5 pullback scalper on the broker's own history.

The same scalper.evaluate() the live engine runs is applied to every closed M5 bar
inside the session. A setup is entered at the next bar's open (longs at the ask =
bid + that bar's recorded spread), then followed bar by bar until the stop, the
target or the time stop (a bar touching both counts as a stop). Trades are then
sized like the real account would size them: % risk of the running balance,
the broker's lot step and minimum lot, and the same min-lot tolerance.

What it does not simulate: the AI confirm/veto step (it cannot be replayed
honestly), the USD-direction guard, news blocks, and cross-symbol risk caps.
The overextension guard is reported separately: results with and without the
setups it would have penalised.

    .venv\\Scripts\\python.exe backtest.py --days 60
    .venv\\Scripts\\python.exe backtest.py --days 90 --symbols EURUSD,GBPUSD --risk 2 --balance 100
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
import scalper

logger = logging.getLogger("hedgefund.backtest")

M5_PER_DAY = 288
IN_SAMPLE_SHARE = 0.7


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def load_history(symbol: str, days: int) -> Dict[str, Any]:
    """Closed bars for every timeframe, with enough warm-up for EMA200 on D1 and H1 (blocking)."""
    counts = {"M5": days * M5_PER_DAY + 400, "M15": days * 96 + 400, "H1": days * 24 + 400, "D1": days + 300}
    return {tf: data_engine.fetch_bars(symbol, scalper.TIMEFRAMES[tf], count) for tf, count in counts.items()}


# -----------------------------------------------------------------------------
# Simulation (per symbol, in R)
# -----------------------------------------------------------------------------
def simulate_symbol(symbol: str, prepared: scalper.Prepared, point: float, days: int,
                    time_stop_bars: int, cooldown_bars: int, max_per_day: int,
                    extra_spread_points: float = 0.0) -> List[Dict[str, Any]]:
    m5 = prepared.a["M5"]
    n = len(m5["close"])
    first = max(60, n - days * M5_PER_DAY)
    utc_close = data_engine.server_epochs_to_utc(m5["close_time"].astype("int64"))
    in_session = scalper.session_mask(utc_close)
    day_keys = [data_engine.trading_day_key(t.to_pydatetime()) if t is not None and t == t else "" for t in utc_close]
    spreads = (m5["spread"] + extra_spread_points) * point  # extra = stress test for wider live spreads

    trades: List[Dict[str, Any]] = []
    busy_until = -1
    last_loss_bar = {"BUY": -10**9, "SELL": -10**9}
    per_day: Dict[str, int] = {}
    for i in range(first, n - 1):
        if i <= busy_until or not in_session[i] or per_day.get(day_keys[i], 0) >= max_per_day:
            continue
        result = scalper.evaluate(prepared, i, spread_price=float(spreads[i]))
        setup = result["setup"]
        if setup is None:
            continue
        side = setup["side"]
        if i - last_loss_bar[side] < cooldown_bars:
            continue

        # Enter at the next bar's open; re-anchor the stop/target distances there, like the live engine.
        j0 = i + 1
        entry_spread = float(spreads[j0])
        entry = m5["open"][j0] + entry_spread if side == "BUY" else m5["open"][j0]
        sl_d, tp_d = setup["sl_distance"], setup["tp_distance"]
        stop = entry - sl_d if side == "BUY" else entry + sl_d
        target = entry + tp_d if side == "BUY" else entry - tp_d
        exit_price, exit_reason, j_exit = None, None, min(n - 1, j0 + time_stop_bars - 1)
        for j in range(j0, j_exit + 1):
            high, low, spread_j = m5["high"][j], m5["low"][j], float(spreads[j])
            if side == "BUY":  # a long exits at the bid
                hit_sl, hit_tp = low <= stop, high >= target
            else:  # a short exits at the ask
                hit_sl, hit_tp = high + spread_j >= stop, low + spread_j <= target
            if hit_sl:
                exit_price, exit_reason, j_exit = stop, "SL", j
                break
            if hit_tp:
                exit_price, exit_reason, j_exit = target, "TP", j
                break
        if exit_price is None:
            exit_price = m5["close"][j_exit] + (float(spreads[j_exit]) if side == "SELL" else 0.0)
            exit_reason = "TIME"
        r_multiple = ((exit_price - entry) if side == "BUY" else (entry - exit_price)) / sl_d

        guards = ai_brain.protective_guards(scalper.guard_context(prepared, m5["close_time"][i]), symbol, side)
        trades.append({
            "symbol": symbol, "side": side, "entry_time": utc_close[i].isoformat(),
            "exit_time": utc_close[j_exit].isoformat(), "day": day_keys[i], "entry": round(entry, 6),
            "exit": round(float(exit_price), 6), "sl_distance": sl_d, "exit_reason": exit_reason,
            "r": round(float(r_multiple), 3), "spread_points": int(m5["spread"][j0]),
            "spread_share_of_stop": round(entry_spread / sl_d, 3),
            "guard_points": sum(g["points"] for g in guards), "guards": [g["id"] for g in guards],
        })
        per_day[day_keys[i]] = per_day.get(day_keys[i], 0) + 1
        busy_until = j_exit
        if r_multiple < 0:
            last_loss_bar[side] = j_exit
    return trades


# -----------------------------------------------------------------------------
# Sizing like the real account
# -----------------------------------------------------------------------------
def size_trades(trades: List[Dict[str, Any]], specs: Dict[str, Dict[str, float]], balance: float,
                risk_percent: float) -> Dict[str, Any]:
    """Apply % risk sizing with lot steps/minimums to the trades in time order; realise P/L at exit."""
    events = sorted(trades, key=lambda t: t["entry_time"])
    open_trades: List[Dict[str, Any]] = []
    curve, skipped = [], 0
    peak = balance
    max_dd = 0.0

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
        spec = specs[trade["symbol"]]
        loss_per_lot = trade["sl_distance"] / spec["tick_size"] * spec["tick_value"]
        budget = balance * risk_percent / 100.0
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


def verdict(summary: Dict[str, Any], out_of_sample: Dict[str, Any]) -> str:
    if summary.get("trades", 0) < 30:
        return "Too few trades to judge; test more days or symbols."
    expectancy, oos = summary["expectancy_r"], out_of_sample.get("expectancy_r")
    if expectancy <= 0:
        return "No edge: the rules lost money after spreads. Do not trade them live as they are."
    if oos is None or oos <= 0:
        return "Positive overall but negative on the most recent 30% of the period: likely unstable; keep testing."
    if expectancy < 0.05:
        return "A thin edge that small cost changes (live spreads, slippage) can erase; demo-test before any live use."
    return "Positive expectancy in both halves of the period; worth forward-testing on demo."


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
def run_backtest(symbols: List[str], days: int, risk_percent: float, balance: float,
                 progress: Optional[Callable[[str], None]] = None,
                 extra_spread_points: float = 0.0) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    time_stop_bars = max(1, config.SCALP_TIME_STOP_MINUTES // 5)
    cooldown_bars = config.LOSS_COOLDOWN_MINUTES // 5
    all_trades: List[Dict[str, Any]] = []
    specs: Dict[str, Dict[str, float]] = {}
    per_symbol: Dict[str, Any] = {}
    period = {"from": None, "to": None}
    data_engine.server_utc_offset_seconds(symbols)
    for number, symbol in enumerate(symbols, start=1):
        if progress:
            progress(f"{symbol} ({number}/{len(symbols)}): loading history")
        try:
            frames = load_history(symbol, days)
            with data_engine.MT5_LOCK:
                info = data_engine.mt5.symbol_info(symbol)
        except Exception as exc:
            per_symbol[symbol] = {"error": str(exc)}
            continue
        if info is None:
            per_symbol[symbol] = {"error": "symbol_info unavailable"}
            continue
        specs[symbol] = {"tick_size": float(info.trade_tick_size) or float(info.point),
                         "tick_value": float(getattr(info, "trade_tick_value_loss", 0) or info.trade_tick_value),
                         "volume_min": float(info.volume_min), "volume_step": float(info.volume_step) or 0.01}
        if progress:
            progress(f"{symbol} ({number}/{len(symbols)}): simulating")
        prepared = scalper.prepare(frames)
        extra = extra_spread_points * (10 if int(info.digits) in (3, 5) else 1)  # given in pips -> points
        trades = simulate_symbol(symbol, prepared, float(info.point), days, time_stop_bars, cooldown_bars,
                                 config.SCALP_MAX_TRADES_PER_SYMBOL, extra)
        all_trades.extend(trades)
        m5_times = data_engine.server_epochs_to_utc(prepared.a["M5"]["close_time"][-days * M5_PER_DAY:].astype("int64"))
        span_from, span_to = m5_times[0].isoformat(), m5_times[-1].isoformat()
        period["from"] = min(filter(None, [period["from"], span_from]))
        period["to"] = max(filter(None, [period["to"], span_to]))
        per_symbol[symbol] = stats(trades, days)

    all_trades.sort(key=lambda t: t["entry_time"])
    split = int(len(all_trades) * IN_SAMPLE_SHARE)
    sizing = size_trades([dict(t) for t in all_trades], specs, balance, risk_percent)
    summary = stats(all_trades, days)
    out_of_sample = stats(all_trades[split:])
    report = {
        "generated_at": started.isoformat(timespec="seconds"),
        "duration_seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "strategy": "SCALP M5 pullback in D1 trend",
        "params": {"days": days, "symbols": symbols, "risk_percent": risk_percent, "start_balance": balance,
                   "reward_risk": config.SCALP_REWARD_RISK, "time_stop_minutes": config.SCALP_TIME_STOP_MINUTES,
                   "rsi_pullback": config.SCALP_RSI_PULLBACK, "max_trades_per_symbol": config.SCALP_MAX_TRADES_PER_SYMBOL,
                   "loss_cooldown_minutes": config.LOSS_COOLDOWN_MINUTES, "extra_spread_pips": extra_spread_points,
                   "session": scalper.session_note().split(" (")[0]},
        "period": period,
        "summary": summary,
        "in_sample": stats(all_trades[:split]),
        "out_of_sample": out_of_sample,
        "account": {key: sizing[key] for key in ("final_balance", "max_drawdown_pct", "skipped_min_lot")},
        "by_symbol": per_symbol,
        "guards": {"not_penalised": stats([t for t in all_trades if not t["guard_points"]]),
                   "penalised": stats([t for t in all_trades if t["guard_points"]])},
        "verdict": verdict(summary, out_of_sample),
        "not_simulated": ["AI confirm/veto", "USD-direction guard", "news guard", "per-currency and daily-loss caps"],
        "equity_curve": sizing["equity_curve"][-400:],
        "trades": all_trades[-300:],
    }
    config.BACKTEST_DIR.mkdir(exist_ok=True)
    path = config.BACKTEST_DIR / f"scalp-{started:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["file"] = path.name
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
    parser = argparse.ArgumentParser(description="Backtest the M5 pullback scalper on MT5 history.")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--risk", type=float, default=None, help="risk %% per trade (default: DEFAULT_RISK_PERCENT)")
    parser.add_argument("--balance", type=float, default=None, help="start balance (default: the account equity)")
    parser.add_argument("--extra-spread", type=float, default=0.0,
                        help="add this many pips to every recorded spread (stress test for a live account's costs)")
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
                              extra_spread_points=args.extra_spread)
    finally:
        data_engine.shutdown_mt5()
    s, a = report["summary"], report["account"]
    print(json.dumps({"period": report["period"], "summary": s, "out_of_sample": report["out_of_sample"],
                      "account": a, "guards": report["guards"], "by_symbol": report["by_symbol"]}, indent=2))
    print("VERDICT:", report["verdict"], "| saved", report["file"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
