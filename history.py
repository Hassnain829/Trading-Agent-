"""
History replay: the live scalp rules over past MT5 bars, so the learning agent pre-trains on months of
markets (trends, ranges, news weeks) instead of only the days it has run live.

    replay      every closed M5 bar of the last --months is judged by scalper.evaluate() (the live rules with
                the live .env/dashboard settings) inside the live entry window. A setup, or one the
                overextension guard blocked, is followed on the next M5 bars: stop, target or time stop, the
                spread paid, a bar touching both stop and target counts as a loss. One trade per pair at a
                time and at most SCALP_MAX_TRADES_PER_SYMBOL per day, like the live engine.
    costs       MT5 history keeps the lowest spread of each bar (often 0), so every trade pays at least the
                pair's typical live spread (75th percentile from the decision journal in data/journal), and
                at least --min-cost-pips.
    output      data/agent/history.jsonl, one reward per line (source "history"). The agent learns from it
                with weight AGENT_HISTORY_WEIGHT; live rewards always count fully.
    report      walk-forward: the agent is fitted on the older months only and judged on the last
                --test-months it never saw, next to the plain rules; plus breakdowns by hour, trend
                strength, pair, side and spread. Saved to data/history/report-<time>.json.

Usage:  .venv\\Scripts\\python.exe history.py [--months 12] [--test-months 3] [--symbols EURUSD,GBPUSD]
        [--min-cost-pips 0.5] [--no-save]
Safe while the bot is running (read-only MT5 access). Restart the bot afterwards so the agent reloads.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import config
import data_engine
import dip
import scalper
import settings_store
from ml import agent

MAX_BARS = 99_000  # copy_rates_from_pos refuses start 1 + count above the terminal's "max bars" (100000)
HISTORY_FILE = config.AGENT_DIR / "history.jsonl"
REPORT_DIR = config.BASE_DIR / "data" / "history"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def fetch_frames(symbol: str, months: int) -> Dict[str, pd.DataFrame]:
    """M5 for the replay window plus enough M15/H1/D1 before it for every indicator to warm up."""
    m5 = min(MAX_BARS, months * 23 * 288 + 1_000)
    counts = {"M5": m5, "M15": min(MAX_BARS, m5 // 3 + 400), "H1": min(MAX_BARS, m5 // 12 + 400),
              "D1": min(MAX_BARS, months * 31 + 320)}
    return {tf: data_engine.fetch_bars(symbol, scalper.TIMEFRAMES[tf], count) for tf, count in counts.items()}


def typical_spreads(symbols: List[str]) -> Dict[str, float]:
    """75th-percentile live spread (in price) per pair during the entry window, from the decision journal."""
    points: Dict[str, List[float]] = defaultdict(list)
    folder = config.JOURNAL_DIR
    for path in sorted(folder.glob("*.jsonl")) if folder.exists() else []:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                symbol, spread = row.get("symbol"), row.get("spread_points")
                if symbol in symbols and spread is not None and 6 <= int(str(row.get("ts", "T00"))[11:13] or 0) < 17:
                    points[symbol].append(float(spread))
    out = {}
    for symbol in symbols:
        info = mt5.symbol_info(symbol)
        point = float(info.point) if info else 0.0
        out[symbol] = float(np.percentile(points[symbol], 75)) * point if points[symbol] else 0.0
    return out


def pip_size(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.0
    return float(info.point) * (10 if info.digits in (3, 5) else 1)


# -----------------------------------------------------------------------------
# Replay
# -----------------------------------------------------------------------------
def _simulate(side: str, entry: float, sl: float, tp: float, rr: float, spread: float, m5: Dict[str, np.ndarray],
              start: int, time_stop_seconds: float) -> Optional[tuple]:
    """(status, R, exit index) on the M5 bars after ``start``: the shadow-trade rules (shadow_store._outcome)."""
    opened = m5["close_time"][start]
    last = None
    for j in range(start + 1, len(m5["close"])):
        if m5["time"][j] >= opened + time_stop_seconds:
            break
        last = j
        high, low = m5["high"][j], m5["low"][j]
        if side == "BUY":  # a long exits at the bid
            hit_sl, hit_tp = low <= sl, high >= tp
        else:  # a short exits at the ask = bid + spread
            hit_sl, hit_tp = high + spread >= sl, low + spread <= tp
        if hit_sl:
            return "LOSS", -1.0, j
        if hit_tp:
            return "WIN", rr, j
    if last is None or last + 1 >= len(m5["close"]):
        return None  # the time stop is not over yet (end of the data)
    close = m5["close"][last] + (spread if side == "SELL" else 0.0)
    moved = (close - entry) if side == "BUY" else (entry - close)
    return "TIMEOUT", round(moved / abs(entry - sl), 3), last


def replay_symbol(symbol: str, months: int, cost: float,
                  frames: Optional[Dict[str, pd.DataFrame]] = None,
                  prepared: Optional[scalper.Prepared] = None) -> List[Dict[str, Any]]:
    p = prepared or scalper.prepare(frames or fetch_frames(symbol, months))
    m5 = p.a["M5"]
    start_epoch = m5["close_time"][-1] - months * 30.4 * 86400
    utc = data_engine.server_epochs_to_utc(m5["close_time"].astype("int64"))
    in_window = scalper.session_mask(pd.DatetimeIndex(utc))
    utc_iso = [None if pd.isna(t) else t.isoformat(timespec="seconds") for t in utc]
    point = float(mt5.symbol_info(symbol).point)
    time_stop = config.SCALP_TIME_STOP_MINUTES * 60
    rows: List[Dict[str, Any]] = []
    # One followed trade per pair at a time: a rules trade holds the pair; a guard-blocked one is followed
    # separately (like the live shadow), so it never stops a later rules trade.
    busy = {"rules": -1, "guard": -1}
    per_day = defaultdict(int)
    for i in range(60, len(m5["close"]) - 1):
        if m5["close_time"][i] < start_epoch or i <= busy["rules"] or not in_window[i] or utc_iso[i] is None:
            continue
        day = utc_iso[i][:10]
        if per_day[day] >= config.SCALP_MAX_TRADES_PER_SYMBOL:
            continue
        spread = max(float(m5["spread"][i]) * point, cost)
        result = scalper.evaluate(p, i, spread_price=spread)
        setup, guards = result.get("setup"), []
        if setup is None and result.get("blocked_setup"):
            setup, guards = result["blocked_setup"], [g["id"] for g in result.get("guards") or []]
        if setup is None and config.SCALP_REVERSION:
            setup = scalper.evaluate_reversion(p, i, spread_price=spread).get("setup")
        if setup is None or (guards and i <= busy["guard"]):
            continue
        if spread > config.MAX_SPREAD_TO_STOP * setup["sl_distance"]:
            continue  # the live engine refuses it for its cost (and does not follow it either)
        side, entry = setup["side"], float(setup["entry_ref"])
        direction = 1.0 if side == "BUY" else -1.0
        sl, tp = entry - direction * setup["sl_distance"], entry + direction * setup["tp_distance"]
        outcome = _simulate(side, entry, sl, tp, float(setup["risk_reward"]), spread, m5, i, time_stop)
        if outcome is None:
            continue
        status, reward, exit_index = outcome
        features = scalper.features(p, i, side=side, spread_price=spread)
        rows.append({
            "id": f"history:{symbol}:{int(m5['time'][i])}", "source": "history", "strategy": "SCALP",
            "symbol": symbol, "side": side, "opened_at": utc_iso[i], "resolved_at": utc_iso[exit_index],
            "reward": float(reward), "outcome": status, "exit": status, "action": "history",
            "decided_by": "guard" if guards else "rules", "blocked_by": guards, "account_mode": "HISTORY",
            "context": agent.make_context("SCALP", features, side, setup),
        })
        busy["guard" if guards else "rules"] = exit_index
        if not guards:
            per_day[day] += 1
    return rows


def replay_dip(symbol: str, months: int, cost: float,
               prepared: Optional[scalper.Prepared] = None) -> List[Dict[str, Any]]:
    """The H1 dip rules (dip.py) over the last ``months``: one trade per pair at a time, the EMA5 exit."""
    p = dip.add_indicators(prepared or scalper.prepare(fetch_frames(symbol, months)))
    h1 = p.a["H1"]
    start_epoch = h1["close_time"][-1] - months * 30.4 * 86400
    utc = data_engine.server_epochs_to_utc(h1["close_time"].astype("int64"))
    in_window = scalper.session_mask(pd.DatetimeIndex(utc))
    utc_iso = [None if pd.isna(t) else t.isoformat(timespec="seconds") for t in utc]
    point = float(mt5.symbol_info(symbol).point)
    rows: List[Dict[str, Any]] = []
    busy, per_day = -1, defaultdict(int)
    for h in range(210, len(h1["close"]) - 1):
        if h1["close_time"][h] < start_epoch or h <= busy or not in_window[h] or utc_iso[h] is None:
            continue
        day = utc_iso[h][:10]
        if per_day[day] >= config.DIP_MAX_TRADES_PER_SYMBOL:
            continue
        spread = max(float(h1["spread"][h]) * point, cost)
        setup = dip.evaluate(p, h, spread_price=spread).get("setup")
        if setup is None or spread > config.MAX_SPREAD_TO_STOP * setup["sl_distance"]:
            continue
        side, entry = setup["side"], float(setup["entry_ref"])
        direction = 1.0 if side == "BUY" else -1.0
        outcome = dip.exit_outcome(h1["time"].astype(float), h1["high"], h1["low"], h1["close"], h1["ema5"],
                                   float(h1["close_time"][h]), side, entry, entry - direction * setup["sl_distance"],
                                   entry + direction * setup["tp_distance"], spread, config.DIP_TIME_STOP_MINUTES * 60)
        if outcome is None:
            continue
        status, reward, exit_index = outcome
        features = dip.features(p, h, side=side, spread_price=spread)
        rows.append({
            "id": f"history:DIP:{symbol}:{int(h1['time'][h])}", "source": "history", "strategy": "DIP",
            "symbol": symbol, "side": side, "opened_at": utc_iso[h], "resolved_at": utc_iso[exit_index],
            "reward": float(reward), "outcome": status, "exit": status, "action": "history", "decided_by": "rules",
            "blocked_by": [], "account_mode": "HISTORY",
            "context": agent.make_context("DIP", features, side, setup),
        })
        busy = exit_index
        per_day[day] += 1
    return rows


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
def stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    rows = sorted(rows, key=lambda r: r["opened_at"])
    r = np.array([row["reward"] for row in rows], dtype=float)
    curve = np.cumsum(r)
    drawdown = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:] - curve)) if n else 0.0
    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    gains, losses = float(r[r > 0].sum()), float(-r[r < 0].sum())
    months: Dict[str, float] = defaultdict(float)
    for row in rows:
        months[row["opened_at"][:7]] += row["reward"]
    return {"n": n, "win_rate": round(float(np.mean([row["outcome"] == "WIN" for row in rows])), 3),
            "avg_r": round(float(r.mean()), 3), "total_r": round(float(r.sum()), 1),
            "t_stat": round(float(r.mean() / (sd / math.sqrt(n))), 2) if sd else 0.0,
            "profit_factor": round(gains / losses, 2) if losses else None, "max_drawdown_r": round(drawdown, 1),
            "months_positive": f"{sum(v > 0 for v in months.values())}/{len(months)}",
            "by_month": {k: round(v, 1) for k, v in sorted(months.items())}}


def _bucket(value: Optional[float], edges: List[float]) -> str:
    if value is None:
        return "n/a"
    for edge in edges:
        if value < edge:
            return f"<{edge:g}"
    return f">={edges[-1]:g}"


def breakdowns(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    keys = {
        "utc_hour": lambda r: f"{int(r['opened_at'][11:13]):02d}h",
        "d1_adx": lambda r: _bucket(r["context"]["market"].get("d1_adx"), [15, 20, 25, 35]),
        "spread_atr": lambda r: _bucket(r["context"]["market"].get("spread_atr"), [0.05, 0.1, 0.2]),
        "symbol": lambda r: r["symbol"],
        "side": lambda r: r["side"],
        "guard": lambda r: ",".join(r["blocked_by"]) or "none",
    }
    out = {}
    for name, key in keys.items():
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[key(row)].append(row)
        out[name] = {k: {f: v for f, v in stats(g).items() if f != "by_month"} for k, g in sorted(groups.items())}
    return out


def walk_forward(rows: List[Dict[str, Any]], test_months: int, strategy: str = "SCALP") -> Dict[str, Any]:
    """Fit the agent on everything before the cutoff, judge it on the last ``test_months`` it never saw."""
    last = max(row["opened_at"] for row in rows)
    cutoff = datetime.fromisoformat(last) - timedelta(days=test_months * 30.4)
    cut = cutoff.isoformat(timespec="seconds")
    train = [row for row in rows if row["resolved_at"] < cut]
    test = [row for row in rows if row["opened_at"] >= cut]
    model = agent.fit(strategy, rows=train, now=cutoff)
    for row in test:
        row["_expected_r"] = (agent.predict(strategy, row["context"], model) or {}).get("expected_r", 0.0)
    rules = [row for row in test if not row["blocked_by"]]
    picks = [row for row in rules if row["_expected_r"] > 0]
    out = {"cutoff": cut, "train_rewards": len(train), "test_rewards": len(test),
           "rules_take_all": stats(rules), "agent_picks": stats(picks),
           "agent_skips": stats([row for row in rules if row["_expected_r"] <= 0]),
           "guard_blocked": stats([row for row in test if row["blocked_by"]]),
           "top_effects": agent._effects(model, top=8)}
    for row in test:
        row.pop("_expected_r", None)
    p = out["agent_picks"]
    checks = {
        "picks >= 100 in the test months": p.get("n", 0) >= 100,
        "picks average >= +0.05R after costs": p.get("avg_r", -1) >= 0.05,
        "picks t-stat >= 2 (unlikely to be luck)": p.get("t_stat", 0) >= 2,
        "most test months positive": _majority(p.get("months_positive", "0/1")),
        "picks beat taking every setup": p.get("avg_r", -1) > out["rules_take_all"].get("avg_r", 0),
    }
    out["readiness"] = {"checks": checks, "ready_for_demo": all(checks.values())}
    return out


def _majority(text: str) -> bool:
    good, total = (int(x) for x in text.split("/"))
    return total > 0 and good * 3 >= total * 2


def _print_stats(label: str, s: Dict[str, Any]) -> None:
    if not s.get("n"):
        print(f"  {label:28s} n=0")
        return
    print(f"  {label:28s} n={s['n']:5d}  win {s['win_rate']:.0%}  avg {s['avg_r']:+.3f}R  total {s['total_r']:+8.1f}R  "
          f"t {s['t_stat']:+.2f}  PF {s['profit_factor']}  maxDD {s['max_drawdown_r']}R  months+ {s['months_positive']}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def save_rows(rows: List[Dict[str, Any]], strategy: str) -> int:
    """Replace ``strategy``'s rows in the history file, keeping the other strategies' rows. Returns the total."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    kept = [row for row in agent.load_history() if row.get("strategy") != strategy] if HISTORY_FILE.exists() else []
    tmp = HISTORY_FILE.with_name(HISTORY_FILE.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in kept + rows:
            handle.write(json.dumps(row, default=str) + "\n")
    os.replace(tmp, HISTORY_FILE)
    return len(kept) + len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--min-cost-pips", type=float, default=0.5)
    parser.add_argument("--strategy", choices=("SCALP", "DIP"), default="SCALP")
    parser.add_argument("--no-save", action="store_true", help="report only; keep data/agent/history.jsonl as it is")
    args = parser.parse_args()
    replay = replay_dip if args.strategy == "DIP" else replay_symbol

    settings_store.load_saved()  # the dashboard's settings (pair list, scalp options) override .env, like live
    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or list(config.SYMBOLS_DEMO)
        for symbol in symbols:
            mt5.symbol_select(symbol, True)
        data_engine.server_utc_offset_seconds(symbols)
        spreads = typical_spreads(symbols)
        rows: List[Dict[str, Any]] = []
        for symbol in symbols:
            cost = max(spreads.get(symbol, 0.0), args.min_cost_pips * pip_size(symbol))
            try:
                found = replay(symbol, args.months, cost)
            except data_engine.DataEngineError as exc:
                print(f"{symbol:8s} skipped: {exc}")
                continue
            rows.extend(found)
            s = stats([r for r in found if not r["blocked_by"]])
            print(f"{symbol:8s} cost {cost / pip_size(symbol):.1f} pips | {len(found):4d} setups | rules "
                  f"avg {s.get('avg_r', 0):+.3f}R total {s.get('total_r', 0):+.1f}R")
    finally:
        mt5.shutdown()
    if not rows:
        raise SystemExit("no setups found in the history")

    rows.sort(key=lambda r: r["opened_at"])
    rules = [r for r in rows if not r["blocked_by"]]
    print(f"\n{args.strategy} REPLAY {args.months} months, {len(symbols)} pairs, window {rows[0]['opened_at'][:10]} -> "
          f"{rows[-1]['opened_at'][:10]}")
    _print_stats("rules (what live would take)", stats(rules))
    _print_stats("guard-blocked setups", stats([r for r in rows if r["blocked_by"]]))
    wf = walk_forward(rows, args.test_months, args.strategy)
    print(f"\nWALK-FORWARD: agent trained on {wf['train_rewards']} rewards before {wf['cutoff'][:10]}, "
          f"tested on the {wf['test_rewards']} after it")
    for label, key in (("rules: take every setup", "rules_take_all"), ("agent picks", "agent_picks"),
                       ("agent skips", "agent_skips"), ("guard-blocked", "guard_blocked")):
        _print_stats(label, wf[key])
    print("\nREADINESS (agent picks in the unseen months):")
    for check, ok in wf["readiness"]["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {check}")
    print(f"  => {'READY for demo' if wf['readiness']['ready_for_demo'] else 'NOT ready for demo yet'}")

    report = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "months": args.months,
              "strategy": args.strategy, "symbols": symbols, "settings": {k: getattr(config, k) for k in (
                  "SCALP_RSI_PULLBACK", "SCALP_REWARD_RISK", "SCALP_TIME_STOP_MINUTES", "SCALP_REVERSION",
                  "SCALP_MEDIUM_TREND", "SCALP_STRICT_GUARD", "SCALP_MIN_ADX", "TRADE_ALL_HOURS",
                  "SCALP_SESSION_START_LONDON", "SCALP_SESSION_END_NEW_YORK", "DIP_RSI_LOW", "DIP_STOP_ATR",
                  "DIP_TIME_STOP_MINUTES")},
              "rules": stats(rules), "walk_forward": wf, "breakdowns": breakdowns(rows)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"report-{args.strategy.lower()}-{datetime.now():%Y%m%d-%H%M}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    if not args.no_save:
        total = save_rows(rows, args.strategy)
        print(f"Saved {len(rows)} {args.strategy} history rewards ({total} in all) to {HISTORY_FILE} "
              f"(restart the bot so the agent loads them)")


if __name__ == "__main__":
    main()
