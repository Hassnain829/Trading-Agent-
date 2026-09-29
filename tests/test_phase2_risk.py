"""Phase 2: validation maths, exit variants, costs, kill-switch/throttle sizing, USD day moves, live kill-switch."""
import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

tmp = Path(tempfile.mkdtemp())
for attr, name in (("MEMORY_FILE", "memory.json"), ("RULES_FILE", "new_rules.json"), ("SETTINGS_FILE", "settings.json"),
                   ("RISK_STATE_FILE", "risk_state.json"), ("SHADOW_FILE", "shadow.json"), ("NEWS_CACHE_FILE", "news.json"),
                   ("AUDIT_LOCK_FILE", ".audit.lock"), ("LOCK_FILE", ".server.lock"), ("BACKTEST_DIR", "backtests"),
                   ("JOURNAL_DIR", "journal")):
    setattr(config, attr, tmp / name)
config.DEEPSEEK_API_KEY = "test"
config.MAX_TOTAL_DRAWDOWN_PERCENT, config.DRAWDOWN_THROTTLE_PERCENT, config.KILL_SWITCH_CLOSE_POSITIONS = 10.0, 5.0, True
config.MAX_DAILY_LOSS_PERCENT = config.MAX_DAILY_PROFIT_PERCENT = 0.0  # keep the daily limits out of the way

import backtest
import data_engine
import execution
import main
import validation

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ============================================================ 1. significance, Monte Carlo, months
rs = [1.5, -1.0, 0.4, -1.0, 1.5, 1.5, -1.0, 0.2, 1.5, -0.3] * 6
sig = validation.significance(rs, trials=1)
mean, sd = np.mean(rs), np.std(rs, ddof=1)
check("t-stat = mean / (sd / sqrt n)", abs(sig["t_stat"] - round(mean / (sd / math.sqrt(len(rs))), 2)) < 1e-9, sig["t_stat"])
check("One trial: DSR equals PSR", sig["dsr"] == sig["psr"] and sig["sharpe_hurdle"] == 0.0, sig)
many = validation.significance(rs, trials=200)
check("More trials raise the bar: DSR falls, t hurdle rises", many["dsr"] < sig["dsr"]
      and many["t_hurdle_for_trials"] > sig["t_hurdle_for_trials"], (many["dsr"], sig["dsr"]))
check("SQN uses at most 100 trades", sig["sqn"] == round(math.sqrt(60) * mean / sd, 2))
check("Too few trades: no statistics", "note" in validation.significance([1, -1, 1]))
mc = validation.monte_carlo(rs, 2.0, runs=500)
check("Monte Carlo: drawdown percentiles ordered, probabilities in [0,1]",
      0 < mc["median_max_drawdown_pct"] <= mc["p95_max_drawdown_pct"] and 0 <= mc["p_loss"] <= 1
      and 0 <= mc["p_hit_kill_switch"] <= 1, mc)
check("Monte Carlo is reproducible (seeded)", validation.monte_carlo(rs, 2.0, runs=500) == mc)
trades = [{"entry_time": f"2026-0{m}-1{d}T10:00:00+00:00", "r": r} for m, r in ((1, 1.0), (2, -1.0), (3, 0.5))
          for d in range(3)]
monthly = validation.monthly(trades)
check("Monthly: 2 of 3 months positive, worst Feb", monthly["positive_share"] == round(2 / 3, 3)
      and monthly["worst_month"]["month"] == "2026-02", monthly)
check("SQN labels", validation.sqn_label(2.54) == "good" and validation.sqn_label(1.0) == "poor")
check("Verdict: proven only with DSR >= 0.95, stable months, positive OOS and +1 pip",
      validation.verdict({"trades": 100, "expectancy_r": 0.3}, {"expectancy_r": 0.2},
                         {"significance": {"dsr": 0.97, "trials": 24}, "monthly": {"positive_share": 0.7}},
                         {"expectancy_r": 0.1}).startswith("Edge looks real")
      and "loses with +1 pip" in validation.verdict({"trades": 100, "expectancy_r": 0.3}, {"expectancy_r": 0.2},
                                                    {"significance": {"dsr": 0.97, "trials": 24},
                                                     "monthly": {"positive_share": 0.7}}, {"expectancy_r": -0.05}))

# ============================================================ 2. walk-forward
t = lambda month, r: {"entry_time": f"2026-{month:02d}-15T10:00:00+00:00", "r": r}
variants = {"A": [t(1, 1.0), t(2, 1.0), t(3, -1.0), t(4, -1.0)], "B": [t(m, 0.2) for m in (1, 2, 3, 4)]}
wf = validation.walk_forward(variants, 2, 1, baseline="A", min_trades=1)
check("Walk-forward picks on the past, scores on the next month",
      [w["pick"] for w in wf["windows"]] == ["A", "B"] and wf["oos"]["net_r"] == -0.8 and wf["picks"] == {"A": 1, "B": 1}, wf)
check("Walk-forward: baseline over the same test months", wf["baseline_same_windows"]["net_r"] == -2.0,
      wf["baseline_same_windows"])
gap = validation.walk_forward({"A": [t(1, 1.0), t(5, 1.0)]}, 2, 1, min_trades=1)
check("Months without trades still count as windows", gap["windows_count"] == 3, gap["windows"])
check("Window plan: 12->3 on years, 3->1 on months",
      validation.window_plan([{"entry_time": "2021-01-01"}, {"entry_time": "2025-12-01"}]) == (12, 3)
      and validation.window_plan([{"entry_time": "2026-03-01"}, {"entry_time": "2026-09-01"}]) == (3, 1))

# ============================================================ 3. exit variants and costs
def path(highs, lows, closes):
    return {"high": np.array(highs), "low": np.array(lows), "close": np.array(closes)}


zero = np.zeros(3)
ENTRY, SL, TP = 1.0, 0.0010, 0.0015
be_path = path([1.0011, 1.0003], [0.9995, 0.9999], [1.0005, 0.9999])  # +1R, then back below entry
results = {mode: backtest._exit(be_path, zero, "BUY", ENTRY, SL, TP, 0, 1, mode, 0.0) for mode in backtest.EXIT_MODES}
check("FIXED: no stop move -> time exit at the close (-0.1R)", results["FIXED"][1] == "TIME"
      and abs(results["FIXED"][0] + 0.1) < 1e-6, results["FIXED"])
check("BREAKEVEN: stop moved to entry after +1R -> 0R", results["BREAKEVEN"][1] == "BE" and abs(results["BREAKEVEN"][0]) < 1e-9,
      results["BREAKEVEN"])
check("PARTIAL: half banked at +1R, rest at entry -> +0.5R", results["PARTIAL"][1] == "BE"
      and abs(results["PARTIAL"][0] - 0.5) < 1e-9, results["PARTIAL"])
check("TRAIL: stop 1R behind the best price -> +0.1R", results["TRAIL"][1] == "TRAIL"
      and abs(results["TRAIL"][0] - 0.1) < 1e-6, results["TRAIL"])
tp_path = path([1.0016], [0.9995], [1.0010])
check("FIXED TP = +1.5R; PARTIAL TP = 0.5 x 1 + 0.5 x 1.5",
      abs(backtest._exit(tp_path, zero, "BUY", ENTRY, SL, TP, 0, 0, "FIXED", 0.0)[0] - 1.5) < 1e-9
      and abs(backtest._exit(tp_path, zero, "BUY", ENTRY, SL, TP, 0, 0, "PARTIAL", 0.0)[0] - 1.25) < 1e-9)
check("Stop checked before target in the same bar (conservative)",
      backtest._exit(path([1.0016], [0.9989], [1.0]), zero, "BUY", ENTRY, SL, TP, 0, 0, "FIXED", 0.0)[1] == "SL")
sl = backtest._exit(path([1.0002], [0.9989], [0.9990]), zero, "BUY", ENTRY, SL, TP, 0, 0, "FIXED", 0.0001)
check("Slippage makes a stop exit worse (-1.1R)", sl[1] == "SL" and abs(sl[0] + 1.1) < 1e-6, sl)
sell = backtest._exit(path([1.0005], [0.9984], [0.9990]), np.full(1, 0.0001), "SELL", ENTRY, SL, TP, 0, 0, "FIXED", 0.0)
check("SELL exits at the ask: the low must reach target - spread", sell[1] == "TP", sell)

# ============================================================ 4. sizing: throttle and kill-switch
spec = {"EURUSD": {"tick_size": 0.00001, "tick_value": 1.0, "volume_min": 0.01, "volume_step": 0.01}}
losers = [{"symbol": "EURUSD", "entry_time": f"2026-09-{d:02d}T08:00", "exit_time": f"2026-09-{d:02d}T09:00",
           "r": -1.0, "sl_distance": 0.0010} for d in range(1, 25)]
protected = backtest.size_trades([dict(x) for x in losers], spec, 10_000.0, 1.0, 10.0, 5.0)
bare = backtest.size_trades([dict(x) for x in losers], spec, 10_000.0, 1.0)
check("Throttle halves risk past -5%; kill-switch stops trading at -10%",
      protected["throttled_trades"] > 0 and protected["kill_switch_at"] is not None
      and protected["skipped_after_kill_switch"] > 0 and 10.0 <= protected["max_drawdown_pct"] < 11.0, protected)
check("Without protection the drawdown keeps going", bare["max_drawdown_pct"] > 20 and bare["kill_switch_at"] is None,
      bare["max_drawdown_pct"])

# ============================================================ 5. USD day moves for the backtest guard
day0 = 20_000 * 86400
def m5(opens_closes):
    times = [day0 + k * 300 for k in range(len(opens_closes))]
    return {"M5": pd.DataFrame({"time": times, "open": [o for o, _ in opens_closes], "close": [c for _, c in opens_closes]})}


market_at = backtest.day_moves({"EURUSD": m5([(1.0, 1.0), (1.0, 0.99)]), "USDJPY": m5([(100.0, 100.0), (100.0, 101.0)]),
                                "GBPUSD": m5([(1.0, 1.0), (1.0, 0.995)])})
seen = market_at("EURUSD", day0 + 600)
check("Day moves: own change + the other pairs, as of that bar", abs(seen["day_change_pct"] + 1.0) < 1e-9
      and abs(seen["correlated_prices"]["USDJPY"]["day_change_pct"] - 1.0) < 1e-9 and "EURUSD" not in seen["correlated_prices"],
      seen)
check("Day moves: nothing known before the bar closed", market_at("EURUSD", day0 + 299)["day_change_pct"] is None)
usd = main.ai_brain.usd_direction(seen)
check("USD direction from the backtest context", usd["label"] == "STRENGTHENING" and usd["pairs"] == 3, usd)

# ============================================================ 6. live kill-switch and throttle
main.bot_state.update(circuit_breaker=False, profit_target_hit=False)
main._update_daily_risk(1000.0, 9)
main._update_daily_risk(1100.0, 9)
check("Peak follows new equity highs", main.bot_state["peak_equity"] == 1100.0 and not main.bot_state["risk_throttled"])
main._update_daily_risk(1040.0, 9)
check("-5.5% from the peak: half risk, no kill-switch", main.bot_state["risk_throttled"] and not main.bot_state["kill_switch"],
      main.bot_state["total_drawdown_pct"])
main._update_daily_risk(985.0, 9)
check("-10.5% from the peak: kill-switch trips", main.bot_state["kill_switch"] and main.bot_state["kill_switch_at"])
main.bot_state.update(guard_login=None, peak_equity=None, kill_switch=False, kill_switch_at=None, day_key=None,
                      day_login=None, day_start_equity=None)  # "restart"
main._peak_saved.update(login=None, peak=None)
main._update_daily_risk(1000.0, 9)
check("After a restart: kill-switch and peak restored", main.bot_state["kill_switch"] and main.bot_state["peak_equity"] == 1100.0)
real_key = data_engine.trading_day_key
data_engine.trading_day_key = lambda now=None: "2099-01-01"
main._update_daily_risk(1000.0, 9)
check("A new trading day does not clear the kill-switch", main.bot_state["kill_switch"] and main.bot_state["peak_equity"] == 1100.0)
data_engine.trading_day_key = real_key
main._update_daily_risk(500.0, 10)
check("Another login has its own peak", main.bot_state["peak_equity"] == 500.0 and not main.bot_state["kill_switch"])
saved = json.loads(config.RISK_STATE_FILE.read_text())
check("Saved per login next to the daily state", set(saved) == {"9", "10"} and saved["9"]["kill_switch"] is True
      and saved["9"]["peak_equity"] == 1100.0 and saved["10"]["peak_equity"] == 500.0, saved)
main._update_daily_risk(985.0, 9)

entry = {}
main.bot_state["is_running"] = True
asyncio.run(main._act_on_decision("EURUSD", {}, {"signal": "BUY", "strategy": "SCALP"}, entry))
check("Kill-switch blocks new entries", entry.get("action") == "blocked: drawdown kill-switch", entry)

closed = []
data_engine.get_account_snapshot = lambda: {"login": 9, "equity": 985.0, "balance": 985.0, "profit": 0.0}
execution.get_open_positions = lambda symbol=None: [{"ticket": 1, "symbol": "EURUSD", "side": "BUY", "profit": -5.0, "managed": True},
                                                    {"ticket": 2, "symbol": "USDJPY", "side": "SELL", "profit": 1.0, "managed": False}]
execution.close_position = lambda ticket: closed.append(ticket) or {"success": True, "ticket": ticket}
main._record_close_execution = lambda result, reason: closed.append(reason)
main._scalp_time_stops = lambda: None
main._watchdog_tick(False)
check("Kill-switch closes the engine's positions only", closed == [1, "KILL_SWITCH"] and main.bot_state["kill_switch_flattened"],
      closed)
main._watchdog_tick(False)
check("...once", closed == [1, "KILL_SWITCH"])

main.bot_state["equity"] = 985.0
main._reset_kill_switch()
saved = json.loads(config.RISK_STATE_FILE.read_text())
check("Reset: new peak at current equity, kill-switch cleared and saved",
      not main.bot_state["kill_switch"] and main.bot_state["peak_equity"] == 985.0
      and saved["9"]["kill_switch"] is False and saved["9"]["peak_equity"] == 985.0, saved["9"])
entry = {}
execution.get_open_positions = lambda symbol=None: []
config.MAX_OPEN_POSITIONS = 0
try:
    asyncio.run(main._act_on_decision("EURUSD", {}, {"signal": "BUY", "strategy": "SCALP"}, entry))
except Exception:
    pass  # past the kill-switch the fake decision is incomplete; only the gate matters here
check("After the reset entries are no longer blocked by it", entry.get("action") != "blocked: drawdown kill-switch", entry)
config.MAX_TOTAL_DRAWDOWN_PERCENT = config.DRAWDOWN_THROTTLE_PERCENT = 0.0
main._update_daily_risk(100.0, 9)
check("0 turns the protection off", not main.bot_state["kill_switch"] and not main.bot_state["risk_throttled"])

# ============================================================ 7. incomplete downloads are refused, not tested
def bars(n, step, gap_at=None):
    t = day0 + step * np.arange(n)
    if gap_at is not None:
        t[gap_at:] += 10 * 86400  # a 10-day hole
    return pd.DataFrame({"time": t.astype("int64"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "spread": 3})


def raises(fn):
    try:
        fn()
    except Exception as exc:
        return str(exc)
    return ""


short = {"M5": bars(500, 300), "D1": bars(46, 86400)}
check("46 daily bars (scattered download) -> refused: EMA200 needs 210",
      "EMA200" in raises(lambda: backtest.check_coverage("EURUSD", short, 1250)))
holey = {"M5": bars(5000, 300, gap_at=2500), "D1": bars(400, 86400)}
check("A 10-day hole -> refused as an incomplete download",
      "holes" in raises(lambda: backtest.check_coverage("EURUSD", holey, 300)))
check("Enough clean history: tradable days = daily bars - warm-up",
      backtest.check_coverage("EURUSD", {"M5": bars(5000, 300), "D1": bars(400, 86400)}, 1250) == 400 - backtest.D1_WARMUP)
backtest.load_history = lambda symbol, days, source="mt5": (_ for _ in ()).throw(FileNotFoundError(f"no bars for {symbol}"))
before = list(config.BACKTEST_DIR.glob("*.json")) if config.BACKTEST_DIR.exists() else []
message = raises(lambda: backtest.run_backtest(["GBPUSD", "USDJPY"], 1250, 2.0, 100.0, source="dukascopy"))
after = list(config.BACKTEST_DIR.glob("*.json")) if config.BACKTEST_DIR.exists() else []
check("No usable symbol -> clear error and NO empty report saved", "no usable dukascopy history" in message
      and "GBPUSD" in message and before == after, message)

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
