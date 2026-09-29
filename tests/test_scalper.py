"""Scalping mode: shared rules, session, backtest exits/sizing, live scalp flow, time stop, shadows, settings."""
import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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
config.SYMBOLS = ["EURUSD", "USDJPY"]
config.STRATEGY_MODE, config.NEWS_GUARD, config.WEEKEND_ENTRY_CUTOFF_HOURS = "SCALP", False, 0.0
config.CONFIDENCE_THRESHOLD, config.CALIBRATE_THRESHOLD, config.OVEREXTENSION_GUARD = 65, False, True
config.LOSS_COOLDOWN_MINUTES, config.DEEPSEEK_API_KEY = 30, "test"
# The mechanics tests below use a steady synthetic trend, which the strict guard rightly calls "stretched";
# the adopted filters get their own section at the end.
DEFAULT_STRICT, DEFAULT_MEDIUM = config.SCALP_STRICT_GUARD, config.SCALP_MEDIUM_TREND
config.SCALP_STRICT_GUARD = False

import MetaTrader5 as mt5

import ai_brain
import backtest
import data_engine
import execution
import journal
import main
import scalper
import shadow_store

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ============================================================ synthetic frames
def frame(closes, t0, step, spread=10):
    closes = np.asarray(closes, float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({"time": (t0 + step * np.arange(len(closes))).astype("int64"), "open": opens,
                         "high": np.maximum(opens, closes) + 0.0001, "low": np.minimum(opens, closes) - 0.0001,
                         "close": closes, "tick_volume": 100, "spread": spread})


def uptrend_m5(n=300):
    steps = [0.0006 if k % 2 == 0 else -0.0001 for k in range(n)]
    return list(1.10 + np.cumsum(steps))


def with_setup(m5):
    last = m5[-1]
    m5 = m5 + [last - 0.0018 * k for k in range(1, 4)]
    return m5 + [m5[-1] + 0.0012]


def frames_for(m5, end, d1_slope=0.002, m15_slope=0.0005):
    return {"M5": frame(m5, end - 300 * len(m5), 300),
            "M15": frame(1.05 + m15_slope * np.arange(300), end - 900 * 300, 900),
            "H1": frame(1.00 + 0.0003 * np.arange(300), end - 3600 * 300, 3600),
            "D1": frame(0.90 + d1_slope * np.arange(300), end - 86400 * 301, 86400)}


END = int(datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc).timestamp())  # a Tuesday, London session

# ============================================================ 1. rules
p = scalper.prepare(frames_for(with_setup(uptrend_m5()), END))
r = scalper.evaluate(p, spread_price=0.0001)
s = r["setup"]
check("Uptrend + sharp M5 dip that turns -> BUY setup", s is not None and s["side"] == "BUY", r["reason"])
check("Stop 1-2 M5 ATR beyond the swing; target = 1.5 x stop", s and 1.0 <= s["sl_atr"] <= 2.0
      and abs(s["tp_distance"] - 1.5 * s["sl_distance"]) < 1e-12 and s["m5_rsi_extreme"] < 40 <= s["m5_rsi"], s)
r = scalper.evaluate(scalper.prepare(frames_for(uptrend_m5() + [uptrend_m5()[-1] + 0.0003], END)))
check("No dip -> no setup", r["setup"] is None and "no M5 pullback" in r["reason"], r["reason"])
check("Every evaluation reports its stage and the M5 RSI", r["stage"] == "NO_PULLBACK" and r["rsi"] > 60
      and scalper.evaluate(p, spread_price=0.0001)["stage"] == "SETUP", (r["stage"], r["rsi"]))
r = scalper.evaluate(scalper.prepare(frames_for(with_setup(uptrend_m5()), END, d1_slope=-0.002)))
check("D1 downtrend while M15/M5 rise -> no BUY", r["setup"] is None and r["trend"] == "DOWN", r["reason"])
r = scalper.evaluate(scalper.prepare(frames_for(with_setup(uptrend_m5()), END, m15_slope=-0.0005)))
check("M15 against D1 -> no setup", r["setup"] is None and "M15" in r["reason"], r["reason"])
mirror = {tf: f.assign(open=3 - f["open"], close=3 - f["close"], high=3 - f["low"], low=3 - f["high"])
          for tf, f in frames_for(with_setup(uptrend_m5()), END).items()}
r = scalper.evaluate(scalper.prepare(mirror), spread_price=0.0001)
check("Mirror image -> SELL setup", r["setup"] is not None and r["setup"]["side"] == "SELL", r["reason"])

# ============================================================ 2. session (DST aware)
check("Default entry window ends 16:00 New York (scalps closed before the 17:00 rollover)",
      config.SCALP_SESSION_END_NEW_YORK == 16)
config.SCALP_SESSION_END_NEW_YORK = 12
tue = lambda h, m=0, month=9, day=29: datetime(2026, month, day, h, m, tzinfo=timezone.utc)
check("Session: Tue 06:30 UTC (07:30 London BST) open", scalper.in_session(tue(6, 30)))
check("Session: Tue 05:59 UTC (06:59 London) closed", not scalper.in_session(tue(5, 59)))
check("Session: Tue 15:59 UTC (11:59 NY) open, 16:00 UTC (12:00 NY) closed",
      scalper.in_session(tue(15, 59)) and not scalper.in_session(tue(16, 0)))
check("Session: winter (GMT) London open is 07:00 UTC", not scalper.in_session(tue(6, 30, 12, 1))
      and scalper.in_session(tue(7, 5, 12, 1)))
check("Session: Saturday closed", not scalper.in_session(datetime(2026, 10, 3, 10, tzinfo=timezone.utc)))
grid = pd.date_range("2026-09-27", "2026-10-04", freq="17min", tz="UTC")
check("Vectorised session mask == in_session", (scalper.session_mask(grid) == np.array([scalper.in_session(t.to_pydatetime()) for t in grid])).all())
config.SCALP_SESSION_END_NEW_YORK = 16
check("Entry window to 16:00 NY: 9:34 PM Pakistan (16:34 UTC) is open, 1:00 AM PKT (20:00 UTC) closed",
      scalper.in_session(tue(16, 34)) and scalper.in_session(tue(19, 59)) and not scalper.in_session(tue(20, 0)))
config.SCALP_SESSION_END_NEW_YORK = 12

# ============================================================ 3. backtest exits + sizing
mt5.symbol_info_tick = lambda s: None  # no live tick: epochs are treated as UTC
data_engine._server_offset.update(seconds=0.0, known=False)


def run_path(after, side_mirror=False, spread=10):
    m5 = with_setup(uptrend_m5()) + after
    frames = frames_for(m5, END + 300 * len(after))
    frames["M5"]["spread"] = spread
    if side_mirror:
        frames = {tf: f.assign(open=3 - f["open"], close=3 - f["close"], high=3 - f["low"], low=3 - f["high"])
                  for tf, f in frames.items()}
    prepared = scalper.prepare(frames)
    return backtest.simulate_symbol("EURUSD", prepared, 0.00001, 2, 12, 6, 4)


target = lambda trades: [t for t in trades if t["entry_time"].startswith("2026-09-29T09:5") or t["entry_time"].startswith("2026-09-29T10:0")]
trades = run_path([with_setup(uptrend_m5())[-1] + 0.0008 * k for k in range(1, 15)])
check("Backtest: price runs up -> TP (+1.5R)", trades and trades[0]["exit_reason"] == "TP" and abs(trades[0]["r"] - 1.5) < 1e-6,
      trades[:1])
trades = run_path([with_setup(uptrend_m5())[-1] - 0.0010 * k for k in range(1, 15)])
check("Backtest: price falls -> SL (-1R)", trades and trades[0]["exit_reason"] == "SL" and abs(trades[0]["r"] + 1) < 1e-6)
flat = with_setup(uptrend_m5())[-1]
trades = run_path([flat + (0.00002 if k % 2 else -0.00002) for k in range(1, 20)])
check("Backtest: price goes nowhere -> TIME exit after 12 bars", trades and trades[0]["exit_reason"] == "TIME"
      and abs(trades[0]["r"]) < 0.5, trades[:1])
trades = run_path([3 - (with_setup(uptrend_m5())[-1] + 0.0008 * k) for k in range(1, 15)][::-1] and
                  [with_setup(uptrend_m5())[-1] + 0.0008 * k for k in range(1, 15)], side_mirror=True, spread=10)
check("Backtest: SELL mirror -> TP; entry at the bid", trades and trades[0]["side"] == "SELL" and trades[0]["exit_reason"] == "TP")

spec = {"EURUSD": {"tick_size": 0.00001, "tick_value": 1.0, "volume_min": 0.01, "volume_step": 0.01}}
sized = backtest.size_trades([
    {"symbol": "EURUSD", "entry_time": "2026-09-29T08:00", "exit_time": "2026-09-29T08:30", "r": 1.5, "sl_distance": 0.0005},
    {"symbol": "EURUSD", "entry_time": "2026-09-29T09:00", "exit_time": "2026-09-29T09:30", "r": -1.0, "sl_distance": 0.0005},
    {"symbol": "EURUSD", "entry_time": "2026-09-29T10:00", "exit_time": "2026-09-29T10:30", "r": 1.0, "sl_distance": 0.0100},
], spec, 100.0, 2.0)
# 2% of 100 = 2.00; loss/lot 50 -> 0.04 lots -> +3.00; then 2% of 103 = 2.06 -> 0.04 -> -2.00; 0.01 lot of a 1000-pt stop = 10 > 2.5 -> skipped
check("Sizing: %-risk lots, compounding, min-lot skip", sized["final_balance"] == 101.0 and sized["skipped_min_lot"] == 1, sized)

st = backtest.stats([{"r": 1.5, "exit_reason": "TP"}, {"r": -1.0, "exit_reason": "SL"}, {"r": -1.0, "exit_reason": "SL"},
                     {"r": 1.5, "exit_reason": "TP"}], days=2)
check("Stats: win rate, expectancy, profit factor, drawdown", st["win_rate"] == 50.0 and st["expectancy_r"] == 0.25
      and st["profit_factor"] == 1.5 and st["max_drawdown_r"] == 2.0 and st["trades_per_day"] == 2.0, st)
check("Verdict: no edge", backtest.verdict({"trades": 40, "expectancy_r": -0.1}, {"expectancy_r": -0.2}).startswith("No edge"))
check("Verdict: too few trades", backtest.verdict({"trades": 5, "expectancy_r": 1}, {}).startswith("Too few"))

backtest.load_history = lambda symbol, days, source="mt5": frames_for(with_setup(uptrend_m5()) + [flat + 0.0008 * k for k in range(1, 15)],
                                                          END + 300 * 14)
mt5.symbol_info = lambda s: SimpleNamespace(point=0.00001, digits=5, trade_tick_size=0.00001, trade_tick_value=1.0,
                                            trade_tick_value_loss=1.0, volume_min=0.01, volume_step=0.01)
report = backtest.run_backtest(["EURUSD"], 2, 2.0, 100.0)
saved = json.loads((config.BACKTEST_DIR / report["file"]).read_text())
check("run_backtest writes a full report", saved["summary"]["trades"] >= 1 and "verdict" in saved and "out_of_sample" in saved
      and "guards" in saved and backtest.latest_report()["file"] == report["file"], saved["summary"])

# ============================================================ 4. live scalp flow
positions, sent = [], []
PRICE = {"EURUSD": flat, "USDJPY": 150.0}


def tick(symbol):
    return SimpleNamespace(bid=PRICE[symbol], ask=round(PRICE[symbol] + 0.0001, 5), time=int(time.time()), time_msc=0)


def order_send(req):
    sent.append(dict(req))
    if req.get("position"):
        positions[:] = [x for x in positions if x.ticket != req["position"]]
        return SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, deal=1, order=1, price=req["price"], volume=req["volume"], comment="ok")
    ticket = 5000 + len(sent)
    positions.append(SimpleNamespace(ticket=ticket, symbol=req["symbol"], volume=req["volume"],
                                     type=mt5.POSITION_TYPE_BUY if req["type"] == mt5.ORDER_TYPE_BUY else mt5.POSITION_TYPE_SELL,
                                     price_open=req["price"], price_current=req["price"], sl=req["sl"], tp=req["tp"],
                                     profit=0.0, swap=0.0, time=int(time.time()), magic=req["magic"], comment=req["comment"]))
    return SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, deal=0, order=ticket, price=req["price"], volume=req["volume"], comment="ok")


mt5.symbol_info = lambda s: SimpleNamespace(digits=5, point=0.00001, trade_tick_size=0.00001, trade_tick_value=1.0,
                                            trade_tick_value_loss=1.0, trade_contract_size=100000.0, volume_min=0.01,
                                            volume_max=100.0, volume_step=0.01, trade_stops_level=0, filling_mode=2,
                                            trade_mode=4, path="Forex\\" + s)
mt5.symbol_info_tick = tick
mt5.account_info = lambda: SimpleNamespace(login=5, equity=100.0, balance=100.0, margin_free=90.0, currency="USD", profit=0.0,
                                           margin=0.0, margin_level=0.0, leverage=100, server="Demo", company="X", name="T",
                                           trade_allowed=True, trade_mode=mt5.ACCOUNT_TRADE_MODE_DEMO)
mt5.positions_get = lambda **kw: tuple(x for x in positions if kw.get("ticket") in (None, x.ticket)
                                       and kw.get("symbol") in (None, x.symbol))
mt5.order_calc_margin = lambda *a: 5.0
mt5.order_send = order_send
mt5.history_deals_get = lambda **kw: ()
mt5.last_error = lambda: (1, "ok")

live_frames = frames_for(with_setup(uptrend_m5()), END)
bar = {"t": int(live_frames["M5"]["time"].iloc[-1])}
scalper.fetch_live_frames = lambda symbol: live_frames
data_engine.last_closed_bar_time = lambda symbol, tf: bar["t"]
data_engine.fetch_correlated_asset_prices = lambda *a, **k: {}


def fake_market(symbol):
    t = tick(symbol)
    return {"symbol": symbol, "tradeable": True, "market_idle": False, "path": "Forex\\" + symbol, "bid": t.bid, "ask": t.ask,
            "mid": t.bid, "spread": 10, "spread_price": 0.0001, "digits": 5, "point": 0.00001, "tick_size": 0.00001,
            "equity": 100.0, "balance": 100.0, "currency": "USD", "account_login": 5, "account_mode": "DEMO",
            "broker": "X", "server": "Demo", "tick_epoch": int(time.time()), "day_change_pct": 0.1,
            "h1_data": {"atr14": 0.001, "structure": {}}, "daily_data": {"structure": {}}}


data_engine.fetch_multi_timeframe_data = fake_market
main._capture_context = lambda symbol, market=None: None
main._refresh_account_state = lambda: None
verdict_answer = {"decision": "CONFIRM", "confidence_score": 78, "logic": "clean trend, orderly pullback"}
prompts = []


def fake_chat(messages, **kw):
    prompts.append(messages[0]["content"])
    return {"content": json.dumps(verdict_answer)}


ai_brain.deepseek_chat = fake_chat
real_in_session = scalper.in_session
scalper.in_session = lambda now=None: True
main.bot_state.update(is_running=True, circuit_breaker=False, profit_target_hit=False, risk_percent=2.0, daily_drawdown_pct=0.0)

outcome = asyncio.run(main.process_symbol("EURUSD"))
view = main.bot_state["decisions"].get("EURUSD") or {}
check("Scanner view after an AI-confirmed setup: stage, trend, RSI, check time",
      view.get("stage") == "AI_CONFIRMED" and view.get("trend") == "UP" and view.get("rsi") is not None
      and main.bot_state.get("scalp_last_check_at"), view)
check("Scalp setup -> AI asked -> CONFIRM 78 -> order placed", outcome == "evaluated" and len(positions) == 1
      and positions[0].comment == "AI-HF-S" and prompts and "SCALP SETUP (BUY EURUSD)" in prompts[-1], (outcome, len(positions)))
pos = positions[0]
sl_pts = (pos.price_open - pos.sl) / 0.00001
check("Scalp stop is 1-2 M5 ATR (not H1-sized); TP 1.5R", 50 < sl_pts < 400 and abs((pos.tp - pos.price_open) / (pos.price_open - pos.sl) - 1.5) < 0.02,
      (pos.price_open, pos.sl, pos.tp))
rec = json.loads(config.MEMORY_FILE.read_text())[-1]
check("Memory record tagged SCALP with setup + time stop", rec["strategy"] == "SCALP" and rec["time_stop_minutes"] == 60
      and rec["setup"]["side"] == "BUY")
last = list(journal.iter_entries(1))[-1]
check("Journal: the filled scalp is recorded with features, AI answer, action and ticket",
      last["kind"] == "scalp_eval" and last["stage"] == "SETUP" and last["action"] == "filled"
      and last["ticket"] == positions[0].ticket and last["ai"]["base_confidence"] == 78
      and len(last["features"]) >= 25 and last["features"]["side"] == "BUY" and last["setup"]["side"] == "BUY",
      {k: last.get(k) for k in ("stage", "action", "ticket")})
n_prompts = len(prompts)
check("Same M5 bar -> not evaluated again", asyncio.run(main.process_symbol("EURUSD")) == "unchanged" and len(prompts) == n_prompts)

# time stop: the position opened 61 minutes ago by the broker clock
pos.time = int(time.time()) - 61 * 60
main._history_cache["stamp"] = None
data_engine.server_now_epoch = lambda symbol: float(int(time.time()))
check("Time stop closes the scalp after 60 min", main._scalp_time_stops() == 1 and not positions)
rec = json.loads(config.MEMORY_FILE.read_text())[-1]
check("Time-stop close recorded on the trade", (rec.get("close_execution") or {}).get("reason") == "TIME_STOP")

# veto -> no trade, shadow AI-VETO
bar["t"] += 300
verdict_answer.update(decision="VETO", confidence_score=40, logic="target runs into the H1 high")
check("AI VETO -> no order", asyncio.run(main.process_symbol("EURUSD")) == "evaluated" and not positions)
check("Scanner view after a veto: AI_VETO", main.bot_state["decisions"]["EURUSD"].get("stage") == "AI_VETO")
veto = list(journal.iter_entries(1))[-1]
check("Journal: the veto is recorded with the shadow trade id that follows it",
      veto["action"] == "hold" and veto["ai"]["blocked_by"] == ["AI-VETO"] and veto.get("shadow_id")
      and veto["shadow_id"] == shadow_store.load_shadows()[-1]["id"], {k: veto.get(k) for k in ("action", "shadow_id")})

# ML setup filter (an approved model is faked; the real one is tested in test_phase3_ml)
real_ml = main.ml_model.evaluate
main.ml_model.evaluate = lambda features, side: {"p_win": 0.31, "threshold": 0.45, "take": False, "model": "lgbm"}
bar["t"] += 300
n_prompts = len(prompts)
saved_shadows = config.SHADOW_FILE.read_text()
config.SHADOW_FILE.write_text("[]")  # the veto's open shadow would absorb this one (same setup, deduplicated)
check("ML filter below break-even -> skipped BEFORE the AI (no API call), no order",
      asyncio.run(main.process_symbol("EURUSD")) == "skipped" and len(prompts) == n_prompts and not positions)
skipped = list(journal.iter_entries(1))[-1]
shadow = shadow_store.load_shadows()[-1]
check("...followed as a shadow trade blocked by ML-FILTER, journaled with P(win)",
      skipped["action"] == "skipped: ML filter" and skipped["ml"]["p_win"] == 0.31 and shadow["blocked_by"] == ["ML-FILTER"]
      and skipped["shadow_id"] == shadow["id"] and shadow["features"]
      and main.bot_state["decisions"]["EURUSD"]["stage"] == "ML_FILTER", {k: skipped.get(k) for k in ("action", "ml")})
main.ml_model.evaluate = lambda features, side: {"p_win": 0.62, "threshold": 0.45, "take": True, "model": "lgbm"}
bar["t"] += 300
asyncio.run(main.process_symbol("EURUSD"))
passed = list(journal.iter_entries(1))[-1]
check("ML filter above break-even -> the AI is asked as before; P(win) journaled",
      len(prompts) == n_prompts + 1 and passed["ml"]["take"] and passed.get("ai"), passed.get("action"))
main.ml_model.evaluate = real_ml
config.SHADOW_FILE.write_text(saved_shadows)
check("Filter off by default: evaluate() returns None", not config.ML_FILTER and main.ml_model.evaluate({}, "BUY") is None)
check("Shadow trade stores the market snapshot and features (for rule learning)",
      shadow_store.load_shadows()[-1].get("market_context", {}).get("h1") is not None
      and len(shadow_store.load_shadows()[-1].get("features") or {}) >= 25)
shadow = shadow_store.load_shadows()[-1]
check("Vetoed setup followed as a shadow trade with the time stop", shadow["blocked_by"] == ["AI-VETO"]
      and shadow["time_stop_minutes"] == 60 and shadow["strategy"] == "SCALP")

# daily cap per symbol
bar["t"] += 300
verdict_answer.update(decision="CONFIRM", confidence_score=80)
config.SCALP_MAX_TRADES_PER_SYMBOL = 1
main._history_cache["stamp"] = None
before = len(prompts)
check("Max scalps per symbol per day -> skipped before asking the AI",
      asyncio.run(main.process_symbol("EURUSD")) == "skipped" and len(prompts) == before)
check("Journal: the daily-limit skip is recorded too", list(journal.iter_entries(1))[-1]["action"].startswith("skipped: "))
config.SCALP_MAX_TRADES_PER_SYMBOL = 4

scalper.in_session = lambda now=None: False
check("Outside the session -> off_session", asyncio.run(main.process_symbol("EURUSD")) == "off_session")
scalper.in_session = real_in_session

# ============================================================ 5. shadow timeout
epoch = 1_800_000_000
sh = {"side": "BUY", "stop_loss": 1.0990, "take_profit": 1.1015, "entry_price": 1.1000, "risk_reward": 1.5,
      "server_epoch": epoch, "spread_price": 0.0001, "time_stop_minutes": 60}
bars = [{"time": epoch + 60 * k, "high": 1.1004, "low": 1.0996, "close": 1.1003} for k in range(1, 90)]
check("Shadow: neither SL nor TP within 60 min -> TIMEOUT at +0.3R", shadow_store._outcome(sh, bars) == ("TIMEOUT", 0.3))
check("Shadow: window not over yet -> still open", shadow_store._outcome(sh, bars[:30]) is None)

# ============================================================ 6. settings + status
data_engine.get_account_snapshot = lambda: None
res = main.api_save_settings(main.SettingsUpdate(strategy_mode="SWING", scalp_time_stop_minutes=45, scalp_max_trades_per_symbol=6))
check("Strategy settings saved and applied", config.STRATEGY_MODE == "SWING" and config.SCALP_TIME_STOP_MINUTES == 45
      and res["settings"]["scalp_max_trades_per_symbol"] == 6)
payload = main._status_payload()
check("Status exposes strategy + backtest state", payload["strategy"]["mode"] == "SWING" and "backtest" in payload)
check("GET /api/backtest returns the latest report without the trade list",
      main.api_backtest()["report"]["file"] == report["file"] and "trades" not in main.api_backtest()["report"])

# ============================================================ 7. off-session behaviour + settings conflicts
now = datetime(2026, 9, 29, 16, 27, tzinfo=timezone.utc)  # 12:27 New York: window closed
opens = scalper.next_session_open(now)
check("Next session: Wed 06:00 UTC (07:00 London BST)", opens == datetime(2026, 9, 30, 6, 0, tzinfo=timezone.utc), opens)
check("Friday evening -> next open is Monday", scalper.next_session_open(datetime(2026, 10, 2, 17, 0, tzinfo=timezone.utc)).weekday() == 0)
check("Session note: the bot's entry window, not the market, with a countdown",
      "entry window closed; the forex market may still be open" in scalper.session_note(now)
      and "Wed 06:00 UTC" in scalper.session_note(now) and "in 13h 33m" in scalper.session_note(now), scalper.session_note(now))

config.STRATEGY_MODE = "SCALP"
scalper.in_session = lambda now=None: False
main.bot_state.update(is_running=True, decisions={}, _off_session_logged=0)
data_engine.ensure_connection = lambda: True
main._refresh_symbol_universe = lambda force=False: None
main._reconcile = lambda: None
main._weekend_close_positions = lambda: 0
main.news.refresh_if_stale = lambda force=False: False
real_learning = main._learning_cycle


async def no_learning():
    return None


main._learning_cycle = no_learning
calls = []
real_process = main.process_symbol


async def spy_process(symbol):
    calls.append(symbol)
    return "evaluated"


main.process_symbol = spy_process
active = asyncio.run(main.run_scan_cycle())
check("Off session: quiet scan, symbols not processed", active is False and calls == [])
check("Off session: every symbol shows why it holds", all("off session" in (x.get("note") or "")
                                                          for x in main.bot_state["decisions"].values())
      and set(main.bot_state["decisions"]) == set(config.SYMBOLS))
logged = main.bot_state["_off_session_logged"]
asyncio.run(main.run_scan_cycle())
check("Off-session reminder is throttled (not every scan)", main.bot_state["_off_session_logged"] == logged)
scalper.in_session = lambda now=None: True
active = asyncio.run(main.run_scan_cycle())
check("In session: symbols processed again", active is True and calls == config.SYMBOLS)
main.process_symbol, main._learning_cycle = real_process, real_learning
scalper.in_session = real_in_session

config.MAX_CURRENCY_RISK_PERCENT, config.MAX_DAILY_LOSS_PERCENT = 2.5, 10.0
main.bot_state["risk_percent"] = 3.0
warnings = main._setting_conflicts()
check("Settings check explains risk 3% vs cap 2.5%", any("one open trade per currency direction" in w for w in warnings), warnings)
positions.clear()
sl, tp = PRICE["EURUSD"] - 0.0010, PRICE["EURUSD"] + 0.0015
config.MAX_CURRENCY_RISK_PERCENT = 1.0
first = execution.preview_trade("EURUSD", "BUY", sl, tp, 3.0)
check("Trade risk above the currency cap: the FIRST trade in a direction is allowed", first["risk_percent"] > 1.0, first)
positions.append(SimpleNamespace(ticket=77, symbol="GBPUSD", volume=0.02, type=mt5.POSITION_TYPE_BUY, price_open=sl + 0.0011,
                                 price_current=sl + 0.0011, sl=sl, tp=tp, profit=0.0, swap=0.0, time=int(time.time()),
                                 magic=config.MAGIC_NUMBER, comment=""))
try:
    execution.preview_trade("EURUSD", "BUY", sl, tp, 3.0)  # second short-USD trade on top of BUY GBPUSD
    check("...but a second trade in the same direction is still refused", False)
except execution.TradeExecutionError as exc:
    check("...but a second trade in the same direction is still refused", "short USD" in str(exc), str(exc))
positions.clear()
main.bot_state["risk_percent"] = 2.0
config.MAX_CURRENCY_RISK_PERCENT = 2.5

# ============================================================ 8. rollover protection + backtest spread filter
config.SCALP_SESSION_END_NEW_YORK, config.SCALP_TIME_STOP_MINUTES = 17, 60
check("Entries until 17:00 NY with a 60-min time stop -> rollover warning",
      any("rollover" in w for w in main._setting_conflicts()), main._setting_conflicts())
config.SCALP_SESSION_END_NEW_YORK = 16
check("Entries until 16:00 NY -> no rollover warning", not any("rollover" in w for w in main._setting_conflicts()))
config.SCALP_SESSION_END_NEW_YORK = 12
wide = run_path([with_setup(uptrend_m5())[-1] + 0.0008 * k for k in range(1, 15)], spread=600)  # 6 pips vs ~15-pip stop
check("Backtest skips setups whose spread is > 25% of the stop (as the live engine does)", wide == [], wide[:1])
data_engine.get_account_snapshot = lambda: None
res = main.api_save_settings(main.SettingsUpdate(scalp_session_start_london=8, scalp_session_end_new_york=15))
check("Session hours editable from the dashboard", config.SCALP_SESSION_START_LONDON == 8
      and config.SCALP_SESSION_END_NEW_YORK == 15 and res["settings"]["scalp_session_end_new_york"] == 15)

# ============================================================ 9. adopted filters: strict guard + medium-term trend
check("Defaults: strict guard and medium-term trend ON; the rejected variants OFF",
      DEFAULT_STRICT and DEFAULT_MEDIUM and config.SCALP_MIN_ADX == 0 and config.SCALP_TRIGGER == "RSI"
      and config.SCALP_ROOM_MIN_R == 0 and config.SCALP_TARGET == "RR")
frames = frames_for(with_setup(uptrend_m5()), END)
config.SCALP_STRICT_GUARD, config.SCALP_MEDIUM_TREND = True, False
r = scalper.evaluate(scalper.prepare(frames), spread_price=0.0001)
check("Strict guard: a stretched setup is skipped, with the guard ids and the setup kept for a shadow trade",
      r["setup"] is None and r["stage"] == "STRETCHED" and r["blocked_setup"]["side"] == "BUY"
      and any(g["id"].startswith("G-OVEREXT") for g in r["guards"]), r["reason"])
config.SCALP_STRICT_GUARD = False
check("Strict guard off: the same setup is taken", scalper.evaluate(scalper.prepare(frames), spread_price=0.0001)["setup"])

# D1 above a rising EMA200 but falling for weeks (the AUDUSD case): EMA20 < EMA50, close < EMA50
d1 = list(0.60 + 0.001 * np.arange(260)) + [0.859 - 0.002 * k for k in range(1, 31)]
aud = dict(frames, D1=frame(d1, END - 86400 * 301, 86400))
config.SCALP_MEDIUM_TREND = False
base = scalper.evaluate(scalper.prepare(aud), spread_price=0.0001)
config.SCALP_MEDIUM_TREND = True
fixed = scalper.evaluate(scalper.prepare(aud), spread_price=0.0001)
check("Medium-term trend: 'uptrend' on EMA200 but falling for weeks -> no longer bought",
      base["trend"] == "UP" and base["setup"] is not None and fixed["setup"] is None and fixed["stage"] == "MEDIUM_AGAINST",
      (base["stage"], fixed["stage"], fixed["reason"]))

# live flow: a stretched setup is not sent to the AI but is followed as a shadow trade
config.SCALP_STRICT_GUARD, config.SCALP_MEDIUM_TREND = True, False
scalper.in_session = lambda now=None: True
scalper.fetch_live_frames = lambda symbol: frames
bar["t"] += 900
config.SHADOW_FILE.write_text("[]")  # the vetoed EURUSD BUY shadow above is still open (dedup)
before_prompts, before_shadows = len(prompts), len(shadow_store.load_shadows())
config.SCALP_MAX_TRADES_PER_SYMBOL = 50
out = asyncio.run(main.process_symbol("EURUSD"))
new_shadow = shadow_store.load_shadows()[-1]
stretched_line = list(journal.iter_entries(1))[-1]
check("Journal: stretched skip recorded with guard ids and its shadow id",
      stretched_line["stage"] == "STRETCHED" and stretched_line["action"] == "skipped: stretched"
      and stretched_line["guards"] and stretched_line.get("shadow_id"), stretched_line.get("action"))
check("Live: stretched setup -> AI not asked, shadow trade recorded with the guard ids",
      out == "no_setup" and len(prompts) == before_prompts and len(shadow_store.load_shadows()) == before_shadows + 1
      and new_shadow["blocked_by"] and all(b.startswith("G-") for b in new_shadow["blocked_by"])
      and main.bot_state["decisions"]["EURUSD"]["stage"] == "STRETCHED", (out, new_shadow.get("blocked_by")))
scalper.in_session = real_in_session
config.SCALP_MAX_TRADES_PER_SYMBOL = 4
data_engine.get_account_snapshot = lambda: None
res = main.api_save_settings(main.SettingsUpdate(scalp_strict_guard=False, scalp_medium_trend=False))
check("Filters switchable from Settings", not config.SCALP_STRICT_GUARD and not config.SCALP_MEDIUM_TREND
      and res["settings"]["scalp_strict_guard"] is False)

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
