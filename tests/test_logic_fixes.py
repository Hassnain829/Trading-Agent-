"""The logic-flaw fixes: persisted daily risk, open-risk budget, currency caps, min-lot tolerance, daily-loss
flatten, new-bar gate, reversal hysteresis + dry run, loss cooldown, drift/spread at send, calibration,
shadow trades, lost-reply duplicates, news guard, UTC exit times and settings."""
import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

tmp = Path(tempfile.mkdtemp())
for attr, name in (("MEMORY_FILE", "memory.json"), ("RULES_FILE", "new_rules.json"), ("SETTINGS_FILE", "settings.json"),
                   ("RISK_STATE_FILE", "risk_state.json"), ("SHADOW_FILE", "shadow.json"),
                   ("NEWS_CACHE_FILE", "news.json"), ("AUDIT_LOCK_FILE", ".audit.lock"), ("LOCK_FILE", ".server.lock")):
    setattr(config, attr, tmp / name)
config.AGENT_DIR = config.SHADOW_FILE.parent / "agent"  # the learning agent's files stay in the temp folder too
config.EXPLORE_FILE = config.AGENT_DIR / "explore_shadows.json"
config.SYMBOLS = ["EURUSD", "USDJPY", "USDCAD", "XAUUSD"]
config.CONFIDENCE_THRESHOLD, config.CALIBRATE_THRESHOLD = 65, True
config.MAX_DAILY_LOSS_PERCENT, config.MAX_CURRENCY_RISK_PERCENT = 5.0, 2.5
config.WEEKEND_ENTRY_CUTOFF_HOURS, config.WEEKEND_CLOSE = 0.0, False
config.STRATEGY_MODE = "SWING"  # these suites test the H1 swing flow
config.MAX_TOTAL_DRAWDOWN_PERCENT = config.DRAWDOWN_THROTTLE_PERCENT = 0.0  # tested in test_phase2_risk

import MetaTrader5 as mt5

import ai_brain
import calibration
import data_engine
import execution
import main
import memory_store
import news
import shadow_store

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ============================================================ fake MT5 account
PRICES = {"EURUSD": 1.10000, "USDJPY": 150.000, "USDCAD": 1.38000, "XAUUSD": 2400.00}
DIGITS = {"EURUSD": 5, "USDJPY": 3, "USDCAD": 5, "XAUUSD": 2}
SPREAD = {"EURUSD": 0.00010, "USDJPY": 0.010, "USDCAD": 0.00012, "XAUUSD": 0.20}
positions, sent = [], []
account = {"equity": 10_000.0, "login": 5}
state = {"lost_reply": False, "next_ticket": 9000}


def info(symbol):
    point = 10 ** -DIGITS[symbol]
    return SimpleNamespace(digits=DIGITS[symbol], point=point, trade_tick_size=point, trade_tick_value=1.0,
                           trade_tick_value_loss=1.0, trade_contract_size=100000.0, volume_min=0.01, volume_max=100.0,
                           volume_step=0.01, trade_stops_level=0, filling_mode=2, trade_mode=4, path="Forex\\" + symbol)


def tick(symbol):
    bid = PRICES[symbol]
    return SimpleNamespace(bid=bid, ask=round(bid + SPREAD[symbol], DIGITS[symbol]), time=int(time.time()),
                           time_msc=int(time.time() * 1000))


def order_send(req):
    sent.append(dict(req))
    if req.get("position"):
        positions[:] = [p for p in positions if p.ticket != req["position"]]
        return None if state["lost_reply"] else SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, deal=1, order=1,
                                                                price=req["price"], volume=req["volume"], comment="ok")
    state["next_ticket"] += 1
    positions.append(SimpleNamespace(ticket=state["next_ticket"], symbol=req["symbol"], volume=req["volume"],
                                     type=mt5.POSITION_TYPE_BUY if req["type"] == mt5.ORDER_TYPE_BUY else mt5.POSITION_TYPE_SELL,
                                     price_open=req["price"], price_current=req["price"], sl=req["sl"], tp=req["tp"],
                                     profit=0.0, swap=0.0, time=int(time.time()), magic=req["magic"], comment="AI-HF"))
    if state["lost_reply"]:
        return None
    return SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, deal=0, order=state["next_ticket"], price=req["price"],
                           volume=req["volume"], comment="ok")


def pos(symbol, side, volume, entry, current, sl, magic=config.MAGIC_NUMBER, ticket=None):
    state["next_ticket"] += 1
    return SimpleNamespace(ticket=ticket or state["next_ticket"], symbol=symbol, volume=volume,
                           type=mt5.POSITION_TYPE_BUY if side == "BUY" else mt5.POSITION_TYPE_SELL,
                           price_open=entry, price_current=current, sl=sl, tp=0.0, profit=0.0, swap=0.0,
                           time=int(time.time()), magic=magic, comment="")


mt5.symbol_info = info
mt5.symbol_info_tick = tick
mt5.account_info = lambda: SimpleNamespace(login=account["login"], equity=account["equity"], balance=10_000.0,
                                           margin_free=9_000.0, currency="USD", profit=0.0, margin=0.0, margin_level=0.0,
                                           leverage=100, server="Demo", company="X", name="T", trade_allowed=True,
                                           trade_mode=mt5.ACCOUNT_TRADE_MODE_DEMO)
mt5.positions_get = lambda **kw: tuple(p for p in positions if kw.get("ticket") in (None, p.ticket)
                                       and kw.get("symbol") in (None, p.symbol))
mt5.order_calc_margin = lambda *a: 50.0
mt5.order_send = order_send
mt5.history_deals_get = lambda **kw: ()
mt5.last_error = lambda: (1, "ok")

# ============================================================ 1. open risk + currency cap + budget
positions[:] = [pos("EURUSD", "SELL", 0.5, 1.1000, 1.1000, 1.1020),   # long USD, 200 pts x 0.5 = 100 = 1.0%
                pos("USDJPY", "BUY", 0.5, 150.0, 150.0, 149.8)]       # long USD, 200 pts x 0.5 = 1.0%
risk = execution.open_risk()
check("Open risk: 2 positions x 1% = 2%, both long USD", risk["total_percent"] == 2.0
      and risk["by_currency"]["USD"]["long"] == 2.0 and risk["by_currency"]["EUR"]["short"] == 1.0, risk)
positions.append(pos("EURUSD", "BUY", 0.5, 1.0950, 1.1000, 1.0990))   # stop in profit: 100 pts left to give back
positions.append(pos("XAUUSD", "BUY", 0.01, 2400, 2400, 0.0, magic=1))  # manual, no stop
risk = execution.open_risk()
rows = {r["symbol"] + r["side"]: r["risk_percent"] for r in risk["positions"]}
check("Stop in profit: only the giveback from here counts (0.5%); unprotected position counts 5%",
      rows["EURUSDBUY"] == 0.5 and rows["XAUUSDBUY"] == 5.0, rows)
positions[:] = positions[:2]


def stops(symbol, side, atr_mult=200):
    point = 10 ** -DIGITS[symbol]
    price = tick(symbol).ask if side == "BUY" else tick(symbol).bid
    d = atr_mult * point
    return (round(price - d, DIGITS[symbol]), round(price + 2 * d, DIGITS[symbol])) if side == "BUY" else \
        (round(price + d, DIGITS[symbol]), round(price - 2 * d, DIGITS[symbol]))


sl, tp = stops("USDCAD", "BUY")
try:
    execution.preview_trade("USDCAD", "BUY", sl, tp, 1.0)
    check("Third long-USD trade refused by the 2.5% currency cap", False)
except execution.TradeExecutionError as exc:
    check("Third long-USD trade refused by the 2.5% currency cap", "long USD" in str(exc), str(exc))
sl, tp = stops("EURUSD", "BUY")
check("A short-USD trade is still allowed", execution.preview_trade("EURUSD", "BUY", sl, tp, 1.0)["risk_percent"] <= 1.0)
try:
    execution.preview_trade("EURUSD", "BUY", sl, tp, 1.0, max_total_open_risk=2.5)
    check("Daily loss budget: 2% open + 1% new > 2.5% left -> refused", False)
except execution.TradeExecutionError as exc:
    check("Daily loss budget: 2% open + 1% new > 2.5% left -> refused", "daily loss budget" in str(exc), str(exc))

# ============================================================ 2. min-lot tolerance 1.25
account["equity"] = 100.0
positions[:] = []
sl, tp = stops("EURUSD", "BUY", 150)  # 0.01 lot risks 1.50 = 1.5% of 100 vs 1% budget
try:
    execution.preview_trade("EURUSD", "BUY", sl, tp, 1.0)
    check("Min lot risking 1.5x budget refused (tolerance 1.25)", False)
except execution.TradeExecutionError as exc:
    check("Min lot risking 1.5x budget refused (tolerance 1.25)", "minimum lot" in str(exc), str(exc))
sl, tp = stops("EURUSD", "BUY", 120)  # 1.20 = 1.2x budget -> allowed
check("Min lot at 1.2x budget allowed", execution.preview_trade("EURUSD", "BUY", sl, tp, 1.0)["volume"] == 0.01)
account["equity"] = 10_000.0

# ============================================================ 3. drift + spread at send
sl, tp = stops("EURUSD", "BUY")
try:
    execution.preview_trade("EURUSD", "BUY", sl, tp, 1.0, entry_reference=1.0990, atr_reference=0.0010)
    check("Price moved 1.1 ATR while deciding -> refused", False)
except execution.TradeExecutionError as exc:
    check("Price moved 1.1 ATR while deciding -> refused", "stale" in str(exc), str(exc))
check("Small drift (0.1 ATR) accepted", execution.preview_trade("EURUSD", "BUY", sl, tp, 1.0, entry_reference=1.1000,
                                                                atr_reference=0.0010)["volume"] > 0)
SPREAD["EURUSD"] = 0.0008
sl, tp = stops("EURUSD", "BUY")
try:
    execution.preview_trade("EURUSD", "BUY", sl, tp, 1.0)
    check("Spread widened to 40% of the stop at send -> refused", False)
except execution.TradeExecutionError as exc:
    check("Spread widened to 40% of the stop at send -> refused", "spread widened" in str(exc), str(exc))
SPREAD["EURUSD"] = 0.00010

# ============================================================ 4. lost reply never duplicates
state["lost_reply"] = True
sent.clear()
sl, tp = stops("EURUSD", "BUY")
trade = execution.execute_trade("EURUSD", "BUY", sl, tp, 1.0)
check("order_send returned None but the fill exists -> recovered, sent once", len(sent) == 1 and trade["position_ticket"]
      == positions[-1].ticket and len(positions) == 1, (len(sent), len(positions)))
sent.clear()
result = execution.close_position(positions[-1].ticket)
check("Close with a lost reply: position gone -> success, sent once", result["success"] and len(sent) == 1 and not positions)
state["lost_reply"] = False

# ============================================================ 5. daily risk survives restarts
main.bot_state.update(day_key=None, day_start_equity=None, day_login=None, circuit_breaker=False, profit_target_hit=False,
                      breaker_flattened=False)
main._update_daily_risk(10_000.0, 5)
main._update_daily_risk(9_400.0, 5)
check("Breaker trips at -6%", main.bot_state["circuit_breaker"])
main.bot_state.update(day_key=None, day_start_equity=None, day_login=None, circuit_breaker=False)  # "restart"
main._update_daily_risk(9_450.0, 5)
check("After a restart: start equity and tripped breaker restored from disk",
      main.bot_state["day_start_equity"] == 10_000.0 and main.bot_state["circuit_breaker"], main.bot_state["day_start_equity"])
main._update_daily_risk(9_450.0, 6)
check("Another login gets its own baseline", main.bot_state["day_start_equity"] == 9_450.0 and not main.bot_state["circuit_breaker"])
saved = json.loads(config.RISK_STATE_FILE.read_text())
check("Risk state saved per login", set(saved) == {"5", "6"} and saved["5"]["circuit_breaker"] is True)
real_key = data_engine.trading_day_key
data_engine.trading_day_key = lambda now=None: "2099-01-01"
main._update_daily_risk(9_450.0, 5)
check("New trading day: fresh baseline, breaker cleared", main.bot_state["day_start_equity"] == 9_450.0
      and not main.bot_state["circuit_breaker"])
data_engine.trading_day_key = real_key
friday_1659 = datetime(2026, 9, 25, 20, 59, tzinfo=timezone.utc)
check("Trading day rolls at 17:00 New York", data_engine.trading_day_key(friday_1659) == "2026-09-25"
      and data_engine.trading_day_key(datetime(2026, 9, 24, 21, 1, tzinfo=timezone.utc)) == "2026-09-25"
      and data_engine.trading_day_key(datetime(2026, 9, 24, 20, 59, tzinfo=timezone.utc)) == "2026-09-24")
check("Budget left = loss limit - today's drawdown", abs(main._daily_risk_budget() - 5.0) < 1e-9)

# ============================================================ 6. daily loss flatten (watchdog)
positions[:] = [pos("EURUSD", "BUY", 0.5, 1.1, 1.1, 1.098), pos("USDJPY", "SELL", 0.2, 150, 150, 150.3, magic=1)]
main.bot_state.update(is_running=True, mt5_connected=True, day_key=None, day_login=None)
data_engine.get_account_snapshot = lambda: {"login": 7, "equity": account["equity"], "balance": 10_000.0, "profit": 0.0,
                                            "server": "Demo", "company": "X", "account_mode": "DEMO"}
config.DAILY_LOSS_CLOSE_POSITIONS = True
main._watchdog_tick(False)                       # baseline 10,000
account["equity"] = 9_400.0
main._watchdog_tick(False)
check("Daily limit hit: the engine's position closed, the manual one kept",
      [p.magic for p in positions] == [1] and main.bot_state["breaker_flattened"], [p.symbol for p in positions])
sent.clear()
main._watchdog_tick(False)
check("Flatten happens once, not every tick", not sent)
account["equity"] = 10_000.0
positions[:] = []

# ============================================================ 7. new-bar gate, reversal, cooldown, news (process_symbol)
main.bot_state.update(circuit_breaker=False, profit_target_hit=False, daily_drawdown_pct=0.0)
config.MAX_DAILY_LOSS_PERCENT = 5.0
market_state = {"bar": "2026-09-25T10:00:00+00:00", "mid": 1.10005}


def fake_market(symbol):
    t = tick(symbol)
    return {"symbol": symbol, "tradeable": True, "market_idle": False, "path": "Forex\\" + symbol, "bid": t.bid, "ask": t.ask,
            "mid": market_state["mid"], "spread": 10, "spread_price": SPREAD[symbol], "digits": DIGITS[symbol],
            "point": 10 ** -DIGITS[symbol], "tick_size": 10 ** -DIGITS[symbol], "equity": account["equity"],
            "balance": 10_000.0, "currency": "USD", "account_login": 5, "account_mode": "DEMO", "broker": "X",
            "server": "Demo", "tick_epoch": int(time.time()), "day_change_pct": 0.1,
            "h1_data": {"atr14": 0.0010, "last_bar_time": market_state["bar"], "structure": {}},
            "daily_data": {"structure": {}}}


asked = []
answer = {"signal": "BUY", "confidence": 78}


def fake_ai(market, symbol):
    asked.append(symbol)
    side = answer["signal"]
    price = market["ask"] if side == "BUY" else market["bid"]
    d = 0.0020
    return {"symbol": symbol, "timestamp": data_engine.utc_now_iso(), "signal": side, "raw_signal": side,
            "confidence_score": answer["confidence"], "base_confidence": answer["confidence"], "threshold": 65,
            "applied_rules": [], "blocked_by": [], "logic": "t", "error": answer.get("error"),
            "stop_loss": round(price - d if side == "BUY" else price + d, 5),
            "take_profit": round(price + 2 * d if side == "BUY" else price - 2 * d, 5), "sl_distance": d, "tp_distance": 2 * d,
            "sl_atr_multiple": 2.0, "tp_atr_multiple": 4.0, "risk_reward": 2.0, "entry_reference": price,
            "atr_reference": 0.0010, "stops_source": "AI_ATR", "stop_notes": [], "atr_source": "H1 ATR14"}


data_engine.fetch_multi_timeframe_data = fake_market
data_engine.fetch_correlated_asset_prices = lambda *a, **k: {}
main.ai_brain.get_ai_decision = fake_ai
main._capture_context = lambda symbol, market=None: None
main._refresh_account_state = lambda: None
config.AI_NEW_BAR_ONLY, config.NEWS_GUARD, config.LOSS_COOLDOWN_MINUTES = True, False, 120
run = lambda: asyncio.run(main.process_symbol("EURUSD"))

check("First look: AI asked, BUY filled", run() == "evaluated" and asked == ["EURUSD"] and len(positions) == 1)
check("Same H1 bar, same price: AI not asked again", run() == "unchanged" and asked == ["EURUSD"])
market_state["mid"] = 1.10060  # +0.55 ATR
check("Price moved 0.55 ATR: AI asked again", run() == "evaluated" and len(asked) == 2)
market_state["bar"] = "2026-09-25T11:00:00+00:00"
answer.update(signal="SELL", confidence=70)
check("New bar, SELL at 70 < 65+10: open BUY kept (hysteresis)", run() == "evaluated" and positions[0].type == mt5.POSITION_TYPE_BUY)
market_state["bar"] = "2026-09-25T12:00:00+00:00"
answer.update(confidence=80)
positions.append(pos("USDJPY", "BUY", 0.5, 150, 150, 149.8))  # long USD 1%
positions.append(pos("USDCAD", "BUY", 0.5, 1.38, 1.38, 1.378))  # long USD 1% -> SELL EURUSD would make 3%
check("Flip whose replacement would break the currency cap: BUY kept, nothing closed",
      run() == "evaluated" and any(p.symbol == "EURUSD" and p.type == mt5.POSITION_TYPE_BUY for p in positions)
      and "currency exposure" in main.bot_state["last_error"], main.bot_state["last_error"])
positions[:] = [p for p in positions if p.symbol == "EURUSD"]
market_state["bar"] = "2026-09-25T13:00:00+00:00"
check("Flip at 80 with room: BUY closed, SELL opened", run() == "evaluated"
      and [(p.symbol, p.type) for p in positions] == [("EURUSD", mt5.POSITION_TYPE_SELL)])
positions[:] = []

answer.update(error="timeout", signal="HOLD")
market_state["bar"] = "2026-09-25T14:00:00+00:00"
before = len(asked)
run()
run()
check("Failed AI call is retried next scan (not cached)", len(asked) == before + 2)
answer.pop("error")

now = datetime.now(timezone.utc)
config.MEMORY_FILE.write_text(json.dumps([
    {"id": "l1", "status": "CLOSED", "outcome": "LOSS", "symbol": "EURUSDm", "side": "BUY", "realized_pnl": -50,
     "timestamp": (now - timedelta(hours=2)).isoformat(), "exit_time_utc": (now - timedelta(minutes=30)).isoformat()}]))
main._history_cache["stamp"] = None
check("Loss cooldown: BUY blocked 30 min after a losing BUY", main._loss_cooldown_block("EURUSD", "BUY") is not None)
check("Loss cooldown: SELL not blocked", main._loss_cooldown_block("EURUSD", "SELL") is None)
answer.update(signal="BUY", confidence=80)
market_state["bar"] = "2026-09-25T15:00:00+00:00"
check("process_symbol respects the cooldown", run() == "evaluated" and not positions)
config.LOSS_COOLDOWN_MINUTES = 20
check("Loss older than the cooldown: allowed", main._loss_cooldown_block("EURUSD", "BUY") is None)

# news
config.NEWS_GUARD = True
event_time = datetime.now(timezone.utc) + timedelta(minutes=10)
news._state.update(events=[{"title": "Non-Farm Employment Change", "currency": "USD",
                            "time": event_time.isoformat(timespec="minutes")}], fetched_at=data_engine.utc_now_iso(), error=None)
check("News: USD event in 10 min blocks EURUSD and XAUUSD", news.blocking_event("EURUSD") and news.blocking_event("XAUUSD"))
check("News: does not block EURJPY... (no USD leg)", news.blocking_event("EURJPY") is None)
check("News: 16 min after the event -> clear", news.blocking_event("EURUSD", event_time + timedelta(minutes=16)) is None)
market_state["bar"] = "2026-09-25T16:00:00+00:00"
before = len(asked)
check("process_symbol skips the AI during the news window", run() == "skipped" and len(asked) == before)
check("AI context lists upcoming high-impact news", "Non-Farm Employment Change" in ai_brain._news_block("EURUSD"))
news._state.update(events=None, fetched_at=None, error=None, checked=0.0)
config.NEWS_CACHE_FILE.write_text(json.dumps({"fetched_at": (datetime.now(timezone.utc) - timedelta(hours=9)).isoformat(),
                                              "events": [{"title": "Old", "currency": "EUR", "time": "2026-01-01T00:00"}]}))
news.requests.get = lambda *a, **k: (_ for _ in ()).throw(news.requests.ConnectionError("offline"))
check("Calendar download fails -> last saved calendar kept, error shown", news.refresh_if_stale() is False
      and news.status()["has_data"] and "offline" in news.status()["error"])
config.NEWS_GUARD = False

# ============================================================ 8. calibration
def closed(conf, pnl):
    return {"status": "CLOSED", "confidence_score": conf, "realized_pnl": pnl, "risk_percent": 1.0,
            "account_mode": "DEMO", "market_context": {"equity": 10_000.0}}


recs = [closed(67, -100.0) for _ in range(8)] + [closed(76, 200.0) for _ in range(4)] + [closed(81, -100) for _ in range(2)]
report = calibration.refresh(recs, "DEMO")
check("Calibration: the 65-69 band lost over 8 trades -> threshold 70",
      report["suggested_threshold"] == 70 and calibration.effective_threshold() == 70, report["reason"])
recs += [closed(77, 150.0) for _ in range(8)]
report = calibration.refresh(recs, "DEMO")
check("Calibration: overall profit does not hide the losing low band -> still 70",
      report["suggested_threshold"] == 70, report["reason"])
bad = [closed(67, -100.0)] * 12 + [closed(72, -100.0)] * 12 + [closed(80, -100.0)] * 10
report = calibration.refresh(bad, "DEMO")
check("Calibration: no level proven profitable -> capped at 85", report["suggested_threshold"] == 85, report["reason"])
calibration.refresh([closed(70, 100.0)] * 3, "DEMO")
check("Calibration: too few trades -> configured threshold", calibration.effective_threshold() == 65)
config.CALIBRATE_THRESHOLD = False
calibration.refresh(recs[:8], "DEMO")
check("Calibration off -> configured threshold", calibration.effective_threshold() == 65)
config.CALIBRATE_THRESHOLD = True
calibration.refresh([], None)

# ============================================================ 9. shadow trades
epoch = 1_800_000_000
mkt = {"tick_epoch": epoch, "spread_price": 0.0001, "account_mode": "DEMO", "broker": "X"}
dec = {"raw_signal": "BUY", "blocked_by": ["R-1", "G-USD"], "stop_loss": 1.0980, "take_profit": 1.1040,
       "entry_reference": 1.1000, "risk_reward": 2.0, "base_confidence": 72, "confidence_score": 50}
check("Blocked trade recorded", shadow_store.record_blocked("EURUSD", mkt, dec) is not None)
check("Same idea within an hour not recorded twice", shadow_store.record_blocked("EURUSD", mkt, dec) is None)
shadow_store.record_blocked("USDJPY", dict(mkt, spread_price=0.02), dict(dec, raw_signal="SELL", stop_loss=150.20,
                                                                            take_profit=149.60, entry_reference=150.0))
bars = {"EURUSD": [{"time": epoch + 60 * i, "high": 1.1005 + 0.0005 * i, "low": 1.0995} for i in range(1, 10)],
        "USDJPY": [{"time": epoch + 60, "high": 150.19, "low": 149.9}]}  # 150.19 + spread 0.02 >= 150.20 -> SL
mt5.copy_rates_from_pos = lambda symbol, tf, start, count: bars[symbol]
shadows = shadow_store.load_shadows()
for s in shadows:
    s["created_at"] = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
config.SHADOW_FILE.write_text(json.dumps(shadows))
check("Two shadows resolved", shadow_store.resolve_open_shadows() == 2)
by = {s["symbol"]: s for s in shadow_store.load_shadows()}
check("BUY reached TP first -> WIN (+2R); SELL stopped by the ask (bid + spread) -> LOSS",
      by["EURUSD"]["status"] == "WIN" and by["EURUSD"]["r_multiple"] == 2.0 and by["USDJPY"]["status"] == "LOSS",
      {k: v["status"] for k, v in by.items()})
stats = shadow_store.stats_by_blocker()
check("Shadow stats per blocker", stats["R-1"]["resolved"] == 2 and stats["R-1"]["expectancy_r"] == 0.5
      and stats["G-USD"]["blocked"] == 2, stats["R-1"])

# ============================================================ 10. UTC exit time
data_engine._server_offset.update(seconds=3 * 3600.0, known=True)
check("Server time 15:00 (GMT+3) -> 12:00 UTC", data_engine.server_epoch_to_utc_iso(
    datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc).timestamp()).startswith("2026-09-25T12:00"))
data_engine._server_offset.update(seconds=0.0)

# ============================================================ 11. settings
data_engine.get_account_snapshot = lambda: None
res = main.api_save_settings(main.SettingsUpdate(max_currency_risk_percent=3.0, daily_loss_close_positions=False,
                                                 loss_cooldown_minutes=60, ai_new_bar_only=False, news_guard=False,
                                                 calibrate_threshold=False))
s = res["settings"]
check("New protection settings saved and applied", s["max_currency_risk_percent"] == 3.0 and config.LOSS_COOLDOWN_MINUTES == 60
      and config.AI_NEW_BAR_ONLY is False and config.CALIBRATE_THRESHOLD is False and config.DAILY_LOSS_CLOSE_POSITIONS is False
      and json.loads(config.SETTINGS_FILE.read_text())["loss_cooldown_minutes"] == 60)
g = main._status_payload()["guardrails"]
check("Status payload exposes the new guardrails", {"effective_threshold", "max_currency_risk_percent", "risk_budget_left_percent",
                                                    "loss_cooldown_minutes", "news_guard"} <= set(g))

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
