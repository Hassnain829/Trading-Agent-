"""End-to-end: an executed trade lands in memory.json with execution-moment volume/volatility/correlation context."""
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

tmp = Path(tempfile.mkdtemp())
config.MEMORY_FILE, config.RULES_FILE = tmp / "memory.json", tmp / "new_rules.json"
config.SYMBOLS = ["EURUSDm", "GBPUSDm", "USDJPYm", "XAUUSDm"]
config.WEEKEND_ENTRY_CUTOFF_HOURS, config.WEEKEND_CLOSE = 0.0, False  # independent of the wall clock
config.AI_NEW_BAR_ONLY, config.NEWS_GUARD = False, False  # this suite tests logging, not the scan gates
config.STRATEGY_MODE = "SWING"
config.RISK_STATE_FILE, config.SHADOW_FILE, config.NEWS_CACHE_FILE = tmp / "risk_state.json", tmp / "shadow.json", tmp / "news.json"
config.AGENT_DIR = config.SHADOW_FILE.parent / "agent"  # the learning agent's files stay in the temp folder too
config.EXPLORE_FILE = config.AGENT_DIR / "explore_shadows.json"
config.TRADING_MODE = "DEMO"  # these checks place (stubbed) orders

import MetaTrader5 as mt5
import data_engine
import execution
import memory_store
import main

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- synthetic MT5 market
NOW = 1_790_000_000  # server epoch seconds, aligned below
NOW -= NOW % 3600
NOW += 1500  # 25 minutes into the current H1 bar
rng = np.random.default_rng(3)
base_moves = rng.normal(0, 0.001, 400)
SPEC = {  # symbol: (digits, point, level, beta vs common factor)
    "EURUSDm": (5, 0.00001, 1.10, 1.0),
    "GBPUSDm": (5, 0.00001, 1.34, 0.9),
    "USDJPYm": (3, 0.001, 148.0, -0.8),
    "XAUUSDm": (3, 0.001, 2650.0, 0.3),
}
DTYPE = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
         ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")]


def bars(symbol, seconds, start_pos, count):
    digits, point, level, beta = SPEC[symbol]
    own = np.random.default_rng(hash((symbol, seconds)) % 2**32).normal(0, 0.0006, 400)
    returns = beta * base_moves + own
    closes = level * np.exp(np.cumsum(returns))
    closes = closes[::-1][start_pos:start_pos + count][::-1]  # newest last
    n = len(closes)
    times = NOW - NOW % seconds - seconds * np.arange(start_pos, start_pos + n)[::-1]
    out = np.zeros(n, dtype=DTYPE)
    out["time"], out["close"] = times, closes
    out["open"] = np.r_[closes[0], closes[:-1]]
    out["high"] = np.maximum(out["open"], closes) * (1 + 0.0004)
    out["low"] = np.minimum(out["open"], closes) * (1 - 0.0004)
    vol = np.full(n, 900)
    if seconds == 60:
        vol[-6:-1] = 2400  # volume burst in the last 5 closed minutes
        vol[-1] = 300
    if seconds == 3600:
        vol[-1] = 700      # forming H1 bar
    out["tick_volume"] = vol
    return out


TF_SECONDS = {mt5.TIMEFRAME_M1: 60, mt5.TIMEFRAME_H1: 3600, mt5.TIMEFRAME_D1: 86400}
mt5.copy_rates_from_pos = lambda s, tf, start, count: bars(s, TF_SECONDS[tf], start, count)
mt5.symbol_select = lambda s, on=True: True
ticks = {s: SimpleNamespace(bid=round(bars(s, 60, 0, 1)["close"][0], SPEC[s][0]),
                            ask=round(bars(s, 60, 0, 1)["close"][0] + 12 * SPEC[s][1], SPEC[s][0]),
                            time=NOW, time_msc=NOW * 1000) for s in SPEC}
mt5.symbol_info_tick = lambda s: ticks[s]
mt5.symbol_info = lambda s: SimpleNamespace(digits=SPEC[s][0], point=SPEC[s][1], trade_tick_size=SPEC[s][1],
                                            trade_tick_value=1.0, trade_tick_value_loss=1.0, trade_contract_size=100000.0,
                                            volume_min=0.01, volume_max=100.0, volume_step=0.01, trade_stops_level=0,
                                            filling_mode=2, trade_mode=4, spread=12)
mt5.account_info = lambda: SimpleNamespace(login=77, equity=10_000.0, balance=10_000.0, margin_free=9000.0, currency="USD",
                                           profit=0.0, margin=0.0, margin_level=0.0, leverage=100, server="Demo",
                                           company="X", name="T", trade_allowed=True)
mt5.order_calc_margin = lambda *a: 100.0
positions = []
mt5.positions_get = lambda **kw: tuple(p for p in positions if kw.get("ticket") in (None, p.ticket)
                                       and kw.get("symbol") in (None, p.symbol))


def order_send(req):
    time.sleep(0.01)
    slip = 2 * SPEC[req["symbol"]][1]
    fill = req["price"] + slip if req["type"] == mt5.ORDER_TYPE_BUY else req["price"] - slip
    if req.get("position"):
        positions[:] = [p for p in positions if p.ticket != req["position"]]
    return SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, deal=5001, order=7001, price=fill, volume=req["volume"], comment="ok")


mt5.order_send = order_send
mt5.history_deals_get = lambda **kw: (SimpleNamespace(position_id=7001),)

# ---------------------------------------------------------------- 1. capture function in isolation
market = data_engine.fetch_multi_timeframe_data("EURUSDm")
ctx = data_engine.capture_execution_context("EURUSDm", config.SYMBOLS, market)
check("Context has quote/volume/volatility/correlated", all(k in ctx for k in ("quote", "volume", "volatility", "correlated_assets")))
check("H1 forming bar pace: 25 min in (41.7%)", ctx["volume"]["h1"]["elapsed_pct"] == 41.7
      and ctx["volume"]["h1"]["forming_volume"] == 700, json.dumps(ctx["volume"]["h1"]))
check("M1 5-minute volume burst detected", ctx["volume"]["m1"]["rel_volume_5m"] > 2.0, ctx["volume"]["m1"]["rel_volume_5m"])
check("Volatility: H1 ATR + M1 ATR + realized vol + 15m range",
      all(ctx["volatility"]["m1"].get(k) is not None for k in ("m1_atr14", "realized_vol_1h_pct", "range_15m", "range_15m_atr"))
      and ctx["volatility"]["h1"]["atr14"] == market["h1_data"]["atr14"])
corr = {s: q["corr_h1"] for s, q in ctx["correlated_assets"].items()}
check("All 3 other symbols with bid/ask/spread/day change/corr", set(corr) == {"GBPUSDm", "USDJPYm", "XAUUSDm"}
      and all(k in ctx["correlated_assets"]["GBPUSDm"] for k in ("bid", "ask", "spread_points", "day_change_pct")))
check("Correlation signs match the synthetic betas (GBP +, JPY -)", corr["GBPUSDm"] > 0.5 and corr["USDJPYm"] < -0.5, corr)
check("Context JSON-serializable", bool(json.dumps(ctx)))

# ---------------------------------------------------------------- 2. process_symbol end-to-end
main.bot_state["is_running"] = True
decision_template = {
    "symbol": "EURUSDm", "timestamp": "2026-09-25T05:00:00+00:00", "signal": "BUY", "raw_signal": "BUY",
    "confidence_score": 78, "base_confidence": 78, "applied_rules": [], "logic": "test", "error": None,
    "sl_atr_multiple": 1.5, "tp_atr_multiple": 3.0, "stops_source": "AI_ATR", "risk_reward": 2.0, "stop_notes": [],
    "atr_source": "H1 ATR14",
}


def fake_decision(market, symbol):
    atr = market["h1_data"]["atr14"]
    entry = market["ask"]
    return {**decision_template, "entry_reference": entry, "atr_reference": atr,
            "stop_loss": round(entry - 1.5 * atr, 5), "take_profit": round(entry + 3 * atr, 5),
            "sl_distance": 1.5 * atr, "tp_distance": 3 * atr}


main.ai_brain.get_ai_decision = fake_decision
asyncio.run(main.process_symbol("EURUSDm"))
records = memory_store.load_trade_memory()
check("Executed trade logged to memory.json", len(records) == 1, len(records))
rec = records[0]
mc = rec["market_context"]
check("Record: context source EXECUTION with capture timestamp", mc["context_source"] == "EXECUTION" and mc["captured_at"])
check("Record: volume block (H1 pace + 5m burst + D1)", mc["volume"]["m1"]["rel_volume_5m"] > 2.0
      and "projected_rel_volume" in mc["volume"]["h1"] and "rel_volume" in mc["volume"]["d1"])
check("Record: volatility block", mc["volatility"]["m1"]["realized_vol_1h_pct"] is not None and mc["volatility"]["h1"]["atr14"])
check("Record: correlated asset prices at fill with corr", len(mc["correlated_prices"]) == 3
      and all("corr_h1" in q and "bid" in q for q in mc["correlated_prices"].values()))
check("Record: decision quote kept for drift analysis", mc["decision_quote"]["bid"] == market["bid"])
ex = rec["execution"]
check("Record: execution quality (slippage +2 pts, latency, IOC)", ex["slippage_points"] == 2.0 and ex["execution_ms"] >= 10
      and ex["type_filling"] == "IOC" and ex["requested_price"] < ex["fill_price"], json.dumps(ex))
check("Record: ATR stops + indicator snapshot kept", rec["sl_atr_multiple"] == 1.5 and mc["h1"]["rsi14"] is not None)

# ---------------------------------------------------------------- 3. reversal close logged with context
positions.append(SimpleNamespace(ticket=7001, symbol="EURUSDm", type=mt5.POSITION_TYPE_BUY, volume=0.5, profit=-12.3,
                                 magic=config.MAGIC_NUMBER, price_open=1.1, price_current=1.099, sl=1.09, tp=1.12,
                                 swap=0.0, time=NOW, comment="AI-HF"))
decision_template["signal"] = decision_template["raw_signal"] = "SELL"


def fake_sell(market, symbol):
    atr, entry = market["h1_data"]["atr14"], market["bid"]
    return {**decision_template, "entry_reference": entry, "atr_reference": atr, "stop_loss": round(entry + 1.5 * atr, 5),
            "take_profit": round(entry - 3 * atr, 5), "sl_distance": 1.5 * atr, "tp_distance": 3 * atr}


main.ai_brain.get_ai_decision = fake_sell
mt5.history_deals_get = lambda **kw: (SimpleNamespace(position_id=7002),)
asyncio.run(main.process_symbol("EURUSDm"))
records = memory_store.load_trade_memory()
first = next(r for r in records if r["ticket"] == 7001)
ce = first.get("close_execution") or {}
check("Reversal close recorded on the original trade", ce.get("reason") == "REVERSAL" and ce.get("deal") == 5001)
check("Close carries its own execution-moment context", (ce.get("market_context") or {}).get("volume", {}).get("m1")
      and len(ce["market_context"]["correlated_prices"]) == 3 and "slippage_points" in ce)
check("New SELL logged as second trade", len(records) == 2 and records[1]["side"] == "SELL")

# ---------------------------------------------------------------- 4. never lose a fill: journal fallback
real_write = memory_store.write_json_atomic
calls = {"n": 0}


def flaky_write(path, data):
    calls["n"] += 1
    if calls["n"] == 1:
        raise PermissionError("memory.json locked by another process")
    return real_write(path, data)


memory_store.write_json_atomic = flaky_write
memory_store.append_trade_memory({"id": "locked-1", "symbol": "XAUUSDm", "side": "BUY", "ticket": 9001,
                                  "timestamp": "2026-09-25T09:00:00+00:00"})
pending = config.MEMORY_FILE.with_name("memory.pending.jsonl")
check("Locked memory.json -> fill journaled, not lost", pending.exists() and "locked-1" in pending.read_text())
memory_store.append_trade_memory({"id": "next-1", "symbol": "GBPUSDm", "side": "SELL", "ticket": 9002,
                                  "timestamp": "2026-09-25T09:05:00+00:00"})
ids = [r["id"] for r in memory_store.load_trade_memory()]
check("Next write merges the journal back", "locked-1" in ids and "next-1" in ids and not pending.exists(), ids)
check("Recovered fill keeps chronological order", ids.index("locked-1") < ids.index("next-1"))
memory_store.write_json_atomic = real_write
memory_store._journal_pending({"id": "boot-1", "symbol": "USDJPYm", "ticket": 9003})
check("Startup flush recovers journal", memory_store.flush_pending_journal() == 1
      and "boot-1" in [r["id"] for r in memory_store.load_trade_memory()])
check("Flush is idempotent", memory_store.flush_pending_journal() == 0)

# ---------------------------------------------------------------- 5. dashboard payloads
main._history_cache["stamp"] = None
cache = main._history_and_stats()
last = cache["last_execution"]
check("Dashboard last_execution = latest fill with context", last and last["side"] == "SELL"
      and last["volume"]["m1"] and len(last["correlated"]) == 3 and last["execution"]["slippage_points"] == 2.0)
row = next(r for r in cache["ledger"] if r["ticket"] == 7001)
check("Ledger row has slippage + close reason", row["slippage_points"] == 2.0 and row["close_reason"] == "REVERSAL")

import auditor
view = auditor._trade_brief(records[0], "L1")
m = view["metrics"]
check("Auditor sees the rule metrics of the fill (volume, volatility, cross-asset incl. corr)",
      m["h1.rel_volume"] > 0 and m["h1.atr_ratio"] > 0 and m["day_change_pct"] is not None
      and all(m.get(f"cross.{s}.corr_h1") is not None for s in ("GBPUSDm", "USDJPYm", "XAUUSDm")), sorted(m))

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
