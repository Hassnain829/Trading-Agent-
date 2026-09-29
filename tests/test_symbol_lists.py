"""Demo/live symbol lists: broker name matching, account-type switching, settings validation, migration."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project folder
import config
config.JOURNAL_DIR = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "journal"  # tests never write the real journal

tmp = Path(tempfile.mkdtemp())
config.RISK_STATE_FILE, config.SHADOW_FILE, config.NEWS_CACHE_FILE = tmp / "risk_state.json", tmp / "shadow.json", tmp / "news.json"
for attr, name in (("MEMORY_FILE", "memory.json"), ("RULES_FILE", "new_rules.json"), ("SETTINGS_FILE", "settings.json")):
    setattr(config, attr, tmp / name)
config.SYMBOLS_DEMO = ["EURUSD", "GBPUSD", "XAUUSD", "BTCUSD"]
config.SYMBOLS_LIVE = ["EURUSD", "USDJPY", "XAUUSD"]

import data_engine
import main
import settings_store
from fastapi import HTTPException

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- name matching
EXNESS = ["EURUSDm", "GBPUSDm", "USDJPYm", "XAUUSDm", "BTCUSDm", "US30m"]
METAQUOTES = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AMD", "MSFT"]
RAW = ["EURUSD.r", "USDJPY.r", "XAUUSD+", "GBPUSD"]
r, m, u = data_engine.resolve_symbols(["EURUSD", "GBPUSD", "XAUUSD"], EXNESS)
check("Plain names -> Exness 'm' names", r == ["EURUSDm", "GBPUSDm", "XAUUSDm"] and not u, m)
r, m, u = data_engine.resolve_symbols(["EURUSDm", "XAUUSDm", "BTCUSDm"], METAQUOTES)
check("Exness names -> MetaQuotes plain names; missing BTC reported", r == ["EURUSD", "XAUUSD"] and u == ["BTCUSDm"], (r, u))
r, m, u = data_engine.resolve_symbols(["EURUSD", "USDJPY", "XAUUSD", "GBPUSD"], RAW)
check("Suffixes .r / + matched, exact name preferred", r == ["EURUSD.r", "USDJPY.r", "XAUUSD+", "GBPUSD"], r)
r, m, u = data_engine.resolve_symbols(["EURUSD"], ["EURUSDm", "EURUSD", "EURUSD.pro"])
check("Exact match wins over suffixed variants", r == ["EURUSD"])
r, m, u = data_engine.resolve_symbols(["eurusd", "EURUSD"], ["EURUSDm"])
check("Case-insensitive and de-duplicated", r == ["EURUSDm"] and m == {"eurusd": "EURUSDm", "EURUSD": "EURUSDm"})
r, m, u = data_engine.resolve_symbols(["US30", "AMD"], ["US30m", "AMDX"])
check("Short names only match exactly (no US30 -> US30m / AMD -> AMDX guessing)", r == [] and u == ["US30", "AMD"], r)

# ---------------------------------------------------------------- account-type switching
accounts = {"demo": {"login": 111, "server": "MetaQuotes-Demo", "account_mode": "DEMO"},
            "live": {"login": 222, "server": "Exness-MT5Real", "account_mode": "LIVE"}}
brokers = {"demo": METAQUOTES, "live": EXNESS}
state = {"who": "demo"}
data_engine.get_account_snapshot = lambda: dict(accounts[state["who"]], equity=100.0, balance=100.0, profit=0.0,
                                                currency="USD", company="x", name="x", leverage=100, margin=0.0,
                                                margin_free=100.0, margin_level=0.0, trade_allowed=True)
data_engine.broker_symbols = lambda: [{"name": n, "description": n, "path": f"Forex\\Majors\\{n}"} for n in brokers[state["who"]]]

main._refresh_symbol_universe(force=True)
check("DEMO account -> demo list resolved (BTCUSD skipped on MetaQuotes)",
      config.SYMBOLS == ["EURUSD", "GBPUSD", "XAUUSD"] and main.bot_state["account_mode"] == "DEMO"
      and main.bot_state["unresolved_symbols"] == ["BTCUSD"], config.SYMBOLS)
state["who"] = "live"
main._refresh_symbol_universe()
check("Logging into a LIVE account switches to the live list and its broker names",
      config.SYMBOLS == ["EURUSDm", "USDJPYm", "XAUUSDm"] and main.bot_state["account_mode"] == "LIVE", config.SYMBOLS)
check("Status payload trades the resolved live symbols", main._status_payload()["symbols"] == ["EURUSDm", "USDJPYm", "XAUUSDm"])
state["who"] = "demo"
main._refresh_symbol_universe()
check("Back to DEMO", config.SYMBOLS == ["EURUSD", "GBPUSD", "XAUUSD"])

# ---------------------------------------------------------------- settings validation per list
try:
    main.api_save_settings(main.SettingsUpdate(symbols_demo=["EURUSD", "EURUSDm", "NOTREAL"]))
    check("Connected (demo) list: unknown symbol rejected", False)
except HTTPException as exc:
    check("Connected (demo) list: only truly unknown names rejected", exc.status_code == 422 and "NOTREAL" in exc.detail
          and "EURUSDm" not in exc.detail, exc.detail)
res = main.api_save_settings(main.SettingsUpdate(symbols_live=["EURUSD", "GBPUSD", "NAS100"]))
check("Other (live) list saved unchecked, with a warning", res["settings"]["symbols_live"] == ["EURUSD", "GBPUSD", "NAS100"]
      and any("LIVE list will be matched" in w for w in res["warnings"]), res["warnings"])
res = main.api_save_settings(main.SettingsUpdate(symbols_demo=["EURUSD", "USDJPY", "eurusd"]))
check("Saving the demo list updates the traded symbols immediately", config.SYMBOLS == ["EURUSD", "USDJPY"]
      and res["active_symbols"] == ["EURUSD", "USDJPY"] and res["settings"]["symbols_demo"] == ["EURUSD", "USDJPY"])
check("Settings payload carries account + mode", res["account"]["mode"] == "DEMO" and res["account"]["server"] == "MetaQuotes-Demo")
saved = json.loads(config.SETTINGS_FILE.read_text())
check("Both lists persisted", saved["symbols_live"] == ["EURUSD", "GBPUSD", "NAS100"] and "symbols_demo" in saved)

sym = main.api_symbols()
check("/api/symbols reports account type", sym["account_mode"] == "DEMO" and sym["server"] == "MetaQuotes-Demo")

# ---------------------------------------------------------------- migration of the old single list
config.SETTINGS_FILE.write_text(json.dumps({"symbols": ["EURUSD", "GBPUSD"], "max_open_positions": 2}))
settings_store.load_saved()
check("Old settings.json 'symbols' becomes the demo list", config.SYMBOLS_DEMO == ["EURUSD", "GBPUSD"]
      and config.MAX_OPEN_POSITIONS == 2)
settings_store.save({"max_open_positions": 3})
check("Next save drops the legacy key", "symbols" not in json.loads(config.SETTINGS_FILE.read_text()))

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILURE(S): {failures}"))
sys.exit(1 if failures else 0)
