---
name: trading-bot-project-state
description: "State of the MT5 AI trading bot as of 2026-09-26 — large uncommitted work, test suites stored outside the repo, open follow-ups"
metadata:
  node_type: memory
  type: project
  originSessionId: 42185519-7770-466e-85fd-b6018ad6b6c9
  modified: 2026-09-26T00:51:08.642Z
---

As of 2026-09-26, everything since commit aa148c5 is **uncommitted**. That includes:
- settings_store.py, the demo/live symbol lists and rule sharing;
- the NVIDIA LLM switch;
- the overextension/weekend/USD guards;
- the 16 logic-flaw fixes: calibration.py, news.py, rule_engine.py, shadow_store.py, risk_state.json persistence, the currency cap and the daily loss budget.

I gave the user a commit message; they commit themselves.

The 10 test suites (test_logic_fixes, test_guards, test_llm_transport, test_rule_sharing, test_symbol_lists, test_settings, test_daily_auditor, test_execution_memory, test_atr_stops, test_pandas_ta_context) and preview_server.py live only in the session scratchpad: `%TEMP%\claude\c--Users-Imran-Alam-Desktop-TR-BOT-self-ritiesh\42185519-7770-466e-85fd-b6018ad6b6c9\scratchpad`. They must redirect RISK_STATE_FILE/SHADOW_FILE/NEWS_CACHE_FILE to temp dirs, or they write into the project folder.

**Why:** the scratchpad is temporary; if it is cleaned, the tests are gone.

**How to apply:** early in the next session, offer to move the tests into a `tests/` folder in the repo (check the scratchpad still exists first).

Other open items:
- The user's NVIDIA API key and MT5 demo password were visible in screenshots; I advised rotating them.
- Only edit `.env` by key name; never print its secrets. A backup is kept as env.backup in the scratchpad.
- Current LLM: nvidia/nemotron-3-super-120b-a12b, with lightning-30b as fallback, via integrate.api.nvidia.com.

Related: [[combined-product-roadmap]]
