---
name: combined-product-roadmap
description: "Planned next phase (as of 2026-09-26) — crypto exchanges via ccxt, TradingView, autoresearch-style strategy research; 4 design questions awaiting the user's decision"
metadata:
  node_type: memory
  type: project
  originSessionId: 42185519-7770-466e-85fd-b6018ad6b6c9
  modified: 2026-09-26T00:50:55.553Z
---

On 2026-09-26 the user paused to decide the design of the next phase. They will come back with answers to these 4 questions. Do not start building before they answer.

1. Spot or futures on Bitget/MEXC?
2. Keep the LLM as the per-trade decider, or switch to "strategy.py rules decide, LLM reviews/vetoes"? I recommended the latter, because LLM decisions cannot be backtested.
3. What comes first: crypto exchanges, or proving an edge with a backtest harness on current markets?
4. Will it run on a VPS, and do they have a paid TradingView plan (needed for webhooks)?

**Why:** the user wants one product that combines their bot with ideas from three repos (see [[reference-trading-repos]]), and asked to trade crypto on Bitget/MEXC and to connect TradingView.

**How to apply:** once they answer, draft a detailed Phase 1 plan. The proposed order was:
1. Commit work, move tests into the repo, rotate exposed keys (see [[trading-bot-project-state]]).
2. Build a backtest harness through the same risk engine, with costs.
3. Add a broker interface: MT5Adapter, PaperAdapter, then CCXTAdapter (Bitget demo trading first).
4. Crypto mode: 24/7 trading day, a combined crypto correlation risk group, funding/fees, exchange-side stops.
5. strategy.py plus an LLM veto.
6. Autoresearch loop (agent edits strategy.py only; train/validation split plus a locked final test period).
7. TradingView webhook endpoint, alerts (Telegram), VPS.
8. Live trading with tiny size after paper-trading criteria are met.

TradingView MCP is for the user's interactive analysis only, never the live feed. For MEXC, verify futures API order access before building on it.
