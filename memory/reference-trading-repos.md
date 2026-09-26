---
name: reference-trading-repos
description: "The three GitHub repos the user wants to combine with their bot, with licences and what to borrow from each (reviewed 2026-09-26)"
metadata:
  node_type: memory
  type: reference
  originSessionId: 42185519-7770-466e-85fd-b6018ad6b6c9
  modified: 2026-09-26T00:51:01.736Z
---

- **github.com/LewisWJackson/tradingview-mcp-jackson** (MIT, Node)
  - MCP server that drives TradingView Desktop via the Chrome DevTools port 9222; 81 tools.
  - Chart and indicator reading, Pine Script dev, replay mode, `morning_brief` from `rules.json`.
  - Cannot place trades. Uses undocumented TradingView internals (terms-of-use risk).
  - Borrow: the interactive analysis workflow only.
- **github.com/jackson-video-resources/claude-tradingview-mcp-trading** (Node, NO licence file, so do not copy code; reimplement the ideas)
  - `bot.js` checks `rules.json` and sends raw REST market orders to Bitget (spot/futures).
  - Paper-trading flag, `MAX_TRADES_PER_DAY`, `trades.csv` tax log, VPS cron on candle close.
  - No SL/TP at all. Sizes as 1% of portfolio notional, not 1% risk at the stop.
- **github.com/karpathy/autoresearch** (MIT)
  - An agent edits only `train.py`, runs a fixed 5-minute experiment, and keeps the change only if `val_bpb` improves.
  - `prepare.py` is fixed; `program.md` is the human's steering brief.
  - Maps to: agent edits `strategy.py`; a fixed backtest harness; a single out-of-sample, after-cost score with a drawdown limit; a locked final test period.

Related: [[combined-product-roadmap]]
