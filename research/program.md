# Research brief — scalper improvement loop

The karpathy/autoresearch pattern, applied to the M5 scalper: an agent (or you) edits **one file**, a
**fixed evaluator** scores it with **one number**, and only improvements are kept. Read this whole
brief before the first experiment.

## Files

| File | Role | Edit? |
|---|---|---|
| `research/candidate.py` | `PARAMS` (strategy knobs), `EXIT_MODE`, `NOTE` | **yes — the only file you edit** |
| `research/evaluate.py` | fixed evaluator: backtest, walk-forward, keep/discard, log | no |
| `research/results.tsv` | every experiment, appended by the evaluator (git-ignored) | no |
| `research/best.json` | the best candidate so far and its score (git-ignored) | no |
| `research/holdout_log.jsonl` | every opening of the locked period (git-ignored) | no |
| `backtest.py`, `scalper.py`, `validation.py` | the harness | no (changes invalidate all earlier scores) |

## The metric

**Walk-forward out-of-sample expectancy, in R per trade, with every spread 1 pip wider than recorded**,
over the research period only. Walk-forward = 12 months to judge → the next 3 months scored (3 → 1 on
short history), rolled forward; only the scored months count. Higher is better.

A candidate is **kept** when it has enough research trades (150 on multi-year data, 40 on short) and
beats the best score by at least **0.01R**. Otherwise it is discarded: restore the best with
`python research/evaluate.py --restore-best` before the next idea.

The log also shows the **Deflated Sharpe Ratio** with the total number of variants ever tried (the
24 tried before this loop plus every row in `results.tsv`). The more experiments, the higher the bar:
prefer few, well-reasoned experiments over many small tweaks.

## The locked year

The most recent 365 days (20% of the history when less than 3 years are loaded) are never scored by
the loop. When the research is done, run `python research/evaluate.py --holdout` **once** on the best
candidate. Positive expectancy there at +1 pip → it may be adopted; otherwise it is not. Every opening
is logged; do not tune after looking at it.

## Loop

1. Read `results.tsv` and `best.json` (what has been tried, what works).
2. Form one hypothesis with a market reason (e.g. "shallower pullbacks catch more trend continuations
   in strong trends"), not a random tweak. Write it in `NOTE`.
3. Edit `PARAMS` / `EXIT_MODE` in `candidate.py` (1–2 knobs per experiment).
4. Run `.venv\Scripts\python.exe research\evaluate.py` (≈1–3 minutes on 5 years).
5. KEPT → continue from it. Discarded → `--restore-best`, try the next idea.
6. Stop after ~20 experiments or when three ideas in a row fail; then run the holdout once.

## Knobs (`evaluate.ALLOWED`)

Pullback: `SCALP_RSI_PULLBACK` (10–50), `SCALP_PULLBACK_BARS` (1–12), `SCALP_PULLBACK_TO_EMA`,
`SCALP_TRIGGER` (RSI | BREAK). Trend filters: `SCALP_MEDIUM_TREND`, `SCALP_MIN_ADX` (0–60),
`SCALP_STRICT_GUARD`. Stops/targets: `SCALP_SL_ATR_MIN`, `SCALP_SL_ATR_MAX`, `SCALP_SWING_BARS`,
`SCALP_REWARD_RISK` (1–5), `SCALP_TARGET` (RR | STRUCTURE), `SCALP_MIN_TARGET_R`, `SCALP_ROOM_MIN_R`,
`SCALP_ROOM_BARS`. Timing: `SCALP_TIME_STOP_MINUTES`, `SCALP_MAX_TRADES_PER_SYMBOL`,
`SCALP_SESSION_START_LONDON`, `SCALP_SESSION_END_NEW_YORK`, `LOSS_COOLDOWN_MINUTES`.
Exits: `EXIT_MODE` FIXED | BREAKEVEN | PARTIAL | TRAIL.

Already rejected on 150 days (retest only with a new reason): ADX filter, pullback-to-EMA, break
trigger, room/structure targets, tight scalp size. Breakeven/partial/trail lost to FIXED on 150 days.

## Rules

- Never edit the evaluator or the harness during a research run. Never touch `.env`, `settings.json`
  or live trading: adopting a result is a separate, manual step after the holdout passes.
- Risk per trade, kill-switch and position sizing are not research knobs (they change money, not edge).
- Scores are only comparable on the same data (source, days, symbols); the evaluator starts a new
  baseline automatically when the data changes (e.g. when the 5-year download completes).
