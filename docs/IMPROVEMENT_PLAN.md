# Improvement Plan — 3 Phases

Goal: turn the M5 scalper into a professionally validated system that learns from every
decision, using 5 years of Dukascopy data and a machine-learning filter (meta-labeling).

Current baseline (150 trading days, broker spreads): 107 trades, +0.28R/trade, SQN 2.54,
max drawdown ~9%. The edge is not yet statistically proven (t = 2.62; ~2.87 needed after
~24 variants were tried). Rules: D1 EMA200 + EMA20/50 trend, M15 momentum, M5 RSI pullback
and turn, strict overextension guard, AI confirm/veto, 1.5R target, 60-minute time stop.

**Rule for every change:** it is switched on only if it improves results on data it was not
tuned on (walk-forward) and stays positive with +1 pip of extra cost.

---

## Phase 1 — Data & learning foundation

Everything the bot sees is saved, and 5 years of realistic history is available.

| # | Task | Files |
|---|---|---|
| 1.1 | Move the test suites into the repo (`tests/`) with a single runner | `tests/`, `run_tests.py` |
| 1.2 | **Decision journal**: save every M5 evaluation (pair, stage, reason, all features, AI answer, outcome link) to a daily file | `journal.py`, `main.py` |
| 1.3 | **Shadow trades feed the learning**: the auditor and confidence calibration also use resolved shadow trades (weighted below real trades) | `auditor.py`, `calibration.py` |
| 1.4 | **Dukascopy downloader**: 5 years of bid/ask ticks → M5/M15/H1/D1 bars with real spreads, for the 8 pairs; resumable, cached in `data/dukascopy/` (git-ignored) | `dukascopy.py` |
| 1.5 | Backtest can run on Dukascopy history as well as MT5 history (`--source dukascopy`) | `backtest.py` |

**Done when:** the journal records every bar live; the 5-year data is on disk; a 5-year backtest
runs; all tests pass from `run_tests.py`.

## Phase 2 — Professional validation & risk

Every result is judged the way professionals judge it, and the account is protected.

| # | Task | Files |
|---|---|---|
| 2.1 | **Rolling walk-forward** (train window → next month, rolled forward), **Monte Carlo** drawdowns and a **multiple-testing-aware significance** (deflated t / Sharpe) in every backtest report | `backtest.py`, `validation.py` |
| 2.2 | Cost realism: slippage and commission settings, stressed runs | `backtest.py`, `config.py` |
| 2.3 | USD-direction guard simulated in the backtest (time-aligned bars of all pairs) | `backtest.py` |
| 2.4 | **Total-drawdown kill-switch** (default 10% from peak) and **risk throttle** (half risk after −5%) | `main.py`, `config.py` |
| 2.5 | Exit variants tested on 5 years: breakeven at +1R, partial exit + trailing stop; adopted only if they pass | `scalper.py`, `backtest.py` |
| 2.6 | Dashboard: walk-forward / Monte Carlo / significance in the Backtest panel; kill-switch in Guardrails and Settings | `templates/index.html` |
| 2.7 | **Research loop (karpathy/autoresearch pattern)**: `research/program.md` (the brief), one editable candidate file (scalper parameters and rule switches), a fixed evaluator (5-year walk-forward at +1 pip, trial-count penalty) and a results log; a change is kept only if it beats the current best and never sees the locked final year | `research/` |

**Done when:** the backtest report shows walk-forward, Monte Carlo and significance; the
kill-switch and throttle are live and tested; exit variants are decided on data.

## Phase 3 — Machine learning (meta-labeling)

A model learns which setups to take, instead of hand-set thresholds.

| # | Task | Files |
|---|---|---|
| 3.1 | **Dataset builder**: every scalper setup over 5 years → ~30 setup-time features + triple-barrier label (TP / SL / time) and R result; **plus every live real trade and resolved shadow trade** (AI vetoes, guard/strict-guard skips, penalty blocks) with the same features from the decision journal, tagged by source and weighted | `ml/dataset.py` |
| 3.2 | **Models**: LightGBM classifier and a logistic-regression baseline; probabilities calibrated | `ml/train.py` |
| 3.3 | **Validation**: purged walk-forward with an embargo; acceptance = beats both the baseline and "no model" out of sample, at +1 pip, with multiple-testing correction | `ml/validate.py` |
| 3.4 | **Live filter**: take a setup only if predicted expected R after costs > 0; switch in Settings; falls back to plain rules if no approved model | `ml/model.py`, `main.py` |
| 3.5 | **Monthly retrain + drift monitor** (live vs predicted win rate) with automatic fallback; each retrain adds the new live and shadow trades | `ml/train.py`, `main.py` |
| 3.6a | **AI-veto check**: compare what vetoed setups (shadows) would have made vs setups the AI confirmed; if the vetoes cost money over enough trades, the dashboard recommends switching the AI to advisory-only | `ml/validate.py`, `templates/index.html` |
| 3.6 | Dashboard ML panel (model status, win probability per setup, drift) and Telegram alerts for fills, kill-switch and drift | `templates/index.html`, `alerts.py` |

**Done when:** a model passes the acceptance test (or we have the evidence that it does not
help, and it stays off); live setups show a win probability; retraining and drift fallback work.

**New dependencies:** `lightgbm`, `scikit-learn` (Phase 3).

---

## Decisions already made

- Strategy: SCALP mode, London 07:00 → New York 16:00 entries, 60-minute time stop.
- Adopted filters: strict overextension guard, D1 medium-term trend. Rejected: ADX, pullback-to-EMA,
  break trigger, room/structure targets, tight scalp size (did not hold up out of sample).
- ML: meta-labeling with LightGBM + logistic baseline. No LSTM / deep learning / RL / LLM fine-tuning.
- Data: Dukascopy, 5 years, 8 pairs (EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD, XAUUSD).
- Kaggle/GitHub datasets reviewed: unsuitable (daily-only, pre-2020, no spreads, or price forecasting).
- Shadow trades (every rejected setup, followed on price data to WIN / LOSS / TIMEOUT) are learning data
  everywhere: auditor and calibration (1.3), ML training and retraining (3.1, 3.5), and judging the AI veto (3.6a).
  "No setup" bars have no trade outcome, so they are journaled for features and drift monitoring, not labels.

## Status

| Phase | Status | Notes |
|---|---|---|
| 1 | **Code done 2026-09-30**; 5-year download in progress | 1.1 tests in `tests/` + `run_tests.py` (12 suites). 1.2 `journal.py` + `scalper.features()` (31 features, side-signed), every scalp/swing evaluation journaled with AI answer, action, ticket, shadow id. 1.3 shadow trades carry a market snapshot + features; auditor window = real + shadow trades (weight `SHADOW_WEIGHT` 0.5, evidence in R); calibration counts shadows. 1.4 `dukascopy.py` (download/build/status; bars on the NY+7 server clock; resumable). 1.5 `backtest.py --source dukascopy`, dashboard source selector. Verified vs MT5 on EURUSD: bar times identical, M5 closes within 0.1 pip median, scalper stage identical on 98.8% of 8,065 bars. Dukascopy rate-limits this IP (503s), so the download runs patiently (1 worker) and can take a day+; rerun `python dukascopy.py download --years 5` anytime, then `python dukascopy.py build`. |
| 2 | **Code done 2026-09-30** (2.1–2.7); decisions on 5 years pending the download | 2.1 `validation.py`: monthly stability, Monte Carlo (5,000 runs, P(hit kill-switch)), t / SQN / PSR / **Deflated Sharpe** with `BACKTEST_TRIALS`, rolling walk-forward (12→3 months, 3→1 on short data) with variant selection; verdict needs DSR ≥ 0.95, 60% positive months, positive recent 30% and positive at +1 pip. 2.2 `--slippage` (pips, stops/time exits and entries) and `--commission` (per lot, converted to R); automatic +1 pip stress in every report. 2.3 USD guard measured with time-aligned day moves of all pairs (`usd_guard` split). 2.4 live kill-switch (10% from the equity peak, persisted per login in `risk_state.json`, closes bot trades, manual reset in Settings) + half-risk throttle past −5%; the backtest account runs with and without it. 2.5 exits FIXED / BREAKEVEN / PARTIAL / TRAIL simulated on the same setups + walk-forward pick. 2.6 dashboard: Backtest tiles (DSR, walk-forward, +1 pip, Monte Carlo, protection) + exit/USD/month line; Guardrails kill-switch row; Settings "Account protection". 2.7 `research/` (program.md, candidate.py, evaluate.py; locked final year / 20%). Tests: `test_phase2_risk` (36 checks). **Broker-history results (330 days, May 2025–Sep 2026):** 194 trades, +0.09R, PF 1.2, t 1.16, DSR 0.21 (not proven); lost Oct 2025–Feb 2026, won Mar–Aug 2026; +1 pip → +0.065R; FIXED beat every exit variant (walk-forward picked FIXED 13/13); USD guard: no evidence either way (8 trades); at 2% risk Monte Carlo median max DD 20%, 99% of runs hit the 10% kill-switch (the real path tripped it Oct 2025). Research baseline (+1 pip, before the locked 2026-06-22+): −0.054R, walk-forward −0.002R. |
| 3 | Not started | |
