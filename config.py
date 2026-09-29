"""
Central configuration for the Autonomous AI Hedge Fund.

Values are loaded from the process environment (optionally populated from a
local ``.env`` file) and coerced into strongly typed module-level constants
with safe defaults. Invalid values never crash the import: they fall back to
the default and are collected in ``CONFIG_WARNINGS`` so the server can log
them once logging is configured.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import MetaTrader5 as mt5
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MEMORY_FILE: Path = BASE_DIR / "memory.json"
RULES_FILE: Path = BASE_DIR / "new_rules.json"
TEMPLATES_DIR: Path = BASE_DIR / "templates"
LOCK_FILE: Path = BASE_DIR / ".server.lock"

CONFIG_WARNINGS: List[str] = []


# -----------------------------------------------------------------------------
# Typed environment readers
# -----------------------------------------------------------------------------
def _is_placeholder(value: str) -> bool:
    """True for empty values and the ``your_...`` placeholders from .env.example."""
    return not value or value.lower().startswith("your_")


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().strip('"').strip("'")
    return default if _is_placeholder(value) else value


def _clamp_number(name: str, value, minimum, maximum):
    if minimum is not None and value < minimum:
        CONFIG_WARNINGS.append(f"{name}={value} is below the minimum {minimum}; clamped")
        return minimum
    if maximum is not None and value > maximum:
        CONFIG_WARNINGS.append(f"{name}={value} is above the maximum {maximum}; clamped")
        return maximum
    return value


def _env_int(name: str, default: int, minimum: Optional[int] = None,
             maximum: Optional[int] = None) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        value = int(float(raw))
    except ValueError:
        CONFIG_WARNINGS.append(f"{name}={raw!r} is not an integer; using default {default}")
        return default
    return _clamp_number(name, value, minimum, maximum)


def _env_float(name: str, default: float, minimum: Optional[float] = None,
               maximum: Optional[float] = None) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        CONFIG_WARNINGS.append(f"{name}={raw!r} is not a number; using default {default}")
        return default
    return _clamp_number(name, value, minimum, maximum)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    CONFIG_WARNINGS.append(f"{name}={raw!r} is not a boolean; using default {default}")
    return default


def _env_list(name: str, default: str) -> List[str]:
    raw = _env_str(name) or default
    seen: Dict[str, None] = {}
    for item in raw.split(","):
        symbol = item.strip()
        if symbol and symbol not in seen:
            seen[symbol] = None
    return list(seen)


# -----------------------------------------------------------------------------
# LLM (any OpenAI-compatible endpoint: DeepSeek, NVIDIA NIM, OpenRouter, ...)
# LLM_* names are preferred; the older DEEPSEEK_* names still work.
# -----------------------------------------------------------------------------
DEEPSEEK_API_KEY: str = _env_str("LLM_API_KEY") or _env_str("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE: str = (_env_str("LLM_API_BASE") or _env_str("DEEPSEEK_API_BASE")
                          or "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL: str = _env_str("LLM_MODEL") or _env_str("DEEPSEEK_MODEL") or "deepseek-chat"
# Tried in order when the main model times out, is overloaded (429/5xx), unknown (404) or answers empty.
LLM_FALLBACK_MODELS: List[str] = [m for m in _env_list("LLM_FALLBACK_MODELS", "") if m != DEEPSEEK_MODEL]
# Free/shared endpoints can queue requests for a long time; reasoning models need room to think.
DEEPSEEK_TIMEOUT_SECONDS: int = _env_int("LLM_TIMEOUT_SECONDS", 120, 10, 600)
LLM_MAX_TOKENS: int = _env_int("LLM_MAX_TOKENS", 4096, 256, 65_536)
# "off" asks reasoning models to skip their thinking phase (faster); "default" leaves the model as is.
_reasoning = _env_str("LLM_REASONING", "default").lower()
LLM_REASONING: str = _reasoning if _reasoning in ("default", "off") else "default"
DEEPSEEK_MAX_RETRIES: int = 2
AI_TEMPERATURE: float = 0.1

# -----------------------------------------------------------------------------
# MetaTrader 5
# -----------------------------------------------------------------------------
MT5_LOGIN: int = _env_int("MT5_LOGIN", 0, minimum=0)
MT5_PASSWORD: str = _env_str("MT5_PASSWORD")
MT5_SERVER: str = _env_str("MT5_SERVER", "Exness-MT5Trial7")
MT5_PATH: Optional[str] = _env_str("MT5_PATH") or None
MT5_TIMEOUT_MS: int = 60_000

# -----------------------------------------------------------------------------
# Engine & risk
# -----------------------------------------------------------------------------
MAX_RISK_PERCENT: float = 5.0
DEFAULT_RISK_PERCENT: float = _env_float("DEFAULT_RISK_PERCENT", 1.0, 0.01, MAX_RISK_PERCENT)
SCAN_INTERVAL_SECONDS: int = _env_int("SCAN_INTERVAL_SECONDS", 30, 5, 86_400)
CONFIDENCE_THRESHOLD: int = _env_int("CONFIDENCE_THRESHOLD", 65, 0, 100)
# Separate lists for demo and live accounts; the list matching the logged-in account's
# type is traded. Plain names (EURUSD) are matched to the broker's own spelling
# (EURUSDm, EURUSD.r, ...) automatically. SYMBOLS is the legacy fallback for both.
_DEFAULT_SYMBOLS = _env_str("SYMBOLS") or "EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,BTCUSD,XAUUSD"
SYMBOLS_DEMO: List[str] = _env_list("SYMBOLS_DEMO", _DEFAULT_SYMBOLS)
SYMBOLS_LIVE: List[str] = _env_list("SYMBOLS_LIVE", _DEFAULT_SYMBOLS)
MAX_SYMBOLS: int = 40
# The symbols actually traded right now: the matching list, resolved to broker names on connect.
SYMBOLS: List[str] = list(SYMBOLS_DEMO)
# The logged-in account ({"mode": "DEMO"|"LIVE", "broker": company, "server": ...}), set on connect.
ACTIVE_ACCOUNT: Dict[str, Optional[str]] = {"mode": None, "broker": None, "server": None}

# Which learned rules apply to the account in use:
#   PAIR   - a rule applies to the same pair at any broker and on demo or live (EURUSD rule -> EURUSDm)
#   BROKER - a rule applies to the same pair only at the broker it was learned at (demo or live)
_rule_scope = _env_str("RULE_SCOPE", "PAIR").upper()
if _rule_scope not in ("PAIR", "BROKER"):
    CONFIG_WARNINGS.append(f"RULE_SCOPE={_rule_scope!r} must be PAIR or BROKER; using PAIR")
    _rule_scope = "PAIR"
RULE_SCOPE: str = _rule_scope

MAX_OPEN_POSITIONS: int = _env_int("MAX_OPEN_POSITIONS", 5, 0, 500)
MAX_DAILY_LOSS_PERCENT: float = _env_float("MAX_DAILY_LOSS_PERCENT", 5.0, 0.0, 100.0)
# Stop opening new trades for the rest of the trading day (resets 17:00 New York) once equity is up this much (0 = off).
MAX_DAILY_PROFIT_PERCENT: float = _env_float("MAX_DAILY_PROFIT_PERCENT", 0.0, 0.0, 1000.0)
# RISK: size each trade so the stop loses DEFAULT_RISK_PERCENT of equity. FIXED: always trade FIXED_LOT.
_sizing_mode = _env_str("POSITION_SIZING_MODE", "RISK").upper()
if _sizing_mode not in ("RISK", "FIXED"):
    CONFIG_WARNINGS.append(f"POSITION_SIZING_MODE={_sizing_mode!r} must be RISK or FIXED; using RISK")
    _sizing_mode = "RISK"
POSITION_SIZING_MODE: str = _sizing_mode
FIXED_LOT: float = _env_float("FIXED_LOT", 0.01, 0.01, 100.0)
AUTO_START_ENGINE: bool = _env_bool("AUTO_START_ENGINE", False)

# Protection guards applied by the engine itself after the AI answers (also in the Settings drawer).
# Overextension: cut confidence when price is stretched far from EMA200 / RSI extreme in the trade direction.
OVEREXTENSION_GUARD: bool = _env_bool("OVEREXTENSION_GUARD", True)
# No new entries this many hours before the Friday forex close (17:00 New York); 0 = off. Crypto is exempt.
WEEKEND_ENTRY_CUTOFF_HOURS: float = _env_float("WEEKEND_ENTRY_CUTOFF_HOURS", 3.0, 0.0, 48.0)
# Close the engine's own non-crypto positions shortly before the Friday close (avoids weekend gaps).
WEEKEND_CLOSE: bool = _env_bool("WEEKEND_CLOSE", False)
WEEKEND_CLOSE_MINUTES: int = 45
# Upper bound for the combined confidence penalty of all guards.
GUARD_MAX_PENALTY: int = 30

# Strategy.
#   SCALP - rules find M5 pullback setups in the D1 trend (M15 confirms); the AI confirms or vetoes each one.
#   SWING - the AI judges every symbol on each closed H1 bar (the original mode).
_strategy_mode = _env_str("STRATEGY_MODE", "SCALP").upper()
if _strategy_mode not in ("SCALP", "SWING"):
    CONFIG_WARNINGS.append(f"STRATEGY_MODE={_strategy_mode!r} must be SCALP or SWING; using SCALP")
    _strategy_mode = "SCALP"
STRATEGY_MODE: str = _strategy_mode
# Scalps are closed after this long if neither the stop nor the target was hit.
SCALP_TIME_STOP_MINUTES: int = _env_int("SCALP_TIME_STOP_MINUTES", 60, 5, 1440)
# At most this many scalps per symbol per trading day.
SCALP_MAX_TRADES_PER_SYMBOL: int = _env_int("SCALP_MAX_TRADES_PER_SYMBOL", 4, 1, 50)
# Entry window for new scalps, DST-aware: from this hour London time until this hour New York time.
# 16 = last entries at 16:00 NY, so the 60-minute time stop has closed every scalp before the 17:00 NY
# rollover (the New York close), when spreads blow out.
SCALP_SESSION_START_LONDON: int = _env_int("SCALP_SESSION_START_LONDON", 7, 0, 23)
SCALP_SESSION_END_NEW_YORK: int = _env_int("SCALP_SESSION_END_NEW_YORK", 16, 1, 17)
# Setup geometry (in M5 ATR14): stop beyond the recent swing, clamped to [min, max]; target = stop x reward/risk.
SCALP_REWARD_RISK: float = _env_float("SCALP_REWARD_RISK", 1.5, 1.0, 5.0)
SCALP_SL_ATR_MIN: float = 1.0
SCALP_SL_ATR_MAX: float = 2.0
SCALP_SWING_BARS: int = 6
# Pullback: M5 RSI14 dipped below this (above 100 - this for shorts) within SCALP_PULLBACK_BARS, then turned back.
SCALP_RSI_PULLBACK: float = _env_float("SCALP_RSI_PULLBACK", 40.0, 10.0, 50.0)
SCALP_PULLBACK_BARS: int = 4
# Refinements, each backtested on 150 trading days (selected on the first 2/3, confirmed on the last 1/3,
# with and without +1 pip of extra cost). ON by default because they held up:
# skip setups the overextension guard would penalise (price already stretched from the H1/D1 EMA200 or RSI
# extreme) before asking the AI; the only scalps that were profitable in both periods;
SCALP_STRICT_GUARD: bool = _env_bool("SCALP_STRICT_GUARD", True)
# the D1 medium-term trend (close vs EMA50, EMA20 vs EMA50) must agree with the EMA200 trend.
SCALP_MEDIUM_TREND: bool = _env_bool("SCALP_MEDIUM_TREND", True)
# OFF by default (looked good on the selection period but did not hold up, or changed nothing):
# minimum D1 ADX14 trend strength (0 = off);
SCALP_MIN_ADX: float = _env_float("SCALP_MIN_ADX", 0.0, 0.0, 60.0)
# the pullback must reach the M5 EMA20 (a real retracement to value, not just an RSI dip);
SCALP_PULLBACK_TO_EMA: bool = _env_bool("SCALP_PULLBACK_TO_EMA", False)
# entry trigger: RSI = RSI turns on a trend-coloured candle; BREAK = that candle also closes beyond
# the previous candle's high (buy) / low (sell), a price-action confirmation that the pullback ended;
_trigger = _env_str("SCALP_TRIGGER", "RSI").upper()
SCALP_TRIGGER: str = _trigger if _trigger in ("RSI", "BREAK") else "RSI"
# skip when the recent swing high/low (last SCALP_ROOM_BARS M5 bars) is closer than this many R (0 = off);
SCALP_ROOM_MIN_R: float = _env_float("SCALP_ROOM_MIN_R", 0.0, 0.0, 5.0)
SCALP_ROOM_BARS: int = 36
# target: RR = stop x SCALP_REWARD_RISK; STRUCTURE = just before the recent swing high/low, capped at
# SCALP_REWARD_RISK and at least SCALP_MIN_TARGET_R (else no trade).
_target = _env_str("SCALP_TARGET", "RR").upper()
SCALP_TARGET: str = _target if _target in ("RR", "STRUCTURE") else "RR"
SCALP_MIN_TARGET_R: float = 1.0
# The AI may take a while: refuse the order if price moved more than this many M5 ATRs meanwhile.
SCALP_MAX_DRIFT_ATR: float = 1.0
# Backtest realism and validation.
# Extra cost per fill in points (entry and stop exits), on top of the recorded spread.
BACKTEST_SLIPPAGE_POINTS: float = _env_float("BACKTEST_SLIPPAGE_POINTS", 0.0, 0.0, 100.0)
# Round-trip commission per standard lot in account currency (0 for spread-only accounts).
BACKTEST_COMMISSION_PER_LOT: float = _env_float("BACKTEST_COMMISSION_PER_LOT", 0.0, 0.0, 100.0)
# How many strategy variants have been tried so far (the Deflated Sharpe Ratio raises the bar with it).
BACKTEST_TRIALS: int = _env_int("BACKTEST_TRIALS", 24, 1, 100_000)
# Account protection (live engine and backtest): stop new entries at this drawdown from peak equity;
# trade at half risk beyond the throttle level. 0 turns either off.
MAX_TOTAL_DRAWDOWN_PERCENT: float = _env_float("MAX_TOTAL_DRAWDOWN_PERCENT", 10.0, 0.0, 90.0)
DRAWDOWN_THROTTLE_PERCENT: float = _env_float("DRAWDOWN_THROTTLE_PERCENT", 5.0, 0.0, 90.0)
THROTTLE_RISK_FACTOR: float = 0.5
KILL_SWITCH_CLOSE_POSITIONS: bool = _env_bool("KILL_SWITCH_CLOSE_POSITIONS", True)
# Machine-learning setup filter (meta-labeling, Phase 3). Datasets, validation reports and the model
# live in data/ml/. The live filter only ever uses a model that passed the walk-forward acceptance test.
ML_DIR: Path = BASE_DIR / "data" / "ml"
ML_FILTER: bool = _env_bool("ML_FILTER", False)  # use an approved model to skip low-probability setups
ML_MIN_GAIN_R: float = 0.02  # a model must beat "no model" and the other model by this much R per trade
ML_MIN_DSR: float = 0.95  # and its selected trades must pass the Deflated Sharpe test
# Decision journal: every evaluation with its features and outcome link (data/journal/*.jsonl).
JOURNAL_ENABLED: bool = _env_bool("JOURNAL_ENABLED", True)
JOURNAL_DIR: Path = BASE_DIR / "data" / "journal"
# Shadow trades count this much relative to a real trade in the auditor and calibration.
SHADOW_WEIGHT: float = _env_float("SHADOW_WEIGHT", 0.5, 0.0, 1.0)
BACKTEST_DIR: Path = BASE_DIR / "backtests"

# Portfolio risk. Open risk = what every open position loses if its stop is hit from the current price.
# Max open risk in one direction of one currency (long USD, short EUR, ...), in % of equity; 0 = off.
MAX_CURRENCY_RISK_PERCENT: float = _env_float("MAX_CURRENCY_RISK_PERCENT", 2.5, 0.0, 50.0)
# A new trade is refused when today's loss + all open risk + the new trade could exceed MAX_DAILY_LOSS_PERCENT.
DAILY_LOSS_RISK_BUDGET: bool = _env_bool("DAILY_LOSS_RISK_BUDGET", True)
# Close the engine's own positions when the daily loss limit trips (manual trades are never touched).
DAILY_LOSS_CLOSE_POSITIONS: bool = _env_bool("DAILY_LOSS_CLOSE_POSITIONS", True)
# A position without a stop loss counts as this much open risk (its real risk is unbounded).
UNPROTECTED_POSITION_RISK_PERCENT: float = MAX_RISK_PERCENT
RISK_WATCHDOG_SECONDS: int = 5
RISK_STATE_FILE: Path = BASE_DIR / "risk_state.json"

# Entry discipline.
# Ask the AI again only when a new H1 bar has closed (or price moved AI_REEVALUATE_ATR_MOVE x ATR since).
AI_NEW_BAR_ONLY: bool = _env_bool("AI_NEW_BAR_ONLY", True)
AI_REEVALUATE_ATR_MOVE: float = 0.5
# No new trade in the same direction on a symbol for this long after a losing close; 0 = off.
LOSS_COOLDOWN_MINUTES: int = _env_int("LOSS_COOLDOWN_MINUTES", 120, 0, 10_080)
# Flipping an open position needs this much more confidence than a fresh entry.
REVERSAL_EXTRA_CONFIDENCE: int = _env_int("REVERSAL_EXTRA_CONFIDENCE", 10, 0, 50)
# Refuse the order if price moved more than this many H1 ATRs while the AI was deciding.
MAX_ENTRY_DRIFT_ATR: float = _env_float("MAX_ENTRY_DRIFT_ATR", 0.5, 0.05, 5.0)

# Economic calendar guard: no new entries around high-impact news for either currency of a symbol.
NEWS_GUARD: bool = _env_bool("NEWS_GUARD", True)
NEWS_BLOCK_BEFORE_MINUTES: int = _env_int("NEWS_BLOCK_BEFORE_MINUTES", 30, 0, 720)
NEWS_BLOCK_AFTER_MINUTES: int = _env_int("NEWS_BLOCK_AFTER_MINUTES", 15, 0, 720)
NEWS_CALENDAR_URL: str = (_env_str("NEWS_CALENDAR_URL")
                          or "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
NEWS_REFRESH_HOURS: float = 4.0
NEWS_CACHE_FILE: Path = BASE_DIR / "news_cache.json"

# Confidence calibration: raise the threshold automatically when executed trades at that confidence
# have lost money (needs CALIBRATION_MIN_TRADES closed trades); never lowers it.
CALIBRATE_THRESHOLD: bool = _env_bool("CALIBRATE_THRESHOLD", True)
CALIBRATION_MIN_TRADES: int = 10
CALIBRATION_MAX_THRESHOLD: int = 85

# The dashboard's Settings drawer saves overrides for the values above here;
# they win over .env until "Reset to .env" is pressed.
SETTINGS_FILE: Path = BASE_DIR / "settings.json"

# Orders placed by this engine carry this magic number; the reversal guard only
# ever closes positions that carry it, so manual trades are never touched.
MAGIC_NUMBER: int = 20_260_925
ORDER_DEVIATION_POINTS: int = 20
ORDER_COMMENT: str = "AI-HF"
# Refuse a trade when the broker's minimum lot would risk more than this
# multiple of the intended risk amount (1.25 = at most 25% over budget).
MIN_LOT_RISK_TOLERANCE: float = _env_float("MIN_LOT_RISK_TOLERANCE", 1.25, 1.0, 3.0)
# A symbol whose last tick has not changed for this long is treated as closed.
MARKET_IDLE_SECONDS: int = 180

# -----------------------------------------------------------------------------
# Market data & indicators
# -----------------------------------------------------------------------------
TIMEFRAMES: Dict[str, int] = {"H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}
BARS_TO_FETCH: int = 250

EMA_PERIOD: int = 200
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
VOL_MA_PERIOD: int = 20
ATR_BASELINE_PERIOD: int = 50
STRUCTURE_LOOKBACK: int = 10

# ATR stop engine: the AI chooses stop/target multiples of H1 ATR(14) inside
# these bounds and the engine converts them into exact prices.
SL_ATR_MULTIPLIER: float = 1.5  # used when the model gives no usable multiple
TP_ATR_MULTIPLIER: float = 3.0
SL_ATR_MIN: float = 1.0
SL_ATR_MAX: float = 3.0
TP_ATR_MIN: float = 1.5
TP_ATR_MAX: float = 6.0
MIN_REWARD_RISK: float = 1.5
# HOLD when the spread would consume more than this fraction of the stop distance.
MAX_SPREAD_TO_STOP: float = 0.25

# -----------------------------------------------------------------------------
# Self-learning auditor
# -----------------------------------------------------------------------------
# The auditor runs once per interval (daily by default), inside the server or via `python auditor.py`.
AUDIT_INTERVAL_HOURS: float = _env_float("AUDIT_INTERVAL_HOURS", 24.0, 1.0, 24 * 30)
for _retired in ("AUDIT_COOLDOWN_HOURS", "AUDIT_TRADE_THRESHOLD"):
    if _env_str(_retired):
        CONFIG_WARNINGS.append(f"{_retired} is no longer used; the auditor runs every AUDIT_INTERVAL_HOURS "
                               f"({AUDIT_INTERVAL_HOURS:g}h)")
AUDIT_LOOKBACK_TRADES: int = 50
AUDIT_LOCK_FILE: Path = BASE_DIR / ".audit.lock"
AUDIT_MIN_TRADES: int = 3
RULE_MIN_PENALTY: int = 10
RULE_MAX_PENALTY: int = 25
# A rule needs this many losing trades that match all of its conditions (checked by code, not the AI) ...
RULE_MIN_SAMPLE_SIZE: int = 3
# ... and at least this share of the matching trades must be losses.
RULE_MIN_LOSS_RATE: float = 0.6
MAX_ACTIVE_RULES: int = 20
# Combined penalty of all learned rules on one decision (guards have their own GUARD_MAX_PENALTY).
RULE_TOTAL_PENALTY_CAP: int = 30
# A rule expires unless an audit reconfirms it within this many days.
RULE_TTL_DAYS: int = _env_int("RULE_TTL_DAYS", 30, 1, 365)
# Trades blocked by a rule are followed on price data ("shadow trades"); once this many are resolved
# and they would have made money overall, the rule is retired.
RULE_SHADOW_MIN_RESOLVED: int = 5
SHADOW_FILE: Path = BASE_DIR / "shadow_trades.json"
SHADOW_MAX_DAYS: int = 7

# -----------------------------------------------------------------------------
# Dashboard server
# -----------------------------------------------------------------------------
HOST: str = _env_str("HOST", "127.0.0.1") or "127.0.0.1"
PORT: int = _env_int("PORT", 8000, 1, 65_535)
