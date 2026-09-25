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
# DeepSeek LLM
# -----------------------------------------------------------------------------
DEEPSEEK_API_KEY: str = _env_str("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE: str = (_env_str("DEEPSEEK_API_BASE", "https://api.deepseek.com")
                          or "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL: str = _env_str("DEEPSEEK_MODEL", "deepseek-chat") or "deepseek-chat"
DEEPSEEK_TIMEOUT_SECONDS: int = 60
DEEPSEEK_MAX_RETRIES: int = 3
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
SYMBOLS: List[str] = _env_list(
    "SYMBOLS",
    "EURUSDm,GBPUSDm,USDJPYm,USDCHFm,USDCADm,AUDUSDm,NZDUSDm,BTCUSDm,XAUUSDm",
)

MAX_OPEN_POSITIONS: int = _env_int("MAX_OPEN_POSITIONS", 5, 0, 500)
MAX_DAILY_LOSS_PERCENT: float = _env_float("MAX_DAILY_LOSS_PERCENT", 5.0, 0.0, 100.0)
AUTO_START_ENGINE: bool = _env_bool("AUTO_START_ENGINE", False)

# Orders placed by this engine carry this magic number; the reversal guard only
# ever closes positions that carry it, so manual trades are never touched.
MAGIC_NUMBER: int = 20_260_925
ORDER_DEVIATION_POINTS: int = 20
ORDER_COMMENT: str = "AI-HF"
# Refuse a trade when the broker's minimum lot would risk more than this
# multiple of the intended risk amount.
MIN_LOT_RISK_TOLERANCE: float = 2.0
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
AUDIT_TRADE_THRESHOLD: int = _env_int("AUDIT_TRADE_THRESHOLD", 10, 1, 10_000)
AUDIT_COOLDOWN_HOURS: float = _env_float("AUDIT_COOLDOWN_HOURS", 12.0, 0.0, 24 * 30)
AUDIT_LOOKBACK_TRADES: int = 50
AUDIT_MIN_TRADES: int = 3
RULE_MIN_PENALTY: int = 15
RULE_MAX_PENALTY: int = 30
RULE_MIN_SAMPLE_SIZE: int = 2
MAX_ACTIVE_RULES: int = 20

# -----------------------------------------------------------------------------
# Dashboard server
# -----------------------------------------------------------------------------
HOST: str = _env_str("HOST", "127.0.0.1") or "127.0.0.1"
PORT: int = _env_int("PORT", 8000, 1, 65_535)
