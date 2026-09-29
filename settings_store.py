"""
Live trading settings edited from the dashboard.

.env provides the defaults; anything saved from the Settings drawer is written
to settings.json and overrides .env (also after restarts) until it is reset.
Settings are applied by updating the matching config attributes, which every
module reads at call time, so changes take effect on the next scan.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import config
from memory_store import read_json_file, write_json_atomic

logger = logging.getLogger("hedgefund.settings")

# setting name -> config attribute it controls
_CONFIG_ATTRIBUTES = {
    "risk_percent": "DEFAULT_RISK_PERCENT",
    "sizing_mode": "POSITION_SIZING_MODE",
    "fixed_lot": "FIXED_LOT",
    "max_open_positions": "MAX_OPEN_POSITIONS",
    "max_daily_loss_percent": "MAX_DAILY_LOSS_PERCENT",
    "max_daily_profit_percent": "MAX_DAILY_PROFIT_PERCENT",
    "confidence_threshold": "CONFIDENCE_THRESHOLD",
    "scan_interval_seconds": "SCAN_INTERVAL_SECONDS",
    "symbols_demo": "SYMBOLS_DEMO",
    "symbols_live": "SYMBOLS_LIVE",
    "rule_scope": "RULE_SCOPE",
    "overextension_guard": "OVEREXTENSION_GUARD",
    "weekend_entry_cutoff_hours": "WEEKEND_ENTRY_CUTOFF_HOURS",
    "weekend_close": "WEEKEND_CLOSE",
    "max_currency_risk_percent": "MAX_CURRENCY_RISK_PERCENT",
    "daily_loss_close_positions": "DAILY_LOSS_CLOSE_POSITIONS",
    "loss_cooldown_minutes": "LOSS_COOLDOWN_MINUTES",
    "ai_new_bar_only": "AI_NEW_BAR_ONLY",
    "news_guard": "NEWS_GUARD",
    "calibrate_threshold": "CALIBRATE_THRESHOLD",
    "strategy_mode": "STRATEGY_MODE",
    "scalp_time_stop_minutes": "SCALP_TIME_STOP_MINUTES",
    "scalp_max_trades_per_symbol": "SCALP_MAX_TRADES_PER_SYMBOL",
    "scalp_session_start_london": "SCALP_SESSION_START_LONDON",
    "scalp_session_end_new_york": "SCALP_SESSION_END_NEW_YORK",
    "scalp_strict_guard": "SCALP_STRICT_GUARD",
    "scalp_medium_trend": "SCALP_MEDIUM_TREND",
    "ml_filter": "ML_FILTER",
    "max_total_drawdown_percent": "MAX_TOTAL_DRAWDOWN_PERCENT",
    "drawdown_throttle_percent": "DRAWDOWN_THROTTLE_PERCENT",
    "kill_switch_close_positions": "KILL_SWITCH_CLOSE_POSITIONS",
}
LIST_SETTINGS = ("symbols_demo", "symbols_live")


def _read(name: str) -> Any:
    value = getattr(config, _CONFIG_ATTRIBUTES[name])
    return list(value) if name in LIST_SETTINGS else value


# Captured at import, before any override is applied.
ENV_DEFAULTS: Dict[str, Any] = {name: _read(name) for name in _CONFIG_ATTRIBUTES}


def _saved_overrides() -> Dict[str, Any]:
    data = read_json_file(config.SETTINGS_FILE, dict)
    if not isinstance(data, dict):
        return {}
    if "symbols" in data and "symbols_demo" not in data:
        data["symbols_demo"] = data["symbols"]  # single list saved by an earlier version
    return {key: value for key, value in data.items() if key in _CONFIG_ATTRIBUTES}


def current() -> Dict[str, Any]:
    """The settings in force right now."""
    return {name: _read(name) for name in _CONFIG_ATTRIBUTES}


def overridden() -> Dict[str, bool]:
    saved = _saved_overrides()
    return {name: name in saved for name in _CONFIG_ATTRIBUTES}


def apply(settings: Dict[str, Any]) -> None:
    for name, value in settings.items():
        attr = _CONFIG_ATTRIBUTES.get(name)
        if attr is not None and value is not None:
            setattr(config, attr, list(value) if name in LIST_SETTINGS else value)


def load_saved() -> Dict[str, Any]:
    """Apply settings.json at startup. Returns the overrides that were applied."""
    saved = _saved_overrides()
    apply(saved)
    if saved:
        logger.info("[SYSTEM] Dashboard settings loaded from %s: %s", config.SETTINGS_FILE.name,
                    ", ".join(f"{k}={v}" for k, v in saved.items()))
    return saved


def save(changes: Dict[str, Any]) -> Dict[str, Any]:
    """Apply and persist changed settings. Returns the full settings now in force."""
    saved = _saved_overrides()
    for name, value in changes.items():
        if name not in _CONFIG_ATTRIBUTES or value is None:
            continue
        if value == ENV_DEFAULTS[name]:
            saved.pop(name, None)  # back to the .env value: no override needed
        else:
            saved[name] = value
    apply(changes)
    write_json_atomic(config.SETTINGS_FILE, saved)
    return current()


def reset() -> Dict[str, Any]:
    """Drop every dashboard override and return to the .env values."""
    apply(ENV_DEFAULTS)
    write_json_atomic(config.SETTINGS_FILE, {})
    logger.info("[SYSTEM] Dashboard settings reset to .env values")
    return current()
