"""
Trading sessions: which of the four forex sessions are open now, and which pairs are "active" in them.

    Sydney     07:00-16:00 Sydney time       Tokyo      09:00-18:00 Tokyo time
    London     08:00-17:00 London time       New York   08:00-17:00 New York time

Local times make the hours follow daylight saving automatically. Every currency has a home session (JPY in
Tokyo, AUD in Sydney, EUR in London, USD in New York...); a pair is active while the home session of either
of its currencies is open, the forex week is open (Sunday 17:00 to Friday 17:00 New York) and it is quoting.
This is for the dashboard only: the engine itself trades 24/7 (config.TRADE_ALL_HOURS).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import data_engine

SESSIONS = {  # name -> (time zone, opens at local hour, closes at local hour)
    "Sydney": (ZoneInfo("Australia/Sydney"), 7, 16),
    "Tokyo": (ZoneInfo("Asia/Tokyo"), 9, 18),
    "London": (ZoneInfo("Europe/London"), 8, 17),
    "New York": (ZoneInfo("America/New_York"), 8, 17),
}
HOME_SESSIONS = {
    "AUD": ("Sydney",), "NZD": ("Sydney",),
    "JPY": ("Tokyo",), "SGD": ("Tokyo",), "HKD": ("Tokyo",), "CNH": ("Tokyo",),
    "EUR": ("London",), "GBP": ("London",), "CHF": ("London",), "NOK": ("London",), "SEK": ("London",),
    "DKK": ("London",), "PLN": ("London",), "HUF": ("London",), "CZK": ("London",), "TRY": ("London",),
    "ZAR": ("London",),
    "USD": ("New York",), "CAD": ("New York",), "MXN": ("New York",),
    # metals are traded mostly in London and New York
    "XAU": ("London", "New York"), "XAG": ("London", "New York"), "XPT": ("London", "New York"),
    "XPD": ("London", "New York"),
}
QUOTE_STALE_SECONDS = 900  # no tick for 15 minutes while others quote = not trading now


def open_sessions(now: Optional[datetime] = None) -> List[str]:
    """The sessions open at ``now`` (UTC), in the day's order; none while the forex week is closed."""
    now = now or datetime.now(timezone.utc)
    if data_engine.fx_week_clock(now).get("closed"):
        return []
    out = []
    for name, (zone, start, end) in SESSIONS.items():
        local = now.astimezone(zone)
        if local.weekday() < 5 and start <= local.hour < end:
            out.append(name)
    return out


def home_sessions(symbol: str) -> List[str]:
    """The sessions where a pair's currencies are at home, e.g. EURJPY -> London, Tokyo."""
    legs = data_engine.currency_legs(symbol) or ()
    out: List[str] = []
    for currency in legs:
        for name in HOME_SESSIONS.get(currency, ()):
            if name not in out:
                out.append(name)
    return out or list(SESSIONS)  # unknown currencies: treat as active whenever any session is open


def pair_status(symbols: List[str], tick_times: Dict[str, Optional[float]],
                now: Optional[datetime] = None) -> Dict[str, object]:
    """
    {"open": [...sessions open now], "pairs": {symbol: {"home": [...], "active": bool, "quoting": bool}}}.
    ``tick_times`` are each symbol's last tick (broker epoch); a symbol is quoting when its tick is at most
    QUOTE_STALE_SECONDS older than the freshest tick of all of them.
    """
    now_open = open_sessions(now)
    freshest = max((t for t in tick_times.values() if t), default=None)
    pairs = {}
    for symbol in symbols:
        tick = tick_times.get(symbol)
        quoting = bool(freshest and tick and freshest - tick <= QUOTE_STALE_SECONDS)
        home = home_sessions(symbol)
        pairs[symbol] = {"home": home, "quoting": quoting, "active": quoting and any(s in now_open for s in home)}
    return {"open": now_open, "all": list(SESSIONS), "pairs": pairs}
