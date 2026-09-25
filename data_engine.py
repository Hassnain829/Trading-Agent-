"""
Market data acquisition and indicator engine for MetaTrader 5.

All calls into the MetaTrader5 package go through ``MT5_LOCK``: the package
talks to the terminal over a single IPC channel and is not safe to use from
several threads at once (the dashboard and the trading loop both query it).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import config

logger = logging.getLogger("hedgefund.data")

MT5_LOCK = threading.RLock()

# symbol -> {"time_msc": last tick time seen, "since": monotonic time it was first seen}
_tick_watch: Dict[str, Dict[str, float]] = {}


class DataEngineError(RuntimeError):
    """Raised when market data cannot be retrieved from MetaTrader 5."""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def mt5_last_error() -> str:
    try:
        code, message = mt5.last_error()
        return f"{code}: {message}"
    except Exception:  # pragma: no cover - defensive
        return "unknown MT5 error"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def server_time_iso(epoch_seconds: float) -> str:
    """MT5 timestamps are broker server time encoded as epoch seconds."""
    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).isoformat(timespec="seconds")


def _clean(value: Any, ndigits: Optional[int] = None) -> Optional[float]:
    """Convert numpy/pandas scalars to JSON-safe floats (NaN/inf -> None)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, ndigits) if ndigits is not None else number


# -----------------------------------------------------------------------------
# Connection management
# -----------------------------------------------------------------------------
def initialize_mt5() -> bool:
    """Start the MT5 bridge, log in when credentials are configured, and log the account."""
    with MT5_LOCK:
        try:
            if config.MT5_PATH:
                ok = mt5.initialize(config.MT5_PATH, timeout=config.MT5_TIMEOUT_MS)
            else:
                ok = mt5.initialize(timeout=config.MT5_TIMEOUT_MS)
        except Exception as exc:
            logger.error("[SYSTEM] MT5 initialize raised: %s", exc)
            return False
        if not ok:
            logger.error("[SYSTEM] MT5 initialize failed (%s). Is the terminal installed%s?",
                         mt5_last_error(), f" at {config.MT5_PATH}" if config.MT5_PATH else "")
            return False

        if config.MT5_LOGIN:
            current = mt5.account_info()
            already_logged_in = (
                current is not None
                and current.login == config.MT5_LOGIN
                and (not config.MT5_SERVER or current.server == config.MT5_SERVER)
            )
            if not already_logged_in:
                try:
                    logged_in = mt5.login(
                        config.MT5_LOGIN,
                        password=config.MT5_PASSWORD,
                        server=config.MT5_SERVER,
                        timeout=config.MT5_TIMEOUT_MS,
                    )
                except Exception as exc:
                    logger.error("[SYSTEM] MT5 login raised: %s", exc)
                    logged_in = False
                if not logged_in:
                    logger.error("[SYSTEM] MT5 login failed for %s@%s (%s)",
                                 config.MT5_LOGIN, config.MT5_SERVER, mt5_last_error())
                    mt5.shutdown()
                    return False

        account = mt5.account_info()
        terminal = mt5.terminal_info()
        if account is None:
            logger.error("[SYSTEM] MT5 connected but no trading account is logged in (%s). "
                         "Set MT5_LOGIN/MT5_PASSWORD/MT5_SERVER in .env.", mt5_last_error())
            mt5.shutdown()
            return False

        logger.info(
            "[SYSTEM] MT5 connected | login %s | server %s | %s | balance %.2f %s | equity %.2f | leverage 1:%s",
            account.login, account.server, account.company, account.balance,
            account.currency, account.equity, account.leverage,
        )
        if terminal is not None:
            logger.info("[SYSTEM] Terminal %s build %s | connected=%s | algo trading %s",
                        terminal.name, terminal.build, terminal.connected,
                        "ENABLED" if terminal.trade_allowed else "DISABLED")
            if not terminal.trade_allowed:
                logger.warning("[SYSTEM] Algo Trading is disabled in the MT5 terminal: every order will be "
                               "rejected until you enable the 'Algo Trading' button.")
        if not account.trade_allowed:
            logger.warning("[SYSTEM] The broker reports trading is not allowed on this account.")
        return True


def ensure_connection() -> bool:
    """Verify the bridge is alive; reinitialize it if the terminal link was lost."""
    with MT5_LOCK:
        try:
            terminal = mt5.terminal_info()
            account = mt5.account_info()
        except Exception:
            terminal = account = None
        if terminal is not None and account is not None:
            if not terminal.connected:
                logger.warning("[SYSTEM] MT5 terminal is running but disconnected from the broker server")
                return False
            return True
        logger.warning("[SYSTEM] MT5 link lost (%s) - reinitializing", mt5_last_error())
        try:
            mt5.shutdown()
        except Exception:
            pass
    return initialize_mt5()


def shutdown_mt5() -> None:
    with MT5_LOCK:
        try:
            mt5.shutdown()
            logger.info("[SYSTEM] MT5 bridge shut down")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[SYSTEM] MT5 shutdown raised: %s", exc)


def get_account_snapshot() -> Optional[Dict[str, Any]]:
    """Live account metrics, or None when MT5 is unavailable."""
    with MT5_LOCK:
        try:
            account = mt5.account_info()
        except Exception:
            account = None
    if account is None:
        return None
    return {
        "login": int(account.login),
        "server": account.server,
        "company": account.company,
        "name": account.name,
        "currency": account.currency,
        "leverage": int(account.leverage),
        "balance": float(account.balance),
        "equity": float(account.equity),
        "profit": float(account.profit),
        "margin": float(account.margin),
        "margin_free": float(account.margin_free),
        "margin_level": float(account.margin_level),
        "trade_allowed": bool(account.trade_allowed),
    }


# -----------------------------------------------------------------------------
# Indicators
# -----------------------------------------------------------------------------
def _ema(series: pd.Series, period: int) -> pd.Series:
    """EMA seeded with the SMA of the first ``period`` values (TA-Lib convention)."""
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < period:
        return pd.Series(out, index=series.index)
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = values[:period].mean()
    for i in range(period, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI: exponential smoothing of gains and losses with alpha = 1/period."""
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # No losses in the window -> RSI 100; flat window -> neutral 50.
    rsi = rsi.where(avg_loss != 0.0, np.where(avg_gain > 0.0, 100.0, 50.0))
    return rsi.where(avg_gain.notna())


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder ATR over the true range max(H-L, |H-prevC|, |L-prevC|)."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _price_structure(bars: pd.DataFrame, digits: Optional[int]) -> Dict[str, Any]:
    """Describe the last N bars: direction, swing structure and range."""
    if bars.empty:
        return {}
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)

    higher_highs = int(np.sum(highs[1:] > highs[:-1]))
    lower_lows = int(np.sum(lows[1:] < lows[:-1]))
    half = max(len(bars) // 2, 1)
    first_high, first_low = highs[:half].max(), lows[:half].min()
    last_high, last_low = highs[half:].max(), lows[half:].min()
    if last_high > first_high and last_low > first_low:
        pattern = "BULLISH_HH_HL"
    elif last_high < first_high and last_low < first_low:
        pattern = "BEARISH_LH_LL"
    elif last_high <= first_high and last_low >= first_low:
        pattern = "CONTRACTING_RANGE"
    else:
        pattern = "EXPANDING_RANGE"

    change_pct = (closes[-1] - closes[0]) / closes[0] * 100.0 if closes[0] else 0.0
    if change_pct > 0.05 and pattern != "BEARISH_LH_LL":
        bias = "UP"
    elif change_pct < -0.05 and pattern != "BULLISH_HH_HL":
        bias = "DOWN"
    else:
        bias = "SIDEWAYS"

    return {
        "bars": int(len(bars)),
        "closes": [_clean(c, digits) for c in closes],
        "high": _clean(highs.max(), digits),
        "low": _clean(lows.min(), digits),
        "change_pct": _clean(change_pct, 3),
        "higher_highs": higher_highs,
        "lower_lows": lower_lows,
        "pattern": pattern,
        "bias": bias,
    }


def calculate_indicators(df: pd.DataFrame, digits: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute EMA(200), RSI(14), ATR(14), relative volume and recent structure.

    ``df`` must hold closed bars with columns time, open, high, low, close,
    tick_volume (oldest first). Returns the latest values as JSON-safe scalars.
    """
    required = {"time", "open", "high", "low", "close", "tick_volume"}
    missing = required - set(df.columns)
    if missing:
        raise DataEngineError(f"OHLCV frame is missing columns: {sorted(missing)}")
    if len(df) < max(config.RSI_PERIOD, config.ATR_PERIOD) + 1:
        raise DataEngineError(f"only {len(df)} bars available; indicators need more history")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    tick_volume = df["tick_volume"].astype(float)

    ema = _ema(close, config.EMA_PERIOD)
    rsi = _rsi(close, config.RSI_PERIOD)
    atr = _atr(high, low, close, config.ATR_PERIOD)
    volume_ma = tick_volume.rolling(config.VOL_MA_PERIOD, min_periods=config.VOL_MA_PERIOD).mean()
    rel_volume = tick_volume / volume_ma.replace(0.0, np.nan)
    atr_baseline = atr.rolling(config.ATR_BASELINE_PERIOD,
                               min_periods=max(config.ATR_BASELINE_PERIOD // 2, 2)).mean()

    last_close = float(close.iloc[-1])
    last_ema = _clean(ema.iloc[-1])
    last_atr = _clean(atr.iloc[-1])

    lookback = min(config.STRUCTURE_LOOKBACK, len(ema) - 1)
    ema_slope = None
    if last_ema is not None and lookback > 0 and not math.isnan(ema.iloc[-1 - lookback]):
        slope = last_ema - float(ema.iloc[-1 - lookback])
        tolerance = (last_atr or 0.0) * 0.05
        ema_slope = "RISING" if slope > tolerance else "FALLING" if slope < -tolerance else "FLAT"

    ema_distance_atr = None
    if last_ema is not None and last_atr:
        ema_distance_atr = (last_close - last_ema) / last_atr

    atr_ratio = None
    baseline = _clean(atr_baseline.iloc[-1])
    if last_atr and baseline:
        atr_ratio = last_atr / baseline

    last_time = pd.Timestamp(df["time"].iloc[-1])
    return {
        "bars": int(len(df)),
        "last_bar_time": last_time.isoformat(),
        "open": _clean(df["open"].iloc[-1], digits),
        "high": _clean(high.iloc[-1], digits),
        "low": _clean(low.iloc[-1], digits),
        "close": _clean(last_close, digits),
        "ema200": _clean(last_ema, digits),
        "price_vs_ema": None if last_ema is None else ("ABOVE" if last_close >= last_ema else "BELOW"),
        "ema_slope": ema_slope,
        "ema_distance_atr": _clean(ema_distance_atr, 2),
        "rsi14": _clean(rsi.iloc[-1], 2),
        "atr14": _clean(last_atr, (digits + 1) if digits is not None else None),
        "atr_pct": _clean(last_atr / last_close * 100.0 if last_atr and last_close else None, 4),
        "atr_ratio": _clean(atr_ratio, 3),
        "tick_volume": int(tick_volume.iloc[-1]),
        "volume_ma20": _clean(volume_ma.iloc[-1], 1),
        "rel_volume": _clean(rel_volume.iloc[-1], 3),
        "structure": _price_structure(df.tail(config.STRUCTURE_LOOKBACK), digits),
    }


# -----------------------------------------------------------------------------
# Market snapshots
# -----------------------------------------------------------------------------
def _rates_to_frame(rates: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(rates)
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    return frame[["time", "open", "high", "low", "close", "tick_volume"]]


def _select_symbol(symbol: str) -> None:
    if not mt5.symbol_select(symbol, True):
        raise DataEngineError(f"{symbol} could not be enabled in MarketWatch ({mt5_last_error()})")


def _live_tick(symbol: str):
    """Tick for a symbol; retries briefly because freshly selected symbols start empty."""
    for attempt in range(3):
        tick = mt5.symbol_info_tick(symbol)
        if tick is not None and (tick.bid > 0 or tick.ask > 0):
            return tick
        time.sleep(0.25 * (attempt + 1))
    raise DataEngineError(f"no live tick for {symbol} ({mt5_last_error()})")


def _closed_bars(symbol: str, timeframe: int, label: str) -> np.ndarray:
    """Most recent closed bars (start_pos=1 skips the bar still forming)."""
    rates = None
    for attempt in range(2):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, config.BARS_TO_FETCH)
        if rates is not None and len(rates) >= config.ATR_PERIOD + 1:
            return rates
        time.sleep(0.5)  # history may still be downloading
    count = 0 if rates is None else len(rates)
    raise DataEngineError(f"{symbol} {label}: only {count} bars returned ({mt5_last_error()})")


def _update_tick_watch(symbol: str, time_msc: int) -> bool:
    """True when the symbol's last tick has been frozen for MARKET_IDLE_SECONDS."""
    now = time.monotonic()
    entry = _tick_watch.get(symbol)
    if entry is None or entry["time_msc"] != time_msc:
        _tick_watch[symbol] = {"time_msc": time_msc, "since": now}
        return False
    return (now - entry["since"]) >= config.MARKET_IDLE_SECONDS


def fetch_multi_timeframe_data(symbol: str) -> Dict[str, Any]:
    """Live quote, account state and H1/D1 indicator snapshots for one symbol."""
    with MT5_LOCK:
        _select_symbol(symbol)
        info = mt5.symbol_info(symbol)
        if info is None:
            raise DataEngineError(f"symbol_info({symbol}) returned None ({mt5_last_error()})")
        tick = _live_tick(symbol)
        raw_frames = {label: _closed_bars(symbol, tf, label) for label, tf in config.TIMEFRAMES.items()}
        account = mt5.account_info()
    if account is None:
        raise DataEngineError(f"account_info unavailable ({mt5_last_error()})")

    digits = int(info.digits)
    point = float(info.point) or 10 ** -digits
    bid, ask = float(tick.bid), float(tick.ask)
    spread_price = max(ask - bid, 0.0)

    h1 = calculate_indicators(_rates_to_frame(raw_frames["H1"]), digits)
    d1 = calculate_indicators(_rates_to_frame(raw_frames["D1"]), digits)

    trade_mode = int(info.trade_mode)
    tradeable = trade_mode not in (mt5.SYMBOL_TRADE_MODE_DISABLED, mt5.SYMBOL_TRADE_MODE_CLOSEONLY)

    return {
        "symbol": symbol,
        "timestamp": utc_now_iso(),
        "tick_time": server_time_iso(tick.time),
        "bid": round(bid, digits),
        "ask": round(ask, digits),
        "mid": round((bid + ask) / 2.0, digits),
        "spread": int(round(spread_price / point)),
        "spread_price": round(spread_price, digits),
        "digits": digits,
        "point": point,
        "contract_size": float(info.trade_contract_size),
        "volume_min": float(info.volume_min),
        "volume_max": float(info.volume_max),
        "volume_step": float(info.volume_step),
        "trade_mode": trade_mode,
        "tradeable": tradeable,
        "market_idle": _update_tick_watch(symbol, int(tick.time_msc)),
        "equity": float(account.equity),
        "balance": float(account.balance),
        "currency": account.currency,
        "account_login": int(account.login),
        "h1_data": h1,
        "daily_data": d1,
    }


def fetch_correlated_asset_prices(current_symbol: Optional[str],
                                  all_symbols: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Bid/ask (and change since the D1 open) for every other configured symbol."""
    snapshot: Dict[str, Dict[str, Any]] = {}
    current = (current_symbol or "").upper()
    symbols: List[str] = list(all_symbols) if all_symbols is not None else list(config.SYMBOLS)
    for symbol in symbols:
        if symbol.upper() == current:
            continue
        try:
            with MT5_LOCK:
                mt5.symbol_select(symbol, True)
                tick = mt5.symbol_info_tick(symbol)
                info = mt5.symbol_info(symbol)
                today = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 1)
        except Exception as exc:
            logger.debug("[SYSTEM] correlated quote for %s failed: %s", symbol, exc)
            continue
        if tick is None or (tick.bid <= 0 and tick.ask <= 0):
            continue
        digits = int(info.digits) if info is not None else 5
        day_open = float(today[0]["open"]) if today is not None and len(today) else None
        change = (float(tick.bid) - day_open) / day_open * 100.0 if day_open else None
        snapshot[symbol] = {
            "bid": round(float(tick.bid), digits),
            "ask": round(float(tick.ask), digits),
            "day_change_pct": _clean(change, 3),
        }
    return snapshot
