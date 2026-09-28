"""
Order execution, position sizing and position management for MetaTrader 5.
"""
from __future__ import annotations

import logging
import math
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import MetaTrader5 as mt5

import config
from data_engine import MT5_LOCK, currency_legs, mt5_last_error, round_to_tick, server_time_iso, utc_now_iso

logger = logging.getLogger("hedgefund.execution")

# symbol_info().filling_mode bit flags (not exported by the MetaTrader5 package).
_SYMBOL_FILLING_FOK = 1
_SYMBOL_FILLING_IOC = 2

_RETRYABLE_RETCODES = {
    mt5.TRADE_RETCODE_REQUOTE,
    mt5.TRADE_RETCODE_PRICE_CHANGED,
    mt5.TRADE_RETCODE_PRICE_OFF,
    getattr(mt5, "TRADE_RETCODE_CONNECTION", 10031),
    getattr(mt5, "TRADE_RETCODE_TIMEOUT", 10012),
}
_SUCCESS_RETCODES = {mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL}
_RETCODE_HINTS = {
    getattr(mt5, "TRADE_RETCODE_CLIENT_DISABLES_AT", 10027): "Algo Trading is disabled in the MT5 terminal",
    getattr(mt5, "TRADE_RETCODE_SERVER_DISABLES_AT", 10026): "the broker disabled algo trading on this account",
    mt5.TRADE_RETCODE_MARKET_CLOSED: "market is closed",
    mt5.TRADE_RETCODE_NO_MONEY: "insufficient margin",
    mt5.TRADE_RETCODE_INVALID_STOPS: "stops rejected by the broker (too close or wrong side)",
}


class TradeExecutionError(RuntimeError):
    """Raised when an order cannot be sized, validated or filled."""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _volume_decimals(step: float) -> int:
    text = f"{step:.10f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0


def _normalize_volume(volume: float, info: Any) -> float:
    step = float(info.volume_step) or 0.01
    steps = math.floor(volume / step + 1e-9)
    normalized = steps * step
    normalized = max(float(info.volume_min), min(normalized, float(info.volume_max)))
    return round(normalized, _volume_decimals(step))


def _filling_candidates(info: Any) -> List[int]:
    """Filling types to try, most-preferred first, based on what the symbol allows."""
    flags = int(getattr(info, "filling_mode", 0) or 0)
    candidates: List[int] = []
    if flags & _SYMBOL_FILLING_IOC:
        candidates.append(mt5.ORDER_FILLING_IOC)
    if flags & _SYMBOL_FILLING_FOK:
        candidates.append(mt5.ORDER_FILLING_FOK)
    candidates.append(mt5.ORDER_FILLING_RETURN)
    for fallback in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _describe(result: Any) -> str:
    if result is None:
        return f"order_send returned None ({mt5_last_error()})"
    hint = _RETCODE_HINTS.get(result.retcode)
    return f"retcode {result.retcode} ({result.comment}){f': {hint}' if hint else ''}"


def _send_with_fallbacks(request: Dict[str, Any], info: Any, refresh_price: Optional[Callable[[], float]] = None,
                        detect_fill: Optional[Callable[[], Any]] = None) -> Any:
    """
    order_send with filling-mode fallback and requote retries. Caller holds MT5_LOCK.

    order_send returning None means the reply was lost, not necessarily that the order
    failed; ``detect_fill`` checks the account for the fill before anything is resent,
    so a lost reply can never turn into a duplicate order.
    """
    last_result = None
    for filling in _filling_candidates(info):
        request["type_filling"] = filling
        for attempt in range(3):
            if refresh_price is not None and attempt > 0:
                request["price"] = refresh_price()
            result = mt5.order_send(request)
            last_result = result
            if result is None:
                recovered = detect_fill() if detect_fill is not None else None
                if recovered is not None:
                    logger.warning("[TRADE] order_send returned no reply (%s) but the order was executed; "
                                   "not resending", mt5_last_error())
                    return recovered
                break
            if result.retcode in _SUCCESS_RETCODES:
                return result
            if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                break  # try the next filling policy
            if result.retcode in _RETRYABLE_RETCODES:
                time.sleep(0.3 * (attempt + 1))
                continue
            return result  # non-retryable rejection
        if last_result is not None and last_result.retcode != mt5.TRADE_RETCODE_INVALID_FILL:
            return last_result
    return last_result


_FILLING_NAMES = {mt5.ORDER_FILLING_FOK: "FOK", mt5.ORDER_FILLING_IOC: "IOC", mt5.ORDER_FILLING_RETURN: "RETURN"}


def _execution_quality(deal_side: str, requested: float, fill: float, info: Any, started: float,
                       request: Dict[str, Any]) -> Dict[str, Any]:
    """Requested vs filled price (slippage in points, positive = worse than requested), latency, fill policy."""
    digits = int(info.digits)
    point = float(info.point) or 10 ** -digits
    direction = 1.0 if deal_side == "BUY" else -1.0
    return {
        "executed_at": utc_now_iso(),
        "requested_price": round(requested, digits),
        "fill_price": round(fill, digits),
        "slippage_points": round((fill - requested) * direction / point, 1),
        "execution_ms": int((time.perf_counter() - started) * 1000),
        "type_filling": _FILLING_NAMES.get(request.get("type_filling"), str(request.get("type_filling"))),
    }


def _position_to_dict(position: Any, digits: int) -> Dict[str, Any]:
    side = "BUY" if position.type == mt5.POSITION_TYPE_BUY else "SELL"
    return {
        "ticket": int(position.ticket),
        "symbol": position.symbol,
        "side": side,
        "volume": float(position.volume),
        "price_open": round(float(position.price_open), digits),
        "price_current": round(float(position.price_current), digits),
        "sl": round(float(position.sl), digits) if position.sl else None,
        "tp": round(float(position.tp), digits) if position.tp else None,
        "profit": round(float(position.profit), 2),
        "swap": round(float(position.swap), 2),
        "time": server_time_iso(position.time),
        "magic": int(position.magic),
        "comment": position.comment,
        "digits": digits,
        "managed": int(position.magic) == config.MAGIC_NUMBER,
    }


# -----------------------------------------------------------------------------
# Position sizing
# -----------------------------------------------------------------------------
def _loss_per_lot(symbol: str, stop_loss_price: float, entry_price: Optional[float]) -> tuple:
    """(symbol info, entry price, stop distance, account-currency loss of 1 lot at the stop)."""
    with MT5_LOCK:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise TradeExecutionError(f"symbol_info({symbol}) unavailable ({mt5_last_error()})")
        if entry_price is None:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise TradeExecutionError(f"no tick for {symbol} ({mt5_last_error()})")
            entry_price = float(tick.ask) if stop_loss_price < tick.bid else float(tick.bid)
        is_buy = stop_loss_price < entry_price
        distance = abs(entry_price - stop_loss_price)
        if distance <= 0:
            raise TradeExecutionError("stop loss equals entry price")

        loss_per_lot = 0.0
        tick_size = float(info.trade_tick_size) or float(info.point)
        tick_value = float(getattr(info, "trade_tick_value_loss", 0.0) or 0.0) or float(info.trade_tick_value)
        if tick_size > 0 and tick_value > 0:
            loss_per_lot = distance / tick_size * tick_value
        if loss_per_lot <= 0:
            order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
            calc = mt5.order_calc_profit(order_type, symbol, 1.0, entry_price, stop_loss_price)
            loss_per_lot = abs(float(calc)) if calc else 0.0
        if loss_per_lot <= 0:
            loss_per_lot = distance * float(info.trade_contract_size)
    if loss_per_lot <= 0:
        raise TradeExecutionError(f"could not determine loss per lot for {symbol}")
    return info, entry_price, distance, loss_per_lot


def fixed_position_size(symbol: str, stop_loss_price: float, lots: float, equity: float,
                        entry_price: Optional[float] = None) -> float:
    """A fixed lot size (snapped to the broker's volume step), refused if its stop would risk > MAX_RISK_PERCENT."""
    if equity <= 0:
        raise TradeExecutionError("equity must be positive to size a position")
    info, _, distance, loss_per_lot = _loss_per_lot(symbol, stop_loss_price, entry_price)
    volume = _normalize_volume(lots, info)
    risk = volume * loss_per_lot
    risk_pct = risk / equity * 100.0
    if risk_pct > config.MAX_RISK_PERCENT:
        raise TradeExecutionError(f"fixed lot {volume} would risk {risk:.2f} ({risk_pct:.2f}% of equity) at the stop, "
                                  f"above the {config.MAX_RISK_PERCENT}% hard cap; trade refused")
    logger.info("[TRADE] Sizing %s: fixed %.2f lots | SL distance %.6g | loss/lot %.2f | risk %.2f (%.2f%% of equity)",
                symbol, volume, distance, loss_per_lot, risk, risk_pct)
    return volume


def calculate_position_size(symbol: str, stop_loss_price: float, risk_percent: float, equity: float,
                            entry_price: Optional[float] = None) -> float:
    """
    Lots such that hitting ``stop_loss_price`` loses ``risk_percent`` of ``equity``.

    Loss per lot = (price distance / tick size) * tick value, falling back to
    the broker's order_calc_profit and finally to distance * contract size.
    """
    if equity <= 0:
        raise TradeExecutionError("equity must be positive to size a position")
    if not 0 < risk_percent <= config.MAX_RISK_PERCENT:
        raise TradeExecutionError(f"risk_percent {risk_percent} outside (0, {config.MAX_RISK_PERCENT}]")

    info, entry_price, distance, loss_per_lot = _loss_per_lot(symbol, stop_loss_price, entry_price)
    risk_amount = equity * (risk_percent / 100.0)
    raw_volume = risk_amount / loss_per_lot
    volume = _normalize_volume(raw_volume, info)
    actual_risk = volume * loss_per_lot

    if actual_risk > risk_amount * config.MIN_LOT_RISK_TOLERANCE:
        raise TradeExecutionError(
            f"minimum lot {info.volume_min} would risk {actual_risk:.2f} "
            f"({actual_risk / equity * 100:.2f}% of equity) vs budget {risk_amount:.2f}; trade refused"
        )
    if actual_risk > risk_amount * 1.05:
        logger.warning("[TRADE] %s volume clamped to broker minimum %.2f: risk %.2f exceeds budget %.2f",
                       symbol, volume, actual_risk, risk_amount)

    logger.info("[TRADE] Sizing %s: equity %.2f x %.2f%% = %.2f risk | SL distance %.6g | "
                "loss/lot %.2f | raw %.4f -> %.2f lots (risk %.2f)",
                symbol, equity, risk_percent, risk_amount, distance, loss_per_lot, raw_volume, volume, actual_risk)
    return volume


# -----------------------------------------------------------------------------
# Positions
# -----------------------------------------------------------------------------
def get_open_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    with MT5_LOCK:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            code, _ = mt5.last_error()
            if code not in (0, 1):  # 1 = RES_S_OK / no positions
                logger.debug("[SYSTEM] positions_get failed (%s)", mt5_last_error())
            return []
        digits_cache: Dict[str, int] = {}
        result = []
        for position in positions:
            if position.symbol not in digits_cache:
                info = mt5.symbol_info(position.symbol)
                digits_cache[position.symbol] = int(info.digits) if info else 5
            result.append(_position_to_dict(position, digits_cache[position.symbol]))
    return result


def close_position(ticket_or_pos: Union[int, Dict[str, Any], Any]) -> Dict[str, Any]:
    """Close a whole position with an opposite market deal."""
    if isinstance(ticket_or_pos, dict):
        ticket = int(ticket_or_pos["ticket"])
    elif hasattr(ticket_or_pos, "ticket"):
        ticket = int(ticket_or_pos.ticket)
    else:
        ticket = int(ticket_or_pos)

    with MT5_LOCK:
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "ticket": ticket, "error": f"position {ticket} not found (already closed?)"}
        position = positions[0]
        info = mt5.symbol_info(position.symbol)
        tick = mt5.symbol_info_tick(position.symbol)
        if info is None or tick is None:
            return {"success": False, "ticket": ticket, "error": f"no market data for {position.symbol}"}

        closing_buy = position.type == mt5.POSITION_TYPE_BUY
        order_type = mt5.ORDER_TYPE_SELL if closing_buy else mt5.ORDER_TYPE_BUY

        def current_price() -> float:
            latest = mt5.symbol_info_tick(position.symbol)
            if latest is None:
                return request["price"]
            return float(latest.bid if closing_buy else latest.ask)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": float(position.volume),
            "type": order_type,
            "position": ticket,
            "price": float(tick.bid if closing_buy else tick.ask),
            "deviation": config.ORDER_DEVIATION_POINTS,
            "magic": config.MAGIC_NUMBER,
            "comment": f"{config.ORDER_COMMENT} close",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        def detect_close() -> Any:
            if mt5.positions_get(ticket=ticket):
                return None
            return SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, price=request["price"], volume=position.volume,
                                   deal=0, order=0, comment="close confirmed after a lost reply")

        started = time.perf_counter()
        result = _send_with_fallbacks(request, info, current_price, detect_close)

    side = "BUY" if closing_buy else "SELL"
    if result is None or result.retcode not in _SUCCESS_RETCODES:
        error = _describe(result)
        logger.error("[TRADE] Close of %s %s ticket %s failed: %s", position.symbol, side, ticket, error)
        return {"success": False, "ticket": ticket, "symbol": position.symbol, "error": error,
                "retcode": getattr(result, "retcode", None)}

    fill_price = float(result.price or request["price"])
    quality = _execution_quality("SELL" if closing_buy else "BUY", float(request["price"]), fill_price,
                                 info, started, request)
    logger.info("[TRADE] Closed %s %s ticket %s | %.2f lots @ %s | floating P/L at close %.2f | slippage %s pts",
                position.symbol, side, ticket, position.volume, fill_price, position.profit, quality["slippage_points"])
    return {
        "success": True,
        "ticket": ticket,
        "symbol": position.symbol,
        "side": side,
        "volume": float(position.volume),
        "price": fill_price,
        "deal": int(result.deal),
        "order": int(result.order),
        "retcode": int(result.retcode),
        "profit_estimate": round(float(position.profit), 2),
        "execution": quality,
    }


# -----------------------------------------------------------------------------
# Portfolio risk: what the open positions lose if every stop is hit from here
# -----------------------------------------------------------------------------
def _tick_value_loss(info: Any) -> float:
    return float(getattr(info, "trade_tick_value_loss", 0.0) or 0.0) or float(info.trade_tick_value or 0.0)


def _position_risk_percent(position: Any, info: Any, equity: float) -> float:
    """% of equity a position can still lose from the current price if its stop is hit (floating profit it would give back counts)."""
    if equity <= 0 or info is None:
        return 0.0
    stop = float(position.sl or 0.0)
    if stop <= 0:
        return config.UNPROTECTED_POSITION_RISK_PERCENT
    is_buy = position.type == mt5.POSITION_TYPE_BUY
    current = float(position.price_current or position.price_open)
    distance = (current - stop) if is_buy else (stop - current)
    if distance <= 0:
        return 0.0
    tick_size = float(info.trade_tick_size) or float(info.point)
    tick_value = _tick_value_loss(info)
    if tick_size > 0 and tick_value > 0:
        loss = distance / tick_size * tick_value * float(position.volume)
    else:
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        calc = mt5.order_calc_profit(order_type, position.symbol, float(position.volume), current, stop)
        loss = abs(float(calc)) if calc else 0.0
    return loss / equity * 100.0


def _add_exposure(by_currency: Dict[str, Dict[str, float]], symbol: str, side: str, risk_percent: float) -> None:
    """A BUY of EURUSD is long EUR and short USD; its full risk counts against both."""
    legs = currency_legs(symbol)
    if not legs:
        return
    base, quote = legs
    for currency, direction in ((base, "long" if side == "BUY" else "short"),
                                (quote, "short" if side == "BUY" else "long")):
        entry = by_currency.setdefault(currency, {"long": 0.0, "short": 0.0})
        entry[direction] += risk_percent


def _open_risk_locked(equity: float, exclude_tickets: Iterable[int] = ()) -> Dict[str, Any]:
    """Open risk of every position on the account (manual ones too: they are real exposure). Caller holds MT5_LOCK."""
    excluded = {int(ticket) for ticket in exclude_tickets}
    by_currency: Dict[str, Dict[str, float]] = {}
    rows: List[Dict[str, Any]] = []
    infos: Dict[str, Any] = {}
    for position in mt5.positions_get() or []:
        if int(position.ticket) in excluded:
            continue
        if position.symbol not in infos:
            infos[position.symbol] = mt5.symbol_info(position.symbol)
        risk = _position_risk_percent(position, infos[position.symbol], equity)
        side = "BUY" if position.type == mt5.POSITION_TYPE_BUY else "SELL"
        _add_exposure(by_currency, position.symbol, side, risk)
        rows.append({"ticket": int(position.ticket), "symbol": position.symbol, "side": side,
                     "risk_percent": round(risk, 3), "protected": bool(position.sl)})
    total = sum(row["risk_percent"] for row in rows)
    return {
        "equity": equity,
        "total_percent": round(total, 3),
        "by_currency": {cur: {k: round(v, 3) for k, v in d.items()} for cur, d in sorted(by_currency.items())},
        "positions": rows,
    }


def open_risk(exclude_tickets: Iterable[int] = ()) -> Dict[str, Any]:
    """Open risk for the dashboard and the risk checks."""
    with MT5_LOCK:
        account = mt5.account_info()
        if account is None:
            return {"equity": 0.0, "total_percent": 0.0, "by_currency": {}, "positions": []}
        return _open_risk_locked(float(account.equity), exclude_tickets)


def _check_portfolio(symbol: str, side: str, new_risk: float, equity: float, exclude_tickets: Iterable[int],
                     max_total_open_risk: Optional[float]) -> None:
    """Refuse a trade that breaks the daily loss budget or piles too much risk onto one currency."""
    risk = _open_risk_locked(equity, exclude_tickets)
    if max_total_open_risk is not None and risk["total_percent"] + new_risk > max_total_open_risk + 1e-9:
        raise TradeExecutionError(
            f"daily loss budget: open risk {risk['total_percent']:.2f}% + this trade {new_risk:.2f}% would exceed "
            f"the {max_total_open_risk:.2f}% still available before the daily loss limit")
    cap = config.MAX_CURRENCY_RISK_PERCENT
    legs = currency_legs(symbol)
    if cap > 0 and legs:
        base, quote = legs
        for currency, direction in ((base, "long" if side == "BUY" else "short"),
                                    (quote, "short" if side == "BUY" else "long")):
            existing = risk["by_currency"].get(currency, {}).get(direction, 0.0)
            if existing + new_risk > cap + 1e-9:
                raise TradeExecutionError(
                    f"currency exposure: {direction} {currency} already carries {existing:.2f}% open risk; "
                    f"+{new_risk:.2f}% would exceed the {cap:.2f}% per-currency cap")


# -----------------------------------------------------------------------------
# Order entry
# -----------------------------------------------------------------------------
def _enforce_stop_distance(side: str, bid: float, ask: float, stop_loss: float, take_profit: float,
                           info: Any) -> tuple:
    """Push SL/TP outside the broker's minimum stop distance (trade_stops_level)."""
    point = float(info.point)
    min_distance = (int(info.trade_stops_level) + 1) * point
    digits = int(info.digits)
    if min_distance <= point:
        return round(stop_loss, digits), round(take_profit, digits)
    if side == "BUY":  # a long closes at bid
        stop_loss = min(stop_loss, bid - min_distance)
        take_profit = max(take_profit, bid + min_distance)
    else:  # a short closes at ask
        stop_loss = max(stop_loss, ask + min_distance)
        take_profit = min(take_profit, ask - min_distance)
    return round(stop_loss, digits), round(take_profit, digits)


def _prepare_order(symbol: str, side: str, stop_loss: float, take_profit: float, risk_percent: float,
                   sl_distance: Optional[float], tp_distance: Optional[float], entry_reference: Optional[float],
                   atr_reference: Optional[float], exclude_tickets: Iterable[int],
                   max_total_open_risk: Optional[float], check_margin: bool,
                   max_drift_atr: Optional[float] = None) -> Dict[str, Any]:
    """
    Every pre-trade check and the sized order request, without sending anything. Caller holds MT5_LOCK.
    Raises TradeExecutionError when the trade must not be placed.
    """
    if side not in ("BUY", "SELL"):
        raise TradeExecutionError(f"cannot execute signal {side!r}")
    if not stop_loss or not take_profit:
        raise TradeExecutionError("stop_loss and take_profit are required")

    account = mt5.account_info()
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if account is None or info is None or tick is None:
        raise TradeExecutionError(f"MT5 data unavailable for {symbol} ({mt5_last_error()})")
    equity = float(account.equity)
    price = float(tick.ask if side == "BUY" else tick.bid)
    digits = int(info.digits)

    # The AI may have taken minutes: refuse if the market has already moved away from its premise.
    drift_limit = max_drift_atr if max_drift_atr is not None else config.MAX_ENTRY_DRIFT_ATR
    if entry_reference and atr_reference and atr_reference > 0:
        drift = abs(price - float(entry_reference))
        if drift > drift_limit * float(atr_reference):
            raise TradeExecutionError(
                f"price moved {drift / float(atr_reference):.2f} ATR ({entry_reference} -> {price}) while the AI "
                f"was deciding (limit {drift_limit} ATR); decision is stale")

    if sl_distance and tp_distance and sl_distance > 0 and tp_distance > 0:
        direction = 1.0 if side == "BUY" else -1.0
        tick_size = float(info.trade_tick_size) or float(info.point)
        anchored_sl = round_to_tick(price - direction * sl_distance, tick_size, digits)
        anchored_tp = round_to_tick(price + direction * tp_distance, tick_size, digits)
        if (anchored_sl, anchored_tp) != (round(float(stop_loss), digits), round(float(take_profit), digits)):
            logger.info("[TRADE] %s ATR stops re-anchored to live price %s: SL %s -> %s | TP %s -> %s",
                        symbol, price, stop_loss, anchored_sl, take_profit, anchored_tp)
        stop_loss, take_profit = anchored_sl, anchored_tp

    stop_loss, take_profit = _enforce_stop_distance(side, float(tick.bid), float(tick.ask),
                                                    float(stop_loss), float(take_profit), info)
    if side == "BUY" and not (stop_loss < price < take_profit):
        raise TradeExecutionError(f"BUY stops invalid at fill price {price}: SL {stop_loss} / TP {take_profit}")
    if side == "SELL" and not (take_profit < price < stop_loss):
        raise TradeExecutionError(f"SELL stops invalid at fill price {price}: SL {stop_loss} / TP {take_profit}")

    # Spreads blow out around the daily rollover and news: check the cost at send time, not decision time.
    spread = max(float(tick.ask) - float(tick.bid), 0.0)
    stop_distance = abs(price - stop_loss)
    if stop_distance > 0 and spread > config.MAX_SPREAD_TO_STOP * stop_distance:
        raise TradeExecutionError(f"spread widened to {spread / stop_distance:.0%} of the stop distance "
                                  f"(limit {config.MAX_SPREAD_TO_STOP:.0%})")

    sizing_mode = config.POSITION_SIZING_MODE
    if sizing_mode == "FIXED":
        volume = fixed_position_size(symbol, stop_loss, config.FIXED_LOT, equity, entry_price=price)
    else:
        volume = calculate_position_size(symbol, stop_loss, risk_percent, equity, entry_price=price)

    order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
    if check_margin:
        margin = mt5.order_calc_margin(order_type, symbol, volume, price)
        free_margin = float(account.margin_free)
        if margin is not None and margin > 0 and margin > free_margin * 0.9:
            scaled = _normalize_volume(volume * (free_margin * 0.9) / margin, info)
            scaled_margin = mt5.order_calc_margin(order_type, symbol, scaled, price) or 0.0
            if scaled_margin > free_margin * 0.9 or scaled <= 0:
                raise TradeExecutionError(f"insufficient free margin ({free_margin:.2f}) for {symbol}; "
                                          f"{volume} lots need {margin:.2f}")
            logger.warning("[TRADE] %s volume reduced %.2f -> %.2f to respect free margin %.2f",
                           symbol, volume, scaled, free_margin)
            volume = scaled

    loss_per_lot = _loss_per_lot(symbol, stop_loss, price)[3]
    new_risk = volume * loss_per_lot / equity * 100.0 if equity > 0 else 0.0
    _check_portfolio(symbol, side, new_risk, equity, exclude_tickets, max_total_open_risk)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": config.ORDER_DEVIATION_POINTS,
        "magic": config.MAGIC_NUMBER,
        "comment": config.ORDER_COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
    }
    return {"request": request, "info": info, "equity": equity, "price": price, "volume": volume,
            "stop_loss": stop_loss, "take_profit": take_profit, "sizing_mode": sizing_mode,
            "risk_percent": round(new_risk, 3), "digits": digits}


def preview_trade(symbol: str, signal: str, stop_loss: float, take_profit: float, risk_percent: float,
                  sl_distance: Optional[float] = None, tp_distance: Optional[float] = None,
                  entry_reference: Optional[float] = None, atr_reference: Optional[float] = None,
                  exclude_tickets: Iterable[int] = (), max_total_open_risk: Optional[float] = None,
                  max_drift_atr: Optional[float] = None) -> Dict[str, Any]:
    """
    Run every pre-trade check without sending an order (used before closing a position for a reversal,
    so a position is never closed for a replacement that would be refused). Raises TradeExecutionError.
    """
    excluded = list(exclude_tickets)
    with MT5_LOCK:
        order = _prepare_order(symbol, str(signal).upper(), stop_loss, take_profit, risk_percent, sl_distance,
                               tp_distance, entry_reference, atr_reference, excluded, max_total_open_risk,
                               check_margin=not excluded,  # margin frees up once the old position is closed
                               max_drift_atr=max_drift_atr)
    return {key: order[key] for key in ("price", "volume", "stop_loss", "take_profit", "risk_percent", "sizing_mode")}


def execute_trade(symbol: str, signal: str, stop_loss: float, take_profit: float,
                  risk_percent: float, sl_distance: Optional[float] = None,
                  tp_distance: Optional[float] = None, entry_reference: Optional[float] = None,
                  atr_reference: Optional[float] = None,
                  max_total_open_risk: Optional[float] = None,
                  max_drift_atr: Optional[float] = None, comment: Optional[str] = None) -> Dict[str, Any]:
    """
    Size and send a market order with SL/TP. Raises TradeExecutionError on any failure.

    When ``sl_distance``/``tp_distance`` (the ATR distances behind the stops) are
    given, SL/TP are re-anchored to the live price at send time so the ATR
    geometry stays exact even if the quote moved while the AI was deciding.
    ``entry_reference``/``atr_reference`` refuse the order if that move was too large;
    ``max_total_open_risk`` is the % of equity left before the daily loss limit.
    """
    side = str(signal).upper()
    with MT5_LOCK:
        order = _prepare_order(symbol, side, stop_loss, take_profit, risk_percent, sl_distance, tp_distance,
                               entry_reference, atr_reference, (), max_total_open_risk, check_margin=True,
                               max_drift_atr=max_drift_atr)
        request, info, equity = order["request"], order["info"], order["equity"]
        if comment:
            request["comment"] = comment[:31]
        volume, stop_loss, take_profit = order["volume"], order["stop_loss"], order["take_profit"]
        sizing_mode = order["sizing_mode"]
        known_tickets = {int(p.ticket) for p in mt5.positions_get(symbol=symbol) or []}

        def current_price() -> float:
            latest = mt5.symbol_info_tick(symbol)
            if latest is None:
                return request["price"]
            return float(latest.ask if side == "BUY" else latest.bid)

        def detect_fill() -> Any:
            for position in mt5.positions_get(symbol=symbol) or []:
                if int(position.ticket) not in known_tickets and int(position.magic) == config.MAGIC_NUMBER:
                    return SimpleNamespace(retcode=mt5.TRADE_RETCODE_DONE, price=float(position.price_open),
                                           volume=float(position.volume), deal=0, order=int(position.ticket),
                                           comment="fill found after a lost reply")
            return None

        started = time.perf_counter()
        result = _send_with_fallbacks(request, info, current_price, detect_fill)
        if result is None or result.retcode not in _SUCCESS_RETCODES:
            raise TradeExecutionError(f"{side} {volume} {symbol} rejected: {_describe(result)}")
        quality = _execution_quality(side, float(request["price"]), float(result.price or request["price"]),
                                     info, started, request)

        position_ticket = int(result.order)
        if result.deal:
            deals = mt5.history_deals_get(ticket=int(result.deal))
            if deals:
                position_ticket = int(deals[0].position_id) or position_ticket

    if result.retcode == mt5.TRADE_RETCODE_DONE_PARTIAL:
        logger.warning("[TRADE] %s %s partially filled: %.2f of %.2f lots", side, symbol, result.volume, volume)

    digits = int(info.digits)
    fill_price = float(result.price or request["price"])
    filled_volume = float(result.volume or volume)
    try:
        loss_per_lot = _loss_per_lot(symbol, stop_loss, fill_price)[3]
    except TradeExecutionError:
        loss_per_lot = 0.0
    return {
        "sizing_mode": sizing_mode,
        "risk_amount": round(filled_volume * loss_per_lot, 2),
        "risk_percent_actual": round(filled_volume * loss_per_lot / equity * 100.0, 3) if equity else None,
        "symbol": symbol,
        "side": side,
        "deal": int(result.deal),
        "order": int(result.order),
        "position_ticket": position_ticket,
        "price": round(fill_price, digits),
        "volume": float(result.volume or volume),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "retcode": int(result.retcode),
        "comment": result.comment,
        "digits": digits,
        "equity_at_entry": equity,
        "execution": quality,
    }
