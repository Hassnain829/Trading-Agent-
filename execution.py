"""
Order execution, position sizing and position management for MetaTrader 5.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Union

import MetaTrader5 as mt5

import config
from data_engine import MT5_LOCK, mt5_last_error, server_time_iso

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


def _send_with_fallbacks(request: Dict[str, Any], info: Any, refresh_price: Optional[Callable[[], float]] = None) -> Any:
    """order_send with filling-mode fallback and requote retries. Caller holds MT5_LOCK."""
    last_result = None
    for filling in _filling_candidates(info):
        request["type_filling"] = filling
        for attempt in range(3):
            if refresh_price is not None and attempt > 0:
                request["price"] = refresh_price()
            result = mt5.order_send(request)
            last_result = result
            if result is None:
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
        result = _send_with_fallbacks(request, info, current_price)

    side = "BUY" if closing_buy else "SELL"
    if result is None or result.retcode not in _SUCCESS_RETCODES:
        error = _describe(result)
        logger.error("[TRADE] Close of %s %s ticket %s failed: %s", position.symbol, side, ticket, error)
        return {"success": False, "ticket": ticket, "symbol": position.symbol, "error": error,
                "retcode": getattr(result, "retcode", None)}

    logger.info("[TRADE] Closed %s %s ticket %s | %.2f lots @ %s | floating P/L at close %.2f",
                position.symbol, side, ticket, position.volume, result.price or request["price"], position.profit)
    return {
        "success": True,
        "ticket": ticket,
        "symbol": position.symbol,
        "side": side,
        "volume": float(position.volume),
        "price": float(result.price or request["price"]),
        "deal": int(result.deal),
        "order": int(result.order),
        "retcode": int(result.retcode),
        "profit_estimate": round(float(position.profit), 2),
    }


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


def execute_trade(symbol: str, signal: str, stop_loss: float, take_profit: float,
                  risk_percent: float) -> Dict[str, Any]:
    """Size and send a market order with SL/TP. Raises TradeExecutionError on any failure."""
    side = str(signal).upper()
    if side not in ("BUY", "SELL"):
        raise TradeExecutionError(f"cannot execute signal {signal!r}")
    if not stop_loss or not take_profit:
        raise TradeExecutionError("stop_loss and take_profit are required")

    with MT5_LOCK:
        account = mt5.account_info()
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if account is None or info is None or tick is None:
            raise TradeExecutionError(f"MT5 data unavailable for {symbol} ({mt5_last_error()})")
        equity = float(account.equity)
        price = float(tick.ask if side == "BUY" else tick.bid)

        stop_loss, take_profit = _enforce_stop_distance(side, float(tick.bid), float(tick.ask),
                                                        float(stop_loss), float(take_profit), info)
        if side == "BUY" and not (stop_loss < price < take_profit):
            raise TradeExecutionError(f"BUY stops invalid at fill price {price}: SL {stop_loss} / TP {take_profit}")
        if side == "SELL" and not (take_profit < price < stop_loss):
            raise TradeExecutionError(f"SELL stops invalid at fill price {price}: SL {stop_loss} / TP {take_profit}")

        volume = calculate_position_size(symbol, stop_loss, risk_percent, equity, entry_price=price)

        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
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

        def current_price() -> float:
            latest = mt5.symbol_info_tick(symbol)
            if latest is None:
                return request["price"]
            return float(latest.ask if side == "BUY" else latest.bid)

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
        result = _send_with_fallbacks(request, info, current_price)
        if result is None or result.retcode not in _SUCCESS_RETCODES:
            raise TradeExecutionError(f"{side} {volume} {symbol} rejected: {_describe(result)}")

        position_ticket = int(result.order)
        if result.deal:
            deals = mt5.history_deals_get(ticket=int(result.deal))
            if deals:
                position_ticket = int(deals[0].position_id) or position_ticket

    if result.retcode == mt5.TRADE_RETCODE_DONE_PARTIAL:
        logger.warning("[TRADE] %s %s partially filled: %.2f of %.2f lots", side, symbol, result.volume, volume)

    digits = int(info.digits)
    fill_price = float(result.price or request["price"])
    return {
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
    }
