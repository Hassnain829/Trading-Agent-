"""
Persistent trade memory (memory.json) and closed-trade reconciliation.

Every filled order is stored with its full execution context. Once MT5 shows
the position as closed, ``reconcile_closed_trades`` pulls the exit deals from
the account history and stamps the record with exit price, realized P/L and
outcome, which is the raw material for the self-learning auditor.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import MetaTrader5 as mt5
import numpy as np

import config
from data_engine import (MT5_LOCK, mt5_last_error, server_epoch_to_utc_iso, server_time_iso,
                         server_utc_offset_seconds, utc_now_iso)

logger = logging.getLogger("hedgefund.memory")

_FILE_LOCKS: Dict[str, threading.RLock] = {}
_FILE_LOCKS_GUARD = threading.Lock()

_EXIT_ENTRIES = {mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY}
_DEAL_REASONS = {
    getattr(mt5, name): label
    for name, label in (
        ("DEAL_REASON_CLIENT", "MANUAL_DESKTOP"),
        ("DEAL_REASON_MOBILE", "MANUAL_MOBILE"),
        ("DEAL_REASON_WEB", "MANUAL_WEB"),
        ("DEAL_REASON_EXPERT", "ENGINE"),
        ("DEAL_REASON_SL", "STOP_LOSS"),
        ("DEAL_REASON_TP", "TAKE_PROFIT"),
        ("DEAL_REASON_SO", "STOP_OUT"),
        ("DEAL_REASON_ROLLOVER", "ROLLOVER"),
        ("DEAL_REASON_VMARGIN", "VARIATION_MARGIN"),
        ("DEAL_REASON_SPLIT", "SPLIT"),
    )
    if hasattr(mt5, name)
}
BREAKEVEN_TOLERANCE = 0.01


# -----------------------------------------------------------------------------
# Generic JSON persistence (shared with the auditor and AI brain)
# -----------------------------------------------------------------------------
def file_lock(path: Path) -> threading.RLock:
    key = str(Path(path).resolve())
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = _FILE_LOCKS[key] = threading.RLock()
        return lock


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def read_json_file(path: Path, default_factory: Callable[[], Any], quarantine_corrupt: bool = True) -> Any:
    """Load JSON; missing -> default. Corrupt files are moved aside instead of overwritten."""
    path = Path(path)
    with file_lock(path):
        if not path.exists():
            return default_factory()
        for attempt in range(5):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.error("[SYSTEM] %s is not valid JSON (%s)", path.name, exc)
                if quarantine_corrupt:
                    backup = path.with_name(f"{path.name}.corrupt-{datetime.now():%Y%m%d-%H%M%S}")
                    try:
                        os.replace(path, backup)
                        logger.error("[SYSTEM] Corrupt file preserved as %s", backup.name)
                    except OSError as move_exc:
                        logger.error("[SYSTEM] Could not quarantine %s: %s", path.name, move_exc)
                return default_factory()
            except OSError as exc:
                logger.error("[SYSTEM] Could not read %s: %s", path.name, exc)
                return default_factory()
        logger.error("[SYSTEM] %s stayed locked by another process; using defaults", path.name)
        return default_factory()


def write_json_atomic(path: Path, data: Any) -> None:
    """Write via a temp file + os.replace so readers never see a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with file_lock(path):
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                # Windows: another process (editor, antivirus) briefly holds the file.
                time.sleep(0.1 * (attempt + 1))
        raise OSError(f"could not replace {path.name}; it is locked by another process")


# -----------------------------------------------------------------------------
# Trade memory
# -----------------------------------------------------------------------------
def _record_key(record: Dict[str, Any]) -> str:
    return str(record.get("id") or f"ticket:{record.get('ticket')}")


def load_trade_memory() -> List[Dict[str, Any]]:
    data = read_json_file(config.MEMORY_FILE, list)
    if not isinstance(data, list):
        logger.error("[SYSTEM] memory.json does not contain a list; ignoring its contents")
        return []
    return [item for item in data if isinstance(item, dict)]


def _pending_path() -> Path:
    """Append-only journal that holds fills whose memory.json write failed."""
    return config.MEMORY_FILE.with_name(f"{config.MEMORY_FILE.stem}.pending.jsonl")


def _json_default_lenient(value: Any) -> Any:
    try:
        return _json_default(value)
    except TypeError:
        return str(value)


def _journal_pending(entry: Dict[str, Any]) -> None:
    path = _pending_path()
    line = json.dumps(entry, ensure_ascii=False, default=_json_default_lenient)
    with file_lock(path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def flush_pending_journal() -> int:
    """Merge journaled fills back into memory.json. Returns how many were recovered."""
    path = _pending_path()
    with file_lock(path):
        if not path.exists():
            return 0
        entries: List[Dict[str, Any]] = []
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.error("[LEARNING] Skipping unreadable journal line %d: %s", line_number, exc)
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        with file_lock(config.MEMORY_FILE):
            records = load_trade_memory()
            known = {_record_key(record) for record in records}
            recovered = [entry for entry in entries if _record_key(entry) not in known]
            if recovered:
                records.extend(recovered)
                records.sort(key=lambda record: str(record.get("timestamp") or ""))  # stable: keeps fill order
                write_json_atomic(config.MEMORY_FILE, records)
        path.unlink()
    if recovered:
        logger.info("[LEARNING] Recovered %d journaled fill(s) into %s", len(recovered), config.MEMORY_FILE.name)
    return len(recovered)


def append_trade_memory(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist an executed trade together with its execution-moment market context.

    A fill must never be lost: if memory.json cannot be written (e.g. the file
    is locked by another program), the record goes to an append-only journal
    and is merged back on the next successful write or at startup.
    """
    entry = dict(record)
    entry.setdefault("id", uuid.uuid4().hex)
    entry.setdefault("status", "CONFIRMED")
    entry.setdefault("timestamp", utc_now_iso())
    for field in ("exit_deal", "exit_price", "exit_time", "exit_reason", "realized_pnl", "outcome"):
        entry.setdefault(field, None)

    try:
        with file_lock(config.MEMORY_FILE):
            records = load_trade_memory()
            records.append(entry)
            write_json_atomic(config.MEMORY_FILE, records)
    except (OSError, TypeError, ValueError) as exc:
        _journal_pending(entry)
        logger.error("[LEARNING] memory.json write failed (%s); %s %s ticket %s journaled to %s",
                     exc, entry.get("symbol"), entry.get("side"), entry.get("ticket"), _pending_path().name)
        return entry

    logger.info("[LEARNING] Memory stored: %s %s ticket %s (%d records total)",
                entry.get("symbol"), entry.get("side"), entry.get("ticket"), len(records))
    if _pending_path().exists():
        try:
            flush_pending_journal()
        except OSError as exc:
            logger.warning("[LEARNING] Journal recovery deferred: %s", exc)
    return entry


def tag_untagged_trades(login: int, account_mode: str, broker: Optional[str], server: Optional[str]) -> int:
    """Label older trades of this login (recorded before DEMO/LIVE tagging) with the account type."""
    with file_lock(config.MEMORY_FILE):
        records = load_trade_memory()
        tagged = 0
        for record in records:
            if record.get("account_mode") or int(record.get("account_login") or 0) != int(login):
                continue
            record.update(account_mode=account_mode, broker=broker, server=server)
            tagged += 1
        if tagged:
            write_json_atomic(config.MEMORY_FILE, records)
    if tagged:
        logger.info("[LEARNING] Tagged %d earlier trade(s) of account %s as %s", tagged, login, account_mode)
    return tagged


def attach_close_execution(position_ticket: int, close_execution: Dict[str, Any]) -> bool:
    """Record a close (reversal or manual) on the trade that opened the position."""
    with file_lock(config.MEMORY_FILE):
        records = load_trade_memory()
        target = next((r for r in reversed(records) if int(r.get("ticket") or 0) == int(position_ticket)), None)
        if target is None:
            return False
        target["close_execution"] = close_execution
        write_json_atomic(config.MEMORY_FILE, records)
    return True


def recent_trades(limit: int = 50) -> List[Dict[str, Any]]:
    return load_trade_memory()[-limit:]


# -----------------------------------------------------------------------------
# Reconciliation against MT5 deal history
# -----------------------------------------------------------------------------
def _deal_cost(deal: Any) -> float:
    return (float(deal.profit) + float(deal.commission) + float(deal.swap)
            + float(getattr(deal, "fee", 0.0) or 0.0))


def _resolve_exit(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return exit fields for a closed position, or None while it is still open/unknown."""
    position_id = int(record.get("ticket") or 0)
    order_ticket = int(record.get("order") or 0)

    with MT5_LOCK:
        if position_id and mt5.positions_get(ticket=position_id):
            return None
        deals = mt5.history_deals_get(position=position_id) if position_id else None
        if not deals and order_ticket:
            orders = mt5.history_orders_get(ticket=order_ticket)
            if orders:
                position_id = int(orders[0].position_id)
                if mt5.positions_get(ticket=position_id):
                    return None
                deals = mt5.history_deals_get(position=position_id)
        if deals is None:
            logger.debug("[LEARNING] history_deals_get(%s) -> None (%s)", position_id, mt5_last_error())

    if not deals:
        return None
    exit_deals = [deal for deal in deals if deal.entry in _EXIT_ENTRIES]
    if not exit_deals:
        return None

    exit_volume = sum(float(deal.volume) for deal in exit_deals)
    exit_price = (sum(float(deal.price) * float(deal.volume) for deal in exit_deals) / exit_volume
                  if exit_volume else float(exit_deals[-1].price))
    last_exit = max(exit_deals, key=lambda deal: deal.time_msc)
    net_pnl = sum(_deal_cost(deal) for deal in deals)
    gross_pnl = sum(float(deal.profit) for deal in deals)

    if net_pnl > BREAKEVEN_TOLERANCE:
        outcome = "WIN"
    elif net_pnl < -BREAKEVEN_TOLERANCE:
        outcome = "LOSS"
    else:
        outcome = "BREAKEVEN"

    entry_deals = [deal for deal in deals if deal.entry == mt5.DEAL_ENTRY_IN]
    holding_minutes = None
    if entry_deals:
        opened = min(deal.time_msc for deal in entry_deals)
        holding_minutes = round((last_exit.time_msc - opened) / 60_000.0, 1)

    digits = int(record.get("digits") or (record.get("market_context") or {}).get("digits") or 5)
    return {
        "status": "CLOSED",
        "ticket": position_id,
        "exit_deal": int(last_exit.ticket),
        "exit_price": round(exit_price, digits),
        "exit_time": server_time_iso(last_exit.time),
        "exit_time_utc": server_epoch_to_utc_iso(last_exit.time),
        "exit_reason": _DEAL_REASONS.get(int(last_exit.reason), f"REASON_{int(last_exit.reason)}"),
        "realized_pnl": round(net_pnl, 2),
        "gross_pnl": round(gross_pnl, 2),
        "costs": round(net_pnl - gross_pnl, 2),
        "outcome": outcome,
        "holding_minutes": holding_minutes,
        "reconciled_at": utc_now_iso(),
    }


def reconcile_closed_trades() -> int:
    """Close out CONFIRMED records whose MT5 positions have exited. Returns the count."""
    pending = [r for r in load_trade_memory() if r.get("status") == "CONFIRMED" and not r.get("exit_deal")]
    if not pending:
        return 0

    with MT5_LOCK:
        try:
            account = mt5.account_info()
        except Exception:
            account = None
    if account is None:
        logger.debug("[LEARNING] Reconciliation skipped: MT5 account unavailable")
        return 0

    server_utc_offset_seconds()  # exit times are stored in real UTC too (loss cooldown)
    updates: Dict[str, Dict[str, Any]] = {}
    for record in pending:
        login = record.get("account_login")
        if login and int(login) != int(account.login):
            continue  # fill belongs to another account; can only reconcile there
        try:
            exit_fields = _resolve_exit(record)
        except Exception as exc:
            logger.warning("[LEARNING] Reconciliation of ticket %s failed: %s", record.get("ticket"), exc)
            continue
        if exit_fields:
            updates[_record_key(record)] = exit_fields

    if not updates:
        return 0

    reconciled: List[Dict[str, Any]] = []
    with file_lock(config.MEMORY_FILE):
        records = load_trade_memory()  # re-read so concurrent appends are preserved
        for record in records:
            fields = updates.get(_record_key(record))
            if fields and record.get("status") == "CONFIRMED":
                record.update(fields)
                reconciled.append(record)
        if reconciled:
            write_json_atomic(config.MEMORY_FILE, records)

    for record in reconciled:
        logger.info("[LEARNING] Reconciled %s %s ticket %s -> %s %+.2f (%s, exit %s)",
                    record.get("symbol"), record.get("side"), record.get("ticket"),
                    record.get("outcome"), record.get("realized_pnl") or 0.0,
                    record.get("exit_reason"), record.get("exit_price"))
    return len(reconciled)
