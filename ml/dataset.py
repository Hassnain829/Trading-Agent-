"""
Dataset for the setup filter (Phase 3.1).

One row per scalper setup:
* backtest rows: every in-session setup the rules find in the history (MT5 or the downloaded FXCM + HistData history), labelled
  with the triple barrier the live engine uses (stop / 1.5R target / 60-minute time stop) through the
  backtest's own fill code, at recorded spreads (``r``) and with every spread 1 pip wider (``r_stress``);
* live rows: real scalp fills from the decision journal joined to their closed trade in memory.json;
* shadow rows: resolved shadow trades (AI vetoes, guard skips, penalty blocks) with their features,
  weighted by SHADOW_WEIGHT.

Features are the 30 side-signed values of scalper.features() plus ``is_buy``. No setup is filtered
by trade overlap here: the validator replays the busy / cooldown / per-day rules on the chosen rows.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import backtest
import calibration
import config
import journal
import memory_store
import scalper
import shadow_store

FEATURES = [
    "d1_ema200_dist_atr", "d1_ema200_slope_atr", "d1_ema20_vs_50_atr", "d1_close_vs_ema50_atr", "d1_rsi", "d1_adx",
    "d1_atr_ratio", "d1_change_20d_pct", "h1_ema200_dist_atr", "h1_rsi", "h1_atr_ratio", "h1_change_24h_pct",
    "m15_close_vs_ema50_pct", "m15_ema50_slope_pct", "m5_rsi", "m5_rsi_extreme", "m5_pullback_bars", "m5_atr_pct",
    "m5_atr_ratio", "m5_rel_volume", "m5_ema20_vs_50_atr", "m5_close_vs_ema50_atr", "m5_body_ratio",
    "m5_close_location", "m5_retrace_pct", "m5_room_atr", "m5_swing_stop_atr", "spread_atr", "server_hour", "weekday",
    "is_buy",
]
STRESS_PIPS = 1.0
COLUMNS = ["source", "symbol", "side", "entry_time", "exit_time", "exit_time_stress", "day", "r", "r_stress", "win",
           "exit_reason", "weight"] + FEATURES


def feature_row(features: Dict[str, Any], side: str) -> Optional[Dict[str, Any]]:
    """The model inputs from a scalper.features() dict (None when it is from another feature version)."""
    if not isinstance(features, dict) or features.get("feature_version") != scalper.FEATURE_VERSION:
        return None
    row = {name: features.get(name) for name in FEATURES if name != "is_buy"}
    row["is_buy"] = 1.0 if side == "BUY" else 0.0
    return row


# -----------------------------------------------------------------------------
# Backtest rows
# -----------------------------------------------------------------------------
def setup_rows(symbols: List[str], days: int, source: str = "mt5",
               progress: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    ny7 = source in backtest.DOWNLOADED
    time_stop_bars = max(1, config.SCALP_TIME_STOP_MINUTES // 5)
    rows: List[Dict[str, Any]] = []
    for number, symbol in enumerate(symbols, start=1):
        if progress:
            progress(f"{symbol} ({number}/{len(symbols)}): labelling setups")
        try:
            frames = backtest.load_history(symbol, days, source)
            spec = backtest._spec(symbol, frames, ny7)
        except Exception as exc:
            if progress:
                progress(f"{symbol}: skipped ({exc})")
            continue
        if spec is None:
            continue
        pip = 10 if spec["digits"] in (3, 5) else 1
        prepared = scalper.prepare(frames)
        scan = backtest.scan_setups(prepared, spec["point"], days, 0.0, ny7)
        m5, spreads = prepared.a["M5"], scan["spreads"]
        stressed = spreads + STRESS_PIPS * pip * spec["point"]
        for i, setup in sorted(scan["setups"].items()):
            base = backtest.fill_setup(m5, spreads, i, setup, time_stop_bars)
            if base is None:  # the live engine would refuse it for its spread
                continue
            stress = backtest.fill_setup(m5, stressed, i, setup, time_stop_bars)
            features = feature_row(scalper.features(prepared, i, setup["side"], spread_price=float(spreads[i])),
                                   setup["side"])
            _, _, r, reason, j_exit = base
            rows.append({
                "source": "backtest", "symbol": symbol, "side": setup["side"],
                "entry_time": scan["utc_close"][i], "exit_time": scan["utc_close"][j_exit],
                "exit_time_stress": scan["utc_close"][stress[4]] if stress else pd.NaT,
                "day": scan["day_keys"][i], "r": round(r, 4), "r_stress": round(stress[2], 4) if stress else np.nan,
                "win": int(r > 0), "exit_reason": reason, "weight": 1.0, **features,
            })
    return _frame(rows)


# -----------------------------------------------------------------------------
# Live and shadow rows
# -----------------------------------------------------------------------------
def _utc(value: Any) -> Optional[pd.Timestamp]:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if stamp is pd.NaT:
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def live_rows(journal_days: int = 3650) -> pd.DataFrame:
    """Real scalp fills (journal features + closed result) and resolved scalp shadow trades."""
    rows: List[Dict[str, Any]] = []
    closed = {}
    for record in memory_store.load_trade_memory():
        r = calibration._r_multiple(record)
        if record.get("ticket") and r is not None:
            closed[int(record["ticket"])] = (record, r)
    for entry in journal.iter_entries(journal_days):
        ticket = entry.get("ticket")
        side = (entry.get("setup") or {}).get("side") or (entry.get("features") or {}).get("side")
        features = feature_row(entry.get("features"), side)
        if not ticket or features is None or int(ticket) not in closed:
            continue
        record, r = closed[int(ticket)]
        entered = _utc(entry.get("ts"))
        exited = _utc(record.get("exit_time_utc") or record.get("exit_time")) or entered
        rows.append({"source": "live", "symbol": entry.get("symbol") or record.get("symbol"), "side": side,
                     "entry_time": entered, "exit_time": exited, "exit_time_stress": exited,
                     "day": entry.get("day") or "", "r": round(float(r), 4), "r_stress": round(float(r), 4),
                     "win": int(r > 0), "exit_reason": record.get("exit_reason"), "weight": 1.0, **features})
    for shadow in shadow_store.load_shadows():
        if shadow.get("status") not in ("WIN", "LOSS", "TIMEOUT") or shadow.get("r_multiple") is None:
            continue
        features = feature_row(shadow.get("features"), shadow.get("side"))
        if features is None:
            continue
        r = float(shadow["r_multiple"])
        entered = _utc(shadow.get("created_at"))
        rows.append({"source": "shadow", "symbol": shadow.get("symbol"), "side": shadow.get("side"),
                     "entry_time": entered, "exit_time": _utc(shadow.get("resolved_at")) or entered,
                     "exit_time_stress": _utc(shadow.get("resolved_at")) or entered, "day": "",
                     "r": round(r, 4), "r_stress": round(r, 4), "win": int(r > 0), "exit_reason": shadow.get("status"),
                     "weight": config.SHADOW_WEIGHT, **features})
    return _frame(rows)


def _frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for column in ("entry_time", "exit_time", "exit_time_stress"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in FEATURES + ["r", "r_stress", "weight"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("entry_time", kind="stable").reset_index(drop=True)


def build(symbols: List[str], days: int, source: str = "mt5",
          progress: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    """Backtest setups plus live and shadow outcomes, oldest first."""
    frames = [setup_rows(symbols, days, source, progress), live_rows()]
    frames = [f for f in frames if len(f)]
    if not frames:
        return _frame([])
    return pd.concat(frames, ignore_index=True).sort_values("entry_time", kind="stable").reset_index(drop=True)


def save(frame: pd.DataFrame, name: str = "dataset") -> str:
    config.ML_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ML_DIR / f"{name}.pkl.gz"
    frame.to_pickle(path, compression="gzip")
    return str(path)


def describe(frame: pd.DataFrame) -> Dict[str, Any]:
    by_source = {source: {"rows": int(len(g)), "win_rate": round(float(g["win"].mean()) * 100, 1),
                          "expectancy_r": round(float(g["r"].mean()), 3)}
                 for source, g in frame.groupby("source")}
    return {"rows": int(len(frame)), "by_source": by_source,
            "from": str(frame["entry_time"].min()) if len(frame) else None,
            "to": str(frame["entry_time"].max()) if len(frame) else None,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
