"""
Dukascopy history: years of real bid/ask data for backtests and machine learning.

Dukascopy (a Swiss bank) publishes its price feed for free. This module downloads the
daily M1 candle files for both the BID and the ASK side, caches every day on disk (so a
download can stop and resume at any time), and builds M5 / M15 / H1 / D1 bid bars with the
real spread in points - the same layout the MT5 backtest uses.

Time alignment: Dukascopy stamps are UTC. Bars are rebuilt on the broker's server clock
(New York time + 7h, the standard MT5 convention), so a D1 bar closes at 17:00 New York
exactly like the broker's, and the backtest reads both sources the same way.

The feed throttles fast clients, so the downloader is deliberately gentle: two workers, a
pause between requests, and growing waits after refusals. Five years of 8 pairs is about
21,000 files and takes several hours; it can run in the background and be restarted.

    .venv\\Scripts\\python.exe dukascopy.py download --years 5
    .venv\\Scripts\\python.exe dukascopy.py build
    .venv\\Scripts\\python.exe dukascopy.py status
"""
from __future__ import annotations

import argparse
import logging
import lzma
import re
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

import config

logger = logging.getLogger("hedgefund.dukascopy")

DATA_DIR: Path = config.BASE_DIR / "data" / "dukascopy"
RAW_DIR, BARS_DIR = DATA_DIR / "raw", DATA_DIR / "bars"
URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month0:02d}/{day:02d}/{side}_candles_min_1.bi5"
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD"]
TIMEFRAMES = {"M5": "5min", "M15": "15min", "H1": "1h", "D1": "1D"}
_NEW_YORK = ZoneInfo("America/New_York")
_RECORD = struct.Struct(">5if")  # seconds into the day, open, close, low, high (scaled ints), volume

# The feed rate-limits an IP for a while after bursts (503 for every file), so the defaults are patient:
# one worker, a pause between files, and long waits when refused. It runs unattended and resumes.
WORKERS = 1
PAUSE_SECONDS = 1.5
TIMEOUT = (8, 40)  # connect, read: the feed sometimes hangs a connection; give up and retry that file
RETRY_SECONDS = (2, 5, 15, 45, 120)  # per file, after a hung connection
THROTTLE_SECONDS = (60, 180, 600, 900)  # everyone waits, when the feed explicitly answers 429 / 503


# -----------------------------------------------------------------------------
# Instruments
# -----------------------------------------------------------------------------
def instrument(symbol: str) -> str:
    """Broker name -> Dukascopy code (EURUSDm / EURUSD.r -> EURUSD)."""
    return re.sub(r"[^A-Z]", "", symbol.upper())[:6]


def price_scale(code: str) -> float:
    """Dukascopy stores prices as integers: 1e5 for most FX, 1e3 for JPY pairs and gold."""
    return 1e3 if code.endswith("JPY") or code.startswith(("XAU", "XAG")) else 1e5


def point_size(code: str) -> float:
    """The broker-style point (EURUSD 0.00001, USDJPY 0.001, XAUUSD 0.01)."""
    if code.startswith(("XAU", "XAG")):
        return 0.01
    return 0.001 if code.endswith("JPY") else 0.00001


# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
class _Throttle:
    """Shared pause for all workers after the feed refuses a request."""

    def __init__(self) -> None:
        self.until = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self.lock:
                remaining = self.until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 5))

    def pause(self, seconds: float) -> None:
        with self.lock:
            self.until = max(self.until, time.monotonic() + seconds)


def _day_file(code: str, day: date, side: str) -> Path:
    return RAW_DIR / code / f"{day:%Y}" / f"{day:%Y-%m-%d}_{side}.npy"


def _decode(content: bytes, scale: float) -> np.ndarray:
    """Rows of [seconds, open, close, low, high, volume] with real prices."""
    if not content:
        return np.empty((0, 6))
    raw = lzma.decompress(content, format=lzma.FORMAT_ALONE)
    rows = np.array([_RECORD.unpack_from(raw, offset) for offset in range(0, len(raw) - len(raw) % 24, 24)],
                    dtype=float).reshape(-1, 6)
    rows[:, 1:5] /= scale
    return rows


def _fetch_day(session: requests.Session, throttle: _Throttle, code: str, day: date, side: str) -> str:
    """Download one day file into the cache. Returns 'ok', 'empty', 'cached' or 'failed'."""
    target = _day_file(code, day, side)
    if target.exists():
        return "cached"
    url = URL.format(symbol=code, year=day.year, month0=day.month - 1, day=day.day, side=side)
    refusals = 0
    for backoff in (*RETRY_SECONDS, None):
        throttle.wait()
        try:
            response = session.get(url, timeout=TIMEOUT)
        except requests.RequestException:
            response = None  # a hung or dropped connection: retry this file soon
        if response is not None and response.status_code in (200, 404):
            try:
                rows = _decode(response.content, price_scale(code)) if response.status_code == 200 else np.empty((0, 6))
            except lzma.LZMAError:
                rows = None  # truncated download: retry
            if rows is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, rows)
                time.sleep(PAUSE_SECONDS)
                return "ok" if len(rows) else "empty"
        elif response is not None and response.status_code in (429, 503):
            throttle.pause(THROTTLE_SECONDS[min(refusals, len(THROTTLE_SECONDS) - 1)])  # the feed asks us to slow down
            refusals += 1
        if backoff is None:
            break
        time.sleep(backoff)
    return "failed"  # left uncached: the next run retries it


def trading_days(years: float, end: Optional[date] = None) -> List[date]:
    """Weekdays (plus Sundays, when the week opens) from ``years`` ago until two days ago, newest first."""
    end = end or (datetime.now(timezone.utc).date() - timedelta(days=2))  # daily files appear with a delay
    start = end - timedelta(days=int(years * 365.25))
    days, day = [], end
    while day >= start:
        if day.weekday() != 5:  # Saturdays are always empty
            days.append(day)
        day -= timedelta(days=1)
    return days


def download(symbols: List[str], years: float, progress: bool = True) -> Dict[str, Dict[str, int]]:
    """Fetch every missing day file (BID and ASK), newest first so a partial run is already useful."""
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    throttle = _Throttle()
    days = trading_days(years)
    summary: Dict[str, Dict[str, int]] = {}
    for symbol in symbols:
        code = instrument(symbol)
        jobs = [(day, side) for day in days for side in ("BID", "ASK") if not _day_file(code, day, side).exists()]
        counts = {"ok": 0, "empty": 0, "cached": 2 * len(days) - len(jobs), "failed": 0}
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for done, status in enumerate(pool.map(lambda job: _fetch_day(session, throttle, code, *job), jobs), 1):
                counts[status] += 1
                if progress and (done % 100 == 0 or done == len(jobs)):
                    rate = done / max(time.monotonic() - started, 1)
                    left = (len(jobs) - done) / rate / 60 if rate else 0
                    print(f"[DUKASCOPY] {code}: {done}/{len(jobs)} files | ok {counts['ok']} empty {counts['empty']} "
                          f"failed {counts['failed']} | ~{left:.0f} min left for this pair", flush=True)
        summary[code] = counts
        print(f"[DUKASCOPY] {code} done: {counts}", flush=True)
    return summary


# -----------------------------------------------------------------------------
# Build bars
# -----------------------------------------------------------------------------
def _m1(code: str) -> pd.DataFrame:
    """All cached M1 bid candles with the ask close, indexed by UTC time."""
    frames = []
    for bid_file in sorted((RAW_DIR / code).glob("*/*_BID.npy")):
        day = datetime.strptime(bid_file.name[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        ask_file = bid_file.with_name(bid_file.name.replace("_BID", "_ASK"))
        bid = np.load(bid_file)
        if not len(bid) or not ask_file.exists():
            continue
        ask = np.load(ask_file)
        frame = pd.DataFrame(bid[:, 1:], columns=["open", "close", "low", "high", "volume"],
                             index=pd.DatetimeIndex(day + pd.to_timedelta(bid[:, 0], unit="s")))
        if len(ask):
            ask_close = pd.Series(ask[:, 2], index=pd.DatetimeIndex(day + pd.to_timedelta(ask[:, 0], unit="s")))
            frame["spread"] = (ask_close.reindex(frame.index) - frame["close"]).clip(lower=0)
        else:
            frame["spread"] = np.nan
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated(keep="last")]
    return m1[m1["volume"] > 0]  # Dukascopy repeats the last price with zero volume while the market is shut


def to_server_time(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC -> the broker server clock (New York time + 7h), as naive timestamps."""
    return index.tz_convert(_NEW_YORK).tz_localize(None) + pd.Timedelta(hours=7)


def build(symbols: List[str]) -> Dict[str, Dict[str, int]]:
    """M5/M15/H1/D1 bid bars with the mean spread in points, on the broker's server clock."""
    BARS_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, int]] = {}
    for symbol in symbols:
        code = instrument(symbol)
        m1 = _m1(code)
        if m1.empty:
            print(f"[DUKASCOPY] {code}: nothing downloaded yet")
            continue
        m1.index = to_server_time(m1.index)
        point = point_size(code)
        summary[code] = {}
        for tf, rule in TIMEFRAMES.items():
            bars = m1.resample(rule, label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "spread": "mean"})
            bars = bars.dropna(subset=["open"])
            out = pd.DataFrame({
                "time": (bars.index.values.astype("datetime64[s]").astype("int64")),
                "open": bars["open"].to_numpy(), "high": bars["high"].to_numpy(),
                "low": bars["low"].to_numpy(), "close": bars["close"].to_numpy(),
                "tick_volume": np.round(bars["volume"].to_numpy() * 100).astype("int64"),
                "spread": np.round(bars["spread"].fillna(bars["spread"].median()).to_numpy() / point).astype("int64"),
            })
            out.to_pickle(BARS_DIR / f"{code}_{tf}.pkl.gz", compression="gzip")
            summary[code][tf] = len(out)
        first, last = m1.index[0], m1.index[-1]
        print(f"[DUKASCOPY] {code}: built {summary[code]} from {first:%Y-%m-%d} to {last:%Y-%m-%d} (server time)")
    return summary


def load_bars(symbol: str, tf: str) -> pd.DataFrame:
    """Built bars for a broker symbol, oldest first (same columns as data_engine.fetch_bars)."""
    path = BARS_DIR / f"{instrument(symbol)}_{tf}.pkl.gz"
    if not path.exists():
        raise FileNotFoundError(f"no Dukascopy {tf} bars for {symbol}; run: python dukascopy.py download && "
                                f"python dukascopy.py build")
    return pd.read_pickle(path, compression="gzip")


def available(symbol: str) -> bool:
    return all((BARS_DIR / f"{instrument(symbol)}_{tf}.pkl.gz").exists() for tf in TIMEFRAMES)


def status(symbols: List[str], years: float) -> None:
    days = trading_days(years)
    for symbol in symbols:
        code = instrument(symbol)
        have = sum(_day_file(code, day, side).exists() for day in days for side in ("BID", "ASK"))
        print(f"{code}: {have}/{2 * len(days)} day files cached ({have / (2 * len(days)):.0%}); "
              f"bars built: {'yes' if available(symbol) else 'no'}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download and build Dukascopy bid/ask history.")
    parser.add_argument("command", choices=("download", "build", "status", "all"))
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = parser.parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.command in ("download", "all"):
        download(symbols, args.years)
    if args.command in ("build", "all"):
        build(symbols)
    if args.command == "status":
        status(symbols, args.years)
    return 0


if __name__ == "__main__":
    sys.exit(main())
