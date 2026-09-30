"""
Five-plus years of M1 history in minutes, from two free public sources (backtest source "history").

* FXCM candle archive (candledata.fxcorporate.com): weekly M1 files with BID and ASK prices, so
  the real spread is known. Covers the seven FX pairs; a few weeks are missing.
* HistData.com: yearly / monthly M1 BID files in New York local time. Supplies gold (XAUUSD, not in the FXCM archive)
  and fills every day FXCM does not have. These minutes have no spread, so they get the pair's
  median FXCM spread for that hour of the day (gold: DEFAULT_SPREAD_POINTS).

Both are cached raw on disk (data/history/raw), so a download can stop and resume. Bars are
rebuilt on the broker's server clock (New York + 7h, like MT5): M5/M15/H1/D1 bid
bars with the spread in points, written to data/history/bars in the layout the backtest reads.

    .venv\\Scripts\\python.exe fxhistory.py all --years 5     (download + build, ~10-20 minutes)
    .venv\\Scripts\\python.exe fxhistory.py status
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

import config

DATA_DIR: Path = config.BASE_DIR / "data" / "history"
RAW_DIR, BARS_DIR = DATA_DIR / "raw", DATA_DIR / "bars"
FXCM_URL = "https://candledata.fxcorporate.com/m1/{code}/{year}/{week}.csv.gz"
HISTDATA_PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{year}{month}"
HISTDATA_GET = "https://www.histdata.com/get.php"
FXCM_CODES = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
              "AUDJPY", "EURCHF", "AUDCAD", "AUDNZD", "CADJPY", "CHFJPY", "EURAUD", "GBPCHF", "NZDJPY", "USDTRY"}
# Spread (points) for minutes without an ask price when the pair has no FXCM spread history at all.
DEFAULT_SPREAD_POINTS = {"XAUUSD": 25, "XAGUSD": 30}
FALLBACK_SPREAD_POINTS = 12
FXCM_WORKERS = 4
RECHECK_DAYS = 21  # recently missing FXCM weeks are asked for again (they may be published late)
NEW_YORK = "America/New_York"  # HistData stamps are New York local time (see _read_histdata)
DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD"]
TIMEFRAMES = {"M5": "5min", "M15": "15min", "H1": "1h", "D1": "1D"}


def instrument(symbol: str) -> str:
    """Broker name -> plain code (EURUSDm / EURUSD.r -> EURUSD)."""
    return re.sub(r"[^A-Z]", "", symbol.upper())[:6]


def point_size(code: str) -> float:
    """The broker-style point (EURUSD 0.00001, USDJPY 0.001, XAUUSD 0.01)."""
    if code.startswith(("XAU", "XAG")):
        return 0.01
    return 0.001 if code.endswith("JPY") else 0.00001


def to_server_time(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC -> the broker server clock (New York time + 7h), as naive timestamps."""
    return index.tz_convert(NEW_YORK).tz_localize(None) + pd.Timedelta(hours=7)
TIMEOUT = (8, 60)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (history downloader for backtests)"
    return session


def _get(session: requests.Session, url: str, data: Optional[Dict] = None, headers: Optional[Dict] = None,
         retries: int = 4) -> Optional[requests.Response]:
    """GET (or POST with ``data``); a 200 or 404 answer is final, anything else is retried."""
    for attempt in range(retries):
        try:
            response = (session.post(url, data=data, headers=headers, timeout=TIMEOUT) if data is not None
                        else session.get(url, headers=headers, timeout=TIMEOUT))
            if response.status_code in (200, 404):
                return response
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None


# -----------------------------------------------------------------------------
# FXCM weekly files
# -----------------------------------------------------------------------------
def _fxcm_file(code: str, year: int, week: int) -> Path:
    return RAW_DIR / "fxcm" / code / str(year) / f"{week:02d}.csv.gz"


def _week_start(year: int, week: int) -> date:
    """FXCM week 1 starts on the first Sunday of the year (approximately; used to skip future weeks)."""
    first = date(year, 1, 1)
    first_sunday = first + timedelta(days=(6 - first.weekday()) % 7)
    return first_sunday + timedelta(weeks=week - 1)


def _missing_path(code: str) -> Path:
    return RAW_DIR / "fxcm" / code / "missing.json"


def _download_fxcm(code: str, start: date, end: date, progress=print) -> Dict[str, int]:
    """Every week file from ``start`` to ``end`` (cached; 404 weeks remembered, recent ones re-asked)."""
    missing_file = _missing_path(code)
    missing = json.loads(missing_file.read_text()) if missing_file.exists() else {}
    jobs = []
    for year in range(start.year, end.year + 1):
        for week in range(1, 54):
            begins = _week_start(year, week)
            if begins > end or begins + timedelta(days=7) < start or _fxcm_file(code, year, week).exists():
                continue
            key = f"{year}-{week}"
            if key in missing and (end - begins).days > RECHECK_DAYS:
                continue  # known gap, filled from HistData
            jobs.append((year, week))
    counts = {"ok": 0, "missing": 0, "failed": 0}
    session = _session()

    def fetch(job):
        year, week = job
        response = _get(session, FXCM_URL.format(code=code, year=year, week=week))
        if response is None:
            return job, "failed"
        if response.status_code == 404 or not response.content.startswith(b"\x1f\x8b"):
            return job, "missing"
        target = _fxcm_file(code, year, week)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        return job, "ok"

    with ThreadPoolExecutor(max_workers=FXCM_WORKERS) as pool:
        for done, ((year, week), status) in enumerate(pool.map(fetch, jobs), 1):
            counts[status] += 1
            if status == "missing":
                missing[f"{year}-{week}"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if progress and (done % 50 == 0 or done == len(jobs)):
                progress(f"[HISTORY] FXCM {code}: {done}/{len(jobs)} weeks | {counts}")
    missing_file.parent.mkdir(parents=True, exist_ok=True)
    missing_file.write_text(json.dumps(missing, indent=0))
    return counts


def _read_fxcm(code: str) -> pd.DataFrame:
    frames = []
    for path in sorted((RAW_DIR / "fxcm" / code).glob("*/*.csv.gz")):
        try:
            df = pd.read_csv(io.BytesIO(gzip.decompress(path.read_bytes())))
        except (OSError, EOFError, ValueError, pd.errors.ParserError):
            continue
        if df.empty:
            continue
        index = pd.to_datetime(df["DateTime"], format="%m/%d/%Y %H:%M:%S.%f", utc=True)
        frames.append(pd.DataFrame({"open": df["BidOpen"].to_numpy(), "high": df["BidHigh"].to_numpy(),
                                    "low": df["BidLow"].to_numpy(), "close": df["BidClose"].to_numpy(),
                                    "spread": (df["AskClose"] - df["BidClose"]).clip(lower=0).to_numpy()},
                                   index=pd.DatetimeIndex(index)))
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "spread"])
    m1 = pd.concat(frames).sort_index()
    return m1[~m1.index.duplicated(keep="last")]


# -----------------------------------------------------------------------------
# HistData yearly / monthly files
# -----------------------------------------------------------------------------
def _histdata_file(code: str, year: int, month: Optional[int]) -> Path:
    return RAW_DIR / "histdata" / code / (f"{year}.zip" if month is None else f"{year}{month:02d}.zip")


def _download_histdata(code: str, start: date, end: date, progress=print) -> Dict[str, int]:
    """Whole past years as one file each; the current year month by month (the last month is refreshed)."""
    session = _session()
    counts = {"ok": 0, "cached": 0, "missing": 0, "failed": 0}
    jobs = [(year, None) for year in range(start.year, end.year)]
    jobs += [(end.year, month) for month in range(1, end.month + 1)]
    for year, month in jobs:
        target = _histdata_file(code, year, month)
        current = month is not None and (year, month) == (end.year, end.month)
        if target.exists() and not current:
            counts["cached"] += 1
            continue
        page_url = HISTDATA_PAGE.format(pair=code.lower(), year=year, month=f"/{month}" if month else "")
        page = _get(session, page_url)
        form = dict(re.findall(r'<input type="hidden" name="(\w+)" id="\w+" value="([^"]*)"', page.text)) if page else {}
        if "tk" not in form:
            counts["failed" if page is None else "missing"] += 1
            continue
        response = _get(session, HISTDATA_GET, data=form, headers={"Referer": page_url})
        if response is None or not response.content.startswith(b"PK"):
            counts["missing" if response is not None else "failed"] += 1  # e.g. the running month is not out yet
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        counts["ok"] += 1
        time.sleep(1.0)  # one request at a time, politely
    if progress:
        progress(f"[HISTORY] HistData {code}: {counts}")
    return counts


def _read_histdata(code: str) -> pd.DataFrame:
    frames = []
    for path in sorted((RAW_DIR / "histdata" / code).glob("*.zip")):
        try:
            archive = zipfile.ZipFile(path)
            name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
            df = pd.read_csv(io.BytesIO(archive.read(name)), sep=";", header=None,
                             names=["t", "open", "high", "low", "close", "volume"])
        except (OSError, StopIteration, zipfile.BadZipFile, ValueError, pd.errors.ParserError):
            continue
        # New York local time WITH daylight saving (checked against FXCM and broker bars: the site says
        # "EST without DST", but summer data is one hour off under that reading). Minutes that do not
        # exist or repeat at the clock change fall in the weekend close; they are dropped.
        local = pd.to_datetime(df["t"], format="%Y%m%d %H%M%S")
        index = local.dt.tz_localize(NEW_YORK, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        keep = index.notna().to_numpy()
        df, index = df[keep], index[keep]
        frames.append(pd.DataFrame({"open": df["open"].to_numpy(), "high": df["high"].to_numpy(),
                                    "low": df["low"].to_numpy(), "close": df["close"].to_numpy()},
                                   index=pd.DatetimeIndex(index)))
    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    m1 = pd.concat(frames).sort_index()
    return m1[~m1.index.duplicated(keep="last")]


# -----------------------------------------------------------------------------
# Merge + bars
# -----------------------------------------------------------------------------
def merged_m1(code: str) -> tuple:
    """FXCM minutes where FXCM has the day, HistData for every other day. Returns (m1, info)."""
    fxcm, hist = _read_fxcm(code), _read_histdata(code)
    point = point_size(code)
    if len(fxcm):
        hourly = fxcm["spread"].groupby(fxcm.index.hour).median()
        overall = float(fxcm["spread"].median())
    else:
        hourly, overall = pd.Series(dtype=float), DEFAULT_SPREAD_POINTS.get(code, FALLBACK_SPREAD_POINTS) * point
    fxcm_days = pd.Index(fxcm.index.normalize().unique()) if len(fxcm) else pd.Index([])
    fill = hist[~hist.index.normalize().isin(fxcm_days)] if len(hist) else hist
    if len(fill):
        fill = fill.assign(spread=fill.index.hour.map(hourly).to_numpy(dtype=float) if len(hourly) else np.nan)
        fill["spread"] = fill["spread"].fillna(overall)
    m1 = (pd.concat([fxcm, fill]).sort_index() if len(fill) else fxcm).astype(float)  # empty frames are object-typed
    info = {"fxcm_minutes": int(len(fxcm)), "histdata_minutes": int(len(fill)),
            "spread_source": "FXCM bid/ask" if len(fxcm) else f"fixed {DEFAULT_SPREAD_POINTS.get(code, FALLBACK_SPREAD_POINTS)} points",
            "median_spread_points": round(overall / point, 1)}
    return m1, info


def build(symbols: List[str], progress=print) -> Dict[str, Dict]:
    """M5/M15/H1/D1 bid bars with the mean spread in points, on the broker's server clock."""
    BARS_DIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict] = {}
    for symbol in symbols:
        code = instrument(symbol)
        m1, info = merged_m1(code)
        if m1.empty:
            if progress:
                progress(f"[HISTORY] {code}: nothing downloaded yet")
            continue
        m1 = m1.copy()
        m1.index = to_server_time(m1.index)
        point = point_size(code)
        counts = {}
        for tf, rule in TIMEFRAMES.items():
            bars = m1.resample(rule, label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean"}).dropna(subset=["open"])
            out = pd.DataFrame({
                "time": bars.index.values.astype("datetime64[s]").astype("int64"),
                "open": bars["open"].to_numpy(), "high": bars["high"].to_numpy(), "low": bars["low"].to_numpy(),
                "close": bars["close"].to_numpy(),
                "tick_volume": np.zeros(len(bars), dtype="int64"),  # neither source has volume
                "spread": np.round(bars["spread"].fillna(bars["spread"].median()).to_numpy() / point).astype("int64"),
            })
            out.to_pickle(BARS_DIR / f"{code}_{tf}.pkl.gz", compression="gzip")
            counts[tf] = len(out)
        gaps = np.diff(m1.index.values.astype("datetime64[s]").astype("int64")) / 86400.0
        info.update(counts, first=f"{m1.index[0]:%Y-%m-%d}", last=f"{m1.index[-1]:%Y-%m-%d}",
                    gaps_over_4_days=int((gaps > 4).sum()))
        (BARS_DIR / f"{code}_info.json").write_text(json.dumps(info, indent=1))
        summary[code] = info
        if progress:
            progress(f"[HISTORY] {code}: built {counts} from {info['first']} to {info['last']} | "
                     f"FXCM {info['fxcm_minutes']:,} + HistData {info['histdata_minutes']:,} minutes | "
                     f"spread {info['spread_source']} (median {info['median_spread_points']} pts) | "
                     f"gaps > 4 days: {info['gaps_over_4_days']}")
    return summary


def load_bars(symbol: str, tf: str) -> pd.DataFrame:
    path = BARS_DIR / f"{instrument(symbol)}_{tf}.pkl.gz"
    if not path.exists():
        raise FileNotFoundError(f"no history {tf} bars for {symbol}; run: python fxhistory.py all --years 5")
    return pd.read_pickle(path, compression="gzip")


def available(symbol: str) -> bool:
    return all((BARS_DIR / f"{instrument(symbol)}_{tf}.pkl.gz").exists() for tf in TIMEFRAMES)


def info(symbol: str) -> Dict:
    path = BARS_DIR / f"{instrument(symbol)}_info.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def download(symbols: List[str], years: float, progress=print) -> None:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=int(years * 365.25) + 7)
    for symbol in symbols:
        code = instrument(symbol)
        if code in FXCM_CODES:
            _download_fxcm(code, start, end, progress)
        _download_histdata(code, start, end, progress)


def status(symbols: List[str]) -> None:
    for symbol in symbols:
        code = instrument(symbol)
        weeks = len(list((RAW_DIR / "fxcm" / code).glob("*/*.csv.gz")))
        months = len(list((RAW_DIR / "histdata" / code).glob("*.zip")))
        built = info(symbol)
        print(f"{code}: FXCM {weeks} weeks, HistData {months} files; bars "
              + (f"{built.get('first')} → {built.get('last')}, spread {built.get('spread_source')}, "
                 f"gaps > 4 days: {built.get('gaps_over_4_days')}" if available(symbol) else "not built"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download and build FXCM + HistData history for backtests and ML.")
    parser.add_argument("command", choices=("download", "build", "status", "all"))
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    args = parser.parse_args(argv)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    started = time.monotonic()
    if args.command in ("download", "all"):
        download(symbols, args.years)
    if args.command in ("build", "all"):
        build(symbols)
    if args.command == "status":
        status(symbols)
    else:
        print(f"[HISTORY] done in {(time.monotonic() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
