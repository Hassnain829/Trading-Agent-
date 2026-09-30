"""
Fixed evaluator of the research loop (karpathy/autoresearch pattern). Do not edit during research.

One number decides: the walk-forward out-of-sample expectancy (R per trade) of research/candidate.py,
with every spread 1 pip wider than recorded, on the research period only. The most recent year
(20% of the history when less than 3 years are loaded) is LOCKED: it is never scored here and is
opened only with --holdout, once, for the final candidate.

A candidate is KEPT (becomes the new best) when it has enough trades and beats the best score by at
least MIN_GAIN. Every run is appended to research/results.tsv; the Deflated Sharpe Ratio uses the
total number of variants ever tried, so the bar rises with each experiment.

    .venv\\Scripts\\python.exe research\\evaluate.py                  # score candidate.py
    .venv\\Scripts\\python.exe research\\evaluate.py --restore-best   # candidate.py <- best so far
    .venv\\Scripts\\python.exe research\\evaluate.py --holdout        # final check on the locked year
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest  # noqa: E402
import config  # noqa: E402
import data_engine  # noqa: E402
import validation  # noqa: E402

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "candidate.py"
RESULTS = HERE / "results.tsv"
BEST = HERE / "best.json"
HOLDOUT_LOG = HERE / "holdout_log.jsonl"

STRESS_PIPS = 1.0
MIN_GAIN = 0.01  # R per trade the walk-forward score must improve by
MIN_TRADES_LONG, MIN_TRADES_SHORT = 150, 40  # research-period trades (3+ years / shorter history)
ALLOWED = {
    "SCALP_RSI_PULLBACK": (10.0, 50.0), "SCALP_PULLBACK_BARS": (1, 12), "SCALP_REWARD_RISK": (1.0, 5.0),
    "SCALP_SL_ATR_MIN": (0.3, 3.0), "SCALP_SL_ATR_MAX": (0.5, 5.0), "SCALP_SWING_BARS": (2, 24),
    "SCALP_TIME_STOP_MINUTES": (5, 480), "SCALP_MAX_TRADES_PER_SYMBOL": (1, 20),
    "SCALP_STRICT_GUARD": bool, "SCALP_MEDIUM_TREND": bool, "SCALP_MIN_ADX": (0.0, 60.0),
    "SCALP_PULLBACK_TO_EMA": bool, "SCALP_TRIGGER": ("RSI", "BREAK"), "SCALP_ROOM_MIN_R": (0.0, 5.0),
    "SCALP_ROOM_BARS": (6, 288), "SCALP_TARGET": ("RR", "STRUCTURE"), "SCALP_MIN_TARGET_R": (0.5, 5.0),
    "SCALP_SESSION_START_LONDON": (0, 23), "SCALP_SESSION_END_NEW_YORK": (1, 17),
    "LOSS_COOLDOWN_MINUTES": (0, 1440),
}
COLUMNS = ["time", "id", "source", "days", "exit", "params", "note", "trades", "wf_expectancy_r", "wf_positive_windows",
           "expectancy_r", "dsr", "trials", "best_before", "status"]


# -----------------------------------------------------------------------------
# Candidate
# -----------------------------------------------------------------------------
def load_candidate(path: Path = CANDIDATE) -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("research_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = dict(getattr(module, "PARAMS", {}) or {})
    exit_mode = str(getattr(module, "EXIT_MODE", "FIXED")).upper()
    problems = []
    for name, value in params.items():
        rule = ALLOWED.get(name)
        if rule is None:
            problems.append(f"{name} is not a research knob")
        elif rule is bool:
            if not isinstance(value, bool):
                problems.append(f"{name} must be True/False")
        elif isinstance(rule[0], str):
            if value not in rule:
                problems.append(f"{name} must be one of {rule}")
        elif not isinstance(value, (int, float)) or isinstance(value, bool) or not rule[0] <= value <= rule[1]:
            problems.append(f"{name}={value!r} outside {rule[0]}..{rule[1]}")
    if exit_mode not in backtest.EXIT_MODES:
        problems.append(f"EXIT_MODE must be one of {backtest.EXIT_MODES}")
    if problems:
        raise SystemExit("candidate.py rejected: " + "; ".join(problems))
    key = json.dumps({"params": params, "exit": exit_mode}, sort_keys=True)
    return {"params": params, "exit": exit_mode, "note": str(getattr(module, "NOTE", "")).strip(),
            "id": hashlib.sha1(key.encode()).hexdigest()[:8]}


def apply(params: Dict[str, Any]) -> None:
    for name, value in params.items():
        setattr(config, name, type(getattr(config, name))(value))


def write_candidate(candidate: Dict[str, Any], path: Path = CANDIDATE) -> None:
    body = "".join(f"    {json.dumps(k)}: {v!r},\n" for k, v in sorted(candidate["params"].items()))
    path.write_text(CANDIDATE_TEMPLATE.format(params=body, exit=candidate["exit"],
                                              note=candidate.get("note", "").replace('"', "'")), encoding="utf-8")


CANDIDATE_TEMPLATE = '''"""
The ONE file the research loop edits (karpathy/autoresearch pattern; see research/program.md).

PARAMS overrides strategy knobs for a backtest only; live trading is never changed from here.
Only the names in evaluate.ALLOWED are accepted. Leave PARAMS empty to test the current rules.
"""

PARAMS = {{
{params}}}
EXIT_MODE = "{exit}"  # FIXED | BREAKEVEN | PARTIAL | TRAIL
NOTE = "{note}"  # one line: the idea behind this candidate
'''


# -----------------------------------------------------------------------------
# Data split
# -----------------------------------------------------------------------------
def holdout_start(period: Dict[str, Optional[str]]) -> str:
    """The locked tail: the last 365 days of 3+ years of history, else the last 20% of it."""
    start, end = datetime.fromisoformat(period["from"]), datetime.fromisoformat(period["to"])
    span = end - start
    locked = timedelta(days=365) if span >= timedelta(days=3 * 365) else span * 0.2
    return (end - locked).isoformat()


def split(trades: List[Dict[str, Any]], boundary: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return ([t for t in trades if t["entry_time"] < boundary], [t for t in trades if t["entry_time"] >= boundary])


def score(trades: List[Dict[str, Any]], trials: int) -> Dict[str, Any]:
    train, test = validation.window_plan(trades)
    wf = validation.walk_forward({"candidate": trades}, train, test)
    sig = validation.significance([t["r"] for t in trades], trials)
    rs = [t["r"] for t in trades]
    return {"trades": len(trades), "wf_expectancy_r": wf["oos"]["expectancy_r"], "wf_trades": wf["oos"]["trades"],
            "wf_positive_windows": wf["positive_windows_share"], "wf_windows": wf["windows_count"],
            "expectancy_r": round(sum(rs) / len(rs), 3) if rs else None, "dsr": sig.get("dsr"),
            "t_stat": sig.get("t_stat"), "trials": trials}


# -----------------------------------------------------------------------------
# Log
# -----------------------------------------------------------------------------
def read_results() -> List[Dict[str, str]]:
    if not RESULTS.exists():
        return []
    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    return [dict(zip(COLUMNS, line.split("\t"))) for line in lines[1:] if line.strip()]


def append_result(row: Dict[str, Any]) -> None:
    new = not RESULTS.exists()
    with RESULTS.open("a", encoding="utf-8") as handle:
        if new:
            handle.write("\t".join(COLUMNS) + "\n")
        handle.write("\t".join(str(row.get(c, "")).replace("\t", " ") for c in COLUMNS) + "\n")


def load_best() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(BEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
def dataset(source: str, days: Optional[int]) -> Tuple[str, int, List[str]]:
    import settings_store
    settings_store.load_saved()  # the same saved strategy settings the live engine uses
    symbols = list(config.SYMBOLS_DEMO)
    if source == "auto":
        source = backtest.best_source(symbols)
    if source == "mt5":
        if not data_engine.initialize_mt5():
            raise SystemExit("MT5 is not available (and no 5-year history is built: python fxhistory.py all --years 5.5)")
        offered = [item["name"] for item in data_engine.broker_symbols() or []]
        symbols, _, missing = data_engine.resolve_symbols(symbols, offered)
        if missing:
            print(f"Not offered and skipped: {', '.join(missing)}")
    return source, int(days or backtest.default_days(source)), symbols


def run(candidate: Dict[str, Any], source: str, days: int, symbols: List[str]) -> Dict[str, Any]:
    apply(candidate["params"])
    report = backtest.run_backtest(symbols, days, 1.0, 10_000.0, extra_spread_points=STRESS_PIPS, source=source,
                                   exit_mode=candidate["exit"], save=False)
    return report


def evaluate(source: str = "auto", days: Optional[int] = None) -> Dict[str, Any]:
    candidate = load_candidate()
    source, days, symbols = dataset(source, days)
    data_key = f"{source}:{days}:{','.join(sorted(symbols))}"
    best = load_best()
    if best and best.get("data_key") != data_key:
        print(f"Best so far was scored on other data ({best.get('data_key')}); this run sets a new baseline.")
        best = None
    trials = config.BACKTEST_TRIALS + len(read_results()) + 1
    report = run(candidate, source, days, symbols)
    boundary = holdout_start(report["period"])
    research, _ = split(report["all_trades"], boundary)
    result = score(research, trials)
    span = datetime.fromisoformat(boundary) - datetime.fromisoformat(report["period"]["from"])
    min_trades = MIN_TRADES_LONG if span >= timedelta(days=2 * 365) else MIN_TRADES_SHORT
    best_score = (best or {}).get("wf_expectancy_r")
    if result["trades"] < min_trades or result["wf_expectancy_r"] is None:
        status = f"rejected: {result['trades']} trades < {min_trades}"
    elif best_score is None or result["wf_expectancy_r"] >= best_score + MIN_GAIN:
        status = "KEPT (new best)"
    else:
        status = f"discarded: not {MIN_GAIN}R better than {best_score}"
    append_result({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "id": candidate["id"],
                   "source": source, "days": days, "exit": candidate["exit"],
                   "params": json.dumps(candidate["params"], sort_keys=True), "note": candidate["note"],
                   **result, "best_before": best_score, "status": status})
    if status.startswith("KEPT"):
        BEST.write_text(json.dumps({**candidate, **result, "data_key": data_key, "holdout_from": boundary,
                                    "kept_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2),
                        encoding="utf-8")
    print(f"{candidate['id']} {status} | walk-forward {result['wf_expectancy_r']}R/trade over {result['wf_trades']} "
          f"unseen trades ({result['wf_positive_windows']} of {result['wf_windows']} windows positive) at +{STRESS_PIPS:g} "
          f"pip | all research trades {result['expectancy_r']}R x {result['trades']} | DSR {result['dsr']} after "
          f"{trials} trials | holdout from {boundary[:10]} locked")
    return {"status": status, **result}


def holdout(source: str = "auto", days: Optional[int] = None) -> Dict[str, Any]:
    """Score the BEST candidate once on the locked period. Every opening is logged."""
    best = load_best()
    if not best:
        raise SystemExit("no best candidate yet: run the evaluator first")
    opened = [json.loads(line) for line in HOLDOUT_LOG.read_text(encoding="utf-8").splitlines()] \
        if HOLDOUT_LOG.exists() else []
    if opened:
        print(f"WARNING: the holdout was opened {len(opened)} time(s) before; results that were tuned after "
              f"looking at it are no longer unseen.")
    source, days, symbols = dataset(source, days)
    report = run(best, source, days, symbols)
    _, locked = split(report["all_trades"], best["holdout_from"])
    trials = config.BACKTEST_TRIALS + len(read_results())
    result = score(locked, trials)
    passed = (result["expectancy_r"] or 0) > 0
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "id": best["id"],
             "params": best["params"], "exit": best["exit"], **result, "passed": passed}
    with HOLDOUT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    print(f"HOLDOUT {best['id']}: {result['expectancy_r']}R/trade x {result['trades']} trades at +{STRESS_PIPS:g} pip "
          f"-> {'PASSED: may be adopted (set the knobs in .env), then demo-test' if passed else 'FAILED: do not adopt'}")
    return entry


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Research loop evaluator (fixed).")
    parser.add_argument("--source", choices=("auto", *backtest.SOURCES), default="auto")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--holdout", action="store_true", help="final one-time check of the best candidate")
    parser.add_argument("--restore-best", action="store_true", help="rewrite candidate.py from the best so far")
    args = parser.parse_args(argv)
    if args.restore_best:
        best = load_best()
        if not best:
            raise SystemExit("no best candidate yet")
        write_candidate(best)
        print(f"candidate.py restored to {best['id']} ({best.get('note')})")
        return 0
    try:
        (holdout if args.holdout else evaluate)(args.source, args.days)
    finally:
        data_engine.shutdown_mt5()
    return 0


if __name__ == "__main__":
    sys.exit(main())
