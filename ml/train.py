"""
Build the dataset, validate both models with the purged walk-forward, and save the result (Phase 3).

    .venv\\Scripts\\python.exe -m ml.train                     # Dukascopy if complete, else broker history
    .venv\\Scripts\\python.exe -m ml.train --source mt5 --days 330

Writes data/ml/dataset.pkl.gz, data/ml/validation-<time>.json and data/ml/model.pkl. The saved model is
marked approved only when the walk-forward accepted it; the live filter ignores unapproved models.
Every run counts as one more trial for the Deflated Sharpe test.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import config
import data_engine
import dukascopy
import scalper
from ml import dataset, validate

MODEL_FILE = "model.pkl"


def trials() -> int:
    runs = len(list(config.ML_DIR.glob("validation-*.json"))) if config.ML_DIR.exists() else 0
    return config.BACKTEST_TRIALS + runs + 1


def train(symbols: List[str], days: int, source: str,
          progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    started = datetime.now(timezone.utc)
    data = dataset.build(symbols, days, source, progress)
    if data.empty:
        raise RuntimeError("no setups found: nothing to train on")
    dataset.save(data)
    if progress:
        progress(f"dataset: {dataset.describe(data)}")
        progress("walk-forward: fitting LightGBM and logistic regression per fold")
    report = validate.walk_forward(data, trials())
    report.update(dataset=dataset.describe(data), source=source, days=days, symbols=symbols,
                  feature_version=scalper.FEATURE_VERSION, generated_at=started.isoformat(timespec="seconds"))
    config.ML_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ML_DIR / f"validation-{started:%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    name = report.get("accepted")
    saved = {"approved": bool(name), "model_name": name, "reason": report.get("reason"),
             "features": dataset.FEATURES, "feature_version": scalper.FEATURE_VERSION,
             "trained_at": started.isoformat(timespec="seconds"), "validation_file": path.name,
             "data_key": f"{source}:{days}:{','.join(sorted(symbols))}", "model": None, "threshold": None}
    if name:  # refit on every row, with the break-even probability of the whole dataset
        saved["model"] = validate.fit(name, data)
        saved["threshold"] = validate.breakeven_probability(data)
        saved["expected"] = report["results"][name]
    with (config.ML_DIR / MODEL_FILE).open("wb") as handle:
        pickle.dump(saved, handle)
    report["file"] = path.name
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train and validate the ML setup filter.")
    parser.add_argument("--source", choices=("auto", "mt5", "dukascopy"), default="auto")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", stream=sys.stdout)
    import settings_store
    settings_store.load_saved()
    symbols = list(config.SYMBOLS_DEMO)
    source = args.source
    if source == "auto":
        source = "dukascopy" if all(dukascopy.available(s) for s in symbols) else "mt5"
    try:
        if source == "mt5":
            if not data_engine.initialize_mt5():
                print("MT5 is not available")
                return 1
            offered = [item["name"] for item in data_engine.broker_symbols() or []]
            symbols, _, _ = data_engine.resolve_symbols(symbols, offered)
        days = args.days or (1250 if source == "dukascopy" else 330)
        report = train(symbols, days, source, progress=print)
    except RuntimeError as exc:
        print(f"TRAINING NOT RUN: {exc}")
        return 1
    finally:
        data_engine.shutdown_mt5()
    print(json.dumps({k: report.get(k) for k in ("results", "quality", "oos_setups", "base_win_rate", "trials")},
                     indent=2, default=str))
    for fold in report.get("folds", []):
        print("  fold", fold)
    print(f"RESULT: {report.get('reason')} | saved {report.get('file')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
