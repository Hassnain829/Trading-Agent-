"""
Professional validation of a trade list (results in R), used by every backtest report.

* rolling stability: results month by month (is the edge spread out, or one lucky month?)
* Monte Carlo: resample the trade sequence many times to see the range of drawdowns and
  the odds of losing or of a deep drawdown at a given risk per trade
* significance: t-statistic, Van Tharp's SQN, and the Deflated Sharpe Ratio (Bailey &
  Lopez de Prado), which asks how likely the edge is real after accounting for how many
  strategy variants were tried (the more ideas tested, the higher the bar)

References: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014); Van Tharp, SQN.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional

import numpy as np

import config

_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649


# -----------------------------------------------------------------------------
# Rolling stability
# -----------------------------------------------------------------------------
def monthly(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Expectancy per calendar month (entry time) and how many months were profitable."""
    months: Dict[str, List[float]] = {}
    for trade in trades:
        months.setdefault(str(trade["entry_time"])[:7], []).append(float(trade["r"]))
    rows = [{"month": m, "trades": len(rs), "net_r": round(sum(rs), 2), "expectancy_r": round(sum(rs) / len(rs), 3)}
            for m, rs in sorted(months.items())]
    active = [row for row in rows if row["trades"] >= 3]  # a month needs a few trades to count
    positive = sum(1 for row in active if row["net_r"] > 0)
    return {
        "months": rows,
        "months_counted": len(active),
        "positive_share": round(positive / len(active), 3) if active else None,
        "worst_month": min(active, key=lambda row: row["net_r"]) if active else None,
        "best_month": max(active, key=lambda row: row["net_r"]) if active else None,
    }


# -----------------------------------------------------------------------------
# Walk-forward
# -----------------------------------------------------------------------------
def _month_range(first: str, last: str) -> List[str]:
    year, month = int(first[:4]), int(first[5:7])
    months = []
    while f"{year:04d}-{month:02d}" <= last:
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _brief(rs: List[float]) -> Dict[str, Any]:
    if not rs:
        return {"trades": 0, "expectancy_r": None, "net_r": 0.0, "win_rate": None}
    return {"trades": len(rs), "expectancy_r": round(sum(rs) / len(rs), 3), "net_r": round(sum(rs), 2),
            "win_rate": round(sum(1 for r in rs if r > 0) / len(rs) * 100.0, 1)}


def window_plan(trades: List[Dict[str, Any]]) -> tuple:
    """(train months, test months): 12 -> 3 on multi-year data, 3 -> 1 on a few months."""
    if not trades:
        return 3, 1
    months = len(_month_range(str(trades[0]["entry_time"])[:7], str(trades[-1]["entry_time"])[:7]))
    return (12, 3) if months >= 30 else (3, 1)


def walk_forward(variants: Dict[str, List[Dict[str, Any]]], train_months: int, test_months: int,
                 baseline: Optional[str] = None, min_trades: int = 20) -> Dict[str, Any]:
    """
    Rolling walk-forward: in each window pick the variant with the best expectancy over the previous
    ``train_months`` (at least ``min_trades`` trades, else the baseline), then record only its results in
    the next ``test_months``. The stitched test results are an honest out-of-sample estimate of
    "choosing by past performance". With one variant it is a rolling out-of-sample stability check.
    """
    names = list(variants)
    baseline = baseline if baseline in variants else names[0]
    stamps = [str(t["entry_time"])[:7] for trades in variants.values() for t in trades]
    if not stamps:
        return {"train_months": train_months, "test_months": test_months, "windows": [], "windows_count": 0,
                "oos": _brief([]), "positive_windows_share": None, "picks": {}}
    months = _month_range(min(stamps), max(stamps))
    by_month = {name: {} for name in names}
    for name, trades in variants.items():
        for t in trades:
            by_month[name].setdefault(str(t["entry_time"])[:7], []).append(float(t["r"]))

    def rs(name: str, window: List[str]) -> List[float]:
        return [r for m in window for r in by_month[name].get(m, [])]

    windows, stitched, picks = [], [], {}
    for start in range(train_months, len(months), test_months):
        train, test = months[start - train_months:start], months[start:start + test_months]
        scores = {name: _brief(rs(name, train))["expectancy_r"] for name in names
                  if len(rs(name, train)) >= min_trades}
        pick = max(scores, key=scores.get) if scores else baseline
        result = rs(pick, test)
        stitched.extend(result)
        picks[pick] = picks.get(pick, 0) + 1
        windows.append({"test": f"{test[0]}..{test[-1]}", "pick": pick,
                        "train_expectancy_r": scores.get(pick), **_brief(result)})
    counted = [w for w in windows if w["trades"]]
    return {
        "train_months": train_months, "test_months": test_months, "windows": windows,
        "windows_count": len(windows), "oos": _brief(stitched), "picks": picks,
        "positive_windows_share": round(sum(1 for w in counted if w["net_r"] > 0) / len(counted), 3) if counted else None,
        "baseline_same_windows": _brief([r for w in windows for r in rs(baseline, [m for m in months
                                         if w["test"][:7] <= m <= w["test"][-7:]])]),
    }


# -----------------------------------------------------------------------------
# Monte Carlo
# -----------------------------------------------------------------------------
def monte_carlo(rs: List[float], risk_percent: float, runs: int = 5000, seed: int = 7,
                ruin_drawdown: float = 0.5) -> Dict[str, Any]:
    """Resample the trades (with replacement) and compound them at ``risk_percent`` per trade."""
    values = np.asarray(rs, dtype=float)
    if len(values) < 10:
        return {"runs": 0, "note": "fewer than 10 trades"}
    rng = np.random.default_rng(seed)
    sims = rng.choice(values, size=(runs, len(values)), replace=True)
    equity = np.cumprod(1.0 + risk_percent / 100.0 * sims, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((runs, 1)), equity], axis=1), axis=1)[:, 1:]
    drawdown = ((peak - equity) / peak).max(axis=1)
    final = equity[:, -1]
    limit = config.MAX_TOTAL_DRAWDOWN_PERCENT / 100.0 if config.MAX_TOTAL_DRAWDOWN_PERCENT > 0 else None
    return {
        "runs": runs, "trades": len(values), "risk_percent": risk_percent,
        "median_growth": round(float(np.median(final)), 3),
        "p_loss": round(float(np.mean(final < 1.0)), 4),
        "median_max_drawdown_pct": round(float(np.median(drawdown)) * 100, 2),
        "p95_max_drawdown_pct": round(float(np.percentile(drawdown, 95)) * 100, 2),
        "p_hit_kill_switch": round(float(np.mean(drawdown >= limit)), 4) if limit else None,
        "p_ruin": round(float(np.mean(drawdown >= ruin_drawdown)), 4),
    }


# -----------------------------------------------------------------------------
# Significance
# -----------------------------------------------------------------------------
def _moments(values: np.ndarray) -> tuple:
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    if sd == 0:
        return mean, sd, 0.0, 3.0
    z = (values - mean) / sd
    return mean, sd, float(np.mean(z ** 3)), float(np.mean(z ** 4))


def deflated_sharpe(values: np.ndarray, trials: int) -> Dict[str, Any]:
    """
    Probabilistic and Deflated Sharpe Ratio (per-trade Sharpe = mean R / stdev R).
    PSR = P(true Sharpe > 0); DSR = P(true Sharpe > the best Sharpe expected by luck among ``trials`` tries).
    """
    n = len(values)
    mean, sd, skew, kurt = _moments(values)
    if sd == 0 or n < 3:
        return {"sharpe_per_trade": None, "psr": None, "dsr": None}
    sr = mean / sd
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    psr = _NORMAL.cdf(sr * math.sqrt(n - 1) / denom)
    sr_std = denom / math.sqrt(n - 1)  # standard error of the Sharpe estimate
    trials = max(1, int(trials))
    if trials > 1:
        expected_max = sr_std * ((1 - _EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / trials)
                                 + _EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (trials * math.e)))
    else:
        expected_max = 0.0
    dsr = _NORMAL.cdf((sr - expected_max) * math.sqrt(n - 1) / denom)
    return {"sharpe_per_trade": round(sr, 4), "psr": round(psr, 4), "dsr": round(dsr, 4),
            "sharpe_hurdle": round(expected_max, 4), "skew": round(skew, 3), "kurtosis": round(kurt, 3)}


def significance(rs: List[float], trials: Optional[int] = None) -> Dict[str, Any]:
    values = np.asarray(rs, dtype=float)
    n = len(values)
    trials = int(trials or config.BACKTEST_TRIALS)
    if n < 10:
        return {"trades": n, "note": "fewer than 10 trades"}
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    t = mean / (sd / math.sqrt(n)) if sd else 0.0
    hurdle = _NORMAL.inv_cdf(1 - 0.05 / max(1, trials))  # Bonferroni-style bar for 95% confidence
    return {
        "trades": n, "trials": trials,
        "t_stat": round(t, 2), "p_value": round(1 - _NORMAL.cdf(t), 4),
        "t_hurdle_for_trials": round(hurdle, 2), "passes_multiple_testing": t > hurdle,
        "sqn": round(math.sqrt(min(n, 100)) * mean / sd, 2) if sd else None,
        **deflated_sharpe(values, trials),
    }


def sqn_label(sqn: Optional[float]) -> str:
    if sqn is None:
        return "n/a"
    for bound, label in ((1.6, "poor"), (2.0, "below average"), (2.5, "average"), (3.0, "good"), (5.0, "excellent")):
        if sqn < bound:
            return label
    return "superb (check for errors)"


def report(trades: List[Dict[str, Any]], risk_percent: float, trials: Optional[int] = None) -> Dict[str, Any]:
    rs = [float(t["r"]) for t in trades]
    sig = significance(rs, trials)
    return {"monthly": monthly(trades), "monte_carlo": monte_carlo(rs, risk_percent),
            "significance": {**sig, "sqn_label": sqn_label(sig.get("sqn"))}}


def verdict(summary: Dict[str, Any], out_of_sample: Dict[str, Any], validation: Dict[str, Any],
            stress: Optional[Dict[str, Any]] = None) -> str:
    """A plain-language verdict that weighs expectancy, stability, significance and cost stress."""
    if summary.get("trades", 0) < 30:
        return "Too few trades to judge; test more days or symbols."
    if summary["expectancy_r"] <= 0:
        return "No edge: the rules lost money after spreads. Do not trade them live as they are."
    sig = validation.get("significance", {})
    stable = (validation.get("monthly", {}).get("positive_share") or 0) >= 0.6
    oos_ok = (out_of_sample.get("expectancy_r") or 0) > 0
    real = (sig.get("dsr") or 0) >= 0.95
    robust = stress is None or (stress.get("expectancy_r") or 0) > 0
    if real and stable and oos_ok and robust:
        return (f"Edge looks real: DSR {sig['dsr']:.2f} after {sig['trials']} trials, "
                f"{validation['monthly']['positive_share']:.0%} of months profitable, positive on the recent 30%"
                f"{' and with +1 pip of extra cost' if stress else ''}. Forward-test on demo before live.")
    parts = []
    if not real:
        parts.append(f"not statistically proven (DSR {sig.get('dsr')} < 0.95 after {sig.get('trials')} trials)")
    if not stable:
        parts.append("profits concentrated in few months")
    if not oos_ok:
        parts.append("negative on the most recent 30%")
    if not robust:
        parts.append(f"loses with +1 pip of extra cost ({stress.get('expectancy_r')}R per trade)")
    return "Positive but " + "; ".join(parts) + ". Keep testing on demo."
