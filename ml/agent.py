"""
The learning agent: a contextual-bandit reinforcement learner, one per rules strategy (SCALP, INTRADAY).

    state    the market when a setup appears (trend, momentum, volatility, spread, hour...), the setup's
             geometry and the AI's answer (asked? confirmed? score, penalty points)
    action   TAKE or SKIP
    reward   the trade's result in R after the spread: +1.5 / +2 for a win at the target, -1 for a stop,
             the move at the time stop otherwise. Rewards come from demo/live trades and from shadow
             trades (skipped setups followed on real prices), so every setup is paid, taken or not.

The model is a Bayesian linear regression of the reward on the standardised state, fitted on every reward
so far with recency weights (half-life AGENT_HALF_LIFE_DAYS), so it follows the current market and needs
no price history. Decisions use Thompson sampling: draw a plausible model from the posterior and take the
setup when that model expects a positive reward. Early on the draws vary a lot (exploration, on demo); as
rewards accumulate they settle on what has been paying. Until a strategy has learned enough (learned():
AGENT_MIN_REWARDS rewards, a third of them from real setups) the AI's confirm/veto decides and the agent only
learns. Whether decisions become real orders is the dashboard's Shadow/Demo switch (config.TRADING_MODE).

Experience (one line per reward) is kept in data/agent/experience.jsonl; sync() adds new rewards from
memory.json (closed trades) and the shadow files.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

import config

logger = logging.getLogger("hedgefund.agent")

STRATEGIES = ("SCALP", "INTRADAY")
EXPERIENCE_VERSION = 2  # 2: samples a real order could not have taken (spread > limit) are left out

# Market state (from scalper.features: side-signed, positive = in the trade's favour).
MARKET_KEYS = (
    "d1_ema200_dist_atr", "d1_adx", "d1_ema20_vs_50_atr", "h1_ema200_dist_atr", "h1_rsi", "h1_change_24h_pct",
    "m15_close_vs_ema50_pct", "m5_rsi", "m5_rsi_extreme", "m5_atr_ratio", "m5_rel_volume", "m5_room_atr",
    "m5_close_location", "spread_atr",
)
COMMON_KEYS = MARKET_KEYS + ("hour_sin", "hour_cos", "is_buy", "sl_atr", "ai_asked", "ai_confirmed", "ai_score",
                             "penalty_pts", "explore")
EXTRA_KEYS = {"SCALP": ("room_r", "reversion"), "INTRADAY": ("level_asia", "bars_since_break")}

PRIOR_PRECISION = 25.0     # shrinkage of each feature's effect (in pseudo-rewards of "no effect")
INTERCEPT_PRECISION = 0.5  # the average reward is learned almost freely
EXPLORATION_SCALE = 0.5    # Thompson draws use half the posterior spread (every setup is paid anyway)
REWARD_CLIP = (-2.0, 5.0)  # one slipped stop or a gap must not dominate the model
CLIP_Z = 4.0

_lock = threading.RLock()
_cache: Dict[str, Any] = {}
_rng = np.random.default_rng()


def keys(strategy: str) -> tuple:
    return COMMON_KEYS + EXTRA_KEYS.get(strategy, ())


def _path():
    return config.AGENT_DIR / "experience.jsonl"


def _utc(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
def make_context(strategy: str, features: Optional[Dict[str, Any]], side: str, setup: Optional[Dict[str, Any]] = None,
                 decision: Optional[Dict[str, Any]] = None, explore: bool = False) -> Dict[str, Any]:
    """Everything the agent sees about one setup (raw values; vector() turns them into numbers)."""
    features = features or {}
    setup = setup or {}
    ai: Dict[str, Any] = {"asked": False}
    if decision is not None and not decision.get("error"):
        penalties = sum(float(r.get("points") or 0) for r in decision.get("applied_rules") or [])
        ai = {"asked": True, "confirmed": "AI-VETO" not in (decision.get("blocked_by") or []),
              "confidence": decision.get("base_confidence"), "penalty_pts": penalties}
    return {
        "v": EXPERIENCE_VERSION, "strategy": strategy, "side": side, "explore": bool(explore),
        "market": {k: _num(features.get(k)) for k in MARKET_KEYS},
        "server_hour": _num(features.get("server_hour")),
        "sl_atr": _num(setup.get("sl_atr")), "room_r": _num(setup.get("room_r")),
        "level_name": setup.get("level_name"), "bars_since_break": _num(setup.get("bars_since_break")),
        "kind": setup.get("kind"),
        "ai": ai,
    }


def vector(strategy: str, ctx: Dict[str, Any]) -> np.ndarray:
    """The raw state as numbers in keys(strategy) order (NaN = unknown; standardised later)."""
    market = ctx.get("market") or {}
    ai = ctx.get("ai") or {}
    hour = _num(ctx.get("server_hour"))
    values: Dict[str, Optional[float]] = {k: _num(market.get(k)) for k in MARKET_KEYS}
    values.update({
        "hour_sin": math.sin(2 * math.pi * hour / 24) if hour is not None else None,
        "hour_cos": math.cos(2 * math.pi * hour / 24) if hour is not None else None,
        "is_buy": 1.0 if ctx.get("side") == "BUY" else 0.0,
        "sl_atr": _num(ctx.get("sl_atr")),
        "ai_asked": 1.0 if ai.get("asked") else 0.0,
        "ai_confirmed": 1.0 if ai.get("asked") and ai.get("confirmed") else 0.0,
        "ai_score": ((_num(ai.get("confidence")) or 50.0) - 50.0) / 50.0 if ai.get("asked") else 0.0,
        "penalty_pts": (_num(ai.get("penalty_pts")) or 0.0) / 10.0 if ai.get("asked") else 0.0,
        "explore": 1.0 if ctx.get("explore") else 0.0,
        "room_r": _num(ctx.get("room_r")),
        "level_asia": 1.0 if str(ctx.get("level_name") or "").startswith("ASIA") else 0.0,
        "bars_since_break": _num(ctx.get("bars_since_break")),
        "reversion": 1.0 if ctx.get("kind") == "REVERSION" else 0.0,
    })
    return np.array([np.nan if values.get(k) is None else float(values[k]) for k in keys(strategy)], dtype=float)


# -----------------------------------------------------------------------------
# Experience (rewards)
# -----------------------------------------------------------------------------
def load_experience(strategy: Optional[str] = None) -> List[Dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and (strategy is None or row.get("strategy") == strategy):
                rows.append(row)
    return rows


def _trade_reward(record: Dict[str, Any]) -> Optional[float]:
    """R of a closed trade from its prices (entry, first stop, exit), else from P&L and the risk taken."""
    entry, stop, exit_ = (_num(record.get(k)) for k in ("entry_price", "stop_loss", "exit_price"))
    if entry is not None and stop is not None and exit_ is not None and abs(entry - stop) > 0:
        moved = (exit_ - entry) if record.get("side") == "BUY" else (entry - exit_)
        return moved / abs(entry - stop)
    import calibration
    return calibration._r_multiple(record)


def _shadow_context(shadow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Agent context of a shadow trade (older shadows without one are rebuilt from their stored fields)."""
    agent = shadow.get("agent") or {}
    if agent.get("context"):
        return agent["context"]
    strategy = str(shadow.get("strategy") or "").upper()
    if strategy not in STRATEGIES or not shadow.get("features"):
        return None
    blocked = shadow.get("blocked_by") or []
    decision = None
    if shadow.get("base_confidence") is not None:
        decision = {"blocked_by": blocked, "base_confidence": shadow.get("base_confidence"), "applied_rules": []}
    return make_context(strategy, shadow.get("features"), shadow.get("side"), None, decision)


def _action(agent: Dict[str, Any], fallback: str) -> str:
    return str(agent.get("action") or fallback)


def _rebuild_if_outdated() -> None:
    """Experience written by an older version is rebuilt once from the trade and shadow files (old file kept)."""
    marker = config.AGENT_DIR / "experience.version"
    current = marker.read_text(encoding="utf-8").strip() if marker.exists() else "1"
    if current == str(EXPERIENCE_VERSION):
        return
    config.AGENT_DIR.mkdir(parents=True, exist_ok=True)
    if _path().exists():
        _path().replace(_path().with_name(f"experience.v{current}.jsonl"))
        logger.warning("[AGENT] Experience rebuilt (version %s -> %s): rewards a real order could not have "
                       "taken (spread too large for the stop) are left out", current, EXPERIENCE_VERSION)
    marker.write_text(str(EXPERIENCE_VERSION), encoding="utf-8")
    _cache.clear()


def sync() -> int:
    """Add every new reward (closed trades, resolved shadow and exploration trades) to the experience log."""
    import memory_store
    import shadow_store
    with _lock:
        _rebuild_if_outdated()
        known = {row.get("id") for row in load_experience()}
        new: List[Dict[str, Any]] = []
        for record in memory_store.load_trade_memory():
            agent = record.get("agent") or {}
            ticket = record.get("ticket")
            if not agent.get("context") or not ticket or record.get("status") != "CLOSED":
                continue
            row_id = f"trade:{ticket}"
            reward = _trade_reward(record)
            if row_id in known or reward is None:
                continue
            new.append({"id": row_id, "source": "trade", "strategy": agent["context"].get("strategy"),
                        "symbol": record.get("symbol"), "side": record.get("side"),
                        "opened_at": record.get("timestamp"),
                        "resolved_at": record.get("exit_time_utc") or record.get("reconciled_at"),
                        "reward": reward, "outcome": record.get("outcome"), "exit": record.get("exit_reason"),
                        "action": "taken", "decided_by": agent.get("decided_by"),
                        "account_mode": record.get("account_mode"), "context": agent["context"]})
        for path, source in ((config.SHADOW_FILE, "shadow"), (config.EXPLORE_FILE, "explore")):
            for shadow in shadow_store.load_shadows(path):
                if shadow.get("status") not in ("WIN", "LOSS", "TIMEOUT") or shadow.get("r_multiple") is None:
                    continue
                row_id = f"{source}:{shadow.get('id')}"
                if row_id in known:
                    continue
                if not shadow_store.tradeable_cost(shadow.get("spread_price"), shadow.get("entry_price"),
                                                   shadow.get("stop_loss")):
                    continue  # the spread alone was bigger than a real order allows: not a valid reward
                ctx = _shadow_context(shadow)
                if ctx is None:
                    continue
                agent = shadow.get("agent") or {}
                fallback = "explore" if source == "explore" else "skipped"
                new.append({"id": row_id, "source": source, "strategy": ctx.get("strategy"),
                            "symbol": shadow.get("symbol"), "side": shadow.get("side"),
                            "opened_at": shadow.get("created_at"), "resolved_at": shadow.get("resolved_at"),
                            "reward": float(shadow["r_multiple"]), "outcome": shadow.get("status"),
                            "exit": shadow.get("status"), "action": _action(agent, fallback),
                            "decided_by": agent.get("decided_by") or ("ai" if source == "shadow" else "explore"),
                            "blocked_by": shadow.get("blocked_by") or [],
                            "account_mode": shadow.get("account_mode"), "context": ctx})
        if not new:
            return 0
        config.AGENT_DIR.mkdir(parents=True, exist_ok=True)
        with _path().open("a", encoding="utf-8") as handle:
            for row in new:
                handle.write(json.dumps(row, default=str) + "\n")
        _cache.clear()
    for row in new:
        logger.info("[AGENT] %s reward %+.2fR: %s %s %s (%s)", str(row["strategy"]).lower(), row["reward"],
                    row["side"], row["symbol"], row["action"], row["source"])
    return len(new)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def _weights(rows: List[Dict[str, Any]], now: datetime) -> np.ndarray:
    half_life = max(float(config.AGENT_HALF_LIFE_DAYS), 0.1)
    out = []
    for row in rows:
        when = _utc(row.get("resolved_at")) or _utc(row.get("opened_at")) or now
        age_days = max(0.0, (now - when).total_seconds() / 86400.0)
        out.append(0.5 ** (age_days / half_life))
    return np.array(out, dtype=float)


def fit(strategy: str, rows: Optional[List[Dict[str, Any]]] = None,
        now: Optional[datetime] = None) -> Dict[str, Any]:
    """Posterior of reward = w . [1, standardised state] from every reward of the strategy."""
    now = now or datetime.now(timezone.utc)
    rows = load_experience(strategy) if rows is None else [r for r in rows if r.get("strategy") == strategy]
    names = keys(strategy)
    model: Dict[str, Any] = {"strategy": strategy, "keys": list(names), "rewards": len(rows), "fitted_at": now.isoformat(),
                             "real_rewards": sum(1 for r in rows if r.get("action") != "explore")}
    if not rows:
        return {**model, "effective": 0.0, "mean_reward": None}
    X = np.vstack([vector(strategy, row.get("context") or {}) for row in rows])
    r = np.clip(np.array([float(row["reward"]) for row in rows]), *REWARD_CLIP)
    w = _weights(rows, now)
    sw = float(w.sum())
    mean = np.zeros(len(names))
    std = np.ones(len(names))
    for j in range(len(names)):
        col, ok = X[:, j], np.isfinite(X[:, j])
        if ok.any():
            m = float(np.average(col[ok], weights=w[ok]))
            s = float(math.sqrt(np.average((col[ok] - m) ** 2, weights=w[ok])))
            mean[j], std[j] = m, s if s > 1e-9 else 1.0
    Z = np.clip(np.nan_to_num((X - mean) / std, nan=0.0), -CLIP_Z, CLIP_Z)
    Z = np.hstack([np.ones((len(rows), 1)), Z])
    mean_r = float(np.average(r, weights=w))
    sigma2 = max(float(np.average((r - mean_r) ** 2, weights=w)), 0.25)
    prior = np.full(Z.shape[1], PRIOR_PRECISION)
    prior[0] = INTERCEPT_PRECISION
    A = np.diag(prior) + (Z * w[:, None]).T @ Z
    A_inv = np.linalg.inv(A)
    mu = A_inv @ ((Z * w[:, None]).T @ r)
    return {**model, "effective": round(sw, 2), "mean_reward": round(mean_r, 4), "sigma2": sigma2,
            "x_mean": mean.tolist(), "x_std": std.tolist(), "mu": mu.tolist(), "cov": (sigma2 * A_inv).tolist()}


def model(strategy: str) -> Dict[str, Any]:
    """The fitted model, cached until the experience log changes (or an hour passes: weights fade)."""
    path = _path()
    stamp = (path.stat().st_mtime, path.stat().st_size) if path.exists() else None
    hour = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    key = (strategy, stamp, hour)
    with _lock:
        cached = _cache.get(strategy)
        if cached and cached[0] == key:
            return cached[1]
        fitted = fit(strategy)
        _cache[strategy] = (key, fitted)
        return fitted


def _standardised(m: Dict[str, Any], ctx: Dict[str, Any]) -> np.ndarray:
    x = vector(m["strategy"], ctx)
    z = np.clip(np.nan_to_num((x - np.array(m["x_mean"])) / np.array(m["x_std"]), nan=0.0), -CLIP_Z, CLIP_Z)
    return np.concatenate([[1.0], z])


def predict(strategy: str, ctx: Dict[str, Any], m: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, float]]:
    """Expected reward (R) of the setup, its uncertainty and P(reward > 0); None without any reward yet."""
    m = m or model(strategy)
    if not m.get("mu"):
        return None
    z = _standardised(m, ctx)
    mu, cov = np.array(m["mu"]), np.array(m["cov"])
    expected = float(z @ mu)
    spread = float(math.sqrt(max(z @ cov @ z, 1e-12)))
    p_positive = 0.5 * (1.0 + math.erf(expected / (spread * math.sqrt(2.0))))
    return {"expected_r": round(expected, 3), "uncertainty_r": round(spread, 3), "p_positive": round(p_positive, 3)}


def min_real_rewards() -> int:
    """Rewards from real setups (not exploration) needed before the agent decides: a third of the minimum."""
    return int(math.ceil(config.AGENT_MIN_REWARDS / 3))


def learned(m: Dict[str, Any]) -> bool:
    """Enough rewards to decide: AGENT_MIN_REWARDS in all, a third of them from real setups the AI reviewed."""
    return (int(m.get("rewards") or 0) >= config.AGENT_MIN_REWARDS
            and int(m.get("real_rewards") or 0) >= min_real_rewards() and bool(m.get("mu")))


def decide(strategy: str, ctx: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """
    TAKE or SKIP by Thompson sampling. {"take": True/False, or None while warming up / switched off (the AI
    decides), "phase", "expected_r", "p_positive", "sampled_r", "rewards", "note"}.
    """
    m = model(strategy)
    rewards = int(m.get("rewards") or 0)
    view = predict(strategy, ctx, m) or {}
    out: Dict[str, Any] = {"strategy": strategy, "rewards": rewards, "take": None, **view}
    if not config.AGENT_ENABLED:
        return {**out, "phase": "off", "note": "agent switched off: the AI decides"}
    if not learned(m) or not view:
        return {**out, "phase": "warmup",
                "note": f"learning ({rewards}/{config.AGENT_MIN_REWARDS} rewards, {int(m.get('real_rewards') or 0)}/"
                        f"{min_real_rewards()} from real setups)"}
    rng = rng or _rng
    z = _standardised(m, ctx)
    draw = rng.multivariate_normal(np.array(m["mu"]), np.array(m["cov"]) * EXPLORATION_SCALE ** 2,
                                   check_valid="ignore", method="cholesky")
    sampled = float(z @ draw)
    take = sampled > 0
    note = (f"{'TAKE' if take else 'SKIP'}: expects {view['expected_r']:+.2f}R "
            f"(P(profit) {view['p_positive']:.0%}, {rewards} rewards learned)")
    return {**out, "phase": "learning", "take": take, "sampled_r": round(sampled, 3), "note": note}


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def _effects(m: Dict[str, Any], top: int = 5) -> List[Dict[str, Any]]:
    """The state features that move the expected reward most (R per 1 standard deviation)."""
    if not m.get("mu"):
        return []
    effects = [{"feature": k, "r_per_sd": round(float(v), 3)} for k, v in zip(m["keys"], m["mu"][1:])]
    effects.sort(key=lambda e: -abs(e["r_per_sd"]))
    return effects[:top]


def status(strategy: str) -> Dict[str, Any]:
    m = model(strategy)
    rewards = int(m.get("rewards") or 0)
    phase = "off" if not config.AGENT_ENABLED else "learning" if learned(m) else "warmup"
    rows = load_experience(strategy)
    return {"strategy": strategy, "phase": phase, "rewards": rewards, "min_rewards": config.AGENT_MIN_REWARDS,
            "real_rewards": int(m.get("real_rewards") or 0), "min_real_rewards": min_real_rewards(),
            "shadow_only": config.TRADING_MODE == "SHADOW",
            "effective_rewards": m.get("effective"), "mean_reward": m.get("mean_reward"),
            "last_reward_at": max((str(r.get("resolved_at") or "") for r in rows), default=None) or None,
            "effects": _effects(m)}


def _group(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    rewards = [float(r["reward"]) for r in rows]
    wins = sum(1 for x in rewards if x > 0)
    return {"n": len(rows), "wins": wins, "win_rate": round(wins / len(rows), 3) if rows else None,
            "total_r": round(sum(rewards), 2), "avg_r": round(sum(rewards) / len(rows), 3) if rows else None}


def scorecard(strategy: str, days: Optional[int] = None, now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Rewards and penalties of one strategy over the last ``days`` (None = all):
      taken    real demo/live trades                    picks    everything the decision-maker chose to take
      skipped  setups it passed on (shadow-followed)    all      every real setup = "take every setup"
      explore  virtual near-miss setups                 ai       setups the AI confirmed vs vetoed
    """
    now = now or datetime.now(timezone.utc)
    rows = load_experience(strategy)
    if days:
        rows = [r for r in rows if (w := _utc(r.get("resolved_at")) or _utc(r.get("opened_at")))
                and (now - w).total_seconds() <= days * 86400]
    real = [r for r in rows if r.get("action") != "explore"]
    picks = [r for r in real if r.get("action") in ("taken", "blocked")]
    asked = [r for r in real if ((r.get("context") or {}).get("ai") or {}).get("asked")]
    groups = {
        "taken": _group(r for r in real if r.get("action") == "taken"),
        "blocked": _group(r for r in real if r.get("action") == "blocked"),
        "skipped": _group(r for r in real if r.get("action") == "skipped"),
        "picks": _group(picks),
        "all": _group(real),
        "explore": _group(r for r in rows if r.get("action") == "explore"),
        "ai_confirmed": _group(r for r in asked if r["context"]["ai"].get("confirmed")),
        "ai_vetoed": _group(r for r in asked if not r["context"]["ai"].get("confirmed")),
        "by_agent": _group(r for r in picks if r.get("decided_by") == "agent"),
    }
    edge = None
    if groups["picks"]["n"] and groups["all"]["n"]:
        edge = round(groups["picks"]["avg_r"] - groups["all"]["avg_r"], 3)
    daily: Dict[str, Dict[str, float]] = {}
    for r in sorted(rows, key=lambda x: str(x.get("resolved_at") or x.get("opened_at") or "")):
        when = _utc(r.get("resolved_at")) or _utc(r.get("opened_at"))
        if when is None:
            continue
        day = daily.setdefault(when.date().isoformat(), {"taken": 0.0, "picks": 0.0, "all": 0.0, "explore": 0.0})
        action, reward = r.get("action"), float(r["reward"])
        if action == "explore":
            day["explore"] += reward
            continue
        day["all"] += reward
        if action in ("taken", "blocked"):
            day["picks"] += reward
        if action == "taken":
            day["taken"] += reward
    curve, totals = [], {"taken": 0.0, "picks": 0.0, "all": 0.0, "explore": 0.0}
    for day in sorted(daily):
        for k in totals:
            totals[k] += daily[day][k]
        curve.append({"day": day, **{k: round(v, 2) for k, v in totals.items()}})
    return {"strategy": strategy, "days": days, "groups": groups, "edge_r": edge, "curve": curve,
            "recent": [{k: r.get(k) for k in ("symbol", "side", "action", "reward", "outcome", "resolved_at",
                                               "source", "decided_by")}
                       for r in sorted(rows, key=lambda x: str(x.get("resolved_at") or ""))[-30:]][::-1]}
