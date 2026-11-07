"""Propose -> verify -> promote.

Claude's proposal (parameters plus a filter set) only replaces the current
version if it beats it on a recent holdout window that neither the analyzer
nor Claude saw, after trading costs. See
docs/superpowers/specs/2026-11-07-walk-forward-promotion-design.md.
"""
import copy
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.data.database import get_connection, load_ohlcv, load_latest_parameters, save_parameters
from src.learning.filter_rules import (
    load_active_filters, parse_analyzer_filters, parse_claude_filter_rules,
    select_filters, init_filter_rules_table,
)

DEFAULTS = {
    "holdout_days": 90,
    "min_holdout_trades": 20,
    "min_trade_ratio": 0.5,
    "min_expectancy_gain": 0.0005,
    "max_drawdown_worsening": 0.05,
}


def promotion_config(config: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update((config.get("learning", {}) or {}).get("promotion", {}) or {})
    return cfg


def holdout_start(config: dict, now: datetime = None) -> pd.Timestamp:
    """Start of the holdout window, naive UTC like the OHLCV index."""
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    return pd.Timestamp(now - timedelta(days=promotion_config(config)["holdout_days"]))


# ── Evaluation ──────────────────────────────────────────────────────────

def evaluate_bundle(strategy: str, params: dict, filters: list, dfs: dict,
                    timeframe: str, trade_start, costs: dict,
                    initial_capital: float) -> dict:
    """Backtest one bundle on the holdout window of every pair.

    Indicators warm up on the full history; entries start at ``trade_start``.
    Nothing is written to the database.
    """
    pnls, drawdowns, per_pair = [], [], {}
    for pair, df in dfs.items():
        engine = BacktestEngine(strategy, copy.deepcopy(params), initial_capital,
                                costs=costs, filters=filters)
        res = engine.run(df.copy(), pair, timeframe, trade_start=trade_start)
        trades = [t["pnl_pct"] for t in engine.trades]
        pnls.extend(trades)
        per_pair[pair] = len(trades)
        if trades:
            drawdowns.append(float(res.get("max_drawdown", 0.0)))
    return {
        "trades": len(pnls),
        "expectancy": float(np.mean(pnls)) if pnls else 0.0,
        "win_rate": float(np.mean([p > 0 for p in pnls])) if pnls else 0.0,
        "max_drawdown": max(drawdowns) if drawdowns else 0.0,
        "per_pair": per_pair,
    }


def decide(incumbent: dict, candidate: dict, cfg: dict, changed: bool) -> tuple[bool, str]:
    """Apply the promotion rule. Returns (promote, reason)."""
    if not changed:
        return False, "no change proposed"
    if candidate["trades"] < cfg["min_holdout_trades"]:
        return False, (f"only {candidate['trades']} holdout trades "
                       f"(need {cfg['min_holdout_trades']})")
    if incumbent["trades"] >= cfg["min_holdout_trades"]:
        if candidate["trades"] < cfg["min_trade_ratio"] * incumbent["trades"]:
            return False, (f"keeps only {candidate['trades'] / incumbent['trades']:.0%} of the "
                           f"current version's trades (need {cfg['min_trade_ratio']:.0%})")
        needed = incumbent["expectancy"] + cfg["min_expectancy_gain"]
        if candidate["expectancy"] < needed:
            return False, (f"expectancy {candidate['expectancy']:+.3%} vs current "
                           f"{incumbent['expectancy']:+.3%} (need ≥ {needed:+.3%})")
        if candidate["max_drawdown"] > incumbent["max_drawdown"] + cfg["max_drawdown_worsening"]:
            return False, (f"max drawdown {candidate['max_drawdown']:.1%} vs current "
                           f"{incumbent['max_drawdown']:.1%}")
        return True, (f"expectancy {candidate['expectancy']:+.3%} vs current "
                      f"{incumbent['expectancy']:+.3%} on {candidate['trades']} holdout trades")
    if candidate["expectancy"] <= 0:
        return False, (f"current version has too few holdout trades and the candidate's "
                       f"expectancy {candidate['expectancy']:+.3%} is not positive")
    return True, (f"expectancy {candidate['expectancy']:+.3%} on {candidate['trades']} "
                  f"holdout trades (current version had {incumbent['trades']})")


# ── Candidate construction ──────────────────────────────────────────────

def candidate_filters(review: dict, findings: dict, limit) -> list:
    """Claude's and the analyzer's proposed rules, capped like live filters."""
    rules = []
    for rule in parse_claude_filter_rules(review.get("filter_rules") or []):
        rules.append(dict(rule, rule_type=rule.get("type", "skip"), source="claude_review"))
    for rule in parse_analyzer_filters((findings or {}).get("suggested_filters") or []):
        rules.append(dict(rule, rule_type=rule.get("type", "skip"), source="analyzer"))
    return select_filters(rules, limit)


def _filter_key(rules: list) -> set:
    def thr(v):
        try:
            return round(float(v), 10)
        except (TypeError, ValueError):
            return str(v)
    return {(r["feature"], r["operator"], thr(r["threshold"])) for r in rules}


def bundle_changed(inc_params: dict, inc_filters: list,
                   cand_params: dict, cand_filters: list) -> bool:
    strip = lambda p: {k: v for k, v in p.items() if k != "risk"}
    return (json.dumps(strip(inc_params), sort_keys=True, default=str)
            != json.dumps(strip(cand_params), sort_keys=True, default=str)
            or _filter_key(inc_filters) != _filter_key(cand_filters))


# ── Persistence ─────────────────────────────────────────────────────────

def replace_active_filters(strategy: str, rules: list, conn=None):
    """Make ``rules`` the strategy's only active filters."""
    own = conn is None
    if own:
        conn = get_connection()
    init_filter_rules_table(conn)
    conn.execute("UPDATE filter_rules SET active = 0 WHERE strategy = ?", (strategy,))
    for r in rules:
        conn.execute("""
            INSERT INTO filter_rules
            (strategy, feature, operator, threshold, rule_type, effect_size, source, rationale)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (strategy, r["feature"], r["operator"], str(r["threshold"]),
              r.get("rule_type", "skip"), r.get("effect_size"),
              r.get("source", "promotion"), r.get("rationale", "")))
    conn.commit()
    if own:
        conn.close()


def record_evaluation(session_id: int, evaluation: dict):
    if session_id is None:
        return
    conn = get_connection()
    conn.execute("UPDATE learning_sessions SET evaluation_json = ?, applied = ? WHERE id = ?",
                 (json.dumps(evaluation, default=str), int(bool(evaluation.get("promoted"))),
                  session_id))
    conn.commit()
    conn.close()


def load_recent_evaluations(strategy: str, n: int = 3) -> list:
    """Last ``n`` evaluated proposals, newest first, for Claude's prompt."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT created_at, recommendations_json, evaluation_json FROM learning_sessions
        WHERE strategy = ? AND evaluation_json IS NOT NULL
        ORDER BY id DESC LIMIT ?
    """, (strategy, n)).fetchall()
    conn.close()
    history = []
    for row in rows:
        try:
            rec = json.loads(row["recommendations_json"] or "{}")
            ev = json.loads(row["evaluation_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        history.append({
            "created_at": row["created_at"],
            "parameter_changes": rec.get("parameter_changes", []),
            "filter_rules": [f.get("rule", "") for f in rec.get("filter_rules", [])],
            "promoted": bool(ev.get("promoted")),
            "reason": ev.get("reason", ""),
            "incumbent": ev.get("incumbent", {}),
            "candidate": ev.get("candidate", {}),
        })
    return history


# ── Orchestration ───────────────────────────────────────────────────────

def run_promotion(strategy: str, review: dict, findings: dict, config: dict,
                  trade_start, dry_run: bool = False) -> dict:
    """Evaluate Claude's proposal against the current version and promote it if better."""
    from src.backtest.costs import trading_costs
    from src.learning.filter_rules import _configured_filter_limit

    cfg = promotion_config(config)
    risk = config.get("risk", {})
    timeframe = config["trading"]["timeframes"][0]
    capital = config.get("simulation", {}).get("initial_capital", 10_000_000)

    conn = get_connection()
    inc_params = load_latest_parameters(conn, strategy)
    if inc_params is None:
        inc_params = copy.deepcopy(config.get("strategy", {}).get(strategy, {}))
    dfs = {}
    for pair in config["trading"]["pairs"]:
        df = load_ohlcv(conn, pair, timeframe)
        if not df.empty:
            dfs[pair] = df
    conn.close()

    inc_filters = load_active_filters(strategy)
    proposed = review.get("updated_parameters")
    cand_params = copy.deepcopy(proposed) if isinstance(proposed, dict) and proposed \
        else copy.deepcopy(inc_params)
    cand_filters = candidate_filters(review, findings, _configured_filter_limit())

    costs = trading_costs(config)
    inc_eval = dict(inc_params, risk=risk)
    cand_eval = dict(cand_params, risk=risk)
    incumbent = evaluate_bundle(strategy, inc_eval, inc_filters, dfs, timeframe,
                                trade_start, costs, capital)
    candidate = evaluate_bundle(strategy, cand_eval, cand_filters, dfs, timeframe,
                                trade_start, costs, capital)
    changed = bundle_changed(inc_params, inc_filters, cand_params, cand_filters)
    promote, reason = decide(incumbent, candidate, cfg, changed)

    evaluation = {
        "holdout_start": str(trade_start),
        "incumbent": incumbent,
        "candidate": candidate,
        "candidate_filters": [f"{f['feature']} {f['operator']} {f['threshold']}"
                              for f in cand_filters],
        "promoted": promote and not dry_run,
        "would_promote": promote,
        "reason": reason,
    }
    if promote and not dry_run:
        conn = get_connection()
        version = save_parameters(
            conn, strategy, cand_params, source="claude_auto_review",
            performance={"holdout": candidate, "win_rate": candidate["win_rate"]},
            notes=json.dumps(review.get("key_insights", [])),
        )
        replace_active_filters(strategy, cand_filters, conn=conn)
        conn.close()
        evaluation["version"] = version
    return evaluation
