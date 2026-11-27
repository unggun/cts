"""Per-strategy regime gate: no new buys while BTC's daily trend is down.

Whether a strategy has the gate is decided from data on every learning run (see
``decide``), never by Claude. The gate sits on top of a strategy's filter rules:
it doesn't count toward their cap and promotion never replaces it. A missing
``btc_st_1d_up`` (stale BTC candles) doesn't block, like ``apply_filters``.

Design: docs/superpowers/specs/2026-11-27-regime-gate-design.md

Usage::

    python -m src.learning.regime_gate                    # evidence for every strategy
    python -m src.learning.regime_gate --strategy cvd
    python -m src.learning.regime_gate --apply            # store the decisions
"""

import argparse
import copy
import json
import math

import numpy as np
import pandas as pd

from src.data.database import get_connection

GATE_FEATURE = "btc_st_1d_up"
GATE_REASON = "regime gate: BTC daily trend down"
DEFAULTS = {"enabled": True, "min_gain": 0.0005, "min_trades": 30}


def gate_config(config: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update((config.get("learning", {}) or {}).get("regime_gate", {}) or {})
    return cfg


def gate_blocks(features: dict) -> bool:
    """True when the gate would block an entry with these features."""
    value = (features or {}).get(GATE_FEATURE)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(value) and value < 0.5


# ── storage ────────────────────────────────────────────────────────────

def init_regime_gate_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS regime_gates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            reason TEXT,
            evidence_json TEXT,
            decided_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_regime_gates_strategy "
                 "ON regime_gates (strategy, id)")


def load_gate_state(strategy: str, conn=None) -> dict | None:
    """The newest stored decision for ``strategy``, or None if never decided."""
    own = conn is None
    if own:
        conn = get_connection()
    try:
        init_regime_gate_table(conn)
        row = conn.execute(
            "SELECT enabled, reason, evidence_json, decided_at FROM regime_gates "
            "WHERE strategy = ? ORDER BY id DESC LIMIT 1", (strategy,)).fetchone()
    finally:
        if own:
            conn.close()
    if row is None:
        return None
    try:
        evidence = json.loads(row[2]) if row[2] else None
    except (TypeError, ValueError):
        evidence = None
    return {"enabled": bool(row[0]), "reason": row[1], "evidence": evidence,
            "decided_at": row[3]}


def gate_enabled(strategy: str, config: dict = None, conn=None) -> bool:
    """Whether new entries of ``strategy`` are gated right now."""
    if config is None:
        from src.config import load_config
        config = load_config()
    if not gate_config(config)["enabled"]:
        return False
    state = load_gate_state(strategy, conn=conn)
    return bool(state and state["enabled"])


def save_gate_state(strategy: str, enabled: bool, reason: str, evidence: dict,
                    conn=None) -> None:
    own = conn is None
    if own:
        conn = get_connection()
    try:
        init_regime_gate_table(conn)
        conn.execute("INSERT INTO regime_gates (strategy, enabled, reason, evidence_json) "
                     "VALUES (?, ?, ?, ?)",
                     (strategy, int(enabled), reason, json.dumps(evidence, default=str)))
        conn.commit()
    finally:
        if own:
            conn.close()


# ── evidence and decision ──────────────────────────────────────────────

def _summary(trades: list, midpoint) -> dict:
    pnls = [t["pnl_pct"] for t in trades]

    def half(first):
        sel = [t["pnl_pct"] for t in trades
               if (pd.Timestamp(t["entry_time"]) < midpoint) == first]
        return {"trades": len(sel), "expectancy": float(np.mean(sel)) if sel else None}

    return {"trades": len(pnls), "expectancy": float(np.mean(pnls)) if pnls else None,
            "first_half": half(True), "second_half": half(False)}


def gather_evidence(strategy: str, config: dict, dfs: dict = None,
                    btc_df: pd.DataFrame = None) -> dict:
    """Backtest the current version (with its filters) over all stored closed
    candles, with and without the gate."""
    from src.backtest.costs import trading_costs
    from src.backtest.engine import BacktestEngine
    from src.data.database import load_latest_parameters
    from src.learning.filter_rules import load_active_filters
    from src.learning.promotion import load_closed_candles

    if dfs is None:
        dfs, btc_df = load_closed_candles(config)
    btc_df = btc_df if btc_df is not None else pd.DataFrame()
    timeframe = config["trading"]["timeframes"][0]
    conn = get_connection()
    try:
        params = load_latest_parameters(conn, strategy)
    finally:
        conn.close()
    if params is None:
        params = copy.deepcopy(config.get("strategy", {}).get(strategy, {}))
    # The legacy nested "filters" dict is unused by every strategy.
    params = {k: v for k, v in params.items() if k not in ("filters", "risk")}
    params["risk"] = config.get("risk", {})
    filters = load_active_filters(strategy)
    costs = trading_costs(config)
    capital = config.get("simulation", {}).get("initial_capital", 10_000_000)

    runs = {}
    for gated in (False, True):
        trades = []
        for pair, df in dfs.items():
            engine = BacktestEngine(strategy, copy.deepcopy(params), capital, costs=costs,
                                    filters=filters, gate=gated)
            # The first 200 bars only warm up the indicators.
            start = df.index[min(200, len(df) - 1)] if len(df) else None
            engine.run(df.copy(), pair, timeframe, trade_start=start, btc_df=btc_df)
            trades.extend(engine.trades)
        runs[gated] = trades
    starts = [d.index[min(200, len(d) - 1)] for d in dfs.values() if len(d)]
    ends = [d.index[-1] for d in dfs.values() if len(d)]
    if not starts:
        return {"error": "no candles"}
    midpoint = min(starts) + (max(ends) - min(starts)) / 2
    return {"window": [str(min(starts)), str(max(ends))], "midpoint": str(midpoint),
            "ungated": _summary(runs[False], midpoint), "gated": _summary(runs[True], midpoint)}


def decide(currently_on: bool, evidence: dict, cfg: dict) -> tuple[bool, str]:
    """Apply the on/off rule. Returns ``(enabled, reason)``.

    On: gated beats ungated by ``min_gain`` in both halves, gated expectancy is
    positive, and at least ``min_trades`` gated trades. Off: gated is worse than
    ungated over the whole window. Otherwise the state is kept (hysteresis).
    """
    g, u = evidence.get("gated") or {}, evidence.get("ungated") or {}
    ge, ue = g.get("expectancy"), u.get("expectancy")
    if ge is None or ue is None:
        return currently_on, "kept: not enough trades to compare"
    if currently_on and ge < ue:
        return False, f"off: gated {ge:+.3%} < ungated {ue:+.3%} over the window"
    halves = []
    for h in ("first_half", "second_half"):
        gh, uh = (g.get(h) or {}).get("expectancy"), (u.get(h) or {}).get("expectancy")
        halves.append(gh is not None and uh is not None and gh >= uh + cfg["min_gain"])
    if all(halves) and ge > 0 and g.get("trades", 0) >= cfg["min_trades"]:
        state = "kept on" if currently_on else "on"
        return True, (f"{state}: gated {ge:+.3%} vs ungated {ue:+.3%} on {g['trades']} trades, "
                      f"better in both halves")
    why = []
    if not all(halves):
        why.append("not better in both halves")
    if ge <= 0:
        why.append(f"gated expectancy {ge:+.3%} not positive")
    if g.get("trades", 0) < cfg["min_trades"]:
        why.append(f"only {g.get('trades', 0)} gated trades (need {cfg['min_trades']})")
    state = "kept on" if currently_on else "stays off"
    return currently_on, f"{state}: {'; '.join(why)} (gated {ge:+.3%} vs ungated {ue:+.3%})"


def review_gate(strategy: str, config: dict, apply: bool = True, dfs: dict = None,
                btc_df: pd.DataFrame = None) -> dict:
    """Gather evidence, decide, and (with ``apply``) store the decision."""
    cfg = gate_config(config)
    state = load_gate_state(strategy)
    currently_on = bool(state and state["enabled"])
    if not cfg["enabled"]:
        return {"strategy": strategy, "enabled": False, "was_enabled": currently_on,
                "changed": False, "reason": "regime gate disabled in config"}
    evidence = gather_evidence(strategy, config, dfs=dfs, btc_df=btc_df)
    if "error" in evidence:
        return {"strategy": strategy, "enabled": currently_on, "was_enabled": currently_on,
                "changed": False, "reason": f"kept: {evidence['error']}"}
    enabled, reason = decide(currently_on, evidence, cfg)
    if apply:
        save_gate_state(strategy, enabled, reason, evidence)
    return {"strategy": strategy, "enabled": enabled, "was_enabled": currently_on,
            "changed": enabled != currently_on, "reason": reason, "evidence": evidence,
            "applied": apply}


def format_review(r: dict) -> str:
    mark = "🛡 ON " if r["enabled"] else "   off"
    change = " (changed)" if r.get("changed") else ""
    lines = [f"{r['strategy']:<15} {mark}{change}: {r['reason']}"]
    ev = r.get("evidence") or {}
    for key in ("ungated", "gated"):
        s = ev.get(key)
        if s:
            def pct(x):
                return "—" if x is None else f"{x:+.2%}"
            lines.append(f"    {key:<8} {s['trades']:>4} trades {pct(s['expectancy'])} "
                         f"(halves {pct(s['first_half']['expectancy'])} / "
                         f"{pct(s['second_half']['expectancy'])})")
    return "\n".join(lines)


def main():
    from src.config import load_config
    from src.data.database import init_db
    from src.execution.live_readiness import KNOWN_STRATEGIES
    from src.learning.promotion import load_closed_candles

    parser = argparse.ArgumentParser(description="Per-strategy regime gate decisions")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--apply", action="store_true", help="store the decisions")
    args = parser.parse_args()
    init_db()
    config = load_config()
    dfs, btc_df = load_closed_candles(config)
    for strategy in [args.strategy] if args.strategy else KNOWN_STRATEGIES:
        print(format_review(review_gate(strategy, config, apply=args.apply,
                                        dfs=dfs, btc_df=btc_df)))
    if not args.apply:
        print("\n(not stored; run with --apply to store these decisions)")


if __name__ == "__main__":
    main()
