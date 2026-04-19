"""Evaluate per-strategy readiness to flip from paper to live trading."""
import json
from datetime import datetime, timedelta

from src.data.database import get_connection, init_db

KNOWN_STRATEGIES = [
    "darvas", "dbw", "candlestick", "macd", "ma_crossover",
    "mean_reversion", "sr_breakout", "cvd", "bb_squeeze", "rsi_vwap",
]

MIN_MC_TRADES = 50
MAX_RUIN_PROB = 0.03
MIN_PAPER_TRADES = 30
MIN_PAPER_WR = 0.50
PAPER_LOOKBACK_DAYS = 30


def load_mc_result(strategy: str, db_path: str = None) -> dict | None:
    """Load latest Monte Carlo simulation result for a strategy."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT verdict, probability_of_ruin, details_json, created_at "
        "FROM simulation_results WHERE strategy = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (strategy,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    details = {}
    if row["details_json"]:
        try:
            details = json.loads(row["details_json"])
        except (json.JSONDecodeError, TypeError):
            details = {}
    return {
        "verdict": row["verdict"],
        "probability_of_ruin": row["probability_of_ruin"],
        "num_source_trades": int(details.get("num_source_trades", 0)),
        "created_at": row["created_at"],
    }


def load_confidence(strategy: str, db_path: str = None) -> str | None:
    """Load confidence level from latest Claude review for a strategy."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT recommendations_json FROM learning_sessions "
        "WHERE strategy = ? AND session_type = 'claude_review' "
        "ORDER BY created_at DESC LIMIT 1",
        (strategy,),
    ).fetchone()
    conn.close()
    if row is None or not row["recommendations_json"]:
        return None
    try:
        recs = json.loads(row["recommendations_json"])
    except (json.JSONDecodeError, TypeError):
        return None
    conf = recs.get("confidence")
    if conf in ("high", "medium", "low"):
        return conf
    return None


def load_param_cutover(strategy: str, db_path: str = None) -> str | None:
    """Load created_at timestamp of the latest parameter version for a strategy."""
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT created_at FROM strategy_parameters "
        "WHERE strategy = ? ORDER BY version DESC LIMIT 1",
        (strategy,),
    ).fetchone()
    conn.close()
    return row["created_at"] if row else None


def _parse_ts(s: str) -> datetime:
    """Parse an ISO timestamp from the DB, accepting either 'T' or ' ' separator."""
    return datetime.fromisoformat(s.replace("T", " "))


def _empty_bucket() -> dict:
    """Return an empty stats bucket template."""
    return {"closed": 0, "wins": 0, "win_rate": None, "total_pnl_pct": 0.0}


def load_paper_stats(strategy: str, cutover: str | None, db_path: str = None) -> dict:
    """Load paper trading stats, split into pre/current around cutover timestamp."""
    cutoff_dt = datetime.utcnow() - timedelta(days=PAPER_LOOKBACK_DAYS)
    cutover_dt = _parse_ts(cutover) if cutover else None

    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT entry_time, exit_time, pnl_pct FROM trade_journal "
        "WHERE mode = 'paper' AND strategy = ?",
        (strategy,),
    ).fetchall()
    conn.close()

    buckets = {"current": _empty_bucket(), "pre_update": _empty_bucket()}
    for row in rows:
        if row["exit_time"] is None:
            continue
        entry_dt = _parse_ts(row["entry_time"])
        if entry_dt < cutoff_dt:
            continue
        bucket_name = "current" if (cutover_dt is None or entry_dt >= cutover_dt) else "pre_update"
        b = buckets[bucket_name]
        b["closed"] += 1
        pnl = row["pnl_pct"] or 0.0
        b["total_pnl_pct"] += pnl
        if pnl > 0:
            b["wins"] += 1

    for b in buckets.values():
        if b["closed"] > 0:
            b["win_rate"] = b["wins"] / b["closed"]
    return buckets


def evaluate_strategy(mc: dict | None, confidence: str | None, paper_stats: dict) -> dict:
    """Evaluate readiness of a strategy to flip from paper to live trading.

    Pure function: takes loaded MC result, confidence level, and paper stats.
    Returns status (INSUFFICIENT_DATA, BLOCKED, WATCH, READY) and gates list.
    """
    cur = paper_stats["current"]
    gates = []

    mc_verdict = mc["verdict"] if mc else None
    mc_ruin = mc["probability_of_ruin"] if mc else None
    mc_trades = mc["num_source_trades"] if mc else 0

    gates.append({"name": "mc_verdict", "value": mc_verdict,
                  "pass": mc_verdict == "ROBUST", "detail": "need ROBUST"})
    gates.append({"name": "probability_of_ruin", "value": mc_ruin,
                  "pass": mc_ruin is not None and mc_ruin < MAX_RUIN_PROB,
                  "detail": f"need < {MAX_RUIN_PROB:.0%}"})
    gates.append({"name": "mc_source_trades", "value": mc_trades,
                  "pass": mc_trades >= MIN_MC_TRADES,
                  "detail": f"need >= {MIN_MC_TRADES}"})
    gates.append({"name": "confidence", "value": confidence,
                  "pass": confidence == "high", "detail": "need high"})
    gates.append({"name": "paper_trades", "value": cur["closed"],
                  "pass": cur["closed"] >= MIN_PAPER_TRADES,
                  "detail": f"need >= {MIN_PAPER_TRADES} closed (current params)"})
    wr_pass = cur["win_rate"] is not None and cur["win_rate"] >= MIN_PAPER_WR
    gates.append({"name": "paper_win_rate", "value": cur["win_rate"],
                  "pass": wr_pass, "detail": f"need >= {MIN_PAPER_WR:.0%}"})
    gates.append({"name": "paper_total_pnl", "value": cur["total_pnl_pct"],
                  "pass": cur["total_pnl_pct"] > 0, "detail": "need > 0"})

    # Status tiers — first match wins.
    if mc is None or mc_trades < MIN_MC_TRADES:
        status = "INSUFFICIENT_DATA"
    elif (
        mc_verdict in ("FRAGILE", "RISKY")
        or (mc_ruin is not None and mc_ruin >= MAX_RUIN_PROB)
        or confidence in (None, "low")
        or (cur["closed"] >= MIN_PAPER_TRADES and cur["total_pnl_pct"] <= 0)
    ):
        status = "BLOCKED"
    elif (
        cur["closed"] < MIN_PAPER_TRADES
        or confidence == "medium"
        or (cur["win_rate"] is not None and cur["win_rate"] < MIN_PAPER_WR
            and cur["total_pnl_pct"] > 0)
    ):
        status = "WATCH"
    else:
        status = "READY"

    return {"status": status, "gates": gates}
