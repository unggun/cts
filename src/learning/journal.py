"""Research journal: each Claude proposal, its holdout result, and how the
promoted version then did on paper.

Read-only. Links paper trades to parameter versions by
``trade_journal.param_id`` (or, for older trades, by time), compares paper
results with the holdout metrics stored on the version, and flags versions
that are clearly behind. Nothing is rolled back automatically.

    python -m src.learning.journal
    python -m src.learning.journal --strategy cvd [--limit 10] [--json]
"""
import argparse
import json
import math
from datetime import datetime, timezone

from src.data.database import get_connection, init_db
from src.execution.live_readiness import KNOWN_STRATEGIES

MIN_PAPER_TRADES = 20
BEHIND_Z = 2.0


# ── Time helpers ────────────────────────────────────────────────────────

def _local(ts: str) -> datetime:
    """``datetime('now')`` (UTC, naive) -> naive local time, like ``entry_time``."""
    dt = datetime.fromisoformat(ts.replace("T", " "))
    return dt.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("T", " "))


def _loads(text):
    try:
        value = json.loads(text) if text else None
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ── Loading ─────────────────────────────────────────────────────────────

def load_versions(conn, strategy: str) -> list:
    """Parameter versions oldest first, with the holdout metrics they won on."""
    rows = conn.execute(
        "SELECT id, version, source, performance_json, created_at FROM strategy_parameters "
        "WHERE strategy = ? ORDER BY created_at, id", (strategy,)).fetchall()
    versions = []
    for row in rows:
        perf = _loads(row["performance_json"]) or {}
        holdout = perf.get("holdout") if isinstance(perf.get("holdout"), dict) else None
        versions.append({"id": row["id"], "version": row["version"], "source": row["source"],
                         "created_at": row["created_at"], "start": _local(row["created_at"]),
                         "holdout": holdout})
    for v, nxt in zip(versions, versions[1:] + [None]):
        v["end"] = nxt["start"] if nxt else None
    return versions


def load_paper_trades(conn, strategy: str) -> list:
    rows = conn.execute(
        "SELECT entry_time, exit_time, pnl_pct, pnl_absolute, param_id FROM trade_journal "
        "WHERE strategy = ? AND mode = 'paper' ORDER BY entry_time", (strategy,)).fetchall()
    return [dict(r) for r in rows]


def assign_versions(trades: list, versions: list) -> dict:
    """{version id: [trades]}; by ``param_id``, else by entry time."""
    by_id = {v["id"]: [] for v in versions}
    for t in trades:
        if t.get("param_id") in by_id:
            by_id[t["param_id"]].append(t)
            continue
        entry = _parse(t["entry_time"])
        for v in versions:
            if v["start"] <= entry and (v["end"] is None or entry < v["end"]):
                by_id[v["id"]].append(t)
                break
    return by_id


# ── Stats and verdict ───────────────────────────────────────────────────

def paper_stats(trades: list, start: datetime, end: datetime | None,
                capital: float, now: datetime = None) -> dict:
    closed = [t for t in trades if t["exit_time"] is not None and t["pnl_pct"] is not None]
    closed.sort(key=lambda t: t["exit_time"])
    pnl = [t["pnl_pct"] for t in closed]
    n = len(pnl)
    mean = sum(pnl) / n if n else None
    se = None
    if n >= 2:
        var = sum((x - mean) ** 2 for x in pnl) / (n - 1)
        se = math.sqrt(var / n)
    equity, peak, max_dd = capital, capital, 0.0
    for t in closed:
        equity += t["pnl_absolute"] or 0.0
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    window_end = end or now or datetime.now()
    days = max((window_end - start).total_seconds() / 86400, 0.0)
    return {"closed": n, "open": len(trades) - len([t for t in trades if t["exit_time"]]),
            "expectancy": mean, "se": se,
            "win_rate": sum(1 for x in pnl if x > 0) / n if n else None,
            "max_drawdown": max_dd if n else None,
            "days": days, "trades_per_week": len(trades) / days * 7 if days >= 1 else None}


def verdict(holdout: dict | None, paper: dict) -> str:
    exp = _num((holdout or {}).get("expectancy"))
    if exp is None:
        return "NO_PREDICTION"
    if paper["closed"] < MIN_PAPER_TRADES or paper["se"] is None:
        return "TOO_FEW"
    if paper["expectancy"] + BEHIND_Z * paper["se"] < exp:
        return "BEHIND"
    return "ON_TRACK"


def hypothesis(rec: dict) -> dict:
    h = rec.get("hypothesis") if isinstance(rec.get("hypothesis"), dict) else {}
    exp_pct = _num(h.get("expected_expectancy_pct"))
    trades = _num(h.get("expected_trades"))
    idea = h.get("idea")
    return {"idea": idea if isinstance(idea, str) and idea.strip() else None,
            "expected_expectancy": exp_pct / 100 if exp_pct is not None else None,
            "expected_trades": int(trades) if trades is not None else None}


def outcome(ev: dict | None) -> str:
    if ev is None:
        return "no holdout test (before promotion testing)"
    reason = ev.get("reason") or ""
    if "candidate" not in ev:
        return f"not evaluated: {reason}"
    if ev.get("promoted"):
        label = "all filters removed" if ev.get("filters_removed") else "promoted"
        return f"{label} → v{ev.get('version')}"
    return f"rejected: {reason}"


# ── Journal ─────────────────────────────────────────────────────────────

def _holdout_days(config: dict) -> int:
    from src.learning.promotion import promotion_config
    return promotion_config(config)["holdout_days"]


def _pct_num(x) -> str:
    """A percent number (already ×100), as stored by the backtest tool."""
    x = _num(x)
    return "—" if x is None else f"{x:+.2f}%"


def backtests(rec: dict) -> list:
    """Claude's train-window backtests for a review, one short line each."""
    bt = (rec or {}).get("backtests")
    calls = bt.get("calls") if isinstance(bt, dict) else None
    lines = []
    for c in calls if isinstance(calls, list) else []:
        if not isinstance(c, dict):
            continue
        changes = c.get("parameter_changes") if isinstance(c.get("parameter_changes"), dict) else {}
        filters = c.get("applied_filters") if isinstance(c.get("applied_filters"), list) else []
        what = "; ".join([f"{k}={v}" for k, v in changes.items()] +
                         [f"skip {f}" for f in filters]) or "current params, no filters"
        res = c.get("result") if isinstance(c.get("result"), dict) else {}
        halves = [h if isinstance(h, dict) else {} for h in
                  (res.get("first_half"), res.get("second_half"))]
        lines.append(f"{what} → {res.get('trades', '—')} trades, "
                     f"{_pct_num(res.get('expectancy_pct'))}/trade "
                     f"(halves {_pct_num(halves[0].get('expectancy_pct'))} / "
                     f"{_pct_num(halves[1].get('expectancy_pct'))})")
    return lines


def build_journal(strategy: str, limit: int = 10, db_path: str = None,
                  config: dict = None, now: datetime = None) -> dict:
    if config is None:
        from src.config import load_config
        config = load_config()
    capital = config.get("simulation", {}).get("initial_capital", 10_000_000)
    holdout_days = _holdout_days(config)
    conn = get_connection(db_path)
    try:
        versions = load_versions(conn, strategy)
        by_version = assign_versions(load_paper_trades(conn, strategy), versions)
        sessions = conn.execute(
            "SELECT id, created_at, recommendations_json, evaluation_json FROM learning_sessions "
            "WHERE strategy = ? AND session_type = 'claude_review' ORDER BY id DESC LIMIT ?",
            (strategy, limit)).fetchall()
    finally:
        conn.close()

    for v in versions:
        v["paper"] = paper_stats(by_version[v["id"]], v["start"], v["end"], capital, now)
        v["verdict"] = verdict(v["holdout"], v["paper"])
        trades = _num((v["holdout"] or {}).get("trades"))
        v["holdout_trades_per_week"] = trades / holdout_days * 7 if trades else None
    by_id = {v["id"]: v for v in versions}
    by_number = {v["version"]: v for v in versions}

    reviews = []
    for s in sessions:
        rec = _loads(s["recommendations_json"]) or {}
        ev = _loads(s["evaluation_json"])
        entry = {"session_id": s["id"], "created_at": s["created_at"],
                 "confidence": rec.get("confidence"), "hypothesis": hypothesis(rec),
                 "outcome": outcome(ev), "candidate": None, "incumbent": None, "version": None,
                 "param_id": None, "backtests": backtests(rec)}
        if ev and "candidate" in ev:
            entry["candidate"] = ev.get("candidate") if isinstance(ev.get("candidate"), dict) else None
            entry["incumbent"] = ev.get("incumbent") if isinstance(ev.get("incumbent"), dict) else None
            if ev.get("promoted"):
                v = _promoted_version(ev, s["created_at"], by_id, by_number)
                if v:
                    entry["version"], entry["param_id"] = v["version"], v["id"]
        reviews.append(entry)
    return {"strategy": strategy, "current": versions[-1] if versions else None,
            "versions": versions, "reviews": reviews, "holdout_days": holdout_days}


# A promotion is saved within seconds of its review; older evaluations carry only
# the version number, which restarts at 1 after ``--reset-params``.
LEGACY_LINK_WINDOW_SECONDS = 3600


def _promoted_version(ev: dict, review_created_at: str, by_id: dict, by_number: dict):
    """The parameter row a promoted review saved, or None if it no longer exists."""
    if "param_id" in ev:
        return by_id.get(ev["param_id"])
    v = by_number.get(ev.get("version"))
    if not v:
        return None
    try:
        gap = (_parse(v["created_at"]) - _parse(review_created_at)).total_seconds()
    except (TypeError, ValueError):
        return None
    return v if 0 <= gap <= LEGACY_LINK_WINDOW_SECONDS else None


def overview(db_path: str = None, config: dict = None, now: datetime = None) -> list:
    rows = []
    for strategy in KNOWN_STRATEGIES:
        j = build_journal(strategy, limit=0, db_path=db_path, config=config, now=now)
        if j["current"]:
            rows.append({"strategy": strategy, "current": j["current"]})
    return rows


def behind_versions(db_path: str = None, config: dict = None) -> list:
    return [r for r in overview(db_path, config) if r["current"]["verdict"] == "BEHIND"]


# ── Formatting ──────────────────────────────────────────────────────────

def _pct(x, digits=2):
    return "—" if x is None else f"{x * 100:+.{digits}f}%"


def _holdout_text(h: dict | None) -> str:
    if not h:
        return "none"
    return (f"{int(h.get('trades') or 0)} trades, {_pct(_num(h.get('expectancy')))}/trade, "
            f"max DD {_pct(_num(h.get('max_drawdown')), 1).lstrip('+')}")


def _paper_text(p: dict) -> str:
    text = (f"{p['closed']} closed ({p['open']} open), {_pct(p['expectancy'])}/trade")
    if p["win_rate"] is not None:
        text += f", WR {p['win_rate']:.0%}, max DD {p['max_drawdown'] * 100:.1f}%"
    return text


def format_version(v: dict, holdout_days: int) -> list:
    rate = v["paper"]["trades_per_week"]
    hrate = v["holdout_trades_per_week"]
    lines = [f"v{v['version']} ({v['source']}) since {v['created_at']} UTC",
             f"  Holdout ({holdout_days}d): {_holdout_text(v['holdout'])}",
             f"  Paper:  {_paper_text(v['paper'])}"]
    if rate is not None or hrate is not None:
        lines.append(f"  Trades/week: paper {'—' if rate is None else f'{rate:.1f}'}"
                     f" vs holdout {'—' if hrate is None else f'{hrate:.1f}'}")
    lines.append(f"  Verdict: {v['verdict']}")
    return lines


def format_strategy(j: dict) -> str:
    lines = [f"RESEARCH JOURNAL: {j['strategy']}", ""]
    if j["current"]:
        lines.append("Current version")
        lines += ["  " + line for line in format_version(j["current"], j["holdout_days"])]
    else:
        lines.append("No saved parameter version.")
    by_id = {v["id"]: v for v in j["versions"]}
    if j["reviews"]:
        lines += ["", "Claude reviews (newest first)"]
    for r in j["reviews"]:
        h = r["hypothesis"]
        lines.append(f"- {r['created_at']} UTC, confidence {r['confidence'] or '—'}")
        if h["idea"]:
            lines.append(f"    Idea: {h['idea']}")
        if h["expected_expectancy"] is not None or h["expected_trades"] is not None:
            lines.append(f"    Predicted: {_pct(h['expected_expectancy'])}/trade, "
                         f"{h['expected_trades'] if h['expected_trades'] is not None else '—'} trades")
        if r.get("backtests"):
            lines.append(f"    Train backtests ({len(r['backtests'])}):")
            lines += [f"      · {b}" for b in r["backtests"]]
        if r["candidate"]:
            lines.append(f"    Holdout: {_holdout_text(r['candidate'])} "
                         f"(current {_holdout_text(r['incumbent'])})")
        lines.append(f"    Outcome: {r['outcome']}")
        v = by_id.get(r.get("param_id"))
        if v:
            lines.append(f"    Paper since: {_paper_text(v['paper'])} → {v['verdict']}")
    return "\n".join(lines)


def format_overview(rows: list) -> str:
    lines = ["RESEARCH JOURNAL — current versions", ""]
    for r in rows:
        v = r["current"]
        hexp = _num((v["holdout"] or {}).get("expectancy"))
        lines.append(f"{r['strategy']:15s} v{v['version']:<3d} holdout {_pct(hexp):>8s} | "
                     f"paper {_pct(v['paper']['expectancy']):>8s} on {v['paper']['closed']:3d} | "
                     f"{v['verdict']}")
    return "\n".join(lines)


def _jsonable(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(type(obj).__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research journal: proposals, holdout and paper results")
    parser.add_argument("--strategy", choices=KNOWN_STRATEGIES)
    parser.add_argument("--limit", type=int, default=10, help="Claude reviews to show")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    init_db()
    if args.strategy:
        data = build_journal(args.strategy, limit=args.limit)
        print(json.dumps(data, default=_jsonable, indent=2) if args.json else format_strategy(data))
    else:
        data = overview()
        print(json.dumps(data, default=_jsonable, indent=2) if args.json else format_overview(data))
