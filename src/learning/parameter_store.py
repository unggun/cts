"""Track the evolution of strategy parameters over time.

This module lets you see how parameters have changed across versions,
what performance each version achieved, and whether the auto-learning
is actually improving things.
"""
import json

from src.data.database import get_connection, init_db


def get_parameter_history(strategy: str) -> list[dict]:
    """Get the full history of parameter changes for a strategy."""
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT version, parameters_json, performance_json, source, notes, created_at
        FROM strategy_parameters
        WHERE strategy = ?
        ORDER BY version ASC
    """, (strategy,))
    rows = cur.fetchall()
    conn.close()

    history = []
    for row in rows:
        entry = {
            "version": row[0],
            "parameters_json": row[1],
            "performance_json": row[2],
            "source": row[3],
            "notes": row[4],
            "created_at": row[5],
            "parameters": json.loads(row[1]) if row[1] else {},
            "performance": json.loads(row[2]) if row[2] else {},
        }
        history.append(entry)
    return history


def compare_versions(strategy: str, v1: int = None, v2: int = None) -> dict:
    """Compare two parameter versions side by side.

    If versions not specified, compares latest with previous.
    """
    history = get_parameter_history(strategy)
    if len(history) < 2:
        return {"error": "Need at least 2 versions to compare"}

    if v1 is None:
        v1 = history[-2]["version"]
    if v2 is None:
        v2 = history[-1]["version"]

    row1 = next(h for h in history if h["version"] == v1)
    row2 = next(h for h in history if h["version"] == v2)

    p1 = row1["parameters"]
    p2 = row2["parameters"]

    changes = []
    all_keys = set(list(p1.keys()) + list(p2.keys()))
    for key in sorted(all_keys):
        old = p1.get(key)
        new = p2.get(key)
        if old != new:
            changes.append({
                "parameter": key,
                "old": old,
                "new": new,
            })

    return {
        "strategy": strategy,
        "version_old": v1,
        "version_new": v2,
        "source_old": row1["source"],
        "source_new": row2["source"],
        "changes": changes,
        "performance_old": row1["performance"],
        "performance_new": row2["performance"],
    }


def get_performance_trend(strategy: str) -> list[dict]:
    """Get win rate trend across parameter versions to see if learning is improving."""
    history = get_parameter_history(strategy)
    trend = []
    for entry in history:
        perf = entry["performance"]
        wr = perf.get("win_rate")
        if isinstance(wr, (int, float)):
            wr_display = f"{wr:.1%}"
        else:
            wr_display = str(wr) if wr else "?"
        trend.append({
            "version": entry["version"],
            "source": entry["source"],
            "win_rate": wr,
            "win_rate_display": wr_display,
            "created_at": entry["created_at"],
        })
    return trend


def format_param_diff(strategy: str) -> str:
    """Format parameter changes between last two versions as a short string for notifications."""
    diff = compare_versions(strategy)
    if "error" in diff:
        return ""
    if not diff["changes"]:
        return "No parameter changes"

    lines = []
    for c in diff["changes"]:
        old_val = c["old"]
        new_val = c["new"]
        if isinstance(old_val, float):
            old_val = f"{old_val:.4g}"
        if isinstance(new_val, float):
            new_val = f"{new_val:.4g}"
        lines.append(f"  {c['parameter']}: {old_val} → {new_val}")
    return "\n".join(lines)


def print_parameter_history(strategy: str):
    """Pretty-print parameter evolution."""
    history = get_parameter_history(strategy)
    if not history:
        print(f"No parameter history for strategy '{strategy}'")
        return

    print(f"\n{'='*60}")
    print(f"PARAMETER HISTORY: {strategy}")
    print(f"{'='*60}")

    for row in history:
        perf = row["performance"]
        wr = perf.get("win_rate", "?")
        if isinstance(wr, float):
            wr = f"{wr:.1%}"
        print(f"\nVersion {row['version']} ({row['source']}) — {row['created_at']}")
        print(f"  Win rate: {wr}")
        params = row["parameters"]
        for k, v in params.items():
            if k != "risk":
                print(f"  {k}: {v}")


def print_performance_trend(strategy: str):
    """Pretty-print win rate trend across versions."""
    trend = get_performance_trend(strategy)
    if not trend:
        print(f"No performance data for strategy '{strategy}'")
        return

    print(f"\n{'='*60}")
    print(f"PERFORMANCE TREND: {strategy}")
    print(f"{'='*60}")

    for t in trend:
        bar = ""
        if isinstance(t["win_rate"], (int, float)) and t["win_rate"] is not None:
            bar_len = int(t["win_rate"] * 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  v{t['version']:>3} ({t['source']:<20}) {t['win_rate_display']:>6}  {bar}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="darvas")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--trend", action="store_true")
    args = parser.parse_args()

    if args.compare:
        result = compare_versions(args.strategy)
        if "error" in result:
            print(result["error"])
        else:
            print(f"\nComparing v{result['version_old']} → v{result['version_new']}")
            print(f"  Performance: {result['performance_old']} → {result['performance_new']}")
            for c in result["changes"]:
                print(f"  {c['parameter']}: {c['old']} → {c['new']}")
    elif args.trend:
        print_performance_trend(args.strategy)
    else:
        print_parameter_history(args.strategy)
