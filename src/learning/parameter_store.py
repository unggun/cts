"""Track the evolution of strategy parameters over time.

This module lets you see how parameters have changed across versions,
what performance each version achieved, and whether the auto-learning
is actually improving things.
"""
import json
import pandas as pd

from src.data.database import get_connection, init_db


def get_parameter_history(strategy: str) -> pd.DataFrame:
    """Get the full history of parameter changes for a strategy."""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query("""
        SELECT version, parameters_json, performance_json, source, notes, created_at
        FROM strategy_parameters
        WHERE strategy = ?
        ORDER BY version ASC
    """, conn, params=[strategy])
    conn.close()

    if not df.empty:
        df["parameters"] = df["parameters_json"].apply(
            lambda x: json.loads(x) if x else {}
        )
        df["performance"] = df["performance_json"].apply(
            lambda x: json.loads(x) if x else {}
        )

    return df


def compare_versions(strategy: str, v1: int = None, v2: int = None) -> dict:
    """Compare two parameter versions side by side.

    If versions not specified, compares latest with previous.
    """
    history = get_parameter_history(strategy)
    if len(history) < 2:
        return {"error": "Need at least 2 versions to compare"}

    if v1 is None:
        v1 = history.iloc[-2]["version"]
    if v2 is None:
        v2 = history.iloc[-1]["version"]

    row1 = history[history["version"] == v1].iloc[0]
    row2 = history[history["version"] == v2].iloc[0]

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
                "v{}_value".format(v1): old,
                "v{}_value".format(v2): new,
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


def print_parameter_history(strategy: str):
    """Pretty-print parameter evolution."""
    history = get_parameter_history(strategy)
    if history.empty:
        print(f"No parameter history for strategy '{strategy}'")
        return

    print(f"\n{'='*60}")
    print(f"PARAMETER HISTORY: {strategy}")
    print(f"{'='*60}")

    for _, row in history.iterrows():
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="darvas")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        result = compare_versions(args.strategy)
        if "error" in result:
            print(result["error"])
        else:
            print(f"\nComparing v{result['version_old']} → v{result['version_new']}")
            for c in result["changes"]:
                print(f"  {c['parameter']}: {list(c.values())[1]} → {list(c.values())[2]}")
    else:
        print_parameter_history(args.strategy)
