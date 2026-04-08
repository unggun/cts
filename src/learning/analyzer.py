"""Winner vs Loser trade analysis — the core auto-learning component.

Analyzes features of winning and losing trades to discover actionable
patterns, then generates filter rules that can be applied to future trades.
"""
import json
from typing import Optional

import numpy as np
import pandas as pd

from src.data.database import get_connection, init_db, load_backtest_trades


def analyze_winners_vs_losers(run_id: str = None, strategy: str = None,
                               min_trades: int = 20) -> dict:
    """Analyze feature distributions of winners vs losers.

    Returns:
        dict with findings, feature importance, and suggested filters
    """
    init_db()
    conn = get_connection()
    trades_df = load_backtest_trades(conn, run_id=run_id, strategy=strategy)
    conn.close()

    if len(trades_df) < min_trades:
        return {"error": f"Need at least {min_trades} trades, got {len(trades_df)}"}

    # Parse features JSON
    features_list = []
    for _, row in trades_df.iterrows():
        try:
            features = json.loads(row["features_json"]) if row["features_json"] else {}
        except (json.JSONDecodeError, TypeError):
            features = {}
        features["pnl_pct"] = row["pnl_pct"]
        features["is_winner"] = 1 if row["pnl_pct"] > 0 else 0
        features_list.append(features)

    if not features_list:
        return {"error": "No feature data found in trades"}

    feat_df = pd.DataFrame(features_list)

    # Split winners and losers
    winners = feat_df[feat_df["is_winner"] == 1]
    losers = feat_df[feat_df["is_winner"] == 0]

    overall_win_rate = len(winners) / len(feat_df)

    # Exit reason breakdown (informational only — not used for filtering
    # since exit_reason is determined after trade closes)
    exit_reason_stats = {}
    if "exit_reason" in trades_df.columns:
        for reason in trades_df["exit_reason"].dropna().unique():
            reason_trades = trades_df[trades_df["exit_reason"] == reason]
            reason_winners = reason_trades[reason_trades["pnl_pct"] > 0]
            exit_reason_stats[reason] = {
                "count": len(reason_trades),
                "win_rate": len(reason_winners) / len(reason_trades) if len(reason_trades) > 0 else 0,
            }

    findings = {
        "total_trades": len(feat_df),
        "winners": len(winners),
        "losers": len(losers),
        "overall_win_rate": overall_win_rate,
        "avg_win_pct": winners["pnl_pct"].mean() if len(winners) > 0 else 0,
        "avg_loss_pct": losers["pnl_pct"].mean() if len(losers) > 0 else 0,
        "exit_reason_stats": exit_reason_stats,
        "feature_analysis": {},
        "suggested_filters": [],
        "warnings": [],
    }

    # Analyze each numeric feature
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
    skip_cols = {"pnl_pct", "is_winner"}

    for col in numeric_cols:
        if col in skip_cols:
            continue
        if feat_df[col].isna().all():
            continue

        w_vals = winners[col].dropna()
        l_vals = losers[col].dropna()

        if len(w_vals) < 3 or len(l_vals) < 3:
            continue

        w_mean = w_vals.mean()
        l_mean = l_vals.mean()
        total_mean = feat_df[col].dropna().mean()
        total_std = feat_df[col].dropna().std()

        if total_std == 0:
            continue

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((w_vals.std()**2 + l_vals.std()**2) / 2)
        if pooled_std > 0:
            effect_size = (w_mean - l_mean) / pooled_std
        else:
            effect_size = 0

        analysis = {
            "winner_mean": round(w_mean, 4),
            "loser_mean": round(l_mean, 4),
            "difference": round(w_mean - l_mean, 4),
            "effect_size": round(effect_size, 4),
            "winner_median": round(w_vals.median(), 4),
            "loser_median": round(l_vals.median(), 4),
        }

        findings["feature_analysis"][col] = analysis

        # Generate filter suggestions for strong effects
        if abs(effect_size) > 0.3:
            direction = "higher" if effect_size > 0 else "lower"
            if effect_size > 0.3:
                # Winners have higher values — suggest minimum threshold
                threshold = l_mean + 0.5 * (w_mean - l_mean)
                findings["suggested_filters"].append({
                    "feature": col,
                    "rule": f"{col} >= {threshold:.4f}",
                    "effect_size": round(effect_size, 3),
                    "rationale": f"Winners avg {w_mean:.4f} vs losers {l_mean:.4f} ({direction} is better)",
                })
            elif effect_size < -0.3:
                # Winners have lower values — suggest maximum threshold
                threshold = w_mean + 0.5 * (l_mean - w_mean)
                findings["suggested_filters"].append({
                    "feature": col,
                    "rule": f"{col} <= {threshold:.4f}",
                    "effect_size": round(effect_size, 3),
                    "rationale": f"Winners avg {w_mean:.4f} vs losers {l_mean:.4f} ({direction} is better)",
                })

    # Analyze categorical features (exclude exit_reason — it's a post-hoc
    # label determined after trade closes, not available at entry time)
    for col in ["rsi_zone", "session"]:
        if col not in feat_df.columns:
            continue

        cross = pd.crosstab(feat_df[col], feat_df["is_winner"], normalize="index")
        if 1 not in cross.columns:
            continue

        for val in cross.index:
            count = len(feat_df[feat_df[col] == val])
            if count < 3:
                continue
            win_rate = cross.loc[val, 1]
            edge = win_rate - overall_win_rate

            if abs(edge) > 0.10:  # 10%+ edge
                if edge < -0.10:
                    findings["suggested_filters"].append({
                        "feature": col,
                        "rule": f"Skip when {col} == '{val}'",
                        "effect_size": round(edge, 3),
                        "rationale": f"{val}: {win_rate:.1%} win rate vs {overall_win_rate:.1%} overall ({count} trades)",
                    })
                    findings["warnings"].append(
                        f"⚠️ {col}='{val}': {win_rate:.1%} win rate ({count} trades) — "
                        f"significantly below overall {overall_win_rate:.1%}"
                    )

    # Sort filters by absolute effect size
    findings["suggested_filters"].sort(key=lambda x: abs(x["effect_size"]), reverse=True)

    return findings


def analyze_time_patterns(run_id: str = None, strategy: str = None) -> dict:
    """Analyze win rates by day of week and hour."""
    init_db()
    conn = get_connection()
    trades_df = load_backtest_trades(conn, run_id=run_id, strategy=strategy)
    conn.close()

    if trades_df.empty:
        return {"error": "No trades found"}

    # Parse features to get time data
    time_data = []
    for _, row in trades_df.iterrows():
        try:
            features = json.loads(row["features_json"]) if row["features_json"] else {}
        except (json.JSONDecodeError, TypeError):
            features = {}
        if "day_of_week" in features:
            time_data.append({
                "day_of_week": features["day_of_week"],
                "hour": features.get("hour", -1),
                "session": features.get("session", "unknown"),
                "is_weekend": features.get("is_weekend", 0),
                "pnl_pct": row["pnl_pct"],
                "is_winner": 1 if row["pnl_pct"] > 0 else 0,
            })

    if not time_data:
        return {"error": "No time data in trade features"}

    time_df = pd.DataFrame(time_data)

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    results = {"by_day": {}, "by_session": {}, "by_hour": {}}

    # By day of week
    for day in range(7):
        day_trades = time_df[time_df["day_of_week"] == day]
        if len(day_trades) > 0:
            results["by_day"][day_names[day]] = {
                "trades": len(day_trades),
                "win_rate": day_trades["is_winner"].mean(),
                "avg_pnl": day_trades["pnl_pct"].mean(),
            }

    # By session
    for session in time_df["session"].unique():
        s_trades = time_df[time_df["session"] == session]
        if len(s_trades) > 0:
            results["by_session"][session] = {
                "trades": len(s_trades),
                "win_rate": s_trades["is_winner"].mean(),
                "avg_pnl": s_trades["pnl_pct"].mean(),
            }

    return results


def print_analysis(findings: dict):
    """Pretty-print the analysis findings."""
    if "error" in findings:
        print(f"Error: {findings['error']}")
        return

    print("=" * 60)
    print("WINNER vs LOSER ANALYSIS")
    print("=" * 60)
    print(f"Total trades: {findings['total_trades']}")
    print(f"Winners: {findings['winners']} | Losers: {findings['losers']}")
    print(f"Win rate: {findings['overall_win_rate']:.1%}")
    print(f"Avg win: {findings['avg_win_pct']:.2%} | Avg loss: {findings['avg_loss_pct']:.2%}")

    if findings["warnings"]:
        print(f"\n--- WARNINGS ---")
        for w in findings["warnings"]:
            print(f"  {w}")

    if findings["suggested_filters"]:
        print(f"\n--- SUGGESTED FILTERS (top 10) ---")
        for f in findings["suggested_filters"][:10]:
            print(f"  Rule: {f['rule']}")
            print(f"    {f['rationale']} (effect: {f['effect_size']})")
            print()

    # Top features by effect size
    if findings["feature_analysis"]:
        print(f"\n--- TOP FEATURES BY EFFECT SIZE ---")
        sorted_features = sorted(
            findings["feature_analysis"].items(),
            key=lambda x: abs(x[1]["effect_size"]),
            reverse=True
        )
        for name, analysis in sorted_features[:15]:
            direction = "↑" if analysis["effect_size"] > 0 else "↓"
            print(f"  {name}: effect={analysis['effect_size']:+.3f} {direction}"
                  f"  (winner={analysis['winner_mean']:.4f}, loser={analysis['loser_mean']:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze winner vs loser patterns")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--strategy", default=None)
    args = parser.parse_args()

    findings = analyze_winners_vs_losers(run_id=args.run_id, strategy=args.strategy)
    print_analysis(findings)

    print("\n")
    time_results = analyze_time_patterns(run_id=args.run_id, strategy=args.strategy)
    if "by_day" in time_results:
        print("--- WIN RATE BY DAY ---")
        for day, stats in time_results["by_day"].items():
            bar = "█" * int(stats["win_rate"] * 20)
            print(f"  {day}: {stats['win_rate']:.0%} ({stats['trades']} trades) {bar}")
