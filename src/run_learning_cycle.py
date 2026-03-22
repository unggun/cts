#!/usr/bin/env python3
"""Master auto-learning loop orchestrator.

This script runs the complete learning cycle:
1. Download latest OHLCV data
2. Run backtest with current parameters
3. Analyze winner/loser patterns
4. Run Monte Carlo simulation
5. Send findings to Claude for review
6. Apply updated parameters
7. Send summary via Telegram

Schedule this via cron or n8n on your VPS:
  # Weekly on Sunday at 00:00
  0 0 * * 0 cd /opt/crypto-trading-system && python -m src.run_learning_cycle
"""
import json
import argparse
from datetime import datetime

from src.config import load_config
from src.data.database import init_db, get_connection
from src.data.downloader import download_all
from src.backtest.runner import run_backtest
from src.simulation.monte_carlo import run_simulation
from src.learning.analyzer import analyze_winners_vs_losers, analyze_time_patterns, print_analysis
from src.learning.claude_review import run_claude_review
from src.learning.parameter_store import print_parameter_history


def run_full_cycle(strategy: str = "darvas", skip_download: bool = False,
                   dry_run: bool = False):
    """Run the complete auto-learning cycle."""
    config = load_config()
    init_db()

    print("=" * 70)
    print(f"AUTO-LEARNING CYCLE — {datetime.now().isoformat()}")
    print(f"Strategy: {strategy}")
    print("=" * 70)

    # ── Step 1: Download latest data ──
    if not skip_download:
        print("\n[1/6] Downloading latest OHLCV data...")
        try:
            download_all(config)
        except Exception as e:
            print(f"  Warning: Download failed: {e}")
            print("  Continuing with existing data...")
    else:
        print("\n[1/6] Skipping download (--skip-download)")

    # ── Step 2: Run backtest ──
    print("\n[2/6] Running backtest...")
    pairs = config["trading"]["pairs"]
    timeframe = config["trading"]["timeframes"][0]

    all_results = {}
    for pair in pairs:
        print(f"\n  Backtesting {pair}...")
        results = run_backtest(strategy, pair, timeframe, config)
        all_results[pair] = results

    # Check if we have enough trades
    total_trades = sum(r.get("total_trades", 0) for r in all_results.values())
    min_trades = config.get("learning", {}).get("min_trades_for_analysis", 30)

    if total_trades < min_trades:
        print(f"\n  Only {total_trades} trades generated (need {min_trades}). "
              f"Consider adding more pairs or longer history.")
        if total_trades == 0:
            print("  Stopping cycle — no trades to analyze.")
            return

    # ── Step 3: Analyze patterns ──
    print(f"\n[3/6] Analyzing winner/loser patterns ({total_trades} trades)...")
    findings = analyze_winners_vs_losers(strategy=strategy)
    print_analysis(findings)

    time_patterns = analyze_time_patterns(strategy=strategy)
    if "by_day" in time_patterns:
        print("\n  Time patterns found:")
        for day, stats in time_patterns["by_day"].items():
            print(f"    {day}: {stats['win_rate']:.0%} ({stats['trades']} trades)")

    # ── Step 4: Monte Carlo simulation ──
    print(f"\n[4/6] Running Monte Carlo simulation...")
    sim_results = run_simulation(strategy=strategy)

    # ── Step 5: Claude review ──
    print(f"\n[5/6] Running Claude AI review...")
    if dry_run:
        print("  (Dry run — showing prompt only)")
    review = run_claude_review(
        strategy=strategy,
        sim_results=sim_results,
        dry_run=dry_run,
    )

    # ── Step 6: Summary ──
    print(f"\n[6/6] Cycle complete.")
    print("\n" + "=" * 70)
    print("CYCLE SUMMARY")
    print("=" * 70)

    for pair, results in all_results.items():
        if results.get("total_trades", 0) > 0:
            print(f"\n  {pair}:")
            print(f"    Trades: {results['total_trades']} | "
                  f"Win rate: {results['win_rate']:.1%} | "
                  f"Return: {results['total_return']:.2%} | "
                  f"Max DD: {results['max_drawdown']:.2%}")

    if sim_results and "verdict" in sim_results:
        print(f"\n  Monte Carlo: {sim_results.get('verdict_emoji', '')} "
              f"{sim_results['verdict']}")
        print(f"    Prob. of ruin: {sim_results.get('probability_of_ruin', 0):.2%}")

    if not dry_run and isinstance(review, dict) and "key_insights" in review:
        print(f"\n  Claude Insights:")
        for insight in review.get("key_insights", [])[:3]:
            print(f"    • {insight}")
        print(f"  Confidence: {review.get('confidence', 'unknown')}")

    print(f"\n  Parameter history:")
    print_parameter_history(strategy)

    # Send Telegram notification (if configured)
    _send_telegram_summary(config, all_results, sim_results, review)


def _send_telegram_summary(config: dict, results: dict,
                           sim_results: dict, review: dict):
    """Send a summary notification via Telegram."""
    tg_cfg = config.get("notifications", {}).get("telegram", {})
    if not tg_cfg.get("enabled", False):
        return

    bot_token = tg_cfg.get("bot_token", "")
    chat_id = tg_cfg.get("chat_id", "")
    if not bot_token or not chat_id or "YOUR" in bot_token:
        return

    try:
        import requests

        lines = ["🤖 *Auto-Learning Cycle Complete*\n"]

        for pair, r in results.items():
            if r.get("total_trades", 0) > 0:
                lines.append(
                    f"*{pair}*: {r['total_trades']} trades | "
                    f"WR: {r['win_rate']:.0%} | "
                    f"Return: {r['total_return']:.1%}"
                )

        if sim_results and "verdict" in sim_results:
            lines.append(f"\nMonte Carlo: {sim_results.get('verdict_emoji', '')} {sim_results['verdict']}")

        if isinstance(review, dict) and "key_insights" in review:
            lines.append("\n*Top Insight:*")
            if review["key_insights"]:
                lines.append(f"_{review['key_insights'][0]}_")

        message = "\n".join(lines)

        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        print("  Telegram notification sent.")
    except Exception as e:
        print(f"  Telegram notification failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full auto-learning cycle")
    parser.add_argument("--strategy", default="darvas",
                        choices=["darvas", "dbw", "candlestick", "sr_breakout", "ma_crossover"])
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip OHLCV data download")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without calling Claude API")
    args = parser.parse_args()

    run_full_cycle(
        strategy=args.strategy,
        skip_download=args.skip_download,
        dry_run=args.dry_run,
    )
