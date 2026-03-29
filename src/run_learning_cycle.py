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
import argparse
from datetime import datetime

from src.config import load_config
from src.data.database import init_db
from src.data.downloader import download_all
from src.backtest.runner import run_backtest
from src.simulation.monte_carlo import run_simulation
from src.learning.analyzer import analyze_winners_vs_losers, analyze_time_patterns, print_analysis
from src.learning.claude_review import run_claude_review
from src.learning.parameter_store import print_parameter_history, format_param_diff, print_performance_trend
from src.learning.filter_rules import init_filter_rules_table


def run_full_cycle(strategy: str = "darvas", skip_download: bool = False,
                   dry_run: bool = False):
    """Run the complete auto-learning cycle."""
    config = load_config()
    init_db()
    init_filter_rules_table()

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
            if results.get("filtered_trades", 0) > 0:
                print(f"    Filtered: {results['filtered_trades']} trades skipped by rules")

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

    print(f"\n  Performance trend:")
    print_performance_trend(strategy)

    # Send Telegram notification (if configured)
    _send_telegram_summary(config, strategy, all_results, sim_results, review)


def run_all_strategies(skip_download: bool = False, dry_run: bool = False):
    """Run learning cycle for all strategies and send a combined summary."""
    config = load_config()
    strategies = ["darvas", "sr_breakout", "ma_crossover", "dbw", "candlestick"]
    # Download once
    if not skip_download:
        init_db()
        print("[MULTI] Downloading data once for all strategies...")
        try:
            download_all(config)
        except Exception as e:
            print(f"  Warning: Download failed: {e}")

    for strategy in strategies:
        print(f"\n{'#' * 70}")
        print(f"# STRATEGY: {strategy}")
        print(f"{'#' * 70}")
        try:
            run_full_cycle(strategy=strategy, skip_download=True, dry_run=dry_run)
        except Exception as e:
            print(f"  Error running {strategy}: {e}")

    # Send combined summary
    _send_multi_strategy_telegram(config, strategies)


def _send_telegram_summary(config: dict, strategy: str, results: dict,
                           sim_results: dict, review: dict):
    """Send a summary notification via Telegram with strategy label and param diff."""
    tg_cfg = config.get("notifications", {}).get("telegram", {})
    if not tg_cfg.get("enabled", False):
        return

    bot_token = tg_cfg.get("bot_token", "")
    chat_id = tg_cfg.get("chat_id", "")
    if not bot_token or not chat_id or "YOUR" in bot_token:
        return

    try:
        import requests

        lines = [f"🤖 *Auto-Learning Cycle Complete*"]
        lines.append(f"📊 Strategy: *{strategy.upper()}*\n")

        for pair, r in results.items():
            if r.get("total_trades", 0) > 0:
                line = (f"*{pair}*: {r['total_trades']} trades | "
                        f"WR: {r['win_rate']:.0%} | "
                        f"Return: {r['total_return']:.1%}")
                if r.get("filtered_trades", 0) > 0:
                    line += f" | Filtered: {r['filtered_trades']}"
                lines.append(line)

        if sim_results and "verdict" in sim_results:
            lines.append(f"\nMonte Carlo: {sim_results.get('verdict_emoji', '')} {sim_results['verdict']}")

        if isinstance(review, dict) and "key_insights" in review:
            lines.append("\n*Top Insight:*")
            if review["key_insights"]:
                lines.append(f"_{review['key_insights'][0]}_")

        # Confidence indicator
        if isinstance(review, dict) and "confidence" in review:
            conf = review["confidence"]
            conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
            lines.append(f"\nConfidence: {conf_emoji} {conf}")
            if conf == "low":
                lines.append("⚠️ _Parameter update SKIPPED (low confidence)_")

        # Parameter changes diff
        param_diff = format_param_diff(strategy)
        if param_diff:
            lines.append(f"\n*Parameter Changes:*")
            lines.append(f"```\n{param_diff}\n```")

        # Filter rules count
        if isinstance(review, dict) and review.get("filter_rules"):
            lines.append(f"\n📋 {len(review['filter_rules'])} filter rules active")

        message = "\n".join(lines)

        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        print("  Telegram notification sent.")
    except Exception as e:
        print(f"  Telegram notification failed: {e}")


def _send_multi_strategy_telegram(config: dict, strategies: list):
    """Send a combined multi-strategy comparison via Telegram."""
    tg_cfg = config.get("notifications", {}).get("telegram", {})
    if not tg_cfg.get("enabled", False):
        return

    bot_token = tg_cfg.get("bot_token", "")
    chat_id = tg_cfg.get("chat_id", "")
    if not bot_token or not chat_id or "YOUR" in bot_token:
        return

    try:
        import requests
        from src.learning.parameter_store import get_performance_trend

        lines = ["📊 *Multi-Strategy Weekly Summary*\n"]

        for strategy in strategies:
            trend = get_performance_trend(strategy)
            if trend:
                latest = trend[-1]
                wr = latest.get("win_rate_display", "?")
                lines.append(f"*{strategy}*: WR {wr} (v{latest['version']})")

        lines.append("\n_Run parameter\\_store --trend for full history_")

        message = "\n".join(lines)
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        print("  Multi-strategy summary sent.")
    except Exception as e:
        print(f"  Multi-strategy summary failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full auto-learning cycle")
    parser.add_argument("--strategy", default="darvas",
                        choices=["darvas", "dbw", "candlestick", "sr_breakout", "ma_crossover"])
    parser.add_argument("--all-strategies", action="store_true",
                        help="Run cycle for all strategies with combined summary")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip OHLCV data download")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without calling Claude API")
    args = parser.parse_args()

    if args.all_strategies:
        run_all_strategies(
            skip_download=args.skip_download,
            dry_run=args.dry_run,
        )
    else:
        run_full_cycle(
            strategy=args.strategy,
            skip_download=args.skip_download,
            dry_run=args.dry_run,
        )
