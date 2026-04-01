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
from src.data.database import init_db, get_connection, load_latest_parameters
from src.data.downloader import download_all
from src.backtest.runner import run_backtest
from src.simulation.monte_carlo import run_simulation
from src.learning.analyzer import analyze_winners_vs_losers, analyze_time_patterns, print_analysis
from src.learning.claude_review import run_claude_review
from src.learning.parameter_store import print_parameter_history, format_param_diff, print_performance_trend
from src.learning.filter_rules import init_filter_rules_table


def check_quality_gates(all_results: dict,
                        min_win_rate: float = 0.55,
                        min_profit_factor: float = 1.5,
                        max_drawdown: float = 0.20,
                        min_total_trades: int = 100) -> tuple:
    """Check if backtest results meet minimum quality thresholds.

    Returns:
        (passed: bool, failures: list[str])
    """
    failures = []

    active = {p: r for p, r in all_results.items() if r.get("total_trades", 0) > 0}
    if not active:
        return False, ["No trades generated across any pair"]

    total_trades = sum(r["total_trades"] for r in active.values())
    if total_trades < min_total_trades:
        failures.append(f"Insufficient trades: {total_trades} < {min_total_trades} minimum")

    total_winners = sum(r.get("winners", 0) for r in active.values())
    agg_win_rate = total_winners / total_trades if total_trades > 0 else 0
    if agg_win_rate < min_win_rate:
        failures.append(f"Win rate too low: {agg_win_rate:.1%} < {min_win_rate:.0%} minimum")

    for pair, r in active.items():
        pf = r.get("profit_factor", 0)
        if pf < min_profit_factor:
            failures.append(f"{pair} profit factor too low: {pf:.2f} < {min_profit_factor}")

    for pair, r in active.items():
        dd = r.get("max_drawdown", 0)
        if dd > max_drawdown:
            failures.append(f"{pair} max drawdown too high: {dd:.1%} > {max_drawdown:.0%}")

    return len(failures) == 0, failures


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

    # ── Quality gates ──
    # Skip gates on first run (no saved parameters yet) — allow initial tuning
    conn = get_connection()
    has_prior_params = load_latest_parameters(conn, strategy) is not None
    conn.close()

    gate_cfg = config.get("learning", {})
    passed, gate_failures = check_quality_gates(
        all_results,
        min_win_rate=gate_cfg.get("min_win_rate", 0.55),
        min_profit_factor=gate_cfg.get("min_profit_factor", 1.5),
        max_drawdown=gate_cfg.get("max_drawdown", 0.20),
        min_total_trades=gate_cfg.get("min_total_trades", 100),
    )
    if not passed:
        if not has_prior_params:
            print("\n  ⚠️ Quality gates FAILED (but first run — allowing initial tuning):")
            for f in gate_failures:
                print(f"    - {f}")
            passed = True  # Allow first-run tuning
        else:
            print("\n  ⚠️ Quality gates FAILED:")
            for f in gate_failures:
                print(f"    - {f}")
            print("  Parameter updates will be blocked this cycle.")

    # ── Monte Carlo gate ──
    mc_fragile = (sim_results and sim_results.get("verdict") == "FRAGILE")
    if mc_fragile and passed:
        print("\n  ⚠️ Monte Carlo verdict is FRAGILE — parameter updates will be blocked.")
        print("    (Filter rules will still be saved.)")
        passed = False

    # ── Step 5: Claude review ──
    print(f"\n[5/6] Running Claude AI review...")
    if dry_run:
        print("  (Dry run — showing prompt only)")
    review = run_claude_review(
        strategy=strategy,
        sim_results=sim_results,
        dry_run=dry_run,
        block_param_updates=not passed,
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
    strategies = ["darvas", "sr_breakout", "ma_crossover", "dbw", "candlestick",
                  "rsi_vwap", "cvd", "macd"]
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

        # Monte Carlo FRAGILE gate
        if sim_results and sim_results.get("verdict") == "FRAGILE":
            lines.append("🛑 _Parameter update BLOCKED (Monte Carlo: FRAGILE)_")

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
                        choices=["darvas", "dbw", "candlestick", "sr_breakout",
                                 "ma_crossover", "rsi_vwap", "cvd", "macd"])
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
