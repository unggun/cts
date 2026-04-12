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
                        min_win_rate: float = 0.35,
                        min_profit_factor: float = 1.2,
                        max_drawdown: float = 0.25,
                        min_total_trades: int = 50,
                        min_expectancy: float = 0.0) -> tuple:
    """Check if backtest results meet minimum quality thresholds.

    Uses expectancy-based gating: a strategy with lower win rate but higher
    avg_win/avg_loss ratio (positive expectancy) is allowed through.

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

    # Aggregate win rate check
    total_winners = sum(r.get("winners", 0) for r in active.values())
    agg_win_rate = total_winners / total_trades if total_trades > 0 else 0
    if agg_win_rate < min_win_rate:
        failures.append(f"Win rate too low: {agg_win_rate:.1%} < {min_win_rate:.0%} minimum")

    # Aggregate expectancy check (WR * avg_win + (1-WR) * avg_loss)
    total_avg_win = sum(r.get("avg_win", 0) * r.get("winners", 0) for r in active.values())
    total_avg_loss = sum(r.get("avg_loss", 0) * r.get("losers", 0) for r in active.values())
    if total_winners > 0:
        total_avg_win /= total_winners
    total_losers = total_trades - total_winners
    if total_losers > 0:
        total_avg_loss /= total_losers
    expectancy = agg_win_rate * total_avg_win + (1 - agg_win_rate) * total_avg_loss
    if expectancy < min_expectancy:
        failures.append(f"Negative expectancy: {expectancy:.3%} (WR {agg_win_rate:.1%} × "
                        f"avg_win {total_avg_win:.2%} + loss component)")

    # Per-pair profit factor — only flag pairs below threshold, not blocking
    pf_warnings = []
    for pair, r in active.items():
        pf = r.get("profit_factor", 0)
        if pf < min_profit_factor:
            pf_warnings.append(f"{pair} profit factor low: {pf:.2f} < {min_profit_factor}")

    # Block only if majority of pairs have poor profit factor
    if len(pf_warnings) > len(active) * 0.7:
        failures.extend(pf_warnings)

    for pair, r in active.items():
        dd = r.get("max_drawdown", 0)
        if dd > max_drawdown:
            failures.append(f"{pair} max drawdown too high: {dd:.1%} > {max_drawdown:.0%}")

    return len(failures) == 0, failures


def reset_filter_rules(strategy: str):
    """Deactivate all filter rules for a strategy."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='filter_rules'")
    if cur.fetchone():
        cur.execute("UPDATE filter_rules SET active = 0 WHERE strategy = ?", (strategy,))
        count = cur.rowcount
        conn.commit()
        print(f"  Deactivated {count} filter rules for {strategy}")
    conn.close()


def reset_parameters(strategy: str):
    """Delete all saved parameter versions for a strategy so it gets first-run bypass."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM strategy_parameters WHERE strategy = ?", (strategy,))
    count = cur.rowcount
    conn.commit()
    conn.close()
    if count > 0:
        print(f"  Deleted {count} parameter version(s) for {strategy}")
    else:
        print(f"  No saved parameters for {strategy}")


def run_full_cycle(strategy: str = "darvas", skip_download: bool = False,
                   dry_run: bool = False, do_reset_filters: bool = False,
                   do_reset_params: bool = False):
    """Run the complete auto-learning cycle."""
    config = load_config()
    init_db()
    init_filter_rules_table()

    if do_reset_filters:
        print(f"\n[RESET] Clearing all filter rules for {strategy}...")
        reset_filter_rules(strategy)

    if do_reset_params:
        print(f"[RESET] Clearing saved parameters for {strategy}...")
        reset_parameters(strategy)

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

    # Collect this cycle's run_ids so downstream analysis only sees
    # trades produced in this cycle (not the full backtest_trades history).
    cycle_run_ids = [r["run_id"] for r in all_results.values() if r.get("run_id")]

    # Check if we have enough trades
    total_trades = sum(r.get("total_trades", 0) for r in all_results.values())
    min_trades = config.get("learning", {}).get("min_trades_for_analysis", 30)

    if total_trades < min_trades:
        print(f"\n  Only {total_trades} trades generated (need {min_trades}). "
              f"Consider adding more pairs or longer history.")
        if total_trades == 0:
            print("  Stopping cycle — no trades to analyze.")
            _send_zero_trade_telegram(config, strategy, all_results)
            return

    # ── Step 3: Analyze patterns ──
    print(f"\n[3/6] Analyzing winner/loser patterns ({total_trades} trades)...")
    findings = analyze_winners_vs_losers(strategy=strategy, run_ids=cycle_run_ids)
    print_analysis(findings)

    time_patterns = analyze_time_patterns(strategy=strategy, run_ids=cycle_run_ids)
    if "by_day" in time_patterns:
        print("\n  Time patterns found:")
        for day, stats in time_patterns["by_day"].items():
            print(f"    {day}: {stats['win_rate']:.0%} ({stats['trades']} trades)")

    # ── Step 4: Monte Carlo simulation ──
    print(f"\n[4/6] Running Monte Carlo simulation...")
    sim_results = run_simulation(strategy=strategy, run_ids=cycle_run_ids)

    # ── Quality gates ──
    # Skip gates on first run (no saved parameters yet) — allow initial tuning
    conn = get_connection()
    has_prior_params = load_latest_parameters(conn, strategy) is not None
    conn.close()

    gate_cfg = config.get("learning", {})
    passed, gate_failures = check_quality_gates(
        all_results,
        min_win_rate=gate_cfg.get("min_win_rate", 0.35),
        min_profit_factor=gate_cfg.get("min_profit_factor", 1.2),
        max_drawdown=gate_cfg.get("max_drawdown", 0.25),
        min_total_trades=gate_cfg.get("min_total_trades", 50),
        min_expectancy=gate_cfg.get("min_expectancy", 0.0),
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
        if not has_prior_params:
            print("\n  ⚠️ Monte Carlo verdict is FRAGILE (but first run — allowing initial tuning).")
            print("    Claude will save parameters so stops/filters can be improved.")
            # Keep passed = True for first run
        else:
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
        run_ids=cycle_run_ids,
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
        else:
            # Diagnostics for zero-trade pairs
            signals = results.get("signal_count", 0)
            rr_rej = results.get("rr_rejected", 0)
            filtered = results.get("filtered_trades", 0)
            if signals > 0:
                print(f"\n  {pair}: 0 trades "
                      f"(signals: {signals}, RR-rejected: {rr_rej}, filtered: {filtered})")

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


def run_all_strategies(skip_download: bool = False, dry_run: bool = False,
                       do_reset_filters: bool = False,
                       do_reset_params: bool = False):
    """Run learning cycle for all strategies and send a combined summary."""
    config = load_config()
    strategies = ["darvas", "sr_breakout", "ma_crossover", "dbw", "candlestick",
                  "rsi_vwap", "cvd", "macd", "mean_reversion", "bb_squeeze"]
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
            run_full_cycle(strategy=strategy, skip_download=True, dry_run=dry_run,
                           do_reset_filters=do_reset_filters,
                           do_reset_params=do_reset_params)
        except Exception as e:
            print(f"  Error running {strategy}: {e}")
        finally:
            # Force WAL checkpoint between strategies to release any lingering locks
            try:
                conn = get_connection()
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.close()
            except Exception:
                pass

    # Send combined summary
    _send_multi_strategy_telegram(config, strategies)


def _send_zero_trade_telegram(config: dict, strategy: str, results: dict):
    """Notify when a strategy's cycle produced zero tradeable signals.

    Without this, strategies that get fully gated by filter rules or
    risk-reward checks silently skip notification and the user can't
    tell them apart from strategies that didn't run at all.
    """
    tg_cfg = config.get("notifications", {}).get("telegram", {})
    if not tg_cfg.get("enabled", False):
        return

    bot_token = tg_cfg.get("bot_token", "")
    chat_id = tg_cfg.get("chat_id", "")
    if not bot_token or not chat_id or "YOUR" in bot_token:
        return

    try:
        import requests

        lines = [f"⚠️ *Auto-Learning Cycle — No Trades*"]
        lines.append(f"📊 Strategy: *{strategy.upper()}*\n")
        lines.append("_Cycle ran but generated 0 tradeable signals._\n")

        lines.append("*Per-pair diagnostics:*")
        any_signals = False
        for pair, r in results.items():
            signals = r.get("signal_count", 0)
            rr_rej = r.get("rr_rejected", 0)
            filtered = r.get("filtered_trades", 0)
            if signals > 0:
                any_signals = True
                lines.append(f"*{pair}*: {signals} signals, "
                             f"{rr_rej} RR-rejected, {filtered} filter-rejected")
            else:
                lines.append(f"*{pair}*: 0 signals")

        total_rr = sum(r.get("rr_rejected", 0) for r in results.values())
        total_filtered = sum(r.get("filtered_trades", 0) for r in results.values())

        lines.append("")
        if not any_signals:
            lines.append("_No signals fired at all — strategy pattern not "
                         "present in recent data._")
        elif total_filtered > total_rr:
            lines.append("_Signals fired but filter rules killed them all. "
                         "Consider `--reset-filters` if rules are stuck._")
        elif total_rr > 0 and total_filtered == 0:
            lines.append("_All signals rejected by risk-reward check. "
                         "The min\\_risk\\_reward or stop/target levels "
                         "need tuning — `--reset-filters` won't help._")
        else:
            lines.append(f"_Rejected by RR: {total_rr}, by filters: "
                         f"{total_filtered}. Check both thresholds._")

        message = "\n".join(lines)
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        print("  Zero-trade Telegram notification sent.")
    except Exception as e:
        print(f"  Zero-trade Telegram notification failed: {e}")


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

        # Monte Carlo FRAGILE gate — check if params were actually saved
        # by looking for a parameter diff (which only exists if save happened)
        param_diff_check = format_param_diff(strategy)
        if sim_results and sim_results.get("verdict") == "FRAGILE":
            if param_diff_check:
                lines.append("⚠️ _Monte Carlo: FRAGILE (first run — params saved for tuning)_")
            else:
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
                                 "ma_crossover", "rsi_vwap", "cvd", "macd",
                                 "mean_reversion", "bb_squeeze"])
    parser.add_argument("--all-strategies", action="store_true",
                        help="Run cycle for all strategies with combined summary")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip OHLCV data download")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without calling Claude API")
    parser.add_argument("--reset-filters", action="store_true",
                        help="Deactivate all filter rules before running (useful for stuck strategies)")
    parser.add_argument("--reset-params", action="store_true",
                        help="Delete saved parameters so strategy gets first-run bypass")
    args = parser.parse_args()

    if args.all_strategies:
        run_all_strategies(
            skip_download=args.skip_download,
            dry_run=args.dry_run,
            do_reset_filters=args.reset_filters,
            do_reset_params=args.reset_params,
        )
    else:
        run_full_cycle(
            strategy=args.strategy,
            skip_download=args.skip_download,
            dry_run=args.dry_run,
            do_reset_filters=args.reset_filters,
            do_reset_params=args.reset_params,
        )
