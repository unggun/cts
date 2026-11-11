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
import re
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
from src.learning.promotion import (
    holdout_start, load_recent_evaluations, promotion_config, record_evaluation,
    run_promotion,
)


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


def _verify_and_promote(strategy: str, review: dict, findings: dict, config: dict,
                        trade_start, dry_run: bool) -> dict:
    """Backtest Claude's proposal against the current version on the holdout.

    Every usable proposal is tested, whatever confidence Claude reports: the
    holdout decides. Confidence is still recorded and gates live readiness.
    In a dry run there is no proposal; the analyzer's filters alone are
    evaluated so the holdout comparison can still be inspected.
    """
    if not dry_run:
        if "error" in review or "raw_response" in review:
            return {"promoted": False, "reason": "no usable proposal from Claude"}
        if not any(isinstance(review.get(k), want) for k, want in
                   (("updated_parameters", dict), ("filter_rules", list))):
            reason = "Claude returned no parameters or filter rules"
            record_evaluation(review.get("_session_id"), {"promoted": False, "reason": reason})
            return {"promoted": False, "reason": reason}

    print(f"\n  Validating proposal on holdout (from {trade_start:%Y-%m-%d})...")
    evaluation = run_promotion(strategy, {} if dry_run else review, findings,
                               config, trade_start, dry_run=dry_run)
    inc, cand = evaluation["incumbent"], evaluation["candidate"]
    print(f"    Current:   {inc['trades']:4d} trades | expectancy {inc['expectancy']:+.3%} "
          f"| max DD {inc['max_drawdown']:.1%}")
    print(f"    Candidate: {cand['trades']:4d} trades | expectancy {cand['expectancy']:+.3%} "
          f"| max DD {cand['max_drawdown']:.1%}")
    if "no_filters" in evaluation:
        nf = evaluation["no_filters"]
        print(f"    No filters:{nf['trades']:4d} trades | expectancy {nf['expectancy']:+.3%} "
              f"| max DD {nf['max_drawdown']:.1%}")
    if evaluation.get("promoted"):
        print(f"  ✅ Promoted as version {evaluation['version']}: {evaluation['reason']}")
    elif dry_run:
        verdict = "would promote" if evaluation["would_promote"] else "would not promote"
        what = "removing all filters" if evaluation.get("filters_removed") \
            else "analyzer-only filters"
        print(f"  (Dry run) {verdict} {what}: {evaluation['reason']}")
    else:
        print(f"  🛑 Not promoted: {evaluation['reason']}")
    if not dry_run:
        record_evaluation(review.get("_session_id"), evaluation)
    return evaluation


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
    # The newest data is held out: analysis and Claude only see the train
    # window, and proposals are judged on the holdout (promotion.py).
    trade_start = holdout_start(config)
    holdout_days = promotion_config(config)["holdout_days"]
    print(f"  Train window ends {trade_start:%Y-%m-%d %H:%M} UTC "
          f"(last {holdout_days} days held out for validation)")

    # Two train backtests: with the active filters (the deployed version, for
    # reports, quality gates and Monte Carlo) and without them (what the
    # analyzer and Claude learn from, so a filter's blind spots stay visible).
    all_results = {}
    for pair in pairs:
        print(f"\n  Backtesting {pair}...")
        results = run_backtest(strategy, pair, timeframe, config, end=trade_start)
        all_results[pair] = results

    if any(r.get("filtered_trades", 0) > 0 for r in all_results.values()):
        unfiltered = {}
        for pair in pairs:
            print(f"\n  Backtesting {pair} without filters (for analysis)...")
            unfiltered[pair] = run_backtest(strategy, pair, timeframe, config,
                                            use_filters=False, end=trade_start)
    else:
        unfiltered = all_results     # no signal was filtered: same trades

    # Collect this cycle's run_ids so downstream analysis only sees
    # trades produced in this cycle (not the full backtest_trades history).
    cycle_run_ids = [r["run_id"] for r in all_results.values() if r.get("run_id")]
    analysis_run_ids = [r["run_id"] for r in unfiltered.values() if r.get("run_id")]

    # Check if we have enough trades
    min_trades = config.get("learning", {}).get("min_trades_for_analysis", 30)
    total_trades = sum(r.get("total_trades", 0) for r in unfiltered.values())

    if total_trades < min_trades:
        print(f"\n  Only {total_trades} trades generated (need {min_trades}). "
              f"Consider adding more pairs or longer history.")
        if total_trades == 0:
            print("  Stopping cycle — no trades to analyze.")
            _send_zero_trade_telegram(config, strategy, unfiltered)
            return

    # ── Step 3: Analyze patterns ──
    print(f"\n[3/6] Analyzing winner/loser patterns ({total_trades} unfiltered trades)...")
    findings = analyze_winners_vs_losers(strategy=strategy, run_ids=analysis_run_ids)
    print_analysis(findings)

    time_patterns = analyze_time_patterns(strategy=strategy, run_ids=analysis_run_ids)
    if "by_day" in time_patterns:
        print("\n  Time patterns found:")
        for day, stats in time_patterns["by_day"].items():
            print(f"    {day}: {stats['win_rate']:.0%} ({stats['trades']} trades)")

    # ── Step 4: Monte Carlo simulation (current version, filters on) ──
    print(f"\n[4/6] Running Monte Carlo simulation...")
    sim_results = run_simulation(strategy=strategy, run_ids=cycle_run_ids)

    # ── Quality gates (reported; promotion below decides what changes) ──
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
        print("\n  ⚠️ Quality gates FAILED (current version is not live-ready):")
        for f in gate_failures:
            print(f"    - {f}")

    # ── Step 5: Claude review ──
    print(f"\n[5/6] Running Claude AI review...")
    if dry_run:
        print("  (Dry run — showing prompt only)")
    review = run_claude_review(
        strategy=strategy,
        sim_results=sim_results,
        dry_run=dry_run,
        run_ids=analysis_run_ids,
        history=load_recent_evaluations(strategy, n=3),
        holdout_days=holdout_days,
        train_end=trade_start,
    )

    # ── Step 5b: Verify the proposal on held-out data, promote if better ──
    review["promotion"] = _verify_and_promote(strategy, review, findings, config,
                                              trade_start, dry_run)

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


def _md(text) -> str:
    """Escape Telegram legacy-Markdown characters.

    Escapes only work outside entities, so escaped text must not be wrapped
    in *bold* or _italic_.
    """
    return re.sub(r"([_*`\[])", r"\\\1", str(text))


def _unmd(text: str) -> str:
    """Undo ``_md`` for the plain-text fallback, so no backslashes show."""
    return re.sub(r"\\([_*`\[])", r"\1", text)


def _promotion_lines(strategy: str, review: dict) -> list:
    """Telegram lines describing the holdout verdict on Claude's proposal."""
    ev = review.get("promotion") if isinstance(review, dict) else None
    if not ev:
        return []
    reason = _md(ev.get("reason", ""))
    if ev.get("promoted"):
        lines = [f"\n✅ *Promoted v{ev.get('version')}* (holdout): {reason}"]
        param_diff = format_param_diff(strategy)
        if param_diff:
            lines.append(f"```\n{param_diff}\n```")
        if ev.get("filters_removed"):
            lines.append("📋 Filters now: none (all removed)")
        elif ev.get("candidate_filters"):
            lines.append(f"📋 Filters now: {_md('; '.join(ev['candidate_filters']))}")
        return lines
    return [f"\n🛑 Not promoted: {reason}"]


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
                lines.append(_md(review["key_insights"][0]))

        # Confidence indicator
        if isinstance(review, dict) and "confidence" in review:
            conf = review["confidence"]
            conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
            lines.append(f"\nConfidence: {conf_emoji} {conf}")
        runs = ((review.get("backtests") or {}).get("calls") or []) \
            if isinstance(review, dict) and isinstance(review.get("backtests"), dict) else []
        if runs:
            lines.append(f"🔬 {len(runs)} train backtest{'s' if len(runs) != 1 else ''} "
                         f"before proposing")

        lines.extend(_promotion_lines(strategy, review))

        message = "\n".join(lines)

        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        if resp.ok:
            print("  Telegram notification sent.")
        else:
            print(f"  Telegram API rejected message: {resp.status_code} {resp.text[:200]}")
            # Retry without Markdown if parsing failed
            resp2 = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": _unmd(message)},
                timeout=10,
            )
            if resp2.ok:
                print("  Sent plain-text fallback.")
            else:
                print(f"  Plain-text fallback also failed: {resp2.status_code}")
    except Exception as e:
        print(f"  Telegram notification failed: {e}")


def _journal_lines() -> list:
    """Versions whose paper results are clearly behind their holdout result."""
    try:
        from src.learning.journal import behind_versions
        behind = behind_versions()
    except Exception as e:
        print(f"  Journal check failed: {e}")
        return []
    if not behind:
        return []
    lines = ["\n⚠️ *Behind holdout on paper* (no automatic rollback):"]
    for r in behind:
        v = r["current"]
        lines.append(_md(f"{r['strategy']} v{v['version']}: paper "
                         f"{v['paper']['expectancy'] * 100:+.2f}% on {v['paper']['closed']} trades "
                         f"vs holdout {v['holdout']['expectancy'] * 100:+.2f}%"))
    lines.append(_md("Use /journal <strategy> for detail."))
    return lines


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
        from src.execution.live_readiness import build_report, short_summary

        lines = ["📊 *Multi-Strategy Weekly Summary*\n"]

        status_counts = {"READY": 0, "WATCH": 0, "BLOCKED": 0, "INSUFFICIENT_DATA": 0}

        for strategy in strategies:
            trend = get_performance_trend(strategy)

            try:
                report = build_report(strategy)
                icon_label = short_summary(report)
                status_counts[report["status"]] += 1
            except Exception:
                report = None
                icon_label = ""

            if trend:
                latest = trend[-1]
                wr = latest.get("win_rate_display", "?")
                suffix = f" {icon_label.split()[0]}" if icon_label else ""
                lines.append(f"*{strategy}*: WR {wr} (v{latest['version']}){suffix}")
            elif report is not None:
                # No trend yet but readiness was computed — still show the row
                lines.append(f"*{strategy}*: {icon_label}")

        lines.append(
            f"\nReadiness: {status_counts['READY']} READY | "
            f"{status_counts['WATCH']} WATCH | "
            f"{status_counts['BLOCKED']} BLOCKED | "
            f"{status_counts['INSUFFICIENT_DATA']} INSUFFICIENT_DATA"
        )
        lines.extend(_journal_lines())
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
